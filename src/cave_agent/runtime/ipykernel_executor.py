"""Code executor using a separate IPython kernel process for isolation."""

from __future__ import annotations

import asyncio
import base64
import logging
from typing import Any

logger = logging.getLogger(__name__)

import dill
from jupyter_client import AsyncKernelManager

from ..security import SecurityChecker
from .executor import ExecutionResult, ErrorFeedbackMode, check_security

# Timeout constants (seconds)
_KERNEL_READY_MAX_RETRIES = 60
_KERNEL_INFO_TIMEOUT = 1.0
_IOPUB_TIMEOUT = 30.0
_SHELL_REPLY_TIMEOUT = 10.0
_DRAIN_TIMEOUT = 0.1


# Executed once after kernel starts to configure IPython behavior
_KERNEL_SETUP = """\
ip = get_ipython()
ip.colors = "nocolor"
ip.InteractiveTB.set_mode("Plain")
"""


def _make_injection_code(name: str, value: Any) -> str:
    """Generate code that recreates *value* as *name* inside the kernel.

    Uses dill for serialization, which handles local functions, closures,
    lambdas, and most Python objects that standard pickle cannot.
    """
    data = base64.b64encode(dill.dumps(value, recurse=True)).decode()
    return (
        "import dill as _dill, base64 as _b64\n"
        f"{name} = _dill.loads(_b64.b64decode({data!r}))"
    )


class IPyKernelExecutor:
    """Executes Python code in an isolated IPython kernel process.

    Same interface as ``IPythonExecutor`` so it can be used as a drop-in
    replacement inside ``Runtime`` / ``IPyKernelRuntime``.
    """

    def __init__(
        self,
        security_checker: SecurityChecker | None = None,
        error_feedback_mode: ErrorFeedbackMode = ErrorFeedbackMode.PLAIN,
    ):
        self._km = AsyncKernelManager(kernel_name="python3")
        self._kernel_client: Any = None
        self._security_checker = security_checker
        self._error_feedback_mode = error_feedback_mode
        self._started = False
        self._pending_injections: dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Start the kernel subprocess and wait until it is ready."""
        await self._km.start_kernel()
        self._kernel_client = self._km.client()
        self._kernel_client.start_channels()

        # Wait for kernel_info_reply to confirm readiness
        for _ in range(_KERNEL_READY_MAX_RETRIES):
            try:
                self._kernel_client.kernel_info()
                msg = await asyncio.wait_for(self._kernel_client.get_shell_msg(), timeout=_KERNEL_INFO_TIMEOUT)
                if msg["header"]["msg_type"] == "kernel_info_reply":
                    break
            except asyncio.TimeoutError:
                continue
        else:
            raise RuntimeError("Kernel did not become ready in time")

        self._started = True

        # Drain any stale IOPub messages from the readiness handshake
        await self._drain_iopub()

        # Configure IPython inside the kernel
        await self._execute_silent(_KERNEL_SETUP)

    async def interrupt(self) -> None:
        """Send SIGINT to the kernel to interrupt running execution."""
        if self._km and await self._km.is_alive():
            await self._km.interrupt_kernel()

    async def stop(self) -> None:
        """Shut down the kernel."""
        try:
            if self._kernel_client is not None:
                self._kernel_client.stop_channels()
                self._kernel_client = None
            if await self._km.is_alive():
                await self._km.shutdown_kernel(now=True)
        except Exception:
            logger.debug("Error during kernel shutdown", exc_info=True)
        self._started = False

    # ------------------------------------------------------------------
    # Namespace (mirrors IPythonExecutor interface)
    # ------------------------------------------------------------------

    def inject_into_namespace(self, name: str, value: Any) -> None:
        """Queue *value* for injection.  Actually sent on the next ``execute()``."""
        self._pending_injections[name] = value

    async def get_from_namespace(self, name: str) -> Any:
        """Retrieve a variable from the kernel namespace via dill.

        Uses IPython's ``display()`` with a custom mime type to transport
        serialized data through the Jupyter display_data channel, rather
        than mixing serialized output into stdout.
        """
        await self._flush_injections()
        code = (
            "import dill as _d, base64 as _b\n"
            f"display({{'application/x-dill': _b.b64encode(_d.dumps({name})).decode()}}, raw=True)"
        )
        msg_id = self._kernel_client.execute(code)
        async for msg in self._iopub_for(msg_id):
            if msg["header"]["msg_type"] == "display_data":
                encoded = msg["content"]["data"].get("application/x-dill")
                if encoded:
                    return dill.loads(base64.b64decode(encoded))
            elif msg["header"]["msg_type"] == "error":
                raise KeyError(
                    f"Variable '{name}': {msg['content']['ename']}: {msg['content']['evalue']}"
                )
        raise KeyError(f"Variable '{name}' not found")

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    async def execute(self, code: str) -> ExecutionResult:
        """Execute *code* on the kernel and return an ``ExecutionResult``."""
        if not self._started:
            raise RuntimeError("Kernel not started — call start() first")

        violation = check_security(self._security_checker, code)
        if violation:
            return violation

        # Flush pending injections
        await self._flush_injections()

        # Send code to kernel
        msg_id = self._kernel_client.execute(code)
        return await self._collect_result(msg_id)

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    async def reset(self) -> None:
        """Restart the kernel (clears all state)."""
        await self.stop()
        # Create a fresh KernelManager (old ZMQ context is closed)
        self._km = AsyncKernelManager(kernel_name="python3")
        await self.start()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    async def _iopub_for(self, msg_id: str, timeout: float = _IOPUB_TIMEOUT):
        """Yield IOPub messages belonging to *msg_id* until kernel is idle."""
        while True:
            msg = await asyncio.wait_for(self._kernel_client.get_iopub_msg(), timeout=timeout)
            if msg["parent_header"].get("msg_id") != msg_id:
                continue
            yield msg
            if msg["header"]["msg_type"] == "status" and msg["content"]["execution_state"] == "idle":
                return

    async def _shell_reply_for(self, msg_id: str, timeout: float = _SHELL_REPLY_TIMEOUT) -> dict:
        """Wait for the shell reply matching *msg_id*."""
        while True:
            msg = await asyncio.wait_for(self._kernel_client.get_shell_msg(), timeout=timeout)
            if msg["parent_header"].get("msg_id") == msg_id:
                return msg

    async def _drain_iopub(self) -> None:
        """Drain any stale messages from the IOPub channel."""
        while True:
            try:
                await asyncio.wait_for(self._kernel_client.get_iopub_msg(), timeout=_DRAIN_TIMEOUT)
            except asyncio.TimeoutError:
                break

    async def _flush_injections(self) -> None:
        if not self._pending_injections:
            return
        for name, value in self._pending_injections.items():
            code = _make_injection_code(name, value)
            await self._execute_silent(code)
        self._pending_injections.clear()

    async def _execute_silent(self, code: str) -> None:
        """Run *code* on the kernel, discard output, raise on error."""
        msg_id = self._kernel_client.execute(code, silent=True)
        reply = await self._shell_reply_for(msg_id)
        if reply["content"].get("status") == "error":
            raise RuntimeError(
                f"Kernel setup error: {reply['content'].get('ename', 'Unknown')}: "
                f"{reply['content'].get('evalue', '')}"
            )

    async def _collect_result(self, msg_id: str) -> ExecutionResult:
        """Read IOPub messages for *msg_id* and build an ``ExecutionResult``."""
        stdout_parts: list[str] = []
        error: BaseException | None = None

        try:
            async for msg in self._iopub_for(msg_id):
                match msg["header"]["msg_type"]:
                    case "stream":
                        stdout_parts.append(msg["content"]["text"])

                    case "execute_result":
                        text = msg["content"].get("data", {}).get("text/plain")
                        if text:
                            stdout_parts.append(text + "\n")

                    case "error":
                        content = msg["content"]
                        error = Exception(f"{content['ename']}: {content['evalue']}")
                        if self._error_feedback_mode == ErrorFeedbackMode.PLAIN:
                            stdout_parts.append("\n".join(content.get("traceback", [])))
                        else:
                            stdout_parts.append(f"{content['ename']}: {content['evalue']}")

        except asyncio.TimeoutError:
            error = TimeoutError(f"Execution timed out after {_IOPUB_TIMEOUT}s")
            stdout_parts.append(f"TimeoutError: Execution timed out after {_IOPUB_TIMEOUT}s")

        stdout = "".join(stdout_parts) or None
        return ExecutionResult(error=error, stdout=stdout)
