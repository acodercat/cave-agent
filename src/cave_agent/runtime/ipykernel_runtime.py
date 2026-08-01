"""IPyKernelRuntime — process-isolated execution via a Jupyter kernel."""

from __future__ import annotations

import logging

from ..security import SecurityChecker
from .base import BaseRuntime
from .executor import ErrorFeedbackMode, RuntimeExecutionError
from .ipykernel_executor import IPyKernelExecutor
from .primitives import Function, Type, Variable

logger = logging.getLogger(__name__)


class IPyKernelRuntime(BaseRuntime):
    """A Python runtime backed by a separate IPython kernel process.

    Drop-in replacement for :class:`~cave_agent.runtime.IPythonRuntime` with
    process isolation — a crash (segfault, OOM) does not bring down the host,
    and :meth:`interrupt` really does stop a runaway cell (SIGINT to the
    kernel), which is what makes ``max_exec_timeout`` enforceable.

    Injected objects are serialized via dill, which handles local functions,
    closures, lambdas and most Python objects. That serialization is the cost:
    objects are copies, not the caller's originals.

    **The subprocess starts on demand.** Constructing the runtime, registering
    functions and variables, and rendering the system prompt all stay free; the
    kernel appears the first time code actually runs, so a conversation that
    never produces a code block never spawns one. Teardown remains explicit —
    use ``async with`` (or call :meth:`stop`) so the subprocess is reclaimed::

        async with IPyKernelRuntime(functions=[...]) as rt:
            await rt.execute("print('hello')")

    Calling :meth:`start` up front is still supported and idempotent, for
    callers who would rather pay the ~1s startup before the first request than
    during it.
    """

    def __init__(
        self,
        functions: list[Function] | None = None,
        variables: list[Variable] | None = None,
        types: list[Type] | None = None,
        security_checker: SecurityChecker | None = None,
        error_feedback_mode: ErrorFeedbackMode = ErrorFeedbackMode.PLAIN,
        iopub_timeout: float | None = None,
    ):
        """
        Args:
            iopub_timeout: Seconds to wait for the next output message from the
                kernel before giving up on a cell. This bounds silence, not
                total runtime — a cell that prints periodically can run
                indefinitely. ``None`` uses the executor default; raise it for
                long silent computations.
        """
        super().__init__(
            IPyKernelExecutor(
                security_checker=security_checker,
                error_feedback_mode=error_feedback_mode,
                iopub_timeout=iopub_timeout,
            ),
            functions=functions,
            variables=variables,
            types=types,
        )

    async def start(self) -> IPyKernelRuntime:
        """Start the kernel subprocess."""
        await self._executor.start()
        return self

    async def stop(self) -> None:
        """Shut down the kernel subprocess, keeping what it was told to hold.

        The kernel takes the namespace with it, but the registry outlives it.
        The executor retains the last successfully staged form of every
        binding and queues it for the next kernel. Shutdown therefore never
        depends on serializing a mutable host object while releasing a process.
        """
        await self._executor.stop()

    async def __aenter__(self) -> IPyKernelRuntime:
        return await self.start()

    async def __aexit__(self, exc_type: type[BaseException] | None, *_: object) -> None:
        """Shut down, without outranking whatever the block was already raising.

        ``stop()`` raises when the kernel outlives its shutdown, which is worth
        knowing — but not at the price of the exception that brought us here.
        Raised from ``__aexit__``, it replaced the caller's ``ValueError`` and
        even a ``CancelledError``, leaving the original reachable only through
        ``__context__``. On a clean exit it still propagates: nothing is being
        hidden, and a leaked kernel is the only news there is.
        """
        if exc_type is None:
            await self.stop()
            return
        try:
            await self.stop()
        except RuntimeExecutionError:
            logger.warning(
                "Kernel shutdown failed while another error was propagating; "
                "the runtime is left marked so a later stop() retries",
                exc_info=True,
            )

    async def interrupt(self) -> None:
        """Send SIGINT to the kernel to interrupt running execution."""
        await self._executor.interrupt()
