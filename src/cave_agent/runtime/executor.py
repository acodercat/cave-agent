import asyncio
import gc
import io
import sys
import threading
from collections import deque
from collections.abc import Callable
from contextlib import asynccontextmanager, contextmanager
from contextvars import ContextVar
from enum import Enum
from typing import Any

from IPython.core.interactiveshell import InteractiveShell
from traitlets.config import Config

from .._placeholders import normalize_identifier
from ..security import SecurityChecker, SecurityError


# Where a cell's output goes while it runs. ``None`` — the default for any
# context that is not executing generated code — means "leave it alone".
#
# Asyncio tasks inherit context variables. Keeping an active flag alongside
# the buffer lets tasks which outlive their cell fall back to the host stream
# instead of writing into a buffer that will never be observed again.
class _CaptureRoute:
    def __init__(self) -> None:
        self.buffer = io.StringIO()
        self.active = True


_EXECUTION_OUTPUT: ContextVar["_CaptureRoute | None"] = ContextVar(
    "cave_execution_output",
    default=None,
)

_ROUTING_INSTALLED = threading.Lock()

# How often a waiter re-checks its place in the queue. Polling rather than
# handing a blocking acquire to a worker thread: an acquire already dispatched
# cannot be cancelled, so a cancelled caller could take the gate and never
# release it. Executions run orders of magnitude longer than this.
_GATE_POLL_SECONDS = 0.005


class _Gate:
    """A cross-thread mutex that serves waiters in arrival order.

    Cross-thread because a runtime may be driven from more than one event loop,
    and an ``asyncio.Lock`` only excludes coroutines sharing its loop. In
    arrival order because an unfair gate lets a tight execution loop starve
    teardown indefinitely — measured, a waiter never got in at all — and
    ``stop()`` has to be able to make progress.
    """

    def __init__(self) -> None:
        self._mutex = threading.Lock()
        self._waiting: deque[object] = deque()

    @asynccontextmanager
    async def held(self):
        ticket = object()
        with self._mutex:
            self._waiting.append(ticket)
        try:
            while True:
                with self._mutex:
                    if self._waiting[0] is ticket:
                        break
                await asyncio.sleep(_GATE_POLL_SECONDS)
        except BaseException:
            with self._mutex:
                self._waiting.remove(ticket)
            raise
        try:
            yield
        finally:
            with self._mutex:
                self._waiting.remove(ticket)


class _RoutedStream:
    """Stands in for ``sys.stdout``/``sys.stderr``, routing per context.

    Writes go to the buffer of the execution running in the current context, and
    to the real stream otherwise. Installed once and left in place, because it is
    a *pass-through* by default — unlike swapping in a buffer for the duration of
    a cell, which claims the whole process's output: a host thread logging while
    generated code ran had its line captured into the model's execution result
    and removed from where it was meant to go.

    Routing per context also removes the reason to serialize executions for
    output correctness, so concurrent runtimes stay concurrent.

    **Known boundary**: a thread the generated code starts does not inherit the
    execution context, so what it prints goes to the host's stream rather than
    into the result. Capturing it would mean wrapping every ``Thread`` target in
    the process — imposing new ``threading`` semantics on the host to serve this
    library, which is the trade this router exists to refuse. ``IPyKernelRuntime``
    captures at the process boundary and has no such gap; use it when generated
    code is expected to print from threads.
    """

    def __init__(self, fallback):
        self._fallback = fallback
        self._binary = _RoutedBinaryStream(self)

    def _target(self):
        route = _EXECUTION_OUTPUT.get()
        return self._fallback if route is None or not route.active else route.buffer

    def write(self, text: str) -> int:
        return self._target().write(text)

    def flush(self) -> None:
        target = self._target()
        if hasattr(target, "flush"):
            target.flush()

    def writelines(self, lines) -> None:
        self._target().writelines(lines)

    @property
    def buffer(self):
        # Match a normal text stream: no underlying binary layer means this
        # attribute itself is unavailable.
        if not hasattr(self._fallback, "buffer"):
            raise AttributeError("underlying stream has no binary buffer")
        return self._binary

    def __getattr__(self, name: str) -> Any:
        # ``isatty``, ``encoding``, ``fileno`` and friends describe the real
        # stream; libraries probe them to decide on colour and buffering.
        return getattr(self._fallback, name)


class _RoutedBinaryStream:
    """Binary companion to :class:`_RoutedStream`."""

    def __init__(self, text_stream: _RoutedStream) -> None:
        self._text_stream = text_stream

    def _fallback(self):
        return self._text_stream._fallback.buffer

    def write(self, data) -> int:
        route = _EXECUTION_OUTPUT.get()
        if route is None or not route.active:
            return self._fallback().write(data)

        raw = bytes(data)
        encoding = getattr(self._text_stream._fallback, "encoding", None) or "utf-8"
        route.buffer.write(raw.decode(encoding, errors="replace"))
        return len(raw)

    def writelines(self, lines) -> None:
        for line in lines:
            self.write(line)

    def flush(self) -> None:
        route = _EXECUTION_OUTPUT.get()
        if route is None or not route.active:
            self._fallback().flush()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._fallback(), name)


def _install_routing() -> None:
    """Wrap ``sys.stdout`` and ``sys.stderr``, each judged on its own.

    Using one of them as the sentinel for both gets it wrong in both
    directions: a host that later replaces only ``stderr`` leaves that stream
    unrouted, so a cell's ``stderr`` lands in the host's buffer instead of the
    result — and a host that replaces only ``stdout`` makes this re-wrap the
    already-wrapped ``stderr`` on every execution, nesting proxies until a
    write overflows the stack.
    """
    with _ROUTING_INSTALLED:
        if not isinstance(sys.stdout, _RoutedStream):
            sys.stdout = _RoutedStream(sys.stdout)
        if not isinstance(sys.stderr, _RoutedStream):
            sys.stderr = _RoutedStream(sys.stderr)


@contextmanager
def _capturing():
    """Collect this context's execution output into one buffer.

    A single buffer for both streams, so ``stderr`` is preserved *and* keeps its
    place relative to ``stdout`` — a warning printed between two lines reads as
    it happened. The kernel backend merges the two the same way.
    """
    _install_routing()
    route = _CaptureRoute()
    token = _EXECUTION_OUTPUT.set(route)
    try:
        yield route.buffer
    finally:
        route.active = False
        _EXECUTION_OUTPUT.reset(token)


def check_security(checker: SecurityChecker | None, code: str) -> "ExecutionResult | None":
    """Run security checks and return an ExecutionResult on violation, or None if clean."""
    if not checker:
        return None
    violations = checker.check_code(code)
    if not violations:
        return None
    details = [str(v) for v in violations]
    msg = f"Code execution blocked: {len(violations)} violations found:\n" + "\n".join(
        f"  - {d}" for d in details
    )
    return ExecutionResult(error=SecurityError(msg))


class RuntimeExecutionError(Exception):
    """The runtime backend failed — not the code it was asked to run.

    Code that raises is a normal outcome: it comes back as an
    :class:`ExecutionResult` with ``error`` set, the model reads the traceback
    and reacts. This is the other case — the kernel would not start, the shell
    reply timed out, the transport broke — where there is no result to report
    and nothing the model could have written differently.

    It ends the run with ``StopReason.RUNTIME_ERROR``, never ``MODEL_ERROR``:
    blaming a kernel fault on the provider corrupts metrics, retry policy and
    what the user is told.
    """


class RuntimeStateLostError(asyncio.CancelledError):
    """Cancellation completed by terminating the runtime process.

    It remains a :class:`asyncio.CancelledError`, so cancellation semantics do
    not change, but a caller waiting for cleanup can distinguish an ordinary
    interrupt from one that destroyed non-registered namespace state.
    """


class ExecutionResult:
    """
    Represents the result of code execution.
    """

    error: BaseException | None = None
    stdout: str | None = None
    state_lost: bool = False

    def __init__(
        self,
        error: BaseException | None = None,
        stdout: str | None = None,
        *,
        state_lost: bool = False,
    ):
        self.error = error
        self.stdout = stdout
        self.state_lost = state_lost

    @property
    def success(self):
        return self.error is None


class ErrorFeedbackMode(Enum):
    """Error feedback modes for LLM agent observation."""

    PLAIN = "Plain"  # Full traceback for agent debugging
    MINIMAL = "Minimal"  # Brief error info for agent efficiency


class IPythonExecutor:
    """
    Handles Python code execution using IPython.
    """

    def __init__(
        self,
        security_checker: SecurityChecker | None = None,
        error_feedback_mode: ErrorFeedbackMode = ErrorFeedbackMode.PLAIN,
    ):
        """Initialize IPython shell for code execution.

        Constructs an independent ``InteractiveShell`` rather than the
        process-wide singleton (``InteractiveShell.instance()``). Using the
        singleton would make every runtime in the process share one namespace
        and silently ignore the config of all but the first — breaking
        multi-runtime / multi-agent isolation.
        """
        ipython_config = self.create_ipython_config(error_feedback_mode=error_feedback_mode)
        self._shell = InteractiveShell(config=ipython_config)
        self._security_checker = security_checker
        # One shell is not reentrant, and execution, allocation and reset all
        # mutate its namespace. Per executor, not per process: separate
        # runtimes have separate shells, and routing output per context means
        # they no longer contend for the process's streams either.
        self._gate = _Gate()
        # Injections wait here until the next gated operation — see
        # ``inject_into_namespace``. Guarded by a plain lock because the public
        # method that fills it is synchronous and may be called from any thread.
        self._pending: dict[str, Any] = {}
        self._pending_lock = threading.Lock()
        # Highest index handed out per prefix, so a long session does not
        # rescan every previous allocation to find the next free name.
        self._allocated: dict[str, int] = {}

    def inject_many_into_namespace(
        self,
        bindings: dict[str, Any],
        *,
        commit: Callable[[], None] | None = None,
    ) -> None:
        """Queue a validated batch without exposing a partial batch."""
        prepared = {normalize_identifier(name): value for name, value in bindings.items()}
        with self._pending_lock:
            self._pending.update(prepared)
            if commit is not None:
                commit()

    def inject_into_namespace(self, name: str, value: Any):
        """Queue *value* for the namespace; landed on the next gated operation.

        Queued rather than written straight in, because this is a synchronous
        public method and the namespace is only consistent inside the gate: a
        resource registered after ``reset`` had snapshotted its bindings but
        before it took the gate was written in and then wiped, leaving the
        registry advertising a name the namespace no longer had. The kernel
        backend has always queued for the same reason.
        """
        self.inject_many_into_namespace({name: value})

    async def bind_unique(
        self,
        prefix: str,
        value: Any,
        *,
        start: int = 1,
        reserve: Callable[[str], bool] | None = None,
        on_bound: Callable[[str], None] | None = None,
        on_failed: Callable[[str], None] | None = None,
    ) -> str:
        """Bind *value* under the first free ``prefix_N`` and return that name.

        Reads the namespace the generated code actually runs in, so a name the
        model created is as visible as one that was injected. Check and bind are
        held together by :attr:`_gate` — cross-thread, fair and cancellable —
        because an absence of ``await`` only excludes coroutines on one loop,
        while the interpreter can switch OS threads between two bytecodes.
        """
        prefix = normalize_identifier(prefix)
        async with self._gate.held():
            self._flush_pending()
            index = max(start, self._allocated.get(prefix, 0) + 1)
            while True:
                name = f"{prefix}_{index}"
                if name not in self._shell.user_ns and (reserve is None or reserve(name)):
                    break
                index += 1
            try:
                self._allocated[prefix] = index
                self._shell.user_ns[name] = value
                if on_bound is not None:
                    on_bound(name)
            except BaseException:
                if on_failed is not None:
                    on_failed(name)
                raise
        return name

    async def execute(self, code: str) -> ExecutionResult:
        """Execute *code*, capturing what it writes to stdout and stderr.

        Serialized against this executor's other shell work only — routing
        output per context means separate runtimes no longer interfere.

        Args:
            code: Python code to execute

        Returns:
            ExecutionResult with success status and output or error
        """
        try:
            violation = check_security(self._security_checker, code)
            if violation:
                return violation

            async with self._gate.held():
                self._flush_pending()
                with _capturing() as output:
                    transformed_code = self._shell.transform_cell(code)
                    result = await self._shell.run_cell_async(
                        transformed_code, transformed_cell=transformed_code
                    )

            error = result.error_before_exec or result.error_in_exec
            if isinstance(error, asyncio.CancelledError):
                # IPython catches cancellation and hands it back as an ordinary
                # result. Returned as one, the agent carried on working after
                # its task was cancelled and reported COMPLETED.
                raise error
            return ExecutionResult(error=error, stdout=output.getvalue())

        except asyncio.CancelledError:
            raise
        except Exception as e:
            return ExecutionResult(error=e)

    async def get_from_namespace(self, name: str) -> Any:
        """Get a value from the execution namespace.

        Normalized because this backend resolves names with a dict lookup while
        generated code goes through the interpreter, which applies NFKC first.
        Without it the two disagree about which variable a name refers to.
        """
        async with self._gate.held():
            self._flush_pending()
            namespace = self._shell.user_ns
            key = normalize_identifier(name)
            if key not in namespace:
                raise KeyError(f"'{name}' is not bound in the namespace")
            return namespace[key]

    def _flush_pending(self) -> None:
        """Move queued injections into the namespace. Caller holds the gate."""
        with self._pending_lock:
            pending, self._pending = self._pending, {}
        self._shell.user_ns.update(pending)

    async def reset(
        self,
        bindings: dict[str, Any] | Callable[[], dict[str, Any]] | None = None,
    ):
        """Clear the namespace and restore *bindings*, as one step.

        The restore is part of the transaction, not a follow-up by the caller:
        clearing under the gate and re-injecting outside it leaves a window
        where the namespace is genuinely empty, and an execution admitted there
        fails on names the runtime promises are always present.
        """
        async with self._gate.held():
            restored = bindings() if callable(bindings) else (bindings or {})
            self._shell.reset()
            gc.collect()
            for name, value in restored.items():
                self._shell.user_ns[normalize_identifier(name)] = value
            # Last, so a registration made while the reset was queueing lands
            # *after* the restore rather than being wiped by it.
            self._flush_pending()

    @staticmethod
    def create_ipython_config(
        error_feedback_mode: ErrorFeedbackMode = ErrorFeedbackMode.PLAIN,
    ) -> Config:
        """Create a clean IPython configuration optimized for code execution."""
        config = Config()
        # Disable the history subsystem outright. ``history_length = 0`` only
        # bounds what is retained — the HistoryManager is still constructed, and
        # each one spawns an `IPythonHistorySavingThread` holding a SQLite
        # connection. Those are per-shell and never reaped, so every runtime
        # created and dropped leaked a thread and a pair of file descriptors,
        # and a long-lived process building many runtimes accumulated both
        # without bound. Nothing here reads the history back, so the whole
        # subsystem is dead weight.
        config.HistoryAccessor.enabled = False
        config.InteractiveShell.cache_size = 0
        config.InteractiveShell.history_length = 0
        config.InteractiveShell.automagic = False
        config.InteractiveShell.separate_in = ""
        config.InteractiveShell.separate_out = ""
        config.InteractiveShell.separate_out2 = ""
        config.InteractiveShell.autocall = 0
        config.InteractiveShell.colors = "nocolor"
        config.InteractiveShell.xmode = error_feedback_mode.value
        config.InteractiveShell.quiet = True
        config.InteractiveShell.autoindent = False

        return config
