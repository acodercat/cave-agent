"""Code executor using a separate IPython kernel process for isolation."""

from __future__ import annotations

import ast
import asyncio
import base64
import logging
import threading
import time
from collections.abc import Awaitable, Callable
from contextlib import asynccontextmanager, contextmanager
from typing import Any

import dill
from jupyter_client.manager import AsyncKernelManager

from .._placeholders import normalize_identifier
from ..security import SecurityChecker
from .executor import (
    ErrorFeedbackMode,
    ExecutionResult,
    RuntimeExecutionError,
    RuntimeStateLostError,
    check_security,
)

logger = logging.getLogger(__name__)

# Timeout constants (seconds)
_KERNEL_READY_TIMEOUT = 60.0
_SHELL_REPLY_TIMEOUT = 10.0

# How long ``stop()`` lets an in-flight request finish before tearing the
# channels down anyway. Matched to DEFAULT_IOPUB_TIMEOUT so a request that is
# going to fail on its own gets to do so first, and shutdown reports one clear
# outcome instead of two racing ones.
_STOP_GRACE_TIMEOUT = 30.0

# How long an abandoned request gets to reach `idle` after SIGINT before the
# kernel is terminated. A cell can catch KeyboardInterrupt, so releasing the
# channel after a longer best-effort wait is neither safe nor a real timeout.
_INTERRUPT_GRACE_TIMEOUT = 0.25

# How many times teardown asks the kernel to die before giving up and saying so.
# A transport hiccup on the first attempt is common; a process that survives
# three is not going to be talked down.
_SHUTDOWN_ATTEMPTS = 3

# How long a request waits for another request's recovery before giving up.
# Derived from the teardown budget it is waiting on, so the two cannot drift.
_RECOVERY_WAIT_TIMEOUT = _STOP_GRACE_TIMEOUT + 10.0

# Poll interval while waiting for that recovery.
_GATE_POLL_SECONDS = 0.005

# Key for the Jupyter user expression carrying a value back on the shell reply.
_USER_EXPRESSION_KEY = "cave_value"

# How many times a request may lose the race against a lifecycle transition
# before giving up. One retry covers a transition that slipped in; more means
# something is tearing the kernel down as fast as we start it.
_ADMISSION_ATTEMPTS = 3

# Default ceiling on silence between two IOPub messages for one execution.
# This bounds how long a cell may produce *nothing* — not its total runtime,
# since every message resets the window. A cell that prints as it goes can run
# for hours; one that computes silently is cut off here. Configurable per
# runtime (``IPyKernelRuntime(iopub_timeout=...)``) because the right value is
# workload-specific: a long silent model fit legitimately needs more.
DEFAULT_IOPUB_TIMEOUT = 30.0

# Infrastructure lives under a dictionary key that cannot be a Python
# identifier, so generated code cannot shadow it with an assignment. The
# callables capture their dependencies while the fresh kernel still has intact
# builtins; later assignments to ``__builtins__``, ``__import__``,
# ``get_ipython`` or ``globals`` therefore cannot break transport.
_KERNEL_CODEC_KEY = "\0cave-agent-codec"

# Executed once after kernel starts to configure IPython behavior
_KERNEL_SETUP = f"""\
import base64 as _cave_base64
import dill as _cave_dill
import concurrent.futures as _cave_futures
import sys as _cave_sys
import threading as _cave_threading
def _cave_dump(value, _dumps=_cave_dill.dumps, _encode=_cave_base64.b64encode):
    return _encode(_dumps(value)).decode()
def _cave_load(payload, _loads=_cave_dill.loads, _decode=_cave_base64.b64decode):
    return _loads(_decode(payload))
def _cave_capture_output_parents(
    _streams=(_cave_sys.stdout, _cave_sys.stderr),
    _hasattr=hasattr,
    _tuple=tuple,
):
    return _tuple(
        (stream._parent_header, stream.parent_header)
        for stream in _streams
        if _hasattr(stream, "_parent_header")
    )
def _cave_run_with_output_parent(fn, args, kwargs, parents):
    tokens = []
    try:
        for context, parent in parents:
            tokens.append((context, context.set(parent)))
        return fn(*args, **kwargs)
    finally:
        for context, token in tokens[::-1]:
            context.reset(token)
_cave_original_thread_start = _cave_threading.Thread.start
def _cave_thread_start(
    self,
    _start=_cave_original_thread_start,
    _capture=_cave_capture_output_parents,
    _run_with_parent=_cave_run_with_output_parent,
):
    parents = _capture()
    run = self.run
    def run_with_output_parent(
        _run=run,
        _parents=parents,
        _thread=self,
        _apply=_run_with_parent,
    ):
        try:
            return _apply(_run, (), {{}}, _parents)
        finally:
            _thread.run = _run
    self.run = run_with_output_parent
    started = False
    try:
        result = _start(self)
        started = True
        return result
    finally:
        if not started:
            self.run = run
_cave_threading.Thread.start = _cave_thread_start
def _cave_make_pool_submit(submit, capture, run_with_parent):
    def pool_submit(self, fn, /, *args, **kwargs):
        return submit(
            self,
            run_with_parent,
            fn,
            args,
            kwargs,
            capture(),
        )
    return pool_submit
_cave_pool_submit = _cave_make_pool_submit(
    _cave_futures.ThreadPoolExecutor.submit,
    _cave_capture_output_parents,
    _cave_run_with_output_parent,
)
_cave_futures.ThreadPoolExecutor.submit = _cave_pool_submit
_cave_original_add_done_callback = _cave_futures.Future.add_done_callback
def _cave_add_done_callback(
    self,
    fn,
    _add=_cave_original_add_done_callback,
    _capture=_cave_capture_output_parents,
    _run_with_parent=_cave_run_with_output_parent,
):
    parents = _capture()
    def callback_with_output_parent(
        future,
        _fn=fn,
        _parents=parents,
        _apply=_run_with_parent,
    ):
        return _apply(_fn, (future,), {{}}, _parents)
    return _add(self, callback_with_output_parent)
_cave_futures.Future.add_done_callback = _cave_add_done_callback
ip = get_ipython()
ip.user_ns[{_KERNEL_CODEC_KEY!r}] = (_cave_dump, _cave_load)
ip.colors = "nocolor"
ip.InteractiveTB.set_mode("Plain")
del (
    _cave_base64, _cave_dill, _cave_futures, _cave_sys, _cave_threading,
    _cave_capture_output_parents, _cave_run_with_output_parent,
    _cave_original_thread_start, _cave_thread_start,
    _cave_make_pool_submit, _cave_pool_submit,
    _cave_original_add_done_callback, _cave_add_done_callback,
    _cave_dump, _cave_load, ip,
)
"""

_USER_NAMESPACE = "(lambda: None).__globals__"
_KERNEL_DUMP = f"{_USER_NAMESPACE}[{_KERNEL_CODEC_KEY!r}][0]"
_KERNEL_LOAD = f"{_USER_NAMESPACE}[{_KERNEL_CODEC_KEY!r}][1]"


async def _finish_despite_cancellation[T](
    awaitable: Awaitable[T],
) -> tuple[T, asyncio.CancelledError | None]:
    """Let *awaitable* finish, remembering cancellation for its caller.

    Shielding once is insufficient: a task may be cancelled repeatedly while
    it is releasing a request or process. The child keeps running until it has
    established a stable state; the caller then performs any synchronous commit
    and re-raises the first cancellation.
    """
    task = asyncio.ensure_future(awaitable)
    cancellation: asyncio.CancelledError | None = None
    while True:
        try:
            return await asyncio.shield(task), cancellation
        except asyncio.CancelledError as error:
            current = asyncio.current_task()
            if current is None or not current.cancelling():
                return task.result(), cancellation
            if cancellation is None:
                cancellation = error
            if task.done():
                try:
                    return task.result(), cancellation
                except BaseException as task_error:
                    raise cancellation from task_error
        except BaseException as task_error:
            if cancellation is not None:
                raise cancellation from task_error
            raise


def _dill_expression(expression: str) -> str:
    """Wrap *expression* so the kernel returns its value as a base64 dill blob.

    Evaluated as a Jupyter user expression. The hidden codec function captured
    its dependencies at kernel setup, and the namespace is reached through a
    fresh function object's globals rather than the rebindable ``globals`` or
    ``get_ipython`` names.
    """
    return f"{_KERNEL_DUMP}({expression})"


def _orphaned_kernel_error() -> RuntimeExecutionError:
    """The failure both lifecycle transitions report when the kernel outlives them."""
    return RuntimeExecutionError(
        f"Kernel is still running after {_SHUTDOWN_ATTEMPTS} shutdown attempts. "
        "Its manager is retained, so calling stop() again retries; until it exits "
        "this runtime refuses to start another kernel."
    )


def _make_injection_code(name: str, value: Any) -> str:
    """Generate code that recreates *value* as *name* inside the kernel.

    Uses dill for serialization, which handles local functions, closures,
    lambdas, and most Python objects that standard pickle cannot. Binds only
    *name* — nothing else is left behind.
    """
    data = base64.b64encode(dill.dumps(value, recurse=True)).decode()
    return f"{name} = {_KERNEL_LOAD}({data!r})"


class IPyKernelExecutor:
    """Executes Python code in an isolated IPython kernel process.

    Same interface as ``IPythonExecutor`` so it can be used as a drop-in
    replacement inside ``Runtime`` / ``IPyKernelRuntime``.

    **Concurrency.** A kernel runs one cell at a time over a single shared
    IOPub queue, so overlapping requests would interleave and could consume
    each other's output. Two locks keep that honest:

    - ``_lifecycle_lock`` — serializes ``start`` / ``stop`` / ``reset``, so
      concurrent first calls collapse into one kernel and two resets cannot
      tear down each other's manager.
    - ``_channel_lock`` — requests (``execute`` / ``get_from_namespace``) are
      serialized, and a lifecycle transition holds it for its *whole* duration
      so teardown does not yank the channels out from under a running request.
      That wait is bounded (:data:`_STOP_GRACE_TIMEOUT`), so a hung cell delays
      shutdown but cannot prevent it.

    A transition is atomic across the channel: ``reset`` does not release the
    lock between tearing the old kernel down and bringing the new one up, so a
    request queued behind it resumes against a live kernel instead of the
    ``None`` client that briefly sits in between. Acquisition order is always
    lifecycle-then-channel, and a request never reaches back for the lifecycle
    lock while holding the channel, so the two cannot deadlock.

    ``interrupt()`` deliberately stays lock-free: its whole purpose is to be
    callable *while* a cell is running, so it is also how you make a shutdown
    immediate — ``interrupt()`` then ``stop()``.
    """

    def __init__(
        self,
        security_checker: SecurityChecker | None = None,
        error_feedback_mode: ErrorFeedbackMode = ErrorFeedbackMode.PLAIN,
        iopub_timeout: float | None = None,
    ):
        self._km = AsyncKernelManager(kernel_name="python3")
        self._kernel_client: Any = None
        self._security_checker = security_checker
        self._error_feedback_mode = error_feedback_mode
        self._iopub_timeout = DEFAULT_IOPUB_TIMEOUT if iopub_timeout is None else iopub_timeout
        self._started = False
        # A kernel that survived its own shutdown. Distinct from "not started":
        # the process is alive and this manager is the only handle to it.
        self._orphaned = False
        self._lifecycle_lock = asyncio.Lock()
        # The loop currently driving this kernel, and how deep. See
        # ``_single_loop``.
        self._active_loop: asyncio.AbstractEventLoop | None = None
        self._active_depth = 0
        self._active_lock = threading.Lock()
        self._channel_lock = asyncio.Lock()
        # Set under the channel before an unresponsive cell releases it. New
        # admissions poll until teardown finishes instead of slipping into the
        # gap and sending work to the kernel that is about to be killed.
        self._recovering = False
        self._pending_lock = threading.Lock()
        self._pending_injections: dict[str, str] = {}
        # A dispatched injection that failed cannot be replayed safely: dill
        # reconstruction is arbitrary Python and may already have produced an
        # external side effect. Refuse further requests until that name is
        # explicitly replaced or reset rebuilds the complete binding set.
        self._failed_injections: set[str] = set()
        # Last value successfully landed for every persistent binding. Unlike
        # ``_pending_injections``, this survives a flush: stop() must be able to
        # rebuild a namespace without re-serializing mutable host objects while
        # it is trying to release the process.
        self._restore_injections: dict[str, str] = {}
        # Highest index handed out per prefix — see ``bind_unique``.
        self._allocated: dict[str, int] = {}

    @contextmanager
    def _single_loop(self):
        """Refuse to run while another event loop is mid-operation here.

        A kernel client's channels, futures and locks belong to the loop that
        created them, and ``jupyter_client``'s do too — so this is not a
        limitation that two ``asyncio`` locks could lift. *Overlapping* loops
        simply hung, and the failure surfaced later and elsewhere as "lock is
        bound to a different event loop".

        Only overlap is refused. Driving one runtime from a succession of loops
        — a module-scoped fixture with a loop per test, a script calling
        ``asyncio.run`` more than once — is ordinary and safe, because nothing
        is in flight across the boundary.
        """
        loop = asyncio.get_running_loop()
        with self._active_lock:
            if self._active_depth and self._active_loop is not loop:
                raise RuntimeExecutionError(
                    "This kernel runtime is busy on another event loop. A "
                    "kernel's channels belong to the loop that created them; "
                    "use one runtime per loop, or marshal your calls onto the "
                    "loop already driving it."
                )
            self._active_loop = loop
            self._active_depth += 1
        try:
            yield
        finally:
            with self._active_lock:
                self._active_depth -= 1

    async def start(self) -> None:
        """Start the kernel subprocess and wait until it is ready.

        Called explicitly, or on demand by the first operation that needs a live
        kernel — which is what makes deferral the default: constructing the
        runtime, registering resources and rendering the system prompt all stay
        free, and a conversation that never runs code never spawns anything.
        Teardown stays the caller's (``async with`` or :meth:`stop`).

        Idempotent and concurrency-safe: the guard is re-checked under
        ``_lifecycle_lock``, so any mix of explicit and on-demand callers
        collapses into one kernel. The lock lives here rather than at the call
        sites because a bare flag check is not atomic — two concurrent
        ``start()``s both got past it and raced inside ``jupyter_client``,
        which sets its readiness future twice and raises ``InvalidStateError``.

        Cleans up after itself if any part of startup fails. ``__aenter__``
        raising means ``__aexit__`` is never called, so without this a kernel
        that never becomes ready would leak its subprocess and ZMQ channels —
        and a caller that retries would leak one per attempt.
        """
        if self._started:
            return
        with self._single_loop():
            async with self._lifecycle_lock:
                if self._started:
                    return
                await self._start_locked()

    async def _start_locked(self) -> None:
        """Actual startup. Caller must hold ``_lifecycle_lock``.

        Refuses to run while a previous kernel is known to have survived its
        shutdown. ``_started`` cannot encode that — it is false in both cases —
        and starting on a manager that still holds a live process spawns a
        replacement that dies on the occupied ports while execution silently
        reconnects to the old kernel.
        """
        if self._orphaned:
            if await self._km.is_alive():
                # Typed like every other runtime failure: ``start()`` and
                # ``__aenter__`` sit outside the ``RuntimeExecutionError``
                # envelope, so a bare ``RuntimeError`` here slipped past callers
                # catching the documented type — ``IPyKernelRuntime.__aexit__``
                # among them. The message stays start-specific.
                raise RuntimeExecutionError(
                    "A previous kernel survived shutdown and is still running; "
                    "refusing to start another that would take its place. Call "
                    "stop() again to retry the shutdown."
                )
            # It exited on its own after all.
            self._orphaned = False
            self._km = AsyncKernelManager(kernel_name="python3")
        try:
            await self._km.start_kernel()
            self._kernel_client = self._km.client()
            self._kernel_client.start_channels()

            # Readiness is delegated, not hand-rolled: a kernel_info_reply on
            # the shell channel does not prove IOPub is usable. IOPub is ZMQ
            # PUB/SUB, and a PUB socket silently drops anything published before
            # the subscription has propagated, so a shell-only check returns
            # while IOPub is still dark and the first cell's output is lost.
            # ``wait_for_ready`` waits for a message on IOPub itself.
            try:
                await self._kernel_client.wait_for_ready(timeout=_KERNEL_READY_TIMEOUT)
            except (RuntimeError, TimeoutError) as error:
                raise RuntimeExecutionError(
                    f"Kernel failed its readiness check: {error}"
                ) from error

            await self._execute_silent(_KERNEL_SETUP)

            # Last, so that ``start()``'s lock-free fast path cannot let another
            # task reach the channels while setup is still in flight — setup
            # does not hold ``_channel_lock``, so an early flag would let the
            # two interleave.
            self._started = True
        except BaseException as start_error:
            # `_stop_locked`, not `stop()`: we already hold `_lifecycle_lock`
            # and `asyncio.Lock` is not reentrant, so the public entry point
            # deadlocks here — and the failure it was cleaning up after is
            # never raised, leaving a live kernel nobody holds a handle to.
            # Nothing can be using the channel: every request waits on the
            # lifecycle lock before touching it.
            _, cancellation = await _finish_despite_cancellation(
                self._stop_locked(),
            )
            self._recovering = False
            if cancellation is not None and not isinstance(start_error, asyncio.CancelledError):
                raise cancellation from start_error
            raise

    async def interrupt(self) -> None:
        """Send SIGINT to the kernel to interrupt running execution."""
        if self._km and await self._km.is_alive():
            await self._km.interrupt_kernel()

    async def stop(self) -> None:
        """Shut down the kernel, letting an in-flight request finish first.

        Graceful by default, which is the Python convention for releasing a
        resource — ``Executor.shutdown`` waits, ``asyncio.Server.wait_closed``
        waits, and ``KernelManager.shutdown_kernel`` takes ``now=False`` as
        *its* default. Tearing the channels down under a running request would
        fail it with ``CancelledError`` for no reason the caller can act on.

        The wait is bounded by :data:`_STOP_GRACE_TIMEOUT` so a runaway cell
        cannot block shutdown forever; past that the teardown proceeds anyway.
        To stop *immediately*, call :meth:`interrupt` first — that is precisely
        what it is for, and it composes without inventing a second shutdown
        mode for callers to choose between.

        Idempotent, and safe on a kernel that was never started — which is how
        a deferred runtime is usually torn down.
        """
        with self._single_loop():
            async with self._lifecycle_lock:
                drained = await self._acquire_channel_for_transition()
                try:
                    orphaned, cancellation = await _finish_despite_cancellation(
                        self._stop_locked(),
                    )
                    with self._pending_lock:
                        restorable = {
                            name: code
                            for name, code in self._restore_injections.items()
                            if name not in self._failed_injections
                        }
                        self._pending_injections = restorable | self._pending_injections
                finally:
                    if drained:
                        self._channel_lock.release()

        if cancellation is not None:
            raise cancellation
        if orphaned:
            raise _orphaned_kernel_error()

    async def _stop_locked(self) -> bool:
        """Actual teardown; returns whether the kernel outlived it.

        Caller holds ``_lifecycle_lock`` and the channel.

        The channel and the process are released **independently**: one ``try``
        around both lets a channel failure skip the shutdown entirely.

        The manager is replaced only once the process is confirmed dead. A
        shut-down manager cannot be started again — its ZMQ context is closed
        and ``start_kernel`` on it hangs — but keeping a *live* kernel's manager
        is the lesser evil, since it is the only handle left to retry with.
        """
        try:
            if self._kernel_client is not None:
                self._kernel_client.stop_channels()
        except Exception:
            logger.warning("Failed to release the kernel channels", exc_info=True)
        finally:
            self._kernel_client = None

        # Committed before termination starts, and pessimistically: the client
        # is already gone, so nothing may claim the kernel is usable. Public
        # lifecycle operations shield termination through repeated cancellation
        # and correct this state once the process outcome is known.
        self._started = False
        self._orphaned = True

        alive = await self._terminate_kernel()
        self._orphaned = alive
        if not alive:
            self._km = AsyncKernelManager(kernel_name="python3")
        return alive

    async def _terminate_kernel(self) -> bool:
        """Ask the kernel to die, up to :data:`_SHUTDOWN_ATTEMPTS` times.

        Returns whether it is *still alive*. A single transport failure is
        common and recoverable; reporting success while the process runs on
        leaves a caller walking away from it.
        """
        for attempt in range(1, _SHUTDOWN_ATTEMPTS + 1):
            try:
                if not await self._km.is_alive():
                    return False
                await self._km.shutdown_kernel(now=True)
                if not await self._km.is_alive():
                    return False
            except Exception:
                logger.warning(
                    "Kernel shutdown attempt %d/%d failed",
                    attempt,
                    _SHUTDOWN_ATTEMPTS,
                    exc_info=True,
                )
        return True

    async def _acquire_channel_for_transition(self) -> bool:
        """Take the channel for a lifecycle transition, warning if it times out."""
        drained = await self._acquire_channel(_STOP_GRACE_TIMEOUT)
        if not drained:
            logger.warning(
                "Kernel lifecycle transition proceeding with a request still in flight after %.0fs",
                _STOP_GRACE_TIMEOUT,
            )
        return drained

    async def _acquire_channel(self, timeout: float) -> bool:
        """Take ``_channel_lock``, giving up after *timeout*. True if acquired.

        Lets teardown wait out an in-flight request without letting a hung one
        wedge it. The caller releases only when this returned True.
        """
        try:
            await asyncio.wait_for(self._channel_lock.acquire(), timeout)
            return True
        except TimeoutError:
            return False

    @staticmethod
    def _prepare_injections(bindings: dict[str, Any]) -> dict[str, str]:
        """Serialize a complete batch before making any of it visible."""
        prepared = {}
        for name, value in bindings.items():
            normalized = normalize_identifier(name)
            prepared[normalized] = _make_injection_code(normalized, value)
        return prepared

    def inject_many_into_namespace(
        self,
        bindings: dict[str, Any],
        *,
        commit: Callable[[], None] | None = None,
    ) -> None:
        """Queue an all-or-nothing serialized batch for the next request."""
        prepared = self._prepare_injections(bindings)
        with self._pending_lock:
            self._pending_injections.update(prepared)
            self._failed_injections.difference_update(prepared)
            if commit is not None:
                commit()

    def inject_into_namespace(self, name: str, value: Any) -> None:
        """Queue *value* for injection.  Actually sent on the next ``execute()``."""
        self.inject_many_into_namespace({name: value})

    async def get_from_namespace(self, name: str) -> Any:
        """Retrieve a variable from the kernel namespace via dill."""
        try:
            async with self._admitted():
                await self._flush_injections()
                payload, failure = await self._evaluate(name)
        except BaseException as error:
            # This path never abandons a cell, so it never destroys state — it
            # may still perform a pending recovery another request left behind,
            # but it does not claim that failure as its own.
            cancellation = await self._recover_if_needed()
            if cancellation is not None and not isinstance(error, asyncio.CancelledError):
                raise cancellation from error
            raise

        if failure is not None:
            raise KeyError(f"Variable '{name}': {failure}")
        if payload is None:
            raise KeyError(f"Variable '{name}' not found")
        return dill.loads(base64.b64decode(payload))

    async def _evaluate(self, expression: str) -> tuple[str | None, str | None]:
        """Evaluate *expression* in the kernel. Caller holds the channel.

        Runs as a Jupyter user expression against an empty cell, so the answer
        arrives on the shell reply and never touches stdout, the display hooks,
        or any name the generated code can rebind.

        Returns ``(payload, failure)`` with the payload **undecoded**, so
        ``None`` means "nothing came back" and nothing else — decoding here
        would collapse that with a variable whose value genuinely is ``None``,
        which is what a declared-but-unset ``Variable`` holds.
        """
        msg_id = self._kernel_client.execute(
            "",
            silent=True,
            user_expressions={_USER_EXPRESSION_KEY: _dill_expression(expression)},
        )
        reply, cancellation = await _finish_despite_cancellation(
            self._request_reply(msg_id, "namespace request"),
        )
        if cancellation is not None:
            raise cancellation

        result = reply["content"].get("user_expressions", {}).get(_USER_EXPRESSION_KEY)
        if result is None:
            return None, "the kernel returned no result for the expression"
        if result.get("status") != "ok":
            return None, f"{result.get('ename', 'Error')}: {result.get('evalue', '')}"
        # ``text/plain`` of a str expression is its repr; literal_eval is the
        # exact inverse and refuses anything that is not a literal.
        return ast.literal_eval(result["data"]["text/plain"]), None

    @asynccontextmanager
    async def _admitted(self):
        """Hold the channel with a live kernel — as one step, not two.

        ``start()`` has a lock-free fast path, so "start, then take the channel"
        is not atomic: a request can pass the flag check, queue behind a
        ``stop()`` already under way, and wake up holding the channel with no
        client. Re-checking under the channel settles it, because a lifecycle
        transition holds that same lock for its whole duration.

        A transition that slipped in is worth one more try; losing the race
        repeatedly means something is stopping the kernel as fast as this can
        start it, and saying so beats spawning kernels in a loop.
        """
        with self._single_loop():
            attempts = 0
            recovery_deadline = time.monotonic() + _RECOVERY_WAIT_TIMEOUT
            while attempts < _ADMISSION_ATTEMPTS:
                with self._pending_lock:
                    failed = sorted(self._failed_injections)
                if failed:
                    names = ", ".join(failed)
                    raise RuntimeExecutionError(
                        "Persistent kernel bindings failed while being "
                        f"restored: {names}. Replace those bindings or "
                        "reset the runtime before executing more code."
                    )
                await self.start()
                async with self._channel_lock:
                    wait_for_recovery = self._recovering
                    if self._started and not wait_for_recovery:
                        yield
                        return
                if wait_for_recovery:
                    # Bounded by the clock, not by attempts: a recovery is a
                    # kernel teardown and legitimately takes seconds, but the
                    # flag is cleared by the task that set it — and if that task
                    # was destroyed, an unbounded wait span at 200 polls/sec
                    # forever instead of reporting anything.
                    if time.monotonic() >= recovery_deadline:
                        raise RuntimeExecutionError(
                            "A previous request's kernel recovery never completed; "
                            "reset or recreate this runtime."
                        )
                    await asyncio.sleep(_GATE_POLL_SECONDS)
                    continue
                attempts += 1
        raise RuntimeExecutionError(
            f"Could not hold the kernel channel: it was stopped during each of "
            f"{_ADMISSION_ATTEMPTS} attempts to start it."
        )

    async def _recover_if_needed(self) -> asyncio.CancelledError | None:
        """Terminate a request that could not be reclaimed after channel release."""
        if not self._recovering:
            return None
        try:
            _, cancellation = await _finish_despite_cancellation(self.stop())
            return cancellation
        finally:
            self._recovering = False

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
        """Bind *value* under the first free ``prefix_N`` in the kernel.

        Two steps — ask which names are taken, then inject under one that is
        not — held together by a single unbroken grip on ``_channel_lock``.
        Nothing in this runtime reaches the kernel without that lock, including
        the model's own code, so the pair is indivisible.
        """
        prefix = normalize_identifier(prefix)
        committed = False
        reserved = False
        name = ""
        try:
            async with self._admitted():
                await self._flush_injections()
                taken = await self._names_starting_with(prefix)
                index = max(start, self._allocated.get(prefix, 0) + 1)
                while True:
                    name = f"{prefix}_{index}"
                    if name not in taken and (reserve is None or reserve(name)):
                        reserved = True
                        break
                    index += 1

                self._allocated[prefix] = index
                injection = _make_injection_code(name, value)
                _, cancellation = await _finish_despite_cancellation(
                    self._execute_silent(injection),
                )
                if on_bound is not None:
                    on_bound(name)
                with self._pending_lock:
                    self._restore_injections[name] = injection
                committed = True
                if cancellation is not None:
                    raise cancellation
        except BaseException as error:
            if reserved and not committed and on_failed is not None:
                on_failed(name)
            # This path finishes its injection despite cancellation rather
            # than abandoning a cell, so it never destroys state. It may still
            # complete a recovery another request left pending — but it does
            # not report that failure as its own.
            cancellation = await self._recover_if_needed()
            if cancellation is not None and not isinstance(error, asyncio.CancelledError):
                raise cancellation from error
            raise
        return name

    async def _names_starting_with(self, prefix: str) -> set[str]:
        """Names currently bound in the kernel that begin with *prefix*.

        Read-only, and the comprehension's variable lives in its own scope, so
        asking the question changes nothing about the answer.

        Reached through :data:`_USER_NAMESPACE` rather than ``globals()``, which
        is itself an ordinary name the generated code can rebind.
        """
        payload, failure = await self._evaluate(
            f"{{k for k in {_USER_NAMESPACE} if k.startswith({prefix!r})}}"
        )
        if failure is not None:
            raise RuntimeError(f"Could not list '{prefix}*' names: {failure}")
        if payload is None:
            raise RuntimeError(f"Kernel did not answer the '{prefix}*' name query")
        return dill.loads(base64.b64decode(payload))

    async def execute(self, code: str) -> ExecutionResult:
        """Execute *code* on the kernel, starting it on first use.

        Concurrent calls are serialized — see ``_channel_lock``.
        """
        # Checked before admission: a rejection touches no channel, so a
        # blocked call shouldn't wait behind a running cell to be told no.
        violation = check_security(self._security_checker, code)
        if violation:
            return violation

        result: ExecutionResult | None = None
        cancellation: asyncio.CancelledError | None = None
        restart_required = False
        cell_idle = False

        try:
            async with self._admitted():
                await self._flush_injections()
                msg_id = self._kernel_client.execute(code)
                try:
                    result = await self._collect_result(msg_id)
                    cell_idle = True
                    _, reap_cancellation = await _finish_despite_cancellation(
                        self._reap_shell_reply(msg_id),
                    )
                    if reap_cancellation is not None:
                        cancellation = reap_cancellation
                except TimeoutError:
                    (reclaimed, _), cleanup_cancellation = await _finish_despite_cancellation(
                        self._abandon(msg_id)
                    )
                    if cleanup_cancellation is not None:
                        cancellation = cleanup_cancellation
                    restart_required = not reclaimed
                    self._recovering = restart_required
                    message = (
                        f"Execution interrupted after producing no output for "
                        f"{self._iopub_timeout}s. Print progress periodically, or raise "
                        "iopub_timeout on the runtime."
                    )
                    result = ExecutionResult(
                        error=TimeoutError(message),
                        stdout=f"TimeoutError: {message}",
                        state_lost=restart_required,
                    )
                except asyncio.CancelledError as error:
                    cancellation = error
                    # Once IOPub reported idle, the cell is finished and its
                    # shell reply is reaped above before cancellation propagates.
                    if cell_idle:
                        restart_required = False
                    else:
                        (reclaimed, _), _ = await _finish_despite_cancellation(
                            self._abandon(msg_id),
                        )
                        restart_required = not reclaimed
                    self._recovering = restart_required
        except BaseException as error:
            # ``restart_required`` is *this* request's own answer. Reading the
            # shared ``_recovering`` flag instead let a request that never
            # dispatched anything report that its cancellation had destroyed
            # the namespace, when another request's cell had.
            state_lost = restart_required
            recovery_cancellation = await self._recover_if_needed()
            if recovery_cancellation is not None and not isinstance(error, asyncio.CancelledError):
                raise recovery_cancellation from error
            if state_lost and isinstance(error, asyncio.CancelledError):
                raise RuntimeStateLostError(
                    "Cancelling this execution terminated the runtime process"
                ) from error
            raise

        if restart_required:
            try:
                # The channel is no longer held, so the ordinary lifecycle path
                # can terminate the uncooperative process safely. Bindings are
                # queued by stop() and restored when the next request starts.
                recovery_cancellation = await self._recover_if_needed()
                if cancellation is None:
                    cancellation = recovery_cancellation
            except Exception:
                if cancellation is None:
                    raise
                logger.warning(
                    "Kernel shutdown failed while execution cancellation was propagating",
                    exc_info=True,
                )

        if cancellation is not None:
            if restart_required:
                raise RuntimeStateLostError(
                    "Cancelling execution terminated the runtime process"
                ) from cancellation
            raise cancellation
        assert result is not None
        return result

    def _prepare_reset_injections(
        self,
        bindings: dict[str, Any] | Callable[[], dict[str, Any]] | None,
    ) -> dict[str, str]:
        """Build the complete restore batch for :meth:`reset`."""
        if bindings is None:
            with self._pending_lock:
                return dict(self._restore_injections)
        values = bindings() if callable(bindings) else bindings
        return self._prepare_injections(values)

    def _stage_reset_injections(self, restored: dict[str, str]) -> None:
        """Queue a reset batch, preserving registrations made while it was built."""
        with self._pending_lock:
            self._pending_injections = restored | self._pending_injections
            self._failed_injections.clear()

    async def reset(
        self,
        bindings: dict[str, Any] | Callable[[], dict[str, Any]] | None = None,
    ) -> None:
        """Restart the kernel, then queue *bindings* for the new namespace.

        A fresh kernel starts empty and the injections are queued rather than
        sent, so they land on the first request that needs them — before any
        generated code can observe the namespace. The in-process backend has to
        hold a lock to get the same guarantee; here a new process provides it.

        A no-op when nothing was ever started: there is no state to clear, and
        starting a kernel here purely to reset it would defeat on-demand
        creation. A stopped runtime carrying a failed persistent binding still
        rebuilds its queued bindings, but remains stopped until the next
        request. Like :meth:`stop`, reset waits out an in-flight request on the
        same event loop (bounded by :data:`_STOP_GRACE_TIMEOUT`) rather than
        tearing the channels out from under one. An overlapping request from a
        different loop is rejected before it touches the channels: those
        futures cannot safely be awaited from here.

        Tears the kernel down and builds a fresh one rather than using
        ``KernelManager.restart_kernel``: that reuses the connection file, but
        leaves the existing client's channels pointed at a process that has
        gone away, and cells then run against a namespace that was never
        actually cleared. Measured, it failed ``test_reset_clears_state``
        outright — the rebuild is slower but correct.

        Teardown and startup happen under one continuous hold of the channel,
        so a request queued behind a reset resumes against the new kernel. When
        the two halves each took the lock separately, a queued ``execute``
        slipped into the gap and found a ``None`` client.
        """
        with self._single_loop():
            async with self._lifecycle_lock:
                if not self._started and not self._orphaned:
                    with self._pending_lock:
                        has_failed_binding = bool(self._failed_injections)
                    if not has_failed_binding:
                        return
                    restored = self._prepare_reset_injections(bindings)
                    self._stage_reset_injections(restored)
                    return
                drained = await self._acquire_channel_for_transition()
                try:
                    restored = self._prepare_reset_injections(bindings)
                    orphaned, cancellation = await _finish_despite_cancellation(
                        self._stop_locked(),
                    )
                    if orphaned:
                        raise _orphaned_kernel_error()
                    # Registrations made while reset was in flight describe
                    # newer intent than its pre-reset binding snapshot.
                    self._stage_reset_injections(restored)
                    # Queue the restore batch before this cancellable operation:
                    # once the old process is gone, a failed or cancelled start
                    # must still leave the next start able to rebuild it.
                    if cancellation is not None:
                        raise cancellation
                    await self._start_locked()
                finally:
                    if drained:
                        self._channel_lock.release()

    async def _iopub_for(self, msg_id: str, timeout: float | None = None):
        """Yield IOPub messages belonging to *msg_id* until kernel is idle.

        *timeout* bounds the gap between consecutive messages from this request.
        Traffic from another parent is consumed but cannot keep a silent cell
        alive.
        """
        timeout = self._iopub_timeout if timeout is None else timeout
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout
        while True:
            remaining = deadline - loop.time()
            msg = await asyncio.wait_for(
                self._kernel_client.get_iopub_msg(),
                timeout=max(0, remaining),
            )
            if msg["parent_header"].get("msg_id") != msg_id:
                continue
            deadline = loop.time() + timeout
            yield msg
            if (
                msg["header"]["msg_type"] == "status"
                and msg["content"]["execution_state"] == "idle"
            ):
                return

    async def _shell_reply_for(self, msg_id: str, timeout: float = _SHELL_REPLY_TIMEOUT) -> dict:
        """Wait for the shell reply matching *msg_id*."""
        while True:
            msg = await asyncio.wait_for(self._kernel_client.get_shell_msg(), timeout=timeout)
            if msg["parent_header"].get("msg_id") == msg_id:
                return msg

    async def _request_reply(self, msg_id: str, operation: str) -> dict:
        """Wait for one internal request without releasing a busy kernel.

        IOPub idle is the execution boundary; a shell reply is not expected
        while a long-running request is still busy. If the IOPub deadline
        expires, interrupt and reclaim this exact request before reporting a
        typed runtime failure. A request that completed during that grace
        period is accepted rather than replayed.
        """
        idle = False
        try:
            await self._drain_until_idle(msg_id)
            idle = True
            return await self._shell_reply_for(msg_id)
        except TimeoutError as error:
            if idle:
                raise RuntimeExecutionError(
                    f"Kernel shell channel timed out during {operation}"
                ) from error

            (reclaimed, reply), cancellation = await _finish_despite_cancellation(
                self._abandon(msg_id)
            )
            if cancellation is not None:
                raise cancellation from error
            if not reclaimed:
                self._recovering = True
                raise RuntimeExecutionError(
                    f"Kernel did not become idle after {operation}; "
                    "the runtime process must be restarted"
                ) from error
            if reply is not None and reply["content"].get("status") == "ok":
                return reply
            raise RuntimeExecutionError(f"Kernel channel timed out during {operation}") from error

    async def _abandon(self, msg_id: str) -> tuple[bool, dict | None]:
        """Interrupt *msg_id* and return whether it reached idle plus its reply.

        A timeout or cancellation stops us *waiting*; the kernel never hears
        about it. Releasing the channel then lets the next request queue behind
        a cell that is still running — which is why a timed-out execution used
        to be followed by a second, spurious timeout.

        SIGINT is given a short grace period. False tells :meth:`execute` to
        terminate the process after releasing the channel; an uncooperative
        cell is never left running beside the next request.
        """
        try:
            await self.interrupt()
            await asyncio.wait_for(
                self._drain_until_idle(msg_id, timeout=_INTERRUPT_GRACE_TIMEOUT),
                _INTERRUPT_GRACE_TIMEOUT,
            )
            reply = await self._reap_shell_reply(
                msg_id,
                timeout=_INTERRUPT_GRACE_TIMEOUT,
            )
            return True, reply
        except TimeoutError:
            logger.warning(
                "Kernel still busy %.2fs after interrupting %s; terminating it",
                _INTERRUPT_GRACE_TIMEOUT,
                msg_id,
            )
        except Exception:
            logger.debug("Failed to reclaim the channel after %s", msg_id, exc_info=True)
        return False, None

    async def _reap_shell_reply(
        self,
        msg_id: str,
        timeout: float = _SHELL_REPLY_TIMEOUT,
    ) -> dict | None:
        """Consume *msg_id*'s shell reply, whose IOPub traffic is already drained.

        Every ``execute_request`` answers on the shell channel as well as IOPub;
        draining only one side leaves the other queued without bound, and
        ``_shell_reply_for`` scans forward through whatever has accumulated.
        Best-effort: the result is already built, so a missing reply must not
        turn a successful execution into a failure.
        """
        try:
            return await self._shell_reply_for(msg_id, timeout=timeout)
        except TimeoutError:
            logger.debug("No shell reply for %s within %.2fs", msg_id, timeout)
            return None

    async def _drain_until_idle(
        self,
        msg_id: str,
        timeout: float | None = None,
    ) -> None:
        """Read *msg_id*'s remaining output until the kernel reports idle.

        Reuses ``_iopub_for``'s parent-id filter and idle terminator so the
        "which messages are mine" rule exists in one place.
        """
        async for _ in self._iopub_for(msg_id, timeout=timeout):
            pass

    async def _flush_injections(self) -> None:
        """Send every queued injection to the kernel.

        Once a batch starts, cancellation waits for it to finish and is then
        re-raised. Re-queuing a request already sent to the kernel would execute
        its deserializer twice on the next call, so the batch must have
        exactly-once completion from the caller's point of view.
        """
        _, cancellation = await _finish_despite_cancellation(
            self._flush_injection_batch(),
        )
        if cancellation is not None:
            raise cancellation

    async def _flush_injection_batch(self) -> None:
        """Send one snapshot of pending injections.

        The batch is swapped out *before* the first await, because ``inject_*``
        is synchronous and public: a registration made mid-flush would otherwise
        mutate the dict being iterated. Anything added during the flush lands in
        the next one, and a failure re-queues what was not sent.
        """
        with self._pending_lock:
            batch, self._pending_injections = self._pending_injections, {}
        pending = list(batch.items())
        for index, (name, code) in enumerate(pending):
            try:
                await self._execute_silent(code)
            except BaseException:
                # This request was dispatched, so replaying it could run a dill
                # deserializer's side effects twice. Preserve only work that was
                # never sent and fail closed until this name is replaced/reset.
                with self._pending_lock:
                    self._pending_injections = dict(pending[index + 1 :]) | self._pending_injections
                    if name not in self._pending_injections:
                        self._failed_injections.add(name)
                raise
            else:
                with self._pending_lock:
                    self._restore_injections[name] = code

    async def _execute_silent(self, code: str) -> None:
        """Run *code* on the kernel, discard output, raise on error."""
        msg_id = self._kernel_client.execute(code, silent=True)
        reply = await self._request_reply(msg_id, "a silent setup request")
        if reply["content"].get("status") == "error":
            raise RuntimeError(
                f"Kernel setup error: {reply['content'].get('ename', 'Unknown')}: "
                f"{reply['content'].get('evalue', '')}"
            )

    async def _collect_result(self, msg_id: str) -> ExecutionResult:
        """Read IOPub messages for *msg_id* and build an ``ExecutionResult``."""
        stdout_parts: list[str] = []
        error: BaseException | None = None

        async for msg in self._iopub_for(msg_id):
            match msg["header"]["msg_type"]:
                case "stream":
                    stdout_parts.append(msg["content"]["text"])

                case "execute_result" | "display_data" | "update_display_data":
                    # ``display()`` and rich reprs are output too; taking only
                    # ``stream`` messages told the model "No output" for a cell
                    # that had produced some.
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

        stdout = "".join(stdout_parts) or None
        return ExecutionResult(error=error, stdout=stdout)
