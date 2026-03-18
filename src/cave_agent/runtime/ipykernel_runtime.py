"""IPyKernelRuntime — process-isolated execution via Jupyter kernel."""

from __future__ import annotations

from ..security import SecurityChecker
from .executor import ErrorFeedbackMode
from .ipykernel_executor import IPyKernelExecutor
from .primitives import Variable, Function, Type
from .runtime import Runtime


class IPyKernelRuntime(Runtime):
    """A Python runtime backed by a separate IPython kernel process.

    Drop-in replacement for ``IPythonRuntime`` with process isolation —
    a crash (segfault, OOM) does not bring down the host.

    Usage::

        async with IPyKernelRuntime(functions=[...]) as rt:
            await rt.execute("print('hello')")

    Injected objects are serialized via dill, which supports local functions,
    closures, lambdas, and most Python objects.
    """

    def __init__(
        self,
        functions: list[Function] | None = None,
        variables: list[Variable] | None = None,
        types: list[Type] | None = None,
        security_checker: SecurityChecker | None = None,
        error_feedback_mode: ErrorFeedbackMode = ErrorFeedbackMode.PLAIN,
    ):
        self._executor = IPyKernelExecutor(
            security_checker=security_checker,
            error_feedback_mode=error_feedback_mode,
        )
        super().__init__(functions=functions, variables=variables, types=types)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> IPyKernelRuntime:
        """Start the kernel subprocess."""
        await self._executor.start()
        return self

    async def stop(self) -> None:
        """Shut down the kernel subprocess."""
        await self._executor.stop()

    async def __aenter__(self) -> IPyKernelRuntime:
        return await self.start()

    async def __aexit__(self, *exc: object) -> None:
        await self.stop()

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    async def reset(self) -> None:  # type: ignore[override]
        """Restart the kernel and re-inject all registered objects."""
        await self._executor.reset()

        for type_obj in self._types.values():
            self._executor.inject_into_namespace(type_obj.name, type_obj.value)
        for func in self._functions.values():
            self._executor.inject_into_namespace(func.name, func.func)
        for var in self._variables.values():
            self._executor.inject_into_namespace(var.name, var.value)
