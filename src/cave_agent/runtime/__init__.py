"""Runtime layer — persistent Python namespaces that execute generated code.

:class:`Runtime` is the protocol the agent depends on; everything else here is
an implementation of it. ``IPyKernelRuntime`` is imported lazily so that
``import cave_agent`` does not require the optional ``dill`` / ``ipykernel``
dependencies.
"""

from typing import TYPE_CHECKING

from .base import BaseRuntime
from .executor import (
    ErrorFeedbackMode,
    ExecutionResult,
    IPythonExecutor,
    RuntimeExecutionError,
    RuntimeStateLostError,
)
from .ipython_runtime import IPythonRuntime
from .primitives import Function, Type, TypeSchemaExtractor, Variable
from .protocol import PreemptibleRuntime, Runtime

if TYPE_CHECKING:
    from .ipykernel_executor import IPyKernelExecutor
    from .ipykernel_runtime import IPyKernelRuntime


def __getattr__(name: str):
    if name in ("IPyKernelRuntime", "IPyKernelExecutor"):
        # The kernel backend's transport (dill, jupyter_client) is the
        # ``ipykernel`` extra. Left unguarded, a base install importing it got
        # a bare ``No module named 'dill'`` — naming a package the user never
        # asked for, with no path out — while the models layer already answers
        # the same situation with the install command.
        try:
            from . import ipykernel_runtime
        except ModuleNotFoundError as error:
            raise ModuleNotFoundError(
                f"{name} needs the 'ipykernel' extra: `pip install 'cave_agent[ipykernel]'`"
            ) from error
        if name == "IPyKernelRuntime":
            return ipykernel_runtime.IPyKernelRuntime
        return ipykernel_runtime.IPyKernelExecutor
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "Runtime",
    "PreemptibleRuntime",
    "BaseRuntime",
    "IPythonRuntime",
    "IPyKernelRuntime",
    "Function",
    "Variable",
    "Type",
    "TypeSchemaExtractor",
    "ExecutionResult",
    "ErrorFeedbackMode",
    "RuntimeExecutionError",
    "RuntimeStateLostError",
    "IPythonExecutor",
    "IPyKernelExecutor",
]
