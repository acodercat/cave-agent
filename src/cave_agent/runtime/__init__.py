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
    if name == "IPyKernelRuntime":
        from .ipykernel_runtime import IPyKernelRuntime

        return IPyKernelRuntime
    if name == "IPyKernelExecutor":
        from .ipykernel_executor import IPyKernelExecutor

        return IPyKernelExecutor
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
