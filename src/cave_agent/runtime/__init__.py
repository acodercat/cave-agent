from .runtime import Runtime
from .ipython_runtime import IPythonRuntime
from .executor import ExecutionResult, ErrorFeedbackMode
from .primitives import Variable, Function, Type, TypeSchemaExtractor


def __getattr__(name: str):
    if name == "IPyKernelRuntime":
        from .ipykernel_runtime import IPyKernelRuntime
        return IPyKernelRuntime
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "Runtime",
    "IPythonRuntime",
    "IPyKernelRuntime",
    "ExecutionResult",
    "ErrorFeedbackMode",
    "Variable",
    "Function",
    "Type",
    "TypeSchemaExtractor",
]
