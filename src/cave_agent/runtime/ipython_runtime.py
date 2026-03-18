"""IPythonRuntime — in-process execution using IPython shell."""

from __future__ import annotations

from ..security import SecurityChecker
from .executor import IPythonExecutor, ErrorFeedbackMode
from .primitives import Variable, Function, Type
from .runtime import Runtime


class IPythonRuntime(Runtime):
    """A Python runtime that executes code in-process via IPython InteractiveShell.

    This is the default runtime for CaveAgent. Code runs in the same process,
    giving direct access to injected Python objects without serialization.
    """

    def __init__(
        self,
        functions: list[Function] | None = None,
        variables: list[Variable] | None = None,
        types: list[Type] | None = None,
        security_checker: SecurityChecker | None = None,
        error_feedback_mode: ErrorFeedbackMode = ErrorFeedbackMode.PLAIN,
    ):
        self._executor = IPythonExecutor(
            security_checker=security_checker,
            error_feedback_mode=error_feedback_mode,
        )
        super().__init__(functions=functions, variables=variables, types=types)
