"""IPythonRuntime — in-process execution using an IPython shell."""

from __future__ import annotations

from ..security import SecurityChecker
from .base import BaseRuntime
from .executor import ErrorFeedbackMode, IPythonExecutor
from .primitives import Function, Type, Variable


class IPythonRuntime(BaseRuntime):
    """A Python runtime that executes code in-process via IPython InteractiveShell.

    The default runtime. Code runs in the same process, so injected objects are
    reached directly with no serialization — a DataFrame stays the same object
    across turns. The tradeoff is isolation: a segfault or OOM in generated code
    takes the host down with it, and :meth:`interrupt` cannot preempt CPU-bound
    code. Use :class:`~cave_agent.runtime.IPyKernelRuntime` when that matters.
    """

    def __init__(
        self,
        functions: list[Function] | None = None,
        variables: list[Variable] | None = None,
        types: list[Type] | None = None,
        security_checker: SecurityChecker | None = None,
        error_feedback_mode: ErrorFeedbackMode = ErrorFeedbackMode.PLAIN,
    ):
        super().__init__(
            IPythonExecutor(
                security_checker=security_checker,
                error_feedback_mode=error_feedback_mode,
            ),
            functions=functions,
            variables=variables,
            types=types,
        )
