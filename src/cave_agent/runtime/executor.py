from typing import Any, Optional
from IPython.core.interactiveshell import InteractiveShell
from IPython.utils.capture import capture_output
from traitlets.config import Config
from enum import Enum
import gc

from ..security import SecurityChecker, SecurityError


class ExecutionResult:
    """
    Represents the result of code execution.
    """
    error: Optional[BaseException] = None
    stdout: Optional[str] = None

    def __init__(self, error: Optional[BaseException] = None, stdout: Optional[str] = None):
        self.error = error
        self.stdout = stdout

    @property
    def success(self):
        return self.error is None


class ErrorFeedbackMode(Enum):
    """Error feedback modes for LLM agent observation."""
    PLAIN = "Plain"      # Full traceback for agent debugging
    MINIMAL = "Minimal"     # Brief error info for agent efficiency


class PythonExecutor:
    """
    Handles Python code execution using IPython.
    """

    def __init__(self, security_checker: Optional[SecurityChecker] = None, error_feedback_mode: ErrorFeedbackMode = ErrorFeedbackMode.PLAIN):
        """Initialize IPython shell for code execution."""
        ipython_config = self.create_ipython_config(error_feedback_mode=error_feedback_mode)
        self._shell = InteractiveShell(config=ipython_config)
        self._security_checker = security_checker

    def inject_into_namespace(self, name: str, value: Any):
        """Inject a value into the execution namespace."""
        self._shell.user_ns[name] = value

    async def execute(self, code: str) -> ExecutionResult:
        """Execute code snippet with optional security checks.

        Args:
            code: Python code to execute

        Returns:
            ExecutionResult with success status and output or error
        """
        try:
            # Perform security check
            if self._security_checker:
                violations = self._security_checker.check_code(code)
                if violations:
                    violation_details = [str(v) for v in violations]
                    error_message = (
                        f"Code execution blocked: {len(violations)} violations found:\n"
                        + "\n".join(f"  - {detail}" for detail in violation_details)
                    )
                    security_error = SecurityError(error_message)
                    return ExecutionResult(error=security_error, stdout=None)

            # Execute the code
            with capture_output() as output:
                transformed_code = self._shell.transform_cell(code)
                result = await self._shell.run_cell_async(
                    transformed_code,
                    transformed_cell=transformed_code
                )

            # Handle execution errors
            if result.error_before_exec:
                return ExecutionResult(
                    error=result.error_before_exec,
                    stdout=output.stdout
                )
            if result.error_in_exec:
                return ExecutionResult(
                    error=result.error_in_exec,
                    stdout=output.stdout
                )

            return ExecutionResult(stdout=output.stdout)

        except Exception as e:
            return ExecutionResult(error=e)

    def get_from_namespace(self, name: str) -> Any:
        """Get a value from the execution namespace."""
        return self._shell.user_ns.get(name)

    def reset(self):
        """Reset the shell"""
        self._shell.reset()
        gc.collect()

    @staticmethod
    def create_ipython_config(error_feedback_mode: ErrorFeedbackMode = ErrorFeedbackMode.PLAIN) -> Config:
        """Create a clean IPython configuration optimized for code execution."""
        config = Config()
        config.InteractiveShell.cache_size = 0
        config.InteractiveShell.history_length = 0
        config.InteractiveShell.automagic = False
        config.InteractiveShell.separate_in = ''
        config.InteractiveShell.separate_out = ''
        config.InteractiveShell.separate_out2 = ''
        config.InteractiveShell.autocall = 0
        config.InteractiveShell.colors = 'nocolor'
        config.InteractiveShell.xmode = error_feedback_mode.value
        config.InteractiveShell.quiet = True
        config.InteractiveShell.autoindent = False

        return config
