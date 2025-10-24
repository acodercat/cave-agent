from typing import Callable, List, Dict, Any, Optional
from IPython.core.interactiveshell import InteractiveShell
from IPython.utils.capture import capture_output
import inspect
from .security_checker import SecurityChecker, SecurityError
from traitlets.config import Config
from enum import Enum

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
                if len(violations) > 0:
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
            
        except SecurityError:
            # Re-raise security errors as-is
            raise
        except Exception as e:
            return ExecutionResult(error=e)
    
    def get_from_namespace(self, name: str) -> Any:
        """Get a value from the execution namespace."""
        return self._shell.user_ns.get(name)
    
    def reset(self):
        """Reset the shell"""
        self._shell.reset()
        import gc
        gc.collect()
        
    @staticmethod
    def create_ipython_config(error_feedback_mode: ErrorFeedbackMode = ErrorFeedbackMode.PLAIN) -> Config:
        """"Create a clean IPython configuration optimized for code execution."""
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


class Variable:
    """Represents a variable in the Python runtime environment."""
    name: str
    description: Optional[str] = None
    value: Optional[Any] = None
    doc: Optional[str] = None
    type: str

    def __init__(self, name: str, value: Optional[Any] = None, description: Optional[str] = None, include_doc: bool = True):
        """Initialize the variable."""
        self.name = name
        self.value = value
        self.description = description
        self.type = type(self.value).__name__

        if include_doc and hasattr(self.value, "__doc__") and self.value.__doc__ and self.value.__doc__.strip():
            self.doc = self.value.__doc__.strip()
        
    def __str__(self):
        """Return a string representation of the variable."""
        parts = [f"- name: {self.name}"]
        parts.append(f"  type: {self.type}")
        if self.description:
            parts.append(f"  description: {self.description}")
        if self.doc:
            parts.append(f"  doc: {self.doc}")

        return "\n".join(parts)

class Function:
    """Represents a function in the Python runtime environment."""
    func: Callable
    description: Optional[str] = None
    doc: Optional[str] = None
    name: str
    signature: str
    include_doc: bool

    def __init__(self, func: Callable, description: Optional[str] = None, include_doc: bool = True):
        """Initialize the function."""
        self.func = func
        self.description = description
        self.name = func.__name__
        self.signature = f"{self.name}{inspect.signature(self.func)}"

        if include_doc and hasattr(self.func, "__doc__") and self.func.__doc__ and self.func.__doc__.strip():
            self.doc = self.func.__doc__
        
    
    def __str__(self):
        """Return a string representation of the function."""
        parts = [f"- function: {self.signature}"]
        if self.description:
            parts.append(f"  description: {self.description}")
        if self.doc:
            parts.append(f"  doc: {self.doc}")

        return "\n".join(parts)
    
class PythonRuntime:
    """
    A Python runtime that executes code snippets in an IPython environment.
    Provides a controlled execution environment with registered functions and objects.
    """
    def __init__(
        self,
        functions: List[Function] = [],
        variables: List[Variable] = [],
        security_checker: Optional[SecurityChecker] = None,
        error_feedback_mode: ErrorFeedbackMode = ErrorFeedbackMode.PLAIN,
    ):
        """
        Initialize runtime with executor and optional initial resources.
        
        Args:
            functions: List of functions to inject into runtime
            variables: List of variables to inject into runtime
            security_checker: Security checker instance to use for code execution
        """
            
        self._executor = PythonExecutor(security_checker=security_checker, error_feedback_mode=error_feedback_mode)
        self._functions: Dict[str, Function] = {}
        self._variables: Dict[str, Variable] = {}

        for function in functions:
            self.inject_function(function)
        
        for variable in variables:
            self.inject_variable(variable)

    def inject_function(self, function: Function):
        """Inject a function in both metadata and execution namespace."""
        if function.name in self._functions:
            raise ValueError(f"Function '{function.name}' already exists")
        self._functions[function.name] = function
        self._executor.inject_into_namespace(function.name, function.func)
    
    def inject_variable(self, variable: Variable):
        """Inject a variable in both metadata and execution namespace."""
        if variable.name in self._variables:
            raise ValueError(f"Variable '{variable.name}' already exists")
        self._variables[variable.name] = variable
        self._executor.inject_into_namespace(variable.name, variable.value)

    async def execute(self, code: str) -> ExecutionResult:
        """Execute code using the executor."""
        return await self._executor.execute(code)

    def get_variable_value(self, name: str) -> Any:
        """Get current value of a variable."""
        if name not in self._variables:
            raise KeyError(f"Variable '{name}' is not managed by this runtime. Available variables: {list(self._variables.keys())}")
        return self._executor.get_from_namespace(name)
    
    def describe_variables(self) -> str:
        """Generate formatted variable descriptions for system prompt."""
        if not self._variables:
            return "No variables available"
        
        descriptions = []
        for variable in self._variables.values():
            descriptions.append(str(variable))
        
        return "\n".join(descriptions)
    
    def describe_functions(self) -> str:
        """Generate formatted function descriptions for system prompt."""
        if not self._functions:
            return "No functions available"
        
        descriptions = []
        for function in self._functions.values():
            descriptions.append(str(function))
        
        return "\n".join(descriptions)
    
    def reset(self):
        """Reset the runtime."""
        self._executor.reset()
        self._functions.clear()
        self._variables.clear()