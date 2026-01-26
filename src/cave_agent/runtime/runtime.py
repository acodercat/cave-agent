from typing import (
    Callable, List, Dict, Any, Optional,
    get_args, get_origin, ForwardRef
)
import inspect

from .executor import PythonExecutor, ExecutionResult, ErrorFeedbackMode
from .primitives import Variable, Function, Type
from ..security import SecurityChecker


class PythonRuntime:
    """
    A Python runtime that executes code snippets in an IPython environment.
    Provides a controlled execution environment with registered functions, variables, and types.
    """

    _executor: PythonExecutor
    _functions: Dict[str, Function]
    _variables: Dict[str, Variable]
    _types: Dict[str, Type]

    def __init__(
        self,
        functions: Optional[List[Function]] = None,
        variables: Optional[List[Variable]] = None,
        types: Optional[List[Type]] = None,
        security_checker: Optional[SecurityChecker] = None,
        error_feedback_mode: ErrorFeedbackMode = ErrorFeedbackMode.PLAIN,
    ):
        """
        Initialize runtime with executor and optional initial resources.

        Args:
            functions: List of functions to inject into runtime
            variables: List of variables to inject into runtime
            types: List of types/classes to inject into runtime
            security_checker: Security checker instance to use for code execution
            error_feedback_mode: Error feedback mode for execution errors
        """
        self._executor = PythonExecutor(security_checker=security_checker, error_feedback_mode=error_feedback_mode)
        self._functions = {}
        self._variables = {}
        self._types = {}

        # Inject explicit types first so they take precedence over auto-injection
        for type_obj in (types or []):
            self.inject_type(type_obj)

        for function in (functions or []):
            self.inject_function(function)

        for variable in (variables or []):
            self.inject_variable(variable)

    # Built-in types that should not be auto-injected
    _BUILTIN_TYPES = frozenset({
        str, int, float, bool, bytes, bytearray,
        list, dict, tuple, set, frozenset,
        type(None), object, type,
    })

    def inject_function(self, function: Function):
        """Inject a function in both metadata and execution namespace.

        Also auto-injects custom types found in the function signature (with schema hidden).
        Use explicit Type injection to show schemas in the prompt.
        """
        if function.name in self._functions:
            raise ValueError(f"Function '{function.name}' already exists")
        self._functions[function.name] = function
        self._executor.inject_into_namespace(function.name, function.func)

        # Auto-inject types from function signature (schema hidden by default)
        self._auto_inject_types_from_signature(function.func)

    def inject_variable(self, variable: Variable):
        """Inject a variable in both metadata and execution namespace.

        Also auto-injects the type of the variable's value (with schema hidden).
        Use explicit Type injection to show schemas in the prompt.
        """
        if variable.name in self._variables:
            raise ValueError(f"Variable '{variable.name}' already exists")
        self._variables[variable.name] = variable
        self._executor.inject_into_namespace(variable.name, variable.value)

        # Auto-inject the type (schema hidden by default)
        if variable.value is not None:
            value_type = type(variable.value)
            self._try_auto_inject_type(value_type, include_schema=False, include_doc=False)

    def update_variable(self, name: str, value: Any):
        """
        Update the value of an existing variable.

        Args:
            name: Name of the variable to update
            value: New value for the variable

        Raises:
            KeyError: If the variable does not exist
            TypeError: If the new value has a different type than the original
        """
        if name not in self._variables:
            raise KeyError(f"Variable '{name}' does not exist. Available variables: {list(self._variables.keys())}")

        # Check type consistency
        original_value = self._variables[name].value
        if original_value is not None and value is not None:
            original_type = type(original_value)
            new_type = type(value)
            if original_type != new_type:
                raise TypeError(
                    f"Cannot update variable '{name}': type mismatch. "
                    f"Expected {original_type.__name__}, got {new_type.__name__}"
                )

        # Update the Variable object's value
        self._variables[name].value = value

        # Update the executor namespace
        self._executor.inject_into_namespace(name, value)

    def inject_type(self, type_obj: Type):
        """Inject a type/class in both metadata and execution namespace."""
        if type_obj.name in self._types:
            raise ValueError(f"Type '{type_obj.name}' already exists")
        self._types[type_obj.name] = type_obj
        self._executor.inject_into_namespace(type_obj.name, type_obj.value)

    def _try_auto_inject_type(
        self,
        cls: type,
        include_schema: bool = False,
        include_doc: bool = False,
    ) -> bool:
        """Try to auto-inject a type if it's injectable and not already present.

        Args:
            cls: The class to inject
            include_schema: Whether to include type schema in describe_types() (default False)
            include_doc: Whether to include docstring in describe_types() (default False)

        Returns True if the type was injected, False otherwise.
        """
        if not self._is_injectable_type(cls):
            return False

        type_name = cls.__name__
        if type_name in self._types:
            return False  # Already injected

        type_obj = Type(cls, include_schema=include_schema, include_doc=include_doc)
        self._types[type_name] = type_obj
        self._executor.inject_into_namespace(type_name, cls)
        return True

    def _is_injectable_type(self, cls: type) -> bool:
        """Check if a type should be auto-injected."""
        if not isinstance(cls, type):
            return False
        if cls in self._BUILTIN_TYPES:
            return False
        # Skip types without proper names (lambdas, etc.)
        if not hasattr(cls, '__name__') or cls.__name__.startswith('<'):
            return False
        return True

    def _auto_inject_types_from_signature(self, func: Callable):
        """Extract and auto-inject types from a function signature (schema hidden)."""
        try:
            sig = inspect.signature(func)
        except (ValueError, TypeError):
            return

        # Process parameter types
        for param in sig.parameters.values():
            if param.annotation != inspect.Parameter.empty:
                self._process_type_for_injection(param.annotation)

        # Process return type
        if sig.return_annotation != inspect.Signature.empty:
            self._process_type_for_injection(sig.return_annotation)

    def _process_type_for_injection(self, type_hint: Any):
        """Process a type hint and inject any custom types found (schema hidden)."""
        if type_hint is None or type_hint is type(None):
            return

        # Handle ForwardRef and string annotations
        if isinstance(type_hint, (str, ForwardRef)):
            return

        # Handle generic types (List[X], Dict[K,V], Optional[X], Union[X,Y], etc.)
        origin = get_origin(type_hint)
        if origin is not None:
            # Process type arguments recursively
            for arg in get_args(type_hint):
                if arg is not type(None):
                    self._process_type_for_injection(arg)
            return

        # Handle actual types (schema hidden by default)
        if isinstance(type_hint, type):
            self._try_auto_inject_type(type_hint, include_schema=False, include_doc=False)

    async def execute(self, code: str) -> ExecutionResult:
        """Execute code using the executor."""
        return await self._executor.execute(code)

    def retrieve(self, name: str) -> Any:
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

    def describe_types(self) -> str:
        """
        Generate type schemas from explicitly injected Types.

        Only Types with include_schema=True or include_doc=True are shown.
        Auto-injected types from Variables/Functions have schema hidden by default.
        """
        if not self._types:
            return "No types available"

        schemas = []
        for type_obj in self._types.values():
            schema = str(type_obj)
            if schema:
                schemas.append(schema)

        return "\n".join(schemas) if schemas else "No types available"

    def reset(self):
        """Reset the runtime."""
        self._executor.reset()
        self._functions.clear()
        self._variables.clear()
        self._types.clear()
