from typing import (
    Callable, List, Dict, Any, Optional, Union,
    get_args, get_origin, ForwardRef
)
from IPython.core.interactiveshell import InteractiveShell
from IPython.utils.capture import capture_output
import inspect
import gc
from .security_checker import SecurityChecker, SecurityError
from traitlets.config import Config
from enum import Enum
from dataclasses import is_dataclass, fields as dataclass_fields, MISSING
from pydantic import BaseModel

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


class Variable:
    """Represents a variable in the Python runtime environment."""

    name: str
    description: Optional[str]
    value: Optional[Any]
    type_name: str

    def __init__(
        self,
        name: str,
        value: Optional[Any] = None,
        description: Optional[str] = None,
    ):
        """
        Initialize the variable.

        Args:
            name: Variable name
            value: The value to store
            description: Optional description of the variable
        """
        self.name = name
        self.value = value
        self.description = description
        self.type_name = type(self.value).__name__ if self.value is not None else "NoneType"

    def __str__(self) -> str:
        """Return a string representation of the variable (without inline schema)."""
        parts = [f"- name: {self.name}"]
        parts.append(f"  type: {self.type_name}")
        if self.description:
            parts.append(f"  description: {self.description}")

        return "\n".join(parts)

class Function:
    """
    Represents a function in the Python runtime environment.
    """

    def __init__(
        self,
        func: Callable,
        description: Optional[str] = None,
        include_doc: bool = True,
    ) -> None:
        """
        Initialize the function.

        Args:
            func: Callable function to wrap
            description: Optional description of the function
            include_doc: Whether to include the function's docstring
        """
        self.func = func
        self.description = description
        self.name = func.__name__
        self.signature = f"{self.name}{inspect.signature(func)}"
        self.doc: Optional[str] = None

        if include_doc and hasattr(func, "__doc__") and func.__doc__:
            self.doc = func.__doc__.strip()
    
    def __str__(self) -> str:
        """Return a string representation of the function."""
        parts = [f"- function: {self.signature}"]
        
        if self.description:
            parts.append(f"  description: {self.description}")

        if self.doc:
            parts.append(self._format_docstring())

        return "\n".join(parts)

    def _format_docstring(self) -> str:
        """
        Format the docstring with proper indentation.

        Returns:
            Formatted docstring
        """
        lines = ["  doc:"]
        for line in self.doc.split('\n'):
            lines.append(f"    {line}")
        return "\n".join(lines)

class Type:
    """Represents a type/class in the Python runtime namespace.

    Use this to inject classes that the LLM can use for:
    - isinstance() checks
    - Creating new instances
    """

    name: str
    value: type
    description: Optional[str]
    include_schema: bool
    include_doc: bool

    def __init__(
        self,
        value: type,
        description: Optional[str] = None,
        include_schema: bool = True,
        include_doc: bool = True,
    ):
        """
        Initialize the type.

        Args:
            value: The class/type to inject
            description: Optional description of the type
            include_schema: Whether to include methods/fields in type schemas section
            include_doc: Whether to include docstring in type schemas section
        """
        if not isinstance(value, type):
            raise ValueError(f"Type value must be a class, got {type(value).__name__}")
        self.name = value.__name__
        self.value = value
        self.description = description
        self.include_schema = include_schema
        self.include_doc = include_doc

    def __str__(self) -> str:
        """Return a string representation of the type with schema if enabled."""
        if not self.include_schema and not self.include_doc:
            return ""

        cls = self.value
        schema = None

        if self.include_schema:
            # Check if it's a Pydantic model
            if issubclass(cls, BaseModel):
                schema = TypeSchemaExtractor._format_pydantic_schema(cls, self.include_doc)

            # Check if it's a dataclass
            elif is_dataclass(cls):
                schema = TypeSchemaExtractor._format_dataclass_schema(cls, self.include_doc)

            # Check if it's an Enum
            elif issubclass(cls, Enum):
                schema = TypeSchemaExtractor._format_enum_schema(cls)

            # Regular class - extract method signatures
            else:
                schema = TypeSchemaExtractor._format_class_methods(cls, self.include_doc)

        elif self.include_doc:
            # Doc only
            if cls.__doc__ and cls.__doc__.strip():
                schema = f"{self.name}:\n  doc: {cls.__doc__.strip()}"

        if not schema:
            return ""

        # Insert description after the type name line if provided
        if self.description:
            lines = schema.split('\n')
            lines.insert(1, f"  description: {self.description}")
            return '\n'.join(lines)

        return schema


class TypeSchemaExtractor:
    """
    Extracts and formats type schema information from Python types.

    Supports Pydantic models, dataclasses, enums, and regular classes.
    """

    @classmethod
    def _format_class_methods(cls, class_type: type, include_doc: bool = True) -> Optional[str]:
        """
        Extract public method signatures from a class.

        Args:
            class_type: The class to analyze
            include_doc: Whether to include docstring

        Returns:
            Formatted method signatures or None if no methods found
        """
        methods = []

        for name, method in inspect.getmembers(class_type, predicate=inspect.isfunction):
            # Skip private and magic methods
            if name.startswith('_'):
                continue

            try:
                sig = inspect.signature(method)
                # Format parameters (skip 'self')
                params = []
                for param_name, param in sig.parameters.items():
                    if param_name == 'self':
                        continue
                    param_str = param_name
                    if param.annotation != inspect.Parameter.empty:
                        param_str += f": {cls._format_type_annotation(param.annotation)}"
                    if param.default != inspect.Parameter.empty:
                        param_str += f" = {param.default!r}"
                    params.append(param_str)

                # Format return type
                return_str = ""
                if sig.return_annotation != inspect.Signature.empty:
                    return_str = f" -> {cls._format_type_annotation(sig.return_annotation)}"

                method_sig = f"{name}({', '.join(params)}){return_str}"
                method_entry = f"    - {method_sig}"

                if include_doc and method.__doc__ and method.__doc__.strip():
                    doc = method.__doc__.strip()
                    method_entry += f"\n        {doc}"

                methods.append(method_entry)
            except (ValueError, TypeError):
                # Skip methods that can't be inspected
                continue

        if not methods:
            return None

        lines = [f"{class_type.__name__}:"]

        # Add docstring if available and requested
        if include_doc and class_type.__doc__ and class_type.__doc__.strip():
            lines.append(f"  doc: {class_type.__doc__.strip()}")

        lines.append("  methods:")
        lines.extend(methods)
        return "\n".join(lines)

    @classmethod
    def _format_pydantic_schema(
        cls,
        model: type,
        include_doc: bool = True
    ) -> str:
        """
        Format a Pydantic model schema.

        Args:
            model: Pydantic model class
            include_doc: Whether to include docstring

        Returns:
            Formatted schema string
        """
        lines = [f"{model.__name__}:"]

        # Add docstring if available and requested
        if include_doc and model.__doc__ and model.__doc__.strip():
            lines.append(f"  doc: {model.__doc__.strip()}")

        lines.append("  fields:")

        for field_name, field_info in model.model_fields.items():
            field_type = field_info.annotation
            type_str = cls._format_type_annotation(field_type)

            field_line = f"    - {field_name}: {type_str}"

            if field_info.description:
                field_line += f"  # {field_info.description}"

            lines.append(field_line)

        return "\n".join(lines)

    @classmethod
    def _format_dataclass_schema(
        cls,
        dataclass_type: type,
        include_doc: bool = True
    ) -> str:
        """
        Format a dataclass schema.

        Args:
            dataclass_type: Dataclass type
            include_doc: Whether to include docstring

        Returns:
            Formatted schema string
        """
        lines = [f"{dataclass_type.__name__}:"]

        # Add docstring if available and requested
        if include_doc and dataclass_type.__doc__ and dataclass_type.__doc__.strip():
            lines.append(f"  doc: {dataclass_type.__doc__.strip()}")

        lines.append("  fields:")

        for field in dataclass_fields(dataclass_type):
            type_str = cls._format_type_annotation(field.type)
            field_line = f"    - {field.name}: {type_str}"

            if field.default is not MISSING:
                field_line += f" = {field.default!r}"
            elif field.default_factory is not MISSING:  # type: ignore
                field_line += " = <factory>"

            lines.append(field_line)

        return "\n".join(lines)

    @classmethod
    def _format_enum_schema(cls, enum_type: type) -> str:
        """
        Format an Enum schema.

        Args:
            enum_type: Enum class

        Returns:
            Formatted schema string
        """
        lines = [f"{enum_type.__name__} (Enum):"]
        for member in enum_type:
            lines.append(f"  - {member.name} = {member.value!r}")
        return "\n".join(lines)

    @classmethod
    def _format_type_annotation(cls, type_hint: Any) -> str:
        """
        Format a type annotation as a readable string.

        Args:
            type_hint: Type annotation to format

        Returns:
            Human-readable type string
        """
        if type_hint is type(None) or type_hint is None:
            return "None"

        # Handle ForwardRef
        if isinstance(type_hint, ForwardRef):
            return type_hint.__forward_arg__

        # Handle string annotations
        if isinstance(type_hint, str):
            return type_hint

        origin = get_origin(type_hint)

        # Handle generic types
        if origin is not None:
            args = get_args(type_hint)

            # Get a readable origin name
            if origin is Union:
                # Special handling for Optional (Union[X, None])
                non_none_args = [a for a in args if a is not type(None)]
                if len(non_none_args) == 1 and len(args) == 2:
                    return f"Optional[{cls._format_type_annotation(non_none_args[0])}]"
                arg_strs = [cls._format_type_annotation(arg) for arg in args]
                return f"Union[{', '.join(arg_strs)}]"

            origin_name = getattr(origin, '__name__', str(origin).replace('typing.', ''))

            if not args:
                return origin_name

            arg_strs = [cls._format_type_annotation(arg) for arg in args]
            return f"{origin_name}[{', '.join(arg_strs)}]"

        # Handle Callable without arguments
        if type_hint is Callable:
            return "Callable"

        # Return the type name
        if hasattr(type_hint, '__name__'):
            return type_hint.__name__

        return str(type_hint)
    
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

    def get_variable(self, name: str) -> Any:
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