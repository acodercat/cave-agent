from typing import (
    Callable, Any, Optional, Union,
    get_args, get_origin, ForwardRef
)
import inspect
from enum import Enum
from dataclasses import is_dataclass, fields as dataclass_fields, MISSING
from pydantic import BaseModel


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
