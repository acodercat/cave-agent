"""Tests for TypeSchemaExtractor class."""

import pytest
from typing import List, Dict, Optional, Union, Callable, Any

from cave_agent.runtime import TypeSchemaExtractor


# =============================================================================
# Tests - Type Annotation Formatting
# =============================================================================

class TestGenericTypeHandling:
    """Test handling of generic types."""

    def test_optional_type_formatting(self):
        """Optional types should be formatted correctly."""
        type_str = TypeSchemaExtractor._format_type_annotation(Optional[str])
        assert type_str == "Optional[str]"

    def test_union_type_formatting(self):
        """Union types should be formatted correctly."""
        type_str = TypeSchemaExtractor._format_type_annotation(Union[str, int, float])
        assert "Union[" in type_str
        assert "str" in type_str
        assert "int" in type_str
        assert "float" in type_str

    def test_list_type_formatting(self):
        """List types should be formatted correctly."""
        type_str = TypeSchemaExtractor._format_type_annotation(List[str])
        assert type_str == "list[str]"

    def test_dict_type_formatting(self):
        """Dict types should be formatted correctly."""
        type_str = TypeSchemaExtractor._format_type_annotation(Dict[str, int])
        assert type_str == "dict[str, int]"

    def test_nested_generic_formatting(self):
        """Nested generics should be formatted correctly."""
        type_str = TypeSchemaExtractor._format_type_annotation(List[Dict[str, List[int]]])
        assert "list[dict[str, list[int]]]" == type_str


# =============================================================================
# Tests - Callable Handling
# =============================================================================

class TestCallableHandling:
    """Test handling of Callable types."""

    def test_callable_type_formatting(self):
        """Callable should be formatted as 'Callable'."""
        type_str = TypeSchemaExtractor._format_type_annotation(Callable)
        assert type_str == "Callable"


# =============================================================================
# Tests - Edge Cases
# =============================================================================

class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_none_type_formatting(self):
        """None type should be formatted as 'None'."""
        type_str = TypeSchemaExtractor._format_type_annotation(None)
        assert type_str == "None"

        type_str = TypeSchemaExtractor._format_type_annotation(type(None))
        assert type_str == "None"

    def test_string_annotation_handling(self):
        """String annotations should be handled gracefully."""
        type_str = TypeSchemaExtractor._format_type_annotation("SomeForwardRef")
        assert type_str == "SomeForwardRef"

    def test_any_type_formatting(self):
        """Any type should be formatted correctly."""
        type_str = TypeSchemaExtractor._format_type_annotation(Any)
        # Any doesn't have __name__, falls back to str()
        assert "Any" in type_str
