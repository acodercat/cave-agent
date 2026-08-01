"""Tests for TypeSchemaExtractor class."""

from typing import Any, Callable, Dict, List, Optional, Union

from cave_agent.runtime import TypeSchemaExtractor


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


class TestCallableHandling:
    """Test handling of Callable types."""

    def test_callable_type_formatting(self):
        """Callable should be formatted as 'Callable'."""
        type_str = TypeSchemaExtractor._format_type_annotation(Callable)
        assert type_str == "Callable"


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


class TestPep604UnionsRenderAsPython:
    """`int | None` and `Optional[int]` mean the same thing but arrive as
    different origins — `types.UnionType` and `typing.Union`. Recognizing only
    the second showed the model `UnionType[int, None]`: not an expression it
    can write back, and inconsistent with the `Optional[...]` beside it."""

    def test_an_optional_pep604_union(self):
        assert TypeSchemaExtractor._format_type_annotation(int | None) == "Optional[int]"

    def test_a_pep604_union_over_a_generic(self):
        assert (
            TypeSchemaExtractor._format_type_annotation(dict[str, int] | None)
            == "Optional[dict[str, int]]"
        )

    def test_a_multi_arm_pep604_union(self):
        assert (
            TypeSchemaExtractor._format_type_annotation(int | str | bool) == "Union[int, str, bool]"
        )

    def test_both_spellings_agree(self):
        from typing import Optional

        assert TypeSchemaExtractor._format_type_annotation(
            int | None
        ) == TypeSchemaExtractor._format_type_annotation(Optional[int])

    def test_a_dataclass_field_renders_runnably(self):
        from dataclasses import dataclass

        @dataclass
        class Config:
            retries: int | None

        schema = TypeSchemaExtractor._format_dataclass_schema(Config)

        assert "UnionType" not in schema
        assert "Optional[int]" in schema
