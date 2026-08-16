"""Validate executable parameter description, parsing, and validation.

The tests cover JSON Schema generation, scalar coercion, choices, bounds, and
the precedence of defaults, documents, and command-line overrides.
"""

from pathlib import Path

import pytest
from beartype.typing import Any, Dict

from diffpes.harness import (
    AutomatonError,
    coerce_param_value,
    describe_param,
    merge_params,
    param_json_schema,
    params_json_schema,
    read_params_document,
    validate_param_value,
)
from diffpes.types import AutomatonParam, make_automaton_param


class TestDescribeParam:
    """Validate :func:`~diffpes.harness.describe_param` output.

    The case scope covers JSON-ready declared metadata.
    """

    def test_describes_default_bounds_choices_and_examples(self) -> None:
        """Build a description with declared metadata fields.

        The result must preserve parameter validation metadata in JSON data.

        Notes
        -----
        Create one bounded float parameter with default choices and an example.
        """
        param: AutomatonParam = make_automaton_param(
            "energy",
            float,
            default=1.5,
            help="Energy value.",
            unit="eV",
            bounds=(0.0, 2.0),
            choices=(0.5, 1.5),
            example=0.5,
        )
        description: Dict[str, Any] = describe_param(param)

        assert description["python_type"] == "float"
        assert description["default"] == 1.5
        assert description["bounds"] == [0.0, 2.0]
        assert description["choices"] == [0.5, 1.5]
        assert description["example"] == 0.5


class TestParamJsonSchema:
    """Validate :func:`~diffpes.harness.param_json_schema` output.

    The case scope covers bounds and primitive JSON type mapping.
    """

    def test_maps_a_bounded_integer_to_json_schema(self) -> None:
        """Map a bounded integer to JSON Schema fields.

        The fragment must retain its integer primitive and inclusive bounds.

        Notes
        -----
        Create one optional integer parameter with an example value.
        """
        param: AutomatonParam = make_automaton_param(
            "count",
            int,
            default=2,
            bounds=(1.0, 4.0),
            example=3,
        )
        schema: Dict[str, Any] = param_json_schema(param)

        assert schema["type"] == "integer"
        assert schema["minimum"] == 1.0
        assert schema["maximum"] == 4.0
        assert schema["examples"] == [3]


class TestParamsJsonSchema:
    """Validate :func:`~diffpes.harness.params_json_schema` output.

    The case scope covers object restrictions and required field declarations.
    """

    def test_rejects_unknown_properties_in_object_schema(self) -> None:
        """Reject unknown properties in the generated object schema.

        Executable parameter documents must have a closed declared key set.

        Notes
        -----
        Create one required parameter and inspect its object schema fields.
        """
        param: AutomatonParam = make_automaton_param("count", int)
        schema: Dict[str, Any] = params_json_schema((param,))

        assert schema["additionalProperties"] is False
        assert schema["required"] == ["count"]
        assert schema["properties"]["count"]["type"] == "integer"


class TestCoerceParamValue:
    """Validate :func:`~diffpes.harness.coerce_param_value` conversion.

    The case scope covers raw command-line scalar and JSON text values.
    """

    def test_converts_boolean_and_json_array_text(self) -> None:
        """Convert boolean and JSON-array command-line text values.

        The conversion must return native Python values with declared types.

        Notes
        -----
        Create boolean and list declarations before coercing raw text values.
        """
        flag: AutomatonParam = make_automaton_param(
            "flag", bool, default=False
        )
        values: AutomatonParam = make_automaton_param(
            "values", list, default=[]
        )
        actual_flag: Any = coerce_param_value(flag, "true")
        actual_values: Any = coerce_param_value(values, "[1, 2]")

        assert actual_flag is True
        assert actual_values == [1, 2]

    def test_rejects_malformed_structured_text(self) -> None:
        """Reject malformed JSON text for a list parameter.

        Structured command-line values must pass through JSON parsing first.

        Notes
        -----
        Create one list declaration and submit malformed JSON text.
        """
        values: AutomatonParam = make_automaton_param(
            "values", list, default=[]
        )

        with pytest.raises(
            AutomatonError,
            match=r"values has an invalid value",
        ):
            coerce_param_value(values, "[")


class TestValidateParamValue:
    """Validate :func:`~diffpes.harness.validate_param_value` restrictions.

    The case scope covers range and choice classification.
    """

    def test_rejects_values_outside_bounds_and_choices(self) -> None:
        """Reject values outside declared bounds and choices.

        The validation helper must reject both unsupported static restrictions.

        Notes
        -----
        Create one bounded integer and one string choice declaration.
        """
        bounded: AutomatonParam = make_automaton_param(
            "count",
            int,
            default=2,
            bounds=(1.0, 5.0),
        )
        selected: AutomatonParam = make_automaton_param(
            "model",
            str,
            default="chain",
            choices=("chain", "graphene"),
        )

        with pytest.raises(
            AutomatonError,
            match=r"count must be at least 1\.0",
        ):
            validate_param_value(bounded, 0)
        with pytest.raises(
            AutomatonError,
            match=r"model must match one declared choice",
        ):
            validate_param_value(selected, "other")


class TestReadParamsDocument:
    """Validate :func:`~diffpes.harness.read_params_document` input.

    The case scope covers inline text, path input, and object validation.
    """

    def test_reads_inline_and_file_json_objects(self, tmp_path: Path) -> None:
        """Read inline and file JSON parameter objects.

        The reader must retain object keys and native JSON scalar values.

        Notes
        -----
        Write one document file and compare it with an inline document result.
        """
        path: Path = tmp_path / "params.json"
        path.write_text('{"count": 3}', encoding="utf-8")
        inline: Dict[str, Any] = read_params_document('{"count": 3}')
        from_path: Dict[str, Any] = read_params_document(str(path))

        assert inline == {"count": 3}
        assert from_path == {"count": 3}

    def test_rejects_a_nonobject_document(self, tmp_path: Path) -> None:
        """Reject a JSON document whose root lacks object structure.

        Parameter input must always provide a JSON object at its root.

        Notes
        -----
        Submit a JSON array through the public reader.
        """
        path: Path = tmp_path / "array.json"
        path.write_text("[1, 2]", encoding="utf-8")

        with pytest.raises(
            AutomatonError,
            match=r"parameter document must contain a JSON object",
        ):
            read_params_document(str(path))


class TestMergeParams:
    """Validate :func:`~diffpes.harness.merge_params` precedence.

    The case scope covers defaults, documents, explicit values, and keys.
    """

    def test_applies_explicit_command_line_precedence(self) -> None:
        """Apply explicit command-line values after document values.

        The final merged value must override both the default and document.

        Notes
        -----
        Merge one declared integer through all three input precedence levels.
        """
        param: AutomatonParam = make_automaton_param("count", int, default=1)
        merged: Dict[str, Any] = merge_params(
            (param,),
            {"count": 2},
            {"count": 3},
        )

        assert merged == {"count": 3}

    def test_rejects_an_unknown_document_key(self) -> None:
        """Reject an unknown key from a parameter document.

        Merge behavior must not silently ignore a caller-provided parameter.

        Notes
        -----
        Merge one declared parameter with one unsupported document key.
        """
        param: AutomatonParam = make_automaton_param("count", int, default=1)

        with pytest.raises(
            AutomatonError,
            match=r"unknown parameter: other",
        ):
            merge_params((param,), {"other": 2}, {})
