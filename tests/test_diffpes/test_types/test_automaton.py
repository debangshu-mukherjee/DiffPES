"""Validate immutable carriers for executable experiment descriptions.

The tests cover factory validation, static metadata, deterministic random keys,
canonical return records, and relative artifact manifest records.
"""

from pathlib import Path

import jax.numpy as jnp
import pytest

from diffpes.types import (
    ArtifactRecord,
    AutomatonContext,
    AutomatonParam,
    AutomatonSpec,
    make_artifact_record,
    make_automaton_context,
    make_automaton_param,
    make_automaton_spec,
)


class TestAutomatonParam:
    """Validate :class:`~diffpes.types.AutomatonParam` static metadata.

    The case scope covers static parameter declaration fields.
    """

    def test_exposes_static_parameter_metadata(self) -> None:
        """Expose static metadata from a parameter factory result.

        The carrier must retain its identifier type required state and help.

        Notes
        -----
        Create one required string parameter with a help message.
        """
        param: AutomatonParam = make_automaton_param(
            "label",
            str,
            help="Label text.",
        )

        assert param.name == "label"
        assert param.python_type is str
        assert param.required is True
        assert param.help == "Label text."


class TestMakeAutomatonParam:
    """Validate :func:`~diffpes.types.make_automaton_param`.

    The case scope covers supported declarations and immutable list metadata.
    """

    def test_normalizes_list_metadata_and_records_default(self) -> None:
        """Normalize list metadata and retain a non-required default.

        A list parameter must store its default and choices as tuples.

        Notes
        -----
        Create one list declaration and compare its public static fields.
        """
        param: AutomatonParam = make_automaton_param(
            "values",
            list,
            default=[1, 2],
            choices=([1, 2], [3, 4]),
            example=[3, 4],
        )

        assert param.required is False
        assert param.default == (1, 2)
        assert param.choices == ((1, 2), (3, 4))
        assert param.example == (3, 4)
        assert param.has_example is True

    def test_rejects_invalid_identifiers_and_parameter_types(self) -> None:
        """Reject invalid identifiers and unsupported parameter types.

        The factory must reserve declarations for supported command-line types.

        Notes
        -----
        Request one keyword identifier and one unsupported complex type.
        """
        with pytest.raises(ValueError, match="non-keyword"):
            make_automaton_param("class", int)
        with pytest.raises(ValueError, match="supported parameter type"):
            make_automaton_param("value", complex)


class TestAutomatonSpec:
    """Validate :class:`~diffpes.types.AutomatonSpec` static metadata.

    The case scope covers experiment descriptions and return declarations.
    """

    def test_exposes_static_description_and_parameter_tuple(self) -> None:
        """Expose static description and parameter tuple values.

        The carrier must retain the ordered declarations from its factory.

        Notes
        -----
        Create one specification with one parameter and a short description.
        """
        param: AutomatonParam = make_automaton_param("count", int, default=1)
        spec: AutomatonSpec = make_automaton_spec(
            "example",
            (param,),
            description="Run one example.",
        )

        assert spec.name == "example"
        assert spec.params == (param,)
        assert spec.description == "Run one example."


class TestMakeAutomatonSpec:
    """Validate :func:`~diffpes.types.make_automaton_spec`.

    The case scope covers deterministic return declaration serialization.
    """

    def test_serializes_declared_returns_with_sorted_keys(self) -> None:
        """Serialize declared return fields with stable key ordering.

        A static description must expose deterministic JSON for the result map.

        Notes
        -----
        Create one empty parameter specification with unsorted return keys.
        """
        spec: AutomatonSpec = make_automaton_spec(
            "example",
            (),
            returns={"z": 1, "a": 2},
        )

        assert spec.returns_json == '{"a":2,"z":1}'

    def test_rejects_duplicate_parameter_names(self) -> None:
        """Reject duplicate parameter names in one specification.

        A parser cannot distinguish two declarations with one destination name.

        Notes
        -----
        Create two valid declarations that intentionally share one name.
        """
        first: AutomatonParam = make_automaton_param("count", int, default=1)
        second: AutomatonParam = make_automaton_param("count", int, default=2)

        with pytest.raises(ValueError, match="unique names"):
            make_automaton_spec("example", (first, second))


class TestAutomatonContext:
    """Validate :class:`~diffpes.types.AutomatonContext` runtime metadata.

    The case scope covers static metadata and one traced random key.
    """

    def test_exposes_output_metadata_and_a_random_key(
        self, tmp_path: Path
    ) -> None:
        """Expose output metadata and a deterministic random key.

        The context must retain its run information beside its JAX key.

        Notes
        -----
        Create one JSON-mode context under pytest's temporary directory.
        """
        context: AutomatonContext = make_automaton_context(
            tmp_path,
            7,
            "example",
            json_mode=True,
        )

        assert context.outdir == str(tmp_path)
        assert context.seed == 7
        assert context.experiment == "example"
        assert context.json_mode is True
        assert context.rng_key.shape == ()


class TestMakeAutomatonContext:
    """Validate :func:`~diffpes.types.make_automaton_context`.

    The case scope covers output-directory creation and deterministic JAX keys.
    """

    def test_creates_output_directory_and_deterministic_key(
        self, tmp_path: Path
    ) -> None:
        """Create the output root and deterministic random key.

        Equal seeds must create equal keys while the requested root exists.

        Notes
        -----
        Create two temporary contexts and compare their random keys.
        """
        output_path: Path = tmp_path / "run"
        first: AutomatonContext = make_automaton_context(
            output_path, 7, "example"
        )
        second: AutomatonContext = make_automaton_context(
            output_path, 7, "example"
        )

        assert output_path.is_dir()
        assert jnp.array_equal(first.rng_key, second.rng_key)

    def test_rejects_a_boolean_seed(self, tmp_path: Path) -> None:
        """Reject a Boolean value for a random seed.

        The context factory must distinguish integer seeds from Boolean values.

        Notes
        -----
        Request one context with a Boolean seed value.
        """
        with pytest.raises(ValueError, match="integer"):
            make_automaton_context(tmp_path, True, "example")


class TestArtifactRecord:
    """Validate :class:`~diffpes.types.ArtifactRecord` manifest metadata.

    The case scope covers immutable role media path and preview fields.
    """

    def test_exposes_each_manifest_field(self) -> None:
        """Expose each immutable artifact manifest field.

        The carrier must retain role media type path and preview text.

        Notes
        -----
        Build one record directly through its public factory.
        """
        record: ArtifactRecord = make_artifact_record(
            "array",
            "application/npz",
            "data.npz",
            preview_b64="YWJj",
        )

        assert record.role == "array"
        assert record.mime == "application/npz"
        assert record.path == "data.npz"
        assert record.preview_b64 == "YWJj"


class TestMakeArtifactRecord:
    """Validate :func:`~diffpes.types.make_artifact_record`.

    The case scope covers safe relative paths and preview storage.
    """

    def test_rejects_absolute_and_parent_manifest_paths(self) -> None:
        """Reject absolute and parent-traversal artifact paths.

        A result record must remain relative to its experiment output root.

        Notes
        -----
        Request one POSIX absolute path and one parent-traversal path.
        """
        with pytest.raises(ValueError, match="relative"):
            make_artifact_record("array", "application/npz", "/outside.npz")
        with pytest.raises(ValueError, match="relative"):
            make_artifact_record("array", "application/npz", "../outside.npz")
