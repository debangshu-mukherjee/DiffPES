"""Validate constants for executable experiment handling.

The tests cover schemas, vocabulary, process codes, runtime settings, media
types, inherited flags, and the dependency-pin pattern.
"""

import re

from diffpes.constants import (
    AUTOMATON_ARTIFACT_MIMES,
    AUTOMATON_CACHE_ENV_VAR,
    AUTOMATON_DEFAULT_OUTDIR,
    AUTOMATON_DESCRIBE_SCHEMA_VERSION,
    AUTOMATON_ERROR_KINDS,
    AUTOMATON_EXIT_CODES,
    AUTOMATON_INHERITED_FLAGS,
    AUTOMATON_JSON_TYPES,
    AUTOMATON_PARAM_TYPES,
    AUTOMATON_PIN_PATTERN,
    AUTOMATON_PREVIEW_MAX_BYTES,
    AUTOMATON_RESULT_SCHEMA_VERSION,
    AUTOMATON_RUNTIME_CHECK_ENV_VAR,
    AUTOMATON_SMOKE_WALL_SECONDS,
    AUTOMATON_STATUS_VALUES,
)


class TestAutomatonConstants:
    """Validate constants exported by :mod:`diffpes.constants.automaton`.

    The cases cover every stable constant in the executable process boundary.
    """

    def test_preserves_schema_and_result_vocabulary(self) -> None:
        """Preserve schema identifiers result statuses and error categories.

        Every public schema and process vocabulary value must retain fixed
        text.

        Notes
        -----
        Compare exported schema strings tuples and exit mappings with fixed
        values.
        """
        assert (
            AUTOMATON_DESCRIBE_SCHEMA_VERSION
            == "diffpes.automaton.describe.v1"
        )
        assert AUTOMATON_RESULT_SCHEMA_VERSION == "diffpes.automaton.result.v1"
        assert AUTOMATON_STATUS_VALUES == ("ok", "error", "timeout")
        assert AUTOMATON_ERROR_KINDS == (
            "InvalidInput",
            "ParamOutOfRange",
            "Unsupported",
            "ResourceExhausted",
            "NumericalFailure",
            "Timeout",
            "Unknown",
        )
        assert AUTOMATON_EXIT_CODES == {
            "InvalidInput": 1,
            "ParamOutOfRange": 2,
            "Unsupported": 2,
            "ResourceExhausted": 1,
            "NumericalFailure": 1,
            "Timeout": 124,
            "Unknown": 1,
        }

    def test_preserves_parameter_and_inherited_flag_contracts(self) -> None:
        """Preserve parameter JSON types and inherited flag order.

        Parser metadata must expose every supported type and flag in order.

        Notes
        -----
        Compare the public type mapping and flag names with fixed values.
        """
        assert AUTOMATON_PARAM_TYPES == (
            "str",
            "int",
            "float",
            "bool",
            "dict",
            "list",
        )
        assert AUTOMATON_JSON_TYPES == {
            "str": "string",
            "int": "integer",
            "float": "number",
            "bool": "boolean",
            "dict": "object",
            "list": "array",
        }
        assert tuple(flag for flag, _ in AUTOMATON_INHERITED_FLAGS) == (
            "--describe",
            "--params",
            "--validate",
            "--estimate",
            "--outdir",
            "--seed",
            "--smoke",
            "--cache",
            "--unchecked",
            "--deadline",
            "--json",
        )

    def test_preserves_runtime_paths_media_and_pin_matching(self) -> None:
        """Preserve runtime paths media types and pin matching behavior.

        Artifact records and script metadata must retain stable constants.

        Notes
        -----
        Compare environment values media types and two dependency-pin matches.
        """
        pattern: re.Pattern[str] = re.compile(AUTOMATON_PIN_PATTERN)

        assert AUTOMATON_DEFAULT_OUTDIR == "automaton_runs"
        assert AUTOMATON_PREVIEW_MAX_BYTES == 65536
        assert AUTOMATON_SMOKE_WALL_SECONDS == 60.0
        assert AUTOMATON_RUNTIME_CHECK_ENV_VAR == "JAXTYPING_DISABLE"
        assert AUTOMATON_CACHE_ENV_VAR == "DIFFPES_JAX_CACHE_DIR"
        assert AUTOMATON_ARTIFACT_MIMES[".json"] == "application/json"
        assert AUTOMATON_ARTIFACT_MIMES[".npz"] == "application/npz"
        assert AUTOMATON_ARTIFACT_MIMES[".png"] == "image/png"
        assert AUTOMATON_ARTIFACT_MIMES[".h5"] == "application/x-hdf5"
        assert (
            AUTOMATON_ARTIFACT_MIMES[".stablehlo"]
            == "application/vnd.stablehlo"
        )
        assert pattern.fullmatch("diffpes==2026.06.13") is not None
        assert pattern.fullmatch("diffpes[cuda]==2026.06.13") is not None
