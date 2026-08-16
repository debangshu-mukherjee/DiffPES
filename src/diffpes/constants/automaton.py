"""Define static values for executable experiment handling.

Extended Summary
----------------
This module centralizes schema identifiers, command-line vocabulary, result
status values, and artifact media types. The harness imports these values
through the public :mod:`diffpes.constants` surface.

Routine Listings
----------------
:obj:`AUTOMATON_ARTIFACT_MIMES`
    Map supported artifact suffixes to Internet media types.
:obj:`AUTOMATON_CACHE_ENV_VAR`
    Name the environment variable for the JAX compilation cache.
:obj:`AUTOMATON_DEFAULT_OUTDIR`
    Name the default root directory for experiment artifacts.
:obj:`AUTOMATON_DESCRIBE_SCHEMA_VERSION`
    Identify the executable description payload schema.
:obj:`AUTOMATON_ERROR_KINDS`
    List accepted executable error categories.
:obj:`AUTOMATON_EXIT_CODES`
    Map executable error categories to process exit codes.
:obj:`AUTOMATON_INHERITED_FLAGS`
    List inherited command-line flags and their help text.
:obj:`AUTOMATON_JSON_TYPES`
    Map supported Python parameter types to JSON Schema types.
:obj:`AUTOMATON_PARAM_TYPES`
    List supported Python parameter type names.
:obj:`AUTOMATON_PIN_PATTERN`
    Match a pinned diffpes requirement in script metadata.
:obj:`AUTOMATON_PREVIEW_MAX_BYTES`
    Limit embedded artifact previews in bytes.
:obj:`AUTOMATON_RESULT_SCHEMA_VERSION`
    Identify the executable result payload schema.
:obj:`AUTOMATON_RUNTIME_CHECK_ENV_VAR`
    Name the environment variable that disables runtime checks.
:obj:`AUTOMATON_SMOKE_WALL_SECONDS`
    Limit a CPU smoke execution wall time in seconds.
:obj:`AUTOMATON_STATUS_VALUES`
    List accepted executable result statuses.
"""

from beartype.typing import Dict, Final, Tuple

AUTOMATON_DESCRIBE_SCHEMA_VERSION: Final[str] = "diffpes.automaton.describe.v1"
AUTOMATON_RESULT_SCHEMA_VERSION: Final[str] = "diffpes.automaton.result.v1"
AUTOMATON_STATUS_VALUES: Final[Tuple[str, ...]] = ("ok", "error", "timeout")
AUTOMATON_ERROR_KINDS: Final[Tuple[str, ...]] = (
    "InvalidInput",
    "ParamOutOfRange",
    "Unsupported",
    "ResourceExhausted",
    "NumericalFailure",
    "Timeout",
    "Unknown",
)
AUTOMATON_EXIT_CODES: Final[Dict[str, int]] = {
    "InvalidInput": 1,
    "ParamOutOfRange": 2,
    "Unsupported": 2,
    "ResourceExhausted": 1,
    "NumericalFailure": 1,
    "Timeout": 124,
    "Unknown": 1,
}
AUTOMATON_PARAM_TYPES: Final[Tuple[str, ...]] = (
    "str",
    "int",
    "float",
    "bool",
    "dict",
    "list",
)
AUTOMATON_JSON_TYPES: Final[Dict[str, str]] = {
    "str": "string",
    "int": "integer",
    "float": "number",
    "bool": "boolean",
    "dict": "object",
    "list": "array",
}
AUTOMATON_INHERITED_FLAGS: Final[Tuple[Tuple[str, str], ...]] = (
    ("--describe", "Emit the parameter schema and result summary."),
    ("--params", "Read a JSON object from a file, standard input, or text."),
    ("--validate", "Validate parameters without running the experiment."),
    ("--estimate", "Emit the declared resource estimate."),
    ("--outdir", "Set the root directory for generated artifacts."),
    ("--seed", "Set the deterministic random seed."),
    ("--smoke", "Use the reduced CPU smoke configuration."),
    ("--cache", "Enable the JAX persistent compilation cache."),
    ("--unchecked", "Re-execute once with runtime checks disabled."),
    ("--deadline", "Set a POSIX wall-time deadline in seconds."),
    ("--json", "Suppress human-readable standard-error messages."),
)
AUTOMATON_DEFAULT_OUTDIR: Final[str] = "automaton_runs"
AUTOMATON_PREVIEW_MAX_BYTES: Final[int] = 65536
AUTOMATON_SMOKE_WALL_SECONDS: Final[float] = 60.0
AUTOMATON_RUNTIME_CHECK_ENV_VAR: Final[str] = "JAXTYPING_DISABLE"
AUTOMATON_CACHE_ENV_VAR: Final[str] = "DIFFPES_JAX_CACHE_DIR"
AUTOMATON_ARTIFACT_MIMES: Final[Dict[str, str]] = {
    ".csv": "text/csv",
    ".h5": "application/x-hdf5",
    ".json": "application/json",
    ".npz": "application/npz",
    ".pdf": "application/pdf",
    ".png": "image/png",
    ".stablehlo": "application/vnd.stablehlo",
    ".svg": "image/svg+xml",
    ".txt": "text/plain",
}
AUTOMATON_PIN_PATTERN: Final[str] = r"diffpes(?:\[cuda\])?==[0-9][^\"']*"

__all__: list[str] = [
    "AUTOMATON_ARTIFACT_MIMES",
    "AUTOMATON_CACHE_ENV_VAR",
    "AUTOMATON_DEFAULT_OUTDIR",
    "AUTOMATON_DESCRIBE_SCHEMA_VERSION",
    "AUTOMATON_ERROR_KINDS",
    "AUTOMATON_EXIT_CODES",
    "AUTOMATON_INHERITED_FLAGS",
    "AUTOMATON_JSON_TYPES",
    "AUTOMATON_PARAM_TYPES",
    "AUTOMATON_PIN_PATTERN",
    "AUTOMATON_PREVIEW_MAX_BYTES",
    "AUTOMATON_RESULT_SCHEMA_VERSION",
    "AUTOMATON_RUNTIME_CHECK_ENV_VAR",
    "AUTOMATON_SMOKE_WALL_SECONDS",
    "AUTOMATON_STATUS_VALUES",
]
