"""Provide the executable experiment process boundary for diffpes.

Extended Summary
----------------
This subpackage provides reusable parameter, artifact, result, and runner
utilities for agent-facing executable experiments. It keeps host-side process
control separate from differentiable scientific computations.

The following submodules organize the executable process boundary:

- :mod:`artifacts`
    Write and record artifacts from executable experiments.
- :mod:`errors`
    Map failures from executable experiment runs.
- :mod:`parameters`
    Parse and validate executable experiment parameters.
- :mod:`reference_models`
    Build compact tight-binding reference models for executable experiments.
- :mod:`results`
    Build JSON result and description payloads for executable experiments.
- :mod:`runner`
    Run executable experiments through one command-line contract.

Routine Listings
----------------
:class:`AutomatonError`
    Report one classified executable experiment failure.
:class:`DeadlineExceededError`
    Report a wall-time deadline failure.
:func:`artifact_path`
    Resolve one safe artifact path below an experiment output root.
:func:`artifact_record_as_dict`
    Convert one artifact record to a JSON-ready dictionary.
:func:`build_parser`
    Build an argument parser for one executable experiment.
:func:`build_result`
    Build one JSON-ready executable result payload.
:func:`classify_exception`
    Map one Python exception to an executable result.
:func:`coerce_param_value`
    Convert one input value to a declared parameter type.
:func:`deadline_context`
    Enforce an optional POSIX wall-time deadline.
:func:`describe_param`
    Build one parameter description for executable introspection.
:func:`describe_payload`
    Build one JSON-ready executable description payload.
:func:`emit`
    Write one sorted JSON payload line to standard output.
:func:`enable_compilation_cache`
    Set the JAX persistent compilation cache.
:func:`exit_code_for`
    Return the process exit code for one error category.
:func:`experiment`
    Decorate one experiment body with the executable command-line contract.
:func:`graphene_pz_model`
    Build the nearest-neighbor graphene pz reference model.
:func:`json_ready`
    Convert host values to strict JSON-ready data.
:func:`linear_chain_model`
    Build the nearest-neighbor one-dimensional chain reference model.
:func:`log_message`
    Write one human-readable experiment message to standard error.
:func:`merge_params`
    Apply defaults, document values, and command-line overrides.
:func:`param_json_schema`
    Build JSON Schema for one executable parameter.
:func:`params_json_schema`
    Build JSON Schema for executable parameter objects.
:func:`read_params_document`
    Read a JSON parameter object from a supported source.
:func:`record_artifact`
    Record an existing artifact below an experiment output root.
:func:`result_key`
    Compute a content-addressed executable result key.
:func:`run_automaton`
    Run one executable experiment with optional argument text.
:func:`save_array_artifact`
    Save compressed NumPy arrays and return a manifest record.
:func:`save_carrier_artifact`
    Save one diffpes carrier in HDF5 and return a manifest record.
:func:`save_figure_artifact`
    Save and close one Matplotlib figure and return a manifest record.
:func:`save_image_artifact`
    Save one image array and return a manifest record.
:func:`save_json_artifact`
    Save JSON data and return a manifest record.
:func:`two_orbital_dirac_model`
    Build a two-orbital lattice Dirac reference model.
:func:`validate_param_value`
    Validate one coerced executable parameter value.
"""

from .artifacts import (
    artifact_path,
    artifact_record_as_dict,
    log_message,
    record_artifact,
    save_array_artifact,
    save_carrier_artifact,
    save_figure_artifact,
    save_image_artifact,
    save_json_artifact,
)
from .errors import (
    AutomatonError,
    DeadlineExceededError,
    classify_exception,
    exit_code_for,
)
from .parameters import (
    coerce_param_value,
    describe_param,
    merge_params,
    param_json_schema,
    params_json_schema,
    read_params_document,
    validate_param_value,
)
from .reference_models import (
    graphene_pz_model,
    linear_chain_model,
    two_orbital_dirac_model,
)
from .results import (
    build_result,
    describe_payload,
    emit,
    json_ready,
    result_key,
)
from .runner import (
    build_parser,
    deadline_context,
    enable_compilation_cache,
    experiment,
    run_automaton,
)

__all__: list[str] = [
    "AutomatonError",
    "DeadlineExceededError",
    "artifact_path",
    "artifact_record_as_dict",
    "build_parser",
    "build_result",
    "classify_exception",
    "coerce_param_value",
    "deadline_context",
    "describe_param",
    "describe_payload",
    "emit",
    "enable_compilation_cache",
    "exit_code_for",
    "experiment",
    "graphene_pz_model",
    "json_ready",
    "linear_chain_model",
    "log_message",
    "merge_params",
    "param_json_schema",
    "params_json_schema",
    "read_params_document",
    "record_artifact",
    "result_key",
    "run_automaton",
    "save_array_artifact",
    "save_carrier_artifact",
    "save_figure_artifact",
    "save_image_artifact",
    "save_json_artifact",
    "two_orbital_dirac_model",
    "validate_param_value",
]
