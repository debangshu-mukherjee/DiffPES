"""Build JSON result and description payloads for executable experiments.

Extended Summary
----------------
This module sanitizes host values for strict JSON output. It builds stable
description and result payloads, including content-addressed result keys. The
emitter writes exactly one sorted JSON object to standard output.

Routine Listings
----------------
:func:`build_result`
    Build one JSON-ready executable result payload.
:func:`describe_payload`
    Build one JSON-ready executable description payload.
:func:`emit`
    Write one sorted JSON payload line to standard output.
:func:`json_ready`
    Convert host values to strict JSON-ready data.
:func:`result_key`
    Compute a content-addressed executable result key.
"""

from __future__ import annotations

import hashlib
import json
import math
from importlib.metadata import version
from pathlib import Path

import jax
import numpy as np
from beartype import beartype
from beartype.typing import Any, Dict, Mapping, Optional, Sequence
from jaxtyping import jaxtyped

from diffpes.certify import canonical_json
from diffpes.constants import (
    AUTOMATON_DESCRIBE_SCHEMA_VERSION,
    AUTOMATON_INHERITED_FLAGS,
    AUTOMATON_RESULT_SCHEMA_VERSION,
    AUTOMATON_STATUS_VALUES,
)
from diffpes.types import ArtifactRecord, AutomatonSpec

from .parameters import params_json_schema


def _package_version() -> str:
    """PRIVATE: Read the installed diffpes version without root-import cycles.

    Returns
    -------
    package_version : str
        Installed diffpes distribution version.

    Notes
    -----
    Uses distribution metadata because the root package imports this module
    before it binds its own ``__version__`` attribute.
    """
    package_version: str = version("diffpes")
    return package_version


def _artifact_payload(record: ArtifactRecord) -> Dict[str, Any]:
    """PRIVATE: Convert one immutable artifact record to JSON data.

    Parameters
    ----------
    record : ArtifactRecord
        Immutable relative artifact manifest record.

    Returns
    -------
    payload : Dict[str, Any]
        JSON-ready artifact mapping.

    Notes
    -----
    Preserves the public manifest field names. The conversion has no file
    system effect and does not inspect preview bytes.
    """
    payload: Dict[str, Any] = {
        "role": record.role,
        "mime": record.mime,
        "path": record.path,
        "preview_b64": record.preview_b64,
    }
    return payload


@jaxtyped(typechecker=beartype)
def json_ready(value: Any) -> Any:
    """Convert host values to strict JSON-ready data.

    The function converts arrays to lists, paths to text, and non-finite
    floating values to null. It recurses through mappings and sequences.

    :see: :class:`~.test_results.TestJsonReady`

    Implementation Logic
    --------------------
    1. **Handle scalar values**::

           ready = None if not math.isfinite(value) else value

       The conversion prevents JSON from emitting invalid NaN or infinity.

    2. **Recurse through containers**::

           ready = {str(key): json_ready(item) for key, item in value.items()}

       The recursion gives nested host data one JSON representation.

    Parameters
    ----------
    value : Any
        Host scalar, path, mapping, sequence, NumPy value, or JAX array.

    Returns
    -------
    ready : Any
        Strict JSON-compatible representation with non-finite values as null.
    """
    if value is None or isinstance(value, (str, bool)):
        ready: Any = value
    elif isinstance(value, Path):
        ready = str(value)
    elif isinstance(value, ArtifactRecord):
        ready = _artifact_payload(value)
    elif isinstance(value, Mapping):
        ready = {str(key): json_ready(item) for key, item in value.items()}
    elif isinstance(value, (list, tuple)):
        ready = [json_ready(item) for item in value]
    elif (
        isinstance(value, np.ndarray)
        or hasattr(value, "shape")
        and hasattr(value, "tolist")
    ):
        ready = json_ready(value.tolist())
    elif isinstance(value, np.generic):
        ready = json_ready(value.item())
    elif isinstance(value, float):
        ready = value if math.isfinite(value) else None
    elif isinstance(value, complex):
        ready = {
            "real": json_ready(float(value.real)),
            "imag": json_ready(float(value.imag)),
        }
    elif isinstance(value, (int,)):
        ready = value
    else:
        ready = str(value)
    return ready


@jaxtyped(typechecker=beartype)
def result_key(
    experiment: str,
    params: Mapping[str, Any],
    seed: int,
    version: str,
) -> str:
    """Compute a content-addressed executable result key.

    The key hashes canonical JSON for the experiment identifier, parameters,
    seed, and installed diffpes version. Artifact contents do not affect it.

    :see: :class:`~.test_results.TestResultKey`

    Implementation Logic
    --------------------
    1. **Build the identity object**::

           identity = {
               "experiment": experiment, "params": params, "seed": seed
           }

       The mapping contains only inputs that identify a logical run.

    2. **Hash canonical bytes**::

           digest = hashlib.sha256(canonical_json(identity)).hexdigest()

       Canonical encoding makes key ordering and numeric text deterministic.

    Parameters
    ----------
    experiment : str
        Stable executable experiment identifier.
    params : Mapping[str, Any]
        Fully validated parameter object.
    seed : int
        Deterministic random seed.
    version : str
        Installed diffpes distribution version.

    Returns
    -------
    digest : str
        Lowercase SHA-256 hexadecimal result key.
    """
    identity: Dict[str, Any] = {
        "experiment": experiment,
        "params": json_ready(dict(params)),
        "seed": seed,
        "diffpes_version": version,
    }
    encoded: bytes = canonical_json(identity)
    digest: str = hashlib.sha256(encoded).hexdigest()
    return digest


@jaxtyped(typechecker=beartype)
def build_result(
    spec: AutomatonSpec,
    status: str,
    params: Mapping[str, Any],
    seed: int,
    metrics: Mapping[str, Any],
    artifacts: Sequence[ArtifactRecord],
    extras: Mapping[str, Any],
    wall_seconds: float,
    *,
    error: Optional[Exception] = None,
) -> Dict[str, Any]:
    """Build one JSON-ready executable result payload.

    The payload records stable identity fields, sanitized metrics, artifact
    records, declared returns, and optional classified error details.

    :see: :class:`~.test_results.TestBuildResult`

    Implementation Logic
    --------------------
    1. **Collect standard fields**::

           payload = {"status": status, "params": json_ready(params)}

       The standard fields remain present for success and failure results.

    2. **Merge body extras**::

           payload.update(json_ready(dict(extras)))

       The merge carries result objects without new schema keys.

    Parameters
    ----------
    spec : AutomatonSpec
        Static metadata for the executable experiment.
    status : str
        ``"ok"``, ``"error"``, or ``"timeout"`` result status.
    params : Mapping[str, Any]
        Fully validated parameter object.
    seed : int
        Deterministic random seed.
    metrics : Mapping[str, Any]
        Result metrics supplied by the experiment body.
    artifacts : Sequence[ArtifactRecord]
        Manifest records supplied by the experiment body.
    extras : Mapping[str, Any]
        Additional experiment-specific top-level result fields.
    wall_seconds : float
        Elapsed body execution time in seconds.
    error : Optional[Exception], optional
        Classified body or validation failure. Default is ``None``.

    Returns
    -------
    payload : Dict[str, Any]
        Strict JSON-ready executable result object.

    Raises
    ------
    ValueError
        Raise for an unsupported status.
    """
    if status not in AUTOMATON_STATUS_VALUES:
        message: str = "status must be an accepted executable status"
        raise ValueError(message)
    package_version: str = _package_version()
    returns: Any = json.loads(spec.returns_json)
    artifact_payloads: Sequence[Dict[str, Any]] = [
        _artifact_payload(record) for record in artifacts
    ]
    payload: Dict[str, Any] = {
        "schema_version": AUTOMATON_RESULT_SCHEMA_VERSION,
        "status": status,
        "experiment": spec.name,
        "diffpes_version": package_version,
        "jax_backend": jax.default_backend(),
        "seed": seed,
        "params": json_ready(dict(params)),
        "metrics": json_ready(dict(metrics)),
        "artifacts": artifact_payloads,
        "wall_seconds": json_ready(float(wall_seconds)),
        "returns": json_ready(returns),
        "result_key": result_key(spec.name, params, seed, package_version),
    }
    extra_payload: Dict[str, Any] = json_ready(dict(extras))
    protected_keys: set[str] = set(payload)
    key: str
    item: Any
    for key, item in extra_payload.items():
        if key not in protected_keys:
            payload[key] = item
    if error is not None:
        payload["error"] = str(error)
        payload["error_kind"] = getattr(error, "error_kind", "Unknown")
        payload["field"] = getattr(error, "field", None)
    return payload


@jaxtyped(typechecker=beartype)
def describe_payload(spec: AutomatonSpec) -> Dict[str, Any]:
    """Build one JSON-ready executable description payload.

    The payload exposes the stable schema version, experiment summary, JSON
    parameter schema, declared returns, and inherited command-line flags.

    :see: :class:`~.test_results.TestDescribePayload`

    Notes
    -----
    Parses the canonical returns text stored by ``make_automaton_spec``. The
    payload does not inspect a user parameter document or run an experiment.

    Parameters
    ----------
    spec : AutomatonSpec
        Static metadata for the executable experiment.

    Returns
    -------
    payload : Dict[str, Any]
        Strict JSON-ready executable description object.
    """
    inherited_flags: Sequence[Sequence[str]] = [
        [flag, help_text] for flag, help_text in AUTOMATON_INHERITED_FLAGS
    ]
    payload: Dict[str, Any] = {
        "schema_version": AUTOMATON_DESCRIBE_SCHEMA_VERSION,
        "experiment": spec.name,
        "summary": spec.description,
        "params_schema": params_json_schema(spec.params),
        "returns": json.loads(spec.returns_json),
        "inherited_flags": inherited_flags,
    }
    return payload


@jaxtyped(typechecker=beartype)
def emit(result: Mapping[str, Any]) -> None:
    """Write one sorted JSON payload line to standard output.

    The function sanitizes values before it prints exactly one flushed line.
    It never emits a prefix or a human-readable status message.

    :see: :class:`~.test_results.TestEmit`

    Notes
    -----
    Uses strict JSON options so callers cannot accidentally write NaN or
    unsorted output that agents cannot parse reproducibly.

    Parameters
    ----------
    result : Mapping[str, Any]
        JSON-like result or description payload.
    """
    ready: Any = json_ready(dict(result))
    text: str = json.dumps(ready, allow_nan=False, sort_keys=True)
    print(text, flush=True)


__all__: list[str] = [
    "build_result",
    "describe_payload",
    "emit",
    "json_ready",
    "result_key",
]
