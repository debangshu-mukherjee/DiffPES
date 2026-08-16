"""Define immutable carriers for executable experiment descriptions.

Extended Summary
----------------
This module stores static experiment metadata and one runtime context. The
factories validate host-side inputs before a script starts its calculation.
The context keeps its JAX random key as the only traced field.

Routine Listings
----------------
:class:`ArtifactRecord`
    Store one manifest record for a saved artifact.
:class:`AutomatonContext`
    Store runtime inputs for one executable experiment.
:class:`AutomatonParam`
    Store one validated executable parameter description.
:class:`AutomatonSpec`
    Store metadata that describes one executable experiment.
:func:`make_artifact_record`
    Create a validated record for one saved artifact.
:func:`make_automaton_context`
    Create a runtime context and its deterministic JAX key.
:func:`make_automaton_param`
    Create a validated executable parameter description.
:func:`make_automaton_spec`
    Create metadata for one executable experiment.
"""

from __future__ import annotations

import json
import keyword
from pathlib import Path

import equinox as eqx
import jax
from beartype import beartype
from beartype.typing import Any, Callable, Mapping, Optional, Sequence, Tuple
from jaxtyping import Array, Key, jaxtyped

from diffpes.constants import AUTOMATON_PARAM_TYPES

from . import (
    aliases as _MISSING,  # noqa: N812 -- module object is the sentinel.
)


def _normalize_static_value(value: Any) -> Any:
    """PRIVATE: Normalize list values to immutable tuples.

    Parameters
    ----------
    value : Any
        Static metadata value that can contain nested lists or tuples.

    Returns
    -------
    normalized : Any
        Value with every list converted to a tuple recursively.

    Notes
    -----
    Keeps scalar values unchanged. The conversion makes static Equinox
    metadata stable during PyTree comparison.
    """
    if isinstance(value, list):
        normalized: Any = tuple(
            _normalize_static_value(item) for item in value
        )
    elif isinstance(value, tuple):
        normalized = tuple(_normalize_static_value(item) for item in value)
    elif isinstance(value, dict):
        normalized = {
            key: _normalize_static_value(item) for key, item in value.items()
        }
    else:
        normalized = value
    return normalized


def _validate_supported_value(value: Any, python_type: type) -> None:
    """PRIVATE: Validate one static value against a declared Python type.

    Parameters
    ----------
    value : Any
        Default, choice, or example value to validate.
    python_type : type
        Declared public parameter type.

    Raises
    ------
    ValueError
        If the value does not satisfy the declared type.

    Notes
    -----
    Accepts integer values for float parameters. It rejects bool values for
    numeric parameters because Python treats bool as an integer subclass.
    """
    is_valid: bool
    if python_type is bool:
        is_valid = isinstance(value, bool)
    elif python_type is int:
        is_valid = isinstance(value, int) and not isinstance(value, bool)
    elif python_type is float:
        is_valid = isinstance(value, (int, float)) and not isinstance(
            value, bool
        )
    elif python_type is list:
        is_valid = isinstance(value, (list, tuple))
    else:
        is_valid = isinstance(value, python_type)
    if not is_valid:
        message: str = (
            f"value for {python_type.__name__!r} must match its declared type"
        )
        raise ValueError(message)


class AutomatonParam(eqx.Module):
    """Store one validated executable parameter description.

    The carrier keeps declaration metadata outside the JAX leaf partition.
    It records whether a default and an example exist without exposing a
    sentinel through the public API.

    :see: :class:`~.test_automaton.TestAutomatonParam`

    Attributes
    ----------
    name : str
        Valid Python identifier used by the executable parameter.
    python_type : type
        One supported Python parameter type.
    required : bool
        Whether callers must provide a value.
    default : Any
        Default value when ``required`` is false.
    help : str
        Human-readable parameter explanation.
    unit : Optional[str]
        Optional physical unit string.
    bounds : Optional[Tuple[Optional[float], Optional[float]]]
        Inclusive numeric lower and upper bounds.
    choices : Optional[Tuple[Any, ...]]
        Optional allowed values.
    example : Any
        Example value when ``has_example`` is true.
    has_example : bool
        Whether the carrier stores an example value.

    See Also
    --------
    make_automaton_param : Create a validated executable parameter description.
    """

    name: str = eqx.field(static=True)
    python_type: type = eqx.field(static=True)
    required: bool = eqx.field(static=True)
    default: Any = eqx.field(static=True)
    help: str = eqx.field(static=True)
    unit: Optional[str] = eqx.field(static=True)
    bounds: Optional[Tuple[Optional[float], Optional[float]]] = eqx.field(
        static=True
    )
    choices: Optional[Tuple[Any, ...]] = eqx.field(static=True)
    example: Any = eqx.field(static=True)
    has_example: bool = eqx.field(static=True)


@jaxtyped(typechecker=beartype)
def make_automaton_param(  # noqa: PLR0912, PLR0915
    name: str,
    python_type: type,
    *,
    default: Any = _MISSING,
    help: str = "",  # noqa: A002
    unit: Optional[str] = None,
    bounds: Optional[Tuple[Optional[float], Optional[float]]] = None,
    choices: Optional[Sequence[Any]] = None,
    example: Any = _MISSING,
) -> AutomatonParam:
    """Create a validated executable parameter description.

    The factory validates static metadata. It turns list metadata into tuples
    so each carrier stays immutable during a JAX PyTree operation.

    :see: :class:`~.test_automaton.TestMakeAutomatonParam`

    Implementation Logic
    --------------------
    1. **Validate the declaration**::

           valid_name = name.isidentifier() and not keyword.iskeyword(name)

       The check keeps generated command-line destinations unambiguous.

    2. **Normalize static sequences**::

           normalized_default = _normalize_static_value(default)

       The conversion gives list metadata an immutable carrier form.

    Parameters
    ----------
    name : str
        Parameter identifier used by Python and the command line.
    python_type : type
        One of ``str``, ``int``, ``float``, ``bool``, ``dict``, or ``list``.
    default : Any, optional
        Optional default value. Omit it to make the parameter required.
    help : str, optional
        Short parameter explanation. Default is an empty string.
    unit : Optional[str], optional
        Optional physical unit. Default is ``None``.
    bounds : Optional[Tuple[Optional[float], Optional[float]]], optional
        Inclusive numeric bounds. Default is ``None``.
    choices : Optional[Sequence[Any]], optional
        Optional allowed values. Default is ``None``.
    example : Any, optional
        Optional example value for description output. Omit it when absent.

    Returns
    -------
    parameter : AutomatonParam
        Immutable validated executable parameter metadata.

    Raises
    ------
    ValueError
        If the identifier, type, bounds, choices, default, or example is
        invalid.
    """
    supported_names: Tuple[str, ...] = AUTOMATON_PARAM_TYPES
    type_name: str = getattr(python_type, "__name__", "")
    if not name.isidentifier() or keyword.iskeyword(name):
        message: str = "name must be a non-keyword Python identifier"
        raise ValueError(message)
    if type_name not in supported_names or python_type not in {
        str,
        int,
        float,
        bool,
        dict,
        list,
    }:
        message = "python_type must be one supported parameter type"
        raise ValueError(message)
    if not isinstance(help, str):
        message = "help must be a string"
        raise ValueError(message)
    if unit is not None and not isinstance(unit, str):
        message = "unit must be a string or None"
        raise ValueError(message)
    if bounds is not None:
        if len(bounds) != 2:  # noqa: PLR2004
            message = "bounds must contain a lower and upper value"
            raise ValueError(message)
        lower: Optional[float]
        upper: Optional[float]
        lower, upper = bounds
        if lower is not None and (
            not isinstance(lower, (int, float)) or isinstance(lower, bool)
        ):
            message = "lower bound must be numeric or None"
            raise ValueError(message)
        if upper is not None and (
            not isinstance(upper, (int, float)) or isinstance(upper, bool)
        ):
            message = "upper bound must be numeric or None"
            raise ValueError(message)
        if lower is not None and upper is not None and lower > upper:
            message = "lower bound must not exceed upper bound"
            raise ValueError(message)
    normalized_choices: Optional[Tuple[Any, ...]]
    if choices is None:
        normalized_choices = None
    else:
        choice: Any
        for choice in choices:
            _validate_supported_value(choice, python_type)
        normalized_choices = tuple(
            _normalize_static_value(choice) for choice in choices
        )
        if not normalized_choices:
            message = "choices must not be empty"
            raise ValueError(message)
    required: bool = default is _MISSING
    if required:
        normalized_default: Any = None
    else:
        _validate_supported_value(default, python_type)
        normalized_default = _normalize_static_value(default)
        if (
            normalized_choices is not None
            and normalized_default not in normalized_choices
        ):
            message = "default must occur in choices"
            raise ValueError(message)
    has_example: bool = example is not _MISSING
    if has_example:
        _validate_supported_value(example, python_type)
        normalized_example: Any = _normalize_static_value(example)
        if (
            normalized_choices is not None
            and normalized_example not in normalized_choices
        ):
            message = "example must occur in choices"
            raise ValueError(message)
    else:
        normalized_example = None
    parameter: AutomatonParam = AutomatonParam(
        name=name,
        python_type=python_type,
        required=required,
        default=normalized_default,
        help=help,
        unit=unit,
        bounds=bounds,
        choices=normalized_choices,
        example=normalized_example,
        has_example=has_example,
    )
    return parameter


class AutomatonSpec(eqx.Module):
    """Store metadata that describes one executable experiment.

    The carrier stores parameter declarations and a canonical returns record.
    All fields stay static because they define a host-side command interface.

    :see: :class:`~.test_automaton.TestAutomatonSpec`

    Attributes
    ----------
    name : str
        Stable experiment identifier.
    params : Tuple[AutomatonParam, ...]
        Ordered parameter declarations.
    returns_json : str
        Canonical JSON description of the declared result fields.
    description : str
        Short experiment description.
    estimate : Optional[Callable[[Any], Mapping[str, Any]]]
        Optional host-side resource estimate callable.

    See Also
    --------
    make_automaton_spec : Create metadata for one executable experiment.
    """

    name: str = eqx.field(static=True)
    params: Tuple[AutomatonParam, ...] = eqx.field(static=True)
    returns_json: str = eqx.field(static=True)
    description: str = eqx.field(static=True)
    estimate: Optional[Callable[[Any], Mapping[str, Any]]] = eqx.field(
        static=True
    )


@jaxtyped(typechecker=beartype)
def make_automaton_spec(
    name: str,
    params: Sequence[AutomatonParam],
    *,
    returns: Optional[Mapping[str, Any]] = None,
    description: str = "",
    estimate: Optional[Callable[[Any], Mapping[str, Any]]] = None,
) -> AutomatonSpec:
    """Create metadata for one executable experiment.

    The factory validates unique parameter names and serializes declared
    result metadata with deterministic JSON key ordering.

    :see: :class:`~.test_automaton.TestMakeAutomatonSpec`

    Implementation Logic
    --------------------
    1. **Collect parameter names**::

           names = tuple(parameter.name for parameter in params)

       The collection detects duplicate command-line destinations.

    2. **Serialize the returns mapping**::

           returns_json = json.dumps(returns_mapping, sort_keys=True)

       Stable ordering gives description output a deterministic representation.

    Parameters
    ----------
    name : str
        Stable executable experiment identifier.
    params : Sequence[AutomatonParam]
        Ordered executable parameter declarations.
    returns : Optional[Mapping[str, Any]], optional
        Declared result fields. Default is an empty mapping.
    description : str, optional
        Short experiment description. Default is an empty string.
    estimate : Optional[Callable[[Any], Mapping[str, Any]]], optional
        Optional host-side resource estimate callable. Default is ``None``.

    Returns
    -------
    spec : AutomatonSpec
        Immutable validated experiment metadata.

    Raises
    ------
    ValueError
        If the name, parameter sequence, returns mapping, or estimate is
        invalid.
    """
    exc: Exception
    if not name or not isinstance(name, str):
        message: str = "name must be a nonempty string"
        raise ValueError(message)
    if not isinstance(description, str):
        message = "description must be a string"
        raise ValueError(message)
    if estimate is not None and not callable(estimate):
        message = "estimate must be callable or None"
        raise ValueError(message)
    normalized_params: Tuple[AutomatonParam, ...] = tuple(params)
    if not all(
        isinstance(parameter, AutomatonParam)
        for parameter in normalized_params
    ):
        message = "params must contain AutomatonParam values"
        raise ValueError(message)
    names: Tuple[str, ...] = tuple(
        parameter.name for parameter in normalized_params
    )
    if len(names) != len(set(names)):
        message = "params must use unique names"
        raise ValueError(message)
    returns_mapping: Mapping[str, Any] = {} if returns is None else returns
    if not isinstance(returns_mapping, Mapping):
        message = "returns must be a mapping or None"
        raise ValueError(message)
    try:
        returns_json: str = json.dumps(
            dict(returns_mapping),
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as exc:
        message = "returns must contain JSON-compatible values"
        raise ValueError(message) from exc
    spec: AutomatonSpec = AutomatonSpec(
        name=name,
        params=normalized_params,
        returns_json=returns_json,
        description=description,
        estimate=estimate,
    )
    return spec


class AutomatonContext(eqx.Module):
    """Store runtime inputs for one executable experiment.

    The carrier gives artifact writers an output root and gives a body one
    deterministic JAX key. The metadata fields remain static across tracing.

    :see: :class:`~.test_automaton.TestAutomatonContext`

    Attributes
    ----------
    outdir : str
        Existing root directory for generated artifacts.
    seed : int
        Deterministic random seed.
    experiment : str
        Stable experiment identifier for logs and result records.
    json_mode : bool
        Whether host-side log messages stay silent.
    rng_key : Key[Array, ""]
        JAX random key derived from ``seed``.

    See Also
    --------
    make_automaton_context : Create a runtime context with a JAX key.
    """

    outdir: str = eqx.field(static=True)
    seed: int = eqx.field(static=True)
    experiment: str = eqx.field(static=True)
    json_mode: bool = eqx.field(static=True)
    rng_key: Key[Array, ""]


@jaxtyped(typechecker=beartype)
def make_automaton_context(
    outdir: str | Path,
    seed: int,
    experiment: str,
    *,
    json_mode: bool = False,
) -> AutomatonContext:
    """Create a runtime context and its deterministic JAX key.

    The factory creates the output directory before it builds a typed JAX
    random key. It keeps all host metadata static in the returned carrier.

    :see: :class:`~.test_automaton.TestMakeAutomatonContext`

    Implementation Logic
    --------------------
    1. **Create the artifact root**::

           output_path.mkdir(parents=True, exist_ok=True)

       The operation lets every writer create nested artifact paths safely.

    2. **Build the random key**::

           rng_key = jax.random.key(seed)

       The key makes repeated seeded experiment runs deterministic.

    Parameters
    ----------
    outdir : str | Path
        Root directory for generated artifacts.
    seed : int
        Deterministic non-boolean random seed.
    experiment : str
        Nonempty experiment identifier.
    json_mode : bool, optional
        Suppress host-side log messages when true. Default is ``False``.

    Returns
    -------
    context : AutomatonContext
        Context with an existing output root and deterministic JAX key.

    Raises
    ------
    ValueError
        If the seed or experiment identifier is invalid.
    """
    if not isinstance(seed, int) or isinstance(seed, bool):
        message: str = "seed must be an integer"
        raise ValueError(message)
    if not isinstance(experiment, str) or not experiment:
        message = "experiment must be a nonempty string"
        raise ValueError(message)
    output_path: Path = Path(outdir)
    output_path.mkdir(parents=True, exist_ok=True)
    rng_key: Key[Array, ""] = jax.random.key(seed)
    context: AutomatonContext = AutomatonContext(
        outdir=str(output_path),
        seed=seed,
        experiment=experiment,
        json_mode=json_mode,
        rng_key=rng_key,
    )
    return context


class ArtifactRecord(eqx.Module):
    """Store one manifest record for a saved artifact.

    The carrier holds relative artifact metadata. It stores an empty preview
    string when the file has no embedded preview data.

    :see: :class:`~.test_automaton.TestArtifactRecord`

    Attributes
    ----------
    role : str
        Domain role declared by the executable experiment.
    mime : str
        Internet media type for the artifact file.
    path : str
        Relative POSIX path below the experiment output root.
    preview_b64 : str
        Optional base64 preview, or an empty string.

    See Also
    --------
    make_artifact_record : Create a validated record for one saved artifact.
    """

    role: str = eqx.field(static=True)
    mime: str = eqx.field(static=True)
    path: str = eqx.field(static=True)
    preview_b64: str = eqx.field(static=True)


@jaxtyped(typechecker=beartype)
def make_artifact_record(
    role: str,
    mime: str,
    path: str,
    *,
    preview_b64: str = "",
) -> ArtifactRecord:
    """Create a validated record for one saved artifact.

    The factory checks the user-visible metadata. Artifact writers perform
    containment checks before they call this factory.

    :see: :class:`~.test_automaton.TestMakeArtifactRecord`

    Implementation Logic
    --------------------
    1. **Validate text fields**::

           valid = bool(role and mime and path)

       The check prevents incomplete manifest records.

    2. **Construct the immutable record**::

           record = ArtifactRecord(role, mime, path, preview_b64)

       The carrier keeps all manifest metadata outside JAX tracing.

    Parameters
    ----------
    role : str
        Domain role declared by the executable experiment.
    mime : str
        Internet media type for the saved file.
    path : str
        Relative POSIX path below the output root.
    preview_b64 : str, optional
        Base64 preview data. Default is an empty string.

    Returns
    -------
    record : ArtifactRecord
        Immutable artifact manifest record.

    Raises
    ------
    ValueError
        If a required text field is empty or the path is absolute.
    """
    relative_path: Path = Path(path)
    if not all(
        isinstance(value, str) for value in (role, mime, path, preview_b64)
    ):
        message: str = "artifact metadata must use strings"
        raise ValueError(message)
    if not role or not mime or not path:
        message = "role, mime, and path must be nonempty"
        raise ValueError(message)
    if relative_path.is_absolute() or ".." in relative_path.parts:
        message = "path must stay relative to the artifact root"
        raise ValueError(message)
    record: ArtifactRecord = ArtifactRecord(
        role=role,
        mime=mime,
        path=relative_path.as_posix(),
        preview_b64=preview_b64,
    )
    return record


__all__: list[str] = [
    "ArtifactRecord",
    "AutomatonContext",
    "AutomatonParam",
    "AutomatonSpec",
    "make_artifact_record",
    "make_automaton_context",
    "make_automaton_param",
    "make_automaton_spec",
]
