"""Parse and validate executable experiment parameters.

Extended Summary
----------------
This module converts static parameter carriers into JSON Schema metadata. It
also merges JSON documents and command-line values into validated Python data.
The module reports field-specific executable errors for invalid input.

Routine Listings
----------------
:func:`coerce_param_value`
    Convert one input value to a declared parameter type.
:func:`describe_param`
    Build one parameter description for executable introspection.
:func:`merge_params`
    Apply defaults, document values, and command-line overrides.
:func:`param_json_schema`
    Build JSON Schema for one executable parameter.
:func:`params_json_schema`
    Build JSON Schema for executable parameter objects.
:func:`read_params_document`
    Read a JSON parameter object from a supported source.
:func:`validate_param_value`
    Validate one coerced executable parameter value.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from beartype import beartype
from beartype.typing import Any, Dict, List, Mapping, Sequence
from jaxtyping import jaxtyped

from diffpes.constants import AUTOMATON_JSON_TYPES
from diffpes.types import AutomatonParam

from .errors import AutomatonError


def _error(message: str, error_kind: str, field: str) -> Exception:
    """PRIVATE: Build one field-specific executable exception.

    Parameters
    ----------
    message : str
        Reader-facing validation message.
    error_kind : str
        Stable executable error category.
    field : str
        Parameter field that caused the rejection.

    Returns
    -------
    error : Exception
        Classified executable exception for the rejected parameter.

    Notes
    -----
    Uses the public lazy exception surface. The helper keeps error creation
    consistent across parsing, coercion, and validation.
    """
    error: Exception = AutomatonError(
        message, error_kind=error_kind, field=field
    )
    return error


def _json_static(value: Any) -> Any:
    """PRIVATE: Convert immutable static metadata to JSON-ready values.

    Parameters
    ----------
    value : Any
        Static carrier metadata value.

    Returns
    -------
    ready : Any
        Equivalent value with tuples converted to JSON lists.

    Notes
    -----
    Recurses through mappings and tuples. The helper preserves scalar values
    because JSON serialization handles them directly.
    """
    if isinstance(value, tuple):
        ready: Any = [_json_static(item) for item in value]
    elif isinstance(value, list):
        ready = [_json_static(item) for item in value]
    elif isinstance(value, Mapping):
        ready = {str(key): _json_static(item) for key, item in value.items()}
    else:
        ready = value
    return ready


def _comparison_value(value: Any) -> Any:
    """PRIVATE: Normalize nested lists before static choice comparison.

    Parameters
    ----------
    value : Any
        Parameter value that can contain JSON containers.

    Returns
    -------
    normalized : Any
        Value with lists converted to tuples recursively.

    Notes
    -----
    Matches the immutable sequence representation stored in parameter carriers.
    It does not alter scalar values or mapping keys.
    """
    if isinstance(value, list):
        normalized: Any = tuple(_comparison_value(item) for item in value)
    elif isinstance(value, tuple):
        normalized = tuple(_comparison_value(item) for item in value)
    elif isinstance(value, Mapping):
        normalized = {
            key: _comparison_value(item) for key, item in value.items()
        }
    else:
        normalized = value
    return normalized


@jaxtyped(typechecker=beartype)
def describe_param(param: AutomatonParam) -> Dict[str, Any]:
    """Build one parameter description for executable introspection.

    The returned object includes the declared type, validation metadata, and
    optional default or example values in JSON-ready form.

    :see: :class:`~.test_parameters.TestDescribeParam`

    Notes
    -----
    Reads only static carrier fields. The function does not coerce or validate
    a caller-provided parameter value.

    Parameters
    ----------
    param : AutomatonParam
        Static declaration of one executable parameter.

    Returns
    -------
    description : Dict[str, Any]
        JSON-ready parameter description.
    """
    description: Dict[str, Any] = {
        "name": param.name,
        "python_type": param.python_type.__name__,
        "required": param.required,
        "help": param.help,
        "unit": param.unit,
        "bounds": _json_static(param.bounds),
        "choices": _json_static(param.choices),
        "has_example": param.has_example,
    }
    if not param.required:
        description["default"] = _json_static(param.default)
    if param.has_example:
        description["example"] = _json_static(param.example)
    return description


@jaxtyped(typechecker=beartype)
def param_json_schema(param: AutomatonParam) -> Dict[str, Any]:
    """Build JSON Schema for one executable parameter.

    The schema maps the supported Python type to its JSON Schema primitive.
    It carries choices, bounds, defaults, and examples when the carrier has
    them.

    :see: :class:`~.test_parameters.TestParamJsonSchema`

    Notes
    -----
    The schema describes values after command-line coercion. It does not
    encode executable error categories or process behavior.

    Parameters
    ----------
    param : AutomatonParam
        Static declaration of one executable parameter.

    Returns
    -------
    schema : Dict[str, Any]
        JSON Schema fragment for the declared parameter.
    """
    type_name: str = param.python_type.__name__
    schema: Dict[str, Any] = {"type": AUTOMATON_JSON_TYPES[type_name]}
    if param.help:
        schema["description"] = param.help
    if param.unit is not None:
        schema["unit"] = param.unit
    if param.bounds is not None:
        lower: float | None
        upper: float | None
        lower, upper = param.bounds
        if lower is not None:
            schema["minimum"] = lower
        if upper is not None:
            schema["maximum"] = upper
    if param.choices is not None:
        schema["enum"] = _json_static(param.choices)
    if not param.required:
        schema["default"] = _json_static(param.default)
    if param.has_example:
        schema["examples"] = [_json_static(param.example)]
    return schema


@jaxtyped(typechecker=beartype)
def params_json_schema(params: Sequence[AutomatonParam]) -> Dict[str, Any]:
    """Build JSON Schema for executable parameter objects.

    The schema requires declared required parameters and rejects unknown
    properties. Each property uses the fragment from ``param_json_schema``.

    :see: :class:`~.test_parameters.TestParamsJsonSchema`

    Implementation Logic
    --------------------
    1. **Build property fragments**::

           properties = {
               param.name: param_json_schema(param) for param in params
           }

       The mapping preserves each named validation contract.

    2. **Collect required fields**::

           required = [param.name for param in params if param.required]

       The list makes missing caller input visible to a schema consumer.

    Parameters
    ----------
    params : Sequence[AutomatonParam]
        Ordered parameter declarations for one executable experiment.

    Returns
    -------
    schema : Dict[str, Any]
        Draft 2020-12 JSON Schema for a parameter object.
    """
    properties: Dict[str, Any] = {
        parameter.name: param_json_schema(parameter) for parameter in params
    }
    required: List[str] = [
        parameter.name for parameter in params if parameter.required
    ]
    schema: Dict[str, Any] = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "additionalProperties": False,
        "properties": properties,
        "required": required,
    }
    return schema


@jaxtyped(typechecker=beartype)
def coerce_param_value(  # noqa: DOC503, PLR0912 -- lazy error type is indirect.
    param: AutomatonParam,
    value: Any,
) -> Any:
    """Convert one input value to a declared parameter type.

    The function accepts JSON values and command-line strings. It decodes JSON
    objects and arrays from text before it validates the resulting type.

    :see: :class:`~.test_parameters.TestCoerceParamValue`

    Implementation Logic
    --------------------
    1. **Decode structured command-line text**::

           decoded = json.loads(value)

       The operation gives object and array flags the same semantics as JSON.

    2. **Convert scalar command-line text**::

           coerced = int(value)

       The conversion applies only to the parameter's declared scalar type.

    Parameters
    ----------
    param : AutomatonParam
        Static declaration that selects the target Python type.
    value : Any
        JSON value or raw command-line text.

    Returns
    -------
    coerced : Any
        Value with the declared Python type.

    Raises
    ------
    AutomatonError
        If the value cannot convert to the declared type.
    """
    exc: Exception
    try:
        coerced: Any
        if param.python_type is str:
            if not isinstance(value, str):
                raise TypeError("expected a string")
            coerced = value
        elif param.python_type is bool:
            if isinstance(value, bool):
                coerced = value
            elif isinstance(value, str) and value.lower() in {
                "true",
                "1",
                "yes",
                "on",
            }:
                coerced = True
            elif isinstance(value, str) and value.lower() in {
                "false",
                "0",
                "no",
                "off",
            }:
                coerced = False
            else:
                raise TypeError("expected a boolean")
        elif param.python_type is int:
            if isinstance(value, bool):
                raise TypeError("expected an integer")
            coerced = int(value)
            if isinstance(value, float) and not value.is_integer():
                raise TypeError("expected an integer")
        elif param.python_type is float:
            if isinstance(value, bool):
                raise TypeError("expected a number")
            coerced = float(value)
        elif param.python_type in (dict, list):
            decoded: Any = (
                json.loads(value) if isinstance(value, str) else value
            )
            if param.python_type is dict and not isinstance(decoded, dict):
                raise TypeError("expected a JSON object")
            if param.python_type is list and not isinstance(
                decoded, (list, tuple)
            ):
                raise TypeError("expected a JSON array")
            coerced = (
                dict(decoded) if param.python_type is dict else list(decoded)
            )
        else:
            raise TypeError("unsupported parameter type")
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        message: str = f"{param.name} has an invalid value: {exc}"
        raise _error(message, "InvalidInput", param.name) from exc
    return coerced


@jaxtyped(typechecker=beartype)
def validate_param_value(  # noqa: DOC503 -- lazy error type is indirect.
    param: AutomatonParam,
    value: Any,
) -> Any:
    """Validate one coerced executable parameter value.

    The function applies declared choices and inclusive numeric bounds. It
    reports unsupported choices separately from out-of-range numeric values.

    :see: :class:`~.test_parameters.TestValidateParamValue`

    Implementation Logic
    --------------------
    1. **Coerce the input**::

           validated = coerce_param_value(param, value)

       The conversion gives document and command-line input one type contract.

    2. **Apply declarative restrictions**::

           if validated not in choices: raise unsupported

       The checks preserve stable categories for callers and result payloads.

    Parameters
    ----------
    param : AutomatonParam
        Static declaration with choices and bounds.
    value : Any
        Raw JSON value or command-line text.

    Returns
    -------
    validated : Any
        Coerced value that satisfies all declared restrictions.

    Raises
    ------
    AutomatonError
        If coercion, a choice check, or a bounds check fails.
    """
    validated: Any = coerce_param_value(param, value)
    comparison: Any = _comparison_value(validated)
    if param.choices is not None and comparison not in param.choices:
        message: str = f"{param.name} must match one declared choice"
        raise _error(message, "Unsupported", param.name)
    if param.bounds is not None:
        if not isinstance(validated, (int, float)) or isinstance(
            validated, bool
        ):
            message = f"{param.name} bounds require a numeric value"
            raise _error(message, "InvalidInput", param.name)
        lower: float | None
        upper: float | None
        lower, upper = param.bounds
        if lower is not None and validated < lower:
            message = f"{param.name} must be at least {lower}"
            raise _error(message, "ParamOutOfRange", param.name)
        if upper is not None and validated > upper:
            message = f"{param.name} must be at most {upper}"
            raise _error(message, "ParamOutOfRange", param.name)
    return validated


@jaxtyped(typechecker=beartype)
def read_params_document(  # noqa: DOC503 -- lazy error type is indirect.
    source: str,
) -> Dict[str, Any]:
    """Read a JSON parameter object from a supported source.

    The function accepts standard input, a readable path, or inline JSON text.
    It rejects documents whose root value is not a JSON object.

    :see: :class:`~.test_parameters.TestReadParamsDocument`

    Notes
    -----
    Reads text only. Unknown fields remain visible for ``merge_params`` to
    classify as unsupported executable input.

    Parameters
    ----------
    source : str
        ``"-"``, a filesystem path, or an inline JSON object.

    Returns
    -------
    document : Dict[str, Any]
        Parsed JSON parameter object.

    Raises
    ------
    AutomatonError
        If the source cannot provide a JSON object.
    """
    exc: Exception
    try:
        text: str
        stripped_source: str = source.strip()
        if source == "-":
            text = sys.stdin.read()
        elif stripped_source.startswith("{"):
            text = source
        else:
            text = Path(source).read_text(encoding="utf-8")
        parsed: Any = json.loads(text)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        message: str = f"cannot read parameter document: {exc}"
        raise _error(message, "InvalidInput", "params") from exc
    if not isinstance(parsed, dict):
        message = "parameter document must contain a JSON object"
        raise _error(message, "InvalidInput", "params")
    document: Dict[str, Any] = dict(parsed)
    return document


@jaxtyped(typechecker=beartype)
def merge_params(  # noqa: DOC503 -- lazy error type is indirect.
    params: Sequence[AutomatonParam],
    document: Mapping[str, Any],
    cli_overrides: Mapping[str, Any],
) -> Dict[str, Any]:
    """Apply defaults, document values, and command-line overrides.

    The function applies precedence in that order. It rejects unknown fields,
    validates every declared value, and rejects missing required parameters.

    :see: :class:`~.test_parameters.TestMergeParams`

    Implementation Logic
    --------------------
    1. **Reject unknown fields**::

           unknown = set(document) - declared_names

       The check prevents ignored caller input from hiding an error.

    2. **Apply precedence and validation**::

           raw_value = cli_overrides.get(param.name, document.get(param.name))

       The final coercion applies one validation path to every input source.

    Parameters
    ----------
    params : Sequence[AutomatonParam]
        Ordered executable parameter declarations.
    document : Mapping[str, Any]
        JSON object from the optional parameter document.
    cli_overrides : Mapping[str, Any]
        Explicit command-line values keyed by parameter name.

    Returns
    -------
    merged : Dict[str, Any]
        Fully validated parameter values.

    Raises
    ------
    AutomatonError
        If a field is unknown, required input is missing, or validation fails.
    """
    parameter: AutomatonParam
    declared_names: set[str] = {parameter.name for parameter in params}
    source_name: str
    for source_name in tuple(document) + tuple(cli_overrides):
        if source_name not in declared_names:
            message: str = f"unknown parameter: {source_name}"
            raise _error(message, "Unsupported", source_name)
    merged: Dict[str, Any] = {}
    for parameter in params:
        raw_value: Any
        if parameter.name in cli_overrides:
            raw_value = cli_overrides[parameter.name]
        elif parameter.name in document:
            raw_value = document[parameter.name]
        elif parameter.required:
            message = f"missing required parameter: {parameter.name}"
            raise _error(message, "InvalidInput", parameter.name)
        else:
            raw_value = parameter.default
        merged[parameter.name] = validate_param_value(parameter, raw_value)
    return merged


__all__: list[str] = [
    "coerce_param_value",
    "describe_param",
    "merge_params",
    "param_json_schema",
    "params_json_schema",
    "read_params_document",
    "validate_param_value",
]
