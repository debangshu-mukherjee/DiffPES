"""Decode portable forward-model certificate documents.

Extended Summary
----------------
This private module validates canonical JSON documents, restores numerical
leaves, and reconstructs certification carriers through types-owned
validation factories.
"""

from __future__ import annotations

import base64
import binascii
import json
import math
from collections.abc import Callable, Mapping
from dataclasses import fields

import jax.numpy as jnp
import numpy as np
from beartype.typing import Any, Dict, List, Tuple
from jaxtyping import Bool, Num
from numpy.typing import NDArray

from diffpes.constants import (
    CERTIFICATE_ARRAY_KINDS,
    CERTIFICATE_DOCUMENT_KEYS,
    CERTIFICATE_FORMAT,
    CERTIFICATE_SCHEMA_MINOR,
)
from diffpes.types import ForwardCertificate

from .certificate import (
    _document_identity,
    _module_factories,
    _module_types,
    _normalize_json_value,
    _parse_schema_version,
    _reject_json_constant,
    _storage_checksum,
    _unique_object,
)


def _read_document(data: bytes) -> Dict[str, Any]:
    """PRIVATE: Parse, structurally validate, and checksum one JSON
    document.

    Implementation Logic
    --------------------
    Decodes UTF-8 JSON with the duplicate-key and constant-rejecting
    hooks.  The document must be an object with every required key,
    the supported ``format`` value, and a valid schema version.
    Unknown top-level keys fail for a minor version at or below the
    reader's version.  The stored consistency checksum must match a
    recomputation, and ``extensions`` must be an object.  The result
    passes through :func:`_normalize_json_value` once.

    Parameters
    ----------
    data : bytes
        Raw JSON document bytes.

    Returns
    -------
    normalized : Dict[str, Any]
        Validated, normalized document mapping.

    Raises
    ------
    ValueError
        If decoding, structure, format, schema, unknown-key, checksum,
        or extension validation fails.
    """
    exc: UnicodeDecodeError | json.JSONDecodeError
    try:
        decoded: Any = json.loads(
            data.decode("utf-8"),
            object_pairs_hook=_unique_object,
            parse_constant=_reject_json_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        msg: str = "certificate is not valid UTF-8 JSON"
        raise ValueError(msg) from exc
    if not isinstance(decoded, dict):
        msg: str = "certificate document must be a JSON object"
        raise ValueError(msg)
    missing: frozenset[str] = CERTIFICATE_DOCUMENT_KEYS - decoded.keys()
    if missing:
        msg: str = f"certificate document is missing fields: {sorted(missing)}"
        raise ValueError(msg)
    if decoded["format"] != CERTIFICATE_FORMAT:
        msg: str = f"unsupported certificate format: {decoded['format']!r}"
        raise ValueError(msg)
    parsed_schema: Tuple[int, int] = _parse_schema_version(
        decoded["schema_version"]
    )
    minor: int = parsed_schema[1]
    extra: frozenset[str] = decoded.keys() - CERTIFICATE_DOCUMENT_KEYS
    if extra and minor <= CERTIFICATE_SCHEMA_MINOR:
        msg: str = (
            f"unknown current-schema certificate fields: {sorted(extra)}"
        )
        raise ValueError(msg)
    expected_checksum: str = _storage_checksum(decoded)
    if decoded["consistency_checksum"] != expected_checksum:
        msg: str = "certificate consistency checksum mismatch"
        raise ValueError(msg)
    extensions: Any = decoded["extensions"]
    if not isinstance(extensions, dict):
        msg: str = "certificate extensions must be a JSON object"
        raise ValueError(msg)
    normalized: Any = _normalize_json_value(decoded)
    return normalized


def _decode_array(node: Mapping[str, Any]) -> Any:
    """PRIVATE: Decode and validate one losslessly represented numerical
    leaf.

    Implementation Logic
    --------------------
    Requires the exact seven-field record shape, little-endian C-order
    base64 storage, a supported non-big-endian dtype, and a shape list
    of nonnegative integers.  Decodes the base64 payload with strict
    validation, checks the byte length against dtype and shape, and
    rejects nonfinite float or complex data.  The buffer copy becomes
    a JAX array.

    Parameters
    ----------
    node : Mapping[str, Any]
        Array record produced by :func:`_encode_array`.

    Returns
    -------
    result : Any
        Reconstructed JAX array with the recorded dtype and shape.

    Raises
    ------
    ValueError
        If any field, dtype, shape, encoding, byte length, or
        finiteness check fails.
    """
    exc: TypeError | binascii.Error | ValueError
    required: frozenset[str] = frozenset(
        {"kind", "dtype", "shape", "byte_order", "order", "encoding", "data"}
    )
    if node.keys() != required:
        msg: str = "array record has missing or unknown fields"
        raise ValueError(msg)
    if node["byte_order"] != "little" or node["order"] != "C":
        msg: str = "certificate arrays require little-endian C-order storage"
        raise ValueError(msg)
    if node["encoding"] != "base64":
        msg: str = "unsupported certificate array encoding"
        raise ValueError(msg)
    try:
        dtype: np.dtype[Any] = np.dtype(node["dtype"])
    except TypeError as exc:
        msg: str = "invalid certificate array dtype"
        raise ValueError(msg) from exc
    if dtype.kind not in CERTIFICATE_ARRAY_KINDS or dtype.byteorder == ">":
        msg: str = f"unsupported certificate array dtype: {dtype}"
        raise ValueError(msg)
    shape_value: Any = node["shape"]
    if not isinstance(shape_value, list) or any(
        not isinstance(item, int) or isinstance(item, bool) or item < 0
        for item in shape_value
    ):
        msg: str = "certificate array shape must contain nonnegative integers"
        raise ValueError(msg)
    shape: Tuple[int, ...] = tuple(shape_value)
    data_value: Any = node["data"]
    if not isinstance(data_value, str):
        msg: str = "certificate array data must be base64 text"
        raise ValueError(msg)
    try:
        payload: bytes = base64.b64decode(data_value, validate=True)
    except (binascii.Error, ValueError) as exc:
        msg: str = "certificate array contains invalid base64 data"
        raise ValueError(msg) from exc
    count: int = math.prod(shape)
    expected_bytes: int = count * dtype.itemsize
    if len(payload) != expected_bytes:
        msg: str = (
            "certificate array byte length does not match dtype and shape"
        )
        raise ValueError(msg)
    array: Bool[NDArray, "..."] | Num[NDArray, "..."] = np.frombuffer(
        payload, dtype=dtype
    ).reshape(shape)
    if dtype.kind in {"f", "c"} and not bool(np.all(np.isfinite(array))):
        msg: str = "certificate persistence rejects nonfinite numerical leaves"
        raise ValueError(msg)
    result: Any = jnp.asarray(array.copy())
    return result


def _decode_value(
    node: Any,
    *,
    schema_minor: int,
    extensions: Dict[str, Any],
    path: str,
) -> Any:
    """PRIVATE: Decode one schema node through the registered carrier
    factories.

    Parameters
    ----------
    node : Any
        Encoded scalar or ``kind``-tagged record.
    schema_minor : int
        Minor schema version of the document.
    extensions : Dict[str, Any]
        Mutable extension mapping that collects unknown fields.
    path : str
        Dotted location for diagnostics.

    Returns
    -------
    decoded : Any
        Reconstructed scalar, array, tuple, list, mapping, or carrier.

    Raises
    ------
    ValueError
        If a node is not a scalar or object, a record shape is
        invalid, or the ``kind`` tag is unknown.

    Notes
    -----
    Scalars pass through.  Objects dispatch on their ``kind`` tag.
    ``array`` records decode losslessly.  ``tuple`` and ``list``
    records recurse with indexed paths.  ``mapping`` records recurse
    with dotted paths.  ``module`` records go to
    :func:`_decode_module`.
    """
    if node is None or isinstance(node, bool | int | float | str):
        decoded: Any = node
        return decoded  # noqa: RET504
    if not isinstance(node, dict):
        msg: str = f"invalid certificate node at {path}"
        raise ValueError(msg)
    kind: Any = node.get("kind")
    if kind == "array":
        decoded = _decode_array(node)
        return decoded  # noqa: RET504
    if kind in {"tuple", "list"}:
        if node.keys() != {"kind", "items"} or not isinstance(
            node["items"], list
        ):
            msg: str = f"invalid {kind} record at {path}"
            raise ValueError(msg)
        values: List[Any] = [
            _decode_value(
                item,
                schema_minor=schema_minor,
                extensions=extensions,
                path=f"{path}[{index}]",
            )
            for index, item in enumerate(node["items"])
        ]
        decoded = tuple(values) if kind == "tuple" else values
        return decoded  # noqa: RET504
    if kind == "mapping":
        if node.keys() != {"kind", "items"} or not isinstance(
            node["items"], dict
        ):
            msg: str = f"invalid mapping record at {path}"
            raise ValueError(msg)
        decoded = {
            key: _decode_value(
                item,
                schema_minor=schema_minor,
                extensions=extensions,
                path=f"{path}.{key}",
            )
            for key, item in node["items"].items()
        }
        return decoded  # noqa: RET504
    if kind != "module":
        msg: str = f"unknown certificate node kind at {path}: {kind!r}"
        raise ValueError(msg)
    decoded = _decode_module(
        node,
        schema_minor=schema_minor,
        extensions=extensions,
        path=path,
    )
    return decoded  # noqa: RET504


def _record_unknown_fields(
    extensions: Dict[str, Any],
    path: str,
    values: Dict[str, Any],
) -> None:
    """PRIVATE: Retain fields introduced by a newer compatible minor schema.

    Parameters
    ----------
    extensions : Dict[str, Any]
        Mutable extension mapping for the certificate under
        reconstruction.
    path : str
        Dotted module location that owns the unknown fields.
    values : Dict[str, Any]
        Encoded unknown fields at that location.

    Raises
    ------
    ValueError
        If the reserved extension key already holds a non-object.

    Notes
    -----
    Stores the values under the reserved
    ``org.diffpes.persistence.unknown_module_fields`` key, so a
    round trip through an older reader preserves newer-minor data.
    """
    key: str = "org.diffpes.persistence.unknown_module_fields"
    existing: Any = extensions.setdefault(key, {})
    if not isinstance(existing, dict):
        msg: str = f"reserved extension key {key!r} must contain an object"
        raise ValueError(msg)
    existing[path] = values


def _decode_module(
    node: Mapping[str, Any],
    *,
    schema_minor: int,
    extensions: Dict[str, Any],
    path: str,
) -> Any:
    """PRIVATE: Decode one whitelisted Equinox carrier via its validation
    factory.

    Implementation Logic
    --------------------
    Requires the exact ``kind``, ``type``, ``fields`` record shape and
    a whitelisted type name.  Missing declared fields fail.  Unknown
    fields fail for a minor version at or below the reader's version;
    a newer minor retains them through
    :func:`_record_unknown_fields`.  Expected fields decode
    recursively.  A ``ForwardCertificate`` root also receives its
    ``extensions_json`` string, re-serialized from the collected
    extensions.  The types-owned factory rebuilds the carrier, so the
    load repeats the full validation contract.

    Parameters
    ----------
    node : Mapping[str, Any]
        Encoded ``module`` record.
    schema_minor : int
        Minor schema version of the document.
    extensions : Dict[str, Any]
        Mutable extension mapping that collects unknown fields.
    path : str
        Dotted location for diagnostics.

    Returns
    -------
    result : Any
        Validated carrier instance.

    Raises
    ------
    ValueError
        If the record shape, type name, or field inventory is invalid,
        or the factory rejects the decoded values.
    """
    exc: TypeError | ValueError
    if node.keys() != {"kind", "type", "fields"}:
        msg: str = f"invalid module record at {path}"
        raise ValueError(msg)
    type_name: Any = node["type"]
    encoded_fields: Any = node["fields"]
    module_types: Dict[str, type[Any]] = _module_types()
    if not isinstance(type_name, str) or type_name not in module_types:
        msg: str = (
            f"unsupported certificate module type at {path}: {type_name!r}"
        )
        raise ValueError(msg)
    if not isinstance(encoded_fields, dict):
        msg: str = f"module fields must be an object at {path}"
        raise ValueError(msg)
    module_type: type[Any] = module_types[type_name]
    expected_names: set[str] = {field.name for field in fields(module_type)}
    if module_type is ForwardCertificate:
        expected_encoded: set[str] = expected_names - {"extensions_json"}
    else:
        expected_encoded = expected_names
    provided_names: set[str] = set(encoded_fields)
    missing: set[str] = expected_encoded - provided_names
    if missing:
        msg: str = f"module {type_name} is missing fields: {sorted(missing)}"
        raise ValueError(msg)
    unknown: set[str] = provided_names - expected_encoded
    if unknown and schema_minor <= CERTIFICATE_SCHEMA_MINOR:
        msg: str = f"module {type_name} has unknown fields: {sorted(unknown)}"
        raise ValueError(msg)
    if unknown:
        _record_unknown_fields(
            extensions,
            path,
            {name: encoded_fields[name] for name in sorted(unknown)},
        )
    values: Dict[str, Any] = {
        name: _decode_value(
            encoded_fields[name],
            schema_minor=schema_minor,
            extensions=extensions,
            path=f"{path}.{name}",
        )
        for name in expected_encoded
    }
    if module_type is ForwardCertificate:
        values["extensions_json"] = json.dumps(
            _normalize_json_value(extensions),
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    factory: Callable[..., Any] = _module_factories()[module_type]
    try:
        result: Any = factory(**values)
    except (TypeError, ValueError) as exc:
        msg: str = f"invalid {type_name} data at {path}: {exc}"
        raise ValueError(msg) from exc
    return result


def _certificate_from_document(document: Dict[str, Any]) -> ForwardCertificate:
    """PRIVATE: Construct a validated certificate from a parsed document.

    Implementation Logic
    --------------------
    Reads the stored canonical identity from the certificate node.
    For a minor version at or below the reader's version, the identity
    must match a recomputation.  Extra top-level
    document fields go into the extensions under a reserved key.  The
    node then decodes through :func:`_decode_value`; the root must be
    a ``ForwardCertificate`` whose manifest schema version equals the
    document's.

    Parameters
    ----------
    document : Dict[str, Any]
        Validated document from :func:`_read_document`.

    Returns
    -------
    decoded : ForwardCertificate
        Reconstructed and revalidated certificate.

    Raises
    ------
    ValueError
        If the identity is absent or mismatched, decoding fails, the
        root is not a certificate, or the schema versions disagree.
    """
    parsed_schema: Tuple[int, int] = _parse_schema_version(
        document["schema_version"]
    )
    minor: int = parsed_schema[1]
    certificate_node: Any = document["certificate"]
    exc: KeyError | TypeError
    try:
        stored_identity: Any = certificate_node["fields"][
            "certificate_checksum"
        ]
    except (KeyError, TypeError) as exc:
        msg: str = "certificate has no canonical identity"
        raise ValueError(msg) from exc
    expected_identity: str = _document_identity(document)
    if (
        minor <= CERTIFICATE_SCHEMA_MINOR
        and stored_identity != expected_identity
    ):
        msg = "certificate canonical identity mismatch"
        raise ValueError(msg)
    extensions: Dict[str, Any] = dict(document["extensions"])
    extra: Dict[str, Any] = {
        key: value
        for key, value in document.items()
        if key not in CERTIFICATE_DOCUMENT_KEYS
    }
    if extra:
        extensions["org.diffpes.persistence.unknown_document_fields"] = extra
    decoded: Any = _decode_value(
        document["certificate"],
        schema_minor=minor,
        extensions=extensions,
        path="certificate",
    )
    if not isinstance(decoded, ForwardCertificate):
        msg: str = "certificate document root is not a ForwardCertificate"
        raise ValueError(msg)
    if decoded.manifest.schema_version != document["schema_version"]:
        msg: str = "document and manifest schema versions disagree"
        raise ValueError(msg)
    return decoded


__all__: list[str] = []
