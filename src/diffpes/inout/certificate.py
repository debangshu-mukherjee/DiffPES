"""Persist forward-model certificates in portable formats.

Extended Summary
----------------
The module stores ``ForwardCertificate`` PyTrees as deterministic, transparent
JSON. It embeds the exact JSON bytes in HDF5 result files. The format
represents numerical leaves without loss. It records the dtype, shape, byte
order, and base64-encoded canonical bytes. A CRC32 checksum detects accidental
storage mismatches. The checksum does not provide authentication or support a
scientific certification claim.

Routine Listings
----------------
:func:`attach_certificate_h5`
    Attach a certificate atomically to an HDF5 result file.
:func:`certificate_identity`
    Compute the scientific identity of a canonical certificate.
:func:`finalize_certificate`
    Replace the kernel placeholder with the canonical identity.
:func:`load_certificate_h5`
    Load a certificate embedded in an HDF5 result file.
:func:`load_certificate_json`
    Load a validated forward certificate from canonical JSON.
:func:`save_certificate_json`
    Save a forward certificate atomically as canonical JSON.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import json
import math
import os
import re
import shutil
import tempfile
import unicodedata
import zlib
from collections.abc import Callable, Mapping
from dataclasses import fields, is_dataclass
from pathlib import Path

import h5py
import jax.numpy as jnp
import numpy as np
from beartype import beartype
from beartype.typing import Any
from jaxtyping import Shaped, UInt, jaxtyped
from numpy.typing import NDArray

from diffpes.types import (
    CERTIFICATE_ARRAY_KINDS,
    CERTIFICATE_DOCUMENT_KEYS,
    CERTIFICATE_FORMAT,
    CERTIFICATE_H5_GROUP,
    CERTIFICATE_SCHEMA_MAJOR,
    CERTIFICATE_SCHEMA_MINOR,
    CERTIFICATE_SCHEMA_PATTERN,
    ArtifactRef,
    CertificationClaim,
    ConventionRef,
    DependencyMap,
    DerivativeEvidence,
    DomainPredicate,
    DomainResult,
    EvidenceLineage,
    EvidenceRef,
    ExecutionManifest,
    ForwardCertificate,
    ForwardModelSpec,
    HumanAttestationRef,
    InformationSpectrum,
    PolicyReport,
    SensitivityMap,
    TransformationRecord,
    WaiverRecord,
    make_artifact_ref,
    make_certification_claim,
    make_convention_ref,
    make_dependency_map,
    make_derivative_evidence,
    make_domain_predicate,
    make_domain_result,
    make_evidence_lineage,
    make_evidence_ref,
    make_execution_manifest,
    make_forward_certificate,
    make_forward_model_spec,
    make_human_attestation_ref,
    make_information_spectrum,
    make_policy_report,
    make_sensitivity_map,
    make_transformation_record,
    make_waiver_record,
)


def _module_factories() -> dict[type[Any], Callable[..., Any]]:
    """Return types-owned carrier factories supported by the codec."""
    factories: dict[type[Any], Callable[..., Any]] = {
        ArtifactRef: make_artifact_ref,
        CertificationClaim: make_certification_claim,
        ConventionRef: make_convention_ref,
        DependencyMap: make_dependency_map,
        DerivativeEvidence: make_derivative_evidence,
        DomainPredicate: make_domain_predicate,
        DomainResult: make_domain_result,
        EvidenceLineage: make_evidence_lineage,
        EvidenceRef: make_evidence_ref,
        ExecutionManifest: make_execution_manifest,
        ForwardCertificate: make_forward_certificate,
        ForwardModelSpec: make_forward_model_spec,
        HumanAttestationRef: make_human_attestation_ref,
        InformationSpectrum: make_information_spectrum,
        PolicyReport: make_policy_report,
        SensitivityMap: make_sensitivity_map,
        TransformationRecord: make_transformation_record,
        WaiverRecord: make_waiver_record,
    }
    return factories


def _module_types() -> dict[str, type[Any]]:
    """Return persisted carrier names mapped to their concrete types."""
    module_types: dict[str, type[Any]] = {
        module_type.__name__: module_type
        for module_type in _module_factories()
    }
    return module_types


def _normalize_text(value: str) -> str:
    """Return NFC-normalized certificate text."""
    normalized: str = unicodedata.normalize("NFC", value)
    return normalized


def _normalize_json_value(value: Any) -> Any:
    """Normalize an extension value and reject non-JSON/nonfinite data."""
    key: Any
    item: Any
    if value is None or isinstance(value, bool | int):
        normalized_value: Any = value
        return normalized_value  # noqa: RET504
    if isinstance(value, float):
        if not math.isfinite(value):
            msg: str = "certificate JSON rejects NaN and infinite values"
            raise ValueError(msg)
        normalized_value = value
        return normalized_value  # noqa: RET504
    if isinstance(value, str):
        normalized_value = _normalize_text(value)
        return normalized_value  # noqa: RET504
    if isinstance(value, list):
        normalized_value = [_normalize_json_value(item) for item in value]
        return normalized_value  # noqa: RET504
    if isinstance(value, Mapping):
        normalized: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                msg: str = "certificate JSON object keys must be strings"
                raise ValueError(msg)
            normalized_key: str = _normalize_text(key)
            if normalized_key in normalized:
                msg: str = (
                    "certificate keys collide after Unicode normalization"
                )
                raise ValueError(msg)
            normalized[normalized_key] = _normalize_json_value(item)
        normalized_value = normalized
        return normalized_value  # noqa: RET504
    msg: str = f"unsupported certificate JSON value: {type(value)!r}"
    raise ValueError(msg)


def _json_bytes(value: Mapping[str, Any], *, newline: bool) -> bytes:
    """Encode a normalized mapping as deterministic UTF-8 JSON."""
    normalized: Any = _normalize_json_value(value)
    encoded: bytes = json.dumps(
        normalized,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    if newline:
        encoded += b"\n"
    return encoded  # noqa: RET504


def _storage_checksum(document: Mapping[str, Any]) -> str:
    """Return the non-security CRC32 of a document without its checksum."""
    payload: dict[str, Any] = dict(document)
    payload.pop("consistency_checksum", None)
    value: int = zlib.crc32(_json_bytes(payload, newline=False))
    checksum: str = f"crc32:certificate-json-v1:{value & 0xFFFFFFFF:08x}"
    return checksum


def _identity_payload(document: Mapping[str, Any]) -> dict[str, Any]:
    """Return canonical scientific fields without audit-only identities."""
    payload: dict[str, Any] = {
        "format": document["format"],
        "schema_version": document["schema_version"],
        "certificate": json.loads(json.dumps(document["certificate"])),
        "extensions": document["extensions"],
    }
    fields_node: dict[str, Any] = payload["certificate"]["fields"]
    fields_node.pop("certificate_checksum", None)
    manifest_fields: dict[str, Any] = fields_node["manifest"]["fields"]
    manifest_fields.pop("execution_id", None)
    manifest_fields.pop("started_at_utc", None)
    return payload


def _document_identity(document: Mapping[str, Any]) -> str:
    """Return the domain-separated SHA-256 certificate identity."""
    payload: bytes = _json_bytes(_identity_payload(document), newline=False)
    digest: Any = hashlib.sha256()
    digest.update(
        b"DIFFPES-SCIENTIFIC-IDENTITY\x00"
        b"sha256\x00"
        b"1\x00"
        b"org.diffpes.identity.certificate.v1\x00"
    )
    digest.update(payload)
    identity: str = f"sha256:1:certificate:{digest.hexdigest()}"
    return identity


def _encode_array(value: object) -> dict[str, Any]:
    """Encode one concrete numerical leaf without decimal conversion."""
    exc: Exception
    try:
        array: Shaped[NDArray, "..."] = np.asarray(value)
    except Exception as exc:
        msg: str = (
            "certificate persistence requires concrete, non-traced arrays"
        )
        raise ValueError(msg) from exc
    if array.dtype.kind not in CERTIFICATE_ARRAY_KINDS:
        msg: str = f"unsupported certificate array dtype: {array.dtype}"
        raise ValueError(msg)
    if array.dtype.kind in {"f", "c"} and not bool(np.all(np.isfinite(array))):
        msg: str = "certificate persistence rejects nonfinite numerical leaves"
        raise ValueError(msg)
    canonical_dtype: np.dtype[Any] = array.dtype.newbyteorder("<")
    canonical: Shaped[NDArray, "..."] = np.asarray(
        array,
        dtype=canonical_dtype,
        order="C",
    )
    payload: str = base64.b64encode(canonical.tobytes(order="C")).decode(
        "ascii"
    )
    result: dict[str, Any] = {
        "kind": "array",
        "dtype": canonical.dtype.str,
        "shape": list(canonical.shape),
        "byte_order": "little",
        "order": "C",
        "encoding": "base64",
        "data": payload,
    }
    return result


def _is_array(value: object) -> bool:
    """Return whether a value exposes a concrete numerical array protocol."""
    if isinstance(value, np.ndarray | np.generic):
        is_array: bool = True
        return is_array
    attributes: tuple[str, ...] = ("__array__", "dtype", "shape")
    is_array: bool = all(hasattr(value, attr) for attr in attributes)
    return is_array


def _encode_value(  # noqa: PLR0911
    value: object,
    *,
    root: bool = False,
) -> Any:
    """Encode one supported carrier field into the transparent schema."""
    field: Any
    if value is None or isinstance(value, bool | int):
        encoded: Any = value
        return encoded  # noqa: RET504
    if isinstance(value, float):
        if not math.isfinite(value):
            msg: str = "certificate persistence rejects nonfinite scalars"
            raise ValueError(msg)
        encoded = value
        return encoded  # noqa: RET504
    if isinstance(value, str):
        encoded = _normalize_text(value)
        return encoded  # noqa: RET504
    if _is_array(value):
        encoded = _encode_array(value)
        return encoded  # noqa: RET504
    if isinstance(value, tuple):
        encoded = {
            "kind": "tuple",
            "items": [_encode_value(item) for item in value],
        }
        return encoded  # noqa: RET504
    if isinstance(value, list):
        encoded = {
            "kind": "list",
            "items": [_encode_value(item) for item in value],
        }
        return encoded  # noqa: RET504
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            msg: str = "certificate mappings require string keys"
            raise ValueError(msg)
        encoded = {
            "kind": "mapping",
            "items": {key: _encode_value(item) for key, item in value.items()},
        }
        return encoded  # noqa: RET504
    value_type: type[Any] = type(value)
    if value_type in _module_factories() and is_dataclass(value):
        encoded_fields: dict[str, Any] = {}
        for field in fields(value):
            if root and field.name == "extensions_json":
                continue
            encoded_fields[field.name] = _encode_value(
                getattr(value, field.name)
            )
        encoded = {
            "kind": "module",
            "type": value_type.__name__,
            "fields": encoded_fields,
        }
        return encoded  # noqa: RET504
    msg: str = f"unsupported certificate field value: {type(value)!r}"
    raise ValueError(msg)


def _parse_extensions(certificate: ForwardCertificate) -> dict[str, Any]:
    """Parse and normalize the certificate extension object."""
    exc: json.JSONDecodeError | TypeError
    try:
        value: Any = json.loads(
            certificate.extensions_json,
            object_pairs_hook=_unique_object,
            parse_constant=_reject_json_constant,
        )
    except (json.JSONDecodeError, TypeError) as exc:
        msg: str = "certificate extensions_json must encode a JSON object"
        raise ValueError(msg) from exc
    if not isinstance(value, dict):
        msg: str = "certificate extensions_json must encode a JSON object"
        raise ValueError(msg)
    normalized: Any = _normalize_json_value(value)
    return normalized


@jaxtyped(typechecker=beartype)
def certificate_identity(certificate: ForwardCertificate) -> str:
    """Compute the scientific identity of a canonical certificate.

    The identity covers scientific and numerical fields. It excludes the
    self-reference, audit execution ID, and wall-clock timestamp. The SHA-256
    digest provides content addressing and does not authenticate the record.

    :see: :class:`~.test_certificate.TestCertificateIdentity`

    Implementation Logic
    --------------------
    1. **Compute the canonical identity**::

           identity = _document_identity(document)

       The canonical payload omits only the self-reference and audit fields.

    Parameters
    ----------
    certificate : ForwardCertificate
        Concrete certificate at the persistence boundary.

    Returns
    -------
    identity : str
        Stable non-security identity for the scientific execution record.
    """
    schema_version: str = certificate.manifest.schema_version
    _parse_schema_version(schema_version)
    document: dict[str, Any] = {
        "format": CERTIFICATE_FORMAT,
        "schema_version": schema_version,
        "certificate": _encode_value(certificate, root=True),
        "extensions": _parse_extensions(certificate),
    }
    identity: str = _document_identity(document)
    return identity


@jaxtyped(typechecker=beartype)
def finalize_certificate(
    certificate: ForwardCertificate,
) -> ForwardCertificate:
    """Replace the kernel placeholder with the canonical identity.

    Canonical encoding stays outside JAX tracing because it has no scientific
    derivative. The returned certificate retains every scientific leaf.

    :see: :class:`~.test_certificate.TestFinalizeCertificate`

    Implementation Logic
    --------------------
    1. **Replace the placeholder**::

           identity = certificate_identity(certificate)

       The factory copies every other certificate field without modification.

    Parameters
    ----------
    certificate : ForwardCertificate
        Concrete certificate produced by the JAX-native execution kernel.

    Returns
    -------
    result : ForwardCertificate
        Equivalent certificate with its final canonical identity.
    """
    identity: str = certificate_identity(certificate)
    result: ForwardCertificate = make_forward_certificate(
        manifest=certificate.manifest,
        model=certificate.model,
        artifacts=certificate.artifacts,
        transformations=certificate.transformations,
        evidence=certificate.evidence,
        claims=certificate.claims,
        domains=certificate.domains,
        derivatives=certificate.derivatives,
        dependencies=certificate.dependencies,
        sensitivities=certificate.sensitivities,
        information=certificate.information,
        policy_report=certificate.policy_report,
        policy_id=certificate.policy_id,
        certificate_checksum=identity,
        extensions_json=certificate.extensions_json,
        waivers=certificate.waivers,
        attestations=certificate.attestations,
    )
    return result


def _certificate_document(certificate: ForwardCertificate) -> dict[str, Any]:
    """Build the complete portable document for one certificate."""
    finalized: ForwardCertificate = finalize_certificate(certificate)
    schema_version: str = finalized.manifest.schema_version
    _parse_schema_version(schema_version)
    document: dict[str, Any] = {
        "format": CERTIFICATE_FORMAT,
        "schema_version": schema_version,
        "certificate": _encode_value(finalized, root=True),
        "extensions": _parse_extensions(finalized),
    }
    document["consistency_checksum"] = _storage_checksum(document)
    return document


def _reject_json_constant(value: str) -> None:
    """Reject JSON's non-standard NaN and Infinity tokens."""
    msg: str = f"certificate JSON contains invalid constant {value!r}"
    raise ValueError(msg)


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    """Build a JSON object while rejecting duplicate names."""
    key: Any
    value: Any
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            msg: str = f"duplicate certificate JSON key: {key!r}"
            raise ValueError(msg)
        result[key] = value
    return result


def _read_document(data: bytes) -> dict[str, Any]:
    """Parse, structurally validate, and checksum one JSON document."""
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
    parsed_schema: tuple[int, int] = _parse_schema_version(
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


def _parse_schema_version(value: object) -> tuple[int, int]:
    """Parse a schema version and reject unsupported major versions."""
    if not isinstance(value, str):
        msg: str = "certificate schema_version must be a string"
        raise ValueError(msg)
    match: re.Match[str] | None = CERTIFICATE_SCHEMA_PATTERN.fullmatch(value)
    if match is None:
        msg: str = f"invalid certificate schema version: {value!r}"
        raise ValueError(msg)
    major: int = int(match.group("major"))
    minor_text: str | None = match.group("minor")
    minor: int = 0 if minor_text is None else int(minor_text)
    if major != CERTIFICATE_SCHEMA_MAJOR:
        msg: str = (
            f"unsupported certificate schema major {major}; "
            f"reader supports {CERTIFICATE_SCHEMA_MAJOR}.x"
        )
        raise ValueError(msg)
    parsed: tuple[int, int] = (major, minor)
    return parsed


def _decode_array(node: Mapping[str, Any]) -> Any:
    """Decode and validate one losslessly represented numerical leaf."""
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
    shape: tuple[int, ...] = tuple(shape_value)
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
    array: Shaped[NDArray, "..."] = np.frombuffer(
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
    extensions: dict[str, Any],
    path: str,
) -> Any:
    """Decode one schema node through the registered carrier factories."""
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
        values: list[Any] = [
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
    extensions: dict[str, Any],
    path: str,
    values: dict[str, Any],
) -> None:
    """Retain fields introduced by a newer compatible minor schema."""
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
    extensions: dict[str, Any],
    path: str,
) -> Any:
    """Decode one whitelisted Equinox carrier via its validation factory."""
    exc: TypeError | ValueError
    if node.keys() != {"kind", "type", "fields"}:
        msg: str = f"invalid module record at {path}"
        raise ValueError(msg)
    type_name: Any = node["type"]
    encoded_fields: Any = node["fields"]
    module_types: dict[str, type[Any]] = _module_types()
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
    values: dict[str, Any] = {
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


def _certificate_from_document(document: dict[str, Any]) -> ForwardCertificate:
    """Construct a validated certificate from a parsed document."""
    parsed_schema: tuple[int, int] = _parse_schema_version(
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
    extensions: dict[str, Any] = dict(document["extensions"])
    extra: dict[str, Any] = {
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
    return decoded  # noqa: RET504


def _atomic_write(path: Path, data: bytes) -> None:
    """Write bytes through a same-directory temporary and atomic replace."""
    stream: Any
    path.parent.mkdir(parents=False, exist_ok=True)
    temporary_record: tuple[int, str] = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    descriptor: int = temporary_record[0]
    temporary_name: str = temporary_record[1]
    temporary: Path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


@jaxtyped(typechecker=beartype)
def save_certificate_json(
    certificate: ForwardCertificate,
    path: str | Path,
) -> None:
    """Save a forward certificate atomically as canonical JSON.

    The persistence operation retains the complete scientific-assurance record
    and its JAX array leaves. Consistency checks detect accidental storage
    corruption.

    :see: :class:`~.test_certificate.TestSaveCertificateJson`


    Implementation Logic
    --------------------
    1. **Build the certificate document**::

           document = _certificate_document(certificate)
           data = _json_bytes(document, newline=True)

       The document includes the schema and a non-security consistency check.
    2. **Replace the destination atomically**::

           _atomic_write(Path(path), data)

       A same-directory temporary prevents a partial JSON record.

    Parameters
    ----------
    certificate : ForwardCertificate
        Validated scientific-assurance record to persist.
    path : str | Path
        Destination JSON path. Its parent directory must already exist.
    """
    document: dict[str, Any] = _certificate_document(certificate)
    data: bytes = _json_bytes(document, newline=True)
    _atomic_write(Path(path), data)


@jaxtyped(typechecker=beartype)
def load_certificate_json(path: str | Path) -> ForwardCertificate:
    """Load a validated forward certificate from canonical JSON.

    The persistence operation retains the complete scientific-assurance record
    and its JAX array leaves. Consistency checks detect accidental storage
    corruption.

    :see: :class:`~.test_certificate.TestLoadCertificateJson`


    Implementation Logic
    --------------------
    1. **Read and validate the document**::

           data = Path(path).read_bytes()
           document = _read_document(data)

       The decoder checks the schema and consistency checksum before use.
    2. **Reconstruct the carrier**::

           certificate = _certificate_from_document(document)

       The decoder restores persisted numerical leaves as JAX arrays.

    Parameters
    ----------
    path : str | Path
        Source JSON path.

    Returns
    -------
    certificate : ForwardCertificate
        Reconstructed certificate with numerical leaves restored as JAX
        arrays.
    """
    data: bytes = Path(path).read_bytes()
    document: dict[str, Any] = _read_document(data)
    certificate: ForwardCertificate = _certificate_from_document(document)
    return certificate


def _validate_h5_name(name: str) -> None:
    """Reject ambiguous or path-like HDF5 certificate names."""
    if not name or name in {".", ".."} or "/" in name or "\x00" in name:
        msg: str = "HDF5 certificate name must be one nonblank group component"
        raise ValueError(msg)


def _write_h5_record(
    path: Path,
    name: str,
    data: bytes,
    certificate: ForwardCertificate,
) -> None:
    """Write one exact JSON record and its convenience index attributes."""
    file: Any
    document: dict[str, Any] = _read_document(data)
    with h5py.File(path, "a") as file:
        root: h5py.Group = file.require_group(CERTIFICATE_H5_GROUP)
        if name in root:
            del root[name]
        group: h5py.Group = root.create_group(name)
        group.create_dataset(
            "canonical_json",
            data=np.frombuffer(data, dtype=np.uint8),
            compression="gzip",
            shuffle=True,
            fletcher32=True,
        )
        group.attrs["format"] = CERTIFICATE_FORMAT
        group.attrs["schema_version"] = certificate.manifest.schema_version
        group.attrs["model_id"] = certificate.model.model_id
        group.attrs["model_version"] = certificate.model.model_version
        group.attrs["policy_id"] = certificate.policy_id
        group.attrs["execution_id"] = certificate.manifest.execution_id
        group.attrs["consistency_checksum"] = document["consistency_checksum"]
        file.flush()


@jaxtyped(typechecker=beartype)
def attach_certificate_h5(
    path: str | Path,
    name: str,
    certificate: ForwardCertificate,
) -> None:
    """Attach a certificate atomically to an HDF5 result file.

    The function updates the complete file through a same-directory temporary.
    It preserves existing numerical result groups.

    :see: :class:`~.test_certificate.TestAttachCertificateH5`


    Implementation Logic
    --------------------
    1. **Encode the certificate**::

           document = _certificate_document(certificate)
           data = _json_bytes(document, newline=True)

       The HDF5 record stores the same canonical bytes as JSON persistence.
    2. **Copy the current container**::

           shutil.copy2(destination, temporary)

       An existing result file remains intact while the copy changes.
    3. **Write and replace the container**::

           _write_h5_record(temporary, name, data, certificate)
           os.replace(temporary, destination)
           temporary.unlink(missing_ok=True)

       Replacement publishes the complete file. Failure removes the temporary.

    Parameters
    ----------
    path : str | Path
        Existing HDF5 result path, or a path for a new HDF5 container.
    name : str
        Name of one result under the certificate index group.
    certificate : ForwardCertificate
        Certificate associated with the named result.

    Raises
    ------
    BaseException
        If copying, writing, or replacing the HDF5 file fails.
    """
    _validate_h5_name(name)
    destination: Path = Path(path)
    destination.parent.mkdir(parents=False, exist_ok=True)
    document: dict[str, Any] = _certificate_document(certificate)
    data: bytes = _json_bytes(document, newline=True)
    temporary_record: tuple[int, str] = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
    )
    descriptor: int = temporary_record[0]
    temporary_name: str = temporary_record[1]
    os.close(descriptor)
    temporary: Path = Path(temporary_name)
    try:
        if destination.exists():
            shutil.copy2(destination, temporary)
        _write_h5_record(temporary, name, data, certificate)
        os.replace(temporary, destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


@jaxtyped(typechecker=beartype)
def load_certificate_h5(
    path: str | Path,
    name: str,
) -> ForwardCertificate:
    """Load a certificate embedded in an HDF5 result file.

    The persistence operation retains the complete scientific-assurance record
    and its JAX array leaves. Consistency checks detect accidental storage
    corruption.

    :see: :class:`~.test_certificate.TestLoadCertificateH5`


    Implementation Logic
    --------------------
    1. **Resolve the stored record**::

           root = file[CERTIFICATE_H5_GROUP]
           group = root[name]

       Missing groups or names raise ``KeyError`` before decoding.
    2. **Decode the canonical bytes**::

           data = stored.tobytes()
           document = _read_document(data)
           certificate = _certificate_from_document(document)

       The decoder validates the persisted schema and consistency check.
    3. **Validate the convenience index**::

           msg: str = f"HDF5 certificate index mismatch for {key!r}"

       Every HDF5 attribute must agree with the canonical JSON record.

    Parameters
    ----------
    path : str | Path
        HDF5 result path.
    name : str
        Certificate name supplied to :func:`attach_certificate_h5`.

    Returns
    -------
    certificate : ForwardCertificate
        Reconstructed and validated certificate.

    Raises
    ------
    KeyError
        If the certificate group or named record is absent.
    ValueError
        If the exact JSON bytes or HDF5 convenience index are inconsistent.
    """
    file: Any
    key: Any
    expected: Any
    _validate_h5_name(name)
    source: Path = Path(path)
    with h5py.File(source, "r") as file:
        if CERTIFICATE_H5_GROUP not in file:
            msg: str = f"No certificates found in {source}"
            raise KeyError(msg)
        root: h5py.Group = file[CERTIFICATE_H5_GROUP]
        if name not in root:
            msg: str = f"Certificate '{name}' not found in {source}"
            raise KeyError(msg)
        group: h5py.Group = root[name]
        if "canonical_json" not in group:
            msg: str = "HDF5 certificate record has no canonical_json dataset"
            raise ValueError(msg)
        stored: UInt[NDArray, " n_byte"] = np.asarray(
            group["canonical_json"][()]
        )
        if stored.dtype != np.dtype(np.uint8) or stored.ndim != 1:
            msg: str = (
                "HDF5 canonical_json dataset must be one-dimensional uint8"
            )
            raise ValueError(msg)
        data: bytes = stored.tobytes()
        document: dict[str, Any] = _read_document(data)
        certificate: ForwardCertificate = _certificate_from_document(document)
        expected_attrs: dict[str, str] = {
            "format": CERTIFICATE_FORMAT,
            "schema_version": certificate.manifest.schema_version,
            "model_id": certificate.model.model_id,
            "model_version": certificate.model.model_version,
            "policy_id": certificate.policy_id,
            "execution_id": certificate.manifest.execution_id,
            "consistency_checksum": document["consistency_checksum"],
        }
        for key, expected in expected_attrs.items():
            actual: Any = group.attrs.get(key)
            if actual is None or str(actual) != expected:
                msg: str = f"HDF5 certificate index mismatch for {key!r}"
                raise ValueError(msg)
    return certificate


__all__: list[str] = [
    "attach_certificate_h5",
    "certificate_identity",
    "finalize_certificate",
    "load_certificate_h5",
    "load_certificate_json",
    "save_certificate_json",
]
