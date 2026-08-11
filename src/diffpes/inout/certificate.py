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
from beartype.typing import Any, Dict, List, Tuple
from jaxtyping import Bool, Num, UInt8, jaxtyped
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


def _module_factories() -> Dict[type[Any], Callable[..., Any]]:
    """PRIVATE: Return types-owned carrier factories for the codec.

    Returns
    -------
    factories : Dict[type[Any], Callable[..., Any]]
        Concrete carrier types mapped to their validating factories.

    Notes
    -----
    The codec reconstructs every certification carrier through its
    types-owned factory, so loading repeats the validation contract.
    """
    factories: Dict[type[Any], Callable[..., Any]] = {
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


def _module_types() -> Dict[str, type[Any]]:
    """PRIVATE: Return persisted carrier names mapped to concrete types.

    Returns
    -------
    module_types : Dict[str, type[Any]]
        Persisted class names mapped to their carrier types.

    Notes
    -----
    The mapping derives from the factory table, so both directions of
    the codec share one carrier inventory.
    """
    module_types: Dict[str, type[Any]] = {
        module_type.__name__: module_type
        for module_type in _module_factories()
    }
    return module_types


def _normalize_text(value: str) -> str:
    """PRIVATE: Return NFC-normalized certificate text.

    Parameters
    ----------
    value : str
        Raw text from a certificate field or JSON key.

    Returns
    -------
    normalized : str
        Unicode NFC normalization of the input.

    Notes
    -----
    One shared normalization keeps byte-level JSON output stable for
    equal text written in different Unicode compositions.
    """
    normalized: str = unicodedata.normalize("NFC", value)
    return normalized


def _normalize_json_value(value: Any) -> Any:
    """PRIVATE: Normalize an extension value and reject non-JSON/nonfinite
    data.

    Implementation Logic
    --------------------
    Recurses structurally.  ``None``, booleans, and integers pass
    through.  Floats must be finite.  Strings normalize to NFC.  Lists
    recurse elementwise.  Mappings normalize each string key, reject
    post-normalization key collisions, and recurse into the values.
    Every other type raises.

    Parameters
    ----------
    value : Any
        Candidate JSON value of any nesting depth.

    Returns
    -------
    normalized_value : Any
        Equivalent value with every string NFC normalized.

    Raises
    ------
    ValueError
        If a float is not finite, an object key is not a string, two
        keys collide after normalization, or a type is not JSON.
    """
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
        normalized: Dict[str, Any] = {}
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
    """PRIVATE: Encode a normalized mapping as deterministic UTF-8 JSON.

    Parameters
    ----------
    value : Mapping[str, Any]
        JSON-ready mapping to serialize.
    newline : bool
        Append one trailing newline byte when true.

    Returns
    -------
    encoded : bytes
        Canonical UTF-8 JSON bytes.

    Notes
    -----
    Normalizes the value first, then serializes with sorted keys,
    compact separators, non-ASCII passthrough, and ``allow_nan``
    disabled.  Equal documents therefore produce identical bytes.
    """
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
    """PRIVATE: Return the non-security CRC32 of a document without its
    checksum.

    Parameters
    ----------
    document : Mapping[str, Any]
        Certificate document, with or without its checksum field.

    Returns
    -------
    checksum : str
        Formatted ``crc32:certificate-json-v1:<8 hex digits>`` value.

    Notes
    -----
    Drops the ``consistency_checksum`` field, serializes the rest with
    :func:`_json_bytes`, and applies :func:`zlib.crc32`.  The checksum
    detects accidental storage corruption only; it is not an
    authentication or certification mechanism.
    """
    payload: Dict[str, Any] = dict(document)
    payload.pop("consistency_checksum", None)
    value: int = zlib.crc32(_json_bytes(payload, newline=False))
    checksum: str = f"crc32:certificate-json-v1:{value & 0xFFFFFFFF:08x}"
    return checksum


def _identity_payload(document: Mapping[str, Any]) -> Dict[str, Any]:
    """PRIVATE: Return canonical scientific fields without audit-only
    identities.

    Parameters
    ----------
    document : Mapping[str, Any]
        Complete encoded certificate document.

    Returns
    -------
    payload : Dict[str, Any]
        Document copy restricted to the identity-relevant fields.

    Notes
    -----
    Keeps ``format``, ``schema_version``, ``certificate``, and
    ``extensions``.  A JSON round trip deep-copies the certificate
    node.  The copy then drops the ``certificate_checksum``
    self-reference and the audit-only manifest fields
    ``execution_id`` and ``started_at_utc``.
    """
    payload: Dict[str, Any] = {
        "format": document["format"],
        "schema_version": document["schema_version"],
        "certificate": json.loads(json.dumps(document["certificate"])),
        "extensions": document["extensions"],
    }
    fields_node: Dict[str, Any] = payload["certificate"]["fields"]
    fields_node.pop("certificate_checksum", None)
    manifest_fields: Dict[str, Any] = fields_node["manifest"]["fields"]
    manifest_fields.pop("execution_id", None)
    manifest_fields.pop("started_at_utc", None)
    return payload


def _document_identity(document: Mapping[str, Any]) -> str:
    """PRIVATE: Return the domain-separated SHA-256 certificate identity.

    Parameters
    ----------
    document : Mapping[str, Any]
        Complete encoded certificate document.

    Returns
    -------
    identity : str
        Formatted ``sha256:1:certificate:<hex digest>`` identity.

    Notes
    -----
    Hashes a fixed NUL-separated domain prefix and then the canonical
    JSON bytes of :func:`_identity_payload`.  The prefix keeps
    certificate identities in a preimage domain disjoint from other
    project checksums.  The digest provides content addressing, not
    authentication.
    """
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


def _encode_array(value: object) -> Dict[str, Any]:
    """PRIVATE: Encode one concrete numerical leaf without decimal
    conversion.

    Implementation Logic
    --------------------
    Converts to a little-endian C-order NumPy array and base64-encodes
    the raw buffer.  The record therefore stores the exact binary
    values; no decimal text conversion can lose precision.

    Parameters
    ----------
    value : object
        Concrete array-like numerical leaf.

    Returns
    -------
    result : Dict[str, Any]
        Array record with ``kind``, ``dtype``, ``shape``,
        ``byte_order``, ``order``, ``encoding``, and base64 ``data``.

    Raises
    ------
    ValueError
        If the input is not a concrete array, the dtype kind lies
        outside the supported set, or an entry is not finite.
    """
    exc: Exception
    try:
        array: Bool[NDArray, "..."] | Num[NDArray, "..."] = np.asarray(value)
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
    canonical: Bool[NDArray, "..."] | Num[NDArray, "..."] = np.asarray(
        array,
        dtype=canonical_dtype,
        order="C",
    )
    payload: str = base64.b64encode(canonical.tobytes(order="C")).decode(
        "ascii"
    )
    result: Dict[str, Any] = {
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
    """PRIVATE: Return whether a value exposes a concrete numerical array
    protocol.

    Parameters
    ----------
    value : object
        Candidate certificate field value.

    Returns
    -------
    is_array : bool
        True for NumPy arrays and scalars, and for objects with
        ``__array__``, ``dtype``, and ``shape`` attributes.

    Notes
    -----
    The attribute test also accepts concrete JAX arrays, so numerical
    leaves route into the lossless array codec.
    """
    if isinstance(value, np.ndarray | np.generic):
        is_array: bool = True
        return is_array
    attributes: Tuple[str, ...] = ("__array__", "dtype", "shape")
    is_array: bool = all(hasattr(value, attr) for attr in attributes)
    return is_array


def _encode_value(  # noqa: PLR0911
    value: object,
    *,
    root: bool = False,
) -> Any:
    """PRIVATE: Encode one supported carrier field into the transparent
    schema.

    Parameters
    ----------
    value : object
        Carrier field value of any supported type.
    root : bool, optional
        True only for the top-level certificate.  The root encoding
        skips the ``extensions_json`` field.  Default is false.

    Returns
    -------
    encoded : Any
        JSON-ready scalar or tagged record.

    Raises
    ------
    ValueError
        If a float is not finite, a mapping key is not a string, or
        the type is not in the supported set.

    Notes
    -----
    Recurses structurally.  ``None``, booleans, and integers pass
    through; floats must be finite; strings normalize to NFC.  Arrays
    encode through :func:`_encode_array`.  Tuples, lists, and mappings
    become ``kind``-tagged records.  A whitelisted dataclass carrier
    becomes a ``module`` record with its encoded fields; the root
    certificate omits ``extensions_json`` because the document stores
    extensions separately.
    """
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
        encoded_fields: Dict[str, Any] = {}
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


def _parse_extensions(certificate: ForwardCertificate) -> Dict[str, Any]:
    """PRIVATE: Parse and normalize the certificate extension object.

    Parameters
    ----------
    certificate : ForwardCertificate
        Carrier whose ``extensions_json`` string holds the extension
        object.

    Returns
    -------
    normalized : Dict[str, Any]
        NFC-normalized extension mapping.

    Raises
    ------
    ValueError
        If the string is not valid JSON or does not encode an object.

    Notes
    -----
    Parses with the duplicate-key and constant-rejecting hooks, so
    ``NaN`` tokens and repeated keys fail, then applies
    :func:`_normalize_json_value`.
    """
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
    document: Dict[str, Any] = {
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


def _certificate_document(certificate: ForwardCertificate) -> Dict[str, Any]:
    """PRIVATE: Build the complete portable document for one certificate.

    Parameters
    ----------
    certificate : ForwardCertificate
        Concrete certificate at the persistence boundary.

    Returns
    -------
    document : Dict[str, Any]
        Mapping with ``format``, ``schema_version``, the encoded
        ``certificate``, ``extensions``, and the storage checksum.

    Notes
    -----
    Finalizes the certificate first, so the stored record carries its
    canonical identity.  The schema helpers raise ``ValueError`` for
    an unsupported schema version or unsupported field values.
    """
    finalized: ForwardCertificate = finalize_certificate(certificate)
    schema_version: str = finalized.manifest.schema_version
    _parse_schema_version(schema_version)
    document: Dict[str, Any] = {
        "format": CERTIFICATE_FORMAT,
        "schema_version": schema_version,
        "certificate": _encode_value(finalized, root=True),
        "extensions": _parse_extensions(finalized),
    }
    document["consistency_checksum"] = _storage_checksum(document)
    return document


def _reject_json_constant(value: str) -> None:
    """PRIVATE: Reject JSON's non-standard NaN and Infinity tokens.

    Parameters
    ----------
    value : str
        Constant token the JSON parser encountered, such as ``"NaN"``.

    Raises
    ------
    ValueError
        Always; the certificate grammar has no nonfinite constants.

    Notes
    -----
    Installed as the ``parse_constant`` hook, so the parser calls this
    function only for ``NaN``, ``Infinity``, and ``-Infinity``.
    """
    msg: str = f"certificate JSON contains invalid constant {value!r}"
    raise ValueError(msg)


def _unique_object(pairs: List[Tuple[str, Any]]) -> Dict[str, Any]:
    """PRIVATE: Build a JSON object while rejecting duplicate names.

    Parameters
    ----------
    pairs : List[Tuple[str, Any]]
        Key-value pairs of one JSON object in document order.

    Returns
    -------
    result : Dict[str, Any]
        Mapping with every pair inserted exactly once.

    Raises
    ------
    ValueError
        If a key repeats within one object.

    Notes
    -----
    Installed as the ``object_pairs_hook``, so a duplicated key fails
    the parse instead of silently keeping the last value.
    """
    key: Any
    value: Any
    result: Dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            msg: str = f"duplicate certificate JSON key: {key!r}"
            raise ValueError(msg)
        result[key] = value
    return result


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


def _parse_schema_version(value: object) -> Tuple[int, int]:
    """PRIVATE: Parse a schema version and reject unsupported major
    versions.

    Parameters
    ----------
    value : object
        Candidate schema version value from a document or manifest.

    Returns
    -------
    parsed : Tuple[int, int]
        Pair of the major and minor version numbers.

    Raises
    ------
    ValueError
        If the value is not a string, does not match the schema
        pattern, or names a different major version.

    Notes
    -----
    A missing minor component parses as zero.  The reader accepts only
    its own major version; minor versions control unknown-field handling
    elsewhere.
    """
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
    parsed: Tuple[int, int] = (major, minor)
    return parsed


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
    return decoded  # noqa: RET504


def _atomic_write(path: Path, data: bytes) -> None:
    """PRIVATE: Write bytes through a same-directory temporary and atomic
    replace.

    Parameters
    ----------
    path : Path
        Destination file path; its parent directory must exist.
    data : bytes
        Exact bytes to publish.

    Raises
    ------
    BaseException
        If temporary creation, writing, syncing, or replacement fails;
        the handler removes the temporary file first.

    Notes
    -----
    Writes into a ``mkstemp`` file in the destination directory,
    flushes and fsyncs it, and publishes with :func:`os.replace`.  The
    same-directory temporary keeps the replace atomic on one
    filesystem, so readers never observe a partial record.
    """
    stream: Any
    path.parent.mkdir(parents=False, exist_ok=True)
    temporary_record: Tuple[int, str] = tempfile.mkstemp(
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
    document: Dict[str, Any] = _certificate_document(certificate)
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
    document: Dict[str, Any] = _read_document(data)
    certificate: ForwardCertificate = _certificate_from_document(document)
    return certificate


def _validate_h5_name(name: str) -> None:
    """PRIVATE: Reject ambiguous or path-like HDF5 certificate names.

    Parameters
    ----------
    name : str
        Requested certificate entry name.

    Raises
    ------
    ValueError
        If the name is empty, ``"."``, ``".."``, or contains a slash
        or NUL character.

    Notes
    -----
    The name must stay one plain group component, so a caller cannot
    address groups outside the certificate index.
    """
    if not name or name in {".", ".."} or "/" in name or "\x00" in name:
        msg: str = "HDF5 certificate name must be one nonblank group component"
        raise ValueError(msg)


def _write_h5_record(
    path: Path,
    name: str,
    data: bytes,
    certificate: ForwardCertificate,
) -> None:
    """PRIVATE: Write one exact JSON record and its convenience index
    attributes.

    Parameters
    ----------
    path : Path
        HDF5 container to open in append mode.
    name : str
        Validated certificate entry name.
    data : bytes
        Canonical JSON bytes of the certificate document.
    certificate : ForwardCertificate
        Finalized certificate that supplies the index attributes.

    Notes
    -----
    Revalidates ``data`` through :func:`_read_document` first.  The
    entry replaces any same-named group under the certificate index
    group and stores the exact bytes as one compressed,
    checksummed ``uint8`` dataset.  Convenience attributes copy the
    format, schema version, model identity, policy, execution ID, and
    storage checksum for quick inspection.
    """
    file: Any
    document: Dict[str, Any] = _read_document(data)
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
    document: Dict[str, Any] = _certificate_document(certificate)
    data: bytes = _json_bytes(document, newline=True)
    temporary_record: Tuple[int, str] = tempfile.mkstemp(
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
        stored: UInt8[NDArray, " n_byte"] = np.asarray(
            group["canonical_json"][()]
        )
        if stored.dtype != np.dtype(np.uint8) or stored.ndim != 1:
            msg: str = (
                "HDF5 canonical_json dataset must be one-dimensional uint8"
            )
            raise ValueError(msg)
        data: bytes = stored.tobytes()
        document: Dict[str, Any] = _read_document(data)
        certificate: ForwardCertificate = _certificate_from_document(document)
        expected_attrs: Dict[str, str] = {
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
