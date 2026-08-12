r"""Encode and identify portable forward-model certificates.

Extended Summary
----------------
This module defines the canonical JSON value codec and scientific identity
boundary for ForwardCertificate PyTrees. Numerical leaves retain their
dtype, shape, byte order, and exact canonical bytes.

Routine Listings
----------------
:func:`certificate_identity`
    Compute the scientific identity of a canonical certificate.
:func:`finalize_certificate`
    Replace the kernel placeholder with the canonical identity.
"""

from __future__ import annotations

import base64
import hashlib
import json
import math
import re
import unicodedata
import zlib
from collections.abc import Callable, Mapping
from dataclasses import fields, is_dataclass

import numpy as np
from beartype import beartype
from beartype.typing import Any, Dict, List, Tuple
from jaxtyping import Bool, Num, jaxtyped
from numpy.typing import NDArray

from diffpes.constants import (
    CERTIFICATE_ARRAY_KINDS,
    CERTIFICATE_FORMAT,
    CERTIFICATE_SCHEMA_MAJOR,
    CERTIFICATE_SCHEMA_PATTERN,
)
from diffpes.types import (
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
    return encoded


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


__all__: list[str] = [
    "certificate_identity",
    "finalize_certificate",
]
