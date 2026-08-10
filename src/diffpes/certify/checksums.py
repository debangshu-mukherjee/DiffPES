"""Compute collision-resistant identities for scientific records.

Extended Summary
----------------
Identities in this module use domain-separated SHA-256 over canonical records.
They provide collision-resistant content addressing, but no evidence of
authorship, authenticity, physical validity, numerical correctness, or
reproducibility. Certification policy must not treat a digest as a scientific
claim.

Every returned string records the algorithm, canonicalization version, record
kind, and digest value. The functions process large carrier and file
payloads in bounded chunks.

Routine Listings
----------------
:func:`checksum_bytes`
    Return a collision-resistant scientific identity for ``data``.
:func:`checksum_chunks`
    Compute a scientific identity over consecutive byte chunks.
:func:`checksum_file`
    Stream exact file bytes into a scientific identity.
:func:`checksum_pytree`
    Stream a canonical carrier into a scientific identity.
:func:`parse_checksum`
    Parse and validate one checksum string.
:func:`artifact_ref`
    Build separate byte, normalized-content, and semantic identities.
:func:`semantic_checksum`
    Identify content together with its declared scientific meaning.
:func:`result_checksum`
    Identify a result under a declared numerical configuration.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable
from pathlib import Path

from beartype import beartype
from beartype.typing import TYPE_CHECKING, Any, Tuple
from jaxtyping import jaxtyped

from diffpes.types import (
    CANONICAL_PYTREE_VERSION,
    CHECKSUM_ALGORITHM,
    CHECKSUM_FILE_CHUNK_BYTES,
    CHECKSUM_PATTERN,
    CHECKSUM_RECORD_KIND_PATTERN,
    make_artifact_ref,
)

from .canonical import iter_canonical_pytree_chunks

if TYPE_CHECKING:
    from diffpes.types import ArtifactRef


def _validate_record_kind(record_kind: str) -> None:
    """PRIVATE: Reject ambiguous or unstable checksum record-kind labels.

    Parameters
    ----------
    record_kind : str
        Candidate record-kind label, such as ``"content"`` or
        ``"result"``.

    Raises
    ------
    ValueError
        If the label does not match ``CHECKSUM_RECORD_KIND_PATTERN``.

    Notes
    -----
    Requires a full match of the lowercase-letter, digit, and hyphen
    pattern. A stable label keeps the domain-separation prefix and the
    formatted checksum unambiguous.
    """
    if CHECKSUM_RECORD_KIND_PATTERN.fullmatch(record_kind) is None:
        msg: str = (
            "record_kind must start with a lowercase letter and contain "
            "only lowercase letters, digits, and hyphens"
        )
        raise ValueError(msg)


def _identity_prefix(record_kind: str) -> bytes:
    """PRIVATE: Return the domain-separated preimage prefix for one
    record kind.

    Parameters
    ----------
    record_kind : str
        Validated record-kind label.

    Returns
    -------
    prefix : bytes
        NUL-separated ASCII fields: the fixed identity banner, the
        checksum algorithm, the canonicalization version, and the
        versioned schema identifier
        ``org.diffpes.identity.<record_kind>.v1``.

    Notes
    -----
    The prefix feeds the SHA-256 state before any payload byte. Distinct
    record kinds therefore hash into disjoint preimage domains, so equal
    content bytes cannot collide across identity roles.
    """
    schema_id: str = f"org.diffpes.identity.{record_kind}.v1"
    prefix: bytes = (
        b"DIFFPES-SCIENTIFIC-IDENTITY\x00"
        + CHECKSUM_ALGORITHM.encode("ascii")
        + b"\x00"
        + CANONICAL_PYTREE_VERSION.encode("ascii")
        + b"\x00"
        + schema_id.encode("ascii")
        + b"\x00"
    )
    return prefix


def _format_checksum(value: str, *, record_kind: str) -> str:
    """PRIVATE: Format a SHA-256 digest with its identity context.

    Parameters
    ----------
    value : str
        Hexadecimal SHA-256 digest.
    record_kind : str
        Validated record-kind label.

    Returns
    -------
    checksum : str
        Colon-joined ``algorithm:version:kind:digest`` identity string.

    Notes
    -----
    Embeds ``CHECKSUM_ALGORITHM`` and ``CANONICAL_PYTREE_VERSION`` so a
    stored identity states how a verifier must recompute it.
    """
    checksum: str = (
        f"{CHECKSUM_ALGORITHM}:{CANONICAL_PYTREE_VERSION}:"
        f"{record_kind}:{value}"
    )
    return checksum


@jaxtyped(typechecker=beartype)
def checksum_chunks(
    chunks: Iterable[bytes | memoryview],
    *,
    record_kind: str,
) -> str:
    """Compute a scientific identity over consecutive byte chunks.

    Domain-separated SHA-256 supplies collision-resistant content addressing.
    It is not authentication or scientific evidence.

    :see: :class:`~.test_checksums.TestChecksumChunks`


    Implementation Logic
    --------------------
    1. **Bind the documented output**::

           checksum: str = _format_checksum(
               digest.hexdigest(), record_kind=record_kind
           )

       The function validates and transforms the inputs before it binds the
       documented output.

    Parameters
    ----------
    chunks : Iterable[bytes | memoryview]
        Consecutive byte-like pieces of one record.
    record_kind : str
        Stable description, such as ``"normalized-content"`` or ``"result"``.

    Returns
    -------
    checksum : str
        Versioned, typed SHA-256 scientific identity.
    """
    chunk: Any
    _validate_record_kind(record_kind)
    digest: Any = hashlib.sha256()
    digest.update(_identity_prefix(record_kind))
    for chunk in chunks:
        digest.update(chunk)
    checksum: str = _format_checksum(
        digest.hexdigest(),
        record_kind=record_kind,
    )
    return checksum


@jaxtyped(typechecker=beartype)
def checksum_bytes(data: bytes, *, record_kind: str) -> str:
    """Return a collision-resistant scientific identity for ``data``.

    Domain-separated SHA-256 supplies collision-resistant content addressing.
    It is not authentication or scientific evidence.

    :see: :class:`~.test_checksums.TestChecksumBytes`


    Implementation Logic
    --------------------
    1. **Bind the documented output**::

           checksum: str = checksum_chunks((data,), record_kind=record_kind)

       The function validates and transforms the inputs before it binds the
       documented output.

    Parameters
    ----------
    data : bytes
        Exact record bytes.
    record_kind : str
        Stable kind distinguishing otherwise equal byte payloads.

    Returns
    -------
    checksum : str
        Typed SHA-256 scientific identity.
    """
    checksum: str = checksum_chunks((data,), record_kind=record_kind)
    return checksum


@jaxtyped(typechecker=beartype)
def checksum_pytree(tree: object, *, record_kind: str) -> str:
    """Stream a canonical carrier into a scientific identity.

    Domain-separated SHA-256 supplies collision-resistant content addressing.
    It is not authentication or scientific evidence.

    :see: :class:`~.test_checksums.TestChecksumPytree`


    Implementation Logic
    --------------------
    1. **Bind the documented output**::

           checksum: str = checksum_chunks(chunks, record_kind=record_kind)

       The function validates and transforms the inputs before it binds the
       documented output.

    Parameters
    ----------
    tree : object
        Supported carrier or nested scientific PyTree.
    record_kind : str
        Stable record-kind label.

    Returns
    -------
    checksum : str
        Typed SHA-256 scientific identity.
    """
    chunks: Iterable[bytes | memoryview] = iter_canonical_pytree_chunks(tree)
    checksum: str = checksum_chunks(chunks, record_kind=record_kind)
    return checksum


@jaxtyped(typechecker=beartype)
def checksum_file(path: str | Path, *, record_kind: str) -> str:
    """Stream exact file bytes into a scientific identity.

    Domain-separated SHA-256 supplies collision-resistant content addressing.
    It is not authentication or scientific evidence.

    :see: :class:`~.test_checksums.TestChecksumFile`


    Implementation Logic
    --------------------
    1. **Bind the documented output**::

           checksum: str = checksum_chunks(chunks(), record_kind=record_kind)

       The function validates and transforms the inputs before it binds the
       documented output.

    Parameters
    ----------
    path : str | Path
        Existing regular file.
    record_kind : str
        Stable record-kind label.

    Returns
    -------
    checksum : str
        Typed SHA-256 scientific identity.
    """
    source: Path = Path(path)

    def chunks() -> Iterable[bytes]:
        stream: Any
        chunk: Any
        with source.open("rb") as stream:
            while chunk := stream.read(CHECKSUM_FILE_CHUNK_BYTES):
                yield chunk

    checksum: str = checksum_chunks(chunks(), record_kind=record_kind)
    return checksum


@jaxtyped(typechecker=beartype)
def parse_checksum(checksum: str) -> Tuple[str, str, str, str]:
    """Parse and validate one checksum string.

    The parser rejects legacy CRC32 strings. Domain-separated SHA-256 supplies
    collision-resistant content addressing, not authentication or scientific
    evidence.

    :see: :class:`~.test_checksums.TestParseChecksum`


    Implementation Logic
    --------------------
    1. **Bind the documented output**::

           parsed: Tuple[str, str, str, str] = (
                   CHECKSUM_ALGORITHM,
                   match.group("canonical"),
                   match.group("kind"),
                   match.group("value"),
               )

       The function validates and transforms the inputs before it binds the
       documented output.

    Parameters
    ----------
    checksum : str
        Value produced by this module.

    Returns
    -------
    parsed : Tuple[str, str, str, str]
        Algorithm, canonicalization version, record kind, and hexadecimal
        value.

    Raises
    ------
    ValueError
        If ``checksum`` is not in the current explicit format.
    """
    match: re.Match[str] | None = CHECKSUM_PATTERN.fullmatch(checksum)
    if match is None:
        msg: str = "invalid DiffPES scientific-identity format"
        raise ValueError(msg)
    parsed: Tuple[str, str, str, str] = (
        CHECKSUM_ALGORITHM,
        match.group("canonical"),
        match.group("kind"),
        match.group("value"),
    )
    return parsed


@jaxtyped(typechecker=beartype)
def semantic_checksum(
    value: object,
    semantics: object,
) -> str:
    """Identify content together with its declared scientific meaning.

    Domain-separated SHA-256 supplies collision-resistant content addressing.
    It is not authentication or scientific evidence.

    :see: :class:`~.test_checksums.TestSemanticChecksum`


    Implementation Logic
    --------------------
    1. **Bind the documented output**::

           checksum: str = checksum_pytree(payload, record_kind="semantic")

       The function validates and transforms the inputs before it binds the
       documented output.

    Parameters
    ----------
    value : object
        Normalized scientific content.
    semantics : object
        Units, axes, frames, conventions, schema, and other meaning-bearing
        declarations.

    Returns
    -------
    checksum : str
        Collision-resistant semantic identity.
    """
    payload: Tuple[str, object, object] = (
        "org.diffpes.semantic-record.v1",
        value,
        semantics,
    )
    checksum: str = checksum_pytree(payload, record_kind="semantic")
    return checksum


@jaxtyped(typechecker=beartype)
def result_checksum(value: object, numerical: object) -> str:
    """Identify a result under a declared numerical configuration.

    Domain-separated SHA-256 supplies collision-resistant content addressing.
    It is not authentication or scientific evidence.

    :see: :class:`~.test_checksums.TestResultChecksum`


    Implementation Logic
    --------------------
    1. **Bind the documented output**::

           checksum: str = checksum_pytree(payload, record_kind="result")

       The function validates and transforms the inputs before it binds the
       documented output.

    Parameters
    ----------
    value : object
        Result carrier.
    numerical : object
        Precision, backend-independent tolerance semantics, and any other
        numerical configuration that defines result identity.

    Returns
    -------
    checksum : str
        Collision-resistant result identity.
    """
    payload: Tuple[str, object, object] = (
        "org.diffpes.result-record.v1",
        value,
        numerical,
    )
    checksum: str = checksum_pytree(payload, record_kind="result")
    return checksum


@jaxtyped(typechecker=beartype)
def artifact_ref(
    path: str | Path,
    normalized: object,
    *,
    role: str,
    media_type: str = "application/octet-stream",
    semantics: object | None = None,
    artifact_id: str | None = None,
) -> ArtifactRef:
    """Build separate byte, normalized-content, and semantic identities.

    Domain-separated SHA-256 supplies collision-resistant content addressing.
    It is not authentication or scientific evidence.

    :see: :class:`~.test_checksums.TestArtifactRef`


    Implementation Logic
    --------------------
    1. **Bind the documented output**::

           reference: ArtifactRef = make_artifact_ref(
                   artifact_id=resolved_id,
                   media_type=media_type,
                   byte_checksum=byte_value,
                   content_checksum=content_value,
                   semantic_checksum=semantic_value,
                   locator=str(source),
                   role=role,
               )

       The function validates and transforms the inputs before it binds the
       documented output.

    Parameters
    ----------
    path : str | Path
        Source artifact with a record of its exact bytes.
    normalized : object
        Parsed, normalized scientific carrier derived from the source.
    role : str
        Scientific role of this artifact in the execution.
    media_type : str, optional
        Declared media type of the source bytes.
    semantics : object | None, optional
        Meaning-bearing declarations. By default, the role and normalized
        carrier type form the minimal semantic descriptor.
    artifact_id : str | None, optional
        Stable caller-owned identity. By default, the function derives a local
        identity from the byte checksum value.

    Returns
    -------
    reference : ArtifactRef
        Immutable certification carrier with three deliberately separate
        scientific identities.
    """
    source: Path = Path(path)
    byte_value: str = checksum_file(source, record_kind="artifact-bytes")
    content_value: str = checksum_pytree(
        normalized,
        record_kind="normalized-content",
    )
    descriptor: object
    if semantics is None:
        descriptor = {
            "carrier_type": (
                f"{type(normalized).__module__}."
                f"{type(normalized).__qualname__}"
            ),
            "role": role,
        }
    else:
        descriptor = semantics
    semantic_value: str = semantic_checksum(normalized, descriptor)
    if artifact_id is None:
        checksum_value: str = parse_checksum(byte_value)[3]
        resolved_id: str = f"artifact-{checksum_value}"
    else:
        resolved_id = artifact_id
    reference: ArtifactRef = make_artifact_ref(
        artifact_id=resolved_id,
        media_type=media_type,
        byte_checksum=byte_value,
        content_checksum=content_value,
        semantic_checksum=semantic_value,
        locator=str(source),
        role=role,
    )
    return reference


__all__: list[str] = [
    "artifact_ref",
    "checksum_bytes",
    "checksum_chunks",
    "checksum_file",
    "checksum_pytree",
    "parse_checksum",
    "result_checksum",
    "semantic_checksum",
]
