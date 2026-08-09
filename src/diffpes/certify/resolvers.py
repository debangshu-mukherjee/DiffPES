"""Resolve certificate artifacts and verify external evidence.

Extended Summary
----------------
Resolvers operate at the eager I/O boundary. They return normalized scientific
content and may also return exact source bytes. The module checks both forms
against their separate SHA-256 identities. No resolver participates
in a traced physics calculation.

Routine Listings
----------------
:func:`filesystem_artifact_resolver`
    Resolve a byte-valued artifact from its local locator.
:func:`mapping_artifact_resolver`
    Build a deterministic resolver from normalized in-memory values.
:func:`resolve_artifact`
    Resolve and validate one referenced artifact.
:func:`verify_evidence`
    Verify referenced artifacts and recorded numerical residuals.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import jax.numpy as jnp
from beartype import beartype
from beartype.typing import Any, Tuple
from jaxtyping import jaxtyped

from diffpes.types import (
    ArtifactRef,
    ArtifactResolver,
    EvidenceRef,
    EvidenceReport,
    make_evidence_report,
)

from .checksums import checksum_bytes, checksum_pytree


@jaxtyped(typechecker=beartype)
def mapping_artifact_resolver(
    artifacts: Mapping[str, Any],
) -> ArtifactResolver:
    """Build a deterministic resolver from normalized in-memory values.

    The resolver uses exact artifact IDs and an immutable mapping copy.

    :see: :class:`~.test_resolvers.TestMappingArtifactResolver`

    Implementation Logic
    --------------------
    1. **Freeze the supplied mapping**::

           frozen = dict(artifacts)

       The resolver reads this copy for each exact artifact ID.

    Parameters
    ----------
    artifacts : Mapping[str, Any]
        Normalized values keyed by exact artifact ID.

    Returns
    -------
    resolver : ArtifactResolver
        Resolver that returns normalized values without source bytes.
    """
    frozen: dict[str, Any] = dict(artifacts)

    def resolver(reference: ArtifactRef) -> Tuple[Any, bytes | None]:
        if reference.artifact_id not in frozen:
            msg: str = f"unresolved artifact: {reference.artifact_id}"
            raise KeyError(msg)
        result: Tuple[Any, bytes | None] = (
            frozen[reference.artifact_id],
            None,
        )
        return result

    return resolver


@jaxtyped(typechecker=beartype)
def filesystem_artifact_resolver(
    reference: ArtifactRef,
) -> Tuple[bytes, bytes]:
    """Resolve a byte-valued artifact from its local locator.

    The resolver returns normalized bytes and exact source bytes separately.

    :see: :class:`~.test_resolvers.TestFilesystemArtifactResolver`

    Implementation Logic
    --------------------
    1. **Read the local artifact**::

           data = Path(reference.locator).read_bytes()

       The calling validator checks both returned byte identities.

    Parameters
    ----------
    reference : ArtifactRef
        Artifact with a local filesystem locator.

    Returns
    -------
    resolved : Tuple[bytes, bytes]
        Normalized byte value and the same exact source bytes.

    Raises
    ------
    ValueError
        If the artifact has no local locator.
    """
    if reference.locator is None:
        msg: str = "artifact has no filesystem locator"
        raise ValueError(msg)
    data: bytes = Path(reference.locator).read_bytes()
    resolved: Tuple[bytes, bytes] = (data, data)
    return resolved


@jaxtyped(typechecker=beartype)
def resolve_artifact(
    reference: ArtifactRef,
    resolver: ArtifactResolver,
) -> Any:
    """Resolve and validate one referenced artifact.

    The function checks normalized content and any available exact bytes.

    :see: :class:`~.test_resolvers.TestResolveArtifact`

    Implementation Logic
    --------------------
    1. **Check normalized content**::

           content_checksum = checksum_pytree(
               value, record_kind="normalized-content"
           )

       The function rejects content that differs from the artifact reference.

    Parameters
    ----------
    reference : ArtifactRef
        Expected byte, normalized-content, and semantic identities.
    resolver : ArtifactResolver
        Eager resolver for the artifact location or backing store.

    Returns
    -------
    value : Any
        Validated normalized scientific content.

    Raises
    ------
    ValueError
        If a returned byte or content identity does not match.
    """
    value: Any
    exact_bytes: bytes | None
    value, exact_bytes = resolver(reference)
    content_checksum: str = checksum_pytree(
        value,
        record_kind="normalized-content",
    )
    if content_checksum != reference.content_checksum:
        msg: str = f"artifact content mismatch: {reference.artifact_id}"
        raise ValueError(msg)
    if reference.byte_checksum is not None and exact_bytes is None:
        msg = f"artifact source bytes unavailable: {reference.artifact_id}"
        raise ValueError(msg)
    if exact_bytes is not None and reference.byte_checksum is not None:
        byte_checksum: str = checksum_bytes(
            exact_bytes,
            record_kind="artifact-bytes",
        )
        if byte_checksum != reference.byte_checksum:
            msg = f"artifact byte mismatch: {reference.artifact_id}"
            raise ValueError(msg)
    return value


@jaxtyped(typechecker=beartype)
def verify_evidence(
    reference: EvidenceRef,
    artifacts: Tuple[ArtifactRef, ...],
    resolver: ArtifactResolver,
) -> EvidenceReport:
    """Verify referenced artifacts and recorded numerical residuals.

    The report keeps resolution, compatibility, and tolerance outcomes
    distinct.

    :see: :class:`~.test_resolvers.TestVerifyEvidence`

    Implementation Logic
    --------------------
    1. **Resolve each evidence artifact**::

           resolve_artifact(artifact, resolver)

       The final outcome also requires every numerical residual to fit.

    Parameters
    ----------
    reference : EvidenceRef
        Numerical evidence and the artifact IDs that support it.
    artifacts : Tuple[ArtifactRef, ...]
        Available artifact records.
    resolver : ArtifactResolver
        Eager resolver for normalized artifact content.

    Returns
    -------
    report : EvidenceReport
        Resolution, compatibility, and numerical tolerance outcome.
    """
    by_id: dict[str, ArtifactRef] = {
        artifact.artifact_id: artifact for artifact in artifacts
    }
    resolved: bool = True
    compatible: bool = True
    artifact_id: str
    for artifact_id in reference.lineage.artifact_refs:
        artifact: ArtifactRef | None = by_id.get(artifact_id)
        if artifact is None:
            resolved = False
            compatible = False
            continue
        try:
            resolve_artifact(artifact, resolver)
        except (KeyError, OSError):
            resolved = False
            compatible = False
        except ValueError:
            compatible = False
    residual_norm: Any = jnp.max(jnp.abs(reference.residual))
    residual_consistent: Any = jnp.array_equal(
        reference.residual,
        reference.measured - reference.reference,
    )
    numerical_passed: Any = residual_consistent & jnp.all(
        jnp.abs(reference.residual) <= reference.tolerance
    )
    report: EvidenceReport = make_evidence_report(
        evidence_id=reference.evidence_id,
        resolved=jnp.asarray(resolved),
        compatible=jnp.asarray(compatible),
        passed=jnp.asarray(resolved & compatible) & numerical_passed,
        residual_norm=residual_norm,
    )
    return report


__all__: list[str] = [
    "filesystem_artifact_resolver",
    "mapping_artifact_resolver",
    "resolve_artifact",
    "verify_evidence",
]
