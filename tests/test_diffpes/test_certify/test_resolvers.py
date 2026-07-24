"""Validate artifact resolution and external evidence checks.

The tests compare normalized and byte content with explicit SHA-256 identities.
They also check missing artifacts and changed content.
"""

from pathlib import Path

import jax.numpy as jnp
import pytest
from beartype.typing import Any

from diffpes.certify import (
    checksum_bytes,
    checksum_pytree,
    filesystem_artifact_resolver,
    mapping_artifact_resolver,
    resolve_artifact,
    verify_evidence,
)
from diffpes.types import (
    make_artifact_ref,
    make_evidence_lineage,
    make_evidence_ref,
)


def _artifact(value: Any, artifact_id: str = "input") -> Any:
    """Build an artifact reference for one normalized value."""
    reference: Any = make_artifact_ref(
        artifact_id=artifact_id,
        media_type="application/x-diffpes-array",
        byte_checksum=None,
        content_checksum=checksum_pytree(
            value,
            record_kind="normalized-content",
        ),
        semantic_checksum=(
            "sha256:1:semantic:"
            "0000000000000000000000000000000000000000000000000000000000000000"
        ),
        locator=None,
        role="normalized-input",
    )
    return reference


class TestMappingArtifactResolver:
    """Verify :func:`~diffpes.certify.mapping_artifact_resolver`.

    The cases cover exact lookup of normalized in-memory artifacts.

    :see: :func:`~diffpes.certify.mapping_artifact_resolver`
    """

    def test_mapping_resolves_exact_artifact_id(self) -> None:
        """Resolve the normalized value through its exact artifact ID.

        The resolver must return the array that the exact ID selects.

        Notes
        -----
        The test creates one array and compares the resolved array exactly.
        """
        value: Any = jnp.array([1.0, 2.0])
        reference: Any = _artifact(value)
        resolver: Any = mapping_artifact_resolver({"input": value})
        resolved: Any = resolve_artifact(reference, resolver)
        assert jnp.array_equal(resolved, value)


class TestFilesystemArtifactResolver:
    """Verify :func:`~diffpes.certify.filesystem_artifact_resolver`.

    The case covers exact byte resolution from a local locator.

    :see: :func:`~diffpes.certify.filesystem_artifact_resolver`
    """

    def test_filesystem_resolves_and_checks_exact_bytes(
        self,
        tmp_path: Path,
    ) -> None:
        """Resolve one file and validate its byte and content identities.

        The resolver must return the four bytes without a conversion.

        Notes
        -----
        The test writes four bytes and checks both independent SHA-256 fields.
        """
        path: Path = tmp_path / "artifact.bin"
        data: bytes = b"data"
        path.write_bytes(data)
        reference: Any = make_artifact_ref(
            artifact_id="bytes",
            media_type="application/octet-stream",
            byte_checksum=checksum_bytes(data, record_kind="artifact-bytes"),
            content_checksum=checksum_pytree(
                data,
                record_kind="normalized-content",
            ),
            semantic_checksum=(
                "sha256:1:semantic:"
                "0000000000000000000000000000000000000000000000000000000000000000"
            ),
            locator=str(path),
            role="source",
        )
        resolved: Any = resolve_artifact(
            reference,
            filesystem_artifact_resolver,
        )
        assert resolved == data


class TestResolveArtifact:
    """Verify :func:`~diffpes.certify.resolve_artifact`.

    The case rejects changed normalized content at the I/O boundary.

    :see: :func:`~diffpes.certify.resolve_artifact`
    """

    def test_changed_content_is_rejected(self) -> None:
        """Reject a value that differs from the referenced normalized value.

        The content identity must detect the changed numerical array.

        Notes
        -----
        The test resolves a different scalar under the same artifact ID.
        """
        reference: Any = _artifact(jnp.array([1.0]))
        resolver: Any = mapping_artifact_resolver({"input": jnp.array([2.0])})
        with pytest.raises(ValueError, match="content mismatch"):
            resolve_artifact(reference, resolver)

    def test_declared_byte_identity_requires_source_bytes(self) -> None:
        """Reject a resolver that omits bytes for a declared byte identity.

        The content value alone cannot verify the separately recorded bytes.

        Notes
        -----
        The mapping resolver intentionally returns no exact source bytes.
        """
        value: Any = jnp.array([1.0])
        reference: Any = make_artifact_ref(
            artifact_id="input",
            media_type="application/x-diffpes-array",
            byte_checksum=checksum_bytes(
                b"source", record_kind="artifact-bytes"
            ),
            content_checksum=checksum_pytree(
                value,
                record_kind="normalized-content",
            ),
            semantic_checksum=(
                "sha256:1:semantic:"
                "0000000000000000000000000000000000000000000000000000000000000000"
            ),
            locator=None,
            role="normalized-input",
        )
        resolver: Any = mapping_artifact_resolver({"input": value})
        with pytest.raises(ValueError, match="source bytes unavailable"):
            resolve_artifact(reference, resolver)


class TestVerifyEvidence:
    """Verify :func:`~diffpes.certify.verify_evidence`.

    The cases combine artifact resolution with the recorded tolerance.

    :see: :func:`~diffpes.certify.verify_evidence`
    """

    def test_evidence_requires_resolved_compatible_artifacts(self) -> None:
        """Pass evidence only when its artifact resolves and its residual fits.

        The report must keep all three Boolean outcomes true.

        Notes
        -----
        The test uses a zero residual and one matching normalized artifact.
        """
        value: Any = jnp.array([1.0])
        artifact: Any = _artifact(value)
        evidence: Any = make_evidence_ref(
            evidence_id="reference",
            method_id="org.diffpes.method.test",
            source_type="analytic_reference",
            measured=jnp.array([1.0]),
            reference=jnp.array([1.0]),
            residual=jnp.array([0.0]),
            tolerance=jnp.array([1e-12]),
            lineage=make_evidence_lineage(
                implementation_refs=("reference.impl",),
                generator_refs=("reference.generator",),
                artifact_refs=(artifact.artifact_id,),
                derivation_refs=("reference.derivation",),
                relationship_ids=(
                    "independent-derivation:reference.derivation",
                ),
            ),
        )
        report: Any = verify_evidence(
            evidence,
            (artifact,),
            mapping_artifact_resolver({artifact.artifact_id: value}),
        )
        assert bool(report.resolved)
        assert bool(report.compatible)
        assert bool(report.passed)

    def test_inconsistent_recorded_residual_fails(self) -> None:
        """Reject evidence whose residual disagrees with its stored values.

        Artifact resolution must not hide an inconsistent numerical record.

        Notes
        -----
        The recorded residual is zero although the measured difference is one.
        """
        value: Any = jnp.array([1.0])
        artifact: Any = _artifact(value)
        evidence: Any = make_evidence_ref(
            evidence_id="reference",
            method_id="org.diffpes.method.test",
            source_type="analytic_reference",
            measured=jnp.array([2.0]),
            reference=jnp.array([1.0]),
            residual=jnp.array([0.0]),
            tolerance=jnp.array([1e-12]),
            lineage=make_evidence_lineage(
                implementation_refs=("reference.impl",),
                generator_refs=("reference.generator",),
                artifact_refs=(artifact.artifact_id,),
                derivation_refs=("reference.derivation",),
                relationship_ids=(
                    "independent-derivation:reference.derivation",
                ),
            ),
        )
        report: Any = verify_evidence(
            evidence,
            (artifact,),
            mapping_artifact_resolver({artifact.artifact_id: value}),
        )
        assert not bool(report.passed)
