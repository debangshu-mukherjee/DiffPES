"""Validate the evidence contracts.

The tests cover public carrier behavior, validation, and JAX
transformations for this implementation module.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest
from beartype.typing import Any
from jaxtyping import Array, Float64

import diffpes.types
from diffpes.types import (
    make_certification_claim,
    make_domain_result,
    make_evidence_lineage,
    make_evidence_ref,
    make_human_attestation_ref,
)
from tests._assertions import assert_rejects, assert_trees_close


class TestCertificationclaim:
    """Verify :class:`~diffpes.types.CertificationClaim`.

    The cases cover the public carrier or factory contract in JAX PyTrees.

    :see: :class:`~diffpes.types.CertificationClaim`
    """

    def test_public_symbol_has_expected_kind(self) -> None:
        """Expose ``CertificationClaim`` through its canonical types package
        path.

        The case uses explicit inputs in the supported certification regime.
        It checks the public result or the documented failure state.

        Notes
        -----
        The test compares the result with explicit numerical or structural
        assertions.
        """
        symbol: object = diffpes.types.CertificationClaim
        assert isinstance(symbol, type)
        assert issubclass(symbol, eqx.Module)


class TestEvidenceLineage:
    """Verify :class:`~diffpes.types.EvidenceLineage`.

    The cases cover the public carrier and its static JAX tree behavior.
    """

    def test_public_symbol_has_expected_kind(self) -> None:
        """Expose the lineage carrier through the types package.

        The case resolves the canonical public import and its carrier type.

        Notes
        -----
        The test checks the symbol directly without constructing authority.
        """
        symbol: object = diffpes.types.EvidenceLineage
        assert isinstance(symbol, type)
        assert issubclass(symbol, eqx.Module)


class TestHumanAttestationRef:
    """Verify :class:`~diffpes.types.HumanAttestationRef`.

    The cases cover the separate public human-review carrier.
    """

    def test_public_symbol_has_expected_kind(self) -> None:
        """Expose the attestation carrier through the types package.

        The case resolves the canonical public import and its carrier type.

        Notes
        -----
        The test checks the symbol without treating review as evidence.
        """
        symbol: object = diffpes.types.HumanAttestationRef
        assert isinstance(symbol, type)
        assert issubclass(symbol, eqx.Module)


class TestEvidenceref:
    """Verify :class:`~diffpes.types.EvidenceRef`.

    The cases cover the public carrier or factory contract in JAX PyTrees.

    :see: :class:`~diffpes.types.EvidenceRef`
    """

    def test_public_symbol_has_expected_kind(self) -> None:
        """Expose ``EvidenceRef`` through its canonical types package path.

        The case uses explicit inputs in the supported certification regime.
        It checks the public result or the documented failure state.

        Notes
        -----
        The test compares the result with explicit numerical or structural
        assertions.
        """
        symbol: object = diffpes.types.EvidenceRef
        assert isinstance(symbol, type)
        assert issubclass(symbol, eqx.Module)


class TestTransformationrecord:
    """Verify :class:`~diffpes.types.TransformationRecord`.

    The cases cover the public carrier or factory contract in JAX PyTrees.

    :see: :class:`~diffpes.types.TransformationRecord`
    """

    def test_public_symbol_has_expected_kind(self) -> None:
        """Expose ``TransformationRecord`` through its canonical types package
        path.

        The case uses explicit inputs in the supported certification regime.
        It checks the public result or the documented failure state.

        Notes
        -----
        The test compares the result with explicit numerical or structural
        assertions.
        """
        symbol: object = diffpes.types.TransformationRecord
        assert isinstance(symbol, type)
        assert issubclass(symbol, eqx.Module)


class TestMakeCertificationClaim:
    """Verify :func:`~diffpes.types.make_certification_claim`.

    The cases cover the public carrier or factory contract in JAX PyTrees.

    :see: :func:`~diffpes.types.make_certification_claim`
    """

    def test_public_symbol_has_expected_kind(self) -> None:
        """Expose ``make_certification_claim`` through its canonical types
        package path.

        The case uses explicit inputs in the supported certification regime.
        It checks the public result or the documented failure state.

        Notes
        -----
        The test compares the result with explicit numerical or structural
        assertions.
        """
        symbol: object = diffpes.types.make_certification_claim
        assert callable(symbol)

    def test_continuous_claim_evidence_is_differentiable(self) -> None:
        """Differentiate a smooth residual and margin through its factory.

        The case uses explicit inputs in the supported certification regime.
        It checks the public result or the documented failure state.

        Notes
        -----
        The test compares the result with explicit numerical or structural
        assertions.
        """
        value: Any
        expected: Any

        def objective(value: Float64[Array, ""]) -> Float64[Array, ""]:
            claim: Any
            claim = make_certification_claim(
                claim_id="claim",
                subject_id="subject",
                predicate_id="agreement",
                evidence_ids=(),
                measured=value[None],
                reference=jnp.ones(1),
                residual=(value - 1.0)[None],
                tolerance=jnp.full(1, 0.1),
                passed=jnp.abs(value - 1.0) <= 0.1,
                margin=0.1 - jnp.abs(value - 1.0),
            )
            result: Float64[Array, ""] = (
                jnp.sum(claim.residual**2) + claim.margin
            )
            return result

        value = jnp.asarray(1.25)
        expected = 2.0 * (value - 1.0) - 1.0
        assert_trees_close(jax.grad(objective)(value), expected)
        assert_trees_close(
            eqx.filter_jit(jax.grad(objective))(value), expected
        )


class TestMakeEvidenceLineage:
    """Verify :func:`~diffpes.types.make_evidence_lineage`.

    The cases cover construction without caller-supplied authority.
    """

    def test_factory_records_named_ancestry(self) -> None:
        """Record all named lineage categories without an authority flag.

        The case constructs a complete external lineage record.

        Notes
        -----
        The test compares static identifiers and confirms no Boolean shortcut.
        """
        lineage: Any = make_evidence_lineage(
            implementation_refs=("reference.impl",),
            generator_refs=("reference.generator",),
            artifact_refs=("reference.artifact",),
            derivation_refs=("reference.derivation",),
            relationship_ids=("independent-derivation:reference.derivation",),
        )
        assert lineage.generator_refs == ("reference.generator",)
        assert not hasattr(lineage, "independent")


class TestMakeHumanAttestationRef:
    """Verify :func:`~diffpes.types.make_human_attestation_ref`.

    The cases cover separate human-review construction and validation.
    """

    def test_factory_requires_review_scope(self) -> None:
        """Reject a human attestation that names no reviewed evidence.

        The case passes an empty review scope to the public factory.

        Notes
        -----
        The test checks the eager structural validation boundary.
        """
        with pytest.raises(ValueError, match="scope_ids must be non-empty"):
            make_human_attestation_ref(
                "attestation",
                "reviewer",
                (),
                "Reviewed.",
                "2026-07-24T00:00:00Z",
            )


class TestMakeEvidenceRef:
    """Verify :func:`~diffpes.types.make_evidence_ref`.

    The cases cover the public carrier or factory contract in JAX PyTrees.

    :see: :func:`~diffpes.types.make_evidence_ref`
    """

    def test_public_symbol_has_expected_kind(self) -> None:
        """Expose ``make_evidence_ref`` through its canonical types package
        path.

        The case uses explicit inputs in the supported certification regime.
        It checks the public result or the documented failure state.

        Notes
        -----
        The test compares the result with explicit numerical or structural
        assertions.
        """
        symbol: object = diffpes.types.make_evidence_ref
        assert callable(symbol)

    def test_factories_reject_bad_numerical_shapes_and_tolerances(
        self,
    ) -> None:
        """Reject malformed evidence eagerly and through compiled execution.

        The case uses explicit inputs in the supported certification regime.
        It checks the public result or the documented failure state.

        Notes
        -----
        The test compares the result with explicit numerical or structural
        assertions.
        """
        with pytest.raises(ValueError, match="equal shapes"):
            make_evidence_ref(
                evidence_id="evidence",
                method_id="method",
                source_type="analytic",
                measured=jnp.ones(2),
                reference=jnp.ones(1),
                residual=jnp.ones(2),
                tolerance=jnp.ones(2),
            )
        assert_rejects(
            make_domain_result,
            "domain",
            0.0,
            0.0,
            0.0,
            -1.0,
            0.0,
            False,
            match="tolerance must be finite and nonnegative",
        )

    def test_empty_evidence_is_rejected(self) -> None:
        """Reject evidence without a numerical measurement.

        Evidence verification requires a defined residual norm.

        Notes
        -----
        All four numerical vectors have the same invalid zero length.
        """
        empty: Any = jnp.asarray([], dtype=jnp.float64)
        with pytest.raises(ValueError, match="must not be empty"):
            make_evidence_ref(
                evidence_id="evidence",
                method_id="method",
                source_type="analytic",
                measured=empty,
                reference=empty,
                residual=empty,
                tolerance=empty,
            )


class TestMakeTransformationRecord:
    """Verify :func:`~diffpes.types.make_transformation_record`.

    The cases cover the public carrier or factory contract in JAX PyTrees.

    :see: :func:`~diffpes.types.make_transformation_record`
    """

    def test_public_symbol_has_expected_kind(self) -> None:
        """Expose ``make_transformation_record`` through its canonical types
        package path.

        The case uses explicit inputs in the supported certification regime.
        It checks the public result or the documented failure state.

        Notes
        -----
        The test compares the result with explicit numerical or structural
        assertions.
        """
        symbol: object = diffpes.types.make_transformation_record
        assert callable(symbol)
