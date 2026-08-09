"""Validate cumulative scientific-certification policies.

The tests cover public behavior, differentiability, validation, and stable
scientific identity in the supported certification regime.
"""

import jax.numpy as jnp
import pytest
from beartype.typing import Any, Tuple

from diffpes.certify import (
    achieved_levels,
    evaluate_claim,
    evaluate_evidence,
    evaluate_policy,
    evidence_is_independent,
)
from diffpes.types import make_evidence_lineage, make_evidence_report


def _claim(
    name: Any,
    predicate: Any,
    passed: Any = True,
    *,
    evidence_ids: Tuple[str, ...] = (),
) -> Any:
    """PRIVATE: Evaluate one certification claim with a chosen outcome.

    Parameters
    ----------
    name : Any
        Claim identity string.
    predicate : Any
        Stable predicate identity string for the claim.
    passed : Any
        Desired outcome; True selects a measured vector that meets the
        zero tolerance.
    evidence_ids : Tuple[str, ...]
        Evidence identities attached to the claim.

    Returns
    -------
    claim : Any
        Evaluated claim record on the subject ``subject.test``.

    Notes
    -----
    A passing claim measures zeros against the zero reference; a failing
    claim measures ones, which exceeds the zero tolerance.
    """
    measured: Any
    measured = jnp.zeros(1) if passed else jnp.ones(1)
    return evaluate_claim(
        name,
        "subject.test",
        predicate,
        measured,
        jnp.zeros(1),
        jnp.zeros(1),
        evidence_ids=evidence_ids,
    )


class TestAchievedLevels:
    """Verify :func:`~diffpes.certify.achieved_levels`.

    The cases cover the public behavior in the supported certification regime.

    :see: :func:`~diffpes.certify.achieved_levels`
    """

    def test_exploratory_reaches_identified_and_validated(self) -> None:
        """Achieve the two exploratory levels from required claims.

        The case uses explicit inputs in the supported certification regime.
        It checks the public result or the documented failure state.

        Notes
        -----
        The test compares the result with explicit numerical or structural assertions.
        """
        report: Any
        report = evaluate_policy(
            (
                _claim("identity", "identity.model"),
                _claim("output", "output.finite"),
            ),
            "org.diffpes.policy.exploratory.v1",
        )
        assert achieved_levels(report) == ("identified", "validated")


class TestEvaluatePolicy:
    """Verify :func:`~diffpes.certify.evaluate_policy`.

    The cases cover the public behavior in the supported certification regime.

    :see: :func:`~diffpes.certify.evaluate_policy`
    """

    def test_failed_lower_level_blocks_higher_levels(self) -> None:
        """Make cumulative outcomes false above a failed identity claim.

        The case uses explicit inputs in the supported certification regime.
        It checks the public result or the documented failure state.

        Notes
        -----
        The test compares the result with explicit numerical or structural assertions.
        """
        report: Any
        report = evaluate_policy(
            (
                _claim("identity", "identity.model", passed=False),
                _claim("output", "output.finite"),
                _claim("derivative", "derivative.fd"),
                _claim("verify", "verification.closed_form"),
            )
        )
        assert not bool(jnp.any(report.achieved))


class TestEvidenceIsIndependent:
    """Verify :func:`~diffpes.certify.evidence_is_independent`.

    The cases cover derived lineage qualification and policy integration.

    :see: :func:`~diffpes.certify.evidence_is_independent`
    """

    @pytest.mark.parametrize(
        ("case", "expected"),
        (
            ("wrapper", False),
            ("renamed-copy", False),
            ("fixture-import", False),
            ("unresolved-conflict", False),
            ("attestation-only", False),
            ("disjoint-control", True),
        ),
    )
    def test_independence_is_derived_from_complete_lineage(
        self,
        case: str,
        expected: bool,
    ) -> None:
        """Reject five false authorities and accept one disjoint control.

        The matrix varies named ancestry while numerical evidence stays equal.

        Notes
        -----
        The test checks both derivation and publication-policy outcomes.
        """
        target: str = "tests.target"
        relationships: Tuple[str, ...] = (
            "resolves-node:reference.impl",
            "resolves-node:reference.generator",
            "resolves-node:reference.derivation",
            "independent-derivation:reference.derivation",
        )
        conflicts: Tuple[str, ...] = ()
        generators: Tuple[str, ...] = ("reference.generator",)
        derivations: Tuple[str, ...] = ("reference.derivation",)
        attestations: Tuple[str, ...] = ()
        if case == "wrapper":
            relationships = (
                "resolves-node:reference.impl",
                "resolves-node:reference.generator",
                "resolves-node:reference.derivation",
                f"wraps:{target}",
            )
        elif case == "renamed-copy":
            relationships = (
                "resolves-node:reference.impl",
                "resolves-node:reference.generator",
                "resolves-node:reference.derivation",
                f"copied-from:{target}",
            )
        elif case == "fixture-import":
            relationships = (
                "resolves-node:reference.impl",
                "resolves-node:reference.generator",
                "resolves-node:reference.derivation",
                f"imports-fixtures-from:{target}",
            )
        elif case == "unresolved-conflict":
            conflicts = ("conflict.analytic",)
        elif case == "attestation-only":
            generators = ()
            derivations = ()
            attestations = ("attestation.review",)
        lineage: Any = make_evidence_lineage(
            implementation_refs=("reference.impl",),
            generator_refs=generators,
            artifact_refs=("reference.artifact",),
            derivation_refs=derivations,
            conflict_refs=conflicts,
            relationship_ids=relationships,
        )
        evidence: Any = evaluate_evidence(
            f"evidence.{case}",
            "method.reference",
            jnp.zeros(1),
            jnp.zeros(1),
            jnp.zeros(1),
            lineage=lineage,
            human_attestation_refs=attestations,
        )
        claims: Tuple[Any, ...] = tuple(
            _claim(
                f"claim-{index}",
                predicate,
                evidence_ids=(
                    (evidence.evidence_id,)
                    if predicate == "benchmark.external"
                    else ()
                ),
            )
            for index, predicate in enumerate(
                (
                    "identity.model",
                    "output.finite",
                    "derivative.fd",
                    "verification.closed_form",
                    "benchmark.external",
                    "reproduction.environment",
                )
            )
        )
        report: Any = evaluate_policy(
            claims,
            "org.diffpes.policy.publication.v1",
            evidence=(evidence,),
            evidence_reports=(
                make_evidence_report(
                    evidence_id=evidence.evidence_id,
                    resolved=True,
                    compatible=True,
                    passed=True,
                    residual_norm=0.0,
                ),
            ),
            implementation_ref=target,
            artifact_ids=("reference.artifact",),
        )
        assert (
            evidence_is_independent(
                evidence,
                target,
                artifact_ids=("reference.artifact",),
            )
            is expected
        )
        assert bool(report.achieved[-1]) is expected

    def test_malformed_or_unresolved_lineage_cannot_qualify(self) -> None:
        """Reject malformed edges and missing artifact resolution.

        Exercise both malformed relationship syntax and unresolved artifacts.

        Notes
        -----
        Require independence qualification to remain false in both cases.
        """
        evidence: Any = evaluate_evidence(
            "evidence.malformed",
            "method.reference",
            jnp.zeros(1),
            jnp.zeros(1),
            jnp.zeros(1),
            lineage=make_evidence_lineage(
                implementation_refs=("reference.impl",),
                generator_refs=("reference.generator",),
                artifact_refs=("reference.artifact",),
                derivation_refs=("reference.derivation",),
                relationship_ids=("not-a-typed-edge",),
            ),
        )
        assert not evidence_is_independent(
            evidence,
            "tests.target",
            artifact_ids=("reference.artifact",),
        )
        resolved_edges: Any = evaluate_evidence(
            "evidence.unresolved-artifact",
            "method.reference",
            jnp.zeros(1),
            jnp.zeros(1),
            jnp.zeros(1),
            lineage=make_evidence_lineage(
                implementation_refs=("reference.impl",),
                generator_refs=("reference.generator",),
                artifact_refs=("reference.artifact",),
                derivation_refs=("reference.derivation",),
                relationship_ids=(
                    "resolves-node:reference.impl",
                    "resolves-node:reference.generator",
                    "resolves-node:reference.derivation",
                    "independent-derivation:reference.derivation",
                ),
            ),
        )
        assert not evidence_is_independent(
            resolved_edges,
            "tests.target",
            artifact_ids=(),
        )

    def test_unrelated_evidence_cannot_unlock_publication(self) -> None:
        """Bind lineage qualification to the benchmark claim's evidence IDs.

        Supply valid evidence whose identity does not match the target claim.

        Notes
        -----
        Require the publication policy to reject the unrelated evidence.
        """
        evidence: Any = evaluate_evidence(
            "evidence.unrelated",
            "method.reference",
            jnp.zeros(1),
            jnp.zeros(1),
            jnp.zeros(1),
            lineage=make_evidence_lineage(
                implementation_refs=("reference.impl",),
                generator_refs=("reference.generator",),
                artifact_refs=("reference.artifact",),
                derivation_refs=("reference.derivation",),
                relationship_ids=(
                    "resolves-node:reference.impl",
                    "resolves-node:reference.generator",
                    "resolves-node:reference.derivation",
                    "independent-derivation:reference.derivation",
                ),
            ),
        )
        claims: Tuple[Any, ...] = tuple(
            _claim(f"claim-{index}", predicate)
            for index, predicate in enumerate(
                (
                    "identity.model",
                    "output.finite",
                    "derivative.fd",
                    "verification.closed_form",
                    "benchmark.external",
                    "reproduction.environment",
                )
            )
        )
        report: Any = evaluate_policy(
            claims,
            "org.diffpes.policy.publication.v1",
            evidence=(evidence,),
            implementation_ref="tests.target",
            artifact_ids=("reference.artifact",),
        )
        assert not bool(report.achieved[-1])

    @pytest.mark.parametrize(
        ("resolved", "compatible"),
        ((False, False), (True, False)),
    )
    def test_unresolved_or_checksum_mismatched_artifact_blocks_publication(
        self,
        resolved: bool,
        compatible: bool,
    ) -> None:
        """Require a passing resolver report before publication can qualify.

        Exercise unresolved and checksum-incompatible artifact reports.

        Notes
        -----
        Require each nonpassing resolver state to block publication.
        """
        evidence: Any = evaluate_evidence(
            "evidence.resolver",
            "method.reference",
            jnp.zeros(1),
            jnp.zeros(1),
            jnp.zeros(1),
            lineage=make_evidence_lineage(
                implementation_refs=("reference.impl",),
                generator_refs=("reference.generator",),
                artifact_refs=("reference.artifact",),
                derivation_refs=("reference.derivation",),
                relationship_ids=(
                    "resolves-node:reference.impl",
                    "resolves-node:reference.generator",
                    "resolves-node:reference.derivation",
                    "independent-derivation:reference.derivation",
                ),
            ),
        )
        claims: Tuple[Any, ...] = tuple(
            _claim(
                f"claim-{index}",
                predicate,
                evidence_ids=(
                    (evidence.evidence_id,)
                    if predicate == "benchmark.external"
                    else ()
                ),
            )
            for index, predicate in enumerate(
                (
                    "identity.model",
                    "output.finite",
                    "derivative.fd",
                    "verification.closed_form",
                    "benchmark.external",
                    "reproduction.environment",
                )
            )
        )
        report: Any = evaluate_policy(
            claims,
            "org.diffpes.policy.publication.v1",
            evidence=(evidence,),
            evidence_reports=(
                make_evidence_report(
                    evidence_id=evidence.evidence_id,
                    resolved=resolved,
                    compatible=compatible,
                    passed=False,
                    residual_norm=0.0,
                ),
            ),
            implementation_ref="tests.target",
            artifact_ids=("reference.artifact",),
        )
        assert not bool(report.achieved[-1])
