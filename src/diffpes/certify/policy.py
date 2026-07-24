"""Evaluate cumulative scientific-certification policies.

Extended Summary
----------------
The module derives certification levels from traced claim outcomes. It does
not store trusted labels. Policies select the required claim categories and
accumulate from identified through reproducible. A failed, unchecked, or
out-of-domain required claim prevents that level and every higher level.

Routine Listings
----------------
:func:`achieved_levels`
    Return certification level names achieved by a concrete report.
:func:`evaluate_policy`
    Derive cumulative certification outcomes from numerical claims.
:func:`evidence_is_independent`
    Derive lineage qualification relative to an implementation under test.
"""

import jax.numpy as jnp
from beartype import beartype
from beartype.typing import Any, Iterable
from jaxtyping import Array, jaxtyped

from diffpes.types import (
    CERTIFICATION_LEVEL_IDS,
    CERTIFICATION_LEVEL_PREFIXES,
    CERTIFICATION_POLICY_IDS,
    CERTIFICATION_POLICY_LEVEL_COUNT,
    CertificationClaim,
    EvidenceLineage,
    EvidenceRef,
    EvidenceReport,
    PolicyReport,
    WaiverRecord,
    make_policy_report,
)

_SHARED_RELATIONSHIPS: frozenset[str] = frozenset(
    {
        "copied-from",
        "derived-from",
        "imports-fixtures-from",
        "shares-generator",
        "shares-implementation",
        "wraps",
    }
)
_LINEAGE_RELATIONSHIPS: frozenset[str] = _SHARED_RELATIONSHIPS | {
    "independent-derivation",
    "resolves-conflict",
    "resolves-node",
}
_INDEPENDENT_CLAIM_PREFIXES: tuple[str, ...] = ("benchmark", "parity")


def _relationship(record: str) -> tuple[str, str] | None:
    """Parse one supported typed lineage relationship."""
    kind: str
    separator: str
    target: str
    kind, separator, target = record.partition(":")
    if (
        not separator
        or not kind
        or not target
        or kind not in _LINEAGE_RELATIONSHIPS
    ):
        return None
    result: tuple[str, str] = (kind, target)
    return result


@jaxtyped(typechecker=beartype)
def evidence_is_independent(
    evidence: EvidenceRef,
    implementation_ref: str,
    *,
    artifact_ids: tuple[str, ...] = (),
) -> bool:
    """Derive lineage qualification relative to an implementation under test.

    Evaluate completeness, shared ancestry, and conflicts from named lineage.

    :see: :class:`~.test_policy.TestEvidenceIsIndependent`

    Parameters
    ----------
    evidence : EvidenceRef
        Evidence whose complete ancestry is evaluated.
    implementation_ref : str
        Exact implementation identity under test.
    artifact_ids : tuple[str, ...]
        Artifact identities present in the evaluated certificate.

    Returns
    -------
    independent : bool
        Derived policy result. This is never accepted from a caller.

    Notes
    -----
    Every implementation, generator, and derivation node must have a typed
    ``resolves-node:<id>`` relationship, and every artifact must be present in
    the certificate. Shared ancestry fails. Conflict references fail unless
    each is named by ``resolves-conflict:<conflict-id>``.
    """
    lineage: EvidenceLineage = evidence.lineage
    complete: bool = all(
        (
            lineage.implementation_refs,
            lineage.generator_refs,
            lineage.artifact_refs,
            lineage.derivation_refs,
            lineage.relationship_ids,
        )
    )
    direct_refs: tuple[str, ...] = (
        *lineage.implementation_refs,
        *lineage.generator_refs,
        *lineage.derivation_refs,
    )
    parsed: tuple[tuple[str, str] | None, ...] = tuple(
        _relationship(item) for item in lineage.relationship_ids
    )
    if not complete or any(item is None for item in parsed):
        return False
    relationships: tuple[tuple[str, str], ...] = tuple(
        item for item in parsed if item is not None
    )
    resolved_nodes: frozenset[str] = frozenset(
        target for kind, target in relationships if kind == "resolves-node"
    )
    nodes_resolved: bool = all(node in resolved_nodes for node in direct_refs)
    artifacts_resolved: bool = set(lineage.artifact_refs).issubset(
        artifact_ids
    )
    shared: bool = implementation_ref in direct_refs or any(
        kind in _SHARED_RELATIONSHIPS and target == implementation_ref
        for kind, target in relationships
    )
    resolved_conflicts: frozenset[str] = frozenset(
        target for kind, target in relationships if kind == "resolves-conflict"
    )
    unresolved: bool = any(
        conflict not in resolved_conflicts
        for conflict in lineage.conflict_refs
    )
    independent: bool = (
        nodes_resolved and artifacts_resolved and not shared and not unresolved
    )
    return independent


def _required_indices(
    claims: tuple[CertificationClaim, ...], policy_id: str
) -> tuple[tuple[tuple[int, ...], ...], tuple[str, ...]]:
    """Select required claims for each cumulative policy level."""
    level_index: Any
    maximum_level: int = CERTIFICATION_POLICY_LEVEL_COUNT[policy_id]
    indices_by_level: list[tuple[int, ...]] = []
    required_ids: list[str] = []
    for level_index in range(len(CERTIFICATION_LEVEL_IDS)):
        if level_index >= maximum_level:
            indices_by_level.append(())
            continue
        prefixes: tuple[str, ...] = CERTIFICATION_LEVEL_PREFIXES[level_index]
        selected: tuple[int, ...] = tuple(
            index
            for index, claim in enumerate(claims)
            if claim.predicate_id.startswith(prefixes)
        )
        indices_by_level.append(selected)
        required_ids.extend(claims[index].claim_id for index in selected)
    result: tuple[tuple[tuple[int, ...], ...], tuple[str, ...]] = (
        tuple(indices_by_level),
        tuple(dict.fromkeys(required_ids)),
    )
    return result


@jaxtyped(typechecker=beartype)
def evaluate_policy(
    claims: Iterable[CertificationClaim],
    policy_id: str = "org.diffpes.policy.research.v1",
    *,
    evidence: tuple[EvidenceRef, ...] = (),
    evidence_reports: tuple[EvidenceReport, ...] = (),
    implementation_ref: str | None = None,
    artifact_ids: tuple[str, ...] = (),
    waivers: tuple[WaiverRecord, ...] = (),
) -> PolicyReport:
    """Derive cumulative certification outcomes from numerical claims.

    The cumulative policy derives named levels from explicit claims. It retains
    the claim truth table as JAX arrays.

    :see: :class:`~.test_policy.TestEvaluatePolicy`

    Parameters
    ----------
    claims : Iterable[CertificationClaim]
        Numerical claims evaluated for one forward execution.
    policy_id : str
        Built-in cumulative policy identity (**static** -- a change retraces).
    evidence : tuple[EvidenceRef, ...]
        Named evidence lineage used by publication and parity policy.
    evidence_reports : tuple[EvidenceReport, ...]
        Resolver-produced reports. Publication and parity cannot qualify
        without one passing report for every attached independent evidence.
    implementation_ref : str | None, optional
        Implementation under test. Required by publication and parity policy.
    artifact_ids : tuple[str, ...]
        Artifact identities present in the evaluated certificate.
    waivers : tuple[WaiverRecord, ...]
        Valid active waiver records. Default is an empty tuple.

    Returns
    -------
    report : PolicyReport
        Traced truth table and cumulative achieved-level vector.

    Raises
    ------
    ValueError
        If ``policy_id`` is not a registered built-in policy.

    Notes
    -----
    Required claim selections are static. Boolean outcomes are JAX leaves, so
    the policy computation remains compatible with ``jit`` and ``vmap``.
    A waiver never changes a claim outcome. Publication and parity policies do
    not achieve their final level when a waiver exists.
    """
    level_index: Any
    indices: Any
    if policy_id not in CERTIFICATION_POLICY_IDS:
        msg: str = f"unknown certification policy: {policy_id}"
        raise ValueError(msg)
    mismatched_waivers: tuple[str, ...] = tuple(
        waiver.waiver_id for waiver in waivers if waiver.policy_id != policy_id
    )
    if mismatched_waivers:
        msg = "waiver policy does not match selected policy: " + ", ".join(
            mismatched_waivers
        )
        raise ValueError(msg)
    claim_tuple: tuple[CertificationClaim, ...] = tuple(claims)
    selection: tuple[tuple[tuple[int, ...], ...], tuple[str, ...]] = (
        _required_indices(claim_tuple, policy_id)
    )
    indices_by_level: tuple[tuple[int, ...], ...] = selection[0]
    required_ids: tuple[str, ...] = selection[1]
    all_passed: Array = jnp.asarray(
        [claim.passed for claim in claim_tuple], dtype=jnp.bool_
    )
    all_checked: Array = jnp.asarray(
        [claim.checked for claim in claim_tuple], dtype=jnp.bool_
    )
    all_in_domain: Array = jnp.asarray(
        [claim.in_domain for claim in claim_tuple], dtype=jnp.bool_
    )
    valid_claim: Array = all_passed & all_checked & all_in_domain
    maximum_level: int = CERTIFICATION_POLICY_LEVEL_COUNT[policy_id]
    achieved_values: list[Array] = []
    cumulative: Array = jnp.asarray(True, dtype=jnp.bool_)
    for level_index, indices in enumerate(indices_by_level):
        if level_index >= maximum_level:
            level_passed: Array = jnp.asarray(False, dtype=jnp.bool_)
        elif not indices:
            level_passed = jnp.asarray(False, dtype=jnp.bool_)
        else:
            level_passed = jnp.all(valid_claim[jnp.asarray(indices)])
        cumulative = cumulative & level_passed
        achieved_values.append(cumulative)
    id_to_index: dict[str, int] = {
        claim.claim_id: index for index, claim in enumerate(claim_tuple)
    }
    required_indices: Array = jnp.asarray(
        [id_to_index[claim_id] for claim_id in required_ids], dtype=jnp.int32
    )
    claim_passed: Array = all_passed[required_indices]
    claim_checked: Array = all_checked[required_indices]
    claim_in_domain: Array = all_in_domain[required_indices]
    achieved: Array = jnp.stack(achieved_values)
    if policy_id in {
        "org.diffpes.policy.publication.v1",
        "org.diffpes.policy.parity.v1",
    }:
        evidence_by_id: dict[str, EvidenceRef] = {
            item.evidence_id: item for item in evidence
        }
        reports_by_id: dict[str, EvidenceReport] = {
            item.evidence_id: item for item in evidence_reports
        }
        independent_claims: tuple[CertificationClaim, ...] = tuple(
            claim
            for claim in claim_tuple
            if claim.predicate_id.startswith(_INDEPENDENT_CLAIM_PREFIXES)
        )
        lineage_qualified: bool = (
            bool(independent_claims)
            and (implementation_ref is not None)
            and all(
                bool(claim.evidence_ids)
                and all(
                    evidence_id in evidence_by_id
                    and evidence_id in reports_by_id
                    and bool(reports_by_id[evidence_id].resolved)
                    and bool(reports_by_id[evidence_id].compatible)
                    and bool(reports_by_id[evidence_id].passed)
                    and evidence_is_independent(
                        evidence_by_id[evidence_id],
                        implementation_ref,
                        artifact_ids=artifact_ids,
                    )
                    for evidence_id in claim.evidence_ids
                )
                for claim in independent_claims
            )
        )
        achieved = achieved.at[4:].set(
            achieved[4:] & jnp.asarray(lineage_qualified)
        )
    if waivers and policy_id in {
        "org.diffpes.policy.publication.v1",
        "org.diffpes.policy.parity.v1",
    }:
        achieved = achieved.at[-1].set(False)
    report: PolicyReport = make_policy_report(
        policy_id=policy_id,
        level_ids=CERTIFICATION_LEVEL_IDS,
        required_claim_ids=required_ids,
        claim_passed=claim_passed,
        claim_checked=claim_checked,
        claim_in_domain=claim_in_domain,
        achieved=achieved,
    )
    return report


@jaxtyped(typechecker=beartype)
def achieved_levels(report: PolicyReport) -> tuple[str, ...]:
    """Return certification level names achieved by a concrete report.

    The cumulative policy derives named levels from explicit claims. It retains
    the claim truth table as JAX arrays.

    :see: :class:`~.test_policy.TestAchievedLevels`

    Parameters
    ----------
    report : PolicyReport
        Concrete policy report inspected at the eager boundary.

    Returns
    -------
    levels : tuple[str, ...]
        Achieved level identities in cumulative policy order.

    Notes
    -----
    This eager inspection helper converts the traced Boolean vector to a
    Python tuple. Do not call the helper inside a compiled kernel.
    """
    levels: tuple[str, ...] = tuple(
        level
        for level, achieved in zip(
            report.level_ids, report.achieved.tolist(), strict=True
        )
        if achieved
    )
    return levels


__all__: list[str] = [
    "achieved_levels",
    "evaluate_policy",
    "evidence_is_independent",
]
