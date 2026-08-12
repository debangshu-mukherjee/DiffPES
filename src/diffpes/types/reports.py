"""Define certification policy and verification reports.

Extended Summary
----------------
This module stores policy results, evidence checks, reproduction
comparisons, and bounded waiver records.

Routine Listings
----------------
:class:`EvidenceReport`
    Store the offline consistency outcome for one evidence record.
:class:`PolicyReport`
    Store a traced policy truth table for derived certification levels.
:class:`ReproductionReport`
    Store a numerical comparison from deliberate forward re-execution.
:class:`VerificationReport`
    Store an offline certificate-verification outcome.
:class:`WaiverRecord`
    Store a bounded policy-waiver declaration without changing claim status.
:class:`WaiverReport`
    Store the temporal validation outcome for one waiver.
:func:`make_evidence_report`
    Create an offline evidence-verification report.
:func:`make_policy_report`
    Create a validated policy truth table.
:func:`make_reproduction_report`
    Create a report comparing a result with its re-execution.
:func:`make_verification_report`
    Create an offline certificate-verification report.
:func:`make_waiver_record`
    Create a bounded policy-waiver declaration.
:func:`make_waiver_report`
    Create a temporal waiver-validation report.
"""

import equinox as eqx
from beartype import beartype
from beartype.typing import Any, Tuple
from jaxtyping import Array, Bool, Float64, jaxtyped

from .certification_validation import (
    _bool,
    _float,
    _nonnegative,
    _require_text,
    _text_tuple,
)


class PolicyReport(eqx.Module):
    """Store a traced policy truth table for derived certification levels.

    Carry scientific vocabulary separately from traced leaves.
    The record remains stable under JIT, VMAP, JVP, and VJP transforms.

    :see: :class:`~.test_reports.TestPolicyreport`

    Attributes
    ----------
    policy_id : str
        Policy id (**static** -- a compile-time constant; changing it
        triggers retracing).
    level_ids : Tuple[str, ...]
        Level ids (**static** -- a compile-time constant; changing it
        triggers retracing).
    required_claim_ids : Tuple[str, ...]
        Required claim ids (**static** -- a compile-time constant;
        changing it triggers retracing).
    claim_passed : Bool[Array, " n_claim"]
        Claim passed retained as a differentiable JAX leaf in the
        declared physical units.
    claim_checked : Bool[Array, " n_claim"]
        Claim checked retained as a differentiable JAX leaf in the
        declared physical units.
    claim_in_domain : Bool[Array, " n_claim"]
        Claim in domain retained as a differentiable JAX leaf in the
        declared physical units.
    achieved : Bool[Array, " n_level"]
        Achieved retained as a differentiable JAX leaf in the declared
        physical units.

    See Also
    --------
    make_policy_report : Validated factory for this type.
    """

    policy_id: str = eqx.field(static=True)
    level_ids: Tuple[str, ...] = eqx.field(static=True)
    required_claim_ids: Tuple[str, ...] = eqx.field(static=True)
    claim_passed: Bool[Array, " n_claim"]
    claim_checked: Bool[Array, " n_claim"]
    claim_in_domain: Bool[Array, " n_claim"]
    achieved: Bool[Array, " n_level"]


class EvidenceReport(eqx.Module):
    """Store the offline consistency outcome for one evidence record.

    Carry scientific vocabulary separately from traced leaves.
    The record remains stable under JIT, VMAP, JVP, and VJP transforms.

    :see: :class:`~.test_reports.TestEvidencereport`

    Attributes
    ----------
    evidence_id : str
        Evidence id (**static** -- a compile-time constant; changing it
        triggers retracing).
    resolved : Bool[Array, ""]
        Resolved retained as a differentiable JAX leaf in the declared
        physical units.
    compatible : Bool[Array, ""]
        Compatible retained as a differentiable JAX leaf in the
        declared physical units.
    passed : Bool[Array, ""]
        Passed retained as a differentiable JAX leaf in the declared
        physical units.
    residual_norm : Float64[Array, ""]
        Residual norm retained as a differentiable JAX leaf in the
        declared physical units.

    See Also
    --------
    make_evidence_report : Validated factory for this type.
    """

    evidence_id: str = eqx.field(static=True)
    resolved: Bool[Array, ""]
    compatible: Bool[Array, ""]
    passed: Bool[Array, ""]
    residual_norm: Float64[Array, ""]


class VerificationReport(eqx.Module):
    """Store an offline certificate-verification outcome.

    Carry scientific vocabulary separately from traced leaves.
    The record remains stable under JIT, VMAP, JVP, and VJP transforms.

    :see: :class:`~.test_reports.TestVerificationreport`

    Attributes
    ----------
    certificate_checksum : str
        Certificate checksum (**static** -- a compile-time constant;
        changing it triggers retracing).
    policy_id : str
        Policy id (**static** -- a compile-time constant; changing it
        triggers retracing).
    structure_valid : Bool[Array, ""]
        Structure valid retained as a differentiable JAX leaf in the
        declared physical units.
    evidence_valid : Bool[Array, ""]
        Evidence valid retained as a differentiable JAX leaf in the
        declared physical units.
    policy_report : PolicyReport
        Policy report retained as a differentiable JAX leaf in the
        declared physical units.

    See Also
    --------
    make_verification_report : Validated factory for this type.
    """

    certificate_checksum: str = eqx.field(static=True)
    policy_id: str = eqx.field(static=True)
    structure_valid: Bool[Array, ""]
    evidence_valid: Bool[Array, ""]
    policy_report: PolicyReport


class ReproductionReport(eqx.Module):
    """Store a numerical comparison from deliberate forward re-execution.

    Carry scientific vocabulary separately from traced leaves.
    The record remains stable under JIT, VMAP, JVP, and VJP transforms.

    :see: :class:`~.test_reports.TestReproductionreport`

    Attributes
    ----------
    execution_id : str
        Execution id (**static** -- a compile-time constant; changing
        it triggers retracing).
    result_checksum : str
        Result checksum (**static** -- a compile-time constant;
        changing it triggers retracing).
    reproduced : Bool[Array, ""]
        Reproduced retained as a differentiable JAX leaf in the
        declared physical units.
    max_abs_error : Float64[Array, ""]
        Max abs error retained as a differentiable JAX leaf in the
        declared physical units.
    max_rel_error : Float64[Array, ""]
        Max rel error retained as a differentiable JAX leaf in the
        declared physical units.
    tolerance : Float64[Array, ""]
        Tolerance retained as a differentiable JAX leaf in the declared
        physical units.

    See Also
    --------
    make_reproduction_report : Validated factory for this type.
    """

    execution_id: str = eqx.field(static=True)
    result_checksum: str = eqx.field(static=True)
    reproduced: Bool[Array, ""]
    max_abs_error: Float64[Array, ""]
    max_rel_error: Float64[Array, ""]
    tolerance: Float64[Array, ""]


class WaiverRecord(eqx.Module):
    """Store a bounded policy-waiver declaration without changing claim status.

    A waiver records review context only. It never changes a failed claim to
    a passed claim.

    :see: :class:`~.test_reports.TestWaiverRecord`

    Attributes
    ----------
    waiver_id : str
        Permanent waiver identity (**static**; changing it causes retracing).
    policy_id : str
        Applicable policy identity (**static**; changing it causes retracing).
    claim_ids : Tuple[str, ...]
        Affected claim identities (**static**; changing them causes retracing).
    author : str
        Responsible reviewer (**static**; changing it causes retracing).
    reason : str
        Technical reason (**static**; changing it causes retracing).
    issued_at_utc : str
        Absolute UTC issue time (**static**; changing it causes retracing).
    expires_at_utc : str
        Absolute UTC expiry time (**static**; changing it causes retracing).

    See Also
    --------
    make_waiver_record : Validated factory for this type.
    """

    waiver_id: str = eqx.field(static=True)
    policy_id: str = eqx.field(static=True)
    claim_ids: Tuple[str, ...] = eqx.field(static=True)
    author: str = eqx.field(static=True)
    reason: str = eqx.field(static=True)
    issued_at_utc: str = eqx.field(static=True)
    expires_at_utc: str = eqx.field(static=True)


class WaiverReport(eqx.Module):
    """Store the temporal validation outcome for one waiver.

    The report distinguishes valid structure from active temporal scope.

    :see: :class:`~.test_reports.TestWaiverReport`

    Attributes
    ----------
    waiver_id : str
        Permanent waiver identity (**static**; changing it causes retracing).
    valid : Bool[Array, ""]
        Whether the record has valid absolute UTC fields.
    active : Bool[Array, ""]
        Whether the waiver covers the selected UTC time.
    errors : Tuple[str, ...]
        Validation errors (**static**; changing them causes retracing).

    See Also
    --------
    make_waiver_report : Validated factory for this type.
    """

    waiver_id: str = eqx.field(static=True)
    valid: Bool[Array, ""]
    active: Bool[Array, ""]
    errors: Tuple[str, ...] = eqx.field(static=True)


@jaxtyped(typechecker=beartype)
def make_policy_report(
    policy_id: str,
    level_ids: Tuple[str, ...],
    required_claim_ids: Tuple[str, ...],
    claim_passed: Any,
    claim_checked: Any,
    claim_in_domain: Any,
    achieved: Any,
) -> PolicyReport:
    """Create a validated policy truth table.

    Carry static scientific vocabulary separately from traced numerical leaves
    while preserving the validation boundary defined by this factory.

    :see: :class:`~.test_reports.TestMakePolicyReport`

    Parameters
    ----------
    policy_id : str
        Policy id used to construct the validated carrier (**static**
        -- a compile-time constant; changing it triggers retracing).
    level_ids : Tuple[str, ...]
        Level ids used to construct the validated carrier (**static**
        -- a compile-time constant; changing it triggers retracing).
    required_claim_ids : Tuple[str, ...]
        Required claim ids used to construct the validated carrier
        (**static** -- a compile-time constant; changing it triggers
        retracing).
    claim_passed : Any
        Claim passed used to construct the validated carrier as a
        traced numerical value in the declared physical units.
    claim_checked : Any
        Claim checked used to construct the validated carrier as a
        traced numerical value in the declared physical units.
    claim_in_domain : Any
        Claim in domain used to construct the validated carrier as a
        traced numerical value in the declared physical units.
    achieved : Any
        Achieved used to construct the validated carrier as a traced
        numerical value in the declared physical units.

    Returns
    -------
    result : PolicyReport
        Validated immutable carrier.

    Raises
    ------
    ValueError
        If static structure or cross-record validation fails.

    Notes
    -----
    The factory checks static structure eagerly. JAX array operations validate
    numerical values and preserve differentiation behavior.
    """
    levels: Tuple[str, ...] = _text_tuple(level_ids, "level_ids")
    claims: Tuple[str, ...] = _text_tuple(
        required_claim_ids,
        "required_claim_ids",
    )
    passed_array: Bool[Array, " n_claim"] = _bool(
        claim_passed, "claim_passed", 1
    )
    checked_array: Bool[Array, " n_claim"] = _bool(
        claim_checked, "claim_checked", 1
    )
    domain_array: Bool[Array, " n_claim"] = _bool(
        claim_in_domain, "claim_in_domain", 1
    )
    achieved_array: Bool[Array, " n_level"] = _bool(achieved, "achieved", 1)
    if not (
        passed_array.shape
        == checked_array.shape
        == domain_array.shape
        == (len(claims),)
    ):
        raise ValueError("policy claim arrays must match required_claim_ids")
    if achieved_array.shape != (len(levels),):
        raise ValueError("achieved must match level_ids")
    result: PolicyReport = PolicyReport(
        policy_id=_require_text(policy_id, "policy_id"),
        level_ids=levels,
        required_claim_ids=claims,
        claim_passed=passed_array,
        claim_checked=checked_array,
        claim_in_domain=domain_array,
        achieved=achieved_array,
    )
    return result


@jaxtyped(typechecker=beartype)
def make_evidence_report(
    evidence_id: str,
    resolved: Any,
    compatible: Any,
    passed: Any,
    residual_norm: Any,
) -> EvidenceReport:
    """Create an offline evidence-verification report.

    Carry static scientific vocabulary separately from traced numerical leaves
    while preserving the validation boundary defined by this factory.

    :see: :class:`~.test_reports.TestMakeEvidenceReport`

    Parameters
    ----------
    evidence_id : str
        Evidence id used to construct the validated carrier (**static**
        -- a compile-time constant; changing it triggers retracing).
    resolved : Any
        Resolved used to construct the validated carrier as a traced
        numerical value in the declared physical units.
    compatible : Any
        Compatible used to construct the validated carrier as a traced
        numerical value in the declared physical units.
    passed : Any
        Passed used to construct the validated carrier as a traced
        numerical value in the declared physical units.
    residual_norm : Any
        Residual norm used to construct the validated carrier as a
        traced numerical value in the declared physical units.

    Returns
    -------
    result : EvidenceReport
        Validated immutable carrier.

    Notes
    -----
    The factory checks static structure eagerly. JAX array operations validate
    numerical values and preserve differentiation behavior.
    """
    result: EvidenceReport = EvidenceReport(
        evidence_id=_require_text(evidence_id, "evidence_id"),
        resolved=_bool(resolved, "resolved", 0),
        compatible=_bool(compatible, "compatible", 0),
        passed=_bool(passed, "passed", 0),
        residual_norm=_float(residual_norm, "residual_norm", 0),
    )
    return result


@jaxtyped(typechecker=beartype)
def make_verification_report(
    certificate_checksum: str,
    policy_id: str,
    structure_valid: Any,
    evidence_valid: Any,
    policy_report: PolicyReport,
) -> VerificationReport:
    """Create an offline certificate-verification report.

    Carry static scientific vocabulary separately from traced numerical leaves
    while preserving the validation boundary defined by this factory.

    :see: :class:`~.test_reports.TestMakeVerificationReport`

    Parameters
    ----------
    certificate_checksum : str
        Certificate checksum used to construct the validated carrier
        (**static** -- a compile-time constant; changing it triggers
        retracing).
    policy_id : str
        Policy id used to construct the validated carrier (**static**
        -- a compile-time constant; changing it triggers retracing).
    structure_valid : Any
        Structure valid used to construct the validated carrier as a
        traced numerical value in the declared physical units.
    evidence_valid : Any
        Evidence valid used to construct the validated carrier as a
        traced numerical value in the declared physical units.
    policy_report : PolicyReport
        Policy report used to construct the validated carrier as a
        traced numerical value in the declared physical units.

    Returns
    -------
    result : VerificationReport
        Validated immutable carrier.

    Raises
    ------
    ValueError
        If static structure or cross-record validation fails.

    Notes
    -----
    The factory checks static structure eagerly. JAX array operations validate
    numerical values and preserve differentiation behavior.
    """
    if policy_report.policy_id != policy_id:
        raise ValueError("policy_report policy_id does not match report")
    result: VerificationReport = VerificationReport(
        certificate_checksum=_require_text(
            certificate_checksum, "certificate_checksum"
        ),
        policy_id=_require_text(policy_id, "policy_id"),
        structure_valid=_bool(structure_valid, "structure_valid", 0),
        evidence_valid=_bool(evidence_valid, "evidence_valid", 0),
        policy_report=policy_report,
    )
    return result


@jaxtyped(typechecker=beartype)
def make_reproduction_report(
    execution_id: str,
    result_checksum: str,
    reproduced: Any,
    max_abs_error: Any,
    max_rel_error: Any,
    tolerance: Any,
) -> ReproductionReport:
    """Create a report comparing a result with its re-execution.

    Carry static scientific vocabulary separately from traced numerical leaves
    while preserving the validation boundary defined by this factory.

    :see: :class:`~.test_reports.TestMakeReproductionReport`

    Parameters
    ----------
    execution_id : str
        Execution id used to construct the validated carrier
        (**static** -- a compile-time constant; changing it triggers
        retracing).
    result_checksum : str
        Result checksum used to construct the validated carrier
        (**static** -- a compile-time constant; changing it triggers
        retracing).
    reproduced : Any
        Reproduced used to construct the validated carrier as a traced
        numerical value in the declared physical units.
    max_abs_error : Any
        Max abs error used to construct the validated carrier as a
        traced numerical value in the declared physical units.
    max_rel_error : Any
        Max rel error used to construct the validated carrier as a
        traced numerical value in the declared physical units.
    tolerance : Any
        Tolerance used to construct the validated carrier as a traced
        numerical value in the declared physical units.

    Returns
    -------
    result : ReproductionReport
        Validated immutable carrier.

    Notes
    -----
    The factory checks static structure eagerly. JAX array operations validate
    numerical values and preserve differentiation behavior.
    """
    result: ReproductionReport = ReproductionReport(
        execution_id=_require_text(execution_id, "execution_id"),
        result_checksum=_require_text(result_checksum, "result_checksum"),
        reproduced=_bool(reproduced, "reproduced", 0),
        max_abs_error=_float(max_abs_error, "max_abs_error", 0),
        max_rel_error=_float(max_rel_error, "max_rel_error", 0),
        tolerance=_nonnegative(_float(tolerance, "tolerance", 0), "tolerance"),
    )
    return result


@jaxtyped(typechecker=beartype)
def make_waiver_record(
    waiver_id: str,
    policy_id: str,
    claim_ids: Tuple[str, ...],
    author: str,
    reason: str,
    issued_at_utc: str,
    expires_at_utc: str,
) -> WaiverRecord:
    """Create a bounded policy-waiver declaration.

    The factory validates static vocabulary. The waiver validator checks the
    absolute UTC interval.

    :see: :class:`~.test_reports.TestMakeWaiverRecord`

    Parameters
    ----------
    waiver_id : str
        Permanent waiver identity (**static**; changing it causes retracing).
    policy_id : str
        Applicable policy identity (**static**; changing it causes retracing).
    claim_ids : Tuple[str, ...]
        Affected claim identities (**static**; changing them causes retracing).
    author : str
        Responsible reviewer (**static**; changing it causes retracing).
    reason : str
        Technical reason (**static**; changing it causes retracing).
    issued_at_utc : str
        Absolute UTC issue time (**static**; changing it causes retracing).
    expires_at_utc : str
        Absolute UTC expiry time (**static**; changing it causes retracing).

    Returns
    -------
    result : WaiverRecord
        Validated static waiver declaration.

    Raises
    ------
    ValueError
        If a required text field or claim identity is empty.

    Notes
    -----
    The factory validates static text before it constructs the waiver record.
    """
    claims: Tuple[str, ...] = _text_tuple(claim_ids, "claim_ids")
    if not claims:
        raise ValueError("claim_ids must contain at least one identity")
    result: WaiverRecord = WaiverRecord(
        waiver_id=_require_text(waiver_id, "waiver_id"),
        policy_id=_require_text(policy_id, "policy_id"),
        claim_ids=claims,
        author=_require_text(author, "author"),
        reason=_require_text(reason, "reason"),
        issued_at_utc=_require_text(issued_at_utc, "issued_at_utc"),
        expires_at_utc=_require_text(expires_at_utc, "expires_at_utc"),
    )
    return result


@jaxtyped(typechecker=beartype)
def make_waiver_report(
    waiver_id: str,
    valid: Any,
    active: Any,
    errors: Tuple[str, ...] = (),
) -> WaiverReport:
    """Create a temporal waiver-validation report.

    The report keeps validation outcomes as JAX Boolean leaves.

    :see: :class:`~.test_reports.TestMakeWaiverReport`

    Parameters
    ----------
    waiver_id : str
        Permanent waiver identity (**static**; changing it causes retracing).
    valid : Any
        Whether the record has valid absolute UTC fields.
    active : Any
        Whether the waiver covers the selected UTC time.
    errors : Tuple[str, ...]
        Validation errors. Default is an empty tuple.

    Returns
    -------
    result : WaiverReport
        Validated temporal outcome and errors.

    Notes
    -----
    The factory converts temporal outcomes to scalar JAX Boolean leaves.
    """
    result: WaiverReport = WaiverReport(
        waiver_id=_require_text(waiver_id, "waiver_id"),
        valid=_bool(valid, "valid", 0),
        active=_bool(active, "active", 0),
        errors=tuple(errors),
    )
    return result


__all__: list[str] = [
    "EvidenceReport",
    "PolicyReport",
    "ReproductionReport",
    "VerificationReport",
    "WaiverRecord",
    "WaiverReport",
    "make_evidence_report",
    "make_policy_report",
    "make_reproduction_report",
    "make_verification_report",
    "make_waiver_record",
    "make_waiver_report",
]
