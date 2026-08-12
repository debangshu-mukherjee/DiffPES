"""Define certification evidence and lineage records.

Extended Summary
----------------
This module stores transformations, evidence lineage, human review,
numerical evidence, and certification claims.

Routine Listings
----------------
:class:`CertificationClaim`
    Store a named claim and its continuous numerical evidence.
:class:`EvidenceLineage`
    Store named implementation, generator, artifact, and derivation lineage.
:class:`EvidenceRef`
    Store numerical evidence with static method and source identity.
:class:`HumanAttestationRef`
    Record a human review separately from computational evidence.
:class:`TransformationRecord`
    Store one transformation and its semantic information effects.
:func:`make_certification_claim`
    Create a claim retaining both continuous and discrete evidence.
:func:`make_evidence_lineage`
    Create named evidence lineage without asserting independence.
:func:`make_evidence_ref`
    Create validated vector-valued numerical evidence.
:func:`make_human_attestation_ref`
    Create a named human-review record.
:func:`make_transformation_record`
    Create a validated information-aware transformation record.
"""

import equinox as eqx
from beartype import beartype
from beartype.typing import Any, Tuple
from jaxtyping import Array, Bool, Float64, Int32, jaxtyped

from .certification_validation import (
    _bool,
    _float,
    _int,
    _nonnegative,
    _require_text,
    _text_tuple,
)


class TransformationRecord(eqx.Module):
    """Store one transformation and its semantic information effects.

    Carry scientific vocabulary separately from traced leaves.
    The record remains stable under JIT, VMAP, JVP, and VJP transforms.

    :see: :class:`~.test_evidence.TestTransformationrecord`

    Attributes
    ----------
    transformation_id : str
        Transformation id (**static** -- a compile-time constant;
        changing it triggers retracing).
    transformation_version : str
        Transformation version (**static** -- a compile-time constant;
        changing it triggers retracing).
    parent_ids : Tuple[str, ...]
        Parent ids (**static** -- a compile-time constant; changing it
        triggers retracing).
    output_ids : Tuple[str, ...]
        Output ids (**static** -- a compile-time constant; changing it
        triggers retracing).
    preserves : Tuple[str, ...]
        Preserves (**static** -- a compile-time constant; changing it
        triggers retracing).
    introduces : Tuple[str, ...]
        Introduces (**static** -- a compile-time constant; changing it
        triggers retracing).
    destroys : Tuple[str, ...]
        Destroys (**static** -- a compile-time constant; changing it
        triggers retracing).
    invalidates_claims : Tuple[str, ...]
        Invalidates claims (**static** -- a compile-time constant;
        changing it triggers retracing).
    parameters_checksum : str
        Parameters checksum (**static** -- a compile-time constant;
        changing it triggers retracing).

    See Also
    --------
    make_transformation_record : Validated factory for this type.
    """

    transformation_id: str = eqx.field(static=True)
    transformation_version: str = eqx.field(static=True)
    parent_ids: Tuple[str, ...] = eqx.field(static=True)
    output_ids: Tuple[str, ...] = eqx.field(static=True)
    preserves: Tuple[str, ...] = eqx.field(static=True)
    introduces: Tuple[str, ...] = eqx.field(static=True)
    destroys: Tuple[str, ...] = eqx.field(static=True)
    invalidates_claims: Tuple[str, ...] = eqx.field(static=True)
    parameters_checksum: str = eqx.field(static=True)


class EvidenceLineage(eqx.Module):
    """Store named implementation, generator, artifact, and derivation lineage.

    The record contains identifiers only. Policy derives independence relative
    to an implementation under test; no field stores a trusted Boolean.

    :see: :class:`~.test_evidence.TestEvidenceLineage`

    Attributes
    ----------
    implementation_refs : Tuple[str, ...]
        Implementations that produced or contributed to the evidence.
    generator_refs : Tuple[str, ...]
        Named generators or execution recipes.
    artifact_refs : Tuple[str, ...]
        Referenced immutable artifacts.
    derivation_refs : Tuple[str, ...]
        Named analytic or numerical derivations.
    conflict_refs : Tuple[str, ...]
        Known conflicts requiring an explicit resolution relationship.
    relationship_ids : Tuple[str, ...]
        Typed lineage relationships.

    See Also
    --------
    make_evidence_lineage : Validated factory for this type.
    """

    implementation_refs: Tuple[str, ...] = eqx.field(static=True)
    generator_refs: Tuple[str, ...] = eqx.field(static=True)
    artifact_refs: Tuple[str, ...] = eqx.field(static=True)
    derivation_refs: Tuple[str, ...] = eqx.field(static=True)
    conflict_refs: Tuple[str, ...] = eqx.field(static=True)
    relationship_ids: Tuple[str, ...] = eqx.field(static=True)


class HumanAttestationRef(eqx.Module):
    """Record a human review separately from computational evidence.

    Keep review identity and scope outside computational lineage authority.

    :see: :class:`~.test_evidence.TestHumanAttestationRef`

    Attributes
    ----------
    attestation_id : str
        Stable attestation identifier.
    reviewer_ref : str
        Named reviewer identity.
    scope_ids : Tuple[str, ...]
        Evidence or lineage identifiers reviewed.
    statement : str
        Review statement.
    recorded_at_utc : str
        Absolute UTC record time.

    See Also
    --------
    make_human_attestation_ref : Validated factory for this type.
    """

    attestation_id: str = eqx.field(static=True)
    reviewer_ref: str = eqx.field(static=True)
    scope_ids: Tuple[str, ...] = eqx.field(static=True)
    statement: str = eqx.field(static=True)
    recorded_at_utc: str = eqx.field(static=True)


class EvidenceRef(eqx.Module):
    """Store numerical evidence with static method and source identity.

    Carry scientific vocabulary separately from traced leaves.
    The record remains stable under JIT, VMAP, JVP, and VJP transforms.

    :see: :class:`~.test_evidence.TestEvidenceref`

    Attributes
    ----------
    evidence_id : str
        Evidence id (**static** -- a compile-time constant; changing it
        triggers retracing).
    method_id : str
        Method id (**static** -- a compile-time constant; changing it
        triggers retracing).
    source_type : str
        Source type (**static** -- a compile-time constant; changing it
        triggers retracing).
    lineage : EvidenceLineage
        Named computational and derivation ancestry.
    human_attestation_refs : Tuple[str, ...]
        Separate human-review references. These never establish independence.
    measured : Float64[Array, " n_measure"]
        Measured retained as a differentiable JAX leaf in the declared
        physical units.
    reference : Float64[Array, " n_measure"]
        Reference retained as a differentiable JAX leaf in the declared
        physical units.
    residual : Float64[Array, " n_measure"]
        Residual retained as a differentiable JAX leaf in the declared
        physical units.
    tolerance : Float64[Array, " n_measure"]
        Tolerance retained as a differentiable JAX leaf in the declared
        physical units.

    See Also
    --------
    make_evidence_ref : Validated factory for this type.
    """

    evidence_id: str = eqx.field(static=True)
    method_id: str = eqx.field(static=True)
    source_type: str = eqx.field(static=True)
    lineage: EvidenceLineage
    human_attestation_refs: Tuple[str, ...] = eqx.field(static=True)
    measured: Float64[Array, " n_measure"]
    reference: Float64[Array, " n_measure"]
    residual: Float64[Array, " n_measure"]
    tolerance: Float64[Array, " n_measure"]


class CertificationClaim(eqx.Module):
    """Store a named claim and its continuous numerical evidence.

    Carry scientific vocabulary separately from traced leaves.
    The record remains stable under JIT, VMAP, JVP, and VJP transforms.

    :see: :class:`~.test_evidence.TestCertificationclaim`

    Attributes
    ----------
    claim_id : str
        Claim id (**static** -- a compile-time constant; changing it
        triggers retracing).
    subject_id : str
        Subject id (**static** -- a compile-time constant; changing it
        triggers retracing).
    predicate_id : str
        Predicate id (**static** -- a compile-time constant; changing
        it triggers retracing).
    evidence_ids : Tuple[str, ...]
        Evidence ids (**static** -- a compile-time constant; changing
        it triggers retracing).
    measured : Float64[Array, " n_measure"]
        Measured retained as a differentiable JAX leaf in the declared
        physical units.
    reference : Float64[Array, " n_measure"]
        Reference retained as a differentiable JAX leaf in the declared
        physical units.
    residual : Float64[Array, " n_measure"]
        Residual retained as a differentiable JAX leaf in the declared
        physical units.
    tolerance : Float64[Array, " n_measure"]
        Tolerance retained as a differentiable JAX leaf in the declared
        physical units.
    passed : Bool[Array, ""]
        Passed retained as a differentiable JAX leaf in the declared
        physical units.
    checked : Bool[Array, ""]
        Checked retained as a differentiable JAX leaf in the declared
        physical units.
    in_domain : Bool[Array, ""]
        In domain retained as a differentiable JAX leaf in the declared
        physical units.
    margin : Float64[Array, ""]
        Margin retained as a differentiable JAX leaf in the declared
        physical units.
    severity_code : Int32[Array, ""]
        Severity code retained as a differentiable JAX leaf in the
        declared physical units.

    See Also
    --------
    make_certification_claim : Validated factory for this type.
    """

    claim_id: str = eqx.field(static=True)
    subject_id: str = eqx.field(static=True)
    predicate_id: str = eqx.field(static=True)
    evidence_ids: Tuple[str, ...] = eqx.field(static=True)
    measured: Float64[Array, " n_measure"]
    reference: Float64[Array, " n_measure"]
    residual: Float64[Array, " n_measure"]
    tolerance: Float64[Array, " n_measure"]
    passed: Bool[Array, ""]
    checked: Bool[Array, ""]
    in_domain: Bool[Array, ""]
    margin: Float64[Array, ""]
    severity_code: Int32[Array, ""]


@jaxtyped(typechecker=beartype)
def make_transformation_record(
    transformation_id: str,
    transformation_version: str,
    parent_ids: Tuple[str, ...],
    output_ids: Tuple[str, ...],
    preserves: Tuple[str, ...] = (),
    introduces: Tuple[str, ...] = (),
    destroys: Tuple[str, ...] = (),
    invalidates_claims: Tuple[str, ...] = (),
    parameters_checksum: str = "none",
) -> TransformationRecord:
    """Create a validated information-aware transformation record.

    Carry static scientific vocabulary separately from traced numerical leaves
    while preserving the validation boundary defined by this factory.

    :see: :class:`~.test_evidence.TestMakeTransformationRecord`

    Parameters
    ----------
    transformation_id : str
        Transformation id used to construct the validated carrier
        (**static** -- a compile-time constant; changing it triggers
        retracing).
    transformation_version : str
        Transformation version used to construct the validated carrier
        (**static** -- a compile-time constant; changing it triggers
        retracing).
    parent_ids : Tuple[str, ...]
        Parent ids used to construct the validated carrier (**static**
        -- a compile-time constant; changing it triggers retracing).
    output_ids : Tuple[str, ...]
        Output ids used to construct the validated carrier (**static**
        -- a compile-time constant; changing it triggers retracing).
    preserves : Tuple[str, ...]
        Preserves used to construct the validated carrier (**static**
        -- a compile-time constant; changing it triggers retracing).
    introduces : Tuple[str, ...]
        Introduces used to construct the validated carrier (**static**
        -- a compile-time constant; changing it triggers retracing).
    destroys : Tuple[str, ...]
        Destroys used to construct the validated carrier (**static** --
        a compile-time constant; changing it triggers retracing).
    invalidates_claims : Tuple[str, ...]
        Invalidates claims used to construct the validated carrier
        (**static** -- a compile-time constant; changing it triggers
        retracing).
    parameters_checksum : str
        Parameters checksum used to construct the validated carrier
        (**static** -- a compile-time constant; changing it triggers
        retracing).

    Returns
    -------
    result : TransformationRecord
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
    if not output_ids:
        raise ValueError("output_ids must be non-empty")
    result: TransformationRecord = TransformationRecord(
        transformation_id=_require_text(
            transformation_id, "transformation_id"
        ),
        transformation_version=_require_text(
            transformation_version, "transformation_version"
        ),
        parent_ids=_text_tuple(parent_ids, "parent_ids"),
        output_ids=_text_tuple(output_ids, "output_ids"),
        preserves=_text_tuple(preserves, "preserves"),
        introduces=_text_tuple(introduces, "introduces"),
        destroys=_text_tuple(destroys, "destroys"),
        invalidates_claims=_text_tuple(
            invalidates_claims, "invalidates_claims"
        ),
        parameters_checksum=_require_text(
            parameters_checksum, "parameters_checksum"
        ),
    )
    return result


@jaxtyped(typechecker=beartype)
def make_evidence_lineage(
    implementation_refs: Tuple[str, ...] = (),
    generator_refs: Tuple[str, ...] = (),
    artifact_refs: Tuple[str, ...] = (),
    derivation_refs: Tuple[str, ...] = (),
    conflict_refs: Tuple[str, ...] = (),
    relationship_ids: Tuple[str, ...] = (),
) -> EvidenceLineage:
    """Create named evidence lineage without asserting independence.

    Validate each named lineage category as static identifier tuples.

    :see: :class:`~.test_evidence.TestMakeEvidenceLineage`

    Parameters
    ----------
    implementation_refs : Tuple[str, ...]
        Contributing implementation identifiers.
    generator_refs : Tuple[str, ...]
        Generator or execution-recipe identifiers.
    artifact_refs : Tuple[str, ...]
        Immutable artifact identifiers.
    derivation_refs : Tuple[str, ...]
        Analytic or numerical derivation identifiers.
    conflict_refs : Tuple[str, ...]
        Known conflict identifiers.
    relationship_ids : Tuple[str, ...]
        Typed relationships such as ``derived-from:<identifier>``.

    Returns
    -------
    result : EvidenceLineage
        Validated static lineage record.

    Notes
    -----
    Empty categories remain explicit and fail policies that require complete
    independent lineage.
    """
    result: EvidenceLineage = EvidenceLineage(
        implementation_refs=_text_tuple(
            implementation_refs, "implementation_refs"
        ),
        generator_refs=_text_tuple(generator_refs, "generator_refs"),
        artifact_refs=_text_tuple(artifact_refs, "artifact_refs"),
        derivation_refs=_text_tuple(derivation_refs, "derivation_refs"),
        conflict_refs=_text_tuple(conflict_refs, "conflict_refs"),
        relationship_ids=_text_tuple(relationship_ids, "relationship_ids"),
    )
    return result


@jaxtyped(typechecker=beartype)
def make_human_attestation_ref(
    attestation_id: str,
    reviewer_ref: str,
    scope_ids: Tuple[str, ...],
    statement: str,
    recorded_at_utc: str,
) -> HumanAttestationRef:
    """Create a named human-review record.

    Validate review identity and scope without changing evidence authority.

    :see: :class:`~.test_evidence.TestMakeHumanAttestationRef`

    Parameters
    ----------
    attestation_id : str
        Stable attestation identifier.
    reviewer_ref : str
        Named reviewer identity.
    scope_ids : Tuple[str, ...]
        Evidence or lineage identifiers reviewed.
    statement : str
        Review statement.
    recorded_at_utc : str
        Absolute UTC record time.

    Returns
    -------
    result : HumanAttestationRef
        Validated static attestation record.

    Raises
    ------
    ValueError
        If the review scope is empty.

    Notes
    -----
    Policy evaluates this record separately from computational lineage.
    """
    if not scope_ids:
        raise ValueError("scope_ids must be non-empty")
    result: HumanAttestationRef = HumanAttestationRef(
        attestation_id=_require_text(attestation_id, "attestation_id"),
        reviewer_ref=_require_text(reviewer_ref, "reviewer_ref"),
        scope_ids=_text_tuple(scope_ids, "scope_ids"),
        statement=_require_text(statement, "statement"),
        recorded_at_utc=_require_text(recorded_at_utc, "recorded_at_utc"),
    )
    return result


@jaxtyped(typechecker=beartype)
def make_evidence_ref(
    evidence_id: str,
    method_id: str,
    source_type: str,
    measured: Any,
    reference: Any,
    residual: Any,
    tolerance: Any,
    *,
    lineage: EvidenceLineage | None = None,
    human_attestation_refs: Tuple[str, ...] = (),
) -> EvidenceRef:
    """Create validated vector-valued numerical evidence.

    Carry static scientific vocabulary separately from traced numerical leaves
    while preserving the validation boundary defined by this factory.

    :see: :class:`~.test_evidence.TestMakeEvidenceRef`

    Parameters
    ----------
    evidence_id : str
        Evidence id used to construct the validated carrier (**static**
        -- a compile-time constant; changing it triggers retracing).
    method_id : str
        Method id used to construct the validated carrier (**static**
        -- a compile-time constant; changing it triggers retracing).
    source_type : str
        Source type used to construct the validated carrier (**static**
        -- a compile-time constant; changing it triggers retracing).
    measured : Any
        Measured used to construct the validated carrier as a traced
        numerical value in the declared physical units.
    reference : Any
        Reference used to construct the validated carrier as a traced
        numerical value in the declared physical units.
    residual : Any
        Residual used to construct the validated carrier as a traced
        numerical value in the declared physical units.
    tolerance : Any
        Tolerance used to construct the validated carrier as a traced
        numerical value in the declared physical units.
    lineage : EvidenceLineage | None, optional
        Named ancestry. An omitted value creates explicitly incomplete
        lineage and cannot satisfy publication or parity policy.
    human_attestation_refs : Tuple[str, ...]
        Separate human-review references.

    Returns
    -------
    result : EvidenceRef
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
    measured_array: Float64[Array, " n_value"] = _float(
        measured, "measured", 1
    )
    reference_array: Float64[Array, " n_value"] = _float(
        reference, "reference", 1
    )
    residual_array: Float64[Array, " n_value"] = _float(
        residual, "residual", 1
    )
    tolerance_array: Float64[Array, " n_value"] = _nonnegative(
        _float(tolerance, "tolerance", 1), "tolerance"
    )
    shape: Tuple[int, ...] = measured_array.shape
    if measured_array.size == 0:
        raise ValueError("evidence numerical arrays must not be empty")
    if not (
        reference_array.shape
        == residual_array.shape
        == tolerance_array.shape
        == shape
    ):
        raise ValueError("evidence numerical arrays must have equal shapes")
    result: EvidenceRef = EvidenceRef(
        evidence_id=_require_text(evidence_id, "evidence_id"),
        method_id=_require_text(method_id, "method_id"),
        source_type=_require_text(source_type, "source_type"),
        lineage=(make_evidence_lineage() if lineage is None else lineage),
        human_attestation_refs=_text_tuple(
            human_attestation_refs, "human_attestation_refs"
        ),
        measured=measured_array,
        reference=reference_array,
        residual=residual_array,
        tolerance=tolerance_array,
    )
    return result


@jaxtyped(typechecker=beartype)
def make_certification_claim(  # noqa: PLR0913, PLR0917
    claim_id: str,
    subject_id: str,
    predicate_id: str,
    evidence_ids: Tuple[str, ...],
    measured: Any,
    reference: Any,
    residual: Any,
    tolerance: Any,
    passed: Any,
    checked: Any = True,
    in_domain: Any = True,
    margin: Any = 0.0,
    severity_code: Any = 0,
) -> CertificationClaim:
    """Create a claim retaining both continuous and discrete evidence.

    Carry static scientific vocabulary separately from traced numerical leaves
    while preserving the validation boundary defined by this factory.

    :see: :class:`~.test_evidence.TestMakeCertificationClaim`

    Parameters
    ----------
    claim_id : str
        Claim id used to construct the validated carrier (**static** --
        a compile-time constant; changing it triggers retracing).
    subject_id : str
        Subject id used to construct the validated carrier (**static**
        -- a compile-time constant; changing it triggers retracing).
    predicate_id : str
        Predicate id used to construct the validated carrier
        (**static** -- a compile-time constant; changing it triggers
        retracing).
    evidence_ids : Tuple[str, ...]
        Evidence ids used to construct the validated carrier
        (**static** -- a compile-time constant; changing it triggers
        retracing).
    measured : Any
        Measured used to construct the validated carrier as a traced
        numerical value in the declared physical units.
    reference : Any
        Reference used to construct the validated carrier as a traced
        numerical value in the declared physical units.
    residual : Any
        Residual used to construct the validated carrier as a traced
        numerical value in the declared physical units.
    tolerance : Any
        Tolerance used to construct the validated carrier as a traced
        numerical value in the declared physical units.
    passed : Any
        Passed used to construct the validated carrier as a traced
        numerical value in the declared physical units.
    checked : Any
        Checked used to construct the validated carrier as a traced
        numerical value in the declared physical units.
    in_domain : Any
        In domain used to construct the validated carrier as a traced
        numerical value in the declared physical units.
    margin : Any
        Margin used to construct the validated carrier as a traced
        numerical value in the declared physical units.
    severity_code : Any
        Severity code used to construct the validated carrier as a
        traced numerical value in the declared physical units.

    Returns
    -------
    result : CertificationClaim
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
    measured_array: Float64[Array, " n_value"] = _float(
        measured, "measured", 1
    )
    reference_array: Float64[Array, " n_value"] = _float(
        reference, "reference", 1
    )
    residual_array: Float64[Array, " n_value"] = _float(
        residual, "residual", 1
    )
    tolerance_array: Float64[Array, " n_value"] = _nonnegative(
        _float(tolerance, "tolerance", 1), "tolerance"
    )
    shape: Tuple[int, ...] = measured_array.shape
    if measured_array.size == 0:
        raise ValueError("claim numerical arrays must not be empty")
    if not (
        reference_array.shape
        == residual_array.shape
        == tolerance_array.shape
        == shape
    ):
        raise ValueError("claim numerical arrays must have equal shapes")
    result: CertificationClaim = CertificationClaim(
        claim_id=_require_text(claim_id, "claim_id"),
        subject_id=_require_text(subject_id, "subject_id"),
        predicate_id=_require_text(predicate_id, "predicate_id"),
        evidence_ids=_text_tuple(evidence_ids, "evidence_ids"),
        measured=measured_array,
        reference=reference_array,
        residual=residual_array,
        tolerance=tolerance_array,
        passed=_bool(passed, "passed", 0),
        checked=_bool(checked, "checked", 0),
        in_domain=_bool(in_domain, "in_domain", 0),
        margin=_float(margin, "margin", 0),
        severity_code=_int(severity_code, "severity_code", 0),
    )
    return result


__all__: list[str] = [
    "CertificationClaim",
    "EvidenceLineage",
    "EvidenceRef",
    "HumanAttestationRef",
    "TransformationRecord",
    "make_certification_claim",
    "make_evidence_lineage",
    "make_evidence_ref",
    "make_human_attestation_ref",
    "make_transformation_record",
]
