"""Store JAX-native carriers for certified forward-model executions.

Extended Summary
----------------
The classes in this module separate static scientific vocabulary from traced
numerical evidence.  Identifiers, conventions, paths, and schema selections
are Equinox static fields; measurements, residuals, margins, sensitivities,
and status arrays remain ordinary JAX leaves.  This lets certification travel
through ``filter_jit``, ``vmap``, JVP, and VJP without moving bookkeeping into
the forward-model kernel.

Routine Listings
----------------
:class:`ArtifactRef`
    Store static identity and role for one source or derived artifact.
:obj:`ArtifactResolver`
    Resolve an artifact to normalized content and optional source bytes.
:class:`CertificationClaim`
    Store a named claim and its continuous numerical evidence.
:class:`CertificationContext`
    Store prepared selections and references for compiled certification.
:class:`CertifiedResult`
    Store a numerical result paired with its differentiable certificate.
:obj:`CheckFunction`
    Callable signature for a pure JAX certification check.
:class:`ConventionRef`
    Store a versioned semantic convention used by a scientific model.
:class:`DependencyMap`
    Store declared and JAXPR-observed dependency relations.
:class:`DerivativeEvidence`
    Store JVP, VJP, reference, and information-spectrum evidence.
:class:`DomainPredicate`
    Store a static declaration of one model-domain predicate.
:class:`DomainResult`
    Store the traced evaluation of one declared domain predicate.
:class:`EvidenceLineage`
    Store named implementation, generator, artifact, and derivation lineage.
:class:`EvidenceRef`
    Store numerical evidence with static method and source identity.
:class:`EvidenceReport`
    Store the offline consistency outcome for one evidence record.
:class:`ExecutionManifest`
    Store software and execution identity prepared at the I/O boundary.
:class:`ForwardCertificate`
    Store the complete assurance record for one forward execution.
:class:`ForwardModelSpec`
    Store the identity of a differentiable forward model.
:class:`HumanAttestationRef`
    Record a human review separately from computational evidence.
:class:`InformationSpectrum`
    Store a matrix-free information spectrum in input coordinates.
:class:`PolicyReport`
    Store a traced policy truth table for derived certification levels.
:class:`RegisteredModel`
    Store a frozen binding between a model spec and its executor.
:class:`RegisteredTransformation`
    Store a frozen transformation and its consistency checksum.
:class:`RegistryReport`
    Store the structural validation result for one registry snapshot.
:class:`RegistrySnapshot`
    Store an immutable deterministic snapshot of registry entries.
:class:`RegistrationHandshake`
    Store declarative registration requirements for one plan owner.
:class:`ReproductionReport`
    Store a numerical comparison from deliberate forward re-execution.
:class:`HandshakeReport`
    Store the validation outcome for one registration handshake.
:class:`SensitivityMap`
    Store scaled sensitivities from inputs to output projections.
:class:`TransformationRecord`
    Store one transformation and its semantic information effects.
:class:`VerificationReport`
    Store an offline certificate-verification outcome.
:class:`WaiverRecord`
    Store a bounded policy-waiver declaration without changing claim status.
:class:`WaiverReport`
    Store the temporal validation outcome for one waiver.
:func:`make_artifact_ref`
    Create a validated artifact reference.
:func:`make_certification_claim`
    Create a claim retaining both continuous and discrete evidence.
:func:`make_certification_context`
    Create a prepared certification context.
:func:`make_certified_result`
    Pair any JAX-compatible result value with a forward certificate.
:func:`make_convention_ref`
    Create a validated convention reference.
:func:`make_dependency_map`
    Create a structural dependency map.
:func:`make_derivative_evidence`
    Create validated derivative and local-information evidence.
:func:`make_domain_predicate`
    Create a validated domain-predicate declaration.
:func:`make_domain_result`
    Create one traced domain evaluation.
:func:`make_evidence_lineage`
    Create named evidence lineage without asserting independence.
:func:`make_evidence_ref`
    Create validated vector-valued numerical evidence.
:func:`make_evidence_report`
    Create an offline evidence-verification report.
:func:`make_execution_manifest`
    Create a validated execution manifest.
:func:`make_forward_certificate`
    Create and cross-validate a complete forward certificate.
:func:`make_forward_model_spec`
    Create a validated stable forward-model specification.
:func:`make_human_attestation_ref`
    Create a named human-review record.
:func:`make_information_spectrum`
    Create a validated local information spectrum.
:func:`make_policy_report`
    Create a validated policy truth table.
:func:`make_registered_model`
    Create a validated model-registry binding.
:func:`make_registered_transformation`
    Create a validated transformation-registry binding.
:func:`make_registry_report`
    Create a validated structural registry report.
:func:`make_registry_snapshot`
    Create an immutable registry snapshot.
:func:`make_registration_handshake`
    Create declarative registration requirements for one plan owner.
:func:`make_reproduction_report`
    Create a report comparing a result with its re-execution.
:func:`make_handshake_report`
    Create a report for one registration handshake.
:func:`make_sensitivity_map`
    Create a named, scaled local-sensitivity map.
:func:`make_transformation_record`
    Create a validated information-aware transformation record.
:func:`make_verification_report`
    Create an offline certificate-verification report.
:func:`make_waiver_record`
    Create a bounded policy-waiver declaration.
:func:`make_waiver_report`
    Create a temporal waiver-validation report.
"""

import json

import equinox as eqx
import jax.numpy as jnp
from beartype import beartype
from beartype.typing import Any, Callable, Optional, Tuple
from jaxtyping import Array, Bool, Float64, Int32, PyTree, jaxtyped

from .contracts import TransformationContract


class ArtifactRef(eqx.Module):
    """Store static identity and role for one source or derived artifact.

    Carry scientific vocabulary separately from traced leaves.
    The record remains stable under JIT, VMAP, JVP, and VJP transforms.

    :see: :class:`~.test_certification.TestArtifactref`

    Attributes
    ----------
    artifact_id : str
        Artifact id (**static** -- a compile-time constant; changing it
        triggers retracing).
    media_type : str
        Media type (**static** -- a compile-time constant; changing it
        triggers retracing).
    byte_checksum : Optional[str]
        Byte checksum (**static** -- a compile-time constant; changing
        it triggers retracing).
    content_checksum : str
        Content checksum (**static** -- a compile-time constant;
        changing it triggers retracing).
    semantic_checksum : str
        Semantic checksum (**static** -- a compile-time constant;
        changing it triggers retracing).
    locator : Optional[str]
        Locator (**static** -- a compile-time constant; changing it
        triggers retracing).
    role : str
        Role (**static** -- a compile-time constant; changing it
        triggers retracing).
    """

    artifact_id: str = eqx.field(static=True)
    media_type: str = eqx.field(static=True)
    byte_checksum: Optional[str] = eqx.field(static=True)
    content_checksum: str = eqx.field(static=True)
    semantic_checksum: str = eqx.field(static=True)
    locator: Optional[str] = eqx.field(static=True)
    role: str = eqx.field(static=True)


type ArtifactResolver = Callable[[ArtifactRef], tuple[Any, Optional[bytes]]]


class ConventionRef(eqx.Module):
    """Store a versioned semantic convention used by a scientific model.

    Carry scientific vocabulary separately from traced leaves.
    The record remains stable under JIT, VMAP, JVP, and VJP transforms.

    :see: :class:`~.test_certification.TestConventionref`

    Attributes
    ----------
    convention_id : str
        Convention id (**static** -- a compile-time constant; changing
        it triggers retracing).
    version : str
        Version (**static** -- a compile-time constant; changing it
        triggers retracing).
    parameters_json : str
        Parameters json (**static** -- a compile-time constant;
        changing it triggers retracing).
    """

    convention_id: str = eqx.field(static=True)
    version: str = eqx.field(static=True)
    parameters_json: str = eqx.field(static=True)


class DomainPredicate(eqx.Module):
    """Store a static declaration of one model-domain predicate.

    Carry scientific vocabulary separately from traced leaves.
    The record remains stable under JIT, VMAP, JVP, and VJP transforms.

    :see: :class:`~.test_certification.TestDomainpredicate`

    Attributes
    ----------
    predicate_id : str
        Predicate id (**static** -- a compile-time constant; changing
        it triggers retracing).
    expression_id : str
        Expression id (**static** -- a compile-time constant; changing
        it triggers retracing).
    units : Optional[str]
        Units (**static** -- a compile-time constant; changing it
        triggers retracing).
    severity : str
        Severity (**static** -- a compile-time constant; changing it
        triggers retracing).
    """

    predicate_id: str = eqx.field(static=True)
    expression_id: str = eqx.field(static=True)
    units: Optional[str] = eqx.field(static=True)
    severity: str = eqx.field(static=True)


class DomainResult(eqx.Module):
    """Store the traced evaluation of one declared domain predicate.

    Carry scientific vocabulary separately from traced leaves.
    The record remains stable under JIT, VMAP, JVP, and VJP transforms.

    :see: :class:`~.test_certification.TestDomainresult`

    Attributes
    ----------
    predicate_id : str
        Predicate id (**static** -- a compile-time constant; changing
        it triggers retracing).
    measured : Float64[Array, ""]
        Measured retained as a differentiable JAX leaf in the declared
        physical units.
    reference : Float64[Array, ""]
        Reference retained as a differentiable JAX leaf in the declared
        physical units.
    residual : Float64[Array, ""]
        Residual retained as a differentiable JAX leaf in the declared
        physical units.
    tolerance : Float64[Array, ""]
        Tolerance retained as a differentiable JAX leaf in the declared
        physical units.
    margin : Float64[Array, ""]
        Margin retained as a differentiable JAX leaf in the declared
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
    severity_code : Int32[Array, ""]
        Severity code retained as a differentiable JAX leaf in the
        declared physical units.
    """

    predicate_id: str = eqx.field(static=True)
    measured: Float64[Array, ""]
    reference: Float64[Array, ""]
    residual: Float64[Array, ""]
    tolerance: Float64[Array, ""]
    margin: Float64[Array, ""]
    passed: Bool[Array, ""]
    checked: Bool[Array, ""]
    in_domain: Bool[Array, ""]
    severity_code: Int32[Array, ""]


type CheckFunction = Callable[[PyTree], DomainResult]


class ForwardModelSpec(eqx.Module):
    """Store the identity of a differentiable forward model.

    Carry scientific vocabulary separately from traced leaves.
    The record remains stable under JIT, VMAP, JVP, and VJP transforms.

    :see: :class:`~.test_certification.TestForwardmodelspec`

    Attributes
    ----------
    model_id : str
        Model id (**static** -- a compile-time constant; changing it
        triggers retracing).
    model_version : str
        Model version (**static** -- a compile-time constant; changing
        it triggers retracing).
    observable_id : str
        Observable id (**static** -- a compile-time constant; changing
        it triggers retracing).
    implementation_ref : str
        Implementation ref (**static** -- a compile-time constant;
        changing it triggers retracing).
    assumptions : Tuple[str, ...]
        Assumptions (**static** -- a compile-time constant; changing it
        triggers retracing).
    conventions : Tuple[ConventionRef, ...]
        Conventions (**static** -- a compile-time constant; changing it
        triggers retracing).
    domain : Tuple[DomainPredicate, ...]
        Domain (**static** -- a compile-time constant; changing it
        triggers retracing).
    differentiable_paths : Tuple[str, ...]
        Differentiable paths (**static** -- a compile-time constant;
        changing it triggers retracing).
    nondifferentiable_paths : Tuple[str, ...]
        Nondifferentiable paths (**static** -- a compile-time constant;
        changing it triggers retracing).
    """

    model_id: str = eqx.field(static=True)
    model_version: str = eqx.field(static=True)
    observable_id: str = eqx.field(static=True)
    implementation_ref: str = eqx.field(static=True)
    assumptions: Tuple[str, ...] = eqx.field(static=True)
    conventions: Tuple[ConventionRef, ...] = eqx.field(static=True)
    domain: Tuple[DomainPredicate, ...] = eqx.field(static=True)
    differentiable_paths: Tuple[str, ...] = eqx.field(static=True)
    nondifferentiable_paths: Tuple[str, ...] = eqx.field(static=True)


class RegisteredModel(eqx.Module):
    """Store a frozen binding between a model spec and its executor.

    Carry scientific vocabulary separately from traced leaves.
    The record remains stable under JIT, VMAP, JVP, and VJP transforms.

    :see: :class:`~.test_certification.TestRegisteredmodel`

    Attributes
    ----------
    spec : ForwardModelSpec
        Spec retained as a differentiable JAX leaf in the declared
        physical units.
    executor : Callable[..., Any]
        Executor (**static** -- a compile-time constant; changing it
        triggers retracing).
    registration_checksum : str
        Registration checksum (**static** -- a compile-time constant;
        changing it triggers retracing).
    """

    spec: ForwardModelSpec
    executor: Callable[..., Any] = eqx.field(static=True)
    registration_checksum: str = eqx.field(static=True)


class RegisteredTransformation(eqx.Module):
    """Store a frozen transformation and its consistency checksum.

    Carry scientific vocabulary separately from traced leaves.
    The record remains stable under JIT, VMAP, JVP, and VJP transforms.

    :see: :class:`~.test_certification.TestRegisteredtransformation`

    Attributes
    ----------
    contract : TransformationContract
        Contract retained as a differentiable JAX leaf in the declared
        physical units.
    registration_checksum : str
        Registration checksum (**static** -- a compile-time constant;
        changing it triggers retracing).
    """

    contract: TransformationContract
    registration_checksum: str = eqx.field(static=True)


class RegistrySnapshot(eqx.Module):
    """Store an immutable deterministic snapshot of registry entries.

    Carry scientific vocabulary separately from traced leaves.
    The record remains stable under JIT, VMAP, JVP, and VJP transforms.

    :see: :class:`~.test_certification.TestRegistrysnapshot`

    Attributes
    ----------
    models : Tuple[RegisteredModel, ...]
        Models retained as a differentiable JAX leaf in the declared
        physical units.
    transformations : Tuple[RegisteredTransformation, ...]
        Transformations retained as a differentiable JAX leaf in the
        declared physical units.
    checksum : str
        Checksum (**static** -- a compile-time constant; changing it
        triggers retracing).
    """

    models: Tuple[RegisteredModel, ...]
    transformations: Tuple[RegisteredTransformation, ...]
    checksum: str = eqx.field(static=True)


class RegistryReport(eqx.Module):
    """Store the structural validation result for one registry snapshot.

    Carry scientific vocabulary separately from traced leaves.
    The record remains stable under JIT, VMAP, JVP, and VJP transforms.

    :see: :class:`~.test_certification.TestRegistryreport`

    Attributes
    ----------
    valid : bool
        Valid (**static** -- a compile-time constant; changing it
        triggers retracing).
    errors : Tuple[str, ...]
        Errors (**static** -- a compile-time constant; changing it
        triggers retracing).
    model_count : int
        Model count (**static** -- a compile-time constant; changing it
        triggers retracing).
    transformation_count : int
        Transformation count (**static** -- a compile-time constant;
        changing it triggers retracing).
    checksum : str
        Checksum (**static** -- a compile-time constant; changing it
        triggers retracing).
    frozen : bool
        Frozen (**static** -- a compile-time constant; changing it
        triggers retracing).
    """

    valid: bool = eqx.field(static=True)
    errors: Tuple[str, ...] = eqx.field(static=True)
    model_count: int = eqx.field(static=True)
    transformation_count: int = eqx.field(static=True)
    checksum: str = eqx.field(static=True)
    frozen: bool = eqx.field(static=True)


class RegistrationHandshake(eqx.Module):
    """Store declarative registration requirements for one plan owner.

    The record names required identities without importing an unfinished
    scientific kernel.

    :see: :class:`~.test_certification.TestRegistrationHandshake`

    Attributes
    ----------
    owner_id : str
        Plan owner identity (**static**; changing it causes retracing).
    model_refs : Tuple[str, ...]
        Required model identities (**static**; changing them causes retracing).
    transformation_refs : Tuple[str, ...]
        Required transformation identities (**static**; changes cause
        retracing).
    convention_refs : Tuple[str, ...]
        Required convention identities (**static**; changes cause retracing).
    evidence_ids : Tuple[str, ...]
        Required evidence identities (**static**; changing them causes
        retracing).
    """

    owner_id: str = eqx.field(static=True)
    model_refs: Tuple[str, ...] = eqx.field(static=True)
    transformation_refs: Tuple[str, ...] = eqx.field(static=True)
    convention_refs: Tuple[str, ...] = eqx.field(static=True)
    evidence_ids: Tuple[str, ...] = eqx.field(static=True)


class HandshakeReport(eqx.Module):
    """Store the validation outcome for one registration handshake.

    The report keeps a JAX Boolean outcome and static missing identities.

    :see: :class:`~.test_certification.TestHandshakeReport`

    Attributes
    ----------
    owner_id : str
        Plan owner identity (**static**; changing it causes retracing).
    complete : Bool[Array, ""]
        Whether every declared identity has a registry binding.
    missing_ids : Tuple[str, ...]
        Missing declared identities (**static**; changes cause retracing).
    """

    owner_id: str = eqx.field(static=True)
    complete: Bool[Array, ""]
    missing_ids: Tuple[str, ...] = eqx.field(static=True)


class TransformationRecord(eqx.Module):
    """Store one transformation and its semantic information effects.

    Carry scientific vocabulary separately from traced leaves.
    The record remains stable under JIT, VMAP, JVP, and VJP transforms.

    :see: :class:`~.test_certification.TestTransformationrecord`

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

    :see: :class:`~.test_certification.TestEvidenceLineage`

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

    :see: :class:`~.test_certification.TestHumanAttestationRef`

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

    :see: :class:`~.test_certification.TestEvidenceref`

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

    :see: :class:`~.test_certification.TestCertificationclaim`

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


class DerivativeEvidence(eqx.Module):
    """Store JVP, VJP, reference, and information-spectrum evidence.

    Carry scientific vocabulary separately from traced leaves.
    The record remains stable under JIT, VMAP, JVP, and VJP transforms.

    :see: :class:`~.test_certification.TestDerivativeevidence`

    Attributes
    ----------
    input_paths : Tuple[str, ...]
        Input paths (**static** -- a compile-time constant; changing it
        triggers retracing).
    output_projection_ids : Tuple[str, ...]
        Output projection ids (**static** -- a compile-time constant;
        changing it triggers retracing).
    method : str
        Method (**static** -- a compile-time constant; changing it
        triggers retracing).
    scales : Float64[Array, " n_input"]
        Scales retained as a differentiable JAX leaf in the declared
        physical units.
    jvp_probes : Float64[Array, "n_probe n_output"]
        Jvp probes retained as a differentiable JAX leaf in the
        declared physical units.
    vjp_probes : Float64[Array, "n_probe n_input"]
        Vjp probes retained as a differentiable JAX leaf in the
        declared physical units.
    reference_derivatives : Float64[Array, "n_probe n_deriv"]
        Reference derivatives retained as a differentiable JAX leaf in
        the declared physical units.
    derivative_residuals : Float64[Array, "n_probe n_deriv"]
        Derivative residuals retained as a differentiable JAX leaf in
        the declared physical units.
    singular_values : Float64[Array, " n_sv"]
        Singular values retained as a differentiable JAX leaf in the
        declared physical units.
    effective_rank : Int32[Array, ""]
        Effective rank retained as a differentiable JAX leaf in the
        declared physical units.
    condition_estimate : Float64[Array, ""]
        Condition estimate retained as a differentiable JAX leaf in the
        declared physical units. Zero means that there is no active information
        direction.
    finite : Bool[Array, ""]
        Finite retained as a differentiable JAX leaf in the declared
        physical units.
    fd_correct : Bool[Array, ""]
        Fd correct retained as a differentiable JAX leaf in the
        declared physical units.
    """

    input_paths: Tuple[str, ...] = eqx.field(static=True)
    output_projection_ids: Tuple[str, ...] = eqx.field(static=True)
    method: str = eqx.field(static=True)
    scales: Float64[Array, " n_input"]
    jvp_probes: Float64[Array, "n_probe n_output"]
    vjp_probes: Float64[Array, "n_probe n_input"]
    reference_derivatives: Float64[Array, "n_probe n_deriv"]
    derivative_residuals: Float64[Array, "n_probe n_deriv"]
    singular_values: Float64[Array, " n_sv"]
    effective_rank: Int32[Array, ""]
    condition_estimate: Float64[Array, ""]
    finite: Bool[Array, ""]
    fd_correct: Bool[Array, ""]


class DependencyMap(eqx.Module):
    """Store declared and JAXPR-observed dependency relations.

    Carry scientific vocabulary separately from traced leaves.
    The record remains stable under JIT, VMAP, JVP, and VJP transforms.

    :see: :class:`~.test_certification.TestDependencymap`

    Attributes
    ----------
    model_id : str
        Model id (**static** -- a compile-time constant; changing it
        triggers retracing).
    input_paths : Tuple[str, ...]
        Input paths (**static** -- a compile-time constant; changing it
        triggers retracing).
    output_paths : Tuple[str, ...]
        Output paths (**static** -- a compile-time constant; changing
        it triggers retracing).
    structural : Bool[Array, "n_output n_input"]
        Structural retained as a differentiable JAX leaf in the
        declared physical units.
    traced : Bool[Array, "n_output n_input"]
        Traced retained as a differentiable JAX leaf in the declared
        physical units.
    """

    model_id: str = eqx.field(static=True)
    input_paths: Tuple[str, ...] = eqx.field(static=True)
    output_paths: Tuple[str, ...] = eqx.field(static=True)
    structural: Bool[Array, "n_output n_input"]
    traced: Bool[Array, "n_output n_input"]


class SensitivityMap(eqx.Module):
    """Store scaled sensitivities from inputs to output projections.

    Carry scientific vocabulary separately from traced leaves.
    The record remains stable under JIT, VMAP, JVP, and VJP transforms.

    :see: :class:`~.test_certification.TestSensitivitymap`

    Attributes
    ----------
    input_paths : Tuple[str, ...]
        Input paths (**static** -- a compile-time constant; changing it
        triggers retracing).
    output_projection_ids : Tuple[str, ...]
        Output projection ids (**static** -- a compile-time constant;
        changing it triggers retracing).
    scales : Float64[Array, " n_input"]
        Scales retained as a differentiable JAX leaf in the declared
        physical units.
    sensitivities : Float64[Array, "n_output n_input"]
        Sensitivities retained as a differentiable JAX leaf in the
        declared physical units.
    threshold : Float64[Array, ""]
        Threshold retained as a differentiable JAX leaf in the declared
        physical units.
    active : Bool[Array, "n_output n_input"]
        Active retained as a differentiable JAX leaf in the declared
        physical units.
    """

    input_paths: Tuple[str, ...] = eqx.field(static=True)
    output_projection_ids: Tuple[str, ...] = eqx.field(static=True)
    scales: Float64[Array, " n_input"]
    sensitivities: Float64[Array, "n_output n_input"]
    threshold: Float64[Array, ""]
    active: Bool[Array, "n_output n_input"]


class InformationSpectrum(eqx.Module):
    """Store a matrix-free information spectrum in input coordinates.

    Carry scientific vocabulary separately from traced leaves.
    The record remains stable under JIT, VMAP, JVP, and VJP transforms.

    :see: :class:`~.test_certification.TestInformationspectrum`

    Attributes
    ----------
    input_paths : Tuple[str, ...]
        Input paths (**static** -- a compile-time constant; changing it
        triggers retracing).
    singular_values : Float64[Array, " n_sv"]
        Singular values retained as a differentiable JAX leaf in the
        declared physical units.
    right_singular_vectors : Float64[Array, "n_sv n_input"]
        Right singular vectors retained as a differentiable JAX leaf in
        the declared physical units.
    effective_rank : Int32[Array, ""]
        Effective rank retained as a differentiable JAX leaf in the
        declared physical units.
    condition_estimate : Float64[Array, ""]
        Condition estimate retained as a differentiable JAX leaf in the
        declared physical units. Zero means that there is no active information
        direction.
    threshold : Float64[Array, ""]
        Threshold retained as a differentiable JAX leaf in the declared
        physical units.
    """

    input_paths: Tuple[str, ...] = eqx.field(static=True)
    singular_values: Float64[Array, " n_sv"]
    right_singular_vectors: Float64[Array, "n_sv n_input"]
    effective_rank: Int32[Array, ""]
    condition_estimate: Float64[Array, ""]
    threshold: Float64[Array, ""]


class ExecutionManifest(eqx.Module):
    """Store software and execution identity prepared at the I/O boundary.

    Carry scientific vocabulary separately from traced leaves.
    The record remains stable under JIT, VMAP, JVP, and VJP transforms.

    :see: :class:`~.test_certification.TestExecutionmanifest`

    Attributes
    ----------
    execution_id : str
        Execution id (**static** -- a compile-time constant; changing
        it triggers retracing).
    model_ref : str
        Model ref (**static** -- a compile-time constant; changing it
        triggers retracing).
    schema_version : str
        Schema version (**static** -- a compile-time constant; changing
        it triggers retracing).
    package_version : str
        Package version (**static** -- a compile-time constant;
        changing it triggers retracing).
    source_checksum : str
        Source checksum (**static** -- a compile-time constant;
        changing it triggers retracing).
    environment_checksum : str
        Environment checksum (**static** -- a compile-time constant;
        changing it triggers retracing).
    backend : str
        Backend (**static** -- a compile-time constant; changing it
        triggers retracing).
    precision_policy : str
        Precision policy (**static** -- a compile-time constant;
        changing it triggers retracing).
    deterministic : bool
        Deterministic (**static** -- a compile-time constant; changing
        it triggers retracing).
    started_at_utc : str
        Started at utc (**static** -- a compile-time constant; changing
        it triggers retracing).
    """

    execution_id: str = eqx.field(static=True)
    model_ref: str = eqx.field(static=True)
    schema_version: str = eqx.field(static=True)
    package_version: str = eqx.field(static=True)
    source_checksum: str = eqx.field(static=True)
    environment_checksum: str = eqx.field(static=True)
    backend: str = eqx.field(static=True)
    precision_policy: str = eqx.field(static=True)
    deterministic: bool = eqx.field(static=True)
    started_at_utc: str = eqx.field(static=True)


class PolicyReport(eqx.Module):
    """Store a traced policy truth table for derived certification levels.

    Carry scientific vocabulary separately from traced leaves.
    The record remains stable under JIT, VMAP, JVP, and VJP transforms.

    :see: :class:`~.test_certification.TestPolicyreport`

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
    """

    policy_id: str = eqx.field(static=True)
    level_ids: Tuple[str, ...] = eqx.field(static=True)
    required_claim_ids: Tuple[str, ...] = eqx.field(static=True)
    claim_passed: Bool[Array, " n_claim"]
    claim_checked: Bool[Array, " n_claim"]
    claim_in_domain: Bool[Array, " n_claim"]
    achieved: Bool[Array, " n_level"]


class CertificationContext(eqx.Module):
    """Store prepared selections and references for compiled certification.

    Carry scientific vocabulary separately from traced leaves.
    The record remains stable under JIT, VMAP, JVP, and VJP transforms.

    :see: :class:`~.test_certification.TestCertificationcontext`

    Attributes
    ----------
    manifest : ExecutionManifest
        Manifest retained as a differentiable JAX leaf in the declared
        physical units.
    model : ForwardModelSpec
        Model retained as a differentiable JAX leaf in the declared
        physical units.
    artifacts : Tuple[ArtifactRef, ...]
        Artifacts retained as a differentiable JAX leaf in the declared
        physical units.
    transformations : Tuple[TransformationRecord, ...]
        Transformations retained as a differentiable JAX leaf in the
        declared physical units.
    evidence : Tuple[EvidenceRef, ...]
        Evidence retained as a differentiable JAX leaf in the declared
        physical units.
    attestations : Tuple[HumanAttestationRef, ...]
        Human-review records kept separate from numerical evidence.
    policy_id : str
        Policy id (**static** -- a compile-time constant; changing it
        triggers retracing).
    check_ids : Tuple[str, ...]
        Check ids (**static** -- a compile-time constant; changing it
        triggers retracing).
    input_checksums : Tuple[str, ...]
        Input checksums (**static** -- a compile-time constant;
        changing it triggers retracing).
    waivers : Tuple[WaiverRecord, ...]
        Policy-waiver records (**static**; changing them causes retracing).
    """

    manifest: ExecutionManifest
    model: ForwardModelSpec
    artifacts: Tuple[ArtifactRef, ...]
    transformations: Tuple[TransformationRecord, ...]
    evidence: Tuple[EvidenceRef, ...]
    attestations: Tuple[HumanAttestationRef, ...]
    policy_id: str = eqx.field(static=True)
    check_ids: Tuple[str, ...] = eqx.field(static=True)
    input_checksums: Tuple[str, ...] = eqx.field(static=True)
    waivers: Tuple["WaiverRecord", ...] = eqx.field(static=True)


class ForwardCertificate(eqx.Module):
    """Store the complete assurance record for one forward execution.

    Carry scientific vocabulary separately from traced leaves.
    The record remains stable under JIT, VMAP, JVP, and VJP transforms.

    :see: :class:`~.test_certification.TestForwardcertificate`

    Attributes
    ----------
    manifest : ExecutionManifest
        Manifest retained as a differentiable JAX leaf in the declared
        physical units.
    model : ForwardModelSpec
        Model retained as a differentiable JAX leaf in the declared
        physical units.
    artifacts : Tuple[ArtifactRef, ...]
        Artifacts retained as a differentiable JAX leaf in the declared
        physical units.
    transformations : Tuple[TransformationRecord, ...]
        Transformations retained as a differentiable JAX leaf in the
        declared physical units.
    evidence : Tuple[EvidenceRef, ...]
        Evidence retained as a differentiable JAX leaf in the declared
        physical units.
    attestations : Tuple[HumanAttestationRef, ...]
        Human-review records kept separate from numerical evidence.
    claims : Tuple[CertificationClaim, ...]
        Claims retained as a differentiable JAX leaf in the declared
        physical units.
    domains : Tuple[DomainResult, ...]
        Domains retained as a differentiable JAX leaf in the declared
        physical units.
    derivatives : DerivativeEvidence
        Derivatives retained as a differentiable JAX leaf in the
        declared physical units.
    dependencies : DependencyMap
        Dependencies retained as a differentiable JAX leaf in the
        declared physical units.
    sensitivities : SensitivityMap
        Sensitivities retained as a differentiable JAX leaf in the
        declared physical units.
    information : InformationSpectrum
        Information retained as a differentiable JAX leaf in the
        declared physical units.
    policy_report : PolicyReport
        Policy report retained as a differentiable JAX leaf in the
        declared physical units.
    policy_id : str
        Policy id (**static** -- a compile-time constant; changing it
        triggers retracing).
    certificate_checksum : str
        Certificate checksum (**static** -- a compile-time constant;
        changing it triggers retracing).
    extensions_json : str
        Extensions json (**static** -- a compile-time constant;
        changing it triggers retracing).
    waivers : Tuple[WaiverRecord, ...]
        Policy-waiver records (**static**; changing them causes retracing).
    """

    manifest: ExecutionManifest
    model: ForwardModelSpec
    artifacts: Tuple[ArtifactRef, ...]
    transformations: Tuple[TransformationRecord, ...]
    evidence: Tuple[EvidenceRef, ...]
    attestations: Tuple[HumanAttestationRef, ...]
    claims: Tuple[CertificationClaim, ...]
    domains: Tuple[DomainResult, ...]
    derivatives: DerivativeEvidence
    dependencies: DependencyMap
    sensitivities: SensitivityMap
    information: InformationSpectrum
    policy_report: PolicyReport
    policy_id: str = eqx.field(static=True)
    certificate_checksum: str = eqx.field(static=True)
    extensions_json: str = eqx.field(static=True)
    waivers: Tuple["WaiverRecord", ...] = eqx.field(static=True)


class CertifiedResult(eqx.Module):
    """Store a numerical result paired with its differentiable certificate.

    Carry scientific vocabulary separately from traced leaves.
    The record remains stable under JIT, VMAP, JVP, and VJP transforms.

    :see: :class:`~.test_certification.TestCertifiedresult`

    Attributes
    ----------
    value : Any
        Value retained as a differentiable JAX leaf in the declared
        physical units.
    certificate : ForwardCertificate
        Certificate retained as a differentiable JAX leaf in the
        declared physical units.
    """

    value: Any
    certificate: ForwardCertificate


class EvidenceReport(eqx.Module):
    """Store the offline consistency outcome for one evidence record.

    Carry scientific vocabulary separately from traced leaves.
    The record remains stable under JIT, VMAP, JVP, and VJP transforms.

    :see: :class:`~.test_certification.TestEvidencereport`

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

    :see: :class:`~.test_certification.TestVerificationreport`

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

    :see: :class:`~.test_certification.TestReproductionreport`

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

    :see: :class:`~.test_certification.TestWaiverRecord`

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

    :see: :class:`~.test_certification.TestWaiverReport`

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
    """

    waiver_id: str = eqx.field(static=True)
    valid: Bool[Array, ""]
    active: Bool[Array, ""]
    errors: Tuple[str, ...] = eqx.field(static=True)


def _require_text(value: str, name: str) -> str:
    """PRIVATE: Reject empty static vocabulary entries.

    Parameters
    ----------
    value : str
        Candidate static vocabulary string.
    name : str
        Field name used in the static error message.

    Returns
    -------
    value : str
        The validated input string, unchanged.

    Raises
    ------
    ValueError
        If ``value`` contains only whitespace or is empty. This is the
        static construction-time contract.

    Notes
    -----
    Apply ``str.strip`` for the check and keep the original
    representation on success.
    """
    if not value.strip():
        raise ValueError(f"{name} must be non-empty")
    return value


def _require_optional_text(value: Optional[str], name: str) -> Optional[str]:
    """PRIVATE: Reject an explicitly supplied empty optional string.

    Parameters
    ----------
    value : Optional[str]
        Candidate optional string, or ``None`` when absent.
    name : str
        Field name passed to the static error message.

    Returns
    -------
    value : Optional[str]
        ``None`` when absent, otherwise the validated nonblank string.

    Notes
    -----
    Pass ``None`` through untouched. Delegate a present value to
    ``_require_text``, which raises the static ``ValueError`` for blank
    text.
    """
    if value is not None:
        result: Optional[str] = _require_text(value, name)
        return result
    return value


def _text_tuple(values: Tuple[str, ...], name: str) -> Tuple[str, ...]:
    """PRIVATE: Normalize and validate a tuple of identifiers.

    Implementation Logic
    --------------------
    Validate every entry through ``_require_text`` while rebuilding the
    tuple. Then compare the set size against the tuple length to reject
    duplicates.

    Parameters
    ----------
    values : Tuple[str, ...]
        Candidate identifier entries.
    name : str
        Field name used in the static error messages.

    Returns
    -------
    result : Tuple[str, ...]
        The validated identifiers frozen into a tuple in input order.

    Raises
    ------
    ValueError
        If the entries are not unique. ``_require_text`` also raises
        ``ValueError`` for a blank entry. This is the static
        construction-time contract.
    """
    result: Tuple[str, ...] = tuple(
        _require_text(value, name) for value in values
    )
    if len(result) != len(set(result)):
        raise ValueError(f"{name} must not contain duplicates")
    return result


def _json_object(value: str, name: str) -> str:
    """PRIVATE: Require a JSON object while preserving the supplied text.

    Implementation Logic
    --------------------
    Decode with ``json.loads`` only to validate. Return the supplied
    string so that checksums over the stored representation stay
    stable.

    Parameters
    ----------
    value : str
        Candidate JSON document as text.
    name : str
        Field name used in the static error messages.

    Returns
    -------
    value : str
        The original JSON text, byte-for-byte unchanged.

    Raises
    ------
    ValueError
        If ``value`` is not valid JSON, or if the decoded document is
        not a JSON object. This is the static construction-time
        contract.
    """
    error: json.JSONDecodeError
    try:
        decoded: Any = json.loads(value)
    except json.JSONDecodeError as error:
        raise ValueError(f"{name} must be valid JSON") from error
    if not isinstance(decoded, dict):
        raise ValueError(f"{name} must encode a JSON object")
    return value


def _float(value: Any, name: str, ndim: int) -> Array:
    """PRIVATE: Cast a numerical value to float64 and enforce its rank.

    Parameters
    ----------
    value : Any
        Numerical value convertible by ``jnp.asarray``.
    name : str
        Field name used in the static error message.
    ndim : int
        Required array rank.

    Returns
    -------
    array : Array
        The value cast to a float64 JAX array.

    Raises
    ------
    ValueError
        If the cast array does not have rank ``ndim``. This is the
        static construction-time contract.

    Notes
    -----
    The rank check is static shape metadata. The numerical content
    stays a traced leaf.
    """
    array: Array = jnp.asarray(value, dtype=jnp.float64)
    if array.ndim != ndim:
        raise ValueError(f"{name} must have rank {ndim}")
    return array


def _bool(value: Any, name: str, ndim: int) -> Array:
    """PRIVATE: Cast a logical value to bool and enforce its rank.

    Parameters
    ----------
    value : Any
        Logical value convertible by ``jnp.asarray``.
    name : str
        Field name used in the static error message.
    ndim : int
        Required array rank.

    Returns
    -------
    array : Array
        The value cast to a boolean JAX array.

    Raises
    ------
    ValueError
        If the cast array does not have rank ``ndim``. This is the
        static construction-time contract.

    Notes
    -----
    The rank check is static shape metadata. The logical content stays
    a traced leaf.
    """
    array: Array = jnp.asarray(value, dtype=jnp.bool_)
    if array.ndim != ndim:
        raise ValueError(f"{name} must have rank {ndim}")
    return array


def _int(value: Any, name: str, ndim: int) -> Array:
    """PRIVATE: Cast an integer value to int32 and enforce its rank.

    Parameters
    ----------
    value : Any
        Integer value convertible by ``jnp.asarray``.
    name : str
        Field name used in the static error message.
    ndim : int
        Required array rank.

    Returns
    -------
    array : Array
        The value cast to an int32 JAX array.

    Raises
    ------
    ValueError
        If the cast array does not have rank ``ndim``. This is the
        static construction-time contract.

    Notes
    -----
    The rank check is static shape metadata. The integer content stays
    a traced leaf.
    """
    array: Array = jnp.asarray(value, dtype=jnp.int32)
    if array.ndim != ndim:
        raise ValueError(f"{name} must have rank {ndim}")
    return array


def _nonnegative(array: Array, name: str) -> Array:
    """PRIVATE: Require finite, nonnegative tolerance-like leaves under JIT.

    Parameters
    ----------
    array : Array
        Traced numerical leaf to guard.
    name : str
        Field name used in the traced error message.

    Returns
    -------
    result : Array
        The same values with the runtime check attached.

    Notes
    -----
    Attach a traced ``eqx.error_if`` guard instead of raising a static
    ``ValueError``. The check runs under JIT and fails at run time when
    any element is nonfinite or negative.
    """
    result: Array = eqx.error_if(
        array,
        ~jnp.all(jnp.isfinite(array) & (array >= 0.0)),
        f"{name} must be finite and nonnegative",
    )
    return result


def _positive(array: Array, name: str) -> Array:
    """PRIVATE: Require finite, positive scale leaves under JIT.

    Parameters
    ----------
    array : Array
        Traced numerical leaf to guard.
    name : str
        Field name used in the traced error message.

    Returns
    -------
    result : Array
        The same values with the runtime check attached.

    Notes
    -----
    Attach a traced ``eqx.error_if`` guard instead of raising a static
    ``ValueError``. The check runs under JIT and fails at run time when
    any element is nonfinite or not strictly positive.
    """
    result: Array = eqx.error_if(
        array,
        ~jnp.all(jnp.isfinite(array) & (array > 0.0)),
        f"{name} must be finite and positive",
    )
    return result


@jaxtyped(typechecker=beartype)
def make_artifact_ref(
    artifact_id: str,
    media_type: str,
    byte_checksum: Optional[str],
    content_checksum: str,
    semantic_checksum: str,
    locator: Optional[str],
    role: str,
) -> ArtifactRef:
    """Create a validated artifact reference.

    Carry static scientific vocabulary separately from traced numerical leaves
    while preserving the validation boundary defined by this factory.

    :see: :class:`~.test_certification.TestMakeArtifactRef`

    Parameters
    ----------
    artifact_id : str
        Artifact id used to construct the validated carrier (**static**
        -- a compile-time constant; changing it triggers retracing).
    media_type : str
        Media type used to construct the validated carrier (**static**
        -- a compile-time constant; changing it triggers retracing).
    byte_checksum : Optional[str]
        Byte checksum used to construct the validated carrier
        (**static** -- a compile-time constant; changing it triggers
        retracing).
    content_checksum : str
        Content checksum used to construct the validated carrier
        (**static** -- a compile-time constant; changing it triggers
        retracing).
    semantic_checksum : str
        Semantic checksum used to construct the validated carrier
        (**static** -- a compile-time constant; changing it triggers
        retracing).
    locator : Optional[str]
        Locator used to construct the validated carrier (**static** --
        a compile-time constant; changing it triggers retracing).
    role : str
        Role used to construct the validated carrier (**static** -- a
        compile-time constant; changing it triggers retracing).

    Returns
    -------
    result : ArtifactRef
        Validated immutable carrier.

    Notes
    -----
    The factory checks static structure eagerly. JAX array operations validate
    numerical values and preserve differentiation behavior.
    """
    result: ArtifactRef = ArtifactRef(
        artifact_id=_require_text(artifact_id, "artifact_id"),
        media_type=_require_text(media_type, "media_type"),
        byte_checksum=_require_optional_text(byte_checksum, "byte_checksum"),
        content_checksum=_require_text(content_checksum, "content_checksum"),
        semantic_checksum=_require_text(
            semantic_checksum, "semantic_checksum"
        ),
        locator=_require_optional_text(locator, "locator"),
        role=_require_text(role, "role"),
    )
    return result


@jaxtyped(typechecker=beartype)
def make_convention_ref(
    convention_id: str,
    version: str,
    parameters_json: str = "{}",
) -> ConventionRef:
    """Create a validated convention reference.

    Carry static scientific vocabulary separately from traced numerical leaves
    while preserving the validation boundary defined by this factory.

    :see: :class:`~.test_certification.TestMakeConventionRef`

    Parameters
    ----------
    convention_id : str
        Convention id used to construct the validated carrier
        (**static** -- a compile-time constant; changing it triggers
        retracing).
    version : str
        Version used to construct the validated carrier (**static** --
        a compile-time constant; changing it triggers retracing).
    parameters_json : str
        Parameters json used to construct the validated carrier
        (**static** -- a compile-time constant; changing it triggers
        retracing).

    Returns
    -------
    result : ConventionRef
        Validated immutable carrier.

    Notes
    -----
    The factory checks static structure eagerly. JAX array operations validate
    numerical values and preserve differentiation behavior.
    """
    result: ConventionRef = ConventionRef(
        convention_id=_require_text(convention_id, "convention_id"),
        version=_require_text(version, "version"),
        parameters_json=_json_object(parameters_json, "parameters_json"),
    )
    return result


@jaxtyped(typechecker=beartype)
def make_domain_predicate(
    predicate_id: str,
    expression_id: str,
    units: Optional[str] = None,
    severity: str = "error",
) -> DomainPredicate:
    """Create a validated domain-predicate declaration.

    Carry static scientific vocabulary separately from traced numerical leaves
    while preserving the validation boundary defined by this factory.

    :see: :class:`~.test_certification.TestMakeDomainPredicate`

    Parameters
    ----------
    predicate_id : str
        Predicate id used to construct the validated carrier
        (**static** -- a compile-time constant; changing it triggers
        retracing).
    expression_id : str
        Expression id used to construct the validated carrier
        (**static** -- a compile-time constant; changing it triggers
        retracing).
    units : Optional[str]
        Units used to construct the validated carrier (**static** -- a
        compile-time constant; changing it triggers retracing).
    severity : str
        Severity used to construct the validated carrier (**static** --
        a compile-time constant; changing it triggers retracing).

    Returns
    -------
    result : DomainPredicate
        Validated immutable carrier.

    Notes
    -----
    The factory checks static structure eagerly. JAX array operations validate
    numerical values and preserve differentiation behavior.
    """
    result: DomainPredicate = DomainPredicate(
        predicate_id=_require_text(predicate_id, "predicate_id"),
        expression_id=_require_text(expression_id, "expression_id"),
        units=_require_optional_text(units, "units"),
        severity=_require_text(severity, "severity"),
    )
    return result


@jaxtyped(typechecker=beartype)
def make_domain_result(
    predicate_id: str,
    measured: Any,
    reference: Any,
    residual: Any,
    tolerance: Any,
    margin: Any,
    passed: Any,
    checked: Any = True,
    in_domain: Any = True,
    severity_code: Any = 0,
) -> DomainResult:
    """Create one traced domain evaluation.

    Carry static scientific vocabulary separately from traced numerical leaves
    while preserving the validation boundary defined by this factory.

    :see: :class:`~.test_certification.TestMakeDomainResult`

    Parameters
    ----------
    predicate_id : str
        Predicate id used to construct the validated carrier
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
    margin : Any
        Margin used to construct the validated carrier as a traced
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
    severity_code : Any
        Severity code used to construct the validated carrier as a
        traced numerical value in the declared physical units.

    Returns
    -------
    result : DomainResult
        Validated immutable carrier.

    Notes
    -----
    The factory checks static structure eagerly. JAX array operations validate
    numerical values and preserve differentiation behavior.
    """
    result: DomainResult = DomainResult(
        predicate_id=_require_text(predicate_id, "predicate_id"),
        measured=_float(measured, "measured", 0),
        reference=_float(reference, "reference", 0),
        residual=_float(residual, "residual", 0),
        tolerance=_nonnegative(_float(tolerance, "tolerance", 0), "tolerance"),
        margin=_float(margin, "margin", 0),
        passed=_bool(passed, "passed", 0),
        checked=_bool(checked, "checked", 0),
        in_domain=_bool(in_domain, "in_domain", 0),
        severity_code=_int(severity_code, "severity_code", 0),
    )
    return result


@jaxtyped(typechecker=beartype)
def make_forward_model_spec(
    model_id: str,
    model_version: str,
    observable_id: str,
    implementation_ref: str,
    assumptions: Tuple[str, ...] = (),
    conventions: Tuple[ConventionRef, ...] = (),
    domain: Tuple[DomainPredicate, ...] = (),
    differentiable_paths: Tuple[str, ...] = (),
    nondifferentiable_paths: Tuple[str, ...] = (),
) -> ForwardModelSpec:
    """Create a validated stable forward-model specification.

    Carry static scientific vocabulary separately from traced numerical leaves
    while preserving the validation boundary defined by this factory.

    :see: :class:`~.test_certification.TestMakeForwardModelSpec`

    Parameters
    ----------
    model_id : str
        Model id used to construct the validated carrier (**static** --
        a compile-time constant; changing it triggers retracing).
    model_version : str
        Model version used to construct the validated carrier
        (**static** -- a compile-time constant; changing it triggers
        retracing).
    observable_id : str
        Observable id used to construct the validated carrier
        (**static** -- a compile-time constant; changing it triggers
        retracing).
    implementation_ref : str
        Implementation ref used to construct the validated carrier
        (**static** -- a compile-time constant; changing it triggers
        retracing).
    assumptions : Tuple[str, ...]
        Assumptions used to construct the validated carrier (**static**
        -- a compile-time constant; changing it triggers retracing).
    conventions : Tuple[ConventionRef, ...]
        Conventions used to construct the validated carrier (**static**
        -- a compile-time constant; changing it triggers retracing).
    domain : Tuple[DomainPredicate, ...]
        Domain used to construct the validated carrier (**static** -- a
        compile-time constant; changing it triggers retracing).
    differentiable_paths : Tuple[str, ...]
        Differentiable paths used to construct the validated carrier
        (**static** -- a compile-time constant; changing it triggers
        retracing).
    nondifferentiable_paths : Tuple[str, ...]
        Nondifferentiable paths used to construct the validated carrier
        (**static** -- a compile-time constant; changing it triggers
        retracing).

    Returns
    -------
    result : ForwardModelSpec
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
    diff_paths: Tuple[str, ...] = _text_tuple(
        differentiable_paths,
        "differentiable_paths",
    )
    nondiff_paths: Tuple[str, ...] = _text_tuple(
        nondifferentiable_paths, "nondifferentiable_paths"
    )
    overlap: set[str] = set(diff_paths).intersection(nondiff_paths)
    if overlap:
        raise ValueError(
            "differentiable_paths and nondifferentiable_paths must be disjoint"
        )
    convention_ids: Tuple[str, ...] = tuple(
        item.convention_id for item in conventions
    )
    predicate_ids: Tuple[str, ...] = tuple(
        item.predicate_id for item in domain
    )
    _text_tuple(convention_ids, "convention ids")
    _text_tuple(predicate_ids, "domain predicate ids")
    result: ForwardModelSpec = ForwardModelSpec(
        model_id=_require_text(model_id, "model_id"),
        model_version=_require_text(model_version, "model_version"),
        observable_id=_require_text(observable_id, "observable_id"),
        implementation_ref=_require_text(
            implementation_ref, "implementation_ref"
        ),
        assumptions=_text_tuple(assumptions, "assumptions"),
        conventions=tuple(conventions),
        domain=tuple(domain),
        differentiable_paths=diff_paths,
        nondifferentiable_paths=nondiff_paths,
    )
    return result


@jaxtyped(typechecker=beartype)
def make_registered_model(
    spec: ForwardModelSpec,
    executor: Callable[..., Any],
    registration_checksum: str,
) -> RegisteredModel:
    """Create a validated model-registry binding.

    Carry static scientific vocabulary separately from traced numerical leaves
    while preserving the validation boundary defined by this factory.

    :see: :class:`~.test_certification.TestMakeRegisteredModel`

    Parameters
    ----------
    spec : ForwardModelSpec
        Spec used to construct the validated carrier as a traced
        numerical value in the declared physical units.
    executor : Callable[..., Any]
        Executor used to construct the validated carrier (**static** --
        a compile-time constant; changing it triggers retracing).
    registration_checksum : str
        Registration checksum used to construct the validated carrier
        (**static** -- a compile-time constant; changing it triggers
        retracing).

    Returns
    -------
    result : RegisteredModel
        Validated immutable carrier.

    Raises
    ------
    TypeError
        If static structure or cross-record validation fails.

    Notes
    -----
    The factory checks static structure eagerly. JAX array operations validate
    numerical values and preserve differentiation behavior.
    """
    if not callable(executor):
        raise TypeError("model executor must be callable")
    result: RegisteredModel = RegisteredModel(
        spec=spec,
        executor=executor,
        registration_checksum=_require_text(
            registration_checksum,
            "registration_checksum",
        ),
    )
    return result


@jaxtyped(typechecker=beartype)
def make_registered_transformation(
    contract: TransformationContract,
    registration_checksum: str,
) -> RegisteredTransformation:
    """Create a validated transformation-registry binding.

    Carry static scientific vocabulary separately from traced numerical leaves
    while preserving the validation boundary defined by this factory.

    :see: :class:`~.test_certification.TestMakeRegisteredTransformation`

    Parameters
    ----------
    contract : TransformationContract
        Contract used to construct the validated carrier as a traced
        numerical value in the declared physical units.
    registration_checksum : str
        Registration checksum used to construct the validated carrier
        (**static** -- a compile-time constant; changing it triggers
        retracing).

    Returns
    -------
    result : RegisteredTransformation
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
    if contract is None:
        raise ValueError("contract must not be None")
    result: RegisteredTransformation = RegisteredTransformation(
        contract=contract,
        registration_checksum=_require_text(
            registration_checksum,
            "registration_checksum",
        ),
    )
    return result


@jaxtyped(typechecker=beartype)
def make_registry_snapshot(
    models: Tuple[RegisteredModel, ...],
    transformations: Tuple[RegisteredTransformation, ...],
    checksum: str,
) -> RegistrySnapshot:
    """Create an immutable registry snapshot.

    Carry static scientific vocabulary separately from traced numerical leaves
    while preserving the validation boundary defined by this factory.

    :see: :class:`~.test_certification.TestMakeRegistrySnapshot`

    Parameters
    ----------
    models : Tuple[RegisteredModel, ...]
        Models used to construct the validated carrier (**static** -- a
        compile-time constant; changing it triggers retracing).
    transformations : Tuple[RegisteredTransformation, ...]
        Transformations used to construct the validated carrier
        (**static** -- a compile-time constant; changing it triggers
        retracing).
    checksum : str
        Checksum used to construct the validated carrier (**static** --
        a compile-time constant; changing it triggers retracing).

    Returns
    -------
    result : RegistrySnapshot
        Validated immutable carrier.

    Notes
    -----
    The factory checks static structure eagerly. JAX array operations validate
    numerical values and preserve differentiation behavior.
    """
    result: RegistrySnapshot = RegistrySnapshot(
        models=tuple(models),
        transformations=tuple(transformations),
        checksum=_require_text(checksum, "checksum"),
    )
    return result


@jaxtyped(typechecker=beartype)
def make_registry_report(
    valid: bool,
    errors: Tuple[str, ...],
    model_count: int,
    transformation_count: int,
    checksum: str,
    frozen: bool,
) -> RegistryReport:
    """Create a validated structural registry report.

    Carry static scientific vocabulary separately from traced numerical leaves
    while preserving the validation boundary defined by this factory.

    :see: :class:`~.test_certification.TestMakeRegistryReport`

    Parameters
    ----------
    valid : bool
        Valid used to construct the validated carrier (**static** -- a
        compile-time constant; changing it triggers retracing).
    errors : Tuple[str, ...]
        Errors used to construct the validated carrier (**static** -- a
        compile-time constant; changing it triggers retracing).
    model_count : int
        Model count used to construct the validated carrier (**static**
        -- a compile-time constant; changing it triggers retracing).
    transformation_count : int
        Transformation count used to construct the validated carrier
        (**static** -- a compile-time constant; changing it triggers
        retracing).
    checksum : str
        Checksum used to construct the validated carrier (**static** --
        a compile-time constant; changing it triggers retracing).
    frozen : bool
        Frozen used to construct the validated carrier (**static** -- a
        compile-time constant; changing it triggers retracing).

    Returns
    -------
    result : RegistryReport
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
    if model_count < 0 or transformation_count < 0:
        raise ValueError("registry entry counts must be nonnegative")
    result: RegistryReport = RegistryReport(
        valid=valid,
        errors=tuple(errors),
        model_count=model_count,
        transformation_count=transformation_count,
        checksum=_require_text(checksum, "checksum"),
        frozen=frozen,
    )
    return result


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

    :see: :class:`~.test_certification.TestMakeTransformationRecord`

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

    :see: :class:`~.test_certification.TestMakeEvidenceLineage`

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

    :see: :class:`~.test_certification.TestMakeHumanAttestationRef`

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

    :see: :class:`~.test_certification.TestMakeEvidenceRef`

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
    measured_array: Array = _float(measured, "measured", 1)
    reference_array: Array = _float(reference, "reference", 1)
    residual_array: Array = _float(residual, "residual", 1)
    tolerance_array: Array = _nonnegative(
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
def make_certification_claim(  # noqa: PLR0913
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

    :see: :class:`~.test_certification.TestMakeCertificationClaim`

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
    measured_array: Array = _float(measured, "measured", 1)
    reference_array: Array = _float(reference, "reference", 1)
    residual_array: Array = _float(residual, "residual", 1)
    tolerance_array: Array = _nonnegative(
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


@jaxtyped(typechecker=beartype)
def make_derivative_evidence(  # noqa: PLR0913
    input_paths: Tuple[str, ...],
    output_projection_ids: Tuple[str, ...],
    method: str,
    scales: Any,
    jvp_probes: Any,
    vjp_probes: Any,
    reference_derivatives: Any,
    derivative_residuals: Any,
    singular_values: Any,
    effective_rank: Any,
    condition_estimate: Any,
    finite: Any,
    fd_correct: Any,
) -> DerivativeEvidence:
    """Create validated derivative and local-information evidence.

    Carry static scientific vocabulary separately from traced numerical leaves
    while preserving the validation boundary defined by this factory.

    :see: :class:`~.test_certification.TestMakeDerivativeEvidence`

    Parameters
    ----------
    input_paths : Tuple[str, ...]
        Input paths used to construct the validated carrier (**static**
        -- a compile-time constant; changing it triggers retracing).
    output_projection_ids : Tuple[str, ...]
        Output projection ids used to construct the validated carrier
        (**static** -- a compile-time constant; changing it triggers
        retracing).
    method : str
        Method used to construct the validated carrier (**static** -- a
        compile-time constant; changing it triggers retracing).
    scales : Any
        Scales used to construct the validated carrier as a traced
        numerical value in the declared physical units.
    jvp_probes : Any
        Jvp probes used to construct the validated carrier as a traced
        numerical value in the declared physical units.
    vjp_probes : Any
        Vjp probes used to construct the validated carrier as a traced
        numerical value in the declared physical units.
    reference_derivatives : Any
        Reference derivatives used to construct the validated carrier
        as a traced numerical value in the declared physical units.
    derivative_residuals : Any
        Derivative residuals used to construct the validated carrier as
        a traced numerical value in the declared physical units.
    singular_values : Any
        Singular values used to construct the validated carrier as a
        traced numerical value in the declared physical units.
    effective_rank : Any
        Effective rank used to construct the validated carrier as a
        traced numerical value in the declared physical units.
    condition_estimate : Any
        Condition estimate used to construct the validated carrier as a
        traced numerical value in the declared physical units. Zero means that
        there is no active information direction.
    finite : Any
        Finite used to construct the validated carrier as a traced
        numerical value in the declared physical units.
    fd_correct : Any
        Fd correct used to construct the validated carrier as a traced
        numerical value in the declared physical units.

    Returns
    -------
    result : DerivativeEvidence
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
    paths: Tuple[str, ...] = _text_tuple(input_paths, "input_paths")
    projections: Tuple[str, ...] = _text_tuple(
        output_projection_ids,
        "output_projection_ids",
    )
    scales_array: Array = _positive(_float(scales, "scales", 1), "scales")
    jvp_array: Array = _float(jvp_probes, "jvp_probes", 2)
    vjp_array: Array = _float(vjp_probes, "vjp_probes", 2)
    reference_array: Array = _float(
        reference_derivatives,
        "reference_derivatives",
        2,
    )
    residual_array: Array = _float(
        derivative_residuals,
        "derivative_residuals",
        2,
    )
    singular_array: Array = _nonnegative(
        _float(singular_values, "singular_values", 1), "singular_values"
    )
    if scales_array.shape[0] != len(paths):
        raise ValueError("scales length must equal input_paths length")
    if vjp_array.shape[1] != len(paths):
        raise ValueError("vjp_probes input dimension must equal input_paths")
    if jvp_array.shape[0] != vjp_array.shape[0]:
        raise ValueError("JVP and VJP probe counts must agree")
    if reference_array.shape != residual_array.shape:
        raise ValueError("reference and derivative residual shapes must agree")
    if reference_array.shape[0] != jvp_array.shape[0]:
        raise ValueError("reference derivative probe count must agree")
    result: DerivativeEvidence = DerivativeEvidence(
        input_paths=paths,
        output_projection_ids=projections,
        method=_require_text(method, "method"),
        scales=scales_array,
        jvp_probes=jvp_array,
        vjp_probes=vjp_array,
        reference_derivatives=reference_array,
        derivative_residuals=residual_array,
        singular_values=singular_array,
        effective_rank=_int(effective_rank, "effective_rank", 0),
        condition_estimate=_float(condition_estimate, "condition_estimate", 0),
        finite=_bool(finite, "finite", 0),
        fd_correct=_bool(fd_correct, "fd_correct", 0),
    )
    return result


@jaxtyped(typechecker=beartype)
def make_dependency_map(
    model_id: str,
    input_paths: Tuple[str, ...],
    output_paths: Tuple[str, ...],
    structural: Any,
    traced: Any,
) -> DependencyMap:
    """Create a structural dependency map.

    Carry static scientific vocabulary separately from traced numerical leaves
    while preserving the validation boundary defined by this factory.

    :see: :class:`~.test_certification.TestMakeDependencyMap`

    Parameters
    ----------
    model_id : str
        Model id used to construct the validated carrier (**static** --
        a compile-time constant; changing it triggers retracing).
    input_paths : Tuple[str, ...]
        Input paths used to construct the validated carrier (**static**
        -- a compile-time constant; changing it triggers retracing).
    output_paths : Tuple[str, ...]
        Output paths used to construct the validated carrier
        (**static** -- a compile-time constant; changing it triggers
        retracing).
    structural : Any
        Structural used to construct the validated carrier as a traced
        numerical value in the declared physical units.
    traced : Any
        Traced used to construct the validated carrier as a traced
        numerical value in the declared physical units.

    Returns
    -------
    result : DependencyMap
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
    inputs: Tuple[str, ...] = _text_tuple(input_paths, "input_paths")
    outputs: Tuple[str, ...] = _text_tuple(output_paths, "output_paths")
    structural_array: Array = _bool(structural, "structural", 2)
    traced_array: Array = _bool(traced, "traced", 2)
    expected: Tuple[int, int] = (len(outputs), len(inputs))
    if structural_array.shape != expected or traced_array.shape != expected:
        raise ValueError(
            "dependency matrices must have shape (outputs, inputs)"
        )
    result: DependencyMap = DependencyMap(
        model_id=_require_text(model_id, "model_id"),
        input_paths=inputs,
        output_paths=outputs,
        structural=structural_array,
        traced=traced_array,
    )
    return result


@jaxtyped(typechecker=beartype)
def make_sensitivity_map(
    input_paths: Tuple[str, ...],
    output_projection_ids: Tuple[str, ...],
    scales: Any,
    sensitivities: Any,
    threshold: Any,
    active: Any,
) -> SensitivityMap:
    """Create a named, scaled local-sensitivity map.

    Carry static scientific vocabulary separately from traced numerical leaves
    while preserving the validation boundary defined by this factory.

    :see: :class:`~.test_certification.TestMakeSensitivityMap`

    Parameters
    ----------
    input_paths : Tuple[str, ...]
        Input paths used to construct the validated carrier (**static**
        -- a compile-time constant; changing it triggers retracing).
    output_projection_ids : Tuple[str, ...]
        Output projection ids used to construct the validated carrier
        (**static** -- a compile-time constant; changing it triggers
        retracing).
    scales : Any
        Scales used to construct the validated carrier as a traced
        numerical value in the declared physical units.
    sensitivities : Any
        Sensitivities used to construct the validated carrier as a
        traced numerical value in the declared physical units.
    threshold : Any
        Threshold used to construct the validated carrier as a traced
        numerical value in the declared physical units.
    active : Any
        Active used to construct the validated carrier as a traced
        numerical value in the declared physical units.

    Returns
    -------
    result : SensitivityMap
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
    inputs: Tuple[str, ...] = _text_tuple(input_paths, "input_paths")
    outputs: Tuple[str, ...] = _text_tuple(
        output_projection_ids,
        "output_projection_ids",
    )
    scales_array: Array = _positive(_float(scales, "scales", 1), "scales")
    sensitivities_array: Array = _float(
        sensitivities,
        "sensitivities",
        2,
    )
    active_array: Array = _bool(active, "active", 2)
    expected: Tuple[int, int] = (len(outputs), len(inputs))
    if sensitivities_array.shape != expected or active_array.shape != expected:
        raise ValueError(
            "sensitivity matrices must have shape (outputs, inputs)"
        )
    if scales_array.shape != (len(inputs),):
        raise ValueError("scales length must equal input_paths length")
    result: SensitivityMap = SensitivityMap(
        input_paths=inputs,
        output_projection_ids=outputs,
        scales=scales_array,
        sensitivities=sensitivities_array,
        threshold=_nonnegative(_float(threshold, "threshold", 0), "threshold"),
        active=active_array,
    )
    return result


@jaxtyped(typechecker=beartype)
def make_information_spectrum(
    input_paths: Tuple[str, ...],
    singular_values: Any,
    right_singular_vectors: Any,
    effective_rank: Any,
    condition_estimate: Any,
    threshold: Any,
) -> InformationSpectrum:
    """Create a validated local information spectrum.

    Carry static scientific vocabulary separately from traced numerical leaves
    while preserving the validation boundary defined by this factory.

    :see: :class:`~.test_certification.TestMakeInformationSpectrum`

    Parameters
    ----------
    input_paths : Tuple[str, ...]
        Input paths used to construct the validated carrier (**static**
        -- a compile-time constant; changing it triggers retracing).
    singular_values : Any
        Singular values used to construct the validated carrier as a
        traced numerical value in the declared physical units.
    right_singular_vectors : Any
        Right singular vectors used to construct the validated carrier
        as a traced numerical value in the declared physical units.
    effective_rank : Any
        Effective rank used to construct the validated carrier as a
        traced numerical value in the declared physical units.
    condition_estimate : Any
        Condition estimate used to construct the validated carrier as a
        traced numerical value in the declared physical units. Zero means that
        there is no active information direction.
    threshold : Any
        Threshold used to construct the validated carrier as a traced
        numerical value in the declared physical units.

    Returns
    -------
    result : InformationSpectrum
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
    paths: Tuple[str, ...] = _text_tuple(input_paths, "input_paths")
    singular_array: Array = _nonnegative(
        _float(singular_values, "singular_values", 1), "singular_values"
    )
    vectors_array: Array = _float(
        right_singular_vectors,
        "right_singular_vectors",
        2,
    )
    if vectors_array.shape != (singular_array.shape[0], len(paths)):
        raise ValueError(
            "right_singular_vectors must have shape (singular values, inputs)"
        )
    result: InformationSpectrum = InformationSpectrum(
        input_paths=paths,
        singular_values=singular_array,
        right_singular_vectors=vectors_array,
        effective_rank=_int(effective_rank, "effective_rank", 0),
        condition_estimate=_float(condition_estimate, "condition_estimate", 0),
        threshold=_nonnegative(_float(threshold, "threshold", 0), "threshold"),
    )
    return result


@jaxtyped(typechecker=beartype)
def make_execution_manifest(
    execution_id: str,
    model_ref: str,
    schema_version: str,
    package_version: str,
    source_checksum: str,
    environment_checksum: str,
    backend: str,
    precision_policy: str,
    deterministic: bool,
    started_at_utc: str,
) -> ExecutionManifest:
    """Create a validated execution manifest.

    Carry static scientific vocabulary separately from traced numerical leaves
    while preserving the validation boundary defined by this factory.

    :see: :class:`~.test_certification.TestMakeExecutionManifest`

    Parameters
    ----------
    execution_id : str
        Execution id used to construct the validated carrier
        (**static** -- a compile-time constant; changing it triggers
        retracing).
    model_ref : str
        Model ref used to construct the validated carrier (**static**
        -- a compile-time constant; changing it triggers retracing).
    schema_version : str
        Schema version used to construct the validated carrier
        (**static** -- a compile-time constant; changing it triggers
        retracing).
    package_version : str
        Package version used to construct the validated carrier
        (**static** -- a compile-time constant; changing it triggers
        retracing).
    source_checksum : str
        Source checksum used to construct the validated carrier
        (**static** -- a compile-time constant; changing it triggers
        retracing).
    environment_checksum : str
        Environment checksum used to construct the validated carrier
        (**static** -- a compile-time constant; changing it triggers
        retracing).
    backend : str
        Backend used to construct the validated carrier (**static** --
        a compile-time constant; changing it triggers retracing).
    precision_policy : str
        Precision policy used to construct the validated carrier
        (**static** -- a compile-time constant; changing it triggers
        retracing).
    deterministic : bool
        Deterministic used to construct the validated carrier
        (**static** -- a compile-time constant; changing it triggers
        retracing).
    started_at_utc : str
        Started at utc used to construct the validated carrier
        (**static** -- a compile-time constant; changing it triggers
        retracing).

    Returns
    -------
    result : ExecutionManifest
        Validated immutable carrier.

    Notes
    -----
    The factory checks static structure eagerly. JAX array operations validate
    numerical values and preserve differentiation behavior.
    """
    result: ExecutionManifest = ExecutionManifest(
        execution_id=_require_text(execution_id, "execution_id"),
        model_ref=_require_text(model_ref, "model_ref"),
        schema_version=_require_text(schema_version, "schema_version"),
        package_version=_require_text(package_version, "package_version"),
        source_checksum=_require_text(source_checksum, "source_checksum"),
        environment_checksum=_require_text(
            environment_checksum, "environment_checksum"
        ),
        backend=_require_text(backend, "backend"),
        precision_policy=_require_text(precision_policy, "precision_policy"),
        deterministic=deterministic,
        started_at_utc=_require_text(started_at_utc, "started_at_utc"),
    )
    return result


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

    :see: :class:`~.test_certification.TestMakePolicyReport`

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
    passed_array: Array = _bool(claim_passed, "claim_passed", 1)
    checked_array: Array = _bool(claim_checked, "claim_checked", 1)
    domain_array: Array = _bool(claim_in_domain, "claim_in_domain", 1)
    achieved_array: Array = _bool(achieved, "achieved", 1)
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
def make_certification_context(
    manifest: ExecutionManifest,
    model: ForwardModelSpec,
    artifacts: Tuple[ArtifactRef, ...] = (),
    transformations: Tuple[TransformationRecord, ...] = (),
    evidence: Tuple[EvidenceRef, ...] = (),
    policy_id: str = "org.diffpes.policy.research.v1",
    check_ids: Tuple[str, ...] = (),
    input_checksums: Tuple[str, ...] = (),
    waivers: Tuple[WaiverRecord, ...] = (),
    attestations: Tuple[HumanAttestationRef, ...] = (),
) -> CertificationContext:
    """Create a prepared certification context.

    Carry static scientific vocabulary separately from traced numerical leaves
    while preserving the validation boundary defined by this factory.

    :see: :class:`~.test_certification.TestMakeCertificationContext`

    Parameters
    ----------
    manifest : ExecutionManifest
        Manifest used to construct the validated carrier as a traced
        numerical value in the declared physical units.
    model : ForwardModelSpec
        Model used to construct the validated carrier as a traced
        numerical value in the declared physical units.
    artifacts : Tuple[ArtifactRef, ...]
        Artifacts used to construct the validated carrier (**static**
        -- a compile-time constant; changing it triggers retracing).
    transformations : Tuple[TransformationRecord, ...]
        Transformations used to construct the validated carrier
        (**static** -- a compile-time constant; changing it triggers
        retracing).
    evidence : Tuple[EvidenceRef, ...]
        Evidence used to construct the validated carrier (**static** --
        a compile-time constant; changing it triggers retracing).
    policy_id : str
        Policy id used to construct the validated carrier (**static**
        -- a compile-time constant; changing it triggers retracing).
    check_ids : Tuple[str, ...]
        Check ids used to construct the validated carrier (**static**
        -- a compile-time constant; changing it triggers retracing).
    input_checksums : Tuple[str, ...]
        Input checksums used to construct the validated carrier
        (**static** -- a compile-time constant; changing it triggers
        retracing).
    waivers : Tuple[WaiverRecord, ...]
        Policy-waiver records. Default is an empty tuple.
    attestations : Tuple[HumanAttestationRef, ...]
        Human-review records kept separate from evidence.

    Returns
    -------
    result : CertificationContext
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
    expected_ref: str = f"{model.model_id}@{model.model_version}"
    if manifest.model_ref not in (model.model_id, expected_ref):
        raise ValueError(
            "manifest model_ref does not match model specification"
        )
    result: CertificationContext = CertificationContext(
        manifest=manifest,
        model=model,
        artifacts=tuple(artifacts),
        transformations=tuple(transformations),
        evidence=tuple(evidence),
        attestations=tuple(attestations),
        policy_id=_require_text(policy_id, "policy_id"),
        check_ids=_text_tuple(check_ids, "check_ids"),
        input_checksums=_text_tuple(input_checksums, "input_checksums"),
        waivers=tuple(waivers),
    )
    return result


def _unique_module_ids(values: Tuple[Any, ...], attribute: str) -> bool:
    """PRIVATE: Return whether a module tuple contains unique named identities.

    Parameters
    ----------
    values : Tuple[Any, ...]
        Module instances that carry the identity attribute.
    attribute : str
        Attribute name that stores each identity.

    Returns
    -------
    unique : bool
        True when no two entries share the same identity value.

    Implementation Logic
    --------------------
    Collect ``getattr(value, attribute)`` for every entry and compare
    the set size against the tuple length.
    """
    identities: Tuple[Any, ...] = tuple(
        getattr(value, attribute) for value in values
    )
    unique: bool = len(identities) == len(set(identities))
    return unique


@jaxtyped(typechecker=beartype)
def make_forward_certificate(  # noqa: PLR0913
    manifest: ExecutionManifest,
    model: ForwardModelSpec,
    artifacts: Tuple[ArtifactRef, ...],
    transformations: Tuple[TransformationRecord, ...],
    evidence: Tuple[EvidenceRef, ...],
    claims: Tuple[CertificationClaim, ...],
    domains: Tuple[DomainResult, ...],
    derivatives: DerivativeEvidence,
    dependencies: DependencyMap,
    sensitivities: SensitivityMap,
    information: InformationSpectrum,
    policy_report: PolicyReport,
    policy_id: str,
    certificate_checksum: str,
    extensions_json: str = "{}",
    waivers: Tuple[WaiverRecord, ...] = (),
    attestations: Tuple[HumanAttestationRef, ...] = (),
) -> ForwardCertificate:
    """Create and cross-validate a complete forward certificate.

    Carry static scientific vocabulary separately from traced numerical leaves
    while preserving the validation boundary defined by this factory.

    :see: :class:`~.test_certification.TestMakeForwardCertificate`

    Parameters
    ----------
    manifest : ExecutionManifest
        Manifest used to construct the validated carrier as a traced
        numerical value in the declared physical units.
    model : ForwardModelSpec
        Model used to construct the validated carrier as a traced
        numerical value in the declared physical units.
    artifacts : Tuple[ArtifactRef, ...]
        Artifacts used to construct the validated carrier (**static**
        -- a compile-time constant; changing it triggers retracing).
    transformations : Tuple[TransformationRecord, ...]
        Transformations used to construct the validated carrier
        (**static** -- a compile-time constant; changing it triggers
        retracing).
    evidence : Tuple[EvidenceRef, ...]
        Evidence used to construct the validated carrier (**static** --
        a compile-time constant; changing it triggers retracing).
    claims : Tuple[CertificationClaim, ...]
        Claims used to construct the validated carrier (**static** -- a
        compile-time constant; changing it triggers retracing).
    domains : Tuple[DomainResult, ...]
        Domains used to construct the validated carrier (**static** --
        a compile-time constant; changing it triggers retracing).
    derivatives : DerivativeEvidence
        Derivatives used to construct the validated carrier as a traced
        numerical value in the declared physical units.
    dependencies : DependencyMap
        Dependencies used to construct the validated carrier as a
        traced numerical value in the declared physical units.
    sensitivities : SensitivityMap
        Sensitivities used to construct the validated carrier as a
        traced numerical value in the declared physical units.
    information : InformationSpectrum
        Information used to construct the validated carrier as a traced
        numerical value in the declared physical units.
    policy_report : PolicyReport
        Policy report used to construct the validated carrier as a
        traced numerical value in the declared physical units.
    policy_id : str
        Policy id used to construct the validated carrier (**static**
        -- a compile-time constant; changing it triggers retracing).
    certificate_checksum : str
        Certificate checksum used to construct the validated carrier
        (**static** -- a compile-time constant; changing it triggers
        retracing).
    extensions_json : str
        Extensions json used to construct the validated carrier
        (**static** -- a compile-time constant; changing it triggers
        retracing).
    waivers : Tuple[WaiverRecord, ...]
        Policy-waiver records. Default is an empty tuple.
    attestations : Tuple[HumanAttestationRef, ...]
        Human-review records kept separate from evidence.

    Returns
    -------
    result : ForwardCertificate
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
    values: Any
    attribute: Any
    expected_ref: str = f"{model.model_id}@{model.model_version}"
    if manifest.model_ref not in (model.model_id, expected_ref):
        raise ValueError(
            "manifest model_ref does not match model specification"
        )
    if policy_report.policy_id != policy_id:
        raise ValueError("policy_report policy_id does not match certificate")
    if dependencies.model_id != model.model_id:
        raise ValueError(
            "dependency model_id does not match model specification"
        )
    attestation_ids: frozenset[str] = frozenset(
        item.attestation_id for item in attestations
    )
    unresolved_attestations: Tuple[str, ...] = tuple(
        reference
        for item in evidence
        for reference in item.human_attestation_refs
        if reference not in attestation_ids
    )
    if unresolved_attestations:
        raise ValueError(
            "evidence references missing human attestations: "
            + ", ".join(unresolved_attestations)
        )
    identity_groups: Tuple[Tuple[Tuple[Any, ...], str], ...] = (
        (artifacts, "artifact_id"),
        (transformations, "transformation_id"),
        (evidence, "evidence_id"),
        (claims, "claim_id"),
        (domains, "predicate_id"),
        (waivers, "waiver_id"),
        (attestations, "attestation_id"),
    )
    for values, attribute in identity_groups:
        if not _unique_module_ids(values, attribute):
            raise ValueError(f"certificate contains duplicate {attribute}")
    result: ForwardCertificate = ForwardCertificate(
        manifest=manifest,
        model=model,
        artifacts=tuple(artifacts),
        transformations=tuple(transformations),
        evidence=tuple(evidence),
        attestations=tuple(attestations),
        claims=tuple(claims),
        domains=tuple(domains),
        derivatives=derivatives,
        dependencies=dependencies,
        sensitivities=sensitivities,
        information=information,
        policy_report=policy_report,
        policy_id=_require_text(policy_id, "policy_id"),
        certificate_checksum=_require_text(
            certificate_checksum, "certificate_checksum"
        ),
        extensions_json=_json_object(extensions_json, "extensions_json"),
        waivers=tuple(waivers),
    )
    return result


@jaxtyped(typechecker=beartype)
def make_certified_result(
    value: Any, certificate: ForwardCertificate
) -> CertifiedResult:
    """Pair any JAX-compatible result value with a forward certificate.

    Carry static scientific vocabulary separately from traced numerical leaves
    while preserving the validation boundary defined by this factory.

    :see: :class:`~.test_certification.TestMakeCertifiedResult`

    Parameters
    ----------
    value : Any
        Value used to construct the validated carrier as a traced
        numerical value in the declared physical units.
    certificate : ForwardCertificate
        Certificate used to construct the validated carrier as a traced
        numerical value in the declared physical units.

    Returns
    -------
    result : CertifiedResult
        Validated immutable carrier.

    Notes
    -----
    The factory checks static structure eagerly. JAX array operations validate
    numerical values and preserve differentiation behavior.
    """
    result: CertifiedResult = CertifiedResult(
        value=value,
        certificate=certificate,
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

    :see: :class:`~.test_certification.TestMakeEvidenceReport`

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

    :see: :class:`~.test_certification.TestMakeVerificationReport`

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

    :see: :class:`~.test_certification.TestMakeReproductionReport`

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
def make_registration_handshake(
    owner_id: str,
    model_refs: Tuple[str, ...] = (),
    transformation_refs: Tuple[str, ...] = (),
    convention_refs: Tuple[str, ...] = (),
    evidence_ids: Tuple[str, ...] = (),
) -> RegistrationHandshake:
    """Create declarative registration requirements for one plan owner.

    The factory validates each static identity without importing a scientific
    executor.

    :see: :class:`~.test_certification.TestMakeRegistrationHandshake`

    Parameters
    ----------
    owner_id : str
        Plan owner identity (**static**; changing it causes retracing).
    model_refs : Tuple[str, ...]
        Required model identities. Default is an empty tuple.
    transformation_refs : Tuple[str, ...]
        Required transformation identities. Default is an empty tuple.
    convention_refs : Tuple[str, ...]
        Required convention identities. Default is an empty tuple.
    evidence_ids : Tuple[str, ...]
        Required evidence identities. Default is an empty tuple.

    Returns
    -------
    result : RegistrationHandshake
        Validated declarative registration requirements.

    Notes
    -----
    The factory validates each static identity before module construction.
    """
    result: RegistrationHandshake = RegistrationHandshake(
        owner_id=_require_text(owner_id, "owner_id"),
        model_refs=_text_tuple(model_refs, "model_refs"),
        transformation_refs=_text_tuple(
            transformation_refs, "transformation_refs"
        ),
        convention_refs=_text_tuple(convention_refs, "convention_refs"),
        evidence_ids=_text_tuple(evidence_ids, "evidence_ids"),
    )
    return result


@jaxtyped(typechecker=beartype)
def make_handshake_report(
    owner_id: str,
    complete: Any,
    missing_ids: Tuple[str, ...] = (),
) -> HandshakeReport:
    """Create a report for one registration handshake.

    The report keeps the completion outcome as a JAX Boolean leaf.

    :see: :class:`~.test_certification.TestMakeHandshakeReport`

    Parameters
    ----------
    owner_id : str
        Plan owner identity (**static**; changing it causes retracing).
    complete : Any
        Whether every declared identity has a registry binding.
    missing_ids : Tuple[str, ...]
        Missing declared identities. Default is an empty tuple.

    Returns
    -------
    result : HandshakeReport
        Validated completion outcome and missing identities.

    Notes
    -----
    The factory converts the completion outcome to a scalar JAX Boolean.
    """
    missing: Tuple[str, ...] = _text_tuple(missing_ids, "missing_ids")
    result: HandshakeReport = HandshakeReport(
        owner_id=_require_text(owner_id, "owner_id"),
        complete=_bool(complete, "complete", 0),
        missing_ids=missing,
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

    :see: :class:`~.test_certification.TestMakeWaiverRecord`

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

    :see: :class:`~.test_certification.TestMakeWaiverReport`

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
    "ArtifactRef",
    "ArtifactResolver",
    "CertificationClaim",
    "CertificationContext",
    "CheckFunction",
    "CertifiedResult",
    "ConventionRef",
    "DependencyMap",
    "DerivativeEvidence",
    "DomainPredicate",
    "DomainResult",
    "EvidenceLineage",
    "EvidenceRef",
    "EvidenceReport",
    "ExecutionManifest",
    "ForwardCertificate",
    "ForwardModelSpec",
    "HandshakeReport",
    "HumanAttestationRef",
    "InformationSpectrum",
    "PolicyReport",
    "RegisteredModel",
    "RegisteredTransformation",
    "RegistryReport",
    "RegistrySnapshot",
    "RegistrationHandshake",
    "ReproductionReport",
    "SensitivityMap",
    "TransformationRecord",
    "VerificationReport",
    "WaiverRecord",
    "WaiverReport",
    "make_artifact_ref",
    "make_certification_claim",
    "make_certification_context",
    "make_certified_result",
    "make_convention_ref",
    "make_dependency_map",
    "make_derivative_evidence",
    "make_domain_predicate",
    "make_domain_result",
    "make_evidence_lineage",
    "make_evidence_ref",
    "make_evidence_report",
    "make_execution_manifest",
    "make_forward_certificate",
    "make_forward_model_spec",
    "make_handshake_report",
    "make_human_attestation_ref",
    "make_information_spectrum",
    "make_policy_report",
    "make_registered_model",
    "make_registered_transformation",
    "make_registry_report",
    "make_registry_snapshot",
    "make_registration_handshake",
    "make_reproduction_report",
    "make_sensitivity_map",
    "make_transformation_record",
    "make_verification_report",
    "make_waiver_record",
    "make_waiver_report",
]
