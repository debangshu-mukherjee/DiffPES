"""Execute JAX-native certified forward models.

Extended Summary
----------------
This module separates filesystem and registry preparation from the pure
compiled scientific kernel. The kernel evaluates the registered forward model
once and retains its linearization. It then derives continuous claims and
derivative evidence. The kernel returns an Equinox certificate with numerical
leaves that support JIT, VMAP, JVP, and VJP transformations.

Routine Listings
----------------
:func:`certify_forward`
    Execute a prepared model and produce a certified JAX PyTree.
:func:`certify_forward_checked`
    Execute certification and return structured hard-domain errors.
:func:`prepare_certification`
    Resolve static scientific records before compiled execution.
:func:`verify_certificate`
    Re-evaluate numerical claim and policy consistency without a rerun.
"""

from functools import cache

import equinox as eqx
import jax
import jax.numpy as jnp
from beartype import beartype
from beartype.typing import Any, Optional
from jax import core
from jax.experimental import checkify
from jaxtyping import Array, Float, PyTree, jaxtyped

from diffpes.types import (
    ArtifactRef,
    ArtifactResolver,
    CertificationClaim,
    CertificationContext,
    CertifiedResult,
    CheckFunction,
    DependencyMap,
    DerivativeEvidence,
    DomainResult,
    EvidenceRef,
    EvidenceReport,
    ExecutionManifest,
    ForwardCertificate,
    HumanAttestationRef,
    InformationSpectrum,
    PolicyReport,
    RegisteredModel,
    SensitivityMap,
    TransformationRecord,
    VerificationReport,
    WaiverRecord,
    make_certification_claim,
    make_certification_context,
    make_certified_result,
    make_forward_certificate,
    make_verification_report,
)

from .checks import get_check
from .dependencies import (
    _dependency_map_from_linearization,
    _dependency_structure,
    _information_spectrum_from_linearization,
    _ravel_real_pytree,
    _sensitivity_map_from_linearization,
)
from .evidence import _derivative_evidence_from_linearization, evaluate_claim
from .policy import evaluate_policy
from .registry import get_model
from .resolvers import verify_evidence
from .waivers import require_active_waivers


def _path_names(tree: PyTree) -> tuple[str, ...]:
    """Return stable real-coordinate paths for numerical input leaves."""
    path: Any
    leaf: Any
    flattened: Any = jax.tree_util.tree_flatten_with_path(tree)
    path_leaves: Any = flattened[0]
    paths: list[str] = []
    for path, leaf in path_leaves:
        name: str = jax.tree_util.keystr(path) or "$"
        if jnp.iscomplexobj(jnp.asarray(leaf)):
            paths.extend((f"{name}.real", f"{name}.imag"))
        else:
            paths.append(name)
    result: tuple[str, ...] = tuple(paths)
    return result


@cache
def _checked_kernel() -> Any:
    """Return the compiled kernel with functional hard-domain checks."""
    transformed: Any = checkify.checkify(_certify_kernel)
    compiled: Any = eqx.filter_jit(transformed)
    return compiled


def _certify_forward_checked(
    context: CertificationContext,
    inputs: PyTree,
    *,
    directions: Optional[PyTree],
    scales: Optional[Float[Array, " n_probe"]],
    spectrum_rank: int,
) -> tuple[Any, CertifiedResult]:
    """Resolve eager records and run one structured checked kernel."""
    registered: RegisteredModel = get_model(
        context.model.model_id,
        context.model.model_version,
    )
    resolved_directions: PyTree = (
        _probe_directions(inputs) if directions is None else directions
    )
    n_probes: int = jax.tree.leaves(resolved_directions)[0].shape[0]
    resolved_scales: Array = (
        jnp.ones(n_probes, dtype=jnp.float64)
        if scales is None
        else jnp.asarray(scales, dtype=jnp.float64)
    )
    domain_checks: tuple[CheckFunction, ...] = tuple(
        get_check(check_id) for check_id in context.check_ids
    )
    structural_evaluation: tuple[PyTree, Array] = _dependency_structure(
        context.model.model_id,
        registered.executor,
        inputs,
    )
    structural: Array = structural_evaluation[1]
    checked: tuple[Any, CertifiedResult] = _checked_kernel()(
        registered.executor,
        context,
        inputs,
        resolved_directions,
        resolved_scales,
        domain_checks,
        structural,
        spectrum_rank=spectrum_rank,
    )
    return checked


def _probe_directions(inputs: PyTree) -> PyTree:
    """Construct one all-ones directional probe per input leaf."""
    array: Any
    flattened: tuple[list[Any], Any] = jax.tree_util.tree_flatten(inputs)
    leaves: list[Any] = flattened[0]
    treedef: Any = flattened[1]
    arrays: list[Array] = [jnp.asarray(leaf) for leaf in leaves]
    n_leaves: int = sum(
        2 if jnp.iscomplexobj(array) else 1 for array in arrays
    )
    batched: list[Array] = []
    selected: int = 0
    for array in arrays:
        shape: tuple[int, ...] = (n_leaves, *array.shape)
        probe: Array = jnp.zeros(shape, dtype=array.dtype)
        probe = probe.at[selected].set(jnp.ones_like(array))
        selected += 1
        if jnp.iscomplexobj(array):
            probe = probe.at[selected].set(1j * jnp.ones_like(array))
            selected += 1
        batched.append(probe)
    directions: PyTree = jax.tree_util.tree_unflatten(treedef, batched)
    return directions


@jaxtyped(typechecker=beartype)
def prepare_certification(  # noqa: PLR0913
    model_id: str,
    model_version: str,
    manifest: ExecutionManifest,
    *,
    policy_id: str = "org.diffpes.policy.research.v1",
    artifacts: tuple[ArtifactRef, ...] = (),
    transformations: tuple[TransformationRecord, ...] = (),
    evidence: tuple[EvidenceRef, ...] = (),
    attestations: tuple[HumanAttestationRef, ...] = (),
    check_ids: tuple[str, ...] = (),
    input_checksums: tuple[str, ...] = (),
    waivers: tuple[WaiverRecord, ...] = (),
) -> CertificationContext:
    """Resolve static scientific records before compiled execution.

    The operation binds a registered forward model to explicit evidence and
    policy records. Numerical outputs and assurance leaves remain
    differentiable.

    :see: :class:`~.test_execution.TestPrepareCertification`

    Implementation Logic
    --------------------
    1. **Resolve the registered model**::

           registered: RegisteredModel = get_model(model_id, model_version)

       Exact lookup binds the context to one stable scientific model identity
       before the function collects evidence and domain checks.

    Parameters
    ----------
    model_id : str
        Permanent model identifier (**static** -- changing it retraces).
    model_version : str
        Exact semantic model version (**static** -- changing it retraces).
    manifest : ExecutionManifest
        Prepared execution identity and numerical environment.
    policy_id : str
        Certification policy identity (**static** -- changing it retraces).
    artifacts : tuple[ArtifactRef, ...]
        Input and derived artifact references.
    transformations : tuple[TransformationRecord, ...]
        Ordered information-flow records.
    evidence : tuple[EvidenceRef, ...]
        Numerical evidence with named lineage.
    attestations : tuple[HumanAttestationRef, ...]
        Human reviews kept separate from computational evidence.
    check_ids : tuple[str, ...]
        Domain checks to run (**static** -- changing them retraces).
    input_checksums : tuple[str, ...]
        Bookkeeping identities for inputs (**static** -- a change retraces).
    waivers : tuple[WaiverRecord, ...]
        Policy-waiver records. Default is an empty tuple.

    Returns
    -------
    context : CertificationContext
        Cross-validated static selections for compiled execution.

    Raises
    ------
    ValueError
        If a waiver does not match the selected policy or active UTC interval.
    """
    registered: RegisteredModel = get_model(model_id, model_version)
    mismatched_waivers: tuple[str, ...] = tuple(
        waiver.waiver_id for waiver in waivers if waiver.policy_id != policy_id
    )
    if mismatched_waivers:
        msg: str = (
            "waiver policy does not match selected policy: "
            + ", ".join(mismatched_waivers)
        )
        raise ValueError(msg)
    require_active_waivers(waivers, as_of_utc=manifest.started_at_utc)
    context: CertificationContext = make_certification_context(
        manifest=manifest,
        model=registered.spec,
        artifacts=artifacts,
        transformations=transformations,
        evidence=evidence,
        attestations=attestations,
        policy_id=policy_id,
        check_ids=(
            check_ids
            if check_ids
            else tuple(item.predicate_id for item in registered.spec.domain)
        ),
        input_checksums=input_checksums,
        waivers=waivers,
    )
    return context


def _evidence_claims(
    model_id: str, evidence: tuple[EvidenceRef, ...]
) -> tuple[CertificationClaim, ...]:
    """Convert registered numerical evidence into verification claims."""
    item: Any
    claims: list[CertificationClaim] = []
    for item in evidence:
        claim: CertificationClaim = evaluate_claim(
            claim_id=f"claim.{item.evidence_id}",
            subject_id=model_id,
            predicate_id="verification.external_reference",
            measured=item.measured,
            reference=item.reference,
            tolerance=item.tolerance,
            evidence_ids=(item.evidence_id,),
        )
        claims.append(claim)
    result: tuple[CertificationClaim, ...] = tuple(claims)
    return result


def _certify_kernel(  # noqa: PLR0915
    executor: Any,
    context: CertificationContext,
    inputs: PyTree,
    directions: PyTree,
    scales: Float[Array, " n_probe"],
    domain_checks: tuple[CheckFunction, ...],
    structural: Array,
    *,
    spectrum_rank: int,
) -> CertifiedResult:
    """Run the pure compiled certification computation."""
    linearized: tuple[PyTree, Any] = jax.linearize(executor, inputs)
    value: PyTree = linearized[0]
    tree_pushforward: Any = linearized[1]

    def vector_pushforward(tangent: PyTree) -> Array:
        output_tangent: PyTree = tree_pushforward(tangent)
        result: Array = _ravel_real_pytree(output_tangent)[0]
        return result

    flat_value: Array = _ravel_real_pytree(value)[0]
    output_size: int = flat_value.size
    input_paths: tuple[str, ...] = _path_names(inputs)
    n_probes: int = len(input_paths)
    cotangent_indices: Array = jnp.arange(n_probes) % output_size
    cotangents: Array = jnp.eye(output_size, dtype=jnp.float64)[
        cotangent_indices
    ]
    output_ids: tuple[str, ...] = tuple(
        f"output[{index}]" for index in range(output_size)
    )
    transposed: Any = jax.linear_transpose(vector_pushforward, inputs)

    def pullback(cotangent: Array) -> PyTree:
        pulled: PyTree = transposed(cotangent)[0]
        return pulled

    flat_inputs: Array
    unravel_inputs: Any
    flat_inputs, unravel_inputs = _ravel_real_pytree(inputs)

    def flat_pushforward(tangent: Array) -> Array:
        result: Array = vector_pushforward(unravel_inputs(tangent))
        return result

    flat_transposed: Any = jax.linear_transpose(
        flat_pushforward,
        flat_inputs,
    )

    def flat_pullback(cotangent: Array) -> Array:
        pulled: Array = flat_transposed(cotangent)[0]
        return pulled

    information: InformationSpectrum
    information = _information_spectrum_from_linearization(
        inputs,
        flat_value,
        flat_pushforward,
        flat_pullback,
        input_paths=input_paths,
        rank=spectrum_rank,
    )

    def forward_vector(candidate: PyTree) -> Array:
        candidate_value: PyTree = executor(candidate)
        result: Array = _ravel_real_pytree(candidate_value)[0]
        return result

    derivatives: DerivativeEvidence = _derivative_evidence_from_linearization(
        forward_vector,
        inputs,
        directions,
        cotangents,
        vector_pushforward,
        pullback,
        information,
        input_paths=input_paths,
        output_projection_ids=output_ids,
        scales=scales,
    )
    dependencies: DependencyMap = _dependency_map_from_linearization(
        context.model.model_id,
        inputs,
        value,
        structural,
        tree_pushforward,
    )
    sensitivities: SensitivityMap = _sensitivity_map_from_linearization(
        input_paths,
        output_ids,
        directions,
        scales,
        vector_pushforward,
    )
    domains: tuple[DomainResult, ...] = tuple(
        check_fn(inputs) for check_fn in domain_checks
    )
    domain: DomainResult
    hard_severity_code: int = 2
    for domain in domains:
        hard_passed: Array = (
            domain.severity_code < hard_severity_code
        ) | domain.passed
        checkify.check(
            hard_passed,
            f"hard domain check failed: {domain.predicate_id}",
        )
    domain_claims: tuple[CertificationClaim, ...] = tuple(
        make_certification_claim(
            claim_id=f"claim.{domain.predicate_id}",
            subject_id=context.model.model_id,
            predicate_id=f"domain.{domain.predicate_id}",
            evidence_ids=(),
            measured=jnp.atleast_1d(domain.measured),
            reference=jnp.atleast_1d(domain.reference),
            residual=jnp.atleast_1d(domain.residual),
            tolerance=jnp.atleast_1d(domain.tolerance),
            passed=domain.passed,
            checked=domain.checked,
            in_domain=domain.in_domain,
            margin=domain.margin,
            severity_code=domain.severity_code,
        )
        for domain in domains
    )
    identity_claim: CertificationClaim = evaluate_claim(
        claim_id="claim.execution.identified",
        subject_id=context.model.model_id,
        predicate_id="identity.model_and_inputs",
        measured=jnp.zeros(1),
        reference=jnp.zeros(1),
        tolerance=jnp.zeros(1),
    )
    nonfinite_count: Array = jnp.sum(~jnp.isfinite(flat_value))
    output_claim: CertificationClaim = evaluate_claim(
        claim_id="claim.output.finite",
        subject_id=context.model.observable_id,
        predicate_id="output.finite",
        measured=jnp.asarray([nonfinite_count], dtype=jnp.float64),
        reference=jnp.zeros(1),
        tolerance=jnp.zeros(1),
    )
    derivative_error: Array = jnp.max(
        jnp.abs(derivatives.derivative_residuals)
    )
    derivative_tolerance: Array = 1e-9 + 1e-6 * jnp.max(
        jnp.abs(derivatives.reference_derivatives)
    )
    derivative_claim: CertificationClaim = evaluate_claim(
        claim_id="claim.derivative.fd_correct",
        subject_id=context.model.model_id,
        predicate_id="derivative.jvp_matches_central_fd",
        measured=jnp.asarray([derivative_error]),
        reference=jnp.zeros(1),
        tolerance=jnp.asarray([derivative_tolerance]),
    )
    external_claims: tuple[CertificationClaim, ...] = _evidence_claims(
        context.model.model_id, context.evidence
    )
    claims: tuple[CertificationClaim, ...] = (
        identity_claim,
        output_claim,
        derivative_claim,
        *domain_claims,
        *external_claims,
    )
    policy_report: PolicyReport = evaluate_policy(
        claims,
        context.policy_id,
        evidence=context.evidence,
        implementation_ref=context.model.implementation_ref,
        artifact_ids=tuple(item.artifact_id for item in context.artifacts),
        waivers=context.waivers,
    )
    certificate: ForwardCertificate = make_forward_certificate(
        manifest=context.manifest,
        model=context.model,
        artifacts=context.artifacts,
        transformations=context.transformations,
        domains=domains,
        evidence=context.evidence,
        attestations=context.attestations,
        claims=claims,
        derivatives=derivatives,
        dependencies=dependencies,
        sensitivities=sensitivities,
        information=information,
        policy_report=policy_report,
        policy_id=context.policy_id,
        certificate_checksum="pending-canonical-serialization",
        extensions_json="{}",
        waivers=context.waivers,
    )
    result: CertifiedResult = make_certified_result(
        value=value,
        certificate=certificate,
    )
    return result


@jaxtyped(typechecker=beartype)
def certify_forward(
    context: CertificationContext,
    inputs: PyTree,
    *,
    directions: Optional[PyTree] = None,
    scales: Optional[Float[Array, " n_probe"]] = None,
    spectrum_rank: int = 8,
) -> CertifiedResult:
    """Execute a prepared model and produce a certified JAX PyTree.

    The operation binds a registered forward model to explicit evidence and
    policy records. Numerical outputs and assurance leaves remain
    differentiable.

    :see: :class:`~.test_execution.TestCertifyForward`

    Parameters
    ----------
    context : CertificationContext
        Prepared model, policy, evidence, and domain-check selections.
    inputs : PyTree
        Numerical model inputs in the model's declared physical units.
    directions : Optional[PyTree]
        Batched tangent probes. By default, the function builds one probe per
        real input coordinate.
    scales : Optional[Float[Array, " n_probe"]]
        Positive physical scale for every tangent probe.
    spectrum_rank : int
        Requested information-spectrum rank (**static** -- a change retraces).

    Returns
    -------
    result : CertifiedResult
        Forward value paired with differentiable evidence and policy outcome.

    Notes
    -----
    The result value, residuals, margins, sensitivities, and information
    spectrum remain differentiable with respect to numerical input leaves.
    """
    error: Any
    result: CertifiedResult
    error, result = _certify_forward_checked(
        context,
        inputs,
        directions=directions,
        scales=scales,
        spectrum_rank=spectrum_rank,
    )
    contains_tracer: bool = any(
        isinstance(leaf, core.Tracer) for leaf in jax.tree.leaves(inputs)
    )
    if not contains_tracer:
        error.throw()
    return result


@jaxtyped(typechecker=beartype)
def certify_forward_checked(
    context: CertificationContext,
    inputs: PyTree,
    *,
    directions: Optional[PyTree] = None,
    scales: Optional[Float[Array, " n_probe"]] = None,
    spectrum_rank: int = 8,
) -> tuple[Any, CertifiedResult]:
    """Execute certification and return structured hard-domain errors.

    The function returns a ``checkify.Error`` with the certified result. The
    caller controls when the structured error becomes an exception.

    :see: :class:`~.test_execution.TestCertifyForwardChecked`

    Parameters
    ----------
    context : CertificationContext
        Prepared model, policy, evidence, and domain-check selections.
    inputs : PyTree
        Numerical model inputs in the declared physical units.
    directions : Optional[PyTree]
        Batched tangent probes. Default None creates one probe per coordinate.
    scales : Optional[Float[Array, " n_probe"]]
        Positive physical scale for every tangent probe. Default None.
    spectrum_rank : int
        Requested information-spectrum rank (**static**). Default 8.

    Returns
    -------
    checked : tuple[Any, CertifiedResult]
        Structured checkify error and complete certified result.

    Notes
    -----
    The structured error remains compatible with JIT and VMAP. Call
    ``error.throw()`` only at an eager boundary.
    """
    checked: tuple[Any, CertifiedResult] = _certify_forward_checked(
        context,
        inputs,
        directions=directions,
        scales=scales,
        spectrum_rank=spectrum_rank,
    )
    return checked


def _claim_is_consistent(
    claim: CertificationClaim,
    domains: dict[str, DomainResult],
) -> bool:
    """Return whether one claim agrees with its continuous evidence."""
    if claim.predicate_id.startswith("domain."):
        domain_id: str = claim.predicate_id.removeprefix("domain.")
        domain: DomainResult | None = domains.get(domain_id)
        if domain is None:
            consistent: bool = False
            return consistent
        comparisons: tuple[tuple[Any, Any], ...] = (
            (claim.measured, jnp.atleast_1d(domain.measured)),
            (claim.reference, jnp.atleast_1d(domain.reference)),
            (claim.residual, jnp.atleast_1d(domain.residual)),
            (claim.tolerance, jnp.atleast_1d(domain.tolerance)),
            (claim.margin, domain.margin),
            (claim.passed, domain.passed),
            (claim.checked, domain.checked),
            (claim.in_domain, domain.in_domain),
            (claim.severity_code, domain.severity_code),
        )
        consistent = all(
            bool(jnp.array_equal(left, right)) for left, right in comparisons
        )
        return consistent  # noqa: RET504
    expected_residual: Array = claim.measured - claim.reference
    expected_margin: Array = jnp.min(
        claim.tolerance - jnp.abs(expected_residual)
    )
    expected_passed: Array = (
        claim.checked
        & claim.in_domain
        & jnp.all(jnp.abs(expected_residual) <= claim.tolerance)
    )
    comparisons = (
        (claim.residual, expected_residual),
        (claim.margin, expected_margin),
        (claim.passed, expected_passed),
    )
    consistent = all(
        bool(jnp.array_equal(left, right)) for left, right in comparisons
    )
    return consistent  # noqa: RET504


def _external_claim_matches_evidence(
    claim: CertificationClaim,
    evidence_by_id: dict[str, EvidenceRef],
) -> bool:
    """Return whether an external claim mirrors its attached evidence."""
    if claim.predicate_id != "verification.external_reference":
        consistent: bool = True
        return consistent  # noqa: RET504
    if len(claim.evidence_ids) != 1:
        consistent = False
        return consistent  # noqa: RET504
    evidence: EvidenceRef | None = evidence_by_id.get(claim.evidence_ids[0])
    if evidence is None:
        consistent = False
        return consistent  # noqa: RET504
    comparisons: tuple[tuple[Any, Any], ...] = (
        (claim.measured, evidence.measured),
        (claim.reference, evidence.reference),
        (claim.residual, evidence.residual),
        (claim.tolerance, evidence.tolerance),
    )
    consistent = all(
        bool(jnp.array_equal(left, right)) for left, right in comparisons
    )
    return consistent  # noqa: RET504


@jaxtyped(typechecker=beartype)
def verify_certificate(
    certificate: ForwardCertificate,
    *,
    resolver: Optional[ArtifactResolver] = None,
) -> VerificationReport:
    """Re-evaluate numerical claim and policy consistency without a rerun.

    The operation binds a registered forward model to explicit evidence and
    policy records. Numerical outputs and assurance leaves remain
    differentiable.

    :see: :class:`~.test_execution.TestVerifyCertificate`

    Parameters
    ----------
    certificate : ForwardCertificate
        Concrete certificate. The function checks its internal numerical
        relations.
    resolver : Optional[ArtifactResolver]
        Artifact resolver for external evidence. Default None checks only
        internal consistency and does not make a resolution claim.

    Returns
    -------
    report : VerificationReport
        Structural and policy consistency outcome.

    Notes
    -----
    Verification recomputes recorded relations only. It does not rerun the
    forward model or convert bookkeeping checksums into scientific evidence.
    """
    domains: dict[str, DomainResult] = {
        domain.predicate_id: domain for domain in certificate.domains
    }
    evidence_by_id: dict[str, EvidenceRef] = {
        evidence.evidence_id: evidence for evidence in certificate.evidence
    }
    claims_consistent: bool = all(
        _claim_is_consistent(claim, domains)
        and _external_claim_matches_evidence(claim, evidence_by_id)
        for claim in certificate.claims
    )
    evidence_ids: frozenset[str] = frozenset(
        item.evidence_id for item in certificate.evidence
    )
    references_consistent: bool = all(
        evidence_id in evidence_ids
        for claim in certificate.claims
        for evidence_id in claim.evidence_ids
    )
    artifact_ids: frozenset[str] = frozenset(
        item.artifact_id for item in certificate.artifacts
    )
    artifact_refs_consistent: bool = all(
        artifact_id in artifact_ids
        for evidence in certificate.evidence
        for artifact_id in evidence.lineage.artifact_refs
    )
    evidence_residuals_consistent: bool = all(
        bool(
            jnp.array_equal(
                evidence.residual,
                evidence.measured - evidence.reference,
            )
        )
        for evidence in certificate.evidence
    )
    evidence_reports: tuple[EvidenceReport, ...] = ()
    if resolver is not None:
        evidence_reports = tuple(
            verify_evidence(item, certificate.artifacts, resolver)
            for item in certificate.evidence
        )
    recomputed: PolicyReport = evaluate_policy(
        certificate.claims,
        certificate.policy_id,
        evidence=certificate.evidence,
        evidence_reports=evidence_reports,
        implementation_ref=certificate.model.implementation_ref,
        artifact_ids=tuple(item.artifact_id for item in certificate.artifacts),
        waivers=certificate.waivers,
    )
    resolved_evidence_valid: bool = resolver is None or all(
        bool(report.passed) for report in evidence_reports
    )
    resolver_may_upgrade_policy: bool = resolver is not None and (
        certificate.policy_id
        in {
            "org.diffpes.policy.publication.v1",
            "org.diffpes.policy.parity.v1",
        }
    )
    achievement_consistent: bool = (
        bool(
            jnp.all(~certificate.policy_report.achieved | recomputed.achieved)
        )
        if resolver_may_upgrade_policy
        else bool(
            jnp.array_equal(
                recomputed.achieved,
                certificate.policy_report.achieved,
            )
        )
    )
    policy_consistent: bool = (
        recomputed.level_ids == certificate.policy_report.level_ids
        and recomputed.required_claim_ids
        == certificate.policy_report.required_claim_ids
        and bool(
            jnp.array_equal(
                recomputed.claim_passed,
                certificate.policy_report.claim_passed,
            )
        )
        and bool(
            jnp.array_equal(
                recomputed.claim_checked,
                certificate.policy_report.claim_checked,
            )
        )
        and bool(
            jnp.array_equal(
                recomputed.claim_in_domain,
                certificate.policy_report.claim_in_domain,
            )
        )
        and achievement_consistent
    )
    report: VerificationReport = make_verification_report(
        certificate_checksum=certificate.certificate_checksum,
        policy_id=certificate.policy_id,
        structure_valid=jnp.asarray(
            claims_consistent
            & references_consistent
            & artifact_refs_consistent
            & evidence_residuals_consistent
        ),
        evidence_valid=jnp.asarray(
            policy_consistent & resolved_evidence_valid
        ),
        policy_report=recomputed,
    )
    return report


__all__: list[str] = [
    "certify_forward",
    "certify_forward_checked",
    "prepare_certification",
    "verify_certificate",
]
