"""Store aggregate carriers for certified forward executions.

Extended Summary
----------------
This module combines model identity, numerical evidence, policy
results, and execution identity into complete certificates.

Routine Listings
----------------
:class:`CertificationContext`
    Store prepared selections and references for compiled certification.
:class:`CertifiedResult`
    Store a numerical result paired with its differentiable certificate.
:class:`ExecutionManifest`
    Store software and execution identity prepared at the I/O boundary.
:class:`ForwardCertificate`
    Store the complete assurance record for one forward execution.
:func:`make_certification_context`
    Create a prepared certification context.
:func:`make_certified_result`
    Pair any JAX-compatible result value with a forward certificate.
:func:`make_execution_manifest`
    Create a validated execution manifest.
:func:`make_forward_certificate`
    Create and cross-validate a complete forward certificate.
"""

import equinox as eqx
from beartype import beartype
from beartype.typing import Any, Tuple
from jaxtyping import jaxtyped

from .certification_validation import _json_object, _require_text, _text_tuple
from .derivatives import (
    DependencyMap,
    DerivativeEvidence,
    InformationSpectrum,
    SensitivityMap,
)
from .evidence import (
    CertificationClaim,
    EvidenceRef,
    HumanAttestationRef,
    TransformationRecord,
)
from .reports import PolicyReport, WaiverRecord
from .specification import ArtifactRef, DomainResult, ForwardModelSpec


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

    See Also
    --------
    make_execution_manifest : Validated factory for this type.
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

    See Also
    --------
    make_certification_context : Validated factory for this type.
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

    See Also
    --------
    make_forward_certificate : Validated factory for this type.
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

    See Also
    --------
    make_certified_result : Validated factory for this type.
    """

    value: Any
    certificate: ForwardCertificate


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
def make_forward_certificate(  # noqa: PLR0913, PLR0917
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


__all__: list[str] = [
    "CertificationContext",
    "CertifiedResult",
    "ExecutionManifest",
    "ForwardCertificate",
    "make_certification_context",
    "make_certified_result",
    "make_execution_manifest",
    "make_forward_certificate",
]
