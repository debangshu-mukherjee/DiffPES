"""Test the JAX-native certification carrier contract.

The tests cover public behavior, differentiability, validation, and stable
scientific identity in the supported certification regime.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest
from beartype.typing import Any

import diffpes.types
from diffpes.types import (
    CertifiedResult,
    ForwardCertificate,
    make_artifact_ref,
    make_certification_claim,
    make_certification_context,
    make_certified_result,
    make_convention_ref,
    make_dependency_map,
    make_derivative_evidence,
    make_domain_predicate,
    make_domain_result,
    make_evidence_lineage,
    make_evidence_ref,
    make_execution_manifest,
    make_forward_certificate,
    make_forward_model_spec,
    make_human_attestation_ref,
    make_information_spectrum,
    make_policy_report,
    make_sensitivity_map,
    make_transformation_record,
)
from tests._assertions import assert_rejects, assert_trees_close


def _certificate() -> ForwardCertificate:
    """PRIVATE: Build one complete, small certificate for carrier tests.

    Returns
    -------
    certificate : ForwardCertificate
        A fully populated certificate with one artifact, one
        transformation, one evidence record, one attestation, one
        claim, and one domain result. Every derivative, dependency,
        sensitivity, information, and policy record is present.

    Notes
    -----
    Builds each sub-record through its public factory with small
    fixed toy values: two differentiable paths, length-two arrays,
    and energies in eV. Assembles them with
    ``make_forward_certificate``, so every carrier test starts from
    one deterministic instance.
    """
    convention: Any
    predicate: Any
    model: Any
    manifest: Any
    artifact: Any
    transformation: Any
    evidence: Any
    attestation: Any
    claim: Any
    domain: Any
    derivatives: Any
    dependencies: Any
    sensitivities: Any
    information: Any
    policy: Any
    convention = make_convention_ref(
        "org.diffpes.convention.energy-fermi", "1", '{"unit":"eV"}'
    )
    predicate = make_domain_predicate(
        "org.diffpes.domain.energy", "closed_interval", "eV"
    )
    model = make_forward_model_spec(
        model_id="org.diffpes.model.toy",
        model_version="1.0.0",
        observable_id="org.diffpes.observable.scalar",
        implementation_ref="tests.toy",
        assumptions=("linear-response",),
        conventions=(convention,),
        domain=(predicate,),
        differentiable_paths=("parameters.x", "parameters.y"),
        nondifferentiable_paths=("configuration.mode",),
    )
    manifest = make_execution_manifest(
        execution_id="run-1",
        model_ref="org.diffpes.model.toy@1.0.0",
        schema_version="1.0",
        package_version="test",
        source_checksum="source-1",
        environment_checksum="environment-1",
        backend="cpu",
        precision_policy="float64",
        deterministic=True,
        started_at_utc="2026-07-21T00:00:00Z",
    )
    artifact = make_artifact_ref(
        artifact_id="input-1",
        media_type="application/x-diffpes-array",
        byte_checksum=None,
        content_checksum="content-1",
        semantic_checksum="semantic-1",
        locator=None,
        role="normalized-input",
    )
    transformation = make_transformation_record(
        transformation_id="org.diffpes.transform.toy",
        transformation_version="1",
        parent_ids=("input-1",),
        output_ids=("output-1",),
        preserves=("units",),
        introduces=("broadening",),
        destroys=("sharp-lines",),
        invalidates_claims=("absolute-resolution",),
        parameters_checksum="parameters-1",
    )
    evidence = make_evidence_ref(
        evidence_id="evidence-1",
        method_id="org.diffpes.method.reference",
        source_type="analytic",
        measured=jnp.array([1.0, 2.0]),
        reference=jnp.array([1.0, 2.0]),
        residual=jnp.zeros(2),
        tolerance=jnp.full(2, 1.0e-12),
        lineage=make_evidence_lineage(
            implementation_refs=("reference.impl",),
            generator_refs=("reference.generator",),
            artifact_refs=("input-1",),
            derivation_refs=("reference.derivation",),
            relationship_ids=("independent-derivation:reference.derivation",),
        ),
        human_attestation_refs=("attestation-1",),
    )
    attestation = make_human_attestation_ref(
        "attestation-1",
        "reviewer.test",
        ("evidence-1",),
        "Lineage reviewed.",
        "2026-07-24T00:00:00Z",
    )
    claim = make_certification_claim(
        claim_id="claim-1",
        subject_id="output-1",
        predicate_id="reference-agreement",
        evidence_ids=("evidence-1",),
        measured=jnp.array([1.0, 2.0]),
        reference=jnp.array([1.0, 2.0]),
        residual=jnp.zeros(2),
        tolerance=jnp.full(2, 1.0e-12),
        passed=True,
        margin=1.0e-12,
    )
    domain = make_domain_result(
        predicate_id=predicate.predicate_id,
        measured=0.5,
        reference=1.0,
        residual=-0.5,
        tolerance=0.0,
        margin=0.5,
        passed=True,
    )
    derivatives = make_derivative_evidence(
        input_paths=model.differentiable_paths,
        output_projection_ids=("scalar",),
        method="jvp-vjp-fd",
        scales=jnp.ones(2),
        jvp_probes=jnp.array([[1.0], [2.0]]),
        vjp_probes=jnp.eye(2),
        reference_derivatives=jnp.eye(2),
        derivative_residuals=jnp.zeros((2, 2)),
        singular_values=jnp.array([2.0, 1.0]),
        effective_rank=2,
        condition_estimate=2.0,
        finite=True,
        fd_correct=True,
    )
    dependencies = make_dependency_map(
        model_id=model.model_id,
        input_paths=model.differentiable_paths,
        output_paths=("value",),
        structural=jnp.array([[True, True]]),
        traced=jnp.array([[True, True]]),
    )
    sensitivities = make_sensitivity_map(
        input_paths=model.differentiable_paths,
        output_projection_ids=("scalar",),
        scales=jnp.ones(2),
        sensitivities=jnp.array([[2.0, 1.0]]),
        threshold=1.0e-12,
        active=jnp.array([[True, True]]),
    )
    information = make_information_spectrum(
        input_paths=model.differentiable_paths,
        singular_values=jnp.array([2.0, 1.0]),
        right_singular_vectors=jnp.eye(2),
        effective_rank=2,
        condition_estimate=2.0,
        threshold=1.0e-12,
    )
    policy = make_policy_report(
        policy_id="org.diffpes.policy.research.v1",
        level_ids=("identified", "validated"),
        required_claim_ids=("claim-1",),
        claim_passed=jnp.array([True]),
        claim_checked=jnp.array([True]),
        claim_in_domain=jnp.array([True]),
        achieved=jnp.array([True, True]),
    )
    return make_forward_certificate(
        manifest=manifest,
        model=model,
        artifacts=(artifact,),
        transformations=(transformation,),
        evidence=(evidence,),
        attestations=(attestation,),
        claims=(claim,),
        domains=(domain,),
        derivatives=derivatives,
        dependencies=dependencies,
        sensitivities=sensitivities,
        information=information,
        policy_report=policy,
        policy_id=policy.policy_id,
        certificate_checksum="certificate-1",
        extensions_json='{"future_field":1}',
    )


class TestArtifactref:
    """Verify :class:`~diffpes.types.ArtifactRef`.

    The cases cover the public carrier or factory contract in JAX PyTrees.

    :see: :class:`~diffpes.types.ArtifactRef`
    """

    def test_public_symbol_has_expected_kind(self) -> None:
        """Expose ``ArtifactRef`` through its canonical types package path.

        The case uses explicit inputs in the supported certification regime.
        It checks the public result or the documented failure state.

        Notes
        -----
        The test compares the result with explicit numerical or structural assertions.
        """
        symbol: object = getattr(diffpes.types, "ArtifactRef")
        assert isinstance(symbol, type)
        assert issubclass(symbol, eqx.Module)


class TestCertificationclaim:
    """Verify :class:`~diffpes.types.CertificationClaim`.

    The cases cover the public carrier or factory contract in JAX PyTrees.

    :see: :class:`~diffpes.types.CertificationClaim`
    """

    def test_public_symbol_has_expected_kind(self) -> None:
        """Expose ``CertificationClaim`` through its canonical types package path.

        The case uses explicit inputs in the supported certification regime.
        It checks the public result or the documented failure state.

        Notes
        -----
        The test compares the result with explicit numerical or structural assertions.
        """
        symbol: object = getattr(diffpes.types, "CertificationClaim")
        assert isinstance(symbol, type)
        assert issubclass(symbol, eqx.Module)


class TestCertificationcontext:
    """Verify :class:`~diffpes.types.CertificationContext`.

    The cases cover the public carrier or factory contract in JAX PyTrees.

    :see: :class:`~diffpes.types.CertificationContext`
    """

    def test_public_symbol_has_expected_kind(self) -> None:
        """Expose ``CertificationContext`` through its canonical types package path.

        The case uses explicit inputs in the supported certification regime.
        It checks the public result or the documented failure state.

        Notes
        -----
        The test compares the result with explicit numerical or structural assertions.
        """
        symbol: object = getattr(diffpes.types, "CertificationContext")
        assert isinstance(symbol, type)
        assert issubclass(symbol, eqx.Module)


class TestCheckfunction:
    """Verify :obj:`~diffpes.types.CheckFunction`.

    The cases cover the public carrier or factory contract in JAX PyTrees.

    :see: :obj:`~diffpes.types.CheckFunction`
    """

    def test_public_symbol_has_expected_kind(self) -> None:
        """Expose ``CheckFunction`` through its canonical types package path.

        The case uses explicit inputs in the supported certification regime.
        It checks the public result or the documented failure state.

        Notes
        -----
        The test compares the result with explicit numerical or structural assertions.
        """
        symbol: object = getattr(diffpes.types, "CheckFunction")
        assert symbol is not None


class TestCertifiedresult:
    """Verify :class:`~diffpes.types.CertifiedResult`.

    The cases cover the public carrier or factory contract in JAX PyTrees.

    :see: :class:`~diffpes.types.CertifiedResult`
    """

    def test_public_symbol_has_expected_kind(self) -> None:
        """Expose ``CertifiedResult`` through its canonical types package path.

        The case uses explicit inputs in the supported certification regime.
        It checks the public result or the documented failure state.

        Notes
        -----
        The test compares the result with explicit numerical or structural assertions.
        """
        symbol: object = getattr(diffpes.types, "CertifiedResult")
        assert isinstance(symbol, type)
        assert issubclass(symbol, eqx.Module)

    def test_complete_result_round_trips_through_filter_jit(self) -> None:
        """Preserve the complete nested PyTree through compiled execution.

        The case uses explicit inputs in the supported certification regime.
        It checks the public result or the documented failure state.

        Notes
        -----
        The test compares the result with explicit numerical or structural assertions.
        """
        result: Any
        compiled: Any
        result = make_certified_result(jnp.array([2.0, 3.0]), _certificate())
        compiled = eqx.filter_jit(lambda item: item)(result)
        assert isinstance(compiled, CertifiedResult)
        assert compiled.certificate.model == result.certificate.model
        assert_trees_close(
            jax.tree.leaves(compiled),
            jax.tree.leaves(result),
            rtol=0.0,
            atol=0.0,
        )


class TestConventionref:
    """Verify :class:`~diffpes.types.ConventionRef`.

    The cases cover the public carrier or factory contract in JAX PyTrees.

    :see: :class:`~diffpes.types.ConventionRef`
    """

    def test_public_symbol_has_expected_kind(self) -> None:
        """Expose ``ConventionRef`` through its canonical types package path.

        The case uses explicit inputs in the supported certification regime.
        It checks the public result or the documented failure state.

        Notes
        -----
        The test compares the result with explicit numerical or structural assertions.
        """
        symbol: object = getattr(diffpes.types, "ConventionRef")
        assert isinstance(symbol, type)
        assert issubclass(symbol, eqx.Module)


class TestDependencymap:
    """Verify :class:`~diffpes.types.DependencyMap`.

    The cases cover the public carrier or factory contract in JAX PyTrees.

    :see: :class:`~diffpes.types.DependencyMap`
    """

    def test_public_symbol_has_expected_kind(self) -> None:
        """Expose ``DependencyMap`` through its canonical types package path.

        The case uses explicit inputs in the supported certification regime.
        It checks the public result or the documented failure state.

        Notes
        -----
        The test compares the result with explicit numerical or structural assertions.
        """
        symbol: object = getattr(diffpes.types, "DependencyMap")
        assert isinstance(symbol, type)
        assert issubclass(symbol, eqx.Module)


class TestDerivativeevidence:
    """Verify :class:`~diffpes.types.DerivativeEvidence`.

    The cases cover the public carrier or factory contract in JAX PyTrees.

    :see: :class:`~diffpes.types.DerivativeEvidence`
    """

    def test_public_symbol_has_expected_kind(self) -> None:
        """Expose ``DerivativeEvidence`` through its canonical types package path.

        The case uses explicit inputs in the supported certification regime.
        It checks the public result or the documented failure state.

        Notes
        -----
        The test compares the result with explicit numerical or structural assertions.
        """
        symbol: object = getattr(diffpes.types, "DerivativeEvidence")
        assert isinstance(symbol, type)
        assert issubclass(symbol, eqx.Module)


class TestDomainpredicate:
    """Verify :class:`~diffpes.types.DomainPredicate`.

    The cases cover the public carrier or factory contract in JAX PyTrees.

    :see: :class:`~diffpes.types.DomainPredicate`
    """

    def test_public_symbol_has_expected_kind(self) -> None:
        """Expose ``DomainPredicate`` through its canonical types package path.

        The case uses explicit inputs in the supported certification regime.
        It checks the public result or the documented failure state.

        Notes
        -----
        The test compares the result with explicit numerical or structural assertions.
        """
        symbol: object = getattr(diffpes.types, "DomainPredicate")
        assert isinstance(symbol, type)
        assert issubclass(symbol, eqx.Module)


class TestDomainresult:
    """Verify :class:`~diffpes.types.DomainResult`.

    The cases cover the public carrier or factory contract in JAX PyTrees.

    :see: :class:`~diffpes.types.DomainResult`
    """

    def test_public_symbol_has_expected_kind(self) -> None:
        """Expose ``DomainResult`` through its canonical types package path.

        The case uses explicit inputs in the supported certification regime.
        It checks the public result or the documented failure state.

        Notes
        -----
        The test compares the result with explicit numerical or structural assertions.
        """
        symbol: object = getattr(diffpes.types, "DomainResult")
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
        symbol: object = getattr(diffpes.types, "EvidenceLineage")
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
        symbol: object = getattr(diffpes.types, "HumanAttestationRef")
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
        The test compares the result with explicit numerical or structural assertions.
        """
        symbol: object = getattr(diffpes.types, "EvidenceRef")
        assert isinstance(symbol, type)
        assert issubclass(symbol, eqx.Module)


class TestEvidencereport:
    """Verify :class:`~diffpes.types.EvidenceReport`.

    The cases cover the public carrier or factory contract in JAX PyTrees.

    :see: :class:`~diffpes.types.EvidenceReport`
    """

    def test_public_symbol_has_expected_kind(self) -> None:
        """Expose ``EvidenceReport`` through its canonical types package path.

        The case uses explicit inputs in the supported certification regime.
        It checks the public result or the documented failure state.

        Notes
        -----
        The test compares the result with explicit numerical or structural assertions.
        """
        symbol: object = getattr(diffpes.types, "EvidenceReport")
        assert isinstance(symbol, type)
        assert issubclass(symbol, eqx.Module)


class TestExecutionmanifest:
    """Verify :class:`~diffpes.types.ExecutionManifest`.

    The cases cover the public carrier or factory contract in JAX PyTrees.

    :see: :class:`~diffpes.types.ExecutionManifest`
    """

    def test_public_symbol_has_expected_kind(self) -> None:
        """Expose ``ExecutionManifest`` through its canonical types package path.

        The case uses explicit inputs in the supported certification regime.
        It checks the public result or the documented failure state.

        Notes
        -----
        The test compares the result with explicit numerical or structural assertions.
        """
        symbol: object = getattr(diffpes.types, "ExecutionManifest")
        assert isinstance(symbol, type)
        assert issubclass(symbol, eqx.Module)


class TestForwardcertificate:
    """Verify :class:`~diffpes.types.ForwardCertificate`.

    The cases cover the public carrier or factory contract in JAX PyTrees.

    :see: :class:`~diffpes.types.ForwardCertificate`
    """

    def test_public_symbol_has_expected_kind(self) -> None:
        """Expose ``ForwardCertificate`` through its canonical types package path.

        The case uses explicit inputs in the supported certification regime.
        It checks the public result or the documented failure state.

        Notes
        -----
        The test compares the result with explicit numerical or structural assertions.
        """
        symbol: object = getattr(diffpes.types, "ForwardCertificate")
        assert isinstance(symbol, type)
        assert issubclass(symbol, eqx.Module)

    def test_complete_graph_has_traced_numerical_leaves(self) -> None:
        """Verify every dynamic leaf in a complete certificate is a JAX array.

        The case uses explicit inputs in the supported certification regime.
        It checks the public result or the documented failure state.

        Notes
        -----
        The test constructs the shared complete fixture and inspects its flattened JAX
        leaves while static vocabulary remains outside the numerical tree.
        """
        certificate: ForwardCertificate = _certificate()
        leaves: list[jax.Array] = jax.tree.leaves(certificate)
        assert leaves
        assert all(isinstance(leaf, jax.Array) for leaf in leaves)

    def test_static_vocabulary_is_absent_from_numerical_leaves(self) -> None:
        """Keep identifiers static while retaining all numerical evidence.

        The case uses explicit inputs in the supported certification regime.
        It checks the public result or the documented failure state.

        Notes
        -----
        The test compares the result with explicit numerical or structural assertions.
        """
        certificate: Any
        leaves: Any
        certificate = _certificate()
        leaves = jax.tree.leaves(certificate)
        assert leaves
        assert all(isinstance(leaf, jax.Array) for leaf in leaves)
        assert not any(isinstance(leaf, str) for leaf in leaves)
        assert certificate.model.model_id == "org.diffpes.model.toy"
        assert certificate.extensions_json == '{"future_field":1}'


class TestForwardmodelspec:
    """Verify :class:`~diffpes.types.ForwardModelSpec`.

    The cases cover the public carrier or factory contract in JAX PyTrees.

    :see: :class:`~diffpes.types.ForwardModelSpec`
    """

    def test_public_symbol_has_expected_kind(self) -> None:
        """Expose ``ForwardModelSpec`` through its canonical types package path.

        The case uses explicit inputs in the supported certification regime.
        It checks the public result or the documented failure state.

        Notes
        -----
        The test compares the result with explicit numerical or structural assertions.
        """
        symbol: object = getattr(diffpes.types, "ForwardModelSpec")
        assert isinstance(symbol, type)
        assert issubclass(symbol, eqx.Module)


class TestInformationspectrum:
    """Verify :class:`~diffpes.types.InformationSpectrum`.

    The cases cover the public carrier or factory contract in JAX PyTrees.

    :see: :class:`~diffpes.types.InformationSpectrum`
    """

    def test_public_symbol_has_expected_kind(self) -> None:
        """Expose ``InformationSpectrum`` through its canonical types package path.

        The case uses explicit inputs in the supported certification regime.
        It checks the public result or the documented failure state.

        Notes
        -----
        The test compares the result with explicit numerical or structural assertions.
        """
        symbol: object = getattr(diffpes.types, "InformationSpectrum")
        assert isinstance(symbol, type)
        assert issubclass(symbol, eqx.Module)


class TestPolicyreport:
    """Verify :class:`~diffpes.types.PolicyReport`.

    The cases cover the public carrier or factory contract in JAX PyTrees.

    :see: :class:`~diffpes.types.PolicyReport`
    """

    def test_public_symbol_has_expected_kind(self) -> None:
        """Expose ``PolicyReport`` through its canonical types package path.

        The case uses explicit inputs in the supported certification regime.
        It checks the public result or the documented failure state.

        Notes
        -----
        The test compares the result with explicit numerical or structural assertions.
        """
        symbol: object = getattr(diffpes.types, "PolicyReport")
        assert isinstance(symbol, type)
        assert issubclass(symbol, eqx.Module)


class TestRegisteredmodel:
    """Verify :class:`~diffpes.types.RegisteredModel`.

    The cases cover the public carrier or factory contract in JAX PyTrees.

    :see: :class:`~diffpes.types.RegisteredModel`
    """

    def test_public_symbol_has_expected_kind(self) -> None:
        """Expose ``RegisteredModel`` through its canonical types package path.

        The case uses explicit inputs in the supported certification regime.
        It checks the public result or the documented failure state.

        Notes
        -----
        The test compares the result with explicit numerical or structural assertions.
        """
        symbol: object = getattr(diffpes.types, "RegisteredModel")
        assert isinstance(symbol, type)
        assert issubclass(symbol, eqx.Module)


class TestRegisteredtransformation:
    """Verify :class:`~diffpes.types.RegisteredTransformation`.

    The cases cover the public carrier or factory contract in JAX PyTrees.

    :see: :class:`~diffpes.types.RegisteredTransformation`
    """

    def test_public_symbol_has_expected_kind(self) -> None:
        """Expose ``RegisteredTransformation`` through its canonical types package path.

        The case uses explicit inputs in the supported certification regime.
        It checks the public result or the documented failure state.

        Notes
        -----
        The test compares the result with explicit numerical or structural assertions.
        """
        symbol: object = getattr(diffpes.types, "RegisteredTransformation")
        assert isinstance(symbol, type)
        assert issubclass(symbol, eqx.Module)


class TestRegistryreport:
    """Verify :class:`~diffpes.types.RegistryReport`.

    The cases cover the public carrier or factory contract in JAX PyTrees.

    :see: :class:`~diffpes.types.RegistryReport`
    """

    def test_public_symbol_has_expected_kind(self) -> None:
        """Expose ``RegistryReport`` through its canonical types package path.

        The case uses explicit inputs in the supported certification regime.
        It checks the public result or the documented failure state.

        Notes
        -----
        The test compares the result with explicit numerical or structural assertions.
        """
        symbol: object = getattr(diffpes.types, "RegistryReport")
        assert isinstance(symbol, type)
        assert issubclass(symbol, eqx.Module)


class TestRegistrysnapshot:
    """Verify :class:`~diffpes.types.RegistrySnapshot`.

    The cases cover the public carrier or factory contract in JAX PyTrees.

    :see: :class:`~diffpes.types.RegistrySnapshot`
    """

    def test_public_symbol_has_expected_kind(self) -> None:
        """Expose ``RegistrySnapshot`` through its canonical types package path.

        The case uses explicit inputs in the supported certification regime.
        It checks the public result or the documented failure state.

        Notes
        -----
        The test compares the result with explicit numerical or structural assertions.
        """
        symbol: object = getattr(diffpes.types, "RegistrySnapshot")
        assert isinstance(symbol, type)
        assert issubclass(symbol, eqx.Module)


class TestReproductionreport:
    """Verify :class:`~diffpes.types.ReproductionReport`.

    The cases cover the public carrier or factory contract in JAX PyTrees.

    :see: :class:`~diffpes.types.ReproductionReport`
    """

    def test_public_symbol_has_expected_kind(self) -> None:
        """Expose ``ReproductionReport`` through its canonical types package path.

        The case uses explicit inputs in the supported certification regime.
        It checks the public result or the documented failure state.

        Notes
        -----
        The test compares the result with explicit numerical or structural assertions.
        """
        symbol: object = getattr(diffpes.types, "ReproductionReport")
        assert isinstance(symbol, type)
        assert issubclass(symbol, eqx.Module)


class TestSensitivitymap:
    """Verify :class:`~diffpes.types.SensitivityMap`.

    The cases cover the public carrier or factory contract in JAX PyTrees.

    :see: :class:`~diffpes.types.SensitivityMap`
    """

    def test_public_symbol_has_expected_kind(self) -> None:
        """Expose ``SensitivityMap`` through its canonical types package path.

        The case uses explicit inputs in the supported certification regime.
        It checks the public result or the documented failure state.

        Notes
        -----
        The test compares the result with explicit numerical or structural assertions.
        """
        symbol: object = getattr(diffpes.types, "SensitivityMap")
        assert isinstance(symbol, type)
        assert issubclass(symbol, eqx.Module)


class TestTransformationrecord:
    """Verify :class:`~diffpes.types.TransformationRecord`.

    The cases cover the public carrier or factory contract in JAX PyTrees.

    :see: :class:`~diffpes.types.TransformationRecord`
    """

    def test_public_symbol_has_expected_kind(self) -> None:
        """Expose ``TransformationRecord`` through its canonical types package path.

        The case uses explicit inputs in the supported certification regime.
        It checks the public result or the documented failure state.

        Notes
        -----
        The test compares the result with explicit numerical or structural assertions.
        """
        symbol: object = getattr(diffpes.types, "TransformationRecord")
        assert isinstance(symbol, type)
        assert issubclass(symbol, eqx.Module)


class TestVerificationreport:
    """Verify :class:`~diffpes.types.VerificationReport`.

    The cases cover the public carrier or factory contract in JAX PyTrees.

    :see: :class:`~diffpes.types.VerificationReport`
    """

    def test_public_symbol_has_expected_kind(self) -> None:
        """Expose ``VerificationReport`` through its canonical types package path.

        The case uses explicit inputs in the supported certification regime.
        It checks the public result or the documented failure state.

        Notes
        -----
        The test compares the result with explicit numerical or structural assertions.
        """
        symbol: object = getattr(diffpes.types, "VerificationReport")
        assert isinstance(symbol, type)
        assert issubclass(symbol, eqx.Module)


class TestMakeArtifactRef:
    """Verify :func:`~diffpes.types.make_artifact_ref`.

    The cases cover the public carrier or factory contract in JAX PyTrees.

    :see: :func:`~diffpes.types.make_artifact_ref`
    """

    def test_public_symbol_has_expected_kind(self) -> None:
        """Expose ``make_artifact_ref`` through its canonical types package path.

        The case uses explicit inputs in the supported certification regime.
        It checks the public result or the documented failure state.

        Notes
        -----
        The test compares the result with explicit numerical or structural assertions.
        """
        symbol: object = getattr(diffpes.types, "make_artifact_ref")
        assert callable(symbol)


class TestMakeCertificationClaim:
    """Verify :func:`~diffpes.types.make_certification_claim`.

    The cases cover the public carrier or factory contract in JAX PyTrees.

    :see: :func:`~diffpes.types.make_certification_claim`
    """

    def test_public_symbol_has_expected_kind(self) -> None:
        """Expose ``make_certification_claim`` through its canonical types package path.

        The case uses explicit inputs in the supported certification regime.
        It checks the public result or the documented failure state.

        Notes
        -----
        The test compares the result with explicit numerical or structural assertions.
        """
        symbol: object = getattr(diffpes.types, "make_certification_claim")
        assert callable(symbol)

    def test_continuous_claim_evidence_is_differentiable(self) -> None:
        """Differentiate a smooth residual and margin through its factory.

        The case uses explicit inputs in the supported certification regime.
        It checks the public result or the documented failure state.

        Notes
        -----
        The test compares the result with explicit numerical or structural assertions.
        """
        value: Any
        expected: Any

        def objective(value: jax.Array) -> jax.Array:
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
            return jnp.sum(claim.residual**2) + claim.margin

        value = jnp.asarray(1.25)
        expected = 2.0 * (value - 1.0) - 1.0
        assert_trees_close(jax.grad(objective)(value), expected)
        assert_trees_close(
            eqx.filter_jit(jax.grad(objective))(value), expected
        )


class TestMakeCertificationContext:
    """Verify :func:`~diffpes.types.make_certification_context`.

    The cases cover the public carrier or factory contract in JAX PyTrees.

    :see: :func:`~diffpes.types.make_certification_context`
    """

    def test_public_symbol_has_expected_kind(self) -> None:
        """Expose ``make_certification_context`` through its canonical types package path.

        The case uses explicit inputs in the supported certification regime.
        It checks the public result or the documented failure state.

        Notes
        -----
        The test compares the result with explicit numerical or structural assertions.
        """
        symbol: object = getattr(diffpes.types, "make_certification_context")
        assert callable(symbol)

    def test_context_cross_validates_model_identity(self) -> None:
        """Reject prepared contexts that combine different model identities.

        The case uses explicit inputs in the supported certification regime.
        It checks the public result or the documented failure state.

        Notes
        -----
        The test compares the result with explicit numerical or structural assertions.
        """
        certificate: Any
        context: Any
        bad_manifest: Any
        certificate = _certificate()
        context = make_certification_context(
            certificate.manifest,
            certificate.model,
            certificate.artifacts,
            certificate.transformations,
            certificate.evidence,
            certificate.policy_id,
            ("domain", "output"),
            ("semantic-1",),
        )
        assert context.model is certificate.model
        bad_manifest = make_execution_manifest(
            execution_id=certificate.manifest.execution_id,
            model_ref="org.diffpes.model.other",
            schema_version=certificate.manifest.schema_version,
            package_version=certificate.manifest.package_version,
            source_checksum=certificate.manifest.source_checksum,
            environment_checksum=certificate.manifest.environment_checksum,
            backend=certificate.manifest.backend,
            precision_policy=certificate.manifest.precision_policy,
            deterministic=certificate.manifest.deterministic,
            started_at_utc=certificate.manifest.started_at_utc,
        )
        with pytest.raises(ValueError, match="model_ref does not match"):
            make_certification_context(bad_manifest, certificate.model)


class TestMakeCertifiedResult:
    """Verify :func:`~diffpes.types.make_certified_result`.

    The cases cover the public carrier or factory contract in JAX PyTrees.

    :see: :func:`~diffpes.types.make_certified_result`
    """

    def test_public_symbol_has_expected_kind(self) -> None:
        """Expose ``make_certified_result`` through its canonical types package path.

        The case uses explicit inputs in the supported certification regime.
        It checks the public result or the documented failure state.

        Notes
        -----
        The test compares the result with explicit numerical or structural assertions.
        """
        symbol: object = getattr(diffpes.types, "make_certified_result")
        assert callable(symbol)

    def test_certified_envelope_preserves_primal_jvp_and_vjp(self) -> None:
        """Show that attaching a certificate does not alter model derivatives.

        The case uses explicit inputs in the supported certification regime.
        It checks the public result or the documented failure state.

        Notes
        -----
        The test compares the result with explicit numerical or structural assertions.
        """
        certificate: Any
        point: Any
        tangent: Any
        ordinary_primal: Any
        ordinary_jvp: Any
        certified_primal: Any
        certified_jvp: Any
        _: Any
        ordinary_pullback: Any
        certified_pullback: Any
        cotangent: Any
        certificate = _certificate()

        def ordinary(value: jax.Array) -> jax.Array:
            return jnp.sin(value) + value**2

        def certified(value: jax.Array) -> jax.Array:
            return make_certified_result(ordinary(value), certificate).value

        point = jnp.asarray(0.4)
        tangent = jnp.asarray(1.7)
        ordinary_primal, ordinary_jvp = jax.jvp(ordinary, (point,), (tangent,))
        certified_primal, certified_jvp = jax.jvp(
            certified, (point,), (tangent,)
        )
        assert_trees_close(
            certified_primal, ordinary_primal, rtol=0.0, atol=0.0
        )
        assert_trees_close(certified_jvp, ordinary_jvp, rtol=0.0, atol=0.0)
        _, ordinary_pullback = jax.vjp(ordinary, point)
        _, certified_pullback = jax.vjp(certified, point)
        cotangent = jnp.asarray(2.0)
        assert_trees_close(
            certified_pullback(cotangent),
            ordinary_pullback(cotangent),
            rtol=0.0,
            atol=0.0,
        )


class TestMakeConventionRef:
    """Verify :func:`~diffpes.types.make_convention_ref`.

    The cases cover the public carrier or factory contract in JAX PyTrees.

    :see: :func:`~diffpes.types.make_convention_ref`
    """

    def test_public_symbol_has_expected_kind(self) -> None:
        """Expose ``make_convention_ref`` through its canonical types package path.

        The case uses explicit inputs in the supported certification regime.
        It checks the public result or the documented failure state.

        Notes
        -----
        The test compares the result with explicit numerical or structural assertions.
        """
        symbol: object = getattr(diffpes.types, "make_convention_ref")
        assert callable(symbol)


class TestMakeDependencyMap:
    """Verify :func:`~diffpes.types.make_dependency_map`.

    The cases cover the public carrier or factory contract in JAX PyTrees.

    :see: :func:`~diffpes.types.make_dependency_map`
    """

    def test_public_symbol_has_expected_kind(self) -> None:
        """Expose ``make_dependency_map`` through its canonical types package path.

        The case uses explicit inputs in the supported certification regime.
        It checks the public result or the documented failure state.

        Notes
        -----
        The test compares the result with explicit numerical or structural assertions.
        """
        symbol: object = getattr(diffpes.types, "make_dependency_map")
        assert callable(symbol)


class TestMakeDerivativeEvidence:
    """Verify :func:`~diffpes.types.make_derivative_evidence`.

    The cases cover the public carrier or factory contract in JAX PyTrees.

    :see: :func:`~diffpes.types.make_derivative_evidence`
    """

    def test_public_symbol_has_expected_kind(self) -> None:
        """Expose ``make_derivative_evidence`` through its canonical types package path.

        The case uses explicit inputs in the supported certification regime.
        It checks the public result or the documented failure state.

        Notes
        -----
        The test compares the result with explicit numerical or structural assertions.
        """
        symbol: object = getattr(diffpes.types, "make_derivative_evidence")
        assert callable(symbol)


class TestMakeDomainPredicate:
    """Verify :func:`~diffpes.types.make_domain_predicate`.

    The cases cover the public carrier or factory contract in JAX PyTrees.

    :see: :func:`~diffpes.types.make_domain_predicate`
    """

    def test_public_symbol_has_expected_kind(self) -> None:
        """Expose ``make_domain_predicate`` through its canonical types package path.

        The case uses explicit inputs in the supported certification regime.
        It checks the public result or the documented failure state.

        Notes
        -----
        The test compares the result with explicit numerical or structural assertions.
        """
        symbol: object = getattr(diffpes.types, "make_domain_predicate")
        assert callable(symbol)


class TestMakeDomainResult:
    """Verify :func:`~diffpes.types.make_domain_result`.

    The cases cover the public carrier or factory contract in JAX PyTrees.

    :see: :func:`~diffpes.types.make_domain_result`
    """

    def test_public_symbol_has_expected_kind(self) -> None:
        """Expose ``make_domain_result`` through its canonical types package path.

        The case uses explicit inputs in the supported certification regime.
        It checks the public result or the documented failure state.

        Notes
        -----
        The test compares the result with explicit numerical or structural assertions.
        """
        symbol: object = getattr(diffpes.types, "make_domain_result")
        assert callable(symbol)

    def test_domain_results_vmap_with_traced_status_and_margin(self) -> None:
        """Batch domain checks without concretizing status arrays.

        The case uses explicit inputs in the supported certification regime.
        It checks the public result or the documented failure state.

        Notes
        -----
        The test compares the result with explicit numerical or structural assertions.
        """
        result: Any

        def evaluate(value: jax.Array) -> Any:
            margin: Any
            margin = 1.0 - jnp.abs(value)
            return make_domain_result(
                "bounded",
                measured=value,
                reference=0.0,
                residual=value,
                tolerance=0.0,
                margin=margin,
                passed=margin >= 0.0,
                in_domain=margin >= 0.0,
            )

        result = eqx.filter_jit(jax.vmap(evaluate))(jnp.array([-0.5, 1.5]))
        assert_trees_close(result.margin, jnp.array([0.5, -0.5]))
        assert jnp.array_equal(result.passed, jnp.array([True, False]))


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
        """Expose ``make_evidence_ref`` through its canonical types package path.

        The case uses explicit inputs in the supported certification regime.
        It checks the public result or the documented failure state.

        Notes
        -----
        The test compares the result with explicit numerical or structural assertions.
        """
        symbol: object = getattr(diffpes.types, "make_evidence_ref")
        assert callable(symbol)

    def test_factories_reject_bad_numerical_shapes_and_tolerances(
        self,
    ) -> None:
        """Reject malformed evidence eagerly and through compiled execution.

        The case uses explicit inputs in the supported certification regime.
        It checks the public result or the documented failure state.

        Notes
        -----
        The test compares the result with explicit numerical or structural assertions.
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


class TestMakeEvidenceReport:
    """Verify :func:`~diffpes.types.make_evidence_report`.

    The cases cover the public carrier or factory contract in JAX PyTrees.

    :see: :func:`~diffpes.types.make_evidence_report`
    """

    def test_public_symbol_has_expected_kind(self) -> None:
        """Expose ``make_evidence_report`` through its canonical types package path.

        The case uses explicit inputs in the supported certification regime.
        It checks the public result or the documented failure state.

        Notes
        -----
        The test compares the result with explicit numerical or structural assertions.
        """
        symbol: object = getattr(diffpes.types, "make_evidence_report")
        assert callable(symbol)


class TestMakeExecutionManifest:
    """Verify :func:`~diffpes.types.make_execution_manifest`.

    The cases cover the public carrier or factory contract in JAX PyTrees.

    :see: :func:`~diffpes.types.make_execution_manifest`
    """

    def test_public_symbol_has_expected_kind(self) -> None:
        """Expose ``make_execution_manifest`` through its canonical types package path.

        The case uses explicit inputs in the supported certification regime.
        It checks the public result or the documented failure state.

        Notes
        -----
        The test compares the result with explicit numerical or structural assertions.
        """
        symbol: object = getattr(diffpes.types, "make_execution_manifest")
        assert callable(symbol)


class TestMakeForwardCertificate:
    """Verify :func:`~diffpes.types.make_forward_certificate`.

    The cases cover the public carrier or factory contract in JAX PyTrees.

    :see: :func:`~diffpes.types.make_forward_certificate`
    """

    def test_public_symbol_has_expected_kind(self) -> None:
        """Expose ``make_forward_certificate`` through its canonical types package path.

        The case uses explicit inputs in the supported certification regime.
        It checks the public result or the documented failure state.

        Notes
        -----
        The test compares the result with explicit numerical or structural assertions.
        """
        symbol: object = getattr(diffpes.types, "make_forward_certificate")
        assert callable(symbol)

    def test_certificate_rejects_cross_record_inconsistency(self) -> None:
        """Reject policy, dependency, and duplicate identity mismatches.

        The case uses explicit inputs in the supported certification regime.
        It checks the public result or the documented failure state.

        Notes
        -----
        The test compares the result with explicit numerical or structural assertions.
        """
        certificate: Any
        certificate = _certificate()
        with pytest.raises(ValueError, match="policy_report policy_id"):
            make_forward_certificate(
                certificate.manifest,
                certificate.model,
                certificate.artifacts,
                certificate.transformations,
                certificate.evidence,
                certificate.claims,
                certificate.domains,
                certificate.derivatives,
                certificate.dependencies,
                certificate.sensitivities,
                certificate.information,
                certificate.policy_report,
                "org.diffpes.policy.other.v1",
                certificate.certificate_checksum,
                attestations=certificate.attestations,
            )
        with pytest.raises(ValueError, match="duplicate artifact_id"):
            make_forward_certificate(
                certificate.manifest,
                certificate.model,
                certificate.artifacts * 2,
                certificate.transformations,
                certificate.evidence,
                certificate.claims,
                certificate.domains,
                certificate.derivatives,
                certificate.dependencies,
                certificate.sensitivities,
                certificate.information,
                certificate.policy_report,
                certificate.policy_id,
                certificate.certificate_checksum,
                attestations=certificate.attestations,
            )


class TestMakeForwardModelSpec:
    """Verify :func:`~diffpes.types.make_forward_model_spec`.

    The cases cover the public carrier or factory contract in JAX PyTrees.

    :see: :func:`~diffpes.types.make_forward_model_spec`
    """

    def test_public_symbol_has_expected_kind(self) -> None:
        """Expose ``make_forward_model_spec`` through its canonical types package path.

        The case uses explicit inputs in the supported certification regime.
        It checks the public result or the documented failure state.

        Notes
        -----
        The test compares the result with explicit numerical or structural assertions.
        """
        symbol: object = getattr(diffpes.types, "make_forward_model_spec")
        assert callable(symbol)

    def test_factories_reject_malformed_static_structure(self) -> None:
        """Reject empty identities, invalid JSON, and overlapping path classes.

        The case uses explicit inputs in the supported certification regime.
        It checks the public result or the documented failure state.

        Notes
        -----
        The test compares the result with explicit numerical or structural assertions.
        """
        with pytest.raises(ValueError, match="artifact_id must be non-empty"):
            make_artifact_ref("", "text/plain", None, "a", "b", None, "input")
        with pytest.raises(ValueError, match="valid JSON"):
            make_convention_ref("convention", "1", "{")
        with pytest.raises(ValueError, match="must be disjoint"):
            make_forward_model_spec(
                "model",
                "1",
                "observable",
                "implementation",
                (),
                (),
                (),
                ("x",),
                ("x",),
            )


class TestMakeInformationSpectrum:
    """Verify :func:`~diffpes.types.make_information_spectrum`.

    The cases cover the public carrier or factory contract in JAX PyTrees.

    :see: :func:`~diffpes.types.make_information_spectrum`
    """

    def test_public_symbol_has_expected_kind(self) -> None:
        """Expose ``make_information_spectrum`` through its canonical types package path.

        The case uses explicit inputs in the supported certification regime.
        It checks the public result or the documented failure state.

        Notes
        -----
        The test compares the result with explicit numerical or structural assertions.
        """
        symbol: object = getattr(diffpes.types, "make_information_spectrum")
        assert callable(symbol)


class TestMakePolicyReport:
    """Verify :func:`~diffpes.types.make_policy_report`.

    The cases cover the public carrier or factory contract in JAX PyTrees.

    :see: :func:`~diffpes.types.make_policy_report`
    """

    def test_public_symbol_has_expected_kind(self) -> None:
        """Expose ``make_policy_report`` through its canonical types package path.

        The case uses explicit inputs in the supported certification regime.
        It checks the public result or the documented failure state.

        Notes
        -----
        The test compares the result with explicit numerical or structural assertions.
        """
        symbol: object = getattr(diffpes.types, "make_policy_report")
        assert callable(symbol)


class TestMakeRegisteredModel:
    """Verify :func:`~diffpes.types.make_registered_model`.

    The cases cover the public carrier or factory contract in JAX PyTrees.

    :see: :func:`~diffpes.types.make_registered_model`
    """

    def test_public_symbol_has_expected_kind(self) -> None:
        """Expose ``make_registered_model`` through its canonical types package path.

        The case uses explicit inputs in the supported certification regime.
        It checks the public result or the documented failure state.

        Notes
        -----
        The test compares the result with explicit numerical or structural assertions.
        """
        symbol: object = getattr(diffpes.types, "make_registered_model")
        assert callable(symbol)


class TestMakeRegisteredTransformation:
    """Verify :func:`~diffpes.types.make_registered_transformation`.

    The cases cover the public carrier or factory contract in JAX PyTrees.

    :see: :func:`~diffpes.types.make_registered_transformation`
    """

    def test_public_symbol_has_expected_kind(self) -> None:
        """Expose ``make_registered_transformation`` through its canonical types package path.

        The case uses explicit inputs in the supported certification regime.
        It checks the public result or the documented failure state.

        Notes
        -----
        The test compares the result with explicit numerical or structural assertions.
        """
        symbol: object = getattr(
            diffpes.types, "make_registered_transformation"
        )
        assert callable(symbol)


class TestMakeRegistryReport:
    """Verify :func:`~diffpes.types.make_registry_report`.

    The cases cover the public carrier or factory contract in JAX PyTrees.

    :see: :func:`~diffpes.types.make_registry_report`
    """

    def test_public_symbol_has_expected_kind(self) -> None:
        """Expose ``make_registry_report`` through its canonical types package path.

        The case uses explicit inputs in the supported certification regime.
        It checks the public result or the documented failure state.

        Notes
        -----
        The test compares the result with explicit numerical or structural assertions.
        """
        symbol: object = getattr(diffpes.types, "make_registry_report")
        assert callable(symbol)


class TestMakeRegistrySnapshot:
    """Verify :func:`~diffpes.types.make_registry_snapshot`.

    The cases cover the public carrier or factory contract in JAX PyTrees.

    :see: :func:`~diffpes.types.make_registry_snapshot`
    """

    def test_public_symbol_has_expected_kind(self) -> None:
        """Expose ``make_registry_snapshot`` through its canonical types package path.

        The case uses explicit inputs in the supported certification regime.
        It checks the public result or the documented failure state.

        Notes
        -----
        The test compares the result with explicit numerical or structural assertions.
        """
        symbol: object = getattr(diffpes.types, "make_registry_snapshot")
        assert callable(symbol)


class TestMakeReproductionReport:
    """Verify :func:`~diffpes.types.make_reproduction_report`.

    The cases cover the public carrier or factory contract in JAX PyTrees.

    :see: :func:`~diffpes.types.make_reproduction_report`
    """

    def test_public_symbol_has_expected_kind(self) -> None:
        """Expose ``make_reproduction_report`` through its canonical types package path.

        The case uses explicit inputs in the supported certification regime.
        It checks the public result or the documented failure state.

        Notes
        -----
        The test compares the result with explicit numerical or structural assertions.
        """
        symbol: object = getattr(diffpes.types, "make_reproduction_report")
        assert callable(symbol)


class TestMakeSensitivityMap:
    """Verify :func:`~diffpes.types.make_sensitivity_map`.

    The cases cover the public carrier or factory contract in JAX PyTrees.

    :see: :func:`~diffpes.types.make_sensitivity_map`
    """

    def test_public_symbol_has_expected_kind(self) -> None:
        """Expose ``make_sensitivity_map`` through its canonical types package path.

        The case uses explicit inputs in the supported certification regime.
        It checks the public result or the documented failure state.

        Notes
        -----
        The test compares the result with explicit numerical or structural assertions.
        """
        symbol: object = getattr(diffpes.types, "make_sensitivity_map")
        assert callable(symbol)


class TestMakeTransformationRecord:
    """Verify :func:`~diffpes.types.make_transformation_record`.

    The cases cover the public carrier or factory contract in JAX PyTrees.

    :see: :func:`~diffpes.types.make_transformation_record`
    """

    def test_public_symbol_has_expected_kind(self) -> None:
        """Expose ``make_transformation_record`` through its canonical types package path.

        The case uses explicit inputs in the supported certification regime.
        It checks the public result or the documented failure state.

        Notes
        -----
        The test compares the result with explicit numerical or structural assertions.
        """
        symbol: object = getattr(diffpes.types, "make_transformation_record")
        assert callable(symbol)


class TestMakeVerificationReport:
    """Verify :func:`~diffpes.types.make_verification_report`.

    The cases cover the public carrier or factory contract in JAX PyTrees.

    :see: :func:`~diffpes.types.make_verification_report`
    """

    def test_public_symbol_has_expected_kind(self) -> None:
        """Expose ``make_verification_report`` through its canonical types package path.

        The case uses explicit inputs in the supported certification regime.
        It checks the public result or the documented failure state.

        Notes
        -----
        The test compares the result with explicit numerical or structural assertions.
        """
        symbol: object = getattr(diffpes.types, "make_verification_report")
        assert callable(symbol)


class TestArtifactResolver:
    """Verify :obj:`~diffpes.types.ArtifactResolver`.

    The case checks the canonical types package path for the resolver alias.

    :see: :obj:`~diffpes.types.ArtifactResolver`
    """

    def test_public_alias_is_available(self) -> None:
        """Expose the artifact resolver alias from the types package.

        The public package must provide one canonical alias object.

        Notes
        -----
        The test resolves the alias by its exact public name.
        """
        symbol: object = getattr(diffpes.types, "ArtifactResolver")
        assert symbol is not None


class TestRegistrationHandshake:
    """Verify :class:`~diffpes.types.RegistrationHandshake`.

    The case checks the canonical type identity for registration requirements.

    :see: :class:`~diffpes.types.RegistrationHandshake`
    """

    def test_public_type_is_an_equinox_module(self) -> None:
        """Expose the handshake type as an Equinox module class.

        The carrier must use the project PyTree base class.

        Notes
        -----
        The test checks class inheritance through the canonical package path.
        """
        symbol: Any = getattr(diffpes.types, "RegistrationHandshake")
        assert issubclass(symbol, eqx.Module)


class TestHandshakeReport:
    """Verify :class:`~diffpes.types.HandshakeReport`.

    The case checks the JAX-native report carrier for handshake validation.

    :see: :class:`~diffpes.types.HandshakeReport`
    """

    def test_public_type_is_an_equinox_module(self) -> None:
        """Expose the handshake report as an Equinox module class.

        The carrier must use the project PyTree base class.

        Notes
        -----
        The test checks class inheritance through the canonical package path.
        """
        symbol: Any = getattr(diffpes.types, "HandshakeReport")
        assert issubclass(symbol, eqx.Module)


class TestWaiverRecord:
    """Verify :class:`~diffpes.types.WaiverRecord`.

    The case checks the static carrier for a bounded policy waiver.

    :see: :class:`~diffpes.types.WaiverRecord`
    """

    def test_public_type_is_an_equinox_module(self) -> None:
        """Expose the waiver record as an Equinox module class.

        The carrier must use the project PyTree base class.

        Notes
        -----
        The test checks class inheritance through the canonical package path.
        """
        symbol: Any = getattr(diffpes.types, "WaiverRecord")
        assert issubclass(symbol, eqx.Module)


class TestWaiverReport:
    """Verify :class:`~diffpes.types.WaiverReport`.

    The case checks the JAX-native carrier for temporal waiver validation.

    :see: :class:`~diffpes.types.WaiverReport`
    """

    def test_public_type_is_an_equinox_module(self) -> None:
        """Expose the waiver report as an Equinox module class.

        The carrier must use the project PyTree base class.

        Notes
        -----
        The test checks class inheritance through the canonical package path.
        """
        symbol: Any = getattr(diffpes.types, "WaiverReport")
        assert issubclass(symbol, eqx.Module)


class TestMakeRegistrationHandshake:
    """Verify :func:`~diffpes.types.make_registration_handshake`.

    The case checks construction of declarative owner requirements.

    :see: :func:`~diffpes.types.make_registration_handshake`
    """

    def test_factory_builds_exact_owner(self) -> None:
        """Build a handshake with one exact required model identity.

        The factory must retain each static identity without modification.

        Notes
        -----
        The test compares the static owner and model reference fields.
        """
        factory: Any = getattr(diffpes.types, "make_registration_handshake")
        result: Any = factory("kinematics", model_refs=("model@1.0.0",))
        assert result.owner_id == "kinematics"
        assert result.model_refs == ("model@1.0.0",)


class TestMakeHandshakeReport:
    """Verify :func:`~diffpes.types.make_handshake_report`.

    The case checks the JAX Boolean outcome and static missing references.

    :see: :func:`~diffpes.types.make_handshake_report`
    """

    def test_factory_retains_missing_references(self) -> None:
        """Build an incomplete report with one missing evidence identity.

        The report must retain the missing identity and false Boolean leaf.

        Notes
        -----
        The test checks both the Boolean leaf and static identity tuple.
        """
        factory: Any = getattr(diffpes.types, "make_handshake_report")
        result: Any = factory("kinematics", False, ("evidence-03",))
        assert not bool(result.complete)
        assert result.missing_ids == ("evidence-03",)


class TestMakeWaiverRecord:
    """Verify :func:`~diffpes.types.make_waiver_record`.

    The case checks construction of a bounded policy-waiver declaration.

    :see: :func:`~diffpes.types.make_waiver_record`
    """

    def test_factory_retains_absolute_utc_limits(self) -> None:
        """Build a waiver with explicit issue and expiry times.

        The factory must retain both absolute UTC strings exactly.

        Notes
        -----
        The test compares the two static UTC fields exactly.
        """
        factory: Any = getattr(diffpes.types, "make_waiver_record")
        result: Any = factory(
            "waiver-1",
            "org.diffpes.policy.research.v1",
            ("claim-1",),
            "reviewer",
            "The external result is pending.",
            "2026-07-20T00:00:00Z",
            "2026-07-22T00:00:00Z",
        )
        assert result.issued_at_utc == "2026-07-20T00:00:00Z"
        assert result.expires_at_utc == "2026-07-22T00:00:00Z"


class TestMakeWaiverReport:
    """Verify :func:`~diffpes.types.make_waiver_report`.

    The case checks the JAX Boolean outcomes for temporal validation.

    :see: :func:`~diffpes.types.make_waiver_report`
    """

    def test_factory_retains_valid_and_active_outcomes(self) -> None:
        """Build a valid and active report without errors.

        The report must contain two true scalar JAX leaves.

        Notes
        -----
        The test converts both scalar JAX leaves to Boolean values.
        """
        factory: Any = getattr(diffpes.types, "make_waiver_report")
        result: Any = factory("waiver-1", True, True)
        assert bool(result.valid)
        assert bool(result.active)
