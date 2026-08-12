"""Validate the certification contracts.

The tests cover public carrier behavior, validation, and JAX
transformations for this implementation module.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest
from beartype.typing import Any, List, Union
from jaxtyping import Array, Bool, Float64, Int32

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
from tests._assertions import assert_trees_close


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
    certificate: ForwardCertificate = make_forward_certificate(
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
    return certificate


class TestCertificationcontext:
    """Verify :class:`~diffpes.types.CertificationContext`.

    The cases cover the public carrier or factory contract in JAX PyTrees.

    :see: :class:`~diffpes.types.CertificationContext`
    """

    def test_public_symbol_has_expected_kind(self) -> None:
        """Expose ``CertificationContext`` through its canonical types package
        path.

        The case uses explicit inputs in the supported certification regime.
        It checks the public result or the documented failure state.

        Notes
        -----
        The test compares the result with explicit numerical or structural
        assertions.
        """
        symbol: object = diffpes.types.CertificationContext
        assert isinstance(symbol, type)
        assert issubclass(symbol, eqx.Module)


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
        The test compares the result with explicit numerical or structural
        assertions.
        """
        symbol: object = diffpes.types.CertifiedResult
        assert isinstance(symbol, type)
        assert issubclass(symbol, eqx.Module)

    def test_complete_result_round_trips_through_filter_jit(self) -> None:
        """Preserve the complete nested PyTree through compiled execution.

        The case uses explicit inputs in the supported certification regime.
        It checks the public result or the documented failure state.

        Notes
        -----
        The test compares the result with explicit numerical or structural
        assertions.
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


class TestExecutionmanifest:
    """Verify :class:`~diffpes.types.ExecutionManifest`.

    The cases cover the public carrier or factory contract in JAX PyTrees.

    :see: :class:`~diffpes.types.ExecutionManifest`
    """

    def test_public_symbol_has_expected_kind(self) -> None:
        """Expose ``ExecutionManifest`` through its canonical types package
        path.

        The case uses explicit inputs in the supported certification regime.
        It checks the public result or the documented failure state.

        Notes
        -----
        The test compares the result with explicit numerical or structural
        assertions.
        """
        symbol: object = diffpes.types.ExecutionManifest
        assert isinstance(symbol, type)
        assert issubclass(symbol, eqx.Module)


class TestForwardcertificate:
    """Verify :class:`~diffpes.types.ForwardCertificate`.

    The cases cover the public carrier or factory contract in JAX PyTrees.

    :see: :class:`~diffpes.types.ForwardCertificate`
    """

    def test_public_symbol_has_expected_kind(self) -> None:
        """Expose ``ForwardCertificate`` through its canonical types package
        path.

        The case uses explicit inputs in the supported certification regime.
        It checks the public result or the documented failure state.

        Notes
        -----
        The test compares the result with explicit numerical or structural
        assertions.
        """
        symbol: object = diffpes.types.ForwardCertificate
        assert isinstance(symbol, type)
        assert issubclass(symbol, eqx.Module)

    def test_complete_graph_has_traced_numerical_leaves(self) -> None:
        """Verify every dynamic leaf in a complete certificate is a JAX array.

        The case uses explicit inputs in the supported certification regime.
        It checks the public result or the documented failure state.

        Notes
        -----
        The test constructs the shared complete fixture and inspects its
        flattened JAX
        leaves while static vocabulary remains outside the numerical tree.
        """
        certificate: ForwardCertificate = _certificate()
        leaves: List[
            Union[
                Float64[Array, "..."],
                Int32[Array, "..."],
                Bool[Array, "..."],
            ]
        ] = jax.tree.leaves(certificate)
        assert leaves
        assert all(isinstance(leaf, jax.Array) for leaf in leaves)

    def test_static_vocabulary_is_absent_from_numerical_leaves(self) -> None:
        """Keep identifiers static while retaining all numerical evidence.

        The case uses explicit inputs in the supported certification regime.
        It checks the public result or the documented failure state.

        Notes
        -----
        The test compares the result with explicit numerical or structural
        assertions.
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


class TestMakeCertificationContext:
    """Verify :func:`~diffpes.types.make_certification_context`.

    The cases cover the public carrier or factory contract in JAX PyTrees.

    :see: :func:`~diffpes.types.make_certification_context`
    """

    def test_public_symbol_has_expected_kind(self) -> None:
        """Expose ``make_certification_context`` through its canonical types
        package path.

        The case uses explicit inputs in the supported certification regime.
        It checks the public result or the documented failure state.

        Notes
        -----
        The test compares the result with explicit numerical or structural
        assertions.
        """
        symbol: object = diffpes.types.make_certification_context
        assert callable(symbol)

    def test_context_cross_validates_model_identity(self) -> None:
        """Reject prepared contexts that combine different model identities.

        The case uses explicit inputs in the supported certification regime.
        It checks the public result or the documented failure state.

        Notes
        -----
        The test compares the result with explicit numerical or structural
        assertions.
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
        """Expose ``make_certified_result`` through its canonical types
        package path.

        The case uses explicit inputs in the supported certification regime.
        It checks the public result or the documented failure state.

        Notes
        -----
        The test compares the result with explicit numerical or structural
        assertions.
        """
        symbol: object = diffpes.types.make_certified_result
        assert callable(symbol)

    def test_certified_envelope_preserves_primal_jvp_and_vjp(self) -> None:
        """Show that attaching a certificate does not alter model derivatives.

        The case uses explicit inputs in the supported certification regime.
        It checks the public result or the documented failure state.

        Notes
        -----
        The test compares the result with explicit numerical or structural
        assertions.
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

        def ordinary(value: Float64[Array, ""]) -> Float64[Array, ""]:
            result: Float64[Array, ""] = jnp.sin(value) + value**2
            return result

        def certified(value: Float64[Array, ""]) -> Float64[Array, ""]:
            result: Float64[Array, ""] = make_certified_result(
                ordinary(value), certificate
            ).value
            return result

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


class TestMakeExecutionManifest:
    """Verify :func:`~diffpes.types.make_execution_manifest`.

    The cases cover the public carrier or factory contract in JAX PyTrees.

    :see: :func:`~diffpes.types.make_execution_manifest`
    """

    def test_public_symbol_has_expected_kind(self) -> None:
        """Expose ``make_execution_manifest`` through its canonical types
        package path.

        The case uses explicit inputs in the supported certification regime.
        It checks the public result or the documented failure state.

        Notes
        -----
        The test compares the result with explicit numerical or structural
        assertions.
        """
        symbol: object = diffpes.types.make_execution_manifest
        assert callable(symbol)


class TestMakeForwardCertificate:
    """Verify :func:`~diffpes.types.make_forward_certificate`.

    The cases cover the public carrier or factory contract in JAX PyTrees.

    :see: :func:`~diffpes.types.make_forward_certificate`
    """

    def test_public_symbol_has_expected_kind(self) -> None:
        """Expose ``make_forward_certificate`` through its canonical types
        package path.

        The case uses explicit inputs in the supported certification regime.
        It checks the public result or the documented failure state.

        Notes
        -----
        The test compares the result with explicit numerical or structural
        assertions.
        """
        symbol: object = diffpes.types.make_forward_certificate
        assert callable(symbol)

    def test_certificate_rejects_cross_record_inconsistency(self) -> None:
        """Reject policy, dependency, and duplicate identity mismatches.

        The case uses explicit inputs in the supported certification regime.
        It checks the public result or the documented failure state.

        Notes
        -----
        The test compares the result with explicit numerical or structural
        assertions.
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
