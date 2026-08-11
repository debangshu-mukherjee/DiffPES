"""Validate compiled JAX-native certified execution.

The tests cover public behavior, differentiability, validation, and stable
scientific identity in the supported certification regime.
"""

import uuid
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
from beartype.typing import Any, Callable
from jaxtyping import PyTree

from diffpes.certify import (
    certify_forward,
    certify_forward_checked,
    evaluate_evidence,
    prepare_certification,
    register_check,
    register_model,
    verify_certificate,
)
from diffpes.inout import load_certificate_json, save_certificate_json
from diffpes.types import (
    CertificationContext,
    ExecutionManifest,
    ForwardModelSpec,
    make_artifact_ref,
    make_domain_result,
    make_evidence_lineage,
    make_execution_manifest,
    make_forward_model_spec,
    make_human_attestation_ref,
)


def _context(executor: Callable[[PyTree], PyTree]) -> CertificationContext:
    """PRIVATE: Prepare a certification context for one fresh test model.

    Parameters
    ----------
    executor : Callable[[PyTree], PyTree]
        Callable that the registry runs as the model forward pass.

    Returns
    -------
    context : CertificationContext
        Prepared context under the exploratory policy.

    Implementation Logic
    --------------------
    Registers a forward-model spec with a unique UUID-suffixed identity.
    Builds a deterministic CPU float64 manifest. Prepares certification
    for model version 1.0.0.
    """
    suffix: str
    model_id: str
    spec: ForwardModelSpec
    manifest: ExecutionManifest
    suffix = uuid.uuid4().hex
    model_id = f"org.diffpes.model.test{suffix}"
    spec = make_forward_model_spec(
        model_id,
        "1.0.0",
        "org.diffpes.observable.test.scalar",
        "tests.quadratic",
        differentiable_paths=("x",),
    )
    register_model(spec, executor)
    manifest = make_execution_manifest(
        f"execution-{suffix}",
        f"{model_id}@1.0.0",
        "1",
        "test",
        "source",
        "environment",
        "cpu",
        "f64",
        True,
        "2026-07-21T00:00:00Z",
    )
    context: CertificationContext = prepare_certification(
        model_id,
        "1.0.0",
        manifest,
        policy_id="org.diffpes.policy.exploratory.v1",
    )
    return context


class TestVerifyCertificate:
    """Verify :func:`~diffpes.certify.verify_certificate`.

    The cases cover the public behavior in the supported certification regime.

    :see: :func:`~diffpes.certify.verify_certificate`
    """

    def test_compiled_quadratic_certificate(self) -> None:
        """Produce the correct value, JVP, dependency, and spectrum.

        The case uses explicit inputs in the supported certification regime.
        It checks the public result or the documented failure state.

        Notes
        -----
        The test checks the result with explicit assertions.
        """
        context: Any
        result: Any
        context = _context(lambda x: x**2)
        result = certify_forward(context, jnp.array([2.0]), spectrum_rank=1)
        assert jnp.allclose(result.value, 4.0)
        assert jnp.allclose(result.certificate.derivatives.jvp_probes, 4.0)
        assert result.certificate.dependencies.structural.tolist() == [[True]]
        assert jnp.allclose(
            result.certificate.information.singular_values, 4.0
        )
        assert bool(verify_certificate(result.certificate).structure_valid)

    def test_lineage_and_attestation_propagate_separately(self) -> None:
        """Carry named lineage and human review through certified execution.

        The case prepares and certifies one reference with both record kinds.

        Notes
        -----
        The assertions keep computational ancestry separate from human review.
        """
        context: CertificationContext = _context(lambda value: value**2)
        evidence: Any = evaluate_evidence(
            "evidence.lineage",
            "method.closed_form",
            jnp.zeros(1),
            jnp.zeros(1),
            jnp.zeros(1),
            lineage=make_evidence_lineage(
                implementation_refs=("reference.impl",),
                generator_refs=("reference.generator",),
                artifact_refs=("reference.artifact",),
                derivation_refs=("reference.derivation",),
                relationship_ids=(
                    "independent-derivation:reference.derivation",
                ),
            ),
            human_attestation_refs=("attestation.review",),
        )
        attestation: Any = make_human_attestation_ref(
            "attestation.review",
            "reviewer.test",
            ("evidence.lineage",),
            "Lineage reviewed.",
            "2026-07-24T00:00:00Z",
        )
        artifact: Any = make_artifact_ref(
            "reference.artifact",
            "application/octet-stream",
            None,
            "content.reference",
            "semantic.reference",
            None,
            "reference",
        )
        prepared: CertificationContext = prepare_certification(
            context.model.model_id,
            context.model.model_version,
            context.manifest,
            policy_id=context.policy_id,
            artifacts=(artifact,),
            evidence=(evidence,),
            attestations=(attestation,),
        )
        certified: Any = certify_forward(
            prepared,
            jnp.array([2.0]),
            spectrum_rank=1,
        )
        assert certified.certificate.evidence[0].lineage == evidence.lineage
        assert certified.certificate.attestations == (attestation,)
        assert bool(verify_certificate(certified.certificate).structure_valid)

    def test_one_sided_domain_claim_is_internally_consistent(self) -> None:
        """Verify a domain claim that uses a signed positive margin.

        Domain claims mirror their predicate result rather than a symmetric
        test.

        Notes
        -----
        The positive-domain residual exceeds its zero comparison tolerance.
        """
        context: CertificationContext = _context(lambda value: value**2)
        check_id: str = f"org.diffpes.domain.test.{uuid.uuid4().hex}"

        def positive_check(inputs: Any) -> Any:
            measured: Any = jnp.min(inputs)
            result: Any = make_domain_result(
                predicate_id=check_id,
                measured=measured,
                reference=0.0,
                residual=measured,
                tolerance=0.0,
                margin=measured,
                passed=measured > 0.0,
                checked=True,
                in_domain=measured > 0.0,
                severity_code=2,
            )
            return result

        register_check(check_id, positive_check)
        checked_context: CertificationContext = prepare_certification(
            context.model.model_id,
            context.model.model_version,
            context.manifest,
            policy_id=context.policy_id,
            check_ids=(check_id,),
        )
        certified: Any = certify_forward(
            checked_context,
            jnp.array([2.0]),
            spectrum_rank=1,
        )
        report: Any = verify_certificate(certified.certificate)
        assert bool(report.structure_valid)

    def test_changed_claim_residual_is_detected(self) -> None:
        """Detect a claim residual that disagrees with measured values.

        The policy Boolean remains unchanged in this single-field mutation.

        Notes
        -----
        The test changes only the execution-identity claim residual.
        """
        context: Any = _context(lambda value: value**2)
        certified: Any = certify_forward(
            context,
            jnp.array([2.0]),
            spectrum_rank=1,
        )
        changed: Any = eqx.tree_at(
            lambda item: item.claims[0].residual,
            certified.certificate,
            jnp.ones(1),
        )
        report: Any = verify_certificate(changed)
        assert not bool(report.structure_valid)

    def test_changed_policy_truth_table_is_detected(self) -> None:
        """Detect a changed policy vector while claim records stay identical.

        Offline verification must compare the complete recomputed truth table.

        Notes
        -----
        The test changes only the stored achieved-level vector.
        """
        context: Any = _context(lambda value: value**2)
        certified: Any = certify_forward(
            context,
            jnp.array([2.0]),
            spectrum_rank=1,
        )
        changed: Any = eqx.tree_at(
            lambda item: item.policy_report.achieved,
            certified.certificate,
            jnp.zeros_like(certified.certificate.policy_report.achieved),
        )
        report: Any = verify_certificate(changed)
        assert not bool(report.evidence_valid)

    def test_external_claim_must_match_attached_evidence(self) -> None:
        """Detect a self-consistent claim that changes attached evidence.

        A claim cannot replace values from its named external evidence record.

        Notes
        -----
        The mutation changes every dependent claim field but leaves evidence
        intact.
        """
        context: Any = _context(lambda value: value**2)
        evidence: Any = evaluate_evidence(
            "evidence.test",
            "method.closed_form",
            jnp.array([1.0]),
            jnp.array([1.0]),
            jnp.array([0.1]),
        )
        evidence_context: Any = prepare_certification(
            context.model.model_id,
            context.model.model_version,
            context.manifest,
            policy_id=context.policy_id,
            evidence=(evidence,),
        )
        certified: Any = certify_forward(
            evidence_context,
            jnp.array([2.0]),
            spectrum_rank=1,
        )
        changed: Any = eqx.tree_at(
            lambda item: (
                item.claims[-1].measured,
                item.claims[-1].residual,
                item.claims[-1].margin,
                item.claims[-1].passed,
            ),
            certified.certificate,
            (
                jnp.array([2.0]),
                jnp.array([1.0]),
                jnp.asarray(-0.9),
                jnp.asarray(False),
            ),
        )
        report: Any = verify_certificate(changed)
        assert not bool(report.structure_valid)


class TestCertifyForward:
    """Verify :func:`~diffpes.certify.certify_forward`.

    The cases cover the public behavior in the supported certification regime.

    :see: :func:`~diffpes.certify.certify_forward`
    """

    def test_gradient_through_result_and_information(self) -> None:
        """Differentiate both the observable and certificate evidence.

        The case uses explicit inputs in the supported certification regime.
        It checks the public result or the documented failure state.

        Notes
        -----
        The test checks the result with explicit assertions.
        """
        context: Any
        value_grad: Any
        information_grad: Any
        context = _context(lambda x: x**3)
        value_grad = jax.grad(
            lambda x: jnp.sum(
                certify_forward(context, x, spectrum_rank=1).value
            )
        )(jnp.array([2.0]))
        information_grad = jax.grad(
            lambda x: jnp.sum(
                certify_forward(
                    context, x, spectrum_rank=1
                ).certificate.information.singular_values
            )
        )(jnp.array([2.0]))
        assert jnp.allclose(value_grad, 12.0)
        assert jnp.allclose(information_grad, 12.0)

    def test_vmap_certified_execution(self) -> None:
        """Batch complete certified executions with JAX VMAP.

        The case uses explicit inputs in the supported certification regime.
        It checks the public result or the documented failure state.

        Notes
        -----
        The test checks the result with explicit assertions.
        """
        context: Any
        values: Any
        context = _context(lambda x: x**2)
        values = jax.vmap(
            lambda x: certify_forward(context, x, spectrum_rank=1).value
        )(jnp.array([1.0, 2.0, 3.0]))
        assert jnp.allclose(values, jnp.array([1.0, 4.0, 9.0]))

    def test_complex_output_keeps_both_real_coordinates(self) -> None:
        """Retain real and imaginary information from a complex output.

        The local information singular value must equal the complex gain norm.

        Notes
        -----
        The analytic map multiplies one real input by ``1 + 2j``.
        """
        context: Any = _context(lambda value: (1.0 + 2.0j) * value)
        result: Any = certify_forward(
            context,
            jnp.array([3.0]),
            spectrum_rank=1,
        )
        assert jnp.allclose(
            result.certificate.derivatives.jvp_probes,
            jnp.array([[1.0, 2.0]]),
        )
        assert jnp.allclose(
            result.certificate.information.singular_values,
            jnp.sqrt(5.0),
        )

    def test_zero_rank_certificate_round_trips(self, tmp_path: Path) -> None:
        """Persist a certificate for a constant zero-information model.

        The finite zero condition sentinel must survive the JSON round trip.

        Notes
        -----
        The model output is constant, so its information spectrum has rank
        zero.
        """

        def constant(value: Any) -> Any:
            constant_value: Any = jnp.zeros_like(value)
            return constant_value

        context: Any = _context(constant)
        result: Any = certify_forward(
            context,
            jnp.array([2.0]),
            spectrum_rank=1,
        )
        path: Path = tmp_path / "zero-rank-certificate.json"
        save_certificate_json(result.certificate, path)
        restored: Any = load_certificate_json(path)
        assert int(restored.information.effective_rank) == 0
        assert float(restored.information.condition_estimate) == 0.0


class TestPrepareCertification:
    """Verify :func:`~diffpes.certify.prepare_certification`.

    The cases cover eager resolution of stable model and policy identities.

    :see: :func:`~diffpes.certify.prepare_certification`
    """

    def test_context_binds_exact_model_identity(self) -> None:
        """Bind a registered model specification into a static context.

        The case uses explicit inputs in the supported certification regime.
        It checks the public result or the documented failure state.

        Notes
        -----
        The test checks the result with explicit assertions.
        """
        context: CertificationContext = _context(lambda value: value**2)
        assert context.model.model_id in context.manifest.model_ref
        assert context.model.model_version in context.manifest.model_ref


class TestCertifyForwardChecked:
    """Verify :func:`~diffpes.certify.certify_forward_checked`.

    The case returns a structured JAX error for a failed hard domain check.

    :see: :func:`~diffpes.certify.certify_forward_checked`
    """

    def test_hard_domain_failure_is_structured(self) -> None:
        """Return a checkify error without Python branching in the kernel.

        The error must name the hard check while the result stays available.

        Notes
        -----
        The test registers one hard check that has a negative margin.
        """
        context: CertificationContext = _context(lambda value: value**2)
        check_id: str = f"org.diffpes.domain.test.{uuid.uuid4().hex}"

        def failed_check(inputs: Any) -> Any:
            del inputs
            result: Any = make_domain_result(
                predicate_id=check_id,
                measured=-1.0,
                reference=0.0,
                residual=-1.0,
                tolerance=0.0,
                margin=-1.0,
                passed=False,
                checked=True,
                in_domain=False,
                severity_code=2,
            )
            return result

        register_check(check_id, failed_check)
        checked_context: CertificationContext = prepare_certification(
            context.model.model_id,
            context.model.model_version,
            context.manifest,
            policy_id=context.policy_id,
            check_ids=(check_id,),
        )
        error: Any
        result: Any
        error, result = certify_forward_checked(
            checked_context,
            jnp.array([2.0]),
            spectrum_rank=1,
        )
        assert "hard domain check failed" in str(error.get())
        assert jnp.allclose(result.value, 4.0)
