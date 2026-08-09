"""Validate resolver-backed reproduction of certified forward results.

The tests run a registered closed-form square model. They compare the rerun
with a separately resolved result artifact.
"""

import uuid

import jax.numpy as jnp
from beartype.typing import Any

from diffpes.certify import (
    certify_forward,
    checksum_pytree,
    mapping_artifact_resolver,
    prepare_certification,
    register_model,
    reproduce_forward,
)
from diffpes.types import (
    make_artifact_ref,
    make_execution_manifest,
    make_forward_certificate,
    make_forward_model_spec,
)


def _certificate(inputs: Any, expected: Any) -> Any:
    """PRIVATE: Build a certificate with resolved input and result artifacts.

    Parameters
    ----------
    inputs : Any
        Normalized model input array for the certified run.
    expected : Any
        Result array that the result artifact must resolve to.

    Returns
    -------
    result : Any
        Forward certificate whose artifact tuple carries content
        checksums for the inputs and the expected result.

    Implementation Logic
    --------------------
    Registers a closed-form square model under a unique UUID-suffixed
    identity, certifies one forward run at spectrum rank 1, and rebuilds
    the certificate with two artifact references that carry
    normalized-content checksums for the inputs and the expected value.
    """
    suffix: str = uuid.uuid4().hex
    model_id: str = f"org.diffpes.model.reproduction_test.{suffix}"
    model: Any = make_forward_model_spec(
        model_id=model_id,
        model_version="1.0.0",
        observable_id="org.diffpes.observable.test.scalar",
        implementation_ref="tests.square",
        differentiable_paths=("x",),
    )
    register_model(model, lambda value: value**2)
    manifest: Any = make_execution_manifest(
        execution_id=f"execution-{suffix}",
        model_ref=f"{model_id}@1.0.0",
        schema_version="1.0.0",
        package_version="test",
        source_checksum="source",
        environment_checksum="environment",
        backend="cpu",
        precision_policy="float64",
        deterministic=True,
        started_at_utc="2026-07-21T00:00:00Z",
    )
    context: Any = prepare_certification(
        model_id,
        "1.0.0",
        manifest,
        policy_id="org.diffpes.policy.exploratory.v1",
    )
    certified: Any = certify_forward(context, inputs, spectrum_rank=1)

    def artifact(artifact_id: str, value: Any, role: str) -> Any:
        reference: Any = make_artifact_ref(
            artifact_id=artifact_id,
            media_type="application/x-diffpes-array",
            byte_checksum=None,
            content_checksum=checksum_pytree(
                value,
                record_kind="normalized-content",
            ),
            semantic_checksum=(
                "sha256:1:semantic:"
                "0000000000000000000000000000000000000000000000000000000000000000"
            ),
            locator=None,
            role=role,
        )
        return reference

    certificate: Any = certified.certificate
    result: Any = make_forward_certificate(
        manifest=certificate.manifest,
        model=certificate.model,
        artifacts=(
            artifact("input", inputs, "normalized-input"),
            artifact("result", expected, "result"),
        ),
        transformations=certificate.transformations,
        evidence=certificate.evidence,
        claims=certificate.claims,
        domains=certificate.domains,
        derivatives=certificate.derivatives,
        dependencies=certificate.dependencies,
        sensitivities=certificate.sensitivities,
        information=certificate.information,
        policy_report=certificate.policy_report,
        policy_id=certificate.policy_id,
        certificate_checksum=certificate.certificate_checksum,
        waivers=certificate.waivers,
    )
    return result


class TestReproduceForward:
    """Verify :func:`~diffpes.certify.reproduce_forward`.

    The cases compare a registered square model with resolved output arrays.

    :see: :func:`~diffpes.certify.reproduce_forward`
    """

    def test_exact_rerun_reports_reproduction(self) -> None:
        """Report success for an exact rerun of the registered square model.

        The maximum absolute error must equal zero.

        Notes
        -----
        The test resolves input 2 and expected output 4 as float64 arrays.
        """
        inputs: Any = jnp.array([2.0])
        expected: Any = jnp.array([4.0])
        certificate: Any = _certificate(inputs, expected)
        resolver: Any = mapping_artifact_resolver(
            {"input": inputs, "result": expected}
        )
        report: Any = reproduce_forward(certificate, resolver=resolver)
        assert bool(report.reproduced)
        assert float(report.max_abs_error) == 0.0

    def test_changed_expected_result_reports_failure(self) -> None:
        """Report failure when the resolved result differs from the rerun.

        The maximum absolute error must equal the analytic difference.

        Notes
        -----
        The test compares the analytic square value 4 with the stored value 5.
        """
        inputs: Any = jnp.array([2.0])
        expected: Any = jnp.array([5.0])
        certificate: Any = _certificate(inputs, expected)
        resolver: Any = mapping_artifact_resolver(
            {"input": inputs, "result": expected}
        )
        report: Any = reproduce_forward(certificate, resolver=resolver)
        assert not bool(report.reproduced)
        assert jnp.allclose(report.max_abs_error, 1.0)

    def test_large_result_uses_combined_tolerance(self) -> None:
        """Verify relative agreement for a large result with absolute error.

        The combined tolerance must accept an error above the absolute term.

        Notes
        -----
        The test compares 1,000,000 with 1,000,000.5 at tolerance 1e-6.
        """
        inputs: Any = jnp.array([1000.0])
        expected: Any = jnp.array([1000000.5])
        certificate: Any = _certificate(inputs, expected)
        resolver: Any = mapping_artifact_resolver(
            {"input": inputs, "result": expected}
        )
        report: Any = reproduce_forward(
            certificate,
            resolver=resolver,
            tolerance=1e-6,
        )
        assert bool(report.reproduced)
        assert jnp.allclose(report.max_abs_error, 0.5)
