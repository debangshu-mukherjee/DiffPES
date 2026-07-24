"""Validate differentiable certification evidence.

The tests cover public behavior, differentiability, validation, and stable
scientific identity in the supported certification regime.
"""

import jax
import jax.numpy as jnp
import pytest
from beartype.typing import Any

from diffpes.certify import (
    derivative_evidence,
    evaluate_claim,
    evaluate_domain,
    evaluate_evidence,
)
from diffpes.types import make_evidence_lineage


class TestEvaluateClaim:
    """Verify :func:`~diffpes.certify.evaluate_claim`.

    The cases cover the public behavior in the supported certification regime.

    :see: :func:`~diffpes.certify.evaluate_claim`
    """

    def test_claim_retains_differentiable_margin(self) -> None:
        """Differentiate the tolerance margin beneath a Boolean claim.

        The case uses explicit inputs in the supported certification regime.
        It checks the public result or the documented failure state.

        Notes
        -----
        The test compares the result with explicit numerical or structural assertions.
        """

        def margin(measured: Any) -> Any:
            claim: Any
            claim = evaluate_claim(
                "claim.test",
                "subject.test",
                "verification.test",
                measured,
                jnp.zeros(1),
                jnp.ones(1),
            )
            return claim.margin

        assert jnp.allclose(jax.grad(margin)(jnp.array([0.25])), -1.0)


class TestEvaluateDomain:
    """Verify :func:`~diffpes.certify.evaluate_domain`.

    The cases cover the public behavior in the supported certification regime.

    :see: :func:`~diffpes.certify.evaluate_domain`
    """

    def test_domain_reports_signed_margin(self) -> None:
        """Preserve distance to the validity boundary.

        The case uses explicit inputs in the supported certification regime.
        It checks the public result or the documented failure state.

        Notes
        -----
        The test compares the result with explicit numerical or structural assertions.
        """
        result: Any
        result = evaluate_domain(
            "domain.test", jnp.array(1.2), jnp.array(1.0), jnp.array(0.5)
        )
        assert bool(result.passed)
        assert jnp.allclose(result.margin, 0.3)


class TestEvaluateEvidence:
    """Verify :func:`~diffpes.certify.evaluate_evidence`.

    The cases cover the public behavior in the supported certification regime.

    :see: :func:`~diffpes.certify.evaluate_evidence`
    """

    def test_external_evidence_preserves_residual(self) -> None:
        """Record numerical comparison arrays without reducing them.

        The case uses explicit inputs in the supported certification regime.
        It checks the public result or the documented failure state.

        Notes
        -----
        The test compares the result with explicit numerical or structural assertions.
        """
        result: Any
        result = evaluate_evidence(
            "evidence.test",
            "method.closed_form",
            jnp.array([1.1, 1.9]),
            jnp.array([1.0, 2.0]),
            jnp.array([0.2, 0.2]),
            lineage=make_evidence_lineage(
                implementation_refs=("reference.impl",),
                generator_refs=("reference.generator",),
                artifact_refs=("reference.artifact",),
                derivation_refs=("reference.derivation",),
                relationship_ids=(
                    "independent-derivation:reference.derivation",
                ),
            ),
        )
        assert jnp.allclose(result.residual, jnp.array([0.1, -0.1]))
        assert not hasattr(result, "independent")


class TestDerivativeEvidence:
    """Verify :func:`~diffpes.certify.derivative_evidence`.

    The cases cover the public behavior in the supported certification regime.

    :see: :func:`~diffpes.certify.derivative_evidence`
    """

    def test_derivative_evidence_matches_quadratic(self) -> None:
        """Compare retained JVPs and VJPs against central differences.

        The case uses explicit inputs in the supported certification regime.
        It checks the public result or the documented failure state.

        Notes
        -----
        The test compares the result with explicit numerical or structural assertions.
        """
        result: Any
        result = derivative_evidence(
            lambda x: x**2,
            jnp.array([2.0]),
            jnp.ones((1, 1)),
            jnp.ones((1, 1)),
            input_paths=("$[0]",),
            output_projection_ids=("square",),
            scales=jnp.ones(1),
            spectrum_rank=1,
        )
        assert bool(result.finite)
        assert bool(result.fd_correct)
        assert jnp.allclose(result.jvp_probes, 4.0)
        assert jnp.allclose(result.vjp_probes, 4.0)

    def test_derivative_evidence_linearizes_once(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Reuse one nonlinear linearization for all information probes.

        Central differences can rerun the model but cannot linearize it again.

        Notes
        -----
        The wrapper counts calls to the public JAX linearization primitive.
        """
        original: Any = jax.linearize
        calls: list[None] = []

        def counted(*args: Any, **kwargs: Any) -> Any:
            calls.append(None)
            return original(*args, **kwargs)

        monkeypatch.setattr(jax, "linearize", counted)
        derivative_evidence(
            lambda value: value**2,
            jnp.array([2.0]),
            jnp.ones((1, 1)),
            jnp.ones((1, 1)),
            input_paths=("$[0]",),
            output_projection_ids=("square",),
            scales=jnp.ones(1),
            spectrum_rank=1,
        )
        assert len(calls) == 1
