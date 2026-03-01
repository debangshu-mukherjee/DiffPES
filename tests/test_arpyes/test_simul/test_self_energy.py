"""Tests for energy-dependent self-energy evaluation."""

import jax
import jax.numpy as jnp
import pytest

from arpyes.simul.self_energy import evaluate_self_energy
from arpyes.types import make_self_energy_config


class TestEvaluateSelfEnergy:
    """Tests for evaluate_self_energy."""

    def test_constant_mode(self):
        """Constant mode returns the coefficient everywhere."""
        config = make_self_energy_config(gamma=0.15, mode="constant")
        energy = jnp.linspace(-3, 1, 100)
        gamma = evaluate_self_energy(energy, config)
        assert jnp.allclose(gamma, 0.15)

    def test_polynomial_mode(self):
        """Polynomial mode evaluates correctly."""
        # gamma(E) = 0.1 + 0.05*E (linear)
        config = make_self_energy_config(
            mode="polynomial",
            coefficients=jnp.array([0.05, 0.1]),  # polyval: highest degree first
        )
        energy = jnp.array([0.0, 1.0, -1.0])
        gamma = evaluate_self_energy(energy, config)
        expected = jnp.array([0.1, 0.15, 0.05])
        assert jnp.allclose(gamma, expected, atol=1e-10)

    def test_tabulated_mode(self):
        """Tabulated mode interpolates correctly."""
        nodes = jnp.array([-3.0, 0.0, 1.0])
        coeffs = jnp.array([0.05, 0.1, 0.2])
        config = make_self_energy_config(
            mode="tabulated",
            coefficients=coeffs,
            energy_nodes=nodes,
        )
        energy = jnp.array([0.0, 0.5])
        gamma = evaluate_self_energy(energy, config)
        assert float(gamma[0]) == pytest.approx(0.1, abs=1e-10)
        assert float(gamma[1]) == pytest.approx(0.15, abs=1e-10)

    def test_constant_gradient(self):
        """Gradient w.r.t. constant coefficient is finite."""
        def loss(coeff):
            config = make_self_energy_config(
                mode="constant",
                coefficients=jnp.array([coeff]),
            )
            energy = jnp.array([0.0, 1.0])
            return jnp.sum(evaluate_self_energy(energy, config))

        grad = jax.grad(loss)(jnp.array(0.1))
        assert jnp.isfinite(grad)

    def test_polynomial_gradient(self):
        """Gradient w.r.t. polynomial coefficients is finite."""
        def loss(coeffs):
            config = make_self_energy_config(
                mode="polynomial",
                coefficients=coeffs,
            )
            energy = jnp.linspace(-1, 1, 50)
            return jnp.sum(evaluate_self_energy(energy, config))

        grad = jax.grad(loss)(jnp.array([0.01, 0.1]))
        assert jnp.all(jnp.isfinite(grad))
