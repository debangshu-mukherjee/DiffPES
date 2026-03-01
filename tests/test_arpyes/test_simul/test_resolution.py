"""Tests for momentum resolution broadening."""

import jax
import jax.numpy as jnp
import pytest

from arpyes.simul.resolution import apply_momentum_broadening


class TestApplyMomentumBroadening:
    """Tests for apply_momentum_broadening."""

    def test_identity_with_zero_dk(self):
        """dk → 0 returns approximately the original intensity."""
        K, E = 20, 50
        intensity = jnp.ones((K, E))
        k_distances = jnp.linspace(0, 1, K)
        # Very small dk = essentially no broadening
        result = apply_momentum_broadening(intensity, k_distances, 1e-15)
        assert jnp.allclose(result, intensity, atol=1e-3)

    def test_smoothing_effect(self):
        """Large dk smooths the k-axis."""
        K, E = 50, 10
        # Create a delta-like peak in k-space
        intensity = jnp.zeros((K, E))
        intensity = intensity.at[25, :].set(1.0)
        k_distances = jnp.linspace(0, 1, K)

        result = apply_momentum_broadening(intensity, k_distances, 0.1)
        # Peak should be spread out
        assert float(result[25, 0]) < 1.0
        # Neighbors should be nonzero
        assert float(result[24, 0]) > 0.0
        assert float(result[26, 0]) > 0.0

    def test_conservation(self):
        """Total intensity is approximately conserved."""
        K, E = 30, 20
        intensity = jnp.abs(
            jnp.sin(jnp.linspace(0, 3, K))[:, None]
        ) * jnp.ones((1, E))
        k_distances = jnp.linspace(0, 1, K)
        result = apply_momentum_broadening(intensity, k_distances, 0.05)
        assert float(jnp.sum(result)) == pytest.approx(
            float(jnp.sum(intensity)), rel=0.1
        )

    def test_output_shape(self):
        """Output shape matches input."""
        K, E = 15, 25
        intensity = jnp.ones((K, E))
        k_distances = jnp.linspace(0, 1, K)
        result = apply_momentum_broadening(intensity, k_distances, 0.1)
        assert result.shape == (K, E)

    def test_gradient_wrt_dk(self):
        """Gradient w.r.t. dk is finite."""
        K, E = 10, 5
        intensity = jnp.ones((K, E))
        k_distances = jnp.linspace(0, 1, K)

        def loss(dk):
            return jnp.sum(
                apply_momentum_broadening(intensity, k_distances, dk)
            )

        grad = jax.grad(loss)(jnp.array(0.1))
        assert jnp.isfinite(grad)
