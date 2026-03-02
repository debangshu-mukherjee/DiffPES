"""Tests for momentum resolution broadening.

Extended Summary
----------------
Validates :func:`diffpes.simul.resolution.apply_momentum_broadening`, which
convolves a 2D ARPES intensity map along the k-axis with a Gaussian kernel
to simulate the finite angular/momentum resolution of an electron analyser.
Tests cover the identity limit (dk -> 0), the smoothing effect on a
delta-like peak, approximate conservation of total intensity, output shape
preservation, and differentiability of the broadening width parameter via
``jax.grad``.

Routine Listings
----------------
:class:`TestApplyMomentumBroadening`
    Tests for apply_momentum_broadening.
"""

import jax
import jax.numpy as jnp
import pytest

from diffpes.simul.resolution import apply_momentum_broadening


class TestApplyMomentumBroadening:
    """Tests for :func:`diffpes.simul.resolution.apply_momentum_broadening`.

    Validates the Gaussian momentum-broadening convolution applied along
    the k-axis of a 2D intensity map. Tests verify limiting behaviour
    (identity at dk -> 0), the physical smoothing effect, approximate
    intensity conservation (since the Gaussian kernel is normalized),
    shape preservation, and JAX differentiability with respect to the
    broadening width dk.
    """

    def test_identity_with_zero_dk(self):
        """Verify that vanishing dk returns approximately the original intensity.

        Test Logic
        ----------
        1. **Setup**: Create a uniform intensity map of shape (20, 50)
           and a linearly spaced k_distances array from 0 to 1. Set
           dk = 1e-15 (effectively zero), so the Gaussian kernel
           collapses to a delta function and the convolution should
           be the identity operation.
        2. **Apply broadening**: Call ``apply_momentum_broadening`` with
           the near-zero dk.
        3. **Compare**: Assert the result is element-wise close to the
           original intensity within atol=1e-3.

        Asserts
        -------
        The broadened intensity equals the input to within 1e-3,
        confirming the correct identity limit when the broadening
        width vanishes.
        """
        K, E = 20, 50
        intensity = jnp.ones((K, E))
        k_distances = jnp.linspace(0, 1, K)
        # Very small dk = essentially no broadening
        result = apply_momentum_broadening(intensity, k_distances, 1e-15)
        assert jnp.allclose(result, intensity, atol=1e-3)

    def test_smoothing_effect(self):
        """Verify that finite dk smooths a delta-like peak along the k-axis.

        Test Logic
        ----------
        1. **Setup**: Create a 2D intensity map (50 x 10) that is zero
           everywhere except at k-index 25, where it is 1.0 across all
           energy channels. This represents a delta-like feature in
           k-space.
        2. **Apply broadening**: Call ``apply_momentum_broadening`` with
           dk=0.1, which should spread the peak to neighbouring
           k-points via the Gaussian kernel.
        3. **Check peak reduction**: Assert the peak value at index 25
           is reduced below 1.0 (the original delta height).
        4. **Check neighbour activation**: Assert that the immediately
           adjacent k-points (indices 24 and 26) have positive
           intensity, confirming the peak has been broadened.

        Asserts
        -------
        The peak is reduced and neighbours are activated, confirming
        that the Gaussian convolution physically smooths sharp
        k-space features as expected from finite analyser resolution.
        """
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
        """Verify that total intensity is approximately conserved under broadening.

        Test Logic
        ----------
        1. **Setup**: Create a 2D intensity map (30 x 20) with a
           spatially varying (sinusoidal) k-profile broadcast across
           all energy channels. Use dk=0.05 for moderate broadening.
        2. **Apply broadening**: Call ``apply_momentum_broadening``.
        3. **Compare totals**: Assert the summed intensity before and
           after broadening agree within 10% relative tolerance.

        Asserts
        -------
        The total intensity is conserved to within 10% relative
        tolerance. Perfect conservation is not expected due to
        edge effects (truncation of the Gaussian kernel at the
        boundaries of the k-grid), but approximate conservation
        confirms the kernel normalization is correct.
        """
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
        """Verify that the output shape matches the input shape.

        Test Logic
        ----------
        1. **Setup**: Create a uniform intensity map of shape (15, 25)
           and apply momentum broadening with dk=0.1.
        2. **Check shape**: Assert the output shape is ``(15, 25)``,
           identical to the input.

        Asserts
        -------
        The output shape equals the input shape ``(K, E)``, confirming
        that the convolution does not alter the grid dimensions.
        """
        K, E = 15, 25
        intensity = jnp.ones((K, E))
        k_distances = jnp.linspace(0, 1, K)
        result = apply_momentum_broadening(intensity, k_distances, 0.1)
        assert result.shape == (K, E)

    def test_gradient_wrt_dk(self):
        """Verify that the gradient of total intensity w.r.t. dk is finite.

        Test Logic
        ----------
        1. **Setup**: Create a uniform intensity map (10 x 5) and define
           a scalar loss function that applies momentum broadening with
           a given dk value and returns the sum of all intensities.
        2. **Differentiate**: Call ``jax.grad(loss)(0.1)`` to compute
           the gradient of the loss with respect to dk.
        3. **Check finiteness**: Assert the gradient is finite.

        Asserts
        -------
        The gradient w.r.t. the momentum broadening width dk is finite,
        confirming that ``apply_momentum_broadening`` is differentiable
        through JAX. This is required for inverse fitting where dk is
        treated as a learnable instrument parameter.
        """
        K, E = 10, 5
        intensity = jnp.ones((K, E))
        k_distances = jnp.linspace(0, 1, K)

        def loss(dk):
            return jnp.sum(
                apply_momentum_broadening(intensity, k_distances, dk)
            )

        grad = jax.grad(loss)(jnp.array(0.1))
        assert jnp.isfinite(grad)
