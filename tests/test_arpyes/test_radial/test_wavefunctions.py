"""Tests for radial wavefunction models."""

import chex
import jax
import jax.numpy as jnp

from arpyes.radial import hydrogenic_radial, slater_radial


class TestSlaterRadial(chex.TestCase):
    """Validate Slater radial normalization and gradients."""

    @chex.variants(with_jit=True, without_jit=True)
    def test_normalization(self):
        """Ensure integral of |R|^2 r^2 is close to unity."""
        r = jnp.linspace(0.0, 30.0, 20000, dtype=jnp.float64)
        fn = self.variant(lambda radius: slater_radial(radius, n=2, zeta=1.3))
        radial = fn(r)
        norm = jnp.trapezoid((radial**2) * (r**2), x=r)
        chex.assert_trees_all_close(norm, jnp.asarray(1.0), atol=2.0e-3)

    def test_gradient_wrt_zeta_matches_finite_difference(self):
        """Check autodiff gradient of Slater radial sum against finite differences."""
        r = jnp.linspace(0.0, 8.0, 500, dtype=jnp.float64)
        zeta0 = jnp.asarray(1.15, dtype=jnp.float64)
        eps = jnp.asarray(1.0e-4, dtype=jnp.float64)

        def objective(zeta: chex.Numeric) -> chex.Array:
            return jnp.sum(slater_radial(r, n=2, zeta=jnp.asarray(zeta)))

        grad_auto = jax.grad(objective)(zeta0)
        fd = (objective(zeta0 + eps) - objective(zeta0 - eps)) / (2.0 * eps)
        chex.assert_trees_all_close(grad_auto, fd, atol=2.0e-4, rtol=2.0e-4)


class TestHydrogenicRadial(chex.TestCase):
    """Validate hydrogenic radial special cases and basic behavior."""

    @chex.variants(with_jit=True, without_jit=True)
    def test_1s_matches_analytic_expression(self):
        """Verify R_10(r) = 2 exp(-r) for Z_eff=1 in atomic units."""
        r = jnp.array([0.0, 0.3, 1.0, 2.5], dtype=jnp.float64)
        fn = self.variant(
            lambda radius: hydrogenic_radial(
                radius,
                n=1,
                angular_momentum=0,
                z_eff=1.0,
            )
        )
        expected = 2.0 * jnp.exp(-r)
        chex.assert_trees_all_close(fn(r), expected, atol=1.0e-10)

    @chex.variants(with_jit=True, without_jit=True)
    def test_2p_is_zero_at_origin(self):
        """Ensure the 2p radial function vanishes at r=0."""
        fn = self.variant(
            lambda radius: hydrogenic_radial(
                radius,
                n=2,
                angular_momentum=1,
                z_eff=1.0,
            )
        )
        value_at_origin = fn(jnp.asarray([0.0], dtype=jnp.float64))
        chex.assert_trees_all_close(
            value_at_origin,
            jnp.asarray([0.0], dtype=jnp.float64),
            atol=1.0e-12,
        )
