"""Tests for radial integrals."""

import chex
import jax
import jax.numpy as jnp

from arpyes.radial import radial_integral, slater_radial


class TestRadialIntegral(chex.TestCase):
    """Validate radial-integral values and derivatives."""

    @chex.variants(with_jit=True, without_jit=True)
    def test_l0_slater_matches_analytic_integral(self):
        """Compare numerical l'=0 integral with analytical expression."""
        zeta = 1.2
        r = jnp.linspace(0.0, 50.0, 25000, dtype=jnp.float64)
        radial = slater_radial(r, n=1, zeta=zeta)
        k = jnp.array([0.2, 0.8, 1.4], dtype=jnp.float64)
        fn = self.variant(
            lambda kvals: radial_integral(kvals, r, radial, l_prime=0)
        )

        numeric = jnp.real(fn(k))
        norm = ((2.0 * zeta) ** 1.5) / jnp.sqrt(2.0)
        expected = (
            2.0 * norm * (3.0 * zeta**2 - k**2) / ((zeta**2 + k**2) ** 3)
        )
        chex.assert_trees_all_close(
            numeric, expected, atol=5.0e-3, rtol=5.0e-3
        )

    def test_gradient_wrt_zeta_matches_finite_difference(self):
        """Check gradient of radial integral with respect to Slater exponent."""
        r = jnp.linspace(0.0, 45.0, 18000, dtype=jnp.float64)
        k = jnp.asarray(0.9, dtype=jnp.float64)
        zeta0 = jnp.asarray(1.1, dtype=jnp.float64)
        eps = jnp.asarray(5.0e-4, dtype=jnp.float64)

        def objective(zeta: chex.Numeric) -> chex.Array:
            radial = slater_radial(r, n=1, zeta=jnp.asarray(zeta))
            value = radial_integral(k, r, radial, l_prime=0)
            return jnp.real(value)

        grad_auto = jax.grad(objective)(zeta0)
        fd = (objective(zeta0 + eps) - objective(zeta0 - eps)) / (2.0 * eps)
        chex.assert_trees_all_close(grad_auto, fd, atol=5.0e-3, rtol=5.0e-3)
