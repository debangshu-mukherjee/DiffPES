"""Tests for spherical Bessel functions."""

import chex
import jax
import jax.numpy as jnp

from arpyes.radial import spherical_bessel_jl


class TestSphericalBessel(chex.TestCase):
    """Validate low-order spherical Bessel behavior and derivatives."""

    @chex.variants(with_jit=True, without_jit=True)
    def test_j0_and_j1_match_closed_form(self):
        """Compare j0/j1 against analytical expressions."""
        x = jnp.array([0.2, 0.7, 1.5], dtype=jnp.float64)
        j0_fn = self.variant(lambda values: spherical_bessel_jl(0, values))
        j1_fn = self.variant(lambda values: spherical_bessel_jl(1, values))

        expected_j0 = jnp.sin(x) / x
        expected_j1 = jnp.sin(x) / (x * x) - jnp.cos(x) / x
        chex.assert_trees_all_close(j0_fn(x), expected_j0, atol=1.0e-10)
        chex.assert_trees_all_close(j1_fn(x), expected_j1, atol=1.0e-10)

    @chex.variants(with_jit=True, without_jit=True)
    def test_j2_matches_closed_form(self):
        """Compare j2 against analytical expression."""
        x = jnp.array([0.4, 1.1, 2.4], dtype=jnp.float64)
        fn = self.variant(lambda values: spherical_bessel_jl(2, values))
        expected = ((3.0 / (x**3)) - (1.0 / x)) * jnp.sin(x) - (
            3.0 / (x * x)
        ) * jnp.cos(x)
        chex.assert_trees_all_close(fn(x), expected, atol=1.0e-10)

    @chex.variants(with_jit=True, without_jit=True)
    def test_zero_argument_limits(self):
        """Check origin limits j0(0)=1 and j_l(0)=0 for l>0."""
        zero = jnp.array([0.0], dtype=jnp.float64)
        j0_fn = self.variant(lambda values: spherical_bessel_jl(0, values))
        j1_fn = self.variant(lambda values: spherical_bessel_jl(1, values))
        j3_fn = self.variant(lambda values: spherical_bessel_jl(3, values))
        chex.assert_trees_all_close(
            j0_fn(zero), jnp.array([1.0]), atol=1.0e-12
        )
        chex.assert_trees_all_close(
            j1_fn(zero), jnp.array([0.0]), atol=1.0e-12
        )
        chex.assert_trees_all_close(
            j3_fn(zero), jnp.array([0.0]), atol=1.0e-12
        )

    def test_j0_gradient_matches_analytic_derivative(self):
        """Validate grad(j0) against d/dx[sin(x)/x]."""
        x0 = jnp.asarray(1.3, dtype=jnp.float64)

        def objective(x: chex.Numeric) -> chex.Array:
            return spherical_bessel_jl(0, jnp.asarray(x))

        grad_fn = jax.grad(objective)
        grad_val = grad_fn(x0)
        expected_grad = (x0 * jnp.cos(x0) - jnp.sin(x0)) / (x0 * x0)
        chex.assert_trees_all_close(grad_val, expected_grad, atol=1.0e-10)
