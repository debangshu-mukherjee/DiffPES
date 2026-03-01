"""Tests for radial wavefunction models.

Extended Summary
----------------
Validates the ``slater_radial`` and ``hydrogenic_radial`` wavefunction
constructors.  Slater tests verify normalization (integral |R|^2 r^2 dr = 1)
and autodiff gradient accuracy against finite differences.  Hydrogenic
tests compare the R_{10} (1s) and R_{21} (2p) radial functions against
known analytical expressions and verify the boundary condition R_{2p}(0) = 0.

Routine Listings
----------------
:class:`TestSlaterRadial`
    Tests for slater_radial.
:class:`TestHydrogenicRadial`
    Tests for hydrogenic_radial.
"""

import chex
import jax
import jax.numpy as jnp

from arpyes.radial import hydrogenic_radial, slater_radial


class TestSlaterRadial(chex.TestCase):
    """Validate Slater radial normalization and autodiff gradients.

    Tests the Slater-type orbital R(r) = N * r^{n-1} * exp(-zeta*r)
    for correct normalization (integral |R|^2 r^2 dr = 1) and verify
    that ``jax.grad`` of a sum-of-values objective with respect to the
    Slater exponent zeta agrees with central finite differences.
    """

    @chex.variants(with_jit=True, without_jit=True)
    def test_normalization(self):
        """Verify the Slater 2s orbital is normalized to unity.

        Constructs R(r) for n=2, zeta=1.3 on a 20000-point grid up to
        r=30 Bohr and numerically integrates |R(r)|^2 * r^2 using the
        trapezoidal rule.  Asserts the integral is within 2e-3 of 1.0.
        The dense grid and large cutoff ensure the exponential tail
        contributes negligibly.  Run under both JIT and eager modes.
        """
        r = jnp.linspace(0.0, 30.0, 20000, dtype=jnp.float64)
        fn = self.variant(lambda radius: slater_radial(radius, n=2, zeta=1.3))
        radial = fn(r)
        norm = jnp.trapezoid((radial**2) * (r**2), x=r)
        chex.assert_trees_all_close(norm, jnp.asarray(1.0), atol=2.0e-3)

    def test_gradient_wrt_zeta_matches_finite_difference(self):
        """Verify autodiff gradient of Slater sum w.r.t. zeta matches FD.

        Defines a scalar objective = sum(R(r; zeta)) for n=2, zeta=1.15
        on a 500-point grid up to r=8 Bohr.  Differentiates with
        ``jax.grad`` and compares against a central finite-difference
        estimate with step eps=1e-4.  Asserts agreement to within 2e-4
        (atol and rtol), confirming the normalization constant, power-law
        prefactor, and exponential are all smoothly differentiable.
        """
        r = jnp.linspace(0.0, 8.0, 500, dtype=jnp.float64)
        zeta0 = jnp.asarray(1.15, dtype=jnp.float64)
        eps = jnp.asarray(1.0e-4, dtype=jnp.float64)

        def objective(zeta: chex.Numeric) -> chex.Array:
            return jnp.sum(slater_radial(r, n=2, zeta=jnp.asarray(zeta)))

        grad_auto = jax.grad(objective)(zeta0)
        fd = (objective(zeta0 + eps) - objective(zeta0 - eps)) / (2.0 * eps)
        chex.assert_trees_all_close(grad_auto, fd, atol=2.0e-4, rtol=2.0e-4)


class TestHydrogenicRadial(chex.TestCase):
    """Validate hydrogenic radial wavefunctions against analytical expressions.

    Tests the ``hydrogenic_radial`` function for the hydrogen atom
    (Z_eff=1) against closed-form R_{nl}(r) expressions.  Covers the
    1s ground state (R_{10} = 2*exp(-r)) and the 2p boundary condition
    (R_{21}(0) = 0, since all l > 0 radial functions vanish at the origin).
    """

    @chex.variants(with_jit=True, without_jit=True)
    def test_1s_matches_analytic_expression(self):
        """Verify R_{10}(r) = 2*exp(-r) for the hydrogen 1s orbital.

        Evaluates the hydrogenic radial function for n=1, l=0, Z_eff=1
        at r = [0.0, 0.3, 1.0, 2.5] Bohr and compares against the
        analytical expression R_{10}(r) = 2*exp(-r) in atomic units.
        Asserts element-wise agreement to within 1e-10.  The r=0 point
        tests the boundary condition R_{10}(0) = 2, and the larger r
        values test the exponential decay.
        """
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
        """Verify the 2p (n=2, l=1) radial function vanishes at r=0.

        Evaluates R_{21}(r=0) for Z_eff=1.  All hydrogenic radial
        functions with l > 0 contain a factor r^l and therefore must
        vanish at the origin.  Asserts the output is zero to within
        1e-12, testing this critical boundary condition / edge case.
        """
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
