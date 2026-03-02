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
import pytest

from diffpes.radial import hydrogenic_radial, slater_radial


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


class TestSlaterRadialErrors:
    """Tests for invalid input handling in slater_radial.

    Validates that ``slater_radial`` raises ``ValueError`` when the
    principal quantum number ``n`` is less than 1.
    """

    def test_n_zero_raises(self):
        """Verify that n=0 raises ValueError.

        Calls ``slater_radial`` with ``n=0`` and asserts a
        ``ValueError`` matching "n must be >= 1" is raised.
        """
        r = jnp.array([1.0], dtype=jnp.float64)
        with pytest.raises(ValueError, match="n must be >= 1"):
            slater_radial(r, n=0, zeta=1.0)


class TestHydrogenicRadialErrors:
    """Tests for invalid input handling in hydrogenic_radial.

    Validates that ``hydrogenic_radial`` raises ``ValueError`` for
    invalid principal quantum numbers (n < 1) and for angular momentum
    that violates 0 <= l < n.
    """

    def test_n_zero_raises(self):
        """Verify that n=0 raises ValueError for hydrogenic_radial.

        Calls ``hydrogenic_radial`` with ``n=0`` and asserts a
        ``ValueError`` matching "n must be >= 1" is raised.
        """
        r = jnp.array([1.0], dtype=jnp.float64)
        with pytest.raises(ValueError, match="n must be >= 1"):
            hydrogenic_radial(r, n=0, angular_momentum=0, z_eff=1.0)

    def test_angular_momentum_equals_n_raises(self):
        """Verify that angular_momentum >= n raises ValueError.

        Calls ``hydrogenic_radial`` with ``n=2, angular_momentum=2``
        which violates the constraint ``angular_momentum < n``, and
        asserts a ``ValueError`` matching "angular_momentum" is raised.
        """
        r = jnp.array([1.0], dtype=jnp.float64)
        with pytest.raises(ValueError, match="angular_momentum"):
            hydrogenic_radial(r, n=2, angular_momentum=2, z_eff=1.0)


class TestLaguerreRecurrence:
    """Tests for the Laguerre polynomial recurrence path.

    Exercises the ``order >= 2`` branch of ``_associated_laguerre``
    that uses ``jax.lax.fori_loop`` for the recurrence, and validates
    the error paths for negative order or alpha.
    """

    def test_negative_order_raises(self):
        """Verify that order < 0 raises ValueError in _associated_laguerre."""
        from diffpes.radial.wavefunctions import _associated_laguerre

        r = jnp.array([1.0], dtype=jnp.float64)
        with pytest.raises(ValueError, match="non-negative"):
            _associated_laguerre(-1, 0.5, r)

    def test_negative_alpha_raises(self):
        """Verify that alpha < 0 raises ValueError in _associated_laguerre."""
        from diffpes.radial.wavefunctions import _associated_laguerre

        r = jnp.array([1.0], dtype=jnp.float64)
        with pytest.raises(ValueError, match="non-negative"):
            _associated_laguerre(2, -0.1, r)

    def test_order_one_early_return(self):
        """Verify that order=1 takes the early-return branch.

        L_1^0(x) = 1 - x, so at x=0 the value is 1.0 and at x=1 it is 0.0.
        This exercises the ``if order == 1: return laguerre_one`` path.
        """
        from diffpes.radial.wavefunctions import _associated_laguerre

        x = jnp.array([0.0, 1.0], dtype=jnp.float64)
        result = _associated_laguerre(1, 0.0, x)
        expected = jnp.array([1.0, 0.0], dtype=jnp.float64)
        chex.assert_trees_all_close(result, expected, atol=1e-10)

    def test_order_two_uses_recurrence(self):
        """Verify that order=2 executes the fori_loop recurrence branch.

        The order=0 and order=1 branches return early; order >= 2
        uses the upward recurrence. Checks the known value
        L_2^0(0) = 1 - 0 + 0 = 1.
        """
        from diffpes.radial.wavefunctions import _associated_laguerre

        x = jnp.array([0.0], dtype=jnp.float64)
        # L_2^0(x) = 1 - x + x^2/2, so L_2^0(0) = 1
        result = _associated_laguerre(2, 0.0, x)
        chex.assert_trees_all_close(result, jnp.array([1.0]), atol=1e-10)
