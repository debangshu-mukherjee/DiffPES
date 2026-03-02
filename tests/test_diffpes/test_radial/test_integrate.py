"""Tests for radial integrals.

Extended Summary
----------------
Validates the ``radial_integral`` function that computes the overlap
integral of a Slater radial wavefunction with a spherical Bessel
function j_{l'}(kr).  Tests compare the l'=0 numerical integral against
a known analytical Fourier-transform result for Slater 1s orbitals, and
verify the autodiff gradient with respect to the Slater exponent zeta
against a central finite-difference estimate.

Routine Listings
----------------
:class:`TestRadialIntegral`
    Tests for radial_integral.
"""

import chex
import jax
import jax.numpy as jnp

from diffpes.radial import radial_integral, slater_radial


class TestRadialIntegral(chex.TestCase):
    """Validate radial-integral values and derivatives.

    Tests the numerical radial overlap integral
    I(k) = integral_0^inf R(r) * j_{l'}(kr) * r^2 dr
    against analytical results for Slater-type orbitals and verifies
    that JAX autodiff gradients with respect to the Slater exponent
    zeta agree with finite-difference estimates.
    """

    @chex.variants(with_jit=True, without_jit=True)
    def test_l0_slater_matches_analytic_integral(self):
        """Verify the l'=0 radial integral matches the analytical Fourier transform.

        For a Slater 1s orbital R(r) = N * r^0 * exp(-zeta*r) with
        zeta=1.2, the j_0(kr) = sin(kr)/(kr) overlap integral has the
        closed-form result 2*N*(3*zeta^2 - k^2)/(zeta^2 + k^2)^3.
        Uses a dense 25000-point grid up to r=50 Bohr and test k-values
        [0.2, 0.8, 1.4].  Asserts the real part of the numerical integral
        agrees with the analytical expression to within 5e-3 in both
        absolute and relative tolerance.  Run under both JIT and eager
        modes via ``chex.variants``.
        """
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
        """Verify autodiff gradient w.r.t. zeta matches finite differences.

        Defines a scalar objective that computes the real part of the
        l'=0 radial integral for a Slater 1s orbital at k=0.9,
        zeta=1.1.  Differentiates with ``jax.grad`` and compares against
        a central finite-difference estimate with step eps=5e-4.  Uses
        an 18000-point grid up to r=45 Bohr.  Asserts agreement to
        within 5e-3 (atol and rtol), confirming that the numerical
        integration (trapezoid rule), Slater radial construction, and
        Bessel evaluation are all smoothly differentiable end-to-end.
        """
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


class TestRadialIntegrateErrors:
    """Tests for invalid input handling in radial_integral.

    Validates that ``radial_integral`` raises ``ValueError`` for
    negative ``l_prime`` values.
    """

    def test_negative_l_prime_raises(self):
        """Verify that l_prime < 0 raises ValueError.

        Calls ``radial_integral`` with ``l_prime=-1`` and asserts a
        ``ValueError`` matching "non-negative" is raised, covering the
        guard at the top of the function.
        """
        import jax.numpy as jnp
        import pytest

        from diffpes.radial import radial_integral, slater_radial

        r = jnp.linspace(0.0, 10.0, 100, dtype=jnp.float64)
        radial = slater_radial(r, n=1, zeta=1.0)
        k = jnp.array([1.0], dtype=jnp.float64)
        with pytest.raises(ValueError, match="non-negative"):
            radial_integral(k, r, radial, l_prime=-1)
