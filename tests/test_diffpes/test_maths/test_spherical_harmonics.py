"""Tests for real spherical harmonics.

Extended Summary
----------------
Validates the real spherical harmonic functions
``real_spherical_harmonic`` and ``real_spherical_harmonics_all``.
Tests cover known analytical values for low-order harmonics (Y_0^0,
Y_1^0, Y_1^{+1}, Y_1^{-1}) using the Condon-Shortley phase convention,
numerical orthonormality via quadrature, JIT compatibility, autodiff
gradients with respect to theta and phi, input validation (negative l,
|m| > l), output shape of the batch function, and consistency between
the single and batch interfaces.

Routine Listings
----------------
:class:`TestRealSphericalHarmonics`
    Tests for real_spherical_harmonic.
:class:`TestRealSphericalHarmonicsAll`
    Tests for real_spherical_harmonics_all.
"""

import math

import jax
import jax.numpy as jnp
import pytest

from diffpes.maths.spherical_harmonics import (
    real_spherical_harmonic,
    real_spherical_harmonics_all,
)


class TestRealSphericalHarmonics:
    """Tests for ``real_spherical_harmonic``.

    Validates the scalar real spherical harmonic Y_l^m(theta, phi) using
    the Condon-Shortley phase convention.  Tests compare against closed-
    form expressions for l=0 and l=1 harmonics, verify orthonormality
    via numerical quadrature, check JIT and autodiff support, and confirm
    proper input validation for invalid quantum numbers.
    """

    def test_y00_constant(self):
        """Verify Y_0^0 = 1/(2*sqrt(pi)) is constant over the sphere.

        Evaluates Y_0^0 at four different (theta, phi) pairs using
        ``jax.vmap``.  The l=0 harmonic is the unique isotropic solution
        with the well-known value 1/(2*sqrt(pi)).  Asserts all four
        outputs match this constant to within 1e-12.
        """
        expected = 1.0 / (2.0 * math.sqrt(math.pi))
        theta = jnp.array([0.0, 0.5, 1.0, 2.0])
        phi = jnp.array([0.0, 0.3, 1.5, 3.0])
        vals = jax.vmap(lambda t, p: real_spherical_harmonic(0, 0, t, p))(
            theta, phi
        )
        assert jnp.allclose(vals, expected, atol=1e-12)

    def test_y10_cosine(self):
        """Verify Y_1^0 = sqrt(3/(4*pi)) * cos(theta) along the meridian.

        Evaluates Y_1^0 at 50 theta values from 0 to pi with phi=0
        using ``jax.vmap``.  Compares against the analytical expression
        sqrt(3/(4*pi)) * cos(theta).  Asserts agreement to within 1e-10,
        validating the associated Legendre polynomial P_1^0 = cos(theta)
        and the normalization constant.
        """
        theta = jnp.linspace(0.0, jnp.pi, 50)
        phi = jnp.zeros(50)
        vals = jax.vmap(lambda t, p: real_spherical_harmonic(1, 0, t, p))(
            theta, phi
        )
        expected = jnp.sqrt(3.0 / (4.0 * jnp.pi)) * jnp.cos(theta)
        assert jnp.allclose(vals, expected, atol=1e-10)

    def test_y11_sin_cos(self):
        """Verify Y_1^1 = -sqrt(3/(4*pi)) * sin(theta) * cos(phi).

        Evaluates Y_1^{+1} at theta=pi/4, phi=0 and compares against the
        Condon-Shortley convention expression.  The negative sign comes
        from the (-1)^m phase factor for m > 0.  Asserts agreement to
        within 1e-10.
        """
        theta = jnp.array(jnp.pi / 4)
        phi = jnp.array(0.0)
        val = real_spherical_harmonic(1, 1, theta, phi)
        # With CS phase: Y_1^1 = -sqrt(3/(4pi)) sin(theta) cos(phi)
        expected = (
            -math.sqrt(3.0 / (4.0 * math.pi))
            * math.sin(math.pi / 4)
            * math.cos(0.0)
        )
        assert abs(float(val) - expected) < 1e-10

    def test_y1m1_sin_sin(self):
        """Verify Y_1^{-1} = +sqrt(3/(4*pi)) * sin(theta) * sin(phi).

        Evaluates Y_1^{-1} at theta=pi/3, phi=pi/4 and compares against
        the analytical expression.  For m < 0, the real spherical harmonic
        uses sin(|m|*phi) and the Condon-Shortley phase cancels, yielding
        a positive prefactor.  Asserts agreement to within 1e-10.
        """
        theta = jnp.array(jnp.pi / 3)
        phi = jnp.array(jnp.pi / 4)
        val = real_spherical_harmonic(1, -1, theta, phi)
        # After CS phase cancellation: Y_1^{-1} = +sqrt(3/(4pi)) sin(theta) sin(phi)
        expected = (
            math.sqrt(3.0 / (4.0 * math.pi))
            * math.sin(math.pi / 3)
            * math.sin(math.pi / 4)
        )
        assert abs(float(val) - expected) < 1e-10

    def test_orthonormality_low_l(self):
        """Verify orthonormality of Y_0^0 and Y_1^0 via numerical quadrature.

        Constructs a (100 x 200) grid in (theta, phi) using midpoint
        quadrature in cos(theta) and uniform spacing in phi.  Computes
        the cross-overlap integral Y_0^0 * Y_1^0 * sin(theta) and the
        self-overlap integral |Y_0^0|^2 * sin(theta).  Asserts the
        cross-overlap is < 0.01 (orthogonality) and the self-overlap
        is within 0.01 of 1.0 (normalization).  The moderate grid
        density gives ~1% numerical accuracy, sufficient for a smoke
        test of the spherical harmonic implementation.
        """
        # Use Gauss-Legendre quadrature for theta, uniform for phi
        n_theta = 100
        n_phi = 200
        x, w = jnp.array(
            list(
                zip(
                    *[
                        (
                            math.cos(math.pi * (i + 0.5) / n_theta),
                            math.pi / n_theta,
                        )
                        for i in range(n_theta)
                    ]
                )
            )
        )
        theta_grid = jnp.arccos(x)
        phi_grid = jnp.linspace(0, 2 * jnp.pi, n_phi, endpoint=False)
        dphi = 2 * jnp.pi / n_phi

        # Test l=0,m=0 vs l=1,m=0
        y00 = jax.vmap(
            lambda t: jax.vmap(lambda p: real_spherical_harmonic(0, 0, t, p))(
                phi_grid
            )
        )(theta_grid)
        y10 = jax.vmap(
            lambda t: jax.vmap(lambda p: real_spherical_harmonic(1, 0, t, p))(
                phi_grid
            )
        )(theta_grid)

        sin_theta = jnp.sin(theta_grid)
        integrand = y00 * y10  # shape (n_theta, n_phi)
        integral = jnp.sum(integrand * sin_theta[:, None] * w[:, None] * dphi)
        assert (
            abs(float(integral)) < 0.01
        ), f"Orthogonality failed: {float(integral)}"

        # Self-overlap of Y_0^0
        integrand_self = y00 * y00
        integral_self = jnp.sum(
            integrand_self * sin_theta[:, None] * w[:, None] * dphi
        )
        assert (
            abs(float(integral_self) - 1.0) < 0.01
        ), f"Normalization failed: {float(integral_self)}"

    def test_jit_compatible(self):
        """Verify ``real_spherical_harmonic`` is JAX-JIT-compatible.

        Wraps a call to Y_2^1(theta, phi) in ``jax.jit`` and evaluates
        at theta=1.0, phi=0.5.  Asserts the output is finite, confirming
        no Python-side operations (e.g., if-branches on traced values)
        break JAX tracing for the l=2 case.
        """
        f = jax.jit(lambda t, p: real_spherical_harmonic(2, 1, t, p))
        val = f(jnp.array(1.0), jnp.array(0.5))
        assert jnp.isfinite(val)

    def test_gradient_theta(self):
        """Verify autodiff gradient of Y_1^0 w.r.t. theta is finite.

        Differentiates Y_1^0(theta, phi=0) at theta=pi/4 using
        ``jax.grad``.  The analytical derivative is
        -sqrt(3/(4*pi)) * sin(theta), which is nonzero at pi/4.  Asserts
        finiteness, confirming the associated Legendre polynomial
        implementation supports reverse-mode AD through theta.
        """
        grad_fn = jax.grad(
            lambda t: real_spherical_harmonic(1, 0, t, jnp.array(0.0))
        )
        g = grad_fn(jnp.array(jnp.pi / 4))
        assert jnp.isfinite(g)

    def test_gradient_phi(self):
        """Verify autodiff gradient of Y_1^1 w.r.t. phi is finite.

        Differentiates Y_1^1(theta=pi/4, phi) at phi=0.5 using
        ``jax.grad``.  The m=1 harmonic has a cos(phi) dependence, so
        the derivative involves -sin(phi) and is nonzero at phi=0.5.
        Asserts finiteness, confirming the azimuthal (phi) branch of
        the implementation supports AD.
        """
        grad_fn = jax.grad(
            lambda p: real_spherical_harmonic(1, 1, jnp.array(jnp.pi / 4), p)
        )
        g = grad_fn(jnp.array(0.5))
        assert jnp.isfinite(g)

    def test_invalid_l_raises(self):
        """Verify that negative l raises ``ValueError``.

        Calls ``real_spherical_harmonic(-1, 0, 0, 0)`` and asserts a
        ``ValueError`` is raised with a message matching "l must be
        non-negative".  This tests the input-validation guard for the
        boundary condition l >= 0.
        """
        with pytest.raises(ValueError, match="l must be non-negative"):
            real_spherical_harmonic(-1, 0, jnp.array(0.0), jnp.array(0.0))

    def test_invalid_m_raises(self):
        """Verify that |m| > l raises ``ValueError``.

        Calls ``real_spherical_harmonic(1, 2, 0, 0)`` where m=2 > l=1
        and asserts a ``ValueError`` is raised with a message matching
        "must be <= l".  This tests the input-validation guard for the
        constraint |m| <= l.
        """
        with pytest.raises(ValueError, match="must be <= l"):
            real_spherical_harmonic(1, 2, jnp.array(0.0), jnp.array(0.0))


class TestRealSphericalHarmonicsAll:
    """Tests for ``real_spherical_harmonics_all``.

    Validates the batch function that returns all (l_max+1)^2 real
    spherical harmonics up to a given l_max at a single (theta, phi)
    point.  Tests verify the output shape and consistency with
    individual ``real_spherical_harmonic`` calls.
    """

    def test_output_shape(self):
        """Verify output has (l_max+1)^2 entries.

        Calls ``real_spherical_harmonics_all(2, theta, phi)`` at a single
        point and asserts the output shape is ``(9,)`` since
        (2+1)^2 = 9 harmonics span l=0, 1, 2.
        """
        theta = jnp.array(0.5)
        phi = jnp.array(0.3)
        vals = real_spherical_harmonics_all(2, theta, phi)
        assert vals.shape == (9,)  # (2+1)^2 = 9

    def test_matches_individual(self):
        """Verify batch results match individual ``real_spherical_harmonic`` calls.

        Evaluates ``real_spherical_harmonics_all(2, theta, phi)`` at
        theta=1.2, phi=0.7 and compares each element against the
        corresponding ``real_spherical_harmonic(l, m, theta, phi)`` call,
        iterating over l in [0..2] and m in [-l..l].  Asserts element-wise
        agreement to within 1e-12, confirming the batch ordering convention
        (l-major, m-minor from -l to +l).
        """
        theta = jnp.array(1.2)
        phi = jnp.array(0.7)
        vals_all = real_spherical_harmonics_all(2, theta, phi)

        idx = 0
        for l in range(3):
            for m in range(-l, l + 1):
                val_single = real_spherical_harmonic(l, m, theta, phi)
                assert jnp.allclose(
                    vals_all[idx], val_single, atol=1e-12
                ), f"Mismatch at l={l}, m={m}: {vals_all[idx]} vs {val_single}"
                idx += 1
