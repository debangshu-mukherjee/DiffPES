"""Tests for real spherical harmonics."""

import math

import jax
import jax.numpy as jnp
import pytest

from arpyes.maths.spherical_harmonics import (
    real_spherical_harmonic,
    real_spherical_harmonics_all,
)


class TestRealSphericalHarmonics:
    """Tests for real_spherical_harmonic."""

    def test_y00_constant(self):
        """Y_0^0 = 1/(2√π) everywhere."""
        expected = 1.0 / (2.0 * math.sqrt(math.pi))
        theta = jnp.array([0.0, 0.5, 1.0, 2.0])
        phi = jnp.array([0.0, 0.3, 1.5, 3.0])
        vals = jax.vmap(lambda t, p: real_spherical_harmonic(0, 0, t, p))(
            theta, phi
        )
        assert jnp.allclose(vals, expected, atol=1e-12)

    def test_y10_cosine(self):
        """Y_1^0 = √(3/(4π)) cos(θ)."""
        theta = jnp.linspace(0.0, jnp.pi, 50)
        phi = jnp.zeros(50)
        vals = jax.vmap(lambda t, p: real_spherical_harmonic(1, 0, t, p))(
            theta, phi
        )
        expected = jnp.sqrt(3.0 / (4.0 * jnp.pi)) * jnp.cos(theta)
        assert jnp.allclose(vals, expected, atol=1e-10)

    def test_y11_sin_cos(self):
        """Y_1^1 = -√(3/(4π)) sin(θ) cos(φ) (Condon-Shortley convention)."""
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
        """Y_1^{-1} = +√(3/(4π)) sin(θ) sin(φ) (CS phase cancelled for m<0)."""
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
        """Check orthonormality ∫ Y_l^m Y_{l'}^{m'} dΩ ≈ δ_{ll'} δ_{mm'}."""
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
        """Spherical harmonics can be JIT-compiled."""
        f = jax.jit(lambda t, p: real_spherical_harmonic(2, 1, t, p))
        val = f(jnp.array(1.0), jnp.array(0.5))
        assert jnp.isfinite(val)

    def test_gradient_theta(self):
        """Gradient with respect to theta is finite."""
        grad_fn = jax.grad(
            lambda t: real_spherical_harmonic(1, 0, t, jnp.array(0.0))
        )
        g = grad_fn(jnp.array(jnp.pi / 4))
        assert jnp.isfinite(g)

    def test_gradient_phi(self):
        """Gradient with respect to phi is finite."""
        grad_fn = jax.grad(
            lambda p: real_spherical_harmonic(1, 1, jnp.array(jnp.pi / 4), p)
        )
        g = grad_fn(jnp.array(0.5))
        assert jnp.isfinite(g)

    def test_invalid_l_raises(self):
        """Negative l raises ValueError."""
        with pytest.raises(ValueError, match="l must be non-negative"):
            real_spherical_harmonic(-1, 0, jnp.array(0.0), jnp.array(0.0))

    def test_invalid_m_raises(self):
        """|m| > l raises ValueError."""
        with pytest.raises(ValueError, match="must be <= l"):
            real_spherical_harmonic(1, 2, jnp.array(0.0), jnp.array(0.0))


class TestRealSphericalHarmonicsAll:
    """Tests for real_spherical_harmonics_all."""

    def test_output_shape(self):
        """Returns (l_max+1)^2 values."""
        theta = jnp.array(0.5)
        phi = jnp.array(0.3)
        vals = real_spherical_harmonics_all(2, theta, phi)
        assert vals.shape == (9,)  # (2+1)^2 = 9

    def test_matches_individual(self):
        """Results match individual calls."""
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
