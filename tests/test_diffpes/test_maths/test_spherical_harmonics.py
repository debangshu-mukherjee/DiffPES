"""Validate real spherical harmonics.

Extended Summary
----------------
The tests validate the real spherical harmonic functions
``real_spherical_harmonic`` and ``real_spherical_harmonics_all``.
They cover known analytical values for low-order harmonics with the
Condon-Shortley phase convention. They verify numerical orthonormality,
JIT compatibility, and autodiff gradients for ``theta`` and ``phi``.
They also verify input validation, batch shape, and consistency between
the single and batch interfaces.

"""

import math

import jax
import jax.numpy as jnp
import pytest
from beartype.typing import Any, Callable
from jaxtyping import Array, Float64

from diffpes.maths import (
    real_spherical_harmonic,
    real_spherical_harmonics_all,
)


class TestRealSphericalHarmonic:
    """Validate ``real_spherical_harmonic``.

    The tests validate the scalar real spherical harmonic with the
    Condon-Shortley phase convention. They compare ``l=0`` and ``l=1``
    harmonics with closed-form expressions. They verify orthonormality with
    numerical quadrature. They also check JIT, autodiff, and input validation.

    :see: :func:`~diffpes.maths.real_spherical_harmonic`
    """

    def test_y00_constant(self) -> None:
        """Verify Y_0^0 = 1/(2*sqrt(pi)) is constant over the sphere.

        The test evaluates Y_0^0 at four different (theta, phi) pairs using
        ``jax.vmap``.  The l=0 harmonic is the unique isotropic solution
        with the well-known value 1/(2*sqrt(pi)).  Asserts all four
        outputs match this constant to within 1e-12.

        Notes
        -----
        The test builds the documented inputs.
        It checks the stated property with explicit assertions.
        """
        expected: float
        theta: Float64[Array, " 4"]
        phi: Float64[Array, " 4"]
        vals: Float64[Array, " 4"]

        expected = 1.0 / (2.0 * math.sqrt(math.pi))
        theta = jnp.array([0.0, 0.5, 1.0, 2.0])
        phi = jnp.array([0.0, 0.3, 1.5, 3.0])
        vals = jax.vmap(lambda t, p: real_spherical_harmonic(0, 0, t, p))(
            theta, phi
        )
        assert jnp.allclose(vals, expected, atol=1e-12)

    def test_y10_cosine(self) -> None:
        """Verify Y_1^0 = sqrt(3/(4*pi)) * cos(theta) along the meridian.

        The test evaluates Y_1^0 at 50 theta values from 0 to pi with phi=0
        using ``jax.vmap``.  Compares against the analytical expression
        sqrt(3/(4*pi)) * cos(theta).  Asserts agreement to within 1e-10,
        validating the associated Legendre polynomial P_1^0 = cos(theta)
        and the normalization constant.

        Notes
        -----
        The test builds the documented inputs.
        It checks the stated property with explicit assertions.
        """
        theta: Float64[Array, " 50"]
        phi: Float64[Array, " 50"]
        vals: Float64[Array, " 50"]
        expected: Float64[Array, " 50"]

        theta = jnp.linspace(0.0, jnp.pi, 50)
        phi = jnp.zeros(50)
        vals = jax.vmap(lambda t, p: real_spherical_harmonic(1, 0, t, p))(
            theta, phi
        )
        expected = jnp.sqrt(3.0 / (4.0 * jnp.pi)) * jnp.cos(theta)
        assert jnp.allclose(vals, expected, atol=1e-10)

    def test_y11_sin_cos(self) -> None:
        """Verify Y_1^1 = +sqrt(3/(4*pi)) * sin(theta) * cos(phi).

        The test evaluates Y_1^{+1} at theta=pi/4, phi=0 and compares against
        the package real-harmonic convention expression. The explicit
        real-basis phase cancels the complex Condon--Shortley sign, so this is
        the positive p_x orbital fixed by the real-harmonic convention. The
        values agree within 1e-10.

        Notes
        -----
        The test builds the documented inputs.
        It checks the stated property with explicit assertions.
        """
        theta: Float64[Array, ""]
        phi: Float64[Array, ""]
        val: Float64[Array, ""]
        expected: float

        theta = jnp.array(jnp.pi / 4)
        phi = jnp.array(0.0)
        val = real_spherical_harmonic(1, 1, theta, phi)

        expected = (
            math.sqrt(3.0 / (4.0 * math.pi))
            * math.sin(math.pi / 4)
            * math.cos(0.0)
        )
        assert abs(float(val) - expected) < 1e-10

    def test_y1m1_sin_sin(self) -> None:
        """Verify Y_1^{-1} = +sqrt(3/(4*pi)) * sin(theta) * sin(phi).

        The test evaluates Y_1^{-1} at theta=pi/3, phi=pi/4 and compares
        against the analytical expression. For m < 0, the real harmonic
        uses ``sin(|m|*phi)`` and the Condon-Shortley phase cancels, yielding
        a positive prefactor.  Asserts agreement to within 1e-10.

        Notes
        -----
        The test builds the documented inputs.
        It checks the stated property with explicit assertions.
        """
        theta: Float64[Array, ""]
        phi: Float64[Array, ""]
        val: Float64[Array, ""]
        expected: float

        theta = jnp.array(jnp.pi / 3)
        phi = jnp.array(jnp.pi / 4)
        val = real_spherical_harmonic(1, -1, theta, phi)

        expected = (
            math.sqrt(3.0 / (4.0 * math.pi))
            * math.sin(math.pi / 3)
            * math.sin(math.pi / 4)
        )
        assert abs(float(val) - expected) < 1e-10

    def test_orthonormality_low_l(self) -> None:
        """Verify orthonormality of Y_0^0 and Y_1^0 via numerical quadrature.

        The test constructs a (100 x 200) grid in (theta, phi) using midpoint
        quadrature in cos(theta) and uniform spacing in phi.  Computes
        the cross-overlap integral Y_0^0 * Y_1^0 * sin(theta) and the
        self-overlap integral ``|Y_0^0|^2 * sin(theta)``.  Asserts the
        cross-overlap is < 0.01 (orthogonality) and the self-overlap
        is within 0.01 of 1.0 (normalization).  The moderate grid
        density gives ~1% numerical accuracy, sufficient for a smoke
        test of the spherical harmonic implementation.

        Notes
        -----
        The test builds the documented inputs.
        It checks the stated property with explicit assertions.
        """
        n_theta: int
        n_phi: int
        x: Float64[Array, " n_theta"]
        w: Float64[Array, " n_theta"]
        theta_grid: Float64[Array, " n_theta"]
        phi_grid: Float64[Array, " n_phi"]
        dphi: float
        y00: Float64[Array, "n_theta n_phi"]
        y10: Float64[Array, "n_theta n_phi"]
        sin_theta: Float64[Array, " n_theta"]
        integrand: Float64[Array, "n_theta n_phi"]
        integral: Float64[Array, ""]
        integrand_self: Float64[Array, "n_theta n_phi"]
        integral_self: Float64[Array, ""]

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
                    ],
                    strict=True,
                )
            )
        )
        theta_grid = jnp.arccos(x)
        phi_grid = jnp.linspace(0, 2 * jnp.pi, n_phi, endpoint=False)
        dphi = 2 * jnp.pi / n_phi

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
        integrand = y00 * y10
        integral = jnp.sum(integrand * sin_theta[:, None] * w[:, None] * dphi)
        assert abs(float(integral)) < 0.01, (
            f"Orthogonality failed: {float(integral)}"
        )

        integrand_self = y00 * y00
        integral_self = jnp.sum(
            integrand_self * sin_theta[:, None] * w[:, None] * dphi
        )
        assert abs(float(integral_self) - 1.0) < 0.01, (
            f"Normalization failed: {float(integral_self)}"
        )

    def test_jit_compatible(self) -> None:
        """Verify ``real_spherical_harmonic`` is JAX-JIT-compatible.

        The test wraps a call to Y_2^1(theta, phi) in ``jax.jit`` and evaluates
        at theta=1.0, phi=0.5.  Asserts the output is finite, confirming
        no Python-side operations (e.g., if-branches on traced values)
        break JAX tracing for the l=2 case.

        Notes
        -----
        The test builds the documented inputs.
        It checks the stated property with explicit assertions.
        """
        f: Callable[..., Any]
        val: Float64[Array, ""]

        f = jax.jit(lambda t, p: real_spherical_harmonic(2, 1, t, p))
        val = f(jnp.array(1.0), jnp.array(0.5))
        assert jnp.isfinite(val)

    def test_gradient_theta(self) -> None:
        """Verify autodiff gradient of Y_1^0 w.r.t. theta is finite.

        The test differentiates Y_1^0(theta, phi=0) at theta=pi/4 using
        ``jax.grad``.  The analytical derivative is
        -sqrt(3/(4*pi)) * sin(theta), which is nonzero at pi/4.  Asserts
        finiteness, confirming the associated Legendre polynomial
        implementation supports reverse-mode AD through theta.

        Notes
        -----
        The test builds the documented inputs.
        It checks the stated property with explicit assertions.
        """
        grad_fn: Callable[..., Any]
        g: Float64[Array, ""]

        grad_fn = jax.grad(
            lambda t: real_spherical_harmonic(1, 0, t, jnp.array(0.0))
        )
        g = grad_fn(jnp.array(jnp.pi / 4))
        assert jnp.isfinite(g)

    def test_gradient_phi(self) -> None:
        """Verify autodiff gradient of Y_1^1 w.r.t. phi is finite.

        The test differentiates Y_1^1(theta=pi/4, phi) at phi=0.5 using
        ``jax.grad``.  The m=1 harmonic has a cos(phi) dependence, so
        the derivative involves -sin(phi) and is nonzero at phi=0.5.
        The test asserts finiteness, confirming the azimuthal (phi) branch of
        the implementation supports AD.

        Notes
        -----
        The test builds the documented inputs.
        It checks the stated property with explicit assertions.
        """
        grad_fn: Callable[..., Any]
        g: Float64[Array, ""]

        grad_fn = jax.grad(
            lambda p: real_spherical_harmonic(1, 1, jnp.array(jnp.pi / 4), p)
        )
        g = grad_fn(jnp.array(0.5))
        assert jnp.isfinite(g)

    def test_invalid_l_raises(self) -> None:
        """Verify that negative l raises ``ValueError``.

        The test calls ``real_spherical_harmonic(-1, 0, 0, 0)`` and expects a
        ``ValueError`` with a matching message. This tests the guard for the
        boundary condition l >= 0.

        Notes
        -----
        The test builds the documented inputs.
        It checks the stated property with explicit assertions.
        """
        with pytest.raises(ValueError, match="l must be non-negative"):
            real_spherical_harmonic(-1, 0, jnp.array(0.0), jnp.array(0.0))

    def test_invalid_m_raises(self) -> None:
        """Verify that ``|m| > l`` raises ``ValueError``.

        The test calls ``real_spherical_harmonic(1, 2, 0, 0)`` with ``m > l``.
        It expects a ``ValueError`` with a matching message. This tests the
        constraint ``|m| <= l``.

        Notes
        -----
        The test builds the documented inputs.
        It checks the stated property with explicit assertions.
        """
        with pytest.raises(ValueError, match="must be <= l"):
            real_spherical_harmonic(1, 2, jnp.array(0.0), jnp.array(0.0))


class TestRealSphericalHarmonicsAll:
    """Validate ``real_spherical_harmonics_all``.

    The batch function returns all real spherical harmonics through ``l_max``
    at one ``(theta, phi)`` point. The tests verify the output shape and
    consistency with individual ``real_spherical_harmonic`` calls.

    :see: :func:`~diffpes.maths.real_spherical_harmonics_all`
    """

    def test_output_shape(self) -> None:
        """Verify output has (l_max+1)^2 entries.

        The test calls ``real_spherical_harmonics_all`` at one point.
        It expects shape ``(9,)`` because nine harmonics span ``l=0, 1, 2``.

        Notes
        -----
        The test builds the documented inputs.
        It checks the stated property with explicit assertions.
        """
        theta: Float64[Array, ""]
        phi: Float64[Array, ""]
        vals: Float64[Array, " 9"]

        theta = jnp.array(0.5)
        phi = jnp.array(0.3)
        vals = real_spherical_harmonics_all(2, theta, phi)
        assert vals.shape == (9,)

    def test_matches_individual(self) -> None:
        """Verify batch results match individual harmonic calls.

        The test evaluates both interfaces at ``theta=1.2`` and ``phi=0.7``.
        It compares each batch element with the corresponding scalar call.
        It iterates through ``l`` and ``m`` in the documented order.
        The values agree within ``1e-12`` and confirm the batch ordering.

        Notes
        -----
        The test builds the documented inputs.
        It checks the stated property with explicit assertions.
        """
        l: int
        m: int

        theta: Float64[Array, ""]
        phi: Float64[Array, ""]
        vals_all: Float64[Array, " 9"]
        idx: int
        val_single: Float64[Array, ""]

        theta = jnp.array(1.2)
        phi = jnp.array(0.7)
        vals_all = real_spherical_harmonics_all(2, theta, phi)

        idx = 0
        for l in range(3):
            for m in range(-l, l + 1):
                val_single = real_spherical_harmonic(l, m, theta, phi)
                assert jnp.allclose(vals_all[idx], val_single, atol=1e-12), (
                    f"Mismatch at l={l}, m={m}: "
                    f"{vals_all[idx]} vs {val_single}"
                )
                idx += 1
