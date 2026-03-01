"""Tests for dipole matrix element assembly.

Extended Summary
----------------
Validates the full dipole-matrix-element pipeline used in ARPES
photoemission intensity simulation: the single-orbital matrix element
``dipole_matrix_element_single``, the per-orbital intensity wrapper
``dipole_intensity_orbital``, and the multi-orbital batch function
``dipole_intensities_all_orbitals``.  Tests cover nonzero output for
allowed transitions, numerical finiteness, JIT compatibility,
non-negativity of intensities, consistency between |M|^2 and the
intensity function, output shapes for multi-orbital bases, and
differentiability with respect to both the electric-field polarization
vector and Slater orbital exponents (zeta).

Routine Listings
----------------
:class:`TestDipoleMatrixElementSingle`
    Tests for dipole_matrix_element_single.
:class:`TestDipoleIntensityOrbital`
    Tests for dipole_intensity_orbital.
:class:`TestDipoleIntensitiesAllOrbitals`
    Tests for dipole_intensities_all_orbitals.
"""

import jax
import jax.numpy as jnp
import pytest

from arpyes.maths.dipole import (
    dipole_intensities_all_orbitals,
    dipole_intensity_orbital,
    dipole_matrix_element_single,
)
from arpyes.radial import slater_radial
from arpyes.types import make_orbital_basis, make_slater_params


@pytest.fixture
def r_grid():
    """Provide a dense radial grid for numerical integration.

    Returns a 5000-point grid from 1e-6 to 50.0 Bohr.  The small but
    nonzero lower bound avoids the 1/r singularity in Slater radial
    functions while the upper bound is large enough for the exponential
    tails to decay below machine precision for typical Slater exponents
    (zeta ~ 1--2).
    """
    return jnp.linspace(1e-6, 50.0, 5000)


@pytest.fixture
def z_polarized():
    """Provide a z-polarized electric-field vector.

    Returns the complex-valued polarization vector (0, 0, 1) which
    selects the q=0 (Delta m = 0) component of the dipole operator,
    corresponding to linearly polarized light along the z-axis.
    """
    return jnp.array([0.0, 0.0, 1.0], dtype=jnp.complex128)


@pytest.fixture
def x_polarized():
    """Provide an x-polarized electric-field vector.

    Returns the complex-valued polarization vector (1, 0, 0) which
    mixes the q=+1 and q=-1 spherical components of the dipole
    operator, corresponding to linearly polarized light along the
    x-axis.
    """
    return jnp.array([1.0, 0.0, 0.0], dtype=jnp.complex128)


class TestDipoleMatrixElementSingle:
    """Tests for ``dipole_matrix_element_single``.

    Validates the core single-orbital dipole matrix element
    M = <k | e . r | n l m> which combines a radial integral with Gaunt
    angular coupling.  Tests verify that allowed transitions produce
    nonzero matrix elements, that outputs are always numerically finite,
    that the function is compatible with JAX JIT compilation, and that
    autodiff gradients with respect to the electric-field polarization
    vector are well-defined.
    """

    def test_s_orbital_nonzero(self, r_grid, z_polarized):
        """Verify the s -> p dipole transition is nonzero for z-polarization.

        Uses an n=1, zeta=1.0 Slater 1s orbital with z-polarized light and
        k along z.  The dipole selection rule allows l=0 -> l'=1 (Delta l = +1)
        with q=0, so the matrix element must be nonzero.  Asserts |M| > 1e-6
        to confirm the angular (Gaunt) and radial integrals both contribute.
        """
        R_vals = slater_radial(r_grid, 1, 1.0)
        k_vec = jnp.array([0.0, 0.0, 1.0])
        M = dipole_matrix_element_single(
            k_vec, r_grid, R_vals, 0, 0, z_polarized
        )
        assert (
            jnp.abs(M) > 1e-6
        ), f"Expected nonzero, got |M|={float(jnp.abs(M))}"

    def test_finite_output(self, r_grid, z_polarized):
        """Verify the matrix element is numerically finite for a p-orbital.

        Uses an n=2, zeta=2.0 Slater 2p (l=1, m=0) orbital with an
        off-axis k-vector (1.0, 0.5, 0.3) and z-polarization.  Asserts
        ``jnp.isfinite(M)`` to guard against NaN or Inf from the radial
        integration or angular coupling at arbitrary k-directions.
        """
        R_vals = slater_radial(r_grid, 2, 2.0)
        k_vec = jnp.array([1.0, 0.5, 0.3])
        M = dipole_matrix_element_single(
            k_vec, r_grid, R_vals, 1, 0, z_polarized
        )
        assert jnp.isfinite(M)

    def test_jit_compatible(self, r_grid, z_polarized):
        """Verify ``dipole_matrix_element_single`` is JAX-JIT-compatible.

        Wraps the function in ``jax.jit`` with k as the traced argument
        (r_grid, R_vals, l, m, and E-field are captured as constants).
        Uses an s-orbital (l=0, m=0) with k along z.  Asserts the JIT-
        compiled output is finite, confirming no Python-side control flow
        or non-JAX operations break tracing.
        """
        R_vals = slater_radial(r_grid, 1, 1.0)
        f = jax.jit(
            lambda k: dipole_matrix_element_single(
                k, r_grid, R_vals, 0, 0, z_polarized
            )
        )
        M = f(jnp.array([0.0, 0.0, 1.0]))
        assert jnp.isfinite(M)

    def test_gradient_wrt_efield(self, r_grid):
        """Verify autodiff gradient of intensity w.r.t. polarization vector.

        Defines a scalar loss = |M(E)|^2 for an s-orbital with k along z
        and differentiates through the full dipole pipeline with respect to
        the complex electric-field vector E using ``jax.grad``.  Asserts
        all three gradient components are finite, confirming the Gaunt
        lookup, radial integral, and complex-valued inner product are all
        smoothly differentiable.
        """
        R_vals = slater_radial(r_grid, 1, 1.0)
        k_vec = jnp.array([0.0, 0.0, 1.0])

        def loss(ef):
            M = dipole_matrix_element_single(k_vec, r_grid, R_vals, 0, 0, ef)
            return jnp.abs(M) ** 2

        grad = jax.grad(loss)(jnp.array([0.0, 0.0, 1.0], dtype=jnp.complex128))
        assert jnp.all(jnp.isfinite(grad))


class TestDipoleIntensityOrbital:
    """Tests for ``dipole_intensity_orbital``.

    Validates the single-orbital intensity function, which returns the
    modulus squared of the dipole matrix element.  Tests verify the
    physical constraint that intensity is non-negative and that the
    function is consistent with computing |M|^2 directly from
    ``dipole_matrix_element_single``.
    """

    def test_non_negative(self, r_grid, z_polarized):
        """Verify photoemission intensity is non-negative.

        Uses an n=2, zeta=1.5 Slater p-orbital (l=1, m=0) with an
        off-axis k-vector and z-polarization.  Since intensity is defined
        as |M|^2, it must be >= 0.  Asserts ``I >= 0.0`` as a basic
        physical sanity check.
        """
        R_vals = slater_radial(r_grid, 2, 1.5)
        k_vec = jnp.array([0.5, 0.3, 0.8])
        I = dipole_intensity_orbital(k_vec, r_grid, R_vals, 1, 0, z_polarized)
        assert float(I) >= 0.0

    def test_equals_abs_squared(self, r_grid, z_polarized):
        """Verify intensity equals the modulus squared of the matrix element.

        Computes both M and I independently for a 1s orbital (n=1,
        zeta=1.0, l=0, m=0) with k along z and z-polarization, then
        asserts ``I == |M|^2`` to within 1e-12.  This cross-checks that
        ``dipole_intensity_orbital`` is a faithful wrapper around
        ``dipole_matrix_element_single``.
        """
        R_vals = slater_radial(r_grid, 1, 1.0)
        k_vec = jnp.array([0.0, 0.0, 1.5])
        M = dipole_matrix_element_single(
            k_vec, r_grid, R_vals, 0, 0, z_polarized
        )
        I = dipole_intensity_orbital(k_vec, r_grid, R_vals, 0, 0, z_polarized)
        assert jnp.allclose(I, jnp.abs(M) ** 2, atol=1e-12)


class TestDipoleIntensitiesAllOrbitals:
    """Tests for ``dipole_intensities_all_orbitals``.

    Validates the batch function that computes dipole intensities for
    every orbital in a ``SlaterParams`` orbital basis.  Tests verify
    output shape (one intensity per orbital), non-negativity for all
    orbitals, and differentiability of the total intensity with respect
    to the Slater exponent parameter zeta.
    """

    def test_output_shape(self, r_grid, z_polarized):
        """Verify output has one intensity per orbital in the basis.

        Constructs a three-orbital basis (1s, 2pz, 2px) with corresponding
        Slater exponents and computes intensities at a single k-point.
        Asserts the output shape is ``(3,)``, matching the number of
        orbitals in the basis.
        """
        basis = make_orbital_basis(
            n_values=(1, 2, 2),
            l_values=(0, 1, 1),
            m_values=(0, 0, 1),
            labels=("1s", "2pz", "2px"),
        )
        sp = make_slater_params(
            zeta=jnp.array([1.0, 1.5, 1.5]),
            orbital_basis=basis,
        )
        k_vec = jnp.array([0.0, 0.0, 1.0])
        I = dipole_intensities_all_orbitals(k_vec, r_grid, sp, z_polarized)
        assert I.shape == (3,)

    def test_all_non_negative(self, r_grid, z_polarized):
        """Verify all orbital intensities are non-negative.

        Constructs a four-orbital basis (1s, 2py, 2pz, 2px) spanning all
        three p sub-orbitals plus s, with an off-axis k-vector and
        z-polarization.  Asserts ``I >= 0`` element-wise, confirming the
        |M|^2 definition holds for every orbital in the batch.
        """
        basis = make_orbital_basis(
            n_values=(1, 2, 2, 2),
            l_values=(0, 1, 1, 1),
            m_values=(0, -1, 0, 1),
            labels=("1s", "2py", "2pz", "2px"),
        )
        sp = make_slater_params(
            zeta=jnp.array([1.0, 1.5, 1.5, 1.5]),
            orbital_basis=basis,
        )
        k_vec = jnp.array([0.5, 0.3, 0.8])
        I = dipole_intensities_all_orbitals(k_vec, r_grid, sp, z_polarized)
        assert jnp.all(I >= 0.0)

    def test_gradient_wrt_zeta(self, r_grid, z_polarized):
        """Verify autodiff gradient of total intensity w.r.t. Slater exponent.

        Defines a scalar loss = sum(I) for a single 1s orbital and
        differentiates with respect to the Slater exponent zeta using
        ``jax.grad``.  Asserts the gradient is finite, confirming the
        entire forward pipeline (Slater radial construction, radial
        integral, Gaunt coupling, intensity summation) supports reverse-
        mode AD -- a prerequisite for inverse fitting of orbital exponents.
        """
        basis = make_orbital_basis(
            n_values=(1,),
            l_values=(0,),
            m_values=(0,),
            labels=("1s",),
        )

        def loss(zeta_val):
            sp = make_slater_params(
                zeta=jnp.array([zeta_val]),
                orbital_basis=basis,
            )
            k_vec = jnp.array([0.0, 0.0, 1.0])
            I = dipole_intensities_all_orbitals(k_vec, r_grid, sp, z_polarized)
            return jnp.sum(I)

        grad = jax.grad(loss)(jnp.array(1.0))
        assert jnp.isfinite(grad), f"Gradient is {grad}"
