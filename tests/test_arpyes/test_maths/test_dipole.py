"""Tests for dipole matrix element assembly."""

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
    """Standard radial grid for tests."""
    return jnp.linspace(1e-6, 50.0, 5000)


@pytest.fixture
def z_polarized():
    """Z-polarized E-field."""
    return jnp.array([0.0, 0.0, 1.0], dtype=jnp.complex128)


@pytest.fixture
def x_polarized():
    """X-polarized E-field."""
    return jnp.array([1.0, 0.0, 0.0], dtype=jnp.complex128)


class TestDipoleMatrixElementSingle:
    """Tests for dipole_matrix_element_single."""

    def test_s_orbital_nonzero(self, r_grid, z_polarized):
        """s-orbital (l=0, m=0) dipole element is nonzero for z-pol."""
        R_vals = slater_radial(r_grid, 1, 1.0)
        k_vec = jnp.array([0.0, 0.0, 1.0])
        M = dipole_matrix_element_single(
            k_vec, r_grid, R_vals, 0, 0, z_polarized
        )
        assert (
            jnp.abs(M) > 1e-6
        ), f"Expected nonzero, got |M|={float(jnp.abs(M))}"

    def test_finite_output(self, r_grid, z_polarized):
        """Output is always finite."""
        R_vals = slater_radial(r_grid, 2, 2.0)
        k_vec = jnp.array([1.0, 0.5, 0.3])
        M = dipole_matrix_element_single(
            k_vec, r_grid, R_vals, 1, 0, z_polarized
        )
        assert jnp.isfinite(M)

    def test_jit_compatible(self, r_grid, z_polarized):
        """Can be JIT-compiled."""
        R_vals = slater_radial(r_grid, 1, 1.0)
        f = jax.jit(
            lambda k: dipole_matrix_element_single(
                k, r_grid, R_vals, 0, 0, z_polarized
            )
        )
        M = f(jnp.array([0.0, 0.0, 1.0]))
        assert jnp.isfinite(M)

    def test_gradient_wrt_efield(self, r_grid):
        """Gradient w.r.t. E-field is finite."""
        R_vals = slater_radial(r_grid, 1, 1.0)
        k_vec = jnp.array([0.0, 0.0, 1.0])

        def loss(ef):
            M = dipole_matrix_element_single(k_vec, r_grid, R_vals, 0, 0, ef)
            return jnp.abs(M) ** 2

        grad = jax.grad(loss)(jnp.array([0.0, 0.0, 1.0], dtype=jnp.complex128))
        assert jnp.all(jnp.isfinite(grad))


class TestDipoleIntensityOrbital:
    """Tests for dipole_intensity_orbital."""

    def test_non_negative(self, r_grid, z_polarized):
        """Intensity is non-negative."""
        R_vals = slater_radial(r_grid, 2, 1.5)
        k_vec = jnp.array([0.5, 0.3, 0.8])
        I = dipole_intensity_orbital(k_vec, r_grid, R_vals, 1, 0, z_polarized)
        assert float(I) >= 0.0

    def test_equals_abs_squared(self, r_grid, z_polarized):
        """Intensity equals |M|^2."""
        R_vals = slater_radial(r_grid, 1, 1.0)
        k_vec = jnp.array([0.0, 0.0, 1.5])
        M = dipole_matrix_element_single(
            k_vec, r_grid, R_vals, 0, 0, z_polarized
        )
        I = dipole_intensity_orbital(k_vec, r_grid, R_vals, 0, 0, z_polarized)
        assert jnp.allclose(I, jnp.abs(M) ** 2, atol=1e-12)


class TestDipoleIntensitiesAllOrbitals:
    """Tests for dipole_intensities_all_orbitals."""

    def test_output_shape(self, r_grid, z_polarized):
        """Returns one intensity per orbital."""
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
        """All intensities are non-negative."""
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
        """Gradient w.r.t. Slater exponent zeta is finite."""
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
