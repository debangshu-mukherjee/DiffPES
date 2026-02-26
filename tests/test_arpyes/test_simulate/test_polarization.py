"""Tests for polarization functions."""

import chex
import jax.numpy as jnp

from arpyes.simulate.polarization import (
    build_efield,
    build_polarization_vectors,
    dipole_matrix_elements,
)
from arpyes.types import make_polarization_config


class TestBuildPolarizationVectors(chex.TestCase):

    def test_orthogonality(self):
        theta = jnp.pi / 4.0
        phi = 0.0
        e_s, e_p = build_polarization_vectors(theta, phi)
        dot_product = jnp.dot(e_s, e_p)
        chex.assert_trees_all_close(
            dot_product, jnp.float64(0.0), atol=1e-10
        )

    def test_unit_vectors(self):
        theta = jnp.pi / 3.0
        phi = jnp.pi / 6.0
        e_s, e_p = build_polarization_vectors(theta, phi)
        chex.assert_trees_all_close(
            jnp.linalg.norm(e_s),
            jnp.float64(1.0),
            atol=1e-10,
        )
        chex.assert_trees_all_close(
            jnp.linalg.norm(e_p),
            jnp.float64(1.0),
            atol=1e-10,
        )

    def test_shape(self):
        e_s, e_p = build_polarization_vectors(0.5, 0.0)
        chex.assert_shape(e_s, (3,))
        chex.assert_shape(e_p, (3,))


class TestBuildEfield(chex.TestCase):

    def test_lvp_equals_s(self):
        config = make_polarization_config(
            theta=jnp.pi / 4.0,
            phi=0.0,
            polarization_type="LVP",
        )
        efield = build_efield(config)
        e_s, _ = build_polarization_vectors(
            config.theta, config.phi
        )
        chex.assert_trees_all_close(
            jnp.real(efield),
            e_s.astype(jnp.float64),
            atol=1e-10,
        )

    def test_lhp_equals_p(self):
        config = make_polarization_config(
            theta=jnp.pi / 4.0,
            phi=0.0,
            polarization_type="LHP",
        )
        efield = build_efield(config)
        _, e_p = build_polarization_vectors(
            config.theta, config.phi
        )
        chex.assert_trees_all_close(
            jnp.real(efield),
            e_p.astype(jnp.float64),
            atol=1e-10,
        )

    def test_rcp_lcp_conjugate(self):
        config_r = make_polarization_config(
            theta=jnp.pi / 4.0,
            phi=0.0,
            polarization_type="RCP",
        )
        config_l = make_polarization_config(
            theta=jnp.pi / 4.0,
            phi=0.0,
            polarization_type="LCP",
        )
        e_rcp = build_efield(config_r)
        e_lcp = build_efield(config_l)
        chex.assert_trees_all_close(
            jnp.real(e_rcp),
            jnp.real(e_lcp),
            atol=1e-10,
        )


class TestDipoleMatrixElements(chex.TestCase):

    def test_shape(self):
        efield = jnp.array(
            [1.0, 0.0, 0.0], dtype=jnp.complex128
        )
        m = dipole_matrix_elements(efield)
        chex.assert_shape(m, (9,))

    def test_s_orbital_zero(self):
        efield = jnp.array(
            [1.0, 0.0, 0.0], dtype=jnp.complex128
        )
        m = dipole_matrix_elements(efield)
        chex.assert_trees_all_close(
            m[0], jnp.float64(0.0), atol=1e-10
        )

    def test_px_with_x_field(self):
        efield = jnp.array(
            [1.0, 0.0, 0.0], dtype=jnp.complex128
        )
        m = dipole_matrix_elements(efield)
        chex.assert_scalar_positive(float(m[3]))

    def test_all_nonnegative(self):
        efield = jnp.array(
            [0.5, 0.3, 0.8], dtype=jnp.complex128
        )
        efield = efield / jnp.linalg.norm(efield)
        m = dipole_matrix_elements(efield)
        for i in range(9):
            assert float(m[i]) >= 0.0
