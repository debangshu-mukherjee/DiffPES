"""Tests for differentiable diagonalization and VASP adapter."""

import jax
import jax.numpy as jnp
import pytest

from arpyes.tightb.diagonalize import (
    diagonalize_single_k,
    diagonalize_tb,
    vasp_to_diagonalized,
)
from arpyes.tightb.hamiltonian import make_1d_chain_model, make_graphene_model
from arpyes.types import (
    make_band_structure,
    make_orbital_basis,
    make_orbital_projection,
)


class TestDiagonalizeSingleK:
    """Tests for diagonalize_single_k."""

    def test_eigenvalues_real(self):
        """Eigenvalues are real."""
        H = jnp.array(
            [[1.0, 0.5 + 0.1j], [0.5 - 0.1j, 2.0]], dtype=jnp.complex128
        )
        evals, evecs = diagonalize_single_k(H)
        assert evals.dtype == jnp.float64

    def test_eigenvectors_orthogonal(self):
        """Eigenvectors are orthonormal."""
        H = jnp.array([[1.0, 0.5], [0.5, 2.0]], dtype=jnp.complex128)
        evals, evecs = diagonalize_single_k(H)
        overlap = evecs.conj().T @ evecs
        assert jnp.allclose(overlap, jnp.eye(2), atol=1e-10)


class TestDiagonalizeTB:
    """Tests for diagonalize_tb."""

    def test_output_shapes(self):
        """Output has correct shapes."""
        model = make_graphene_model()
        kpoints = jnp.array([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]])
        diag = diagonalize_tb(model, kpoints)
        assert diag.eigenvalues.shape == (2, 2)
        assert diag.eigenvectors.shape == (2, 2, 2)
        assert diag.kpoints.shape == (2, 3)

    def test_eigenvalues_sorted(self):
        """Eigenvalues are sorted in ascending order per k-point."""
        model = make_graphene_model()
        kpoints = jnp.array(
            [[0.0, 0.0, 0.0], [0.1, 0.2, 0.0], [0.3, 0.1, 0.0]]
        )
        diag = diagonalize_tb(model, kpoints)
        for i in range(3):
            assert float(diag.eigenvalues[i, 0]) <= float(
                diag.eigenvalues[i, 1]
            )

    def test_differentiable(self):
        """Eigenvalues are differentiable w.r.t. hopping params."""
        model = make_1d_chain_model(t=-1.0)
        kpoints = jnp.array([[0.25, 0.0, 0.0]])

        def loss(hop):
            m = model._replace(hopping_params=hop)
            d = diagonalize_tb(m, kpoints)
            return jnp.sum(d.eigenvalues)

        grad = jax.grad(loss)(model.hopping_params)
        assert jnp.all(jnp.isfinite(grad))

    def test_eigenvectors_shape_convention(self):
        """eigenvectors[k, b, o] = coefficient of orbital o in band b."""
        model = make_graphene_model()
        kpoints = jnp.array([[0.0, 0.0, 0.0]])
        diag = diagonalize_tb(model, kpoints)
        # K=1, B=2, O=2
        assert diag.eigenvectors.shape == (1, 2, 2)
        # Eigenvectors should be normalized per band
        norms = jnp.sum(jnp.abs(diag.eigenvectors[0]) ** 2, axis=1)
        assert jnp.allclose(norms, 1.0, atol=1e-10)


class TestVaspToDiagonalized:
    """Tests for vasp_to_diagonalized."""

    def test_output_shapes(self):
        """Output shapes match input band structure."""
        K, B, A = 5, 3, 2
        bands = make_band_structure(
            eigenvalues=jnp.zeros((K, B)),
            kpoints=jnp.zeros((K, 3)),
        )
        orb_proj = make_orbital_projection(
            projections=jnp.ones((K, B, A, 9)) / 9.0
        )
        basis = make_orbital_basis(
            n_values=(1, 2, 2),
            l_values=(0, 1, 1),
            m_values=(0, 0, 1),
        )
        diag = vasp_to_diagonalized(bands, orb_proj, basis)
        assert diag.eigenvalues.shape == (K, B)
        assert diag.eigenvectors.shape == (K, B, 3)

    def test_eigenvectors_normalized(self):
        """Approximate eigenvectors are normalized."""
        K, B, A = 3, 2, 1
        bands = make_band_structure(
            eigenvalues=jnp.zeros((K, B)),
            kpoints=jnp.zeros((K, 3)),
        )
        proj = jnp.zeros((K, B, A, 9))
        proj = proj.at[:, :, :, 0].set(0.3)  # s
        proj = proj.at[:, :, :, 2].set(0.7)  # pz
        orb_proj = make_orbital_projection(projections=proj)
        basis = make_orbital_basis(
            n_values=(1, 2),
            l_values=(0, 1),
            m_values=(0, 0),
        )
        diag = vasp_to_diagonalized(bands, orb_proj, basis)
        norms = jnp.sum(jnp.abs(diag.eigenvectors) ** 2, axis=-1)
        assert jnp.allclose(norms, 1.0, atol=1e-10)
