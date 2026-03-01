"""Tests for tight-binding Hamiltonian builder."""

import jax
import jax.numpy as jnp
import pytest

from arpyes.tb.hamiltonian import (
    build_hamiltonian_k,
    make_1d_chain_model,
    make_graphene_model,
)


class TestBuildHamiltonianK:
    """Tests for build_hamiltonian_k."""

    def test_hermitian(self):
        """H(k) is Hermitian."""
        model = make_graphene_model()
        k = jnp.array([0.3, 0.2, 0.0])
        H = build_hamiltonian_k(
            k,
            model.hopping_params,
            model.hopping_indices,
            model.n_orbitals,
            model.lattice_vectors,
        )
        assert jnp.allclose(H, H.conj().T, atol=1e-12)

    def test_correct_shape(self):
        """H(k) has shape (n_orbitals, n_orbitals)."""
        model = make_graphene_model()
        k = jnp.array([0.1, 0.2, 0.0])
        H = build_hamiltonian_k(
            k,
            model.hopping_params,
            model.hopping_indices,
            model.n_orbitals,
            model.lattice_vectors,
        )
        assert H.shape == (2, 2)

    def test_jit_compatible(self):
        """Can be JIT-compiled."""
        model = make_1d_chain_model()
        f = jax.jit(
            lambda k: build_hamiltonian_k(
                k,
                model.hopping_params,
                model.hopping_indices,
                model.n_orbitals,
                model.lattice_vectors,
            )
        )
        H = f(jnp.array([0.25, 0.0, 0.0]))
        assert jnp.all(jnp.isfinite(H))


class TestMake1DChainModel:
    """Tests for 1D chain model."""

    def test_cosine_dispersion(self):
        """1D chain: E(k) = 2t cos(2πk)."""
        model = make_1d_chain_model(t=-1.0)
        kpoints = jnp.linspace(-0.5, 0.5, 101)[:, None] * jnp.array([[1.0, 0.0, 0.0]])

        from arpyes.tb.diagonalize import diagonalize_tb

        diag = diagonalize_tb(model, kpoints)
        expected = -2.0 * jnp.cos(2.0 * jnp.pi * jnp.linspace(-0.5, 0.5, 101))
        assert jnp.allclose(diag.eigenvalues[:, 0], expected, atol=1e-10)

    def test_eigenvalue_range(self):
        """Eigenvalues span [-2|t|, 2|t|]."""
        model = make_1d_chain_model(t=-1.5)
        kpoints = jnp.linspace(-0.5, 0.5, 201)[:, None] * jnp.array([[1.0, 0.0, 0.0]])

        from arpyes.tb.diagonalize import diagonalize_tb

        diag = diagonalize_tb(model, kpoints)
        assert float(diag.eigenvalues.min()) == pytest.approx(-3.0, abs=0.05)
        assert float(diag.eigenvalues.max()) == pytest.approx(3.0, abs=0.05)


class TestMakeGrapheneModel:
    """Tests for graphene model."""

    def test_gamma_point(self):
        """At Γ, eigenvalues are ±3|t|."""
        model = make_graphene_model(t=-2.7)
        Gamma = jnp.array([[0.0, 0.0, 0.0]])

        from arpyes.tb.diagonalize import diagonalize_tb

        diag = diagonalize_tb(model, Gamma)
        evals = jnp.sort(diag.eigenvalues[0])
        assert float(evals[0]) == pytest.approx(-8.1, abs=0.01)
        assert float(evals[1]) == pytest.approx(8.1, abs=0.01)

    def test_k_point_dirac(self):
        """At K=(2/3, 1/3, 0), eigenvalues are 0 (Dirac point)."""
        model = make_graphene_model(t=-2.7)
        K = jnp.array([[2.0 / 3, 1.0 / 3, 0.0]])

        from arpyes.tb.diagonalize import diagonalize_tb

        diag = diagonalize_tb(model, K)
        assert jnp.allclose(diag.eigenvalues[0], 0.0, atol=1e-10)

    def test_gradient_wrt_hopping(self):
        """Gradient of eigenvalues w.r.t. hopping is finite."""
        model = make_graphene_model(t=-2.7)
        kpoints = jnp.array([[0.1, 0.2, 0.0]])

        from arpyes.tb.diagonalize import diagonalize_tb

        def loss(hop):
            m = model._replace(hopping_params=hop)
            d = diagonalize_tb(m, kpoints)
            return jnp.sum(d.eigenvalues ** 2)

        grad = jax.grad(loss)(model.hopping_params)
        assert jnp.all(jnp.isfinite(grad))
