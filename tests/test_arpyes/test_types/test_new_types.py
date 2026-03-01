"""Tests for new PyTree types: OrbitalBasis, SlaterParams, DiagonalizedBands, TBModel, SelfEnergyConfig."""

import jax
import jax.numpy as jnp
import pytest

from arpyes.types import (
    DiagonalizedBands,
    OrbitalBasis,
    SelfEnergyConfig,
    SlaterParams,
    TBModel,
    make_diagonalized_bands,
    make_orbital_basis,
    make_self_energy_config,
    make_slater_params,
    make_tb_model,
)


class TestOrbitalBasis:
    """Tests for OrbitalBasis PyTree."""

    def test_creation(self):
        """Can create an OrbitalBasis."""
        basis = make_orbital_basis(
            n_values=(1, 2),
            l_values=(0, 1),
            m_values=(0, 0),
        )
        assert basis.n_values == (1, 2)
        assert basis.l_values == (0, 1)

    def test_default_labels(self):
        """Labels default to 'orb_0', 'orb_1', ..."""
        basis = make_orbital_basis(
            n_values=(1,),
            l_values=(0,),
            m_values=(0,),
        )
        assert basis.labels == ("orb_0",)

    def test_pytree_flatten_unflatten(self):
        """PyTree round-trip preserves data."""
        basis = make_orbital_basis(
            n_values=(1, 2, 3),
            l_values=(0, 1, 2),
            m_values=(0, 0, 0),
            labels=("s", "p", "d"),
        )
        leaves, treedef = jax.tree_util.tree_flatten(basis)
        basis2 = jax.tree_util.tree_unflatten(treedef, leaves)
        assert basis2.n_values == basis.n_values
        assert basis2.labels == basis.labels

    def test_length_mismatch_raises(self):
        """Mismatched lengths raise ValueError."""
        with pytest.raises(ValueError, match="same length"):
            make_orbital_basis(
                n_values=(1, 2),
                l_values=(0,),
                m_values=(0, 0),
            )


class TestSlaterParams:
    """Tests for SlaterParams PyTree."""

    def test_creation(self):
        """Can create SlaterParams."""
        basis = make_orbital_basis(
            n_values=(1,),
            l_values=(0,),
            m_values=(0,),
        )
        sp = make_slater_params(
            zeta=jnp.array([1.5]),
            orbital_basis=basis,
        )
        assert sp.zeta.shape == (1,)
        assert sp.coefficients.shape == (1, 1)

    def test_pytree_gradient(self):
        """Gradient flows through SlaterParams."""
        basis = make_orbital_basis(
            n_values=(1,),
            l_values=(0,),
            m_values=(0,),
        )

        def loss(sp):
            return jnp.sum(sp.zeta ** 2)

        sp = make_slater_params(
            zeta=jnp.array([2.0]),
            orbital_basis=basis,
        )
        grad = jax.grad(loss)(sp)
        assert jnp.allclose(grad.zeta, 4.0)

    def test_float64_casting(self):
        """Arrays are cast to float64."""
        basis = make_orbital_basis(
            n_values=(1,),
            l_values=(0,),
            m_values=(0,),
        )
        sp = make_slater_params(
            zeta=jnp.array([1.0], dtype=jnp.float32),
            orbital_basis=basis,
        )
        assert sp.zeta.dtype == jnp.float64


class TestDiagonalizedBands:
    """Tests for DiagonalizedBands PyTree."""

    def test_creation(self):
        """Can create DiagonalizedBands."""
        K, B, O = 5, 3, 4
        diag = make_diagonalized_bands(
            eigenvalues=jnp.zeros((K, B)),
            eigenvectors=jnp.ones((K, B, O), dtype=jnp.complex128),
            kpoints=jnp.zeros((K, 3)),
            fermi_energy=0.0,
        )
        assert diag.eigenvalues.shape == (K, B)
        assert diag.eigenvectors.shape == (K, B, O)

    def test_pytree_round_trip(self):
        """PyTree flatten/unflatten preserves data."""
        K, B, O = 2, 2, 2
        diag = make_diagonalized_bands(
            eigenvalues=jnp.array([[1.0, 2.0], [3.0, 4.0]]),
            eigenvectors=jnp.eye(2, dtype=jnp.complex128)[None].repeat(2, axis=0),
            kpoints=jnp.zeros((2, 3)),
        )
        leaves, treedef = jax.tree_util.tree_flatten(diag)
        diag2 = jax.tree_util.tree_unflatten(treedef, leaves)
        assert jnp.allclose(diag2.eigenvalues, diag.eigenvalues)


class TestTBModel:
    """Tests for TBModel PyTree."""

    def test_creation(self):
        """Can create TBModel."""
        basis = make_orbital_basis(
            n_values=(1,),
            l_values=(0,),
            m_values=(0,),
        )
        model = make_tb_model(
            hopping_params=jnp.array([1.0, 1.0]),
            lattice_vectors=jnp.eye(3),
            hopping_indices=((0, 0, (1, 0, 0)), (0, 0, (-1, 0, 0))),
            n_orbitals=1,
            orbital_basis=basis,
        )
        assert model.hopping_params.shape == (2,)

    def test_pytree_gradient(self):
        """Gradient flows through TBModel hopping params."""
        basis = make_orbital_basis(
            n_values=(1,),
            l_values=(0,),
            m_values=(0,),
        )
        model = make_tb_model(
            hopping_params=jnp.array([1.0]),
            lattice_vectors=jnp.eye(3),
            hopping_indices=((0, 0, (1, 0, 0)),),
            n_orbitals=1,
            orbital_basis=basis,
        )

        def loss(m):
            return jnp.sum(m.hopping_params ** 2)

        grad = jax.grad(loss)(model)
        assert jnp.allclose(grad.hopping_params, 2.0)


class TestSelfEnergyConfig:
    """Tests for SelfEnergyConfig PyTree."""

    def test_constant_default(self):
        """Default creates a constant config."""
        config = make_self_energy_config()
        assert config.mode == "constant"
        assert config.coefficients.shape == (1,)
        assert float(config.coefficients[0]) == pytest.approx(0.1)

    def test_polynomial(self):
        """Polynomial mode works."""
        config = make_self_energy_config(
            mode="polynomial",
            coefficients=jnp.array([0.01, 0.1]),
        )
        assert config.mode == "polynomial"

    def test_tabulated_requires_nodes(self):
        """Tabulated mode without nodes raises ValueError."""
        with pytest.raises(ValueError, match="energy_nodes required"):
            make_self_energy_config(mode="tabulated")

    def test_invalid_mode_raises(self):
        """Invalid mode raises ValueError."""
        with pytest.raises(ValueError, match="mode must be"):
            make_self_energy_config(mode="invalid")

    def test_pytree_round_trip(self):
        """PyTree flatten/unflatten preserves data."""
        config = make_self_energy_config(gamma=0.2)
        leaves, treedef = jax.tree_util.tree_flatten(config)
        config2 = jax.tree_util.tree_unflatten(treedef, leaves)
        assert config2.mode == config.mode
        assert jnp.allclose(config2.coefficients, config.coefficients)
