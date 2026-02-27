"""Tests for parser-adjacent workflow helpers."""

import chex
import jax.numpy as jnp
import pytest

from arpyes.inout import aggregate_atoms, check_consistency, reduce_orbitals, select_atoms
from arpyes.types import (
    SpinOrbitalProjection,
    make_band_structure,
    make_kpath_info,
    make_orbital_projection,
    make_spin_orbital_projection,
)


def _make_test_orb():
    """Create a test OrbitalProjection with 2 k-points, 2 bands, 3 atoms."""
    proj = jnp.zeros((2, 2, 3, 9), dtype=jnp.float64)
    # Set s-orbital for atom 0 to 1.0, atom 1 to 2.0, atom 2 to 3.0
    proj = proj.at[:, :, 0, 0].set(1.0)
    proj = proj.at[:, :, 1, 0].set(2.0)
    proj = proj.at[:, :, 2, 0].set(3.0)
    # Set p-orbitals (indices 1-3) for all atoms
    proj = proj.at[:, :, :, 1].set(0.1)
    proj = proj.at[:, :, :, 2].set(0.2)
    proj = proj.at[:, :, :, 3].set(0.3)
    # Set d-orbitals (indices 4-8) for all atoms
    proj = proj.at[:, :, :, 4].set(0.01)
    proj = proj.at[:, :, :, 5].set(0.02)
    proj = proj.at[:, :, :, 6].set(0.03)
    proj = proj.at[:, :, :, 7].set(0.04)
    proj = proj.at[:, :, :, 8].set(0.05)
    return make_orbital_projection(projections=proj)


class TestSelectAtoms(chex.TestCase):
    """Tests for select_atoms helper."""

    def test_select_single_atom(self):
        """Select a single atom and verify shape and values."""
        orb = _make_test_orb()
        sub = select_atoms(orb, [1])
        chex.assert_shape(sub.projections, (2, 2, 1, 9))
        chex.assert_trees_all_close(
            sub.projections[0, 0, 0, 0], jnp.float64(2.0), atol=1e-12
        )

    def test_select_multiple_atoms(self):
        """Select two atoms and verify shape."""
        orb = _make_test_orb()
        sub = select_atoms(orb, [0, 2])
        chex.assert_shape(sub.projections, (2, 2, 2, 9))
        chex.assert_trees_all_close(
            sub.projections[0, 0, 0, 0], jnp.float64(1.0), atol=1e-12
        )
        chex.assert_trees_all_close(
            sub.projections[0, 0, 1, 0], jnp.float64(3.0), atol=1e-12
        )

    def test_preserves_spin_orbital_projection_type(self):
        """SpinOrbitalProjection input returns SpinOrbitalProjection."""
        proj = jnp.ones((2, 2, 3, 9), dtype=jnp.float64)
        spin = jnp.ones((2, 2, 3, 6), dtype=jnp.float64)
        orb = make_spin_orbital_projection(projections=proj, spin=spin)
        sub = select_atoms(orb, [0, 2])
        assert isinstance(sub, SpinOrbitalProjection)
        chex.assert_shape(sub.projections, (2, 2, 2, 9))
        chex.assert_shape(sub.spin, (2, 2, 2, 6))


class TestAggregateAtoms(chex.TestCase):
    """Tests for aggregate_atoms helper."""

    def test_aggregate_all(self):
        """Sum over all atoms."""
        orb = _make_test_orb()
        agg = aggregate_atoms(orb)
        chex.assert_shape(agg, (2, 2, 9))
        # s-orbital: 1+2+3 = 6
        chex.assert_trees_all_close(
            agg[0, 0, 0], jnp.float64(6.0), atol=1e-12
        )

    def test_aggregate_subset(self):
        """Sum over a subset of atoms."""
        orb = _make_test_orb()
        agg = aggregate_atoms(orb, [0, 1])
        chex.assert_shape(agg, (2, 2, 9))
        # s-orbital: 1+2 = 3
        chex.assert_trees_all_close(
            agg[0, 0, 0], jnp.float64(3.0), atol=1e-12
        )


class TestReduceOrbitals(chex.TestCase):
    """Tests for reduce_orbitals helper."""

    def test_reduces_to_spd(self):
        """Reduce 9 orbitals to s/p/d totals."""
        orb = _make_test_orb()
        reduced = reduce_orbitals(orb.projections)
        chex.assert_shape(reduced, (2, 2, 3, 3))
        # For atom 0: s=1.0, p=0.1+0.2+0.3=0.6, d=0.01+...+0.05=0.15
        chex.assert_trees_all_close(
            reduced[0, 0, 0, 0], jnp.float64(1.0), atol=1e-12
        )
        chex.assert_trees_all_close(
            reduced[0, 0, 0, 1], jnp.float64(0.6), atol=1e-12
        )
        chex.assert_trees_all_close(
            reduced[0, 0, 0, 2], jnp.float64(0.15), atol=1e-12
        )


class TestCheckConsistency(chex.TestCase):
    """Tests for check_consistency helper."""

    def test_consistent_inputs(self):
        """No error when dimensions agree."""
        bands = make_band_structure(
            eigenvalues=jnp.zeros((2, 3)),
            kpoints=jnp.zeros((2, 3)),
        )
        orb = make_orbital_projection(
            projections=jnp.zeros((2, 3, 1, 9)),
        )
        check_consistency(bands, orb)

    def test_kpoint_mismatch(self):
        """Raise ValueError on k-point count mismatch."""
        bands = make_band_structure(
            eigenvalues=jnp.zeros((2, 3)),
            kpoints=jnp.zeros((2, 3)),
        )
        orb = make_orbital_projection(
            projections=jnp.zeros((4, 3, 1, 9)),
        )
        with pytest.raises(ValueError, match="K-point count mismatch"):
            check_consistency(bands, orb)

    def test_band_mismatch(self):
        """Raise ValueError on band count mismatch."""
        bands = make_band_structure(
            eigenvalues=jnp.zeros((2, 3)),
            kpoints=jnp.zeros((2, 3)),
        )
        orb = make_orbital_projection(
            projections=jnp.zeros((2, 5, 1, 9)),
        )
        with pytest.raises(ValueError, match="Band count mismatch"):
            check_consistency(bands, orb)

    def test_with_kpath(self):
        """Check consistency with KPathInfo."""
        bands = make_band_structure(
            eigenvalues=jnp.zeros((10, 3)),
            kpoints=jnp.zeros((10, 3)),
        )
        orb = make_orbital_projection(
            projections=jnp.zeros((10, 3, 1, 9)),
        )
        kpath = make_kpath_info(
            num_kpoints=10,
            label_indices=[0, 9],
            mode="Line-mode",
        )
        check_consistency(bands, orb, kpath)
