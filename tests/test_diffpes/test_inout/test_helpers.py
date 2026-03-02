"""Tests for parser-adjacent workflow helpers."""

import chex
import jax.numpy as jnp
import pytest

from diffpes.inout import (
    aggregate_atoms,
    check_consistency,
    reduce_orbitals,
    select_atoms,
)
from diffpes.types import (
    SpinOrbitalProjection,
    make_band_structure,
    make_kpath_info,
    make_orbital_projection,
    make_spin_orbital_projection,
)


def _make_test_orb():
    """Create a test OrbitalProjection with 2 k-points, 2 bands, 3 atoms.

    Constructs a synthetic OrbitalProjection fixture with shape
    (2, 2, 3, 9) where each atom has distinct s-orbital values
    (1.0, 2.0, 3.0 for atoms 0, 1, 2 respectively), uniform p-orbital
    values (0.1, 0.2, 0.3 for py, pz, px), and uniform d-orbital
    values (0.01--0.05 for the five d channels). This provides
    deterministic, analytically verifiable test data for select, aggregate,
    and reduce operations.

    Returns
    -------
    OrbitalProjection
        PyTree with ``projections`` of shape (2, 2, 3, 9) and no spin
        or OAM data.
    """
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
    """Tests for :func:`diffpes.inout.select_atoms`.

    Validates atom-axis slicing of OrbitalProjection and
    SpinOrbitalProjection PyTrees. Covers single-atom selection,
    multi-atom selection with value preservation, and type preservation
    for the SpinOrbitalProjection subclass (ensuring both ``projections``
    and ``spin`` arrays are sliced consistently along the atom axis).
    """

    def test_select_single_atom(self):
        """Select a single atom by index and verify output shape and s-orbital value.

        Uses the ``_make_test_orb`` fixture (3 atoms with distinct s-orbital
        weights) and selects atom index 1. Asserts that the output projection
        shape collapses from 3 atoms to 1 along axis 2, and that the s-orbital
        value for the selected atom equals 2.0 (the value assigned to atom 1
        in the fixture), verified to within ``atol=1e-12``.
        """
        orb = _make_test_orb()
        sub = select_atoms(orb, [1])
        chex.assert_shape(sub.projections, (2, 2, 1, 9))
        chex.assert_trees_all_close(
            sub.projections[0, 0, 0, 0], jnp.float64(2.0), atol=1e-12
        )

    def test_select_multiple_atoms(self):
        """Select two non-contiguous atoms and verify shape and ordering.

        Selects atoms 0 and 2 from the 3-atom fixture. Asserts the output
        shape is (2, 2, 2, 9) -- two atoms retained -- and that the
        s-orbital values appear in selection order: atom 0 maps to output
        index 0 with value 1.0, atom 2 maps to output index 1 with value
        3.0. Both values are verified to within ``atol=1e-12``.
        """
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
        """Verify that selecting atoms from a SpinOrbitalProjection returns SpinOrbitalProjection.

        Constructs a SpinOrbitalProjection with ``projections`` shape
        (2, 2, 3, 9) and ``spin`` shape (2, 2, 3, 6), then selects atoms
        0 and 2. Asserts the result is an instance of SpinOrbitalProjection
        (not the base OrbitalProjection), and that both ``projections`` and
        ``spin`` arrays are sliced to 2 atoms (shapes (2, 2, 2, 9) and
        (2, 2, 2, 6) respectively). This is a regression guard ensuring the
        function dispatches correctly on the input subtype.
        """
        proj = jnp.ones((2, 2, 3, 9), dtype=jnp.float64)
        spin = jnp.ones((2, 2, 3, 6), dtype=jnp.float64)
        orb = make_spin_orbital_projection(projections=proj, spin=spin)
        sub = select_atoms(orb, [0, 2])
        assert isinstance(sub, SpinOrbitalProjection)
        chex.assert_shape(sub.projections, (2, 2, 2, 9))
        chex.assert_shape(sub.spin, (2, 2, 2, 6))


class TestAggregateAtoms(chex.TestCase):
    """Tests for :func:`diffpes.inout.aggregate_atoms`.

    Validates summation of orbital projections over the atom axis. Covers
    aggregation over all atoms (full sum) and over an explicit subset of
    atom indices, verifying both output shape collapse and numerical
    correctness of the summed s-orbital values.
    """

    def test_aggregate_all(self):
        """Sum projections over all atoms and verify collapsed shape and total.

        Calls ``aggregate_atoms`` with no atom indices (default: all atoms).
        Asserts the output shape is (2, 2, 9) -- the atom dimension is
        removed -- and that the s-orbital value at [0, 0, 0] equals 6.0
        (the sum 1.0 + 2.0 + 3.0 from atoms 0, 1, 2 in the fixture),
        verified to within ``atol=1e-12``.
        """
        orb = _make_test_orb()
        agg = aggregate_atoms(orb)
        chex.assert_shape(agg, (2, 2, 9))
        # s-orbital: 1+2+3 = 6
        chex.assert_trees_all_close(agg[0, 0, 0], jnp.float64(6.0), atol=1e-12)

    def test_aggregate_subset(self):
        """Sum projections over a specified subset of atoms.

        Calls ``aggregate_atoms`` with atom indices [0, 1]. Asserts the
        output shape is (2, 2, 9) and the s-orbital value at [0, 0, 0]
        equals 3.0 (1.0 + 2.0 from atoms 0 and 1 only), verified to
        within ``atol=1e-12``. This exercises the explicit atom-selection
        branch prior to summation.
        """
        orb = _make_test_orb()
        agg = aggregate_atoms(orb, [0, 1])
        chex.assert_shape(agg, (2, 2, 9))
        # s-orbital: 1+2 = 3
        chex.assert_trees_all_close(agg[0, 0, 0], jnp.float64(3.0), atol=1e-12)


class TestReduceOrbitals(chex.TestCase):
    """Tests for :func:`diffpes.inout.reduce_orbitals`.

    Validates reduction of the 9-channel orbital decomposition
    (s, py, pz, px, dxy, dyz, dz2, dxz, dx2-y2) into three aggregate
    channels (s-total, p-total, d-total) by summing the appropriate
    groups.
    """

    def test_reduces_to_spd(self):
        """Reduce 9-channel orbital projections to s/p/d totals and verify sums.

        Applies ``reduce_orbitals`` to the fixture's raw projections array.
        Asserts the output shape is (2, 2, 3, 3) -- 9 orbital channels
        collapsed to 3 -- and checks atom 0's reduced values analytically:

        - s = 1.0 (single channel, index 0)
        - p = 0.1 + 0.2 + 0.3 = 0.6 (channels 1--3 summed)
        - d = 0.01 + 0.02 + 0.03 + 0.04 + 0.05 = 0.15 (channels 4--8 summed)

        All comparisons use ``atol=1e-12``.
        """
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
    """Tests for :func:`diffpes.inout.check_consistency`.

    Validates the dimension-compatibility checker that ensures
    BandStructure, OrbitalProjection, and (optionally) KPathInfo
    agree on k-point and band counts. Covers the happy path (all
    dimensions match), k-point mismatch, band mismatch, and the
    optional KPathInfo consistency check.
    """

    def test_consistent_inputs(self):
        """Verify no error is raised when BandStructure and OrbitalProjection agree.

        Constructs a BandStructure with 2 k-points and 3 bands, and an
        OrbitalProjection with matching leading dimensions (2, 3, 1, 9).
        Calls ``check_consistency`` and asserts it returns without
        raising, confirming the positive validation path.
        """
        bands = make_band_structure(
            eigenvalues=jnp.zeros((2, 3)),
            kpoints=jnp.zeros((2, 3)),
        )
        orb = make_orbital_projection(
            projections=jnp.zeros((2, 3, 1, 9)),
        )
        check_consistency(bands, orb)

    def test_kpoint_mismatch(self):
        """Verify ValueError is raised when k-point counts disagree.

        Constructs a BandStructure with 2 k-points but an
        OrbitalProjection with 4 k-points. Asserts that
        ``check_consistency`` raises ``ValueError`` matching
        ``"K-point count mismatch"``, exercising the first dimension
        check.
        """
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
        """Verify ValueError is raised when band counts disagree.

        Constructs a BandStructure with 3 bands but an
        OrbitalProjection with 5 bands (k-points match at 2). Asserts
        that ``check_consistency`` raises ``ValueError`` matching
        ``"Band count mismatch"``, exercising the second dimension
        check independently from the k-point check.
        """
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
        """Verify consistency check passes when optional KPathInfo is included.

        Constructs a BandStructure (10 k-points, 3 bands), a matching
        OrbitalProjection (10, 3, 1, 9), and a KPathInfo with
        ``num_kpoints=10``, label indices at start and end, and
        ``"Line-mode"`` mode. Calls ``check_consistency`` with all three
        arguments and asserts no error is raised, confirming the three-way
        dimension agreement path.
        """
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

    def test_kpath_line_mode_mismatch_raises(self):
        """Verify ValueError when Line-mode KPathInfo has a different k-point count.

        Constructs a BandStructure with 10 k-points and an OrbitalProjection
        with matching dimensions, but a KPathInfo with ``num_kpoints=5`` in
        ``"Line-mode"``. Asserts that ``check_consistency`` raises
        ``ValueError`` matching ``"K-point count mismatch"``, covering the
        ``kpath.mode == "Line-mode"`` branch at helpers.py lines 283-287.
        """
        bands = make_band_structure(
            eigenvalues=jnp.zeros((10, 3)),
            kpoints=jnp.zeros((10, 3)),
        )
        orb = make_orbital_projection(
            projections=jnp.zeros((10, 3, 1, 9)),
        )
        kpath = make_kpath_info(
            num_kpoints=5,
            label_indices=[0, 4],
            mode="Line-mode",
        )
        with pytest.raises(ValueError, match="K-point count mismatch"):
            check_consistency(bands, orb, kpath)


class TestSelectAtomsWithOAM(chex.TestCase):
    """Test that select_atoms correctly propagates the OAM field."""

    def test_select_atoms_preserves_oam(self):
        """Verify OAM is sliced along the atom axis when selecting atoms.

        Constructs an OrbitalProjection with OAM shape (2, 2, 3, 3) and
        selects atoms 0 and 2. Asserts the resulting OAM has shape
        (2, 2, 2, 3), confirming helpers.py line 89 is executed when
        ``orb.oam is not None``.
        """
        proj = jnp.ones((2, 2, 3, 9), dtype=jnp.float64)
        oam = jnp.ones((2, 2, 3, 3), dtype=jnp.float64)
        orb = make_orbital_projection(projections=proj, oam=oam)
        sub = select_atoms(orb, [0, 2])
        assert sub.oam is not None
        chex.assert_shape(sub.oam, (2, 2, 2, 3))
