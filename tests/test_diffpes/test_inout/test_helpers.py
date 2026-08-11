"""Validate parser-adjacent workflow helpers.

Extended Summary
----------------
The tests validate atom selection, atom aggregation, shell reduction, and
dimension checks across files. They use synthetic projection carriers.
"""

import chex
import jax.numpy as jnp
import pytest
from beartype import beartype
from jaxtyping import Array, Float64, jaxtyped

import diffpes
from diffpes.inout import (
    aggregate_atoms,
    check_consistency,
    reduce_orbitals,
    select_atoms,
)
from diffpes.types import (
    OrbitalProjection,
    SpinOrbitalProjection,
    make_band_structure,
    make_kpath_info,
    make_orbital_projection,
    make_spin_orbital_projection,
)


@jaxtyped(typechecker=beartype)
def _make_test_orb() -> OrbitalProjection:
    """PRIVATE: Create a deterministic OrbitalProjection fixture.

    The helper constructs an ``OrbitalProjection`` with shape ``(2, 2, 3, 9)``.
    Each atom has a distinct s-orbital value. All atoms share fixed p-orbital
    and d-orbital values. This fixture provides deterministic data for
    selection, aggregation, and reduction.

    Returns
    -------
    OrbitalProjection
        PyTree with ``projections`` of shape (2, 2, 3, 9) and no spin
        or OAM data.
    """
    proj: Float64[Array, "2 2 3 9"] = jnp.zeros(
        (2, 2, 3, 9), dtype=jnp.float64
    )
    proj = proj.at[:, :, 0, 0].set(1.0)
    proj = proj.at[:, :, 1, 0].set(2.0)
    proj = proj.at[:, :, 2, 0].set(3.0)
    proj = proj.at[:, :, :, 1].set(0.1)
    proj = proj.at[:, :, :, 2].set(0.2)
    proj = proj.at[:, :, :, 3].set(0.3)
    proj = proj.at[:, :, :, 4].set(0.01)
    proj = proj.at[:, :, :, 5].set(0.02)
    proj = proj.at[:, :, :, 6].set(0.03)
    proj = proj.at[:, :, :, 7].set(0.04)
    proj = proj.at[:, :, :, 8].set(0.05)
    orbital_projection: OrbitalProjection = make_orbital_projection(
        projections=proj
    )
    return orbital_projection


class TestSelectAtoms(chex.TestCase):
    """Validate :func:`diffpes.inout.select_atoms`.

    The tests validate atom-axis slicing of both projection carriers.
    They cover one atom and multiple atoms while preserving values.
    They also verify the ``SpinOrbitalProjection`` type and both sliced arrays.

    :see: :func:`~diffpes.inout.select_atoms`
    """

    def test_select_single_atom(self) -> None:
        """Select one atom and verify its shape and s-orbital value.

        The fixture has three atoms with distinct s-orbital weights.
        The test selects atom 1 and checks the reduced atom axis.
        It compares the selected s-orbital value with 2.0 by using
        ``atol=1e-12``.

        Notes
        -----
        The test builds the inputs in the test body and checks the stated
        property with the documented numerical or structural assertions.
        """
        orb: (
            diffpes.types.OrbitalProjection
            | diffpes.types.SpinOrbitalProjection
        )
        sub: (
            diffpes.types.OrbitalProjection
            | diffpes.types.SpinOrbitalProjection
        )

        orb = _make_test_orb()
        sub = select_atoms(orb, [1])
        chex.assert_shape(sub.projections, (2, 2, 1, 9))
        chex.assert_trees_all_close(
            sub.projections[0, 0, 0, 0], jnp.float64(2.0), atol=1e-12
        )

    def test_select_multiple_atoms(self) -> None:
        """Select two non-contiguous atoms and verify shape and ordering.

        The test selects atoms 0 and 2 from the three-atom fixture.
        It checks the output shape and selection order. Output indices 0 and 1
        contain s-orbital values 1.0 and 3.0. The comparisons use
        ``atol=1e-12``.

        Notes
        -----
        The test builds the inputs in the test body and checks the stated
        property with the documented numerical or structural assertions.
        """
        orb: (
            diffpes.types.OrbitalProjection
            | diffpes.types.SpinOrbitalProjection
        )
        sub: (
            diffpes.types.OrbitalProjection
            | diffpes.types.SpinOrbitalProjection
        )

        orb = _make_test_orb()
        sub = select_atoms(orb, [0, 2])
        chex.assert_shape(sub.projections, (2, 2, 2, 9))
        chex.assert_trees_all_close(
            sub.projections[0, 0, 0, 0], jnp.float64(1.0), atol=1e-12
        )
        chex.assert_trees_all_close(
            sub.projections[0, 0, 1, 0], jnp.float64(3.0), atol=1e-12
        )

    def test_preserves_spin_orbital_projection_type(self) -> None:
        """Preserve the SpinOrbitalProjection type after selection.

        The test constructs a SpinOrbitalProjection with ``projections`` shape
        (2, 2, 3, 9) and ``spin`` shape (2, 2, 3, 6), then selects atoms
        0 and 2. The test verifies the ``SpinOrbitalProjection`` result type.
        It also verifies the two-atom shapes of both arrays. This test checks
        dispatch for the input subtype.

        Notes
        -----
        The test builds the inputs in the test body and checks the stated
        property with the documented numerical or structural assertions.
        """
        proj: Float64[Array, "2 2 3 9"]
        spin: Float64[Array, "2 2 3 6"]
        orb: diffpes.types.SpinOrbitalProjection
        sub: (
            diffpes.types.OrbitalProjection
            | diffpes.types.SpinOrbitalProjection
        )

        proj = jnp.ones((2, 2, 3, 9), dtype=jnp.float64)
        spin = jnp.ones((2, 2, 3, 6), dtype=jnp.float64)
        orb = make_spin_orbital_projection(projections=proj, spin=spin)
        sub = select_atoms(orb, [0, 2])
        assert isinstance(sub, SpinOrbitalProjection)
        chex.assert_shape(sub.projections, (2, 2, 2, 9))
        chex.assert_shape(sub.spin, (2, 2, 2, 6))


class TestAggregateAtoms(chex.TestCase):
    """Validate :func:`diffpes.inout.aggregate_atoms`.

    The tests validate the sum of orbital projections over the atom axis.
    They cover all atoms and an explicit subset. They verify the output shape
    and the summed s-orbital values.

    :see: :func:`~diffpes.inout.aggregate_atoms`
    """

    def test_aggregate_all(self) -> None:
        """Sum projections over all atoms and verify collapsed shape and total.

        The test calls ``aggregate_atoms`` without atom indices.
        It checks the output shape after removal of the atom axis.
        It compares the summed s-orbital value with 6.0 by using
        ``atol=1e-12``.

        Notes
        -----
        The test builds the inputs in the test body and checks the stated
        property with the documented numerical or structural assertions.
        """
        orb: (
            diffpes.types.OrbitalProjection
            | diffpes.types.SpinOrbitalProjection
        )
        agg: (
            diffpes.types.OrbitalProjection
            | diffpes.types.SpinOrbitalProjection
        )

        orb = _make_test_orb()
        agg = aggregate_atoms(orb)
        chex.assert_shape(agg, (2, 2, 9))

        chex.assert_trees_all_close(agg[0, 0, 0], jnp.float64(6.0), atol=1e-12)

    def test_aggregate_subset(self) -> None:
        """Sum projections over a specified subset of atoms.

        The test calls ``aggregate_atoms`` with atom indices 0 and 1.
        It checks the output shape and compares the s-orbital value with 3.0.
        The comparison uses ``atol=1e-12``. This input covers explicit
        atom selection before the sum.

        Notes
        -----
        The test builds the inputs in the test body and checks the stated
        property with the documented numerical or structural assertions.
        """
        orb: (
            diffpes.types.OrbitalProjection
            | diffpes.types.SpinOrbitalProjection
        )
        agg: (
            diffpes.types.OrbitalProjection
            | diffpes.types.SpinOrbitalProjection
        )

        orb = _make_test_orb()
        agg = aggregate_atoms(orb, [0, 1])
        chex.assert_shape(agg, (2, 2, 9))

        chex.assert_trees_all_close(agg[0, 0, 0], jnp.float64(3.0), atol=1e-12)


class TestReduceOrbitals(chex.TestCase):
    """Validate :func:`diffpes.inout.reduce_orbitals`.

    The tests reduce nine orbital channels to aggregate s, p, and d channels.
    Each aggregate channel sums the applicable input channels.

    :see: :func:`~diffpes.inout.reduce_orbitals`
    """

    def test_reduces_to_spd(self) -> None:
        """Reduce nine orbital channels to s/p/d totals.

        The test applies ``reduce_orbitals`` to the fixture's raw projections
        array.
        The test asserts the output shape is (2, 2, 3, 3) -- 9 orbital channels
        collapsed to 3 -- and checks atom 0's reduced values analytically:

        The expected s, p, and d totals are 1.0, 0.6, and 0.15.

        All comparisons use ``atol=1e-12``.

        Notes
        -----
        The test builds the inputs in the test body and checks the stated
        property with the documented numerical or structural assertions.
        """
        orb: (
            diffpes.types.OrbitalProjection
            | diffpes.types.SpinOrbitalProjection
        )
        reduced: (
            diffpes.types.OrbitalProjection
            | diffpes.types.SpinOrbitalProjection
        )

        orb = _make_test_orb()
        reduced = reduce_orbitals(orb.projections)
        chex.assert_shape(reduced, (2, 2, 3, 3))

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
    """Validate :func:`diffpes.inout.check_consistency`.

    Validates the dimension-compatibility checker that ensures
    BandStructure, OrbitalProjection, and (optionally) KPathInfo
    agree on k-point and band counts. Covers the happy path (all
    dimensions match), k-point mismatch, band mismatch, and the
    optional KPathInfo consistency check.

    :see: :func:`~diffpes.inout.check_consistency`
    """

    def test_consistent_inputs(self) -> None:
        """Verify successful checks for matching band and projection carriers.

        The test constructs a BandStructure with 2 k-points and 3 bands, and an
        OrbitalProjection with matching leading dimensions (2, 3, 1, 9).
        The test calls ``check_consistency`` and asserts it returns without
        raising, confirming the positive validation path.

        Notes
        -----
        The test builds the inputs in the test body and checks the stated
        property with the documented numerical or structural assertions.
        """
        bands: diffpes.types.BandStructure
        orb: diffpes.types.OrbitalProjection

        bands = make_band_structure(
            eigenvalues=jnp.zeros((2, 3)),
            kpoints=jnp.zeros((2, 3)),
        )
        orb = make_orbital_projection(
            projections=jnp.zeros((2, 3, 1, 9)),
        )
        check_consistency(bands, orb)

    def test_kpoint_mismatch(self) -> None:
        """Verify ``ValueError`` for different k-point counts.

        The test constructs a BandStructure with 2 k-points but an
        OrbitalProjection with 4 k-points. Asserts that
        ``check_consistency`` raises ``ValueError`` matching
        ``"K-point count mismatch"``, exercising the first dimension
        check.

        Notes
        -----
        The test builds the inputs in the test body and checks the stated
        property with the documented numerical or structural assertions.
        """
        bands: diffpes.types.BandStructure
        orb: diffpes.types.OrbitalProjection

        bands = make_band_structure(
            eigenvalues=jnp.zeros((2, 3)),
            kpoints=jnp.zeros((2, 3)),
        )
        orb = make_orbital_projection(
            projections=jnp.zeros((4, 3, 1, 9)),
        )
        with pytest.raises(ValueError, match="K-point count mismatch"):
            check_consistency(bands, orb)

    def test_band_mismatch(self) -> None:
        """Verify ``ValueError`` for different band counts.

        The test constructs a BandStructure with 3 bands but an
        OrbitalProjection with 5 bands (k-points match at 2). Asserts
        that ``check_consistency`` raises ``ValueError`` matching
        ``"Band count mismatch"``, exercising the second dimension
        check independently from the k-point check.

        Notes
        -----
        The test builds the inputs in the test body and checks the stated
        property with the documented numerical or structural assertions.
        """
        bands: diffpes.types.BandStructure
        orb: diffpes.types.OrbitalProjection

        bands = make_band_structure(
            eigenvalues=jnp.zeros((2, 3)),
            kpoints=jnp.zeros((2, 3)),
        )
        orb = make_orbital_projection(
            projections=jnp.zeros((2, 5, 1, 9)),
        )
        with pytest.raises(ValueError, match="Band count mismatch"):
            check_consistency(bands, orb)

    def test_with_kpath(self) -> None:
        """Verify consistency with an optional ``KPathInfo``.

        The test constructs matching band, projection, and k-path carriers.
        The carriers have ten k-points and three bands. The test calls
        ``check_consistency`` and expects no error. This result confirms that
        all three dimensions agree.

        Notes
        -----
        The test builds the inputs in the test body and checks the stated
        property with the documented numerical or structural assertions.
        """
        bands: diffpes.types.BandStructure
        orb: diffpes.types.OrbitalProjection
        kpath: diffpes.types.KPathInfo

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
            segments=1,
            mode="Line-mode",
        )
        check_consistency(bands, orb, kpath)

    def test_kpath_line_mode_mismatch_raises(self) -> None:
        """Reject a mismatched line-mode k-point count.

        The test constructs a BandStructure with 10 k-points and a matching
        OrbitalProjection, but a KPathInfo with ``num_kpoints=5`` in
        ``"Line-mode"``. Asserts that ``check_consistency`` raises
        ``ValueError`` matching ``"K-point count mismatch"``, covering the
        ``kpath.mode == "Line-mode"`` branch at helpers.py lines 283-287.

        Notes
        -----
        The test builds the inputs in the test body and checks the stated
        property with the documented numerical or structural assertions.
        """
        bands: diffpes.types.BandStructure
        orb: diffpes.types.OrbitalProjection
        kpath: diffpes.types.KPathInfo

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
            segments=1,
            mode="Line-mode",
        )
        with pytest.raises(ValueError, match="K-point count mismatch"):
            check_consistency(bands, orb, kpath)


class TestSelectAtomsWithOAM(chex.TestCase):
    """Test that select_atoms correctly propagates the OAM field.

    :see: :func:`~diffpes.inout.select_atoms`
    """

    def test_select_atoms_preserves_oam(self) -> None:
        """Verify OAM slicing along the atom axis during selection.

        The test constructs an OrbitalProjection with OAM shape (2, 2, 3, 3)
        and selects atoms 0 and 2. The test checks the resulting OAM shape.
        This input covers the path where ``orb.oam`` is not ``None``.

        Notes
        -----
        The test builds the inputs in the test body and checks the stated
        property with the documented numerical or structural assertions.
        """
        proj: Float64[Array, "2 2 3 9"]
        oam: Float64[Array, "2 2 3 3"]
        orb: diffpes.types.OrbitalProjection
        sub: (
            diffpes.types.OrbitalProjection
            | diffpes.types.SpinOrbitalProjection
        )

        proj = jnp.ones((2, 2, 3, 9), dtype=jnp.float64)
        oam = jnp.ones((2, 2, 3, 3), dtype=jnp.float64)
        orb = make_orbital_projection(projections=proj, oam=oam)
        sub = select_atoms(orb, [0, 2])
        assert sub.oam is not None
        chex.assert_shape(sub.oam, (2, 2, 2, 3))
