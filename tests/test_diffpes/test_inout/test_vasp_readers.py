"""Tests for VASP file readers.

Extended Summary
----------------
Exercises the VASP file parsing API: read_doscar, read_eigenval,
read_kpoints, read_poscar, and read_procar. Each reader is tested
against minimal but valid fixture files under fixtures/ so that
parsing logic, shape construction, and numeric values can be
asserted without external data. Tests cover Line-mode, Automatic,
and Explicit KPOINTS; VASP4 and VASP5 POSCAR formats; and
multi-k-point EIGENVAL and label-extraction branches in KPOINTS.
All test logic is documented in the docstrings of each class
and test method.

Routine Listings
----------------
:class:`TestReadDoscar`
    Tests for read_doscar.
:class:`TestReadEigenval`
    Tests for read_eigenval.
:class:`TestReadKpoints`
    Tests for read_kpoints.
:class:`TestReadPoscar`
    Tests for read_poscar.
:class:`TestReadProcar`
    Tests for read_procar.
"""

from pathlib import Path

import chex
import jax.numpy as jnp

import io
import tempfile

import pytest

from diffpes.inout import (
    read_chgcar,
    read_doscar,
    read_eigenval,
    read_kpoints,
    read_poscar,
    read_procar,
)
from diffpes.types import (
    BandStructure,
    FullDensityOfStates,
    SOCVolumetricData,
    SpinBandStructure,
    SpinOrbitalProjection,
    VolumetricData,
    make_orbital_projection,
    make_spin_orbital_projection,
)

_FIXTURES_DIR: Path = Path(__file__).resolve().parent / "fixtures"


class TestReadDoscar(chex.TestCase):
    """Tests for :func:`diffpes.inout.read_doscar`.

    Verifies that the DOSCAR parser produces a valid DensityOfStates
    PyTree with correct array shapes and expected numeric values from
    the minimal fixture.
    """

    def test_parses_minimal_doscar(self):
        """Read minimal DOSCAR fixture and assert shape and key values of DensityOfStates.

        Loads the fixtures/DOSCAR file and asserts that energy and
        total_dos have shape (5,), fermi_energy is scalar, and
        selected elements match the known fixture values (fermi 0.5,
        first energy -2.0, middle DOS 0.5). Ensures the parser
        correctly interprets the DOSCAR header and data block.
        """
        path = _FIXTURES_DIR / "DOSCAR"
        dos = read_doscar(str(path))
        chex.assert_shape(dos.energy, (5,))
        chex.assert_shape(dos.total_dos, (5,))
        chex.assert_shape(dos.fermi_energy, ())
        chex.assert_trees_all_close(
            dos.fermi_energy, jnp.float64(0.5), atol=1e-12
        )
        chex.assert_trees_all_close(
            dos.energy[0], jnp.float64(-2.0), atol=1e-12
        )
        chex.assert_trees_all_close(
            dos.total_dos[2], jnp.float64(0.5), atol=1e-12
        )


class TestReadEigenval(chex.TestCase):
    """Tests for :func:`diffpes.inout.read_eigenval`.

    Covers single- and multi-k-point EIGENVAL parsing, including
    the loop branch for multiple k-points, and asserts BandStructure
    shapes and eigenvalue/k-point values.
    """

    def test_parses_minimal_eigenval(self):
        """Read minimal EIGENVAL (1 k-point, 1 band) and assert BandStructure shape and values.

        Uses the minimal EIGENVAL fixture and fermi_energy=-0.5.
        Asserts eigenvalues shape (1, 1), kpoints (1, 3),
        kpoint_weights (1,), k-point [0,0,0], fermi -0.5, and
        eigenvalue -1.5. Validates header and per-k-point block parsing.
        """
        path = _FIXTURES_DIR / "EIGENVAL"
        bands = read_eigenval(str(path), fermi_energy=-0.5)
        chex.assert_shape(bands.eigenvalues, (1, 1))
        chex.assert_shape(bands.kpoints, (1, 3))
        chex.assert_shape(bands.kpoint_weights, (1,))
        chex.assert_trees_all_close(
            bands.kpoints[0], jnp.array([0.0, 0.0, 0.0]), atol=1e-12
        )
        chex.assert_trees_all_close(
            bands.fermi_energy, jnp.float64(-0.5), atol=1e-12
        )
        chex.assert_trees_all_close(
            bands.eigenvalues[0, 0], jnp.float64(-1.5), atol=1e-12
        )

    def test_parses_eigenval_two_kpoints(self):
        """Read EIGENVAL with 2 k-points and assert both k-points and eigenvalues.

        Uses EIGENVAL_two_kp fixture to exercise the parser's loop
        over multiple k-points (including the branch between k-point
        blocks). Asserts eigenvalues shape (2, 1), k-points at
        [0,0,0] and [0.5,0,0], and eigenvalues -1.0 and -0.5.
        """
        path = _FIXTURES_DIR / "EIGENVAL_two_kp"
        bands = read_eigenval(str(path), fermi_energy=0.0)
        chex.assert_shape(bands.eigenvalues, (2, 1))
        chex.assert_shape(bands.kpoints, (2, 3))
        chex.assert_trees_all_close(
            bands.kpoints[0], jnp.array([0.0, 0.0, 0.0]), atol=1e-12
        )
        chex.assert_trees_all_close(
            bands.kpoints[1], jnp.array([0.5, 0.0, 0.0]), atol=1e-12
        )
        chex.assert_trees_all_close(
            bands.eigenvalues[0, 0], jnp.float64(-1.0), atol=1e-12
        )
        chex.assert_trees_all_close(
            bands.eigenvalues[1, 0], jnp.float64(-0.5), atol=1e-12
        )

    def test_spin_polarized_legacy(self):
        """Read spin-polarized EIGENVAL in legacy mode and verify only spin-up eigenvalues.

        Parses the EIGENVAL_spin fixture with ``return_mode="legacy"``,
        which should discard spin-down data and return a plain
        ``BandStructure``. Asserts the result is a ``BandStructure``
        (not ``SpinBandStructure``), eigenvalues shape is (2, 2)
        (2 k-points, 2 bands), and the spin-up eigenvalues at k=0 are
        -1.5 and -0.5 respectively, verified to within ``atol=1e-12``.
        This exercises the legacy backward-compatibility path.
        """
        path = _FIXTURES_DIR / "EIGENVAL_spin"
        bands = read_eigenval(
            str(path), fermi_energy=0.0, return_mode="legacy"
        )
        assert isinstance(bands, BandStructure)
        chex.assert_shape(bands.eigenvalues, (2, 2))
        # spin-up eigenvalues sorted: [-1.5, -0.5] and [-1.0, -0.2]
        chex.assert_trees_all_close(
            bands.eigenvalues[0, 0], jnp.float64(-1.5), atol=1e-12
        )
        chex.assert_trees_all_close(
            bands.eigenvalues[0, 1], jnp.float64(-0.5), atol=1e-12
        )

    def test_spin_polarized_full(self):
        """Read spin-polarized EIGENVAL in full mode and verify both spin channels.

        Parses the EIGENVAL_spin fixture with ``return_mode="full"``,
        which returns a ``SpinBandStructure`` containing separate
        ``eigenvalues_up`` and ``eigenvalues_down`` arrays. Asserts the
        result type is ``SpinBandStructure``, both arrays have shape
        (2, 2), spin-up k=0 eigenvalues are [-1.5, -0.5], and spin-down
        k=0 eigenvalues are [-1.2, -0.3], all verified to within
        ``atol=1e-12``.
        """
        path = _FIXTURES_DIR / "EIGENVAL_spin"
        bands = read_eigenval(str(path), fermi_energy=0.0, return_mode="full")
        assert isinstance(bands, SpinBandStructure)
        chex.assert_shape(bands.eigenvalues_up, (2, 2))
        chex.assert_shape(bands.eigenvalues_down, (2, 2))
        # spin-up k=0: sorted [-1.5, -0.5]
        chex.assert_trees_all_close(
            bands.eigenvalues_up[0, 0], jnp.float64(-1.5), atol=1e-12
        )
        # spin-down k=0: sorted [-1.2, -0.3]
        chex.assert_trees_all_close(
            bands.eigenvalues_down[0, 0], jnp.float64(-1.2), atol=1e-12
        )
        chex.assert_trees_all_close(
            bands.eigenvalues_down[0, 1], jnp.float64(-0.3), atol=1e-12
        )

    def test_nonspin_full_returns_bandstructure(self):
        """Verify that full mode with a non-spin-polarized file returns plain BandStructure.

        Parses the standard EIGENVAL fixture (ISPIN=1) with
        ``return_mode="full"``. Asserts the result is a plain
        ``BandStructure`` rather than ``SpinBandStructure``, confirming
        that the full-mode path correctly detects single-spin data and
        falls back to the standard type.
        """
        path = _FIXTURES_DIR / "EIGENVAL"
        bands = read_eigenval(str(path), fermi_energy=0.0, return_mode="full")
        assert isinstance(bands, BandStructure)


class TestReadKpoints(chex.TestCase):
    """Tests for :func:`diffpes.inout.read_kpoints`.

    Covers Line-mode (with and without label fallback), Automatic,
    and Explicit KPOINTS formats. Asserts both legacy plotting fields
    (mode, labels, label_indices) and richer mode-specific metadata
    (grid/shift, explicit k-points/weights, line-mode endpoints).
    """

    def test_line_mode(self):
        """Read Line-mode KPOINTS and assert mode, num_kpoints, and symmetry labels.

        Parses KPOINTS_line fixture. Asserts mode is "Line-mode",
        num_kpoints is 4, labels include G/X/M, and line-mode metadata
        (segments, points_per_segment, endpoints, coordinate_mode,
        comment) are populated.
        """
        path = _FIXTURES_DIR / "KPOINTS_line"
        kpath = read_kpoints(str(path))
        assert kpath.mode == "Line-mode"
        chex.assert_equal(kpath.num_kpoints, 4)
        chex.assert_equal(kpath.points_per_segment, 2)
        chex.assert_equal(kpath.segments, 2)
        assert "G" in kpath.labels
        assert "X" in kpath.labels
        assert "M" in kpath.labels
        assert len(kpath.labels) >= 2
        assert len(kpath.label_indices) >= 2
        assert kpath.coordinate_mode.lower() == "reciprocal"
        assert kpath.comment == "k-path"
        chex.assert_shape(kpath.kpoints, (3, 3))
        chex.assert_trees_all_close(
            kpath.kpoints[0], jnp.array([0.0, 0.0, 0.0]), atol=1e-12
        )
        chex.assert_trees_all_close(
            kpath.kpoints[-1], jnp.array([0.5, 0.5, 0.0]), atol=1e-12
        )
        assert kpath.weights is None
        assert kpath.grid is None
        assert kpath.shift is None

    def test_automatic_mode(self):
        """Read Automatic (Monkhorst-Pack) KPOINTS and assert mode and zero k-point count.

        Parses KPOINTS_auto. Asserts mode is "Automatic", num_kpoints
        is 0, and automatic metadata (grid and shift) is populated.
        """
        path = _FIXTURES_DIR / "KPOINTS_auto"
        kpath = read_kpoints(str(path))
        assert kpath.mode == "Automatic"
        chex.assert_equal(kpath.num_kpoints, 0)
        chex.assert_trees_all_close(
            kpath.grid, jnp.array([4, 4, 4], dtype=jnp.int32), atol=0
        )
        chex.assert_trees_all_close(
            kpath.shift, jnp.array([0.0, 0.0, 0.0]), atol=1e-12
        )
        assert kpath.coordinate_mode.lower() == "monkhorst-pack"
        assert kpath.kpoints is None
        assert kpath.weights is None

    def test_line_mode_label_fallback(self):
        """Read Line-mode KPOINTS using fallback label extraction (no "!" prefix).

        Uses KPOINTS_line_fallback where one line has five tokens
        (coordinates plus weight and label "G") and another has three
        (no label). Asserts mode is "Line-mode", "G" appears in
        labels, and the empty or second label is present, exercising
        _extract_label branches for len(parts) > 4 and return "".
        """
        path = _FIXTURES_DIR / "KPOINTS_line_fallback"
        kpath = read_kpoints(str(path))
        assert kpath.mode == "Line-mode"
        assert "G" in kpath.labels
        assert "" in kpath.labels or len(kpath.labels) == 2
        chex.assert_equal(kpath.segments, 1)
        chex.assert_shape(kpath.kpoints, (2, 3))

    def test_explicit_mode(self):
        """Read Explicit KPOINTS and assert mode and k-point count.

        Parses KPOINTS_explicit. Asserts mode is "Explicit",
        num_kpoints is 3, and explicit metadata (k-points + weights)
        is parsed.
        """
        path = _FIXTURES_DIR / "KPOINTS_explicit"
        kpath = read_kpoints(str(path))
        assert kpath.mode == "Explicit"
        chex.assert_equal(kpath.num_kpoints, 3)
        chex.assert_shape(kpath.kpoints, (3, 3))
        chex.assert_shape(kpath.weights, (3,))
        chex.assert_trees_all_close(
            kpath.kpoints[1], jnp.array([0.5, 0.0, 0.0]), atol=1e-12
        )
        chex.assert_trees_all_close(
            kpath.weights,
            jnp.array([1.0, 0.5, 0.5], dtype=jnp.float64),
            atol=1e-12,
        )
        assert kpath.coordinate_mode.lower() == "cartesian"

    def test_explicit_mode_with_mode_header(self):
        """Read Explicit KPOINTS with an explicit mode header line and separate coordinate line.

        Parses KPOINTS_explicit_mode_header, which uses a distinct
        format where the mode is declared on a separate header line
        before the coordinate system line. Asserts mode is ``"Explicit"``,
        ``num_kpoints`` is 3, k-points shape is (3, 3), weights shape
        is (3,), the first k-point is [0, 0, 0], weights are
        [1.0, 0.5, 0.5], and coordinate mode is Cartesian. This exercises
        an alternative file layout branch in the KPOINTS parser.
        """
        path = _FIXTURES_DIR / "KPOINTS_explicit_mode_header"
        kpath = read_kpoints(str(path))
        assert kpath.mode == "Explicit"
        chex.assert_equal(kpath.num_kpoints, 3)
        chex.assert_shape(kpath.kpoints, (3, 3))
        chex.assert_shape(kpath.weights, (3,))
        chex.assert_trees_all_close(
            kpath.kpoints[0], jnp.array([0.0, 0.0, 0.0]), atol=1e-12
        )
        chex.assert_trees_all_close(
            kpath.weights,
            jnp.array([1.0, 0.5, 0.5], dtype=jnp.float64),
            atol=1e-12,
        )
        assert kpath.coordinate_mode.lower() == "cartesian"


class TestReadPoscar(chex.TestCase):
    """Tests for :func:`diffpes.inout.read_poscar`.

    Covers VASP5 (species + Direct), VASP4 (Cartesian), and
    selective-dynamics POSCAR formats. Asserts lattice, coords,
    symbols, and atom_counts as appropriate.
    """

    def test_parses_vasp5_direct(self):
        """Read VASP-5 POSCAR with species and Direct coordinates and assert geometry.

        Parses the default POSCAR fixture. Asserts lattice (3,3),
        coords (6,3), symbols ("Si", "O"), and atom_counts [2, 4].
        Validates species line parsing and direct-coordinate scaling.
        """
        path = _FIXTURES_DIR / "POSCAR"
        geom = read_poscar(str(path))
        chex.assert_shape(geom.lattice, (3, 3))
        chex.assert_shape(geom.coords, (6, 3))
        assert geom.symbols == ("Si", "O")
        chex.assert_trees_all_close(
            geom.atom_counts, jnp.array([2, 4], dtype=jnp.int32)
        )

    def test_parses_vasp4_cartesian(self):
        """Read VASP-4 POSCAR with Cartesian coordinates and assert geometry.

        Parses POSCAR_cartesian (no species line). Asserts coords
        shape (2, 3), empty symbols, and atom_counts [2]. Validates
        Cartesian path and single-species fallback.
        """
        path = _FIXTURES_DIR / "POSCAR_cartesian"
        geom = read_poscar(str(path))
        chex.assert_shape(geom.coords, (2, 3))
        assert geom.symbols == ()
        chex.assert_trees_all_close(
            geom.atom_counts, jnp.array([2], dtype=jnp.int32)
        )

    def test_parses_selective_dynamics(self):
        """Read POSCAR with Selective dynamics line and assert coordinates.

        Parses POSCAR_selective. Asserts coords shape (1, 3) and
        first coordinate [0, 0, 0]. Validates that the selective
        dynamics line is consumed and coordinates are still read
        correctly.
        """
        path = _FIXTURES_DIR / "POSCAR_selective"
        geom = read_poscar(str(path))
        chex.assert_shape(geom.coords, (1, 3))
        chex.assert_trees_all_close(
            geom.coords[0], jnp.array([0.0, 0.0, 0.0]), atol=1e-12
        )


class TestReadProcar(chex.TestCase):
    """Tests for :func:`diffpes.inout.read_procar`.

    Verifies that the PROCAR parser produces an OrbitalProjection
    with correct projection array shape and optional spin/oam
    absent when not present in the file.
    """

    def test_parses_minimal_procar(self):
        """Read minimal PROCAR and assert OrbitalProjection shape and sample values.

        Loads the minimal PROCAR fixture. Asserts projections shape
        (2, 2, 1, 9), selected projection values (0.1 and 0.18),
        and that spin and oam are None. Validates k-point/band/ion
        block parsing and orbital channel ordering.
        """
        path = _FIXTURES_DIR / "PROCAR"
        orb = read_procar(str(path))
        chex.assert_shape(orb.projections, (2, 2, 1, 9))
        chex.assert_trees_all_close(
            orb.projections[0, 0, 0, 0], jnp.float64(0.1), atol=1e-12
        )
        chex.assert_trees_all_close(
            orb.projections[1, 1, 0, 0], jnp.float64(0.18), atol=1e-12
        )
        assert orb.spin is None
        assert orb.oam is None

    def test_spin_procar_legacy(self):
        """Read spin-polarized PROCAR in legacy mode and verify only first spin block.

        Parses the PROCAR_spin fixture with ``return_mode="legacy"``,
        which extracts only the first (spin-up) block. Asserts
        projections shape is (2, 2, 1, 9), the s-orbital value at
        [0, 0, 0, 0] equals 0.1 (matching the fixture's spin-up data),
        and ``spin`` is ``None`` (no spin decomposition in legacy mode).
        This exercises the backward-compatible single-block extraction.
        """
        path = _FIXTURES_DIR / "PROCAR_spin"
        orb = read_procar(str(path), return_mode="legacy")
        chex.assert_shape(orb.projections, (2, 2, 1, 9))
        # Should get first spin block values
        chex.assert_trees_all_close(
            orb.projections[0, 0, 0, 0], jnp.float64(0.1), atol=1e-12
        )
        assert orb.spin is None

    def test_spin_procar_full(self):
        """Read spin-polarized PROCAR in full mode and verify SpinOrbitalProjection output.

        Parses the PROCAR_spin fixture with ``return_mode="full"``,
        returning a ``SpinOrbitalProjection`` with both ``projections``
        and ``spin`` arrays. Asserts the result type is
        ``SpinOrbitalProjection``, projections shape is (2, 2, 1, 9),
        spin shape is (2, 2, 1, 6), and the averaged s-orbital value
        at [0, 0, 0, 0] equals 0.09 (the mean of spin-up 0.1 and
        spin-down 0.08), verified to within ``atol=1e-12``.
        """
        path = _FIXTURES_DIR / "PROCAR_spin"
        orb = read_procar(str(path), return_mode="full")
        assert isinstance(orb, SpinOrbitalProjection)
        chex.assert_shape(orb.projections, (2, 2, 1, 9))
        chex.assert_shape(orb.spin, (2, 2, 1, 6))
        # Projections should be average of up and down
        # up s[0,0,0]=0.1, down s[0,0,0]=0.08 -> avg 0.09
        chex.assert_trees_all_close(
            orb.projections[0, 0, 0, 0], jnp.float64(0.09), atol=1e-12
        )


class TestReadDoscarFull(chex.TestCase):
    """Tests for :func:`diffpes.inout.read_doscar` with ``return_mode='full'``.

    Validates the full-mode DOSCAR parser that returns
    ``FullDensityOfStates`` objects containing spin-resolved total DOS,
    integrated DOS, and optionally per-atom projected DOS (PDOS).
    Covers spin-polarized files (both channels), non-spin files
    (spin-down fields are None), PDOS-containing files, and the
    legacy fallback for spin-polarized data.
    """

    def test_spin_doscar_full(self):
        """Read spin-polarized DOSCAR in full mode and verify both spin channels.

        Parses the DOSCAR_spin fixture with ``return_mode="full"``.
        Asserts the result is a ``FullDensityOfStates``, energy has
        shape (5,), both ``total_dos_up`` and ``total_dos_down`` have
        shape (5,) and are not None, both integrated DOS arrays have
        shape (5,), ``fermi_energy`` is 0.5, the first spin-up DOS
        value is 0.10, and the first spin-down DOS value is 0.08,
        all verified to within ``atol=1e-12``.
        """
        path = _FIXTURES_DIR / "DOSCAR_spin"
        dos = read_doscar(str(path), return_mode="full")
        assert isinstance(dos, FullDensityOfStates)
        chex.assert_shape(dos.energy, (5,))
        chex.assert_shape(dos.total_dos_up, (5,))
        assert dos.total_dos_down is not None
        chex.assert_shape(dos.total_dos_down, (5,))
        chex.assert_shape(dos.integrated_dos_up, (5,))
        assert dos.integrated_dos_down is not None
        chex.assert_shape(dos.integrated_dos_down, (5,))
        chex.assert_trees_all_close(
            dos.fermi_energy, jnp.float64(0.5), atol=1e-12
        )
        chex.assert_trees_all_close(
            dos.total_dos_up[0], jnp.float64(0.10), atol=1e-12
        )
        chex.assert_trees_all_close(
            dos.total_dos_down[0], jnp.float64(0.08), atol=1e-12
        )

    def test_nonspin_doscar_full(self):
        """Read non-spin-polarized DOSCAR in full mode and verify spin-down fields are None.

        Parses the standard DOSCAR fixture (non-spin-polarized) with
        ``return_mode="full"``. Asserts the result is a
        ``FullDensityOfStates``, energy and ``total_dos_up`` both have
        shape (5,), ``integrated_dos_up`` has shape (5,), and both
        ``total_dos_down`` and ``integrated_dos_down`` are ``None``,
        confirming the parser correctly detects non-spin data and
        leaves spin-down fields unset.
        """
        path = _FIXTURES_DIR / "DOSCAR"
        dos = read_doscar(str(path), return_mode="full")
        assert isinstance(dos, FullDensityOfStates)
        chex.assert_shape(dos.energy, (5,))
        chex.assert_shape(dos.total_dos_up, (5,))
        assert dos.total_dos_down is None
        chex.assert_shape(dos.integrated_dos_up, (5,))
        assert dos.integrated_dos_down is None

    def test_pdos_doscar_full(self):
        """Read DOSCAR containing per-atom PDOS blocks in full mode.

        Parses the DOSCAR_pdos fixture with ``return_mode="full"``.
        Asserts the result is a ``FullDensityOfStates``, ``pdos`` is
        not None, its shape is (2, 3, 10) corresponding to 2 atoms,
        3 energy points, and 10 data columns (9 orbital channels plus
        a total column), and ``natoms`` is 2. This validates the PDOS
        block parsing path which reads per-atom decomposed DOS after
        the total DOS header.
        """
        path = _FIXTURES_DIR / "DOSCAR_pdos"
        dos = read_doscar(str(path), return_mode="full")
        assert isinstance(dos, FullDensityOfStates)
        assert dos.pdos is not None
        # 2 atoms, 3 energy points, 10 data columns (9 orbitals + total)
        chex.assert_shape(dos.pdos, (2, 3, 10))
        assert dos.natoms == 2

    def test_spin_doscar_legacy(self):
        """Read spin-polarized DOSCAR in legacy mode and verify only spin-up is returned.

        Parses the DOSCAR_spin fixture with ``return_mode="legacy"``.
        Asserts the result is a plain ``DensityOfStates`` (not
        ``FullDensityOfStates``), energy has shape (5,), ``total_dos``
        has shape (5,), and the first total DOS value is 0.10
        (matching the spin-up channel), verified to within
        ``atol=1e-12``. This exercises the backward-compatible path
        that discards spin-down data.
        """
        path = _FIXTURES_DIR / "DOSCAR_spin"
        dos = read_doscar(str(path), return_mode="legacy")
        # Legacy returns DensityOfStates with only spin-up
        chex.assert_shape(dos.energy, (5,))
        chex.assert_shape(dos.total_dos, (5,))
        chex.assert_trees_all_close(
            dos.total_dos[0], jnp.float64(0.10), atol=1e-12
        )


class TestReadChgcar(chex.TestCase):
    """Tests for :func:`diffpes.inout.read_chgcar`.

    Validates the CHGCAR parser across three file variants: charge-only
    (single data block yielding ``VolumetricData``), ISPIN=2
    spin-polarized (two blocks yielding ``VolumetricData`` with scalar
    magnetization), and SOC (four blocks yielding
    ``SOCVolumetricData`` with vector magnetization). Asserts lattice,
    coordinate, and grid shapes, type discrimination, and numerical
    values of the volumetric data grids.
    """

    def test_charge_only(self):
        """Read charge-only CHGCAR and verify VolumetricData output with no magnetization.

        Parses CHGCAR_charge (single data block). Asserts the result
        type is ``VolumetricData``, lattice shape is (3, 3), coords
        shape is (1, 3), ``grid_shape`` is (2, 2, 2), charge shape is
        (2, 2, 2), magnetization is ``None``, symbols is ``("Si",)``,
        and ``atom_counts`` matches ``[1]``. This validates the
        single-block CHGCAR parsing path.
        """
        path = _FIXTURES_DIR / "CHGCAR_charge"
        vol = read_chgcar(str(path))
        assert isinstance(vol, VolumetricData)
        chex.assert_shape(vol.lattice, (3, 3))
        chex.assert_shape(vol.coords, (1, 3))
        assert vol.grid_shape == (2, 2, 2)
        chex.assert_shape(vol.charge, (2, 2, 2))
        assert vol.magnetization is None
        assert vol.symbols == ("Si",)
        chex.assert_trees_all_close(
            vol.atom_counts, jnp.array([1], dtype=jnp.int32)
        )

    def test_charge_with_magnetization(self):
        """Read ISPIN=2 CHGCAR and verify VolumetricData includes scalar magnetization.

        Parses CHGCAR_spin (two data blocks: charge and magnetization
        density). Asserts the result type is ``VolumetricData``,
        ``grid_shape`` is (2, 2, 2), charge shape is (2, 2, 2),
        magnetization is not ``None``, and magnetization shape is
        (2, 2, 2). This validates the two-block ISPIN=2 parsing path
        where magnetization = rho_up - rho_down.
        """
        path = _FIXTURES_DIR / "CHGCAR_spin"
        vol = read_chgcar(str(path))
        assert isinstance(vol, VolumetricData)
        assert vol.grid_shape == (2, 2, 2)
        chex.assert_shape(vol.charge, (2, 2, 2))
        assert vol.magnetization is not None
        chex.assert_shape(vol.magnetization, (2, 2, 2))

    def test_soc_chgcar(self):
        """Read SOC CHGCAR and verify SOCVolumetricData with 3-component magnetization.

        Parses CHGCAR_soc (four data blocks: charge, mx, my, mz).
        Asserts the result type is ``SOCVolumetricData``, ``grid_shape``
        is (2, 2, 2), charge shape is (2, 2, 2), scalar magnetization
        shape is (2, 2, 2), and ``magnetization_vector`` shape is
        (2, 2, 2, 3). Further verifies that the scalar magnetization
        equals the z-component (block 4), and that the first element
        of the x-component matches the expected volume-normalized value
        ``0.10 / 27.0``, both to within ``atol=1e-12``. This validates
        the four-block SOC parsing path and volume normalization.
        """
        path = _FIXTURES_DIR / "CHGCAR_soc"
        vol = read_chgcar(str(path))
        assert isinstance(vol, SOCVolumetricData)
        assert vol.grid_shape == (2, 2, 2)
        chex.assert_shape(vol.charge, (2, 2, 2))
        chex.assert_shape(vol.magnetization, (2, 2, 2))
        chex.assert_shape(vol.magnetization_vector, (2, 2, 2, 3))
        # magnetization should equal mz (4th block)
        chex.assert_trees_all_close(
            vol.magnetization,
            vol.magnetization_vector[..., 2],
            atol=1e-12,
        )
        # mx is block 2 (values 0.10-0.80), check first element
        # volume = 27.0, raw value 0.10, so grid value = 0.10/27
        chex.assert_trees_all_close(
            vol.magnetization_vector[0, 0, 0, 0],
            jnp.float64(0.10 / 27.0),
            atol=1e-12,
        )


# ---------------------------------------------------------------------------
# DOSCAR: PDOS with header line (covers doscar.py lines 204-221, 230)
# ---------------------------------------------------------------------------


class TestReadDoscarPdosHeader(chex.TestCase):
    """Tests for :func:`read_doscar` PDOS parsing with per-atom header lines."""

    def test_pdos_with_header_line(self):
        """DOSCAR with per-atom header line exercises doscar.py lines 204-221.

        Uses the DOSCAR_pdos_header fixture where each atom's PDOS block
        starts with a 4-value header line (EMIN EMAX NEDOS EFERMI).
        The condition ``_NONSPIN_COLS <= len(line_vals) <= _SPIN_COLS``
        is True (4 values), so lines 204-221 are executed. Asserts the
        PDOS array has the correct shape (1 atom, 3 energy points, 9
        orbital columns).
        """
        path = _FIXTURES_DIR / "DOSCAR_pdos_header"
        dos = read_doscar(str(path), return_mode="full")
        assert isinstance(dos, FullDensityOfStates)
        assert dos.pdos is not None
        # 1 atom, 3 energy points, 9 orbital columns (energy skipped)
        chex.assert_shape(dos.pdos, (1, 3, 9))

    def test_pdos_early_break_on_empty_line(self):
        """PDOS block with fewer data rows than NEDOS exercises doscar.py line 230.

        Writes a DOSCAR where natoms=1, NEDOS=3 but the PDOS block has
        only 1 data row followed by an empty line. This causes the
        ``for j in range(1, nedos):`` loop to hit an empty ``row_line``
        and execute the ``break`` on line 230. Asserts the result is
        a ``FullDensityOfStates`` (parse continues with truncated data).
        """
        content = (
            "         1\n"
            " unknown system\n"
            " 1.0\n"
            " 0.0 0.0 0.0\n"
            " 0.0 0.0 0.0\n"
            " -2.0 1.0 3 0.5\n"
            " -2.0 0.1 0.0\n"
            " -1.0 0.2 0.1\n"
            "  1.0 0.3 0.4\n"
            " -2.0 0.05 0.01 0.01 0.01 0.01 0.01 0.01 0.01 0.01\n"
            "\n"
        )
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".DOSCAR", delete=False
        ) as fh:
            fh.write(content)
            tmpname = fh.name
        try:
            dos = read_doscar(tmpname, return_mode="full")
            assert isinstance(dos, FullDensityOfStates)
        finally:
            import os

            os.unlink(tmpname)

    def test_pdos_header_then_empty_line_breaks(self):
        """PDOS block whose header is followed by an empty line exercises doscar.py line 206.

        Writes a DOSCAR with natoms=1, NEDOS=3 where the PDOS block has a
        4-value header line (satisfying ``_NONSPIN_COLS <= len <= _SPIN_COLS``)
        immediately followed by an empty line. The parser reads the header
        (line 204), then reads the empty line as ``pdos_ncols_check`` and
        hits ``if not pdos_ncols_check.strip(): break`` on line 205-206.
        Asserts the result is a ``FullDensityOfStates`` with ``pdos=None``
        (no PDOS data collected before the break).
        """
        content = (
            "         1\n"
            " unknown system\n"
            " 1.0\n"
            " 0.0 0.0 0.0\n"
            " 0.0 0.0 0.0\n"
            " -2.0 1.0 3 0.5\n"
            " -2.0 0.1 0.0\n"
            " -1.0 0.2 0.1\n"
            "  1.0 0.3 0.4\n"
            " -2.0 1.0 3 0.5\n"  # 4-value PDOS header -> goes into header branch
            "\n"  # empty pdos_ncols_check -> break on line 206
        )
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".DOSCAR", delete=False
        ) as fh:
            fh.write(content)
            tmpname = fh.name
        try:
            dos = read_doscar(tmpname, return_mode="full")
            assert isinstance(dos, FullDensityOfStates)
            assert dos.pdos is None
        finally:
            import os

            os.unlink(tmpname)

    def test_pdos_header_inner_loop_empty_row_breaks(self):
        """Empty data row inside PDOS-with-header loop exercises doscar.py line 218.

        Writes a DOSCAR with natoms=1, NEDOS=3 where the PDOS block starts
        with a 4-value header line, then a valid first data row, then an
        empty line for the second data row (j=1). The inner loop reads the
        empty ``row_line`` and hits ``if not row_line.strip(): break`` on
        line 217-218. Asserts the result is a ``FullDensityOfStates``.
        """
        content = (
            "         1\n"
            " unknown system\n"
            " 1.0\n"
            " 0.0 0.0 0.0\n"
            " 0.0 0.0 0.0\n"
            " -2.0 1.0 3 0.5\n"
            " -2.0 0.1 0.0\n"
            " -1.0 0.2 0.1\n"
            "  1.0 0.3 0.4\n"
            " -2.0 1.0 3 0.5\n"  # 4-value PDOS header -> header branch
            " -2.0 0.05 0.01 0.01 0.01 0.01 0.01 0.01 0.01 0.01\n"  # first data row (j=0)
            "\n"  # empty row for j=1 -> break on line 218
        )
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".DOSCAR", delete=False
        ) as fh:
            fh.write(content)
            tmpname = fh.name
        try:
            dos = read_doscar(tmpname, return_mode="full")
            assert isinstance(dos, FullDensityOfStates)
        finally:
            import os

            os.unlink(tmpname)


# ---------------------------------------------------------------------------
# PROCAR: empty file error, SOC 4-block parsing
# ---------------------------------------------------------------------------


class TestReadProcarErrors(chex.TestCase):
    """Tests for error handling in :func:`read_procar`."""

    def test_empty_procar_raises(self):
        """An empty PROCAR file raises ValueError (procar.py lines 155-156).

        Writes a file with no valid k-points block and calls
        ``read_procar``. Asserts a ``ValueError`` matching
        ``"No valid PROCAR blocks found"``.
        """
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".PROCAR", delete=False
        ) as fh:
            fh.write("PROCAR empty\n")
            tmpname = fh.name
        try:
            with pytest.raises(ValueError, match="No valid PROCAR blocks"):
                read_procar(tmpname)
        finally:
            import os

            os.unlink(tmpname)


class TestReadProcarSOC(chex.TestCase):
    """Tests for SOC PROCAR parsing (procar.py lines 191-210, 282-283, 307)."""

    def test_soc_procar_full(self):
        """Read SOC PROCAR (4 blocks) in full mode and verify spin components.

        Uses the PROCAR_soc fixture which has a title line before the
        first "k-points" block (covers lines 282-283), a blank line
        between the k-point header and the band header (covers line 307),
        and 4 blocks (covers lines 191-210). Asserts the result is a
        ``SpinOrbitalProjection`` with projections shape (1, 1, 1, 9)
        and spin shape (1, 1, 1, 6).
        """
        path = _FIXTURES_DIR / "PROCAR_soc"
        orb = read_procar(str(path), return_mode="full")
        assert isinstance(orb, SpinOrbitalProjection)
        chex.assert_shape(orb.projections, (1, 1, 1, 9))
        chex.assert_shape(orb.spin, (1, 1, 1, 6))
        # Total projections equal block-0 values
        chex.assert_trees_all_close(
            orb.projections[0, 0, 0, 0], jnp.float64(0.1), atol=1e-12
        )


# ---------------------------------------------------------------------------
# EIGENVAL: error paths (lines 169-184, 191-195, 255)
# ---------------------------------------------------------------------------


def _write_tmpfile(content: str) -> str:
    """Write content to a temporary file and return its path."""
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".EIGENVAL", delete=False
    ) as fh:
        fh.write(content)
        return fh.name


_EIGENVAL_HEADER: str = (
    "     1     1     1     1\n"
    "unknown\n"
    "  1.0  1.0  1.0\n"
    "  0  0  0\n"
    "  0  0  0\n"
    "  1     1     1\n"
    "\n"
)


class TestReadEigenvalErrors(chex.TestCase):
    """Error-path tests for :func:`read_eigenval` (lines 169-184, 191-195, 255)."""

    def _cleanup(self, path: str) -> None:
        import os

        os.unlink(path)

    def test_eof_on_kpoint_line_raises(self):
        """EOF while reading k-point block raises ValueError (lines 169-170, 255).

        Writes an EIGENVAL with a valid 6-line header followed by a blank
        separator but no k-point data line. ``_read_next_nonempty_line``
        returns ``""`` at EOF (line 255), then line 169 detects the empty
        string and raises ValueError.
        """
        content = _EIGENVAL_HEADER  # header only, no k-point data
        path = _write_tmpfile(content)
        try:
            with pytest.raises(ValueError, match="Unexpected EOF"):
                read_eigenval(path, fermi_energy=0.0)
        finally:
            self._cleanup(path)

    def test_invalid_kpoint_line_raises(self):
        """K-point line with < 4 values raises ValueError (lines 173-174).

        Writes an EIGENVAL where the first k-point line has only 2 values.
        Asserts ``ValueError`` matching ``"Invalid EIGENVAL k-point line"``.
        """
        content = _EIGENVAL_HEADER + "  0.0  0.0\n"
        path = _write_tmpfile(content)
        try:
            with pytest.raises(ValueError, match="Invalid EIGENVAL k-point line"):
                read_eigenval(path, fermi_energy=0.0)
        finally:
            self._cleanup(path)

    def test_eof_on_band_line_raises(self):
        """EOF while reading band line raises ValueError (lines 179-180, 255).

        Writes an EIGENVAL with a valid k-point line but no band data.
        ``_read_next_nonempty_line`` returns ``""`` at EOF (line 255),
        then line 179 detects it and raises ValueError.
        """
        content = _EIGENVAL_HEADER + "  0.0  0.0  0.0  1.0\n"
        path = _write_tmpfile(content)
        try:
            with pytest.raises(ValueError, match="Unexpected EOF"):
                read_eigenval(path, fermi_energy=0.0)
        finally:
            self._cleanup(path)

    def test_invalid_band_line_raises(self):
        """Band line with no numeric values raises ValueError (lines 183-184).

        Writes an EIGENVAL with a valid k-point line but an empty-looking
        band line that has no parseable eigenvalue. Asserts ``ValueError``
        matching ``"Invalid EIGENVAL band line"``.
        """
        content = _EIGENVAL_HEADER + "  0.0  0.0  0.0  1.0\n" + "  \n"
        # Actually need a non-empty but invalid line; use a word
        content2 = _EIGENVAL_HEADER + "  0.0  0.0  0.0  1.0\nNaN\n"
        path = _write_tmpfile(content2)
        try:
            with pytest.raises(ValueError, match="Invalid EIGENVAL band line"):
                read_eigenval(path, fermi_energy=0.0)
        finally:
            self._cleanup(path)

    def test_spin_polarized_band_missing_spin_down_raises(self):
        """Spin-polarized band line with only 1 value raises ValueError (lines 191-195).

        Writes an ISPIN=2 EIGENVAL (header has `2` as first value)
        where the band line has only the spin-up energy but no
        spin-down energy. Asserts ``ValueError`` matching
        ``"Invalid spin-polarized"``.
        """
        spin_header = (
            "     1     1     1     2\n"  # ISPIN=2
            "unknown\n"
            "  1.0  1.0  1.0\n"
            "  0  0  0\n"
            "  0  0  0\n"
            "  1     1     1\n"
            "\n"
        )
        content = spin_header + "  0.0  0.0  0.0  1.0\n" + "  1  -1.5\n"
        path = _write_tmpfile(content)
        try:
            with pytest.raises(ValueError, match="spin-polarized"):
                read_eigenval(path, fermi_energy=0.0, return_mode="full")
        finally:
            self._cleanup(path)


# ---------------------------------------------------------------------------
# KPOINTS: error paths (lines 227, 231-236, 241, 270-276, 307-308, 343-344, 375-376)
# ---------------------------------------------------------------------------


def _write_kpoints_tmpfile(content: str) -> str:
    """Write content to a temp KPOINTS file and return its path."""
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".KPOINTS", delete=False
    ) as fh:
        fh.write(content)
        return fh.name


class TestReadKpointsErrors(chex.TestCase):
    """Error-path and edge-case tests for :func:`read_kpoints`."""

    def _cleanup(self, path: str) -> None:
        import os

        os.unlink(path)

    def test_explicit_break_when_excess_lines(self):
        """Explicit KPOINTS with more lines than num_kpoints hits the break (line 227).

        Writes a 3-kpoint Explicit KPOINTS file with 4 coordinate lines.
        Asserts mode is 'Explicit' and only 3 k-points are returned (the
        4th line is ignored by the ``break`` at line 227).
        """
        content = (
            "Excess explicit kpoints\n"
            "3\n"
            "Cartesian\n"
            "  0.0  0.0  0.0  1.0\n"
            "  0.5  0.0  0.0  0.5\n"
            "  0.5  0.5  0.0  0.5\n"
            "  0.0  0.5  0.0  0.5\n"  # extra line
        )
        path = _write_kpoints_tmpfile(content)
        try:
            kpath = read_kpoints(path)
            assert kpath.mode == "Explicit"
            assert kpath.num_kpoints == 3
        finally:
            self._cleanup(path)

    def test_explicit_invalid_float_raises(self):
        """Explicit KPOINTS with non-numeric token raises ValueError (lines 231-233).

        Writes an Explicit KPOINTS with "abc 0.0 0.0" as a k-point line.
        Asserts ``ValueError`` matching ``"Invalid explicit KPOINTS"``.
        """
        content = (
            "Bad explicit\n"
            "2\n"
            "Cartesian\n"
            "  abc  0.0  0.0  1.0\n"
            "  0.5  0.0  0.0  0.5\n"
        )
        path = _write_kpoints_tmpfile(content)
        try:
            with pytest.raises(ValueError, match="Invalid explicit KPOINTS"):
                read_kpoints(path)
        finally:
            self._cleanup(path)

    def test_explicit_too_few_coords_raises(self):
        """Explicit KPOINTS with < 3 coordinates per line raises ValueError (lines 235-236).

        Writes an Explicit KPOINTS where a k-point line has only 2 values.
        Asserts ``ValueError`` matching ``"at least 3 coordinates"``.
        """
        content = (
            "Too few coords\n"
            "2\n"
            "Cartesian\n"
            "  0.0  0.0\n"
            "  0.5  0.0  0.0\n"
        )
        path = _write_kpoints_tmpfile(content)
        try:
            with pytest.raises(ValueError, match="at least 3 coordinates"):
                read_kpoints(path)
        finally:
            self._cleanup(path)

    def test_explicit_no_weight_column_defaults_to_one(self):
        """Explicit KPOINTS without a 4th weight column uses default weight 1.0 (line 241).

        Writes an Explicit KPOINTS where k-point lines have only 3
        columns (no weight). Asserts that all weights equal 1.0.
        """
        content = (
            "No weight\n"
            "2\n"
            "Cartesian\n"
            "  0.0  0.0  0.0\n"
            "  0.5  0.0  0.0\n"
        )
        path = _write_kpoints_tmpfile(content)
        try:
            kpath = read_kpoints(path)
            assert kpath.mode == "Explicit"
            assert kpath.weights is not None
            chex.assert_trees_all_close(
                kpath.weights,
                jnp.array([1.0, 1.0], dtype=jnp.float64),
                atol=1e-12,
            )
        finally:
            self._cleanup(path)

    def test_looks_like_kpoint_line_value_error_branch(self):
        """_looks_like_kpoint_line returns False for non-numeric 3-token lines (lines 270-276).

        Uses an Explicit KPOINTS where ``mode_line`` is not in
        ``_COORDINATE_MODE_TOKENS`` (mode = "Explicit") and the first
        remaining line has 3 tokens but the first is non-numeric.
        Asserts the mode header is consumed correctly as coord_mode.
        """
        content = (
            "Test non-numeric\n"
            "2\n"
            "Explicit\n"
            "Reciprocal 0.5 0.5\n"  # 3 tokens, first non-float -> looks_like=False
            "  0.0  0.0  0.0\n"
            "  0.5  0.0  0.0\n"
        )
        path = _write_kpoints_tmpfile(content)
        try:
            kpath = read_kpoints(path)
            assert kpath.mode == "Explicit"
            assert kpath.num_kpoints == 2
        finally:
            self._cleanup(path)

    def test_looks_like_kpoint_line_returns_true(self):
        """_looks_like_kpoint_line returns True when all three tokens are floats (lines 272-273, 276).

        Uses an Explicit KPOINTS where ``mode_line`` ("explicit") is not in
        ``_COORDINATE_MODE_TOKENS`` and the first remaining line has three
        numeric tokens ("0.0 0.0 0.0 1.0"). All three float() calls succeed
        (lines 272-273 executed), so the function reaches ``return True``
        (line 276). This means ``remaining_lines.pop(0)`` is NOT called and
        ``coord_mode`` stays as ``scheme_or_mode`` ("Explicit").
        """
        content = (
            "Test numeric first line\n"
            "2\n"
            "Explicit\n"
            "0.0 0.0 0.0 1.0\n"  # 3 float tokens -> looks_like=True -> no pop
            "0.5 0.0 0.0 0.5\n"
        )
        path = _write_kpoints_tmpfile(content)
        try:
            kpath = read_kpoints(path)
            assert kpath.mode == "Explicit"
            assert kpath.num_kpoints == 2
        finally:
            self._cleanup(path)

    def test_automatic_grid_too_few_values_raises(self):
        """Automatic KPOINTS grid line with < 3 values raises ValueError (lines 307-308).

        Writes an Automatic KPOINTS where the grid line has only 2
        integers. Asserts ``ValueError`` matching ``"3 values"``.
        """
        content = (
            "Bad auto\n"
            "0\n"
            "Monkhorst-Pack\n"
            "4 4\n"  # < 3 values
            "0 0 0\n"
        )
        path = _write_kpoints_tmpfile(content)
        try:
            with pytest.raises(ValueError, match="3 values"):
                read_kpoints(path)
        finally:
            self._cleanup(path)

    def test_automatic_shift_too_few_values_raises(self):
        """Automatic KPOINTS shift line with < 3 values raises ValueError (lines 343-344).

        Writes an Automatic KPOINTS where the shift line has only 2
        values. Asserts ``ValueError`` matching ``"3 values"``.
        """
        content = (
            "Bad auto shift\n"
            "0\n"
            "Monkhorst-Pack\n"
            "4 4 4\n"
            "0 0\n"  # < 3 values
        )
        path = _write_kpoints_tmpfile(content)
        try:
            with pytest.raises(ValueError, match="3 values"):
                read_kpoints(path)
        finally:
            self._cleanup(path)

    def test_line_mode_bad_coord_line_raises(self):
        """Line-mode KPOINTS with an unparseable coordinate raises ValueError (lines 375-376).

        Writes a Line-mode KPOINTS where a coordinate line has fewer
        than 3 float tokens. Asserts ``ValueError`` matching
        ``"Could not parse k-point coordinates"``.
        """
        content = (
            "Bad line mode\n"
            "2\n"
            "Line-mode\n"
            "Reciprocal\n"
            "  text only\n"  # < 3 float tokens
            "  0.5  0.5  0.5  M\n"
        )
        path = _write_kpoints_tmpfile(content)
        try:
            with pytest.raises(ValueError, match="Could not parse k-point"):
                read_kpoints(path)
        finally:
            self._cleanup(path)


# ---------------------------------------------------------------------------
# CHGCAR: error paths and edge cases (many lines)
# ---------------------------------------------------------------------------


def _write_chgcar_tmpfile(content: str) -> str:
    """Write content to a temp CHGCAR file and return its path."""
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".CHGCAR", delete=False
    ) as fh:
        fh.write(content)
        return fh.name


_CHGCAR_POSCAR_HEADER = (
    "Test system\n"
    " 1.0\n"
    "  3.0  0.0  0.0\n"
    "  0.0  3.0  0.0\n"
    "  0.0  0.0  3.0\n"
    "Si\n"
    "1\n"
    "Direct\n"
    "  0.0  0.0  0.0\n"
)


class TestReadChgcarErrors(chex.TestCase):
    """Error-path tests for :func:`read_chgcar` (chgcar.py lines 143-144, 150-151, etc.)."""

    def _cleanup(self, path: str) -> None:
        import os

        os.unlink(path)

    def test_zero_volume_lattice_raises(self):
        """Zero-volume lattice raises ValueError (chgcar.py lines 143-144).

        Writes a CHGCAR where the lattice vectors are all zero, making
        the unit cell volume = 0. Asserts ``ValueError`` matching
        ``"volume is zero"``.
        """
        content = (
            "Test system\n"
            " 1.0\n"
            "  0.0  0.0  0.0\n"  # zero lattice
            "  0.0  0.0  0.0\n"
            "  0.0  0.0  0.0\n"
            "Si\n"
            "1\n"
            "Direct\n"
            "  0.0  0.0  0.0\n"
            "\n"
            "   2   2   2\n"
            " 1.0 2.0 3.0 4.0\n"
            " 5.0 6.0 7.0 8.0\n"
        )
        path = _write_chgcar_tmpfile(content)
        try:
            with pytest.raises(ValueError, match="volume is zero"):
                read_chgcar(path)
        finally:
            self._cleanup(path)

    def test_no_grid_dimensions_raises(self):
        """Missing grid dimension line raises ValueError (chgcar.py lines 150-151).

        Writes a CHGCAR with a valid POSCAR header but no grid dimension
        line after the coordinates. Asserts ``ValueError`` matching
        ``"Could not locate"``.
        """
        content = _CHGCAR_POSCAR_HEADER  # no grid line follows
        path = _write_chgcar_tmpfile(content)
        try:
            with pytest.raises(ValueError, match="Could not locate"):
                read_chgcar(path)
        finally:
            self._cleanup(path)

    def test_invalid_lattice_line_raises(self):
        """Lattice line with < 3 values raises ValueError (chgcar.py lines 292-293).

        Writes a CHGCAR where the first lattice row has only 2 values.
        Asserts ``ValueError`` matching ``"Invalid CHGCAR lattice"``.
        """
        content = (
            "Test system\n"
            " 1.0\n"
            "  3.0  0.0\n"  # only 2 values
            "  0.0  3.0  0.0\n"
            "  0.0  0.0  3.0\n"
            "Si\n"
            "1\n"
            "Direct\n"
            "  0.0  0.0  0.0\n"
        )
        path = _write_chgcar_tmpfile(content)
        try:
            with pytest.raises(ValueError, match="Invalid CHGCAR lattice"):
                read_chgcar(path)
        finally:
            self._cleanup(path)

    def test_selective_dynamics_line_consumed(self):
        """Selective dynamics line is consumed before reading coordinates (chgcar.py line 307).

        Writes a CHGCAR with a 'Selective dynamics' line before the
        coordinate-mode line. Asserts the file is parsed successfully
        and the coord_line is correctly identified as 'Direct'.
        """
        content = (
            "Test system\n"
            " 1.0\n"
            "  3.0  0.0  0.0\n"
            "  0.0  3.0  0.0\n"
            "  0.0  0.0  3.0\n"
            "Si\n"
            "1\n"
            "Selective dynamics\n"  # triggers line 307
            "Direct\n"
            "  0.0  0.0  0.0\n"
            "\n"
            "   2   2   2\n"
            " 1.0 2.0 3.0 4.0\n"
            " 5.0 6.0 7.0 8.0\n"
        )
        path = _write_chgcar_tmpfile(content)
        try:
            vol = read_chgcar(path)
            chex.assert_shape(vol.charge, (2, 2, 2))
        finally:
            self._cleanup(path)

    def test_invalid_coordinate_line_raises(self):
        """Atomic coordinate line with < 3 values raises ValueError (chgcar.py lines 316-317).

        Writes a CHGCAR where the atomic coordinate line has only 2
        values. Asserts ``ValueError`` matching ``"Invalid CHGCAR coordinate"``.
        """
        content = (
            "Test system\n"
            " 1.0\n"
            "  3.0  0.0  0.0\n"
            "  0.0  3.0  0.0\n"
            "  0.0  0.0  3.0\n"
            "Si\n"
            "1\n"
            "Direct\n"
            "  0.0  0.0\n"  # only 2 values
        )
        path = _write_chgcar_tmpfile(content)
        try:
            with pytest.raises(ValueError, match="Invalid CHGCAR coordinate"):
                read_chgcar(path)
        finally:
            self._cleanup(path)

    def test_cartesian_coordinates_transform(self):
        """Cartesian coordinates are correctly converted to fractional (chgcar.py lines 321-322).

        Writes a CHGCAR with Cartesian coordinate mode. The atom at
        [3.0, 0.0, 0.0] Cartesian in a [3,3,3] lattice should map to
        fractional [1.0, 0.0, 0.0]. Asserts the parsed coordinate
        equals [1.0, 0.0, 0.0] to within atol=1e-10.
        """
        content = (
            "Test system\n"
            " 1.0\n"
            "  3.0  0.0  0.0\n"
            "  0.0  3.0  0.0\n"
            "  0.0  0.0  3.0\n"
            "Si\n"
            "1\n"
            "Cartesian\n"  # triggers lines 321-322
            "  3.0  0.0  0.0\n"  # Cartesian [3,0,0] -> fractional [1,0,0]
            "\n"
            "   2   2   2\n"
            " 1.0 2.0 3.0 4.0\n"
            " 5.0 6.0 7.0 8.0\n"
        )
        path = _write_chgcar_tmpfile(content)
        try:
            vol = read_chgcar(path)
            chex.assert_trees_all_close(
                vol.coords[0], jnp.array([1.0, 0.0, 0.0]), atol=1e-10
            )
        finally:
            self._cleanup(path)

    def test_find_next_grid_skips_non_matching_lines(self):
        """_find_next_grid_line skips lines with != 3 parts and non-int 3-part lines.

        Writes a CHGCAR where the data section contains lines with != 3
        parts (covers chgcar.py line 374) and a 3-float line like
        "1.5 2.5 3.5" (covers lines 381-382, ValueError -> continue)
        before the actual grid dimensions.
        """
        content = (
            _CHGCAR_POSCAR_HEADER
            + "\n"
            + "   2   2   2\n"
            + " 1.0 2.0 3.0 4.0\n"
            + " 5.0 6.0 7.0 8.0\n"
            + "\n"
            + "some augmentation text line here\n"  # != 3 parts -> line 374
            + "1.5 2.5 3.5\n"  # 3 non-int floats -> line 381-382
            + "   2   2   2\n"  # actual second grid
            + " 0.1 0.2 0.3 0.4\n"
            + " 0.5 0.6 0.7 0.8\n"
        )
        path = _write_chgcar_tmpfile(content)
        try:
            vol = read_chgcar(path)
            assert isinstance(vol, VolumetricData)
            assert vol.magnetization is not None
        finally:
            self._cleanup(path)

    def test_parse_float_block_skips_blank_lines(self):
        """_parse_float_block skips blank lines in data (chgcar.py lines 449-450).

        Writes a CHGCAR where the data block has a blank line after the
        first row of values. Asserts the grid is parsed correctly.
        """
        content = (
            _CHGCAR_POSCAR_HEADER
            + "\n"
            + "   2   2   2\n"
            + " 1.0 2.0 3.0 4.0\n"
            + "\n"  # blank line inside data block -> lines 449-450
            + " 5.0 6.0 7.0 8.0\n"
        )
        path = _write_chgcar_tmpfile(content)
        try:
            vol = read_chgcar(path)
            chex.assert_shape(vol.charge, (2, 2, 2))
        finally:
            self._cleanup(path)

    def test_parse_float_block_stops_on_non_float_token(self):
        """Non-float token in data triggers row_valid=False (chgcar.py lines 458-460).

        Writes a CHGCAR where an augmentation header line appears BETWEEN
        the two rows of float data (after 4 values, before the remaining
        4). The non-float token sets ``row_valid=False`` at line 458,
        breaks the inner token loop (line 459), and skips appending
        (line 460 is just after the except; the outer loop still advances
        idx and the subsequent valid rows complete the 8-value collection).
        """
        content = (
            _CHGCAR_POSCAR_HEADER
            + "\n"
            + "   2   2   2\n"
            + " 1.0 2.0 3.0 4.0\n"
            + "augmentation occupancies\n"  # non-float mid-block -> lines 458-460
            + " 5.0 6.0 7.0 8.0\n"
        )
        path = _write_chgcar_tmpfile(content)
        try:
            vol = read_chgcar(path)
            chex.assert_shape(vol.charge, (2, 2, 2))
        finally:
            self._cleanup(path)

    def test_parse_float_block_truncated_raises(self):
        """Truncated data block raises ValueError (chgcar.py lines 467-468).

        Writes a CHGCAR where the grid is 2x2x2 (needs 8 values) but
        only 4 values are present. Asserts ``ValueError`` matching
        ``"Unexpected end of CHGCAR data block"``.
        """
        content = (
            _CHGCAR_POSCAR_HEADER
            + "\n"
            + "   2   2   2\n"
            + " 1.0 2.0 3.0 4.0\n"  # only 4 values, needs 8
        )
        path = _write_chgcar_tmpfile(content)
        try:
            with pytest.raises(ValueError, match="Unexpected end of CHGCAR"):
                read_chgcar(path)
        finally:
            self._cleanup(path)
