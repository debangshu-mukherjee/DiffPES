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

from arpyes.inout import (
    read_doscar,
    read_eigenval,
    read_kpoints,
    read_poscar,
    read_procar,
)

_FIXTURES_DIR: Path = Path(__file__).resolve().parent / "fixtures"


class TestReadDoscar(chex.TestCase):
    """Tests for :func:`arpyes.inout.read_doscar`.

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
        chex.assert_trees_all_close(dos.fermi_energy, jnp.float64(0.5), atol=1e-12)
        chex.assert_trees_all_close(dos.energy[0], jnp.float64(-2.0), atol=1e-12)
        chex.assert_trees_all_close(dos.total_dos[2], jnp.float64(0.5), atol=1e-12)


class TestReadEigenval(chex.TestCase):
    """Tests for :func:`arpyes.inout.read_eigenval`.

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
        chex.assert_trees_all_close(bands.fermi_energy, jnp.float64(-0.5), atol=1e-12)
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


class TestReadKpoints(chex.TestCase):
    """Tests for :func:`arpyes.inout.read_kpoints`.

    Covers Line-mode (with and without label fallback), Automatic,
    and Explicit KPOINTS formats. Asserts mode string, num_kpoints,
    and presence of expected labels.
    """

    def test_line_mode(self):
        """Read Line-mode KPOINTS and assert mode, num_kpoints, and symmetry labels.

        Parses KPOINTS_line fixture. Asserts mode is "Line-mode",
        num_kpoints is 4, and labels include G, X, M with at least
        two labels and two label indices. Validates segment parsing
        and label extraction from lines with "! LABEL" convention.
        """
        path = _FIXTURES_DIR / "KPOINTS_line"
        kpath = read_kpoints(str(path))
        assert kpath.mode == "Line-mode"
        chex.assert_equal(kpath.num_kpoints, 4)
        assert "G" in kpath.labels
        assert "X" in kpath.labels
        assert "M" in kpath.labels
        assert len(kpath.labels) >= 2
        assert len(kpath.label_indices) >= 2

    def test_automatic_mode(self):
        """Read Automatic (Monkhorst-Pack) KPOINTS and assert mode and zero k-point count.

        Parses KPOINTS_auto. Asserts mode is "Automatic" and
        num_kpoints is 0, as required for grid-based sampling
        where VASP determines the grid size internally.
        """
        path = _FIXTURES_DIR / "KPOINTS_auto"
        kpath = read_kpoints(str(path))
        assert kpath.mode == "Automatic"
        chex.assert_equal(kpath.num_kpoints, 0)

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

    def test_explicit_mode(self):
        """Read Explicit KPOINTS and assert mode and k-point count.

        Parses KPOINTS_explicit. Asserts mode is "Explicit" and
        num_kpoints is 3, confirming that the header value is
        used when the file lists k-points explicitly.
        """
        path = _FIXTURES_DIR / "KPOINTS_explicit"
        kpath = read_kpoints(str(path))
        assert kpath.mode == "Explicit"
        chex.assert_equal(kpath.num_kpoints, 3)


class TestReadPoscar(chex.TestCase):
    """Tests for :func:`arpyes.inout.read_poscar`.

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
    """Tests for :func:`arpyes.inout.read_procar`.

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
