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
    read_chgcar,
    read_doscar,
    read_eigenval,
    read_kpoints,
    read_poscar,
    read_procar,
)
from arpyes.types import (
    BandStructure,
    FullDensityOfStates,
    SOCVolumetricData,
    SpinBandStructure,
    SpinOrbitalProjection,
    VolumetricData,
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
        """Read spin-polarized EIGENVAL in legacy mode and get only spin-up."""
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
        """Read spin-polarized EIGENVAL in full mode to get both channels."""
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
        """In full mode, ISPIN=1 file still returns BandStructure."""
        path = _FIXTURES_DIR / "EIGENVAL"
        bands = read_eigenval(str(path), fermi_energy=0.0, return_mode="full")
        assert isinstance(bands, BandStructure)


class TestReadKpoints(chex.TestCase):
    """Tests for :func:`arpyes.inout.read_kpoints`.

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
        """Read Explicit KPOINTS with mode header and separate coord line."""
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

    def test_spin_procar_legacy(self):
        """Read spin-polarized PROCAR in legacy mode (first block only)."""
        path = _FIXTURES_DIR / "PROCAR_spin"
        orb = read_procar(str(path), return_mode="legacy")
        chex.assert_shape(orb.projections, (2, 2, 1, 9))
        # Should get first spin block values
        chex.assert_trees_all_close(
            orb.projections[0, 0, 0, 0], jnp.float64(0.1), atol=1e-12
        )
        assert orb.spin is None

    def test_spin_procar_full(self):
        """Read spin-polarized PROCAR in full mode returns SpinOrbitalProjection."""
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
    """Tests for :func:`arpyes.inout.read_doscar` with return_mode='full'."""

    def test_spin_doscar_full(self):
        """Read spin-polarized DOSCAR in full mode with both channels."""
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
        """Read non-spin DOSCAR in full mode."""
        path = _FIXTURES_DIR / "DOSCAR"
        dos = read_doscar(str(path), return_mode="full")
        assert isinstance(dos, FullDensityOfStates)
        chex.assert_shape(dos.energy, (5,))
        chex.assert_shape(dos.total_dos_up, (5,))
        assert dos.total_dos_down is None
        chex.assert_shape(dos.integrated_dos_up, (5,))
        assert dos.integrated_dos_down is None

    def test_pdos_doscar_full(self):
        """Read DOSCAR with PDOS blocks in full mode."""
        path = _FIXTURES_DIR / "DOSCAR_pdos"
        dos = read_doscar(str(path), return_mode="full")
        assert isinstance(dos, FullDensityOfStates)
        assert dos.pdos is not None
        # 2 atoms, 3 energy points, 10 data columns (9 orbitals + total)
        chex.assert_shape(dos.pdos, (2, 3, 10))
        assert dos.natoms == 2

    def test_spin_doscar_legacy(self):
        """Read spin DOSCAR in legacy mode returns DensityOfStates."""
        path = _FIXTURES_DIR / "DOSCAR_spin"
        dos = read_doscar(str(path), return_mode="legacy")
        # Legacy returns DensityOfStates with only spin-up
        chex.assert_shape(dos.energy, (5,))
        chex.assert_shape(dos.total_dos, (5,))
        chex.assert_trees_all_close(
            dos.total_dos[0], jnp.float64(0.10), atol=1e-12
        )


class TestReadChgcar(chex.TestCase):
    """Tests for :func:`arpyes.inout.read_chgcar`."""

    def test_charge_only(self):
        """Read charge-only CHGCAR returns VolumetricData."""
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
        """Read ISPIN=2 CHGCAR returns VolumetricData with magnetization."""
        path = _FIXTURES_DIR / "CHGCAR_spin"
        vol = read_chgcar(str(path))
        assert isinstance(vol, VolumetricData)
        assert vol.grid_shape == (2, 2, 2)
        chex.assert_shape(vol.charge, (2, 2, 2))
        assert vol.magnetization is not None
        chex.assert_shape(vol.magnetization, (2, 2, 2))

    def test_soc_chgcar(self):
        """Read SOC CHGCAR returns SOCVolumetricData with vector magnetization."""
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
