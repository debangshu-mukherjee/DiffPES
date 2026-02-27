"""Tests for VASP file readers.

Uses minimal fixture files under fixtures/ to exercise the parsing
paths of read_doscar, read_eigenval, read_kpoints, read_poscar,
and read_procar.
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
    """Tests for :func:`arpyes.inout.read_doscar`."""

    def test_parses_minimal_doscar(self):
        """Parse minimal DOSCAR fixture and check DensityOfStates."""
        path = _FIXTURES_DIR / "DOSCAR"
        dos = read_doscar(str(path))
        chex.assert_shape(dos.energy, (5,))
        chex.assert_shape(dos.total_dos, (5,))
        chex.assert_shape(dos.fermi_energy, ())
        chex.assert_trees_all_close(dos.fermi_energy, jnp.float64(0.5), atol=1e-12)
        chex.assert_trees_all_close(dos.energy[0], jnp.float64(-2.0), atol=1e-12)
        chex.assert_trees_all_close(dos.total_dos[2], jnp.float64(0.5), atol=1e-12)


class TestReadEigenval(chex.TestCase):
    """Tests for :func:`arpyes.inout.read_eigenval`."""

    def test_parses_minimal_eigenval(self):
        """Parse minimal EIGENVAL fixture (1 k-point, 1 band) and check BandStructure."""
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
        """Parse EIGENVAL with 2 k-points to cover multi-k-point loop branch."""
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
    """Tests for :func:`arpyes.inout.read_kpoints`."""

    def test_line_mode(self):
        """Parse Line-mode KPOINTS and check labels and indices."""
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
        """Parse Automatic (Monkhorst-Pack) KPOINTS."""
        path = _FIXTURES_DIR / "KPOINTS_auto"
        kpath = read_kpoints(str(path))
        assert kpath.mode == "Automatic"
        chex.assert_equal(kpath.num_kpoints, 0)

    def test_line_mode_label_fallback(self):
        """Parse Line-mode KPOINTS with 5-token line (label from last token) and 3-token line (no label)."""
        path = _FIXTURES_DIR / "KPOINTS_line_fallback"
        kpath = read_kpoints(str(path))
        assert kpath.mode == "Line-mode"
        assert "G" in kpath.labels
        assert "" in kpath.labels or len(kpath.labels) == 2

    def test_explicit_mode(self):
        """Parse Explicit KPOINTS."""
        path = _FIXTURES_DIR / "KPOINTS_explicit"
        kpath = read_kpoints(str(path))
        assert kpath.mode == "Explicit"
        chex.assert_equal(kpath.num_kpoints, 3)


class TestReadPoscar(chex.TestCase):
    """Tests for :func:`arpyes.inout.read_poscar`."""

    def test_parses_vasp5_direct(self):
        """Parse VASP-5 POSCAR with species and Direct coordinates."""
        path = _FIXTURES_DIR / "POSCAR"
        geom = read_poscar(str(path))
        chex.assert_shape(geom.lattice, (3, 3))
        chex.assert_shape(geom.coords, (6, 3))
        assert geom.symbols == ("Si", "O")
        chex.assert_trees_all_close(
            geom.atom_counts, jnp.array([2, 4], dtype=jnp.int32)
        )

    def test_parses_vasp4_cartesian(self):
        """Parse VASP-4 POSCAR with Cartesian coordinates."""
        path = _FIXTURES_DIR / "POSCAR_cartesian"
        geom = read_poscar(str(path))
        chex.assert_shape(geom.coords, (2, 3))
        assert geom.symbols == ()
        chex.assert_trees_all_close(
            geom.atom_counts, jnp.array([2], dtype=jnp.int32)
        )

    def test_parses_selective_dynamics(self):
        """Parse POSCAR with Selective dynamics line."""
        path = _FIXTURES_DIR / "POSCAR_selective"
        geom = read_poscar(str(path))
        chex.assert_shape(geom.coords, (1, 3))
        chex.assert_trees_all_close(
            geom.coords[0], jnp.array([0.0, 0.0, 0.0]), atol=1e-12
        )


class TestReadProcar(chex.TestCase):
    """Tests for :func:`arpyes.inout.read_procar`."""

    def test_parses_minimal_procar(self):
        """Parse minimal PROCAR and check OrbitalProjection."""
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
