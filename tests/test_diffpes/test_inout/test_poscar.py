"""Validate VASP POSCAR parsing.

Covers VASP 4 and VASP 5 headers, direct and Cartesian coordinates, and
selective-dynamics records.
"""

import tempfile
from pathlib import Path

import chex
import jax.numpy as jnp

import diffpes
from diffpes.inout import (
    read_poscar,
)

_FIXTURES_DIR: Path = Path(__file__).resolve().parent / "fixtures"


class TestReadPoscar(chex.TestCase):
    """Validate :func:`diffpes.inout.read_poscar`.

    Covers VASP5 (species + Direct), VASP4 (Cartesian), and
    selective-dynamics POSCAR formats. Asserts lattice, coords,
    symbols, and atom_counts as appropriate.

    :see: :func:`~diffpes.inout.read_poscar`
    """

    def test_parses_vasp5_direct(self) -> None:
        """Read VASP-5 species and direct coordinates.

        Parses the default POSCAR fixture. Asserts lattice (3,3), positions
        (6,3), and the expanded per-atom species tuple.
        The test validates species line parsing and direct-coordinate scaling.

        Notes
        -----
        The test builds the inputs in the test body and checks the stated
        property with the documented numerical or structural assertions.
        """
        path: Path
        geom: diffpes.types.CrystalGeometry

        path = _FIXTURES_DIR / "POSCAR"
        geom = read_poscar(str(path))
        chex.assert_shape(geom.lattice, (3, 3))
        chex.assert_shape(geom.positions, (6, 3))
        assert geom.species == ("Si", "Si", "O", "O", "O", "O")

    def test_parses_vasp4_cartesian(self) -> None:
        """Read VASP-4 POSCAR with Cartesian coordinates and assert geometry.

        Parses POSCAR_cartesian (no species line). Asserts positions
        shape (2, 3) and empty species. Validates
        Cartesian path and single-species fallback.

        Notes
        -----
        The test builds the inputs in the test body and checks the stated
        property with the documented numerical or structural assertions.
        """
        path: Path
        geom: diffpes.types.CrystalGeometry

        path = _FIXTURES_DIR / "POSCAR_cartesian"
        geom = read_poscar(str(path))
        chex.assert_shape(geom.positions, (2, 3))
        assert geom.species == ()

    def test_parses_selective_dynamics(self) -> None:
        """Read POSCAR with Selective dynamics line and assert coordinates.

        Parses POSCAR_selective. Asserts positions shape (1, 3) and
        first coordinate [0, 0, 0]. The test verifies consumption of the
        selective-dynamics line. It also checks the parsed coordinates.

        Notes
        -----
        The test builds the inputs in the test body and checks the stated
        property with the documented numerical or structural assertions.
        """
        path: Path
        geom: diffpes.types.CrystalGeometry

        path = _FIXTURES_DIR / "POSCAR_selective"
        geom = read_poscar(str(path))
        chex.assert_shape(geom.positions, (1, 3))
        chex.assert_trees_all_close(
            geom.positions[0], jnp.array([0.0, 0.0, 0.0]), atol=1e-12
        )

    def test_negative_scale_sets_the_target_cell_volume(self) -> None:
        """Use a negative POSCAR scale as a positive target volume.

        A raw cubic cell with volume eight and a requested volume of 64 needs
        a linear scale of two. Cartesian coordinates must use the same scale
        before conversion to fractional coordinates.

        Notes
        -----
        Write one Cartesian site in a synthetic POSCAR. Check the positive
        lattice, target volume, and resulting fractional position.
        """
        directory: str
        with tempfile.TemporaryDirectory() as directory:
            path: Path = Path(directory) / "POSCAR-negative-scale"
            path.write_text(
                "negative scale\n"
                "-64.0\n"
                "2.0 0.0 0.0\n"
                "0.0 2.0 0.0\n"
                "0.0 0.0 2.0\n"
                "X\n"
                "1\n"
                "Cartesian\n"
                "1.0 2.0 3.0\n",
                encoding="utf-8",
            )
            geometry: diffpes.types.CrystalGeometry = read_poscar(str(path))

        chex.assert_trees_all_close(
            geometry.lattice,
            4.0 * jnp.eye(3),
            rtol=0.0,
            atol=1e-14,
        )
        chex.assert_trees_all_close(
            jnp.linalg.det(geometry.lattice),
            64.0,
            rtol=0.0,
            atol=1e-13,
        )
        chex.assert_trees_all_close(
            geometry.positions,
            jnp.asarray([[0.5, 1.0, 1.5]]),
            rtol=0.0,
            atol=1e-14,
        )
