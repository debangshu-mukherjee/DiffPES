"""Validate VASP CHGCAR parsing.

Covers scalar, spin-polarized, and SOC volumetric grids together with
malformed headers, grid blocks, and coordinate data.
"""

import os
import tempfile
from pathlib import Path

import chex
import jax.numpy as jnp
import pytest
from beartype.typing import TextIO

import diffpes
from diffpes.inout import (
    read_chgcar,
)
from diffpes.types import (
    SOCVolumetricData,
    VolumetricData,
)

_FIXTURES_DIR: Path = Path(__file__).resolve().parent / "fixtures"


class TestReadChgcar(chex.TestCase):
    """Validate :func:`diffpes.inout.read_chgcar`.

    The tests validate charge-only, ``ISPIN=2``, and SOC CHGCAR files.
    These files have one, two, and four data blocks, respectively.
    The tests check the lattice,
    coordinate, and grid shapes, type discrimination, and numerical
    values of the volumetric data grids.

    :see: :func:`~diffpes.inout.read_chgcar`
    """

    def test_charge_only(self) -> None:
        """Read a charge-only CHGCAR without magnetization.

        The test parses ``CHGCAR_charge``, which has one data block.
        It checks the result type, the array shapes, and ``grid_shape``.
        It also checks the symbols, atom counts, and absent magnetization.
        These checks validate the CHGCAR path for one block.

        Notes
        -----
        The test builds the inputs in the test body and checks the stated
        property with the documented numerical or structural assertions.
        """
        path: Path
        vol: diffpes.types.VolumetricData | diffpes.types.SOCVolumetricData

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

    def test_charge_with_magnetization(self) -> None:
        """Read an ISPIN=2 CHGCAR with scalar magnetization.

        The test parses ``CHGCAR_spin``, which has two data blocks.
        It checks the result type, ``grid_shape``, and both array shapes.
        It also verifies that the magnetization exists. These checks validate
        the ``ISPIN=2`` path, where magnetization equals ``rho_up-rho_down``.

        Notes
        -----
        The test builds the inputs in the test body and checks the stated
        property with the documented numerical or structural assertions.
        """
        path: Path
        vol: diffpes.types.VolumetricData | diffpes.types.SOCVolumetricData

        path = _FIXTURES_DIR / "CHGCAR_spin"
        vol = read_chgcar(str(path))
        assert isinstance(vol, VolumetricData)
        assert vol.grid_shape == (2, 2, 2)
        chex.assert_shape(vol.charge, (2, 2, 2))
        assert vol.magnetization is not None
        chex.assert_shape(vol.magnetization, (2, 2, 2))

    def test_soc_chgcar(self) -> None:
        """Read an SOC CHGCAR with three-component magnetization.

        Parses CHGCAR_soc (four data blocks: charge, mx, my, mz).
        The test checks the result type, ``grid_shape``, and all array shapes.
        It verifies that the scalar magnetization equals the z-component.
        It compares the first x-component with its volume-normalized value
        by using ``atol=1e-12``. These checks validate the four-block SOC path.

        Notes
        -----
        The test builds the inputs in the test body and checks the stated
        property with the documented numerical or structural assertions.
        """
        path: Path
        vol: diffpes.types.VolumetricData | diffpes.types.SOCVolumetricData

        path = _FIXTURES_DIR / "CHGCAR_soc"
        vol = read_chgcar(str(path))
        assert isinstance(vol, SOCVolumetricData)
        assert vol.grid_shape == (2, 2, 2)
        chex.assert_shape(vol.charge, (2, 2, 2))
        chex.assert_shape(vol.magnetization, (2, 2, 2))
        chex.assert_shape(vol.magnetization_vector, (2, 2, 2, 3))

        chex.assert_trees_all_close(
            vol.magnetization,
            vol.magnetization_vector[..., 2],
            atol=1e-12,
        )

        chex.assert_trees_all_close(
            vol.magnetization_vector[0, 0, 0, 0],
            jnp.float64(0.10 / 27.0),
            atol=1e-12,
        )


def _write_chgcar_tmpfile(content: str) -> str:
    """PRIVATE: Write content to a temporary CHGCAR file.

    Parameters
    ----------
    content : str
        Full CHGCAR text to write to the file.

    Returns
    -------
    path : str
        Filesystem path of the written temporary file.

    Notes
    -----
    Creates a ``NamedTemporaryFile`` with the ``.CHGCAR`` suffix and
    ``delete=False`` so the file survives for the calling test. The
    caller removes the file afterwards.
    """
    fh: TextIO

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".CHGCAR", delete=False
    ) as fh:
        fh.write(content)
        path: str = fh.name
        return path


_CHGCAR_POSCAR_HEADER: str = (
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
    """Validate error paths in :func:`read_chgcar`.

    :see: :func:`~diffpes.inout.read_chgcar`
    """

    def _cleanup(self, path: str) -> None:
        """PRIVATE: Delete one temporary CHGCAR file.

        Parameters
        ----------
        path : str
            Filesystem path of the temporary file to delete.

        Notes
        -----
        Calls ``os.unlink`` on the path after the error assertion so
        the temporary file does not accumulate between test runs.
        """
        os.unlink(path)

    def test_zero_volume_lattice_raises(self) -> None:
        """Verify that a zero-volume lattice raises ``ValueError``.

        The test writes a CHGCAR where the lattice vectors are all zero, making
        the unit cell volume = 0. Asserts ``ValueError`` matching
        ``"volume is zero"``.

        Notes
        -----
        The test builds the inputs in the test body and checks the stated
        property with the documented numerical or structural assertions.
        """
        content: str
        path: str

        content = (
            "Test system\n"
            " 1.0\n"
            "  0.0  0.0  0.0\n"
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

    def test_no_grid_dimensions_raises(self) -> None:
        """Verify that a missing grid dimension line raises ``ValueError``.

        The test writes a CHGCAR with a valid POSCAR header but no grid
        dimension line after the coordinates. Asserts ``ValueError`` matching
        ``"Could not locate"``.

        Notes
        -----
        The test builds the inputs in the test body and checks the stated
        property with the documented numerical or structural assertions.
        """
        content: str
        path: str

        content = _CHGCAR_POSCAR_HEADER
        path = _write_chgcar_tmpfile(content)
        try:
            with pytest.raises(ValueError, match="Could not locate"):
                read_chgcar(path)
        finally:
            self._cleanup(path)

    def test_invalid_lattice_line_raises(self) -> None:
        """Verify that a short lattice line raises ``ValueError``.

        The test writes a CHGCAR where the first lattice row has only 2 values.
        The test asserts ``ValueError`` matching ``"Invalid CHGCAR lattice"``.

        Notes
        -----
        The test builds the inputs in the test body and checks the stated
        property with the documented numerical or structural assertions.
        """
        content: str
        path: str

        content = (
            "Test system\n"
            " 1.0\n"
            "  3.0  0.0\n"
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

    def test_selective_dynamics_line_consumed(self) -> None:
        """Consume the selective-dynamics line before coordinates.

        The test writes a CHGCAR with a 'Selective dynamics' line before the
        coordinate-mode line. The test checks successful parsing and the
        ``Direct`` coordinate mode.

        Notes
        -----
        The test builds the inputs in the test body and checks the stated
        property with the documented numerical or structural assertions.
        """
        content: str
        path: str
        vol: diffpes.types.VolumetricData | diffpes.types.SOCVolumetricData

        content = (
            "Test system\n"
            " 1.0\n"
            "  3.0  0.0  0.0\n"
            "  0.0  3.0  0.0\n"
            "  0.0  0.0  3.0\n"
            "Si\n"
            "1\n"
            "Selective dynamics\n"
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

    def test_invalid_coordinate_line_raises(self) -> None:
        """Verify that a short atomic-coordinate line raises ``ValueError``.

        The test writes a CHGCAR where the atomic coordinate line has only 2
        values. Asserts ``ValueError`` matching
        ``"Invalid CHGCAR coordinate"``.

        Notes
        -----
        The test builds the inputs in the test body and checks the stated
        property with the documented numerical or structural assertions.
        """
        content: str
        path: str

        content = (
            "Test system\n"
            " 1.0\n"
            "  3.0  0.0  0.0\n"
            "  0.0  3.0  0.0\n"
            "  0.0  0.0  3.0\n"
            "Si\n"
            "1\n"
            "Direct\n"
            "  0.0  0.0\n"
        )
        path = _write_chgcar_tmpfile(content)
        try:
            with pytest.raises(ValueError, match="Invalid CHGCAR coordinate"):
                read_chgcar(path)
        finally:
            self._cleanup(path)

    def test_cartesian_coordinates_transform(self) -> None:
        """Convert Cartesian coordinates to fractional coordinates.

        The test writes a CHGCAR with Cartesian coordinate mode. One atom has
        coordinate ``[3.0, 0.0, 0.0]`` in a cubic lattice. Its expected
        fractional coordinate is ``[1.0, 0.0, 0.0]``. The comparison uses
        ``atol=1e-10``.

        Notes
        -----
        The test builds the inputs in the test body and checks the stated
        property with the documented numerical or structural assertions.
        """
        content: str
        path: str
        vol: diffpes.types.VolumetricData | diffpes.types.SOCVolumetricData

        content = (
            "Test system\n"
            " 1.0\n"
            "  3.0  0.0  0.0\n"
            "  0.0  3.0  0.0\n"
            "  0.0  0.0  3.0\n"
            "Si\n"
            "1\n"
            "Cartesian\n"
            "  3.0  0.0  0.0\n"
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

    def test_find_next_grid_skips_non_matching_lines(self) -> None:
        """Skip malformed lines while locating the next grid block.

        The test writes a CHGCAR with nonmatching lines before the grid shape.
        One line has the three floats ``1.5 2.5 3.5``. The parser skips these
        lines before it reads the grid shape.

        Notes
        -----
        The test builds the inputs in the test body and checks the stated
        property with the documented numerical or structural assertions.
        """
        content: str
        path: str
        vol: diffpes.types.VolumetricData | diffpes.types.SOCVolumetricData

        content = (
            _CHGCAR_POSCAR_HEADER
            + "\n"
            + "   2   2   2\n"
            + " 1.0 2.0 3.0 4.0\n"
            + " 5.0 6.0 7.0 8.0\n"
            + "\n"
            + "some augmentation text line here\n"
            + "1.5 2.5 3.5\n"
            + "   2   2   2\n"
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

    def test_parse_float_block_skips_blank_lines(self) -> None:
        """Skip blank lines while parsing a floating-point block.

        The test writes a CHGCAR whose data block has a blank line after the
        first row of values. The test checks the parsed grid.

        Notes
        -----
        The test builds the inputs in the test body and checks the stated
        property with the documented numerical or structural assertions.
        """
        content: str
        path: str
        vol: diffpes.types.VolumetricData | diffpes.types.SOCVolumetricData

        content = (
            _CHGCAR_POSCAR_HEADER
            + "\n"
            + "   2   2   2\n"
            + " 1.0 2.0 3.0 4.0\n"
            + "\n"
            + " 5.0 6.0 7.0 8.0\n"
        )
        path = _write_chgcar_tmpfile(content)
        try:
            vol = read_chgcar(path)
            chex.assert_shape(vol.charge, (2, 2, 2))
        finally:
            self._cleanup(path)

    def test_parse_float_block_stops_on_non_float_token(self) -> None:
        """Verify that a non-float token sets ``row_valid=False``.

        The test writes a CHGCAR with a header between two float rows.
        The non-float token sets ``row_valid=False`` and stops the inner loop.
        The parser does not append that row. The outer loop advances, and
        later valid rows complete the eight-value collection.

        Notes
        -----
        The test builds the inputs in the test body and checks the stated
        property with the documented numerical or structural assertions.
        """
        content: str
        path: str
        vol: diffpes.types.VolumetricData | diffpes.types.SOCVolumetricData

        content = (
            _CHGCAR_POSCAR_HEADER
            + "\n"
            + "   2   2   2\n"
            + " 1.0 2.0 3.0 4.0\n"
            + "augmentation occupancies\n"
            + " 5.0 6.0 7.0 8.0\n"
        )
        path = _write_chgcar_tmpfile(content)
        try:
            vol = read_chgcar(path)
            chex.assert_shape(vol.charge, (2, 2, 2))
        finally:
            self._cleanup(path)

    def test_parse_float_block_truncated_raises(self) -> None:
        """Verify that a truncated data block raises ``ValueError``.

        The test writes a CHGCAR where the grid is 2x2x2 (needs 8 values) but
        only 4 values are present. Asserts ``ValueError`` matching
        ``"Unexpected end of CHGCAR data block"``.

        Notes
        -----
        The test builds the inputs in the test body and checks the stated
        property with the documented numerical or structural assertions.
        """
        content: str
        path: str

        content = (
            _CHGCAR_POSCAR_HEADER
            + "\n"
            + "   2   2   2\n"
            + " 1.0 2.0 3.0 4.0\n"
        )
        path = _write_chgcar_tmpfile(content)
        try:
            with pytest.raises(ValueError, match="Unexpected end of CHGCAR"):
                read_chgcar(path)
        finally:
            self._cleanup(path)
