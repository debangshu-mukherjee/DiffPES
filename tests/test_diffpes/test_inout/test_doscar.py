"""Validate VASP DOSCAR parsing.

Covers legacy and full density-of-states carriers, spin channels, projected
blocks, and malformed or truncated input.
"""

import os
import tempfile
from pathlib import Path

import chex
import jax.numpy as jnp
from beartype.typing import TextIO

import diffpes
from diffpes.inout import (
    read_doscar,
)
from diffpes.types import (
    FullDensityOfStates,
)

_FIXTURES_DIR: Path = Path(__file__).resolve().parent / "fixtures"


class TestReadDoscar(chex.TestCase):
    """Validate :func:`diffpes.inout.read_doscar`.

    Verifies that the DOSCAR parser produces a valid DensityOfStates
    PyTree with correct array shapes and expected numeric values from
    the minimal fixture.

    :see: :func:`~diffpes.inout.read_doscar`
    """

    def test_parses_minimal_doscar(self) -> None:
        """Read a minimal DOSCAR and assert its key values.

        The test loads the ``fixtures/DOSCAR`` file. It checks the array
        shapes and the scalar Fermi energy. It compares selected elements
        with the known fixture values. These checks verify the DOSCAR header
        and data block.

        Notes
        -----
        The test builds the inputs in the test body and checks the stated
        property with the documented numerical or structural assertions.
        """
        path: Path
        dos: diffpes.types.DensityOfStates | diffpes.types.FullDensityOfStates

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


class TestReadDoscarFull(chex.TestCase):
    """Validate :func:`diffpes.inout.read_doscar` with ``return_mode='full'``.

    Validates the full-mode DOSCAR parser that returns
    ``FullDensityOfStates`` objects containing spin-resolved total DOS,
    integrated DOS, and optionally per-atom projected DOS (PDOS).
    Covers spin-polarized files (both channels), non-spin files
    (spin-down fields are None), PDOS-containing files, and the
    legacy fallback for spin-polarized data.

    :see: :func:`~diffpes.inout.read_doscar`
    """

    def test_spin_doscar_full(self) -> None:
        """Read both spin channels from a spin-polarized DOSCAR.

        Parses the DOSCAR_spin fixture with ``return_mode="full"``.
        The test checks the result type and the shape of each DOS array.
        It checks that both spin-down arrays exist. It compares the Fermi
        energy and the first value of each spin channel with fixture values.
        The comparisons use ``atol=1e-12``.

        Notes
        -----
        The test builds the inputs in the test body and checks the stated
        property with the documented numerical or structural assertions.
        """
        path: Path
        dos: diffpes.types.DensityOfStates | diffpes.types.FullDensityOfStates

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

    def test_nonspin_doscar_full(self) -> None:
        """Read a non-spin DOSCAR and verify absent spin-down fields.

        Parses the standard DOSCAR fixture (non-spin-polarized) with
        ``return_mode="full"``. The test checks the result type and the
        three spin-up array shapes. It verifies that both spin-down arrays
        are ``None``. These checks confirm detection of non-spin data.

        Notes
        -----
        The test builds the inputs in the test body and checks the stated
        property with the documented numerical or structural assertions.
        """
        path: Path
        dos: diffpes.types.DensityOfStates | diffpes.types.FullDensityOfStates

        path = _FIXTURES_DIR / "DOSCAR"
        dos = read_doscar(str(path), return_mode="full")
        assert isinstance(dos, FullDensityOfStates)
        chex.assert_shape(dos.energy, (5,))
        chex.assert_shape(dos.total_dos_up, (5,))
        assert dos.total_dos_down is None
        chex.assert_shape(dos.integrated_dos_up, (5,))
        assert dos.integrated_dos_down is None

    def test_pdos_doscar_full(self) -> None:
        """Read DOSCAR containing per-atom PDOS blocks in full mode.

        Parses the DOSCAR_pdos fixture with ``return_mode="full"``.
        The test checks the result type and verifies that ``pdos`` exists.
        The PDOS shape represents two atoms, three energies, and ten columns.
        The test also verifies ``natoms=2``. These checks validate the PDOS
        block after the total DOS header.

        Notes
        -----
        The test builds the inputs in the test body and checks the stated
        property with the documented numerical or structural assertions.
        """
        path: Path
        dos: diffpes.types.DensityOfStates | diffpes.types.FullDensityOfStates

        path = _FIXTURES_DIR / "DOSCAR_pdos"
        dos = read_doscar(str(path), return_mode="full")
        assert isinstance(dos, FullDensityOfStates)
        assert dos.pdos is not None

        chex.assert_shape(dos.pdos, (2, 3, 10))
        assert dos.natoms == 2

    def test_spin_doscar_legacy(self) -> None:
        """Read a legacy spin-polarized DOSCAR and verify the spin-up result.

        Parses the DOSCAR_spin fixture with ``return_mode="legacy"``.
        The test verifies a plain ``DensityOfStates`` result and both array
        shapes. It compares the first total DOS value with the spin-up value.
        The comparison uses ``atol=1e-12``. This test covers the compatible
        path that discards spin-down data.

        Notes
        -----
        The test builds the inputs in the test body and checks the stated
        property with the documented numerical or structural assertions.
        """
        path: Path
        dos: diffpes.types.DensityOfStates | diffpes.types.FullDensityOfStates

        path = _FIXTURES_DIR / "DOSCAR_spin"
        dos = read_doscar(str(path), return_mode="legacy")

        chex.assert_shape(dos.energy, (5,))
        chex.assert_shape(dos.total_dos, (5,))
        chex.assert_trees_all_close(
            dos.total_dos[0], jnp.float64(0.10), atol=1e-12
        )


class TestReadDoscarPdosHeader(chex.TestCase):
    """Validate :func:`read_doscar` PDOS parsing with per-atom header lines.

    :see: :func:`~diffpes.inout.read_doscar`
    """

    def test_pdos_with_header_line(self) -> None:
        """Exercise DOSCAR parsing with a header for each atom.

        The test uses the DOSCAR_pdos_header fixture where each atom's PDOS
        block starts with a 4-value header line (EMIN EMAX NEDOS EFERMI).
        The four values satisfy the column-count condition. Therefore, the
        parser executes the applicable header path. The test checks the PDOS
        shape for one atom, three energies, and nine orbital columns.

        Notes
        -----
        The test builds the inputs in the test body and checks the stated
        property with the documented numerical or structural assertions.
        """
        path: Path
        dos: diffpes.types.DensityOfStates | diffpes.types.FullDensityOfStates

        path = _FIXTURES_DIR / "DOSCAR_pdos_header"
        dos = read_doscar(str(path), return_mode="full")
        assert isinstance(dos, FullDensityOfStates)
        assert dos.pdos is not None

        chex.assert_shape(dos.pdos, (1, 3, 9))

    def test_pdos_early_break_on_empty_line(self) -> None:
        """Exercise PDOS parsing with fewer rows than ``NEDOS``.

        The test writes a DOSCAR where natoms=1, NEDOS=3 but the PDOS block has
        only 1 data row followed by an empty line. This causes the
        ``for j in range(1, nedos):`` loop to hit an empty ``row_line``
        and execute the ``break`` on line 230. Asserts the result is
        a ``FullDensityOfStates`` (parse continues with truncated data).

        Notes
        -----
        The test builds the inputs in the test body and checks the stated
        property with the documented numerical or structural assertions.
        """
        fh: TextIO

        content: str
        tmpname: str
        dos: diffpes.types.DensityOfStates | diffpes.types.FullDensityOfStates

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
            os.unlink(tmpname)

    def test_pdos_header_then_empty_line_breaks(self) -> None:
        """Exercise PDOS parsing with an empty line after the header.

        The test writes a DOSCAR with ``natoms=1`` and ``NEDOS=3``.
        Its PDOS block has a four-value header followed by an empty line.
        The parser reads the header
        (line 204), then reads the empty line as ``pdos_ncols_check`` and
        hits ``if not pdos_ncols_check.strip(): break`` on line 205-206.
        The test asserts a ``FullDensityOfStates`` result with ``pdos=None``
        because the break occurs before the parser collects any PDOS data.

        Notes
        -----
        The test builds the inputs in the test body and checks the stated
        property with the documented numerical or structural assertions.
        """
        fh: TextIO

        content: str
        tmpname: str
        dos: diffpes.types.DensityOfStates | diffpes.types.FullDensityOfStates

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
            " -2.0 1.0 3 0.5\n"
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
            assert dos.pdos is None
        finally:
            os.unlink(tmpname)

    def test_pdos_header_inner_loop_empty_row_breaks(self) -> None:
        """Exercise the PDOS header loop with an empty data row.

        The test writes a DOSCAR with ``natoms=1`` and ``NEDOS=3``.
        The PDOS block has a header, one valid row, and one empty row.
        The inner loop reads the
        empty ``row_line`` and hits ``if not row_line.strip(): break`` on
        line 217-218. Asserts the result is a ``FullDensityOfStates``.

        Notes
        -----
        The test builds the inputs in the test body and checks the stated
        property with the documented numerical or structural assertions.
        """
        fh: TextIO

        content: str
        tmpname: str
        dos: diffpes.types.DensityOfStates | diffpes.types.FullDensityOfStates

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
            " -2.0 1.0 3 0.5\n"
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
            os.unlink(tmpname)
