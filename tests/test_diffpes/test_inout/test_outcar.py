"""Validate VASP OUTCAR summary parsing.

Covers last-occurrence selection for repeated markers and the explicit
errors for files without both markers.
"""

import os
import tempfile

import chex
import jax.numpy as jnp
import pytest
from beartype.typing import TextIO

from diffpes.inout import (
    read_outcar,
)
from diffpes.types import (
    OutcarData,
)


def _write_outcar(text: str) -> str:
    """PRIVATE: Write one temporary OUTCAR-style file and return its path.

    Parameters
    ----------
    text : str
        Complete file content to write.

    Returns
    -------
    str
        Path of the written temporary file.
    """
    fh: TextIO

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".OUTCAR", delete=False
    ) as fh:
        fh.write(text)
        tmpname: str = fh.name
    return tmpname


class TestReadOutcar(chex.TestCase):
    """Validate :func:`diffpes.inout.read_outcar`.

    Verifies last-occurrence marker selection and the explicit missing
    marker errors.

    :see: :func:`~diffpes.inout.read_outcar`
    """

    def test_keeps_last_marker_values(self) -> None:
        """Keep the last E-fermi and NELECT values in the file.

        The test writes both markers twice with different values. The
        parser returns the later pair, matching the final electronic
        step.

        Notes
        -----
        The test builds the inputs in the test body and checks the stated
        property with the documented numerical or structural assertions.
        """
        summary: OutcarData

        tmpname: str = _write_outcar(
            "   NELECT =      84.0000000    total number of electrons\n"
            " E-fermi :   1.0000     XC(G=0):  -9.0\n"
            " some unrelated line\n"
            " E-fermi :   2.3919     XC(G=0):  -9.1234\n"
        )
        try:
            summary = read_outcar(tmpname)
        finally:
            os.unlink(tmpname)
        chex.assert_trees_all_close(
            summary.fermi_energy, jnp.float64(2.3919), atol=1e-12
        )
        chex.assert_trees_all_close(
            summary.nelect, jnp.float64(84.0), atol=1e-12
        )

    def test_missing_fermi_marker_raises(self) -> None:
        """Reject a file without an E-fermi line.

        The parser raises the documented ``ValueError`` when the scan
        finds no Fermi marker.

        Notes
        -----
        The test builds the inputs in the test body and checks the stated
        property with the documented numerical or structural assertions.
        """
        tmpname: str = _write_outcar(
            "   NELECT =      84.0000000    total number of electrons\n"
        )
        try:
            with pytest.raises(
                ValueError,
                match="No E-fermi line found in OUTCAR file.",
            ):
                read_outcar(tmpname)
        finally:
            os.unlink(tmpname)

    def test_missing_nelect_marker_raises(self) -> None:
        """Reject a file without a NELECT line.

        The parser raises the documented ``ValueError`` when the scan
        finds no electron-count marker.

        Notes
        -----
        The test builds the inputs in the test body and checks the stated
        property with the documented numerical or structural assertions.
        """
        tmpname: str = _write_outcar(
            " E-fermi :   2.3919     XC(G=0):  -9.1234\n"
        )
        try:
            with pytest.raises(
                ValueError,
                match="No NELECT line found in OUTCAR file.",
            ):
                read_outcar(tmpname)
        finally:
            os.unlink(tmpname)
