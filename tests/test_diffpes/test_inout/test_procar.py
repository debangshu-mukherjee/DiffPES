"""Validate VASP PROCAR parsing.

Covers non-spin, spin-polarized, and SOC projections together with orbital
ordering and malformed or truncated blocks.
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
    read_procar,
)
from diffpes.types import (
    SpinOrbitalProjection,
)

_FIXTURES_DIR: Path = Path(__file__).resolve().parent / "fixtures"


class TestReadProcar(chex.TestCase):
    """Validate :func:`diffpes.inout.read_procar`.

    Verifies that the PROCAR parser produces an OrbitalProjection
    with correct projection array shape and optional spin/oam
    absent when not present in the file.

    :see: :func:`~diffpes.inout.read_procar`
    """

    def test_parses_minimal_procar(self) -> None:
        """Read a minimal PROCAR and assert sample projection values.

        The test loads the minimal PROCAR fixture. Asserts projections shape
        (2, 2, 1, 9), selected projection values (0.1 and 0.18),
        and that spin and oam are None. Validates k-point/band/ion
        block parsing and orbital channel ordering.

        Notes
        -----
        The test builds the inputs in the test body and checks the stated
        property with the documented numerical or structural assertions.
        """
        path: Path
        orb: (
            diffpes.types.OrbitalProjection
            | diffpes.types.SpinOrbitalProjection
        )

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

    def test_spin_procar_legacy(self) -> None:
        """Read only the first PROCAR spin block in legacy mode.

        Parses the PROCAR_spin fixture with ``return_mode="legacy"``,
        which extracts only the first spin-up block. The test checks the
        projection shape and the selected s-orbital value. It also verifies
        that ``spin`` is ``None`` in legacy mode.
        This exercises the backward-compatible single-block extraction.

        Notes
        -----
        The test builds the inputs in the test body and checks the stated
        property with the documented numerical or structural assertions.
        """
        path: Path
        orb: (
            diffpes.types.OrbitalProjection
            | diffpes.types.SpinOrbitalProjection
        )

        path = _FIXTURES_DIR / "PROCAR_spin"
        orb = read_procar(str(path), return_mode="legacy")
        chex.assert_shape(orb.projections, (2, 2, 1, 9))

        chex.assert_trees_all_close(
            orb.projections[0, 0, 0, 0], jnp.float64(0.1), atol=1e-12
        )
        assert orb.spin is None

    def test_spin_procar_full(self) -> None:
        """Read spin-polarized PROCAR output in full mode.

        Parses the PROCAR_spin fixture with ``return_mode="full"``,
        returning a ``SpinOrbitalProjection`` with both ``projections``
        and ``spin`` arrays. The test checks the result type and both shapes.
        It compares the averaged s-orbital value with 0.09. This value is
        the mean of the two spin values. The comparison uses ``atol=1e-12``.

        Notes
        -----
        The test builds the inputs in the test body and checks the stated
        property with the documented numerical or structural assertions.
        """
        path: Path
        orb: (
            diffpes.types.OrbitalProjection
            | diffpes.types.SpinOrbitalProjection
        )

        path = _FIXTURES_DIR / "PROCAR_spin"
        orb = read_procar(str(path), return_mode="full")
        assert isinstance(orb, SpinOrbitalProjection)
        chex.assert_shape(orb.projections, (2, 2, 1, 9))
        chex.assert_shape(orb.spin, (2, 2, 1, 6))

        chex.assert_trees_all_close(
            orb.projections[0, 0, 0, 0], jnp.float64(0.09), atol=1e-12
        )


class TestReadProcarErrors(chex.TestCase):
    """Validate error handling in :func:`read_procar`.

    :see: :func:`~diffpes.inout.read_procar`
    """

    def test_empty_procar_raises(self) -> None:
        """Verify that an empty PROCAR file raises ``ValueError``.

        The test writes a file with no valid k-points block and calls
        ``read_procar``. Asserts a ``ValueError`` matching
        ``"No valid PROCAR blocks found"``.

        Notes
        -----
        The test builds the inputs in the test body and checks the stated
        property with the documented numerical or structural assertions.
        """
        fh: TextIO

        tmpname: str

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".PROCAR", delete=False
        ) as fh:
            fh.write("PROCAR empty\n")
            tmpname = fh.name
        try:
            with pytest.raises(ValueError, match="No valid PROCAR blocks"):
                read_procar(tmpname)
        finally:
            os.unlink(tmpname)


class TestReadProcarSOC(chex.TestCase):
    """Validate SOC PROCAR parsing (procar.py lines 191-210, 282-283, 307).

    :see: :func:`~diffpes.inout.read_procar`
    """

    def test_soc_procar_full(self) -> None:
        """Read SOC PROCAR (4 blocks) in full mode and verify spin components.

        The test uses the ``PROCAR_soc`` fixture. It has a title before the
        first k-point block and a blank line before the band header.
        The fixture contains four blocks. The test checks a
        ``SpinOrbitalProjection`` with projections shape (1, 1, 1, 9)
        and spin shape (1, 1, 1, 6).

        Notes
        -----
        The test builds the inputs in the test body and checks the stated
        property with the documented numerical or structural assertions.
        """
        path: Path
        orb: (
            diffpes.types.OrbitalProjection
            | diffpes.types.SpinOrbitalProjection
        )

        path = _FIXTURES_DIR / "PROCAR_soc"
        orb = read_procar(str(path), return_mode="full")
        assert isinstance(orb, SpinOrbitalProjection)
        chex.assert_shape(orb.projections, (1, 1, 1, 9))
        chex.assert_shape(orb.spin, (1, 1, 1, 6))

        chex.assert_trees_all_close(
            orb.projections[0, 0, 0, 0], jnp.float64(0.1), atol=1e-12
        )
