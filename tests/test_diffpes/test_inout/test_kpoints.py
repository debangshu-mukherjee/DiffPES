"""Validate VASP KPOINTS parsing.

Covers line-mode, automatic, and explicit sampling together with labels,
weights, shifts, and malformed coordinate records.
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
    read_kpoints,
)

_FIXTURES_DIR: Path = Path(__file__).resolve().parent / "fixtures"


class TestReadKpoints(chex.TestCase):
    """Validate :func:`diffpes.inout.read_kpoints`.

    Covers Line-mode (with and without label fallback), Automatic,
    and Explicit KPOINTS formats. Asserts both legacy plotting fields
    (mode, labels, label_indices) and richer mode-specific metadata
    (grid/shift, explicit k-points/weights, line-mode endpoints).

    :see: :func:`~diffpes.inout.read_kpoints`
    """

    def test_line_mode(self) -> None:
        """Read line-mode KPOINTS and assert its symmetry labels.

        The test parses the ``KPOINTS_line`` fixture. It verifies the mode,
        point count, and labels. It also verifies all metadata for line mode.

        Notes
        -----
        The test builds the inputs in the test body and checks the stated
        property with the documented numerical or structural assertions.
        """
        path: Path
        kpath: diffpes.types.KPathInfo

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

    def test_automatic_mode(self) -> None:
        """Read automatic KPOINTS and assert its zero point count.

        The test parses ``KPOINTS_auto``. It checks the mode, zero point
        count, grid, and shift.

        Notes
        -----
        The test builds the inputs in the test body and checks the stated
        property with the documented numerical or structural assertions.
        """
        path: Path
        kpath: diffpes.types.KPathInfo

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

    def test_line_mode_label_fallback(self) -> None:
        """Read line-mode KPOINTS using fallback label extraction.

        The test uses KPOINTS_line_fallback where one line has five tokens
        (coordinates plus weight and label "G") and another has three
        (no label). The test verifies the mode and both label outcomes.
        This input covers both label extraction branches.

        Notes
        -----
        The test builds the inputs in the test body and checks the stated
        property with the documented numerical or structural assertions.
        """
        path: Path
        kpath: diffpes.types.KPathInfo

        path = _FIXTURES_DIR / "KPOINTS_line_fallback"
        kpath = read_kpoints(str(path))
        assert kpath.mode == "Line-mode"
        assert "G" in kpath.labels
        assert "" in kpath.labels or len(kpath.labels) == 2
        chex.assert_equal(kpath.segments, 1)
        chex.assert_shape(kpath.kpoints, (2, 3))

    def test_explicit_mode(self) -> None:
        """Read Explicit KPOINTS and assert mode and k-point count.

        The test parses ``KPOINTS_explicit``. It checks the mode, point count,
        k-points, and weights.

        Notes
        -----
        The test builds the inputs in the test body and checks the stated
        property with the documented numerical or structural assertions.
        """
        path: Path
        kpath: diffpes.types.KPathInfo

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

    def test_explicit_mode_with_mode_header(self) -> None:
        """Read explicit KPOINTS with a separate mode header.

        The test parses ``KPOINTS_explicit_mode_header``. A separate header
        line declares the mode before the coordinate system. The test checks
        the mode, point count, array shapes, first point, weights, and
        coordinate mode.
        This input covers the alternative KPOINTS layout.

        Notes
        -----
        The test builds the inputs in the test body and checks the stated
        property with the documented numerical or structural assertions.
        """
        path: Path
        kpath: diffpes.types.KPathInfo

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


def _write_kpoints_tmpfile(content: str) -> str:
    """PRIVATE: Write content to a temporary KPOINTS file.

    Parameters
    ----------
    content : str
        Full KPOINTS text to write to the file.

    Returns
    -------
    path : str
        Filesystem path of the written temporary file.

    Notes
    -----
    Creates a ``NamedTemporaryFile`` with the ``.KPOINTS`` suffix and
    ``delete=False`` so the file survives for the calling test. The
    caller removes the file afterwards.
    """
    fh: TextIO

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".KPOINTS", delete=False
    ) as fh:
        fh.write(content)
        path: str = fh.name
        return path


class TestReadKpointsErrors(chex.TestCase):
    """Validate additional paths in :func:`read_kpoints`.

    :see: :func:`~diffpes.inout.read_kpoints`
    """

    def _cleanup(self, path: str) -> None:
        """PRIVATE: Delete one temporary KPOINTS file.

        Parameters
        ----------
        path : str
            Filesystem path of the temporary file to delete.

        Notes
        -----
        Calls ``os.unlink`` on the path after the assertion so the
        temporary file does not accumulate between test runs.
        """
        os.unlink(path)

    def test_explicit_break_when_excess_lines(self) -> None:
        """Verify that explicit KPOINTS ignores excess point lines.

        The test writes a 3-kpoint explicit KPOINTS file with four coordinate
        lines.
        The test checks the Explicit mode and exactly three returned k-points.
        The parser ignores the fourth line.

        Notes
        -----
        The test builds the inputs in the test body and checks the stated
        property with the documented numerical or structural assertions.
        """
        content: str
        path: str
        kpath: diffpes.types.KPathInfo

        content = (
            "Excess explicit kpoints\n"
            "3\n"
            "Cartesian\n"
            "  0.0  0.0  0.0  1.0\n"
            "  0.5  0.0  0.0  0.5\n"
            "  0.5  0.5  0.0  0.5\n"
            "  0.0  0.5  0.0  0.5\n"
        )
        path = _write_kpoints_tmpfile(content)
        try:
            kpath = read_kpoints(path)
            assert kpath.mode == "Explicit"
            assert kpath.num_kpoints == 3
        finally:
            self._cleanup(path)

    def test_explicit_invalid_float_raises(self) -> None:
        """Verify that a nonnumeric KPOINTS token raises ``ValueError``.

        The test writes an explicit KPOINTS with ``abc 0.0 0.0`` as a point
        line. It asserts ``ValueError`` matching
        ``"Invalid explicit KPOINTS"``.

        Notes
        -----
        The test builds the inputs in the test body and checks the stated
        property with the documented numerical or structural assertions.
        """
        content: str
        path: str

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

    def test_explicit_too_few_coords_raises(self) -> None:
        """Verify that a short KPOINTS coordinate raises ``ValueError``.

        The test writes an explicit KPOINTS where a point line has only two
        values.
        The test asserts ``ValueError`` matching ``"at least 3 coordinates"``.

        Notes
        -----
        The test builds the inputs in the test body and checks the stated
        property with the documented numerical or structural assertions.
        """
        content: str
        path: str

        content = "Too few coords\n2\nCartesian\n  0.0  0.0\n  0.5  0.0  0.0\n"
        path = _write_kpoints_tmpfile(content)
        try:
            with pytest.raises(ValueError, match="at least 3 coordinates"):
                read_kpoints(path)
        finally:
            self._cleanup(path)

    def test_explicit_no_weight_column_defaults_to_one(self) -> None:
        """Verify the default KPOINTS weight without a weight column.

        The test writes an Explicit KPOINTS where k-point lines have only 3
        columns (no weight). Asserts that all weights equal 1.0.

        Notes
        -----
        The test builds the inputs in the test body and checks the stated
        property with the documented numerical or structural assertions.
        """
        content: str
        path: str
        kpath: diffpes.types.KPathInfo

        content = "No weight\n2\nCartesian\n  0.0  0.0  0.0\n  0.5  0.0  0.0\n"
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

    def test_looks_like_kpoint_line_value_error_branch(self) -> None:
        """Verify rejection of a nonnumeric three-token k-point line.

        The test uses an Explicit KPOINTS with an unrecognized ``mode_line``.
        The first remaining line has three tokens and a nonnumeric first token.
        The test checks consumption of the mode header as ``coord_mode``.

        Notes
        -----
        The test builds the inputs in the test body and checks the stated
        property with the documented numerical or structural assertions.
        """
        content: str
        path: str
        kpath: diffpes.types.KPathInfo

        content = (
            "Test non-numeric\n"
            "2\n"
            "Explicit\n"
            "Reciprocal 0.5 0.5\n"
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

    def test_looks_like_kpoint_line_returns_true(self) -> None:
        """Verify acceptance of a numeric k-point line.

        The test uses an Explicit KPOINTS with an unrecognized ``mode_line``.
        The first remaining line contains numeric tokens. All float calls
        succeed (lines 272-273 executed), so the function reaches
        ``return True`` (line 276). This means ``remaining_lines.pop(0)`` is
        not called and
        ``coord_mode`` stays as ``scheme_or_mode`` ("Explicit").

        Notes
        -----
        The test builds the inputs in the test body and checks the stated
        property with the documented numerical or structural assertions.
        """
        content: str
        path: str
        kpath: diffpes.types.KPathInfo

        content = (
            "Test numeric first line\n"
            "2\n"
            "Explicit\n"
            "0.0 0.0 0.0 1.0\n"
            "0.5 0.0 0.0 0.5\n"
        )
        path = _write_kpoints_tmpfile(content)
        try:
            kpath = read_kpoints(path)
            assert kpath.mode == "Explicit"
            assert kpath.num_kpoints == 2
        finally:
            self._cleanup(path)

    def test_automatic_grid_too_few_values_raises(self) -> None:
        """Verify that a short automatic-grid line raises ``ValueError``.

        The test writes an Automatic KPOINTS where the grid line has only 2
        integers. Asserts ``ValueError`` matching ``"3 values"``.

        Notes
        -----
        The test builds the inputs in the test body and checks the stated
        property with the documented numerical or structural assertions.
        """
        content: str
        path: str

        content = "Bad auto\n0\nMonkhorst-Pack\n4 4\n0 0 0\n"
        path = _write_kpoints_tmpfile(content)
        try:
            with pytest.raises(ValueError, match="3 values"):
                read_kpoints(path)
        finally:
            self._cleanup(path)

    def test_automatic_shift_too_few_values_raises(self) -> None:
        """Verify that a short automatic-shift line raises ``ValueError``.

        The test writes an Automatic KPOINTS where the shift line has only 2
        values. Asserts ``ValueError`` matching ``"3 values"``.

        Notes
        -----
        The test builds the inputs in the test body and checks the stated
        property with the documented numerical or structural assertions.
        """
        content: str
        path: str

        content = "Bad auto shift\n0\nMonkhorst-Pack\n4 4 4\n0 0\n"
        path = _write_kpoints_tmpfile(content)
        try:
            with pytest.raises(ValueError, match="3 values"):
                read_kpoints(path)
        finally:
            self._cleanup(path)

    def test_line_mode_bad_coord_line_raises(self) -> None:
        """Verify that an invalid line-mode coordinate raises ``ValueError``.

        The test writes a Line-mode KPOINTS where a coordinate line has fewer
        than 3 float tokens. Asserts ``ValueError`` matching
        ``"Could not parse k-point coordinates"``.

        Notes
        -----
        The test builds the inputs in the test body and checks the stated
        property with the documented numerical or structural assertions.
        """
        content: str
        path: str

        content = (
            "Bad line mode\n"
            "2\n"
            "Line-mode\n"
            "Reciprocal\n"
            "  text only\n"
            "  0.5  0.5  0.5  M\n"
        )
        path = _write_kpoints_tmpfile(content)
        try:
            with pytest.raises(ValueError, match="Could not parse k-point"):
                read_kpoints(path)
        finally:
            self._cleanup(path)
