"""Validate Wannier operator sidecar structure and traced/static partitioning.

The tests cover both source formats, PyTree leaves, exact serialization
metadata, format-specific arrays, shape validation, and finite values.
"""

from pathlib import Path

import chex
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from beartype.typing import List, Optional
from jaxtyping import Array, Complex128, Int64
from numpy.typing import NDArray

from diffpes.types import (
    HamiltonianBlocks,
    HoppingRecord,
    PyTreeDef,
    TextLineCursor,
    WannierOperatorData,
    make_hamiltonian_blocks,
    make_hopping_record,
    make_text_line_cursor,
    make_wannier_operator_data,
)


class TestHamiltonianBlocks:
    """Validate :class:`~diffpes.types.HamiltonianBlocks` storage.

    The carrier must retain normalized matrices, source lines, cells, and
    degeneracies without changing parser values.

    :see: :class:`~diffpes.types.HamiltonianBlocks`
    """

    def test_stores_arrays_and_exact_metadata(self) -> None:
        """Preserve Hamiltonian arrays and exact block metadata.

        The check covers every field of one single-orbital parser block.

        Notes
        -----
        The test constructs the carrier directly with one complex matrix and
        its physical source line, then compares each retained field.
        """
        matrices: Complex128[NDArray, "1 1 1"] = np.asarray(
            [[[2.0 + 0.0j]]], dtype=np.complex128
        )
        source_lines: Int64[NDArray, "1 1 1"] = np.asarray(
            [[[7]]], dtype=np.int64
        )
        blocks: HamiltonianBlocks = HamiltonianBlocks(
            matrices=matrices,
            source_lines=source_lines,
            cells=((0, 0, 0),),
            degeneracies=(1,),
        )

        assert blocks.matrices is matrices
        assert blocks.source_lines is source_lines
        assert blocks.cells == ((0, 0, 0),)
        assert blocks.degeneracies == (1,)


class TestHoppingRecord:
    """Validate :class:`~diffpes.types.HoppingRecord` storage.

    The carrier must retain one orbital pair, cell, complex amplitude, and
    physical source line.

    :see: :class:`~diffpes.types.HoppingRecord`
    """

    def test_stores_one_directed_hopping(self) -> None:
        """Preserve all values of one directed hopping record.

        The check covers the exact indices, translation, amplitude, and line
        identity consumed by Hermitian-closure validation.

        Notes
        -----
        The test constructs the carrier directly and compares every field
        against a distinct deterministic value.
        """
        record: HoppingRecord = HoppingRecord(
            pair=(1, 2),
            cell=(-1, 0, 1),
            amplitude=0.25 - 0.5j,
            line_number=13,
        )

        assert record.pair == (1, 2)
        assert record.cell == (-1, 0, 1)
        assert record.amplitude == 0.25 - 0.5j
        assert record.line_number == 13


class TestTextLineCursor:
    """Validate :class:`~diffpes.types.TextLineCursor` parsing behavior.

    The carrier must advance by physical lines, skip blanks only on request,
    and reject unread records at exhaustion checks.

    :see: :class:`~diffpes.types.TextLineCursor`
    """

    def test_advances_across_physical_and_nonblank_lines(
        self, tmp_path: Path
    ) -> None:
        """Advance exactly while preserving one-based source line numbers.

        The check distinguishes a physical-line read from a nonblank-line
        read and verifies successful exhaustion after both operations.

        Notes
        -----
        The test writes a three-line UTF-8 file with one blank middle line,
        creates the cursor through its public factory, and consumes it.
        """
        path: Path = tmp_path / "records.dat"
        path.write_text("header\n\nrecord\n", encoding="utf-8")
        cursor: TextLineCursor = make_text_line_cursor(path)

        assert cursor.next_line("header") == (1, "header")
        assert cursor.next_nonempty("record") == (3, "record")
        cursor.ensure_exhausted()
        assert cursor.index == 3


class TestWannierOperatorData:
    """Validate :class:`diffpes.types.WannierOperatorData`.

    The cases compare numerical PyTree leaves for Hamiltonian-only and
    operator-bearing inputs.
    """

    def test_hr_tree_keeps_only_centres_as_a_numerical_leaf(self) -> None:
        """Preserve explicit centres and exact static serialization metadata.

        An hr sidecar must expose no absent position-matrix leaf.

        Notes
        -----
        Flatten and rebuild the carrier before comparing all metadata.
        """
        data: WannierOperatorData = make_wannier_operator_data(
            position_matrices=None,
            centres_cart=jnp.asarray([[0.2, 0.3, 0.4]]),
            cells=((-1, 0, 0), (0, 0, 0), (1, 0, 0)),
            degeneracies=(2, 1, 2),
            spin_layout="block_down_up",
            source_format="hr",
        )
        leaves: List[object]
        tree: PyTreeDef
        leaves, tree = jax.tree.flatten(data)
        restored: WannierOperatorData = jax.tree.unflatten(tree, leaves)

        assert len(leaves) == 1
        chex.assert_trees_all_close(restored, data)
        assert restored.cells == data.cells
        assert restored.degeneracies == (2, 1, 2)
        assert restored.source_format == "hr"

    def test_tb_tree_traces_position_matrices_and_centres(self) -> None:
        """Expose both numerical operator arrays as complex/real JAX leaves.

        A tb sidecar must retain declared double-precision dtypes.

        Notes
        -----
        Flatten one minimal carrier and inspect its two numerical arrays.
        """
        data: WannierOperatorData = make_wannier_operator_data(
            position_matrices=jnp.zeros(
                (1, 2, 2, 3),
                dtype=jnp.complex128,
            ),
            centres_cart=jnp.zeros((2, 3), dtype=jnp.float64),
            cells=((0, 0, 0),),
            degeneracies=(1,),
            spin_layout="interleaved_up_down",
            source_format="tb",
        )
        leaves: List[object] = jax.tree.leaves(data)

        assert len(leaves) == 2
        assert data.position_matrices is not None
        assert data.position_matrices.dtype == jnp.complex128
        assert data.centres_cart.dtype == jnp.float64


class TestMakeHamiltonianBlocks:
    """Validate :func:`~diffpes.types.make_hamiltonian_blocks`.

    The factory must keep validated parser arrays and exact metadata without
    an implicit copy or normalization.

    :see: :func:`~diffpes.types.make_hamiltonian_blocks`
    """

    def test_retains_supplied_arrays_by_identity(self) -> None:
        """Retain supplied Hamiltonian and source-line arrays by identity.

        The check prevents the carrier boundary from changing parser values
        through an implicit cast or copy.

        Notes
        -----
        The test creates width-qualified single-orbital NumPy arrays, passes
        them through the factory, and compares their object identities.
        """
        matrices: Complex128[NDArray, "1 1 1"] = np.zeros(
            (1, 1, 1), dtype=np.complex128
        )
        source_lines: Int64[NDArray, "1 1 1"] = np.ones(
            (1, 1, 1), dtype=np.int64
        )
        blocks: HamiltonianBlocks = make_hamiltonian_blocks(
            matrices=matrices,
            source_lines=source_lines,
            cells=((0, 0, 0),),
            degeneracies=(1,),
        )

        assert blocks.matrices is matrices
        assert blocks.source_lines is source_lines


class TestMakeHoppingRecord:
    """Validate :func:`~diffpes.types.make_hopping_record`.

    The factory must retain each validated parser value without changing its
    numeric or exact-integer representation.

    :see: :func:`~diffpes.types.make_hopping_record`
    """

    def test_retains_supplied_hopping_values(self) -> None:
        """Retain one orbital pair, cell, amplitude, and source line.

        The check covers the full value boundary of a non-onsite hopping
        record.

        Notes
        -----
        The test passes four distinct deterministic values through the public
        factory and compares the resulting fields exactly.
        """
        record: HoppingRecord = make_hopping_record(
            pair=(0, 1),
            cell=(1, -1, 0),
            amplitude=-0.75 + 0.125j,
            line_number=9,
        )

        assert record.pair == (0, 1)
        assert record.cell == (1, -1, 0)
        assert record.amplitude == -0.75 + 0.125j
        assert record.line_number == 9


class TestMakeTextLineCursor:
    """Validate :func:`~diffpes.types.make_text_line_cursor`.

    The factory must decode one UTF-8 file into exact physical lines and
    initialize its unread index at zero.

    :see: :func:`~diffpes.types.make_text_line_cursor`
    """

    def test_reads_utf8_lines_at_the_initial_index(
        self, tmp_path: Path
    ) -> None:
        """Read exact UTF-8 lines and start before the first line.

        The check covers Unicode decoding, newline removal, tuple storage, and
        initial cursor position.

        Notes
        -----
        The test writes two deterministic physical lines, invokes the public
        factory, and compares the path, line tuple, and index.
        """
        path: Path = tmp_path / "unicode.dat"
        path.write_text("alpha\nβeta\n", encoding="utf-8")
        cursor: TextLineCursor = make_text_line_cursor(path)

        assert cursor.path == path
        assert cursor.lines == ("alpha", "βeta")
        assert cursor.index == 0


class TestMakeWannierOperatorData:
    """Validate :func:`diffpes.types.make_wannier_operator_data`.

    The cases enforce source-specific matrices, consistent axes, exact cells,
    and finite numerical values.
    """

    @pytest.mark.parametrize(
        ("source_format", "position_matrices", "match"),
        [
            (
                "hr",
                jnp.zeros((1, 1, 1, 3), dtype=jnp.complex128),
                "hr operator data must not contain",
            ),
            (
                "tb",
                None,
                "tb operator data requires",
            ),
        ],
    )
    def test_enforces_source_specific_position_matrix_presence(
        self,
        source_format: str,
        position_matrices: Optional[Complex128[Array, "n_cell n_orb n_orb 3"]],
        match: str,
    ) -> None:
        """Reject position matrices on hr data and their absence on tb data.

        Each source grammar requires one unambiguous operator-data layout.

        Notes
        -----
        Parameterize both mismatched source and array-presence combinations.
        """
        with pytest.raises(ValueError, match=match):
            make_wannier_operator_data(
                position_matrices=position_matrices,
                centres_cart=jnp.zeros((1, 3)),
                cells=((0, 0, 0),),
                degeneracies=(1,),
                spin_layout="block_down_up",
                source_format=source_format,
            )

    def test_rejects_inconsistent_cells_weights_and_axes(self) -> None:
        """Reject duplicate cells, nonpositive weights, and wrong matrix axes.

        Static serialization metadata must agree with every numerical
        dimension.

        Notes
        -----
        Construct three independent malformed carriers and match diagnostics.
        """
        with pytest.raises(ValueError, match="cells must be unique"):
            make_wannier_operator_data(
                position_matrices=None,
                centres_cart=jnp.zeros((1, 3)),
                cells=((0, 0, 0), (0, 0, 0)),
                degeneracies=(1, 1),
                spin_layout="block_down_up",
                source_format="hr",
            )
        with pytest.raises(ValueError, match="positive integers"):
            make_wannier_operator_data(
                position_matrices=None,
                centres_cart=jnp.zeros((1, 3)),
                cells=((0, 0, 0),),
                degeneracies=(0,),
                spin_layout="block_down_up",
                source_format="hr",
            )
        with pytest.raises(ValueError, match="position_matrices must have"):
            make_wannier_operator_data(
                position_matrices=jnp.zeros(
                    (2, 2, 2, 3),
                    dtype=jnp.complex128,
                ),
                centres_cart=jnp.zeros((2, 3)),
                cells=((0, 0, 0),),
                degeneracies=(1,),
                spin_layout="block_down_up",
                source_format="tb",
            )

    def test_rejects_nonfinite_numerical_values(self) -> None:
        """Reject NaN centres through the carrier runtime validation.

        Shape-valid numerical arrays still require finite entries.

        Notes
        -----
        Inject one NaN into a minimal hr centre array.
        """
        with pytest.raises(RuntimeError, match="centres finite"):
            make_wannier_operator_data(
                position_matrices=None,
                centres_cart=jnp.asarray([[jnp.nan, 0.0, 0.0]]),
                cells=((0, 0, 0),),
                degeneracies=(1,),
                spin_layout="block_down_up",
                source_format="hr",
            )
