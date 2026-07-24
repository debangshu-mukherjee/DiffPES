"""Validate Wannier operator sidecar structure and traced/static partitioning.

The tests cover both source formats, PyTree leaves, exact serialization
metadata, format-specific arrays, shape validation, and finite values.
"""

import chex
import jax
import jax.numpy as jnp
import pytest

from diffpes.types.wannier import (
    WannierOperatorData,
    make_wannier_operator_data,
)


class TestWannierOperatorData:
    """Validate :class:`diffpes.types.WannierOperatorData`."""

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
        leaves: list[object]
        tree: jax.tree_util.PyTreeDef
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
        leaves: list[object] = jax.tree.leaves(data)

        assert len(leaves) == 2
        assert data.position_matrices is not None
        assert data.position_matrices.dtype == jnp.complex128
        assert data.centres_cart.dtype == jnp.float64


class TestMakeWannierOperatorData:
    """Validate :func:`diffpes.types.make_wannier_operator_data`."""

    @pytest.mark.parametrize(
        ("source_format", "position_matrices", "match"),
        (
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
        ),
    )
    def test_enforces_source_specific_position_matrix_presence(
        self,
        source_format: str,
        position_matrices: jax.Array | None,
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
