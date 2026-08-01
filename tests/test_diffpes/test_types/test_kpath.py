"""Validate k-path metadata storage and mode consistency.

The cases cover immutable plotting metadata and rejection of unsupported
KPOINTS mode selectors in ``diffpes.types.kpath``.
"""

import chex
import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float64

from diffpes.types import (
    KGrid,
    KPath,
    KPathInfo,
    make_kgrid,
    make_kpath,
    make_kpath_info,
)
from tests._assertions import assert_rejects
from tests._gradients import gradient_gate


class TestKPathInfo:
    """Validate :class:`~diffpes.types.KPathInfo` field storage.

    The carrier must preserve line-mode labels and their integer indices.

    :see: :class:`~diffpes.types.KPathInfo`
    """

    def test_stores_line_mode_metadata(self) -> None:
        """Preserve line-mode indices, labels, and mode text.

        The check verifies the three label positions for a 100-point path and
        the corresponding static symmetry labels.

        Notes
        -----
        The test constructs the carrier through its validated factory, checks the array
        shape with Chex, and compares the static metadata exactly.
        """
        kpath: KPathInfo = make_kpath_info(
            num_kpoints=100,
            label_indices=[0, 49, 99],
            segments=2,
            mode="Line-mode",
            labels=("G", "M", "K"),
        )

        chex.assert_shape(kpath.label_indices, (3,))
        chex.assert_equal(kpath.mode, "Line-mode")
        chex.assert_equal(kpath.labels, ("G", "M", "K"))


class TestMakeKPathInfo:
    """Validate :func:`~diffpes.types.make_kpath_info`.

    The factory must reject mode strings outside the supported parser
    conventions.

    :see: :func:`~diffpes.types.make_kpath_info`
    """

    def test_rejects_unknown_mode(self) -> None:
        """Reject a k-path mode outside the supported static set.

        The check isolates the mode selector contract before interpretation
        of optional arrays for a mode.

        Notes
        -----
        Supplies otherwise valid minimal metadata with ``mode="unknown"`` and
        matches the allowed-mode diagnostic.
        """
        assert_rejects(
            make_kpath_info,
            num_kpoints=2,
            label_indices=[0],
            segments=1,
            mode="unknown",
            match="mode must be one of",
        )

    def test_rejects_incomplete_line_metadata(self) -> None:
        """Reject missing labels, unequal label counts, and zero segments.

        The parser carrier requires complete static metadata for line mode.

        Notes
        -----
        The shared helper checks each independent structure error in eager and
        compiled execution.
        """
        assert_rejects(
            make_kpath_info,
            num_kpoints=2,
            label_indices=[],
            segments=1,
            match="at least one label index",
        )
        assert_rejects(
            make_kpath_info,
            num_kpoints=2,
            label_indices=[0, 1],
            segments=1,
            labels=("G",),
            match="labels and label_indices",
        )
        assert_rejects(
            make_kpath_info,
            num_kpoints=2,
            label_indices=[0],
            segments=0,
            match="at least one segment",
        )

    def test_validates_each_optional_numeric_field(self) -> None:
        """Store finite optional arrays and reject each non-finite array.

        The factory must apply its traced finite-value checks to k-points,
        weights, and the grid shift.

        Notes
        -----
        The test first constructs all optional fields. It then changes one
        floating field to NaN in each shared rejection check.
        """
        kpath: KPathInfo = make_kpath_info(
            num_kpoints=2,
            label_indices=[0],
            segments=1,
            kpoints=jnp.zeros((2, 3)),
            weights=jnp.ones((2,)),
            grid=[2, 2, 1],
            shift=jnp.zeros((3,)),
        )
        chex.assert_shape(kpath.kpoints, (2, 3))
        chex.assert_shape(kpath.weights, (2,))
        chex.assert_shape(kpath.grid, (3,))
        chex.assert_shape(kpath.shift, (3,))

        assert_rejects(
            make_kpath_info,
            num_kpoints=2,
            label_indices=[0],
            segments=1,
            kpoints=jnp.zeros((2, 3)).at[0, 0].set(jnp.nan),
            match="kpoints must be finite",
        )
        assert_rejects(
            make_kpath_info,
            num_kpoints=2,
            label_indices=[0],
            segments=1,
            weights=jnp.array([1.0, jnp.nan]),
            match="weights must be finite",
        )
        assert_rejects(
            make_kpath_info,
            num_kpoints=2,
            label_indices=[0],
            segments=1,
            shift=jnp.array([0.0, jnp.nan, 0.0]),
            match="shift must be finite",
        )


class TestKPath:
    """Validate :class:`~diffpes.types.KPath` as a JAX PyTree.

    The cases cover reconstruction and the division between traced path data
    and static plotting metadata.

    :see: :class:`~diffpes.types.KPath`
    """

    def test_preserves_static_and_traced_fields_in_a_tree_round_trip(
        self,
    ) -> None:
        """Preserve path coordinates and plotting metadata after reconstruction.

        The tree must expose only the k-points and fixed ``kz`` as numerical
        leaves.

        Notes
        -----
        The test flattens one path and restores it with the same JAX tree
        definition. Chex compares the complete carriers.
        """
        kpath: KPath = make_kpath(
            jnp.array([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]]),
            labels=("G", "X"),
            label_indices=(0, 1),
            n_per_segment=2,
            kz=0.4,
        )
        leaves: list[Array]
        tree: jax.tree_util.PyTreeDef
        leaves, tree = jax.tree.flatten(kpath)
        restored: KPath = jax.tree.unflatten(tree, leaves)

        chex.assert_equal(len(leaves), 2)
        chex.assert_trees_all_close(restored, kpath)
        chex.assert_equal(restored.labels, ("G", "X"))


class TestKGrid:
    """Validate :class:`~diffpes.types.KGrid` as a JAX PyTree.

    The cases cover the static raster shape and optional traced axis values.

    :see: :class:`~diffpes.types.KGrid`
    """

    def test_keeps_only_the_mesh_shape_static(self) -> None:
        """Keep the raster coordinates and optional values as traced leaves.

        The dynamic partition must contain the k-points, ``kz``, and the
        photon-energy axis.

        Notes
        -----
        The test partitions one two-by-two grid with
        :func:`equinox.is_array` and counts its JAX leaves.
        """
        kgrid: KGrid = make_kgrid(
            jnp.zeros((4, 3)),
            (2, 2),
            kz=0.5,
            photon_energy_axis_ev=jnp.array([20.0, 30.0]),
        )
        dynamic: KGrid
        static: KGrid
        dynamic, static = eqx.partition(kgrid, eqx.is_array)

        chex.assert_equal(len(jax.tree.leaves(dynamic)), 3)
        chex.assert_equal(dynamic.mesh_shape, (2, 2))
        chex.assert_equal(static.mesh_shape, (2, 2))


class TestMakeKPath:
    """Validate :func:`~diffpes.types.make_kpath`.

    The cases cover static metadata checks, compiled finite-value validation,
    and coordinate sensitivity.

    :see: :func:`~diffpes.types.make_kpath`
    """

    def test_rejects_nonfinite_path_data_under_jit(self) -> None:
        """Reject non-finite path data in eager and compiled modes.

        Runtime validation must identify either the k-point or fixed ``kz``
        field in both modes.

        Notes
        -----
        The shared helper sends one invalid k-point path and one invalid
        ``kz`` value through both execution modes.
        """
        assert_rejects(
            make_kpath,
            jnp.array([[0.0, 0.0, 0.0], [jnp.nan, 0.0, 0.0]]),
            labels=("G", "X"),
            label_indices=(0, 1),
            n_per_segment=2,
            match="kpoints must be finite",
        )
        assert_rejects(
            make_kpath,
            jnp.zeros((2, 3)),
            labels=("G", "X"),
            label_indices=(0, 1),
            n_per_segment=2,
            kz=jnp.nan,
            match="kz must be finite",
        )

    def test_rejects_invalid_static_metadata(self) -> None:
        """Reject unordered indices or indices outside the path.

        Static validation must fail before traced coordinate validation.

        Notes
        -----
        The test supplies two descending indices and matches the ordering
        diagnostic in eager and compiled modes.
        """
        assert_rejects(
            make_kpath,
            jnp.zeros((3, 3)),
            labels=("A", "B"),
            label_indices=(2, 1),
            n_per_segment=2,
            match="strictly increasing",
        )

    def test_rejects_each_other_invalid_static_setting(self) -> None:
        """Reject a zero segment size and inconsistent label metadata.

        The factory must reject each invalid structure before it validates the
        traced coordinates.

        Notes
        -----
        The test sends three independent invalid cases through eager and
        compiled execution with the shared rejection helper.
        """
        points: Float64[Array, "3 3"] = jnp.zeros((3, 3))
        assert_rejects(
            make_kpath,
            points,
            labels=("A",),
            label_indices=(0,),
            n_per_segment=0,
            match="n_per_segment",
        )
        assert_rejects(
            make_kpath,
            points,
            labels=("A",),
            label_indices=(0, 2),
            match="equal lengths",
        )
        assert_rejects(
            make_kpath,
            points,
            labels=("A",),
            label_indices=(3,),
            match="k-point range",
        )

    def test_preserves_gradients_through_all_kpoint_components(self) -> None:
        """Preserve finite nonzero gradients through all path coordinates.

        Automatic derivatives must agree with central differences for a
        weighted coordinate reduction.

        Notes
        -----
        The shared gradient gate checks reverse mode and every component of a
        generic two-point path.
        """
        points: Float64[Array, "2 3"] = jnp.array(
            [[0.1, 0.2, 0.3], [0.4, -0.2, 0.5]]
        )
        weights: Float64[Array, "2 3"] = jnp.arange(1.0, 7.0).reshape((2, 3))

        def loss(candidate: Float64[Array, "2 3"]) -> Float64[Array, ""]:
            """Reduce all traced path coordinates with distinct weights."""
            kpath: KPath = make_kpath(
                candidate,
                labels=("A", "B"),
                label_indices=(0, 1),
                n_per_segment=2,
            )
            result: Float64[Array, ""] = jnp.sum(kpath.kpoints * weights)
            return result

        gradient_gate(loss, points, modes=("rev",))


class TestMakeKGrid:
    """Validate :func:`~diffpes.types.make_kgrid`.

    The cases cover raster consistency, compiled axis validation, and
    coordinate sensitivity.

    :see: :func:`~diffpes.types.make_kgrid`
    """

    def test_rejects_an_inconsistent_static_mesh_shape(self) -> None:
        """Reject a mesh shape whose product differs from the point count.

        The static contract must hold before any consumer reshapes the raster.

        Notes
        -----
        The test gives six points and a four-element mesh shape. The shared
        helper checks eager and compiled rejection.
        """
        assert_rejects(
            make_kgrid,
            jnp.zeros((6, 3)),
            (2, 2),
            match="mesh_shape product",
        )

    def test_rejects_each_other_invalid_static_shape(self) -> None:
        """Reject a nonpositive dimension and an axis with the wrong length.

        The factory must validate both parts of the fixed raster contract.

        Notes
        -----
        The shared helper checks eager and compiled rejection for two
        independent invalid grid structures.
        """
        assert_rejects(
            make_kgrid,
            jnp.zeros((2, 3)),
            (0, 2),
            match="dimensions must be positive",
        )
        assert_rejects(
            make_kgrid,
            jnp.zeros((4, 3)),
            (2, 2),
            photon_energy_axis_ev=jnp.array([20.0]),
            match="length must equal n_rows",
        )

    def test_rejects_invalid_traced_grid_data_under_jit(self) -> None:
        """Reject invalid traced grid data in eager and compiled modes.

        Checks for the energy axis, k-points, and fixed ``kz`` must remain
        active after JAX compilation.

        Notes
        -----
        The shared helper checks one invalid value in each traced grid field.
        """
        assert_rejects(
            make_kgrid,
            jnp.zeros((4, 3)),
            (2, 2),
            photon_energy_axis_ev=jnp.array([20.0, 0.0]),
            match="photon_energy_axis_ev",
        )
        assert_rejects(
            make_kgrid,
            jnp.zeros((4, 3)).at[0, 0].set(jnp.nan),
            (2, 2),
            match="kpoints must be finite",
        )
        assert_rejects(
            make_kgrid,
            jnp.zeros((4, 3)),
            (2, 2),
            kz=jnp.nan,
            match="kz must be finite",
        )

    def test_preserves_gradients_through_all_grid_components(self) -> None:
        """Preserve finite nonzero gradients through all grid coordinates.

        Automatic derivatives must agree with central differences for a
        weighted coordinate reduction.

        Notes
        -----
        The shared gradient gate checks reverse mode and every component of a
        generic two-by-two raster.
        """
        points: Float64[Array, "4 3"] = jnp.arange(1.0, 13.0).reshape((4, 3))
        weights: Float64[Array, "4 3"] = jnp.linspace(0.2, 1.3, 12).reshape(
            (4, 3)
        )

        def loss(candidate: Float64[Array, "4 3"]) -> Float64[Array, ""]:
            """Reduce all traced grid coordinates with distinct weights."""
            kgrid: KGrid = make_kgrid(candidate, (2, 2))
            result: Float64[Array, ""] = jnp.sum(kgrid.kpoints * weights)
            return result

        gradient_gate(loss, points, modes=("rev",))
