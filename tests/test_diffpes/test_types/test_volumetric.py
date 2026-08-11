"""Validate scalar and spin-orbit volumetric carriers and factories.

The cases cover PyTree reconstruction, default atom counts, and consistency
between static grid metadata and scalar or vector-valued volumetric arrays.
"""

import chex
import jax
import jax.numpy as jnp
from beartype.typing import List

from diffpes.types import (
    PyTreeDef,
    SOCVolumetricData,
    VolumetricData,
    make_soc_volumetric_data,
    make_volumetric_data,
)
from tests._assertions import assert_rejects


class TestVolumetricData:
    """Validate :class:`~diffpes.types.VolumetricData` as a JAX PyTree.

    Charge-density leaves and static grid and species metadata must survive
    flattening and reconstruction together.

    :see: :class:`~diffpes.types.VolumetricData`
    """

    def test_pytree_round_trip(self) -> None:
        """Preserve charge values and static metadata through reconstruction.

        The check compares a ``4 x 4 x 4`` unit charge grid, its grid shape,
        and two species labels after a JAX PyTree round trip.

        Notes
        -----
        The test constructs the scalar carrier through its factory, uses JAX
        tree
        utilities for reconstruction, and compares numerical and static fields.
        """
        volume: VolumetricData = make_volumetric_data(
            lattice=jnp.eye(3),
            coords=jnp.zeros((2, 3)),
            charge=jnp.ones((4, 4, 4)),
            grid_shape=(4, 4, 4),
            symbols=("Fe", "Co"),
        )
        leaves: List[object]
        tree: PyTreeDef
        leaves, tree = jax.tree_util.tree_flatten(volume)
        restored: VolumetricData = jax.tree_util.tree_unflatten(tree, leaves)

        chex.assert_trees_all_close(restored.charge, volume.charge)
        chex.assert_equal(restored.grid_shape, volume.grid_shape)
        chex.assert_equal(restored.symbols, volume.symbols)


class TestSOCVolumetricData:
    """Validate :class:`~diffpes.types.SOCVolumetricData` as a JAX PyTree.

    Scalar and vector magnetization leaves must remain associated with their
    common spin-orbit grid through JAX transformations.

    :see: :class:`~diffpes.types.SOCVolumetricData`
    """

    def test_pytree_round_trip(self) -> None:
        """Preserve charge and magnetization grids through reconstruction.

        The check uses a ``2 x 2 x 2`` scalar grid and a final vector axis of
        length three for the spin-orbit magnetization components.

        Notes
        -----
        The test constructs the SOC carrier, applies a JAX flatten and
        unflatten round
        trip, and compares all numerical grid leaves with Chex.
        """
        volume: SOCVolumetricData = make_soc_volumetric_data(
            lattice=jnp.eye(3),
            coords=jnp.zeros((1, 3)),
            charge=jnp.ones((2, 2, 2)),
            magnetization=jnp.zeros((2, 2, 2)),
            magnetization_vector=jnp.zeros((2, 2, 2, 3)),
            grid_shape=(2, 2, 2),
            symbols=("Fe",),
        )
        leaves: List[object]
        tree: PyTreeDef
        leaves, tree = jax.tree_util.tree_flatten(volume)
        restored: SOCVolumetricData = jax.tree_util.tree_unflatten(
            tree, leaves
        )

        chex.assert_trees_all_close(restored.charge, volume.charge)
        chex.assert_trees_all_close(
            restored.magnetization_vector, volume.magnetization_vector
        )


class TestMakeVolumetricData:
    """Validate :func:`~diffpes.types.make_volumetric_data`.

    The scalar factory must supply an empty integer atom-count array and reject
    charge grids inconsistent with static shape metadata.

    :see: :func:`~diffpes.types.make_volumetric_data`
    """

    def test_supplies_empty_atom_counts(self) -> None:
        """Create a zero-length int32 atom-count array when omitted.

        The check verifies both dtype and shape of the stable array sentinel
        used instead of ``None``.

        Notes
        -----
        The test constructs a one-species volume without ``atom_counts`` and
        uses Chex
        to compare the resulting array shape and dtype.
        """
        volume: VolumetricData = make_volumetric_data(
            lattice=jnp.eye(3),
            coords=jnp.zeros((1, 3)),
            charge=jnp.ones((2, 2, 2)),
            grid_shape=(2, 2, 2),
            symbols=("Fe",),
        )

        chex.assert_shape(volume.atom_counts, (0,))
        assert volume.atom_counts.dtype == jnp.int32

    def test_rejects_grid_shape_mismatch(self) -> None:
        """Reject charge data inconsistent with the declared grid shape.

        The check verifies that a ``2 x 2 x 2`` charge array cannot have a
        ``1 x 1 x 1`` grid label.

        Notes
        -----
        Supplies otherwise compatible geometry and matches the factory's
        static charge-shape diagnostic.
        """
        assert_rejects(
            make_volumetric_data,
            lattice=jnp.eye(3),
            coords=jnp.zeros((1, 3)),
            charge=jnp.zeros((2, 2, 2)),
            grid_shape=(1, 1, 1),
            match="grid_shape must match charge shape",
        )


class TestMakeSOCVolumetricData:
    """Validate :func:`~diffpes.types.make_soc_volumetric_data`.

    The SOC factory must provide the same atom-count sentinel. It must require
    vector magnetization dimensions that match the static grid.

    :see: :func:`~diffpes.types.make_soc_volumetric_data`
    """

    def test_supplies_empty_atom_counts(self) -> None:
        """Create a zero-length int32 atom-count array when omitted.

        The check verifies the stable array sentinel for a minimal SOC volume
        with all four required grid blocks.

        Notes
        -----
        The test constructs a one-point SOC volume without ``atom_counts`` and
        checks
        the resulting array dtype and shape.
        """
        volume: SOCVolumetricData = make_soc_volumetric_data(
            lattice=jnp.eye(3),
            coords=jnp.zeros((1, 3)),
            charge=jnp.ones((1, 1, 1)),
            magnetization=jnp.zeros((1, 1, 1)),
            magnetization_vector=jnp.zeros((1, 1, 1, 3)),
            grid_shape=(1, 1, 1),
            symbols=("Fe",),
        )

        chex.assert_shape(volume.atom_counts, (0,))
        assert volume.atom_counts.dtype == jnp.int32

    def test_rejects_vector_grid_shape_mismatch(self) -> None:
        """Reject vector magnetization with incompatible spatial dimensions.

        The check verifies all three spatial dimensions against the declared
        grid independently of the final three-component axis.

        Notes
        -----
        Supplies a vector grid with a doubled first dimension and matches the
        factory's static spatial-shape diagnostic.
        """
        assert_rejects(
            make_soc_volumetric_data,
            lattice=jnp.eye(3),
            coords=jnp.zeros((1, 3)),
            charge=jnp.zeros((1, 1, 1)),
            magnetization=jnp.zeros((1, 1, 1)),
            magnetization_vector=jnp.zeros((2, 1, 1, 3)),
            grid_shape=(1, 1, 1),
            match="grid_shape must match magnetization_vector spatial shape",
        )
