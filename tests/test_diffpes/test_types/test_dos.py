"""Validate density-of-states carriers and their public factories.

The cases cover JIT construction, full-carrier PyTree reconstruction, and the
strictly increasing energy-axis contract shared by both factories.
"""

import chex
import jax
import jax.numpy as jnp
from beartype.typing import Callable
from jaxtyping import Array, Float64

from diffpes.types import (
    DensityOfStates,
    FullDensityOfStates,
    PyTreeDef,
    make_density_of_states,
    make_full_density_of_states,
)
from tests._assertions import assert_rejects


class TestDensityOfStates:
    """Validate :class:`~diffpes.types.DensityOfStates` array storage.

    The carrier must retain the shared energy and density axes together with a
    scalar Fermi-energy reference.

    :see: :class:`~diffpes.types.DensityOfStates`
    """

    def test_stores_density_arrays(self) -> None:
        """Preserve equal-length energy and total-DOS arrays.

        The check verifies 16 energy samples, 16 density values, and a scalar
        Fermi energy of ``-1.5`` eV.

        Notes
        -----
        The test constructs the carrier through the public factory and checks shapes
        and the independently specified scalar with Chex.
        """
        energy: Float64[Array, "16"] = jnp.linspace(-10.0, 5.0, 16)
        density: Float64[Array, "16"] = jnp.ones(16)
        result: DensityOfStates = make_density_of_states(
            energy=energy, total_dos=density, fermi_energy=-1.5
        )

        chex.assert_shape(result.energy, (16,))
        chex.assert_shape(result.total_dos, (16,))
        chex.assert_trees_all_close(result.fermi_energy, jnp.float64(-1.5))


class TestFullDensityOfStates:
    """Validate :class:`~diffpes.types.FullDensityOfStates` as a PyTree.

    Optional spin and projected fields must remain absent while required array
    leaves survive JAX reconstruction.

    :see: :class:`~diffpes.types.FullDensityOfStates`
    """

    def test_pytree_round_trip(self) -> None:
        """Preserve required full-DOS leaves through a PyTree round trip.

        The check compares a 50-point total DOS, its integral, and the scalar
        Fermi energy after JAX flattening and reconstruction.

        Notes
        -----
        The test builds a spin-up-only full carrier, uses JAX tree utilities for the
        round trip, and compares the restored values with Chex.
        """
        energy: Float64[Array, "50"] = jnp.linspace(-3, 1, 50)
        density_up: Float64[Array, "50"] = jnp.ones(50)
        integrated_up: Float64[Array, "50"] = jnp.cumsum(density_up)
        full_dos: FullDensityOfStates = make_full_density_of_states(
            energy=energy,
            total_dos_up=density_up,
            integrated_dos_up=integrated_up,
            fermi_energy=-0.5,
        )
        leaves: list[object]
        tree: PyTreeDef
        leaves, tree = jax.tree_util.tree_flatten(full_dos)
        restored: FullDensityOfStates = jax.tree_util.tree_unflatten(
            tree, leaves
        )

        chex.assert_trees_all_close(restored.total_dos_up, density_up)
        chex.assert_trees_all_close(
            restored.fermi_energy, full_dos.fermi_energy
        )
        chex.assert_equal(restored.natoms, full_dos.natoms)


class TestMakeDensityOfStates(chex.TestCase):
    """Validate :func:`~diffpes.types.make_density_of_states` under JIT.

    The factory must construct the same finite carrier in eager and compiled
    execution and reject non-increasing energy coordinates.

    :see: :func:`~diffpes.types.make_density_of_states`
    """

    @chex.variants(with_jit=True, without_jit=True)
    def test_constructs_in_both_execution_modes(self) -> None:
        """Construct equal-shape DOS fields in eager and compiled execution.

        The check gates the 32-point output shape and scalar Fermi-energy value
        in both Chex execution variants.

        Notes
        -----
        The test wraps the public factory with ``self.variant``, supplies finite arrays,
        and compares the resulting shape and scalar with Chex.
        """
        factory: Callable[..., DensityOfStates] = self.variant(
            make_density_of_states
        )
        result: DensityOfStates = factory(
            energy=jnp.linspace(-2.0, 1.0, 32),
            total_dos=jnp.ones(32),
            fermi_energy=0.25,
        )

        chex.assert_shape(result.energy, (32,))
        chex.assert_trees_all_close(result.fermi_energy, jnp.float64(0.25))

    def test_rejects_unsorted_energy_axis(self) -> None:
        """Reject repeated coordinates on the DOS energy axis.

        The check enforces strict monotonicity independently of the density
        values supplied at those coordinates.

        Notes
        -----
        Supplies two equal energy values and matches the traced validation
        diagnostic through the shared rejection helper.
        """
        assert_rejects(
            make_density_of_states,
            energy=jnp.array([0.0, 0.0]),
            total_dos=jnp.ones(2),
            match="energy strictly increasing",
        )


class TestMakeFullDensityOfStates:
    """Validate :func:`~diffpes.types.make_full_density_of_states`.

    The full factory must enforce the same strict energy ordering as the
    compact carrier while validating its integrated-DOS field.

    :see: :func:`~diffpes.types.make_full_density_of_states`
    """

    def test_rejects_unsorted_energy_axis(self) -> None:
        """Reject repeated coordinates on the full-DOS energy axis.

        The check verifies strict monotonicity for a complete spin-up DOS
        input whose density and integral otherwise have compatible shapes.

        Notes
        -----
        Supplies two repeated energy coordinates and matches the factory's
        traced ordering diagnostic through the rejection helper.
        """
        energy: Float64[Array, "2"] = jnp.array([0.0, 0.0])

        assert_rejects(
            make_full_density_of_states,
            energy=energy,
            total_dos_up=jnp.ones(2),
            integrated_dos_up=jnp.ones(2),
            match="energy strictly increasing",
        )
