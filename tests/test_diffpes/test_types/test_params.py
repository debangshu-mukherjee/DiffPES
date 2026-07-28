"""Validate simulation-parameter carriers and factories.

The cases cover PyTree reconstruction, documented defaults, data-derived
energy windows, and rejection of invalid parameters.
"""

import chex
import jax
import jax.numpy as jnp

from diffpes.types import (
    SimulationParams,
    make_expanded_simulation_params,
    make_simulation_params,
)
from tests._assertions import assert_rejects


class TestSimulationParams:
    """Validate :class:`~diffpes.types.SimulationParams` as a JAX PyTree.

    Differentiable broadening leaves and the static fidelity must survive JAX
    flattening and reconstruction.

    :see: :class:`~diffpes.types.SimulationParams`
    """

    def test_pytree_round_trip(self) -> None:
        """Preserve broadening and fidelity fields through reconstruction.

        The check compares the 0.04 eV Gaussian width and 25,000-point static
        fidelity before and after a JAX PyTree round trip.

        Notes
        -----
        The test constructs default parameters, flattens and unflattens them with JAX,
        and compares traced and static fields independently.
        """
        params: SimulationParams = make_simulation_params()
        leaves: list[object]
        tree: jax.tree_util.PyTreeDef
        leaves, tree = jax.tree.flatten(params)
        restored: SimulationParams = jax.tree.unflatten(tree, leaves)

        chex.assert_trees_all_close(restored.sigma, params.sigma)
        chex.assert_equal(restored.fidelity, params.fidelity)


class TestMakeSimulationParams:
    """Validate :func:`~diffpes.types.make_simulation_params`.

    The factory must provide the documented ARPES defaults, preserve custom
    values, and reject nonphysical windows and broadenings.

    :see: :func:`~diffpes.types.make_simulation_params`
    """

    def test_constructs_documented_defaults(self) -> None:
        """Construct the documented energy window and broadening defaults.

        The check verifies ``[-3, 1]`` eV, 25,000 samples, and Gaussian and
        Lorentzian widths of 0.04 eV and 0.1 eV.

        Notes
        -----
        The test calls the factory without arguments and compares all defining default
        fields with independent constants using Chex.
        """
        params: SimulationParams = make_simulation_params()

        chex.assert_trees_all_close(params.energy_min, jnp.float64(-3.0))
        chex.assert_trees_all_close(params.energy_max, jnp.float64(1.0))
        chex.assert_equal(params.fidelity, 25000)
        chex.assert_trees_all_close(params.sigma, jnp.float64(0.04))
        chex.assert_trees_all_close(params.gamma, jnp.float64(0.1))

    def test_rejects_nonphysical_parameters(self) -> None:
        """Reject negative broadening and a reversed energy window.

        The check covers both independent physical validation boundaries of
        the base simulation-parameter factory.

        Notes
        -----
        The test uses the eager-and-JIT rejection helper with a negative Gaussian width
        and then with ``energy_min`` greater than ``energy_max``.
        """
        assert_rejects(
            make_simulation_params, sigma=-1.0, match="sigma must be positive"
        )
        assert_rejects(
            make_simulation_params,
            energy_min=5.0,
            energy_max=-5.0,
            match="energy_min must be less than energy_max",
        )


class TestMakeExpandedSimulationParams:
    """Validate :func:`~diffpes.types.make_expanded_simulation_params`.

    The factory must derive its energy window from band extrema and symmetric
    padding. It must retain differentiable dependence on these inputs.

    :see: :func:`~diffpes.types.make_expanded_simulation_params`
    """

    def test_derives_energy_window_from_bands(self) -> None:
        """Expand band extrema by the requested energy padding.

        For bands spanning ``[-2, 3]`` eV and padding 0.5 eV, the expected
        simulation window is ``[-2.5, 3.5]`` eV.

        Notes
        -----
        Supplies a two-by-two band array, constructs the expanded parameters,
        and compares both derived bounds with the analytic extrema.
        """
        eigenbands: jax.Array = jnp.array([[-2.0, 0.0], [1.0, 3.0]])
        params: SimulationParams = make_expanded_simulation_params(
            eigenbands, energy_padding=0.5
        )

        chex.assert_trees_all_close(params.energy_min, jnp.float64(-2.5))
        chex.assert_trees_all_close(params.energy_max, jnp.float64(3.5))
