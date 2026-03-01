"""Self-energy configuration data structures.

Extended Summary
----------------
Defines the PyTree type for energy-dependent self-energy
(lifetime broadening) used by the differentiable forward simulator.

Routine Listings
----------------
:class:`SelfEnergyConfig`
    PyTree for energy-dependent self-energy.
:func:`make_self_energy_config`
    Factory for SelfEnergyConfig.

Notes
-----
The self-energy coefficients are differentiable (JAX children)
while the mode string is static (auxiliary data) because it
selects different code branches.
"""

import jax.numpy as jnp
from beartype import beartype
from beartype.typing import NamedTuple, Optional, Tuple
from jax.tree_util import register_pytree_node_class
from jaxtyping import Array, Float, jaxtyped

from .aliases import ScalarFloat


@register_pytree_node_class
class SelfEnergyConfig(NamedTuple):
    """PyTree for energy-dependent self-energy (lifetime broadening).

    Models the imaginary part of the electronic self-energy as a
    function of energy. In the forward simulator this replaces the
    scalar ``gamma`` with an energy-dependent broadening.

    Attributes
    ----------
    coefficients : Float[Array, " P"]
        Parameters for the self-energy model. Differentiable.
        - mode="constant": P=1, ``[gamma]``.
        - mode="polynomial": P=degree+1, ``[a0, a1, ...]``.
        - mode="tabulated": P=N, ``[gamma_1, ..., gamma_N]``.
    energy_nodes : Optional[Float[Array, " P"]]
        Energy grid for tabulated mode. None for other modes.
    mode : str
        One of ``"constant"``, ``"polynomial"``, ``"tabulated"``.
        Static (auxiliary data).

    Notes
    -----
    Registered as a JAX PyTree with ``@register_pytree_node_class``.
    ``coefficients`` and ``energy_nodes`` are children;
    ``mode`` is auxiliary data.
    """

    coefficients: Float[Array, " P"]
    energy_nodes: Optional[Float[Array, " P"]]
    mode: str

    def tree_flatten(
        self,
    ) -> Tuple[
        Tuple[Float[Array, " P"], Optional[Float[Array, " P"]]],
        str,
    ]:
        """Flatten into JAX children and auxiliary data.

        Returns
        -------
        children : tuple
            ``(coefficients, energy_nodes)``.
        aux_data : str
            The mode string.
        """
        return ((self.coefficients, self.energy_nodes), self.mode)

    @classmethod
    def tree_unflatten(
        cls,
        aux_data: str,
        children: Tuple[Float[Array, " P"], Optional[Float[Array, " P"]]],
    ) -> "SelfEnergyConfig":
        """Reconstruct a SelfEnergyConfig from flattened components.

        Parameters
        ----------
        aux_data : str
            The mode string.
        children : tuple
            ``(coefficients, energy_nodes)``.

        Returns
        -------
        config : SelfEnergyConfig
            Reconstructed instance.
        """
        coefficients, energy_nodes = children
        return cls(
            coefficients=coefficients,
            energy_nodes=energy_nodes,
            mode=aux_data,
        )


@jaxtyped(typechecker=beartype)
def make_self_energy_config(
    gamma: ScalarFloat = 0.1,
    mode: str = "constant",
    coefficients: Optional[Float[Array, " P"]] = None,
    energy_nodes: Optional[Float[Array, " P"]] = None,
) -> SelfEnergyConfig:
    """Create a validated SelfEnergyConfig instance.

    Parameters
    ----------
    gamma : ScalarFloat, optional
        Constant broadening in eV. Used when ``coefficients`` is
        None and mode is ``"constant"``. Default is 0.1.
    mode : str, optional
        Self-energy model. Default is ``"constant"``.
    coefficients : Float[Array, " P"], optional
        Explicit coefficients. If None, uses ``[gamma]``.
    energy_nodes : Float[Array, " P"], optional
        Energy grid for tabulated mode.

    Returns
    -------
    config : SelfEnergyConfig
        Validated self-energy configuration.
    """
    if mode not in ("constant", "polynomial", "tabulated"):
        msg = f"mode must be 'constant', 'polynomial', or 'tabulated', got '{mode}'"
        raise ValueError(msg)
    if coefficients is None:
        coeff_arr = jnp.asarray([gamma], dtype=jnp.float64)
    else:
        coeff_arr = jnp.asarray(coefficients, dtype=jnp.float64)
    nodes_arr: Optional[Float[Array, " P"]] = None
    if energy_nodes is not None:
        nodes_arr = jnp.asarray(energy_nodes, dtype=jnp.float64)
    if mode == "tabulated" and nodes_arr is None:
        msg = "energy_nodes required for tabulated mode"
        raise ValueError(msg)
    return SelfEnergyConfig(
        coefficients=coeff_arr,
        energy_nodes=nodes_arr,
        mode=mode,
    )


__all__: list[str] = [
    "SelfEnergyConfig",
    "make_self_energy_config",
]
