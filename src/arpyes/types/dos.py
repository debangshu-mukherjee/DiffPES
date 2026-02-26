"""Density of states data structure.

Extended Summary
----------------
Defines the :class:`DensityOfStates` PyTree for storing total
and projected density of states from VASP DOSCAR files.

Routine Listings
----------------
:class:`DensityOfStates`
    PyTree for density of states data.
:func:`make_density_of_states`
    Factory function for DensityOfStates.

Notes
-----
All energy values are in electron-volts (eV).
"""

import jax.numpy as jnp
from beartype import beartype
from beartype.typing import NamedTuple, Tuple
from jax.tree_util import register_pytree_node_class
from jaxtyping import Array, Float, jaxtyped

from .aliases import ScalarNumeric


@register_pytree_node_class
class DensityOfStates(NamedTuple):
    """PyTree for density of states.

    Attributes
    ----------
    energy : Float[Array, " E"]
        Energy axis in eV.
    total_dos : Float[Array, " E"]
        Total density of states.
    fermi_energy : Float[Array, " "]
        Fermi level energy in eV.
    """

    energy: Float[Array, " E"]
    total_dos: Float[Array, " E"]
    fermi_energy: Float[Array, " "]

    def tree_flatten(
        self,
    ) -> Tuple[
        Tuple[
            Float[Array, " E"],
            Float[Array, " E"],
            Float[Array, " "],
        ],
        None,
    ]:
        """Flatten into JAX children and auxiliary data."""
        return (
            (self.energy, self.total_dos, self.fermi_energy),
            None,
        )

    @classmethod
    def tree_unflatten(
        cls,
        _aux_data: None,
        children: Tuple[
            Float[Array, " E"],
            Float[Array, " E"],
            Float[Array, " "],
        ],
    ) -> "DensityOfStates":
        """Reconstruct from flattened components."""
        return cls(*children)


@jaxtyped(typechecker=beartype)
def make_density_of_states(
    energy: Float[Array, " E"],
    total_dos: Float[Array, " E"],
    fermi_energy: ScalarNumeric = 0.0,
) -> DensityOfStates:
    """Create a validated DensityOfStates instance.

    Parameters
    ----------
    energy : Float[Array, " E"]
        Energy axis in eV.
    total_dos : Float[Array, " E"]
        Total density of states.
    fermi_energy : ScalarNumeric, optional
        Fermi level in eV. Default is 0.0.

    Returns
    -------
    dos : DensityOfStates
        Validated density of states instance.
    """
    energy_arr: Float[Array, " E"] = jnp.asarray(
        energy, dtype=jnp.float64
    )
    dos_arr: Float[Array, " E"] = jnp.asarray(
        total_dos, dtype=jnp.float64
    )
    fermi_arr: Float[Array, " "] = jnp.asarray(
        fermi_energy, dtype=jnp.float64
    )
    dos: DensityOfStates = DensityOfStates(
        energy=energy_arr,
        total_dos=dos_arr,
        fermi_energy=fermi_arr,
    )
    return dos


__all__: list[str] = [
    "DensityOfStates",
    "make_density_of_states",
]
