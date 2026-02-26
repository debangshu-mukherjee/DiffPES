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

    Stores total density of states (DOS) data parsed from VASP DOSCAR
    files. The DOS is represented as a pair of 1-D arrays sharing the
    same energy axis, together with a scalar Fermi energy reference.

    This class is registered as a JAX PyTree via
    ``@register_pytree_node_class``, allowing instances to be passed
    through ``jax.jit``, ``jax.grad``, ``jax.vmap``, and other JAX
    transformations. All fields are JAX arrays (no auxiliary data),
    so every field participates in autodiff tracing.

    Attributes
    ----------
    energy : Float[Array, " E"]
        Energy axis in eV.
    total_dos : Float[Array, " E"]
        Total density of states.
    fermi_energy : Float[Array, " "]
        Fermi level energy in eV.

    Notes
    -----
    Registered as a JAX PyTree with ``@register_pytree_node_class``.
    All three fields are JAX arrays and stored as children (no
    auxiliary data). This means every field is visible to JAX's
    tracing and transformation machinery.

    See Also
    --------
    make_density_of_states : Factory function with validation and
        float64 casting.
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
        """Flatten into JAX children and auxiliary data.

        Separates the PyTree into children (JAX-traced arrays) and
        auxiliary data (static Python values). For DensityOfStates,
        all fields are JAX arrays, so the auxiliary data is ``None``.

        Implementation Logic
        --------------------
        1. **Children** (JAX arrays, participate in autodiff):
           ``(energy, total_dos, fermi_energy)``
        2. **Auxiliary data** (static, not traced): ``None``
           -- No static fields exist on this type.

        Returns
        -------
        children : tuple of Array
            Tuple of ``(energy, total_dos, fermi_energy)`` JAX arrays.
        aux_data : None
            No auxiliary data for this type.
        """
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
        """Reconstruct a DensityOfStates from flattened components.

        Inverse of :meth:`tree_flatten`. JAX calls this method
        automatically when unflattening a PyTree after a
        transformation (e.g., inside ``jax.jit`` or ``jax.grad``).

        Implementation Logic
        --------------------
        1. Receive ``children`` tuple of three JAX arrays and
           ``_aux_data = None``.
        2. Splat ``children`` directly into the constructor via
           ``cls(*children)`` because field order matches the
           children order from :meth:`tree_flatten`.

        Parameters
        ----------
        _aux_data : None
            Unused -- DensityOfStates has no auxiliary data.
        children : tuple of Array
            Tuple of ``(energy, total_dos, fermi_energy)`` JAX arrays.

        Returns
        -------
        dos : DensityOfStates
            Reconstructed instance with identical data.
        """
        return cls(*children)


@jaxtyped(typechecker=beartype)
def make_density_of_states(
    energy: Float[Array, " E"],
    total_dos: Float[Array, " E"],
    fermi_energy: ScalarNumeric = 0.0,
) -> DensityOfStates:
    """Create a validated DensityOfStates instance.

    Validates and normalises the inputs before constructing the
    DensityOfStates PyTree. All numeric inputs are cast to float64
    JAX arrays so that downstream JAX transformations operate at
    double precision without silent dtype promotion surprises.

    Implementation Logic
    --------------------
    1. **Cast energy** to ``jnp.float64`` via ``jnp.asarray``.
    2. **Cast total_dos** to ``jnp.float64`` via ``jnp.asarray``.
    3. **Cast fermi_energy** scalar to a 0-D ``jnp.float64`` array.
    4. **Construct** the ``DensityOfStates`` NamedTuple from the
       three validated arrays and return it.

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

    See Also
    --------
    DensityOfStates : The PyTree class constructed by this factory.
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
