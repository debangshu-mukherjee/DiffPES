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
from beartype.typing import NamedTuple, Optional, Tuple
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
    energy_arr: Float[Array, " E"] = jnp.asarray(energy, dtype=jnp.float64)
    dos_arr: Float[Array, " E"] = jnp.asarray(total_dos, dtype=jnp.float64)
    fermi_arr: Float[Array, " "] = jnp.asarray(fermi_energy, dtype=jnp.float64)
    dos: DensityOfStates = DensityOfStates(
        energy=energy_arr,
        total_dos=dos_arr,
        fermi_energy=fermi_arr,
    )
    return dos


@register_pytree_node_class
class FullDensityOfStates(NamedTuple):
    """PyTree for complete density of states with spin and PDOS.

    Stores the full DOS data from a VASP DOSCAR file including
    spin-resolved total DOS, integrated DOS, and per-atom
    site-projected DOS. Returned by ``read_doscar`` when
    ``return_mode="full"``.

    Attributes
    ----------
    energy : Float[Array, " E"]
        Energy axis in eV.
    total_dos_up : Float[Array, " E"]
        Total DOS for spin-up (or only channel if non-spin).
    total_dos_down : Optional[Float[Array, " E"]]
        Total DOS for spin-down, or None if ISPIN=1.
    integrated_dos_up : Float[Array, " E"]
        Integrated DOS for spin-up.
    integrated_dos_down : Optional[Float[Array, " E"]]
        Integrated DOS for spin-down, or None if ISPIN=1.
    pdos : Optional[Float[Array, "A E C"]]
        Per-atom projected DOS. A atoms, E energies, C columns.
        None if no PDOS blocks present.
    fermi_energy : Float[Array, " "]
        Fermi level energy in eV.
    natoms : int
        Number of atoms in the cell.
    """

    energy: Float[Array, " E"]
    total_dos_up: Float[Array, " E"]
    total_dos_down: Optional[Float[Array, " E"]]
    integrated_dos_up: Float[Array, " E"]
    integrated_dos_down: Optional[Float[Array, " E"]]
    pdos: Optional[Float[Array, "A E C"]]
    fermi_energy: Float[Array, " "]
    natoms: int

    def tree_flatten(
        self,
    ) -> Tuple[
        Tuple[
            Float[Array, " E"],
            Float[Array, " E"],
            Optional[Float[Array, " E"]],
            Float[Array, " E"],
            Optional[Float[Array, " E"]],
            Optional[Float[Array, "A E C"]],
            Float[Array, " "],
        ],
        int,
    ]:
        """Flatten into JAX leaf arrays and auxiliary data.

        Returns
        -------
        children : tuple of (jax.Array or None)
            All numeric fields.
        aux_data : int
            natoms (static).
        """
        return (
            (
                self.energy,
                self.total_dos_up,
                self.total_dos_down,
                self.integrated_dos_up,
                self.integrated_dos_down,
                self.pdos,
                self.fermi_energy,
            ),
            self.natoms,
        )

    @classmethod
    def tree_unflatten(
        cls,
        aux_data: int,
        children: Tuple[
            Float[Array, " E"],
            Float[Array, " E"],
            Optional[Float[Array, " E"]],
            Float[Array, " E"],
            Optional[Float[Array, " E"]],
            Optional[Float[Array, "A E C"]],
            Float[Array, " "],
        ],
    ) -> "FullDensityOfStates":
        """Reconstruct from flattened components.

        Parameters
        ----------
        aux_data : int
            natoms.
        children : tuple of (jax.Array or None)

        Returns
        -------
        FullDensityOfStates
        """
        (
            energy,
            total_dos_up,
            total_dos_down,
            integrated_dos_up,
            integrated_dos_down,
            pdos,
            fermi_energy,
        ) = children
        return cls(
            energy=energy,
            total_dos_up=total_dos_up,
            total_dos_down=total_dos_down,
            integrated_dos_up=integrated_dos_up,
            integrated_dos_down=integrated_dos_down,
            pdos=pdos,
            fermi_energy=fermi_energy,
            natoms=aux_data,
        )


@jaxtyped(typechecker=beartype)
def make_full_density_of_states(
    energy: Float[Array, " E"],
    total_dos_up: Float[Array, " E"],
    integrated_dos_up: Float[Array, " E"],
    fermi_energy: ScalarNumeric = 0.0,
    total_dos_down: Optional[Float[Array, " E"]] = None,
    integrated_dos_down: Optional[Float[Array, " E"]] = None,
    pdos: Optional[Float[Array, "A E C"]] = None,
    natoms: int = 0,
) -> FullDensityOfStates:
    """Create a validated ``FullDensityOfStates`` instance.

    Parameters
    ----------
    energy : Float[Array, " E"]
        Energy axis in eV.
    total_dos_up : Float[Array, " E"]
        Spin-up total DOS.
    integrated_dos_up : Float[Array, " E"]
        Spin-up integrated DOS.
    fermi_energy : ScalarNumeric, optional
        Fermi level in eV.
    total_dos_down : Optional[Float[Array, " E"]], optional
        Spin-down total DOS.
    integrated_dos_down : Optional[Float[Array, " E"]], optional
        Spin-down integrated DOS.
    pdos : Optional[Float[Array, "A E C"]], optional
        Per-atom projected DOS.
    natoms : int, optional
        Number of atoms.

    Returns
    -------
    dos : FullDensityOfStates
    """
    energy_arr = jnp.asarray(energy, dtype=jnp.float64)
    up_arr = jnp.asarray(total_dos_up, dtype=jnp.float64)
    int_up_arr = jnp.asarray(integrated_dos_up, dtype=jnp.float64)
    fermi_arr = jnp.asarray(fermi_energy, dtype=jnp.float64)
    down_arr = None
    if total_dos_down is not None:
        down_arr = jnp.asarray(total_dos_down, dtype=jnp.float64)
    int_down_arr = None
    if integrated_dos_down is not None:
        int_down_arr = jnp.asarray(integrated_dos_down, dtype=jnp.float64)
    pdos_arr = None
    if pdos is not None:
        pdos_arr = jnp.asarray(pdos, dtype=jnp.float64)
    return FullDensityOfStates(
        energy=energy_arr,
        total_dos_up=up_arr,
        total_dos_down=down_arr,
        integrated_dos_up=int_up_arr,
        integrated_dos_down=int_down_arr,
        pdos=pdos_arr,
        fermi_energy=fermi_arr,
        natoms=natoms,
    )


__all__: list[str] = [
    "DensityOfStates",
    "FullDensityOfStates",
    "make_density_of_states",
    "make_full_density_of_states",
]
