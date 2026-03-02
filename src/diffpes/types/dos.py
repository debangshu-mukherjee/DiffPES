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
        dos: DensityOfStates = cls(*children)
        return dos


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

    Extended Summary
    ----------------
    Stores the full DOS data from a VASP DOSCAR file including
    spin-resolved total DOS, integrated DOS, and per-atom
    site-projected DOS. Returned by ``read_doscar`` when
    ``return_mode="full"``. This is the comprehensive counterpart to
    the simpler :class:`DensityOfStates`, which only stores a single
    total-DOS channel.

    This class is registered as a JAX PyTree via
    ``@register_pytree_node_class``. All numeric fields (seven
    arrays, some optional) are stored as children visible to JAX
    tracing, while ``natoms`` is a plain Python ``int`` stored as
    auxiliary data because it is a structural constant that JAX
    cannot trace.

    Attributes
    ----------
    energy : Float[Array, " E"]
        Energy axis in eV, shared by all DOS channels.
        JAX-traced (differentiable).
    total_dos_up : Float[Array, " E"]
        Total DOS for spin-up channel (or the only channel if
        ISPIN=1). Units are states/eV. JAX-traced (differentiable).
    total_dos_down : Optional[Float[Array, " E"]]
        Total DOS for spin-down channel, or ``None`` if the
        calculation is non-spin-polarized (ISPIN=1). Units are
        states/eV. JAX-traced when present.
    integrated_dos_up : Float[Array, " E"]
        Integrated (cumulative) DOS for spin-up channel. The value
        at the Fermi energy gives the number of electrons for this
        spin channel. JAX-traced (differentiable).
    integrated_dos_down : Optional[Float[Array, " E"]]
        Integrated DOS for spin-down channel, or ``None`` if
        ISPIN=1. JAX-traced when present.
    pdos : Optional[Float[Array, "A E C"]]
        Per-atom site-projected DOS. A is the number of atoms, E the
        number of energy points, and C the number of orbital columns
        (depends on LORBIT setting in VASP, typically 9 or 16).
        ``None`` if no PDOS blocks are present in the DOSCAR file.
        JAX-traced when present.
    fermi_energy : Float[Array, " "]
        Fermi level energy in eV. A 0-D scalar array.
        JAX-traced (differentiable).
    natoms : int
        Number of atoms in the unit cell. Stored as auxiliary data
        (not a JAX array) because it is a structural constant.

    Notes
    -----
    Registered as a JAX PyTree with ``@register_pytree_node_class``.
    The ``natoms`` field is a Python ``int`` (not a JAX array) and
    is therefore stored as auxiliary data rather than as a child. JAX
    treats auxiliary data as a compile-time constant: changing
    ``natoms`` triggers recompilation of any ``jit``-compiled
    function that receives this PyTree.

    Optional fields (``total_dos_down``, ``integrated_dos_down``,
    ``pdos``) may be ``None`` for non-spin-polarized calculations or
    DOSCAR files without projected DOS. JAX handles ``None`` leaves
    transparently.

    See Also
    --------
    DensityOfStates : Simplified single-channel variant.
    make_full_density_of_states : Factory function with validation
        and float64 casting.
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

        Separates JAX-traced arrays (children) from static Python
        values (auxiliary data) for ``jax.tree_util`` compatibility.

        Implementation Logic
        --------------------
        1. **Children** (JAX arrays, participate in autodiff):
           ``(energy, total_dos_up, total_dos_down, integrated_dos_up,
           integrated_dos_down, pdos, fermi_energy)`` -- seven fields,
           some of which may be ``None``. JAX treats ``None`` leaves
           as empty subtrees and skips them during tracing.
        2. **Auxiliary data** (static, not traced by JAX):
           ``natoms`` -- a Python ``int`` representing the number of
           atoms. Stored as aux_data because JAX cannot trace Python
           ints and because this value is a structural constant.

        Returns
        -------
        children : tuple of (jax.Array or None)
            ``(energy, total_dos_up, total_dos_down,
            integrated_dos_up, integrated_dos_down, pdos,
            fermi_energy)``.
        aux_data : int
            The ``natoms`` value, stored outside JAX tracing.
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
        """Reconstruct a ``FullDensityOfStates`` from flattened components.

        Inverse of :meth:`tree_flatten`. JAX calls this method
        automatically when unflattening a PyTree after a
        transformation (e.g., inside ``jax.jit`` or ``jax.grad``).

        Implementation Logic
        --------------------
        1. Unpack ``children`` into seven array-valued fields:
           ``(energy, total_dos_up, total_dos_down, integrated_dos_up,
           integrated_dos_down, pdos, fermi_energy)``.
        2. Receive ``aux_data`` as the ``natoms`` Python int.
        3. Pass all eight fields to the constructor, restoring
           ``natoms`` from ``aux_data`` into its correct position.

        Parameters
        ----------
        aux_data : int
            The ``natoms`` value recovered from auxiliary data.
        children : tuple of (jax.Array or None)
            ``(energy, total_dos_up, total_dos_down,
            integrated_dos_up, integrated_dos_down, pdos,
            fermi_energy)`` as returned by :meth:`tree_flatten`.

        Returns
        -------
        dos : FullDensityOfStates
            Reconstructed instance with identical data.
        """
        energy: Float[Array, " E"]
        total_dos_up: Float[Array, " E"]
        total_dos_down: Optional[Float[Array, " E"]]
        integrated_dos_up: Float[Array, " E"]
        integrated_dos_down: Optional[Float[Array, " E"]]
        pdos: Optional[Float[Array, "A E C"]]
        fermi_energy: Float[Array, " "]
        (
            energy,
            total_dos_up,
            total_dos_down,
            integrated_dos_up,
            integrated_dos_down,
            pdos,
            fermi_energy,
        ) = children
        dos: FullDensityOfStates = cls(
            energy=energy,
            total_dos_up=total_dos_up,
            total_dos_down=total_dos_down,
            integrated_dos_up=integrated_dos_up,
            integrated_dos_down=integrated_dos_down,
            pdos=pdos,
            fermi_energy=fermi_energy,
            natoms=aux_data,
        )
        return dos


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

    Extended Summary
    ----------------
    Factory function that validates and normalises full density of
    states data before constructing a ``FullDensityOfStates`` PyTree.
    All non-None numeric arrays are cast to ``float64`` for numerical
    stability. Optional fields (``total_dos_down``,
    ``integrated_dos_down``, ``pdos``) are cast only when present,
    preserving the ``None`` sentinel for non-spin-polarized or
    non-projected calculations.

    The function is decorated with ``@jaxtyped(typechecker=beartype)``
    so that the energy dimension *E* is checked for consistency
    across all provided arrays at call time.

    Use this factory when you need the complete DOSCAR output
    including spin-resolved channels and site-projected DOS. For the
    simpler case of a single total-DOS channel, prefer
    :func:`make_density_of_states` instead.

    Implementation Logic
    --------------------
    1. **Cast energy** to ``jnp.float64`` via ``jnp.asarray``.
    2. **Cast total_dos_up** to ``jnp.float64``.
    3. **Cast integrated_dos_up** to ``jnp.float64``.
    4. **Cast fermi_energy** scalar to a 0-D ``jnp.float64`` array.
    5. **Cast optional fields** (``total_dos_down``,
       ``integrated_dos_down``, ``pdos``) to ``jnp.float64`` when
       not ``None``; otherwise leave as ``None``.
    6. **Keep natoms** as a Python ``int`` (not cast). It is stored
       as auxiliary data in the resulting PyTree.
    7. **Construct** the ``FullDensityOfStates`` NamedTuple from all
       fields and return it.

    Parameters
    ----------
    energy : Float[Array, " E"]
        Energy axis in eV.
    total_dos_up : Float[Array, " E"]
        Spin-up total DOS (states/eV).
    integrated_dos_up : Float[Array, " E"]
        Spin-up integrated (cumulative) DOS.
    fermi_energy : ScalarNumeric, optional
        Fermi level in eV. Default is 0.0.
    total_dos_down : Optional[Float[Array, " E"]], optional
        Spin-down total DOS. Default is None (ISPIN=1).
    integrated_dos_down : Optional[Float[Array, " E"]], optional
        Spin-down integrated DOS. Default is None.
    pdos : Optional[Float[Array, "A E C"]], optional
        Per-atom site-projected DOS with C orbital columns.
        Default is None.
    natoms : int, optional
        Number of atoms in the unit cell. Default is 0.

    Returns
    -------
    dos : FullDensityOfStates
        Validated full density of states with all non-None arrays
        in ``float64``.

    See Also
    --------
    make_density_of_states : Factory for the simplified variant.
    FullDensityOfStates : The PyTree class constructed by this
        factory.
    """
    energy_arr: Float[Array, " E"] = jnp.asarray(energy, dtype=jnp.float64)
    up_arr: Float[Array, " E"] = jnp.asarray(total_dos_up, dtype=jnp.float64)
    int_up_arr: Float[Array, " E"] = jnp.asarray(integrated_dos_up, dtype=jnp.float64)
    fermi_arr: Float[Array, " "] = jnp.asarray(fermi_energy, dtype=jnp.float64)
    down_arr: Optional[Float[Array, " E"]] = None
    if total_dos_down is not None:
        down_arr = jnp.asarray(total_dos_down, dtype=jnp.float64)
    int_down_arr: Optional[Float[Array, " E"]] = None
    if integrated_dos_down is not None:
        int_down_arr = jnp.asarray(integrated_dos_down, dtype=jnp.float64)
    pdos_arr: Optional[Float[Array, "A E C"]] = None
    if pdos is not None:
        pdos_arr = jnp.asarray(pdos, dtype=jnp.float64)
    dos: FullDensityOfStates = FullDensityOfStates(
        energy=energy_arr,
        total_dos_up=up_arr,
        total_dos_down=down_arr,
        integrated_dos_up=int_up_arr,
        integrated_dos_down=int_down_arr,
        pdos=pdos_arr,
        fermi_energy=fermi_arr,
        natoms=natoms,
    )
    return dos


__all__: list[str] = [
    "DensityOfStates",
    "FullDensityOfStates",
    "make_density_of_states",
    "make_full_density_of_states",
]
