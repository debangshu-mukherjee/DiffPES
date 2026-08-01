"""Define density-of-states data structures.

Extended Summary
----------------
This module defines the :class:`DensityOfStates` and
:class:`FullDensityOfStates` PyTrees. These types store total and projected
density of states from VASP DOSCAR files.

Routine Listings
----------------
:class:`DensityOfStates`
    Store density-of-states data in a JAX PyTree.
:class:`FullDensityOfStates`
    Store spin-resolved total and projected DOS data in a JAX PyTree.
:func:`make_density_of_states`
    Create a validated DensityOfStates instance.
:func:`make_full_density_of_states`
    Create a validated ``FullDensityOfStates`` instance.

Notes
-----
All energy values are in electron-volts (eV).
"""

import equinox as eqx
import jax.numpy as jnp
from beartype import beartype
from beartype.typing import Optional
from jaxtyping import Array, Float64, jaxtyped

from .aliases import ScalarNumeric


class DensityOfStates(eqx.Module):
    """Store density-of-states data in a JAX PyTree.

    This type stores total density-of-states (DOS) data from VASP DOSCAR
    files. Two 1-D arrays share the same energy axis. A scalar value specifies
    the Fermi energy.

    This JAX PyTree passes through ``jax.jit``, ``jax.grad``, ``jax.vmap``,
    and other JAX transformations. All fields contain JAX arrays and no
    auxiliary data. Therefore, every field participates in autodiff tracing.


    :see: :class:`~.test_dos.TestDensityOfStates`

    Attributes
    ----------
    energy : Float64[Array, " E"]
        Energy axis in eV.
    total_dos : Float64[Array, " E"]
        Total density of states.
    fermi_energy : Float64[Array, " "]
        Fermi level energy in eV.

    Notes
    -----
    JAX stores all three array fields as children and uses no auxiliary data.
    Therefore, JAX tracing and transformations can access every field.

    See Also
    --------
    make_density_of_states : Factory function with validation and
        float64 casting.
    """

    energy: Float64[Array, " E"]
    total_dos: Float64[Array, " E"]
    fermi_energy: Float64[Array, " "]


@jaxtyped(typechecker=beartype)
def make_density_of_states(  # noqa: DOC503
    energy: Float64[Array, " Ee"],
    total_dos: Float64[Array, " Ed"],
    fermi_energy: ScalarNumeric = 0.0,
) -> DensityOfStates:
    """Create a validated DensityOfStates instance.

    The factory validates and normalizes the inputs before it constructs the
    DensityOfStates PyTree. It casts all numerical inputs to float64 JAX
    arrays. Downstream JAX transformations therefore use double precision
    without silent dtype promotion.

    :see: :class:`~.test_dos.TestMakeDensityOfStates`

    Implementation Logic
    --------------------
    1. **Prepare the normalized values**::

           energy_arr = jnp.asarray(energy, dtype=jnp.float64)

       This expression gives the later validation steps a stable shape and
       dtype.

    2. **Apply static validation**::

           energy_arr.shape[0] != dos_arr.shape[0]

       This predicate rejects invalid structure before JAX traces the
       numerical checks.

    3. **Apply traced validation**::

           ~jnp.all(jnp.diff(energy_arr) > 0.0)

       This predicate remains active during eager and compiled execution.

    4. **Return the named instance**::

           return dos

       The explicit name keeps the implementation and the Returns section
       synchronized.

    Parameters
    ----------
    energy : Float64[Array, " Ee"]
        Energy axis in eV.
    total_dos : Float64[Array, " Ed"]
        Total density of states.
    fermi_energy : ScalarNumeric, optional
        Fermi level in eV. Default is 0.0.

    Returns
    -------
    dos : DensityOfStates
        Validated density of states instance.

    Raises
    ------
    ValueError
        If the energy and DOS channel lengths disagree.
    EquinoxRuntimeError
        If the energy axis is not strictly increasing or DOS values are
        non-finite.

    Notes
    -----
    Static validation raises ``ValueError`` before traced construction when
    the energy and DOS lengths differ. Traced validation uses
    ``eqx.error_if`` and raises ``EquinoxRuntimeError`` when the energy axis
    is not increasing or the DOS contains a non-finite value.

    See Also
    --------
    DensityOfStates : The PyTree class constructed by this factory.
    """
    energy_arr: Float64[Array, " E"] = jnp.asarray(energy, dtype=jnp.float64)
    dos_arr: Float64[Array, " E"] = jnp.asarray(total_dos, dtype=jnp.float64)
    fermi_arr: Float64[Array, " "] = jnp.asarray(
        fermi_energy, dtype=jnp.float64
    )

    if energy_arr.shape[0] != dos_arr.shape[0]:
        raise ValueError(
            "make_density_of_states: energy and total_dos lengths disagree"
        )

    def validate_and_create() -> DensityOfStates:
        nonlocal dos_arr, energy_arr, fermi_arr
        energy_arr = eqx.error_if(
            energy_arr,
            ~(jnp.all(jnp.diff(energy_arr) > 0.0)),
            "make_density_of_states: energy strictly increasing",
        )
        dos_arr = eqx.error_if(
            dos_arr,
            ~(jnp.all(jnp.isfinite(dos_arr))),
            "make_density_of_states: dos finite",
        )
        validated_dos: DensityOfStates = DensityOfStates(
            energy=energy_arr,
            total_dos=dos_arr,
            fermi_energy=fermi_arr,
        )
        return validated_dos

    dos: DensityOfStates = validate_and_create()
    return dos


class FullDensityOfStates(eqx.Module):
    """Store spin-resolved total and projected DOS data in a JAX PyTree.

    This type stores the full DOS data from a VASP DOSCAR file, including
    spin-resolved total DOS, integrated DOS, and per-atom
    site-projected DOS. ``read_doscar`` returns this type when
    ``return_mode="full"``. This type is the comprehensive counterpart to
    the simpler :class:`DensityOfStates`, which only stores a single
    total-DOS channel.

    JAX stores the array fields as PyTree children. JAX tracing can access
    these required and optional fields. JAX stores ``natoms`` as auxiliary
    data because this Python ``int`` is a structural constant.


    :see: :class:`~.test_dos.TestFullDensityOfStates`

    Attributes
    ----------
    energy : Float64[Array, " E"]
        Energy axis in eV, shared by all DOS channels.
        JAX-traced (differentiable).
    total_dos_up : Float64[Array, " E"]
        Total DOS for spin-up channel (or the only channel if
        ISPIN=1). Units are states/eV. JAX-traced (differentiable).
    total_dos_down : Optional[Float64[Array, " E"]]
        Total DOS for spin-down channel, or ``None`` if the
        calculation is non-spin-polarized (ISPIN=1). Units are
        states/eV. JAX-traced when present.
    integrated_dos_up : Float64[Array, " E"]
        Integrated (cumulative) DOS for spin-up channel. The value
        at the Fermi energy gives the number of electrons for this
        spin channel. JAX-traced (differentiable).
    integrated_dos_down : Optional[Float64[Array, " E"]]
        Integrated DOS for spin-down channel, or ``None`` if
        ISPIN=1. JAX-traced when present.
    pdos : Optional[Float64[Array, "A E C"]]
        Per-atom site-projected DOS. A specifies the atom count. E specifies
        the energy-point count. C specifies the orbital-column count, which
        depends on the VASP LORBIT setting.
        ``None`` if no PDOS blocks are present in the DOSCAR file.
        JAX-traced when present.
    fermi_energy : Float64[Array, " "]
        Fermi level energy in eV. A 0-D scalar array.
        JAX-traced (differentiable).
    natoms : int
        Number of atoms in the unit cell (**static** -- a compile-time
        constant; changing it triggers retracing).

    Notes
    -----
    The ``natoms`` field is a Python ``int`` declared with
    ``eqx.field(static=True)`` rather than as a traced leaf. Changing
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

    energy: Float64[Array, " E"]
    total_dos_up: Float64[Array, " E"]
    total_dos_down: Optional[Float64[Array, " E"]]
    integrated_dos_up: Float64[Array, " E"]
    integrated_dos_down: Optional[Float64[Array, " E"]]
    pdos: Optional[Float64[Array, "A E C"]]
    fermi_energy: Float64[Array, " "]
    natoms: int = eqx.field(static=True)


@jaxtyped(typechecker=beartype)
def make_full_density_of_states(  # noqa: DOC503
    energy: Float64[Array, " Ee"],
    total_dos_up: Float64[Array, " Eu"],
    integrated_dos_up: Float64[Array, " Eiu"],
    fermi_energy: ScalarNumeric = 0.0,
    total_dos_down: Optional[Float64[Array, " Ed"]] = None,
    integrated_dos_down: Optional[Float64[Array, " Eid"]] = None,
    pdos: Optional[Float64[Array, "A Ep C"]] = None,
    natoms: int = 0,
) -> FullDensityOfStates:
    """Create a validated ``FullDensityOfStates`` instance.

    The factory validates and normalizes full density-of-states
    states data before constructing a ``FullDensityOfStates`` PyTree.
    The factory casts all present numerical arrays to ``float64`` for
    numerical stability. It casts optional fields only when they are present.
    Thus, ``None`` continues to identify non-spin-polarized or non-projected
    calculations.

    ``@jaxtyped(typechecker=beartype)`` checks the energy dimension *E* across
    all provided arrays at call time.

    Use this factory when you need the complete DOSCAR output
    including spin-resolved channels and site-projected DOS. For the
    simpler case of a single total-DOS channel, prefer
    :func:`make_density_of_states` instead.

    :see: :class:`~.test_dos.TestMakeFullDensityOfStates`

    Implementation Logic
    --------------------
    1. **Prepare the normalized values**::

           energy_arr = jnp.asarray(energy, dtype=jnp.float64)

       This expression gives the later validation steps a stable shape and
       dtype.

    2. **Apply static validation**::

           up_arr.shape[0] != nenergy

       This predicate rejects invalid structure before JAX traces the
       numerical checks.

    3. **Apply traced validation**::

           ~jnp.all(jnp.diff(energy_arr) > 0.0)

       This predicate remains active during eager and compiled execution.

    4. **Return the named instance**::

           return dos

       The explicit name keeps the implementation and the Returns section
       synchronized.

    Parameters
    ----------
    energy : Float64[Array, " Ee"]
        Energy axis in eV.
    total_dos_up : Float64[Array, " Eu"]
        Spin-up total DOS (states/eV).
    integrated_dos_up : Float64[Array, " Eiu"]
        Spin-up integrated (cumulative) DOS.
    fermi_energy : ScalarNumeric, optional
        Fermi level in eV. Default is 0.0.
    total_dos_down : Optional[Float64[Array, " Ed"]], optional
        Spin-down total DOS. Default is None (ISPIN=1).
    integrated_dos_down : Optional[Float64[Array, " Eid"]], optional
        Spin-down integrated DOS. Default is None.
    pdos : Optional[Float64[Array, "A Ep C"]], optional
        Per-atom site-projected DOS with C orbital columns.
        Default is None.
    natoms : int, optional
        Number of atoms in the unit cell (**static** -- a compile-time
        constant; changing it triggers retracing). Default is 0.

    Returns
    -------
    dos : FullDensityOfStates
        Validated full density of states with all non-None arrays
        in ``float64``.

    Raises
    ------
    ValueError
        If a total, integrated, or projected DOS channel disagrees with the
        energy-axis length.
    EquinoxRuntimeError
        If the energy axis is not strictly increasing or any DOS channel
        contains non-finite values.

    Notes
    -----
    Static validation raises ``ValueError`` before traced construction when
    any present DOS channel disagrees with the energy-axis length. Traced
    validation uses ``eqx.error_if`` and raises ``EquinoxRuntimeError`` when
    the energy axis is not increasing or a present DOS channel is non-finite.

    See Also
    --------
    make_density_of_states : Factory for the simplified variant.
    FullDensityOfStates : The PyTree class constructed by this
        factory.
    """
    energy_arr: Float64[Array, " E"] = jnp.asarray(energy, dtype=jnp.float64)
    up_arr: Float64[Array, " E"] = jnp.asarray(total_dos_up, dtype=jnp.float64)
    int_up_arr: Float64[Array, " E"] = jnp.asarray(
        integrated_dos_up, dtype=jnp.float64
    )
    fermi_arr: Float64[Array, " "] = jnp.asarray(
        fermi_energy, dtype=jnp.float64
    )
    down_arr: Optional[Float64[Array, " E"]] = None
    if total_dos_down is not None:
        down_arr = jnp.asarray(total_dos_down, dtype=jnp.float64)
    int_down_arr: Optional[Float64[Array, " E"]] = None
    if integrated_dos_down is not None:
        int_down_arr = jnp.asarray(integrated_dos_down, dtype=jnp.float64)
    pdos_arr: Optional[Float64[Array, "A E C"]] = None
    if pdos is not None:
        pdos_arr = jnp.asarray(pdos, dtype=jnp.float64)

    nenergy: int = energy_arr.shape[0]
    if up_arr.shape[0] != nenergy:
        raise ValueError(
            "make_full_density_of_states: energy and total_dos_up "
            "lengths disagree"
        )
    if int_up_arr.shape[0] != nenergy:
        raise ValueError(
            "make_full_density_of_states: energy and integrated_dos_up "
            "lengths disagree"
        )
    if down_arr is not None and down_arr.shape[0] != nenergy:
        raise ValueError(
            "make_full_density_of_states: energy and total_dos_down "
            "lengths disagree"
        )
    if int_down_arr is not None and int_down_arr.shape[0] != nenergy:
        raise ValueError(
            "make_full_density_of_states: energy and integrated_dos_down "
            "lengths disagree"
        )
    if pdos_arr is not None and pdos_arr.shape[1] != nenergy:
        raise ValueError(
            "make_full_density_of_states: energy and pdos E axes disagree"
        )

    def validate_and_create() -> FullDensityOfStates:
        nonlocal \
            down_arr, \
            energy_arr, \
            int_down_arr, \
            int_up_arr, \
            pdos_arr, \
            up_arr
        energy_arr = eqx.error_if(
            energy_arr,
            ~(jnp.all(jnp.diff(energy_arr) > 0.0)),
            "make_full_density_of_states: energy strictly increasing",
        )
        up_arr = eqx.error_if(
            up_arr,
            ~(jnp.all(jnp.isfinite(up_arr))),
            "make_full_density_of_states: dos up finite",
        )
        int_up_arr = eqx.error_if(
            int_up_arr,
            ~(jnp.all(jnp.isfinite(int_up_arr))),
            "make_full_density_of_states: integrated dos up finite",
        )
        if down_arr is not None:
            down_arr = eqx.error_if(
                down_arr,
                ~(jnp.all(jnp.isfinite(down_arr))),
                "make_full_density_of_states: dos down finite",
            )
        if int_down_arr is not None:
            int_down_arr = eqx.error_if(
                int_down_arr,
                ~(jnp.all(jnp.isfinite(int_down_arr))),
                "make_full_density_of_states: integrated dos down finite",
            )
        if pdos_arr is not None:
            pdos_arr = eqx.error_if(
                pdos_arr,
                ~(jnp.all(jnp.isfinite(pdos_arr))),
                "make_full_density_of_states: pdos finite",
            )
        validated_dos: FullDensityOfStates = FullDensityOfStates(
            energy=energy_arr,
            total_dos_up=up_arr,
            total_dos_down=down_arr,
            integrated_dos_up=int_up_arr,
            integrated_dos_down=int_down_arr,
            pdos=pdos_arr,
            fermi_energy=fermi_arr,
            natoms=natoms,
        )
        return validated_dos

    dos: FullDensityOfStates = validate_and_create()
    return dos


__all__: list[str] = [
    "DensityOfStates",
    "FullDensityOfStates",
    "make_density_of_states",
    "make_full_density_of_states",
]
