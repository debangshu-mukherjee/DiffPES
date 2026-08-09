"""Define volumetric data structures for VASP CHGCAR files.

Extended Summary
----------------
This module defines PyTree types for real-space volumetric grid data from VASP
CHGCAR files. It provides two variants:

- :class:`VolumetricData` -- for non-SOC calculations (ISPIN=1 or
  ISPIN=2), with an optional scalar magnetization density.
- :class:`SOCVolumetricData` -- for spin-orbit coupling calculations.
  VASP writes four grid blocks: total charge, mx, my, and mz. This type stores
  scalar mz and the complete 3-component magnetization vector.

Routine Listings
----------------
:class:`SOCVolumetricData`
    Store SOC CHGCAR volumetric-grid data in a JAX PyTree.
:class:`VolumetricData`
    Store CHGCAR volumetric-grid data in a JAX PyTree.
:func:`make_soc_volumetric_data`
    Create a validated ``SOCVolumetricData`` instance.
:func:`make_volumetric_data`
    Create a validated ``VolumetricData`` instance.

Notes
-----
All real-space grid data uses the VASP output units. The charge density uses
electrons per unit cell volume, not per Angstrom^3. JAX stores ``grid_shape``
and ``symbols`` as auxiliary data because it cannot trace their Python tuples.
"""

import equinox as eqx
import jax.numpy as jnp
from beartype import beartype
from beartype.typing import Optional, Tuple
from jaxtyping import Array, Float64, Int32, jaxtyped


class VolumetricData(eqx.Module):
    """Store CHGCAR volumetric-grid data in a JAX PyTree.

    This type stores the charge density on a 3-D real-space grid together with
    the crystal lattice needed to interpret the grid coordinates.
    It includes an optional magnetization density grid for a spin-polarized
    CHGCAR calculation (ISPIN=2). The magnetization equals the difference
    between spin-up and spin-down charge densities.

    This class is an immutable :class:`equinox.Module` PyTree. JAX stores the
    numerical fields as children that support tracing. It stores
    ``grid_shape`` and ``symbols`` as auxiliary data because JAX cannot trace
    their Python tuples.


    :see: :class:`~.test_volumetric.TestVolumetricData`

    Attributes
    ----------
    lattice : Float64[Array, "3 3"]
        Real-space lattice vectors as rows, in Angstroms. Defines
        the unit cell geometry for interpreting grid coordinates.
        JAX-traced (differentiable).
    coords : Float64[Array, "N 3"]
        Fractional atomic coordinates for all N atoms in the cell.
        JAX-traced (differentiable).
    charge : Float64[Array, "Nx Ny Nz"]
        Charge density on the 3-D real-space grid, in units of
        electrons per unit cell volume (VASP convention). Nx, Ny, Nz
        are the grid dimensions along the three lattice directions.
        JAX-traced (differentiable).
    magnetization : Optional[Float64[Array, "Nx Ny Nz"]]
        Scalar magnetization density (spin-up minus spin-down), or
        ``None`` for non-spin-polarized calculations (ISPIN=1).
        Same units and grid as ``charge``. JAX-traced when present.
    atom_counts : Int32[Array, " S"]
        Number of atoms per species, with S = number of species.
        JAX-traced (differentiable, int32).
    grid_shape : Tuple[int, int, int]
        Grid dimensions ``(Nx, Ny, Nz)`` (**static** -- a compile-time
        constant; changing it triggers retracing).
    symbols : Tuple[str, ...]
        Element symbols for each species (e.g. ``("Bi", "Se")``).
        **static** -- compile-time constants; changing them triggers retracing.

    Notes
    -----
    Equinox derives the PyTree structure from the annotated fields.
    JAX stores the Python tuples ``grid_shape`` and ``symbols`` as PyTree
    auxiliary data. JAX treats
    auxiliary data as compile-time constants: changing them triggers
    recompilation of any ``jit``-compiled function.

    See Also
    --------
    SOCVolumetricData : Variant for spin-orbit coupling with vector
        magnetization.
    make_volumetric_data : Factory function with validation and
        float64 casting.
    """

    lattice: Float64[Array, "3 3"]
    coords: Float64[Array, "N 3"]
    charge: Float64[Array, "Nx Ny Nz"]
    magnetization: Optional[Float64[Array, "Nx Ny Nz"]]
    atom_counts: Int32[Array, " S"]
    grid_shape: Tuple[int, int, int] = eqx.field(static=True)
    symbols: Tuple[str, ...] = eqx.field(static=True)


@jaxtyped(typechecker=beartype)
def make_volumetric_data(  # noqa: DOC503
    lattice: Float64[Array, "3 3"],
    coords: Float64[Array, "N 3"],
    charge: Float64[Array, "Cx Cy Cz"],
    magnetization: Optional[Float64[Array, "Mx My Mz"]] = None,
    grid_shape: Tuple[int, int, int] = (1, 1, 1),
    symbols: Tuple[str, ...] = (),
    atom_counts: Optional[Int32[Array, " S"]] = None,
) -> VolumetricData:
    """Create a validated ``VolumetricData`` instance.

    The factory validates and normalizes CHGCAR volumetric
    data before it constructs a ``VolumetricData`` PyTree. It casts numerical
    arrays to ``float64`` and atom counts to ``int32``. The factory casts
    ``magnetization`` only when it is present. Thus, ``None`` continues to
    identify non-spin-polarized calculations. The factory substitutes an empty
    int32 array when ``atom_counts`` is ``None``.

    ``@jaxtyped(typechecker=beartype)`` checks the shape and dtype constraints
    at call time.

    :see: :class:`~.test_volumetric.TestMakeVolumetricData`

    Implementation Logic
    --------------------
    1. **Prepare the normalized values**::

           lattice_arr = jnp.asarray(lattice, dtype=jnp.float64)

       This expression gives the later validation steps a stable shape and
       dtype.

    2. **Apply static validation**::

           charge_arr.shape != grid_shape

       This predicate rejects invalid structure before JAX traces the
       numerical checks.

    3. **Apply traced validation**::

           ~jnp.all(jnp.isfinite(lattice_arr))

       This predicate remains active during eager and compiled execution.

    4. **Return the named instance**::

           return vol

       The explicit name keeps the implementation and the Returns section
       synchronized.

    Parameters
    ----------
    lattice : Float64[Array, "3 3"]
        Real-space lattice vectors as rows, in Angstroms.
    coords : Float64[Array, "N 3"]
        Fractional atomic coordinates.
    charge : Float64[Array, "Cx Cy Cz"]
        Charge density on 3-D grid (electrons per unit cell volume).
    magnetization : Optional[Float64[Array, "Mx My Mz"]], optional
        Magnetization density (spin-up minus spin-down).
        Default is ``None`` (non-spin-polarized).
    grid_shape : Tuple[int, int, int], optional
        Grid dimensions ``(Nx, Ny, Nz)`` (**static** -- a compile-time
        constant; changing it triggers retracing). Default is ``(1, 1, 1)``.
    symbols : Tuple[str, ...], optional
        Element symbols per species (**static** -- compile-time constants;
        changing them triggers retracing). Default is empty tuple.
    atom_counts : Optional[Int32[Array, " S"]], optional
        Number of atoms per species. Default is ``None`` (replaced
        by an empty int32 array).

    Returns
    -------
    vol : VolumetricData
        Validated volumetric data with ``float64``/``int32`` arrays.

    Raises
    ------
    ValueError
        If ``grid_shape`` does not match ``charge`` or the optional
        ``magnetization`` grid.
    EquinoxRuntimeError
        If the lattice or any volumetric grid contains a non-finite
        value.

    Notes
    -----
    Static validation raises ``ValueError`` before traced construction when
    ``grid_shape`` differs from the charge or magnetization shape. Traced
    validation uses ``eqx.error_if`` and raises ``EquinoxRuntimeError`` when
    the lattice, charge, or present magnetization contains a non-finite value.

    See Also
    --------
    VolumetricData : The PyTree class constructed by this factory.
    make_soc_volumetric_data : Factory for the SOC variant with
        vector magnetization.
    """
    lattice_arr: Float64[Array, "3 3"] = jnp.asarray(
        lattice, dtype=jnp.float64
    )
    coords_arr: Float64[Array, "N 3"] = jnp.asarray(coords, dtype=jnp.float64)
    charge_arr: Float64[Array, "Nx Ny Nz"] = jnp.asarray(
        charge, dtype=jnp.float64
    )
    mag_arr: Optional[Float64[Array, "Nx Ny Nz"]] = None
    if magnetization is not None:
        mag_arr = jnp.asarray(magnetization, dtype=jnp.float64)
    if atom_counts is None:
        counts_arr: Int32[Array, " S"] = jnp.zeros(0, dtype=jnp.int32)
    else:
        counts_arr = jnp.asarray(atom_counts, dtype=jnp.int32)
    if charge_arr.shape != grid_shape:
        msg: str = "grid_shape must match charge shape"
        raise ValueError(msg)
    if mag_arr is not None and mag_arr.shape != grid_shape:
        msg: str = "grid_shape must match magnetization shape"
        raise ValueError(msg)

    def validate_and_create() -> VolumetricData:
        nonlocal charge_arr, lattice_arr, mag_arr
        lattice_arr = eqx.error_if(
            lattice_arr,
            ~(jnp.all(jnp.isfinite(lattice_arr))),
            "make_volumetric_data: lattice finite",
        )
        charge_arr = eqx.error_if(
            charge_arr,
            ~(jnp.all(jnp.isfinite(charge_arr))),
            "make_volumetric_data: charge finite",
        )
        if mag_arr is not None:
            mag_arr = eqx.error_if(
                mag_arr,
                ~(jnp.all(jnp.isfinite(mag_arr))),
                "make_volumetric_data: magnetization finite",
            )
        validated_volume: VolumetricData = VolumetricData(
            lattice=lattice_arr,
            coords=coords_arr,
            charge=charge_arr,
            magnetization=mag_arr,
            grid_shape=grid_shape,
            symbols=symbols,
            atom_counts=counts_arr,
        )
        return validated_volume

    vol: VolumetricData = validate_and_create()
    return vol


class SOCVolumetricData(eqx.Module):
    """Store SOC CHGCAR volumetric-grid data in a JAX PyTree.

    This type is a :class:`VolumetricData` variant for spin-orbit coupling
    calculations. VASP writes four CHGCAR grid blocks: total charge, mx, my,
    and mz magnetization components. The
    ``magnetization`` contains the mz component for compatibility with ISPIN=2
    consumers. ``magnetization_vector`` contains the complete 3-component
    vector ``(mx, my, mz)`` at each grid point.

    This class is an immutable :class:`equinox.Module` PyTree. JAX stores
    numerical fields as children that support tracing. It stores
    ``grid_shape`` and ``symbols`` as auxiliary data.


    :see: :class:`~.test_volumetric.TestSOCVolumetricData`

    Attributes
    ----------
    lattice : Float64[Array, "3 3"]
        Real-space lattice vectors as rows, in Angstroms.
        JAX-traced (differentiable).
    coords : Float64[Array, "N 3"]
        Fractional atomic coordinates for all N atoms.
        JAX-traced (differentiable).
    charge : Float64[Array, "Nx Ny Nz"]
        Total charge density on the 3-D grid (electrons per unit
        cell volume). JAX-traced (differentiable).
    magnetization : Float64[Array, "Nx Ny Nz"]
        Scalar magnetization density, specifically the mz component.
        Provided for backward compatibility with code that expects
        ISPIN=2-style scalar magnetization. JAX-traced
        (differentiable).
    magnetization_vector : Float64[Array, "Nx Ny Nz 3"]
        Full vector magnetization ``(mx, my, mz)`` at each grid
        point. The last axis indexes the three Cartesian components.
        JAX-traced (differentiable).
    atom_counts : Int32[Array, " S"]
        Number of atoms per species. JAX-traced (int32).
    grid_shape : Tuple[int, int, int]
        Grid dimensions ``(Nx, Ny, Nz)`` (**static** -- a compile-time
        constant; changing it triggers retracing).
    symbols : Tuple[str, ...]
        Element symbols per species (**static** -- compile-time constants;
        changing them triggers retracing).

    Notes
    -----
    Equinox derives the PyTree structure from the annotated fields.
    JAX stores six numerical fields as children. It stores ``grid_shape`` and
    ``symbols`` as auxiliary data. Unlike :class:`VolumetricData`, both
    ``magnetization`` and ``magnetization_vector`` are mandatory
    (non-optional) because SOC calculations always produce all four
    grid blocks.

    See Also
    --------
    VolumetricData : Non-SOC variant with optional scalar
        magnetization.
    make_soc_volumetric_data : Factory function with validation and
        float64 casting.
    """

    lattice: Float64[Array, "3 3"]
    coords: Float64[Array, "N 3"]
    charge: Float64[Array, "Nx Ny Nz"]
    magnetization: Float64[Array, "Nx Ny Nz"]
    magnetization_vector: Float64[Array, "Nx Ny Nz 3"]
    atom_counts: Int32[Array, " S"]
    grid_shape: Tuple[int, int, int] = eqx.field(static=True)
    symbols: Tuple[str, ...] = eqx.field(static=True)


@jaxtyped(typechecker=beartype)
def make_soc_volumetric_data(  # noqa: DOC503
    lattice: Float64[Array, "3 3"],
    coords: Float64[Array, "N 3"],
    charge: Float64[Array, "Cx Cy Cz"],
    magnetization: Float64[Array, "Mx My Mz"],
    magnetization_vector: Float64[Array, "Vx Vy Vz 3"],
    grid_shape: Tuple[int, int, int] = (1, 1, 1),
    symbols: Tuple[str, ...] = (),
    atom_counts: Optional[Int32[Array, " S"]] = None,
) -> SOCVolumetricData:
    """Create a validated ``SOCVolumetricData`` instance.

    The factory validates and normalizes SOC CHGCAR
    volumetric data before constructing a ``SOCVolumetricData``
    PyTree. It casts numerical arrays to ``float64`` and atom counts to
    ``int32``. Unlike :func:`make_volumetric_data`, both
    ``magnetization`` and ``magnetization_vector`` are mandatory
    because SOC calculations always produce all four grid blocks.

    ``@jaxtyped(typechecker=beartype)`` checks shape constraints at call time.
    The grid dimensions must agree across ``charge``, ``magnetization``, and
    ``magnetization_vector``.

    :see: :class:`~.test_volumetric.TestMakeSOCVolumetricData`

    Implementation Logic
    --------------------
    1. **Prepare the normalized values**::

           soc_lattice_arr = jnp.asarray(lattice, dtype=jnp.float64)

       This expression gives the later validation steps a stable shape and
       dtype.

    2. **Apply static validation**::

           soc_charge_arr.shape != grid_shape

       This predicate rejects invalid structure before JAX traces the
       numerical checks.

    3. **Apply traced validation**::

           ~jnp.all(jnp.isfinite(soc_lattice_arr))

       This predicate remains active during eager and compiled execution.

    4. **Return the named instance**::

           return vol

       The explicit name keeps the implementation and the Returns section
       synchronized.

    Parameters
    ----------
    lattice : Float64[Array, "3 3"]
        Real-space lattice vectors as rows, in Angstroms.
    coords : Float64[Array, "N 3"]
        Fractional atomic coordinates.
    charge : Float64[Array, "Cx Cy Cz"]
        Total charge density on 3-D grid (electrons per unit cell
        volume).
    magnetization : Float64[Array, "Mx My Mz"]
        Scalar magnetization density (mz component), for backward
        compatibility with ISPIN=2 consumers.
    magnetization_vector : Float64[Array, "Vx Vy Vz 3"]
        Full vector magnetization ``(mx, my, mz)`` at each grid
        point.
    grid_shape : Tuple[int, int, int], optional
        Grid dimensions ``(Nx, Ny, Nz)`` (**static** -- a compile-time
        constant; changing it triggers retracing). Default is ``(1, 1, 1)``.
    symbols : Tuple[str, ...], optional
        Element symbols per species (**static** -- compile-time constants;
        changing them triggers retracing). Default is empty tuple.
    atom_counts : Optional[Int32[Array, " S"]], optional
        Number of atoms per species. Default is ``None`` (replaced
        by an empty int32 array).

    Returns
    -------
    vol : SOCVolumetricData
        Validated SOC volumetric data with ``float64``/``int32``
        arrays.

    Raises
    ------
    ValueError
        If ``grid_shape`` does not match the charge, scalar
        magnetization, or vector-magnetization grids.
    EquinoxRuntimeError
        If the lattice or any volumetric grid contains a non-finite
        value.

    Notes
    -----
    Static validation raises ``ValueError`` before traced construction when
    ``grid_shape`` differs from a scalar or vector grid shape. Traced
    validation uses ``eqx.error_if`` and raises ``EquinoxRuntimeError`` when
    the lattice or any charge or magnetization grid is non-finite.

    See Also
    --------
    SOCVolumetricData : The PyTree class constructed by this factory.
    make_volumetric_data : Factory for the non-SOC variant.
    """
    if atom_counts is None:
        counts_arr: Int32[Array, " S"] = jnp.zeros(0, dtype=jnp.int32)
    else:
        counts_arr = jnp.asarray(atom_counts, dtype=jnp.int32)
    soc_lattice_arr: Float64[Array, "3 3"] = jnp.asarray(
        lattice, dtype=jnp.float64
    )
    soc_charge_arr: Float64[Array, "Nx Ny Nz"] = jnp.asarray(
        charge, dtype=jnp.float64
    )
    soc_mag_arr: Float64[Array, "Nx Ny Nz"] = jnp.asarray(
        magnetization, dtype=jnp.float64
    )
    soc_mag_vector_arr: Float64[Array, "Nx Ny Nz 3"] = jnp.asarray(
        magnetization_vector, dtype=jnp.float64
    )
    if soc_charge_arr.shape != grid_shape:
        msg: str = "grid_shape must match charge shape"
        raise ValueError(msg)
    if soc_mag_arr.shape != grid_shape:
        msg: str = "grid_shape must match magnetization shape"
        raise ValueError(msg)
    if soc_mag_vector_arr.shape[:3] != grid_shape:
        msg: str = "grid_shape must match magnetization_vector spatial shape"
        raise ValueError(msg)

    def validate_and_create() -> SOCVolumetricData:
        nonlocal soc_charge_arr, soc_lattice_arr, soc_mag_arr
        nonlocal soc_mag_vector_arr
        soc_lattice_arr = eqx.error_if(
            soc_lattice_arr,
            ~(jnp.all(jnp.isfinite(soc_lattice_arr))),
            "make_soc_volumetric_data: lattice finite",
        )
        soc_charge_arr = eqx.error_if(
            soc_charge_arr,
            ~(jnp.all(jnp.isfinite(soc_charge_arr))),
            "make_soc_volumetric_data: charge finite",
        )
        soc_mag_arr = eqx.error_if(
            soc_mag_arr,
            ~(jnp.all(jnp.isfinite(soc_mag_arr))),
            "make_soc_volumetric_data: magnetization finite",
        )
        soc_mag_vector_arr = eqx.error_if(
            soc_mag_vector_arr,
            ~(jnp.all(jnp.isfinite(soc_mag_vector_arr))),
            "make_soc_volumetric_data: magnetization vector finite",
        )
        validated_volume: SOCVolumetricData = SOCVolumetricData(
            lattice=soc_lattice_arr,
            coords=jnp.asarray(coords, dtype=jnp.float64),
            charge=soc_charge_arr,
            magnetization=soc_mag_arr,
            magnetization_vector=soc_mag_vector_arr,
            grid_shape=grid_shape,
            symbols=symbols,
            atom_counts=counts_arr,
        )
        return validated_volume

    vol: SOCVolumetricData = validate_and_create()
    return vol


__all__: list[str] = [
    "SOCVolumetricData",
    "VolumetricData",
    "make_soc_volumetric_data",
    "make_volumetric_data",
]
