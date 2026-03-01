"""Volumetric data structures for VASP CHGCAR files.

Defines the :class:`VolumetricData` PyTree for storing charge density
and optional magnetization density grids parsed from VASP CHGCAR files.
"""

import jax.numpy as jnp
from beartype import beartype
from beartype.typing import NamedTuple, Optional, Tuple
from jax.tree_util import register_pytree_node_class
from jaxtyping import Array, Float, Int, jaxtyped


@register_pytree_node_class
class VolumetricData(NamedTuple):
    """PyTree for volumetric grid data from CHGCAR.

    Stores the charge density on a 3-D real-space grid together with
    the crystal lattice needed to interpret the grid coordinates.
    An optional magnetization density grid is included when the
    CHGCAR comes from a spin-polarized calculation.

    Attributes
    ----------
    lattice : Float[Array, "3 3"]
        Real-space lattice vectors as rows (Angstroms).
    coords : Float[Array, "N 3"]
        Fractional atomic coordinates.
    charge : Float[Array, "Nx Ny Nz"]
        Charge density on a 3-D grid.
    magnetization : Optional[Float[Array, "Nx Ny Nz"]]
        Magnetization density (spin-up minus spin-down), or None.
    grid_shape : tuple[int, int, int]
        Grid dimensions (Nx, Ny, Nz).
    symbols : tuple[str, ...]
        Element symbols per species.
    atom_counts : Int[Array, " S"]
        Atoms per species.
    """

    lattice: Float[Array, "3 3"]
    coords: Float[Array, "N 3"]
    charge: Float[Array, "Nx Ny Nz"]
    magnetization: Optional[Float[Array, "Nx Ny Nz"]]
    grid_shape: tuple[int, int, int]
    symbols: tuple[str, ...]
    atom_counts: Int[Array, " S"]

    def tree_flatten(
        self,
    ) -> Tuple[
        Tuple[
            Float[Array, "3 3"],
            Float[Array, "N 3"],
            Float[Array, "Nx Ny Nz"],
            Optional[Float[Array, "Nx Ny Nz"]],
            Int[Array, " S"],
        ],
        Tuple[tuple[int, int, int], tuple[str, ...]],
    ]:
        """Flatten into JAX leaf arrays and auxiliary data.

        Returns
        -------
        children : tuple of (jax.Array or None)
            Numeric fields.
        aux_data : tuple
            (grid_shape, symbols) static metadata.
        """
        return (
            (
                self.lattice,
                self.coords,
                self.charge,
                self.magnetization,
                self.atom_counts,
            ),
            (self.grid_shape, self.symbols),
        )

    @classmethod
    def tree_unflatten(
        cls,
        aux_data: Tuple[tuple[int, int, int], tuple[str, ...]],
        children: Tuple[
            Float[Array, "3 3"],
            Float[Array, "N 3"],
            Float[Array, "Nx Ny Nz"],
            Optional[Float[Array, "Nx Ny Nz"]],
            Int[Array, " S"],
        ],
    ) -> "VolumetricData":
        """Reconstruct from flattened components.

        Parameters
        ----------
        aux_data : tuple
            (grid_shape, symbols).
        children : tuple of (jax.Array or None)

        Returns
        -------
        VolumetricData
        """
        lattice, coords, charge, magnetization, atom_counts = children
        grid_shape, symbols = aux_data
        return cls(
            lattice=lattice,
            coords=coords,
            charge=charge,
            magnetization=magnetization,
            grid_shape=grid_shape,
            symbols=symbols,
            atom_counts=atom_counts,
        )


@jaxtyped(typechecker=beartype)
def make_volumetric_data(
    lattice: Float[Array, "3 3"],
    coords: Float[Array, "N 3"],
    charge: Float[Array, "Nx Ny Nz"],
    magnetization: Optional[Float[Array, "Nx Ny Nz"]] = None,
    grid_shape: tuple[int, int, int] = (1, 1, 1),
    symbols: tuple[str, ...] = (),
    atom_counts: Optional[Int[Array, " S"]] = None,
) -> VolumetricData:
    """Create a validated ``VolumetricData`` instance.

    Parameters
    ----------
    lattice : Float[Array, "3 3"]
        Real-space lattice vectors (rows, Angstroms).
    coords : Float[Array, "N 3"]
        Fractional atomic coordinates.
    charge : Float[Array, "Nx Ny Nz"]
        Charge density on 3-D grid.
    magnetization : Optional[Float[Array, "Nx Ny Nz"]], optional
        Magnetization density. Default is None.
    grid_shape : tuple[int, int, int], optional
        Grid dimensions. Default is (1,1,1).
    symbols : tuple[str, ...], optional
        Element symbols. Default is empty.
    atom_counts : Optional[Int[Array, " S"]], optional
        Atoms per species. Default is None.

    Returns
    -------
    vol : VolumetricData
    """
    lattice_arr = jnp.asarray(lattice, dtype=jnp.float64)
    coords_arr = jnp.asarray(coords, dtype=jnp.float64)
    charge_arr = jnp.asarray(charge, dtype=jnp.float64)
    mag_arr = None
    if magnetization is not None:
        mag_arr = jnp.asarray(magnetization, dtype=jnp.float64)
    if atom_counts is None:
        counts_arr = jnp.zeros(0, dtype=jnp.int32)
    else:
        counts_arr = jnp.asarray(atom_counts, dtype=jnp.int32)
    return VolumetricData(
        lattice=lattice_arr,
        coords=coords_arr,
        charge=charge_arr,
        magnetization=mag_arr,
        grid_shape=grid_shape,
        symbols=symbols,
        atom_counts=counts_arr,
    )


@register_pytree_node_class
class SOCVolumetricData(NamedTuple):
    """PyTree for volumetric data from SOC CHGCAR files.

    Variant of :class:`VolumetricData` for spin-orbit coupling
    calculations where VASP writes 4 grid blocks: total charge,
    mx, my, mz. The ``magnetization`` field holds mz (backward
    compatible with ISPIN=2 consumers) and
    ``magnetization_vector`` holds the full 3-component vector.

    Attributes
    ----------
    lattice : Float[Array, "3 3"]
        Real-space lattice vectors as rows (Angstroms).
    coords : Float[Array, "N 3"]
        Fractional atomic coordinates.
    charge : Float[Array, "Nx Ny Nz"]
        Charge density on a 3-D grid.
    magnetization : Float[Array, "Nx Ny Nz"]
        Scalar magnetization density (mz component).
    magnetization_vector : Float[Array, "Nx Ny Nz 3"]
        Vector magnetization (mx, my, mz).
    grid_shape : tuple[int, int, int]
        Grid dimensions (Nx, Ny, Nz).
    symbols : tuple[str, ...]
        Element symbols per species.
    atom_counts : Int[Array, " S"]
        Atoms per species.
    """

    lattice: Float[Array, "3 3"]
    coords: Float[Array, "N 3"]
    charge: Float[Array, "Nx Ny Nz"]
    magnetization: Float[Array, "Nx Ny Nz"]
    magnetization_vector: Float[Array, "Nx Ny Nz 3"]
    grid_shape: tuple[int, int, int]
    symbols: tuple[str, ...]
    atom_counts: Int[Array, " S"]

    def tree_flatten(
        self,
    ) -> Tuple[
        Tuple[
            Float[Array, "3 3"],
            Float[Array, "N 3"],
            Float[Array, "Nx Ny Nz"],
            Float[Array, "Nx Ny Nz"],
            Float[Array, "Nx Ny Nz 3"],
            Int[Array, " S"],
        ],
        Tuple[tuple[int, int, int], tuple[str, ...]],
    ]:
        """Flatten into JAX leaf arrays and auxiliary data."""
        return (
            (
                self.lattice,
                self.coords,
                self.charge,
                self.magnetization,
                self.magnetization_vector,
                self.atom_counts,
            ),
            (self.grid_shape, self.symbols),
        )

    @classmethod
    def tree_unflatten(
        cls,
        aux_data: Tuple[tuple[int, int, int], tuple[str, ...]],
        children: Tuple[
            Float[Array, "3 3"],
            Float[Array, "N 3"],
            Float[Array, "Nx Ny Nz"],
            Float[Array, "Nx Ny Nz"],
            Float[Array, "Nx Ny Nz 3"],
            Int[Array, " S"],
        ],
    ) -> "SOCVolumetricData":
        """Reconstruct from flattened components."""
        (
            lattice,
            coords,
            charge,
            magnetization,
            magnetization_vector,
            atom_counts,
        ) = children
        grid_shape, symbols = aux_data
        return cls(
            lattice=lattice,
            coords=coords,
            charge=charge,
            magnetization=magnetization,
            magnetization_vector=magnetization_vector,
            grid_shape=grid_shape,
            symbols=symbols,
            atom_counts=atom_counts,
        )


@jaxtyped(typechecker=beartype)
def make_soc_volumetric_data(
    lattice: Float[Array, "3 3"],
    coords: Float[Array, "N 3"],
    charge: Float[Array, "Nx Ny Nz"],
    magnetization: Float[Array, "Nx Ny Nz"],
    magnetization_vector: Float[Array, "Nx Ny Nz 3"],
    grid_shape: tuple[int, int, int] = (1, 1, 1),
    symbols: tuple[str, ...] = (),
    atom_counts: Optional[Int[Array, " S"]] = None,
) -> SOCVolumetricData:
    """Create a validated ``SOCVolumetricData`` instance.

    Parameters
    ----------
    lattice : Float[Array, "3 3"]
        Real-space lattice vectors (rows, Angstroms).
    coords : Float[Array, "N 3"]
        Fractional atomic coordinates.
    charge : Float[Array, "Nx Ny Nz"]
        Charge density on 3-D grid.
    magnetization : Float[Array, "Nx Ny Nz"]
        Scalar magnetization density (mz).
    magnetization_vector : Float[Array, "Nx Ny Nz 3"]
        Vector magnetization (mx, my, mz).
    grid_shape : tuple[int, int, int], optional
        Grid dimensions. Default is (1,1,1).
    symbols : tuple[str, ...], optional
        Element symbols. Default is empty.
    atom_counts : Optional[Int[Array, " S"]], optional
        Atoms per species. Default is None.

    Returns
    -------
    vol : SOCVolumetricData
    """
    if atom_counts is None:
        counts_arr = jnp.zeros(0, dtype=jnp.int32)
    else:
        counts_arr = jnp.asarray(atom_counts, dtype=jnp.int32)
    return SOCVolumetricData(
        lattice=jnp.asarray(lattice, dtype=jnp.float64),
        coords=jnp.asarray(coords, dtype=jnp.float64),
        charge=jnp.asarray(charge, dtype=jnp.float64),
        magnetization=jnp.asarray(magnetization, dtype=jnp.float64),
        magnetization_vector=jnp.asarray(
            magnetization_vector, dtype=jnp.float64
        ),
        grid_shape=grid_shape,
        symbols=symbols,
        atom_counts=counts_arr,
    )


__all__: list[str] = [
    "SOCVolumetricData",
    "VolumetricData",
    "make_soc_volumetric_data",
    "make_volumetric_data",
]
