"""Crystal geometry data structure for VASP crystal structures.

Extended Summary
----------------
Defines the :class:`CrystalGeometry` PyTree for representing
crystal structures parsed from VASP POSCAR files. Includes real-space
lattice vectors, reciprocal lattice, atomic coordinates, element
symbols, and atom counts per species.

Routine Listings
----------------
:class:`CrystalGeometry`
    PyTree NamedTuple for crystal geometry data.
:func:`make_crystal_geometry`
    Factory function with validation and reciprocal lattice computation.

Notes
-----
The ``symbols`` field is stored as auxiliary data in the PyTree
since JAX cannot trace Python strings. All numeric fields are
stored as JAX arrays for compatibility with JAX transformations.
"""

import jax.numpy as jnp
from beartype import beartype
from beartype.typing import NamedTuple, Tuple, Union
from jax.tree_util import register_pytree_node_class
from jaxtyping import Array, Float, Int, jaxtyped

from .aliases import ScalarNumeric


@register_pytree_node_class
class CrystalGeometry(NamedTuple):
    """PyTree for crystal geometry from VASP POSCAR.

    Attributes
    ----------
    lattice : Float[Array, "3 3"]
        Real-space lattice vectors as rows (angstroms).
    reciprocal_lattice : Float[Array, "3 3"]
        Reciprocal lattice vectors as rows (1/angstroms).
    coords : Float[Array, "N 3"]
        Fractional atomic coordinates.
    symbols : tuple[str, ...]
        Element symbols for each species.
    atom_counts : Int[Array, " S"]
        Number of atoms per species.
    """

    lattice: Float[Array, "3 3"]
    reciprocal_lattice: Float[Array, "3 3"]
    coords: Float[Array, "N 3"]
    symbols: tuple[str, ...]
    atom_counts: Int[Array, " S"]

    def tree_flatten(
        self,
    ) -> Tuple[
        Tuple[
            Float[Array, "3 3"],
            Float[Array, "3 3"],
            Float[Array, "N 3"],
            Int[Array, " S"],
        ],
        tuple[str, ...],
    ]:
        """Flatten into JAX children and auxiliary data."""
        return (
            (
                self.lattice,
                self.reciprocal_lattice,
                self.coords,
                self.atom_counts,
            ),
            self.symbols,
        )

    @classmethod
    def tree_unflatten(
        cls,
        aux_data: tuple[str, ...],
        children: Tuple[
            Float[Array, "3 3"],
            Float[Array, "3 3"],
            Float[Array, "N 3"],
            Int[Array, " S"],
        ],
    ) -> "CrystalGeometry":
        """Reconstruct from flattened components."""
        lattice, reciprocal_lattice, coords, atom_counts = (
            children
        )
        return cls(
            lattice=lattice,
            reciprocal_lattice=reciprocal_lattice,
            coords=coords,
            symbols=aux_data,
            atom_counts=atom_counts,
        )


def _compute_reciprocal_lattice(
    lattice: Float[Array, "3 3"],
) -> Float[Array, "3 3"]:
    """Compute reciprocal lattice vectors from real-space lattice.

    Parameters
    ----------
    lattice : Float[Array, "3 3"]
        Real-space lattice vectors as rows.

    Returns
    -------
    reciprocal : Float[Array, "3 3"]
        Reciprocal lattice vectors as rows.
    """
    a1: Float[Array, " 3"] = lattice[0]
    a2: Float[Array, " 3"] = lattice[1]
    a3: Float[Array, " 3"] = lattice[2]
    volume: Float[Array, " "] = jnp.dot(a1, jnp.cross(a2, a3))
    b1: Float[Array, " 3"] = (
        2.0 * jnp.pi * jnp.cross(a2, a3) / volume
    )
    b2: Float[Array, " 3"] = (
        2.0 * jnp.pi * jnp.cross(a3, a1) / volume
    )
    b3: Float[Array, " 3"] = (
        2.0 * jnp.pi * jnp.cross(a1, a2) / volume
    )
    reciprocal: Float[Array, "3 3"] = jnp.stack([b1, b2, b3])
    return reciprocal


@jaxtyped(typechecker=beartype)
def make_crystal_geometry(
    lattice: Union[
        Float[Array, "3 3"], "list[list[ScalarNumeric]]"
    ],
    coords: Float[Array, "N 3"],
    symbols: tuple[str, ...],
    atom_counts: Union[Int[Array, " S"], "list[int]"],
) -> CrystalGeometry:
    """Create a validated CrystalGeometry instance.

    Parameters
    ----------
    lattice : Union[Float[Array, "3 3"], list]
        Real-space lattice vectors as rows (angstroms).
    coords : Float[Array, "N 3"]
        Fractional atomic coordinates.
    symbols : tuple[str, ...]
        Element symbols for each species.
    atom_counts : Union[Int[Array, " S"], list[int]]
        Number of atoms per species.

    Returns
    -------
    geometry : CrystalGeometry
        Validated crystal geometry instance.
    """
    lattice_arr: Float[Array, "3 3"] = jnp.asarray(
        lattice, dtype=jnp.float64
    )
    coords_arr: Float[Array, "N 3"] = jnp.asarray(
        coords, dtype=jnp.float64
    )
    counts_arr: Int[Array, " S"] = jnp.asarray(
        atom_counts, dtype=jnp.int32
    )
    reciprocal: Float[Array, "3 3"] = _compute_reciprocal_lattice(
        lattice_arr
    )
    geometry: CrystalGeometry = CrystalGeometry(
        lattice=lattice_arr,
        reciprocal_lattice=reciprocal,
        coords=coords_arr,
        symbols=symbols,
        atom_counts=counts_arr,
    )
    return geometry


__all__: list[str] = [
    "CrystalGeometry",
    "make_crystal_geometry",
]
