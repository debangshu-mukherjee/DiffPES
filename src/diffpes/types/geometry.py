"""Define crystal-geometry data structures for VASP crystal structures.

Extended Summary
----------------
This module defines the :class:`CrystalGeometry` PyTree for crystal
structures parsed from VASP POSCAR files. It stores real-space lattice
vectors, reciprocal lattice vectors, atomic positions, and species.

Routine Listings
----------------
:class:`CrystalGeometry`
    Store VASP POSCAR crystal geometry in a JAX PyTree.
:func:`make_crystal_geometry`
    Create a validated CrystalGeometry instance.

Notes
-----
JAX stores the ``species`` field as auxiliary PyTree data because JAX cannot
trace Python strings. JAX stores all numerical fields as arrays.
"""

import equinox as eqx
import jax.numpy as jnp
from beartype import beartype
from beartype.typing import List, Tuple, Union
from jaxtyping import Array, Float64, jaxtyped

from .aliases import ScalarNumeric

_MAX_LATTICE_CONDITION_NUMBER: float = 1e12
_MIN_SCALED_SINGULAR_VALUE: float = 1e-12


class CrystalGeometry(eqx.Module):
    """Store VASP POSCAR crystal geometry in a JAX PyTree.

    This type contains the complete crystal structure from a VASP POSCAR file.
    It contains real-space lattice vectors, the reciprocal lattice,
    fractional positions, and per-atom species. Together, these fields
    describe the periodic crystal for an ARPES simulation.

    JAX stores the numerical fields as PyTree children. JAX tracing can access
    these fields. JAX stores the ``species`` tuple as auxiliary data because
    JAX cannot trace Python strings.


    :see: :class:`~.test_geometry.TestCrystalGeometry`

    Attributes
    ----------
    lattice : Float64[Array, "3 3"]
        Real-space lattice vectors as rows (angstroms).
    reciprocal : Float64[Array, "3 3"]
        Reciprocal lattice vectors as rows (1/angstroms).
    positions : Float64[Array, "N 3"]
        Fractional atomic positions.
    species : Tuple[str, ...]
        Per-atom species symbols (**static** -- compile-time constants;
        changing them triggers retracing). An empty tuple records unknown
        species.

    Notes
    -----
    The ``species`` field is a tuple of Python strings declared with
    ``eqx.field(static=True)`` rather than as a traced leaf. Changing it
    triggers recompilation of any ``jit``-compiled function that
    receives this PyTree.

    See Also
    --------
    make_crystal_geometry : Factory function with validation, float64
        casting, and automatic reciprocal lattice computation.
    """

    lattice: Float64[Array, "3 3"]
    reciprocal: Float64[Array, "3 3"]
    positions: Float64[Array, "N 3"]
    species: Tuple[str, ...] = eqx.field(static=True)


def _compute_reciprocal_lattice(
    lattice: Float64[Array, "3 3"],
) -> Float64[Array, "3 3"]:
    r"""PRIVATE: Compute reciprocal lattice vectors from real-space lattice.

    Derives the reciprocal lattice from the real-space lattice using
    the standard cross-product formula. The reciprocal lattice vectors
    ``b_i`` satisfy ``a_i . b_j = 2 pi delta_{ij}``.

    Implementation Logic
    --------------------
    Given real-space lattice vectors ``a1, a2, a3`` (rows of
    ``lattice``):

    1. **Extract rows**: ``a1 = lattice[0]``, ``a2 = lattice[1]``,
       ``a3 = lattice[2]``.
    2. **Compute unit-cell volume**:
       ``V = a1 . (a2 x a3)`` (scalar triple product).
    3. **Compute reciprocal vectors** via the cross-product formula:

       .. math::

           b_1 = 2\\pi \\, (a_2 \\times a_3) / V

           b_2 = 2\\pi \\, (a_3 \\times a_1) / V

           b_3 = 2\\pi \\, (a_1 \\times a_2) / V

    4. **Stack** ``[b1, b2, b3]`` into a (3, 3) array with
       reciprocal vectors as rows.

    Parameters
    ----------
    lattice : Float64[Array, "3 3"]
        Real-space lattice vectors as rows.

    Returns
    -------
    reciprocal : Float64[Array, "3 3"]
        Reciprocal lattice vectors as rows (units of 1/angstrom,
        with the 2 pi prefactor included).
    """
    a1: Float64[Array, " 3"] = lattice[0]
    a2: Float64[Array, " 3"] = lattice[1]
    a3: Float64[Array, " 3"] = lattice[2]
    volume: Float64[Array, " "] = jnp.dot(a1, jnp.cross(a2, a3))
    b1: Float64[Array, " 3"] = 2.0 * jnp.pi * jnp.cross(a2, a3) / volume
    b2: Float64[Array, " 3"] = 2.0 * jnp.pi * jnp.cross(a3, a1) / volume
    b3: Float64[Array, " 3"] = 2.0 * jnp.pi * jnp.cross(a1, a2) / volume
    reciprocal: Float64[Array, "3 3"] = jnp.stack([b1, b2, b3])
    return reciprocal


@jaxtyped(typechecker=beartype)
def make_crystal_geometry(  # noqa: DOC503
    lattice: Union[Float64[Array, "3 3"], "List[List[ScalarNumeric]]"],
    positions: Float64[Array, "N 3"],
    species: Tuple[str, ...],
) -> CrystalGeometry:
    """Create a validated CrystalGeometry instance.

    The factory validates the real-space lattice and fractional positions. It
    computes the reciprocal lattice and casts numerical arrays to float64.

    :see: :class:`~.test_geometry.TestMakeCrystalGeometry`

    Implementation Logic
    --------------------
    1. **Prepare the normalized values**::

           lattice_arr = jnp.asarray(lattice, dtype=jnp.float64)

       This expression gives the later validation steps a stable shape and
       dtype.

    2. **Apply static validation**::

           species and len(species) != positions_arr.shape[0]

       This predicate rejects invalid structure before JAX traces the
       numerical checks.

    3. **Apply traced validation**::

           ~jnp.all(jnp.isfinite(positions_arr))

       This predicate remains active during eager and compiled execution.

    4. **Return the named instance**::

           return geometry

       The explicit name keeps the implementation and the Returns section
       synchronized.

    Parameters
    ----------
    lattice : Union[Float64[Array, "3 3"], "List[List[ScalarNumeric]]"]
        Real-space lattice vectors as rows (angstroms).
    positions : Float64[Array, "N 3"]
        Fractional atomic positions.
    species : Tuple[str, ...]
        Per-atom species symbols (**static** -- compile-time constants;
        changing them triggers retracing). Use an empty tuple when the source
        does not identify species.

    Returns
    -------
    geometry : CrystalGeometry
        Validated crystal geometry instance with the reciprocal
        lattice pre-computed.

    Raises
    ------
    ValueError
        If a non-empty species tuple differs from the number of positions.
        Empty species remain valid for VASP 4 files without a species line.
    EquinoxRuntimeError
        If positions or lattice entries are non-finite. The function also
        rejects a left-handed lattice or a lattice that violates the numerical
        stability limits.

    Notes
    -----
    Static validation raises ``ValueError`` before traced construction when
    species and position counts disagree. Traced validation uses
    ``eqx.error_if`` and raises ``EquinoxRuntimeError`` for non-finite data or
    a lattice that fails the orientation and conditioning limits.

    See Also
    --------
    CrystalGeometry : The PyTree class constructed by this factory.
    _compute_reciprocal_lattice : Cross-product formula used to
        derive the reciprocal lattice.
    """
    lattice_arr: Float64[Array, "3 3"] = jnp.asarray(
        lattice, dtype=jnp.float64
    )
    positions_arr: Float64[Array, "N 3"] = jnp.asarray(
        positions,
        dtype=jnp.float64,
    )

    if species and len(species) != positions_arr.shape[0]:
        msg: str = "make_crystal_geometry: species must match positions"
        raise ValueError(msg)

    def validate_and_create() -> CrystalGeometry:
        nonlocal lattice_arr, positions_arr
        lattice_arr = eqx.error_if(
            lattice_arr,
            ~(jnp.all(jnp.isfinite(lattice_arr))),
            "make_crystal_geometry: lattice finite",
        )
        determinant: Float64[Array, " "] = jnp.linalg.det(lattice_arr)
        lattice_arr = eqx.error_if(
            lattice_arr,
            ~(determinant > 0.0),
            "make_crystal_geometry: lattice must be right-handed",
        )
        singular_values: Float64[Array, " 3"] = jnp.linalg.svdvals(lattice_arr)
        largest: Float64[Array, " "] = singular_values[0]
        smallest: Float64[Array, " "] = singular_values[-1]
        scaled_smallest: Float64[Array, " "] = smallest / largest
        lattice_arr = eqx.error_if(
            lattice_arr,
            scaled_smallest < _MIN_SCALED_SINGULAR_VALUE,
            "make_crystal_geometry: scaled singular value below limit",
        )
        condition_number: Float64[Array, " "] = largest / smallest
        lattice_arr = eqx.error_if(
            lattice_arr,
            condition_number > _MAX_LATTICE_CONDITION_NUMBER,
            "make_crystal_geometry: lattice condition number exceeds limit",
        )
        positions_arr = eqx.error_if(
            positions_arr,
            ~(jnp.all(jnp.isfinite(positions_arr))),
            "make_crystal_geometry: positions must be finite",
        )
        reciprocal: Float64[Array, "3 3"] = _compute_reciprocal_lattice(
            lattice_arr
        )
        validated_geometry: CrystalGeometry = CrystalGeometry(
            lattice=lattice_arr,
            reciprocal=reciprocal,
            positions=positions_arr,
            species=species,
        )
        return validated_geometry

    geometry: CrystalGeometry = validate_and_create()
    return geometry


__all__: list[str] = [
    "CrystalGeometry",
    "make_crystal_geometry",
]
