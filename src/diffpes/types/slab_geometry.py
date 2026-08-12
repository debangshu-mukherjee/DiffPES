"""Define exact surface-cell and slab geometry metadata.

Extended Summary
----------------
This module stores validated surface frames and slab-construction
choices with exact integer cell coefficients.

Routine Listings
----------------
:class:`SlabSpec`
    Store static slab construction choices and provenance.
:class:`SurfaceCell`
    Store a validated Cartesian surface-cell frame.
:func:`make_slab_spec`
    Create a validated slab-construction sidecar.
:func:`make_surface_cell`
    Create a validated Cartesian surface-cell carrier.
"""

import math

import equinox as eqx
import jax.numpy as jnp
from beartype import beartype
from beartype.typing import Tuple
from jaxtyping import Array, Float64, jaxtyped

from diffpes.constants import (
    CARTESIAN_COMPONENTS,
    ROTATION_ORTHOGONALITY_TOLERANCE,
    SURFACE_VECTOR_COUNT,
)

from .aliases import ScalarNumeric
from .geometry import CrystalGeometry


def _validate_integer_triple(
    values: Tuple[int, int, int],
    name: str,
) -> None:
    """PRIVATE: Validate one exact integer coefficient triple.

    Parameters
    ----------
    values : Tuple[int, int, int]
        Candidate integer coefficient triple.
    name : str
        Field name used in the static error message.

    Raises
    ------
    ValueError
        If ``values`` is not a tuple of exactly three Python integers.
        This is the static construction-time contract.

    Notes
    -----
    Use exact ``type`` comparisons to reject bools and NumPy integers.
    The triple stays exact under integer arithmetic.
    """
    if (
        type(values) is not tuple
        or len(values) != CARTESIAN_COMPONENTS
        or any(type(value) is not int for value in values)
    ):
        message: str = f"{name} must be a tuple of three integers"
        raise ValueError(message)


def _integer_dot(
    left: Tuple[int, int, int],
    right: Tuple[int, int, int],
) -> int:
    """PRIVATE: Return an exact dot product between integer triples.

    Parameters
    ----------
    left : Tuple[int, int, int]
        First integer triple.
    right : Tuple[int, int, int]
        Second integer triple.

    Returns
    -------
    result : int
        Exact integer dot product of the two triples.

    Notes
    -----
    Sum the three componentwise products in Python integer arithmetic,
    which is exact at any magnitude.
    """
    result: int = sum(
        left[index] * right[index] for index in range(CARTESIAN_COMPONENTS)
    )
    return result


def _integer_determinant(
    rows: Tuple[
        Tuple[int, int, int],
        Tuple[int, int, int],
        Tuple[int, int, int],
    ],
) -> int:
    """PRIVATE: Return the exact determinant of three integer row vectors.

    Parameters
    ----------
    rows : Tuple[Tuple[int, int, int], Tuple[int, int, int], \
Tuple[int, int, int]]
        Three integer row vectors of the coefficient matrix.

    Returns
    -------
    determinant : int
        Exact integer determinant of the 3 x 3 coefficient matrix.

    Notes
    -----
    Expand the determinant along the first row in Python integer
    arithmetic, which is exact at any magnitude.
    """
    first: Tuple[int, int, int]
    second: Tuple[int, int, int]
    third: Tuple[int, int, int]
    first, second, third = rows
    determinant: int = (
        first[0] * (second[1] * third[2] - second[2] * third[1])
        - first[1] * (second[0] * third[2] - second[2] * third[0])
        + first[2] * (second[0] * third[1] - second[1] * third[0])
    )
    return determinant


def _validate_surface_cell_structure(
    in_plane_vectors: Float64[Array, "2 3"],
    stacking_vector: Float64[Array, " 3"],
    rotation: Float64[Array, "3 3"],
    interlayer_spacing_ang: Float64[Array, ""],
    miller: Tuple[int, int, int],
    in_plane_coeffs: Tuple[
        Tuple[int, int, int],
        Tuple[int, int, int],
    ],
    stacking_coeffs: Tuple[int, int, int],
) -> None:
    """PRIVATE: Validate surface-cell shapes and exact integer invariants.

    Implementation Logic
    --------------------
    Check the traced arrays only through ``ndim`` and ``shape``. Then
    check the integer metadata with ``_validate_integer_triple``,
    ``_integer_dot``, and ``_integer_determinant``, which stay exact at
    any magnitude.

    Parameters
    ----------
    in_plane_vectors : Float64[Array, "2 3"]
        Cartesian in-plane surface vectors in Angstrom, as rows.
    stacking_vector : Float64[Array, " 3"]
        Cartesian stacking vector in Angstrom.
    rotation : Float64[Array, "3 3"]
        Active Cartesian rotation from the bulk to the surface frame.
    interlayer_spacing_ang : Float64[Array, ""]
        Interlayer spacing in Angstrom.
    miller : Tuple[int, int, int]
        GCD-reduced Miller tuple.
    in_plane_coeffs : Tuple[Tuple[int, int, int], Tuple[int, int, int]]
        Exact bulk-lattice coefficients of the in-plane vectors.
    stacking_coeffs : Tuple[int, int, int]
        Exact bulk-lattice coefficients of the stacking vector.

    Raises
    ------
    ValueError
        If a numerical field has the wrong shape. If the coefficient
        metadata lacks exact integer triples. If an in-plane row lies
        outside the Miller plane. If the stacking dot product differs
        from one. If the three coefficient rows are linearly
        dependent. This is the static construction-time contract.
    """
    if (
        in_plane_vectors.ndim != SURFACE_VECTOR_COUNT
        or in_plane_vectors.shape
        != (SURFACE_VECTOR_COUNT, CARTESIAN_COMPONENTS)
    ):
        message: str = "in_plane_vectors must have shape (2, 3)"
        raise ValueError(message)
    if stacking_vector.ndim != 1 or stacking_vector.shape != (3,):
        message = "stacking_vector must have shape (3,)"
        raise ValueError(message)
    if rotation.ndim != SURFACE_VECTOR_COUNT or rotation.shape != (
        CARTESIAN_COMPONENTS,
        CARTESIAN_COMPONENTS,
    ):
        message = "rotation must have shape (3, 3)"
        raise ValueError(message)
    if interlayer_spacing_ang.ndim != 0:
        message = "interlayer_spacing_ang must be scalar"
        raise ValueError(message)

    _validate_integer_triple(miller, "miller")
    if (
        type(in_plane_coeffs) is not tuple
        or len(in_plane_coeffs) != SURFACE_VECTOR_COUNT
    ):
        message = "in_plane_coeffs must contain two integer triples"
        raise ValueError(message)
    _validate_integer_triple(in_plane_coeffs[0], "in_plane_coeffs[0]")
    _validate_integer_triple(in_plane_coeffs[1], "in_plane_coeffs[1]")
    _validate_integer_triple(stacking_coeffs, "stacking_coeffs")

    if (
        _integer_dot(miller, in_plane_coeffs[0]) != 0
        or _integer_dot(
            miller,
            in_plane_coeffs[1],
        )
        != 0
    ):
        message = "in_plane_coeffs must lie in the Miller plane"
        raise ValueError(message)
    if _integer_dot(miller, stacking_coeffs) != 1:
        message = (
            "the gcd-reduced miller tuple dotted with stacking_coeffs "
            "must equal one"
        )
        raise ValueError(message)
    coefficient_rows: Tuple[
        Tuple[int, int, int],
        Tuple[int, int, int],
        Tuple[int, int, int],
    ] = (
        in_plane_coeffs[0],
        in_plane_coeffs[1],
        stacking_coeffs,
    )
    if _integer_determinant(coefficient_rows) == 0:
        message = (
            "surface-cell coefficient tuples must be linearly independent"
        )
        raise ValueError(message)


def _validate_slab_spec_structure(
    surface_cell: "SurfaceCell",
    thickness_ang: float,
    vacuum_ang: float,
    fine: Tuple[float, float],
    termination: Tuple[str, str],
    n_layers: int,
    bulk_atom_of_slab_atom: Tuple[int, ...],
    layer_of_slab_atom: Tuple[int, ...],
) -> None:
    """PRIVATE: Validate static slab provenance and selection metadata.

    Implementation Logic
    --------------------
    Use exact ``type`` comparisons on the plain Python metadata and
    ``math.isfinite`` on the float choices. The two provenance tuples
    must agree elementwise on length so each slab atom keeps one bulk
    atom and one layer.

    Parameters
    ----------
    surface_cell : "SurfaceCell"
        Validated surface-frame carrier.
    thickness_ang : float
        Requested slab thickness in Angstrom.
    vacuum_ang : float
        Vacuum padding in Angstrom.
    fine : Tuple[float, float]
        Top and bottom cut shifts in Angstrom.
    termination : Tuple[str, str]
        Top and bottom species labels.
    n_layers : int
        Number of slab layers.
    bulk_atom_of_slab_atom : Tuple[int, ...]
        Bulk-atom provenance index for each slab atom.
    layer_of_slab_atom : Tuple[int, ...]
        Layer index for each slab atom.

    Raises
    ------
    ValueError
        If ``surface_cell`` has the wrong type. If a length is not a
        finite nonnegative float. If ``fine`` or ``termination`` has
        invalid pair metadata. If ``n_layers`` is not positive. If the
        provenance maps have invalid lengths, integers, or layer
        bounds. This is the static construction-time contract.
    """
    if not isinstance(surface_cell, SurfaceCell):
        message: str = "surface_cell must be a SurfaceCell"
        raise ValueError(message)
    if (
        type(thickness_ang) is not float
        or type(vacuum_ang) is not float
        or not math.isfinite(thickness_ang)
        or not math.isfinite(vacuum_ang)
    ):
        message = "thickness_ang and vacuum_ang must be finite floats"
        raise ValueError(message)
    if thickness_ang < 0.0 or vacuum_ang < 0.0:
        message = "thickness_ang and vacuum_ang must both be nonnegative"
        raise ValueError(message)
    if (
        type(fine) is not tuple
        or len(fine) != SURFACE_VECTOR_COUNT
        or any(type(value) is not float for value in fine)
        or not all(math.isfinite(value) for value in fine)
    ):
        message = "fine must contain two finite floats"
        raise ValueError(message)
    if (
        type(termination) is not tuple
        or len(termination) != SURFACE_VECTOR_COUNT
        or any(type(species) is not str for species in termination)
    ):
        message = "termination must contain two species labels"
        raise ValueError(message)
    if type(n_layers) is not int or n_layers <= 0:
        message = "n_layers must be a positive integer"
        raise ValueError(message)
    if any(
        type(values) is not tuple
        for values in (bulk_atom_of_slab_atom, layer_of_slab_atom)
    ):
        message = "slab provenance maps must be tuples"
        raise ValueError(message)
    if len(bulk_atom_of_slab_atom) != len(layer_of_slab_atom):
        message = "slab provenance maps must have the same length"
        raise ValueError(message)
    if any(
        type(index) is not int or index < 0 for index in bulk_atom_of_slab_atom
    ):
        message = "bulk_atom_of_slab_atom must contain nonnegative integers"
        raise ValueError(message)
    if any(
        type(layer) is not int or layer < 0 or layer >= n_layers
        for layer in layer_of_slab_atom
    ):
        message = "layer_of_slab_atom entries must lie in [0, n_layers)"
        raise ValueError(message)


class SurfaceCell(eqx.Module):
    """Store a validated Cartesian surface-cell frame.

    Keep traced Cartesian geometry together with exact Miller-frame
    coefficients selected by the host topology stage.

    :see: :class:`~.test_slab_geometry.TestSurfaceCell`

    Attributes
    ----------
    in_plane_vectors : Float64[Array, "2 3"]
        Cartesian in-plane vectors in Angstrom, as rows.
    stacking_vector : Float64[Array, "3"]
        Cartesian stacking vector in Angstrom.
    rotation : Float64[Array, "3 3"]
        Active Cartesian rotation from bulk to surface frame.
    interlayer_spacing_ang : Float64[Array, ""]
        Positive interlayer spacing in Angstrom.
    miller : Tuple[int, int, int]
        GCD-reduced Miller tuple (**static** -- changing it triggers
        retracing).
    in_plane_coeffs : Tuple[Tuple[int, int, int], Tuple[int, int, int]]
        Exact bulk-lattice coefficients of the in-plane vectors (**static**).
    stacking_coeffs : Tuple[int, int, int]
        Exact bulk-lattice coefficients of the stacking vector (**static**).

    Notes
    -----
    The integer coefficient rows are linearly independent, the in-plane rows
    are orthogonal to ``miller``, and ``miller · stacking_coeffs == 1``.

    See Also
    --------
    make_surface_cell : Validated factory for this type.
    """

    in_plane_vectors: Float64[Array, "2 3"]
    stacking_vector: Float64[Array, " 3"]
    rotation: Float64[Array, "3 3"]
    interlayer_spacing_ang: Float64[Array, ""]
    miller: Tuple[int, int, int] = eqx.field(static=True)
    in_plane_coeffs: Tuple[
        Tuple[int, int, int],
        Tuple[int, int, int],
    ] = eqx.field(static=True)
    stacking_coeffs: Tuple[int, int, int] = eqx.field(static=True)

    def __check_init__(self) -> None:
        """Validate exact surface-cell structure on direct construction."""
        _validate_surface_cell_structure(
            self.in_plane_vectors,
            self.stacking_vector,
            self.rotation,
            self.interlayer_spacing_ang,
            self.miller,
            self.in_plane_coeffs,
            self.stacking_coeffs,
        )


class SlabSpec(eqx.Module):
    """Store static slab construction choices and provenance.

    Carry the traced surface frame alongside immutable cut, layer, and
    bulk-to-slab provenance metadata.

    :see: :class:`~.test_slab_geometry.TestSlabSpec`

    Attributes
    ----------
    surface_cell : SurfaceCell
        Differentiable surface-frame carrier.
    thickness_ang : float
        Requested slab thickness in Angstrom (**static**).
    vacuum_ang : float
        Vacuum padding in Angstrom (**static**).
    fine : Tuple[float, float]
        Top and bottom cut shifts in Angstrom (**static**).
    termination : Tuple[str, str]
        Top and bottom species labels (**static**).
    n_layers : int
        Number of slab layers (**static**).
    bulk_atom_of_slab_atom : Tuple[int, ...]
        Bulk-atom provenance for each slab atom (**static**).
    layer_of_slab_atom : Tuple[int, ...]
        Layer index for each slab atom (**static**).

    See Also
    --------
    make_slab_spec : Validated factory for this type.
    """

    surface_cell: SurfaceCell
    thickness_ang: float = eqx.field(static=True)
    vacuum_ang: float = eqx.field(static=True)
    fine: Tuple[float, float] = eqx.field(static=True)
    termination: Tuple[str, str] = eqx.field(static=True)
    n_layers: int = eqx.field(static=True)
    bulk_atom_of_slab_atom: Tuple[int, ...] = eqx.field(static=True)
    layer_of_slab_atom: Tuple[int, ...] = eqx.field(static=True)

    def __check_init__(self) -> None:
        """Validate slab metadata invariants on direct construction."""
        _validate_slab_spec_structure(
            self.surface_cell,
            self.thickness_ang,
            self.vacuum_ang,
            self.fine,
            self.termination,
            self.n_layers,
            self.bulk_atom_of_slab_atom,
            self.layer_of_slab_atom,
        )


@jaxtyped(typechecker=beartype)
def make_surface_cell(  # noqa: DOC502, DOC503
    in_plane_vectors: Float64[Array, "2 3"],
    stacking_vector: Float64[Array, " 3"],
    rotation: Float64[Array, "3 3"],
    interlayer_spacing_ang: ScalarNumeric,
    miller: Tuple[int, int, int],
    in_plane_coeffs: Tuple[
        Tuple[int, int, int],
        Tuple[int, int, int],
    ],
    stacking_coeffs: Tuple[int, int, int],
) -> SurfaceCell:
    """Create a validated Cartesian surface-cell carrier.

    Convert continuous inputs to JAX arrays, validate the exact integer frame,
    and enforce the finite, orthogonal surface-frame contract.

    :see: :class:`~.test_slab_geometry.TestMakeSurfaceCell`

    Parameters
    ----------
    in_plane_vectors : Float64[Array, "2 3"]
        Cartesian in-plane vectors in Angstrom, as rows.
    stacking_vector : Float64[Array, "3"]
        Cartesian stacking vector in Angstrom.
    rotation : Float64[Array, "3 3"]
        Active Cartesian rotation from bulk to surface frame.
    interlayer_spacing_ang : ScalarNumeric
        Positive interlayer spacing in Angstrom.
    miller : Tuple[int, int, int]
        GCD-reduced Miller tuple.
    in_plane_coeffs : Tuple[Tuple[int, int, int], Tuple[int, int, int]]
        Exact integer coefficients for the in-plane vectors.
    stacking_coeffs : Tuple[int, int, int]
        Exact integer coefficients for the stacking vector.

    Returns
    -------
    surface_cell : SurfaceCell
        Validated surface-cell carrier.

    Raises
    ------
    ValueError
        If array shapes or exact integer invariants are invalid.
    EquinoxRuntimeError
        If numerical leaves are non-finite, the spacing is not positive, or
        the rotation is not orthogonal within ``1e-10``.

    Notes
    -----
    Exact Miller coefficients remain static metadata while Cartesian vectors,
    rotation, and spacing remain differentiable leaves.
    """
    in_plane_array: Float64[Array, "2 3"] = jnp.asarray(
        in_plane_vectors,
        dtype=jnp.float64,
    )
    stacking_array: Float64[Array, " 3"] = jnp.asarray(
        stacking_vector,
        dtype=jnp.float64,
    )
    rotation_array: Float64[Array, "3 3"] = jnp.asarray(
        rotation,
        dtype=jnp.float64,
    )
    spacing_array: Float64[Array, ""] = jnp.asarray(
        interlayer_spacing_ang,
        dtype=jnp.float64,
    )
    _validate_surface_cell_structure(
        in_plane_array,
        stacking_array,
        rotation_array,
        spacing_array,
        miller,
        in_plane_coeffs,
        stacking_coeffs,
    )

    in_plane_array = eqx.error_if(
        in_plane_array,
        ~jnp.all(jnp.isfinite(in_plane_array)),
        "make_surface_cell: in-plane vectors must be finite",
    )
    stacking_array = eqx.error_if(
        stacking_array,
        ~jnp.all(jnp.isfinite(stacking_array)),
        "make_surface_cell: stacking vector must be finite",
    )
    rotation_array = eqx.error_if(
        rotation_array,
        ~jnp.all(jnp.isfinite(rotation_array)),
        "make_surface_cell: rotation must be finite",
    )
    orthogonality_error: Float64[Array, ""] = jnp.linalg.norm(
        rotation_array.T @ rotation_array
        - jnp.eye(CARTESIAN_COMPONENTS, dtype=jnp.float64)
    )
    rotation_array = eqx.error_if(
        rotation_array,
        orthogonality_error >= ROTATION_ORTHOGONALITY_TOLERANCE,
        "make_surface_cell: rotation must be orthogonal",
    )
    spacing_array = eqx.error_if(
        spacing_array,
        ~jnp.isfinite(spacing_array) | (spacing_array <= 0.0),
        "make_surface_cell: interlayer spacing must be finite and positive",
    )
    surface_cell: SurfaceCell = SurfaceCell(
        in_plane_vectors=in_plane_array,
        stacking_vector=stacking_array,
        rotation=rotation_array,
        interlayer_spacing_ang=spacing_array,
        miller=miller,
        in_plane_coeffs=in_plane_coeffs,
        stacking_coeffs=stacking_coeffs,
    )
    return surface_cell


@jaxtyped(typechecker=beartype)
def make_slab_spec(
    surface_cell: SurfaceCell,
    geometry: CrystalGeometry,
    thickness_ang: float,
    vacuum_ang: float,
    fine: Tuple[float, float],
    termination: Tuple[str, str],
    n_layers: int,
    bulk_atom_of_slab_atom: Tuple[int, ...],
    layer_of_slab_atom: Tuple[int, ...],
) -> SlabSpec:
    """Create a validated slab-construction sidecar.

    ``geometry`` supplies the bulk species and atom count used to validate
    termination labels and provenance. It is validation context and is not
    stored in the returned sidecar. A geometry with no species metadata may
    use only the internal natural-cut sentinel ``("X", "X")``; explicit
    species termination still requires declared species.

    :see: :class:`~.test_slab_geometry.TestMakeSlabSpec`

    Parameters
    ----------
    surface_cell : SurfaceCell
        Validated surface-cell frame.
    geometry : CrystalGeometry
        Bulk geometry used to validate species and atom provenance.
    thickness_ang : float
        Nonnegative requested minimum slab span in Angstrom.
    vacuum_ang : float
        Nonnegative vacuum padding in Angstrom.
    fine : Tuple[float, float]
        Finite top and bottom cut shifts in Angstrom.
    termination : Tuple[str, str]
        Top and bottom species labels.
    n_layers : int
        Positive number of slab layers.
    bulk_atom_of_slab_atom : Tuple[int, ...]
        Bulk-atom index for each slab atom.
    layer_of_slab_atom : Tuple[int, ...]
        Layer index for each slab atom.

    Returns
    -------
    slab_spec : SlabSpec
        Validated static slab provenance with a traced surface-cell child.

    Raises
    ------
    ValueError
        If slab choices, species, or provenance mappings are inconsistent.

    Notes
    -----
    The factory validates host-selected provenance and preserves the
    ``SurfaceCell`` as the only traced child of the static slab sidecar.
    """
    if not isinstance(geometry, CrystalGeometry):
        message: str = "geometry must be a CrystalGeometry"
        raise ValueError(message)
    _validate_slab_spec_structure(
        surface_cell,
        thickness_ang,
        vacuum_ang,
        fine,
        termination,
        n_layers,
        bulk_atom_of_slab_atom,
        layer_of_slab_atom,
    )
    unknown_species_termination: bool = (
        not geometry.species and termination == ("X", "X")
    )
    if not unknown_species_termination and any(
        species not in geometry.species for species in termination
    ):
        message = "termination species must occur in geometry.species"
        raise ValueError(message)
    n_bulk_atoms: int = geometry.positions.shape[0]
    if any(index >= n_bulk_atoms for index in bulk_atom_of_slab_atom):
        message = "bulk_atom_of_slab_atom entries must refer to geometry atoms"
        raise ValueError(message)
    slab_spec: SlabSpec = SlabSpec(
        surface_cell=surface_cell,
        thickness_ang=thickness_ang,
        vacuum_ang=vacuum_ang,
        fine=fine,
        termination=termination,
        n_layers=n_layers,
        bulk_atom_of_slab_atom=bulk_atom_of_slab_atom,
        layer_of_slab_atom=layer_of_slab_atom,
    )
    return slab_spec


__all__: list[str] = [
    "SlabSpec",
    "SurfaceCell",
    "make_slab_spec",
    "make_surface_cell",
]
