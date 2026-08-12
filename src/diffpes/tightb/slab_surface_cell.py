"""Construct exact Miller-index surface cells.

Extended Summary
----------------
This module owns the exact host integer choices.
It separates them from the differentiable Cartesian assembly.

Routine Listings
----------------
:func:`find_surface_cell`
    Build an exact primitive surface cell for one Miller plane.
"""

from __future__ import annotations

import math

import jax.numpy as jnp
import numpy as np
from beartype import beartype
from beartype.typing import Tuple
from jaxtyping import Array, Bool, Float64, Int64, jaxtyped
from numpy.typing import NDArray

from diffpes.maths import safe_norm
from diffpes.types import CrystalGeometry, SurfaceCell, make_surface_cell


def _extended_gcd(first: int, second: int) -> Tuple[int, int, int]:
    """PRIVATE: Return nonnegative gcd and signed Bezout coefficients.

    Parameters
    ----------
    first : int
        First integer, any sign.
    second : int
        Second integer, any sign.

    Returns
    -------
    result : Tuple[int, int, int]
        Values ``(g, x, y)`` with nonnegative ``g = gcd(first, second)``
        and exactly ``x * first + y * second == g``.

    Notes
    -----
    The iterative extended Euclidean algorithm runs on absolute values;
    a final sign flip moves the coefficients back to the signed inputs.
    All arithmetic is exact Python integer arithmetic, so the Bezout
    identity holds without rounding.
    """
    old_remainder: int = abs(first)
    remainder: int = abs(second)
    old_coefficient: int = 1
    coefficient: int = 0
    old_other: int = 0
    other: int = 1
    while remainder:
        quotient: int = old_remainder // remainder
        old_remainder, remainder = (
            remainder,
            old_remainder - quotient * remainder,
        )
        old_coefficient, coefficient = (
            coefficient,
            old_coefficient - quotient * coefficient,
        )
        old_other, other = other, old_other - quotient * other
    signed_first: int = old_coefficient if first >= 0 else -old_coefficient
    signed_second: int = old_other if second >= 0 else -old_other
    result: Tuple[int, int, int] = (
        old_remainder,
        signed_first,
        signed_second,
    )
    return result


def _determinant_3x3(matrix: Int64[NDArray, "3 3"]) -> int:
    """PRIVATE: Evaluate a three-dimensional integer determinant exactly.

    Parameters
    ----------
    matrix : Int64[NDArray, "3 3"]
        Integer matrix.

    Returns
    -------
    determinant : int
        Exact determinant value.

    Notes
    -----
    Cofactor expansion on Python integers avoids floating-point rounding
    and int64 overflow, so unimodularity checks on surface frames are
    exact.
    """
    a: int = int(matrix[0, 0])
    b: int = int(matrix[0, 1])
    c: int = int(matrix[0, 2])
    d: int = int(matrix[1, 0])
    e: int = int(matrix[1, 1])
    f: int = int(matrix[1, 2])
    g: int = int(matrix[2, 0])
    h: int = int(matrix[2, 1])
    i: int = int(matrix[2, 2])
    determinant: int = (
        a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)
    )
    return determinant


def _primitive_integer_frame(
    miller: Tuple[int, int, int],
) -> Tuple[Int64[NDArray, "2 3"], Int64[NDArray, " 3"]]:
    """PRIVATE: Construct an oriented integer kernel basis and unit advance.

    Parameters
    ----------
    miller : Tuple[int, int, int]
        Primitive (gcd-reduced) Miller indices.

    Returns
    -------
    result : Tuple[Int64[NDArray, "2 3"], Int64[NDArray, " 3"]]
        Two integer lattice vectors spanning the Miller plane and one
        stacking vector that advances exactly one plane.

    Raises
    ------
    ValueError
        If the indices are not gcd-reduced or the assembled frame is not
        unimodular.

    Notes
    -----
    Nested extended-gcd calls give exact Bezout coefficients: both
    kernel rows satisfy ``row . miller == 0`` and the stacking vector
    satisfies ``stacking . miller == 1``. The stacked frame must have
    determinant ``+-1``. A negative determinant flips the second kernel
    row so the frame is right handed.
    """
    first: int
    second: int
    third: int
    first, second, third = miller
    gcd_first_two: int
    bezout_first: int
    bezout_second: int
    gcd_first_two, bezout_first, bezout_second = _extended_gcd(
        first,
        second,
    )
    if gcd_first_two == 0:
        kernel: Int64[NDArray, "2 3"] = np.asarray(
            ((1, 0, 0), (0, 1, 0)),
            dtype=np.int64,
        )
        stacking: Int64[NDArray, " 3"] = np.asarray(
            (0, 0, third),
            dtype=np.int64,
        )
    else:
        total_gcd: int
        bezout_pair: int
        bezout_third: int
        total_gcd, bezout_pair, bezout_third = _extended_gcd(
            gcd_first_two,
            third,
        )
        if total_gcd != 1:
            message: str = "miller must be gcd-reduced"
            raise ValueError(message)
        kernel = np.asarray(
            (
                (second // gcd_first_two, -first // gcd_first_two, 0),
                (
                    -bezout_first * third,
                    -bezout_second * third,
                    gcd_first_two,
                ),
            ),
            dtype=np.int64,
        )
        stacking = np.asarray(
            (
                bezout_pair * bezout_first,
                bezout_pair * bezout_second,
                bezout_third,
            ),
            dtype=np.int64,
        )
    frame: Int64[NDArray, "3 3"] = np.vstack((kernel, stacking))
    determinant: int = _determinant_3x3(frame)
    if abs(determinant) != 1:
        message = "surface integer frame must be unimodular"
        raise ValueError(message)
    if determinant < 0:
        kernel[1] *= -1
    result: Tuple[Int64[NDArray, "2 3"], Int64[NDArray, " 3"]] = (
        kernel,
        stacking,
    )
    return result


def _gauss_reduce(
    kernel: Int64[NDArray, "2 3"],
    lattice: Float64[NDArray, "3 3"],
) -> Int64[NDArray, "2 3"]:
    """PRIVATE: Compute an exact reduced basis in the Cartesian metric.

    Parameters
    ----------
    kernel : Int64[NDArray, "2 3"]
        Integer basis of the surface plane.
    lattice : Float64[NDArray, "3 3"]
        Concrete bulk lattice rows in angstroms.

    Returns
    -------
    reduced : Int64[NDArray, "2 3"]
        Lagrange--Gauss-reduced integer basis of the same plane.

    Raises
    ------
    RuntimeError
        If the reduction does not converge within 256 sweeps.

    Notes
    -----
    Each sweep works on the Cartesian images of the integer rows. It
    swaps the rows when the second is shorter, negates one row to keep
    the orientation, and subtracts the nearest-integer projection
    multiple. The row operations stay integer, so the reduced rows span
    exactly the original plane lattice.
    """
    reduced: Int64[NDArray, "2 3"] = kernel.copy()
    for _ in range(256):
        vectors: Float64[NDArray, "2 3"] = reduced @ lattice
        first_norm: float = float(vectors[0] @ vectors[0])
        second_norm: float = float(vectors[1] @ vectors[1])
        if second_norm < first_norm:
            reduced[[0, 1]] = reduced[[1, 0]]
            reduced[1] *= -1
            continue
        nearest: int = int(
            np.rint(float(vectors[0] @ vectors[1]) / first_norm)
        )
        if nearest == 0:
            break
        reduced[1] -= nearest * reduced[0]
    else:
        message: str = "surface-basis metric reduction did not converge"
        raise RuntimeError(message)
    return reduced


def _closest_stacking_vector(
    stacking: Int64[NDArray, " 3"],
    kernel: Int64[NDArray, "2 3"],
    lattice: Float64[NDArray, "3 3"],
) -> Int64[NDArray, " 3"]:
    """PRIVATE: Compute the closest unit-advance vector to the plane normal.

    Parameters
    ----------
    stacking : Int64[NDArray, " 3"]
        Any integer vector advancing exactly one Miller plane.
    kernel : Int64[NDArray, "2 3"]
        Reduced integer basis of the plane.
    lattice : Float64[NDArray, "3 3"]
        Concrete bulk lattice rows in angstroms.

    Returns
    -------
    closest : Int64[NDArray, " 3"]
        Unit-advance vector of minimum Cartesian length, with ties
        broken by the smallest coefficient pair.

    Notes
    -----
    Adding in-plane kernel vectors never changes the plane advance, so
    the task is a two-dimensional closest-vector search. A Gram
    least-squares solve seeds the search. An enumeration box certifies
    completeness: its radius derives from the seed norm and the smallest
    in-plane singular value, so no closer pair exists outside it.
    Shortening the in-plane component aligns the stacking vector as
    closely as possible with the surface normal.
    """
    in_plane: Float64[NDArray, "2 3"] = kernel @ lattice
    seed: Float64[NDArray, " 3"] = stacking @ lattice
    gram: Float64[NDArray, "2 2"] = in_plane @ in_plane.T
    continuous: Float64[NDArray, " 2"] = np.linalg.solve(
        gram, -(in_plane @ seed)
    )
    rounded: Int64[NDArray, " 2"] = np.rint(continuous).astype(np.int64)

    def perpendicular_norm_squared(
        coefficients: Int64[NDArray, " 2"],
    ) -> float:
        candidate: Float64[NDArray, " 3"] = seed + coefficients @ in_plane
        norm_squared: float = float(candidate @ candidate)
        return norm_squared

    best_coefficients: Int64[NDArray, " 2"] = rounded
    best_norm: float = perpendicular_norm_squared(best_coefficients)
    smallest_singular: float = float(
        np.linalg.svd(in_plane, compute_uv=False)[-1]
    )
    radius: float = math.sqrt(best_norm) / smallest_singular + 1.0
    lower: Int64[NDArray, " 2"] = np.floor(continuous - radius).astype(
        np.int64
    )
    upper: Int64[NDArray, " 2"] = np.ceil(continuous + radius).astype(np.int64)
    first_coefficient: int
    second_coefficient: int
    coefficients: Int64[NDArray, " 2"]
    for first_coefficient in range(int(lower[0]), int(upper[0]) + 1):
        for second_coefficient in range(int(lower[1]), int(upper[1]) + 1):
            coefficients = np.asarray(
                (first_coefficient, second_coefficient),
                dtype=np.int64,
            )
            candidate_norm: float = perpendicular_norm_squared(coefficients)
            candidate_key: Tuple[float, int, int] = (
                candidate_norm,
                first_coefficient,
                second_coefficient,
            )
            best_key: Tuple[float, int, int] = (
                best_norm,
                int(best_coefficients[0]),
                int(best_coefficients[1]),
            )
            if candidate_key < best_key:
                best_norm = candidate_norm
                best_coefficients = coefficients
    closest: Int64[NDArray, " 3"] = stacking + best_coefficients @ kernel
    return closest


def _validate_miller(
    miller: Tuple[int, int, int],
) -> Tuple[int, int, int]:
    """PRIVATE: Validate a static primitive Miller tuple.

    Parameters
    ----------
    miller : Tuple[int, int, int]
        Candidate Miller indices.

    Returns
    -------
    miller : Tuple[int, int, int]
        The unchanged, validated tuple.

    Raises
    ------
    ValueError
        If the value is not a tuple of three integers, equals
        ``(0, 0, 0)``, or is not gcd-reduced.

    Notes
    -----
    Exact integer gcd arithmetic performs the primitivity check on the
    static host value before any surface construction.
    """
    if (
        type(miller) is not tuple
        or len(miller) != 3  # noqa: PLR2004
        or any(type(component) is not int for component in miller)
    ):
        message: str = "miller must be a tuple of three integers"
        raise ValueError(message)
    divisor: int = math.gcd(
        math.gcd(abs(miller[0]), abs(miller[1])),
        abs(miller[2]),
    )
    if divisor == 0:
        message = "miller must not be (0, 0, 0)"
        raise ValueError(message)
    if divisor != 1:
        message = "miller must be gcd-reduced"
        raise ValueError(message)
    return miller


def _surface_rotation(
    reciprocal_normal: Float64[Array, " 3"],
) -> Float64[Array, "3 3"]:
    """PRIVATE: Construct the guarded active rotation from a normal to +z.

    Parameters
    ----------
    reciprocal_normal : Float64[Array, " 3"]
        Reciprocal-lattice surface normal in 1/Angstrom, any nonzero
        scale.

    Returns
    -------
    rotation : Float64[Array, "3 3"]
        Proper rotation that maps the unit normal onto ``(0, 0, 1)``.

    Notes
    -----
    The generic branch is Rodrigues' formula built from the skew matrix
    of ``normal x z``: ``I + K + K @ K / (1 + cos)``. ``jnp.where``
    guards the antipode. Within ``1e-12`` of ``normal = -z`` the
    selection sanitizes the denominator and picks the fixed pi rotation
    about x, ``diag(1, -1, -1)``, so no singular branch runs.
    :func:`diffpes.maths.safe_norm` normalizes the input
    under its registered boundary convention. At ``normal = +z`` the
    formula reduces exactly to the identity.
    """
    normal_norm: Float64[Array, ""] = safe_norm(reciprocal_normal)
    normal: Float64[Array, " 3"] = reciprocal_normal / normal_norm
    z_axis: Float64[Array, " 3"] = jnp.asarray(
        (0.0, 0.0, 1.0),
        dtype=normal.dtype,
    )
    cross_vector: Float64[Array, " 3"] = jnp.cross(normal, z_axis)
    cross_x: Float64[Array, ""] = cross_vector[0]
    cross_y: Float64[Array, ""] = cross_vector[1]
    cross_z: Float64[Array, ""] = cross_vector[2]
    zero: Float64[Array, ""] = jnp.zeros_like(cross_x)
    skew: Float64[Array, "3 3"] = jnp.stack(
        (
            jnp.stack((zero, -cross_z, cross_y)),
            jnp.stack((cross_z, zero, -cross_x)),
            jnp.stack((-cross_y, cross_x, zero)),
        )
    )
    cosine: Float64[Array, ""] = jnp.dot(normal, z_axis)
    away_from_antipode: Bool[Array, ""] = cosine > -1.0 + 1e-12
    sanitized_denominator: Float64[Array, ""] = jnp.where(
        away_from_antipode,
        1.0 + cosine,
        1.0,
    )
    identity: Float64[Array, "3 3"] = jnp.eye(3, dtype=normal.dtype)
    generic: Float64[Array, "3 3"] = (
        identity + skew + (skew @ skew) / sanitized_denominator
    )
    antiparallel: Float64[Array, "3 3"] = jnp.diag(
        jnp.asarray((1.0, -1.0, -1.0), dtype=normal.dtype)
    )
    rotation: Float64[Array, "3 3"] = jnp.where(
        away_from_antipode,
        generic,
        antiparallel,
    )
    return rotation


def _assemble_surface_cell(
    geometry: CrystalGeometry,
    miller: Tuple[int, int, int],
    in_plane_coeffs: Tuple[
        Tuple[int, int, int],
        Tuple[int, int, int],
    ],
    stacking_coeffs: Tuple[int, int, int],
) -> SurfaceCell:
    """PRIVATE: Assemble continuous surface geometry from frozen topology.

    Parameters
    ----------
    geometry : CrystalGeometry
        Differentiable bulk geometry.
    miller : Tuple[int, int, int]
        Static primitive Miller indices.
    in_plane_coeffs : Tuple[Tuple[int, int, int], Tuple[int, int, int]]
        Frozen integer coefficients of the two in-plane vectors.
    stacking_coeffs : Tuple[int, int, int]
        Frozen integer coefficients of the one-plane stacking vector.

    Returns
    -------
    surface_cell : SurfaceCell
        Rotated surface-frame vectors, interplanar spacing in Angstrom,
        and the frozen integer provenance.

    Notes
    -----
    The traced stage multiplies the frozen integer coefficients into the
    differentiable lattice and computes the reciprocal normal
    ``miller @ reciprocal``. It rotates all three vectors into the
    surface frame and derives the interplanar spacing ``2 * pi / |G|``.
    Within one frozen integer choice every output is differentiable in
    the bulk geometry.
    """
    coefficient_array: Float64[Array, "3 3"] = jnp.asarray(
        (*in_plane_coeffs, stacking_coeffs),
        dtype=jnp.float64,
    )
    bulk_vectors: Float64[Array, "3 3"] = coefficient_array @ geometry.lattice
    miller_array: Float64[Array, " 3"] = jnp.asarray(
        miller,
        dtype=jnp.float64,
    )
    reciprocal_normal: Float64[Array, " 3"] = (
        miller_array @ geometry.reciprocal
    )
    rotation: Float64[Array, "3 3"] = _surface_rotation(reciprocal_normal)
    surface_vectors: Float64[Array, "3 3"] = bulk_vectors @ rotation.T
    spacing: Float64[Array, ""] = 2.0 * jnp.pi / safe_norm(reciprocal_normal)
    surface_cell: SurfaceCell = make_surface_cell(
        in_plane_vectors=surface_vectors[:2],
        stacking_vector=surface_vectors[2],
        rotation=rotation,
        interlayer_spacing_ang=spacing,
        miller=miller,
        in_plane_coeffs=in_plane_coeffs,
        stacking_coeffs=stacking_coeffs,
    )
    return surface_cell


@jaxtyped(typechecker=beartype)
def find_surface_cell(  # noqa: DOC502
    geometry: CrystalGeometry,
    miller: Tuple[int, int, int],
) -> SurfaceCell:
    """Build an exact primitive surface cell for one Miller plane.

    The static host stage constructs a unimodular integer frame. It reduces
    the in-plane basis and solves the two-dimensional closest-vector problem
    without a fixed search radius. The traced stage assembles Cartesian
    vectors and the surface rotation.

    Parameters
    ----------
    geometry : CrystalGeometry
        Bulk geometry. Continuous lattice leaves support differentiation
        within one selected integer surface topology.
    miller : Tuple[int, int, int]
        Primitive Miller indices. The factory rejects nonprimitive tuples.

    Returns
    -------
    surface_cell : SurfaceCell
        Surface-frame vectors and their exact bulk integer coefficients.

    Raises
    ------
    ValueError
        If the Miller tuple is zero, nonintegral, or not gcd-reduced.

    Notes
    -----
    This is a host-only topology-selection factory: do not call it inside
    ``jit``, ``grad``, or ``vmap``. Metric reduction and closest-vector
    selection inspect concrete lattice values. The returned exact integer
    coefficients can subsequently drive traced continuous assembly.

    Metric reduction is a discrete host-side choice. Derivatives with respect
    to the lattice are exact within the selected integer cell; a perturbation
    that changes the selected cell is a new topology-selection event.

    :see: :class:`~.test_slab_surface_cell.TestFindSurfaceCell`
    """
    primitive_miller: Tuple[int, int, int] = _validate_miller(miller)
    lattice_snapshot: Float64[NDArray, "3 3"] = np.asarray(
        geometry.lattice,
        dtype=np.float64,
    )
    kernel: Int64[NDArray, "2 3"]
    stacking: Int64[NDArray, " 3"]
    kernel, stacking = _primitive_integer_frame(primitive_miller)
    kernel = _gauss_reduce(kernel, lattice_snapshot)
    stacking = _closest_stacking_vector(
        stacking,
        kernel,
        lattice_snapshot,
    )
    in_plane_coeffs: Tuple[
        Tuple[int, int, int],
        Tuple[int, int, int],
    ] = (
        tuple(int(value) for value in kernel[0]),
        tuple(int(value) for value in kernel[1]),
    )
    stacking_coeffs: Tuple[int, int, int] = tuple(
        int(value) for value in stacking
    )
    surface_cell: SurfaceCell = _assemble_surface_cell(
        geometry=geometry,
        miller=primitive_miller,
        in_plane_coeffs=in_plane_coeffs,
        stacking_coeffs=stacking_coeffs,
    )
    return surface_cell


__all__: list[str] = [
    "find_surface_cell",
]
