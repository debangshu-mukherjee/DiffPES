r"""Construct exact Miller-index surface cells and rotate TB models.

Extended Summary
----------------
Surface construction has a deliberately split implementation. Exact integer
bookkeeping selects a primitive surface basis and a one-plane stacking
vector on the host. JAX then assembles the Cartesian vectors, interplanar
spacing, and bulk-to-surface rotation from the differentiable crystal
geometry. The integer choice is piecewise constant under lattice changes;
within one choice all continuous geometry operations remain differentiable.

The whole-model rotation applies the corresponding real-harmonic Wigner
representation to every complete atomic shell. Spinor models additionally
rotate their spin channels. Incomplete shells can only pass through the
identity no-op. A sliced Wigner matrix does not define a covariant orbital
transformation.

Routine Listings
----------------
:func:`find_surface_cell`
    Build an exact primitive surface cell for one Miller plane.
:func:`freeze_slab_topology`
    Freeze every discrete choice required to rebuild one slab.
:func:`rebuild_slab`
    Construct a slab from frozen topology using only JAX geometry.
:func:`gen_slab`
    Construct a finite Miller-index slab with exact open-normal topology.
:func:`gen_slab_with_operators`
    Construct a slab while preserving its Wannier operator sidecar.
:func:`rotate_tb_model`
    Construct a rotated complete-shell tight-binding model.
:func:`validate_open_surface_adjacency`
    Reject direct or component-propagated periodic normal images.

Notes
-----
For a unit normal ``n`` and ``z = (0, 0, 1)``, the active Cartesian rotation
uses Rodrigues' formula with axis ``n x z`` and angle
``atan2(|n x z|, n . z)``. Safe norm and arctangent primitives give the
identity at ``n = z`` and a fixed pi rotation around x at ``n = -z`` without
evaluating a singular normalization branch.
"""

from __future__ import annotations

import math

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from beartype import beartype
from beartype.typing import Dict, Tuple
from jaxtyping import Array, Complex128, Float64, Int64, jaxtyped
from numpy.typing import NDArray

from diffpes.maths import (
    real_harmonic_unitary,
    safe_arccos,
    safe_arctan2,
    safe_norm,
    wigner_d,
)
from diffpes.types import (
    CrystalGeometry,
    OrbitalBasis,
    SlabSpec,
    SlabTopology,
    SurfaceCell,
    TBModel,
    WannierOperatorData,
    make_crystal_geometry,
    make_orbital_basis,
    make_slab_spec,
    make_surface_cell,
    make_tb_model,
    make_wannier_operator_data,
)


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
    away_from_antipode: Array = cosine > -1.0 + 1e-12
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

    :see: :class:`~.test_slab.TestFindSurfaceCell`
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


def _shell_groups(
    model: TBModel,
) -> Dict[Tuple[int, int, int, int], list[int]]:
    """PRIVATE: Compute orbital groups by site, principal shell, l, and spin.

    Parameters
    ----------
    model : TBModel
        Model that supplies the static basis metadata.

    Returns
    -------
    groups : Dict[Tuple[int, int, int, int], list[int]]
        Map from ``(atom, n, l, spin)`` to the basis positions of that
        shell's orbitals.

    Notes
    -----
    A spinless model uses the placeholder spin ``0`` for every orbital,
    so each spatial shell forms one group. The groups are the units on
    which Wigner rotation blocks act.
    """
    groups: Dict[Tuple[int, int, int, int], list[int]] = {}
    orbital: int
    atom: int
    principal: int
    angular: int
    spin: int
    for orbital, (
        atom,
        principal,
        angular,
        spin,
    ) in enumerate(
        zip(
            model.basis.atom_indices,
            model.basis.n,
            model.basis.l,
            model.basis.spin if model.spinor else (0,) * len(model.basis.n),
            strict=True,
        )
    ):
        groups.setdefault((atom, principal, angular, spin), []).append(orbital)
    return groups


def _missing_magnetic_numbers(
    model: TBModel,
) -> Dict[
    Tuple[int, int, int, int],
    Tuple[int, ...],
]:
    """PRIVATE: Return missing m values for every incomplete shell.

    Parameters
    ----------
    model : TBModel
        Model carrying the registered shells to audit.

    Returns
    -------
    missing : Dict[Tuple[int, int, int, int], Tuple[int, ...]]
        Map from every incomplete ``(atom, n, l, spin)`` shell to its
        absent magnetic numbers. Complete shells do not appear.

    Notes
    -----
    A shell is incomplete when an ``m`` value from ``-l..+l`` is absent,
    an ``m`` value repeats, or the orbital count differs from
    ``2*l + 1``. A duplicated shell can report an empty absent tuple, so
    callers must test dictionary membership, not tuple truth. Only
    complete shells support a covariant Wigner rotation.
    """
    missing: Dict[Tuple[int, int, int, int], Tuple[int, ...]] = {}
    key: Tuple[int, int, int, int]
    indices: list[int]
    for key, indices in _shell_groups(model).items():
        angular: int = key[2]
        present: list[int] = [model.basis.m[index] for index in indices]
        expected: set[int] = set(range(-angular, angular + 1))
        absent: Tuple[int, ...] = tuple(sorted(expected - set(present)))
        duplicated: bool = len(present) != len(set(present))
        if absent or duplicated or len(present) != 2 * angular + 1:
            missing[key] = absent
    return missing


def _rotation_euler_zyz(
    rotation: Float64[Array, "3 3"],
) -> Tuple[Float64[Array, ""], Float64[Array, ""], Float64[Array, ""]]:
    """PRIVATE: Convert one active Cartesian rotation to guarded z-y-z angles.

    Parameters
    ----------
    rotation : Float64[Array, "3 3"]
        Proper active rotation matrix.

    Returns
    -------
    result : tuple
        Euler angles ``(alpha, beta, gamma)`` in radians for the z-y-z
        convention, each a scalar array.

    Notes
    -----
    ``beta`` comes from :func:`diffpes.maths.safe_arccos` of the lower
    right element. Away from the poles, generic ``arctan2`` expressions
    read the third row and column. Within ``1e-12`` of
    ``sin(beta) = 0`` the two z rotations degenerate. The guarded branch
    assigns the whole in-plane angle to ``alpha`` from the upper block
    and sets ``gamma`` to zero. The sign branch follows the pole. All
    selections use ``jnp.where``, so the angles stay traced.
    """
    beta: Float64[Array, ""] = safe_arccos(rotation[2, 2])
    sine_beta: Float64[Array, ""] = jnp.sin(beta)
    generic_alpha: Float64[Array, ""] = safe_arctan2(
        rotation[1, 2],
        rotation[0, 2],
    )
    generic_gamma: Float64[Array, ""] = safe_arctan2(
        rotation[2, 1],
        -rotation[2, 0],
    )
    positive_alpha: Float64[Array, ""] = safe_arctan2(
        rotation[1, 0],
        rotation[0, 0],
    )
    negative_alpha: Float64[Array, ""] = safe_arctan2(
        -rotation[1, 0],
        -rotation[0, 0],
    )
    pole_alpha: Float64[Array, ""] = jnp.where(
        rotation[2, 2] >= 0.0,
        positive_alpha,
        negative_alpha,
    )
    alpha: Float64[Array, ""] = jnp.where(
        jnp.abs(sine_beta) > 1e-12,  # noqa: PLR2004
        generic_alpha,
        pole_alpha,
    )
    gamma: Float64[Array, ""] = jnp.where(
        jnp.abs(sine_beta) > 1e-12,  # noqa: PLR2004
        generic_gamma,
        0.0,
    )
    result: Tuple[
        Float64[Array, ""],
        Float64[Array, ""],
        Float64[Array, ""],
    ] = (alpha, beta, gamma)
    return result


def _real_wigner(
    angular: int,
    alpha: Float64[Array, ""],
    beta: Float64[Array, ""],
    gamma: Float64[Array, ""],
) -> Complex128[Array, "m1 m2"]:
    """PRIVATE: Convert the complex Wigner representation to real harmonics.

    Parameters
    ----------
    angular : int
        Static shell angular momentum.
    alpha : Float64[Array, ""]
        First z-y-z Euler angle in radians.
    beta : Float64[Array, ""]
        Second z-y-z Euler angle in radians.
    gamma : Float64[Array, ""]
        Third z-y-z Euler angle in radians.

    Returns
    -------
    real_matrix : Complex128[Array, "m1 m2"]
        Wigner rotation of the shell in the package real-harmonic basis.

    Notes
    -----
    The transform is the package operator convention
    ``U.conj() @ D @ U.T``, with :func:`diffpes.maths.wigner_d`
    supplying the complex-harmonic matrix and
    :func:`diffpes.maths.real_harmonic_unitary` supplying ``U``.
    """
    complex_matrix: Complex128[Array, "m1 m2"] = wigner_d(
        angular,
        alpha,
        beta,
        gamma,
    )
    unitary: Complex128[Array, "m1 m2"] = real_harmonic_unitary(angular)
    real_matrix: Complex128[Array, "m1 m2"] = (
        unitary.conj() @ complex_matrix @ unitary.T
    )
    return real_matrix


def _spin_half_wigner(
    alpha: Float64[Array, ""],
    beta: Float64[Array, ""],
    gamma: Float64[Array, ""],
) -> Complex128[Array, "2 2"]:
    """PRIVATE: Construct the spin-half Wigner matrix in (-1, +1) order.

    Parameters
    ----------
    alpha : Float64[Array, ""]
        First z-y-z Euler angle in radians.
    beta : Float64[Array, ""]
        Second z-y-z Euler angle in radians.
    gamma : Float64[Array, ""]
        Third z-y-z Euler angle in radians.

    Returns
    -------
    matrix : Complex128[Array, "2 2"]
        Spinor rotation with rows and columns ordered as spin down then
        spin up.

    Notes
    -----
    The small-d factor uses ``cos(beta/2)`` and ``sin(beta/2)``. The
    phases ``exp(-i m alpha)`` and ``exp(-i m gamma)`` multiply from the
    left and right with ``m = (-1/2, +1/2)``. The down--up ordering
    matches the package C8 spin storage convention.
    """
    cosine: Float64[Array, ""] = jnp.cos(0.5 * beta)
    sine: Float64[Array, ""] = jnp.sin(0.5 * beta)
    small: Float64[Array, "2 2"] = jnp.stack(
        (
            jnp.stack((cosine, sine)),
            jnp.stack((-sine, cosine)),
        )
    )
    magnetic: Float64[Array, " 2"] = jnp.asarray(
        (-0.5, 0.5),
        dtype=beta.dtype,
    )
    alpha_phase: Complex128[Array, " 2"] = jnp.exp(-1j * magnetic * alpha)
    gamma_phase: Complex128[Array, " 2"] = jnp.exp(-1j * magnetic * gamma)
    matrix: Complex128[Array, "2 2"] = (
        alpha_phase[:, None] * small * gamma_phase[None, :]
    )
    return matrix


def _orbital_rotation(
    model: TBModel,
    rotation: Float64[Array, "3 3"],
) -> Complex128[Array, "n_orb n_orb"]:
    """PRIVATE: Assemble the block-diagonal orbital and spin representation.

    Parameters
    ----------
    model : TBModel
        Model whose static basis metadata defines the shell blocks.
    rotation : Float64[Array, "3 3"]
        Proper active Cartesian rotation.

    Returns
    -------
    representation : Complex128[Array, "n_orb n_orb"]
        Unitary rotation representation in the model basis order.

    Notes
    -----
    Guarded z-y-z Euler angles feed one real-harmonic Wigner matrix per
    distinct ``l`` and, for a spinor model, one spin-half matrix. Every
    entry couples two orbitals of the same ``(atom, n, l)`` shell. It
    multiplies the angular factor at their magnetic numbers by the
    spin-half factor at their spin labels; a spinless model uses a unit
    spin factor. Orbitals of different shells never mix, so the matrix
    is block diagonal over complete shells in any basis order.
    """
    alpha: Float64[Array, ""]
    beta: Float64[Array, ""]
    gamma: Float64[Array, ""]
    alpha, beta, gamma = _rotation_euler_zyz(rotation)
    n_orbitals: int = len(model.basis.n)
    representation: Complex128[Array, "n_orb n_orb"] = jnp.zeros(
        (n_orbitals, n_orbitals),
        dtype=jnp.complex128,
    )
    angular_matrices: Dict[int, Complex128[Array, "m1 m2"]] = {
        angular: _real_wigner(angular, alpha, beta, gamma)
        for angular in set(model.basis.l)
    }
    spin_matrix: Complex128[Array, "2 2"] | None = (
        _spin_half_wigner(alpha, beta, gamma) if model.spinor else None
    )
    row: int
    column: int
    for row in range(n_orbitals):
        for column in range(n_orbitals):
            same_shell: bool = (
                model.basis.atom_indices[row]
                == model.basis.atom_indices[column]
                and model.basis.n[row] == model.basis.n[column]
                and model.basis.l[row] == model.basis.l[column]
            )
            if not same_shell:
                continue
            angular: int = model.basis.l[row]
            angular_factor: Complex128[Array, ""] = angular_matrices[angular][
                model.basis.m[row] + angular,
                model.basis.m[column] + angular,
            ]
            if spin_matrix is None:
                spin_factor: complex | Complex128[Array, ""] = (
                    1.0 if row == column or same_shell else 0.0
                )
            else:
                row_spin: int = 0 if model.basis.spin[row] == -1 else 1
                column_spin: int = 0 if model.basis.spin[column] == -1 else 1
                spin_factor = spin_matrix[row_spin, column_spin]
            representation = representation.at[row, column].set(
                angular_factor * spin_factor
            )
    return representation


def _translation_blocks(
    model: TBModel,
) -> Tuple[Tuple[Tuple[int, int, int], ...], Complex128[Array, "n_r n_o n_o"]]:
    """PRIVATE: Materialize translation blocks with diagonal onsite terms.

    Parameters
    ----------
    model : TBModel
        Model supplying the sparse hopping records to densify.

    Returns
    -------
    result : tuple
        Sorted static cell tuples and one dense complex block per cell.
        Each block holds the hoppings in eV scattered by pair; the
        zero-cell block also carries the onsite energies on its
        diagonal.

    Notes
    -----
    The zero cell is always present even without home-cell hoppings, so
    the onsite diagonal has a destination. Dense blocks let one unitary
    conjugation rotate arbitrary onsite and hopping structure without
    assuming shell degeneracy.
    """
    zero_cell: Tuple[int, int, int] = (0, 0, 0)
    cells: Tuple[Tuple[int, int, int], ...] = tuple(
        sorted(set(model.hopping_cells) | {zero_cell})
    )
    n_orbitals: int = len(model.basis.n)
    blocks: Complex128[Array, "n_r n_o n_o"] = jnp.zeros(
        (len(cells), n_orbitals, n_orbitals),
        dtype=jnp.complex128,
    )
    cell_lookup: Dict[Tuple[int, int, int], int] = {
        cell: index for index, cell in enumerate(cells)
    }
    hopping: int
    pair: Tuple[int, int]
    cell: Tuple[int, int, int]
    for hopping, (pair, cell) in enumerate(
        zip(model.hopping_pairs, model.hopping_cells, strict=True)
    ):
        blocks = blocks.at[
            cell_lookup[cell],
            pair[0],
            pair[1],
        ].add(model.hopping_amplitudes[hopping])
    diagonal: Array = jnp.arange(n_orbitals, dtype=jnp.int32)
    blocks = blocks.at[cell_lookup[zero_cell], diagonal, diagonal].add(
        model.onsite_energies
    )
    result: Tuple[
        Tuple[Tuple[int, int, int], ...],
        Complex128[Array, "n_r n_o n_o"],
    ] = (cells, blocks)
    return result


@jaxtyped(typechecker=beartype)
def rotate_tb_model(  # noqa: DOC503
    model: TBModel,
    rotation: Float64[Array, "3 3"],
) -> TBModel:
    """Construct a rotated complete-shell tight-binding model.

    Parameters
    ----------
    model : TBModel
        Bulk model in a real-harmonic orbital basis.
    rotation : Float64[Array, "3 3"]
        Proper active Cartesian rotation from the old frame to the new frame.

    Returns
    -------
    rotated_model : TBModel
        Ordinary model in the rotated Cartesian and orbital frame.

    Raises
    ------
    ValueError
        If an incomplete ``m`` shell receives a nonidentity rotation.
    EquinoxRuntimeError
        If ``rotation`` is non-finite, improper, or nonorthogonal.

    Notes
    -----
    The block-diagonal real-harmonic representation conjugates every exact
    translation block, including the onsite block. Dense translation blocks
    preserve generic onsite and hopping matrices without assuming shell
    degeneracy. The added static connectivity changes representation, not
    physical coupling.

    :see: :class:`~.test_slab.TestRotateTbModel`
    """
    rotation_array: Float64[Array, "3 3"] = jnp.asarray(
        rotation,
        dtype=jnp.float64,
    )
    missing: Dict[
        Tuple[int, int, int, int],
        Tuple[int, ...],
    ] = _missing_magnetic_numbers(model)
    if missing:
        error: TypeError
        try:
            identity_rotation: bool = bool(
                np.allclose(
                    np.asarray(rotation_array),
                    np.eye(3),
                    rtol=0.0,
                    atol=1e-12,
                )
            )
        except TypeError as error:
            message: str = (
                "incomplete shells require an eager identity/no-op rotation"
            )
            raise ValueError(message) from error
        if identity_rotation:
            return model
        diagnostics: str = "; ".join(
            f"{key}: missing m={values}"
            for key, values in sorted(missing.items())
        )
        message = (
            "nonidentity rotation requires complete registered m shells; "
            f"{diagnostics}"
        )
        raise ValueError(message)

    rotation_array = eqx.error_if(
        rotation_array,
        ~jnp.all(jnp.isfinite(rotation_array)),
        "rotate_tb_model: rotation must be finite",
    )
    identity: Float64[Array, "3 3"] = jnp.eye(3, dtype=jnp.float64)
    rotation_array = eqx.error_if(
        rotation_array,
        jnp.max(jnp.abs(rotation_array.T @ rotation_array - identity)) > 1e-10,  # noqa: PLR2004
        "rotate_tb_model: rotation must be orthogonal",
    )
    rotation_array = eqx.error_if(
        rotation_array,
        jnp.abs(jnp.linalg.det(rotation_array) - 1.0) > 1e-10,  # noqa: PLR2004
        "rotate_tb_model: rotation must be proper",
    )
    representation: Complex128[Array, "n_orb n_orb"] = _orbital_rotation(
        model,
        rotation_array,
    )
    cells: Tuple[Tuple[int, int, int], ...]
    blocks: Complex128[Array, "n_r n_o n_o"]
    cells, blocks = _translation_blocks(model)
    rotated_blocks: Complex128[Array, "n_r n_o n_o"] = (
        representation[None, :, :]
        @ blocks
        @ representation.conj().T[None, :, :]
    )
    n_orbitals: int = len(model.basis.n)
    hopping_pairs: Tuple[Tuple[int, int], ...] = tuple(
        (row, column)
        for _cell in cells
        for row in range(n_orbitals)
        for column in range(n_orbitals)
    )
    hopping_cells: Tuple[Tuple[int, int, int], ...] = tuple(
        cell
        for cell in cells
        for _row in range(n_orbitals)
        for _column in range(n_orbitals)
    )
    hopping_amplitudes: Complex128[Array, " n_hop"] = rotated_blocks.reshape(
        -1
    )
    rotated_lattice: Float64[Array, "3 3"] = (
        model.geometry.lattice @ rotation_array.T
    )
    rotated_geometry: CrystalGeometry = make_crystal_geometry(
        lattice=rotated_lattice,
        positions=model.geometry.positions,
        species=model.geometry.species,
    )
    rotated_model: TBModel = make_tb_model(
        hopping_amplitudes=hopping_amplitudes,
        onsite_energies=jnp.zeros_like(model.onsite_energies),
        soc_lambdas=model.soc_lambdas,
        geometry=rotated_geometry,
        basis=model.basis,
        hopping_pairs=hopping_pairs,
        hopping_cells=hopping_cells,
        shell_index=model.shell_index,
        spinor=model.spinor,
        orbital_positions=model.orbital_positions,
        depths=model.depths,
    )
    return rotated_model


def _surface_integer_matrix(
    surface_cell: SurfaceCell,
) -> Int64[NDArray, "3 3"]:
    """PRIVATE: Return the exact row-wise bulk-to-surface integer frame.

    Parameters
    ----------
    surface_cell : SurfaceCell
        Surface cell carrying frozen integer coefficients.

    Returns
    -------
    coefficients : Int64[NDArray, "3 3"]
        Rows holding the two in-plane vectors and the stacking vector in
        bulk fractional coordinates.

    Raises
    ------
    ValueError
        If the exact integer determinant is not one.

    Notes
    -----
    Determinant ``+1`` certifies an oriented unimodular frame, so bulk
    and surface integer coordinates stay exactly interconvertible.
    """
    coefficients: Int64[NDArray, "3 3"] = np.asarray(
        (
            surface_cell.in_plane_coeffs[0],
            surface_cell.in_plane_coeffs[1],
            surface_cell.stacking_coeffs,
        ),
        dtype=np.int64,
    )
    determinant: int = _determinant_3x3(coefficients)
    if determinant != 1:
        message: str = (
            "surface integer coefficients must form an oriented "
            "unimodular frame"
        )
        raise ValueError(message)
    return coefficients


def _inverse_integer_matrix(
    coefficients: Int64[NDArray, "3 3"],
) -> Int64[NDArray, "3 3"]:
    """PRIVATE: Compute a unimodular inverse and verify exact recovery.

    Parameters
    ----------
    coefficients : Int64[NDArray, "3 3"]
        Integer frame with determinant one.

    Returns
    -------
    inverse : Int64[NDArray, "3 3"]
        Exact integer inverse of the frame.

    Raises
    ------
    ValueError
        If the determinant is not one or the product with the candidate
        inverse is not exactly the identity.

    Notes
    -----
    For a determinant-one matrix the inverse equals the adjugate, so
    every entry is an exact integer cofactor. The final multiplication
    check guards against silent integer overflow in the ``int64`` cast.
    """
    determinant: int = _determinant_3x3(coefficients)
    if determinant != 1:
        message: str = "surface integer frame must have determinant one"
        raise ValueError(message)
    a: int = int(coefficients[0, 0])
    b: int = int(coefficients[0, 1])
    c: int = int(coefficients[0, 2])
    d: int = int(coefficients[1, 0])
    e: int = int(coefficients[1, 1])
    f: int = int(coefficients[1, 2])
    g: int = int(coefficients[2, 0])
    h: int = int(coefficients[2, 1])
    i: int = int(coefficients[2, 2])
    inverse: Int64[NDArray, "3 3"] = np.asarray(
        (
            (e * i - f * h, c * h - b * i, b * f - c * e),
            (f * g - d * i, a * i - c * g, c * d - a * f),
            (d * h - e * g, b * g - a * h, a * e - b * d),
        ),
        dtype=np.int64,
    )
    identity: Int64[NDArray, "3 3"] = coefficients @ inverse
    if not np.array_equal(identity, np.eye(3, dtype=np.int64)):
        message: str = "surface integer frame inverse is not exact"
        raise ValueError(message)
    return inverse


def _base_surface_coordinates(
    geometry: CrystalGeometry,
    inverse_coefficients: Int64[NDArray, "3 3"],
) -> Tuple[Float64[NDArray, "n_atom 3"], Int64[NDArray, "n_atom 3"]]:
    """PRIVATE: Compute atom representatives and integer surface-cell shifts.

    Parameters
    ----------
    geometry : CrystalGeometry
        Concrete bulk geometry.
    inverse_coefficients : Int64[NDArray, "3 3"]
        Exact inverse of the bulk-to-surface integer frame.

    Returns
    -------
    result : Tuple[Float64[NDArray, "n_atom 3"], Int64[NDArray, "n_atom 3"]]
        Surface fractional representatives in ``[0, 1)`` and the exact
        integer shifts removed from every atom.

    Notes
    -----
    Multiplying bulk fractional positions by the inverse frame gives
    surface fractional coordinates. A ``floor`` with a ``1e-10`` guard
    extracts the integer surface-cell part. Coordinates within the same
    tolerance of one snap back to zero, so atoms on the upper boundary
    join the home cell deterministically.
    """
    surface_coordinates: Float64[NDArray, "n_atom 3"] = (
        np.asarray(geometry.positions, dtype=np.float64) @ inverse_coefficients
    )
    shifts: Int64[NDArray, "n_atom 3"] = np.floor(
        surface_coordinates + 1e-10
    ).astype(np.int64)
    base: Float64[NDArray, "n_atom 3"] = surface_coordinates - shifts
    base[np.isclose(base, 1.0, atol=1e-10)] = 0.0
    result: Tuple[Float64[NDArray, "n_atom 3"], Int64[NDArray, "n_atom 3"]] = (
        base,
        shifts,
    )
    return result


def _natural_atom_copies(
    geometry: CrystalGeometry,
    base_coordinates: Float64[NDArray, "n_atom 3"],
    surface_vectors: Float64[NDArray, "3 3"],
    n_layers: int,
    thickness_ang: float,
    fine: Tuple[float, float],
) -> Tuple[Tuple[int, ...], Tuple[int, ...], Tuple[str, str]]:
    """PRIVATE: Select the smallest natural stack meeting the minimum span.

    Parameters
    ----------
    geometry : CrystalGeometry
        Concrete bulk geometry supplying species labels.
    base_coordinates : Float64[NDArray, "n_atom 3"]
        Surface fractional atom representatives in ``[0, 1)``.
    surface_vectors : Float64[NDArray, "3 3"]
        Concrete surface-frame vectors in angstroms.
    n_layers : int
        Starting number of stacked one-plane layers.
    thickness_ang : float
        Minimum retained material span in Angstrom.
    fine : Tuple[float, float]
        Inward ``(top, bottom)`` cut shifts in Angstrom.

    Returns
    -------
    result : Tuple[Tuple[int, ...], Tuple[int, ...], Tuple[str, str]]
        Bulk atom index and normalized layer of every retained copy, and
        the resolved ``(top, bottom)`` termination species.

    Raises
    ------
    ValueError
        If no expanded stack keeps the requested span after the fine
        cuts.

    Notes
    -----
    The search grows the layer count from ``n_layers`` upward. For each
    candidate count it keeps every atom copy whose Cartesian height lies
    between the fine-shifted bottom and top cuts, with a ``1e-10``
    tolerance. The search accepts the first count whose kept span
    reaches ``thickness_ang``. Kept copies sort by layer then atom, and
    layers renumber densely from zero. The extreme-height atoms name the
    termination; a species-free geometry reports the placeholder
    ``"X"``.
    """
    spacing: float = abs(float(surface_vectors[2, 2]))
    base_heights: Float64[NDArray, " n_atom"] = (
        base_coordinates @ surface_vectors[:, 2]
    )
    baseline_bottom: float = float(np.min(base_heights))
    base_top: float = float(np.max(base_heights))
    positive_fine_span: float = max(fine[0], 0.0) + max(fine[1], 0.0)
    maximum_extra_layers: int = (
        int(math.ceil(positive_fine_span / spacing)) + 3
    )
    extra_span: float = max(abs(fine[0]), abs(fine[1]))
    padding: int = int(math.ceil(extra_span / spacing)) + 1
    candidates: list[Tuple[float, int, int]] = []
    expanded_layers: int
    for expanded_layers in range(
        n_layers,
        n_layers + maximum_extra_layers + 1,
    ):
        baseline_top: float = base_top + (expanded_layers - 1) * spacing
        bottom_cut: float = baseline_bottom + fine[1]
        top_cut: float = baseline_top - fine[0]
        candidates = []
        layer: int
        atom: int
        for layer in range(-padding, expanded_layers + padding):
            for atom in range(base_coordinates.shape[0]):
                height: float = float(base_heights[atom] + layer * spacing)
                if height >= bottom_cut - 1e-10 and height <= top_cut + 1e-10:
                    candidates.append((height, atom, layer))
        if candidates and (
            max(row[0] for row in candidates)
            - min(row[0] for row in candidates)
            + 1e-10
            >= thickness_ang
        ):
            break
    else:
        message: str = (
            "fine shifts cannot preserve the requested minimum slab thickness"
        )
        raise ValueError(message)
    candidates.sort(key=lambda row: (row[2], row[1]))
    unique_layers: Tuple[int, ...] = tuple(
        sorted({row[2] for row in candidates})
    )
    layer_map: Dict[int, int] = {
        original: normalized
        for normalized, original in enumerate(unique_layers)
    }
    bulk_atoms: Tuple[int, ...] = tuple(row[1] for row in candidates)
    layers: Tuple[int, ...] = tuple(layer_map[row[2]] for row in candidates)
    heights: Float64[NDArray, " n_candidate"] = np.asarray(
        [row[0] for row in candidates],
        dtype=np.float64,
    )
    bottom_index: int = int(np.argmin(heights))
    top_index: int = int(np.argmax(heights))
    species: Tuple[str, ...] = geometry.species
    if not species:
        species = tuple("X" for _ in range(geometry.positions.shape[0]))
    termination: Tuple[str, str] = (
        species[candidates[top_index][1]],
        species[candidates[bottom_index][1]],
    )
    result: Tuple[
        Tuple[int, ...],
        Tuple[int, ...],
        Tuple[str, str],
    ] = (bulk_atoms, layers, termination)
    return result


def _terminated_atom_copies(  # noqa: PLR0913
    geometry: CrystalGeometry,
    base_coordinates: Float64[NDArray, "n_atom 3"],
    surface_vectors: Float64[NDArray, "3 3"],
    n_layers: int,
    thickness_ang: float,
    termination: Tuple[str, str],
    fine: Tuple[float, float],
) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
    """PRIVATE: Compute a post-fine stack with requested endpoint species.

    Parameters
    ----------
    geometry : CrystalGeometry
        Concrete bulk geometry; must declare species.
    base_coordinates : Float64[NDArray, "n_atom 3"]
        Surface fractional atom representatives in ``[0, 1)``.
    surface_vectors : Float64[NDArray, "3 3"]
        Concrete surface-frame vectors in angstroms.
    n_layers : int
        Starting number of stacked one-plane layers.
    thickness_ang : float
        Minimum retained material span in Angstrom.
    termination : Tuple[str, str]
        Requested ``(top, bottom)`` species.
    fine : Tuple[float, float]
        Inward ``(top, bottom)`` cut shifts in Angstrom.

    Returns
    -------
    result : Tuple[Tuple[int, ...], Tuple[int, ...]]
        Bulk atom index and normalized layer of every retained copy.

    Raises
    ------
    ValueError
        If the geometry declares no species, or no expanded stack
        provides both requested endpoint species with the minimum span.

    Notes
    -----
    For each candidate layer count the search collects fine-cut atom
    copies. It takes the lowest copy of the bottom species and the
    highest copy of the top species. It keeps every copy between them
    when their separation reaches ``thickness_ang``. The layer count
    grows until this succeeds. Kept copies sort by layer then atom and
    layers renumber densely from zero.
    """
    if not geometry.species:
        message: str = "explicit termination requires geometry species"
        raise ValueError(message)
    top_species: str
    bottom_species: str
    top_species, bottom_species = termination
    spacing: float = abs(float(surface_vectors[2, 2]))
    base_heights: Float64[NDArray, " n_atom"] = (
        base_coordinates @ surface_vectors[:, 2]
    )
    baseline_bottom: float = float(np.min(base_heights))
    base_top: float = float(np.max(base_heights))
    positive_fine_span: float = max(fine[0], 0.0) + max(fine[1], 0.0)
    maximum_extra_layers: int = (
        int(math.ceil(positive_fine_span / spacing)) + 5
    )
    extra_span: float = max(abs(fine[0]), abs(fine[1]))
    padding: int = int(math.ceil(extra_span / spacing)) + 2
    kept: list[Tuple[float, int, int, str]] = []
    expanded_layers: int
    for expanded_layers in range(
        n_layers,
        n_layers + maximum_extra_layers + 1,
    ):
        baseline_top: float = base_top + (expanded_layers - 1) * spacing
        bottom_cut: float = baseline_bottom + fine[1]
        top_cut: float = baseline_top - fine[0]
        candidates: list[Tuple[float, int, int, str]] = []
        layer: int
        atom: int
        for layer in range(-padding, expanded_layers + padding):
            for atom in range(base_coordinates.shape[0]):
                height: float = float(base_heights[atom] + layer * spacing)
                if height >= bottom_cut - 1e-10 and height <= top_cut + 1e-10:
                    candidates.append(
                        (height, atom, layer, geometry.species[atom])
                    )
        bottom_rows: Tuple[Tuple[float, int, int, str], ...] = tuple(
            row for row in candidates if row[3] == bottom_species
        )
        top_rows: Tuple[Tuple[float, int, int, str], ...] = tuple(
            row for row in candidates if row[3] == top_species
        )
        if not bottom_rows or not top_rows:
            continue
        bottom: Tuple[float, int, int, str] = min(
            bottom_rows,
            key=lambda row: (row[0], row[2], row[1]),
        )
        top: Tuple[float, int, int, str] = max(
            top_rows,
            key=lambda row: (row[0], row[2], row[1]),
        )
        if top[0] - bottom[0] + 1e-10 < thickness_ang:
            continue
        kept = [
            row
            for row in candidates
            if row[0] >= bottom[0] - 1e-10 and row[0] <= top[0] + 1e-10
        ]
        break
    else:
        message = (
            "requested terminations cannot preserve the minimum slab "
            "thickness after fine shifts"
        )
        raise ValueError(message)
    kept.sort(key=lambda row: (row[2], row[1]))
    unique_layers: Tuple[int, ...] = tuple(sorted({row[2] for row in kept}))
    layer_map: Dict[int, int] = {
        original: normalized
        for normalized, original in enumerate(unique_layers)
    }
    bulk_atoms: Tuple[int, ...] = tuple(row[1] for row in kept)
    layers: Tuple[int, ...] = tuple(layer_map[row[2]] for row in kept)
    result: Tuple[Tuple[int, ...], Tuple[int, ...]] = (bulk_atoms, layers)
    return result


def _orbital_copy_metadata(
    basis: OrbitalBasis,
    bulk_atom_of_slab_atom: Tuple[int, ...],
    layer_of_slab_atom: Tuple[int, ...],
) -> Tuple[
    Tuple[int, ...],
    Tuple[int, ...],
    Tuple[int, ...],
]:
    """PRIVATE: Compute each slab orbital's bulk, atom, and layer mapping.

    Parameters
    ----------
    basis : OrbitalBasis
        Bulk orbital basis.
    bulk_atom_of_slab_atom : Tuple[int, ...]
        Frozen bulk atom index of every slab atom.
    layer_of_slab_atom : Tuple[int, ...]
        Frozen layer of every slab atom.

    Returns
    -------
    result : Tuple[Tuple[int, ...], Tuple[int, ...], Tuple[int, ...]]
        Bulk orbital, slab atom index, and layer for every slab orbital.

    Notes
    -----
    The walk visits slab atoms in frozen order and copies each bulk
    atom's orbitals in bulk basis order. The slab orbital ordering is
    therefore deterministic and reproducible from the topology alone.
    """
    orbitals_by_atom: Dict[int, list[int]] = {}
    bulk_orbital: int
    for bulk_orbital, atom in enumerate(basis.atom_indices):
        orbitals_by_atom.setdefault(atom, []).append(bulk_orbital)
    slab_to_bulk: list[int] = []
    slab_atom_indices: list[int] = []
    slab_layers: list[int] = []
    slab_atom: int
    bulk_atom: int
    atom: int
    layer: int
    for slab_atom, (bulk_atom, layer) in enumerate(
        zip(
            bulk_atom_of_slab_atom,
            layer_of_slab_atom,
            strict=True,
        )
    ):
        for bulk_orbital in orbitals_by_atom.get(bulk_atom, []):
            slab_to_bulk.append(bulk_orbital)
            slab_atom_indices.append(slab_atom)
            slab_layers.append(layer)
    result: Tuple[
        Tuple[int, ...],
        Tuple[int, ...],
        Tuple[int, ...],
    ] = (
        tuple(slab_to_bulk),
        tuple(slab_atom_indices),
        tuple(slab_layers),
    )
    return result


def _slab_basis(
    basis: OrbitalBasis,
    slab_to_bulk: Tuple[int, ...],
    slab_atom_indices: Tuple[int, ...],
    slab_layers: Tuple[int, ...],
) -> OrbitalBasis:
    """PRIVATE: Create static orbital metadata for one frozen slab topology.

    Parameters
    ----------
    basis : OrbitalBasis
        Bulk orbital basis.
    slab_to_bulk : Tuple[int, ...]
        Bulk orbital behind every slab orbital.
    slab_atom_indices : Tuple[int, ...]
        Slab atom index of every slab orbital.
    slab_layers : Tuple[int, ...]
        Layer of every slab orbital.

    Returns
    -------
    slab_basis : OrbitalBasis
        Validated slab basis with layer-tagged labels.

    Notes
    -----
    Quantum numbers and spin gather from the bulk basis. Labels append
    ``@L<layer>`` to the bulk label, with the fallback ``orb<index>``
    for an unlabeled bulk basis, so every layer copy keeps a distinct
    name.
    """
    labels: Tuple[str, ...] = tuple(
        f"{basis.labels[bulk] if basis.labels else f'orb{bulk}'}@L{layer}"
        for bulk, layer in zip(slab_to_bulk, slab_layers, strict=True)
    )
    spin: Tuple[int, ...] = (
        tuple(basis.spin[bulk] for bulk in slab_to_bulk) if basis.spin else ()
    )
    slab_basis: OrbitalBasis = make_orbital_basis(
        atom_indices=slab_atom_indices,
        n=tuple(basis.n[bulk] for bulk in slab_to_bulk),
        l=tuple(basis.l[bulk] for bulk in slab_to_bulk),
        m=tuple(basis.m[bulk] for bulk in slab_to_bulk),
        spin=spin,
        labels=labels,
    )
    return slab_basis


def _slab_shell_metadata(
    model: TBModel,
    slab_to_bulk: Tuple[int, ...],
    slab_atom_indices: Tuple[int, ...],
) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
    """PRIVATE: Create SOC shell IDs and their bulk-shell gather.

    Parameters
    ----------
    model : TBModel
        Bulk model carrying the original shell map.
    slab_to_bulk : Tuple[int, ...]
        Bulk orbital behind every slab orbital.
    slab_atom_indices : Tuple[int, ...]
        Slab atom index of every slab orbital.

    Returns
    -------
    result : Tuple[Tuple[int, ...], Tuple[int, ...]]
        Contiguous slab shell ID per orbital, with ``-1`` preserved for
        excluded orbitals, and the bulk shell behind every new ID.

    Notes
    -----
    Every ``(slab atom, bulk shell)`` combination receives one fresh
    contiguous ID in first-appearance order. The gather tuple lets the
    rebuild index bulk ``soc_lambdas``, so each atom copy keeps its bulk
    coupling strength.
    """
    shell_lookup: Dict[Tuple[int, int], int] = {}
    slab_shells: list[int] = []
    bulk_shell_gather: list[int] = []
    slab_orbital: int
    bulk_orbital: int
    for slab_orbital, bulk_orbital in enumerate(slab_to_bulk):
        bulk_shell: int = model.shell_index[bulk_orbital]
        if bulk_shell < 0:
            slab_shells.append(-1)
            continue
        key: Tuple[int, int] = (
            slab_atom_indices[slab_orbital],
            bulk_shell,
        )
        if key not in shell_lookup:
            shell_lookup[key] = len(shell_lookup)
            bulk_shell_gather.append(bulk_shell)
        slab_shells.append(shell_lookup[key])
    result: Tuple[Tuple[int, ...], Tuple[int, ...]] = (
        tuple(slab_shells),
        tuple(bulk_shell_gather),
    )
    return result


def _orbital_lookup(
    model: TBModel,
    spec: SlabSpec,
) -> Tuple[
    Tuple[int, ...],
    Tuple[int, ...],
    Tuple[int, ...],
    Dict[Tuple[int, int], int],
]:
    """PRIVATE: Create deterministic slab-orbital lookup metadata.

    Parameters
    ----------
    model : TBModel
        Bulk model that provides the orbital basis to replicate.
    spec : SlabSpec
        Slab provenance carrying frozen atom and layer choices.

    Returns
    -------
    result : tuple
        Bulk orbital, slab atom index, and layer per slab orbital, plus
        a ``(bulk orbital, layer)`` to slab-orbital dictionary.

    Notes
    -----
    The dictionary inverts the deterministic copy order from
    ``_orbital_copy_metadata``, so hopping propagation resolves a target
    orbital copy in constant time.
    """
    slab_to_bulk: Tuple[int, ...]
    slab_atom_indices: Tuple[int, ...]
    slab_layers: Tuple[int, ...]
    slab_to_bulk, slab_atom_indices, slab_layers = _orbital_copy_metadata(
        model.basis,
        spec.bulk_atom_of_slab_atom,
        spec.layer_of_slab_atom,
    )
    lookup: Dict[Tuple[int, int], int] = {
        (bulk, layer): slab
        for slab, (bulk, layer) in enumerate(
            zip(slab_to_bulk, slab_layers, strict=True)
        )
    }
    result: Tuple[
        Tuple[int, ...],
        Tuple[int, ...],
        Tuple[int, ...],
        Dict[Tuple[int, int], int],
    ] = (slab_to_bulk, slab_atom_indices, slab_layers, lookup)
    return result


def _propagate_hoppings_with_shifts(
    rotated_bulk: TBModel,
    spec: SlabSpec,
    atom_shifts: Int64[NDArray, "n_atom 3"],
) -> Tuple[
    Complex128[Array, " n_hop_slab"],
    Tuple[Tuple[int, int], ...],
    Tuple[Tuple[int, int, int], ...],
    Tuple[int, ...],
]:
    """PRIVATE: Propagate exact bulk hoppings through one frozen topology.

    Parameters
    ----------
    rotated_bulk : TBModel
        Surface-frame bulk model supplying the hoppings to replicate.
    spec : SlabSpec
        Slab provenance carrying the frozen surface cell and copies.
    atom_shifts : Int64[NDArray, "n_atom 3"]
        Exact integer surface-cell shifts of the bulk atoms.

    Returns
    -------
    result : tuple
        Gathered slab hopping amplitudes in eV, slab orbital pairs, and
        exact in-plane slab cells with a zero normal component. The last
        member gives the bulk hopping index behind every slab record.

    Notes
    -----
    Every bulk hopping cell transforms into surface coordinates through
    the exact inverse integer frame. The mapping keeps a record only
    when the target layer exists in the slab. Bonds leaving the finite
    stack therefore drop out, and the retained graph stays open along
    the normal by construction. Amplitudes come from one differentiable
    gather into the bulk amplitudes, so slab hoppings keep bulk
    parameter derivatives.
    """
    coefficients: Int64[NDArray, "3 3"] = _surface_integer_matrix(
        spec.surface_cell
    )
    inverse: Int64[NDArray, "3 3"] = _inverse_integer_matrix(coefficients)
    slab_to_bulk: Tuple[int, ...]
    slab_atom_indices: Tuple[int, ...]
    slab_layers: Tuple[int, ...]
    lookup: Dict[Tuple[int, int], int]
    (
        slab_to_bulk,
        slab_atom_indices,
        slab_layers,
        lookup,
    ) = _orbital_lookup(rotated_bulk, spec)
    pairs: list[Tuple[int, int]] = []
    cells: list[Tuple[int, int, int]] = []
    gather: list[int] = []
    slab_source: int
    bulk_source: int
    pair: Tuple[int, int]
    bulk_cell: Tuple[int, int, int]
    for slab_source, bulk_source in enumerate(slab_to_bulk):
        source_atom: int = rotated_bulk.basis.atom_indices[bulk_source]
        source_layer: int = slab_layers[slab_source]
        source_surface_cell: Int64[NDArray, " 3"] = np.asarray(
            (
                -atom_shifts[source_atom, 0],
                -atom_shifts[source_atom, 1],
                source_layer - atom_shifts[source_atom, 2],
            ),
            dtype=np.int64,
        )
        hopping_index: int
        for hopping_index, (
            pair,
            bulk_cell,
        ) in enumerate(
            zip(
                rotated_bulk.hopping_pairs,
                rotated_bulk.hopping_cells,
                strict=True,
            )
        ):
            if pair[0] != bulk_source:
                continue
            bulk_target: int = pair[1]
            target_atom: int = rotated_bulk.basis.atom_indices[bulk_target]
            transformed_cell: Int64[NDArray, " 3"] = (
                np.asarray(bulk_cell, dtype=np.int64) @ inverse
            )
            target_surface_cell: Int64[NDArray, " 3"] = (
                source_surface_cell + transformed_cell
            )
            target_layer: int = int(
                target_surface_cell[2] + atom_shifts[target_atom, 2]
            )
            slab_target: int | None = lookup.get((bulk_target, target_layer))
            if slab_target is None:
                continue
            slab_cell: Tuple[int, int, int] = (
                int(target_surface_cell[0] + atom_shifts[target_atom, 0]),
                int(target_surface_cell[1] + atom_shifts[target_atom, 1]),
                0,
            )
            pairs.append((slab_source, slab_target))
            cells.append(slab_cell)
            gather.append(hopping_index)
    gather_tuple: Tuple[int, ...] = tuple(gather)
    gather_array: Array = jnp.asarray(gather_tuple, dtype=jnp.int32)
    amplitudes: Complex128[Array, " n_hop_slab"] = (
        rotated_bulk.hopping_amplitudes[gather_array]
    )
    result: Tuple[
        Complex128[Array, " n_hop_slab"],
        Tuple[Tuple[int, int], ...],
        Tuple[Tuple[int, int, int], ...],
        Tuple[int, ...],
    ] = (amplitudes, tuple(pairs), tuple(cells), gather_tuple)
    return result


@jaxtyped(typechecker=beartype)
def _propagate_hoppings(
    rotated_bulk: TBModel,
    spec: SlabSpec,
) -> Tuple[
    Complex128[Array, " n_hop_slab"],
    Tuple[Tuple[int, int], ...],
    Tuple[Tuple[int, int, int], ...],
    Tuple[int, ...],
]:
    """PRIVATE: Propagate hoppings after eagerly selecting representatives.

    Parameters
    ----------
    rotated_bulk : TBModel
        Surface-frame bulk model supplying the hoppings to replicate.
    spec : SlabSpec
        Slab provenance carrying the frozen surface cell and copies.

    Returns
    -------
    result : tuple
        Gathered slab amplitudes in eV, slab orbital pairs, exact
        in-plane slab cells, and the bulk hopping index per record.

    Notes
    -----
    The wrapper derives the integer atom shifts from the concrete
    geometry with ``_base_surface_coordinates`` and then delegates to
    ``_propagate_hoppings_with_shifts``. Rebuilds that already carry
    frozen shifts call the latter directly and skip this host step.
    """
    coefficients: Int64[NDArray, "3 3"] = _surface_integer_matrix(
        spec.surface_cell
    )
    inverse: Int64[NDArray, "3 3"] = _inverse_integer_matrix(coefficients)
    atom_shifts: Int64[NDArray, "n_atom 3"]
    _, atom_shifts = _base_surface_coordinates(
        rotated_bulk.geometry,
        inverse,
    )
    result: Tuple[
        Complex128[Array, " n_hop_slab"],
        Tuple[Tuple[int, int], ...],
        Tuple[Tuple[int, int, int], ...],
        Tuple[int, ...],
    ] = _propagate_hoppings_with_shifts(
        rotated_bulk,
        spec,
        atom_shifts,
    )
    return result


@jaxtyped(typechecker=beartype)
def validate_open_surface_adjacency(model: TBModel) -> None:
    """Reject direct or component-propagated periodic normal images.

    This is a host-side validation diagnostic over the slab's exact static
    hopping graph. It is not a transformed numerical kernel.

    :see: :class:`~.test_slab.TestValidateOpenSurfaceAdjacency`

    Notes
    -----
    The validator first locates every direct nonzero normal-image edge. It
    then searches the unfolded connected component for a top-to-bottom path
    carrying a nonzero accumulated image and reports the exact edge witness.
    """
    offending: Tuple[
        Tuple[int, Tuple[int, int], Tuple[int, int, int]],
        ...,
    ] = tuple(
        (index, pair, cell)
        for index, (pair, cell) in enumerate(
            zip(model.hopping_pairs, model.hopping_cells, strict=True)
        )
        if cell[2] != 0
    )
    adjacency: Dict[
        int,
        list[Tuple[int, int, int]],
    ] = {}
    index: int
    pair: Tuple[int, int]
    cell: Tuple[int, int, int]
    for index, (pair, cell) in enumerate(
        zip(model.hopping_pairs, model.hopping_cells, strict=True)
    ):
        adjacency.setdefault(pair[0], []).append((index, pair[1], cell[2]))
    if offending:
        if model.orbital_positions is None:
            orbital_fractional: Array = model.geometry.positions[
                jnp.asarray(model.basis.atom_indices, dtype=jnp.int32)
            ]
        else:
            orbital_fractional = model.orbital_positions
        orbital_heights: Float64[NDArray, " n_orb"] = np.asarray(
            orbital_fractional @ model.geometry.lattice,
            dtype=np.float64,
        )[:, 2]
        top_height: float = float(np.max(orbital_heights))
        bottom_height: float = float(np.min(orbital_heights))
        top_orbitals: Tuple[int, ...] = tuple(
            int(index)
            for index in np.flatnonzero(
                np.isclose(
                    orbital_heights,
                    top_height,
                    rtol=0.0,
                    atol=1e-10,
                )
            )
        )
        bottom_orbitals: set[int] = {
            int(index)
            for index in np.flatnonzero(
                np.isclose(
                    orbital_heights,
                    bottom_height,
                    rtol=0.0,
                    atol=1e-10,
                )
            )
        }
        maximum_steps: int = max(1, len(model.basis.n) - 1)
        witness: Tuple[int, ...] | None = None
        root: int
        source: int
        normal_image: int
        path: Tuple[int, ...]
        edge_index: int
        target: int
        delta: int
        for root in top_orbitals:
            queue: list[Tuple[int, int, Tuple[int, ...]]] = [(root, 0, ())]
            visited: set[Tuple[int, int]] = {(root, 0)}
            while queue and witness is None:
                source, normal_image, path = queue.pop(0)
                if len(path) >= maximum_steps:
                    continue
                for edge_index, target, delta in adjacency.get(source, ()):
                    target_image: int = normal_image + delta
                    target_path: Tuple[int, ...] = (*path, edge_index)
                    if target in bottom_orbitals and target_image != 0:
                        witness = target_path
                        break
                    state: Tuple[int, int] = (target, target_image)
                    if state not in visited:
                        visited.add(state)
                        queue.append((target, target_image, target_path))
            if witness is not None:
                break
        if witness is None:
            witness = (offending[0][0],)
        index, pair, cell = offending[0]
        message: str = (
            "open slab contains a nonzero normal-image hopping: "
            f"index={index}, pair={pair}, cell={cell}; "
            f"component_path={witness}"
        )
        raise ValueError(message)


def _slab_geometry_and_centres(  # noqa: PLR0913
    rotated_bulk: TBModel,
    surface_cell: SurfaceCell,
    inverse_coefficients: Int64[NDArray, "3 3"],
    atom_shifts: Int64[NDArray, "n_atom 3"],
    bulk_atoms: Tuple[int, ...],
    atom_layers: Tuple[int, ...],
    slab_to_bulk: Tuple[int, ...],
    slab_atom_indices: Tuple[int, ...],
    n_layers: int,
    vacuum_ang: float,
) -> Tuple[
    CrystalGeometry,
    Float64[Array, " n_orb"],
    Float64[Array, "n_orb 3"] | None,
]:
    """PRIVATE: Assemble differentiable slab positions, centres, and depths.

    Parameters
    ----------
    rotated_bulk : TBModel
        Surface-frame bulk model.
    surface_cell : SurfaceCell
        Differentiable surface-frame vectors and spacing.
    inverse_coefficients : Int64[NDArray, "3 3"]
        Exact inverse of the bulk-to-surface integer frame.
    atom_shifts : Int64[NDArray, "n_atom 3"]
        Frozen integer surface-cell shifts of the bulk atoms.
    bulk_atoms : Tuple[int, ...]
        Frozen bulk atom index of every slab atom.
    atom_layers : Tuple[int, ...]
        Frozen layer of every slab atom.
    slab_to_bulk : Tuple[int, ...]
        Bulk orbital behind every slab orbital.
    slab_atom_indices : Tuple[int, ...]
        Slab atom index of every slab orbital.
    n_layers : int
        Frozen number of stacked one-plane layers.
    vacuum_ang : float
        Static vacuum padding in Angstrom.

    Returns
    -------
    result : tuple
        Validated slab geometry, per-orbital depths below the top
        surface in Angstrom, and slab fractional orbital centres or
        ``None`` when the bulk declares no explicit centres.

    Notes
    -----
    Frozen integers only index and shift; every coordinate operation
    runs on traced arrays, so positions, centres, and depths stay
    differentiable within the topology. The assembly lifts the stack so
    its lowest atom sits at zero height. The slab cell closes with
    ``n_layers`` interlayer spacings plus the vacuum, and in-plane
    fractional coordinates wrap modulo one. Explicit bulk orbital
    centres propagate with per-orbital integer shifts; otherwise orbital
    depths inherit their atom heights. Depth runs from the topmost
    centre downward.
    """
    inverse_array: Float64[Array, "3 3"] = jnp.asarray(
        inverse_coefficients,
        dtype=jnp.float64,
    )
    surface_vectors: Float64[Array, "3 3"] = jnp.vstack(
        (
            surface_cell.in_plane_vectors,
            surface_cell.stacking_vector[None, :],
        )
    )
    bulk_atom_array: Array = jnp.asarray(bulk_atoms, dtype=jnp.int32)
    base_surface: Float64[Array, "n_atoms 3"] = (
        rotated_bulk.geometry.positions @ inverse_array
        - jnp.asarray(atom_shifts, dtype=jnp.float64)
    )
    atom_layer_array: Float64[Array, " n_slab_atoms"] = jnp.asarray(
        atom_layers,
        dtype=jnp.float64,
    )
    atom_surface: Float64[Array, "n_slab_atoms 3"] = (
        base_surface[bulk_atom_array].at[:, 2].add(atom_layer_array)
    )
    atom_cart: Float64[Array, "n_slab_atoms 3"] = (
        atom_surface @ surface_vectors
    )
    bottom: Float64[Array, ""] = jnp.min(atom_cart[:, 2])
    atom_cart = atom_cart.at[:, 2].add(-bottom)
    material_period: Float64[Array, ""] = (
        jnp.asarray(float(n_layers), dtype=jnp.float64)
        * surface_cell.interlayer_spacing_ang
    )
    height: Float64[Array, ""] = material_period + jnp.asarray(
        vacuum_ang,
        dtype=jnp.float64,
    )
    slab_lattice: Float64[Array, "3 3"] = jnp.vstack(
        (
            surface_cell.in_plane_vectors,
            jnp.stack(
                (jnp.zeros_like(height), jnp.zeros_like(height), height)
            ),
        )
    )
    inverse_lattice: Float64[Array, "3 3"] = jnp.linalg.inv(slab_lattice)
    atom_fractional: Float64[Array, "n_slab_atoms 3"] = (
        atom_cart @ inverse_lattice
    )
    atom_fractional = atom_fractional.at[:, :2].set(
        jnp.mod(atom_fractional[:, :2], 1.0)
    )
    species_source: Tuple[str, ...] = rotated_bulk.geometry.species
    if not species_source:
        species_source = tuple(
            "X" for _ in range(rotated_bulk.geometry.positions.shape[0])
        )
    slab_species: Tuple[str, ...] = tuple(
        species_source[index] for index in bulk_atoms
    )
    slab_geometry: CrystalGeometry = make_crystal_geometry(
        lattice=slab_lattice,
        positions=atom_fractional,
        species=slab_species,
    )

    orbital_position_array: Float64[Array, "n_orb 3"] | None = None
    if rotated_bulk.orbital_positions is not None:
        bulk_orbital_array: Array = jnp.asarray(
            slab_to_bulk,
            dtype=jnp.int32,
        )
        bulk_centre_surface: Float64[Array, "n_bulk_orb 3"] = (
            rotated_bulk.orbital_positions @ inverse_array
        )
        centre_shifts: Float64[Array, "n_orb 3"] = jnp.asarray(
            [
                (
                    -atom_shifts[
                        rotated_bulk.basis.atom_indices[bulk],
                        0,
                    ],
                    -atom_shifts[
                        rotated_bulk.basis.atom_indices[bulk],
                        1,
                    ],
                    layer
                    - atom_shifts[
                        rotated_bulk.basis.atom_indices[bulk],
                        2,
                    ],
                )
                for bulk, layer in zip(
                    slab_to_bulk,
                    (atom_layers[index] for index in slab_atom_indices),
                    strict=True,
                )
            ],
            dtype=jnp.float64,
        )
        centre_surface: Float64[Array, "n_orb 3"] = (
            bulk_centre_surface[bulk_orbital_array] + centre_shifts
        )
        centre_cart: Float64[Array, "n_orb 3"] = (
            centre_surface @ surface_vectors
        )
        centre_cart = centre_cart.at[:, 2].add(-bottom)
        orbital_position_array = centre_cart @ inverse_lattice
        orbital_position_array = orbital_position_array.at[:, :2].set(
            jnp.mod(orbital_position_array[:, :2], 1.0)
        )
        depth_coordinates: Float64[Array, " n_orb"] = centre_cart[:, 2]
    else:
        slab_atom_index_array: Array = jnp.asarray(
            slab_atom_indices,
            dtype=jnp.int32,
        )
        depth_coordinates = atom_cart[slab_atom_index_array, 2]
    depths: Float64[Array, " n_orb"] = (
        jnp.max(depth_coordinates) - depth_coordinates
    )
    result: Tuple[
        CrystalGeometry,
        Float64[Array, " n_orb"],
        Float64[Array, "n_orb 3"] | None,
    ] = (slab_geometry, depths, orbital_position_array)
    return result


@jaxtyped(typechecker=beartype)
def freeze_slab_topology(  # noqa: PLR0913
    bulk_model: TBModel,
    miller: Tuple[int, int, int],
    thickness_ang: float,
    vacuum_ang: float,
    termination: Tuple[str, str] | None = None,
    fine: Tuple[float, float] = (0.0, 0.0),
) -> SlabTopology:
    """Freeze every discrete choice required to rebuild one slab.

    This host-side setup function is intentionally outside JAX transforms.
    Reuse its result with :func:`rebuild_slab`. Lattice, positions, model
    parameters, orbital centres, and depths can then vary continuously within
    the frozen topology.

    :see: :class:`~.test_slab.TestFreezeSlabTopology`

    Notes
    -----
    Eager code selects integer surface coefficients, atom copies, endpoint
    species, and exact hopping gathers. The types-owned carrier makes those
    choices static for subsequent continuous reconstruction.
    """
    if not math.isfinite(thickness_ang) or thickness_ang < 0.0:
        message: str = "thickness_ang must be finite and nonnegative"
        raise ValueError(message)
    if not math.isfinite(vacuum_ang) or vacuum_ang < 0.0:
        message = "vacuum_ang must be finite and nonnegative"
        raise ValueError(message)
    if (
        type(fine) is not tuple
        or len(fine) != 2  # noqa: PLR2004
        or any(not math.isfinite(value) for value in fine)
    ):
        message = "fine must contain two finite floats"
        raise ValueError(message)
    if termination is not None and (
        type(termination) is not tuple
        or len(termination) != 2  # noqa: PLR2004
        or any(type(species) is not str for species in termination)
    ):
        message = "termination must be None or a pair of species strings"
        raise ValueError(message)

    surface_cell: SurfaceCell = find_surface_cell(
        bulk_model.geometry,
        miller,
    )
    coefficients: Int64[NDArray, "3 3"] = _surface_integer_matrix(surface_cell)
    inverse: Int64[NDArray, "3 3"] = _inverse_integer_matrix(coefficients)
    base_coordinates: Float64[NDArray, "n_atom 3"]
    atom_shifts: Int64[NDArray, "n_atom 3"]
    base_coordinates, atom_shifts = _base_surface_coordinates(
        bulk_model.geometry,
        inverse,
    )
    surface_vectors_snapshot: Float64[NDArray, "3 3"] = np.vstack(
        (
            np.asarray(surface_cell.in_plane_vectors),
            np.asarray(surface_cell.stacking_vector)[None, :],
        )
    )
    spacing_snapshot: float = float(surface_cell.interlayer_spacing_ang)
    n_layers: int = (
        int(math.ceil(thickness_ang / spacing_snapshot - 1e-12)) + 1
    )
    bulk_atoms: Tuple[int, ...]
    atom_layers: Tuple[int, ...]
    resolved_termination: Tuple[str, str]
    if termination is None:
        (
            bulk_atoms,
            atom_layers,
            resolved_termination,
        ) = _natural_atom_copies(
            bulk_model.geometry,
            base_coordinates,
            surface_vectors_snapshot,
            n_layers,
            thickness_ang,
            fine,
        )
        n_layers = max(atom_layers) + 1
    else:
        bulk_atoms, atom_layers = _terminated_atom_copies(
            bulk_model.geometry,
            base_coordinates,
            surface_vectors_snapshot,
            n_layers,
            thickness_ang,
            termination,
            fine,
        )
        n_layers = max(atom_layers) + 1
        resolved_termination = termination
    topology: SlabTopology = SlabTopology(
        miller=surface_cell.miller,
        in_plane_coeffs=surface_cell.in_plane_coeffs,
        stacking_coeffs=surface_cell.stacking_coeffs,
        atom_shifts=tuple(
            tuple(int(value) for value in row) for row in atom_shifts
        ),
        bulk_atom_of_slab_atom=bulk_atoms,
        layer_of_slab_atom=atom_layers,
        termination=resolved_termination,
        thickness_ang=thickness_ang,
        vacuum_ang=vacuum_ang,
        fine=fine,
        n_layers=n_layers,
        bulk_atom_count=bulk_model.geometry.positions.shape[0],
        basis_atom_indices=bulk_model.basis.atom_indices,
    )
    return topology


@jaxtyped(typechecker=beartype)
def rebuild_slab(
    bulk_model: TBModel,
    topology: SlabTopology,
) -> Tuple[TBModel, SlabSpec]:
    """Construct a slab from frozen topology using only JAX geometry.

    The function contains no host conversion of a traced model leaf. It is
    therefore the differentiable slab-rebuild seam. Call
    :func:`freeze_slab_topology` once at a representative geometry, then pass
    compatible continuously perturbed models here.

    :see: :class:`~.test_slab.TestRebuildSlab`

    Notes
    -----
    Static topology indices gather the rotated bulk geometry, onsite terms,
    SOC shells, and exact hopping records. All numerical leaves remain JAX
    arrays, so differentiation never re-enters discrete topology selection.
    """
    if bulk_model.geometry.positions.shape[0] != topology.bulk_atom_count:
        message: str = "bulk atom count does not match frozen slab topology"
        raise ValueError(message)
    if bulk_model.basis.atom_indices != topology.basis_atom_indices:
        message = "basis atom mapping does not match frozen slab topology"
        raise ValueError(message)
    surface_cell: SurfaceCell = _assemble_surface_cell(
        geometry=bulk_model.geometry,
        miller=topology.miller,
        in_plane_coeffs=topology.in_plane_coeffs,
        stacking_coeffs=topology.stacking_coeffs,
    )
    rotated_bulk: TBModel = rotate_tb_model(
        bulk_model,
        surface_cell.rotation,
    )
    coefficients: Int64[NDArray, "3 3"] = _surface_integer_matrix(surface_cell)
    inverse: Int64[NDArray, "3 3"] = _inverse_integer_matrix(coefficients)
    atom_shifts: Int64[NDArray, "n_atom 3"] = np.asarray(
        topology.atom_shifts,
        dtype=np.int64,
    )
    spec: SlabSpec = make_slab_spec(
        surface_cell=surface_cell,
        geometry=rotated_bulk.geometry,
        thickness_ang=topology.thickness_ang,
        vacuum_ang=topology.vacuum_ang,
        fine=topology.fine,
        termination=topology.termination,
        n_layers=topology.n_layers,
        bulk_atom_of_slab_atom=topology.bulk_atom_of_slab_atom,
        layer_of_slab_atom=topology.layer_of_slab_atom,
    )
    slab_to_bulk: Tuple[int, ...]
    slab_atom_indices: Tuple[int, ...]
    slab_layers: Tuple[int, ...]
    slab_to_bulk, slab_atom_indices, slab_layers = _orbital_copy_metadata(
        rotated_bulk.basis,
        topology.bulk_atom_of_slab_atom,
        topology.layer_of_slab_atom,
    )
    slab_basis: OrbitalBasis = _slab_basis(
        rotated_bulk.basis,
        slab_to_bulk,
        slab_atom_indices,
        slab_layers,
    )
    slab_shells: Tuple[int, ...]
    shell_gather: Tuple[int, ...]
    slab_shells, shell_gather = _slab_shell_metadata(
        rotated_bulk,
        slab_to_bulk,
        slab_atom_indices,
    )
    slab_geometry: CrystalGeometry
    depths: Float64[Array, " n_orb"]
    orbital_positions: Float64[Array, "n_orb 3"] | None
    slab_geometry, depths, orbital_positions = _slab_geometry_and_centres(
        rotated_bulk,
        surface_cell,
        inverse,
        atom_shifts,
        topology.bulk_atom_of_slab_atom,
        topology.layer_of_slab_atom,
        slab_to_bulk,
        slab_atom_indices,
        topology.n_layers,
        topology.vacuum_ang,
    )
    amplitudes: Complex128[Array, " n_hop_slab"]
    hopping_pairs: Tuple[Tuple[int, int], ...]
    hopping_cells: Tuple[Tuple[int, int, int], ...]
    _gather: Tuple[int, ...]
    (
        amplitudes,
        hopping_pairs,
        hopping_cells,
        _gather,
    ) = _propagate_hoppings_with_shifts(
        rotated_bulk,
        spec,
        atom_shifts,
    )
    slab_to_bulk_array: Array = jnp.asarray(
        slab_to_bulk,
        dtype=jnp.int32,
    )
    shell_gather_array: Array = jnp.asarray(
        shell_gather,
        dtype=jnp.int32,
    )
    slab_model: TBModel = make_tb_model(
        hopping_amplitudes=amplitudes,
        onsite_energies=rotated_bulk.onsite_energies[slab_to_bulk_array],
        soc_lambdas=rotated_bulk.soc_lambdas[shell_gather_array],
        geometry=slab_geometry,
        basis=slab_basis,
        hopping_pairs=hopping_pairs,
        hopping_cells=hopping_cells,
        shell_index=slab_shells,
        spinor=rotated_bulk.spinor,
        orbital_positions=orbital_positions,
        depths=depths,
    )
    validate_open_surface_adjacency(slab_model)
    result: Tuple[TBModel, SlabSpec] = (slab_model, spec)
    return result


@jaxtyped(typechecker=beartype)
def gen_slab(  # noqa: DOC105, DOC502, PLR0913, PLR0915
    bulk_model: TBModel,
    miller: Tuple[int, int, int],
    thickness_ang: float,
    vacuum_ang: float,
    termination: Tuple[str, str] | None = None,
    fine: Tuple[float, float] = (0.0, 0.0),
) -> Tuple[TBModel, SlabSpec]:
    """Construct a finite Miller-index slab with exact open-normal topology.

    Eager input selects discrete atoms, orbitals, and hoppings. Every assembled
    geometry, centre, depth, amplitude, onsite, and SOC value remains a JAX
    array. Re-run the factory after a continuous perturbation crosses a layer,
    termination, or metric-reduction boundary.

    Parameters
    ----------
    bulk_model : TBModel
        Validated bulk tight-binding model.
    miller : Tuple[int, int, int]
        Primitive Miller indices (**static**).
    thickness_ang : float
        Nonnegative minimum material span in Angstrom (**static**). Zero
        requests the one-plane limiting slab.
    vacuum_ang : float
        Nonnegative vacuum padding in Angstrom (**static**).
    termination : Tuple[str, str] or None, optional
        Requested ``(top, bottom)`` species. ``None`` retains a natural
        complete stack. Default is ``None``.
    fine : Tuple[float, float], optional
        Static ``(top, bottom)`` inward cut shifts in Angstrom.

    Returns
    -------
    result : Tuple[TBModel, SlabSpec]
        Slab model and its immutable construction provenance.

    Raises
    ------
    ValueError
        If geometry choices fail, no requested termination exists, or the
        propagated graph retains a normal image.

    Examples
    --------
    Build a nearest-neighbour graphene model and cut a finite zigzag ribbon
    normal to the first reciprocal-lattice vector. The third bulk vector
    provides a noninteracting embedding direction. Exact cells certify slab
    openness independently of its length.

    >>> import jax.numpy as jnp
    >>> from diffpes.types import (
    ...     make_crystal_geometry,
    ...     make_orbital_basis,
    ...     make_tb_model,
    ... )
    >>> bond = 1.42
    >>> root_three = jnp.sqrt(3.0)
    >>> geometry = make_crystal_geometry(
    ...     lattice=jnp.asarray(
    ...         (
    ...             (root_three * bond, 0.0, 0.0),
    ...             (root_three * bond / 2.0, 1.5 * bond, 0.0),
    ...             (0.0, 0.0, 20.0),
    ...         )
    ...     ),
    ...     positions=jnp.asarray(((0.0, 0.0, 0.0), (1 / 3, 1 / 3, 0.0))),
    ...     species=("C", "C"),
    ... )
    >>> basis = make_orbital_basis(
    ...     atom_indices=(0, 1),
    ...     n=(2, 2),
    ...     l=(1, 1),
    ...     m=(0, 0),
    ...     labels=("pz_A", "pz_B"),
    ... )
    >>> graphene = make_tb_model(
    ...     hopping_amplitudes=-2.7 * jnp.ones(6, dtype=jnp.complex128),
    ...     onsite_energies=jnp.zeros(2),
    ...     soc_lambdas=jnp.zeros(0),
    ...     geometry=geometry,
    ...     basis=basis,
    ...     hopping_pairs=((0, 1), (0, 1), (0, 1), (1, 0), (1, 0), (1, 0)),
    ...     hopping_cells=(
    ...         (0, 0, 0),
    ...         (-1, 0, 0),
    ...         (0, -1, 0),
    ...         (0, 0, 0),
    ...         (1, 0, 0),
    ...         (0, 1, 0),
    ...     ),
    ...     shell_index=(-1, -1),
    ... )
    >>> ribbon, ribbon_spec = gen_slab(
    ...     graphene,
    ...     miller=(1, 0, 0),
    ...     thickness_ang=15.0,
    ...     vacuum_ang=12.0,
    ... )
    >>> ribbon_spec.n_layers > 1
    True
    >>> all(cell[2] == 0 for cell in ribbon.hopping_cells)
    True

    Notes
    -----
    This convenience factory performs host-only topology selection before
    calling :func:`rebuild_slab`; it is not itself a ``jit``/``grad``/``vmap``
    target. For transformed calculations, call :func:`freeze_slab_topology`
    eagerly once and transform :func:`rebuild_slab`.

    :see: :class:`~.test_slab.TestGenSlab`
    """
    topology: SlabTopology = freeze_slab_topology(
        bulk_model=bulk_model,
        miller=miller,
        thickness_ang=thickness_ang,
        vacuum_ang=vacuum_ang,
        termination=termination,
        fine=fine,
    )
    result: Tuple[TBModel, SlabSpec] = rebuild_slab(bulk_model, topology)
    return result


def _slab_operator_centres(
    bulk_model: TBModel,
    operator_data: WannierOperatorData,
    spec: SlabSpec,
) -> Float64[Array, "n_orb_slab 3"]:
    """PRIVATE: Convert explicit Wannier centres into the slab cell.

    Parameters
    ----------
    bulk_model : TBModel
        Bulk model whose basis orders the centres.
    operator_data : WannierOperatorData
        Bulk Wannier centres and optional position matrices.
    spec : SlabSpec
        Slab provenance carrying the frozen surface cell and copies.

    Returns
    -------
    shifted_centres : Float64[Array, "n_orb_slab 3"]
        Cartesian slab centres in angstroms, lifted by the same bottom
        offset as the slab atoms.

    Raises
    ------
    ValueError
        If a nonidentity rotation meets matrix-free data whose shells do
        not share one centre.

    Notes
    -----
    Matrix-free (hr) data rotates the centres directly; that is
    covariant only when every shell shares one centre, which the guard
    enforces. Full (tb) data instead conjugates the position matrices
    with the orbital representation and rotates their Cartesian
    component. The real zero-cell diagonal then supplies the rotated
    centres. The frozen integer shifts and layers translate every
    orbital copy into place through the surface vectors.
    """
    coefficients: Int64[NDArray, "3 3"] = _surface_integer_matrix(
        spec.surface_cell
    )
    inverse: Int64[NDArray, "3 3"] = _inverse_integer_matrix(coefficients)
    atom_shifts: Int64[NDArray, "n_atom 3"]
    _, atom_shifts = _base_surface_coordinates(
        bulk_model.geometry,
        inverse,
    )
    slab_to_bulk: Tuple[int, ...]
    slab_layers: Tuple[int, ...]
    slab_to_bulk, _, slab_layers, _ = _orbital_lookup(bulk_model, spec)
    inverse_array: Float64[Array, "3 3"] = jnp.asarray(
        inverse,
        dtype=jnp.float64,
    )
    surface_vectors: Float64[Array, "3 3"] = jnp.vstack(
        (
            spec.surface_cell.in_plane_vectors,
            spec.surface_cell.stacking_vector[None, :],
        )
    )
    identity_rotation: bool = bool(
        np.allclose(
            np.asarray(spec.surface_cell.rotation),
            np.eye(3),
            rtol=0.0,
            atol=1e-12,
        )
    )
    if operator_data.position_matrices is None:
        if not identity_rotation:
            indices: list[int]
            for indices in _shell_groups(bulk_model).values():
                shell_centres: Float64[NDArray, "n_shell 3"] = np.asarray(
                    operator_data.centres_cart
                )[indices]
                if not np.allclose(
                    shell_centres,
                    shell_centres[:1],
                    rtol=0.0,
                    atol=1e-10,
                ):
                    message: str = (
                        "matrix-free operator data requires one shared centre "
                        "per shell under nonidentity rotation"
                    )
                    raise ValueError(message)
        bulk_centres_rotated: Float64[Array, "n_orb 3"] = (
            operator_data.centres_cart @ spec.surface_cell.rotation.T
        )
    else:
        representation: Complex128[Array, "n_orb n_orb"] = (
            jnp.eye(len(bulk_model.basis.n), dtype=jnp.complex128)
            if identity_rotation
            else _orbital_rotation(
                bulk_model,
                spec.surface_cell.rotation,
            )
        )
        orbital_rotated: Complex128[Array, "n_R n_orb n_orb 3"] = jnp.einsum(
            "ai,rijc,bj->rabc",
            representation,
            operator_data.position_matrices,
            representation.conj(),
        )
        rotated_blocks: Complex128[Array, "n_R n_orb n_orb 3"] = jnp.einsum(
            "rijc,ac->rija",
            orbital_rotated,
            spec.surface_cell.rotation,
        )
        zero_index: int = operator_data.cells.index((0, 0, 0))
        diagonal: Array = jnp.arange(
            len(bulk_model.basis.n),
            dtype=jnp.int32,
        )
        bulk_centres_rotated = jnp.real(
            rotated_blocks[zero_index, diagonal, diagonal]
        )
    translations: Float64[Array, "n_orb_slab 3"] = jnp.asarray(
        [
            (
                -atom_shifts[bulk_model.basis.atom_indices[bulk], 0],
                -atom_shifts[bulk_model.basis.atom_indices[bulk], 1],
                layer - atom_shifts[bulk_model.basis.atom_indices[bulk], 2],
            )
            for bulk, layer in zip(
                slab_to_bulk,
                slab_layers,
                strict=True,
            )
        ],
        dtype=jnp.float64,
    )
    bulk_indices: Array = jnp.asarray(slab_to_bulk, dtype=jnp.int32)
    centres_cart: Float64[Array, "n_orb_slab 3"] = (
        bulk_centres_rotated[bulk_indices] + translations @ surface_vectors
    )

    base_atoms: Float64[Array, "n_atoms 3"] = (
        bulk_model.geometry.positions @ inverse_array
        - jnp.asarray(atom_shifts, dtype=jnp.float64)
    )
    slab_atom_surface: Float64[Array, "n_slab_atoms 3"] = base_atoms[
        jnp.asarray(spec.bulk_atom_of_slab_atom, dtype=jnp.int32)
    ]
    slab_atom_surface = slab_atom_surface.at[:, 2].add(
        jnp.asarray(spec.layer_of_slab_atom, dtype=jnp.float64)
    )
    bottom: Float64[Array, ""] = jnp.min(
        (slab_atom_surface @ surface_vectors)[:, 2]
    )
    shifted_centres: Float64[Array, "n_orb_slab 3"] = centres_cart.at[
        :, 2
    ].add(-bottom)
    return shifted_centres


def _propagated_operator_cells(
    bulk_model: TBModel,
    operator_data: WannierOperatorData,
    spec: SlabSpec,
) -> Tuple[Tuple[int, int, int], ...]:
    """PRIVATE: Compute an operator cell grid for the exact slab topology.

    Parameters
    ----------
    bulk_model : TBModel
        Bulk model providing atom and orbital metadata.
    operator_data : WannierOperatorData
        Bulk operator data supplying the cell grid to propagate.
    spec : SlabSpec
        Slab provenance carrying the frozen surface cell and copies.

    Returns
    -------
    result : Tuple[Tuple[int, int, int], ...]
        Sorted in-plane slab cells, all with a zero normal component,
        reachable from any retained orbital copy.

    Notes
    -----
    The walk repeats the exact hopping-propagation bookkeeping on the
    operator cell grid. Every bulk cell transforms through the inverse
    integer frame from every slab source orbital. The walk records a
    slab cell whenever the target orbital copy exists. Matrix-free
    sidecars use this grid because they carry no per-cell matrices to
    scatter.
    """
    coefficients: Int64[NDArray, "3 3"] = _surface_integer_matrix(
        spec.surface_cell
    )
    inverse: Int64[NDArray, "3 3"] = _inverse_integer_matrix(coefficients)
    atom_shifts: Int64[NDArray, "n_atom 3"]
    _, atom_shifts = _base_surface_coordinates(
        bulk_model.geometry,
        inverse,
    )
    slab_to_bulk: Tuple[int, ...]
    slab_layers: Tuple[int, ...]
    lookup: Dict[Tuple[int, int], int]
    slab_to_bulk, _, slab_layers, lookup = _orbital_lookup(
        bulk_model,
        spec,
    )
    cells: set[Tuple[int, int, int]] = set()
    slab_source: int
    bulk_source: int
    for slab_source, bulk_source in enumerate(slab_to_bulk):
        source_atom: int = bulk_model.basis.atom_indices[bulk_source]
        source_layer: int = slab_layers[slab_source]
        source_surface_cell: Int64[NDArray, " 3"] = np.asarray(
            (
                -atom_shifts[source_atom, 0],
                -atom_shifts[source_atom, 1],
                source_layer - atom_shifts[source_atom, 2],
            ),
            dtype=np.int64,
        )
        bulk_cell: Tuple[int, int, int]
        for bulk_cell in operator_data.cells:
            transformed_cell: Int64[NDArray, " 3"] = (
                np.asarray(bulk_cell, dtype=np.int64) @ inverse
            )
            target_surface_cell: Int64[NDArray, " 3"] = (
                source_surface_cell + transformed_cell
            )
            bulk_target: int
            for bulk_target in range(len(bulk_model.basis.n)):
                target_atom: int = bulk_model.basis.atom_indices[bulk_target]
                target_layer: int = int(
                    target_surface_cell[2] + atom_shifts[target_atom, 2]
                )
                if (bulk_target, target_layer) not in lookup:
                    continue
                cells.add(
                    (
                        int(
                            target_surface_cell[0]
                            + atom_shifts[target_atom, 0]
                        ),
                        int(
                            target_surface_cell[1]
                            + atom_shifts[target_atom, 1]
                        ),
                        0,
                    )
                )
    result: Tuple[Tuple[int, int, int], ...] = tuple(sorted(cells))
    return result


def _propagate_position_matrices(  # noqa: PLR0915
    bulk_model: TBModel,
    operator_data: WannierOperatorData,
    spec: SlabSpec,
    slab_centres_cart: Float64[Array, "n_orb_slab 3"],
) -> Tuple[
    Complex128[Array, "n_R n_orb n_orb 3"],
    Tuple[Tuple[int, int, int], ...],
]:
    """PRIVATE: Propagate real-space position matrices with exact bookkeeping.

    Parameters
    ----------
    bulk_model : TBModel
        Bulk model providing atom and orbital metadata.
    operator_data : WannierOperatorData
        Bulk operator data; must carry position matrices.
    spec : SlabSpec
        Slab provenance carrying the frozen surface cell and copies.
    slab_centres_cart : Float64[Array, "n_orb_slab 3"]
        Cartesian slab centres in angstroms for the diagonal reset.

    Returns
    -------
    result : tuple
        Dense slab position matrices per exact in-plane cell, in
        angstroms, and the sorted cell tuple.

    Raises
    ------
    ValueError
        If the sidecar has no position matrices, or a nonidentity
        rotation meets an incomplete shell.

    Notes
    -----
    The orbital representation conjugates the bulk matrices and the
    surface rotation rotates their Cartesian component. The exact
    hopping-propagation bookkeeping then scatters every surviving
    element into its slab cell. A final pass retranslates the zero-cell
    diagonal so its real part equals the supplied slab centres. This
    matches the shift and bottom lift of the slab geometry.
    """
    if operator_data.position_matrices is None:
        message: str = "position-matrix propagation requires tb data"
        raise ValueError(message)
    missing_shells: Dict[
        Tuple[int, int, int, int],
        Tuple[int, ...],
    ] = _missing_magnetic_numbers(bulk_model)
    identity_rotation: bool = bool(
        np.allclose(
            np.asarray(spec.surface_cell.rotation),
            np.eye(3),
            rtol=0.0,
            atol=1e-12,
        )
    )
    if missing_shells and not identity_rotation:
        message = (
            "operator propagation requires complete shells for a "
            "nonidentity rotation"
        )
        raise ValueError(message)
    representation: Complex128[Array, "n_orb n_orb"] = (
        jnp.eye(len(bulk_model.basis.n), dtype=jnp.complex128)
        if identity_rotation
        else _orbital_rotation(
            bulk_model,
            spec.surface_cell.rotation,
        )
    )
    orbital_rotated: Complex128[Array, "n_R n_orb n_orb 3"] = jnp.einsum(
        "ai,rijc,bj->rabc",
        representation,
        operator_data.position_matrices,
        representation.conj(),
    )
    rotated_position_matrices: Complex128[
        Array,
        "n_R n_orb n_orb 3",
    ] = jnp.einsum(
        "rijc,ac->rija",
        orbital_rotated,
        spec.surface_cell.rotation,
    )
    coefficients: Int64[NDArray, "3 3"] = _surface_integer_matrix(
        spec.surface_cell
    )
    inverse: Int64[NDArray, "3 3"] = _inverse_integer_matrix(coefficients)
    atom_shifts: Int64[NDArray, "n_atom 3"]
    _, atom_shifts = _base_surface_coordinates(
        bulk_model.geometry,
        inverse,
    )
    slab_to_bulk: Tuple[int, ...]
    slab_layers: Tuple[int, ...]
    lookup: Dict[Tuple[int, int], int]
    slab_to_bulk, _, slab_layers, lookup = _orbital_lookup(
        bulk_model,
        spec,
    )
    emitted: list[
        Tuple[
            Tuple[int, int, int],
            int,
            int,
            int,
            int,
            int,
        ]
    ] = []
    slab_source: int
    bulk_source: int
    for slab_source, bulk_source in enumerate(slab_to_bulk):
        source_atom: int = bulk_model.basis.atom_indices[bulk_source]
        source_layer: int = slab_layers[slab_source]
        source_surface_cell: Int64[NDArray, " 3"] = np.asarray(
            (
                -atom_shifts[source_atom, 0],
                -atom_shifts[source_atom, 1],
                source_layer - atom_shifts[source_atom, 2],
            ),
            dtype=np.int64,
        )
        cell_index: int
        bulk_cell: Tuple[int, int, int]
        for cell_index, bulk_cell in enumerate(operator_data.cells):
            transformed_cell: Int64[NDArray, " 3"] = (
                np.asarray(bulk_cell, dtype=np.int64) @ inverse
            )
            target_surface_cell: Int64[NDArray, " 3"] = (
                source_surface_cell + transformed_cell
            )
            bulk_target: int
            for bulk_target in range(len(bulk_model.basis.n)):
                target_atom: int = bulk_model.basis.atom_indices[bulk_target]
                target_layer: int = int(
                    target_surface_cell[2] + atom_shifts[target_atom, 2]
                )
                slab_target: int | None = lookup.get(
                    (bulk_target, target_layer)
                )
                if slab_target is None:
                    continue
                slab_cell: Tuple[int, int, int] = (
                    int(target_surface_cell[0] + atom_shifts[target_atom, 0]),
                    int(target_surface_cell[1] + atom_shifts[target_atom, 1]),
                    0,
                )
                emitted.append(
                    (
                        slab_cell,
                        slab_source,
                        slab_target,
                        cell_index,
                        bulk_source,
                        bulk_target,
                    )
                )
    cells: Tuple[Tuple[int, int, int], ...] = tuple(
        sorted({record[0] for record in emitted})
    )
    cell_lookup: Dict[Tuple[int, int, int], int] = {
        cell: index for index, cell in enumerate(cells)
    }
    n_slab_orbitals: int = len(slab_to_bulk)
    matrices: Complex128[Array, "n_R n_orb n_orb 3"] = jnp.zeros(
        (len(cells), n_slab_orbitals, n_slab_orbitals, 3),
        dtype=jnp.complex128,
    )
    record: Tuple[
        Tuple[int, int, int],
        int,
        int,
        int,
        int,
        int,
    ]
    for record in emitted:
        (
            slab_cell,
            slab_source,
            slab_target,
            cell_index,
            bulk_source,
            bulk_target,
        ) = record
        vector: Complex128[Array, " 3"] = rotated_position_matrices[
            cell_index,
            bulk_source,
            bulk_target,
        ]
        matrices = matrices.at[
            cell_lookup[slab_cell],
            slab_source,
            slab_target,
        ].add(vector)
    zero_cell_index: int | None = cell_lookup.get((0, 0, 0))
    if zero_cell_index is not None:
        bulk_zero_index: int = operator_data.cells.index((0, 0, 0))
        diagonal: Array = jnp.arange(n_slab_orbitals, dtype=jnp.int32)
        current_diagonal: Complex128[Array, "n_orb 3"] = matrices[
            zero_cell_index,
            diagonal,
            diagonal,
        ]
        bulk_indices: Array = jnp.asarray(
            slab_to_bulk,
            dtype=jnp.int32,
        )
        original_diagonal: Complex128[Array, "n_orb 3"] = (
            rotated_position_matrices[
                bulk_zero_index,
                bulk_indices,
                bulk_indices,
            ]
        )
        translation: Complex128[Array, "n_orb 3"] = (
            slab_centres_cart.astype(jnp.complex128) - original_diagonal
        )
        matrices = matrices.at[
            zero_cell_index,
            diagonal,
            diagonal,
        ].set(current_diagonal + translation)
    result: Tuple[
        Complex128[Array, "n_R n_orb n_orb 3"],
        Tuple[Tuple[int, int, int], ...],
    ] = (matrices, cells)
    return result


@jaxtyped(typechecker=beartype)
def gen_slab_with_operators(  # noqa: DOC105, PLR0913
    bulk_model: TBModel,
    operator_data: WannierOperatorData,
    miller: Tuple[int, int, int],
    thickness_ang: float,
    vacuum_ang: float,
    termination: Tuple[str, str] | None = None,
    fine: Tuple[float, float] = (0.0, 0.0),
) -> Tuple[TBModel, SlabSpec, WannierOperatorData]:
    """Construct a slab while preserving its Wannier operator sidecar.

    The Hamiltonian and operator paths share one frozen atom, orbital, layer,
    and exact-cell mapping. The surface rotation acts on Cartesian centres
    and vector-operator components. The function preserves every populated
    position matrix.

    Parameters
    ----------
    bulk_model : TBModel
        Validated bulk tight-binding model.
    operator_data : WannierOperatorData
        Paired Wannier centres and optional real-space position matrices.
    miller : Tuple[int, int, int]
        Primitive Miller indices.
    thickness_ang : float
        Nonnegative minimum post-fine material span in Angstrom.
    vacuum_ang : float
        Nonnegative vacuum padding in Angstrom.
    termination : Tuple[str, str] or None, optional
        Requested top and bottom species, or ``None`` for a natural cut.
    fine : Tuple[float, float], optional
        Static top and bottom inward cut shifts in Angstrom.

    Returns
    -------
    result : Tuple[TBModel, SlabSpec, WannierOperatorData]
        Slab model, construction provenance, and propagated operator data.

    Raises
    ------
    ValueError
        If the model and sidecar have incompatible orbital or cell grids, the
        position sidecar lacks its zero cell, or slab construction fails.

    Notes
    -----
    This is a host-only convenience factory because it selects slab topology
    and exact operator-cell bookkeeping. Transform continuous slab rebuilds
    through :func:`rebuild_slab`; regenerate this sidecar after a topology
    change.

    :see: :class:`~.test_slab.TestGenSlabWithOperators`
    """
    if operator_data.centres_cart.shape[0] != len(bulk_model.basis.n):
        message: str = (
            "operator_data centres must match the bulk orbital count"
        )
        raise ValueError(message)
    missing_cells: set[Tuple[int, int, int]] = set(
        bulk_model.hopping_cells
    ) - set(operator_data.cells)
    if missing_cells:
        message = (
            "operator_data cells must cover the bulk Hamiltonian cell grid; "
            f"missing={tuple(sorted(missing_cells))}"
        )
        raise ValueError(message)
    if (
        operator_data.position_matrices is not None
        and (0, 0, 0) not in operator_data.cells
    ):
        message = "tb operator data must contain the zero cell"
        raise ValueError(message)
    slab_model: TBModel
    spec: SlabSpec
    slab_model, spec = gen_slab(
        bulk_model,
        miller,
        thickness_ang,
        vacuum_ang,
        termination,
        fine,
    )
    centres_cart: Float64[Array, "n_orb_slab 3"] = _slab_operator_centres(
        bulk_model,
        operator_data,
        spec,
    )
    centre_fractional: Float64[Array, "n_orb_slab 3"] = (
        centres_cart @ jnp.linalg.inv(slab_model.geometry.lattice)
    )
    centre_fractional = centre_fractional.at[:, :2].set(
        jnp.mod(centre_fractional[:, :2], 1.0)
    )
    operator_depths: Float64[Array, " n_orb_slab"] = (
        jnp.max(centres_cart[:, 2]) - centres_cart[:, 2]
    )
    slab_model = make_tb_model(
        hopping_amplitudes=slab_model.hopping_amplitudes,
        onsite_energies=slab_model.onsite_energies,
        soc_lambdas=slab_model.soc_lambdas,
        geometry=slab_model.geometry,
        basis=slab_model.basis,
        hopping_pairs=slab_model.hopping_pairs,
        hopping_cells=slab_model.hopping_cells,
        shell_index=slab_model.shell_index,
        spinor=slab_model.spinor,
        orbital_positions=centre_fractional,
        depths=operator_depths,
    )
    position_matrices: Complex128[Array, "n_R n_orb n_orb 3"] | None
    cells: Tuple[Tuple[int, int, int], ...]
    source_format: str
    if operator_data.position_matrices is None:
        position_matrices = None
        cells = _propagated_operator_cells(
            bulk_model,
            operator_data,
            spec,
        )
        source_format = "hr"
    else:
        position_matrices, cells = _propagate_position_matrices(
            bulk_model,
            operator_data,
            spec,
            centres_cart,
        )
        source_format = "tb"
    propagated: WannierOperatorData = make_wannier_operator_data(
        position_matrices=position_matrices,
        centres_cart=centres_cart,
        cells=cells,
        degeneracies=(1,) * len(cells),
        spin_layout=operator_data.spin_layout,
        source_format=source_format,
    )
    result: Tuple[TBModel, SlabSpec, WannierOperatorData] = (
        slab_model,
        spec,
        propagated,
    )
    return result


__all__: list[str] = [
    "find_surface_cell",
    "freeze_slab_topology",
    "gen_slab",
    "gen_slab_with_operators",
    "rebuild_slab",
    "rotate_tb_model",
    "validate_open_surface_adjacency",
]
