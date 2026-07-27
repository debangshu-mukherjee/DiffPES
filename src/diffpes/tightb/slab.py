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
from jaxtyping import Array, Complex, Float, jaxtyped

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


def _extended_gcd(first: int, second: int) -> tuple[int, int, int]:
    """Return nonnegative gcd and signed Bezout coefficients."""
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
    result: tuple[int, int, int] = (
        old_remainder,
        signed_first,
        signed_second,
    )
    return result


def _determinant_3x3(matrix: np.ndarray) -> int:
    """Evaluate a three-dimensional integer determinant exactly."""
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
    miller: tuple[int, int, int],
) -> tuple[np.ndarray, np.ndarray]:
    """Construct an oriented integer kernel basis and unit-advance vector."""
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
        kernel: np.ndarray = np.asarray(
            ((1, 0, 0), (0, 1, 0)),
            dtype=np.int64,
        )
        stacking: np.ndarray = np.asarray(
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
    frame: np.ndarray = np.vstack((kernel, stacking))
    determinant: int = _determinant_3x3(frame)
    if abs(determinant) != 1:
        message = "surface integer frame must be unimodular"
        raise ValueError(message)
    if determinant < 0:
        kernel[1] *= -1
    result: tuple[np.ndarray, np.ndarray] = (kernel, stacking)
    return result


def _gauss_reduce(
    kernel: np.ndarray,
    lattice: np.ndarray,
) -> np.ndarray:
    """Compute an exact reduced basis in the Cartesian lattice metric."""
    reduced: np.ndarray = kernel.copy()
    for _ in range(256):
        vectors: np.ndarray = reduced @ lattice
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
    stacking: np.ndarray,
    kernel: np.ndarray,
    lattice: np.ndarray,
) -> np.ndarray:
    """Compute the closest unit-advance vector to the plane normal."""
    in_plane: np.ndarray = kernel @ lattice
    seed: np.ndarray = stacking @ lattice
    gram: np.ndarray = in_plane @ in_plane.T
    continuous: np.ndarray = np.linalg.solve(gram, -(in_plane @ seed))
    rounded: np.ndarray = np.rint(continuous).astype(np.int64)

    def perpendicular_norm_squared(coefficients: np.ndarray) -> float:
        candidate: np.ndarray = seed + coefficients @ in_plane
        norm_squared: float = float(candidate @ candidate)
        return norm_squared

    best_coefficients: np.ndarray = rounded
    best_norm: float = perpendicular_norm_squared(best_coefficients)
    smallest_singular: float = float(
        np.linalg.svd(in_plane, compute_uv=False)[-1]
    )
    radius: float = math.sqrt(best_norm) / smallest_singular + 1.0
    lower: np.ndarray = np.floor(continuous - radius).astype(np.int64)
    upper: np.ndarray = np.ceil(continuous + radius).astype(np.int64)
    first_coefficient: int
    second_coefficient: int
    coefficients: np.ndarray
    for first_coefficient in range(int(lower[0]), int(upper[0]) + 1):
        for second_coefficient in range(int(lower[1]), int(upper[1]) + 1):
            coefficients = np.asarray(
                (first_coefficient, second_coefficient),
                dtype=np.int64,
            )
            candidate_norm: float = perpendicular_norm_squared(coefficients)
            candidate_key: tuple[float, int, int] = (
                candidate_norm,
                first_coefficient,
                second_coefficient,
            )
            best_key: tuple[float, int, int] = (
                best_norm,
                int(best_coefficients[0]),
                int(best_coefficients[1]),
            )
            if candidate_key < best_key:
                best_norm = candidate_norm
                best_coefficients = coefficients
    closest: np.ndarray = stacking + best_coefficients @ kernel
    return closest


def _validate_miller(
    miller: tuple[int, int, int],
) -> tuple[int, int, int]:
    """Validate a static primitive Miller tuple."""
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
    reciprocal_normal: Float[Array, " 3"],
) -> Float[Array, "3 3"]:
    """Construct the guarded active rotation from a normal to positive z."""
    normal_norm: Float[Array, ""] = safe_norm(reciprocal_normal)
    normal: Float[Array, " 3"] = reciprocal_normal / normal_norm
    z_axis: Float[Array, " 3"] = jnp.asarray(
        (0.0, 0.0, 1.0),
        dtype=normal.dtype,
    )
    cross_vector: Float[Array, " 3"] = jnp.cross(normal, z_axis)
    cross_x: Float[Array, ""] = cross_vector[0]
    cross_y: Float[Array, ""] = cross_vector[1]
    cross_z: Float[Array, ""] = cross_vector[2]
    zero: Float[Array, ""] = jnp.zeros_like(cross_x)
    skew: Float[Array, "3 3"] = jnp.stack(
        (
            jnp.stack((zero, -cross_z, cross_y)),
            jnp.stack((cross_z, zero, -cross_x)),
            jnp.stack((-cross_y, cross_x, zero)),
        )
    )
    cosine: Float[Array, ""] = jnp.dot(normal, z_axis)
    away_from_antipode: Array = cosine > -1.0 + 1e-12
    sanitized_denominator: Float[Array, ""] = jnp.where(
        away_from_antipode,
        1.0 + cosine,
        1.0,
    )
    identity: Float[Array, "3 3"] = jnp.eye(3, dtype=normal.dtype)
    generic: Float[Array, "3 3"] = (
        identity + skew + (skew @ skew) / sanitized_denominator
    )
    antiparallel: Float[Array, "3 3"] = jnp.diag(
        jnp.asarray((1.0, -1.0, -1.0), dtype=normal.dtype)
    )
    rotation: Float[Array, "3 3"] = jnp.where(
        away_from_antipode,
        generic,
        antiparallel,
    )
    return rotation


def _assemble_surface_cell(
    geometry: CrystalGeometry,
    miller: tuple[int, int, int],
    in_plane_coeffs: tuple[
        tuple[int, int, int],
        tuple[int, int, int],
    ],
    stacking_coeffs: tuple[int, int, int],
) -> SurfaceCell:
    """Assemble continuous surface geometry from frozen integer topology."""
    coefficient_array: Float[Array, "3 3"] = jnp.asarray(
        (*in_plane_coeffs, stacking_coeffs),
        dtype=jnp.float64,
    )
    bulk_vectors: Float[Array, "3 3"] = coefficient_array @ geometry.lattice
    miller_array: Float[Array, " 3"] = jnp.asarray(
        miller,
        dtype=jnp.float64,
    )
    reciprocal_normal: Float[Array, " 3"] = miller_array @ geometry.reciprocal
    rotation: Float[Array, "3 3"] = _surface_rotation(reciprocal_normal)
    surface_vectors: Float[Array, "3 3"] = bulk_vectors @ rotation.T
    spacing: Float[Array, ""] = 2.0 * jnp.pi / safe_norm(reciprocal_normal)
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
    miller: tuple[int, int, int],
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
    miller : tuple[int, int, int]
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
    primitive_miller: tuple[int, int, int] = _validate_miller(miller)
    lattice_snapshot: np.ndarray = np.asarray(
        geometry.lattice,
        dtype=np.float64,
    )
    kernel: np.ndarray
    stacking: np.ndarray
    kernel, stacking = _primitive_integer_frame(primitive_miller)
    kernel = _gauss_reduce(kernel, lattice_snapshot)
    stacking = _closest_stacking_vector(
        stacking,
        kernel,
        lattice_snapshot,
    )
    in_plane_coeffs: tuple[
        tuple[int, int, int],
        tuple[int, int, int],
    ] = (
        tuple(int(value) for value in kernel[0]),
        tuple(int(value) for value in kernel[1]),
    )
    stacking_coeffs: tuple[int, int, int] = tuple(
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
) -> dict[tuple[int, int, int, int], list[int]]:
    """Compute orbital groups by site, principal shell, l, and spin."""
    groups: dict[tuple[int, int, int, int], list[int]] = {}
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
) -> dict[
    tuple[int, int, int, int],
    tuple[int, ...],
]:
    """Return missing m values for every incomplete registered shell."""
    missing: dict[tuple[int, int, int, int], tuple[int, ...]] = {}
    key: tuple[int, int, int, int]
    indices: list[int]
    for key, indices in _shell_groups(model).items():
        angular: int = key[2]
        present: list[int] = [model.basis.m[index] for index in indices]
        expected: set[int] = set(range(-angular, angular + 1))
        absent: tuple[int, ...] = tuple(sorted(expected - set(present)))
        duplicated: bool = len(present) != len(set(present))
        if absent or duplicated or len(present) != 2 * angular + 1:
            missing[key] = absent
    return missing


def _rotation_euler_zyz(
    rotation: Float[Array, "3 3"],
) -> tuple[Float[Array, ""], Float[Array, ""], Float[Array, ""]]:
    """Convert one active Cartesian rotation to guarded z-y-z angles."""
    beta: Float[Array, ""] = safe_arccos(rotation[2, 2])
    sine_beta: Float[Array, ""] = jnp.sin(beta)
    generic_alpha: Float[Array, ""] = safe_arctan2(
        rotation[1, 2],
        rotation[0, 2],
    )
    generic_gamma: Float[Array, ""] = safe_arctan2(
        rotation[2, 1],
        -rotation[2, 0],
    )
    positive_alpha: Float[Array, ""] = safe_arctan2(
        rotation[1, 0],
        rotation[0, 0],
    )
    negative_alpha: Float[Array, ""] = safe_arctan2(
        -rotation[1, 0],
        -rotation[0, 0],
    )
    pole_alpha: Float[Array, ""] = jnp.where(
        rotation[2, 2] >= 0.0,
        positive_alpha,
        negative_alpha,
    )
    alpha: Float[Array, ""] = jnp.where(
        jnp.abs(sine_beta) > 1e-12,  # noqa: PLR2004
        generic_alpha,
        pole_alpha,
    )
    gamma: Float[Array, ""] = jnp.where(
        jnp.abs(sine_beta) > 1e-12,  # noqa: PLR2004
        generic_gamma,
        0.0,
    )
    result: tuple[
        Float[Array, ""],
        Float[Array, ""],
        Float[Array, ""],
    ] = (alpha, beta, gamma)
    return result


def _real_wigner(
    angular: int,
    alpha: Float[Array, ""],
    beta: Float[Array, ""],
    gamma: Float[Array, ""],
) -> Complex[Array, "m1 m2"]:
    """Convert the complex Wigner representation to real harmonics."""
    complex_matrix: Complex[Array, "m1 m2"] = wigner_d(
        angular,
        alpha,
        beta,
        gamma,
    )
    unitary: Complex[Array, "m1 m2"] = real_harmonic_unitary(angular)
    real_matrix: Complex[Array, "m1 m2"] = (
        unitary.conj() @ complex_matrix @ unitary.T
    )
    return real_matrix


def _spin_half_wigner(
    alpha: Float[Array, ""],
    beta: Float[Array, ""],
    gamma: Float[Array, ""],
) -> Complex[Array, "2 2"]:
    """Construct the spin-half Wigner matrix in (-1, +1) order."""
    cosine: Float[Array, ""] = jnp.cos(0.5 * beta)
    sine: Float[Array, ""] = jnp.sin(0.5 * beta)
    small: Float[Array, "2 2"] = jnp.stack(
        (
            jnp.stack((cosine, sine)),
            jnp.stack((-sine, cosine)),
        )
    )
    magnetic: Float[Array, " 2"] = jnp.asarray(
        (-0.5, 0.5),
        dtype=beta.dtype,
    )
    alpha_phase: Complex[Array, " 2"] = jnp.exp(-1j * magnetic * alpha)
    gamma_phase: Complex[Array, " 2"] = jnp.exp(-1j * magnetic * gamma)
    matrix: Complex[Array, "2 2"] = (
        alpha_phase[:, None] * small * gamma_phase[None, :]
    )
    return matrix


def _orbital_rotation(
    model: TBModel,
    rotation: Float[Array, "3 3"],
) -> Complex[Array, "n_orb n_orb"]:
    """Assemble the block-diagonal orbital and spin representation."""
    alpha: Float[Array, ""]
    beta: Float[Array, ""]
    gamma: Float[Array, ""]
    alpha, beta, gamma = _rotation_euler_zyz(rotation)
    n_orbitals: int = len(model.basis.n)
    representation: Complex[Array, "n_orb n_orb"] = jnp.zeros(
        (n_orbitals, n_orbitals),
        dtype=jnp.complex128,
    )
    angular_matrices: dict[int, Complex[Array, "m1 m2"]] = {
        angular: _real_wigner(angular, alpha, beta, gamma)
        for angular in set(model.basis.l)
    }
    spin_matrix: Complex[Array, "2 2"] | None = (
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
            angular_factor: Complex[Array, ""] = angular_matrices[angular][
                model.basis.m[row] + angular,
                model.basis.m[column] + angular,
            ]
            if spin_matrix is None:
                spin_factor: complex | Complex[Array, ""] = (
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
) -> tuple[tuple[tuple[int, int, int], ...], Complex[Array, "n_r n_o n_o"]]:
    """Materialize translation blocks, including diagonal onsite terms."""
    zero_cell: tuple[int, int, int] = (0, 0, 0)
    cells: tuple[tuple[int, int, int], ...] = tuple(
        sorted(set(model.hopping_cells) | {zero_cell})
    )
    n_orbitals: int = len(model.basis.n)
    blocks: Complex[Array, "n_r n_o n_o"] = jnp.zeros(
        (len(cells), n_orbitals, n_orbitals),
        dtype=jnp.complex128,
    )
    cell_lookup: dict[tuple[int, int, int], int] = {
        cell: index for index, cell in enumerate(cells)
    }
    hopping: int
    pair: tuple[int, int]
    cell: tuple[int, int, int]
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
    result: tuple[
        tuple[tuple[int, int, int], ...],
        Complex[Array, "n_r n_o n_o"],
    ] = (cells, blocks)
    return result


@jaxtyped(typechecker=beartype)
def rotate_tb_model(  # noqa: DOC503
    model: TBModel,
    rotation: Float[Array, "3 3"],
) -> TBModel:
    """Construct a rotated complete-shell tight-binding model.

    Parameters
    ----------
    model : TBModel
        Bulk model in a real-harmonic orbital basis.
    rotation : Float[Array, "3 3"]
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
    rotation_array: Float[Array, "3 3"] = jnp.asarray(
        rotation,
        dtype=jnp.float64,
    )
    missing: dict[
        tuple[int, int, int, int],
        tuple[int, ...],
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
    identity: Float[Array, "3 3"] = jnp.eye(3, dtype=jnp.float64)
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
    representation: Complex[Array, "n_orb n_orb"] = _orbital_rotation(
        model,
        rotation_array,
    )
    cells: tuple[tuple[int, int, int], ...]
    blocks: Complex[Array, "n_r n_o n_o"]
    cells, blocks = _translation_blocks(model)
    rotated_blocks: Complex[Array, "n_r n_o n_o"] = (
        representation[None, :, :]
        @ blocks
        @ representation.conj().T[None, :, :]
    )
    n_orbitals: int = len(model.basis.n)
    hopping_pairs: tuple[tuple[int, int], ...] = tuple(
        (row, column)
        for _cell in cells
        for row in range(n_orbitals)
        for column in range(n_orbitals)
    )
    hopping_cells: tuple[tuple[int, int, int], ...] = tuple(
        cell
        for cell in cells
        for _row in range(n_orbitals)
        for _column in range(n_orbitals)
    )
    hopping_amplitudes: Complex[Array, " n_hop"] = rotated_blocks.reshape(-1)
    rotated_lattice: Float[Array, "3 3"] = (
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


def _surface_integer_matrix(surface_cell: SurfaceCell) -> np.ndarray:
    """Return the exact row-wise bulk-to-surface integer frame."""
    coefficients: np.ndarray = np.asarray(
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


def _inverse_integer_matrix(coefficients: np.ndarray) -> np.ndarray:
    """Compute a unimodular inverse and verify exact integer recovery."""
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
    inverse: np.ndarray = np.asarray(
        (
            (e * i - f * h, c * h - b * i, b * f - c * e),
            (f * g - d * i, a * i - c * g, c * d - a * f),
            (d * h - e * g, b * g - a * h, a * e - b * d),
        ),
        dtype=np.int64,
    )
    identity: np.ndarray = coefficients @ inverse
    if not np.array_equal(identity, np.eye(3, dtype=np.int64)):
        message: str = "surface integer frame inverse is not exact"
        raise ValueError(message)
    return inverse


def _base_surface_coordinates(
    geometry: CrystalGeometry,
    inverse_coefficients: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute atom representatives and integer surface-cell shifts."""
    surface_coordinates: np.ndarray = (
        np.asarray(geometry.positions, dtype=np.float64) @ inverse_coefficients
    )
    shifts: np.ndarray = np.floor(surface_coordinates + 1e-10).astype(np.int64)
    base: np.ndarray = surface_coordinates - shifts
    base[np.isclose(base, 1.0, atol=1e-10)] = 0.0
    result: tuple[np.ndarray, np.ndarray] = (base, shifts)
    return result


def _natural_atom_copies(
    geometry: CrystalGeometry,
    base_coordinates: np.ndarray,
    surface_vectors: np.ndarray,
    n_layers: int,
    thickness_ang: float,
    fine: tuple[float, float],
) -> tuple[tuple[int, ...], tuple[int, ...], tuple[str, str]]:
    """Select the smallest post-fine natural stack meeting the minimum span."""
    spacing: float = abs(float(surface_vectors[2, 2]))
    base_heights: np.ndarray = base_coordinates @ surface_vectors[:, 2]
    baseline_bottom: float = float(np.min(base_heights))
    base_top: float = float(np.max(base_heights))
    positive_fine_span: float = max(fine[0], 0.0) + max(fine[1], 0.0)
    maximum_extra_layers: int = (
        int(math.ceil(positive_fine_span / spacing)) + 3
    )
    extra_span: float = max(abs(fine[0]), abs(fine[1]))
    padding: int = int(math.ceil(extra_span / spacing)) + 1
    candidates: list[tuple[float, int, int]] = []
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
    unique_layers: tuple[int, ...] = tuple(
        sorted({row[2] for row in candidates})
    )
    layer_map: dict[int, int] = {
        original: normalized
        for normalized, original in enumerate(unique_layers)
    }
    bulk_atoms: tuple[int, ...] = tuple(row[1] for row in candidates)
    layers: tuple[int, ...] = tuple(layer_map[row[2]] for row in candidates)
    heights: np.ndarray = np.asarray(
        [row[0] for row in candidates],
        dtype=np.float64,
    )
    bottom_index: int = int(np.argmin(heights))
    top_index: int = int(np.argmax(heights))
    species: tuple[str, ...] = geometry.species
    if not species:
        species = tuple("X" for _ in range(geometry.positions.shape[0]))
    termination: tuple[str, str] = (
        species[candidates[top_index][1]],
        species[candidates[bottom_index][1]],
    )
    result: tuple[
        tuple[int, ...],
        tuple[int, ...],
        tuple[str, str],
    ] = (bulk_atoms, layers, termination)
    return result


def _terminated_atom_copies(  # noqa: PLR0913
    geometry: CrystalGeometry,
    base_coordinates: np.ndarray,
    surface_vectors: np.ndarray,
    n_layers: int,
    thickness_ang: float,
    termination: tuple[str, str],
    fine: tuple[float, float],
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Compute a post-fine stack with requested endpoint species."""
    if not geometry.species:
        message: str = "explicit termination requires geometry species"
        raise ValueError(message)
    top_species: str
    bottom_species: str
    top_species, bottom_species = termination
    spacing: float = abs(float(surface_vectors[2, 2]))
    base_heights: np.ndarray = base_coordinates @ surface_vectors[:, 2]
    baseline_bottom: float = float(np.min(base_heights))
    base_top: float = float(np.max(base_heights))
    positive_fine_span: float = max(fine[0], 0.0) + max(fine[1], 0.0)
    maximum_extra_layers: int = (
        int(math.ceil(positive_fine_span / spacing)) + 5
    )
    extra_span: float = max(abs(fine[0]), abs(fine[1]))
    padding: int = int(math.ceil(extra_span / spacing)) + 2
    kept: list[tuple[float, int, int, str]] = []
    expanded_layers: int
    for expanded_layers in range(
        n_layers,
        n_layers + maximum_extra_layers + 1,
    ):
        baseline_top: float = base_top + (expanded_layers - 1) * spacing
        bottom_cut: float = baseline_bottom + fine[1]
        top_cut: float = baseline_top - fine[0]
        candidates: list[tuple[float, int, int, str]] = []
        layer: int
        atom: int
        for layer in range(-padding, expanded_layers + padding):
            for atom in range(base_coordinates.shape[0]):
                height: float = float(base_heights[atom] + layer * spacing)
                if height >= bottom_cut - 1e-10 and height <= top_cut + 1e-10:
                    candidates.append(
                        (height, atom, layer, geometry.species[atom])
                    )
        bottom_rows: tuple[tuple[float, int, int, str], ...] = tuple(
            row for row in candidates if row[3] == bottom_species
        )
        top_rows: tuple[tuple[float, int, int, str], ...] = tuple(
            row for row in candidates if row[3] == top_species
        )
        if not bottom_rows or not top_rows:
            continue
        bottom: tuple[float, int, int, str] = min(
            bottom_rows,
            key=lambda row: (row[0], row[2], row[1]),
        )
        top: tuple[float, int, int, str] = max(
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
    unique_layers: tuple[int, ...] = tuple(sorted({row[2] for row in kept}))
    layer_map: dict[int, int] = {
        original: normalized
        for normalized, original in enumerate(unique_layers)
    }
    bulk_atoms: tuple[int, ...] = tuple(row[1] for row in kept)
    layers: tuple[int, ...] = tuple(layer_map[row[2]] for row in kept)
    result: tuple[tuple[int, ...], tuple[int, ...]] = (bulk_atoms, layers)
    return result


def _orbital_copy_metadata(
    basis: OrbitalBasis,
    bulk_atom_of_slab_atom: tuple[int, ...],
    layer_of_slab_atom: tuple[int, ...],
) -> tuple[
    tuple[int, ...],
    tuple[int, ...],
    tuple[int, ...],
]:
    """Compute each slab orbital's bulk, atom, and layer mapping."""
    orbitals_by_atom: dict[int, list[int]] = {}
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
    result: tuple[
        tuple[int, ...],
        tuple[int, ...],
        tuple[int, ...],
    ] = (
        tuple(slab_to_bulk),
        tuple(slab_atom_indices),
        tuple(slab_layers),
    )
    return result


def _slab_basis(
    basis: OrbitalBasis,
    slab_to_bulk: tuple[int, ...],
    slab_atom_indices: tuple[int, ...],
    slab_layers: tuple[int, ...],
) -> OrbitalBasis:
    """Create static orbital metadata for one frozen slab topology."""
    labels: tuple[str, ...] = tuple(
        f"{basis.labels[bulk] if basis.labels else f'orb{bulk}'}@L{layer}"
        for bulk, layer in zip(slab_to_bulk, slab_layers, strict=True)
    )
    spin: tuple[int, ...] = (
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
    slab_to_bulk: tuple[int, ...],
    slab_atom_indices: tuple[int, ...],
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Create SOC shell IDs and their bulk-shell gather."""
    shell_lookup: dict[tuple[int, int], int] = {}
    slab_shells: list[int] = []
    bulk_shell_gather: list[int] = []
    slab_orbital: int
    bulk_orbital: int
    for slab_orbital, bulk_orbital in enumerate(slab_to_bulk):
        bulk_shell: int = model.shell_index[bulk_orbital]
        if bulk_shell < 0:
            slab_shells.append(-1)
            continue
        key: tuple[int, int] = (
            slab_atom_indices[slab_orbital],
            bulk_shell,
        )
        if key not in shell_lookup:
            shell_lookup[key] = len(shell_lookup)
            bulk_shell_gather.append(bulk_shell)
        slab_shells.append(shell_lookup[key])
    result: tuple[tuple[int, ...], tuple[int, ...]] = (
        tuple(slab_shells),
        tuple(bulk_shell_gather),
    )
    return result


def _orbital_lookup(
    model: TBModel,
    spec: SlabSpec,
) -> tuple[
    tuple[int, ...],
    tuple[int, ...],
    tuple[int, ...],
    dict[tuple[int, int], int],
]:
    """Create deterministic slab-orbital lookup metadata."""
    slab_to_bulk: tuple[int, ...]
    slab_atom_indices: tuple[int, ...]
    slab_layers: tuple[int, ...]
    slab_to_bulk, slab_atom_indices, slab_layers = _orbital_copy_metadata(
        model.basis,
        spec.bulk_atom_of_slab_atom,
        spec.layer_of_slab_atom,
    )
    lookup: dict[tuple[int, int], int] = {
        (bulk, layer): slab
        for slab, (bulk, layer) in enumerate(
            zip(slab_to_bulk, slab_layers, strict=True)
        )
    }
    result: tuple[
        tuple[int, ...],
        tuple[int, ...],
        tuple[int, ...],
        dict[tuple[int, int], int],
    ] = (slab_to_bulk, slab_atom_indices, slab_layers, lookup)
    return result


def _propagate_hoppings_with_shifts(
    rotated_bulk: TBModel,
    spec: SlabSpec,
    atom_shifts: np.ndarray,
) -> tuple[
    Complex[Array, " n_hop_slab"],
    tuple[tuple[int, int], ...],
    tuple[tuple[int, int, int], ...],
    tuple[int, ...],
]:
    """Propagate exact bulk hoppings through one frozen slab topology."""
    coefficients: np.ndarray = _surface_integer_matrix(spec.surface_cell)
    inverse: np.ndarray = _inverse_integer_matrix(coefficients)
    slab_to_bulk: tuple[int, ...]
    slab_atom_indices: tuple[int, ...]
    slab_layers: tuple[int, ...]
    lookup: dict[tuple[int, int], int]
    (
        slab_to_bulk,
        slab_atom_indices,
        slab_layers,
        lookup,
    ) = _orbital_lookup(rotated_bulk, spec)
    pairs: list[tuple[int, int]] = []
    cells: list[tuple[int, int, int]] = []
    gather: list[int] = []
    slab_source: int
    bulk_source: int
    pair: tuple[int, int]
    bulk_cell: tuple[int, int, int]
    for slab_source, bulk_source in enumerate(slab_to_bulk):
        source_atom: int = rotated_bulk.basis.atom_indices[bulk_source]
        source_layer: int = slab_layers[slab_source]
        source_surface_cell: np.ndarray = np.asarray(
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
            transformed_cell: np.ndarray = (
                np.asarray(bulk_cell, dtype=np.int64) @ inverse
            )
            target_surface_cell: np.ndarray = (
                source_surface_cell + transformed_cell
            )
            target_layer: int = int(
                target_surface_cell[2] + atom_shifts[target_atom, 2]
            )
            slab_target: int | None = lookup.get((bulk_target, target_layer))
            if slab_target is None:
                continue
            slab_cell: tuple[int, int, int] = (
                int(target_surface_cell[0] + atom_shifts[target_atom, 0]),
                int(target_surface_cell[1] + atom_shifts[target_atom, 1]),
                0,
            )
            pairs.append((slab_source, slab_target))
            cells.append(slab_cell)
            gather.append(hopping_index)
    gather_tuple: tuple[int, ...] = tuple(gather)
    gather_array: Array = jnp.asarray(gather_tuple, dtype=jnp.int32)
    amplitudes: Complex[Array, " n_hop_slab"] = (
        rotated_bulk.hopping_amplitudes[gather_array]
    )
    result: tuple[
        Complex[Array, " n_hop_slab"],
        tuple[tuple[int, int], ...],
        tuple[tuple[int, int, int], ...],
        tuple[int, ...],
    ] = (amplitudes, tuple(pairs), tuple(cells), gather_tuple)
    return result


@jaxtyped(typechecker=beartype)
def _propagate_hoppings(
    rotated_bulk: TBModel,
    spec: SlabSpec,
) -> tuple[
    Complex[Array, " n_hop_slab"],
    tuple[tuple[int, int], ...],
    tuple[tuple[int, int, int], ...],
    tuple[int, ...],
]:
    """Propagate hoppings after eagerly selecting atom representatives."""
    coefficients: np.ndarray = _surface_integer_matrix(spec.surface_cell)
    inverse: np.ndarray = _inverse_integer_matrix(coefficients)
    atom_shifts: np.ndarray
    _, atom_shifts = _base_surface_coordinates(
        rotated_bulk.geometry,
        inverse,
    )
    result: tuple[
        Complex[Array, " n_hop_slab"],
        tuple[tuple[int, int], ...],
        tuple[tuple[int, int, int], ...],
        tuple[int, ...],
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
    offending: tuple[
        tuple[int, tuple[int, int], tuple[int, int, int]],
        ...,
    ] = tuple(
        (index, pair, cell)
        for index, (pair, cell) in enumerate(
            zip(model.hopping_pairs, model.hopping_cells, strict=True)
        )
        if cell[2] != 0
    )
    adjacency: dict[
        int,
        list[tuple[int, int, int]],
    ] = {}
    index: int
    pair: tuple[int, int]
    cell: tuple[int, int, int]
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
        orbital_heights: np.ndarray = np.asarray(
            orbital_fractional @ model.geometry.lattice,
            dtype=np.float64,
        )[:, 2]
        top_height: float = float(np.max(orbital_heights))
        bottom_height: float = float(np.min(orbital_heights))
        top_orbitals: tuple[int, ...] = tuple(
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
        witness: tuple[int, ...] | None = None
        root: int
        source: int
        normal_image: int
        path: tuple[int, ...]
        edge_index: int
        target: int
        delta: int
        for root in top_orbitals:
            queue: list[tuple[int, int, tuple[int, ...]]] = [(root, 0, ())]
            visited: set[tuple[int, int]] = {(root, 0)}
            while queue and witness is None:
                source, normal_image, path = queue.pop(0)
                if len(path) >= maximum_steps:
                    continue
                for edge_index, target, delta in adjacency.get(source, ()):
                    target_image: int = normal_image + delta
                    target_path: tuple[int, ...] = (*path, edge_index)
                    if target in bottom_orbitals and target_image != 0:
                        witness = target_path
                        break
                    state: tuple[int, int] = (target, target_image)
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
    inverse_coefficients: np.ndarray,
    atom_shifts: np.ndarray,
    bulk_atoms: tuple[int, ...],
    atom_layers: tuple[int, ...],
    slab_to_bulk: tuple[int, ...],
    slab_atom_indices: tuple[int, ...],
    n_layers: int,
    vacuum_ang: float,
) -> tuple[
    CrystalGeometry,
    Float[Array, " n_orb"],
    Float[Array, "n_orb 3"] | None,
]:
    """Assemble differentiable slab positions, centres, and depth tags."""
    inverse_array: Float[Array, "3 3"] = jnp.asarray(
        inverse_coefficients,
        dtype=jnp.float64,
    )
    surface_vectors: Float[Array, "3 3"] = jnp.vstack(
        (
            surface_cell.in_plane_vectors,
            surface_cell.stacking_vector[None, :],
        )
    )
    bulk_atom_array: Array = jnp.asarray(bulk_atoms, dtype=jnp.int32)
    base_surface: Float[Array, "n_atoms 3"] = (
        rotated_bulk.geometry.positions @ inverse_array
        - jnp.asarray(atom_shifts, dtype=jnp.float64)
    )
    atom_layer_array: Float[Array, " n_slab_atoms"] = jnp.asarray(
        atom_layers,
        dtype=jnp.float64,
    )
    atom_surface: Float[Array, "n_slab_atoms 3"] = (
        base_surface[bulk_atom_array].at[:, 2].add(atom_layer_array)
    )
    atom_cart: Float[Array, "n_slab_atoms 3"] = atom_surface @ surface_vectors
    bottom: Float[Array, ""] = jnp.min(atom_cart[:, 2])
    atom_cart = atom_cart.at[:, 2].add(-bottom)
    material_period: Float[Array, ""] = (
        jnp.asarray(float(n_layers), dtype=jnp.float64)
        * surface_cell.interlayer_spacing_ang
    )
    height: Float[Array, ""] = material_period + jnp.asarray(
        vacuum_ang,
        dtype=jnp.float64,
    )
    slab_lattice: Float[Array, "3 3"] = jnp.vstack(
        (
            surface_cell.in_plane_vectors,
            jnp.stack(
                (jnp.zeros_like(height), jnp.zeros_like(height), height)
            ),
        )
    )
    inverse_lattice: Float[Array, "3 3"] = jnp.linalg.inv(slab_lattice)
    atom_fractional: Float[Array, "n_slab_atoms 3"] = (
        atom_cart @ inverse_lattice
    )
    atom_fractional = atom_fractional.at[:, :2].set(
        jnp.mod(atom_fractional[:, :2], 1.0)
    )
    species_source: tuple[str, ...] = rotated_bulk.geometry.species
    if not species_source:
        species_source = tuple(
            "X" for _ in range(rotated_bulk.geometry.positions.shape[0])
        )
    slab_species: tuple[str, ...] = tuple(
        species_source[index] for index in bulk_atoms
    )
    slab_geometry: CrystalGeometry = make_crystal_geometry(
        lattice=slab_lattice,
        positions=atom_fractional,
        species=slab_species,
    )

    orbital_position_array: Float[Array, "n_orb 3"] | None = None
    if rotated_bulk.orbital_positions is not None:
        bulk_orbital_array: Array = jnp.asarray(
            slab_to_bulk,
            dtype=jnp.int32,
        )
        bulk_centre_surface: Float[Array, "n_bulk_orb 3"] = (
            rotated_bulk.orbital_positions @ inverse_array
        )
        centre_shifts: Float[Array, "n_orb 3"] = jnp.asarray(
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
        centre_surface: Float[Array, "n_orb 3"] = (
            bulk_centre_surface[bulk_orbital_array] + centre_shifts
        )
        centre_cart: Float[Array, "n_orb 3"] = centre_surface @ surface_vectors
        centre_cart = centre_cart.at[:, 2].add(-bottom)
        orbital_position_array = centre_cart @ inverse_lattice
        orbital_position_array = orbital_position_array.at[:, :2].set(
            jnp.mod(orbital_position_array[:, :2], 1.0)
        )
        depth_coordinates: Float[Array, " n_orb"] = centre_cart[:, 2]
    else:
        slab_atom_index_array: Array = jnp.asarray(
            slab_atom_indices,
            dtype=jnp.int32,
        )
        depth_coordinates = atom_cart[slab_atom_index_array, 2]
    depths: Float[Array, " n_orb"] = (
        jnp.max(depth_coordinates) - depth_coordinates
    )
    result: tuple[
        CrystalGeometry,
        Float[Array, " n_orb"],
        Float[Array, "n_orb 3"] | None,
    ] = (slab_geometry, depths, orbital_position_array)
    return result


@jaxtyped(typechecker=beartype)
def freeze_slab_topology(  # noqa: PLR0913
    bulk_model: TBModel,
    miller: tuple[int, int, int],
    thickness_ang: float,
    vacuum_ang: float,
    termination: tuple[str, str] | None = None,
    fine: tuple[float, float] = (0.0, 0.0),
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
    coefficients: np.ndarray = _surface_integer_matrix(surface_cell)
    inverse: np.ndarray = _inverse_integer_matrix(coefficients)
    base_coordinates: np.ndarray
    atom_shifts: np.ndarray
    base_coordinates, atom_shifts = _base_surface_coordinates(
        bulk_model.geometry,
        inverse,
    )
    surface_vectors_snapshot: np.ndarray = np.vstack(
        (
            np.asarray(surface_cell.in_plane_vectors),
            np.asarray(surface_cell.stacking_vector)[None, :],
        )
    )
    spacing_snapshot: float = float(surface_cell.interlayer_spacing_ang)
    n_layers: int = (
        int(math.ceil(thickness_ang / spacing_snapshot - 1e-12)) + 1
    )
    bulk_atoms: tuple[int, ...]
    atom_layers: tuple[int, ...]
    resolved_termination: tuple[str, str]
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
) -> tuple[TBModel, SlabSpec]:
    """Construct a slab from frozen topology using only JAX geometry.

    The function contains no host conversion of a traced model leaf. It is
    therefore the differentiable Plan-05 D2 seam. Call
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
    coefficients: np.ndarray = _surface_integer_matrix(surface_cell)
    inverse: np.ndarray = _inverse_integer_matrix(coefficients)
    atom_shifts: np.ndarray = np.asarray(
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
    slab_to_bulk: tuple[int, ...]
    slab_atom_indices: tuple[int, ...]
    slab_layers: tuple[int, ...]
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
    slab_shells: tuple[int, ...]
    shell_gather: tuple[int, ...]
    slab_shells, shell_gather = _slab_shell_metadata(
        rotated_bulk,
        slab_to_bulk,
        slab_atom_indices,
    )
    slab_geometry: CrystalGeometry
    depths: Float[Array, " n_orb"]
    orbital_positions: Float[Array, "n_orb 3"] | None
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
    amplitudes: Complex[Array, " n_hop_slab"]
    hopping_pairs: tuple[tuple[int, int], ...]
    hopping_cells: tuple[tuple[int, int, int], ...]
    _gather: tuple[int, ...]
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
    result: tuple[TBModel, SlabSpec] = (slab_model, spec)
    return result


@jaxtyped(typechecker=beartype)
def gen_slab(  # noqa: DOC105, DOC502, PLR0913, PLR0915
    bulk_model: TBModel,
    miller: tuple[int, int, int],
    thickness_ang: float,
    vacuum_ang: float,
    termination: tuple[str, str] | None = None,
    fine: tuple[float, float] = (0.0, 0.0),
) -> tuple[TBModel, SlabSpec]:
    """Construct a finite Miller-index slab with exact open-normal topology.

    Eager input selects discrete atoms, orbitals, and hoppings. Every assembled
    geometry, centre, depth, amplitude, onsite, and SOC value remains a JAX
    array. Re-run the factory after a continuous perturbation crosses a layer,
    termination, or metric-reduction boundary.

    Parameters
    ----------
    bulk_model : TBModel
        Validated bulk tight-binding model.
    miller : tuple[int, int, int]
        Primitive Miller indices (**static**).
    thickness_ang : float
        Nonnegative minimum material span in Angstrom (**static**). Zero
        requests the one-plane limiting slab.
    vacuum_ang : float
        Nonnegative vacuum padding in Angstrom (**static**).
    termination : tuple[str, str] or None, optional
        Requested ``(top, bottom)`` species. ``None`` retains a natural
        complete stack. Default is ``None``.
    fine : tuple[float, float], optional
        Static ``(top, bottom)`` inward cut shifts in Angstrom.

    Returns
    -------
    result : tuple[TBModel, SlabSpec]
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
    result: tuple[TBModel, SlabSpec] = rebuild_slab(bulk_model, topology)
    return result


def _slab_operator_centres(
    bulk_model: TBModel,
    operator_data: WannierOperatorData,
    spec: SlabSpec,
) -> Float[Array, "n_orb_slab 3"]:
    """Convert explicit Wannier centres into the slab cell."""
    coefficients: np.ndarray = _surface_integer_matrix(spec.surface_cell)
    inverse: np.ndarray = _inverse_integer_matrix(coefficients)
    atom_shifts: np.ndarray
    _, atom_shifts = _base_surface_coordinates(
        bulk_model.geometry,
        inverse,
    )
    slab_to_bulk: tuple[int, ...]
    slab_layers: tuple[int, ...]
    slab_to_bulk, _, slab_layers, _ = _orbital_lookup(bulk_model, spec)
    inverse_array: Float[Array, "3 3"] = jnp.asarray(
        inverse,
        dtype=jnp.float64,
    )
    surface_vectors: Float[Array, "3 3"] = jnp.vstack(
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
                shell_centres: np.ndarray = np.asarray(
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
        bulk_centres_rotated: Float[Array, "n_orb 3"] = (
            operator_data.centres_cart @ spec.surface_cell.rotation.T
        )
    else:
        representation: Complex[Array, "n_orb n_orb"] = (
            jnp.eye(len(bulk_model.basis.n), dtype=jnp.complex128)
            if identity_rotation
            else _orbital_rotation(
                bulk_model,
                spec.surface_cell.rotation,
            )
        )
        orbital_rotated: Complex[Array, "n_R n_orb n_orb 3"] = jnp.einsum(
            "ai,rijc,bj->rabc",
            representation,
            operator_data.position_matrices,
            representation.conj(),
        )
        rotated_blocks: Complex[Array, "n_R n_orb n_orb 3"] = jnp.einsum(
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
    translations: Float[Array, "n_orb_slab 3"] = jnp.asarray(
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
    centres_cart: Float[Array, "n_orb_slab 3"] = (
        bulk_centres_rotated[bulk_indices] + translations @ surface_vectors
    )

    base_atoms: Float[Array, "n_atoms 3"] = (
        bulk_model.geometry.positions @ inverse_array
        - jnp.asarray(atom_shifts, dtype=jnp.float64)
    )
    slab_atom_surface: Float[Array, "n_slab_atoms 3"] = base_atoms[
        jnp.asarray(spec.bulk_atom_of_slab_atom, dtype=jnp.int32)
    ]
    slab_atom_surface = slab_atom_surface.at[:, 2].add(
        jnp.asarray(spec.layer_of_slab_atom, dtype=jnp.float64)
    )
    bottom: Float[Array, ""] = jnp.min(
        (slab_atom_surface @ surface_vectors)[:, 2]
    )
    shifted_centres: Float[Array, "n_orb_slab 3"] = centres_cart.at[:, 2].add(
        -bottom
    )
    return shifted_centres


def _propagated_operator_cells(
    bulk_model: TBModel,
    operator_data: WannierOperatorData,
    spec: SlabSpec,
) -> tuple[tuple[int, int, int], ...]:
    """Compute an operator cell grid for the exact slab topology."""
    coefficients: np.ndarray = _surface_integer_matrix(spec.surface_cell)
    inverse: np.ndarray = _inverse_integer_matrix(coefficients)
    atom_shifts: np.ndarray
    _, atom_shifts = _base_surface_coordinates(
        bulk_model.geometry,
        inverse,
    )
    slab_to_bulk: tuple[int, ...]
    slab_layers: tuple[int, ...]
    lookup: dict[tuple[int, int], int]
    slab_to_bulk, _, slab_layers, lookup = _orbital_lookup(
        bulk_model,
        spec,
    )
    cells: set[tuple[int, int, int]] = set()
    slab_source: int
    bulk_source: int
    for slab_source, bulk_source in enumerate(slab_to_bulk):
        source_atom: int = bulk_model.basis.atom_indices[bulk_source]
        source_layer: int = slab_layers[slab_source]
        source_surface_cell: np.ndarray = np.asarray(
            (
                -atom_shifts[source_atom, 0],
                -atom_shifts[source_atom, 1],
                source_layer - atom_shifts[source_atom, 2],
            ),
            dtype=np.int64,
        )
        bulk_cell: tuple[int, int, int]
        for bulk_cell in operator_data.cells:
            transformed_cell: np.ndarray = (
                np.asarray(bulk_cell, dtype=np.int64) @ inverse
            )
            target_surface_cell: np.ndarray = (
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
    result: tuple[tuple[int, int, int], ...] = tuple(sorted(cells))
    return result


def _propagate_position_matrices(  # noqa: PLR0915
    bulk_model: TBModel,
    operator_data: WannierOperatorData,
    spec: SlabSpec,
    slab_centres_cart: Float[Array, "n_orb_slab 3"],
) -> tuple[
    Complex[Array, "n_R n_orb n_orb 3"],
    tuple[tuple[int, int, int], ...],
]:
    """Propagate real-space position matrices with exact cell bookkeeping."""
    if operator_data.position_matrices is None:
        message: str = "position-matrix propagation requires tb data"
        raise ValueError(message)
    missing_shells: dict[
        tuple[int, int, int, int],
        tuple[int, ...],
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
    representation: Complex[Array, "n_orb n_orb"] = (
        jnp.eye(len(bulk_model.basis.n), dtype=jnp.complex128)
        if identity_rotation
        else _orbital_rotation(
            bulk_model,
            spec.surface_cell.rotation,
        )
    )
    orbital_rotated: Complex[Array, "n_R n_orb n_orb 3"] = jnp.einsum(
        "ai,rijc,bj->rabc",
        representation,
        operator_data.position_matrices,
        representation.conj(),
    )
    rotated_position_matrices: Complex[
        Array,
        "n_R n_orb n_orb 3",
    ] = jnp.einsum(
        "rijc,ac->rija",
        orbital_rotated,
        spec.surface_cell.rotation,
    )
    coefficients: np.ndarray = _surface_integer_matrix(spec.surface_cell)
    inverse: np.ndarray = _inverse_integer_matrix(coefficients)
    atom_shifts: np.ndarray
    _, atom_shifts = _base_surface_coordinates(
        bulk_model.geometry,
        inverse,
    )
    slab_to_bulk: tuple[int, ...]
    slab_layers: tuple[int, ...]
    lookup: dict[tuple[int, int], int]
    slab_to_bulk, _, slab_layers, lookup = _orbital_lookup(
        bulk_model,
        spec,
    )
    emitted: list[
        tuple[
            tuple[int, int, int],
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
        source_surface_cell: np.ndarray = np.asarray(
            (
                -atom_shifts[source_atom, 0],
                -atom_shifts[source_atom, 1],
                source_layer - atom_shifts[source_atom, 2],
            ),
            dtype=np.int64,
        )
        cell_index: int
        bulk_cell: tuple[int, int, int]
        for cell_index, bulk_cell in enumerate(operator_data.cells):
            transformed_cell: np.ndarray = (
                np.asarray(bulk_cell, dtype=np.int64) @ inverse
            )
            target_surface_cell: np.ndarray = (
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
                slab_cell: tuple[int, int, int] = (
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
    cells: tuple[tuple[int, int, int], ...] = tuple(
        sorted({record[0] for record in emitted})
    )
    cell_lookup: dict[tuple[int, int, int], int] = {
        cell: index for index, cell in enumerate(cells)
    }
    n_slab_orbitals: int = len(slab_to_bulk)
    matrices: Complex[Array, "n_R n_orb n_orb 3"] = jnp.zeros(
        (len(cells), n_slab_orbitals, n_slab_orbitals, 3),
        dtype=jnp.complex128,
    )
    record: tuple[
        tuple[int, int, int],
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
        vector: Complex[Array, " 3"] = rotated_position_matrices[
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
        current_diagonal: Complex[Array, "n_orb 3"] = matrices[
            zero_cell_index,
            diagonal,
            diagonal,
        ]
        bulk_indices: Array = jnp.asarray(
            slab_to_bulk,
            dtype=jnp.int32,
        )
        original_diagonal: Complex[Array, "n_orb 3"] = (
            rotated_position_matrices[
                bulk_zero_index,
                bulk_indices,
                bulk_indices,
            ]
        )
        translation: Complex[Array, "n_orb 3"] = (
            slab_centres_cart.astype(jnp.complex128) - original_diagonal
        )
        matrices = matrices.at[
            zero_cell_index,
            diagonal,
            diagonal,
        ].set(current_diagonal + translation)
    result: tuple[
        Complex[Array, "n_R n_orb n_orb 3"],
        tuple[tuple[int, int, int], ...],
    ] = (matrices, cells)
    return result


@jaxtyped(typechecker=beartype)
def gen_slab_with_operators(  # noqa: DOC105, PLR0913
    bulk_model: TBModel,
    operator_data: WannierOperatorData,
    miller: tuple[int, int, int],
    thickness_ang: float,
    vacuum_ang: float,
    termination: tuple[str, str] | None = None,
    fine: tuple[float, float] = (0.0, 0.0),
) -> tuple[TBModel, SlabSpec, WannierOperatorData]:
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
    miller : tuple[int, int, int]
        Primitive Miller indices.
    thickness_ang : float
        Nonnegative minimum post-fine material span in Angstrom.
    vacuum_ang : float
        Nonnegative vacuum padding in Angstrom.
    termination : tuple[str, str] or None, optional
        Requested top and bottom species, or ``None`` for a natural cut.
    fine : tuple[float, float], optional
        Static top and bottom inward cut shifts in Angstrom.

    Returns
    -------
    result : tuple[TBModel, SlabSpec, WannierOperatorData]
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
    missing_cells: set[tuple[int, int, int]] = set(
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
    centres_cart: Float[Array, "n_orb_slab 3"] = _slab_operator_centres(
        bulk_model,
        operator_data,
        spec,
    )
    centre_fractional: Float[Array, "n_orb_slab 3"] = (
        centres_cart @ jnp.linalg.inv(slab_model.geometry.lattice)
    )
    centre_fractional = centre_fractional.at[:, :2].set(
        jnp.mod(centre_fractional[:, :2], 1.0)
    )
    operator_depths: Float[Array, " n_orb_slab"] = (
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
    position_matrices: Complex[Array, "n_R n_orb n_orb 3"] | None
    cells: tuple[tuple[int, int, int], ...]
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
    result: tuple[TBModel, SlabSpec, WannierOperatorData] = (
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
