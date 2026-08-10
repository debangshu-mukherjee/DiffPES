r"""Build tight-binding hoppings from Slater--Koster integrals.

Extended Summary
----------------
The two-center kernel implements the Slater--Koster decomposition for
``s``, ``p``, and ``d`` shells in the package-wide real-harmonic order

``s; (p_y, p_z, p_x); (d_xy, d_yz, d_z2, d_xz, d_x2-y2)``.

Rather than differentiating singular Euler angles, the implementation rotates
orthonormal Cartesian vector and symmetric-traceless-tensor representations.
Two smooth rotation charts cover the north and south bond poles. Axial
degeneracy of the pi and delta channels makes the composed block independent
of the chart's residual rotation about the bond. The result is the same
bond-rotation construction as canon Eq. (18), but it retains the analytic
transverse position gradients at both poles.

Routine Listings
----------------
:func:`sk_block`
    Construct a real-harmonic Slater--Koster hopping block.
:func:`neighbor_shells`
    Find unique undirected neighbor bonds at host setup time.
:func:`build_sk_model`
    Build a validated tight-binding model from two-center integrals.

Notes
-----
Neighbor selection is discrete. The selected atom/cell tuples define static
topology. Derivatives remain meaningful while perturbations neither cross the
cutoff nor merge a nonzero bond into zero length. The host setup derives a
complete finite translation bound from the lattice singular values, the basis
diameter, and the cutoff. The builder rejects transformations without concrete
geometry because no neighbor-topology certificate exists.
"""

from collections.abc import Sequence

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from beartype import beartype
from beartype.typing import Dict, Tuple
from jax import core
from jaxtyping import Array, Bool, Complex128, Float64, jaxtyped
from numpy.typing import NDArray

from diffpes.types import (
    CARTESIAN_COMPONENTS,
    CHANNELS_BY_PAIR,
    KNOWN_CHANNELS,
    MAX_SK_ANGULAR_MOMENTUM,
    MIN_BOND_DISTANCE,
    PARAMETER_KEY_PARTS,
    SHELL_ATOLERANCE,
    SHELL_RTOLERANCE,
    SPECIES_PAIR_PARTS,
    CrystalGeometry,
    OrbitalBasis,
    SlaterKosterParams,
    TBModel,
    make_tb_model,
)


def _validate_angular_momentum(l: int, name: str) -> None:  # noqa: E741
    """PRIVATE: Validate one static Slater--Koster shell angular momentum.

    Parameters
    ----------
    l : int
        Candidate static shell angular momentum.
    name : str
        Argument name used in the error message.

    Raises
    ------
    ValueError
        If ``l`` is not an integer from zero through
        ``MAX_SK_ANGULAR_MOMENTUM``.

    Notes
    -----
    The check reads the static Python value, so an unsupported shell
    fails before any block construction. The implemented two-center
    kernel covers s, p, and d shells only.
    """
    if type(l) is not int or not 0 <= l <= MAX_SK_ANGULAR_MOMENTUM:
        message: str = (
            f"{name} must be an integer from 0 through "
            f"{MAX_SK_ANGULAR_MOMENTUM}"
        )
        raise ValueError(message)


def _rotation_z_to_direction(
    direction: Float64[Array, " 3"],
) -> Float64[Array, "3 3"]:
    """PRIVATE: Construct a proper rotation taking positive z onto the bond.

    Parameters
    ----------
    direction : Float64[Array, " 3"]
        Cartesian unit vector along the bond.

    Returns
    -------
    rotation : Float64[Array, "3 3"]
        Proper rotation matrix that maps ``(0, 0, 1)`` onto
        ``direction``.

    Notes
    -----
    Two smooth Rodrigues charts cover the bond sphere. The north chart
    rotates ``+z`` directly onto the direction and is singular only at
    the south pole. The south chart composes a half turn about x with a
    rotation from ``-z`` and is singular only at the north pole.
    ``jnp.where`` selects the chart by the sign of the z component and
    sanitizes the unused denominator. The value and the transverse
    position gradients therefore stay finite at both poles. The charts
    differ by a residual rotation about the bond axis. The axial
    degeneracy of the pi and delta channels makes the composed
    Slater--Koster block independent of that residual gauge.
    """
    dtype: jnp.dtype = direction.dtype
    identity: Float64[Array, "3 3"] = jnp.eye(
        CARTESIAN_COMPONENTS,
        dtype=dtype,
    )
    direction_x: Float64[Array, ""] = direction[0]
    direction_y: Float64[Array, ""] = direction[1]
    direction_z: Float64[Array, ""] = direction[2]
    use_north_chart: Array = direction_z >= 0.0

    north_denominator: Float64[Array, ""] = jnp.where(
        use_north_chart,
        1.0 + direction_z,
        1.0,
    )
    north_cross: Float64[Array, " 3"] = jnp.stack(
        (-direction_y, direction_x, jnp.zeros_like(direction_z))
    )
    north_skew: Float64[Array, "3 3"] = jnp.asarray(
        (
            (0.0, -north_cross[2], north_cross[1]),
            (north_cross[2], 0.0, -north_cross[0]),
            (-north_cross[1], north_cross[0], 0.0),
        ),
        dtype=dtype,
    )
    north_rotation: Float64[Array, "3 3"] = (
        identity + north_skew + north_skew @ north_skew / north_denominator
    )

    south_denominator: Float64[Array, ""] = jnp.where(
        use_north_chart,
        1.0,
        1.0 - direction_z,
    )
    south_cross: Float64[Array, " 3"] = jnp.stack(
        (direction_y, -direction_x, jnp.zeros_like(direction_z))
    )
    south_skew: Float64[Array, "3 3"] = jnp.asarray(
        (
            (0.0, -south_cross[2], south_cross[1]),
            (south_cross[2], 0.0, -south_cross[0]),
            (-south_cross[1], south_cross[0], 0.0),
        ),
        dtype=dtype,
    )
    half_turn_x: Float64[Array, "3 3"] = jnp.diag(
        jnp.asarray((1.0, -1.0, -1.0), dtype=dtype)
    )
    from_negative_z: Float64[Array, "3 3"] = (
        identity + south_skew + south_skew @ south_skew / south_denominator
    )
    south_rotation: Float64[Array, "3 3"] = from_negative_z @ half_turn_x
    rotation: Float64[Array, "3 3"] = jnp.where(
        use_north_chart,
        north_rotation,
        south_rotation,
    )
    return rotation


def _p_axes(dtype: jnp.dtype) -> Float64[Array, "3 3"]:
    """PRIVATE: Return Cartesian unit vectors in real-harmonic p-shell order.

    Parameters
    ----------
    dtype : jnp.dtype
        Floating dtype for the returned constant.

    Returns
    -------
    axes : Float64[Array, "3 3"]
        Rows ``(y, z, x)`` matching the package p-orbital order
        ``(p_y, p_z, p_x)``.

    Notes
    -----
    The rows are the change of basis between the Cartesian vector
    representation and the real p orbitals. A Cartesian rotation ``R``
    therefore acts on the shell as ``axes @ R @ axes.T``.
    """
    axes: Float64[Array, "3 3"] = jnp.asarray(
        (
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
            (1.0, 0.0, 0.0),
        ),
        dtype=dtype,
    )
    return axes


def _d_tensors(dtype: jnp.dtype) -> Float64[Array, "5 3 3"]:
    """PRIVATE: Return orthonormal traceless tensors in real d-shell order.

    Parameters
    ----------
    dtype : jnp.dtype
        Floating dtype for the returned constant.

    Returns
    -------
    tensors : Float64[Array, "5 3 3"]
        Symmetric traceless tensors for the package d-orbital order
        ``(d_xy, d_yz, d_z2, d_xz, d_x2-y2)``, orthonormal under the
        Frobenius inner product.

    Notes
    -----
    The ``1/sqrt(2)`` and ``1/sqrt(6)`` factors normalize every tensor
    to unit Frobenius norm. A Cartesian rotation acts on the d shell
    through the congruence ``R T R.T`` of each tensor followed by
    projection back onto this basis.
    """
    inverse_sqrt_two: float = 1.0 / np.sqrt(2.0)
    inverse_sqrt_six: float = 1.0 / np.sqrt(6.0)
    tensors: Float64[Array, "5 3 3"] = jnp.asarray(
        (
            (
                (0.0, inverse_sqrt_two, 0.0),
                (inverse_sqrt_two, 0.0, 0.0),
                (0.0, 0.0, 0.0),
            ),
            (
                (0.0, 0.0, 0.0),
                (0.0, 0.0, inverse_sqrt_two),
                (0.0, inverse_sqrt_two, 0.0),
            ),
            (
                (-inverse_sqrt_six, 0.0, 0.0),
                (0.0, -inverse_sqrt_six, 0.0),
                (0.0, 0.0, 2.0 * inverse_sqrt_six),
            ),
            (
                (0.0, 0.0, inverse_sqrt_two),
                (0.0, 0.0, 0.0),
                (inverse_sqrt_two, 0.0, 0.0),
            ),
            (
                (inverse_sqrt_two, 0.0, 0.0),
                (0.0, -inverse_sqrt_two, 0.0),
                (0.0, 0.0, 0.0),
            ),
        ),
        dtype=dtype,
    )
    return tensors


def _orbital_rotation(
    l: int,  # noqa: E741
    rotation: Float64[Array, "3 3"],
) -> Float64[Array, "m1 m2"]:
    """PRIVATE: Represent a Cartesian rotation in a real s, p, or d shell.

    Parameters
    ----------
    l : int
        Static shell angular momentum: ``0``, ``1``, or ``2``.
    rotation : Float64[Array, "3 3"]
        Proper Cartesian rotation matrix.

    Returns
    -------
    representation : Float64[Array, "m1 m2"]
        Orthogonal ``(2*l + 1)``-dimensional representation of the
        rotation in the package real-harmonic order.

    Notes
    -----
    The s shell returns the trivial one-by-one identity. The p shell
    conjugates the rotation with the real-harmonic axis rows. The d
    shell rotates every orthonormal traceless tensor by congruence and
    projects the result back onto the tensor basis. Rotating vector and
    tensor representations avoids differentiating singular Euler angles.
    """
    if l == 0:
        scalar: Float64[Array, "1 1"] = jnp.ones(
            (1, 1),
            dtype=rotation.dtype,
        )
        return scalar
    if l == 1:
        axes: Float64[Array, "3 3"] = _p_axes(rotation.dtype)
        vector_representation: Float64[Array, "3 3"] = axes @ rotation @ axes.T
        return vector_representation

    tensors: Float64[Array, "5 3 3"] = _d_tensors(rotation.dtype)
    rotated_tensors: Float64[Array, "5 3 3"] = jnp.einsum(
        "ik,akl,jl->aij",
        rotation,
        tensors,
        rotation,
    )
    tensor_representation: Float64[Array, "5 5"] = jnp.einsum(
        "bij,aij->ba",
        tensors,
        rotated_tensors,
    )
    return tensor_representation


@jaxtyped(typechecker=beartype)
def sk_block(  # noqa: DOC502, DOC503
    l1: int,
    l2: int,
    v_llm: Float64[Array, " n_m"],
    bond_cart: Float64[Array, " 3"],
) -> Float64[Array, "m1 m2"]:
    r"""Construct a real-harmonic Slater--Koster hopping block.

    Rotate the bond-axis sigma, pi, and delta channels into the declared
    laboratory-frame real-harmonic order.

    :see: :class:`~.test_slaterkoster.TestSkBlock`

    Parameters
    ----------
    l1 : int
        Angular momentum of the row shell: ``0`` (s), ``1`` (p), or ``2``
        (d).
    l2 : int
        Angular momentum of the column shell, with the same range.
    v_llm : Float64[Array, " n_m"]
        Fundamental integrals ordered by ``abs(m)``: sigma, pi, then delta.
        Its length must be ``min(l1, l2) + 1``.
    bond_cart : Float64[Array, " 3"]
        Nonzero Cartesian displacement from the row-shell atom to the
        column-shell atom.

    Returns
    -------
    block : Float64[Array, "m1 m2"]
        Hopping matrix with shape ``(2*l1 + 1, 2*l2 + 1)`` in the declared
        real-harmonic order.

    Raises
    ------
    ValueError
        If an angular momentum lies outside the implemented s/p/d range or
        the integral vector has the wrong rank or length.
    EquinoxRuntimeError
        If the bond is zero/non-finite or an integral is non-finite.

    Notes
    -----
    A bond-axis matrix couples equal magnetic numbers with
    ``v_llm[abs(m)]``. Cartesian vector/tensor representations rotate it into
    the laboratory real-harmonic basis. For a swapped shell order, the
    radial parity convention
    :math:`V_{l_2l_1m}=(-1)^{l_1+l_2}V_{l_1l_2m}` is applied automatically
    when ``l1 > l2``.
    """
    _validate_angular_momentum(l1, "l1")
    _validate_angular_momentum(l2, "l2")
    if v_llm.ndim != 1:
        message: str = "v_llm must be one-dimensional"
        raise ValueError(message)
    expected_integrals: int = min(l1, l2) + 1
    if v_llm.shape[0] != expected_integrals:
        message = (
            f"v_llm length must equal min(l1, l2) + 1 ({expected_integrals})"
        )
        raise ValueError(message)

    values: Float64[Array, " n_m"] = eqx.error_if(
        v_llm,
        ~jnp.all(jnp.isfinite(v_llm)),
        "sk_block: integrals finite",
    )
    bond: Float64[Array, " 3"] = eqx.error_if(
        bond_cart,
        ~jnp.all(jnp.isfinite(bond_cart)),
        "sk_block: bond finite",
    )
    bond_norm: Float64[Array, ""] = jnp.linalg.norm(bond)
    bond = eqx.error_if(
        bond,
        ~(bond_norm > 0.0),
        "sk_block: bond nonzero",
    )
    direction: Float64[Array, " 3"] = bond / bond_norm
    rotation: Float64[Array, "3 3"] = _rotation_z_to_direction(direction)
    left_rotation: Float64[Array, "m1 m1"] = _orbital_rotation(l1, rotation)
    right_rotation: Float64[Array, "m2 m2"] = _orbital_rotation(l2, rotation)

    bond_axis: Float64[Array, "m1 m2"] = jnp.zeros(
        (2 * l1 + 1, 2 * l2 + 1),
        dtype=values.dtype,
    )
    magnetic_number: int
    for magnetic_number in range(-min(l1, l2), min(l1, l2) + 1):
        bond_axis = bond_axis.at[
            magnetic_number + l1,
            magnetic_number + l2,
        ].set(values[abs(magnetic_number)])

    radial_parity: int = (-1) ** (l1 + l2) if l1 > l2 else 1
    block: Float64[Array, "m1 m2"] = (
        radial_parity * left_rotation @ bond_axis @ right_rotation.T
    )
    return block


def _candidate_topology(
    n_atoms: int,
    supercell_radius: int,
) -> Tuple[
    Tuple[Tuple[int, int], ...],
    Tuple[Tuple[int, int, int], ...],
]:
    """PRIVATE: Build one canonical representative of every undirected bond.

    Parameters
    ----------
    n_atoms : int
        Number of atoms in the home cell.
    supercell_radius : int
        Inclusive integer search radius along every lattice direction.

    Returns
    -------
    topology : tuple
        Canonical ordered atom pairs and their exact integer cell
        translations, as two parallel static tuples.

    Notes
    -----
    The loop enumerates every atom pair in every cell of the
    ``(2*radius + 1)**3`` cube and skips the self-bond in the home cell.
    It keeps a record only when the record precedes its reverse
    ``(j, i, -R)`` lexicographically. Every undirected bond therefore
    appears exactly once.
    """
    atom_pairs: list[Tuple[int, int]] = []
    cells: list[Tuple[int, int, int]] = []
    cell_x: int
    cell_y: int
    cell_z: int
    atom_i: int
    atom_j: int
    for cell_x in range(-supercell_radius, supercell_radius + 1):
        for cell_y in range(-supercell_radius, supercell_radius + 1):
            for cell_z in range(-supercell_radius, supercell_radius + 1):
                cell: Tuple[int, int, int] = (cell_x, cell_y, cell_z)
                for atom_i in range(n_atoms):
                    for atom_j in range(n_atoms):
                        if atom_i == atom_j and cell == (0, 0, 0):
                            continue
                        record: Tuple[int, int, int, int, int] = (
                            atom_i,
                            atom_j,
                            cell_x,
                            cell_y,
                            cell_z,
                        )
                        reverse: Tuple[int, int, int, int, int] = (
                            atom_j,
                            atom_i,
                            -cell_x,
                            -cell_y,
                            -cell_z,
                        )
                        if record < reverse:
                            atom_pairs.append((atom_i, atom_j))
                            cells.append(cell)
    topology: Tuple[
        Tuple[Tuple[int, int], ...],
        Tuple[Tuple[int, int, int], ...],
    ] = (tuple(atom_pairs), tuple(cells))
    return topology


def _certified_supercell_radius(
    geometry: CrystalGeometry,
    cutoff: float,
) -> int:
    r"""PRIVATE: Return a complete integer-translation search radius.

    Parameters
    ----------
    geometry : CrystalGeometry
        Concrete crystal lattice and fractional atom positions.
    cutoff : float
        Positive inclusive Cartesian distance cutoff in angstroms.

    Returns
    -------
    radius : int
        Cube radius that contains every integer translation able to
        carry a retained bond.

    Raises
    ------
    ValueError
        If the lattice is singular or non-finite, or the derived bound
        is not finite.

    Notes
    -----
    If ``A`` stores lattice vectors as rows, a retained displacement
    obeys ``||(n + delta) A|| <= cutoff``. Therefore

    ``||n|| <= (cutoff + ||delta A||) / sigma_min(A)``.

    Maximizing the second term over all basis pairs gives one cube
    radius that contains every possible retained translation. The
    outward floating-point rounding keeps the host certificate
    conservative.
    """
    lattice: Float64[NDArray, "3 3"] = np.asarray(
        geometry.lattice, dtype=np.float64
    )
    positions: Float64[NDArray, "n_atom 3"] = np.asarray(
        geometry.positions, dtype=np.float64
    )
    singular_values: Float64[NDArray, " 3"] = np.linalg.svd(
        lattice,
        compute_uv=False,
    )
    sigma_min: float = float(singular_values[-1])
    if not np.isfinite(sigma_min) or sigma_min <= 0.0:
        message: str = (
            "neighbor-shell search requires a finite nonsingular lattice"
        )
        raise ValueError(message)

    if positions.shape[0] <= 1:
        basis_diameter: float = 0.0
    else:
        fractional_pairs: Float64[NDArray, "n_atom n_atom 3"] = (
            positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
        )
        cartesian_pairs: Float64[NDArray, "n_atom n_atom 3"] = (
            fractional_pairs @ lattice
        )
        basis_diameter = float(
            np.max(np.linalg.norm(cartesian_pairs, axis=-1))
        )
    bound: float = (cutoff + basis_diameter) / sigma_min
    if not np.isfinite(bound):
        message = "neighbor-shell search bound must be finite"
        raise ValueError(message)
    conservative_bound: float = float(np.nextafter(bound, np.inf))
    radius: int = int(np.ceil(conservative_bound))
    return radius


def _displacements_and_distances(
    geometry: CrystalGeometry,
    atom_pairs: Tuple[Tuple[int, int], ...],
    cells: Tuple[Tuple[int, int, int], ...],
) -> Tuple[Float64[Array, "n_bond 3"], Float64[Array, " n_bond"]]:
    """PRIVATE: Derive differentiable fractional bonds and Cartesian lengths.

    Parameters
    ----------
    geometry : CrystalGeometry
        Crystal lattice and fractional atom positions.
    atom_pairs : Tuple[Tuple[int, int], ...]
        Static ordered atom pairs.
    cells : Tuple[Tuple[int, int, int], ...]
        Exact integer cell translations for every pair.

    Returns
    -------
    result : Tuple[Float64[Array, "n_bond 3"], Float64[Array, " n_bond"]]
        Fractional displacements ``R + tau_j - tau_i`` and their
        Cartesian lengths in angstroms.

    Notes
    -----
    The static tuples only index; the arithmetic runs on the traced
    geometry, so displacements and distances carry position and lattice
    derivatives. An empty pair list returns empty arrays of the correct
    shape and dtype.
    """
    if not atom_pairs:
        empty_displacements: Float64[Array, "0 3"] = jnp.zeros(
            (0, CARTESIAN_COMPONENTS),
            dtype=geometry.positions.dtype,
        )
        empty_distances: Float64[Array, " 0"] = jnp.zeros(
            (0,),
            dtype=geometry.positions.dtype,
        )
        empty_result: Tuple[
            Float64[Array, "0 3"],
            Float64[Array, " 0"],
        ] = (empty_displacements, empty_distances)
        return empty_result
    atom_i: Array = jnp.asarray(
        tuple(pair[0] for pair in atom_pairs),
        dtype=jnp.int32,
    )
    atom_j: Array = jnp.asarray(
        tuple(pair[1] for pair in atom_pairs),
        dtype=jnp.int32,
    )
    cell_array: Float64[Array, "n_bond 3"] = jnp.asarray(
        cells,
        dtype=geometry.positions.dtype,
    )
    displacements: Float64[Array, "n_bond 3"] = (
        cell_array + geometry.positions[atom_j] - geometry.positions[atom_i]
    )
    cartesian: Float64[Array, "n_bond 3"] = displacements @ geometry.lattice
    distances: Float64[Array, " n_bond"] = jnp.linalg.norm(
        cartesian,
        axis=1,
    )
    result: Tuple[
        Float64[Array, "n_bond 3"],
        Float64[Array, " n_bond"],
    ] = (displacements, distances)
    return result


def _geometry_is_traced(geometry: CrystalGeometry) -> bool:
    """PRIVATE: Detect JAX tracers in the topology-defining geometry.

    Parameters
    ----------
    geometry : CrystalGeometry
        Geometry whose lattice and positions the check inspects.

    Returns
    -------
    traced : bool
        ``True`` when the positions or the lattice is a JAX tracer.

    Notes
    -----
    Host neighbor selection cannot run on tracers. Callers use this test
    to reject traced geometry or to fall back to a concrete primal.
    """
    traced: bool = isinstance(geometry.positions, core.Tracer) or isinstance(
        geometry.lattice, core.Tracer
    )
    return traced


def _concrete_primal(value: Array) -> Array | None:
    """PRIVATE: Recover the concrete primal carried by an eager AD tracer.

    Parameters
    ----------
    value : Array
        Possibly traced array leaf.

    Returns
    -------
    primal : Array | None
        Concrete array after unwrapping every nested ``primal``
        attribute. ``None`` signals that a tracer level carries no
        primal or that the chain does not end in a concrete array.

    Notes
    -----
    Eager forward- and reverse-mode AD tracers stack ``primal``
    attributes; the loop unwraps them until a concrete
    :class:`jax.Array` appears. Tracers made by :func:`jax.jit` carry no
    ``primal`` attribute, so compiled tracing correctly reports no
    concrete value.
    """
    candidate: object = value
    while isinstance(candidate, core.Tracer):
        if not hasattr(candidate, "primal"):
            missing: Array | None = None
            return missing
        candidate = candidate.primal
    if isinstance(candidate, jax.Array):
        primal: Array | None = candidate
        return primal
    missing = None
    return missing  # noqa: RET504 -- repository returns bound names.


def _primal_geometry(
    geometry: CrystalGeometry,
) -> CrystalGeometry | None:
    """PRIVATE: Recover topology values from eager forward/reverse tracers.

    Parameters
    ----------
    geometry : CrystalGeometry
        Geometry whose lattice or positions may be eager AD tracers.

    Returns
    -------
    primal : CrystalGeometry | None
        Copy of the geometry with concrete lattice and positions, or
        ``None`` when either leaf has no concrete primal.

    Notes
    -----
    :func:`equinox.tree_at` swaps both leaves at once and preserves all
    other geometry metadata. Host neighbor certification then runs on
    the concrete primal while derivatives keep flowing through the
    original traced leaves.
    """
    primal_lattice: Array | None = _concrete_primal(geometry.lattice)
    primal_positions: Array | None = _concrete_primal(geometry.positions)
    if primal_lattice is None or primal_positions is None:
        missing: CrystalGeometry | None = None
        return missing
    primal: CrystalGeometry = eqx.tree_at(
        lambda item: (item.lattice, item.positions),
        geometry,
        (primal_lattice, primal_positions),
    )
    return primal


@jaxtyped(typechecker=beartype)
def neighbor_shells(  # noqa: DOC502
    geometry: CrystalGeometry,
    cutoff: float,
    supercell_radius: int | None = None,
) -> Tuple[
    Tuple[Tuple[int, int], ...],
    Tuple[Tuple[int, int, int], ...],
    Float64[Array, "n_bond 3"],
    Float64[Array, " n_bond"],
]:
    """Find unique undirected neighbor bonds at host setup time.

    Enumerate a finite translation supercell and retain one canonical record
    for every bond inside the inclusive Cartesian cutoff.

    :see: :class:`~.test_slaterkoster.TestNeighborShells`

    Parameters
    ----------
    geometry : CrystalGeometry
        Crystal lattice and fractional atom positions.
    cutoff : float
        Positive inclusive Cartesian distance cutoff in angstroms.
    supercell_radius : int | None, optional
        Requested number of translated cells in each positive and negative
        lattice direction. ``None`` uses the certified complete radius.
        The function rejects explicit radii smaller than the certificate.

    Returns
    -------
    atom_pairs : Tuple[Tuple[int, int], ...]
        Canonical undirected atom pairs. Every physical bond appears once.
    cells : Tuple[Tuple[int, int, int], ...]
        Exact integer cell translations from the first atom to the second.
    displacements : Float64[Array, "n_bond 3"]
        Fractional displacements ``R + tau_j - tau_i``.
    distances : Float64[Array, " n_bond"]
        Cartesian bond lengths in angstroms.

    Raises
    ------
    ValueError
        If the cutoff or radius fails validation, an explicit radius lacks
        completeness, tracing reaches this host routine, or distinct atoms
        produce a zero-length bond.

    Notes
    -----
    NumPy selects the tuples on the host, making their topology static. JAX
    re-derives displacements and distances from those tuples. The returned
    arrays retain local derivatives after topology freezing.
    :func:`build_sk_model` adds reverse directed hopping records later.
    """
    if not np.isfinite(cutoff) or cutoff <= 0.0:
        message: str = "cutoff must be a positive finite float"
        raise ValueError(message)
    if supercell_radius is not None and (
        type(supercell_radius) is not int or supercell_radius < 0
    ):
        message = "supercell_radius must be a non-negative integer"
        raise ValueError(message)
    if _geometry_is_traced(geometry):
        message = (
            "neighbor_shells is a host setup operation; freeze topology "
            "before tracing geometry"
        )
        raise ValueError(message)

    certified_radius: int = _certified_supercell_radius(geometry, cutoff)
    if supercell_radius is not None and supercell_radius < certified_radius:
        message = (
            f"supercell_radius={supercell_radius} is incomplete; "
            f"certified minimum is {certified_radius}"
        )
        raise ValueError(message)
    search_radius: int = (
        certified_radius if supercell_radius is None else supercell_radius
    )
    candidates: Tuple[Tuple[int, int], ...]
    candidate_cells: Tuple[Tuple[int, int, int], ...]
    candidates, candidate_cells = _candidate_topology(
        geometry.positions.shape[0],
        search_radius,
    )
    with jax.ensure_compile_time_eval():
        candidate_displacements: Float64[Array, "n_candidate 3"]
        candidate_distances: Float64[Array, " n_candidate"]
        candidate_displacements, candidate_distances = (
            _displacements_and_distances(
                geometry,
                candidates,
                candidate_cells,
            )
        )
        host_distances: Float64[NDArray, " n_candidate"] = np.asarray(
            candidate_distances
        )
    if np.any(host_distances <= MIN_BOND_DISTANCE):
        message = "neighbor_shells encountered a zero-length atom pair"
        raise ValueError(message)
    keep: Bool[NDArray, " n_candidate"] = host_distances <= cutoff
    kept_indices: Tuple[int, ...] = tuple(
        int(index) for index in np.flatnonzero(keep)
    )
    atom_pairs: Tuple[Tuple[int, int], ...] = tuple(
        candidates[index] for index in kept_indices
    )
    cells: Tuple[Tuple[int, int, int], ...] = tuple(
        candidate_cells[index] for index in kept_indices
    )
    displacements: Float64[Array, "n_bond 3"]
    distances: Float64[Array, " n_bond"]
    displacements, distances = _displacements_and_distances(
        geometry,
        atom_pairs,
        cells,
    )
    result: Tuple[
        Tuple[Tuple[int, int], ...],
        Tuple[Tuple[int, int, int], ...],
        Float64[Array, "n_bond 3"],
        Float64[Array, " n_bond"],
    ] = (atom_pairs, cells, displacements, distances)
    return result


def _parse_parameter_keys(
    keys: Tuple[str, ...],
) -> Dict[Tuple[str | None, int | None, str], int]:
    """PRIVATE: Parse material, optional shell, and channel identifiers.

    Parameters
    ----------
    keys : Tuple[str, ...]
        Static Slater--Koster parameter keys.

    Returns
    -------
    parsed : Dict[Tuple[str | None, int | None, str], int]
        Map from ``(species_pair, shell, channel)`` to the position of
        the key in ``keys``. Generic entries store ``None`` for the
        missing parts.

    Raises
    ------
    ValueError
        If a shell selector is not a positive decimal integer or a
        species pair does not hold exactly two nonempty names. Also
        raised when the key grammar is invalid or the channel is
        unknown.

    Notes
    -----
    Supported grammars are ``"<channel>"``, ``"<A>-<B>:<channel>"``, and
    the one-based distance-shell form ``"<A>-<B>@<shell>:<channel>"``.
    Channels must belong to the types-owned ``KNOWN_CHANNELS`` set.
    """
    parsed: Dict[Tuple[str | None, int | None, str], int] = {}
    index: int
    key: str
    for index, key in enumerate(keys):
        pieces: list[str] = key.split(":")
        pair: str | None
        shell: int | None = None
        channel: str
        if len(pieces) == 1:
            pair = None
            channel = pieces[0]
        elif len(pieces) == PARAMETER_KEY_PARTS:
            pair, channel = pieces
            if "@" in pair:
                shell_text: str
                pair, shell_text = pair.rsplit("@", maxsplit=1)
                if not shell_text.isdecimal() or int(shell_text) < 1:
                    message: str = (
                        f"invalid neighbor-shell selector in SK key {key!r}"
                    )
                    raise ValueError(message)
                shell = int(shell_text)
            species: list[str] = pair.split("-")
            if len(species) != SPECIES_PAIR_PARTS or not all(species):
                message = f"invalid species pair in SK key {key!r}"
                raise ValueError(message)
        else:
            message = f"invalid SK key grammar {key!r}"
            raise ValueError(message)
        if channel not in KNOWN_CHANNELS:
            message = f"unknown Slater--Koster channel {channel!r}"
            raise ValueError(message)
        parsed[(pair, shell, channel)] = index
    return parsed


def _species_pair(
    geometry: CrystalGeometry,
    atom_pair: Tuple[int, int],
) -> Tuple[str | None, str | None]:
    """PRIVATE: Return forward and reversed material-pair identifiers.

    Parameters
    ----------
    geometry : CrystalGeometry
        Geometry that may carry species labels.
    atom_pair : Tuple[int, int]
        Static ordered atom pair.

    Returns
    -------
    pairs : Tuple[str | None, str | None]
        Strings ``"A-B"`` and ``"B-A"`` for the two atom species, or
        ``(None, None)`` when the geometry declares no species.

    Notes
    -----
    The function returns both orders because parameter lookup accepts a
    key written for either direction of the same bond.
    """
    if not geometry.species:
        empty_pairs: Tuple[str | None, str | None] = (None, None)
        return empty_pairs
    species_i: str = geometry.species[atom_pair[0]]
    species_j: str = geometry.species[atom_pair[1]]
    pairs: Tuple[str | None, str | None] = (
        f"{species_i}-{species_j}",
        f"{species_j}-{species_i}",
    )
    return pairs


def _shell_numbers(
    geometry: CrystalGeometry,
    atom_pairs: Tuple[Tuple[int, int], ...],
    distances: Float64[Array, " n_bond"],
) -> Tuple[int, ...]:
    """PRIVATE: Create one-based distance-shell numbers per species pair.

    Parameters
    ----------
    geometry : CrystalGeometry
        Geometry that may carry species labels.
    atom_pairs : Tuple[Tuple[int, int], ...]
        Static ordered atom pairs.
    distances : Float64[Array, " n_bond"]
        Cartesian bond lengths in angstroms.

    Returns
    -------
    result : Tuple[int, ...]
        One-based distance-shell number for every bond.

    Notes
    -----
    Bonds group by their unordered species pair; without species labels
    one shared group applies. Within a group, sorted host distances
    merge into shells when they agree within ``SHELL_RTOLERANCE`` and
    ``SHELL_ATOLERANCE``, and each bond takes the first matching shell.
    The result is static host metadata: it certifies the topology and
    never enters tracing.
    """
    host_distances: Float64[NDArray, " n_bond"] = np.asarray(distances)
    grouped: Dict[Tuple[str, str], list[float]] = {}
    pair_groups: list[Tuple[str, str]] = []
    atom_pair: Tuple[int, int]
    distance: np.float64
    for atom_pair in atom_pairs:
        if geometry.species:
            group: Tuple[str, str] = tuple(
                sorted(
                    (
                        geometry.species[atom_pair[0]],
                        geometry.species[atom_pair[1]],
                    )
                )
            )
        else:
            group = ("", "")
        pair_groups.append(group)
        grouped.setdefault(group, [])

    group: Tuple[str, str]
    values: list[float]
    for group in grouped:
        group_distances: list[float] = sorted(
            float(distance)
            for distance, candidate_group in zip(
                host_distances,
                pair_groups,
                strict=True,
            )
            if candidate_group == group
        )
        values = []
        for distance in group_distances:
            if not values or not np.isclose(
                distance,
                values[-1],
                rtol=SHELL_RTOLERANCE,
                atol=SHELL_ATOLERANCE,
            ):
                values.append(distance)
        grouped[group] = values

    shell_numbers: list[int] = []
    for distance, group in zip(
        host_distances,
        pair_groups,
        strict=True,
    ):
        values = grouped[group]
        matching: list[int] = [
            index
            for index, reference in enumerate(values)
            if np.isclose(
                distance,
                reference,
                rtol=SHELL_RTOLERANCE,
                atol=SHELL_ATOLERANCE,
            )
        ]
        shell_numbers.append(matching[0] + 1)
    result: Tuple[int, ...] = tuple(shell_numbers)
    return result


def _parameter_index(
    lookup: Dict[Tuple[str | None, int | None, str], int],
    forward_pair: str | None,
    reverse_pair: str | None,
    shell: int,
    channel: str,
) -> int | None:
    """PRIVATE: Resolve one integral with specific-to-generic precedence.

    Parameters
    ----------
    lookup : Dict[Tuple[str | None, int | None, str], int]
        Parsed key map from ``_parse_parameter_keys``.
    forward_pair : str | None
        Species pair ``"A-B"`` in bond order, or ``None``.
    reverse_pair : str | None
        Reversed species pair ``"B-A"``, or ``None``.
    shell : int
        One-based distance-shell number of the bond.
    channel : str
        Slater--Koster channel name such as ``"pp_pi"``.

    Returns
    -------
    index : int | None
        Position of the matched key, or ``None`` when no candidate
        matches.

    Notes
    -----
    The lookup tries candidates in a fixed order: forward pair with
    shell, reverse pair with shell, then both pairs without shell. The
    generic channel-only key comes last, and the first hit wins.
    """
    candidates: Sequence[Tuple[str | None, int | None, str]] = (
        (forward_pair, shell, channel),
        (reverse_pair, shell, channel),
        (forward_pair, None, channel),
        (reverse_pair, None, channel),
        (None, None, channel),
    )
    candidate: Tuple[str | None, int | None, str]
    for candidate in candidates:
        if candidate in lookup:
            index: int | None = lookup[candidate]
            return index
    missing: int | None = None
    return missing


def _integral_vector(
    sk_params: SlaterKosterParams,
    lookup: Dict[Tuple[str | None, int | None, str], int],
    forward_pair: str | None,
    reverse_pair: str | None,
    shell: int,
    channels: Tuple[str, ...],
) -> Tuple[Float64[Array, " n_m"], bool]:
    """PRIVATE: Collect channel values, treating omitted channels as zero.

    Parameters
    ----------
    sk_params : SlaterKosterParams
        Differentiable fundamental integral values and static keys.
    lookup : Dict[Tuple[str | None, int | None, str], int]
        Parsed key map from ``_parse_parameter_keys``.
    forward_pair : str | None
        Species pair in bond order, or ``None``.
    reverse_pair : str | None
        Reversed species pair, or ``None``.
    shell : int
        One-based distance-shell number of the bond.
    channels : Tuple[str, ...]
        Sigma, pi, and delta channel names for the angular pair.

    Returns
    -------
    result : Tuple[Float64[Array, " n_m"], bool]
        Stacked integral vector in eV ordered as the channels, and a
        flag that is ``True`` when at least one channel has a key.

    Notes
    -----
    Omitted channels contribute exact zeros, so a sparse parameter set
    stays valid. The flag lets the builder skip orbital pairs whose
    every channel is absent instead of materializing zero hoppings.
    """
    values: list[Float64[Array, ""]] = []
    found_any: bool = False
    channel: str
    for channel in channels:
        index: int | None = _parameter_index(
            lookup,
            forward_pair,
            reverse_pair,
            shell,
            channel,
        )
        if index is None:
            values.append(jnp.zeros((), dtype=sk_params.values.dtype))
        else:
            values.append(sk_params.values[index])
            found_any = True
    vector: Float64[Array, " n_m"] = jnp.stack(values)
    result: Tuple[Float64[Array, " n_m"], bool] = (vector, found_any)
    return result


def _freeze_neighbor_topology(
    geometry: CrystalGeometry,
    cutoff: float,
) -> Tuple[
    Tuple[Tuple[int, int], ...],
    Tuple[Tuple[int, int, int], ...],
    Tuple[int, ...],
]:
    """PRIVATE: Certify and freeze atom pairs, cells, and distance shells.

    Parameters
    ----------
    geometry : CrystalGeometry
        Concrete geometry that defines the neighbor topology.
    cutoff : float
        Positive inclusive neighbor cutoff in angstroms.

    Returns
    -------
    result : tuple
        Canonical atom pairs, exact integer cells, and one-based
        distance-shell numbers, as three parallel static tuples.

    Notes
    -----
    :func:`neighbor_shells` performs the certified host search. Distance
    evaluation for shell numbering runs under
    :func:`jax.ensure_compile_time_eval`, so freezing works inside
    traced callers while the shell metadata stays static. Rebuilding
    closures reuse this frozen topology and never repeat discrete
    neighbor selection.
    """
    atom_pairs: Tuple[Tuple[int, int], ...]
    cells: Tuple[Tuple[int, int, int], ...]
    atom_pairs, cells, _, _ = neighbor_shells(geometry, cutoff)
    with jax.ensure_compile_time_eval():
        distances: Float64[Array, " n_bond"]
        _, distances = _displacements_and_distances(
            geometry,
            atom_pairs,
            cells,
        )
        shell_numbers: Tuple[int, ...] = _shell_numbers(
            geometry,
            atom_pairs,
            distances,
        )
    result: Tuple[
        Tuple[Tuple[int, int], ...],
        Tuple[Tuple[int, int, int], ...],
        Tuple[int, ...],
    ] = (atom_pairs, cells, shell_numbers)
    return result


def _build_sk_model_from_topology(  # noqa: PLR0913, PLR0915
    geometry: CrystalGeometry,
    basis: OrbitalBasis,
    sk_params: SlaterKosterParams,
    onsite_energies: Float64[Array, " n_orb"],
    soc_lambdas: Float64[Array, " n_shells"],
    shell_index: Tuple[int, ...],
    atom_pairs: Tuple[Tuple[int, int], ...],
    cells: Tuple[Tuple[int, int, int], ...],
    shell_numbers: Tuple[int, ...],
    *,
    spinor: bool,
) -> TBModel:
    """PRIVATE: Assemble a model on one previously certified static topology.

    Parameters
    ----------
    geometry : CrystalGeometry
        Possibly traced geometry used for differentiable bond vectors.
    basis : OrbitalBasis
        Orbital metadata in package real-harmonic order.
    sk_params : SlaterKosterParams
        Differentiable fundamental two-center integrals in eV.
    onsite_energies : Float64[Array, " n_orb"]
        Onsite orbital energies in eV.
    soc_lambdas : Float64[Array, " n_shells"]
        Atomic spin--orbit strengths in eV.
    shell_index : Tuple[int, ...]
        Orbital-to-SOC-shell map passed to the model factory.
    atom_pairs : Tuple[Tuple[int, int], ...]
        Certified canonical atom pairs.
    cells : Tuple[Tuple[int, int, int], ...]
        Certified exact integer cell translations.
    shell_numbers : Tuple[int, ...]
        Certified one-based distance-shell numbers.
    spinor : bool
        Whether the basis carries explicit spin channels.

    Returns
    -------
    model : TBModel
        Validated model with explicit conjugate reverse hopping records.

    Notes
    -----
    For every certified bond the builder evaluates one Slater--Koster
    block per angular pair and reads the element addressed by the two
    orbital magnetic numbers. Spinor bases keep hoppings spin diagonal.
    The builder skips entirely any orbital pair whose channels have no
    key. It emits every retained amplitude twice: the forward record and
    the conjugated reverse record on the negated cell. The model
    therefore reaches the factory exactly Hermitian-closed. Bond vectors
    derive from the traced geometry, so amplitudes keep position and
    lattice derivatives on the frozen topology.
    """
    lookup: Dict[Tuple[str | None, int | None, str], int] = (
        _parse_parameter_keys(sk_params.keys)
    )
    displacements: Float64[Array, "n_bond 3"]
    displacements, _ = _displacements_and_distances(
        geometry,
        atom_pairs,
        cells,
    )
    orbitals_by_atom: Tuple[Tuple[int, ...], ...] = tuple(
        tuple(
            orbital
            for orbital, atom in enumerate(basis.atom_indices)
            if atom == atom_index
        )
        for atom_index in range(geometry.positions.shape[0])
    )
    amplitudes: list[Complex128[Array, ""]] = []
    hopping_pairs: list[Tuple[int, int]] = []
    hopping_cells: list[Tuple[int, int, int]] = []
    bond_index: int
    atom_pair: Tuple[int, int]
    cell: Tuple[int, int, int]
    for bond_index, (atom_pair, cell) in enumerate(
        zip(atom_pairs, cells, strict=True)
    ):
        forward_pair: str | None
        reverse_pair: str | None
        forward_pair, reverse_pair = _species_pair(geometry, atom_pair)
        cartesian_bond: Float64[Array, " 3"] = (
            displacements[bond_index] @ geometry.lattice
        )
        block_cache: Dict[Tuple[int, int], Float64[Array, "m1 m2"]] = {}
        orbital_i: int
        orbital_j: int
        for orbital_i in orbitals_by_atom[atom_pair[0]]:
            for orbital_j in orbitals_by_atom[atom_pair[1]]:
                if spinor and basis.spin[orbital_i] != basis.spin[orbital_j]:
                    continue
                l1: int = basis.l[orbital_i]
                l2: int = basis.l[orbital_j]
                angular_pair: Tuple[int, int] = tuple(sorted((l1, l2)))
                channels: Tuple[str, ...] = CHANNELS_BY_PAIR[angular_pair]
                integral_vector: Float64[Array, " n_m"]
                found_any: bool
                integral_vector, found_any = _integral_vector(
                    sk_params,
                    lookup,
                    forward_pair,
                    reverse_pair,
                    shell_numbers[bond_index],
                    channels,
                )
                if not found_any:
                    continue
                cache_key: Tuple[int, int] = (l1, l2)
                if cache_key not in block_cache:
                    block_cache[cache_key] = sk_block(
                        l1,
                        l2,
                        integral_vector,
                        cartesian_bond,
                    )
                block: Float64[Array, "m1 m2"] = block_cache[cache_key]
                amplitude: Float64[Array, ""] = block[
                    basis.m[orbital_i] + l1,
                    basis.m[orbital_j] + l2,
                ]
                complex_amplitude: Complex128[Array, ""] = jnp.asarray(
                    amplitude,
                    dtype=jnp.complex128,
                )
                hopping_pairs.extend(
                    ((orbital_i, orbital_j), (orbital_j, orbital_i))
                )
                hopping_cells.extend(
                    (
                        cell,
                        (-cell[0], -cell[1], -cell[2]),
                    )
                )
                amplitudes.extend(
                    (complex_amplitude, jnp.conj(complex_amplitude))
                )

    if amplitudes:
        hopping_array: Complex128[Array, " n_hop"] = jnp.stack(amplitudes)
    else:
        hopping_array = jnp.zeros((0,), dtype=jnp.complex128)
    model: TBModel = make_tb_model(
        hopping_amplitudes=hopping_array,
        onsite_energies=onsite_energies,
        soc_lambdas=soc_lambdas,
        geometry=geometry,
        basis=basis,
        hopping_pairs=tuple(hopping_pairs),
        hopping_cells=tuple(hopping_cells),
        shell_index=shell_index,
        spinor=spinor,
    )
    return model


@jaxtyped(typechecker=beartype)
def build_sk_model(  # noqa: DOC502, DOC503, PLR0912, PLR0915
    geometry: CrystalGeometry,
    basis: OrbitalBasis,
    sk_params: SlaterKosterParams,
    onsite_energies: Float64[Array, " n_orb"],
    soc_lambdas: Float64[Array, " n_shells"],
    shell_index: Tuple[int, ...],
    cutoff: float,
    spinor: bool = False,
) -> TBModel:
    """Build a validated tight-binding model from two-center integrals.

    Select static neighbor topology, evaluate every requested two-center
    block, and emit explicit reverse records for exact Hermitian closure.

    :see: :class:`~.test_slaterkoster.TestBuildSkModel`

    Parameters
    ----------
    geometry : CrystalGeometry
        Crystal lattice, fractional atom positions, and species.
    basis : OrbitalBasis
        Orbital metadata in package real-harmonic order.
    sk_params : SlaterKosterParams
        Fundamental two-center values and their static keys.
    onsite_energies : Float64[Array, " n_orb"]
        Onsite orbital energies in eV.
    soc_lambdas : Float64[Array, " n_shells"]
        Atomic spin--orbit strengths in eV.
    shell_index : Tuple[int, ...]
        Orbital-to-SOC-shell map passed to the tight-binding carrier.
    cutoff : float
        Positive inclusive neighbor cutoff in angstroms.
    spinor : bool, optional
        Whether ``basis`` already contains explicit spin channels. Hoppings
        preserve spin; this flag never doubles the basis a second time.

    Returns
    -------
    model : TBModel
        Validated model with exact integer cells and explicit reverse hopping
        records.

    Raises
    ------
    ValueError
        If keys fail their grammar, the basis contains an orbital beyond d,
        fully traced geometry cannot certify topology, or topology setup fails.
    EquinoxRuntimeError
        If a traced numerical invariant of the resulting model fails.

    Notes
    -----
    Supported keys are ``"<A>-<B>:<channel>"`` and the optional one-based
    distance-shell form ``"<A>-<B>@<shell>:<channel>"``. Generic channel-only
    keys such as ``"pp_pi"`` apply to every species pair. Omitted companion
    channels are exactly zero. For example, a graphene model may provide only
    ``"C-C:pp_pi"``.

    Concrete setup prunes the hopping metadata to the cutoff using a complete
    singular-value search certificate. Eager automatic differentiation can
    recover the concrete primal geometry and retains local derivatives on the
    selected topology. The builder rejects fully traced geometry without a
    concrete primal. Use :func:`~diffpes.tightb.sk_model_parameter_view` to
    capture certified topology before compiling geometry optimization.
    """
    if any(
        angular < 0 or angular > MAX_SK_ANGULAR_MOMENTUM for angular in basis.l
    ):
        message: str = "build_sk_model supports only s, p, and d orbitals"
        raise ValueError(message)
    if not np.isfinite(cutoff) or cutoff <= 0.0:
        message = "cutoff must be a positive finite float"
        raise ValueError(message)
    traced_geometry: bool = _geometry_is_traced(geometry)
    topology_geometry: CrystalGeometry | None = (
        _primal_geometry(geometry) if traced_geometry else geometry
    )
    if topology_geometry is None:
        message = (
            "build_sk_model cannot certify neighbor topology from fully "
            "traced geometry; freeze topology before compilation"
        )
        raise ValueError(message)
    atom_pairs: Tuple[Tuple[int, int], ...]
    cells: Tuple[Tuple[int, int, int], ...]
    shell_numbers: Tuple[int, ...]
    (
        atom_pairs,
        cells,
        shell_numbers,
    ) = _freeze_neighbor_topology(
        topology_geometry,
        cutoff,
    )
    model: TBModel = _build_sk_model_from_topology(
        geometry,
        basis=basis,
        sk_params=sk_params,
        onsite_energies=onsite_energies,
        soc_lambdas=soc_lambdas,
        shell_index=shell_index,
        atom_pairs=atom_pairs,
        cells=cells,
        shell_numbers=shell_numbers,
        spinor=spinor,
    )
    return model


__all__: list[str] = [
    "build_sk_model",
    "neighbor_shells",
    "sk_block",
]
