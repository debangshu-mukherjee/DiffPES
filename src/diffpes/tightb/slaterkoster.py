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
cutoff nor merge a nonzero bond into zero length. A transformed builder uses
the fixed candidate topology of the default ``5 x 5 x 5`` supercell. It masks
bonds outside the cutoff. This keeps the same Hamiltonian away from the
boundary and retains the local position derivative.
"""

from collections.abc import Sequence

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from beartype import beartype
from jax import core
from jaxtyping import Array, Complex, Float, jaxtyped

from diffpes.types import (
    CrystalGeometry,
    OrbitalBasis,
    SlaterKosterParams,
    TBModel,
    make_tb_model,
)

_CARTESIAN_COMPONENTS: int = 3
_DEFAULT_SUPERCELL_RADIUS: int = 2
_MAX_SK_ANGULAR_MOMENTUM: int = 2
_MIN_BOND_DISTANCE: float = 1e-12
_PARAMETER_KEY_PARTS: int = 2
_SHELL_ATOLERANCE: float = 1e-8
_SHELL_RTOLERANCE: float = 1e-8
_SPECIES_PAIR_PARTS: int = 2

_CHANNELS_BY_PAIR: dict[tuple[int, int], tuple[str, ...]] = {
    (0, 0): ("ss_sigma",),
    (0, 1): ("sp_sigma",),
    (0, 2): ("sd_sigma",),
    (1, 1): ("pp_sigma", "pp_pi"),
    (1, 2): ("pd_sigma", "pd_pi"),
    (2, 2): ("dd_sigma", "dd_pi", "dd_delta"),
}
_KNOWN_CHANNELS: frozenset[str] = frozenset(
    channel for channels in _CHANNELS_BY_PAIR.values() for channel in channels
)


def _validate_angular_momentum(l: int, name: str) -> None:  # noqa: E741
    """Validate one static Slater--Koster shell angular momentum."""
    if type(l) is not int or not 0 <= l <= _MAX_SK_ANGULAR_MOMENTUM:
        message: str = (
            f"{name} must be an integer from 0 through "
            f"{_MAX_SK_ANGULAR_MOMENTUM}"
        )
        raise ValueError(message)


def _rotation_z_to_direction(
    direction: Float[Array, " 3"],
) -> Float[Array, "3 3"]:
    """Construct a proper rotation taking positive z onto ``direction``."""
    dtype: jnp.dtype = direction.dtype
    identity: Float[Array, "3 3"] = jnp.eye(
        _CARTESIAN_COMPONENTS,
        dtype=dtype,
    )
    direction_x: Float[Array, ""] = direction[0]
    direction_y: Float[Array, ""] = direction[1]
    direction_z: Float[Array, ""] = direction[2]
    use_north_chart: Array = direction_z >= 0.0

    north_denominator: Float[Array, ""] = jnp.where(
        use_north_chart,
        1.0 + direction_z,
        1.0,
    )
    north_cross: Float[Array, " 3"] = jnp.stack(
        (-direction_y, direction_x, jnp.zeros_like(direction_z))
    )
    north_skew: Float[Array, "3 3"] = jnp.asarray(
        (
            (0.0, -north_cross[2], north_cross[1]),
            (north_cross[2], 0.0, -north_cross[0]),
            (-north_cross[1], north_cross[0], 0.0),
        ),
        dtype=dtype,
    )
    north_rotation: Float[Array, "3 3"] = (
        identity + north_skew + north_skew @ north_skew / north_denominator
    )

    south_denominator: Float[Array, ""] = jnp.where(
        use_north_chart,
        1.0,
        1.0 - direction_z,
    )
    south_cross: Float[Array, " 3"] = jnp.stack(
        (direction_y, -direction_x, jnp.zeros_like(direction_z))
    )
    south_skew: Float[Array, "3 3"] = jnp.asarray(
        (
            (0.0, -south_cross[2], south_cross[1]),
            (south_cross[2], 0.0, -south_cross[0]),
            (-south_cross[1], south_cross[0], 0.0),
        ),
        dtype=dtype,
    )
    half_turn_x: Float[Array, "3 3"] = jnp.diag(
        jnp.asarray((1.0, -1.0, -1.0), dtype=dtype)
    )
    from_negative_z: Float[Array, "3 3"] = (
        identity + south_skew + south_skew @ south_skew / south_denominator
    )
    south_rotation: Float[Array, "3 3"] = from_negative_z @ half_turn_x
    rotation: Float[Array, "3 3"] = jnp.where(
        use_north_chart,
        north_rotation,
        south_rotation,
    )
    return rotation


def _p_axes(dtype: jnp.dtype) -> Float[Array, "3 3"]:
    """Return Cartesian unit vectors in real-harmonic p-shell order."""
    axes: Float[Array, "3 3"] = jnp.asarray(
        (
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
            (1.0, 0.0, 0.0),
        ),
        dtype=dtype,
    )
    return axes


def _d_tensors(dtype: jnp.dtype) -> Float[Array, "5 3 3"]:
    """Return orthonormal traceless tensors in real d-shell order."""
    inverse_sqrt_two: float = 1.0 / np.sqrt(2.0)
    inverse_sqrt_six: float = 1.0 / np.sqrt(6.0)
    tensors: Float[Array, "5 3 3"] = jnp.asarray(
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
    rotation: Float[Array, "3 3"],
) -> Float[Array, "m1 m2"]:
    """Represent a Cartesian rotation in a real s, p, or d shell."""
    if l == 0:
        scalar: Float[Array, "1 1"] = jnp.ones(
            (1, 1),
            dtype=rotation.dtype,
        )
        return scalar
    if l == 1:
        axes: Float[Array, "3 3"] = _p_axes(rotation.dtype)
        vector_representation: Float[Array, "3 3"] = axes @ rotation @ axes.T
        return vector_representation

    tensors: Float[Array, "5 3 3"] = _d_tensors(rotation.dtype)
    rotated_tensors: Float[Array, "5 3 3"] = jnp.einsum(
        "ik,akl,jl->aij",
        rotation,
        tensors,
        rotation,
    )
    tensor_representation: Float[Array, "5 5"] = jnp.einsum(
        "bij,aij->ba",
        tensors,
        rotated_tensors,
    )
    return tensor_representation


@jaxtyped(typechecker=beartype)
def sk_block(  # noqa: DOC502, DOC503
    l1: int,
    l2: int,
    v_llm: Float[Array, " n_m"],
    bond_cart: Float[Array, " 3"],
) -> Float[Array, "m1 m2"]:
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
    v_llm : Float[Array, " n_m"]
        Fundamental integrals ordered by ``abs(m)``: sigma, pi, then delta.
        Its length must be ``min(l1, l2) + 1``.
    bond_cart : Float[Array, " 3"]
        Nonzero Cartesian displacement from the row-shell atom to the
        column-shell atom.

    Returns
    -------
    block : Float[Array, "m1 m2"]
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

    values: Float[Array, " n_m"] = eqx.error_if(
        v_llm,
        ~jnp.all(jnp.isfinite(v_llm)),
        "sk_block: integrals finite",
    )
    bond: Float[Array, " 3"] = eqx.error_if(
        bond_cart,
        ~jnp.all(jnp.isfinite(bond_cart)),
        "sk_block: bond finite",
    )
    bond_norm: Float[Array, ""] = jnp.linalg.norm(bond)
    bond = eqx.error_if(
        bond,
        ~(bond_norm > 0.0),
        "sk_block: bond nonzero",
    )
    direction: Float[Array, " 3"] = bond / bond_norm
    rotation: Float[Array, "3 3"] = _rotation_z_to_direction(direction)
    left_rotation: Float[Array, "m1 m1"] = _orbital_rotation(l1, rotation)
    right_rotation: Float[Array, "m2 m2"] = _orbital_rotation(l2, rotation)

    bond_axis: Float[Array, "m1 m2"] = jnp.zeros(
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
    block: Float[Array, "m1 m2"] = (
        radial_parity * left_rotation @ bond_axis @ right_rotation.T
    )
    return block


def _candidate_topology(
    n_atoms: int,
    supercell_radius: int,
) -> tuple[
    tuple[tuple[int, int], ...],
    tuple[tuple[int, int, int], ...],
]:
    """Build one canonical representative of every undirected bond."""
    atom_pairs: list[tuple[int, int]] = []
    cells: list[tuple[int, int, int]] = []
    cell_x: int
    cell_y: int
    cell_z: int
    atom_i: int
    atom_j: int
    for cell_x in range(-supercell_radius, supercell_radius + 1):
        for cell_y in range(-supercell_radius, supercell_radius + 1):
            for cell_z in range(-supercell_radius, supercell_radius + 1):
                cell: tuple[int, int, int] = (cell_x, cell_y, cell_z)
                for atom_i in range(n_atoms):
                    for atom_j in range(n_atoms):
                        if atom_i == atom_j and cell == (0, 0, 0):
                            continue
                        record: tuple[int, int, int, int, int] = (
                            atom_i,
                            atom_j,
                            cell_x,
                            cell_y,
                            cell_z,
                        )
                        reverse: tuple[int, int, int, int, int] = (
                            atom_j,
                            atom_i,
                            -cell_x,
                            -cell_y,
                            -cell_z,
                        )
                        if record < reverse:
                            atom_pairs.append((atom_i, atom_j))
                            cells.append(cell)
    return tuple(atom_pairs), tuple(cells)


def _displacements_and_distances(
    geometry: CrystalGeometry,
    atom_pairs: tuple[tuple[int, int], ...],
    cells: tuple[tuple[int, int, int], ...],
) -> tuple[Float[Array, "n_bond 3"], Float[Array, " n_bond"]]:
    """Derive differentiable fractional bonds and Cartesian distances."""
    if not atom_pairs:
        empty_displacements: Float[Array, "0 3"] = jnp.zeros(
            (0, _CARTESIAN_COMPONENTS),
            dtype=geometry.positions.dtype,
        )
        empty_distances: Float[Array, " 0"] = jnp.zeros(
            (0,),
            dtype=geometry.positions.dtype,
        )
        return empty_displacements, empty_distances
    atom_i: Array = jnp.asarray(
        tuple(pair[0] for pair in atom_pairs),
        dtype=jnp.int32,
    )
    atom_j: Array = jnp.asarray(
        tuple(pair[1] for pair in atom_pairs),
        dtype=jnp.int32,
    )
    cell_array: Float[Array, "n_bond 3"] = jnp.asarray(
        cells,
        dtype=geometry.positions.dtype,
    )
    displacements: Float[Array, "n_bond 3"] = (
        cell_array + geometry.positions[atom_j] - geometry.positions[atom_i]
    )
    cartesian: Float[Array, "n_bond 3"] = displacements @ geometry.lattice
    distances: Float[Array, " n_bond"] = jnp.linalg.norm(
        cartesian,
        axis=1,
    )
    return displacements, distances


def _geometry_is_traced(geometry: CrystalGeometry) -> bool:
    """Return whether topology-defining geometry values are JAX tracers."""
    traced: bool = isinstance(geometry.positions, core.Tracer) or isinstance(
        geometry.lattice, core.Tracer
    )
    return traced


def _concrete_primal(value: Array) -> Array | None:
    """Recover the concrete primal carried by an eager AD tracer."""
    candidate: object = value
    while isinstance(candidate, core.Tracer):
        if not hasattr(candidate, "primal"):
            return None
        candidate = candidate.primal
    if isinstance(candidate, jax.Array):
        return candidate
    return None


def _primal_geometry(
    geometry: CrystalGeometry,
) -> CrystalGeometry | None:
    """Recover topology values from eager forward/reverse AD tracers."""
    primal_lattice: Array | None = _concrete_primal(geometry.lattice)
    primal_positions: Array | None = _concrete_primal(geometry.positions)
    if primal_lattice is None or primal_positions is None:
        return None
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
    supercell_radius: int = _DEFAULT_SUPERCELL_RADIUS,
) -> tuple[
    tuple[tuple[int, int], ...],
    tuple[tuple[int, int, int], ...],
    Float[Array, "n_bond 3"],
    Float[Array, " n_bond"],
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
    supercell_radius : int, optional
        Number of translated cells to enumerate in each positive and negative
        lattice direction. The default ``2`` scans a ``5 x 5 x 5``
        supercell.

    Returns
    -------
    atom_pairs : tuple[tuple[int, int], ...]
        Canonical undirected atom pairs. Every physical bond appears once.
    cells : tuple[tuple[int, int, int], ...]
        Exact integer cell translations from the first atom to the second.
    displacements : Float[Array, "n_bond 3"]
        Fractional displacements ``R + tau_j - tau_i``.
    distances : Float[Array, " n_bond"]
        Cartesian bond lengths in angstroms.

    Raises
    ------
    ValueError
        If the cutoff or radius fails validation, tracing reaches this host
        setup routine, or distinct atom records produce a zero-length bond.

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
    if type(supercell_radius) is not int or supercell_radius < 0:
        message = "supercell_radius must be a non-negative integer"
        raise ValueError(message)
    if _geometry_is_traced(geometry):
        message = (
            "neighbor_shells is a host setup operation; freeze topology "
            "before tracing geometry"
        )
        raise ValueError(message)

    candidates: tuple[tuple[int, int], ...]
    candidate_cells: tuple[tuple[int, int, int], ...]
    candidates, candidate_cells = _candidate_topology(
        geometry.positions.shape[0],
        supercell_radius,
    )
    with jax.ensure_compile_time_eval():
        candidate_displacements: Float[Array, "n_candidate 3"]
        candidate_distances: Float[Array, " n_candidate"]
        candidate_displacements, candidate_distances = (
            _displacements_and_distances(
                geometry,
                candidates,
                candidate_cells,
            )
        )
        host_distances: np.ndarray = np.asarray(candidate_distances)
    if np.any(host_distances <= _MIN_BOND_DISTANCE):
        message = "neighbor_shells encountered a zero-length atom pair"
        raise ValueError(message)
    keep: np.ndarray = host_distances <= cutoff
    kept_indices: tuple[int, ...] = tuple(
        int(index) for index in np.flatnonzero(keep)
    )
    atom_pairs: tuple[tuple[int, int], ...] = tuple(
        candidates[index] for index in kept_indices
    )
    cells: tuple[tuple[int, int, int], ...] = tuple(
        candidate_cells[index] for index in kept_indices
    )
    displacements: Float[Array, "n_bond 3"]
    distances: Float[Array, " n_bond"]
    displacements, distances = _displacements_and_distances(
        geometry,
        atom_pairs,
        cells,
    )
    return atom_pairs, cells, displacements, distances


def _parse_parameter_keys(
    keys: tuple[str, ...],
) -> dict[tuple[str | None, int | None, str], int]:
    """Parse and validate material, optional shell, and channel identifiers."""
    parsed: dict[tuple[str | None, int | None, str], int] = {}
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
        elif len(pieces) == _PARAMETER_KEY_PARTS:
            pair, channel = pieces
            if "@" in pair:
                pair, shell_text = pair.rsplit("@", maxsplit=1)
                if not shell_text.isdecimal() or int(shell_text) < 1:
                    message: str = (
                        f"invalid neighbor-shell selector in SK key {key!r}"
                    )
                    raise ValueError(message)
                shell = int(shell_text)
            species: list[str] = pair.split("-")
            if len(species) != _SPECIES_PAIR_PARTS or not all(species):
                message = f"invalid species pair in SK key {key!r}"
                raise ValueError(message)
        else:
            message = f"invalid SK key grammar {key!r}"
            raise ValueError(message)
        if channel not in _KNOWN_CHANNELS:
            message = f"unknown Slater--Koster channel {channel!r}"
            raise ValueError(message)
        parsed[(pair, shell, channel)] = index
    return parsed


def _species_pair(
    geometry: CrystalGeometry,
    atom_pair: tuple[int, int],
) -> tuple[str | None, str | None]:
    """Return forward and reversed material-pair identifiers."""
    if not geometry.species:
        return None, None
    species_i: str = geometry.species[atom_pair[0]]
    species_j: str = geometry.species[atom_pair[1]]
    return f"{species_i}-{species_j}", f"{species_j}-{species_i}"


def _shell_numbers(
    geometry: CrystalGeometry,
    atom_pairs: tuple[tuple[int, int], ...],
    distances: Float[Array, " n_bond"],
) -> tuple[int, ...]:
    """Create one-based distance-shell numbers within each species pair."""
    host_distances: np.ndarray = np.asarray(distances)
    grouped: dict[tuple[str, str], list[float]] = {}
    pair_groups: list[tuple[str, str]] = []
    atom_pair: tuple[int, int]
    for atom_pair in atom_pairs:
        if geometry.species:
            group: tuple[str, str] = tuple(
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

    group: tuple[str, str]
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
                rtol=_SHELL_RTOLERANCE,
                atol=_SHELL_ATOLERANCE,
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
                rtol=_SHELL_RTOLERANCE,
                atol=_SHELL_ATOLERANCE,
            )
        ]
        shell_numbers.append(matching[0] + 1)
    return tuple(shell_numbers)


def _parameter_index(
    lookup: dict[tuple[str | None, int | None, str], int],
    forward_pair: str | None,
    reverse_pair: str | None,
    shell: int,
    channel: str,
) -> int | None:
    """Resolve one integral with shell-specific-to-generic precedence."""
    candidates: Sequence[tuple[str | None, int | None, str]] = (
        (forward_pair, shell, channel),
        (reverse_pair, shell, channel),
        (forward_pair, None, channel),
        (reverse_pair, None, channel),
        (None, None, channel),
    )
    candidate: tuple[str | None, int | None, str]
    for candidate in candidates:
        if candidate in lookup:
            return lookup[candidate]
    return None


def _integral_vector(
    sk_params: SlaterKosterParams,
    lookup: dict[tuple[str | None, int | None, str], int],
    forward_pair: str | None,
    reverse_pair: str | None,
    shell: int,
    channels: tuple[str, ...],
) -> tuple[Float[Array, " n_m"], bool]:
    """Collect sigma/pi/delta values, treating omitted channels as zero."""
    values: list[Float[Array, ""]] = []
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
    vector: Float[Array, " n_m"] = jnp.stack(values)
    return vector, found_any


@jaxtyped(typechecker=beartype)
def build_sk_model(  # noqa: DOC502, DOC503, PLR0912, PLR0915
    geometry: CrystalGeometry,
    basis: OrbitalBasis,
    sk_params: SlaterKosterParams,
    onsite_energies: Float[Array, " n_orb"],
    soc_lambdas: Float[Array, " n_shells"],
    shell_index: tuple[int, ...],
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
    onsite_energies : Float[Array, " n_orb"]
        Onsite orbital energies in eV.
    soc_lambdas : Float[Array, " n_shells"]
        Atomic spin--orbit strengths in eV.
    shell_index : tuple[int, ...]
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
        traced geometry requests a later distance shell, or topology setup
        fails.
    EquinoxRuntimeError
        If a traced numerical invariant of the resulting model fails.

    Notes
    -----
    Supported keys are ``"<A>-<B>:<channel>"`` and the optional one-based
    distance-shell form ``"<A>-<B>@<shell>:<channel>"``. Generic channel-only
    keys such as ``"pp_pi"`` apply to every species pair. Omitted companion
    channels are exactly zero. For example, a graphene model may provide only
    ``"C-C:pp_pi"``.

    Concrete setup prunes the hopping metadata to the cutoff. If positions or
    lattice are tracers, the builder keeps the static radius-two candidate
    topology and multiplies outside-cutoff amplitudes by zero. This is the
    topology-freeze contract: gradients are local and make no claim at a
    cutoff crossing.
    """
    if any(
        angular < 0 or angular > _MAX_SK_ANGULAR_MOMENTUM
        for angular in basis.l
    ):
        message: str = "build_sk_model supports only s, p, and d orbitals"
        raise ValueError(message)
    if not np.isfinite(cutoff) or cutoff <= 0.0:
        message = "cutoff must be a positive finite float"
        raise ValueError(message)
    lookup: dict[tuple[str | None, int | None, str], int] = (
        _parse_parameter_keys(sk_params.keys)
    )

    traced_geometry: bool = _geometry_is_traced(geometry)
    topology_geometry: CrystalGeometry | None = (
        _primal_geometry(geometry) if traced_geometry else geometry
    )
    atom_pairs: tuple[tuple[int, int], ...]
    cells: tuple[tuple[int, int, int], ...]
    displacements: Float[Array, "n_bond 3"]
    distances: Float[Array, " n_bond"]
    if topology_geometry is None:
        if any(shell is not None and shell != 1 for _, shell, _ in lookup):
            message = (
                "position/lattice differentiation supports base or @1 SK "
                "keys; freeze higher distance-shell topology outside tracing"
            )
            raise ValueError(message)
        atom_pairs, cells = _candidate_topology(
            geometry.positions.shape[0],
            _DEFAULT_SUPERCELL_RADIUS,
        )
        displacements, distances = _displacements_and_distances(
            geometry,
            atom_pairs,
            cells,
        )
        shell_numbers: tuple[int, ...] = (1,) * len(atom_pairs)
        active: Array = (distances > _MIN_BOND_DISTANCE) & (
            distances <= cutoff
        )
    else:
        topology_pairs: tuple[tuple[int, int], ...]
        topology_cells: tuple[tuple[int, int, int], ...]
        topology_displacements: Float[Array, "n_bond 3"]
        topology_distances: Float[Array, " n_bond"]
        (
            topology_pairs,
            topology_cells,
            topology_displacements,
            topology_distances,
        ) = neighbor_shells(
            topology_geometry,
            cutoff,
            _DEFAULT_SUPERCELL_RADIUS,
        )
        del topology_displacements, topology_distances
        atom_pairs = topology_pairs
        cells = topology_cells
        displacements, distances = _displacements_and_distances(
            geometry,
            atom_pairs,
            cells,
        )
        with jax.ensure_compile_time_eval():
            _, shell_distances = _displacements_and_distances(
                topology_geometry,
                atom_pairs,
                cells,
            )
            shell_numbers = _shell_numbers(
                topology_geometry,
                atom_pairs,
                shell_distances,
            )
        active = jnp.ones((len(atom_pairs),), dtype=jnp.bool_)

    orbitals_by_atom: tuple[tuple[int, ...], ...] = tuple(
        tuple(
            orbital
            for orbital, atom in enumerate(basis.atom_indices)
            if atom == atom_index
        )
        for atom_index in range(geometry.positions.shape[0])
    )
    amplitudes: list[Complex[Array, ""]] = []
    hopping_pairs: list[tuple[int, int]] = []
    hopping_cells: list[tuple[int, int, int]] = []
    bond_index: int
    atom_pair: tuple[int, int]
    cell: tuple[int, int, int]
    for bond_index, (atom_pair, cell) in enumerate(
        zip(atom_pairs, cells, strict=True)
    ):
        forward_pair: str | None
        reverse_pair: str | None
        forward_pair, reverse_pair = _species_pair(geometry, atom_pair)
        cartesian_bond: Float[Array, " 3"] = (
            displacements[bond_index] @ geometry.lattice
        )
        safe_bond: Float[Array, " 3"] = jnp.where(
            active[bond_index],
            cartesian_bond,
            jnp.asarray((0.0, 0.0, 1.0), dtype=cartesian_bond.dtype),
        )
        block_cache: dict[tuple[int, int], Float[Array, "m1 m2"]] = {}
        orbital_i: int
        orbital_j: int
        for orbital_i in orbitals_by_atom[atom_pair[0]]:
            for orbital_j in orbitals_by_atom[atom_pair[1]]:
                if spinor and basis.spin[orbital_i] != basis.spin[orbital_j]:
                    continue
                l1: int = basis.l[orbital_i]
                l2: int = basis.l[orbital_j]
                angular_pair: tuple[int, int] = tuple(sorted((l1, l2)))
                channels: tuple[str, ...] = _CHANNELS_BY_PAIR[angular_pair]
                integral_vector: Float[Array, " n_m"]
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
                cache_key: tuple[int, int] = (l1, l2)
                if cache_key not in block_cache:
                    block_cache[cache_key] = sk_block(
                        l1,
                        l2,
                        integral_vector,
                        safe_bond,
                    )
                block: Float[Array, "m1 m2"] = block_cache[cache_key]
                amplitude: Float[Array, ""] = (
                    block[
                        basis.m[orbital_i] + l1,
                        basis.m[orbital_j] + l2,
                    ]
                    * active[bond_index]
                )
                complex_amplitude: Complex[Array, ""] = jnp.asarray(
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
        hopping_array: Complex[Array, " n_hop"] = jnp.stack(amplitudes)
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


__all__: list[str] = [
    "build_sk_model",
    "neighbor_shells",
    "sk_block",
]
