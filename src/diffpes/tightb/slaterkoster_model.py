"""Build tight-binding models from two-center integrals.

Extended Summary
----------------
This module maps registered integrals onto frozen bonds.
It assembles one model from those values.

Routine Listings
----------------
:func:`build_sk_model`
    Build a validated tight-binding model from two-center integrals.
"""

from collections.abc import Sequence

import jax
import jax.numpy as jnp
import numpy as np
from beartype import beartype
from beartype.typing import Dict, List, Tuple
from jaxtyping import Array, Complex128, Float64, jaxtyped
from numpy.typing import NDArray

from diffpes.constants import (
    CHANNELS_BY_PAIR,
    KNOWN_CHANNELS,
    MAX_SK_ANGULAR_MOMENTUM,
    PARAMETER_KEY_PARTS,
    SHELL_ATOLERANCE,
    SHELL_RTOLERANCE,
    SPECIES_PAIR_PARTS,
)
from diffpes.types import (
    CrystalGeometry,
    OrbitalBasis,
    SlaterKosterParams,
    TBModel,
    make_tb_model,
)

from .neighbor_shells import (
    _displacements_and_distances,
    _geometry_is_traced,
    _primal_geometry,
    neighbor_shells,
)
from .slaterkoster import sk_block


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
        pieces: List[str] = key.split(":")
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
            species: List[str] = pair.split("-")
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
    grouped: Dict[Tuple[str, str], List[float]] = {}
    pair_groups: List[Tuple[str, str]] = []
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
    values: List[float]
    for group in grouped:
        group_distances: List[float] = sorted(
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

    shell_numbers: List[int] = []
    for distance, group in zip(
        host_distances,
        pair_groups,
        strict=True,
    ):
        values = grouped[group]
        matching: List[int] = [
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
    values: List[Float64[Array, ""]] = []
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
    amplitudes: List[Complex128[Array, ""]] = []
    hopping_pairs: List[Tuple[int, int]] = []
    hopping_cells: List[Tuple[int, int, int]] = []
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

    :see: :class:`~.test_slaterkoster_model.TestBuildSkModel`

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
]
