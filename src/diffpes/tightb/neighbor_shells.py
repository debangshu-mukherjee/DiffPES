"""Find finite neighbor-shell topology.

Extended Summary
----------------
This module freezes undirected neighbor bonds from concrete crystal geometry.

Routine Listings
----------------
:func:`neighbor_shells`
    Find unique undirected neighbor bonds at host setup time.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from beartype import beartype
from beartype.typing import List, Tuple
from jax import core
from jaxtyping import Array, Bool, Float64, Int32, jaxtyped
from numpy.typing import NDArray

from diffpes.constants import CARTESIAN_COMPONENTS, MIN_BOND_DISTANCE
from diffpes.types import CrystalGeometry


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
    atom_pairs: List[Tuple[int, int]] = []
    cells: List[Tuple[int, int, int]] = []
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
    atom_i: Int32[Array, " n_bond"] = jnp.asarray(
        tuple(pair[0] for pair in atom_pairs),
        dtype=jnp.int32,
    )
    atom_j: Int32[Array, " n_bond"] = jnp.asarray(
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


def _concrete_primal(
    value: Float64[Array, "..."],
) -> Float64[Array, "..."] | None:
    """PRIVATE: Recover the concrete primal carried by an eager AD tracer.

    Parameters
    ----------
    value : Float64[Array, "..."]
        Possibly traced array leaf.

    Returns
    -------
    primal : Float64[Array, "..."] | None
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
            missing: Float64[Array, "..."] | None = None
            return missing
        candidate = candidate.primal
    if isinstance(candidate, jax.Array):
        primal: Float64[Array, "..."] | None = candidate
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
    primal_lattice: Float64[Array, "3 3"] | None = _concrete_primal(
        geometry.lattice
    )
    primal_positions: Float64[Array, "n_atoms 3"] | None = _concrete_primal(
        geometry.positions
    )
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

    :see: :class:`~.test_neighbor_shells.TestNeighborShells`

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


__all__: list[str] = [
    "neighbor_shells",
]
