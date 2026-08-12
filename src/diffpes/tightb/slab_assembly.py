"""Rebuild slabs from frozen topology.

Extended Summary
----------------
This module assembles slab bases, hoppings, geometry, and adjacency validation.

Routine Listings
----------------
:func:`rebuild_slab`
    Construct a slab from frozen topology using only JAX geometry.
:func:`validate_open_surface_adjacency`
    Reject direct or component-propagated periodic normal images.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
from beartype import beartype
from beartype.typing import Dict, List, Tuple
from jaxtyping import Array, Complex128, Float64, Int32, Int64, jaxtyped
from numpy.typing import NDArray

from diffpes.types import (
    CrystalGeometry,
    OrbitalBasis,
    SlabSpec,
    SlabTopology,
    SurfaceCell,
    TBModel,
    make_crystal_geometry,
    make_orbital_basis,
    make_slab_spec,
    make_tb_model,
)

from .slab_rotation import rotate_tb_model
from .slab_surface_cell import _assemble_surface_cell
from .slab_topology import (
    _base_surface_coordinates,
    _inverse_integer_matrix,
    _surface_integer_matrix,
)


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
    orbitals_by_atom: Dict[int, List[int]] = {}
    bulk_orbital: int
    for bulk_orbital, atom in enumerate(basis.atom_indices):
        orbitals_by_atom.setdefault(atom, []).append(bulk_orbital)
    slab_to_bulk: List[int] = []
    slab_atom_indices: List[int] = []
    slab_layers: List[int] = []
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
    slab_shells: List[int] = []
    bulk_shell_gather: List[int] = []
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
    pairs: List[Tuple[int, int]] = []
    cells: List[Tuple[int, int, int]] = []
    gather: List[int] = []
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
    gather_array: Int32[Array, " n_hop_slab"] = jnp.asarray(
        gather_tuple, dtype=jnp.int32
    )
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

    :see: :class:`~.test_slab_assembly.TestValidateOpenSurfaceAdjacency`

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
        List[Tuple[int, int, int]],
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
            orbital_fractional: Float64[Array, "n_orb 3"] = (
                model.geometry.positions[
                    jnp.asarray(model.basis.atom_indices, dtype=jnp.int32)
                ]
            )
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
            queue: List[Tuple[int, int, Tuple[int, ...]]] = [(root, 0, ())]
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
    bulk_atom_array: Int32[Array, " n_slab_atoms"] = jnp.asarray(
        bulk_atoms, dtype=jnp.int32
    )
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
        bulk_orbital_array: Int32[Array, " n_orb"] = jnp.asarray(
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
        slab_atom_index_array: Int32[Array, " n_orb"] = jnp.asarray(
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
def rebuild_slab(
    bulk_model: TBModel,
    topology: SlabTopology,
) -> Tuple[TBModel, SlabSpec]:
    """Construct a slab from frozen topology using only JAX geometry.

    The function contains no host conversion of a traced model leaf. It is
    therefore the differentiable slab-rebuild seam. Call
    :func:`freeze_slab_topology` once at a representative geometry, then pass
    compatible continuously perturbed models here.

    :see: :class:`~.test_slab_assembly.TestRebuildSlab`

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
    slab_to_bulk_array: Int32[Array, " n_orb"] = jnp.asarray(
        slab_to_bulk,
        dtype=jnp.int32,
    )
    shell_gather_array: Int32[Array, " n_shell"] = jnp.asarray(
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


__all__: list[str] = [
    "rebuild_slab",
    "validate_open_surface_adjacency",
]
