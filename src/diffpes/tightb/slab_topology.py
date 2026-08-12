"""Freeze discrete slab topology choices.

Extended Summary
----------------
This module records every discrete slab choice.
These choices support a differentiable slab rebuild.

Routine Listings
----------------
:func:`freeze_slab_topology`
    Freeze every discrete choice required to rebuild one slab.
"""

from __future__ import annotations

import math

import numpy as np
from beartype import beartype
from beartype.typing import Dict, List, Tuple
from jaxtyping import Float64, Int64, jaxtyped
from numpy.typing import NDArray

from diffpes.types import (
    CrystalGeometry,
    SlabTopology,
    SurfaceCell,
    TBModel,
    make_slab_topology,
)

from .slab_surface_cell import _determinant_3x3, find_surface_cell


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
    candidates: List[Tuple[float, int, int]] = []
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
    kept: List[Tuple[float, int, int, str]] = []
    expanded_layers: int
    for expanded_layers in range(
        n_layers,
        n_layers + maximum_extra_layers + 1,
    ):
        baseline_top: float = base_top + (expanded_layers - 1) * spacing
        bottom_cut: float = baseline_bottom + fine[1]
        top_cut: float = baseline_top - fine[0]
        candidates: List[Tuple[float, int, int, str]] = []
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

    :see: :class:`~.test_slab_topology.TestFreezeSlabTopology`

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
    topology: SlabTopology = make_slab_topology(
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


__all__: list[str] = [
    "freeze_slab_topology",
]
