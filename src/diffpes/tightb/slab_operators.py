"""Propagate Wannier operators into finite slabs.

Extended Summary
----------------
This module preserves position and Wannier operator sidecars.
It propagates them during slab construction.

Routine Listings
----------------
:func:`gen_slab_with_operators`
    Construct a slab while preserving its Wannier operator sidecar.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
from beartype import beartype
from beartype.typing import Dict, List, Tuple
from jaxtyping import Array, Complex128, Float64, Int32, Int64, jaxtyped
from numpy.typing import NDArray

from diffpes.types import (
    SlabSpec,
    TBModel,
    WannierOperatorData,
    make_tb_model,
    make_wannier_operator_data,
)

from .slab import gen_slab
from .slab_assembly import _orbital_lookup
from .slab_rotation import (
    _missing_magnetic_numbers,
    _orbital_rotation,
    _shell_groups,
)
from .slab_topology import (
    _base_surface_coordinates,
    _inverse_integer_matrix,
    _surface_integer_matrix,
)


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
            indices: List[int]
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
        diagonal: Int32[Array, " n_bulk_orb"] = jnp.arange(
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
    bulk_indices: Int32[Array, " n_orb_slab"] = jnp.asarray(
        slab_to_bulk, dtype=jnp.int32
    )
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
    emitted: List[
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
        diagonal: Int32[Array, " n_orb"] = jnp.arange(
            n_slab_orbitals, dtype=jnp.int32
        )
        current_diagonal: Complex128[Array, "n_orb 3"] = matrices[
            zero_cell_index,
            diagonal,
            diagonal,
        ]
        bulk_indices: Int32[Array, " n_orb"] = jnp.asarray(
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

    :see: :class:`~.test_slab_operators.TestGenSlabWithOperators`
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
    "gen_slab_with_operators",
]
