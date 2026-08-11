"""Define tight-binding model and diagonalized-band data structures.

Extended Summary
----------------
This module defines the native tight-binding carrier and the diagonalized
electronic-structure interface consumed by later ARPES stages. Tight-binding
connectivity is exact static metadata, while energies, complex amplitudes,
geometry, and eigensystems remain differentiable JAX leaves.

Routine Listings
----------------
:class:`DiagonalizedBands`
    Store diagonalized electronic-structure data in a JAX PyTree.
:class:`SlabSpec`
    Store static slab construction choices and provenance.
:class:`SlabTopology`
    Store host-selected discrete slab topology for pure-JAX rebuilding.
:class:`SurfaceCell`
    Store a validated Cartesian surface-cell frame.
:class:`TBModel`
    Store tight-binding parameters in a JAX PyTree.
:func:`make_diagonalized_bands`
    Create a validated ``DiagonalizedBands`` instance.
:func:`make_slab_spec`
    Create a validated slab-construction sidecar.
:func:`make_slab_topology`
    Create validated host-selected slab topology metadata.
:func:`make_surface_cell`
    Create a validated Cartesian surface-cell carrier.
:func:`make_tb_model`
    Create a validated ``TBModel`` instance.

Notes
-----
The tight-binding phase convention is the basis-position gauge. Each physical
fractional bond displacement follows ``R + tau_j - tau_i`` from exact
integer-cell metadata and either explicit orbital centres or atomic positions.
"""

import math

import equinox as eqx
import jax.numpy as jnp
from beartype import beartype
from beartype.typing import Dict, List, Optional, Tuple
from jaxtyping import (
    Array,
    Complex128,
    Float64,
    Int32,
    jaxtyped,
)

from .aliases import ScalarNumeric
from .geometry import CrystalGeometry
from .radial_params import OrbitalBasis

_HERMITICITY_TOLERANCE: float = 1e-12
_PAIR_LENGTH: int = 2
_SURFACE_VECTOR_COUNT: int = 2
_CELL_COMPONENTS: int = 3
_EIGENVALUE_NDIM: int = 2
_EIGENVECTOR_NDIM: int = 3
_ORBITAL_POSITION_NDIM: int = 2
_DEPTH_TOLERANCE_ANG: float = 1e-12
_ROTATION_ORTHOGONALITY_TOLERANCE: float = 1e-10


def _validate_depths_shape(
    depths: Optional[Float64[Array, " n_depth"]],
    n_orbitals: int,
) -> None:
    """PRIVATE: Validate the optional orbital-depth axis.

    Parameters
    ----------
    depths : Optional[Float64[Array, " n_depth"]]
        Orbital depths in Angstrom below the top surface, or ``None``
        for a bulk model.
    n_orbitals : int
        Number of orbitals the depth axis must match.

    Raises
    ------
    ValueError
        If ``depths`` is present and is not one-dimensional with one
        entry per orbital. This is the static construction-time
        contract.

    Notes
    -----
    ``None`` passes untouched. Checks only static shape metadata here.
    The factory keeps value nonnegativity traced.
    """
    if depths is not None and (
        depths.ndim != 1 or depths.shape[0] != n_orbitals
    ):
        message: str = "depths must have shape (n_orbitals,)"
        raise ValueError(message)


def _validate_integer_triple(
    values: Tuple[int, int, int],
    name: str,
) -> None:
    """PRIVATE: Validate one exact integer coefficient triple.

    Parameters
    ----------
    values : Tuple[int, int, int]
        Candidate integer coefficient triple.
    name : str
        Field name used in the static error message.

    Raises
    ------
    ValueError
        If ``values`` is not a tuple of exactly three Python integers.
        This is the static construction-time contract.

    Notes
    -----
    Use exact ``type`` comparisons to reject bools and NumPy integers.
    The triple stays exact under integer arithmetic.
    """
    if (
        type(values) is not tuple
        or len(values) != _CELL_COMPONENTS
        or any(type(value) is not int for value in values)
    ):
        message: str = f"{name} must be a tuple of three integers"
        raise ValueError(message)


def _integer_dot(
    left: Tuple[int, int, int],
    right: Tuple[int, int, int],
) -> int:
    """PRIVATE: Return an exact dot product between integer triples.

    Parameters
    ----------
    left : Tuple[int, int, int]
        First integer triple.
    right : Tuple[int, int, int]
        Second integer triple.

    Returns
    -------
    result : int
        Exact integer dot product of the two triples.

    Notes
    -----
    Sum the three componentwise products in Python integer arithmetic,
    which is exact at any magnitude.
    """
    result: int = sum(
        left[index] * right[index] for index in range(_CELL_COMPONENTS)
    )
    return result


def _integer_determinant(
    rows: Tuple[
        Tuple[int, int, int],
        Tuple[int, int, int],
        Tuple[int, int, int],
    ],
) -> int:
    """PRIVATE: Return the exact determinant of three integer row vectors.

    Parameters
    ----------
    rows : Tuple[Tuple[int, int, int], Tuple[int, int, int], \
Tuple[int, int, int]]
        Three integer row vectors of the coefficient matrix.

    Returns
    -------
    determinant : int
        Exact integer determinant of the 3 x 3 coefficient matrix.

    Notes
    -----
    Expand the determinant along the first row in Python integer
    arithmetic, which is exact at any magnitude.
    """
    first: Tuple[int, int, int]
    second: Tuple[int, int, int]
    third: Tuple[int, int, int]
    first, second, third = rows
    determinant: int = (
        first[0] * (second[1] * third[2] - second[2] * third[1])
        - first[1] * (second[0] * third[2] - second[2] * third[0])
        + first[2] * (second[0] * third[1] - second[1] * third[0])
    )
    return determinant


def _validate_surface_cell_structure(
    in_plane_vectors: Float64[Array, "2 3"],
    stacking_vector: Float64[Array, " 3"],
    rotation: Float64[Array, "3 3"],
    interlayer_spacing_ang: Float64[Array, ""],
    miller: Tuple[int, int, int],
    in_plane_coeffs: Tuple[
        Tuple[int, int, int],
        Tuple[int, int, int],
    ],
    stacking_coeffs: Tuple[int, int, int],
) -> None:
    """PRIVATE: Validate surface-cell shapes and exact integer invariants.

    Implementation Logic
    --------------------
    Check the traced arrays only through ``ndim`` and ``shape``. Then
    check the integer metadata with ``_validate_integer_triple``,
    ``_integer_dot``, and ``_integer_determinant``, which stay exact at
    any magnitude.

    Parameters
    ----------
    in_plane_vectors : Float64[Array, "2 3"]
        Cartesian in-plane surface vectors in Angstrom, as rows.
    stacking_vector : Float64[Array, " 3"]
        Cartesian stacking vector in Angstrom.
    rotation : Float64[Array, "3 3"]
        Active Cartesian rotation from the bulk to the surface frame.
    interlayer_spacing_ang : Float64[Array, ""]
        Interlayer spacing in Angstrom.
    miller : Tuple[int, int, int]
        GCD-reduced Miller tuple.
    in_plane_coeffs : Tuple[Tuple[int, int, int], Tuple[int, int, int]]
        Exact bulk-lattice coefficients of the in-plane vectors.
    stacking_coeffs : Tuple[int, int, int]
        Exact bulk-lattice coefficients of the stacking vector.

    Raises
    ------
    ValueError
        If a numerical field has the wrong shape. If the coefficient
        metadata lacks exact integer triples. If an in-plane row lies
        outside the Miller plane. If the stacking dot product differs
        from one. If the three coefficient rows are linearly
        dependent. This is the static construction-time contract.
    """
    if (
        in_plane_vectors.ndim != _SURFACE_VECTOR_COUNT
        or in_plane_vectors.shape != (_SURFACE_VECTOR_COUNT, _CELL_COMPONENTS)
    ):
        message: str = "in_plane_vectors must have shape (2, 3)"
        raise ValueError(message)
    if stacking_vector.ndim != 1 or stacking_vector.shape != (3,):
        message = "stacking_vector must have shape (3,)"
        raise ValueError(message)
    if rotation.ndim != _SURFACE_VECTOR_COUNT or rotation.shape != (
        _CELL_COMPONENTS,
        _CELL_COMPONENTS,
    ):
        message = "rotation must have shape (3, 3)"
        raise ValueError(message)
    if interlayer_spacing_ang.ndim != 0:
        message = "interlayer_spacing_ang must be scalar"
        raise ValueError(message)

    _validate_integer_triple(miller, "miller")
    if (
        type(in_plane_coeffs) is not tuple
        or len(in_plane_coeffs) != _SURFACE_VECTOR_COUNT
    ):
        message = "in_plane_coeffs must contain two integer triples"
        raise ValueError(message)
    _validate_integer_triple(in_plane_coeffs[0], "in_plane_coeffs[0]")
    _validate_integer_triple(in_plane_coeffs[1], "in_plane_coeffs[1]")
    _validate_integer_triple(stacking_coeffs, "stacking_coeffs")

    if (
        _integer_dot(miller, in_plane_coeffs[0]) != 0
        or _integer_dot(
            miller,
            in_plane_coeffs[1],
        )
        != 0
    ):
        message = "in_plane_coeffs must lie in the Miller plane"
        raise ValueError(message)
    if _integer_dot(miller, stacking_coeffs) != 1:
        message = (
            "the gcd-reduced miller tuple dotted with stacking_coeffs "
            "must equal one"
        )
        raise ValueError(message)
    coefficient_rows: Tuple[
        Tuple[int, int, int],
        Tuple[int, int, int],
        Tuple[int, int, int],
    ] = (
        in_plane_coeffs[0],
        in_plane_coeffs[1],
        stacking_coeffs,
    )
    if _integer_determinant(coefficient_rows) == 0:
        message = (
            "surface-cell coefficient tuples must be linearly independent"
        )
        raise ValueError(message)


def _validate_slab_spec_structure(
    surface_cell: "SurfaceCell",
    thickness_ang: float,
    vacuum_ang: float,
    fine: Tuple[float, float],
    termination: Tuple[str, str],
    n_layers: int,
    bulk_atom_of_slab_atom: Tuple[int, ...],
    layer_of_slab_atom: Tuple[int, ...],
) -> None:
    """PRIVATE: Validate static slab provenance and selection metadata.

    Implementation Logic
    --------------------
    Use exact ``type`` comparisons on the plain Python metadata and
    ``math.isfinite`` on the float choices. The two provenance tuples
    must agree elementwise on length so each slab atom keeps one bulk
    atom and one layer.

    Parameters
    ----------
    surface_cell : "SurfaceCell"
        Validated surface-frame carrier.
    thickness_ang : float
        Requested slab thickness in Angstrom.
    vacuum_ang : float
        Vacuum padding in Angstrom.
    fine : Tuple[float, float]
        Top and bottom cut shifts in Angstrom.
    termination : Tuple[str, str]
        Top and bottom species labels.
    n_layers : int
        Number of slab layers.
    bulk_atom_of_slab_atom : Tuple[int, ...]
        Bulk-atom provenance index for each slab atom.
    layer_of_slab_atom : Tuple[int, ...]
        Layer index for each slab atom.

    Raises
    ------
    ValueError
        If ``surface_cell`` has the wrong type. If a length is not a
        finite nonnegative float. If ``fine`` or ``termination`` has
        invalid pair metadata. If ``n_layers`` is not positive. If the
        provenance maps have invalid lengths, integers, or layer
        bounds. This is the static construction-time contract.
    """
    if not isinstance(surface_cell, SurfaceCell):
        message: str = "surface_cell must be a SurfaceCell"
        raise ValueError(message)
    if (
        type(thickness_ang) is not float
        or type(vacuum_ang) is not float
        or not math.isfinite(thickness_ang)
        or not math.isfinite(vacuum_ang)
    ):
        message = "thickness_ang and vacuum_ang must be finite floats"
        raise ValueError(message)
    if thickness_ang < 0.0 or vacuum_ang < 0.0:
        message = "thickness_ang and vacuum_ang must both be nonnegative"
        raise ValueError(message)
    if (
        type(fine) is not tuple
        or len(fine) != _SURFACE_VECTOR_COUNT
        or any(type(value) is not float for value in fine)
        or not all(math.isfinite(value) for value in fine)
    ):
        message = "fine must contain two finite floats"
        raise ValueError(message)
    if (
        type(termination) is not tuple
        or len(termination) != _SURFACE_VECTOR_COUNT
        or any(type(species) is not str for species in termination)
    ):
        message = "termination must contain two species labels"
        raise ValueError(message)
    if type(n_layers) is not int or n_layers <= 0:
        message = "n_layers must be a positive integer"
        raise ValueError(message)
    if any(
        type(values) is not tuple
        for values in (bulk_atom_of_slab_atom, layer_of_slab_atom)
    ):
        message = "slab provenance maps must be tuples"
        raise ValueError(message)
    if len(bulk_atom_of_slab_atom) != len(layer_of_slab_atom):
        message = "slab provenance maps must have the same length"
        raise ValueError(message)
    if any(
        type(index) is not int or index < 0 for index in bulk_atom_of_slab_atom
    ):
        message = "bulk_atom_of_slab_atom must contain nonnegative integers"
        raise ValueError(message)
    if any(
        type(layer) is not int or layer < 0 or layer >= n_layers
        for layer in layer_of_slab_atom
    ):
        message = "layer_of_slab_atom entries must lie in [0, n_layers)"
        raise ValueError(message)


def _validate_slab_topology_structure(  # noqa: DOC503, PLR0912, PLR0913, PLR0915
    *,
    miller: Tuple[int, int, int],
    in_plane_coeffs: Tuple[
        Tuple[int, int, int],
        Tuple[int, int, int],
    ],
    stacking_coeffs: Tuple[int, int, int],
    atom_shifts: Tuple[Tuple[int, int, int], ...],
    bulk_atom_of_slab_atom: Tuple[int, ...],
    layer_of_slab_atom: Tuple[int, ...],
    termination: Tuple[str, str],
    thickness_ang: float,
    vacuum_ang: float,
    fine: Tuple[float, float],
    n_layers: int,
    bulk_atom_count: int,
    basis_atom_indices: Tuple[int, ...],
) -> None:
    """PRIVATE: Validate static topology selected for a slab rebuild.

    Parameters
    ----------
    miller : Tuple[int, int, int]
        GCD-reduced surface Miller tuple.
    in_plane_coeffs : Tuple[Tuple[int, int, int], Tuple[int, int, int]]
        Exact in-plane lattice-coefficient rows.
    stacking_coeffs : Tuple[int, int, int]
        Exact one-plane stacking coefficient row.
    atom_shifts : Tuple[Tuple[int, int, int], ...]
        Surface-cell shift for every source bulk atom.
    bulk_atom_of_slab_atom : Tuple[int, ...]
        Bulk-atom provenance for every slab atom.
    layer_of_slab_atom : Tuple[int, ...]
        Layer index for every slab atom.
    termination : Tuple[str, str]
        Top and bottom termination labels.
    thickness_ang : float
        Requested slab thickness in Angstrom.
    vacuum_ang : float
        Vacuum padding in Angstrom.
    fine : Tuple[float, float]
        Top and bottom cut shifts in Angstrom.
    n_layers : int
        Positive number of selected layers.
    bulk_atom_count : int
        Positive atom count of the source bulk geometry.
    basis_atom_indices : Tuple[int, ...]
        Bulk-atom assignment for every source basis orbital.

    Raises
    ------
    ValueError
        If exact surface coefficients, atom provenance, layer bounds, lengths,
        termination metadata, or finite slab dimensions are inconsistent.

    Implementation Logic
    --------------------
    Validate the exact surface basis first. Then require one coordinate shift
    per frozen bulk atom and paired provenance for every selected slab atom.
    Finally, check all bulk and layer indices against their frozen bounds.
    """
    _validate_integer_triple(miller, "miller")
    if (
        type(in_plane_coeffs) is not tuple
        or len(in_plane_coeffs) != _SURFACE_VECTOR_COUNT
    ):
        message: str = "in_plane_coeffs must contain two integer triples"
        raise ValueError(message)
    _validate_integer_triple(in_plane_coeffs[0], "in_plane_coeffs[0]")
    _validate_integer_triple(in_plane_coeffs[1], "in_plane_coeffs[1]")
    _validate_integer_triple(stacking_coeffs, "stacking_coeffs")
    if (
        _integer_dot(miller, in_plane_coeffs[0]) != 0
        or _integer_dot(miller, in_plane_coeffs[1]) != 0
    ):
        message = "in_plane_coeffs must lie in the Miller plane"
        raise ValueError(message)
    if _integer_dot(miller, stacking_coeffs) != 1:
        message = "miller dotted with stacking_coeffs must equal one"
        raise ValueError(message)
    coefficient_rows: Tuple[
        Tuple[int, int, int],
        Tuple[int, int, int],
        Tuple[int, int, int],
    ] = (in_plane_coeffs[0], in_plane_coeffs[1], stacking_coeffs)
    if _integer_determinant(coefficient_rows) == 0:
        message = "slab topology coefficient rows must be independent"
        raise ValueError(message)
    if type(atom_shifts) is not tuple:
        message = "atom_shifts must be a tuple of integer triples"
        raise ValueError(message)
    atom_shift: Tuple[int, int, int]
    for atom_shift in atom_shifts:
        _validate_integer_triple(atom_shift, "atom_shifts entry")
    if any(
        type(values) is not tuple
        for values in (
            bulk_atom_of_slab_atom,
            layer_of_slab_atom,
            basis_atom_indices,
        )
    ):
        message = "slab topology provenance maps must be tuples"
        raise ValueError(message)
    if len(bulk_atom_of_slab_atom) != len(layer_of_slab_atom):
        message = "slab atom provenance maps must agree in length"
        raise ValueError(message)
    if (
        type(termination) is not tuple
        or len(termination) != _SURFACE_VECTOR_COUNT
        or any(type(species) is not str for species in termination)
    ):
        message = "termination must contain two species labels"
        raise ValueError(message)
    if (
        type(thickness_ang) is not float
        or type(vacuum_ang) is not float
        or not math.isfinite(thickness_ang)
        or not math.isfinite(vacuum_ang)
        or thickness_ang < 0.0
        or vacuum_ang < 0.0
    ):
        message = "thickness_ang and vacuum_ang must be finite and nonnegative"
        raise ValueError(message)
    if (
        type(fine) is not tuple
        or len(fine) != _SURFACE_VECTOR_COUNT
        or any(type(value) is not float for value in fine)
        or not all(math.isfinite(value) for value in fine)
    ):
        message = "fine must contain two finite floats"
        raise ValueError(message)
    if type(n_layers) is not int or n_layers <= 0:
        message = "n_layers must be a positive integer"
        raise ValueError(message)
    if type(bulk_atom_count) is not int or bulk_atom_count <= 0:
        message = "bulk_atom_count must be a positive integer"
        raise ValueError(message)
    if len(atom_shifts) != bulk_atom_count:
        message = "atom_shifts must contain one entry per frozen bulk atom"
        raise ValueError(message)
    if any(
        type(index) is not int or index < 0 or index >= bulk_atom_count
        for index in bulk_atom_of_slab_atom
    ):
        message = "bulk atom provenance must lie in the frozen bulk geometry"
        raise ValueError(message)
    if any(
        type(layer) is not int or layer < 0 or layer >= n_layers
        for layer in layer_of_slab_atom
    ):
        message = "layer provenance must lie in the frozen layer range"
        raise ValueError(message)
    if any(
        type(index) is not int or index < 0 or index >= bulk_atom_count
        for index in basis_atom_indices
    ):
        message = "basis atom indices must lie in the frozen bulk geometry"
        raise ValueError(message)


def _validate_basis_geometry(
    basis: OrbitalBasis,
    geometry: CrystalGeometry,
) -> None:
    """PRIVATE: Validate the orbital-to-atom mapping against a geometry.

    Parameters
    ----------
    basis : OrbitalBasis
        Orbital metadata that supplies ``atom_indices`` for checking.
    geometry : CrystalGeometry
        Crystal geometry that provides the atomic position rows.

    Raises
    ------
    ValueError
        If any ``basis.atom_indices`` entry is not a valid row index of
        ``geometry.positions``. This is the static construction-time
        contract.

    Notes
    -----
    Compare each index against the static atom count
    ``geometry.positions.shape[0]``. Nonnegativity is already
    guaranteed by the ``OrbitalBasis`` invariants.
    """
    n_atoms: int = geometry.positions.shape[0]
    if any(index >= n_atoms for index in basis.atom_indices):
        message: str = (
            "basis atom_indices must refer to geometry.positions rows"
        )
        raise ValueError(message)


def _validate_hopping_metadata(
    hopping_pairs: Tuple[Tuple[int, int], ...],
    hopping_cells: Tuple[Tuple[int, int, int], ...],
    n_orbitals: int,
) -> Tuple[int, ...]:
    """PRIVATE: Validate exact connectivity and derive its closure permutation.

    Implementation Logic
    --------------------
    Build the ``(i, j, R)`` key of every record, then bucket record
    indices by key. For every key require a reversed-key bucket of
    equal size and map matching occurrences positionally. The resulting
    permutation lets the factory compare each amplitude with the
    conjugate of its Hermitian partner.

    Parameters
    ----------
    hopping_pairs : Tuple[Tuple[int, int], ...]
        Directed orbital pairs ``(i, j)``, one per hopping record.
    hopping_cells : Tuple[Tuple[int, int, int], ...]
        Exact integer lattice translations ``R``, one per record.
    n_orbitals : int
        Number of orbitals that bounds the pair indices.

    Returns
    -------
    closure_permutation : Tuple[int, ...]
        Index of the ``(j, i, -R)`` partner record for every
        ``(i, j, R)`` record.

    Raises
    ------
    ValueError
        If a pair or cell has invalid integer metadata. If a pair index
        lies outside ``[0, n_orbitals)``. If duplicate records exist.
        If any record lacks a ``(j, i, -R)`` partner. This is the static
        construction-time contract.
    """
    keys: List[Tuple[int, int, Tuple[int, int, int]]] = []
    pair: Tuple[int, int]
    cell: Tuple[int, int, int]
    for pair, cell in zip(hopping_pairs, hopping_cells, strict=True):
        if (
            type(pair) is not tuple
            or len(pair) != _PAIR_LENGTH
            or any(type(index) is not int for index in pair)
        ):
            message: str = "hopping_pairs must contain pairs of integers"
            raise ValueError(message)
        if (
            type(cell) is not tuple
            or len(cell) != _CELL_COMPONENTS
            or any(type(component) is not int for component in cell)
        ):
            message = "hopping_cells must contain integer triples"
            raise ValueError(message)
        if any(index < 0 or index >= n_orbitals for index in pair):
            message = "hopping pair indices must be in [0, n_orbitals)"
            raise ValueError(message)
        keys.append((pair[0], pair[1], cell))

    if len(set(keys)) != len(keys):
        message = "duplicate (i, j, R) hopping records are not allowed"
        raise ValueError(message)

    buckets: Dict[
        Tuple[int, int, Tuple[int, int, int]],
        List[int],
    ] = {}
    index: int
    key: Tuple[int, int, Tuple[int, int, int]]
    for index, key in enumerate(keys):
        buckets.setdefault(key, []).append(index)

    closure: List[int] = [-1] * len(keys)
    indices: List[int]
    for key, indices in buckets.items():
        orbital_i: int
        orbital_j: int
        orbital_i, orbital_j, cell = key
        reverse_key: Tuple[int, int, Tuple[int, int, int]] = (
            orbital_j,
            orbital_i,
            (-cell[0], -cell[1], -cell[2]),
        )
        reverse_indices: List[int] | None = buckets.get(reverse_key)
        if reverse_indices is None or len(reverse_indices) != len(indices):
            message = (
                "hopping metadata must be Hermitian-closed with one "
                "(j, i, -R) partner per (i, j, R) entry"
            )
            raise ValueError(message)
        occurrence: int
        for occurrence, index in enumerate(indices):
            closure[index] = reverse_indices[occurrence]
    closure_permutation: Tuple[int, ...] = tuple(closure)
    return closure_permutation


def _validate_shell_metadata(
    soc_lambdas: Float64[Array, " n_shells"],
    basis: OrbitalBasis,
    shell_index: Tuple[int, ...],
) -> None:
    """PRIVATE: Validate contiguous atomic-shell identifiers and their groups.

    Implementation Logic
    --------------------
    Walk every orbital with a nonnegative shell and record its
    ``(atom_index, n, l)`` group in two dictionaries. Reject a shell
    that mixes groups and a group that is split across shells. Spin
    copies of one group therefore share one ``soc_lambdas`` entry.

    Parameters
    ----------
    soc_lambdas : Float64[Array, " n_shells"]
        Atomic spin-orbit couplings in eV, one per shell.
    basis : OrbitalBasis
        Orbital metadata that provides ``(atom, n, l)`` per orbital.
    shell_index : Tuple[int, ...]
        Orbital-to-SOC-shell map; ``-1`` denotes no shell.

    Raises
    ------
    ValueError
        If ``shell_index`` contains an integer below ``-1``. If
        ``soc_lambdas`` has the wrong length. If nonnegative identifiers
        are not contiguous from zero. If shells and ``(atom, n, l)``
        groups lack a one-to-one map. This is the static
        construction-time contract.
    """
    if any(type(index) is not int or index < -1 for index in shell_index):
        message: str = (
            "shell_index entries must be integers greater than or equal to -1"
        )
        raise ValueError(message)
    expected_shells: int = max(shell_index, default=-1) + 1
    if soc_lambdas.shape[0] != expected_shells:
        message = (
            "soc_lambdas length must equal max(shell_index) + 1, with -1 "
            "denoting no shell"
        )
        raise ValueError(message)
    active_shells: set[int] = {index for index in shell_index if index >= 0}
    if active_shells != set(range(expected_shells)):
        message = "nonnegative shell_index IDs must be contiguous from 0"
        raise ValueError(message)

    shell_groups: Dict[int, Tuple[int, int, int]] = {}
    group_shells: Dict[Tuple[int, int, int], int] = {}
    orbital: int
    shell: int
    for orbital, shell in enumerate(shell_index):
        if shell < 0:
            continue
        group: Tuple[int, int, int] = (
            basis.atom_indices[orbital],
            basis.n[orbital],
            basis.l[orbital],
        )
        existing_group: Tuple[int, int, int] | None = shell_groups.get(shell)
        if existing_group is not None and existing_group != group:
            message = "each shell_index ID must map to one (atom, n, l) group"
            raise ValueError(message)
        existing_shell: int | None = group_shells.get(group)
        if existing_shell is not None and existing_shell != shell:
            message = "each (atom, n, l) group must map to one shell_index ID"
            raise ValueError(message)
        shell_groups[shell] = group
        group_shells[group] = shell


def _validate_tb_structure(  # noqa: PLR0913, PLR0917
    hopping_amplitudes: Complex128[Array, " n_hop"],
    onsite_energies: Float64[Array, " n_orb"],
    soc_lambdas: Float64[Array, " n_shells"],
    geometry: CrystalGeometry,
    basis: OrbitalBasis,
    hopping_pairs: Tuple[Tuple[int, int], ...],
    hopping_cells: Tuple[Tuple[int, int, int], ...],
    shell_index: Tuple[int, ...],
    spinor: bool,
    orbital_positions: Optional[Float64[Array, "n_orb 3"]],
    depths: Optional[Float64[Array, " n_depth"]],
) -> Tuple[int, ...]:
    """PRIVATE: Validate tight-binding structure and return its closure.

    Implementation Logic
    --------------------
    Check types, ranks, and axis agreements first. Then delegate to
    ``_validate_basis_geometry``, ``_validate_depths_shape``,
    ``_validate_shell_metadata``, and ``_validate_hopping_metadata``,
    and return the closure permutation of the last one.

    Parameters
    ----------
    hopping_amplitudes : Complex128[Array, " n_hop"]
        Complex hopping amplitudes in eV, one per hopping record.
    onsite_energies : Float64[Array, " n_orb"]
        Onsite orbital energies in eV.
    soc_lambdas : Float64[Array, " n_shells"]
        Atomic spin-orbit couplings in eV, one per shell.
    geometry : CrystalGeometry
        Differentiable lattice and atomic positions.
    basis : OrbitalBasis
        Orbital-to-atom and quantum-number metadata.
    hopping_pairs : Tuple[Tuple[int, int], ...]
        Directed orbital pairs ``(i, j)``, one per record.
    hopping_cells : Tuple[Tuple[int, int, int], ...]
        Exact integer lattice translations ``R``, one per record.
    shell_index : Tuple[int, ...]
        Orbital-to-SOC-shell map; ``-1`` denotes no shell.
    spinor : bool
        Whether the basis carries explicit spin channels.
    orbital_positions : Optional[Float64[Array, "n_orb 3"]]
        Explicit fractional orbital centres, or ``None``.
    depths : Optional[Float64[Array, " n_depth"]]
        Orbital depths in Angstrom, or ``None`` for a bulk model.

    Returns
    -------
    closure : Tuple[int, ...]
        Hermitian-closure permutation: the index of the ``(j, i, -R)``
        partner record for every hopping record.

    Raises
    ------
    ValueError
        If geometry, basis, or connectivity metadata has the wrong
        type. If a numerical array has the wrong rank or length. If
        ``orbital_positions`` has the wrong shape. If ``spinor`` is not
        Boolean or contradicts the spin channels. The delegated
        validators raise ``ValueError`` for their own static contracts.
    """
    if not isinstance(geometry, CrystalGeometry):
        message: str = "geometry must be a CrystalGeometry"
        raise ValueError(message)
    if not isinstance(basis, OrbitalBasis):
        message = "basis must be an OrbitalBasis"
        raise ValueError(message)
    if any(
        type(values) is not tuple
        for values in (hopping_pairs, hopping_cells, shell_index)
    ):
        message = (
            "hopping_pairs, hopping_cells, and shell_index must be tuples"
        )
        raise ValueError(message)
    if hopping_amplitudes.ndim != 1:
        message = "hopping_amplitudes must be one-dimensional"
        raise ValueError(message)
    if onsite_energies.ndim != 1:
        message = "onsite_energies must be one-dimensional"
        raise ValueError(message)
    if soc_lambdas.ndim != 1:
        message = "soc_lambdas must be one-dimensional"
        raise ValueError(message)

    n_hoppings: int = hopping_amplitudes.shape[0]
    n_orbitals: int = onsite_energies.shape[0]
    if len(hopping_pairs) != n_hoppings or len(hopping_cells) != n_hoppings:
        message = (
            "hopping_amplitudes, hopping_pairs, and hopping_cells must have "
            "the same length"
        )
        raise ValueError(message)
    if len(basis.n) != n_orbitals or len(shell_index) != n_orbitals:
        message = (
            "onsite_energies, basis, and shell_index must have the same "
            "orbital count"
        )
        raise ValueError(message)
    _validate_basis_geometry(basis, geometry)
    if orbital_positions is not None and (
        orbital_positions.ndim != _ORBITAL_POSITION_NDIM
        or orbital_positions.shape != (n_orbitals, _CELL_COMPONENTS)
    ):
        message = "orbital_positions must have shape (n_orbitals, 3)"
        raise ValueError(message)
    _validate_depths_shape(depths, n_orbitals)

    _validate_shell_metadata(soc_lambdas, basis, shell_index)
    if type(spinor) is not bool:
        message = "spinor must be a bool"
        raise ValueError(message)
    if spinor and (
        len(basis.spin) != n_orbitals
        or any(channel not in (-1, 1) for channel in basis.spin)
    ):
        message = "spinor models require one +1 or -1 basis spin per orbital"
        raise ValueError(message)
    if not spinor and basis.spin:
        message = "spinless models require an empty basis spin tuple"
        raise ValueError(message)

    closure: Tuple[int, ...] = _validate_hopping_metadata(
        hopping_pairs,
        hopping_cells,
        n_orbitals,
    )
    return closure


def _checked_geometry(
    geometry: CrystalGeometry, context: str
) -> CrystalGeometry:
    """PRIVATE: Attach finite-value runtime checks to every geometry leaf.

    Parameters
    ----------
    geometry : CrystalGeometry
        Geometry that supplies ``lattice``, ``reciprocal``, and
        ``positions`` leaves for guarding.
    context : str
        Caller name used as the prefix of each traced error message.

    Returns
    -------
    checked : CrystalGeometry
        The same geometry with the runtime checks attached.

    Notes
    -----
    Attach one traced ``eqx.error_if`` guard per array leaf instead of
    raising a static ``ValueError``. Each guard fails at run time under
    JIT when its leaf contains a nonfinite element. Rebuild the carrier
    with ``eqx.tree_at`` so that the guarded leaves replace the
    originals.
    """
    lattice: Float64[Array, "3 3"] = eqx.error_if(
        geometry.lattice,
        ~jnp.all(jnp.isfinite(geometry.lattice)),
        f"{context}: geometry lattice finite",
    )
    reciprocal: Float64[Array, "3 3"] = eqx.error_if(
        geometry.reciprocal,
        ~jnp.all(jnp.isfinite(geometry.reciprocal)),
        f"{context}: geometry reciprocal finite",
    )
    positions: Float64[Array, "n_atoms 3"] = eqx.error_if(
        geometry.positions,
        ~jnp.all(jnp.isfinite(geometry.positions)),
        f"{context}: geometry positions finite",
    )
    checked: CrystalGeometry = eqx.tree_at(
        lambda item: (item.lattice, item.reciprocal, item.positions),
        geometry,
        (lattice, reciprocal, positions),
    )
    return checked


class DiagonalizedBands(eqx.Module):
    """Store diagonalized electronic-structure data in a JAX PyTree.

    The carrier is the tight-binding-to-ARPES interface. Geometry and orbital
    metadata travel with each eigensystem so later matrix-element stages can
    form Cartesian momenta, atomic interference phases, and orbital operators.

    :see: :class:`~.test_tb_model.TestDiagonalizedBands`

    Attributes
    ----------
    eigenvalues : Float64[Array, "n_k n_bands"]
        Band energies in eV.
    eigenvectors : Complex128[Array, "n_k n_bands n_orb"]
        Complex orbital coefficients in the basis-position gauge.
    kpoints : Float64[Array, "n_k 3"]
        Fractional reciprocal-space coordinates.
    fermi_energy : Float64[Array, ""]
        Fermi energy in eV.
    geometry : CrystalGeometry
        Crystal geometry. Its numerical fields are differentiable children.
    basis : OrbitalBasis
        Orbital and atom metadata (**static** -- changing it triggers
        retracing).
    orbital_positions : Optional[Float64[Array, "n_orb 3"]]
        Explicit fractional orbital centres associated with the
        basis-position-gauge coefficients. ``None`` ties centres to atoms.
    depths : Optional[Float64[Array, "n_orb"]]
        Orbital depths in Angstrom below the top surface. ``None`` denotes a
        bulk model. Native tight-binding diagonalization propagates this
        differentiable leaf without transformation.

    Notes
    -----
    The numerical eigensystem, geometry, and optional orbital-position fields
    remain JAX leaves. ``basis`` is static because its quantum numbers and
    atom mapping shape compiled operator construction.

    See Also
    --------
    TBModel : Tight-binding carrier whose diagonalization produces bands.
    make_diagonalized_bands : Validating carrier factory.
    """

    eigenvalues: Float64[Array, "n_k n_bands"]
    eigenvectors: Complex128[Array, "n_k n_bands n_orb"]
    kpoints: Float64[Array, "n_k 3"]
    fermi_energy: Float64[Array, ""]
    geometry: CrystalGeometry
    basis: OrbitalBasis = eqx.field(static=True)
    orbital_positions: Optional[Float64[Array, "n_orb 3"]] = None
    depths: Optional[Float64[Array, " n_orb"]] = None

    def __check_init__(self) -> None:
        """Validate the static eigensystem invariants again."""
        _validate_diagonalized_structure(
            self.eigenvalues,
            self.eigenvectors,
            self.kpoints,
            self.fermi_energy,
            self.geometry,
            self.basis,
            self.orbital_positions,
            self.depths,
        )


class TBModel(eqx.Module):
    r"""Store tight-binding parameters in a JAX PyTree.

    Each hopping record is ``(i, j, R, t)`` with exact integer lattice
    translation ``R``. In the pinned basis-position gauge, Hamiltonian phases
    use the physical fractional displacement
    :math:`R + \tau_j - \tau_i`. Explicit ``orbital_positions`` provide one
    :math:`\tau` per Wannier orbital. Otherwise they are derived from
    ``geometry.positions`` and ``basis.atom_indices``. Physical displacements
    are never stored in place of ``R`` or rounded back into connectivity.

    :see: :class:`~.test_tb_model.TestTBModel`

    Attributes
    ----------
    hopping_amplitudes : Complex128[Array, "n_hop"]
        Complex hopping amplitudes in eV. These differentiable values support
        spin-orbit and other intrinsically complex couplings.
    onsite_energies : Float64[Array, "n_orb"]
        Onsite orbital energies in eV.
    soc_lambdas : Float64[Array, "n_shells"]
        Atomic spin-orbit couplings in eV, one per ``(atom, n, l)`` shell.
    geometry : CrystalGeometry
        Differentiable lattice and fractional atomic positions.
    basis : OrbitalBasis
        Orbital-to-atom and quantum-number metadata (**static** -- changing it
        triggers retracing).
    hopping_pairs : Tuple[Tuple[int, int], ...]
        Directed orbital pairs ``(i, j)`` (**static** -- changing them triggers
        retracing).
    hopping_cells : Tuple[Tuple[int, int, int], ...]
        Exact integer translations ``R`` (**static** -- changing them triggers
        retracing).
    shell_index : Tuple[int, ...]
        Orbital-to-SOC-shell mapping; ``-1`` means no shell. Nonnegative IDs
        are contiguous and map one-to-one to ``(atom, n, l)`` groups, with
        spin copies sharing an ID (**static** -- changing it triggers
        retracing).
    spinor : bool
        Whether the basis carries explicit spin channels (**static** --
        changing it triggers retracing).
    orbital_positions : Optional[Float64[Array, "n_orb 3"]]
        Explicit fractional orbital or Wannier centres. ``None`` ties every
        orbital centre to its assigned atomic position. Explicit centres are
        differentiable independently of the atomic geometry.
    depths : Optional[Float64[Array, "n_orb"]]
        Orbital depths in Angstrom below the top surface. Values are finite
        and nonnegative up to the numerical boundary tolerance. ``None``
        denotes a bulk model.

    Notes
    -----
    Hopping metadata excludes duplicate ``(i, j, R)`` records and includes a
    ``(j, i, -R)`` partner for every entry. The factory checks corresponding
    amplitudes elementwise against their complex conjugates. Hamiltonian
    assembly therefore needs no Hermitianization repair.

    See Also
    --------
    DiagonalizedBands : Eigensystem carrier produced from this model.
    make_tb_model : Validating carrier factory.
    """

    hopping_amplitudes: Complex128[Array, " n_hop"]
    onsite_energies: Float64[Array, " n_orb"]
    soc_lambdas: Float64[Array, " n_shells"]
    geometry: CrystalGeometry
    basis: OrbitalBasis = eqx.field(static=True)
    hopping_pairs: Tuple[Tuple[int, int], ...] = eqx.field(static=True)
    hopping_cells: Tuple[Tuple[int, int, int], ...] = eqx.field(static=True)
    shell_index: Tuple[int, ...] = eqx.field(static=True)
    spinor: bool = eqx.field(static=True)
    orbital_positions: Optional[Float64[Array, "n_orb 3"]] = None
    depths: Optional[Float64[Array, " n_orb"]] = None

    def __check_init__(self) -> None:
        """Validate the static tight-binding invariants again."""
        _validate_tb_structure(
            self.hopping_amplitudes,
            self.onsite_energies,
            self.soc_lambdas,
            self.geometry,
            self.basis,
            self.hopping_pairs,
            self.hopping_cells,
            self.shell_index,
            self.spinor,
            self.orbital_positions,
            self.depths,
        )


class SlabTopology(eqx.Module):
    """Store host-selected discrete slab topology for pure-JAX rebuilding.

    The carrier contains only static integer choices, endpoint metadata, and
    design values selected before a JAX transformation.

    :see: :class:`~.test_tb_model.TestSlabTopology`

    Attributes
    ----------
    miller : Tuple[int, int, int]
        GCD-reduced Miller tuple (**static**).
    in_plane_coeffs : Tuple[Tuple[int, int, int], Tuple[int, int, int]]
        Exact coefficient rows for the surface plane (**static**).
    stacking_coeffs : Tuple[int, int, int]
        Exact coefficient row for one stacking period (**static**).
    atom_shifts : Tuple[Tuple[int, int, int], ...]
        Surface-cell shift for every source bulk atom (**static**).
    bulk_atom_of_slab_atom : Tuple[int, ...]
        Bulk-atom provenance for every slab atom (**static**).
    layer_of_slab_atom : Tuple[int, ...]
        Layer index for every slab atom (**static**).
    termination : Tuple[str, str]
        Top and bottom species labels (**static**).
    thickness_ang : float
        Requested slab thickness in Angstrom (**static**).
    vacuum_ang : float
        Vacuum padding in Angstrom (**static**).
    fine : Tuple[float, float]
        Top and bottom cut shifts in Angstrom (**static**).
    n_layers : int
        Number of selected layers (**static**).
    bulk_atom_count : int
        Atom count of the source bulk geometry (**static**).
    basis_atom_indices : Tuple[int, ...]
        Bulk-atom assignment for every source orbital (**static**).

    See Also
    --------
    make_slab_topology : Validated factory for this type.
    """

    miller: Tuple[int, int, int] = eqx.field(static=True)
    in_plane_coeffs: Tuple[
        Tuple[int, int, int],
        Tuple[int, int, int],
    ] = eqx.field(static=True)
    stacking_coeffs: Tuple[int, int, int] = eqx.field(static=True)
    atom_shifts: Tuple[Tuple[int, int, int], ...] = eqx.field(static=True)
    bulk_atom_of_slab_atom: Tuple[int, ...] = eqx.field(static=True)
    layer_of_slab_atom: Tuple[int, ...] = eqx.field(static=True)
    termination: Tuple[str, str] = eqx.field(static=True)
    thickness_ang: float = eqx.field(static=True)
    vacuum_ang: float = eqx.field(static=True)
    fine: Tuple[float, float] = eqx.field(static=True)
    n_layers: int = eqx.field(static=True)
    bulk_atom_count: int = eqx.field(static=True)
    basis_atom_indices: Tuple[int, ...] = eqx.field(static=True)

    def __check_init__(self) -> None:
        """Validate frozen slab topology on direct construction."""
        _validate_slab_topology_structure(
            miller=self.miller,
            in_plane_coeffs=self.in_plane_coeffs,
            stacking_coeffs=self.stacking_coeffs,
            atom_shifts=self.atom_shifts,
            bulk_atom_of_slab_atom=self.bulk_atom_of_slab_atom,
            layer_of_slab_atom=self.layer_of_slab_atom,
            termination=self.termination,
            thickness_ang=self.thickness_ang,
            vacuum_ang=self.vacuum_ang,
            fine=self.fine,
            n_layers=self.n_layers,
            bulk_atom_count=self.bulk_atom_count,
            basis_atom_indices=self.basis_atom_indices,
        )


class SurfaceCell(eqx.Module):
    """Store a validated Cartesian surface-cell frame.

    Keep traced Cartesian geometry together with exact Miller-frame
    coefficients selected by the host topology stage.

    :see: :class:`~.test_tb_model.TestSurfaceCell`

    Attributes
    ----------
    in_plane_vectors : Float64[Array, "2 3"]
        Cartesian in-plane vectors in Angstrom, as rows.
    stacking_vector : Float64[Array, "3"]
        Cartesian stacking vector in Angstrom.
    rotation : Float64[Array, "3 3"]
        Active Cartesian rotation from bulk to surface frame.
    interlayer_spacing_ang : Float64[Array, ""]
        Positive interlayer spacing in Angstrom.
    miller : Tuple[int, int, int]
        GCD-reduced Miller tuple (**static** -- changing it triggers
        retracing).
    in_plane_coeffs : Tuple[Tuple[int, int, int], Tuple[int, int, int]]
        Exact bulk-lattice coefficients of the in-plane vectors (**static**).
    stacking_coeffs : Tuple[int, int, int]
        Exact bulk-lattice coefficients of the stacking vector (**static**).

    Notes
    -----
    The integer coefficient rows are linearly independent, the in-plane rows
    are orthogonal to ``miller``, and ``miller · stacking_coeffs == 1``.

    See Also
    --------
    make_surface_cell : Validated factory for this type.
    """

    in_plane_vectors: Float64[Array, "2 3"]
    stacking_vector: Float64[Array, " 3"]
    rotation: Float64[Array, "3 3"]
    interlayer_spacing_ang: Float64[Array, ""]
    miller: Tuple[int, int, int] = eqx.field(static=True)
    in_plane_coeffs: Tuple[
        Tuple[int, int, int],
        Tuple[int, int, int],
    ] = eqx.field(static=True)
    stacking_coeffs: Tuple[int, int, int] = eqx.field(static=True)

    def __check_init__(self) -> None:
        """Validate exact surface-cell structure on direct construction."""
        _validate_surface_cell_structure(
            self.in_plane_vectors,
            self.stacking_vector,
            self.rotation,
            self.interlayer_spacing_ang,
            self.miller,
            self.in_plane_coeffs,
            self.stacking_coeffs,
        )


class SlabSpec(eqx.Module):
    """Store static slab construction choices and provenance.

    Carry the traced surface frame alongside immutable cut, layer, and
    bulk-to-slab provenance metadata.

    :see: :class:`~.test_tb_model.TestSlabSpec`

    Attributes
    ----------
    surface_cell : SurfaceCell
        Differentiable surface-frame carrier.
    thickness_ang : float
        Requested slab thickness in Angstrom (**static**).
    vacuum_ang : float
        Vacuum padding in Angstrom (**static**).
    fine : Tuple[float, float]
        Top and bottom cut shifts in Angstrom (**static**).
    termination : Tuple[str, str]
        Top and bottom species labels (**static**).
    n_layers : int
        Number of slab layers (**static**).
    bulk_atom_of_slab_atom : Tuple[int, ...]
        Bulk-atom provenance for each slab atom (**static**).
    layer_of_slab_atom : Tuple[int, ...]
        Layer index for each slab atom (**static**).

    See Also
    --------
    make_slab_spec : Validated factory for this type.
    """

    surface_cell: SurfaceCell
    thickness_ang: float = eqx.field(static=True)
    vacuum_ang: float = eqx.field(static=True)
    fine: Tuple[float, float] = eqx.field(static=True)
    termination: Tuple[str, str] = eqx.field(static=True)
    n_layers: int = eqx.field(static=True)
    bulk_atom_of_slab_atom: Tuple[int, ...] = eqx.field(static=True)
    layer_of_slab_atom: Tuple[int, ...] = eqx.field(static=True)

    def __check_init__(self) -> None:
        """Validate slab metadata invariants on direct construction."""
        _validate_slab_spec_structure(
            self.surface_cell,
            self.thickness_ang,
            self.vacuum_ang,
            self.fine,
            self.termination,
            self.n_layers,
            self.bulk_atom_of_slab_atom,
            self.layer_of_slab_atom,
        )


def _validate_diagonalized_structure(
    eigenvalues: Float64[Array, "n_k_e n_bands_e"],
    eigenvectors: Complex128[Array, "n_k_v n_bands_v n_orb"],
    kpoints: Float64[Array, "n_k_p 3"],
    fermi_energy: Float64[Array, ""],
    geometry: CrystalGeometry,
    basis: OrbitalBasis,
    orbital_positions: Optional[Float64[Array, "n_orb 3"]],
    depths: Optional[Float64[Array, " n_depth"]],
) -> None:
    """PRIVATE: Validate static eigensystem shapes and context.

    Implementation Logic
    --------------------
    Check types, ranks, and axis agreements through static shape
    metadata only. Then delegate the depth axis to
    ``_validate_depths_shape`` and the orbital-to-atom map to
    ``_validate_basis_geometry``.

    Parameters
    ----------
    eigenvalues : Float64[Array, "n_k_e n_bands_e"]
        Band energies in eV.
    eigenvectors : Complex128[Array, "n_k_v n_bands_v n_orb"]
        Complex orbital coefficients per k-point and band.
    kpoints : Float64[Array, "n_k_p 3"]
        Fractional reciprocal-space coordinates.
    fermi_energy : Float64[Array, ""]
        Fermi energy in eV.
    geometry : CrystalGeometry
        Differentiable lattice and atomic positions.
    basis : OrbitalBasis
        Orbital-to-atom and quantum-number metadata.
    orbital_positions : Optional[Float64[Array, "n_orb 3"]]
        Explicit fractional orbital centres, or ``None``.
    depths : Optional[Float64[Array, " n_depth"]]
        Orbital depths in Angstrom, or ``None`` for a bulk model.

    Raises
    ------
    ValueError
        If geometry or basis has the wrong type. If an eigensystem
        array has the wrong rank or shape. If k-point, band, or orbital
        axes disagree. If ``orbital_positions`` has the wrong shape.
        The delegated validators raise ``ValueError`` for their own
        static contracts.
    """
    if not isinstance(geometry, CrystalGeometry):
        message: str = "geometry must be a CrystalGeometry"
        raise ValueError(message)
    if not isinstance(basis, OrbitalBasis):
        message = "basis must be an OrbitalBasis"
        raise ValueError(message)
    if eigenvalues.ndim != _EIGENVALUE_NDIM:
        message = "eigenvalues must be two-dimensional"
        raise ValueError(message)
    if eigenvectors.ndim != _EIGENVECTOR_NDIM:
        message = "eigenvectors must be three-dimensional"
        raise ValueError(message)
    if (
        kpoints.ndim != _EIGENVALUE_NDIM
        or kpoints.shape[1] != _CELL_COMPONENTS
    ):
        message = "kpoints must have shape (n_k, 3)"
        raise ValueError(message)
    if fermi_energy.ndim != 0:
        message = "fermi_energy must be scalar"
        raise ValueError(message)
    if eigenvalues.shape != eigenvectors.shape[:2]:
        message = "eigenvalues and eigenvectors must agree on n_k and n_bands"
        raise ValueError(message)
    if eigenvalues.shape[0] != kpoints.shape[0]:
        message = "eigenvalues and kpoints must agree on n_k"
        raise ValueError(message)
    if eigenvectors.shape[2] != len(basis.n):
        message = "eigenvector orbital axis must match basis"
        raise ValueError(message)
    if orbital_positions is not None and (
        orbital_positions.ndim != _ORBITAL_POSITION_NDIM
        or orbital_positions.shape != (len(basis.n), _CELL_COMPONENTS)
    ):
        message = "orbital_positions must have shape (n_orbitals, 3)"
        raise ValueError(message)
    _validate_depths_shape(depths, len(basis.n))
    _validate_basis_geometry(basis, geometry)


@jaxtyped(typechecker=beartype)
def make_diagonalized_bands(  # noqa: DOC502, DOC503
    eigenvalues: Float64[Array, "n_k_e n_bands_e"],
    eigenvectors: Complex128[Array, "n_k_v n_bands_v n_orb"],
    kpoints: Float64[Array, "n_k_p 3"],
    geometry: CrystalGeometry,
    basis: OrbitalBasis,
    fermi_energy: ScalarNumeric = 0.0,
    orbital_positions: Optional[Float64[Array, "n_orb 3"]] = None,
    depths: Optional[Float64[Array, " n_depth"]] = None,
) -> DiagonalizedBands:
    """Create a validated ``DiagonalizedBands`` instance.

    The factory normalizes every numerical array and validates its axes
    against the supplied geometry and orbital basis.

    :see: :class:`~.test_tb_model.TestMakeDiagonalizedBands`

    Parameters
    ----------
    eigenvalues : Float64[Array, "n_k_e n_bands_e"]
        Band energies in eV.
    eigenvectors : Complex128[Array, "n_k_v n_bands_v n_orb"]
        Complex orbital coefficients in the basis-position gauge.
    kpoints : Float64[Array, "n_k_p 3"]
        Fractional reciprocal-space coordinates.
    geometry : CrystalGeometry
        Crystal geometry whose numerical leaves remain differentiable.
    basis : OrbitalBasis
        Orbital and atom metadata (**static** -- changing it triggers
        retracing).
    fermi_energy : ScalarNumeric, optional
        Fermi energy in eV. Default is 0.0.
    orbital_positions : Optional[Float64[Array, "n_orb 3"]], optional
        Explicit fractional orbital centres associated with the
        basis-position-gauge coefficients. ``None`` derives centres from
        atom assignments. Default is ``None``.
    depths : Optional[Float64[Array, "n_depth"]], optional
        Orbital depths in Angstrom below the top surface. ``None`` denotes a
        bulk model. Default is ``None``.

    Returns
    -------
    bands : DiagonalizedBands
        Validated double-precision eigensystem and its structural context.

    Raises
    ------
    ValueError
        If eigensystem axes, basis size, or atom assignments disagree.
    EquinoxRuntimeError
        If any numerical leaf is non-finite.

    Notes
    -----
    Static validation checks array axes and context before tracing. Runtime
    validation uses :func:`equinox.error_if` for every numerical leaf, so the
    same rejection behavior remains active under JIT.

    See Also
    --------
    DiagonalizedBands : Carrier constructed by this factory.
    make_tb_model : Construct the model diagonalized by native TB producers.
    """
    eigenvalue_array: Float64[Array, "n_k n_bands"] = jnp.asarray(
        eigenvalues,
        dtype=jnp.float64,
    )
    eigenvector_array: Complex128[Array, "n_k n_bands n_orb"] = jnp.asarray(
        eigenvectors,
        dtype=jnp.complex128,
    )
    kpoint_array: Float64[Array, "n_k 3"] = jnp.asarray(
        kpoints,
        dtype=jnp.float64,
    )
    fermi_array: Float64[Array, ""] = jnp.asarray(
        fermi_energy,
        dtype=jnp.float64,
    )
    orbital_position_array: Optional[Float64[Array, "n_orb 3"]] = None
    if orbital_positions is not None:
        orbital_position_array = jnp.asarray(
            orbital_positions,
            dtype=jnp.float64,
        )
    depth_array: Optional[Float64[Array, " n_depth"]] = None
    if depths is not None:
        depth_array = jnp.asarray(depths, dtype=jnp.float64)
    _validate_diagonalized_structure(
        eigenvalue_array,
        eigenvector_array,
        kpoint_array,
        fermi_array,
        geometry,
        basis,
        orbital_position_array,
        depth_array,
    )

    eigenvalue_array = eqx.error_if(
        eigenvalue_array,
        ~jnp.all(jnp.isfinite(eigenvalue_array)),
        "make_diagonalized_bands: eigenvalues finite",
    )
    eigenvector_array = eqx.error_if(
        eigenvector_array,
        ~jnp.all(jnp.isfinite(eigenvector_array)),
        "make_diagonalized_bands: eigenvectors finite",
    )
    kpoint_array = eqx.error_if(
        kpoint_array,
        ~jnp.all(jnp.isfinite(kpoint_array)),
        "make_diagonalized_bands: kpoints finite",
    )
    fermi_array = eqx.error_if(
        fermi_array,
        ~jnp.isfinite(fermi_array),
        "make_diagonalized_bands: fermi energy finite",
    )
    if orbital_position_array is not None:
        orbital_position_array = eqx.error_if(
            orbital_position_array,
            ~jnp.all(jnp.isfinite(orbital_position_array)),
            "make_diagonalized_bands: orbital positions finite",
        )
    if depth_array is not None:
        depth_array = eqx.error_if(
            depth_array,
            ~jnp.all(jnp.isfinite(depth_array)),
            "make_diagonalized_bands: depths must be finite",
        )
        depth_array = eqx.error_if(
            depth_array,
            jnp.any(depth_array < -_DEPTH_TOLERANCE_ANG),
            "make_diagonalized_bands: depths must be nonnegative",
        )
    checked_geometry: CrystalGeometry = _checked_geometry(
        geometry,
        "make_diagonalized_bands",
    )
    bands: DiagonalizedBands = DiagonalizedBands(
        eigenvalues=eigenvalue_array,
        eigenvectors=eigenvector_array,
        kpoints=kpoint_array,
        fermi_energy=fermi_array,
        geometry=checked_geometry,
        basis=basis,
        orbital_positions=orbital_position_array,
        depths=depth_array,
    )
    return bands


@jaxtyped(typechecker=beartype)
def make_tb_model(  # noqa: DOC502, DOC503, PLR0913, PLR0917
    hopping_amplitudes: Complex128[Array, "n_hop"],
    onsite_energies: Float64[Array, "n_orb"],
    soc_lambdas: Float64[Array, "n_shells"],
    geometry: CrystalGeometry,
    basis: OrbitalBasis,
    hopping_pairs: Tuple[Tuple[int, int], ...],
    hopping_cells: Tuple[Tuple[int, int, int], ...],
    shell_index: Tuple[int, ...],
    spinor: bool = False,
    orbital_positions: Optional[Float64[Array, "n_orb 3"]] = None,
    depths: Optional[Float64[Array, " n_depth"]] = None,
) -> TBModel:
    r"""Create a validated ``TBModel`` instance.

    The factory normalizes numerical leaves, validates exact connectivity,
    and enforces complex-conjugate hopping closure under JAX transformations.

    :see: :class:`~.test_tb_model.TestMakeTBModel`

    Parameters
    ----------
    hopping_amplitudes : Complex128[Array, "n_hop"]
        Directed hopping amplitudes in eV.
    onsite_energies : Float64[Array, "n_orb"]
        Onsite orbital energies in eV.
    soc_lambdas : Float64[Array, "n_shells"]
        Spin-orbit coupling energies in eV, one per atomic shell.
    geometry : CrystalGeometry
        Crystal lattice and fractional atomic positions.
    basis : OrbitalBasis
        Orbital-to-atom and quantum-number metadata (**static** -- changing it
        triggers retracing).
    hopping_pairs : Tuple[Tuple[int, int], ...]
        Directed ``(i, j)`` orbital pairs (**static** -- changing them
        triggers retracing).
    hopping_cells : Tuple[Tuple[int, int, int], ...]
        Exact integer translations ``R`` (**static** -- changing them triggers
        retracing).
    shell_index : Tuple[int, ...]
        Orbital-to-SOC-shell map with ``-1`` denoting no shell (**static** --
        changing it triggers retracing).
    spinor : bool, optional
        Whether the basis has explicit spin channels (**static** -- changing
        it triggers retracing). Default is ``False``.
    orbital_positions : Optional[Float64[Array, "n_orb 3"]], optional
        Explicit fractional orbital centres for the basis-position gauge.
        ``None`` derives centres from atom assignments. Default is ``None``.
    depths : Optional[Float64[Array, "n_depth"]], optional
        Orbital depths in Angstrom below the top surface. ``None`` denotes a
        bulk model. Default is ``None``.

    Returns
    -------
    model : TBModel
        Validated complex tight-binding model.

    Raises
    ------
    ValueError
        If structural lengths, indices, spin metadata, shell metadata, or
        Hermiticity closure metadata are inconsistent.
    EquinoxRuntimeError
        If a numerical leaf is non-finite or reverse hopping amplitudes are
        not complex conjugates to absolute tolerance ``1e-12`` eV.

    Notes
    -----
    The algorithm is:

    1. Check static dimensions, unique exact integer metadata, contiguous
       atomic-shell IDs, spin semantics, and closure under
       ``(i, j, R) -> (j, i, -R)``.
    2. Derive a static reverse-entry permutation from exact metadata.
    3. Use :func:`equinox.error_if` to reject non-finite leaves and enforce
       :math:`t_{ji}(-R) = t_{ij}(R)^*` under eager and compiled execution.

    The physical displacement is not stored. Hamiltonian consumers derive
    ``R + tau_j - tau_i`` from explicit orbital positions when present and
    otherwise from assigned atomic positions.

    See Also
    --------
    TBModel : Carrier constructed by this factory.
    make_diagonalized_bands : Construct the downstream eigensystem carrier.
    """
    hopping_array: Complex128[Array, " n_hop"] = jnp.asarray(
        hopping_amplitudes,
        dtype=jnp.complex128,
    )
    onsite_array: Float64[Array, " n_orb"] = jnp.asarray(
        onsite_energies,
        dtype=jnp.float64,
    )
    soc_array: Float64[Array, " n_shells"] = jnp.asarray(
        soc_lambdas,
        dtype=jnp.float64,
    )
    orbital_position_array: Optional[Float64[Array, "n_orb 3"]] = None
    if orbital_positions is not None:
        orbital_position_array = jnp.asarray(
            orbital_positions,
            dtype=jnp.float64,
        )
    depth_array: Optional[Float64[Array, " n_depth"]] = None
    if depths is not None:
        depth_array = jnp.asarray(depths, dtype=jnp.float64)
    closure: Tuple[int, ...] = _validate_tb_structure(
        hopping_array,
        onsite_array,
        soc_array,
        geometry,
        basis,
        hopping_pairs,
        hopping_cells,
        shell_index,
        spinor,
        orbital_position_array,
        depth_array,
    )

    hopping_array = eqx.error_if(
        hopping_array,
        ~jnp.all(jnp.isfinite(hopping_array)),
        "make_tb_model: hopping amplitudes finite",
    )
    onsite_array = eqx.error_if(
        onsite_array,
        ~jnp.all(jnp.isfinite(onsite_array)),
        "make_tb_model: onsite energies finite",
    )
    soc_array = eqx.error_if(
        soc_array,
        ~jnp.all(jnp.isfinite(soc_array)),
        "make_tb_model: soc lambdas finite",
    )
    if orbital_position_array is not None:
        orbital_position_array = eqx.error_if(
            orbital_position_array,
            ~jnp.all(jnp.isfinite(orbital_position_array)),
            "make_tb_model: orbital positions finite",
        )
    if depth_array is not None:
        depth_array = eqx.error_if(
            depth_array,
            ~jnp.all(jnp.isfinite(depth_array)),
            "make_tb_model: depths must be finite",
        )
        depth_array = eqx.error_if(
            depth_array,
            jnp.any(depth_array < -_DEPTH_TOLERANCE_ANG),
            "make_tb_model: depths must be nonnegative",
        )
    closure_indices: Int32[Array, " n_hop"] = jnp.asarray(
        closure,
        dtype=jnp.int32,
    )
    reverse_amplitudes: Complex128[Array, " n_hop"] = hopping_array[
        closure_indices
    ]
    closure_error: Float64[Array, " n_hop"] = jnp.abs(
        reverse_amplitudes - jnp.conj(hopping_array)
    )
    hopping_array = eqx.error_if(
        hopping_array,
        ~jnp.all(closure_error <= _HERMITICITY_TOLERANCE),
        "make_tb_model: reverse hopping amplitudes must be complex conjugates",
    )
    checked_geometry: CrystalGeometry = _checked_geometry(
        geometry,
        "make_tb_model",
    )
    model: TBModel = TBModel(
        hopping_amplitudes=hopping_array,
        onsite_energies=onsite_array,
        soc_lambdas=soc_array,
        geometry=checked_geometry,
        basis=basis,
        hopping_pairs=hopping_pairs,
        hopping_cells=hopping_cells,
        shell_index=shell_index,
        spinor=spinor,
        orbital_positions=orbital_position_array,
        depths=depth_array,
    )
    return model


@jaxtyped(typechecker=beartype)
def make_slab_topology(  # noqa: DOC502, PLR0913
    *,
    miller: Tuple[int, int, int],
    in_plane_coeffs: Tuple[
        Tuple[int, int, int],
        Tuple[int, int, int],
    ],
    stacking_coeffs: Tuple[int, int, int],
    atom_shifts: Tuple[Tuple[int, int, int], ...],
    bulk_atom_of_slab_atom: Tuple[int, ...],
    layer_of_slab_atom: Tuple[int, ...],
    termination: Tuple[str, str],
    thickness_ang: float,
    vacuum_ang: float,
    fine: Tuple[float, float],
    n_layers: int,
    bulk_atom_count: int,
    basis_atom_indices: Tuple[int, ...],
) -> SlabTopology:
    """Create validated host-selected slab topology metadata.

    The factory binds one exact surface basis to atom, layer, termination,
    and source-model provenance. All returned fields are static PyTree
    metadata because topology selection occurs before JAX transformation.

    :see: :class:`~.test_tb_model.TestMakeSlabTopology`

    Parameters
    ----------
    miller : Tuple[int, int, int]
        GCD-reduced surface Miller tuple.
    in_plane_coeffs : Tuple[Tuple[int, int, int], Tuple[int, int, int]]
        Exact in-plane lattice-coefficient rows.
    stacking_coeffs : Tuple[int, int, int]
        Exact one-plane stacking coefficient row.
    atom_shifts : Tuple[Tuple[int, int, int], ...]
        Surface-cell shift for every source bulk atom.
    bulk_atom_of_slab_atom : Tuple[int, ...]
        Bulk-atom provenance for every slab atom.
    layer_of_slab_atom : Tuple[int, ...]
        Layer index for every slab atom.
    termination : Tuple[str, str]
        Top and bottom termination labels.
    thickness_ang : float
        Requested slab thickness in Angstrom.
    vacuum_ang : float
        Vacuum padding in Angstrom.
    fine : Tuple[float, float]
        Top and bottom cut shifts in Angstrom.
    n_layers : int
        Positive number of selected layers.
    bulk_atom_count : int
        Positive atom count of the source bulk geometry.
    basis_atom_indices : Tuple[int, ...]
        Bulk-atom assignment for every source basis orbital.

    Returns
    -------
    topology : SlabTopology
        Validated static topology for pure-JAX slab rebuilding.

    Raises
    ------
    ValueError
        If exact surface coefficients, atom provenance, layer bounds, lengths,
        termination metadata, or finite slab dimensions are inconsistent.

    Notes
    -----
    Every check is static because all topology fields are immutable host
    choices. Numerical lattice and Hamiltonian leaves remain outside this
    carrier and retain their existing differentiable rebuild path.
    """
    _validate_slab_topology_structure(
        miller=miller,
        in_plane_coeffs=in_plane_coeffs,
        stacking_coeffs=stacking_coeffs,
        atom_shifts=atom_shifts,
        bulk_atom_of_slab_atom=bulk_atom_of_slab_atom,
        layer_of_slab_atom=layer_of_slab_atom,
        termination=termination,
        thickness_ang=thickness_ang,
        vacuum_ang=vacuum_ang,
        fine=fine,
        n_layers=n_layers,
        bulk_atom_count=bulk_atom_count,
        basis_atom_indices=basis_atom_indices,
    )
    topology: SlabTopology = SlabTopology(
        miller=miller,
        in_plane_coeffs=in_plane_coeffs,
        stacking_coeffs=stacking_coeffs,
        atom_shifts=atom_shifts,
        bulk_atom_of_slab_atom=bulk_atom_of_slab_atom,
        layer_of_slab_atom=layer_of_slab_atom,
        termination=termination,
        thickness_ang=thickness_ang,
        vacuum_ang=vacuum_ang,
        fine=fine,
        n_layers=n_layers,
        bulk_atom_count=bulk_atom_count,
        basis_atom_indices=basis_atom_indices,
    )
    return topology


@jaxtyped(typechecker=beartype)
def make_surface_cell(  # noqa: DOC502, DOC503
    in_plane_vectors: Float64[Array, "2 3"],
    stacking_vector: Float64[Array, " 3"],
    rotation: Float64[Array, "3 3"],
    interlayer_spacing_ang: ScalarNumeric,
    miller: Tuple[int, int, int],
    in_plane_coeffs: Tuple[
        Tuple[int, int, int],
        Tuple[int, int, int],
    ],
    stacking_coeffs: Tuple[int, int, int],
) -> SurfaceCell:
    """Create a validated Cartesian surface-cell carrier.

    Convert continuous inputs to JAX arrays, validate the exact integer frame,
    and enforce the finite, orthogonal surface-frame contract.

    :see: :class:`~.test_tb_model.TestMakeSurfaceCell`

    Parameters
    ----------
    in_plane_vectors : Float64[Array, "2 3"]
        Cartesian in-plane vectors in Angstrom, as rows.
    stacking_vector : Float64[Array, "3"]
        Cartesian stacking vector in Angstrom.
    rotation : Float64[Array, "3 3"]
        Active Cartesian rotation from bulk to surface frame.
    interlayer_spacing_ang : ScalarNumeric
        Positive interlayer spacing in Angstrom.
    miller : Tuple[int, int, int]
        GCD-reduced Miller tuple.
    in_plane_coeffs : Tuple[Tuple[int, int, int], Tuple[int, int, int]]
        Exact integer coefficients for the in-plane vectors.
    stacking_coeffs : Tuple[int, int, int]
        Exact integer coefficients for the stacking vector.

    Returns
    -------
    surface_cell : SurfaceCell
        Validated surface-cell carrier.

    Raises
    ------
    ValueError
        If array shapes or exact integer invariants are invalid.
    EquinoxRuntimeError
        If numerical leaves are non-finite, the spacing is not positive, or
        the rotation is not orthogonal within ``1e-10``.

    Notes
    -----
    Exact Miller coefficients remain static metadata while Cartesian vectors,
    rotation, and spacing remain differentiable leaves.
    """
    in_plane_array: Float64[Array, "2 3"] = jnp.asarray(
        in_plane_vectors,
        dtype=jnp.float64,
    )
    stacking_array: Float64[Array, " 3"] = jnp.asarray(
        stacking_vector,
        dtype=jnp.float64,
    )
    rotation_array: Float64[Array, "3 3"] = jnp.asarray(
        rotation,
        dtype=jnp.float64,
    )
    spacing_array: Float64[Array, ""] = jnp.asarray(
        interlayer_spacing_ang,
        dtype=jnp.float64,
    )
    _validate_surface_cell_structure(
        in_plane_array,
        stacking_array,
        rotation_array,
        spacing_array,
        miller,
        in_plane_coeffs,
        stacking_coeffs,
    )

    in_plane_array = eqx.error_if(
        in_plane_array,
        ~jnp.all(jnp.isfinite(in_plane_array)),
        "make_surface_cell: in-plane vectors must be finite",
    )
    stacking_array = eqx.error_if(
        stacking_array,
        ~jnp.all(jnp.isfinite(stacking_array)),
        "make_surface_cell: stacking vector must be finite",
    )
    rotation_array = eqx.error_if(
        rotation_array,
        ~jnp.all(jnp.isfinite(rotation_array)),
        "make_surface_cell: rotation must be finite",
    )
    orthogonality_error: Float64[Array, ""] = jnp.linalg.norm(
        rotation_array.T @ rotation_array
        - jnp.eye(_CELL_COMPONENTS, dtype=jnp.float64)
    )
    rotation_array = eqx.error_if(
        rotation_array,
        orthogonality_error >= _ROTATION_ORTHOGONALITY_TOLERANCE,
        "make_surface_cell: rotation must be orthogonal",
    )
    spacing_array = eqx.error_if(
        spacing_array,
        ~jnp.isfinite(spacing_array) | (spacing_array <= 0.0),
        "make_surface_cell: interlayer spacing must be finite and positive",
    )
    surface_cell: SurfaceCell = SurfaceCell(
        in_plane_vectors=in_plane_array,
        stacking_vector=stacking_array,
        rotation=rotation_array,
        interlayer_spacing_ang=spacing_array,
        miller=miller,
        in_plane_coeffs=in_plane_coeffs,
        stacking_coeffs=stacking_coeffs,
    )
    return surface_cell


@jaxtyped(typechecker=beartype)
def make_slab_spec(
    surface_cell: SurfaceCell,
    geometry: CrystalGeometry,
    thickness_ang: float,
    vacuum_ang: float,
    fine: Tuple[float, float],
    termination: Tuple[str, str],
    n_layers: int,
    bulk_atom_of_slab_atom: Tuple[int, ...],
    layer_of_slab_atom: Tuple[int, ...],
) -> SlabSpec:
    """Create a validated slab-construction sidecar.

    ``geometry`` supplies the bulk species and atom count used to validate
    termination labels and provenance. It is validation context and is not
    stored in the returned sidecar. A geometry with no species metadata may
    use only the internal natural-cut sentinel ``("X", "X")``; explicit
    species termination still requires declared species.

    :see: :class:`~.test_tb_model.TestMakeSlabSpec`

    Parameters
    ----------
    surface_cell : SurfaceCell
        Validated surface-cell frame.
    geometry : CrystalGeometry
        Bulk geometry used to validate species and atom provenance.
    thickness_ang : float
        Nonnegative requested minimum slab span in Angstrom.
    vacuum_ang : float
        Nonnegative vacuum padding in Angstrom.
    fine : Tuple[float, float]
        Finite top and bottom cut shifts in Angstrom.
    termination : Tuple[str, str]
        Top and bottom species labels.
    n_layers : int
        Positive number of slab layers.
    bulk_atom_of_slab_atom : Tuple[int, ...]
        Bulk-atom index for each slab atom.
    layer_of_slab_atom : Tuple[int, ...]
        Layer index for each slab atom.

    Returns
    -------
    slab_spec : SlabSpec
        Validated static slab provenance with a traced surface-cell child.

    Raises
    ------
    ValueError
        If slab choices, species, or provenance mappings are inconsistent.

    Notes
    -----
    The factory validates host-selected provenance and preserves the
    ``SurfaceCell`` as the only traced child of the static slab sidecar.
    """
    if not isinstance(geometry, CrystalGeometry):
        message: str = "geometry must be a CrystalGeometry"
        raise ValueError(message)
    _validate_slab_spec_structure(
        surface_cell,
        thickness_ang,
        vacuum_ang,
        fine,
        termination,
        n_layers,
        bulk_atom_of_slab_atom,
        layer_of_slab_atom,
    )
    unknown_species_termination: bool = (
        not geometry.species and termination == ("X", "X")
    )
    if not unknown_species_termination and any(
        species not in geometry.species for species in termination
    ):
        message = "termination species must occur in geometry.species"
        raise ValueError(message)
    n_bulk_atoms: int = geometry.positions.shape[0]
    if any(index >= n_bulk_atoms for index in bulk_atom_of_slab_atom):
        message = "bulk_atom_of_slab_atom entries must refer to geometry atoms"
        raise ValueError(message)
    slab_spec: SlabSpec = SlabSpec(
        surface_cell=surface_cell,
        thickness_ang=thickness_ang,
        vacuum_ang=vacuum_ang,
        fine=fine,
        termination=termination,
        n_layers=n_layers,
        bulk_atom_of_slab_atom=bulk_atom_of_slab_atom,
        layer_of_slab_atom=layer_of_slab_atom,
    )
    return slab_spec


__all__: list[str] = [
    "DiagonalizedBands",
    "SlabSpec",
    "SlabTopology",
    "SurfaceCell",
    "TBModel",
    "make_diagonalized_bands",
    "make_slab_spec",
    "make_slab_topology",
    "make_surface_cell",
    "make_tb_model",
]
