r"""Read normative Wannier90 tight-binding files.

Extended Summary
----------------
This module converts strictly parsed seedname_hr.dat and seedname_tb.dat
records into validated tight-binding and Wannier operator carriers.

Routine Listings
----------------
:func:`read_wannier90_hr`
    Parse a normative Wannier90 ``seedname_hr.dat`` file.
:func:`read_wannier90_tb`
    Parse a normative Wannier90 ``seedname_tb.dat`` file.

Notes
-----
The two formats have intentionally separate public readers. No generic
dat-file dispatcher exists.
"""

from pathlib import Path

import jax.numpy as jnp
import numpy as np
from beartype import beartype
from beartype.typing import List, Optional, Tuple
from jaxtyping import Array, Complex128, Float64, Int64, jaxtyped
from numpy.typing import NDArray

from diffpes.constants import (
    WANNIER_CENTRE_CONSISTENCY_TOLERANCE,
    WANNIER_HERMITICITY_TOLERANCE,
    WANNIER_HR_SUFFIX,
    WANNIER_TB_SUFFIX,
)
from diffpes.types import (
    CrystalGeometry,
    HamiltonianBlocks,
    HoppingRecord,
    OrbitalBasis,
    TBModel,
    TextLineCursor,
    WannierOperatorData,
    make_crystal_geometry,
    make_hamiltonian_blocks,
    make_hopping_record,
    make_tb_model,
    make_text_line_cursor,
    make_wannier_operator_data,
)

from .tb_files import (
    _line_error,
    _spin_permutation,
    _validate_hopping_closure,
)
from .wannier90_parser import (
    _parse_header,
    _parse_hr_hamiltonian_blocks,
    _parse_tb_hamiltonian_blocks,
    _parse_tb_lattice,
    _parse_tb_position_blocks,
    _parse_wannier_dimensions,
    _require_filename_suffix,
    _validate_basis_size,
)


def _extract_model_data(  # noqa: DOC503 -- raises a prebuilt line error.
    blocks: HamiltonianBlocks,
    path: Path,
) -> Tuple[Float64[NDArray, " n_orb"], Tuple[HoppingRecord, ...]]:
    """PRIVATE: Extract real origin diagonals and directed hopping records.

    Implementation Logic
    --------------------
    Requires exactly one origin cell ``(0, 0, 0)``.  Origin-diagonal
    entries must be real within the ``1e-12`` eV tolerance and become
    the onsite energies.  Every other entry, including origin
    off-diagonals, becomes one directed hopping record that keeps its
    source line.  The record set then passes
    :func:`_validate_hopping_closure` before it is returned.

    Parameters
    ----------
    blocks : HamiltonianBlocks
        Weight-normalized Hamiltonian blocks with source-line maps.
    path : Path
        Source file for the diagnostics.

    Returns
    -------
    result : Tuple[Float64[NDArray, " n_orb"], \
Tuple[HoppingRecord, ...]]
        Real onsite energies in eV and the validated directed hopping
        records in eV.

    Raises
    ------
    ValueError
        If the origin cell is absent or repeated, an onsite entry has
        an imaginary part above tolerance, or Hermitian closure fails.
    """
    origin_indices: List[int] = [
        index for index, cell in enumerate(blocks.cells) if cell == (0, 0, 0)
    ]
    if len(origin_indices) != 1:
        message: str = (
            f"{path}: Wannier file must contain exactly one origin cell"
        )
        raise ValueError(message)
    origin: int = origin_indices[0]
    n_orbitals: int = blocks.matrices.shape[1]
    onsite: Float64[NDArray, " n_orb"] = np.empty(
        (n_orbitals,), dtype=np.float64
    )
    records: List[HoppingRecord] = []
    cell_index: int
    cell: Tuple[int, int, int]
    orbital_i: int
    orbital_j: int
    for cell_index, cell in enumerate(blocks.cells):
        for orbital_i in range(n_orbitals):
            for orbital_j in range(n_orbitals):
                amplitude: complex = complex(
                    blocks.matrices[cell_index, orbital_i, orbital_j]
                )
                line_number: int = int(
                    blocks.source_lines[cell_index, orbital_i, orbital_j]
                )
                if cell_index == origin and orbital_i == orbital_j:
                    if abs(amplitude.imag) > WANNIER_HERMITICITY_TOLERANCE:
                        message: ValueError = _line_error(
                            path,
                            line_number,
                            "onsite Hamiltonian entry must be real",
                        )
                        raise message
                    onsite[orbital_i] = amplitude.real
                else:
                    records.append(
                        make_hopping_record(
                            pair=(orbital_i, orbital_j),
                            cell=cell,
                            amplitude=amplitude,
                            line_number=line_number,
                        )
                    )
    record_tuple: Tuple[HoppingRecord, ...] = tuple(records)
    _validate_hopping_closure(record_tuple, path)
    result: Tuple[Float64[NDArray, " n_orb"], Tuple[HoppingRecord, ...]] = (
        onsite,
        record_tuple,
    )
    return result


def _make_model(
    blocks: HamiltonianBlocks,
    geometry: CrystalGeometry,
    basis: OrbitalBasis,
    path: Path,
    orbital_positions: Optional[Float64[Array, "n_orb 3"]] = None,
) -> TBModel:
    """PRIVATE: Convert normalized matrix blocks to a validated native model.

    Parameters
    ----------
    blocks : HamiltonianBlocks
        Weight-normalized Hamiltonian blocks with source-line maps.
    geometry : CrystalGeometry
        Resolved lattice and atomic positions.
    basis : OrbitalBasis
        Orbital metadata matching the file orbital count.
    path : Path
        Source file for diagnostics.
    orbital_positions : Optional[Float64[Array, "n_orb 3"]], optional
        Explicit Cartesian Wannier centres in Angstrom, or ``None``
        for the basis-position gauge.  Default is ``None``.

    Returns
    -------
    model : TBModel
        Validated native carrier with hopping amplitudes and onsite
        energies in eV.

    Notes
    -----
    Splits the blocks with :func:`_extract_model_data` and forwards
    everything to :func:`make_tb_model`.  The call passes no SOC
    shells (``shell_index`` all ``-1``, empty ``soc_lambdas``) and
    takes the spinor flag from ``basis.spin``.  The factory performs
    the final carrier validation.
    """
    onsite: Float64[NDArray, " n_orb"]
    records: Tuple[HoppingRecord, ...]
    onsite, records = _extract_model_data(blocks, path)
    n_orbitals: int = onsite.shape[0]
    model: TBModel = make_tb_model(
        hopping_amplitudes=jnp.asarray(
            [record.amplitude for record in records],
            dtype=jnp.complex128,
        ),
        onsite_energies=jnp.asarray(onsite, dtype=jnp.float64),
        soc_lambdas=jnp.zeros((0,), dtype=jnp.float64),
        geometry=geometry,
        basis=basis,
        hopping_pairs=tuple(record.pair for record in records),
        hopping_cells=tuple(record.cell for record in records),
        shell_index=(-1,) * n_orbitals,
        spinor=bool(basis.spin),
        orbital_positions=orbital_positions,
    )
    return model


def _permute_hamiltonian_blocks(
    blocks: HamiltonianBlocks,
    permutation: Tuple[int, ...],
) -> HamiltonianBlocks:
    """PRIVATE: Apply one state permutation to both Hamiltonian axes and
    line maps.

    Parameters
    ----------
    blocks : HamiltonianBlocks
        Parsed blocks in the serialized orbital order.
    permutation : Tuple[int, ...]
        Serialized index for each native index, from
        :func:`_spin_permutation`.

    Returns
    -------
    permuted_blocks : HamiltonianBlocks
        Blocks with both orbital axes of the matrices and of the
        source-line map gathered into the native order; cells and
        degeneracies pass through unchanged.

    Notes
    -----
    Uses :func:`np.take` on axes one and two, so rows and columns
    reorder consistently and every entry keeps its recorded source
    line.
    """
    matrices: Complex128[NDArray, "n_cell n_orb n_orb"] = np.take(
        blocks.matrices, permutation, axis=1
    )
    matrices = np.take(matrices, permutation, axis=2)
    source_lines: Int64[NDArray, "n_cell n_orb n_orb"] = np.take(
        blocks.source_lines,
        permutation,
        axis=1,
    )
    source_lines = np.take(source_lines, permutation, axis=2)
    permuted_blocks: HamiltonianBlocks = make_hamiltonian_blocks(
        matrices=matrices,
        source_lines=source_lines,
        cells=blocks.cells,
        degeneracies=blocks.degeneracies,
    )
    return permuted_blocks


def _permute_position_matrices(
    matrices: Complex128[NDArray, "n_cell n_orb n_orb 3"],
    permutation: Tuple[int, ...],
) -> Complex128[NDArray, "n_cell n_orb n_orb 3"]:
    """PRIVATE: Apply one state permutation to both position-operator axes.

    Parameters
    ----------
    matrices : Complex128[NDArray, "n_cell n_orb n_orb 3"]
        Position-operator matrix elements in Angstrom in the
        serialized orbital order.
    permutation : Tuple[int, ...]
        Serialized index for each native index, from
        :func:`_spin_permutation`.

    Returns
    -------
    result : Complex128[NDArray, "n_cell n_orb n_orb 3"]
        Matrix elements gathered into the native order on both orbital
        axes; the Cartesian axis stays untouched.

    Notes
    -----
    Uses :func:`np.take` on axes one and two exactly as the
    Hamiltonian permutation does.
    """
    permuted: Complex128[NDArray, "n_cell n_orb n_orb 3"] = np.take(
        matrices, permutation, axis=1
    )
    result: Complex128[NDArray, "n_cell n_orb n_orb 3"] = np.take(
        permuted, permutation, axis=2
    )
    return result


def _centres_from_position_matrices(
    matrices: Complex128[NDArray, "n_cell n_orb n_orb 3"],
    cells: Tuple[Tuple[int, int, int], ...],
    path: Path,
) -> Float64[NDArray, "n_orb 3"]:
    """PRIVATE: Extract real origin-diagonal Wannier centres in Angstrom.

    Parameters
    ----------
    matrices : Complex128[NDArray, "n_cell n_orb n_orb 3"]
        Weight-normalized position-operator matrix elements in
        Angstrom, aligned to ``cells``.
    cells : Tuple[Tuple[int, int, int], ...]
        Cell tuple in the same order as the matrix leading axis.
    path : Path
        Source file for diagnostics.

    Returns
    -------
    centres : Float64[NDArray, "n_orb 3"]
        Real Cartesian Wannier centres in Angstrom, one row per
        orbital.

    Raises
    ------
    ValueError
        If no origin cell exists or an origin-diagonal entry has an
        imaginary part above the ``1e-12`` tolerance.

    Notes
    -----
    The centre of Wannier function ``n`` is the origin-cell diagonal
    ``<n0|r|n0>``.  The helper takes the diagonal of the origin block
    and keeps the real part after the imaginary-part check.
    """
    try:
        origin: int = cells.index((0, 0, 0))
    except ValueError as error:
        error: ValueError
        message: str = f"{path}: position matrices require an origin cell"
        raise ValueError(message) from error
    diagonal: Complex128[NDArray, "n_orb 3"] = np.diagonal(
        matrices[origin],
        axis1=0,
        axis2=1,
    ).T
    if np.any(np.abs(diagonal.imag) > WANNIER_HERMITICITY_TOLERANCE):
        message = (
            f"{path}: origin-diagonal position entries must be real "
            f"within {WANNIER_HERMITICITY_TOLERANCE:.0e} Angstrom"
        )
        raise ValueError(message)
    centres: Float64[NDArray, "n_orb 3"] = np.asarray(
        diagonal.real, dtype=np.float64
    )
    return centres


def _geometry_from_centres(
    lattice: Float64[NDArray, "3 3"],
    centres_cart: Float64[NDArray, "n_orb 3"],
    basis: OrbitalBasis,
    path: Path,
) -> CrystalGeometry:
    """PRIVATE: Build atom positions after validating orbital-centre
    assumptions.

    Implementation Logic
    --------------------
    Requires nonempty, zero-based, contiguous ``basis.atom_indices``.
    Wannier centres assigned to one atom must agree within the
    ``1e-10`` Angstrom tolerance.  The first orbital centre of each
    atom becomes the atom position.  Cartesian atom positions
    convert to fractional coordinates through the inverse lattice, and
    the geometry factory receives an empty species tuple.

    Parameters
    ----------
    lattice : Float64[NDArray, "3 3"]
        Row-vector lattice matrix in Angstrom.
    centres_cart : Float64[NDArray, "n_orb 3"]
        Cartesian Wannier centres in Angstrom.
    basis : OrbitalBasis
        Orbital metadata that assigns each orbital to an atom.
    path : Path
        Source file for diagnostics.

    Returns
    -------
    geometry : CrystalGeometry
        Validated geometry with fractional atomic positions.

    Raises
    ------
    ValueError
        If the basis is empty, atom indices are not contiguous from
        zero, centres of one atom disagree, or the lattice is
        singular.
    """
    atom_indices: Tuple[int, ...] = basis.atom_indices
    if not atom_indices:
        message: str = f"{path}: basis must contain at least one orbital"
        raise ValueError(message)
    n_atoms: int = max(atom_indices) + 1
    if set(atom_indices) != set(range(n_atoms)):
        message = f"{path}: basis atom_indices must be contiguous from zero"
        raise ValueError(message)
    atom_centres: Float64[NDArray, "n_atom 3"] = np.empty(
        (n_atoms, 3), dtype=np.float64
    )
    atom: int
    for atom in range(n_atoms):
        orbital_rows: List[int] = [
            index
            for index, assigned_atom in enumerate(atom_indices)
            if assigned_atom == atom
        ]
        reference: Float64[NDArray, " 3"] = centres_cart[orbital_rows[0]]
        differences: Float64[NDArray, "n_row 3"] = np.abs(
            centres_cart[orbital_rows] - reference
        )
        if np.any(differences > WANNIER_CENTRE_CONSISTENCY_TOLERANCE):
            message = (
                f"{path}: Wannier centres assigned to atom {atom} differ by "
                "more than "
                f"{WANNIER_CENTRE_CONSISTENCY_TOLERANCE:.0e} Angstrom"
            )
            raise ValueError(message)
        atom_centres[atom] = reference
    try:
        fractional: Float64[NDArray, "n_atom 3"] = (
            atom_centres @ np.linalg.inv(lattice)
        )
    except np.linalg.LinAlgError as error:
        error: np.linalg.LinAlgError
        message = f"{path}: lattice must be nonsingular"
        raise ValueError(message) from error
    geometry: CrystalGeometry = make_crystal_geometry(
        lattice=jnp.asarray(lattice, dtype=jnp.float64),
        positions=jnp.asarray(fractional, dtype=jnp.float64),
        species=(),
    )
    return geometry


def _validated_explicit_centres(
    centres_cart: Float64[Array, "n_orb 3"],
    n_orbitals: int,
    path: Path,
) -> Float64[NDArray, "n_orb 3"]:
    """PRIVATE: Validate explicit ``hr.dat`` centres without altering
    connectivity.

    Parameters
    ----------
    centres_cart : Float64[Array, "n_orb 3"]
        Caller-supplied Cartesian Wannier centres in Angstrom.
    n_orbitals : int
        ``num_wann`` count declared by the file.
    path : Path
        Source file for diagnostics.

    Returns
    -------
    centres : Float64[NDArray, "n_orb 3"]
        The same centres as a ``float64`` NumPy array.

    Raises
    ------
    ValueError
        If the shape differs from ``(n_orbitals, 3)`` or any value is
        not finite.

    Notes
    -----
    Only converts and checks; hopping cells and pairs stay exactly as
    parsed from the file.
    """
    centres: Float64[NDArray, "n_orb 3"] = np.asarray(
        centres_cart, dtype=np.float64
    )
    if centres.shape != (n_orbitals, 3):
        message: str = (
            f"{path}: centres_cart must have shape ({n_orbitals}, 3)"
        )
        raise ValueError(message)
    if not np.all(np.isfinite(centres)):
        message = f"{path}: centres_cart must be finite"
        raise ValueError(message)
    return centres


def _fractional_wannier_centres(
    lattice: Float64[NDArray, "3 3"],
    centres_cart: Float64[NDArray, "n_orb 3"],
    path: Path,
) -> Float64[NDArray, "n_orb 3"]:
    """PRIVATE: Convert Cartesian Wannier centres to fractional coordinates.

    Parameters
    ----------
    lattice : Float64[NDArray, "3 3"]
        Row-vector lattice matrix in Angstrom.
    centres_cart : Float64[NDArray, "n_orb 3"]
        Cartesian Wannier centres in Angstrom.
    path : Path
        Source file for the diagnostic.

    Returns
    -------
    fractional_array : Float64[NDArray, "n_orb 3"]
        Dimensionless fractional centres ``centres_cart @ inv(L)``.

    Raises
    ------
    ValueError
        If the lattice matrix is singular.

    Notes
    -----
    Right-multiplies by the inverse lattice because both the lattice
    rows and the centre rows are Cartesian row vectors.
    """
    try:
        inverse_lattice: Float64[NDArray, "3 3"] = np.linalg.inv(lattice)
    except np.linalg.LinAlgError as error:
        error: np.linalg.LinAlgError
        message: str = f"{path}: lattice must be nonsingular"
        raise ValueError(message) from error
    fractional: Float64[NDArray, "n_orb 3"] = centres_cart @ inverse_lattice
    fractional_array: Float64[NDArray, "n_orb 3"] = np.asarray(
        fractional, dtype=np.float64
    )
    return fractional_array


def _resolve_tb_geometry(
    lattice: Float64[NDArray, "3 3"],
    centres_cart: Float64[NDArray, "n_orb 3"],
    basis: OrbitalBasis,
    path: Path,
    geometry: Optional[CrystalGeometry],
) -> CrystalGeometry:
    """PRIVATE: Resolve atomic geometry without conflating atoms and Wannier
    centres.

    Parameters
    ----------
    lattice : Float64[NDArray, "3 3"]
        Lattice matrix in Angstrom parsed from ``tb.dat``.
    centres_cart : Float64[NDArray, "n_orb 3"]
        Cartesian Wannier centres in Angstrom.
    basis : OrbitalBasis
        Orbital metadata that assigns each orbital to an atom.
    path : Path
        Source file for diagnostics.
    geometry : Optional[CrystalGeometry]
        Caller-supplied geometry, or ``None`` to derive one.

    Returns
    -------
    resolved_geometry : CrystalGeometry
        The supplied geometry after validation, or a geometry derived
        from the Wannier centres.

    Raises
    ------
    ValueError
        If the supplied geometry does not cover all basis atom indices
        or its lattice differs from the ``tb.dat`` lattice beyond the
        ``1e-10`` Angstrom tolerance.

    Notes
    -----
    With ``geometry is None`` the atom positions come from
    :func:`_geometry_from_centres`; otherwise the supplied atoms stay
    authoritative and only consistency checks run.
    """
    resolved_geometry: CrystalGeometry
    if geometry is None:
        resolved_geometry = _geometry_from_centres(
            lattice,
            centres_cart,
            basis,
            path,
        )
    else:
        if geometry.positions.shape[0] <= max(basis.atom_indices, default=-1):
            message: str = (
                f"{path}: geometry positions do not cover basis atom_indices"
            )
            raise ValueError(message)
        if not np.allclose(
            np.asarray(geometry.lattice),
            lattice,
            rtol=0.0,
            atol=WANNIER_CENTRE_CONSISTENCY_TOLERANCE,
        ):
            message = f"{path}: supplied geometry lattice differs from tb.dat"
            raise ValueError(message)
        resolved_geometry = geometry
    return resolved_geometry


@jaxtyped(typechecker=beartype)
def read_wannier90_hr(  # noqa: DOC502
    filename: str,
    geometry: CrystalGeometry,
    basis: OrbitalBasis,
    centres_cart: Float64[Array, "n_orb 3"],
) -> Tuple[TBModel, WannierOperatorData]:
    """Parse a normative Wannier90 ``seedname_hr.dat`` file.

    Read exact real-space Hamiltonian cells and combine them with required
    external Wannier centres.

    :see: :class:`~.test_wannier90.TestReadWannier90Hr`

    Parameters
    ----------
    filename : str
        Explicit path ending in ``"_hr.dat"``.
    geometry : CrystalGeometry
        Geometry used by basis-position-gauge Bloch assembly.
    basis : OrbitalBasis
        Native orbital ordering.
    centres_cart : Float64[Array, "n_orb 3"]
        Required explicit Wannier centres in Cartesian Angstrom.

    Returns
    -------
    result : Tuple[TBModel, WannierOperatorData]
        Validated Hamiltonian and an ``hr`` sidecar whose position matrices
        are ``None``.

    Raises
    ------
    ValueError
        If validation rejects the header, weight block, indexed matrix rows,
        centres, origin onsite entries, or Hermitian closure.
    EquinoxRuntimeError
        If a final carrier rejects a traced numerical invariant.

    Notes
    -----
    After the free-form header, the format stores ``num_wann``, ``nrpts``,
    positive degeneracies, and exactly ``num_wann**2 * nrpts`` rows
    ``R1 R2 R3 i j Re Im`` with one-based indices. Divide every Hamiltonian
    entry once by its cell degeneracy. Retain parsed integer cells verbatim;
    centres never replace or modify connectivity.
    """
    path: Path = Path(filename)
    _require_filename_suffix(path, WANNIER_HR_SUFFIX)
    cursor: TextLineCursor = make_text_line_cursor(path)
    _parse_header(cursor)
    n_orbitals: int
    n_cells: int
    degeneracies: Tuple[int, ...]
    n_orbitals, n_cells, degeneracies = _parse_wannier_dimensions(cursor)
    _validate_basis_size(basis, n_orbitals, path)
    _spin_permutation(basis, "block_down_up", path)
    blocks: HamiltonianBlocks = _parse_hr_hamiltonian_blocks(
        cursor,
        n_orbitals,
        n_cells,
        degeneracies,
    )
    cursor.ensure_exhausted()
    centres: Float64[NDArray, "n_orb 3"] = _validated_explicit_centres(
        centres_cart,
        n_orbitals,
        path,
    )
    orbital_positions: Float64[NDArray, "n_orb 3"] = (
        _fractional_wannier_centres(
            np.asarray(geometry.lattice),
            centres,
            path,
        )
    )
    model: TBModel = _make_model(
        blocks,
        geometry,
        basis,
        path,
        jnp.asarray(orbital_positions, dtype=jnp.float64),
    )
    operator_data: WannierOperatorData = make_wannier_operator_data(
        position_matrices=None,
        centres_cart=jnp.asarray(centres, dtype=jnp.float64),
        cells=blocks.cells,
        degeneracies=blocks.degeneracies,
        spin_layout="block_down_up",
        source_format="hr",
    )
    result: Tuple[TBModel, WannierOperatorData] = (model, operator_data)
    return result


@jaxtyped(typechecker=beartype)
def read_wannier90_tb(  # noqa: DOC502
    filename: str,
    basis: OrbitalBasis,
    spin_layout: str,
    geometry: Optional[CrystalGeometry] = None,
) -> Tuple[TBModel, WannierOperatorData]:
    """Parse a normative Wannier90 ``seedname_tb.dat`` file.

    Read lattice, Hamiltonian, and position-operator blocks while normalizing
    all serialized spin axes into the native convention.

    :see: :class:`~.test_wannier90.TestReadWannier90Tb`

    Parameters
    ----------
    filename : str
        Explicit path ending in ``"_tb.dat"``.
    basis : OrbitalBasis
        Native DiffPES basis in block-down/up order for spinors.
    spin_layout : str
        Serialized ordering, ``"block_down_up"`` or
        ``"interleaved_up_down"``.
    geometry : Optional[CrystalGeometry], optional
        Atomic geometry. Noncoincident Wannier centres on one atom require
        this value. Without it, coincident assigned centres determine each
        atomic position.

    Returns
    -------
    result : Tuple[TBModel, WannierOperatorData]
        Validated model plus full position-operator sidecar.

    Raises
    ------
    ValueError
        If validation rejects lattice data, dimensions, cells, weights,
        matrix indices, spin metadata, centres, or Hermitian closure.
    EquinoxRuntimeError
        If a final carrier rejects a traced numerical invariant.

    Notes
    -----
    ``tb.dat`` is not an ``hr.dat`` row variant. It stores a header and three
    Cartesian lattice rows in Angstrom. A shared dimension/degeneracy block
    precedes cell-headed Hamiltonian and position matrices. Hamiltonian and
    position rows carry one-based matrix indices. This parser validates each
    pair independently of the writer's loop order.

    Both operator axes, Hamiltonian axes, and centres receive the same
    setup-time serialized-to-native spin permutation. No later consumer
    repeats that permutation.
    """
    path: Path = Path(filename)
    _require_filename_suffix(path, WANNIER_TB_SUFFIX)
    cursor: TextLineCursor = make_text_line_cursor(path)
    _parse_header(cursor)
    lattice: Float64[NDArray, "3 3"] = _parse_tb_lattice(cursor)
    n_orbitals: int
    n_cells: int
    degeneracies: Tuple[int, ...]
    n_orbitals, n_cells, degeneracies = _parse_wannier_dimensions(cursor)
    _validate_basis_size(basis, n_orbitals, path)
    serialized_blocks: HamiltonianBlocks = _parse_tb_hamiltonian_blocks(
        cursor,
        n_orbitals,
        n_cells,
        degeneracies,
    )
    serialized_positions: Complex128[NDArray, "n_cell n_orb n_orb 3"] = (
        _parse_tb_position_blocks(
            cursor,
            n_orbitals,
            serialized_blocks,
        )
    )
    cursor.ensure_exhausted()
    permutation: Tuple[int, ...] = _spin_permutation(
        basis,
        spin_layout,
        path,
    )
    blocks: HamiltonianBlocks = _permute_hamiltonian_blocks(
        serialized_blocks,
        permutation,
    )
    position_matrices: Complex128[NDArray, "n_cell n_orb n_orb 3"] = (
        _permute_position_matrices(
            serialized_positions,
            permutation,
        )
    )
    centres: Float64[NDArray, "n_orb 3"] = _centres_from_position_matrices(
        position_matrices,
        blocks.cells,
        path,
    )
    resolved_geometry: CrystalGeometry = _resolve_tb_geometry(
        lattice,
        centres,
        basis,
        path,
        geometry,
    )
    orbital_positions: Float64[NDArray, "n_orb 3"] = (
        _fractional_wannier_centres(
            lattice,
            centres,
            path,
        )
    )
    model: TBModel = _make_model(
        blocks,
        resolved_geometry,
        basis,
        path,
        jnp.asarray(orbital_positions, dtype=jnp.float64),
    )
    operator_data: WannierOperatorData = make_wannier_operator_data(
        position_matrices=jnp.asarray(
            position_matrices,
            dtype=jnp.complex128,
        ),
        centres_cart=jnp.asarray(centres, dtype=jnp.float64),
        cells=blocks.cells,
        degeneracies=blocks.degeneracies,
        spin_layout=spin_layout,
        source_format="tb",
    )
    result: Tuple[TBModel, WannierOperatorData] = (model, operator_data)
    return result


__all__: list[str] = [
    "read_wannier90_hr",
    "read_wannier90_tb",
]
