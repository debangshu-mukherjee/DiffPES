"""Parse normative Wannier90 text records.

Extended Summary
----------------
This private module owns the strict line-oriented grammar for Wannier90
header, lattice, Hamiltonian, position, and degeneracy records.
"""

from pathlib import Path

import numpy as np
from beartype.typing import Dict, List, Optional, Tuple
from jaxtyping import Complex128, Float64, Int64
from numpy.typing import NDArray

from diffpes.constants import (
    WANNIER_CELL_FIELDS,
    WANNIER_DEGENERACIES_PER_LINE,
    WANNIER_HR_HAMILTONIAN_FIELDS,
    WANNIER_TB_HAMILTONIAN_FIELDS,
    WANNIER_TB_POSITION_FIELDS,
)
from diffpes.types import (
    HamiltonianBlocks,
    OrbitalBasis,
    TextLineCursor,
    make_hamiltonian_blocks,
)

from .tb_files import _line_error, _parse_finite_float, _parse_integer


def _parse_single_positive_integer(  # noqa: DOC503 -- prebuilt line error.
    cursor: TextLineCursor,
    context: str,
) -> int:
    """PRIVATE: Parse a one-token positive integer line.

    Parameters
    ----------
    cursor : TextLineCursor
        Line cursor positioned before the expected counter line.
    context : str
        Field name, such as ``"num_wann"`` or ``"nrpts"``, used in
        diagnostics.

    Returns
    -------
    value : int
        Positive integer from the next nonblank line.

    Raises
    ------
    ValueError
        If the next nonblank line does not hold exactly one token, the
        token is not an integer, or the value is not positive.

    Notes
    -----
    Advances the cursor past the consumed line.  Wannier90 writes the
    ``num_wann`` and ``nrpts`` counters in this one-token form.
    """
    line_number: int
    text: str
    line_number, text = cursor.next_nonempty(context)
    tokens: List[str] = text.split()
    if len(tokens) != 1:
        message: ValueError = _line_error(
            cursor.path,
            line_number,
            f"{context} line must contain exactly one integer",
        )
        raise message
    value: int = _parse_integer(
        tokens[0],
        cursor.path,
        line_number,
        context,
    )
    if value <= 0:
        message = _line_error(
            cursor.path,
            line_number,
            f"{context} must be positive",
        )
        raise message
    return value


def _parse_cell(  # noqa: DOC503 -- raises a prebuilt line error.
    tokens: List[str],
    path: Path,
    line_number: int,
    context: str,
) -> Tuple[int, int, int]:
    """PRIVATE: Parse one exact three-integer cell.

    Parameters
    ----------
    tokens : List[str]
        Split tokens of one cell record.
    path : Path
        Source file for the diagnostic.
    line_number : int
        One-based physical line number for the diagnostic.
    context : str
        Record name the diagnostic quotes.

    Returns
    -------
    values : Tuple[int, int, int]
        Integer lattice translation ``(R1, R2, R3)`` in lattice-vector
        units.

    Raises
    ------
    ValueError
        If the token count differs from three or any token is not an
        exact integer.

    Notes
    -----
    Delegates each component to :func:`_parse_integer` with an ``R1``,
    ``R2``, or ``R3`` label.
    """
    if len(tokens) != WANNIER_CELL_FIELDS:
        message: ValueError = _line_error(
            path,
            line_number,
            f"{context} must contain exactly three integers",
        )
        raise message
    values: Tuple[int, int, int] = (
        _parse_integer(tokens[0], path, line_number, f"{context} R1"),
        _parse_integer(tokens[1], path, line_number, f"{context} R2"),
        _parse_integer(tokens[2], path, line_number, f"{context} R3"),
    )
    return values


def _parse_one_based_pair(  # noqa: DOC503 -- raises a prebuilt line error.
    tokens: List[str],
    n_orbitals: int,
    path: Path,
    line_number: int,
) -> Tuple[int, int]:
    """PRIVATE: Parse and validate a one-based Wannier matrix index pair.

    Parameters
    ----------
    tokens : List[str]
        Two index tokens from one matrix row.
    n_orbitals : int
        Declared ``num_wann`` orbital count.
    path : Path
        Source file for the diagnostic.
    line_number : int
        One-based physical line number for the diagnostic.

    Returns
    -------
    pair : Tuple[int, int]
        Zero-based ``(row, column)`` matrix indices.

    Raises
    ------
    ValueError
        If either token is not an integer or lies outside
        ``[1, n_orbitals]``.

    Notes
    -----
    Wannier90 writes one-based indices; the return value subtracts one
    from each after the range check.
    """
    first: int = _parse_integer(
        tokens[0],
        path,
        line_number,
        "first matrix index",
    )
    second: int = _parse_integer(
        tokens[1],
        path,
        line_number,
        "second matrix index",
    )
    if not (1 <= first <= n_orbitals and 1 <= second <= n_orbitals):
        message: ValueError = _line_error(
            path,
            line_number,
            f"matrix indices must be in [1, {n_orbitals}]",
        )
        raise message
    pair: Tuple[int, int] = (first - 1, second - 1)
    return pair


def _parse_degeneracies(  # noqa: DOC503 -- raises a prebuilt line error.
    cursor: TextLineCursor,
    n_cells: int,
) -> Tuple[int, ...]:
    """PRIVATE: Parse the normative 15-integer-per-line degeneracy block.

    Parameters
    ----------
    cursor : TextLineCursor
        Line cursor positioned at the start of the degeneracy block.
    n_cells : int
        Declared ``nrpts`` cell count.

    Returns
    -------
    degeneracies : Tuple[int, ...]
        One positive Wigner--Seitz degeneracy weight per cell, in file
        order.

    Raises
    ------
    ValueError
        If a block line does not hold exactly the expected token
        count, a token is not an integer, or a weight is not positive.

    Notes
    -----
    Wannier90 writes fifteen weights per line; the final line holds
    the remainder.  The parser demands exactly
    ``min(15, n_cells - parsed)`` tokens per line, so a malformed
    block cannot silently shift the Hamiltonian rows that follow.
    """
    values: List[int] = []
    while len(values) < n_cells:
        line_number: int
        text: str
        line_number, text = cursor.next_nonempty("degeneracy weights")
        tokens: List[str] = text.split()
        expected: int = min(
            WANNIER_DEGENERACIES_PER_LINE,
            n_cells - len(values),
        )
        if len(tokens) != expected:
            message: ValueError = _line_error(
                cursor.path,
                line_number,
                f"degeneracy line must contain exactly {expected} integer(s)",
            )
            raise message
        token: str
        for token in tokens:
            weight: int = _parse_integer(
                token,
                cursor.path,
                line_number,
                "degeneracy",
            )
            if weight <= 0:
                message = _line_error(
                    cursor.path,
                    line_number,
                    "degeneracies must be positive",
                )
                raise message
            values.append(weight)
    degeneracies: Tuple[int, ...] = tuple(values)
    return degeneracies


def _parse_wannier_dimensions(
    cursor: TextLineCursor,
) -> Tuple[int, int, Tuple[int, ...]]:
    """PRIVATE: Parse ``num_wann``, ``nrpts``, and their weight block.

    Parameters
    ----------
    cursor : TextLineCursor
        Line cursor positioned after the header line.

    Returns
    -------
    dimensions : Tuple[int, int, Tuple[int, ...]]
        Orbital count ``num_wann``, cell count ``nrpts``, and the
        per-cell degeneracy weights.

    Notes
    -----
    Reads the two one-token counter lines and then the degeneracy
    block; both Wannier90 grammars share this layout.  The counter and
    degeneracy helpers raise line-numbered ``ValueError`` diagnostics
    for malformed lines.
    """
    n_orbitals: int = _parse_single_positive_integer(cursor, "num_wann")
    n_cells: int = _parse_single_positive_integer(cursor, "nrpts")
    degeneracies: Tuple[int, ...] = _parse_degeneracies(cursor, n_cells)
    dimensions: Tuple[int, int, Tuple[int, ...]] = (
        n_orbitals,
        n_cells,
        degeneracies,
    )
    return dimensions


def _parse_header(cursor: TextLineCursor) -> None:  # noqa: DOC503
    """PRIVATE: Consume and validate the required free-form header line.

    Parameters
    ----------
    cursor : TextLineCursor
        Line cursor positioned at the first physical line.

    Raises
    ------
    ValueError
        If the first physical line is blank.

    Notes
    -----
    Wannier90 always writes a date comment as line one.  The check
    reads the physical line without blank skipping.  A file that
    starts with data fails instead of losing its first record.
    """
    line_number: int
    text: str
    line_number, text = cursor.next_line("header")
    if not text.strip():
        message: ValueError = _line_error(
            cursor.path,
            line_number,
            "header must be nonempty",
        )
        raise message


def _require_filename_suffix(path: Path, suffix: str) -> None:
    """PRIVATE: Prevent accidental cross-format or generic ``.dat`` dispatch.

    Parameters
    ----------
    path : Path
        File path the caller wants to parse.
    suffix : str
        Required filename ending, ``"_hr.dat"`` or ``"_tb.dat"``.

    Raises
    ------
    ValueError
        If the filename does not end with ``suffix``.

    Notes
    -----
    The two Wannier90 grammars are intentionally not auto-detected;
    the filename must state the format explicitly.
    """
    if not path.name.endswith(suffix):
        message: str = (
            f"{path}: expected an explicit {suffix} filename; "
            "generic .dat dispatch is not supported"
        )
        raise ValueError(message)


def _parse_hr_hamiltonian_blocks(  # noqa: DOC503, PLR0913
    cursor: TextLineCursor,
    n_orbitals: int,
    n_cells: int,
    degeneracies: Tuple[int, ...],
) -> HamiltonianBlocks:
    """PRIVATE: Parse cell-bearing ``hr.dat`` Hamiltonian rows.

    Implementation Logic
    --------------------
    Reads ``n_cells`` blocks of ``n_orbitals**2`` rows.  Each row
    holds the seven fields ``R1 R2 R3 i j Re Im`` and repeats the
    cell on every line.  The cell must stay constant inside one
    block.  No ``(i, j)`` pair may repeat inside a block, and no cell
    may repeat across blocks.  Every stored entry divides by the
    block degeneracy weight once, and a parallel array records the
    one-based source line of each entry for later diagnostics.

    Parameters
    ----------
    cursor : TextLineCursor
        Line cursor positioned at the first Hamiltonian row.
    n_orbitals : int
        Declared ``num_wann`` orbital count.
    n_cells : int
        Declared ``nrpts`` cell count.
    degeneracies : Tuple[int, ...]
        Per-cell Wigner--Seitz degeneracy weights in file order.

    Returns
    -------
    blocks : HamiltonianBlocks
        Weight-normalized complex matrices in eV, their source-line
        map, the cell tuple in file order, and the degeneracies.

    Raises
    ------
    ValueError
        If a row has the wrong field count, a value fails to parse,
        or the cell changes inside a block.  Also if an index pair
        repeats, a block is empty, or a cell repeats.
    """
    matrices: Complex128[NDArray, "n_cell n_orb n_orb"] = np.empty(
        (n_cells, n_orbitals, n_orbitals),
        dtype=np.complex128,
    )
    source_lines: Int64[NDArray, "n_cell n_orb n_orb"] = np.empty(
        (n_cells, n_orbitals, n_orbitals),
        dtype=np.int64,
    )
    cells: List[Tuple[int, int, int]] = []
    cell_index: int
    for cell_index in range(n_cells):
        block_cell: Optional[Tuple[int, int, int]] = None
        seen_pairs: set[Tuple[int, int]] = set()
        weight: int = degeneracies[cell_index]
        for _ in range(n_orbitals * n_orbitals):
            line_number: int
            text: str
            line_number, text = cursor.next_nonempty("hr Hamiltonian row")
            tokens: List[str] = text.split()
            if len(tokens) != WANNIER_HR_HAMILTONIAN_FIELDS:
                message: ValueError = _line_error(
                    cursor.path,
                    line_number,
                    "hr Hamiltonian row must contain seven fields",
                )
                raise message
            cell: Tuple[int, int, int] = _parse_cell(
                tokens[:3],
                cursor.path,
                line_number,
                "Hamiltonian cell",
            )
            if block_cell is None:
                block_cell = cell
            elif cell != block_cell:
                message = _line_error(
                    cursor.path,
                    line_number,
                    "cell changed inside one hr Hamiltonian block",
                )
                raise message
            pair: Tuple[int, int] = _parse_one_based_pair(
                tokens[3:5],
                n_orbitals,
                cursor.path,
                line_number,
            )
            if pair in seen_pairs:
                message = _line_error(
                    cursor.path,
                    line_number,
                    f"duplicate matrix indices {pair[0] + 1}, {pair[1] + 1}",
                )
                raise message
            seen_pairs.add(pair)
            real: float = _parse_finite_float(
                tokens[5],
                cursor.path,
                line_number,
                "Hamiltonian real part",
            )
            imaginary: float = _parse_finite_float(
                tokens[6],
                cursor.path,
                line_number,
                "Hamiltonian imaginary part",
            )
            matrices[cell_index, pair[0], pair[1]] = (
                real + 1j * imaginary
            ) / weight
            source_lines[cell_index, pair[0], pair[1]] = line_number
        if block_cell is None:
            message = f"{cursor.path}: empty hr Hamiltonian block"
            raise ValueError(message)
        if block_cell in cells:
            message = f"{cursor.path}: duplicate Hamiltonian cell {block_cell}"
            raise ValueError(message)
        cells.append(block_cell)
    blocks: HamiltonianBlocks = make_hamiltonian_blocks(
        matrices=matrices,
        source_lines=source_lines,
        cells=tuple(cells),
        degeneracies=degeneracies,
    )
    return blocks


def _parse_tb_lattice(  # noqa: DOC503
    cursor: TextLineCursor,
) -> Float64[NDArray, "3 3"]:
    """PRIVATE: Parse three Cartesian lattice rows in Angstrom.

    Parameters
    ----------
    cursor : TextLineCursor
        Line cursor positioned at the first lattice row.

    Returns
    -------
    lattice : Float64[NDArray, "3 3"]
        Row-vector lattice matrix in Angstrom.

    Raises
    ------
    ValueError
        If a lattice row does not hold exactly three finite floats.

    Notes
    -----
    ``tb.dat`` stores the lattice directly after the header; each of
    the three nonblank lines carries one Cartesian lattice vector.
    """
    lattice: Float64[NDArray, "3 3"] = np.empty((3, 3), dtype=np.float64)
    row: int
    for row in range(3):
        line_number: int
        text: str
        line_number, text = cursor.next_nonempty("tb lattice")
        tokens: List[str] = text.split()
        if len(tokens) != WANNIER_CELL_FIELDS:
            message: ValueError = _line_error(
                cursor.path,
                line_number,
                "tb lattice row must contain three floats",
            )
            raise message
        lattice[row] = [
            _parse_finite_float(
                token,
                cursor.path,
                line_number,
                "lattice component",
            )
            for token in tokens
        ]
    return lattice


def _parse_tb_hamiltonian_blocks(  # noqa: DOC503 -- prebuilt line error.
    cursor: TextLineCursor,
    n_orbitals: int,
    n_cells: int,
    degeneracies: Tuple[int, ...],
) -> HamiltonianBlocks:
    """PRIVATE: Parse block-headed ``tb.dat`` Hamiltonian matrices.

    Implementation Logic
    --------------------
    Reads ``n_cells`` blocks.  Each block starts with one
    three-integer cell header and continues with ``n_orbitals**2``
    rows of four fields ``i j Re Im``.  No cell may repeat and no
    ``(i, j)`` pair may repeat inside a block.  Every stored entry
    divides by the block degeneracy weight once, and a parallel array
    records the one-based source line of each entry.

    Parameters
    ----------
    cursor : TextLineCursor
        Line cursor positioned at the first cell header.
    n_orbitals : int
        Declared ``num_wann`` orbital count.
    n_cells : int
        Declared ``nrpts`` cell count.
    degeneracies : Tuple[int, ...]
        Per-cell Wigner--Seitz degeneracy weights in file order.

    Returns
    -------
    blocks : HamiltonianBlocks
        Weight-normalized complex matrices in eV, their source-line
        map, the cell tuple in file order, and the degeneracies.

    Raises
    ------
    ValueError
        If a header or row has an invalid shape, a value fails to
        parse, an index pair repeats inside a block, or a cell
        repeats.
    """
    matrices: Complex128[NDArray, "n_cell n_orb n_orb"] = np.empty(
        (n_cells, n_orbitals, n_orbitals),
        dtype=np.complex128,
    )
    source_lines: Int64[NDArray, "n_cell n_orb n_orb"] = np.empty(
        (n_cells, n_orbitals, n_orbitals),
        dtype=np.int64,
    )
    cells: List[Tuple[int, int, int]] = []
    cell_index: int
    for cell_index in range(n_cells):
        cell_line: int
        cell_text: str
        cell_line, cell_text = cursor.next_nonempty("tb Hamiltonian cell")
        cell: Tuple[int, int, int] = _parse_cell(
            cell_text.split(),
            cursor.path,
            cell_line,
            "Hamiltonian cell",
        )
        if cell in cells:
            message: ValueError = _line_error(
                cursor.path,
                cell_line,
                f"duplicate Hamiltonian cell {cell}",
            )
            raise message
        cells.append(cell)
        seen_pairs: set[Tuple[int, int]] = set()
        weight: int = degeneracies[cell_index]
        for _ in range(n_orbitals * n_orbitals):
            line_number: int
            text: str
            line_number, text = cursor.next_nonempty("tb Hamiltonian row")
            tokens: List[str] = text.split()
            if len(tokens) != WANNIER_TB_HAMILTONIAN_FIELDS:
                message = _line_error(
                    cursor.path,
                    line_number,
                    "tb Hamiltonian row must contain four fields",
                )
                raise message
            pair: Tuple[int, int] = _parse_one_based_pair(
                tokens[:2],
                n_orbitals,
                cursor.path,
                line_number,
            )
            if pair in seen_pairs:
                message = _line_error(
                    cursor.path,
                    line_number,
                    f"duplicate matrix indices {pair[0] + 1}, {pair[1] + 1}",
                )
                raise message
            seen_pairs.add(pair)
            real: float = _parse_finite_float(
                tokens[2],
                cursor.path,
                line_number,
                "Hamiltonian real part",
            )
            imaginary: float = _parse_finite_float(
                tokens[3],
                cursor.path,
                line_number,
                "Hamiltonian imaginary part",
            )
            matrices[cell_index, pair[0], pair[1]] = (
                real + 1j * imaginary
            ) / weight
            source_lines[cell_index, pair[0], pair[1]] = line_number
    blocks: HamiltonianBlocks = make_hamiltonian_blocks(
        matrices=matrices,
        source_lines=source_lines,
        cells=tuple(cells),
        degeneracies=degeneracies,
    )
    return blocks


def _parse_tb_position_blocks(  # noqa: DOC503 -- prebuilt line error.
    cursor: TextLineCursor,
    n_orbitals: int,
    hamiltonian_blocks: HamiltonianBlocks,
) -> Complex128[NDArray, "n_cell n_orb n_orb 3"]:
    """PRIVATE: Parse and align block-headed ``tb.dat`` position matrices.

    Implementation Logic
    --------------------
    Reads one position block per Hamiltonian cell.  Each block starts
    with one three-integer cell header.  The header must name an
    existing Hamiltonian cell and must not repeat.  The block then
    holds ``n_orbitals**2`` rows of eight fields: the index pair,
    then real and imaginary parts of the x, y, and z components.
    Every component divides by the degeneracy weight of its cell
    once.  The stacked result follows the Hamiltonian cell order, not
    the position-block file order.

    Parameters
    ----------
    cursor : TextLineCursor
        Line cursor positioned at the first position cell header.
    n_orbitals : int
        Declared ``num_wann`` orbital count.
    hamiltonian_blocks : HamiltonianBlocks
        Parsed Hamiltonian blocks that define the cell set, the cell
        order, and the degeneracy weights.

    Returns
    -------
    aligned : Complex128[NDArray, "n_cell n_orb n_orb 3"]
        Weight-normalized position-operator matrix elements in
        Angstrom, aligned to the Hamiltonian cell order.

    Raises
    ------
    ValueError
        If a header names an unknown or repeated cell, a row has an
        invalid shape, a value fails to parse, or an index pair
        repeats.
    """
    by_cell: Dict[
        Tuple[int, int, int], Complex128[NDArray, "n_orb n_orb 3"]
    ] = {}
    weight_by_cell: Dict[Tuple[int, int, int], int] = dict(
        zip(
            hamiltonian_blocks.cells,
            hamiltonian_blocks.degeneracies,
            strict=True,
        )
    )
    for _ in hamiltonian_blocks.cells:
        cell_line: int
        cell_text: str
        cell_line, cell_text = cursor.next_nonempty("tb position cell")
        cell: Tuple[int, int, int] = _parse_cell(
            cell_text.split(),
            cursor.path,
            cell_line,
            "position cell",
        )
        if cell not in weight_by_cell:
            message: ValueError = _line_error(
                cursor.path,
                cell_line,
                f"position cell {cell} has no Hamiltonian block",
            )
            raise message
        if cell in by_cell:
            message = _line_error(
                cursor.path,
                cell_line,
                f"duplicate position cell {cell}",
            )
            raise message
        matrix: Complex128[NDArray, "n_orb n_orb 3"] = np.empty(
            (n_orbitals, n_orbitals, 3),
            dtype=np.complex128,
        )
        seen_pairs: set[Tuple[int, int]] = set()
        weight: int = weight_by_cell[cell]
        for _ in range(n_orbitals * n_orbitals):
            line_number: int
            text: str
            line_number, text = cursor.next_nonempty("tb position row")
            tokens: List[str] = text.split()
            if len(tokens) != WANNIER_TB_POSITION_FIELDS:
                message = _line_error(
                    cursor.path,
                    line_number,
                    "tb position row must contain eight fields",
                )
                raise message
            pair: Tuple[int, int] = _parse_one_based_pair(
                tokens[:2],
                n_orbitals,
                cursor.path,
                line_number,
            )
            if pair in seen_pairs:
                message = _line_error(
                    cursor.path,
                    line_number,
                    f"duplicate matrix indices {pair[0] + 1}, {pair[1] + 1}",
                )
                raise message
            seen_pairs.add(pair)
            components: List[complex] = []
            axis: int
            for axis in range(3):
                real: float = _parse_finite_float(
                    tokens[2 + 2 * axis],
                    cursor.path,
                    line_number,
                    f"position axis {axis} real part",
                )
                imaginary: float = _parse_finite_float(
                    tokens[3 + 2 * axis],
                    cursor.path,
                    line_number,
                    f"position axis {axis} imaginary part",
                )
                components.append((real + 1j * imaginary) / weight)
            matrix[pair[0], pair[1]] = components
        by_cell[cell] = matrix
    aligned: Complex128[NDArray, "n_cell n_orb n_orb 3"] = np.stack(
        tuple(by_cell[cell] for cell in hamiltonian_blocks.cells),
        axis=0,
    )
    return aligned


def _validate_basis_size(
    basis: OrbitalBasis,
    n_orbitals: int,
    path: Path,
) -> None:
    """PRIVATE: Require one basis entry per serialized Wannier function.

    Parameters
    ----------
    basis : OrbitalBasis
        Caller-supplied orbital metadata.
    n_orbitals : int
        ``num_wann`` count declared by the file.
    path : Path
        Source file for the diagnostic.

    Raises
    ------
    ValueError
        If ``len(basis.n)`` differs from ``n_orbitals``.

    Notes
    -----
    Compares only the lengths; the carrier constructors validate
    per-orbital quantum numbers.
    """
    if len(basis.n) != n_orbitals:
        message: str = (
            f"{path}: basis has {len(basis.n)} orbitals but file declares "
            f"{n_orbitals}"
        )
        raise ValueError(message)


__all__: list[str] = []
