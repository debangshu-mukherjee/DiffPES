"""Parse explicit hopping lists and normative Wannier90 TB files.

Extended Summary
----------------
This module is the host-side airlock for three distinct tight-binding file
grammars: a Cartesian hopping list, Wannier90 ``seedname_hr.dat``, and
Wannier90 ``seedname_tb.dat``. Parsing uses NumPy and exact Python integer
metadata; validated outputs are native JAX/Equinox carriers.

Routine Listings
----------------
:func:`read_hopping_list`
    Parse a zero-based Cartesian tight-binding hopping list.
:func:`read_wannier90_hr`
    Parse a normative Wannier90 ``seedname_hr.dat`` file.
:func:`read_wannier90_tb`
    Parse a normative Wannier90 ``seedname_tb.dat`` file.

Notes
-----
The two Wannier90 formats have intentionally separate public readers.
No generic ``.dat`` dispatcher exists. ``hr.dat`` rows carry their cell on
every Hamiltonian line, whereas ``tb.dat`` uses explicit cell-block headers
for both Hamiltonian and position matrices.
"""

from dataclasses import dataclass
from pathlib import Path

import jax.numpy as jnp
import numpy as np
from beartype import beartype
from beartype.typing import Optional
from jaxtyping import Array, Complex128, Float64, Int64, jaxtyped
from numpy.typing import NDArray

from diffpes.types import (
    HOPPING_LIST_COMPLEX_FIELDS,
    HOPPING_LIST_REAL_FIELDS,
    WANNIER_CELL_FIELDS,
    WANNIER_CENTRE_CONSISTENCY_TOLERANCE,
    WANNIER_DEGENERACIES_PER_LINE,
    WANNIER_HERMITICITY_TOLERANCE,
    WANNIER_HR_HAMILTONIAN_FIELDS,
    WANNIER_HR_SUFFIX,
    WANNIER_INTEGER_RECOVERY_TOLERANCE,
    WANNIER_TB_HAMILTONIAN_FIELDS,
    WANNIER_TB_POSITION_FIELDS,
    WANNIER_TB_SUFFIX,
    CrystalGeometry,
    OrbitalBasis,
    TBModel,
    WannierOperatorData,
    make_crystal_geometry,
    make_tb_model,
    make_wannier_operator_data,
)


@dataclass(frozen=True)
class _HoppingRecord:
    """Store one parsed non-onsite hopping and its source line."""

    pair: tuple[int, int]
    cell: tuple[int, int, int]
    amplitude: complex
    line_number: int


@dataclass(frozen=True)
class _HamiltonianBlocks:
    """Store normalized Hamiltonian matrices with exact block metadata."""

    matrices: Complex128[NDArray, "n_cell n_orb n_orb"]
    source_lines: Int64[NDArray, "n_cell n_orb n_orb"]
    cells: tuple[tuple[int, int, int], ...]
    degeneracies: tuple[int, ...]


@dataclass
class _LineCursor:
    """Record strict, line-numbered parsing of one text file."""

    path: Path
    lines: tuple[str, ...]
    index: int = 0

    @classmethod
    def from_path(cls, path: Path) -> "_LineCursor":
        """Read one UTF-8 text file into a line cursor."""
        text: str = path.read_text(encoding="utf-8")
        cursor: _LineCursor = cls(
            path=path,
            lines=tuple(text.splitlines()),
        )
        return cursor

    def next_line(self, context: str) -> tuple[int, str]:
        """Return the next physical line without skipping blanks."""
        if self.index >= len(self.lines):
            message: str = (
                f"{self.path}: unexpected end of file while reading {context}"
            )
            raise ValueError(message)
        line_number: int = self.index + 1
        text: str = self.lines[self.index]
        self.index += 1
        result: tuple[int, str] = (line_number, text)
        return result

    def next_nonempty(self, context: str) -> tuple[int, str]:
        """Return the next nonblank line."""
        while self.index < len(self.lines):
            line_number: int
            text: str
            line_number, text = self.next_line(context)
            if text.strip():
                result: tuple[int, str] = (line_number, text)
                return result
        message: str = (
            f"{self.path}: unexpected end of file while reading {context}"
        )
        raise ValueError(message)

    def ensure_exhausted(self) -> None:
        """Reject any trailing nonblank record."""
        while self.index < len(self.lines):
            line_number: int
            text: str
            line_number, text = self.next_line("trailing records")
            if text.strip():
                message: str = (
                    f"{self.path}: line {line_number}: unexpected trailing "
                    "record"
                )
                raise ValueError(message)


def _line_error(path: Path, line_number: int, message: str) -> ValueError:
    """Build a line-numbered parser error."""
    error: ValueError = ValueError(f"{path}: line {line_number}: {message}")
    return error


def _parse_integer(
    token: str,
    path: Path,
    line_number: int,
    label: str,
) -> int:
    """Parse one exact integer token."""
    try:
        value: int = int(token)
    except ValueError as error:
        error: ValueError
        message: ValueError = _line_error(
            path,
            line_number,
            f"{label} must be an integer, got {token!r}",
        )
        raise message from error
    return value


def _parse_finite_float(
    token: str,
    path: Path,
    line_number: int,
    label: str,
) -> float:
    """Parse one finite floating-point token."""
    try:
        value: float = float(token)
    except ValueError as error:
        error: ValueError
        message: ValueError = _line_error(
            path,
            line_number,
            f"{label} must be a float, got {token!r}",
        )
        raise message from error
    if not np.isfinite(value):
        message = _line_error(
            path,
            line_number,
            f"{label} must be finite",
        )
        raise message
    return value


def _parse_single_positive_integer(
    cursor: _LineCursor,
    context: str,
) -> int:
    """Parse a one-token positive integer line."""
    line_number: int
    text: str
    line_number, text = cursor.next_nonempty(context)
    tokens: list[str] = text.split()
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


def _parse_cell(
    tokens: list[str],
    path: Path,
    line_number: int,
    context: str,
) -> tuple[int, int, int]:
    """Parse one exact three-integer cell."""
    if len(tokens) != WANNIER_CELL_FIELDS:
        message: ValueError = _line_error(
            path,
            line_number,
            f"{context} must contain exactly three integers",
        )
        raise message
    values: tuple[int, int, int] = (
        _parse_integer(tokens[0], path, line_number, f"{context} R1"),
        _parse_integer(tokens[1], path, line_number, f"{context} R2"),
        _parse_integer(tokens[2], path, line_number, f"{context} R3"),
    )
    return values


def _parse_one_based_pair(
    tokens: list[str],
    n_orbitals: int,
    path: Path,
    line_number: int,
) -> tuple[int, int]:
    """Parse and validate a one-based Wannier matrix index pair."""
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
    pair: tuple[int, int] = (first - 1, second - 1)
    return pair


def _parse_degeneracies(
    cursor: _LineCursor,
    n_cells: int,
) -> tuple[int, ...]:
    """Parse the normative 15-integer-per-line degeneracy block."""
    values: list[int] = []
    while len(values) < n_cells:
        line_number: int
        text: str
        line_number, text = cursor.next_nonempty("degeneracy weights")
        tokens: list[str] = text.split()
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
    degeneracies: tuple[int, ...] = tuple(values)
    return degeneracies


def _parse_wannier_dimensions(
    cursor: _LineCursor,
) -> tuple[int, int, tuple[int, ...]]:
    """Parse ``num_wann``, ``nrpts``, and their weight block."""
    n_orbitals: int = _parse_single_positive_integer(cursor, "num_wann")
    n_cells: int = _parse_single_positive_integer(cursor, "nrpts")
    degeneracies: tuple[int, ...] = _parse_degeneracies(cursor, n_cells)
    dimensions: tuple[int, int, tuple[int, ...]] = (
        n_orbitals,
        n_cells,
        degeneracies,
    )
    return dimensions


def _parse_header(cursor: _LineCursor) -> None:
    """Consume and validate the required free-form header line."""
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
    """Prevent accidental cross-format or generic ``.dat`` dispatch."""
    if not path.name.endswith(suffix):
        message: str = (
            f"{path}: expected an explicit {suffix} filename; "
            "generic .dat dispatch is not supported"
        )
        raise ValueError(message)


def _parse_hr_hamiltonian_blocks(  # noqa: PLR0913
    cursor: _LineCursor,
    n_orbitals: int,
    n_cells: int,
    degeneracies: tuple[int, ...],
) -> _HamiltonianBlocks:
    """Parse cell-bearing ``hr.dat`` Hamiltonian rows."""
    matrices: Complex128[NDArray, "n_cell n_orb n_orb"] = np.empty(
        (n_cells, n_orbitals, n_orbitals),
        dtype=np.complex128,
    )
    source_lines: Int64[NDArray, "n_cell n_orb n_orb"] = np.empty(
        (n_cells, n_orbitals, n_orbitals),
        dtype=np.int64,
    )
    cells: list[tuple[int, int, int]] = []
    cell_index: int
    for cell_index in range(n_cells):
        block_cell: Optional[tuple[int, int, int]] = None
        seen_pairs: set[tuple[int, int]] = set()
        weight: int = degeneracies[cell_index]
        for _ in range(n_orbitals * n_orbitals):
            line_number: int
            text: str
            line_number, text = cursor.next_nonempty("hr Hamiltonian row")
            tokens: list[str] = text.split()
            if len(tokens) != WANNIER_HR_HAMILTONIAN_FIELDS:
                message: ValueError = _line_error(
                    cursor.path,
                    line_number,
                    "hr Hamiltonian row must contain seven fields",
                )
                raise message
            cell: tuple[int, int, int] = _parse_cell(
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
            pair: tuple[int, int] = _parse_one_based_pair(
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
    blocks: _HamiltonianBlocks = _HamiltonianBlocks(
        matrices=matrices,
        source_lines=source_lines,
        cells=tuple(cells),
        degeneracies=degeneracies,
    )
    return blocks


def _parse_tb_lattice(cursor: _LineCursor) -> Float64[NDArray, "3 3"]:
    """Parse three Cartesian lattice rows in Angstrom."""
    lattice: Float64[NDArray, "3 3"] = np.empty((3, 3), dtype=np.float64)
    row: int
    for row in range(3):
        line_number: int
        text: str
        line_number, text = cursor.next_nonempty("tb lattice")
        tokens: list[str] = text.split()
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


def _parse_tb_hamiltonian_blocks(
    cursor: _LineCursor,
    n_orbitals: int,
    n_cells: int,
    degeneracies: tuple[int, ...],
) -> _HamiltonianBlocks:
    """Parse block-headed ``tb.dat`` Hamiltonian matrices."""
    matrices: Complex128[NDArray, "n_cell n_orb n_orb"] = np.empty(
        (n_cells, n_orbitals, n_orbitals),
        dtype=np.complex128,
    )
    source_lines: Int64[NDArray, "n_cell n_orb n_orb"] = np.empty(
        (n_cells, n_orbitals, n_orbitals),
        dtype=np.int64,
    )
    cells: list[tuple[int, int, int]] = []
    cell_index: int
    for cell_index in range(n_cells):
        cell_line: int
        cell_text: str
        cell_line, cell_text = cursor.next_nonempty("tb Hamiltonian cell")
        cell: tuple[int, int, int] = _parse_cell(
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
        seen_pairs: set[tuple[int, int]] = set()
        weight: int = degeneracies[cell_index]
        for _ in range(n_orbitals * n_orbitals):
            line_number: int
            text: str
            line_number, text = cursor.next_nonempty("tb Hamiltonian row")
            tokens: list[str] = text.split()
            if len(tokens) != WANNIER_TB_HAMILTONIAN_FIELDS:
                message = _line_error(
                    cursor.path,
                    line_number,
                    "tb Hamiltonian row must contain four fields",
                )
                raise message
            pair: tuple[int, int] = _parse_one_based_pair(
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
    blocks: _HamiltonianBlocks = _HamiltonianBlocks(
        matrices=matrices,
        source_lines=source_lines,
        cells=tuple(cells),
        degeneracies=degeneracies,
    )
    return blocks


def _parse_tb_position_blocks(
    cursor: _LineCursor,
    n_orbitals: int,
    hamiltonian_blocks: _HamiltonianBlocks,
) -> Complex128[NDArray, "n_cell n_orb n_orb 3"]:
    """Parse and align block-headed ``tb.dat`` position matrices."""
    by_cell: dict[
        tuple[int, int, int], Complex128[NDArray, "n_orb n_orb 3"]
    ] = {}
    weight_by_cell: dict[tuple[int, int, int], int] = dict(
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
        cell: tuple[int, int, int] = _parse_cell(
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
        seen_pairs: set[tuple[int, int]] = set()
        weight: int = weight_by_cell[cell]
        for _ in range(n_orbitals * n_orbitals):
            line_number: int
            text: str
            line_number, text = cursor.next_nonempty("tb position row")
            tokens: list[str] = text.split()
            if len(tokens) != WANNIER_TB_POSITION_FIELDS:
                message = _line_error(
                    cursor.path,
                    line_number,
                    "tb position row must contain eight fields",
                )
                raise message
            pair: tuple[int, int] = _parse_one_based_pair(
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
            components: list[complex] = []
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
    """Require one basis entry per serialized Wannier function."""
    if len(basis.n) != n_orbitals:
        message: str = (
            f"{path}: basis has {len(basis.n)} orbitals but file declares "
            f"{n_orbitals}"
        )
        raise ValueError(message)


def _validate_hopping_closure(
    records: tuple[_HoppingRecord, ...],
    path: Path,
) -> None:
    """Name duplicate, missing, or numerically inconsistent reverse rows."""
    lookup: dict[
        tuple[int, int, tuple[int, int, int]],
        _HoppingRecord,
    ] = {}
    record: _HoppingRecord
    for record in records:
        key: tuple[int, int, tuple[int, int, int]] = (
            record.pair[0],
            record.pair[1],
            record.cell,
        )
        previous: Optional[_HoppingRecord] = lookup.get(key)
        if previous is not None:
            message: str = (
                f"{path}: rows {previous.line_number} and "
                f"{record.line_number}: duplicate hopping record {key}"
            )
            raise ValueError(message)
        lookup[key] = record
    for key, record in lookup.items():
        orbital_i: int
        orbital_j: int
        cell: tuple[int, int, int]
        orbital_i, orbital_j, cell = key
        reverse_key: tuple[int, int, tuple[int, int, int]] = (
            orbital_j,
            orbital_i,
            (-cell[0], -cell[1], -cell[2]),
        )
        reverse: Optional[_HoppingRecord] = lookup.get(reverse_key)
        if reverse is None:
            message = (
                f"{path}: row {record.line_number}: missing Hermitian reverse "
                f"record {reverse_key}"
            )
            raise ValueError(message)
        difference: float = abs(reverse.amplitude - np.conj(record.amplitude))
        if difference > WANNIER_HERMITICITY_TOLERANCE:
            message = (
                f"{path}: rows {record.line_number} and "
                f"{reverse.line_number}: reverse hopping amplitudes differ "
                f"by {difference:.6e} eV"
            )
            raise ValueError(message)


def _extract_model_data(
    blocks: _HamiltonianBlocks,
    path: Path,
) -> tuple[Float64[NDArray, " n_orb"], tuple[_HoppingRecord, ...]]:
    """Extract real origin diagonals and directed hopping records."""
    origin_indices: list[int] = [
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
    records: list[_HoppingRecord] = []
    cell_index: int
    cell: tuple[int, int, int]
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
                        _HoppingRecord(
                            pair=(orbital_i, orbital_j),
                            cell=cell,
                            amplitude=amplitude,
                            line_number=line_number,
                        )
                    )
    record_tuple: tuple[_HoppingRecord, ...] = tuple(records)
    _validate_hopping_closure(record_tuple, path)
    result: tuple[Float64[NDArray, " n_orb"], tuple[_HoppingRecord, ...]] = (
        onsite,
        record_tuple,
    )
    return result


def _make_model(
    blocks: _HamiltonianBlocks,
    geometry: CrystalGeometry,
    basis: OrbitalBasis,
    path: Path,
    orbital_positions: Optional[Float64[Array, "n_orb 3"]] = None,
) -> TBModel:
    """Convert normalized matrix blocks to a validated native model."""
    onsite: Float64[NDArray, " n_orb"]
    records: tuple[_HoppingRecord, ...]
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


def _spin_permutation(
    basis: OrbitalBasis,
    spin_layout: str,
    path: Path,
) -> tuple[int, ...]:
    """Convert native block-down/up axes to serialized axes."""
    if spin_layout not in ("block_down_up", "interleaved_up_down"):
        message: str = (
            f"{path}: spin_layout must be 'block_down_up' or "
            "'interleaved_up_down'"
        )
        raise ValueError(message)
    n_orbitals: int = len(basis.n)
    if not basis.spin:
        if spin_layout != "block_down_up":
            message = (
                f"{path}: interleaved spin layout requires a spinor basis"
            )
            raise ValueError(message)
        permutation: tuple[int, ...] = tuple(range(n_orbitals))
        return permutation  # noqa: RET504
    if n_orbitals % 2:
        message = f"{path}: spinor basis must contain an even orbital count"
        raise ValueError(message)
    n_spatial: int = n_orbitals // 2
    expected_spin: tuple[int, ...] = (-1,) * n_spatial + (1,) * n_spatial
    if basis.spin != expected_spin:
        message = f"{path}: basis must use native block_down_up spin metadata"
        raise ValueError(message)
    field_name: str
    values: tuple[int, ...]
    for field_name, values in (
        ("atom_indices", basis.atom_indices),
        ("n", basis.n),
        ("l", basis.l),
        ("m", basis.m),
    ):
        if values[:n_spatial] != values[n_spatial:]:
            message = (
                f"{path}: spin-copy {field_name} metadata must match "
                "between native blocks"
            )
            raise ValueError(message)
    if spin_layout == "block_down_up":
        permutation = tuple(range(n_orbitals))
        return permutation  # noqa: RET504
    down_serialized: tuple[int, ...] = tuple(
        2 * index + 1 for index in range(n_spatial)
    )
    up_serialized: tuple[int, ...] = tuple(
        2 * index for index in range(n_spatial)
    )
    permutation = down_serialized + up_serialized
    return permutation  # noqa: RET504


def _permute_hamiltonian_blocks(
    blocks: _HamiltonianBlocks,
    permutation: tuple[int, ...],
) -> _HamiltonianBlocks:
    """Apply one state permutation to both Hamiltonian axes and line maps."""
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
    permuted_blocks: _HamiltonianBlocks = _HamiltonianBlocks(
        matrices=matrices,
        source_lines=source_lines,
        cells=blocks.cells,
        degeneracies=blocks.degeneracies,
    )
    return permuted_blocks


def _permute_position_matrices(
    matrices: Complex128[NDArray, "n_cell n_orb n_orb 3"],
    permutation: tuple[int, ...],
) -> Complex128[NDArray, "n_cell n_orb n_orb 3"]:
    """Apply one state permutation to both position-operator axes."""
    permuted: Complex128[NDArray, "n_cell n_orb n_orb 3"] = np.take(
        matrices, permutation, axis=1
    )
    result: Complex128[NDArray, "n_cell n_orb n_orb 3"] = np.take(
        permuted, permutation, axis=2
    )
    return result


def _centres_from_position_matrices(
    matrices: Complex128[NDArray, "n_cell n_orb n_orb 3"],
    cells: tuple[tuple[int, int, int], ...],
    path: Path,
) -> Float64[NDArray, "n_orb 3"]:
    """Extract real origin-diagonal Wannier centres in Angstrom."""
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
    """Build atom positions after validating orbital-centre assumptions."""
    atom_indices: tuple[int, ...] = basis.atom_indices
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
        orbital_rows: list[int] = [
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
    """Validate explicit ``hr.dat`` centres without altering connectivity."""
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
    """Convert Cartesian Wannier centres to fractional coordinates."""
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
    """Resolve atomic geometry without conflating atoms and Wannier centres."""
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
def read_hopping_list(  # noqa: DOC502, DOC503, PLR0913, PLR0915
    filename: str,
    geometry: CrystalGeometry,
    basis: OrbitalBasis,
    shell_index: tuple[int, ...] = (),
    soc_lambdas: Optional[Float64[Array, " n_shells"]] = None,
) -> TBModel:
    r"""Parse a zero-based Cartesian tight-binding hopping list.

    Each nonblank comma-separated row has
    ``o1,o2,x,y,z,real[,imag]``. Orbital indices are zero-based, Cartesian
    bond components are in Angstrom, and amplitudes are in eV.

    :see: :class:`~.test_tb_files.TestReadHoppingList`

    Parameters
    ----------
    filename : str
        Hopping-list path.
    geometry : CrystalGeometry
        Lattice and fractional atomic positions used to recover exact cells.
    basis : OrbitalBasis
        Orbital-to-atom metadata.
    shell_index : tuple[int, ...], optional
        Orbital-to-SOC-shell map. An empty tuple selects ``-1`` for every
        orbital. Default is empty.
    soc_lambdas : Optional[Float64[Array, " n_shells"]], optional
        SOC energies in eV. Default is an empty array.

    Returns
    -------
    model : TBModel
        Validated native model in the basis-position gauge.

    Raises
    ------
    ValueError
        If validation rejects rows, orbital indices, recovered cells, onsite
        entries, or Hermitian closure.
    EquinoxRuntimeError
        If the final native carrier rejects a traced numerical invariant.

    Notes
    -----
    Convert Cartesian ``d`` once to fractional coordinates. The exact cell
    candidate is
    :math:`R=d-\tau_j+\tau_i`. The parser checks every component against its
    nearest integer at tolerance ``1e-10`` and only then casts it to an
    integer. A failing diagnostic names the offending source row.
    """
    path: Path = Path(filename)
    lines: tuple[str, ...] = tuple(
        path.read_text(encoding="utf-8").splitlines()
    )
    n_orbitals: int = len(basis.n)
    lattice: Float64[NDArray, "3 3"] = np.asarray(
        geometry.lattice, dtype=np.float64
    )
    positions: Float64[NDArray, "n_atom 3"] = np.asarray(
        geometry.positions, dtype=np.float64
    )
    if any(index >= positions.shape[0] for index in basis.atom_indices):
        message: str = (
            f"{path}: basis atom_indices must refer to geometry position rows"
        )
        raise ValueError(message)
    _spin_permutation(basis, "block_down_up", path)
    try:
        inverse_lattice: Float64[NDArray, "3 3"] = np.linalg.inv(lattice)
    except np.linalg.LinAlgError as error:
        error: np.linalg.LinAlgError
        message: str = f"{path}: geometry lattice must be nonsingular"
        raise ValueError(message) from error
    onsite: Float64[NDArray, " n_orb"] = np.zeros(
        (n_orbitals,), dtype=np.float64
    )
    onsite_lines: dict[int, int] = {}
    records: list[_HoppingRecord] = []
    saw_row: bool = False
    line_number: int
    text: str
    for line_number, text in enumerate(lines, start=1):
        if not text.strip():
            continue
        saw_row = True
        tokens: list[str] = [token.strip() for token in text.split(",")]
        if len(tokens) not in (
            HOPPING_LIST_REAL_FIELDS,
            HOPPING_LIST_COMPLEX_FIELDS,
        ) or any(not token for token in tokens):
            message: ValueError = _line_error(
                path,
                line_number,
                "hopping row must contain six or seven comma-separated fields",
            )
            raise message
        orbital_i: int = _parse_integer(
            tokens[0],
            path,
            line_number,
            "first zero-based orbital index",
        )
        orbital_j: int = _parse_integer(
            tokens[1],
            path,
            line_number,
            "second zero-based orbital index",
        )
        if not (0 <= orbital_i < n_orbitals and 0 <= orbital_j < n_orbitals):
            message = _line_error(
                path,
                line_number,
                f"orbital indices must be in [0, {n_orbitals})",
            )
            raise message
        cartesian: Float64[NDArray, " 3"] = np.asarray(
            [
                _parse_finite_float(
                    tokens[index],
                    path,
                    line_number,
                    "Cartesian bond component",
                )
                for index in range(2, 5)
            ],
            dtype=np.float64,
        )
        real: float = _parse_finite_float(
            tokens[5],
            path,
            line_number,
            "hopping real part",
        )
        imaginary: float = (
            _parse_finite_float(
                tokens[6],
                path,
                line_number,
                "hopping imaginary part",
            )
            if len(tokens) == HOPPING_LIST_COMPLEX_FIELDS
            else 0.0
        )
        atom_i: int = basis.atom_indices[orbital_i]
        atom_j: int = basis.atom_indices[orbital_j]
        fractional_displacement: Float64[NDArray, " 3"] = (
            cartesian @ inverse_lattice
        )
        candidate: Float64[NDArray, " 3"] = (
            fractional_displacement - positions[atom_j] + positions[atom_i]
        )
        nearest: Float64[NDArray, " 3"] = np.rint(candidate)
        deviation: Float64[NDArray, " 3"] = np.abs(candidate - nearest)
        if np.any(deviation > WANNIER_INTEGER_RECOVERY_TOLERANCE):
            message = _line_error(
                path,
                line_number,
                "bond vector gives noninteger cell candidate "
                f"{candidate.tolist()} beyond "
                f"{WANNIER_INTEGER_RECOVERY_TOLERANCE:.0e} tolerance",
            )
            raise message
        cell: tuple[int, int, int] = tuple(
            int(component) for component in nearest
        )
        amplitude: complex = complex(real, imaginary)
        if orbital_i == orbital_j and cell == (0, 0, 0):
            previous_line: Optional[int] = onsite_lines.get(orbital_i)
            if previous_line is not None:
                message = ValueError(
                    f"{path}: rows {previous_line} and {line_number}: "
                    f"duplicate onsite entry for orbital {orbital_i}"
                )
                raise message
            if abs(imaginary) > WANNIER_HERMITICITY_TOLERANCE:
                message = _line_error(
                    path,
                    line_number,
                    "onsite hopping entry must be real",
                )
                raise message
            onsite[orbital_i] = real
            onsite_lines[orbital_i] = line_number
        else:
            records.append(
                _HoppingRecord(
                    pair=(orbital_i, orbital_j),
                    cell=cell,
                    amplitude=amplitude,
                    line_number=line_number,
                )
            )
    if not saw_row:
        message = f"{path}: hopping list contains no records"
        raise ValueError(message)
    record_tuple: tuple[_HoppingRecord, ...] = tuple(records)
    _validate_hopping_closure(record_tuple, path)
    resolved_shell_index: tuple[int, ...] = (
        shell_index if shell_index else (-1,) * n_orbitals
    )
    resolved_soc: Float64[Array, " n_shells"] = (
        jnp.zeros((0,), dtype=jnp.float64)
        if soc_lambdas is None
        else jnp.asarray(soc_lambdas, dtype=jnp.float64)
    )
    model: TBModel = make_tb_model(
        hopping_amplitudes=jnp.asarray(
            [record.amplitude for record in record_tuple],
            dtype=jnp.complex128,
        ),
        onsite_energies=jnp.asarray(onsite, dtype=jnp.float64),
        soc_lambdas=resolved_soc,
        geometry=geometry,
        basis=basis,
        hopping_pairs=tuple(record.pair for record in record_tuple),
        hopping_cells=tuple(record.cell for record in record_tuple),
        shell_index=resolved_shell_index,
        spinor=bool(basis.spin),
    )
    return model


@jaxtyped(typechecker=beartype)
def read_wannier90_hr(  # noqa: DOC502
    filename: str,
    geometry: CrystalGeometry,
    basis: OrbitalBasis,
    centres_cart: Float64[Array, "n_orb 3"],
) -> tuple[TBModel, WannierOperatorData]:
    """Parse a normative Wannier90 ``seedname_hr.dat`` file.

    Read exact real-space Hamiltonian cells and combine them with required
    external Wannier centres.

    :see: :class:`~.test_tb_files.TestReadWannier90Hr`

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
    result : tuple[TBModel, WannierOperatorData]
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
    cursor: _LineCursor = _LineCursor.from_path(path)
    _parse_header(cursor)
    n_orbitals: int
    n_cells: int
    degeneracies: tuple[int, ...]
    n_orbitals, n_cells, degeneracies = _parse_wannier_dimensions(cursor)
    _validate_basis_size(basis, n_orbitals, path)
    _spin_permutation(basis, "block_down_up", path)
    blocks: _HamiltonianBlocks = _parse_hr_hamiltonian_blocks(
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
    result: tuple[TBModel, WannierOperatorData] = (model, operator_data)
    return result


@jaxtyped(typechecker=beartype)
def read_wannier90_tb(  # noqa: DOC502
    filename: str,
    basis: OrbitalBasis,
    spin_layout: str,
    geometry: Optional[CrystalGeometry] = None,
) -> tuple[TBModel, WannierOperatorData]:
    """Parse a normative Wannier90 ``seedname_tb.dat`` file.

    Read lattice, Hamiltonian, and position-operator blocks while normalizing
    all serialized spin axes into the native convention.

    :see: :class:`~.test_tb_files.TestReadWannier90Tb`

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
    result : tuple[TBModel, WannierOperatorData]
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
    cursor: _LineCursor = _LineCursor.from_path(path)
    _parse_header(cursor)
    lattice: Float64[NDArray, "3 3"] = _parse_tb_lattice(cursor)
    n_orbitals: int
    n_cells: int
    degeneracies: tuple[int, ...]
    n_orbitals, n_cells, degeneracies = _parse_wannier_dimensions(cursor)
    _validate_basis_size(basis, n_orbitals, path)
    serialized_blocks: _HamiltonianBlocks = _parse_tb_hamiltonian_blocks(
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
    permutation: tuple[int, ...] = _spin_permutation(
        basis,
        spin_layout,
        path,
    )
    blocks: _HamiltonianBlocks = _permute_hamiltonian_blocks(
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
    result: tuple[TBModel, WannierOperatorData] = (model, operator_data)
    return result


__all__: list[str] = [
    "read_hopping_list",
    "read_wannier90_hr",
    "read_wannier90_tb",
]
