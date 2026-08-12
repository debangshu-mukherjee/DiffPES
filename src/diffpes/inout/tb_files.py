r"""Parse explicit Cartesian tight-binding hopping lists.

Extended Summary
----------------
This module is the host-side airlock for the zero-based Cartesian hopping
list grammar. Parsing uses NumPy and exact Python integer metadata;
validated outputs are native JAX/Equinox carriers.

Routine Listings
----------------
:func:`read_hopping_list`
    Parse a zero-based Cartesian tight-binding hopping list.
"""

from pathlib import Path

import jax.numpy as jnp
import numpy as np
from beartype import beartype
from beartype.typing import Dict, List, Optional, Tuple
from jaxtyping import Array, Float64, jaxtyped
from numpy.typing import NDArray

from diffpes.constants import (
    HOPPING_LIST_COMPLEX_FIELDS,
    HOPPING_LIST_REAL_FIELDS,
    WANNIER_HERMITICITY_TOLERANCE,
    WANNIER_INTEGER_RECOVERY_TOLERANCE,
)
from diffpes.types import (
    CrystalGeometry,
    HoppingRecord,
    OrbitalBasis,
    TBModel,
    make_hopping_record,
    make_tb_model,
)


def _line_error(path: Path, line_number: int, message: str) -> ValueError:
    """PRIVATE: Build a line-numbered parser error.

    Parameters
    ----------
    path : Path
        Source file the diagnostic names.
    line_number : int
        One-based physical line number of the offending record.
    message : str
        Human-readable description of the defect.

    Returns
    -------
    error : ValueError
        Constructed exception with the ``path: line N: message`` text;
        the caller raises it.

    Notes
    -----
    Only formats and returns the exception.  Returning instead of
    raising lets callers attach ``from`` causes at the raise site.
    """
    error: ValueError = ValueError(f"{path}: line {line_number}: {message}")
    return error


def _parse_integer(  # noqa: DOC503 -- raises a prebuilt line error.
    token: str,
    path: Path,
    line_number: int,
    label: str,
) -> int:
    """PRIVATE: Parse one exact integer token.

    Parameters
    ----------
    token : str
        Whitespace-stripped text token to convert.
    path : Path
        Source file for the diagnostic.
    line_number : int
        One-based physical line number for the diagnostic.
    label : str
        Field name the diagnostic quotes.

    Returns
    -------
    value : int
        Exact Python integer parsed from the token.

    Raises
    ------
    ValueError
        If ``int(token)`` rejects the token.

    Notes
    -----
    Uses the strict ``int`` constructor, so decimal notation such as
    ``"2.0"`` fails and keeps file cells exact.
    """
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


def _parse_finite_float(  # noqa: DOC503 -- raises a prebuilt line error.
    token: str,
    path: Path,
    line_number: int,
    label: str,
) -> float:
    """PRIVATE: Parse one finite floating-point token.

    Parameters
    ----------
    token : str
        Whitespace-stripped text token to convert.
    path : Path
        Source file for the diagnostic.
    line_number : int
        One-based physical line number for the diagnostic.
    label : str
        Field name the diagnostic quotes.

    Returns
    -------
    value : float
        Finite Python float parsed from the token.

    Raises
    ------
    ValueError
        If ``float(token)`` rejects the token or the value is NaN or
        infinite.

    Notes
    -----
    Accepts any ``float``-parseable notation and then applies
    :func:`np.isfinite`, so ``"nan"`` and ``"inf"`` fail with a
    line-numbered diagnostic.
    """
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


def _validate_hopping_closure(
    records: Tuple[HoppingRecord, ...],
    path: Path,
) -> None:
    """PRIVATE: Name duplicate, missing, or numerically inconsistent reverse
    rows.

    Implementation Logic
    --------------------
    Indexes every record by ``(i, j, R)`` and rejects duplicates.  For
    each record it then requires the Hermitian partner
    ``(j, i, -R)`` to exist and checks
    ``abs(reverse - conj(amplitude))`` against the ``1e-12`` eV
    Hermiticity tolerance.  Diagnostics quote the one-based source
    lines of both offending rows.

    Parameters
    ----------
    records : Tuple[HoppingRecord, ...]
        Directed non-onsite hopping records with source lines.
    path : Path
        Source file for the diagnostics.

    Raises
    ------
    ValueError
        If a record repeats, its reverse partner is missing, or the
        reverse amplitude is not the complex conjugate within
        tolerance.
    """
    lookup: Dict[
        Tuple[int, int, Tuple[int, int, int]],
        HoppingRecord,
    ] = {}
    record: HoppingRecord
    for record in records:
        key: Tuple[int, int, Tuple[int, int, int]] = (
            record.pair[0],
            record.pair[1],
            record.cell,
        )
        previous: Optional[HoppingRecord] = lookup.get(key)
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
        cell: Tuple[int, int, int]
        orbital_i, orbital_j, cell = key
        reverse_key: Tuple[int, int, Tuple[int, int, int]] = (
            orbital_j,
            orbital_i,
            (-cell[0], -cell[1], -cell[2]),
        )
        reverse: Optional[HoppingRecord] = lookup.get(reverse_key)
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


def _spin_permutation(
    basis: OrbitalBasis,
    spin_layout: str,
    path: Path,
) -> Tuple[int, ...]:
    """PRIVATE: Convert native block-down/up axes to serialized axes.

    Implementation Logic
    --------------------
    Validates the layout label, then the basis: a non-spinor basis
    admits only ``"block_down_up"`` and yields the identity.  A spinor
    basis must hold an even orbital count and native spin metadata
    ``(-1, ..., -1, 1, ..., 1)``.  The per-spin copies of the
    ``atom_indices``, ``n``, ``l``, and ``m`` tuples must match.
    ``"block_down_up"`` then also yields the identity.
    ``"interleaved_up_down"`` maps native down index ``i`` to
    serialized index ``2 i + 1`` and native up index
    ``n_spatial + i`` to serialized index ``2 i``.

    Parameters
    ----------
    basis : OrbitalBasis
        Orbital metadata in the native block-down/up order.
    spin_layout : str
        Serialized layout: ``"block_down_up"`` or
        ``"interleaved_up_down"``.
    path : Path
        Source file for diagnostics.

    Returns
    -------
    permutation : Tuple[int, ...]
        For each native basis index, the serialized file index; a
        gather with this permutation reorders serialized matrices into
        the native layout.

    Raises
    ------
    ValueError
        If the layout label is unknown, interleaved layout meets a
        non-spinor basis, the spinor count is odd, or spin metadata
        breaks the convention.
    """
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
        permutation: Tuple[int, ...] = tuple(range(n_orbitals))
        return permutation  # noqa: RET504
    if n_orbitals % 2:
        message = f"{path}: spinor basis must contain an even orbital count"
        raise ValueError(message)
    n_spatial: int = n_orbitals // 2
    expected_spin: Tuple[int, ...] = (-1,) * n_spatial + (1,) * n_spatial
    if basis.spin != expected_spin:
        message = f"{path}: basis must use native block_down_up spin metadata"
        raise ValueError(message)
    field_name: str
    values: Tuple[int, ...]
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
    down_serialized: Tuple[int, ...] = tuple(
        2 * index + 1 for index in range(n_spatial)
    )
    up_serialized: Tuple[int, ...] = tuple(
        2 * index for index in range(n_spatial)
    )
    permutation = down_serialized + up_serialized
    return permutation  # noqa: RET504


@jaxtyped(typechecker=beartype)
def read_hopping_list(  # noqa: DOC502, DOC503, PLR0913, PLR0915
    filename: str,
    geometry: CrystalGeometry,
    basis: OrbitalBasis,
    shell_index: Tuple[int, ...] = (),
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
    shell_index : Tuple[int, ...], optional
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
    lines: Tuple[str, ...] = tuple(
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
    onsite_lines: Dict[int, int] = {}
    records: List[HoppingRecord] = []
    saw_row: bool = False
    line_number: int
    text: str
    for line_number, text in enumerate(lines, start=1):
        if not text.strip():
            continue
        saw_row = True
        tokens: List[str] = [token.strip() for token in text.split(",")]
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
        cell: Tuple[int, int, int] = tuple(
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
                make_hopping_record(
                    pair=(orbital_i, orbital_j),
                    cell=cell,
                    amplitude=amplitude,
                    line_number=line_number,
                )
            )
    if not saw_row:
        message = f"{path}: hopping list contains no records"
        raise ValueError(message)
    record_tuple: Tuple[HoppingRecord, ...] = tuple(records)
    _validate_hopping_closure(record_tuple, path)
    resolved_shell_index: Tuple[int, ...] = (
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


__all__: list[str] = ["read_hopping_list"]
