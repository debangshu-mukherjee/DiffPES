"""Parse a VASP CHGCAR file.

Extended Summary
----------------
The module reads VASP CHGCAR files and returns a carrier with the crystal
geometry, charge density, and optional magnetization density.
Scalar data uses :class:`~diffpes.types.VolumetricData`.
Four-block vector data uses :class:`~diffpes.types.SOCVolumetricData`.

Routine Listings
----------------
:func:`read_chgcar`
    Parse a VASP CHGCAR file.
"""

from pathlib import Path

import jax.numpy as jnp
import numpy as np
from beartype import beartype
from beartype.typing import List, Optional, TextIO, Tuple
from jaxtyping import Array, Float64, Int32, jaxtyped
from numpy.typing import NDArray

from diffpes.constants import (
    LATTICE_ROWS,
    N_SOC_MAG_BLOCKS,
    SCALAR_LINE_COMPONENTS,
    XYZ_COMPONENTS,
)
from diffpes.types import (
    SOCVolumetricData,
    VolumetricData,
    make_soc_volumetric_data,
    make_volumetric_data,
)


@jaxtyped(typechecker=beartype)
def read_chgcar(
    filename: str = "CHGCAR",
) -> VolumetricData | SOCVolumetricData:
    r"""Parse a VASP CHGCAR file.

    The parser supports three layouts:

    - **ISPIN=1**: 1 grid block (charge only).
    - **ISPIN=2**: 2 grid blocks (charge + scalar magnetization).
    - **SOC** (LSORBIT): 4 grid blocks (charge, mx, my, mz).

    The CHGCAR file starts with a POSCAR-style structural header. One or
    more volumetric blocks follow the header on a real-space FFT grid.
    Each block starts with the three grid dimensions. VASP then writes
    the flattened values in Fortran column-major order.

    :see: :class:`~.test_chgcar.TestReadChgcar`

    Implementation Logic
    --------------------
    1. **Read the structural header and remaining records**::

           path: Path = Path(filename)
           with path.open("r") as fid:
               lattice, coords, symbols, atom_counts = _read_poscar_header(fid)
               rest_lines: List[str] = [
                   line.rstrip("\n") for line in fid
               ]

       This separates the geometry from the subsequent volumetric blocks.

    2. **Normalize the charge grid by the cell volume**::

           charge_grid: Float64[NDArray, "Nx Ny Nz"] = (
               charge_vals.reshape(grid_shape, order="F") / volume
           )

       This converts VASP charge-times-volume values to charge density.

    3. **Return the carrier for the detected layout**::

           return volumetric

       The grid-count branch binds scalar or SOC data to one output name.

    Parameters
    ----------
    filename : str, optional
        Path to the CHGCAR file. Default is ``"CHGCAR"``.

    Returns
    -------
    volumetric : VolumetricData or SOCVolumetricData
        ``VolumetricData`` for ISPIN=1 or ISPIN=2 files.
        ``SOCVolumetricData`` for SOC files with four grid blocks.

    Raises
    ------
    ValueError
        If the lattice volume is zero, the grid dimensions are absent,
        or a data block is incomplete.

    Notes
    -----
    The returned densities use electrons per cubic Angstrom. The parser
    divides the raw VASP values by the cell volume. It stores coordinates
    in fractional form and preserves the VASP Fortran grid order.
    """
    fid: TextIO

    path: Path = Path(filename)
    with path.open("r") as fid:
        lattice: Float64[NDArray, "3 3"]
        coords: Float64[NDArray, "N 3"]
        symbols: Tuple[str, ...]
        atom_counts: List[int]
        lattice, coords, symbols, atom_counts = _read_poscar_header(fid)
        rest_lines: List[str] = [line.rstrip("\n") for line in fid]

    volume: float = abs(
        float(
            np.dot(
                lattice[0, :],
                np.cross(lattice[1, :], lattice[2, :]),
            )
        )
    )
    if volume == 0.0:
        msg: str = "CHGCAR lattice volume is zero."
        raise ValueError(msg)

    first_grid_idx: Optional[int]
    grid_shape: Tuple[int, int, int]
    first_grid_idx, grid_shape = _find_next_grid_line(rest_lines, 0)
    if first_grid_idx is None:
        msg: str = "Could not locate CHGCAR charge-density grid dimensions."
        raise ValueError(msg)

    ngrid: int = int(np.prod(np.asarray(grid_shape, dtype=np.int64)))
    charge_vals: Float64[NDArray, " ngrid"]
    end_idx: int
    charge_vals, end_idx = _parse_float_block(
        rest_lines,
        first_grid_idx + 1,
        ngrid,
    )
    charge_grid: Float64[NDArray, "Nx Ny Nz"] = (
        charge_vals.reshape(grid_shape, order="F") / volume
    )

    mag_grids: List[Float64[NDArray, "Nx Ny Nz"]] = []
    scan_idx: int = end_idx
    while len(mag_grids) < N_SOC_MAG_BLOCKS:
        next_idx: Optional[int]
        next_shape: Tuple[int, int, int]
        next_idx, next_shape = _find_next_grid_line(rest_lines, scan_idx)
        if next_idx is None:
            break
        ngrid_mag: int = int(np.prod(np.asarray(next_shape, dtype=np.int64)))
        mag_vals: Float64[NDArray, " ngrid"]
        mag_vals, scan_idx = _parse_float_block(
            rest_lines,
            next_idx + 1,
            ngrid_mag,
        )
        mag_grids.append(mag_vals.reshape(next_shape, order="F") / volume)

    lattice_arr: Float64[Array, "3 3"] = jnp.asarray(
        lattice, dtype=jnp.float64
    )
    coords_arr: Float64[Array, "N 3"] = jnp.asarray(coords, dtype=jnp.float64)
    charge_arr: Float64[Array, "Nx Ny Nz"] = jnp.asarray(
        charge_grid, dtype=jnp.float64
    )
    counts_arr: Int32[Array, " S"] = jnp.asarray(atom_counts, dtype=jnp.int32)

    volumetric: VolumetricData | SOCVolumetricData
    if len(mag_grids) == N_SOC_MAG_BLOCKS:
        mag_vector: Float64[NDArray, "Nx Ny Nz 3"] = np.stack(
            mag_grids, axis=-1
        )
        volumetric = make_soc_volumetric_data(
            lattice=lattice_arr,
            coords=coords_arr,
            charge=charge_arr,
            magnetization=jnp.asarray(mag_grids[2], dtype=jnp.float64),
            magnetization_vector=jnp.asarray(mag_vector, dtype=jnp.float64),
            grid_shape=grid_shape,
            symbols=symbols,
            atom_counts=counts_arr,
        )
        return volumetric

    volumetric = make_volumetric_data(
        lattice=lattice_arr,
        coords=coords_arr,
        charge=charge_arr,
        magnetization=(
            None
            if not mag_grids
            else jnp.asarray(mag_grids[0], dtype=jnp.float64)
        ),
        grid_shape=grid_shape,
        symbols=symbols,
        atom_counts=counts_arr,
    )
    return volumetric


@jaxtyped(typechecker=beartype)
def _read_poscar_header(
    fid: TextIO,
) -> Tuple[
    Float64[NDArray, "3 3"],
    Float64[NDArray, "N 3"],
    Tuple[str, ...],
    List[int],
]:
    """PRIVATE: Read the POSCAR-like header section at the start of a CHGCAR
    file.

    A CHGCAR file starts with a POSCAR-compatible section. This section
    contains a comment, a scale, lattice vectors, atom data, and coordinates.
    It can also contain species names and a selective-dynamics flag. The helper
    reads these fields from the open file. It returns NumPy arrays and Python
    containers.

    Implementation Logic
    --------------------
    1. Read and discard the comment line.
    2. Read the universal scaling factor (float).
    3. Read three lattice vectors with three values each.
    4. Multiply the lattice vectors by the scaling factor.
    5. Read the next line.
    6. If the line contains no digits, read it as the VASP-5 symbol line.
    7. Read the atom counts from the next line.
    8. Otherwise, read the current line as the atom counts.
    9. Compute the total atom count as the sum of the species counts.
    10. Read the coordinate-type line.
    11. If the line starts with ``'s'`` or ``'S'``, read the next line.
    12. Determine whether coordinates are Cartesian (``'c'``/``'k'``)
       or direct (fractional).
    13. Read ``natoms`` coordinate lines with three values each.
    14. If Cartesian, apply the scaling factor and convert to
       fractional via ``np.linalg.solve(lattice.T, coords.T).T``.

    Parameters
    ----------
    fid : TextIO
        Open file handle positioned at the start of the CHGCAR file.

    Returns
    -------
    lattice : Float64[NDArray, "3 3"]
        Scaled lattice vectors, shape ``(3, 3)``.
    coords : Float64[NDArray, "N 3"]
        Fractional atomic coordinates, shape ``(natoms, 3)``.
    symbols : Tuple[str, ...]
        Element symbols (empty tuple for VASP-4 style files).
    atom_counts : List[int]
        Number of atoms per species.

    Raises
    ------
    ValueError
        If a lattice line has fewer than 3 components or a coordinate
        line has fewer than 3 components.
    """
    row: int
    atom_idx: int

    _comment: str = fid.readline().strip()
    scale: float = float(fid.readline().strip())

    lattice: Float64[NDArray, "3 3"] = np.zeros(
        (LATTICE_ROWS, XYZ_COMPONENTS),
        dtype=np.float64,
    )
    for row in range(LATTICE_ROWS):
        vals: List[float] = [float(x) for x in fid.readline().split()]
        if len(vals) < XYZ_COMPONENTS:
            msg: str = "Invalid CHGCAR lattice line."
            raise ValueError(msg)
        lattice[row, :] = vals[:XYZ_COMPONENTS]
    lattice = lattice * scale

    line: str = fid.readline().strip()
    symbols: Tuple[str, ...] = ()
    if line and not any(char.isdigit() for char in line):
        symbols = tuple(line.split())
        line = fid.readline().strip()
    atom_counts: List[int] = [int(x) for x in line.split()]
    natoms: int = sum(atom_counts)

    coord_line: str = fid.readline().strip()
    if coord_line and coord_line[0].lower() == "s":
        coord_line = fid.readline().strip()
    cartesian: bool = bool(coord_line) and coord_line[0].lower() in ("c", "k")

    coords: Float64[NDArray, "N 3"] = np.zeros(
        (natoms, XYZ_COMPONENTS), dtype=np.float64
    )
    for atom_idx in range(natoms):
        vals = [float(x) for x in fid.readline().split()[:XYZ_COMPONENTS]]
        if len(vals) < XYZ_COMPONENTS:
            msg: str = "Invalid CHGCAR coordinate line."
            raise ValueError(msg)
        coords[atom_idx, :] = vals

    if cartesian:
        coords = coords * scale
        coords = np.linalg.solve(lattice.T, coords.T).T

    header_data: Tuple[
        Float64[NDArray, "3 3"],
        Float64[NDArray, "N 3"],
        Tuple[str, ...],
        List[int],
    ] = (lattice, coords, symbols, atom_counts)
    return header_data


@jaxtyped(typechecker=beartype)
def _find_next_grid_line(
    lines: List[str],
    start_idx: int,
) -> Tuple[Optional[int], Tuple[int, int, int]]:
    """PRIVATE: Find the next line containing exactly three positive
    integers.

    The helper scans ``lines`` from ``start_idx`` for the next FFT grid
    header. One line with three positive integers precedes each CHGCAR
    volumetric block.

    Implementation Logic
    --------------------
    1. Iterate from ``start_idx`` to the end of ``lines``.
    2. Skip blank lines and lines that do not split into exactly 3
       tokens.
    3. Attempt to parse all three tokens as integers. If any token
       fails to convert, skip the line.
    4. If all three integers are positive, return the line index and the
       ``(NGX, NGY, NGZ)`` tuple.
    5. Return ``(None, (0, 0, 0))`` if no line matches.

    Parameters
    ----------
    lines : List[str]
        All CHGCAR lines after the parser consumes the POSCAR header.
    start_idx : int
        Index within ``lines`` at which to begin scanning.

    Returns
    -------
    idx : int or None
        Line index of the grid header, or ``None`` if not found.
    grid_shape : Tuple[int, int, int]
        ``(NGX, NGY, NGZ)`` grid dimensions, or ``(0, 0, 0)`` if
        not found.
    """
    idx: int

    for idx in range(start_idx, len(lines)):
        stripped: str = lines[idx].strip()
        if not stripped:
            continue
        parts: List[str] = stripped.split()
        if len(parts) != SCALAR_LINE_COMPONENTS:
            continue
        try:
            values: Tuple[int, int, int] = (
                int(parts[0]),
                int(parts[1]),
                int(parts[2]),
            )
        except ValueError:
            continue
        if values[0] > 0 and values[1] > 0 and values[2] > 0:
            grid_line: Tuple[Optional[int], Tuple[int, int, int]] = (
                idx,
                values,
            )
            return grid_line
    grid_line = (None, (0, 0, 0))
    return grid_line  # noqa: RET504 -- assign-before-return is required.


@jaxtyped(typechecker=beartype)
def _parse_float_block(
    lines: List[str],
    start_idx: int,
    nvals: int,
) -> Tuple[Float64[NDArray, " nvals"], int]:
    """PRIVATE: Parse ``nvals`` whitespace-separated floats starting at
    ``start_idx``.

    The helper reads a continuous block of floating-point values from multiple
    CHGCAR lines. VASP usually writes 5 or 10 values per line. The exact count
    can vary. The helper reads lines until it collects ``nvals`` values.

    Implementation Logic
    --------------------
    1. Initialize an empty collection and a running line index.
    2. For each line starting at ``start_idx``:

       a. Skip blank lines.
       b. Attempt to parse every whitespace-separated token as a float.
       c. If one token is invalid, treat the complete line as non-data.
       d. Stop before the invalid line.
       e. Append only the values that the collection still needs.

    3. Verify that the collection contains exactly ``nvals`` values.
    4. Raise ``ValueError`` after a short read.

    Parameters
    ----------
    lines : List[str]
        All remaining lines of the CHGCAR file.
    start_idx : int
        Index within ``lines`` at which to begin reading floats.
    nvals : int
        Total number of floats to collect.

    Returns
    -------
    value_arr : Float64[NDArray, " nvals"]
        1D array of shape ``(nvals,)`` with dtype ``float64``.
    end_idx : int
        Index of the first line *after* the last consumed line,
        suitable for passing as ``start_idx`` to a subsequent call.

    Raises
    ------
    ValueError
        If the helper reaches the end of ``lines`` before it collects
        ``nvals`` floats.
    """
    token: str

    values: List[float] = []
    idx: int = start_idx

    while idx < len(lines) and len(values) < nvals:
        stripped: str = lines[idx].strip()
        if not stripped:
            idx += 1
            continue

        parts: List[str] = stripped.split()
        row_vals: List[float] = []
        row_valid: bool = True
        for token in parts:
            try:
                row_vals.append(float(token))
            except ValueError:
                row_valid = False
                break
        if row_valid:
            needed: int = nvals - len(values)
            values.extend(row_vals[:needed])
        idx += 1

    if len(values) != nvals:
        msg: str = "Unexpected end of CHGCAR data block."
        raise ValueError(msg)
    value_arr: Float64[NDArray, " nvals"] = np.asarray(
        values, dtype=np.float64
    )
    parsed_block: Tuple[Float64[NDArray, " nvalues"], int] = (value_arr, idx)
    return parsed_block


__all__: list[str] = [
    "read_chgcar",
]
