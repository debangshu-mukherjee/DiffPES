"""VASP CHGCAR file parser.

Extended Summary
----------------
Reads VASP CHGCAR volumetric files and returns a
:class:`~diffpes.types.VolumetricData` PyTree containing the crystal
geometry, charge density, and optional magnetization density.
For SOC calculations (4 grid blocks), returns an
:class:`~diffpes.types.SOCVolumetricData` with vector magnetization.

Routine Listings
----------------
:func:`read_chgcar`
    Parse a VASP CHGCAR file into VolumetricData or SOCVolumetricData.
"""

from pathlib import Path

import jax.numpy as jnp
import numpy as np
from beartype.typing import Optional, Tuple

from diffpes.types import (
    SOCVolumetricData,
    VolumetricData,
    make_soc_volumetric_data,
    make_volumetric_data,
)

_LATTICE_ROWS: int = 3
_XYZ_COMPONENTS: int = 3
_SCALAR_LINE_COMPONENTS: int = 3


_N_SOC_MAG_BLOCKS: int = 3


def read_chgcar(
    filename: str = "CHGCAR",
) -> VolumetricData | SOCVolumetricData:
    """Parse a VASP CHGCAR file.

    Supports three layouts:

    - **ISPIN=1**: 1 grid block (charge only).
    - **ISPIN=2**: 2 grid blocks (charge + scalar magnetization).
    - **SOC** (LSORBIT): 4 grid blocks (charge, mx, my, mz).

    Extended Summary
    ----------------
    The CHGCAR file produced by VASP begins with a POSCAR-style header
    (comment, scaling factor, lattice vectors, species, atom counts,
    coordinate type, and atomic positions) followed by one or more
    volumetric data blocks on a real-space FFT grid. Each block starts
    with a line of three integers (``NGX NGY NGZ``) giving the grid
    dimensions, followed by the flattened grid values written in
    Fortran column-major order (x fastest, z slowest).

    Implementation Logic
    --------------------
    1. **Read POSCAR header** -- delegates to :func:`_read_poscar_header`
       to obtain the lattice matrix, fractional coordinates, element
       symbols, and atom counts.

    2. **Compute cell volume** -- ``|a . (b x c)|`` from the three
       lattice vectors. Raises if the volume is zero (degenerate cell).

    3. **Locate first grid block** -- scans remaining lines for the
       first line containing exactly three positive integers (the FFT
       grid dimensions) via :func:`_find_next_grid_line`.

    4. **Parse charge density** -- reads ``NGX * NGY * NGZ`` floats
       from subsequent lines via :func:`_parse_float_block`, reshapes
       in Fortran order, and divides by the cell volume to convert
       from the VASP convention (charge * volume) to charge density
       (electrons / Angstrom^3).

    5. **Parse magnetization blocks** -- repeats the grid-search and
       float-parsing loop up to three more times (for ISPIN=2 or SOC
       calculations). The loop terminates early if no further grid
       header is found.

    6. **Construct return value** -- based on the number of
       magnetization grids found:

       * 0 grids: ISPIN=1. Returns ``VolumetricData`` with
         ``magnetization=None``.
       * 1 grid: ISPIN=2. Returns ``VolumetricData`` with a scalar
         magnetization density (spin-up minus spin-down).
       * 3 grids: SOC. Returns ``SOCVolumetricData`` with the
         three grids stacked into a ``(NGX, NGY, NGZ, 3)``
         magnetization vector field (mx, my, mz). The scalar
         ``magnetization`` field is set to the mz component for
         backward compatibility.

    Parameters
    ----------
    filename : str, optional
        Path to CHGCAR file. Default is ``"CHGCAR"``.

    Returns
    -------
    volumetric : VolumetricData or SOCVolumetricData
        ``VolumetricData`` for ISPIN=1 or ISPIN=2 files.
        ``SOCVolumetricData`` for SOC files (4 grid blocks).

    Raises
    ------
    ValueError
        If the lattice volume is zero or if the grid dimensions cannot
        be located or the data block is truncated.

    Notes
    -----
    All returned densities are in units of electrons / Angstrom^3 (the
    raw VASP values, which encode charge * volume, are divided by the
    cell volume). Coordinates are stored in fractional form after
    conversion from Cartesian if necessary. The grid arrays use
    Fortran (column-major) ordering to match the VASP write convention.
    """
    path: Path = Path(filename)
    with path.open("r") as fid:
        lattice: np.ndarray
        coords: np.ndarray
        symbols: tuple[str, ...]
        atom_counts: list[int]
        lattice, coords, symbols, atom_counts = _read_poscar_header(fid)
        rest_lines: list[str] = [line.rstrip("\n") for line in fid]

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
    grid_shape: tuple[int, int, int]
    first_grid_idx, grid_shape = _find_next_grid_line(rest_lines, 0)
    if first_grid_idx is None:
        msg: str = "Could not locate CHGCAR charge-density grid dimensions."
        raise ValueError(msg)

    ngrid: int = int(np.prod(np.asarray(grid_shape, dtype=np.int64)))
    charge_vals: np.ndarray
    end_idx: int
    charge_vals, end_idx = _parse_float_block(
        rest_lines,
        first_grid_idx + 1,
        ngrid,
    )
    charge_grid: np.ndarray = (
        charge_vals.reshape(grid_shape, order="F") / volume
    )

    # Read all remaining grid blocks (up to 3 for SOC: mx, my, mz)
    mag_grids: list[np.ndarray] = []
    scan_idx: int = end_idx
    while len(mag_grids) < _N_SOC_MAG_BLOCKS:
        next_idx: Optional[int]
        next_shape: tuple[int, int, int]
        next_idx, next_shape = _find_next_grid_line(rest_lines, scan_idx)
        if next_idx is None:
            break
        ngrid_mag: int = int(np.prod(np.asarray(next_shape, dtype=np.int64)))
        mag_vals: np.ndarray
        mag_vals, scan_idx = _parse_float_block(
            rest_lines,
            next_idx + 1,
            ngrid_mag,
        )
        mag_grids.append(mag_vals.reshape(next_shape, order="F") / volume)

    lattice_arr: jnp.ndarray = jnp.asarray(lattice, dtype=jnp.float64)
    coords_arr: jnp.ndarray = jnp.asarray(coords, dtype=jnp.float64)
    charge_arr: jnp.ndarray = jnp.asarray(charge_grid, dtype=jnp.float64)
    counts_arr: jnp.ndarray = jnp.asarray(atom_counts, dtype=jnp.int32)

    if len(mag_grids) == _N_SOC_MAG_BLOCKS:
        # SOC: blocks are mx, my, mz
        mag_vector: np.ndarray = np.stack(mag_grids, axis=-1)
        result_soc: SOCVolumetricData = make_soc_volumetric_data(
            lattice=lattice_arr,
            coords=coords_arr,
            charge=charge_arr,
            magnetization=jnp.asarray(mag_grids[2], dtype=jnp.float64),
            magnetization_vector=jnp.asarray(mag_vector, dtype=jnp.float64),
            grid_shape=grid_shape,
            symbols=symbols,
            atom_counts=counts_arr,
        )
        return result_soc

    result: VolumetricData = make_volumetric_data(
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
    return result


def _read_poscar_header(
    fid,  # noqa: ANN001
) -> tuple[np.ndarray, np.ndarray, tuple[str, ...], list[int]]:
    """Read the POSCAR-like header section at the start of a CHGCAR file.

    Extended Summary
    ----------------
    The first section of a CHGCAR file is identical to a POSCAR file:
    comment, scaling factor, 3x3 lattice matrix, optional species names,
    atom counts, optional selective-dynamics flag, coordinate type, and
    atomic positions. This helper reads all of those fields from the
    already-opened file handle and returns them as NumPy arrays / Python
    containers.

    Implementation Logic
    --------------------
    1. Read and discard the comment line.
    2. Read the universal scaling factor (float).
    3. Read three lines of three floats each for the lattice vectors and
       multiply by the scaling factor.
    4. Read the next line. If it contains no digits, treat it as the
       VASP-5 element-symbol line and advance to the following line for
       atom counts. Otherwise parse atom counts directly.
    5. Compute total atom count as the sum of per-species counts.
    6. Read the coordinate-type line. If it starts with ``'s'`` or
       ``'S'``, selective dynamics is present; consume it and read the
       next line for the actual coordinate type.
    7. Determine whether coordinates are Cartesian (``'c'``/``'k'``)
       or direct (fractional).
    8. Read ``natoms`` coordinate lines of three floats each.
    9. If Cartesian, scale by the scaling factor and convert to
       fractional via ``np.linalg.solve(lattice.T, coords.T).T``.

    Parameters
    ----------
    fid : file-like
        Open file handle positioned at the start of the CHGCAR file.

    Returns
    -------
    lattice : np.ndarray
        Scaled lattice vectors, shape ``(3, 3)``.
    coords : np.ndarray
        Fractional atomic coordinates, shape ``(natoms, 3)``.
    symbols : tuple[str, ...]
        Element symbols (empty tuple for VASP-4 style files).
    atom_counts : list[int]
        Number of atoms per species.

    Raises
    ------
    ValueError
        If a lattice line has fewer than 3 components or a coordinate
        line has fewer than 3 components.
    """
    _comment: str = fid.readline().strip()
    scale: float = float(fid.readline().strip())

    lattice: np.ndarray = np.zeros(
        (_LATTICE_ROWS, _XYZ_COMPONENTS),
        dtype=np.float64,
    )
    for row in range(_LATTICE_ROWS):
        vals: list[float] = [float(x) for x in fid.readline().split()]
        if len(vals) < _XYZ_COMPONENTS:
            msg: str = "Invalid CHGCAR lattice line."
            raise ValueError(msg)
        lattice[row, :] = vals[:_XYZ_COMPONENTS]
    lattice = lattice * scale

    line: str = fid.readline().strip()
    symbols: tuple[str, ...] = ()
    if line and not any(char.isdigit() for char in line):
        symbols = tuple(line.split())
        line = fid.readline().strip()
    atom_counts: list[int] = [int(x) for x in line.split()]
    natoms: int = sum(atom_counts)

    coord_line: str = fid.readline().strip()
    if coord_line and coord_line[0].lower() == "s":
        coord_line = fid.readline().strip()
    cartesian: bool = bool(coord_line) and coord_line[0].lower() in ("c", "k")

    coords: np.ndarray = np.zeros((natoms, _XYZ_COMPONENTS), dtype=np.float64)
    for atom_idx in range(natoms):
        vals = [float(x) for x in fid.readline().split()[:_XYZ_COMPONENTS]]
        if len(vals) < _XYZ_COMPONENTS:
            msg: str = "Invalid CHGCAR coordinate line."
            raise ValueError(msg)
        coords[atom_idx, :] = vals

    if cartesian:
        coords = coords * scale
        coords = np.linalg.solve(lattice.T, coords.T).T

    return lattice, coords, symbols, atom_counts


def _find_next_grid_line(
    lines: list[str],
    start_idx: int,
) -> Tuple[Optional[int], Tuple[int, int, int]]:
    """Find the next line containing exactly three positive integers.

    Extended Summary
    ----------------
    Scans forward through ``lines`` starting at ``start_idx`` to locate
    the next FFT grid-dimension header. In CHGCAR files, each
    volumetric data block is preceded by a single line of three positive
    integers ``NGX NGY NGZ``.

    Implementation Logic
    --------------------
    1. Iterate from ``start_idx`` to the end of ``lines``.
    2. Skip blank lines and lines that do not split into exactly 3
       tokens.
    3. Attempt to parse all three tokens as integers. If any token
       fails to convert, skip the line.
    4. If all three integers are positive, return the line index and the
       ``(NGX, NGY, NGZ)`` tuple.
    5. If no matching line is found, return ``(None, (0, 0, 0))``.

    Parameters
    ----------
    lines : list[str]
        All remaining lines of the CHGCAR file (after the POSCAR
        header has been consumed).
    start_idx : int
        Index within ``lines`` at which to begin scanning.

    Returns
    -------
    idx : int or None
        Line index of the grid header, or ``None`` if not found.
    grid_shape : tuple[int, int, int]
        ``(NGX, NGY, NGZ)`` grid dimensions, or ``(0, 0, 0)`` if
        not found.
    """
    for idx in range(start_idx, len(lines)):
        stripped: str = lines[idx].strip()
        if not stripped:
            continue
        parts: list[str] = stripped.split()
        if len(parts) != _SCALAR_LINE_COMPONENTS:
            continue
        try:
            values: tuple[int, int, int] = (
                int(parts[0]),
                int(parts[1]),
                int(parts[2]),
            )
        except ValueError:
            continue
        if values[0] > 0 and values[1] > 0 and values[2] > 0:
            return idx, values
    return None, (0, 0, 0)


def _parse_float_block(
    lines: list[str],
    start_idx: int,
    nvals: int,
) -> tuple[np.ndarray, int]:
    """Parse ``nvals`` whitespace-separated floats starting at ``start_idx``.

    Extended Summary
    ----------------
    Reads a contiguous block of floating-point values spread across
    multiple lines in the CHGCAR file. VASP writes volumetric data in
    rows of typically 5 or 10 values per line, but the exact count can
    vary. This function consumes lines until exactly ``nvals`` values
    have been collected.

    Implementation Logic
    --------------------
    1. Initialize an empty collection and a running line index.
    2. For each line starting at ``start_idx``:

       a. Skip blank lines.
       b. Attempt to parse every whitespace-separated token as a float.
       c. If any token fails to parse, treat the entire line as
          non-data (e.g., an augmentation-charge header) and stop
          consuming from it.
       d. Append parsed floats to the collection, taking at most as
          many as needed to reach ``nvals``.

    3. After the loop, verify that exactly ``nvals`` values were
       collected. Raise ``ValueError`` on a short read.

    Parameters
    ----------
    lines : list[str]
        All remaining lines of the CHGCAR file.
    start_idx : int
        Index within ``lines`` at which to begin reading floats.
    nvals : int
        Total number of floats to collect.

    Returns
    -------
    values : np.ndarray
        1D array of shape ``(nvals,)`` with dtype ``float64``.
    end_idx : int
        Index of the first line *after* the last consumed line,
        suitable for passing as ``start_idx`` to a subsequent call.

    Raises
    ------
    ValueError
        If the end of ``lines`` is reached before ``nvals`` floats
        have been collected.
    """
    values: list[float] = []
    idx: int = start_idx

    while idx < len(lines) and len(values) < nvals:
        stripped: str = lines[idx].strip()
        if not stripped:
            idx += 1
            continue

        parts: list[str] = stripped.split()
        row_vals: list[float] = []
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

    return np.asarray(values, dtype=np.float64), idx


__all__: list[str] = [
    "read_chgcar",
]
