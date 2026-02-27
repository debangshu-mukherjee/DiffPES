"""VASP CHGCAR file parser.

Extended Summary
----------------
Reads VASP CHGCAR volumetric files and returns a
:class:`~arpyes.types.VolumetricData` PyTree containing the crystal
geometry, charge density, and optional magnetization density.
"""

from pathlib import Path

import jax.numpy as jnp
import numpy as np
from beartype.typing import Optional, Tuple

from arpyes.types import VolumetricData, make_volumetric_data

_LATTICE_ROWS: int = 3
_XYZ_COMPONENTS: int = 3
_SCALAR_LINE_COMPONENTS: int = 3


def read_chgcar(
    filename: str = "CHGCAR",
) -> VolumetricData:
    """Parse a VASP CHGCAR file.

    Parameters
    ----------
    filename : str, optional
        Path to CHGCAR file. Default is ``"CHGCAR"``.

    Returns
    -------
    volumetric : VolumetricData
        Parsed lattice, coordinates, charge grid, and optional
        magnetization grid.
    """
    path: Path = Path(filename)
    with path.open("r") as fid:
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
        msg = "CHGCAR lattice volume is zero."
        raise ValueError(msg)

    first_grid_idx, grid_shape = _find_next_grid_line(rest_lines, 0)
    if first_grid_idx is None:
        msg = "Could not locate CHGCAR charge-density grid dimensions."
        raise ValueError(msg)

    ngrid: int = int(np.prod(np.asarray(grid_shape, dtype=np.int64)))
    charge_vals, end_idx = _parse_float_block(
        rest_lines,
        first_grid_idx + 1,
        ngrid,
    )
    charge_grid: np.ndarray = (
        charge_vals.reshape(grid_shape, order="F") / volume
    )

    magnetization_grid: Optional[np.ndarray] = None
    second_grid_idx, second_shape = _find_next_grid_line(rest_lines, end_idx)
    if second_grid_idx is not None:
        ngrid_mag: int = int(np.prod(np.asarray(second_shape, dtype=np.int64)))
        mag_vals, _ = _parse_float_block(
            rest_lines,
            second_grid_idx + 1,
            ngrid_mag,
        )
        magnetization_grid = mag_vals.reshape(second_shape, order="F") / volume

    volumetric: VolumetricData = make_volumetric_data(
        lattice=jnp.asarray(lattice, dtype=jnp.float64),
        coords=jnp.asarray(coords, dtype=jnp.float64),
        charge=jnp.asarray(charge_grid, dtype=jnp.float64),
        magnetization=(
            None
            if magnetization_grid is None
            else jnp.asarray(magnetization_grid, dtype=jnp.float64)
        ),
        grid_shape=grid_shape,
        symbols=symbols,
        atom_counts=jnp.asarray(atom_counts, dtype=jnp.int32),
    )
    return volumetric


def _read_poscar_header(
    fid,  # noqa: ANN001
) -> tuple[np.ndarray, np.ndarray, tuple[str, ...], list[int]]:
    """Read POSCAR-like header section at the start of CHGCAR."""
    _comment: str = fid.readline().strip()
    scale: float = float(fid.readline().strip())

    lattice: np.ndarray = np.zeros(
        (_LATTICE_ROWS, _XYZ_COMPONENTS),
        dtype=np.float64,
    )
    for row in range(_LATTICE_ROWS):
        vals: list[float] = [float(x) for x in fid.readline().split()]
        if len(vals) < _XYZ_COMPONENTS:
            msg = "Invalid CHGCAR lattice line."
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
            msg = "Invalid CHGCAR coordinate line."
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
    """Find the next line containing three positive integers."""
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
    """Parse ``nvals`` floats starting at ``start_idx`` across lines."""
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
        msg = "Unexpected end of CHGCAR data block."
        raise ValueError(msg)

    return np.asarray(values, dtype=np.float64), idx


__all__: list[str] = [
    "read_chgcar",
]
