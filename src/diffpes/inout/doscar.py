"""VASP DOSCAR file parser.

Extended Summary
----------------
Reads VASP DOSCAR files and returns a
:class:`~diffpes.types.DensityOfStates` or
:class:`~diffpes.types.FullDensityOfStates` PyTree depending on
the ``return_mode`` parameter.

Routine Listings
----------------
:func:`read_doscar`
    Parse a VASP DOSCAR file into a DensityOfStates or
    FullDensityOfStates.

Notes
-----
Handles both spin-polarized (ISPIN=2) and non-polarized (ISPIN=1)
DOSCAR formats. The Fermi level is extracted directly from the
file header.
"""

from pathlib import Path

import jax.numpy as jnp
import numpy as np
from beartype.typing import Literal, Optional, Union
from jaxtyping import Array, Float
from numpy import ndarray as NDArray  # noqa: N812

from diffpes.types import (
    DensityOfStates,
    FullDensityOfStates,
    make_density_of_states,
    make_full_density_of_states,
)

_NONSPIN_COLS: int = 3
_SPIN_COLS: int = 5


def read_doscar(  # noqa: PLR0912, PLR0915
    filename: str = "DOSCAR",
    return_mode: Literal["legacy", "full"] = "legacy",
) -> Union[DensityOfStates, FullDensityOfStates]:
    """Parse a VASP DOSCAR file.

    Reads a VASP DOSCAR file containing total (and optionally
    site-projected) density of states on a uniform energy grid.

    Extended Summary
    ----------------
    The DOSCAR file format written by VASP consists of:

    * **Header** (6 lines):
      - Line 1: system header with ``NATOMS`` as the first integer.
      - Lines 2-5: additional metadata (unused here).
      - Line 6: ``EMIN  EMAX  NEDOS  EFERMI  ...`` -- energy window
        bounds, number of DOS grid points, and the Fermi energy.

    * **Total DOS block** (``NEDOS`` lines): each line contains the
      energy value followed by density-of-states columns.

      - ISPIN=1: 3 columns -- ``energy, DOS_up, intDOS_up``.
      - ISPIN=2: 5 columns -- ``energy, DOS_up, DOS_down, intDOS_up,
        intDOS_down``.

    * **Per-atom PDOS blocks** (optional, ``NATOMS`` blocks of
      ``NEDOS`` lines each): each block begins with a header line
      identical in format to line 6 of the main header, followed by
      orbital-projected DOS values. Column counts vary depending on
      ``LORBIT`` and spin polarization.

    Implementation Logic
    --------------------
    1. **Parse header** -- read the first 6 lines. Extract ``NATOMS``
       from line 1 and ``NEDOS`` / ``EFERMI`` from line 6.

    2. **Read total DOS block** -- peek at the first data line to
       determine the column count (3 for non-spin, 5 for spin).
       Read all ``NEDOS`` rows into a ``(NEDOS, ncols)`` array.

    3. **Legacy mode early return** -- if ``return_mode == "legacy"``,
       extract only ``energy`` (column 0) and ``total_dos`` (column 1,
       i.e. spin-up DOS) and return a ``DensityOfStates``.

    4. **Full mode column extraction** -- split columns into spin-up
       DOS, optional spin-down DOS, and integrated DOS arrays.

    5. **Read PDOS blocks** -- for each of the ``NATOMS`` atoms:

       a. Read the per-atom header line (EMIN/EMAX/NEDOS/EFERMI).
       b. Read the first data line to determine the PDOS column count.
       c. Read the remaining ``NEDOS - 1`` lines.
       d. Strip the energy column (column 0) and store the orbital
          columns.

    6. **Stack PDOS** -- if any PDOS blocks were successfully read,
       stack them into a ``(NATOMS, NEDOS, C)`` array where ``C`` is
       the number of orbital columns.

    7. **Construct return value** -- build a ``FullDensityOfStates``
       PyTree containing all extracted arrays.

    Parameters
    ----------
    filename : str, optional
        Path to DOSCAR file. Default is ``"DOSCAR"``.
    return_mode : {"legacy", "full"}, optional
        ``"legacy"`` (default) returns a ``DensityOfStates`` with
        only spin-up total DOS (backward-compatible). ``"full"``
        returns a ``FullDensityOfStates`` with both spin channels,
        integrated DOS, and PDOS blocks when present.

    Returns
    -------
    dos : DensityOfStates or FullDensityOfStates
        Density of states data.

    Notes
    -----
    In ``"full"`` mode the parser also reads per-atom PDOS blocks
    that follow the total DOS section. Each atom's PDOS block has
    the same number of energy grid points (``NEDOS``) as the total
    DOS block. The PDOS orbital ordering follows the VASP convention
    determined by ``LORBIT`` (e.g. ``s, p_y, p_z, p_x, d_{xy}, ...``
    for ``LORBIT=11``). The Fermi energy stored in the returned
    object is taken directly from the file header (line 6, column 4).
    """
    path: Path = Path(filename)
    with path.open("r") as fid:
        header: list[str] = fid.readline().split()
        natoms: int = int(header[0])
        fid.readline()
        fid.readline()
        fid.readline()
        fid.readline()
        meta: list[float] = [float(x) for x in fid.readline().split()]
        nedos: int = int(meta[2])
        efermi: float = meta[3]

        # Read total DOS block
        first_line: str = fid.readline()
        first_vals: list[float] = [float(x) for x in first_line.split()]
        ncols: int = len(first_vals)
        data: Float[NDArray, "E C"] = np.zeros(
            (nedos, ncols), dtype=np.float64
        )
        data[0, :] = first_vals
        for i in range(1, nedos):
            vals: list[float] = [float(x) for x in fid.readline().split()]
            data[i, :] = vals

        if return_mode == "legacy":
            energy: Float[Array, " E"] = jnp.asarray(
                data[:, 0], dtype=jnp.float64
            )
            total_dos: Float[Array, " E"] = jnp.asarray(
                data[:, 1], dtype=jnp.float64
            )
            result_legacy: DensityOfStates = make_density_of_states(
                energy=energy,
                total_dos=total_dos,
                fermi_energy=efermi,
            )
            return result_legacy

        # Full mode: extract all columns
        is_spin: bool = ncols == _SPIN_COLS
        energy_arr: Float[Array, " E"] = jnp.asarray(data[:, 0], dtype=jnp.float64)
        dos_up_arr: Float[Array, " E"] = jnp.asarray(data[:, 1], dtype=jnp.float64)
        dos_down_arr: Optional[Float[Array, " E"]] = None
        int_up_arr: Float[Array, " E"]
        int_down_arr: Optional[Float[Array, " E"]] = None

        if is_spin:
            dos_down_arr = jnp.asarray(data[:, 2], dtype=jnp.float64)
            int_up_arr = jnp.asarray(data[:, 3], dtype=jnp.float64)
            int_down_arr = jnp.asarray(data[:, 4], dtype=jnp.float64)
        else:
            int_up_arr = jnp.asarray(data[:, 2], dtype=jnp.float64)

        # Read PDOS blocks if present
        pdos_arr: Optional[Float[Array, "A E C"]] = None
        pdos_blocks: list[Float[NDArray, "E C"]] = []
        for _atom in range(natoms):
            # Each PDOS block may have a header line
            # repeating EMIN EMAX NEDOS EFERMI
            # or just start with data lines
            line: str = fid.readline()
            if not line or not line.strip():
                break
            # Check if this is a PDOS header (same format as total DOS header)
            line_vals: list[float] = [float(x) for x in line.split()]
            if _NONSPIN_COLS <= len(line_vals) <= _SPIN_COLS:
                # Could be either a header or short PDOS line
                # PDOS header has same EMIN EMAX NEDOS EFERMI format
                # We detect by checking if first value matches energy range
                # Actually, DOSCAR PDOS blocks always have a header line
                pdos_ncols_check: str = fid.readline()
                if not pdos_ncols_check.strip():
                    break
                pdos_first: list[float] = [
                    float(x) for x in pdos_ncols_check.split()
                ]
                pdos_ncols: int = len(pdos_first)
                atom_data: Float[NDArray, "E C"] = np.zeros(
                    (nedos, pdos_ncols), dtype=np.float64
                )
                atom_data[0, :] = pdos_first
                for j in range(1, nedos):
                    row_line: str = fid.readline()
                    if not row_line.strip():
                        break
                    atom_data[j, :] = [float(x) for x in row_line.split()]
                # Store only the orbital columns (skip energy column 0)
                pdos_blocks.append(atom_data[:, 1:])
            else:
                # This line is the first PDOS data line (no header)
                pdos_ncols = len(line_vals)
                atom_data = np.zeros((nedos, pdos_ncols), dtype=np.float64)
                atom_data[0, :] = line_vals
                for j in range(1, nedos):
                    row_line = fid.readline()
                    if not row_line.strip():
                        break
                    atom_data[j, :] = [float(x) for x in row_line.split()]
                pdos_blocks.append(atom_data[:, 1:])

        if pdos_blocks:
            # Stack into (A, E, C) array
            pdos_arr = jnp.asarray(
                np.stack(pdos_blocks, axis=0), dtype=jnp.float64
            )

    result_full: FullDensityOfStates = make_full_density_of_states(
        energy=energy_arr,
        total_dos_up=dos_up_arr,
        integrated_dos_up=int_up_arr,
        fermi_energy=efermi,
        total_dos_down=dos_down_arr,
        integrated_dos_down=int_down_arr,
        pdos=pdos_arr,
        natoms=natoms,
    )
    return result_full


__all__: list[str] = [
    "read_doscar",
]
