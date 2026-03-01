"""VASP DOSCAR file parser.

Extended Summary
----------------
Reads VASP DOSCAR files and returns a
:class:`~arpyes.types.DensityOfStates` or
:class:`~arpyes.types.FullDensityOfStates` PyTree depending on
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
from beartype.typing import Literal, Union
from jaxtyping import Array, Float

from arpyes.types import (
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
    DOS block.
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
        data: np.ndarray = np.zeros((nedos, ncols), dtype=np.float64)
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
            return make_density_of_states(
                energy=energy,
                total_dos=total_dos,
                fermi_energy=efermi,
            )

        # Full mode: extract all columns
        is_spin: bool = ncols == _SPIN_COLS
        energy_arr = jnp.asarray(data[:, 0], dtype=jnp.float64)
        dos_up_arr = jnp.asarray(data[:, 1], dtype=jnp.float64)
        dos_down_arr = None
        int_up_arr: Float[Array, " E"]
        int_down_arr = None

        if is_spin:
            dos_down_arr = jnp.asarray(data[:, 2], dtype=jnp.float64)
            int_up_arr = jnp.asarray(data[:, 3], dtype=jnp.float64)
            int_down_arr = jnp.asarray(data[:, 4], dtype=jnp.float64)
        else:
            int_up_arr = jnp.asarray(data[:, 2], dtype=jnp.float64)

        # Read PDOS blocks if present
        pdos_arr = None
        pdos_blocks: list[np.ndarray] = []
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
                atom_data: np.ndarray = np.zeros(
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

    return make_full_density_of_states(
        energy=energy_arr,
        total_dos_up=dos_up_arr,
        integrated_dos_up=int_up_arr,
        fermi_energy=efermi,
        total_dos_down=dos_down_arr,
        integrated_dos_down=int_down_arr,
        pdos=pdos_arr,
        natoms=natoms,
    )


__all__: list[str] = [
    "read_doscar",
]
