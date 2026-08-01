"""Parse a VASP DOSCAR file.

Extended Summary
----------------
The module reads VASP DOSCAR files and returns a density-of-states carrier.
Legacy data uses :class:`~diffpes.types.DensityOfStates`.
Full data uses :class:`~diffpes.types.FullDensityOfStates`.
The ``return_mode`` parameter selects the carrier.

Routine Listings
----------------
:func:`read_doscar`
    Parse a VASP DOSCAR file.

Notes
-----
The parser supports spin-polarized and nonpolarized DOSCAR formats. It reads
the Fermi level directly from the file header.
"""

from pathlib import Path

import jax.numpy as jnp
import numpy as np
from beartype import beartype
from beartype.typing import Literal, Optional, TextIO, Union
from jaxtyping import Array, Float, jaxtyped
from numpy.typing import NDArray

from diffpes.types import (
    NONSPIN_COLS,
    SPIN_COLS,
    DensityOfStates,
    FullDensityOfStates,
    make_density_of_states,
    make_full_density_of_states,
)


@jaxtyped(typechecker=beartype)
def read_doscar(  # noqa: PLR0912, PLR0915
    filename: str = "DOSCAR",
    return_mode: Literal["legacy", "full"] = "legacy",
) -> Union[DensityOfStates, FullDensityOfStates]:
    """Parse a VASP DOSCAR file.

    The function reads a VASP DOSCAR file that contains total and optional
    site-projected) density of states on a uniform energy grid.

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

    * **Per-atom PDOS blocks**: Each optional block contains ``NEDOS`` lines.
      The header has the same format as line 6 of the main header.
      Orbital-projected DOS values follow the header. ``LORBIT`` and spin
      polarization determine the column count.

    :see: :class:`~.test_doscar.TestReadDoscar`

    Implementation Logic
    --------------------
    1. **Read the header and total-DOS dimensions**::

           path: Path = Path(filename)
           with path.open("r") as fid:
               header: list[str] = fid.readline().split()
               natoms: int = int(header[0])

       This establishes the atom count before the function allocates the
       total and projected data.

    2. **Allocate and populate the total-DOS table**::

           data: Float[NDArray, "E C"] = np.zeros(
               (nedos, ncols), dtype=np.float64
           )

       This preserves each column until the function knows the return mode.

    3. **Return the selected DOS carrier**::

           return dos

       Both branches bind their validated result to ``dos``.

    Parameters
    ----------
    filename : str, optional
        Path to DOSCAR file. Default is ``"DOSCAR"``.
    return_mode : Literal["legacy", "full"], optional
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
    In ``"full"`` mode, the parser also reads each PDOS block after the total
    DOS section. Each PDOS block has ``NEDOS`` energy points. ``LORBIT``
    determines the VASP orbital order. For example, ``LORBIT=11`` starts with
    ``s, p_y, p_z, p_x, d_{xy}``. The parser reads the Fermi energy from
    column 4 of line 6.
    """
    fid: TextIO
    i: int
    _atom: int
    j: int

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

        dos: DensityOfStates | FullDensityOfStates
        if return_mode == "legacy":
            energy: Float[Array, " E"] = jnp.asarray(
                data[:, 0], dtype=jnp.float64
            )
            total_dos: Float[Array, " E"] = jnp.asarray(
                data[:, 1], dtype=jnp.float64
            )
            dos = make_density_of_states(
                energy=energy,
                total_dos=total_dos,
                fermi_energy=efermi,
            )
            return dos

        is_spin: bool = ncols == SPIN_COLS
        energy_arr: Float[Array, " E"] = jnp.asarray(
            data[:, 0], dtype=jnp.float64
        )
        dos_up_arr: Float[Array, " E"] = jnp.asarray(
            data[:, 1], dtype=jnp.float64
        )
        dos_down_arr: Optional[Float[Array, " E"]] = None
        int_up_arr: Float[Array, " E"]
        int_down_arr: Optional[Float[Array, " E"]] = None

        if is_spin:
            dos_down_arr = jnp.asarray(data[:, 2], dtype=jnp.float64)
            int_up_arr = jnp.asarray(data[:, 3], dtype=jnp.float64)
            int_down_arr = jnp.asarray(data[:, 4], dtype=jnp.float64)
        else:
            int_up_arr = jnp.asarray(data[:, 2], dtype=jnp.float64)

        pdos_arr: Optional[Float[Array, "A E C"]] = None
        pdos_blocks: list[Float[NDArray, "E C"]] = []
        for _atom in range(natoms):
            line: str = fid.readline()
            if not line or not line.strip():
                break
            line_vals: list[float] = [float(x) for x in line.split()]
            if NONSPIN_COLS <= len(line_vals) <= SPIN_COLS:
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
                pdos_blocks.append(atom_data[:, 1:])
            else:
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
            pdos_arr = jnp.asarray(
                np.stack(pdos_blocks, axis=0), dtype=jnp.float64
            )

    dos = make_full_density_of_states(
        energy=energy_arr,
        total_dos_up=dos_up_arr,
        integrated_dos_up=int_up_arr,
        fermi_energy=efermi,
        total_dos_down=dos_down_arr,
        integrated_dos_down=int_down_arr,
        pdos=pdos_arr,
        natoms=natoms,
    )
    return dos


__all__: list[str] = [
    "read_doscar",
]
