"""VASP DOSCAR file parser.

Extended Summary
----------------
Reads VASP DOSCAR files and returns a
:class:`~arpyes.types.DensityOfStates` PyTree with energy axis,
total density of states, and Fermi energy.

Routine Listings
----------------
:func:`read_doscar`
    Parse a VASP DOSCAR file into a DensityOfStates.

Notes
-----
Handles both spin-polarized and non-polarized DOSCAR formats.
The Fermi level is extracted directly from the file header.
"""

from pathlib import Path

import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float

from arpyes.types import DensityOfStates, make_density_of_states


def read_doscar(
    filename: str = "DOSCAR",
) -> DensityOfStates:
    """Parse a VASP DOSCAR file.

    Reads a VASP DOSCAR file that contains the total (and optionally
    site-projected) density of states on a uniform energy grid. The
    DOSCAR format has a 6-line header followed by ``NEDOS`` data lines
    for the total DOS, and optionally ``NATOMS`` additional blocks for
    site-projected DOS. This function extracts only the total DOS and
    returns a :class:`~arpyes.types.DensityOfStates` PyTree.

    Implementation Logic
    --------------------
    1. **Read header line 1** -- extract ``_natoms`` from the first
       integer on line 1 (number of atoms in the cell).

    2. **Skip lines 2-5** -- consume four lines containing system
       title and INCAR metadata that are not needed.

    3. **Read metadata line (line 6)** -- parse floats: EMIN, EMAX,
       ``nedos`` (number of energy grid points, cast to int), and
       ``efermi`` (Fermi energy in eV).

    4. **Detect column count** -- read the first data line and count
       the number of whitespace-separated tokens (``ncols``). For
       non-spin-polarized calculations ncols = 3 (energy, DOS,
       integrated DOS); for spin-polarized ncols = 5 (energy,
       DOS-up, DOS-down, integrated-up, integrated-down).

    5. **Parse total-DOS block** -- allocate an array of shape
       ``(nedos, ncols)`` and fill it row-by-row. The first data line
       (already read in step 4) is stored as row 0.

    6. **Extract columns** -- column 0 is the energy axis, column 1
       is the total DOS (spin-up for spin-polarized).

    7. **Construct PyTree** -- call ``make_density_of_states`` with
       the energy array, total DOS array, and Fermi energy.

    Parameters
    ----------
    filename : str, optional
        Path to DOSCAR file. Default is ``"DOSCAR"``.

    Returns
    -------
    dos : DensityOfStates
        Density of states with energy axis and Fermi level.

    Notes
    -----
    For spin-polarized calculations (ISPIN=2), the DOSCAR contains
    separate spin-up and spin-down columns. This parser currently
    returns only column 1 (spin-up DOS) as the ``total_dos`` field.
    To obtain the full spin-resolved DOS, the raw data array would
    need to be returned directly. Site-projected DOS blocks
    (one per atom, appearing after the total-DOS block) are not
    parsed.
    """
    path: Path = Path(filename)
    with path.open("r") as fid:
        header: list[str] = fid.readline().split()
        _natoms: int = int(header[0])
        fid.readline()
        fid.readline()
        fid.readline()
        fid.readline()
        meta: list[float] = [
            float(x) for x in fid.readline().split()
        ]
        nedos: int = int(meta[2])
        efermi: float = meta[3]
        first_line: str = fid.readline()
        first_vals: list[float] = [
            float(x) for x in first_line.split()
        ]
        ncols: int = len(first_vals)
        data: np.ndarray = np.zeros(
            (nedos, ncols), dtype=np.float64
        )
        data[0, :] = first_vals
        for i in range(1, nedos):
            vals: list[float] = [
                float(x) for x in fid.readline().split()
            ]
            data[i, :] = vals
    energy: Float[Array, " E"] = jnp.asarray(data[:, 0], dtype=jnp.float64)
    total_dos: Float[Array, " E"] = jnp.asarray(data[:, 1], dtype=jnp.float64)
    dos: DensityOfStates = make_density_of_states(
        energy=energy,
        total_dos=total_dos,
        fermi_energy=efermi,
    )
    return dos


__all__: list[str] = [
    "read_doscar",
]
