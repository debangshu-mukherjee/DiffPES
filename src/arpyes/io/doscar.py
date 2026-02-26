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

import numpy as np

from arpyes.types import DensityOfStates, make_density_of_states


def read_doscar(
    filename: str = "DOSCAR",
) -> DensityOfStates:
    """Parse a VASP DOSCAR file.

    Parameters
    ----------
    filename : str, optional
        Path to DOSCAR file. Default is ``"DOSCAR"``.

    Returns
    -------
    dos : DensityOfStates
        Density of states with energy axis and Fermi level.
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
    energy: np.ndarray = data[:, 0]
    total_dos: np.ndarray = data[:, 1]
    dos: DensityOfStates = make_density_of_states(
        energy=energy,
        total_dos=total_dos,
        fermi_energy=efermi,
    )
    return dos


__all__: list[str] = [
    "read_doscar",
]
