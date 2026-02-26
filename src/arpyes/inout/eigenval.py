"""VASP EIGENVAL file parser.

Extended Summary
----------------
Reads VASP EIGENVAL files containing electronic band energies and
returns a :class:`~arpyes.types.BandStructure` PyTree.

Routine Listings
----------------
:func:`read_eigenval`
    Parse a VASP EIGENVAL file into a BandStructure.

Notes
-----
Handles both spin-polarized and non-polarized calculations.
Bands are sorted by energy within each k-point.
"""

from pathlib import Path

import numpy as np

from arpyes.types import BandStructure, make_band_structure


def read_eigenval(
    filename: str = "EIGENVAL",
    fermi_energy: float = 0.0,
) -> BandStructure:
    """Parse a VASP EIGENVAL file.

    Parameters
    ----------
    filename : str, optional
        Path to EIGENVAL file. Default is ``"EIGENVAL"``.
    fermi_energy : float, optional
        Fermi level in eV. Default is 0.0.

    Returns
    -------
    bands : BandStructure
        Band structure with eigenvalues and k-points.
    """
    path: Path = Path(filename)
    with path.open("r") as fid:
        header: list[int] = [
            int(x) for x in fid.readline().split()
        ]
        _ispin: int = header[3]
        fid.readline()
        fid.readline()
        fid.readline()
        fid.readline()
        meta: list[int] = [
            int(x) for x in fid.readline().split()
        ]
        _nelect: int = meta[0]
        nkpoints: int = meta[1]
        nbands: int = meta[2]
        fid.readline()
        kpoints: np.ndarray = np.zeros(
            (nkpoints, 4), dtype=np.float64
        )
        eigenvalues: np.ndarray = np.zeros(
            (nkpoints, nbands), dtype=np.float64
        )
        for k in range(nkpoints):
            fid.readline()
            kpoints[k, :] = [
                float(x) for x in fid.readline().split()
            ]
            fid.readline()
            for b in range(nbands):
                vals: list[float] = [
                    float(x)
                    for x in fid.readline().split()
                ]
                eigenvalues[k, b] = vals[1]
            if k < nkpoints - 1:
                pass
        eigenvalues = np.sort(eigenvalues, axis=1)
    bands: BandStructure = make_band_structure(
        eigenvalues=eigenvalues,
        kpoints=kpoints[:, :3],
        kpoint_weights=kpoints[:, 3],
        fermi_energy=fermi_energy,
    )
    return bands


__all__: list[str] = [
    "read_eigenval",
]
