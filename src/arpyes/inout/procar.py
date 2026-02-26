"""VASP PROCAR file parser.

Extended Summary
----------------
Reads VASP PROCAR files containing orbital-resolved band
projections and returns an
:class:`~arpyes.types.OrbitalProjection` PyTree.

Routine Listings
----------------
:func:`read_procar`
    Parse a VASP PROCAR file into an OrbitalProjection.

Notes
-----
Orbital ordering follows VASP convention:
``[s, py, pz, px, dxy, dyz, dz2, dxz, dx2-y2]``.
"""

import re
from pathlib import Path

import numpy as np

from arpyes.types import OrbitalProjection, make_orbital_projection

_NORBS: int = 9


def read_procar(
    filename: str = "PROCAR",
) -> OrbitalProjection:
    """Parse a VASP PROCAR file.

    Parameters
    ----------
    filename : str, optional
        Path to PROCAR file. Default is ``"PROCAR"``.

    Returns
    -------
    orb_proj : OrbitalProjection
        Orbital projections with shape (K, B, A, 9).
    """
    path: Path = Path(filename)
    with path.open("r") as fid:
        header: str = ""
        while "k-points" not in header:
            header = fid.readline()
        params: list[int] = [
            int(x)
            for x in re.findall(r"\d+", header)
        ]
        nkpts: int = params[0]
        nbands: int = params[1]
        natoms: int = params[2]
        projections: np.ndarray = np.zeros(
            (nkpts, nbands, natoms, _NORBS),
            dtype=np.float64,
        )
        k_re: str = (
            r"k-point\s+(\d+)\s*:\s*"
            r"([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)"
        )
        for line in fid:
            k_match: re.Match[str] | None = re.search(
                k_re, line
            )
            if k_match is None:
                continue
            k_idx: int = int(k_match.group(1)) - 1
            for b in range(nbands):
                band_line: str = ""
                while "band" not in band_line:
                    band_line = fid.readline()
                fid.readline()
                for a in range(natoms):
                    data_line: str = fid.readline()
                    vals: list[float] = [
                        float(x)
                        for x in data_line.split()
                    ]
                    projections[k_idx, b, a, :] = vals[
                        1 : _NORBS + 1
                    ]
                fid.readline()
                fid.readline()
    orb_proj: OrbitalProjection = make_orbital_projection(
        projections=projections,
    )
    return orb_proj


__all__: list[str] = [
    "read_procar",
]
