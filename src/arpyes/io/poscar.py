"""VASP POSCAR file parser.

Extended Summary
----------------
Reads VASP POSCAR/CONTCAR crystal structure files and returns
a :class:`~arpyes.types.CrystalGeometry` PyTree containing lattice
vectors, atomic coordinates, element symbols, and atom counts.

Routine Listings
----------------
:func:`read_poscar`
    Parse a VASP POSCAR file into a CrystalGeometry.

Notes
-----
Handles both direct (fractional) and Cartesian coordinate formats,
optional selective dynamics, and automatic reciprocal lattice
computation.
"""

from pathlib import Path

import numpy as np

from arpyes.types import CrystalGeometry, make_crystal_geometry


def read_poscar(
    filename: str = "POSCAR",
) -> CrystalGeometry:
    """Parse a VASP POSCAR/CONTCAR file.

    Parameters
    ----------
    filename : str, optional
        Path to POSCAR file. Default is ``"POSCAR"``.

    Returns
    -------
    geometry : CrystalGeometry
        Crystal geometry with lattice, coordinates, symbols.
    """
    path: Path = Path(filename)
    with path.open("r") as fid:
        _comment: str = fid.readline().strip()
        scale: float = float(fid.readline().strip())
        lattice: np.ndarray = np.zeros((3, 3), dtype=np.float64)
        for i in range(3):
            vals: list[float] = [
                float(x) for x in fid.readline().split()
            ]
            lattice[i, :] = vals
        lattice = lattice * scale
        line: str = fid.readline().strip()
        symbols: tuple[str, ...] = ()
        if not any(c.isdigit() for c in line):
            symbols = tuple(line.split())
            line = fid.readline().strip()
        atom_counts: list[int] = [
            int(x) for x in line.split()
        ]
        natoms: int = sum(atom_counts)
        line = fid.readline().strip()
        selective: bool = False
        if line[0].lower() == "s":
            selective = True  # noqa: F841
            line = fid.readline().strip()
        cartesian: bool = line[0].lower() in ("c", "k")
        coords: np.ndarray = np.zeros(
            (natoms, 3), dtype=np.float64
        )
        for i in range(natoms):
            vals = [
                float(x)
                for x in fid.readline().split()[:3]
            ]
            coords[i, :] = vals
        if cartesian:
            coords = coords * scale
            coords = np.linalg.solve(
                lattice.T, coords.T
            ).T
    geometry: CrystalGeometry = make_crystal_geometry(
        lattice=lattice,
        coords=coords,
        symbols=symbols,
        atom_counts=atom_counts,
    )
    return geometry


__all__: list[str] = [
    "read_poscar",
]
