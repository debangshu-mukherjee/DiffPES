"""VASP file parsers for ARPES simulation input.

Extended Summary
----------------
Provides parsers for VASP output files (POSCAR, EIGENVAL, KPOINTS,
DOSCAR, PROCAR) that return PyTree data structures suitable for
ARPES simulations.

Routine Listings
----------------
:func:`read_poscar`
    Parse VASP POSCAR into CrystalGeometry.
:func:`read_eigenval`
    Parse VASP EIGENVAL into BandStructure.
:func:`read_kpoints`
    Parse VASP KPOINTS into KPathInfo.
:func:`read_doscar`
    Parse VASP DOSCAR into DensityOfStates.
:func:`read_procar`
    Parse VASP PROCAR into OrbitalProjection.

Notes
-----
All parsers use standard Python I/O (not JAX) since file
parsing is inherently sequential. They convert parsed data
to JAX arrays via factory functions.
"""

from .doscar import read_doscar
from .eigenval import read_eigenval
from .kpoints import read_kpoints
from .poscar import read_poscar
from .procar import read_procar

__all__: list[str] = [
    "read_doscar",
    "read_eigenval",
    "read_kpoints",
    "read_poscar",
    "read_procar",
]
