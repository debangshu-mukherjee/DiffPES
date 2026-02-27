"""VASP file parsers for ARPES simulation input.

Extended Summary
----------------
Provides parsers for VASP output files (POSCAR, EIGENVAL, KPOINTS,
DOSCAR, PROCAR) that return PyTree data structures suitable for
ARPES simulations.

Routine Listings
----------------
:func:`apply_kpath_ticks`
    Annotate a plot axis with KPathInfo symmetry labels.
:func:`load_from_h5`
    Load PyTrees from an HDF5 file.
:func:`plot_arpes_spectrum`
    Plot an ARPES map from an ArpesSpectrum PyTree.
:func:`plot_arpes_with_kpath`
    Plot an ARPES map and apply KPathInfo axis annotations.
:func:`read_doscar`
    Parse VASP DOSCAR into DensityOfStates.
:func:`read_eigenval`
    Parse VASP EIGENVAL into BandStructure.
:func:`read_kpoints`
    Parse VASP KPOINTS into KPathInfo.
:func:`read_poscar`
    Parse VASP POSCAR into CrystalGeometry.
:func:`read_procar`
    Parse VASP PROCAR into OrbitalProjection.
:func:`save_to_h5`
    Save one or more named PyTrees to an HDF5 file.

Notes
-----
All parsers use standard Python I/O (not JAX) since file
parsing is inherently sequential. They convert parsed data
to JAX arrays via factory functions.
"""

from .doscar import read_doscar
from .eigenval import read_eigenval
from .hdf5 import load_from_h5, save_to_h5
from .kpoints import read_kpoints
from .plotting import (
    apply_kpath_ticks,
    plot_arpes_spectrum,
    plot_arpes_with_kpath,
)
from .poscar import read_poscar
from .procar import read_procar

__all__: list[str] = [
    "apply_kpath_ticks",
    "load_from_h5",
    "plot_arpes_spectrum",
    "plot_arpes_with_kpath",
    "read_doscar",
    "read_eigenval",
    "read_kpoints",
    "read_poscar",
    "read_procar",
    "save_to_h5",
]
