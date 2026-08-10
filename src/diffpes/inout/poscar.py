"""Parse a VASP POSCAR/CONTCAR file.

Extended Summary
----------------
The module reads VASP POSCAR and CONTCAR crystal structure files. It returns
a :class:`~diffpes.types.CrystalGeometry` PyTree containing lattice vectors,
fractional positions, and per-atom species.

Routine Listings
----------------
:func:`read_poscar`
    Parse a VASP POSCAR/CONTCAR file.

Notes
-----
The parser supports direct and Cartesian coordinate formats,
optional selective dynamics, and automatic reciprocal lattice
computation.
"""

from pathlib import Path

import jax.numpy as jnp
import numpy as np
from beartype import beartype
from beartype.typing import TextIO, Tuple
from jaxtyping import Float64, jaxtyped
from numpy.typing import NDArray

from diffpes.types import CrystalGeometry, make_crystal_geometry


@jaxtyped(typechecker=beartype)
def read_poscar(
    filename: str = "POSCAR",
) -> CrystalGeometry:
    """Parse a VASP POSCAR/CONTCAR file.

    The function reads a VASP POSCAR or CONTCAR file. The file defines lattice
    vectors, element symbols, atom counts, and atomic coordinates. The
    coordinates can use direct or Cartesian form. The function returns a
    :class:`~diffpes.types.CrystalGeometry` PyTree with fractional coordinates.
    It applies the universal scaling factor to the lattice. A negative scalar
    specifies the target cell volume, as defined by the VASP format.

    The POSCAR file format used by VASP has the following structure:

    * **Line 1**: Comment / system name (discarded).
    * **Line 2**: Universal scaling factor applied to all lattice
      vectors (and to Cartesian coordinates, if applicable).
    * **Lines 3-5**: Three rows of three floats defining the lattice
      vectors ``a``, ``b``, ``c`` in Angstroms (before scaling).
    * **Line 6**: Either element symbols (VASP >= 5) or atom counts
      (VASP 4). If non-numeric, it is the species line.
    * **Line 6 or 7**: Atom counts per species (list of integers).
    * **Next line**: Optional ``"Selective dynamics"`` flag. If
      present, the following line is the coordinate-type specifier.
    * **Coordinate-type line**: ``"Direct"`` / ``"Fractional"`` for
      fractional coordinates, or ``"Cartesian"`` / ``"Cart"`` for
      Cartesian.
    * **Coordinate lines**: ``natoms`` lines, each with at least three
      floats. The parser ignores additional selective-dynamics columns.

    :see: :class:`~.test_poscar.TestReadPoscar`

    Implementation Logic
    --------------------
    1. **Read and scale the lattice**::

           lattice = lattice * scaling_factor

       A positive value is the direct scale. For a negative value, derive the
       positive scale that gives a cell volume of ``abs(raw_scale)``.

    2. **Convert Cartesian coordinates when required**::

           coords = np.linalg.solve(lattice.T, coords.T).T

       Solving against the scaled lattice produces fractional coordinates.

    3. **Return validated crystal geometry**::

           return geometry

       The result includes the reciprocal lattice and per-atom species.

    Parameters
    ----------
    filename : str, optional
        Path to POSCAR file. Default is ``"POSCAR"``.

    Returns
    -------
    geometry : CrystalGeometry
        Crystal geometry with lattice, reciprocal lattice, fractional
        positions, and per-atom species.

    Raises
    ------
    ValueError
        If a negative scale accompanies a zero-volume raw lattice.

    Notes
    -----
    The function always returns fractional coordinates. For Cartesian input,
    ``np.linalg.solve`` computes
    ``frac = lattice^{-T} @ cart^T`` with numerical stability. The parser
    detects and ignores optional selective-dynamics flags. VASP-4 files can
    omit the element-symbol line. In this case, the function returns an empty
    ``species`` tuple. Supply the species from an external source such as
    POTCAR.
    """
    fid: TextIO
    i: int

    path: Path = Path(filename)
    with path.open("r") as fid:
        _comment: str = fid.readline().strip()
        raw_scale: float = float(fid.readline().strip())
        lattice: Float64[NDArray, "3 3"] = np.zeros((3, 3), dtype=np.float64)
        for i in range(3):
            vals: list[float] = [float(x) for x in fid.readline().split()]
            lattice[i, :] = vals
        scale: float = raw_scale
        if raw_scale < 0.0:
            raw_volume: float = abs(float(np.linalg.det(lattice)))
            if raw_volume == 0.0:
                msg: str = "negative POSCAR scale requires nonzero cell volume"
                raise ValueError(msg)
            scale = (abs(raw_scale) / raw_volume) ** (1.0 / 3.0)
        lattice = lattice * scale
        line: str = fid.readline().strip()
        symbols: Tuple[str, ...] = ()
        if not any(c.isdigit() for c in line):
            symbols = tuple(line.split())
            line = fid.readline().strip()
        atom_counts: list[int] = [int(x) for x in line.split()]
        natoms: int = sum(atom_counts)
        line = fid.readline().strip()
        selective: bool = False
        if line[0].lower() == "s":
            selective = True  # noqa: F841
            line = fid.readline().strip()
        cartesian: bool = line[0].lower() in ("c", "k")
        coords: Float64[NDArray, "N 3"] = np.zeros(
            (natoms, 3), dtype=np.float64
        )
        for i in range(natoms):
            vals = [float(x) for x in fid.readline().split()[:3]]
            coords[i, :] = vals
        if cartesian:
            coords = coords * scale
            coords = np.linalg.solve(lattice.T, coords.T).T
    species: Tuple[str, ...] = ()
    if symbols:
        species = tuple(
            symbol
            for symbol, count in zip(symbols, atom_counts, strict=True)
            for _ in range(count)
        )
    geometry: CrystalGeometry = make_crystal_geometry(
        lattice=jnp.asarray(lattice),
        positions=jnp.asarray(coords),
        species=species,
    )
    return geometry


__all__: list[str] = [
    "read_poscar",
]
