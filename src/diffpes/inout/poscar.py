"""VASP POSCAR file parser.

Extended Summary
----------------
Reads VASP POSCAR/CONTCAR crystal structure files and returns
a :class:`~diffpes.types.CrystalGeometry` PyTree containing lattice
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

import jax.numpy as jnp
import numpy as np
from jaxtyping import Float
from numpy import ndarray as NDArray  # noqa: N812

from diffpes.types import CrystalGeometry, make_crystal_geometry


def read_poscar(
    filename: str = "POSCAR",
) -> CrystalGeometry:
    """Parse a VASP POSCAR/CONTCAR file.

    Reads a VASP POSCAR (or CONTCAR) file that defines the crystal
    structure: lattice vectors, element symbols, atom counts, and
    atomic coordinates in either direct (fractional) or Cartesian
    form. The function returns a :class:`~diffpes.types.CrystalGeometry`
    PyTree with coordinates always stored in fractional form and the
    lattice pre-scaled by the universal scaling factor.

    Extended Summary
    ----------------
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
      floats. Additional columns (selective-dynamics T/F flags) are
      silently ignored.

    Implementation Logic
    --------------------
    1. **Read comment line** (line 1) -- consumed and discarded.

    2. **Read scaling factor** (line 2) -- a single float that
       multiplies all lattice vectors.

    3. **Read lattice vectors** (lines 3-5) -- three rows of three
       floats each, stored as a (3, 3) NumPy array and immediately
       multiplied by the scaling factor.

    4. **Detect optional species line** (line 6) -- if the line
       contains no digits it is the VASP-5 style element-symbol line
       (e.g. ``"Si O"``). The symbols are stored and the next line is
       read. If the line contains digits, it is the atom-count line
       (VASP-4 style) and ``symbols`` remains empty.

    5. **Read atom counts** -- parse the current line as a list of
       integers giving the number of atoms per species.

    6. **Detect optional Selective Dynamics line** -- if the next line
       starts with ``"s"`` or ``"S"``, selective-dynamics flags are
       present; consume the line and read the coordinate-type line
       that follows.

    7. **Determine coordinate type** -- if the coordinate-type line
       starts with ``"c"``, ``"C"``, ``"k"``, or ``"K"`` the
       coordinates are Cartesian; otherwise they are direct
       (fractional).

    8. **Read coordinates** -- parse ``natoms`` lines, each with at
       least three floats. Only the first three tokens are used
       (selective-dynamics flags on positions 4-6 are ignored).

    9. **Convert Cartesian to fractional** (if needed) -- scale the
       Cartesian coordinates by the scaling factor, then solve
       ``lattice^T @ frac^T = cart^T`` via ``np.linalg.solve`` to
       obtain fractional coordinates.

    10. **Construct PyTree** -- call ``make_crystal_geometry`` with the
        scaled lattice, fractional coordinates, element symbols, and
        atom counts. The reciprocal lattice is computed automatically
        inside ``make_crystal_geometry`` as ``2 * pi * inv(lattice)^T``.

    Parameters
    ----------
    filename : str, optional
        Path to POSCAR file. Default is ``"POSCAR"``.

    Returns
    -------
    geometry : CrystalGeometry
        Crystal geometry with lattice, reciprocal lattice, fractional
        coordinates, element symbols, and atom counts.

    Raises
    ------
    IndexError
        If the coordinate-type line is empty (malformed file).

    Notes
    -----
    Coordinates are always returned in **fractional** (direct) form,
    regardless of the input format. If the input is Cartesian, the
    conversion ``frac = lattice^{-T} @ cart^T`` is performed via
    ``np.linalg.solve`` for numerical stability. The optional
    selective-dynamics flags (``T T F`` appended to coordinate lines)
    are detected and skipped but not stored. VASP-4 style files that
    lack the element-symbol line will produce an empty ``symbols``
    tuple; the caller should supply symbols from an external source
    (e.g. POTCAR) in that case.
    """
    path: Path = Path(filename)
    with path.open("r") as fid:
        _comment: str = fid.readline().strip()
        scale: float = float(fid.readline().strip())
        lattice: Float[NDArray, "3 3"] = np.zeros(
            (3, 3), dtype=np.float64
        )
        for i in range(3):
            vals: list[float] = [float(x) for x in fid.readline().split()]
            lattice[i, :] = vals
        lattice = lattice * scale
        line: str = fid.readline().strip()
        symbols: tuple[str, ...] = ()
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
        coords: Float[NDArray, "N 3"] = np.zeros(
            (natoms, 3), dtype=np.float64
        )
        for i in range(natoms):
            vals = [float(x) for x in fid.readline().split()[:3]]
            coords[i, :] = vals
        if cartesian:
            coords = coords * scale
            coords = np.linalg.solve(lattice.T, coords.T).T
    geometry: CrystalGeometry = make_crystal_geometry(
        lattice=jnp.asarray(lattice),
        coords=jnp.asarray(coords),
        symbols=symbols,
        atom_counts=atom_counts,
    )
    return geometry


__all__: list[str] = [
    "read_poscar",
]
