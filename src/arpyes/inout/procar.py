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

import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float

from arpyes.types import OrbitalProjection, make_orbital_projection

_NORBS: int = 9


def read_procar(
    filename: str = "PROCAR",
) -> OrbitalProjection:
    r"""Parse a VASP PROCAR file.

    Reads a VASP PROCAR file that contains the orbital-resolved
    projections of Kohn-Sham wave functions onto site-centred
    spherical harmonics. The PROCAR file is generated when
    ``LORBIT >= 10`` and provides a 4-D array of projection weights
    indexed by (k-point, band, atom, orbital). This function returns
    an :class:`~arpyes.types.OrbitalProjection` PyTree.

    Implementation Logic
    --------------------
    1. **Locate header line** -- scan forward until a line containing
       ``"k-points"`` is found. This header line has the format
       ``"# of k-points:  K  # of bands:  B  # of ions:  A"`` and
       is parsed with ``re.findall(r"\\d+", header)`` to extract
       the three integers ``nkpts``, ``nbands``, ``natoms``.

    2. **Allocate output array** -- create a zero-filled NumPy array
       of shape ``(nkpts, nbands, natoms, 9)`` with dtype float64.

    3. **Iterate over k-point blocks** -- scan the remaining lines
       for k-point headers matching the regex
       ``r"k-point\\s+(\\d+)\\s*:\\s*([-\\d.]+)\\s+([-\\d.]+)\\s+([-\\d.]+)"``.
       The first capture group gives the 1-based k-point index
       (converted to 0-based).

    4. **Iterate over band blocks** -- for each k-point, loop
       ``nbands`` times. Within each iteration, scan forward until
       a line containing ``"band"`` is found (the band header), then
       skip one orbital-name header line.

    5. **Read per-atom projection rows** -- read ``natoms`` lines.
       Each line has the format ``"atom_idx  s  py  pz  px  dxy  dyz
       dz2  dxz  dx2  tot"``. Parse all tokens as floats and store
       columns 1 through 9 (the 9 orbital weights, excluding the
       atom index at position 0 and the total at position 10) into
       ``projections[k, b, a, :]``.

    6. **Skip footer lines** -- after each band's atom block,
       consume two lines (the ``"tot"`` summation line and a blank
       separator).

    7. **Construct PyTree** -- call ``make_orbital_projection`` with
       the filled projection array.

    Parameters
    ----------
    filename : str, optional
        Path to PROCAR file. Default is ``"PROCAR"``.

    Returns
    -------
    orb_proj : OrbitalProjection
        Orbital projections with shape ``(K, B, A, 9)`` where K is
        the number of k-points, B the number of bands, A the number
        of atoms, and 9 the number of orbital channels.

    Notes
    -----
    Orbital ordering follows the VASP convention:
    ``[s, py, pz, px, dxy, dyz, dz2, dxz, dx2-y2]``. This differs
    from the standard real-spherical-harmonic ordering used by some
    other DFT codes. For spin-polarized calculations (ISPIN=2) the
    PROCAR file contains two consecutive sets of k-point blocks (one
    per spin channel). The current parser does **not** separate spin
    channels; it will overwrite the first spin's data with the
    second's because both share the same k-point indices.
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
    proj_arr: Float[Array, " K B A 9"] = jnp.asarray(
        projections, dtype=jnp.float64)
    orb_proj: OrbitalProjection = make_orbital_projection(
        projections=proj_arr,
    )
    return orb_proj


__all__: list[str] = [
    "read_procar",
]
