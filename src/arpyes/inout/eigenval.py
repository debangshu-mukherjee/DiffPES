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

    Reads a VASP EIGENVAL file that contains electronic eigenvalues
    (band energies) at each k-point sampled during a self-consistent
    or non-self-consistent calculation. The EIGENVAL format consists
    of a fixed header block followed by repeating per-k-point blocks
    of band energies. This function returns a
    :class:`~arpyes.types.BandStructure` PyTree.

    Implementation Logic
    --------------------
    1. **Read header** -- parse the first line to extract the 4-integer
       header (NIONS, unknown, unknown, ISPIN). The fourth integer
       ``_ispin`` indicates spin-polarization (1 = non-polarized,
       2 = spin-polarized).

    2. **Skip lines 2-5** -- consume four lines that contain the
       system title, a blank line, and POTCAR-related metadata. These
       are not used.

    3. **Read metadata line (line 6)** -- parse three integers:
       ``_nelect`` (number of electrons), ``nkpoints`` (number of
       k-points), and ``nbands`` (number of bands).

    4. **Skip one blank line** following the metadata.

    5. **Parse per-k-point blocks** -- for each of the ``nkpoints``
       k-points:
       a. Skip one blank separator line.
       b. Read the k-point coordinate line: four floats
          (kx, ky, kz, weight) stored in ``kpoints[k, :]``.
       c. Skip one blank line after the k-point header.
       d. Read ``nbands`` eigenvalue lines. Each line has at least
          two columns (band index, energy); store ``vals[1]`` in
          ``eigenvalues[k, b]``.

    6. **Sort eigenvalues** -- sort bands within each k-point by
       energy (ascending) via ``np.sort(eigenvalues, axis=1)`` to
       ensure a consistent band ordering.

    7. **Construct PyTree** -- call ``make_band_structure`` with the
       eigenvalue matrix, k-point coordinates (first 3 columns),
       k-point weights (column 4), and the user-supplied Fermi energy.

    Parameters
    ----------
    filename : str, optional
        Path to EIGENVAL file. Default is ``"EIGENVAL"``.
    fermi_energy : float, optional
        Fermi level in eV used to reference the eigenvalues.
        Default is 0.0.

    Returns
    -------
    bands : BandStructure
        Band structure with eigenvalues and k-points.

    Notes
    -----
    For spin-polarized calculations (ISPIN=2), each eigenvalue line
    contains three columns (index, energy-up, energy-down). Only the
    spin-up energy (``vals[1]``) is currently extracted; spin-down
    channels are silently dropped. The eigenvalue sort in step 6 may
    re-order bands across spin channels. The Fermi energy is **not**
    embedded in the EIGENVAL file itself; it must be obtained
    separately (e.g. from a DOSCAR or OUTCAR) and passed via the
    ``fermi_energy`` parameter.
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
