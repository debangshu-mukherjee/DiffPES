"""VASP EIGENVAL file parser.

Extended Summary
----------------
Reads VASP EIGENVAL files containing electronic band energies and
returns a :class:`~arpyes.types.BandStructure` or
:class:`~arpyes.types.SpinBandStructure` PyTree depending on the
``return_mode`` parameter.

Routine Listings
----------------
:func:`read_eigenval`
    Parse a VASP EIGENVAL file into a BandStructure or
    SpinBandStructure.

Notes
-----
Handles both spin-polarized (ISPIN=2) and non-polarized (ISPIN=1)
calculations. Bands are sorted by energy within each k-point.
"""

from pathlib import Path

import jax.numpy as jnp
import numpy as np
from beartype.typing import Literal, Optional, Union

from arpyes.types import (
    BandStructure,
    SpinBandStructure,
    make_band_structure,
    make_spin_band_structure,
)

_ISPIN_SPIN_POLARIZED: int = 2
_KPOINT_LINE_VALUES: int = 4
_BAND_LINE_MIN_VALUES: int = 2
_BAND_LINE_SPIN_VALUES: int = 3
_EIG_UP_INDEX: int = 1
_EIG_DOWN_INDEX: int = 2


def read_eigenval(
    filename: str = "EIGENVAL",
    fermi_energy: float = 0.0,
    return_mode: Literal["legacy", "full"] = "legacy",
) -> Union[BandStructure, SpinBandStructure]:
    """Parse a VASP EIGENVAL file.

    Reads a VASP EIGENVAL file that contains electronic eigenvalues
    (band energies) at each k-point. Supports both ISPIN=1 (non-
    polarized) and ISPIN=2 (spin-polarized) calculations.

    Extended Summary
    ----------------
    The EIGENVAL file format written by VASP consists of:

    * **Header** (6 lines):
      - Line 1: four integers including ``ISPIN`` as the 4th value.
        ``ISPIN=2`` indicates spin-polarized eigenvalues.
      - Lines 2-5: additional metadata (unused here).
      - Line 6: ``NELECT  NKPOINTS  NBANDS`` -- number of electrons,
        k-points, and bands.

    * **K-point / band blocks** (``NKPOINTS`` blocks): each block
      starts with a k-point line of 4 floats (``kx ky kz weight``),
      followed by ``NBANDS`` eigenvalue lines.

      - ISPIN=1: each band line has 2 values -- ``band_index energy``.
      - ISPIN=2: each band line has 3 values --
        ``band_index energy_up energy_down``.

    Implementation Logic
    --------------------
    1. **Parse header** -- read the first 6 lines. Extract ``ISPIN``
       from line 1 (4th integer) and ``NKPOINTS`` / ``NBANDS`` from
       line 6.

    2. **Allocate arrays** -- create ``(NKPOINTS, 4)`` for k-point
       coordinates+weights and ``(NKPOINTS, NBANDS)`` for spin-up
       eigenvalues. If ``ISPIN=2``, allocate a matching spin-down
       array.

    3. **Read k-point blocks** -- for each k-point:

       a. Read the next non-empty line and parse 4 floats for the
          k-point coordinates and weight.
       b. For each band, read the next non-empty line and extract the
          eigenvalue(s). For spin-polarized files, store both spin-up
          (column 1) and spin-down (column 2).

    4. **Sort eigenvalues** -- sort bands by energy within each
       k-point (ascending) for both spin channels. This ensures
       consistent band ordering even if VASP wrote them unordered.

    5. **Construct return value**:

       * ``return_mode="legacy"`` or ISPIN=1: return a
         ``BandStructure`` with only spin-up eigenvalues, 3D k-point
         coordinates, and k-point weights.
       * ``return_mode="full"`` with ISPIN=2: return a
         ``SpinBandStructure`` containing both spin channels.

    Parameters
    ----------
    filename : str, optional
        Path to EIGENVAL file. Default is ``"EIGENVAL"``.
    fermi_energy : float, optional
        Fermi level in eV used to reference the eigenvalues.
        Default is 0.0.
    return_mode : {"legacy", "full"}, optional
        ``"legacy"`` (default) returns a ``BandStructure`` with only
        spin-up eigenvalues (backward-compatible). ``"full"`` returns
        a ``SpinBandStructure`` with both spin channels when ISPIN=2,
        or a ``BandStructure`` when ISPIN=1.

    Returns
    -------
    bands : BandStructure or SpinBandStructure
        Band structure with eigenvalues and k-points. The type
        depends on ``return_mode`` and the spin polarization.

    Raises
    ------
    ValueError
        If a k-point line has fewer than 4 values, or a band line
        has fewer values than expected for the given ISPIN, or if
        EOF is encountered before all blocks are read.

    Notes
    -----
    For spin-polarized calculations (ISPIN=2), each eigenvalue line
    contains three columns (index, energy-up, energy-down). In
    ``"legacy"`` mode only the spin-up energy is extracted. In
    ``"full"`` mode both channels are preserved in a
    ``SpinBandStructure``. The Fermi energy is **not** embedded in
    the EIGENVAL file; it must be obtained separately (e.g. from
    DOSCAR or OUTCAR). Eigenvalues are stored as-is from the file
    (not shifted by ``fermi_energy``); the Fermi energy is carried
    alongside the eigenvalues in the returned PyTree for downstream
    consumers to apply the shift if needed.
    """
    path: Path = Path(filename)
    with path.open("r") as fid:
        header: list[int] = [int(x) for x in fid.readline().split()]
        ispin: int = header[3]
        fid.readline()
        fid.readline()
        fid.readline()
        fid.readline()
        meta: list[int] = [int(x) for x in fid.readline().split()]
        _nelect: int = meta[0]
        nkpoints: int = meta[1]
        nbands: int = meta[2]
        kpoints: np.ndarray = np.zeros((nkpoints, 4), dtype=np.float64)
        eigenvalues_up: np.ndarray = np.zeros(
            (nkpoints, nbands), dtype=np.float64
        )
        eigenvalues_down: Optional[np.ndarray] = None
        if ispin == _ISPIN_SPIN_POLARIZED:
            eigenvalues_down = np.zeros((nkpoints, nbands), dtype=np.float64)
        for k in range(nkpoints):
            kpoint_line: str = _read_next_nonempty_line(fid)
            if not kpoint_line:
                msg = "Unexpected EOF while reading EIGENVAL k-point block."
                raise ValueError(msg)
            kpoint_vals: list[float] = [float(x) for x in kpoint_line.split()]
            if len(kpoint_vals) < _KPOINT_LINE_VALUES:
                msg = "Invalid EIGENVAL k-point line; expected 4 values."
                raise ValueError(msg)
            kpoints[k, :] = kpoint_vals[:_KPOINT_LINE_VALUES]
            for b in range(nbands):
                band_line: str = _read_next_nonempty_line(fid)
                if not band_line:
                    msg = "Unexpected EOF while reading EIGENVAL band line."
                    raise ValueError(msg)
                vals: list[float] = [float(x) for x in band_line.split()]
                if len(vals) < _BAND_LINE_MIN_VALUES:
                    msg = "Invalid EIGENVAL band line; expected band energy."
                    raise ValueError(msg)
                eigenvalues_up[k, b] = vals[_EIG_UP_INDEX]
                if (
                    ispin == _ISPIN_SPIN_POLARIZED
                    and eigenvalues_down is not None
                ):
                    if len(vals) < _BAND_LINE_SPIN_VALUES:
                        msg = (
                            "Invalid spin-polarized EIGENVAL band line; "
                            "expected spin-down energy."
                        )
                        raise ValueError(msg)
                    eigenvalues_down[k, b] = vals[_EIG_DOWN_INDEX]
        eigenvalues_up = np.sort(eigenvalues_up, axis=1)
        if eigenvalues_down is not None:
            eigenvalues_down = np.sort(eigenvalues_down, axis=1)

    if (
        return_mode == "full"
        and ispin == _ISPIN_SPIN_POLARIZED
        and eigenvalues_down is not None
    ):
        return make_spin_band_structure(
            eigenvalues_up=jnp.asarray(eigenvalues_up),
            eigenvalues_down=jnp.asarray(eigenvalues_down),
            kpoints=jnp.asarray(kpoints[:, :3]),
            kpoint_weights=jnp.asarray(kpoints[:, 3]),
            fermi_energy=fermi_energy,
        )
    bands: BandStructure = make_band_structure(
        eigenvalues=jnp.asarray(eigenvalues_up),
        kpoints=jnp.asarray(kpoints[:, :3]),
        kpoint_weights=jnp.asarray(kpoints[:, 3]),
        fermi_energy=fermi_energy,
    )
    return bands


def _read_next_nonempty_line(fid) -> str:  # noqa: ANN001
    """Read and return the next non-empty line, or ``""`` at EOF.

    Extended Summary
    ----------------
    Helper that skips blank lines in the EIGENVAL file. VASP separates
    k-point blocks with empty lines; this function transparently
    consumes them so that the main parsing loop does not need to handle
    blank-line logic.

    Implementation Logic
    --------------------
    1. Call ``fid.readline()`` in a loop.
    2. If the returned string is falsy (empty string ``""``), EOF has
       been reached -- return ``""``.
    3. If the stripped line is non-empty, return the original
       (unstripped) line.
    4. Otherwise continue to the next line.

    Parameters
    ----------
    fid : file-like
        Open file handle positioned somewhere within the EIGENVAL file.

    Returns
    -------
    line : str
        The next non-empty line (with trailing newline), or ``""`` if
        the end of the file is reached.
    """
    while True:
        line: str = fid.readline()
        if not line:
            return ""
        if line.strip():
            return line


__all__: list[str] = [
    "read_eigenval",
]
