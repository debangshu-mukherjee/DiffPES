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
from typing import Literal

import jax.numpy as jnp
import numpy as np

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
) -> BandStructure | SpinBandStructure:
    """Parse a VASP EIGENVAL file.

    Reads a VASP EIGENVAL file that contains electronic eigenvalues
    (band energies) at each k-point. Supports both ISPIN=1 (non-
    polarized) and ISPIN=2 (spin-polarized) calculations.

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

    Notes
    -----
    For spin-polarized calculations (ISPIN=2), each eigenvalue line
    contains three columns (index, energy-up, energy-down). In
    ``"legacy"`` mode only the spin-up energy is extracted. In
    ``"full"`` mode both channels are preserved in a
    ``SpinBandStructure``. The Fermi energy is **not** embedded in
    the EIGENVAL file; it must be obtained separately (e.g. from
    DOSCAR or OUTCAR).
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
        eigenvalues_down: np.ndarray | None = None
        if ispin == _ISPIN_SPIN_POLARIZED:
            eigenvalues_down = np.zeros(
                (nkpoints, nbands), dtype=np.float64
            )
        for k in range(nkpoints):
            kpoint_line: str = _read_next_nonempty_line(fid)
            if not kpoint_line:
                msg = "Unexpected EOF while reading EIGENVAL k-point block."
                raise ValueError(msg)
            kpoint_vals: list[float] = [
                float(x) for x in kpoint_line.split()
            ]
            if len(kpoint_vals) < _KPOINT_LINE_VALUES:
                msg = "Invalid EIGENVAL k-point line; expected 4 values."
                raise ValueError(msg)
            kpoints[k, :] = kpoint_vals[:_KPOINT_LINE_VALUES]
            for b in range(nbands):
                band_line: str = _read_next_nonempty_line(fid)
                if not band_line:
                    msg = "Unexpected EOF while reading EIGENVAL band line."
                    raise ValueError(msg)
                vals: list[float] = [
                    float(x) for x in band_line.split()
                ]
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
    """Read and return the next non-empty line, or ``""`` at EOF."""
    while True:
        line: str = fid.readline()
        if not line:
            return ""
        if line.strip():
            return line


__all__: list[str] = [
    "read_eigenval",
]
