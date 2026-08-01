"""Parse a VASP EIGENVAL file.

Extended Summary
----------------
The module reads VASP EIGENVAL files with electronic band energies. It returns
a band carrier. The :class:`~diffpes.types.BandStructure` carrier represents
non-spin data. The :class:`~diffpes.types.SpinBandStructure` carrier represents
spin data selected by the ``return_mode`` parameter.

Routine Listings
----------------
:func:`read_eigenval`
    Parse a VASP EIGENVAL file.

Notes
-----
The parser supports spin-polarized and nonpolarized computations. It sorts the
bands by energy within each k-point.
"""

from pathlib import Path

import jax.numpy as jnp
import numpy as np
from beartype import beartype
from beartype.typing import Literal, Optional, TextIO, Union
from jaxtyping import Float64, jaxtyped
from numpy.typing import NDArray

from diffpes.types import (
    BAND_LINE_MIN_VALUES,
    BAND_LINE_SPIN_VALUES,
    EIG_DOWN_INDEX,
    EIG_UP_INDEX,
    ISPIN_SPIN_POLARIZED,
    KPOINT_LINE_VALUES,
    BandStructure,
    ScalarFloat,
    SpinBandStructure,
    make_band_structure,
    make_spin_band_structure,
)


@jaxtyped(typechecker=beartype)
def read_eigenval(  # noqa: PLR0915
    filename: str = "EIGENVAL",
    fermi_energy: ScalarFloat = 0.0,
    return_mode: Literal["legacy", "full"] = "legacy",
) -> Union[BandStructure, SpinBandStructure]:
    """Parse a VASP EIGENVAL file.

    The function reads the electronic eigenvalues at each k-point from a VASP
    EIGENVAL file. It supports nonpolarized ISPIN=1 and spin-polarized ISPIN=2
    computations.

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

    :see: :class:`~.test_eigenval.TestReadEigenval`

    Implementation Logic
    --------------------
    1. **Read the spin and dimension metadata**::

           path: Path = Path(filename)
           with path.open("r") as fid:
               first_line: list[str] = fid.readline().split()
               ispin: int = int(first_line[3])

       This step fixes the static file layout before the function allocates
       the band arrays.

    2. **Sort each spin channel by band energy**::

           eigen_up = np.take_along_axis(eigen_up, order_up, axis=1)

       This gives every k-point the ascending band convention used downstream.

    3. **Return the requested band carrier**::

           return bands

       The return-mode branch binds scalar or spin-resolved data to one output.

    Parameters
    ----------
    filename : str, optional
        Path to EIGENVAL file. Default is ``"EIGENVAL"``.
    fermi_energy : ScalarFloat, optional
        Fermi level in eV used to reference the eigenvalues.
        The function accepts Python scalars and traced scalar arrays. The
        default is 0.0.
    return_mode : Literal["legacy", "full"], optional
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
        If a k-point line has fewer than four values.
        If a band line has fewer values than the selected ISPIN mode requires.
        If the parser reaches EOF before it reads all blocks.

    Notes
    -----
    For ISPIN=2 computations, each eigenvalue line contains the index and two
    energy columns. In ``"legacy"`` mode, the parser reads only the spin-up
    energy. In ``"full"`` mode, the parser retains both channels in a
    ``SpinBandStructure``. The EIGENVAL file does not contain the Fermi energy.
    Read it separately from a DOSCAR or OUTCAR file. The parser retains the
    eigenvalues without a ``fermi_energy`` shift. It stores the Fermi energy in
    the returned PyTree for downstream shifts.
    """
    fid: TextIO
    k: int
    b: int

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
        kpoints: Float64[NDArray, "K 4"] = np.zeros(
            (nkpoints, 4), dtype=np.float64
        )
        eigenvalues_up: Float64[NDArray, "K B"] = np.zeros(
            (nkpoints, nbands), dtype=np.float64
        )
        eigenvalues_down: Optional[Float64[NDArray, "K B"]] = None
        if ispin == ISPIN_SPIN_POLARIZED:
            eigenvalues_down = np.zeros((nkpoints, nbands), dtype=np.float64)
        for k in range(nkpoints):
            kpoint_line: str = _read_next_nonempty_line(fid)
            if not kpoint_line:
                msg: str = (
                    "Unexpected EOF while reading EIGENVAL k-point block."
                )
                raise ValueError(msg)
            kpoint_vals: list[float] = [float(x) for x in kpoint_line.split()]
            if len(kpoint_vals) < KPOINT_LINE_VALUES:
                msg: str = "Invalid EIGENVAL k-point line; expected 4 values."
                raise ValueError(msg)
            kpoints[k, :] = kpoint_vals[:KPOINT_LINE_VALUES]
            for b in range(nbands):
                band_line: str = _read_next_nonempty_line(fid)
                if not band_line:
                    msg: str = (
                        "Unexpected EOF while reading EIGENVAL band line."
                    )
                    raise ValueError(msg)
                vals: list[float] = [float(x) for x in band_line.split()]
                if len(vals) < BAND_LINE_MIN_VALUES:
                    msg: str = (
                        "Invalid EIGENVAL band line; expected band energy."
                    )
                    raise ValueError(msg)
                eigenvalues_up[k, b] = vals[EIG_UP_INDEX]
                if (
                    ispin == ISPIN_SPIN_POLARIZED
                    and eigenvalues_down is not None
                ):
                    if len(vals) < BAND_LINE_SPIN_VALUES:
                        msg: str = (
                            "Invalid spin-polarized EIGENVAL band line; "
                            "expected spin-down energy."
                        )
                        raise ValueError(msg)
                    eigenvalues_down[k, b] = vals[EIG_DOWN_INDEX]
        eigenvalues_up = np.sort(eigenvalues_up, axis=1)
        if eigenvalues_down is not None:
            eigenvalues_down = np.sort(eigenvalues_down, axis=1)

    if (
        return_mode == "full"
        and ispin == ISPIN_SPIN_POLARIZED
        and eigenvalues_down is not None
    ):
        bands: Union[BandStructure, SpinBandStructure] = (
            make_spin_band_structure(
                eigenvalues_up=jnp.asarray(eigenvalues_up),
                eigenvalues_down=jnp.asarray(eigenvalues_down),
                kpoints=jnp.asarray(kpoints[:, :3]),
                kpoint_weights=jnp.asarray(kpoints[:, 3]),
                fermi_energy=fermi_energy,
            )
        )
    else:
        bands = make_band_structure(
            eigenvalues=jnp.asarray(eigenvalues_up),
            kpoints=jnp.asarray(kpoints[:, :3]),
            kpoint_weights=jnp.asarray(kpoints[:, 3]),
            fermi_energy=fermi_energy,
        )
    return bands


def _read_next_nonempty_line(fid: TextIO) -> str:
    """Read and return the next non-empty line, or ``""`` at EOF.

    Extended Summary
    ----------------
    The helper skips blank lines in the EIGENVAL file. VASP separates k-point
    blocks with empty lines. The helper consumes them before the main parsing
    loop processes the data.

    Implementation Logic
    --------------------
    1. Call ``fid.readline()`` in a loop.
    2. If the returned string is empty, return ``""`` for EOF.
    3. If the stripped line is non-empty, return the original
       (unstripped) line.
    4. Otherwise continue to the next line.

    Parameters
    ----------
    fid : TextIO
        Open file handle positioned somewhere within the EIGENVAL file.

    Returns
    -------
    line : str
        The next nonempty line with its trailing newline, or ``""`` at EOF.
    """
    result: str = ""
    while True:
        line: str = fid.readline()
        if not line:
            break
        if line.strip():
            result = line
            break
    return result


__all__: list[str] = [
    "read_eigenval",
]
