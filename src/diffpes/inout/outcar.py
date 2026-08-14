"""Parse scalar summary values from a VASP OUTCAR file.

Extended Summary
----------------
The module reads a VASP OUTCAR file and returns the scalar summary
carrier :class:`~diffpes.types.OutcarData`. The parser keeps the last
reported Fermi energy and electron count. The last values describe the
final electronic step.

Routine Listings
----------------
:func:`read_outcar`
    Parse scalar summary values from a VASP OUTCAR file.
"""

from pathlib import Path

from beartype import beartype
from beartype.typing import List, Optional, TextIO
from jaxtyping import jaxtyped

from diffpes.types import (
    OutcarData,
    make_outcar_data,
)


@jaxtyped(typechecker=beartype)
def read_outcar(
    filename: str = "OUTCAR",
) -> OutcarData:
    """Parse scalar summary values from a VASP OUTCAR file.

    The function scans an OUTCAR file line by line. It keeps the last
    ``E-fermi`` value and the last ``NELECT`` value. VASP repeats these
    lines during ionic relaxation, so the last occurrence describes the
    final step.

    :see: :class:`~.test_outcar.TestReadOutcar`

    Implementation Logic
    --------------------
    1. **Scan the retained marker lines**::

           if "E-fermi" in line:
               fermi_energy = float(line.split()[2])

       The scan overwrites earlier matches, so the final electronic
       step wins.

    2. **Reject files without both markers**::

           raise ValueError("No E-fermi line found in OUTCAR file.")

       The explicit error separates a truncated file from a parse bug.

    3. **Return the validated carrier**::

           return summary

       The factory binds finiteness checks to the returned values.

    Parameters
    ----------
    filename : str, optional
        Path to the OUTCAR file. Default is ``"OUTCAR"``.

    Returns
    -------
    summary : OutcarData
        Carrier with the Fermi energy in eV and the electron count.

    Raises
    ------
    ValueError
        If the file contains no ``E-fermi`` line or no ``NELECT``
        line, or if a marker line does not parse.

    Notes
    -----
    The ``E-fermi`` line has the form ``E-fermi :   2.3919  ...``. The
    ``NELECT`` line has the form ``NELECT =  84.0000  ...``. Both
    values sit in the third whitespace-separated token.
    """
    fid: TextIO
    line: str
    exc: ValueError

    _value_token_index: int = 2
    fermi_energy: Optional[float] = None
    nelect: Optional[float] = None
    path: Path = Path(filename)
    with path.open("r", encoding="utf-8", errors="replace") as fid:
        for line in fid:
            tokens: List[str] = line.split()
            if len(tokens) <= _value_token_index:
                continue
            if "E-fermi" in line:
                try:
                    fermi_energy = float(tokens[_value_token_index])
                except ValueError as exc:
                    msg: str = f"Invalid E-fermi line in OUTCAR: {line!r}"
                    raise ValueError(msg) from exc
            elif tokens[0] == "NELECT":
                try:
                    nelect = float(tokens[_value_token_index])
                except ValueError as exc:
                    msg = f"Invalid NELECT line in OUTCAR: {line!r}"
                    raise ValueError(msg) from exc
    if fermi_energy is None:
        raise ValueError("No E-fermi line found in OUTCAR file.")
    if nelect is None:
        raise ValueError("No NELECT line found in OUTCAR file.")
    summary: OutcarData = make_outcar_data(
        fermi_energy=fermi_energy,
        nelect=nelect,
    )
    return summary


__all__: list[str] = [
    "read_outcar",
]
