"""VASP KPOINTS file parser.

Extended Summary
----------------
Reads VASP KPOINTS files and returns a
:class:`~arpyes.types.KPathInfo` PyTree with symmetry point
labels and their indices along the Brillouin zone path.

Routine Listings
----------------
:func:`read_kpoints`
    Parse a VASP KPOINTS file into a KPathInfo.

Notes
-----
Supports Automatic (Monkhorst-Pack), Line-mode, and Explicit
k-point specifications.
"""

import re
from pathlib import Path

from arpyes.types import KPathInfo, make_kpath_info


def read_kpoints(
    filename: str = "KPOINTS",
) -> KPathInfo:
    """Parse a VASP KPOINTS file.

    Reads a VASP KPOINTS file that specifies the Brillouin-zone sampling
    for a band-structure calculation. The KPOINTS file has three possible
    modes: Line-mode (explicit path segments between high-symmetry
    points), Automatic (Monkhorst-Pack or Gamma-centred grid), and
    Explicit (individually listed k-points). This function extracts
    symmetry-point labels and their indices along the path and returns
    a :class:`~arpyes.types.KPathInfo` PyTree.

    Implementation Logic
    --------------------
    1. **Read header** -- consume the comment line (line 1), the
       number-of-k-points line (line 2, parsed as ``num_kpts``), and
       the mode line (line 3).

    2. **Determine mode** -- if the mode line contains ``"line"`` the
       file is Line-mode; otherwise if ``num_kpts == 0`` it is
       Automatic; otherwise Explicit.

    3. **Skip coordinate-type line** -- consume line 4 (Reciprocal /
       Cartesian indicator) without storing it.

    4. **Parse k-point data** (mode-dependent):
       - *Line-mode*: collect all remaining non-blank lines. Lines are
         grouped in pairs (start, end) forming path segments. For each
         segment, extract the symmetry label from the start and end
         lines using ``_extract_label``. The running index advances by
         ``num_kpts`` per segment; the start label of the first segment
         gets index 0, and each segment's end label gets index
         ``idx + num_kpts - 1``.
       - *Automatic*: ``total_kpts`` is set to 0 (grid size determined
         by VASP internally).
       - *Explicit*: ``total_kpts`` equals the header value.

    5. **Construct PyTree** -- call ``make_kpath_info`` with the
       accumulated labels, label indices, mode string, and total
       k-point count.

    Parameters
    ----------
    filename : str, optional
        Path to KPOINTS file. Default is ``"KPOINTS"``.

    Returns
    -------
    kpath : KPathInfo
        K-point path metadata with labels and indices.

    Notes
    -----
    In Line-mode the number on line 2 is the number of k-points
    **per segment**, not the total. The total equals
    ``n_segments * num_kpts``. Labels are extracted via
    ``_extract_label``, which looks for the ``! LABEL`` comment
    convention used by tools such as AFLOW and SeeK-path.
    """
    path: Path = Path(filename)
    with path.open("r") as fid:
        _comment: str = fid.readline().strip()
        num_line: str = fid.readline().strip()
        num_kpts: int = int(num_line.split(maxsplit=1)[0])
        mode_line: str = fid.readline().strip().lower()
        if "line" in mode_line:
            mode: str = "Line-mode"
        elif num_kpts == 0:
            mode = "Automatic"
        else:
            mode = "Explicit"
        _coord_type: str = fid.readline().strip()
        labels: list[str] = []
        label_indices: list[int] = []
        if mode == "Line-mode":
            raw_lines: list[str] = []
            for line in fid:
                stripped: str = line.strip()
                if stripped:
                    raw_lines.append(stripped)
            n_endpoints: int = len(raw_lines)
            n_segments: int = n_endpoints // 2
            idx: int = 0
            for seg in range(n_segments):
                start_line: str = raw_lines[2 * seg]
                end_line: str = raw_lines[2 * seg + 1]
                start_label: str = _extract_label(start_line)
                end_label: str = _extract_label(end_line)
                if seg == 0:
                    labels.append(start_label)
                    label_indices.append(idx)
                idx += num_kpts
                labels.append(end_label)
                label_indices.append(idx - 1)
            total_kpts: int = n_segments * num_kpts
        elif mode == "Automatic":
            total_kpts = 0
        else:
            total_kpts = num_kpts
    kpath: KPathInfo = make_kpath_info(
        num_kpoints=total_kpts,
        label_indices=label_indices if label_indices else [0],
        mode=mode,
        labels=tuple(labels),
    )
    return kpath


def _extract_label(line: str) -> str:
    r"""Extract symmetry label from a KPOINTS line.

    Parses a single coordinate line from a VASP KPOINTS file and
    attempts to recover the human-readable symmetry label (e.g.
    ``"G"``, ``"X"``, ``"M"``) that is conventionally appended after
    the three fractional coordinates.

    Implementation Logic
    --------------------
    1. **Regex match** -- search for the pattern ``! <label>`` using
       ``re.search(r"!\\s*(\\S+)", line)``. The ``!`` delimiter is the
       standard VASP comment marker used by AFLOW, SeeK-path, and most
       KPOINTS generators. If a match is found, return the first
       captured non-whitespace group.

    2. **Fallback heuristic** -- if no ``!`` marker is present, split
       the line on whitespace. If there are more than 4 tokens (three
       coordinates plus at least a weight and a label), return the last
       token as the label.

    3. **Default** -- if neither strategy yields a label, return an
       empty string.

    Parameters
    ----------
    line : str
        A single k-point line from the KPOINTS file, typically of the
        form ``"0.0  0.0  0.0  ! G"`` or ``"0.0  0.0  0.0  1  G"``.

    Returns
    -------
    label : str
        Extracted symmetry label, or ``""`` if none is found.

    Notes
    -----
    The ``! LABEL`` convention is the most reliable. The fallback
    heuristic can misidentify a numeric weight as a label when the
    line has exactly 5 whitespace-separated tokens, but this situation
    is rare in practice.
    """
    _min_parts_with_label: int = 4
    match: re.Match[str] | None = re.search(r"!\s*(\S+)", line)
    if match:
        return match.group(1)
    parts: list[str] = line.split()
    if len(parts) > _min_parts_with_label:
        return parts[-1]
    return ""


__all__: list[str] = [
    "read_kpoints",
]
