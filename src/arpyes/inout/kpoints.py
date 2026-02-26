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

    Parameters
    ----------
    filename : str, optional
        Path to KPOINTS file. Default is ``"KPOINTS"``.

    Returns
    -------
    kpath : KPathInfo
        K-point path metadata with labels and indices.
    """
    path: Path = Path(filename)
    with path.open("r") as fid:
        _comment: str = fid.readline().strip()
        num_line: str = fid.readline().strip()
        num_kpts: int = int(
            num_line.split(maxsplit=1)[0]
        )
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
                start_label: str = _extract_label(
                    start_line
                )
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
    """Extract symmetry label from a KPOINTS line.

    Parameters
    ----------
    line : str
        A line from the KPOINTS file.

    Returns
    -------
    label : str
        Extracted label or empty string.
    """
    _min_parts_with_label: int = 4
    match: re.Match[str] | None = re.search(
        r"!\s*(\S+)", line
    )
    if match:
        return match.group(1)
    parts: list[str] = line.split()
    if len(parts) > _min_parts_with_label:
        return parts[-1]
    return ""


__all__: list[str] = [
    "read_kpoints",
]
