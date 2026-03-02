"""VASP KPOINTS file parser.

Extended Summary
----------------
Reads VASP KPOINTS files and returns a
:class:`~diffpes.types.KPathInfo` PyTree containing plotting labels and
mode-specific metadata (automatic grid/shift, explicit weights, and
line-mode segment endpoints).

Routine Listings
----------------
:func:`read_kpoints`
    Parse a VASP KPOINTS file into KPathInfo.
"""

import re
from pathlib import Path

import jax.numpy as jnp
from beartype.typing import Optional
from jaxtyping import Array, Float, Int

from diffpes.types import KPathInfo, make_kpath_info

_FLOAT_TOKEN_RE: re.Pattern[str] = re.compile(
    r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
)
_XYZ_COMPONENTS: int = 3
_WEIGHT_COMPONENT_INDEX: int = 3
_WEIGHT_COMPONENT_COUNT: int = 4
_COORDINATE_MODE_TOKENS: set[str] = {
    "cartesian",
    "reciprocal",
    "direct",
    "fractional",
}


def read_kpoints(  # noqa: PLR0915
    filename: str = "KPOINTS",
) -> KPathInfo:
    """Parse a VASP KPOINTS file.

    Reads a VASP KPOINTS file that specifies Brillouin-zone sampling.
    Supports the three standard modes:
    Line-mode (path segments), Automatic (Monkhorst-Pack/Gamma grids),
    and Explicit (listed k-points with optional weights).

    Implementation Logic
    --------------------
    1. Parse comment, line-2 integer, and line-3 mode/scheme.
    2. Dispatch by mode:
       - Line-mode: parse paired endpoint lines, derive segment count,
         endpoint k-points, labels, and label indices.
       - Automatic: parse grid and shift vectors.
       - Explicit: parse listed k-points and weights.
    3. Construct ``KPathInfo`` with both legacy plotting fields and
       richer mode-specific metadata.

    Parameters
    ----------
    filename : str, optional
        Path to KPOINTS file. Default is ``"KPOINTS"``.

    Returns
    -------
    kpath : KPathInfo
        K-point metadata including labels/indices and mode-specific
        parsed fields.

    Notes
    -----
    In Line-mode, line 2 is points-per-segment. ``num_kpoints`` in the
    returned object is the total count ``segments * points_per_segment``,
    preserving existing plotting behavior.
    """
    path: Path = Path(filename)
    with path.open("r") as fid:
        comment: str = fid.readline().strip()
        num_line: str = fid.readline().strip()
        points_per_segment: int = int(num_line.split(maxsplit=1)[0])
        scheme_or_mode: str = fid.readline().strip()
        mode_line: str = scheme_or_mode.lower()

        labels: list[str] = []
        label_indices: list[int] = []
        line_endpoints: list[list[float]] = []
        explicit_kpoints: list[list[float]] = []
        explicit_weights: list[float] = []
        grid: Optional[list[int]] = None
        shift: Optional[list[float]] = None
        coord_mode: str = ""
        segments: int = 0
        total_kpts: int

        if "line" in mode_line:
            mode: str = "Line-mode"
            coord_mode = fid.readline().strip()
            raw_lines: list[str] = [
                line.strip() for line in fid if line.strip()
            ]
            segments = len(raw_lines) // 2

            if segments > 0:
                line_endpoints.append(_extract_coords(raw_lines[0]))
                labels.append(_extract_label(raw_lines[0]))
                label_indices.append(0)

            idx: int = 0
            for seg in range(segments):
                end_line: str = raw_lines[2 * seg + 1]
                line_endpoints.append(_extract_coords(end_line))
                labels.append(_extract_label(end_line))
                idx += points_per_segment
                label_indices.append(idx - 1)

            total_kpts = segments * points_per_segment
        elif points_per_segment == 0:
            mode = "Automatic"
            coord_mode = scheme_or_mode
            grid = _parse_grid(fid.readline())
            shift = _parse_shift(fid.readline())
            total_kpts = 0
        else:
            mode = "Explicit"
            remaining_lines: list[str] = [
                line.strip() for line in fid if line.strip()
            ]
            coord_mode = scheme_or_mode
            if (
                mode_line not in _COORDINATE_MODE_TOKENS
                and remaining_lines
                and not _looks_like_kpoint_line(remaining_lines[0])
            ):
                coord_mode = remaining_lines.pop(0)
            explicit_kpoints, explicit_weights = _parse_explicit_kpoints(
                remaining_lines, points_per_segment
            )
            total_kpts = points_per_segment

    line_endpoints_arr: Optional[Float[Array, "K 3"]] = None
    if line_endpoints:
        line_endpoints_arr = jnp.asarray(line_endpoints, dtype=jnp.float64)
    explicit_kpoints_arr: Optional[Float[Array, "K 3"]] = None
    if explicit_kpoints:
        explicit_kpoints_arr = jnp.asarray(explicit_kpoints, dtype=jnp.float64)
    explicit_weights_arr: Optional[Float[Array, " K"]] = None
    if explicit_weights:
        explicit_weights_arr = jnp.asarray(explicit_weights, dtype=jnp.float64)
    grid_arr: Optional[Int[Array, " 3"]] = None
    if grid is not None:
        grid_arr = jnp.asarray(grid, dtype=jnp.int32)
    shift_arr: Optional[Float[Array, " 3"]] = None
    if shift is not None:
        shift_arr = jnp.asarray(shift, dtype=jnp.float64)

    parsed_kpoints: Optional[Float[Array, "K 3"]] = line_endpoints_arr
    parsed_weights: Optional[Float[Array, " K"]] = None
    if mode == "Explicit":
        parsed_kpoints = explicit_kpoints_arr
        parsed_weights = explicit_weights_arr

    kpath: KPathInfo = make_kpath_info(
        num_kpoints=total_kpts,
        label_indices=label_indices if label_indices else [0],
        points_per_segment=points_per_segment,
        segments=segments,
        kpoints=parsed_kpoints,
        weights=parsed_weights,
        grid=grid_arr,
        shift=shift_arr,
        mode=mode,
        labels=tuple(labels),
        comment=comment,
        coordinate_mode=coord_mode,
    )
    return kpath


def _parse_explicit_kpoints(
    lines: list[str],
    num_kpoints: int,
) -> tuple[list[list[float]], list[float]]:
    """Parse explicit-mode k-point coordinates and optional weights.

    Extended Summary
    ----------------
    In VASP explicit KPOINTS mode, each k-point is listed on its own
    line with at least three fractional/Cartesian coordinates and an
    optional fourth column for the integration weight.

    Implementation Logic
    --------------------
    1. Iterate over ``lines``, stopping once ``num_kpoints`` k-points
       have been collected.
    2. Split each line and parse all tokens as floats. Raise
       ``ValueError`` if any token is not numeric or if fewer than 3
       values are present.
    3. Store the first 3 values as coordinates.
    4. If a 4th value is present, use it as the weight; otherwise
       default to 1.0.

    Parameters
    ----------
    lines : list[str]
        Remaining lines from the KPOINTS file after the mode line.
    num_kpoints : int
        Expected number of k-points to parse.

    Returns
    -------
    points : list[list[float]]
        Parsed k-point coordinates, each a 3-element list.
    weights : list[float]
        Corresponding k-point weights.

    Raises
    ------
    ValueError
        If a coordinate line cannot be parsed or has fewer than 3
        numeric tokens.
    """
    points: list[list[float]] = []
    weights: list[float] = []
    for stripped in lines:
        if len(points) >= num_kpoints:
            break
        parts: list[float]
        try:
            parts = [float(x) for x in stripped.split()]
        except ValueError as exc:
            msg = f"Invalid explicit KPOINTS coordinate line: {stripped!r}"
            raise ValueError(msg) from exc
        if len(parts) < _XYZ_COMPONENTS:
            msg = "Explicit KPOINTS line must contain at least 3 coordinates."
            raise ValueError(msg)
        points.append(parts[:_XYZ_COMPONENTS])
        if len(parts) >= _WEIGHT_COMPONENT_COUNT:
            weights.append(parts[_WEIGHT_COMPONENT_INDEX])
        else:
            weights.append(1.0)
    return points, weights


def _looks_like_kpoint_line(line: str) -> bool:
    """Return ``True`` when the first three tokens are parseable floats.

    Extended Summary
    ----------------
    Heuristic used to distinguish a k-point coordinate line from a
    coordinate-mode descriptor line (e.g. ``"Reciprocal"``) when the
    KPOINTS file uses explicit mode and the mode keyword also happens
    to appear on line 3. Lines with fewer than 3 tokens or with
    non-numeric leading tokens return ``False``.

    Parameters
    ----------
    line : str
        A single line from the KPOINTS file.

    Returns
    -------
    bool
        ``True`` if the first three whitespace-separated tokens can
        all be converted to ``float``; ``False`` otherwise.
    """
    parts: list[str] = line.split()
    if len(parts) < _XYZ_COMPONENTS:
        return False
    try:
        float(parts[0])
        float(parts[1])
        float(parts[2])
    except ValueError:
        return False
    return True


def _parse_grid(line: str) -> list[int]:
    """Parse automatic-mode grid line into three integers.

    Extended Summary
    ----------------
    In VASP automatic KPOINTS mode, the line immediately after the
    scheme keyword contains the three Monkhorst-Pack grid subdivisions
    ``N1 N2 N3``. Tokens are parsed as floats first and then rounded
    to the nearest integer to tolerate files written with decimal
    points (e.g. ``"4.0 4.0 4.0"``).

    Parameters
    ----------
    line : str
        The grid-dimension line from the KPOINTS file.

    Returns
    -------
    list[int]
        Three-element list ``[N1, N2, N3]``.

    Raises
    ------
    ValueError
        If the line contains fewer than 3 whitespace-separated tokens.
    """
    vals: list[str] = line.split()
    if len(vals) < _XYZ_COMPONENTS:
        msg = "Automatic KPOINTS grid line must have 3 values."
        raise ValueError(msg)
    return [
        int(round(float(vals[0]))),
        int(round(float(vals[1]))),
        int(round(float(vals[2]))),
    ]


def _parse_shift(line: str) -> list[float]:
    """Parse automatic-mode shift line into three floats.

    Extended Summary
    ----------------
    In VASP automatic KPOINTS mode, the line after the grid line
    contains the shift vector applied to the Monkhorst-Pack mesh.
    Values of ``0 0 0`` indicate a Gamma-centred grid; non-zero
    values shift the mesh in fractional reciprocal-space coordinates.

    Parameters
    ----------
    line : str
        The shift line from the KPOINTS file.

    Returns
    -------
    list[float]
        Three-element list ``[s1, s2, s3]``.

    Raises
    ------
    ValueError
        If the line contains fewer than 3 whitespace-separated tokens.
    """
    vals: list[str] = line.split()
    if len(vals) < _XYZ_COMPONENTS:
        msg = "Automatic KPOINTS shift line must have 3 values."
        raise ValueError(msg)
    return [float(vals[0]), float(vals[1]), float(vals[2])]


def _extract_coords(line: str) -> list[float]:
    """Extract first three float tokens from a KPOINTS coordinate line.

    Extended Summary
    ----------------
    Uses a regex (``_FLOAT_TOKEN_RE``) to find all floating-point
    tokens in the line, regardless of surrounding non-numeric text
    (such as symmetry labels or comment markers). This makes the
    extraction robust to various KPOINTS formatting styles.

    Parameters
    ----------
    line : str
        A single k-point coordinate line.

    Returns
    -------
    list[float]
        Three-element list of the first three float values found.

    Raises
    ------
    ValueError
        If fewer than 3 float tokens are found on the line.
    """
    tokens: list[str] = _FLOAT_TOKEN_RE.findall(line)
    if len(tokens) < _XYZ_COMPONENTS:
        msg = f"Could not parse k-point coordinates from line: {line!r}"
        raise ValueError(msg)
    return [float(tokens[0]), float(tokens[1]), float(tokens[2])]


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
    match: Optional[re.Match[str]] = re.search(r"!\s*(\S+)", line)
    if match:
        return match.group(1)
    parts: list[str] = line.split()
    if len(parts) > _min_parts_with_label:
        return parts[-1]
    return ""


__all__: list[str] = [
    "read_kpoints",
]
