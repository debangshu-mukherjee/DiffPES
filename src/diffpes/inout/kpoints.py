"""Parse a VASP KPOINTS file.

Extended Summary
----------------
The module reads VASP KPOINTS files. It returns a ``KPathInfo`` carrier with
plotting labels and metadata for each mode. The metadata includes grids,
shifts, weights, and segment endpoints.

Routine Listings
----------------
:func:`read_kpoints`
    Parse a VASP KPOINTS file.
"""

import re
from pathlib import Path

import jax.numpy as jnp
from beartype import beartype
from beartype.typing import Optional, TextIO
from jaxtyping import Array, Float64, Int32, jaxtyped

from diffpes.types import (
    COORDINATE_MODE_TOKENS,
    FLOAT_TOKEN_RE,
    WEIGHT_COMPONENT_COUNT,
    WEIGHT_COMPONENT_INDEX,
    XYZ_COMPONENTS,
    KPathInfo,
    make_kpath_info,
)


@jaxtyped(typechecker=beartype)
def read_kpoints(  # noqa: PLR0915
    filename: str = "KPOINTS",
) -> KPathInfo:
    """Parse a VASP KPOINTS file.

    The function reads a VASP KPOINTS file that specifies Brillouin-zone
    sampling. It supports three standard modes:
    Line-mode (path segments), Automatic (Monkhorst-Pack/Gamma grids),
    and Explicit (listed k-points with optional weights).

    :see: :class:`~.test_kpoints.TestReadKpoints`

    Implementation Logic
    --------------------
    1. **Read the mode selector**::

           mode_token: str = lines[2].strip()
           mode_lower: str = mode_token.lower()

       This determines the static parser branch from the third KPOINTS record.

    2. **Dispatch the selected layout**::

           if "line" in mode_lower:
               mode = "Line-mode"

       The other branches parse automatic grids or explicit weighted points.

    3. **Return normalized path metadata**::

           return kpath

       Each layout binds its fields through one factory.

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
    fid: TextIO
    seg: int

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
                mode_line not in COORDINATE_MODE_TOKENS
                and remaining_lines
                and not _looks_like_kpoint_line(remaining_lines[0])
            ):
                coord_mode = remaining_lines.pop(0)
            explicit_kpoints, explicit_weights = _parse_explicit_kpoints(
                remaining_lines, points_per_segment
            )
            total_kpts = points_per_segment

    line_endpoints_arr: Optional[Float64[Array, "K 3"]] = None
    if line_endpoints:
        line_endpoints_arr = jnp.asarray(line_endpoints, dtype=jnp.float64)
    explicit_kpoints_arr: Optional[Float64[Array, "K 3"]] = None
    if explicit_kpoints:
        explicit_kpoints_arr = jnp.asarray(explicit_kpoints, dtype=jnp.float64)
    explicit_weights_arr: Optional[Float64[Array, " K"]] = None
    if explicit_weights:
        explicit_weights_arr = jnp.asarray(explicit_weights, dtype=jnp.float64)
    grid_arr: Optional[Int32[Array, " 3"]] = None
    if grid is not None:
        grid_arr = jnp.asarray(grid, dtype=jnp.int32)
    shift_arr: Optional[Float64[Array, " 3"]] = None
    if shift is not None:
        shift_arr = jnp.asarray(shift, dtype=jnp.float64)

    parsed_kpoints: Optional[Float64[Array, "K 3"]] = line_endpoints_arr
    parsed_weights: Optional[Float64[Array, " K"]] = None
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
    """PRIVATE: Parse explicit-mode k-point coordinates and optional weights.

    Extended Summary
    ----------------
    In the VASP explicit mode, one line defines each k-point. The line contains
    at least three coordinates and can contain an integration weight.

    Implementation Logic
    --------------------
    1. Iterate over ``lines`` until the function collects ``num_kpoints``
       k-points.
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
        If the parser cannot read a coordinate line.
        If a coordinate line has fewer than three numeric tokens.
    """
    stripped: str

    exc: ValueError
    points: list[list[float]] = []
    weights: list[float] = []
    for stripped in lines:
        if len(points) >= num_kpoints:
            break
        parts: list[float]
        try:
            parts = [float(x) for x in stripped.split()]
        except ValueError as exc:
            msg: str = (
                f"Invalid explicit KPOINTS coordinate line: {stripped!r}"
            )
            raise ValueError(msg) from exc
        if len(parts) < XYZ_COMPONENTS:
            msg = "Explicit KPOINTS line must contain at least 3 coordinates."
            raise ValueError(msg)
        points.append(parts[:XYZ_COMPONENTS])
        if len(parts) >= WEIGHT_COMPONENT_COUNT:
            weights.append(parts[WEIGHT_COMPONENT_INDEX])
        else:
            weights.append(1.0)
    explicit_kpoints: tuple[list[list[float]], list[float]] = (
        points,
        weights,
    )
    return explicit_kpoints


def _looks_like_kpoint_line(line: str) -> bool:
    """PRIVATE: Return ``True`` when the first three tokens are parseable
    floats.

    Extended Summary
    ----------------
    The heuristic distinguishes a k-point coordinate line from a coordinate
    mode descriptor. Explicit files can place the mode keyword on line 3.
    Lines with fewer than three tokens return ``False``. Lines with nonnumeric
    leading tokens also return ``False``.

    Parameters
    ----------
    line : str
        A single line from the KPOINTS file.

    Returns
    -------
    bool
        ``True`` if ``float`` can convert each of the first three tokens;
        otherwise, ``False``.
    """
    parts: list[str] = line.split()
    looks_like_kpoint: bool = False
    if len(parts) < XYZ_COMPONENTS:
        looks_like_kpoint = False
    else:
        try:
            float(parts[0])
            float(parts[1])
            float(parts[2])
        except ValueError:
            looks_like_kpoint = False
        else:
            looks_like_kpoint = True
    return looks_like_kpoint


def _parse_grid(line: str) -> list[int]:
    """PRIVATE: Parse automatic-mode grid line into three integers.

    Extended Summary
    ----------------
    In the VASP automatic mode, the line after the scheme contains the three
    Monkhorst-Pack subdivisions. The helper converts the tokens to floats. It
    then rounds them to accept decimal forms such as ``"4.0 4.0 4.0"``.

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
    if len(vals) < XYZ_COMPONENTS:
        msg: str = "Automatic KPOINTS grid line must have 3 values."
        raise ValueError(msg)
    grid: list[int] = [
        int(round(float(vals[0]))),
        int(round(float(vals[1]))),
        int(round(float(vals[2]))),
    ]
    return grid


def _parse_shift(line: str) -> list[float]:
    """PRIVATE: Parse automatic-mode shift line into three floats.

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
    if len(vals) < XYZ_COMPONENTS:
        msg: str = "Automatic KPOINTS shift line must have 3 values."
        raise ValueError(msg)
    shift: list[float] = [float(vals[0]), float(vals[1]), float(vals[2])]
    return shift


def _extract_coords(line: str) -> list[float]:
    """PRIVATE: Extract first three float tokens from a KPOINTS coordinate
    line.

    Extended Summary
    ----------------
    The helper uses ``FLOAT_TOKEN_RE`` to find each floating-point token in the
    line. It ignores surrounding nonnumeric text, including labels and comment
    markers. This behavior supports different KPOINTS formats.

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
        If the helper finds fewer than three float tokens on the line.
    """
    tokens: list[str] = FLOAT_TOKEN_RE.findall(line)
    if len(tokens) < XYZ_COMPONENTS:
        msg: str = f"Could not parse k-point coordinates from line: {line!r}"
        raise ValueError(msg)
    coordinates: list[float] = [
        float(tokens[0]),
        float(tokens[1]),
        float(tokens[2]),
    ]
    return coordinates


def _extract_label(line: str) -> str:
    r"""PRIVATE: Extract symmetry label from a KPOINTS line.

    The helper parses one coordinate line from a VASP KPOINTS file. It tries to
    read the symmetry label after the three fractional coordinates.

    Implementation Logic
    --------------------
    1. **Match the expression**: Search for ``! <label>`` with
       ``re.search(r"!\\s*(\\S+)", line)``. AFLOW, SeeK-path, and many KPOINTS
       generators use the ``!`` delimiter. Return the first captured group.

    2. **Apply the fallback**: Split a line without ``!`` on whitespace.
       Return the last token when the line contains more than four tokens.

    3. **Use the default**: Return an empty string when neither strategy finds
       a label.

    Parameters
    ----------
    line : str
        A single k-point line from the KPOINTS file, typically of the
        form ``"0.0  0.0  0.0  ! G"`` or ``"0.0  0.0  0.0  1  G"``.

    Returns
    -------
    label : str
        The symmetry label, or ``""`` when no label exists.

    Notes
    -----
    The ``! LABEL`` convention gives the most reliable result. The fallback
    can incorrectly read a numeric weight as a label in a five-token line.
    """
    _min_parts_with_label: int = 4
    match: Optional[re.Match[str]] = re.search(r"!\s*(\S+)", line)
    if match:
        label: str = match.group(1)
    else:
        parts: list[str] = line.split()
        label = parts[-1] if len(parts) > _min_parts_with_label else ""
    return label


__all__: list[str] = [
    "read_kpoints",
]
