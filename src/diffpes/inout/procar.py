"""Parse a VASP PROCAR file.

Extended Summary
----------------
The module reads VASP PROCAR files with orbital-resolved band projections. It
returns an :class:`~diffpes.types.OrbitalProjection` carrier. It supports
non-spin, spin-polarized (ISPIN=2), and SOC layouts.

Routine Listings
----------------
:func:`read_procar`
    Parse a VASP PROCAR file.

Notes
-----
Orbital ordering follows VASP convention:
``[s, py, pz, px, dxy, dyz, dz2, dxz, dx2-y2]``.
"""

import re
from pathlib import Path

import jax.numpy as jnp
import numpy as np
from beartype import beartype
from beartype.typing import (
    Any,
    Dict,
    List,
    Literal,
    Optional,
    TextIO,
    Tuple,
    Union,
)
from jaxtyping import Array, Float64, jaxtyped
from numpy.typing import NDArray

from diffpes.constants import (
    ISPIN2_BLOCKS,
    N_ORBITALS,
    N_SPIN_COMPONENTS,
    SOC_BLOCKS,
)
from diffpes.types import (
    OrbitalProjection,
    SpinOrbitalProjection,
    make_orbital_projection,
    make_spin_orbital_projection,
)


@jaxtyped(typechecker=beartype)
def read_procar(
    filename: str = "PROCAR",
    return_mode: Literal["legacy", "full"] = "legacy",
) -> Union[OrbitalProjection, SpinOrbitalProjection]:
    r"""Parse a VASP PROCAR file.

    The function reads a VASP PROCAR file that contains the orbital-resolved
    projections of Kohn-Sham wave functions onto site-centred
    spherical harmonics. Supports three layouts:

    - **Non-spin** (ISPIN=1, no SOC): single block of k-points.
    - **Spin-polarized** (ISPIN=2): two consecutive blocks of
      k-points (one per spin channel).
    - **SOC** (LSORBIT=.TRUE.): four consecutive blocks per k-point
      (total, Sx, Sy, Sz projections).

    The PROCAR file written by VASP (when ``LORBIT=11`` or ``12``)
    contains site- and orbital-resolved projections of each Kohn-Sham
    eigenstate. The file contains one or more blocks with the following data:

    * A header line with ``# of k-points``, ``# of bands``,
      ``# of ions``.
    * For each k-point: one coordinate line and one record for each band.
      Each band record contains the energy and an orbital header. It also
      contains one projection line for each atom. The projection columns are
      the ion index, nine orbitals, and ``tot``. A ``tot`` line and a blank
      line follow each record.

    The number of projection tables determines the spin layout:

    * 1 table: non-spin-polarized (ISPIN=1).
    * 2 tables: spin-polarized (ISPIN=2), table 0 = spin-up,
      table 1 = spin-down.
    * 4 tables: spin-orbit coupling (SOC), tables = total, Sx, Sy, Sz.

    SOC files arrive in two layouts. Legacy files repeat the whole
    k-point block four times. Modern files write one header and stack the
    four projection tables under each band record. The parser accepts
    both layouts and returns the same four-table structure.

    :see: :class:`~.test_procar.TestReadProcar`

    Implementation Logic
    --------------------
    1. **Parse the file blocks**::

           content = fid.read()
           blocks = _parse_procar_blocks(content)

       Each block carries one orbital projection table and its dimensions.
    2. **Identify the spin layout**::

           is_spin_polarized = nblocks == ISPIN2_BLOCKS
           is_soc = nblocks == SOC_BLOCKS

       The number of blocks distinguishes non-spin, ISPIN=2, and SOC data.
    3. **Build the projection and spin arrays**::

           avg = (proj_up + proj_down) / 2.0
           sx_sum = np.sum(proj_sx, axis=-1)
           sy_sum = np.sum(proj_sy, axis=-1)
           sz_sum = np.sum(proj_sz, axis=-1)

       Full mode stores signed spin components as separate non-negative pairs.
    4. **Construct the matching carrier**::

           projection_result = make_orbital_projection(projections=proj_arr)
           projection_result = make_spin_orbital_projection(
               projections=proj_arr, spin=spin_arr
           )

       Legacy and non-spin data use the orbital carrier. Spin data uses the
       mandatory-spin carrier.

    Parameters
    ----------
    filename : str, optional
        Path to PROCAR file. Default is ``"PROCAR"``.
    return_mode : Literal["legacy", "full"], optional
        ``"legacy"`` (default) returns an ``OrbitalProjection``
        from the first spin block only (backward-compatible).
        ``"full"`` returns a ``SpinOrbitalProjection`` (with
        mandatory spin field) for ISPIN=2 and SOC data, or an
        ``OrbitalProjection`` for non-spin data.

    Returns
    -------
    projection_result : Union[OrbitalProjection, SpinOrbitalProjection]
        ``OrbitalProjection`` for legacy mode or non-spin data.
        ``SpinOrbitalProjection`` for full mode with spin data.

    Raises
    ------
    ValueError
        If the parser finds no valid PROCAR blocks in the file.
        If the parsed projection-table count is not one, two, or four.

    Notes
    -----
    The 9 orbital channels follow the VASP convention:
    ``[s, py, pz, px, dxy, dyz, dz2, dxz, dx2-y2]``.
    The parser does not store the VASP ``tot`` column. It retains only the
    individual orbital columns. In the ISPIN=2 full mode, the parser uses
    ``(up + down) / 2`` as the orbital weight. Downstream consumers expect one
    projection array instead of separate spin channels. The parser encodes the
    spin texture as six nonnegative channels. These channels follow the ARPES
    simulation convention ``[Sx+, Sx-, Sy+, Sy-, Sz+, Sz-]``.
    """
    fid: TextIO

    path: Path = Path(filename)
    with path.open("r") as fid:
        content: str = fid.read()

    blocks: List[Dict[str, Any]] = _parse_procar_blocks(content)

    if not blocks:
        msg: str = "No valid PROCAR blocks found."
        raise ValueError(msg)

    nblocks: int = len(blocks)
    if nblocks not in (1, ISPIN2_BLOCKS, SOC_BLOCKS):
        count_msg: str = (
            f"unsupported PROCAR projection-table count: {nblocks}"
        )
        raise ValueError(count_msg)
    nkpts: int = blocks[0]["nkpts"]
    nbands: int = blocks[0]["nbands"]
    natoms: int = blocks[0]["natoms"]

    is_spin_polarized: bool = nblocks == ISPIN2_BLOCKS
    is_soc: bool = nblocks == SOC_BLOCKS

    if return_mode == "legacy" or (not is_spin_polarized and not is_soc):
        proj_arr: Float64[Array, " K B A 9"] = jnp.asarray(
            blocks[0]["projections"], dtype=jnp.float64
        )
        projection_result: Union[OrbitalProjection, SpinOrbitalProjection] = (
            make_orbital_projection(projections=proj_arr)
        )
    elif is_spin_polarized:
        proj_up: Float64[NDArray, "K B A O"] = blocks[0]["projections"]
        proj_down: Float64[NDArray, "K B A O"] = blocks[1]["projections"]
        avg: Float64[NDArray, "K B A O"] = (proj_up + proj_down) / 2.0
        proj_arr = jnp.asarray(avg, dtype=jnp.float64)
        spin_data: Float64[NDArray, "K B A 6"] = np.zeros(
            (nkpts, nbands, natoms, N_SPIN_COMPONENTS), dtype=np.float64
        )
        sz_diff: Float64[NDArray, "K B A"] = np.sum(
            proj_up - proj_down, axis=-1
        )
        spin_data[:, :, :, 4] = np.maximum(sz_diff, 0.0)
        spin_data[:, :, :, 5] = np.maximum(-sz_diff, 0.0)
        spin_arr: Float64[Array, " K B A 6"] = jnp.asarray(
            spin_data, dtype=jnp.float64
        )
        projection_result = make_spin_orbital_projection(
            projections=proj_arr, spin=spin_arr
        )
    else:
        proj_total: Float64[NDArray, "K B A O"] = blocks[0]["projections"]
        proj_sx: Float64[NDArray, "K B A O"] = blocks[1]["projections"]
        proj_sy: Float64[NDArray, "K B A O"] = blocks[2]["projections"]
        proj_sz: Float64[NDArray, "K B A O"] = blocks[3]["projections"]
        proj_arr = jnp.asarray(proj_total, dtype=jnp.float64)

        spin_data = np.zeros(
            (nkpts, nbands, natoms, N_SPIN_COMPONENTS), dtype=np.float64
        )
        sx_sum: Float64[NDArray, "K B A"] = np.sum(proj_sx, axis=-1)
        sy_sum: Float64[NDArray, "K B A"] = np.sum(proj_sy, axis=-1)
        sz_sum: Float64[NDArray, "K B A"] = np.sum(proj_sz, axis=-1)
        spin_data[:, :, :, 0] = np.maximum(sx_sum, 0.0)
        spin_data[:, :, :, 1] = np.maximum(-sx_sum, 0.0)
        spin_data[:, :, :, 2] = np.maximum(sy_sum, 0.0)
        spin_data[:, :, :, 3] = np.maximum(-sy_sum, 0.0)
        spin_data[:, :, :, 4] = np.maximum(sz_sum, 0.0)
        spin_data[:, :, :, 5] = np.maximum(-sz_sum, 0.0)
        spin_arr = jnp.asarray(spin_data, dtype=jnp.float64)
        projection_result = make_spin_orbital_projection(
            projections=proj_arr, spin=spin_arr
        )
    return projection_result


def _read_ion_rows(
    lines: List[str],
    start: int,
) -> Tuple[List[List[float]], int]:
    """PRIVATE: Read consecutive ion and total rows from one band record.

    Parameters
    ----------
    lines : List[str]
        Complete PROCAR text split into lines.
    start : int
        First candidate ion-row index.

    Returns
    -------
    result : Tuple[List[List[float]], int]
        Parsed orbital rows and the first unread line index.

    Implementation Logic
    --------------------
    Parse numeric ion rows, skip total rows, and stop at the next record.
    """
    band_rows: List[List[float]] = []
    i: int = start
    while i < len(lines):
        tokens: List[str] = lines[i].split()
        if tokens and tokens[0].isdigit():
            row: List[float] = [
                float(value) for value in tokens[1 : N_ORBITALS + 1]
            ]
            band_rows.append(row)
            i += 1
        elif tokens and tokens[0] == "tot":
            i += 1
        else:
            break
    result: Tuple[List[List[float]], int] = (band_rows, i)
    return result


def _read_projection_tables(
    lines: List[str],
    start: int,
    nkpts: int,
    nbands: int,
    natoms: int,
) -> Tuple[List[Float64[NDArray, "K B A O"]], int]:
    """PRIVATE: Read every projection table under one PROCAR header.

    Parameters
    ----------
    lines : List[str]
        Complete PROCAR text split into lines.
    start : int
        First line after the parsed dimensions header.
    nkpts : int
        Number of k-points declared by the header.
    nbands : int
        Number of bands declared by the header.
    natoms : int
        Number of ion rows in each projection table.

    Returns
    -------
    result : Tuple[List[Float64[NDArray, "K B A O"]], int]
        Projection tables and the first unread line index.

    Implementation Logic
    --------------------
    Locate each k-point and band header. Collect consecutive ion rows, split
    stacked SOC tables by atom count, and require one stable table count.

    Raises
    ------
    ValueError
        If ion-row counts are malformed or table counts vary by band.
    """
    b: int
    s: int
    sub_tables: List[Float64[NDArray, "K B A O"]] = []
    i: int = start
    k_re: str = r"k-point\s+(\d+)\s*:\s*" r"([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)"
    kpts_found: int = 0
    while i < len(lines) and kpts_found < nkpts:
        k_match: Optional[re.Match[str]] = re.search(k_re, lines[i])
        if k_match is None:
            i += 1
            continue
        k_idx: int = int(k_match.group(1)) - 1
        i += 1
        for b in range(nbands):
            while i < len(lines) and not lines[i].lstrip().startswith("band"):
                i += 1
            while i < len(lines) and not lines[i].lstrip().startswith("ion"):
                i += 1
            i += 1
            band_rows: List[List[float]]
            band_rows, i = _read_ion_rows(lines, i)
            if not band_rows or len(band_rows) % natoms != 0:
                rows_msg: str = (
                    "PROCAR band record has a malformed ion-row count"
                )
                raise ValueError(rows_msg)
            n_sub: int = len(band_rows) // natoms
            if not sub_tables:
                sub_tables = [
                    np.zeros(
                        (nkpts, nbands, natoms, N_ORBITALS),
                        dtype=np.float64,
                    )
                    for _ in range(n_sub)
                ]
            if n_sub != len(sub_tables):
                tables_msg: str = (
                    "PROCAR band records disagree on the "
                    "projection-table count"
                )
                raise ValueError(tables_msg)
            for s in range(n_sub):
                row_block: Float64[NDArray, "A O"] = np.asarray(
                    band_rows[s * natoms : (s + 1) * natoms],
                    dtype=np.float64,
                )
                sub_tables[s][k_idx, b] = row_block
        kpts_found += 1
    result: Tuple[List[Float64[NDArray, "K B A O"]], int] = (sub_tables, i)
    return result


def _parse_procar_blocks(
    content: str,
) -> List[Dict[str, Any]]:
    """PRIVATE: Parse all PROCAR blocks from the full file content string.

    A PROCAR file may contain one, two, or four projection tables. A
    header line matching ``"# of k-points: K  # of bands: B  # of
    ions: A"`` starts each k-point block. Legacy layouts repeat the
    whole header block per table. Modern SOC layouts write one header
    and stack four ion tables under each band record. The parser emits
    one dict per projection table for both layouts.

    Implementation Logic
    --------------------
    1. Split the content into lines and scan for lines containing the
       substring ``"k-points"`` (the block header).
    2. Extract ``(nkpts, nbands, natoms)`` from the header using a
       regex that captures all integers on the line.
    3. For each k-point within the block, search forward for a line
       matching ``k-point <index> : kx ky kz`` with a regex.
    4. For each band within the k-point:

       a. Scan forward to the band energy header, then to the orbital
          header line (``ion  s  py ...``). The scan tolerates blank
          lines between the anchors.
       b. Read every consecutive ion row (first token is the ion
          index), parsing columns 1 through 9 and skipping each
          ``tot`` summation row. Consecutive stacked tables produce
          ``natoms`` rows per table.
       c. Reject a row count that is not a positive multiple of
          ``natoms``, and reject a table count that changes between
          band records.

    5. Split the collected rows into per-table arrays and append one
       dict per table with keys ``'nkpts'``, ``'nbands'``,
       ``'natoms'``, and ``'projections'``.
    6. Return the list of table dicts.

    Parameters
    ----------
    content : str
        The entire PROCAR file content as a single string.

    Returns
    -------
    blocks : List[Dict[str, Any]]
        List of parsed blocks. Each dict contains:

        * ``'nkpts'`` (int): number of k-points.
        * ``'nbands'`` (int): number of bands.
        * ``'natoms'`` (int): number of atoms (ions).
        * ``'projections'`` (``Float64[NDArray, "nkpts nbands natoms 9"]``):
          orbital projections with dtype ``float64``.

    Notes
    -----
    The parser uses 1-based k-point indices from the file to place
    data into the 0-based NumPy array (``k_idx = parsed_index - 1``).
    The parser reads band and atom lines in sequence, not by their parsed
    indices. It does not store the ``tot`` column.
    """
    blocks: List[Dict[str, Any]] = []
    lines: List[str] = content.splitlines()
    i: int = 0

    while i < len(lines):
        if "k-points" not in lines[i]:
            i += 1
            continue
        header: str = lines[i]
        params: List[int] = [int(x) for x in re.findall(r"\d+", header)]
        nkpts: int = params[0]
        nbands: int = params[1]
        natoms: int = params[2]
        sub_tables: List[Float64[NDArray, "K B A O"]]
        sub_tables, i = _read_projection_tables(
            lines,
            i + 1,
            nkpts,
            nbands,
            natoms,
        )

        sub_table: Float64[NDArray, "K B A O"]
        for sub_table in sub_tables:
            blocks.append(
                {
                    "nkpts": nkpts,
                    "nbands": nbands,
                    "natoms": natoms,
                    "projections": sub_table,
                }
            )

    return blocks


__all__: list[str] = [
    "read_procar",
]
