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
from beartype.typing import Any, Dict, List, Literal, Optional, TextIO, Union
from jaxtyping import Array, Float64, jaxtyped
from numpy.typing import NDArray

from diffpes.types import (
    ISPIN2_BLOCKS,
    N_ORBITALS,
    N_SPIN_COMPONENTS,
    SOC_BLOCKS,
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

    The number of blocks determines the spin layout:

    * 1 block: non-spin-polarized (ISPIN=1).
    * 2 blocks: spin-polarized (ISPIN=2), block 0 = spin-up,
      block 1 = spin-down.
    * 4 blocks: spin-orbit coupling (SOC), blocks = total, Sx, Sy, Sz.

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


def _parse_procar_blocks(
    content: str,
) -> List[Dict[str, Any]]:
    """PRIVATE: Parse all PROCAR blocks from the full file content string.

    Extended Summary
    ----------------
    A PROCAR file may contain 1, 2, or 4 consecutive blocks depending
    on the spin configuration. A header line starts each block. The header
    matches ``"# of k-points: K  # of bands: B  # of ions: A"``. Nested
    projection data follows the header.

    Implementation Logic
    --------------------
    1. Split the content into lines and scan for lines containing the
       substring ``"k-points"`` (the block header).
    2. Extract ``(nkpts, nbands, natoms)`` from the header using a
       regex that captures all integers on the line.
    3. Allocate a ``(nkpts, nbands, natoms, 9)`` NumPy array for the
       orbital projections.
    4. For each k-point within the block:

       a. Search forward for a line matching the pattern
          ``k-point <index> : kx ky kz`` using a regex.
       b. For each band within the k-point:

          i.   Skip lines until the helper finds the band energy header.
          ii.  Skip the orbital-name header line (``ion  s  py ...``).
          iii. Read ``natoms`` lines, parsing columns 1 through 9
               (skipping the ion index in column 0) as the orbital
               projections.
          iv.  Skip the ``tot`` summation line and the trailing blank
               line.

    5. Append a dict with keys ``'nkpts'``, ``'nbands'``,
       ``'natoms'``, and ``'projections'`` for each block.
    6. Return the list of block dicts.

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
    b: int
    a: int

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
        projections: Float64[NDArray, "K B A O"] = np.zeros(
            (nkpts, nbands, natoms, N_ORBITALS), dtype=np.float64
        )
        i += 1

        k_re: str = (
            r"k-point\s+(\d+)\s*:\s*" r"([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)"
        )
        kpts_found: int = 0
        while i < len(lines) and kpts_found < nkpts:
            k_match: Optional[re.Match[str]] = re.search(k_re, lines[i])
            if k_match is None:
                i += 1
                continue
            k_idx: int = int(k_match.group(1)) - 1
            i += 1
            for b in range(nbands):
                while i < len(lines) and "band" not in lines[i]:
                    i += 1
                i += 1
                i += 1
                for a in range(natoms):
                    vals: List[float] = [float(x) for x in lines[i].split()]
                    projections[k_idx, b, a, :] = vals[1 : N_ORBITALS + 1]
                    i += 1
                i += 1
                i += 1
            kpts_found += 1

        blocks.append(
            {
                "nkpts": nkpts,
                "nbands": nbands,
                "natoms": natoms,
                "projections": projections,
            }
        )

    return blocks


__all__: list[str] = [
    "read_procar",
]
