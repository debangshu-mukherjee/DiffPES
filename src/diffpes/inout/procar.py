"""VASP PROCAR file parser.

Extended Summary
----------------
Reads VASP PROCAR files containing orbital-resolved band
projections and returns an
:class:`~diffpes.types.OrbitalProjection` PyTree. Supports
non-spin, spin-polarized (ISPIN=2), and SOC layouts.

Routine Listings
----------------
:func:`read_procar`
    Parse a VASP PROCAR file into an OrbitalProjection.

Notes
-----
Orbital ordering follows VASP convention:
``[s, py, pz, px, dxy, dyz, dz2, dxz, dx2-y2]``.
"""

import re
from pathlib import Path

import jax.numpy as jnp
import numpy as np
from beartype.typing import Literal, Optional, Union
from jaxtyping import Array, Float
from numpy import ndarray as NDArray  # noqa: N812

from diffpes.types import (
    OrbitalProjection,
    SpinOrbitalProjection,
    make_orbital_projection,
    make_spin_orbital_projection,
)

_NORBS: int = 9
_NSPIN_COMPONENTS: int = 6
_ISPIN2_BLOCKS: int = 2
_SOC_BLOCKS: int = 4


def read_procar(
    filename: str = "PROCAR",
    return_mode: Literal["legacy", "full"] = "legacy",
) -> Union[OrbitalProjection, SpinOrbitalProjection]:
    r"""Parse a VASP PROCAR file.

    Reads a VASP PROCAR file that contains the orbital-resolved
    projections of Kohn-Sham wave functions onto site-centred
    spherical harmonics. Supports three layouts:

    - **Non-spin** (ISPIN=1, no SOC): single block of k-points.
    - **Spin-polarized** (ISPIN=2): two consecutive blocks of
      k-points (one per spin channel).
    - **SOC** (LSORBIT=.TRUE.): four consecutive blocks per k-point
      (total, Sx, Sy, Sz projections).

    Extended Summary
    ----------------
    The PROCAR file written by VASP (when ``LORBIT=11`` or ``12``)
    contains site- and orbital-resolved projections of each Kohn-Sham
    eigenstate. The file is organised into one or more *blocks*, each
    containing:

    * A header line with ``# of k-points``, ``# of bands``,
      ``# of ions``.
    * For each k-point: a coordinate line, then for each band: a
      band-energy line, an orbital-name header, one projection line
      per atom (columns: ion index, s, py, pz, px, dxy, dyz, dz2,
      dxz, dx2-y2, tot), a ``tot`` summation line, and a blank line.

    The number of blocks determines the spin layout:

    * 1 block: non-spin-polarized (ISPIN=1).
    * 2 blocks: spin-polarized (ISPIN=2), block 0 = spin-up,
      block 1 = spin-down.
    * 4 blocks: spin-orbit coupling (SOC), blocks = total, Sx, Sy, Sz.

    Implementation Logic
    --------------------
    1. **Read entire file** into a single string and delegate to
       :func:`_parse_procar_blocks` to extract structured block data.

    2. **Determine spin layout** from the number of blocks.

    3. **Legacy mode or non-spin**: return an ``OrbitalProjection``
       wrapping only the first block's projection array (shape
       ``(K, B, A, 9)``).

    4. **Spin-polarized (ISPIN=2), full mode**:

       a. Average spin-up and spin-down projections to get a
          spin-averaged orbital weight per atom.
       b. Construct a ``(K, B, A, 6)`` spin texture array where
          channels 4-5 encode ``Sz+`` and ``Sz-`` (the positive and
          negative parts of the spin-up minus spin-down difference
          summed over orbitals).
       c. Return a ``SpinOrbitalProjection``.

    5. **SOC (4 blocks), full mode**:

       a. Use the total-projection block (block 0) as the orbital
          weights.
       b. Construct a ``(K, B, A, 6)`` spin texture array where
          channels 0-1 encode ``Sx+`` / ``Sx-``, 2-3 encode
          ``Sy+`` / ``Sy-``, and 4-5 encode ``Sz+`` / ``Sz-``.
          Each component is the positive/negative part of the
          orbital-summed projection from the corresponding
          spin block.
       c. Return a ``SpinOrbitalProjection``.

    Parameters
    ----------
    filename : str, optional
        Path to PROCAR file. Default is ``"PROCAR"``.
    return_mode : {"legacy", "full"}, optional
        ``"legacy"`` (default) returns an ``OrbitalProjection``
        from the first spin block only (backward-compatible).
        ``"full"`` returns a ``SpinOrbitalProjection`` (with
        mandatory spin field) for ISPIN=2 and SOC data, or an
        ``OrbitalProjection`` for non-spin data.

    Returns
    -------
    orb_proj : OrbitalProjection or SpinOrbitalProjection
        ``OrbitalProjection`` for legacy mode or non-spin data.
        ``SpinOrbitalProjection`` for full mode with spin data.

    Raises
    ------
    ValueError
        If no valid PROCAR blocks are found in the file.

    Notes
    -----
    The 9 orbital channels follow the VASP convention:
    ``[s, py, pz, px, dxy, dyz, dz2, dxz, dx2-y2]``.
    The ``tot`` column printed by VASP is **not** stored; only the
    individual orbital columns are kept. For ISPIN=2 in full mode, the
    spin-averaged projection ``(up + down) / 2`` is used as the
    orbital weight because many downstream consumers expect a single
    projection array rather than separate spin channels. The spin
    texture is encoded as 6 non-negative channels
    ``[Sx+, Sx-, Sy+, Sy-, Sz+, Sz-]`` following the convention used
    by the ARPES simulation pipeline.
    """
    path: Path = Path(filename)
    with path.open("r") as fid:
        content: str = fid.read()

    blocks: list[dict] = _parse_procar_blocks(content)

    if not blocks:
        msg = "No valid PROCAR blocks found."
        raise ValueError(msg)

    nblocks: int = len(blocks)
    nkpts: int = blocks[0]["nkpts"]
    nbands: int = blocks[0]["nbands"]
    natoms: int = blocks[0]["natoms"]

    is_spin_polarized: bool = nblocks == _ISPIN2_BLOCKS
    is_soc: bool = nblocks == _SOC_BLOCKS

    if return_mode == "legacy" or (not is_spin_polarized and not is_soc):
        proj_arr: Float[Array, " K B A 9"] = jnp.asarray(
            blocks[0]["projections"], dtype=jnp.float64
        )
        return make_orbital_projection(projections=proj_arr)

    if is_spin_polarized:
        proj_up: Float[NDArray, "K B A O"] = blocks[0]["projections"]
        proj_down: Float[NDArray, "K B A O"] = blocks[1]["projections"]
        avg: Float[NDArray, "K B A O"] = (proj_up + proj_down) / 2.0
        proj_arr = jnp.asarray(avg, dtype=jnp.float64)
        spin_data: Float[NDArray, "K B A 6"] = np.zeros(
            (nkpts, nbands, natoms, _NSPIN_COMPONENTS), dtype=np.float64
        )
        sz_diff: Float[NDArray, "K B A"] = np.sum(
            proj_up - proj_down, axis=-1
        )
        spin_data[:, :, :, 4] = np.maximum(sz_diff, 0.0)
        spin_data[:, :, :, 5] = np.maximum(-sz_diff, 0.0)
        spin_arr: Float[Array, " K B A 6"] = jnp.asarray(
            spin_data, dtype=jnp.float64
        )
        return make_spin_orbital_projection(
            projections=proj_arr, spin=spin_arr
        )

    # SOC: 4 blocks = total, Sx, Sy, Sz
    proj_total: Float[NDArray, "K B A O"] = blocks[0]["projections"]
    proj_sx: Float[NDArray, "K B A O"] = blocks[1]["projections"]
    proj_sy: Float[NDArray, "K B A O"] = blocks[2]["projections"]
    proj_sz: Float[NDArray, "K B A O"] = blocks[3]["projections"]
    proj_arr = jnp.asarray(proj_total, dtype=jnp.float64)

    spin_data = np.zeros(
        (nkpts, nbands, natoms, _NSPIN_COMPONENTS), dtype=np.float64
    )
    sx_sum: Float[NDArray, "K B A"] = np.sum(proj_sx, axis=-1)
    sy_sum: Float[NDArray, "K B A"] = np.sum(proj_sy, axis=-1)
    sz_sum: Float[NDArray, "K B A"] = np.sum(proj_sz, axis=-1)
    spin_data[:, :, :, 0] = np.maximum(sx_sum, 0.0)
    spin_data[:, :, :, 1] = np.maximum(-sx_sum, 0.0)
    spin_data[:, :, :, 2] = np.maximum(sy_sum, 0.0)
    spin_data[:, :, :, 3] = np.maximum(-sy_sum, 0.0)
    spin_data[:, :, :, 4] = np.maximum(sz_sum, 0.0)
    spin_data[:, :, :, 5] = np.maximum(-sz_sum, 0.0)
    spin_arr = jnp.asarray(spin_data, dtype=jnp.float64)
    return make_spin_orbital_projection(projections=proj_arr, spin=spin_arr)


def _parse_procar_blocks(
    content: str,
) -> list[dict]:
    """Parse all PROCAR blocks from the full file content string.

    Extended Summary
    ----------------
    A PROCAR file may contain 1, 2, or 4 consecutive blocks depending
    on the spin configuration. Each block begins with a header line
    matching ``"# of k-points: K  # of bands: B  # of ions: A"`` and
    is followed by nested k-point / band / atom projection data.

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

          i.   Skip forward until a line containing ``"band"`` is
               found (the band energy header).
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
    blocks : list[dict]
        List of parsed blocks. Each dict contains:

        * ``'nkpts'`` (int): number of k-points.
        * ``'nbands'`` (int): number of bands.
        * ``'natoms'`` (int): number of atoms (ions).
        * ``'projections'`` (np.ndarray): orbital projections with
          shape ``(nkpts, nbands, natoms, 9)`` and dtype ``float64``.

    Notes
    -----
    The parser uses 1-based k-point indices from the file to place
    data into the 0-based NumPy array (``k_idx = parsed_index - 1``).
    Band and atom lines are read positionally (sequential order) rather
    than by parsed index. The ``tot`` column (column 10 in the PROCAR
    line) is not stored.
    """
    blocks: list[dict] = []
    lines: list[str] = content.splitlines()
    i: int = 0

    while i < len(lines):
        if "k-points" not in lines[i]:
            i += 1
            continue
        header: str = lines[i]
        params: list[int] = [int(x) for x in re.findall(r"\d+", header)]
        nkpts: int = params[0]
        nbands: int = params[1]
        natoms: int = params[2]
        projections: Float[NDArray, "K B A O"] = np.zeros(
            (nkpts, nbands, natoms, _NORBS), dtype=np.float64
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
                i += 1  # skip band header
                i += 1  # skip orbital-name header
                for a in range(natoms):
                    vals: list[float] = [float(x) for x in lines[i].split()]
                    projections[k_idx, b, a, :] = vals[1 : _NORBS + 1]
                    i += 1
                i += 1  # skip tot line
                i += 1  # skip blank line
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
