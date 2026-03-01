"""VASP PROCAR file parser.

Extended Summary
----------------
Reads VASP PROCAR files containing orbital-resolved band
projections and returns an
:class:`~arpyes.types.OrbitalProjection` PyTree. Supports
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

from arpyes.types import (
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
        proj_up: np.ndarray = blocks[0]["projections"]
        proj_down: np.ndarray = blocks[1]["projections"]
        avg: np.ndarray = (proj_up + proj_down) / 2.0
        proj_arr = jnp.asarray(avg, dtype=jnp.float64)
        spin_data: np.ndarray = np.zeros(
            (nkpts, nbands, natoms, _NSPIN_COMPONENTS), dtype=np.float64
        )
        sz_diff: np.ndarray = np.sum(proj_up - proj_down, axis=-1)
        spin_data[:, :, :, 4] = np.maximum(sz_diff, 0.0)
        spin_data[:, :, :, 5] = np.maximum(-sz_diff, 0.0)
        spin_arr: Float[Array, " K B A 6"] = jnp.asarray(
            spin_data, dtype=jnp.float64
        )
        return make_spin_orbital_projection(
            projections=proj_arr, spin=spin_arr
        )

    # SOC: 4 blocks = total, Sx, Sy, Sz
    proj_total: np.ndarray = blocks[0]["projections"]
    proj_sx: np.ndarray = blocks[1]["projections"]
    proj_sy: np.ndarray = blocks[2]["projections"]
    proj_sz: np.ndarray = blocks[3]["projections"]
    proj_arr = jnp.asarray(proj_total, dtype=jnp.float64)

    spin_data = np.zeros(
        (nkpts, nbands, natoms, _NSPIN_COMPONENTS), dtype=np.float64
    )
    sx_sum: np.ndarray = np.sum(proj_sx, axis=-1)
    sy_sum: np.ndarray = np.sum(proj_sy, axis=-1)
    sz_sum: np.ndarray = np.sum(proj_sz, axis=-1)
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
    """Parse all PROCAR blocks from file content.

    Each block starts with a header line containing k-points count,
    followed by k-point/band/atom data. Returns a list of dicts,
    each with keys 'nkpts', 'nbands', 'natoms', 'projections'.
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
        projections: np.ndarray = np.zeros(
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
