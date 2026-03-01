"""Parser-adjacent workflow helpers for assembling simulation-ready arrays.

Provides utilities for atom-subset aggregation, orbital channel
reductions, and cross-file consistency checks between EIGENVAL,
PROCAR, and KPOINTS parsed data.
"""

import jax.numpy as jnp
from beartype.typing import Optional, Union
from jaxtyping import Array, Float, Int

from arpyes.types import (
    BandStructure,
    KPathInfo,
    OrbitalProjection,
    SpinOrbitalProjection,
)

_S_IDX: int = 0
_P_SLICE: slice = slice(1, 4)
_D_SLICE: slice = slice(4, 9)


def select_atoms(
    orb: Union[OrbitalProjection, SpinOrbitalProjection],
    atom_indices: list[int],
) -> Union[OrbitalProjection, SpinOrbitalProjection]:
    """Extract orbital projections for a subset of atoms.

    Parameters
    ----------
    orb : OrbitalProjection or SpinOrbitalProjection
        Full orbital projections with shape ``(K, B, A, 9)``.
    atom_indices : list[int]
        0-based indices of atoms to select.

    Returns
    -------
    OrbitalProjection or SpinOrbitalProjection
        Projections restricted to the specified atoms.
        Shape ``(K, B, len(atom_indices), 9)``.
        Preserves the input type.
    """
    idx: Int[Array, " N"] = jnp.asarray(atom_indices, dtype=jnp.int32)
    proj_sub: Float[Array, "K B N 9"] = orb.projections[:, :, idx, :]
    spin_sub = None
    if orb.spin is not None:
        spin_sub = orb.spin[:, :, idx, :]
    oam_sub = None
    if orb.oam is not None:
        oam_sub = orb.oam[:, :, idx, :]
    if isinstance(orb, SpinOrbitalProjection):
        return SpinOrbitalProjection(
            projections=proj_sub,
            spin=spin_sub,  # type: ignore[arg-type]
            oam=oam_sub,
        )
    return OrbitalProjection(
        projections=proj_sub,
        spin=spin_sub,
        oam=oam_sub,
    )


def aggregate_atoms(
    orb: OrbitalProjection,
    atom_indices: Optional[list[int]] = None,
) -> Float[Array, "K B 9"]:
    """Sum orbital projections over a set of atoms.

    Parameters
    ----------
    orb : OrbitalProjection
        Full orbital projections with shape ``(K, B, A, 9)``.
    atom_indices : list[int] or None, optional
        0-based indices of atoms to sum over. If None, sums over
        all atoms.

    Returns
    -------
    Float[Array, "K B 9"]
        Atom-summed orbital projections.
    """
    if atom_indices is not None:
        idx = jnp.asarray(atom_indices, dtype=jnp.int32)
        proj = orb.projections[:, :, idx, :]
    else:
        proj = orb.projections
    return jnp.sum(proj, axis=2)


def reduce_orbitals(
    projections: Float[Array, "K B A 9"],
) -> Float[Array, "K B A 3"]:
    """Reduce 9 orbital channels to s/p/d totals.

    Parameters
    ----------
    projections : Float[Array, "K B A 9"]
        Full 9-channel orbital projections.

    Returns
    -------
    Float[Array, "K B A 3"]
        Reduced projections: ``[s_total, p_total, d_total]``.
    """
    s_total: Float[Array, "K B A"] = projections[..., _S_IDX]
    p_total: Float[Array, "K B A"] = jnp.sum(
        projections[..., _P_SLICE], axis=-1
    )
    d_total: Float[Array, "K B A"] = jnp.sum(
        projections[..., _D_SLICE], axis=-1
    )
    return jnp.stack([s_total, p_total, d_total], axis=-1)


def check_consistency(
    bands: BandStructure,
    orb: OrbitalProjection,
    kpath: Optional[KPathInfo] = None,
) -> None:
    """Check dimension agreement across parsed VASP files.

    Parameters
    ----------
    bands : BandStructure
        Parsed EIGENVAL data.
    orb : OrbitalProjection
        Parsed PROCAR data.
    kpath : KPathInfo or None, optional
        Parsed KPOINTS data.

    Raises
    ------
    ValueError
        If k-point or band counts disagree between files.
    """
    nk_bands: int = int(bands.eigenvalues.shape[0])
    nb_bands: int = int(bands.eigenvalues.shape[1])
    nk_procar: int = int(orb.projections.shape[0])
    nb_procar: int = int(orb.projections.shape[1])

    if nk_bands != nk_procar:
        msg = (
            f"K-point count mismatch: EIGENVAL has {nk_bands}, "
            f"PROCAR has {nk_procar}."
        )
        raise ValueError(msg)

    if nb_bands != nb_procar:
        msg = (
            f"Band count mismatch: EIGENVAL has {nb_bands}, "
            f"PROCAR has {nb_procar}."
        )
        raise ValueError(msg)

    if kpath is not None and kpath.mode == "Line-mode":
        nk_kpath: int = int(kpath.num_kpoints)
        if nk_kpath > 0 and nk_bands != nk_kpath:
            msg = (
                f"K-point count mismatch: EIGENVAL has {nk_bands}, "
                f"KPOINTS has {nk_kpath}."
            )
            raise ValueError(msg)


__all__: list[str] = [
    "aggregate_atoms",
    "check_consistency",
    "reduce_orbitals",
    "select_atoms",
]
