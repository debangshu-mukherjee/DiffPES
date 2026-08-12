"""Provide workflow helpers for simulation-ready parser arrays.

Extended Summary
----------------
The module provides utilities for atom-subset aggregation, orbital channel
reductions, and cross-file consistency checks between EIGENVAL,
PROCAR, and KPOINTS parsed data.

Routine Listings
----------------
:func:`aggregate_atoms`
    Sum orbital projections over a set of atoms.
:func:`check_consistency`
    Check dimension agreement across parsed VASP files.
:func:`reduce_orbitals`
    Reduce 9 orbital channels to s/p/d totals.
:func:`select_atoms`
    Extract orbital projections for a subset of atoms.
"""

import jax.numpy as jnp
from beartype import beartype
from beartype.typing import List, Optional, Union
from jaxtyping import Array, Float64, Int32, jaxtyped

from diffpes.constants import (
    D_ORBITAL_SLICE,
    P_ORBITAL_SLICE,
    S_IDX,
)
from diffpes.types import (
    BandStructure,
    KPathInfo,
    OrbitalProjection,
    SpinOrbitalProjection,
    make_orbital_projection,
    make_spin_orbital_projection,
)


@jaxtyped(typechecker=beartype)
def select_atoms(
    orb: Union[OrbitalProjection, SpinOrbitalProjection],
    atom_indices: List[int],
) -> Union[OrbitalProjection, SpinOrbitalProjection]:
    """Extract orbital projections for a subset of atoms.

    The function creates a projection object that contains only the requested
    atoms. Use it to isolate contributions from specified sites. Examples
    include surface atoms and atoms of one element.

    :see: :class:`~.test_helpers.TestSelectAtoms`

    Implementation Logic
    --------------------
    1. **Build the atom index**::

           idx = jnp.asarray(atom_indices, dtype=jnp.int32)

       The JAX index keeps the selection compatible with traced array access.
    2. **Select each present array leaf**::

           proj_sub = orb.projections[:, :, idx, :]
           spin_sub = orb.spin[:, :, idx, :]
           oam_sub = orb.oam[:, :, idx, :]

       Each selection uses the same atom axis and preserves the other axes.
    3. **Preserve the carrier type**::

           result = SpinOrbitalProjection(...)
           result = OrbitalProjection(...)

       The branch returns the same projection carrier variant as the input.

    Parameters
    ----------
    orb : Union[OrbitalProjection, SpinOrbitalProjection]
        Full orbital projections with shape ``(K, B, A, 9)``.
    atom_indices : List[int]
        0-based indices of atoms to select.

    Returns
    -------
    result : Union[OrbitalProjection, SpinOrbitalProjection]
        Projections restricted to the specified atoms.
        Shape ``(K, B, len(atom_indices), 9)``.
        Preserves the input type.

    Notes
    -----
    The returned object shares no memory with the original because JAX
    advanced indexing always produces a copy. The pure function works inside
    code that ``jax.jit`` compiles.
    """
    idx: Int32[Array, " N"] = jnp.asarray(atom_indices, dtype=jnp.int32)
    proj_sub: Float64[Array, "K B N 9"] = orb.projections[:, :, idx, :]
    spin_sub: Optional[Float64[Array, "K B N 6"]] = None
    if orb.spin is not None:
        spin_sub = orb.spin[:, :, idx, :]
    oam_sub: Optional[Float64[Array, "K B N 3"]] = None
    if orb.oam is not None:
        oam_sub = orb.oam[:, :, idx, :]
    if isinstance(orb, SpinOrbitalProjection):
        result: Union[OrbitalProjection, SpinOrbitalProjection] = (
            make_spin_orbital_projection(
                projections=proj_sub,
                spin=orb.spin[:, :, idx, :],
                oam=oam_sub,
            )
        )
    else:
        result = make_orbital_projection(
            projections=proj_sub,
            spin=spin_sub,
            oam=oam_sub,
        )
    return result


@jaxtyped(typechecker=beartype)
def aggregate_atoms(
    orb: OrbitalProjection,
    atom_indices: Optional[List[int]] = None,
) -> Float64[Array, "K B 9"]:
    """Sum orbital projections over a set of atoms.

    The function sums the atom axis and produces a ``(K, B, 9)`` array.
    Therefore, each k-point and band pair contains the total orbital weight.
    The ARPES intensity computation uses this reduction because it needs the
    aggregate orbital character instead of individual atom contributions.

    :see: :class:`~.test_helpers.TestAggregateAtoms`

    Implementation Logic
    --------------------
    1. **Select the requested atom data**::

           proj = orb.projections[:, :, idx, :]
           proj = orb.projections

       The optional index limits the reduction to the requested atoms.
    2. **Sum the atom axis**::

           result = jnp.sum(proj, axis=2)

       The reduction keeps the k-point, band, and orbital axes explicit.

    Parameters
    ----------
    orb : OrbitalProjection
        Full orbital projections with shape ``(K, B, A, 9)``.
    atom_indices : Optional[List[int]], optional
        0-based indices of atoms to sum over. If None, sums over
        all atoms.

    Returns
    -------
    result : Float64[Array, "K B 9"]
        Atom-summed orbital projections.

    Notes
    -----
    This function operates only on the ``projections`` field and
    ignores ``spin`` and ``oam`` fields. For spin-resolved
    aggregation, use :func:`select_atoms` first and then perform
    the reduction manually.
    """
    if atom_indices is not None:
        idx: Int32[Array, " N"] = jnp.asarray(atom_indices, dtype=jnp.int32)
        proj: Float64[Array, "K B N 9"] = orb.projections[:, :, idx, :]
    else:
        proj = orb.projections
    result: Float64[Array, "K B 9"] = jnp.sum(proj, axis=2)
    return result


@jaxtyped(typechecker=beartype)
def reduce_orbitals(
    projections: Float64[Array, "K B A 9"],
) -> Float64[Array, "K B A 3"]:
    """Reduce 9 orbital channels to s/p/d totals.

    Collapses the 9-channel VASP orbital decomposition
    (``s, py, pz, px, dxy, dyz, dz2, dxz, dx2-y2``) into three
    angular-momentum shell totals. This is useful for coarse-grained
    orbital-character analysis, for example fat-band colors from s/p/d
    weight).

    :see: :class:`~.test_helpers.TestReduceOrbitals`

    Implementation Logic
    --------------------
    1. **Compute shell totals**::

           s_total = projections[..., S_IDX]
           p_total = jnp.sum(projections[..., P_ORBITAL_SLICE], axis=-1)
           d_total = jnp.sum(projections[..., D_ORBITAL_SLICE], axis=-1)

       The fixed slices apply the public VASP orbital ordering.
    2. **Stack the shell axis**::

           reduced = jnp.stack([s_total, p_total, d_total], axis=-1)

       The new trailing axis stores the s, p, and d totals in that order.

    Parameters
    ----------
    projections : Float64[Array, "K B A 9"]
        Full 9-channel orbital projections.

    Returns
    -------
    reduced : Float64[Array, "K B A 3"]
        Reduced projections: ``[s_total, p_total, d_total]``.

    Notes
    -----
    The VASP orbital ordering assumed here is:
    ``[s, py, pz, px, dxy, dyz, dz2, dxz, dx2-y2]`` (indices 0-8).
    This matches the standard PROCAR output when ``LORBIT=11`` or
    ``LORBIT=12``.
    """
    s_total: Float64[Array, "K B A"] = projections[..., S_IDX]
    p_total: Float64[Array, "K B A"] = jnp.sum(
        projections[..., P_ORBITAL_SLICE], axis=-1
    )
    d_total: Float64[Array, "K B A"] = jnp.sum(
        projections[..., D_ORBITAL_SLICE], axis=-1
    )
    reduced: Float64[Array, "K B A 3"] = jnp.stack(
        [s_total, p_total, d_total], axis=-1
    )
    return reduced


@jaxtyped(typechecker=beartype)
def check_consistency(
    bands: BandStructure,
    orb: Union[OrbitalProjection, SpinOrbitalProjection],
    kpath: Optional[KPathInfo] = None,
) -> None:
    """Check dimension agreement across parsed VASP files.

    Validates that the k-point and band dimensions are consistent
    between the EIGENVAL-derived band structure, the PROCAR-derived
    orbital projections, and (optionally) the KPOINTS-derived path
    metadata. This is a defensive check intended to catch mismatches
    caused by mixing output files from different VASP runs.

    :see: :class:`~.test_helpers.TestCheckConsistency`

    Implementation Logic
    --------------------
    1. **Read the shared dimensions**::

           nk_bands = int(bands.eigenvalues.shape[0])
           nb_bands = int(bands.eigenvalues.shape[1])
           nk_procar = int(orb.projections.shape[0])
           nb_procar = int(orb.projections.shape[1])

       These static dimensions identify incompatible parser outputs early.
    2. **Compare the band and k-point axes**::

           if nk_bands != nk_procar:
           if nb_bands != nb_procar:

       Each mismatch raises ``ValueError`` with both observed sizes.
    3. **Check optional line-mode metadata**::

           if nk_kpath > 0 and nk_bands != nk_kpath:

       A positive KPOINTS count must agree with the EIGENVAL count.

    Parameters
    ----------
    bands : BandStructure
        Parsed EIGENVAL data.
    orb : Union[OrbitalProjection, SpinOrbitalProjection]
        Parsed PROCAR data.
    kpath : Optional[KPathInfo], optional
        Parsed KPOINTS data.

    Raises
    ------
    ValueError
        If k-point or band counts disagree between files.

    Notes
    -----
    This function does not verify atom counts (PROCAR vs POSCAR)
    because ``BandStructure`` does not carry atom information.
    For atom-count checks, compare ``orb.projections.shape[2]`` against
    ``geometry.positions.shape[0]`` manually.
    """
    nk_bands: int = int(bands.eigenvalues.shape[0])
    nb_bands: int = int(bands.eigenvalues.shape[1])
    nk_procar: int = int(orb.projections.shape[0])
    nb_procar: int = int(orb.projections.shape[1])

    if nk_bands != nk_procar:
        msg: str = (
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
