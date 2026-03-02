"""Parser-adjacent workflow helpers for assembling simulation-ready arrays.

Provides utilities for atom-subset aggregation, orbital channel
reductions, and cross-file consistency checks between EIGENVAL,
PROCAR, and KPOINTS parsed data.

Routine Listings
----------------
:func:`select_atoms`
    Extract orbital projections for a subset of atoms.
:func:`aggregate_atoms`
    Sum orbital projections over a set of atoms.
:func:`reduce_orbitals`
    Reduce 9 orbital channels to s/p/d totals.
:func:`check_consistency`
    Check dimension agreement across parsed VASP files.
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

    Extended Summary
    ----------------
    Creates a new projection object containing only the atoms at the
    requested indices. This is the primary mechanism for isolating the
    contribution of specific atomic sites (e.g. surface atoms, a
    particular element) to the simulated ARPES spectrum.

    Implementation Logic
    --------------------
    1. Convert ``atom_indices`` to a JAX ``int32`` index array.
    2. Fancy-index the ``projections`` array along axis 2 (atom axis)
       to extract the requested atoms.
    3. If ``spin`` data is present (non-``None``), apply the same
       fancy-indexing to the spin array.
    4. If ``oam`` (orbital angular momentum) data is present, apply
       the same fancy-indexing.
    5. Construct and return a new object of the same type as the input
       (``SpinOrbitalProjection`` or ``OrbitalProjection``) so that
       downstream code can treat the result identically to the
       original.

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

    Notes
    -----
    The returned object shares no memory with the original because
    JAX fancy-indexing always produces a copy. The function is
    pure and can be safely used inside ``jax.jit``-compiled code.
    """
    idx: Int[Array, " N"] = jnp.asarray(atom_indices, dtype=jnp.int32)
    proj_sub: Float[Array, "K B N 9"] = orb.projections[:, :, idx, :]
    spin_sub: Optional[Float[Array, "K B N 9"]] = None
    if orb.spin is not None:
        spin_sub = orb.spin[:, :, idx, :]
    oam_sub: Optional[Float[Array, "K B N 9"]] = None
    if orb.oam is not None:
        oam_sub = orb.oam[:, :, idx, :]
    if isinstance(orb, SpinOrbitalProjection):
        result_soc: SpinOrbitalProjection = SpinOrbitalProjection(
            projections=proj_sub,
            spin=spin_sub,  # type: ignore[arg-type]
            oam=oam_sub,
        )
        return result_soc
    result: OrbitalProjection = OrbitalProjection(
        projections=proj_sub,
        spin=spin_sub,
        oam=oam_sub,
    )
    return result


def aggregate_atoms(
    orb: OrbitalProjection,
    atom_indices: Optional[list[int]] = None,
) -> Float[Array, "K B 9"]:
    """Sum orbital projections over a set of atoms.

    Extended Summary
    ----------------
    Produces a ``(K, B, 9)`` array where the atom axis has been
    summed out, giving the total orbital weight at each (k-point,
    band) pair. This is the standard reduction used before computing
    cross-section-weighted ARPES intensities, because the simulation
    only needs the aggregate orbital character rather than per-atom
    contributions.

    Implementation Logic
    --------------------
    1. If ``atom_indices`` is not ``None``, fancy-index the
       ``projections`` array along axis 2 to restrict to the
       requested atoms before summing.
    2. If ``atom_indices`` is ``None``, use the full projections
       array (all atoms).
    3. Sum along axis 2 (the atom axis) using ``jnp.sum`` and
       return the resulting ``(K, B, 9)`` array.

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

    Notes
    -----
    This function operates only on the ``projections`` field and
    ignores ``spin`` and ``oam`` fields. For spin-resolved
    aggregation, use :func:`select_atoms` first and then perform
    the reduction manually.
    """
    if atom_indices is not None:
        idx: Int[Array, " N"] = jnp.asarray(atom_indices, dtype=jnp.int32)
        proj: Float[Array, "K B N 9"] = orb.projections[:, :, idx, :]
    else:
        proj = orb.projections
    result: Float[Array, "K B 9"] = jnp.sum(proj, axis=2)
    return result


def reduce_orbitals(
    projections: Float[Array, "K B A 9"],
) -> Float[Array, "K B A 3"]:
    """Reduce 9 orbital channels to s/p/d totals.

    Extended Summary
    ----------------
    Collapses the 9-channel VASP orbital decomposition
    (``s, py, pz, px, dxy, dyz, dz2, dxz, dx2-y2``) into three
    angular-momentum shell totals. This is useful for coarse-grained
    orbital-character analysis (e.g. fat-band coloring by s/p/d
    weight).

    Implementation Logic
    --------------------
    1. Extract the ``s`` channel (index 0) as-is -- shape ``(K, B, A)``.
    2. Sum the three ``p`` channels (indices 1:4) along the last axis.
    3. Sum the five ``d`` channels (indices 4:9) along the last axis.
    4. Stack ``[s_total, p_total, d_total]`` along a new trailing axis
       to produce shape ``(K, B, A, 3)``.

    Parameters
    ----------
    projections : Float[Array, "K B A 9"]
        Full 9-channel orbital projections.

    Returns
    -------
    Float[Array, "K B A 3"]
        Reduced projections: ``[s_total, p_total, d_total]``.

    Notes
    -----
    The VASP orbital ordering assumed here is:
    ``[s, py, pz, px, dxy, dyz, dz2, dxz, dx2-y2]`` (indices 0-8).
    This matches the standard PROCAR output when ``LORBIT=11`` or
    ``LORBIT=12``.
    """
    s_total: Float[Array, "K B A"] = projections[..., _S_IDX]
    p_total: Float[Array, "K B A"] = jnp.sum(
        projections[..., _P_SLICE], axis=-1
    )
    d_total: Float[Array, "K B A"] = jnp.sum(
        projections[..., _D_SLICE], axis=-1
    )
    reduced: Float[Array, "K B A 3"] = jnp.stack(
        [s_total, p_total, d_total], axis=-1
    )
    return reduced


def check_consistency(
    bands: BandStructure,
    orb: OrbitalProjection,
    kpath: Optional[KPathInfo] = None,
) -> None:
    """Check dimension agreement across parsed VASP files.

    Extended Summary
    ----------------
    Validates that the k-point and band dimensions are consistent
    between the EIGENVAL-derived band structure, the PROCAR-derived
    orbital projections, and (optionally) the KPOINTS-derived path
    metadata. This is a defensive check intended to catch mismatches
    caused by mixing output files from different VASP runs.

    Implementation Logic
    --------------------
    1. Extract ``nkpoints`` and ``nbands`` from ``bands.eigenvalues``
       (shape ``(K, B)``) and ``orb.projections`` (shape
       ``(K, B, A, 9)``).
    2. Compare the k-point counts between EIGENVAL and PROCAR. Raise
       ``ValueError`` with a descriptive message if they disagree.
    3. Compare the band counts between EIGENVAL and PROCAR. Raise
       ``ValueError`` if they disagree.
    4. If ``kpath`` is provided and its mode is ``"Line-mode"``,
       compare the total k-point count from KPOINTS
       (``kpath.num_kpoints``) against the EIGENVAL k-point count.
       Only check when ``num_kpoints > 0`` (i.e. the KPOINTS file
       provides a concrete count).

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

    Notes
    -----
    This function does not verify atom counts (PROCAR vs POSCAR)
    because ``BandStructure`` does not carry atom information.
    For atom-count checks, compare ``orb.projections.shape[2]``
    against ``geometry.atom_counts.sum()`` manually.
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
