"""Differentiable band diagonalization and VASP adapter.

Extended Summary
----------------
Wraps ``jnp.linalg.eigh`` with vmap over k-points to produce
a ``DiagonalizedBands`` PyTree from a ``TBModel``. Also provides
an adapter to convert VASP ``BandStructure`` + ``OrbitalProjection``
to the same interface.

Routine Listings
----------------
:func:`diagonalize_single_k`
    Diagonalize H(k) at a single k-point.
:func:`diagonalize_tb`
    Diagonalize a TB model at all k-points.
:func:`vasp_to_diagonalized`
    Convert VASP BandStructure + OrbitalProjection to DiagonalizedBands.
"""

import jax
import jax.numpy as jnp
from beartype import beartype
from beartype.typing import Literal
from jaxtyping import Array, Complex, Float, Int, jaxtyped

from arpyes.types import (
    BandStructure,
    DiagonalizedBands,
    OrbitalBasis,
    OrbitalProjection,
    TBModel,
    make_diagonalized_bands,
)

from .hamiltonian import build_hamiltonian_k

_PHASE_LOSS_MESSAGE: str = (
    "vasp_to_diagonalized uses sqrt(|c|^2) magnitudes from PROCAR and "
    "cannot recover complex eigenvector phases. Matrix elements that "
    "depend on phase interference are approximate."
)


@jaxtyped(typechecker=beartype)
def diagonalize_single_k(
    H_k: Complex[Array, "O O"],
) -> tuple[Float[Array, " O"], Complex[Array, "O O"]]:
    """Diagonalize H(k) at a single k-point.

    Extended Summary
    ----------------
    Delegates to ``jnp.linalg.eigh``, which is the standard LAPACK-style
    Hermitian eigensolver.  ``eigh`` is used (not ``eig``) because the
    Hamiltonian is guaranteed Hermitian by construction, which ensures
    real eigenvalues and an orthonormal eigenvector basis.

    ``jnp.linalg.eigh`` returns eigenvalues in **ascending** order and
    eigenvectors as **columns** of the returned matrix: that is,
    ``eigenvectors[:, i]`` is the eigenvector corresponding to
    ``eigenvalues[i]``.  This column-eigenvector convention is the
    LAPACK/NumPy/JAX standard but differs from some physics textbooks
    that store eigenvectors as rows.  The caller (``diagonalize_tb``)
    transposes to a band-major layout after vmapping.

    This function is intentionally thin -- a one-liner around ``eigh``
    -- so that it can be individually tested, mocked, or replaced with
    a custom differentiable eigensolver if needed.

    Parameters
    ----------
    H_k : Complex[Array, "O O"]
        Hermitian Hamiltonian matrix.

    Returns
    -------
    eigenvalues : Float[Array, " O"]
        Eigenvalues in ascending order.
    eigenvectors : Complex[Array, "O O"]
        Eigenvector columns (eigenvectors[:, i] is the i-th).

    Notes
    -----
    JAX provides analytical gradients through ``eigh`` via implicit
    differentiation of the eigenvalue equation, so
    ``jax.grad(lambda p: eigenvalues(p).sum())`` works out of the box.
    Degenerate eigenvalues can cause numerical instability in the
    backward pass; this is a known JAX limitation.
    """
    eigenvalues: Float[Array, " O"]
    eigenvectors: Complex[Array, "O O"]
    eigenvalues, eigenvectors = jnp.linalg.eigh(H_k)
    result: tuple[Float[Array, " O"], Complex[Array, "O O"]] = (
        eigenvalues,
        eigenvectors,
    )
    return result


@jaxtyped(typechecker=beartype)
def diagonalize_tb(
    tb_model: TBModel,
    kpoints: Float[Array, "K 3"],
) -> DiagonalizedBands:
    """Diagonalize a TB model at all k-points.

    Builds H(k) for each k-point and diagonalizes via
    ``jnp.linalg.eigh``. Both operations are vmapped and
    fully differentiable with respect to ``tb_model.hopping_params``.

    Extended Summary
    ----------------
    Internally a closure ``_build_and_diag`` is defined that captures
    the model parameters (hopping amplitudes, indices, lattice vectors)
    and maps a single k-point to its ``(eigenvalues, eigenvectors)``
    pair.  ``jax.vmap`` is then applied over the leading k-point axis,
    so the entire band structure is computed in one vectorized call
    without Python loops.

    After vmapping, the eigenvector array has shape ``(K, O, O)`` in
    the column-eigenvector convention of ``eigh`` (i.e.
    ``evecs[k, :, b]`` is band ``b`` at k-point ``k``).  A transpose
    ``(0, 2, 1)`` converts this to the band-major convention
    ``(K, B, O)`` where ``evecs[k, b, :]`` gives the orbital
    coefficients of band ``b`` at k-point ``k``.  This layout matches
    the ``DiagonalizedBands.eigenvectors`` contract used by the rest
    of the ARPYES pipeline (projections, matrix elements, spectral
    function).

    The Fermi energy is set to 0.0 by default because bare
    tight-binding models have no absolute energy reference.

    Parameters
    ----------
    tb_model : TBModel
        Tight-binding model.
    kpoints : Float[Array, "K 3"]
        k-points in fractional coordinates.

    Returns
    -------
    bands : DiagonalizedBands
        Diagonalized electronic structure.

    Notes
    -----
    Because ``jax.vmap`` traces the function once and broadcasts over
    the batch dimension, the number of k-points does not affect
    compilation time -- only runtime.  The full forward + backward
    pass (eigenvalues and their gradients with respect to hopping
    parameters) is differentiable end-to-end.
    """

    def _build_and_diag(k: Float[Array, " 3"]) -> tuple:
        H: Complex[Array, "O O"] = build_hamiltonian_k(
            k,
            tb_model.hopping_params,
            tb_model.hopping_indices,
            tb_model.n_orbitals,
            tb_model.lattice_vectors,
        )
        evals: Float[Array, " O"]
        evecs: Complex[Array, "O O"]
        evals, evecs = diagonalize_single_k(H)
        return evals, evecs

    eigenvalues: Float[Array, "K O"]
    eigenvectors: Complex[Array, "K O O"]
    eigenvalues, eigenvectors = jax.vmap(_build_and_diag)(kpoints)
    # eigenvectors from eigh: shape (K, O, O) where evecs[:, :, i] is i-th
    # We want (K, B, O) where B=O (number of bands = number of orbitals)
    # Transpose so evecs[k, b, o] = coefficient of orbital o in band b
    eigenvectors = jnp.transpose(eigenvectors, (0, 2, 1))

    bands: DiagonalizedBands = make_diagonalized_bands(
        eigenvalues=eigenvalues,
        eigenvectors=eigenvectors,
        kpoints=kpoints,
        fermi_energy=0.0,
    )
    return bands


@jaxtyped(typechecker=beartype)
def vasp_to_diagonalized(
    bands: BandStructure,
    orb_proj: OrbitalProjection,
    orbital_basis: OrbitalBasis,
    phase_loss: Literal["warn", "ignore", "error"] = "warn",
) -> DiagonalizedBands:
    """Convert VASP BandStructure + OrbitalProjection to DiagonalizedBands.

    VASP PROCAR gives |c_{k,b,orb}|^2, not the complex coefficients.
    This adapter uses sqrt(|c|^2) with positive sign as an
    approximation. Phase information is lost.

    The orbital projections are summed over atoms and mapped to
    the orbital basis ordering.

    Extended Summary
    ----------------
    VASP's PROCAR file provides site- and orbital-projected squared
    moduli ``|c_{k,b,atom,orb}|^2`` for each eigenstate.  Because
    the complex phase of each coefficient is discarded during the VASP
    projection, it is **not possible** to recover the true complex
    eigenvectors from PROCAR data alone.  This adapter applies the
    simplest possible approximation: it takes ``sqrt(|c|^2)`` with a
    uniformly **positive** sign, yielding purely real, non-negative
    approximate coefficients.

    The approximation is exact for any observable that depends only
    on the modulus of the coefficients (e.g. orbital-resolved density
    of states), but it introduces errors in any quantity sensitive to
    relative phases between orbitals (e.g. interference terms in
    photoemission matrix elements).

    **Orbital mapping.** VASP stores projections in a fixed 9-orbital
    ordering for the s, p, and d channels::

        [s, py, pz, px, dxy, dyz, dz2, dxz, dx2-y2]

    This differs from some standard orderings (e.g. the ``(l, m)``
    convention with m running from -l to +l).  The function maps from
    ``(l, m)`` quantum numbers in the ``OrbitalBasis`` to VASP's
    9-orbital column index via a lookup table.

    **Atom summation.** Before the orbital mapping, the projections are
    summed over the atom axis (axis 2), collapsing
    ``(K, B, A, 9) -> (K, B, 9)``.  This is correct when the
    ``OrbitalBasis`` describes a single composite orbital per
    ``(l, m)`` channel rather than per-atom resolution.

    **Normalization.** After taking the square root, each eigenvector
    (per k-point and band) is normalized so that
    ``sum_orb |c_{k,b,orb}|^2 = 1``.  A safe-division guard
    (threshold ``1e-12``) prevents NaN gradients when all projections
    are zero (e.g. core states with no s/p/d weight).

    Parameters
    ----------
    bands : BandStructure
        VASP eigenvalues and k-points.
    orb_proj : OrbitalProjection
        VASP orbital projections of shape (K, B, A, 9).
    orbital_basis : OrbitalBasis
        Quantum number metadata defining which VASP orbital
        indices to use.
    phase_loss : {"warn", "ignore", "error"}, optional
        Policy for handling lost phase information:
        - ``"warn"`` (default): emit a runtime warning.
        - ``"ignore"``: proceed silently.
        - ``"error"``: raise ``ValueError`` and abort.

    Returns
    -------
    diag : DiagonalizedBands
        Approximate diagonalized bands.

    Notes
    -----
    The VASP 9-orbital ordering used here covers s, p, and d channels
    only.  f-orbital projections (l=3) are not supported and will raise
    a ``ValueError``.  If VASP was run with ``LORBIT=11`` or higher,
    the PROCAR contains all nine channels; ``LORBIT=10`` may omit
    per-m decomposition and is not compatible with this adapter.

    The resulting eigenvectors are cast to ``complex128`` for
    compatibility with the ``DiagonalizedBands`` type even though they
    are purely real (zero imaginary part).
    """
    if phase_loss not in ("warn", "ignore", "error"):
        msg: str = (
            "phase_loss must be one of {'warn', 'ignore', 'error'}, "
            f"got '{phase_loss}'"
        )
        raise ValueError(msg)

    if phase_loss == "error":
        raise ValueError(_PHASE_LOSS_MESSAGE)
    if phase_loss == "warn":
        import warnings

        warnings.warn(_PHASE_LOSS_MESSAGE, RuntimeWarning, stacklevel=2)

    # Sum projections over atoms: (K, B, A, 9) -> (K, B, 9)
    proj_summed: Float[Array, "K B 9"] = jnp.sum(orb_proj.projections, axis=2)

    # Map orbital basis to VASP orbital indices
    # VASP ordering: [s, py, pz, px, dxy, dyz, dz2, dxz, dx2-y2]
    vasp_lm_to_idx: dict[tuple[int, int], int] = {
        (0, 0): 0,  # s
        (1, -1): 1,  # py
        (1, 0): 2,  # pz
        (1, 1): 3,  # px
        (2, -2): 4,  # dxy
        (2, -1): 5,  # dyz
        (2, 0): 6,  # dz2
        (2, 1): 7,  # dxz
        (2, 2): 8,  # dx2-y2
    }

    n_orbs: int = len(orbital_basis.l_values)
    orbital_indices: list[int] = []
    for i in range(n_orbs):
        l_val: int = orbital_basis.l_values[i]
        m_val: int = orbital_basis.m_values[i]
        idx: int | None = vasp_lm_to_idx.get((l_val, m_val))
        if idx is None:
            msg = f"Orbital (l={l_val}, m={m_val}) not in VASP 9-orbital set"
            raise ValueError(msg)
        orbital_indices.append(idx)

    # Extract and take sqrt as approximate coefficients
    # proj_summed shape: (K, B, 9), select columns -> (K, B, n_orbs)
    idx_arr: Int[Array, " N"] = jnp.array(orbital_indices)
    approx_c2: Float[Array, "K B N"] = proj_summed[:, :, idx_arr]
    approx_c: Float[Array, "K B N"] = jnp.sqrt(jnp.maximum(approx_c2, 0.0))

    # Normalize eigenvectors per (k, band) so they sum to 1
    norm: Float[Array, "K B 1"] = jnp.sqrt(
        jnp.sum(approx_c**2, axis=-1, keepdims=True)
    )
    safe_norm: Float[Array, "K B 1"] = jnp.where(norm > 1e-12, norm, 1.0)
    eigenvectors: Complex[Array, "K B N"] = (approx_c / safe_norm).astype(
        jnp.complex128
    )

    diag: DiagonalizedBands = make_diagonalized_bands(
        eigenvalues=bands.eigenvalues,
        eigenvectors=eigenvectors,
        kpoints=bands.kpoints,
        fermi_energy=bands.fermi_energy,
    )
    return diag


__all__: list[str] = [
    "diagonalize_single_k",
    "diagonalize_tb",
    "vasp_to_diagonalized",
]
