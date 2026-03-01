"""Differentiable band diagonalization and VASP adapter.

Extended Summary
----------------
Wraps ``jnp.linalg.eigh`` with vmap over k-points to produce
a ``DiagonalizedBands`` PyTree from a ``TBModel``. Also provides
an adapter to convert VASP ``BandStructure`` + ``OrbitalProjection``
to the same interface.
"""

import jax
import jax.numpy as jnp
from beartype import beartype
from jaxtyping import Array, Complex, Float, jaxtyped

from arpyes.types import (
    BandStructure,
    DiagonalizedBands,
    OrbitalBasis,
    OrbitalProjection,
    TBModel,
    make_diagonalized_bands,
)

from .hamiltonian import build_hamiltonian_k


@jaxtyped(typechecker=beartype)
def diagonalize_single_k(
    H_k: Complex[Array, "O O"],
) -> tuple[Float[Array, " O"], Complex[Array, "O O"]]:
    """Diagonalize H(k) at a single k-point.

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
    """
    eigenvalues, eigenvectors = jnp.linalg.eigh(H_k)
    return eigenvalues, eigenvectors


@jaxtyped(typechecker=beartype)
def diagonalize_tb(
    tb_model: TBModel,
    kpoints: Float[Array, "K 3"],
) -> DiagonalizedBands:
    """Diagonalize a TB model at all k-points.

    Builds H(k) for each k-point and diagonalizes via
    ``jnp.linalg.eigh``. Both operations are vmapped and
    fully differentiable with respect to ``tb_model.hopping_params``.

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
    """

    def _build_and_diag(k: Float[Array, " 3"]) -> tuple:
        H = build_hamiltonian_k(
            k,
            tb_model.hopping_params,
            tb_model.hopping_indices,
            tb_model.n_orbitals,
            tb_model.lattice_vectors,
        )
        evals, evecs = diagonalize_single_k(H)
        return evals, evecs

    eigenvalues, eigenvectors = jax.vmap(_build_and_diag)(kpoints)
    # eigenvectors from eigh: shape (K, O, O) where evecs[:, :, i] is i-th
    # We want (K, B, O) where B=O (number of bands = number of orbitals)
    # Transpose so evecs[k, b, o] = coefficient of orbital o in band b
    eigenvectors = jnp.transpose(eigenvectors, (0, 2, 1))

    return make_diagonalized_bands(
        eigenvalues=eigenvalues,
        eigenvectors=eigenvectors,
        kpoints=kpoints,
        fermi_energy=0.0,
    )


@jaxtyped(typechecker=beartype)
def vasp_to_diagonalized(
    bands: BandStructure,
    orb_proj: OrbitalProjection,
    orbital_basis: OrbitalBasis,
) -> DiagonalizedBands:
    """Convert VASP BandStructure + OrbitalProjection to DiagonalizedBands.

    VASP PROCAR gives |c_{k,b,orb}|^2, not the complex coefficients.
    This adapter uses sqrt(|c|^2) with positive sign as an
    approximation. Phase information is lost.

    The orbital projections are summed over atoms and mapped to
    the orbital basis ordering.

    Parameters
    ----------
    bands : BandStructure
        VASP eigenvalues and k-points.
    orb_proj : OrbitalProjection
        VASP orbital projections of shape (K, B, A, 9).
    orbital_basis : OrbitalBasis
        Quantum number metadata defining which VASP orbital
        indices to use.

    Returns
    -------
    diag : DiagonalizedBands
        Approximate diagonalized bands.
    """
    # Sum projections over atoms: (K, B, A, 9) -> (K, B, 9)
    proj_summed = jnp.sum(orb_proj.projections, axis=2)

    # Map orbital basis to VASP orbital indices
    # VASP ordering: [s, py, pz, px, dxy, dyz, dz2, dxz, dx2-y2]
    vasp_lm_to_idx = {
        (0, 0): 0,    # s
        (1, -1): 1,   # py
        (1, 0): 2,    # pz
        (1, 1): 3,    # px
        (2, -2): 4,   # dxy
        (2, -1): 5,   # dyz
        (2, 0): 6,    # dz2
        (2, 1): 7,    # dxz
        (2, 2): 8,    # dx2-y2
    }

    n_orbs = len(orbital_basis.l_values)
    orbital_indices = []
    for i in range(n_orbs):
        l_val = orbital_basis.l_values[i]
        m_val = orbital_basis.m_values[i]
        idx = vasp_lm_to_idx.get((l_val, m_val))
        if idx is None:
            msg = f"Orbital (l={l_val}, m={m_val}) not in VASP 9-orbital set"
            raise ValueError(msg)
        orbital_indices.append(idx)

    # Extract and take sqrt as approximate coefficients
    # proj_summed shape: (K, B, 9), select columns -> (K, B, n_orbs)
    idx_arr = jnp.array(orbital_indices)
    approx_c2 = proj_summed[:, :, idx_arr]
    approx_c = jnp.sqrt(jnp.maximum(approx_c2, 0.0))

    # Normalize eigenvectors per (k, band) so they sum to 1
    norm = jnp.sqrt(jnp.sum(approx_c**2, axis=-1, keepdims=True))
    safe_norm = jnp.where(norm > 1e-12, norm, 1.0)
    eigenvectors = (approx_c / safe_norm).astype(jnp.complex128)

    return make_diagonalized_bands(
        eigenvalues=bands.eigenvalues,
        eigenvectors=eigenvectors,
        kpoints=bands.kpoints,
        fermi_energy=bands.fermi_energy,
    )


__all__: list[str] = [
    "diagonalize_single_k",
    "diagonalize_tb",
    "vasp_to_diagonalized",
]
