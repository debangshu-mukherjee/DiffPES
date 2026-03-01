r"""Minimal tight-binding infrastructure for testing.

Extended Summary
----------------
Provides a JAX-traceable Hamiltonian builder, differentiable
diagonalization, and convenience models (graphene, 1D chain)
for testing the forward simulator pipeline.

Routine Listings
----------------
:func:`build_hamiltonian_k`
    Build the Bloch Hamiltonian H(k) at a single k-point from
    hopping parameters and lattice vectors.
:func:`diagonalize_single_k`
    Diagonalize a Hermitian Hamiltonian at one k-point via
    ``jnp.linalg.eigh``.
:func:`diagonalize_tb`
    Diagonalize a ``TBModel`` at all k-points (vmapped).
:func:`vasp_to_diagonalized`
    Convert VASP ``BandStructure`` + ``OrbitalProjection`` to
    ``DiagonalizedBands`` (phase-less sqrt approximation).
:func:`eigenvector_orbital_weights`
    Compute orbital weights :math:`|c_{k,b,\\mathrm{orb}}|^2` from
    eigenvectors.
:func:`orbital_coefficients`
    Return raw complex orbital coefficients (identity accessor).
:func:`make_1d_chain_model`
    Create a one-orbital 1D chain ``TBModel``.
:func:`make_graphene_model`
    Create a two-orbital honeycomb graphene ``TBModel``.

Notes
-----
All functions are JAX-compatible and fully differentiable with
respect to ``TBModel.hopping_params``.  The Hamiltonian builder
and diagonalizer are vmapped over k-points.
"""

from .diagonalize import (
    diagonalize_single_k,
    diagonalize_tb,
    vasp_to_diagonalized,
)
from .hamiltonian import (
    build_hamiltonian_k,
    make_1d_chain_model,
    make_graphene_model,
)
from .projections import eigenvector_orbital_weights, orbital_coefficients

__all__: list[str] = [
    "build_hamiltonian_k",
    "diagonalize_single_k",
    "diagonalize_tb",
    "eigenvector_orbital_weights",
    "make_1d_chain_model",
    "make_graphene_model",
    "orbital_coefficients",
    "vasp_to_diagonalized",
]
