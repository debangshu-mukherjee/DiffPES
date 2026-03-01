"""Minimal tight-binding infrastructure for testing.

Extended Summary
----------------
Provides a JAX-traceable Hamiltonian builder, differentiable
diagonalization, and convenience models (graphene, 1D chain)
for testing the forward simulator pipeline.
"""

from .diagonalize import diagonalize_single_k, diagonalize_tb, vasp_to_diagonalized
from .hamiltonian import build_hamiltonian_k, make_1d_chain_model, make_graphene_model
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
