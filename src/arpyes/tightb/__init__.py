r"""Legacy tight-binding shim and ARPES-side adapters.

Extended Summary
----------------
Tight-binding Hamiltonian construction and diagonalization are
owned by the companion project **DiffTB** (``../DiffTB``).
ARPyES is purely an ARPES calculator that consumes
``DiagonalizedBands`` from any electronic-structure source.

This module retains:

- **ARPES-side adapters** that stay here permanently:
  ``vasp_to_diagonalized``, ``eigenvector_orbital_weights``,
  ``orbital_coefficients``.
- **Legacy test fixtures** (no new development):
  ``build_hamiltonian_k``, ``diagonalize_single_k``,
  ``diagonalize_tb``, ``make_1d_chain_model``,
  ``make_graphene_model``.

Routine Listings
----------------
:func:`vasp_to_diagonalized`
    Convert VASP ``BandStructure`` + ``OrbitalProjection`` to
    ``DiagonalizedBands`` (phase-less sqrt approximation).
:func:`eigenvector_orbital_weights`
    Compute orbital weights :math:`|c_{k,b,\\mathrm{orb}}|^2` from
    eigenvectors.
:func:`orbital_coefficients`
    Return raw complex orbital coefficients (identity accessor).
:func:`build_hamiltonian_k`
    (Legacy) Build the Bloch Hamiltonian H(k) at a single k-point.
:func:`diagonalize_single_k`
    (Legacy) Diagonalize a Hermitian Hamiltonian at one k-point.
:func:`diagonalize_tb`
    (Legacy) Diagonalize a ``TBModel`` at all k-points (vmapped).
:func:`make_1d_chain_model`
    (Legacy) Create a one-orbital 1D chain ``TBModel``.
:func:`make_graphene_model`
    (Legacy) Create a two-orbital honeycomb graphene ``TBModel``.

Notes
-----
For production tight-binding workflows, use DiffTB to produce
``DiagonalizedBands`` and pass them to ``simulate_tb_radial``.
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
