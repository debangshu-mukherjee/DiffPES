"""ARPES simulation functions at five complexity levels.

Extended Summary
----------------
Provides a complete ARPES simulation pipeline from basic Voigt
convolution to full polarization-dependent dipole matrix element
calculations. Also exports broadening functions, cross-section
models, polarization utilities, and orbital angular momentum.

Routine Listings
----------------
:func:`simulate_novice`
    Voigt broadening with uniform orbital weights.
:func:`simulate_basic`
    Gaussian broadening with heuristic orbital weights.
:func:`simulate_basicplus`
    Gaussian broadening with Yeh-Lindau cross-sections.
:func:`simulate_advanced`
    Gaussian with Yeh-Lindau and polarization selection rules.
:func:`simulate_expert`
    Voigt with Yeh-Lindau, polarization, and dipole elements.
:func:`simulate_soc`
    Expert plus spin-orbit (S·k_photon) correction.
:func:`simulate_expanded`
    Expanded-input dispatcher for the five simulation levels.
:func:`simulate_novice_expanded`
    Expanded-input novice wrapper.
:func:`simulate_basic_expanded`
    Expanded-input basic wrapper.
:func:`simulate_basicplus_expanded`
    Expanded-input basicplus wrapper.
:func:`simulate_advanced_expanded`
    Expanded-input advanced wrapper.
:func:`simulate_expert_expanded`
    Expanded-input expert wrapper.
:func:`simulate_soc_expanded`
    Expanded-input SOC wrapper (requires surface_spin).
:func:`make_expanded_simulation_params`
    Build expanded-input simulation parameters from eigenbands.
:func:`gaussian`
    Normalized Gaussian broadening profile.
:func:`voigt`
    Normalized Voigt profile via the Faddeeva function.
:func:`fermi_dirac`
    Fermi-Dirac thermal distribution function.
:func:`heuristic_weights`
    Energy-dependent heuristic orbital weights.
:func:`yeh_lindau_weights`
    Interpolated Yeh-Lindau cross-section weights.
:func:`build_polarization_vectors`
    Construct s- and p-polarization basis vectors.
:func:`build_efield`
    Compute electric field vector from polarization config.
:func:`dipole_matrix_elements`
    Compute dipole matrix elements for all 9 orbitals.
:func:`compute_oam`
    Compute orbital angular momentum from projections.

Notes
-----
All simulation functions are JAX-compatible and use ``jax.vmap``
for vectorized evaluation across k-points and bands.
"""

from .broadening import fermi_dirac, gaussian, voigt
from .crosssections import heuristic_weights, yeh_lindau_weights
from .expanded import (
    make_expanded_simulation_params,
    simulate_advanced_expanded,
    simulate_basic_expanded,
    simulate_basicplus_expanded,
    simulate_expanded,
    simulate_expert_expanded,
    simulate_novice_expanded,
    simulate_soc_expanded,
)
from .oam import compute_oam
from .polarization import (
    build_efield,
    build_polarization_vectors,
    dipole_matrix_elements,
    photon_wavevector,
)
from .spectrum import (
    simulate_advanced,
    simulate_basic,
    simulate_basicplus,
    simulate_expert,
    simulate_novice,
    simulate_soc,
)

__all__: list[str] = [
    "build_efield",
    "build_polarization_vectors",
    "compute_oam",
    "dipole_matrix_elements",
    "fermi_dirac",
    "gaussian",
    "heuristic_weights",
    "make_expanded_simulation_params",
    "photon_wavevector",
    "simulate_advanced_expanded",
    "simulate_advanced",
    "simulate_basic_expanded",
    "simulate_basic",
    "simulate_basicplus_expanded",
    "simulate_basicplus",
    "simulate_expert_expanded",
    "simulate_expert",
    "simulate_expanded",
    "simulate_novice_expanded",
    "simulate_novice",
    "simulate_soc_expanded",
    "simulate_soc",
    "voigt",
    "yeh_lindau_weights",
]
