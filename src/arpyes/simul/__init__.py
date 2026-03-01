"""ARPES simulation functions at five complexity levels.

Extended Summary
----------------
Provides a complete ARPES simulation pipeline from basic Voigt
convolution to full polarization-dependent dipole matrix element
calculations. Also exports broadening functions, cross-section
models, polarization utilities, and orbital angular momentum.

Routine Listings
----------------
:func:`build_efield`
    Compute electric field vector from polarization config.
:func:`build_polarization_vectors`
    Construct s- and p-polarization basis vectors.
:func:`compute_oam`
    Compute orbital angular momentum from projections.
:func:`dipole_matrix_elements`
    Compute dipole matrix elements for all 9 orbitals.
:func:`fermi_dirac`
    Fermi-Dirac thermal distribution function.
:func:`gaussian`
    Normalized Gaussian broadening profile.
:func:`heuristic_weights`
    Energy-dependent heuristic orbital weights.
:func:`make_expanded_simulation_params`
    Build expanded-input simulation parameters from eigenbands.
:func:`photon_wavevector`
    Unit photon propagation vector from incidence angles.
:func:`simulate_advanced`
    Gaussian with Yeh-Lindau and polarization selection rules.
:func:`simulate_advanced_expanded`
    Expanded-input advanced wrapper.
:func:`simulate_basic`
    Gaussian broadening with heuristic orbital weights.
:func:`simulate_basic_expanded`
    Expanded-input basic wrapper.
:func:`simulate_basicplus`
    Gaussian broadening with Yeh-Lindau cross-sections.
:func:`simulate_basicplus_expanded`
    Expanded-input basicplus wrapper.
:func:`simulate_expanded`
    Expanded-input dispatcher by complexity level.
:func:`simulate_expert`
    Voigt with Yeh-Lindau, polarization, and dipole elements.
:func:`simulate_expert_expanded`
    Expanded-input expert wrapper.
:func:`simulate_novice`
    Voigt broadening with uniform orbital weights.
:func:`simulate_novice_expanded`
    Expanded-input novice wrapper.
:func:`simulate_soc`
    Expert plus spin-orbit (S·k_photon) correction.
:func:`simulate_soc_expanded`
    Expanded-input SOC wrapper (requires surface_spin).
:func:`voigt`
    Normalized Voigt profile via the Faddeeva function.
:func:`yeh_lindau_weights`
    Interpolated Yeh-Lindau cross-section weights.

Notes
-----
All simulation functions are JAX-compatible and use ``jax.vmap``
for vectorized evaluation across k-points and bands.
"""

from .broadening import fermi_dirac, gaussian, voigt
from .crosssections import heuristic_weights, yeh_lindau_weights
from .forward import simulate_tb_radial
from .resolution import apply_momentum_broadening
from .self_energy import evaluate_self_energy
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
    "apply_momentum_broadening",
    "build_efield",
    "build_polarization_vectors",
    "compute_oam",
    "dipole_matrix_elements",
    "evaluate_self_energy",
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
    "simulate_tb_radial",
    "voigt",
    "yeh_lindau_weights",
]
