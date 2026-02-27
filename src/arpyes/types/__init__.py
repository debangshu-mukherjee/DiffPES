"""Type definitions and factory functions for arpyes.

Extended Summary
----------------
Provides PyTree-compatible data structures and their factory
functions for representing ARPES simulation data including crystal
geometry, band structures, orbital projections, simulation
parameters, and polarization configurations. Fields that
participate in autodiff are stored as JAX array children, while
shape-determining values (e.g., ``SimulationParams.fidelity``)
and code-path selectors (e.g., ``PolarizationConfig.polarization_type``)
are stored as auxiliary data so they remain concrete at trace time
and trigger recompilation only when changed.

Routine Listings
----------------
:class:`ArpesSpectrum`
    PyTree for ARPES simulation output.
:class:`BandStructure`
    PyTree for electronic band structure.
:class:`CrystalGeometry`
    PyTree for crystal geometry from VASP POSCAR.
:class:`DensityOfStates`
    PyTree for density of states.
:class:`KPathInfo`
    PyTree for k-point path metadata.
:class:`OrbitalProjection`
    PyTree for orbital-resolved band projections.
:class:`PolarizationConfig`
    PyTree for photon polarization geometry.
:class:`SimulationParams`
    PyTree for ARPES simulation parameters.
:func:`make_arpes_spectrum`
    Factory for ArpesSpectrum.
:func:`make_band_structure`
    Factory for BandStructure with validation.
:func:`make_crystal_geometry`
    Factory for CrystalGeometry with reciprocal lattice.
:func:`make_density_of_states`
    Factory for DensityOfStates.
:func:`make_kpath_info`
    Factory for KPathInfo.
:func:`make_orbital_projection`
    Factory for OrbitalProjection.
:func:`make_polarization_config`
    Factory for PolarizationConfig.
:func:`make_simulation_params`
    Factory for SimulationParams.

Notes
-----
All PyTree types use ``@register_pytree_node_class`` for
compatibility with JAX transformations (jit, grad, vmap).
"""

from .aliases import (
    NonJaxNumber,
    ScalarBool,
    ScalarComplex,
    ScalarFloat,
    ScalarInteger,
    ScalarNumeric,
)
from .bands import (
    ArpesSpectrum,
    BandStructure,
    OrbitalProjection,
    SpinOrbitalProjection,
    make_arpes_spectrum,
    make_band_structure,
    make_orbital_projection,
    make_spin_orbital_projection,
)
from .dos import (
    DensityOfStates,
    make_density_of_states,
)
from .geometry import (
    CrystalGeometry,
    make_crystal_geometry,
)
from .kpath import (
    KPathInfo,
    make_kpath_info,
)
from .params import (
    PolarizationConfig,
    SimulationParams,
    make_polarization_config,
    make_simulation_params,
)

__all__: list[str] = [
    "ArpesSpectrum",
    "BandStructure",
    "CrystalGeometry",
    "DensityOfStates",
    "KPathInfo",
    "NonJaxNumber",
    "OrbitalProjection",
    "SpinOrbitalProjection",
    "PolarizationConfig",
    "ScalarBool",
    "ScalarComplex",
    "ScalarFloat",
    "ScalarInteger",
    "ScalarNumeric",
    "SimulationParams",
    "make_arpes_spectrum",
    "make_band_structure",
    "make_crystal_geometry",
    "make_density_of_states",
    "make_kpath_info",
    "make_orbital_projection",
    "make_spin_orbital_projection",
    "make_polarization_config",
    "make_simulation_params",
]
