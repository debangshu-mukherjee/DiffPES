"""Angular matrix elements for dipole photoemission.

Extended Summary
----------------
Provides Gaunt coefficients, real spherical harmonics, and full
dipole matrix element assembly for the differentiable ARPES
forward model.
"""

from .dipole import (
    dipole_intensities_all_orbitals,
    dipole_intensity_orbital,
    dipole_matrix_element_single,
)
from .gaunt import GAUNT_TABLE, L_MAX, build_gaunt_table, gaunt_lookup
from .spherical_harmonics import (
    real_spherical_harmonic,
    real_spherical_harmonics_all,
)

__all__: list[str] = [
    "GAUNT_TABLE",
    "L_MAX",
    "build_gaunt_table",
    "dipole_intensities_all_orbitals",
    "dipole_intensity_orbital",
    "dipole_matrix_element_single",
    "gaunt_lookup",
    "real_spherical_harmonic",
    "real_spherical_harmonics_all",
]
