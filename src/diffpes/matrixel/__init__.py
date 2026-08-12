"""Assemble coherent orbital and band photoemission matrix elements.

Extended Summary
----------------
The subpackage separates optimizer-facing parameter diagnostics from the
coherent transition assembly. Both modules preserve complex amplitudes
through polarization contraction and band projection. The public surface
keeps every matrix-element import at ``diffpes.matrixel``.

The following list describes the submodules:

- :mod:`parameters`
    Compute matrix-element parameters and sensitivity diagnostics.
- :mod:`transition`
    Assemble coherent orbital and band photoemission matrix elements.

Routine Listings
----------------
:func:`assemble_orbital_transition_channels`
    Assemble the validated orbital transition tensor.
:func:`band_group_weight_sensitivity`
    Compute complete isolated band-group weights and their Jacobian.
:func:`contract_polarization`
    Compute the sample-frame polarization contraction.
:func:`log_band_group_weight_sensitivity`
    Convert positive group-weight derivatives to logarithmic derivatives.
:func:`matrix_element_intensity`
    Sum outgoing-spin modulus squares exactly once.
:func:`matrix_element_phase_gauge_direction`
    Build the unit overall-phase tangent in packed coordinates.
:func:`orbital_transition_channels`
    Assemble coherent orbital transition channels.
:func:`pack_matrixel_params`
    Pack active matrix-element parameters into one real vector.
:func:`project_band_channels`
    Compute band channels without conjugating orbital coefficients.
:func:`radial_coefficient_scale_gauge_directions`
    Build normalized radial coefficient-scale gauge tangents.
:func:`real_spherical_harmonics_cartesian_all`
    Evaluate all real spherical harmonics from Cartesian directions.
:func:`resolve_orbital_positions_cart`
    Resolve orbital centres in Cartesian Angstrom coordinates.
:func:`transition_source`
    Build conjugated outgoing-spin rows as full source kets.
:func:`unpack_matrixel_params`
    Construct active matrix-element parameters from one real vector.
"""

from .parameters import (
    band_group_weight_sensitivity,
    log_band_group_weight_sensitivity,
    matrix_element_phase_gauge_direction,
    pack_matrixel_params,
    radial_coefficient_scale_gauge_directions,
    unpack_matrixel_params,
)
from .transition import (
    assemble_orbital_transition_channels,
    contract_polarization,
    matrix_element_intensity,
    orbital_transition_channels,
    project_band_channels,
    real_spherical_harmonics_cartesian_all,
    resolve_orbital_positions_cart,
    transition_source,
)

__all__: list[str] = [
    "assemble_orbital_transition_channels",
    "band_group_weight_sensitivity",
    "contract_polarization",
    "log_band_group_weight_sensitivity",
    "matrix_element_intensity",
    "matrix_element_phase_gauge_direction",
    "orbital_transition_channels",
    "pack_matrixel_params",
    "project_band_channels",
    "radial_coefficient_scale_gauge_directions",
    "real_spherical_harmonics_cartesian_all",
    "resolve_orbital_positions_cart",
    "transition_source",
    "unpack_matrixel_params",
]
