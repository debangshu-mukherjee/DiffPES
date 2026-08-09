r"""Provide differentiable ARPES simulation primitives.

Extended Summary
----------------
The subpackage provides two deliberately incoherent projection-spectrum
tiers, authenticated atomic cross sections, detector kinematics, polarization
frame transforms, broadening primitives, and orbital angular momentum.
Coherent photoemission amplitudes live in :mod:`matrixel`.

The following list describes the submodules:

- :mod:`broadening`
    Compute energy broadening functions for ARPES simulations.
- :mod:`crosssections`
    Interpolate authenticated Yeh--Lindau photoionization cross sections.
- :mod:`expanded`
    Run the incoherent ARPES tiers from plain band/projection arrays.
- :mod:`kinematics`
    Compute free-electron photoemission kinematics.
- :mod:`matrixel`
    Assemble coherent orbital and band photoemission matrix elements.
- :mod:`oam`
    Compute orbital angular momentum.
- :mod:`polarization`
    Compute photon polarization and explicit frame transformations.
- :mod:`resolution`
    Apply momentum resolution broadening to ARPES simulations.
- :mod:`spectral`
    Evaluate the complex retarded self-energy through the certified KK map.
- :mod:`spectrum`
    Simulate deliberately incoherent ARPES projection spectra.
- :mod:`workflow`
    Run high-level workflows for VASP-to-ARPES simulation.

Routine Listings
----------------
:func:`apply_momentum_broadening`
    Convolve I(k, E) with a Gaussian in k-space.
:func:`build_polarization_vectors`
    Construct s- and p-polarization basis vectors.
:func:`detector_angles_to_kpar`
    Convert detector angles to parallel momentum.
:func:`detector_axis_to_sample`
    Convert a detector-fixed axis to sample coordinates.
:func:`detector_rotation`
    Build the detector-frame rotation.
:func:`compute_oam`
    Compute orbital angular momentum z-component.
:func:`evaluate_self_energy`
    Evaluate the complex retarded self-energy for one causal model.
:func:`emission_angles`
    Convert Cartesian momentum to emission angles.
:func:`fermi_dirac`
    Compute Fermi-Dirac distribution value.
:func:`final_state_k_inv_ang`
    Convert kinetic energy to momentum and return its validity mask.
:func:`gaussian`
    Compute normalized Gaussian broadening profile.
:func:`kinetic_energy_ev`
    Compute signed photoelectron kinetic energy and its validity mask.
:func:`assemble_orbital_transition_channels`
    Assemble the validated orbital transition tensor.
:func:`band_group_weight_sensitivity`
    Compute complete isolated band-group weights and their Jacobian.
:func:`contract_experiment_polarization`
    Rotate laboratory polarization to the sample and contract it late.
:func:`contract_polarization`
    Compute the sample-frame polarization contraction.
:func:`lab_polarization_to_sample`
    Convert fixed laboratory polarization to sample coordinates.
:func:`kpar_to_detector_angles`
    Convert parallel momentum to detector angles.
:func:`kz_from_inner_potential`
    Compute complex out-of-plane momentum from the inner potential.
:func:`kz_from_inner_potential_at_fermi`
    Evaluate the named Fermi-level ``kz`` approximation.
:func:`load_vasp_context`
    Load a simulation-ready context from VASP output files.
:func:`matrix_element_intensity`
    Sum outgoing-spin modulus squares exactly once.
:func:`log_band_group_weight_sensitivity`
    Convert positive group-weight derivatives to logarithmic derivatives.
:func:`matrix_element_phase_gauge_direction`
    Build the unit overall-phase tangent in packed coordinates.
:func:`orbital_transition_channels`
    Assemble coherent orbital transition channels.
:func:`pack_matrixel_params`
    Pack active matrix-element parameters into one real vector.
:func:`photon_wavevector`
    Build the unit photon wavevector from incidence angles.
:func:`polarization_from_angles`
    Construct polarization from incidence angles.
:func:`polarization_to_spherical`
    Convert Cartesian polarization to spherical components.
:func:`prepare_projection`
    Prepare orbital projections for simulation.
:func:`project_band_channels`
    Compute band channels without conjugating orbital coefficients.
:func:`radial_coefficient_scale_gauge_directions`
    Build normalized radial coefficient-scale gauge tangents.
:func:`real_spherical_harmonics_cartesian_all`
    Evaluate all real spherical harmonics from Cartesian directions.
:func:`resolve_orbital_positions_cart`
    Resolve orbital centres in Cartesian Angstrom coordinates.
:func:`run_vasp_workflow`
    Run an end-to-end VASP-to-ARPES workflow in one call.
:func:`rotate_frame_vectors`
    Rotate a detector-fixed real axis across a detector-angle grid.
:func:`sample_azimuth_rotation`
    Build the active sample-to-laboratory azimuth rotation.
:func:`simulate_basic`
    Simulate an incoherent spectrum with Yeh--Lindau weights.
:func:`simulate_basic_expanded`
    Run the Yeh--Lindau-weighted incoherent tier from plain arrays.
:func:`simulate_context`
    Run a level-dispatched simulation from a loaded workflow context.
:func:`simulate_expanded`
    Dispatch one of the two retained incoherent simulation tiers.
:func:`simulate_novice`
    Simulate an incoherent spectrum with uniform orbital weights.
:func:`simulate_novice_expanded`
    Run the uniformly weighted incoherent tier from plain arrays.
:func:`transition_source`
    Build conjugated outgoing-spin rows as full source kets.
:func:`unpack_matrixel_params`
    Construct active matrix-element parameters from one real vector.
:func:`voigt`
    Compute a normalized Voigt profile through the Faddeeva function.
:func:`yeh_lindau_cross_section`
    Interpolate an atomic subshell photoionization cross section.
:func:`yeh_lindau_cross_section_table`
    Return one raw Yeh--Lindau subshell row.
:func:`yeh_lindau_orbital_weights`
    Return Yeh--Lindau weights for every orbital in a basis.

Notes
-----
The retained simulation functions are JAX-compatible and use ``jax.vmap``
across k-points and bands.
"""

from .broadening import fermi_dirac, gaussian, voigt
from .crosssections import (
    yeh_lindau_cross_section,
    yeh_lindau_cross_section_table,
    yeh_lindau_orbital_weights,
)
from .expanded import (
    simulate_basic_expanded,
    simulate_expanded,
    simulate_novice_expanded,
)
from .kinematics import (
    detector_angles_to_kpar,
    emission_angles,
    final_state_k_inv_ang,
    kinetic_energy_ev,
    kpar_to_detector_angles,
    kz_from_inner_potential,
    kz_from_inner_potential_at_fermi,
)
from .matrixel import (
    assemble_orbital_transition_channels,
    band_group_weight_sensitivity,
    contract_experiment_polarization,
    contract_polarization,
    log_band_group_weight_sensitivity,
    matrix_element_intensity,
    matrix_element_phase_gauge_direction,
    orbital_transition_channels,
    pack_matrixel_params,
    project_band_channels,
    radial_coefficient_scale_gauge_directions,
    real_spherical_harmonics_cartesian_all,
    resolve_orbital_positions_cart,
    transition_source,
    unpack_matrixel_params,
)
from .oam import compute_oam
from .polarization import (
    build_polarization_vectors,
    detector_axis_to_sample,
    detector_rotation,
    lab_polarization_to_sample,
    photon_wavevector,
    polarization_from_angles,
    polarization_to_spherical,
    rotate_frame_vectors,
    sample_azimuth_rotation,
)
from .resolution import apply_momentum_broadening
from .spectral import evaluate_self_energy
from .spectrum import (
    simulate_basic,
    simulate_novice,
)
from .workflow import (
    load_vasp_context,
    prepare_projection,
    run_vasp_workflow,
    simulate_context,
)

__all__: list[str] = [
    "apply_momentum_broadening",
    "assemble_orbital_transition_channels",
    "band_group_weight_sensitivity",
    "build_polarization_vectors",
    "compute_oam",
    "contract_experiment_polarization",
    "contract_polarization",
    "detector_angles_to_kpar",
    "detector_axis_to_sample",
    "detector_rotation",
    "emission_angles",
    "evaluate_self_energy",
    "fermi_dirac",
    "final_state_k_inv_ang",
    "gaussian",
    "kinetic_energy_ev",
    "lab_polarization_to_sample",
    "kpar_to_detector_angles",
    "kz_from_inner_potential",
    "kz_from_inner_potential_at_fermi",
    "load_vasp_context",
    "log_band_group_weight_sensitivity",
    "matrix_element_intensity",
    "matrix_element_phase_gauge_direction",
    "orbital_transition_channels",
    "photon_wavevector",
    "pack_matrixel_params",
    "polarization_from_angles",
    "polarization_to_spherical",
    "prepare_projection",
    "project_band_channels",
    "radial_coefficient_scale_gauge_directions",
    "real_spherical_harmonics_cartesian_all",
    "resolve_orbital_positions_cart",
    "run_vasp_workflow",
    "rotate_frame_vectors",
    "sample_azimuth_rotation",
    "simulate_basic",
    "simulate_basic_expanded",
    "simulate_context",
    "simulate_expanded",
    "simulate_novice",
    "simulate_novice_expanded",
    "transition_source",
    "unpack_matrixel_params",
    "voigt",
    "yeh_lindau_cross_section",
    "yeh_lindau_cross_section_table",
    "yeh_lindau_orbital_weights",
]
