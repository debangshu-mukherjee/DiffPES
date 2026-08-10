r"""Provide differentiable ARPES simulation primitives.

Extended Summary
----------------
The subpackage provides coherent spectral assembly, authenticated atomic
cross sections, detector kinematics, polarization frame transforms,
broadening primitives, and orbital angular momentum. Coherent photoemission
amplitudes live in :mod:`matrixel`; the resolvent and eigen assembly in
:mod:`spectral` combines those amplitudes with a causal self-energy and
sampled-energy Fermi occupation.

The following list describes the submodules:

- :mod:`_detector_map`
    Compute conservative source-to-detector finite-volume maps.
- :mod:`broadening`
    Compute energy broadening functions for ARPES simulations.
- :mod:`crosssections`
    Interpolate authenticated Yeh--Lindau photoionization cross sections.
- :mod:`effects`
    Apply calibrated instrument effects and assemble expected counts.
- :mod:`kinematics`
    Compute free-electron photoemission kinematics.
- :mod:`matrixel`
    Assemble coherent orbital and band photoemission matrix elements.
- :mod:`oam`
    Compute orbital angular momentum.
- :mod:`polarization`
    Compute photon polarization and explicit frame transformations.
- :mod:`spectrum`
    Compose the coherent single-:math:`k_z` ARPES forward driver.
- :mod:`spectral`
    Evaluate the complex retarded self-energy through the certified KK map.
- :mod:`workflow`
    Load VASP metadata and compose the explicit-H coherent cut workflow.

Routine Listings
----------------
:func:`apply_detector_effects`
    Apply the complete deterministic source-to-count detector chain.
:func:`apply_post_count_response`
    Convolve expected counts along the recorded-energy index.
:func:`apply_resolution`
    Apply analytic finite-volume resolution in native detector coordinates.
:func:`apply_transmission`
    Apply analyser transmission to intensity at true kinetic energy.
:func:`assemble_spectral_intensity_bands_chunk`
    Assemble occupied intrinsic intensity from eigenvalues and band weights.
:func:`assemble_spectral_intensity_chunk`
    Assemble occupied intrinsic intensity from Hamiltonians and sources.
:func:`build_polarization_vectors`
    Construct s- and p-polarization basis vectors.
:func:`detector_angles_to_kpar`
    Convert detector angles to parallel momentum.
:func:`detector_axis_to_sample`
    Convert a detector-fixed axis to sample coordinates.
:func:`detector_rotation`
    Build the detector-frame rotation.
:func:`detector_bin_volumes`
    Compute explicit native detector-bin volumes.
:func:`compute_oam`
    Compute orbital angular momentum z-component.
:func:`convolve_energy`
    Convolve a uniform energy axis with the sampled parity stencil.
:func:`convolve_kpath`
    Convolve physical-k path-cell densities with analytic boundary loss.
:func:`convolve_momentum_map`
    Convolve a uniform Cartesian momentum map with sampled stencils.
:func:`evaluate_self_energy`
    Evaluate the complex retarded self-energy for one causal model.
:func:`expected_counts`
    Assemble deterministic expected detector counts.
:func:`emission_angles`
    Convert Cartesian momentum to emission angles.
:func:`fermi_dirac`
    Compute Fermi-Dirac distribution value.
:func:`final_state_k_inv_ang`
    Convert kinetic energy to momentum and return its validity mask.
:func:`fixed_total_probabilities`
    Normalize all detector rates to one event-probability tensor.
:func:`gaussian`
    Compute normalized Gaussian broadening profile.
:func:`gaussian_kernel_1d`
    Build a sampled, sum-normalized Gaussian stencil.
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
:func:`normalize_intensity`
    Return an explicit display-only normalization of carrier values.
:func:`background_density`
    Evaluate a nonnegative detector-coordinate background.
:func:`log_band_group_weight_sensitivity`
    Convert positive group-weight derivatives to logarithmic derivatives.
:func:`matrix_element_phase_gauge_direction`
    Build the unit overall-phase tangent in packed coordinates.
:func:`map_source_to_detector`
    Convert one source density to native detector bins conservatively.
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
:func:`projected_spectral_density_resolvent`
    Compute the projected Hermitian resolvent spectral density.
:func:`project_band_channels`
    Compute band channels without conjugating orbital coefficients.
:func:`radial_coefficient_scale_gauge_directions`
    Build normalized radial coefficient-scale gauge tangents.
:func:`real_spherical_harmonics_cartesian_all`
    Evaluate all real spherical harmonics from Cartesian directions.
:func:`resolve_orbital_positions_cart`
    Resolve orbital centres in Cartesian Angstrom coordinates.
:func:`rotate_frame_vectors`
    Rotate a detector-fixed real axis across a detector-angle grid.
:func:`run_vasp_workflow`
    Run the explicit-H coherent cut workflow with VASP metadata.
:func:`sample_azimuth_rotation`
    Build the active sample-to-laboratory azimuth rotation.
:func:`sample_fixed_total_counts`
    Generate one fixed-total multinomial count tensor.
:func:`sample_poisson_counts`
    Generate independent Poisson counts for a rate tensor.
:func:`sensitivity_field`
    Evaluate the positive normalized detector sensitivity field.
:func:`simulate_arpes`
    Simulate the canonical coherent single-kz detector raster.
:func:`simulate_arpes_cut`
    Simulate the canonical coherent single-kz path-cut detector raster.
:func:`spectral_intensity_eigen`
    Evaluate spectral intensity from eigenvalues and invariant weights.
:func:`spectral_intensity_resolvent`
    Evaluate degeneracy-safe spectral intensity through a linear solve.
:func:`transition_source`
    Build conjugated outgoing-spin rows as full source kets.
:func:`transmission_shape`
    Evaluate positive monotone analyser transmission with fixed mean one.
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
The spectral functions are JAX-compatible and preserve coherent source
amplitudes through the final spectral reduction.
"""

from .broadening import fermi_dirac, gaussian, voigt
from .crosssections import (
    yeh_lindau_cross_section,
    yeh_lindau_cross_section_table,
    yeh_lindau_orbital_weights,
)
from .effects import (
    apply_detector_effects,
    apply_post_count_response,
    apply_resolution,
    apply_transmission,
    background_density,
    convolve_energy,
    convolve_kpath,
    convolve_momentum_map,
    detector_bin_volumes,
    expected_counts,
    fixed_total_probabilities,
    gaussian_kernel_1d,
    map_source_to_detector,
    sample_fixed_total_counts,
    sample_poisson_counts,
    sensitivity_field,
    transmission_shape,
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
from .spectral import (
    assemble_spectral_intensity_bands_chunk,
    assemble_spectral_intensity_chunk,
    evaluate_self_energy,
    projected_spectral_density_resolvent,
    spectral_intensity_eigen,
    spectral_intensity_resolvent,
)
from .spectrum import normalize_intensity, simulate_arpes, simulate_arpes_cut
from .workflow import (
    load_vasp_context,
    prepare_projection,
    run_vasp_workflow,
)

__all__: list[str] = [
    "apply_detector_effects",
    "apply_post_count_response",
    "apply_resolution",
    "apply_transmission",
    "assemble_orbital_transition_channels",
    "assemble_spectral_intensity_bands_chunk",
    "assemble_spectral_intensity_chunk",
    "band_group_weight_sensitivity",
    "background_density",
    "build_polarization_vectors",
    "compute_oam",
    "convolve_energy",
    "convolve_kpath",
    "convolve_momentum_map",
    "contract_experiment_polarization",
    "contract_polarization",
    "detector_angles_to_kpar",
    "detector_axis_to_sample",
    "detector_bin_volumes",
    "detector_rotation",
    "emission_angles",
    "evaluate_self_energy",
    "expected_counts",
    "fermi_dirac",
    "final_state_k_inv_ang",
    "fixed_total_probabilities",
    "gaussian",
    "gaussian_kernel_1d",
    "kinetic_energy_ev",
    "lab_polarization_to_sample",
    "kpar_to_detector_angles",
    "kz_from_inner_potential",
    "kz_from_inner_potential_at_fermi",
    "load_vasp_context",
    "log_band_group_weight_sensitivity",
    "matrix_element_intensity",
    "matrix_element_phase_gauge_direction",
    "map_source_to_detector",
    "normalize_intensity",
    "orbital_transition_channels",
    "photon_wavevector",
    "pack_matrixel_params",
    "polarization_from_angles",
    "polarization_to_spherical",
    "prepare_projection",
    "projected_spectral_density_resolvent",
    "project_band_channels",
    "radial_coefficient_scale_gauge_directions",
    "real_spherical_harmonics_cartesian_all",
    "resolve_orbital_positions_cart",
    "rotate_frame_vectors",
    "run_vasp_workflow",
    "sample_azimuth_rotation",
    "sample_fixed_total_counts",
    "sample_poisson_counts",
    "sensitivity_field",
    "simulate_arpes",
    "simulate_arpes_cut",
    "spectral_intensity_eigen",
    "spectral_intensity_resolvent",
    "transition_source",
    "transmission_shape",
    "unpack_matrixel_params",
    "voigt",
    "yeh_lindau_cross_section",
    "yeh_lindau_cross_section_table",
    "yeh_lindau_orbital_weights",
]
