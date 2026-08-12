r"""Provide differentiable ARPES simulation primitives.

Extended Summary
----------------
The subpackage provides coherent spectral assembly, authenticated atomic
cross sections, detector kinematics, polarization frame transforms,
broadening primitives, and orbital angular momentum. The resolvent and eigen
assembly in :mod:`spectral` combines matrix-element amplitudes with a causal
self-energy and sampled-energy Fermi occupation.

The following list describes the submodules:

- :mod:`_detector_cube`
    PRIVATE: Map Cartesian source cubes to native detector bins.
- :mod:`_detector_geometry`
    PRIVATE: Compute detector-map geometry and finite-volume quadrature.
- :mod:`_detector_map`
    PRIVATE: Dispatch conservative source-to-detector maps.
- :mod:`_detector_spectrum`
    PRIVATE: Map path spectra to native detector bins.
- :mod:`_kramers_kronig`
    PRIVATE: Apply the certified retarded Kramers--Kronig map.
- :mod:`_kz_spectrum`
    PRIVATE: Assemble exact and broadened bulk out-of-plane spectra.
- :mod:`_principal_value`
    PRIVATE: Evaluate static principal-value quadrature primitives.
- :mod:`_source_carriers`
    PRIVATE: Construct physical ARPES source carriers.
- :mod:`_spectrum_stream`
    PRIVATE: Stream bounded coherent source-intensity chunks.
- :mod:`_spectrum_validation`
    PRIVATE: Validate coherent ARPES driver structure.
- :mod:`broadening`
    Compute energy broadening functions for ARPES simulations.
- :mod:`counting`
    Sample detector counts from expected rates.
- :mod:`crosssections`
    Interpolate authenticated Yeh--Lindau photoionization cross sections.
- :mod:`detector_response`
    Assemble calibrated detector response fields and rates.
- :mod:`effects`
    Compose source mapping and deterministic detector effects.
- :mod:`factorized`
    Compose typed electronic-state and factorized-current evaluation.
- :mod:`generalized_spectral`
    Evaluate metric-aware retarded Green functions and spectral projections.
- :mod:`kinematics`
    Compute free-electron photoemission kinematics.
- :mod:`kz_broadening`
    Apply wrapped out-of-plane momentum broadening.
- :mod:`oam`
    Compute orbital angular momentum.
- :mod:`polarization`
    Compute photon polarization and explicit frame transformations.
- :mod:`plane_wave`
    Compute bounded pseudo-wave and PAW-restored ARPES amplitudes.
- :mod:`resolution`
    Apply finite-volume detector resolution.
- :mod:`retarded_self_energy`
    Evaluate causal retarded self-energy models.
- :mod:`spectral`
    Assemble chunked occupied intrinsic spectral intensity.
- :mod:`spectral_eigen`
    Evaluate nondegenerate eigenvalue spectral observables.
- :mod:`spectral_resolvent`
    Evaluate degeneracy-safe resolvent spectral observables.
- :mod:`spectrum`
    Compose coherent ARPES and photon-energy-scan drivers.
- :mod:`transmission`
    Apply calibrated analyser transmission.
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
:func:`background_density`
    Evaluate a nonnegative detector-coordinate background.
:func:`broaden_kz`
    Apply wrapped-Cauchy bin masses to node-resolved bulk intensity.
:func:`build_polarization_vectors`
    Construct s- and p-polarization basis vectors.
:func:`compute_oam`
    Compute orbital angular momentum z-component.
:func:`contract_experiment_polarization`
    Rotate laboratory polarization to the sample and contract it late.
:func:`convolve_energy`
    Convolve a uniform energy axis with the sampled parity stencil.
:func:`convolve_kpath`
    Convolve physical-k path-cell densities with analytic boundary loss.
:func:`convolve_momentum_map`
    Convolve a uniform Cartesian momentum map with sampled stencils.
:func:`detector_angles_to_kpar`
    Convert detector angles to parallel momentum.
:func:`detector_axis_to_sample`
    Convert a detector-fixed axis to sample coordinates.
:func:`detector_bin_volumes`
    Compute explicit native detector-bin volumes.
:func:`detector_rotation`
    Build the detector-frame rotation.
:func:`emission_angles`
    Convert Cartesian momentum to emission angles.
:func:`evaluate_self_energy`
    Evaluate the complex retarded self-energy for one causal model.
:func:`expected_counts`
    Assemble deterministic expected detector counts.
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
:func:`hv_map_at_energy`
    Interpolate a photon-energy scan at one sampled binding energy.
:func:`kinetic_energy_ev`
    Compute signed photoelectron kinetic energy and its validity mask.
:func:`kpar_to_detector_angles`
    Convert parallel momentum to detector angles.
:func:`kz_fractional_nodes`
    Build static uniform surface-fractional kz bin centres.
:func:`kz_from_inner_potential`
    Compute complex out-of-plane momentum from the inner potential.
:func:`kz_from_inner_potential_at_fermi`
    Evaluate the named Fermi-level ``kz`` approximation.
:func:`kz_wrapped_lorentzian_bin_weights`
    Integrate wrapped-Lorentzian mass over fractional kz bins.
:func:`lab_polarization_to_sample`
    Convert fixed laboratory polarization to sample coordinates.
:func:`load_vasp_context`
    Load a simulation-ready context from VASP output files.
:func:`map_source_to_detector`
    Convert one source density to native detector bins conservatively.
:func:`normalize_intensity`
    Return an explicit display-only normalization of carrier values.
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
:func:`projected_spectral_density`
    Contract transition sources with a metric-aware spectral matrix.
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
    Simulate the canonical detector raster.
:func:`simulate_arpes_cut`
    Simulate the canonical path-cut detector raster.
:func:`simulate_hv_scan`
    Simulate a single-domain pre-detector photon-energy scan.
:func:`spectral_intensity_eigen`
    Evaluate spectral intensity from eigenvalues and invariant weights.
:func:`spectral_intensity_resolvent`
    Evaluate degeneracy-safe spectral intensity through a linear solve.
:func:`transmission_shape`
    Evaluate positive monotone analyser transmission with fixed mean one.
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
from .counting import (
    fixed_total_probabilities,
    sample_fixed_total_counts,
    sample_poisson_counts,
)
from .crosssections import (
    yeh_lindau_cross_section,
    yeh_lindau_cross_section_table,
    yeh_lindau_orbital_weights,
)
from .detector_response import (
    apply_post_count_response,
    background_density,
    detector_bin_volumes,
    expected_counts,
    sensitivity_field,
)
from .effects import apply_detector_effects, map_source_to_detector
from .factorized import evaluate_spectral_projection
from .generalized_spectral import (
    projected_spectral_density,
    solve_retarded_dyson,
    spectral_density_matrix,
    total_spectral_density,
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
from .kz_broadening import (
    broaden_kz,
    kz_fractional_nodes,
    kz_wrapped_lorentzian_bin_weights,
)
from .oam import compute_oam
from .plane_wave import (
    plane_wave_mask,
    plane_wave_pseudo_amplitude,
    surface_window_transform,
)
from .polarization import (
    build_polarization_vectors,
    contract_experiment_polarization,
    detector_axis_to_sample,
    detector_rotation,
    lab_polarization_to_sample,
    photon_wavevector,
    polarization_from_angles,
    polarization_to_spherical,
    rotate_frame_vectors,
    sample_azimuth_rotation,
)
from .resolution import (
    apply_resolution,
    convolve_energy,
    convolve_kpath,
    convolve_momentum_map,
    gaussian_kernel_1d,
)
from .retarded_self_energy import evaluate_self_energy
from .spectral import (
    assemble_spectral_intensity_bands_chunk,
    assemble_spectral_intensity_chunk,
)
from .spectral_eigen import spectral_intensity_eigen
from .spectral_resolvent import (
    projected_spectral_density_resolvent,
    spectral_intensity_resolvent,
)
from .spectrum import (
    hv_map_at_energy,
    normalize_intensity,
    simulate_arpes,
    simulate_arpes_cut,
    simulate_hv_scan,
)
from .transmission import apply_transmission, transmission_shape
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
    "assemble_spectral_intensity_bands_chunk",
    "assemble_spectral_intensity_chunk",
    "background_density",
    "broaden_kz",
    "build_polarization_vectors",
    "compute_oam",
    "contract_experiment_polarization",
    "convolve_energy",
    "convolve_kpath",
    "convolve_momentum_map",
    "detector_angles_to_kpar",
    "detector_axis_to_sample",
    "detector_bin_volumes",
    "detector_rotation",
    "emission_angles",
    "evaluate_self_energy",
    "evaluate_spectral_projection",
    "expected_counts",
    "fermi_dirac",
    "final_state_k_inv_ang",
    "fixed_total_probabilities",
    "gaussian",
    "gaussian_kernel_1d",
    "hv_map_at_energy",
    "kinetic_energy_ev",
    "kpar_to_detector_angles",
    "kz_fractional_nodes",
    "kz_from_inner_potential",
    "kz_from_inner_potential_at_fermi",
    "kz_wrapped_lorentzian_bin_weights",
    "lab_polarization_to_sample",
    "load_vasp_context",
    "map_source_to_detector",
    "normalize_intensity",
    "photon_wavevector",
    "polarization_from_angles",
    "polarization_to_spherical",
    "prepare_projection",
    "projected_spectral_density_resolvent",
    "projected_spectral_density",
    "plane_wave_mask",
    "plane_wave_pseudo_amplitude",
    "rotate_frame_vectors",
    "run_vasp_workflow",
    "sample_azimuth_rotation",
    "sample_fixed_total_counts",
    "sample_poisson_counts",
    "sensitivity_field",
    "simulate_arpes",
    "simulate_arpes_cut",
    "simulate_hv_scan",
    "spectral_intensity_eigen",
    "spectral_density_matrix",
    "spectral_intensity_resolvent",
    "transmission_shape",
    "surface_window_transform",
    "solve_retarded_dyson",
    "total_spectral_density",
    "voigt",
    "yeh_lindau_cross_section",
    "yeh_lindau_cross_section_table",
    "yeh_lindau_orbital_weights",
]
