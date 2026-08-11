"""Register built-in certified DiffPES forward models.

Extended Summary
----------------
This module defines built-in transformation contracts and domain-owner
handshakes.
Registration is explicit and idempotent. Importing DiffPES does not mutate
the registry.

Routine Listings
----------------
:func:`register_builtin_models`
    Register built-in transformations and owner handshakes.
"""

from beartype import beartype
from beartype.typing import Any, Dict, Tuple
from jaxtyping import jaxtyped

from diffpes.types import (
    RegistrationHandshake,
    TransformationContract,
    make_registration_handshake,
    make_transformation_contract,
)

from .registry import (
    list_handshakes,
    list_transformations,
    register_handshake,
    register_transformation,
    registry_manifest,
)


def _register_transformations() -> None:
    """PRIVATE: Register built-in semantic and information-loss
    contracts.

    Notes
    -----
    Builds the full tuple of built-in transformation contracts. Covers
    amplitude, band, resolution, normalization, k-space, kinematics,
    polarization, geometry, and tight-binding transformations. Then
    registers only the identity-version pairs that
    :func:`~.registry.list_transformations` does not already report.
    The skip step makes repeated registration idempotent against the
    append-only registry.
    """
    contract: Any
    contracts: Tuple[TransformationContract, ...] = (
        make_transformation_contract(
            "org.diffpes.transform.amplitude.intensity",
            "1.0.0",
            requires=("complex_photoemission_amplitude",),
            produces=("arpes_intensity",),
            preserves=("energy_reference", "momentum_coordinates"),
            destroys=("overall_matrix_element_phase",),
            invalidates_claims=("claim.amplitude.phase_recoverable",),
        ),
        make_transformation_contract(
            "org.diffpes.transform.band.incoherent_sum",
            "1.0.0",
            requires=("band_resolved_intensity",),
            produces=("summed_intensity",),
            preserves=("energy_reference", "momentum_coordinates"),
            destroys=("band_component_attribution",),
            invalidates_claims=("claim.band.attribution_preserved",),
        ),
        make_transformation_contract(
            "org.diffpes.transform.resolution.energy_voigt",
            "1.0.0",
            requires=("unbroadened_spectrum",),
            produces=("energy_broadened_spectrum",),
            preserves=("energy_reference", "momentum_coordinates"),
            introduces=("finite_energy_resolution",),
            destroys=("unresolved_energy_information",),
            invalidates_claims=("claim.spectrum.unbroadened",),
        ),
        make_transformation_contract(
            "org.diffpes.transform.normalization.zscore",
            "1.0.0",
            requires=("absolute_intensity",),
            produces=("standardized_intensity",),
            preserves=("energy_reference", "momentum_coordinates"),
            introduces=("dimensionless_standardization",),
            destroys=("absolute_intensity_calibration",),
            invalidates_claims=("claim.intensity.absolute_calibration",),
        ),
        make_transformation_contract(
            "org.diffpes.transform.self_energy.kk_causal",
            "1.0.0",
            requires=(
                "retarded_imaginary_self_energy",
                "declared_relative_energy_domain",
                "declared_subtraction_point",
                "serialized_power2_tail_coordinates",
            ),
            produces=("complex_retarded_self_energy",),
            preserves=(
                "relative_energy_reference",
                "self_energy_parameter_identity",
                "retarded_imaginary_sign",
            ),
            introduces=(
                "once_subtracted_kramers_kronig_real_part",
                "matrix_free_cell_integrated_principal_value",
                "smooth_mode_piecewise_cubic_core",
                "grid_mode_piecewise_linear_hat_core",
                "uniform_even_4096_node_default",
                "kk_selection_domain_minus8_plus8_ev",
                "kk_phase_aligned_integer_cell_domain_extension",
                "power2_tail_gauss_legendre_256_per_side",
                "trusted_interval_two_cell_margin",
                "direct_query_evaluation_without_interpolation",
                "kk_pair_mixed_bound_atol_2e_8_ev_rtol_1e_6",
                "kk_refinement_bounds_value_2e_6_ev_derivative_2e_5_jvp_2e_5_ev",
                "kk_tail_refinement_value_derivative_bound_1e_13",
                "kk_wigner_min_order_1p4_base_error_1e_5_ev",
                "faddeeva_upper_half_plane_closed_radius_1e8",
                "voigt_shared_faddeeva_guard",
                "voigt_zero_width_endpoints_value_only",
            ),
            invalidates_claims=(
                "claim.self_energy.imaginary_only",
                "claim.kk.query_window_defines_domain",
                "claim.kk.truncated_tail",
            ),
        ),
        make_transformation_contract(
            "org.diffpes.transform.spectral.resolvent",
            "1.0.0",
            requires=(
                "complex_hermitian_hamiltonian_relative_to_fermi",
                "complex_retarded_self_energy",
                "complex_transition_source",
                "strictly_positive_regulator",
            ),
            produces=("intrinsic_spectral_intensity",),
            preserves=(
                "relative_energy_reference",
                "degenerate_subspace_unitary_gauge",
                "outgoing_channel_incoherent_reduction",
            ),
            introduces=(
                "complex128_resolvent_linear_solve",
                "degeneracy_safe_hamiltonian_gradient",
                "scalar_retarded_self_energy_broadening",
                "hermitian_relative_tolerance_1e_12",
                "positive_regulator_default_1e_4_ev",
                "checkpointed_nested_k_omega_scan",
                "static_padded_256k_512omega_schedule",
                "static_k_chunk_32_omega_chunk_32",
                "explicit_nonempty_n_out_source_axis",
                "independent_scalar_rhs_solve_before_sum",
                "xla_memory_analysis_allocation_authority",
                "registered_spinless_solve_tape_1p5x_ceiling",
                "one_compile_per_padded_schedule",
            ),
            destroys=("overall_transition_source_phase",),
            invalidates_claims=(
                "claim.spectral.raw_eigenvector_gauge_observable",
                "claim.spectral.instrument_broadened",
            ),
        ),
        make_transformation_contract(
            "org.diffpes.transform.occupation.fermi_at_omega",
            "1.0.0",
            requires=(
                "intrinsic_spectral_intensity",
                "sampled_relative_energy",
                "strictly_positive_temperature",
            ),
            produces=("occupied_intrinsic_spectral_intensity",),
            preserves=(
                "relative_energy_reference",
                "sampled_energy_axis",
                "momentum_coordinates",
            ),
            introduces=("fermi_dirac_occupation_at_sampled_energy",),
            invalidates_claims=(
                "claim.occupation.evaluated_at_band_eigenvalue",
                "claim.spectral.instrument_broadened",
            ),
        ),
        make_transformation_contract(
            "org.diffpes.transform.kz.wrapped_lorentzian_integral",
            "1.0.0",
            requires=(
                "normal_reciprocal_covariant_full_integrand",
                "dimensionless_surface_fractional_kz_nodes",
                "exact_finite_energy_inner_potential_center",
                "positive_intensity_mean_free_path_angstrom",
                "primitive_surface_cell",
            ),
            produces=("kz_broadened_intrinsic_spectral_intensity",),
            preserves=(
                "relative_energy_reference",
                "surface_parallel_momentum",
                "normal_integration_coordinate_reciprocal_invariance",
                "fixed_parallel_and_outgoing_final_momenta",
                "hamiltonian_spectral_source_gauge_covariance",
                "physical_surface_repeated_zone_matrix_element_contrast",
                "constant_observable_unit_mass",
            ),
            introduces=(
                "wrapped_cauchy_analytic_bin_masses",
                "gauge_covariant_surface_reciprocal_folding",
                "gamma_hwhm_one_over_two_mean_free_path",
                "exact_finite_omega_kz_center",
                "fixed_surface_fractional_node_schedule",
                "g6_calibrated_n_kz_2048_or_explicit_caller_recalibration",
                "no_crop_renormalization_or_finite_image_sum",
                "checkpointed_node_local_lax_scan",
                "single_k_by_energy_accumulator",
                "no_complete_all_node_carrier",
            ),
            destroys=("resolved_bulk_kz_attribution",),
            invalidates_claims=(
                "claim.kz.sharp_direct_distribution",
                "claim.kz.coherent_slab_depth_sum",
                "claim.kz.crop_normalized_lorentzian",
            ),
        ),
        make_transformation_contract(
            "org.diffpes.transform.kz.photon_energy_scan",
            "1.0.0",
            requires=(
                "single_domain_pre_detector_intensity_program",
                "sampled_photon_energy_axis",
                "exact_finite_energy_inner_potential_kinematics",
            ),
            produces=("pre_detector_photon_energy_scan",),
            preserves=(
                "relative_energy_reference",
                "surface_parallel_momentum",
                "sampled_photon_energy_axis",
                "explicit_hamiltonian_derivative_authority",
            ),
            introduces=(
                "photon_energy_lax_scan",
                "finite_omega_kinematics_per_scan_sample",
                "elementwise_photon_energy_gradients",
                "flat_auxiliary_memory_beyond_returned_scan",
                "no_five_point_photon_energy_interpolation",
            ),
            invalidates_claims=(
                "claim.kz.fermi_level_center_broadcast_over_energy",
                "claim.kz.scan_includes_detector_response",
            ),
        ),
        make_transformation_contract(
            "org.diffpes.transform.detector.source_density_map",
            "1.0.0",
            requires=(
                "self_describing_source_density",
                "registered_sample_cartesian_frame",
                "experiment_geometry",
                "explicit_detector_calibration",
            ),
            produces=("detector_native_coordinate_density",),
            preserves=(
                "relative_energy_reference",
                "in_aperture_integrated_flux",
                "domain_identity_until_detector_space_mixture",
            ),
            introduces=(
                "active_zyz_domain_rotation",
                "active_sample_to_laboratory_azimuth_rotation",
                "projected_rotation_absolute_determinant_density_factor",
                "analytic_detector_inverse_jacobian",
                "four_point_gauss_legendre_finite_volume_map",
                "signed_diagonal_antidiagonal_boundary_seam_splitting",
                "clamped_linear_exterior_half_source_cells",
                "explicit_boundary_loss_captured_fraction",
                "general_rotation_strict_source_enclosure",
                "general_rotation_four_vs_eight_node_convergence",
                "enclosed_smooth_interior_coordinate_derivative_only",
                "slit_line_density_integrated_over_declared_v_aperture",
            ),
            destroys=("source_grid_sample_identity",),
            invalidates_claims=(
                "claim.detector.source_array_is_detector_raster",
                "claim.detector.target_inferred_from_source_extrema",
                "claim.detector.boundary_loss_renormalized",
                "claim.detector.general_rotation_boundary_intersection_certified",
                "claim.detector.general_rotation_support_crossing_accepted",
                "claim.detector.coordinate_map_topology_switch_differentiable",
            ),
        ),
        make_transformation_contract(
            "org.diffpes.transform.instrument.transmission_fixed_domain",
            "1.0.0",
            requires=(
                "true_kinetic_energy",
                "transmission_raw_slopes",
                "fixed_transmission_calibration_domain",
            ),
            produces=("transmission_weighted_detector_density",),
            preserves=(
                "detector_native_coordinates",
                "relative_energy_reference",
                "calibration_domain_mean_response",
            ),
            introduces=(
                "monotone_integrated_bernstein_log_response",
                "softplus_slope_coordinates",
                "fixed_64_point_gauss_legendre_normalization",
                "fixed_domain_mean_one_response",
                "caller_crop_invariant_transmission",
            ),
            invalidates_claims=(
                "claim.instrument.transmission_normalized_on_query_crop",
                "claim.instrument.transmission_applied_after_resolution",
            ),
        ),
        make_transformation_contract(
            "org.diffpes.transform.instrument.native_detector_resolution",
            "1.0.0",
            requires=(
                "detector_native_coordinate_density",
                "explicit_detector_bin_edges",
                "positive_native_fwhm_widths",
            ),
            produces=("native_resolution_broadened_detector_density",),
            preserves=(
                "detector_coordinate_units",
                "relative_energy_reference",
                "finite_window_flux_accounting",
            ),
            introduces=(
                "piecewise_constant_finite_volume_psf",
                "analytic_gaussian_bin_integrals",
                "separable_native_u_v_energy_resolution",
                "fwhm_to_sigma_single_owner_conversion",
                "boundary_loss_without_row_renormalization",
            ),
            destroys=("sub_psf_detector_structure",),
            invalidates_claims=(
                "claim.instrument.cartesian_k_psf_is_native_angular_psf",
                "claim.instrument.boundary_flux_preserved_by_renormalization",
            ),
        ),
        make_transformation_contract(
            "org.diffpes.transform.detector.expected_counts",
            "1.0.0",
            requires=(
                "post_resolution_detector_density",
                "detector_effects_parameters",
                "explicit_native_bin_volumes",
            ),
            produces=("detector_expected_counts",),
            preserves=(
                "detector_channel_identity",
                "native_detector_bin_identity",
                "nonnegative_rate_domain",
            ),
            introduces=(
                "softmax_detector_space_domain_mixture",
                "nonnegative_detector_background",
                "volume_mean_one_detector_sensitivity",
                "explicit_exposure_and_native_bin_volume",
                "optional_calibrated_post_count_response",
                "explicit_poisson_or_fixed_total_acquisition_mode",
            ),
            destroys=("pre_count_detector_density_units",),
            invalidates_claims=(
                "claim.detector.source_space_domain_mixture",
                "claim.detector.omitted_native_bin_volume",
                "claim.detector.display_normalization_is_likelihood",
            ),
        ),
        make_transformation_contract(
            "org.diffpes.transform.kspace.fractional_cartesian",
            "1.0.0",
            requires=(
                "fractional_reciprocal_coordinates",
                "reciprocal_lattice_basis",
            ),
            produces=("cartesian_reciprocal_coordinates",),
            preserves=(
                "physical_wavevector",
                "convention.reciprocal_basis_rows",
            ),
        ),
        make_transformation_contract(
            "org.diffpes.transform.kinematics.detector_angle_kpar",
            "1.0.0",
            requires=(
                "detector_angles",
                "photoelectron_kinetic_energy",
                "sample_detector_geometry",
            ),
            produces=("parallel_momentum",),
            preserves=(
                "photoelectron_direction",
                "convention.detector_slit_frame",
            ),
        ),
        make_transformation_contract(
            "org.diffpes.transform.kinematics.inner_potential",
            "1.0.0",
            requires=(
                "photoelectron_kinetic_energy",
                "parallel_momentum",
                "inner_potential",
            ),
            produces=("surface_normal_momentum",),
            preserves=(
                "complex_evanescent_branch",
                "convention.positive_kz_branch",
            ),
        ),
        make_transformation_contract(
            "org.diffpes.transform.polarization.lab_polarization_to_sample",
            "1.0.0",
            requires=(
                "laboratory_polarization",
                "sample_orientation",
            ),
            produces=("sample_frame_polarization",),
            preserves=(
                "polarization_norm",
                "polarization_helicity",
                "optical_phase",
                "fixed_beam_across_detector_pixels",
                "convention.sample_orientation_inverse",
            ),
        ),
        make_transformation_contract(
            "org.diffpes.transform.geometry.detector_axis_to_sample",
            "1.0.0",
            requires=(
                "detector_frame_axis",
                "detector_orientation",
                "sample_orientation",
            ),
            produces=("sample_frame_detector_axis",),
            preserves=(
                "axis_norm",
                "detector_axis_identity",
                "convention.detector_to_lab_then_lab_to_sample",
            ),
        ),
        make_transformation_contract(
            "org.diffpes.transform.tightb.bloch_basis_position",
            "1.0.0",
            requires=(
                "validated_hermitian_closed_hopping_list",
                "exact_integer_hopping_cells",
                "basis_fractional_positions",
                "fractional_reciprocal_coordinates",
            ),
            produces=("complex_hermitian_bloch_hamiltonian",),
            preserves=(
                "energy_reference",
                "fractional_reciprocal_coordinates",
                "orbital_spin_basis",
            ),
            introduces=("convention.bloch.basis_position_gauge",),
            destroys=("real_space_hopping_record_attribution",),
            invalidates_claims=(
                "claim.tightb.single_k_real_space_hoppings_recoverable",
            ),
        ),
        make_transformation_contract(
            "org.diffpes.transform.tightb.dos_gaussian",
            "1.0.0",
            requires=(
                "band_eigenvalues",
                "normalized_kpoint_weights",
                "energy_axis",
                "positive_gaussian_width",
            ),
            produces=("gaussian_broadened_density_of_states",),
            preserves=(
                "energy_reference",
                "integrated_state_count",
            ),
            introduces=("finite_gaussian_energy_resolution",),
            destroys=(
                "delta_resolved_spectral_information",
                "kpoint_resolved_attribution",
                "band_resolved_attribution",
            ),
            invalidates_claims=(
                "claim.dos.delta_resolved",
                "claim.dos.kpoint_attribution_preserved",
            ),
        ),
        make_transformation_contract(
            "org.diffpes.transform.tightb.eigensystem_fixed_group",
            "1.0.0",
            requires=("complex_hermitian_bloch_hamiltonian",),
            produces=(
                "sorted_band_eigenvalues",
                "degeneracy_safe_eigensystem",
                "fixed_group_gauge_invariant_observable",
            ),
            preserves=(
                "energy_reference",
                "fractional_reciprocal_coordinates",
                "orbital_spin_basis",
                "degenerate_group_subspace",
            ),
            introduces=(
                "regularized_eigenvector_differential",
                "fixed_group_gauge_invariance",
            ),
            destroys=(
                "individual_eigenvector_phase",
                "degenerate_subspace_basis_choice",
            ),
            invalidates_claims=(
                "claim.band.individual_eigenvector_phase_observable",
                "claim.band.degenerate_sorted_band_derivative",
            ),
        ),
        make_transformation_contract(
            "org.diffpes.transform.tightb.filling_fermi_level",
            "1.0.0",
            requires=(
                "band_eigenvalues",
                "normalized_kpoint_weights",
                "target_filling",
                "positive_electronic_temperature",
            ),
            produces=("finite_temperature_fermi_level",),
            preserves=(
                "energy_reference",
                "target_filling_constraint",
            ),
            introduces=(
                "fermi_dirac_occupation",
                "implicit_root_differential",
            ),
            destroys=(
                "band_resolved_attribution",
                "kpoint_resolved_attribution",
                "full_spectrum_recoverability",
            ),
            invalidates_claims=(
                "claim.fermi_level.full_band_structure_recoverable",
            ),
        ),
        make_transformation_contract(
            "org.diffpes.transform.tightb.depth_carrier",
            "1.0.0",
            requires=(
                "validated_optional_orbital_depths_angstrom",
                "tight_binding_model",
            ),
            produces=("diagonalized_bands_with_optional_orbital_depths",),
            preserves=(
                "depth_values_exactly",
                "depth_ordering",
                "depth_units_angstrom",
                "bulk_none_sentinel",
            ),
        ),
        make_transformation_contract(
            "org.diffpes.transform.tightb.slab_surface",
            "1.0.0",
            requires=(
                "validated_bulk_tight_binding_model",
                "primitive_miller_index",
                "frozen_slab_topology",
                "exact_integer_real_space_cells",
            ),
            produces=(
                "finite_open_slab_model",
                "orbital_depths_angstrom",
                "slab_construction_provenance",
            ),
            preserves=(
                "bulk_parameter_identity",
                "surface_translation_symmetry",
                "basis_position_gauge",
                "wannier_operator_sidecar",
            ),
            introduces=(
                "surface_cartesian_frame",
                "open_normal_boundary",
                "top_surface_depth_origin",
            ),
            destroys=("normal_translation_symmetry",),
            invalidates_claims=(
                "claim.slab.normal_bloch_momentum_observable",
            ),
        ),
        make_transformation_contract(
            "org.diffpes.transform.tightb.surface_projection",
            "1.0.0",
            requires=(
                "orbital_depths_angstrom",
                "positive_intensity_escape_length_angstrom",
                "diagonalized_slab_eigenvectors",
                "registered_isolated_band_groups",
            ),
            produces=(
                "surface_resolved_band_weights",
                "fixed_group_surface_traces",
            ),
            preserves=(
                "complete_group_unitary_gauge",
                "eigenvector_phase_gauge",
                "intensity_escape_length_convention",
            ),
            introduces=("exponential_intensity_depth_weight",),
        ),
    )
    existing: set[Tuple[str, str]] = {
        (item.transformation_id, item.transformation_version)
        for item in list_transformations()
    }
    for contract in contracts:
        key: Tuple[str, str] = (
            contract.transformation_id,
            contract.transformation_version,
        )
        if key not in existing:
            register_transformation(contract)


def _register_kspace_handshake() -> None:
    """PRIVATE: Register the k-space certification handshake
    idempotently.

    Notes
    -----
    Returns without effect when a handshake for the
    ``org.diffpes.kspace`` owner already exists. Otherwise registers the
    owner's versioned coordinate, kinematics, polarization, and geometry
    transformation references together with the named k-space evidence
    identities.
    """
    owner_id: str = "org.diffpes.kspace"
    existing: set[str] = {item.owner_id for item in list_handshakes()}
    if owner_id in existing:
        return
    handshake: Any = make_registration_handshake(
        owner_id=owner_id,
        transformation_refs=(
            "org.diffpes.transform.kspace.fractional_cartesian@1.0.0",
            "org.diffpes.transform.kinematics.detector_angle_kpar@1.0.0",
            "org.diffpes.transform.kinematics.inner_potential@1.0.0",
            "org.diffpes.transform.polarization.lab_polarization_to_sample@1.0.0",
            "org.diffpes.transform.geometry.detector_axis_to_sample@1.0.0",
        ),
        evidence_ids=(
            "org.diffpes.evidence.kspace.graphene.closed_form",
            "org.diffpes.evidence.kspace.damascelli.kinematics",
            "org.diffpes.evidence.kspace.chinook.kz",
            "org.diffpes.evidence.kspace.chinook.tilt",
            "org.diffpes.evidence.kspace.chinook.mesh",
            "org.diffpes.evidence.kspace.polarization.spherical_basis",
        ),
    )
    register_handshake(handshake)


def _register_tightb_handshake() -> None:
    """PRIVATE: Register the tight-binding certification handshake
    idempotently.

    Notes
    -----
    Returns without effect when a handshake for the
    ``org.diffpes.tightb`` owner already exists. Otherwise registers the
    owner's versioned Bloch, eigensystem, density-of-states, and Fermi
    level transformation references together with the named
    tight-binding evidence identities.
    """
    owner_id: str = "org.diffpes.tightb"
    existing: set[str] = {item.owner_id for item in list_handshakes()}
    if owner_id in existing:
        return
    handshake: Any = make_registration_handshake(
        owner_id=owner_id,
        transformation_refs=(
            "org.diffpes.transform.tightb.bloch_basis_position@1.0.0",
            "org.diffpes.transform.tightb.eigensystem_fixed_group@1.0.0",
            "org.diffpes.transform.tightb.dos_gaussian@1.0.0",
            "org.diffpes.transform.tightb.filling_fermi_level@1.0.0",
        ),
        evidence_ids=(
            "org.diffpes.evidence.tightb.hopping_structure",
            "org.diffpes.evidence.tightb.wigner_rotation",
            "org.diffpes.evidence.tightb.slater_koster_table_i",
            "org.diffpes.evidence.tightb.analytic_bands",
            "org.diffpes.evidence.tightb.atomic_soc_kramers",
            "org.diffpes.evidence.tightb.chinook_k_compatibility_resolved",
            "org.diffpes.evidence.tightb.wannier90_normative_ingestion",
            "org.diffpes.evidence.tightb.fixed_group_gauge_invariance",
            "org.diffpes.evidence.tightb.dos_filling_closed_form",
            "org.diffpes.evidence.tightb.generic_parameter_gradients",
            "org.diffpes.evidence.tightb.degenerate_invariant_gradients",
            "org.diffpes.evidence.tightb.eigh_regularization_bias",
            "org.diffpes.evidence.tightb.complex_holomorphic_gradients",
            "org.diffpes.evidence.tightb.fermi_implicit_gradient",
            "org.diffpes.evidence.tightb.bloch_jaxpr_compile_count",
            "org.diffpes.evidence.tightb.batch_memory_shapes",
            "org.diffpes.evidence.tightb.eigvalsh_reverse_memory",
        ),
    )
    register_handshake(handshake)


def _register_slab_surface_handshakes() -> None:
    """PRIVATE: Register the slab carrier and surface handshakes.

    Notes
    -----
    Reads the existing owner set once. Registers ``org.diffpes.slab``
    with its depth-carrier transformation and evidence. Registers
    ``org.diffpes.surface`` with its construction, projection, and
    surface evidence. Skips any handshake whose owner already exists.
    """
    existing: set[str] = {item.owner_id for item in list_handshakes()}
    if "org.diffpes.slab" not in existing:
        register_handshake(
            make_registration_handshake(
                owner_id="org.diffpes.slab",
                transformation_refs=(
                    "org.diffpes.transform.tightb.depth_carrier@1.0.0",
                ),
                evidence_ids=(
                    "org.diffpes.evidence.slab.depth_carrier_persistence",
                    "org.diffpes.evidence.slab.depth_identity_jacobian",
                ),
            )
        )
    if "org.diffpes.surface" not in existing:
        register_handshake(
            make_registration_handshake(
                owner_id="org.diffpes.surface",
                transformation_refs=(
                    "org.diffpes.transform.tightb.slab_surface@1.0.0",
                    "org.diffpes.transform.tightb.surface_projection@1.0.0",
                ),
                evidence_ids=(
                    "org.diffpes.evidence.surface.finite_chain",
                    "org.diffpes.evidence.surface.rotation_covariance",
                    "org.diffpes.evidence.surface.graphene_edges",
                    "org.diffpes.evidence.surface.chinook_slab",
                    "org.diffpes.evidence.surface.primitive_depths",
                    "org.diffpes.evidence.surface.inversion_covariance",
                    "org.diffpes.evidence.surface.open_surface",
                    "org.diffpes.evidence.surface.depth_handoff",
                    "org.diffpes.evidence.surface.surface_projection",
                    "org.diffpes.evidence.surface.exact_operator_gather",
                    "org.diffpes.evidence.surface.incomplete_shell_rejection",
                    "org.diffpes.evidence.surface.fixed_group_gauge",
                    "org.diffpes.evidence.surface.unfolded_graph",
                    "org.diffpes.evidence.surface.acyclic_lifecycle",
                    "org.diffpes.evidence.surface.bulk_parameter_gradients",
                    "org.diffpes.evidence.surface.lattice_depth_gradients",
                    "org.diffpes.evidence.surface.probe_depth_gradients",
                    "org.diffpes.evidence.surface.random_group_gauges",
                    "org.diffpes.evidence.surface.chunked_memory",
                    "org.diffpes.evidence.surface.compile_count",
                ),
            )
        )


def _register_matrixel_handshake() -> None:
    """PRIVATE: Register the matrix-element evidence handshake.

    Notes
    -----
    Returns without effect when a handshake for the
    ``org.diffpes.matrixel`` owner already exists. Otherwise registers
    the owner's versioned amplitude, polarization, and depth-carrier
    transformation references together with the verification,
    derivative, scaling, and lifecycle evidence identities of the
    matrix-element work.
    """
    owner_id: str = "org.diffpes.matrixel"
    existing: set[str] = {item.owner_id for item in list_handshakes()}
    if owner_id in existing:
        return
    verification_evidence: Tuple[str, ...] = (
        "org.diffpes.evidence.matrixel.spherical_bessel_values_derivatives",
        "org.diffpes.evidence.matrixel.radial_integral_dipole_measure",
        "org.diffpes.evidence.matrixel.real_gaunt_dipole_table",
        "org.diffpes.evidence.matrixel.slater_effective_charge",
        "org.diffpes.evidence.matrixel.yeh_lindau_cross_sections",
        "org.diffpes.evidence.matrixel.graphene_interference",
        "org.diffpes.evidence.matrixel.polarization_geometry",
        "org.diffpes.evidence.matrixel.channel_basis_spin_reduction",
        "org.diffpes.evidence.matrixel.mean_free_path_attenuation",
        "org.diffpes.evidence.matrixel.resolvent_dual_convention",
        "org.diffpes.evidence.matrixel.coulomb_functions_phase_shift",
        "org.diffpes.evidence.matrixel.dipole_gauge_equivalence",
        "org.diffpes.evidence.matrixel.free_final_state_hermite_tabulation_rejection",
        "org.diffpes.evidence.matrixel.polarization_basis_geometry",
        "org.diffpes.evidence.matrixel.complete_shell_covariance",
        "org.diffpes.evidence.matrixel.radial_profile_convergence",
        "org.diffpes.evidence.matrixel.band_group_weight_sensitivity",
        "org.diffpes.evidence.matrixel.orbital_position_vacuum_momentum",
    )
    derivative_evidence: Tuple[str, ...] = (
        "org.diffpes.evidence.matrixel.radial_parameter_gradients",
        "org.diffpes.evidence.matrixel.photon_energy_kinematics_gradients",
        "org.diffpes.evidence.matrixel.polarization_geometry_gradients",
        "org.diffpes.evidence.matrixel.orbital_position_gradients",
        "org.diffpes.evidence.matrixel.mean_free_path_gradients",
        "org.diffpes.evidence.matrixel.phase_shift_gradients",
        "org.diffpes.evidence.matrixel.holomorphic_phase_gradients",
        "org.diffpes.evidence.matrixel.eigenvector_group_gradients",
        "org.diffpes.evidence.matrixel.dark_corridor_weight_gradients",
        "org.diffpes.evidence.matrixel.depth_capstone_gradients",
        "org.diffpes.evidence.matrixel.coulomb_assembly_gradients",
        "org.diffpes.evidence.matrixel.dipole_gauge_gradients",
        "org.diffpes.evidence.matrixel.hermite_acceleration_not_applicable",
    )
    scaling_evidence: Tuple[str, ...] = (
        "org.diffpes.evidence.matrixel.orbital_channel_compile_scaling",
        "org.diffpes.evidence.matrixel.chunk_memory_allocation",
        "org.diffpes.evidence.matrixel.late_polarization_performance",
    )
    handshake: Any = make_registration_handshake(
        owner_id=owner_id,
        transformation_refs=(
            "org.diffpes.transform.amplitude.intensity@1.0.0",
            "org.diffpes.transform.polarization.lab_polarization_to_sample@1.0.0",
            "org.diffpes.transform.tightb.depth_carrier@1.0.0",
        ),
        evidence_ids=(
            *verification_evidence,
            *derivative_evidence,
            *scaling_evidence,
            "org.diffpes.evidence.matrixel.spin_axis.incoherent_reduction",
            "org.diffpes.evidence.matrixel.band_group.complete_sensitivity",
            "org.diffpes.evidence.matrixel.radial.certified_profile",
            "org.diffpes.evidence.matrixel.coulomb.phase_assembly",
            "org.diffpes.evidence.matrixel.gauge.length_momentum",
            "org.diffpes.evidence.matrixel.lifecycle.kspace_kg_e",
            "org.diffpes.evidence.matrixel.handoff.transition_rows",
        ),
    )
    register_handshake(handshake)


def _spectral_handshake() -> RegistrationHandshake:
    """PRIVATE: Construct the exact spectral owner handshake.

    Notes
    -----
    Centralizes the immutable upstream identity used both by spectral
    registration and by the detector owner's hard dependency check.
    """
    owner_id: str = "org.diffpes.spectral"
    evidence_ids: Tuple[str, ...] = (
        "org.diffpes.evidence.spectral.faddeeva.full_envelope",
        "org.diffpes.evidence.spectral.voigt.shared_guard",
        "org.diffpes.evidence.spectral.kk.analytic_pair_truth",
        "org.diffpes.evidence.spectral.kk.refinement_convergence",
        "org.diffpes.evidence.spectral.kk.carrier_consistency",
        "org.diffpes.evidence.spectral.kk.derivative_composite_route",
        "org.diffpes.evidence.spectral.kk.reverse_mode_consistency",
        "org.diffpes.evidence.spectral.kk.singularity_stress_witness",
        "org.diffpes.evidence.spectral.kk.spectral_observable_stability",
        "org.diffpes.evidence.spectral.kk.rejected_control_reference",
        "org.diffpes.evidence.spectral.two_pole_closed_form",
        "org.diffpes.evidence.spectral.resolvent_eigen_complex_hermitian",
        "org.diffpes.evidence.spectral.full_line_weight",
        "org.diffpes.evidence.spectral.regulator_limit",
        "org.diffpes.evidence.spectral.resolvent_eigen_consistency",
        "org.diffpes.evidence.spectral.chinook_compatibility",
        "org.diffpes.evidence.spectral.self_energy.causal_model_truth",
        "org.diffpes.evidence.spectral.derivative.faddeeva_voigt",
        "org.diffpes.evidence.spectral.derivative.kk_parameters",
        "org.diffpes.evidence.spectral.derivative.assembly_eta_temperature",
        "org.diffpes.evidence.spectral.derivative.degenerate_resolvent",
        "org.diffpes.evidence.spectral.derivative.complex_adjoint",
        "org.diffpes.evidence.spectral.derivative.causal_parameters",
        "org.diffpes.evidence.spectral.scaling.rematerialized_chunk_memory",
        "org.diffpes.evidence.spectral.scaling.padded_compile_count",
        "org.diffpes.evidence.spectral.scaling.complex128_solve",
        "org.diffpes.evidence.spectral.handoff.matrix_element_transition_rows",
    )
    handshake: RegistrationHandshake = make_registration_handshake(
        owner_id=owner_id,
        transformation_refs=(
            "org.diffpes.transform.self_energy.kk_causal@1.0.0",
            "org.diffpes.transform.spectral.resolvent@1.0.0",
            "org.diffpes.transform.occupation.fermi_at_omega@1.0.0",
        ),
        evidence_ids=evidence_ids,
    )
    return handshake


def _register_spectral_handshake() -> None:
    """PRIVATE: Register the spectral transformation and evidence handshake.

    Notes
    -----
    Returns without effect when ``org.diffpes.spectral`` is already present.
    Otherwise records the exact immutable handshake consumed by the detector
    owner.
    """
    handshake: RegistrationHandshake = _spectral_handshake()
    existing: set[str] = {item.owner_id for item in list_handshakes()}
    if handshake.owner_id in existing:
        return
    register_handshake(handshake)


def _register_detector_handshake() -> None:
    """PRIVATE: Register the detector-chain evidence handshake.

    Notes
    -----
    Requires the exact ``org.diffpes.spectral`` owner handshake before doing
    anything else. Returns without effect when ``org.diffpes.detector`` already
    exists. Otherwise binds conservative detector mapping, fixed-domain
    transmission, native finite-volume resolution, and expected-count
    construction to the complete detector evidence wall.
    """
    expected_upstream: RegistrationHandshake = _spectral_handshake()
    upstream: RegistrationHandshake | None = next(
        (
            item
            for item in list_handshakes()
            if item.owner_id == expected_upstream.owner_id
        ),
        None,
    )
    if upstream != expected_upstream:
        msg: str = (
            "org.diffpes.detector requires the exact "
            "org.diffpes.spectral handshake"
        )
        raise RuntimeError(msg)
    owner_id: str = "org.diffpes.detector"
    existing: set[str] = {item.owner_id for item in list_handshakes()}
    if owner_id in existing:
        return
    evidence_ids: Tuple[str, ...] = (
        "org.diffpes.evidence.detector.resolution.sampled_scipy_parity",
        "org.diffpes.evidence.detector.resolution.finite_volume_energy",
        "org.diffpes.evidence.detector.resolution.finite_volume_angular",
        "org.diffpes.evidence.detector.resolution.finite_volume_path",
        "org.diffpes.evidence.detector.manufactured.source_cube",
        "org.diffpes.evidence.detector.manufactured.complete_chain",
        "org.diffpes.evidence.detector.chinook.compatibility",
        "org.diffpes.evidence.detector.slicer.interpolation",
        "org.diffpes.evidence.detector.map.signed_permutation_boundary_seams",
        "org.diffpes.evidence.detector.map.general_rotation_strict_enclosure",
        "org.diffpes.evidence.detector.acquisition.moments",
        "org.diffpes.evidence.detector.transmission.fixed_domain",
        "org.diffpes.evidence.detector.derivative.resolution_widths",
        "org.diffpes.evidence.detector.derivative.source_temperature",
        "org.diffpes.evidence.detector.derivative.source_geometry",
        "org.diffpes.evidence.detector.derivative.slicer_queries",
        "org.diffpes.evidence.detector.derivative.transmission_coefficients",
        "org.diffpes.evidence.detector.derivative.effects_parameters",
        "org.diffpes.evidence.detector.derivative.coordinate_map_enclosed_interior",
        "org.diffpes.evidence.detector.scaling.driver_memory",
        "org.diffpes.evidence.detector.scaling.rematerialization",
        "org.diffpes.evidence.detector.scaling.compile_count",
        "org.diffpes.evidence.detector.scaling.vmap",
        "org.diffpes.evidence.detector.counterexample.battery",
        "org.diffpes.evidence.detector.lifecycle.zero_legacy",
        "org.diffpes.evidence.detector.lifecycle.documentation_wall",
        "org.diffpes.evidence.detector.handoff.spectral",
        "org.diffpes.evidence.detector.driver.coherent_composition",
    )
    register_handshake(
        make_registration_handshake(
            owner_id=owner_id,
            transformation_refs=(
                "org.diffpes.transform.detector.source_density_map@1.0.0",
                "org.diffpes.transform.instrument.transmission_fixed_domain@1.0.0",
                "org.diffpes.transform.instrument.native_detector_resolution@1.0.0",
                "org.diffpes.transform.detector.expected_counts@1.0.0",
            ),
            evidence_ids=evidence_ids,
        )
    )


def _packaged_handshake(owner_id: str) -> RegistrationHandshake:
    """PRIVATE: Return one immutable packaged owner handshake.

    Parameters
    ----------
    owner_id : str
        Exact owner identity to resolve from the packaged registry manifest.

    Returns
    -------
    handshake : RegistrationHandshake
        Exact declaration stored in the packaged registry manifest.

    Raises
    ------
    RuntimeError
        If the packaged manifest does not declare ``owner_id`` exactly once.

    Notes
    -----
    Plan-owned downstream handshakes compare these complete immutable records,
    rather than accepting a matching owner label with drifted transformations
    or evidence.
    """
    manifest: Dict[str, Any] = registry_manifest()
    declarations: Tuple[Dict[str, Any], ...] = tuple(
        item
        for item in manifest.get("handshakes", ())
        if item.get("owner_id") == owner_id
    )
    if len(declarations) != 1:
        msg: str = f"packaged registry must declare {owner_id} exactly once"
        raise RuntimeError(msg)
    declaration: Dict[str, Any] = declarations[0]
    handshake: RegistrationHandshake = make_registration_handshake(
        owner_id=owner_id,
        model_refs=tuple(declaration["model_refs"]),
        transformation_refs=tuple(declaration["transformation_refs"]),
        convention_refs=tuple(declaration["convention_refs"]),
        evidence_ids=tuple(declaration["evidence_ids"]),
    )
    return handshake


def _register_kz_handshake() -> None:
    """PRIVATE: Register the finite-kz and photon-energy-scan handshake.

    Notes
    -----
    Requires exact packaged k-space, surface, matrix-element, spectral, and
    detector owner records. A matching label with missing or drifted semantic
    evidence cannot release the downstream ``org.diffpes.kz`` owner.
    Registration is otherwise idempotent.
    """
    owner_id: str = "org.diffpes.kz"
    required_upstream: Tuple[str, ...] = (
        "org.diffpes.kspace",
        "org.diffpes.surface",
        "org.diffpes.matrixel",
        "org.diffpes.spectral",
        "org.diffpes.detector",
    )
    live: Dict[str, RegistrationHandshake] = {
        item.owner_id: item for item in list_handshakes()
    }
    invalid_upstream: Tuple[str, ...] = tuple(
        upstream_id
        for upstream_id in required_upstream
        if live.get(upstream_id) != _packaged_handshake(upstream_id)
    )
    if invalid_upstream:
        msg: str = (
            f"{owner_id} requires exact upstream handshakes: "
            + ", ".join(invalid_upstream)
        )
        raise RuntimeError(msg)
    if owner_id in live:
        return
    register_handshake(
        make_registration_handshake(
            owner_id=owner_id,
            transformation_refs=(
                "org.diffpes.transform.kz.wrapped_lorentzian_integral@1.0.0",
                "org.diffpes.transform.kz.photon_energy_scan@1.0.0",
            ),
            evidence_ids=(
                "org.diffpes.evidence.kz.exact_finite_energy_center",
                "org.diffpes.evidence.kz.wrapped_cauchy_voigt_truth",
                "org.diffpes.evidence.kz.direct_delta_boundary",
                "org.diffpes.evidence.kz.quadrature_calibrated_profile",
                "org.diffpes.evidence.kz.oblique_normal_integrand_covariance",
                "org.diffpes.evidence.kz.driver_owned_carriers",
                "org.diffpes.evidence.kz.derivative.mean_free_path",
                "org.diffpes.evidence.kz.derivative.kinematics",
                "org.diffpes.evidence.kz.derivative.nonzero_tripwires",
                "org.diffpes.evidence.kz.scaling.node_local_memory",
                "org.diffpes.evidence.kz.scaling.remat_hv_scan",
                "org.diffpes.evidence.kz.driver.canonical_extension",
                "org.diffpes.evidence.kz.lifecycle.no_all_node_carrier",
                "org.diffpes.evidence.kz.lifecycle.documentation_wall",
                "org.diffpes.evidence.kz.handoff.kspace",
                "org.diffpes.evidence.kz.handoff.surface",
                "org.diffpes.evidence.kz.handoff.matrixel",
                "org.diffpes.evidence.kz.handoff.spectral",
                "org.diffpes.evidence.kz.handoff.detector",
            ),
        )
    )


@jaxtyped(typechecker=beartype)
def register_builtin_models() -> None:
    """Register built-in transformations and owner handshakes.

    The eager operation adds each immutable registry identity at most once.
    Repeated calls preserve the same registry contents.

    :see: :class:`~.test_models.TestRegisterBuiltinModels`

    Notes
    -----
    Registration is explicit and idempotent at the eager application boundary.
    Numerical model execution and domain predicates remain pure JAX programs.
    """
    _register_transformations()
    _register_kspace_handshake()
    _register_matrixel_handshake()
    _register_slab_surface_handshakes()
    _register_spectral_handshake()
    _register_detector_handshake()
    _register_tightb_handshake()
    _register_kz_handshake()


__all__: list[str] = [
    "register_builtin_models",
]
