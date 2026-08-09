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
from beartype.typing import Any, Tuple
from jaxtyping import jaxtyped

from diffpes.types import (
    TransformationContract,
    make_registration_handshake,
    make_transformation_contract,
)

from .registry import (
    list_handshakes,
    list_transformations,
    register_handshake,
    register_transformation,
)


def _register_transformations() -> None:
    """PRIVATE: Register built-in semantic and information-loss
    contracts.

    Notes
    -----
    Builds the full tuple of built-in transformation contracts
    (amplitude, band, resolution, normalization, k-space, kinematics,
    polarization, geometry, and tight-binding transformations), then
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
    Reads the existing owner set once, then registers the
    ``org.diffpes.slab`` handshake (depth-carrier transformation and
    depth evidence) and the ``org.diffpes.surface`` handshake (slab
    construction and surface-projection transformations with the
    surface evidence identities), each only when its owner is not
    present.
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
    _register_tightb_handshake()


__all__: list[str] = [
    "register_builtin_models",
]
