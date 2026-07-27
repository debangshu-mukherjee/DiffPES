"""Register built-in certified DiffPES forward models.

Extended Summary
----------------
This module defines the stable scientific identity for the current radial
tight-binding ARPES forward model. It also defines the one-PyTree executor
adapter, versioned transformation contracts, and Plan 03/04 owner handshakes.
Registration is explicit and idempotent. Importing DiffPES does not mutate
the registry.

Routine Listings
----------------
:func:`execute_tb_radial`
    Execute the radial ARPES model from one certification input PyTree.
:func:`register_builtin_models`
    Register built-in models, transformations, and owner handshakes.
:func:`tb_radial_model_spec`
    Return the stable scientific specification for radial ARPES.
"""

import jax.numpy as jnp
from beartype import beartype
from beartype.typing import Any
from jaxtyping import Array, jaxtyped

from diffpes.simul import simulate_tb_radial
from diffpes.types import (
    TB_RADIAL_INPUT_COUNT,
    TB_RADIAL_MODEL_ID,
    TB_RADIAL_MODEL_VERSION,
    ArpesSpectrum,
    DomainResult,
    ForwardModelSpec,
    TransformationContract,
    make_convention_ref,
    make_domain_predicate,
    make_domain_result,
    make_forward_model_spec,
    make_registration_handshake,
    make_transformation_contract,
)

from .checks import list_checks, register_check
from .registry import (
    list_handshakes,
    list_models,
    list_transformations,
    register_handshake,
    register_model,
    register_transformation,
)


@jaxtyped(typechecker=beartype)
def execute_tb_radial(inputs: tuple[Any, ...]) -> ArpesSpectrum:
    """Execute the radial ARPES model from one certification input PyTree.

    The tuple order is ``(bands, radial, simulation, polarization,
    work_function, self_energy, radial_grid, momentum_width)``. Optional
    entries use ``None`` and therefore become empty JAX subtrees.

    :see: :class:`~.test_models.TestExecuteTbRadial`

    Implementation Logic
    --------------------
    1. **Bind the documented output**::

           spectrum: ArpesSpectrum = simulate_tb_radial(
                   bands,
                   radial,
                   simulation,
                   polarization,
                   work_function=work_function,
                   self_energy=self_energy,
                   r_grid=grid,
                   dk=dk,
               )

       The function validates and transforms the inputs before it binds the
       documented output.

    Parameters
    ----------
    inputs : tuple[Any, ...]
        Eight-position radial-model input PyTree. Its tuple structure is
        **static**; a change to the structure triggers retracing.

    Returns
    -------
    spectrum : ArpesSpectrum
        Simulated ARPES intensity and axes in their declared physical units.

    Raises
    ------
    ValueError
        If ``inputs`` does not contain exactly eight entries.

    Notes
    -----
    Gradients pass through every numerical input accepted by
    :func:`~diffpes.simul.simulate_tb_radial`.
    """
    if len(inputs) != TB_RADIAL_INPUT_COUNT:
        msg: str = "tb_radial certification inputs must contain eight entries"
        raise ValueError(msg)
    bands: Any = inputs[0]
    radial: Any = inputs[1]
    simulation: Any = inputs[2]
    polarization: Any = inputs[3]
    work_function: Any = inputs[4]
    self_energy: Any = inputs[5]
    grid: Any = inputs[6]
    dk: Any = inputs[7]
    spectrum: ArpesSpectrum = simulate_tb_radial(
        bands,
        radial,
        simulation,
        polarization,
        work_function=work_function,
        self_energy=self_energy,
        r_grid=grid,
        dk=dk,
    )
    return spectrum


@jaxtyped(typechecker=beartype)
def tb_radial_model_spec() -> ForwardModelSpec:
    """Return the stable scientific specification for radial ARPES.

    The operation uses the built-in radial ARPES adapter and its declared
    conventions. It preserves JAX differentiation through numerical model
    inputs.

    :see: :class:`~.test_models.TestTbRadialModelSpec`

    Returns
    -------
    spec : ForwardModelSpec
        Versioned assumptions, conventions, domain, and differentiable paths.

    Notes
    -----
    The specification identifies the current tight-binding plus radial
    dipole model. It does not itself evaluate or certify a numerical result.
    """
    conventions: tuple = (
        make_convention_ref(
            "org.diffpes.convention.energy.fermi_referenced_ev",
            "1.0.0",
            "{}",
        ),
        make_convention_ref(
            "org.diffpes.convention.length.angstrom",
            "1.0.0",
            "{}",
        ),
        make_convention_ref(
            "org.diffpes.convention.orbital.real_harmonics",
            "1.0.0",
            "{}",
        ),
    )
    domain: tuple = (
        make_domain_predicate(
            "org.diffpes.domain.photon_energy.positive",
            "photon_energy_ev > 0",
            "eV",
            "error",
        ),
        make_domain_predicate(
            "org.diffpes.domain.radial_grid.positive",
            "all(r_grid > 0)",
            "bohr",
            "error",
        ),
    )
    spec: ForwardModelSpec = make_forward_model_spec(
        model_id=TB_RADIAL_MODEL_ID,
        model_version=TB_RADIAL_MODEL_VERSION,
        observable_id="org.diffpes.observable.arpes.intensity",
        implementation_ref="diffpes.simul.simulate_tb_radial",
        assumptions=(
            "dipole_approximation",
            "independent_particle_initial_state",
            "free_electron_final_state",
            "slater_radial_basis",
            "fermi_dirac_occupation",
            "voigt_energy_resolution",
        ),
        conventions=conventions,
        domain=domain,
        differentiable_paths=(
            "bands.eigenvalues",
            "bands.eigenvectors",
            "radial.zeta",
            "simulation.sigma",
            "simulation.gamma",
            "simulation.temperature",
            "simulation.photon_energy",
            "polarization.theta",
            "polarization.phi",
            "work_function",
        ),
        nondifferentiable_paths=(
            "simulation.fidelity",
            "polarization.polarization_type",
        ),
    )
    return spec


def _register_transformations() -> None:
    """Register built-in semantic and information-loss contracts."""
    contract: Any
    contracts: tuple[TransformationContract, ...] = (
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
            produces=("out_of_plane_momentum",),
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
                "in_plane_translation_symmetry",
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
    existing: set[tuple[str, str]] = {
        (item.transformation_id, item.transformation_version)
        for item in list_transformations()
    }
    for contract in contracts:
        key: tuple[str, str] = (
            contract.transformation_id,
            contract.transformation_version,
        )
        if key not in existing:
            register_transformation(contract)


def _register_plan03_handshake() -> None:
    """Register the Plan 03 certification handshake idempotently."""
    owner_id: str = "org.diffpes.plan.03"
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
            "org.diffpes.evidence.03.graphene.closed_form",
            "org.diffpes.evidence.03.damascelli.kinematics",
            "org.diffpes.evidence.03.chinook.kz",
            "org.diffpes.evidence.03.chinook.tilt",
            "org.diffpes.evidence.03.chinook.mesh",
            "org.diffpes.evidence.03.polarization.spherical_basis",
        ),
    )
    register_handshake(handshake)


def _register_plan04_handshake() -> None:
    """Register the Plan 04 certification handshake idempotently."""
    owner_id: str = "org.diffpes.plan.04"
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
            "org.diffpes.evidence.04.g1.hopping_structure",
            "org.diffpes.evidence.04.g2.wigner_rotation",
            "org.diffpes.evidence.04.g3.slater_koster_table_i",
            "org.diffpes.evidence.04.g4.analytic_bands",
            "org.diffpes.evidence.04.g5.atomic_soc_kramers",
            "org.diffpes.evidence.04.g6.chinook_k_compatibility_resolved",
            "org.diffpes.evidence.04.g7.wannier90_normative_ingestion",
            "org.diffpes.evidence.04.g8.fixed_group_gauge_invariance",
            "org.diffpes.evidence.04.g9.dos_filling_closed_form",
            "org.diffpes.evidence.04.d1.generic_parameter_gradients",
            "org.diffpes.evidence.04.d2.degenerate_invariant_gradients",
            "org.diffpes.evidence.04.d3.eigh_regularization_bias",
            "org.diffpes.evidence.04.d4.complex_holomorphic_gradients",
            "org.diffpes.evidence.04.d5.fermi_implicit_gradient",
            "org.diffpes.evidence.04.s1.bloch_jaxpr_compile_count",
            "org.diffpes.evidence.04.s2.batch_memory_shapes",
            "org.diffpes.evidence.04.s3.eigvalsh_reverse_memory",
        ),
    )
    register_handshake(handshake)


def _register_plan05_handshakes() -> None:
    """Register the split Plan 05 carrier and slab handshakes."""
    existing: set[str] = {item.owner_id for item in list_handshakes()}
    if "org.diffpes.plan.05a" not in existing:
        register_handshake(
            make_registration_handshake(
                owner_id="org.diffpes.plan.05a",
                transformation_refs=(
                    "org.diffpes.transform.tightb.depth_carrier@1.0.0",
                ),
                evidence_ids=(
                    "org.diffpes.evidence.05a.g1.depth_carrier_persistence",
                    "org.diffpes.evidence.05a.d1.depth_identity_jacobian",
                ),
            )
        )
    if "org.diffpes.plan.05b" not in existing:
        register_handshake(
            make_registration_handshake(
                owner_id="org.diffpes.plan.05b",
                transformation_refs=(
                    "org.diffpes.transform.tightb.slab_surface@1.0.0",
                    "org.diffpes.transform.tightb.surface_projection@1.0.0",
                ),
                evidence_ids=(
                    "org.diffpes.evidence.05.g1.finite_chain",
                    "org.diffpes.evidence.05.g2.rotation_covariance",
                    "org.diffpes.evidence.05.g3.graphene_edges",
                    "org.diffpes.evidence.05.g4.chinook_slab",
                    "org.diffpes.evidence.05.g5.primitive_depths",
                    "org.diffpes.evidence.05.g6.inversion_covariance",
                    "org.diffpes.evidence.05.g7.open_surface",
                    "org.diffpes.evidence.05.g8.depth_handoff",
                    "org.diffpes.evidence.05.g9.surface_projection",
                    "org.diffpes.evidence.05.g10.exact_operator_gather",
                    "org.diffpes.evidence.05.g11.incomplete_shell_rejection",
                    "org.diffpes.evidence.05.g12.fixed_group_gauge",
                    "org.diffpes.evidence.05.g13.unfolded_graph",
                    "org.diffpes.evidence.05.g14.acyclic_lifecycle",
                    "org.diffpes.evidence.05.d1.bulk_parameter_gradients",
                    "org.diffpes.evidence.05.d2.lattice_depth_gradients",
                    "org.diffpes.evidence.05.d4.probe_depth_gradients",
                    "org.diffpes.evidence.05.d5.random_group_gauges",
                    "org.diffpes.evidence.05.s1.chunked_memory",
                    "org.diffpes.evidence.05.s2.compile_count",
                ),
            )
        )


def _check_positive_photon_energy(inputs: tuple[Any, ...]) -> DomainResult:
    """Check that the radial model uses a positive photon energy."""
    photon_energy: Array = inputs[2].photon_energy
    margin: Array = photon_energy
    result: DomainResult = make_domain_result(
        predicate_id="org.diffpes.domain.photon_energy.positive",
        measured=photon_energy,
        reference=jnp.zeros(()),
        residual=photon_energy,
        tolerance=jnp.zeros(()),
        margin=margin,
        passed=margin > 0.0,
        in_domain=margin > 0.0,
        severity_code=jnp.asarray(2, dtype=jnp.int32),
    )
    return result


def _check_positive_radial_grid(inputs: tuple[Any, ...]) -> DomainResult:
    """Check that every explicit radial-grid point is positive."""
    radial_grid: Any = inputs[6]
    margin: Array = (
        jnp.asarray(1e-6, dtype=jnp.float64)
        if radial_grid is None
        else jnp.min(radial_grid)
    )
    result: DomainResult = make_domain_result(
        predicate_id="org.diffpes.domain.radial_grid.positive",
        measured=margin,
        reference=jnp.zeros(()),
        residual=margin,
        tolerance=jnp.zeros(()),
        margin=margin,
        passed=margin > 0.0,
        in_domain=margin > 0.0,
        severity_code=jnp.asarray(2, dtype=jnp.int32),
    )
    return result


def _register_checks() -> None:
    """Register built-in radial-model domain checks idempotently."""
    check_id: Any
    check_fn: Any
    registered: set[str] = set(list_checks())
    checks: tuple[tuple[str, Any], ...] = (
        (
            "org.diffpes.domain.photon_energy.positive",
            _check_positive_photon_energy,
        ),
        (
            "org.diffpes.domain.radial_grid.positive",
            _check_positive_radial_grid,
        ),
    )
    for check_id, check_fn in checks:
        if check_id not in registered:
            register_check(check_id, check_fn)


@jaxtyped(typechecker=beartype)
def register_builtin_models() -> None:
    """Register built-in models, transformations, and owner handshakes.

    The operation uses the built-in radial ARPES adapter and its declared
    conventions. It preserves JAX differentiation through numerical model
    inputs.

    :see: :class:`~.test_models.TestRegisterBuiltinModels`

    Notes
    -----
    Registration is explicit and idempotent at the eager application boundary.
    Numerical model execution and domain predicates remain pure JAX programs.
    """
    existing: set[tuple[str, str]] = {
        (item.model_id, item.model_version) for item in list_models()
    }
    key: tuple[str, str] = (TB_RADIAL_MODEL_ID, TB_RADIAL_MODEL_VERSION)
    if key not in existing:
        register_model(tb_radial_model_spec(), execute_tb_radial)
    _register_transformations()
    _register_checks()
    _register_plan03_handshake()
    _register_plan04_handshake()
    _register_plan05_handshakes()


__all__: list[str] = [
    "execute_tb_radial",
    "register_builtin_models",
    "tb_radial_model_spec",
]
