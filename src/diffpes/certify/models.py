r"""Register built-in certified DiffPES forward models.

Extended Summary
----------------
This module defines built-in domain-owner handshakes.
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
    make_registration_handshake,
)

from .builtin_transformations import _register_transformations
from .registry import (
    list_handshakes,
    register_handshake,
)
from .registry_resources import registry_manifest


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
    Downstream handshakes compare these complete immutable records,
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


__all__: list[str] = ["register_builtin_models"]
