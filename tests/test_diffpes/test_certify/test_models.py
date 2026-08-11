"""Validate built-in transformation and owner-handshake registration.

The test checks idempotent eager registration after radial-model retirement.
"""

import subprocess
import sys

from beartype.typing import Any, Dict, Tuple

from diffpes.certify import (
    get_transformation,
    list_handshakes,
    list_transformations,
    register_builtin_models,
    registry_manifest,
    validate_handshake,
)
from diffpes.types import RegistrationHandshake, TransformationContract


class TestRegisterBuiltinModels:
    """Verify :func:`~diffpes.certify.register_builtin_models`."""

    def test_builtin_registration_is_idempotent(self) -> None:
        """Register each transformation and owner handshake exactly once.

        Repeated registration must preserve unique transformation and owner
        identities.

        Notes
        -----
        Call the public registrar twice. Compare collected keys with sets and
        inspect the registered information-loss declarations.
        """
        register_builtin_models()
        register_builtin_models()
        registered_transformations: Tuple[TransformationContract, ...] = (
            list_transformations()
        )
        transformations: Tuple[Tuple[str, str], ...] = tuple(
            (item.transformation_id, item.transformation_version)
            for item in registered_transformations
        )
        handshakes: Tuple[RegistrationHandshake, ...] = list_handshakes()
        owner_ids: Tuple[str, ...] = tuple(
            item.owner_id for item in handshakes
        )
        assert len(transformations) == len(set(transformations))
        assert len(owner_ids) == len(set(owner_ids))
        assert "org.diffpes.detector" in owner_ids
        assert "org.diffpes.kz" in owner_ids
        assert "org.diffpes.matrixel" in owner_ids
        assert "org.diffpes.spectral" in owner_ids
        matrix_element: RegistrationHandshake = next(
            item
            for item in handshakes
            if item.owner_id == "org.diffpes.matrixel"
        )
        assert (
            "org.diffpes.evidence.matrixel.spherical_bessel_values_derivatives"
            in matrix_element.evidence_ids
        )
        assert (
            "org.diffpes.evidence.matrixel.orbital_position_vacuum_momentum"
            in matrix_element.evidence_ids
        )
        assert (
            "org.diffpes.evidence.matrixel.radial_parameter_gradients"
            in matrix_element.evidence_ids
        )
        assert (
            "org.diffpes.evidence.matrixel.dipole_gauge_gradients"
            in matrix_element.evidence_ids
        )
        assert (
            "org.diffpes.evidence.matrixel.hermite_acceleration_not_applicable"
            in matrix_element.evidence_ids
        )
        assert len(matrix_element.evidence_ids) == 41
        assert (
            "org.diffpes.evidence.matrixel.orbital_channel_compile_scaling"
            in matrix_element.evidence_ids
        )
        assert (
            "org.diffpes.evidence.matrixel.chunk_memory_allocation"
            in matrix_element.evidence_ids
        )
        assert (
            "org.diffpes.evidence.matrixel.late_polarization_performance"
            in matrix_element.evidence_ids
        )
        assert (
            "org.diffpes.evidence.matrixel.lifecycle.kspace_kg_e"
            in matrix_element.evidence_ids
        )
        assert (
            "org.diffpes.evidence.matrixel.handoff.transition_rows"
            in matrix_element.evidence_ids
        )
        spectral: RegistrationHandshake = next(
            item
            for item in handshakes
            if item.owner_id == "org.diffpes.spectral"
        )
        assert len(spectral.transformation_refs) == 3
        assert len(spectral.evidence_ids) == 27
        assert (
            "org.diffpes.evidence.spectral.kk.singularity_stress_witness"
            in spectral.evidence_ids
        )
        detector: RegistrationHandshake = next(
            item
            for item in handshakes
            if item.owner_id == "org.diffpes.detector"
        )
        assert len(detector.transformation_refs) == 4
        assert len(detector.evidence_ids) == 28
        assert (
            "org.diffpes.evidence.detector.manufactured.complete_chain"
            in detector.evidence_ids
        )
        assert (
            "org.diffpes.evidence.detector.derivative.coordinate_map_enclosed_interior"
            in detector.evidence_ids
        )
        assert (
            "org.diffpes.evidence.detector.map.signed_permutation_boundary_seams"
            in detector.evidence_ids
        )
        assert (
            "org.diffpes.evidence.detector.map.general_rotation_strict_enclosure"
            in detector.evidence_ids
        )
        destroyed: set[str] = {
            loss
            for entry in registered_transformations
            for loss in entry.destroys
        }
        assert "overall_matrix_element_phase" in destroyed
        assert "absolute_intensity_calibration" in destroyed

    def test_kz_handshake_pins_semantics_and_upstream_handoffs(self) -> None:
        """Bind wrapped-kz and photon-energy scans to exact owner evidence.

        The packaged and live declarations must agree exactly. Their
        transformations record finite-energy centers, the calibrated static
        node count, and the prohibition on a complete all-node carrier.

        Notes
        -----
        Register every built-in identity. Resolve the permanent owner ID and
        inspect both transformation contracts. Validate only the evidence in
        the packaged handshake.
        """
        register_builtin_models()
        manifest: Dict[str, Any] = registry_manifest()
        declaration: Dict[str, Any] = next(
            item
            for item in manifest["handshakes"]
            if item["owner_id"] == "org.diffpes.kz"
        )
        handshake: RegistrationHandshake = next(
            item
            for item in list_handshakes()
            if item.owner_id == "org.diffpes.kz"
        )
        expected_refs: Tuple[str, ...] = (
            "org.diffpes.transform.kz.wrapped_lorentzian_integral@1.0.0",
            "org.diffpes.transform.kz.photon_energy_scan@1.0.0",
        )
        assert handshake.model_refs == ()
        assert handshake.convention_refs == ()
        assert handshake.transformation_refs == expected_refs
        assert tuple(declaration["transformation_refs"]) == expected_refs
        evidence_ids: Tuple[str, ...] = tuple(declaration["evidence_ids"])
        assert handshake.evidence_ids == evidence_ids
        assert len(evidence_ids) == 19
        assert len(set(evidence_ids)) == 19
        assert {
            "org.diffpes.evidence.kz.handoff.kspace",
            "org.diffpes.evidence.kz.handoff.surface",
            "org.diffpes.evidence.kz.handoff.matrixel",
            "org.diffpes.evidence.kz.handoff.spectral",
            "org.diffpes.evidence.kz.handoff.detector",
        } <= set(evidence_ids)

        wrapped: TransformationContract = get_transformation(
            "org.diffpes.transform.kz.wrapped_lorentzian_integral",
            "1.0.0",
        ).contract
        scan: TransformationContract = get_transformation(
            "org.diffpes.transform.kz.photon_energy_scan",
            "1.0.0",
        ).contract
        assert {
            "exact_finite_omega_kz_center",
            "gauge_covariant_surface_reciprocal_folding",
            "g6_calibrated_n_kz_2048_or_explicit_caller_recalibration",
            "checkpointed_node_local_lax_scan",
            "single_k_by_energy_accumulator",
            "no_complete_all_node_carrier",
        } <= set(wrapped.introduces)
        assert "normal_reciprocal_covariant_full_integrand" in wrapped.requires
        assert {
            "normal_integration_coordinate_reciprocal_invariance",
            "fixed_parallel_and_outgoing_final_momenta",
            "hamiltonian_spectral_source_gauge_covariance",
            "physical_surface_repeated_zone_matrix_element_contrast",
        } <= set(wrapped.preserves)
        assert {
            "photon_energy_lax_scan",
            "finite_omega_kinematics_per_scan_sample",
            "flat_auxiliary_memory_beyond_returned_scan",
            "no_five_point_photon_energy_interpolation",
        } <= set(scan.introduces)
        report: Any = validate_handshake(
            handshake,
            evidence_ids=evidence_ids,
        )
        assert bool(report.complete), report.missing_ids

    def test_kz_owner_rejects_missing_upstream_handshakes(self) -> None:
        """Refuse kz registration before its five upstream owners exist.

        A missing prerequisite must prevent the downstream kz owner from
        entering the immutable registry.

        Notes
        -----
        Use a fresh process because the registry is append-only. Invoke the
        private lifecycle boundary directly and require every missing owner to
        appear in the deterministic error.
        """
        program: str = """
from diffpes.certify.models import _register_kz_handshake
try:
    _register_kz_handshake()
except RuntimeError as exc:
    assert str(exc) == (
        'org.diffpes.kz requires exact upstream handshakes: '
        'org.diffpes.kspace, org.diffpes.surface, org.diffpes.matrixel, '
        'org.diffpes.spectral, org.diffpes.detector'
    )
else:
    raise AssertionError('kz owner accepted missing upstream handshakes')
"""
        completed: subprocess.CompletedProcess[str] = subprocess.run(
            [sys.executable, "-c", program],
            check=False,
            capture_output=True,
            text=True,
        )
        assert completed.returncode == 0, completed.stderr

    def test_kz_owner_rejects_drifted_upstream_handshake(self) -> None:
        """Refuse a matching detector label with a drifted declaration.

        A matching owner ID must not conceal a changed upstream semantic
        declaration.

        Notes
        -----
        Register exact packaged declarations for four owners in a fresh
        process and an empty declaration for the detector owner. The kz owner
        must detect semantic drift rather than accepting the matching label.
        """
        program: str = """
from diffpes.certify import register_handshake
from diffpes.certify.models import _packaged_handshake, _register_kz_handshake
from diffpes.types import make_registration_handshake
for owner_id in (
    'org.diffpes.kspace',
    'org.diffpes.surface',
    'org.diffpes.matrixel',
    'org.diffpes.spectral',
):
    register_handshake(_packaged_handshake(owner_id))
register_handshake(make_registration_handshake('org.diffpes.detector'))
try:
    _register_kz_handshake()
except RuntimeError as exc:
    assert str(exc) == (
        'org.diffpes.kz requires exact upstream handshakes: '
        'org.diffpes.detector'
    )
else:
    raise AssertionError('kz owner accepted a drifted upstream handshake')
"""
        completed: subprocess.CompletedProcess[str] = subprocess.run(
            [sys.executable, "-c", program],
            check=False,
            capture_output=True,
            text=True,
        )
        assert completed.returncode == 0, completed.stderr
