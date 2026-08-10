"""Validate built-in transformation and owner-handshake registration.

The test checks idempotent eager registration after radial-model retirement.
"""

from beartype.typing import Tuple

from diffpes.certify import (
    list_handshakes,
    list_transformations,
    register_builtin_models,
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
