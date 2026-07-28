"""Validate built-in transformation and owner-handshake registration.

The test checks idempotent eager registration after radial-model retirement.
"""

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
        registered_transformations: tuple[TransformationContract, ...] = (
            list_transformations()
        )
        transformations: tuple[tuple[str, str], ...] = tuple(
            (item.transformation_id, item.transformation_version)
            for item in registered_transformations
        )
        handshakes: tuple[RegistrationHandshake, ...] = list_handshakes()
        owner_ids: tuple[str, ...] = tuple(
            item.owner_id for item in handshakes
        )
        assert len(transformations) == len(set(transformations))
        assert len(owner_ids) == len(set(owner_ids))
        assert "org.diffpes.plan.06" in owner_ids
        plan06: RegistrationHandshake = next(
            item
            for item in handshakes
            if item.owner_id == "org.diffpes.plan.06"
        )
        assert all(
            f"org.diffpes.evidence.06.g{index}" in plan06.evidence_ids
            for index in range(1, 19)
        )
        assert all(
            f"org.diffpes.evidence.06.d{index}" in plan06.evidence_ids
            for index in range(1, 13)
        )
        assert (
            "org.diffpes.evidence.06.d13.not_applicable.g13_rejected"
            in plan06.evidence_ids
        )
        assert "org.diffpes.evidence.06.d13" not in plan06.evidence_ids
        assert all(
            f"org.diffpes.evidence.06.s{index}" in plan06.evidence_ids
            for index in range(1, 4)
        )
        assert (
            "org.diffpes.evidence.06.lifecycle.plan03_kg_e"
            in plan06.evidence_ids
        )
        assert (
            "org.diffpes.evidence.06.handoff.plan07_transition_rows"
            in plan06.evidence_ids
        )
        destroyed: set[str] = {
            loss
            for entry in registered_transformations
            for loss in entry.destroys
        }
        assert "overall_matrix_element_phase" in destroyed
        assert "absolute_intensity_calibration" in destroyed
