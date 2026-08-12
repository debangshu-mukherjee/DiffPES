"""Validate packaged certification registry resources.

The tests cover deterministic manifests, generated model cards, and live
registry drift detection in the supported certification regime.
"""

import pytest
from beartype.typing import Any, Dict, Tuple

from diffpes.certify import (
    packaged_model_card,
    register_builtin_models,
    registry_manifest,
    render_model_card,
    validate_registry_manifest,
)
from tests._factories import registry_model_spec


class TestRegistryManifest:
    """Verify :func:`~diffpes.certify.registry_manifest`.

    The case reads the packaged manifest without process-local mutation.

    :see: :func:`~diffpes.certify.registry_manifest`
    """

    def test_manifest_has_scientific_handshakes_and_no_retired_model(
        self,
    ) -> None:
        """Read the schema and current owner handshakes from package resources.

        The manifest omits the stale radial model and declares matrix-element,
        spectral, detector, and finite-kz ownership explicitly.

        Notes
        -----
        Compare manifest identities before validating the complete live drift.
        """
        manifest: Dict[str, Any] = registry_manifest()
        assert manifest["schema_version"] == "1.0.0"
        assert manifest["models"] == []
        owners: Tuple[str, ...] = tuple(
            item["owner_id"] for item in manifest["handshakes"]
        )
        assert owners == tuple(sorted(owners))
        assert "org.diffpes.detector" in owners
        assert "org.diffpes.kz" in owners
        assert "org.diffpes.matrixel" in owners
        assert "org.diffpes.spectral" in owners
        matrix_element: Dict[str, Any] = next(
            item
            for item in manifest["handshakes"]
            if item["owner_id"] == "org.diffpes.matrixel"
        )
        assert len(matrix_element["evidence_ids"]) == 41
        assert (
            "org.diffpes.evidence.matrixel.orbital_position_vacuum_momentum"
            in matrix_element["evidence_ids"]
        )
        assert (
            "org.diffpes.evidence.matrixel.hermite_acceleration_not_applicable"
            in matrix_element["evidence_ids"]
        )
        assert len(set(matrix_element["evidence_ids"])) == 41
        assert (
            "org.diffpes.evidence.matrixel.late_polarization_performance"
            in matrix_element["evidence_ids"]
        )
        spectral: Dict[str, Any] = next(
            item
            for item in manifest["handshakes"]
            if item["owner_id"] == "org.diffpes.spectral"
        )
        assert len(spectral["transformation_refs"]) == 3
        assert len(spectral["evidence_ids"]) == 27
        detector: Dict[str, Any] = next(
            item
            for item in manifest["handshakes"]
            if item["owner_id"] == "org.diffpes.detector"
        )
        assert len(detector["transformation_refs"]) == 4
        assert len(detector["evidence_ids"]) == 28
        kz: Dict[str, Any] = next(
            item
            for item in manifest["handshakes"]
            if item["owner_id"] == "org.diffpes.kz"
        )
        assert len(kz["transformation_refs"]) == 2
        assert len(kz["evidence_ids"]) == 19


class TestRenderModelCard:
    """Verify :func:`~diffpes.certify.render_model_card`.

    The cases render Markdown directly from a model specification.

    :see: :func:`~diffpes.certify.render_model_card`
    """

    def test_card_contains_exact_model_identity(self) -> None:
        """Render an exact model identity and its scientific fields.

        The case uses an isolated model specification.

        Notes
        -----
        Compare the generated header and required registry fields.
        """
        spec: Any = registry_model_spec("render_card")
        card: str = render_model_card(spec)
        assert card.startswith("# org.diffpes.model.registry_test.render_card")
        assert "Version: `1.0.0`." in card
        assert "Observable: `org.diffpes.observable.arpes.intensity`." in card
        assert "Implementation: `tests.registry:render_card`." in card


class TestPackagedModelCard:
    """Verify :func:`~diffpes.certify.packaged_model_card`.

    The cases read generated model cards from package resources.

    :see: :func:`~diffpes.certify.packaged_model_card`
    """

    def test_missing_card_raises_file_not_found(self) -> None:
        """Reject a model identity without a packaged card.

        The case uses an identity outside the packaged manifest.

        Notes
        -----
        Confirm the resource layer reports a missing generated card.
        """
        with pytest.raises(
            FileNotFoundError,
            match=r"registry_test\.missing@1\.0\.0\.md",
        ):
            packaged_model_card(
                "org.diffpes.model.registry_test.missing",
                "1.0.0",
            )


class TestValidateRegistryManifest:
    """Verify :func:`~diffpes.certify.validate_registry_manifest`.

    The case checks every packaged entry and generated model card for drift.

    :see: :func:`~diffpes.certify.validate_registry_manifest`
    """

    def test_builtin_registry_has_no_packaged_drift(self) -> None:
        """Find no missing built-in entry or changed generated model card.

        The validator must return an empty tuple after built-in registration.

        Notes
        -----
        The test registers all built-ins before it validates the package files.
        """
        register_builtin_models()
        assert validate_registry_manifest() == ()
