"""Validate semantic transformation contracts.

The tests cover public behavior, differentiability, validation, and stable
scientific identity in the supported certification regime.
"""

import pytest
from beartype.typing import Any

from diffpes.certify import (
    compose_transformations,
    validate_composition,
)
from diffpes.types import make_transformation_contract


class TestValidateComposition:
    """Verify :func:`~diffpes.certify.validate_composition`.

    The cases cover the public behavior in the supported certification regime.

    :see: :func:`~diffpes.certify.validate_composition`
    """

    def test_empty_composition_is_valid(self) -> None:
        """Verify an empty pipeline preserves its declared input semantics.

        The case uses explicit inputs in the supported certification regime.
        It checks the public result or the documented failure state.

        Notes
        -----
        Composes no transforms and checks the deterministic report directly.
        """
        report: Any
        report = validate_composition((), initial_semantics=("energy",))
        assert report.valid


class TestComposeTransformations:
    """Verify :func:`~diffpes.certify.compose_transformations`.

    The cases cover the public behavior in the supported certification regime.

    :see: :func:`~diffpes.certify.compose_transformations`
    """

    def test_composition_propagates_information_loss_and_invalidations(
        self,
    ) -> None:
        """Carry explicit semantics and all downstream losses conservatively.

        The case uses explicit inputs in the supported certification regime.
        It checks the public result or the documented failure state.

        Notes
        -----
        The test checks the result with explicit assertions.
        """
        amplitude: Any
        intensity: Any
        report: Any
        amplitude = make_transformation_contract(
            "org.diffpes.transform.matrix_element.amplitude",
            "1.0.0",
            requires=("energy-axis", "orbital-phase"),
            produces=("amplitude",),
            preserves=("energy-axis", "orbital-phase"),
            introduces=("dipole-approximation",),
        )
        intensity = make_transformation_contract(
            "org.diffpes.transform.observable.intensity",
            "1.0.0",
            requires=("amplitude",),
            produces=("intensity",),
            preserves=("energy-axis",),
            destroys=("orbital-phase",),
            invalidates_claims=("org.diffpes.claim.phase.resolved",),
        )
        report = compose_transformations(
            (amplitude, intensity),
            initial_semantics=("energy-axis", "orbital-phase"),
        )
        assert report.valid
        assert report.available_semantics == ("energy-axis", "intensity")
        assert set(report.destroyed_information) == {
            "amplitude",
            "dipole-approximation",
            "orbital-phase",
        }
        assert report.invalidated_claims == (
            "org.diffpes.claim.phase.resolved",
        )

    def test_missing_requirement_is_reported_and_strict_composition_raises(
        self,
    ) -> None:
        """Refuse a transformation whose parent semantics do not satisfy it.

        The case uses explicit inputs in the supported certification regime.
        It checks the public result or the documented failure state.

        Notes
        -----
        The test checks the result with explicit assertions.
        """
        contract: Any
        report: Any
        contract = make_transformation_contract(
            "org.diffpes.transform.resolution.energy",
            "1.0.0",
            requires=("energy-axis",),
            produces=("broadened-intensity",),
        )
        report = validate_composition(
            (contract,), initial_semantics=("intensity",)
        )
        assert not report.valid
        assert "energy-axis" in report.errors[0]
        with pytest.raises(ValueError, match="energy-axis"):
            compose_transformations(
                (contract,), initial_semantics=("intensity",)
            )


class TestValidateContract:
    """Verify :func:`~diffpes.certify.validate_contract`.

    The cases cover the public behavior in the supported certification regime.

    :see: :func:`~diffpes.certify.validate_contract`
    """

    def test_contradictory_contract_is_rejected(self) -> None:
        """Reject a property declared both preserved and destroyed.

        The case uses explicit inputs in the supported certification regime.
        It checks the public result or the documented failure state.

        Notes
        -----
        The test checks the result with explicit assertions.
        """
        with pytest.raises(ValueError, match="preserve/introduce and destroy"):
            make_transformation_contract(
                "org.diffpes.transform.invalid.example",
                "1.0.0",
                preserves=("absolute-intensity",),
                destroys=("absolute-intensity",),
            )

    @pytest.mark.parametrize(
        ("identifier", "version"),
        [("not-an-id", "1.0.0"), ("org.diffpes.transform.valid", "version")],
    )
    def test_contract_identity_and_version_are_validated(
        self, identifier: Any, version: Any
    ) -> None:
        """Require stable reverse-DNS identities and semantic versions.

        The case uses explicit inputs in the supported certification regime.
        It checks the public result or the documented failure state.

        Notes
        -----
        The test checks the result with explicit assertions.
        """
        with pytest.raises(
            ValueError,
            match=r"transformation_(?:id|version)",
        ):
            make_transformation_contract(identifier, version)
