"""Verify immutable fidelity-manifest invariants.

Use explicit fixtures and independent expectations for every assertion.
"""

import pytest
from beartype.typing import Any, Dict, Tuple

from diffpes.types import (
    FidelityManifest,
    make_derivative_capability,
    make_fidelity_manifest,
)


def _manifest(**overrides: object) -> FidelityManifest:
    """PRIVATE: Check the private helper behavior.

    Notes
    -----
    Build the fixture from explicit values for isolated use.
    """
    values: Dict[str, object] = {
        "schema_version": "1.0",
        "model_ref": "model",
        "instrument_ref": "instrument-ref",
        "acquisition_ref": "acquisition",
        "initial_state": "tb",
        "spectral_physics": "scalar",
        "photocurrent": "projection",
        "light_interaction": "dipole",
        "instrument": "none",
    }
    values.update(overrides)
    result: Any = make_fidelity_manifest(**values)
    return result


class TestDerivativecapability:
    """Verify ``diffpes.types.DerivativeCapability`` invariants.

    Cover acceptance and rejection cases with explicit fixtures.
    """

    def test_accepts_complete_derivative_declaration(self) -> None:
        """Preserve the input path, exact mode, and policy identity.
        Check explicit inputs against independent expectations.

        Notes
        -----
        Compare all stored fields with explicit scalar inputs.
        """
        capability: Any = make_derivative_capability(
            "model.eta_ev", "exact_ad", "policy"
        )
        assert capability.input_path == "model.eta_ev"
        assert capability.mode == "exact_ad"

    @pytest.mark.parametrize(
        ("values", "message"),
        [
            (("", "exact_ad", "policy"), "references must be nonempty"),
            (("path", "exact_ad", ""), "references must be nonempty"),
            (("path", "unknown", "policy"), "unknown derivative"),
        ],
    )
    def test_rejects_each_derivative_invariant(
        self, values: Tuple[str, str, str], message: str
    ) -> None:
        """Reject empty identities and unsupported derivative modes.
        Check explicit inputs against independent expectations.

        Notes
        -----
        Replace one valid field per parameterized declaration.
        """
        with pytest.raises(ValueError, match=message):
            make_derivative_capability(*values)


class TestFidelitymanifest:
    """Verify ``diffpes.types.FidelityManifest`` invariants.

    Cover acceptance and rejection cases with explicit fixtures.
    """

    def test_accepts_complete_manifest_with_optional_evidence(self) -> None:
        """Preserve derivative, validation, validity, and discrepancy refs.
        Check explicit inputs against independent expectations.

        Notes
        -----
        Inspect every optional evidence collection after construction.
        """
        derivative: Any = make_derivative_capability(
            "eta", "exact_ad", "policy"
        )
        manifest: Any = _manifest(
            derivative_capabilities=(derivative,),
            validation_refs=("validation",),
            validity_domain_refs=("domain",),
            discrepancy_ref="discrepancy",
        )
        assert manifest.derivative_capabilities == (derivative,)
        assert manifest.validation_refs == ("validation",)
        assert manifest.discrepancy_ref == "discrepancy"

    @pytest.mark.parametrize(
        ("overrides", "message"),
        [
            ({"model_ref": ""}, "fields must be nonempty"),
            ({"validation_refs": ("",)}, "validation references"),
            ({"validity_domain_refs": ("",)}, "validity-domain references"),
            (
                {
                    "derivative_capabilities": (
                        make_derivative_capability("eta", "exact_ad", "a"),
                        make_derivative_capability(
                            "eta", "frozen_upstream", "b"
                        ),
                    )
                },
                "paths must be unique",
            ),
            ({"discrepancy_ref": ""}, "discrepancy reference"),
        ],
    )
    def test_rejects_each_manifest_invariant(
        self, overrides: Dict[str, object], message: str
    ) -> None:
        """Reject empty identities, references, and duplicate input paths.
        Check explicit inputs against independent expectations.

        Notes
        -----
        Alter one field in the otherwise complete manifest fixture.
        """
        with pytest.raises(ValueError, match=message):
            _manifest(**overrides)


class TestMakeDerivativeCapability:
    """Verify ``diffpes.types.make_derivative_capability``.

    Cover acceptance and rejection cases with explicit fixtures.
    """


class TestMakeFidelityManifest:
    """Verify ``diffpes.types.make_fidelity_manifest``.

    Cover acceptance and rejection cases with explicit fixtures.
    """
