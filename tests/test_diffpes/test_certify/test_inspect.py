"""Validate stable human-readable certificate inspection.

The tests cover public behavior, differentiability, validation, and stable
scientific identity in the supported certification regime.
"""

import pytest
from beartype.typing import Any

from diffpes.certify import (
    diff_certificates,
    explain_claim,
    summarize_certificate,
)
from tests._factories import sample_forward_certificate


class TestSummarizeCertificate:
    """Verify :func:`~diffpes.certify.summarize_certificate`.

    The cases cover the public behavior in the supported certification regime.

    :see: :func:`~diffpes.certify.summarize_certificate`
    """

    def test_summary_identifies_model(self) -> None:
        """Verify summaries expose the permanent scientific model identity.

        The case uses explicit inputs in the supported certification regime.
        It checks the public result or the documented failure state.

        Notes
        -----
        The test builds the shared certificate fixture and searches its
        compact text.
        """
        summary: str = summarize_certificate(sample_forward_certificate())
        assert "org.diffpes.model.arpes.test" in summary

    def test_summary_answers_certification_questions(
        self,
    ) -> None:
        """Verify the summary answers key certification questions.

        The case uses explicit inputs in the supported certification regime.
        It checks the public result or the documented failure state.

        Notes
        -----
        The test checks the result with explicit assertions.
        """
        summary: Any
        summary = summarize_certificate(sample_forward_certificate())

        assert "Model: org.diffpes.model.arpes.test@1.0.0" in summary
        assert "Observable: org.diffpes.observable.arpes.intensity" in summary
        assert (
            "org.diffpes.convention.energy.fermi_referenced_ev@1.0.0"
            in summary
        )
        assert "bands [initial_state; application/x-vasp-eigenval]" in summary
        assert "destroys: overall_phase" in summary
        assert "Claims: passed=1" in summary
        assert (
            "Achieved levels: identified, validated, differentiable, verified"
            in summary
        )
        assert "Unachieved levels: benchmarked, reproducible" in summary
        assert "claim.output.finite: passed (margin=0.5)" in summary
        assert "jax.linearize+jvp+vjp+central_fd" in summary
        assert "Structural dependencies: 2/2 input-output pairs" in summary
        assert "Locally active sensitivities: 2/2 projections" in summary
        assert "effective_rank=2" in summary
        assert "/private/data" not in summary


class TestExplainClaim:
    """Verify :func:`~diffpes.certify.explain_claim`.

    The cases cover the public behavior in the supported certification regime.

    :see: :func:`~diffpes.certify.explain_claim`
    """

    def test_explain_claim_reports_bounded_numerical_evidence(self) -> None:
        """Verify explain claim reports bounded numerical evidence.

        The case uses explicit inputs in the supported certification regime.
        It checks the public result or the documented failure state.

        Notes
        -----
        The test checks the result with explicit assertions.
        """
        rationale: Any
        rationale = explain_claim(
            sample_forward_certificate(), "claim.output.finite"
        )

        assert "Status: passed" in rationale
        assert "Margin: 0.5" in rationale
        assert "reference-spectrum" in rationale
        assert "method: org.diffpes.method.reference" in rationale
        assert "lineage independence: qualified" in rationale
        assert "generators: reference.generator" in rationale
        assert "tolerance: shape=(2,), values=[1.e-08, 1.e-08]" in rationale

    def test_explain_claim_rejects_missing_id(self) -> None:
        """Verify explain claim rejects missing id.

        The case uses explicit inputs in the supported certification regime.
        It checks the public result or the documented failure state.

        Notes
        -----
        The test checks the result with explicit assertions.
        """
        with pytest.raises(KeyError, match="not present"):
            explain_claim(sample_forward_certificate(), "claim.missing")


class TestDiffCertificates:
    """Verify :func:`~diffpes.certify.diff_certificates`.

    The cases cover the public behavior in the supported certification regime.

    :see: :func:`~diffpes.certify.diff_certificates`
    """

    def test_diff_categorizes_environment_audit_and_scientific_changes(
        self,
    ) -> None:
        """Verify diff categorizes environment audit and scientific changes.

        The case uses explicit inputs in the supported certification regime.
        It checks the public result or the documented failure state.

        Notes
        -----
        The test checks the result with explicit assertions.
        """
        original: Any
        audit_changed: Any
        scientific_changed: Any
        environment_manifest: Any
        audit: Any
        scientific: Any
        environment: Any
        original = sample_forward_certificate()
        audit_changed = sample_forward_certificate(
            execution_id="run-002",
            started_at_utc="2026-07-21T13:00:00Z",
        )
        scientific_changed = sample_forward_certificate(model_version="1.1.0")
        environment_manifest = sample_forward_certificate(
            environment_checksum=(
                "sha256:1:environment:"
                "0000000000000000000000000000000000000000000000000000000000000000"
            )
        )

        audit = diff_certificates(original, audit_changed)
        scientific = diff_certificates(original, scientific_changed)
        environment = diff_certificates(original, environment_manifest)

        assert audit.scientific == ()
        assert audit.audit == ("execution_id", "started_at_utc")
        assert scientific.scientific == ("model",)
        assert scientific.environment == ()
        assert environment.environment == ("environment_checksum",)
        assert "audit: execution_id, started_at_utc" in audit.summary

    def test_diff_identical_certificate_has_clear_summary(self) -> None:
        """Verify diff identical certificate has clear summary.

        The case uses explicit inputs in the supported certification regime.
        It checks the public result or the documented failure state.

        Notes
        -----
        The test checks the result with explicit assertions.
        """
        certificate: Any
        difference: Any
        certificate = sample_forward_certificate()
        difference = diff_certificates(certificate, certificate)

        assert difference.identical
        assert difference.summary == "Certificates are identical."
