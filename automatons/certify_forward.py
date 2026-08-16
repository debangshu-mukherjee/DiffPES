# /// script
# requires-python = ">=3.12,<3.15"
# dependencies = ["diffpes==2026.06.13"]
# ///
"""Certify and reproduce a deterministic compact forward calculation.

The automaton registers a local differentiable scalar forward model through the
public certification registry. It persists a certificate, verifies its internal
relations, and reproduces the stored result through an in-memory artifact
resolver. Smoke mode uses one float64 scalar input.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace

import jax.numpy as jnp
from beartype import beartype
from beartype.typing import Any, Dict, List
from jaxtyping import Array, Float64, jaxtyped

import diffpes as dp


@jaxtyped(typechecker=beartype)
def _compact_forward(
    values: Float64[Array, " n_value"],
) -> Float64[Array, " n_value"]:
    """PRIVATE: Evaluate one deterministic nonlinear forward vector.

    Parameters
    ----------
    values : Float64[Array, " n_value"]
        Input scalar observables in normalized units.

    Returns
    -------
    result : Float64[Array, " n_value"]
        Squared forward observables in normalized units.

    Notes
    -----
    The compact quadratic retains a nonzero JVP and an exact reproduction
    oracle. It keeps certification smoke execution small and deterministic.
    """
    result: Float64[Array, " n_value"] = values**2
    return result


def _register_compact_model(model_id: str) -> None:
    """PRIVATE: Register the compact model only when this process lacks it.

    Parameters
    ----------
    model_id : str
        Reverse-DNS identity for the process-local model.

    Returns
    -------
    None
        The registry receives the model when no exact identity exists.

    Notes
    -----
    Repeated in-process calls reuse the registered model. Independent command
    processes each begin with an empty process-local registration state.
    """
    try:
        dp.certify.get_model(model_id, "1.0.0")
    except KeyError:
        specification: dp.types.ForwardModelSpec
        specification = dp.types.make_forward_model_spec(
            model_id=model_id,
            model_version="1.0.0",
            observable_id="org.diffpes.observable.arpes.intensity",
            implementation_ref="automatons.certify_forward:compact_forward",
            differentiable_paths=("intensity",),
        )
        dp.certify.register_model(specification, _compact_forward)


def _certificate_artifact(
    artifact_id: str,
    value: Float64[Array, " n_value"],
    role: str,
) -> dp.types.ArtifactRef:
    """PRIVATE: Build one resolver-backed certificate artifact reference.

    Parameters
    ----------
    artifact_id : str
        Stable resolver key for the normalized value.
    value : Float64[Array, " n_value"]
        Numeric array represented by the artifact.
    role : str
        Certification artifact role.

    Returns
    -------
    reference : ArtifactRef
        Checked reference with a normalized-content checksum.

    Notes
    -----
    The resolver later supplies the same in-memory arrays. A fixed semantic
    digest denotes this small normalized array representation.
    """
    reference: dp.types.ArtifactRef = dp.types.make_artifact_ref(
        artifact_id=artifact_id,
        media_type="application/x-diffpes-array",
        byte_checksum=None,
        content_checksum=dp.certify.checksum_pytree(
            value,
            record_kind="normalized-content",
        ),
        semantic_checksum=(
            "sha256:1:semantic:"
            "0000000000000000000000000000000000000000000000000000000000000000"
        ),
        locator=None,
        role=role,
    )
    return reference


@dp.harness.experiment(
    name="certify-forward",
    params=(
        dp.types.make_automaton_param(
            "model_id",
            str,
            default="compact-square",
            help="Registered compact forward model identifier.",
            choices=("compact-square",),
            example="compact-square",
        ),
        dp.types.make_automaton_param(
            "evidence_level",
            str,
            default="exploratory",
            help="Certification policy level for the compact model.",
            choices=("exploratory",),
            example="exploratory",
        ),
    ),
    returns={
        "metrics": {
            "verified": {"type": "boolean"},
            "achieved_levels": {"type": "array"},
            "certificate_sha256": {"type": "string"},
        },
        "artifacts": {
            "roles": ["certificate", "certificate_report", "metrics"]
        },
    },
)
def main(
    args: SimpleNamespace,
    ctx: dp.types.AutomatonContext,
) -> Dict[str, Any]:
    """Certify a compact forward value and return persisted evidence.

    The body registers a deterministic model and prepares a CPU manifest. It
    saves and reloads the certificate before verification and reproduction.
    """
    del args.evidence_level
    registry_model_id: str = "org.diffpes.model.automaton.compact_square"
    _register_compact_model(registry_model_id)
    manifest: dp.types.ExecutionManifest = dp.types.make_execution_manifest(
        execution_id="automaton-compact-square",
        model_ref=f"{registry_model_id}@1.0.0",
        schema_version="1.0.0",
        package_version=dp.__version__,
        source_checksum="automaton-compact-forward",
        environment_checksum="cpu-float64",
        backend="cpu",
        precision_policy="float64",
        deterministic=True,
        started_at_utc="2026-08-16T00:00:00Z",
    )
    certification_context: dp.types.CertificationContext = (
        dp.certify.prepare_certification(
            registry_model_id,
            "1.0.0",
            manifest,
            policy_id="org.diffpes.policy.exploratory.v1",
        )
    )
    model_input: Float64[Array, " 1"] = jnp.asarray(
        (2.0,),
        dtype=jnp.float64,
    )
    check_error: Any
    certified: dp.types.CertifiedResult
    check_error, certified = dp.certify.certify_forward_checked(
        certification_context,
        model_input,
        spectrum_rank=1,
    )
    check_error.throw()
    raw_certificate: dp.types.ForwardCertificate = certified.certificate
    complete_certificate: dp.types.ForwardCertificate = (
        dp.types.make_forward_certificate(
            manifest=raw_certificate.manifest,
            model=raw_certificate.model,
            artifacts=(
                _certificate_artifact(
                    "model-input",
                    model_input,
                    "normalized-input",
                ),
                _certificate_artifact(
                    "forward-result",
                    certified.value,
                    "result",
                ),
            ),
            transformations=raw_certificate.transformations,
            evidence=raw_certificate.evidence,
            claims=raw_certificate.claims,
            domains=raw_certificate.domains,
            derivatives=raw_certificate.derivatives,
            dependencies=raw_certificate.dependencies,
            sensitivities=raw_certificate.sensitivities,
            information=raw_certificate.information,
            policy_report=raw_certificate.policy_report,
            policy_id=raw_certificate.policy_id,
            certificate_checksum=raw_certificate.certificate_checksum,
            waivers=raw_certificate.waivers,
        )
    )
    certificate_path: Path = dp.harness.artifact_path(ctx, "certificate.json")
    dp.inout.save_certificate_json(complete_certificate, certificate_path)
    persisted_certificate: dp.types.ForwardCertificate = (
        dp.inout.load_certificate_json(certificate_path)
    )
    verification: dp.types.VerificationReport = dp.certify.verify_certificate(
        persisted_certificate
    )
    resolver: dp.types.ArtifactResolver = dp.certify.mapping_artifact_resolver(
        {
            "model-input": model_input,
            "forward-result": certified.value,
        }
    )
    reproduction: dp.types.ReproductionReport = dp.certify.reproduce_forward(
        persisted_certificate,
        resolver=resolver,
    )
    achieved_levels: List[str] = [
        level
        for level, achieved in zip(
            verification.policy_report.level_ids,
            verification.policy_report.achieved,
            strict=True,
        )
        if bool(achieved)
    ]
    certificate_checksum: str = persisted_certificate.certificate_checksum
    certificate_sha256: str = certificate_checksum.rsplit(":", maxsplit=1)[-1]
    certificate_report: Dict[str, Any] = {
        "certificate_checksum": certificate_checksum,
        "structure_valid": bool(verification.structure_valid),
        "evidence_valid": bool(verification.evidence_valid),
        "reproduced": bool(reproduction.reproduced),
        "reproduction_max_abs_error": float(reproduction.max_abs_error),
        "achieved_levels": achieved_levels,
    }
    report_bytes: bytes = dp.certify.canonical_json(certificate_report)
    report_sha256: str = hashlib.sha256(report_bytes).hexdigest()
    metrics: Dict[str, Any] = {
        "verified": bool(verification.structure_valid)
        and bool(verification.evidence_valid)
        and bool(reproduction.reproduced),
        "achieved_levels": achieved_levels,
        "certificate_sha256": certificate_sha256,
        "certificate_report_sha256": report_sha256,
        "reproduction_max_abs_error": float(reproduction.max_abs_error),
    }
    artifacts: List[dp.types.ArtifactRecord] = [
        dp.harness.record_artifact(
            ctx,
            certificate_path,
            role="certificate",
            mime="application/json",
            preview=True,
        ),
        dp.harness.save_json_artifact(
            ctx,
            "certificate_report.json",
            certificate_report,
            role="certificate_report",
        ),
        dp.harness.save_json_artifact(ctx, "metrics.json", metrics),
    ]
    result: Dict[str, Any] = {"metrics": metrics, "artifacts": artifacts}
    return result


if __name__ == "__main__":
    main()
