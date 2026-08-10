"""Render forward-model certificates in a human-readable form.

Extended Summary
----------------
This module provides deterministic, compact views of a ``ForwardCertificate``.
The views cover the model, provenance, claims, differentiability evidence, and
policy outcome. Inspection is deliberately an eager boundary utility:
it formats already-computed certificate leaves and never evaluates a physics
claim or loads the associated forward result.

Routine Listings
----------------
:func:`diff_certificates`
    Compare two certificates by scientific meaning and record class.
:func:`explain_claim`
    Explain one claim and the numerical evidence supporting it.
:func:`summarize_certificate`
    Return a deterministic human-readable certificate summary.
"""

from __future__ import annotations

import numpy as np
from beartype import beartype
from beartype.typing import Any, Dict, Tuple
from jaxtyping import Bool, Shaped, jaxtyped
from numpy.typing import NDArray

from diffpes.types import (
    CERTIFICATE_ARRAY_PREVIEW_ITEMS,
    CertificateDiff,
    ForwardCertificate,
    make_certificate_diff,
)

from .canonical import canonical_pytree
from .policy import evidence_is_independent


def _scalar_bool(value: Any) -> bool:
    """PRIVATE: Convert one concrete scalar array to ``bool`` for
    display.

    Parameters
    ----------
    value : Any
        Concrete zero-dimensional array or Python scalar.

    Returns
    -------
    result : bool
        Python Boolean value of the scalar.

    Raises
    ------
    ValueError
        If the value is not zero-dimensional.

    Notes
    -----
    Materializes through :func:`numpy.asarray` and extracts the single
    item, so claim status logic works on plain Python Booleans.
    """
    array: Shaped[NDArray, "..."] = np.asarray(value)
    if array.shape != ():
        msg: str = (
            f"expected scalar certificate field, received shape {array.shape}"
        )
        raise ValueError(msg)
    result: bool = bool(array.item())
    return result


def _scalar_text(value: Any) -> str:
    """PRIVATE: Format one concrete scalar numerical certificate field.

    Parameters
    ----------
    value : Any
        Concrete numerical certificate leaf.

    Returns
    -------
    text : str
        Compact rendering: a non-scalar value becomes a shape and dtype
        placeholder, a float uses the ``.8g`` format, and any other
        scalar uses ``str``.

    Notes
    -----
    The fixed ``.8g`` float format keeps summaries deterministic across
    platforms and print options.
    """
    array: Shaped[NDArray, "..."] = np.asarray(value)
    if array.shape != ():
        text: str = f"array(shape={array.shape}, dtype={array.dtype})"
        return text  # noqa: RET504
    item: Any = array.item()
    if isinstance(item, float):
        text = f"{item:.8g}"
        return text  # noqa: RET504
    text: str = str(item)
    return text  # noqa: RET504


def _array_text(value: Any) -> str:
    """PRIVATE: Format a bounded preview of a numerical evidence array.

    Parameters
    ----------
    value : Any
        Concrete numerical evidence array.

    Returns
    -------
    text : str
        ``shape=..., values=[...]`` preview of at most
        ``CERTIFICATE_ARRAY_PREVIEW_ITEMS`` leading flattened values at
        precision 8. A truncated preview ends with an ellipsis marker.

    Notes
    -----
    The bound keeps claim explanations readable for large evidence
    arrays while the full shape stays visible.
    """
    array: Shaped[NDArray, "..."] = np.asarray(value)
    flat: Shaped[NDArray, " n_flat"] = array.reshape(-1)
    preview: str = np.array2string(
        flat[:CERTIFICATE_ARRAY_PREVIEW_ITEMS],
        separator=", ",
        precision=8,
        suppress_small=False,
    )
    if flat.size > CERTIFICATE_ARRAY_PREVIEW_ITEMS:
        preview = f"{preview[:-1]}, ...]"
    text: str = f"shape={array.shape}, values={preview}"
    return text  # noqa: RET504


def _claim_status(claim: Any) -> str:
    """PRIVATE: Return the bounded status label for one claim.

    Parameters
    ----------
    claim : Any
        Claim record with scalar ``checked``, ``in_domain``, and
        ``passed`` leaves.

    Returns
    -------
    status : str
        One of ``"not_checked"``, ``"out_of_domain"``, ``"passed"``, or
        ``"failed"``, tested in that order of precedence.

    Notes
    -----
    An unchecked claim reports ``not_checked`` even with a true pass
    flag. This rule stops summaries from presenting unevaluated claims
    as outcomes.
    """
    if not _scalar_bool(claim.checked):
        status: str = "not_checked"
        return status  # noqa: RET504
    if not _scalar_bool(claim.in_domain):
        status = "out_of_domain"
        return status  # noqa: RET504
    if _scalar_bool(claim.passed):
        status = "passed"
        return status  # noqa: RET504
    status: str = "failed"
    return status  # noqa: RET504


def _optional_static_tuple(value: object, name: str) -> Tuple[str, ...]:
    """PRIVATE: Read an optional tuple-valued static inspection field.

    Parameters
    ----------
    value : object
        Record that may carry the named field.
    name : str
        Attribute name to read.

    Returns
    -------
    result : Tuple[str, ...]
        String forms of the tuple entries, or the empty tuple when the
        field is absent or not a tuple.

    Notes
    -----
    Tolerant reading lets the summary render certificates from schema
    versions that lack an optional field.
    """
    field_value: object = getattr(value, name, ())
    if not isinstance(field_value, tuple):
        result: Tuple[str, ...] = ()
        return result
    result: Tuple[str, ...] = tuple(str(item) for item in field_value)
    return result


def _append_tuple(
    lines: list[str],
    heading: str,
    values: Tuple[str, ...],
) -> None:
    """PRIVATE: Append one heading and its static values when nonempty.

    Parameters
    ----------
    lines : list[str]
        Summary lines to extend in place.
    heading : str
        Section heading without a trailing colon.
    values : Tuple[str, ...]
        Values to list under the heading.

    Notes
    -----
    Writes the heading with a colon and one indented dash line per
    value. An empty tuple appends nothing, so empty sections leave no
    trace in the summary.
    """
    if not values:
        return
    lines.append(f"{heading}:")
    lines.extend(f"  - {value}" for value in values)


@jaxtyped(typechecker=beartype)
def summarize_certificate(certificate: ForwardCertificate) -> str:
    r"""Return a deterministic human-readable certificate summary.

    The inspection result answers a bounded scientific question without loading
    the associated result arrays. It excludes private source locators.

    :see: :class:`~.test_inspect.TestSummarizeCertificate`


    Implementation Logic
    --------------------
    1. **Bind the documented output**::

           summary: str = "\n".join(lines)

       The function validates and transforms the inputs before it binds the
       documented output.

    Parameters
    ----------
    certificate : ForwardCertificate
        Certificate to summarize. The function does not need or load the
        associated result arrays.

    Returns
    -------
    summary : str
        Compact multiline summary of identity, semantics, provenance, claim
        status, and derivative diagnostics.
    """
    item: Any
    claim: Any
    model: Any = certificate.model
    manifest: Any = certificate.manifest
    lines: list[str] = [
        "Forward certificate",
        f"Model: {model.model_id}@{model.model_version}",
        f"Observable: {model.observable_id}",
        f"Implementation: {model.implementation_ref}",
        f"Schema: {manifest.schema_version}",
        f"Policy: {certificate.policy_id}",
        f"Execution: {manifest.execution_id} at {manifest.started_at_utc}",
        "Environment: "
        f"package={manifest.package_version}, backend={manifest.backend}, "
        f"precision={manifest.precision_policy}, "
        f"deterministic={manifest.deterministic}",
    ]

    policy_report: Any = certificate.policy_report
    achieved_flags: Tuple[bool, ...] = tuple(
        bool(item) for item in np.asarray(policy_report.achieved).tolist()
    )
    levels: Tuple[str, ...] = tuple(
        level
        for level, achieved in zip(
            policy_report.level_ids,
            achieved_flags,
            strict=True,
        )
        if achieved
    )
    if levels:
        lines.append(f"Achieved levels: {', '.join(levels)}")
    unachieved: Tuple[str, ...] = tuple(
        level
        for level, achieved in zip(
            policy_report.level_ids,
            achieved_flags,
            strict=True,
        )
        if not achieved
    )
    if unachieved:
        lines.append(f"Unachieved levels: {', '.join(unachieved)}")

    unmet_requirements: Tuple[str, ...] = tuple(
        claim_id
        for claim_id, passed, checked, in_domain in zip(
            policy_report.required_claim_ids,
            np.asarray(policy_report.claim_passed).tolist(),
            np.asarray(policy_report.claim_checked).tolist(),
            np.asarray(policy_report.claim_in_domain).tolist(),
            strict=True,
        )
        if not (passed and checked and in_domain)
    )
    if unmet_requirements:
        lines.append(f"Unmet requirements: {', '.join(unmet_requirements)}")

    _append_tuple(lines, "Assumptions", tuple(model.assumptions))

    if model.conventions:
        lines.append("Conventions:")
        lines.extend(
            f"  - {item.convention_id}@{item.version}"
            for item in model.conventions
        )

    if certificate.artifacts:
        lines.append("Artifacts:")
        lines.extend(
            f"  - {item.artifact_id} [{item.role}; {item.media_type}]"
            for item in certificate.artifacts
        )

    if certificate.transformations:
        lines.append("Transformations:")
        for item in certificate.transformations:
            lines.append(
                f"  - {item.transformation_id}@{item.transformation_version}"
            )
            if item.destroys:
                lines.append(f"    destroys: {', '.join(item.destroys)}")
            if item.invalidates_claims:
                lines.append(
                    f"    invalidates: {', '.join(item.invalidates_claims)}"
                )

    status_counts: Dict[str, int] = {
        "passed": 0,
        "failed": 0,
        "not_checked": 0,
        "out_of_domain": 0,
    }
    for claim in certificate.claims:
        status_counts[_claim_status(claim)] += 1
    lines.append(
        "Claims: "
        + ", ".join(
            f"{status}={count}" for status, count in status_counts.items()
        )
    )
    lines.extend(
        f"  - {claim.claim_id}: {_claim_status(claim)} "
        f"(margin={_scalar_text(claim.margin)})"
        for claim in certificate.claims
    )

    derivatives: Any = certificate.derivatives
    dependency_structural: Bool[NDArray, "n_output n_input"] = np.asarray(
        certificate.dependencies.structural
    )
    dependency_traced: Bool[NDArray, "n_output n_input"] = np.asarray(
        certificate.dependencies.traced
    )
    sensitivity_active: Bool[NDArray, "n_output n_input"] = np.asarray(
        certificate.sensitivities.active
    )
    information: Any = certificate.information
    lines.extend(
        (
            f"Derivative method: {derivatives.method}",
            "Derivative inputs: "
            + (
                ", ".join(derivatives.input_paths)
                if derivatives.input_paths
                else "none declared"
            ),
            "Derivative checks: "
            f"finite={_scalar_bool(derivatives.finite)}, "
            f"reference_agreement={_scalar_bool(derivatives.fd_correct)}",
            "Structural dependencies: "
            f"{int(np.sum(dependency_structural))}/"
            f"{dependency_structural.size} input-output pairs",
            "Locally active dependencies: "
            f"{int(np.sum(dependency_traced))}/"
            f"{dependency_traced.size} input-output pairs",
            "Locally active sensitivities: "
            f"{int(np.sum(sensitivity_active))}/"
            f"{sensitivity_active.size} projections "
            f"(threshold={_scalar_text(certificate.sensitivities.threshold)})",
            "Information diagnostics: "
            f"effective_rank={_scalar_text(information.effective_rank)}, "
            "condition_estimate="
            f"{_scalar_text(information.condition_estimate)}",
        )
    )
    summary: str = "\n".join(lines)
    return summary


@jaxtyped(typechecker=beartype)
def explain_claim(
    certificate: ForwardCertificate,
    claim_id: str,
) -> str:
    r"""Explain one claim and the numerical evidence supporting it.

    The inspection result answers a bounded scientific question without loading
    the associated result arrays. It excludes private source locators.

    :see: :class:`~.test_inspect.TestExplainClaim`


    Implementation Logic
    --------------------
    1. **Bind the documented output**::

           rationale: str = "\n".join(lines)

       The function validates and transforms the inputs before it binds the
       documented output.

    Parameters
    ----------
    certificate : ForwardCertificate
        Certificate containing the claim.
    claim_id : str
        Stable claim identifier to explain.

    Returns
    -------
    rationale : str
        Multiline rationale with status, margin, and evidence arrays.

    Raises
    ------
    KeyError
        If the certificate does not contain ``claim_id``.
    """
    evidence_id: Any
    matching: Tuple[Any, ...] = tuple(
        claim for claim in certificate.claims if claim.claim_id == claim_id
    )
    if not matching:
        msg: str = f"Claim '{claim_id}' is not present in this certificate."
        raise KeyError(msg)
    claim: Any = matching[0]
    evidence_by_id: Dict[str, Any] = {
        item.evidence_id: item for item in certificate.evidence
    }
    artifact_ids: Tuple[str, ...] = tuple(
        item.artifact_id for item in certificate.artifacts
    )
    lines: list[str] = [
        f"Claim: {claim.claim_id}",
        f"Subject: {claim.subject_id}",
        f"Predicate: {claim.predicate_id}",
        f"Status: {_claim_status(claim)}",
        f"Margin: {_scalar_text(claim.margin)}",
        f"Severity code: {_scalar_text(claim.severity_code)}",
    ]
    if not claim.evidence_ids:
        lines.append("Evidence: none recorded")
    else:
        lines.append("Evidence:")
    for evidence_id in claim.evidence_ids:
        evidence: Any | None = evidence_by_id.get(evidence_id)
        if evidence is None:
            lines.append(f"  - {evidence_id}: missing from certificate")
            continue
        lines.extend(
            (
                f"  - {evidence.evidence_id}",
                f"    method: {evidence.method_id}",
                f"    source: {evidence.source_type}",
                "    lineage independence: "
                + (
                    "qualified"
                    if evidence_is_independent(
                        evidence,
                        certificate.model.implementation_ref,
                        artifact_ids=artifact_ids,
                    )
                    else "not qualified"
                ),
                "    implementations: "
                + ", ".join(evidence.lineage.implementation_refs),
                "    generators: "
                + ", ".join(evidence.lineage.generator_refs),
                "    artifacts: " + ", ".join(evidence.lineage.artifact_refs),
                "    derivations: "
                + ", ".join(evidence.lineage.derivation_refs),
                "    conflicts: " + ", ".join(evidence.lineage.conflict_refs),
                "    attestations: "
                + ", ".join(evidence.human_attestation_refs),
                f"    measured: {_array_text(evidence.measured)}",
                f"    reference: {_array_text(evidence.reference)}",
                f"    residual: {_array_text(evidence.residual)}",
                f"    tolerance: {_array_text(evidence.tolerance)}",
            )
        )
    rationale: str = "\n".join(lines)
    return rationale


def _different(left: object, right: object) -> bool:
    """PRIVATE: Return deterministic semantic inequality for supported
    records.

    Parameters
    ----------
    left : object
        Reference record.
    right : object
        Candidate record.

    Returns
    -------
    result : bool
        ``True`` when the canonical byte records of the two values
        differ.

    Notes
    -----
    Compares :func:`~.canonical.canonical_pytree` bytes, so the test
    sees type, dtype, shape, and bit-exact value identity rather than
    Python object equality.
    """
    result: bool = canonical_pytree(left) != canonical_pytree(right)
    return result


def _field_differences(
    left: object,
    right: object,
    names: Tuple[str, ...],
) -> Tuple[str, ...]:
    """PRIVATE: Return names whose values differ under canonical
    comparison.

    Parameters
    ----------
    left : object
        Reference record.
    right : object
        Candidate record.
    names : Tuple[str, ...]
        Attribute names to compare.

    Returns
    -------
    result : Tuple[str, ...]
        The names, in input order, whose attribute values differ under
        :func:`_different`.

    Notes
    -----
    The stable name order keeps certificate diffs deterministic for one
    field grouping.
    """
    result: Tuple[str, ...] = tuple(
        name
        for name in names
        if _different(getattr(left, name), getattr(right, name))
    )
    return result


def _artifact_identity(certificate: ForwardCertificate) -> Tuple[object, ...]:
    """PRIVATE: Return meaning-bearing artifact fields without local
    locators.

    Parameters
    ----------
    certificate : ForwardCertificate
        Certificate whose artifacts enter a scientific comparison.

    Returns
    -------
    result : Tuple[object, ...]
        Per artifact: identifier, media type, byte checksum, content
        checksum, semantic checksum, and role.

    Notes
    -----
    Omits the locator on purpose. Two certificates that reference the
    same content at different filesystem paths compare as scientifically
    equal; the locator difference surfaces in the audit category
    instead.
    """
    result: Tuple[object, ...] = tuple(
        (
            item.artifact_id,
            item.media_type,
            item.byte_checksum,
            item.content_checksum,
            item.semantic_checksum,
            item.role,
        )
        for item in certificate.artifacts
    )
    return result


@jaxtyped(typechecker=beartype)
def diff_certificates(
    left: ForwardCertificate,
    right: ForwardCertificate,
) -> CertificateDiff:
    """Compare two certificates by scientific meaning and record class.

    The inspection result answers a bounded scientific question without loading
    the associated result arrays. It excludes private source locators.

    :see: :class:`~.test_inspect.TestDiffCertificates`


    Implementation Logic
    --------------------
    1. **Bind the documented output**::

           difference: CertificateDiff = make_certificate_diff(
                   scientific=tuple(scientific),
                   numerical=_field_differences(left, right, numerical_names),
                   environment=_field_differences(
                       left.manifest,
                       right.manifest,
                       environment_names,
                   ),
                   audit=tuple(audit),
               )

       The function validates and transforms the inputs before it binds the
       documented output.

    Parameters
    ----------
    left : ForwardCertificate
        Reference certificate.
    right : ForwardCertificate
        Candidate certificate.

    Returns
    -------
    difference : CertificateDiff
        Stable field-name differences grouped into scientific, numerical,
        environment, and audit categories.
    """
    scientific_names: Tuple[str, ...] = (
        "model",
        "transformations",
        "domains",
        "policy_id",
        "extensions_json",
    )
    numerical_names: Tuple[str, ...] = (
        "evidence",
        "claims",
        "derivatives",
        "dependencies",
        "sensitivities",
        "information",
        "policy_report",
        "certificate_checksum",
    )
    environment_names: Tuple[str, ...] = (
        "schema_version",
        "package_version",
        "source_checksum",
        "environment_checksum",
        "backend",
        "precision_policy",
        "deterministic",
    )
    audit_names: Tuple[str, ...] = ("execution_id", "started_at_utc")
    scientific: list[str] = list(
        _field_differences(left, right, scientific_names)
    )
    if _different(_artifact_identity(left), _artifact_identity(right)):
        scientific.append("artifacts")
    audit: list[str] = list(
        _field_differences(
            left.manifest,
            right.manifest,
            audit_names,
        )
    )
    left_locators: Tuple[str | None, ...] = tuple(
        item.locator for item in left.artifacts
    )
    right_locators: Tuple[str | None, ...] = tuple(
        item.locator for item in right.artifacts
    )
    if _different(left_locators, right_locators):
        audit.append("artifact_locators")
    difference: CertificateDiff = make_certificate_diff(
        scientific=tuple(scientific),
        numerical=_field_differences(left, right, numerical_names),
        environment=_field_differences(
            left.manifest,
            right.manifest,
            environment_names,
        ),
        audit=tuple(audit),
    )
    return difference


__all__: list[str] = [
    "diff_certificates",
    "explain_claim",
    "summarize_certificate",
]
