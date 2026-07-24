"""Certify differentiable DiffPES forward-model executions.

Extended Summary
----------------
The certification package combines static scientific identity and provenance
with numerical evidence from the actual JAX forward program.
Canonical records and CRC32 checksums provide non-security bookkeeping at the
I/O boundary; they do not establish physical validity.  Runtime claims,
derivative checks, dependency maps, sensitivities, and information spectra are
JAX-native numerical carriers.

The package contains these submodules:

- :mod:`canonical`
    Represent scientific records canonically for certification.
- :mod:`checksums`
    Compute non-security consistency checksums for scientific records.
- :mod:`checks`
    Register pure JAX scientific certification checks.
- :mod:`contracts`
    Define semantic contracts for composable certified transformations.
- :mod:`registry`
    Register certified models and transformations deterministically.
- :mod:`provenance`
    Trace artifact lineage and semantic information loss.
- :mod:`dependencies`
    Trace differentiable information flow through forward models.
- :mod:`evidence`
    Build differentiable evidence for certified forward models.
- :mod:`resolvers`
    Resolve certificate artifacts and verify external evidence.
- :mod:`reproduction`
    Reproduce a certified forward result from resolved artifacts.
- :mod:`policy`
    Evaluate cumulative scientific-certification policies.
- :mod:`execution`
    Execute JAX-native certified forward models.
- :mod:`inspect`
    Render forward-model certificates in a human-readable form.
- :mod:`models`
    Register built-in certified DiffPES forward models.
- :mod:`waivers`
    Validate bounded policy-waiver records at the I/O boundary.

Routine Listings
----------------
:func:`achieved_levels`
    Return certification level names achieved by a concrete report.
:func:`artifact_ref`
    Build separate byte, normalized-content, and semantic identities.
:func:`build_provenance`
    Build a deterministic provenance DAG and propagate information.
:func:`canonical_json`
    Return deterministic typed JSON bytes for ``value``.
:func:`canonical_pytree`
    Return canonical bytes for a supported carrier or PyTree.
:func:`certify_forward`
    Execute a prepared model and produce a certified JAX PyTree.
:func:`certify_forward_checked`
    Execute certification and return structured hard-domain errors.
:func:`checksum_bytes`
    Return a non-security consistency checksum for ``data``.
:func:`checksum_chunks`
    Compute a consistency checksum over consecutive byte chunks.
:func:`checksum_file`
    Stream exact file bytes into a consistency checksum.
:func:`checksum_pytree`
    Stream a canonical carrier into a consistency checksum.
:func:`compose_transformations`
    Compose contracts and raise for unsatisfied requirements.
:func:`clear_dependency_cache`
    Clear the eager cache for structural dependency analyses.
:func:`dependency_cache_info`
    Return cache size, hit count, and miss count.
:func:`dependency_map`
    Trace leaf-level structural and local numerical dependencies.
:func:`derivative_evidence`
    Compare JAX information flow with batched finite differences.
:func:`diff_certificates`
    Compare two certificates by scientific meaning and record class.
:func:`effective_information`
    Return propagated semantics, losses, and invalidations for one node.
:func:`evaluate_claim`
    Evaluate a numerical claim and preserve residual and margin leaves.
:func:`evaluate_domain`
    Evaluate a symmetric domain predicate around a reference value.
:func:`evaluate_evidence`
    Compare measured values with an external numerical reference.
:func:`evaluate_policy`
    Derive cumulative certification outcomes from numerical claims.
:func:`execute_tb_radial`
    Execute the radial ARPES model from one certification input PyTree.
:func:`explain_claim`
    Explain one claim and the numerical evidence supporting it.
:func:`freeze_registry`
    Prevent later registration and return the final immutable snapshot.
:func:`filesystem_artifact_resolver`
    Resolve a byte-valued artifact from its local locator.
:func:`get_check`
    Resolve a registered JAX certification check.
:func:`get_model`
    Resolve an exact registered model.
:func:`get_transformation`
    Resolve an exact registered transformation contract.
:func:`information_spectrum`
    Estimate the leading local information spectrum matrix-free.
:func:`invalidated_claims`
    Return every claim invalidated at or upstream of one output.
:func:`iter_canonical_pytree_chunks`
    Yield canonical carrier bytes in bounded chunks.
:func:`lineage`
    Return the transitive parent-node lineage of one output.
:func:`linearized_forward`
    Evaluate a forward model and retain its JVP linearization.
:func:`list_checks`
    List registered checks in deterministic identity order.
:func:`list_handshakes`
    Return owner handshakes in deterministic identity order.
:func:`list_models`
    Return model specifications in deterministic identity order.
:func:`list_registered_models`
    Return an immutable deterministic snapshot including executors.
:func:`list_transformations`
    Return transformation contracts in deterministic identity order.
:func:`parse_checksum`
    Parse and validate one checksum string.
:func:`packaged_model_card`
    Read the packaged generated card for one model identity.
:func:`prepare_certification`
    Resolve static scientific records before compiled execution.
:func:`mapping_artifact_resolver`
    Build a deterministic resolver from normalized in-memory values.
:func:`register_handshake`
    Register declarative requirements from one owning plan.
:func:`register_builtin_models`
    Register built-in models, transformations, and owner handshakes.
:func:`register_check`
    Register a stable predicate identity and pure JAX callable.
:func:`register_model`
    Register an exact model identity once.
:func:`register_transformation`
    Register an exact transformation contract once.
:func:`registry_snapshot`
    Return one internally consistent immutable registry snapshot.
:func:`registry_manifest`
    Read the packaged registry manifest.
:func:`render_model_card`
    Render a model card directly from a model specification.
:func:`reproduce_forward`
    Re-execute a registered model and compare its recorded result.
:func:`require_active_waivers`
    Reject malformed, expired, or premature waiver records.
:func:`resolve_artifact`
    Resolve and validate one referenced artifact.
:func:`result_checksum`
    Identify a result under a declared numerical configuration.
:func:`semantic_checksum`
    Identify content together with its declared scientific meaning.
:func:`sensitivity_map`
    Measure scaled JVP sensitivities for a batch of tangent directions.
:func:`summarize_certificate`
    Return a deterministic human-readable certificate summary.
:func:`tb_radial_model_spec`
    Return the stable scientific specification for radial ARPES.
:func:`validate_composition`
    Validate and conservatively compose transformation semantics.
:func:`validate_contract`
    Return structural errors for a raw or deserialized contract.
:func:`validate_handshake`
    Validate one owner handshake against available records.
:func:`validate_provenance`
    Re-evaluate graph structure and derived state independently.
:func:`validate_registry`
    Recompute registry structure and consistency checksums.
:func:`validate_registry_manifest`
    Compare the packaged registry manifest with live entries.
:func:`validate_waiver`
    Validate one waiver against an explicit absolute UTC time.
:func:`validate_waivers`
    Validate multiple waivers against one explicit absolute UTC time.
:func:`verify_evidence`
    Verify referenced artifacts and recorded numerical residuals.
:func:`verify_certificate`
    Re-evaluate numerical claim and policy consistency without a rerun.
"""

from .canonical import (
    canonical_json,
    canonical_pytree,
    iter_canonical_pytree_chunks,
)
from .checks import get_check, list_checks, register_check
from .checksums import (
    artifact_ref,
    checksum_bytes,
    checksum_chunks,
    checksum_file,
    checksum_pytree,
    parse_checksum,
    result_checksum,
    semantic_checksum,
)
from .contracts import (
    compose_transformations,
    validate_composition,
    validate_contract,
)
from .dependencies import (
    clear_dependency_cache,
    dependency_cache_info,
    dependency_map,
    information_spectrum,
    linearized_forward,
    sensitivity_map,
)
from .evidence import (
    derivative_evidence,
    evaluate_claim,
    evaluate_domain,
    evaluate_evidence,
)
from .execution import (
    certify_forward,
    certify_forward_checked,
    prepare_certification,
    verify_certificate,
)
from .inspect import diff_certificates, explain_claim, summarize_certificate
from .models import (
    execute_tb_radial,
    register_builtin_models,
    tb_radial_model_spec,
)
from .policy import achieved_levels, evaluate_policy
from .provenance import (
    build_provenance,
    effective_information,
    invalidated_claims,
    lineage,
    validate_provenance,
)
from .registry import (
    freeze_registry,
    get_model,
    get_transformation,
    list_handshakes,
    list_models,
    list_registered_models,
    list_transformations,
    packaged_model_card,
    register_handshake,
    register_model,
    register_transformation,
    registry_manifest,
    registry_snapshot,
    render_model_card,
    validate_handshake,
    validate_registry,
    validate_registry_manifest,
)
from .reproduction import reproduce_forward
from .resolvers import (
    filesystem_artifact_resolver,
    mapping_artifact_resolver,
    resolve_artifact,
    verify_evidence,
)
from .waivers import require_active_waivers, validate_waiver, validate_waivers

__all__: list[str] = [
    "achieved_levels",
    "artifact_ref",
    "build_provenance",
    "canonical_json",
    "canonical_pytree",
    "certify_forward",
    "certify_forward_checked",
    "checksum_bytes",
    "checksum_chunks",
    "checksum_file",
    "checksum_pytree",
    "compose_transformations",
    "clear_dependency_cache",
    "dependency_cache_info",
    "dependency_map",
    "derivative_evidence",
    "diff_certificates",
    "effective_information",
    "evaluate_claim",
    "evaluate_domain",
    "evaluate_evidence",
    "evaluate_policy",
    "execute_tb_radial",
    "explain_claim",
    "filesystem_artifact_resolver",
    "freeze_registry",
    "get_check",
    "get_model",
    "get_transformation",
    "information_spectrum",
    "invalidated_claims",
    "iter_canonical_pytree_chunks",
    "lineage",
    "linearized_forward",
    "list_handshakes",
    "list_models",
    "list_checks",
    "list_registered_models",
    "list_transformations",
    "parse_checksum",
    "packaged_model_card",
    "prepare_certification",
    "mapping_artifact_resolver",
    "register_handshake",
    "register_model",
    "register_builtin_models",
    "register_check",
    "register_transformation",
    "registry_manifest",
    "registry_snapshot",
    "render_model_card",
    "reproduce_forward",
    "require_active_waivers",
    "resolve_artifact",
    "result_checksum",
    "semantic_checksum",
    "sensitivity_map",
    "summarize_certificate",
    "tb_radial_model_spec",
    "validate_composition",
    "validate_contract",
    "validate_handshake",
    "validate_provenance",
    "validate_registry",
    "validate_registry_manifest",
    "validate_waiver",
    "validate_waivers",
    "verify_evidence",
    "verify_certificate",
]
