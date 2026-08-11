"""Build differentiable evidence for certified forward models.

Extended Summary
----------------
This module turns continuous JAX measurements into explicit evidence and
claim PyTrees. It retains residuals and margins before it derives Boolean
outcomes. Optimization and experiment-design code can therefore differentiate
the quantities below a certification threshold.

Routine Listings
----------------
:func:`derivative_evidence`
    Compare JAX information flow with batched finite differences.
:func:`evaluate_claim`
    Evaluate a numerical claim and preserve residual and margin leaves.
:func:`evaluate_domain`
    Evaluate a symmetric domain predicate around a reference value.
:func:`evaluate_evidence`
    Compare measured values with an external numerical reference.
"""

import jax
import jax.numpy as jnp
from beartype import beartype
from beartype.typing import Any, Callable, List, Tuple
from jaxtyping import Array, Bool, Float, Float64, PyTree, Shaped, jaxtyped

from diffpes.types import (
    CertificationClaim,
    DerivativeEvidence,
    DomainResult,
    EvidenceLineage,
    EvidenceRef,
    InformationSpectrum,
    make_certification_claim,
    make_derivative_evidence,
    make_domain_result,
    make_evidence_ref,
)

from .dependencies import (
    _information_spectrum_from_linearization,
    _ravel_real_pytree,
)


def _as_vector(value: Shaped[Array, "..."]) -> Float64[Array, " n_value"]:
    """PRIVATE: Flatten a numerical value to a one-dimensional JAX
    array.

    Parameters
    ----------
    value : Shaped[Array, "..."]
        Scalar or array value in its declared physical units.

    Returns
    -------
    vector : Float64[Array, " n_value"]
        One-dimensional ``float64`` view of ``value``.

    Notes
    -----
    Ravels after a cast to ``float64`` so evidence comparisons run in
    one dtype and one flat coordinate layout.
    """
    vector: Float64[Array, " n_value"] = jnp.ravel(
        jnp.asarray(value, dtype=jnp.float64)
    )
    return vector


@jaxtyped(typechecker=beartype)
def evaluate_evidence(
    evidence_id: str,
    method_id: str,
    measured: Shaped[Array, "..."],
    reference: Shaped[Array, "..."],
    tolerance: Shaped[Array, "..."],
    *,
    source_type: str = "external_reference",
    lineage: EvidenceLineage | None = None,
    human_attestation_refs: Tuple[str, ...] = (),
) -> EvidenceRef:
    """Compare measured values with an external numerical reference.

    The operation retains continuous residuals and margins before it derives
    Boolean certification outcomes. These leaves remain available to JAX
    transformations.

    :see: :class:`~.test_evidence.TestEvaluateEvidence`

    Parameters
    ----------
    evidence_id : str
        Stable evidence identifier (**static** -- changing it retraces).
    method_id : str
        Stable comparison-method identifier (**static** -- a change retraces).
    measured : Shaped[Array, "..."]
        Computed numerical values in the declared observable units.
    reference : Shaped[Array, "..."]
        Independent reference values in the same units.
    tolerance : Shaped[Array, "..."]
        Nonnegative component tolerances in the same units.
    source_type : str
        Evidence-source category (**static** -- changing it retraces).
    lineage : EvidenceLineage | None, optional
        Named implementation, generator, artifact, derivation, conflict, and
        relationship ancestry. Omission is explicitly incomplete lineage.
    human_attestation_refs : Tuple[str, ...]
        Separate human-review references. They never establish independence.

    Returns
    -------
    evidence : EvidenceRef
        Vector evidence retaining measured, reference, residual, and tolerance.

    Raises
    ------
    ValueError
        If measured and reference shapes differ.

    Notes
    -----
    Residuals remain JAX leaves, so losses may differentiate through the
    comparison before any Boolean policy reduction.
    """
    measured_array: Float64[Array, " n_measure"] = _as_vector(measured)
    reference_array: Float64[Array, " n_measure"] = _as_vector(reference)
    tolerance_array: Float64[Array, " n_measure"] = jnp.broadcast_to(
        _as_vector(tolerance), measured_array.shape
    )
    if measured_array.shape != reference_array.shape:
        msg: str = "measured and reference evidence arrays must agree"
        raise ValueError(msg)
    residual: Float64[Array, " n_measure"] = measured_array - reference_array
    evidence: EvidenceRef = make_evidence_ref(
        evidence_id=evidence_id,
        method_id=method_id,
        source_type=source_type,
        measured=measured_array,
        reference=reference_array,
        residual=residual,
        tolerance=tolerance_array,
        lineage=lineage,
        human_attestation_refs=human_attestation_refs,
    )
    return evidence


@jaxtyped(typechecker=beartype)
def evaluate_claim(
    claim_id: str,
    subject_id: str,
    predicate_id: str,
    measured: Shaped[Array, "..."],
    reference: Shaped[Array, "..."],
    tolerance: Shaped[Array, "..."],
    *,
    evidence_ids: Tuple[str, ...] = (),
    checked: bool = True,
    in_domain: bool = True,
    severity_code: int = 1,
) -> CertificationClaim:
    """Evaluate a numerical claim and preserve residual and margin leaves.

    The operation retains continuous residuals and margins before it derives
    Boolean certification outcomes. These leaves remain available to JAX
    transformations.

    :see: :class:`~.test_evidence.TestEvaluateClaim`

    Parameters
    ----------
    claim_id : str
        Stable claim identifier (**static** -- changing it retraces).
    subject_id : str
        Scientific subject identifier (**static** -- changing it retraces).
    predicate_id : str
        Predicate identity (**static** -- changing it retraces).
    measured : Shaped[Array, "..."]
        Computed numerical values in the predicate units.
    reference : Shaped[Array, "..."]
        Reference values in the same units.
    tolerance : Shaped[Array, "..."]
        Nonnegative component tolerances in the same units.
    evidence_ids : Tuple[str, ...]
        Supporting evidence identifiers (**static** -- changing them retraces).
    checked : bool
        Whether evaluation occurred (**static** -- changing it retraces).
    in_domain : bool
        Whether the subject lies in the validity domain (**static**).
    severity_code : int
        Numerical severity code (**static** -- changing it retraces).

    Returns
    -------
    claim : CertificationClaim
        Claim with continuous residual and signed minimum margin leaves.

    Raises
    ------
    ValueError
        If measured and reference shapes differ.

    Notes
    -----
    The margin is differentiable almost everywhere with respect to measured
    values. The function derives the Boolean outcome after it retains that
    margin.
    """
    measured_array: Float64[Array, " n_measure"] = _as_vector(measured)
    reference_array: Float64[Array, " n_measure"] = _as_vector(reference)
    tolerance_array: Float64[Array, " n_measure"] = jnp.broadcast_to(
        _as_vector(tolerance), measured_array.shape
    )
    if measured_array.shape != reference_array.shape:
        msg: str = "claim measured and reference arrays must agree"
        raise ValueError(msg)
    residual: Float64[Array, " n_measure"] = measured_array - reference_array
    component_margin: Float64[Array, " n_measure"] = tolerance_array - jnp.abs(
        residual
    )
    margin: Float64[Array, ""] = jnp.min(component_margin)
    checked_array: Bool[Array, ""] = jnp.asarray(checked, dtype=jnp.bool_)
    domain_array: Bool[Array, ""] = jnp.asarray(in_domain, dtype=jnp.bool_)
    passed: Bool[Array, ""] = (
        checked_array & domain_array & jnp.all(component_margin >= 0)
    )
    claim: CertificationClaim = make_certification_claim(
        claim_id=claim_id,
        subject_id=subject_id,
        predicate_id=predicate_id,
        evidence_ids=evidence_ids,
        measured=measured_array,
        reference=reference_array,
        residual=residual,
        tolerance=tolerance_array,
        passed=passed,
        checked=checked_array,
        in_domain=domain_array,
        margin=margin,
        severity_code=jnp.asarray(severity_code, dtype=jnp.int32),
    )
    return claim


@jaxtyped(typechecker=beartype)
def evaluate_domain(
    predicate_id: str,
    measured: Shaped[Array, ""],
    reference: Shaped[Array, ""],
    tolerance: Shaped[Array, ""],
    *,
    checked: bool = True,
    severity_code: int = 1,
) -> DomainResult:
    """Evaluate a symmetric domain predicate around a reference value.

    The operation retains continuous residuals and margins before it derives
    Boolean certification outcomes. These leaves remain available to JAX
    transformations.

    :see: :class:`~.test_evidence.TestEvaluateDomain`

    Parameters
    ----------
    predicate_id : str
        Stable domain-predicate identity (**static** -- changing it retraces).
    measured : Shaped[Array, ""]
        Measured physical quantity in the predicate units.
    reference : Shaped[Array, ""]
        Domain-center value in the same units.
    tolerance : Shaped[Array, ""]
        Symmetric half-width in the same units.
    checked : bool
        Whether evaluation occurred (**static** -- changing it retraces).
    severity_code : int
        Numerical severity code (**static** -- changing it retraces).

    Returns
    -------
    result : DomainResult
        Traced outcome retaining residual and signed boundary margin.

    Notes
    -----
    The signed margin remains differentiable away from the absolute-value
    cusp. Optimization can use it directly without differentiating a Boolean.
    """
    measured_array: Float64[Array, ""] = jnp.asarray(
        measured, dtype=jnp.float64
    )
    reference_array: Float64[Array, ""] = jnp.asarray(
        reference, dtype=jnp.float64
    )
    tolerance_array: Float64[Array, ""] = jnp.asarray(
        tolerance, dtype=jnp.float64
    )
    residual: Float64[Array, ""] = measured_array - reference_array
    margin: Float64[Array, ""] = tolerance_array - jnp.abs(residual)
    checked_array: Bool[Array, ""] = jnp.asarray(checked, dtype=jnp.bool_)
    passed: Bool[Array, ""] = checked_array & (margin >= 0)
    result: DomainResult = make_domain_result(
        predicate_id=predicate_id,
        measured=measured_array,
        reference=reference_array,
        residual=residual,
        tolerance=tolerance_array,
        margin=margin,
        passed=passed,
        checked=checked_array,
        in_domain=passed,
        severity_code=jnp.asarray(severity_code, dtype=jnp.int32),
    )
    return result


def _perturb(
    inputs: PyTree, direction: PyTree, step: Float[Array, ""]
) -> PyTree:
    """PRIVATE: Add one scaled tangent PyTree to an input PyTree.

    Parameters
    ----------
    inputs : PyTree
        Numerical inputs at the evaluation point.
    direction : PyTree
        Tangent with the same structure as ``inputs``.
    step : Float[Array, ""]
        Scalar step size in the units of each perturbed leaf.

    Returns
    -------
    perturbed : PyTree
        Leafwise ``value + step * tangent``.

    Notes
    -----
    Supplies the shifted evaluation points for the central-difference
    reference derivatives.
    """
    perturbed: PyTree = jax.tree.map(
        lambda value, tangent: value + step * tangent,
        inputs,
        direction,
    )
    return perturbed


@jaxtyped(typechecker=beartype)
def _derivative_evidence_from_linearization(  # noqa: PLR0913
    forward_fn: Callable[[PyTree], Float[Array, " n_output"]],
    inputs: PyTree,
    directions: PyTree,
    cotangents: Float[Array, "n_cotangent n_output"],
    pushforward: Callable[[PyTree], Float[Array, " n_output"]],
    pullback: Callable[[Float[Array, " n_output"]], PyTree],
    spectrum: InformationSpectrum,
    *,
    input_paths: Tuple[str, ...],
    output_projection_ids: Tuple[str, ...],
    scales: Float[Array, " n_probe"],
    step: float = 6e-6,
) -> DerivativeEvidence:
    """PRIVATE: Build derivative evidence from one retained JAX
    linearization.

    The function reuses supplied JVP, transpose, and spectrum results. Batched
    central differences remain an independent numerical reference.

    Parameters
    ----------
    forward_fn : Callable[[PyTree], Float[Array, " n_output"]]
        Pure differentiable forward model for central differences.
    inputs : PyTree
        Numerical inputs at the linearization point.
    directions : PyTree
        Tangent probes with one leading probe axis.
    cotangents : Float[Array, "n_cotangent n_output"]
        Output-space transpose probes.
    pushforward : Callable[[PyTree], Float[Array, " n_output"]]
        Retained JVP linear map.
    pullback : Callable[[Float[Array, " n_output"]], PyTree]
        Retained transpose linear map.
    spectrum : InformationSpectrum
        Information spectrum from the same retained linearization.
    input_paths : Tuple[str, ...]
        Stable input-coordinate names (**static**).
    output_projection_ids : Tuple[str, ...]
        Stable output-projection names (**static**).
    scales : Float[Array, " n_probe"]
        Positive physical scale for each tangent probe.
    step : float
        Relative central-difference step. Default 6e-6.

    Returns
    -------
    evidence : DerivativeEvidence
        JVP, transpose, finite-difference, and spectrum evidence.

    Notes
    -----
    The function differentiates continuous evidence. Threshold decisions remain
    discrete diagnostics.
    """
    jvp_values: Float[Array, "n_probe n_output"] = jax.vmap(pushforward)(
        directions
    )

    def finite_difference(
        direction: PyTree, local_step: Float[Array, ""]
    ) -> Float[Array, " n_output"]:
        plus: Float[Array, " n_output"] = forward_fn(
            _perturb(inputs, direction, local_step)
        )
        minus: Float[Array, " n_output"] = forward_fn(
            _perturb(inputs, direction, -local_step)
        )
        derivative: Float[Array, " n_output"] = (plus - minus) / (
            2.0 * local_step
        )
        return derivative

    steps: Float[Array, " n_probe"] = step * jnp.maximum(jnp.abs(scales), 1e-3)
    reference: Float[Array, "n_probe n_output"] = jax.vmap(finite_difference)(
        directions, steps
    )
    residuals: Float[Array, "n_probe n_output"] = jvp_values - reference

    def pull_one(
        cotangent: Float[Array, " n_output"],
    ) -> Float[Array, " n_summary"]:
        leaf: Any
        pulled: PyTree = pullback(cotangent)
        values: List[Float[Array, ""]] = []
        for leaf in jax.tree.leaves(pulled):
            array: Shaped[Array, "..."] = jnp.asarray(leaf)
            values.append(jnp.linalg.norm(jnp.ravel(jnp.real(array))))
            if jnp.iscomplexobj(array):
                values.append(jnp.linalg.norm(jnp.ravel(jnp.imag(array))))
        summaries: Float[Array, " n_summary"] = jnp.asarray(values)
        return summaries

    vjp_values: Float[Array, "n_cotangent n_summary"] = jax.vmap(pull_one)(
        cotangents
    )
    finite: Bool[Array, ""] = (
        jnp.all(jnp.isfinite(jvp_values))
        & jnp.all(jnp.isfinite(vjp_values))
        & jnp.all(jnp.isfinite(reference))
    )
    fd_correct: Bool[Array, ""] = jnp.allclose(
        jvp_values,
        reference,
        rtol=1e-6,
        atol=1e-9,
    )
    evidence: DerivativeEvidence = make_derivative_evidence(
        input_paths=input_paths,
        output_projection_ids=output_projection_ids,
        method="jax.linearize+jvp+linear_transpose+central_fd",
        scales=scales,
        jvp_probes=jnp.reshape(jvp_values, (jvp_values.shape[0], -1)),
        vjp_probes=jnp.reshape(vjp_values, (vjp_values.shape[0], -1)),
        reference_derivatives=jnp.reshape(reference, (reference.shape[0], -1)),
        derivative_residuals=jnp.reshape(residuals, (residuals.shape[0], -1)),
        singular_values=spectrum.singular_values,
        effective_rank=spectrum.effective_rank,
        condition_estimate=spectrum.condition_estimate,
        finite=finite,
        fd_correct=fd_correct,
    )
    return evidence


@jaxtyped(typechecker=beartype)
def derivative_evidence(
    forward_fn: Callable[[PyTree], Float[Array, " n_output"]],
    inputs: PyTree,
    directions: PyTree,
    cotangents: Float[Array, "n_cotangent n_output"],
    *,
    input_paths: Tuple[str, ...],
    output_projection_ids: Tuple[str, ...],
    scales: Float[Array, " n_probe"],
    step: float = 6e-6,
    spectrum_rank: int = 8,
) -> DerivativeEvidence:
    """Compare JAX information flow with batched finite differences.

    The tangent tree has a leading probe axis on every leaf. The function
    reuses one ``jax.linearize`` result for all JVP probes. It applies
    ``jax.vmap`` to the central differences and the retained VJP. The result
    also contains a matrix-free local information spectrum.

    :see: :class:`~.test_evidence.TestDerivativeEvidence`

    Implementation Logic
    --------------------
    1. **Compute the derivative residuals**::

           residuals: Float[Array, "n_probe n_output"] = jvp_values - reference

       The residual retains the difference between JAX linearization and the
       finite-difference reference before construction of the evidence.

    Parameters
    ----------
    forward_fn : Callable[[PyTree], Float[Array, " n_output"]]
        Pure differentiable forward model.
    inputs : PyTree
        Numerical model inputs in their declared physical units.
    directions : PyTree
        Tangent probes with one leading probe axis on each numerical leaf.
    cotangents : Float[Array, "n_cotangent n_output"]
        Output-space probes in the units reciprocal to the forward output.
    input_paths : Tuple[str, ...]
        Stable names for the probed input coordinates (**static**).
    output_projection_ids : Tuple[str, ...]
        Stable names for the output projections (**static**).
    scales : Float[Array, " n_probe"]
        Positive physical scale for each tangent probe.
    step : float
        Relative central-difference step. Default 6e-6.
    spectrum_rank : int
        Requested information-spectrum rank (**static**). Default 8.

    Returns
    -------
    evidence : DerivativeEvidence
        JVP, VJP, finite-difference, and local information-spectrum evidence.

    Notes
    -----
    JVP, VJP, residual, and spectrum leaves remain differentiable. The function
    derives Boolean evidence fields after it retains the continuous values.
    """
    linearized: Tuple[
        Float[Array, " n_output"],
        Callable[[PyTree], Float[Array, " n_output"]],
    ] = jax.linearize(
        forward_fn,
        inputs,
    )
    output: Float[Array, " n_output"] = linearized[0]
    pushforward: Callable[[PyTree], Float[Array, " n_output"]] = linearized[1]
    transposed: Callable[[Float[Array, " n_output"]], Tuple[PyTree]] = (
        jax.linear_transpose(
            pushforward,
            inputs,
        )
    )

    def pullback(cotangent: Float[Array, " n_output"]) -> PyTree:
        pulled: PyTree = transposed(cotangent)[0]
        return pulled

    flat_inputs: Float[Array, " n_input"]
    unravel_inputs: Callable[[Float[Array, " n_input"]], PyTree]
    flat_inputs, unravel_inputs = _ravel_real_pytree(inputs)

    def flat_pushforward(
        tangent: Float[Array, " n_input"],
    ) -> Float[Array, " n_output"]:
        response: Float[Array, " n_output"] = pushforward(
            unravel_inputs(tangent)
        )
        flattened: Float[Array, " n_output"] = _ravel_real_pytree(response)[0]
        return flattened

    flat_transposed: Callable[
        [Float[Array, " n_output"]], Tuple[Float[Array, " n_input"]]
    ] = jax.linear_transpose(flat_pushforward, flat_inputs)

    def flat_pullback(
        cotangent: Float[Array, " n_output"],
    ) -> Float[Array, " n_input"]:
        pulled: Float[Array, " n_input"] = flat_transposed(cotangent)[0]
        return pulled

    flat_output: Float[Array, " n_output"] = _ravel_real_pytree(output)[0]
    spectrum: InformationSpectrum = _information_spectrum_from_linearization(
        inputs,
        flat_output,
        flat_pushforward,
        flat_pullback,
        input_paths=input_paths,
        rank=spectrum_rank,
    )
    evidence: DerivativeEvidence = _derivative_evidence_from_linearization(
        forward_fn,
        inputs,
        directions,
        cotangents,
        pushforward,
        pullback,
        spectrum,
        input_paths=input_paths,
        output_projection_ids=output_projection_ids,
        scales=scales,
        step=step,
    )
    return evidence


__all__: list[str] = [
    "derivative_evidence",
    "evaluate_claim",
    "evaluate_domain",
    "evaluate_evidence",
]
