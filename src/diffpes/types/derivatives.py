"""Define derivative and information evidence records.

Extended Summary
----------------
This module stores derivative checks, dependency maps, scaled
sensitivities, and matrix-free information spectra.

Routine Listings
----------------
:class:`DependencyMap`
    Store declared and JAXPR-observed dependency relations.
:class:`DerivativeEvidence`
    Store JVP, VJP, reference, and information-spectrum evidence.
:class:`InformationSpectrum`
    Store a matrix-free information spectrum in input coordinates.
:class:`SensitivityMap`
    Store scaled sensitivities from inputs to output projections.
:func:`make_dependency_map`
    Create a structural dependency map.
:func:`make_derivative_evidence`
    Create validated derivative and local-information evidence.
:func:`make_information_spectrum`
    Create a validated local information spectrum.
:func:`make_sensitivity_map`
    Create a named, scaled local-sensitivity map.
"""

import equinox as eqx
from beartype import beartype
from beartype.typing import Any, Tuple
from jaxtyping import Array, Bool, Float64, Int32, jaxtyped

from .certification_validation import (
    _bool,
    _float,
    _int,
    _nonnegative,
    _positive,
    _require_text,
    _text_tuple,
)


class DerivativeEvidence(eqx.Module):
    """Store JVP, VJP, reference, and information-spectrum evidence.

    Carry scientific vocabulary separately from traced leaves.
    The record remains stable under JIT, VMAP, JVP, and VJP transforms.

    :see: :class:`~.test_derivatives.TestDerivativeevidence`

    Attributes
    ----------
    input_paths : Tuple[str, ...]
        Input paths (**static** -- a compile-time constant; changing it
        triggers retracing).
    output_projection_ids : Tuple[str, ...]
        Output projection ids (**static** -- a compile-time constant;
        changing it triggers retracing).
    method : str
        Method (**static** -- a compile-time constant; changing it
        triggers retracing).
    scales : Float64[Array, " n_input"]
        Scales retained as a differentiable JAX leaf in the declared
        physical units.
    jvp_probes : Float64[Array, "n_probe n_output"]
        Jvp probes retained as a differentiable JAX leaf in the
        declared physical units.
    vjp_probes : Float64[Array, "n_probe n_input"]
        Vjp probes retained as a differentiable JAX leaf in the
        declared physical units.
    reference_derivatives : Float64[Array, "n_probe n_deriv"]
        Reference derivatives retained as a differentiable JAX leaf in
        the declared physical units.
    derivative_residuals : Float64[Array, "n_probe n_deriv"]
        Derivative residuals retained as a differentiable JAX leaf in
        the declared physical units.
    singular_values : Float64[Array, " n_sv"]
        Singular values retained as a differentiable JAX leaf in the
        declared physical units.
    effective_rank : Int32[Array, ""]
        Effective rank retained as a differentiable JAX leaf in the
        declared physical units.
    condition_estimate : Float64[Array, ""]
        Condition estimate retained as a differentiable JAX leaf in the
        declared physical units. Zero means that there is no active information
        direction.
    finite : Bool[Array, ""]
        Finite retained as a differentiable JAX leaf in the declared
        physical units.
    fd_correct : Bool[Array, ""]
        Fd correct retained as a differentiable JAX leaf in the
        declared physical units.

    See Also
    --------
    make_derivative_evidence : Validated factory for this type.
    """

    input_paths: Tuple[str, ...] = eqx.field(static=True)
    output_projection_ids: Tuple[str, ...] = eqx.field(static=True)
    method: str = eqx.field(static=True)
    scales: Float64[Array, " n_input"]
    jvp_probes: Float64[Array, "n_probe n_output"]
    vjp_probes: Float64[Array, "n_probe n_input"]
    reference_derivatives: Float64[Array, "n_probe n_deriv"]
    derivative_residuals: Float64[Array, "n_probe n_deriv"]
    singular_values: Float64[Array, " n_sv"]
    effective_rank: Int32[Array, ""]
    condition_estimate: Float64[Array, ""]
    finite: Bool[Array, ""]
    fd_correct: Bool[Array, ""]


class DependencyMap(eqx.Module):
    """Store declared and JAXPR-observed dependency relations.

    Carry scientific vocabulary separately from traced leaves.
    The record remains stable under JIT, VMAP, JVP, and VJP transforms.

    :see: :class:`~.test_derivatives.TestDependencymap`

    Attributes
    ----------
    model_id : str
        Model id (**static** -- a compile-time constant; changing it
        triggers retracing).
    input_paths : Tuple[str, ...]
        Input paths (**static** -- a compile-time constant; changing it
        triggers retracing).
    output_paths : Tuple[str, ...]
        Output paths (**static** -- a compile-time constant; changing
        it triggers retracing).
    structural : Bool[Array, "n_output n_input"]
        Structural retained as a differentiable JAX leaf in the
        declared physical units.
    traced : Bool[Array, "n_output n_input"]
        Traced retained as a differentiable JAX leaf in the declared
        physical units.

    See Also
    --------
    make_dependency_map : Validated factory for this type.
    """

    model_id: str = eqx.field(static=True)
    input_paths: Tuple[str, ...] = eqx.field(static=True)
    output_paths: Tuple[str, ...] = eqx.field(static=True)
    structural: Bool[Array, "n_output n_input"]
    traced: Bool[Array, "n_output n_input"]


class SensitivityMap(eqx.Module):
    """Store scaled sensitivities from inputs to output projections.

    Carry scientific vocabulary separately from traced leaves.
    The record remains stable under JIT, VMAP, JVP, and VJP transforms.

    :see: :class:`~.test_derivatives.TestSensitivitymap`

    Attributes
    ----------
    input_paths : Tuple[str, ...]
        Input paths (**static** -- a compile-time constant; changing it
        triggers retracing).
    output_projection_ids : Tuple[str, ...]
        Output projection ids (**static** -- a compile-time constant;
        changing it triggers retracing).
    scales : Float64[Array, " n_input"]
        Scales retained as a differentiable JAX leaf in the declared
        physical units.
    sensitivities : Float64[Array, "n_output n_input"]
        Sensitivities retained as a differentiable JAX leaf in the
        declared physical units.
    threshold : Float64[Array, ""]
        Threshold retained as a differentiable JAX leaf in the declared
        physical units.
    active : Bool[Array, "n_output n_input"]
        Active retained as a differentiable JAX leaf in the declared
        physical units.

    See Also
    --------
    make_sensitivity_map : Validated factory for this type.
    """

    input_paths: Tuple[str, ...] = eqx.field(static=True)
    output_projection_ids: Tuple[str, ...] = eqx.field(static=True)
    scales: Float64[Array, " n_input"]
    sensitivities: Float64[Array, "n_output n_input"]
    threshold: Float64[Array, ""]
    active: Bool[Array, "n_output n_input"]


class InformationSpectrum(eqx.Module):
    """Store a matrix-free information spectrum in input coordinates.

    Carry scientific vocabulary separately from traced leaves.
    The record remains stable under JIT, VMAP, JVP, and VJP transforms.

    :see: :class:`~.test_derivatives.TestInformationspectrum`

    Attributes
    ----------
    input_paths : Tuple[str, ...]
        Input paths (**static** -- a compile-time constant; changing it
        triggers retracing).
    singular_values : Float64[Array, " n_sv"]
        Singular values retained as a differentiable JAX leaf in the
        declared physical units.
    right_singular_vectors : Float64[Array, "n_sv n_input"]
        Right singular vectors retained as a differentiable JAX leaf in
        the declared physical units.
    effective_rank : Int32[Array, ""]
        Effective rank retained as a differentiable JAX leaf in the
        declared physical units.
    condition_estimate : Float64[Array, ""]
        Condition estimate retained as a differentiable JAX leaf in the
        declared physical units. Zero means that there is no active information
        direction.
    threshold : Float64[Array, ""]
        Threshold retained as a differentiable JAX leaf in the declared
        physical units.

    See Also
    --------
    make_information_spectrum : Validated factory for this type.
    """

    input_paths: Tuple[str, ...] = eqx.field(static=True)
    singular_values: Float64[Array, " n_sv"]
    right_singular_vectors: Float64[Array, "n_sv n_input"]
    effective_rank: Int32[Array, ""]
    condition_estimate: Float64[Array, ""]
    threshold: Float64[Array, ""]


@jaxtyped(typechecker=beartype)
def make_derivative_evidence(  # noqa: PLR0913, PLR0917
    input_paths: Tuple[str, ...],
    output_projection_ids: Tuple[str, ...],
    method: str,
    scales: Any,
    jvp_probes: Any,
    vjp_probes: Any,
    reference_derivatives: Any,
    derivative_residuals: Any,
    singular_values: Any,
    effective_rank: Any,
    condition_estimate: Any,
    finite: Any,
    fd_correct: Any,
) -> DerivativeEvidence:
    """Create validated derivative and local-information evidence.

    Carry static scientific vocabulary separately from traced numerical leaves
    while preserving the validation boundary defined by this factory.

    :see: :class:`~.test_derivatives.TestMakeDerivativeEvidence`

    Parameters
    ----------
    input_paths : Tuple[str, ...]
        Input paths used to construct the validated carrier (**static**
        -- a compile-time constant; changing it triggers retracing).
    output_projection_ids : Tuple[str, ...]
        Output projection ids used to construct the validated carrier
        (**static** -- a compile-time constant; changing it triggers
        retracing).
    method : str
        Method used to construct the validated carrier (**static** -- a
        compile-time constant; changing it triggers retracing).
    scales : Any
        Scales used to construct the validated carrier as a traced
        numerical value in the declared physical units.
    jvp_probes : Any
        Jvp probes used to construct the validated carrier as a traced
        numerical value in the declared physical units.
    vjp_probes : Any
        Vjp probes used to construct the validated carrier as a traced
        numerical value in the declared physical units.
    reference_derivatives : Any
        Reference derivatives used to construct the validated carrier
        as a traced numerical value in the declared physical units.
    derivative_residuals : Any
        Derivative residuals used to construct the validated carrier as
        a traced numerical value in the declared physical units.
    singular_values : Any
        Singular values used to construct the validated carrier as a
        traced numerical value in the declared physical units.
    effective_rank : Any
        Effective rank used to construct the validated carrier as a
        traced numerical value in the declared physical units.
    condition_estimate : Any
        Condition estimate used to construct the validated carrier as a
        traced numerical value in the declared physical units. Zero means that
        there is no active information direction.
    finite : Any
        Finite used to construct the validated carrier as a traced
        numerical value in the declared physical units.
    fd_correct : Any
        Fd correct used to construct the validated carrier as a traced
        numerical value in the declared physical units.

    Returns
    -------
    result : DerivativeEvidence
        Validated immutable carrier.

    Raises
    ------
    ValueError
        If static structure or cross-record validation fails.

    Notes
    -----
    The factory checks static structure eagerly. JAX array operations validate
    numerical values and preserve differentiation behavior.
    """
    paths: Tuple[str, ...] = _text_tuple(input_paths, "input_paths")
    projections: Tuple[str, ...] = _text_tuple(
        output_projection_ids,
        "output_projection_ids",
    )
    scales_array: Float64[Array, " n_input"] = _positive(
        _float(scales, "scales", 1), "scales"
    )
    jvp_array: Float64[Array, "n_probe n_output"] = _float(
        jvp_probes, "jvp_probes", 2
    )
    vjp_array: Float64[Array, "n_probe n_input"] = _float(
        vjp_probes, "vjp_probes", 2
    )
    reference_array: Float64[Array, "n_probe n_reference"] = _float(
        reference_derivatives,
        "reference_derivatives",
        2,
    )
    residual_array: Float64[Array, "n_probe n_reference"] = _float(
        derivative_residuals,
        "derivative_residuals",
        2,
    )
    singular_array: Float64[Array, " n_singular"] = _nonnegative(
        _float(singular_values, "singular_values", 1), "singular_values"
    )
    if scales_array.shape[0] != len(paths):
        raise ValueError("scales length must equal input_paths length")
    if vjp_array.shape[1] != len(paths):
        raise ValueError("vjp_probes input dimension must equal input_paths")
    if jvp_array.shape[0] != vjp_array.shape[0]:
        raise ValueError("JVP and VJP probe counts must agree")
    if reference_array.shape != residual_array.shape:
        raise ValueError("reference and derivative residual shapes must agree")
    if reference_array.shape[0] != jvp_array.shape[0]:
        raise ValueError("reference derivative probe count must agree")
    result: DerivativeEvidence = DerivativeEvidence(
        input_paths=paths,
        output_projection_ids=projections,
        method=_require_text(method, "method"),
        scales=scales_array,
        jvp_probes=jvp_array,
        vjp_probes=vjp_array,
        reference_derivatives=reference_array,
        derivative_residuals=residual_array,
        singular_values=singular_array,
        effective_rank=_int(effective_rank, "effective_rank", 0),
        condition_estimate=_float(condition_estimate, "condition_estimate", 0),
        finite=_bool(finite, "finite", 0),
        fd_correct=_bool(fd_correct, "fd_correct", 0),
    )
    return result


@jaxtyped(typechecker=beartype)
def make_dependency_map(
    model_id: str,
    input_paths: Tuple[str, ...],
    output_paths: Tuple[str, ...],
    structural: Any,
    traced: Any,
) -> DependencyMap:
    """Create a structural dependency map.

    Carry static scientific vocabulary separately from traced numerical leaves
    while preserving the validation boundary defined by this factory.

    :see: :class:`~.test_derivatives.TestMakeDependencyMap`

    Parameters
    ----------
    model_id : str
        Model id used to construct the validated carrier (**static** --
        a compile-time constant; changing it triggers retracing).
    input_paths : Tuple[str, ...]
        Input paths used to construct the validated carrier (**static**
        -- a compile-time constant; changing it triggers retracing).
    output_paths : Tuple[str, ...]
        Output paths used to construct the validated carrier
        (**static** -- a compile-time constant; changing it triggers
        retracing).
    structural : Any
        Structural used to construct the validated carrier as a traced
        numerical value in the declared physical units.
    traced : Any
        Traced used to construct the validated carrier as a traced
        numerical value in the declared physical units.

    Returns
    -------
    result : DependencyMap
        Validated immutable carrier.

    Raises
    ------
    ValueError
        If static structure or cross-record validation fails.

    Notes
    -----
    The factory checks static structure eagerly. JAX array operations validate
    numerical values and preserve differentiation behavior.
    """
    inputs: Tuple[str, ...] = _text_tuple(input_paths, "input_paths")
    outputs: Tuple[str, ...] = _text_tuple(output_paths, "output_paths")
    structural_array: Bool[Array, "n_output n_input"] = _bool(
        structural, "structural", 2
    )
    traced_array: Bool[Array, "n_output n_input"] = _bool(traced, "traced", 2)
    expected: Tuple[int, int] = (len(outputs), len(inputs))
    if structural_array.shape != expected or traced_array.shape != expected:
        raise ValueError(
            "dependency matrices must have shape (outputs, inputs)"
        )
    result: DependencyMap = DependencyMap(
        model_id=_require_text(model_id, "model_id"),
        input_paths=inputs,
        output_paths=outputs,
        structural=structural_array,
        traced=traced_array,
    )
    return result


@jaxtyped(typechecker=beartype)
def make_sensitivity_map(
    input_paths: Tuple[str, ...],
    output_projection_ids: Tuple[str, ...],
    scales: Any,
    sensitivities: Any,
    threshold: Any,
    active: Any,
) -> SensitivityMap:
    """Create a named, scaled local-sensitivity map.

    Carry static scientific vocabulary separately from traced numerical leaves
    while preserving the validation boundary defined by this factory.

    :see: :class:`~.test_derivatives.TestMakeSensitivityMap`

    Parameters
    ----------
    input_paths : Tuple[str, ...]
        Input paths used to construct the validated carrier (**static**
        -- a compile-time constant; changing it triggers retracing).
    output_projection_ids : Tuple[str, ...]
        Output projection ids used to construct the validated carrier
        (**static** -- a compile-time constant; changing it triggers
        retracing).
    scales : Any
        Scales used to construct the validated carrier as a traced
        numerical value in the declared physical units.
    sensitivities : Any
        Sensitivities used to construct the validated carrier as a
        traced numerical value in the declared physical units.
    threshold : Any
        Threshold used to construct the validated carrier as a traced
        numerical value in the declared physical units.
    active : Any
        Active used to construct the validated carrier as a traced
        numerical value in the declared physical units.

    Returns
    -------
    result : SensitivityMap
        Validated immutable carrier.

    Raises
    ------
    ValueError
        If static structure or cross-record validation fails.

    Notes
    -----
    The factory checks static structure eagerly. JAX array operations validate
    numerical values and preserve differentiation behavior.
    """
    inputs: Tuple[str, ...] = _text_tuple(input_paths, "input_paths")
    outputs: Tuple[str, ...] = _text_tuple(
        output_projection_ids,
        "output_projection_ids",
    )
    scales_array: Float64[Array, " n_input"] = _positive(
        _float(scales, "scales", 1), "scales"
    )
    sensitivities_array: Float64[Array, "n_output n_input"] = _float(
        sensitivities,
        "sensitivities",
        2,
    )
    active_array: Bool[Array, "n_output n_input"] = _bool(active, "active", 2)
    expected: Tuple[int, int] = (len(outputs), len(inputs))
    if sensitivities_array.shape != expected or active_array.shape != expected:
        raise ValueError(
            "sensitivity matrices must have shape (outputs, inputs)"
        )
    if scales_array.shape != (len(inputs),):
        raise ValueError("scales length must equal input_paths length")
    result: SensitivityMap = SensitivityMap(
        input_paths=inputs,
        output_projection_ids=outputs,
        scales=scales_array,
        sensitivities=sensitivities_array,
        threshold=_nonnegative(_float(threshold, "threshold", 0), "threshold"),
        active=active_array,
    )
    return result


@jaxtyped(typechecker=beartype)
def make_information_spectrum(
    input_paths: Tuple[str, ...],
    singular_values: Any,
    right_singular_vectors: Any,
    effective_rank: Any,
    condition_estimate: Any,
    threshold: Any,
) -> InformationSpectrum:
    """Create a validated local information spectrum.

    Carry static scientific vocabulary separately from traced numerical leaves
    while preserving the validation boundary defined by this factory.

    :see: :class:`~.test_derivatives.TestMakeInformationSpectrum`

    Parameters
    ----------
    input_paths : Tuple[str, ...]
        Input paths used to construct the validated carrier (**static**
        -- a compile-time constant; changing it triggers retracing).
    singular_values : Any
        Singular values used to construct the validated carrier as a
        traced numerical value in the declared physical units.
    right_singular_vectors : Any
        Right singular vectors used to construct the validated carrier
        as a traced numerical value in the declared physical units.
    effective_rank : Any
        Effective rank used to construct the validated carrier as a
        traced numerical value in the declared physical units.
    condition_estimate : Any
        Condition estimate used to construct the validated carrier as a
        traced numerical value in the declared physical units. Zero means that
        there is no active information direction.
    threshold : Any
        Threshold used to construct the validated carrier as a traced
        numerical value in the declared physical units.

    Returns
    -------
    result : InformationSpectrum
        Validated immutable carrier.

    Raises
    ------
    ValueError
        If static structure or cross-record validation fails.

    Notes
    -----
    The factory checks static structure eagerly. JAX array operations validate
    numerical values and preserve differentiation behavior.
    """
    paths: Tuple[str, ...] = _text_tuple(input_paths, "input_paths")
    singular_array: Float64[Array, " n_singular"] = _nonnegative(
        _float(singular_values, "singular_values", 1), "singular_values"
    )
    vectors_array: Float64[Array, "n_singular n_input"] = _float(
        right_singular_vectors,
        "right_singular_vectors",
        2,
    )
    if vectors_array.shape != (singular_array.shape[0], len(paths)):
        raise ValueError(
            "right_singular_vectors must have shape (singular values, inputs)"
        )
    result: InformationSpectrum = InformationSpectrum(
        input_paths=paths,
        singular_values=singular_array,
        right_singular_vectors=vectors_array,
        effective_rank=_int(effective_rank, "effective_rank", 0),
        condition_estimate=_float(condition_estimate, "condition_estimate", 0),
        threshold=_nonnegative(_float(threshold, "threshold", 0), "threshold"),
    )
    return result


__all__: list[str] = [
    "DependencyMap",
    "DerivativeEvidence",
    "InformationSpectrum",
    "SensitivityMap",
    "make_dependency_map",
    "make_derivative_evidence",
    "make_information_spectrum",
    "make_sensitivity_map",
]
