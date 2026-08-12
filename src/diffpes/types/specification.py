"""Define certified forward-model specifications.

Extended Summary
----------------
This module stores artifact, convention, domain, and forward-model
identity for certified executions.

Routine Listings
----------------
:class:`ArtifactRef`
    Store static identity and role for one source or derived artifact.
:class:`ConventionRef`
    Store a versioned semantic convention used by a scientific model.
:class:`DomainPredicate`
    Store a static declaration of one model-domain predicate.
:class:`DomainResult`
    Store the traced evaluation of one declared domain predicate.
:class:`ForwardModelSpec`
    Store the identity of a differentiable forward model.
:func:`make_artifact_ref`
    Create a validated artifact reference.
:func:`make_convention_ref`
    Create a validated convention reference.
:func:`make_domain_predicate`
    Create a validated domain-predicate declaration.
:func:`make_domain_result`
    Create one traced domain evaluation.
:func:`make_forward_model_spec`
    Create a validated stable forward-model specification.
:obj:`ArtifactResolver`
    Resolve an artifact to normalized content and optional source bytes.
:obj:`CheckFunction`
    Callable signature for a pure JAX certification check.
"""

import equinox as eqx
from beartype import beartype
from beartype.typing import Any, Callable, Optional, Tuple
from jaxtyping import Array, Bool, Float64, Int32, PyTree, jaxtyped

from .certification_validation import (
    _bool,
    _float,
    _int,
    _json_object,
    _nonnegative,
    _require_optional_text,
    _require_text,
    _text_tuple,
)


class ArtifactRef(eqx.Module):
    """Store static identity and role for one source or derived artifact.

    Carry scientific vocabulary separately from traced leaves.
    The record remains stable under JIT, VMAP, JVP, and VJP transforms.

    :see: :class:`~.test_specification.TestArtifactref`

    Attributes
    ----------
    artifact_id : str
        Artifact id (**static** -- a compile-time constant; changing it
        triggers retracing).
    media_type : str
        Media type (**static** -- a compile-time constant; changing it
        triggers retracing).
    byte_checksum : Optional[str]
        Byte checksum (**static** -- a compile-time constant; changing
        it triggers retracing).
    content_checksum : str
        Content checksum (**static** -- a compile-time constant;
        changing it triggers retracing).
    semantic_checksum : str
        Semantic checksum (**static** -- a compile-time constant;
        changing it triggers retracing).
    locator : Optional[str]
        Locator (**static** -- a compile-time constant; changing it
        triggers retracing).
    role : str
        Role (**static** -- a compile-time constant; changing it
        triggers retracing).

    See Also
    --------
    make_artifact_ref : Validated factory for this type.
    """

    artifact_id: str = eqx.field(static=True)
    media_type: str = eqx.field(static=True)
    byte_checksum: Optional[str] = eqx.field(static=True)
    content_checksum: str = eqx.field(static=True)
    semantic_checksum: str = eqx.field(static=True)
    locator: Optional[str] = eqx.field(static=True)
    role: str = eqx.field(static=True)


type ArtifactResolver = Callable[[ArtifactRef], Tuple[Any, Optional[bytes]]]


class ConventionRef(eqx.Module):
    """Store a versioned semantic convention used by a scientific model.

    Carry scientific vocabulary separately from traced leaves.
    The record remains stable under JIT, VMAP, JVP, and VJP transforms.

    :see: :class:`~.test_specification.TestConventionref`

    Attributes
    ----------
    convention_id : str
        Convention id (**static** -- a compile-time constant; changing
        it triggers retracing).
    version : str
        Version (**static** -- a compile-time constant; changing it
        triggers retracing).
    parameters_json : str
        Parameters json (**static** -- a compile-time constant;
        changing it triggers retracing).

    See Also
    --------
    make_convention_ref : Validated factory for this type.
    """

    convention_id: str = eqx.field(static=True)
    version: str = eqx.field(static=True)
    parameters_json: str = eqx.field(static=True)


class DomainPredicate(eqx.Module):
    """Store a static declaration of one model-domain predicate.

    Carry scientific vocabulary separately from traced leaves.
    The record remains stable under JIT, VMAP, JVP, and VJP transforms.

    :see: :class:`~.test_specification.TestDomainpredicate`

    Attributes
    ----------
    predicate_id : str
        Predicate id (**static** -- a compile-time constant; changing
        it triggers retracing).
    expression_id : str
        Expression id (**static** -- a compile-time constant; changing
        it triggers retracing).
    units : Optional[str]
        Units (**static** -- a compile-time constant; changing it
        triggers retracing).
    severity : str
        Severity (**static** -- a compile-time constant; changing it
        triggers retracing).

    See Also
    --------
    make_domain_predicate : Validated factory for this type.
    """

    predicate_id: str = eqx.field(static=True)
    expression_id: str = eqx.field(static=True)
    units: Optional[str] = eqx.field(static=True)
    severity: str = eqx.field(static=True)


class DomainResult(eqx.Module):
    """Store the traced evaluation of one declared domain predicate.

    Carry scientific vocabulary separately from traced leaves.
    The record remains stable under JIT, VMAP, JVP, and VJP transforms.

    :see: :class:`~.test_specification.TestDomainresult`

    Attributes
    ----------
    predicate_id : str
        Predicate id (**static** -- a compile-time constant; changing
        it triggers retracing).
    measured : Float64[Array, ""]
        Measured retained as a differentiable JAX leaf in the declared
        physical units.
    reference : Float64[Array, ""]
        Reference retained as a differentiable JAX leaf in the declared
        physical units.
    residual : Float64[Array, ""]
        Residual retained as a differentiable JAX leaf in the declared
        physical units.
    tolerance : Float64[Array, ""]
        Tolerance retained as a differentiable JAX leaf in the declared
        physical units.
    margin : Float64[Array, ""]
        Margin retained as a differentiable JAX leaf in the declared
        physical units.
    passed : Bool[Array, ""]
        Passed retained as a differentiable JAX leaf in the declared
        physical units.
    checked : Bool[Array, ""]
        Checked retained as a differentiable JAX leaf in the declared
        physical units.
    in_domain : Bool[Array, ""]
        In domain retained as a differentiable JAX leaf in the declared
        physical units.
    severity_code : Int32[Array, ""]
        Severity code retained as a differentiable JAX leaf in the
        declared physical units.

    See Also
    --------
    make_domain_result : Validated factory for this type.
    """

    predicate_id: str = eqx.field(static=True)
    measured: Float64[Array, ""]
    reference: Float64[Array, ""]
    residual: Float64[Array, ""]
    tolerance: Float64[Array, ""]
    margin: Float64[Array, ""]
    passed: Bool[Array, ""]
    checked: Bool[Array, ""]
    in_domain: Bool[Array, ""]
    severity_code: Int32[Array, ""]


type CheckFunction = Callable[[PyTree], DomainResult]


class ForwardModelSpec(eqx.Module):
    """Store the identity of a differentiable forward model.

    Carry scientific vocabulary separately from traced leaves.
    The record remains stable under JIT, VMAP, JVP, and VJP transforms.

    :see: :class:`~.test_specification.TestForwardmodelspec`

    Attributes
    ----------
    model_id : str
        Model id (**static** -- a compile-time constant; changing it
        triggers retracing).
    model_version : str
        Model version (**static** -- a compile-time constant; changing
        it triggers retracing).
    observable_id : str
        Observable id (**static** -- a compile-time constant; changing
        it triggers retracing).
    implementation_ref : str
        Implementation ref (**static** -- a compile-time constant;
        changing it triggers retracing).
    assumptions : Tuple[str, ...]
        Assumptions (**static** -- a compile-time constant; changing it
        triggers retracing).
    conventions : Tuple[ConventionRef, ...]
        Conventions (**static** -- a compile-time constant; changing it
        triggers retracing).
    domain : Tuple[DomainPredicate, ...]
        Domain (**static** -- a compile-time constant; changing it
        triggers retracing).
    differentiable_paths : Tuple[str, ...]
        Differentiable paths (**static** -- a compile-time constant;
        changing it triggers retracing).
    nondifferentiable_paths : Tuple[str, ...]
        Nondifferentiable paths (**static** -- a compile-time constant;
        changing it triggers retracing).

    See Also
    --------
    make_forward_model_spec : Validated factory for this type.
    """

    model_id: str = eqx.field(static=True)
    model_version: str = eqx.field(static=True)
    observable_id: str = eqx.field(static=True)
    implementation_ref: str = eqx.field(static=True)
    assumptions: Tuple[str, ...] = eqx.field(static=True)
    conventions: Tuple[ConventionRef, ...] = eqx.field(static=True)
    domain: Tuple[DomainPredicate, ...] = eqx.field(static=True)
    differentiable_paths: Tuple[str, ...] = eqx.field(static=True)
    nondifferentiable_paths: Tuple[str, ...] = eqx.field(static=True)


@jaxtyped(typechecker=beartype)
def make_artifact_ref(
    artifact_id: str,
    media_type: str,
    byte_checksum: Optional[str],
    content_checksum: str,
    semantic_checksum: str,
    locator: Optional[str],
    role: str,
) -> ArtifactRef:
    """Create a validated artifact reference.

    Carry static scientific vocabulary separately from traced numerical leaves
    while preserving the validation boundary defined by this factory.

    :see: :class:`~.test_specification.TestMakeArtifactRef`

    Parameters
    ----------
    artifact_id : str
        Artifact id used to construct the validated carrier (**static**
        -- a compile-time constant; changing it triggers retracing).
    media_type : str
        Media type used to construct the validated carrier (**static**
        -- a compile-time constant; changing it triggers retracing).
    byte_checksum : Optional[str]
        Byte checksum used to construct the validated carrier
        (**static** -- a compile-time constant; changing it triggers
        retracing).
    content_checksum : str
        Content checksum used to construct the validated carrier
        (**static** -- a compile-time constant; changing it triggers
        retracing).
    semantic_checksum : str
        Semantic checksum used to construct the validated carrier
        (**static** -- a compile-time constant; changing it triggers
        retracing).
    locator : Optional[str]
        Locator used to construct the validated carrier (**static** --
        a compile-time constant; changing it triggers retracing).
    role : str
        Role used to construct the validated carrier (**static** -- a
        compile-time constant; changing it triggers retracing).

    Returns
    -------
    result : ArtifactRef
        Validated immutable carrier.

    Notes
    -----
    The factory checks static structure eagerly. JAX array operations validate
    numerical values and preserve differentiation behavior.
    """
    result: ArtifactRef = ArtifactRef(
        artifact_id=_require_text(artifact_id, "artifact_id"),
        media_type=_require_text(media_type, "media_type"),
        byte_checksum=_require_optional_text(byte_checksum, "byte_checksum"),
        content_checksum=_require_text(content_checksum, "content_checksum"),
        semantic_checksum=_require_text(
            semantic_checksum, "semantic_checksum"
        ),
        locator=_require_optional_text(locator, "locator"),
        role=_require_text(role, "role"),
    )
    return result


@jaxtyped(typechecker=beartype)
def make_convention_ref(
    convention_id: str,
    version: str,
    parameters_json: str = "{}",
) -> ConventionRef:
    """Create a validated convention reference.

    Carry static scientific vocabulary separately from traced numerical leaves
    while preserving the validation boundary defined by this factory.

    :see: :class:`~.test_specification.TestMakeConventionRef`

    Parameters
    ----------
    convention_id : str
        Convention id used to construct the validated carrier
        (**static** -- a compile-time constant; changing it triggers
        retracing).
    version : str
        Version used to construct the validated carrier (**static** --
        a compile-time constant; changing it triggers retracing).
    parameters_json : str
        Parameters json used to construct the validated carrier
        (**static** -- a compile-time constant; changing it triggers
        retracing).

    Returns
    -------
    result : ConventionRef
        Validated immutable carrier.

    Notes
    -----
    The factory checks static structure eagerly. JAX array operations validate
    numerical values and preserve differentiation behavior.
    """
    result: ConventionRef = ConventionRef(
        convention_id=_require_text(convention_id, "convention_id"),
        version=_require_text(version, "version"),
        parameters_json=_json_object(parameters_json, "parameters_json"),
    )
    return result


@jaxtyped(typechecker=beartype)
def make_domain_predicate(
    predicate_id: str,
    expression_id: str,
    units: Optional[str] = None,
    severity: str = "error",
) -> DomainPredicate:
    """Create a validated domain-predicate declaration.

    Carry static scientific vocabulary separately from traced numerical leaves
    while preserving the validation boundary defined by this factory.

    :see: :class:`~.test_specification.TestMakeDomainPredicate`

    Parameters
    ----------
    predicate_id : str
        Predicate id used to construct the validated carrier
        (**static** -- a compile-time constant; changing it triggers
        retracing).
    expression_id : str
        Expression id used to construct the validated carrier
        (**static** -- a compile-time constant; changing it triggers
        retracing).
    units : Optional[str]
        Units used to construct the validated carrier (**static** -- a
        compile-time constant; changing it triggers retracing).
    severity : str
        Severity used to construct the validated carrier (**static** --
        a compile-time constant; changing it triggers retracing).

    Returns
    -------
    result : DomainPredicate
        Validated immutable carrier.

    Notes
    -----
    The factory checks static structure eagerly. JAX array operations validate
    numerical values and preserve differentiation behavior.
    """
    result: DomainPredicate = DomainPredicate(
        predicate_id=_require_text(predicate_id, "predicate_id"),
        expression_id=_require_text(expression_id, "expression_id"),
        units=_require_optional_text(units, "units"),
        severity=_require_text(severity, "severity"),
    )
    return result


@jaxtyped(typechecker=beartype)
def make_domain_result(
    predicate_id: str,
    measured: Any,
    reference: Any,
    residual: Any,
    tolerance: Any,
    margin: Any,
    passed: Any,
    checked: Any = True,
    in_domain: Any = True,
    severity_code: Any = 0,
) -> DomainResult:
    """Create one traced domain evaluation.

    Carry static scientific vocabulary separately from traced numerical leaves
    while preserving the validation boundary defined by this factory.

    :see: :class:`~.test_specification.TestMakeDomainResult`

    Parameters
    ----------
    predicate_id : str
        Predicate id used to construct the validated carrier
        (**static** -- a compile-time constant; changing it triggers
        retracing).
    measured : Any
        Measured used to construct the validated carrier as a traced
        numerical value in the declared physical units.
    reference : Any
        Reference used to construct the validated carrier as a traced
        numerical value in the declared physical units.
    residual : Any
        Residual used to construct the validated carrier as a traced
        numerical value in the declared physical units.
    tolerance : Any
        Tolerance used to construct the validated carrier as a traced
        numerical value in the declared physical units.
    margin : Any
        Margin used to construct the validated carrier as a traced
        numerical value in the declared physical units.
    passed : Any
        Passed used to construct the validated carrier as a traced
        numerical value in the declared physical units.
    checked : Any
        Checked used to construct the validated carrier as a traced
        numerical value in the declared physical units.
    in_domain : Any
        In domain used to construct the validated carrier as a traced
        numerical value in the declared physical units.
    severity_code : Any
        Severity code used to construct the validated carrier as a
        traced numerical value in the declared physical units.

    Returns
    -------
    result : DomainResult
        Validated immutable carrier.

    Notes
    -----
    The factory checks static structure eagerly. JAX array operations validate
    numerical values and preserve differentiation behavior.
    """
    result: DomainResult = DomainResult(
        predicate_id=_require_text(predicate_id, "predicate_id"),
        measured=_float(measured, "measured", 0),
        reference=_float(reference, "reference", 0),
        residual=_float(residual, "residual", 0),
        tolerance=_nonnegative(_float(tolerance, "tolerance", 0), "tolerance"),
        margin=_float(margin, "margin", 0),
        passed=_bool(passed, "passed", 0),
        checked=_bool(checked, "checked", 0),
        in_domain=_bool(in_domain, "in_domain", 0),
        severity_code=_int(severity_code, "severity_code", 0),
    )
    return result


@jaxtyped(typechecker=beartype)
def make_forward_model_spec(
    model_id: str,
    model_version: str,
    observable_id: str,
    implementation_ref: str,
    assumptions: Tuple[str, ...] = (),
    conventions: Tuple[ConventionRef, ...] = (),
    domain: Tuple[DomainPredicate, ...] = (),
    differentiable_paths: Tuple[str, ...] = (),
    nondifferentiable_paths: Tuple[str, ...] = (),
) -> ForwardModelSpec:
    """Create a validated stable forward-model specification.

    Carry static scientific vocabulary separately from traced numerical leaves
    while preserving the validation boundary defined by this factory.

    :see: :class:`~.test_specification.TestMakeForwardModelSpec`

    Parameters
    ----------
    model_id : str
        Model id used to construct the validated carrier (**static** --
        a compile-time constant; changing it triggers retracing).
    model_version : str
        Model version used to construct the validated carrier
        (**static** -- a compile-time constant; changing it triggers
        retracing).
    observable_id : str
        Observable id used to construct the validated carrier
        (**static** -- a compile-time constant; changing it triggers
        retracing).
    implementation_ref : str
        Implementation ref used to construct the validated carrier
        (**static** -- a compile-time constant; changing it triggers
        retracing).
    assumptions : Tuple[str, ...]
        Assumptions used to construct the validated carrier (**static**
        -- a compile-time constant; changing it triggers retracing).
    conventions : Tuple[ConventionRef, ...]
        Conventions used to construct the validated carrier (**static**
        -- a compile-time constant; changing it triggers retracing).
    domain : Tuple[DomainPredicate, ...]
        Domain used to construct the validated carrier (**static** -- a
        compile-time constant; changing it triggers retracing).
    differentiable_paths : Tuple[str, ...]
        Differentiable paths used to construct the validated carrier
        (**static** -- a compile-time constant; changing it triggers
        retracing).
    nondifferentiable_paths : Tuple[str, ...]
        Nondifferentiable paths used to construct the validated carrier
        (**static** -- a compile-time constant; changing it triggers
        retracing).

    Returns
    -------
    result : ForwardModelSpec
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
    diff_paths: Tuple[str, ...] = _text_tuple(
        differentiable_paths,
        "differentiable_paths",
    )
    nondiff_paths: Tuple[str, ...] = _text_tuple(
        nondifferentiable_paths, "nondifferentiable_paths"
    )
    overlap: set[str] = set(diff_paths).intersection(nondiff_paths)
    if overlap:
        raise ValueError(
            "differentiable_paths and nondifferentiable_paths must be disjoint"
        )
    convention_ids: Tuple[str, ...] = tuple(
        item.convention_id for item in conventions
    )
    predicate_ids: Tuple[str, ...] = tuple(
        item.predicate_id for item in domain
    )
    _text_tuple(convention_ids, "convention ids")
    _text_tuple(predicate_ids, "domain predicate ids")
    result: ForwardModelSpec = ForwardModelSpec(
        model_id=_require_text(model_id, "model_id"),
        model_version=_require_text(model_version, "model_version"),
        observable_id=_require_text(observable_id, "observable_id"),
        implementation_ref=_require_text(
            implementation_ref, "implementation_ref"
        ),
        assumptions=_text_tuple(assumptions, "assumptions"),
        conventions=tuple(conventions),
        domain=tuple(domain),
        differentiable_paths=diff_paths,
        nondifferentiable_paths=nondiff_paths,
    )
    return result


__all__: list[str] = [
    "ArtifactRef",
    "ConventionRef",
    "DomainPredicate",
    "DomainResult",
    "ForwardModelSpec",
    "make_artifact_ref",
    "make_convention_ref",
    "make_domain_predicate",
    "make_domain_result",
    "make_forward_model_spec",
    "ArtifactResolver",
    "CheckFunction",
]
