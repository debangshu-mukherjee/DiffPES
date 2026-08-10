"""Define static carriers for certified transformation contracts.

Extended Summary
----------------
Transformation contracts are declarative Equinox PyTrees. They record how a
transformation requires, produces, preserves, introduces, or destroys
scientific semantics. They keep runtime bookkeeping outside JAX kernels.

Routine Listings
----------------
:class:`CompositionReport`
    Store a conservative transformation-composition result.
:class:`TransformationContract`
    Store the static semantic contract for one registered transformation.
:func:`make_composition_report`
    Create a validated immutable transformation-composition report.
:func:`make_transformation_contract`
    Create a validated immutable transformation contract.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import equinox as eqx
from beartype import beartype
from beartype.typing import Tuple
from jaxtyping import jaxtyped

from .certification_constants import (
    CERTIFICATION_IDENTIFIER_PATTERN,
    CERTIFICATION_SEMVER_PATTERN,
)


class TransformationContract(eqx.Module):
    """Store the static semantic contract for one registered transformation.

    This carrier declares how one transformation changes the named scientific
    semantics available to downstream forward-model claims.

    :see: :class:`~.test_contracts.TestTransformationContract`

    Attributes
    ----------
    transformation_id : str
        Permanent transformation identifier (**static** -- a compile-time
        constant; changing it triggers retracing).
    transformation_version : str
        Semantic version (**static** -- a compile-time constant; changing it
        triggers retracing).
    requires : Tuple[str, ...]
        Semantics required on input (**static** -- compile-time constants;
        changing them triggers retracing).
    produces : Tuple[str, ...]
        Output semantics created by the transformation (**static** --
        compile-time constants; changing them triggers retracing).
    preserves : Tuple[str, ...]
        Input semantics preserved exactly (**static** -- compile-time
        constants; changing them triggers retracing).
    introduces : Tuple[str, ...]
        New semantics introduced on output (**static** -- compile-time
        constants; changing them triggers retracing).
    destroys : Tuple[str, ...]
        Information made unavailable on output (**static** -- compile-time
        constants; changing them triggers retracing).
    invalidates_claims : Tuple[str, ...]
        Claim identifiers invalidated by the transformation (**static** --
        compile-time constants; changing them triggers retracing).
    jax_pure : bool
        Whether the contract declares the transformation JAX-pure (**static**
        -- a compile-time constant; changing it triggers retracing).

    Notes
    -----
    Every field is static metadata. The carrier has no differentiable leaves;
    it describes a JAX transformation but does not participate in its gradient.

    See Also
    --------
    make_transformation_contract : Create a validated immutable transformation
        contract.
    """

    transformation_id: str = eqx.field(static=True)
    transformation_version: str = eqx.field(static=True)
    requires: Tuple[str, ...] = eqx.field(static=True)
    produces: Tuple[str, ...] = eqx.field(static=True)
    preserves: Tuple[str, ...] = eqx.field(static=True)
    introduces: Tuple[str, ...] = eqx.field(static=True)
    destroys: Tuple[str, ...] = eqx.field(static=True)
    invalidates_claims: Tuple[str, ...] = eqx.field(static=True)
    jax_pure: bool = eqx.field(static=True)


class CompositionReport(eqx.Module):
    """Store a conservative transformation-composition result.

    This carrier records whether a sequence of transformation contracts can
    compose without losing a required scientific semantic.

    :see: :class:`~.test_contracts.TestCompositionReport`

    Attributes
    ----------
    valid : bool
        Whether composition succeeds (**static** -- a compile-time constant;
        changing it triggers retracing).
    errors : Tuple[str, ...]
        Composition failures (**static** -- compile-time constants; changing
        them triggers retracing).
    available_semantics : Tuple[str, ...]
        Semantics available after composition (**static** -- compile-time
        constants; changing them triggers retracing).
    destroyed_information : Tuple[str, ...]
        Information destroyed along the path (**static** -- compile-time
        constants; changing them triggers retracing).
    invalidated_claims : Tuple[str, ...]
        Claims invalidated along the path (**static** -- compile-time
        constants; changing them triggers retracing).
    transformation_refs : Tuple[str, ...]
        Ordered transformation references (**static** -- compile-time
        constants; changing them triggers retracing).

    Notes
    -----
    Every field is static metadata. Code outside numerical kernels evaluates
    composition. Therefore, the report introduces no gradient path or
    reduction.

    See Also
    --------
    make_composition_report : Create a validated immutable
        transformation-composition report.
    """

    valid: bool = eqx.field(static=True)
    errors: Tuple[str, ...] = eqx.field(static=True)
    available_semantics: Tuple[str, ...] = eqx.field(static=True)
    destroyed_information: Tuple[str, ...] = eqx.field(static=True)
    invalidated_claims: Tuple[str, ...] = eqx.field(static=True)
    transformation_refs: Tuple[str, ...] = eqx.field(static=True)


def _normalize_terms(
    values: Iterable[str],
    *,
    field_name: str,
) -> Tuple[str, ...]:
    """PRIVATE: Normalize a semantic term sequence without hiding duplicates.

    Implementation Logic
    --------------------
    Freeze ``values`` into a tuple first. Reject non-string or blank
    entries. Compare the set size against the tuple length so
    duplicates fail without silent removal.

    Parameters
    ----------
    values : Iterable[str]
        Semantic term entries to freeze.
    field_name : str
        Field name used in the static error messages.

    Returns
    -------
    normalized : Tuple[str, ...]
        The input entries frozen into a tuple in their original order.

    Raises
    ------
    ValueError
        If an entry is not a nonblank string, or if the entries are not
        unique. This is the static construction-time contract.
    """
    normalized: Tuple[str, ...] = tuple(values)
    if any(
        not isinstance(value, str) or not value.strip() for value in normalized
    ):
        msg: str = f"{field_name} entries must be nonblank strings"
        raise ValueError(msg)
    if len(set(normalized)) != len(normalized):
        msg: str = f"{field_name} entries must be unique"
        raise ValueError(msg)
    return normalized


def _validate_identity(transformation_id: str, version: str) -> None:
    """PRIVATE: Validate permanent identity and scientific semantic version.

    Implementation Logic
    --------------------
    Apply ``fullmatch`` with the shared certification identifier and
    semantic-version patterns. Require at least one dot so that the
    identifier keeps a namespace segment.

    Parameters
    ----------
    transformation_id : str
        Permanent lowercase reverse-DNS-like transformation identifier.
    version : str
        Semantic version string of the transformation.

    Raises
    ------
    ValueError
        If ``transformation_id`` does not match
        ``CERTIFICATION_IDENTIFIER_PATTERN`` or contains no dot, or if
        ``version`` does not match ``CERTIFICATION_SEMVER_PATTERN``. This
        is the static construction-time contract.
    """
    if (
        CERTIFICATION_IDENTIFIER_PATTERN.fullmatch(transformation_id) is None
        or "." not in transformation_id
    ):
        msg: str = "transformation_id must be a lowercase reverse-DNS-like ID"
        raise ValueError(msg)
    if CERTIFICATION_SEMVER_PATTERN.fullmatch(version) is None:
        msg: str = "transformation_version must be a semantic version"
        raise ValueError(msg)


@jaxtyped(typechecker=beartype)
def make_transformation_contract(
    transformation_id: str,
    transformation_version: str,
    *,
    requires: Sequence[str] = (),
    produces: Sequence[str] = (),
    preserves: Sequence[str] = (),
    introduces: Sequence[str] = (),
    destroys: Sequence[str] = (),
    invalidates_claims: Sequence[str] = (),
    jax_pure: bool = True,
) -> TransformationContract:
    """Create a validated immutable transformation contract.

    Validate a permanent identity and freeze each semantic term sequence
    without hiding duplicates or contradictory information-loss declarations.

    :see: :class:`~.test_contracts.TestMakeTransformationContract`

    Implementation Logic
    --------------------
    1. **Validate identity**::

           _validate_identity(transformation_id, transformation_version)

       Require a permanent reverse-DNS-like identifier and semantic version.
    2. **Normalize semantic terms**::

           normalized_requires = _normalize_terms(requires, ...)

       Freeze each semantic sequence while retaining duplicate detection.
    3. **Reject contradictory loss declarations**::

           contradictory = set(normalized_preserves) & set(normalized_destroys)

       A term cannot remain available while also being declared destroyed.
    4. **Construct the carrier**::

           contract = TransformationContract(...)

       Bind and return the immutable static contract.

    Parameters
    ----------
    transformation_id : str
        Permanent reverse-DNS-like identifier (**static** -- a compile-time
        constant; changing it triggers retracing).
    transformation_version : str
        Semantic version (**static** -- a compile-time constant; changing it
        triggers retracing).
    requires : Sequence[str]
        Required input semantics (**static** -- compile-time constants;
        changing them triggers retracing). Default is empty.
    produces : Sequence[str]
        Produced output semantics (**static** -- compile-time constants;
        changing them triggers retracing). Default is empty.
    preserves : Sequence[str]
        Preserved input semantics (**static** -- compile-time constants;
        changing them triggers retracing). Default is empty.
    introduces : Sequence[str]
        Introduced output semantics (**static** -- compile-time constants;
        changing them triggers retracing). Default is empty.
    destroys : Sequence[str]
        Destroyed information labels (**static** -- compile-time constants;
        changing them triggers retracing). Default is empty.
    invalidates_claims : Sequence[str]
        Invalidated claim identifiers (**static** -- compile-time constants;
        changing them triggers retracing). Default is empty.
    jax_pure : bool
        JAX-purity declaration (**static** -- a compile-time constant; changing
        it triggers retracing). Default is ``True``.

    Returns
    -------
    contract : TransformationContract
        Validated immutable transformation contract.

    Raises
    ------
    ValueError
        If the identity or version is invalid; a semantic term is blank or
        duplicated; or a term is both preserved or introduced and destroyed.

    Notes
    -----
    The factory performs only static validation because all inputs become
    static fields. It does not convert a traced numerical value to Python
    control flow.
    """
    _validate_identity(transformation_id, transformation_version)
    normalized_requires: Tuple[str, ...] = _normalize_terms(
        requires,
        field_name="requires",
    )
    normalized_produces: Tuple[str, ...] = _normalize_terms(
        produces,
        field_name="produces",
    )
    normalized_preserves: Tuple[str, ...] = _normalize_terms(
        preserves,
        field_name="preserves",
    )
    normalized_introduces: Tuple[str, ...] = _normalize_terms(
        introduces,
        field_name="introduces",
    )
    normalized_destroys: Tuple[str, ...] = _normalize_terms(
        destroys,
        field_name="destroys",
    )
    normalized_invalidates: Tuple[str, ...] = _normalize_terms(
        invalidates_claims,
        field_name="invalidates_claims",
    )
    contradictory: set[str] = set(normalized_preserves) & set(
        normalized_destroys
    )
    contradictory |= set(normalized_introduces) & set(normalized_destroys)
    if contradictory:
        terms: str = ", ".join(sorted(contradictory))
        msg: str = f"a contract cannot preserve/introduce and destroy: {terms}"
        raise ValueError(msg)
    contract: TransformationContract = TransformationContract(
        transformation_id=transformation_id,
        transformation_version=transformation_version,
        requires=normalized_requires,
        produces=normalized_produces,
        preserves=normalized_preserves,
        introduces=normalized_introduces,
        destroys=normalized_destroys,
        invalidates_claims=normalized_invalidates,
        jax_pure=bool(jax_pure),
    )
    return contract


@jaxtyped(typechecker=beartype)
def make_composition_report(
    valid: bool,
    errors: Sequence[str],
    available_semantics: Sequence[str],
    destroyed_information: Sequence[str],
    invalidated_claims: Sequence[str],
    transformation_refs: Sequence[str],
) -> CompositionReport:
    """Create a validated immutable transformation-composition report.

    Freeze the cumulative semantic state and require the validity flag to
    agree exactly with whether the report contains errors.

    :see: :class:`~.test_contracts.TestMakeCompositionReport`

    Implementation Logic
    --------------------
    1. **Normalize errors**::

           normalized_errors = _normalize_terms(errors, field_name="errors")

       Freeze validation failures and reject blank or duplicate entries.
    2. **Check validity consistency**::

           if valid == bool(normalized_errors):

       Require success exactly when no composition error is present.
    3. **Construct the report**::

           report = CompositionReport(...)

       Normalize the cumulative semantic fields and bind the result.

    Parameters
    ----------
    valid : bool
        Whether composition succeeded (**static** -- a compile-time constant;
        changing it triggers retracing).
    errors : Sequence[str]
        Composition errors (**static** -- compile-time constants; changing
        them triggers retracing).
    available_semantics : Sequence[str]
        Final available semantics (**static** -- compile-time constants;
        changing them triggers retracing).
    destroyed_information : Sequence[str]
        Cumulative information losses (**static** -- compile-time constants;
        changing them triggers retracing).
    invalidated_claims : Sequence[str]
        Cumulative invalidated claims (**static** -- compile-time constants;
        changing them triggers retracing).
    transformation_refs : Sequence[str]
        Ordered transformation references (**static** -- compile-time
        constants; changing them triggers retracing).

    Returns
    -------
    report : CompositionReport
        Validated immutable composition report.

    Raises
    ------
    ValueError
        If a sequence contains blank or duplicate entries, or ``valid`` does
        not agree with whether ``errors`` is empty.

    Notes
    -----
    Validation is entirely static because the report contains no numerical
    JAX leaves.
    """
    normalized_errors: Tuple[str, ...] = _normalize_terms(
        errors,
        field_name="errors",
    )
    if valid == bool(normalized_errors):
        msg: str = "valid must be true exactly when errors is empty"
        raise ValueError(msg)
    report: CompositionReport = CompositionReport(
        valid=valid,
        errors=normalized_errors,
        available_semantics=_normalize_terms(
            available_semantics,
            field_name="available_semantics",
        ),
        destroyed_information=_normalize_terms(
            destroyed_information,
            field_name="destroyed_information",
        ),
        invalidated_claims=_normalize_terms(
            invalidated_claims,
            field_name="invalidated_claims",
        ),
        transformation_refs=_normalize_terms(
            transformation_refs,
            field_name="transformation_refs",
        ),
    )
    return report


__all__: list[str] = [
    "CompositionReport",
    "TransformationContract",
    "make_composition_report",
    "make_transformation_contract",
]
