"""Define semantic contracts for composable certified transformations.

Extended Summary
----------------
Transformation contracts state the meaning that a step requires and
produces. They identify the preserved properties, destroyed information, and
invalid prior claims. Composition applies a conservative rule. An inherited
property survives only when the next transformation names it in
``preserves``.

Contracts are static Equinox PyTrees.  They select and describe JAX programs;
continuous numerical checks remain in the certification carriers and kernels.

Routine Listings
----------------
:func:`validate_contract`
    Return structural errors for a raw or deserialized contract.
:func:`validate_composition`
    Validate and conservatively compose transformation semantics.
:func:`compose_transformations`
    Compose contracts and raise for unsatisfied requirements.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

from beartype import beartype
from beartype.typing import Any, Tuple
from jaxtyping import jaxtyped

from diffpes.types import (
    CompositionReport,
    TransformationContract,
    make_composition_report,
    make_transformation_contract,
)


def _normalize_initial_semantics(values: Iterable[str]) -> Tuple[str, ...]:
    """PRIVATE: Validate initial semantics without constructing a
    carrier.

    Parameters
    ----------
    values : Iterable[str]
        Declared semantic properties that hold before the first
        transformation.

    Returns
    -------
    normalized : Tuple[str, ...]
        The properties as a tuple in input order.

    Raises
    ------
    ValueError
        If an entry is not a nonblank string, or if entries repeat.

    Notes
    -----
    Materializes the iterable once and checks each entry is a nonblank
    string and that no entry repeats. The tuple then seeds the semantic
    state of the composition walk.
    """
    normalized: Tuple[str, ...] = tuple(values)
    if any(
        not isinstance(value, str) or not value.strip() for value in normalized
    ):
        msg: str = "initial_semantics entries must be nonblank strings"
        raise ValueError(msg)
    if len(set(normalized)) != len(normalized):
        msg: str = "initial_semantics entries must be unique"
        raise ValueError(msg)
    return normalized


@jaxtyped(typechecker=beartype)
def validate_contract(contract: TransformationContract) -> Tuple[str, ...]:
    """Return structural errors for a raw or deserialized contract.

    The operation propagates declared semantics and information loss
    conservatively. It never infers that an undeclared scientific property
    survives.

    :see: :class:`~.test_contracts.TestValidateContract`

    Implementation Logic
    --------------------
    1. **Bind the documented output**::

           result: Tuple[str, ...] = tuple(errors)

       The function validates and transforms the inputs before it binds the
       documented output.

    Parameters
    ----------
    contract : TransformationContract
        Semantic and information-loss declaration to validate.

    Returns
    -------
    result : Tuple[str, ...]
        Deterministic structural error messages, empty when valid.
    """
    exc: ValueError | TypeError
    errors: list[str] = []
    try:
        make_transformation_contract(
            contract.transformation_id,
            contract.transformation_version,
            requires=contract.requires,
            produces=contract.produces,
            preserves=contract.preserves,
            introduces=contract.introduces,
            destroys=contract.destroys,
            invalidates_claims=contract.invalidates_claims,
            jax_pure=contract.jax_pure,
        )
    except (ValueError, TypeError) as exc:
        errors.append(str(exc))
    result: Tuple[str, ...] = tuple(errors)
    return result


@jaxtyped(typechecker=beartype)
def validate_composition(
    contracts: Sequence[TransformationContract],
    *,
    initial_semantics: Iterable[str] = (),
) -> CompositionReport:
    """Validate and conservatively compose transformation semantics.

    The operation propagates declared semantics and information loss
    conservatively. It never infers that an undeclared scientific property
    survives.

    :see: :class:`~.test_contracts.TestValidateComposition`


    Parameters
    ----------
    contracts : Sequence[TransformationContract]
        Ordered transformation pipeline.
    initial_semantics : Iterable[str], optional
        Properties available on the pipeline inputs.

    Returns
    -------
    report : CompositionReport
        Deterministic semantics, information-loss, and claim-invalidation
        summary. The report contains invalid composition instead of raising an
        exception.

    Notes
    -----
    The operation treats a property as lost when ``preserves`` does not name
    it at that step. This rule prevents an undeclared transformation from
    retaining a scientific claim.
    """
    index: Any
    contract: Any
    error: Any
    initial: Tuple[str, ...] = _normalize_initial_semantics(initial_semantics)
    current: set[str] = set(initial)
    losses: set[str] = set()
    invalidated: set[str] = set()
    errors: list[str] = []
    references: list[str] = []
    for index, contract in enumerate(contracts):
        reference: str = (
            f"{contract.transformation_id}@{contract.transformation_version}"
        )
        references.append(reference)
        for error in validate_contract(contract):
            errors.append(f"step {index} ({reference}): {error}")
        missing: set[str] = set(contract.requires) - current
        if missing:
            details: str = ", ".join(sorted(missing))
            errors.append(
                f"step {index} ({reference}) is missing required semantics: "
                f"{details}"
            )
        retained: set[str] = current & set(contract.preserves)
        losses.update(current - retained)
        losses.update(contract.destroys)
        current = retained
        current.difference_update(contract.destroys)
        current.update(contract.introduces)
        current.update(contract.produces)
        invalidated.update(contract.invalidates_claims)
    report: CompositionReport = make_composition_report(
        valid=not errors,
        errors=tuple(errors),
        available_semantics=tuple(sorted(current)),
        destroyed_information=tuple(sorted(losses)),
        invalidated_claims=tuple(sorted(invalidated)),
        transformation_refs=tuple(references),
    )
    return report


@jaxtyped(typechecker=beartype)
def compose_transformations(
    contracts: Sequence[TransformationContract],
    *,
    initial_semantics: Iterable[str] = (),
) -> CompositionReport:
    """Compose contracts and raise for unsatisfied requirements.

    The operation propagates declared semantics and information loss
    conservatively. It never infers that an undeclared scientific property
    survives.

    :see: :class:`~.test_contracts.TestComposeTransformations`


    Implementation Logic
    --------------------
    1. **Bind the documented output**::

           msg: str = "; ".join(report.errors)

       The function validates and transforms the inputs before it binds the
       documented output.

    Parameters
    ----------
    contracts : Sequence[TransformationContract]
        Ordered transformation pipeline.
    initial_semantics : Iterable[str], optional
        Properties present before the first step.

    Returns
    -------
    report : CompositionReport
        Valid composition report.

    Raises
    ------
    ValueError
        If any registered semantic requirement is missing.
    """
    report: CompositionReport = validate_composition(
        contracts,
        initial_semantics=initial_semantics,
    )
    if not report.valid:
        msg: str = "; ".join(report.errors)
        raise ValueError(msg)
    return report


__all__: list[str] = [
    "compose_transformations",
    "validate_composition",
    "validate_contract",
]
