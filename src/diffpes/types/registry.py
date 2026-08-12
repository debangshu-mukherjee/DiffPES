"""Define immutable certification registry records.

Extended Summary
----------------
This module stores registered models, transformations, snapshots,
handshakes, and their structural reports.

Routine Listings
----------------
:class:`HandshakeReport`
    Store the validation outcome for one registration handshake.
:class:`RegisteredModel`
    Store a frozen binding between a model spec and its executor.
:class:`RegisteredTransformation`
    Store a frozen transformation and its consistency checksum.
:class:`RegistrationHandshake`
    Store registration requirements for one certification owner.
:class:`RegistryReport`
    Store the structural validation result for one registry snapshot.
:class:`RegistrySnapshot`
    Store an immutable deterministic snapshot of registry entries.
:func:`make_handshake_report`
    Create a report for one registration handshake.
:func:`make_registered_model`
    Create a validated model-registry binding.
:func:`make_registered_transformation`
    Create a validated transformation-registry binding.
:func:`make_registration_handshake`
    Create registration requirements for one certification owner.
:func:`make_registry_report`
    Create a validated structural registry report.
:func:`make_registry_snapshot`
    Create an immutable registry snapshot.
"""

import equinox as eqx
from beartype import beartype
from beartype.typing import Any, Callable, Tuple
from jaxtyping import Array, Bool, jaxtyped

from .certification_validation import _bool, _require_text, _text_tuple
from .contracts import TransformationContract
from .specification import ForwardModelSpec


class RegisteredModel(eqx.Module):
    """Store a frozen binding between a model spec and its executor.

    Carry scientific vocabulary separately from traced leaves.
    The record remains stable under JIT, VMAP, JVP, and VJP transforms.

    :see: :class:`~.test_registry.TestRegisteredmodel`

    Attributes
    ----------
    spec : ForwardModelSpec
        Spec retained as a differentiable JAX leaf in the declared
        physical units.
    executor : Callable[..., Any]
        Executor (**static** -- a compile-time constant; changing it
        triggers retracing).
    registration_checksum : str
        Registration checksum (**static** -- a compile-time constant;
        changing it triggers retracing).

    See Also
    --------
    make_registered_model : Validated factory for this type.
    """

    spec: ForwardModelSpec
    executor: Callable[..., Any] = eqx.field(static=True)
    registration_checksum: str = eqx.field(static=True)


class RegisteredTransformation(eqx.Module):
    """Store a frozen transformation and its consistency checksum.

    Carry scientific vocabulary separately from traced leaves.
    The record remains stable under JIT, VMAP, JVP, and VJP transforms.

    :see: :class:`~.test_registry.TestRegisteredtransformation`

    Attributes
    ----------
    contract : TransformationContract
        Contract retained as a differentiable JAX leaf in the declared
        physical units.
    registration_checksum : str
        Registration checksum (**static** -- a compile-time constant;
        changing it triggers retracing).

    See Also
    --------
    make_registered_transformation : Validated factory for this type.
    """

    contract: TransformationContract
    registration_checksum: str = eqx.field(static=True)


class RegistrySnapshot(eqx.Module):
    """Store an immutable deterministic snapshot of registry entries.

    Carry scientific vocabulary separately from traced leaves.
    The record remains stable under JIT, VMAP, JVP, and VJP transforms.

    :see: :class:`~.test_registry.TestRegistrysnapshot`

    Attributes
    ----------
    models : Tuple[RegisteredModel, ...]
        Models retained as a differentiable JAX leaf in the declared
        physical units.
    transformations : Tuple[RegisteredTransformation, ...]
        Transformations retained as a differentiable JAX leaf in the
        declared physical units.
    checksum : str
        Checksum (**static** -- a compile-time constant; changing it
        triggers retracing).

    See Also
    --------
    make_registry_snapshot : Validated factory for this type.
    """

    models: Tuple[RegisteredModel, ...]
    transformations: Tuple[RegisteredTransformation, ...]
    checksum: str = eqx.field(static=True)


class RegistryReport(eqx.Module):
    """Store the structural validation result for one registry snapshot.

    Carry scientific vocabulary separately from traced leaves.
    The record remains stable under JIT, VMAP, JVP, and VJP transforms.

    :see: :class:`~.test_registry.TestRegistryreport`

    Attributes
    ----------
    valid : bool
        Valid (**static** -- a compile-time constant; changing it
        triggers retracing).
    errors : Tuple[str, ...]
        Errors (**static** -- a compile-time constant; changing it
        triggers retracing).
    model_count : int
        Model count (**static** -- a compile-time constant; changing it
        triggers retracing).
    transformation_count : int
        Transformation count (**static** -- a compile-time constant;
        changing it triggers retracing).
    checksum : str
        Checksum (**static** -- a compile-time constant; changing it
        triggers retracing).
    frozen : bool
        Frozen (**static** -- a compile-time constant; changing it
        triggers retracing).

    See Also
    --------
    make_registry_report : Validated factory for this type.
    """

    valid: bool = eqx.field(static=True)
    errors: Tuple[str, ...] = eqx.field(static=True)
    model_count: int = eqx.field(static=True)
    transformation_count: int = eqx.field(static=True)
    checksum: str = eqx.field(static=True)
    frozen: bool = eqx.field(static=True)


class RegistrationHandshake(eqx.Module):
    """Store registration requirements for one certification owner.

    The record names required identities without importing an unfinished
    scientific kernel.

    :see: :class:`~.test_registry.TestRegistrationHandshake`

    Attributes
    ----------
    owner_id : str
        Certification owner identity (**static**; changing it causes
        retracing).
    model_refs : Tuple[str, ...]
        Required model identities (**static**; changing them causes retracing).
    transformation_refs : Tuple[str, ...]
        Required transformation identities (**static**; changes cause
        retracing).
    convention_refs : Tuple[str, ...]
        Required convention identities (**static**; changes cause retracing).
    evidence_ids : Tuple[str, ...]
        Required evidence identities (**static**; changing them causes
        retracing).

    See Also
    --------
    make_registration_handshake : Validated factory for this type.
    """

    owner_id: str = eqx.field(static=True)
    model_refs: Tuple[str, ...] = eqx.field(static=True)
    transformation_refs: Tuple[str, ...] = eqx.field(static=True)
    convention_refs: Tuple[str, ...] = eqx.field(static=True)
    evidence_ids: Tuple[str, ...] = eqx.field(static=True)


class HandshakeReport(eqx.Module):
    """Store the validation outcome for one registration handshake.

    The report keeps a JAX Boolean outcome and static missing identities.

    :see: :class:`~.test_registry.TestHandshakeReport`

    Attributes
    ----------
    owner_id : str
        Certification owner identity (**static**; changing it causes
        retracing).
    complete : Bool[Array, ""]
        Whether every declared identity has a registry binding.
    missing_ids : Tuple[str, ...]
        Missing declared identities (**static**; changes cause retracing).

    See Also
    --------
    make_handshake_report : Validated factory for this type.
    """

    owner_id: str = eqx.field(static=True)
    complete: Bool[Array, ""]
    missing_ids: Tuple[str, ...] = eqx.field(static=True)


@jaxtyped(typechecker=beartype)
def make_registered_model(
    spec: ForwardModelSpec,
    executor: Callable[..., Any],
    registration_checksum: str,
) -> RegisteredModel:
    """Create a validated model-registry binding.

    Carry static scientific vocabulary separately from traced numerical leaves
    while preserving the validation boundary defined by this factory.

    :see: :class:`~.test_registry.TestMakeRegisteredModel`

    Parameters
    ----------
    spec : ForwardModelSpec
        Spec used to construct the validated carrier as a traced
        numerical value in the declared physical units.
    executor : Callable[..., Any]
        Executor used to construct the validated carrier (**static** --
        a compile-time constant; changing it triggers retracing).
    registration_checksum : str
        Registration checksum used to construct the validated carrier
        (**static** -- a compile-time constant; changing it triggers
        retracing).

    Returns
    -------
    result : RegisteredModel
        Validated immutable carrier.

    Raises
    ------
    TypeError
        If static structure or cross-record validation fails.

    Notes
    -----
    The factory checks static structure eagerly. JAX array operations validate
    numerical values and preserve differentiation behavior.
    """
    if not callable(executor):
        raise TypeError("model executor must be callable")
    result: RegisteredModel = RegisteredModel(
        spec=spec,
        executor=executor,
        registration_checksum=_require_text(
            registration_checksum,
            "registration_checksum",
        ),
    )
    return result


@jaxtyped(typechecker=beartype)
def make_registered_transformation(
    contract: TransformationContract,
    registration_checksum: str,
) -> RegisteredTransformation:
    """Create a validated transformation-registry binding.

    Carry static scientific vocabulary separately from traced numerical leaves
    while preserving the validation boundary defined by this factory.

    :see: :class:`~.test_registry.TestMakeRegisteredTransformation`

    Parameters
    ----------
    contract : TransformationContract
        Contract used to construct the validated carrier as a traced
        numerical value in the declared physical units.
    registration_checksum : str
        Registration checksum used to construct the validated carrier
        (**static** -- a compile-time constant; changing it triggers
        retracing).

    Returns
    -------
    result : RegisteredTransformation
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
    if contract is None:
        raise ValueError("contract must not be None")
    result: RegisteredTransformation = RegisteredTransformation(
        contract=contract,
        registration_checksum=_require_text(
            registration_checksum,
            "registration_checksum",
        ),
    )
    return result


@jaxtyped(typechecker=beartype)
def make_registry_snapshot(
    models: Tuple[RegisteredModel, ...],
    transformations: Tuple[RegisteredTransformation, ...],
    checksum: str,
) -> RegistrySnapshot:
    """Create an immutable registry snapshot.

    Carry static scientific vocabulary separately from traced numerical leaves
    while preserving the validation boundary defined by this factory.

    :see: :class:`~.test_registry.TestMakeRegistrySnapshot`

    Parameters
    ----------
    models : Tuple[RegisteredModel, ...]
        Models used to construct the validated carrier (**static** -- a
        compile-time constant; changing it triggers retracing).
    transformations : Tuple[RegisteredTransformation, ...]
        Transformations used to construct the validated carrier
        (**static** -- a compile-time constant; changing it triggers
        retracing).
    checksum : str
        Checksum used to construct the validated carrier (**static** --
        a compile-time constant; changing it triggers retracing).

    Returns
    -------
    result : RegistrySnapshot
        Validated immutable carrier.

    Notes
    -----
    The factory checks static structure eagerly. JAX array operations validate
    numerical values and preserve differentiation behavior.
    """
    result: RegistrySnapshot = RegistrySnapshot(
        models=tuple(models),
        transformations=tuple(transformations),
        checksum=_require_text(checksum, "checksum"),
    )
    return result


@jaxtyped(typechecker=beartype)
def make_registry_report(
    valid: bool,
    errors: Tuple[str, ...],
    model_count: int,
    transformation_count: int,
    checksum: str,
    frozen: bool,
) -> RegistryReport:
    """Create a validated structural registry report.

    Carry static scientific vocabulary separately from traced numerical leaves
    while preserving the validation boundary defined by this factory.

    :see: :class:`~.test_registry.TestMakeRegistryReport`

    Parameters
    ----------
    valid : bool
        Valid used to construct the validated carrier (**static** -- a
        compile-time constant; changing it triggers retracing).
    errors : Tuple[str, ...]
        Errors used to construct the validated carrier (**static** -- a
        compile-time constant; changing it triggers retracing).
    model_count : int
        Model count used to construct the validated carrier (**static**
        -- a compile-time constant; changing it triggers retracing).
    transformation_count : int
        Transformation count used to construct the validated carrier
        (**static** -- a compile-time constant; changing it triggers
        retracing).
    checksum : str
        Checksum used to construct the validated carrier (**static** --
        a compile-time constant; changing it triggers retracing).
    frozen : bool
        Frozen used to construct the validated carrier (**static** -- a
        compile-time constant; changing it triggers retracing).

    Returns
    -------
    result : RegistryReport
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
    if model_count < 0 or transformation_count < 0:
        raise ValueError("registry entry counts must be nonnegative")
    result: RegistryReport = RegistryReport(
        valid=valid,
        errors=tuple(errors),
        model_count=model_count,
        transformation_count=transformation_count,
        checksum=_require_text(checksum, "checksum"),
        frozen=frozen,
    )
    return result


@jaxtyped(typechecker=beartype)
def make_registration_handshake(
    owner_id: str,
    model_refs: Tuple[str, ...] = (),
    transformation_refs: Tuple[str, ...] = (),
    convention_refs: Tuple[str, ...] = (),
    evidence_ids: Tuple[str, ...] = (),
) -> RegistrationHandshake:
    """Create registration requirements for one certification owner.

    The factory validates each static identity without importing a scientific
    executor.

    :see: :class:`~.test_registry.TestMakeRegistrationHandshake`

    Parameters
    ----------
    owner_id : str
        Certification owner identity (**static**; changing it causes
        retracing).
    model_refs : Tuple[str, ...]
        Required model identities. Default is an empty tuple.
    transformation_refs : Tuple[str, ...]
        Required transformation identities. Default is an empty tuple.
    convention_refs : Tuple[str, ...]
        Required convention identities. Default is an empty tuple.
    evidence_ids : Tuple[str, ...]
        Required evidence identities. Default is an empty tuple.

    Returns
    -------
    result : RegistrationHandshake
        Validated declarative registration requirements.

    Notes
    -----
    The factory validates each static identity before module construction.
    """
    result: RegistrationHandshake = RegistrationHandshake(
        owner_id=_require_text(owner_id, "owner_id"),
        model_refs=_text_tuple(model_refs, "model_refs"),
        transformation_refs=_text_tuple(
            transformation_refs, "transformation_refs"
        ),
        convention_refs=_text_tuple(convention_refs, "convention_refs"),
        evidence_ids=_text_tuple(evidence_ids, "evidence_ids"),
    )
    return result


@jaxtyped(typechecker=beartype)
def make_handshake_report(
    owner_id: str,
    complete: Any,
    missing_ids: Tuple[str, ...] = (),
) -> HandshakeReport:
    """Create a report for one registration handshake.

    The report keeps the completion outcome as a JAX Boolean leaf.

    :see: :class:`~.test_registry.TestMakeHandshakeReport`

    Parameters
    ----------
    owner_id : str
        Certification owner identity (**static**; changing it causes
        retracing).
    complete : Any
        Whether every declared identity has a registry binding.
    missing_ids : Tuple[str, ...]
        Missing declared identities. Default is an empty tuple.

    Returns
    -------
    result : HandshakeReport
        Validated completion outcome and missing identities.

    Notes
    -----
    The factory converts the completion outcome to a scalar JAX Boolean.
    """
    missing: Tuple[str, ...] = _text_tuple(missing_ids, "missing_ids")
    result: HandshakeReport = HandshakeReport(
        owner_id=_require_text(owner_id, "owner_id"),
        complete=_bool(complete, "complete", 0),
        missing_ids=missing,
    )
    return result


__all__: list[str] = [
    "HandshakeReport",
    "RegisteredModel",
    "RegisteredTransformation",
    "RegistrationHandshake",
    "RegistryReport",
    "RegistrySnapshot",
    "make_handshake_report",
    "make_registered_model",
    "make_registered_transformation",
    "make_registration_handshake",
    "make_registry_report",
    "make_registry_snapshot",
]
