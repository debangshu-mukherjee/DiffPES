"""Register certified models and transformations deterministically.

Extended Summary
----------------
The registry binds permanent scientific identities to model executors and
semantic transformation contracts. It stores entries as sorted immutable
tuples and rejects duplicate identities. Callers receive frozen snapshots.
Registration order therefore cannot alter lookup, listing, or the registry
consistency checksum.

Registry checksums are non-security bookkeeping values.  They identify an
accidental record mismatch and never establish scientific validity or the
identity of an author.

Routine Listings
----------------
:func:`register_model`
    Register an exact model identity once.
:func:`get_model`
    Resolve an exact registered model.
:func:`register_transformation`
    Register an exact transformation contract once.
:func:`register_handshake`
    Register declarative requirements from one domain owner.
:func:`get_transformation`
    Resolve an exact registered transformation contract.
:func:`list_models`
    Return model specifications in deterministic identity order.
:func:`list_registered_models`
    Return an immutable deterministic snapshot including executors.
:func:`list_transformations`
    Return transformation contracts in deterministic identity order.
:func:`list_handshakes`
    Return owner handshakes in deterministic identity order.
:func:`registry_snapshot`
    Return one internally consistent immutable registry snapshot.
:func:`freeze_registry`
    Prevent later registration and return the final immutable snapshot.
:func:`validate_registry`
    Recompute registry structure and consistency checksums.
:func:`validate_handshake`
    Validate one owner handshake against available records.
:func:`validate_registry_manifest`
    Compare the packaged registry manifest with live entries.
:func:`render_model_card`
    Render a model card directly from a model specification.
:func:`packaged_model_card`
    Read the packaged generated card for one model identity.
:func:`registry_manifest`
    Read the packaged registry manifest.
"""

from __future__ import annotations

import json
import threading
from functools import cache
from importlib import resources

from beartype import beartype
from beartype.typing import Any, Callable
from jaxtyping import jaxtyped

from diffpes.types import (
    CERTIFICATION_IDENTIFIER_PATTERN,
    CERTIFICATION_SEMVER_PATTERN,
    ForwardModelSpec,
    HandshakeReport,
    RegisteredModel,
    RegisteredTransformation,
    RegistrationHandshake,
    RegistryReport,
    RegistrySnapshot,
    TransformationContract,
    make_handshake_report,
    make_registered_model,
    make_registered_transformation,
    make_registration_handshake,
    make_registry_report,
    make_registry_snapshot,
)

from .checksums import checksum_pytree
from .contracts import validate_contract


class _RegistryState:
    """Store mutable process-local state behind immutable public snapshots."""

    def __init__(self) -> None:
        self.models: tuple[RegisteredModel, ...] = ()
        self.transformations: tuple[RegisteredTransformation, ...] = ()
        self.handshakes: tuple[RegistrationHandshake, ...] = ()
        self.frozen = False
        self.lock = threading.RLock()


@cache
def _registry_state() -> _RegistryState:
    """Return the lazily initialized process-local registry state."""
    state: _RegistryState = _RegistryState()
    return state


def _model_key(entry: RegisteredModel) -> tuple[str, str]:
    """Return a sortable exact scientific model identity."""
    key: tuple[str, str] = (entry.spec.model_id, entry.spec.model_version)
    return key


def _transformation_key(
    entry: RegisteredTransformation,
) -> tuple[str, str]:
    """Return a sortable exact transformation identity."""
    contract: TransformationContract = entry.contract
    key: tuple[str, str] = (
        contract.transformation_id,
        contract.transformation_version,
    )
    return key


def _handshake_key(entry: RegistrationHandshake) -> str:
    """Return the stable owner key for a registration handshake."""
    key: str = entry.owner_id
    return key


def _validate_model_spec(spec: ForwardModelSpec) -> None:
    """Validate the registry-facing identity fields of a model spec."""
    if CERTIFICATION_IDENTIFIER_PATTERN.fullmatch(spec.model_id) is None:
        msg: str = "model_id must be a lowercase reverse-DNS-like ID"
        raise ValueError(msg)
    if CERTIFICATION_SEMVER_PATTERN.fullmatch(spec.model_version) is None:
        msg: str = "model_version must be a semantic version"
        raise ValueError(msg)
    if CERTIFICATION_IDENTIFIER_PATTERN.fullmatch(spec.observable_id) is None:
        msg: str = "observable_id must be a lowercase reverse-DNS-like ID"
        raise ValueError(msg)
    if not spec.implementation_ref.strip():
        msg: str = "implementation_ref must be nonblank"
        raise ValueError(msg)


def _ensure_open() -> None:
    """Reject mutation once an application has frozen registration."""
    if _registry_state().frozen:
        msg: str = "the certification registry is frozen"
        raise ValueError(msg)


@jaxtyped(typechecker=beartype)
def register_model(
    spec: ForwardModelSpec,
    executor: Callable[..., Any],
) -> None:
    """Register an exact model identity once.

    The process-local registry uses exact scientific identities and
    deterministic ordering. Freezing prevents later mutation of the selected
    programs.

    :see: :class:`~.test_registry.TestRegisterModel`


    Implementation Logic
    --------------------
    1. **Bind the documented output**::

           state.models = tuple(sorted((*state.models, entry), key=_model_key))

       The function validates and transforms the inputs before it binds the
       documented output.

    Parameters
    ----------
    spec : ForwardModelSpec
        Immutable scientific model specification.
    executor : Callable[..., Any]
        Pure model implementation. Certification later verifies its JAX
        behavior. Registration only requires a callable.

    Raises
    ------
    ValueError
        If the identity is invalid or already exists. The function also raises
        when the registry no longer accepts changes.
    """
    _validate_model_spec(spec)
    checksum: str = checksum_pytree(
        spec,
        record_kind="model-registration",
    )
    entry: RegisteredModel = make_registered_model(
        spec=spec,
        executor=executor,
        registration_checksum=checksum,
    )
    key: tuple[str, str] = _model_key(entry)
    state: _RegistryState = _registry_state()
    with state.lock:
        _ensure_open()
        if any(_model_key(existing) == key for existing in state.models):
            msg: str = f"duplicate model identity: {key[0]}@{key[1]}"
            raise ValueError(msg)
        state.models = tuple(sorted((*state.models, entry), key=_model_key))


@jaxtyped(typechecker=beartype)
def register_transformation(contract: TransformationContract) -> None:
    """Register an exact transformation contract once.

    The process-local registry uses exact scientific identities and
    deterministic ordering. Freezing prevents later mutation of the selected
    programs.

    :see: :class:`~.test_registry.TestRegisterTransformation`


    Implementation Logic
    --------------------
    1. **Bind the documented output**::

           state.transformations = tuple(
                       sorted(
                           (*state.transformations, entry),
                           key=_transformation_key,
                       )
                   )

       The function validates and transforms the inputs before it binds the
       documented output.

    Parameters
    ----------
    contract : TransformationContract
        Validated semantic and information-loss declaration.

    Raises
    ------
    ValueError
        If the contract is invalid or duplicated. The function also raises
        when the registry no longer accepts changes.
    """
    errors: tuple[str, ...] = validate_contract(contract)
    if errors:
        msg: str = "; ".join(errors)
        raise ValueError(msg)
    checksum: str = checksum_pytree(
        contract,
        record_kind="transformation-registration",
    )
    entry: RegisteredTransformation = make_registered_transformation(
        contract=contract,
        registration_checksum=checksum,
    )
    key: tuple[str, str] = _transformation_key(entry)
    state: _RegistryState = _registry_state()
    with state.lock:
        _ensure_open()
        if any(
            _transformation_key(existing) == key
            for existing in state.transformations
        ):
            msg: str = f"duplicate transformation identity: {key[0]}@{key[1]}"
            raise ValueError(msg)
        state.transformations = tuple(
            sorted(
                (*state.transformations, entry),
                key=_transformation_key,
            )
        )


@jaxtyped(typechecker=beartype)
def get_model(model_id: str, model_version: str) -> RegisteredModel:
    """Resolve an exact registered model.

    The process-local registry uses exact scientific identities and
    deterministic ordering. Freezing prevents later mutation of the selected
    programs.

    :see: :class:`~.test_registry.TestGetModel`


    Implementation Logic
    --------------------
    1. **Bind the documented output**::

           msg: str = f"unknown model identity: {model_id}@{model_version}"

       The function validates and transforms the inputs before it binds the
       documented output.

    Parameters
    ----------
    model_id : str
        Permanent scientific model identifier.
    model_version : str
        Exact semantic model version.

    Returns
    -------
    model : RegisteredModel
        Immutable registered model binding.

    Raises
    ------
    KeyError
        If the exact ID and semantic version are absent.
    """
    entry: Any
    key: tuple[str, str] = (model_id, model_version)
    state: _RegistryState = _registry_state()
    with state.lock:
        for entry in state.models:
            if _model_key(entry) == key:
                model: RegisteredModel = entry
                return model
    msg: str = f"unknown model identity: {model_id}@{model_version}"
    raise KeyError(msg)


@jaxtyped(typechecker=beartype)
def get_transformation(
    transformation_id: str,
    transformation_version: str,
) -> RegisteredTransformation:
    """Resolve an exact registered transformation contract.

    The process-local registry uses exact scientific identities and
    deterministic ordering. Freezing prevents later mutation of the selected
    programs.

    :see: :class:`~.test_registry.TestGetTransformation`


    Implementation Logic
    --------------------
    1. **Bind the documented output**::

           msg: str = (
                   "unknown transformation identity: "
                   f"{transformation_id}@{transformation_version}"
               )

       The function validates and transforms the inputs before it binds the
       documented output.

    Parameters
    ----------
    transformation_id : str
        Permanent transformation identifier.
    transformation_version : str
        Exact semantic transformation version.

    Returns
    -------
    transformation : RegisteredTransformation
        Immutable registered transformation binding.

    Raises
    ------
    KeyError
        If the exact ID and semantic version are absent.
    """
    entry: Any
    key: tuple[str, str] = (transformation_id, transformation_version)
    state: _RegistryState = _registry_state()
    with state.lock:
        for entry in state.transformations:
            if _transformation_key(entry) == key:
                transformation: RegisteredTransformation = entry
                return transformation
    msg: str = (
        "unknown transformation identity: "
        f"{transformation_id}@{transformation_version}"
    )
    raise KeyError(msg)


@jaxtyped(typechecker=beartype)
def list_models() -> tuple[ForwardModelSpec, ...]:
    """Return model specifications in deterministic identity order.

    The process-local registry uses exact scientific identities and
    deterministic ordering. Freezing prevents later mutation of the selected
    programs.

    :see: :class:`~.test_registry.TestListModels`

    Implementation Logic
    --------------------
    1. **Bind the documented output**::

           models: tuple[ForwardModelSpec, ...] = tuple(
                       entry.spec for entry in state.models
                   )

       The function validates and transforms the inputs before it binds the
       documented output.

    Returns
    -------
    models : tuple[ForwardModelSpec, ...]
        Immutable sorted model specifications without executor callables.
    """
    state: _RegistryState = _registry_state()
    with state.lock:
        models: tuple[ForwardModelSpec, ...] = tuple(
            entry.spec for entry in state.models
        )
    return models


@jaxtyped(typechecker=beartype)
def list_registered_models() -> tuple[RegisteredModel, ...]:
    """Return an immutable deterministic snapshot including executors.

    The process-local registry uses exact scientific identities and
    deterministic ordering. Freezing prevents later mutation of the selected
    programs.

    :see: :class:`~.test_registry.TestListRegisteredModels`

    Implementation Logic
    --------------------
    1. **Bind the documented output**::

           models: tuple[RegisteredModel, ...] = state.models

       The function validates and transforms the inputs before it binds the
       documented output.

    Returns
    -------
    models : tuple[RegisteredModel, ...]
        Immutable sorted model bindings including static executor callables.
    """
    state: _RegistryState = _registry_state()
    with state.lock:
        models: tuple[RegisteredModel, ...] = state.models
    return models


@jaxtyped(typechecker=beartype)
def list_transformations() -> tuple[TransformationContract, ...]:
    """Return transformation contracts in deterministic identity order.

    The process-local registry uses exact scientific identities and
    deterministic ordering. Freezing prevents later mutation of the selected
    programs.

    :see: :class:`~.test_registry.TestListTransformations`

    Implementation Logic
    --------------------
    1. **Bind the documented output**::

           transformations: tuple[TransformationContract, ...] = tuple(
                       entry.contract for entry in state.transformations
                   )

       The function validates and transforms the inputs before it binds the
       documented output.

    Returns
    -------
    transformations : tuple[TransformationContract, ...]
        Immutable sorted semantic and information-loss contracts.
    """
    state: _RegistryState = _registry_state()
    with state.lock:
        transformations: tuple[TransformationContract, ...] = tuple(
            entry.contract for entry in state.transformations
        )
    return transformations


@jaxtyped(typechecker=beartype)
def register_handshake(handshake: RegistrationHandshake) -> None:
    """Register declarative requirements from one domain owner.

    The registry stores requirements before or after the owner registers them.

    :see: :class:`~.test_registry.TestRegisterHandshake`

    Implementation Logic
    --------------------
    1. **Store the sorted declaration**::

           state.handshakes = tuple(
               sorted((*state.handshakes, handshake), key=_handshake_key)
           )

       Stable owner ordering removes import-order effects.

    Parameters
    ----------
    handshake : RegistrationHandshake
        Exact model, transformation, convention, and evidence references.

    Raises
    ------
    ValueError
        If the owner has a handshake or the registry has a frozen state.
    """
    state: _RegistryState = _registry_state()
    with state.lock:
        _ensure_open()
        if any(
            item.owner_id == handshake.owner_id for item in state.handshakes
        ):
            msg: str = f"duplicate handshake owner: {handshake.owner_id}"
            raise ValueError(msg)
        state.handshakes = tuple(
            sorted((*state.handshakes, handshake), key=_handshake_key)
        )


@jaxtyped(typechecker=beartype)
def list_handshakes() -> tuple[RegistrationHandshake, ...]:
    """Return owner handshakes in deterministic identity order.

    The process-local registry returns one immutable tuple.

    :see: :class:`~.test_registry.TestListHandshakes`

    Returns
    -------
    handshakes : tuple[RegistrationHandshake, ...]
        Immutable sorted handshake declarations.

    Notes
    -----
    The function reads the tuple while the registry lock is active.
    """
    state: _RegistryState = _registry_state()
    with state.lock:
        handshakes: tuple[RegistrationHandshake, ...] = state.handshakes
    return handshakes


@jaxtyped(typechecker=beartype)
def validate_handshake(
    handshake: RegistrationHandshake,
    *,
    evidence_ids: tuple[str, ...] = (),
) -> HandshakeReport:
    """Validate one owner handshake against available records.

    The report names each exact reference that has no available binding.

    :see: :class:`~.test_registry.TestValidateHandshake`

    Implementation Logic
    --------------------
    1. **Collect missing references**::

           missing = tuple(
               reference
               for required, available in available_groups
               for reference in required
               if reference not in available
           )

       The comparison uses exact versioned identities from each registry.

    Parameters
    ----------
    handshake : RegistrationHandshake
        Declarative requirements supplied by the domain owner.
    evidence_ids : tuple[str, ...]
        Available evidence record IDs. Default is an empty tuple.

    Returns
    -------
    report : HandshakeReport
        Completion state and sorted missing exact references.
    """
    model_refs: set[str] = {
        f"{item.model_id}@{item.model_version}" for item in list_models()
    }
    transformation_refs: set[str] = {
        f"{item.transformation_id}@{item.transformation_version}"
        for item in list_transformations()
    }
    convention_refs: set[str] = {
        f"{item.convention_id}@{item.version}"
        for model in list_models()
        for item in model.conventions
    }
    available_groups: tuple[tuple[tuple[str, ...], set[str]], ...] = (
        (handshake.model_refs, model_refs),
        (handshake.transformation_refs, transformation_refs),
        (handshake.convention_refs, convention_refs),
        (handshake.evidence_ids, set(evidence_ids)),
    )
    missing: tuple[str, ...] = tuple(
        sorted(
            reference
            for required, available in available_groups
            for reference in required
            if reference not in available
        )
    )
    report: HandshakeReport = make_handshake_report(
        owner_id=handshake.owner_id,
        complete=not missing,
        missing_ids=missing,
    )
    return report


@jaxtyped(typechecker=beartype)
def registry_manifest() -> dict[str, Any]:
    """Read the packaged registry manifest.

    The manifest records generated model and transformation identities.

    :see: :class:`~.test_registry.TestRegistryManifest`

    Implementation Logic
    --------------------
    1. **Parse the package resource**::

           decoded = json.loads(text)

       The function rejects a root that is not a JSON object.

    Returns
    -------
    manifest : dict[str, Any]
        Parsed manifest with generated model and transformation identities.

    Raises
    ------
    ValueError
        If the manifest root is not a JSON object.
    """
    text: str = (
        resources.files("diffpes.certify")
        .joinpath("_registry", "manifest.json")
        .read_text(encoding="utf-8")
    )
    decoded: Any = json.loads(text)
    if not isinstance(decoded, dict):
        msg: str = "registry manifest root must be an object"
        raise ValueError(msg)
    return decoded


@jaxtyped(typechecker=beartype)
def render_model_card(spec: ForwardModelSpec) -> str:
    r"""Render a model card directly from a model specification.

    The generated Markdown contains no separately maintained scientific data.

    :see: :class:`~.test_registry.TestRenderModelCard`

    Implementation Logic
    --------------------
    1. **Render registry fields**::

           card = f"# {spec.model_id}\\n\\nVersion: `{spec.model_version}`."

       The complete output also lists assumptions, conventions, and domains.

    Parameters
    ----------
    spec : ForwardModelSpec
        Registered scientific model specification.

    Returns
    -------
    card : str
        Deterministic Markdown generated only from registry truth.
    """
    assumptions: str = "\n".join(
        f"- The model uses `{item}`." for item in spec.assumptions
    )
    conventions: str = "\n".join(
        f"- The model uses `{item.convention_id}@{item.version}`."
        for item in spec.conventions
    )
    domains: str = "\n".join(
        f"- `{item.predicate_id}` uses `{item.expression_id}` with "
        f"`{item.severity}` severity."
        for item in spec.domain
    )
    card: str = (
        f"# {spec.model_id}\n\n"
        f"Version: `{spec.model_version}`.\n\n"
        f"Observable: `{spec.observable_id}`.\n\n"
        f"Implementation: `{spec.implementation_ref}`.\n\n"
        "## Assumptions\n\n"
        f"{assumptions}\n\n"
        "## Conventions\n\n"
        f"{conventions}\n\n"
        "## Domain\n\n"
        f"{domains}\n"
    )
    return card


@jaxtyped(typechecker=beartype)
def packaged_model_card(model_id: str, model_version: str) -> str:
    """Read the packaged generated card for one model identity.

    The filename combines the permanent model ID with its semantic version.

    :see: :class:`~.test_registry.TestPackagedModelCard`

    Implementation Logic
    --------------------
    1. **Read the generated resource**::

           filename = f"{model_id}@{model_version}.md"

       The package resource contains the canonical generated Markdown view.

    Parameters
    ----------
    model_id : str
        Exact permanent model ID.
    model_version : str
        Exact semantic model version.

    Returns
    -------
    card : str
        Packaged Markdown model card.
    """
    filename: str = f"{model_id}@{model_version}.md"
    card: str = (
        resources.files("diffpes.certify")
        .joinpath("_registry", "model-cards", filename)
        .read_text(encoding="utf-8")
    )
    return card


@jaxtyped(typechecker=beartype)
def validate_registry_manifest() -> tuple[str, ...]:
    """Compare the packaged registry manifest with live entries.

    The comparison detects missing entries and generated model-card drift.

    :see: :class:`~.test_registry.TestValidateRegistryManifest`

    Implementation Logic
    --------------------
    1. **Compare packaged entries**::

           manifest = registry_manifest()

       The function compares each manifest identity with the live registry.

    Returns
    -------
    errors : tuple[str, ...]
        Sorted missing-entry and generated-card drift messages.
    """
    manifest: dict[str, Any] = registry_manifest()
    errors: list[str] = []
    models: dict[tuple[str, str], ForwardModelSpec] = {
        (item.model_id, item.model_version): item for item in list_models()
    }
    transformations: set[tuple[str, str]] = {
        (item.transformation_id, item.transformation_version)
        for item in list_transformations()
    }
    handshakes: dict[str, RegistrationHandshake] = {
        item.owner_id: item for item in list_handshakes()
    }
    entry: Any
    for entry in manifest.get("models", ()):
        key: tuple[str, str] = (entry["model_id"], entry["model_version"])
        if key not in models:
            errors.append(f"missing packaged model: {key[0]}@{key[1]}")
            continue
        generated: str = render_model_card(models[key])
        packaged: str = packaged_model_card(*key)
        if generated != packaged:
            errors.append(f"model card drift: {key[0]}@{key[1]}")
    for entry in manifest.get("transformations", ()):
        key = (
            entry["transformation_id"],
            entry["transformation_version"],
        )
        if key not in transformations:
            errors.append(
                f"missing packaged transformation: {key[0]}@{key[1]}"
            )
    for entry in manifest.get("handshakes", ()):
        owner_id: str = entry["owner_id"]
        expected: RegistrationHandshake = make_registration_handshake(
            owner_id=owner_id,
            model_refs=tuple(entry["model_refs"]),
            transformation_refs=tuple(entry["transformation_refs"]),
            convention_refs=tuple(entry["convention_refs"]),
            evidence_ids=tuple(entry["evidence_ids"]),
        )
        actual: RegistrationHandshake | None = handshakes.get(owner_id)
        if actual is None:
            errors.append(f"missing packaged handshake: {owner_id}")
        elif actual != expected:
            errors.append(f"packaged handshake drift: {owner_id}")
    result: tuple[str, ...] = tuple(sorted(errors))
    return result


def _registry_checksum(
    models: tuple[RegisteredModel, ...],
    transformations: tuple[RegisteredTransformation, ...],
) -> str:
    """Identify sorted entry contents without serializing callables."""
    payload: tuple[tuple[tuple[str, str, str], ...], ...] = (
        tuple(
            (
                entry.spec.model_id,
                entry.spec.model_version,
                entry.registration_checksum,
            )
            for entry in models
        ),
        tuple(
            (
                entry.contract.transformation_id,
                entry.contract.transformation_version,
                entry.registration_checksum,
            )
            for entry in transformations
        ),
    )
    checksum: str = checksum_pytree(payload, record_kind="registry")
    return checksum


@jaxtyped(typechecker=beartype)
def registry_snapshot() -> RegistrySnapshot:
    """Return one internally consistent immutable registry snapshot.

    The process-local registry uses exact scientific identities and
    deterministic ordering. Freezing prevents later mutation of the selected
    programs.

    :see: :class:`~.test_registry.TestRegistrySnapshot`

    Implementation Logic
    --------------------
    1. **Bind the documented output**::

           snapshot: RegistrySnapshot = make_registry_snapshot(
                   models=models,
                   transformations=transformations,
                   checksum=checksum,
               )

       The function validates and transforms the inputs before it binds the
       documented output.

    Returns
    -------
    snapshot : RegistrySnapshot
        Models, transformations, and their non-security consistency checksum.
    """
    state: _RegistryState = _registry_state()
    with state.lock:
        models: tuple[RegisteredModel, ...] = state.models
        transformations: tuple[RegisteredTransformation, ...] = (
            state.transformations
        )
        checksum: str = _registry_checksum(models, transformations)
    snapshot: RegistrySnapshot = make_registry_snapshot(
        models=models,
        transformations=transformations,
        checksum=checksum,
    )
    return snapshot


@jaxtyped(typechecker=beartype)
def freeze_registry() -> RegistrySnapshot:
    """Prevent later registration and return the final immutable snapshot.

    The process-local registry uses exact scientific identities and
    deterministic ordering. Freezing prevents later mutation of the selected
    programs.

    :see: :class:`~.test_registry.TestFreezeRegistry`

    Returns
    -------
    snapshot : RegistrySnapshot
        Final immutable contents after the function disables mutation.

    Notes
    -----
    Freezing is process-local eager registry control and is never invoked from
    a traced numerical kernel.
    """
    state: _RegistryState = _registry_state()
    with state.lock:
        state.frozen = True
        snapshot: RegistrySnapshot = registry_snapshot()
    return snapshot


@jaxtyped(typechecker=beartype)
def validate_registry() -> RegistryReport:
    """Recompute registry structure and consistency checksums.

    The process-local registry uses exact scientific identities and
    deterministic ordering. Freezing prevents later mutation of the selected
    programs.

    :see: :class:`~.test_registry.TestValidateRegistry`


    Implementation Logic
    --------------------
    1. **Bind the documented output**::

           report: RegistryReport = make_registry_report(
                   valid=not errors,
                   errors=tuple(errors),
                   model_count=len(models),
                   transformation_count=len(transformations),
                   checksum=checksum,
                   frozen=frozen,
               )

       The function validates and transforms the inputs before it binds the
       documented output.

    Returns
    -------
    report : RegistryReport
        Validation errors, entry counts, deterministic checksum, and frozen
        state. The checksum has bookkeeping meaning only.
    """
    entry: Any
    state: _RegistryState = _registry_state()
    with state.lock:
        models: tuple[RegisteredModel, ...] = state.models
        transformations: tuple[RegisteredTransformation, ...] = (
            state.transformations
        )
        frozen: bool = state.frozen
    errors: list[str] = []
    model_keys: tuple[tuple[str, str], ...] = tuple(
        _model_key(entry) for entry in models
    )
    transformation_keys: tuple[tuple[str, str], ...] = tuple(
        _transformation_key(entry) for entry in transformations
    )
    if model_keys != tuple(sorted(model_keys)):
        errors.append("model entries are not deterministically sorted")
    if len(set(model_keys)) != len(model_keys):
        errors.append("model identities are not unique")
    if transformation_keys != tuple(sorted(transformation_keys)):
        errors.append(
            "transformation entries are not deterministically sorted"
        )
    if len(set(transformation_keys)) != len(transformation_keys):
        errors.append("transformation identities are not unique")
    exc: ValueError
    for entry in models:
        try:
            _validate_model_spec(entry.spec)
        except ValueError as exc:
            errors.append(str(exc))
        expected: str = checksum_pytree(
            entry.spec,
            record_kind="model-registration",
        )
        if expected != entry.registration_checksum:
            errors.append(
                f"model registration mismatch: "
                f"{entry.spec.model_id}@{entry.spec.model_version}"
            )
    for entry in transformations:
        errors.extend(validate_contract(entry.contract))
        expected = checksum_pytree(
            entry.contract,
            record_kind="transformation-registration",
        )
        if expected != entry.registration_checksum:
            contract: TransformationContract = entry.contract
            errors.append(
                "transformation registration mismatch: "
                f"{contract.transformation_id}@"
                f"{contract.transformation_version}"
            )
    checksum: str = _registry_checksum(models, transformations)
    report: RegistryReport = make_registry_report(
        valid=not errors,
        errors=tuple(errors),
        model_count=len(models),
        transformation_count=len(transformations),
        checksum=checksum,
        frozen=frozen,
    )
    return report


__all__: list[str] = [
    "freeze_registry",
    "get_model",
    "get_transformation",
    "list_handshakes",
    "list_models",
    "list_registered_models",
    "list_transformations",
    "packaged_model_card",
    "register_handshake",
    "register_model",
    "register_transformation",
    "registry_manifest",
    "registry_snapshot",
    "render_model_card",
    "validate_handshake",
    "validate_registry",
    "validate_registry_manifest",
]
