r"""Load and validate packaged certification registry resources.

Extended Summary
----------------
This module renders model cards from registered scientific records and
compares the packaged generated views with the live registry.

Routine Listings
----------------
:func:`packaged_model_card`
    Read the packaged generated card for one model identity.
:func:`registry_manifest`
    Read the packaged registry manifest.
:func:`render_model_card`
    Render a model card directly from a model specification.
:func:`validate_registry_manifest`
    Compare the packaged registry manifest with live entries.
"""

from __future__ import annotations

import json
from importlib import resources

from beartype import beartype
from beartype.typing import Any, Dict, List, Tuple
from jaxtyping import jaxtyped

from diffpes.types import (
    ForwardModelSpec,
    RegistrationHandshake,
    make_registration_handshake,
)

from .registry import list_handshakes, list_models, list_transformations


@jaxtyped(typechecker=beartype)
def registry_manifest() -> Dict[str, Any]:
    """Read the packaged registry manifest.

    The manifest records generated model and transformation identities.

    :see: :class:`~.test_registry_resources.TestRegistryManifest`

    Implementation Logic
    --------------------
    1. **Parse the package resource**::

           decoded = json.loads(text)

       The function rejects a root that is not a JSON object.

    Returns
    -------
    manifest : Dict[str, Any]
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

    :see: :class:`~.test_registry_resources.TestRenderModelCard`

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

    :see: :class:`~.test_registry_resources.TestPackagedModelCard`

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
def validate_registry_manifest() -> Tuple[str, ...]:
    """Compare the packaged registry manifest with live entries.

    The comparison detects missing entries and generated model-card drift.

    :see: :class:`~.test_registry_resources.TestValidateRegistryManifest`

    Implementation Logic
    --------------------
    1. **Compare packaged entries**::

           manifest = registry_manifest()

       The function compares each manifest identity with the live registry.

    Returns
    -------
    errors : Tuple[str, ...]
        Sorted missing-entry and generated-card drift messages.
    """
    manifest: Dict[str, Any] = registry_manifest()
    errors: List[str] = []
    models: Dict[Tuple[str, str], ForwardModelSpec] = {
        (item.model_id, item.model_version): item for item in list_models()
    }
    transformations: set[Tuple[str, str]] = {
        (item.transformation_id, item.transformation_version)
        for item in list_transformations()
    }
    handshakes: Dict[str, RegistrationHandshake] = {
        item.owner_id: item for item in list_handshakes()
    }
    entry: Any
    for entry in manifest.get("models", ()):
        key: Tuple[str, str] = (entry["model_id"], entry["model_version"])
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
    result: Tuple[str, ...] = tuple(sorted(errors))
    return result


__all__: list[str] = [
    "packaged_model_card",
    "registry_manifest",
    "render_model_card",
    "validate_registry_manifest",
]
