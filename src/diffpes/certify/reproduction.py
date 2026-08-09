"""Reproduce a certified forward result from resolved artifacts.

Extended Summary
----------------
The runner resolves normalized inputs and the recorded result at the eager I/O
boundary. It executes the exact registered model and compares the resulting
PyTree numerically. A successful report is resolver-backed evidence; a stored
checksum alone is not a reproduction claim.

Routine Listings
----------------
:func:`reproduce_forward`
    Re-execute a registered model and compare its recorded result.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from beartype import beartype
from beartype.typing import Any, Tuple
from jax.flatten_util import ravel_pytree
from jaxtyping import jaxtyped

from diffpes.types import (
    ArtifactRef,
    ArtifactResolver,
    ForwardCertificate,
    ReproductionReport,
    make_reproduction_report,
)

from .checksums import checksum_pytree
from .registry import get_model
from .resolvers import resolve_artifact


def _unique_role(
    artifacts: Tuple[ArtifactRef, ...],
    roles: frozenset[str],
    label: str,
) -> ArtifactRef:
    """PRIVATE: Return the single artifact whose role is in ``roles``.

    Parameters
    ----------
    artifacts : Tuple[ArtifactRef, ...]
        Certificate artifact references.
    roles : frozenset[str]
        Accepted role labels for one reproduction slot.
    label : str
        Human-readable slot name for the error message.

    Returns
    -------
    result : ArtifactRef
        The one artifact whose role lies in ``roles``.

    Raises
    ------
    ValueError
        If no artifact or more than one artifact carries an accepted
        role.

    Notes
    -----
    Reproduction needs exactly one normalized-input artifact and one
    recorded-result artifact; an ambiguous certificate must fail before
    any resolver I/O runs.
    """
    matches: Tuple[ArtifactRef, ...] = tuple(
        artifact for artifact in artifacts if artifact.role in roles
    )
    if len(matches) != 1:
        msg: str = f"certificate requires exactly one {label} artifact"
        raise ValueError(msg)
    result: ArtifactRef = matches[0]
    return result


@jaxtyped(typechecker=beartype)
def reproduce_forward(
    certificate: ForwardCertificate,
    *,
    resolver: ArtifactResolver,
    tolerance: float = 1e-10,
) -> ReproductionReport:
    """Re-execute a registered model and compare its recorded result.

    The runner resolves one normalized input and one separately stored result.

    :see: :class:`~.test_reproduction.TestReproduceForward`

    Implementation Logic
    --------------------
    1. **Resolve and re-execute**::

           reproduced_value = registered.executor(inputs)

       Exact registry lookup binds the rerun to the certificate model.
    2. **Compare numerical leaves**::

           absolute = jnp.abs(actual_flat - expected_flat)

       The report stores maximum absolute and relative errors.

    Parameters
    ----------
    certificate : ForwardCertificate
        Certificate with one normalized-input and one result artifact.
    resolver : ArtifactResolver
        Eager resolver for both normalized artifacts.
    tolerance : float
        Nonnegative absolute and relative comparison tolerance. Default 1e-10.

    Returns
    -------
    report : ReproductionReport
        Numerical comparison and reproduced result identity.

    Raises
    ------
    ValueError
        If required artifact roles or result tree structure are inconsistent.

    Notes
    -----
    The comparison uses ``tolerance`` for both the absolute and relative terms.
    Each leaf must satisfy ``absolute <= tolerance * (1 + abs(expected))``.
    """
    if tolerance < 0.0:
        msg: str = "tolerance must be nonnegative"
        raise ValueError(msg)
    input_roles: frozenset[str] = frozenset(
        ("model-input", "normalized-input")
    )
    result_roles: frozenset[str] = frozenset(("forward-result", "result"))
    input_ref: ArtifactRef = _unique_role(
        certificate.artifacts,
        input_roles,
        "normalized-input",
    )
    result_ref: ArtifactRef = _unique_role(
        certificate.artifacts,
        result_roles,
        "result",
    )
    inputs: Any = resolve_artifact(input_ref, resolver)
    expected: Any = resolve_artifact(result_ref, resolver)
    registered: Any = get_model(
        certificate.model.model_id,
        certificate.model.model_version,
    )
    reproduced_value: Any = registered.executor(inputs)
    if jax.tree.structure(reproduced_value) != jax.tree.structure(expected):
        msg = "reproduced result tree structure does not match"
        raise ValueError(msg)
    actual_flat: Any = ravel_pytree(reproduced_value)[0]
    expected_flat: Any = ravel_pytree(expected)[0]
    absolute: Any = jnp.abs(actual_flat - expected_flat)
    denominator: Any = jnp.maximum(
        jnp.abs(expected_flat),
        jnp.finfo(absolute.dtype).tiny,
    )
    relative: Any = absolute / denominator
    max_abs: Any = jnp.max(absolute)
    max_rel: Any = jnp.max(relative)
    allowed: Any = tolerance + tolerance * jnp.abs(expected_flat)
    reproduced: Any = jnp.all(absolute <= allowed)
    result_identity: str = checksum_pytree(
        reproduced_value,
        record_kind="normalized-content",
    )
    report: ReproductionReport = make_reproduction_report(
        execution_id=certificate.manifest.execution_id,
        result_checksum=result_identity,
        reproduced=reproduced,
        max_abs_error=max_abs,
        max_rel_error=max_rel,
        tolerance=jnp.asarray(tolerance),
    )
    return report


__all__: list[str] = ["reproduce_forward"]
