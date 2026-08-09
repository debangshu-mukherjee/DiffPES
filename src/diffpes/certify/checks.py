"""Register pure JAX scientific certification checks.

Extended Summary
----------------
This module connects stable predicate identities to pure functions. The
compiled certification kernel evaluates these functions. Registration selects
programs statically. Every check returns a ``DomainResult`` with a continuous
margin and a Boolean outcome.

Routine Listings
----------------
:func:`get_check`
    Resolve a registered JAX certification check.
:func:`list_checks`
    List registered checks in deterministic identity order.
:func:`register_check`
    Register a stable predicate identity and pure JAX callable.
"""

from functools import cache

from beartype import beartype
from beartype.typing import Tuple
from jaxtyping import jaxtyped

from diffpes.types import CheckFunction


@cache
def _check_registry() -> dict[str, CheckFunction]:
    """PRIVATE: Return process-local pure-JAX check bindings.

    Returns
    -------
    registry : dict[str, CheckFunction]
        The single mutable mapping from stable predicate identifiers to
        registered pure JAX predicates.

    Notes
    -----
    The :func:`functools.cache` decorator makes the empty dictionary a
    process-local singleton. :func:`register_check` mutates the returned
    mapping in place; :func:`get_check` and :func:`list_checks` read it.
    """
    registry: dict[str, CheckFunction] = {}
    return registry


@jaxtyped(typechecker=beartype)
def register_check(check_id: str, check_fn: CheckFunction) -> None:
    """Register a stable predicate identity and pure JAX callable.

    The process-local registry selects pure JAX predicates by stable scientific
    identity. Registered functions retain continuous margins in compiled
    execution.

    :see: :class:`~.test_checks.TestRegisterCheck`

    Parameters
    ----------
    check_id : str
        Stable predicate identifier (**static** -- a compile-time selection).
    check_fn : CheckFunction
        Pure JAX predicate implementation (**static** -- a compiled program).

    Raises
    ------
    ValueError
        If the identity is unstable or already registered.

    Notes
    -----
    Registration occurs at the eager application boundary. The registered
    callable itself remains pure and traceable inside certification kernels.
    """
    if not check_id or not check_id.startswith("org.diffpes."):
        msg: str = "check_id must be a stable org.diffpes identifier"
        raise ValueError(msg)
    checks: dict[str, CheckFunction] = _check_registry()
    if check_id in checks:
        msg: str = f"duplicate certification check: {check_id}"
        raise ValueError(msg)
    checks[check_id] = check_fn


@jaxtyped(typechecker=beartype)
def get_check(check_id: str) -> CheckFunction:
    """Resolve a registered JAX certification check.

    The process-local registry selects pure JAX predicates by stable scientific
    identity. Registered functions retain continuous margins in compiled
    execution.

    :see: :class:`~.test_checks.TestGetCheck`

    Implementation Logic
    --------------------
    1. **Bind the documented output**::

           check: CheckFunction = _check_registry()[check_id]

       The function validates and transforms the inputs before it binds the
       documented output.

    Parameters
    ----------
    check_id : str
        Exact predicate identifier (**static** -- a compile-time selection).

    Returns
    -------
    check : CheckFunction
        Registered pure JAX predicate implementation.

    Raises
    ------
    KeyError
        If no check has the requested identity.
    """
    exc: KeyError
    try:
        check: CheckFunction = _check_registry()[check_id]
    except KeyError as exc:
        raise KeyError(f"unknown certification check: {check_id}") from exc
    return check


@jaxtyped(typechecker=beartype)
def list_checks() -> Tuple[str, ...]:
    """List registered checks in deterministic identity order.

    The process-local registry selects pure JAX predicates by stable scientific
    identity. Registered functions retain continuous margins in compiled
    execution.

    :see: :class:`~.test_checks.TestListChecks`

    Implementation Logic
    --------------------
    1. **Bind the documented output**::

           checks: tuple[str, ...] = tuple(sorted(_check_registry()))

       The function validates and transforms the inputs before it binds the
       documented output.

    Returns
    -------
    checks : Tuple[str, ...]
        Sorted stable predicate identifiers.
    """
    checks: Tuple[str, ...] = tuple(sorted(_check_registry()))
    return checks


__all__: list[str] = [
    "get_check",
    "list_checks",
    "register_check",
]
