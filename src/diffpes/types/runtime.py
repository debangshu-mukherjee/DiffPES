"""Store mutable host-side state for certification services.

Extended Summary
----------------
The carriers in this module own the process-local cache and registry state
used by ``diffpes.certify``. They remain outside compiled numerical kernels
and retain their existing host-side mutation behavior.

Routine Listings
----------------
:class:`CertificationRegistryState`
    Store mutable entries for the process-local certification registry.
:class:`DependencyAnalysisCache`
    Store cached structural dependency analyses and access counters.
:func:`make_certification_registry_state`
    Create empty mutable state for the certification registry.
:func:`make_dependency_analysis_cache`
    Create an empty mutable cache for dependency analyses.

Notes
-----
Every field is static because these carriers coordinate host execution. The
certification services synchronize mutations with each carrier's reentrant
lock and never pass the carriers through JAX transformations.
"""

import threading

import equinox as eqx
from beartype import beartype
from beartype.typing import Any, Dict, Tuple
from jaxtyping import Array, Bool, PyTree, jaxtyped

from .certification import (
    RegisteredModel,
    RegisteredTransformation,
    RegistrationHandshake,
)

type _DependencyCacheEntries = Dict[
    Tuple[Any, ...],
    Tuple[PyTree, Bool[Array, "n_output n_input"]],
]


class CertificationRegistryState(eqx.Module):
    """Store mutable entries for the process-local certification registry.

    The state collects registered models, transformations, and owner
    handshakes behind one reentrant lock. The registry service replaces the
    immutable tuples while it holds that lock.

    :see: :class:`~.test_runtime.TestCertificationRegistryState`

    Attributes
    ----------
    models : Tuple[RegisteredModel, ...]
        Registered model bindings in deterministic order (**static** -- host
        registry state that does not enter compiled kernels).
    transformations : Tuple[RegisteredTransformation, ...]
        Registered transformation bindings in deterministic order
        (**static** -- host registry state that does not enter compiled
        kernels).
    handshakes : Tuple[RegistrationHandshake, ...]
        Registered owner handshakes in deterministic order (**static** --
        host registry state that does not enter compiled kernels).
    frozen : bool
        Registry flag that forbids later registration (**static** -- host
        registry state that does not enter compiled kernels).
    lock : Any
        Reentrant lock that synchronizes every registry read and mutation
        (**static** -- host synchronization state).

    Notes
    -----
    Equinox makes module attributes immutable through normal assignment. The
    registry service uses ``object.__setattr__`` while it holds ``lock`` to
    retain the original process-local mutation semantics.

    See Also
    --------
    make_certification_registry_state : Create empty mutable state for the
        certification registry.
    """

    models: Tuple[RegisteredModel, ...] = eqx.field(static=True)
    transformations: Tuple[RegisteredTransformation, ...] = eqx.field(
        static=True
    )
    handshakes: Tuple[RegistrationHandshake, ...] = eqx.field(static=True)
    frozen: bool = eqx.field(static=True)
    lock: Any = eqx.field(static=True)


class DependencyAnalysisCache(eqx.Module):
    """Store cached structural dependency analyses and access counters.

    The cache maps one static model and input signature to its abstract
    output and Boolean dependency matrix. It counts eager cache hits and
    misses behind one reentrant lock.

    :see: :class:`~.test_runtime.TestDependencyAnalysisCache`

    Attributes
    ----------
    entries : _DependencyCacheEntries
        Cached abstract outputs and output-by-input dependency matrices
        (**static** -- host cache state that does not enter compiled kernels).
    hits : int
        Number of successful eager cache lookups (**static** -- host cache
        state that does not enter compiled kernels).
    misses : int
        Number of structural analyses inserted into the cache (**static** --
        host cache state that does not enter compiled kernels).
    lock : Any
        Reentrant lock that synchronizes every cache read and mutation
        (**static** -- host synchronization state).

    Notes
    -----
    Equinox makes module attributes immutable through normal assignment. The
    dependency service uses ``object.__setattr__`` while it holds ``lock`` to
    retain the original process-local mutation semantics.

    See Also
    --------
    make_dependency_analysis_cache : Create an empty mutable cache for
        dependency analyses.
    """

    entries: _DependencyCacheEntries = eqx.field(static=True)
    hits: int = eqx.field(static=True)
    misses: int = eqx.field(static=True)
    lock: Any = eqx.field(static=True)


@jaxtyped(typechecker=beartype)
def make_certification_registry_state() -> CertificationRegistryState:
    """Create empty mutable state for the certification registry.

    Construct one host-side registry holder with empty immutable entry tuples,
    an open registration state, and a new reentrant lock.

    :see: :class:`~.test_runtime.TestMakeCertificationRegistryState`

    Returns
    -------
    state : CertificationRegistryState
        Empty process-local registry state with an independent lock.

    Notes
    -----
    The factory creates a fresh lock and does not install a singleton. The
    consuming service owns singleton lifetime through ``functools.cache``.
    """
    lock: Any = threading.RLock()
    state: CertificationRegistryState = CertificationRegistryState(
        models=(),
        transformations=(),
        handshakes=(),
        frozen=False,
        lock=lock,
    )
    return state


@jaxtyped(typechecker=beartype)
def make_dependency_analysis_cache() -> DependencyAnalysisCache:
    """Create an empty mutable cache for dependency analyses.

    Construct one host-side cache with no entries, zero counters, and a new
    reentrant lock.

    :see: :class:`~.test_runtime.TestMakeDependencyAnalysisCache`

    Returns
    -------
    cache : DependencyAnalysisCache
        Empty process-local dependency cache with an independent lock.

    Notes
    -----
    The factory creates a fresh dictionary and lock and does not install a
    singleton. The consuming service owns singleton lifetime through
    ``functools.cache``.
    """
    entries: _DependencyCacheEntries = {}
    lock: Any = threading.RLock()
    cache: DependencyAnalysisCache = DependencyAnalysisCache(
        entries=entries,
        hits=0,
        misses=0,
        lock=lock,
    )
    return cache


__all__: list[str] = [
    "CertificationRegistryState",
    "DependencyAnalysisCache",
    "make_certification_registry_state",
    "make_dependency_analysis_cache",
]
