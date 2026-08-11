"""Validate host-side certification state carriers and factories.

The tests cover empty construction, independent synchronization objects, and
the in-place collection mutations used by certification services.
"""

from diffpes.types import (
    CertificationRegistryState,
    DependencyAnalysisCache,
    make_certification_registry_state,
    make_dependency_analysis_cache,
)


class TestCertificationRegistryState:
    """Validate :class:`~diffpes.types.CertificationRegistryState` storage.

    The carrier must expose empty immutable registry collections and an open
    host-side mutation state.

    :see: :class:`~diffpes.types.CertificationRegistryState`
    """

    def test_stores_empty_registry_collections(self) -> None:
        """Preserve the empty registry collections and open state.

        The check covers every scientific entry collection before the first
        registration.

        Notes
        -----
        The test constructs the carrier through its public factory and reads
        each field while it holds the carrier's reentrant lock.
        """
        state: CertificationRegistryState = make_certification_registry_state()

        with state.lock:
            assert state.models == ()
            assert state.transformations == ()
            assert state.handshakes == ()
            assert state.frozen is False


class TestDependencyAnalysisCache:
    """Validate :class:`~diffpes.types.DependencyAnalysisCache` storage.

    The carrier must expose one mutable entry dictionary and exact integer
    access counters.

    :see: :class:`~diffpes.types.DependencyAnalysisCache`
    """

    def test_stores_empty_entries_and_zero_counters(self) -> None:
        """Preserve the empty cache and its zero access counters.

        The check covers every cache field before the first structural
        dependency analysis.

        Notes
        -----
        The test constructs the carrier through its public factory and reads
        its dictionary and counters while it holds the reentrant lock.
        """
        cache: DependencyAnalysisCache = make_dependency_analysis_cache()

        with cache.lock:
            assert cache.entries == {}
            assert cache.hits == 0
            assert cache.misses == 0


class TestMakeCertificationRegistryState:
    """Validate :func:`~diffpes.types.make_certification_registry_state`.

    The factory must allocate independent state and synchronization objects
    for separate registry owners.

    :see: :func:`~diffpes.types.make_certification_registry_state`
    """

    def test_creates_independent_registry_state(self) -> None:
        """Create independent registry carriers and reentrant locks.

        The check prevents one factory call from sharing mutable host state
        with another call.

        Notes
        -----
        The test compares the identity of two carriers and their lock fields
        after two consecutive public factory calls.
        """
        first: CertificationRegistryState = make_certification_registry_state()
        second: CertificationRegistryState = (
            make_certification_registry_state()
        )

        assert first is not second
        assert first.lock is not second.lock


class TestMakeDependencyAnalysisCache:
    """Validate :func:`~diffpes.types.make_dependency_analysis_cache`.

    The factory must allocate independent dictionaries and synchronization
    objects for separate cache owners.

    :see: :func:`~diffpes.types.make_dependency_analysis_cache`
    """

    def test_creates_independent_cache_state(self) -> None:
        """Create independent dependency caches and reentrant locks.

        The check prevents one factory call from sharing mutable host state
        with another call.

        Notes
        -----
        The test compares the identity of two dictionaries and locks after
        two consecutive public factory calls.
        """
        first: DependencyAnalysisCache = make_dependency_analysis_cache()
        second: DependencyAnalysisCache = make_dependency_analysis_cache()

        assert first.entries is not second.entries
        assert first.lock is not second.lock
