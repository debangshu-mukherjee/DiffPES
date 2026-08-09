"""Provide the shared pytest runtime foundation.

Extended Summary
----------------
The autouse fixtures in this module enforce the numerical and resource
contracts shared by the complete test suite. They verify JAX 64-bit mode,
clear compilation caches between tests, and reject excessive retained memory.
The collection hook serializes memory-intensive tests under pytest-xdist, and
the random-key fixture supplies stable, worker-independent test randomness.
"""

import hashlib
import os
import subprocess
import sys
from pathlib import Path
from importlib.abc import MetaPathFinder
from importlib.machinery import ModuleSpec
from types import ModuleType

# ruff: noqa: I001 -- diffpes must configure JAX before JAX is imported.
import diffpes  # noqa: F401 -- Import activates the package-wide x64 contract.

import jax
import jax.numpy as jnp
import psutil
import pytest
from beartype import beartype
from beartype.typing import Iterator, Tuple
from jaxtyping import PRNGKeyArray, jaxtyped

pytest_plugins: Tuple[str, ...] = ("pytester",)

_PREFLIGHT_SKIP_VARIABLE: str = "DIFFPES_SKIP_PREFLIGHT"
RSS_LEAK_LIMIT_MB: float = 500.0
_BYTES_PER_MEBIBYTE: int = 1024**2


class _ChinookImportBlocker(MetaPathFinder):
    """Reject Chinook imports throughout pytest collection and execution."""

    def find_spec(
        self,
        fullname: str,
        path: object | None,
        target: ModuleType | None = None,
    ) -> ModuleSpec | None:
        """Raise for the forbidden Chinook module namespace."""
        del path, target
        if fullname == "chinook" or fullname.startswith("chinook."):
            raise RuntimeError(
                "DiffPES tests must consume frozen Chinook artifacts; "
                f"importing {fullname!r} is forbidden"
            )
        return None


_CHINOOK_IMPORT_BLOCKER = _ChinookImportBlocker()


def _run_annotation_preflight(config: pytest.Config) -> None:
    """PRIVATE: Reject invalid annotations before pytest collects test modules.

    The gate runs ``tests/_preflight_types.py`` in a subprocess. A subprocess
    keeps the jaxtyping import hook out of this process. An in-process run
    leaves decorated modules in ``sys.modules``. Pytest then collects wrapped
    fixtures.

    The gate runs once for each session. It does not run on a pytest-xdist
    worker, and it does not run inside a nested ``pytester`` session.

    Parameters
    ----------
    config : pytest.Config
        Active pytest configuration. A worker configuration carries
        ``workerinput``.

    Raises
    ------
    pytest.UsageError
        If one module or more carries an invalid annotation.
    """
    if hasattr(config, "workerinput"):
        return
    if os.environ.get(_PREFLIGHT_SKIP_VARIABLE):
        return
    os.environ[_PREFLIGHT_SKIP_VARIABLE] = "1"
    script: Path = Path(__file__).resolve().parent / "_preflight_types.py"
    if not script.is_file():
        return
    completed: subprocess.CompletedProcess[str] = subprocess.run(
        [sys.executable, str(script)],
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        message: str = (
            "annotation pre-flight failed; fix these before running tests:\n"
            + completed.stdout
            + completed.stderr
        )
        raise pytest.UsageError(message)


def pytest_configure(config: pytest.Config) -> None:
    """Install the test-suite firewall before pytest collects test modules."""
    _run_annotation_preflight(config)
    imported: Tuple[str, ...] = tuple(
        sorted(
            name
            for name in sys.modules
            if name == "chinook" or name.startswith("chinook.")
        )
    )
    if imported:
        raise RuntimeError(
            "Chinook was imported before DiffPES test collection: "
            + ", ".join(imported)
        )
    if _CHINOOK_IMPORT_BLOCKER not in sys.meta_path:
        sys.meta_path.insert(0, _CHINOOK_IMPORT_BLOCKER)


def pytest_unconfigure(config: pytest.Config) -> None:
    """Remove the test-suite firewall when pytest shuts down."""
    del config
    if _CHINOOK_IMPORT_BLOCKER in sys.meta_path:
        sys.meta_path.remove(_CHINOOK_IMPORT_BLOCKER)


@pytest.fixture(autouse=True, scope="session")
def assert_x64_enabled() -> Iterator[None]:
    """Require JAX 64-bit precision for the complete test session.

    Yields
    ------
    None
        Control to the test session after validating the default JAX dtype.

    Notes
    -----
    Asserts that importing diffpes selected ``jax.numpy.float64`` as the
    default scalar dtype before the suite begins.
    """
    actual_dtype: jnp.dtype = jnp.zeros(()).dtype
    assert actual_dtype == jnp.float64, (
        "diffpes tests require JAX 64-bit mode; "
        f"the default scalar dtype is {actual_dtype}."
    )
    yield


@pytest.fixture(autouse=True)
def rss_leak_guard(request: pytest.FixtureRequest) -> Iterator[None]:
    """Reject process memory growth beyond the test-specific RSS limit.

    The default limit is 500 MiB. Use
    ``@pytest.mark.rss_limit_mb(limit)`` for a different expected allocation.
    The fixture clears JAX caches before the baseline and final measurement.
    Thus, compiled executables do not hide application-level leaks.

    Parameters
    ----------
    request : pytest.FixtureRequest
        Pytest request carrying the current test node and its markers.

    Yields
    ------
    None
        Control to the test after recording its baseline RSS.
    """
    limit_marker: pytest.Mark | None = request.node.get_closest_marker(
        "rss_limit_mb"
    )
    limit_mb: float = RSS_LEAK_LIMIT_MB
    if limit_marker is not None:
        if len(limit_marker.args) != 1:
            pytest.fail("rss_limit_mb requires exactly one numeric argument")
        limit_mb = float(limit_marker.args[0])

    process: psutil.Process = psutil.Process()
    jax.clear_caches()
    rss_before: int = process.memory_info().rss
    yield
    jax.clear_caches()
    rss_after: int = process.memory_info().rss
    growth_mb: float = (rss_after - rss_before) / _BYTES_PER_MEBIBYTE
    assert growth_mb <= limit_mb, (
        f"{request.node.nodeid} retained {growth_mb:.1f} MiB RSS; "
        f"limit is {limit_mb:.1f} MiB"
    )


@pytest.fixture(autouse=True)
def clear_jax_caches(rss_leak_guard: None) -> Iterator[None]:
    """Clear JAX caches after every test and before its RSS measurement.

    Parameters
    ----------
    rss_leak_guard : None
        Dependency that orders cache clearing before the guard's teardown.

    Yields
    ------
    None
        Control to the test before clearing all JAX compilation caches.
    """
    yield
    jax.clear_caches()


@jaxtyped(typechecker=beartype)
def pytest_collection_modifyitems(
    config: pytest.Config,
    items: list[pytest.Item],
) -> None:
    """Serialize memory-intensive tests on one pytest-xdist worker.

    Parameters
    ----------
    config : pytest.Config
        Active pytest configuration. Required by the pytest hook contract.
    items : list[pytest.Item]
        Collected test items to augment with xdist grouping metadata.
    """
    del config
    xdist_group: pytest.MarkDecorator = pytest.mark.xdist_group("big_mem")
    item: pytest.Item
    for item in items:
        if item.get_closest_marker("big_mem") is not None:
            item.add_marker(xdist_group)


@pytest.fixture
@jaxtyped(typechecker=beartype)
def rng_key(request: pytest.FixtureRequest) -> PRNGKeyArray:
    """Create a deterministic random key from the current test node ID.

    Hashing the fully qualified node ID makes the seed stable across Python
    processes, repeated runs, and xdist workers while keeping distinct tests
    statistically independent.

    Parameters
    ----------
    request : pytest.FixtureRequest
        Pytest request identifying the current test node.

    Returns
    -------
    PRNGKeyArray
        Typed JAX random key derived from the first 32 hash bits.
    """
    digest: bytes = hashlib.sha256(request.node.nodeid.encode()).digest()
    seed: int = int.from_bytes(digest[:4], byteorder="big")
    key: PRNGKeyArray = jax.random.key(seed)
    return key
