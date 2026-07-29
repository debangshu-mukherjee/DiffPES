"""Provide shared assertions for numerical test trees.

Extended Summary
----------------
This module wraps Chex's PyTree assertions with the strict test defaults.
It loads behavioral regression arrays from the WP6.1 reference directory.
It also provides the shared eager and JIT rejection contract for factory tests.
"""

import re
from pathlib import Path

import chex
import equinox as eqx
import jax
import numpy as np
from beartype import beartype
from beartype.typing import Any, Callable
from jaxtyping import PyTree, jaxtyped

_REFERENCE_DIRECTORY: Path = (
    Path(__file__).parent / "test_diffpes" / "_reference_data"
)


def _assert_rejection(
    fn: Callable[..., Any],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    match: str,
) -> None:
    """Assert one callable invocation raises the expected validation error."""
    error: ValueError | RuntimeError
    try:
        fn(*args, **kwargs)
    except (ValueError, RuntimeError) as error:
        if re.search(match, str(error)) is None:
            message: str = (
                f"error message {str(error)!r} does not match {match!r}"
            )
            raise AssertionError(message) from error
    else:
        message = "DID NOT RAISE ValueError or RuntimeError"
        raise AssertionError(message)


@jaxtyped(typechecker=beartype)
def assert_rejects(
    fn: Callable[..., Any],
    *args: Any,
    match: str,
    under_jit: bool = True,
    **kwargs: Any,
) -> None:
    """Assert a factory rejects the same invalid input eagerly and under JIT.

    :see: :class:`~.test_assertions.TestAssertRejects`

    Parameters
    ----------
    fn : Callable[..., Any]
        Factory or other validation callable expected to reject its inputs.
    *args : Any
        Positional arguments forwarded to ``fn``.
    match : str
        Regular expression that must match the eager and JIT error messages.
    under_jit : bool, optional
        Whether to repeat the assertion through :func:`equinox.filter_jit`.
        Default is ``True``.
    **kwargs : Any
        Keyword arguments forwarded to ``fn``.

    Notes
    -----
    The delegated eager and compiled checks raise ``AssertionError`` when the
    callable accepts the invalid input or its message does not match.
    """
    _assert_rejection(fn, args, kwargs, match)
    if under_jit:
        jitted_fn: Callable[..., Any] = eqx.filter_jit(fn)
        _assert_rejection(jitted_fn, args, kwargs, match)


@jaxtyped(typechecker=beartype)
def assert_trees_close(
    actual: PyTree,
    desired: PyTree,
    *,
    rtol: float = 1e-12,
    atol: float = 0.0,
) -> None:
    """Assert that corresponding numerical PyTree leaves are close.

    The helper uses a relative tolerance of ``1e-12`` and zero absolute
    tolerance by default. These settings preserve sensitivity near zero.
    They also permit small variation from the order of floating-point reductions.
    """
    chex.assert_trees_all_close(actual, desired, rtol=rtol, atol=atol)


@jaxtyped(typechecker=beartype)
def assert_tree_finite(tree: PyTree) -> None:
    """Assert that every numerical leaf in a PyTree is finite.

    Delegates recursively to Chex so nested production carriers and ordinary
    array collections share one NaN and infinity gate.
    """
    leaves: tuple[object, ...] = tuple(jax.tree.leaves(tree))
    chex.assert_tree_all_finite(leaves)


@jaxtyped(typechecker=beartype)
def assert_matches_reference(
    tree: PyTree,
    name: str,
    *,
    rtol: float = 1e-12,
) -> None:
    """Compare numerical leaves with a named deterministic NPZ reference.

    Loads ``tests/test_diffpes/_reference_data/<name>.npz`` and compares the
    archive arrays in stored order with the numerical leaves of ``tree``.
    ``name`` must be a bare filename stem, preventing paths outside the
    pinned reference directory.

    Parameters
    ----------
    tree : PyTree
        Numerical production carrier or nested collection to compare.
    name : str
        Bare filename stem of the NPZ artifact.
    rtol : float, optional
        Relative comparison tolerance. Default is ``1e-12``.
    Raises
    ------
    ValueError
        If ``name`` is not a bare filename stem or the leaf count differs from
        the archive array count.
    """
    if Path(name).name != name or Path(name).suffix:
        message: str = "name must be a bare reference filename stem"
        raise ValueError(message)
    reference_path: Path = _REFERENCE_DIRECTORY / f"{name}.npz"
    archive: Any
    with np.load(reference_path, allow_pickle=False) as archive:
        desired_leaves: tuple[np.ndarray, ...] = tuple(
            archive[key] for key in archive.files
        )
    actual_leaves: tuple[object, ...] = tuple(jax.tree.leaves(tree))
    if len(actual_leaves) != len(desired_leaves):
        message = (
            "reference leaf count differs: "
            f"actual={len(actual_leaves)}, desired={len(desired_leaves)}"
        )
        raise ValueError(message)
    assert_trees_close(actual_leaves, desired_leaves, rtol=rtol)
