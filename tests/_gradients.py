"""Provide the program-wide differentiability gate.

Extended Summary
----------------
Every differentiability gate in the diffpes plan series calls
``gradient_gate`` or ``assert_grad_matches_fd`` from this module. Ad-hoc finite
differences in a gate are a review-blocking defect. A harness failure on a
physics function indicates a physics failure. The certified gradients
construct the Fisher matrix. A false zero in the parameter Jacobian removes
the applicable Fisher row and column.

The harness combines JAX's randomized directional checks with scale-aware,
elementwise central differences and explicit zero-gradient tripwires. Complex
leaves follow JAX's complex-to-real Wirtinger convention.

Notes
-----
The finite-difference scaling follows Nocedal and Wright, *Numerical
Optimization*, section 8.1. The complex-gradient convention follows Martins
et al. (2003) and the JAX advanced-autodiff cookbook.
"""

import jax
import jax.numpy as jnp
from beartype import beartype
from beartype.typing import Any, Callable, Dict, Optional, Tuple
from jax import test_util
from jaxtyping import (
    Array,
    Complex128,
    Float64,
    PRNGKeyArray,
    PyTree,
    jaxtyped,
)

from diffpes.types import PyTreeDef, ScalarFloat
from tests._assertions import assert_tree_finite
from tests._types import GradRegime, ScalarLoss

RTOL_LADDER: Dict[GradRegime, float] = {
    "smooth": 1e-6,
    "stiff": 1e-5,
    "singular": 1e-4,
}
EPS_F64: float = 2.220446049250313e-16
FD_ROUNDOFF_FACTOR: float = 1.0


@jaxtyped(typechecker=beartype)
def fd_step(
    theta: Float64[Array, "..."], *, scale_floor: ScalarFloat = 1e-3
) -> Float64[Array, "..."]:
    """Calculate a scale-aware central-finite-difference step.

    Uses ``EPS_F64**(1/3) * maximum(abs(theta), scale_floor)`` elementwise,
    which balances truncation and round-off error for central differences.
    """
    step: Float64[Array, "..."] = EPS_F64 ** (1.0 / 3.0) * jnp.maximum(
        jnp.abs(theta), scale_floor
    )
    return step


def _path_name(path: Tuple[object, ...]) -> str:
    """PRIVATE: Render a JAX key path in the stable tree-path notation.

    Parameters
    ----------
    path : tuple[object, ...]
        Key-path entries from JAX tree traversal.

    Returns
    -------
    path_name : str
        Stable bracketed path string from ``jax.tree_util.keystr``.

    Notes
    -----
    ``jax.tree_util.keystr`` formats every entry in traversal order,
    so error messages name the exact failing leaf.
    """
    path_name: str = jax.tree_util.keystr(path)
    return path_name


def _as_jax_arrays(tree: PyTree) -> PyTree:
    """PRIVATE: Normalize numerical-check inputs before type validation.

    Parameters
    ----------
    tree : PyTree
        Arbitrary tree of array-like numerical leaves.

    Returns
    -------
    normalized : PyTree
        The same tree with every leaf as a JAX array.

    Notes
    -----
    ``jax.tree.map`` applies ``jnp.asarray`` to every leaf, which
    gives later checks one uniform array type.
    """
    normalized: PyTree = jax.tree.map(jnp.asarray, tree)
    return normalized


def _central_leaf_grad(
    jitted_fn: ScalarLoss,
    treedef: PyTreeDef,
    leaves: list[Array],
    leaf_index: int,
    scale_floor: ScalarFloat,
) -> Array:
    """PRIVATE: Differentiate one leaf by elementwise central differences.

    Parameters
    ----------
    jitted_fn : ScalarLoss
        Jitted scalar loss over the full tree.
    treedef : PyTreeDef
        Tree definition for reassembly of perturbed leaves.
    leaves : list[Array]
        All numerical leaves in flattening order.
    leaf_index : int
        Index of the differentiated leaf.
    scale_floor : ScalarFloat
        Lower bound for the finite-difference scale.

    Returns
    -------
    gradient : Array
        Gradient with the leaf shape; complex leaves follow the
        ``d/dRe - 1j*d/dIm`` convention.

    Implementation Logic
    --------------------
    The helper perturbs one flattened component at a time along an
    identity basis and evaluates symmetric differences under
    ``jax.vmap``. Real leaves need one diagonal; complex leaves also
    perturb the imaginary component and combine both diagonals as
    ``d/dRe - 1j*d/dIm``. Step sizes come from :func:`fd_step` on the
    real part.
    """
    leaf: Array = jnp.asarray(leaves[leaf_index])
    steps: Array = fd_step(jnp.real(leaf), scale_floor=scale_floor)
    flat_leaf: Array = jnp.ravel(leaf)
    flat_steps: Array = jnp.ravel(steps)
    basis: Array = jnp.eye(flat_leaf.size, dtype=flat_leaf.dtype)

    def evaluate(delta: Array) -> Float64[Array, ""]:
        perturbed_leaves: list[Array] = list(leaves)
        perturbed_leaves[leaf_index] = jnp.reshape(
            flat_leaf + delta, leaf.shape
        )
        perturbed_tree: PyTree = jax.tree_util.tree_unflatten(
            treedef, perturbed_leaves
        )
        value: Float64[Array, ""] = jitted_fn(perturbed_tree)
        return value

    real_deltas: Array = basis * flat_steps[:, None]
    real_gradient: Array = jax.vmap(
        lambda delta: (evaluate(delta) - evaluate(-delta)) / (2.0 * flat_steps)
    )(real_deltas)
    real_diagonal: Array = jnp.diag(real_gradient)
    if not jnp.issubdtype(leaf.dtype, jnp.complexfloating):
        gradient: Array = jnp.reshape(real_diagonal, leaf.shape)
        return gradient

    imaginary_deltas: Array = 1j * basis * flat_steps[:, None]
    imaginary_gradient: Array = jax.vmap(
        lambda delta: (evaluate(delta) - evaluate(-delta)) / (2.0 * flat_steps)
    )(imaginary_deltas)
    imaginary_diagonal: Array = jnp.diag(imaginary_gradient)
    gradient = jnp.reshape(real_diagonal - 1j * imaginary_diagonal, leaf.shape)
    return gradient


@jaxtyped(typechecker=beartype)
def central_fd_grad(
    fn: ScalarLoss, theta: PyTree, *, scale_floor: ScalarFloat = 1e-3
) -> PyTree:
    """Calculate an elementwise central-FD gradient over a numerical PyTree.

    Real leaves use symmetric perturbations with :func:`fd_step`. Complex
    leaves separately perturb real and imaginary components and combine them
    as ``d/dRe - 1j*d/dIm``, matching JAX's complex-to-real convention. The
    cost is two forward evaluations per real parameter and four per complex
    parameter. This cost restricts the helper to toy-model gates.
    """
    leaves: list[Any]
    treedef: PyTreeDef
    leaves, treedef = jax.tree_util.tree_flatten(theta)
    array_leaves: list[Array] = [jnp.asarray(leaf) for leaf in leaves]
    jitted_fn: ScalarLoss = jax.jit(fn)
    gradient_leaves: list[Array] = [
        _central_leaf_grad(
            jitted_fn, treedef, array_leaves, index, scale_floor
        )
        for index in range(len(array_leaves))
    ]
    gradient: PyTree = jax.tree_util.tree_unflatten(treedef, gradient_leaves)
    return gradient


@jaxtyped(typechecker=beartype)
def assert_grad_matches_fd(
    fn: ScalarLoss,
    theta: PyTree,
    *,
    regime: GradRegime = "smooth",
    atol: Optional[ScalarFloat] = None,
    directional_atol: Optional[ScalarFloat] = None,
    scale_floor: ScalarFloat = 1e-3,
    modes: Tuple[str, ...] = ("fwd", "rev"),
) -> None:
    """Assert autodiff agrees with directional and elementwise FD checks.

    :data:`RTOL_LADDER` selects the relative tolerance. If the caller omits
    ``atol``, each parameter uses the central-FD round-off bound
    ``EPS_F64 * max(1, abs(fn(theta))) / h_i``. Its units are loss per
    parameter, matching a gradient; the relative term covers the
    ``O(h_i**2)`` truncation error. The randomized directional check has its
    own scalar ``directional_atol`` and median step because
    :func:`jax.test_util.check_grads` does not accept elementwise steps.
    Failures identify the exact PyTree leaf path and largest discrepancy.
    """
    step_leaves: list[Array] = [
        fd_step(jnp.real(jnp.asarray(leaf)), scale_floor=scale_floor)
        for leaf in jax.tree.leaves(theta)
    ]
    median_step: Float64[Array, ""] = jnp.median(
        jnp.concatenate([jnp.ravel(step) for step in step_leaves])
    )
    relative_tolerance: float = RTOL_LADDER[regime]
    value: Float64[Array, ""] = fn(theta)
    directional_absolute_tolerance: ScalarFloat = (
        FD_ROUNDOFF_FACTOR
        * EPS_F64
        * jnp.maximum(1.0, jnp.abs(value))
        / median_step
        if directional_atol is None
        else directional_atol
    )

    def checked_fn(candidate: PyTree) -> Float64[Array, ""]:
        normalized: PyTree = _as_jax_arrays(candidate)
        checked_value: Float64[Array, ""] = fn(normalized)
        return checked_value

    test_util.check_grads(
        checked_fn,
        (theta,),
        order=1,
        modes=modes,
        eps=float(median_step),
        atol=float(directional_absolute_tolerance),
        rtol=relative_tolerance,
    )
    automatic: PyTree = jax.grad(fn)(theta)
    finite_difference: PyTree = central_fd_grad(
        fn, theta, scale_floor=scale_floor
    )
    automatic_paths: list[Tuple[Tuple[object, ...], Array]]
    automatic_treedef: PyTreeDef
    automatic_paths, automatic_treedef = jax.tree_util.tree_flatten_with_path(
        automatic
    )
    finite_leaves: list[Array]
    finite_treedef: PyTreeDef
    finite_leaves, finite_treedef = jax.tree_util.tree_flatten(
        finite_difference
    )
    step_treedef: PyTreeDef
    step_treedef = jax.tree_util.tree_structure(theta)
    if (
        automatic_treedef != finite_treedef
        or automatic_treedef != step_treedef
    ):
        raise AssertionError("autodiff and finite-difference trees differ")
    path: Tuple[object, ...]
    actual: Array
    expected: Array
    step: Array
    for (path, actual), expected, step in zip(
        automatic_paths, finite_leaves, step_leaves, strict=True
    ):
        absolute_tolerance: Array = (
            FD_ROUNDOFF_FACTOR
            * EPS_F64
            * jnp.maximum(1.0, jnp.abs(value))
            / step
            if atol is None
            else jnp.full_like(step, atol)
        )
        tolerance: Array = absolute_tolerance + relative_tolerance * jnp.abs(
            expected
        )
        difference: Array = jnp.abs(actual - expected)
        if not bool(jnp.all(difference <= tolerance)):
            message: str = (
                f"gradient mismatch at {_path_name(path)}: "
                f"max_abs_error={float(jnp.max(difference)):.6e}, "
                f"max_atol={float(jnp.max(absolute_tolerance)):.6e}, "
                f"rtol={relative_tolerance:.6e}"
            )
            raise AssertionError(message)


@jaxtyped(typechecker=beartype)
def assert_nonzero_grad(
    fn: ScalarLoss,
    theta: PyTree,
    *,
    sensitive_paths: Optional[Tuple[str, ...]] = None,
    min_norm: ScalarFloat = 1e-12,
    elementwise: bool = False,
) -> None:
    """Assert selected gradients have physically useful sensitivity.

    The helper checks every leaf by default. ``sensitive_paths`` selects exact JAX
    key-path strings. By default, require each selected leaf to exceed
    ``min_norm`` in Euclidean norm. This rule permits physical structural zeros
    within a sensitive leaf. If ``elementwise`` is true, require every
    coordinate to exceed ``min_norm`` in magnitude. Reserve elementwise mode
    for contracts that explicitly register every coordinate as sensitive.
    """
    gradient: PyTree = jax.grad(fn)(theta)
    path_leaves: list[Tuple[Tuple[object, ...], Array]]
    path_leaves, _ = jax.tree_util.tree_flatten_with_path(gradient)
    available_paths: set[str] = {_path_name(path) for path, _ in path_leaves}
    selected_paths: set[str] = (
        available_paths if sensitive_paths is None else set(sensitive_paths)
    )
    missing_paths: set[str] = selected_paths - available_paths
    if missing_paths:
        message: str = (
            f"unknown sensitive gradient paths: {sorted(missing_paths)}"
        )
        raise ValueError(message)
    path: Tuple[object, ...]
    leaf: Array
    for path, leaf in path_leaves:
        path_name: str = _path_name(path)
        if path_name in selected_paths:
            if elementwise:
                magnitudes: Float64[Array, " ..."] = jnp.abs(jnp.ravel(leaf))
                if magnitudes.size == 0:
                    message = (
                        f"gradient at {path_name} is empty; "
                        "elementwise sensitivity requires a coordinate"
                    )
                    raise AssertionError(message)
                insensitive: Array = magnitudes <= min_norm
                if bool(jnp.any(insensitive)):
                    flat_index: int = int(jnp.argmax(insensitive))
                    magnitude: float = float(magnitudes[flat_index])
                    message = (
                        f"gradient at {path_name} coordinate {flat_index} "
                        f"has magnitude {magnitude:.6e}; "
                        f"required > {float(min_norm):.6e}"
                    )
                    raise AssertionError(message)
                continue
            norm: Float64[Array, ""] = jnp.linalg.norm(jnp.ravel(leaf))
            if not bool(norm > min_norm):
                message = (
                    f"gradient at {path_name} has norm {float(norm):.6e}; "
                    f"required > {float(min_norm):.6e}"
                )
                raise AssertionError(message)


@jaxtyped(typechecker=beartype)
def gradient_gate(
    fn: ScalarLoss,
    theta: PyTree,
    *,
    regime: GradRegime = "smooth",
    sensitive_paths: Optional[Tuple[str, ...]] = None,
    elementwise: bool = False,
    **kwargs: Any,
) -> None:
    """Run finite, finite-difference, and nonzero gradient checks together.

    Sibling-plan differentiability gates use this single entry point. The helper
    passes ``elementwise`` to :func:`assert_nonzero_grad`. It passes all other
    keyword arguments to :func:`assert_grad_matches_fd`.
    """
    gradient: PyTree = jax.grad(fn)(theta)
    assert_tree_finite(gradient)
    assert_grad_matches_fd(fn, theta, regime=regime, **kwargs)
    assert_nonzero_grad(
        fn,
        theta,
        sensitive_paths=sensitive_paths,
        elementwise=elementwise,
    )


@jaxtyped(typechecker=beartype)
def random_generic_complex(
    key: PRNGKeyArray,
    shape: Tuple[int, ...],
    *,
    scale: ScalarFloat = 1.0,
) -> Complex128[Array, "..."]:
    """Generate generic complex data with asymmetric independent components.

    Real and imaginary components use independent normal draws at scales
    ``scale`` and ``0.7 * scale``. The asymmetry prevents conjugation errors
    from passing accidentally through Hermitian or equal-component inputs.
    """
    real_key: PRNGKeyArray
    imaginary_key: PRNGKeyArray
    real_key, imaginary_key = jax.random.split(key)
    real_part: Float64[Array, "..."] = scale * jax.random.normal(
        real_key, shape
    )
    imaginary_part: Float64[Array, "..."] = (
        0.7 * scale * jax.random.normal(imaginary_key, shape)
    )
    values: Complex128[Array, "..."] = real_part + 1j * imaginary_part
    return values


@jaxtyped(typechecker=beartype)
def complex_step_derivative(
    fn: Callable[[Array], Array],
    x: Float64[Array, "..."],
    *,
    direction: Optional[Float64[Array, "..."]] = None,
    h: ScalarFloat = 1e-20,
) -> Float64[Array, "..."]:
    """Estimate a directional derivative by complex step.

    Evaluate ``imag(fn(x + 1j*h*direction)) / h`` for a holomorphic sub-block
    that is numerically real on the real axis. The default direction is an
    all-ones array. The result is an estimator, not a holomorphy detector:
    zero is a valid derivative and conjugation can return a nonzero but wrong
    value. Every certification use must compare against an independent
    analytic derivative or JVP. General complex-to-real maps use stacked-real
    central finite differences instead.
    """
    resolved_direction: Float64[Array, "..."] = (
        jnp.ones_like(x) if direction is None else direction
    )
    complex_value: Array = fn(
        x.astype(jnp.complex128) + 1j * h * resolved_direction
    )
    imaginary_part: Array = jnp.imag(complex_value)
    derivative: Float64[Array, "..."] = imaginary_part / h
    return derivative
