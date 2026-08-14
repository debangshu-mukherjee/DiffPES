"""Trace differentiable information flow through forward models.

Extended Summary
----------------
This module provides JAX-native tools for structural dependencies, local
sensitivity, retained linearization, and matrix-free information spectra.
Certified forward executions use these tools. Typed JAXPR provides the
structural flow. JVP and VJP operations measure numerical flow through the
actual forward program.

Routine Listings
----------------
:func:`clear_dependency_cache`
    Clear the eager cache for structural dependency analyses.
:func:`dependency_cache_info`
    Return cache size, hit count, and miss count.
:func:`dependency_map`
    Trace leaf-level structural and local numerical dependencies.
:func:`information_spectrum`
    Estimate the leading local information spectrum matrix-free.
:func:`linearized_forward`
    Evaluate a forward model and retain its JVP linearization.
:func:`sensitivity_map`
    Measure scaled JVP sensitivities for a batch of tangent directions.
"""

from functools import cache

import jax
import jax.numpy as jnp
from beartype import beartype
from beartype.typing import Any, Callable, Dict, List, Optional, Tuple
from jax import core
from jaxtyping import Array, Bool, Float, Int, Int32, PyTree, Shaped, jaxtyped

from diffpes.maths import pack_complex, unpack_complex
from diffpes.types import (
    DependencyAnalysisCache,
    DependencyMap,
    InformationSpectrum,
    SensitivityMap,
    make_dependency_analysis_cache,
    make_dependency_map,
    make_information_spectrum,
    make_sensitivity_map,
)


@cache
def _dependency_cache_state() -> DependencyAnalysisCache:
    """PRIVATE: Return the process-local cache for structural dependency
    analyses.

    Returns
    -------
    state : DependencyAnalysisCache
        The single mutable holder of cached analyses, hit and miss
        counters, and their reentrant lock.

    Notes
    -----
    The :func:`functools.cache` decorator makes the state object a
    process-local singleton. :func:`_dependency_structure` reads and
    fills it; :func:`clear_dependency_cache` resets it.
    """
    state: DependencyAnalysisCache = make_dependency_analysis_cache()
    return state


def _abstract_signature(tree: PyTree) -> Tuple[Any, ...]:
    """PRIVATE: Return a hashable signature for one static input
    configuration.

    Parameters
    ----------
    tree : PyTree
        Numerical input PyTree.

    Returns
    -------
    signature : Tuple[Any, ...]
        The tree-structure string and one ``(shape, dtype)`` pair per
        leaf.

    Notes
    -----
    Captures only the static configuration (structure, shapes, and
    dtypes), so cache entries stay valid across different numerical
    values. The signature joins the model identity and callable identity
    in the structural-cache key.
    """
    leaves: List[Any] = jax.tree.leaves(tree)
    leaf_signatures: Tuple[Tuple[Tuple[int, ...], str], ...] = tuple(
        (tuple(jnp.shape(leaf)), str(jnp.asarray(leaf).dtype))
        for leaf in leaves
    )
    signature: Tuple[Any, ...] = (
        str(jax.tree.structure(tree)),
        leaf_signatures,
    )
    return signature


def _path_names(tree: PyTree) -> Tuple[str, ...]:
    """PRIVATE: Return stable JAX key-path names for all leaves.

    Parameters
    ----------
    tree : PyTree
        Input or output PyTree.

    Returns
    -------
    names : Tuple[str, ...]
        One :func:`jax.tree_util.keystr` name per leaf in flattening
        order. A bare root leaf gets the name ``"$"``.

    Notes
    -----
    The names label the rows and columns of dependency matrices, so they
    must be deterministic for one tree structure.
    """
    flattened: Any = jax.tree_util.tree_flatten_with_path(tree)
    path_leaves: Any = flattened[0]
    names: Tuple[str, ...] = tuple(
        jax.tree_util.keystr(path) or "$" for path, _ in path_leaves
    )
    return names


def _structural_dependencies(
    forward_fn: Callable[[PyTree], PyTree], inputs: PyTree
) -> Tuple[PyTree, Bool[Array, "n_output n_input"]]:
    """PRIVATE: Propagate input-leaf dependency sets through a closed
    JAXPR.

    Implementation Logic
    --------------------
    Obtains the abstract output with :func:`jax.eval_shape` and the
    typed JAXPR with :func:`jax.make_jaxpr`. Seeds each input variable
    with its own index set and each constant with the empty set. Walks
    the equations in order. Each output receives the union of its
    equation's input index sets. The
    final matrix marks, for each JAXPR output, which input leaves can
    reach it structurally.

    Parameters
    ----------
    forward_fn : Callable[[PyTree], PyTree]
        Pure JAX forward function.
    inputs : PyTree
        Numerical inputs for one static shape and dtype configuration.

    Returns
    -------
    result : Tuple[PyTree, Bool[Array, "n_output n_input"]]
        Abstract output PyTree and the Boolean output-by-input
        structural dependency matrix.
    """
    constvar: Any
    equation: Any
    variable: Any
    output: PyTree = jax.eval_shape(forward_fn, inputs)
    closed: Any = jax.make_jaxpr(forward_fn)(inputs)
    jaxpr: Any = closed.jaxpr
    n_inputs: int = len(jaxpr.invars)
    dependency: Dict[Any, frozenset[int]] = {
        var: frozenset((index,)) for index, var in enumerate(jaxpr.invars)
    }
    for constvar in jaxpr.constvars:
        dependency[constvar] = frozenset()

    def variable_dependencies(variable: Any) -> frozenset[int]:
        try:
            result: frozenset[int] = dependency.get(variable, frozenset())
        except TypeError:
            result = frozenset()
        return result

    for equation in jaxpr.eqns:
        incoming: frozenset[int] = frozenset().union(
            *(variable_dependencies(variable) for variable in equation.invars)
        )
        for variable in equation.outvars:
            dependency[variable] = incoming
    rows: List[Bool[Array, " n_input"]] = []
    for variable in jaxpr.outvars:
        indices: frozenset[int] = dependency.get(variable, frozenset())
        rows.append(
            jnp.asarray(
                [index in indices for index in range(n_inputs)],
                dtype=jnp.bool_,
            )
        )
    structural: Bool[Array, "n_output n_input"] = jnp.stack(rows, axis=0)
    result: Tuple[PyTree, Bool[Array, "n_output n_input"]] = (
        output,
        structural,
    )
    return result


def _dependency_structure(
    model_id: str,
    forward_fn: Callable[[PyTree], PyTree],
    inputs: PyTree,
) -> Tuple[PyTree, Bool[Array, "n_output n_input"]]:
    """PRIVATE: Resolve one cached structural analysis outside compiled
    execution.

    The cache key includes the model ID, callable, tree, shapes, and dtypes.

    Parameters
    ----------
    model_id : str
        Permanent model identity and cache namespace.
    forward_fn : Callable[[PyTree], PyTree]
        Pure JAX forward function.
    inputs : PyTree
        Numerical inputs for one static shape and dtype configuration.

    Returns
    -------
    result : Tuple[PyTree, Bool[Array, "n_output n_input"]]
        Abstract output and output-by-input structural dependency matrix.

    Notes
    -----
    The cache holds concrete abstract records only. Traced calls do not update
    process state.
    """
    contains_tracer: bool = any(
        isinstance(leaf, core.Tracer) for leaf in jax.tree.leaves(inputs)
    )
    if contains_tracer:
        result: Tuple[PyTree, Bool[Array, "n_output n_input"]] = (
            _structural_dependencies(forward_fn, inputs)
        )
        return result
    key: Tuple[Any, ...] = (
        model_id,
        id(forward_fn),
        *_abstract_signature(inputs),
    )
    state: DependencyAnalysisCache = _dependency_cache_state()
    with state.lock:
        cached: Tuple[PyTree, Bool[Array, "n_output n_input"]] | None = (
            state.entries.get(key)
        )
        if cached is not None:
            object.__setattr__(state, "hits", state.hits + 1)
            return cached
    result = _structural_dependencies(forward_fn, inputs)
    with state.lock:
        state.entries[key] = result
        object.__setattr__(state, "misses", state.misses + 1)
    return result


@jaxtyped(typechecker=beartype)
def clear_dependency_cache() -> None:
    """Clear the eager cache for structural dependency analyses.

    The function resets process-local orchestration state between evaluations.

    :see: :class:`~.test_dependencies.TestClearDependencyCache`

    Notes
    -----
    The function changes orchestration state only. It does not run in a JAX
    numerical kernel.
    """
    state: DependencyAnalysisCache = _dependency_cache_state()
    with state.lock:
        state.entries.clear()
        object.__setattr__(state, "hits", 0)
        object.__setattr__(state, "misses", 0)


@jaxtyped(typechecker=beartype)
def dependency_cache_info() -> Tuple[int, int, int]:
    """Return cache size, hit count, and miss count.

    The tuple gives deterministic counters for structural cache tests.

    :see: :class:`~.test_dependencies.TestDependencyCacheInfo`

    Returns
    -------
    info : Tuple[int, int, int]
        Entry count, hit count, and miss count.

    Notes
    -----
    The counters measure eager structural analysis. They do not measure a JAX
    compilation cache.
    """
    state: DependencyAnalysisCache = _dependency_cache_state()
    with state.lock:
        info: Tuple[int, int, int] = (
            len(state.entries),
            state.hits,
            state.misses,
        )
    return info


def _leaf_direction(inputs: PyTree, leaf_index: int) -> PyTree:
    """PRIVATE: Build an all-ones tangent for one input leaf.

    Parameters
    ----------
    inputs : PyTree
        Numerical input PyTree that fixes the tangent structure.
    leaf_index : int
        Flattened index of the leaf to probe.

    Returns
    -------
    tangent : PyTree
        PyTree with the structure of ``inputs``: ones at the selected
        leaf and zeros at every other leaf.

    Notes
    -----
    The tangent probes one input leaf at a time through the retained
    pushforward when :func:`_dependency_map_from_linearization` builds
    the traced dependency matrix.
    """
    flattened: Tuple[List[Any], Any] = jax.tree_util.tree_flatten(inputs)
    leaves: List[Any] = flattened[0]
    treedef: Any = flattened[1]
    tangent_leaves: List[Shaped[Array, "..."]] = [
        jnp.zeros_like(leaf) for leaf in leaves
    ]
    tangent_leaves[leaf_index] = jnp.ones_like(leaves[leaf_index])
    tangent: PyTree = jax.tree_util.tree_unflatten(treedef, tangent_leaves)
    return tangent


@jaxtyped(typechecker=beartype)
def linearized_forward(
    forward_fn: Callable[[PyTree], PyTree], inputs: PyTree
) -> Tuple[PyTree, Callable[[PyTree], PyTree]]:
    """Evaluate a forward model and retain its JVP linearization.

    The operation exposes local information flow with JAX linearization and
    PyTree coordinates. Numerical leaves remain differentiable.

    :see: :class:`~.test_dependencies.TestLinearizedForward`


    Implementation Logic
    --------------------
    1. **Bind the documented output**::

           linearized = jax.linearize(forward_fn, inputs)

       The function validates and transforms the inputs before it binds the
       documented output.

    Parameters
    ----------
    forward_fn : Callable[[PyTree], PyTree]
        Pure JAX forward function accepting one input PyTree.
    inputs : PyTree
        Evaluation point for the forward model.

    Returns
    -------
    output : PyTree
        Forward-model value at ``inputs``.
    pushforward : Callable[[PyTree], PyTree]
        Reusable linear map from input tangents to output tangents.
    """
    linearized: Tuple[PyTree, Callable[[PyTree], PyTree]] = jax.linearize(
        forward_fn,
        inputs,
    )
    return linearized


@jaxtyped(typechecker=beartype)
def dependency_map(
    model_id: str,
    forward_fn: Callable[[PyTree], PyTree],
    inputs: PyTree,
    *,
    threshold: float = 1e-12,
) -> DependencyMap:
    """Trace leaf-level structural and local numerical dependencies.

    Structural dependencies come from typed JAXPR variable flow. ``traced``
    records whether an all-ones tangent produces a local response above
    ``threshold``. A false entry is local evidence, not global independence.

    :see: :class:`~.test_dependencies.TestDependencyMap`

    Implementation Logic
    --------------------
    1. **Bind the documented output**::

           result: DependencyMap = make_dependency_map(
                   model_id=model_id,
                   input_paths=_path_names(inputs),
                   output_paths=_path_names(abstract_output),
                   structural=structural,
                   traced=traced,
               )

       The function validates and transforms the inputs before it binds the
       documented output.

    Parameters
    ----------
    model_id : str
        Permanent scientific model identity (**static**).
    forward_fn : Callable[[PyTree], PyTree]
        Pure differentiable forward model.
    inputs : PyTree
        Numerical model inputs in their declared physical units.
    threshold : float
        Positive local-response threshold. Default 1e-12.

    Returns
    -------
    result : DependencyMap
        Structural and local numerical dependency matrices.

    Notes
    -----
    The traced matrix is differentiable only through its continuous JVP
    source. Thresholded Boolean entries do not carry useful gradients.
    """
    structural_evaluation: Tuple[PyTree, Bool[Array, "n_output n_input"]] = (
        _dependency_structure(model_id, forward_fn, inputs)
    )
    abstract_output: PyTree = structural_evaluation[0]
    structural: Bool[Array, "n_output n_input"] = structural_evaluation[1]
    linearized: Tuple[PyTree, Callable[[PyTree], PyTree]] = linearized_forward(
        forward_fn,
        inputs,
    )
    pushforward: Callable[[PyTree], PyTree] = linearized[1]
    result: DependencyMap = _dependency_map_from_linearization(
        model_id,
        inputs,
        abstract_output,
        structural,
        pushforward,
        threshold=threshold,
    )
    return result


@jaxtyped(typechecker=beartype)
def _dependency_map_from_linearization(
    model_id: str,
    inputs: PyTree,
    output: PyTree,
    structural: Bool[Array, "n_output n_input"],
    pushforward: Callable[[PyTree], PyTree],
    *,
    threshold: float = 1e-12,
) -> DependencyMap:
    """PRIVATE: Build a dependency map from one retained JAX
    linearization.

    The function reuses the supplied pushforward for every numerical probe.

    Parameters
    ----------
    model_id : str
        Permanent scientific model identity (**static**).
    inputs : PyTree
        Numerical inputs at the linearization point.
    output : PyTree
        Forward output at the linearization point.
    structural : Bool[Array, "n_output n_input"]
        Output-by-input structural dependency matrix.
    pushforward : Callable[[PyTree], PyTree]
        Retained JVP linear map.
    threshold : float
        Positive local-response threshold. Default 1e-12.

    Returns
    -------
    result : DependencyMap
        Structural and local numerical dependency matrices.

    Notes
    -----
    A false traced entry gives local evidence only. It does not establish
    global independence.
    """
    index: Any
    n_inputs: int = len(jax.tree.leaves(inputs))
    numerical_rows: List[Bool[Array, " n_output"]] = []
    for index in range(n_inputs):
        tangent: PyTree = _leaf_direction(inputs, index)
        response: PyTree = pushforward(tangent)
        activity: Bool[Array, " n_output"] = jnp.asarray(
            [
                jnp.linalg.norm(jnp.ravel(jnp.asarray(leaf))) > threshold
                for leaf in jax.tree.leaves(response)
            ],
            dtype=jnp.bool_,
        )
        numerical_rows.append(activity)
    traced: Bool[Array, "n_output n_input"] = jnp.stack(
        numerical_rows, axis=0
    ).T
    result: DependencyMap = make_dependency_map(
        model_id=model_id,
        input_paths=_path_names(inputs),
        output_paths=_path_names(output),
        structural=structural,
        traced=traced,
    )
    return result


@jaxtyped(typechecker=beartype)
def sensitivity_map(
    input_paths: Tuple[str, ...],
    output_projection_ids: Tuple[str, ...],
    forward_fn: Callable[[PyTree], Float[Array, " n_output"]],
    inputs: PyTree,
    directions: PyTree,
    scales: Float[Array, " n_input"],
    *,
    threshold: float = 1e-12,
) -> SensitivityMap:
    """Measure scaled JVP sensitivities for a batch of tangent directions.

    ``directions`` has the same tree structure as ``inputs`` and a leading
    probe axis on every leaf. The flattened output must correspond to
    ``output_projection_ids``.

    :see: :class:`~.test_dependencies.TestSensitivityMap`

    Implementation Logic
    --------------------
    1. **Bind the documented output**::

           result: SensitivityMap = make_sensitivity_map(
                   input_paths=input_paths,
                   output_projection_ids=output_projection_ids,
                   scales=scales,
                   sensitivities=scaled,
                   threshold=threshold,
                   active=active,
               )

       The function validates and transforms the inputs before it binds the
       documented output.

    Parameters
    ----------
    input_paths : Tuple[str, ...]
        Stable input-coordinate names (**static**).
    output_projection_ids : Tuple[str, ...]
        Stable output-projection names (**static**).
    forward_fn : Callable[[PyTree], Float[Array, " n_output"]]
        Pure differentiable forward model.
    inputs : PyTree
        Numerical model inputs in their declared physical units.
    directions : PyTree
        Batched tangent directions with a leading probe axis.
    scales : Float[Array, " n_input"]
        Positive physical scale for each input direction.
    threshold : float
        Absolute activity threshold. Default 1e-12.

    Returns
    -------
    result : SensitivityMap
        Scaled output-by-input sensitivities and activity indicators.

    Notes
    -----
    The sensitivity values carry gradients through ``jax.linearize``. The
    thresholded activity matrix is a derived diagnostic.
    """
    linearized: Tuple[PyTree, Callable[[PyTree], PyTree]] = linearized_forward(
        forward_fn,
        inputs,
    )
    pushforward: Callable[[PyTree], PyTree] = linearized[1]
    result: SensitivityMap = _sensitivity_map_from_linearization(
        input_paths,
        output_projection_ids,
        directions,
        scales,
        pushforward,
        threshold=threshold,
    )
    return result


@jaxtyped(typechecker=beartype)
def _sensitivity_map_from_linearization(
    input_paths: Tuple[str, ...],
    output_projection_ids: Tuple[str, ...],
    directions: PyTree,
    scales: Float[Array, " n_input"],
    pushforward: Callable[[PyTree], PyTree],
    *,
    threshold: float = 1e-12,
) -> SensitivityMap:
    """PRIVATE: Measure scaled sensitivities from one retained JAX
    linearization.

    Parameters
    ----------
    input_paths : Tuple[str, ...]
        Stable input-coordinate names (**static**).
    output_projection_ids : Tuple[str, ...]
        Stable output-projection names (**static**).
    directions : PyTree
        Batched tangent directions with a leading probe axis.
    scales : Float[Array, " n_input"]
        Positive physical scale for each input direction.
    pushforward : Callable[[PyTree], PyTree]
        Retained JVP linear map.
    threshold : float
        Absolute activity threshold. Default 1e-12.

    Returns
    -------
    result : SensitivityMap
        Scaled output-by-input sensitivities and activity indicators.

    Notes
    -----
    The function does not evaluate or linearize the nonlinear model again.
    """
    responses: Float[Array, "n_input n_output"] = jax.vmap(pushforward)(
        directions
    )
    response_array: Float[Array, "n_input n_output"] = jnp.reshape(
        responses, (responses.shape[0], -1)
    )
    scaled: Float[Array, "n_output n_input"] = (
        response_array * scales[:, None]
    ).T
    active: Bool[Array, "n_output n_input"] = jnp.abs(scaled) > threshold
    result: SensitivityMap = make_sensitivity_map(
        input_paths=input_paths,
        output_projection_ids=output_projection_ids,
        scales=scales,
        sensitivities=scaled,
        threshold=threshold,
        active=active,
    )
    return result


def _deterministic_subspace(
    size: int, rank: int, dtype: Any
) -> Float[Array, "size rank"]:
    """PRIVATE: Construct a deterministic full-rank starting subspace.

    Parameters
    ----------
    size : int
        Number of real input coordinates (matrix row count).
    rank : int
        Number of subspace columns.
    dtype : Any
        Real floating dtype of the subspace.

    Returns
    -------
    orthogonal : Float[Array, "size rank"]
        ``(size, rank)`` matrix with orthonormal columns.

    Notes
    -----
    Fills the matrix with incommensurate sine and cosine values of the
    index products and orthonormalizes it with a reduced QR
    decomposition. The construction needs no random key, so repeated
    spectrum estimates are bit-reproducible.
    """
    rows: Float[Array, "size 1"] = jnp.arange(1, size + 1, dtype=dtype)[
        :, None
    ]
    cols: Float[Array, "1 rank"] = jnp.arange(1, rank + 1, dtype=dtype)[
        None, :
    ]
    initial: Float[Array, "size rank"] = jnp.sin(rows * cols) + jnp.cos(
        rows * (cols + 0.5)
    )
    decomposition: Tuple[
        Float[Array, "size rank"], Float[Array, "rank rank"]
    ] = jnp.linalg.qr(
        initial,
        mode="reduced",
    )
    orthogonal: Float[Array, "size rank"] = decomposition[0]
    return orthogonal


def _element_paths(tree: PyTree) -> Tuple[str, ...]:
    """PRIVATE: Expand leaf paths to one stable name per scalar
    parameter.

    Parameters
    ----------
    tree : PyTree
        Numerical input PyTree.

    Returns
    -------
    paths : Tuple[str, ...]
        One name per real coordinate, in the order of
        :func:`_ravel_real_pytree`.

    Notes
    -----
    Starts from the leaf key path. Appends ``[index]`` when a leaf has
    more than one element. Appends ``.real`` and ``.imag`` for each
    complex element. The names label the rows of the right
    singular vectors in an information spectrum.
    """
    path: Any
    leaf: Any
    index: Any
    flattened: Any = jax.tree_util.tree_flatten_with_path(tree)
    path_leaves: Any = flattened[0]
    names: List[str] = []
    for path, leaf in path_leaves:
        base: str = jax.tree_util.keystr(path) or "$"
        array: Shaped[Array, "..."] = jnp.asarray(leaf)
        size: int = array.size
        components: Tuple[str, ...] = (
            ("real", "imag") if jnp.iscomplexobj(array) else ("",)
        )
        for index in range(size):
            indexed: str = base if size == 1 else f"{base}[{index}]"
            names.extend(
                indexed if not component else f"{indexed}.{component}"
                for component in components
            )
    paths: Tuple[str, ...] = tuple(names)
    return paths


def _ravel_real_pytree(
    tree: PyTree,
) -> Tuple[
    Float[Array, " n_coordinate"],
    Callable[[Float[Array, " n_coordinate"]], PyTree],
]:
    """PRIVATE: Ravel a numerical PyTree in independent real
    coordinates.

    Implementation Logic
    --------------------
    Flattens the tree and converts each leaf: a complex leaf becomes
    interleaved real and imaginary pairs through
    :func:`~diffpes.maths.pack_complex`; a real inexact leaf ravels
    directly; any other dtype raises. The returned ``unravel`` closure
    inverts the layout leaf by leaf and restores the original dtypes and
    tree structure.

    Parameters
    ----------
    tree : PyTree
        Numerical PyTree with inexact (float or complex) array leaves.

    Returns
    -------
    result : Tuple[
        Float[Array, " n_coordinate"],
        Callable[[Float[Array, " n_coordinate"]], PyTree],
        ]
        Flat real coordinate vector and the inverse map back to the
        PyTree.

    Raises
    ------
    TypeError
        If a leaf has a non-inexact dtype.
    """
    array: Any
    flattened: Tuple[List[Any], Any] = jax.tree_util.tree_flatten(tree)
    leaves: List[Any] = flattened[0]
    treedef: Any = flattened[1]
    arrays: List[Shaped[Array, "..."]] = [jnp.asarray(leaf) for leaf in leaves]
    parts: List[Float[Array, " n_part"]] = []
    for array in arrays:
        if jnp.iscomplexobj(array):
            parts.append(jnp.ravel(pack_complex(array)))
        elif jnp.issubdtype(array.dtype, jnp.inexact):
            parts.append(jnp.ravel(array))
        else:
            msg: str = "information inputs must contain inexact array leaves"
            raise TypeError(msg)
    flat: Float[Array, " n_coordinate"] = (
        jnp.concatenate(parts) if parts else jnp.zeros(0)
    )

    def unravel(vector: Float[Array, " n_coordinate"]) -> PyTree:
        array: Any
        offset: int = 0
        rebuilt: List[Shaped[Array, "..."]] = []
        for array in arrays:
            size: int = array.size
            if jnp.iscomplexobj(array):
                count: int = 2 * size
                packed: Float[Array, "... 2"] = jnp.reshape(
                    vector[offset : offset + count], (*array.shape, 2)
                )
                rebuilt.append(unpack_complex(packed).astype(array.dtype))
            else:
                count = size
                rebuilt.append(
                    jnp.reshape(
                        vector[offset : offset + count], array.shape
                    ).astype(array.dtype)
                )
            offset += count
        result: PyTree = jax.tree_util.tree_unflatten(treedef, rebuilt)
        return result

    result: Tuple[
        Float[Array, " n_coordinate"],
        Callable[[Float[Array, " n_coordinate"]], PyTree],
    ] = (flat, unravel)
    return result


@jaxtyped(typechecker=beartype)
def information_spectrum(  # noqa: PLR0915
    forward_fn: Callable[[PyTree], PyTree],
    inputs: PyTree,
    *,
    input_paths: Optional[Tuple[str, ...]] = None,
    output_weights: Optional[Float[Array, " n_output"]] = None,
    rank: int = 8,
    iterations: int = 8,
    threshold: float = 1e-10,
) -> InformationSpectrum:
    """Estimate the leading local information spectrum matrix-free.

    The function computes leading eigenpairs of ``J.T @ W @ J`` through JVP
    and VJP maps. It never materializes the dense Jacobian. Singular values
    are square roots of the nonnegative information eigenvalues.

    :see: :class:`~.test_dependencies.TestInformationSpectrum`

    Implementation Logic
    --------------------
    1. **Bind the documented output**::

           result: InformationSpectrum = make_information_spectrum(
                   input_paths=paths,
                   singular_values=singular_values,
                   right_singular_vectors=right_vectors,
                   effective_rank=effective_rank,
                   condition_estimate=condition,
                   threshold=threshold,
               )

       The function validates and transforms the inputs before it binds the
       documented output.

    Parameters
    ----------
    forward_fn : Callable[[PyTree], PyTree]
        Pure differentiable forward model.
    inputs : PyTree
        Numerical model inputs in their declared physical units.
    input_paths : Optional[Tuple[str, ...]]
        Names for flattened real input coordinates (**static**). Default None.
    output_weights : Optional[Float[Array, " n_output"]]
        Nonnegative metric weights for flattened outputs. Default None.
    rank : int
        Requested leading spectrum rank (**static**). Default 8.
    iterations : int
        Number of subspace iterations (**static**). Default 8.
    threshold : float
        Singular-value activity threshold. Default 1e-10.

    Returns
    -------
    result : InformationSpectrum
        Leading singular values, right vectors, rank, and condition estimate.

    Notes
    -----
    The subspace iteration and eigendecomposition remain JAX differentiable.
    Degenerate eigenvalues can make individual singular vectors non-unique.
    """
    flattened_inputs: Tuple[
        Float[Array, " n_input"], Callable[[Float[Array, " n_input"]], PyTree]
    ] = _ravel_real_pytree(inputs)
    flat_inputs: Float[Array, " n_input"] = flattened_inputs[0]
    unravel_inputs: Callable[[Float[Array, " n_input"]], PyTree] = (
        flattened_inputs[1]
    )

    def flat_forward(
        flat: Float[Array, " n_input"],
    ) -> Float[Array, " n_output"]:
        output: PyTree = forward_fn(unravel_inputs(flat))
        flattened_output: Tuple[
            Float[Array, " n_output"],
            Callable[[Float[Array, " n_output"]], PyTree],
        ] = _ravel_real_pytree(output)
        flat_output: Float[Array, " n_output"] = flattened_output[0]
        return flat_output

    linearized: Tuple[
        Float[Array, " n_output"],
        Callable[[Float[Array, " n_input"]], Float[Array, " n_output"]],
    ] = jax.linearize(flat_forward, flat_inputs)
    flat_output: Float[Array, " n_output"] = linearized[0]
    pushforward: Callable[
        [Float[Array, " n_input"]], Float[Array, " n_output"]
    ] = linearized[1]
    transposed: Callable[
        [Float[Array, " n_output"]], Tuple[Float[Array, " n_input"]]
    ] = jax.linear_transpose(
        pushforward,
        flat_inputs,
    )

    def pullback(
        cotangent: Float[Array, " n_output"],
    ) -> Float[Array, " n_input"]:
        pulled: Float[Array, " n_input"] = transposed(cotangent)[0]
        return pulled

    result: InformationSpectrum = _information_spectrum_from_linearization(
        inputs,
        flat_output,
        pushforward,
        pullback,
        input_paths=input_paths,
        output_weights=output_weights,
        rank=rank,
        iterations=iterations,
        threshold=threshold,
    )
    return result


@jaxtyped(typechecker=beartype)
def _information_spectrum_from_linearization(  # noqa: PLR0913
    inputs: PyTree,
    flat_output: Float[Array, " n_output"],
    pushforward: Callable[[Float[Array, "n_input"]], Float[Array, "n_output"]],
    pullback: Callable[[Float[Array, "n_output"]], Float[Array, "n_input"]],
    *,
    input_paths: Optional[Tuple[str, ...]] = None,
    output_weights: Optional[Float[Array, " n_output"]] = None,
    rank: int = 8,
    iterations: int = 8,
    threshold: float = 1e-10,
) -> InformationSpectrum:
    """PRIVATE: Estimate an information spectrum from one retained
    linearization.

    The function applies supplied JVP and transpose maps. It does not evaluate
    or linearize the nonlinear model again.

    Parameters
    ----------
    inputs : PyTree
        Numerical inputs at the linearization point.
    flat_output : Float[Array, " n_output"]
        Forward output in independent real coordinates.
    pushforward : Callable[[Float[Array, "n_input"]], Float[Array, "n_output"]]
        Linear map from real input coordinates to real output coordinates.
    pullback : Callable[[Float[Array, "n_output"]], Float[Array, "n_input"]]
        Transpose map from real output coordinates to real input coordinates.
    input_paths : Optional[Tuple[str, ...]]
        Names for flattened real input coordinates. Default None.
    output_weights : Optional[Float[Array, " n_output"]]
        Nonnegative output metric weights. Default None.
    rank : int
        Requested leading spectrum rank (**static**). Default 8.
    iterations : int
        Number of subspace iterations (**static**). Default 8.
    threshold : float
        Singular-value activity threshold. Default 1e-10.

    Returns
    -------
    result : InformationSpectrum
        Leading singular values, right vectors, rank, and condition estimate.

    Raises
    ------
    ValueError
        If metric weights have the wrong shape or the inputs are empty.
    """
    flat_inputs: Float[Array, " n_input"] = _ravel_real_pytree(inputs)[0]
    weights: Float[Array, " n_output"] = (
        jnp.ones_like(flat_output)
        if output_weights is None
        else jnp.asarray(output_weights, dtype=flat_output.real.dtype)
    )
    if weights.shape != flat_output.shape:
        raise ValueError("output_weights must match the flattened output")
    effective_rank_limit: int = min(rank, flat_inputs.size, flat_output.size)
    if effective_rank_limit < 1:
        raise ValueError("rank requires non-empty inputs and outputs")

    def normal(
        vector: Float[Array, " n_input"],
    ) -> Float[Array, " n_input"]:
        response: Float[Array, " n_output"] = pushforward(vector)
        pulled: Float[Array, " n_input"] = pullback(weights * response)
        result: Float[Array, " n_input"] = jnp.real(pulled)
        return result

    def apply_columns(
        matrix: Float[Array, "n_input rank"],
    ) -> Float[Array, "n_input rank"]:
        applied: Float[Array, "n_input rank"] = jax.vmap(
            normal, in_axes=1, out_axes=1
        )(matrix)
        return applied

    subspace: Float[Array, "n_input rank"] = _deterministic_subspace(
        flat_inputs.size, effective_rank_limit, flat_inputs.real.dtype
    )

    def iteration(
        _: Int[Array, ""], basis: Float[Array, "n_input rank"]
    ) -> Float[Array, "n_input rank"]:
        updated: Float[Array, "n_input rank"] = apply_columns(basis)
        orthogonal: Float[Array, "n_input rank"] = jnp.linalg.qr(
            updated, mode="reduced"
        )[0]
        return orthogonal

    subspace = jax.lax.fori_loop(0, iterations, iteration, subspace)
    projected: Float[Array, "rank rank"] = subspace.T @ apply_columns(subspace)
    eigenvalues: Float[Array, " rank"]
    eigenvectors_small: Float[Array, "rank rank"]
    eigenvalues, eigenvectors_small = jnp.linalg.eigh(projected)
    order: Int[Array, " rank"] = jnp.argsort(eigenvalues)[::-1]
    eigenvalues = jnp.maximum(eigenvalues[order], 0.0)
    right_vectors: Float[Array, "rank n_input"] = (
        subspace @ eigenvectors_small[:, order]
    ).T
    singular_values: Float[Array, " rank"] = jnp.sqrt(eigenvalues)
    active: Bool[Array, " rank"] = singular_values > threshold
    effective_rank: Int32[Array, ""] = jnp.sum(active, dtype=jnp.int32)
    smallest_active: Float[Array, ""] = jnp.min(
        jnp.where(active, singular_values, jnp.inf)
    )
    condition: Float[Array, ""] = jnp.where(
        effective_rank > 0,
        singular_values[0] / smallest_active,
        0.0,
    )
    paths: Tuple[str, ...] = (
        _element_paths(inputs) if input_paths is None else input_paths
    )
    if len(paths) != flat_inputs.size:
        paths = _element_paths(inputs)
    result: InformationSpectrum = make_information_spectrum(
        input_paths=paths,
        singular_values=singular_values,
        right_singular_vectors=right_vectors,
        effective_rank=effective_rank,
        condition_estimate=condition,
        threshold=threshold,
    )
    return result


__all__: list[str] = [
    "clear_dependency_cache",
    "dependency_cache_info",
    "dependency_map",
    "information_spectrum",
    "linearized_forward",
    "sensitivity_map",
]
