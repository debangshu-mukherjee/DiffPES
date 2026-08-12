"""Compute static-shape sharding operations.

Extended Summary
----------------
This module pads physical k points with finite repeated values. It then uses
``lax.scan`` and ``jax.checkpoint`` to bound live memory by the chunk shape.
The callers own device mesh placement and scientific reductions.

Routine Listings
----------------
:func:`pad_with_mask`
    Pad k points with finite repeated values and return an f64 mask.
:func:`sharded_kmap`
    Map a checkpointed body across static k-point chunks.
:func:`sharded_ksum`
    Sum scalar chunk outputs across static k-point chunks.
"""

import jax
import jax.numpy as jnp
from beartype import beartype
from beartype.typing import Callable, Tuple
from jaxtyping import Array, Float64, jaxtyped

from diffpes.constants import ARRAY_MATRIX_NDIM, CARTESIAN_COMPONENTS
from diffpes.types import ShardSpec


@jaxtyped(typechecker=beartype)
def pad_with_mask(
    kpoints: Float64[Array, "n_k 3"],
    nk_max: int,
) -> Tuple[Float64[Array, "nk_max 3"], Float64[Array, " nk_max"]]:
    """Pad k points with finite repeated values and return an f64 mask.

    The function repeats the final valid point. It never creates an invalid
    padded coordinate. The zero mask removes the repeated contributions.

    :see: :class:`~.test_sharding.TestPadWithMask`

    Parameters
    ----------
    kpoints : Float64[Array, "n_k 3"]
        Valid Cartesian or fractional k points.
    nk_max : int
        **Static.** Padded k-point capacity.

    Returns
    -------
    padded_kpoints : Float64[Array, "nk_max 3"]
        Finite padded k points.
    k_mask : Float64[Array, " nk_max"]
        One for valid points and zero for padded points.

    Raises
    ------
    ValueError
        If k points are empty, have an invalid shape, or exceed ``nk_max``.

    Notes
    -----
    The repeated padding makes both primal and tangent values finite. This
    property prevents a masked invalid branch from producing a NaN gradient.
    """
    points: Float64[Array, "n_k 3"] = jnp.asarray(kpoints, dtype=jnp.float64)
    if (
        points.ndim != ARRAY_MATRIX_NDIM
        or points.shape[1] != CARTESIAN_COMPONENTS
        or points.shape[0] == 0
        or points.shape[0] > nk_max
    ):
        raise ValueError("k points must be nonempty 3-vectors within nk_max")
    padding: int = nk_max - points.shape[0]
    padded_kpoints: Float64[Array, "nk_max 3"] = jnp.concatenate(
        (
            points,
            jnp.broadcast_to(points[-1], (padding, CARTESIAN_COMPONENTS)),
        ),
        axis=0,
    )
    k_mask: Float64[Array, " nk_max"] = jnp.concatenate(
        (
            jnp.ones((points.shape[0],), dtype=jnp.float64),
            jnp.zeros((padding,), dtype=jnp.float64),
        )
    )
    return padded_kpoints, k_mask


def _checkpoint_body(  # noqa: DOC105
    body: Callable, spec: ShardSpec
) -> Callable:
    """PRIVATE: Apply the static rematerialization policy to a chunk body.

    Parameters
    ----------
    body : Callable
        Pure JAX chunk function.
    spec : ShardSpec
        Static rematerialization selector.

    Returns
    -------
    checkpointed_body : Callable
        Body wrapped by the configured JAX checkpoint operation.
    """
    checkpointed_body: Callable
    if spec.checkpoint_policy == "dots_saveable":
        checkpointed_body = jax.checkpoint(
            body,
            policy=jax.checkpoint_policies.dots_with_no_batch_dims_saveable,
        )
    else:
        checkpointed_body = jax.checkpoint(body)
    return checkpointed_body


@jaxtyped(typechecker=beartype)
def sharded_kmap(  # noqa: DOC105
    body: Callable[
        [Float64[Array, "chunk 3"], Float64[Array, " chunk"]],
        Float64[Array, "chunk n_omega"],
    ],
    kpoints: Float64[Array, "nk_max 3"],
    kweights: Float64[Array, " nk_max"],
    spec: ShardSpec,
) -> Float64[Array, "nk_max n_omega"]:
    """Map a checkpointed body across static k-point chunks.

    The function reshapes a capacity-fixed k axis into chunks. It uses
    ``lax.scan`` so the JAXPR size does not grow with the chunk count.

    :see: :class:`~.test_sharding.TestShardedKmap`

    Parameters
    ----------
    body : Callable
        Pure chunk function that returns one row for each input k point.
    kpoints : Float64[Array, "nk_max 3"]
        Capacity-padded k points.
    kweights : Float64[Array, " nk_max"]
        Mask-folded quadrature weights.
    spec : ShardSpec
        **Static.** Execution and chunk policy.

    Returns
    -------
    mapped_values : Float64[Array, "nk_max n_omega"]
        Concatenated output from every k-point chunk.

    Raises
    ------
    ValueError
        If the input axes disagree with the static capacity.
    """
    if kpoints.shape != (spec.nk_max, 3) or kweights.shape != (spec.nk_max,):
        raise ValueError("sharded k map inputs must match ShardSpec.nk_max")
    chunk_count: int = spec.nk_max // spec.chunk_size
    chunk_points: Float64[Array, "n_chunk chunk 3"] = jnp.reshape(
        kpoints, (chunk_count, spec.chunk_size, CARTESIAN_COMPONENTS)
    )
    chunk_weights: Float64[Array, "n_chunk chunk"] = jnp.reshape(
        kweights, (chunk_count, spec.chunk_size)
    )
    checkpointed_body: Callable = _checkpoint_body(body, spec)

    def scan_body(
        carry: None,
        inputs: Tuple[Float64[Array, "chunk 3"], Float64[Array, " chunk"]],
    ) -> Tuple[None, Float64[Array, "chunk n_omega"]]:
        """PRIVATE: Compute one checkpointed k-point chunk.

        Parameters
        ----------
        carry : None
            Empty scan carry.
        inputs : Tuple[Float64[Array, "chunk 3"], Float64[Array, " chunk"]]
            K points and weights for one static chunk.

        Returns
        -------
        next_carry : None
            Empty scan carry.
        chunk_values : Float64[Array, "chunk n_omega"]
            Chunk output from ``body``.
        """
        del carry
        chunk_values: Float64[Array, "chunk n_omega"] = checkpointed_body(
            inputs[0], inputs[1]
        )
        next_carry: None = None
        return next_carry, chunk_values

    _: None
    scanned_values: Float64[Array, "n_chunk chunk n_omega"]
    _, scanned_values = jax.lax.scan(
        scan_body, None, (chunk_points, chunk_weights)
    )
    mapped_values: Float64[Array, "nk_max n_omega"] = jnp.reshape(
        scanned_values, (spec.nk_max, scanned_values.shape[-1])
    )
    return mapped_values


@jaxtyped(typechecker=beartype)
def sharded_ksum(  # noqa: DOC105
    body: Callable[
        [Float64[Array, "chunk 3"], Float64[Array, " chunk"]],
        Float64[Array, " chunk"],
    ],
    kpoints: Float64[Array, "nk_max 3"],
    kweights: Float64[Array, " nk_max"],
    spec: ShardSpec,
) -> Float64[Array, ""]:
    """Sum scalar chunk outputs across static k-point chunks.

    The function adapts a scalar-per-k body to :func:`sharded_kmap`. It
    preserves the caller-provided mask-folded weights.

    :see: :class:`~.test_sharding.TestShardedKsum`

    Parameters
    ----------
    body : Callable
        Pure function that returns one scalar per k point.
    kpoints : Float64[Array, "nk_max 3"]
        Capacity-padded k points.
    kweights : Float64[Array, " nk_max"]
        Mask-folded quadrature weights.
    spec : ShardSpec
        **Static.** Execution and chunk policy.

    Returns
    -------
    summed_value : Float64[Array, ""]
        Sum of the scalar body values.
    """
    mapped_values: Float64[Array, "nk_max 1"] = sharded_kmap(
        lambda points, weights: body(points, weights)[:, None],
        kpoints,
        kweights,
        spec,
    )
    summed_value: Float64[Array, ""] = jnp.sum(mapped_values[:, 0])
    return summed_value


__all__: list[str] = [
    "pad_with_mask",
    "sharded_kmap",
    "sharded_ksum",
]
