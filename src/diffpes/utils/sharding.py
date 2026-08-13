"""Compute static-shape sharding operations.

Extended Summary
----------------
Use this module for its validated public contracts and operations.

Routine Listings
----------------
:func:`pad_with_mask`
    Compute the ``pad_with_mask`` public contract.
:func:`sharded_kmap`
    Compute the ``sharded_kmap`` public contract.
:func:`sharded_ksum`
    Compute the ``sharded_ksum`` public contract.
"""

import jax
import jax.numpy as jnp
from beartype import beartype
from beartype.typing import Callable, Tuple
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P
from jaxtyping import Array, Float64, jaxtyped

from diffpes.constants import ARRAY_MATRIX_NDIM, CARTESIAN_COMPONENTS
from diffpes.types import ShardSpec


@jaxtyped(typechecker=beartype)
def pad_with_mask(
    kpoints: Float64[Array, "n_k 3"],
    nk_max: int,
) -> Tuple[Float64[Array, "nk_max 3"], Float64[Array, " nk_max"]]:
    """Compute the ``pad_with_mask`` public contract.

    Validate documented inputs and preserve the declared scientific identity.

    :see: :class:`~.test_sharding.TestPadWithMask`

    Notes
    -----
    Fold the physical-lane mask into ``kweights``. Multiply those weights into
    every returned lane. This utility does not suppress padding again.

    Parameters
    ----------
    kpoints : Float64[Array, 'n_k 3']
        Input value for this operation.
    nk_max : int
        Input value for this operation.

    Returns
    -------
    result : Tuple[Float64[Array, 'nk_max 3'], Float64[Array, ' nk_max']]
        Validated operation result.

    Raises
    ------
    ValueError
        If k points are empty, malformed, or exceed the padded capacity.
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
    result: Tuple[Float64[Array, "nk_max 3"], Float64[Array, " nk_max"]] = (
        padded_kpoints,
        k_mask,
    )
    return result


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
    """Compute the ``sharded_kmap`` public contract.

    Validate documented inputs and preserve the declared scientific identity.

    :see: :class:`~.test_sharding.TestShardedKmap`

    Notes
    -----
    Fold the physical-lane mask into ``kweights``. Multiply those weights into
    every returned lane before reduction. Otherwise, padded values and their
    gradients remain observable.

    Parameters
    ----------
    body : Callable
        Input value for this operation.
    kpoints : Float64[Array, 'nk_max 3']
        Input value for this operation.
    kweights : Float64[Array, ' nk_max']
        Input value for this operation.
    spec : ShardSpec
        Input value for this operation.

    Returns
    -------
    result : Float64[Array, 'nk_max n_omega']
        Validated operation result.

    Raises
    ------
    ValueError
        If input arrays do not match the static sharding capacity.
    """
    if kpoints.shape != (spec.nk_max, 3) or kweights.shape != (spec.nk_max,):
        raise ValueError("sharded k map inputs must match ShardSpec.nk_max")
    checkpointed_body: Callable = _checkpoint_body(body, spec)

    def local_map(
        local_points: Float64[Array, "n_local 3"],
        local_weights: Float64[Array, " n_local"],
    ) -> Float64[Array, "n_local n_omega"]:
        """Evaluate checkpointed chunks on one mesh shard.

        Notes
        -----
        Reshape the local shard into static chunks before scanning.
        """
        local_capacity: int = spec.nk_max // spec.n_devices
        local_chunk_count: int = local_capacity // spec.chunk_size
        chunk_points: Float64[Array, "n_chunk chunk 3"] = jnp.reshape(
            local_points,
            (local_chunk_count, spec.chunk_size, CARTESIAN_COMPONENTS),
        )
        chunk_weights: Float64[Array, "n_chunk chunk"] = jnp.reshape(
            local_weights,
            (local_chunk_count, spec.chunk_size),
        )

        def scan_body(
            carry: None,
            inputs: Tuple[Float64[Array, "chunk 3"], Float64[Array, " chunk"]],
        ) -> Tuple[None, Float64[Array, "chunk n_omega"]]:
            """Evaluate one checkpointed momentum chunk.

            Notes
            -----
            Pass the chunk arrays to the rematerialized body.
            """
            del carry
            chunk_values: Float64[Array, "chunk n_omega"] = checkpointed_body(
                inputs[0], inputs[1]
            )
            result: Tuple[None, Float64[Array, "chunk n_omega"]] = (
                None,
                chunk_values,
            )
            return result

        scanned_values: Float64[Array, "n_chunk chunk n_omega"]
        _, scanned_values = jax.lax.scan(
            scan_body, None, (chunk_points, chunk_weights)
        )
        local_values: Float64[Array, "n_local n_omega"] = jnp.reshape(
            scanned_values, (local_capacity, scanned_values.shape[-1])
        )
        return local_values

    mesh: Mesh = jax.make_mesh((spec.n_devices,), (spec.device_axis,))
    with mesh:
        mapped_values: Float64[Array, "nk_max n_omega"] = jax.shard_map(
            local_map,
            mesh=mesh,
            in_specs=(P(spec.device_axis), P(spec.device_axis)),
            out_specs=P(spec.device_axis),
        )(kpoints, kweights)
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
    """Compute the ``sharded_ksum`` public contract.

    Validate documented inputs and preserve the declared scientific identity.

    :see: :class:`~.test_sharding.TestShardedKsum`

    Notes
    -----
    Validate inputs before returning the named result.

    Parameters
    ----------
    body : Callable
        Input value for this operation.
    kpoints : Float64[Array, 'nk_max 3']
        Input value for this operation.
    kweights : Float64[Array, ' nk_max']
        Input value for this operation.
    spec : ShardSpec
        Input value for this operation.

    Returns
    -------
    result : Float64[Array, '']
        Validated operation result.

    Raises
    ------
    ValueError
        If input arrays do not match the static sharding capacity.
    """
    if kpoints.shape != (spec.nk_max, 3) or kweights.shape != (spec.nk_max,):
        raise ValueError("sharded k sum inputs must match ShardSpec.nk_max")
    checkpointed_body: Callable = _checkpoint_body(body, spec)

    def local_sum(
        local_points: Float64[Array, "n_local 3"],
        local_weights: Float64[Array, " n_local"],
    ) -> Float64[Array, ""]:
        """Evaluate local chunks and sum across the device mesh.

        Notes
        -----
        Accumulate local values before the collective reduction.
        """
        local_capacity: int = spec.nk_max // spec.n_devices
        local_chunk_count: int = local_capacity // spec.chunk_size
        chunk_points: Float64[Array, "n_chunk chunk 3"] = jnp.reshape(
            local_points,
            (local_chunk_count, spec.chunk_size, CARTESIAN_COMPONENTS),
        )
        chunk_weights: Float64[Array, "n_chunk chunk"] = jnp.reshape(
            local_weights,
            (local_chunk_count, spec.chunk_size),
        )

        def scan_body(
            partial: Float64[Array, ""],
            inputs: Tuple[Float64[Array, "chunk 3"], Float64[Array, " chunk"]],
        ) -> Tuple[Float64[Array, ""], None]:
            """Add one checkpointed chunk to a scalar partial.

            Notes
            -----
            Preserve a scalar scan carry for bounded accumulation.
            """
            values: Float64[Array, " chunk"] = checkpointed_body(
                inputs[0], inputs[1]
            )
            next_partial: Float64[Array, ""] = partial + jnp.sum(values)
            result: Tuple[Float64[Array, ""], None] = (next_partial, None)
            return result

        initial: Float64[Array, ""] = jax.lax.pcast(
            jnp.asarray(0.0, dtype=jnp.float64),
            (spec.device_axis,),
            to="varying",
        )
        local_total: Float64[Array, ""]
        local_total, _ = jax.lax.scan(
            scan_body, initial, (chunk_points, chunk_weights)
        )
        mesh_total: Float64[Array, ""] = jax.lax.psum(
            local_total, spec.device_axis
        )
        return mesh_total

    mesh: Mesh = jax.make_mesh((spec.n_devices,), (spec.device_axis,))
    with mesh:
        summed_value: Float64[Array, ""] = jax.shard_map(
            local_sum,
            mesh=mesh,
            in_specs=(P(spec.device_axis), P(spec.device_axis)),
            out_specs=P(),
        )(kpoints, kweights)
    return summed_value


__all__: list[str] = [
    "pad_with_mask",
    "sharded_kmap",
    "sharded_ksum",
]
