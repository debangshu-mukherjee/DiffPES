"""Define static execution policies for JAX sharding.

Extended Summary
----------------
Use this module for its validated public contracts and operations.

Routine Listings
----------------
:class:`ShardSpec`
    Define the ``ShardSpec`` public contract.
:func:`make_shard_spec`
    Compute the ``make_shard_spec`` public contract.
"""

import equinox as eqx
import jax
from beartype import beartype
from beartype.typing import Optional
from jaxtyping import jaxtyped

from diffpes.constants import SHARD_CHECKPOINT_POLICIES


class ShardSpec(eqx.Module):
    """Define the ``ShardSpec`` public contract.

    Validate documented inputs and preserve the declared scientific identity.

    :see: :class:`~.test_sharding.TestShardSpec`

    Attributes
    ----------
    device_axis : str
        Store the mesh-axis name.
    n_devices : int
        Store the device count.
    chunk_size : int
        Store the per-device chunk size.
    nk_max : int
        Store the padded capacity.
    checkpoint_policy : str
        Store the checkpoint policy.
    demote_accumulation : bool
        Store the accumulation-precision policy.

    See Also
    --------
    make_shard_spec
        Construct a validated sharding specification.
    """

    device_axis: str = eqx.field(static=True)
    n_devices: int = eqx.field(static=True)
    chunk_size: int = eqx.field(static=True)
    nk_max: int = eqx.field(static=True)
    checkpoint_policy: str = eqx.field(static=True)
    demote_accumulation: bool = eqx.field(static=True)

    def __check_init__(self) -> None:
        """PRIVATE: Validate static mesh and chunk compatibility.

        Raises
        ------
        ValueError
            If a mesh field is nonpositive, a selector is unknown, or the
            padded capacity does not divide into local chunks.
        """
        local_capacity: int = self.n_devices * self.chunk_size
        if not self.device_axis:
            raise ValueError("sharding device axis must be nonempty")
        if self.n_devices <= 0:
            raise ValueError("sharding device count must be positive")
        if self.chunk_size <= 0:
            raise ValueError("sharding chunk size must be positive")
        if self.nk_max <= 0:
            raise ValueError("sharding capacity must be positive")
        if self.nk_max % local_capacity != 0:
            raise ValueError("sharding capacity must divide into local chunks")
        if self.checkpoint_policy not in SHARD_CHECKPOINT_POLICIES:
            raise ValueError("sharding checkpoint policy is unsupported")


@jaxtyped(typechecker=beartype)
def make_shard_spec(
    n_devices: Optional[int] = None,
    chunk_size: int = 256,
    nk_max: Optional[int] = None,
    checkpoint_policy: str = "everything",
    demote_accumulation: bool = False,
    *,
    device_axis: str = "k",
) -> ShardSpec:
    """Compute the ``make_shard_spec`` public contract.

    Validate documented inputs and preserve the declared scientific identity.

    :see: :class:`~.test_sharding.TestMakeShardSpec`

    Notes
    -----
    Validate inputs before returning the named result.

    Parameters
    ----------
    n_devices : Optional[int]
        Input value for this operation.
    chunk_size : int
        Input value for this operation.
    nk_max : Optional[int]
        Input value for this operation.
    checkpoint_policy : str
        Input value for this operation.
    demote_accumulation : bool
        Input value for this operation.
    device_axis : str
        Input value for this operation.

    Returns
    -------
    result : ShardSpec
        Validated operation result.

    Raises
    ------
    ValueError
        If accumulation demotion lacks an implementation.
    """
    resolved_devices: int
    if n_devices is None:
        resolved_devices = len(jax.local_devices())
    else:
        resolved_devices = n_devices
    resolved_capacity: int = (
        resolved_devices * chunk_size if nk_max is None else nk_max
    )
    if demote_accumulation:
        raise ValueError("f32 accumulation is not implemented")
    spec: ShardSpec = ShardSpec(
        device_axis=device_axis,
        n_devices=resolved_devices,
        chunk_size=chunk_size,
        nk_max=resolved_capacity,
        checkpoint_policy=checkpoint_policy,
        demote_accumulation=demote_accumulation,
    )
    return spec


__all__: list[str] = [
    "ShardSpec",
    "make_shard_spec",
]
