"""Define static execution policies for JAX sharding.

Extended Summary
----------------
This module stores the static mesh, chunk, rematerialization, and precision
choices for one bounded single-host evaluation. Numerical physics leaves do
not live in this carrier.

Routine Listings
----------------
:class:`ShardSpec`
    Store a static execution policy for JAX sharding.
:func:`make_shard_spec`
    Create a validated static execution policy for JAX sharding.
"""

import equinox as eqx
import jax
from beartype import beartype
from beartype.typing import Optional
from jaxtyping import jaxtyped

from diffpes.constants import SHARD_CHECKPOINT_POLICIES


class ShardSpec(eqx.Module):
    """Store a static execution policy for JAX sharding.

    The fields define compilation structure only. A changed field causes
    retracing. The carrier does not select scientific physics.

    :see: :class:`~.test_sharding.TestShardSpec`

    Attributes
    ----------
    device_axis : str
        **Static.** Mesh axis name. Changing it triggers retracing.
    n_devices : int
        **Static.** Number of local mesh devices. Changing it triggers
        retracing.
    chunk_size : int
        **Static.** Number of k points per compiled chunk. Changing it
        triggers retracing.
    nk_max : int
        **Static.** Padded k-point capacity. Changing it triggers retracing.
    checkpoint_policy : str
        **Static.** ``"everything"`` or ``"dots_saveable"``. Changing it
        triggers retracing.
    demote_accumulation : bool
        **Static.** Opt-in final real accumulation demotion. Changing it
        triggers retracing.

    See Also
    --------
    make_shard_spec : Create a validated static execution policy for JAX
        sharding.
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
        if (
            not self.device_axis
            or self.n_devices <= 0
            or self.chunk_size <= 0
            or self.nk_max <= 0
            or self.nk_max % local_capacity != 0
            or self.checkpoint_policy not in SHARD_CHECKPOINT_POLICIES
        ):
            raise ValueError("invalid static sharding execution policy")


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
    """Create a validated static execution policy for JAX sharding.

    The factory uses the local device count only when callers omit it. The
    padded capacity remains static for the complete compiled model family.

    :see: :class:`~.test_sharding.TestMakeShardSpec`

    Parameters
    ----------
    n_devices : Optional[int]
        **Static.** Local device count. Default ``None`` uses JAX local
        devices.
    chunk_size : int
        **Static.** K points per chunk. Default 256.
    nk_max : Optional[int]
        **Static.** Padded k-point capacity. Default ``None`` selects one
        chunk.
    checkpoint_policy : str
        **Static.** Rematerialization selector. Default ``"everything"``.
    demote_accumulation : bool
        **Static.** Enable approved final real accumulation demotion. Default
        ``False``.
    device_axis : str
        **Static.** Mesh axis label. Default ``"k"``.

    Returns
    -------
    spec : ShardSpec
        Validated static execution policy.

    Raises
    ------
    ValueError
        If the static capacity is incompatible with device-local chunks.
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
