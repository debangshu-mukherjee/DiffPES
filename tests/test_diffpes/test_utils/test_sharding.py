"""Specify finite padding and static-shape sharding primitives.

Extended Summary
----------------
The contracts protect finite repeated padding and masked reductions before the
factorized forward model consumes them.

Routine Listings
----------------
:class:`TestPadding`
    Specify finite repeated k-point padding.
:class:`TestShardedReductions`
    Specify static-shape mapping and scalar reduction.
"""

import jax.numpy as jnp
from jaxtyping import Array, Float64

from diffpes.types import make_shard_spec
from diffpes.utils import pad_with_mask, sharded_kmap, sharded_ksum


class TestPadding:
    """Specify finite repeated k-point padding.

    See Also
    --------
    diffpes.utils.pad_with_mask
        Production padding primitive covered by these contract tests.
    """

    def test_repeats_final_physical_kpoint_and_masks_padding(self) -> None:
        """Keep padded lanes finite before their contribution is zero
        weighted.
        """
        points = jnp.asarray([[0.0, 0.0, 0.0], [0.25, 0.0, 0.0]])
        padded, mask = pad_with_mask(points, 4)
        assert jnp.allclose(padded[-1], points[-1])
        assert jnp.array_equal(mask, jnp.asarray([1.0, 1.0, 0.0, 0.0]))


class TestShardedReductions:
    """Specify static-shape mapping and scalar reduction.

    See Also
    --------
    diffpes.utils.sharded_kmap
        Production mapping primitive covered by these contract tests.
    """

    def test_preserves_masked_reference(self) -> None:
        """Match direct weighted arithmetic through mapped and summed paths."""
        spec = make_shard_spec(n_devices=1, chunk_size=2, nk_max=4)
        points, mask = pad_with_mask(
            jnp.asarray([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]]),
            4,
        )

        def body(
            kpoints: Float64[Array, "chunk 3"],
            weights: Float64[Array, " chunk"],
        ) -> Float64[Array, " chunk"]:
            """Return a weighted scalar value for each supplied k point."""
            return weights * kpoints[:, 0]

        mapped = sharded_kmap(
            lambda kpoints, weights: body(kpoints, weights)[:, None],
            points,
            mask,
            spec,
        )
        summed = sharded_ksum(body, points, mask, spec)
        assert jnp.allclose(mapped[:, 0], jnp.asarray([1.0, 2.0, 0.0, 0.0]))
        assert jnp.allclose(summed, 3.0)
