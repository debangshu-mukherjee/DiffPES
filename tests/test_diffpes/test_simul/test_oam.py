"""Validate orbital angular momentum calculation.

Extended Summary
----------------
The tests exercise :func:`diffpes.simul.compute_oam`. They verify the
``(K, B, A, 3)`` output shape and finite values. They also verify that the
total channel equals the sum of the p and d channels.

"""

import chex
import jax.numpy as jnp
from jaxtyping import Array, Float64

from diffpes.simul import compute_oam


class TestComputeOam(chex.TestCase):
    """Validate :func:`diffpes.simul.oam.compute_oam`.

    The tests verify the output shape and finite p, d, and total contributions.

    :see: :func:`~diffpes.simul.compute_oam`
    """

    def test_output_shape(self) -> None:
        """Verify OAM array has shape (K, B, A, 3).

        The test establishes the output shape contract for compute oam with the
        concrete values and array shapes described below.

        Notes
        -----
        1. **Setup**: Create projections of shape (4, 3, 2, 9)
           (K=4, B=3, A=2) with uniform values.
        2. **Call**: compute_oam(projections).
        3. **Check**: Assert output shape (4, 3, 2, 3) and all finite.

        **Expected assertions**

        OAM shape is (K, B, A, 3) with [p_oam, d_oam, total_oam].
        """
        k: int
        b: int
        a: int
        projections: Float64[Array, "..."]
        oam: Float64[Array, "..."]

        k, b, a = 4, 3, 2
        projections = jnp.ones((k, b, a, 9), dtype=jnp.float64) * 0.1
        oam = compute_oam(projections)
        chex.assert_shape(oam, (k, b, a, 3))
        chex.assert_tree_all_finite(oam)

    def test_total_is_p_plus_d(self) -> None:
        """Verify the total OAM channel equals the p- and d-channel sum.

        The test establishes the p-plus-d contract for compute OAM with the
        concrete values and array shapes described below.

        Notes
        -----
        1. **Setup**: Create projections of shape (2, 2, 1, 9) with
           uniform value 0.2.
        2. **Call**: compute_oam(projections) yields oam with shape
           (2, 2, 1, 3).
        3. **Check**: The third channel (index 2) should equal the sum
           of the first (p) and second (d) channels.

        **Expected assertions**

        oam[..., 0] + oam[..., 1] equals oam[..., 2] to within 1e-12,
        confirming the total OAM is the sum of p and d contributions.
        """
        projections: Float64[Array, "..."]
        oam: Float64[Array, "..."]

        projections = jnp.ones((2, 2, 1, 9), dtype=jnp.float64) * 0.2
        oam = compute_oam(projections)
        chex.assert_trees_all_close(
            oam[..., 0] + oam[..., 1],
            oam[..., 2],
            atol=1e-12,
        )
