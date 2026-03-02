"""Tests for orbital angular momentum calculation.

Extended Summary
----------------
Exercises :func:`diffpes.simul.compute_oam`. Verifies that the output
array has shape (K, B, A, 3) with channels [p_oam, d_oam, total_oam],
that all values are finite, and that the third channel equals the sum
of the first two. All test logic and assertions are documented in the
docstrings of each test class and method.

Routine Listings
----------------
:class:`TestComputeOam`
    Tests for compute_oam.
"""

import chex
import jax.numpy as jnp

from diffpes.simul import compute_oam


class TestComputeOam(chex.TestCase):
    """Tests for :func:`diffpes.simul.oam.compute_oam`.

    Verifies output shape (K, B, A, 3) and that p/d/total
    contributions are computed and finite.
    """

    def test_output_shape(self):
        """Verify OAM array has shape (K, B, A, 3).

        Test Logic
        ----------
        1. **Setup**: Create projections of shape (4, 3, 2, 9)
           (K=4, B=3, A=2) with uniform values.
        2. **Call**: compute_oam(projections).
        3. **Check**: Assert output shape (4, 3, 2, 3) and all finite.

        Asserts
        -------
        OAM shape is (K, B, A, 3) with [p_oam, d_oam, total_oam].
        """
        k, b, a = 4, 3, 2
        projections = jnp.ones((k, b, a, 9), dtype=jnp.float64) * 0.1
        oam = compute_oam(projections)
        chex.assert_shape(oam, (k, b, a, 3))
        chex.assert_tree_all_finite(oam)

    def test_total_is_p_plus_d(self):
        """Verify that the total OAM channel equals the sum of p and d channels.

        Test Logic
        ----------
        1. **Setup**: Create projections of shape (2, 2, 1, 9) with
           uniform value 0.2.
        2. **Call**: compute_oam(projections) yields oam with shape
           (2, 2, 1, 3).
        3. **Check**: The third channel (index 2) should equal the sum
           of the first (p) and second (d) channels.

        Asserts
        -------
        oam[..., 0] + oam[..., 1] equals oam[..., 2] to within 1e-12,
        confirming the total OAM is the sum of p and d contributions.
        """
        projections = jnp.ones((2, 2, 1, 9), dtype=jnp.float64) * 0.2
        oam = compute_oam(projections)
        chex.assert_trees_all_close(
            oam[..., 0] + oam[..., 1],
            oam[..., 2],
            atol=1e-12,
        )
