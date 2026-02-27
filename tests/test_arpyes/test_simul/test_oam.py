"""Tests for orbital angular momentum calculation."""

import chex
import jax.numpy as jnp

from arpyes.simul import compute_oam


class TestComputeOam(chex.TestCase):
    """Tests for :func:`arpyes.simul.oam.compute_oam`.

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
        """Verify third channel is sum of first two (p + d = total)."""
        projections = jnp.ones((2, 2, 1, 9), dtype=jnp.float64) * 0.2
        oam = compute_oam(projections)
        chex.assert_trees_all_close(
            oam[..., 0] + oam[..., 1],
            oam[..., 2],
            atol=1e-12,
        )
