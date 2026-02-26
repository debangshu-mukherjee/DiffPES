"""Tests for cross-section weight functions."""

import chex
import jax.numpy as jnp

from arpyes.simulate.crosssections import (
    heuristic_weights,
    yeh_lindau_weights,
)


class TestHeuristicWeights(chex.TestCase):

    @chex.variants(with_jit=True, without_jit=True)
    def test_low_energy_p_enhanced(self):
        var_fn = self.variant(heuristic_weights)
        w = var_fn(30.0)
        chex.assert_shape(w, (9,))
        assert float(w[1]) == 2.0
        assert float(w[2]) == 2.0
        assert float(w[3]) == 2.0
        assert float(w[4]) == 1.0

    @chex.variants(with_jit=True, without_jit=True)
    def test_high_energy_d_enhanced(self):
        var_fn = self.variant(heuristic_weights)
        w = var_fn(60.0)
        assert float(w[1]) == 1.0
        assert float(w[4]) == 2.0
        assert float(w[8]) == 2.0


class TestYehLindauWeights(chex.TestCase):

    @chex.variants(with_jit=True, without_jit=True)
    def test_at_20_eV(self):
        var_fn = self.variant(yeh_lindau_weights)
        w = var_fn(20.0)
        chex.assert_shape(w, (9,))
        chex.assert_trees_all_close(
            w[0], jnp.float64(0.1), atol=1e-5
        )
        chex.assert_trees_all_close(
            w[1], jnp.float64(0.6), atol=1e-5
        )
        chex.assert_trees_all_close(
            w[4], jnp.float64(2.0), atol=1e-5
        )

    @chex.variants(with_jit=True, without_jit=True)
    def test_interpolated(self):
        var_fn = self.variant(yeh_lindau_weights)
        w = var_fn(30.0)
        chex.assert_shape(w, (9,))
        assert float(w[0]) > 0.0
        assert float(w[1]) > 0.0

    @chex.variants(with_jit=True, without_jit=True)
    def test_all_positive(self):
        var_fn = self.variant(yeh_lindau_weights)
        w = var_fn(40.0)
        for i in range(9):
            chex.assert_scalar_positive(float(w[i]))
