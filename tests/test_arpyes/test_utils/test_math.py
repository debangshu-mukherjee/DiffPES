"""Tests for math utility functions."""

import chex
import jax.numpy as jnp

from arpyes.utils.math import faddeeva, zscore_normalize


class TestFaddeeva(chex.TestCase):

    @chex.variants(with_jit=True, without_jit=True)
    def test_real_axis(self):
        x = jnp.linspace(-3.0, 3.0, 100)
        z = x + 0j
        var_fn = self.variant(faddeeva)
        w = var_fn(z)
        chex.assert_shape(w, (100,))
        chex.assert_tree_all_finite(jnp.real(w))

    @chex.variants(with_jit=True, without_jit=True)
    def test_zero(self):
        z = jnp.array(0.0 + 0j)
        var_fn = self.variant(faddeeva)
        w = var_fn(z)
        chex.assert_trees_all_close(
            jnp.real(w), jnp.float64(1.0), atol=0.05
        )

    @chex.variants(with_jit=True, without_jit=True)
    def test_imaginary_axis(self):
        y = jnp.array([0.5, 1.0, 2.0])
        z = 1j * y
        var_fn = self.variant(faddeeva)
        w = var_fn(z)
        chex.assert_shape(w, (3,))
        chex.assert_tree_all_finite(jnp.real(w))


class TestZscoreNormalize(chex.TestCase):

    @chex.variants(with_jit=True, without_jit=True)
    def test_normalized_stats(self):
        data = jnp.array(
            [1.0, 2.0, 3.0, 4.0, 5.0], dtype=jnp.float64
        )
        var_fn = self.variant(zscore_normalize)
        result = var_fn(data)
        chex.assert_trees_all_close(
            jnp.mean(result), jnp.float64(0.0), atol=1e-10
        )
        chex.assert_trees_all_close(
            jnp.std(result), jnp.float64(1.0), atol=1e-10
        )

    @chex.variants(with_jit=True, without_jit=True)
    def test_constant_input(self):
        data = jnp.ones(10, dtype=jnp.float64)
        var_fn = self.variant(zscore_normalize)
        result = var_fn(data)
        chex.assert_trees_all_close(
            result, jnp.zeros(10, dtype=jnp.float64), atol=1e-10
        )

    @chex.variants(with_jit=True, without_jit=True)
    def test_2d_input(self):
        data = jnp.arange(12.0).reshape(3, 4)
        var_fn = self.variant(zscore_normalize)
        result = var_fn(data)
        chex.assert_shape(result, (3, 4))
        chex.assert_trees_all_close(
            jnp.mean(result), jnp.float64(0.0), atol=1e-10
        )
