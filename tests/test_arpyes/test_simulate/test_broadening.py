"""Tests for broadening functions."""

import chex
import jax.numpy as jnp

from arpyes.simulate.broadening import (
    fermi_dirac,
    gaussian,
    voigt,
)


class TestGaussian(chex.TestCase):

    @chex.variants(with_jit=True, without_jit=True)
    def test_normalization(self):
        e_range = jnp.linspace(-10.0, 10.0, 100000)
        sigma = 0.5
        var_fn = self.variant(gaussian)
        profile = var_fn(e_range, 0.0, sigma)
        de = e_range[1] - e_range[0]
        integral = jnp.sum(profile) * de
        chex.assert_trees_all_close(
            integral, jnp.float64(1.0), atol=1e-3
        )

    @chex.variants(with_jit=True, without_jit=True)
    def test_peak_position(self):
        e_range = jnp.linspace(-5.0, 5.0, 10001)
        center = 1.5
        var_fn = self.variant(gaussian)
        profile = var_fn(e_range, center, 0.3)
        peak_idx = jnp.argmax(profile)
        peak_energy = e_range[peak_idx]
        chex.assert_trees_all_close(
            peak_energy, jnp.float64(center), atol=0.01
        )

    @chex.variants(with_jit=True, without_jit=True)
    def test_symmetry(self):
        e_range = jnp.linspace(-5.0, 5.0, 1001)
        var_fn = self.variant(gaussian)
        profile = var_fn(e_range, 0.0, 0.5)
        chex.assert_trees_all_close(
            profile, profile[::-1], atol=1e-10
        )


class TestVoigt(chex.TestCase):

    @chex.variants(with_jit=True, without_jit=True)
    def test_reduces_to_gaussian(self):
        e_range = jnp.linspace(-5.0, 5.0, 10001)
        sigma = 0.5
        gamma = 1e-10
        var_fn = self.variant(voigt)
        v_profile = var_fn(e_range, 0.0, sigma, gamma)
        g_profile = gaussian(e_range, 0.0, sigma)
        chex.assert_trees_all_close(
            v_profile, g_profile, atol=1e-3
        )

    @chex.variants(with_jit=True, without_jit=True)
    def test_peak_position(self):
        e_range = jnp.linspace(-5.0, 5.0, 10001)
        center = -1.0
        var_fn = self.variant(voigt)
        profile = var_fn(e_range, center, 0.3, 0.1)
        peak_idx = jnp.argmax(profile)
        peak_energy = e_range[peak_idx]
        chex.assert_trees_all_close(
            peak_energy, jnp.float64(center), atol=0.01
        )

    @chex.variants(with_jit=True, without_jit=True)
    def test_positive_values(self):
        e_range = jnp.linspace(-5.0, 5.0, 1001)
        var_fn = self.variant(voigt)
        profile = var_fn(e_range, 0.0, 0.5, 0.2)
        chex.assert_tree_all_finite(profile)


class TestFermiDirac(chex.TestCase):

    @chex.variants(with_jit=True, without_jit=True)
    def test_at_fermi_level(self):
        var_fn = self.variant(fermi_dirac)
        result = var_fn(0.0, 0.0, 300.0)
        chex.assert_trees_all_close(
            result, jnp.float64(0.5), atol=1e-5
        )

    @chex.variants(with_jit=True, without_jit=True)
    def test_deep_below_fermi(self):
        var_fn = self.variant(fermi_dirac)
        result = var_fn(-5.0, 0.0, 15.0)
        chex.assert_trees_all_close(
            result, jnp.float64(1.0), atol=1e-5
        )

    @chex.variants(with_jit=True, without_jit=True)
    def test_high_above_fermi(self):
        var_fn = self.variant(fermi_dirac)
        result = var_fn(5.0, 0.0, 15.0)
        chex.assert_trees_all_close(
            result, jnp.float64(0.0), atol=1e-5
        )

    @chex.variants(with_jit=True, without_jit=True)
    def test_range_0_to_1(self):
        var_fn = self.variant(fermi_dirac)
        result = var_fn(-0.5, 0.0, 300.0)
        assert float(result) >= 0.0
        assert float(result) <= 1.0
