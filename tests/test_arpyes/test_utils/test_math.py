"""Tests for math utility functions."""

import chex
import jax.numpy as jnp

from arpyes.utils.math import faddeeva, zscore_normalize


class TestFaddeeva(chex.TestCase):
    """Tests for :func:`arpyes.utils.math.faddeeva`.

    Verifies correctness of the Faddeeva function (w(z) = exp(-z^2) erfc(-iz))
    implementation including evaluation on the real axis, the known value at the
    origin, and evaluation on the imaginary axis. Each test is run both with and
    without JIT compilation to ensure consistent behavior.
    """

    @chex.variants(with_jit=True, without_jit=True)
    def test_real_axis(self):
        """Verify that faddeeva returns finite values with correct shape on the real axis.

        Test Logic
        ----------
        1. **Build real-valued input**:
           Create 100 points linearly spaced in [-3, 3] and cast to complex
           by adding 0j.

        2. **Evaluate**:
           Call the faddeeva function (via the chex variant wrapper) on the
           complex array.

        3. **Check output**:
           Assert the result shape is (100,) and that all real parts are
           finite (no NaN or Inf).

        Asserts
        -------
        Output shape matches (100,) and all real components are finite,
        confirming the Taylor-series evaluation is numerically stable on
        the real line within |x| <= 3.
        """
        x = jnp.linspace(-3.0, 3.0, 100)
        z = x + 0j
        var_fn = self.variant(faddeeva)
        w = var_fn(z)
        chex.assert_shape(w, (100,))
        chex.assert_tree_all_finite(jnp.real(w))

    @chex.variants(with_jit=True, without_jit=True)
    def test_zero(self):
        """Verify that faddeeva(0) returns approximately 1.0.

        Test Logic
        ----------
        1. **Build scalar input**:
           Create the complex scalar z = 0 + 0j.

        2. **Evaluate**:
           Call the faddeeva function (via the chex variant wrapper) on the
           scalar input.

        3. **Check known value**:
           The Faddeeva function at the origin satisfies w(0) = erfc(0) = 1.
           Assert that the real part of the result is close to 1.0 within
           an absolute tolerance of 0.05.

        Asserts
        -------
        Real part of w(0) is within 0.05 of 1.0, validating the seed
        coefficient a_0 = 1 of the Taylor expansion.
        """
        z = jnp.array(0.0 + 0j)
        var_fn = self.variant(faddeeva)
        w = var_fn(z)
        chex.assert_trees_all_close(
            jnp.real(w), jnp.float64(1.0), atol=0.05
        )

    @chex.variants(with_jit=True, without_jit=True)
    def test_imaginary_axis(self):
        """Verify that faddeeva returns finite values with correct shape on the imaginary axis.

        Test Logic
        ----------
        1. **Build purely imaginary input**:
           Create an array of three imaginary values z = i * [0.5, 1.0, 2.0].

        2. **Evaluate**:
           Call the faddeeva function (via the chex variant wrapper) on the
           imaginary array.

        3. **Check output**:
           Assert the result shape is (3,) and that all real parts are
           finite (no NaN or Inf).

        Asserts
        -------
        Output shape matches (3,) and all real components are finite,
        confirming numerical stability along the imaginary axis where
        w(iy) grows exponentially as exp(y^2) erfc(y).
        """
        y = jnp.array([0.5, 1.0, 2.0])
        z = 1j * y
        var_fn = self.variant(faddeeva)
        w = var_fn(z)
        chex.assert_shape(w, (3,))
        chex.assert_tree_all_finite(jnp.real(w))


class TestZscoreNormalize(chex.TestCase):
    """Tests for :func:`arpyes.utils.math.zscore_normalize`.

    Verifies correctness of z-score normalization including the standard case
    (mean becomes 0, std becomes 1), the degenerate constant-input case
    (zero-variance guard), and multi-dimensional input support.
    """

    @chex.variants(with_jit=True, without_jit=True)
    def test_normalized_stats(self):
        """Verify that zscore_normalize produces zero mean and unit standard deviation.

        Test Logic
        ----------
        1. **Build a simple 1-D array**:
           Create the float64 array [1, 2, 3, 4, 5] with non-zero variance.

        2. **Normalize**:
           Call zscore_normalize (via the chex variant wrapper) on the data.

        3. **Check statistics**:
           Assert the mean of the result is 0.0 and the standard deviation
           is 1.0, both within an absolute tolerance of 1e-10.

        Asserts
        -------
        Output mean is approximately 0 and output std is approximately 1,
        confirming the core z-score transformation (x - mu) / sigma.
        """
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
        """Verify that zscore_normalize returns all zeros for a constant array.

        Test Logic
        ----------
        1. **Build a constant array**:
           Create a length-10 array of all ones (std = 0).

        2. **Normalize**:
           Call zscore_normalize (via the chex variant wrapper). The function
           should detect the zero standard deviation and use the safe divisor
           of 1.0 to avoid division by zero.

        3. **Check output**:
           Assert the result equals a length-10 zero array within an absolute
           tolerance of 1e-10.

        Asserts
        -------
        Output is all zeros, confirming the zero-variance guard path
        (jnp.where(std > 0, std, 1.0)) produces the correct degenerate result.
        """
        data = jnp.ones(10, dtype=jnp.float64)
        var_fn = self.variant(zscore_normalize)
        result = var_fn(data)
        chex.assert_trees_all_close(
            result, jnp.zeros(10, dtype=jnp.float64), atol=1e-10
        )

    @chex.variants(with_jit=True, without_jit=True)
    def test_2d_input(self):
        """Verify that zscore_normalize handles 2-D arrays with global statistics.

        Test Logic
        ----------
        1. **Build a 2-D array**:
           Create a (3, 4) float array from arange(12) so that the data has
           non-trivial variance across both axes.

        2. **Normalize**:
           Call zscore_normalize (via the chex variant wrapper) on the 2-D
           input. The function computes global (not per-axis) mean and std.

        3. **Check shape and mean**:
           Assert the output shape is preserved as (3, 4) and the global mean
           of the result is approximately 0.0 within an absolute tolerance
           of 1e-10.

        Asserts
        -------
        Output shape is (3, 4) and global mean is approximately 0, confirming
        that zscore_normalize operates element-wise over arbitrary-rank inputs
        using global statistics.
        """
        data = jnp.arange(12.0).reshape(3, 4)
        var_fn = self.variant(zscore_normalize)
        result = var_fn(data)
        chex.assert_shape(result, (3, 4))
        chex.assert_trees_all_close(
            jnp.mean(result), jnp.float64(0.0), atol=1e-10
        )
