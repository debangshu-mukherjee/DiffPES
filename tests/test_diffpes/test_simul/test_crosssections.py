"""Tests for ARPES cross-section weight functions.

Extended Summary
----------------
Exercises heuristic_weights and yeh_lindau_weights. Heuristic tests
verify the two-regime model (p-enhanced below 50 eV, d-enhanced above),
output shape (9,), and JIT compatibility. Yeh-Lindau tests verify
interpolation at a fixed photon energy, output shape, and non-negative
weights. All test logic and assertions are documented in the docstrings
of each test class and method.

Routine Listings
----------------
:class:`TestHeuristicWeights`
    Tests for heuristic_weights.
:class:`TestYehLindauWeights`
    Tests for yeh_lindau_weights.
"""

import chex
import jax.numpy as jnp

from diffpes.simul.crosssections import (
    heuristic_weights,
    yeh_lindau_weights,
)


class TestHeuristicWeights(chex.TestCase):
    """Tests for :func:`diffpes.simul.crosssections.heuristic_weights`.

    Verifies the two-regime heuristic model that assigns enhanced weights
    to p-orbitals below 50 eV and to d-orbitals above 50 eV, including
    output shape, correct orbital enhancement in each regime, and JIT
    compatibility.
    """

    @chex.variants(with_jit=True, without_jit=True)
    def test_low_energy_p_enhanced(self):
        """Verify that p-orbital weights are enhanced in the low-energy regime.

        Test Logic
        ----------
        1. **Compute weights at 30 eV**:
           Calls ``heuristic_weights(30.0)``, which is below the 50 eV
           threshold and should select the low-energy weight vector.

        2. **Check output shape**:
           Confirms the result has shape ``(9,)`` matching the 9-orbital basis.

        3. **Check p-orbital enhancement**:
           Verifies that p-orbital indices (1, 2, 3) have weight 2.0 and
           the first d-orbital index (4) has weight 1.0.

        Asserts
        -------
        Output shape is ``(9,)``, p-orbitals (indices 1-3) equal 2.0, and
        d-orbital (index 4) equals 1.0.
        """
        var_fn = self.variant(heuristic_weights)
        w = var_fn(30.0)
        chex.assert_shape(w, (9,))
        assert float(w[1]) == 2.0
        assert float(w[2]) == 2.0
        assert float(w[3]) == 2.0
        assert float(w[4]) == 1.0

    @chex.variants(with_jit=True, without_jit=True)
    def test_high_energy_d_enhanced(self):
        """Verify that d-orbital weights are enhanced in the high-energy regime.

        Test Logic
        ----------
        1. **Compute weights at 60 eV**:
           Calls ``heuristic_weights(60.0)``, which is above the 50 eV
           threshold and should select the high-energy weight vector.

        2. **Check p- and d-orbital values**:
           Verifies that p-orbital index 1 has weight 1.0, while d-orbital
           indices 4 and 8 have weight 2.0.

        Asserts
        -------
        p-orbital (index 1) equals 1.0, and d-orbitals (indices 4, 8)
        equal 2.0.
        """
        var_fn = self.variant(heuristic_weights)
        w = var_fn(60.0)
        assert float(w[1]) == 1.0
        assert float(w[4]) == 2.0
        assert float(w[8]) == 2.0


class TestYehLindauWeights(chex.TestCase):
    """Tests for :func:`diffpes.simul.crosssections.yeh_lindau_weights`.

    Verifies the interpolated Yeh-Lindau cross-section weights including
    exact values at tabulation points, correct interpolation at intermediate
    energies, positivity of all weights, and JIT compatibility.
    """

    @chex.variants(with_jit=True, without_jit=True)
    def test_at_20_eV(self):
        """Verify exact cross-section values at the 20 eV tabulation point.

        Test Logic
        ----------
        1. **Compute weights at 20 eV**:
           Calls ``yeh_lindau_weights(20.0)`` at the lowest tabulation
           energy, where no interpolation is needed.

        2. **Check output shape**:
           Confirms the result has shape ``(9,)``.

        3. **Check known tabulated values**:
           Compares the s-orbital weight (index 0) against 0.1, the
           first p-orbital weight (index 1) against 0.6, and the first
           d-orbital weight (index 4) against 2.0, matching the
           _SIGMA_S, _SIGMA_P, and _SIGMA_D tables at 20 eV.

        Asserts
        -------
        Output shape is ``(9,)`` and weights at indices 0, 1, 4 match
        the tabulated values within tolerance 1e-5.
        """
        var_fn = self.variant(yeh_lindau_weights)
        w = var_fn(20.0)
        chex.assert_shape(w, (9,))
        chex.assert_trees_all_close(w[0], jnp.float64(0.1), atol=1e-5)
        chex.assert_trees_all_close(w[1], jnp.float64(0.6), atol=1e-5)
        chex.assert_trees_all_close(w[4], jnp.float64(2.0), atol=1e-5)

    @chex.variants(with_jit=True, without_jit=True)
    def test_interpolated(self):
        """Verify that interpolation at 30 eV produces positive s and p weights.

        Test Logic
        ----------
        1. **Compute weights at 30 eV**:
           Calls ``yeh_lindau_weights(30.0)``, an energy between the
           20 eV and 40 eV tabulation points, requiring interpolation.

        2. **Check output shape**:
           Confirms the result has shape ``(9,)``.

        3. **Check s- and p-orbital weights are positive**:
           Verifies that the interpolated s-orbital (index 0) and
           p-orbital (index 1) weights are strictly greater than zero.

        Asserts
        -------
        Output shape is ``(9,)`` and weights at indices 0 and 1 are
        strictly positive.
        """
        var_fn = self.variant(yeh_lindau_weights)
        w = var_fn(30.0)
        chex.assert_shape(w, (9,))
        assert float(w[0]) > 0.0
        assert float(w[1]) > 0.0

    @chex.variants(with_jit=True, without_jit=True)
    def test_all_positive(self):
        """Verify that all 9 orbital weights are strictly positive at 40 eV.

        Test Logic
        ----------
        1. **Compute weights at 40 eV**:
           Calls ``yeh_lindau_weights(40.0)`` at a tabulation point.

        2. **Check positivity of every element**:
           Iterates over all 9 orbital weight entries and asserts each
           is strictly positive using ``chex.assert_scalar_positive``.

        Asserts
        -------
        Every element of the 9-element weight vector is strictly positive,
        ensuring no orbital receives a zero or negative cross-section.
        """
        var_fn = self.variant(yeh_lindau_weights)
        w = var_fn(40.0)
        for i in range(9):
            chex.assert_scalar_positive(float(w[i]))
