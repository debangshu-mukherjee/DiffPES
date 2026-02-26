"""Tests for broadening functions."""

import chex
import jax.numpy as jnp

from arpyes.simul.broadening import (
    fermi_dirac,
    gaussian,
    voigt,
)


class TestGaussian(chex.TestCase):
    """Tests for :func:`arpyes.simul.broadening.gaussian`.

    Verifies the normalized Gaussian broadening profile, including
    normalization (unit integral), peak position accuracy, and
    symmetry about the center energy.
    """

    @chex.variants(with_jit=True, without_jit=True)
    def test_normalization(self):
        """Verify that the Gaussian profile integrates to unity.

        Test Logic
        ----------
        1. **Create dense energy grid**:
           A wide energy range [-10, 10] eV with 100,000 points ensures
           accurate numerical integration via the rectangle rule.

        2. **Evaluate Gaussian**:
           Computes the profile centered at 0.0 eV with sigma = 0.5 eV.

        3. **Numerical integration**:
           Sums profile values multiplied by the energy step size to
           approximate the integral.

        Asserts
        -------
        The numerical integral is within 1e-3 of 1.0, confirming
        proper normalization of the Gaussian density.
        """
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
        """Verify that the Gaussian peak occurs at the specified center energy.

        Test Logic
        ----------
        1. **Create energy grid**:
           A symmetric range [-5, 5] eV with 10,001 points gives a step
           size of 0.001 eV for precise peak location.

        2. **Evaluate Gaussian**:
           Computes the profile centered at 1.5 eV with sigma = 0.3 eV.

        3. **Locate peak**:
           Finds the energy corresponding to the maximum profile value
           using ``jnp.argmax``.

        Asserts
        -------
        The peak energy is within 0.01 eV of the requested center (1.5 eV),
        confirming the centering parameter works correctly.
        """
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
        """Verify that the Gaussian profile is symmetric about its center.

        Test Logic
        ----------
        1. **Create symmetric energy grid**:
           A range [-5, 5] eV centered at zero with an odd number of
           points (1001) ensures the center point lies exactly at 0.0 eV.

        2. **Evaluate Gaussian**:
           Computes the profile centered at 0.0 eV with sigma = 0.5 eV.

        3. **Compare with reversed profile**:
           Checks that the profile array equals its reverse, which holds
           for a symmetric function on a symmetric grid.

        Asserts
        -------
        Each element matches its mirror element to within 1e-10,
        confirming the even-function symmetry G(-E) = G(E).
        """
        e_range = jnp.linspace(-5.0, 5.0, 1001)
        var_fn = self.variant(gaussian)
        profile = var_fn(e_range, 0.0, 0.5)
        chex.assert_trees_all_close(
            profile, profile[::-1], atol=1e-10
        )


class TestVoigt(chex.TestCase):
    """Tests for :func:`arpyes.simul.broadening.voigt`.

    Verifies the pseudo-Voigt broadening profile, including its
    limiting behavior (reduction to Gaussian when gamma approaches zero),
    peak position accuracy, and output finiteness.
    """

    @chex.variants(with_jit=True, without_jit=True)
    def test_reduces_to_gaussian(self):
        """Verify that the Voigt profile reduces to a Gaussian when gamma is negligible.

        Test Logic
        ----------
        1. **Set near-zero Lorentzian width**:
           Uses gamma = 1e-10 eV so the Lorentzian contribution vanishes,
           leaving only the Gaussian component.

        2. **Evaluate both profiles**:
           Computes the Voigt profile and a pure Gaussian with the same
           sigma on the same energy grid.

        3. **Compare element-wise**:
           Checks that the Voigt output matches the pure Gaussian.

        Asserts
        -------
        All values agree to within 1e-3, confirming the correct
        Gaussian limiting behavior of the pseudo-Voigt approximation.
        """
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
        """Verify that the Voigt profile peaks at the specified center energy.

        Test Logic
        ----------
        1. **Create energy grid**:
           A range [-5, 5] eV with 10,001 points for sub-meV resolution.

        2. **Evaluate Voigt profile**:
           Computes the profile centered at -1.0 eV with sigma = 0.3 eV
           and gamma = 0.1 eV (mixed Gaussian-Lorentzian regime).

        3. **Locate peak**:
           Finds the energy corresponding to the maximum profile value.

        Asserts
        -------
        The peak energy is within 0.01 eV of the requested center (-1.0 eV),
        confirming that both Gaussian and Lorentzian components share the
        same center.
        """
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
        """Verify that the Voigt profile produces finite values everywhere.

        Test Logic
        ----------
        1. **Evaluate Voigt profile**:
           Computes the profile on [-5, 5] eV with sigma = 0.5 eV and
           gamma = 0.2 eV, a typical mixed-broadening regime.

        2. **Check finiteness**:
           Ensures no NaN or infinity values appear in the output, which
           could arise from division-by-zero in the Lorentzian component
           or overflow in intermediate calculations.

        Asserts
        -------
        All profile values are finite (no NaN or Inf), confirming
        numerical stability of the pseudo-Voigt implementation.
        """
        e_range = jnp.linspace(-5.0, 5.0, 1001)
        var_fn = self.variant(voigt)
        profile = var_fn(e_range, 0.0, 0.5, 0.2)
        chex.assert_tree_all_finite(profile)


class TestFermiDirac(chex.TestCase):
    """Tests for :func:`arpyes.simul.broadening.fermi_dirac`.

    Verifies the Fermi-Dirac thermal distribution function, including
    the exact value at the Fermi level, asymptotic limits deep below
    and high above the Fermi energy, and the bounded output range [0, 1].
    """

    @chex.variants(with_jit=True, without_jit=True)
    def test_at_fermi_level(self):
        """Verify that the Fermi-Dirac function equals 0.5 at the Fermi energy.

        Test Logic
        ----------
        1. **Evaluate at E = Ef**:
           Calls ``fermi_dirac(0.0, 0.0, 300.0)`` where energy and Fermi
           level are both 0.0 eV at room temperature (300 K).

        2. **Check analytic result**:
           At E = Ef the exponent is zero, so f(Ef) = 1/(1+1) = 0.5
           regardless of temperature.

        Asserts
        -------
        The result is within 1e-5 of 0.5, confirming the fundamental
        property f(Ef) = 0.5 at finite temperature.
        """
        var_fn = self.variant(fermi_dirac)
        result = var_fn(0.0, 0.0, 300.0)
        chex.assert_trees_all_close(
            result, jnp.float64(0.5), atol=1e-5
        )

    @chex.variants(with_jit=True, without_jit=True)
    def test_deep_below_fermi(self):
        """Verify that states deep below the Fermi level are fully occupied.

        Test Logic
        ----------
        1. **Evaluate far below Ef**:
           Calls ``fermi_dirac(-5.0, 0.0, 15.0)`` where E - Ef = -5.0 eV
           at T = 15 K (kT ~ 1.3 meV), making the exponent ~ -3900.

        2. **Check saturation**:
           For large negative exponents, exp(x) approaches 0 and the
           occupation approaches 1/(1+0) = 1.

        Asserts
        -------
        The result is within 1e-5 of 1.0, confirming full occupation
        of deeply bound states.
        """
        var_fn = self.variant(fermi_dirac)
        result = var_fn(-5.0, 0.0, 15.0)
        chex.assert_trees_all_close(
            result, jnp.float64(1.0), atol=1e-5
        )

    @chex.variants(with_jit=True, without_jit=True)
    def test_high_above_fermi(self):
        """Verify that states far above the Fermi level are unoccupied.

        Test Logic
        ----------
        1. **Evaluate far above Ef**:
           Calls ``fermi_dirac(5.0, 0.0, 15.0)`` where E - Ef = +5.0 eV
           at T = 15 K, making the exponent ~ +3900.

        2. **Check vanishing occupation**:
           For large positive exponents, exp(x) dominates and the
           occupation approaches 1/(1+inf) = 0.

        Asserts
        -------
        The result is within 1e-5 of 0.0, confirming that states
        well above the Fermi energy are effectively empty.
        """
        var_fn = self.variant(fermi_dirac)
        result = var_fn(5.0, 0.0, 15.0)
        chex.assert_trees_all_close(
            result, jnp.float64(0.0), atol=1e-5
        )

    @chex.variants(with_jit=True, without_jit=True)
    def test_range_0_to_1(self):
        """Verify that the Fermi-Dirac output lies within [0, 1].

        Test Logic
        ----------
        1. **Evaluate at a typical energy**:
           Calls ``fermi_dirac(-0.5, 0.0, 300.0)`` at E = -0.5 eV
           relative to Ef = 0.0 eV at room temperature (300 K), where
           the result should be between 0 and 1 (closer to 1).

        2. **Bound checks**:
           Asserts the scalar result is non-negative and does not
           exceed unity using plain Python comparisons.

        Asserts
        -------
        The occupation value satisfies 0 <= f(E) <= 1, which must
        hold for any valid probability/occupation function.
        """
        var_fn = self.variant(fermi_dirac)
        result = var_fn(-0.5, 0.0, 300.0)
        assert float(result) >= 0.0
        assert float(result) <= 1.0
