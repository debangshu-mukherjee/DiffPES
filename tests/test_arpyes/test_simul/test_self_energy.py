"""Tests for energy-dependent self-energy evaluation.

Extended Summary
----------------
Validates :func:`arpyes.simul.self_energy.evaluate_self_energy`, which
computes the imaginary part of the electron self-energy (lifetime
broadening gamma) as a function of binding energy. Three evaluation
modes are tested: constant (energy-independent), polynomial (power-series
in energy), and tabulated (piecewise-linear interpolation from energy
nodes). Tests verify correct functional values and, for the constant and
polynomial modes, finite gradients via ``jax.grad`` to ensure the
self-energy is differentiable and usable in inverse fitting pipelines.

Routine Listings
----------------
:class:`TestEvaluateSelfEnergy`
    Tests for evaluate_self_energy.
"""

import jax
import jax.numpy as jnp
import pytest

from arpyes.simul.self_energy import evaluate_self_energy
from arpyes.types import make_self_energy_config


class TestEvaluateSelfEnergy:
    """Tests for :func:`arpyes.simul.self_energy.evaluate_self_energy`.

    Validates the three self-energy evaluation modes -- constant, polynomial,
    and tabulated -- by checking exact functional values at known energy
    points. Also verifies JAX differentiability of the constant and
    polynomial modes by asserting that ``jax.grad`` produces finite
    gradients with respect to the self-energy coefficients.
    """

    def test_constant_mode(self):
        """Verify that constant mode returns the same gamma at every energy.

        Test Logic
        ----------
        1. **Setup**: Create a self-energy config with mode="constant"
           and gamma=0.15 eV. Evaluate at 100 energy points spanning
           [-3, 1] eV.
        2. **Check**: Assert all 100 returned gamma values equal 0.15.

        Asserts
        -------
        Every element of the output array equals 0.15, confirming that
        constant mode ignores energy dependence and returns a uniform
        broadening width, as expected for a simple Lorentzian lifetime.
        """
        config = make_self_energy_config(gamma=0.15, mode="constant")
        energy = jnp.linspace(-3, 1, 100)
        gamma = evaluate_self_energy(energy, config)
        assert jnp.allclose(gamma, 0.15)

    def test_polynomial_mode(self):
        """Verify that polynomial mode evaluates gamma(E) = c0 + c1*E correctly.

        Test Logic
        ----------
        1. **Setup**: Create a self-energy config with mode="polynomial"
           and coefficients [0.05, 0.1] (highest degree first, matching
           ``jnp.polyval`` convention), giving gamma(E) = 0.05*E + 0.1.
           Evaluate at E = [0.0, 1.0, -1.0].
        2. **Check**: Assert the returned values match the analytically
           expected [0.1, 0.15, 0.05] to within atol=1e-10.

        Asserts
        -------
        The polynomial self-energy is evaluated correctly at three
        test energies, confirming that the coefficient ordering and
        polynomial evaluation match ``jnp.polyval`` semantics.
        """
        # gamma(E) = 0.1 + 0.05*E (linear)
        config = make_self_energy_config(
            mode="polynomial",
            coefficients=jnp.array(
                [0.05, 0.1]
            ),  # polyval: highest degree first
        )
        energy = jnp.array([0.0, 1.0, -1.0])
        gamma = evaluate_self_energy(energy, config)
        expected = jnp.array([0.1, 0.15, 0.05])
        assert jnp.allclose(gamma, expected, atol=1e-10)

    def test_tabulated_mode(self):
        """Verify that tabulated mode interpolates gamma between energy nodes.

        Test Logic
        ----------
        1. **Setup**: Create a self-energy config with mode="tabulated",
           energy_nodes=[-3.0, 0.0, 1.0] and coefficients=[0.05, 0.1, 0.2].
           Evaluate at E = [0.0, 0.5], where E=0.0 is an exact node and
           E=0.5 lies midway between nodes 0.0 and 1.0.
        2. **Check exact node**: Assert gamma(0.0) = 0.1 (direct lookup).
        3. **Check interpolated point**: Assert gamma(0.5) = 0.15, the
           linear interpolation midpoint between 0.1 and 0.2.

        Asserts
        -------
        The tabulated self-energy returns exact values at nodes and
        correct linearly interpolated values between nodes, confirming
        that the piecewise-linear interpolation is implemented correctly.
        """
        nodes = jnp.array([-3.0, 0.0, 1.0])
        coeffs = jnp.array([0.05, 0.1, 0.2])
        config = make_self_energy_config(
            mode="tabulated",
            coefficients=coeffs,
            energy_nodes=nodes,
        )
        energy = jnp.array([0.0, 0.5])
        gamma = evaluate_self_energy(energy, config)
        assert float(gamma[0]) == pytest.approx(0.1, abs=1e-10)
        assert float(gamma[1]) == pytest.approx(0.15, abs=1e-10)

    def test_constant_gradient(self):
        """Verify that the gradient w.r.t. constant self-energy coefficient is finite.

        Test Logic
        ----------
        1. **Setup**: Define a scalar loss function that creates a
           constant-mode self-energy config from a single coefficient,
           evaluates it at E = [0.0, 1.0], and returns the sum of the
           resulting gamma values.
        2. **Differentiate**: Call ``jax.grad(loss)(0.1)`` to compute
           the gradient w.r.t. the constant coefficient.
        3. **Check finiteness**: Assert the gradient is finite.

        Asserts
        -------
        The gradient w.r.t. the constant self-energy coefficient is
        finite, confirming that the constant evaluation path is
        differentiable through JAX and suitable for inverse fitting.
        """

        def loss(coeff):
            config = make_self_energy_config(
                mode="constant",
                coefficients=jnp.array([coeff]),
            )
            energy = jnp.array([0.0, 1.0])
            return jnp.sum(evaluate_self_energy(energy, config))

        grad = jax.grad(loss)(jnp.array(0.1))
        assert jnp.isfinite(grad)

    def test_polynomial_gradient(self):
        """Verify that gradients w.r.t. polynomial coefficients are finite.

        Test Logic
        ----------
        1. **Setup**: Define a scalar loss function that creates a
           polynomial-mode self-energy config from a 2-element
           coefficient array [c1, c0], evaluates it at 50 energy points
           in [-1, 1] eV, and returns the sum of the resulting gamma
           values.
        2. **Differentiate**: Call ``jax.grad(loss)`` with initial
           coefficients [0.01, 0.1] to compute the gradient vector.
        3. **Check finiteness**: Assert all gradient components are
           finite.

        Asserts
        -------
        All gradient components w.r.t. the polynomial coefficients are
        finite, confirming that the polynomial evaluation via
        ``jnp.polyval`` is differentiable through JAX. This enables
        fitting energy-dependent self-energy models to experimental
        linewidths.
        """

        def loss(coeffs):
            config = make_self_energy_config(
                mode="polynomial",
                coefficients=coeffs,
            )
            energy = jnp.linspace(-1, 1, 50)
            return jnp.sum(evaluate_self_energy(energy, config))

        grad = jax.grad(loss)(jnp.array([0.01, 0.1]))
        assert jnp.all(jnp.isfinite(grad))
