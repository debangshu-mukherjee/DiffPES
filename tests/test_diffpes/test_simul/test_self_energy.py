"""Validate the temporary WP7.3 linewidth evaluator compatibility seam.

The tests cover the three modes that the compatibility evaluator implements.
"""

import jax
import jax.numpy as jnp

from diffpes.simul import evaluate_self_energy
from diffpes.types import SelfEnergyModel, make_self_energy_model

_DOMAIN: jax.Array = jnp.array([-4.0, 4.0])
_TAIL: jax.Array = jnp.zeros(2)


class TestEvaluateSelfEnergy:
    """Test :func:`diffpes.simul.evaluate_self_energy` behavior.

    The evaluator tests below verify values and gradients for supported modes.

    :see: :func:`~diffpes.simul.evaluate_self_energy`
    """


def test_constant_mode_and_gradient() -> None:
    """Evaluate the gamma shortcut and retain its coefficient gradient.

    The test covers the constant value and its finite gradient.

    Notes
    -----
    The test evaluates a grid and differentiates its summed linewidth.
    """
    energy: jax.Array = jnp.linspace(-3.0, 1.0, 100)
    model: SelfEnergyModel = make_self_energy_model(
        gamma=0.15, kk_consistent=False
    )
    assert jnp.allclose(evaluate_self_energy(energy, model), 0.15)

    def loss(raw: jax.Array) -> jax.Array:
        current: SelfEnergyModel = make_self_energy_model(
            coefficients=jnp.atleast_1d(raw), kk_consistent=False
        )
        return jnp.sum(evaluate_self_energy(energy, current))

    assert jnp.isfinite(jax.grad(loss)(jnp.array(0.1)))


def test_poly_mode_and_gradient() -> None:
    """Apply softplus to the highest-degree-first polynomial.

    The test covers the polynomial values and coefficient gradients.

    Notes
    -----
    The test compares an explicit formula and differentiates its sum.
    """
    energy: jax.Array = jnp.array([-1.0, 0.0, 1.0])
    raw: jax.Array = jnp.array([0.05, 0.1])

    def loss(coefficients: jax.Array) -> jax.Array:
        model: SelfEnergyModel = make_self_energy_model(
            mode="poly",
            coefficients=coefficients,
            kk_domain_rel_fermi_ev=_DOMAIN,
            tail_coefficients=_TAIL,
            tail_mode="power2",
        )
        return jnp.sum(evaluate_self_energy(energy, model))

    assert jnp.allclose(
        loss(raw), jnp.sum(jax.nn.softplus(jnp.polyval(raw, energy)))
    )
    assert jnp.all(jnp.isfinite(jax.grad(loss)(raw)))


def test_grid_mode() -> None:
    """Interpolate smoothly reparameterized grid linewidths.

    The test covers interpolation at one node and one midpoint.

    Notes
    -----
    The test compares the evaluator with a direct JAX interpolation.
    """
    nodes: jax.Array = jnp.array([-4.0, 0.0, 4.0])
    raw: jax.Array = jnp.array([-2.0, -1.0, 0.0])
    model: SelfEnergyModel = make_self_energy_model(
        mode="grid",
        coefficients=raw,
        energy_nodes_rel_fermi_ev=nodes,
        kk_domain_rel_fermi_ev=_DOMAIN,
        tail_coefficients=_TAIL,
        tail_mode="power2",
    )
    energy: jax.Array = jnp.array([0.0, 2.0])
    expected: jax.Array = jnp.interp(energy, nodes, jax.nn.softplus(raw))
    assert jnp.allclose(evaluate_self_energy(energy, model), expected)


TestEvaluateSelfEnergy.test_constant_mode_and_gradient = staticmethod(
    test_constant_mode_and_gradient
)
TestEvaluateSelfEnergy.test_poly_mode_and_gradient = staticmethod(
    test_poly_mode_and_gradient
)
TestEvaluateSelfEnergy.test_grid_mode = staticmethod(test_grid_mode)

del test_constant_mode_and_gradient
del test_poly_mode_and_gradient
del test_grid_mode
