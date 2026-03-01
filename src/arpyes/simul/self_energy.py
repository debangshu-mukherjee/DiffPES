"""Energy-dependent self-energy evaluation.

Extended Summary
----------------
Evaluates the imaginary part of the electronic self-energy
(lifetime broadening) as a function of energy. Supports constant,
polynomial, and tabulated modes, all JAX-differentiable.
"""

import jax.numpy as jnp
from beartype import beartype
from jaxtyping import Array, Float, jaxtyped

from arpyes.types import SelfEnergyConfig


@jaxtyped(typechecker=beartype)
def evaluate_self_energy(
    energy: Float[Array, " ..."],
    config: SelfEnergyConfig,
) -> Float[Array, " ..."]:
    r"""Evaluate :math:`\Gamma(E)` from the self-energy config.

    Parameters
    ----------
    energy : Float[Array, " ..."]
        Energy values in eV.
    config : SelfEnergyConfig
        Self-energy model configuration.

    Returns
    -------
    gamma : Float[Array, " ..."]
        Energy-dependent broadening in eV, same shape as ``energy``.
    """
    mode = config.mode

    if mode == "constant":
        return jnp.broadcast_to(config.coefficients[0], energy.shape)
    elif mode == "polynomial":
        return jnp.polyval(config.coefficients, energy)
    elif mode == "tabulated":
        return jnp.interp(energy, config.energy_nodes, config.coefficients)
    else:
        msg = f"Unknown self-energy mode: {mode}"
        raise ValueError(msg)


__all__: list[str] = ["evaluate_self_energy"]
