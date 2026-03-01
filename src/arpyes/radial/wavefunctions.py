"""Atomic radial wavefunction models in JAX.

Extended Summary
----------------
Provides normalized Slater-type and hydrogenic radial wavefunctions
for use in differentiable ARPES matrix-element calculations.
"""

import math

import jax
import jax.numpy as jnp
from beartype import beartype
from jaxtyping import Array, Float, jaxtyped

from arpyes.types import ScalarFloat


def _associated_laguerre(
    order: int,
    alpha: int,
    x: Float[Array, " ..."],
) -> Float[Array, " ..."]:
    """Evaluate generalized Laguerre polynomial ``L_order^alpha(x)``."""
    if order < 0:
        msg = "order must be non-negative"
        raise ValueError(msg)
    if alpha < 0:
        msg = "alpha must be non-negative"
        raise ValueError(msg)

    x_arr: Float[Array, " ..."] = jnp.asarray(x, dtype=jnp.float64)
    laguerre_zero: Float[Array, " ..."] = jnp.ones_like(x_arr)
    if order == 0:
        return laguerre_zero

    alpha_arr: Float[Array, " "] = jnp.asarray(alpha, dtype=jnp.float64)
    laguerre_one: Float[Array, " ..."] = 1.0 + alpha_arr - x_arr
    if order == 1:
        return laguerre_one

    def _recurrence_step(
        current_order: int,
        state: tuple[Float[Array, " ..."], Float[Array, " ..."]],
    ) -> tuple[Float[Array, " ..."], Float[Array, " ..."]]:
        laguerre_prev_prev, laguerre_prev = state
        order_arr: Float[Array, " "] = jnp.asarray(
            current_order, dtype=jnp.float64
        )
        prefactor: Float[Array, " ..."] = (
            2.0 * order_arr - 1.0 + alpha_arr - x_arr
        ) / order_arr
        correction: Float[Array, " ..."] = (
            (order_arr - 1.0 + alpha_arr) / order_arr
        ) * laguerre_prev_prev
        laguerre_curr: Float[Array, " ..."] = (
            prefactor * laguerre_prev - correction
        )
        return laguerre_prev, laguerre_curr

    _, laguerre_final = jax.lax.fori_loop(
        2,
        order + 1,
        _recurrence_step,
        (laguerre_zero, laguerre_one),
    )
    return laguerre_final


@jaxtyped(typechecker=beartype)
def slater_radial(
    r: Float[Array, " ..."],
    n: int,
    zeta: ScalarFloat,
) -> Float[Array, " ..."]:
    """Evaluate normalized Slater-type radial function.

    Parameters
    ----------
    r : Float[Array, " ..."]
        Radial coordinate in atomic units.
    n : int
        Principal quantum number (``n >= 1``).
    zeta : Float[Array, " "]
        Slater exponent.

    Returns
    -------
    values : Float[Array, " ..."]
        Normalized radial function
        ``R(r) = N r^(n-1) exp(-zeta * r)``.
    """
    if n < 1:
        msg = "n must be >= 1"
        raise ValueError(msg)

    r_arr: Float[Array, " ..."] = jnp.asarray(r, dtype=jnp.float64)
    zeta_arr: Float[Array, " "] = jnp.asarray(zeta, dtype=jnp.float64)
    factorial_term: Float[Array, " "] = jnp.asarray(
        math.factorial(2 * n), dtype=jnp.float64
    )
    norm: Float[Array, " "] = ((2.0 * zeta_arr) ** (n + 0.5)) / jnp.sqrt(
        factorial_term
    )
    values: Float[Array, " ..."] = (
        norm * (r_arr ** (n - 1)) * jnp.exp(-zeta_arr * r_arr)
    )
    return values


@jaxtyped(typechecker=beartype)
def hydrogenic_radial(
    r: Float[Array, " ..."],
    n: int,
    angular_momentum: int,
    z_eff: ScalarFloat,
) -> Float[Array, " ..."]:
    """Evaluate normalized hydrogenic radial function.

    Parameters
    ----------
    r : Float[Array, " ..."]
        Radial coordinate in atomic units.
    n : int
        Principal quantum number.
    angular_momentum : int
        Angular momentum quantum number (``0 <= angular_momentum < n``).
    z_eff : Float[Array, " "]
        Effective nuclear charge.

    Returns
    -------
    values : Float[Array, " ..."]
        ``R_{n,l}(r)`` for hydrogenic orbitals.
    """
    if n < 1:
        msg = "n must be >= 1"
        raise ValueError(msg)
    if angular_momentum < 0 or angular_momentum >= n:
        msg = "angular_momentum must satisfy 0 <= angular_momentum < n"
        raise ValueError(msg)

    r_arr: Float[Array, " ..."] = jnp.asarray(r, dtype=jnp.float64)
    z_arr: Float[Array, " "] = jnp.asarray(z_eff, dtype=jnp.float64)
    n_float: float = float(n)
    rho: Float[Array, " ..."] = 2.0 * z_arr * r_arr / n_float

    laguerre_order: int = n - angular_momentum - 1
    laguerre_alpha: int = 2 * angular_momentum + 1
    laguerre_values: Float[Array, " ..."] = _associated_laguerre(
        laguerre_order, laguerre_alpha, rho
    )

    factorial_ratio: float = math.factorial(laguerre_order) / (
        2.0 * n_float * math.factorial(n + angular_momentum)
    )
    prefactor: Float[Array, " "] = ((2.0 * z_arr) / n_float) ** 1.5
    norm: Float[Array, " "] = prefactor * jnp.sqrt(
        jnp.asarray(factorial_ratio, dtype=jnp.float64)
    )
    values: Float[Array, " ..."] = (
        norm * jnp.exp(-0.5 * rho) * (rho**angular_momentum) * laguerre_values
    )
    return values


__all__: list[str] = ["hydrogenic_radial", "slater_radial"]
