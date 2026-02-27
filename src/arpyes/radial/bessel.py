"""Spherical Bessel functions in JAX.

Extended Summary
----------------
Implements :math:`j_l(x)` using stable low-order seeds and upward
recurrence in a JIT-compatible form. A small-argument limit
:math:`j_l(x) ~ x^l / (2l + 1)!!` avoids divide-by-zero issues at
the origin.
"""

import math

import jax
import jax.numpy as jnp
from beartype import beartype
from jaxtyping import Array, Float, jaxtyped

_SMALL_ARGUMENT: float = 1.0e-8


def _odd_double_factorial(order: int) -> float:
    """Compute odd double factorial ``(order)!!`` for odd ``order``."""
    if order < 1 or order % 2 == 0:
        msg = "order must be a positive odd integer"
        raise ValueError(msg)
    product: int = math.prod(range(1, order + 1, 2))
    return float(product)


@jaxtyped(typechecker=beartype)
def spherical_bessel_jl(
    order: int,
    x: Float[Array, " ..."],
) -> Float[Array, " ..."]:
    """Evaluate spherical Bessel function :math:`j_l(x)`.

    Parameters
    ----------
    order : int
        Non-negative angular momentum order.
    x : Float[Array, " ..."]
        Real argument array.

    Returns
    -------
    values : Float[Array, " ..."]
        ``j_l(x)`` evaluated element-wise with the same shape as ``x``.
    """
    if order < 0:
        msg = "order must be non-negative"
        raise ValueError(msg)

    x_arr: Float[Array, " ..."] = jnp.asarray(x, dtype=jnp.float64)
    small_mask: Float[Array, " ..."] = jnp.abs(x_arr) < _SMALL_ARGUMENT
    x_safe: Float[Array, " ..."] = jnp.where(small_mask, 1.0, x_arr)

    j0_nonzero: Float[Array, " ..."] = jnp.sin(x_safe) / x_safe
    if order == 0:
        return jnp.where(small_mask, 1.0, j0_nonzero)

    j1_nonzero: Float[Array, " ..."] = (
        jnp.sin(x_safe) / (x_safe * x_safe) - jnp.cos(x_safe) / x_safe
    )
    if order == 1:
        j1_limit: Float[Array, " ..."] = x_arr / 3.0
        return jnp.where(small_mask, j1_limit, j1_nonzero)

    def _recurrence_step(
        index: int,
        state: tuple[Float[Array, " ..."], Float[Array, " ..."]],
    ) -> tuple[Float[Array, " ..."], Float[Array, " ..."]]:
        previous, current = state
        index_arr: Float[Array, " "] = jnp.asarray(index, dtype=jnp.float64)
        next_value: Float[Array, " ..."] = (
            ((2.0 * index_arr + 1.0) / x_safe) * current - previous
        )
        return current, next_value

    _, jl_nonzero = jax.lax.fori_loop(
        1,
        order,
        _recurrence_step,
        (j0_nonzero, j1_nonzero),
    )
    double_factorial: float = _odd_double_factorial(2 * order + 1)
    small_limit: Float[Array, " ..."] = (x_arr**order) / double_factorial
    values: Float[Array, " ..."] = jnp.where(
        small_mask, small_limit, jl_nonzero
    )
    return values


__all__: list[str] = ["spherical_bessel_jl"]
