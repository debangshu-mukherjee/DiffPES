"""Spherical Bessel functions in JAX.

Extended Summary
----------------
Implements :math:`j_l(x)` using stable low-order seeds and upward
recurrence in a JIT-compatible form. A small-argument limit
:math:`j_l(x) ~ x^l / (2l + 1)!!` avoids divide-by-zero issues at
the origin.

Routine Listings
----------------
:func:`spherical_bessel_jl`
    Evaluate spherical Bessel function j_l(x).
"""

import math

import jax
import jax.numpy as jnp
from beartype import beartype
from jaxtyping import Array, Float, jaxtyped

_SMALL_ARGUMENT: float = 1.0e-8


def _odd_double_factorial(order: int) -> float:
    r"""Compute odd double factorial ``(order)!!`` for odd ``order``.

    Extended Summary
    ----------------
    Evaluates the double factorial :math:`n!!` for odd positive
    integers, defined as:

    .. math::

        n!! = 1 \cdot 3 \cdot 5 \cdots n = \prod_{k=0}^{(n-1)/2} (2k + 1)

    This is used as the denominator in the small-argument limit of
    the spherical Bessel function:

    .. math::

        j_l(x) \approx \frac{x^l}{(2l+1)!!} \quad \text{for } |x| \ll 1

    The computation uses ``math.prod`` over ``range(1, order+1, 2)``
    for exact integer arithmetic, then casts to float.

    Parameters
    ----------
    order : int
        A positive odd integer.

    Returns
    -------
    value : float
        The double factorial ``order!!``.

    Raises
    ------
    ValueError
        If ``order`` is not a positive odd integer.
    """
    if order < 1 or order % 2 == 0:
        msg: str = "order must be a positive odd integer"
        raise ValueError(msg)
    product: int = math.prod(range(1, order + 1, 2))
    value: float = float(product)
    return value


@jaxtyped(typechecker=beartype)
def spherical_bessel_jl(
    order: int,
    x: Float[Array, " ..."],
) -> Float[Array, " ..."]:
    r"""Evaluate spherical Bessel function :math:`j_l(x)`.

    Extended Summary
    ----------------
    Computes the spherical Bessel function of the first kind
    :math:`j_l(x)` using closed-form seeds for l=0 and l=1, followed
    by upward recurrence for l >= 2.

    **Seed values:**

    .. math::

        j_0(x) = \frac{\sin x}{x}

        j_1(x) = \frac{\sin x}{x^2} - \frac{\cos x}{x}

    **Upward recurrence** (for l >= 2, starting from l=1):

    .. math::

        j_{l+1}(x) = \frac{2l + 1}{x} \, j_l(x) - j_{l-1}(x)

    This is implemented via ``jax.lax.fori_loop`` for JIT
    compatibility. The loop runs from index 1 to ``order - 1``,
    carrying the pair :math:`(j_{l-1}, j_l)` and producing
    :math:`j_{l+1}` at each step.

    **Small-argument limit:**

    For :math:`|x| < 10^{-8}` (the module constant ``_SMALL_ARGUMENT``),
    the function uses the leading-order Taylor expansion:

    .. math::

        j_l(x) \approx \frac{x^l}{(2l+1)!!}

    to avoid the divide-by-zero singularity in the seed formulas.
    A boolean mask ``small_mask`` selects between the recurrence
    result and the small-argument limit element-wise via
    ``jnp.where``. The "safe" input ``x_safe`` replaces masked-out
    entries with 1.0 so that the recurrence branch does not produce
    NaN or Inf values that could pollute gradients.

    **Numerical stability notes:**

    - Upward recurrence for :math:`j_l` is stable because
      :math:`j_l(x)` is the dominant solution of the recurrence
      relation for moderate x. For very large l relative to x,
      downward (Miller) recurrence would be more stable, but this
      regime is not encountered in ARPES simulations where l <= 5
      and kr is typically O(1)--O(10).
    - The ``jnp.where``-based masking ensures that gradients are
      well-defined everywhere, including at x = 0.

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

    Notes
    -----
    The ``order`` parameter is a Python-level integer (not a JAX
    tracer), so changing it requires re-tracing. This is by design
    because the loop bounds and the double-factorial constant depend
    on ``order`` statically.
    """
    if order < 0:
        msg: str = "order must be non-negative"
        raise ValueError(msg)

    x_arr: Float[Array, " ..."] = jnp.asarray(x, dtype=jnp.float64)
    small_mask: Float[Array, " ..."] = jnp.abs(x_arr) < _SMALL_ARGUMENT
    x_safe: Float[Array, " ..."] = jnp.where(small_mask, 1.0, x_arr)

    j0_nonzero: Float[Array, " ..."] = jnp.sin(x_safe) / x_safe
    if order == 0:
        j0_result: Float[Array, " ..."] = jnp.where(
            small_mask, 1.0, j0_nonzero
        )
        return j0_result

    j1_nonzero: Float[Array, " ..."] = (
        jnp.sin(x_safe) / (x_safe * x_safe) - jnp.cos(x_safe) / x_safe
    )
    if order == 1:
        j1_limit: Float[Array, " ..."] = x_arr / 3.0
        j1_result: Float[Array, " ..."] = jnp.where(
            small_mask, j1_limit, j1_nonzero
        )
        return j1_result

    def _recurrence_step(
        index: int,
        state: tuple[Float[Array, " ..."], Float[Array, " ..."]],
    ) -> tuple[Float[Array, " ..."], Float[Array, " ..."]]:
        previous: Float[Array, " ..."]
        current: Float[Array, " ..."]
        previous, current = state
        index_arr: Float[Array, " "] = jnp.asarray(index, dtype=jnp.float64)
        next_value: Float[Array, " ..."] = (
            (2.0 * index_arr + 1.0) / x_safe
        ) * current - previous
        return current, next_value

    jl_nonzero: Float[Array, " ..."]
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
