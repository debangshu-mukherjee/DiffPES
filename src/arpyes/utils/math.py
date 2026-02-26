"""Mathematical utility functions for ARPES simulations.

Extended Summary
----------------
Provides JAX-compatible implementations of the Faddeeva function
(complex error function) and data normalization routines used
throughout the ARPES simulation pipeline.

Routine Listings
----------------
:func:`faddeeva`
    Faddeeva function via Taylor series expansion.
:func:`zscore_normalize`
    Z-score (zero-mean, unit-variance) normalization.

Notes
-----
The Faddeeva implementation uses a 64-term Taylor series
derived from the ODE w'(z) = -2z w(z) + 2i/sqrt(pi),
giving double-precision accuracy for |z| < 6.
"""

import math

import jax.numpy as jnp
from beartype import beartype
from jaxtyping import Array, Complex, Float, jaxtyped

_N_TAYLOR: int = 64


def _faddeeva_taylor_coeffs() -> list[complex]:
    """Taylor coefficients of w(z) = exp(-z^2) erfc(-iz).

    Uses the recurrence from the ODE
    w'(z) = -2z w(z) + 2i/sqrt(pi):
      a_0 = 1,  a_1 = 2i/sqrt(pi),
      a_{n+1} = -2 a_{n-1} / (n+1)  for n >= 1.
    """
    c: list[complex] = [0j] * _N_TAYLOR
    c[0] = 1.0 + 0j
    c[1] = 2.0j / math.sqrt(math.pi)
    for n in range(1, _N_TAYLOR - 1):
        c[n + 1] = -2.0 * c[n - 1] / (n + 1)
    return c


_W_POLY: Complex[Array, " N"] = jnp.array(
    _faddeeva_taylor_coeffs()[::-1],
    dtype=jnp.complex128,
)


@jaxtyped(typechecker=beartype)
def faddeeva(
    z: Complex[Array, " ..."],
) -> Complex[Array, " ..."]:
    """Evaluate the Faddeeva function w(z) = exp(-z^2) erfc(-iz).

    Parameters
    ----------
    z : Complex[Array, " ..."]
        Complex argument(s).

    Returns
    -------
    w : Complex[Array, " ..."]
        Faddeeva function values.

    Notes
    -----
    Uses a 64-term Taylor series derived from the ODE
    w'(z) = -2z w(z) + 2i/sqrt(pi). Accurate to double
    precision for |z| < 6.
    """
    z_c: Complex[Array, " ..."] = jnp.asarray(
        z, dtype=jnp.complex128
    )
    w: Complex[Array, " ..."] = jnp.polyval(
        _W_POLY, z_c, unroll=8
    )
    return w


@jaxtyped(typechecker=beartype)
def zscore_normalize(
    data: Float[Array, " ..."],
) -> Float[Array, " ..."]:
    """Apply z-score normalization (zero-mean, unit-variance).

    Parameters
    ----------
    data : Float[Array, " ..."]
        Input data array.

    Returns
    -------
    normalized : Float[Array, " ..."]
        Normalized data with mean 0 and standard deviation 1.
    """
    mean_val: Float[Array, " "] = jnp.mean(data)
    std_val: Float[Array, " "] = jnp.std(data)
    safe_std: Float[Array, " "] = jnp.where(
        std_val > 0.0, std_val, 1.0
    )
    normalized: Float[Array, " ..."] = (
        data - mean_val
    ) / safe_std
    return normalized


__all__: list[str] = [
    "faddeeva",
    "zscore_normalize",
]
