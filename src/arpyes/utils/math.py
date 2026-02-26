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

    Computes the first ``_N_TAYLOR`` coefficients of the Taylor expansion
    of the Faddeeva function about the origin using a three-term recurrence
    relation derived from the defining ODE w'(z) = -2z w(z) + 2i/sqrt(pi).

    Implementation Logic
    --------------------
    The recurrence is derived by substituting the power series
    w(z) = sum_n a_n z^n into the ODE and matching coefficients:

    1. **Seed values** -- set the first two coefficients from the known
       boundary conditions of the Faddeeva function:
       a_0 = 1  (w(0) = erfc(0) = 1),
       a_1 = 2i / sqrt(pi)  (from w'(0) = 2i/sqrt(pi)).

    2. **Three-term recurrence** -- for n >= 1 iterate:
       a_{n+1} = -2 * a_{n-1} / (n + 1).
       This follows directly from the ODE: equating the z^n coefficient
       on both sides yields (n+1) a_{n+1} = -2 a_{n-1}.

    3. **Return** the coefficient list of length ``_N_TAYLOR``.

    Parameters
    ----------
    None

    Returns
    -------
    c : list[complex]
        Taylor coefficients [a_0, a_1, ..., a_{N-1}] where
        N = ``_N_TAYLOR``.

    Notes
    -----
    The recurrence skips every other coefficient (even indices are real,
    odd indices are purely imaginary), producing an alternating pattern
    that mirrors the parity structure of the Faddeeva function.
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

    Computes the Faddeeva (scaled complex complementary error) function
    for arbitrary complex arrays using a precomputed Taylor polynomial
    evaluated via Horner's method.

    Implementation Logic
    --------------------
    1. **Cast to complex128** -- convert the input to ``jnp.complex128``
       via ``jnp.asarray`` to ensure sufficient precision for the 64-term
       polynomial evaluation.

    2. **Evaluate via Horner's method** -- call ``jnp.polyval`` with
       the module-level constant ``_W_POLY`` (coefficients stored in
       descending-power order, i.e. reversed from the natural a_0, ...,
       a_{N-1} ordering) and the cast input. The ``unroll=8`` hint
       allows XLA to pipeline the inner loop for better throughput on
       accelerators.

    Parameters
    ----------
    z : Complex[Array, " ..."]
        Complex argument(s), arbitrary shape.

    Returns
    -------
    w : Complex[Array, " ..."]
        Faddeeva function values, same shape as ``z``.

    Notes
    -----
    Accuracy is approximately double-precision (~15 significant digits)
    for |z| < 6. For |z| >= 6 the Taylor series converges slowly and
    an asymptotic expansion or continued-fraction representation should
    be used instead. The current implementation does **not** fall back
    to an asymptotic form, so callers must ensure inputs stay within the
    convergence domain.
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

    Transforms an arbitrary float array so that the output has zero mean
    and unit standard deviation, which is a common preprocessing step
    before comparing simulated and experimental ARPES spectra.

    Implementation Logic
    --------------------
    1. **Compute statistics** -- calculate the global mean and standard
       deviation of the input array using ``jnp.mean`` and ``jnp.std``
       (population std, i.e. ddof=0).

    2. **Guard against zero std** -- if the standard deviation is
       exactly zero (constant array), replace it with 1.0 via
       ``jnp.where(std > 0, std, 1.0)`` to avoid division-by-zero.
       This produces an all-zeros output for constant inputs, which is
       the mathematically sensible limit.

    3. **Normalize** -- compute (data - mean) / safe_std element-wise
       and return.

    Parameters
    ----------
    data : Float[Array, " ..."]
        Input data array of any shape.

    Returns
    -------
    normalized : Float[Array, " ..."]
        Normalized data with mean 0 and standard deviation 1
        (or all zeros if the input is constant).

    Notes
    -----
    The normalization is computed over **all** elements (global mean
    and std), not per-axis. For per-axis normalization, reshape the
    data and call this function on each slice separately.
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
