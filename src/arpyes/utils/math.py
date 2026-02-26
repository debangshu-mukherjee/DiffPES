"""Mathematical utility functions for ARPES simulations.

Extended Summary
----------------
Provides JAX-compatible implementations of the Faddeeva function
(complex error function) and data normalization routines used
throughout the ARPES simulation pipeline.

Routine Listings
----------------
:func:`faddeeva`
    Faddeeva function via Weideman's 32-term rational approximation.
:func:`zscore_normalize`
    Z-score (zero-mean, unit-variance) normalization.

Notes
-----
The Faddeeva implementation uses the algorithm from
Weideman (1994), SIAM J. Numer. Anal. 31(5), pp. 1497-1518,
which provides high accuracy for complex arguments via a
32-term rational approximation using Chebyshev nodes.
"""

import jax.numpy as jnp
from beartype import beartype
from jaxtyping import Array, Complex, Float, jaxtyped

_FADDEEVA_N: int = 32
_FADDEEVA_L: float = 6.0

_K_IDX: Float[Array, " N"] = jnp.arange(
    1, _FADDEEVA_N + 1, dtype=jnp.float64
)
_THETA: Float[Array, " N"] = (
    jnp.pi * (_K_IDX - 0.5) / _FADDEEVA_N
)
_T_NODES: Float[Array, " N"] = (
    _FADDEEVA_L * jnp.tan(_THETA / 2.0)
)
_F_VALS: Float[Array, " N"] = (
    jnp.exp(-_T_NODES**2)
    * (_FADDEEVA_L**2 + _T_NODES**2)
)
_FFT_INPUT: Float[Array, " M"] = jnp.concatenate(
    [_F_VALS, _F_VALS[::-1]]
)
_COEFFS: Float[Array, " N"] = (
    (2.0 / _FADDEEVA_N)
    * jnp.real(jnp.fft.fft(_FFT_INPUT))[: _FADDEEVA_N]
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
    Uses Weideman's 32-term rational approximation with
    FFT-computed coefficients (L=6, N=32) for double precision
    accuracy. Reference: Weideman (1994), SIAM J. Numer. Anal.
    31(5), pp. 1497-1518.
    """
    z_c: Complex[Array, " ..."] = jnp.asarray(
        z, dtype=jnp.complex128
    )
    poles: Complex[Array, " N"] = 1j * _T_NODES
    diffs: Complex[Array, " ... N"] = (
        z_c[..., jnp.newaxis] - poles[jnp.newaxis, :]
    )
    terms: Complex[Array, " ... N"] = (
        _COEFFS[jnp.newaxis, :] / diffs
    )
    partial_sum: Complex[Array, " ..."] = jnp.sum(
        terms, axis=-1
    )
    w: Complex[Array, " ..."] = (
        (2.0 / jnp.sqrt(jnp.pi))
        * partial_sum
        * jnp.exp(-(z_c**2))
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
