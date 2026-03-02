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

import jax
import jax.numpy as jnp
from beartype import beartype
from jaxtyping import Array, Complex, Float, Int, jaxtyped

_N_TAYLOR: int = 64


def _faddeeva_taylor_coeffs() -> Complex[Array, " N"]:
    r"""Taylor coefficients of w(z) = exp(-z^2) erfc(-iz) via JAX scan.

    Extended Summary
    ----------------
    Computes the first ``_N_TAYLOR`` coefficients of the Taylor expansion
    of the Faddeeva function about the origin.

    **Mathematical derivation:**

    The Faddeeva function satisfies the ODE:

    .. math::

        w'(z) = -2z \, w(z) + \frac{2i}{\sqrt{\pi}}

    Substituting the power series ansatz :math:`w(z) = \sum_{n=0}^\infty
    a_n z^n` and matching coefficients of :math:`z^n` on both sides
    yields the two-term recurrence:

    .. math::

        a_0 = 1, \quad a_1 = \frac{2i}{\sqrt{\pi}}, \quad
        a_{n+1} = \frac{-2 \, a_{n-1}}{n+1} \quad (n \ge 1)

    Note that the recurrence couples even-indexed coefficients among
    themselves and odd-indexed coefficients among themselves (the
    series separates into even and odd parts). Even coefficients are
    real and odd coefficients are purely imaginary, reflecting the
    symmetry :math:`w(-z) = 2 e^{-z^2} - w(z)`.

    **Implementation:**

    The recurrence is implemented with ``jax.lax.scan`` (no Python
    for-loop) for efficiency and XLA compatibility. The scan carry
    is the pair :math:`(a_{n-1}, a_n)`, and at each step n (0-indexed),
    the next coefficient is :math:`a_{n+2} = -2 a_n / (n + 2)`. The
    scan produces coefficients :math:`a_2` through :math:`a_{N-1}`,
    which are concatenated with the seed values :math:`a_0, a_1` to
    form the complete coefficient vector.

    The resulting array is reversed (descending power order) at module
    level and stored in ``_W_POLY`` for use with ``jnp.polyval``
    (Horner's method).

    Returns
    -------
    coeffs : Complex[Array, " N"]
        Taylor coefficients :math:`[a_0, a_1, \ldots, a_{N-1}]` in
        ascending power order, where ``N = _N_TAYLOR``.
    """
    c0: Complex[Array, " "] = jnp.array(1.0 + 0j, dtype=jnp.complex128)
    c1: Complex[Array, " "] = jnp.array(
        2.0j / math.sqrt(math.pi), dtype=jnp.complex128
    )

    def body(
        carry: tuple[Complex[Array, " "], Complex[Array, " "]],
        n: Int[Array, ""],
    ) -> tuple[
        tuple[Complex[Array, " "], Complex[Array, " "]],
        Complex[Array, " "],
    ]:
        c_prev, c_curr = carry
        next_c: Complex[Array, " "] = (-2.0 * c_prev) / (
            jnp.asarray(n, dtype=jnp.float64) + 2.0
        )
        return (c_curr, next_c), next_c

    rest: Complex[Array, " N2"]
    _, rest = jax.lax.scan(
        body,
        (c0, c1),
        jnp.arange(_N_TAYLOR - 2, dtype=jnp.int32),
    )
    full: Complex[Array, " N"] = jnp.concatenate([c0[None], c1[None], rest])
    return full


_W_POLY: Complex[Array, " N"] = _faddeeva_taylor_coeffs()[::-1]


@jaxtyped(typechecker=beartype)
def faddeeva(
    z: Complex[Array, " ..."],
) -> Complex[Array, " ..."]:
    r"""Evaluate the Faddeeva function w(z) = exp(-z^2) erfc(-iz).

    Extended Summary
    ----------------
    Computes the Faddeeva (scaled complex complementary error) function
    for arbitrary complex arrays using a precomputed Taylor polynomial
    evaluated via Horner's method.

    The Faddeeva function is defined as:

    .. math::

        w(z) = e^{-z^2} \operatorname{erfc}(-iz)
             = e^{-z^2} \left(1 + \frac{2i}{\sqrt{\pi}}
               \int_0^z e^{t^2} dt \right)

    It is closely related to the Voigt profile used in spectroscopy:
    the Voigt function is the real part of the Faddeeva function
    evaluated along the imaginary axis. In ARPES simulations, it
    appears when convolving Lorentzian lifetime broadening with
    Gaussian instrumental resolution.

    **Implementation:**

    1. **Cast to complex128** -- convert the input to ``jnp.complex128``
       via ``jnp.asarray`` to ensure sufficient precision for the 64-term
       polynomial evaluation.

    2. **Evaluate via Horner's method** -- call ``jnp.polyval`` with
       the module-level constant ``_W_POLY`` (coefficients stored in
       descending-power order, i.e. reversed from the natural
       :math:`a_0, \ldots, a_{N-1}` ordering) and the cast input. The
       ``unroll=8`` hint allows XLA to pipeline the inner loop for
       better throughput on accelerators.

    The Taylor coefficients are derived from the ODE
    :math:`w'(z) = -2z w(z) + 2i/\sqrt{\pi}` and precomputed at
    module import time by `_faddeeva_taylor_coeffs`. See that
    function's docstring for the full recurrence derivation.

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

    The function is fully differentiable via JAX autodiff, which
    is important for gradient-based optimization of broadening
    parameters in inverse-fitting workflows.
    """
    z_c: Complex[Array, " ..."] = jnp.asarray(z, dtype=jnp.complex128)
    w: Complex[Array, " ..."] = jnp.polyval(_W_POLY, z_c, unroll=8)
    return w


@jaxtyped(typechecker=beartype)
def zscore_normalize(
    data: Float[Array, " ..."],
) -> Float[Array, " ..."]:
    r"""Apply z-score normalization (zero-mean, unit-variance).

    Extended Summary
    ----------------
    Transforms an arbitrary float array so that the output has zero mean
    and unit standard deviation, which is a common preprocessing step
    before comparing simulated and experimental ARPES spectra. The
    z-score transformation is:

    .. math::

        \hat{x}_i = \frac{x_i - \bar{x}}{\sigma}

    where :math:`\bar{x}` is the global mean and :math:`\sigma` is
    the population standard deviation (:math:`\text{ddof}=0`).

    **Implementation details:**

    1. **Compute statistics** -- calculate the global mean and standard
       deviation of the input array using ``jnp.mean`` and ``jnp.std``
       (population std, i.e. ddof=0).

    2. **Guard against zero std** -- if the standard deviation is
       exactly zero (constant array), replace it with 1.0 via
       ``jnp.where(std > 0, std, 1.0)`` to avoid division-by-zero.
       This produces an all-zeros output for constant inputs, which is
       the mathematically sensible limit. The ``jnp.where`` formulation
       is gradient-safe: JAX will not propagate gradients through the
       zero-std branch.

    3. **Normalize** -- compute ``(data - mean) / safe_std`` element-wise
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

    This function is differentiable via JAX autodiff with respect to
    the input ``data``. The gradient propagates through both the
    mean-subtraction and the division by standard deviation.
    """
    mean_val: Float[Array, " "] = jnp.mean(data)
    std_val: Float[Array, " "] = jnp.std(data)
    safe_std: Float[Array, " "] = jnp.where(std_val > 0.0, std_val, 1.0)
    normalized: Float[Array, " ..."] = (data - mean_val) / safe_std
    return normalized


__all__: list[str] = [
    "faddeeva",
    "zscore_normalize",
]
