"""Compute mathematical utilities for ARPES simulations.

Extended Summary
----------------
The module provides JAX-compatible implementations of the Faddeeva function
and data normalization routines. The ARPES simulation pipeline uses these
functions. Complex parameter packing provides the required optimizer boundary
for complex physics.

Routine Listings
----------------
:func:`faddeeva`
    Evaluate the Faddeeva function w(z) = exp(-z^2) erfc(-iz).
:func:`pack_complex`
    Pack complex parameters as stacked real values.
:func:`unpack_complex`
    Unpack stacked real parameters into complex values.
:func:`zscore_normalize`
    Apply z-score normalization (zero-mean, unit-variance).

Notes
-----
The Faddeeva implementation uses a fixed-order Weideman rational
approximation. It covers the declared upper-half-plane envelope without a
region seam.
"""

import math

import equinox as eqx
import jax
import jax.numpy as jnp
from beartype import beartype
from jaxtyping import Array, Complex, Complex128, Float, Float64, jaxtyped

from diffpes.maths import safe_divide


def _faddeeva_weideman_coefficients() -> Float64[Array, " N"]:
    r"""Generate fixed-order Weideman rational coefficients.

    The construction samples the mapped Gaussian on a fixed tangent grid.
    A discrete Fourier transform produces the rational-basis coefficients.

    Returns
    -------
    coeffs : Float64[Array, " N"]
        Real coefficients in descending polynomial order.

    Notes
    -----
    The algorithm-selection sweep freezes order 40 before production.
    The transform follows Weideman's published rational construction.
    """
    order: int = 40
    scale: float = math.sqrt(order / math.sqrt(2.0))
    doubled_order: int = 2 * order
    indices: Float64[Array, " grid"] = jnp.arange(
        -doubled_order + 1,
        doubled_order,
        dtype=jnp.float64,
    )
    angles: Float64[Array, " grid"] = indices * math.pi / doubled_order
    mapped: Float64[Array, " grid"] = scale * jnp.tan(angles / 2.0)
    samples: Float64[Array, " grid"] = jnp.exp(-(mapped**2)) * (
        scale**2 + mapped**2
    )
    padded: Float64[Array, " fft_grid"] = jnp.concatenate(
        (jnp.zeros(1, dtype=jnp.float64), samples)
    )
    transformed: Complex128[Array, " fft_grid"] = jnp.fft.fft(
        jnp.fft.fftshift(padded)
    )
    ascending: Float64[Array, " fft_grid"] = jnp.real(transformed) / (
        2 * doubled_order
    )
    result: Float64[Array, " N"] = ascending[1 : order + 1][::-1]
    return result


_W_POLY: Float64[Array, " N"] = _faddeeva_weideman_coefficients()


@jaxtyped(typechecker=beartype)
def faddeeva(  # noqa: DOC502 -- eqx.error_if raises under JAX execution.
    z: Complex[Array, " ..."],
) -> Complex128[Array, " ..."]:
    r"""Evaluate the Faddeeva function w(z) = exp(-z^2) erfc(-iz).

    The function evaluates a fixed-order rational approximation on complex
    arrays in the closed upper half-plane.

    The following equation defines the Faddeeva function:

    .. math::

        w(z) = e^{-z^2} \operatorname{erfc}(-iz)
             = e^{-z^2} \left(1 + \frac{2i}{\sqrt{\pi}}
               \int_0^z e^{t^2} dt \right)

    The Voigt profile uses the real part of the Faddeeva function along the
    imaginary axis. ARPES simulations use it to convolve Lorentzian lifetime
    broadening with Gaussian instrument resolution.

    :see: :class:`~.test_math.TestFaddeeva`

    Parameters
    ----------
    z : Complex[Array, " ..."]
        Complex arguments with ``Im(z) >= 0`` and ``abs(z) <= 1e8``.

    Returns
    -------
    w : Complex128[Array, " ..."]
        Faddeeva function values, same shape as ``z``.

    Raises
    ------
    EquinoxRuntimeError
        If an argument is nonfinite, below the real axis, or outside the
        certified magnitude envelope.

    Notes
    -----
    The order-40 Weideman approximation uses one rational region, so no
    selector seam enters the primal or its JAX derivative. The denominator
    has positive real part throughout the declared upper-half-plane domain.
    Inputs promote to complex128 before validation and evaluation.
    """
    z_c: Complex128[Array, " ..."] = jnp.asarray(z, dtype=jnp.complex128)
    maximum_absolute_value: float = 1.0e8
    invalid: Array = (
        ~jnp.all(jnp.isfinite(z_c))
        | jnp.any(jnp.imag(z_c) < 0.0)
        | jnp.any(jnp.abs(z_c) > maximum_absolute_value)
    )
    checked: Complex128[Array, " ..."] = eqx.error_if(
        z_c,
        invalid,
        "z must be finite with Im(z) >= 0 and abs(z) <= 1e8",
    )
    scale: float = math.sqrt(40 / math.sqrt(2.0))
    denominator: Complex128[Array, " ..."] = scale - 1j * checked
    transformed: Complex128[Array, " ..."] = (
        scale + 1j * checked
    ) / denominator
    polynomial: Complex128[Array, " ..."] = jnp.polyval(
        _W_POLY,
        transformed,
        unroll=8,
    )
    w: Complex128[Array, " ..."] = 2.0 * polynomial / denominator**2 + 1.0 / (
        jnp.sqrt(jnp.pi) * denominator
    )
    return w


@jaxtyped(typechecker=beartype)
def pack_complex(
    z: Complex[Array, " ..."],
) -> Float[Array, " ... 2"]:
    """Pack complex parameters as stacked real values.

    Complex parameters cross the optimizer and Fisher-information boundary as
    stacked reals, while values remain complex inside the physics pipeline.
    This function is the sanctioned complex-to-real crossing point.

    :see: :class:`~.test_math.TestPackComplex`

    Parameters
    ----------
    z : Complex[Array, " ..."]
        Complex-valued physics parameters of arbitrary shape.

    Returns
    -------
    packed : Float[Array, " ... 2"]
        Real-valued parameters with real and imaginary components in the final
        axis, in that order.

    Notes
    -----
    The function forms the final axis with
    ``jnp.stack([z.real, z.imag], axis=-1)``. This operation preserves the
    component dtype and exposes independent real optimizer coordinates.

    See Also
    --------
    unpack_complex : Restore complex values inside the physics pipeline.
    """
    packed: Float[Array, " ... 2"] = jnp.stack([z.real, z.imag], axis=-1)
    return packed


@jaxtyped(typechecker=beartype)
def unpack_complex(
    p: Float[Array, " ... 2"],
) -> Complex[Array, " ..."]:
    """Unpack stacked real parameters into complex values.

    Complex parameters cross the optimizer and Fisher-information boundary as
    stacked reals, while values remain complex inside the physics pipeline.
    This function is the sanctioned real-to-complex crossing point.

    :see: :class:`~.test_math.TestUnpackComplex`

    Parameters
    ----------
    p : Float[Array, " ... 2"]
        Real-valued optimizer parameters whose final axis stores real and
        imaginary components, in that order.

    Returns
    -------
    unpacked : Complex[Array, " ..."]
        Complex-valued physics parameters with the packing axis removed.

    Notes
    -----
    ``jax.lax.complex`` combines the final-axis components without changing
    their precision. For a real loss, JAX differentiates the two packed
    components as ordinary real optimizer coordinates.

    See Also
    --------
    pack_complex : Expose complex parameters at the real optimizer boundary.
    """
    unpacked: Complex[Array, " ..."] = jax.lax.complex(p[..., 0], p[..., 1])
    return unpacked


@jaxtyped(typechecker=beartype)
def zscore_normalize(
    data: Float[Array, " ..."],
) -> Float[Array, " ..."]:
    r"""Apply z-score normalization (zero-mean, unit-variance).

    The function transforms a float array to zero mean and unit standard
    deviation. This transformation prepares simulated and experimental ARPES
    spectra for comparison. The z-score transformation is:

    .. math::

        \hat{x}_i = \frac{x_i - \bar{x}}{\sigma}

    where :math:`\bar{x}` is the global mean and :math:`\sigma` is
    the population standard deviation (:math:`\text{ddof}=0`).

    **Implementation details:**

    1. **Compute the statistics**: Compute the global mean with ``jnp.mean``.
       Compute the population standard deviation with ``jnp.std`` and ddof=0.

    2. **Guard against zero deviation**: Pass the centered values and standard
       deviation to :func:`~diffpes.maths.safe_divide`. The function selects
       zero for a constant input and defines a zero subgradient.

    3. **Normalize the values**: Compute ``(data - mean) / safe_std`` for each
       element and return the result.

    :see: :class:`~.test_math.TestZscoreNormalize`

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
    The function computes one global mean and standard deviation over all
    elements. For each-axis normalization, reshape the data and call the
    function on each slice.

    This function is differentiable via JAX autodiff with respect to
    the input ``data``. The gradient propagates through both the
    mean-subtraction and the division by standard deviation.
    """
    mean_val: Float[Array, " "] = jnp.mean(data)
    std_val: Float[Array, " "] = jnp.std(data)
    centered: Float[Array, " ..."] = data - mean_val
    normalized: Float[Array, " ..."] = safe_divide(centered, std_val)
    return normalized


__all__: list[str] = [
    "faddeeva",
    "pack_complex",
    "unpack_complex",
    "zscore_normalize",
]
