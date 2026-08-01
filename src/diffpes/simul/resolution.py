"""Apply momentum resolution broadening to ARPES simulations.

Extended Summary
----------------
The module applies a Gaussian convolution along the k-axis. This operation
simulates the finite angular acceptance of the ARPES electron analyzer. A dense
kernel matrix multiplication implements the convolution. It supports typical
k-point counts and JAX differentiation.

Routine Listings
----------------
:func:`apply_momentum_broadening`
    Convolve I(k, E) with a Gaussian in k-space.

Notes
-----
The kernel matrix has shape ``(K, K)``. The cumulative k-path distances define
it, so it supports nonuniform k-point spacing. JAX can compile all operations
and differentiate them with respect to ``dk``.
"""

import jax.numpy as jnp
from beartype import beartype
from jaxtyping import Array, Float64, jaxtyped

from diffpes.maths import safe_divide
from diffpes.types import EPS, ScalarFloat


@jaxtyped(typechecker=beartype)
def apply_momentum_broadening(
    intensity: Float64[Array, "K E"],
    k_distances: Float64[Array, " K"],
    dk: ScalarFloat,
) -> Float64[Array, "K E"]:
    r"""Convolve I(k, E) with a Gaussian in k-space.

    Simulates the finite angular (momentum) resolution of an ARPES
    analyser by applying a Gaussian convolution along the k-axis.
    This smears sharp spectral features in momentum, mimicking the
    experimental point-spread function in the angular direction.

    In an ARPES experiment, the electron analyzer has a finite angular
    acceptance. The photon beam also has a finite spot size. These properties
    produce a momentum-space resolution function. A Gaussian with width ``dk``
    approximates this function. The implementation multiplies the intensity by
    a normalized Gaussian kernel. JAX can differentiate this operation.

    :see: :class:`~.test_resolution.TestApplyMomentumBroadening`

    Implementation Logic
    --------------------
    1. **Guard the Gaussian width**::

           safe_dk: Float64[Array, ""] = jnp.maximum(dk_arr, EPS)

       This guard prevents division by zero. A zero width produces an
       effectively diagonal kernel.

    2. **Build the Gaussian kernel**::

           kernel: Float64[Array, " K K"] = jnp.exp(
               -0.5 * scaled_distances**2
           )

       Each matrix element measures the Gaussian overlap between two
       k-points. The cumulative distances support nonuniform spacing.

    3. **Normalize each row**::

           kernel = safe_divide(kernel, row_sum)

       The normalization preserves the spectral weight at each output
       k-point. The safe division also protects an empty numerical row.

    4. **Apply the kernel**::

           broadened: Float64[Array, "K E"] = kernel @ intensity

       The matrix product applies the momentum response independently to
       every energy column.

    Parameters
    ----------
    intensity : Float64[Array, "K E"]
        ARPES intensity map with shape ``(n_kpoints, n_energies)``.
    k_distances : Float64[Array, " K"]
        Cumulative k-path distances of shape ``(n_kpoints,)``,
        in inverse Angstroms. Must be monotonically increasing.
    dk : ScalarFloat
        Gaussian broadening standard deviation in inverse Angstroms.
        Typical experimental values range from 0.01 to 0.05.

    Returns
    -------
    broadened : Float64[Array, "K E"]
        Momentum-broadened intensity map, same shape as ``intensity``.

    Notes
    -----
    The kernel matrix is dense ``(K, K)`` and fully traced by JAX,
    so this function supports ``jax.grad`` with respect to ``dk``
    and ``intensity``. For very large numbers of k-points, memory
    usage scales as ``O(K^2)``.
    """
    dk_arr: Float64[Array, ""] = jnp.asarray(dk, dtype=jnp.float64)
    safe_dk: Float64[Array, ""] = jnp.maximum(dk_arr, EPS)
    k_i: Float64[Array, " K 1"] = k_distances[:, jnp.newaxis]
    k_j: Float64[Array, " 1 K"] = k_distances[jnp.newaxis, :]
    scaled_distances: Float64[Array, " K K"] = safe_divide(k_i - k_j, safe_dk)
    kernel: Float64[Array, " K K"] = jnp.exp(-0.5 * scaled_distances**2)
    row_sum: Float64[Array, " K 1"] = jnp.sum(kernel, axis=1, keepdims=True)
    kernel = safe_divide(kernel, row_sum)
    broadened: Float64[Array, " K E"] = kernel @ intensity
    return broadened


__all__: list[str] = ["apply_momentum_broadening"]
