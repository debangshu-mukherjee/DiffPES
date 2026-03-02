"""Momentum resolution broadening for ARPES simulations.

Extended Summary
----------------
Applies a Gaussian convolution along the k-axis to simulate the
finite angular acceptance of the ARPES electron analyser. The
convolution is implemented as a dense kernel matrix multiply,
which is efficient for typical k-point counts and fully
JAX-differentiable.

Routine Listings
----------------
:func:`apply_momentum_broadening`
    Convolve an intensity map with a Gaussian kernel in k-space.

Notes
-----
The kernel matrix has shape ``(K, K)`` and is built from the
cumulative k-path distances, so it correctly handles non-uniform
k-point spacing. All operations are JIT-compilable and support
``jax.grad`` with respect to the broadening width ``dk``.
"""

import jax.numpy as jnp
from beartype import beartype
from jaxtyping import Array, Float, jaxtyped

from diffpes.types import ScalarFloat

_EPS: float = 1e-12
_MIN_SUM: float = 1e-30


@jaxtyped(typechecker=beartype)
def apply_momentum_broadening(
    intensity: Float[Array, "K E"],
    k_distances: Float[Array, " K"],
    dk: ScalarFloat,
) -> Float[Array, "K E"]:
    r"""Convolve I(k, E) with a Gaussian in k-space.

    Simulates the finite angular (momentum) resolution of an ARPES
    analyser by applying a Gaussian convolution along the k-axis.
    This smears sharp spectral features in momentum, mimicking the
    experimental point-spread function in the angular direction.

    Extended Summary
    ----------------
    In a real ARPES experiment, the finite angular acceptance of the
    electron analyser and the finite spot size of the photon beam
    lead to a momentum-space resolution function. This is well
    approximated by a Gaussian of width ``dk`` (in inverse
    Angstroms). The convolution is implemented as a matrix multiply
    with a normalized Gaussian kernel, which is efficient and
    JAX-differentiable.

    Implementation Logic
    --------------------
    1. **Guard against zero dk**:
       ``safe_dk = max(dk, 1e-12)``
       Prevents division by zero if ``dk`` is exactly zero. The
       resulting extremely narrow kernel effectively becomes an
       identity operation.

    2. **Build Gaussian kernel matrix**:
       ``G_{ij} = exp(-0.5 * ((k_i - k_j) / safe_dk)^2)``
       The kernel is a ``(K, K)`` matrix where each element measures
       the Gaussian overlap between k-points ``i`` and ``j``. The
       k-distances are cumulative path lengths along the k-path,
       accounting for non-uniform k-point spacing.

    3. **Row-normalize the kernel**:
       Each row of the kernel is divided by its sum so that the
       convolution conserves total spectral weight. A safety guard
       replaces zero row sums with 1.0 to avoid division by zero
       (which could occur if k-points are extremely far apart
       relative to ``dk``).

    4. **Apply via matrix multiplication**:
       ``I_broadened = G @ I``
       The ``(K, K) @ (K, E)`` product applies the normalized
       Gaussian weights to each energy column independently,
       producing the momentum-broadened intensity of shape ``(K, E)``.

    Parameters
    ----------
    intensity : Float[Array, "K E"]
        ARPES intensity map with shape ``(n_kpoints, n_energies)``.
    k_distances : Float[Array, " K"]
        Cumulative k-path distances of shape ``(n_kpoints,)``,
        in inverse Angstroms. Must be monotonically increasing.
    dk : ScalarFloat
        Gaussian broadening standard deviation in inverse Angstroms.
        Typical experimental values range from 0.01 to 0.05.

    Returns
    -------
    broadened : Float[Array, "K E"]
        Momentum-broadened intensity map, same shape as ``intensity``.

    Notes
    -----
    The kernel matrix is dense ``(K, K)`` and fully traced by JAX,
    so this function supports ``jax.grad`` with respect to ``dk``
    and ``intensity``. For very large numbers of k-points, memory
    usage scales as ``O(K^2)``.
    """
    dk_arr: Float[Array, ""] = jnp.asarray(dk, dtype=jnp.float64)
    safe_dk: Float[Array, ""] = jnp.where(dk_arr > _EPS, dk_arr, _EPS)
    k_i: Float[Array, " K 1"] = k_distances[:, jnp.newaxis]
    k_j: Float[Array, " 1 K"] = k_distances[jnp.newaxis, :]
    kernel: Float[Array, " K K"] = jnp.exp(-0.5 * ((k_i - k_j) / safe_dk) ** 2)
    row_sum: Float[Array, " K 1"] = jnp.sum(kernel, axis=1, keepdims=True)
    safe_sum: Float[Array, " K 1"] = jnp.where(
        row_sum > _MIN_SUM, row_sum, 1.0
    )
    kernel: Float[Array, " K K"] = kernel / safe_sum
    broadened: Float[Array, " K E"] = kernel @ intensity
    return broadened


__all__: list[str] = ["apply_momentum_broadening"]
