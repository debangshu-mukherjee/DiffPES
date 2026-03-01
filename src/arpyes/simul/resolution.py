"""Momentum resolution broadening.

Extended Summary
----------------
Applies a Gaussian convolution along the k-axis to simulate
finite angular acceptance of the ARPES detector.
"""

import jax.numpy as jnp
from beartype import beartype
from jaxtyping import Array, Float, jaxtyped

from arpyes.types import ScalarFloat


@jaxtyped(typechecker=beartype)
def apply_momentum_broadening(
    intensity: Float[Array, "K E"],
    k_distances: Float[Array, " K"],
    dk: ScalarFloat,
) -> Float[Array, "K E"]:
    r"""Convolve I(k, E) with a Gaussian in k-space.

    Builds a Gaussian kernel matrix :math:`G_{ij}` of shape (K, K)
    and multiplies: ``I_broadened = G @ I``.

    Parameters
    ----------
    intensity : Float[Array, "K E"]
        ARPES intensity map.
    k_distances : Float[Array, " K"]
        Cumulative k-path distances (1D).
    dk : ScalarFloat
        Gaussian broadening width in inverse Angstroms.

    Returns
    -------
    broadened : Float[Array, "K E"]
        Momentum-broadened intensity.
    """
    dk_arr = jnp.asarray(dk, dtype=jnp.float64)
    safe_dk = jnp.where(dk_arr > 1e-12, dk_arr, 1e-12)

    # Build Gaussian kernel matrix
    k_i = k_distances[:, jnp.newaxis]  # (K, 1)
    k_j = k_distances[jnp.newaxis, :]  # (1, K)
    kernel = jnp.exp(-0.5 * ((k_i - k_j) / safe_dk) ** 2)

    # Normalize each row
    row_sum = jnp.sum(kernel, axis=1, keepdims=True)
    safe_sum = jnp.where(row_sum > 1e-30, row_sum, 1.0)
    kernel = kernel / safe_sum

    broadened = kernel @ intensity
    return broadened


__all__: list[str] = ["apply_momentum_broadening"]
