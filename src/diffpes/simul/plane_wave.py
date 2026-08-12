"""Compute pseudo-wave point-detector ARPES amplitudes.

This module implements only the smooth-orbital pseudo-wave tier. It does not
implement PAW all-electron restoration, spin projection, or detector
deposition.
"""

import jax.numpy as jnp
from beartype import beartype
from jaxtyping import Array, Complex128, Float64, jaxtyped

from diffpes.types import PlaneWaveBatch


@jaxtyped(typechecker=beartype)
def plane_wave_mask(batch: PlaneWaveBatch) -> Float64[Array, "n_state n_pw"]:
    """Return the multiplicative validity mask for a padded plane-wave
    batch.
    """
    index: Float64[Array, " n_pw"] = jnp.arange(
        batch.coefficients.shape[1], dtype=jnp.float64
    )
    return (index[None, :] < batch.plane_wave_counts[:, None]).astype(
        jnp.float64
    )


@jaxtyped(typechecker=beartype)
def surface_window_transform(
    delta_k_cart: Float64[Array, "... 3"],
    lateral_coherence_ang: Float64[Array, ""],
    mean_free_path_ang: Float64[Array, ""],
) -> Complex128[Array, "..."]:
    """Evaluate a Gaussian-lateral damped-half-space window transform."""
    parallel_sq: Float64[Array, "..."] = jnp.sum(
        delta_k_cart[..., :2] ** 2, axis=-1
    )
    lateral: Float64[Array, "..."] = jnp.exp(
        -0.5 * lateral_coherence_ang**2 * parallel_sq
    )
    denominator: Complex128[Array, "..."] = (
        1.0 / mean_free_path_ang - 1.0j * delta_k_cart[..., 2]
    )
    return lateral / denominator


@jaxtyped(typechecker=beartype)
def plane_wave_pseudo_amplitude(
    batch: PlaneWaveBatch,
    final_k_cart_inv_ang: Float64[Array, "n_detector 3"],
    polarization: Complex128[Array, " 3"],
    lateral_coherence_ang: Float64[Array, ""],
    mean_free_path_ang: Float64[Array, ""],
) -> Complex128[Array, "n_state n_spinor n_detector"]:
    """Evaluate the coherent finite-window pseudo-wave velocity amplitude."""
    mask: Float64[Array, "n_state n_pw"] = plane_wave_mask(batch)
    reciprocal: Float64[Array, "3 3"] = batch.geometry.reciprocal
    g_cart: Float64[Array, "n_state n_pw 3"] = jnp.einsum(
        "spj,jk->spk", batch.g_vectors_frac.astype(jnp.float64), reciprocal
    )
    k_cart: Float64[Array, "n_state 3"] = jnp.einsum(
        "sj,jk->sk", batch.kpoints_frac, reciprocal
    )
    wavevector: Float64[Array, "n_state n_pw 3"] = k_cart[:, None, :] + g_cart
    delta: Float64[Array, "n_state n_pw n_detector 3"] = (
        final_k_cart_inv_ang[None, None, :, :] - wavevector[:, :, None, :]
    )
    window: Complex128[Array, "n_state n_pw n_detector"] = (
        surface_window_transform(
            delta, lateral_coherence_ang, mean_free_path_ang
        )
    )
    dipole: Complex128[Array, "n_state n_pw"] = jnp.einsum(
        "spj,j->sp", wavevector, polarization
    )
    amplitude: Complex128[Array, "n_state n_spinor n_detector"] = jnp.einsum(
        "spi,sp,spd,sp->sid",
        batch.coefficients,
        dipole,
        window,
        mask,
    )
    return amplitude


__all__: list[str] = [
    "plane_wave_mask",
    "plane_wave_pseudo_amplitude",
    "surface_window_transform",
]
