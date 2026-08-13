"""Compute pseudo-wave point-detector ARPES amplitudes.

Extended Summary
----------------
Evaluate only the smooth-orbital pseudo tier with a point-detector kernel.
This module does not restore all-electron PAW terms, project spin, or deposit
intensity into detector bins.

Routine Listings
----------------
:func:`plane_wave_mask`
    Compute the ``plane_wave_mask`` public contract.
:func:`plane_wave_pseudo_amplitude`
    Compute the ``plane_wave_pseudo_amplitude`` public contract.
:func:`surface_window_transform`
    Compute the ``surface_window_transform`` public contract.
"""

import equinox as eqx
import jax.numpy as jnp
from beartype import beartype
from jaxtyping import Array, Complex128, Float64, jaxtyped

from diffpes.types import PlaneWaveBatch


@jaxtyped(typechecker=beartype)
def plane_wave_mask(batch: PlaneWaveBatch) -> Float64[Array, "n_state n_pw"]:
    """Compute the ``plane_wave_mask`` public contract.

    Validate documented inputs and preserve the declared scientific identity.

    :see: :class:`~.test_plane_wave.TestPlaneWaveMask`

    Notes
    -----
    Validate inputs before returning the named result.

    Parameters
    ----------
    batch : PlaneWaveBatch
        Input value for this operation.

    Returns
    -------
    result : Float64[Array, 'n_state n_pw']
        Validated operation result.
    """
    index: Float64[Array, " n_pw"] = jnp.arange(
        batch.coefficients.shape[1], dtype=jnp.float64
    )
    result: Float64[Array, "n_state n_pw"] = (
        index[None, :] < batch.plane_wave_counts[:, None]
    ).astype(jnp.float64)
    return result


@jaxtyped(typechecker=beartype)
def surface_window_transform(
    delta_k_cart: Float64[Array, "... 3"],
    lateral_coherence_ang: Float64[Array, ""],
    mean_free_path_ang: Float64[Array, ""],
) -> Complex128[Array, "..."]:
    """Compute the ``surface_window_transform`` public contract.

    Validate documented inputs and preserve the declared scientific identity.

    :see: :class:`~.test_plane_wave.TestSurfaceWindowTransform`

    Notes
    -----
    Validate inputs before returning the named result.

    Parameters
    ----------
    delta_k_cart : Float64[Array, '... 3']
        Input value for this operation.
    lateral_coherence_ang : Float64[Array, '']
        Input value for this operation.
    mean_free_path_ang : Float64[Array, '']
        Input value for this operation.

    Returns
    -------
    result : Complex128[Array, '...']
        Validated operation result.
    """
    checked_delta: Float64[Array, "... 3"] = eqx.error_if(
        delta_k_cart,
        ~jnp.all(jnp.isfinite(delta_k_cart)),
        "window momentum mismatch must be finite",
    )
    checked_coherence: Float64[Array, ""] = eqx.error_if(
        lateral_coherence_ang,
        ~jnp.isfinite(lateral_coherence_ang) | (lateral_coherence_ang <= 0.0),
        "lateral coherence must be finite and positive",
    )
    checked_mean_free_path: Float64[Array, ""] = eqx.error_if(
        mean_free_path_ang,
        ~jnp.isfinite(mean_free_path_ang) | (mean_free_path_ang <= 0.0),
        "mean free path must be finite and positive",
    )
    parallel_sq: Float64[Array, "..."] = jnp.sum(
        checked_delta[..., :2] ** 2, axis=-1
    )
    lateral: Float64[Array, "..."] = jnp.exp(
        -0.5 * checked_coherence**2 * parallel_sq
    )
    denominator: Complex128[Array, "..."] = (
        1.0 / checked_mean_free_path - 1.0j * checked_delta[..., 2]
    )
    result: Complex128[Array, "..."] = lateral / denominator
    return result


@jaxtyped(typechecker=beartype)
def plane_wave_pseudo_amplitude(
    batch: PlaneWaveBatch,
    final_k_cart_inv_ang: Float64[Array, "n_detector 3"],
    polarization: Complex128[Array, " 3"],
    lateral_coherence_ang: Float64[Array, ""],
    mean_free_path_ang: Float64[Array, ""],
) -> Complex128[Array, "n_state n_spinor n_detector"]:
    """Compute the ``plane_wave_pseudo_amplitude`` public contract.

    Validate documented inputs and preserve the declared scientific identity.

    :see: :class:`~.test_plane_wave.TestPlaneWavePseudoAmplitude`

    Notes
    -----
    Validate inputs before returning the named result.

    Parameters
    ----------
    batch : PlaneWaveBatch
        Input value for this operation.
    final_k_cart_inv_ang : Float64[Array, 'n_detector 3']
        Input value for this operation.
    polarization : Complex128[Array, ' 3']
        Input value for this operation.
    lateral_coherence_ang : Float64[Array, '']
        Input value for this operation.
    mean_free_path_ang : Float64[Array, '']
        Input value for this operation.

    Returns
    -------
    result : Complex128[Array, 'n_state n_spinor n_detector']
        Validated operation result.
    """
    checked_final_k: Float64[Array, "n_detector 3"] = eqx.error_if(
        final_k_cart_inv_ang,
        ~jnp.all(jnp.isfinite(final_k_cart_inv_ang)),
        "final-state momenta must be finite",
    )
    checked_polarization: Complex128[Array, " 3"] = eqx.error_if(
        polarization,
        ~jnp.all(jnp.isfinite(polarization)),
        "polarization must be finite",
    )
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
        checked_final_k[None, None, :, :] - wavevector[:, :, None, :]
    )
    window: Complex128[Array, "n_state n_pw n_detector"] = (
        surface_window_transform(
            delta, lateral_coherence_ang, mean_free_path_ang
        )
    )
    dipole: Complex128[Array, "n_state n_pw"] = jnp.einsum(
        "spj,j->sp", wavevector, checked_polarization
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
