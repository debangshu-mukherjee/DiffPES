"""Replay Chinook's frozen lithium-chain sampled Gaussian response in JAX.

This module is an isolated parity adapter, not a production DiffPES path.  It
accepts Chinook's authenticated pre-resolution cut and reproduces only the
declared sampled-response convention.
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np
from beartype.typing import Dict, Tuple
from jaxtyping import Array, Bool, Float64
from numpy.typing import NDArray

from diffpes.simul import convolve_energy

MINIMUM_AXIS_SAMPLES: int = 2


def _sigma_pixels(
    axis: Float64[NDArray, " n"],
    fwhm: float,
    *,
    chinook_count_over_range: bool,
) -> float:
    """PRIVATE: Convert physical FWHM to sampled Gaussian sigma.

    Parameters
    ----------
    axis : Float64[NDArray, " n"]
        Strictly increasing sampled coordinate axis.
    fwhm : float
        Gaussian full width at half maximum in axis units.
    chinook_count_over_range : bool
        Whether to use Chinook's binding N/range discretization.

    Returns
    -------
    sigma_pixels : float
        Gaussian standard deviation in sample pixels.

    Raises
    ------
    ValueError
        If the axis has fewer than two strictly increasing samples.

    Notes
    -----
    The false branch is a planted N-1/range control and has no acceptance
    authority.
    """
    samples: int = int(axis.size)
    if samples < MINIMUM_AXIS_SAMPLES or not np.all(np.diff(axis) > 0.0):
        raise ValueError("axis must contain at least two increasing samples")
    span: float = float(axis[-1] - axis[0])
    count: int = samples if chinook_count_over_range else samples - 1
    sigma_pixels: float = float(
        fwhm * count / (span * math.sqrt(8.0 * math.log(2.0)))
    )
    return sigma_pixels


def _sampled_kernel(sigma_pixels: float) -> Float64[Array, " n_kernel"]:
    """PRIVATE: Build SciPy's normalized default four-sigma kernel.

    Parameters
    ----------
    sigma_pixels : float
        Positive Gaussian standard deviation in sample pixels.

    Returns
    -------
    kernel : Float64[Array, " n_kernel"]
        Odd, unit-sum, float64 sampled Gaussian kernel.

    Raises
    ------
    ValueError
        If the standard deviation is not positive.

    Notes
    -----
    SciPy rounds the support radius with int(truncate*sigma + 0.5).
    """
    if sigma_pixels <= 0.0:
        raise ValueError("sigma_pixels must be positive")
    radius: int = int(4.0 * sigma_pixels + 0.5)
    offsets: Float64[Array, " n_kernel"] = jnp.arange(
        -radius, radius + 1, dtype=jnp.float64
    )
    unnormalized: Float64[Array, " n_kernel"] = jnp.exp(
        -0.5 * jnp.square(offsets / sigma_pixels)
    )
    kernel: Float64[Array, " n_kernel"] = unnormalized / jnp.sum(unnormalized)
    return kernel


def _convolve_trailing_reflect(
    values: Float64[Array, "... n_sample"],
    kernel: Float64[Array, " n_kernel"],
) -> Float64[Array, "... n_sample"]:
    """PRIVATE: Convolve the trailing axis with reflect boundaries.

    Parameters
    ----------
    values : Float64[Array, "... n_sample"]
        Values on a uniform trailing-axis grid.
    kernel : Float64[Array, " n_kernel"]
        Odd one-dimensional correlation kernel.

    Returns
    -------
    convolved : Float64[Array, "... n_sample"]
        Array with the same shape as the input.

    Notes
    -----
    JAX's symmetric padding is half-sample symmetric, matching
    scipy.ndimage's reflect mode.  The symmetric Gaussian makes correlation
    and convolution identical.
    """
    length: int = values.shape[-1]
    flattened: Float64[Array, "n_row n_sample"] = values.reshape((-1, length))
    radius: int = (kernel.shape[0] - 1) // 2
    padded: Float64[Array, "n_row n_padded"] = jnp.pad(
        flattened,
        ((0, 0), (radius, radius)),
        mode="symmetric",
    )
    correlated: Float64[Array, "n_row n_sample 1"] = (
        jax.lax.conv_general_dilated(
            padded[..., None],
            kernel[:, None, None],
            window_strides=(1,),
            padding="VALID",
            dimension_numbers=("NWC", "WIO", "NWC"),
        )
    )
    convolved: Float64[Array, "... n_sample"] = correlated[..., 0].reshape(
        values.shape
    )
    return convolved


def matched_resolution(
    raw: Float64[NDArray, "ky kx omega"],
    kx_axis: Float64[NDArray, " kx"],
    omega_axis: Float64[NDArray, " omega"],
    *,
    energy_fwhm_ev: float,
    momentum_fwhm_inv_ang: float,
) -> Float64[NDArray, "ky kx omega"]:
    """Apply the isolated matched sampled-Gaussian adapter.

    Parameters
    ----------
    raw : Float64[NDArray, "ky kx omega"]
        Authenticated Chinook pre-resolution intensity cut.
    kx_axis : Float64[NDArray, " kx"]
        Frozen momentum axis in inverse Angstrom.
    omega_axis : Float64[NDArray, " omega"]
        Frozen relative-energy axis in eV.
    energy_fwhm_ev : float
        Energy-resolution FWHM in eV.
    momentum_fwhm_inv_ang : float
        Momentum-resolution FWHM in inverse Angstrom.

    Returns
    -------
    broadened : Float64[NDArray, "ky kx omega"]
        Response-broadened cut in Chinook's ky-kx-omega axis order.

    Notes
    -----
    Apply the energy kernel first and the momentum kernel second. This order
    matches the pinned Chinook call to ``scipy.ndimage.gaussian_filter``.
    """
    sigma_energy: float = _sigma_pixels(
        omega_axis,
        energy_fwhm_ev,
        chinook_count_over_range=True,
    )
    sigma_momentum: float = _sigma_pixels(
        kx_axis,
        momentum_fwhm_inv_ang,
        chinook_count_over_range=True,
    )
    values: Float64[Array, "ky kx omega"] = _convolve_trailing_reflect(
        jnp.asarray(raw),
        _sampled_kernel(sigma_energy),
    )
    momentum_last: Float64[Array, "ky omega kx"] = jnp.swapaxes(values, -1, -2)
    convolved: Float64[Array, "ky omega kx"] = _convolve_trailing_reflect(
        momentum_last,
        _sampled_kernel(sigma_momentum),
    )
    broadened: Float64[NDArray, "ky kx omega"] = np.asarray(
        jnp.swapaxes(convolved, -1, -2)
    )
    return broadened


def wrong_nominal_spacing_resolution(
    raw: Float64[NDArray, "ky kx omega"],
    kx_axis: Float64[NDArray, " kx"],
    omega_axis: Float64[NDArray, " omega"],
    *,
    energy_fwhm_ev: float,
    momentum_fwhm_inv_ang: float,
) -> Float64[NDArray, "ky kx omega"]:
    """Apply the planted N-1/range spacing defect.

    Parameters
    ----------
    raw : Float64[NDArray, "ky kx omega"]
        Authenticated Chinook pre-resolution intensity cut.
    kx_axis : Float64[NDArray, " kx"]
        Frozen momentum axis in inverse Angstrom.
    omega_axis : Float64[NDArray, " omega"]
        Frozen relative-energy axis in eV.
    energy_fwhm_ev : float
        Energy-resolution FWHM in eV.
    momentum_fwhm_inv_ang : float
        Momentum-resolution FWHM in inverse Angstrom.

    Returns
    -------
    broadened : Float64[NDArray, "ky kx omega"]
        Deliberately incorrect response-broadened cut.

    Notes
    -----
    Change only the pixels-per-coordinate conversion.
    """
    sigma_energy: float = _sigma_pixels(
        omega_axis,
        energy_fwhm_ev,
        chinook_count_over_range=False,
    )
    sigma_momentum: float = _sigma_pixels(
        kx_axis,
        momentum_fwhm_inv_ang,
        chinook_count_over_range=False,
    )
    values: Float64[Array, "ky kx omega"] = _convolve_trailing_reflect(
        jnp.asarray(raw),
        _sampled_kernel(sigma_energy),
    )
    convolved: Float64[Array, "ky omega kx"] = _convolve_trailing_reflect(
        jnp.swapaxes(values, -1, -2),
        _sampled_kernel(sigma_momentum),
    )
    broadened: Float64[NDArray, "ky kx omega"] = np.asarray(
        jnp.swapaxes(convolved, -1, -2)
    )
    return broadened


def wrong_axis_order_resolution(
    raw: Float64[NDArray, "ky kx omega"],
    kx_axis: Float64[NDArray, " kx"],
    omega_axis: Float64[NDArray, " omega"],
    *,
    energy_fwhm_ev: float,
    momentum_fwhm_inv_ang: float,
) -> Float64[NDArray, "ky kx omega"]:
    """Apply the planted energy/momentum kernel-axis swap.

    Parameters
    ----------
    raw : Float64[NDArray, "ky kx omega"]
        Authenticated Chinook pre-resolution intensity cut.
    kx_axis : Float64[NDArray, " kx"]
        Frozen momentum axis in inverse Angstrom.
    omega_axis : Float64[NDArray, " omega"]
        Frozen relative-energy axis in eV.
    energy_fwhm_ev : float
        Energy-resolution FWHM in eV.
    momentum_fwhm_inv_ang : float
        Momentum-resolution FWHM in inverse Angstrom.

    Returns
    -------
    broadened : Float64[NDArray, "ky kx omega"]
        Deliberately incorrect response-broadened cut.

    Notes
    -----
    Retain the correct sigma values. Assign them deliberately to the wrong
    axes.
    """
    sigma_energy: float = _sigma_pixels(
        omega_axis,
        energy_fwhm_ev,
        chinook_count_over_range=True,
    )
    sigma_momentum: float = _sigma_pixels(
        kx_axis,
        momentum_fwhm_inv_ang,
        chinook_count_over_range=True,
    )
    values: Float64[Array, "ky kx omega"] = _convolve_trailing_reflect(
        jnp.asarray(raw),
        _sampled_kernel(sigma_momentum),
    )
    convolved: Float64[Array, "ky omega kx"] = _convolve_trailing_reflect(
        jnp.swapaxes(values, -1, -2),
        _sampled_kernel(sigma_energy),
    )
    broadened: Float64[NDArray, "ky kx omega"] = np.asarray(
        jnp.swapaxes(convolved, -1, -2)
    )
    return broadened


def public_sampled_resolution(
    raw: Float64[NDArray, "ky kx omega"],
    kx_axis: Float64[NDArray, " kx"],
    omega_axis: Float64[NDArray, " omega"],
    *,
    energy_fwhm_ev: float,
    momentum_fwhm_inv_ang: float,
) -> Float64[NDArray, "ky kx omega"]:
    """Evaluate the production sampled long-tail convolution diagnostic.

    Parameters
    ----------
    raw : Float64[NDArray, "ky kx omega"]
        Authenticated Chinook pre-resolution intensity cut.
    kx_axis : Float64[NDArray, " kx"]
        Frozen momentum axis in inverse Angstrom.
    omega_axis : Float64[NDArray, " omega"]
        Frozen relative-energy axis in eV.
    energy_fwhm_ev : float
        Energy-resolution FWHM in eV.
    momentum_fwhm_inv_ang : float
        Momentum-resolution FWHM in inverse Angstrom.

    Returns
    -------
    broadened : Float64[NDArray, "ky kx omega"]
        Production-helper diagnostic, never a parity acceptance candidate.

    Notes
    -----
    Adjust coordinate sigmas for Chinook's N/range pixel convention. Retain
    the public helper's longer fixed support intentionally.
    """
    sigma_energy_pixels: float = _sigma_pixels(
        omega_axis,
        energy_fwhm_ev,
        chinook_count_over_range=True,
    )
    sigma_momentum_pixels: float = _sigma_pixels(
        kx_axis,
        momentum_fwhm_inv_ang,
        chinook_count_over_range=True,
    )
    energy_step: float = float(omega_axis[1] - omega_axis[0])
    momentum_step: float = float(kx_axis[1] - kx_axis[0])
    energy_blurred: Float64[Array, "ky kx omega"] = convolve_energy(
        jnp.asarray(raw),
        jnp.asarray(omega_axis),
        sigma_energy_pixels * energy_step,
    )
    momentum_blurred: Float64[Array, "ky omega kx"] = convolve_energy(
        jnp.swapaxes(energy_blurred, -1, -2),
        jnp.asarray(kx_axis),
        sigma_momentum_pixels * momentum_step,
    )
    broadened: Float64[NDArray, "ky kx omega"] = np.asarray(
        jnp.swapaxes(momentum_blurred, -1, -2)
    )
    return broadened


def comparison_metrics(
    candidate: Float64[NDArray, "ky kx omega"],
    reference: Float64[NDArray, "ky kx omega"],
    kx_axis: Float64[NDArray, " kx"],
    omega_axis: Float64[NDArray, " omega"],
    *,
    energy_fwhm_ev: float,
    momentum_fwhm_inv_ang: float,
) -> Dict[str, object]:
    """Measure one scale and evaluate the frozen parity envelopes.

    Parameters
    ----------
    candidate : Float64[NDArray, "ky kx omega"]
        Candidate response-broadened cut.
    reference : Float64[NDArray, "ky kx omega"]
        Chinook response-broadened reference cut.
    kx_axis : Float64[NDArray, " kx"]
        Frozen momentum axis in inverse Angstrom.
    omega_axis : Float64[NDArray, " omega"]
        Frozen relative-energy axis in eV.
    energy_fwhm_ev : float
        Energy-resolution FWHM in eV.
    momentum_fwhm_inv_ang : float
        Momentum-resolution FWHM in inverse Angstrom.

    Returns
    -------
    metrics : Dict[str, object]
        Crop, fitted scale, error statistics, and both acceptance booleans.

    Notes
    -----
    Fit exactly one nonnegative least-squares scale on the five-sigma
    interior. Relative diagnostics mask values at or below the registered
    elementwise absolute tolerance.
    """
    sigma_energy: float = _sigma_pixels(
        omega_axis,
        energy_fwhm_ev,
        chinook_count_over_range=True,
    )
    sigma_momentum: float = _sigma_pixels(
        kx_axis,
        momentum_fwhm_inv_ang,
        chinook_count_over_range=True,
    )
    crop_energy: int = int(math.ceil(5.0 * sigma_energy))
    crop_momentum: int = int(math.ceil(5.0 * sigma_momentum))
    selection: Tuple[slice, slice, slice] = (
        slice(None),
        slice(crop_momentum, -crop_momentum),
        slice(crop_energy, -crop_energy),
    )
    candidate_interior: Float64[NDArray, "ky kx omega"] = candidate[selection]
    reference_interior: Float64[NDArray, "ky kx omega"] = reference[selection]
    denominator: float = float(np.vdot(candidate_interior, candidate_interior))
    numerator: float = float(np.vdot(candidate_interior, reference_interior))
    scale: float = max(0.0, numerator / denominator)
    delta: Float64[NDArray, "ky kx omega"] = (
        scale * candidate_interior - reference_interior
    )
    absolute_error: Float64[NDArray, "ky kx omega"] = np.abs(delta)
    peak: float = float(np.max(np.abs(reference_interior)))
    atol: float = 1.0e-7 * peak
    envelope: Float64[NDArray, "ky kx omega"] = atol + 1.0e-4 * np.abs(
        reference_interior
    )
    relative_mask: Bool[NDArray, "ky kx omega"] = (
        np.abs(reference_interior) > atol
    )
    relative_error: Float64[NDArray, " relative"] = absolute_error[
        relative_mask
    ] / np.abs(reference_interior[relative_mask])
    metrics: Dict[str, object] = {
        "crop_cells_axis_order_ky_kx_omega": [
            0,
            crop_momentum,
            crop_energy,
        ],
        "interior_shape": list(reference_interior.shape),
        "profiled_nonnegative_scale": scale,
        "max_absolute_error": float(np.max(absolute_error)),
        "max_absolute_error_over_peak": float(np.max(absolute_error) / peak),
        "max_relative_error_above_atol": float(np.max(relative_error)),
        "p999_relative_error_above_atol": float(
            np.quantile(relative_error, 0.999)
        ),
        "candidate_to_reference_integral_ratio_full_grid": float(
            np.sum(candidate) / np.sum(reference)
        ),
        "elementwise_pass": bool(np.all(absolute_error <= envelope)),
        "strict_peak_scaled_pass": bool(
            np.max(absolute_error) <= 1.0e-6 * peak
        ),
    }
    return metrics
