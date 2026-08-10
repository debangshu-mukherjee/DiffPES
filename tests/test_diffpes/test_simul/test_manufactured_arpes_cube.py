"""Verify a manufactured coherent ARPES cube against analytic equations.

The frozen one-band fixture separates intrinsic spectral assembly, sampled
Fermi occupation, fixed-domain analyser transmission, Cartesian momentum
resolution, and sampled-energy resolution.  The checks compare every
intermediate voxel and the final cube without fitting an intensity scale.
Independent planted alternatives demonstrate that each seam is observable.
"""

from __future__ import annotations

import math

import jax.numpy as jnp
import numpy as np
import pytest
from beartype.typing import Any, Tuple
from jaxtyping import Bool, Float64, Int64
from scipy import integrate

from diffpes.simul import (
    apply_transmission,
    assemble_spectral_intensity_bands_chunk,
    convolve_energy,
    convolve_momentum_map,
)
from diffpes.types import (
    KB_EV_PER_K,
    DetectorCalibration,
    make_detector_calibration,
    make_self_energy_model,
)

_SOURCE_RTOL: float = 1.0e-10
_FINAL_RTOL: float = 1.0e-8
_SOURCE_ATOL: float = 1.0e-14
_FINAL_ATOL: float = 1.0e-13
_SPECTRAL_WEIGHT_RTOL: float = 1.0e-8
_N_K: int = 9
_N_ENERGY: int = 401
_ETA_EV: float = 1.0e-4
_GAMMA_EV: float = 0.04
_TEMPERATURE_K: float = 30.0
_PHOTON_ENERGY_EV: float = 50.0
_WORK_FUNCTION_EV: float = 4.5
_MOMENTUM_HALF_WIDTH: int = 12
_ENERGY_HALF_WIDTH: int = 12


def _axes_and_band() -> Tuple[
    Float64[np.ndarray, " kx"],
    Float64[np.ndarray, " ky"],
    Float64[np.ndarray, " energy"],
    Float64[np.ndarray, "kx ky"],
]:
    """PRIVATE: Return the frozen Cartesian axes and diagonal band.

    Returns
    -------
    fixture : Tuple[Float64[np.ndarray, "..."], ...]
        Momentum axes, sampled relative-energy axis, and band-energy raster.
    """
    kx: Float64[np.ndarray, " kx"] = np.linspace(
        -0.20, 0.20, _N_K, dtype=np.float64
    )
    ky: Float64[np.ndarray, " ky"] = np.linspace(
        -0.20, 0.20, _N_K, dtype=np.float64
    )
    energy: Float64[np.ndarray, " energy"] = np.linspace(
        -0.60, 0.40, _N_ENERGY, dtype=np.float64
    )
    mesh_x: Float64[np.ndarray, "kx ky"]
    mesh_y: Float64[np.ndarray, "kx ky"]
    mesh_x, mesh_y = np.meshgrid(kx, ky, indexing="ij")
    band: Float64[np.ndarray, "kx ky"] = -0.12 + 0.80 * (mesh_x**2 + mesh_y**2)
    return kx, ky, energy, band


def _analytic_source(
    band: Float64[np.ndarray, "kx ky"],
    energy: Float64[np.ndarray, " energy"],
) -> Tuple[
    Float64[np.ndarray, "kx ky energy"],
    Float64[np.ndarray, " energy"],
    Float64[np.ndarray, "kx ky energy"],
]:
    """PRIVATE: Evaluate the analytic unit-matrix-element source cube.

    Parameters
    ----------
    band : Float64[np.ndarray, "kx ky"]
        One diagonal band on the Cartesian momentum raster.
    energy : Float64[np.ndarray, " energy"]
        Sampled energy relative to the Fermi level.

    Returns
    -------
    result : Tuple[Float64[np.ndarray, "..."], ...]
        Lorentzian spectral cube, sampled occupation, and occupied source.
    """
    width: float = _GAMMA_EV + _ETA_EV
    displacement: Float64[np.ndarray, "kx ky energy"] = (
        energy[None, None, :] - band[:, :, None]
    )
    spectral: Float64[np.ndarray, "kx ky energy"] = width / (
        np.pi * (displacement**2 + width**2)
    )
    occupation: Float64[np.ndarray, " energy"] = 1.0 / (
        1.0
        + np.exp(
            energy / (float(KB_EV_PER_K) * _TEMPERATURE_K),
            dtype=np.float64,
        )
    )
    source: Float64[np.ndarray, "kx ky energy"] = (
        spectral * occupation[None, None, :]
    )
    return spectral, occupation, source


def _calibration() -> DetectorCalibration:
    """PRIVATE: Build the frozen transmission-domain carrier.

    Returns
    -------
    calibration : DetectorCalibration
        Fixed-domain calibration used only by the declared transmission.
    """
    calibration: DetectorCalibration = make_detector_calibration(
        u_bin_edges=jnp.asarray([-0.25, 0.25]),
        v_bin_edges=jnp.asarray([-0.25, 0.25]),
        energy_bin_edges_ev=jnp.asarray([-0.605, 0.405]),
        psf_fwhm_u=0.01,
        psf_fwhm_v=0.01,
        psf_fwhm_energy_ev=0.01,
        transmission_reference_domain_ev=jnp.asarray([44.9, 45.9]),
    )
    return calibration


def _analytic_transmission(
    kinetic_energy: Float64[np.ndarray, " energy"],
    raw_slopes: Float64[np.ndarray, " slopes"],
) -> Float64[np.ndarray, " energy"]:
    """PRIVATE: Evaluate the two-coordinate transmission independently.

    Parameters
    ----------
    kinetic_energy : Float64[np.ndarray, " energy"]
        True kinetic-energy samples on the fixed calibration domain.
    raw_slopes : Float64[np.ndarray, " slopes"]
        Two raw Bernstein derivative coordinates.

    Returns
    -------
    transmission : Float64[np.ndarray, " energy"]
        Positive increasing response with continuous-domain mean one.

    Notes
    -----
    For degree one, integrating the Bernstein derivative gives
    ``s0 * (x - x**2 / 2) + s1 * x**2 / 2``.  SciPy quadrature supplies an
    independent normalization rather than reusing production quadrature.
    """
    slopes: Float64[np.ndarray, " slopes"] = np.logaddexp(0.0, raw_slopes)

    def anchored_log_response(x: Any) -> Any:
        """Return the analytic integrated-Bernstein log response."""
        return slopes[0] * (x - 0.5 * x**2) + 0.5 * slopes[1] * x**2

    mean: float = integrate.quad(
        lambda x: math.exp(float(anchored_log_response(x))),
        0.0,
        1.0,
        epsabs=1.0e-13,
        epsrel=1.0e-13,
        limit=100,
    )[0]
    normalized_energy: Float64[np.ndarray, " energy"] = (
        kinetic_energy - 44.9
    ) / (45.9 - 44.9)
    transmission: Float64[np.ndarray, " energy"] = (
        np.exp(anchored_log_response(normalized_energy)) / mean
    )
    return transmission


def _sampled_gaussian_matrix(
    size: int, sigma_pixels: float, half_width: int
) -> Float64[np.ndarray, "target source"]:
    """PRIVATE: Build an independent zero-exterior convolution matrix.

    Parameters
    ----------
    size : int
        Number of source and target samples.
    sigma_pixels : float
        Gaussian sigma in sample-spacing units.
    half_width : int
        Static sampled support on either side of the centre.

    Returns
    -------
    matrix : Float64[np.ndarray, "target source"]
        Target-by-source sampled convolution matrix without edge renormalizing.
    """
    offsets: Int64[np.ndarray, " taps"] = np.arange(
        -half_width, half_width + 1
    )
    kernel: Float64[np.ndarray, " taps"] = np.exp(
        -0.5 * (offsets / sigma_pixels) ** 2
    )
    kernel = kernel / np.sum(kernel)
    target: Int64[np.ndarray, "target 1"] = np.arange(size)[:, None]
    source: Int64[np.ndarray, "1 source"] = np.arange(size)[None, :]
    difference: Int64[np.ndarray, "target source"] = target - source
    inside: Bool[np.ndarray, "target source"] = (
        np.abs(difference) <= half_width
    )
    matrix: Float64[np.ndarray, "target source"] = np.zeros(
        (size, size), dtype=np.float64
    )
    matrix[inside] = kernel[difference[inside] + half_width]
    return matrix


def _direct_separable_convolution(
    intensity: Float64[np.ndarray, "kx ky energy"],
    momentum_matrix: Float64[np.ndarray, "k_target k_source"],
    energy_matrix: Float64[np.ndarray, "e_target e_source"],
) -> Tuple[
    Float64[np.ndarray, "kx ky energy"],
    Float64[np.ndarray, "kx ky energy"],
]:
    """PRIVATE: Apply the three sampled matrices from their equations.

    Parameters
    ----------
    intensity : Float64[np.ndarray, "kx ky energy"]
        Pre-resolution transmitted cube.
    momentum_matrix : Float64[np.ndarray, "k_target k_source"]
        Shared Cartesian target-by-source momentum matrix.
    energy_matrix : Float64[np.ndarray, "e_target e_source"]
        Target-by-source sampled-energy matrix.

    Returns
    -------
    result : Tuple[Float64[np.ndarray, "kx ky energy"], ...]
        Post-momentum cube and final post-energy cube.
    """
    after_kx: Float64[np.ndarray, "kx ky energy"] = np.einsum(
        "ia,abe->ibe", momentum_matrix, intensity
    )
    after_momentum: Float64[np.ndarray, "kx ky energy"] = np.einsum(
        "jb,ibe->ije", momentum_matrix, after_kx
    )
    final: Float64[np.ndarray, "kx ky energy"] = np.einsum(
        "me,ije->ijm", energy_matrix, after_momentum
    )
    return after_momentum, final


def _assert_planted_alternative_fails(
    alternative: Float64[np.ndarray, "..."],
    truth: Float64[np.ndarray, "..."],
    *,
    rtol: float,
    atol: float,
) -> None:
    """PRIVATE: Assert one planted alternative cannot pass the truth gate.

    Parameters
    ----------
    alternative : Float64[np.ndarray, "..."]
        Deliberately incorrect seam or final cube.
    truth : Float64[np.ndarray, "..."]
        Independently derived accepted value.
    rtol : float
        Relative tolerance of the corresponding accepted comparison.
    atol : float
        Absolute tolerance of the corresponding accepted comparison.
    """
    with pytest.raises(AssertionError):
        np.testing.assert_allclose(alternative, truth, rtol=rtol, atol=atol)


class TestManufacturedArpesCubeTruth:
    """Verify every seam of the frozen unit-matrix-element cube."""

    def test_every_intrinsic_voxel_and_finite_window_weight(self) -> None:
        """Match the analytic Lorentzian and sampled Fermi occupation.

        Every source voxel is absolute-scale checked.  Adaptive quadrature of
        the same normalized spectral equation independently matches its
        finite-window arctangent integral for all 81 momenta.

        Notes
        -----
        Planted alternatives evaluate occupation at the band pole, omit eta,
        or replace the finite-window weight by the infinite-domain value.
        """
        energy: Float64[np.ndarray, " energy"]
        band: Float64[np.ndarray, "kx ky"]
        _, _, energy, band = _axes_and_band()
        spectral: Float64[np.ndarray, "kx ky energy"]
        occupation: Float64[np.ndarray, " energy"]
        expected_source: Float64[np.ndarray, "kx ky energy"]
        spectral, occupation, expected_source = _analytic_source(band, energy)
        flat_band: Float64[np.ndarray, "k 1"] = band.reshape((-1, 1))
        unit_weights: Float64[np.ndarray, "k energy 1"] = np.ones(
            (flat_band.shape[0], energy.size, 1), dtype=np.float64
        )
        produced_flat: Float64[np.ndarray, "k energy"] = np.asarray(
            assemble_spectral_intensity_bands_chunk(
                jnp.asarray(flat_band),
                jnp.asarray(unit_weights),
                jnp.asarray(energy),
                make_self_energy_model(gamma=_GAMMA_EV),
                jnp.asarray(0.0),
                _TEMPERATURE_K,
                _ETA_EV,
            )
        )
        produced: Float64[np.ndarray, "kx ky energy"] = produced_flat.reshape(
            (_N_K, _N_K, _N_ENERGY)
        )
        np.testing.assert_allclose(
            produced,
            expected_source,
            rtol=_SOURCE_RTOL,
            atol=_SOURCE_ATOL,
        )

        width: float = _GAMMA_EV + _ETA_EV
        analytic_weights: Float64[np.ndarray, "kx ky"] = (
            np.arctan((energy[-1] - band) / width)
            - np.arctan((energy[0] - band) / width)
        ) / np.pi
        quadrature_weights: Float64[np.ndarray, "kx ky"] = np.empty_like(band)
        index: Tuple[int, ...]
        for index in np.ndindex(band.shape):
            pole: float = float(band[index])
            quadrature_weights[index] = integrate.quad(
                lambda omega: (
                    width / (np.pi * ((omega - pole) ** 2 + width**2))
                ),
                float(energy[0]),
                float(energy[-1]),
                epsabs=1.0e-12,
                epsrel=1.0e-12,
                points=(pole,),
                limit=200,
            )[0]
        maximum_relative_error: float = float(
            np.max(
                np.abs(quadrature_weights - analytic_weights)
                / analytic_weights
            )
        )
        assert maximum_relative_error <= _SPECTRAL_WEIGHT_RTOL

        pole_occupation: Float64[np.ndarray, "kx ky"] = 1.0 / (
            1.0
            + np.exp(
                band / (float(KB_EV_PER_K) * _TEMPERATURE_K),
                dtype=np.float64,
            )
        )
        _assert_planted_alternative_fails(
            spectral * pole_occupation[:, :, None],
            expected_source,
            rtol=_SOURCE_RTOL,
            atol=_SOURCE_ATOL,
        )
        width_without_eta: float = _GAMMA_EV
        spectral_without_eta: Float64[np.ndarray, "kx ky energy"] = (
            width_without_eta
            / (
                np.pi
                * (
                    (energy[None, None, :] - band[:, :, None]) ** 2
                    + width_without_eta**2
                )
            )
        )
        _assert_planted_alternative_fails(
            spectral_without_eta * occupation[None, None, :],
            expected_source,
            rtol=_SOURCE_RTOL,
            atol=_SOURCE_ATOL,
        )
        _assert_planted_alternative_fails(
            np.ones_like(analytic_weights),
            analytic_weights,
            rtol=_SPECTRAL_WEIGHT_RTOL,
            atol=0.0,
        )

    def test_transmission_resolution_and_every_final_voxel(self) -> None:
        """Match declared transmission and direct separable convolution.

        Production and independent calculations agree after transmission,
        after both momentum passes, and in every final energy-broadened voxel.

        Notes
        -----
        Planted alternatives omit transmission, renormalize lost edge mass,
        omit one momentum pass, or apply transmission after resolution.
        """
        kx: Float64[np.ndarray, " kx"]
        ky: Float64[np.ndarray, " ky"]
        energy: Float64[np.ndarray, " energy"]
        band: Float64[np.ndarray, "kx ky"]
        kx, ky, energy, band = _axes_and_band()
        source: Float64[np.ndarray, "kx ky energy"]
        _, _, source = _analytic_source(band, energy)
        raw_slopes: Float64[np.ndarray, " slopes"] = np.asarray(
            [-0.4, 0.2], dtype=np.float64
        )
        kinetic_energy: Float64[np.ndarray, " energy"] = (
            _PHOTON_ENERGY_EV - _WORK_FUNCTION_EV + energy
        )
        expected_transmission: Float64[np.ndarray, " energy"] = (
            _analytic_transmission(kinetic_energy, raw_slopes)
        )
        expected_transmitted: Float64[np.ndarray, "kx ky energy"] = (
            source * expected_transmission[None, None, :]
        )
        produced_transmitted: Float64[np.ndarray, "kx ky energy"] = np.asarray(
            apply_transmission(
                jnp.asarray(source),
                jnp.asarray(kinetic_energy),
                jnp.asarray(raw_slopes),
                _calibration(),
            )
        )
        np.testing.assert_allclose(
            produced_transmitted,
            expected_transmitted,
            rtol=_SOURCE_RTOL,
            atol=_SOURCE_ATOL,
        )

        momentum_spacing: float = float(kx[1] - kx[0])
        energy_spacing: float = float(energy[1] - energy[0])
        sigma_momentum: float = 1.15 * momentum_spacing
        sigma_energy: float = 1.20 * energy_spacing
        momentum_matrix: Float64[np.ndarray, "k_target k_source"] = (
            _sampled_gaussian_matrix(
                _N_K,
                sigma_momentum / momentum_spacing,
                _MOMENTUM_HALF_WIDTH,
            )
        )
        energy_matrix: Float64[np.ndarray, "e_target e_source"] = (
            _sampled_gaussian_matrix(
                _N_ENERGY,
                sigma_energy / energy_spacing,
                _ENERGY_HALF_WIDTH,
            )
        )
        expected_momentum: Float64[np.ndarray, "kx ky energy"]
        expected_final: Float64[np.ndarray, "kx ky energy"]
        expected_momentum, expected_final = _direct_separable_convolution(
            expected_transmitted, momentum_matrix, energy_matrix
        )
        produced_momentum: Float64[np.ndarray, "kx ky energy"] = np.asarray(
            convolve_momentum_map(
                jnp.asarray(produced_transmitted),
                jnp.asarray(kx),
                jnp.asarray(ky),
                sigma_momentum,
                half_width=_MOMENTUM_HALF_WIDTH,
            )
        )
        np.testing.assert_allclose(
            produced_momentum,
            expected_momentum,
            rtol=_SOURCE_RTOL,
            atol=_SOURCE_ATOL,
        )
        produced_final: Float64[np.ndarray, "kx ky energy"] = np.asarray(
            convolve_energy(
                jnp.asarray(produced_momentum),
                jnp.asarray(energy),
                sigma_energy,
                half_width=_ENERGY_HALF_WIDTH,
            )
        )
        np.testing.assert_allclose(
            produced_final,
            expected_final,
            rtol=_FINAL_RTOL,
            atol=_FINAL_ATOL,
        )

        _assert_planted_alternative_fails(
            source,
            expected_transmitted,
            rtol=_SOURCE_RTOL,
            atol=_SOURCE_ATOL,
        )
        row_normalized_momentum: Float64[np.ndarray, "k_target k_source"] = (
            momentum_matrix / np.sum(momentum_matrix, axis=1, keepdims=True)
        )
        wrong_boundary: Float64[np.ndarray, "kx ky energy"]
        wrong_boundary, _ = _direct_separable_convolution(
            expected_transmitted, row_normalized_momentum, energy_matrix
        )
        _assert_planted_alternative_fails(
            wrong_boundary,
            expected_momentum,
            rtol=_SOURCE_RTOL,
            atol=_SOURCE_ATOL,
        )
        one_momentum_pass: Float64[np.ndarray, "kx ky energy"] = np.einsum(
            "ia,abe->ibe", momentum_matrix, expected_transmitted
        )
        _assert_planted_alternative_fails(
            one_momentum_pass,
            expected_momentum,
            rtol=_SOURCE_RTOL,
            atol=_SOURCE_ATOL,
        )
        untransmitted_final: Float64[np.ndarray, "kx ky energy"]
        _, untransmitted_final = _direct_separable_convolution(
            source, momentum_matrix, energy_matrix
        )
        transmission_after_resolution: Float64[np.ndarray, "kx ky energy"] = (
            untransmitted_final * expected_transmission[None, None, :]
        )
        _assert_planted_alternative_fails(
            transmission_after_resolution,
            expected_final,
            rtol=_FINAL_RTOL,
            atol=_FINAL_ATOL,
        )
