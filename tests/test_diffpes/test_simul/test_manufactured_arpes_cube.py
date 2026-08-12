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
from numpy.typing import NDArray
from scipy import integrate

from diffpes.constants import (
    KB_EV_PER_K,
)
from diffpes.simul import (
    apply_transmission,
    assemble_spectral_intensity_bands_chunk,
    convolve_energy,
    convolve_momentum_map,
)
from diffpes.types import (
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
    Float64[NDArray, " kx"],
    Float64[NDArray, " ky"],
    Float64[NDArray, " energy"],
    Float64[NDArray, "kx ky"],
]:
    """PRIVATE: Return the frozen Cartesian axes and diagonal band.

    Returns
    -------
    fixture : Tuple[Float64[NDArray, "..."], ...]
        Momentum axes, sampled relative-energy axis, and band-energy raster.
    """
    kx: Float64[NDArray, " kx"] = np.linspace(
        -0.20, 0.20, _N_K, dtype=np.float64
    )
    ky: Float64[NDArray, " ky"] = np.linspace(
        -0.20, 0.20, _N_K, dtype=np.float64
    )
    energy: Float64[NDArray, " energy"] = np.linspace(
        -0.60, 0.40, _N_ENERGY, dtype=np.float64
    )
    mesh_x: Float64[NDArray, "kx ky"]
    mesh_y: Float64[NDArray, "kx ky"]
    mesh_x, mesh_y = np.meshgrid(kx, ky, indexing="ij")
    band: Float64[NDArray, "kx ky"] = -0.12 + 0.80 * (mesh_x**2 + mesh_y**2)
    returned: Tuple[
        Float64[NDArray, " kx"],
        Float64[NDArray, " ky"],
        Float64[NDArray, " energy"],
        Float64[NDArray, "kx ky"],
    ] = kx, ky, energy, band
    return returned


def _analytic_source(
    band: Float64[NDArray, "kx ky"],
    energy: Float64[NDArray, " energy"],
) -> Tuple[
    Float64[NDArray, "kx ky energy"],
    Float64[NDArray, " energy"],
    Float64[NDArray, "kx ky energy"],
]:
    """PRIVATE: Evaluate the analytic unit-matrix-element source cube.

    Parameters
    ----------
    band : Float64[NDArray, "kx ky"]
        One diagonal band on the Cartesian momentum raster.
    energy : Float64[NDArray, " energy"]
        Sampled energy relative to the Fermi level.

    Returns
    -------
    result : Tuple[Float64[NDArray, "..."], ...]
        Lorentzian spectral cube, sampled occupation, and occupied source.
    """
    width: float = _GAMMA_EV + _ETA_EV
    displacement: Float64[NDArray, "kx ky energy"] = (
        energy[None, None, :] - band[:, :, None]
    )
    spectral: Float64[NDArray, "kx ky energy"] = width / (
        np.pi * (displacement**2 + width**2)
    )
    occupation: Float64[NDArray, " energy"] = 1.0 / (
        1.0
        + np.exp(
            energy / (float(KB_EV_PER_K) * _TEMPERATURE_K),
            dtype=np.float64,
        )
    )
    source: Float64[NDArray, "kx ky energy"] = (
        spectral * occupation[None, None, :]
    )
    returned: Tuple[
        Float64[NDArray, "kx ky energy"],
        Float64[NDArray, " energy"],
        Float64[NDArray, "kx ky energy"],
    ] = spectral, occupation, source
    return returned


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
    kinetic_energy: Float64[NDArray, " energy"],
    raw_slopes: Float64[NDArray, " slopes"],
) -> Float64[NDArray, " energy"]:
    """PRIVATE: Evaluate the two-coordinate transmission independently.

    Parameters
    ----------
    kinetic_energy : Float64[NDArray, " energy"]
        True kinetic-energy samples on the fixed calibration domain.
    raw_slopes : Float64[NDArray, " slopes"]
        Two raw Bernstein derivative coordinates.

    Returns
    -------
    transmission : Float64[NDArray, " energy"]
        Positive increasing response with continuous-domain mean one.

    Notes
    -----
    For degree one, integrating the Bernstein derivative gives
    ``s0 * (x - x**2 / 2) + s1 * x**2 / 2``.  SciPy quadrature supplies an
    independent normalization rather than reusing production quadrature.
    """
    slopes: Float64[NDArray, " slopes"] = np.logaddexp(0.0, raw_slopes)

    def anchored_log_response(x: Any) -> Any:
        """Return the analytic integrated-Bernstein log response."""
        returned: Any = slopes[0] * (x - 0.5 * x**2) + 0.5 * slopes[1] * x**2
        return returned

    mean: float = integrate.quad(
        lambda x: math.exp(float(anchored_log_response(x))),
        0.0,
        1.0,
        epsabs=1.0e-13,
        epsrel=1.0e-13,
        limit=100,
    )[0]
    normalized_energy: Float64[NDArray, " energy"] = (
        kinetic_energy - 44.9
    ) / (45.9 - 44.9)
    transmission: Float64[NDArray, " energy"] = (
        np.exp(anchored_log_response(normalized_energy)) / mean
    )
    return transmission


def _sampled_gaussian_matrix(
    size: int, sigma_pixels: float, half_width: int
) -> Float64[NDArray, "target source"]:
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
    matrix : Float64[NDArray, "target source"]
        Target-by-source sampled convolution matrix without edge renormalizing.
    """
    offsets: Int64[NDArray, " taps"] = np.arange(-half_width, half_width + 1)
    kernel: Float64[NDArray, " taps"] = np.exp(
        -0.5 * (offsets / sigma_pixels) ** 2
    )
    kernel = kernel / np.sum(kernel)
    target: Int64[NDArray, "target 1"] = np.arange(size)[:, None]
    source: Int64[NDArray, "1 source"] = np.arange(size)[None, :]
    difference: Int64[NDArray, "target source"] = target - source
    inside: Bool[NDArray, "target source"] = np.abs(difference) <= half_width
    matrix: Float64[NDArray, "target source"] = np.zeros(
        (size, size), dtype=np.float64
    )
    matrix[inside] = kernel[difference[inside] + half_width]
    return matrix


def _direct_separable_convolution(
    intensity: Float64[NDArray, "kx ky energy"],
    momentum_matrix: Float64[NDArray, "k_target k_source"],
    energy_matrix: Float64[NDArray, "e_target e_source"],
) -> Tuple[
    Float64[NDArray, "kx ky energy"],
    Float64[NDArray, "kx ky energy"],
]:
    """PRIVATE: Apply the three sampled matrices from their equations.

    Parameters
    ----------
    intensity : Float64[NDArray, "kx ky energy"]
        Pre-resolution transmitted cube.
    momentum_matrix : Float64[NDArray, "k_target k_source"]
        Shared Cartesian target-by-source momentum matrix.
    energy_matrix : Float64[NDArray, "e_target e_source"]
        Target-by-source sampled-energy matrix.

    Returns
    -------
    result : Tuple[Float64[NDArray, "kx ky energy"], ...]
        Post-momentum cube and final post-energy cube.
    """
    after_kx: Float64[NDArray, "kx ky energy"] = np.einsum(
        "ia,abe->ibe", momentum_matrix, intensity
    )
    after_momentum: Float64[NDArray, "kx ky energy"] = np.einsum(
        "jb,ibe->ije", momentum_matrix, after_kx
    )
    final: Float64[NDArray, "kx ky energy"] = np.einsum(
        "me,ije->ijm", energy_matrix, after_momentum
    )
    returned: Tuple[
        Float64[NDArray, "kx ky energy"],
        Float64[NDArray, "kx ky energy"],
    ] = after_momentum, final
    return returned


def _assert_planted_alternative_fails(
    alternative: Float64[NDArray, "..."],
    truth: Float64[NDArray, "..."],
    *,
    rtol: float,
    atol: float,
) -> None:
    """PRIVATE: Assert one planted alternative cannot pass the truth check.

    Parameters
    ----------
    alternative : Float64[NDArray, "..."]
        Deliberately incorrect seam or final cube.
    truth : Float64[NDArray, "..."]
        Independently derived accepted value.
    rtol : float
        Relative tolerance of the corresponding accepted comparison.
    atol : float
        Absolute tolerance of the corresponding accepted comparison.
    """
    with pytest.raises(AssertionError, match="Not equal to tolerance"):
        np.testing.assert_allclose(alternative, truth, rtol=rtol, atol=atol)


class TestManufacturedArpesCubeTruth:
    """Verify every seam of the frozen unit-matrix-element cube.

    The cases compare intrinsic voxels, finite-window weights, transmission,
    resolution, and final detector voxels with independent manufactured truth.
    """

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
        energy: Float64[NDArray, " energy"]
        band: Float64[NDArray, "kx ky"]
        _, _, energy, band = _axes_and_band()
        spectral: Float64[NDArray, "kx ky energy"]
        occupation: Float64[NDArray, " energy"]
        expected_source: Float64[NDArray, "kx ky energy"]
        spectral, occupation, expected_source = _analytic_source(band, energy)
        flat_band: Float64[NDArray, "k 1"] = band.reshape((-1, 1))
        unit_weights: Float64[NDArray, "k energy 1"] = np.ones(
            (flat_band.shape[0], energy.size, 1), dtype=np.float64
        )
        produced_flat: Float64[NDArray, "k energy"] = np.asarray(
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
        produced: Float64[NDArray, "kx ky energy"] = produced_flat.reshape(
            (_N_K, _N_K, _N_ENERGY)
        )
        np.testing.assert_allclose(
            produced,
            expected_source,
            rtol=_SOURCE_RTOL,
            atol=_SOURCE_ATOL,
        )

        width: float = _GAMMA_EV + _ETA_EV
        analytic_weights: Float64[NDArray, "kx ky"] = (
            np.arctan((energy[-1] - band) / width)
            - np.arctan((energy[0] - band) / width)
        ) / np.pi
        quadrature_weights: Float64[NDArray, "kx ky"] = np.empty_like(band)
        index: Tuple[int, ...]
        for index in np.ndindex(band.shape):
            pole: float = float(band[index])

            def lorentzian_integrand(
                omega: float,
                pole_value: float = pole,
            ) -> float:
                """Evaluate the manufactured Lorentzian integrand."""
                value: float = width / (
                    np.pi * ((omega - pole_value) ** 2 + width**2)
                )
                return value

            quadrature_weights[index] = integrate.quad(
                lorentzian_integrand,
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

        pole_occupation: Float64[NDArray, "kx ky"] = 1.0 / (
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
        spectral_without_eta: Float64[NDArray, "kx ky energy"] = (
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
        kx: Float64[NDArray, " kx"]
        ky: Float64[NDArray, " ky"]
        energy: Float64[NDArray, " energy"]
        band: Float64[NDArray, "kx ky"]
        kx, ky, energy, band = _axes_and_band()
        source: Float64[NDArray, "kx ky energy"]
        _, _, source = _analytic_source(band, energy)
        raw_slopes: Float64[NDArray, " slopes"] = np.asarray(
            [-0.4, 0.2], dtype=np.float64
        )
        kinetic_energy: Float64[NDArray, " energy"] = (
            _PHOTON_ENERGY_EV - _WORK_FUNCTION_EV + energy
        )
        expected_transmission: Float64[NDArray, " energy"] = (
            _analytic_transmission(kinetic_energy, raw_slopes)
        )
        expected_transmitted: Float64[NDArray, "kx ky energy"] = (
            source * expected_transmission[None, None, :]
        )
        produced_transmitted: Float64[NDArray, "kx ky energy"] = np.asarray(
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
        momentum_matrix: Float64[NDArray, "k_target k_source"] = (
            _sampled_gaussian_matrix(
                _N_K,
                sigma_momentum / momentum_spacing,
                _MOMENTUM_HALF_WIDTH,
            )
        )
        energy_matrix: Float64[NDArray, "e_target e_source"] = (
            _sampled_gaussian_matrix(
                _N_ENERGY,
                sigma_energy / energy_spacing,
                _ENERGY_HALF_WIDTH,
            )
        )
        expected_momentum: Float64[NDArray, "kx ky energy"]
        expected_final: Float64[NDArray, "kx ky energy"]
        expected_momentum, expected_final = _direct_separable_convolution(
            expected_transmitted, momentum_matrix, energy_matrix
        )
        produced_momentum: Float64[NDArray, "kx ky energy"] = np.asarray(
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
        produced_final: Float64[NDArray, "kx ky energy"] = np.asarray(
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
        row_normalized_momentum: Float64[NDArray, "k_target k_source"] = (
            momentum_matrix / np.sum(momentum_matrix, axis=1, keepdims=True)
        )
        wrong_boundary: Float64[NDArray, "kx ky energy"]
        wrong_boundary, _ = _direct_separable_convolution(
            expected_transmitted, row_normalized_momentum, energy_matrix
        )
        _assert_planted_alternative_fails(
            wrong_boundary,
            expected_momentum,
            rtol=_SOURCE_RTOL,
            atol=_SOURCE_ATOL,
        )
        one_momentum_pass: Float64[NDArray, "kx ky energy"] = np.einsum(
            "ia,abe->ibe", momentum_matrix, expected_transmitted
        )
        _assert_planted_alternative_fails(
            one_momentum_pass,
            expected_momentum,
            rtol=_SOURCE_RTOL,
            atol=_SOURCE_ATOL,
        )
        untransmitted_final: Float64[NDArray, "kx ky energy"]
        _, untransmitted_final = _direct_separable_convolution(
            source, momentum_matrix, energy_matrix
        )
        transmission_after_resolution: Float64[NDArray, "kx ky energy"] = (
            untransmitted_final * expected_transmission[None, None, :]
        )
        _assert_planted_alternative_fails(
            transmission_after_resolution,
            expected_final,
            rtol=_FINAL_RTOL,
            atol=_FINAL_ATOL,
        )
