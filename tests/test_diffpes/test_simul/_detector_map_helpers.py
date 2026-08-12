"""Provide shared fixtures for the mirrored split test modules.

The helpers preserve the inputs and independent reference calculations.
"""

import jax.numpy as jnp
from jaxtyping import Array, Float64

from diffpes.types import (
    ArpesCube,
    ArpesSpectrum,
    DetectorCalibration,
    DetectorEffects,
    ExperimentGeometry,
    make_arpes_cube,
    make_arpes_spectrum,
    make_detector_calibration,
    make_detector_effects,
    make_experiment_geometry,
)

_FRAME_ID: str = "org.diffpes.frame.sample_cartesian"


def _geometry(
    *, sample_azimuth: Float64[Array, ""] = jnp.array(0.0), slit: str = "H"
) -> ExperimentGeometry:
    """PRIVATE: Build a generic positive-energy experiment geometry.

    Parameters
    ----------
    sample_azimuth : Float64[Array, ""], optional
        Traced sample-to-laboratory azimuth. Default is zero.
    slit : str, optional
        Static detector slit orientation. Default is ``"H"``.

    Returns
    -------
    geometry : ExperimentGeometry
        Validated geometry at 50 eV photon energy.
    """
    geometry: ExperimentGeometry = make_experiment_geometry(
        photon_energy_ev=50.0,
        polarization=jnp.array([1.0 + 0.0j, 0.0j, 0.0j]),
        sample_azimuth=sample_azimuth,
        work_function_ev=4.0,
        slit=slit,
    )
    return geometry


def _cube() -> ArpesCube:
    """PRIVATE: Build an asymmetric positive affine source cube.

    Returns
    -------
    cube : ArpesCube
        Three-by-three-by-three sample-frame source density.
    """
    kx_axis: Float64[Array, " x"] = jnp.array([-0.5, 0.0, 0.5])
    ky_axis: Float64[Array, " y"] = jnp.array([-0.45, 0.05, 0.55])
    energy_axis: Float64[Array, " e"] = jnp.array([-0.4, 0.0, 0.4])
    intensity: Float64[Array, "x y e"] = (
        2.0
        + 0.35 * kx_axis[:, None, None]
        + 0.22 * ky_axis[None, :, None]
        + 0.17 * energy_axis[None, None, :]
    )
    cube: ArpesCube = make_arpes_cube(intensity, kx_axis, ky_axis, energy_axis)
    return cube


def _map_calibration() -> DetectorCalibration:
    """PRIVATE: Build an unequal-bin two-dimensional detector target.

    Returns
    -------
    calibration : DetectorCalibration
        Explicit target fully inside the affine cube fixture.
    """
    calibration: DetectorCalibration = make_detector_calibration(
        u_bin_edges=jnp.array([-0.055, -0.012, 0.047]),
        v_bin_edges=jnp.array([-0.048, 0.009, 0.052]),
        energy_bin_edges_ev=jnp.array([-0.24, -0.03, 0.21]),
        psf_fwhm_u=0.01,
        psf_fwhm_v=0.012,
        psf_fwhm_energy_ev=0.02,
        transmission_reference_domain_ev=jnp.array([40.0, 55.0]),
    )
    return calibration


def _spectrum(direction: str = "x") -> ArpesSpectrum:
    """PRIVATE: Build equal-length Gamma-to-X or Gamma-to-Y source cuts.

    Parameters
    ----------
    direction : str, optional
        Cartesian path direction, exactly ``"x"`` or ``"y"``. Default is
        ``"x"``.

    Returns
    -------
    spectrum : ArpesSpectrum
        Self-describing three-point line density.

    Raises
    ------
    ValueError
        The direction selector accepts only ``"x"`` or ``"y"``.
    """
    if direction == "x":
        points: Float64[Array, "k 3"] = jnp.array(
            [[-0.2, 0.0, 0.0], [0.0, 0.0, 0.0], [0.2, 0.0, 0.0]]
        )
    elif direction == "y":
        points = jnp.array(
            [[0.0, -0.2, 0.0], [0.0, 0.0, 0.0], [0.0, 0.2, 0.0]]
        )
    else:
        raise ValueError("spectrum direction must be x or y")
    k_axis: Float64[Array, " k"] = jnp.array([0.0, 0.2, 0.4])
    energy_axis: Float64[Array, " e"] = jnp.array([-0.2, 0.0, 0.2])
    intensity: Float64[Array, "k e"] = (
        1.4 + 0.5 * k_axis[:, None] + 0.2 * energy_axis[None, :]
    )
    spectrum: ArpesSpectrum = make_arpes_spectrum(
        intensity, energy_axis, k_axis, points
    )
    return spectrum


def _slit_calibration() -> DetectorCalibration:
    """PRIVATE: Build an explicit one-bin transverse slit target.

    Returns
    -------
    calibration : DetectorCalibration
        Unequal active ``u``/energy bins and one declared ``v`` aperture.
    """
    calibration: DetectorCalibration = make_detector_calibration(
        u_bin_edges=jnp.array([-0.048, -0.007, 0.044]),
        v_bin_edges=jnp.array([-0.10, 0.10]),
        energy_bin_edges_ev=jnp.array([-0.18, 0.01, 0.17]),
        psf_fwhm_u=0.01,
        psf_fwhm_v=0.012,
        psf_fwhm_energy_ev=0.02,
        transmission_reference_domain_ev=jnp.array([40.0, 55.0]),
    )
    return calibration


def _effects(
    logits: Float64[Array, " d"], rotations: Float64[Array, "d 3"]
) -> DetectorEffects:
    """PRIVATE: Build detector state for a domain-mixture fixture.

    Parameters
    ----------
    logits : Float64[Array, " d"]
        Traced domain logits.
    rotations : Float64[Array, "d 3"]
        Active z--y--z rotations.

    Returns
    -------
    effects : DetectorEffects
        Validated effects with otherwise inert bounded stages.
    """
    effects: DetectorEffects = make_detector_effects(
        domain_logits=logits,
        domain_euler_angles_rad=rotations,
        transmission_raw_slopes=jnp.array([0.0, 0.0]),
        background_coefficients=jnp.array([0.0]),
        sensitivity_coefficients=jnp.array([]),
        exposure=1.0,
        background_mode="flat",
        sensitivity_mode="constant",
        domain_frame_ids=tuple(_FRAME_ID for _ in range(logits.shape[0])),
    )
    return effects
