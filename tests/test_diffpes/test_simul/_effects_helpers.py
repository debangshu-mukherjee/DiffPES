"""Provide shared fixtures for the mirrored split test modules.

The helpers preserve the inputs and independent reference calculations.
"""

import math

import jax.numpy as jnp
import numpy as np
from beartype.typing import Dict, List, Tuple
from jaxtyping import Array, Float64
from numpy.typing import NDArray
from scipy import special

from diffpes.types import (
    ArpesCube,
    CrystalGeometry,
    DetectorCalibration,
    DetectorEffects,
    ExperimentGeometry,
    SurfaceCell,
    make_arpes_cube,
    make_crystal_geometry,
    make_detector_calibration,
    make_detector_effects,
    make_experiment_geometry,
    make_surface_cell,
)

_FRAME_ID: str = "org.diffpes.frame.sample_cartesian"


_DETERMINISTIC_RTOL: float = 1.0e-10


_SAMPLE_DRAWS: int = 200_000


_FWHM_TO_SIGMA: float = 1.0 / (2.0 * np.sqrt(2.0 * np.log(2.0)))


_FINITE_VOLUME_RTOL: float = 1.0e-11


_SAMPLED_HALF_WIDTH: int = 48


def _calibration(*, slit: bool = False) -> DetectorCalibration:
    """PRIVATE: Build an unequal-bin detector calibration.

    The fixture switches only the native ``v`` bin count.

    Parameters
    ----------
    slit : bool, optional
        Whether to create one native ``v`` bin. Default is ``False``.

    Returns
    -------
    calibration : DetectorCalibration
        Validated unequal-bin detector calibration.
    """
    v_edges: Float64[Array, "..."] = (
        jnp.array([-0.4, 0.6]) if slit else jnp.array([-0.4, 0.1, 0.8])
    )
    calibration: DetectorCalibration = make_detector_calibration(
        u_bin_edges=jnp.array([-1.0, -0.25, 1.5]),
        v_bin_edges=v_edges,
        energy_bin_edges_ev=jnp.array([-2.0, -0.75, 0.5, 2.5]),
        psf_fwhm_u=0.08,
        psf_fwhm_v=0.11,
        psf_fwhm_energy_ev=0.06,
        transmission_reference_domain_ev=jnp.array([12.0, 42.0]),
    )
    return calibration


def _effects(**overrides: object) -> DetectorEffects:
    """PRIVATE: Build valid one-domain detector effects.

    Keyword overrides select each focused physical fixture.

    Parameters
    ----------
    **overrides : object
        Values that replace valid flat-background defaults.

    Returns
    -------
    effects : DetectorEffects
        Validated detector-effects carrier.
    """
    parameters: Dict[str, object] = {
        "domain_logits": jnp.array([0.2]),
        "domain_euler_angles_rad": jnp.array([[0.1, -0.2, 0.3]]),
        "transmission_raw_slopes": jnp.array([0.15, -0.25]),
        "background_coefficients": jnp.array([0.1]),
        "sensitivity_coefficients": jnp.array([]),
        "exposure": 2.5,
        "background_mode": "flat",
        "sensitivity_mode": "constant",
        "domain_frame_ids": (_FRAME_ID,),
    }
    parameters.update(overrides)
    effects: DetectorEffects = make_detector_effects(**parameters)
    return effects


def _inverse_softplus(value: float) -> Float64[Array, "..."]:
    """PRIVATE: Return the raw coordinate for one positive amplitude.

    The transform creates exact physical amplitudes for analytic fixtures.

    Parameters
    ----------
    value : float
        Positive physical amplitude.

    Returns
    -------
    raw_value : Float64[Array, "..."]
        Unconstrained softplus coordinate.
    """
    raw_value: Float64[Array, "..."] = jnp.log(jnp.expm1(value))
    return raw_value


def _smooth_effects_fixture() -> Tuple[
    DetectorCalibration,
    Float64[Array, "..."],
    Float64[Array, "..."],
    Tuple[
        Float64[Array, "..."],
        Float64[Array, "..."],
        Float64[Array, "..."],
        Float64[Array, "..."],
    ],
]:
    """PRIVATE: Build the shared smooth detector-effects fixture.

    The fixture uses asymmetric values to expose every implemented leaf.

    Returns
    -------
    fixture : Tuple
        Calibration, density, loss weights, and continuous effects leaves.
    """
    calibration: DetectorCalibration = _calibration(slit=False)
    density: Float64[Array, "..."] = jnp.linspace(0.2, 1.4, 12).reshape(
        (1, 2, 2, 3)
    )
    weights: Float64[Array, "1 2 2 3"] = jnp.array(
        [
            [
                [[0.7, -0.2, 0.4], [0.1, 0.9, -0.5]],
                [[-0.3, 0.6, 1.1], [0.8, -0.7, 0.2]],
            ]
        ]
    )
    theta: Tuple[
        Float64[Array, "..."],
        Float64[Array, "..."],
        Float64[Array, "..."],
        Float64[Array, "..."],
    ] = (
        jnp.array([-0.2, 0.08, -0.05, 0.12, 0.04, -0.07, 0.03]),
        jnp.array([0.11, -0.06, 0.08, 0.03, -0.09, 0.05]),
        jnp.array(2.3),
        jnp.array([0.35, 0.7, 0.25]),
    )
    fixture: Tuple[
        DetectorCalibration,
        Float64[Array, "..."],
        Float64[Array, "..."],
        Tuple[
            Float64[Array, "..."],
            Float64[Array, "..."],
            Float64[Array, "..."],
            Float64[Array, "..."],
        ],
    ] = (calibration, density, weights, theta)
    return fixture


def _resolution_calibration(
    *,
    sign: int = 1,
    widths: Tuple[float, float, float] = (0.23, 0.17, 0.29),
) -> DetectorCalibration:
    """PRIVATE: Build the asymmetric native-resolution fixture.

    Parameters
    ----------
    sign : int, optional
        Registered analyser-transmission direction. Default is increasing.
    widths : Tuple[float, float, float], optional
        Native ``u``, ``v``, and energy FWHM values.

    Returns
    -------
    calibration : DetectorCalibration
        Unequal-bin calibration with a fixed kinetic-energy domain.
    """
    calibration: DetectorCalibration = make_detector_calibration(
        u_bin_edges=jnp.array([-0.7, -0.2, 0.15, 0.9]),
        v_bin_edges=jnp.array([-0.5, -0.05, 0.8]),
        energy_bin_edges_ev=jnp.array([-1.2, -0.6, 0.1, 0.55, 1.4]),
        psf_fwhm_u=widths[0],
        psf_fwhm_v=widths[1],
        psf_fwhm_energy_ev=widths[2],
        transmission_reference_domain_ev=jnp.array([12.0, 48.0]),
        transmission_monotonic_sign=sign,
    )
    return calibration


def _normal_second_antiderivative(
    displacement: Float64[NDArray, "..."], sigma: float
) -> Float64[NDArray, "..."]:
    """PRIVATE: Evaluate the independent Gaussian second antiderivative.

    Parameters
    ----------
    displacement : Float64[NDArray, "..."]
        Edge-to-edge displacements.
    sigma : float
        Positive Gaussian standard deviation.

    Returns
    -------
    values : Float64[NDArray, "..."]
        ``z * Phi(z/sigma) + sigma * phi(z/sigma)``.
    """
    scaled: Float64[NDArray, "..."] = displacement / sigma
    values: Float64[NDArray, "..."] = displacement * special.ndtr(
        scaled
    ) + sigma * np.exp(-0.5 * scaled**2) / np.sqrt(2.0 * np.pi)
    return values


def _reference_finite_volume_matrix(
    edges: Float64[NDArray, " Np1"], sigma: float
) -> Float64[NDArray, "N N"]:
    """PRIVATE: Build an independent analytic finite-volume Gaussian matrix.

    Parameters
    ----------
    edges : Float64[NDArray, " Np1"]
        Explicit common source and target cell edges.
    sigma : float
        Positive Gaussian standard deviation.

    Returns
    -------
    matrix : Float64[NDArray, "N N"]
        Output-density by input-density matrix without row normalization.
    """
    left: Float64[NDArray, " N"] = edges[:-1]
    right: Float64[NDArray, " N"] = edges[1:]
    integrated: Float64[NDArray, "N N"] = (
        _normal_second_antiderivative(right[:, None] - left[None, :], sigma)
        - _normal_second_antiderivative(left[:, None] - left[None, :], sigma)
        - _normal_second_antiderivative(right[:, None] - right[None, :], sigma)
        + _normal_second_antiderivative(left[:, None] - right[None, :], sigma)
    )
    matrix: Float64[NDArray, "N N"] = (
        np.maximum(integrated, 0.0) / np.diff(edges)[:, None]
    )
    return matrix


def _reference_resolution(
    density: Float64[NDArray, "... U V E"],
    calibration: DetectorCalibration,
) -> Tuple[Float64[NDArray, "... U V E"], Float64[NDArray, " 3"]]:
    """PRIVATE: Apply independent separable finite-volume resolution.

    Parameters
    ----------
    density : Float64[NDArray, "... U V E"]
        Input native-bin density.
    calibration : DetectorCalibration
        Native edges and FWHM widths.

    Returns
    -------
    blurred : Float64[NDArray, "... U V E"]
        Independently assembled blurred density.
    fractions : Float64[NDArray, " 3"]
        Sequential native-axis captured fractions.
    """
    edges: Tuple[Float64[NDArray, " Np1"], ...] = (
        np.asarray(calibration.u_bin_edges),
        np.asarray(calibration.v_bin_edges),
        np.asarray(calibration.energy_bin_edges_ev),
    )
    fwhm: Tuple[float, float, float] = (
        float(calibration.psf_fwhm_u),
        float(calibration.psf_fwhm_v),
        float(calibration.psf_fwhm_energy_ev),
    )
    matrices: Tuple[Float64[NDArray, "N N"], ...] = tuple(
        _reference_finite_volume_matrix(axis_edges, width * _FWHM_TO_SIGMA)
        for axis_edges, width in zip(edges, fwhm, strict=True)
    )
    volumes: Float64[NDArray, "U V E"] = (
        np.diff(edges[0])[:, None, None]
        * np.diff(edges[1])[None, :, None]
        * np.diff(edges[2])[None, None, :]
    )

    def flux(candidate: Float64[NDArray, "... U V E"]) -> float:
        returned: float = float(np.sum(candidate * volumes))
        return returned

    initial_flux: float = flux(density)
    after_u: Float64[NDArray, "... U V E"] = np.einsum(
        "ij,...jve->...ive", matrices[0], density
    )
    flux_u: float = flux(after_u)
    after_v: Float64[NDArray, "... U V E"] = np.einsum(
        "ij,...uje->...uie", matrices[1], after_u
    )
    flux_v: float = flux(after_v)
    blurred: Float64[NDArray, "... U V E"] = np.einsum(
        "ij,...uvj->...uvi", matrices[2], after_v
    )
    flux_energy: float = flux(blurred)
    fractions: Float64[NDArray, " 3"] = np.asarray(
        [flux_u / initial_flux, flux_v / flux_u, flux_energy / flux_v]
    )
    returned: Tuple[Float64[NDArray, "... U V E"], Float64[NDArray, " 3"]] = (
        blurred,
        fractions,
    )
    return returned


def _reference_integrated_bernstein(
    x: Float64[NDArray, "..."], degree: int
) -> Float64[NDArray, "... q"]:
    """PRIVATE: Integrate Bernstein basis functions independently.

    Parameters
    ----------
    x : Float64[NDArray, "..."]
        Normalized domain coordinates.
    degree : int
        Bernstein derivative degree.

    Returns
    -------
    values : Float64[NDArray, "... q"]
        Integrated basis values for every slope coordinate.
    """
    elevated_degree: int = degree + 1
    elevated: List[Float64[NDArray, "..."]] = [
        float(math.comb(elevated_degree, index))
        * x**index
        * (1.0 - x) ** (elevated_degree - index)
        for index in range(elevated_degree + 1)
    ]
    values: Float64[NDArray, "... q"] = np.stack(
        [
            np.sum(np.stack(elevated[index + 1 :]), axis=0) / elevated_degree
            for index in range(degree + 1)
        ],
        axis=-1,
    )
    return values


def _wrapped_cauchy_fourier_bin_masses(
    edges: Float64[NDArray, " Np1"],
    center: float,
    gamma_frac: float,
    *,
    n_harmonics: int = 256,
) -> Float64[NDArray, " N"]:
    """PRIVATE: Integrate wrapped-Cauchy bins by a Fourier reference.

    Parameters
    ----------
    edges : Float64[NDArray, " Np1"]
        Fractional edges spanning one period.
    center : float
        Folded fractional distribution centre.
    gamma_frac : float
        Positive HWHM divided by the reciprocal period.
    n_harmonics : int, optional
        Number of retained positive Fourier harmonics. Default is 256.

    Returns
    -------
    masses : Float64[NDArray, " N"]
        Independently integrated Fourier-series bin masses.

    Notes
    -----
    The coefficient of harmonic ``m`` is ``exp(-2*pi*gamma_frac*m)``.
    Integrating each cosine analytically avoids sharing the production CDF.
    """
    harmonics: Float64[NDArray, " M"] = np.arange(
        1, n_harmonics + 1, dtype=np.float64
    )
    coefficients: Float64[NDArray, " M"] = np.exp(
        -2.0 * np.pi * gamma_frac * harmonics
    ) / (np.pi * harmonics)
    right_phases: Float64[NDArray, "M N"] = (
        2.0 * np.pi * (harmonics[:, None] * (edges[None, 1:] - center))
    )
    left_phases: Float64[NDArray, "M N"] = (
        2.0 * np.pi * (harmonics[:, None] * (edges[None, :-1] - center))
    )
    masses: Float64[NDArray, " N"] = np.diff(edges) + np.sum(
        coefficients[:, None] * (np.sin(right_phases) - np.sin(left_phases)),
        axis=0,
    )
    return masses


def _surface_kz_fixture(
    *,
    oblique: bool,
    doubled_stacking: bool = False,
) -> Tuple[CrystalGeometry, SurfaceCell]:
    """PRIVATE: Build a cubic or oblique unit-advance surface fixture.

    Parameters
    ----------
    oblique : bool
        Whether the stacking vector has lateral components and length two.
    doubled_stacking : bool, optional
        Whether to plant stale numerical ``g=2`` stacking data while keeping
        the exact unit-advance metadata. Default is ``False``.

    Returns
    -------
    fixture : Tuple[CrystalGeometry, SurfaceCell]
        Bulk geometry and its nominal surface-frame carrier.

    Notes
    -----
    The oblique direct rows are ``(1,0,0)``, ``(0,1,0)``, and
    ``(0.4,0.2,2)``. Identity coefficients and rotation make every expected
    row independently explicit.
    """
    lattice: Float64[Array, "..."] = (
        jnp.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.4, 0.2, 2.0],
            ]
        )
        if oblique
        else jnp.eye(3, dtype=jnp.float64)
    )
    geometry: CrystalGeometry = make_crystal_geometry(
        lattice,
        jnp.zeros((1, 3), dtype=jnp.float64),
        ("X",),
    )
    stacking_scale: float = 2.0 if doubled_stacking else 1.0
    nominal_spacing: float = 2.0 if oblique else 1.0
    cell: SurfaceCell = make_surface_cell(
        in_plane_vectors=lattice[:2],
        stacking_vector=stacking_scale * lattice[2],
        rotation=jnp.eye(3, dtype=jnp.float64),
        interlayer_spacing_ang=stacking_scale * nominal_spacing,
        miller=(0, 0, 1),
        in_plane_coeffs=((1, 0, 0), (0, 1, 0)),
        stacking_coeffs=(0, 0, 1),
    )
    fixture: Tuple[CrystalGeometry, SurfaceCell] = (geometry, cell)
    return fixture


def _detector_chain_fixture() -> Tuple[
    ArpesCube,
    ExperimentGeometry,
    DetectorCalibration,
    DetectorEffects,
]:
    """PRIVATE: Build one compact public detector-chain fixture.

    Returns
    -------
    fixture : Tuple
        Source cube, geometry, explicit target calibration, and effects state.
    """
    kx: Float64[Array, "3"] = jnp.array([-0.5, 0.0, 0.5])
    ky: Float64[Array, "3"] = jnp.array([-0.45, 0.05, 0.55])
    energy: Float64[Array, "3"] = jnp.array([-0.4, 0.0, 0.4])
    intensity: Float64[Array, "..."] = (
        2.0
        + 0.35 * kx[:, None, None]
        + 0.22 * ky[None, :, None]
        + 0.17 * energy[None, None, :]
    )
    source: ArpesCube = make_arpes_cube(intensity, kx, ky, energy)
    geometry: ExperimentGeometry = make_experiment_geometry(
        photon_energy_ev=50.0,
        polarization=jnp.array([1.0 + 0.0j, 0.0j, 0.0j]),
        work_function_ev=4.0,
    )
    calibration: DetectorCalibration = make_detector_calibration(
        u_bin_edges=jnp.array([-0.055, -0.012, 0.047]),
        v_bin_edges=jnp.array([-0.048, 0.009, 0.052]),
        energy_bin_edges_ev=jnp.array([-0.24, -0.03, 0.21]),
        psf_fwhm_u=0.01,
        psf_fwhm_v=0.012,
        psf_fwhm_energy_ev=0.02,
        transmission_reference_domain_ev=jnp.array([40.0, 55.0]),
    )
    effects: DetectorEffects = make_detector_effects(
        domain_logits=jnp.array([0.0]),
        domain_euler_angles_rad=jnp.zeros((1, 3)),
        transmission_raw_slopes=jnp.array([-0.4, 0.2]),
        background_coefficients=jnp.array([-2.0]),
        sensitivity_coefficients=jnp.array([]),
        exposure=2.5,
        background_mode="flat",
        sensitivity_mode="constant",
        domain_frame_ids=(_FRAME_ID,),
    )
    fixture: Tuple[
        ArpesCube,
        ExperimentGeometry,
        DetectorCalibration,
        DetectorEffects,
    ] = (source, geometry, calibration, effects)
    return fixture
