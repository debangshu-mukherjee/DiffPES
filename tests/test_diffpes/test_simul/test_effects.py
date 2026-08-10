"""Verify resolution, transmission, detector effects, and count sampling.

The tests pin the bounded WP8.8 deterministic and stochastic contracts. They
also cover the implemented expected-rate and event-probability derivatives.
"""

import math

import chex
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from beartype.typing import Callable, Dict, Tuple
from hypothesis import given, settings
from hypothesis import strategies as st
from jaxtyping import Float64
from scipy import ndimage, special

from diffpes.simul.effects import (
    apply_detector_effects,
    apply_post_count_response,
    apply_resolution,
    apply_transmission,
    background_density,
    convolve_energy,
    convolve_kpath,
    convolve_momentum_map,
    detector_bin_volumes,
    expected_counts,
    fixed_total_probabilities,
    gaussian_kernel_1d,
    map_source_to_detector,
    sample_fixed_total_counts,
    sample_poisson_counts,
    sensitivity_field,
    transmission_shape,
)
from diffpes.types import (
    K_PREFACTOR_INV_ANG_SQRT_EV,
    ArpesCube,
    DetectorCalibration,
    DetectorEffects,
    DetectorRaster,
    ExperimentGeometry,
    constant_energy_map,
    fermi_surface_map,
    make_arpes_cube,
    make_detector_calibration,
    make_detector_effects,
    make_experiment_geometry,
)
from tests._assertions import assert_rejects
from tests._gradients import gradient_gate

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
    v_edges: jax.Array = (
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


def _inverse_softplus(value: float) -> jax.Array:
    """PRIVATE: Return the raw coordinate for one positive amplitude.

    The transform creates exact physical amplitudes for analytic fixtures.

    Parameters
    ----------
    value : float
        Positive physical amplitude.

    Returns
    -------
    raw_value : jax.Array
        Unconstrained softplus coordinate.
    """
    raw_value: jax.Array = jnp.log(jnp.expm1(value))
    return raw_value


def _d8_fixture() -> Tuple[
    DetectorCalibration,
    jax.Array,
    jax.Array,
    Tuple[jax.Array, jax.Array, jax.Array, jax.Array],
]:
    """PRIVATE: Build the shared smooth-effects D8 fixture.

    The fixture uses asymmetric values to expose every implemented leaf.

    Returns
    -------
    fixture : Tuple
        Calibration, density, loss weights, and continuous effects leaves.
    """
    calibration: DetectorCalibration = _calibration(slit=False)
    density: jax.Array = jnp.linspace(0.2, 1.4, 12).reshape((1, 2, 2, 3))
    weights: jax.Array = jnp.array(
        [
            [
                [[0.7, -0.2, 0.4], [0.1, 0.9, -0.5]],
                [[-0.3, 0.6, 1.1], [0.8, -0.7, 0.2]],
            ]
        ]
    )
    theta: Tuple[jax.Array, jax.Array, jax.Array, jax.Array] = (
        jnp.array([-0.2, 0.08, -0.05, 0.12, 0.04, -0.07, 0.03]),
        jnp.array([0.11, -0.06, 0.08, 0.03, -0.09, 0.05]),
        jnp.array(2.3),
        jnp.array([0.35, 0.7, 0.25]),
    )
    fixture: Tuple[
        DetectorCalibration,
        jax.Array,
        jax.Array,
        Tuple[jax.Array, jax.Array, jax.Array, jax.Array],
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
    displacement: Float64[np.ndarray, "..."], sigma: float
) -> Float64[np.ndarray, "..."]:
    """PRIVATE: Evaluate the independent Gaussian second antiderivative.

    Parameters
    ----------
    displacement : Float64[np.ndarray, "..."]
        Edge-to-edge displacements.
    sigma : float
        Positive Gaussian standard deviation.

    Returns
    -------
    values : Float64[np.ndarray, "..."]
        ``z * Phi(z/sigma) + sigma * phi(z/sigma)``.
    """
    scaled: Float64[np.ndarray, "..."] = displacement / sigma
    values: Float64[np.ndarray, "..."] = displacement * special.ndtr(
        scaled
    ) + sigma * np.exp(-0.5 * scaled**2) / np.sqrt(2.0 * np.pi)
    return values


def _reference_finite_volume_matrix(
    edges: Float64[np.ndarray, " Np1"], sigma: float
) -> Float64[np.ndarray, "N N"]:
    """PRIVATE: Build an independent analytic finite-volume Gaussian matrix.

    Parameters
    ----------
    edges : Float64[np.ndarray, " Np1"]
        Explicit common source and target cell edges.
    sigma : float
        Positive Gaussian standard deviation.

    Returns
    -------
    matrix : Float64[np.ndarray, "N N"]
        Output-density by input-density matrix without row normalization.
    """
    left: Float64[np.ndarray, " N"] = edges[:-1]
    right: Float64[np.ndarray, " N"] = edges[1:]
    integrated: Float64[np.ndarray, "N N"] = (
        _normal_second_antiderivative(right[:, None] - left[None, :], sigma)
        - _normal_second_antiderivative(left[:, None] - left[None, :], sigma)
        - _normal_second_antiderivative(right[:, None] - right[None, :], sigma)
        + _normal_second_antiderivative(left[:, None] - right[None, :], sigma)
    )
    matrix: Float64[np.ndarray, "N N"] = (
        np.maximum(integrated, 0.0) / np.diff(edges)[:, None]
    )
    return matrix


def _reference_resolution(
    density: Float64[np.ndarray, "... U V E"],
    calibration: DetectorCalibration,
) -> Tuple[Float64[np.ndarray, "... U V E"], Float64[np.ndarray, " 3"]]:
    """PRIVATE: Apply independent separable finite-volume resolution.

    Parameters
    ----------
    density : Float64[np.ndarray, "... U V E"]
        Input native-bin density.
    calibration : DetectorCalibration
        Native edges and FWHM widths.

    Returns
    -------
    blurred : Float64[np.ndarray, "... U V E"]
        Independently assembled blurred density.
    fractions : Float64[np.ndarray, " 3"]
        Sequential native-axis captured fractions.
    """
    edges: Tuple[Float64[np.ndarray, " Np1"], ...] = (
        np.asarray(calibration.u_bin_edges),
        np.asarray(calibration.v_bin_edges),
        np.asarray(calibration.energy_bin_edges_ev),
    )
    fwhm: Tuple[float, float, float] = (
        float(calibration.psf_fwhm_u),
        float(calibration.psf_fwhm_v),
        float(calibration.psf_fwhm_energy_ev),
    )
    matrices: Tuple[Float64[np.ndarray, "N N"], ...] = tuple(
        _reference_finite_volume_matrix(axis_edges, width * _FWHM_TO_SIGMA)
        for axis_edges, width in zip(edges, fwhm, strict=True)
    )
    volumes: Float64[np.ndarray, "U V E"] = (
        np.diff(edges[0])[:, None, None]
        * np.diff(edges[1])[None, :, None]
        * np.diff(edges[2])[None, None, :]
    )

    def flux(candidate: Float64[np.ndarray, "... U V E"]) -> float:
        return float(np.sum(candidate * volumes))

    initial_flux: float = flux(density)
    after_u: Float64[np.ndarray, "... U V E"] = np.einsum(
        "ij,...jve->...ive", matrices[0], density
    )
    flux_u: float = flux(after_u)
    after_v: Float64[np.ndarray, "... U V E"] = np.einsum(
        "ij,...uje->...uie", matrices[1], after_u
    )
    flux_v: float = flux(after_v)
    blurred: Float64[np.ndarray, "... U V E"] = np.einsum(
        "ij,...uvj->...uvi", matrices[2], after_v
    )
    flux_energy: float = flux(blurred)
    fractions: Float64[np.ndarray, " 3"] = np.asarray(
        [flux_u / initial_flux, flux_v / flux_u, flux_energy / flux_v]
    )
    return blurred, fractions


def _reference_integrated_bernstein(
    x: Float64[np.ndarray, "..."], degree: int
) -> Float64[np.ndarray, "... q"]:
    """PRIVATE: Integrate Bernstein basis functions independently.

    Parameters
    ----------
    x : Float64[np.ndarray, "..."]
        Normalized domain coordinates.
    degree : int
        Bernstein derivative degree.

    Returns
    -------
    values : Float64[np.ndarray, "... q"]
        Integrated basis values for every slope coordinate.
    """
    elevated_degree: int = degree + 1
    elevated: list[Float64[np.ndarray, "..."]] = [
        float(math.comb(elevated_degree, index))
        * x**index
        * (1.0 - x) ** (elevated_degree - index)
        for index in range(elevated_degree + 1)
    ]
    values: Float64[np.ndarray, "... q"] = np.stack(
        [
            np.sum(np.stack(elevated[index + 1 :]), axis=0) / elevated_degree
            for index in range(degree + 1)
        ],
        axis=-1,
    )
    return values


class TestGaussianKernel1D:
    """Verify :func:`diffpes.simul.gaussian_kernel_1d`.

    The class owns static sampled Gaussian support and normalization.
    """

    def test_normalizes_default_support_and_rejects_false_accuracy(
        self,
    ) -> None:
        """Normalize the 97-tap kernel and reject the frozen 65-tap claim.

        The case pins the registered support, symmetry, and unit mass.

        Notes
        -----
        The test compares the sampled kernel directly before exercising the
        support rejection.
        """
        kernel: jax.Array = gaussian_kernel_1d(6.0)

        chex.assert_shape(kernel, (97,))
        np.testing.assert_allclose(np.sum(kernel), 1.0, rtol=0.0, atol=3.0e-16)
        np.testing.assert_array_equal(kernel, kernel[::-1])
        assert_rejects(
            gaussian_kernel_1d,
            jnp.array(6.0),
            half_width=32,
            match="registered envelope",
        )

    def test_rejects_singular_width_eager_and_jit(self) -> None:
        """Reject zero and negative widths in both execution modes.

        The case fixes the lower boundary of the sampled-width envelope.

        Notes
        -----
        The shared rejection helper evaluates both eager and compiled calls.
        """
        assert_rejects(
            gaussian_kernel_1d,
            jnp.array(0.0),
            match="registered envelope",
        )
        assert_rejects(
            gaussian_kernel_1d,
            jnp.array(-0.2),
            match="registered envelope",
        )


class TestConvolveEnergy:
    """Verify :func:`diffpes.simul.convolve_energy`.

    The class owns G1 sampled energy parity and validation.
    """

    @pytest.mark.parametrize("sigma_over_dx", [0.5, 1.0, 2.0, 6.0])
    def test_g1_matches_scipy_sampled_stencil(
        self, sigma_over_dx: float
    ) -> None:
        """Match SciPy constant-boundary filtering at all registered widths.

        The case covers every preregistered ratio of width to spacing.

        Notes
        -----
        The test applies the same static radius and zero-boundary convention to
        independent SciPy output.
        """
        rng: np.random.Generator = np.random.default_rng(8101)
        samples: Float64[np.ndarray, "A B E"] = rng.normal(size=(2, 3, 31))
        spacing: float = 0.04
        energy: jax.Array = jnp.arange(samples.shape[-1]) * spacing - 0.6
        actual: jax.Array = convolve_energy(
            jnp.asarray(samples),
            energy,
            sigma_over_dx * spacing,
        )
        desired: Float64[np.ndarray, "A B E"] = ndimage.gaussian_filter1d(
            samples,
            sigma=sigma_over_dx,
            axis=-1,
            mode="constant",
            radius=_SAMPLED_HALF_WIDTH,
        )

        np.testing.assert_allclose(actual, desired, rtol=1.0e-10, atol=1.0e-13)

    def test_rejects_nonuniform_axis_eager_and_jit(self) -> None:
        """Reject a finite but nonuniform energy grid in both modes.

        The case prevents sampled convolution on an invalid physical axis.

        Notes
        -----
        The shared rejection helper traces the same nonuniform coordinates
        through eager and compiled calls.
        """
        assert_rejects(
            convolve_energy,
            jnp.ones((2, 4)),
            jnp.array([0.0, 0.1, 0.21, 0.3]),
            jnp.array(0.05),
            match="finite, increasing, and uniform",
        )


class TestConvolveMomentumMap:
    """Verify :func:`diffpes.simul.convolve_momentum_map`.

    The class owns G2 Cartesian-map SciPy parity with explicit physical axes.
    """

    def test_g2_matches_separable_scipy_filter(self) -> None:
        """Match SciPy on unequal uniform Cartesian momentum spacings.

        The case pins separable physical-axis scaling on a nonsquare map.

        Notes
        -----
        The test builds independent SciPy output with one sigma per momentum
        spacing and the registered static radius.
        """
        rng: np.random.Generator = np.random.default_rng(8102)
        samples: Float64[np.ndarray, "Kx Ky E"] = rng.normal(size=(9, 7, 4))
        kx: jax.Array = jnp.linspace(-0.24, 0.24, 9)
        ky: jax.Array = jnp.linspace(-0.15, 0.15, 7)
        sigma: float = 0.06
        actual: jax.Array = convolve_momentum_map(
            jnp.asarray(samples), kx, ky, sigma
        )
        desired: Float64[np.ndarray, "Kx Ky E"] = ndimage.gaussian_filter(
            samples,
            sigma=(sigma / 0.06, sigma / 0.05),
            axes=(0, 1),
            mode="constant",
            radius=(_SAMPLED_HALF_WIDTH, _SAMPLED_HALF_WIDTH),
        )

        np.testing.assert_allclose(actual, desired, rtol=1.0e-10, atol=1.0e-13)

    def test_rejects_fractional_or_nonuniform_coordinate_surrogates(
        self,
    ) -> None:
        """Require explicit uniformly calibrated physical momentum axes.

        The case rejects a fractional-coordinate surrogate with uneven steps.

        Notes
        -----
        The shared helper evaluates the coordinate validation in eager and
        compiled execution.
        """
        assert_rejects(
            convolve_momentum_map,
            jnp.ones((4, 3, 2)),
            jnp.array([-0.2, -0.1, 0.03, 0.2]),
            jnp.array([-0.1, 0.0, 0.1]),
            jnp.array(0.04),
            match="finite, increasing, and uniform",
        )


class TestConvolveKPath:
    """Verify :func:`diffpes.simul.convolve_kpath`.

    The class owns G3 numerical parity and G3C physical path semantics.
    """

    def test_g3_shared_sampled_stencil_matches_scipy(self) -> None:
        """Match SciPy through the shared uniform-axis sampled implementation.

        The case pins the private G3 sampled-parity path without changing G3C.

        Notes
        -----
        The test transposes the path axis through the public sampled energy
        helper and compares independent SciPy output.
        """
        rng: np.random.Generator = np.random.default_rng(8103)
        samples: Float64[np.ndarray, "K E"] = rng.normal(size=(25, 3))
        centres: jax.Array = jnp.linspace(-0.6, 0.6, 25)
        sigma: float = 0.075
        sampled: jax.Array = convolve_energy(
            jnp.asarray(samples).T,
            centres,
            sigma,
        ).T
        desired: Float64[np.ndarray, "K E"] = ndimage.gaussian_filter1d(
            samples,
            sigma=sigma / 0.05,
            axis=0,
            mode="constant",
            radius=_SAMPLED_HALF_WIDTH,
        )

        np.testing.assert_allclose(
            sampled, desired, rtol=1.0e-10, atol=1.0e-13
        )

    def test_g3c_matches_nonuniform_analytic_finite_volume(self) -> None:
        """Match analytic nonuniform cells without row normalization.

        The case verifies physical density transport and captured boundary flux.

        Notes
        -----
        The test constructs cell edges and an independent analytic Gaussian
        matrix before comparing density and flux diagnostics.
        """
        centres_np: Float64[np.ndarray, " K"] = np.array(
            [-0.8, -0.35, -0.1, 0.4, 1.1]
        )
        density_np: Float64[np.ndarray, "K E"] = np.array(
            [
                [0.2, 0.7],
                [1.1, 0.4],
                [0.8, 1.5],
                [1.7, 0.3],
                [0.4, 0.9],
            ]
        )
        sigma: float = 0.22
        interior: Float64[np.ndarray, " Km1"] = 0.5 * (
            centres_np[:-1] + centres_np[1:]
        )
        edges: Float64[np.ndarray, " Kp1"] = np.concatenate(
            (
                [centres_np[0] - 0.5 * (centres_np[1] - centres_np[0])],
                interior,
                [centres_np[-1] + 0.5 * (centres_np[-1] - centres_np[-2])],
            )
        )
        matrix: Float64[np.ndarray, "K K"] = _reference_finite_volume_matrix(
            edges, sigma
        )
        desired: Float64[np.ndarray, "K E"] = matrix @ density_np
        widths: Float64[np.ndarray, " K"] = np.diff(edges)
        desired_fraction: float = float(
            np.sum(desired * widths[:, None])
            / np.sum(density_np * widths[:, None])
        )

        actual: jax.Array
        fraction: jax.Array
        valid: jax.Array
        actual, fraction, valid = convolve_kpath(
            jnp.asarray(density_np), jnp.asarray(centres_np), sigma
        )

        np.testing.assert_allclose(
            actual, desired, rtol=_FINITE_VOLUME_RTOL, atol=2.0e-15
        )
        np.testing.assert_allclose(
            fraction, desired_fraction, rtol=_FINITE_VOLUME_RTOL, atol=0.0
        )
        assert bool(valid)
        assert 0.0 <= float(fraction) <= 1.0

    def test_g3c_two_center_counterexample_rejects_sampled_flux_creation(
        self,
    ) -> None:
        """Pin the coarse two-centre analytic result and sampled-rule failure.

        The case exposes flux creation from the retired sampled-cell rule.

        Notes
        -----
        The test compares the finite-volume captured fraction with the analytic
        value produced by the planted sampled counterexample.
        """
        centres: jax.Array = jnp.array([0.0, 1.0])
        density: jax.Array = jnp.ones((2, 1))
        sigma: float = 0.01
        finite_volume_fraction: jax.Array
        valid: jax.Array
        _, finite_volume_fraction, valid = convolve_kpath(
            density, centres, sigma
        )
        former_sampled_fraction: float = 0.5 / (sigma * np.sqrt(2.0 * np.pi))

        assert bool(valid)
        np.testing.assert_allclose(
            finite_volume_fraction,
            0.9960105771959856,
            rtol=_FINITE_VOLUME_RTOL,
            atol=0.0,
        )
        assert former_sampled_fraction == pytest.approx(19.947114020071634)
        assert former_sampled_fraction > 1.0

    def test_g3c_domain_enlargement_recovers_flux_and_zero_is_invalid(
        self,
    ) -> None:
        """Recover escaped mass with source-covered padding and flag zero input.

        The case distinguishes physical boundary loss from an invalid zero rate.

        Notes
        -----
        The test enlarges the source domain and compares captured fractions
        before checking exact zero-output diagnostics.
        """
        compact_centres: jax.Array = jnp.arange(-1.0, 2.0)
        compact_density: jax.Array = jnp.array([[0.0], [1.0], [0.0]])
        extended_centres: jax.Array = jnp.arange(-3.0, 4.0)
        extended_density: jax.Array = jnp.array(
            [[0.0], [0.0], [0.0], [1.0], [0.0], [0.0], [0.0]]
        )
        compact_fraction: jax.Array
        _, compact_fraction, _ = convolve_kpath(
            compact_density, compact_centres, 0.7
        )
        extended_fraction: jax.Array
        _, extended_fraction, _ = convolve_kpath(
            extended_density, extended_centres, 0.7
        )
        zero: jax.Array
        zero_fraction: jax.Array
        zero_valid: jax.Array
        zero, zero_fraction, zero_valid = convolve_kpath(
            jnp.zeros_like(compact_density), compact_centres, 0.7
        )

        assert float(extended_fraction) > float(compact_fraction)
        assert float(extended_fraction) < 1.0
        chex.assert_trees_all_equal(zero, jnp.zeros_like(zero))
        assert float(zero_fraction) == 0.0
        assert not bool(zero_valid)

    @settings(max_examples=12, deadline=None, derandomize=True)
    @given(
        narrower=st.floats(
            min_value=0.08,
            max_value=0.35,
            allow_nan=False,
            allow_infinity=False,
        ),
        increment=st.floats(
            min_value=0.03,
            max_value=0.25,
            allow_nan=False,
            allow_infinity=False,
        ),
    )
    def test_broader_width_monotonically_increases_variance(
        self, narrower: float, increment: float
    ) -> None:
        """Verify finite-volume variance increases over a bounded width sweep.

        The property also keeps both captured fractions inside the declared
        loss interval and requires the broader kernel to capture no more mass.

        Notes
        -----
        Compare two positive widths on one deterministic impulse density.
        """
        centres: jax.Array = jnp.linspace(-4.0, 4.0, 17)
        density: jax.Array = jnp.zeros((17, 1)).at[8, 0].set(1.0)
        broader: float = narrower + increment
        narrow_density: jax.Array
        narrow_fraction: jax.Array
        broad_density: jax.Array
        broad_fraction: jax.Array
        narrow_density, narrow_fraction, _ = convolve_kpath(
            density, centres, narrower
        )
        broad_density, broad_fraction, _ = convolve_kpath(
            density, centres, broader
        )
        widths: jax.Array = jnp.full((17,), centres[1] - centres[0])

        def variance(candidate: jax.Array) -> jax.Array:
            """Return the captured-density variance."""
            mass: jax.Array = candidate[:, 0] * widths
            probability: jax.Array = mass / jnp.sum(mass)
            mean: jax.Array = jnp.sum(probability * centres)
            return jnp.sum(probability * (centres - mean) ** 2)

        assert float(variance(broad_density)) > float(variance(narrow_density))
        fraction_tolerance: float = 5.0e-15
        assert (
            -fraction_tolerance
            <= float(narrow_fraction)
            <= 1.0 + fraction_tolerance
        )
        assert (
            -fraction_tolerance
            <= float(broad_fraction)
            <= 1.0 + fraction_tolerance
        )
        assert (
            float(broad_fraction)
            <= float(narrow_fraction) + fraction_tolerance
        )


class TestApplyResolution:
    """Verify :func:`diffpes.simul.apply_resolution`.

    The class owns G1C/G2C finite-volume resolution and D1 gradients.
    """

    @pytest.mark.parametrize("profile", ["delta", "constant", "translated"])
    def test_g1c_g2c_matches_independent_analytic_reference(
        self, profile: str
    ) -> None:
        """Match separable analytic truth for three edge-sensitive fixtures.

        The case covers delta, constant, and translated native-bin densities.

        Notes
        -----
        The test assembles independent finite-volume matrices and compares all
        blurred values and sequential captured fractions.
        """
        calibration: DetectorCalibration = _resolution_calibration()
        density_np: Float64[np.ndarray, "C U V E"] = np.zeros((1, 3, 2, 4))
        if profile == "delta":
            density_np[0, 0, 0, 0] = 2.3
        elif profile == "constant":
            density_np[...] = 0.7
        else:
            density_np[0, 1, 1, 2] = 1.4
            density_np[0, 2, 0, 1] = 0.6
        desired: Float64[np.ndarray, "C U V E"]
        desired_fractions: Float64[np.ndarray, " 3"]
        desired, desired_fractions = _reference_resolution(
            density_np, calibration
        )

        actual: jax.Array
        fractions: jax.Array
        valid: jax.Array
        actual, fractions, valid = apply_resolution(
            jnp.asarray(density_np), calibration
        )

        np.testing.assert_allclose(
            actual, desired, rtol=_FINITE_VOLUME_RTOL, atol=2.0e-15
        )
        np.testing.assert_allclose(
            fractions,
            desired_fractions,
            rtol=_FINITE_VOLUME_RTOL,
            atol=2.0e-15,
        )
        assert bool(valid)
        assert bool(jnp.all((fractions >= 0.0) & (fractions <= 1.0)))

    def test_g2c_anisotropic_native_widths_do_not_become_stationary_k(
        self,
    ) -> None:
        """Expose swapped native angular widths on unequal detector bins.

        The case prevents replacing native-coordinate widths with stationary k.

        Notes
        -----
        The test swaps the two angular FWHMs and requires a measurable change
        in the resolved detector density.
        """
        density: jax.Array = jnp.zeros((1, 3, 2, 4)).at[0, 0, 1, 2].set(1.0)
        calibrated: DetectorCalibration = _resolution_calibration(
            widths=(0.31, 0.08, 0.2)
        )
        swapped: DetectorCalibration = _resolution_calibration(
            widths=(0.08, 0.31, 0.2)
        )
        desired: jax.Array
        desired, _, _ = apply_resolution(density, calibrated)
        planted_stationary_k: jax.Array
        planted_stationary_k, _, _ = apply_resolution(density, swapped)

        assert float(jnp.max(jnp.abs(desired - planted_stationary_k))) > 1.0e-3

    def test_g2c_nonlinear_kinematics_rejects_stationary_k_width(self) -> None:
        """Reject a fixed momentum PSF across energy-dependent angle maps.

        An independent native-angle calculation supplies the accepted result.
        The planted alternative conserves each bin's mass while transforming
        to ``k = p(E) sin(u)``, applies one fixed momentum width, and transforms
        back. Its energy-dependent angular profiles must fail the native truth.

        Notes
        -----
        Evaluate all energies and compare the transformed profiles directly.
        """
        u_edges: Float64[np.ndarray, " U1"] = np.array(
            [-0.30, -0.20, -0.11, -0.03, 0.05, 0.14, 0.24, 0.36]
        )
        calibration: DetectorCalibration = make_detector_calibration(
            u_bin_edges=jnp.asarray(u_edges),
            v_bin_edges=jnp.array([-0.2, 0.2]),
            energy_bin_edges_ev=jnp.array([-20.0, -10.0, 10.0, 20.0]),
            psf_fwhm_u=0.12,
            psf_fwhm_v=0.003,
            psf_fwhm_energy_ev=0.03,
            transmission_reference_domain_ev=jnp.array([20.0, 70.0]),
        )
        profile: Float64[np.ndarray, " U"] = np.array(
            [0.1, 0.4, 1.7, 0.8, 0.25, 0.05, 0.02]
        )
        density_np: Float64[np.ndarray, "C U V E"] = np.broadcast_to(
            profile[None, :, None, None], (1, 7, 1, 3)
        ).copy()
        native_truth: Float64[np.ndarray, "C U V E"]
        native_fractions: Float64[np.ndarray, " 3"]
        native_truth, native_fractions = _reference_resolution(
            density_np, calibration
        )
        actual: jax.Array
        actual_fractions: jax.Array
        actual, actual_fractions, _ = apply_resolution(
            jnp.asarray(density_np), calibration
        )

        energy_centres: Float64[np.ndarray, " E"] = np.array(
            [-15.0, 0.0, 15.0]
        )
        kinetic_energy: Float64[np.ndarray, " E"] = 46.0 + energy_centres
        momenta: Float64[np.ndarray, " E"] = float(
            K_PREFACTOR_INV_ANG_SQRT_EV
        ) * np.sqrt(kinetic_energy)
        sigma_u: float = 0.12 * _FWHM_TO_SIGMA
        sigma_k: float = momenta[1] * sigma_u
        angular_widths: Float64[np.ndarray, " U"] = np.diff(u_edges)
        stationary_u: Float64[np.ndarray, "C U V E"] = np.empty_like(
            density_np
        )
        energy_index: int
        momentum: np.float64
        for energy_index, momentum in enumerate(momenta):
            k_edges: Float64[np.ndarray, " U1"] = momentum * np.sin(u_edges)
            k_widths: Float64[np.ndarray, " U"] = np.diff(k_edges)
            source_k: Float64[np.ndarray, " U"] = (
                density_np[0, :, 0, energy_index] * angular_widths / k_widths
            )
            stationary_k: Float64[np.ndarray, " U"] = (
                _reference_finite_volume_matrix(k_edges, sigma_k) @ source_k
            )
            stationary_u[0, :, 0, energy_index] = (
                stationary_k * k_widths / angular_widths
            )
        v_matrix: Float64[np.ndarray, "V V"] = _reference_finite_volume_matrix(
            np.asarray(calibration.v_bin_edges),
            float(calibration.psf_fwhm_v) * _FWHM_TO_SIGMA,
        )
        energy_matrix: Float64[np.ndarray, "E E"] = (
            _reference_finite_volume_matrix(
                np.asarray(calibration.energy_bin_edges_ev),
                float(calibration.psf_fwhm_energy_ev) * _FWHM_TO_SIGMA,
            )
        )
        stationary_u = np.einsum("ij,...uje->...uie", v_matrix, stationary_u)
        stationary_u = np.einsum(
            "ij,...uvj->...uvi", energy_matrix, stationary_u
        )

        np.testing.assert_allclose(
            actual,
            native_truth,
            rtol=_FINITE_VOLUME_RTOL,
            atol=2.0e-15,
        )
        np.testing.assert_allclose(
            actual_fractions,
            native_fractions,
            rtol=_FINITE_VOLUME_RTOL,
            atol=2.0e-15,
        )
        assert not np.allclose(
            stationary_u,
            native_truth,
            rtol=1.0e-10,
            atol=1.0e-12,
        )
        assert not np.allclose(
            stationary_u[0, :, 0, 0],
            stationary_u[0, :, 0, -1],
            rtol=1.0e-6,
            atol=1.0e-12,
        )

    def test_zero_density_has_exact_invalid_mask(self) -> None:
        """Return exact zeros and false validity for a zero detector raster.

        The case fixes zero-rate diagnostic semantics for native resolution.

        Notes
        -----
        The test compares every output with exact zeros and checks the false
        validity flag.
        """
        calibration: DetectorCalibration = _resolution_calibration()
        density: jax.Array = jnp.zeros((2, 3, 2, 4))
        blurred: jax.Array
        fractions: jax.Array
        valid: jax.Array
        blurred, fractions, valid = apply_resolution(density, calibration)

        chex.assert_trees_all_equal(blurred, jnp.zeros_like(density))
        chex.assert_trees_all_equal(fractions, jnp.zeros(3))
        assert not bool(valid)

    def test_d1_width_and_intensity_gradients_match_finite_differences(
        self,
    ) -> None:
        """Check fwd/rev gradients through all three FWHMs and density.

        The case covers every calibrated width and the full intensity tensor.

        Notes
        -----
        The shared gradient gate compares forward and reverse autodiff with its
        finite-difference ladder on a smooth fixture.
        """
        calibration: DetectorCalibration = _resolution_calibration()
        widths: jax.Array = jnp.array([0.23, 0.17, 0.29])
        density: jax.Array = jnp.linspace(0.2, 1.3, 24).reshape((1, 3, 2, 4))
        weights: jax.Array = jnp.linspace(-0.4, 0.8, 24).reshape(density.shape)

        def loss(theta: Tuple[jax.Array, jax.Array]) -> jax.Array:
            candidate_widths: jax.Array
            candidate_density: jax.Array
            candidate_widths, candidate_density = theta
            candidate: DetectorCalibration = eqx.tree_at(
                lambda item: (
                    item.psf_fwhm_u,
                    item.psf_fwhm_v,
                    item.psf_fwhm_energy_ev,
                ),
                calibration,
                (
                    candidate_widths[0],
                    candidate_widths[1],
                    candidate_widths[2],
                ),
            )
            blurred: jax.Array
            fractions: jax.Array
            blurred, fractions, _ = apply_resolution(
                candidate_density, candidate
            )
            return jnp.sum(blurred * weights) + jnp.dot(
                fractions, jnp.array([0.3, -0.2, 0.4])
            )

        gradient_gate(loss, (widths, density), regime="smooth")

    def test_d1_decreasing_widths_converge_one_sided_to_identity(self) -> None:
        """Verify value convergence as every positive native width decreases.

        The sequence stays above the registered positive-width floor and
        deliberately makes no derivative claim at the rejected zero limit.

        Notes
        -----
        Measure successive errors against the unchanged input density.
        """
        density: jax.Array = jnp.array(
            [
                [
                    [[0.2, 1.1, 0.4, 0.8], [1.3, 0.1, 0.7, 0.5]],
                    [[0.9, 0.3, 1.4, 0.2], [0.4, 1.2, 0.6, 1.0]],
                    [[1.5, 0.2, 0.8, 0.4], [0.3, 1.1, 0.5, 1.3]],
                ]
            ]
        )
        width_scales: Tuple[float, ...] = (0.18, 0.09, 0.045, 0.0225)
        errors: list[float] = []
        width: float
        for width in width_scales:
            calibration: DetectorCalibration = _resolution_calibration(
                widths=(width, width, width)
            )
            blurred: jax.Array = apply_resolution(density, calibration)[0]
            errors.append(float(jnp.linalg.norm(blurred - density)))

        assert all(
            later < earlier for earlier, later in zip(errors, errors[1:])
        )
        assert errors[-1] < 0.25 * errors[0]


class TestDisplayTopHatDerivatives:
    """Verify the deliberately nonsmooth display-window derivative contract."""

    def test_d6_top_hat_coordinates_are_documented_exact_zeros(self) -> None:
        """Assert zero centre and tolerance gradients away from membership seams.

        The case covers both constant-energy and Fermi-surface display helpers.

        Notes
        -----
        Differentiate interior membership regions and inspect their documentation.
        """
        energy: jax.Array = jnp.array([-1.0, -0.2, 0.4, 1.2])
        intensity: jax.Array = jnp.arange(16.0).reshape((2, 2, 4)) + 1.0
        cube: ArpesCube = make_arpes_cube(
            intensity,
            jnp.array([-0.3, 0.4]),
            jnp.array([-0.5, 0.2]),
            energy,
        )
        window_grad: jax.Array = jax.grad(
            lambda window: jnp.sum(
                constant_energy_map(cube, window[0], window[1])
            )
        )(jnp.array([-0.15, 0.1]))
        tolerance_grad: jax.Array = jax.grad(
            lambda tolerance: jnp.sum(fermi_surface_map(cube, tolerance))
        )(jnp.array(0.25))

        chex.assert_trees_all_equal(window_grad, jnp.zeros(2))
        chex.assert_trees_all_equal(tolerance_grad, jnp.array(0.0))
        assert "zero almost everywhere by design" in (
            constant_energy_map.__doc__ or ""
        )
        assert "documented zero derivative" in (
            fermi_surface_map.__doc__ or ""
        )


class TestTransmissionShape:
    """Verify :func:`diffpes.simul.transmission_shape`.

    The class owns G9 fixed-domain calibration and D7 shape derivatives.
    """

    @pytest.mark.parametrize("sign", [-1, 1])
    @pytest.mark.parametrize("n_slopes", [2, 3])
    def test_g9_positive_monotone_fixed_domain_mean(
        self, sign: int, n_slopes: int
    ) -> None:
        """Normalize the full domain and enforce the registered slope sign.

        The case covers both monotonic directions and supported shape degrees.

        Notes
        -----
        The test evaluates a dense quadrature grid and checks positivity,
        strict monotonicity, and unit domain mean.
        """
        calibration: DetectorCalibration = _resolution_calibration(sign=sign)
        raw: jax.Array = jnp.linspace(-0.4, 0.25, n_slopes)
        nodes128: Float64[np.ndarray, " Q"]
        weights128: Float64[np.ndarray, " Q"]
        nodes128, weights128 = np.polynomial.legendre.leggauss(128)
        energies: jax.Array = 30.0 + 18.0 * jnp.asarray(nodes128)
        transmission: jax.Array = transmission_shape(
            energies, raw, calibration
        )
        weighted_mean: jax.Array = 0.5 * jnp.sum(
            jnp.asarray(weights128) * transmission
        )
        differences: jax.Array = jnp.diff(transmission)

        assert bool(jnp.all(transmission > 0.0))
        assert bool(jnp.all(sign * differences > 0.0))
        np.testing.assert_allclose(
            weighted_mean, 1.0, rtol=1.0e-12, atol=1.0e-14
        )

    def test_g9_crop_and_padding_invariance_is_bitwise(self) -> None:
        """Keep retained transmission bins bitwise identical across windows.

        The case pins normalization to the fixed calibration domain.

        Notes
        -----
        The test evaluates one full query and requires its retained slice to
        equal an independently cropped query bitwise.
        """
        calibration: DetectorCalibration = _resolution_calibration()
        raw: jax.Array = jnp.array([-0.4, 0.2, -0.1])
        full_energy: jax.Array = jnp.linspace(12.0, 48.0, 13)
        full: jax.Array = transmission_shape(full_energy, raw, calibration)
        cropped: jax.Array = transmission_shape(
            full_energy[3:10], raw, calibration
        )

        np.testing.assert_array_equal(cropped, full[3:10])

    def test_g9_matches_independent_integrated_bernstein_reference(
        self,
    ) -> None:
        """Match an independent 128-node basis and normalization calculation.

        The case verifies the monotone shape parameterization numerically.

        Notes
        -----
        The test constructs integrated Bernstein values and Gauss-Legendre
        normalization with NumPy before comparing production output.
        """
        calibration: DetectorCalibration = _resolution_calibration(sign=-1)
        raw_np: Float64[np.ndarray, " q"] = np.array([-0.3, 0.15, 0.4])
        query_np: Float64[np.ndarray, " E"] = np.array(
            [12.0, 18.5, 31.0, 48.0]
        )
        normalized_query: Float64[np.ndarray, " E"] = (query_np - 12.0) / 36.0
        slopes: Float64[np.ndarray, " q"] = np.logaddexp(0.0, raw_np)
        log_query: Float64[np.ndarray, " E"] = -np.sum(
            _reference_integrated_bernstein(normalized_query, 2) * slopes,
            axis=-1,
        )
        nodes: Float64[np.ndarray, " Q"]
        weights: Float64[np.ndarray, " Q"]
        nodes, weights = np.polynomial.legendre.leggauss(128)
        normalized_nodes: Float64[np.ndarray, " Q"] = 0.5 * (nodes + 1.0)
        log_nodes: Float64[np.ndarray, " Q"] = -np.sum(
            _reference_integrated_bernstein(normalized_nodes, 2) * slopes,
            axis=-1,
        )
        denominator128: float = float(
            0.5 * np.sum(weights * np.exp(log_nodes))
        )
        desired: Float64[np.ndarray, " E"] = np.exp(log_query) / denominator128
        actual: jax.Array = transmission_shape(
            jnp.asarray(query_np), jnp.asarray(raw_np), calibration
        )

        np.testing.assert_allclose(actual, desired, rtol=1.0e-12, atol=1.0e-14)

    def test_g9_rejects_extrapolation_eager_and_jit(self) -> None:
        """Reject any query outside the fixed calibration domain in both modes.

        The case prevents silent transmission extrapolation beyond calibration.

        Notes
        -----
        The shared rejection helper submits one below-domain query to eager and
        compiled execution.
        """
        calibration: DetectorCalibration = _resolution_calibration()
        assert_rejects(
            transmission_shape,
            jnp.array([11.99, 20.0]),
            jnp.array([-0.2, 0.3]),
            calibration,
            match="inside the calibration domain",
        )

    @pytest.mark.parametrize("n_slopes", [2, 3])
    def test_d7_every_shape_coefficient_matches_fd_and_is_nonzero(
        self, n_slopes: int
    ) -> None:
        """Check each raw-slope derivative with the shared f64 FD ladder.

        The case verifies all supported transmission-shape coordinates.

        Notes
        -----
        The shared gradient gate compares every coordinate with finite
        differences and requires a nonzero smooth derivative.
        """
        calibration: DetectorCalibration = _resolution_calibration(sign=1)
        raw: jax.Array = jnp.linspace(-0.35, 0.28, n_slopes)
        energy: jax.Array = jnp.array([13.0, 19.0, 28.0, 39.0, 47.0])
        weights: jax.Array = jnp.array([0.8, -0.2, 0.5, -0.7, 1.1])

        def loss(candidate: jax.Array) -> jax.Array:
            return jnp.sum(
                transmission_shape(energy, candidate, calibration) * weights
            )

        gradient_gate(loss, raw, regime="smooth", elementwise=True)

    def test_d7_energy_gradients_match_fd(self) -> None:
        """Check transmission derivatives for every query energy.

        The case verifies the continuous kinetic-energy dependence directly.

        Notes
        -----
        The shared gradient gate compares every energy coordinate with its
        finite-difference estimate.
        """
        calibration: DetectorCalibration = _resolution_calibration(sign=-1)
        raw: jax.Array = jnp.array([-0.3, 0.15, 0.4])
        energy: jax.Array = jnp.array([13.0, 20.0, 31.0, 45.0])
        weights: jax.Array = jnp.array([0.7, -0.4, 1.1, 0.3])

        def loss(candidate: jax.Array) -> jax.Array:
            return jnp.sum(
                transmission_shape(candidate, raw, calibration) * weights
            )

        gradient_gate(loss, energy, regime="smooth", elementwise=True)


class TestApplyTransmission:
    """Verify :func:`diffpes.simul.apply_transmission`.

    The class owns multiplicative true-energy transmission semantics.
    """

    def test_multiplies_only_the_trailing_energy_axis(self) -> None:
        """Apply one fixed transmission curve over arbitrary leading axes.

        The case pins broadcasting to the trailing true-energy coordinate.

        Notes
        -----
        The test multiplies the input by the independently evaluated shape and
        requires exact tree equality.
        """
        calibration: DetectorCalibration = _resolution_calibration()
        energy: jax.Array = jnp.array([14.0, 25.0, 37.0, 46.0])
        raw: jax.Array = jnp.array([-0.4, 0.2])
        intensity: jax.Array = jnp.arange(24.0).reshape((2, 3, 4)) + 0.2
        shape: jax.Array = transmission_shape(energy, raw, calibration)
        actual: jax.Array = apply_transmission(
            intensity, energy, raw, calibration
        )

        chex.assert_trees_all_equal(actual, intensity * shape)


class TestResolutionTransmissionVariants(chex.TestCase):
    """Verify eager and JIT success paths for both canonical effects."""

    @chex.variants(with_jit=True, without_jit=True)
    def test_apply_resolution_success_path(self) -> None:
        """Return equal-shape finite results in eager and compiled execution.

        The case verifies the canonical resolution success path in both modes.

        Notes
        -----
        The Chex variant applies one fixture eagerly and through JIT before
        checking shape, finiteness, and validity.
        """
        operator: Callable[..., object] = self.variant(apply_resolution)
        calibration: DetectorCalibration = _resolution_calibration()
        density: jax.Array = jnp.linspace(0.2, 1.3, 24).reshape((1, 3, 2, 4))
        blurred: jax.Array
        fractions: jax.Array
        valid: jax.Array
        blurred, fractions, valid = operator(density, calibration)

        chex.assert_shape(blurred, density.shape)
        chex.assert_shape(fractions, (3,))
        assert bool(jnp.all(jnp.isfinite(blurred)))
        assert bool(valid)

    @chex.variants(with_jit=True, without_jit=True)
    def test_apply_transmission_success_path(self) -> None:
        """Apply finite transmission in eager and compiled execution.

        The case verifies shape preservation for the multiplicative stage.

        Notes
        -----
        The Chex variant applies one fixture eagerly and through JIT before
        checking shape and finiteness.
        """
        operator: Callable[..., jax.Array] = self.variant(apply_transmission)
        calibration: DetectorCalibration = _resolution_calibration()
        energy: jax.Array = jnp.array([14.0, 25.0, 37.0, 46.0])
        intensity: jax.Array = jnp.ones((2, 3, 4))
        actual: jax.Array = operator(
            intensity, energy, jnp.array([-0.4, 0.2]), calibration
        )

        chex.assert_shape(actual, intensity.shape)
        assert bool(jnp.all(jnp.isfinite(actual)))


class TestDetectorBinVolumes:
    """Verify :func:`diffpes.simul.detector_bin_volumes`.

    The class owns unequal-width and slit-volume behavior.
    """

    def test_preserves_every_unequal_native_width(self) -> None:
        """Preserve explicit unequal native-bin volumes.

        The case compares all products with independent NumPy edge differences.

        Notes
        -----
        The test constructs a two-dimensional detector map and compares every
        target bin at the deterministic G8 tolerance.
        """
        calibration: DetectorCalibration = _calibration(slit=False)
        actual: jax.Array = detector_bin_volumes(calibration)
        desired: Float64[np.ndarray, "U V E"] = (
            np.diff(np.array([-1.0, -0.25, 1.5]))[:, None, None]
            * np.diff(np.array([-0.4, 0.1, 0.8]))[None, :, None]
            * np.diff(np.array([-2.0, -0.75, 0.5, 2.5]))[None, None, :]
        )

        np.testing.assert_allclose(
            actual, desired, rtol=_DETERMINISTIC_RTOL, atol=0.0
        )


class TestBackgroundDensity:
    """Verify :func:`diffpes.simul.background_density`.

    The class owns smooth positivity and the weighted Shirley tail.
    """

    @pytest.mark.parametrize("slit", [False, True])
    def test_smooth_background_remains_nonnegative(self, slit: bool) -> None:
        """Keep smooth map and slit backgrounds nonnegative.

        The parameterized case exercises both active-axis coefficient lengths.

        Notes
        -----
        The test evaluates an asymmetric raw Legendre field and checks every
        physical background value after the softplus transform.
        """
        calibration: DetectorCalibration = _calibration(slit=slit)
        active_axes: int = 2 if slit else 3
        effects: DetectorEffects = _effects(
            background_mode="smooth",
            background_coefficients=jnp.linspace(
                -0.3, 0.4, 1 + 2 * active_axes
            ),
        )
        signal: jax.Array = jnp.ones(
            (
                1,
                calibration.u_bin_edges.size - 1,
                calibration.v_bin_edges.size - 1,
                calibration.energy_bin_edges_ev.size - 1,
            )
        )

        background: jax.Array = background_density(
            signal, calibration, effects
        )
        assert bool(jnp.all(background >= 0.0))

    def test_shirley_tail_uses_largest_recorded_energy(self) -> None:
        """Integrate the Shirley tail toward the largest energy.

        The case also pins the exact zero-signal branch derivative to zero.

        Notes
        -----
        The test compares unequal-width cumulative mass with NumPy. It then
        differentiates the production background at an all-zero signal.
        """
        calibration: DetectorCalibration = _calibration(slit=True)
        density: jax.Array = jnp.broadcast_to(
            jnp.array([[[[1.0, 2.0, 4.0]]]]), (1, 2, 1, 3)
        )
        base: float = 0.3
        scale: float = 0.8
        effects: DetectorEffects = _effects(
            background_mode="shirley",
            background_coefficients=jnp.array(
                [_inverse_softplus(base), _inverse_softplus(scale)]
            ),
        )
        delta_energy: Float64[np.ndarray, " E"] = np.array([1.25, 1.25, 2.0])
        weighted: Float64[np.ndarray, " E"] = (
            np.array([1.0, 2.0, 4.0]) * delta_energy
        )
        tail: Float64[np.ndarray, " E"] = np.flip(
            np.cumsum(np.flip(weighted))
        ) / np.sum(weighted)
        desired: Float64[np.ndarray, " E"] = base + scale * tail

        actual: jax.Array = background_density(density, calibration, effects)
        np.testing.assert_allclose(
            actual[0, 0, 0],
            desired,
            rtol=_DETERMINISTIC_RTOL,
            atol=0.0,
        )
        assert bool(actual[0, 0, 0, 0] > actual[0, 0, 0, -1])

        def zero_branch_loss(candidate: jax.Array) -> jax.Array:
            loss: jax.Array = jnp.sum(
                background_density(candidate, calibration, effects)
            )
            return loss

        zero_density: jax.Array = jnp.zeros_like(density)
        gradient: jax.Array = jax.grad(zero_branch_loss)(zero_density)
        chex.assert_trees_all_equal(gradient, jnp.zeros_like(zero_density))


class TestSensitivityField:
    """Verify :func:`diffpes.simul.sensitivity_field`.

    The class owns positivity and full-calibration volume normalization.
    """

    @pytest.mark.parametrize("slit", [False, True])
    def test_normalizes_full_volume_mean_for_map_and_slit(
        self, slit: bool
    ) -> None:
        """Normalize the full detector volume mean to one.

        The parameterized case covers both active-axis Legendre layouts.

        Notes
        -----
        The test evaluates the complete field before it computes the explicit
        native-volume-weighted mean.
        """
        calibration: DetectorCalibration = _calibration(slit=slit)
        active_axes: int = 2 if slit else 3
        effects: DetectorEffects = _effects(
            sensitivity_mode="smooth",
            sensitivity_coefficients=jnp.linspace(
                -0.17, 0.23, 2 * active_axes
            ),
        )

        sensitivity: jax.Array = sensitivity_field(calibration, effects)
        volumes: jax.Array = detector_bin_volumes(calibration)
        weighted_mean: jax.Array = jnp.sum(sensitivity * volumes) / jnp.sum(
            volumes
        )

        assert bool(jnp.all(sensitivity > 0.0))
        np.testing.assert_allclose(
            weighted_mean, 1.0, rtol=_DETERMINISTIC_RTOL, atol=0.0
        )


class TestApplyPostCountResponse:
    """Verify :func:`diffpes.simul.apply_post_count_response`.

    The class owns energy-only convolution, edge loss, and channel validation.
    """

    def test_convolves_energy_with_zero_padding_and_edge_loss(self) -> None:
        """Convolve only energy with zero exterior padding.

        The asymmetric kernel distinguishes convolution from correlation.

        Notes
        -----
        The test compares one detector row with NumPy and checks that exterior
        response leaves the recorded domain.
        """
        effects: DetectorEffects = _effects(
            post_count_mode="calibrated",
            post_count_kernel=jnp.array([1.0, 2.0, 4.0]),
        )
        rates: jax.Array = jnp.array([[[[1.0, 3.0, 5.0, 9.0]]]])
        actual: jax.Array = apply_post_count_response(rates, effects)
        desired: Float64[np.ndarray, " E"] = np.convolve(
            np.array([1.0, 3.0, 5.0, 9.0]),
            np.array([1.0, 2.0, 4.0]) / 7.0,
            mode="same",
        )

        np.testing.assert_allclose(
            actual[0, 0, 0],
            desired,
            rtol=_DETERMINISTIC_RTOL,
            atol=0.0,
        )
        assert float(jnp.sum(actual)) < float(jnp.sum(rates))

    def test_rejects_an_empty_channel_axis(self) -> None:
        """Reject response arrays without a physical channel.

        The case pins the audit-required nonempty channel boundary.

        Notes
        -----
        The shared rejection helper checks the same structural error eagerly
        and under JIT.
        """
        effects: DetectorEffects = _effects()
        empty_rates: jax.Array = jnp.empty((0, 2, 1, 3))
        assert_rejects(
            apply_post_count_response,
            empty_rates,
            effects,
            match="channel axis cannot be empty",
        )


class TestExpectedCounts:
    """Verify :func:`diffpes.simul.expected_counts`.

    The class owns physical count units and implemented rate derivatives.
    """

    def test_applies_flat_background_exposure_and_native_volume(self) -> None:
        """Apply every deterministic scalar and native-bin factor.

        The unequal-bin fixture exposes omission of detector volume.

        Notes
        -----
        The test compares every channel and bin with an independent analytic
        rate expression at the G8 tolerance.
        """
        calibration: DetectorCalibration = _calibration(slit=False)
        density: jax.Array = jnp.full((2, 2, 2, 3), 1.75)
        background_amplitude: float = 0.4
        effects: DetectorEffects = _effects(
            background_coefficients=jnp.array(
                [_inverse_softplus(background_amplitude)]
            ),
            exposure=3.2,
        )
        volumes: jax.Array = detector_bin_volumes(calibration)
        desired: jax.Array = (
            3.2 * (1.75 + background_amplitude) * volumes[None, ...]
        )
        actual: jax.Array = expected_counts(density, calibration, effects)

        np.testing.assert_allclose(
            actual,
            jnp.broadcast_to(desired, density.shape),
            rtol=_DETERMINISTIC_RTOL,
            atol=0.0,
        )

    def test_rejects_an_empty_channel_axis(self) -> None:
        """Reject detector densities without a physical channel.

        The case pins the audit-required expected-count boundary.

        Notes
        -----
        The shared helper runs the structural rejection eagerly and under JIT
        with a valid calibration and effects carrier.
        """
        calibration: DetectorCalibration = _calibration(slit=True)
        effects: DetectorEffects = _effects()
        empty_density: jax.Array = jnp.empty((0, 2, 1, 3))
        assert_rejects(
            expected_counts,
            empty_density,
            calibration,
            effects,
            match="channel axis cannot be empty",
        )

    def test_rates_pass_fd_jit_and_vmap(self) -> None:
        """Differentiate every implemented continuous rate leaf.

        The gate covers background, sensitivity, exposure, and response kernel.

        Notes
        -----
        The test applies the shared finite-difference harness. It then compares
        JIT output and vmaps a batch over every tested leaf.
        """
        calibration: DetectorCalibration
        density: jax.Array
        weights: jax.Array
        theta: Tuple[jax.Array, jax.Array, jax.Array, jax.Array]
        calibration, density, weights, theta = _d8_fixture()

        def rate_loss(
            candidate: Tuple[jax.Array, jax.Array, jax.Array, jax.Array],
        ) -> jax.Array:
            background: jax.Array
            sensitivity: jax.Array
            exposure: jax.Array
            kernel: jax.Array
            background, sensitivity, exposure, kernel = candidate
            effects: DetectorEffects = _effects(
                background_mode="smooth",
                background_coefficients=background,
                sensitivity_mode="smooth",
                sensitivity_coefficients=sensitivity,
                exposure=exposure,
                post_count_mode="calibrated",
                post_count_kernel=kernel,
            )
            rates: jax.Array = expected_counts(density, calibration, effects)
            loss: jax.Array = jnp.sum(rates * weights)
            return loss

        gradient_gate(rate_loss, theta, regime="smooth")
        eager_loss: jax.Array = rate_loss(theta)
        compiled_loss: jax.Array = jax.jit(rate_loss)(theta)
        chex.assert_trees_all_close(
            compiled_loss,
            eager_loss,
            rtol=_DETERMINISTIC_RTOL,
            atol=0.0,
        )
        batched_theta: Tuple[jax.Array, ...] = jax.tree.map(
            lambda leaf: jnp.stack((leaf, leaf * 1.04)), theta
        )
        batched_loss: jax.Array = jax.jit(jax.vmap(rate_loss))(batched_theta)
        chex.assert_shape(batched_loss, (2,))

    def test_stage_local_counts_exclude_map_and_transmission_leaves(
        self,
    ) -> None:
        """Keep map and transmission leaves outside stage-local counts.

        The case distinguishes the post-resolution primitive from the complete
        public detector chain.

        Notes
        -----
        Differentiate ``expected_counts`` alone with respect to domain logits,
        rotations, and transmission coordinates and require structural zeros.
        """
        calibration: DetectorCalibration = _calibration(slit=True)
        density: jax.Array = jnp.linspace(0.3, 1.1, 6).reshape((1, 2, 1, 3))
        theta: Tuple[jax.Array, jax.Array, jax.Array] = (
            jnp.array([0.2]),
            jnp.array([[0.1, -0.2, 0.3]]),
            jnp.array([0.15, -0.25]),
        )

        def loss(
            candidate: Tuple[jax.Array, jax.Array, jax.Array],
        ) -> jax.Array:
            logits: jax.Array
            rotations: jax.Array
            transmission: jax.Array
            logits, rotations, transmission = candidate
            effects: DetectorEffects = _effects(
                domain_logits=logits,
                domain_euler_angles_rad=rotations,
                transmission_raw_slopes=transmission,
            )
            rates: jax.Array = expected_counts(density, calibration, effects)
            total: jax.Array = jnp.sum(rates)
            return total

        gradient: Tuple[jax.Array, jax.Array, jax.Array] = jax.grad(loss)(
            theta
        )
        zeros: Tuple[jax.Array, ...] = jax.tree.map(jnp.zeros_like, theta)
        chex.assert_trees_all_equal(gradient, zeros)


class TestFixedTotalProbabilities:
    """Verify :func:`diffpes.simul.fixed_total_probabilities`.

    The class owns global normalization and probability derivatives.
    """

    def test_normalizes_one_global_event_vector(self) -> None:
        """Normalize all rates into one probability tensor.

        The case preserves the input shape and checks a unit global sum.

        Notes
        -----
        The test compares a nonnormalized matrix with its direct global ratio.
        It also rejects an all-zero rate tensor.
        """
        rates: jax.Array = jnp.array([[2.0, 3.0], [1.0, 4.0]])
        probabilities: jax.Array = fixed_total_probabilities(rates)
        desired: jax.Array = rates / jnp.sum(rates)

        np.testing.assert_allclose(
            probabilities,
            desired,
            rtol=_DETERMINISTIC_RTOL,
            atol=0.0,
        )
        assert_rejects(
            fixed_total_probabilities,
            jnp.zeros(3),
            match="positive sum",
        )

    def test_probabilities_pass_fd_jit_and_vmap(self) -> None:
        """Differentiate implemented leaves through event probabilities.

        Exposure stays fixed because global normalization removes its scale.

        Notes
        -----
        The test applies the shared finite-difference gate to background,
        sensitivity, and kernel leaves before JIT and vmap comparisons.
        """
        calibration: DetectorCalibration
        density: jax.Array
        weights: jax.Array
        rate_theta: Tuple[jax.Array, jax.Array, jax.Array, jax.Array]
        calibration, density, weights, rate_theta = _d8_fixture()
        theta: Tuple[jax.Array, jax.Array, jax.Array] = (
            rate_theta[0],
            rate_theta[1],
            rate_theta[3],
        )

        def probability_loss(
            candidate: Tuple[jax.Array, jax.Array, jax.Array],
        ) -> jax.Array:
            background: jax.Array
            sensitivity: jax.Array
            kernel: jax.Array
            background, sensitivity, kernel = candidate
            effects: DetectorEffects = _effects(
                background_mode="smooth",
                background_coefficients=background,
                sensitivity_mode="smooth",
                sensitivity_coefficients=sensitivity,
                exposure=2.3,
                post_count_mode="calibrated",
                post_count_kernel=kernel,
            )
            rates: jax.Array = expected_counts(density, calibration, effects)
            probabilities: jax.Array = fixed_total_probabilities(rates)
            loss: jax.Array = jnp.sum(probabilities * weights)
            return loss

        gradient_gate(probability_loss, theta, regime="smooth")
        eager_loss: jax.Array = probability_loss(theta)
        compiled_loss: jax.Array = jax.jit(probability_loss)(theta)
        chex.assert_trees_all_close(
            compiled_loss,
            eager_loss,
            rtol=_DETERMINISTIC_RTOL,
            atol=0.0,
        )
        batched_theta: Tuple[jax.Array, ...] = jax.tree.map(
            lambda leaf: jnp.stack((leaf, leaf * 1.04)), theta
        )
        batched_loss: jax.Array = jax.jit(jax.vmap(probability_loss))(
            batched_theta
        )
        chex.assert_shape(batched_loss, (2,))


class TestSamplePoissonCounts:
    """Verify :func:`diffpes.simul.sample_poisson_counts`.

    The class owns Poisson moments, replay, and the integer gradient boundary.
    """

    @pytest.mark.big_mem
    @pytest.mark.rss_limit_mb(900)
    def test_moments_stay_within_five_standard_errors(self) -> None:
        """Match Poisson means and variances at three rate scales.

        The fixed-seed gate uses 200,000 draws at rates 0.5, 5, and 50.

        Notes
        -----
        The test computes analytic standard errors from exact Poisson fourth
        moments and applies the preregistered five-error bound.
        """
        rates: jax.Array = jnp.array([0.5, 5.0, 50.0])
        keys: jax.Array = jax.random.split(jax.random.key(8201), _SAMPLE_DRAWS)
        draws: jax.Array = jax.jit(
            jax.vmap(sample_poisson_counts, in_axes=(0, None))
        )(keys, rates)
        empirical_mean: jax.Array = jnp.mean(draws, axis=0)
        empirical_variance: jax.Array = jnp.mean(
            jnp.square(draws - rates), axis=0
        )
        mean_error: jax.Array = jnp.sqrt(rates / _SAMPLE_DRAWS)
        variance_error: jax.Array = jnp.sqrt(
            (rates + 2.0 * jnp.square(rates)) / _SAMPLE_DRAWS
        )

        assert bool(
            jnp.all(jnp.abs(empirical_mean - rates) <= 5.0 * mean_error)
        )
        assert bool(
            jnp.all(
                jnp.abs(empirical_variance - rates) <= 5.0 * variance_error
            )
        )

    def test_replays_and_rejects_a_gradient_claim(self) -> None:
        """Replay one key and keep integer draws outside autodiff.

        The case requires bitwise equality and an integer-output gradient error.

        Notes
        -----
        The test calls the public sampler twice with one key. It then asks JAX
        for an unsupported gradient of the integer sum.
        """
        rates: jax.Array = jnp.array([0.2, 0.3, 0.5])
        key: jax.Array = jax.random.key(881)
        first: jax.Array = sample_poisson_counts(key, rates)
        second: jax.Array = sample_poisson_counts(key, rates)

        chex.assert_trees_all_equal(first, second)
        assert jnp.issubdtype(first.dtype, jnp.integer)
        with pytest.raises(TypeError, match="real-valued outputs"):
            jax.grad(
                lambda candidate: jnp.sum(
                    sample_poisson_counts(key, candidate)
                )
            )(rates)


class TestSampleFixedTotalCounts:
    """Verify :func:`diffpes.simul.sample_fixed_total_counts`.

    The class owns multinomial moments, exact totals, replay, and gradient scope.
    """

    @pytest.mark.big_mem
    @pytest.mark.rss_limit_mb(900)
    def test_moments_stay_within_five_standard_errors(self) -> None:
        """Match multinomial means and full covariance.

        The fixed-seed gate uses 200,000 draws with total 100.

        Notes
        -----
        The test derives covariance standard errors from exact categorical
        fourth moments and checks all nine covariance entries.
        """
        total_count: int = 100
        probabilities: jax.Array = jnp.array([0.2, 0.3, 0.5])
        keys: jax.Array = jax.random.split(jax.random.key(8202), _SAMPLE_DRAWS)
        draws: jax.Array = jax.jit(
            jax.vmap(sample_fixed_total_counts, in_axes=(0, None, None)),
            static_argnums=2,
        )(keys, probabilities, total_count)
        totals: jax.Array = jnp.sum(draws, axis=1)
        event_covariance: jax.Array = jnp.diag(probabilities) - jnp.outer(
            probabilities, probabilities
        )
        covariance: jax.Array = total_count * event_covariance
        expected_mean: jax.Array = total_count * probabilities
        centred: jax.Array = draws - expected_mean
        empirical_mean: jax.Array = jnp.mean(draws, axis=0)
        empirical_covariance: jax.Array = (
            jnp.einsum("ni,nj->ij", centred, centred) / _SAMPLE_DRAWS
        )
        one_hot: jax.Array = jnp.eye(probabilities.size)
        centred_event: jax.Array = one_hot - probabilities[None, :]
        event_fourth: jax.Array = jnp.einsum(
            "k,ki,ki,kj,kj->ij",
            probabilities,
            centred_event,
            centred_event,
            centred_event,
            centred_event,
        )
        event_variance: jax.Array = probabilities * (1.0 - probabilities)
        count_fourth: jax.Array = total_count * event_fourth + total_count * (
            total_count - 1
        ) * (
            jnp.outer(event_variance, event_variance)
            + 2.0 * jnp.square(event_covariance)
        )
        mean_error: jax.Array = jnp.sqrt(jnp.diag(covariance) / _SAMPLE_DRAWS)
        covariance_error: jax.Array = jnp.sqrt(
            (count_fourth - jnp.square(covariance)) / _SAMPLE_DRAWS
        )

        assert bool(jnp.all(totals == total_count))
        assert bool(
            jnp.all(
                jnp.abs(empirical_mean - expected_mean) <= 5.0 * mean_error
            )
        )
        assert bool(
            jnp.all(
                jnp.abs(empirical_covariance - covariance)
                <= 5.0 * covariance_error
            )
        )

    def test_replays_exact_total_and_rejects_a_gradient_claim(self) -> None:
        """Replay one key and preserve the declared event total.

        The case also keeps integer multinomial draws outside autodiff.

        Notes
        -----
        The test compares two fixed-key draws bitwise and checks their dtype and
        sum. It then requests an unsupported gradient of the integer sum.
        """
        rates: jax.Array = jnp.array([0.2, 0.3, 0.5])
        key: jax.Array = jax.random.key(881)
        first: jax.Array = sample_fixed_total_counts(key, rates, 113)
        second: jax.Array = sample_fixed_total_counts(key, rates, 113)

        chex.assert_trees_all_equal(first, second)
        assert int(jnp.sum(first)) == 113
        assert jnp.issubdtype(first.dtype, jnp.integer)
        with pytest.raises(TypeError, match="real-valued outputs"):
            jax.grad(
                lambda candidate: jnp.sum(
                    sample_fixed_total_counts(key, candidate, 113)
                )
            )(rates)


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
    kx: jax.Array = jnp.array([-0.5, 0.0, 0.5])
    ky: jax.Array = jnp.array([-0.45, 0.05, 0.55])
    energy: jax.Array = jnp.array([-0.4, 0.0, 0.4])
    intensity: jax.Array = (
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


class TestMapSourceToDetector:
    """Verify :func:`diffpes.simul.map_source_to_detector`.

    The class owns the public conservative-mapping surface and diagnostics.
    """

    def test_returns_finite_density_and_captured_flux(self) -> None:
        """Convert a named Cartesian source without inferring target bins.

        The case verifies the conservative public mapping boundary.

        Notes
        -----
        The focused public gate complements the private analytic/Jacobian
        battery and preserves the reported boundary-loss diagnostic.
        """
        source: ArpesCube
        geometry: ExperimentGeometry
        calibration: DetectorCalibration
        source, geometry, calibration, _ = _detector_chain_fixture()
        density: jax.Array
        captured: jax.Array
        density, captured = map_source_to_detector(
            source, geometry, calibration
        )

        chex.assert_shape(density, (2, 2, 2))
        assert bool(jnp.all(jnp.isfinite(density)))
        assert bool(jnp.all(density >= 0.0))
        assert 0.0 < float(captured) <= 1.0


class TestApplyDetectorEffects:
    """Verify :func:`diffpes.simul.apply_detector_effects`.

    The class owns the public stage order and native-count carrier boundary.
    """

    def test_matches_explicit_ordered_stage_composition(self) -> None:
        """Match the complete deterministic chain stage by stage.

        The case pins transmission before resolution and count assembly.

        Notes
        -----
        Transmission uses true kinetic energy before resolution; expected
        counts then apply background, sensitivity, exposure, volume, and the
        optional recorded response.
        """
        source: ArpesCube
        geometry: ExperimentGeometry
        calibration: DetectorCalibration
        effects: DetectorEffects
        source, geometry, calibration, effects = _detector_chain_fixture()
        mapped: jax.Array
        mapped, _ = map_source_to_detector(source, geometry, calibration)
        recorded_energy: jax.Array = 0.5 * (
            calibration.energy_bin_edges_ev[:-1]
            + calibration.energy_bin_edges_ev[1:]
        )
        kinetic_energy: jax.Array = (
            geometry.photon_energy_ev
            - geometry.work_function_ev
            + recorded_energy
        )
        transmitted: jax.Array = apply_transmission(
            mapped,
            kinetic_energy,
            effects.transmission_raw_slopes,
            calibration,
        )
        resolved: jax.Array = apply_resolution(transmitted, calibration)[0]
        desired: jax.Array = expected_counts(
            resolved[None, ...], calibration, effects
        )

        raster: DetectorRaster = apply_detector_effects(
            (source,), geometry, calibration, effects
        )

        chex.assert_trees_all_close(raster.expected_counts, desired)
        chex.assert_trees_all_equal(
            raster.detector_u_axis,
            0.5 * (calibration.u_bin_edges[:-1] + calibration.u_bin_edges[1:]),
        )
        chex.assert_trees_all_equal(
            raster.detector_v_axis,
            0.5 * (calibration.v_bin_edges[:-1] + calibration.v_bin_edges[1:]),
        )
        chex.assert_trees_all_equal(raster.energy_axis, recorded_energy)
        assert raster.channel_labels == ("intensity",)

    def test_jit_success_path_preserves_counts(self) -> None:
        """Compile the whole deterministic source-to-count chain.

        The case requires compiled and eager detector carriers to agree.

        Notes
        -----
        The compiled result includes mapping, transmission, native resolution,
        and expected-count construction rather than a stage-local surrogate.
        """
        source: ArpesCube
        geometry: ExperimentGeometry
        calibration: DetectorCalibration
        effects: DetectorEffects
        source, geometry, calibration, effects = _detector_chain_fixture()
        eager: DetectorRaster = apply_detector_effects(
            (source,), geometry, calibration, effects
        )
        compiled: DetectorRaster = jax.jit(apply_detector_effects)(
            (source,), geometry, calibration, effects
        )

        chex.assert_trees_all_close(compiled, eager)
