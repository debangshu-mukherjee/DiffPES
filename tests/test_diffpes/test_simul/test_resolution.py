"""Validate the resolution module.

The cases use analytic values, invariants, and finite differences.
"""

from itertools import pairwise

import chex
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from beartype.typing import Callable, List, Tuple
from hypothesis import given, settings
from hypothesis import strategies as st
from jaxtyping import Array, Bool, Float64
from numpy.typing import NDArray
from scipy import ndimage

from diffpes.constants import (
    K_PREFACTOR_INV_ANG_SQRT_EV,
)
from diffpes.simul import (
    apply_resolution,
    apply_transmission,
    convolve_energy,
    convolve_kpath,
    convolve_momentum_map,
    gaussian_kernel_1d,
)
from diffpes.types import (
    ArpesCube,
    DetectorCalibration,
    constant_energy_map,
    fermi_surface_map,
    make_arpes_cube,
    make_detector_calibration,
)
from tests._assertions import assert_rejects
from tests._gradients import assert_gradients_match_finite_differences

from ._effects_helpers import (
    _FINITE_VOLUME_RTOL,
    _FWHM_TO_SIGMA,
    _SAMPLED_HALF_WIDTH,
    _reference_finite_volume_matrix,
    _reference_resolution,
    _resolution_calibration,
)


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
        kernel: Float64[Array, "..."] = gaussian_kernel_1d(6.0)

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

    The class owns sampled-energy parity and validation.
    """

    @pytest.mark.parametrize("sigma_over_dx", [0.5, 1.0, 2.0, 6.0])
    def test_matches_scipy_sampled_energy_stencil(
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
        samples: Float64[NDArray, "A B E"] = rng.normal(size=(2, 3, 31))
        spacing: float = 0.04
        energy: Float64[Array, "..."] = (
            jnp.arange(samples.shape[-1]) * spacing - 0.6
        )
        actual: Float64[Array, "..."] = convolve_energy(
            jnp.asarray(samples),
            energy,
            sigma_over_dx * spacing,
        )
        desired: Float64[NDArray, "A B E"] = ndimage.gaussian_filter1d(
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

    The class owns Cartesian-map SciPy parity with explicit physical axes.
    """

    def test_matches_separable_scipy_filter(self) -> None:
        """Match SciPy on unequal uniform Cartesian momentum spacings.

        The case pins separable physical-axis scaling on a nonsquare map.

        Notes
        -----
        The test builds independent SciPy output with one sigma per momentum
        spacing and the registered static radius.
        """
        rng: np.random.Generator = np.random.default_rng(8102)
        samples: Float64[NDArray, "Kx Ky E"] = rng.normal(size=(9, 7, 4))
        kx: Float64[Array, "9"] = jnp.linspace(-0.24, 0.24, 9)
        ky: Float64[Array, "7"] = jnp.linspace(-0.15, 0.15, 7)
        sigma: float = 0.06
        actual: Float64[Array, "..."] = convolve_momentum_map(
            jnp.asarray(samples), kx, ky, sigma
        )
        desired: Float64[NDArray, "Kx Ky E"] = ndimage.gaussian_filter(
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

    The class owns sampled numerical parity and finite-volume path semantics.
    """

    def test_shared_sampled_stencil_matches_scipy(self) -> None:
        """Match SciPy through the shared uniform-axis sampled implementation.

        The case pins the sampled-parity path without changing finite-volume
        behavior.

        Notes
        -----
        The test transposes the path axis through the public sampled energy
        helper and compares independent SciPy output.
        """
        rng: np.random.Generator = np.random.default_rng(8103)
        samples: Float64[NDArray, "K E"] = rng.normal(size=(25, 3))
        centres: Float64[Array, "25"] = jnp.linspace(-0.6, 0.6, 25)
        sigma: float = 0.075
        sampled: Float64[Array, "..."] = convolve_energy(
            jnp.asarray(samples).T,
            centres,
            sigma,
        ).T
        desired: Float64[NDArray, "K E"] = ndimage.gaussian_filter1d(
            samples,
            sigma=sigma / 0.05,
            axis=0,
            mode="constant",
            radius=_SAMPLED_HALF_WIDTH,
        )

        np.testing.assert_allclose(
            sampled, desired, rtol=1.0e-10, atol=1.0e-13
        )

    def test_matches_nonuniform_analytic_finite_volume(self) -> None:
        """Match analytic nonuniform cells without row normalization.

        The case verifies physical density transport and captured boundary
        flux.

        Notes
        -----
        The test constructs cell edges and an independent analytic Gaussian
        matrix before comparing density and flux diagnostics.
        """
        centres_np: Float64[NDArray, " K"] = np.array(
            [-0.8, -0.35, -0.1, 0.4, 1.1]
        )
        density_np: Float64[NDArray, "K E"] = np.array(
            [
                [0.2, 0.7],
                [1.1, 0.4],
                [0.8, 1.5],
                [1.7, 0.3],
                [0.4, 0.9],
            ]
        )
        sigma: float = 0.22
        interior: Float64[NDArray, " Km1"] = 0.5 * (
            centres_np[:-1] + centres_np[1:]
        )
        edges: Float64[NDArray, " Kp1"] = np.concatenate(
            (
                [centres_np[0] - 0.5 * (centres_np[1] - centres_np[0])],
                interior,
                [centres_np[-1] + 0.5 * (centres_np[-1] - centres_np[-2])],
            )
        )
        matrix: Float64[NDArray, "K K"] = _reference_finite_volume_matrix(
            edges, sigma
        )
        desired: Float64[NDArray, "K E"] = matrix @ density_np
        widths: Float64[NDArray, " K"] = np.diff(edges)
        desired_fraction: float = float(
            np.sum(desired * widths[:, None])
            / np.sum(density_np * widths[:, None])
        )

        actual: Float64[Array, "..."]
        fraction: Float64[Array, "..."]
        valid: Bool[Array, "..."]
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

    def test_two_center_counterexample_rejects_sampled_flux_creation(
        self,
    ) -> None:
        """Pin the coarse two-centre analytic result and sampled-rule failure.

        The case exposes flux creation from the retired sampled-cell rule.

        Notes
        -----
        The test compares the finite-volume captured fraction with the analytic
        value produced by the planted sampled counterexample.
        """
        centres: Float64[Array, "2"] = jnp.array([0.0, 1.0])
        density: Float64[Array, "2 1"] = jnp.ones((2, 1))
        sigma: float = 0.01
        finite_volume_fraction: Float64[Array, "..."]
        valid: Bool[Array, "..."]
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

    def test_domain_enlargement_recovers_flux_and_zero_is_invalid(
        self,
    ) -> None:
        """Recover escaped mass with source padding and flag zero input.

        The case distinguishes physical boundary loss from an invalid zero
        rate.

        Notes
        -----
        The test enlarges the source domain and compares captured fractions
        before checking exact zero-output diagnostics.
        """
        compact_centres: Float64[Array, "..."] = jnp.arange(-1.0, 2.0)
        compact_density: Float64[Array, "3 1"] = jnp.array(
            [[0.0], [1.0], [0.0]]
        )
        extended_centres: Float64[Array, "..."] = jnp.arange(-3.0, 4.0)
        extended_density: Float64[Array, "7 1"] = jnp.array(
            [[0.0], [0.0], [0.0], [1.0], [0.0], [0.0], [0.0]]
        )
        compact_fraction: Float64[Array, "..."]
        _, compact_fraction, _ = convolve_kpath(
            compact_density, compact_centres, 0.7
        )
        extended_fraction: Float64[Array, "..."]
        _, extended_fraction, _ = convolve_kpath(
            extended_density, extended_centres, 0.7
        )
        zero: Float64[Array, "..."]
        zero_fraction: Float64[Array, "..."]
        zero_valid: Bool[Array, "..."]
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
        centres: Float64[Array, "17"] = jnp.linspace(-4.0, 4.0, 17)
        density: Float64[Array, "..."] = jnp.zeros((17, 1)).at[8, 0].set(1.0)
        broader: float = narrower + increment
        narrow_density: Float64[Array, "..."]
        narrow_fraction: Float64[Array, "..."]
        broad_density: Float64[Array, "..."]
        broad_fraction: Float64[Array, "..."]
        narrow_density, narrow_fraction, _ = convolve_kpath(
            density, centres, narrower
        )
        broad_density, broad_fraction, _ = convolve_kpath(
            density, centres, broader
        )
        widths: Float64[Array, "17"] = jnp.full((17,), centres[1] - centres[0])

        def variance(
            candidate: Float64[Array, "..."],
        ) -> Float64[Array, "..."]:
            """Return the captured-density variance."""
            mass: Float64[Array, "..."] = candidate[:, 0] * widths
            probability: Float64[Array, "..."] = mass / jnp.sum(mass)
            mean: Float64[Array, "..."] = jnp.sum(probability * centres)
            returned: Float64[Array, "..."] = jnp.sum(
                probability * (centres - mean) ** 2
            )
            return returned

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

    The class owns finite-volume energy/momentum resolution and width
    gradients.
    """

    @pytest.mark.parametrize("profile", ["delta", "constant", "translated"])
    def test_finite_volume_matches_independent_analytic_reference(
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
        density_np: Float64[NDArray, "C U V E"] = np.zeros((1, 3, 2, 4))
        if profile == "delta":
            density_np[0, 0, 0, 0] = 2.3
        elif profile == "constant":
            density_np[...] = 0.7
        else:
            density_np[0, 1, 1, 2] = 1.4
            density_np[0, 2, 0, 1] = 0.6
        desired: Float64[NDArray, "C U V E"]
        desired_fractions: Float64[NDArray, " 3"]
        desired, desired_fractions = _reference_resolution(
            density_np, calibration
        )

        actual: Float64[Array, "..."]
        fractions: Float64[Array, "..."]
        valid: Bool[Array, "..."]
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

    def test_anisotropic_native_widths_do_not_become_stationary_k(
        self,
    ) -> None:
        """Expose swapped native angular widths on unequal detector bins.

        The case prevents replacing native-coordinate widths with stationary k.

        Notes
        -----
        The test swaps the two angular FWHMs and requires a measurable change
        in the resolved detector density.
        """
        density: Float64[Array, "..."] = (
            jnp.zeros((1, 3, 2, 4)).at[0, 0, 1, 2].set(1.0)
        )
        calibrated: DetectorCalibration = _resolution_calibration(
            widths=(0.31, 0.08, 0.2)
        )
        swapped: DetectorCalibration = _resolution_calibration(
            widths=(0.08, 0.31, 0.2)
        )
        desired: Float64[Array, "..."]
        desired, _, _ = apply_resolution(density, calibrated)
        planted_stationary_k: Float64[Array, "..."]
        planted_stationary_k, _, _ = apply_resolution(density, swapped)

        assert float(jnp.max(jnp.abs(desired - planted_stationary_k))) > 1.0e-3

    def test_nonlinear_kinematics_rejects_stationary_k_width(self) -> None:
        """Reject a fixed momentum PSF across energy-dependent angle maps.

        An independent native-angle calculation supplies the accepted result.
        The planted alternative conserves each bin's mass while transforming
        to ``k = p(E) sin(u)``, applies one fixed momentum width, and
        transforms
        back. Its energy-dependent angular profiles must fail the native truth.

        Notes
        -----
        Evaluate all energies and compare the transformed profiles directly.
        """
        u_edges: Float64[NDArray, " U1"] = np.array(
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
        profile: Float64[NDArray, " U"] = np.array(
            [0.1, 0.4, 1.7, 0.8, 0.25, 0.05, 0.02]
        )
        density_np: Float64[NDArray, "C U V E"] = np.broadcast_to(
            profile[None, :, None, None], (1, 7, 1, 3)
        ).copy()
        native_truth: Float64[NDArray, "C U V E"]
        native_fractions: Float64[NDArray, " 3"]
        native_truth, native_fractions = _reference_resolution(
            density_np, calibration
        )
        actual: Float64[Array, "..."]
        actual_fractions: Float64[Array, "..."]
        actual, actual_fractions, _ = apply_resolution(
            jnp.asarray(density_np), calibration
        )

        energy_centres: Float64[NDArray, " E"] = np.array([-15.0, 0.0, 15.0])
        kinetic_energy: Float64[NDArray, " E"] = 46.0 + energy_centres
        momenta: Float64[NDArray, " E"] = float(
            K_PREFACTOR_INV_ANG_SQRT_EV
        ) * np.sqrt(kinetic_energy)
        sigma_u: float = 0.12 * _FWHM_TO_SIGMA
        sigma_k: float = momenta[1] * sigma_u
        angular_widths: Float64[NDArray, " U"] = np.diff(u_edges)
        stationary_u: Float64[NDArray, "C U V E"] = np.empty_like(density_np)
        energy_index: int
        momentum: np.float64
        for energy_index, momentum in enumerate(momenta):
            k_edges: Float64[NDArray, " U1"] = momentum * np.sin(u_edges)
            k_widths: Float64[NDArray, " U"] = np.diff(k_edges)
            source_k: Float64[NDArray, " U"] = (
                density_np[0, :, 0, energy_index] * angular_widths / k_widths
            )
            stationary_k: Float64[NDArray, " U"] = (
                _reference_finite_volume_matrix(k_edges, sigma_k) @ source_k
            )
            stationary_u[0, :, 0, energy_index] = (
                stationary_k * k_widths / angular_widths
            )
        v_matrix: Float64[NDArray, "V V"] = _reference_finite_volume_matrix(
            np.asarray(calibration.v_bin_edges),
            float(calibration.psf_fwhm_v) * _FWHM_TO_SIGMA,
        )
        energy_matrix: Float64[NDArray, "E E"] = (
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
        density: Float64[Array, "2 3 2 4"] = jnp.zeros((2, 3, 2, 4))
        blurred: Float64[Array, "..."]
        fractions: Float64[Array, "..."]
        valid: Bool[Array, "..."]
        blurred, fractions, valid = apply_resolution(density, calibration)

        chex.assert_trees_all_equal(blurred, jnp.zeros_like(density))
        chex.assert_trees_all_equal(fractions, jnp.zeros(3))
        assert not bool(valid)

    def test_width_and_intensity_gradients_match_finite_differences(
        self,
    ) -> None:
        """Check fwd/rev gradients through all three FWHMs and density.

        The case covers every calibrated width and the full intensity tensor.

        Notes
        -----
        The shared gradient check compares forward and reverse autodiff with
        its
        finite-difference ladder on a smooth fixture.
        """
        calibration: DetectorCalibration = _resolution_calibration()
        widths: Float64[Array, "3"] = jnp.array([0.23, 0.17, 0.29])
        density: Float64[Array, "..."] = jnp.linspace(0.2, 1.3, 24).reshape(
            (1, 3, 2, 4)
        )
        weights: Float64[Array, "..."] = jnp.linspace(-0.4, 0.8, 24).reshape(
            density.shape
        )

        def loss(
            theta: Tuple[Float64[Array, "..."], Float64[Array, "..."]],
        ) -> Float64[Array, "..."]:
            candidate_widths: Float64[Array, "..."]
            candidate_density: Float64[Array, "..."]
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
            blurred: Float64[Array, "..."]
            fractions: Float64[Array, "..."]
            blurred, fractions, _ = apply_resolution(
                candidate_density, candidate
            )
            returned: Float64[Array, "..."] = jnp.sum(
                blurred * weights
            ) + jnp.dot(fractions, jnp.array([0.3, -0.2, 0.4]))
            return returned

        assert_gradients_match_finite_differences(
            loss, (widths, density), regime="smooth"
        )

    def test_decreasing_widths_converge_one_sided_to_identity(self) -> None:
        """Verify value convergence as every positive native width decreases.

        The sequence stays above the registered positive-width floor and
        deliberately makes no derivative claim at the rejected zero limit.

        Notes
        -----
        Measure successive errors against the unchanged input density.
        """
        density: Float64[Array, "1 3 2 4"] = jnp.array(
            [
                [
                    [[0.2, 1.1, 0.4, 0.8], [1.3, 0.1, 0.7, 0.5]],
                    [[0.9, 0.3, 1.4, 0.2], [0.4, 1.2, 0.6, 1.0]],
                    [[1.5, 0.2, 0.8, 0.4], [0.3, 1.1, 0.5, 1.3]],
                ]
            ]
        )
        width_scales: Tuple[float, ...] = (0.18, 0.09, 0.045, 0.0225)
        errors: List[float] = []
        width: float
        for width in width_scales:
            calibration: DetectorCalibration = _resolution_calibration(
                widths=(width, width, width)
            )
            blurred: Float64[Array, "..."] = apply_resolution(
                density, calibration
            )[0]
            errors.append(float(jnp.linalg.norm(blurred - density)))

        assert all(later < earlier for earlier, later in pairwise(errors))
        assert errors[-1] < 0.25 * errors[0]


class TestDisplayTopHatDerivatives:
    """Verify the deliberately nonsmooth display-window derivative contract.

    The case differentiates the top-hat coordinates and requires the documented
    zero gradients at the nonsmooth display boundaries.
    """

    def test_top_hat_coordinates_are_documented_exact_zeros(self) -> None:
        """Assert zero gradients away from membership seams.

        The case covers both constant-energy and Fermi-surface display helpers.

        Notes
        -----
        Differentiate interior membership regions and inspect their
        documentation.
        """
        energy: Float64[Array, "4"] = jnp.array([-1.0, -0.2, 0.4, 1.2])
        intensity: Float64[Array, "..."] = (
            jnp.arange(16.0).reshape((2, 2, 4)) + 1.0
        )
        cube: ArpesCube = make_arpes_cube(
            intensity,
            jnp.array([-0.3, 0.4]),
            jnp.array([-0.5, 0.2]),
            energy,
        )
        window_grad: Float64[Array, "..."] = jax.grad(
            lambda window: jnp.sum(
                constant_energy_map(cube, window[0], window[1])
            )
        )(jnp.array([-0.15, 0.1]))
        tolerance_grad: Float64[Array, "..."] = jax.grad(
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


class TestResolutionTransmissionVariants(chex.TestCase):
    """Verify eager and JIT success paths for both canonical effects.

    The cases apply the resolution and transmission operators in eager and
    compiled execution, then compare the results with expected arrays.
    """

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
        density: Float64[Array, "..."] = jnp.linspace(0.2, 1.3, 24).reshape(
            (1, 3, 2, 4)
        )
        blurred: Float64[Array, "..."]
        fractions: Float64[Array, "..."]
        valid: Bool[Array, "..."]
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
        operator: Callable[..., Float64[Array, "..."]] = self.variant(
            apply_transmission
        )
        calibration: DetectorCalibration = _resolution_calibration()
        energy: Float64[Array, "4"] = jnp.array([14.0, 25.0, 37.0, 46.0])
        intensity: Float64[Array, "2 3 4"] = jnp.ones((2, 3, 4))
        actual: Float64[Array, "..."] = operator(
            intensity, energy, jnp.array([-0.4, 0.2]), calibration
        )

        chex.assert_shape(actual, intensity.shape)
        assert bool(jnp.all(jnp.isfinite(actual)))
