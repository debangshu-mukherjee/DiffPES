"""Verify detector effects, expected counts, and acquisition sampling.

The tests pin the bounded WP8.8 deterministic and stochastic contracts. They
also cover the implemented expected-rate and event-probability derivatives.
"""

import chex
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from beartype.typing import Dict, Tuple
from jaxtyping import Float64

from diffpes.simul.effects import (
    apply_post_count_response,
    background_density,
    detector_bin_volumes,
    expected_counts,
    fixed_total_probabilities,
    sample_fixed_total_counts,
    sample_poisson_counts,
    sensitivity_field,
)
from diffpes.types import (
    DetectorCalibration,
    DetectorEffects,
    make_detector_calibration,
    make_detector_effects,
)
from tests._assertions import assert_rejects
from tests._gradients import gradient_gate

_FRAME_ID: str = "org.diffpes.frame.sample_cartesian"
_DETERMINISTIC_RTOL: float = 1.0e-10
_SAMPLE_DRAWS: int = 200_000


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

    def test_upstream_effects_leaves_remain_deferred(self) -> None:
        """Keep upstream effects leaves outside this bounded D8 claim.

        The case documents structural zeros before mapping and transmission land.

        Notes
        -----
        The test differentiates expected counts with respect to domain logits,
        rotations, and transmission coordinates and requires exact zeros.
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
