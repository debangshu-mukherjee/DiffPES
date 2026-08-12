"""Validate the detector response module.

The cases use analytic values, invariants, and finite differences.
"""

import chex
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from beartype.typing import Tuple
from jaxtyping import Array, Float64
from numpy.typing import NDArray

from diffpes.simul import (
    apply_post_count_response,
    background_density,
    detector_bin_volumes,
    expected_counts,
    sensitivity_field,
)
from diffpes.types import (
    DetectorCalibration,
    DetectorEffects,
)
from tests._assertions import assert_rejects
from tests._gradients import assert_gradients_match_finite_differences

from ._effects_helpers import (
    _DETERMINISTIC_RTOL,
    _calibration,
    _effects,
    _inverse_softplus,
    _smooth_effects_fixture,
)


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
        target bin at the deterministic registered tolerance.
        """
        calibration: DetectorCalibration = _calibration(slit=False)
        actual: Float64[Array, "..."] = detector_bin_volumes(calibration)
        desired: Float64[NDArray, "U V E"] = (
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
        signal: Float64[Array, "..."] = jnp.ones(
            (
                1,
                calibration.u_bin_edges.size - 1,
                calibration.v_bin_edges.size - 1,
                calibration.energy_bin_edges_ev.size - 1,
            )
        )

        background: Float64[Array, "..."] = background_density(
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
        density: Float64[Array, "..."] = jnp.broadcast_to(
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
        delta_energy: Float64[NDArray, " E"] = np.array([1.25, 1.25, 2.0])
        weighted: Float64[NDArray, " E"] = (
            np.array([1.0, 2.0, 4.0]) * delta_energy
        )
        tail: Float64[NDArray, " E"] = np.flip(
            np.cumsum(np.flip(weighted))
        ) / np.sum(weighted)
        desired: Float64[NDArray, " E"] = base + scale * tail

        actual: Float64[Array, "..."] = background_density(
            density, calibration, effects
        )
        np.testing.assert_allclose(
            actual[0, 0, 0],
            desired,
            rtol=_DETERMINISTIC_RTOL,
            atol=0.0,
        )
        assert bool(actual[0, 0, 0, 0] > actual[0, 0, 0, -1])

        def zero_branch_loss(
            candidate: Float64[Array, "..."],
        ) -> Float64[Array, "..."]:
            loss: Float64[Array, "..."] = jnp.sum(
                background_density(candidate, calibration, effects)
            )
            return loss

        zero_density: Float64[Array, "..."] = jnp.zeros_like(density)
        gradient: Float64[Array, "..."] = jax.grad(zero_branch_loss)(
            zero_density
        )
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

        sensitivity: Float64[Array, "..."] = sensitivity_field(
            calibration, effects
        )
        volumes: Float64[Array, "..."] = detector_bin_volumes(calibration)
        weighted_mean: Float64[Array, "..."] = jnp.sum(
            sensitivity * volumes
        ) / jnp.sum(volumes)

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
        rates: Float64[Array, "1 1 1 4"] = jnp.array(
            [[[[1.0, 3.0, 5.0, 9.0]]]]
        )
        actual: Float64[Array, "..."] = apply_post_count_response(
            rates, effects
        )
        desired: Float64[NDArray, " E"] = np.convolve(
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
        empty_rates: Float64[Array, "0 2 1 3"] = jnp.empty((0, 2, 1, 3))
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
        rate expression at the registered tolerance.
        """
        calibration: DetectorCalibration = _calibration(slit=False)
        density: Float64[Array, "2 2 2 3"] = jnp.full((2, 2, 2, 3), 1.75)
        background_amplitude: float = 0.4
        effects: DetectorEffects = _effects(
            background_coefficients=jnp.array(
                [_inverse_softplus(background_amplitude)]
            ),
            exposure=3.2,
        )
        volumes: Float64[Array, "..."] = detector_bin_volumes(calibration)
        desired: Float64[Array, "..."] = (
            3.2 * (1.75 + background_amplitude) * volumes[None, ...]
        )
        actual: Float64[Array, "..."] = expected_counts(
            density, calibration, effects
        )

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
        empty_density: Float64[Array, "0 2 1 3"] = jnp.empty((0, 2, 1, 3))
        assert_rejects(
            expected_counts,
            empty_density,
            calibration,
            effects,
            match="channel axis cannot be empty",
        )

    def test_rates_pass_fd_jit_and_vmap(self) -> None:
        """Differentiate every implemented continuous rate leaf.

        The check covers background, sensitivity, exposure, and response
        kernel.

        Notes
        -----
        The test applies the shared finite-difference harness. It then compares
        JIT output and vmaps a batch over every tested leaf.
        """
        calibration: DetectorCalibration
        density: Float64[Array, "..."]
        weights: Float64[Array, "..."]
        theta: Tuple[
            Float64[Array, "..."],
            Float64[Array, "..."],
            Float64[Array, "..."],
            Float64[Array, "..."],
        ]
        calibration, density, weights, theta = _smooth_effects_fixture()

        def rate_loss(
            candidate: Tuple[
                Float64[Array, "..."],
                Float64[Array, "..."],
                Float64[Array, "..."],
                Float64[Array, "..."],
            ],
        ) -> Float64[Array, "..."]:
            background: Float64[Array, "..."]
            sensitivity: Float64[Array, "..."]
            exposure: Float64[Array, "..."]
            kernel: Float64[Array, "..."]
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
            rates: Float64[Array, "..."] = expected_counts(
                density, calibration, effects
            )
            loss: Float64[Array, "..."] = jnp.sum(rates * weights)
            return loss

        assert_gradients_match_finite_differences(
            rate_loss, theta, regime="smooth"
        )
        eager_loss: Float64[Array, "..."] = rate_loss(theta)
        compiled_loss: Float64[Array, "..."] = jax.jit(rate_loss)(theta)
        chex.assert_trees_all_close(
            compiled_loss,
            eager_loss,
            rtol=_DETERMINISTIC_RTOL,
            atol=0.0,
        )
        batched_theta: Tuple[Float64[Array, "..."], ...] = jax.tree.map(
            lambda leaf: jnp.stack((leaf, leaf * 1.04)), theta
        )
        batched_loss: Float64[Array, "..."] = jax.jit(jax.vmap(rate_loss))(
            batched_theta
        )
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
        density: Float64[Array, "..."] = jnp.linspace(0.3, 1.1, 6).reshape(
            (1, 2, 1, 3)
        )
        theta: Tuple[
            Float64[Array, "..."], Float64[Array, "..."], Float64[Array, "..."]
        ] = (
            jnp.array([0.2]),
            jnp.array([[0.1, -0.2, 0.3]]),
            jnp.array([0.15, -0.25]),
        )

        def loss(
            candidate: Tuple[
                Float64[Array, "..."],
                Float64[Array, "..."],
                Float64[Array, "..."],
            ],
        ) -> Float64[Array, "..."]:
            logits: Float64[Array, "..."]
            rotations: Float64[Array, "..."]
            transmission: Float64[Array, "..."]
            logits, rotations, transmission = candidate
            effects: DetectorEffects = _effects(
                domain_logits=logits,
                domain_euler_angles_rad=rotations,
                transmission_raw_slopes=transmission,
            )
            rates: Float64[Array, "..."] = expected_counts(
                density, calibration, effects
            )
            total: Float64[Array, "..."] = jnp.sum(rates)
            return total

        gradient: Tuple[
            Float64[Array, "..."], Float64[Array, "..."], Float64[Array, "..."]
        ] = jax.grad(loss)(theta)
        zeros: Tuple[Float64[Array, "..."], ...] = jax.tree.map(
            jnp.zeros_like, theta
        )
        chex.assert_trees_all_equal(gradient, zeros)
