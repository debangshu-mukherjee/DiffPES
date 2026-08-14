"""Validate the counting module.

The cases use analytic values, invariants, and finite differences.
"""

import chex
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from beartype.typing import Tuple
from jaxtyping import Array, Float64

from diffpes.simul import (
    expected_counts,
    fixed_total_probabilities,
    sample_fixed_total_counts,
    sample_poisson_counts,
)
from diffpes.types import (
    DetectorCalibration,
    DetectorEffects,
)
from tests._assertions import assert_rejects
from tests._gradients import assert_gradients_match_finite_differences

from ._effects_helpers import (
    _DETERMINISTIC_RTOL,
    _SAMPLE_DRAWS,
    _effects,
    _smooth_effects_fixture,
)


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
        rates: Float64[Array, "2 2"] = jnp.array([[2.0, 3.0], [1.0, 4.0]])
        probabilities: Float64[Array, "..."] = fixed_total_probabilities(rates)
        desired: Float64[Array, "..."] = rates / jnp.sum(rates)

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

    @pytest.mark.slow
    def test_probabilities_pass_fd_jit_and_vmap(self) -> None:
        """Differentiate implemented leaves through event probabilities.

        Exposure stays fixed because global normalization removes its scale.

        Notes
        -----
        The test applies the shared finite-difference check to background,
        sensitivity, and kernel leaves before JIT and vmap comparisons.
        """
        calibration: DetectorCalibration
        density: Float64[Array, "..."]
        weights: Float64[Array, "..."]
        rate_theta: Tuple[
            Float64[Array, "..."],
            Float64[Array, "..."],
            Float64[Array, "..."],
            Float64[Array, "..."],
        ]
        calibration, density, weights, rate_theta = _smooth_effects_fixture()
        theta: Tuple[
            Float64[Array, "..."], Float64[Array, "..."], Float64[Array, "..."]
        ] = (
            rate_theta[0],
            rate_theta[1],
            rate_theta[3],
        )

        def probability_loss(
            candidate: Tuple[
                Float64[Array, "..."],
                Float64[Array, "..."],
                Float64[Array, "..."],
            ],
        ) -> Float64[Array, "..."]:
            background: Float64[Array, "..."]
            sensitivity: Float64[Array, "..."]
            kernel: Float64[Array, "..."]
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
            rates: Float64[Array, "..."] = expected_counts(
                density, calibration, effects
            )
            probabilities: Float64[Array, "..."] = fixed_total_probabilities(
                rates
            )
            loss: Float64[Array, "..."] = jnp.sum(probabilities * weights)
            return loss

        assert_gradients_match_finite_differences(
            probability_loss, theta, regime="smooth"
        )
        eager_loss: Float64[Array, "..."] = probability_loss(theta)
        compiled_loss: Float64[Array, "..."] = jax.jit(probability_loss)(theta)
        chex.assert_trees_all_close(
            compiled_loss,
            eager_loss,
            rtol=_DETERMINISTIC_RTOL,
            atol=0.0,
        )
        batched_theta: Tuple[Float64[Array, "..."], ...] = jax.tree.map(
            lambda leaf: jnp.stack((leaf, leaf * 1.04)), theta
        )
        batched_loss: Float64[Array, "..."] = jax.jit(
            jax.vmap(probability_loss)
        )(batched_theta)
        chex.assert_shape(batched_loss, (2,))


class TestSamplePoissonCounts:
    """Verify :func:`diffpes.simul.sample_poisson_counts`.

    The class owns Poisson moments, replay, and the integer gradient boundary.
    """

    @pytest.mark.big_mem
    @pytest.mark.rss_limit_mb(900)
    def test_moments_stay_within_five_standard_errors(self) -> None:
        """Match Poisson means and variances at three rate scales.

        The fixed-seed check uses 200,000 draws at rates 0.5, 5, and 50.

        Notes
        -----
        The test computes analytic standard errors from exact Poisson fourth
        moments and applies the preregistered five-error bound.
        """
        rates: Float64[Array, "3"] = jnp.array([0.5, 5.0, 50.0])
        keys: Float64[Array, "..."] = jax.random.split(
            jax.random.key(8201), _SAMPLE_DRAWS
        )
        draws: Float64[Array, "..."] = jax.jit(
            jax.vmap(sample_poisson_counts, in_axes=(0, None))
        )(keys, rates)
        empirical_mean: Float64[Array, "..."] = jnp.mean(draws, axis=0)
        empirical_variance: Float64[Array, "..."] = jnp.mean(
            jnp.square(draws - rates), axis=0
        )
        mean_error: Float64[Array, "..."] = jnp.sqrt(rates / _SAMPLE_DRAWS)
        variance_error: Float64[Array, "..."] = jnp.sqrt(
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

        The case requires bitwise equality and an integer-output gradient
        error.

        Notes
        -----
        The test calls the public sampler twice with one key. It then asks JAX
        for an unsupported gradient of the integer sum.
        """
        rates: Float64[Array, "3"] = jnp.array([0.2, 0.3, 0.5])
        key: Float64[Array, "..."] = jax.random.key(881)
        first: Float64[Array, "..."] = sample_poisson_counts(key, rates)
        second: Float64[Array, "..."] = sample_poisson_counts(key, rates)

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

    The class owns multinomial moments, exact totals, replay, and gradient
    scope.
    """

    @pytest.mark.big_mem
    @pytest.mark.rss_limit_mb(900)
    def test_moments_stay_within_five_standard_errors(self) -> None:
        """Match multinomial means and full covariance.

        The fixed-seed check uses 200,000 draws with total 100.

        Notes
        -----
        The test derives covariance standard errors from exact categorical
        fourth moments and checks all nine covariance entries.
        """
        total_count: int = 100
        probabilities: Float64[Array, "3"] = jnp.array([0.2, 0.3, 0.5])
        keys: Float64[Array, "..."] = jax.random.split(
            jax.random.key(8202), _SAMPLE_DRAWS
        )
        draws: Float64[Array, "..."] = jax.jit(
            jax.vmap(sample_fixed_total_counts, in_axes=(0, None, None)),
            static_argnums=2,
        )(keys, probabilities, total_count)
        totals: Float64[Array, "..."] = jnp.sum(draws, axis=1)
        event_covariance: Float64[Array, "..."] = jnp.diag(
            probabilities
        ) - jnp.outer(probabilities, probabilities)
        covariance: Float64[Array, "..."] = total_count * event_covariance
        expected_mean: Float64[Array, "..."] = total_count * probabilities
        centred: Float64[Array, "..."] = draws - expected_mean
        empirical_mean: Float64[Array, "..."] = jnp.mean(draws, axis=0)
        empirical_covariance: Float64[Array, "..."] = (
            jnp.einsum("ni,nj->ij", centred, centred) / _SAMPLE_DRAWS
        )
        one_hot: Float64[Array, "..."] = jnp.eye(probabilities.size)
        centred_event: Float64[Array, "..."] = one_hot - probabilities[None, :]
        event_fourth: Float64[Array, "..."] = jnp.einsum(
            "k,ki,ki,kj,kj->ij",
            probabilities,
            centred_event,
            centred_event,
            centred_event,
            centred_event,
        )
        event_variance: Float64[Array, "..."] = probabilities * (
            1.0 - probabilities
        )
        count_fourth: Float64[Array, "..."] = (
            total_count * event_fourth
            + total_count
            * (total_count - 1)
            * (
                jnp.outer(event_variance, event_variance)
                + 2.0 * jnp.square(event_covariance)
            )
        )
        mean_error: Float64[Array, "..."] = jnp.sqrt(
            jnp.diag(covariance) / _SAMPLE_DRAWS
        )
        covariance_error: Float64[Array, "..."] = jnp.sqrt(
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
        The test compares two fixed-key draws bitwise and checks their dtype
        and
        sum. It then requests an unsupported gradient of the integer sum.
        """
        rates: Float64[Array, "3"] = jnp.array([0.2, 0.3, 0.5])
        key: Float64[Array, "..."] = jax.random.key(881)
        first: Float64[Array, "..."] = sample_fixed_total_counts(
            key, rates, 113
        )
        second: Float64[Array, "..."] = sample_fixed_total_counts(
            key, rates, 113
        )

        chex.assert_trees_all_equal(first, second)
        assert int(jnp.sum(first)) == 113
        assert jnp.issubdtype(first.dtype, jnp.integer)
        with pytest.raises(TypeError, match="real-valued outputs"):
            jax.grad(
                lambda candidate: jnp.sum(
                    sample_fixed_total_counts(key, candidate, 113)
                )
            )(rates)
