"""Validate the transmission module.

The cases use analytic values, invariants, and finite differences.
"""

import chex
import jax.numpy as jnp
import numpy as np
import pytest
from jaxtyping import Array, Float64
from numpy.typing import NDArray

from diffpes.simul import (
    apply_transmission,
    transmission_shape,
)
from diffpes.types import (
    DetectorCalibration,
)
from tests._assertions import assert_rejects
from tests._gradients import assert_gradients_match_finite_differences

from ._effects_helpers import (
    _reference_integrated_bernstein,
    _resolution_calibration,
)


class TestTransmissionShape:
    """Verify :func:`diffpes.simul.transmission_shape`.

    The class owns fixed-domain calibration and shape derivatives.
    """

    @pytest.mark.parametrize("sign", [-1, 1])
    @pytest.mark.parametrize("n_slopes", [2, 3])
    def test_positive_monotone_fixed_domain_mean(
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
        raw: Float64[Array, "..."] = jnp.linspace(-0.4, 0.25, n_slopes)
        nodes128: Float64[NDArray, " Q"]
        weights128: Float64[NDArray, " Q"]
        nodes128, weights128 = np.polynomial.legendre.leggauss(128)
        energies: Float64[Array, "..."] = 30.0 + 18.0 * jnp.asarray(nodes128)
        transmission: Float64[Array, "..."] = transmission_shape(
            energies, raw, calibration
        )
        weighted_mean: Float64[Array, "..."] = 0.5 * jnp.sum(
            jnp.asarray(weights128) * transmission
        )
        differences: Float64[Array, "..."] = jnp.diff(transmission)

        assert bool(jnp.all(transmission > 0.0))
        assert bool(jnp.all(sign * differences > 0.0))
        np.testing.assert_allclose(
            weighted_mean, 1.0, rtol=1.0e-12, atol=1.0e-14
        )

    def test_crop_and_padding_invariance_is_bitwise(self) -> None:
        """Keep retained transmission bins bitwise identical across windows.

        The case pins normalization to the fixed calibration domain.

        Notes
        -----
        The test evaluates one full query and requires its retained slice to
        equal an independently cropped query bitwise.
        """
        calibration: DetectorCalibration = _resolution_calibration()
        raw: Float64[Array, "3"] = jnp.array([-0.4, 0.2, -0.1])
        full_energy: Float64[Array, "13"] = jnp.linspace(12.0, 48.0, 13)
        full: Float64[Array, "..."] = transmission_shape(
            full_energy, raw, calibration
        )
        cropped: Float64[Array, "..."] = transmission_shape(
            full_energy[3:10], raw, calibration
        )

        np.testing.assert_array_equal(cropped, full[3:10])

    def test_matches_independent_integrated_bernstein_reference(
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
        raw_np: Float64[NDArray, " q"] = np.array([-0.3, 0.15, 0.4])
        query_np: Float64[NDArray, " E"] = np.array([12.0, 18.5, 31.0, 48.0])
        normalized_query: Float64[NDArray, " E"] = (query_np - 12.0) / 36.0
        slopes: Float64[NDArray, " q"] = np.logaddexp(0.0, raw_np)
        log_query: Float64[NDArray, " E"] = -np.sum(
            _reference_integrated_bernstein(normalized_query, 2) * slopes,
            axis=-1,
        )
        nodes: Float64[NDArray, " Q"]
        weights: Float64[NDArray, " Q"]
        nodes, weights = np.polynomial.legendre.leggauss(128)
        normalized_nodes: Float64[NDArray, " Q"] = 0.5 * (nodes + 1.0)
        log_nodes: Float64[NDArray, " Q"] = -np.sum(
            _reference_integrated_bernstein(normalized_nodes, 2) * slopes,
            axis=-1,
        )
        denominator128: float = float(
            0.5 * np.sum(weights * np.exp(log_nodes))
        )
        desired: Float64[NDArray, " E"] = np.exp(log_query) / denominator128
        actual: Float64[Array, "..."] = transmission_shape(
            jnp.asarray(query_np), jnp.asarray(raw_np), calibration
        )

        np.testing.assert_allclose(actual, desired, rtol=1.0e-12, atol=1.0e-14)

    def test_rejects_extrapolation_eager_and_jit(self) -> None:
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
    def test_every_shape_coefficient_matches_fd_and_is_nonzero(
        self, n_slopes: int
    ) -> None:
        """Check each raw-slope derivative with the shared f64 FD ladder.

        The case verifies all supported transmission-shape coordinates.

        Notes
        -----
        The shared gradient check compares every coordinate with finite
        differences and requires a nonzero smooth derivative.
        """
        calibration: DetectorCalibration = _resolution_calibration(sign=1)
        raw: Float64[Array, "..."] = jnp.linspace(-0.35, 0.28, n_slopes)
        energy: Float64[Array, "5"] = jnp.array([13.0, 19.0, 28.0, 39.0, 47.0])
        weights: Float64[Array, "5"] = jnp.array([0.8, -0.2, 0.5, -0.7, 1.1])

        def loss(candidate: Float64[Array, "..."]) -> Float64[Array, "..."]:
            returned: Float64[Array, "..."] = jnp.sum(
                transmission_shape(energy, candidate, calibration) * weights
            )
            return returned

        assert_gradients_match_finite_differences(
            loss, raw, regime="smooth", elementwise=True
        )

    def test_energy_gradients_match_fd(self) -> None:
        """Check transmission derivatives for every query energy.

        The case verifies the continuous kinetic-energy dependence directly.

        Notes
        -----
        The shared gradient check compares every energy coordinate with its
        finite-difference estimate.
        """
        calibration: DetectorCalibration = _resolution_calibration(sign=-1)
        raw: Float64[Array, "3"] = jnp.array([-0.3, 0.15, 0.4])
        energy: Float64[Array, "4"] = jnp.array([13.0, 20.0, 31.0, 45.0])
        weights: Float64[Array, "4"] = jnp.array([0.7, -0.4, 1.1, 0.3])

        def loss(candidate: Float64[Array, "..."]) -> Float64[Array, "..."]:
            returned: Float64[Array, "..."] = jnp.sum(
                transmission_shape(candidate, raw, calibration) * weights
            )
            return returned

        assert_gradients_match_finite_differences(
            loss, energy, regime="smooth", elementwise=True
        )


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
        energy: Float64[Array, "4"] = jnp.array([14.0, 25.0, 37.0, 46.0])
        raw: Float64[Array, "2"] = jnp.array([-0.4, 0.2])
        intensity: Float64[Array, "..."] = (
            jnp.arange(24.0).reshape((2, 3, 4)) + 0.2
        )
        shape: Float64[Array, "..."] = transmission_shape(
            energy, raw, calibration
        )
        actual: Float64[Array, "..."] = apply_transmission(
            intensity, energy, raw, calibration
        )

        chex.assert_trees_all_equal(actual, intensity * shape)
