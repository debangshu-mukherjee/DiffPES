"""Validate JAX-native certification dependency analysis.

The tests cover public behavior, differentiability, validation, and stable
scientific identity in the supported certification regime.
"""

import jax
import jax.numpy as jnp
from beartype.typing import Any, Tuple

from diffpes.certify import (
    clear_dependency_cache,
    dependency_cache_info,
    dependency_map,
    information_spectrum,
    linearized_forward,
    sensitivity_map,
)


class TestDependencyMap:
    """Verify :func:`~diffpes.certify.dependency_map`.

    The cases cover the public behavior in the supported certification regime.

    :see: :func:`~diffpes.certify.dependency_map`
    """

    def test_dependency_map_distinguishes_disconnected_leaf(self) -> None:
        """Trace only leaves consumed by the output JAXPR.

        The case uses explicit inputs in the supported certification regime.
        It checks the public result or the documented failure state.

        Notes
        -----
        The test compares the result with explicit numerical or structural assertions.
        """
        inputs: Any
        result: Any

        def forward(inputs: Any) -> Any:
            x: Any
            unused: Any
            x, unused = inputs
            del unused
            return x**2

        inputs = (jnp.array([2.0]), jnp.array([7.0]))
        result = dependency_map("org.diffpes.model.test", forward, inputs)
        assert result.structural.shape == (1, 2)
        assert result.structural.tolist() == [[True, False]]
        assert result.traced.tolist() == [[True, False]]


class TestLinearizedForward:
    """Verify :func:`~diffpes.certify.linearized_forward`.

    The cases cover the public behavior in the supported certification regime.

    :see: :func:`~diffpes.certify.linearized_forward`
    """

    def test_linearization_reuses_exact_jvp(self) -> None:
        """Retain the linear map of a cubic function.

        The case uses explicit inputs in the supported certification regime.
        It checks the public result or the documented failure state.

        Notes
        -----
        The test compares the result with explicit numerical or structural assertions.
        """
        value: Any
        pushforward: Any
        value, pushforward = linearized_forward(
            lambda x: x**3, jnp.array([2.0])
        )
        assert jnp.allclose(value, 8.0)
        assert jnp.allclose(pushforward(jnp.ones(1)), 12.0)


class TestInformationSpectrum:
    """Verify :func:`~diffpes.certify.information_spectrum`.

    The cases cover the public behavior in the supported certification regime.

    :see: :func:`~diffpes.certify.information_spectrum`
    """

    def test_information_spectrum_and_gradient(self) -> None:
        """Recover and differentiate the singular value of x squared.

        The case uses explicit inputs in the supported certification regime.
        It checks the public result or the documented failure state.

        Notes
        -----
        The test compares the result with explicit numerical or structural assertions.
        """
        x: Any

        def singular(x: Any) -> Any:
            spectrum: Any
            spectrum = information_spectrum(
                lambda candidate: candidate**2,
                x,
                rank=1,
                iterations=3,
            )
            return spectrum.singular_values[0]

        x = jnp.array([2.0])
        assert jnp.allclose(singular(x), 4.0, rtol=1e-12)
        assert jnp.allclose(jax.grad(singular)(x), 2.0, rtol=1e-9)

    def test_zero_rank_uses_finite_condition_sentinel(self) -> None:
        """Represent a zero-information model with a finite condition value.

        Zero marks the absence of an active information direction.

        Notes
        -----
        The constant model has a zero Jacobian at every input value.
        """

        def constant(value: Any) -> Any:
            constant_value: Any = jnp.zeros_like(value)
            return constant_value

        result: Any = information_spectrum(
            constant,
            jnp.array([2.0]),
            rank=1,
            iterations=3,
        )
        assert int(result.effective_rank) == 0
        assert float(result.condition_estimate) == 0.0
        assert bool(jnp.isfinite(result.condition_estimate))


class TestSensitivityMap:
    """Verify :func:`~diffpes.certify.sensitivity_map`.

    The cases cover the public behavior in the supported certification regime.

    :see: :func:`~diffpes.certify.sensitivity_map`
    """

    def test_sensitivity_map_has_output_by_input_orientation(self) -> None:
        """Store output projections along rows and input probes in columns.

        The case uses explicit inputs in the supported certification regime.
        It checks the public result or the documented failure state.

        Notes
        -----
        The test compares the result with explicit numerical or structural assertions.
        """
        inputs: Any
        directions: Any
        result: Any
        inputs = jnp.array([2.0])
        directions = jnp.ones((1, 1))
        result = sensitivity_map(
            ("$[0]",),
            ("square",),
            lambda x: x**2,
            inputs,
            directions,
            jnp.ones(1),
        )
        assert result.sensitivities.shape == (1, 1)
        assert jnp.allclose(result.sensitivities, 4.0)


class TestClearDependencyCache:
    """Verify :func:`~diffpes.certify.clear_dependency_cache`.

    The case clears all structural entries and all counters.

    :see: :func:`~diffpes.certify.clear_dependency_cache`
    """

    def test_clear_removes_entries_and_counters(self) -> None:
        """Clear the cache after one structural dependency analysis.

        The cache must report no entries, hits, or misses after the clear.

        Notes
        -----
        The test computes one linear dependency map before it clears the cache.
        """

        def forward(value: Any) -> Any:
            result: Any = 2.0 * value
            return result

        dependency_map(
            "org.diffpes.model.cache.clear",
            forward,
            jnp.array([1.0]),
        )
        clear_dependency_cache()
        info: Tuple[int, int, int] = dependency_cache_info()
        assert info == (0, 0, 0)


class TestDependencyCacheInfo:
    """Verify :func:`~diffpes.certify.dependency_cache_info`.

    The case counts one cache hit for one repeated static model shape.

    :see: :func:`~diffpes.certify.dependency_cache_info`
    """

    def test_static_model_shape_has_one_structural_miss(self) -> None:
        """Verify one structural miss across two map evaluations.

        The repeated model ID, callable, shape, and dtype use one cache entry.

        Notes
        -----
        The test clears the cache and calls the same linear model two times.
        """

        def forward(value: Any) -> Any:
            result: Any = 2.0 * value
            return result

        clear_dependency_cache()
        inputs: Any = jnp.array([1.0, 2.0])
        dependency_map("org.diffpes.model.cache", forward, inputs)
        dependency_map("org.diffpes.model.cache", forward, inputs)
        info: Tuple[int, int, int] = dependency_cache_info()
        assert info == (1, 1, 1)
