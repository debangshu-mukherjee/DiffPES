"""Exercise metric-aware generalized spectral contracts.

Extended Summary
----------------
The manufactured one-level Green function pins the spectral sign, metric
trace, and covariant transition-source contraction independently.

Routine Listings
----------------
:class:`TestMetricAwareSpectralDensity`
    Specify the one-level metric-aware spectral formula.
"""

import jax.numpy as jnp

from diffpes.simul import (
    projected_spectral_density,
    spectral_density_matrix,
    total_spectral_density,
)
from diffpes.types import (
    make_measurement_coordinates,
    make_retarded_green_batch,
    make_spectral_evaluation_request,
)


class TestMetricAwareSpectralDensity:
    """Specify the one-level metric-aware spectral formula.

    See Also
    --------
    diffpes.simul.spectral_density_matrix
        Production spectral-matrix routine covered by this contract test.
    """

    def test_metric_trace_and_covariant_projection_match_formula(self) -> None:
        """Retain the retarded sign convention in all public contractions."""
        coordinates = make_measurement_coordinates(
            (jnp.asarray([0.0]),),
            coordinate_names=("energy",),
            coordinate_units=("eV",),
            coordinate_dimensions=(("energy",),),
            dimension_names=("energy",),
            coordinate_system="relative_energy",
            frame_id="org.diffpes.frame.test@1",
        )
        request = make_spectral_evaluation_request(
            coordinates,
            jnp.asarray([0.0]),
            jnp.asarray([20.0]),
            jnp.asarray(0.1),
            basis_ref="test-basis",
        )
        green = make_retarded_green_batch(
            jnp.asarray([[[[[0.0 - 10.0j]]]]]),
            jnp.asarray([[[1.0 + 0.0j]]]),
            request,
            basis_ref="test-basis",
            source_ref="test",
            derivative_mode="exact_ad",
            validation_ref="test",
        )
        spectral = spectral_density_matrix(green)
        total = total_spectral_density(green)
        source = jnp.asarray([[[[[1.0 + 0.0j]]]]])
        projected = projected_spectral_density(green, source)
        assert jnp.allclose(spectral, 10.0 / jnp.pi)
        assert jnp.allclose(total, 10.0 / jnp.pi)
        assert jnp.allclose(projected, 10.0 / jnp.pi)
