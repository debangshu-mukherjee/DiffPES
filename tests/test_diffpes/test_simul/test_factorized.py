"""Verify the factorized spectral-projection evaluator.

Use explicit fixtures and independent expectations for every assertion.
"""

import jax.numpy as jnp
import numpy as np
import pytest
from beartype.typing import Any, List, Tuple
from jaxtyping import Array, Complex128, Float64
from numpy.typing import NDArray

from diffpes.simul import evaluate_spectral_projection
from diffpes.types import (
    FactorizedArpesModel,
    MeasurementCoordinates,
    make_factorized_arpes_model,
    make_final_state_spec,
    make_matrix_element_params,
    make_measurement_coordinates,
    make_orbital_basis,
    make_radial_quadrature_spec,
    make_radial_spec,
    make_self_energy_model,
)

_RTOL = 1.0e-12
_ATOL = 1.0e-13
_GAMMA_EV = 0.07
_ETA_EV = 0.03


class _TwoOrbitalSource:
    """PRIVATE: Check every structural capability protocol."""

    capabilities: Tuple[str, ...]
    state_ref: str = "fixture-state"
    derivative_mode: str = "exact_ad"

    def __init__(self, capabilities: Tuple[str, ...]) -> None:
        self.capabilities = capabilities
        self.numerical_calls: int = 0

    def hamiltonian(
        self, coordinates: MeasurementCoordinates
    ) -> Complex128[Array, "n_k 2 2"]:
        """Check the private helper behavior."""
        self.numerical_calls += 1
        n_k: int = coordinates.coordinate_arrays[0].shape[0]
        matrix: Complex128[Array, "2 2"] = jnp.asarray(
            [[-0.45, 0.18], [0.18, 0.55]], dtype=jnp.complex128
        )
        result: Complex128[Array, "n_k 2 2"] = jnp.broadcast_to(
            matrix, (n_k, 2, 2)
        )
        return result

    def overlap(
        self, coordinates: MeasurementCoordinates
    ) -> Complex128[Array, "n_k 2 2"]:
        """Check the private helper behavior."""
        self.numerical_calls += 1
        n_k: int = coordinates.coordinate_arrays[0].shape[0]
        matrix: Complex128[Array, "2 2"] = jnp.asarray(
            [[1.0, 0.16], [0.16, 1.08]], dtype=jnp.complex128
        )
        result: Complex128[Array, "n_k 2 2"] = jnp.broadcast_to(
            matrix, (n_k, 2, 2)
        )
        return result


def _coordinates() -> MeasurementCoordinates:
    """PRIVATE: Check the private helper behavior.

    Notes
    -----
    Build the fixture from explicit values for isolated use.
    """
    k_points: Float64[Array, "2 3"] = jnp.asarray(
        [[0.0, 0.0, 0.0], [0.2, -0.1, 0.0]]
    )
    coordinates: MeasurementCoordinates = make_measurement_coordinates(
        (k_points,),
        coordinate_names=("k_points_frac",),
        coordinate_units=("1",),
        coordinate_dimensions=(("k", "cart"),),
        dimension_names=("k", "cart"),
        coordinate_system="fractional",
        frame_id="fixture",
    )
    return coordinates


def _model() -> FactorizedArpesModel:
    """PRIVATE: Check the private helper behavior.

    Notes
    -----
    Build the fixture from explicit values for isolated use.
    """
    basis: Any = make_orbital_basis(
        atom_indices=(0, 1),
        n=(1, 1),
        l=(0, 0),
        m=(0, 0),
        labels=("a:1s", "b:1s"),
    )
    radial: Any = make_radial_spec(
        basis,
        (0, 1),
        mode="fixed",
        fixed_integrals_shell=jnp.asarray([[0.0, 1.0], [0.0, 1.0]]),
    )
    matrix_elements: Any = make_matrix_element_params(
        basis,
        (0, 1),
        sigma_shell=jnp.asarray([1.0, 1.0]),
        phase_shift_angles_shell=jnp.asarray([0.0, 0.0]),
    )
    model: FactorizedArpesModel = make_factorized_arpes_model(
        radial,
        matrix_elements,
        make_radial_quadrature_spec(),
        make_final_state_spec(),
        make_self_energy_model(gamma=_GAMMA_EV),
        eta_ev=jnp.asarray(_ETA_EV),
    )
    return model


class TestEvaluateSpectralProjection:
    """Verify ``diffpes.simul.evaluate_spectral_projection``.

    Cover acceptance and rejection cases with explicit fixtures.
    """

    def test_matches_independent_two_orbital_numpy_reference(self) -> None:
        """Match diagonal Green projections computed directly with NumPy.
        Check explicit inputs against independent expectations.

        Notes
        -----
        Invert the two-level Dyson operator independently at every energy.
        """
        model: FactorizedArpesModel = _model()
        source: Any = _TwoOrbitalSource(("hamiltonian", "overlap"))
        omega: Float64[Array, " 3"] = jnp.asarray([-0.7, 0.0, 0.8])
        temperature: Float64[Array, " 2"] = jnp.asarray([15.0, 90.0])
        actual: Any = evaluate_spectral_projection(
            model, source, _coordinates(), omega, temperature
        )

        hamiltonian: Complex128[NDArray, "2 2"] = np.asarray(
            [[-0.45, 0.18], [0.18, 0.55]], dtype=np.complex128
        )
        overlap: Complex128[NDArray, "2 2"] = np.asarray(
            [[1.0, 0.16], [0.16, 1.08]], dtype=np.complex128
        )
        rows: List[Float64[NDArray, " 2"]] = []
        energy: float
        for energy in np.asarray(omega):
            operator: Complex128[NDArray, "2 2"] = (
                (energy + 1.0j * _ETA_EV) * overlap
                - hamiltonian
                + 1.0j * _GAMMA_EV * np.eye(2)
            )
            rows.append(-np.imag(np.diag(np.linalg.inv(operator))) / np.pi)
        energy_channel: Float64[NDArray, "3 2"] = np.asarray(rows)
        expected: Float64[NDArray, "2 2 2 3"] = np.broadcast_to(
            energy_channel.T[:, None, None, :], (2, 2, 2, 3)
        )
        assert np.allclose(
            actual.scalar_intensity_by_domain[0],
            expected,
            rtol=_RTOL,
            atol=_ATOL,
        )

    @pytest.mark.parametrize("missing", ["hamiltonian", "overlap"])
    def test_rejects_each_missing_capability_before_numerics(
        self, missing: str
    ) -> None:
        """Reject each absent capability before invoking source methods.
        Check explicit inputs against independent expectations.

        Notes
        -----
        Remove one declared capability while retaining both structural methods.
        """
        capabilities: Tuple[str, ...] = tuple(
            item for item in ("hamiltonian", "overlap") if item != missing
        )
        source: Any = _TwoOrbitalSource(capabilities)
        with pytest.raises(
            ValueError, match=f"lacks required capabilities: {missing}"
        ):
            evaluate_spectral_projection(
                _model(),
                source,
                _coordinates(),
                jnp.asarray([0.0]),
                jnp.asarray([20.0]),
            )
        assert source.numerical_calls == 0

    def test_model_method_routes_to_the_public_evaluator(self) -> None:
        """Return the same payload through the model protocol method.
        Check explicit inputs against independent expectations.

        Notes
        -----
        Evaluate identical immutable inputs through both public entry points.
        """
        model: FactorizedArpesModel = _model()
        coordinates: MeasurementCoordinates = _coordinates()
        omega: Float64[Array, " 1"] = jnp.asarray([0.1])
        temperature: Float64[Array, " 1"] = jnp.asarray([30.0])
        direct: Any = evaluate_spectral_projection(
            model,
            _TwoOrbitalSource(("hamiltonian", "overlap")),
            coordinates,
            omega,
            temperature,
        )
        routed: Any = model.evaluate(
            _TwoOrbitalSource(("hamiltonian", "overlap")),
            coordinates,
            omega,
            temperature,
        )
        assert np.array_equal(
            np.asarray(routed.scalar_intensity_by_domain[0]),
            np.asarray(direct.scalar_intensity_by_domain[0]),
        )
