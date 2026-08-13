"""Verify metric-aware generalized spectral kernels.

The tests compare two-level nonorthogonal solves with independent NumPy
linear algebra and exercise each rejected Dyson-domain boundary.
"""

import equinox as eqx
import jax.numpy as jnp
import numpy as np
import pytest
from beartype.typing import Any, List, Tuple
from jaxtyping import Array, Complex128, Float64
from numpy.typing import NDArray

from diffpes.simul import (
    projected_spectral_density,
    projected_spectral_density_solve,
    solve_retarded_dyson,
    spectral_density_matrix,
    total_spectral_density,
    total_spectral_density_solve,
)
from diffpes.types import (
    RetardedGreenBatch,
    SelfEnergyBatch,
    SpectralEvaluationRequest,
    make_measurement_coordinates,
    make_retarded_green_batch,
    make_self_energy_batch,
    make_spectral_evaluation_request,
)

_RTOL = 1.0e-12
_ATOL = 1.0e-13


def _fixture() -> Tuple[
    Complex128[Array, "1 2 2"],
    Complex128[Array, "1 2 2"],
    SelfEnergyBatch,
    SpectralEvaluationRequest,
]:
    """PRIVATE: Check the private helper behavior.

    Notes
    -----
    Build the fixture from explicit values for isolated use.
    """
    kpoints: Float64[Array, "1 3"] = jnp.asarray([[0.1, -0.2, 0.0]])
    omega: Float64[Array, " 3"] = jnp.asarray([-0.8, 0.0, 0.9])
    temperature: Float64[Array, " 1"] = jnp.asarray([25.0])
    coordinates: Any = make_measurement_coordinates(
        (kpoints, omega, temperature),
        coordinate_names=(
            "k_points_frac",
            "omega_rel_fermi_ev",
            "temperature_k",
        ),
        coordinate_units=("1", "eV", "K"),
        coordinate_dimensions=(("k", "cart"), ("omega",), ("temperature",)),
        dimension_names=("k", "cart", "omega", "temperature"),
        coordinate_system="fractional_energy_temperature",
        frame_id="fixture",
    )
    request: SpectralEvaluationRequest = make_spectral_evaluation_request(
        coordinates,
        omega,
        temperature,
        jnp.asarray(0.04),
        basis_ref="fixture-basis",
    )
    hamiltonian: Complex128[Array, "1 2 2"] = jnp.asarray(
        [[[-0.45, 0.17], [0.17, 0.62]]], dtype=jnp.complex128
    )
    overlap: Complex128[Array, "1 2 2"] = jnp.asarray(
        [[[1.0, 0.18], [0.18, 1.12]]], dtype=jnp.complex128
    )
    identity: Complex128[Array, "2 2"] = jnp.eye(2, dtype=jnp.complex128)
    values: Complex128[Array, "1 1 3 2 2"] = jnp.broadcast_to(
        -0.07j * identity,
        (1, 1, omega.shape[0], 2, 2),
    )
    self_energy: SelfEnergyBatch = make_self_energy_batch(
        values,
        request,
        basis_ref="fixture-basis",
        source_ref="fixture",
        derivative_mode="exact_ad",
    )
    result: Tuple[
        Complex128[Array, "1 2 2"],
        Complex128[Array, "1 2 2"],
        SelfEnergyBatch,
        SpectralEvaluationRequest,
    ] = (hamiltonian, overlap, self_energy, request)
    return result


def _numpy_green(
    hamiltonian: Complex128[Array, "1 2 2"],
    overlap: Complex128[Array, "1 2 2"],
    self_energy: SelfEnergyBatch,
    request: SpectralEvaluationRequest,
) -> Complex128[NDArray, "1 1 3 2 2"]:
    """PRIVATE: Check the private helper behavior.

    Notes
    -----
    Build the fixture from explicit values for isolated use.
    """
    h_np: Complex128[NDArray, "2 2"] = np.asarray(hamiltonian[0])
    s_np: Complex128[NDArray, "2 2"] = np.asarray(overlap[0])
    sigma_np: Complex128[NDArray, "3 2 2"] = np.asarray(
        self_energy.values_ev[0, 0]
    )
    omega_np: Float64[NDArray, " 3"] = np.asarray(request.omega_rel_fermi_ev)
    rows: List[Complex128[NDArray, "2 2"]] = []
    energy: float
    sigma: Complex128[NDArray, "2 2"]
    for energy, sigma in zip(omega_np, sigma_np, strict=True):
        operator: Complex128[NDArray, "2 2"] = (
            (energy + 1.0j * float(request.eta_ev)) * s_np - h_np - sigma
        )
        rows.append(np.linalg.inv(operator))
    result: Complex128[NDArray, "1 1 3 2 2"] = np.asarray(rows)[None, None]
    return result


class TestSolveRetardedDyson:
    """Verify ``diffpes.simul.solve_retarded_dyson`` against NumPy.

    Cover acceptance and rejection cases with explicit fixtures.
    """

    def test_matches_independent_nonorthogonal_inverse(self) -> None:
        """Match every two-level pole with an independently inverted operator.
        Check explicit inputs against independent expectations.

        Notes
        -----
        Build each complex matrix in NumPy and compare the complete batch.
        """
        hamiltonian: Any
        overlap: Any
        self_energy: Any
        request: Any
        hamiltonian, overlap, self_energy, request = _fixture()
        green: RetardedGreenBatch = solve_retarded_dyson(
            hamiltonian, overlap, self_energy, request
        )
        expected: Complex128[NDArray, "1 1 3 2 2"] = _numpy_green(
            hamiltonian, overlap, self_energy, request
        )
        assert np.allclose(
            green.values_per_ev, expected, rtol=_RTOL, atol=_ATOL
        )

    @pytest.mark.parametrize(
        ("hamiltonian", "overlap", "message"),
        [
            (
                jnp.asarray([[[-0.4, 0.2], [0.1, 0.6]]], dtype=jnp.complex128),
                jnp.asarray([[[1.0, 0.1], [0.1, 1.0]]], dtype=jnp.complex128),
                "Hamiltonian must be finite and Hermitian",
            ),
            (
                jnp.asarray([[[-0.4, 0.1], [0.1, 0.6]]], dtype=jnp.complex128),
                jnp.asarray([[[1.0, 2.0], [2.0, 1.0]]], dtype=jnp.complex128),
                "overlap must be finite, Hermitian, and positive definite",
            ),
        ],
    )
    def test_rejects_invalid_dyson_domains(
        self,
        hamiltonian: Complex128[Array, "1 2 2"],
        overlap: Complex128[Array, "1 2 2"],
        message: str,
    ) -> None:
        """Reject non-Hermitian or indefinite inputs through traced guards.
        Check explicit inputs against independent expectations.

        Notes
        -----
        Replace one valid fixture matrix and match its specific diagnostic.
        """
        self_energy: Any
        request: Any
        _, _, self_energy, request = _fixture()
        with pytest.raises(eqx.EquinoxRuntimeError, match=message):
            solve_retarded_dyson(hamiltonian, overlap, self_energy, request)


class TestProjectedSpectralDensitySolve:
    """Verify ``diffpes.simul.projected_spectral_density_solve`` directly.

    Cover acceptance and rejection cases with explicit fixtures.
    """

    def test_matches_numpy_transition_source_solves(self) -> None:
        """Match complex transition rows without materializing production G.
        Check explicit inputs against independent expectations.

        Notes
        -----
        Contract independent NumPy inverse matrices with fixed complex rows.
        """
        hamiltonian: Any
        overlap: Any
        self_energy: Any
        request: Any
        hamiltonian, overlap, self_energy, request = _fixture()
        rows: Complex128[Array, "1 1 3 2 2"] = jnp.broadcast_to(
            jnp.asarray([[1.0 + 0.2j, -0.3j], [0.4, 0.8 - 0.1j]])[
                None, None, None
            ],
            (1, 1, 3, 2, 2),
        )
        actual: Float64[Array, "1 1 3 2"] = projected_spectral_density_solve(
            hamiltonian, overlap, self_energy, request, rows
        )
        green_np: Complex128[NDArray, "1 1 3 2 2"] = _numpy_green(
            hamiltonian, overlap, self_energy, request
        )
        rows_np: Complex128[NDArray, "1 1 3 2 2"] = np.asarray(rows)
        expected: Float64[NDArray, "1 1 3 2"] = (
            -np.imag(
                np.einsum(
                    "tkwai,tkwij,tkwaj->tkwa",
                    rows_np.conj(),
                    green_np,
                    rows_np,
                )
            )
            / np.pi
        )
        assert np.allclose(actual, expected, rtol=_RTOL, atol=_ATOL)


class TestTotalSpectralDensitySolve:
    """Verify ``diffpes.simul.total_spectral_density_solve`` directly.

    Cover acceptance and rejection cases with explicit fixtures.
    """

    def test_matches_numpy_metric_trace(self) -> None:
        """Match the nonorthogonal spectral trace from independent inverses.
        Check explicit inputs against independent expectations.

        Notes
        -----
        Evaluate minus imaginary trace of the NumPy Green-metric product.
        """
        hamiltonian: Any
        overlap: Any
        self_energy: Any
        request: Any
        hamiltonian, overlap, self_energy, request = _fixture()
        actual: Float64[Array, "1 1 3"] = total_spectral_density_solve(
            hamiltonian, overlap, self_energy, request
        )
        green_np: Complex128[NDArray, "1 1 3 2 2"] = _numpy_green(
            hamiltonian, overlap, self_energy, request
        )
        expected: Float64[NDArray, "1 1 3"] = (
            -np.imag(
                np.einsum("tkwij,kji->tkw", green_np, np.asarray(overlap))
            )
            / np.pi
        )
        assert np.allclose(actual, expected, rtol=_RTOL, atol=_ATOL)


class TestSpectralDensityMatrix:
    """Verify ``diffpes.simul.spectral_density_matrix`` sign conventions.

    Cover acceptance and rejection cases with explicit fixtures.
    """

    def test_returns_hermitian_positive_one_level_density(self) -> None:
        """Recover ten over pi from a Green value of minus ten i.
        Check explicit inputs against independent expectations.

        Notes
        -----
        Compare the result with the scalar retarded spectral formula.
        """
        coordinates: Any = make_measurement_coordinates(
            (jnp.asarray([0.0]),),
            coordinate_names=("energy",),
            coordinate_units=("eV",),
            coordinate_dimensions=(("energy",),),
            dimension_names=("energy",),
            coordinate_system="relative_energy",
            frame_id="fixture",
        )
        request: Any = make_spectral_evaluation_request(
            coordinates,
            jnp.asarray([0.0]),
            jnp.asarray([20.0]),
            jnp.asarray(0.1),
            basis_ref="fixture-basis",
        )
        green: Any = make_retarded_green_batch(
            jnp.asarray([[[[[0.0 - 10.0j]]]]]),
            jnp.asarray([[[1.0 + 0.0j]]]),
            request,
            basis_ref="fixture-basis",
            source_ref="fixture",
            derivative_mode="exact_ad",
            validation_ref="fixture",
        )
        spectral: Complex128[Array, "1 1 1 1 1"] = spectral_density_matrix(
            green
        )
        assert jnp.allclose(spectral, 10.0 / jnp.pi)


class TestTotalSpectralDensity:
    """Verify ``diffpes.simul.total_spectral_density`` metric contraction.

    Cover acceptance and rejection cases with explicit fixtures.
    """

    def test_matches_direct_solve_for_the_two_level_fixture(self) -> None:
        """Match materialized-G and direct-solve metric traces.
        Check explicit inputs against independent expectations.

        Notes
        -----
        Compare two public paths on the same validated two-level fixture.
        """
        hamiltonian: Any
        overlap: Any
        self_energy: Any
        request: Any
        hamiltonian, overlap, self_energy, request = _fixture()
        green: Any = solve_retarded_dyson(
            hamiltonian, overlap, self_energy, request
        )
        assert jnp.allclose(
            total_spectral_density(green),
            total_spectral_density_solve(
                hamiltonian, overlap, self_energy, request
            ),
            rtol=_RTOL,
            atol=_ATOL,
        )


class TestProjectedSpectralDensity:
    """Verify ``diffpes.simul.projected_spectral_density`` contractions.

    Cover acceptance and rejection cases with explicit fixtures.
    """

    def test_matches_direct_solve_for_fixed_transition_rows(self) -> None:
        """Match materialized-G and direct-solve projected densities.
        Check explicit inputs against independent expectations.

        Notes
        -----
        Compare both paths with identical covariant transition-source rows.
        """
        hamiltonian: Any
        overlap: Any
        self_energy: Any
        request: Any
        hamiltonian, overlap, self_energy, request = _fixture()
        rows: Complex128[Array, "1 1 3 1 2"] = jnp.broadcast_to(
            jnp.asarray([1.0 + 0.1j, 0.4 - 0.2j])[None, None, None, None],
            (1, 1, 3, 1, 2),
        )
        green: Any = solve_retarded_dyson(
            hamiltonian, overlap, self_energy, request
        )
        assert jnp.allclose(
            projected_spectral_density(green, rows),
            projected_spectral_density_solve(
                hamiltonian, overlap, self_energy, request, rows
            ),
            rtol=_RTOL,
            atol=_ATOL,
        )
