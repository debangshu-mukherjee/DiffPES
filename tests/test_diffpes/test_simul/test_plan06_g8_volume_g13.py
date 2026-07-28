"""Certify the conservative G8 volume witness and G13 lifecycle.

Extended Summary
----------------
An independent three-dimensional Cartesian product quadrature integrates a
normalized mixed-parity orbital against a plane wave and the length-gauge
dipole operator. A frozen artifact summary then proves that the optional
Hermite accelerator rejects and leaves D13 inactive.
"""

import json
import math
from pathlib import Path

import chex
import jax.numpy as jnp
import numpy as np
import pytest
from jaxtyping import Complex
from scipy.special import gamma

from diffpes.radial import radial_bvals
from diffpes.simul import contract_polarization, orbital_transition_channels
from diffpes.types import (
    FinalStateSpec,
    MatrixElementParams,
    OrbitalBasis,
    RadialQuadratureSpec,
    RadialSpec,
    make_final_state_spec,
    make_matrix_element_params,
    make_orbital_basis,
    make_radial_quadrature_spec,
    make_radial_spec,
)

_SPATIAL_PLANE_WAVE_FACTOR: float = math.sqrt(3.0 / (4.0 * math.pi)) / (
    4.0 * math.pi
)
_SPATIAL_AMPLITUDE_REFERENCE: complex = 3.9264107296525 - 2.7583112509178j


def _normalized_sto(
    principal: int,
    zeta: float,
    radius: np.ndarray,
) -> np.ndarray:
    """Evaluate one independently normalized integer-n Slater row."""
    normalization: float = math.sqrt(
        (2.0 * zeta) ** (2 * principal + 1) / gamma(2 * principal + 1)
    )
    values: np.ndarray = (
        normalization * radius ** (principal - 1) * np.exp(-zeta * radius)
    )
    return values


def _spatial_volume_amplitude(
    momentum_bohr_inv: float,
    direction_cart: np.ndarray,
    polarization_cart: np.ndarray,
    coefficients: np.ndarray,
) -> tuple[complex, float]:
    """Integrate the normalized mixed s-plus-pz orbital in Cartesian space."""
    radial_abscissa: np.ndarray
    radial_weights_raw: np.ndarray
    radial_abscissa, radial_weights_raw = np.polynomial.legendre.leggauss(400)
    radius: np.ndarray = 25.0 * (radial_abscissa + 1.0)
    radial_weights: np.ndarray = 25.0 * radial_weights_raw
    cosine: np.ndarray
    cosine_weights: np.ndarray
    cosine, cosine_weights = np.polynomial.legendre.leggauss(72)
    azimuth: np.ndarray = (
        2.0 * np.pi * np.arange(144, dtype=np.float64) / 144.0
    )
    azimuth_weight: float = 2.0 * np.pi / 144.0
    sine: np.ndarray = np.sqrt(1.0 - cosine[:, None] ** 2)
    x_direction: np.ndarray = sine * np.cos(azimuth[None, :])
    y_direction: np.ndarray = sine * np.sin(azimuth[None, :])
    z_direction: np.ndarray = np.broadcast_to(
        cosine[:, None],
        x_direction.shape,
    )
    direction: np.ndarray = np.asarray(direction_cart, dtype=np.float64)
    direction = direction / np.linalg.norm(direction)
    momentum_projection: np.ndarray = (
        direction[0] * x_direction
        + direction[1] * y_direction
        + direction[2] * z_direction
    )
    polarization_projection: np.ndarray = (
        polarization_cart[0] * x_direction
        + polarization_cart[1] * y_direction
        + polarization_cart[2] * z_direction
    )
    y00: np.ndarray = np.full_like(
        x_direction,
        1.0 / math.sqrt(4.0 * math.pi),
    )
    y10: np.ndarray = math.sqrt(3.0 / (4.0 * math.pi)) * z_direction
    radial_s: np.ndarray = _normalized_sto(1, 1.1, radius)
    radial_p: np.ndarray = _normalized_sto(2, 0.9, radius)
    angular_weights: np.ndarray = cosine_weights[:, None] * azimuth_weight
    amplitude: complex = 0.0j
    norm: float = 0.0
    radial_point: np.float64
    radial_weight: np.float64
    s_value: np.float64
    p_value: np.float64
    for radial_point, radial_weight, s_value, p_value in zip(
        radius,
        radial_weights,
        radial_s,
        radial_p,
        strict=True,
    ):
        wavefunction: np.ndarray = (
            coefficients[0] * s_value * y00 + coefficients[1] * p_value * y10
        )
        plane_wave: np.ndarray = np.exp(
            1j * momentum_bohr_inv * radial_point * momentum_projection
        )
        amplitude += complex(
            radial_weight
            * radial_point**3
            * np.sum(
                angular_weights
                * plane_wave
                * polarization_projection
                * wavefunction
            )
        )
        norm += float(
            radial_weight
            * radial_point**2
            * np.sum(angular_weights * np.abs(wavefunction) ** 2)
        )
    return amplitude, norm


def _production_amplitude(
    radial_values: np.ndarray,
    direction_cart: np.ndarray,
    polarization_cart: np.ndarray,
    coefficients: np.ndarray,
    basis: OrbitalBasis,
    params: MatrixElementParams,
) -> complex:
    """Assemble the production mixed-parity amplitude from supplied branches."""
    channels: Complex[jnp.ndarray, "1 1 2 3"] = orbital_transition_channels(
        jnp.zeros((1, 3)),
        jnp.asarray(direction_cart[None, :]),
        jnp.zeros((2, 3)),
        jnp.zeros((2,)),
        jnp.asarray(radial_values[None, :, :]),
        params,
        jnp.asarray(9.0),
        basis,
    )
    polarized: Complex[jnp.ndarray, " 2"] = contract_polarization(
        channels[0, 0],
        jnp.asarray(polarization_cart),
    )
    amplitude: complex = complex(
        jnp.sum(polarized * jnp.asarray(coefficients))
    )
    return amplitude


def test_g8_full_cartesian_volume_matches_production() -> None:
    """Match a normalized full-volume mixed-parity plane-wave amplitude.

    The independent oracle integrates radius and two Cartesian direction
    coordinates without Gaunt tables or a spherical-wave decomposition.

    Notes
    -----
    Compare the declared normalization and reject three partial-wave phases.
    """
    basis: OrbitalBasis = make_orbital_basis(
        atom_indices=(0, 0),
        n=(1, 2),
        l=(0, 1),
        m=(0, 0),
    )
    radial: RadialSpec = make_radial_spec(
        basis,
        (0, 1),
        zeta_shell=jnp.asarray([[1.1], [0.9]]),
        coefficients_shell=jnp.ones((2, 1)),
    )
    params: MatrixElementParams = make_matrix_element_params(basis, (0, 1))
    quadrature: RadialQuadratureSpec = make_radial_quadrature_spec()
    final_state: FinalStateSpec = make_final_state_spec()
    momentum: float = 0.73
    direction: np.ndarray = np.asarray([0.41, -0.36, 0.84])
    direction = direction / np.linalg.norm(direction)
    polarization: np.ndarray = np.asarray(
        [0.31 + 0.27j, -0.42 + 0.19j, 0.53 - 0.11j],
        dtype=np.complex128,
    )
    coefficients: np.ndarray = np.asarray(
        [0.8 + 0.2j, -0.35 + 0.6j],
        dtype=np.complex128,
    )
    coefficients = coefficients / np.linalg.norm(coefficients)
    spatial_amplitude: complex
    spatial_norm: float
    spatial_amplitude, spatial_norm = _spatial_volume_amplitude(
        momentum,
        direction,
        polarization,
        coefficients,
    )
    np.testing.assert_allclose(
        spatial_amplitude,
        _SPATIAL_AMPLITUDE_REFERENCE,
        rtol=1.0e-12,
        atol=1.0e-12,
    )
    expected: complex = _SPATIAL_PLANE_WAVE_FACTOR * spatial_amplitude
    correct_radial: np.ndarray = np.asarray(
        radial_bvals(
            radial,
            jnp.asarray([momentum]),
            quadrature,
            final_state,
        )[0]
    )
    actual: complex = _production_amplitude(
        correct_radial,
        direction,
        polarization,
        coefficients,
        basis,
        params,
    )
    np.testing.assert_allclose(spatial_norm, 1.0, rtol=0.0, atol=1.0e-12)
    np.testing.assert_allclose(actual, expected, rtol=1.0e-12, atol=1.0e-13)

    final_degrees: tuple[tuple[int, int], ...] = ((-1, 1), (0, 2))
    phase_free: np.ndarray = np.zeros_like(correct_radial)
    orbital: int
    degrees: tuple[int, int]
    branch: int
    final_degree: int
    for orbital, degrees in enumerate(final_degrees):
        for branch, final_degree in enumerate(degrees):
            if final_degree >= 0:
                phase_free[orbital, branch] = correct_radial[
                    orbital, branch
                ] / (1j**final_degree)
    controls: dict[str, np.ndarray] = {
        "omitted": phase_free,
        "flipped": np.asarray(
            [
                [
                    0.0,
                    phase_free[0, 1] * (-1j),
                ],
                [
                    phase_free[1, 0],
                    phase_free[1, 1] * (-1j) ** 2,
                ],
            ]
        ),
        "doubled": np.asarray(
            [
                [
                    0.0,
                    phase_free[0, 1] * (1j**1) ** 2,
                ],
                [
                    phase_free[1, 0],
                    phase_free[1, 1] * (1j**2) ** 2,
                ],
            ]
        ),
    }
    name: str
    planted: np.ndarray
    for name, planted in controls.items():
        control: complex = _production_amplitude(
            planted,
            direction,
            polarization,
            coefficients,
            basis,
            params,
        )
        assert abs(control - expected) > 1.0e-3, name


def test_g13_rejection_makes_d13_inactive() -> None:
    """Bind the decisive G13 artifact to runtime rejection and D13 status.

    The frozen 1025-to-2049 witness exceeds the preregistered half-budget
    threshold, so no Hermite resolution exists for derivative certification.

    Notes
    -----
    Read the compact artifact summary and exercise every candidate runtime guard.
    """
    reference_path: Path = (
        Path(__file__).resolve().parents[1]
        / "_reference_data"
        / "plan06_g13_rejection.json"
    )
    evidence: dict[str, object] = json.loads(reference_path.read_text())
    decision: dict[str, object] = evidence["decision"]
    ratio: float = float(evidence["next_rung_value_budget_ratio"])
    threshold: float = float(evidence["next_rung_fraction_limit"])
    assert evidence["schema"] == "diffpes.plan06.radial-hermite-rejection.v1"
    assert ratio > threshold
    assert decision["accepted"] is False
    assert decision["selected_node_count"] is None
    assert decision["runtime"] == "reject-hermite-use-direct"
    assert (
        decision["d13_status"]
        == "not activated because no G13 resolution may ship"
    )
    node_count: int
    for node_count in evidence["candidate_node_counts"]:
        with pytest.raises(ValueError, match="failed the frozen G13"):
            make_final_state_spec(
                radial_accelerator="hermite",
                table_n_points=node_count,
            )
    direct: FinalStateSpec = make_final_state_spec()
    assert direct.radial_accelerator == "direct"
