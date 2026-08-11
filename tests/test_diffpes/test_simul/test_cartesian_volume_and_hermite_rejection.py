"""Certify the Cartesian volume witness and Hermite rejection lifecycle.

Extended Summary
----------------
An independent three-dimensional Cartesian product quadrature integrates a
normalized mixed-parity orbital against a plane wave and the length-gauge
dipole operator. A frozen artifact summary then proves rejection of the
optional Hermite accelerator and inactivity of its derivative path.
"""

import json
import math
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest
from beartype.typing import Dict, Tuple
from jaxtyping import Array, Complex128, Float64
from numpy.typing import NDArray
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
    radius: Float64[NDArray, " n_r"],
) -> Float64[NDArray, " n_r"]:
    """PRIVATE: Evaluate one independently normalized integer-n Slater row.

    Parameters
    ----------
    principal : int
        Principal quantum number n.
    zeta : float
        Slater exponent in 1/Bohr.
    radius : Float64[NDArray, " n_r"]
        Radial samples in Bohr.

    Returns
    -------
    values : Float64[NDArray, " n_r"]
        Normalized Slater radial function samples.

    Notes
    -----
    Computes the closed-form normalization sqrt((2 zeta)**(2n + 1) /
    (2n)!) through the gamma function and multiplies r**(n - 1) times
    exp(-zeta r).
    """
    normalization: float = math.sqrt(
        (2.0 * zeta) ** (2 * principal + 1) / gamma(2 * principal + 1)
    )
    values: Float64[NDArray, " n_r"] = (
        normalization * radius ** (principal - 1) * np.exp(-zeta * radius)
    )
    return values


def _spatial_volume_amplitude(
    momentum_bohr_inv: float,
    direction_cart: Float64[NDArray, " 3"],
    polarization_cart: Complex128[NDArray, " 3"],
    coefficients: Complex128[NDArray, " 2"],
) -> Tuple[complex, float]:
    """PRIVATE: Integrate the mixed s-plus-pz orbital over the full volume.

    Parameters
    ----------
    momentum_bohr_inv : float
        Plane-wave momentum magnitude in 1/Bohr.
    direction_cart : Float64[NDArray, " 3"]
        Cartesian emission direction; normalized inside.
    polarization_cart : Complex128[NDArray, " 3"]
        Cartesian complex polarization.
    coefficients : Complex128[NDArray, " 2"]
        Mixing coefficients of the s and p_z components.

    Returns
    -------
    amplitude : complex
        Length-gauge plane-wave dipole amplitude of the mixed orbital.
    norm : float
        Squared norm of the mixed orbital under the same quadrature.

    Implementation Logic
    --------------------
    Builds a product quadrature from 400 radial nodes on [0, 50] Bohr.
    Adds 72 Gauss-Legendre polar-cosine nodes and 144 uniform azimuth
    nodes. The orbital combines the
    normalized n=1, zeta=1.1 s row and the n=2, zeta=0.9 p row with
    the real harmonics. Each radial shell accumulates the plane-wave
    phase, polarization projection, and orbital product. It uses volume
    factor r**3 for the dipole radius and r**2 for the norm.
    """
    radial_abscissa: Float64[NDArray, " n_r"]
    radial_weights_raw: Float64[NDArray, " n_r"]
    radial_abscissa, radial_weights_raw = np.polynomial.legendre.leggauss(400)
    radius: Float64[NDArray, " n_r"] = 25.0 * (radial_abscissa + 1.0)
    radial_weights: Float64[NDArray, " n_r"] = 25.0 * radial_weights_raw
    cosine: Float64[NDArray, " n_cos"]
    cosine_weights: Float64[NDArray, " n_cos"]
    cosine, cosine_weights = np.polynomial.legendre.leggauss(72)
    azimuth: Float64[NDArray, " n_phi"] = (
        2.0 * np.pi * np.arange(144, dtype=np.float64) / 144.0
    )
    azimuth_weight: float = 2.0 * np.pi / 144.0
    sine: Float64[NDArray, "n_cos 1"] = np.sqrt(1.0 - cosine[:, None] ** 2)
    x_direction: Float64[NDArray, "n_cos n_phi"] = sine * np.cos(
        azimuth[None, :]
    )
    y_direction: Float64[NDArray, "n_cos n_phi"] = sine * np.sin(
        azimuth[None, :]
    )
    z_direction: Float64[NDArray, "n_cos n_phi"] = np.broadcast_to(
        cosine[:, None],
        x_direction.shape,
    )
    direction: Float64[NDArray, " 3"] = np.asarray(
        direction_cart, dtype=np.float64
    )
    direction = direction / np.linalg.norm(direction)
    momentum_projection: Float64[NDArray, "n_cos n_phi"] = (
        direction[0] * x_direction
        + direction[1] * y_direction
        + direction[2] * z_direction
    )
    polarization_projection: Complex128[NDArray, "n_cos n_phi"] = (
        polarization_cart[0] * x_direction
        + polarization_cart[1] * y_direction
        + polarization_cart[2] * z_direction
    )
    y00: Float64[NDArray, "n_cos n_phi"] = np.full_like(
        x_direction,
        1.0 / math.sqrt(4.0 * math.pi),
    )
    y10: Float64[NDArray, "n_cos n_phi"] = (
        math.sqrt(3.0 / (4.0 * math.pi)) * z_direction
    )
    radial_s: Float64[NDArray, " n_r"] = _normalized_sto(1, 1.1, radius)
    radial_p: Float64[NDArray, " n_r"] = _normalized_sto(2, 0.9, radius)
    angular_weights: Float64[NDArray, "n_cos 1"] = (
        cosine_weights[:, None] * azimuth_weight
    )
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
        wavefunction: Complex128[NDArray, "n_cos n_phi"] = (
            coefficients[0] * s_value * y00 + coefficients[1] * p_value * y10
        )
        plane_wave: Complex128[NDArray, "n_cos n_phi"] = np.exp(
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
    returned: Tuple[complex, float] = amplitude, norm
    return returned


def _production_amplitude(
    radial_values: Complex128[NDArray, "n_orb 2"],
    direction_cart: Float64[NDArray, " 3"],
    polarization_cart: Complex128[NDArray, " 3"],
    coefficients: Complex128[NDArray, " n_orb"],
    basis: OrbitalBasis,
    params: MatrixElementParams,
) -> complex:
    """PRIVATE: Assemble the production mixed-parity amplitude from branches.

    Parameters
    ----------
    radial_values : Complex128[NDArray, "n_orb 2"]
        Radial transition integrals for the two dipole branches of each
        orbital.
    direction_cart : Float64[NDArray, " 3"]
        Unit Cartesian final-momentum direction for one detector point.
    polarization_cart : Complex128[NDArray, " 3"]
        Cartesian complex polarization.
    coefficients : Complex128[NDArray, " n_orb"]
        Orbital mixing coefficients of the bound state.
    basis : OrbitalBasis
        Orbital metadata for the sampled basis.
    params : MatrixElementParams
        Matrix-element parameter carrier for the basis.

    Returns
    -------
    amplitude : complex
        Coefficient-weighted polarized transition amplitude.

    Implementation Logic
    --------------------
    Evaluates production channels at one k-point with zero positions
    and depths. Uses a 9 Angstrom mean free path. Contracts Cartesian
    channels with polarization and sums orbital amplitudes against the
    supplied coefficients.
    """
    channels: Complex128[Array, "1 1 2 3"] = orbital_transition_channels(
        jnp.zeros((1, 3)),
        jnp.asarray(direction_cart[None, :]),
        jnp.zeros((2, 3)),
        jnp.zeros((2,)),
        jnp.asarray(radial_values[None, :, :]),
        params,
        jnp.asarray(9.0),
        basis,
    )
    polarized: Complex128[Array, " 2"] = contract_polarization(
        channels[0, 0],
        jnp.asarray(polarization_cart),
    )
    amplitude: complex = complex(
        jnp.sum(polarized * jnp.asarray(coefficients))
    )
    return amplitude


def test_full_cartesian_volume_matches_production() -> None:
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
    direction: Float64[NDArray, " 3"] = np.asarray([0.41, -0.36, 0.84])
    direction = direction / np.linalg.norm(direction)
    polarization: Complex128[NDArray, " 3"] = np.asarray(
        [0.31 + 0.27j, -0.42 + 0.19j, 0.53 - 0.11j],
        dtype=np.complex128,
    )
    coefficients: Complex128[NDArray, " 2"] = np.asarray(
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
    correct_radial: Complex128[NDArray, "n_orb 2"] = np.asarray(
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

    final_degrees: Tuple[Tuple[int, int], ...] = ((-1, 1), (0, 2))
    phase_free: Complex128[NDArray, "n_orb 2"] = np.zeros_like(correct_radial)
    orbital: int
    degrees: Tuple[int, int]
    branch: int
    final_degree: int
    for orbital, degrees in enumerate(final_degrees):
        for branch, final_degree in enumerate(degrees):
            if final_degree >= 0:
                phase_free[orbital, branch] = correct_radial[
                    orbital, branch
                ] / (1j**final_degree)
    controls: Dict[str, Complex128[NDArray, "n_orb 2"]] = {
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
    planted: Complex128[NDArray, "n_orb 2"]
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


def test_hermite_rejection_makes_derivative_path_inactive() -> None:
    """Bind the decisive artifact to runtime rejection and derivative status.

    The frozen 1025-to-2049 witness exceeds the preregistered half-budget
    threshold, so no Hermite resolution exists for derivative certification.

    Notes
    -----
    Read the compact artifact summary and exercise every candidate runtime
    guard.
    """
    reference_path: Path = (
        Path(__file__).resolve().parents[1]
        / "_reference_data"
        / "radial_hermite_rejection.json"
    )
    evidence: Dict[str, object] = json.loads(reference_path.read_text())
    decision: Dict[str, object] = evidence["decision"]
    ratio: float = float(evidence["next_rung_value_budget_ratio"])
    threshold: float = float(evidence["next_rung_fraction_limit"])
    assert evidence["schema"] == "diffpes.radial-hermite-rejection.v1"
    assert ratio > threshold
    assert decision["accepted"] is False
    assert decision["selected_node_count"] is None
    assert decision["runtime"] == "reject-hermite-use-direct"
    assert (
        decision["derivative_status"]
        == "not activated because no accepted Hermite resolution may ship"
    )
    node_count: int
    for node_count in evidence["candidate_node_counts"]:
        with pytest.raises(
            ValueError, match="failed the frozen radial accelerator"
        ):
            make_final_state_spec(
                radial_accelerator="hermite",
                table_n_points=node_count,
            )
    direct: FinalStateSpec = make_final_state_spec()
    assert direct.radial_accelerator == "direct"
