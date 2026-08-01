"""Replay matrix-element and polarization parity against pinned Chinook data.

Extended Summary
----------------
The tests reconstruct current public-API results from a frozen K-reference.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from beartype.typing import Any
from jaxtyping import Bool, Complex128, Float64, Int64, Shaped
from numpy.typing import NDArray

from diffpes.simul import (
    contract_polarization,
    orbital_transition_channels,
    project_band_channels,
)
from diffpes.types import (
    MatrixElementParams,
    OrbitalBasis,
    make_matrix_element_params,
    make_orbital_basis,
)

_REFERENCE_DIRECTORY = (
    Path(__file__).parents[1]
    / "_reference_data"
    / "chinook_matrix_element_parity"
)
_RTOL = 1.0e-6
_ATOL = 1.0e-12
_RING_RADIUS_INV_ANG = 0.12
_RING_HALF_WIDTH_INV_ANG = 0.018


def _transition_band_channels(
    k_i_state: Float64[NDArray, "n_state 3"],
    k_f_state: Float64[NDArray, "n_state 3"],
    positions_cartesian: Float64[NDArray, "n_orb 3"],
    coefficients_state: Complex128[NDArray, "n_state n_orb"],
) -> jax.Array:
    """Build transition and band channels through public DiffPES APIs."""
    n_states: int = len(k_i_state)
    n_orbitals: int = positions_cartesian.shape[0]
    basis: OrbitalBasis = make_orbital_basis(
        atom_indices=tuple(range(n_orbitals)),
        n=(2,) * n_orbitals,
        l=(1,) * n_orbitals,
        m=(0,) * n_orbitals,
    )
    parameters: MatrixElementParams = make_matrix_element_params(
        basis,
        tuple(range(n_orbitals)),
    )
    bvals: jax.Array = jnp.broadcast_to(
        jnp.asarray([1.0 + 0.0j, -1.0 + 0.0j]),
        (n_states, n_orbitals, 2),
    )
    transition: jax.Array = orbital_transition_channels(
        jnp.asarray(k_i_state),
        jnp.asarray(k_f_state),
        jnp.asarray(positions_cartesian),
        jnp.zeros((n_orbitals,), dtype=jnp.float64),
        bvals,
        parameters,
        jnp.asarray(10.0, dtype=jnp.float64),
        basis,
    )
    band_channels: jax.Array = project_band_channels(
        transition,
        jnp.asarray(coefficients_state[:, None, :]),
    )
    return band_channels


def _polarized_amplitudes(
    band_channels: jax.Array,
    polarizations: Complex128[NDArray, "n_pol 3"],
) -> Complex128[NDArray, "n_state n_pol"]:
    """Return public band amplitudes for each supplied polarization."""
    amplitudes: list[Complex128[NDArray, " n_state"]] = []
    polarization: Complex128[NDArray, " 3"]
    for polarization in polarizations:
        contracted: jax.Array = contract_polarization(
            band_channels,
            jnp.asarray(polarization),
        )
        amplitudes.append(np.asarray(contracted[:, 0, 0]))
    result: Complex128[NDArray, "n_state n_pol"] = np.stack(
        amplitudes, axis=-1
    )
    return result


def _canonical_cartesian_amplitudes(
    k_i_state: Float64[NDArray, "n_state 3"],
    k_f_state: Float64[NDArray, "n_state 3"],
    positions_cartesian: Float64[NDArray, "n_orb 3"],
    coefficients_state: Complex128[NDArray, "n_state n_orb"],
) -> Complex128[NDArray, "n_state 3"]:
    """Evaluate Cartesian components through the public low-level assembly."""
    band_channels: jax.Array = _transition_band_channels(
        k_i_state,
        k_f_state,
        positions_cartesian,
        coefficients_state,
    )
    result: Complex128[NDArray, "n_state 3"] = _polarized_amplitudes(
        band_channels,
        np.eye(3, dtype=complex),
    )
    return result


def _dark_ring_angles(
    k_i_state: Float64[NDArray, "n_state 3"],
    band_indices: Int64[NDArray, " n_state"],
    valley_indices: Int64[NDArray, " n_state"],
    valley_centres: Float64[NDArray, "n_valley 3"],
    intensities: Float64[NDArray, " n_state"],
) -> Float64[NDArray, " n_angle"]:
    """Measure the darkest sampled annular angle for each valley and band."""
    angles: list[float] = []
    valley_index: int
    band_index: int
    for valley_index in range(2):
        centre: Float64[NDArray, " 2"] = valley_centres[valley_index, :2]
        for band_index in range(2):
            selected: Bool[NDArray, " n_state"] = (
                valley_indices == valley_index
            ) & (band_indices == band_index)
            offsets: Float64[NDArray, "n_selected 2"] = (
                k_i_state[selected, :2] - centre
            )
            radii: Float64[NDArray, " n_selected"] = np.linalg.norm(
                offsets, axis=1
            )
            annulus: Bool[NDArray, " n_selected"] = (
                np.abs(radii - _RING_RADIUS_INV_ANG)
                <= _RING_HALF_WIDTH_INV_ANG
            )
            annulus_offsets: Float64[NDArray, "n_annulus 2"] = offsets[annulus]
            annulus_intensities: Float64[NDArray, " n_annulus"] = intensities[
                selected
            ][annulus]
            minimum: int = int(np.argmin(annulus_intensities))
            angle: float = math.atan2(
                float(annulus_offsets[minimum, 1]),
                float(annulus_offsets[minimum, 0]),
            )
            angles.append(angle)
    result: Float64[NDArray, " n_angle"] = np.asarray(angles)
    return result


def test_graphene_pointwise_map_and_valley_orientation() -> None:
    """Validate the frozen graphene map and its opposite-valley orientation.

    The test checks amplitudes, intensities, and darkest annular directions.

    Notes
    -----
    Reconstruct every state through the public low-level assembly.
    """
    archive: Any
    with np.load(_REFERENCE_DIRECTORY / "chinook_reference.npz") as archive:
        reference: dict[str, Shaped[NDArray, "..."]] = {
            key: archive[key] for key in archive.files
        }
    measured: Complex128[NDArray, "n_state 3"] = (
        _canonical_cartesian_amplitudes(
            reference["graphene_k_i_state"],
            reference["graphene_k_f_state"],
            reference["graphene_positions_cartesian_angstrom"],
            reference["graphene_coefficients_state_orb"],
        )
    )
    expected: Complex128[NDArray, "n_state 3"] = reference[
        "graphene_canonical_amplitudes_state_spin_cartesian"
    ][:, 0, :]
    np.testing.assert_allclose(
        measured,
        expected,
        rtol=_RTOL,
        atol=_ATOL,
    )
    measured_intensity: Float64[NDArray, " n_state"] = (
        np.abs(measured[:, 0]) ** 2
    )
    expected_intensity: Float64[NDArray, " n_state"] = (
        np.abs(expected[:, 0]) ** 2
    )
    np.testing.assert_allclose(
        measured_intensity,
        expected_intensity,
        rtol=_RTOL,
        atol=_ATOL,
    )
    measured_angles: Float64[NDArray, " n_angle"] = _dark_ring_angles(
        reference["graphene_k_i_state"],
        reference["graphene_band_indices"],
        reference["graphene_valley_index_state"],
        reference["graphene_valley_centres_inverse_angstrom"],
        measured_intensity,
    )
    expected_angles: Float64[NDArray, " n_angle"] = _dark_ring_angles(
        reference["graphene_k_i_state"],
        reference["graphene_band_indices"],
        reference["graphene_valley_index_state"],
        reference["graphene_valley_centres_inverse_angstrom"],
        expected_intensity,
    )
    np.testing.assert_array_equal(measured_angles, expected_angles)


def test_polarization_intensities_and_ratios() -> None:
    """Validate s, p, and both helicity rows against the frozen reference.

    The test compares absolute intensities and p-normalized ratios.

    Notes
    -----
    Contract the reconstructed Cartesian transition with four polarizations.
    """
    archive: Any
    with np.load(_REFERENCE_DIRECTORY / "chinook_reference.npz") as archive:
        reference: dict[str, Shaped[NDArray, "..."]] = {
            key: archive[key] for key in archive.files
        }
    band_channels: jax.Array = _transition_band_channels(
        reference["polarization_k_i_cart_inverse_angstrom"],
        reference["polarization_k_f_cart_inverse_angstrom"],
        np.zeros((1, 3), dtype=float),
        np.ones((1, 1), dtype=complex),
    )
    amplitudes: Complex128[NDArray, " n_pol"] = _polarized_amplitudes(
        band_channels,
        reference["polarization_cartesian"],
    )[0]
    intensities: Float64[NDArray, " n_pol"] = np.abs(amplitudes) ** 2
    ratios: Float64[NDArray, " n_pol"] = intensities / intensities[1]
    np.testing.assert_allclose(
        intensities,
        reference["polarization_intensities_physical"],
        rtol=_RTOL,
        atol=_ATOL,
    )
    np.testing.assert_allclose(
        ratios,
        reference["polarization_ratios_to_p_physical"],
        rtol=_RTOL,
        atol=_ATOL,
    )


def test_chinook_artifact_provenance_and_digest() -> None:
    """Validate the immutable archive digest and pinned comparator identity.

    The test binds the data to its model specification and Chinook commit.

    Notes
    -----
    Hash both tracked files before inspecting the provenance fields.
    """
    manifest_path: Path = _REFERENCE_DIRECTORY / "manifest.json"
    manifest: dict[str, Any] = json.loads(
        manifest_path.read_text(encoding="utf-8")
    )
    archive_path: Path = _REFERENCE_DIRECTORY / str(manifest["archive"])
    archive_sha256: str = hashlib.sha256(archive_path.read_bytes()).hexdigest()
    model_path: Path = _REFERENCE_DIRECTORY / str(
        manifest["model_specification"]
    )
    model_sha256: str = hashlib.sha256(model_path.read_bytes()).hexdigest()
    environment_path: Path = _REFERENCE_DIRECTORY / "chinook_env_freeze.txt"
    environment_sha256: str = hashlib.sha256(
        environment_path.read_bytes()
    ).hexdigest()
    assert manifest["requirements"] == [
        "chinook-pointwise-matrix-element-parity",
        "chinook-polarization-intensity-parity",
    ]
    assert manifest["classification"] == "K-type behavioral compatibility"
    assert manifest["chinook_commit"] == (
        "24913de8cc5b8c162f7c1b4acc64bd1b54dd548b"
    )
    assert archive_sha256 == manifest["archive_sha256"]
    assert model_sha256 == manifest["model_specification_sha256"]
    assert environment_sha256 == manifest["environment_sha256"]
