"""Certify the Plan-07 true-Voigt implementation against frozen evidence.

The module freezes artifact and analytic checks before production changes.
Production assertions exercise the same immutable SciPy and analytic truths.
"""

from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
from typing import Any, Callable

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import test_util
from jaxtyping import Array, Float
from scipy import special

from diffpes.simul import simulate_novice, voigt
from tests._assertions import assert_rejects
from tests._factories import (
    toy_band_structure,
    toy_orbital_projection,
    toy_simulation_params,
)

_REFERENCE_DIRECTORY: Path = (
    Path(__file__).resolve().parents[1] / "_reference_data"
)
_REFERENCE_PATH: Path = (
    _REFERENCE_DIRECTORY / "plan07_voigt_scipy_reference.npz"
)
_MANIFEST_PATH: Path = _REFERENCE_DIRECTORY / "plan07_voigt_manifest.json"
_NOVICE_PATH: Path = _REFERENCE_DIRECTORY / "novice_toy_plan07_true_voigt.npz"
_HISTORICAL_PATH: Path = (
    _REFERENCE_DIRECTORY / "novice_toy_plan02_pseudo_voigt.npz"
)
_RETIRED_PLAN02_PATH: Path = _REFERENCE_DIRECTORY / "novice_toy.npz"
_GENERATOR_PATH: Path = (
    Path(__file__).resolve().parents[2]
    / "_reference_tools"
    / "generate_plan07_voigt_reference.py"
)
_CENTER: float = 0.137
_POSITIVE_RTL: float = 1.0e-10
_POSITIVE_G1_FLOOR: float = 2.0e-15
_ENDPOINT_RTL: float = 1.0e-12
_ENDPOINT_FLOOR: float = 5.0e-15
_D1_RTL: float = 1.0e-6
_D1_ATL: float = 2.0e-10


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    """Load one inert NPZ into ordinary arrays without pickle."""
    archive: Any
    with np.load(path, allow_pickle=False) as archive:
        return {name: archive[name] for name in archive.files}


def _sha256(path: Path) -> str:
    """Return the SHA-256 digest of one evidence file."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _positive_bound(
    reference: np.ndarray,
    sigma: float,
) -> np.ndarray:
    """Return the preregistered G1-propagated positive-width bound."""
    return _POSITIVE_RTL * np.abs(reference) + _POSITIVE_G1_FLOOR / (
        sigma * np.sqrt(2.0 * np.pi)
    )


def _profile(
    energy: np.ndarray,
    center: float,
    sigma: float,
    gamma: float,
) -> np.ndarray:
    """Evaluate the independent SciPy profile with analytic endpoints."""
    displacement: np.ndarray = energy - center
    if gamma == 0.0:
        return np.exp(-((displacement / sigma) ** 2) / 2.0) / (
            sigma * np.sqrt(2.0 * np.pi)
        )
    if sigma == 0.0:
        return gamma / (np.pi * (displacement**2 + gamma**2))
    return special.voigt_profile(displacement, sigma, gamma)


def _stable_fermi(energy: np.ndarray) -> np.ndarray:
    """Evaluate the registered overflow-safe analytic Fermi function."""
    exponent: np.ndarray = energy / (8.617333e-5 * 15.0)
    occupation: np.ndarray = np.empty_like(exponent)
    positive: np.ndarray = exponent >= 0.0
    decaying: np.ndarray = np.exp(-exponent[positive])
    occupation[positive] = decaying / (1.0 + decaying)
    occupation[~positive] = 1.0 / (1.0 + np.exp(exponent[~positive]))
    return occupation


class TestPlan07VoigtEvidence:
    """Validate the frozen independent artifacts before production editing."""

    def test_generator_boundary_and_manifest_are_frozen(self) -> None:
        """Require a production-independent generator and authenticated files.

        Extended Summary
        ----------------
        The test verifies the generator import boundary and every registered
        archive digest.

        Notes
        -----
        Parse the generator AST, load the JSON manifest, and inspect each
        deterministic archive without pickle.
        """
        source: str = _GENERATOR_PATH.read_text(encoding="utf-8")
        tree: ast.Module = ast.parse(source)
        imported_roots: set[str] = set()
        node: ast.AST
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported_roots.update(
                    alias.name.split(".", maxsplit=1)[0]
                    for alias in node.names
                )
            elif isinstance(node, ast.ImportFrom) and node.module is not None:
                imported_roots.add(node.module.split(".", maxsplit=1)[0])
        assert "diffpes" not in imported_roots
        assert "jax" not in imported_roots

        manifest: dict[str, Any] = json.loads(
            _MANIFEST_PATH.read_text(encoding="utf-8")
        )
        assert manifest["schema"] == "diffpes.plan07.voigt-reference.v1"
        assert manifest["stage"] == (
            "preregistered-before-WP7.2-production-edit"
        )
        assert manifest["generator_sha256"] == _sha256(_GENERATOR_PATH)
        archive_key: str
        archive_path: Path
        for archive_key, archive_path in (
            ("voigt_reference", _REFERENCE_PATH),
            ("novice_plan07", _NOVICE_PATH),
            ("historical_plan02", _HISTORICAL_PATH),
        ):
            assert manifest["archives"][archive_key]["sha256"] == _sha256(
                archive_path
            )
            arrays: dict[str, np.ndarray] = _load_npz(archive_path)
            assert arrays
            assert all(array.dtype == np.float64 for array in arrays.values())
        assert not _RETIRED_PLAN02_PATH.exists()
        assert manifest["archives"]["historical_plan02"]["classification"] == (
            "superseded pseudo-Voigt evidence; not a compatibility shim"
        )

    def test_positive_reference_table_replays_scipy(self) -> None:
        """Recompute all 360 positive-width values and their G1 coordinates.

        Extended Summary
        ----------------
        The test confirms the frozen positive table matches SciPy throughout
        the closed Faddeeva envelope.

        Notes
        -----
        Evaluate each width row with SciPy, reconstruct its complex
        coordinates, and check value finiteness and nonnegativity.
        """
        reference: dict[str, np.ndarray] = _load_npz(_REFERENCE_PATH)
        widths: np.ndarray = reference["positive_widths"]
        energies: np.ndarray = reference["positive_energies"]
        desired: np.ndarray = reference["positive_values"]
        actual: np.ndarray = np.stack(
            [
                special.voigt_profile(energy - _CENTER, sigma, gamma)
                for energy, (sigma, gamma) in zip(
                    energies,
                    widths,
                    strict=True,
                )
            ]
        )
        np.testing.assert_array_equal(actual, desired)
        z_values: np.ndarray = (
            reference["positive_z_real"] + 1j * reference["positive_z_imag"]
        )
        assert np.max(np.abs(z_values)) <= 1.0e8
        assert np.all(np.isfinite(desired))
        assert np.all(desired >= 0.0)

    def test_endpoint_reference_rows_are_exact_analytic_values(self) -> None:
        """Recompute every value-only Gaussian and Cauchy endpoint row.

        Extended Summary
        ----------------
        The test confirms both endpoint families use the representable
        displacement coordinates stored in the artifact.

        Notes
        -----
        Reconstruct each normalized coordinate, evaluate its analytic density,
        and compare the resulting arrays exactly.
        """
        reference: dict[str, np.ndarray] = _load_npz(_REFERENCE_PATH)
        rows: list[np.ndarray] = []
        energy: np.ndarray
        sigma: float
        gamma: float
        for energy, (sigma, gamma) in zip(
            reference["endpoint_energies"],
            reference["endpoint_widths"],
            strict=True,
        ):
            nonzero_width: float = max(sigma, gamma)
            q_hat: np.ndarray = (energy - _CENTER) / nonzero_width
            if gamma == 0.0:
                row: np.ndarray = np.exp(-(q_hat**2) / 2.0) / (
                    nonzero_width * np.sqrt(2.0 * np.pi)
                )
            else:
                row = 1.0 / (np.pi * nonzero_width * (1.0 + q_hat**2))
            rows.append(row)
        actual: np.ndarray = np.stack(rows)
        np.testing.assert_array_equal(actual, reference["endpoint_values"])

    def test_one_sided_reference_rates_are_registered(self) -> None:
        """Require quadratic sigma and linear gamma endpoint convergence.

        Extended Summary
        ----------------
        The test verifies each positive rung against SciPy and certifies both
        preregistered convergence intervals.

        Notes
        -----
        Recompute every rung, require strict error decay, and bound successive
        ratios for both endpoint directions.
        """
        reference: dict[str, np.ndarray] = _load_npz(_REFERENCE_PATH)
        actual: np.ndarray = np.stack(
            [
                special.voigt_profile(energy - _CENTER, sigma, gamma)
                for energy, (sigma, gamma) in zip(
                    reference["onesided_energies"],
                    reference["onesided_widths"],
                    strict=True,
                )
            ]
        )
        np.testing.assert_array_equal(actual, reference["onesided_values"])
        differences: np.ndarray = reference["onesided_differences"]
        ratios: np.ndarray = reference["onesided_ratios"]
        assert np.all(np.diff(differences, axis=-1) < 0.0)
        assert np.all((ratios[0] >= 15.5) & (ratios[0] <= 16.5))
        assert np.all((ratios[1] >= 3.9) & (ratios[1] <= 4.1))

    def test_scaled_full_line_reference_mass_is_unity(self) -> None:
        """Recompute the frozen 256-to-512 tangent-map mass battery.

        Extended Summary
        ----------------
        The test verifies normalization for every frozen interior and endpoint
        row at both quadrature orders.

        Notes
        -----
        Apply the scaled tangent map, include its Jacobian, and compare both
        mass estimates with the committed reference.
        """
        reference: dict[str, np.ndarray] = _load_npz(_REFERENCE_PATH)
        widths: np.ndarray = reference["normalization_widths"]
        scales: np.ndarray = reference["normalization_scales"]
        expected_masses: np.ndarray = reference["normalization_masses"]
        actual_masses: np.ndarray = np.empty_like(expected_masses)
        row: int
        sigma: float
        gamma: float
        scale: float
        column: int
        order_float: float
        for row, ((sigma, gamma), scale) in enumerate(
            zip(widths, scales, strict=True)
        ):
            for column, order_float in enumerate(
                reference["normalization_orders"]
            ):
                order: int = int(order_float)
                nodes: np.ndarray
                weights: np.ndarray
                nodes, weights = np.polynomial.legendre.leggauss(order)
                angle: np.ndarray = np.pi * nodes / 2.0
                energy: np.ndarray = _CENTER + scale * np.tan(angle)
                jacobian: np.ndarray = scale * np.pi / 2.0 / np.cos(angle) ** 2
                actual_masses[row, column] = np.sum(
                    weights
                    * _profile(energy, _CENTER, sigma, gamma)
                    * jacobian
                )
        np.testing.assert_array_equal(actual_masses, expected_masses)
        assert np.max(np.abs(actual_masses - 1.0)) <= 2.0e-10
        assert (
            np.max(np.abs(actual_masses[:, 1] - actual_masses[:, 0]))
            <= 2.0e-10
        )
        assert np.max(reference["normalization_maximum_z"][:6]) <= 1.0e8

    def test_envelope_reference_reconstructs_registered_radii(self) -> None:
        """Verify the frozen pass, boundary, and rejection coordinates.

        Extended Summary
        ----------------
        The test confirms every stored energy row reconstructs its intended
        Faddeeva radius within the float64 budget.

        Notes
        -----
        Map energies back to complex arguments, select each row maximum, and
        compare against the registered radii.
        """
        reference: dict[str, np.ndarray] = _load_npz(_REFERENCE_PATH)
        sigma: float
        gamma: float
        sigma, gamma = reference["envelope_widths"]
        energies: np.ndarray = reference["envelope_energies"]
        reconstructed: np.ndarray = np.max(
            np.abs((energies - _CENTER + 1j * gamma) / (sigma * np.sqrt(2.0))),
            axis=1,
        )
        radii: np.ndarray = reference["envelope_radii"]
        np.testing.assert_array_equal(
            reconstructed,
            reference["envelope_reconstructed_radii"],
        )
        assert np.all(
            np.abs(reconstructed - radii)
            <= 4.0 * np.finfo(np.float64).eps * radii
        )

    def test_d1_artifact_matches_analytic_rows_and_fd(self) -> None:
        """Recompute analytic wofz derivatives and validate the FD plateau.

        Extended Summary
        ----------------
        The test confirms point derivatives, contracted sensitivities, and
        multistep finite differences agree with independent analytic truth.

        Notes
        -----
        Apply the Faddeeva ODE derivative, contract each probe, and inspect the
        median and spread of all stencil rungs.
        """
        reference: dict[str, np.ndarray] = _load_npz(_REFERENCE_PATH)
        probes: np.ndarray = reference["d1_probes"]
        energies: np.ndarray = reference["d1_energies"]
        desired_values: np.ndarray = reference["d1_point_values"]
        desired_derivatives: np.ndarray = reference["d1_point_derivatives"]
        actual_values: np.ndarray = np.empty_like(desired_values)
        actual_derivatives: np.ndarray = np.empty_like(desired_derivatives)
        row: int
        probe: np.ndarray
        energy: np.ndarray
        for row, (probe, energy) in enumerate(
            zip(probes, energies, strict=True)
        ):
            center: float
            sigma: float
            gamma: float
            center, sigma, gamma = probe
            z_values: np.ndarray = (energy - center + 1j * gamma) / (
                sigma * np.sqrt(2.0)
            )
            w_values: np.ndarray = special.wofz(z_values)
            w_prime: np.ndarray = -2.0 * z_values * w_values + 2j / np.sqrt(
                np.pi
            )
            prefactor: float = 1.0 / (sigma * np.sqrt(2.0 * np.pi))
            values: np.ndarray = prefactor * np.real(w_values)
            actual_values[row] = values
            actual_derivatives[row] = np.stack(
                (
                    prefactor
                    * np.real(w_prime * (-1.0 / (sigma * np.sqrt(2.0)))),
                    -values / sigma
                    + prefactor * np.real(w_prime * (-z_values / sigma)),
                    prefactor
                    * np.real(w_prime * (1j / (sigma * np.sqrt(2.0)))),
                ),
                axis=1,
            )
        np.testing.assert_array_equal(actual_values, desired_values)
        np.testing.assert_array_equal(actual_derivatives, desired_derivatives)

        contracted: np.ndarray = reference["d1_analytic_contracted"]
        estimates: np.ndarray = reference["d1_fd_estimates"]
        median: np.ndarray = np.median(estimates, axis=1)
        spread: np.ndarray = np.ptp(estimates, axis=1)
        np.testing.assert_allclose(
            median,
            contracted,
            rtol=_D1_RTL,
            atol=_D1_ATL,
        )
        assert np.all(spread <= _D1_ATL + _D1_RTL * np.abs(contracted))
        assert np.all(np.abs(contracted) > 1.0e-4)

    def test_novice_artifact_is_manual_scipy_truth(self) -> None:
        """Build the fixed-input novice spectrum without production.

        Extended Summary
        ----------------
        The test confirms manual SciPy broadening and analytic occupation
        reproduce the committed novice artifact.

        Notes
        -----
        Load frozen eigenvalues and weights, broaden every band, and reduce the
        contributions over the band axis.
        """
        reference: dict[str, np.ndarray] = _load_npz(_REFERENCE_PATH)
        novice: dict[str, np.ndarray] = _load_npz(_NOVICE_PATH)
        eigenvalues: np.ndarray = reference["novice_eigenvalues"]
        band_weights: np.ndarray = reference["novice_band_weights"]
        energy_axis: np.ndarray = np.linspace(-3.0, 0.5, 512)
        occupations: np.ndarray = _stable_fermi(eigenvalues)
        profiles: np.ndarray = special.voigt_profile(
            energy_axis[None, None, :] - eigenvalues[..., None],
            0.04,
            0.1,
        )
        intensity: np.ndarray = np.sum(
            band_weights[..., None] * occupations[..., None] * profiles,
            axis=1,
        )
        np.testing.assert_array_equal(
            intensity,
            novice["leaf_000_intensity"],
        )
        np.testing.assert_array_equal(
            energy_axis,
            novice["leaf_001_energy_axis"],
        )


class TestPlan07VoigtProduction:
    """Certify production against the preregistered G2/D1 witnesses."""

    def test_positive_width_table_matches_true_voigt(self) -> None:
        """Match all positive production rows under the G1-derived bound.

        Extended Summary
        ----------------
        The test verifies production values remain finite, nonnegative, and
        within the propagated Faddeeva error budget.

        Notes
        -----
        Evaluate each frozen energy row and compare every element with its
        independent SciPy value.
        """
        reference: dict[str, np.ndarray] = _load_npz(_REFERENCE_PATH)
        energy: np.ndarray
        desired: np.ndarray
        sigma: float
        gamma: float
        for energy, desired, (sigma, gamma) in zip(
            reference["positive_energies"],
            reference["positive_values"],
            reference["positive_widths"],
            strict=True,
        ):
            actual: np.ndarray = np.asarray(
                voigt(jnp.asarray(energy), _CENTER, sigma, gamma)
            )
            assert np.all(np.isfinite(actual))
            assert np.all(actual >= 0.0)
            assert np.all(
                np.abs(actual - desired) <= _positive_bound(desired, sigma)
            )

    def test_exact_endpoint_rows_match_analytic_values(self) -> None:
        """Match endpoint values without differentiating either selector.

        Extended Summary
        ----------------
        The test verifies production follows the exact Gaussian and Cauchy
        value conventions at zero component width.

        Notes
        -----
        Evaluate each endpoint row and apply the dedicated mixed endpoint
        tolerance without invoking any derivative transform.
        """
        reference: dict[str, np.ndarray] = _load_npz(_REFERENCE_PATH)
        energy: np.ndarray
        desired: np.ndarray
        sigma: float
        gamma: float
        for energy, desired, (sigma, gamma) in zip(
            reference["endpoint_energies"],
            reference["endpoint_values"],
            reference["endpoint_widths"],
            strict=True,
        ):
            actual: np.ndarray = np.asarray(
                voigt(jnp.asarray(energy), _CENTER, sigma, gamma)
            )
            nonzero_width: float = max(sigma, gamma)
            bound: np.ndarray = (
                _ENDPOINT_FLOOR / nonzero_width
                + _ENDPOINT_RTL * np.abs(desired)
            )
            assert np.all(np.abs(actual - desired) <= bound)

    def test_one_sided_rows_match_scipy_and_converge(self) -> None:
        """Require each positive rung and its registered endpoint rate.

        Extended Summary
        ----------------
        The test verifies production values and both one-sided convergence
        orders against the frozen tables.

        Notes
        -----
        Collect every positive profile, measure endpoint errors, and bound
        their successive ratios after strict decay checks.
        """
        reference: dict[str, np.ndarray] = _load_npz(_REFERENCE_PATH)
        actual_values: list[np.ndarray] = []
        energy: np.ndarray
        desired: np.ndarray
        sigma: float
        gamma: float
        for energy, desired, (sigma, gamma) in zip(
            reference["onesided_energies"],
            reference["onesided_values"],
            reference["onesided_widths"],
            strict=True,
        ):
            actual: np.ndarray = np.asarray(
                voigt(jnp.asarray(energy), _CENTER, sigma, gamma)
            )
            assert np.all(
                np.abs(actual - desired) <= _positive_bound(desired, sigma)
            )
            actual_values.append(actual)
        differences: np.ndarray = np.empty((2, 3, 3), dtype=np.float64)
        rows: np.ndarray = np.asarray(actual_values).reshape(2, 3, 3, 10)
        endpoints: np.ndarray = reference["onesided_endpoint_values"].reshape(
            2, 3, 3, 10
        )
        anchors: np.ndarray = reference["anchors"]
        direction: int
        anchor_index: int
        anchor: float
        for direction in range(2):
            for anchor_index, anchor in enumerate(anchors):
                differences[direction, anchor_index] = anchor * np.max(
                    np.abs(
                        rows[direction, anchor_index]
                        - endpoints[direction, anchor_index]
                    ),
                    axis=1,
                )
        ratios: np.ndarray = differences[..., :-1] / differences[..., 1:]
        assert np.all(np.diff(differences, axis=-1) < 0.0)
        assert np.all((ratios[0] >= 15.5) & (ratios[0] <= 16.5))
        assert np.all((ratios[1] >= 3.9) & (ratios[1] <= 4.1))

    def test_scaled_full_line_production_mass_is_unity(self) -> None:
        """Require both quadrature orders and their delta to meet G2.

        Extended Summary
        ----------------
        The test verifies production integrates to unit mass for every
        interior and analytic endpoint row.

        Notes
        -----
        Evaluate production on both scaled tangent grids, apply quadrature
        weights, and check each mass and order delta.
        """
        reference: dict[str, np.ndarray] = _load_npz(_REFERENCE_PATH)
        widths: np.ndarray = reference["normalization_widths"]
        scales: np.ndarray = reference["normalization_scales"]
        masses: np.ndarray = np.empty((widths.shape[0], 2))
        row: int
        sigma: float
        gamma: float
        scale: float
        column: int
        order_float: float
        for row, ((sigma, gamma), scale) in enumerate(
            zip(widths, scales, strict=True)
        ):
            for column, order_float in enumerate(
                reference["normalization_orders"]
            ):
                nodes: np.ndarray
                weights: np.ndarray
                nodes, weights = np.polynomial.legendre.leggauss(
                    int(order_float)
                )
                angle: np.ndarray = np.pi * nodes / 2.0
                energy: np.ndarray = _CENTER + scale * np.tan(angle)
                jacobian: np.ndarray = scale * np.pi / 2.0 / np.cos(angle) ** 2
                profile: np.ndarray = np.asarray(
                    voigt(jnp.asarray(energy), _CENTER, sigma, gamma)
                )
                masses[row, column] = np.sum(weights * profile * jacobian)
        assert np.max(np.abs(masses - 1.0)) <= 2.0e-10
        assert np.max(np.abs(masses[:, 1] - masses[:, 0])) <= 2.0e-10

    def test_shared_envelope_passes_and_rejects_complete_arrays(self) -> None:
        """Enforce the closed G1 envelope eagerly and under JIT.

        Extended Summary
        ----------------
        The test confirms accepted radii evaluate while boundary violations
        reject every element in the submitted array.

        Notes
        -----
        Exercise interior and closed-boundary rows in both modes, then plant
        isolated offenders and require the registered diagnostic.
        """
        reference: dict[str, np.ndarray] = _load_npz(_REFERENCE_PATH)
        sigma: float
        gamma: float
        sigma, gamma = reference["envelope_widths"]
        energies: np.ndarray = reference["envelope_energies"]
        accepted: np.ndarray
        for accepted in energies[:2]:
            eager: Array = voigt(
                jnp.asarray(accepted),
                _CENTER,
                sigma,
                gamma,
            )
            compiled: Array = eqx.filter_jit(voigt)(
                jnp.asarray(accepted),
                _CENTER,
                sigma,
                gamma,
            )
            assert np.all(np.isfinite(np.asarray(eager)))
            assert np.all(np.isfinite(np.asarray(compiled)))
        assert_rejects(
            voigt,
            jnp.asarray(energies[2]),
            _CENTER,
            sigma,
            gamma,
            match="Faddeeva envelope",
        )
        planted: np.ndarray = np.concatenate(
            (energies[0, :2], energies[2, 2:])
        )
        assert_rejects(
            voigt,
            jnp.asarray(planted),
            _CENTER,
            sigma,
            gamma,
            match="Faddeeva envelope",
        )

    def test_width_domain_empty_output_and_endpoint_bypass(self) -> None:
        """Retain width rejection, empty shape, and analytic endpoint bypass.

        Extended Summary
        ----------------
        The test verifies invalid widths reject while valid empty arrays and
        far-tail endpoint calls remain supported.

        Notes
        -----
        Submit each invalid width pair, inspect the empty result contract, and
        evaluate both analytic endpoints outside the shared envelope.
        """
        energy: Float[Array, "3"] = jnp.asarray(
            [_CENTER - 1.0, _CENTER, _CENTER + 1.0],
            dtype=jnp.float64,
        )
        invalid: tuple[tuple[float, float, str], ...] = (
            (-1.0e-6, 2.0e-6, "sigma must be finite and nonnegative"),
            (1.0e-6, -2.0e-6, "gamma must be finite and nonnegative"),
            (np.nan, 2.0e-6, "sigma must be finite and nonnegative"),
            (1.0e-6, np.inf, "gamma must be finite and nonnegative"),
            (0.0, 0.0, "sigma and gamma must not both be zero"),
        )
        sigma: float
        gamma: float
        message: str
        for sigma, gamma, message in invalid:
            assert_rejects(
                voigt,
                energy,
                _CENTER,
                sigma,
                gamma,
                match=message,
            )
        empty: Array = voigt(
            jnp.empty((0,), dtype=jnp.float64),
            _CENTER,
            1.0e-6,
            2.0e-6,
        )
        assert empty.shape == (0,)
        assert empty.dtype == jnp.float64
        for sigma, gamma in ((1.0e-6, 0.0), (0.0, 2.0e-6)):
            endpoint: Array = voigt(
                jnp.asarray([1.0e9], dtype=jnp.float64),
                _CENTER,
                sigma,
                gamma,
            )
            assert np.all(np.isfinite(np.asarray(endpoint)))

    def test_nonfinite_energy_and_center_reject(self) -> None:
        """Reject nonfinite profile coordinates eagerly and under JIT.

        Extended Summary
        ----------------
        The test verifies every nonfinite energy and center variant fails with
        its coordinate-specific diagnostic.

        Notes
        -----
        Plant each nonfinite scalar into an otherwise valid call and repeat the
        rejection check through compiled execution.
        """
        finite: Float[Array, "3"] = jnp.asarray(
            [_CENTER - 1.0, _CENTER, _CENTER + 1.0],
            dtype=jnp.float64,
        )
        invalid_energy: float
        for invalid_energy in (np.nan, np.inf, -np.inf):
            planted: Array = finite.at[1].set(invalid_energy)
            assert_rejects(
                voigt,
                planted,
                _CENTER,
                1.0e-6,
                2.0e-6,
                match="energy.*finite",
            )
        invalid_center: float
        for invalid_center in (np.nan, np.inf, -np.inf):
            assert_rejects(
                voigt,
                finite,
                invalid_center,
                1.0e-6,
                2.0e-6,
                match="center.*finite",
            )

    def test_d1_jacfwd_jacrev_and_check_grads_match_truth(self) -> None:
        """Match all positive D1 probes in dimensionless coordinates.

        Extended Summary
        ----------------
        The test verifies forward mode, reverse mode, and directional checks
        against the contracted analytic derivative truth.

        Notes
        -----
        Build a fixed-energy loss for each probe, differentiate its
        dimensionless parameters, and apply the preregistered comparison
        budget.
        """
        reference: dict[str, np.ndarray] = _load_npz(_REFERENCE_PATH)
        weights: Array = jnp.asarray(reference["d1_weights"])
        zero: Float[Array, "3"] = jnp.zeros(3, dtype=jnp.float64)
        probe: np.ndarray
        energy: np.ndarray
        desired: np.ndarray
        for probe, energy, desired in zip(
            reference["d1_probes"],
            reference["d1_energies"],
            reference["d1_analytic_contracted"],
            strict=True,
        ):
            center: float
            sigma: float
            gamma: float
            center, sigma, gamma = probe
            scale: float = max(sigma, gamma)
            energy_array: Array = jnp.asarray(energy)

            def loss(parameters: Float[Array, "3"]) -> Float[Array, ""]:
                profile: Array = voigt(
                    energy_array,
                    center + scale * parameters[0],
                    sigma * (1.0 + parameters[1]),
                    gamma * (1.0 + parameters[2]),
                )
                return scale * jnp.sum(weights * profile)

            forward: Array = jax.jacfwd(loss)(zero)
            reverse: Array = jax.jacrev(loss)(zero)
            np.testing.assert_allclose(
                np.asarray(forward),
                desired,
                rtol=_D1_RTL,
                atol=_D1_ATL,
            )
            np.testing.assert_allclose(
                np.asarray(reverse),
                desired,
                rtol=_D1_RTL,
                atol=_D1_ATL,
            )
            test_util.check_grads(
                loss,
                (zero,),
                order=1,
                modes=("fwd", "rev"),
                eps=2.0**-14,
                rtol=_D1_RTL,
                atol=_D1_ATL,
            )

    @pytest.mark.big_mem
    @pytest.mark.rss_limit_mb(1200)
    def test_novice_replay_matches_true_voigt_artifact(self) -> None:
        """Replay the retained novice inputs against independent true truth.

        Extended Summary
        ----------------
        The test verifies the complete production novice path reproduces the
        manually assembled Plan-07 artifact.

        Notes
        -----
        Build the fixed-seed carriers, simulate the spectrum, and compare both
        carrier arrays with their independent references.
        """
        key: Array = jax.random.key(20260713)
        spectrum: Any = simulate_novice(
            toy_band_structure(key),
            toy_orbital_projection(key),
            toy_simulation_params(fidelity=512),
            15.0,
        )
        desired: dict[str, np.ndarray] = _load_npz(_NOVICE_PATH)
        np.testing.assert_allclose(
            np.asarray(spectrum.intensity),
            desired["leaf_000_intensity"],
            rtol=1.0e-12,
            atol=0.0,
        )
        np.testing.assert_allclose(
            np.asarray(spectrum.energy_axis),
            desired["leaf_001_energy_axis"],
            rtol=1.0e-12,
            atol=2.0 * np.finfo(np.float64).eps,
        )
