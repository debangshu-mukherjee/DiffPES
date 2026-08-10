"""Certify the analytic, gauge, and derivative gates for Plan-07 WP7.5.

Independent finite differences, gauge rotations, full-line quadrature, and a
regulator ladder exercise the public eigen and resolvent primitives.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from beartype.typing import Tuple
from jaxtyping import Array, Complex128, Float64
from numpy.typing import NDArray

from diffpes.simul import (
    spectral_intensity_eigen,
    spectral_intensity_resolvent,
)


def _quadratic_peak_and_hwhm(
    values: Float64[NDArray, " n"],
    omega: Float64[NDArray, " n"],
) -> Tuple[float, float]:
    """PRIVATE: Return a quadratic peak and linear-crossing HWHM.

    Notes
    -----
    Three samples around the maximum determine the sub-grid peak. Independent
    linear crossings on both flanks determine the half width.
    """
    index: int = int(np.argmax(values))
    left: float = float(values[index - 1])
    middle: float = float(values[index])
    right: float = float(values[index + 1])
    curvature: float = left - 2.0 * middle + right
    offset: float = 0.5 * (left - right) / curvature
    spacing: float = float(omega[1] - omega[0])
    peak_position: float = float(omega[index] + offset * spacing)
    peak_height: float = middle - 0.25 * (left - right) * offset
    half_height: float = 0.5 * peak_height

    left_index: int = index
    while values[left_index] >= half_height:
        left_index -= 1
    left_fraction: float = (half_height - values[left_index]) / (
        values[left_index + 1] - values[left_index]
    )
    left_crossing: float = float(omega[left_index] + left_fraction * spacing)

    right_index: int = index
    while values[right_index] >= half_height:
        right_index += 1
    right_fraction: float = (half_height - values[right_index - 1]) / (
        values[right_index] - values[right_index - 1]
    )
    right_crossing: float = float(
        omega[right_index - 1] + right_fraction * spacing
    )
    hwhm: float = 0.5 * (right_crossing - left_crossing)
    return peak_position, hwhm


class TestWp75ResolventDifferentiability:
    """Pin generic source, energy, self-energy, and regulator derivatives."""

    def test_generic_coordinates_match_central_differences(self) -> None:
        """Match reverse mode to central differences on every scalar input.

        The coordinate vector spans source, energy, self-energy, and regulator.

        Notes
        -----
        A vectorized centered stencil provides independent derivatives for all
        eight real coordinates of one generic complex solve.
        """
        hamiltonian: Complex128[Array, "2 2"] = jnp.asarray(
            [[-0.21, 0.13 + 0.09j], [0.13 - 0.09j, 0.34]],
            dtype=jnp.complex128,
        )
        coordinates: Float64[Array, " 8"] = jnp.asarray(
            [0.71, -0.33, 0.24, 0.58, 0.06, -0.014, 0.037, 0.002]
        )

        def loss(candidate: Float64[Array, " 8"]) -> Float64[Array, ""]:
            """Convert real coordinates to one generic complex solve."""
            source: Complex128[Array, " 2"] = jax.lax.complex(
                candidate[:2], candidate[2:4]
            )
            sigma: Complex128[Array, ""] = jax.lax.complex(
                candidate[5], -candidate[6]
            )
            return spectral_intensity_resolvent(
                hamiltonian,
                source[None, :],
                candidate[4],
                sigma,
                candidate[7],
            )

        gradient: Float64[Array, " 8"] = jax.grad(loss)(coordinates)
        step: float = 2.0**-14
        directions: Float64[Array, "8 8"] = jnp.eye(8, dtype=jnp.float64)
        finite_difference: Float64[Array, " 8"] = jax.vmap(
            lambda direction: (
                (
                    loss(coordinates + step * direction)
                    - loss(coordinates - step * direction)
                )
                / (2.0 * step)
            )
        )(directions)
        np.testing.assert_allclose(
            np.asarray(gradient),
            np.asarray(finite_difference),
            rtol=1.0e-6,
            atol=1.0e-8,
        )
        assert bool(jnp.all(jnp.abs(gradient) > 1.0e-8))


class TestWp75GaugeAndSumRules:
    """Pin exact-degeneracy gauge invariance and normalized spectral mass."""

    def test_degenerate_unitary_and_phase_invariance(self) -> None:
        """Preserve the observable under U(1) and degenerate U(2) gauges.

        The test rotates both the Hamiltonian basis and transition source.

        Notes
        -----
        The test compares resolvent values before and after the rotation. The
        explicit value-only eigen seam compares invariant weight sums.
        """
        eigenvalue: float = 0.17
        hamiltonian: Complex128[Array, "2 2"] = eigenvalue * jnp.eye(
            2, dtype=jnp.complex128
        )
        source: Complex128[Array, " 2"] = jnp.asarray(
            [0.8 + 0.3j, -0.2 + 0.7j]
        )
        unitary: Complex128[Array, "2 2"] = jnp.asarray(
            [[1.0, 1.0j], [1.0j, 1.0]], dtype=jnp.complex128
        ) / jnp.sqrt(2.0)
        transformed_hamiltonian: Complex128[Array, "2 2"] = (
            unitary.conj().T @ hamiltonian @ unitary
        )
        transformed_source: Complex128[Array, " 2"] = unitary.conj().T @ source
        phase: Complex128[Array, ""] = jnp.exp(0.731j)
        arguments: Tuple[Float64[Array, ""], Complex128[Array, ""], float] = (
            jnp.asarray(-0.04),
            jnp.asarray(-0.031j),
            2.0e-4,
        )
        baseline: Float64[Array, ""] = spectral_intensity_resolvent(
            hamiltonian, source[None, :], *arguments
        )
        rotated: Float64[Array, ""] = spectral_intensity_resolvent(
            transformed_hamiltonian, transformed_source[None, :], *arguments
        )
        phased: Float64[Array, ""] = spectral_intensity_resolvent(
            hamiltonian, (phase * source)[None, :], *arguments
        )
        np.testing.assert_allclose(
            np.asarray([rotated, phased]),
            np.asarray(baseline),
            rtol=1.0e-12,
            atol=1.0e-13,
        )

        eigenvalues: Float64[Array, " 2"] = jnp.full(2, eigenvalue)
        original_weights: Float64[Array, " 2"] = jnp.abs(source) ** 2
        rotated_weights: Float64[Array, " 2"] = (
            jnp.abs(transformed_source) ** 2
        )
        original_eigen: Float64[Array, ""] = spectral_intensity_eigen(
            eigenvalues,
            original_weights,
            *arguments,
            allow_degenerate_value_only=True,
        )
        rotated_eigen: Float64[Array, ""] = spectral_intensity_eigen(
            eigenvalues,
            rotated_weights,
            *arguments,
            allow_degenerate_value_only=True,
        )
        np.testing.assert_allclose(
            np.asarray(rotated_eigen),
            np.asarray(original_eigen),
            rtol=1.0e-12,
            atol=1.0e-13,
        )

    def test_full_line_mass_and_quadrature_refinement(self) -> None:
        """Match unit Lorentzian mass at both frozen tan-map orders.

        The test integrates five weighted production Lorentzians over R.

        Notes
        -----
        Gauss--Legendre nodes reach the full line through a tangent map. Both
        frozen orders recover each supplied projected weight.
        """
        centers: Float64[Array, " 5"] = jnp.asarray(
            [-0.07, -0.23, -0.8539392, 1.0539392, 0.10]
        )
        widths: Float64[Array, " 5"] = jnp.asarray(
            [0.0301, 0.0301, 0.0201, 0.0201, 0.0501]
        )
        projected_weights: Float64[Array, " 5"] = jnp.asarray(
            [0.685, 1.025, 1.78415201, 0.10584799, 1.0]
        )
        eta: float = 1.0e-4

        def masses(order: int) -> Float64[Array, " 5"]:
            """Integrate every row after a full-line tan map."""
            nodes_np: Float64[NDArray, " n_quad"]
            quadrature_np: Float64[NDArray, " n_quad"]
            nodes_np, quadrature_np = np.polynomial.legendre.leggauss(order)
            nodes: Float64[Array, " n"] = jnp.asarray(nodes_np)
            quadrature: Float64[Array, " n"] = jnp.asarray(quadrature_np)
            angle: Float64[Array, " n"] = 0.5 * jnp.pi * nodes

            def one(
                center: Float64[Array, ""],
                width: Float64[Array, ""],
                weight: Float64[Array, ""],
            ) -> Float64[Array, ""]:
                """Integrate one production Lorentzian row."""
                omega: Float64[Array, " n"] = center + width * jnp.tan(angle)
                jacobian: Float64[Array, " n"] = (
                    0.5 * jnp.pi * width / jnp.cos(angle) ** 2
                )
                sigma: Complex128[Array, ""] = jax.lax.complex(
                    jnp.asarray(0.0), -(width - eta)
                )
                intensity: Float64[Array, " n"] = jax.vmap(
                    spectral_intensity_eigen,
                    in_axes=(None, None, 0, None, None),
                )(
                    jnp.atleast_1d(center),
                    jnp.atleast_1d(weight),
                    omega,
                    sigma,
                    eta,
                )
                return jnp.sum(quadrature * intensity * jacobian)

            return jax.vmap(one)(centers, widths, projected_weights)

        coarse: Float64[Array, " 5"] = masses(256)
        fine: Float64[Array, " 5"] = masses(512)
        np.testing.assert_allclose(
            np.asarray(coarse),
            np.asarray(projected_weights),
            rtol=1.0e-8,
            atol=1.0e-12,
        )
        np.testing.assert_allclose(
            np.asarray(fine),
            np.asarray(projected_weights),
            rtol=1.0e-8,
            atol=1.0e-12,
        )
        assert float(jnp.max(jnp.abs(fine - coarse))) <= 1.0e-12

    def test_regulator_limit_acceptance_battery(self) -> None:
        """Enforce every frozen G5b mass, width, peak, and extrapolation gate.

        The test measures four decreasing regulator rungs on one dense grid.

        Notes
        -----
        Analytic captured mass, interpolated peak and HWHM, convergence ratios,
        and linear extrapolants jointly pin the preregistered limits.
        """
        omega: Float64[Array, " n"] = jnp.linspace(-1.0, 1.0, 8001)
        etas: Float64[Array, " 4"] = jnp.asarray(
            [8.0e-4, 4.0e-4, 2.0e-4, 1.0e-4]
        )
        center: float = 0.10
        physical_width: float = 0.05

        def row(eta: Float64[Array, ""]) -> Float64[Array, " n"]:
            """Evaluate one regulator rung on the frozen grid."""
            return jax.vmap(
                spectral_intensity_eigen,
                in_axes=(None, None, 0, None, None),
            )(
                jnp.asarray([center]),
                jnp.ones(1, dtype=jnp.float64),
                omega,
                jnp.asarray(-physical_width * 1.0j),
                eta,
            )

        intensities: Float64[Array, "4 n"] = jax.jit(jax.vmap(row))(etas)
        captured: Float64[Array, " 4"] = jnp.trapezoid(intensities, omega)
        widths: Float64[Array, " 4"] = physical_width + etas
        exact_captured: Float64[Array, " 4"] = (
            jnp.arctan((1.0 - center) / widths)
            - jnp.arctan((-1.0 - center) / widths)
        ) / jnp.pi
        np.testing.assert_allclose(
            np.asarray(captured),
            np.asarray(exact_captured),
            rtol=1.0e-10,
            atol=1.0e-9,
        )

        omega_np: Float64[NDArray, " n"] = np.asarray(omega)
        measurements: list[Tuple[float, float]] = [
            _quadratic_peak_and_hwhm(np.asarray(row_values), omega_np)
            for row_values in np.asarray(intensities)
        ]
        peak_positions: Float64[NDArray, " rung"] = np.asarray(
            [measurement[0] for measurement in measurements]
        )
        measured_hwhm: Float64[NDArray, " rung"] = np.asarray(
            [measurement[1] for measurement in measurements]
        )
        assert float(np.max(np.abs(peak_positions - center))) <= 2.0e-6

        eta_zero_mass: float = float(
            (
                np.arctan((1.0 - center) / physical_width)
                - np.arctan((-1.0 - center) / physical_width)
            )
            / np.pi
        )
        mass_errors: Float64[NDArray, " rung"] = np.abs(
            np.asarray(captured) - eta_zero_mass
        )
        hwhm_errors: Float64[NDArray, " rung"] = np.abs(
            measured_hwhm - physical_width
        )
        mass_ratios: Float64[NDArray, " rung_minus_one"] = (
            mass_errors[:-1] / mass_errors[1:]
        )
        hwhm_ratios: Float64[NDArray, " rung_minus_one"] = (
            hwhm_errors[:-1] / hwhm_errors[1:]
        )
        assert bool(np.all((mass_ratios >= 1.98) & (mass_ratios <= 2.02)))
        assert bool(np.all((hwhm_ratios >= 1.98) & (hwhm_ratios <= 2.02)))
        assert abs(float(captured[-1] - captured[-2])) <= 6.5e-5
        assert abs(float(measured_hwhm[-1] - measured_hwhm[-2])) <= 1.01e-4
        extrapolated_mass: float = float(2.0 * captured[-1] - captured[-2])
        extrapolated_hwhm: float = float(
            2.0 * measured_hwhm[-1] - measured_hwhm[-2]
        )
        assert abs(extrapolated_mass - eta_zero_mass) <= 2.0e-6
        assert abs(extrapolated_hwhm - physical_width) <= 2.0e-6
