# /// script
# requires-python = ">=3.12,<3.15"
# dependencies = ["diffpes==2026.06.13"]
# ///
"""Recover planted polarization angles and an intensity-scale nuisance value.

The automaton forms a synthetic matrix-element intensity map from public
polarization and coherent contraction functions. It fits two incidence angles
with an explicit logarithmic intensity scale. It writes fit, covariance, map
overlay, and metrics artifacts. Smoke mode uses a compact in-code map.
"""

from __future__ import annotations

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import optimistix as optx
from beartype import beartype
from beartype.typing import Any, Dict, List
from jaxtyping import Array, Complex128, Float64, jaxtyped

import diffpes as dp


@jaxtyped(typechecker=beartype)
def _geometry_intensity(
    parameters: Float64[Array, " 3"],
    transition_channels: Complex128[Array, "n_x n_y 3"],
) -> Float64[Array, "n_x n_y"]:
    """PRIVATE: Build a coherent intensity map from geometry coordinates.

    Parameters
    ----------
    parameters : Float64[Array, " 3"]
        Incidence theta, incidence phi, and logarithmic intensity scale.
    transition_channels : Complex128[Array, "n_x n_y 3"]
        Synthetic Cartesian transition channels on the map grid.

    Returns
    -------
    intensity : Float64[Array, "n_x n_y"]
        Unresolved-spin matrix-element intensity map.

    Notes
    -----
    Builds a linear polarization vector from the two angles. It contracts the
    channels before the public final-spin intensity reduction. The logarithmic
    nuisance coordinate keeps the fitted scale strictly positive.
    """
    incidence_theta: Float64[Array, ""] = parameters[0]
    incidence_phi: Float64[Array, ""] = parameters[1]
    log_scale: Float64[Array, ""] = parameters[2]
    polarization: Complex128[Array, " 3"] = dp.simul.polarization_from_angles(
        incidence_theta,
        incidence_phi,
        "linear",
        polarization_angle=0.37,
    )
    amplitude: Complex128[Array, "n_x n_y"] = (
        dp.matrixel.contract_polarization(
            transition_channels,
            polarization,
        )
    )
    spin_amplitudes: Complex128[Array, "n_x n_y 2"] = jnp.stack(
        (amplitude, (0.38 + 0.21j) * amplitude),
        axis=-1,
    )
    intensity: Float64[Array, "n_x n_y"] = jnp.exp(log_scale) * (
        dp.matrixel.matrix_element_intensity(spin_amplitudes)
    )
    return intensity


@dp.harness.experiment(
    name="fit-experiment-geometry",
    params=(
        dp.types.make_automaton_param(
            "true_polarization_angles_rad",
            list,
            default=[0.62, 0.31],
            help="Planted incidence theta and phi angles in radians.",
            example=[0.62, 0.31],
        ),
        dp.types.make_automaton_param(
            "initial_angles_rad",
            list,
            default=[0.50, 0.16],
            help="Initial incidence theta and phi angles in radians.",
            example=[0.50, 0.16],
        ),
        dp.types.make_automaton_param(
            "noise_sigma",
            float,
            default=0.0,
            help="Additive intensity-map noise scale.",
            bounds=(0.0, 1.0),
            example=0.0,
        ),
        dp.types.make_automaton_param(
            "max_steps",
            int,
            default=128,
            help="Maximum Levenberg-Marquardt steps.",
            bounds=(1.0, 2048.0),
            example=128,
        ),
        dp.types.make_automaton_param(
            "rtol",
            float,
            default=1.0e-10,
            help="Relative solver tolerance.",
            bounds=(1.0e-14, 1.0e-2),
            example=1.0e-10,
        ),
        dp.types.make_automaton_param(
            "atol",
            float,
            default=1.0e-10,
            help="Absolute solver tolerance.",
            bounds=(1.0e-14, 1.0e-2),
            example=1.0e-10,
        ),
    ),
    returns={
        "fit": {"type": "object"},
        "metrics": {
            "angle_abs_error_rad": {"type": "number"},
            "scale_rel_error": {"type": "number"},
            "residual_rms": {"type": "number"},
            "converged": {"type": "boolean"},
        },
        "artifacts": {
            "roles": ["fit", "covariance", "geometry_overlay", "metrics"]
        },
    },
)
def main(  # noqa: PLR0915
    args: SimpleNamespace,
    ctx: dp.types.AutomatonContext,
) -> Dict[str, Any]:  # noqa: PLR0915
    """Fit polarization geometry and return the quotient-aware artifacts.

    The body fits a planted coherent intensity map with a scale nuisance term.
    It evaluates the public phase-gauge tangent.
    It checks all Jacobian columns.
    """
    if len(args.true_polarization_angles_rad) != 2:  # noqa: PLR2004
        message: str = "true_polarization_angles_rad must contain two values"
        raise ValueError(message)
    if len(args.initial_angles_rad) != 2:  # noqa: PLR2004
        message = "initial_angles_rad must contain two values"
        raise ValueError(message)
    n_x: int = 10 if args.smoke else 32
    n_y: int = 9 if args.smoke else 28
    x_axis: Float64[Array, " n_x"] = jnp.linspace(
        -1.0,
        1.0,
        n_x,
        dtype=jnp.float64,
    )
    y_axis: Float64[Array, " n_y"] = jnp.linspace(
        -0.8,
        0.8,
        n_y,
        dtype=jnp.float64,
    )
    x_grid: Float64[Array, "n_x n_y"]
    y_grid: Float64[Array, "n_x n_y"]
    x_grid, y_grid = jnp.meshgrid(x_axis, y_axis, indexing="ij")
    transition_channels: Complex128[Array, "n_x n_y 3"] = jnp.stack(
        (
            jnp.cos(1.3 * x_grid) + 0.22j * jnp.sin(y_grid),
            0.55 * jnp.sin(1.1 * y_grid) + 0.18j * jnp.cos(x_grid),
            0.30 + 0.25 * x_grid * y_grid + 0.10j * x_grid,
        ),
        axis=-1,
    ).astype(jnp.complex128)
    true_angles: Float64[Array, " 2"] = jnp.asarray(
        args.true_polarization_angles_rad,
        dtype=jnp.float64,
    )
    initial_angles: Float64[Array, " 2"] = jnp.asarray(
        args.initial_angles_rad,
        dtype=jnp.float64,
    )
    true_scale: Float64[Array, ""] = jnp.asarray(1.4, dtype=jnp.float64)
    initial_scale: Float64[Array, ""] = jnp.asarray(0.8, dtype=jnp.float64)
    true_parameters: Float64[Array, " 3"] = jnp.concatenate(
        (true_angles, jnp.log(true_scale)[None])
    )
    initial_parameters: Float64[Array, " 3"] = jnp.concatenate(
        (initial_angles, jnp.log(initial_scale)[None])
    )
    target_clean: Float64[Array, "n_x n_y"] = _geometry_intensity(
        true_parameters,
        transition_channels,
    )
    target_noise: Float64[Array, "n_x n_y"] = args.noise_sigma * (
        jax.random.normal(ctx.rng_key, target_clean.shape, dtype=jnp.float64)
    )
    target: Float64[Array, "n_x n_y"] = target_clean + target_noise

    def residual(
        candidate_parameters: Float64[Array, " 3"],
        _: Any,
    ) -> Float64[Array, " n_residual"]:
        """PRIVATE: Return map residuals for one geometry candidate.

        Parameters
        ----------
        candidate_parameters : Float64[Array, " 3"]
            Candidate angles and logarithmic intensity scale.
        _ : Any
            Unused solver auxiliary argument.

        Returns
        -------
        residual_values : Float64[Array, " n_residual"]
            Flattened coherent-intensity residuals.

        Notes
        -----
        Uses the same public polarization and matrix-element route as the
        planted map. The flattening exposes one residual per map pixel.
        """
        candidate_intensity: Float64[Array, "n_x n_y"] = _geometry_intensity(
            candidate_parameters,
            transition_channels,
        )
        residual_values: Float64[Array, " n_residual"] = jnp.ravel(
            candidate_intensity - target
        )
        return residual_values

    solver: optx.LevenbergMarquardt = optx.LevenbergMarquardt(
        rtol=args.rtol,
        atol=args.atol,
    )
    solution: Any = optx.least_squares(
        residual,
        solver,
        initial_parameters,
        max_steps=args.max_steps,
        throw=False,
    )
    fitted_parameters: Float64[Array, " 3"] = solution.value
    fitted_intensity: Float64[Array, "n_x n_y"] = _geometry_intensity(
        fitted_parameters,
        transition_channels,
    )
    residual_values: Float64[Array, " n_residual"] = residual(
        fitted_parameters,
        None,
    )
    jacobian: Float64[Array, "n_residual 3"] = jax.jacfwd(
        lambda coordinate: residual(coordinate, None)
    )(true_parameters)
    finite_difference_steps: Float64[Array, " 3"] = 1.0e-5 * jnp.maximum(
        1.0,
        jnp.abs(true_parameters),
    )
    theta_basis: Float64[Array, " 3"] = jnp.asarray(
        (finite_difference_steps[0], 0.0, 0.0),
        dtype=jnp.float64,
    )
    phi_basis: Float64[Array, " 3"] = jnp.asarray(
        (0.0, finite_difference_steps[1], 0.0),
        dtype=jnp.float64,
    )
    scale_basis: Float64[Array, " 3"] = jnp.asarray(
        (0.0, 0.0, finite_difference_steps[2]),
        dtype=jnp.float64,
    )
    theta_fd: Float64[Array, " n_residual"] = (
        residual(true_parameters + theta_basis, None)
        - residual(true_parameters - theta_basis, None)
    ) / (2.0 * finite_difference_steps[0])
    phi_fd: Float64[Array, " n_residual"] = (
        residual(true_parameters + phi_basis, None)
        - residual(true_parameters - phi_basis, None)
    ) / (2.0 * finite_difference_steps[1])
    scale_fd: Float64[Array, " n_residual"] = (
        residual(true_parameters + scale_basis, None)
        - residual(true_parameters - scale_basis, None)
    ) / (2.0 * finite_difference_steps[2])
    finite_difference: Float64[Array, "n_residual 3"] = jnp.stack(
        (theta_fd, phi_fd, scale_fd),
        axis=-1,
    )
    jacobian_fd_relative_error: Float64[Array, ""] = jnp.linalg.norm(
        jacobian - finite_difference
    ) / jnp.maximum(jnp.linalg.norm(finite_difference), 1.0e-12)
    jacobian_column_norms: Float64[Array, " 3"] = jnp.linalg.norm(
        jacobian,
        axis=0,
    )
    residual_rms: Float64[Array, ""] = jnp.sqrt(jnp.mean(residual_values**2))
    residual_variance: Float64[Array, ""] = jnp.maximum(
        jnp.mean(residual_values**2),
        1.0e-24,
    )
    covariance: Float64[Array, "3 3"] = residual_variance * jnp.linalg.pinv(
        jacobian.T @ jacobian
    )
    gauge_basis: dp.types.OrbitalBasis = dp.types.make_orbital_basis(
        atom_indices=(0, 0, 0, 0),
        n=(1, 2, 2, 2),
        l=(0, 1, 1, 1),
        m=(0, -1, 0, 1),
    )
    gauge_radial: dp.types.RadialSpec = dp.types.make_radial_spec(
        gauge_basis,
        (0, 1, 1, 1),
        zeta_shell=jnp.asarray(((1.2, 2.1), (0.9, 1.7))),
        coefficients_shell=jnp.asarray(((0.8, -0.3), (0.6, 0.4))),
    )
    gauge_parameters: dp.types.MatrixElementParams = (
        dp.types.make_matrix_element_params(
            gauge_basis,
            (0, 1, 1, 1),
            sigma_shell=jnp.asarray((1.3, 0.7)),
            phase_shift_angles_shell=jnp.asarray((0.2, -0.4, 0.6)),
        )
    )
    phase_gauge_direction: Float64[Array, " n_gauge"] = (
        dp.matrixel.matrix_element_phase_gauge_direction(
            gauge_radial,
            gauge_parameters,
            jnp.asarray(8.5, dtype=jnp.float64),
        )
    )
    fitted_angles: Float64[Array, " 2"] = fitted_parameters[:2]
    fitted_scale: Float64[Array, ""] = jnp.exp(fitted_parameters[2])
    angle_abs_error: Float64[Array, ""] = jnp.max(
        jnp.abs(fitted_angles - true_angles)
    )
    scale_rel_error: Float64[Array, ""] = (
        jnp.abs(fitted_scale - true_scale) / true_scale
    )
    converged: bool = bool(
        (angle_abs_error < 1.0e-5)  # noqa: PLR2004
        & (scale_rel_error < 1.0e-6)  # noqa: PLR2004
        & (residual_rms <= jnp.maximum(args.atol, 1.0e-12))
    )
    overlay_figure: Any
    overlay_figure, _, _ = dp.plots.plot_difference_map(
        target - fitted_intensity,
        x_axis,
        y_axis,
        title="Planted minus fitted geometry intensity",
    )
    metrics: Dict[str, Any] = {
        "angle_abs_error_rad": float(angle_abs_error),
        "scale_rel_error": float(scale_rel_error),
        "residual_rms": float(residual_rms),
        "n_steps": int(solution.stats["num_steps"]),
        "converged": converged,
        "jacobian_finite": bool(jnp.all(jnp.isfinite(jacobian))),
        "jacobian_min_column_norm": float(jnp.min(jacobian_column_norms)),
        "jacobian_fd_relative_error": float(jacobian_fd_relative_error),
        "phase_gauge_norm": float(jnp.linalg.norm(phase_gauge_direction)),
    }
    fit: Dict[str, Any] = {
        "true_angles_rad": true_angles,
        "initial_angles_rad": initial_angles,
        "fitted_angles_rad": fitted_angles,
        "true_scale": true_scale,
        "fitted_scale": fitted_scale,
        "phase_gauge_direction": phase_gauge_direction,
        "solver_result": str(solution.result),
    }
    artifacts: List[dp.types.ArtifactRecord] = [
        dp.harness.save_json_artifact(ctx, "fit.json", fit, role="fit"),
        dp.harness.save_array_artifact(
            ctx,
            "covariance.npz",
            {
                "covariance": covariance,
                "jacobian": jacobian,
                "finite_difference": finite_difference,
                "target_intensity": target,
                "fitted_intensity": fitted_intensity,
            },
            role="covariance",
        ),
        dp.harness.save_figure_artifact(
            ctx,
            "geometry_overlay.png",
            overlay_figure,
            role="geometry_overlay",
        ),
        dp.harness.save_json_artifact(ctx, "metrics.json", metrics),
    ]
    result: Dict[str, Any] = {
        "metrics": metrics,
        "artifacts": artifacts,
        "fit": fit,
    }
    return result


if __name__ == "__main__":
    main()
