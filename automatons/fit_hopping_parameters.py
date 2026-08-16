# /// script
# requires-python = ">=3.12,<3.15"
# dependencies = ["diffpes==2026.06.13"]
# ///
"""Recover a planted graphene hopping from an intrinsic spectral cut.

The automaton creates a graphene reference model and obtains its independent
tight-binding coordinates through the public parameter view. It fits one
shared nearest-neighbor coordinate against a synthetic spectral cut. It writes
the fit, covariance, overlay, and metrics artifacts. Smoke mode uses four
nondegenerate momenta and 48 energy samples.
"""

from __future__ import annotations

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import optimistix as optx
from beartype import beartype
from beartype.typing import Any, Callable, Dict, List
from jaxtyping import Array, Float64, jaxtyped

import diffpes as dp


@jaxtyped(typechecker=beartype)
def _spectral_cut_from_hopping(
    hopping_ev: Float64[Array, ""],
    packed_parameters: Float64[Array, " n_parameters"],
    rebuild: Callable[[Float64[Array, " n_parameters"]], dp.types.TBModel],
    kpoints: Float64[Array, "n_k 3"],
    energy_axis: Float64[Array, " n_energy"],
) -> Float64[Array, "n_k n_energy"]:
    """PRIVATE: Build one intrinsic spectral cut from a shared hopping.

    Parameters
    ----------
    hopping_ev : Float64[Array, ""]
        Shared nearest-neighbor hopping coordinate in eV.
    packed_parameters : Float64[Array, " n_parameters"]
        Independent coordinates from ``tb_parameter_view``.
    rebuild : Callable
        Public parameter-view closure that reconstructs a TB model.
    kpoints : Float64[Array, "n_k 3"]
        Fixed nondegenerate fractional momentum samples.
    energy_axis : Float64[Array, " n_energy"]
        Relative-energy samples in eV.

    Returns
    -------
    intensity : Float64[Array, "n_k n_energy"]
        Intrinsic spectral intensity in inverse eV.

    Notes
    -----
    Replaces the three independent real hopping representatives with one
    shared coordinate. The public reconstruction closure restores Hermitian
    partners before the public spectral assembler evaluates the cut.
    """
    candidate_parameters: Float64[Array, " n_parameters"] = (
        packed_parameters.at[0]
        .set(hopping_ev)
        .at[2]
        .set(hopping_ev)
        .at[4]
        .set(hopping_ev)
    )
    model: dp.types.TBModel = rebuild(candidate_parameters)
    eigenvalues: Float64[Array, "n_k n_bands"] = dp.tightb.eigvalsh_bands(
        model,
        kpoints,
    )
    band_weights: Float64[Array, "n_k n_energy n_bands"] = jnp.broadcast_to(
        jnp.ones_like(eigenvalues)[:, None, :],
        (eigenvalues.shape[0], energy_axis.shape[0], eigenvalues.shape[1]),
    )
    self_energy: dp.types.SelfEnergyModel = dp.types.make_self_energy_model(
        gamma=0.25,
    )
    intensity: Float64[Array, "n_k n_energy"] = (
        dp.simul.assemble_spectral_intensity_bands_chunk(
            eigenvalues,
            band_weights,
            energy_axis,
            self_energy,
            jnp.asarray(0.0, dtype=jnp.float64),
            30.0,
        )
    )
    return intensity


@dp.harness.experiment(
    name="fit-hopping-parameters",
    params=(
        dp.types.make_automaton_param(
            "true_hopping_ev",
            float,
            default=-2.7,
            help="Planted nearest-neighbor hopping in eV.",
            bounds=(-10.0, -0.01),
            example=-2.7,
        ),
        dp.types.make_automaton_param(
            "initial_hopping_ev",
            float,
            default=-2.1,
            help="Initial nearest-neighbor hopping in eV.",
            bounds=(-10.0, -0.01),
            example=-2.1,
        ),
        dp.types.make_automaton_param(
            "noise_sigma",
            float,
            default=0.0,
            help="Additive spectral noise scale in inverse eV.",
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
            "hopping_abs_error_ev": {"type": "number"},
            "hopping_rel_error": {"type": "number"},
            "residual_rms": {"type": "number"},
            "n_steps": {"type": "integer"},
            "converged": {"type": "boolean"},
        },
        "artifacts": {
            "roles": ["fit", "covariance", "fit_overlay", "metrics"]
        },
    },
)
def main(
    args: SimpleNamespace,
    ctx: dp.types.AutomatonContext,
) -> Dict[str, Any]:
    """Fit one graphene hopping coordinate and return inversion artifacts.

    The body fits a planted spectral cut with Levenberg-Marquardt. It checks
    the scalar residual derivative against a central finite difference.
    """
    n_energy: int = 48 if args.smoke else 96
    kpoints: Float64[Array, "4 3"] = jnp.asarray(
        (
            (0.113, 0.217, 0.0),
            (0.287, 0.143, 0.0),
            (0.371, 0.411, 0.0),
            (0.190, 0.350, 0.0),
        ),
        dtype=jnp.float64,
    )
    energy_axis: Float64[Array, " n_energy"] = jnp.linspace(
        -7.0,
        7.0,
        n_energy,
        dtype=jnp.float64,
    )
    initial_model: dp.types.TBModel = dp.harness.graphene_pz_model(
        hopping_ev=args.initial_hopping_ev,
    )
    packed_parameters: Float64[Array, " n_parameters"]
    rebuild: Callable[[Float64[Array, " n_parameters"]], dp.types.TBModel]
    packed_parameters, rebuild = dp.tightb.tb_parameter_view(initial_model)
    true_hopping: Float64[Array, ""] = jnp.asarray(
        args.true_hopping_ev,
        dtype=jnp.float64,
    )
    initial_hopping: Float64[Array, ""] = jnp.asarray(
        args.initial_hopping_ev,
        dtype=jnp.float64,
    )
    target_clean: Float64[Array, "4 n_energy"] = _spectral_cut_from_hopping(
        true_hopping,
        packed_parameters,
        rebuild,
        kpoints,
        energy_axis,
    )
    target_noise: Float64[Array, "4 n_energy"] = args.noise_sigma * (
        jax.random.normal(ctx.rng_key, target_clean.shape, dtype=jnp.float64)
    )
    target: Float64[Array, "4 n_energy"] = target_clean + target_noise

    def residual(
        candidate_hopping: Float64[Array, ""],
        _: Any,
    ) -> Float64[Array, " n_residual"]:
        """PRIVATE: Return spectral residuals for one hopping candidate.

        Parameters
        ----------
        candidate_hopping : Float64[Array, ""]
            Candidate shared hopping coordinate in eV.
        _ : Any
            Unused solver auxiliary argument.

        Returns
        -------
        residual_values : Float64[Array, " n_residual"]
            Flattened target-minus-model spectral residuals.

        Notes
        -----
        Calls the public spectral-cut composition for the candidate coordinate.
        Flattening gives the least-squares solver one residual vector.
        """
        candidate_intensity: Float64[Array, "4 n_energy"] = (
            _spectral_cut_from_hopping(
                candidate_hopping,
                packed_parameters,
                rebuild,
                kpoints,
                energy_axis,
            )
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
        initial_hopping,
        max_steps=args.max_steps,
        throw=False,
    )
    fitted_hopping: Float64[Array, ""] = solution.value
    fitted_intensity: Float64[Array, "4 n_energy"] = (
        _spectral_cut_from_hopping(
            fitted_hopping,
            packed_parameters,
            rebuild,
            kpoints,
            energy_axis,
        )
    )
    residual_values: Float64[Array, " n_residual"] = residual(
        fitted_hopping,
        None,
    )
    jacobian: Float64[Array, " n_residual"] = jax.jacfwd(
        lambda coordinate: residual(coordinate, None)
    )(true_hopping)
    finite_difference_step: Float64[Array, ""] = 1.0e-5 * jnp.maximum(
        1.0,
        jnp.abs(true_hopping),
    )
    finite_difference: Float64[Array, " n_residual"] = (
        residual(true_hopping + finite_difference_step, None)
        - residual(true_hopping - finite_difference_step, None)
    ) / (2.0 * finite_difference_step)
    jacobian_fd_relative_error: Float64[Array, ""] = jnp.linalg.norm(
        jacobian - finite_difference
    ) / jnp.maximum(jnp.linalg.norm(finite_difference), 1.0e-12)
    residual_rms: Float64[Array, ""] = jnp.sqrt(jnp.mean(residual_values**2))
    covariance: Float64[Array, "1 1"] = jnp.asarray(
        [[jnp.maximum(jnp.mean(residual_values**2), 1.0e-24)]],
        dtype=jnp.float64,
    ) / jnp.maximum(jnp.sum(jacobian**2), 1.0e-24)
    fitted_parameters: Float64[Array, " n_parameters"] = (
        packed_parameters.at[0]
        .set(fitted_hopping)
        .at[2]
        .set(fitted_hopping)
        .at[4]
        .set(fitted_hopping)
    )
    fitted_model: dp.types.TBModel = rebuild(fitted_parameters)
    fitted_bands: Float64[Array, "4 n_bands"] = dp.tightb.eigvalsh_bands(
        fitted_model,
        kpoints,
    )
    momentum_axis: Float64[Array, " 4"] = jnp.arange(
        kpoints.shape[0],
        dtype=jnp.float64,
    )
    overlay_figure: Any
    overlay_figure, _, _ = dp.plots.plot_bands_over_spectrum(
        target,
        momentum_axis,
        energy_axis,
        fitted_bands,
        title="Recovered graphene hopping",
    )
    hopping_abs_error: Float64[Array, ""] = jnp.abs(
        fitted_hopping - true_hopping
    )
    hopping_rel_error: Float64[Array, ""] = hopping_abs_error / jnp.maximum(
        jnp.abs(true_hopping),
        1.0e-12,
    )
    jacobian_column_norm: Float64[Array, ""] = jnp.linalg.norm(jacobian)
    converged: bool = bool(
        (hopping_rel_error < 1.0e-6)  # noqa: PLR2004
        & (residual_rms <= jnp.maximum(args.atol, 1.0e-12))
    )
    metrics: Dict[str, Any] = {
        "hopping_abs_error_ev": float(hopping_abs_error),
        "hopping_rel_error": float(hopping_rel_error),
        "residual_rms": float(residual_rms),
        "n_steps": int(solution.stats["num_steps"]),
        "converged": converged,
        "jacobian_finite": bool(jnp.all(jnp.isfinite(jacobian))),
        "jacobian_min_column_norm": float(jacobian_column_norm),
        "jacobian_fd_relative_error": float(jacobian_fd_relative_error),
    }
    fit: Dict[str, Any] = {
        "true_hopping_ev": true_hopping,
        "initial_hopping_ev": initial_hopping,
        "fitted_hopping_ev": fitted_hopping,
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
            "fit_overlay.png",
            overlay_figure,
            role="fit_overlay",
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
