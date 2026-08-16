# /// script
# requires-python = ">=3.12,<3.15"
# dependencies = ["diffpes==2026.06.13"]
# ///
"""Recover a planted constant self-energy linewidth from spectral profiles.

The automaton evaluates the public causal self-energy carrier and intrinsic
spectral-intensity function. It fits one linewidth to synthetic EDC and MDC
profiles. It writes the fit, covariance, linewidth comparison, and metrics.
Smoke mode uses compact profiles with no external data.
"""

from __future__ import annotations

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import optimistix as optx
from beartype import beartype
from beartype.typing import Any, Dict, List
from jaxtyping import Array, Float64, jaxtyped

import diffpes as dp


@jaxtyped(typechecker=beartype)
def _line_profiles_from_gamma(
    gamma_ev: Float64[Array, ""],
    energy_axis: Float64[Array, " n_energy"],
    momentum_axis: Float64[Array, " n_momentum"],
) -> Float64[Array, " n_profile"]:
    """PRIVATE: Build EDC and MDC values from a constant linewidth.

    Parameters
    ----------
    gamma_ev : Float64[Array, ""]
        Positive constant linewidth in eV.
    energy_axis : Float64[Array, " n_energy"]
        Relative-energy samples for the EDC in eV.
    momentum_axis : Float64[Array, " n_momentum"]
        Reduced momentum samples for the MDC.

    Returns
    -------
    profile : Float64[Array, " n_profile"]
        Concatenated EDC and MDC spectral intensities.

    Notes
    -----
    Evaluates the public self-energy model at each EDC energy. The MDC fixes
    energy at the Fermi level while its two-band dispersion changes with k.
    """
    self_energy: dp.types.SelfEnergyModel = dp.types.make_self_energy_model(
        gamma=gamma_ev,
        mode="constant",
    )
    edc_eigenvalues: Float64[Array, " 2"] = jnp.asarray(
        (-0.45, 0.32),
        dtype=jnp.float64,
    )
    band_weights: Float64[Array, " 2"] = jnp.asarray(
        (1.0, 0.55),
        dtype=jnp.float64,
    )
    edc_sigma: Any = dp.simul.evaluate_self_energy(energy_axis, self_energy)
    edc: Float64[Array, " n_energy"] = jax.vmap(
        lambda energy_value, sigma_value: dp.simul.spectral_intensity_eigen(
            edc_eigenvalues,
            band_weights,
            energy_value,
            sigma_value,
            0.003,
        )
    )(energy_axis, edc_sigma)
    fermi_axis: Float64[Array, " 1"] = jnp.zeros(
        (1,),
        dtype=jnp.float64,
    )
    mdc_sigma: Any = dp.simul.evaluate_self_energy(fermi_axis, self_energy)

    def mdc_value(momentum: Float64[Array, ""]) -> Float64[Array, ""]:
        """PRIVATE: Evaluate one Fermi-level MDC intensity value.

        Parameters
        ----------
        momentum : Float64[Array, ""]
            Reduced momentum coordinate.

        Returns
        -------
        intensity : Float64[Array, ""]
            Two-band intrinsic intensity at the Fermi level.

        Notes
        -----
        Moves the two dispersive poles with the reduced momentum coordinate.
        The public spectral evaluator retains the causal linewidth response.
        """
        mdc_eigenvalues: Float64[Array, " 2"] = jnp.asarray(
            (-0.45 + 0.50 * momentum, 0.32 + 0.10 * momentum),
            dtype=jnp.float64,
        )
        intensity: Float64[Array, ""] = dp.simul.spectral_intensity_eigen(
            mdc_eigenvalues,
            band_weights,
            fermi_axis[0],
            mdc_sigma[0],
            0.003,
        )
        return intensity

    mdc: Float64[Array, " n_momentum"] = jax.vmap(mdc_value)(momentum_axis)
    profile: Float64[Array, " n_profile"] = jnp.concatenate((edc, mdc))
    return profile


@dp.harness.experiment(
    name="fit-self-energy",
    params=(
        dp.types.make_automaton_param(
            "true_gamma_ev",
            float,
            default=0.075,
            help="Planted constant linewidth in eV.",
            bounds=(1.0e-4, 1.0),
            example=0.075,
        ),
        dp.types.make_automaton_param(
            "initial_gamma_ev",
            float,
            default=0.040,
            help="Initial constant linewidth in eV.",
            bounds=(1.0e-4, 1.0),
            example=0.040,
        ),
        dp.types.make_automaton_param(
            "mode",
            str,
            default="constant",
            help="Supported self-energy mode selector.",
            choices=("constant",),
            example="constant",
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
            "gamma_abs_error_ev": {"type": "number"},
            "coefficient_rel_error": {"type": "number"},
            "residual_rms": {"type": "number"},
            "converged": {"type": "boolean"},
        },
        "artifacts": {
            "roles": ["fit", "covariance", "linewidth_overlay", "metrics"]
        },
    },
)
def main(
    args: SimpleNamespace,
    ctx: dp.types.AutomatonContext,
) -> Dict[str, Any]:
    """Fit a constant linewidth and return causal-profile artifacts.

    The body fits EDC and MDC intensities from one public self-energy model.
    It checks the linewidth residual derivative against a central difference.
    """
    n_energy: int = 40 if args.smoke else 96
    n_momentum: int = 32 if args.smoke else 80
    energy_axis: Float64[Array, " n_energy"] = jnp.linspace(
        -1.2,
        0.5,
        n_energy,
        dtype=jnp.float64,
    )
    momentum_axis: Float64[Array, " n_momentum"] = jnp.linspace(
        -0.8,
        0.8,
        n_momentum,
        dtype=jnp.float64,
    )
    true_gamma: Float64[Array, ""] = jnp.asarray(
        args.true_gamma_ev,
        dtype=jnp.float64,
    )
    initial_gamma: Float64[Array, ""] = jnp.asarray(
        args.initial_gamma_ev,
        dtype=jnp.float64,
    )
    target_clean: Float64[Array, " n_profile"] = _line_profiles_from_gamma(
        true_gamma,
        energy_axis,
        momentum_axis,
    )
    target_noise: Float64[Array, " n_profile"] = args.noise_sigma * (
        jax.random.normal(ctx.rng_key, target_clean.shape, dtype=jnp.float64)
    )
    target: Float64[Array, " n_profile"] = target_clean + target_noise

    def residual(
        candidate_gamma: Float64[Array, ""],
        _: Any,
    ) -> Float64[Array, " n_profile"]:
        """PRIVATE: Return EDC and MDC residuals for one linewidth.

        Parameters
        ----------
        candidate_gamma : Float64[Array, ""]
            Candidate constant linewidth in eV.
        _ : Any
            Unused solver auxiliary argument.

        Returns
        -------
        residual_values : Float64[Array, " n_profile"]
            Model-minus-target EDC and MDC residuals.

        Notes
        -----
        Reuses the same public line-profile composition as the planted target.
        The shared path keeps the inversion and forward definitions aligned.
        """
        profile: Float64[Array, " n_profile"] = _line_profiles_from_gamma(
            candidate_gamma,
            energy_axis,
            momentum_axis,
        )
        residual_values: Float64[Array, " n_profile"] = profile - target
        return residual_values

    solver: optx.LevenbergMarquardt = optx.LevenbergMarquardt(
        rtol=args.rtol,
        atol=args.atol,
    )
    solution: Any = optx.least_squares(
        residual,
        solver,
        initial_gamma,
        max_steps=args.max_steps,
        throw=False,
    )
    fitted_gamma: Float64[Array, ""] = solution.value
    fitted_profile: Float64[Array, " n_profile"] = _line_profiles_from_gamma(
        fitted_gamma,
        energy_axis,
        momentum_axis,
    )
    residual_values: Float64[Array, " n_profile"] = residual(
        fitted_gamma,
        None,
    )
    jacobian: Float64[Array, " n_profile"] = jax.jacfwd(
        lambda coordinate: residual(coordinate, None)
    )(true_gamma)
    finite_difference_step: Float64[Array, ""] = 1.0e-5 * jnp.maximum(
        1.0,
        jnp.abs(true_gamma),
    )
    finite_difference: Float64[Array, " n_profile"] = (
        residual(true_gamma + finite_difference_step, None)
        - residual(true_gamma - finite_difference_step, None)
    ) / (2.0 * finite_difference_step)
    jacobian_fd_relative_error: Float64[Array, ""] = jnp.linalg.norm(
        jacobian - finite_difference
    ) / jnp.maximum(jnp.linalg.norm(finite_difference), 1.0e-12)
    residual_rms: Float64[Array, ""] = jnp.sqrt(jnp.mean(residual_values**2))
    covariance: Float64[Array, "1 1"] = jnp.asarray(
        [[jnp.maximum(jnp.mean(residual_values**2), 1.0e-24)]],
        dtype=jnp.float64,
    ) / jnp.maximum(jnp.sum(jacobian**2), 1.0e-24)
    gamma_abs_error: Float64[Array, ""] = jnp.abs(fitted_gamma - true_gamma)
    coefficient_rel_error: Float64[Array, ""] = gamma_abs_error / jnp.maximum(
        jnp.abs(true_gamma),
        1.0e-12,
    )
    jacobian_column_norm: Float64[Array, ""] = jnp.linalg.norm(jacobian)
    converged: bool = bool(
        (coefficient_rel_error < 1.0e-6)  # noqa: PLR2004
        & (residual_rms <= jnp.maximum(args.atol, 1.0e-12))
    )
    target_edc: Float64[Array, " n_energy"] = target[:n_energy]
    fitted_edc: Float64[Array, " n_energy"] = fitted_profile[:n_energy]
    overlay_figure: Any
    overlay_figure, _, _ = dp.plots.plot_curve_family(
        energy_axis,
        (target_edc, fitted_edc),
        labels=("planted EDC", "fitted EDC"),
        xlabel="relative energy (eV)",
        ylabel="intensity (1/eV)",
        title="Recovered constant linewidth",
    )
    metrics: Dict[str, Any] = {
        "gamma_abs_error_ev": float(gamma_abs_error),
        "coefficient_rel_error": float(coefficient_rel_error),
        "residual_rms": float(residual_rms),
        "n_steps": int(solution.stats["num_steps"]),
        "converged": converged,
        "jacobian_finite": bool(jnp.all(jnp.isfinite(jacobian))),
        "jacobian_min_column_norm": float(jacobian_column_norm),
        "jacobian_fd_relative_error": float(jacobian_fd_relative_error),
        "mode": args.mode,
    }
    fit: Dict[str, Any] = {
        "true_gamma_ev": true_gamma,
        "initial_gamma_ev": initial_gamma,
        "fitted_gamma_ev": fitted_gamma,
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
                "target_profile": target,
                "fitted_profile": fitted_profile,
            },
            role="covariance",
        ),
        dp.harness.save_figure_artifact(
            ctx,
            "linewidth_overlay.png",
            overlay_figure,
            role="linewidth_overlay",
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
