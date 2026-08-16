# /// script
# requires-python = ">=3.12,<3.15"
# dependencies = ["diffpes==2026.06.13"]
# ///
"""Measure spectral convergence across compact momentum and energy ladders.

The automaton evaluates an intrinsic chain spectrum at paired resolutions. It
compares integrated occupied weight and an FWHM proxy with the finest sample.
It writes the residual curves, numerical arrays, and a metrics record. Smoke
mode uses at most three compact resolution levels.
"""

from __future__ import annotations

from types import SimpleNamespace

import jax.numpy as jnp
from beartype import beartype
from beartype.typing import Any, Dict, List, Tuple
from jaxtyping import Array, Float64, jaxtyped

import diffpes as dp


def _resolution_ladder(values: List[int], name: str) -> Tuple[int, ...]:
    """PRIVATE: Validate one ordered positive resolution ladder.

    Parameters
    ----------
    values : List[int]
        Requested resolution values.
    name : str
        Human-readable parameter name for failure messages.

    Returns
    -------
    ladder : Tuple[int, ...]
        Strictly increasing positive integer resolution values.

    Raises
    ------
    ValueError
        If the ladder has fewer than two levels or lacks strict ordering.

    Notes
    -----
    The host-side validation keeps JAX array shapes static and unambiguous.
    """
    minimum_levels: int = 2
    minimum_resolution: int = 4
    ladder: Tuple[int, ...] = tuple(int(value) for value in values)
    if len(ladder) < minimum_levels:
        message: str = f"{name} must contain at least two levels"
        raise ValueError(message)
    if any(value < minimum_resolution for value in ladder):
        message = f"{name} values must be at least four"
        raise ValueError(message)
    if any(
        left >= right
        for left, right in zip(ladder[:-1], ladder[1:], strict=True)
    ):
        message = f"{name} must be strictly increasing"
        raise ValueError(message)
    return ladder


@jaxtyped(typechecker=beartype)
def _spectral_observables(
    n_k: int,
    n_energy: int,
    window_ev: float,
) -> Tuple[Float64[Array, ""], Float64[Array, ""]]:
    """PRIVATE: Compute two intrinsic observables at one resolution.

    Parameters
    ----------
    n_k : int
        Number of chain momentum samples.
    n_energy : int
        Number of relative-energy samples.
    window_ev : float
        Half-width of the occupied integration window in eV.

    Returns
    -------
    observables : Tuple[Float64[Array, ""], Float64[Array, ""]]
        Integrated intensity and a sampled MDC FWHM proxy.

    Notes
    -----
    The function composes the public chain model, eigenvalue path, and
    intrinsic spectral assembler. The sampled half-maximum width avoids a
    non-differentiable fitted peak model during this operational check.
    """
    model: dp.types.TBModel = dp.harness.linear_chain_model()
    momentum_axis: Float64[Array, " n_k"] = jnp.linspace(
        0.0,
        0.5,
        n_k,
        dtype=jnp.float64,
    )
    kpoints: Float64[Array, "n_k 3"] = jnp.stack(
        (
            momentum_axis,
            jnp.zeros_like(momentum_axis),
            jnp.zeros_like(momentum_axis),
        ),
        axis=1,
    )
    energy_axis: Float64[Array, " n_energy"] = jnp.linspace(
        -2.0,
        0.2,
        n_energy,
        dtype=jnp.float64,
    )
    eigenvalues: Float64[Array, "n_k n_band"] = dp.tightb.eigvalsh_bands(
        model,
        kpoints,
    )
    band_weights: Float64[Array, "n_k n_energy n_band"] = jnp.broadcast_to(
        jnp.ones_like(eigenvalues)[:, None, :],
        (n_k, n_energy, eigenvalues.shape[1]),
    )
    self_energy: dp.types.SelfEnergyModel = dp.types.make_self_energy_model(
        gamma=0.06,
    )
    intensity: Float64[Array, "n_k n_energy"] = (
        dp.simul.assemble_spectral_intensity_bands_chunk(
            eigenvalues,
            band_weights,
            energy_axis,
            self_energy,
            jnp.asarray(0.0, dtype=jnp.float64),
            30.0,
            allow_degenerate_value_only=True,
        )
    )
    energy_step: Float64[Array, ""] = energy_axis[1] - energy_axis[0]
    in_window: Float64[Array, " n_energy"] = jnp.asarray(
        jnp.abs(energy_axis) <= window_ev,
        dtype=jnp.float64,
    )
    integrated_intensity: Float64[Array, ""] = (
        energy_step * jnp.sum(intensity * in_window[None, :]) / n_k
    )
    fermi_index: int = int(jnp.argmin(jnp.abs(energy_axis)))
    mdc: Float64[Array, " n_k"] = intensity[:, fermi_index]
    half_maximum: Float64[Array, ""] = 0.5 * jnp.max(mdc)
    momentum_step: Float64[Array, ""] = momentum_axis[1] - momentum_axis[0]
    fwhm_proxy: Float64[Array, ""] = momentum_step * jnp.sum(
        jnp.asarray(mdc >= half_maximum, dtype=jnp.float64)
    )
    observables: Tuple[Float64[Array, ""], Float64[Array, ""]] = (
        integrated_intensity,
        fwhm_proxy,
    )
    return observables


@dp.harness.experiment(
    name="convergence-study",
    params=(
        dp.types.make_automaton_param(
            "n_k_ladder",
            list,
            default=[8, 16, 24],
            help="Increasing momentum sample counts.",
            example=[8, 16, 24],
        ),
        dp.types.make_automaton_param(
            "n_energy_ladder",
            list,
            default=[16, 32, 48],
            help="Increasing energy sample counts.",
            example=[16, 32, 48],
        ),
        dp.types.make_automaton_param(
            "window_ev",
            float,
            default=0.25,
            help="Half-width of the occupied energy window in eV.",
            bounds=(0.01, 1.0),
            example=0.25,
        ),
    ),
    returns={
        "metrics": {
            "residuals": {"type": "object"},
            "converged_level": {"type": "integer"},
            "monotone": {"type": "boolean"},
        },
        "artifacts": {
            "roles": [
                "convergence_curves",
                "convergence_arrays",
                "metrics",
            ]
        },
    },
)
def main(
    args: SimpleNamespace,
    ctx: dp.types.AutomatonContext,
) -> Dict[str, Any]:
    """Measure intrinsic resolution convergence and return diagnostics.

    The body evaluates paired resolution levels against the finest result. It
    writes residual curves and identifies the first level below a tight error.
    """
    n_k_ladder: Tuple[int, ...] = _resolution_ladder(
        list(args.n_k_ladder),
        "n_k_ladder",
    )
    n_energy_ladder: Tuple[int, ...] = _resolution_ladder(
        list(args.n_energy_ladder),
        "n_energy_ladder",
    )
    n_levels: int = min(len(n_k_ladder), len(n_energy_ladder))
    if args.smoke:
        n_levels = min(n_levels, 3)
    selected_k: Tuple[int, ...] = n_k_ladder[:n_levels]
    selected_energy: Tuple[int, ...] = n_energy_ladder[:n_levels]
    integrated_values_list: List[float] = []
    fwhm_values_list: List[float] = []
    n_k: int
    n_energy: int
    for n_k, n_energy in zip(selected_k, selected_energy, strict=True):
        integrated_value: Float64[Array, ""]
        fwhm_value: Float64[Array, ""]
        integrated_value, fwhm_value = _spectral_observables(
            n_k,
            n_energy,
            args.window_ev,
        )
        integrated_values_list.append(float(integrated_value))
        fwhm_values_list.append(float(fwhm_value))
    integrated_values: Float64[Array, " n_level"] = jnp.asarray(
        integrated_values_list,
        dtype=jnp.float64,
    )
    fwhm_values: Float64[Array, " n_level"] = jnp.asarray(
        fwhm_values_list,
        dtype=jnp.float64,
    )
    integrated_residuals: Float64[Array, " n_level"] = jnp.abs(
        integrated_values - integrated_values[-1]
    )
    fwhm_residuals: Float64[Array, " n_level"] = jnp.abs(
        fwhm_values - fwhm_values[-1]
    )
    monotonicity_tolerance: float = 1.0e-12
    monotone_integrated: bool = bool(
        jnp.all(jnp.diff(integrated_residuals) <= monotonicity_tolerance)
    )
    monotone_fwhm: bool = bool(
        jnp.all(jnp.diff(fwhm_residuals) <= monotonicity_tolerance)
    )
    monotone: bool = monotone_integrated and monotone_fwhm
    normalized_integrated: Float64[Array, " n_level"] = (
        integrated_residuals
        / jnp.maximum(jnp.max(integrated_residuals), 1.0e-15)
    )
    normalized_fwhm: Float64[Array, " n_level"] = fwhm_residuals / jnp.maximum(
        jnp.max(fwhm_residuals),
        1.0e-15,
    )
    level_axis: Float64[Array, " n_level"] = jnp.arange(
        n_levels,
        dtype=jnp.float64,
    )
    figure: Any
    figure, _, _ = dp.plots.plot_curve_family(
        level_axis,
        (normalized_integrated, normalized_fwhm),
        labels=("integrated intensity", "MDC FWHM"),
        xlabel="resolution level",
        ylabel="normalized residual",
        title="Intrinsic spectral convergence",
    )
    convergence_tolerance: float = 5.0e-3
    converged_level: int = n_levels - 1
    level_index: int
    for level_index in range(n_levels):
        if (
            float(integrated_residuals[level_index]) <= convergence_tolerance
            and float(fwhm_residuals[level_index]) <= convergence_tolerance
        ):
            converged_level = level_index
            break
    metrics: Dict[str, Any] = {
        "residuals": {
            "integrated_intensity": [
                float(value) for value in integrated_residuals
            ],
            "mdc_fwhm": [float(value) for value in fwhm_residuals],
        },
        "converged_level": converged_level,
        "monotone": monotone,
        "n_k_ladder": list(selected_k),
        "n_energy_ladder": list(selected_energy),
    }
    artifacts: List[dp.types.ArtifactRecord] = [
        dp.harness.save_figure_artifact(
            ctx,
            "convergence.png",
            figure,
            role="convergence_curves",
        ),
        dp.harness.save_array_artifact(
            ctx,
            "convergence.npz",
            {
                "n_k": selected_k,
                "n_energy": selected_energy,
                "integrated_intensity": integrated_values,
                "mdc_fwhm": fwhm_values,
                "integrated_residual": integrated_residuals,
                "mdc_fwhm_residual": fwhm_residuals,
            },
            role="convergence_arrays",
        ),
        dp.harness.save_json_artifact(ctx, "metrics.json", metrics),
    ]
    result: Dict[str, Any] = {"metrics": metrics, "artifacts": artifacts}
    return result


if __name__ == "__main__":
    main()
