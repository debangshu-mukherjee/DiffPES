# /// script
# requires-python = ">=3.12,<3.15"
# dependencies = ["diffpes==2026.06.13"]
# ///
"""Match a measured spectral cut against simulated candidates.

This experiment loads NPZ measurement data or synthesizes a graphene cut. It
compares candidate linewidth and temperature pairs with normalized correlation,
mean-square error, and Poisson chi-square. It writes the best residual and a
machine-readable candidate table.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
from beartype import beartype
from beartype.typing import Any, Dict, List, Tuple
from jaxtyping import Array, Float64, jaxtyped

import diffpes as dp


@jaxtyped(typechecker=beartype)
def _synthetic_cut(
    gamma_ev: dp.types.ScalarFloat,
    temperature_k: dp.types.ScalarFloat,
    n_k: int,
    n_energy: int,
) -> Tuple[
    Float64[Array, " n_k n_energy"],
    Float64[Array, " n_k"],
    Float64[Array, " n_energy"],
]:
    """PRIVATE: Build one graphene spectral cut for a candidate pair.

    Parameters
    ----------
    gamma_ev : dp.types.ScalarFloat
        Constant linewidth in eV.
    temperature_k : dp.types.ScalarFloat
        Fermi occupation temperature in kelvin.
    n_k : int
        Number of momentum path points.
    n_energy : int
        Number of energy samples.

    Returns
    -------
    intensity : Float64[Array, " n_k n_energy"]
        Occupied graphene spectral intensity.
    k_axis : Float64[Array, " n_k"]
        Cumulative Cartesian path distance in inverse angstroms.
    energy_axis : Float64[Array, " n_energy"]
        Relative energy coordinates in eV.

    Notes
    -----
    Calls the public reference model, tight-binding diagonalizer, and spectral
    assembler. The two candidate parameters enter the shared forward path.
    """
    model: dp.types.TBModel = dp.harness.graphene_pz_model()
    path_fractional: Float64[Array, " n_k 3"] = jnp.linspace(
        jnp.asarray([0.28, 1.0 / 3.0, 0.0], dtype=jnp.float64),
        jnp.asarray([0.39, 1.0 / 3.0, 0.0], dtype=jnp.float64),
        n_k,
    )
    kpoints: Float64[Array, " n_k 3"] = dp.tightb.kpoints_frac_to_cart(
        path_fractional,
        model.geometry,
    )
    k_axis: Float64[Array, " n_k"] = jnp.linalg.norm(
        kpoints - kpoints[0],
        axis=1,
    )
    energy_axis: Float64[Array, " n_energy"] = jnp.linspace(
        -0.25,
        0.10,
        n_energy,
        dtype=jnp.float64,
    )
    eigenvalues: Float64[Array, " n_k n_bands"] = dp.tightb.eigvalsh_bands(
        model,
        path_fractional,
    )
    band_weights: Float64[Array, " n_k n_energy n_bands"] = jnp.broadcast_to(
        jnp.ones_like(eigenvalues)[:, None, :],
        (n_k, n_energy, eigenvalues.shape[1]),
    )
    self_energy: dp.types.SelfEnergyModel = dp.types.make_self_energy_model(
        gamma=gamma_ev,
    )
    intensity: Float64[Array, " n_k n_energy"] = (
        dp.simul.assemble_spectral_intensity_bands_chunk(
            eigenvalues,
            band_weights,
            energy_axis,
            self_energy,
            jnp.asarray(0.0, dtype=jnp.float64),
            temperature_k,
            allow_degenerate_value_only=True,
        )
    )
    return intensity, k_axis, energy_axis


@jaxtyped(typechecker=beartype)
def _comparison_metrics(
    measured: Float64[Array, " n_k n_energy"],
    simulated: Float64[Array, " n_k n_energy"],
) -> Tuple[Float64[Array, ""], Float64[Array, ""], Float64[Array, ""]]:
    """PRIVATE: Compute three normalized comparison statistics.

    Parameters
    ----------
    measured : Float64[Array, " n_k n_energy"]
        Measurement intensity on shared axes.
    simulated : Float64[Array, " n_k n_energy"]
        Candidate intensity on shared axes.

    Returns
    -------
    ncc : Float64[Array, ""]
        Normalized cross-correlation coefficient.
    mse : Float64[Array, ""]
        Mean-square difference after sum normalization.
    chi_square : Float64[Array, ""]
        Poisson chi-square per degree of freedom.

    Notes
    -----
    Divides each map by its complete intensity sum. This compares shapes while
    preserving a Poisson-weighted score for the grid ranking.
    """
    measured_sum: Float64[Array, ""] = jnp.maximum(jnp.sum(measured), 1.0e-30)
    simulated_sum: Float64[Array, ""] = jnp.maximum(
        jnp.sum(simulated),
        1.0e-30,
    )
    measured_normalized: Float64[Array, " n_k n_energy"] = (
        measured / measured_sum
    )
    simulated_normalized: Float64[Array, " n_k n_energy"] = (
        simulated / simulated_sum
    )
    measured_centered: Float64[Array, " n_k n_energy"] = (
        measured_normalized - jnp.mean(measured_normalized)
    )
    simulated_centered: Float64[Array, " n_k n_energy"] = (
        simulated_normalized - jnp.mean(simulated_normalized)
    )
    ncc_denominator: Float64[Array, ""] = jnp.maximum(
        jnp.linalg.norm(measured_centered)
        * jnp.linalg.norm(simulated_centered),
        1.0e-30,
    )
    ncc: Float64[Array, ""] = (
        jnp.sum(measured_centered * simulated_centered) / ncc_denominator
    )
    residual: Float64[Array, " n_k n_energy"] = (
        measured_normalized - simulated_normalized
    )
    mse: Float64[Array, ""] = jnp.mean(residual**2)
    chi_square: Float64[Array, ""] = jnp.sum(
        residual**2 / jnp.maximum(simulated_normalized, 1.0e-18)
    ) / float(residual.size - 1)
    result: Tuple[
        Float64[Array, ""],
        Float64[Array, ""],
        Float64[Array, ""],
    ] = (
        ncc,
        mse,
        chi_square,
    )
    return result


@dp.harness.experiment(
    name="match-measured-to-simulated",
    params=[
        dp.types.make_automaton_param(
            "measured_path",
            str,
            default="",
            help="NPZ measurement path. Empty builds a planted graphene cut.",
            example="measurement.npz",
        ),
        dp.types.make_automaton_param(
            "gamma_grid",
            list,
            default=[0.015, 0.025, 0.040],
            help="Candidate constant linewidth values in eV.",
            example=[0.015, 0.025, 0.040],
        ),
        dp.types.make_automaton_param(
            "temperature_grid",
            list,
            default=[30.0, 55.0, 90.0],
            help="Candidate Fermi temperatures in kelvin.",
            example=[30.0, 55.0, 90.0],
        ),
    ],
    returns={
        "metrics": {
            "best_gamma_ev": {"type": "number"},
            "best_temperature_k": {"type": "number"},
            "best_ncc": {"type": "number"},
            "best_chi2_per_dof": {"type": "number"},
        },
        "artifacts": {
            "roles": ["residual_map", "match_table", "match_arrays"],
        },
    },
)
def main(  # noqa: PLR0915 -- direct grid values preserve result provenance.
    args: SimpleNamespace,
    ctx: dp.types.AutomatonContext,
) -> Dict[str, Any]:
    """Run the candidate comparison and return the selected match."""
    gamma_grid: Tuple[Any, ...] = tuple(args.gamma_grid)
    temperature_grid: Tuple[Any, ...] = tuple(args.temperature_grid)
    if not gamma_grid or not temperature_grid:
        raise ValueError("gamma_grid and temperature_grid must be nonempty")
    maximum_grid_size: int = 3
    if (
        len(gamma_grid) > maximum_grid_size
        or len(temperature_grid) > maximum_grid_size
    ):
        raise ValueError("candidate grids may contain at most three values")
    planted_gamma_ev: float | None
    planted_temperature_k: float | None
    if args.measured_path:
        source_path: Path = Path(args.measured_path)
        if source_path.suffix != ".npz":
            raise ValueError("measured_path must end in .npz")
        with np.load(source_path, allow_pickle=False) as archive:
            measured: Float64[Array, " n_k n_energy"] = jnp.asarray(
                archive["intensity"],
                dtype=jnp.float64,
            )
            k_axis: Float64[Array, " n_k"] = jnp.asarray(
                archive["k_axis"],
                dtype=jnp.float64,
            )
            energy_axis: Float64[Array, " n_energy"] = jnp.asarray(
                archive["energy_axis"],
                dtype=jnp.float64,
            )
        planted_gamma_ev = None
        planted_temperature_k = None
    else:
        n_k: int = 18 if args.smoke else 72
        n_energy: int = 36 if args.smoke else 144
        planted_gamma_ev = 0.025
        planted_temperature_k = 55.0
        measured, k_axis, energy_axis = _synthetic_cut(
            planted_gamma_ev,
            planted_temperature_k,
            n_k,
            n_energy,
        )
    best_gamma_ev: float = 0.0
    best_temperature_k: float = 0.0
    best_ncc: float = -1.0
    best_chi_square: float = float("inf")
    best_simulated: Float64[Array, " n_k n_energy"] = jnp.zeros_like(measured)
    table: List[Dict[str, float]] = []
    gamma_value: Any
    temperature_value: Any
    for gamma_value in gamma_grid:
        for temperature_value in temperature_grid:
            simulated: Float64[Array, " n_k n_energy"]
            _, _, _ = k_axis, energy_axis, measured
            simulated, _, _ = _synthetic_cut(
                float(gamma_value),
                float(temperature_value),
                measured.shape[0],
                measured.shape[1],
            )
            ncc: Float64[Array, ""]
            mse: Float64[Array, ""]
            chi_square: Float64[Array, ""]
            ncc, mse, chi_square = _comparison_metrics(measured, simulated)
            ncc_value: float = float(ncc)
            mse_value: float = float(mse)
            chi_square_value: float = float(chi_square)
            row: Dict[str, float] = {
                "gamma_ev": float(gamma_value),
                "temperature_k": float(temperature_value),
                "ncc": ncc_value,
                "mse": mse_value,
                "chi2_per_dof": chi_square_value,
            }
            table.append(row)
            if chi_square_value < best_chi_square:
                best_gamma_ev = float(gamma_value)
                best_temperature_k = float(temperature_value)
                best_ncc = ncc_value
                best_chi_square = chi_square_value
                best_simulated = simulated
    residual: Float64[Array, " n_k n_energy"] = measured - best_simulated
    metrics: Dict[str, Any] = {
        "best_gamma_ev": best_gamma_ev,
        "best_temperature_k": best_temperature_k,
        "best_ncc": best_ncc,
        "best_chi2_per_dof": best_chi_square,
        "planted_gamma_ev": planted_gamma_ev,
        "planted_temperature_k": planted_temperature_k,
    }
    residual_figure: Any
    residual_figure, _, _ = dp.plots.plot_difference_map(
        residual,
        k_axis,
        energy_axis,
        colorbar=False,
        title="measured minus best simulated cut",
    )
    artifacts: List[dp.types.ArtifactRecord] = [
        dp.harness.save_figure_artifact(
            ctx,
            "residual_map.png",
            residual_figure,
            role="residual_map",
        ),
        dp.harness.save_json_artifact(
            ctx,
            "match_table.json",
            {"candidates": table},
            role="match_table",
        ),
        dp.harness.save_array_artifact(
            ctx,
            "match_arrays.npz",
            {
                "best_simulated": best_simulated,
                "energy_axis": energy_axis,
                "k_axis": k_axis,
                "measured": measured,
                "residual": residual,
            },
            role="match_arrays",
        ),
        dp.harness.save_json_artifact(
            ctx,
            "metrics.json",
            metrics,
            role="metrics",
        ),
    ]
    result: Dict[str, Any] = {"metrics": metrics, "artifacts": artifacts}
    return result


if __name__ == "__main__":
    main()
