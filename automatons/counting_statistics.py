# /// script
# requires-python = ">=3.12,<3.15"
# dependencies = ["diffpes==2026.06.13"]
# ///
"""Measure Poisson signal-to-noise scaling for detector-like counts.

This experiment creates a smooth detector-rate tensor and acquires Poisson
realizations over a short exposure ladder. It records integrated counts,
signal-to-noise values, and the log-log slope while saving raw count arrays.
"""

from __future__ import annotations

from types import SimpleNamespace

import jax
import jax.numpy as jnp
from beartype import beartype
from beartype.typing import Any, Dict, List, Tuple
from jaxtyping import Array, Float64, Int64, jaxtyped

import diffpes as dp


@jaxtyped(typechecker=beartype)
def _expected_rate_tensor(
    n_u: int,
    n_v: int,
    n_energy: int,
) -> Float64[Array, " n_u n_v n_energy"]:
    """PRIVATE: Build a smooth nonnegative detector event-rate tensor.

    Parameters
    ----------
    n_u : int
        Number of horizontal detector samples.
    n_v : int
        Number of vertical detector samples.
    n_energy : int
        Number of energy-channel samples.

    Returns
    -------
    rates : Float64[Array, " n_u n_v n_energy"]
        Positive expected events per second in each detector bin.

    Notes
    -----
    The Gaussian-like ridge and nonzero background make every Poisson mean
    finite and nonnegative while resembling a dispersive detector intensity.
    """
    u_axis: Float64[Array, " n_u"] = jnp.linspace(
        -1.0,
        1.0,
        n_u,
        dtype=jnp.float64,
    )
    v_axis: Float64[Array, " n_v"] = jnp.linspace(
        -1.0,
        1.0,
        n_v,
        dtype=jnp.float64,
    )
    energy_axis: Float64[Array, " n_energy"] = jnp.linspace(
        -0.16,
        0.10,
        n_energy,
        dtype=jnp.float64,
    )
    angular_profile: Float64[Array, " n_u n_v"] = jnp.exp(
        -0.5 * ((u_axis[:, None] - 0.28 * v_axis[None, :]) / 0.38) ** 2
        - 0.5 * (v_axis[None, :] / 0.60) ** 2
    )
    energy_profile: Float64[Array, " n_energy"] = jnp.exp(
        -0.5 * ((energy_axis + 0.045) / 0.050) ** 2
    )
    rates: Float64[Array, " n_u n_v n_energy"] = (
        180.0 * angular_profile[:, :, None] * energy_profile[None, None, :]
        + 0.5
    )
    return rates


@jaxtyped(typechecker=beartype)
def _loglog_slope(
    exposures: Float64[Array, " n_settings"],
    snr_values: Float64[Array, " n_settings"],
) -> Float64[Array, ""]:
    """PRIVATE: Fit a least-squares slope in log exposure and log SNR.

    Parameters
    ----------
    exposures : Float64[Array, " n_settings"]
        Strictly positive exposure values in seconds.
    snr_values : Float64[Array, " n_settings"]
        Strictly positive signal-to-noise values.

    Returns
    -------
    slope : Float64[Array, ""]
        Ordinary least-squares slope of log SNR against log exposure.

    Notes
    -----
    For independent Poisson counts, the integrated SNR is expected to scale
    with the square root of exposure and therefore have a slope near one half.
    """
    log_exposure: Float64[Array, " n_settings"] = jnp.log(exposures)
    log_snr: Float64[Array, " n_settings"] = jnp.log(snr_values)
    centred_exposure: Float64[Array, " n_settings"] = log_exposure - jnp.mean(
        log_exposure
    )
    centred_snr: Float64[Array, " n_settings"] = log_snr - jnp.mean(log_snr)
    numerator: Float64[Array, ""] = jnp.sum(centred_exposure * centred_snr)
    denominator: Float64[Array, ""] = jnp.sum(centred_exposure**2)
    slope: Float64[Array, ""] = numerator / denominator
    return slope


@dp.harness.experiment(
    name="counting-statistics",
    params=[
        dp.types.make_automaton_param(
            "exposures_s",
            list,
            default=[0.25, 1.0, 4.0],
            help="Increasing detector exposure values in seconds.",
            example=[0.25, 1.0, 4.0],
        ),
        dp.types.make_automaton_param(
            "n_repeats",
            int,
            default=2,
            help="Independent Poisson acquisitions at each exposure.",
            example=2,
        ),
    ],
    returns={
        "metrics": {
            "snr": {"type": "array"},
            "snr_loglog_slope": {"type": "number"},
            "total_counts": {"type": "array"},
        },
        "artifacts": {
            "roles": ["counts_comparison", "snr_curve", "counting_arrays"],
        },
    },
)
def main(  # noqa: PLR0915 -- each acquisition remains explicitly reproducible.
    args: SimpleNamespace,
    ctx: dp.types.AutomatonContext,
) -> Dict[str, Any]:
    """Run the exposure series and return observed Poisson scaling metrics."""
    exposure_values: Tuple[Any, ...] = tuple(args.exposures_s)
    minimum_ladder_size: int = 2
    maximum_ladder_size: int = 3
    maximum_repeats: int = 2
    if len(exposure_values) < minimum_ladder_size:
        raise ValueError("exposures_s must contain at least two values")
    if len(exposure_values) > maximum_ladder_size:
        raise ValueError("exposures_s may contain at most three values")
    if args.n_repeats < 1 or args.n_repeats > maximum_repeats:
        raise ValueError("n_repeats must be between one and two")
    exposures: Float64[Array, " n_settings"] = jnp.asarray(
        exposure_values,
        dtype=jnp.float64,
    )
    if bool(jnp.any(exposures <= 0.0)):
        raise ValueError("exposures_s values must be positive")
    if bool(jnp.any(exposures[1:] <= exposures[:-1])):
        raise ValueError("exposures_s values must be strictly increasing")
    n_u: int = 8 if args.smoke else 24
    n_v: int = 7 if args.smoke else 20
    n_energy: int = 12 if args.smoke else 48
    rates_per_second: Float64[Array, " n_u n_v n_energy"] = (
        _expected_rate_tensor(n_u, n_v, n_energy)
    )
    total_counts: List[float] = []
    snr_values: List[float] = []
    expected_blocks: List[Float64[Array, " n_u n_v n_energy"]] = []
    observed_blocks: List[Float64[Array, " n_u n_v n_energy"]] = []
    exposure_index: int
    exposure: Float64[Array, ""]
    for exposure_index, exposure in enumerate(exposures):
        expected_counts: Float64[Array, " n_u n_v n_energy"] = (
            rates_per_second * exposure
        )
        repeat_totals: List[float] = []
        repeat_index: int
        first_observed: Float64[Array, " n_u n_v n_energy"] | None = None
        for repeat_index in range(args.n_repeats):
            acquisition_key: jax.Array = jax.random.fold_in(
                ctx.rng_key,
                exposure_index * args.n_repeats + repeat_index,
            )
            sampled: Int64[Array, " n_u n_v n_energy"] = (
                dp.simul.sample_poisson_counts(
                    acquisition_key,
                    expected_counts,
                )
            )
            observed: Float64[Array, " n_u n_v n_energy"] = jnp.asarray(
                sampled,
                dtype=jnp.float64,
            )
            if first_observed is None:
                first_observed = observed
            repeat_totals.append(float(jnp.sum(observed)))
        mean_total: float = sum(repeat_totals) / float(args.n_repeats)
        expected_total: float = float(jnp.sum(expected_counts))
        snr: float = mean_total / float(jnp.sqrt(expected_total))
        total_counts.append(mean_total)
        snr_values.append(snr)
        expected_blocks.append(expected_counts)
        if first_observed is None:
            raise RuntimeError("Poisson acquisition did not return a sample")
        observed_blocks.append(first_observed)
    snr_array: Float64[Array, " n_settings"] = jnp.asarray(
        snr_values,
        dtype=jnp.float64,
    )
    slope: Float64[Array, ""] = _loglog_slope(exposures, snr_array)
    metrics: Dict[str, Any] = {
        "snr": snr_values,
        "snr_loglog_slope": float(slope),
        "total_counts": total_counts,
    }
    comparison_figure: Any
    comparison_figure, _, _ = dp.plots.plot_detector_comparison(
        expected_blocks[-1],
        observed_blocks[-1],
        view="energy",
        log_counts=True,
        colorbar=False,
        titles=("expected counts", "Poisson acquisition"),
    )
    snr_figure: Any
    snr_figure, _, _ = dp.plots.plot_curve_family(
        exposures,
        (snr_array,),
        labels=("integrated SNR",),
        log_x=True,
        log_y=True,
        xlabel="exposure (s)",
        ylabel="integrated SNR",
        title="Poisson exposure scaling",
    )
    artifacts: List[dp.types.ArtifactRecord] = [
        dp.harness.save_figure_artifact(
            ctx,
            "counts_comparison.png",
            comparison_figure,
            role="counts_comparison",
        ),
        dp.harness.save_figure_artifact(
            ctx,
            "snr_curve.png",
            snr_figure,
            role="snr_curve",
        ),
        dp.harness.save_array_artifact(
            ctx,
            "counting_arrays.npz",
            {
                "exposures_s": exposures,
                "expected_0": expected_blocks[0],
                "expected_1": expected_blocks[
                    min(1, len(expected_blocks) - 1)
                ],
                "expected_2": expected_blocks[
                    min(2, len(expected_blocks) - 1)
                ],
                "observed_0": observed_blocks[0],
                "observed_1": observed_blocks[
                    min(1, len(observed_blocks) - 1)
                ],
                "observed_2": observed_blocks[
                    min(2, len(observed_blocks) - 1)
                ],
                "snr": snr_array,
            },
            role="counting_arrays",
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
