# /// script
# requires-python = ">=3.12,<3.15"
# dependencies = ["diffpes==2026.06.13"]
# ///
"""Measure spectral-cut broadening across resolution settings.

This experiment creates a dispersive occupied spectral cut. It applies public
energy and momentum convolutions for paired resolution values. It reports the
apparent FWHM of the Fermi-level momentum distribution and saves comparison
figures with the resolved arrays.
"""

from __future__ import annotations

from types import SimpleNamespace

import jax
import jax.numpy as jnp
from beartype import beartype
from beartype.typing import Any, Dict, List, Tuple
from jaxtyping import Array, Float64, jaxtyped

import diffpes as dp


@jaxtyped(typechecker=beartype)
def _fermi_values(
    energy_axis: Float64[Array, " n_energy"],
) -> Float64[Array, " n_energy"]:
    """PRIVATE: Evaluate a finite-temperature Fermi occupation on one axis.

    Parameters
    ----------
    energy_axis : Float64[Array, " n_energy"]
        Relative energy coordinates in eV.

    Returns
    -------
    occupation : Float64[Array, " n_energy"]
        Occupied fraction at 45 K.

    Notes
    -----
    Vectorizes the scalar public Fermi-Dirac function. The common occupation
    factor changes spectral weight but leaves each momentum width physical.
    """
    occupation: Float64[Array, " n_energy"] = jax.vmap(
        dp.simul.fermi_dirac,
        in_axes=(0, None, None),
    )(energy_axis, 0.0, 45.0)
    return occupation


@jaxtyped(typechecker=beartype)
def _source_cut(
    n_k: int,
    n_energy: int,
) -> Tuple[
    Float64[Array, " n_k n_energy"],
    Float64[Array, " n_k"],
    Float64[Array, " n_energy"],
]:
    """PRIVATE: Build an occupied dispersive source cut.

    Parameters
    ----------
    n_k : int
        Number of momentum samples.
    n_energy : int
        Number of energy samples.

    Returns
    -------
    intensity : Float64[Array, " n_k n_energy"]
        Nonnegative source intensity.
    k_axis : Float64[Array, " n_k"]
        Strictly increasing momentum coordinates in inverse angstroms.
    energy_axis : Float64[Array, " n_energy"]
        Strictly increasing energy coordinates in eV.

    Notes
    -----
    Uses a linear dispersion with a Gaussian momentum profile. Energy
    broadening changes the Fermi-level MDC width through the dispersion slope.
    """
    k_axis: Float64[Array, " n_k"] = jnp.linspace(
        -0.25,
        0.25,
        n_k,
        dtype=jnp.float64,
    )
    energy_axis: Float64[Array, " n_energy"] = jnp.linspace(
        -0.12,
        0.08,
        n_energy,
        dtype=jnp.float64,
    )
    dispersion_velocity: float = 1.4
    intrinsic_sigma_k: float = 0.008
    centre_k: Float64[Array, " n_energy"] = energy_axis / dispersion_velocity
    profile: Float64[Array, " n_k n_energy"] = jnp.exp(
        -0.5 * ((k_axis[:, None] - centre_k[None, :]) / intrinsic_sigma_k) ** 2
    )
    occupation: Float64[Array, " n_energy"] = _fermi_values(energy_axis)
    intensity: Float64[Array, " n_k n_energy"] = profile * occupation[None, :]
    return intensity, k_axis, energy_axis


@jaxtyped(typechecker=beartype)
def _moment_fwhm(
    profile: Float64[Array, " n_k"],
    k_axis: Float64[Array, " n_k"],
) -> Float64[Array, ""]:
    """PRIVATE: Estimate an apparent Gaussian FWHM from a momentum profile.

    Parameters
    ----------
    profile : Float64[Array, " n_k"]
        Nonnegative momentum distribution values.
    k_axis : Float64[Array, " n_k"]
        Momentum coordinates in inverse angstroms.

    Returns
    -------
    fwhm : Float64[Array, ""]
        Second-moment Gaussian FWHM in inverse angstroms.

    Notes
    -----
    Calculates the profile mean and variance after normalizing its complete
    sampled weight. The result remains stable for slightly broadened peaks.
    """
    profile_sum: Float64[Array, ""] = jnp.maximum(jnp.sum(profile), 1.0e-30)
    weights: Float64[Array, " n_k"] = profile / profile_sum
    mean_k: Float64[Array, ""] = jnp.sum(weights * k_axis)
    variance_k: Float64[Array, ""] = jnp.sum(weights * (k_axis - mean_k) ** 2)
    fwhm: Float64[Array, ""] = (
        2.0 * jnp.sqrt(2.0 * jnp.log(2.0)) * jnp.sqrt(variance_k)
    )
    return fwhm


@dp.harness.experiment(
    name="resolution-sweep",
    params=[
        dp.types.make_automaton_param(
            "energy_resolutions_ev",
            list,
            default=[0.010, 0.030, 0.060],
            help="Paired energy-resolution FWHM values in eV.",
            example=[0.010, 0.030, 0.060],
        ),
        dp.types.make_automaton_param(
            "momentum_resolutions_inv_ang",
            list,
            default=[0.004, 0.010, 0.018],
            help=(
                "Paired momentum-resolution FWHM values in inverse angstroms."
            ),
            example=[0.004, 0.010, 0.018],
        ),
    ],
    returns={
        "metrics": {
            "apparent_fwhm_inv_ang": {"type": "array"},
            "monotone_in_energy_resolution": {"type": "boolean"},
        },
        "artifacts": {
            "roles": ["resolution_series", "fwhm_curve", "resolution_arrays"],
        },
    },
)
def main(  # noqa: PLR0915 -- paired settings retain their explicit provenance.
    args: SimpleNamespace,
    ctx: dp.types.AutomatonContext,
) -> Dict[str, Any]:
    """Run paired resolution convolutions and return their width trend."""
    energy_resolutions: Tuple[Any, ...] = tuple(args.energy_resolutions_ev)
    momentum_resolutions: Tuple[Any, ...] = tuple(
        args.momentum_resolutions_inv_ang
    )
    maximum_ladder_size: int = 3
    if not energy_resolutions or not momentum_resolutions:
        raise ValueError("resolution ladders must be nonempty")
    if len(energy_resolutions) != len(momentum_resolutions):
        raise ValueError("resolution ladders must have equal lengths")
    if len(energy_resolutions) > maximum_ladder_size:
        raise ValueError("resolution ladders may contain at most three values")
    if any(float(value) <= 0.0 for value in energy_resolutions):
        raise ValueError("energy resolutions must be positive")
    if any(float(value) <= 0.0 for value in momentum_resolutions):
        raise ValueError("momentum resolutions must be positive")
    n_k: int = 33 if args.smoke else 129
    n_energy: int = 48 if args.smoke else 192
    source: Float64[Array, " n_k n_energy"]
    k_axis: Float64[Array, " n_k"]
    energy_axis: Float64[Array, " n_energy"]
    source, k_axis, energy_axis = _source_cut(n_k, n_energy)
    fwhm_to_sigma: float = 1.0 / (2.0 * float(jnp.sqrt(2.0 * jnp.log(2.0))))
    fermi_index: int = int(jnp.argmin(jnp.abs(energy_axis)))
    resolved_cuts: List[Float64[Array, " n_k n_energy"]] = []
    fwhm_values: List[float] = []
    energy_value: Any
    momentum_value: Any
    for energy_value, momentum_value in zip(
        energy_resolutions,
        momentum_resolutions,
        strict=True,
    ):
        energy_sigma_ev: float = float(energy_value) * fwhm_to_sigma
        momentum_sigma_inv_ang: float = float(momentum_value) * fwhm_to_sigma
        energy_resolved: Float64[Array, " n_k n_energy"] = (
            dp.simul.convolve_energy(
                source,
                energy_axis,
                energy_sigma_ev,
                half_width=48,
            )
        )
        momentum_resolved: Float64[Array, " n_k n_energy"]
        momentum_resolved, _, _ = dp.simul.convolve_kpath(
            energy_resolved,
            k_axis,
            momentum_sigma_inv_ang,
        )
        mdc: Float64[Array, " n_k"] = momentum_resolved[:, fermi_index]
        apparent_fwhm: Float64[Array, ""] = _moment_fwhm(mdc, k_axis)
        resolved_cuts.append(momentum_resolved)
        fwhm_values.append(float(apparent_fwhm))
    monotone: bool = all(
        later + 1.0e-12 >= earlier
        for earlier, later in zip(fwhm_values, fwhm_values[1:], strict=False)
    )
    energy_resolution_array: Float64[Array, " n_settings"] = jnp.asarray(
        energy_resolutions,
        dtype=jnp.float64,
    )
    fwhm_array: Float64[Array, " n_settings"] = jnp.asarray(
        fwhm_values,
        dtype=jnp.float64,
    )
    labels: Tuple[str, ...] = tuple(
        f"{float(value) * 1000.0:.0f} meV" for value in energy_resolutions
    )
    metrics: Dict[str, Any] = {
        "apparent_fwhm_inv_ang": fwhm_values,
        "monotone_in_energy_resolution": monotone,
    }
    series_figure: Any
    series_figure, _, _ = dp.plots.plot_spectral_cut_series(
        tuple(resolved_cuts),
        k_axis,
        energy_axis,
        titles=labels,
        colorbar=False,
        xlabel="momentum (inverse angstroms)",
        ylabel="energy relative to the Fermi level (eV)",
    )
    fwhm_figure: Any
    fwhm_figure, _, _ = dp.plots.plot_curve_family(
        energy_resolution_array,
        (fwhm_array,),
        labels=("apparent MDC FWHM",),
        xlabel="energy-resolution FWHM (eV)",
        ylabel="apparent MDC FWHM (inverse angstroms)",
        title="resolution broadening trend",
    )
    artifacts: List[dp.types.ArtifactRecord] = [
        dp.harness.save_figure_artifact(
            ctx,
            "resolution_series.png",
            series_figure,
            role="resolution_series",
        ),
        dp.harness.save_figure_artifact(
            ctx,
            "fwhm_curve.png",
            fwhm_figure,
            role="fwhm_curve",
        ),
        dp.harness.save_array_artifact(
            ctx,
            "resolution_arrays.npz",
            {
                "energy_axis": energy_axis,
                "energy_resolutions_ev": energy_resolution_array,
                "fwhm": fwhm_array,
                "k_axis": k_axis,
                "source": source,
                "resolved_0": resolved_cuts[0],
                "resolved_1": resolved_cuts[min(1, len(resolved_cuts) - 1)],
                "resolved_2": resolved_cuts[min(2, len(resolved_cuts) - 1)],
            },
            role="resolution_arrays",
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
