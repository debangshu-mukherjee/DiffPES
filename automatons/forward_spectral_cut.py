# /// script
# requires-python = ">=3.12,<3.15"
# dependencies = ["diffpes==2026.06.13"]
# ///
"""Simulate an intrinsic graphene spectral cut with occupied Lorentzians.

The automaton diagonalizes a graphene path and applies the public spectral
assembler. It writes a spectrum carrier, arrays, two figures, and metrics.
Smoke mode uses at most 24 momenta and 48 energy samples.
"""

from __future__ import annotations

from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
from beartype.typing import Any, Dict, List

import diffpes as dp


@dp.harness.experiment(
    name="forward-spectral-cut",
    params=(
        dp.types.make_automaton_param(
            "gamma_ev",
            float,
            default=0.025,
            help="Constant self-energy width in eV.",
            bounds=(1.0e-5, 2.0),
            example=0.025,
        ),
        dp.types.make_automaton_param(
            "temperature_k",
            float,
            default=40.0,
            help="Sample temperature in kelvin.",
            bounds=(1.0, 2000.0),
            example=40.0,
        ),
        dp.types.make_automaton_param(
            "energy_min_ev",
            float,
            default=-3.0,
            help="Lower relative-energy limit in eV.",
            bounds=(-20.0, 19.0),
            example=-3.0,
        ),
        dp.types.make_automaton_param(
            "energy_max_ev",
            float,
            default=0.18,
            help="Upper relative-energy limit in eV.",
            bounds=(-19.0, 20.0),
            example=0.18,
        ),
        dp.types.make_automaton_param(
            "n_energy",
            int,
            default=401,
            help="Number of relative-energy samples.",
            bounds=(8.0, 4096.0),
            example=401,
        ),
        dp.types.make_automaton_param(
            "n_k",
            int,
            default=201,
            help="Number of momentum samples through K.",
            bounds=(8.0, 4096.0),
            example=201,
        ),
    ),
    returns={
        "metrics": {
            "max_intensity": {"type": "number"},
            "integrated_intensity": {"type": "number"},
            "mdc_fwhm_inv_ang_at_ef": {"type": "number"},
            "edc_peak_ev_at_k_index": {"type": "number"},
        },
        "artifacts": {
            "roles": [
                "spectral_cut",
                "edc_mdc_panels",
                "spectrum_h5",
                "spectrum_arrays",
            ]
        },
    },
)
def main(
    args: SimpleNamespace,
    ctx: dp.types.AutomatonContext,
) -> Dict[str, Any]:
    """Run one intrinsic spectral-cut calculation and return its artifacts.

    The body uses uniform invariant band weights with a causal constant width.
    It retains the physical axes inside an ArpesSpectrum carrier.
    """
    n_k: int = min(args.n_k, 24) if args.smoke else args.n_k
    n_energy: int = min(args.n_energy, 48) if args.smoke else args.n_energy
    model: Any = dp.harness.graphene_pz_model()
    path_fractional: Any = jnp.linspace(
        jnp.asarray((0.47, 1.0 / 3.0, 0.0), dtype=jnp.float64),
        jnp.asarray((0.86, 1.0 / 3.0, 0.0), dtype=jnp.float64),
        n_k,
    )
    path_cartesian: Any = dp.tightb.kpoints_frac_to_cart(
        path_fractional,
        model.geometry,
    )
    momentum_axis: Any = jnp.linalg.norm(
        path_cartesian - path_cartesian[0],
        axis=1,
    )
    eigenvalues: Any = dp.tightb.eigvalsh_bands(model, path_fractional)
    energy_axis: Any = jnp.linspace(
        args.energy_min_ev,
        args.energy_max_ev,
        n_energy,
    )
    band_weights: Any = jnp.broadcast_to(
        jnp.ones_like(eigenvalues)[:, None, :],
        (n_k, n_energy, eigenvalues.shape[1]),
    )
    self_energy: Any = dp.types.make_self_energy_model(gamma=args.gamma_ev)
    intensity: Any = dp.simul.assemble_spectral_intensity_bands_chunk(
        eigenvalues,
        band_weights,
        energy_axis,
        self_energy,
        jnp.asarray(0.0, dtype=jnp.float64),
        args.temperature_k,
        allow_degenerate_value_only=True,
    )
    spectrum: Any = dp.types.make_arpes_spectrum(
        intensity,
        energy_axis,
        momentum_axis,
        path_cartesian,
    )
    fermi_index: int = int(np.argmin(np.abs(np.asarray(energy_axis))))
    center_index: int = n_k // 2
    mdc_values: Any = np.asarray(intensity[:, fermi_index])
    mdc_half_maximum: float = 0.5 * float(np.max(mdc_values))
    fwhm_indices: Any = np.flatnonzero(mdc_values >= mdc_half_maximum)
    minimum_width_samples: int = 2
    mdc_fwhm: float
    if fwhm_indices.size >= minimum_width_samples:
        mdc_fwhm = float(
            np.asarray(momentum_axis)[fwhm_indices[-1]]
            - np.asarray(momentum_axis)[fwhm_indices[0]]
        )
    else:
        mdc_fwhm = 0.0
    edc_peak_index: int = int(np.argmax(np.asarray(intensity[center_index])))
    spectral_figure: Any
    spectral_figure, _, _ = dp.plots.plot_arpes_spectrum(
        spectrum,
        title="Intrinsic graphene spectral cut",
    )
    panel_figure: Any
    panel_figure, _, _ = dp.plots.plot_edc_mdc_panels(
        intensity,
        momentum_axis,
        energy_axis,
        k_value=float(momentum_axis[center_index]),
        energy_value=0.0,
        suptitle="Intrinsic EDC and MDC",
    )
    integrated_intensity: float = float(
        jnp.trapezoid(
            jnp.trapezoid(intensity, energy_axis, axis=1),
            momentum_axis,
        )
    )
    metrics: Dict[str, Any] = {
        "max_intensity": float(jnp.max(intensity)),
        "integrated_intensity": integrated_intensity,
        "mdc_fwhm_inv_ang_at_ef": mdc_fwhm,
        "edc_peak_ev_at_k_index": float(energy_axis[edc_peak_index]),
    }
    artifacts: List[Any] = [
        dp.harness.save_figure_artifact(
            ctx,
            "spectral_cut.png",
            spectral_figure,
            role="spectral_cut",
        ),
        dp.harness.save_figure_artifact(
            ctx,
            "edc_mdc_panels.png",
            panel_figure,
            role="edc_mdc_panels",
        ),
        dp.harness.save_carrier_artifact(
            ctx,
            "spectrum.h5",
            spectrum,
            role="spectrum_h5",
        ),
        dp.harness.save_array_artifact(
            ctx,
            "spectrum.npz",
            {
                "energy_axis_ev": energy_axis,
                "intensity": intensity,
                "momentum_axis_inv_ang": momentum_axis,
            },
            role="spectrum_arrays",
        ),
        dp.harness.save_json_artifact(ctx, "metrics.json", metrics),
    ]
    result: Dict[str, Any] = {"metrics": metrics, "artifacts": artifacts}
    return result


if __name__ == "__main__":
    main()
