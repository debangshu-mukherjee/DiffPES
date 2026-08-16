# /// script
# requires-python = ">=3.12,<3.15"
# dependencies = ["diffpes==2026.06.13"]
# ///
"""Simulate a graphene source cube around one Dirac point.

The automaton builds a public ARPES mesh and a spectral-intensity cube. It
writes a cube carrier, a Fermi map, orthogonal cuts, and scalar metrics.
Smoke mode uses no more than a 16 by 16 by 32 source raster.
"""

from __future__ import annotations

from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
from beartype.typing import Any, Dict, List
from matplotlib import pyplot as plt

import diffpes as dp


@dp.harness.experiment(
    name="forward-arpes-cube",
    params=(
        dp.types.make_automaton_param(
            "n_kx",
            int,
            default=81,
            help="Number of Cartesian kx samples.",
            bounds=(2.0, 1024.0),
            example=81,
        ),
        dp.types.make_automaton_param(
            "n_ky",
            int,
            default=81,
            help="Number of Cartesian ky samples.",
            bounds=(2.0, 1024.0),
            example=81,
        ),
        dp.types.make_automaton_param(
            "n_energy",
            int,
            default=181,
            help="Number of relative-energy samples.",
            bounds=(2.0, 1024.0),
            example=181,
        ),
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
            "slice_energy_ev",
            float,
            default=0.0,
            help="Energy of the rendered momentum map in eV.",
            bounds=(-1.6, 0.12),
            example=0.0,
        ),
    ),
    returns={
        "metrics": {
            "cube_shape": {"type": "array"},
            "fermi_map_fraction_above_half_max": {"type": "number"},
            "max_intensity": {"type": "number"},
        },
        "artifacts": {"roles": ["fermi_map", "orthogonal_cuts", "cube_h5"]},
    },
)
def main(
    args: SimpleNamespace,
    ctx: dp.types.AutomatonContext,
) -> Dict[str, Any]:
    """Run one source-cube calculation and return its artifacts.

    The body maps a Cartesian raster into fractional coordinates before it
    assembles intrinsic intensity. It keeps the public cube carrier intact.
    """
    n_kx: int = min(args.n_kx, 16) if args.smoke else args.n_kx
    n_ky: int = min(args.n_ky, 16) if args.smoke else args.n_ky
    n_energy: int = min(args.n_energy, 32) if args.smoke else args.n_energy
    model: Any = dp.harness.graphene_pz_model()
    dirac_fractional: Any = jnp.asarray(
        ((2.0 / 3.0, 1.0 / 3.0, 0.0),),
        dtype=jnp.float64,
    )
    dirac_cartesian: Any = dp.tightb.kpoints_frac_to_cart(
        dirac_fractional,
        model.geometry,
    )[0]
    half_width_inv_ang: float = 0.55
    kx_axis: Any = jnp.linspace(
        dirac_cartesian[0] - half_width_inv_ang,
        dirac_cartesian[0] + half_width_inv_ang,
        n_kx,
    )
    ky_axis: Any = jnp.linspace(
        dirac_cartesian[1] - half_width_inv_ang,
        dirac_cartesian[1] + half_width_inv_ang,
        n_ky,
    )
    kgrid: Any = dp.tightb.build_arpes_kmesh(
        kx_axis,
        ky_axis,
        0.0,
        0.0,
        model.geometry,
    )
    eigenvalues: Any = dp.tightb.eigvalsh_bands(model, kgrid.kpoints)
    energy_axis: Any = jnp.linspace(-1.6, 0.12, n_energy)
    band_weights: Any = jnp.broadcast_to(
        jnp.ones_like(eigenvalues)[:, None, :],
        (eigenvalues.shape[0], n_energy, eigenvalues.shape[1]),
    )
    self_energy: Any = dp.types.make_self_energy_model(gamma=args.gamma_ev)
    flat_intensity: Any = dp.simul.assemble_spectral_intensity_bands_chunk(
        eigenvalues,
        band_weights,
        energy_axis,
        self_energy,
        jnp.asarray(0.0, dtype=jnp.float64),
        args.temperature_k,
        allow_degenerate_value_only=True,
    )
    grid_intensity: Any = jnp.reshape(
        flat_intensity,
        (n_ky, n_kx, n_energy),
    )
    cube_intensity: Any = jnp.transpose(grid_intensity, (1, 0, 2))
    cube: Any = dp.types.make_arpes_cube(
        cube_intensity,
        kx_axis,
        ky_axis,
        energy_axis,
        provenance="intrinsic graphene source cube",
    )
    fermi_map: Any = dp.simul.constant_energy_slice(
        cube,
        args.slice_energy_ev,
    )
    fermi_figure: Any
    fermi_figure, _, _ = dp.plots.plot_momentum_map(
        fermi_map,
        cube.kx_axis,
        cube.ky_axis,
        title="Graphene momentum map",
    )
    center_kx: int = n_kx // 2
    center_ky: int = n_ky // 2
    kx_cut: Any = cube.intensity[:, center_ky, :]
    ky_cut: Any = cube.intensity[center_kx, :, :]
    cuts_figure: Any
    if n_kx == n_ky:
        cuts_figure, _, _ = dp.plots.plot_spectral_cut_series(
            (kx_cut, ky_cut),
            cube.kx_axis,
            cube.energy_axis,
            titles=("Central kx cut", "Central ky cut"),
        )
    else:
        cuts_figure, axes = plt.subplots(1, 2, constrained_layout=True)
        dp.plots.plot_spectral_cut(
            kx_cut,
            cube.kx_axis,
            cube.energy_axis,
            ax=axes[0],
            title="Central kx cut",
        )
        dp.plots.plot_spectral_cut(
            ky_cut,
            cube.ky_axis,
            cube.energy_axis,
            ax=axes[1],
            title="Central ky cut",
        )
    fermi_values: Any = np.asarray(fermi_map)
    fermi_half_maximum: float = 0.5 * float(np.max(fermi_values))
    fermi_fraction: float = float(np.mean(fermi_values >= fermi_half_maximum))
    metrics: Dict[str, Any] = {
        "cube_shape": [int(axis) for axis in cube.intensity.shape],
        "fermi_map_fraction_above_half_max": fermi_fraction,
        "max_intensity": float(jnp.max(cube.intensity)),
    }
    artifacts: List[Any] = [
        dp.harness.save_figure_artifact(
            ctx,
            "fermi_map.png",
            fermi_figure,
            role="fermi_map",
        ),
        dp.harness.save_figure_artifact(
            ctx,
            "orthogonal_cuts.png",
            cuts_figure,
            role="orthogonal_cuts",
        ),
        dp.harness.save_carrier_artifact(
            ctx,
            "cube.h5",
            cube,
            role="cube_h5",
        ),
        dp.harness.save_json_artifact(ctx, "metrics.json", metrics),
    ]
    result: Dict[str, Any] = {"metrics": metrics, "artifacts": artifacts}
    return result


if __name__ == "__main__":
    main()
