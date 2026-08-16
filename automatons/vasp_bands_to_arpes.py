# /// script
# requires-python = ">=3.12,<3.15"
# dependencies = ["diffpes==2026.06.13"]
# ///
"""Render VASP band metadata as an occupied spectral cut.

The automaton reads the public EIGENVAL and PROCAR carriers, forms a weighted
spectral cut, and records the metadata needed to inspect that result. In smoke
mode it writes a minimal local VASP fixture and also exercises the explicit-H
detector workflow with a separately supplied synthetic Hamiltonian.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import jax.numpy as jnp
from beartype.typing import Any, Dict, List

import diffpes as dp


@dp.harness.experiment(
    name="vasp-bands-to-arpes",
    params=(
        dp.types.make_automaton_param(
            "vasp_dir",
            str,
            default="data/DFT/Bi2Se3/6QL/Output few bands/MGM",
            help="Directory containing EIGENVAL, PROCAR, and KPOINTS.",
            example="data/DFT/Bi2Se3/6QL/Output few bands/MGM",
        ),
        dp.types.make_automaton_param(
            "gamma_ev",
            float,
            default=0.04,
            help="Constant spectral half width in eV.",
            bounds=(1.0e-4, 1.0),
            example=0.04,
        ),
        dp.types.make_automaton_param(
            "temperature_k",
            float,
            default=35.0,
            help="Sample temperature in kelvin.",
            bounds=(0.1, 1000.0),
            example=35.0,
        ),
        dp.types.make_automaton_param(
            "n_energy",
            int,
            default=81,
            help="Number of relative-energy samples.",
            bounds=(3.0, 1024.0),
            example=81,
        ),
    ),
    returns={
        "metrics": {
            "n_bands": {"type": "integer"},
            "n_kpoints": {"type": "integer"},
            "fermi_energy_ev": {"type": "number"},
            "max_intensity": {"type": "number"},
        },
        "artifacts": {
            "roles": [
                "vasp_spectral_cut",
                "vasp_band_weights",
                "vasp_arrays",
            ]
        },
    },
)
def main(
    args: SimpleNamespace,
    ctx: dp.types.AutomatonContext,
) -> Dict[str, Any]:
    """Read VASP metadata, make a spectral cut, and save artifacts.

    The production path uses the band and orbital-projection carriers directly.
    The compact local fixture invokes the public coherent workflow with a
    separately supplied explicit Hamiltonian tensor.
    """
    n_energy: int = min(args.n_energy, 9) if args.smoke else args.n_energy
    input_directory: Path
    if args.smoke:
        eigenval_path: Path = dp.harness.artifact_path(
            ctx,
            "smoke_vasp/EIGENVAL",
        )
        input_directory = eigenval_path.parent
        eigenval_path.write_text(
            "     1     1     1     1\n"
            "unknown\n"
            "  1.0  1.0  1.0\n"
            "  0  0  0\n"
            "  0  0  0\n"
            "  1     2     1\n"
            "\n"
            "\n"
            "  0.000  0.000  0.000  0.5\n"
            "\n"
            "    1   -1.0\n"
            "\n"
            "  0.500  0.000  0.000  0.5\n"
            "\n"
            "    1   -0.5\n",
            encoding="utf-8",
        )
        (input_directory / "PROCAR").write_text(
            " # of k-points:   2 # of bands:   1 # of ions:   1\n"
            "\n"
            " k-point    1 :    0.00000000 0.00000000 0.00000000\n"
            " band    1\n"
            " ion    s     py     pz     px    dxy    dyz    dz2    dxz   "
            "dx2-y2    tot\n"
            "   1  0.1  0.05  0.05  0.05  0.02  0.02  0.02  0.02  "
            "0.02  0.35\n"
            " tot  0.1  0.05  0.05  0.05  0.02  0.02  0.02  0.02  "
            "0.02  0.35\n"
            "\n"
            " k-point    2 :    0.50000000 0.00000000 0.00000000\n"
            " band    1\n"
            " ion    s     py     pz     px    dxy    dyz    dz2    dxz   "
            "dx2-y2    tot\n"
            "   1  0.15  0.06  0.06  0.06  0.02  0.02  0.02  0.02  "
            "0.02  0.43\n"
            " tot  0.15  0.06  0.06  0.06  0.02  0.02  0.02  0.02  "
            "0.02  0.43\n",
            encoding="utf-8",
        )
        (input_directory / "KPOINTS").write_text(
            "k-path\n"
            "2\n"
            "Line-mode\n"
            "Reciprocal\n"
            "0.0 0.0 0.0 ! G\n"
            "0.5 0.0 0.0 ! X\n",
            encoding="utf-8",
        )
        (input_directory / "POSCAR").write_text(
            "Synthetic two-atom fixture\n"
            "1.0\n"
            "2.0 0.0 0.0\n"
            "0.0 2.0 0.0\n"
            "0.0 0.0 20.0\n"
            "X\n"
            "2\n"
            "Direct\n"
            "0.0 0.0 0.0\n"
            "0.5 0.5 0.0\n",
            encoding="utf-8",
        )
        (input_directory / "OUTCAR").write_text(
            " E-fermi : 0.0000 eV\n",
            encoding="utf-8",
        )
    else:
        input_directory = Path(args.vasp_dir)
    context: Any = dp.simul.load_vasp_context(
        directory=str(input_directory),
        eigenval_file="EIGENVAL",
        procar_file="PROCAR",
        doscar_file=None,
        kpoints_file="KPOINTS",
        fermi_energy=0.0,
    )
    energy_axis: Any = jnp.linspace(-1.20, 0.20, n_energy)
    orbital_weights: Any = jnp.sum(
        context.orb_proj.projections,
        axis=(2, 3),
    )
    spectral_weights: Any = jnp.broadcast_to(
        orbital_weights[:, None, :],
        (
            context.bands.eigenvalues.shape[0],
            n_energy,
            context.bands.eigenvalues.shape[1],
        ),
    )
    self_energy: Any = dp.types.make_self_energy_model(gamma=args.gamma_ev)
    spectral_intensity: Any = dp.simul.assemble_spectral_intensity_bands_chunk(
        context.bands.eigenvalues,
        spectral_weights,
        energy_axis,
        self_energy,
        context.bands.fermi_energy,
        args.temperature_k,
        allow_degenerate_value_only=True,
    )
    momentum_axis: Any = jnp.arange(
        context.bands.eigenvalues.shape[0],
        dtype=jnp.float64,
    )
    workflow_counts: Any = jnp.zeros((0,), dtype=jnp.float64)
    if args.smoke:
        crystal_geometry: Any = dp.types.make_crystal_geometry(
            2.0 * jnp.pi * jnp.eye(3, dtype=jnp.float64),
            jnp.zeros((1, 3), dtype=jnp.float64),
            ("X",),
        )
        orbital_basis: Any = dp.types.make_orbital_basis(
            atom_indices=(0,),
            n=(1,),
            l=(0,),
            m=(0,),
            labels=("1s",),
        )
        radial_spec: Any = dp.types.make_radial_spec(
            orbital_basis,
            (0,),
            mode="fixed",
            fixed_integrals_shell=jnp.asarray(((0.0, 1.0),)),
        )
        matrix_element_params: Any = dp.types.make_matrix_element_params(
            orbital_basis,
            (0,),
            sigma_shell=jnp.asarray((1.0,)),
            phase_shift_angles_shell=jnp.asarray((0.0,)),
        )
        polarization: Any = dp.simul.polarization_from_angles(0.4, 0.0, "p")
        geometry: Any = dp.types.make_experiment_geometry(
            photon_energy_ev=50.0,
            polarization=polarization,
            incidence_theta=0.4,
            work_function_ev=4.5,
            temperature_k=args.temperature_k,
        )
        calibration: Any = dp.types.make_detector_calibration(
            u_bin_edges=jnp.asarray((-0.05, 0.08, 0.18)),
            v_bin_edges=jnp.asarray((-0.02, 0.02)),
            energy_bin_edges_ev=jnp.linspace(-1.25, 0.25, n_energy + 1),
            psf_fwhm_u=0.01,
            psf_fwhm_v=0.01,
            psf_fwhm_energy_ev=0.02,
            transmission_reference_domain_ev=jnp.asarray((44.0, 46.0)),
        )
        detector_effects: Any = dp.types.make_detector_effects(
            domain_logits=jnp.asarray((0.0,)),
            domain_euler_angles_rad=jnp.zeros((1, 3)),
            transmission_raw_slopes=jnp.asarray((-0.2, 0.1)),
            background_coefficients=jnp.asarray((-3.0,)),
            sensitivity_coefficients=jnp.asarray(()),
            exposure=1.0,
            background_mode="flat",
            sensitivity_mode="constant",
            domain_frame_ids=("org.diffpes.frame.sample_cartesian",),
        )
        explicit_hamiltonians: Any = jnp.asarray(
            (((-1.25 + 0.0j,),), ((-0.75 + 0.0j,),)),
            dtype=jnp.complex128,
        )
        workflow_raster: Any = dp.simul.run_vasp_workflow(
            explicit_hamiltonians,
            crystal_geometry=crystal_geometry,
            orbital_basis=orbital_basis,
            radial_spec=radial_spec,
            matrix_element_params=matrix_element_params,
            radial_quadrature=dp.types.make_radial_quadrature_spec(),
            final_state=dp.types.make_final_state_spec(),
            experiment_geometry=geometry,
            self_energy=self_energy,
            energy_axis=energy_axis,
            detector_calibration=calibration,
            detector_effects=detector_effects,
            directory=str(input_directory),
            eigenval_file="EIGENVAL",
            procar_file="PROCAR",
            doscar_file=None,
            kpoints_file="KPOINTS",
            fermi_energy=0.0,
            phase_loss="ignore",
            k_chunk=2,
            energy_chunk=3,
            checkpoint=False,
        )
        workflow_counts = workflow_raster.expected_counts
    spectral_figure: Any
    spectral_figure, _, _ = dp.plots.plot_spectral_cut(
        spectral_intensity,
        momentum_axis,
        energy_axis,
        title="VASP projection-weighted spectral cut",
    )
    weight_figure: Any
    weight_figure, _, _ = dp.plots.plot_band_scatter_weights(
        context.bands,
        orbital_weights,
        momentum_axis=momentum_axis,
        title="VASP orbital projection weights",
    )
    metrics: Dict[str, Any] = {
        "n_bands": int(context.bands.eigenvalues.shape[1]),
        "n_kpoints": int(context.bands.eigenvalues.shape[0]),
        "fermi_energy_ev": float(context.bands.fermi_energy),
        "max_intensity": float(jnp.max(spectral_intensity)),
    }
    artifacts: List[Any] = [
        dp.harness.save_figure_artifact(
            ctx,
            "vasp_spectral_cut.png",
            spectral_figure,
            role="vasp_spectral_cut",
        ),
        dp.harness.save_figure_artifact(
            ctx,
            "vasp_band_weights.png",
            weight_figure,
            role="vasp_band_weights",
        ),
        dp.harness.save_array_artifact(
            ctx,
            "vasp_arrays.npz",
            {
                "energy_axis_ev": energy_axis,
                "momentum_index": momentum_axis,
                "orbital_weights": orbital_weights,
                "spectral_intensity": spectral_intensity,
                "workflow_expected_counts": workflow_counts,
            },
            role="vasp_arrays",
        ),
        dp.harness.save_json_artifact(ctx, "metrics.json", metrics),
    ]
    result: Dict[str, Any] = {"metrics": metrics, "artifacts": artifacts}
    return result


if __name__ == "__main__":
    main()
