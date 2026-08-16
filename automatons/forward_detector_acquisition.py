# /// script
# requires-python = ">=3.12,<3.15"
# dependencies = ["diffpes==2026.06.13"]
# ///
"""Simulate a calibrated detector acquisition from a Dirac reference model.

The automaton evaluates the public coherent source and detector pipeline. It
writes sampled counts, expected counts, detector figures, and summary metrics.
Smoke mode uses a raster with no more than 24 bins on each axis.
"""

from __future__ import annotations

from types import SimpleNamespace

import jax.numpy as jnp
from beartype.typing import Any, Dict, List

import diffpes as dp


@dp.harness.experiment(
    name="forward-detector-acquisition",
    params=(
        dp.types.make_automaton_param(
            "photon_energy_ev",
            float,
            default=21.2,
            help="Photon energy in eV.",
            bounds=(10.0, 150.0),
            example=21.2,
        ),
        dp.types.make_automaton_param(
            "work_function_ev",
            float,
            default=4.5,
            help="Sample work function in eV.",
            bounds=(1.0, 8.0),
            example=4.5,
        ),
        dp.types.make_automaton_param(
            "polarization",
            str,
            default="p",
            help="Public polarization selector.",
            choices=("p", "s", "c+", "c-", "linear"),
            example="p",
        ),
        dp.types.make_automaton_param(
            "exposure_s",
            float,
            default=1.0,
            help="Relative detector exposure in seconds.",
            bounds=(1.0e-6, 1000.0),
            example=1.0,
        ),
        dp.types.make_automaton_param(
            "n_u",
            int,
            default=48,
            help="Number of detector u bins.",
            bounds=(2.0, 512.0),
            example=48,
        ),
        dp.types.make_automaton_param(
            "n_v",
            int,
            default=48,
            help="Number of detector v bins.",
            bounds=(2.0, 512.0),
            example=48,
        ),
        dp.types.make_automaton_param(
            "n_energy",
            int,
            default=48,
            help="Number of recorded energy bins.",
            bounds=(2.0, 512.0),
            example=48,
        ),
    ),
    returns={
        "metrics": {
            "total_counts": {"type": "integer"},
            "max_counts": {"type": "integer"},
            "poisson_chi2_per_dof": {"type": "number"},
            "mean_expected_counts": {"type": "number"},
        },
        "artifacts": {
            "roles": [
                "detector_image",
                "detector_energy_cut",
                "counts_arrays",
            ]
        },
    },
)
def main(
    args: SimpleNamespace,
    ctx: dp.types.AutomatonContext,
) -> Dict[str, Any]:
    """Run one detector acquisition and return its artifacts.

    The body uses the public coherent source driver before detector effects.
    It samples Poisson counts only after the expected-rate calculation.
    """
    n_u: int = min(args.n_u, 24) if args.smoke else args.n_u
    n_v: int = min(args.n_v, 24) if args.smoke else args.n_v
    n_energy: int = min(args.n_energy, 24) if args.smoke else args.n_energy
    model: Any = dp.harness.two_orbital_dirac_model(
        velocity_ev_ang=3.3,
        lattice_a_ang=2.0,
    )
    source_u_axis: Any = jnp.linspace(-0.24, 0.24, n_u)
    source_v_axis: Any = jnp.linspace(-0.24, 0.24, n_v)
    kgrid: Any = dp.tightb.build_arpes_kmesh(
        source_u_axis,
        source_v_axis,
        0.0,
        0.0,
        model.geometry,
    )
    hamiltonians: Any = dp.tightb.bloch_hamiltonian_batch(
        model,
        kgrid.kpoints,
    )
    bands: Any = dp.tightb.diagonalize_tb(model, kgrid.kpoints)
    polarization: Any = dp.simul.polarization_from_angles(
        0.4,
        0.0,
        args.polarization,
    )
    geometry: Any = dp.types.make_experiment_geometry(
        photon_energy_ev=args.photon_energy_ev,
        polarization=polarization,
        incidence_theta=0.4,
        work_function_ev=args.work_function_ev,
        temperature_k=45.0,
        mean_free_path_ang=9.0,
    )
    radial_spec: Any = dp.types.make_radial_spec(
        model.basis,
        (0, 0),
        mode="fixed",
        fixed_integrals_shell=jnp.asarray(((0.0, 1.0),)),
    )
    matrix_element_params: Any = dp.types.make_matrix_element_params(
        model.basis,
        (0, 0),
        sigma_shell=jnp.asarray((1.0,)),
        phase_shift_angles_shell=jnp.asarray((0.15,)),
    )
    calibration: Any = dp.types.make_detector_calibration(
        u_bin_edges=jnp.linspace(-0.075, 0.075, n_u + 1),
        v_bin_edges=jnp.linspace(-0.075, 0.075, n_v + 1),
        energy_bin_edges_ev=jnp.linspace(-0.72, 0.10, n_energy + 1),
        psf_fwhm_u=0.005,
        psf_fwhm_v=0.005,
        psf_fwhm_energy_ev=0.012,
        transmission_reference_domain_ev=jnp.asarray(
            (
                args.photon_energy_ev - args.work_function_ev - 1.0,
                args.photon_energy_ev - args.work_function_ev + 1.0,
            )
        ),
    )
    detector_effects: Any = dp.types.make_detector_effects(
        domain_logits=jnp.asarray((0.0,)),
        domain_euler_angles_rad=jnp.zeros((1, 3)),
        transmission_raw_slopes=jnp.asarray((-0.35, 0.15)),
        background_coefficients=jnp.asarray((-8.0,)),
        sensitivity_coefficients=jnp.asarray(()),
        exposure=args.exposure_s * 1.0e8,
        background_mode="flat",
        sensitivity_mode="constant",
        domain_frame_ids=("org.diffpes.frame.sample_cartesian",),
    )
    energy_axis: Any = jnp.linspace(-0.72, 0.10, n_energy)
    raster: Any = dp.simul.simulate_arpes(
        (hamiltonians,),
        (bands,),
        radial_spec,
        matrix_element_params,
        dp.types.make_radial_quadrature_spec(),
        dp.types.make_final_state_spec(),
        geometry,
        dp.types.make_self_energy_model(gamma=0.025),
        kgrid,
        energy_axis,
        calibration,
        detector_effects,
        k_chunk=64,
        energy_chunk=16,
        checkpoint=True,
    )
    expected_counts: Any = raster.expected_counts[0]
    sampled_counts: Any = dp.simul.sample_poisson_counts(
        ctx.rng_key,
        expected_counts,
    )
    residual: Any = sampled_counts - expected_counts
    chi_square: Any = jnp.sum(
        residual * residual / jnp.maximum(expected_counts, 1.0)
    )
    image_figure: Any
    image_figure, _, _ = dp.plots.plot_detector_image(
        raster,
        log_counts=True,
        title="Poisson-sampled detector acquisition",
    )
    cut_figure: Any
    cut_figure, _, _ = dp.plots.plot_detector_energy_cut(
        raster,
        cut_axis="v",
        log_counts=True,
        title="Detector energy cut",
    )
    metrics: Dict[str, Any] = {
        "total_counts": int(jnp.sum(sampled_counts)),
        "max_counts": int(jnp.max(sampled_counts)),
        "poisson_chi2_per_dof": float(chi_square / sampled_counts.size),
        "mean_expected_counts": float(jnp.mean(expected_counts)),
    }
    artifacts: List[Any] = [
        dp.harness.save_figure_artifact(
            ctx,
            "detector_image.png",
            image_figure,
            role="detector_image",
        ),
        dp.harness.save_figure_artifact(
            ctx,
            "detector_energy_cut.png",
            cut_figure,
            role="detector_energy_cut",
        ),
        dp.harness.save_array_artifact(
            ctx,
            "counts.npz",
            {
                "expected_counts": expected_counts,
                "sampled_counts": sampled_counts,
                "energy_axis_ev": raster.energy_axis,
            },
            role="counts_arrays",
        ),
        dp.harness.save_json_artifact(ctx, "metrics.json", metrics),
    ]
    result: Dict[str, Any] = {"metrics": metrics, "artifacts": artifacts}
    return result


if __name__ == "__main__":
    main()
