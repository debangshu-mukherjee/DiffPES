# /// script
# requires-python = ">=3.12,<3.15"
# dependencies = ["diffpes==2026.06.13"]
# ///
"""Compare detector maps from two public photon-polarization states.

The automaton builds both polarization vectors from incidence angles. It runs
the public coherent matrix-element pipeline and writes the signed asymmetry.
Smoke mode uses a compact detector raster with physical calibration effects.
"""

from __future__ import annotations

from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
from beartype.typing import Any, Dict, List

import diffpes as dp


@dp.harness.experiment(
    name="polarization-dichroism",
    params=(
        dp.types.make_automaton_param(
            "polarization_a",
            str,
            default="p",
            help="First public polarization selector.",
            choices=("p", "s", "c+", "c-", "linear"),
            example="p",
        ),
        dp.types.make_automaton_param(
            "polarization_b",
            str,
            default="s",
            help="Second public polarization selector.",
            choices=("p", "s", "c+", "c-", "linear"),
            example="s",
        ),
        dp.types.make_automaton_param(
            "photon_energy_ev",
            float,
            default=21.2,
            help="Photon energy in eV.",
            bounds=(10.0, 150.0),
            example=21.2,
        ),
        dp.types.make_automaton_param(
            "incidence_theta_rad",
            float,
            default=0.4,
            help="Photon incidence angle in radians.",
            bounds=(0.0, 1.5),
            example=0.4,
        ),
        dp.types.make_automaton_param(
            "incidence_phi_rad",
            float,
            default=0.0,
            help="Photon incidence azimuth in radians.",
            bounds=(-3.2, 3.2),
            example=0.0,
        ),
        dp.types.make_automaton_param(
            "polarization_angle_a_rad",
            float,
            default=0.0,
            help="Linear-basis angle for the first state in radians.",
            bounds=(-3.2, 3.2),
            example=0.0,
        ),
        dp.types.make_automaton_param(
            "polarization_angle_b_rad",
            float,
            default=0.5,
            help="Linear-basis angle for the second state in radians.",
            bounds=(-3.2, 3.2),
            example=0.5,
        ),
    ),
    returns={
        "metrics": {
            "max_abs_asymmetry": {"type": "number"},
            "mean_asymmetry": {"type": "number"},
            "sign_change_count": {"type": "integer"},
        },
        "artifacts": {
            "roles": [
                "intensity_a",
                "intensity_b",
                "asymmetry_map",
                "dichroism_arrays",
            ]
        },
    },
)
def main(
    args: SimpleNamespace,
    ctx: dp.types.AutomatonContext,
) -> Dict[str, Any]:
    """Run a two-polarization detector comparison and return its artifacts.

    The source driver applies the public matrix-element intensity reduction.
    The body forms a signed detector-map asymmetry after that physical chain.
    """
    n_raster: int = 8 if args.smoke else 24
    n_energy: int = 8 if args.smoke else 24
    model: Any = dp.harness.two_orbital_dirac_model(
        velocity_ev_ang=3.3,
        lattice_a_ang=2.0,
    )
    source_axis: Any = jnp.linspace(-0.24, 0.24, n_raster)
    kgrid: Any = dp.tightb.build_arpes_kmesh(
        source_axis,
        source_axis,
        0.0,
        0.0,
        model.geometry,
    )
    hamiltonians: Any = dp.tightb.bloch_hamiltonian_batch(
        model,
        kgrid.kpoints,
    )
    bands: Any = dp.tightb.diagonalize_tb(model, kgrid.kpoints)
    polarization_a: Any = dp.simul.polarization_from_angles(
        args.incidence_theta_rad,
        args.incidence_phi_rad,
        args.polarization_a,
        args.polarization_angle_a_rad,
    )
    polarization_b: Any = dp.simul.polarization_from_angles(
        args.incidence_theta_rad,
        args.incidence_phi_rad,
        args.polarization_b,
        args.polarization_angle_b_rad,
    )
    geometry_a: Any = dp.types.make_experiment_geometry(
        photon_energy_ev=args.photon_energy_ev,
        polarization=polarization_a,
        incidence_theta=args.incidence_theta_rad,
        incidence_phi=args.incidence_phi_rad,
        work_function_ev=4.5,
        temperature_k=45.0,
        mean_free_path_ang=9.0,
    )
    geometry_b: Any = dp.types.make_experiment_geometry(
        photon_energy_ev=args.photon_energy_ev,
        polarization=polarization_b,
        incidence_theta=args.incidence_theta_rad,
        incidence_phi=args.incidence_phi_rad,
        work_function_ev=4.5,
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
        u_bin_edges=jnp.linspace(-0.075, 0.075, n_raster + 1),
        v_bin_edges=jnp.linspace(-0.075, 0.075, n_raster + 1),
        energy_bin_edges_ev=jnp.linspace(-0.72, 0.10, n_energy + 1),
        psf_fwhm_u=0.005,
        psf_fwhm_v=0.005,
        psf_fwhm_energy_ev=0.012,
        transmission_reference_domain_ev=jnp.asarray(
            (args.photon_energy_ev - 5.5, args.photon_energy_ev - 3.5)
        ),
    )
    detector_effects: Any = dp.types.make_detector_effects(
        domain_logits=jnp.asarray((0.0,)),
        domain_euler_angles_rad=jnp.zeros((1, 3)),
        transmission_raw_slopes=jnp.asarray((-0.35, 0.15)),
        background_coefficients=jnp.asarray((-8.0,)),
        sensitivity_coefficients=jnp.asarray(()),
        exposure=1.0e8,
        background_mode="flat",
        sensitivity_mode="constant",
        domain_frame_ids=("org.diffpes.frame.sample_cartesian",),
    )
    energy_axis: Any = jnp.linspace(-0.72, 0.10, n_energy)
    common_inputs: Any = (
        (hamiltonians,),
        (bands,),
        radial_spec,
        matrix_element_params,
        dp.types.make_radial_quadrature_spec(),
        dp.types.make_final_state_spec(),
    )
    raster_a: Any = dp.simul.simulate_arpes(
        *common_inputs,
        geometry_a,
        dp.types.make_self_energy_model(gamma=0.025),
        kgrid,
        energy_axis,
        calibration,
        detector_effects,
        k_chunk=64,
        energy_chunk=16,
        checkpoint=True,
    )
    raster_b: Any = dp.simul.simulate_arpes(
        *common_inputs,
        geometry_b,
        dp.types.make_self_energy_model(gamma=0.025),
        kgrid,
        energy_axis,
        calibration,
        detector_effects,
        k_chunk=64,
        energy_chunk=16,
        checkpoint=True,
    )
    intensity_a: Any = jnp.sum(raster_a.expected_counts[0], axis=-1)
    intensity_b: Any = jnp.sum(raster_b.expected_counts[0], axis=-1)
    asymmetry: Any = (intensity_a - intensity_b) / jnp.maximum(
        intensity_a + intensity_b,
        1.0e-20,
    )
    flattened_signs: Any = np.sign(np.asarray(asymmetry).reshape(-1))
    sign_changes: int = int(
        np.count_nonzero(flattened_signs[1:] * flattened_signs[:-1] < 0.0)
    )
    intensity_a_figure: Any
    intensity_a_figure, _, _ = dp.plots.plot_detector_image(
        raster_a,
        title=f"{args.polarization_a} detector map",
    )
    intensity_b_figure: Any
    intensity_b_figure, _, _ = dp.plots.plot_detector_image(
        raster_b,
        title=f"{args.polarization_b} detector map",
    )
    asymmetry_figure: Any
    asymmetry_figure, _, _ = dp.plots.plot_difference_map(
        asymmetry,
        raster_a.detector_u_axis,
        raster_a.detector_v_axis,
        zero_lines=True,
        title="Polarization asymmetry",
    )
    metrics: Dict[str, Any] = {
        "max_abs_asymmetry": float(jnp.max(jnp.abs(asymmetry))),
        "mean_asymmetry": float(jnp.mean(asymmetry)),
        "sign_change_count": sign_changes,
    }
    artifacts: List[Any] = [
        dp.harness.save_figure_artifact(
            ctx,
            "intensity_a.png",
            intensity_a_figure,
            role="intensity_a",
        ),
        dp.harness.save_figure_artifact(
            ctx,
            "intensity_b.png",
            intensity_b_figure,
            role="intensity_b",
        ),
        dp.harness.save_figure_artifact(
            ctx,
            "asymmetry_map.png",
            asymmetry_figure,
            role="asymmetry_map",
        ),
        dp.harness.save_array_artifact(
            ctx,
            "dichroism.npz",
            {
                "asymmetry": asymmetry,
                "intensity_a": intensity_a,
                "intensity_b": intensity_b,
            },
            role="dichroism_arrays",
        ),
        dp.harness.save_json_artifact(ctx, "metrics.json", metrics),
    ]
    result: Dict[str, Any] = {"metrics": metrics, "artifacts": artifacts}
    return result


if __name__ == "__main__":
    main()
