# /// script
# requires-python = ">=3.12,<3.15"
# dependencies = ["diffpes==2026.06.13"]
# ///
"""Simulate a photon-energy scan through a compact Dirac reference path.

The automaton evaluates the public photon-energy scan driver and its Fermi
level momentum mesh. It writes a momentum-by-photon-energy map and raw arrays.
Smoke mode uses at most six photon energies and a short momentum path.
"""

from __future__ import annotations

from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
from beartype.typing import Any, Dict, List

import diffpes as dp


@dp.harness.experiment(
    name="photon-energy-scan",
    params=(
        dp.types.make_automaton_param(
            "hv_min_ev",
            float,
            default=20.0,
            help="Lowest photon energy in eV.",
            bounds=(10.0, 200.0),
            example=20.0,
        ),
        dp.types.make_automaton_param(
            "hv_max_ev",
            float,
            default=60.0,
            help="Highest photon energy in eV.",
            bounds=(10.1, 250.0),
            example=60.0,
        ),
        dp.types.make_automaton_param(
            "n_hv",
            int,
            default=13,
            help="Number of photon-energy samples.",
            bounds=(2.0, 256.0),
            example=13,
        ),
        dp.types.make_automaton_param(
            "inner_potential_ev",
            float,
            default=12.0,
            help="Free-electron inner potential in eV.",
            bounds=(0.1, 80.0),
            example=12.0,
        ),
        dp.types.make_automaton_param(
            "kz_broadening_inv_ang",
            float,
            default=0.08,
            help="Out-of-plane Lorentzian width in inverse Angstrom.",
            bounds=(0.001, 2.0),
            example=0.08,
        ),
    ),
    returns={
        "metrics": {
            "kz_min_inv_ang": {"type": "number"},
            "kz_max_inv_ang": {"type": "number"},
            "n_hv": {"type": "integer"},
            "intensity_periodicity_ev": {"type": "number"},
        },
        "artifacts": {"roles": ["hv_map", "hv_arrays"]},
    },
)
def main(
    args: SimpleNamespace,
    ctx: dp.types.AutomatonContext,
) -> Dict[str, Any]:
    """Run one photon-energy scan and return its artifacts.

    The body keeps photon energy explicit in the public scan driver. It maps
    the same energies to at-Fermi vertical momenta for the reported range.
    """
    if args.hv_max_ev <= args.hv_min_ev:
        message: str = "hv_max_ev must exceed hv_min_ev"
        raise ValueError(message)
    n_hv: int = min(args.n_hv, 6) if args.smoke else args.n_hv
    n_k: int = 8 if args.smoke else 32
    n_energy: int = 12 if args.smoke else 61
    model: Any = dp.harness.two_orbital_dirac_model(
        velocity_ev_ang=3.3,
        lattice_a_ang=2.0,
    )
    fractional_path: Any = jnp.stack(
        (
            jnp.linspace(-0.08, 0.08, n_k),
            jnp.zeros((n_k,)),
            jnp.zeros((n_k,)),
        ),
        axis=1,
    )
    kpath: Any = dp.types.make_kpath(fractional_path, kz=0.0)
    hamiltonian: Any = dp.tightb.bloch_hamiltonian_batch(
        model,
        fractional_path,
    )
    bands: Any = dp.tightb.diagonalize_tb(model, fractional_path)
    photon_energies: Any = jnp.linspace(
        args.hv_min_ev,
        args.hv_max_ev,
        n_hv,
    )
    polarization: Any = dp.simul.polarization_from_angles(0.4, 0.0, "p")
    geometry: Any = dp.types.make_experiment_geometry(
        photon_energy_ev=args.hv_min_ev,
        polarization=polarization,
        incidence_theta=0.4,
        work_function_ev=4.5,
        inner_potential_ev=args.inner_potential_ev,
        temperature_k=35.0,
        mean_free_path_ang=0.5 / args.kz_broadening_inv_ang,
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
    energy_axis: Any = jnp.linspace(-0.70, 0.10, n_energy)
    scan: Any = dp.simul.simulate_hv_scan(
        hamiltonian,
        bands,
        radial_spec,
        matrix_element_params,
        dp.types.make_radial_quadrature_spec(),
        dp.types.make_final_state_spec(),
        geometry,
        dp.types.make_self_energy_model(gamma=0.030),
        kpath,
        energy_axis,
        photon_energies,
        k_chunk=16,
        energy_chunk=12,
        checkpoint=True,
    )
    hv_map: Any = dp.simul.hv_map_at_energy(scan, energy_axis, 0.0)
    kpar_axis: Any = jnp.linspace(-0.24, 0.24, n_k)
    fermi_mesh: Any = dp.tightb.build_kmesh_hv_at_fermi(
        kpar_axis,
        photon_energies,
        4.5,
        args.inner_potential_ev,
        0.0,
        jnp.asarray((1.0, 0.0)),
        model.geometry,
    )
    mesh_cartesian: Any = dp.tightb.kpoints_frac_to_cart(
        fermi_mesh.kpoints,
        model.geometry,
    )
    vertical_momentum: Any = mesh_cartesian[:, 2]
    path_cartesian: Any = dp.tightb.kpoints_frac_to_cart(
        fractional_path,
        model.geometry,
    )
    momentum_axis: Any = jnp.linalg.norm(
        path_cartesian - path_cartesian[0],
        axis=1,
    )
    intensity_trace: Any = np.asarray(jnp.sum(hv_map, axis=0))
    minimum_peak_count: int = 2
    local_peak_indices: Any = (
        np.flatnonzero(
            (intensity_trace[1:-1] > intensity_trace[:-2])
            & (intensity_trace[1:-1] > intensity_trace[2:])
        )
        + 1
    )
    periodicity_ev: float
    if local_peak_indices.size >= minimum_peak_count:
        periodicity_ev = float(
            np.median(np.diff(np.asarray(photon_energies)[local_peak_indices]))
        )
    else:
        periodicity_ev = 0.0
    map_figure: Any
    map_figure, _, _ = dp.plots.plot_momentum_map(
        hv_map,
        momentum_axis,
        photon_energies,
        title="Dirac photon-energy intensity map",
    )
    metrics: Dict[str, Any] = {
        "kz_min_inv_ang": float(jnp.min(vertical_momentum)),
        "kz_max_inv_ang": float(jnp.max(vertical_momentum)),
        "n_hv": n_hv,
        "intensity_periodicity_ev": periodicity_ev,
    }
    artifacts: List[Any] = [
        dp.harness.save_figure_artifact(
            ctx,
            "hv_map.png",
            map_figure,
            role="hv_map",
        ),
        dp.harness.save_array_artifact(
            ctx,
            "hv_scan.npz",
            {
                "energy_axis_ev": energy_axis,
                "hv_map": hv_map,
                "photon_energies_ev": photon_energies,
                "scan": scan,
                "vertical_momentum_inv_ang": vertical_momentum,
            },
            role="hv_arrays",
        ),
        dp.harness.save_json_artifact(ctx, "metrics.json", metrics),
    ]
    result: Dict[str, Any] = {"metrics": metrics, "artifacts": artifacts}
    return result


if __name__ == "__main__":
    main()
