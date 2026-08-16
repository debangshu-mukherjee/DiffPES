# /// script
# requires-python = ">=3.12,<3.15"
# dependencies = ["diffpes==2026.06.13"]
# ///
"""Render a tight-binding band path and persist its native eigensystem.

The automaton selects a graphene, chain, or Wannier model. It calls public
DiffPES path and diagonalization functions, then writes a figure and carriers.
Smoke mode uses graphene with no more than 32 points per segment.
"""

from __future__ import annotations

import lzma
from pathlib import Path
from types import SimpleNamespace

import jax.numpy as jnp
from beartype.typing import Any, Dict, List, Tuple

import diffpes as dp


@dp.harness.experiment(
    name="forward-bands",
    params=(
        dp.types.make_automaton_param(
            "model",
            str,
            default="graphene",
            help="Reference model selector.",
            choices=("graphene", "chain", "wannier"),
            example="graphene",
        ),
        dp.types.make_automaton_param(
            "hr_path",
            str,
            default=(
                "tests/test_diffpes/_reference_data/"
                "wannier90_wse2_soc_11bnd_hr.dat.xz"
            ),
            help="Wannier90 hr file for the Wannier model.",
            example=(
                "tests/test_diffpes/_reference_data/"
                "wannier90_wse2_soc_11bnd_hr.dat.xz"
            ),
        ),
        dp.types.make_automaton_param(
            "hopping_ev",
            float,
            default=-2.7,
            help="Nearest-neighbor hopping in eV.",
            bounds=(-20.0, 20.0),
            example=-2.7,
        ),
        dp.types.make_automaton_param(
            "n_k",
            int,
            default=201,
            help="Points in each path segment.",
            bounds=(8.0, 4096.0),
            example=201,
        ),
        dp.types.make_automaton_param(
            "path_labels",
            list,
            default=["Gamma", "K", "M", "Gamma"],
            help="Labels for the fixed symmetry path.",
            example=["Gamma", "K", "M", "Gamma"],
        ),
    ),
    returns={
        "metrics": {
            "n_bands": {"type": "integer"},
            "bandwidth_ev": {"type": "number"},
            "min_direct_gap_ev": {"type": "number"},
            "fermi_energy_ev": {"type": "number"},
        },
        "artifacts": {"roles": ["band_dispersion", "band_arrays", "bands_h5"]},
    },
)
def main(
    args: SimpleNamespace,
    ctx: dp.types.AutomatonContext,
) -> Dict[str, Any]:
    """Run one band-path calculation and return its artifacts.

    The body diagonalizes a public tight-binding model on a labelled path.
    It stores an HDF5 carrier, compact arrays, a band figure, and metrics.
    """
    n_per_segment: int = min(args.n_k, 32) if args.smoke else args.n_k
    path_labels: Tuple[str, ...] = tuple(
        str(label) for label in args.path_labels
    )
    anchors: Any = jnp.asarray(
        (
            (0.0, 0.0, 0.0),
            (2.0 / 3.0, 1.0 / 3.0, 0.0),
            (0.5, 0.0, 0.0),
            (0.0, 0.0, 0.0),
        ),
        dtype=jnp.float64,
    )
    model: Any
    if args.model == "graphene":
        model = dp.harness.graphene_pz_model(hopping_ev=args.hopping_ev)
    elif args.model == "chain":
        model = dp.harness.linear_chain_model(hopping_ev=args.hopping_ev)
    else:
        source_path: Path = Path(args.hr_path)
        if source_path.suffix == ".xz":
            decompressed_path: Path = dp.harness.artifact_path(
                ctx,
                "wannier_hr.dat",
            )
            decompressed_bytes: bytes = lzma.decompress(
                source_path.read_bytes()
            )
            decompressed_path.write_bytes(decompressed_bytes)
            source_path = decompressed_path
        source_lines: List[str] = [
            line
            for line in source_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        minimum_header_lines: int = 2
        if len(source_lines) < minimum_header_lines:
            message: str = "Wannier90 hr file lacks its orbital-count line"
            raise ValueError(message)
        n_wannier: int = int(source_lines[1])
        geometry: Any = dp.types.make_crystal_geometry(
            lattice=jnp.eye(3, dtype=jnp.float64),
            positions=jnp.zeros((1, 3), dtype=jnp.float64),
            species=("Wannier cell",),
        )
        basis: Any = dp.types.make_orbital_basis(
            atom_indices=(0,) * n_wannier,
            n=(1,) * n_wannier,
            l=(0,) * n_wannier,
            m=(0,) * n_wannier,
            labels=tuple(f"wannier_{index}" for index in range(n_wannier)),
        )
        model, _ = dp.inout.read_wannier90_hr(
            str(source_path),
            geometry,
            basis,
            jnp.zeros((n_wannier, 3), dtype=jnp.float64),
        )
    if len(path_labels) != anchors.shape[0]:
        message = "path_labels must contain four labels"
        raise ValueError(message)
    path: Any = dp.tightb.build_kpath(
        anchors,
        model.geometry,
        n_per_segment,
        path_labels,
    )
    bands: Any = dp.tightb.diagonalize_tb(model, path.kpoints)
    momentum_axis: Any = dp.tightb.kpath_arc_length(path, model.geometry)
    band_range: Any = jnp.max(bands.eigenvalues) - jnp.min(bands.eigenvalues)
    min_direct_gap_ev: float
    if bands.eigenvalues.shape[1] > 1:
        min_direct_gap_ev = float(jnp.min(jnp.diff(bands.eigenvalues, axis=1)))
    else:
        min_direct_gap_ev = 0.0
    band_figure: Any
    band_figure, _, _ = dp.plots.plot_band_dispersion(
        bands,
        momentum_axis=momentum_axis,
        title="Tight-binding band dispersion",
    )
    metrics: Dict[str, Any] = {
        "n_bands": int(bands.eigenvalues.shape[1]),
        "bandwidth_ev": float(band_range),
        "min_direct_gap_ev": min_direct_gap_ev,
        "fermi_energy_ev": float(bands.fermi_energy),
    }
    artifacts: List[Any] = [
        dp.harness.save_figure_artifact(
            ctx,
            "bands.png",
            band_figure,
            role="band_dispersion",
        ),
        dp.harness.save_array_artifact(
            ctx,
            "bands.npz",
            {
                "eigenvalues_ev": bands.eigenvalues,
                "momentum_axis_inv_ang": momentum_axis,
            },
            role="band_arrays",
        ),
        dp.harness.save_carrier_artifact(
            ctx,
            "bands.h5",
            bands,
            role="bands_h5",
        ),
        dp.harness.save_json_artifact(ctx, "metrics.json", metrics),
    ]
    result: Dict[str, Any] = {"metrics": metrics, "artifacts": artifacts}
    return result


if __name__ == "__main__":
    main()
