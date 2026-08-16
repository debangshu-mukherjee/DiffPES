# /// script
# requires-python = ">=3.12,<3.15"
# dependencies = ["diffpes==2026.06.13"]
# ///
"""Rank compact spectral acquisitions by their local information content.

The automaton evaluates chain spectral cuts for candidate acquisition settings.
It forms a Fisher block for the requested physical coordinates and ranks the
designs by its log determinant. Smoke mode evaluates three small candidates.
"""

from __future__ import annotations

from types import SimpleNamespace

import jax
import jax.numpy as jnp
from beartype import beartype
from beartype.typing import Any, Dict, List, Mapping, Sequence, Tuple
from jaxtyping import Array, Float64, jaxtyped

import diffpes as dp


@jaxtyped(typechecker=beartype)
def _weighted_chain_spectrum(
    parameters: Float64[Array, " n_parameter"],
    energy_axis: Float64[Array, " n_energy"],
    kpoints: Float64[Array, "n_k 3"],
    acquisition_weight: Float64[Array, ""],
) -> Float64[Array, " n_output"]:
    """PRIVATE: Build a weighted chain spectrum for one acquisition.

    Parameters
    ----------
    parameters : Float64[Array, " n_parameter"]
        Hopping and linewidth coordinates for the reference model.
    energy_axis : Float64[Array, " n_energy"]
        Relative-energy samples in eV.
    kpoints : Float64[Array, "n_k 3"]
        Fixed fractional momenta away from a band degeneracy.
    acquisition_weight : Float64[Array, ""]
        Positive exposure and resolution weight for one design.

    Returns
    -------
    spectrum : Float64[Array, " n_output"]
        Flattened weighted spectral intensity.

    Notes
    -----
    Uses public chain and spectral-assembly functions. The acquisition weight
    represents the independent count information of the selected energy range.
    """
    model: dp.types.TBModel = dp.harness.linear_chain_model(
        hopping_ev=parameters[0]
    )
    eigenvalues: Float64[Array, "n_k n_band"] = dp.tightb.eigvalsh_bands(
        model,
        kpoints,
    )
    band_weights: Float64[Array, "n_k n_energy n_band"] = jnp.broadcast_to(
        jnp.ones_like(eigenvalues)[:, None, :],
        (eigenvalues.shape[0], energy_axis.shape[0], eigenvalues.shape[1]),
    )
    self_energy: dp.types.SelfEnergyModel = dp.types.make_self_energy_model(
        gamma=parameters[1]
    )
    intensity: Float64[Array, "n_k n_energy"] = (
        dp.simul.assemble_spectral_intensity_bands_chunk(
            eigenvalues,
            band_weights,
            energy_axis,
            self_energy,
            jnp.asarray(0.0, dtype=jnp.float64),
            30.0,
        )
    )
    weighted_intensity: Float64[Array, "n_k n_energy"] = (
        acquisition_weight * intensity
    )
    spectrum: Float64[Array, " n_output"] = jnp.ravel(weighted_intensity)
    return spectrum


@dp.harness.experiment(
    name="experiment-design-compare",
    params=(
        dp.types.make_automaton_param(
            "designs",
            list,
            default=[
                {
                    "hv_ev": 22.0,
                    "polarization": "p",
                    "temperature_k": 30.0,
                    "energy_resolution_ev": 0.02,
                },
                {
                    "hv_ev": 26.0,
                    "polarization": "p",
                    "temperature_k": 30.0,
                    "energy_resolution_ev": 0.02,
                },
                {
                    "hv_ev": 30.0,
                    "polarization": "p",
                    "temperature_k": 30.0,
                    "energy_resolution_ev": 0.02,
                },
            ],
            help="Candidate acquisition dictionaries.",
            example=[
                {
                    "hv_ev": 22.0,
                    "polarization": "p",
                    "temperature_k": 30.0,
                    "energy_resolution_ev": 0.02,
                }
            ],
        ),
        dp.types.make_automaton_param(
            "target_parameters",
            list,
            default=["hopping_ev", "gamma_ev"],
            help="Physical coordinates used in each Fisher block.",
            example=["hopping_ev", "gamma_ev"],
        ),
    ),
    returns={
        "ranked": {"type": "array"},
        "metrics": {
            "best_design_index": {"type": "integer"},
            "logdet_information": {"type": "array"},
            "crb_trace": {"type": "array"},
        },
        "artifacts": {
            "roles": ["design_ranking", "design_bars", "design_arrays"]
        },
    },
)
def main(  # noqa: PLR0912, PLR0915
    args: SimpleNamespace,
    ctx: dp.types.AutomatonContext,
) -> Dict[str, Any]:
    """Rank candidate acquisitions and return the design artifacts.

    The body computes a public information spectrum for each design. It uses
    the Fisher block to rank candidates and estimate the CRB trace.
    """
    designs: Sequence[Any] = args.designs
    target_names: Tuple[str, ...] = tuple(
        str(name) for name in args.target_parameters
    )
    supported_targets: Tuple[str, ...] = ("hopping_ev", "gamma_ev")
    if not designs:
        message: str = "designs must contain at least one acquisition"
        raise ValueError(message)
    if not target_names:
        message = "target_parameters must contain at least one coordinate"
        raise ValueError(message)
    if any(name not in supported_targets for name in target_names):
        message = "target_parameters support hopping_ev and gamma_ev"
        raise ValueError(message)
    if len(set(target_names)) != len(target_names):
        message = "target_parameters must not repeat a coordinate"
        raise ValueError(message)
    max_smoke_designs: int = 3
    if len(designs) > max_smoke_designs and args.smoke:
        designs = designs[:max_smoke_designs]
    n_energy: int = 16 if args.smoke else 48
    energy_axis: Float64[Array, " n_energy"] = jnp.linspace(
        -3.0,
        3.0,
        n_energy,
        dtype=jnp.float64,
    )
    kpoints: Float64[Array, "4 3"] = jnp.asarray(
        (
            (0.07, 0.0, 0.0),
            (0.18, 0.0, 0.0),
            (0.31, 0.0, 0.0),
            (0.44, 0.0, 0.0),
        ),
        dtype=jnp.float64,
    )
    parameters: Float64[Array, " 2"] = jnp.asarray(
        (-1.2, 0.18),
        dtype=jnp.float64,
    )
    target_indices: Tuple[int, ...] = tuple(
        (0 if name == "hopping_ev" else 1) for name in target_names
    )
    target_index_array: Array = jnp.asarray(
        target_indices,
        dtype=jnp.int32,
    )
    target_values: Float64[Array, " n_target"] = parameters[target_index_array]
    logdets: List[float] = []
    crb_traces: List[float] = []
    singular_rows: List[Float64[Array, " n_singular"]] = []
    ranking_rows: List[Dict[str, Any]] = []
    design_index: int
    design_value: Any
    for design_index, design_value in enumerate(designs):
        if not isinstance(design_value, Mapping):
            message = "each design must be a JSON object"
            raise ValueError(message)
        design: Mapping[str, Any] = design_value
        photon_energy: float = float(design.get("hv_ev", 22.0))
        temperature: float = float(design.get("temperature_k", 30.0))
        resolution: float = float(design.get("energy_resolution_ev", 0.02))
        polarization: str = str(design.get("polarization", "p"))
        minimum_photon_energy: float = 20.0
        if (
            photon_energy <= minimum_photon_energy
            or temperature <= 0.0
            or resolution <= 0.0
        ):
            message = (
                "each design needs positive photon, temperature, and "
                "resolution values"
            )
            raise ValueError(message)
        polarization_factor: float
        if polarization == "p":
            polarization_factor = 1.0
        elif polarization == "s":
            polarization_factor = 0.9
        elif polarization in {"c+", "c-"}:
            polarization_factor = 0.85
        elif polarization == "linear":
            polarization_factor = 0.95
        else:
            message = "polarization must use a public selector"
            raise ValueError(message)
        energy_window: float = photon_energy - minimum_photon_energy
        acquisition_weight: Float64[Array, ""] = jnp.asarray(
            jnp.sqrt(energy_window)
            * polarization_factor
            / jnp.sqrt((temperature / 30.0) * (resolution / 0.02)),
            dtype=jnp.float64,
        )

        def forward(
            candidate: Float64[Array, " n_target"],
            weight: Float64[Array, ""] = acquisition_weight,
        ) -> Float64[Array, " n_output"]:
            """PRIVATE: Evaluate one design at candidate physical coordinates.

            Parameters
            ----------
            candidate : Float64[Array, " n_target"]
                Values for the requested physical coordinates.
            weight : Float64[Array, ""]
                Bound acquisition weight for the selected candidate design.

            Returns
            -------
            spectrum : Float64[Array, " n_output"]
                Flattened weighted spectral intensity.

            Notes
            -----
            Restores the selected coordinates in the physical reference vector
            before the public matrix-free information calculation.
            """
            full_parameters: Float64[Array, " 2"] = parameters.at[
                target_index_array
            ].set(candidate)
            spectrum: Float64[Array, " n_output"] = _weighted_chain_spectrum(
                full_parameters,
                energy_axis,
                kpoints,
                weight,
            )
            return spectrum

        spectrum: dp.types.InformationSpectrum = (
            dp.certify.information_spectrum(
                forward,
                target_values,
                input_paths=target_names,
                rank=len(target_names),
                iterations=4,
                threshold=1.0e-10,
            )
        )
        jacobian: Float64[Array, "n_output n_target"] = jax.jacfwd(forward)(
            target_values
        )
        fisher: Float64[Array, "n_target n_target"] = jacobian.T @ jacobian
        eigenvalues: Float64[Array, " n_target"] = jnp.linalg.eigvalsh(fisher)
        stable_eigenvalues: Float64[Array, " n_target"] = jnp.maximum(
            eigenvalues,
            1.0e-30,
        )
        logdet: Float64[Array, ""] = jnp.sum(jnp.log(stable_eigenvalues))
        crb_trace: Float64[Array, ""] = jnp.trace(jnp.linalg.pinv(fisher))
        logdets.append(float(logdet))
        crb_traces.append(float(crb_trace))
        singular_rows.append(spectrum.singular_values)
        ranking_rows.append(
            {
                "design_index": design_index,
                "hv_ev": photon_energy,
                "polarization": polarization,
                "temperature_k": temperature,
                "energy_resolution_ev": resolution,
                "logdet_information": float(logdet),
                "crb_trace": float(crb_trace),
            }
        )
    ranked: List[Dict[str, Any]] = sorted(
        ranking_rows,
        key=lambda row: (
            -float(row["logdet_information"]),
            int(row["design_index"]),
        ),
    )
    best_design_index: int = int(ranked[0]["design_index"])
    design_axis: Float64[Array, " n_design"] = jnp.arange(
        len(logdets),
        dtype=jnp.float64,
    )
    logdet_array: Float64[Array, " n_design"] = jnp.asarray(
        logdets,
        dtype=jnp.float64,
    )
    crb_array: Float64[Array, " n_design"] = jnp.asarray(
        crb_traces,
        dtype=jnp.float64,
    )
    figure: Any
    figure, axis, _ = dp.plots.plot_curve_family(
        design_axis,
        (logdet_array,),
        labels=("log determinant",),
        xlabel="Design index",
        ylabel="Log determinant",
        title="Acquisition information comparison",
    )
    axis.bar(design_axis, logdet_array, alpha=0.3)
    metrics: Dict[str, Any] = {
        "best_design_index": best_design_index,
        "logdet_information": logdets,
        "crb_trace": crb_traces,
    }
    artifacts: List[dp.types.ArtifactRecord] = [
        dp.harness.save_json_artifact(
            ctx,
            "design_ranking.json",
            {"ranked": ranked, "target_parameters": list(target_names)},
            role="design_ranking",
        ),
        dp.harness.save_figure_artifact(
            ctx,
            "design_bars.png",
            figure,
            role="design_bars",
        ),
        dp.harness.save_array_artifact(
            ctx,
            "design_arrays.npz",
            {
                "logdet_information": logdet_array,
                "crb_trace": crb_array,
                "singular_values": jnp.stack(singular_rows),
            },
            role="design_arrays",
        ),
        dp.harness.save_json_artifact(ctx, "metrics.json", metrics),
    ]
    result: Dict[str, Any] = {
        "metrics": metrics,
        "artifacts": artifacts,
        "ranked": ranked,
    }
    return result


if __name__ == "__main__":
    main()
