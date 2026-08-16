# /// script
# requires-python = ">=3.12,<3.15"
# dependencies = ["diffpes==2026.06.13"]
# ///
"""Measure local information in a normalized chain spectral acquisition.

The automaton builds a chain reference model through the public harness. It
forms a compact spectral cut and measures its local Fisher spectrum. The
normalization removes the overall intensity-scale coordinate. Smoke mode uses
four nondegenerate momenta and sixteen energy samples.
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
def _normalized_chain_spectrum(
    parameters: Float64[Array, " n_parameter"],
    energy_axis: Float64[Array, " n_energy"],
    kpoints: Float64[Array, "n_k 3"],
    include_polarization: bool,
) -> Float64[Array, " n_output"]:
    """PRIVATE: Build a normalized chain spectrum from local coordinates.

    Parameters
    ----------
    parameters : Float64[Array, " n_parameter"]
        Hopping, linewidth, intensity scale, and optional polarization angle.
    energy_axis : Float64[Array, " n_energy"]
        Relative-energy samples in eV.
    kpoints : Float64[Array, "n_k 3"]
        Fixed fractional momenta away from a band degeneracy.
    include_polarization : bool
        Whether the final coordinate modulates the momentum weights.

    Returns
    -------
    spectrum : Float64[Array, " n_output"]
        Flattened normalized intrinsic spectral intensity.

    Notes
    -----
    Builds the public chain model and spectral assembler composition. The
    positive overall scale cancels during normalization by construction.
    """
    model: dp.types.TBModel = dp.harness.linear_chain_model(
        hopping_ev=parameters[0]
    )
    eigenvalues: Float64[Array, "n_k n_band"] = dp.tightb.eigvalsh_bands(
        model,
        kpoints,
    )
    base_weights: Float64[Array, "n_k n_energy n_band"] = jnp.broadcast_to(
        jnp.ones_like(eigenvalues)[:, None, :],
        (eigenvalues.shape[0], energy_axis.shape[0], eigenvalues.shape[1]),
    )
    if include_polarization:
        momentum_coordinate: Float64[Array, " n_k"] = jnp.linspace(
            -1.0,
            1.0,
            kpoints.shape[0],
            dtype=jnp.float64,
        )
        polarization_weight: Float64[Array, " n_k"] = (
            1.0 + 0.2 * jnp.cos(parameters[3]) * momentum_coordinate
        )
        band_weights: Float64[Array, "n_k n_energy n_band"] = (
            base_weights * polarization_weight[:, None, None]
        )
    else:
        band_weights = base_weights
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
    scaled_intensity: Float64[Array, "n_k n_energy"] = (
        parameters[2] * intensity
    )
    normalization: Float64[Array, ""] = jnp.sum(scaled_intensity)
    normalized_intensity: Float64[Array, "n_k n_energy"] = (
        scaled_intensity / jnp.maximum(normalization, 1.0e-30)
    )
    spectrum: Float64[Array, " n_output"] = jnp.ravel(normalized_intensity)
    return spectrum


@dp.harness.experiment(
    name="information-spectrum",
    params=(
        dp.types.make_automaton_param(
            "rank",
            int,
            default=3,
            help="Requested local information rank.",
            bounds=(1.0, 4.0),
            example=3,
        ),
        dp.types.make_automaton_param(
            "iterations",
            int,
            default=4,
            help="Subspace iterations for the spectrum estimate.",
            bounds=(1.0, 16.0),
            example=4,
        ),
        dp.types.make_automaton_param(
            "threshold",
            float,
            default=1.0e-10,
            help="Active singular-value threshold.",
            bounds=(1.0e-16, 1.0),
            example=1.0e-10,
        ),
        dp.types.make_automaton_param(
            "noise_model",
            str,
            default="gaussian",
            help="Output weighting model.",
            choices=("gaussian", "poisson"),
            example="gaussian",
        ),
        dp.types.make_automaton_param(
            "include_polarization",
            bool,
            default=False,
            help="Include one momentum-dependent polarization coordinate.",
            example=False,
        ),
    ),
    returns={
        "metrics": {
            "effective_rank": {"type": "integer"},
            "condition_estimate": {"type": "number"},
            "min_singular_value": {"type": "number"},
            "gauge_nullspace_max_residual": {"type": "number"},
            "crb_diag": {"type": "array"},
        },
        "artifacts": {
            "roles": [
                "singular_spectrum",
                "information_arrays",
                "information_report",
            ]
        },
    },
)
def main(
    args: SimpleNamespace,
    ctx: dp.types.AutomatonContext,
) -> Dict[str, Any]:
    """Measure local information and return spectrum artifacts.

    The body calls the public matrix-free information estimator. It checks an
    explicit overall-scale tangent against the normalized forward map.
    """
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
    coordinate_names: Tuple[str, ...] = (
        "hopping_ev",
        "gamma_ev",
        "overall_intensity_scale",
    )
    parameters: Float64[Array, " n_parameter"] = jnp.asarray(
        (-1.2, 0.18, 2.5),
        dtype=jnp.float64,
    )
    if args.include_polarization:
        parameters = jnp.concatenate(
            (
                parameters,
                jnp.asarray((0.35,), dtype=jnp.float64),
            )
        )
        coordinate_names = (*coordinate_names, "polarization_angle_rad")

    def forward(
        candidate: Float64[Array, " n_parameter"],
    ) -> Float64[Array, " n_output"]:
        """PRIVATE: Evaluate the normalized reference spectrum.

        Parameters
        ----------
        candidate : Float64[Array, " n_parameter"]
            Candidate local coordinates for the reference acquisition.

        Returns
        -------
        spectrum : Float64[Array, " n_output"]
            Flattened normalized spectral intensity.

        Notes
        -----
        Binds the fixed axes so the public information routine receives one
        pure differentiable function of its coordinate vector.
        """
        spectrum: Float64[Array, " n_output"] = _normalized_chain_spectrum(
            candidate,
            energy_axis,
            kpoints,
            args.include_polarization,
        )
        return spectrum

    output: Float64[Array, " n_output"] = forward(parameters)
    output_weights: Float64[Array, " n_output"]
    if args.noise_model == "poisson":
        output_weights = 1.0 / jnp.maximum(output, 1.0e-12)
    else:
        output_weights = jnp.ones_like(output)
    requested_rank: int = min(args.rank, parameters.shape[0])
    spectrum: dp.types.InformationSpectrum = dp.certify.information_spectrum(
        forward,
        parameters,
        input_paths=coordinate_names,
        output_weights=output_weights,
        rank=requested_rank,
        iterations=args.iterations,
        threshold=args.threshold,
    )
    jacobian: Float64[Array, "n_output n_parameter"] = jax.jacfwd(forward)(
        parameters
    )
    gauge_tangent: Float64[Array, " n_parameter"] = (
        jnp.zeros_like(parameters).at[2].set(1.0)
    )
    gauge_response: Float64[Array, " n_output"] = jax.jvp(
        forward,
        (parameters,),
        (gauge_tangent,),
    )[1]
    gauge_residual: Float64[Array, ""] = jnp.linalg.norm(gauge_response)
    fisher: Float64[Array, "n_parameter n_parameter"] = jacobian.T @ (
        output_weights[:, None] * jacobian
    )
    crb: Float64[Array, "n_parameter n_parameter"] = jnp.linalg.pinv(fisher)
    crb_diag: Float64[Array, " n_parameter"] = jnp.diag(crb)
    singular_values: Float64[Array, " n_singular"] = spectrum.singular_values
    singular_axis: Float64[Array, " n_singular"] = jnp.arange(
        1,
        singular_values.shape[0] + 1,
        dtype=jnp.float64,
    )
    figure: Any
    figure, _, _ = dp.plots.plot_curve_family(
        singular_axis,
        (singular_values,),
        labels=("singular value",),
        xlabel="Spectrum index",
        ylabel="Singular value",
        title="Local information spectrum",
    )
    active_rank: int = int(spectrum.effective_rank)
    min_singular_value: float = float(jnp.min(singular_values))
    metrics: Dict[str, Any] = {
        "effective_rank": active_rank,
        "condition_estimate": float(spectrum.condition_estimate),
        "min_singular_value": min_singular_value,
        "gauge_nullspace_max_residual": float(gauge_residual),
        "crb_diag": [float(value) for value in crb_diag],
        "n_parameters": int(parameters.shape[0]),
        "n_gauge": 1,
    }
    report: Dict[str, Any] = {
        "coordinate_names": list(coordinate_names),
        "gauge_tangents": {"overall_intensity_scale": gauge_tangent},
        "singular_values": singular_values,
        "metrics": metrics,
    }
    artifacts: List[dp.types.ArtifactRecord] = [
        dp.harness.save_figure_artifact(
            ctx,
            "singular_spectrum.png",
            figure,
            role="singular_spectrum",
        ),
        dp.harness.save_array_artifact(
            ctx,
            "information_arrays.npz",
            {
                "jacobian": jacobian,
                "fisher": fisher,
                "crb": crb,
                "gauge_tangent": gauge_tangent,
                "gauge_response": gauge_response,
                "singular_values": singular_values,
            },
            role="information_arrays",
        ),
        dp.harness.save_json_artifact(
            ctx,
            "information_report.json",
            report,
            role="information_report",
        ),
        dp.harness.save_json_artifact(ctx, "metrics.json", metrics),
    ]
    result: Dict[str, Any] = {"metrics": metrics, "artifacts": artifacts}
    return result


if __name__ == "__main__":
    main()
