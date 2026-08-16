# /// script
# requires-python = ">=3.12,<3.15"
# dependencies = ["diffpes==2026.06.13"]
# ///
"""Ingest a spectral cut and recover its Fermi edge.

This experiment reads an ARPES spectrum from HDF5 or NPZ data. It also builds
a graphene-derived in-code spectrum when no path is supplied. It normalizes
the cut, fits its angle-integrated edge, and writes figures and arrays.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import optimistix as optx
from beartype import beartype
from beartype.typing import Any, Dict, List, Tuple
from jaxtyping import Array, Float64, jaxtyped

import diffpes as dp


@jaxtyped(typechecker=beartype)
def _fermi_values(
    energy_axis: Float64[Array, " n_energy"],
    fermi_edge_ev: dp.types.ScalarFloat,
    temperature_k: dp.types.ScalarFloat,
) -> Float64[Array, " n_energy"]:
    """PRIVATE: Evaluate the scalar Fermi function across an energy axis.

    Parameters
    ----------
    energy_axis : Float64[Array, " n_energy"]
        Increasing energy coordinates in eV.
    fermi_edge_ev : dp.types.ScalarFloat
        Fermi edge energy in eV.
    temperature_k : dp.types.ScalarFloat
        Positive edge temperature in kelvin.

    Returns
    -------
    occupations : Float64[Array, " n_energy"]
        Fermi occupations at the supplied energy coordinates.

    Notes
    -----
    Vectorizes the public scalar Fermi function. This preserves its temperature
    validation and its stable sigmoid evaluation.
    """
    occupations: Float64[Array, " n_energy"] = jax.vmap(
        dp.simul.fermi_dirac,
        in_axes=(0, None, None),
    )(energy_axis, fermi_edge_ev, temperature_k)
    return occupations


@jaxtyped(typechecker=beartype)
def _graphene_envelope(
    n_k: int,
    n_energy: int,
) -> Tuple[
    Float64[Array, " n_k n_energy"],
    Float64[Array, " n_k"],
    Float64[Array, " n_energy"],
    Float64[Array, " n_k 3"],
]:
    """PRIVATE: Build an unoccupied graphene spectral envelope.

    Parameters
    ----------
    n_k : int
        Number of path points.
    n_energy : int
        Number of energy points.

    Returns
    -------
    envelope : Float64[Array, " n_k n_energy"]
        Positive graphene spectral envelope.
    k_axis : Float64[Array, " n_k"]
        Cumulative Cartesian path distance in inverse angstroms.
    energy_axis : Float64[Array, " n_energy"]
        Relative energy coordinates in eV.
    kpoints : Float64[Array, " n_k 3"]
        Cartesian path coordinates in inverse angstroms.

    Notes
    -----
    Diagonalizes the public graphene reference model along a path through K.
    It evaluates one Lorentzian spectral value for each energy and path point.
    """
    model: dp.types.TBModel = dp.harness.graphene_pz_model()
    path_fractional: Float64[Array, " n_k 3"] = jnp.linspace(
        jnp.asarray([0.28, 1.0 / 3.0, 0.0], dtype=jnp.float64),
        jnp.asarray([0.39, 1.0 / 3.0, 0.0], dtype=jnp.float64),
        n_k,
    )
    kpoints: Float64[Array, " n_k 3"] = dp.tightb.kpoints_frac_to_cart(
        path_fractional,
        model.geometry,
    )
    k_axis: Float64[Array, " n_k"] = jnp.linalg.norm(
        kpoints - kpoints[0],
        axis=1,
    )
    energy_axis: Float64[Array, " n_energy"] = jnp.linspace(
        -0.20,
        0.10,
        n_energy,
        dtype=jnp.float64,
    )
    eigenvalues: Float64[Array, " n_k n_bands"] = dp.tightb.eigvalsh_bands(
        model,
        path_fractional,
    )
    band_weights: Float64[Array, " n_bands"] = jnp.ones(
        eigenvalues.shape[1],
        dtype=jnp.float64,
    )

    def at_energy(
        eigenvalue_row: Float64[Array, " n_bands"],
        energy_ev: Float64[Array, ""],
    ) -> Float64[Array, ""]:
        """PRIVATE: Evaluate one graphene spectral density value.

        Parameters
        ----------
        eigenvalue_row : Float64[Array, " n_bands"]
            Band energies at one path point in eV.
        energy_ev : Float64[Array, ""]
            Query energy in eV.

        Returns
        -------
        intensity : Float64[Array, ""]
            Unoccupied spectral density at the query point.

        Notes
        -----
        Uses the public eigenvalue spectral function with a constant imaginary
        self-energy. The calculation leaves occupation for the edge model.
        """
        intensity: Float64[Array, ""] = dp.simul.spectral_intensity_eigen(
            eigenvalue_row,
            band_weights,
            energy_ev,
            jnp.asarray(-0.025j, dtype=jnp.complex128),
            1.0e-4,
            allow_degenerate_value_only=True,
        )
        return intensity

    envelope: Float64[Array, " n_k n_energy"] = jax.vmap(
        lambda eigenvalue_row: jax.vmap(
            lambda energy_ev: at_energy(eigenvalue_row, energy_ev)
        )(energy_axis)
    )(eigenvalues)
    return envelope, k_axis, energy_axis, kpoints


@jaxtyped(typechecker=beartype)
def _edge_residual(
    parameters: Float64[Array, " 4"],
    data: Tuple[
        Float64[Array, " n_energy"],
        Float64[Array, " n_energy"],
        Float64[Array, " n_energy"],
    ],
) -> Float64[Array, " n_energy"]:
    """PRIVATE: Compute residuals for the edge-fit model.

    Parameters
    ----------
    parameters : Float64[Array, " 4"]
        Edge, temperature, amplitude, and linear-background slope.
    data : tuple
        Energy axis, spectral envelope, and normalized EDC values.

    Returns
    -------
    residual : Float64[Array, " n_energy"]
        Model-minus-observation residual values.

    Notes
    -----
    Multiplies the known graphene envelope by a Fermi edge and a linear
    background. Optimistix minimizes the returned least-squares residual.
    """
    energy_axis: Float64[Array, " n_energy"]
    envelope: Float64[Array, " n_energy"]
    observed: Float64[Array, " n_energy"]
    energy_axis, envelope, observed = data
    fermi_edge_ev: Float64[Array, ""] = parameters[0]
    temperature_k: Float64[Array, ""] = parameters[1]
    amplitude: Float64[Array, ""] = parameters[2]
    slope: Float64[Array, ""] = parameters[3]
    occupation: Float64[Array, " n_energy"] = _fermi_values(
        energy_axis,
        fermi_edge_ev,
        temperature_k,
    )
    background: Float64[Array, " n_energy"] = 1.0 + slope * (
        energy_axis - fermi_edge_ev
    )
    fitted: Float64[Array, " n_energy"] = (
        amplitude * envelope * occupation * background
    )
    residual: Float64[Array, " n_energy"] = fitted - observed
    return residual


@dp.harness.experiment(
    name="arpes-ingest",
    params=[
        dp.types.make_automaton_param(
            "spectrum_path",
            str,
            default="",
            help="HDF5 or NPZ spectrum path. Empty builds a graphene cut.",
            example="measurement.npz",
        ),
        dp.types.make_automaton_param(
            "edge_window_ev",
            float,
            default=0.12,
            help="Symmetric Fermi-edge fitting window in eV.",
            bounds=(0.02, 0.20),
            example=0.12,
        ),
        dp.types.make_automaton_param(
            "initial_temperature_k",
            float,
            default=45.0,
            help="Initial edge temperature in kelvin.",
            bounds=(1.0, 600.0),
            example=45.0,
        ),
    ],
    returns={
        "metrics": {
            "fermi_edge_ev": {"type": "number"},
            "effective_temperature_k": {"type": "number"},
            "k_range_inv_ang": {"type": "number"},
            "energy_range_ev": {"type": "number"},
            "total_intensity": {"type": "number"},
        },
        "artifacts": {
            "roles": ["ingested_cut", "edge_fit_curve", "ingest_arrays"],
        },
    },
)
def main(  # noqa: PLR0915 -- direct CLI branches keep input provenance clear.
    args: SimpleNamespace,
    ctx: dp.types.AutomatonContext,
) -> Dict[str, Any]:
    """Run the spectral ingest and return recovered edge artifacts."""
    if args.spectrum_path:
        source_path: Path = Path(args.spectrum_path)
        if source_path.suffix == ".h5":
            loaded: Any = dp.inout.load_from_h5(
                source_path,
                name="spectrum",
            )
            if not isinstance(loaded, dp.types.ArpesSpectrum):
                raise ValueError("HDF5 input must contain an ArpesSpectrum")
            intensity: Float64[Array, " n_k n_energy"] = loaded.intensity
            k_axis: Float64[Array, " n_k"] = loaded.k_axis
            energy_axis: Float64[Array, " n_energy"] = loaded.energy_axis
            kpoints: Float64[Array, " n_k 3"] = loaded.kpoints_cart_inv_ang
            envelope: Float64[Array, " n_energy"] = jnp.ones_like(energy_axis)
        elif source_path.suffix == ".npz":
            with np.load(source_path, allow_pickle=False) as archive:
                intensity = jnp.asarray(
                    archive["intensity"],
                    dtype=jnp.float64,
                )
                k_axis = jnp.asarray(archive["k_axis"], dtype=jnp.float64)
                energy_axis = jnp.asarray(
                    archive["energy_axis"],
                    dtype=jnp.float64,
                )
            kpoints = jnp.stack(
                (k_axis, jnp.zeros_like(k_axis), jnp.zeros_like(k_axis)),
                axis=1,
            )
            envelope = jnp.ones_like(energy_axis)
        else:
            raise ValueError("spectrum_path must end in .h5 or .npz")
        planted_edge_ev: float | None = None
        planted_temperature_k: float | None = None
    else:
        n_k: int = 20 if args.smoke else 80
        n_energy: int = 48 if args.smoke else 192
        raw_envelope: Float64[Array, " n_k n_energy"]
        raw_envelope, k_axis, energy_axis, kpoints = _graphene_envelope(
            n_k,
            n_energy,
        )
        planted_edge_ev = 0.012
        planted_temperature_k = 52.0
        planted_slope: float = 0.25
        occupation: Float64[Array, " n_energy"] = _fermi_values(
            energy_axis,
            planted_edge_ev,
            planted_temperature_k,
        )
        background: Float64[Array, " n_energy"] = 1.0 + planted_slope * (
            energy_axis - planted_edge_ev
        )
        intensity = raw_envelope * occupation[None, :] * background[None, :]
        envelope = jnp.mean(raw_envelope, axis=0)

    spectrum: dp.types.ArpesSpectrum = dp.types.make_arpes_spectrum(
        intensity,
        energy_axis,
        k_axis,
        kpoints,
    )
    normalized: Float64[Array, " n_k n_energy"] = dp.simul.normalize_intensity(
        spectrum,
        mode="sum",
    )
    edc: Float64[Array, " n_energy"] = jnp.mean(normalized, axis=0)
    fit_mask: jax.Array = jnp.abs(energy_axis) <= args.edge_window_ev
    fitted_energy: Float64[Array, " n_fit"] = energy_axis[fit_mask]
    fitted_envelope: Float64[Array, " n_fit"] = envelope[fit_mask]
    fitted_edc: Float64[Array, " n_fit"] = edc[fit_mask]
    initial_parameters: Float64[Array, " 4"] = jnp.asarray(
        [
            0.0,
            args.initial_temperature_k,
            float(jnp.max(fitted_edc / fitted_envelope)),
            0.0,
        ],
        dtype=jnp.float64,
    )
    solver: optx.LevenbergMarquardt = optx.LevenbergMarquardt(
        rtol=1.0e-10,
        atol=1.0e-12,
    )
    solution: optx.Solution[Float64[Array, " 4"], Any] = optx.least_squares(
        _edge_residual,
        solver,
        initial_parameters,
        args=(fitted_energy, fitted_envelope, fitted_edc),
        max_steps=64 if args.smoke else 256,
        throw=False,
    )
    fitted_parameters: Float64[Array, " 4"] = solution.value
    fitted_model: Float64[Array, " n_energy"] = edc - _edge_residual(
        fitted_parameters,
        (energy_axis, envelope, edc),
    )
    fermi_edge_ev: float = float(fitted_parameters[0])
    effective_temperature_k: float = float(fitted_parameters[1])
    metrics: Dict[str, Any] = {
        "fermi_edge_ev": fermi_edge_ev,
        "effective_temperature_k": effective_temperature_k,
        "k_range_inv_ang": float(k_axis[-1] - k_axis[0]),
        "energy_range_ev": float(energy_axis[-1] - energy_axis[0]),
        "total_intensity": float(jnp.sum(normalized)),
        "fit_rms": float(jnp.sqrt(jnp.mean((fitted_model - edc) ** 2))),
        "planted_fermi_edge_ev": planted_edge_ev,
        "planted_temperature_k": planted_temperature_k,
    }
    cut_figure: Any
    cut_figure, _, _ = dp.plots.plot_spectral_cut(
        normalized,
        k_axis,
        energy_axis,
        colorbar=False,
        title="normalized ARPES ingest",
    )
    edge_figure: Any
    edge_figure, _, _ = dp.plots.plot_curve_family(
        energy_axis,
        (edc, fitted_model),
        labels=("ingested EDC", "Fermi-edge fit"),
        xlabel="energy relative to the Fermi level (eV)",
        ylabel="normalized intensity",
        title="angle-integrated Fermi edge",
    )
    artifacts: List[dp.types.ArtifactRecord] = [
        dp.harness.save_figure_artifact(
            ctx,
            "ingested_cut.png",
            cut_figure,
            role="ingested_cut",
        ),
        dp.harness.save_figure_artifact(
            ctx,
            "edge_fit_curve.png",
            edge_figure,
            role="edge_fit_curve",
        ),
        dp.harness.save_array_artifact(
            ctx,
            "ingest_arrays.npz",
            {
                "energy_axis": energy_axis,
                "fitted_model": fitted_model,
                "intensity": normalized,
                "k_axis": k_axis,
            },
            role="ingest_arrays",
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
