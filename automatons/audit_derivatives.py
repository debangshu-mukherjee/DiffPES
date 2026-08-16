# /// script
# requires-python = ">=3.12,<3.15"
# dependencies = ["diffpes==2026.06.13"]
# ///
"""Audit local spectral derivatives against central finite differences.

The automaton evaluates a chain spectral cut at nondegenerate momenta. It uses
the public derivative-evidence, linearization, and dependency APIs. Smoke mode
uses two active coordinates with four momenta and sixteen energy samples.
"""

from __future__ import annotations

from types import SimpleNamespace

import jax
import jax.numpy as jnp
from beartype import beartype
from beartype.typing import Any, Callable, Dict, List, Tuple
from jaxtyping import Array, Bool, Float64, jaxtyped

import diffpes as dp

type _ForwardFunction = Callable[
    [Float64[Array, " n_parameter"]],
    Float64[Array, " n_output"],
]


@jaxtyped(typechecker=beartype)
def _reference_spectrum(
    parameters: Float64[Array, " 2"],
    energy_axis: Float64[Array, " n_energy"],
    kpoints: Float64[Array, "n_k 3"],
) -> Float64[Array, " n_output"]:
    """PRIVATE: Build an intrinsic chain spectrum for derivative checks.

    Parameters
    ----------
    parameters : Float64[Array, " 2"]
        Hopping and linewidth coordinates for the reference model.
    energy_axis : Float64[Array, " n_energy"]
        Relative-energy samples in eV.
    kpoints : Float64[Array, "n_k 3"]
        Fixed fractional momenta away from a band degeneracy.

    Returns
    -------
    spectrum : Float64[Array, " n_output"]
        Flattened intrinsic spectral intensity.

    Notes
    -----
    Combines the public chain reference model with the spectral assembler.
    Both coordinates change the line shape at the selected finite momenta.
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
    spectrum: Float64[Array, " n_output"] = jnp.ravel(intensity)
    return spectrum


@jaxtyped(typechecker=beartype)
def _evaluate_derivatives(
    forward_fn: _ForwardFunction,
    parameters: Float64[Array, " n_parameter"],
    step: float,
    rtol: float,
    fail_on_violation: bool,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """PRIVATE: Evaluate derivative evidence and enforce optional failure.

    Parameters
    ----------
    forward_fn : _ForwardFunction
        Pure spectral forward map with one coordinate-vector argument.
    parameters : Float64[Array, " n_parameter"]
        Reference coordinate vector for the derivative audit.
    step : float
        Relative central-difference step.
    rtol : float
        Relative tolerance for retained finite-difference residuals.
    fail_on_violation : bool
        Whether a failed derivative condition raises an exception.

    Returns
    -------
    metrics : Dict[str, Any]
        JSON-ready derivative summary metrics.
    arrays : Dict[str, Any]
        Continuous evidence and dependency arrays.

    Raises
    ------
    RuntimeError
        If ``fail_on_violation`` is true and a derivative condition fails.

    Notes
    -----
    Supplies explicit tangent and cotangent probes to the public evidence API.
    It separately counts silent zero columns in the dense local Jacobian.
    """
    output: Float64[Array, " n_output"] = forward_fn(parameters)
    n_parameters: int = int(parameters.shape[0])
    identity: Float64[Array, "n_parameter n_parameter"] = jnp.eye(
        n_parameters,
        dtype=jnp.float64,
    )
    evidence_inputs: Tuple[Float64[Array, ""], ...] = tuple(
        parameters[index] for index in range(n_parameters)
    )
    directions: Tuple[Float64[Array, " n_parameter"], ...] = tuple(
        identity[:, index] for index in range(n_parameters)
    )
    cotangents: Float64[Array, "n_parameter n_output"] = jnp.eye(
        output.shape[0],
        dtype=jnp.float64,
    )[:n_parameters]
    input_paths: Tuple[str, ...] = tuple(
        f"parameter_{index}" for index in range(n_parameters)
    )
    output_ids: Tuple[str, ...] = tuple(
        f"output_{index}" for index in range(n_parameters)
    )
    scales: Float64[Array, " n_parameter"] = jnp.maximum(
        jnp.abs(parameters),
        1.0,
    )

    def evidence_forward(
        coordinates: Tuple[Float64[Array, ""], ...],
    ) -> Float64[Array, " n_output"]:
        """PRIVATE: Adapt scalar leaves to the vector reference function.

        Parameters
        ----------
        coordinates : Tuple[Float64[Array, ""], ...]
            Scalar coordinate leaves for the public evidence interface.

        Returns
        -------
        spectrum : Float64[Array, " n_output"]
            Flattened spectral intensity from the vector forward function.

        Notes
        -----
        Stacks scalar leaves to expose one named PyTree leaf per coordinate.
        This representation matches the public VJP evidence carrier layout.
        """
        coordinate_vector: Float64[Array, " n_parameter"] = jnp.stack(
            coordinates
        )
        spectrum: Float64[Array, " n_output"] = forward_fn(coordinate_vector)
        return spectrum

    evidence: dp.types.DerivativeEvidence = dp.certify.derivative_evidence(
        evidence_forward,
        evidence_inputs,
        directions,
        cotangents,
        input_paths=input_paths,
        output_projection_ids=output_ids,
        scales=scales,
        step=step,
        spectrum_rank=n_parameters,
    )
    linearized: Tuple[
        Float64[Array, " n_output"],
        Callable[
            [Float64[Array, " n_parameter"]], Float64[Array, " n_output"]
        ],
    ] = dp.certify.linearized_forward(forward_fn, parameters)
    linearized_output: Float64[Array, " n_output"] = linearized[0]
    dependencies: dp.types.DependencyMap = dp.certify.dependency_map(
        "org.diffpes.identifiability.derivative-audit",
        forward_fn,
        parameters,
    )
    jacobian: Float64[Array, "n_output n_parameter"] = jax.jacfwd(forward_fn)(
        parameters
    )
    column_norms: Float64[Array, " n_parameter"] = jnp.linalg.norm(
        jacobian,
        axis=0,
    )
    zero_column_threshold: float = 1.0e-12
    zero_columns: Bool[Array, " n_parameter"] = (
        column_norms <= zero_column_threshold
    )
    zero_column_count: int = int(jnp.sum(zero_columns))
    residual_magnitudes: Float64[Array, "n_probe n_output"] = jnp.abs(
        evidence.derivative_residuals
    )
    reference_magnitudes: Float64[Array, "n_probe n_output"] = jnp.abs(
        evidence.reference_derivatives
    )
    relative_errors: Float64[Array, "n_probe n_output"] = (
        residual_magnitudes / jnp.maximum(reference_magnitudes, 1.0e-12)
    )
    max_relative_error: Float64[Array, ""] = jnp.max(relative_errors)
    coordinate_passes: Bool[Array, " n_parameter"] = (
        jnp.all(relative_errors <= rtol, axis=1) & ~zero_columns
    )
    n_passed: int = int(jnp.sum(coordinate_passes))
    all_passed: bool = bool(
        evidence.finite
        & evidence.fd_correct
        & (max_relative_error <= rtol)
        & (zero_column_count == 0)
    )
    metrics: Dict[str, Any] = {
        "n_parameters": n_parameters,
        "n_passed": n_passed,
        "all_passed": all_passed,
        "max_relative_error": float(max_relative_error),
        "zero_column_count": zero_column_count,
    }
    arrays: Dict[str, Any] = {
        "jacobian": jacobian,
        "column_norms": column_norms,
        "jvp_probes": evidence.jvp_probes,
        "reference_derivatives": evidence.reference_derivatives,
        "derivative_residuals": evidence.derivative_residuals,
        "linearized_output": linearized_output,
        "dependency_structural": dependencies.structural,
        "dependency_traced": dependencies.traced,
    }
    if fail_on_violation and not all_passed:
        message: str = "derivative audit detects a failed coordinate condition"
        raise RuntimeError(message)
    result: Tuple[Dict[str, Any], Dict[str, Any]] = (metrics, arrays)
    return result


@dp.harness.experiment(
    name="audit-derivatives",
    params=(
        dp.types.make_automaton_param(
            "fail_on_violation",
            bool,
            default=False,
            help="Raise when a finite-difference or zero-column check fails.",
            example=False,
        ),
        dp.types.make_automaton_param(
            "step",
            float,
            default=1.0e-5,
            help="Relative central-difference step.",
            bounds=(1.0e-8, 1.0e-2),
            example=1.0e-5,
        ),
        dp.types.make_automaton_param(
            "rtol",
            float,
            default=1.0e-6,
            help="Relative derivative agreement tolerance.",
            bounds=(1.0e-10, 1.0e-2),
            example=1.0e-6,
        ),
    ),
    returns={
        "metrics": {
            "n_parameters": {"type": "integer"},
            "n_passed": {"type": "integer"},
            "all_passed": {"type": "boolean"},
            "max_relative_error": {"type": "number"},
            "zero_column_count": {"type": "integer"},
        },
        "artifacts": {"roles": ["derivative_report", "derivative_arrays"]},
    },
)
def main(
    args: SimpleNamespace,
    ctx: dp.types.AutomatonContext,
) -> Dict[str, Any]:
    """Audit one reference spectrum and return derivative artifacts.

    The body creates a nondegenerate reference model. It records automatic and
    central finite-difference evidence for every active coordinate.
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
    parameters: Float64[Array, " 2"] = jnp.asarray(
        (-1.2, 0.18),
        dtype=jnp.float64,
    )

    def forward(
        candidate: Float64[Array, " 2"],
    ) -> Float64[Array, " n_output"]:
        """PRIVATE: Evaluate the fixed reference spectral cut.

        Parameters
        ----------
        candidate : Float64[Array, " 2"]
            Hopping and linewidth coordinates for the reference model.

        Returns
        -------
        spectrum : Float64[Array, " n_output"]
            Flattened intrinsic spectral intensity.

        Notes
        -----
        Captures the fixed axes so the derivative evidence receives one pure
        differentiable function of two physical coordinates.
        """
        spectrum: Float64[Array, " n_output"] = _reference_spectrum(
            candidate,
            energy_axis,
            kpoints,
        )
        return spectrum

    metrics: Dict[str, Any]
    arrays: Dict[str, Any]
    metrics, arrays = _evaluate_derivatives(
        forward,
        parameters,
        args.step,
        args.rtol,
        args.fail_on_violation,
    )
    report: Dict[str, Any] = {"metrics": metrics, "parameters": parameters}
    artifacts: List[dp.types.ArtifactRecord] = [
        dp.harness.save_json_artifact(
            ctx,
            "derivative_report.json",
            report,
            role="derivative_report",
        ),
        dp.harness.save_array_artifact(
            ctx,
            "derivative_arrays.npz",
            arrays,
            role="derivative_arrays",
        ),
        dp.harness.save_json_artifact(ctx, "metrics.json", metrics),
    ]
    result: Dict[str, Any] = {"metrics": metrics, "artifacts": artifacts}
    return result


if __name__ == "__main__":
    main()
