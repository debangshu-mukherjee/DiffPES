"""Define typed sources and evaluated batches for generalized spectra."""

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from beartype import beartype
from beartype.typing import Tuple, Union
from jaxtyping import Array, Complex128, Float64, jaxtyped

from diffpes.constants import HERMITICITY_RELATIVE_TOLERANCE

from .coordinates import MeasurementCoordinates
from .self_energy import SelfEnergyModel


class RetardedValidationReport(eqx.Module):
    """Record necessary-condition checks for a retarded numerical source."""

    report_ref: str = eqx.field(static=True)
    check_ids: Tuple[str, ...] = eqx.field(static=True)
    metric_values: Tuple[float, ...] = eqx.field(static=True)
    tolerance_values: Tuple[float, ...] = eqx.field(static=True)
    metric_units: Tuple[str, ...] = eqx.field(static=True)
    assumptions: Tuple[str, ...] = eqx.field(static=True)
    excluded_claims: Tuple[str, ...] = eqx.field(static=True)
    evidence_refs: Tuple[str, ...] = eqx.field(static=True)
    schema_version: str = eqx.field(static=True)

    def __check_init__(self) -> None:
        """Validate aligned numeric report fields."""
        if not self.report_ref or not self.schema_version:
            raise ValueError("validation report identity must be nonempty")
        size: int = len(self.check_ids)
        if any(
            len(values) != size
            for values in (
                self.metric_values,
                self.tolerance_values,
                self.metric_units,
            )
        ):
            raise ValueError(
                "validation report metrics must align with check_ids"
            )


class SpectralEvaluationRequest(eqx.Module):
    """Specify coordinates, energy, temperature, metric basis, and
    regulator.
    """

    coordinates: MeasurementCoordinates
    omega_rel_fermi_ev: Float64[Array, " n_omega"]
    temperature_k: Float64[Array, " n_temperature"]
    eta_ev: Float64[Array, ""]
    basis_ref: str = eqx.field(static=True)

    def __check_init__(self) -> None:
        """Require finite increasing axes and a positive regulator."""
        if not self.basis_ref:
            raise ValueError("spectral request basis_ref must be nonempty")
        if self.omega_rel_fermi_ev.ndim != 1 or self.temperature_k.ndim != 1:
            raise ValueError("spectral axes must be one-dimensional")
        omega: Float64[Array, " n_omega"] = eqx.error_if(
            self.omega_rel_fermi_ev,
            ~jnp.all(jnp.isfinite(self.omega_rel_fermi_ev))
            | ~jnp.all(jnp.diff(self.omega_rel_fermi_ev) > 0.0),
            "energy axis must be finite and strictly increasing",
        )
        temperature: Float64[Array, " n_temperature"] = eqx.error_if(
            self.temperature_k,
            ~jnp.all(jnp.isfinite(self.temperature_k))
            | ~jnp.all(self.temperature_k >= 0.0)
            | ~jnp.all(jnp.diff(self.temperature_k) > 0.0),
            "temperature axis must be finite, nonnegative, and increasing",
        )
        eta: Float64[Array, ""] = eqx.error_if(
            self.eta_ev,
            ~jnp.isfinite(self.eta_ev) | (self.eta_ev <= 0.0),
            "eta must be finite and strictly positive",
        )
        object.__setattr__(self, "omega_rel_fermi_ev", omega)
        object.__setattr__(self, "temperature_k", temperature)
        object.__setattr__(self, "eta_ev", eta)


class SelfEnergyBatch(eqx.Module):
    """Store covariant orbital self-energy values on one request grid."""

    values_ev: Complex128[Array, "n_temperature n_k n_omega n_orb n_orb"]
    request: SpectralEvaluationRequest
    basis_ref: str = eqx.field(static=True)
    source_ref: str = eqx.field(static=True)
    derivative_mode: str = eqx.field(static=True)


class RetardedGreenBatch(eqx.Module):
    """Store contravariant retarded Green matrices and their overlap metric."""

    values_per_ev: Complex128[Array, "n_temperature n_k n_omega n_orb n_orb"]
    overlap: Complex128[Array, "n_k n_orb n_orb"]
    request: SpectralEvaluationRequest
    basis_ref: str = eqx.field(static=True)
    source_ref: str = eqx.field(static=True)
    derivative_mode: str = eqx.field(static=True)
    validation_ref: str = eqx.field(static=True)


class ParametricSelfEnergy(eqx.Module):
    """Wrap a scalar self-energy parameterization as a typed source.

    The legacy carrier is deliberately an implementation detail of this
    source: generalized evaluators consume only this wrapper.
    """

    parameterization: SelfEnergyModel
    source_ref: str = eqx.field(static=True)
    basis_ref: str = eqx.field(static=True)
    provenance_ref: str = eqx.field(static=True)
    derivative_mode: str = eqx.field(static=True, default="exact_ad")


class TabulatedMatrixSelfEnergy(eqx.Module):
    """Store a covariant causal matrix self-energy table."""

    values_ev: Complex128[Array, "n_temperature n_k n_omega n_orb n_orb"]
    k_points_frac: Float64[Array, "n_k 3"]
    omega_rel_fermi_ev: Float64[Array, " n_omega"]
    temperature_k: Float64[Array, " n_temperature"]
    basis_ref: str = eqx.field(static=True)
    k_frame_id: str = eqx.field(static=True)
    interpolation: str = eqx.field(static=True)
    extrapolation: str = eqx.field(static=True)
    source_ref: str = eqx.field(static=True)
    provenance_ref: str = eqx.field(static=True)
    source_sha256: str = eqx.field(static=True)
    derivative_mode: str = eqx.field(static=True)
    validation: RetardedValidationReport = eqx.field(static=True)


class TabulatedRetardedGreenFunctionSource(eqx.Module):
    """Store a validated direct retarded Green-function table."""

    values_per_ev: Complex128[Array, "n_temperature n_k n_omega n_orb n_orb"]
    overlap: Complex128[Array, "n_k n_orb n_orb"]
    k_points_frac: Float64[Array, "n_k 3"]
    omega_rel_fermi_ev: Float64[Array, " n_omega"]
    temperature_k: Float64[Array, " n_temperature"]
    basis_ref: str = eqx.field(static=True)
    k_frame_id: str = eqx.field(static=True)
    interpolation: str = eqx.field(static=True)
    extrapolation: str = eqx.field(static=True)
    source_ref: str = eqx.field(static=True)
    provenance_ref: str = eqx.field(static=True)
    source_sha256: str = eqx.field(static=True)
    derivative_mode: str = eqx.field(static=True)
    validation: RetardedValidationReport = eqx.field(static=True)
    required_capabilities: Tuple[str, ...] = eqx.field(static=True)
    state_ref: str = eqx.field(static=True)

    @property
    def capabilities(self) -> Tuple[str, ...]:
        """Return the electronic-state capabilities served by this table."""
        return self.required_capabilities

    @jaxtyped(typechecker=beartype)
    def retarded_green_function(
        self,
        coordinates: MeasurementCoordinates,
    ) -> Complex128[Array, "n_k n_omega n_orb n_orb"]:
        """Expose a singleton-temperature table through the capability seam."""
        del coordinates
        if self.values_per_ev.shape[0] != 1:
            raise ValueError(
                "direct table capability requires an explicit temperature"
                " selection"
            )
        return self.values_per_ev[0]


RetardedSelfEnergySource = Union[
    ParametricSelfEnergy, TabulatedMatrixSelfEnergy
]


class DysonSpectralSource(eqx.Module):
    """Bind a typed self-energy source to metric-aware Dyson evaluation."""

    self_energy: RetardedSelfEnergySource
    source_ref: str = eqx.field(static=True)
    derivative_mode: str = eqx.field(static=True)
    required_capabilities: Tuple[str, ...] = eqx.field(static=True)


def _validate_table_axes(
    values: Complex128[Array, "n_temperature n_k n_omega n_orb n_orb"],
    k_points: Float64[Array, "n_k 3"],
    omega: Float64[Array, " n_omega"],
    temperature: Float64[Array, " n_temperature"],
) -> None:
    """PRIVATE: Validate common table dimensions before construction."""
    if (
        values.ndim != 5  # noqa: PLR2004
        or values.shape[-1] != values.shape[-2]
        or values.shape[:3]
        != (temperature.shape[0], k_points.shape[0], omega.shape[0])
        or k_points.shape[1:] != (3,)
        or omega.ndim != 1
        or temperature.ndim != 1
    ):
        raise ValueError(
            "spectral table axes and square matrix axes must agree"
        )


def _eager_matrix_validation(
    values: Complex128[Array, "n_temperature n_k n_omega n_orb n_orb"],
    *,
    matrix_kind: str,
) -> RetardedValidationReport:
    """PRIVATE: Validate matrix symmetry and semidefinite physics eagerly.

    Parameters
    ----------
    values : Complex128[Array, "n_temperature n_k n_omega n_orb n_orb"]
        Matrix-valued table evaluated on temperature, momentum, and energy
        nodes.
    matrix_kind : str
        ``"self_energy"`` for causal loss validation or ``"green"`` for
        spectral positive-semidefiniteness validation.

    Returns
    -------
    RetardedValidationReport
        Measured residual and minimum-eigenvalue diagnostics.

    Raises
    ------
    ValueError
        If a table is nonfinite, non-Hermitian in the required derived
        matrix, or violates its semidefinite condition.
    """
    table = np.asarray(values)
    if not np.all(np.isfinite(table)):
        raise ValueError(
            "finite_table check failed: table contains nonfinite values"
        )
    dagger = np.swapaxes(np.conj(table), -1, -2)
    difference = table - dagger
    residual_numerator = np.linalg.norm(difference, axis=(-2, -1))
    residual_denominator = np.maximum(
        np.linalg.norm(table, axis=(-2, -1)),
        np.finfo(np.float64).eps,
    )
    hermitian_residual = float(
        np.max(residual_numerator / residual_denominator)
    )
    if matrix_kind == "self_energy":
        derived = -(table - dagger) / (2.0j)
        check_id = "causal_loss"
        unit = "eV"
    elif matrix_kind == "green":
        derived = -(table - dagger) / (2.0j * np.pi)
        check_id = "spectral_psd"
        unit = "1/eV"
    else:
        raise ValueError("unknown matrix validation kind")
    derived_dagger = np.swapaxes(np.conj(derived), -1, -2)
    derived_residual = float(
        np.max(
            np.linalg.norm(derived - derived_dagger, axis=(-2, -1))
            / np.maximum(
                np.linalg.norm(derived, axis=(-2, -1)),
                np.finfo(np.float64).eps,
            )
        )
    )
    minimum_eigenvalue = float(np.min(np.linalg.eigvalsh(derived)))
    scale = max(float(np.max(np.linalg.norm(derived, axis=(-2, -1)))), 1.0)
    semidefinite_tolerance = HERMITICITY_RELATIVE_TOLERANCE * scale
    if derived_residual > HERMITICITY_RELATIVE_TOLERANCE:
        raise ValueError(
            f"{check_id} Hermiticity check failed: residual={derived_residual}"
        )
    if minimum_eigenvalue < -semidefinite_tolerance:
        raise ValueError(
            f"{check_id} check failed: minimum_eigenvalue={minimum_eigenvalue}"
        )
    return RetardedValidationReport(
        report_ref="pending",
        check_ids=("finite_table", "matrix_hermiticity", check_id),
        metric_values=(0.0, hermitian_residual, minimum_eigenvalue),
        tolerance_values=(
            0.0,
            HERMITICITY_RELATIVE_TOLERANCE,
            semidefinite_tolerance,
        ),
        metric_units=("1", "1", unit),
        assumptions=("eager_node_validation",),
        excluded_claims=("analyticity_proven",),
        evidence_refs=(),
        schema_version="1.0",
    )


@jaxtyped(typechecker=beartype)
def make_retarded_validation_report(
    *,
    report_ref: str,
    check_ids: Tuple[str, ...] = (),
    metric_values: Tuple[float, ...] = (),
    tolerance_values: Tuple[float, ...] = (),
    metric_units: Tuple[str, ...] = (),
    assumptions: Tuple[str, ...] = (),
    excluded_claims: Tuple[str, ...] = (),
    evidence_refs: Tuple[str, ...] = (),
    schema_version: str = "1.0",
) -> RetardedValidationReport:
    """Create a typed validation report; acceptance is never a Boolean."""
    return RetardedValidationReport(
        report_ref,
        check_ids,
        metric_values,
        tolerance_values,
        metric_units,
        assumptions,
        excluded_claims,
        evidence_refs,
        schema_version,
    )


@jaxtyped(typechecker=beartype)
def make_spectral_evaluation_request(
    coordinates: MeasurementCoordinates,
    omega_rel_fermi_ev: Float64[Array, " n_omega"],
    temperature_k: Float64[Array, " n_temperature"],
    eta_ev: Float64[Array, ""],
    *,
    basis_ref: str,
) -> SpectralEvaluationRequest:
    """Create a f64 retarded spectral evaluation request."""
    return SpectralEvaluationRequest(
        coordinates,
        jnp.asarray(omega_rel_fermi_ev, dtype=jnp.float64),
        jnp.asarray(temperature_k, dtype=jnp.float64),
        jnp.asarray(eta_ev, dtype=jnp.float64),
        basis_ref,
    )


@jaxtyped(typechecker=beartype)
def make_parametric_self_energy(
    parameterization: SelfEnergyModel,
    *,
    source_ref: str,
    basis_ref: str,
    provenance_ref: str,
) -> ParametricSelfEnergy:
    """Lift a scalar self-energy parameterization into a spectral source."""
    if not source_ref or not basis_ref or not provenance_ref:
        raise ValueError(
            "parametric self-energy identity fields must be nonempty"
        )
    return ParametricSelfEnergy(
        parameterization, source_ref, basis_ref, provenance_ref
    )


@jaxtyped(typechecker=beartype)
def make_self_energy_batch(
    values_ev: Complex128[Array, "n_temperature n_k n_omega n_orb n_orb"],
    request: SpectralEvaluationRequest,
    *,
    basis_ref: str,
    source_ref: str,
    derivative_mode: str,
) -> SelfEnergyBatch:
    """Create a self-energy batch after shape validation."""
    values: Complex128[Array, "n_temperature n_k n_omega n_orb n_orb"] = (
        jnp.asarray(values_ev, dtype=jnp.complex128)
    )
    if (
        values.ndim != 5  # noqa: PLR2004
        or values.shape[-1] != values.shape[-2]
        or values.shape[0] != request.temperature_k.shape[0]
        or values.shape[2] != request.omega_rel_fermi_ev.shape[0]
    ):
        raise ValueError("self-energy batch axes do not match its request")
    if basis_ref != request.basis_ref:
        raise ValueError("self-energy and request basis identities must match")
    return SelfEnergyBatch(
        values, request, basis_ref, source_ref, derivative_mode
    )


@jaxtyped(typechecker=beartype)
def make_retarded_green_batch(
    values_per_ev: Complex128[Array, "n_temperature n_k n_omega n_orb n_orb"],
    overlap: Complex128[Array, "n_k n_orb n_orb"],
    request: SpectralEvaluationRequest,
    *,
    basis_ref: str,
    source_ref: str,
    derivative_mode: str,
    validation_ref: str,
) -> RetardedGreenBatch:
    """Create a Green batch with a matching overlap metric."""
    values: Complex128[Array, "n_temperature n_k n_omega n_orb n_orb"] = (
        jnp.asarray(values_per_ev, dtype=jnp.complex128)
    )
    metric: Complex128[Array, "n_k n_orb n_orb"] = jnp.asarray(
        overlap, dtype=jnp.complex128
    )
    if (
        values.ndim != 5  # noqa: PLR2004
        or metric.ndim != 3  # noqa: PLR2004
        or values.shape[1] != metric.shape[0]
        or values.shape[-2:] != metric.shape[-2:]
        or values.shape[0] != request.temperature_k.shape[0]
        or values.shape[2] != request.omega_rel_fermi_ev.shape[0]
    ):
        raise ValueError("Green batch axes do not match request and overlap")
    if basis_ref != request.basis_ref:
        raise ValueError("Green batch and request basis identities must match")
    return RetardedGreenBatch(
        values,
        metric,
        request,
        basis_ref,
        source_ref,
        derivative_mode,
        validation_ref,
    )


@jaxtyped(typechecker=beartype)
def make_dyson_spectral_source(
    self_energy: RetardedSelfEnergySource, *, source_ref: str
) -> DysonSpectralSource:
    """Create a Dyson source with registry-derived state requirements."""
    required: Tuple[str, ...] = (
        ("hamiltonian", "overlap")
        if self_energy.basis_ref != "scalar"
        else ("hamiltonian",)
    )
    return DysonSpectralSource(
        self_energy, source_ref, self_energy.derivative_mode, required
    )


@jaxtyped(typechecker=beartype)  # noqa: PLR0913
def make_tabulated_matrix_self_energy(  # noqa: PLR0913
    values_ev: Complex128[Array, "n_temperature n_k n_omega n_orb n_orb"],
    k_points_frac: Float64[Array, "n_k 3"],
    omega_rel_fermi_ev: Float64[Array, " n_omega"],
    temperature_k: Float64[Array, " n_temperature"],
    *,
    basis_ref: str,
    k_frame_id: str,
    interpolation: str,
    source_ref: str,
    provenance_ref: str,
    source_sha256: str,
    derivative_mode: str,
    validation_policy_ref: str,
) -> TabulatedMatrixSelfEnergy:
    """Create a validated causal table carrier with a derived report
    identity.
    """
    values = jnp.asarray(values_ev, dtype=jnp.complex128)
    kpoints = jnp.asarray(k_points_frac, dtype=jnp.float64)
    omega = jnp.asarray(omega_rel_fermi_ev, dtype=jnp.float64)
    temperature = jnp.asarray(temperature_k, dtype=jnp.float64)
    _validate_table_axes(values, kpoints, omega, temperature)
    if interpolation not in ("multilinear_v1", "barycentric_mesh_v1"):
        raise ValueError("unregistered spectral table interpolation")
    if not all(
        (
            basis_ref,
            k_frame_id,
            source_ref,
            provenance_ref,
            source_sha256,
            validation_policy_ref,
        )
    ):
        raise ValueError(
            "tabulated self-energy identity fields must be nonempty"
        )
    preliminary_report = _eager_matrix_validation(
        values,
        matrix_kind="self_energy",
    )
    report = RetardedValidationReport(
        validation_policy_ref,
        preliminary_report.check_ids,
        preliminary_report.metric_values,
        preliminary_report.tolerance_values,
        preliminary_report.metric_units,
        preliminary_report.assumptions,
        preliminary_report.excluded_claims,
        preliminary_report.evidence_refs,
        preliminary_report.schema_version,
    )
    return TabulatedMatrixSelfEnergy(
        values,
        kpoints,
        omega,
        temperature,
        basis_ref,
        k_frame_id,
        interpolation,
        "reject",
        source_ref,
        provenance_ref,
        source_sha256,
        derivative_mode,
        report,
    )


@jaxtyped(typechecker=beartype)  # noqa: PLR0913
def make_tabulated_retarded_green_function_source(  # noqa: PLR0913
    values_per_ev: Complex128[Array, "n_temperature n_k n_omega n_orb n_orb"],
    overlap: Complex128[Array, "n_k n_orb n_orb"],
    k_points_frac: Float64[Array, "n_k 3"],
    omega_rel_fermi_ev: Float64[Array, " n_omega"],
    temperature_k: Float64[Array, " n_temperature"],
    *,
    basis_ref: str,
    k_frame_id: str,
    interpolation: str,
    source_ref: str,
    provenance_ref: str,
    source_sha256: str,
    derivative_mode: str,
    validation_policy_ref: str,
) -> TabulatedRetardedGreenFunctionSource:
    """Create a direct Green-function source with a capability identity."""
    values = jnp.asarray(values_per_ev, dtype=jnp.complex128)
    metric = jnp.asarray(overlap, dtype=jnp.complex128)
    kpoints = jnp.asarray(k_points_frac, dtype=jnp.float64)
    omega = jnp.asarray(omega_rel_fermi_ev, dtype=jnp.float64)
    temperature = jnp.asarray(temperature_k, dtype=jnp.float64)
    _validate_table_axes(values, kpoints, omega, temperature)
    if metric.shape != (kpoints.shape[0], values.shape[-1], values.shape[-1]):
        raise ValueError("direct Green overlap axes must match table orbitals")
    preliminary_report = _eager_matrix_validation(
        values,
        matrix_kind="green",
    )
    report = RetardedValidationReport(
        validation_policy_ref,
        preliminary_report.check_ids,
        preliminary_report.metric_values,
        preliminary_report.tolerance_values,
        preliminary_report.metric_units,
        preliminary_report.assumptions,
        preliminary_report.excluded_claims,
        preliminary_report.evidence_refs,
        preliminary_report.schema_version,
    )
    return TabulatedRetardedGreenFunctionSource(
        values,
        metric,
        kpoints,
        omega,
        temperature,
        basis_ref,
        k_frame_id,
        interpolation,
        "reject",
        source_ref,
        provenance_ref,
        source_sha256,
        derivative_mode,
        report,
        ("retarded_green_function", "overlap"),
        f"org.diffpes.electronic_state.{source_ref}",
    )


__all__: list[str] = [
    "DysonSpectralSource",
    "ParametricSelfEnergy",
    "RetardedGreenBatch",
    "RetardedSelfEnergySource",
    "RetardedValidationReport",
    "SelfEnergyBatch",
    "SpectralEvaluationRequest",
    "TabulatedMatrixSelfEnergy",
    "TabulatedRetardedGreenFunctionSource",
    "make_dyson_spectral_source",
    "make_parametric_self_energy",
    "make_retarded_green_batch",
    "make_retarded_validation_report",
    "make_self_energy_batch",
    "make_spectral_evaluation_request",
    "make_tabulated_matrix_self_energy",
    "make_tabulated_retarded_green_function_source",
]
