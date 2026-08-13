"""Define typed sources and evaluated batches for generalized spectra.

Extended Summary
----------------
Use this module for its validated public contracts and operations.

Routine Listings
----------------
:class:`DysonSpectralSource`
    Define the ``DysonSpectralSource`` public contract.
:class:`ParametricSelfEnergy`
    Define the ``ParametricSelfEnergy`` public contract.
:class:`RetardedGreenBatch`
    Define the ``RetardedGreenBatch`` public contract.
:class:`SelfEnergyBatch`
    Define the ``SelfEnergyBatch`` public contract.
:class:`SpectralEvaluationRequest`
    Define the ``SpectralEvaluationRequest`` public contract.
:class:`TabulatedMatrixSelfEnergy`
    Define the ``TabulatedMatrixSelfEnergy`` public contract.
:class:`TabulatedRetardedGreenFunctionSource`
    Define the ``TabulatedRetardedGreenFunctionSource`` public contract.
:func:`make_dyson_spectral_source`
    Compute the ``make_dyson_spectral_source`` public contract.
:func:`make_parametric_self_energy`
    Compute the ``make_parametric_self_energy`` public contract.
:func:`make_retarded_green_batch`
    Compute the ``make_retarded_green_batch`` public contract.
:func:`make_self_energy_batch`
    Compute the ``make_self_energy_batch`` public contract.
:func:`make_spectral_evaluation_request`
    Compute the ``make_spectral_evaluation_request`` public contract.
:func:`make_tabulated_matrix_self_energy`
    Compute the ``make_tabulated_matrix_self_energy`` public contract.
:func:`make_tabulated_retarded_green_function_source`
    Create a tabulated retarded Green-function source.
"""

# ruff: noqa: E501
import equinox as eqx
import jax.numpy as jnp
from beartype import beartype
from beartype.typing import Tuple, Union
from jaxtyping import Array, Bool, Complex128, Float64, jaxtyped

from .coordinates import MeasurementCoordinates
from .retarded_validation import (
    RetardedValidationReport,
    _eager_matrix_validation,
    _eager_overlap_validation,
    _validate_table_axes,
)
from .self_energy import SelfEnergyModel


class SpectralEvaluationRequest(eqx.Module):
    """Define the ``SpectralEvaluationRequest`` public contract.

    Validate documented inputs and preserve the declared scientific identity.

    :see: :class:`~.test_generalized_spectral.TestSpectralevaluationrequest`

    Attributes
    ----------
    coordinates : MeasurementCoordinates
        Store measurement coordinates.
    omega_rel_fermi_ev : Float64[Array, " n_omega"]
        Store relative energies.
    temperature_k : Float64[Array, " n_temperature"]
        Store temperatures.
    eta_ev : Float64[Array, ""]
        Store the retarded regulator.
    basis_ref : str
        Store the basis identity.

    See Also
    --------
    make_spectral_evaluation_request
        Construct a validated request.
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
    """Define the ``SelfEnergyBatch`` public contract.

    Validate documented inputs and preserve the declared scientific identity.

    :see: :class:`~.test_generalized_spectral.TestSelfenergybatch`

    Attributes
    ----------
    values_ev : Complex128[Array, "n_temperature n_k n_omega n_orb n_orb"]
        Store self-energy matrices.
    request : SpectralEvaluationRequest
        Store the evaluation request.
    basis_ref : str
        Store the basis identity.
    source_ref : str
        Store the source identity.
    derivative_mode : str
        Store the derivative mode.

    See Also
    --------
    make_self_energy_batch
        Construct a validated self-energy batch.
    """

    values_ev: Complex128[Array, "n_temperature n_k n_omega n_orb n_orb"]
    request: SpectralEvaluationRequest
    basis_ref: str = eqx.field(static=True)
    source_ref: str = eqx.field(static=True)
    derivative_mode: str = eqx.field(static=True)


class RetardedGreenBatch(eqx.Module):
    """Define the ``RetardedGreenBatch`` public contract.

    Validate documented inputs and preserve the declared scientific identity.

    :see: :class:`~.test_generalized_spectral.TestRetardedgreenbatch`

    Attributes
    ----------
    values_per_ev : Complex128[Array, "n_temperature n_k n_omega n_orb n_orb"]
        Store Green-function matrices.
    overlap : Complex128[Array, "n_k n_orb n_orb"]
        Store overlap matrices.
    request : SpectralEvaluationRequest
        Store the evaluation request.
    basis_ref : str
        Store the basis identity.
    source_ref : str
        Store the source identity.
    derivative_mode : str
        Store the derivative mode.
    validation_ref : str
        Store the validation identity.

    See Also
    --------
    make_retarded_green_batch
        Construct a validated Green-function batch.
    """

    values_per_ev: Complex128[Array, "n_temperature n_k n_omega n_orb n_orb"]
    overlap: Complex128[Array, "n_k n_orb n_orb"]
    request: SpectralEvaluationRequest
    basis_ref: str = eqx.field(static=True)
    source_ref: str = eqx.field(static=True)
    derivative_mode: str = eqx.field(static=True)
    validation_ref: str = eqx.field(static=True)


class ParametricSelfEnergy(eqx.Module):
    """Define the ``ParametricSelfEnergy`` public contract.

    Validate documented inputs and preserve the declared scientific identity.

    :see: :class:`~.test_generalized_spectral.TestParametricselfenergy`

    Attributes
    ----------
    parameterization : SelfEnergyModel
        Store the parameterization.
    source_ref : str
        Store the source identity.
    basis_ref : str
        Store the basis identity.
    provenance_ref : str
        Store the provenance identity.
    derivative_mode : str
        Store the derivative mode.

    See Also
    --------
    make_parametric_self_energy
        Construct a validated parametric source.
    """

    parameterization: SelfEnergyModel
    source_ref: str = eqx.field(static=True)
    basis_ref: str = eqx.field(static=True)
    provenance_ref: str = eqx.field(static=True)
    derivative_mode: str = eqx.field(static=True, default="exact_ad")


class TabulatedMatrixSelfEnergy(eqx.Module):
    """Define the ``TabulatedMatrixSelfEnergy`` public contract.

    Validate documented inputs and preserve the declared scientific identity.

    :see: :class:`~.test_generalized_spectral.TestTabulatedmatrixselfenergy`

    Attributes
    ----------
    values_ev : Complex128[Array, "n_temperature n_k n_omega n_orb n_orb"]
        Store self-energy matrices.
    k_points_frac : Float64[Array, "n_k 3"]
        Store fractional momenta.
    omega_rel_fermi_ev : Float64[Array, " n_omega"]
        Store relative energies.
    temperature_k : Float64[Array, " n_temperature"]
        Store temperatures.
    basis_ref : str
        Store the basis identity.
    k_frame_id : str
        Store the momentum-frame identity.
    interpolation : str
        Store the interpolation policy.
    extrapolation : str
        Store the extrapolation policy.
    source_ref : str
        Store the source identity.
    provenance_ref : str
        Store the provenance identity.
    source_sha256 : str
        Store the source digest.
    derivative_mode : str
        Store the derivative mode.
    validation : RetardedValidationReport
        Store validation evidence.

    See Also
    --------
    make_tabulated_matrix_self_energy
        Construct a validated tabulated self-energy.
    """

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
    """Define the ``TabulatedRetardedGreenFunctionSource`` public contract.

    Validate documented inputs and preserve the declared scientific identity.

    :see: :class:`~.test_generalized_spectral.TestTabulatedretardedgreenfunctionsource`

    Attributes
    ----------
    values_per_ev : Complex128[Array, "n_temperature n_k n_omega n_orb n_orb"]
        Store Green-function matrices.
    overlap : Complex128[Array, "n_k n_orb n_orb"]
        Store overlap matrices.
    k_points_frac : Float64[Array, "n_k 3"]
        Store fractional momenta.
    omega_rel_fermi_ev : Float64[Array, " n_omega"]
        Store relative energies.
    temperature_k : Float64[Array, " n_temperature"]
        Store temperatures.
    basis_ref : str
        Store the basis identity.
    k_frame_id : str
        Store the momentum-frame identity.
    interpolation : str
        Store the interpolation policy.
    extrapolation : str
        Store the extrapolation policy.
    source_ref : str
        Store the source identity.
    provenance_ref : str
        Store the provenance identity.
    source_sha256 : str
        Store the source digest.
    derivative_mode : str
        Store the derivative mode.
    validation : RetardedValidationReport
        Store validation evidence.
    required_capabilities : Tuple[str, ...]
        Store required capabilities.
    state_ref : str
        Store the state identity.

    See Also
    --------
    make_tabulated_retarded_green_function_source
        Construct a validated Green-function source.
    """

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
        result: Tuple[str, ...] = self.required_capabilities
        return result

    @jaxtyped(typechecker=beartype)
    def retarded_green_function(
        self,
        coordinates: MeasurementCoordinates,
    ) -> Complex128[Array, "n_k n_omega n_orb n_orb"]:
        """Return an exact-node singleton-temperature Green-function table."""
        required_names: Tuple[str, ...] = (
            "k_points_frac",
            "omega_rel_fermi_ev",
            "temperature_k",
        )
        missing_names: Tuple[str, ...] = tuple(
            name
            for name in required_names
            if name not in coordinates.coordinate_names
        )
        if missing_names:
            missing_text: str = ", ".join(missing_names)
            raise ValueError(
                f"direct table coordinates lack exact axes: {missing_text}"
            )
        if self.values_per_ev.shape[0] != 1:
            raise ValueError(
                "direct table capability requires an explicit temperature"
                " selection"
            )
        k_points: Float64[Array, "n_k 3"] = coordinates.coordinate_arrays[
            coordinates.coordinate_names.index("k_points_frac")
        ]
        omega: Float64[Array, " n_omega"] = coordinates.coordinate_arrays[
            coordinates.coordinate_names.index("omega_rel_fermi_ev")
        ]
        temperature: Float64[Array, " n_temperature"] = (
            coordinates.coordinate_arrays[
                coordinates.coordinate_names.index("temperature_k")
            ]
        )
        exact_nodes: Bool[Array, ""] = (
            jnp.array_equal(k_points, self.k_points_frac)
            & jnp.array_equal(omega, self.omega_rel_fermi_ev)
            & jnp.array_equal(temperature, self.temperature_k)
        )
        values: Complex128[Array, "n_temperature n_k n_omega n_orb n_orb"] = (
            eqx.error_if(
                self.values_per_ev,
                ~exact_nodes,
                "direct Green source requires exact-node evaluation",
            )
        )
        selected_values: Complex128[Array, "n_k n_omega n_orb n_orb"] = values[
            0
        ]
        return selected_values


class DysonSpectralSource(eqx.Module):
    """Define the ``DysonSpectralSource`` public contract.

    Validate documented inputs and preserve the declared scientific identity.

    :see: :class:`~.test_generalized_spectral.TestDysonspectralsource`

    Attributes
    ----------
    self_energy : Union[ParametricSelfEnergy, TabulatedMatrixSelfEnergy]
        Store the self-energy source.
    source_ref : str
        Store the source identity.
    derivative_mode : str
        Store the derivative mode.
    required_capabilities : Tuple[str, ...]
        Store required capabilities.

    See Also
    --------
    make_dyson_spectral_source
        Construct a validated Dyson source.
    """

    self_energy: Union[ParametricSelfEnergy, TabulatedMatrixSelfEnergy]
    source_ref: str = eqx.field(static=True)
    derivative_mode: str = eqx.field(static=True)
    required_capabilities: Tuple[str, ...] = eqx.field(static=True)


@jaxtyped(typechecker=beartype)
def make_spectral_evaluation_request(
    coordinates: MeasurementCoordinates,
    omega_rel_fermi_ev: Float64[Array, " n_omega"],
    temperature_k: Float64[Array, " n_temperature"],
    eta_ev: Float64[Array, ""],
    *,
    basis_ref: str,
) -> SpectralEvaluationRequest:
    """Compute the ``make_spectral_evaluation_request`` public contract.

    Validate documented inputs and preserve the declared scientific identity.

    :see: :class:`~.test_generalized_spectral.TestMakeSpectralEvaluationRequest`

    Notes
    -----
    Validate inputs before returning the named result.

    Parameters
    ----------
    coordinates : MeasurementCoordinates
        Input value for this operation.
    omega_rel_fermi_ev : Float64[Array, ' n_omega']
        Input value for this operation.
    temperature_k : Float64[Array, ' n_temperature']
        Input value for this operation.
    eta_ev : Float64[Array, '']
        Input value for this operation.
    basis_ref : str
        Input value for this operation.

    Returns
    -------
    result : SpectralEvaluationRequest
        Validated operation result.
    """
    result: SpectralEvaluationRequest = SpectralEvaluationRequest(
        coordinates,
        jnp.asarray(omega_rel_fermi_ev, dtype=jnp.float64),
        jnp.asarray(temperature_k, dtype=jnp.float64),
        jnp.asarray(eta_ev, dtype=jnp.float64),
        basis_ref,
    )
    return result


@jaxtyped(typechecker=beartype)
def make_parametric_self_energy(
    parameterization: SelfEnergyModel,
    *,
    source_ref: str,
    basis_ref: str,
    provenance_ref: str,
) -> ParametricSelfEnergy:
    """Compute the ``make_parametric_self_energy`` public contract.

    Validate documented inputs and preserve the declared scientific identity.

    :see: :class:`~.test_generalized_spectral.TestMakeParametricSelfEnergy`

    Notes
    -----
    Validate inputs before returning the named result.

    Parameters
    ----------
    parameterization : SelfEnergyModel
        Input value for this operation.
    source_ref : str
        Input value for this operation.
    basis_ref : str
        Input value for this operation.
    provenance_ref : str
        Input value for this operation.

    Returns
    -------
    result : ParametricSelfEnergy
        Validated operation result.

    Raises
    ------
    ValueError
        If a source, basis, or provenance identity is empty.
    """
    if not source_ref or not basis_ref or not provenance_ref:
        raise ValueError(
            "parametric self-energy identity fields must be nonempty"
        )
    source: ParametricSelfEnergy = ParametricSelfEnergy(
        parameterization, source_ref, basis_ref, provenance_ref
    )
    return source


@jaxtyped(typechecker=beartype)
def make_self_energy_batch(
    values_ev: Complex128[Array, "n_temperature n_k n_omega n_orb n_orb"],
    request: SpectralEvaluationRequest,
    *,
    basis_ref: str,
    source_ref: str,
    derivative_mode: str,
) -> SelfEnergyBatch:
    """Compute the ``make_self_energy_batch`` public contract.

    Validate documented inputs and preserve the declared scientific identity.

    :see: :class:`~.test_generalized_spectral.TestMakeSelfEnergyBatch`

    Notes
    -----
    Validate inputs before returning the named result.

    Parameters
    ----------
    values_ev : Complex128[Array, 'n_temperature n_k n_omega n_orb n_orb']
        Input value for this operation.
    request : SpectralEvaluationRequest
        Input value for this operation.
    basis_ref : str
        Input value for this operation.
    source_ref : str
        Input value for this operation.
    derivative_mode : str
        Input value for this operation.

    Returns
    -------
    result : SelfEnergyBatch
        Validated operation result.

    Raises
    ------
    ValueError
        If batch axes, basis identity, or source identity are inconsistent.
    """
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
    if not source_ref or not derivative_mode:
        raise ValueError("self-energy batch identity fields must be nonempty")
    batch: SelfEnergyBatch = SelfEnergyBatch(
        values, request, basis_ref, source_ref, derivative_mode
    )
    return batch


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
    """Compute the ``make_retarded_green_batch`` public contract.

    Validate documented inputs and preserve the declared scientific identity.

    :see: :class:`~.test_generalized_spectral.TestMakeRetardedGreenBatch`

    Notes
    -----
    Validate inputs before returning the named result.

    Parameters
    ----------
    values_per_ev : Complex128[Array, 'n_temperature n_k n_omega n_orb n_orb']
        Input value for this operation.
    overlap : Complex128[Array, 'n_k n_orb n_orb']
        Input value for this operation.
    request : SpectralEvaluationRequest
        Input value for this operation.
    basis_ref : str
        Input value for this operation.
    source_ref : str
        Input value for this operation.
    derivative_mode : str
        Input value for this operation.
    validation_ref : str
        Input value for this operation.

    Returns
    -------
    result : RetardedGreenBatch
        Validated operation result.

    Raises
    ------
    ValueError
        If Green and overlap axes or their identities are inconsistent.
    """
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
    if not source_ref or not derivative_mode or not validation_ref:
        raise ValueError("Green batch identity fields must be nonempty")
    batch: RetardedGreenBatch = RetardedGreenBatch(
        values,
        metric,
        request,
        basis_ref,
        source_ref,
        derivative_mode,
        validation_ref,
    )
    return batch


@jaxtyped(typechecker=beartype)
def make_dyson_spectral_source(
    self_energy: Union[ParametricSelfEnergy, TabulatedMatrixSelfEnergy],
    *,
    electronic_state_capabilities: Tuple[str, ...],
    source_ref: str,
) -> DysonSpectralSource:
    """Compute the ``make_dyson_spectral_source`` public contract.

    Validate documented inputs and preserve the declared scientific identity.

    :see: :class:`~.test_generalized_spectral.TestMakeDysonSpectralSource`

    Notes
    -----
    Validate inputs before returning the named result.

    Parameters
    ----------
    self_energy : Union[ParametricSelfEnergy, TabulatedMatrixSelfEnergy]
        Input value for this operation.
    electronic_state_capabilities : Tuple[str, ...]
        Input value for this operation.
    source_ref : str
        Input value for this operation.

    Returns
    -------
    result : DysonSpectralSource
        Validated operation result.

    Raises
    ------
    ValueError
        If the electronic-state capabilities or source identity are empty.
    """
    if not electronic_state_capabilities:
        raise ValueError("electronic-state capabilities must be nonempty")
    if not source_ref:
        raise ValueError("Dyson spectral source_ref must be nonempty")
    required: Tuple[str, ...] = (
        ("hamiltonian",)
        if "orthonormal_basis" in electronic_state_capabilities
        else ("hamiltonian", "overlap")
    )
    result: DysonSpectralSource = DysonSpectralSource(
        self_energy, source_ref, self_energy.derivative_mode, required
    )
    return result


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
    """Compute the ``make_tabulated_matrix_self_energy`` public contract.

    Validate documented inputs and preserve the declared scientific identity.

    :see: :class:`~.test_generalized_spectral.TestMakeTabulatedMatrixSelfEnergy`

    Notes
    -----
    Validate inputs before returning the named result.

    Parameters
    ----------
    values_ev : Complex128[Array, 'n_temperature n_k n_omega n_orb n_orb']
        Input value for this operation.
    k_points_frac : Float64[Array, 'n_k 3']
        Input value for this operation.
    omega_rel_fermi_ev : Float64[Array, ' n_omega']
        Input value for this operation.
    temperature_k : Float64[Array, ' n_temperature']
        Input value for this operation.
    basis_ref : str
        Input value for this operation.
    k_frame_id : str
        Input value for this operation.
    interpolation : str
        Input value for this operation.
    source_ref : str
        Input value for this operation.
    provenance_ref : str
        Input value for this operation.
    source_sha256 : str
        Input value for this operation.
    derivative_mode : str
        Input value for this operation.
    validation_policy_ref : str
        Input value for this operation.

    Returns
    -------
    result : TabulatedMatrixSelfEnergy
        Validated operation result.

    Raises
    ------
    ValueError
        If table axes, exact-node policy, identity, or causal loss is invalid.
    """
    values: Complex128[Array, "n_temperature n_k n_omega n_orb n_orb"] = (
        jnp.asarray(values_ev, dtype=jnp.complex128)
    )
    kpoints: Float64[Array, "n_k 3"] = jnp.asarray(
        k_points_frac, dtype=jnp.float64
    )
    omega: Float64[Array, " n_omega"] = jnp.asarray(
        omega_rel_fermi_ev, dtype=jnp.float64
    )
    temperature: Float64[Array, " n_temperature"] = jnp.asarray(
        temperature_k, dtype=jnp.float64
    )
    _validate_table_axes(values, kpoints, omega, temperature)
    if interpolation != "exact_nodes_v1":
        raise ValueError("spectral tables support exact-node evaluation only")
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
    preliminary_report: RetardedValidationReport = _eager_matrix_validation(
        values,
        matrix_kind="self_energy",
    )
    report: RetardedValidationReport = RetardedValidationReport(
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
    result: TabulatedMatrixSelfEnergy = TabulatedMatrixSelfEnergy(
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
    return result


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
    """Create a tabulated retarded Green-function source.

    Validate documented inputs and preserve the declared scientific identity.

    :see: :class:`~.test_generalized_spectral.TestMakeTabulatedRetardedGreenFunctionSource`

    Notes
    -----
    Validate inputs before returning the named result.

    Parameters
    ----------
    values_per_ev : Complex128[Array, 'n_temperature n_k n_omega n_orb n_orb']
        Input value for this operation.
    overlap : Complex128[Array, 'n_k n_orb n_orb']
        Input value for this operation.
    k_points_frac : Float64[Array, 'n_k 3']
        Input value for this operation.
    omega_rel_fermi_ev : Float64[Array, ' n_omega']
        Input value for this operation.
    temperature_k : Float64[Array, ' n_temperature']
        Input value for this operation.
    basis_ref : str
        Input value for this operation.
    k_frame_id : str
        Input value for this operation.
    interpolation : str
        Input value for this operation.
    source_ref : str
        Input value for this operation.
    provenance_ref : str
        Input value for this operation.
    source_sha256 : str
        Input value for this operation.
    derivative_mode : str
        Input value for this operation.
    validation_policy_ref : str
        Input value for this operation.

    Returns
    -------
    result : TabulatedRetardedGreenFunctionSource
        Validated operation result.

    Raises
    ------
    ValueError
        If table axes, overlap, identity, exact-node policy, or spectral
        positivity is invalid.
    """
    values: Complex128[Array, "n_temperature n_k n_omega n_orb n_orb"] = (
        jnp.asarray(values_per_ev, dtype=jnp.complex128)
    )
    metric: Complex128[Array, "n_k n_orb n_orb"] = jnp.asarray(
        overlap, dtype=jnp.complex128
    )
    kpoints: Float64[Array, "n_k 3"] = jnp.asarray(
        k_points_frac, dtype=jnp.float64
    )
    omega: Float64[Array, " n_omega"] = jnp.asarray(
        omega_rel_fermi_ev, dtype=jnp.float64
    )
    temperature: Float64[Array, " n_temperature"] = jnp.asarray(
        temperature_k, dtype=jnp.float64
    )
    _validate_table_axes(values, kpoints, omega, temperature)
    if metric.shape != (kpoints.shape[0], values.shape[-1], values.shape[-1]):
        raise ValueError("direct Green overlap axes must match table orbitals")
    _eager_overlap_validation(metric)
    if interpolation != "exact_nodes_v1":
        raise ValueError("direct Green tables support exact nodes only")
    if not all(
        (
            basis_ref,
            k_frame_id,
            source_ref,
            provenance_ref,
            source_sha256,
            derivative_mode,
            validation_policy_ref,
        )
    ):
        raise ValueError("direct Green identity fields must be nonempty")
    preliminary_report: RetardedValidationReport = _eager_matrix_validation(
        values,
        matrix_kind="green",
    )
    report: RetardedValidationReport = RetardedValidationReport(
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
    result: TabulatedRetardedGreenFunctionSource = (
        TabulatedRetardedGreenFunctionSource(
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
    )
    return result


__all__: list[str] = [
    "DysonSpectralSource",
    "ParametricSelfEnergy",
    "RetardedGreenBatch",
    "SelfEnergyBatch",
    "SpectralEvaluationRequest",
    "TabulatedMatrixSelfEnergy",
    "TabulatedRetardedGreenFunctionSource",
    "make_dyson_spectral_source",
    "make_parametric_self_energy",
    "make_retarded_green_batch",
    "make_self_energy_batch",
    "make_spectral_evaluation_request",
    "make_tabulated_matrix_self_energy",
    "make_tabulated_retarded_green_function_source",
]
