"""Evaluate metric-aware retarded Green functions and spectral projections.

Extended Summary
----------------
Use this module for its validated public contracts and operations.

Routine Listings
----------------
:func:`projected_spectral_density`
    Compute the ``projected_spectral_density`` public contract.
:func:`projected_spectral_density_solve`
    Compute the ``projected_spectral_density_solve`` public contract.
:func:`solve_retarded_dyson`
    Compute the ``solve_retarded_dyson`` public contract.
:func:`spectral_density_matrix`
    Compute the ``spectral_density_matrix`` public contract.
:func:`total_spectral_density`
    Compute the ``total_spectral_density`` public contract.
:func:`total_spectral_density_solve`
    Compute the ``total_spectral_density_solve`` public contract.
"""

# The repository floor requires unsplittable reciprocal Sphinx class links.
# ruff: noqa: E501

import equinox as eqx
import jax
import jax.numpy as jnp
from beartype import beartype
from beartype.typing import Literal, Tuple
from jaxtyping import Array, Bool, Complex128, Float64, jaxtyped

from diffpes.constants import (
    CARTESIAN_COMPONENTS,
    HERMITICITY_RELATIVE_TOLERANCE,
)
from diffpes.types import (
    HamiltonianSource,
    MeasurementCoordinates,
    ParametricSelfEnergy,
    RetardedGreenBatch,
    RetardedSelfEnergySource,
    SelfEnergyBatch,
    SpectralEvaluationRequest,
    TabulatedMatrixSelfEnergy,
    make_retarded_green_batch,
    make_self_energy_batch,
)

from .retarded_self_energy import evaluate_self_energy


def _dagger(
    value: Complex128[Array, "... n_orb n_orb"],
) -> Complex128[Array, "... n_orb n_orb"]:
    """PRIVATE: Return the conjugate transpose on matrix axes.

    Notes
    -----
    Preserve all leading batch axes.
    """
    result: Complex128[Array, "... n_orb n_orb"] = jnp.swapaxes(
        jnp.conj(value), -1, -2
    )
    return result


def _coordinate_array(
    coordinates: MeasurementCoordinates,
    name: str,
) -> Float64[Array, "..."]:
    """PRIVATE: Return one statically named coordinate array.

    Parameters
    ----------
    coordinates : MeasurementCoordinates
        Coordinate carrier to query.
    name : str
        Coordinate name to select.

    Returns
    -------
    values : Float64[Array, "..."]
        Selected coordinate array.

    Raises
    ------
    ValueError
        If the coordinates omit the requested name.
    """
    if name not in coordinates.coordinate_names:
        raise ValueError(f"measurement coordinates lack required axis: {name}")
    values: Float64[Array, "..."] = coordinates.coordinate_arrays[
        coordinates.coordinate_names.index(name)
    ]
    return values


@jaxtyped(typechecker=beartype)
def _evaluate_retarded_self_energy(
    source: RetardedSelfEnergySource,
    electronic_state: HamiltonianSource,
    request: SpectralEvaluationRequest,
) -> SelfEnergyBatch:
    """PRIVATE: Evaluate a retarded self-energy in its covariant basis.

    Notes
    -----
    Retain scalar self-energies as the singleton orbital specialization.
    Select tabulated values exactly on their declared grid.
    """
    if source.basis_ref != request.basis_ref:
        raise ValueError(
            "self-energy source and request basis identities differ"
        )
    if isinstance(source, ParametricSelfEnergy):
        scalar: Complex128[Array, " n_omega"] = evaluate_self_energy(
            request.omega_rel_fermi_ev,
            source.parameterization,
        )
        hamiltonian: Complex128[Array, "n_k n_orb n_orb"] = (
            electronic_state.hamiltonian(request.coordinates)
        )
        identity: Complex128[Array, "n_orb n_orb"] = jnp.eye(
            hamiltonian.shape[-1], dtype=jnp.complex128
        )
        values: Complex128[Array, "n_temperature n_k n_omega n_orb n_orb"] = (
            scalar[None, None, :, None, None] * identity[None, None, None]
        )
        values = jnp.broadcast_to(
            values,
            (
                request.temperature_k.shape[0],
                hamiltonian.shape[0],
                request.omega_rel_fermi_ev.shape[0],
                hamiltonian.shape[-1],
                hamiltonian.shape[-1],
            ),
        )
        result: SelfEnergyBatch = make_self_energy_batch(
            values,
            request,
            basis_ref=source.basis_ref,
            source_ref=source.source_ref,
            derivative_mode=source.derivative_mode,
        )
        return result  # noqa: RET504
    if isinstance(source, TabulatedMatrixSelfEnergy):
        request_k_points: Float64[Array, "n_k 3"] = _coordinate_array(
            request.coordinates, "k_points_frac"
        )
        axes_match: Bool[Array, ""] = (
            jnp.array_equal(
                source.omega_rel_fermi_ev,
                request.omega_rel_fermi_ev,
            )
            & jnp.array_equal(source.temperature_k, request.temperature_k)
            & jnp.array_equal(source.k_points_frac, request_k_points)
        )
        values: Complex128[Array, "n_temperature n_k n_omega n_orb n_orb"] = (
            eqx.error_if(
                source.values_ev,
                ~axes_match,
                "tabulated source requires exact-node evaluation",
            )
        )
        result = make_self_energy_batch(
            values,
            request,
            basis_ref=source.basis_ref,
            source_ref=source.source_ref,
            derivative_mode=source.derivative_mode,
        )
        return result  # noqa: RET504
    raise TypeError("unrecognized closed retarded self-energy source")


def _validate_dyson_domain(
    hamiltonian: Complex128[Array, "n_k n_orb n_orb"],
    overlap: Complex128[Array, "n_k n_orb n_orb"],
    eta_ev: Float64[Array, ""],
) -> Tuple[
    Complex128[Array, "n_k n_orb n_orb"],
    Complex128[Array, "n_k n_orb n_orb"],
]:
    """PRIVATE: Reject invalid Hamiltonian, metric, and regulator values.

    Parameters
    ----------
    hamiltonian : Complex128[Array, "n_k n_orb n_orb"]
        Hamiltonian matrices to validate.
    overlap : Complex128[Array, "n_k n_orb n_orb"]
        Overlap matrices to validate.
    eta_ev : Float64[Array, ""]
        Retarded regulator to validate.

    Returns
    -------
    hamiltonian : Complex128[Array, "n_k n_orb n_orb"]
        Validated Hamiltonian matrices.
    overlap : Complex128[Array, "n_k n_orb n_orb"]
        Validated overlap matrices.

    Notes
    -----
    Use traced Equinox checks for matrix-domain and regulator constraints.
    """
    hamiltonian_scale: Float64[Array, " n_k"] = jnp.maximum(
        jnp.linalg.norm(hamiltonian, axis=(-2, -1)),
        jnp.finfo(jnp.float64).eps,
    )
    overlap_scale: Float64[Array, " n_k"] = jnp.maximum(
        jnp.linalg.norm(overlap, axis=(-2, -1)),
        jnp.finfo(jnp.float64).eps,
    )
    hamiltonian_residual: Float64[Array, ""] = jnp.max(
        jnp.linalg.norm(hamiltonian - _dagger(hamiltonian), axis=(-2, -1))
        / hamiltonian_scale
    )
    overlap_residual: Float64[Array, ""] = jnp.max(
        jnp.linalg.norm(overlap - _dagger(overlap), axis=(-2, -1))
        / overlap_scale
    )
    checked_hamiltonian: Complex128[Array, "n_k n_orb n_orb"] = eqx.error_if(
        hamiltonian,
        ~jnp.all(jnp.isfinite(hamiltonian))
        | (hamiltonian_residual > HERMITICITY_RELATIVE_TOLERANCE),
        "Dyson Hamiltonian must be finite and Hermitian",
    )
    cholesky: Complex128[Array, "n_k n_orb n_orb"] = jnp.linalg.cholesky(
        overlap
    )
    checked_overlap: Complex128[Array, "n_k n_orb n_orb"] = eqx.error_if(
        overlap,
        ~jnp.all(jnp.isfinite(overlap))
        | (overlap_residual > HERMITICITY_RELATIVE_TOLERANCE)
        | ~jnp.all(jnp.isfinite(cholesky)),
        "Dyson overlap must be finite, Hermitian, and positive definite",
    )
    checked_hamiltonian = eqx.error_if(
        checked_hamiltonian,
        ~jnp.isfinite(eta_ev) | (eta_ev <= 0.0),
        "Dyson regulator must be finite and positive",
    )
    result: Tuple[
        Complex128[Array, "n_k n_orb n_orb"],
        Complex128[Array, "n_k n_orb n_orb"],
    ] = (checked_hamiltonian, checked_overlap)
    return result


@jaxtyped(typechecker=beartype)
def solve_retarded_dyson(
    hamiltonian_rel_fermi_ev: Complex128[Array, "n_k n_orb n_orb"],
    overlap: Complex128[Array, "n_k n_orb n_orb"],
    self_energy: SelfEnergyBatch,
    request: SpectralEvaluationRequest,
) -> RetardedGreenBatch:
    """Compute the ``solve_retarded_dyson`` public contract.

    Validate documented inputs and preserve the declared scientific identity.

    :see: :class:`~.test_generalized_spectral.TestSolveRetardedDyson`

    Notes
    -----
    Validate inputs before returning the named result.

    Parameters
    ----------
    hamiltonian_rel_fermi_ev : Complex128[Array, 'n_k n_orb n_orb']
        Input value for this operation.
    overlap : Complex128[Array, 'n_k n_orb n_orb']
        Input value for this operation.
    self_energy : SelfEnergyBatch
        Input value for this operation.
    request : SpectralEvaluationRequest
        Input value for this operation.

    Returns
    -------
    result : RetardedGreenBatch
        Validated operation result.

    Raises
    ------
    ValueError
        If matrix axes or spectral-request basis identities disagree.
    """
    hamiltonian: Complex128[Array, "n_k n_orb n_orb"] = jnp.asarray(
        hamiltonian_rel_fermi_ev, dtype=jnp.complex128
    )
    metric: Complex128[Array, "n_k n_orb n_orb"] = jnp.asarray(
        overlap, dtype=jnp.complex128
    )
    if (
        hamiltonian.shape != metric.shape
        or hamiltonian.ndim != CARTESIAN_COMPONENTS
    ):
        raise ValueError(
            "Hamiltonian and overlap must be matching square matrices"
        )
    hamiltonian, metric = _validate_dyson_domain(
        hamiltonian, metric, request.eta_ev
    )
    if self_energy.request.basis_ref != request.basis_ref:
        raise ValueError(
            "Dyson self-energy and request basis identities differ"
        )
    request_axes_match: Bool[Array, ""] = jnp.array_equal(
        self_energy.request.omega_rel_fermi_ev,
        request.omega_rel_fermi_ev,
    ) & jnp.array_equal(
        self_energy.request.temperature_k,
        request.temperature_k,
    )
    sigma_values: Complex128[
        Array, "n_temperature n_k n_omega n_orb n_orb"
    ] = eqx.error_if(
        self_energy.values_ev,
        ~request_axes_match,
        "Dyson self-energy must be evaluated for this request",
    )
    n_orb: int = hamiltonian.shape[-1]
    eye: Complex128[Array, "n_orb n_orb"] = jnp.eye(
        n_orb, dtype=jnp.complex128
    )
    frequency: Complex128[Array, " n_omega"] = (
        request.omega_rel_fermi_ev + 1.0j * request.eta_ev
    )
    operator: Complex128[Array, "n_temperature n_k n_omega n_orb n_orb"] = (
        frequency[None, None, :, None, None] * metric[None, :, None]
        - hamiltonian[None, :, None]
        - sigma_values
    )
    right_hand_side: Complex128[Array, "n_orb n_orb"] = eye
    values: Complex128[Array, "n_temperature n_k n_omega n_orb n_orb"] = (
        jax.vmap(
            jax.vmap(
                jax.vmap(jnp.linalg.solve, in_axes=(0, None)),
                in_axes=(0, None),
            ),
            in_axes=(0, None),
        )(operator, right_hand_side)
    )
    result: RetardedGreenBatch = make_retarded_green_batch(
        values,
        metric,
        request,
        basis_ref=self_energy.basis_ref,
        source_ref=self_energy.source_ref,
        derivative_mode=self_energy.derivative_mode,
        validation_ref="org.diffpes.validation.dyson@1.0.0",
    )
    return result


@jaxtyped(typechecker=beartype)
def projected_spectral_density_solve(
    hamiltonian_rel_fermi_ev: Complex128[Array, "n_k n_orb n_orb"],
    overlap: Complex128[Array, "n_k n_orb n_orb"],
    self_energy: SelfEnergyBatch,
    request: SpectralEvaluationRequest,
    transition_sources: Complex128[
        Array, "n_temperature n_k n_omega n_out n_orb"
    ],
) -> Float64[Array, "n_temperature n_k n_omega n_out"]:
    """Compute the ``projected_spectral_density_solve`` public contract.

    Validate documented inputs and preserve the declared scientific identity.

    :see: :class:`~.test_generalized_spectral.TestProjectedSpectralDensitySolve`

    Notes
    -----
    Validate inputs before returning the named result.

    Parameters
    ----------
    hamiltonian_rel_fermi_ev : Complex128[Array, 'n_k n_orb n_orb']
        Input value for this operation.
    overlap : Complex128[Array, 'n_k n_orb n_orb']
        Input value for this operation.
    self_energy : SelfEnergyBatch
        Input value for this operation.
    request : SpectralEvaluationRequest
        Input value for this operation.
    transition_sources : Complex128[Array, 'n_temperature n_k n_omega n_out n_orb']
        Input value for this operation.

    Returns
    -------
    result : Float64[Array, 'n_temperature n_k n_omega n_out']
        Validated operation result.

    Raises
    ------
    ValueError
        If transition-source or Dyson matrix axes disagree.
    """
    hamiltonian: Complex128[Array, "n_k n_orb n_orb"] = jnp.asarray(
        hamiltonian_rel_fermi_ev, dtype=jnp.complex128
    )
    metric: Complex128[Array, "n_k n_orb n_orb"] = jnp.asarray(
        overlap, dtype=jnp.complex128
    )
    if hamiltonian.shape != metric.shape:
        raise ValueError("projected solve matrices must have matching axes")
    hamiltonian, metric = _validate_dyson_domain(
        hamiltonian, metric, request.eta_ev
    )
    sources: Complex128[Array, "n_temperature n_k n_omega n_out n_orb"] = (
        jnp.asarray(transition_sources, dtype=jnp.complex128)
    )
    if (
        sources.shape[:3] != self_energy.values_ev.shape[:3]
        or sources.shape[-1] != hamiltonian.shape[-1]
    ):
        raise ValueError("transition sources must match the Dyson batch axes")
    frequency: Complex128[Array, " n_omega"] = (
        request.omega_rel_fermi_ev + 1.0j * request.eta_ev
    )
    operator: Complex128[Array, "n_temperature n_k n_omega n_orb n_orb"] = (
        frequency[None, None, :, None, None] * metric[None, :, None]
        - hamiltonian[None, :, None]
        - self_energy.values_ev
    )
    right_hand_sides: Complex128[
        Array, "n_temperature n_k n_omega n_orb n_out"
    ] = jnp.swapaxes(sources, -1, -2)
    solutions: Complex128[Array, "n_temperature n_k n_omega n_orb n_out"] = (
        jnp.linalg.solve(operator, right_hand_sides)
    )
    projected: Float64[Array, "n_temperature n_k n_omega n_out"] = (
        -jnp.imag(
            jnp.einsum(
                "tkwai,tkwia->tkwa",
                jnp.conj(sources),
                solutions,
            )
        )
        / jnp.pi
    )
    return projected


@jaxtyped(typechecker=beartype)
def total_spectral_density_solve(
    hamiltonian_rel_fermi_ev: Complex128[Array, "n_k n_orb n_orb"],
    overlap: Complex128[Array, "n_k n_orb n_orb"],
    self_energy: SelfEnergyBatch,
    request: SpectralEvaluationRequest,
) -> Float64[Array, "n_temperature n_k n_omega"]:
    """Compute the ``total_spectral_density_solve`` public contract.

    Validate documented inputs and preserve the declared scientific identity.

    :see: :class:`~.test_generalized_spectral.TestTotalSpectralDensitySolve`

    Notes
    -----
    Validate inputs before returning the named result.

    Parameters
    ----------
    hamiltonian_rel_fermi_ev : Complex128[Array, 'n_k n_orb n_orb']
        Input value for this operation.
    overlap : Complex128[Array, 'n_k n_orb n_orb']
        Input value for this operation.
    self_energy : SelfEnergyBatch
        Input value for this operation.
    request : SpectralEvaluationRequest
        Input value for this operation.

    Returns
    -------
    result : Float64[Array, 'n_temperature n_k n_omega']
        Validated operation result.

    Raises
    ------
    ValueError
        If the Hamiltonian and overlap axes disagree.
    """
    hamiltonian: Complex128[Array, "n_k n_orb n_orb"] = jnp.asarray(
        hamiltonian_rel_fermi_ev, dtype=jnp.complex128
    )
    metric: Complex128[Array, "n_k n_orb n_orb"] = jnp.asarray(
        overlap, dtype=jnp.complex128
    )
    if hamiltonian.shape != metric.shape:
        raise ValueError("total solve matrices must have matching axes")
    hamiltonian, metric = _validate_dyson_domain(
        hamiltonian, metric, request.eta_ev
    )
    if self_energy.values_ev.shape[1] != hamiltonian.shape[0]:
        raise ValueError("self-energy k axis must match the Dyson matrices")
    frequency: Complex128[Array, " n_omega"] = (
        request.omega_rel_fermi_ev + 1.0j * request.eta_ev
    )
    operator: Complex128[Array, "n_temperature n_k n_omega n_orb n_orb"] = (
        frequency[None, None, :, None, None] * metric[None, :, None]
        - hamiltonian[None, :, None]
        - self_energy.values_ev
    )
    metric_rhs: Complex128[Array, "n_temperature n_k n_omega n_orb n_orb"] = (
        jnp.broadcast_to(metric[None, :, None], operator.shape)
    )
    solutions: Complex128[Array, "n_temperature n_k n_omega n_orb n_orb"] = (
        jnp.linalg.solve(operator, metric_rhs)
    )
    total: Float64[Array, "n_temperature n_k n_omega"] = (
        -jnp.imag(jnp.trace(solutions, axis1=-2, axis2=-1)) / jnp.pi
    )
    return total


@jaxtyped(typechecker=beartype)
def spectral_density_matrix(
    green: RetardedGreenBatch,
) -> Complex128[Array, "n_temperature n_k n_omega n_orb n_orb"]:
    """Compute the ``spectral_density_matrix`` public contract.

    Validate documented inputs and preserve the declared scientific identity.

    :see: :class:`~.test_generalized_spectral.TestSpectralDensityMatrix`

    Notes
    -----
    Validate inputs before returning the named result.

    Parameters
    ----------
    green : RetardedGreenBatch
        Input value for this operation.

    Returns
    -------
    result : Complex128[Array, 'n_temperature n_k n_omega n_orb n_orb']
        Validated operation result.
    """
    spectral: Complex128[Array, "n_temperature n_k n_omega n_orb n_orb"] = -(
        green.values_per_ev - _dagger(green.values_per_ev)
    ) / (2.0j * jnp.pi)
    return spectral


@jaxtyped(typechecker=beartype)
def total_spectral_density(
    green: RetardedGreenBatch,
) -> Float64[Array, "n_temperature n_k n_omega"]:
    """Compute the ``total_spectral_density`` public contract.

    Validate documented inputs and preserve the declared scientific identity.

    :see: :class:`~.test_generalized_spectral.TestTotalSpectralDensity`

    Notes
    -----
    Validate inputs before returning the named result.

    Parameters
    ----------
    green : RetardedGreenBatch
        Input value for this operation.

    Returns
    -------
    result : Float64[Array, 'n_temperature n_k n_omega']
        Validated operation result.
    """
    spectral: Complex128[Array, "n_temperature n_k n_omega n_orb n_orb"] = (
        spectral_density_matrix(green)
    )
    total: Float64[Array, "n_temperature n_k n_omega"] = jnp.real(
        jnp.einsum("kij,tkwji->tkw", green.overlap, spectral)
    )
    return total


@jaxtyped(typechecker=beartype)
def projected_spectral_density(  # noqa: DOC105 -- Napoleon breaks quoted Literal.
    green: RetardedGreenBatch,
    transition_sources: Complex128[
        Array, "n_temperature n_k n_omega n_out n_orb"
    ],
    *,
    source_variance: Literal["covariant"] = "covariant",
) -> Float64[Array, "n_temperature n_k n_omega n_out"]:
    """Compute the ``projected_spectral_density`` public contract.

    Validate documented inputs and preserve the declared scientific identity.

    :see: :class:`~.test_generalized_spectral.TestProjectedSpectralDensity`

    Notes
    -----
    Validate inputs before returning the named result.

    Parameters
    ----------
    green : RetardedGreenBatch
        Input value for this operation.
    transition_sources : Complex128[Array, 'n_temperature n_k n_omega n_out n_orb']
        Input value for this operation.
    source_variance : Literal[covariant]
        Input value for this operation.

    Returns
    -------
    result : Float64[Array, 'n_temperature n_k n_omega n_out']
        Validated operation result.

    Raises
    ------
    ValueError
        If transition-source axes disagree with the Green batch.
    """
    if source_variance != "covariant":
        raise ValueError("transition sources must be declared covariant")
    sources: Complex128[Array, "n_temperature n_k n_omega n_out n_orb"] = (
        jnp.asarray(transition_sources, dtype=jnp.complex128)
    )
    spectral: Complex128[Array, "n_temperature n_k n_omega n_orb n_orb"] = (
        spectral_density_matrix(green)
    )
    if (
        sources.shape[:3] != spectral.shape[:3]
        or sources.shape[-1] != spectral.shape[-1]
    ):
        raise ValueError("transition sources must share the Green batch axes")
    projected: Float64[Array, "n_temperature n_k n_omega n_out"] = jnp.real(
        jnp.einsum(
            "tkwai,tkwij,tkwaj->tkwa", jnp.conj(sources), spectral, sources
        )
    )
    return projected


__all__: list[str] = [
    "projected_spectral_density",
    "projected_spectral_density_solve",
    "solve_retarded_dyson",
    "spectral_density_matrix",
    "total_spectral_density",
    "total_spectral_density_solve",
]
