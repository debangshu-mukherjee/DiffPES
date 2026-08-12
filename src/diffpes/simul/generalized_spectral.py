"""Evaluate metric-aware retarded Green functions and spectral projections."""

import equinox as eqx
import jax
import jax.numpy as jnp
from beartype import beartype
from beartype.typing import Literal
from jaxtyping import Array, Complex128, Float64, jaxtyped

from diffpes.constants import CARTESIAN_COMPONENTS
from diffpes.types import (
    HamiltonianSource,
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
    """PRIVATE: Return the conjugate transpose on matrix axes."""
    return jnp.swapaxes(jnp.conj(value), -1, -2)


@jaxtyped(typechecker=beartype)
def _evaluate_retarded_self_energy(
    source: RetardedSelfEnergySource,
    electronic_state: HamiltonianSource,
    request: SpectralEvaluationRequest,
) -> SelfEnergyBatch:
    """Evaluate a typed retarded self-energy in its covariant orbital basis.

    The scalar self-energy family is retained only as the singleton orbital
    specialization. Tabulated values are selected exactly on their declared
    grid; interpolation belongs to the source-specific host/I/O airlock.
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
        return make_self_energy_batch(
            values,
            request,
            basis_ref=source.basis_ref,
            source_ref=source.source_ref,
            derivative_mode=source.derivative_mode,
        )
    if isinstance(source, TabulatedMatrixSelfEnergy):
        axes_match = jnp.array_equal(
            source.omega_rel_fermi_ev,
            request.omega_rel_fermi_ev,
        ) & jnp.array_equal(source.temperature_k, request.temperature_k)
        values: Complex128[Array, "n_temperature n_k n_omega n_orb n_orb"] = (
            eqx.error_if(
                source.values_ev,
                ~axes_match,
                "tabulated source requires exact-node evaluation",
            )
        )
        return make_self_energy_batch(
            values,
            request,
            basis_ref=source.basis_ref,
            source_ref=source.source_ref,
            derivative_mode=source.derivative_mode,
        )
    raise TypeError("unrecognized closed retarded self-energy source")


@jaxtyped(typechecker=beartype)
def solve_retarded_dyson(
    hamiltonian_rel_fermi_ev: Complex128[Array, "n_k n_orb n_orb"],
    overlap: Complex128[Array, "n_k n_orb n_orb"],
    self_energy: SelfEnergyBatch,
    request: SpectralEvaluationRequest,
) -> RetardedGreenBatch:
    """Solve the nonorthogonal retarded Dyson equation without an inverse."""
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
    if self_energy.request != request:
        raise ValueError(
            "Dyson self-energy must be evaluated for this request"
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
        - self_energy.values_ev
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
    return make_retarded_green_batch(
        values,
        metric,
        request,
        basis_ref=self_energy.basis_ref,
        source_ref=self_energy.source_ref,
        derivative_mode=self_energy.derivative_mode,
        validation_ref="org.diffpes.validation.dyson@1.0.0",
    )


@jaxtyped(typechecker=beartype)
def spectral_density_matrix(
    green: RetardedGreenBatch,
) -> Complex128[Array, "n_temperature n_k n_omega n_orb n_orb"]:
    """Return the Hermitian contravariant spectral-density matrix."""
    spectral: Complex128[Array, "n_temperature n_k n_omega n_orb n_orb"] = -(
        green.values_per_ev - _dagger(green.values_per_ev)
    ) / (2.0j * jnp.pi)
    return spectral


@jaxtyped(typechecker=beartype)
def total_spectral_density(
    green: RetardedGreenBatch,
) -> Float64[Array, "n_temperature n_k n_omega"]:
    """Return the metric trace of the retarded spectral-density matrix."""
    spectral: Complex128[Array, "n_temperature n_k n_omega n_orb n_orb"] = (
        spectral_density_matrix(green)
    )
    total: Float64[Array, "n_temperature n_k n_omega"] = jnp.real(
        jnp.einsum("kij,tkwji->tkw", green.overlap, spectral)
    )
    return total


@jaxtyped(typechecker=beartype)
def projected_spectral_density(
    green: RetardedGreenBatch,
    transition_sources: Complex128[
        Array, "n_temperature n_k n_omega n_out n_orb"
    ],
    *,
    source_variance: Literal["covariant"] = "covariant",
) -> Float64[Array, "n_temperature n_k n_omega n_out"]:
    """Contract covariant transition sources with the spectral matrix."""
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
    "solve_retarded_dyson",
    "spectral_density_matrix",
    "total_spectral_density",
]
