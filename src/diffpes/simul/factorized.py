"""Compose typed electronic-state, factorized-current, and observation
stages.
"""

import jax.numpy as jnp
from beartype import beartype
from jaxtyping import Array, Complex128, Float64, jaxtyped

from diffpes.types import (
    FactorizedArpesModel,
    HamiltonianOverlapSource,
    IntrinsicPhotocurrent,
    MeasurementCoordinates,
    ParametricSelfEnergy,
    SpectralEvaluationRequest,
    make_fidelity_manifest,
    make_intrinsic_photocurrent,
    make_parametric_self_energy,
    make_spectral_evaluation_request,
)

from .generalized_spectral import (
    _evaluate_retarded_self_energy,
    projected_spectral_density,
    solve_retarded_dyson,
)


@jaxtyped(typechecker=beartype)
def evaluate_spectral_projection(
    model: FactorizedArpesModel,
    electronic_state: HamiltonianOverlapSource,
    coordinates: MeasurementCoordinates,
    omega_rel_fermi_ev: Float64[Array, " n_omega"],
    temperature_k: Float64[Array, " n_temperature"],
) -> IntrinsicPhotocurrent:
    """Evaluate orbital-projected spectral density through the Dyson path.

    This source-level observable remains separate from the later instrument
    response. It returns scalar intrinsic intensity only.
    """
    required = set(model.required_capabilities)
    available = set(electronic_state.capabilities)
    if not required <= available:
        raise ValueError(
            "electronic state lacks factorized model capabilities"
        )
    hamiltonian: Complex128[Array, "n_k n_orb n_orb"] = (
        electronic_state.hamiltonian(coordinates)
    )
    overlap: Complex128[Array, "n_k n_orb n_orb"] = electronic_state.overlap(
        coordinates
    )
    request: SpectralEvaluationRequest = make_spectral_evaluation_request(
        coordinates,
        omega_rel_fermi_ev,
        temperature_k,
        model.eta_ev,
        basis_ref="scalar",
    )
    source: ParametricSelfEnergy = make_parametric_self_energy(
        model.self_energy,
        source_ref="org.diffpes.spectral.scalar@1.0.0",
        basis_ref="scalar",
        provenance_ref="org.diffpes.provenance.native@1.0.0",
    )
    sigma = _evaluate_retarded_self_energy(source, electronic_state, request)
    green = solve_retarded_dyson(hamiltonian, overlap, sigma, request)
    n_orb: int = hamiltonian.shape[-1]
    sources: Complex128[Array, "n_temperature n_k n_omega n_out n_orb"] = (
        jnp.broadcast_to(
            jnp.eye(n_orb, dtype=jnp.complex128)[None, None, None],
            (
                temperature_k.shape[0],
                hamiltonian.shape[0],
                omega_rel_fermi_ev.shape[0],
                n_orb,
                n_orb,
            ),
        )
    )
    intensity: Float64[Array, "n_temperature n_k n_omega n_out"] = (
        projected_spectral_density(green, sources)
    )
    payload: Float64[Array, "n_channel n_temperature n_k n_omega"] = (
        jnp.moveaxis(intensity, -1, 0)
    )
    fidelity = make_fidelity_manifest(
        schema_version="1.0",
        model_ref="org.diffpes.model.arpes.spectral_projection@0.1.0",
        instrument_ref="org.diffpes.instrument.none@0.1.0",
        acquisition_ref="org.diffpes.acquisition.none@0.1.0",
        initial_state="tb",
        spectral_physics="scalar_sigma",
        photocurrent="spectral_projection",
        light_interaction="none",
        instrument="none",
    )
    return make_intrinsic_photocurrent(
        (payload,),
        coordinates,
        channel_labels=tuple(f"orbital_{index}" for index in range(n_orb)),
        intensity_units="1/eV",
        model_ref=model.model_ref,
        state_ref=electronic_state.state_ref,
        fidelity=fidelity,
    )


__all__: list[str] = ["evaluate_spectral_projection"]
