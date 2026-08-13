"""Compose typed electronic-state, factorized-current, and observation.

Extended Summary
----------------
Use this module for its validated public contracts and operations.

Routine Listings
----------------
:func:`evaluate_spectral_projection`
    Compute the ``evaluate_spectral_projection`` public contract.
"""

import jax.numpy as jnp
from beartype import beartype
from beartype.typing import Tuple
from jaxtyping import Array, Complex128, Float64, jaxtyped

from diffpes.types import (
    FactorizedArpesModel,
    FidelityManifest,
    HamiltonianOverlapSource,
    IntrinsicPhotocurrent,
    MeasurementCoordinates,
    ParametricSelfEnergy,
    SelfEnergyBatch,
    SpectralEvaluationRequest,
    make_fidelity_manifest,
    make_intrinsic_photocurrent,
    make_parametric_self_energy,
    make_spectral_evaluation_request,
)

from .generalized_spectral import (
    _evaluate_retarded_self_energy,
    projected_spectral_density_solve,
)


@jaxtyped(typechecker=beartype)
def evaluate_spectral_projection(
    model: FactorizedArpesModel,
    electronic_state: HamiltonianOverlapSource,
    coordinates: MeasurementCoordinates,
    omega_rel_fermi_ev: Float64[Array, " n_omega"],
    temperature_k: Float64[Array, " n_temperature"],
) -> IntrinsicPhotocurrent:
    """Compute the ``evaluate_spectral_projection`` public contract.

    Validate documented inputs and preserve the declared scientific identity.

    :see: :class:`~.test_factorized.TestEvaluateSpectralProjection`

    Notes
    -----
    Validate inputs before returning the named result.

    Parameters
    ----------
    model : FactorizedArpesModel
        Input value for this operation.
    electronic_state : HamiltonianOverlapSource
        Input value for this operation.
    coordinates : MeasurementCoordinates
        Input value for this operation.
    omega_rel_fermi_ev : Float64[Array, ' n_omega']
        Input value for this operation.
    temperature_k : Float64[Array, ' n_temperature']
        Input value for this operation.

    Returns
    -------
    result : IntrinsicPhotocurrent
        Validated operation result.

    Raises
    ------
    ValueError
        If the electronic-state source lacks a required capability.
    """
    required: set[str] = set(model.required_capabilities)
    available: set[str] = set(electronic_state.capabilities)
    missing: Tuple[str, ...] = tuple(sorted(required - available))
    if missing:
        missing_text: str = ", ".join(missing)
        raise ValueError(
            f"electronic state lacks required capabilities: {missing_text}"
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
    sigma: SelfEnergyBatch = _evaluate_retarded_self_energy(
        source, electronic_state, request
    )
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
        projected_spectral_density_solve(
            hamiltonian,
            overlap,
            sigma,
            request,
            sources,
        )
    )
    payload: Float64[Array, "n_channel n_temperature n_k n_omega"] = (
        jnp.moveaxis(intensity, -1, 0)
    )
    fidelity: FidelityManifest = make_fidelity_manifest(
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
    result: IntrinsicPhotocurrent = make_intrinsic_photocurrent(
        (payload,),
        coordinates,
        channel_labels=tuple(f"orbital_{index}" for index in range(n_orb)),
        intensity_units="1/eV",
        model_ref=model.model_ref,
        state_ref=electronic_state.state_ref,
        fidelity=fidelity,
    )
    return result


__all__: list[str] = ["evaluate_spectral_projection"]
