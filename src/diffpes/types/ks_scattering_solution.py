"""Define scattering solver policies and evaluated result batches.

Extended Summary
----------------
Use this module for validated solver controls and scattering outputs.

Routine Listings
----------------
:class:`KSScatteringBatch`
    Define the ``KSScatteringBatch`` public contract.
:class:`KSScatteringSolverSpec`
    Define the ``KSScatteringSolverSpec`` public contract.
:func:`make_ks_scattering_batch`
    Compute the ``make_ks_scattering_batch`` public contract.
:func:`make_ks_scattering_solver_spec`
    Compute the ``make_ks_scattering_solver_spec`` public contract.
"""

# Exact pydoclint attribute types cannot split across physical lines.
# ruff: noqa: E501

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from beartype import beartype
from jaxtyping import Array, Bool, Complex128, Float64, jaxtyped


class KSScatteringSolverSpec(eqx.Module):
    """Define the ``KSScatteringSolverSpec`` public contract.

    Validate documented inputs and preserve the declared scientific identity.

    :see: :class:`~.test_ks_scattering_solution.TestKsscatteringsolverspec`

    Attributes
    ----------
    relative_residual : float
        Store relative tolerance.
    absolute_residual : float
        Store absolute tolerance.
    max_iterations : int
        Store the iteration limit.
    krylov_dimension : int
        Store the Krylov dimension.
    preconditioner_ref : str
        Store the preconditioner identity.
    threshold_guard_ev : float
        Store the threshold guard.

    See Also
    --------
    make_ks_scattering_solver_spec
        Construct a validated solver specification.
    """

    relative_residual: float = eqx.field(static=True)
    absolute_residual: float = eqx.field(static=True)
    max_iterations: int = eqx.field(static=True)
    krylov_dimension: int = eqx.field(static=True)
    preconditioner_ref: str = eqx.field(static=True)
    threshold_guard_ev: float = eqx.field(static=True)


class KSScatteringBatch(eqx.Module):
    """Define the ``KSScatteringBatch`` public contract.

    Validate documented inputs and preserve the declared scientific identity.

    :see: :class:`~.test_ks_scattering_solution.TestKsscatteringbatch`

    Attributes
    ----------
    states : Complex128[Array, "n_state n_slice n_chan n_out_spin"]
        Store scattering states.
    reflection_amplitudes : Complex128[Array, "n_state n_open n_out_spin"]
        Store reflection amplitudes.
    transmission_amplitudes : Complex128[Array, "n_state n_open n_out_spin"]
        Store transmission amplitudes.
    open_channel_mask : Bool[Array, "n_state n_chan"]
        Store open-channel masks.
    residual_norm : Float64[Array, " n_state"]
        Store residual norms.
    incident_flux : Float64[Array, " n_state"]
        Store incident fluxes.
    reflected_flux : Float64[Array, " n_state"]
        Store reflected fluxes.
    transmitted_flux : Float64[Array, " n_state"]
        Store transmitted fluxes.
    absorbed_flux : Float64[Array, " n_state"]
        Store absorbed fluxes.
    state_ref : str
        Store the state identity.

    See Also
    --------
    make_ks_scattering_batch
        Construct a validated scattering batch.
    """

    states: Complex128[Array, "n_state n_slice n_chan n_out_spin"]
    reflection_amplitudes: Complex128[Array, "n_state n_open n_out_spin"]
    transmission_amplitudes: Complex128[Array, "n_state n_open n_out_spin"]
    open_channel_mask: Bool[Array, "n_state n_chan"]
    residual_norm: Float64[Array, " n_state"]
    incident_flux: Float64[Array, " n_state"]
    reflected_flux: Float64[Array, " n_state"]
    transmitted_flux: Float64[Array, " n_state"]
    absorbed_flux: Float64[Array, " n_state"]
    state_ref: str = eqx.field(static=True)


@jaxtyped(typechecker=beartype)
def make_ks_scattering_solver_spec(
    *,
    relative_residual: float = 1.0e-10,
    absolute_residual: float = 1.0e-12,
    max_iterations: int = 500,
    krylov_dimension: int = 32,
    preconditioner_ref: str = "org.diffpes.preconditioner.kinetic@1.0.0",
    threshold_guard_ev: float = 1.0e-5,
) -> KSScatteringSolverSpec:
    """Compute the ``make_ks_scattering_solver_spec`` public contract.

    Validate documented inputs and preserve the declared scientific identity.

    :see: :class:`~.test_ks_scattering_solution.TestMakeKsScatteringSolverSpec`

    Notes
    -----
    Validate inputs before returning the named result.

    Parameters
    ----------
    relative_residual : float
        Input value for this operation.
    absolute_residual : float
        Input value for this operation.
    max_iterations : int
        Input value for this operation.
    krylov_dimension : int
        Input value for this operation.
    preconditioner_ref : str
        Input value for this operation.
    threshold_guard_ev : float
        Input value for this operation.

    Returns
    -------
    result : KSScatteringSolverSpec
        Validated operation result.

    Raises
    ------
    ValueError
        If solver tolerances, dimensions, or the preconditioner identity is
        invalid.
    """
    if (
        not all(
            np.isfinite(value)
            for value in (
                relative_residual,
                absolute_residual,
                threshold_guard_ev,
            )
        )
        or min(relative_residual, absolute_residual, threshold_guard_ev) <= 0.0
        or max_iterations <= 0
        or krylov_dimension <= 0
    ):
        raise ValueError(
            "scattering solver tolerances and dimensions must be positive"
        )
    if not preconditioner_ref:
        raise ValueError("scattering preconditioner reference is required")
    result: KSScatteringSolverSpec = KSScatteringSolverSpec(
        relative_residual,
        absolute_residual,
        max_iterations,
        krylov_dimension,
        preconditioner_ref,
        threshold_guard_ev,
    )
    return result


@jaxtyped(typechecker=beartype)  # noqa: PLR0913
def make_ks_scattering_batch(  # noqa: PLR0913
    states: Complex128[Array, "n_state n_slice n_chan n_out_spin"],
    reflection_amplitudes: Complex128[Array, "n_state n_open n_out_spin"],
    transmission_amplitudes: Complex128[Array, "n_state n_open n_out_spin"],
    open_channel_mask: Bool[Array, "n_state n_chan"],
    residual_norm: Float64[Array, " n_state"],
    incident_flux: Float64[Array, " n_state"],
    reflected_flux: Float64[Array, " n_state"],
    transmitted_flux: Float64[Array, " n_state"],
    absorbed_flux: Float64[Array, " n_state"],
    *,
    state_ref: str,
) -> KSScatteringBatch:
    """Compute the ``make_ks_scattering_batch`` public contract.

    Validate documented inputs and preserve the declared scientific identity.

    :see: :class:`~.test_ks_scattering_solution.TestMakeKsScatteringBatch`

    Notes
    -----
    Validate inputs before returning the named result.

    Parameters
    ----------
    states : Complex128[Array, 'n_state n_slice n_chan n_out_spin']
        Input value for this operation.
    reflection_amplitudes : Complex128[Array, 'n_state n_open n_out_spin']
        Input value for this operation.
    transmission_amplitudes : Complex128[Array, 'n_state n_open n_out_spin']
        Input value for this operation.
    open_channel_mask : Bool[Array, 'n_state n_chan']
        Input value for this operation.
    residual_norm : Float64[Array, ' n_state']
        Input value for this operation.
    incident_flux : Float64[Array, ' n_state']
        Input value for this operation.
    reflected_flux : Float64[Array, ' n_state']
        Input value for this operation.
    transmitted_flux : Float64[Array, ' n_state']
        Input value for this operation.
    absorbed_flux : Float64[Array, ' n_state']
        Input value for this operation.
    state_ref : str
        Input value for this operation.

    Returns
    -------
    result : KSScatteringBatch
        Validated operation result.

    Raises
    ------
    ValueError
        If the batch identity or numerical axes are inconsistent.
    """
    if not state_ref:
        raise ValueError("scattering batch state_ref must be nonempty")
    state_values: Complex128[Array, "n_state n_slice n_chan n_out_spin"] = (
        jnp.asarray(states, dtype=jnp.complex128)
    )
    reflection: Complex128[Array, "n_state n_open n_out_spin"] = jnp.asarray(
        reflection_amplitudes, dtype=jnp.complex128
    )
    transmission: Complex128[Array, "n_state n_open n_out_spin"] = jnp.asarray(
        transmission_amplitudes, dtype=jnp.complex128
    )
    mask: Bool[Array, "n_state n_chan"] = jnp.asarray(
        open_channel_mask, dtype=jnp.bool_
    )
    residual: Float64[Array, " n_state"] = jnp.asarray(
        residual_norm, dtype=jnp.float64
    )
    incident: Float64[Array, " n_state"] = jnp.asarray(
        incident_flux, dtype=jnp.float64
    )
    reflected: Float64[Array, " n_state"] = jnp.asarray(
        reflected_flux, dtype=jnp.float64
    )
    transmitted: Float64[Array, " n_state"] = jnp.asarray(
        transmitted_flux, dtype=jnp.float64
    )
    absorbed: Float64[Array, " n_state"] = jnp.asarray(
        absorbed_flux, dtype=jnp.float64
    )
    n_states: int = state_values.shape[0]
    if (
        state_values.ndim != 4  # noqa: PLR2004
        or reflection.shape[0] != n_states
        or transmission.shape != reflection.shape
        or mask.shape[0] != n_states
        or mask.shape[1] != state_values.shape[2]
        or reflection.shape[-1] != state_values.shape[-1]
        or any(
            value.shape != (n_states,)
            for value in (
                residual,
                incident,
                reflected,
                transmitted,
                absorbed,
            )
        )
    ):
        raise ValueError("scattering batch axes are inconsistent")
    state_values = eqx.error_if(
        state_values,
        ~jnp.all(jnp.isfinite(state_values))
        | ~jnp.all(jnp.isfinite(reflection))
        | ~jnp.all(jnp.isfinite(transmission))
        | ~jnp.all(jnp.isfinite(residual))
        | ~jnp.all(jnp.isfinite(incident))
        | ~jnp.all(jnp.isfinite(reflected))
        | ~jnp.all(jnp.isfinite(transmitted))
        | ~jnp.all(jnp.isfinite(absorbed))
        | jnp.any(residual < 0.0)
        | jnp.any(incident <= 0.0)
        | jnp.any(reflected < 0.0)
        | jnp.any(transmitted < 0.0)
        | jnp.any(absorbed < 0.0),
        "scattering batch diagnostics must be finite and physical",
    )
    result: KSScatteringBatch = KSScatteringBatch(
        state_values,
        reflection,
        transmission,
        mask,
        residual,
        incident,
        reflected,
        transmitted,
        absorbed,
        state_ref,
    )
    return result


__all__: list[str] = [
    "KSScatteringBatch",
    "KSScatteringSolverSpec",
    "make_ks_scattering_batch",
    "make_ks_scattering_solver_spec",
]
