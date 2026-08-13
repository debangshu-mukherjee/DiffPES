"""Define intrinsic and observed ARPES result carriers.

Extended Summary
----------------
Use this module for its validated public contracts and operations.

Routine Listings
----------------
:class:`IntrinsicPhotocurrent`
    Define the ``IntrinsicPhotocurrent`` public contract.
:class:`SimulationResult`
    Define the ``SimulationResult`` public contract.
:func:`make_intrinsic_photocurrent`
    Compute the ``make_intrinsic_photocurrent`` public contract.
:func:`make_simulation_result`
    Compute the ``make_simulation_result`` public contract.
"""

import equinox as eqx
import jax.numpy as jnp
from beartype import beartype
from beartype.typing import Tuple
from jaxtyping import Array, Float64, jaxtyped

from .coordinates import MeasurementCoordinates
from .fidelity import FidelityManifest


class IntrinsicPhotocurrent(eqx.Module):
    """Define the ``IntrinsicPhotocurrent`` public contract.

    Validate documented inputs and preserve the declared scientific identity.

    :see: :class:`~.test_result.TestIntrinsicphotocurrent`

    Attributes
    ----------
    scalar_intensity_by_domain : Tuple[Float64[Array, "n_channel ..."], ...]
        Store domain intensities.
    coordinates : MeasurementCoordinates
        Store measurement coordinates.
    channel_labels : Tuple[str, ...]
        Store channel labels.
    payload_kind : str
        Store the payload kind.
    intensity_units : str
        Store intensity units.
    model_ref : str
        Store the model identity.
    state_ref : str
        Store the state identity.
    fidelity : FidelityManifest
        Store the fidelity declaration.

    See Also
    --------
    make_intrinsic_photocurrent
        Construct a validated photocurrent.
    """

    scalar_intensity_by_domain: Tuple[Float64[Array, "n_channel ..."], ...]
    coordinates: MeasurementCoordinates
    channel_labels: Tuple[str, ...] = eqx.field(static=True)
    payload_kind: str = eqx.field(static=True)
    intensity_units: str = eqx.field(static=True)
    model_ref: str = eqx.field(static=True)
    state_ref: str = eqx.field(static=True)
    fidelity: FidelityManifest = eqx.field(static=True)


class SimulationResult(eqx.Module):
    """Define the ``SimulationResult`` public contract.

    Validate documented inputs and preserve the declared scientific identity.

    :see: :class:`~.test_result.TestSimulationresult`

    Attributes
    ----------
    expected_counts : Float64[Array, "n_channel ..."]
        Store expected counts.
    coordinates : MeasurementCoordinates
        Store measurement coordinates.
    channel_labels : Tuple[str, ...]
        Store channel labels.
    fidelity : FidelityManifest
        Store the fidelity declaration.

    See Also
    --------
    make_simulation_result
        Construct a validated simulation result.
    """

    expected_counts: Float64[Array, "n_channel ..."]
    coordinates: MeasurementCoordinates
    channel_labels: Tuple[str, ...] = eqx.field(static=True)
    fidelity: FidelityManifest = eqx.field(static=True)


@jaxtyped(typechecker=beartype)
def make_intrinsic_photocurrent(
    scalar_intensity_by_domain: Tuple[Float64[Array, "n_channel ..."], ...],
    coordinates: MeasurementCoordinates,
    *,
    channel_labels: Tuple[str, ...],
    intensity_units: str,
    model_ref: str,
    state_ref: str,
    fidelity: FidelityManifest,
) -> IntrinsicPhotocurrent:
    """Compute the ``make_intrinsic_photocurrent`` public contract.

    Validate documented inputs and preserve the declared scientific identity.

    :see: :class:`~.test_result.TestMakeIntrinsicPhotocurrent`

    Notes
    -----
    Validate inputs before returning the named result.

    Parameters
    ----------
    scalar_intensity_by_domain : Tuple[Float64[Array, 'n_channel ...'], ...]
        Input value for this operation.
    coordinates : MeasurementCoordinates
        Input value for this operation.
    channel_labels : Tuple[str, ...]
        Input value for this operation.
    intensity_units : str
        Input value for this operation.
    model_ref : str
        Input value for this operation.
    state_ref : str
        Input value for this operation.
    fidelity : FidelityManifest
        Input value for this operation.

    Returns
    -------
    result : IntrinsicPhotocurrent
        Validated operation result.

    Raises
    ------
    ValueError
        If payload domains, channels, or scientific identities are
        inconsistent.
    """
    arrays: Tuple[Float64[Array, "n_channel ..."], ...] = tuple(
        jnp.asarray(value, dtype=jnp.float64)
        for value in scalar_intensity_by_domain
    )
    if not arrays:
        raise ValueError("at least one intrinsic domain payload is required")
    if not channel_labels or any(not label for label in channel_labels):
        raise ValueError("channel labels must be nonempty")
    if not intensity_units or not model_ref or not state_ref:
        raise ValueError("intrinsic result identity fields must be nonempty")
    if any(value.shape[0] != len(channel_labels) for value in arrays):
        raise ValueError(
            "intrinsic domain payloads must share the channel labels"
        )
    checked_arrays: Tuple[Float64[Array, "n_channel ..."], ...] = tuple(
        eqx.error_if(
            value,
            jnp.any(~jnp.isfinite(value)) | jnp.any(value < 0.0),
            "intrinsic intensity must be finite and nonnegative",
        )
        for value in arrays
    )
    result: IntrinsicPhotocurrent = IntrinsicPhotocurrent(
        checked_arrays,
        coordinates,
        channel_labels,
        "scalar_intensity",
        intensity_units,
        model_ref,
        state_ref,
        fidelity,
    )
    return result


@jaxtyped(typechecker=beartype)
def make_simulation_result(
    expected_counts: Float64[Array, "n_channel ..."],
    coordinates: MeasurementCoordinates,
    *,
    channel_labels: Tuple[str, ...],
    fidelity: FidelityManifest,
) -> SimulationResult:
    """Compute the ``make_simulation_result`` public contract.

    Validate documented inputs and preserve the declared scientific identity.

    :see: :class:`~.test_result.TestMakeSimulationResult`

    Notes
    -----
    Validate inputs before returning the named result.

    Parameters
    ----------
    expected_counts : Float64[Array, 'n_channel ...']
        Input value for this operation.
    coordinates : MeasurementCoordinates
        Input value for this operation.
    channel_labels : Tuple[str, ...]
        Input value for this operation.
    fidelity : FidelityManifest
        Input value for this operation.

    Returns
    -------
    result : SimulationResult
        Validated operation result.

    Raises
    ------
    ValueError
        If channel labels are empty or disagree with the count axis.
    """
    counts: Float64[Array, "n_channel ..."] = jnp.asarray(
        expected_counts, dtype=jnp.float64
    )
    if not channel_labels or any(not label for label in channel_labels):
        raise ValueError("channel labels must be nonempty")
    if counts.shape[0] != len(channel_labels):
        raise ValueError("result channels must match labels")
    checked_counts: Float64[Array, "n_channel ..."] = eqx.error_if(
        counts,
        jnp.any(~jnp.isfinite(counts)) | jnp.any(counts < 0.0),
        "expected counts must be finite and nonnegative",
    )
    result: SimulationResult = SimulationResult(
        checked_counts, coordinates, channel_labels, fidelity
    )
    return result


__all__: list[str] = [
    "IntrinsicPhotocurrent",
    "SimulationResult",
    "make_intrinsic_photocurrent",
    "make_simulation_result",
]
