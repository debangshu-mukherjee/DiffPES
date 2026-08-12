"""Define intrinsic and observed ARPES result carriers.

Extended Summary
----------------
This module separates intrinsic scalar photocurrent from detector-domain
expectation values.  The split prevents instrument response code from
mistaking channel-resolved physics for measured count data.

Routine Listings
----------------
:class:`IntrinsicPhotocurrent`
    Store channel-resolved intrinsic scalar intensities by domain.
:class:`SimulationResult`
    Store channel-resolved detector expectation values.
:func:`make_intrinsic_photocurrent`
    Construct a validated intrinsic-photocurrent carrier.
:func:`make_simulation_result`
    Construct a validated detector-result carrier.
"""

import equinox as eqx
import jax.numpy as jnp
from beartype import beartype
from beartype.typing import Tuple
from jaxtyping import Array, Float64, jaxtyped

from .coordinates import MeasurementCoordinates
from .fidelity import FidelityManifest


class IntrinsicPhotocurrent(eqx.Module):
    """Store late-domain-mixed channel-resolved scalar intensities.

    Attributes
    ----------
    scalar_intensity_by_domain : Tuple[Float64[Array, "n_channel ..."], ...]
        Intrinsic nonnegative intensity arrays, one for each domain.
    coordinates : MeasurementCoordinates
        Measurement coordinates shared by all domain payloads.
    channel_labels : Tuple[str, ...]
        Stable labels for the leading channel dimension.
    payload_kind : str
        Static discriminator; always ``"scalar_intensity"``.
    intensity_units : str
        Physical unit declaration for the intensity payload.
    model_ref : str
        Stable forward-model identity.
    state_ref : str
        Stable initial-state identity.
    fidelity : FidelityManifest
        Resolved scientific-fidelity declaration.

    See Also
    --------
    SimulationResult
        Stores the detector-domain expectation after instrument response.
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
    """Store channel-resolved detector expectation values.

    Attributes
    ----------
    expected_counts : Float64[Array, "n_channel ..."]
        Finite nonnegative detector expectation values by channel.
    coordinates : MeasurementCoordinates
        Detector-domain measurement coordinates.
    channel_labels : Tuple[str, ...]
        Stable labels for the leading channel dimension.
    fidelity : FidelityManifest
        Resolved scientific-fidelity declaration.

    See Also
    --------
    IntrinsicPhotocurrent
        Supplies scalar intensity before instrument response.
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
    """Construct a validated intrinsic-photocurrent carrier.

    Parameters
    ----------
    scalar_intensity_by_domain : Tuple[Float64[Array, "n_channel ..."], ...]
        Per-domain scalar intensity arrays with a leading channel axis.
    coordinates : MeasurementCoordinates
        Coordinates shared by all intensity arrays.
    channel_labels : Tuple[str, ...]
        Labels corresponding to the leading channel axis.
    intensity_units : str
        Physical intensity-unit declaration.
    model_ref : str
        Stable forward-model identity.
    state_ref : str
        Stable initial-state identity.
    fidelity : FidelityManifest
        Resolved scientific-fidelity declaration.

    Returns
    -------
    IntrinsicPhotocurrent
        Immutable, finite, nonnegative intrinsic payload.

    Raises
    ------
    ValueError
        If no domain payload is supplied or labels are inconsistent.
    """
    arrays = tuple(
        jnp.asarray(value, dtype=jnp.float64)
        for value in scalar_intensity_by_domain
    )
    if not arrays:
        raise ValueError("at least one intrinsic domain payload is required")
    if not channel_labels or any(not label for label in channel_labels):
        raise ValueError("channel labels must be nonempty")
    if any(value.shape[0] != len(channel_labels) for value in arrays):
        raise ValueError(
            "intrinsic domain payloads must share the channel labels"
        )
    checked_arrays = tuple(
        eqx.error_if(
            value,
            jnp.any(~jnp.isfinite(value)) | jnp.any(value < 0.0),
            "intrinsic intensity must be finite and nonnegative",
        )
        for value in arrays
    )
    return IntrinsicPhotocurrent(
        checked_arrays,
        coordinates,
        channel_labels,
        "scalar_intensity",
        intensity_units,
        model_ref,
        state_ref,
        fidelity,
    )


@jaxtyped(typechecker=beartype)
def make_simulation_result(
    expected_counts: Float64[Array, "n_channel ..."],
    coordinates: MeasurementCoordinates,
    *,
    channel_labels: Tuple[str, ...],
    fidelity: FidelityManifest,
) -> SimulationResult:
    """Construct a validated detector-result carrier.

    Parameters
    ----------
    expected_counts : Float64[Array, "n_channel ..."]
        Detector expectation values with a leading channel axis.
    coordinates : MeasurementCoordinates
        Detector-domain coordinates for the result.
    channel_labels : Tuple[str, ...]
        Labels corresponding to the leading channel axis.
    fidelity : FidelityManifest
        Resolved scientific-fidelity declaration.

    Returns
    -------
    SimulationResult
        Immutable finite, nonnegative detector expectation values.

    Raises
    ------
    ValueError
        If labels are missing or do not match the channel dimension.
    """
    counts = jnp.asarray(expected_counts, dtype=jnp.float64)
    if not channel_labels or any(not label for label in channel_labels):
        raise ValueError("channel labels must be nonempty")
    if counts.shape[0] != len(channel_labels):
        raise ValueError("result channels must match labels")
    checked_counts = eqx.error_if(
        counts,
        jnp.any(~jnp.isfinite(counts)) | jnp.any(counts < 0.0),
        "expected counts must be finite and nonnegative",
    )
    return SimulationResult(
        checked_counts, coordinates, channel_labels, fidelity
    )


__all__: list[str] = [
    "IntrinsicPhotocurrent",
    "SimulationResult",
    "make_intrinsic_photocurrent",
    "make_simulation_result",
]
