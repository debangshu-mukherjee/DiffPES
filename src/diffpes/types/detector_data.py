"""Define detector calibration and raster data structures.

Extended Summary
----------------
This module stores native detector coordinates, calibration data,
and expected counts on the recorded detector raster.

Routine Listings
----------------
:class:`DetectorCalibration`
    Store native detector-bin and point-spread calibration.
:class:`DetectorRaster`
    Store expected detector counts on native recorded coordinates.
:func:`make_detector_calibration`
    Create a validated ``DetectorCalibration`` instance.
:func:`make_detector_raster`
    Create a validated ``DetectorRaster`` instance.
"""

import equinox as eqx
import jax.numpy as jnp
from beartype import beartype
from beartype.typing import Optional, Tuple
from jaxtyping import Array, Float64, jaxtyped

from diffpes.constants import (
    DETECTOR_BOUNDARY_POLICY,
    DETECTOR_COORDINATE_SYSTEM,
    MIN_INTERPOLATION_AXIS_POINTS,
)

from .aliases import ScalarFloat


class DetectorCalibration(eqx.Module):
    """Store native detector-bin and point-spread calibration.

    The carrier defines recorded-coordinate edges, point-spread widths, and a
    fixed transmission domain. Static selectors make the coordinate and
    boundary conventions explicit at tracing time.

    :see: :class:`~.test_detector_data.TestDetectorCalibration`

    Attributes
    ----------
    u_bin_edges : Float64[Array, " n_u_plus_1"]
        Native detector ``T_x`` bin edges in radians.
    v_bin_edges : Float64[Array, " n_v_plus_1"]
        Native detector ``T_y`` bin edges in radians. A slit has two edges.
    energy_bin_edges_ev : Float64[Array, " n_e_plus_1"]
        Recorded energy-bin edges relative to the Fermi level in eV.
    psf_fwhm_u : Float64[Array, ""]
        Positive point-spread FWHM in native ``u`` radians.
    psf_fwhm_v : Float64[Array, ""]
        Positive point-spread FWHM in native ``v`` radians.
    psf_fwhm_energy_ev : Float64[Array, ""]
        Positive energy point-spread FWHM in eV.
    transmission_reference_domain_ev : Float64[Array, " 2"]
        Fixed true-kinetic-energy calibration domain in eV.
    coordinate_system : str
        **Static.** V1 is exactly ``"hemispherical_angles"``. Changing it
        triggers retracing.
    transmission_monotonic_sign : int
        **Static.** Declared transmission direction, exactly ``-1`` or ``1``.
        Changing it triggers retracing.
    boundary_policy : str
        **Static.** V1 is exactly ``"loss"``. Changing it triggers retracing.

    Notes
    -----
    This carrier is the sole authority for target bins and native detector
    PSF widths. The source mesh never infers or replaces these values.

    See Also
    --------
    make_detector_calibration : Validated factory for this type.
    """

    u_bin_edges: Float64[Array, " n_u_plus_1"]
    v_bin_edges: Float64[Array, " n_v_plus_1"]
    energy_bin_edges_ev: Float64[Array, " n_e_plus_1"]
    psf_fwhm_u: Float64[Array, ""]
    psf_fwhm_v: Float64[Array, ""]
    psf_fwhm_energy_ev: Float64[Array, ""]
    transmission_reference_domain_ev: Float64[Array, " 2"]
    coordinate_system: str = eqx.field(static=True)
    transmission_monotonic_sign: int = eqx.field(static=True)
    boundary_policy: str = eqx.field(static=True)


class DetectorRaster(eqx.Module):
    """Store expected detector counts on native recorded coordinates.

    The carrier associates a nonempty channel axis with native detector and
    energy coordinates. Optional quantization vectors remain attached to
    their detector pixels without relabeling them as momentum coordinates.

    :see: :class:`~.test_detector_data.TestDetectorRaster`

    Attributes
    ----------
    expected_counts : Float64[Array, "n_channel n_u n_v n_e"]
        Nonnegative expected counts per native recorded bin.
    detector_u_axis : Float64[Array, " n_u"]
        Native detector ``T_x`` bin centres in radians.
    detector_v_axis : Float64[Array, " n_v"]
        Native detector ``T_y`` bin centres in radians. A slit has length one.
    energy_axis : Float64[Array, " n_e"]
        Recorded energy-bin centres relative to the Fermi level in eV.
    quantization_axis : Optional[Float64[Array, "n_u n_v 3"]]
        Optional per-pixel laboratory spin-quantization axes. A constant axis
        is explicitly broadcast over native detector pixels.
    channel_labels : Tuple[str, ...]
        **Static.** One unique nonempty label per leading channel. Changing
        the labels triggers retracing.
    coordinate_system : str
        **Static.** V1 is exactly ``"hemispherical_angles"``. Changing it
        triggers retracing.

    Notes
    -----
    Detector arrays are not relabeled Cartesian momentum grids. This carrier
    keeps native coordinates and expected-count units explicit.

    See Also
    --------
    make_detector_raster : Validated factory for this type.
    """

    expected_counts: Float64[Array, "n_channel n_u n_v n_e"]
    detector_u_axis: Float64[Array, " n_u"]
    detector_v_axis: Float64[Array, " n_v"]
    energy_axis: Float64[Array, " n_e"]
    quantization_axis: Optional[Float64[Array, "n_u n_v 3"]]
    channel_labels: Tuple[str, ...] = eqx.field(static=True)
    coordinate_system: str = eqx.field(static=True)


@jaxtyped(typechecker=beartype)
def make_detector_raster(  # noqa: DOC503
    expected_counts: Float64[Array, "C U V Ei"],
    detector_u_axis: Float64[Array, " Ua"],
    detector_v_axis: Float64[Array, " Va"],
    energy_axis: Float64[Array, " Ea"],
    channel_labels: Tuple[str, ...],
    coordinate_system: str,
    quantization_axis: Optional[Float64[Array, "Uq Vq 3"]] = None,
) -> DetectorRaster:
    """Create a validated ``DetectorRaster`` instance.

    The factory aligns expected counts with native coordinate axes and static
    channel labels. It also validates optional per-pixel quantization vectors.

    :see: :class:`~.test_detector_data.TestMakeDetectorRaster`

    Parameters
    ----------
    expected_counts : Float64[Array, "C U V Ei"]
        Nonnegative expected counts in native detector bins.
    detector_u_axis : Float64[Array, " Ua"]
        Native detector ``T_x`` bin centres in radians.
    detector_v_axis : Float64[Array, " Va"]
        Native detector ``T_y`` bin centres in radians.
    energy_axis : Float64[Array, " Ea"]
        Recorded energy-bin centres relative to the Fermi level in eV.
    channel_labels : Tuple[str, ...]
        **Static.** One unique nonempty label per channel. Changing it triggers
        retracing.
    coordinate_system : str
        **Static.** Must be ``"hemispherical_angles"``. Changing it triggers
        retracing.
    quantization_axis : Optional[Float64[Array, "Uq Vq 3"]], optional
        Per-pixel laboratory spin-quantization axes.

    Returns
    -------
    raster : DetectorRaster
        Validated native-coordinate expected-count raster.

    Raises
    ------
    ValueError
        If dimensions or labels disagree, the channel axis is empty, or the
        coordinate system is invalid.
    EquinoxRuntimeError
        If values are non-finite, counts are negative, or a multi-point axis
        is not strictly increasing.

    Notes
    -----
    Static shape checks reject an empty channel axis before numerical
    validation runs in eager or compiled execution.
    """
    counts_arr: Float64[Array, "C U V E"] = jnp.asarray(
        expected_counts, dtype=jnp.float64
    )
    u_arr: Float64[Array, " U"] = jnp.asarray(
        detector_u_axis, dtype=jnp.float64
    )
    v_arr: Float64[Array, " V"] = jnp.asarray(
        detector_v_axis, dtype=jnp.float64
    )
    energy_arr: Float64[Array, " E"] = jnp.asarray(
        energy_axis, dtype=jnp.float64
    )
    quantization_arr: Optional[Float64[Array, "U V 3"]] = None
    if quantization_axis is not None:
        quantization_arr = jnp.asarray(quantization_axis, dtype=jnp.float64)
    if counts_arr.shape[1:] != (
        u_arr.shape[0],
        v_arr.shape[0],
        energy_arr.shape[0],
    ):
        raise ValueError("make_detector_raster: counts and axes disagree")
    if min(u_arr.shape[0], v_arr.shape[0], energy_arr.shape[0]) < 1:
        raise ValueError("make_detector_raster: axes cannot be empty")
    if counts_arr.shape[0] < 1:
        raise ValueError(
            "make_detector_raster: channel axis requires at least one channel"
        )
    if len(channel_labels) != counts_arr.shape[0]:
        raise ValueError("make_detector_raster: channel labels disagree")
    labels_invalid: bool = any(not label for label in channel_labels) or len(
        set(channel_labels)
    ) != len(channel_labels)
    if labels_invalid:
        raise ValueError(
            "make_detector_raster: channel labels must be unique and nonempty"
        )
    if (
        quantization_arr is not None
        and quantization_arr.shape[:2] != counts_arr.shape[1:3]
    ):
        raise ValueError(
            "make_detector_raster: quantization and detector axes disagree"
        )
    if coordinate_system != DETECTOR_COORDINATE_SYSTEM:
        raise ValueError("make_detector_raster: unknown coordinate system")

    def validate_and_create() -> DetectorRaster:
        """Validate traced leaves and construct the detector raster.

        Returns
        -------
        raster : DetectorRaster
            Validated native-coordinate expected-count raster.
        """
        nonlocal counts_arr, energy_arr, quantization_arr, u_arr, v_arr
        counts_arr = eqx.error_if(
            counts_arr,
            ~jnp.all(jnp.isfinite(counts_arr)),
            "make_detector_raster: expected counts finite",
        )
        counts_arr = eqx.error_if(
            counts_arr,
            ~jnp.all(counts_arr >= 0.0),
            "make_detector_raster: expected counts non negative",
        )
        u_arr = eqx.error_if(
            u_arr,
            ~jnp.all(jnp.isfinite(u_arr)) | ~jnp.all(jnp.diff(u_arr) > 0.0),
            "make_detector_raster: u axis finite and strictly increasing",
        )
        v_arr = eqx.error_if(
            v_arr,
            ~jnp.all(jnp.isfinite(v_arr)) | ~jnp.all(jnp.diff(v_arr) > 0.0),
            "make_detector_raster: v axis finite and strictly increasing",
        )
        energy_arr = eqx.error_if(
            energy_arr,
            ~jnp.all(jnp.isfinite(energy_arr))
            | ~jnp.all(jnp.diff(energy_arr) > 0.0),
            "make_detector_raster: energy axis finite and strictly increasing",
        )
        if quantization_arr is not None:
            quantization_arr = eqx.error_if(
                quantization_arr,
                ~jnp.all(jnp.isfinite(quantization_arr)),
                "make_detector_raster: quantization axis finite",
            )
        validated_raster: DetectorRaster = DetectorRaster(
            expected_counts=counts_arr,
            detector_u_axis=u_arr,
            detector_v_axis=v_arr,
            energy_axis=energy_arr,
            quantization_axis=quantization_arr,
            channel_labels=channel_labels,
            coordinate_system=coordinate_system,
        )
        return validated_raster

    raster: DetectorRaster = validate_and_create()
    return raster


@jaxtyped(typechecker=beartype)
def make_detector_calibration(  # noqa: DOC503
    u_bin_edges: Float64[Array, " U"],
    v_bin_edges: Float64[Array, " V"],
    energy_bin_edges_ev: Float64[Array, " E"],
    psf_fwhm_u: ScalarFloat,
    psf_fwhm_v: ScalarFloat,
    psf_fwhm_energy_ev: ScalarFloat,
    transmission_reference_domain_ev: Float64[Array, " 2"],
    coordinate_system: str = DETECTOR_COORDINATE_SYSTEM,
    transmission_monotonic_sign: int = 1,
    boundary_policy: str = DETECTOR_BOUNDARY_POLICY,
) -> DetectorCalibration:
    """Create a validated ``DetectorCalibration`` instance.

    The factory normalizes detector edges and point-spread widths to float64.
    It binds monotonicity, positivity, and fixed-policy checks to one carrier.

    :see: :class:`~.test_detector_data.TestMakeDetectorCalibration`

    Parameters
    ----------
    u_bin_edges : Float64[Array, " U"]
        Native detector ``T_x`` bin edges in radians.
    v_bin_edges : Float64[Array, " V"]
        Native detector ``T_y`` bin edges in radians.
    energy_bin_edges_ev : Float64[Array, " E"]
        Recorded energy-bin edges relative to the Fermi level in eV.
    psf_fwhm_u : ScalarFloat
        Positive native ``u`` point-spread FWHM in radians.
    psf_fwhm_v : ScalarFloat
        Positive native ``v`` point-spread FWHM in radians.
    psf_fwhm_energy_ev : ScalarFloat
        Positive energy point-spread FWHM in eV.
    transmission_reference_domain_ev : Float64[Array, " 2"]
        Fixed true-kinetic-energy calibration domain in eV.
    coordinate_system : str, optional
        **Static.** Must be ``"hemispherical_angles"``. Changing it triggers
        retracing.
    transmission_monotonic_sign : int, optional
        **Static.** Exactly ``-1`` or ``1``. Changing it triggers retracing.
    boundary_policy : str, optional
        **Static.** Must be ``"loss"``. Changing it triggers retracing.

    Returns
    -------
    calibration : DetectorCalibration
        Validated detector calibration with float64 numerical leaves.

    Raises
    ------
    ValueError
        If an edge axis has fewer than two entries or static metadata is not
        supported.
    EquinoxRuntimeError
        If values are non-finite, edges or the transmission domain are not
        strictly increasing, or a point-spread width is not positive.

    Notes
    -----
    Static policy checks run before value-threaded Equinox validation of the
    numerical calibration leaves.
    """
    u_edges: Float64[Array, " U"] = jnp.asarray(u_bin_edges, dtype=jnp.float64)
    v_edges: Float64[Array, " V"] = jnp.asarray(v_bin_edges, dtype=jnp.float64)
    energy_edges: Float64[Array, " E"] = jnp.asarray(
        energy_bin_edges_ev, dtype=jnp.float64
    )
    fwhm_u: Float64[Array, ""] = jnp.asarray(psf_fwhm_u, dtype=jnp.float64)
    fwhm_v: Float64[Array, ""] = jnp.asarray(psf_fwhm_v, dtype=jnp.float64)
    fwhm_energy: Float64[Array, ""] = jnp.asarray(
        psf_fwhm_energy_ev, dtype=jnp.float64
    )
    transmission_domain: Float64[Array, " 2"] = jnp.asarray(
        transmission_reference_domain_ev, dtype=jnp.float64
    )
    if (
        min(u_edges.shape[0], v_edges.shape[0], energy_edges.shape[0])
        < MIN_INTERPOLATION_AXIS_POINTS
    ):
        raise ValueError(
            "make_detector_calibration: bin edges require two points"
        )
    if coordinate_system != DETECTOR_COORDINATE_SYSTEM:
        raise ValueError(
            "make_detector_calibration: unknown coordinate system"
        )
    if transmission_monotonic_sign not in (-1, 1):
        raise ValueError(
            "make_detector_calibration: transmission sign must be -1 or 1"
        )
    if boundary_policy != DETECTOR_BOUNDARY_POLICY:
        raise ValueError(
            "make_detector_calibration: boundary policy must be loss"
        )

    def validate_and_create() -> DetectorCalibration:
        """Validate traced leaves and construct the calibration.

        Returns
        -------
        calibration : DetectorCalibration
            Validated detector-bin and response calibration.
        """
        nonlocal energy_edges, fwhm_energy, fwhm_u, fwhm_v
        nonlocal transmission_domain, u_edges, v_edges
        u_edges = eqx.error_if(
            u_edges,
            ~jnp.all(jnp.isfinite(u_edges))
            | ~jnp.all(jnp.diff(u_edges) > 0.0),
            "make_detector_calibration: u edges must be finite/increasing",
        )
        v_edges = eqx.error_if(
            v_edges,
            ~jnp.all(jnp.isfinite(v_edges))
            | ~jnp.all(jnp.diff(v_edges) > 0.0),
            "make_detector_calibration: v edges must be finite/increasing",
        )
        energy_edges = eqx.error_if(
            energy_edges,
            ~jnp.all(jnp.isfinite(energy_edges))
            | ~jnp.all(jnp.diff(energy_edges) > 0.0),
            "make_detector_calibration: energy edges finite/increasing",
        )
        fwhm_u = eqx.error_if(
            fwhm_u,
            ~jnp.isfinite(fwhm_u) | (fwhm_u <= 0.0),
            "make_detector_calibration: u FWHM finite and positive",
        )
        fwhm_v = eqx.error_if(
            fwhm_v,
            ~jnp.isfinite(fwhm_v) | (fwhm_v <= 0.0),
            "make_detector_calibration: v FWHM finite and positive",
        )
        fwhm_energy = eqx.error_if(
            fwhm_energy,
            ~jnp.isfinite(fwhm_energy) | (fwhm_energy <= 0.0),
            "make_detector_calibration: energy FWHM finite and positive",
        )
        transmission_domain = eqx.error_if(
            transmission_domain,
            ~jnp.all(jnp.isfinite(transmission_domain))
            | ~(transmission_domain[1] > transmission_domain[0]),
            "make_detector_calibration: transmission domain increasing",
        )
        validated_calibration: DetectorCalibration = DetectorCalibration(
            u_bin_edges=u_edges,
            v_bin_edges=v_edges,
            energy_bin_edges_ev=energy_edges,
            psf_fwhm_u=fwhm_u,
            psf_fwhm_v=fwhm_v,
            psf_fwhm_energy_ev=fwhm_energy,
            transmission_reference_domain_ev=transmission_domain,
            coordinate_system=coordinate_system,
            transmission_monotonic_sign=transmission_monotonic_sign,
            boundary_policy=boundary_policy,
        )
        return validated_calibration

    calibration: DetectorCalibration = validate_and_create()
    return calibration


__all__: list[str] = [
    "DetectorCalibration",
    "DetectorRaster",
    "make_detector_calibration",
    "make_detector_raster",
]
