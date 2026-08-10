r"""Assemble detector-coordinate backgrounds and expected counts.

Extended Summary
----------------
This module implements the bounded WP8.8 detector-effects foundation after
conservative mapping and native-coordinate resolution. It turns a
nonnegative detector density into expected bin counts with the binding
background, sensitivity, bin-volume, exposure, and optional calibrated
energy-response stages. Sampling takes explicit JAX PRNG keys and remains
outside the differentiable expected-rate graph.

Routine Listings
----------------
:func:`apply_post_count_response`
    Convolve expected counts along the recorded-energy index.
:func:`background_density`
    Evaluate a nonnegative detector-coordinate background.
:func:`detector_bin_volumes`
    Compute explicit native detector-bin volumes.
:func:`expected_counts`
    Assemble deterministic expected detector counts.
:func:`fixed_total_probabilities`
    Normalize all detector rates to one event-probability tensor.
:func:`sample_fixed_total_counts`
    Generate one fixed-total multinomial count tensor.
:func:`sample_poisson_counts`
    Generate independent Poisson counts for a rate tensor.
:func:`sensitivity_field`
    Evaluate the positive normalized detector sensitivity field.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
from beartype import beartype
from beartype.typing import Tuple
from jaxtyping import Array, Float64, Int, PRNGKeyArray, jaxtyped

from diffpes.types import DetectorCalibration, DetectorEffects

__all__: list[str] = [
    "apply_post_count_response",
    "background_density",
    "detector_bin_volumes",
    "expected_counts",
    "fixed_total_probabilities",
    "sample_fixed_total_counts",
    "sample_poisson_counts",
    "sensitivity_field",
]


def _validated_detector_density(  # noqa: DOC503
    detector_density: Float64[Array, "C U V E"],
    calibration: DetectorCalibration,
) -> Float64[Array, "C U V E"]:
    """PRIVATE: Validate one post-resolution detector-density raster.

    The helper checks dimensions, target-bin agreement, finite values, and
    nonnegative density before later detector stages consume the raster.

    Parameters
    ----------
    detector_density : Float64[Array, "C U V E"]
        Candidate post-resolution detector density.
    calibration : DetectorCalibration
        Target calibration that fixes the detector dimensions.

    Returns
    -------
    validated_density : Float64[Array, "C U V E"]
        Validated detector density in float64 precision.

    Raises
    ------
    ValueError
        If the raster rank, channel count, or target dimensions are invalid.
    EquinoxRuntimeError
        If a value is non-finite or negative.
    """
    density: Float64[Array, "C U V E"] = jnp.asarray(
        detector_density, dtype=jnp.float64
    )
    if density.ndim != 4:  # noqa: PLR2004
        raise ValueError("detector density must have four dimensions")
    if density.shape[0] < 1:
        raise ValueError("detector density channel axis cannot be empty")
    expected_shape: Tuple[int, int, int] = (
        calibration.u_bin_edges.shape[0] - 1,
        calibration.v_bin_edges.shape[0] - 1,
        calibration.energy_bin_edges_ev.shape[0] - 1,
    )
    if density.shape[1:] != expected_shape:
        raise ValueError("detector density and calibration bins disagree")
    validated_density: Float64[Array, "C U V E"] = eqx.error_if(
        density,
        ~jnp.all(jnp.isfinite(density)) | ~jnp.all(density >= 0.0),
        "detector density must be finite and nonnegative",
    )
    return validated_density


def _normalized_bin_centres(
    edges: Float64[Array, " Np1"],
) -> Float64[Array, " N"]:
    """PRIVATE: Normalize bin centres from the outer edges to [-1, 1].

    The affine coordinate supports the fixed detector Legendre basis.

    Parameters
    ----------
    edges : Float64[Array, " Np1"]
        Strictly increasing detector-bin edges.

    Returns
    -------
    normalized : Float64[Array, " N"]
        Normalized coordinate at every bin centre.
    """
    centres: Float64[Array, " N"] = 0.5 * (edges[:-1] + edges[1:])
    span: Float64[Array, ""] = edges[-1] - edges[0]
    normalized: Float64[Array, " N"] = 2.0 * (centres - edges[0]) / span - 1.0
    return normalized


def _active_legendre_fields(
    calibration: DetectorCalibration,
) -> Tuple[Float64[Array, "U V E"], ...]:
    """PRIVATE: Return ordered P1/P2 fields for active detector axes.

    The function omits the one-bin slit axis and preserves ``(u, v, E)``
    ordering for a detector map.

    Parameters
    ----------
    calibration : DetectorCalibration
        Complete detector calibration and target-bin edges.

    Returns
    -------
    basis_tuple : Tuple[Float64[Array, "U V E"], ...]
        Ordered first- and second-degree Legendre fields.
    """
    volumes: Float64[Array, "U V E"] = detector_bin_volumes(calibration)
    target_shape: Tuple[int, int, int] = volumes.shape
    u_coordinate: Float64[Array, " U"] = _normalized_bin_centres(
        calibration.u_bin_edges
    )
    energy_coordinate: Float64[Array, " E"] = _normalized_bin_centres(
        calibration.energy_bin_edges_ev
    )
    u_field: Float64[Array, "U V E"] = jnp.broadcast_to(
        u_coordinate[:, None, None], target_shape
    )
    energy_field: Float64[Array, "U V E"] = jnp.broadcast_to(
        energy_coordinate[None, None, :], target_shape
    )
    coordinate_fields: Tuple[Float64[Array, "U V E"], ...]
    if target_shape[1] > 1:
        v_coordinate: Float64[Array, " V"] = _normalized_bin_centres(
            calibration.v_bin_edges
        )
        v_field: Float64[Array, "U V E"] = jnp.broadcast_to(
            v_coordinate[None, :, None], target_shape
        )
        coordinate_fields = (u_field, v_field, energy_field)
    else:
        coordinate_fields = (u_field, energy_field)
    basis_fields: list[Float64[Array, "U V E"]] = []
    coordinate: Float64[Array, "U V E"]
    for coordinate in coordinate_fields:
        basis_fields.extend(
            (coordinate, 0.5 * (3.0 * jnp.square(coordinate) - 1.0))
        )
    basis_tuple: Tuple[Float64[Array, "U V E"], ...] = tuple(basis_fields)
    return basis_tuple


@jaxtyped(typechecker=beartype)
def detector_bin_volumes(
    calibration: DetectorCalibration,
) -> Float64[Array, "U V E"]:
    """Compute explicit native detector-bin volumes.

    The calculation retains every declared native width, including the
    single-bin slit width.

    :see: :class:`~.test_effects.TestDetectorBinVolumes`

    Parameters
    ----------
    calibration : DetectorCalibration
        Target detector edges. A one-bin native ``v`` axis still contributes
        its declared slit width.

    Returns
    -------
    volumes : Float64[Array, "U V E"]
        Products ``Delta u * Delta v * Delta E`` for all target bins.

    Notes
    -----
    The function multiplies edge differences without inferring widths from
    source arrays.
    """
    delta_u: Float64[Array, " U"] = jnp.diff(calibration.u_bin_edges)
    delta_v: Float64[Array, " V"] = jnp.diff(calibration.v_bin_edges)
    delta_energy: Float64[Array, " E"] = jnp.diff(
        calibration.energy_bin_edges_ev
    )
    volumes: Float64[Array, "U V E"] = (
        delta_u[:, None, None]
        * delta_v[None, :, None]
        * delta_energy[None, None, :]
    )
    validated_volumes: Float64[Array, "U V E"] = eqx.error_if(
        volumes,
        ~jnp.all(jnp.isfinite(volumes)) | ~jnp.all(volumes > 0.0),
        "detector bin volumes must be finite and positive",
    )
    return validated_volumes


@jaxtyped(typechecker=beartype)
def background_density(  # noqa: DOC503
    detector_density: Float64[Array, "C U V E"],
    calibration: DetectorCalibration,
    effects: DetectorEffects,
) -> Float64[Array, "C U V E"]:
    """Evaluate a nonnegative detector-coordinate background.

    The selected v1 basis produces a detector density after native-coordinate
    resolution and before sensitivity or exposure.

    :see: :class:`~.test_effects.TestBackgroundDensity`

    Parameters
    ----------
    detector_density : Float64[Array, "C U V E"]
        Finite nonnegative post-resolution density per native volume.
    calibration : DetectorCalibration
        Complete target-bin calibration.
    effects : DetectorEffects
        Background selector and raw coefficients.

    Returns
    -------
    background : Float64[Array, "C U V E"]
        Nonnegative background in the same density unit as the signal.

    Raises
    ------
    ValueError
        If detector dimensions or calibration-specific coefficient lengths
        disagree.
    EquinoxRuntimeError
        If the input density or result is non-finite or negative.

    Notes
    -----
    The Shirley branch integrates energy-bin mass toward the largest recorded
    energy. Its exact zero-signal branch has zero derivative.
    """
    density: Float64[Array, "C U V E"] = _validated_detector_density(
        detector_density, calibration
    )
    coefficients: Float64[Array, " B"] = effects.background_coefficients
    expected_length: int
    background: Float64[Array, "C U V E"]
    if effects.background_mode == "flat":
        expected_length = 1
        if coefficients.shape[0] != expected_length:
            raise ValueError("flat background requires one coefficient")
        background = jnp.broadcast_to(
            jax.nn.softplus(coefficients[0]), density.shape
        )
    elif effects.background_mode == "shirley":
        expected_length = 2
        if coefficients.shape[0] != expected_length:
            raise ValueError("Shirley background requires two coefficients")
        delta_energy: Float64[Array, " E"] = jnp.diff(
            calibration.energy_bin_edges_ev
        )
        weighted_density: Float64[Array, "C U V E"] = (
            density * delta_energy[None, None, None, :]
        )
        tail: Float64[Array, "C U V E"] = jnp.flip(
            jnp.cumsum(jnp.flip(weighted_density, axis=-1), axis=-1),
            axis=-1,
        )
        denominator: Float64[Array, "C U V 1"] = jnp.sum(
            weighted_density, axis=-1, keepdims=True
        )
        safe_denominator: Float64[Array, "C U V 1"] = jnp.where(
            denominator == 0.0, 1.0, denominator
        )
        quotient: Float64[Array, "C U V E"] = tail / safe_denominator
        quotient = jnp.where(denominator == 0.0, 0.0, quotient)
        background = (
            jax.nn.softplus(coefficients[0])
            + jax.nn.softplus(coefficients[1]) * quotient
        )
    elif effects.background_mode == "smooth":
        basis: Tuple[Float64[Array, "U V E"], ...] = _active_legendre_fields(
            calibration
        )
        expected_length = 1 + len(basis)
        if coefficients.shape[0] != expected_length:
            raise ValueError(
                "smooth background coefficient length disagrees with "
                "detector dimensionality"
            )
        raw_background: Float64[Array, "U V E"] = jnp.broadcast_to(
            coefficients[0], basis[0].shape
        )
        coefficient: Float64[Array, ""]
        basis_field: Float64[Array, "U V E"]
        for coefficient, basis_field in zip(
            coefficients[1:], basis, strict=True
        ):
            raw_background = raw_background + coefficient * basis_field
        background = jnp.broadcast_to(
            jax.nn.softplus(raw_background)[None, ...], density.shape
        )
    else:
        raise ValueError("unsupported detector background mode")
    background = eqx.error_if(
        background,
        ~jnp.all(jnp.isfinite(background)) | ~jnp.all(background >= 0.0),
        "detector background must be finite and nonnegative",
    )
    return background


@jaxtyped(typechecker=beartype)
def sensitivity_field(  # noqa: DOC503
    calibration: DetectorCalibration,
    effects: DetectorEffects,
) -> Float64[Array, "U V E"]:
    """Evaluate the positive normalized detector sensitivity field.

    The smooth mode removes its constant gauge through one full-calibration
    volume-weighted normalization.

    :see: :class:`~.test_effects.TestSensitivityField`

    Parameters
    ----------
    calibration : DetectorCalibration
        Complete detector calibration used for the fixed normalization.
    effects : DetectorEffects
        Sensitivity selector and raw coefficients.

    Returns
    -------
    sensitivity : Float64[Array, "U V E"]
        Positive field with exact full-bin-volume-weighted mean one.

    Raises
    ------
    ValueError
        If the coefficient length disagrees with detector dimensionality.
    EquinoxRuntimeError
        If exponentiation or normalization produces an invalid field.

    Notes
    -----
    The function always uses every target bin in the supplied calibration.
    Caller-side display crops do not enter this operation.
    """
    volumes: Float64[Array, "U V E"] = detector_bin_volumes(calibration)
    coefficients: Float64[Array, " S"] = effects.sensitivity_coefficients
    if effects.sensitivity_mode == "constant":
        if coefficients.shape[0] != 0:
            raise ValueError("constant sensitivity requires no coefficients")
        sensitivity: Float64[Array, "U V E"] = jnp.ones_like(volumes)
    elif effects.sensitivity_mode == "smooth":
        basis: Tuple[Float64[Array, "U V E"], ...] = _active_legendre_fields(
            calibration
        )
        if coefficients.shape[0] != len(basis):
            raise ValueError(
                "smooth sensitivity coefficient length disagrees with "
                "detector dimensionality"
            )
        log_sensitivity: Float64[Array, "U V E"] = jnp.zeros_like(volumes)
        coefficient: Float64[Array, ""]
        basis_field: Float64[Array, "U V E"]
        for coefficient, basis_field in zip(coefficients, basis, strict=True):
            log_sensitivity = log_sensitivity + coefficient * basis_field
        unnormalized: Float64[Array, "U V E"] = jnp.exp(log_sensitivity)
        unnormalized = eqx.error_if(
            unnormalized,
            ~jnp.all(jnp.isfinite(unnormalized))
            | ~jnp.all(unnormalized > 0.0),
            "detector sensitivity must be finite and positive",
        )
        volume_mean: Float64[Array, ""] = jnp.sum(
            unnormalized * volumes
        ) / jnp.sum(volumes)
        volume_mean = eqx.error_if(
            volume_mean,
            ~jnp.isfinite(volume_mean) | ~(volume_mean > 0.0),
            "detector sensitivity normalization must be positive",
        )
        sensitivity = unnormalized / volume_mean
    else:
        raise ValueError("unsupported detector sensitivity mode")
    return sensitivity


@jaxtyped(typechecker=beartype)
def apply_post_count_response(  # noqa: DOC503
    rates: Float64[Array, "C U V E"],
    effects: DetectorEffects,
) -> Float64[Array, "C U V E"]:
    """Convolve expected counts along the recorded-energy index.

    Calibrated mode applies one normalized odd kernel with zero exterior
    padding and retains physical edge loss.

    :see: :class:`~.test_effects.TestApplyPostCountResponse`

    Parameters
    ----------
    rates : Float64[Array, "C U V E"]
        Finite nonnegative expected counts before MCP/ADC spreading.
    effects : DetectorEffects
        Post-count selector and normalized odd kernel.

    Returns
    -------
    convolved : Float64[Array, "C U V E"]
        Counts after zero-padded energy-only convolution. Lost response at
        the recorded edges is not renormalized.

    Raises
    ------
    ValueError
        If rates are not four-dimensional or calibrated mode lacks a valid
        odd one-dimensional kernel.
    EquinoxRuntimeError
        If rates or the convolved result are invalid.

    Notes
    -----
    The operation never spreads counts across detector channels or angular
    bins. ``none`` mode returns the validated input unchanged.
    """
    rate_array: Float64[Array, "C U V E"] = jnp.asarray(
        rates, dtype=jnp.float64
    )
    if rate_array.ndim != 4:  # noqa: PLR2004
        raise ValueError("post-count rates must have four dimensions")
    if rate_array.shape[0] < 1:
        raise ValueError("post-count rate channel axis cannot be empty")
    rate_array = eqx.error_if(
        rate_array,
        ~jnp.all(jnp.isfinite(rate_array)) | ~jnp.all(rate_array >= 0.0),
        "post-count rates must be finite and nonnegative",
    )
    if effects.post_count_mode == "none":
        if effects.post_count_kernel is not None:
            raise ValueError("none post-count mode requires no kernel")
        return rate_array
    if effects.post_count_mode != "calibrated":
        raise ValueError("unsupported post-count mode")
    kernel: Float64[Array, " K"] | None = effects.post_count_kernel
    if (
        kernel is None
        or kernel.ndim != 1
        or kernel.shape[0] < 1
        or kernel.shape[0] % 2 != 1
    ):
        raise ValueError("calibrated post-count mode requires an odd kernel")
    half_width: int = kernel.shape[0] // 2
    rows: Float64[Array, "N 1 E"] = rate_array.reshape(
        (-1, 1, rate_array.shape[-1])
    )
    response: Float64[Array, "1 1 K"] = kernel[::-1].reshape(
        (1, 1, kernel.shape[0])
    )
    convolved_rows: Float64[Array, "N 1 E"] = jax.lax.conv_general_dilated(
        rows,
        response,
        window_strides=(1,),
        padding=((half_width, half_width),),
        dimension_numbers=("NCH", "OIH", "NCH"),
    )
    convolved: Float64[Array, "C U V E"] = convolved_rows.reshape(
        rate_array.shape
    )
    validated_convolved: Float64[Array, "C U V E"] = eqx.error_if(
        convolved,
        ~jnp.all(jnp.isfinite(convolved)) | ~jnp.all(convolved >= 0.0),
        "post-count response must be finite and nonnegative",
    )
    return validated_convolved


@jaxtyped(typechecker=beartype)
def expected_counts(  # noqa: DOC502
    detector_density: Float64[Array, "C U V E"],
    calibration: DetectorCalibration,
    effects: DetectorEffects,
) -> Float64[Array, "C U V E"]:
    """Assemble deterministic expected detector counts.

    The stage combines post-resolution density, background, normalized
    sensitivity, exposure, native bin volume, and calibrated response.

    :see: :class:`~.test_effects.TestExpectedCounts`

    Parameters
    ----------
    detector_density : Float64[Array, "C U V E"]
        Post-resolution detector density per native coordinate volume.
    calibration : DetectorCalibration
        Complete native target bins.
    effects : DetectorEffects
        Background, sensitivity, exposure, and response state.

    Returns
    -------
    rates : Float64[Array, "C U V E"]
        Nonnegative expected count rate in every native detector bin.

    Raises
    ------
    ValueError
        If density shape or a mode-specific coefficient length is invalid.
    EquinoxRuntimeError
        If a numerical input or result is non-finite or outside its domain.

    Notes
    -----
    The returned array contains differentiable expected rates. Integer
    acquisition remains an explicit downstream operation.
    """
    density: Float64[Array, "C U V E"] = _validated_detector_density(
        detector_density, calibration
    )
    background: Float64[Array, "C U V E"] = background_density(
        density, calibration, effects
    )
    sensitivity: Float64[Array, "U V E"] = sensitivity_field(
        calibration, effects
    )
    volumes: Float64[Array, "U V E"] = detector_bin_volumes(calibration)
    rates: Float64[Array, "C U V E"] = (
        effects.exposure
        * sensitivity[None, ...]
        * (density + background)
        * volumes[None, ...]
    )
    rates = eqx.error_if(
        rates,
        ~jnp.all(jnp.isfinite(rates)) | ~jnp.all(rates >= 0.0),
        "expected detector counts must be finite and nonnegative",
    )
    final_rates: Float64[Array, "C U V E"] = apply_post_count_response(
        rates, effects
    )
    return final_rates


@jaxtyped(typechecker=beartype)
def fixed_total_probabilities(  # noqa: DOC503
    rates: Float64[Array, "..."],
) -> Float64[Array, "..."]:
    """Normalize all detector rates to one event-probability tensor.

    The operation treats every array entry as one category in a single
    fixed-total acquisition.

    :see: :class:`~.test_effects.TestFixedTotalProbabilities`

    Parameters
    ----------
    rates : Float64[Array, "..."]
        Finite nonnegative rates with a positive global sum.

    Returns
    -------
    probabilities : Float64[Array, "..."]
        Same-shaped probabilities whose global sum is one.

    Raises
    ------
    ValueError
        If rates are scalar or empty.
    EquinoxRuntimeError
        If a rate is non-finite or negative, or their global sum is not
        positive.

    Notes
    -----
    Global normalization preserves the tensor shape and removes the overall
    exposure scale from multinomial probabilities.
    """
    rate_array: Float64[Array, "..."] = jnp.asarray(rates, dtype=jnp.float64)
    if rate_array.ndim < 1 or rate_array.size < 1:
        raise ValueError("fixed-total rates must be a nonempty array")
    rate_array = eqx.error_if(
        rate_array,
        ~jnp.all(jnp.isfinite(rate_array)) | ~jnp.all(rate_array >= 0.0),
        "fixed-total rates must be finite and nonnegative",
    )
    total_rate: Float64[Array, ""] = jnp.sum(rate_array)
    total_rate = eqx.error_if(
        total_rate,
        ~(total_rate > 0.0),
        "fixed-total rates must have a positive sum",
    )
    probabilities: Float64[Array, "..."] = rate_array / total_rate
    return probabilities


@jaxtyped(typechecker=beartype)
def sample_poisson_counts(  # noqa: DOC503
    key: PRNGKeyArray,
    rates: Float64[Array, "..."],
) -> Int[Array, "..."]:
    """Generate independent Poisson counts for a rate tensor.

    The sampler maps each expected rate to an independent integer variate
    using one explicit JAX key.

    :see: :class:`~.test_effects.TestSamplePoissonCounts`

    Parameters
    ----------
    key : PRNGKeyArray
        Explicit JAX random key.
    rates : Float64[Array, "..."]
        Finite nonnegative Poisson means.

    Returns
    -------
    counts : Int[Array, "..."]
        Integer sample with the same shape as ``rates``.

    Raises
    ------
    ValueError
        If rates are scalar or empty.
    EquinoxRuntimeError
        If a rate is non-finite or negative.

    Notes
    -----
    Integer draws are intentionally outside the differentiable graph.
    """
    rate_array: Float64[Array, "..."] = jnp.asarray(rates, dtype=jnp.float64)
    if rate_array.ndim < 1 or rate_array.size < 1:
        raise ValueError("Poisson rates must be a nonempty array")
    rate_array = eqx.error_if(
        rate_array,
        ~jnp.all(jnp.isfinite(rate_array)) | ~jnp.all(rate_array >= 0.0),
        "Poisson rates must be finite and nonnegative",
    )
    counts: Int[Array, "..."] = jax.random.poisson(
        key, rate_array, dtype=jnp.int64
    )
    return counts


@jaxtyped(typechecker=beartype)
def sample_fixed_total_counts(  # noqa: DOC503
    key: PRNGKeyArray,
    rates: Float64[Array, "..."],
    total_count: int,
) -> Int[Array, "..."]:
    """Generate one fixed-total multinomial count tensor.

    The sampler normalizes all rates and returns one integer realization with
    an exact declared total.

    :see: :class:`~.test_effects.TestSampleFixedTotalCounts`

    Parameters
    ----------
    key : PRNGKeyArray
        Explicit JAX random key.
    rates : Float64[Array, "..."]
        Finite nonnegative rates with positive global sum.
    total_count : int
        Positive static number of acquired events.

    Returns
    -------
    counts : Int[Array, "..."]
        One multinomial count tensor summing exactly to ``total_count``.

    Raises
    ------
    ValueError
        If ``total_count`` is not a positive integer or rates are empty.
    EquinoxRuntimeError
        If rates are non-finite, negative, or have a nonpositive sum.

    Notes
    -----
    This is one multinomial draw over all bins, not independent Poisson
    sampling. Integer draws are intentionally outside the derivative graph.
    """
    if type(total_count) is not int or total_count <= 0:
        raise ValueError("total_count must be a positive integer")
    probabilities: Float64[Array, "..."] = fixed_total_probabilities(rates)
    flat_counts: Int[Array, " N"] = jax.random.multinomial(
        key,
        total_count,
        probabilities.reshape((-1,)),
        dtype=jnp.float64,
    ).astype(jnp.int64)
    counts: Int[Array, "..."] = flat_counts.reshape(probabilities.shape)
    return counts
