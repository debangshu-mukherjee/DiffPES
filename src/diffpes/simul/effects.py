r"""Apply calibrated instrument effects and assemble expected counts.

Extended Summary
----------------
This module implements native-coordinate Gaussian resolution, the fixed-domain
analyser-transmission model, and the WP8.8 detector-effects foundation. The
canonical ordering is true-kinetic-energy transmission, finite-volume detector
resolution, background, sensitivity, bin-volume conversion, exposure, and an
optional calibrated post-count response. Sampling takes explicit JAX PRNG keys
and remains outside the differentiable expected-rate graph.

The physical resolution operator integrates piecewise-constant bin densities
against a continuous Gaussian. It converts calibrated FWHM values through
``sigma = FWHM / (2 * sqrt(2 * log(2)))`` and reports boundary loss. The
sampled uniform-grid helpers are separate SciPy/Chinook parity approximations;
they use zero padding and a static, sum-normalized stencil.

Routine Listings
----------------
:func:`apply_resolution`
    Apply analytic finite-volume resolution in native detector coordinates.
:func:`apply_post_count_response`
    Convolve expected counts along the recorded-energy index.
:func:`apply_detector_effects`
    Apply the complete deterministic source-to-count detector chain.
:func:`apply_transmission`
    Apply analyser transmission to intensity at true kinetic energy.
:func:`background_density`
    Evaluate a nonnegative detector-coordinate background.
:func:`convolve_energy`
    Convolve a uniform energy axis with the sampled parity stencil.
:func:`convolve_kpath`
    Convolve physical-k path-cell densities with analytic boundary loss.
:func:`convolve_momentum_map`
    Convolve a uniform Cartesian momentum map with sampled stencils.
:func:`detector_bin_volumes`
    Compute explicit native detector-bin volumes.
:func:`expected_counts`
    Assemble deterministic expected detector counts.
:func:`fixed_total_probabilities`
    Normalize all detector rates to one event-probability tensor.
:func:`gaussian_kernel_1d`
    Build a sampled, sum-normalized Gaussian stencil.
:func:`map_source_to_detector`
    Convert one source density to native detector bins conservatively.
:func:`sample_fixed_total_counts`
    Generate one fixed-total multinomial count tensor.
:func:`sample_poisson_counts`
    Generate independent Poisson counts for a rate tensor.
:func:`sensitivity_field`
    Evaluate the positive normalized detector sensitivity field.
:func:`transmission_shape`
    Evaluate positive monotone analyser transmission with fixed mean one.
"""

import math

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from beartype import beartype
from beartype.typing import Tuple, Union
from jaxtyping import Array, Bool, Float64, Int, PRNGKeyArray, jaxtyped

from diffpes.types import (
    ArpesCube,
    ArpesSpectrum,
    DetectorCalibration,
    DetectorEffects,
    DetectorRaster,
    ExperimentGeometry,
    ScalarFloat,
    make_detector_raster,
)

from ._detector_map import _map_and_mix_domains, _map_source_to_detector

__all__: list[str] = [
    "apply_detector_effects",
    "apply_resolution",
    "apply_post_count_response",
    "apply_transmission",
    "background_density",
    "convolve_energy",
    "convolve_kpath",
    "convolve_momentum_map",
    "detector_bin_volumes",
    "expected_counts",
    "fixed_total_probabilities",
    "gaussian_kernel_1d",
    "map_source_to_detector",
    "sample_fixed_total_counts",
    "sample_poisson_counts",
    "sensitivity_field",
    "transmission_shape",
]


def _validate_half_width(half_width: int) -> None:
    """PRIVATE: Validate one static sampled-kernel half-width.

    Parameters
    ----------
    half_width : int
        Requested number of taps on either side of the kernel centre.

    Raises
    ------
    ValueError
        If the support is not a positive integer.
    """
    if isinstance(half_width, bool) or half_width < 1:
        raise ValueError("half_width must be a positive static integer")


def _validate_uniform_axis(  # noqa: DOC503
    axis: Float64[Array, " N"],
    *,
    name: str,
) -> Float64[Array, ""]:
    """PRIVATE: Validate one increasing uniform coordinate axis.

    Parameters
    ----------
    axis : Float64[Array, " N"]
        Candidate one-dimensional coordinate axis.
    name : str
        Name used in traced diagnostics.

    Returns
    -------
    spacing : Float64[Array, ""]
        Positive common grid spacing.

    Raises
    ------
    ValueError
        If the axis is not one-dimensional or has fewer than two points.
    EquinoxRuntimeError
        If the axis is non-finite, non-increasing, or nonuniform.
    """
    if axis.ndim != 1 or axis.shape[0] < 2:  # noqa: PLR2004
        raise ValueError(f"{name} must be one-dimensional with two points")
    differences: Float64[Array, " Nm1"] = jnp.diff(axis)
    spacing: Float64[Array, ""] = differences[0]
    tolerance: Float64[Array, ""] = 1.0e-14 + 1.0e-12 * jnp.maximum(
        1.0, jnp.abs(spacing)
    )
    invalid: Bool[Array, ""] = (
        ~jnp.all(jnp.isfinite(axis))
        | ~(spacing > 0.0)
        | jnp.any(jnp.abs(differences - spacing) > tolerance)
    )
    checked_spacing: Float64[Array, ""] = eqx.error_if(
        spacing, invalid, f"{name} must be finite, increasing, and uniform"
    )
    return checked_spacing


def _sampled_convolve_axis(
    values: Float64[Array, "..."],
    kernel: Float64[Array, " K"],
    *,
    axis: int,
    half_width: int,
) -> Float64[Array, "..."]:
    """PRIVATE: Convolve one array axis with a zero-padded stencil.

    Parameters
    ----------
    values : Float64[Array, "..."]
        Array carrying the axis to convolve.
    kernel : Float64[Array, " K"]
        Odd, sum-normalized correlation stencil.
    axis : int
        Axis index to move to the convolution dimension.
    half_width : int
        Static number of zero-padded elements on either side.

    Returns
    -------
    convolved : Float64[Array, "..."]
        Array with the original shape and axis order.
    """
    moved: Float64[Array, "..."] = jnp.moveaxis(values, axis, -1)
    length: int = moved.shape[-1]
    flattened: Float64[Array, "batch N"] = moved.reshape((-1, length))
    lhs: Float64[Array, "batch N 1"] = flattened[..., None]
    rhs: Float64[Array, "K 1 1"] = kernel[:, None, None]
    correlated: Float64[Array, "batch N 1"] = jax.lax.conv_general_dilated(
        lhs,
        rhs,
        window_strides=(1,),
        padding=((half_width, half_width),),
        dimension_numbers=("NWC", "WIO", "NWC"),
    )
    restored: Float64[Array, "..."] = correlated[..., 0].reshape(moved.shape)
    convolved: Float64[Array, "..."] = jnp.moveaxis(restored, -1, axis)
    return convolved


def _gaussian_integral_antiderivative(
    displacement: Float64[Array, "..."],
    sigma: Float64[Array, ""],
) -> Float64[Array, "..."]:
    r"""PRIVATE: Evaluate a Gaussian second-antiderivative.

    Parameters
    ----------
    displacement : Float64[Array, "..."]
        Edge-to-edge coordinate displacements.
    sigma : Float64[Array, ""]
        Strictly positive Gaussian standard deviation.

    Returns
    -------
    antiderivative : Float64[Array, "..."]
        Values of :math:`z\Phi(z/\sigma)+\sigma\phi(z/\sigma)`.
    """
    scaled: Float64[Array, "..."] = displacement / sigma
    cdf: Float64[Array, "..."] = 0.5 * (
        1.0 + jax.scipy.special.erf(scaled / math.sqrt(2.0))
    )
    density: Float64[Array, "..."] = jnp.exp(
        -0.5 * jnp.square(scaled)
    ) / math.sqrt(2.0 * math.pi)
    antiderivative: Float64[Array, "..."] = (
        displacement * cdf + sigma * density
    )
    return antiderivative


def _finite_volume_gaussian_matrix(  # noqa: DOC502, DOC503
    edges: Float64[Array, " Np1"],
    sigma: Float64[Array, ""],
    *,
    name: str,
) -> Float64[Array, "N N"]:
    """PRIVATE: Build the analytic density-to-density Gaussian matrix.

    Parameters
    ----------
    edges : Float64[Array, " Np1"]
        Explicit increasing source and target cell edges.
    sigma : Float64[Array, ""]
        Gaussian standard deviation in the edge coordinate.
    name : str
        Axis name used in traced diagnostics.

    Returns
    -------
    matrix : Float64[Array, "N N"]
        Finite-volume matrix. Rows are not normalized at boundaries.

    Raises
    ------
    EquinoxRuntimeError
        If the width is non-finite or below the registered smooth interior.
    """
    widths: Float64[Array, " N"] = jnp.diff(edges)
    sigma_checked: Float64[Array, ""] = eqx.error_if(
        sigma,
        ~jnp.isfinite(sigma) | ~(sigma >= 1.0e-3 * jnp.min(widths)),
        f"{name} sigma must be finite and at least 1e-3 of the smallest bin",
    )
    left: Float64[Array, " N"] = edges[:-1]
    right: Float64[Array, " N"] = edges[1:]
    integrated: Float64[Array, "N N"] = (
        _gaussian_integral_antiderivative(
            right[:, None] - left[None, :], sigma_checked
        )
        - _gaussian_integral_antiderivative(
            left[:, None] - left[None, :], sigma_checked
        )
        - _gaussian_integral_antiderivative(
            right[:, None] - right[None, :], sigma_checked
        )
        + _gaussian_integral_antiderivative(
            left[:, None] - right[None, :], sigma_checked
        )
    )
    # Roundoff can make analytically zero, far-off-diagonal cells negative by
    # a few ulps. This projection does not renormalize rows or columns.
    integrated = jnp.maximum(integrated, 0.0)
    matrix: Float64[Array, "N N"] = integrated / widths[:, None]
    return matrix


def _apply_finite_volume_axis(
    values: Float64[Array, "..."],
    matrix: Float64[Array, "N N"],
    *,
    axis: int,
) -> Float64[Array, "..."]:
    """PRIVATE: Apply one density-to-density finite-volume matrix.

    Parameters
    ----------
    values : Float64[Array, "..."]
        Input bin-average densities.
    matrix : Float64[Array, "N N"]
        Output-by-source finite-volume matrix.
    axis : int
        Array axis corresponding to the matrix source dimension.

    Returns
    -------
    convolved : Float64[Array, "..."]
        Densities with the same shape and axis order.
    """
    moved: Float64[Array, "..."] = jnp.moveaxis(values, axis, -1)
    transformed: Float64[Array, "..."] = jnp.einsum(
        "ij,...j->...i", matrix, moved
    )
    convolved: Float64[Array, "..."] = jnp.moveaxis(transformed, -1, axis)
    return convolved


def _path_cell_edges(
    centres: Float64[Array, " K"],
) -> Float64[Array, " Kp1"]:
    """PRIVATE: Construct midpoint cells with exterior half-cell faces.

    Parameters
    ----------
    centres : Float64[Array, " K"]
        Strictly increasing path-cell centres.

    Returns
    -------
    edges : Float64[Array, " Kp1"]
        Midpoint interior faces and symmetric exterior half cells.
    """
    interior: Float64[Array, " Km1"] = 0.5 * (centres[:-1] + centres[1:])
    first: Float64[Array, " 1"] = centres[:1] - 0.5 * (
        centres[1:2] - centres[:1]
    )
    last: Float64[Array, " 1"] = centres[-1:] + 0.5 * (
        centres[-1:] - centres[-2:-1]
    )
    edges: Float64[Array, " Kp1"] = jnp.concatenate((first, interior, last))
    return edges


def _integrated_bernstein(
    x: Float64[Array, "..."],
    degree: int,
) -> Float64[Array, "... q"]:
    r"""PRIVATE: Integrate every Bernstein basis function from zero to x.

    Parameters
    ----------
    x : Float64[Array, "..."]
        Normalized calibration-domain coordinate.
    degree : int
        Static Bernstein degree, one or two for the public model.

    Returns
    -------
    integrals : Float64[Array, "... q"]
        Analytic values :math:`\int_0^x B_j^{degree}(t)\,dt`.
    """
    elevated_degree: int = degree + 1
    elevated: list[Float64[Array, "..."]] = [
        math.comb(elevated_degree, index)
        * x**index
        * (1.0 - x) ** (elevated_degree - index)
        for index in range(elevated_degree + 1)
    ]
    integrals: Float64[Array, "... q"] = jnp.stack(
        [
            sum(elevated[index + 1 :], start=jnp.zeros_like(x))
            / elevated_degree
            for index in range(degree + 1)
        ],
        axis=-1,
    )
    return integrals


def _transmission_log_response(
    normalized_energy: Float64[Array, "..."],
    raw_slopes: Float64[Array, " q"],
    sign: int,
) -> Float64[Array, "..."]:
    """PRIVATE: Evaluate the anchored monotone log-transmission polynomial.

    Parameters
    ----------
    normalized_energy : Float64[Array, "..."]
        Calibration-domain coordinate in ``[0, 1]``.
    raw_slopes : Float64[Array, " q"]
        Two or three unconstrained Bernstein derivative coordinates.
    sign : int
        Registered monotonic direction, ``-1`` or ``1``.

    Returns
    -------
    log_response : Float64[Array, "..."]
        Anchored integrated-Bernstein log response.
    """
    basis_integrals: Float64[Array, "... q"] = _integrated_bernstein(
        normalized_energy, raw_slopes.shape[0] - 1
    )
    slopes: Float64[Array, " q"] = jax.nn.softplus(raw_slopes)
    log_response: Float64[Array, "..."] = sign * jnp.sum(
        basis_integrals * slopes, axis=-1
    )
    return log_response


@jaxtyped(typechecker=beartype)
def gaussian_kernel_1d(  # noqa: DOC502, DOC503
    sigma_over_dx: ScalarFloat,
    half_width: int = 48,
) -> Float64[Array, " n_taps"]:
    """Build a sampled, sum-normalized Gaussian stencil.

    The static support and zero-padding convention match SciPy's
    ``gaussian_filter1d(mode="constant", radius=half_width)`` approximation.

    :see: :class:`~.test_effects.TestGaussianKernel1D`

    Parameters
    ----------
    sigma_over_dx : ScalarFloat
        Gaussian sigma in uniform-grid-spacing units.
    half_width : int, optional
        Static number of taps on either side. Default is 48.

    Returns
    -------
    kernel : Float64[Array, " n_taps"]
        Positive odd stencil with sum one.

    Raises
    ------
    ValueError
        If ``half_width`` is not a positive static integer.
    EquinoxRuntimeError
        If sigma is invalid or the declared support omits more than ``1e-15``
        of the continuous Gaussian mass.

    Notes
    -----
    Widths below ``1e-3`` pixels are outside the smooth fitted domain. The
    singular zero-width limit has no derivative claim.
    """
    _validate_half_width(half_width)
    sigma: Float64[Array, ""] = jnp.asarray(sigma_over_dx, dtype=jnp.float64)
    minimum_sigma: float = 1.0e-3
    maximum_tail_mass: float = 1.0e-15
    tail_mass: Float64[Array, ""] = jax.scipy.special.erfc(
        (half_width + 0.5) / (math.sqrt(2.0) * sigma)
    )
    sigma = eqx.error_if(
        sigma,
        ~jnp.isfinite(sigma)
        | ~(sigma >= minimum_sigma)
        | ~jnp.isfinite(tail_mass)
        | (tail_mass > maximum_tail_mass),
        "sampled Gaussian sigma/support violates the registered envelope",
    )
    offsets: Float64[Array, " n_taps"] = jnp.arange(
        -half_width, half_width + 1, dtype=jnp.float64
    )
    unnormalized: Float64[Array, " n_taps"] = jnp.exp(
        -0.5 * jnp.square(offsets / sigma)
    )
    kernel: Float64[Array, " n_taps"] = unnormalized / jnp.sum(unnormalized)
    return kernel


@jaxtyped(typechecker=beartype)
def convolve_energy(  # noqa: DOC503
    intensity: Float64[Array, "... n_e"],
    energy_axis: Float64[Array, " n_e"],
    sigma_e_ev: ScalarFloat,
    half_width: int = 48,
) -> Float64[Array, "... n_e"]:
    """Convolve a uniform energy axis with the sampled parity stencil.

    This helper reproduces SciPy's sampled Gaussian and zero-padding boundary
    convention. Use :func:`apply_resolution` for calibrated physical bins.

    :see: :class:`~.test_effects.TestConvolveEnergy`

    Parameters
    ----------
    intensity : Float64[Array, "... n_e"]
        Values sampled on a uniform trailing energy axis.
    energy_axis : Float64[Array, " n_e"]
        Strictly increasing uniform energy coordinates in eV.
    sigma_e_ev : ScalarFloat
        Gaussian standard deviation in eV.
    half_width : int, optional
        Static stencil half-width. Default is 48.

    Returns
    -------
    convolved : Float64[Array, "... n_e"]
        Zero-padded sampled convolution with the input shape.

    Raises
    ------
    ValueError
        If dimensions disagree or the axis is too short.
    EquinoxRuntimeError
        If the axis, width, support, or input values are invalid.

    Notes
    -----
    The function converts the physical sigma to pixel units once, builds the
    static sampled stencil, and correlates the trailing axis with explicit
    zero padding.
    """
    if intensity.ndim < 1 or intensity.shape[-1] != energy_axis.shape[0]:
        raise ValueError("intensity trailing axis must match energy_axis")
    spacing: Float64[Array, ""] = _validate_uniform_axis(
        energy_axis, name="energy_axis"
    )
    values: Float64[Array, "... n_e"] = eqx.error_if(
        intensity,
        ~jnp.all(jnp.isfinite(intensity)),
        "energy-convolution intensity must be finite",
    )
    sigma: Float64[Array, ""] = jnp.asarray(sigma_e_ev, dtype=jnp.float64)
    kernel: Float64[Array, " n_taps"] = gaussian_kernel_1d(
        sigma / spacing, half_width
    )
    convolved: Float64[Array, "... n_e"] = _sampled_convolve_axis(
        values, kernel, axis=-1, half_width=half_width
    )
    return convolved


@jaxtyped(typechecker=beartype)
def convolve_momentum_map(  # noqa: DOC503
    intensity: Float64[Array, "n_kx n_ky n_e"],
    kx_axis_inv_ang: Float64[Array, " n_kx"],
    ky_axis_inv_ang: Float64[Array, " n_ky"],
    sigma_k_inv_ang: ScalarFloat,
    half_width: int = 48,
) -> Float64[Array, "n_kx n_ky n_e"]:
    """Convolve a uniform Cartesian momentum map with sampled stencils.

    This is an energy-independent physical-k calibration approximation for
    SciPy/Chinook parity, not the canonical angular analyser model. Its two
    explicit axes carry uniform Cartesian-calibration coordinates in inverse
    angstroms. Use :func:`apply_resolution` for native angular FWHM values and
    nonlinear kinematic maps.

    :see: :class:`~.test_effects.TestConvolveMomentumMap`

    Parameters
    ----------
    intensity : Float64[Array, "n_kx n_ky n_e"]
        Cartesian momentum-map samples.
    kx_axis_inv_ang : Float64[Array, " n_kx"]
        Strictly increasing uniform first-axis coordinates in inverse
        angstroms.
    ky_axis_inv_ang : Float64[Array, " n_ky"]
        Strictly increasing uniform second-axis coordinates in inverse
        angstroms.
    sigma_k_inv_ang : ScalarFloat
        Isotropic Gaussian sigma in inverse angstroms.
    half_width : int, optional
        Static stencil half-width on both map axes. Default is 48.

    Returns
    -------
    convolved : Float64[Array, "n_kx n_ky n_e"]
        Two-pass zero-padded sampled convolution.

    Raises
    ------
    ValueError
        If raster dimensions disagree or either map axis is too short.
    EquinoxRuntimeError
        If coordinates, width, support, or intensity are invalid.

    Notes
    -----
    The helper infers only the two uniform mesh increments; it does not infer
    an angular PSF or an energy-dependent momentum Jacobian.
    """
    if intensity.ndim != 3:  # noqa: PLR2004
        raise ValueError("momentum-map intensity must have three dimensions")
    if intensity.shape[:2] != (
        kx_axis_inv_ang.shape[0],
        ky_axis_inv_ang.shape[0],
    ):
        raise ValueError("momentum-map axes must match explicit k axes")
    values: Float64[Array, "n_kx n_ky n_e"] = eqx.error_if(
        intensity,
        ~jnp.all(jnp.isfinite(intensity)),
        "momentum-map intensity must be finite",
    )
    spacing_first: Float64[Array, ""] = _validate_uniform_axis(
        kx_axis_inv_ang, name="kx_axis_inv_ang"
    )
    spacing_second: Float64[Array, ""] = _validate_uniform_axis(
        ky_axis_inv_ang, name="ky_axis_inv_ang"
    )
    sigma: Float64[Array, ""] = jnp.asarray(sigma_k_inv_ang, dtype=jnp.float64)
    kernel_first: Float64[Array, " n_taps"] = gaussian_kernel_1d(
        sigma / spacing_first, half_width
    )
    kernel_second: Float64[Array, " n_taps"] = gaussian_kernel_1d(
        sigma / spacing_second, half_width
    )
    first_pass: Float64[Array, "n_kx n_ky n_e"] = _sampled_convolve_axis(
        values, kernel_first, axis=0, half_width=half_width
    )
    convolved: Float64[Array, "n_kx n_ky n_e"] = _sampled_convolve_axis(
        first_pass, kernel_second, axis=1, half_width=half_width
    )
    return convolved


@jaxtyped(typechecker=beartype)
def convolve_kpath(  # noqa: DOC503
    intensity: Float64[Array, "n_k n_e"],
    k_distances: Float64[Array, " n_k"],
    sigma_k_inv_ang: ScalarFloat,
) -> Tuple[Float64[Array, "n_k n_e"], Float64[Array, ""], Bool[Array, ""]]:
    """Convolve physical-k path-cell densities with analytic boundary loss.

    This O(K-squared) operator is for cut-sized physical-k calibrations only.
    A calibrated angular width must use detector-axis resolution through
    :func:`apply_resolution`.

    :see: :class:`~.test_effects.TestConvolveKPath`

    Parameters
    ----------
    intensity : Float64[Array, "n_k n_e"]
        Nonnegative path-cell-average densities.
    k_distances : Float64[Array, " n_k"]
        Strictly increasing cumulative path centres in inverse angstroms.
    sigma_k_inv_ang : ScalarFloat
        Positive Gaussian standard deviation in inverse angstroms.

    Returns
    -------
    convolved : Float64[Array, "n_k n_e"]
        Finite-volume density without row normalization.
    captured_fraction : Float64[Array, ""]
        In-domain output flux divided by nonzero input flux.
    valid : Bool[Array, ""]
        Whether the input carries nonzero integrated flux.

    Raises
    ------
    ValueError
        If path dimensions disagree or the path contains fewer than two
        centres.
    EquinoxRuntimeError
        If centres, input density, or the width are outside the valid domain.

    Notes
    -----
    Interior faces are adjacent-centre midpoints; exterior faces lie one half
    of the nearest spacing beyond each endpoint. Escaped Gaussian mass is
    absent under the binding ``loss`` boundary policy.
    """
    if intensity.ndim != 2 or k_distances.ndim != 1:  # noqa: PLR2004
        raise ValueError("k-path intensity and centres must be rank two/one")
    if (
        intensity.shape[0] != k_distances.shape[0] or intensity.shape[0] < 2  # noqa: PLR2004
    ):
        raise ValueError(
            "k-path intensity requires at least two matching cells"
        )
    centres: Float64[Array, " n_k"] = eqx.error_if(
        k_distances,
        ~jnp.all(jnp.isfinite(k_distances))
        | ~jnp.all(jnp.diff(k_distances) > 0.0),
        "k-path centres must be finite and strictly increasing",
    )
    values: Float64[Array, "n_k n_e"] = eqx.error_if(
        intensity,
        ~jnp.all(jnp.isfinite(intensity)) | ~jnp.all(intensity >= 0.0),
        "k-path density must be finite and nonnegative",
    )
    edges: Float64[Array, " k_edges"] = _path_cell_edges(centres)
    sigma: Float64[Array, ""] = jnp.asarray(sigma_k_inv_ang, dtype=jnp.float64)
    matrix: Float64[Array, "n_k n_k"] = _finite_volume_gaussian_matrix(
        edges, sigma, name="k-path"
    )
    convolved: Float64[Array, "n_k n_e"] = _apply_finite_volume_axis(
        values, matrix, axis=0
    )
    cell_widths: Float64[Array, " n_k"] = jnp.diff(edges)
    input_flux: Float64[Array, ""] = jnp.sum(values * cell_widths[:, None])
    output_flux: Float64[Array, ""] = jnp.sum(convolved * cell_widths[:, None])
    valid: Bool[Array, ""] = input_flux > 0.0
    safe_input_flux: Float64[Array, ""] = jnp.where(valid, input_flux, 1.0)
    captured_fraction: Float64[Array, ""] = jnp.where(
        valid, output_flux / safe_input_flux, 0.0
    )
    result: Tuple[
        Float64[Array, "n_k n_e"], Float64[Array, ""], Bool[Array, ""]
    ] = (convolved, captured_fraction, valid)
    return result


@jaxtyped(typechecker=beartype)
def apply_resolution(  # noqa: DOC503
    intensity_detector: Float64[Array, "... n_e"],
    calibration: DetectorCalibration,
) -> Tuple[
    Float64[Array, "... n_e"],
    Float64[Array, " 3"],
    Bool[Array, ""],
]:
    """Apply analytic finite-volume resolution in native detector coordinates.

    The last three input axes are native ``(u, v, E)`` bins. Leading axes,
    such as detector channels, pass through independently. The returned three
    captured fractions are the sequential ``u``, ``v``, and energy flux
    retentions; their product equals the total retained-flux fraction.

    :see: :class:`~.test_effects.TestApplyResolution`

    Parameters
    ----------
    intensity_detector : Float64[Array, "... n_e"]
        Nonnegative detector-bin-average density with trailing ``(u, v, E)``
        dimensions fixed by ``calibration``.
    calibration : DetectorCalibration
        Explicit native bin edges and positive FWHM widths.

    Returns
    -------
    blurred : Float64[Array, "... n_e"]
        Native-coordinate finite-volume density with the input shape.
    captured_fractions : Float64[Array, " 3"]
        Sequential captured-flux fractions in ``(u, v, E)`` order.
    valid : Bool[Array, ""]
        Whether the input carries nonzero integrated flux.

    Raises
    ------
    ValueError
        If the input lacks the three native axes or bin counts disagree.
    EquinoxRuntimeError
        If density or a calibrated width is outside the valid domain.

    Notes
    -----
    The operator uses analytic Gaussian double integrals over every source and
    target bin. Preserve boundary loss without row normalization, reflection,
    replication, or wrapping.
    """
    expected_shape: Tuple[int, int, int] = (
        calibration.u_bin_edges.shape[0] - 1,
        calibration.v_bin_edges.shape[0] - 1,
        calibration.energy_bin_edges_ev.shape[0] - 1,
    )
    if (
        intensity_detector.ndim < 3  # noqa: PLR2004
        or intensity_detector.shape[-3:] != expected_shape
        or intensity_detector.size == 0
    ):
        raise ValueError(
            "detector density trailing axes must match calibration (u, v, E)"
        )
    density: Float64[Array, "... n_e"] = eqx.error_if(
        intensity_detector,
        ~jnp.all(jnp.isfinite(intensity_detector))
        | ~jnp.all(intensity_detector >= 0.0),
        "detector density must be finite and nonnegative",
    )
    fwhm_to_sigma: float = 1.0 / (2.0 * math.sqrt(2.0 * math.log(2.0)))
    sigma_u: Float64[Array, ""] = calibration.psf_fwhm_u * fwhm_to_sigma
    sigma_v: Float64[Array, ""] = calibration.psf_fwhm_v * fwhm_to_sigma
    sigma_energy: Float64[Array, ""] = (
        calibration.psf_fwhm_energy_ev * fwhm_to_sigma
    )
    matrix_u: Float64[Array, "U U"] = _finite_volume_gaussian_matrix(
        calibration.u_bin_edges, sigma_u, name="detector-u"
    )
    matrix_v: Float64[Array, "V V"] = _finite_volume_gaussian_matrix(
        calibration.v_bin_edges, sigma_v, name="detector-v"
    )
    matrix_energy: Float64[Array, "E E"] = _finite_volume_gaussian_matrix(
        calibration.energy_bin_edges_ev, sigma_energy, name="detector-energy"
    )
    volumes: Float64[Array, "U V E"] = detector_bin_volumes(calibration)
    leading_shape: Tuple[int, ...] = (1,) * (density.ndim - 3)
    broadcast_volumes: Float64[Array, "..."] = volumes.reshape(
        (*leading_shape, *volumes.shape)
    )

    def integrated_flux(
        candidate: Float64[Array, "..."],
    ) -> Float64[Array, ""]:
        flux: Float64[Array, ""] = jnp.sum(candidate * broadcast_volumes)
        return flux

    initial_flux: Float64[Array, ""] = integrated_flux(density)
    after_u: Float64[Array, "... n_e"] = _apply_finite_volume_axis(
        density, matrix_u, axis=-3
    )
    flux_u: Float64[Array, ""] = integrated_flux(after_u)
    after_v: Float64[Array, "... n_e"] = _apply_finite_volume_axis(
        after_u, matrix_v, axis=-2
    )
    flux_v: Float64[Array, ""] = integrated_flux(after_v)
    blurred: Float64[Array, "... n_e"] = _apply_finite_volume_axis(
        after_v, matrix_energy, axis=-1
    )
    flux_energy: Float64[Array, ""] = integrated_flux(blurred)
    valid: Bool[Array, ""] = initial_flux > 0.0
    safe_initial: Float64[Array, ""] = jnp.where(valid, initial_flux, 1.0)
    safe_flux_u: Float64[Array, ""] = jnp.where(flux_u > 0.0, flux_u, 1.0)
    safe_flux_v: Float64[Array, ""] = jnp.where(flux_v > 0.0, flux_v, 1.0)
    captured_fractions: Float64[Array, " 3"] = jnp.where(
        valid,
        jnp.stack(
            (
                flux_u / safe_initial,
                flux_v / safe_flux_u,
                flux_energy / safe_flux_v,
            )
        ),
        jnp.zeros(3, dtype=jnp.float64),
    )
    result: Tuple[
        Float64[Array, "... n_e"], Float64[Array, " 3"], Bool[Array, ""]
    ] = (blurred, captured_fractions, valid)
    return result


@jaxtyped(typechecker=beartype)
def transmission_shape(  # noqa: DOC503
    kinetic_energy_axis_ev: Float64[Array, " n_e"],
    raw_slopes: Float64[Array, " q"],
    calibration: DetectorCalibration,
) -> Float64[Array, " n_e"]:
    """Evaluate positive monotone analyser transmission with fixed mean one.

    Two or three softplus slope coordinates weight a Bernstein basis for the
    derivative of log transmission. The calibration fixes its sign. A
    64-point Gauss--Legendre mean over the complete calibration domain removes
    its constant mode. The caller's query window never controls normalization.

    :see: :class:`~.test_effects.TestTransmissionShape`

    Parameters
    ----------
    kinetic_energy_axis_ev : Float64[Array, " n_e"]
        True kinetic energies inside the fixed calibration domain in eV.
    raw_slopes : Float64[Array, " q"]
        Exactly two or three unconstrained log-slope coordinates.
    calibration : DetectorCalibration
        Fixed domain and registered monotonic direction.

    Returns
    -------
    transmission : Float64[Array, " n_e"]
        Positive normalized analyser response at every query energy.

    Raises
    ------
    ValueError
        If either numerical input is not one-dimensional or ``q`` is invalid.
    EquinoxRuntimeError
        If values are non-finite, outside the domain, or overflow the model.

    Notes
    -----
    Cropping, padding, or rebinning the query cannot change the affine basis or
    normalization. Version one provides no extrapolation mode.
    """
    if kinetic_energy_axis_ev.ndim != 1 or raw_slopes.ndim != 1:
        raise ValueError(
            "transmission energy and slopes must be one-dimensional"
        )
    if raw_slopes.shape[0] not in (2, 3):  # noqa: PLR2004
        raise ValueError("transmission requires exactly two or three slopes")
    domain: Float64[Array, " 2"] = calibration.transmission_reference_domain_ev
    energies: Float64[Array, " n_e"] = eqx.error_if(
        kinetic_energy_axis_ev,
        ~jnp.all(jnp.isfinite(kinetic_energy_axis_ev))
        | jnp.any(kinetic_energy_axis_ev < domain[0])
        | jnp.any(kinetic_energy_axis_ev > domain[1]),
        "transmission queries must stay inside the calibration domain",
    )
    slopes: Float64[Array, " q"] = eqx.error_if(
        raw_slopes,
        ~jnp.all(jnp.isfinite(raw_slopes)),
        "transmission slopes must be finite",
    )
    span: Float64[Array, ""] = domain[1] - domain[0]
    normalized_query: Float64[Array, " n_e"] = (energies - domain[0]) / span
    query_log_response: Float64[Array, " n_e"] = _transmission_log_response(
        normalized_query, slopes, calibration.transmission_monotonic_sign
    )
    gauss_nodes: Float64[np.ndarray, " n_quad"]
    gauss_weights: Float64[np.ndarray, " n_quad"]
    gauss_nodes, gauss_weights = np.polynomial.legendre.leggauss(64)
    quadrature_nodes: Float64[Array, " n_quad"] = jnp.asarray(
        0.5 * (gauss_nodes + 1.0), dtype=jnp.float64
    )
    quadrature_weights: Float64[Array, " n_quad"] = jnp.asarray(
        0.5 * gauss_weights, dtype=jnp.float64
    )
    quadrature_log_response: Float64[Array, " n_quad"] = (
        _transmission_log_response(
            quadrature_nodes, slopes, calibration.transmission_monotonic_sign
        )
    )
    denominator: Float64[Array, ""] = jnp.sum(
        quadrature_weights * jnp.exp(quadrature_log_response)
    )
    transmission: Float64[Array, " n_e"] = (
        jnp.exp(query_log_response) / denominator
    )
    validated_transmission: Float64[Array, " n_e"] = eqx.error_if(
        transmission,
        ~jnp.all(jnp.isfinite(transmission))
        | ~jnp.all(transmission > 0.0)
        | ~jnp.isfinite(denominator)
        | ~(denominator > 0.0),
        "transmission response and normalization must be finite and positive",
    )
    return validated_transmission


@jaxtyped(typechecker=beartype)
def apply_transmission(  # noqa: DOC503
    intensity: Float64[Array, "... n_e"],
    kinetic_energy_axis_ev: Float64[Array, " n_e"],
    raw_slopes: Float64[Array, " q"],
    calibration: DetectorCalibration,
) -> Float64[Array, "... n_e"]:
    """Apply analyser transmission to intensity at true kinetic energy.

    The fixed-domain shape broadcasts across every leading intensity axis.

    :see: :class:`~.test_effects.TestApplyTransmission`

    Parameters
    ----------
    intensity : Float64[Array, "... n_e"]
        Finite pre-resolution physical intensity.
    kinetic_energy_axis_ev : Float64[Array, " n_e"]
        True kinetic-energy coordinates for the trailing intensity axis.
    raw_slopes : Float64[Array, " q"]
        Two or three unconstrained analyser shape coordinates.
    calibration : DetectorCalibration
        Fixed transmission calibration domain and monotonic sign.

    Returns
    -------
    transmitted : Float64[Array, "... n_e"]
        Intensity multiplied along its trailing energy axis.

    Raises
    ------
    ValueError
        If the intensity and kinetic-energy dimensions disagree.
    EquinoxRuntimeError
        If the intensity or transmission inputs are invalid.

    Notes
    -----
    The canonical chain calls this function before detector resolution. It
    therefore evaluates throughput at true, not recorded, kinetic energy.
    """
    if (
        intensity.ndim < 1
        or intensity.shape[-1] != kinetic_energy_axis_ev.shape[0]
    ):
        raise ValueError("intensity trailing axis must match kinetic energy")
    values: Float64[Array, "... n_e"] = eqx.error_if(
        intensity,
        ~jnp.all(jnp.isfinite(intensity)),
        "transmission input intensity must be finite",
    )
    shape: Float64[Array, " n_e"] = transmission_shape(
        kinetic_energy_axis_ev, raw_slopes, calibration
    )
    transmitted: Float64[Array, "... n_e"] = values * shape
    return transmitted


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
def map_source_to_detector(  # noqa: DOC502, DOC503
    source: Union[ArpesCube, ArpesSpectrum],
    geometry: ExperimentGeometry,
    calibration: DetectorCalibration,
) -> Tuple[Float64[Array, "u v e"], Float64[Array, ""]]:
    """Convert one source density to native detector bins conservatively.

    The source carries its complete Cartesian axes and registered sample
    frame. The calibration independently owns every native target edge. The
    returned density uses the declared per-native-volume convention, while
    the scalar reports the fraction of source flux captured under the
    calibrated ``loss`` boundary policy.

    :see: :class:`~.test_effects.TestMapSourceToDetector`

    Parameters
    ----------
    source : Union[ArpesCube, ArpesSpectrum]
        Self-describing source-coordinate physical intensity.
    geometry : ExperimentGeometry
        Traced sample and photoemission geometry.
    calibration : DetectorCalibration
        Explicit native detector target and boundary convention.

    Returns
    -------
    density : Float64[Array, "u v e"]
        Native detector density before transmission and resolution.
    captured_fraction : Float64[Array, ""]
        Captured source-flux fraction under the ``loss`` policy.

    Raises
    ------
    ValueError
        If the source carrier, registered frame, target dimensionality, or
        slit/map contract is invalid.
    EquinoxRuntimeError
        If a traced geometry, source, or calibration value leaves the valid
        detector chart.

    Notes
    -----
    Production uses four Gauss--Legendre nodes on every active target axis.
    The mapper performs no target inference, row normalization, reflection,
    or source-axis relabeling. Signed diagonal and antidiagonal domain maps
    may cross source-support boundaries: those branches split quadrature at
    every support and exterior-face seam before integration. General domain
    rotations instead require a conservative interval enclosure. Every
    inverse-mapped target bin must lie strictly inside the source exterior
    faces. Eager and compiled calls reject rotations whose enclosures touch or
    cross that boundary. Coordinate derivatives for the general branch
    therefore claim only the smooth, strictly enclosed interior chart, never
    a support crossing or topology switch.

    An :class:`~diffpes.types.ArpesSpectrum` is already a line density along
    its declared path, integrated over exactly one transverse ``v`` aperture.
    Its cumulative path coordinate must be strictly increasing and its full
    Cartesian path must remain inside that aperture. The slit mapper applies
    the absolute path-to-detector Jacobian exactly once. It also divides by the
    declared aperture width once. It never promotes a cut into an inferred 2-D
    source.
    """
    mapped: Tuple[Float64[Array, "u v e"], Float64[Array, ""]] = (
        _map_source_to_detector(source, geometry, calibration)
    )
    return mapped


@jaxtyped(typechecker=beartype)
def apply_detector_effects(  # noqa: DOC502, DOC503
    physical_by_domain: Tuple[Union[ArpesCube, ArpesSpectrum], ...],
    geometry: ExperimentGeometry,
    calibration: DetectorCalibration,
    effects: DetectorEffects,
) -> DetectorRaster:
    """Apply the complete deterministic source-to-count detector chain.

    The chain actively rotates and conservatively maps each source domain
    before traced softmax mixing on common detector bins. The mixed density
    then passes through true-kinetic-energy transmission, native-coordinate
    finite-volume resolution, background, sensitivity, exposure, explicit
    bin-volume conversion, and the optional calibrated post-count response.

    :see: :class:`~.test_effects.TestApplyDetectorEffects`

    Parameters
    ----------
    physical_by_domain : Tuple[Union[ArpesCube, ArpesSpectrum], ...]
        Nonempty static tuple of self-describing source densities.
    geometry : ExperimentGeometry
        Traced photoemission and sample geometry.
    calibration : DetectorCalibration
        Explicit native detector target, PSF, and transmission domain.
    effects : DetectorEffects
        Domain, analyser, background, sensitivity, and acquisition state.

    Returns
    -------
    raster : DetectorRaster
        Single-channel native-coordinate expected counts.

    Raises
    ------
    ValueError
        If domains, frames, source carriers, or target dimensions disagree.
    EquinoxRuntimeError
        If a traced physical or detector coordinate is invalid.

    Notes
    -----
    Integer acquisition and display normalization are intentionally separate.
    Evaluate transmission at true kinetic energy before the recorded-bin PSF.
    Apply background and sensitivity only after that resolution. Domain
    mapping inherits :func:`map_source_to_detector`'s complete coordinate
    contract. Signed diagonal and antidiagonal maps split support seams.
    General rotations require strict enclosure and claim smooth-interior
    derivatives only. Slit spectra retain single-aperture line-density
    semantics.
    """
    mixed_density: Float64[Array, "u v e"]
    mixed_density, _ = _map_and_mix_domains(
        physical_by_domain, geometry, calibration, effects
    )
    recorded_energy: Float64[Array, " e"] = 0.5 * (
        calibration.energy_bin_edges_ev[:-1]
        + calibration.energy_bin_edges_ev[1:]
    )
    kinetic_energy: Float64[Array, " e"] = (
        geometry.photon_energy_ev - geometry.work_function_ev + recorded_energy
    )
    transmitted: Float64[Array, "u v e"] = apply_transmission(
        mixed_density,
        kinetic_energy,
        effects.transmission_raw_slopes,
        calibration,
    )
    resolved: Float64[Array, "u v e"] = apply_resolution(
        transmitted, calibration
    )[0]
    rates: Float64[Array, "1 u v e"] = expected_counts(
        resolved[None, ...], calibration, effects
    )
    detector_u: Float64[Array, " u"] = 0.5 * (
        calibration.u_bin_edges[:-1] + calibration.u_bin_edges[1:]
    )
    detector_v: Float64[Array, " v"] = 0.5 * (
        calibration.v_bin_edges[:-1] + calibration.v_bin_edges[1:]
    )
    raster: DetectorRaster = make_detector_raster(
        rates,
        detector_u,
        detector_v,
        recorded_energy,
        channel_labels=("intensity",),
        coordinate_system=calibration.coordinate_system,
    )
    return raster


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
