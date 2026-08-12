"""Apply finite-volume detector resolution.

Extended Summary
----------------
This module provides sampled parity stencils.
It also provides analytic finite-volume Gaussian resolution.

Routine Listings
----------------
:func:`apply_resolution`
    Apply analytic finite-volume resolution in native detector coordinates.
:func:`convolve_energy`
    Convolve a uniform energy axis with the sampled parity stencil.
:func:`convolve_kpath`
    Convolve physical-k path-cell densities with analytic boundary loss.
:func:`convolve_momentum_map`
    Convolve a uniform Cartesian momentum map with sampled stencils.
:func:`gaussian_kernel_1d`
    Build a sampled, sum-normalized Gaussian stencil.
"""

import math

import equinox as eqx
import jax
import jax.numpy as jnp
from beartype import beartype
from beartype.typing import Tuple
from jaxtyping import Array, Bool, Float64, jaxtyped

from diffpes.types import DetectorCalibration, ScalarFloat

from .detector_response import detector_bin_volumes


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


@jaxtyped(typechecker=beartype)
def gaussian_kernel_1d(  # noqa: DOC502, DOC503
    sigma_over_dx: ScalarFloat,
    half_width: int = 48,
) -> Float64[Array, " n_taps"]:
    """Build a sampled, sum-normalized Gaussian stencil.

    The static support and zero-padding convention match SciPy's
    ``gaussian_filter1d(mode="constant", radius=half_width)`` approximation.

    :see: :class:`~.test_resolution.TestGaussianKernel1D`

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

    :see: :class:`~.test_resolution.TestConvolveEnergy`

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

    :see: :class:`~.test_resolution.TestConvolveMomentumMap`

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

    :see: :class:`~.test_resolution.TestConvolveKPath`

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

    :see: :class:`~.test_resolution.TestApplyResolution`

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


__all__: list[str] = [
    "apply_resolution",
    "convolve_energy",
    "convolve_kpath",
    "convolve_momentum_map",
    "gaussian_kernel_1d",
]
