"""PRIVATE: Compute detector-map geometry and finite-volume quadrature.

Extended Summary
----------------
This private module owns shared rotations and Jacobians.
It also owns interpolation and flux accounting.
"""

import equinox as eqx
import jax.numpy as jnp
from beartype.typing import Tuple, Union
from jaxtyping import Array, Bool, Float64, Int32

from diffpes.types import (
    ArpesCube,
    ArpesSpectrum,
    DetectorCalibration,
    ExperimentGeometry,
)

from .polarization import sample_azimuth_rotation


def _gauss_legendre_rule(
    order: int,
) -> Tuple[Float64[Array, " q"], Float64[Array, " q"]]:
    """PRIVATE: Return the registered unit-interval quadrature rule.

    Parameters
    ----------
    order : int
        Static quadrature order, exactly four or eight.

    Returns
    -------
    rule : Tuple[Float64[Array, " q"], Float64[Array, " q"]]
        Nodes on ``[-1, 1]`` and their positive weights.

    Raises
    ------
    ValueError
        If ``order`` is neither four nor eight.
    """
    if order == 4:  # noqa: PLR2004 -- registered quadrature order.
        nodes_tuple: Tuple[float, ...] = (
            -0.8611363115940526,
            -0.3399810435848563,
            0.3399810435848563,
            0.8611363115940526,
        )
        weights_tuple: Tuple[float, ...] = (
            0.34785484513745385,
            0.6521451548625461,
            0.6521451548625461,
            0.34785484513745385,
        )
    elif order == 8:  # noqa: PLR2004 -- eight-point comparison order.
        nodes_tuple = (
            -0.9602898564975363,
            -0.7966664774136267,
            -0.525532409916329,
            -0.1834346424956498,
            0.1834346424956498,
            0.525532409916329,
            0.7966664774136267,
            0.9602898564975363,
        )
        weights_tuple = (
            0.10122853629037626,
            0.22238103445337448,
            0.31370664587788727,
            0.362683783378362,
            0.362683783378362,
            0.31370664587788727,
            0.22238103445337448,
            0.10122853629037626,
        )
    else:
        raise ValueError("detector mapping quadrature order must be 4 or 8")
    nodes: Float64[Array, " q"] = jnp.asarray(nodes_tuple, dtype=jnp.float64)
    weights: Float64[Array, " q"] = jnp.asarray(
        weights_tuple, dtype=jnp.float64
    )
    rule: Tuple[Float64[Array, " q"], Float64[Array, " q"]] = (
        nodes,
        weights,
    )
    return rule


def _bin_quadrature(
    edges: Float64[Array, " np1"], order: int
) -> Tuple[Float64[Array, "n q"], Float64[Array, "n q"]]:
    """PRIVATE: Apply one Gauss--Legendre rule to explicit target bins.

    Parameters
    ----------
    edges : Float64[Array, " np1"]
        Strictly increasing target-bin edges.
    order : int
        Static registered quadrature order.

    Returns
    -------
    quadrature : Tuple[Float64[Array, "n q"], Float64[Array, "n q"]]
        Physical nodes and integration weights for every bin.
    """
    reference_nodes: Float64[Array, " q"]
    reference_weights: Float64[Array, " q"]
    reference_nodes, reference_weights = _gauss_legendre_rule(order)
    centres: Float64[Array, " n"] = 0.5 * (edges[:-1] + edges[1:])
    half_widths: Float64[Array, " n"] = 0.5 * (edges[1:] - edges[:-1])
    nodes: Float64[Array, "n q"] = (
        centres[:, None] + half_widths[:, None] * reference_nodes[None, :]
    )
    weights: Float64[Array, "n q"] = (
        half_widths[:, None] * reference_weights[None, :]
    )
    quadrature: Tuple[Float64[Array, "n q"], Float64[Array, "n q"]] = (
        nodes,
        weights,
    )
    return quadrature


def _source_faces(axis: Float64[Array, " n"]) -> Float64[Array, " np1"]:
    """PRIVATE: Construct midpoint and exterior-half-cell source faces.

    Parameters
    ----------
    axis : Float64[Array, " n"]
        Strictly increasing source-centre coordinates with at least two nodes.

    Returns
    -------
    faces : Float64[Array, " np1"]
        Interior midpoint faces and one exterior half spacing at each end.
    """
    interior: Float64[Array, " nm1"] = 0.5 * (axis[:-1] + axis[1:])
    lower: Float64[Array, " 1"] = jnp.asarray(
        [axis[0] - 0.5 * (axis[1] - axis[0])]
    )
    upper: Float64[Array, " 1"] = jnp.asarray(
        [axis[-1] + 0.5 * (axis[-1] - axis[-2])]
    )
    faces: Float64[Array, " np1"] = jnp.concatenate((lower, interior, upper))
    return faces


def _source_cell_widths(axis: Float64[Array, " n"]) -> Float64[Array, " n"]:
    """PRIVATE: Return finite-volume widths implied by source centres.

    Parameters
    ----------
    axis : Float64[Array, " n"]
        Strictly increasing source-centre coordinates.

    Returns
    -------
    widths : Float64[Array, " n"]
        Differences of the midpoint/exterior faces.
    """
    widths: Float64[Array, " n"] = jnp.diff(_source_faces(axis))
    return widths


def _clamped_bracket(
    axis: Float64[Array, " n"], query: Float64[Array, " ..."]
) -> Tuple[
    Int32[Array, " ..."],
    Int32[Array, " ..."],
    Float64[Array, " ..."],
    Bool[Array, " ..."],
]:
    """PRIVATE: Find a clamped-linear bracket and support mask.

    Parameters
    ----------
    axis : Float64[Array, " n"]
        Strictly increasing source-centre axis.
    query : Float64[Array, " ..."]
        Arbitrarily shaped query coordinates.

    Returns
    -------
    bracket : Tuple
        Lower and upper integer indices, upper interpolation weight, and the
        closed exterior-face support mask.

    Notes
    -----
    Queries between an endpoint centre and its exterior face clamp to the
    endpoint value.  The support mask excludes queries strictly beyond a face.
    """
    faces: Float64[Array, " np1"] = _source_faces(axis)
    in_domain: Bool[Array, " ..."] = (query >= faces[0]) & (query <= faces[-1])
    clamped: Float64[Array, " ..."] = jnp.clip(query, axis[0], axis[-1])
    upper: Int32[Array, " ..."] = jnp.clip(
        jnp.searchsorted(axis, clamped, side="right"), 1, axis.size - 1
    )
    lower: Int32[Array, " ..."] = upper - 1
    denominator: Float64[Array, " ..."] = axis[upper] - axis[lower]
    safe_denominator: Float64[Array, " ..."] = jnp.where(
        denominator > 0.0, denominator, 1.0
    )
    weight: Float64[Array, " ..."] = (clamped - axis[lower]) / safe_denominator
    weight = jnp.where(denominator > 0.0, weight, 0.0)
    bracket: Tuple[
        Int32[Array, " ..."],
        Int32[Array, " ..."],
        Float64[Array, " ..."],
        Bool[Array, " ..."],
    ] = (lower, upper, weight, in_domain)
    return bracket


def _interpolate_cube(
    source: ArpesCube,
    kx_query: Float64[Array, " ..."],
    ky_query: Float64[Array, " ..."],
    energy_query: Float64[Array, " ..."],
) -> Float64[Array, " ..."]:
    """PRIVATE: Evaluate a cube with clamped trilinear reconstruction.

    Parameters
    ----------
    source : ArpesCube
        Validated source-coordinate cube.
    kx_query : Float64[Array, " ..."]
        Source ``kx`` coordinates.
    ky_query : Float64[Array, " ..."]
        Source ``ky`` coordinates.
    energy_query : Float64[Array, " ..."]
        Source relative-energy coordinates, broadcast-compatible with the
        momentum coordinates.

    Returns
    -------
    values : Float64[Array, " ..."]
        Reconstructed density, exactly zero outside any exterior face.
    """
    ix0: Int32[Array, " ..."]
    ix1: Int32[Array, " ..."]
    tx: Float64[Array, " ..."]
    valid_x: Bool[Array, " ..."]
    ix0, ix1, tx, valid_x = _clamped_bracket(source.kx_axis, kx_query)
    iy0: Int32[Array, " ..."]
    iy1: Int32[Array, " ..."]
    ty: Float64[Array, " ..."]
    valid_y: Bool[Array, " ..."]
    iy0, iy1, ty, valid_y = _clamped_bracket(source.ky_axis, ky_query)
    ie0: Int32[Array, " ..."]
    ie1: Int32[Array, " ..."]
    te: Float64[Array, " ..."]
    valid_e: Bool[Array, " ..."]
    ie0, ie1, te, valid_e = _clamped_bracket(source.energy_axis, energy_query)

    lower_energy: Float64[Array, " ..."] = (1.0 - tx) * (
        (1.0 - ty) * source.intensity[ix0, iy0, ie0]
        + ty * source.intensity[ix0, iy1, ie0]
    ) + tx * (
        (1.0 - ty) * source.intensity[ix1, iy0, ie0]
        + ty * source.intensity[ix1, iy1, ie0]
    )
    upper_energy: Float64[Array, " ..."] = (1.0 - tx) * (
        (1.0 - ty) * source.intensity[ix0, iy0, ie1]
        + ty * source.intensity[ix0, iy1, ie1]
    ) + tx * (
        (1.0 - ty) * source.intensity[ix1, iy0, ie1]
        + ty * source.intensity[ix1, iy1, ie1]
    )
    interpolated: Float64[Array, " ..."] = (
        1.0 - te
    ) * lower_energy + te * upper_energy
    values: Float64[Array, " ..."] = jnp.where(
        valid_x & valid_y & valid_e, interpolated, 0.0
    )
    return values


def _interpolate_spectrum(
    source: ArpesSpectrum,
    path_query: Float64[Array, " ..."],
    energy_query: Float64[Array, " ..."],
) -> Float64[Array, " ..."]:
    """PRIVATE: Evaluate a slit spectrum with clamped bilinear reconstruction.

    Parameters
    ----------
    source : ArpesSpectrum
        Validated source line density.
    path_query : Float64[Array, " ..."]
        Source path coordinates.
    energy_query : Float64[Array, " ..."]
        Source relative-energy coordinates, broadcast-compatible with the
        path coordinates.

    Returns
    -------
    values : Float64[Array, " ..."]
        Reconstructed line density, zero beyond either exterior face.
    """
    ik0: Int32[Array, " ..."]
    ik1: Int32[Array, " ..."]
    tk: Float64[Array, " ..."]
    valid_k: Bool[Array, " ..."]
    ik0, ik1, tk, valid_k = _clamped_bracket(source.k_axis, path_query)
    ie0: Int32[Array, " ..."]
    ie1: Int32[Array, " ..."]
    te: Float64[Array, " ..."]
    valid_e: Bool[Array, " ..."]
    ie0, ie1, te, valid_e = _clamped_bracket(source.energy_axis, energy_query)
    lower_energy: Float64[Array, " ..."] = (1.0 - tk) * source.intensity[
        ik0, ie0
    ] + tk * source.intensity[ik1, ie0]
    upper_energy: Float64[Array, " ..."] = (1.0 - tk) * source.intensity[
        ik0, ie1
    ] + tk * source.intensity[ik1, ie1]
    interpolated: Float64[Array, " ..."] = (
        1.0 - te
    ) * lower_energy + te * upper_energy
    values: Float64[Array, " ..."] = jnp.where(
        valid_k & valid_e, interpolated, 0.0
    )
    return values


def _active_euler_zyz(
    angles_rad: Float64[Array, " 3"],
) -> Float64[Array, "3 3"]:
    """PRIVATE: Build an active z--y--z Cartesian Euler rotation.

    Parameters
    ----------
    angles_rad : Float64[Array, " 3"]
        Active ``(alpha, beta, gamma)`` angles in radians.

    Returns
    -------
    rotation : Float64[Array, "3 3"]
        Matrix ``Rz(alpha) @ Ry(beta) @ Rz(gamma)`` for column vectors.
    """
    alpha: Float64[Array, ""] = angles_rad[0]
    beta: Float64[Array, ""] = angles_rad[1]
    gamma: Float64[Array, ""] = angles_rad[2]
    ca: Float64[Array, ""] = jnp.cos(alpha)
    sa: Float64[Array, ""] = jnp.sin(alpha)
    cb: Float64[Array, ""] = jnp.cos(beta)
    sb: Float64[Array, ""] = jnp.sin(beta)
    cg: Float64[Array, ""] = jnp.cos(gamma)
    sg: Float64[Array, ""] = jnp.sin(gamma)
    zero: Float64[Array, ""] = jnp.zeros_like(alpha)
    one: Float64[Array, ""] = jnp.ones_like(alpha)
    rotation_z_alpha: Float64[Array, "3 3"] = jnp.stack(
        (
            jnp.stack((ca, -sa, zero)),
            jnp.stack((sa, ca, zero)),
            jnp.stack((zero, zero, one)),
        )
    )
    rotation_y_beta: Float64[Array, "3 3"] = jnp.stack(
        (
            jnp.stack((cb, zero, sb)),
            jnp.stack((zero, one, zero)),
            jnp.stack((-sb, zero, cb)),
        )
    )
    rotation_z_gamma: Float64[Array, "3 3"] = jnp.stack(
        (
            jnp.stack((cg, -sg, zero)),
            jnp.stack((sg, cg, zero)),
            jnp.stack((zero, zero, one)),
        )
    )
    rotation: Float64[Array, "3 3"] = (
        rotation_z_alpha @ rotation_y_beta @ rotation_z_gamma
    )
    return rotation


def _source_to_lab_rotation(
    geometry: ExperimentGeometry,
    euler_angles_rad: Float64[Array, " 3"],
) -> Float64[Array, "3 3"]:
    """PRIVATE: Compose domain and sample-to-laboratory active rotations.

    Parameters
    ----------
    geometry : ExperimentGeometry
        Experiment carrying the traced sample azimuth.
    euler_angles_rad : Float64[Array, " 3"]
        Active source-domain z--y--z rotation.

    Returns
    -------
    rotation : Float64[Array, "3 3"]
        Domain-source to laboratory rotation.
    """
    domain_rotation: Float64[Array, "3 3"] = _active_euler_zyz(
        euler_angles_rad
    )
    sample_rotation: Float64[Array, "3 3"] = sample_azimuth_rotation(
        geometry.sample_azimuth
    )
    rotation: Float64[Array, "3 3"] = sample_rotation @ domain_rotation
    return rotation


def _detector_forward_jacobian(
    u: Float64[Array, " ..."],
    v: Float64[Array, " ..."],
    momentum: Float64[Array, " ..."],
    slit: str,
) -> Float64[Array, "... 2 2"]:
    """PRIVATE: Evaluate the analytic detector momentum Jacobian.

    Parameters
    ----------
    u : Float64[Array, " ..."]
        Native detector ``u`` angles in radians.
    v : Float64[Array, " ..."]
        Native detector ``v`` angles in radians.
    momentum : Float64[Array, " ..."]
        Positive photoelectron momentum magnitude in inverse angstroms.
    slit : str
        Static detector slit orientation.

    Returns
    -------
    jacobian : Float64[Array, "... 2 2"]
        Analytic ``d(kx, ky) / d(u, v)`` in the laboratory frame.

    Raises
    ------
    ValueError
        The slit selector accepts only ``"H"`` or ``"V"``.
    """
    cosine_u: Float64[Array, " ..."] = jnp.cos(u)
    sine_u: Float64[Array, " ..."] = jnp.sin(u)
    cosine_v: Float64[Array, " ..."] = jnp.cos(v)
    sine_v: Float64[Array, " ..."] = jnp.sin(v)
    zero: Float64[Array, " ..."] = jnp.zeros_like(momentum)
    if slit == "H":
        first_row: Float64[Array, "... 2"] = jnp.stack(
            (momentum * cosine_u, zero), axis=-1
        )
        second_row: Float64[Array, "... 2"] = jnp.stack(
            (
                momentum * sine_u * sine_v,
                -momentum * cosine_u * cosine_v,
            ),
            axis=-1,
        )
    elif slit == "V":
        first_row = jnp.stack((zero, momentum * cosine_v), axis=-1)
        second_row = jnp.stack(
            (
                -momentum * cosine_u * cosine_v,
                momentum * sine_u * sine_v,
            ),
            axis=-1,
        )
    else:
        raise ValueError("detector mapping slit must be 'H' or 'V'")
    jacobian: Float64[Array, "... 2 2"] = jnp.stack(
        (first_row, second_row), axis=-2
    )
    return jacobian


def _inverse_map_abs_jacobian(
    u: Float64[Array, " ..."],
    v: Float64[Array, " ..."],
    momentum: Float64[Array, " ..."],
    slit: str,
) -> Float64[Array, " ..."]:
    """PRIVATE: Return the analytic detector-to-momentum volume factor.

    Parameters
    ----------
    u : Float64[Array, " ..."]
        Native detector ``u`` angles in radians.
    v : Float64[Array, " ..."]
        Native detector ``v`` angles in radians.
    momentum : Float64[Array, " ..."]
        Positive photoelectron momentum magnitude.
    slit : str
        Static detector slit orientation.

    Returns
    -------
    absolute_jacobian : Float64[Array, " ..."]
        ``abs(det(d(kx, ky) / d(u, v)))``.

    Raises
    ------
    ValueError
        The slit selector accepts only ``"H"`` or ``"V"``.
    """
    if slit == "H":
        absolute_jacobian: Float64[Array, " ..."] = (
            jnp.square(momentum) * jnp.square(jnp.cos(u)) * jnp.cos(v)
        )
    elif slit == "V":
        absolute_jacobian = (
            jnp.square(momentum) * jnp.cos(u) * jnp.square(jnp.cos(v))
        )
    else:
        raise ValueError("detector mapping slit must be 'H' or 'V'")
    return absolute_jacobian


def _analytic_angle_jacobian(
    u: Float64[Array, ""],
    v: Float64[Array, ""],
    momentum: Float64[Array, ""],
    slit: str,
) -> Float64[Array, "2 2"]:
    """PRIVATE: Compute the analytic inverse detector Jacobian.

    Parameters
    ----------
    u : Float64[Array, ""]
        Native detector ``u`` angle at one point on the open chart.
    v : Float64[Array, ""]
        Native detector ``v`` angle at one point on the open chart.
    momentum : Float64[Array, ""]
        Positive photoelectron momentum magnitude.
    slit : str
        Static detector slit orientation.

    Returns
    -------
    inverse : Float64[Array, "2 2"]
        Analytic ``d(u, v) / d(kx, ky)``.
    """
    forward: Float64[Array, "2 2"] = _detector_forward_jacobian(
        u, v, momentum, slit
    )
    determinant: Float64[Array, ""] = (
        forward[0, 0] * forward[1, 1] - forward[0, 1] * forward[1, 0]
    )
    safe_determinant: Float64[Array, ""] = jnp.where(
        jnp.abs(determinant) > 0.0, determinant, 1.0
    )
    inverse: Float64[Array, "2 2"] = (
        jnp.stack(
            (
                jnp.stack((forward[1, 1], -forward[0, 1])),
                jnp.stack((-forward[1, 0], forward[0, 0])),
            )
        )
        / safe_determinant
    )
    return inverse


def _validate_common(  # noqa: DOC503 -- eqx.error_if raises at runtime.
    source: Union[ArpesCube, ArpesSpectrum],
    geometry: ExperimentGeometry,
    calibration: DetectorCalibration,
    euler_angles_rad: Float64[Array, " 3"],
) -> Float64[Array, " 3"]:
    """PRIVATE: Validate static mapping metadata and traced coordinates.

    Parameters
    ----------
    source : Union[ArpesCube, ArpesSpectrum]
        Self-describing source carrier.
    geometry : ExperimentGeometry
        Experiment geometry selecting the detector slit.
    calibration : DetectorCalibration
        Explicit native detector target.
    euler_angles_rad : Float64[Array, " 3"]
        Active source-domain rotation.

    Returns
    -------
    validated_angles : Float64[Array, " 3"]
        Rotation angles carrying compiled finite/chart checks.

    Raises
    ------
    ValueError
        Static metadata, dimensions, or source type violates the v1 contract.
    """
    if not isinstance(source, (ArpesCube, ArpesSpectrum)):
        raise ValueError(
            "detector mapping requires ArpesCube or ArpesSpectrum"
        )
    if source.cartesian_frame_id != "org.diffpes.frame.sample_cartesian":
        raise ValueError(
            "detector mapping requires the registered source frame"
        )
    if calibration.coordinate_system != "hemispherical_angles":
        raise ValueError("detector mapping requires hemispherical angles")
    if calibration.boundary_policy != "loss":
        raise ValueError(
            "detector mapping supports only boundary_policy='loss'"
        )
    if geometry.slit not in {"H", "V"}:
        raise ValueError("detector mapping slit must be 'H' or 'V'")
    if euler_angles_rad.shape != (3,):
        raise ValueError("domain Euler angles must have shape (3,)")
    if isinstance(source, ArpesCube):
        if calibration.v_bin_edges.shape[0] <= 2:  # noqa: PLR2004
            raise ValueError(
                "ArpesCube mapping requires an active v target axis"
            )
        if (
            min(
                source.kx_axis.shape[0],
                source.ky_axis.shape[0],
                source.energy_axis.shape[0],
            )
            < 2  # noqa: PLR2004 -- linear interpolation requires two nodes.
        ):
            raise ValueError("cube mapping source axes require two centres")
    else:
        if calibration.v_bin_edges.shape[0] != 2:  # noqa: PLR2004
            raise ValueError("ArpesSpectrum mapping requires one v aperture")
        if (
            min(source.k_axis.shape[0], source.energy_axis.shape[0]) < 2  # noqa: PLR2004 -- linear interpolation requires two nodes.
        ):
            raise ValueError(
                "spectrum mapping source axes require two centres"
            )

    validated_angles: Float64[Array, " 3"] = eqx.error_if(
        euler_angles_rad,
        ~jnp.all(jnp.isfinite(euler_angles_rad)),
        "detector mapping Euler angles must be finite",
    )
    target_chart_valid: Bool[Array, ""] = (
        (calibration.u_bin_edges[0] > -0.5 * jnp.pi)
        & (calibration.u_bin_edges[-1] < 0.5 * jnp.pi)
        & (calibration.v_bin_edges[0] > -0.5 * jnp.pi)
        & (calibration.v_bin_edges[-1] < 0.5 * jnp.pi)
    )
    validated_angles = eqx.error_if(
        validated_angles,
        ~target_chart_valid,
        "detector target edges must lie inside the open principal chart",
    )
    minimum_kinetic_energy: Float64[Array, ""] = (
        geometry.photon_energy_ev
        - geometry.work_function_ev
        + calibration.energy_bin_edges_ev[0]
    )
    result: Float64[Array, " 3"] = eqx.error_if(
        validated_angles,
        ~(minimum_kinetic_energy > 0.0),
        (
            "detector mapping requires positive kinetic energy in every "
            "target bin"
        ),
    )
    return result


def _source_flux(
    source: Union[ArpesCube, ArpesSpectrum],
) -> Float64[Array, ""]:
    """PRIVATE: Integrate one clamped-linear source over its full support.

    Parameters
    ----------
    source : Union[ArpesCube, ArpesSpectrum]
        Source density carrier.

    Returns
    -------
    flux : Float64[Array, ""]
        Full source mass in its declared coordinate measure.
    """
    energy_widths: Float64[Array, " e"] = _source_cell_widths(
        source.energy_axis
    )
    if isinstance(source, ArpesCube):
        kx_widths: Float64[Array, " x"] = _source_cell_widths(source.kx_axis)
        ky_widths: Float64[Array, " y"] = _source_cell_widths(source.ky_axis)
        flux: Float64[Array, ""] = jnp.einsum(
            "x,y,e,xye->",
            kx_widths,
            ky_widths,
            energy_widths,
            source.intensity,
        )
    else:
        path_widths: Float64[Array, " k"] = _source_cell_widths(source.k_axis)
        flux = jnp.einsum(
            "k,e,ke->", path_widths, energy_widths, source.intensity
        )
    return flux


def _target_volumes(
    calibration: DetectorCalibration,
) -> Float64[Array, "u v e"]:
    """PRIVATE: Compute explicit native target-bin volumes.

    Parameters
    ----------
    calibration : DetectorCalibration
        Explicit target-bin edges.

    Returns
    -------
    volumes : Float64[Array, "u v e"]
        Products of native ``u``, ``v``, and energy widths.
    """
    volumes: Float64[Array, "u v e"] = (
        jnp.diff(calibration.u_bin_edges)[:, None, None]
        * jnp.diff(calibration.v_bin_edges)[None, :, None]
        * jnp.diff(calibration.energy_bin_edges_ev)[None, None, :]
    )
    return volumes


def _segmented_quadrature(
    boundaries: Float64[Array, "... n_boundary"],
    lower: Float64[Array, " ..."],
    upper: Float64[Array, " ..."],
    order: int,
) -> Tuple[
    Float64[Array, "... n_segment q"],
    Float64[Array, "... n_segment q"],
]:
    """PRIVATE: Integrate smooth pieces between declared seam coordinates.

    Parameters
    ----------
    boundaries : Float64[Array, "... n_boundary"]
        Sorted seam coordinates with one more boundary than segment.
    lower : Float64[Array, " ..."]
        Lower edge of the target interval.
    upper : Float64[Array, " ..."]
        Upper edge of the target interval.
    order : int
        Static registered quadrature order.

    Returns
    -------
    quadrature : Tuple
        Nodes and weights for every clipped smooth segment.

    Notes
    -----
    Empty intersections retain static shape with exactly zero weights.
    """
    reference_nodes: Float64[Array, " q"]
    reference_weights: Float64[Array, " q"]
    reference_nodes, reference_weights = _gauss_legendre_rule(order)
    segment_lower: Float64[Array, "... n_segment"] = jnp.maximum(
        boundaries[..., :-1], lower[..., None]
    )
    segment_upper: Float64[Array, "... n_segment"] = jnp.minimum(
        boundaries[..., 1:], upper[..., None]
    )
    widths: Float64[Array, "... n_segment"] = jnp.maximum(
        segment_upper - segment_lower, 0.0
    )
    centres: Float64[Array, "... n_segment"] = 0.5 * (
        segment_lower + segment_upper
    )
    half_widths: Float64[Array, "... n_segment"] = 0.5 * widths
    nodes: Float64[Array, "... n_segment q"] = (
        centres[..., None] + half_widths[..., None] * reference_nodes
    )
    weights: Float64[Array, "... n_segment q"] = (
        half_widths[..., None] * reference_weights
    )
    quadrature: Tuple[
        Float64[Array, "... n_segment q"],
        Float64[Array, "... n_segment q"],
    ] = (nodes, weights)
    return quadrature


def _endpoint_seams(
    axis: Float64[Array, " n"],
) -> Float64[Array, " 4"]:
    """PRIVATE: Return support faces and endpoint-clamp seams.

    Parameters
    ----------
    axis : Float64[Array, " n"]
        Strictly increasing source-centre coordinates.

    Returns
    -------
    seams : Float64[Array, " 4"]
        Lower face, endpoint centres, and upper face.

    Notes
    -----
    Clamped reconstruction changes formula at both endpoint centres.
    """
    faces: Float64[Array, " np1"] = _source_faces(axis)
    seams: Float64[Array, " 4"] = jnp.stack(
        (faces[0], axis[0], axis[-1], faces[-1])
    )
    return seams


def _captured_fraction(
    detector_density: Float64[Array, "u v e"],
    source_flux: Float64[Array, ""],
    calibration: DetectorCalibration,
) -> Float64[Array, ""]:
    """PRIVATE: Compare captured detector mass with full source mass.

    Parameters
    ----------
    detector_density : Float64[Array, "u v e"]
        Mapped native-coordinate density.
    source_flux : Float64[Array, ""]
        Full source mass before aperture loss.
    calibration : DetectorCalibration
        Explicit target volumes.

    Returns
    -------
    fraction : Float64[Array, ""]
        Captured mass fraction.  The zero-source convention is zero.
    """
    captured_flux: Float64[Array, ""] = jnp.sum(
        detector_density * _target_volumes(calibration)
    )
    positive_source: Bool[Array, ""] = source_flux > 0.0
    safe_source: Float64[Array, ""] = jnp.where(
        positive_source, source_flux, 1.0
    )
    fraction: Float64[Array, ""] = jnp.where(
        positive_source, captured_flux / safe_source, 0.0
    )
    return fraction


__all__: list[str] = []
