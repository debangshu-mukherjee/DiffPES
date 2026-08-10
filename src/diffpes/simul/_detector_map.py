r"""Compute conservative source-to-detector finite-volume maps.

Extended Summary
----------------
This private module implements the coordinate-density seam used by the
Plan-08a detector chain.  It maps a self-describing source carrier into the
explicit bins of :class:`~diffpes.types.DetectorCalibration`, using four
Gauss--Legendre nodes per active target axis.  Clamped-linear source
interpolation spans exterior half-cells and vanishes beyond their faces.
No source array supplies target coordinates.

Cartesian source domains first undergo an active z--y--z Euler rotation and
then the active sample-to-laboratory azimuth rotation.  The exact Plan-03
detector map supplies laboratory parallel momentum.  Its analytic inverse-map
Jacobian converts source density to native angular density.

An :class:`~diffpes.types.ArpesSpectrum` represents a line density already
integrated over the declared transverse slit aperture.  Accept it only with
one native ``v`` bin.  Require strict monotonicity across its rotated forward
``u(s)`` domain.  Keep the entire path inside the declared ``v`` aperture.
Divide the returned density by the aperture width.  Later native-bin-volume
multiplication then recovers the source line flux.

Notes
-----
The public wrappers live in :mod:`diffpes.simul.effects`.  This isolation
allows independent finite-volume tests during complete detector-chain
assembly.  Production always uses four quadrature points.  The
order-selecting helper supports only the binding four-versus-eight convergence
gate.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
from beartype.typing import Optional, Tuple, Union
from jaxtyping import Array, Bool, Float64

from diffpes.simul.kinematics import (
    detector_angles_to_kpar,
    final_state_k_inv_ang,
    kpar_to_detector_angles,
)
from diffpes.simul.polarization import sample_azimuth_rotation
from diffpes.types import (
    ArpesCube,
    ArpesSpectrum,
    DetectorCalibration,
    DetectorEffects,
    ExperimentGeometry,
)

__all__: list[str] = []


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
    elif order == 8:  # noqa: PLR2004 -- convergence-gate quadrature order.
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
) -> Tuple[Array, Array, Float64[Array, " ..."], Bool[Array, " ..."]]:
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
    upper: Array = jnp.clip(
        jnp.searchsorted(axis, clamped, side="right"), 1, axis.size - 1
    )
    lower: Array = upper - 1
    denominator: Float64[Array, " ..."] = axis[upper] - axis[lower]
    safe_denominator: Float64[Array, " ..."] = jnp.where(
        denominator > 0.0, denominator, 1.0
    )
    weight: Float64[Array, " ..."] = (clamped - axis[lower]) / safe_denominator
    weight = jnp.where(denominator > 0.0, weight, 0.0)
    bracket: Tuple[
        Array,
        Array,
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
    ix0: Array
    ix1: Array
    tx: Float64[Array, " ..."]
    valid_x: Bool[Array, " ..."]
    ix0, ix1, tx, valid_x = _clamped_bracket(source.kx_axis, kx_query)
    iy0: Array
    iy1: Array
    ty: Float64[Array, " ..."]
    valid_y: Bool[Array, " ..."]
    iy0, iy1, ty, valid_y = _clamped_bracket(source.ky_axis, ky_query)
    ie0: Array
    ie1: Array
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
    ik0: Array
    ik1: Array
    tk: Float64[Array, " ..."]
    valid_k: Bool[Array, " ..."]
    ik0, ik1, tk, valid_k = _clamped_bracket(source.k_axis, path_query)
    ie0: Array
    ie1: Array
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
    """PRIVATE: Evaluate the analytic Plan-03 detector momentum Jacobian.

    Parameters
    ----------
    u : Float64[Array, " ..."]
        Native detector ``u`` angles in radians.
    v : Float64[Array, " ..."]
        Native detector ``v`` angles in radians.
    momentum : Float64[Array, " ..."]
        Positive photoelectron momentum magnitude in inverse angstroms.
    slit : str
        Static Plan-03 slit orientation.

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
        Static Plan-03 slit orientation.

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
        Static Plan-03 slit orientation.

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


def _validate_general_target_enclosure(  # noqa: DOC502, DOC503
    source: ArpesCube,
    geometry: ExperimentGeometry,
    calibration: DetectorCalibration,
    inverse_projected: Float64[Array, "2 2"],
) -> Float64[Array, "2 2"]:
    """PRIVATE: Require a general rotated target inside source support.

    Parameters
    ----------
    source : ArpesCube
        Source cube with exterior-half-cell support faces.
    geometry : ExperimentGeometry
        Photon energy, work function, and slit orientation.
    calibration : DetectorCalibration
        Complete target angle and energy edges.
    inverse_projected : Float64[Array, "2 2"]
        Laboratory-to-source in-plane map.

    Returns
    -------
    validated_inverse : Float64[Array, "2 2"]
        Input map carrying the compiled enclosure assertion.

    Raises
    ------
    EquinoxRuntimeError
        If the conservative enclosure reaches a source support face.

    Notes
    -----
    Signed interval arithmetic retains offset target and source ranges.
    Conservative products may reject valid oblique targets but admit no face
    crossing.
    """
    energy_faces: Float64[Array, " ep1"] = _source_faces(source.energy_axis)
    target_energy_lower: Float64[Array, ""] = calibration.energy_bin_edges_ev[
        0
    ]
    target_energy_upper: Float64[Array, ""] = calibration.energy_bin_edges_ev[
        -1
    ]
    minimum_kinetic_energy: Float64[Array, ""] = (
        geometry.photon_energy_ev
        - geometry.work_function_ev
        + target_energy_lower
    )
    maximum_kinetic_energy: Float64[Array, ""] = (
        geometry.photon_energy_ev
        - geometry.work_function_ev
        + target_energy_upper
    )
    minimum_momentum: Float64[Array, ""] = final_state_k_inv_ang(
        minimum_kinetic_energy
    )[0]
    maximum_momentum: Float64[Array, ""] = final_state_k_inv_ang(
        maximum_kinetic_energy
    )[0]
    primary_lower: Float64[Array, ""]
    primary_upper: Float64[Array, ""]
    secondary_lower: Float64[Array, ""]
    secondary_upper: Float64[Array, ""]
    if geometry.slit == "H":
        primary_lower = calibration.u_bin_edges[0]
        primary_upper = calibration.u_bin_edges[-1]
        secondary_lower = calibration.v_bin_edges[0]
        secondary_upper = calibration.v_bin_edges[-1]
    else:
        primary_lower = calibration.v_bin_edges[0]
        primary_upper = calibration.v_bin_edges[-1]
        secondary_lower = calibration.u_bin_edges[0]
        secondary_upper = calibration.u_bin_edges[-1]
    primary_sines: Float64[Array, " 2"] = jnp.sin(
        jnp.stack((primary_lower, primary_upper))
    )
    x_products: Float64[Array, " 4"] = jnp.asarray(
        (
            minimum_momentum * primary_sines[0],
            minimum_momentum * primary_sines[1],
            maximum_momentum * primary_sines[0],
            maximum_momentum * primary_sines[1],
        )
    )
    laboratory_x_interval: Float64[Array, " 2"] = jnp.stack(
        (jnp.min(x_products), jnp.max(x_products))
    )
    primary_cosines: Float64[Array, " 2"] = jnp.cos(
        jnp.stack((primary_lower, primary_upper))
    )
    interval_crosses_zero: Bool[Array, ""] = (primary_lower <= 0.0) & (
        primary_upper >= 0.0
    )
    cosine_interval: Float64[Array, " 2"] = jnp.stack(
        (
            jnp.min(primary_cosines),
            jnp.where(interval_crosses_zero, 1.0, jnp.max(primary_cosines)),
        )
    )
    radial_cosine_interval: Float64[Array, " 2"] = jnp.stack(
        (
            minimum_momentum * cosine_interval[0],
            maximum_momentum * cosine_interval[1],
        )
    )
    negative_secondary_sines: Float64[Array, " 2"] = -jnp.sin(
        jnp.stack((secondary_upper, secondary_lower))
    )
    y_products: Float64[Array, " 4"] = jnp.asarray(
        (
            radial_cosine_interval[0] * negative_secondary_sines[0],
            radial_cosine_interval[0] * negative_secondary_sines[1],
            radial_cosine_interval[1] * negative_secondary_sines[0],
            radial_cosine_interval[1] * negative_secondary_sines[1],
        )
    )
    laboratory_y_interval: Float64[Array, " 2"] = jnp.stack(
        (jnp.min(y_products), jnp.max(y_products))
    )
    source_x_products: Float64[Array, "2 2"] = (
        inverse_projected[:, :1] * laboratory_x_interval[None, :]
    )
    source_y_products: Float64[Array, "2 2"] = (
        inverse_projected[:, 1:] * laboratory_y_interval[None, :]
    )
    source_lower: Float64[Array, " 2"] = jnp.min(
        source_x_products, axis=1
    ) + jnp.min(source_y_products, axis=1)
    source_upper: Float64[Array, " 2"] = jnp.max(
        source_x_products, axis=1
    ) + jnp.max(source_y_products, axis=1)
    x_faces: Float64[Array, " xp1"] = _source_faces(source.kx_axis)
    y_faces: Float64[Array, " yp1"] = _source_faces(source.ky_axis)
    enclosed: Bool[Array, ""] = (
        (target_energy_lower > energy_faces[0])
        & (target_energy_upper < energy_faces[-1])
        & (source_lower[0] > x_faces[0])
        & (source_upper[0] < x_faces[-1])
        & (source_lower[1] > y_faces[0])
        & (source_upper[1] < y_faces[-1])
    )
    validated_inverse: Float64[Array, "2 2"] = eqx.error_if(
        inverse_projected,
        ~enclosed,
        (
            "general rotated detector target must lie strictly inside "
            "source exterior faces"
        ),
    )
    return validated_inverse


def _map_cube_general(
    source: ArpesCube,
    geometry: ExperimentGeometry,
    calibration: DetectorCalibration,
    euler_angles_rad: Float64[Array, " 3"],
    order: int,
) -> Tuple[Float64[Array, "u v e"], Float64[Array, ""]]:
    """PRIVATE: Compute the general rotated Cartesian source-cube map.

    Parameters
    ----------
    source : ArpesCube
        Source density on separable sample-frame Cartesian axes.
    geometry : ExperimentGeometry
        Kinematics and sample azimuth.
    calibration : DetectorCalibration
        Explicit native detector target.
    euler_angles_rad : Float64[Array, " 3"]
        Active source-domain z--y--z rotation.
    order : int
        Static quadrature order.

    Returns
    -------
    mapped : Tuple[Float64[Array, "u v e"], Float64[Array, ""]]
        Detector density and captured source-flux fraction.
    """
    source_to_lab: Float64[Array, "3 3"] = _source_to_lab_rotation(
        geometry, euler_angles_rad
    )
    projected: Float64[Array, "2 2"] = source_to_lab[:2, :2]
    determinant: Float64[Array, ""] = (
        projected[0, 0] * projected[1, 1] - projected[0, 1] * projected[1, 0]
    )
    projected = eqx.error_if(
        projected,
        ~jnp.isfinite(determinant) | (jnp.abs(determinant) <= 1.0e-10),  # noqa: PLR2004
        "cube domain rotation has a singular projected in-plane map",
    )
    inverse_projected: Float64[Array, "2 2"] = (
        jnp.stack(
            (
                jnp.stack((projected[1, 1], -projected[0, 1])),
                jnp.stack((-projected[1, 0], projected[0, 0])),
            )
        )
        / determinant
    )
    inverse_projected = _validate_general_target_enclosure(
        source, geometry, calibration, inverse_projected
    )

    u_nodes: Float64[Array, "u q"]
    u_weights: Float64[Array, "u q"]
    u_nodes, u_weights = _bin_quadrature(calibration.u_bin_edges, order)
    v_nodes: Float64[Array, "v q"]
    v_weights: Float64[Array, "v q"]
    v_nodes, v_weights = _bin_quadrature(calibration.v_bin_edges, order)
    energy_nodes: Float64[Array, "e q"]
    energy_weights: Float64[Array, "e q"]
    energy_nodes, energy_weights = _bin_quadrature(
        calibration.energy_bin_edges_ev, order
    )
    n_v: int = v_nodes.shape[0]
    n_e: int = energy_nodes.shape[0]
    flat_bins: Array = jnp.arange(u_nodes.shape[0] * n_v * n_e)

    def integrate_bin(flat_index: Array) -> Float64[Array, ""]:
        """Integrate one explicit native target bin."""
        u_index: Array = flat_index // (n_v * n_e)
        remainder: Array = flat_index % (n_v * n_e)
        v_index: Array = remainder // n_e
        energy_index: Array = remainder % n_e
        u_grid: Float64[Array, "q 1 1"] = u_nodes[u_index, :, None, None]
        v_grid: Float64[Array, "1 q 1"] = v_nodes[v_index, None, :, None]
        energy_grid: Float64[Array, "1 1 q"] = energy_nodes[
            energy_index, None, None, :
        ]
        kinetic_energy: Float64[Array, "1 1 q"] = (
            geometry.photon_energy_ev - geometry.work_function_ev + energy_grid
        )
        lab_k_parallel: Float64[Array, "q q q 2"] = detector_angles_to_kpar(
            u_grid, v_grid, kinetic_energy, geometry.slit
        )
        source_k_parallel: Float64[Array, "q q q 2"] = jnp.einsum(
            "ij,...j->...i", inverse_projected, lab_k_parallel
        )
        momentum: Float64[Array, "1 1 q"] = final_state_k_inv_ang(
            kinetic_energy
        )[0]
        jacobian: Float64[Array, "q q q"] = _inverse_map_abs_jacobian(
            u_grid, v_grid, momentum, geometry.slit
        ) / jnp.abs(determinant)
        source_density: Float64[Array, "q q q"] = _interpolate_cube(
            source,
            source_k_parallel[..., 0],
            source_k_parallel[..., 1],
            energy_grid,
        )
        weight_grid: Float64[Array, "q q q"] = (
            u_weights[u_index, :, None, None]
            * v_weights[v_index, None, :, None]
            * energy_weights[energy_index, None, None, :]
        )
        mass: Float64[Array, ""] = jnp.sum(
            weight_grid * source_density * jacobian
        )
        return mass

    masses: Float64[Array, " b"] = jax.lax.map(
        jax.checkpoint(integrate_bin), flat_bins
    )
    bin_shape: Tuple[int, int, int] = (
        u_nodes.shape[0],
        n_v,
        n_e,
    )
    volumes: Float64[Array, "u v e"] = _target_volumes(calibration)
    detector_density: Float64[Array, "u v e"] = (
        masses.reshape(bin_shape) / volumes
    )
    source_flux: Float64[Array, ""] = _source_flux(source)
    fraction: Float64[Array, ""] = _captured_fraction(
        detector_density, source_flux, calibration
    )
    mapped: Tuple[Float64[Array, "u v e"], Float64[Array, ""]] = (
        detector_density,
        fraction,
    )
    return mapped


def _map_cube_axis_aligned_momentum(  # noqa: PLR0915
    source: ArpesCube,
    geometry: ExperimentGeometry,
    calibration: DetectorCalibration,
    inverse_projected: Float64[Array, "2 2"],
    determinant: Float64[Array, ""],
    primary_source_axis: Float64[Array, " n_primary"],
    secondary_source_axis: Float64[Array, " n_secondary"],
    primary_coefficient: Float64[Array, ""],
    secondary_coefficient: Float64[Array, ""],
    order: int,
) -> Tuple[Float64[Array, "u v e"], Float64[Array, ""]]:
    """PRIVATE: Integrate an aligned map in laboratory momentum coordinates.

    Parameters
    ----------
    source : ArpesCube
        Source density on separable sample-frame Cartesian axes.
    geometry : ExperimentGeometry
        Kinematics and static slit orientation.
    calibration : DetectorCalibration
        Explicit native detector target.
    inverse_projected : Float64[Array, "2 2"]
        Signed-permutation laboratory-to-source map.
    determinant : Float64[Array, ""]
        Projected domain-map determinant.
    primary_source_axis : Float64[Array, " n_primary"]
        Source axis controlled by the detector's first rotation.
    secondary_source_axis : Float64[Array, " n_secondary"]
        Source axis controlled by the detector's second rotation.
    primary_coefficient : Float64[Array, ""]
        Nonzero primary inverse-map coefficient.
    secondary_coefficient : Float64[Array, ""]
        Nonzero secondary inverse-map coefficient.
    order : int
        Static registered quadrature order on each smooth subcell.

    Returns
    -------
    mapped : Tuple[Float64[Array, "u v e"], Float64[Array, ""]]
        Detector density and captured source-flux fraction.

    Notes
    -----
    The coordinate change cancels the native detector Jacobian exactly.
    Static seam intervals retain zero weights when they miss a target bin.
    """
    energy_faces: Float64[Array, " ep1"] = _source_faces(source.energy_axis)
    energy_seams: Float64[Array, " 2"] = jnp.stack(
        (energy_faces[0], energy_faces[-1])
    )
    primary_lab_seams: Float64[Array, " 4"] = jnp.sort(
        _endpoint_seams(primary_source_axis) / primary_coefficient
    )
    secondary_lab_seams: Float64[Array, " 4"] = jnp.sort(
        _endpoint_seams(secondary_source_axis) / secondary_coefficient
    )
    n_u: int = calibration.u_bin_edges.shape[0] - 1
    n_v: int = calibration.v_bin_edges.shape[0] - 1
    n_e: int = calibration.energy_bin_edges_ev.shape[0] - 1
    flat_bins: Array = jnp.arange(n_u * n_v * n_e)

    def integrate_bin(flat_index: Array) -> Float64[Array, ""]:
        """Integrate one native bin between mapped momentum seams."""
        u_index: Array = flat_index // (n_v * n_e)
        remainder: Array = flat_index % (n_v * n_e)
        v_index: Array = remainder // n_e
        energy_index: Array = remainder % n_e
        energy_nodes: Float64[Array, "1 q"]
        energy_weights: Float64[Array, "1 q"]
        energy_nodes, energy_weights = _segmented_quadrature(
            energy_seams,
            calibration.energy_bin_edges_ev[energy_index],
            calibration.energy_bin_edges_ev[energy_index + 1],
            order,
        )
        kinetic_energy: Float64[Array, "1 q"] = (
            geometry.photon_energy_ev
            - geometry.work_function_ev
            + energy_nodes
        )
        momentum: Float64[Array, "1 q"] = final_state_k_inv_ang(
            kinetic_energy
        )[0]
        primary_lower: Float64[Array, ""]
        primary_upper: Float64[Array, ""]
        secondary_lower: Float64[Array, ""]
        secondary_upper: Float64[Array, ""]
        if geometry.slit == "H":
            primary_lower = calibration.u_bin_edges[u_index]
            primary_upper = calibration.u_bin_edges[u_index + 1]
            secondary_lower = calibration.v_bin_edges[v_index]
            secondary_upper = calibration.v_bin_edges[v_index + 1]
        else:
            primary_lower = calibration.v_bin_edges[v_index]
            primary_upper = calibration.v_bin_edges[v_index + 1]
            secondary_lower = calibration.u_bin_edges[u_index]
            secondary_upper = calibration.u_bin_edges[u_index + 1]
        target_x_lower: Float64[Array, "1 q"] = momentum * jnp.sin(
            primary_lower
        )
        target_x_upper: Float64[Array, "1 q"] = momentum * jnp.sin(
            primary_upper
        )
        x_boundaries: Float64[Array, "1 q 4"] = jnp.broadcast_to(
            primary_lab_seams,
            (*momentum.shape, primary_lab_seams.shape[0]),
        )
        x_nodes: Float64[Array, "1 q sx qx"]
        x_weights: Float64[Array, "1 q sx qx"]
        x_nodes, x_weights = _segmented_quadrature(
            x_boundaries, target_x_lower, target_x_upper, order
        )
        transverse_momentum: Float64[Array, "1 q sx qx"] = jnp.sqrt(
            jnp.maximum(
                jnp.square(momentum[..., None, None]) - jnp.square(x_nodes),
                0.0,
            )
        )
        target_y_lower: Float64[Array, "1 q sx qx"] = (
            -transverse_momentum * jnp.sin(secondary_upper)
        )
        target_y_upper: Float64[Array, "1 q sx qx"] = (
            -transverse_momentum * jnp.sin(secondary_lower)
        )
        y_boundaries: Float64[Array, "1 q sx qx 4"] = jnp.broadcast_to(
            secondary_lab_seams,
            (*x_nodes.shape, secondary_lab_seams.shape[0]),
        )
        y_nodes: Float64[Array, "1 q sx qx sy qy"]
        y_weights: Float64[Array, "1 q sx qx sy qy"]
        y_nodes, y_weights = _segmented_quadrature(
            y_boundaries, target_y_lower, target_y_upper, order
        )
        x_grid: Float64[Array, "1 q sx qx 1 1"] = x_nodes[..., None, None]
        lab_k_parallel: Float64[Array, "1 q sx qx sy qy 2"] = jnp.stack(
            (jnp.broadcast_to(x_grid, y_nodes.shape), y_nodes), axis=-1
        )
        source_k_parallel: Float64[Array, "1 q sx qx sy qy 2"] = jnp.einsum(
            "ij,...j->...i", inverse_projected, lab_k_parallel
        )
        energy_grid: Float64[Array, "1 q 1 1 1 1"] = energy_nodes[
            ..., None, None, None, None
        ]
        source_density: Float64[Array, "1 q sx qx sy qy"] = _interpolate_cube(
            source,
            source_k_parallel[..., 0],
            source_k_parallel[..., 1],
            energy_grid,
        )
        weight_grid: Float64[Array, "1 q sx qx sy qy"] = (
            energy_weights[..., None, None, None, None]
            * x_weights[..., None, None]
            * y_weights
        )
        mass: Float64[Array, ""] = jnp.sum(
            weight_grid * source_density / jnp.abs(determinant)
        )
        return mass

    masses: Float64[Array, " b"] = jax.lax.map(
        jax.checkpoint(integrate_bin), flat_bins
    )
    volumes: Float64[Array, "u v e"] = _target_volumes(calibration)
    detector_density: Float64[Array, "u v e"] = (
        masses.reshape((n_u, n_v, n_e)) / volumes
    )
    source_flux: Float64[Array, ""] = _source_flux(source)
    fraction: Float64[Array, ""] = _captured_fraction(
        detector_density, source_flux, calibration
    )
    mapped: Tuple[Float64[Array, "u v e"], Float64[Array, ""]] = (
        detector_density,
        fraction,
    )
    return mapped


def _map_cube(
    source: ArpesCube,
    geometry: ExperimentGeometry,
    calibration: DetectorCalibration,
    euler_angles_rad: Float64[Array, " 3"],
    order: int,
) -> Tuple[Float64[Array, "u v e"], Float64[Array, ""]]:
    """PRIVATE: Compute a boundary-aware conservative source-cube map.

    Parameters
    ----------
    source : ArpesCube
        Source density on separable sample-frame Cartesian axes.
    geometry : ExperimentGeometry
        Kinematics and sample azimuth.
    calibration : DetectorCalibration
        Explicit native detector target.
    euler_angles_rad : Float64[Array, " 3"]
        Active source-domain z--y--z rotation.
    order : int
        Static quadrature order.

    Returns
    -------
    mapped : Tuple[Float64[Array, "u v e"], Float64[Array, ""]]
        Detector density and captured source-flux fraction.

    Notes
    -----
    Signed-permutation rotations use exact support-seam segmentation.
    General rotations require strict interior enclosure before bounded
    cubature.
    """
    source_to_lab: Float64[Array, "3 3"] = _source_to_lab_rotation(
        geometry, euler_angles_rad
    )
    projected: Float64[Array, "2 2"] = source_to_lab[:2, :2]
    determinant: Float64[Array, ""] = (
        projected[0, 0] * projected[1, 1] - projected[0, 1] * projected[1, 0]
    )
    projected = eqx.error_if(
        projected,
        ~jnp.isfinite(determinant) | (jnp.abs(determinant) <= 1.0e-10),  # noqa: PLR2004
        "cube domain rotation has a singular projected in-plane map",
    )
    inverse_projected: Float64[Array, "2 2"] = (
        jnp.stack(
            (
                jnp.stack((projected[1, 1], -projected[0, 1])),
                jnp.stack((-projected[1, 0], projected[0, 0])),
            )
        )
        / determinant
    )
    diagonal: Bool[Array, ""] = (
        (jnp.abs(inverse_projected[0, 1]) <= 1.0e-12)  # noqa: PLR2004
        & (jnp.abs(inverse_projected[1, 0]) <= 1.0e-12)  # noqa: PLR2004
        & (jnp.abs(inverse_projected[0, 0]) > 1.0e-12)  # noqa: PLR2004
        & (jnp.abs(inverse_projected[1, 1]) > 1.0e-12)  # noqa: PLR2004
    )
    antidiagonal: Bool[Array, ""] = (
        (jnp.abs(inverse_projected[0, 0]) <= 1.0e-12)  # noqa: PLR2004
        & (jnp.abs(inverse_projected[1, 1]) <= 1.0e-12)  # noqa: PLR2004
        & (jnp.abs(inverse_projected[1, 0]) > 1.0e-12)  # noqa: PLR2004
        & (jnp.abs(inverse_projected[0, 1]) > 1.0e-12)  # noqa: PLR2004
    )
    mapped: Tuple[Float64[Array, "u v e"], Float64[Array, ""]] = jax.lax.cond(
        diagonal | antidiagonal,
        lambda _: jax.lax.cond(
            diagonal,
            lambda __: _map_cube_axis_aligned_momentum(
                source,
                geometry,
                calibration,
                inverse_projected,
                determinant,
                source.kx_axis,
                source.ky_axis,
                inverse_projected[0, 0],
                inverse_projected[1, 1],
                order,
            ),
            lambda __: _map_cube_axis_aligned_momentum(
                source,
                geometry,
                calibration,
                inverse_projected,
                determinant,
                source.ky_axis,
                source.kx_axis,
                inverse_projected[1, 0],
                inverse_projected[0, 1],
                order,
            ),
            operand=None,
        ),
        lambda _: _map_cube_general(
            source, geometry, calibration, euler_angles_rad, order
        ),
        operand=None,
    )
    return mapped


def _extended_path(
    source: ArpesSpectrum,
) -> Tuple[Float64[Array, " kp2"], Float64[Array, "kp2 3"]]:
    """PRIVATE: Construct a source path through both exterior half-cells.

    Parameters
    ----------
    source : ArpesSpectrum
        Source line density and full Cartesian path.

    Returns
    -------
    extended : Tuple[Float64[Array, " kp2"], Float64[Array, "kp2 3"]]
        Path coordinates and linearly extended Cartesian points at both faces.
    """
    faces: Float64[Array, " kp1"] = _source_faces(source.k_axis)
    lower_scale: Float64[Array, ""] = (faces[0] - source.k_axis[0]) / (
        source.k_axis[1] - source.k_axis[0]
    )
    upper_scale: Float64[Array, ""] = (faces[-1] - source.k_axis[-1]) / (
        source.k_axis[-1] - source.k_axis[-2]
    )
    lower_point: Float64[Array, "3"] = source.kpoints_cart_inv_ang[0] + (
        lower_scale
        * (source.kpoints_cart_inv_ang[1] - source.kpoints_cart_inv_ang[0])
    )
    upper_point: Float64[Array, "3"] = source.kpoints_cart_inv_ang[-1] + (
        upper_scale
        * (source.kpoints_cart_inv_ang[-1] - source.kpoints_cart_inv_ang[-2])
    )
    extended_axis: Float64[Array, " kp2"] = jnp.concatenate(
        (faces[:1], source.k_axis, faces[-1:])
    )
    extended_points: Float64[Array, "kp2 3"] = jnp.concatenate(
        (
            lower_point[None, :],
            source.kpoints_cart_inv_ang,
            upper_point[None, :],
        ),
        axis=0,
    )
    extended: Tuple[Float64[Array, " kp2"], Float64[Array, "kp2 3"]] = (
        extended_axis,
        extended_points,
    )
    return extended


def _path_angle_and_derivative(
    source_point: Float64[Array, " 3"],
    source_tangent: Float64[Array, " 3"],
    source_to_lab: Float64[Array, "3 3"],
    kinetic_energy: Float64[Array, ""],
    slit: str,
) -> Tuple[
    Float64[Array, ""],
    Float64[Array, ""],
    Float64[Array, ""],
]:
    """PRIVATE: Evaluate slit angles and analytic ``du/ds`` on a path.

    Parameters
    ----------
    source_point : Float64[Array, " 3"]
        One sample-frame path point.
    source_tangent : Float64[Array, " 3"]
        Derivative of the sample-frame point with respect to path length.
    source_to_lab : Float64[Array, "3 3"]
        Active domain/sample composition.
    kinetic_energy : Float64[Array, ""]
        Positive photoelectron kinetic energy.
    slit : str
        Static Plan-03 slit orientation.

    Returns
    -------
    result : Tuple[Float64[Array, ""], ...]
        Native ``u``, native ``v``, and the analytic path derivative ``du/ds``.
    """
    lab_point: Float64[Array, "3"] = source_to_lab @ source_point
    lab_tangent: Float64[Array, "3"] = source_to_lab @ source_tangent
    u: Float64[Array, ""]
    v: Float64[Array, ""]
    u, v = kpar_to_detector_angles(lab_point[:2], kinetic_energy, slit)
    momentum: Float64[Array, ""] = final_state_k_inv_ang(kinetic_energy)[0]
    angle_jacobian: Float64[Array, "2 2"] = _analytic_angle_jacobian(
        u, v, momentum, slit
    )
    du_ds: Float64[Array, ""] = jnp.dot(angle_jacobian[0], lab_tangent[:2])
    result: Tuple[
        Float64[Array, ""],
        Float64[Array, ""],
        Float64[Array, ""],
    ] = (u, v, du_ds)
    return result


def _validate_spectrum_chart(  # noqa: DOC502 -- eqx.error_if raises at runtime.
    source: ArpesSpectrum,
    source_to_lab: Float64[Array, "3 3"],
    energy_nodes: Float64[Array, "e q"],
    calibration: DetectorCalibration,
    geometry: ExperimentGeometry,
    order: int,
) -> Float64[Array, "e q"]:
    """PRIVATE: Enforce monotone slit mapping and transverse containment.

    Parameters
    ----------
    source : ArpesSpectrum
        Slit-integrated source line density.
    source_to_lab : Float64[Array, "3 3"]
        Active domain/sample composition.
    energy_nodes : Float64[Array, "e q"]
        Every target-energy quadrature node.
    calibration : DetectorCalibration
        Declared single transverse aperture.
    geometry : ExperimentGeometry
        Kinematics and slit selector.
    order : int
        Static quadrature order used for within-segment validation.

    Returns
    -------
    validated_nodes : Float64[Array, "e q"]
        Input nodes carrying compiled chart checks.

    """
    extended_axis: Float64[Array, " kp2"]
    extended_points: Float64[Array, "kp2 3"]
    extended_axis, extended_points = _extended_path(source)
    reference_nodes: Float64[Array, " q"] = _gauss_legendre_rule(order)[0]
    fractions: Float64[Array, " q"] = 0.5 * (reference_nodes + 1.0)
    segment_points: Float64[Array, "segment q 3"] = (
        extended_points[:-1, None, :]
        + fractions[None, :, None]
        * (extended_points[1:] - extended_points[:-1])[:, None, :]
    )
    segment_tangents: Float64[Array, "segment 3"] = (
        extended_points[1:] - extended_points[:-1]
    ) / (extended_axis[1:] - extended_axis[:-1])[:, None]
    flat_energy: Float64[Array, " n"] = energy_nodes.reshape((-1,))

    def check_energy(
        omega: Float64[Array, ""],
    ) -> Tuple[Bool[Array, ""], Bool[Array, ""]]:
        """Check all path segments at one target-energy node."""
        kinetic_energy: Float64[Array, ""] = (
            geometry.photon_energy_ev - geometry.work_function_ev + omega
        )

        def check_segment(
            data: Tuple[Float64[Array, "q 3"], Float64[Array, " 3"]],
        ) -> Tuple[
            Float64[Array, " q"],
            Float64[Array, " q"],
            Float64[Array, " q"],
        ]:
            """Evaluate chart quantities within one path segment."""
            points: Float64[Array, "q 3"]
            tangent: Float64[Array, " 3"]
            points, tangent = data
            values: Tuple[Array, Array, Array] = jax.vmap(
                lambda point: _path_angle_and_derivative(
                    point,
                    tangent,
                    source_to_lab,
                    kinetic_energy,
                    geometry.slit,
                )
            )(points)
            return values

        u_values: Float64[Array, "segment q"]
        v_values: Float64[Array, "segment q"]
        derivatives: Float64[Array, "segment q"]
        u_values, v_values, derivatives = jax.vmap(check_segment)(
            (segment_points, segment_tangents)
        )
        orientation: Float64[Array, ""] = jnp.sign(
            u_values[-1, -1] - u_values[0, 0]
        )
        monotone: Bool[Array, ""] = (orientation != 0.0) & jnp.all(
            orientation * derivatives > 1.0e-12  # noqa: PLR2004
        )
        inside_aperture: Bool[Array, ""] = jnp.all(
            (v_values >= calibration.v_bin_edges[0])
            & (v_values <= calibration.v_bin_edges[-1])
        )
        checks: Tuple[Bool[Array, ""], Bool[Array, ""]] = (
            monotone,
            inside_aperture,
        )
        return checks

    monotone_checks: Bool[Array, " n"]
    aperture_checks: Bool[Array, " n"]
    monotone_checks, aperture_checks = jax.vmap(check_energy)(flat_energy)
    validated_nodes: Float64[Array, "e q"] = eqx.error_if(
        energy_nodes,
        ~jnp.all(monotone_checks),
        "spectrum forward u(path) must be strictly monotone",
    )
    result: Float64[Array, "e q"] = eqx.error_if(
        validated_nodes,
        ~jnp.all(aperture_checks),
        "spectrum path leaves the declared transverse v aperture",
    )
    return result


def _solve_path_coordinate(
    target_u: Float64[Array, ""],
    omega: Float64[Array, ""],
    source: ArpesSpectrum,
    source_to_lab: Float64[Array, "3 3"],
    geometry: ExperimentGeometry,
) -> Tuple[Float64[Array, ""], Float64[Array, ""], Bool[Array, ""]]:
    """PRIVATE: Resolve a monotone slit path and return ``abs(ds/du)``.

    Parameters
    ----------
    target_u : Float64[Array, ""]
        Native target angle.
    omega : Float64[Array, ""]
        Fermi-relative energy.
    source : ArpesSpectrum
        Slit-integrated source line density.
    source_to_lab : Float64[Array, "3 3"]
        Active domain/sample composition.
    geometry : ExperimentGeometry
        Kinematics and slit selector.

    Returns
    -------
    inverse : Tuple[Float64[Array, ""], Float64[Array, ""], Bool[Array, ""]]
        Source path coordinate, analytic absolute inverse derivative, and
        exterior-face support mask.

    Notes
    -----
    A fixed Newton solve refines the piecewise-linear inverse guess.  Its
    derivative uses the explicit analytic Plan-03 angle Jacobian rather than
    differentiating a dense coordinate map.
    """
    extended_axis: Float64[Array, " kp2"]
    extended_points: Float64[Array, "kp2 3"]
    extended_axis, extended_points = _extended_path(source)
    kinetic_energy: Float64[Array, ""] = (
        geometry.photon_energy_ev - geometry.work_function_ev + omega
    )

    def angles_at_point(
        point: Float64[Array, " 3"],
    ) -> Tuple[Float64[Array, ""], Float64[Array, ""]]:
        """Compute native detector angles for one source point."""
        lab_point: Float64[Array, " 3"] = source_to_lab @ point
        angles: Tuple[Float64[Array, ""], Float64[Array, ""]] = (
            kpar_to_detector_angles(
                lab_point[:2], kinetic_energy, geometry.slit
            )
        )
        return angles

    u_extended: Float64[Array, " kp2"] = jax.vmap(
        lambda point: angles_at_point(point)[0]
    )(extended_points)
    increasing: Bool[Array, ""] = u_extended[-1] > u_extended[0]
    ordered_u: Float64[Array, " kp2"] = jnp.where(
        increasing, u_extended, u_extended[::-1]
    )
    ordered_axis: Float64[Array, " kp2"] = jnp.where(
        increasing, extended_axis, extended_axis[::-1]
    )
    ordered_points: Float64[Array, "kp2 3"] = jnp.where(
        increasing, extended_points, extended_points[::-1]
    )
    in_support: Bool[Array, ""] = (target_u >= ordered_u[0]) & (
        target_u <= ordered_u[-1]
    )
    bounded_u: Float64[Array, ""] = jnp.clip(
        target_u, ordered_u[0], ordered_u[-1]
    )
    upper: Array = jnp.clip(
        jnp.searchsorted(ordered_u, bounded_u, side="right"),
        1,
        ordered_u.size - 1,
    )
    lower: Array = upper - 1
    s0: Float64[Array, ""] = ordered_axis[lower]
    s1: Float64[Array, ""] = ordered_axis[upper]
    point0: Float64[Array, "3"] = ordered_points[lower]
    point1: Float64[Array, "3"] = ordered_points[upper]
    tangent: Float64[Array, "3"] = (point1 - point0) / (s1 - s0)
    initial_fraction: Float64[Array, ""] = (bounded_u - ordered_u[lower]) / (
        ordered_u[upper] - ordered_u[lower]
    )
    initial_s: Float64[Array, ""] = s0 + initial_fraction * (s1 - s0)

    def newton_step(
        _: Array, candidate_s: Float64[Array, ""]
    ) -> Float64[Array, ""]:
        """Apply one analytic-Jacobian Newton refinement."""
        point: Float64[Array, "3"] = point0 + (candidate_s - s0) * tangent
        candidate_u: Float64[Array, ""]
        candidate_v: Float64[Array, ""]
        derivative: Float64[Array, ""]
        candidate_u, candidate_v, derivative = _path_angle_and_derivative(
            point,
            tangent,
            source_to_lab,
            kinetic_energy,
            geometry.slit,
        )
        del candidate_v
        safe_derivative: Float64[Array, ""] = jnp.where(
            jnp.abs(derivative) > 1.0e-12,  # noqa: PLR2004
            derivative,
            1.0,
        )
        refined: Float64[Array, ""] = (
            candidate_s - (candidate_u - bounded_u) / safe_derivative
        )
        return refined

    path_coordinate: Float64[Array, ""] = jax.lax.fori_loop(
        0, 10, newton_step, initial_s
    )
    final_point: Float64[Array, "3"] = (
        point0 + (path_coordinate - s0) * tangent
    )
    du_ds: Float64[Array, ""]
    _, _, du_ds = _path_angle_and_derivative(
        final_point,
        tangent,
        source_to_lab,
        kinetic_energy,
        geometry.slit,
    )
    inverse_derivative: Float64[Array, ""] = 1.0 / jnp.abs(du_ds)
    inverse: Tuple[Float64[Array, ""], Float64[Array, ""], Bool[Array, ""]] = (
        path_coordinate,
        inverse_derivative,
        in_support,
    )
    return inverse


def _map_spectrum(
    source: ArpesSpectrum,
    geometry: ExperimentGeometry,
    calibration: DetectorCalibration,
    euler_angles_rad: Float64[Array, " 3"],
    order: int,
) -> Tuple[Float64[Array, "u 1 e"], Float64[Array, ""]]:
    """PRIVATE: Compute a conservative slit-line-density map.

    Parameters
    ----------
    source : ArpesSpectrum
        Source line density already integrated over the transverse aperture.
    geometry : ExperimentGeometry
        Kinematics and sample azimuth.
    calibration : DetectorCalibration
        Explicit one-bin transverse target.
    euler_angles_rad : Float64[Array, " 3"]
        Active source-domain z--y--z rotation.
    order : int
        Static quadrature order.

    Returns
    -------
    mapped : Tuple[Float64[Array, "u 1 e"], Float64[Array, ""]]
        Native detector density and captured line-flux fraction.
    """
    source_to_lab: Float64[Array, "3 3"] = _source_to_lab_rotation(
        geometry, euler_angles_rad
    )
    u_nodes: Float64[Array, "u q"]
    u_weights: Float64[Array, "u q"]
    u_nodes, u_weights = _bin_quadrature(calibration.u_bin_edges, order)
    energy_nodes: Float64[Array, "e q"]
    energy_weights: Float64[Array, "e q"]
    energy_nodes, energy_weights = _bin_quadrature(
        calibration.energy_bin_edges_ev, order
    )
    energy_nodes = _validate_spectrum_chart(
        source,
        source_to_lab,
        energy_nodes,
        calibration,
        geometry,
        order,
    )
    n_e: int = energy_nodes.shape[0]
    flat_bins: Array = jnp.arange(u_nodes.shape[0] * n_e)

    def integrate_bin(flat_index: Array) -> Float64[Array, ""]:
        """Integrate one active ``u``/energy target bin."""
        u_index: Array = flat_index // n_e
        energy_index: Array = flat_index % n_e
        u_grid: Float64[Array, "q 1"] = u_nodes[u_index, :, None]
        energy_grid: Float64[Array, "1 q"] = energy_nodes[
            energy_index, None, :
        ]
        broadcast_u: Float64[Array, "q q"]
        broadcast_energy: Float64[Array, "q q"]
        broadcast_u, broadcast_energy = jnp.broadcast_arrays(
            u_grid, energy_grid
        )

        def evaluate_node(
            target: Tuple[Float64[Array, ""], Float64[Array, ""]],
        ) -> Float64[Array, ""]:
            """Evaluate transformed line density at one quadrature node."""
            target_u: Float64[Array, ""]
            omega: Float64[Array, ""]
            target_u, omega = target
            path_coordinate: Float64[Array, ""]
            inverse_derivative: Float64[Array, ""]
            in_support: Bool[Array, ""]
            path_coordinate, inverse_derivative, in_support = (
                _solve_path_coordinate(
                    target_u,
                    omega,
                    source,
                    source_to_lab,
                    geometry,
                )
            )
            source_density: Float64[Array, ""] = _interpolate_spectrum(
                source, path_coordinate, omega
            )
            transformed: Float64[Array, ""] = jnp.where(
                in_support,
                source_density * inverse_derivative,
                0.0,
            )
            return transformed

        transformed_flat: Float64[Array, " n"] = jax.vmap(evaluate_node)(
            (broadcast_u.reshape((-1,)), broadcast_energy.reshape((-1,)))
        )
        transformed: Float64[Array, "q q"] = transformed_flat.reshape(
            broadcast_u.shape
        )
        weight_grid: Float64[Array, "q q"] = (
            u_weights[u_index, :, None] * energy_weights[energy_index, None, :]
        )
        mass: Float64[Array, ""] = jnp.sum(weight_grid * transformed)
        return mass

    masses: Float64[Array, " b"] = jax.lax.map(
        jax.checkpoint(integrate_bin), flat_bins
    )
    delta_u: Float64[Array, " u"] = jnp.diff(calibration.u_bin_edges)
    delta_energy: Float64[Array, " e"] = jnp.diff(
        calibration.energy_bin_edges_ev
    )
    delta_v: Float64[Array, ""] = (
        calibration.v_bin_edges[1] - calibration.v_bin_edges[0]
    )
    active_volumes: Float64[Array, "u e"] = (
        delta_u[:, None] * delta_energy[None, :]
    )
    line_density: Float64[Array, "u e"] = (
        masses.reshape((u_nodes.shape[0], n_e)) / active_volumes
    )
    detector_density: Float64[Array, "u 1 e"] = (
        line_density[:, None, :] / delta_v
    )
    source_flux: Float64[Array, ""] = _source_flux(source)
    fraction: Float64[Array, ""] = _captured_fraction(
        detector_density, source_flux, calibration
    )
    mapped: Tuple[Float64[Array, "u 1 e"], Float64[Array, ""]] = (
        detector_density,
        fraction,
    )
    return mapped


def _map_source_to_detector_with_order(
    source: Union[ArpesCube, ArpesSpectrum],
    geometry: ExperimentGeometry,
    calibration: DetectorCalibration,
    euler_angles_rad: Optional[Float64[Array, " 3"]] = None,
    *,
    order: int = 4,
) -> Tuple[Float64[Array, "u v e"], Float64[Array, ""]]:
    """PRIVATE: Compute one source map with a registered quadrature order.

    Parameters
    ----------
    source : Union[ArpesCube, ArpesSpectrum]
        Self-describing source density.
    geometry : ExperimentGeometry
        Experiment kinematics and sample azimuth.
    calibration : DetectorCalibration
        Explicit native detector target.
    euler_angles_rad : Optional[Float64[Array, " 3"]], optional
        Active source-domain z--y--z rotation.  ``None`` selects identity.
    order : int, optional
        Static quadrature order, four for production and eight for the
        convergence gate.  Default is four.

    Returns
    -------
    mapped : Tuple[Float64[Array, "u v e"], Float64[Array, ""]]
        Native detector density and captured source-flux fraction.
    """
    _gauss_legendre_rule(order)
    angles: Float64[Array, " 3"] = (
        jnp.zeros(3, dtype=jnp.float64)
        if euler_angles_rad is None
        else jnp.asarray(euler_angles_rad, dtype=jnp.float64)
    )
    angles = _validate_common(source, geometry, calibration, angles)
    mapped: Tuple[Float64[Array, "u v e"], Float64[Array, ""]]
    if isinstance(source, ArpesCube):
        mapped = _map_cube(source, geometry, calibration, angles, order)
    else:
        mapped = _map_spectrum(source, geometry, calibration, angles, order)
    density: Float64[Array, "u v e"] = eqx.error_if(
        mapped[0],
        ~jnp.all(jnp.isfinite(mapped[0])) | ~jnp.all(mapped[0] >= 0.0),
        "mapped detector density must be finite and nonnegative",
    )
    fraction: Float64[Array, ""] = eqx.error_if(
        mapped[1],
        ~jnp.isfinite(mapped[1]) | (mapped[1] < 0.0) | (mapped[1] > 1.0),
        (
            "captured detector fraction must lie in [0, 1]; refine target "
            "bins if quadrature creates flux"
        ),
    )
    validated: Tuple[Float64[Array, "u v e"], Float64[Array, ""]] = (
        density,
        fraction,
    )
    return validated


def _map_source_to_detector(
    source: Union[ArpesCube, ArpesSpectrum],
    geometry: ExperimentGeometry,
    calibration: DetectorCalibration,
) -> Tuple[Float64[Array, "u v e"], Float64[Array, ""]]:
    """PRIVATE: Compute one unrotated source map with production quadrature.

    Parameters
    ----------
    source : Union[ArpesCube, ArpesSpectrum]
        Self-describing source density.
    geometry : ExperimentGeometry
        Experiment kinematics and sample azimuth.
    calibration : DetectorCalibration
        Explicit native detector target.

    Returns
    -------
    mapped : Tuple[Float64[Array, "u v e"], Float64[Array, ""]]
        Native detector density and captured source-flux fraction.
    """
    mapped: Tuple[Float64[Array, "u v e"], Float64[Array, ""]] = (
        _map_source_to_detector_with_order(
            source, geometry, calibration, order=4
        )
    )
    return mapped


def _map_and_mix_domains(
    physical_by_domain: Tuple[Union[ArpesCube, ArpesSpectrum], ...],
    geometry: ExperimentGeometry,
    calibration: DetectorCalibration,
    effects: DetectorEffects,
) -> Tuple[Float64[Array, "u v e"], Float64[Array, ""]]:
    """PRIVATE: Compute rotated domain maps before detector-space mixing.

    Parameters
    ----------
    physical_by_domain : Tuple[Union[ArpesCube, ArpesSpectrum], ...]
        Nonempty static tuple of self-describing source densities.
    geometry : ExperimentGeometry
        Experiment kinematics and sample azimuth.
    calibration : DetectorCalibration
        Common explicit native detector target.
    effects : DetectorEffects
        Traced domain rotations/logits and registered frame identities.

    Returns
    -------
    mixed : Tuple[Float64[Array, "u v e"], Float64[Array, ""]]
        Softmax-weighted detector density and its flux-weighted captured
        fraction.

    Raises
    ------
    ValueError
        If the domain count or frame identities disagree.

    Notes
    -----
    Mapping precedes mixing.  Differently oriented source arrays are never
    combined pointwise.
    """
    n_domain: int = len(physical_by_domain)
    if n_domain < 1:
        raise ValueError("detector domain tuple cannot be empty")
    if effects.domain_logits.shape[0] != n_domain:
        raise ValueError("detector domain logits and source count disagree")
    if effects.domain_euler_angles_rad.shape != (n_domain, 3):
        raise ValueError("detector rotations and source count disagree")
    if effects.coordinate_density != "per_native_volume":
        raise ValueError("detector mapping requires per-native-volume density")
    if len(effects.domain_frame_ids) != n_domain:
        raise ValueError(
            "detector frame identifiers and source count disagree"
        )
    domain_index: int
    densities: list[Float64[Array, "u v e"]] = []
    source_fluxes: list[Float64[Array, ""]] = []
    for domain_index in range(n_domain):
        source: Union[ArpesCube, ArpesSpectrum] = physical_by_domain[
            domain_index
        ]
        if effects.domain_frame_ids[domain_index] != source.cartesian_frame_id:
            raise ValueError(
                "detector domain and source frame identifiers disagree"
            )
        density: Float64[Array, "u v e"]
        fraction: Float64[Array, ""]
        density, fraction = _map_source_to_detector_with_order(
            source,
            geometry,
            calibration,
            effects.domain_euler_angles_rad[domain_index],
            order=4,
        )
        del fraction
        densities.append(density)
        source_fluxes.append(_source_flux(source))
    density_stack: Float64[Array, "d u v e"] = jnp.stack(densities)
    source_flux_stack: Float64[Array, " d"] = jnp.stack(source_fluxes)
    weights: Float64[Array, " d"] = jax.nn.softmax(effects.domain_logits)
    mixed_density: Float64[Array, "u v e"] = jnp.einsum(
        "d,duve->uve", weights, density_stack
    )
    mixed_source_flux: Float64[Array, ""] = jnp.dot(weights, source_flux_stack)
    mixed_fraction: Float64[Array, ""] = _captured_fraction(
        mixed_density, mixed_source_flux, calibration
    )
    mixed_fraction = eqx.error_if(
        mixed_fraction,
        ~jnp.isfinite(mixed_fraction)
        | (mixed_fraction < 0.0)
        | (mixed_fraction > 1.0),
        "mixed captured detector fraction must lie in [0, 1]",
    )
    mixed: Tuple[Float64[Array, "u v e"], Float64[Array, ""]] = (
        mixed_density,
        mixed_fraction,
    )
    return mixed
