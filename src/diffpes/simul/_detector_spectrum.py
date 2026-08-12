"""PRIVATE: Map path spectra to native detector bins.

Extended Summary
----------------
This private module maps one-dimensional path densities.
The map uses the declared slit aperture.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
from beartype.typing import Tuple
from jaxtyping import Array, Bool, Float64, Int32, Int64

from diffpes.types import (
    ArpesSpectrum,
    DetectorCalibration,
    ExperimentGeometry,
)

from ._detector_geometry import (
    _analytic_angle_jacobian,
    _bin_quadrature,
    _captured_fraction,
    _gauss_legendre_rule,
    _interpolate_spectrum,
    _source_faces,
    _source_flux,
    _source_to_lab_rotation,
)
from .kinematics import final_state_k_inv_ang, kpar_to_detector_angles


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
        Static detector slit orientation.

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
            values: Tuple[
                Float64[Array, " q"],
                Float64[Array, " q"],
                Float64[Array, " q"],
            ] = jax.vmap(
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
    derivative uses the explicit analytic detector-angle Jacobian rather than
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
    upper: Int32[Array, ""] = jnp.clip(
        jnp.searchsorted(ordered_u, bounded_u, side="right"),
        1,
        ordered_u.size - 1,
    )
    lower: Int32[Array, ""] = upper - 1
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
        _: Int64[Array, ""], candidate_s: Float64[Array, ""]
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
    flat_bins: Int64[Array, " n_bin"] = jnp.arange(u_nodes.shape[0] * n_e)

    def integrate_bin(flat_index: Int64[Array, ""]) -> Float64[Array, ""]:
        """Integrate one active ``u``/energy target bin."""
        u_index: Int64[Array, ""] = flat_index // n_e
        energy_index: Int64[Array, ""] = flat_index % n_e
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


__all__: list[str] = []
