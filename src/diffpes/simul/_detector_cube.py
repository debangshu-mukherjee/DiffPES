"""PRIVATE: Map Cartesian source cubes to native detector bins.

Extended Summary
----------------
This private module maps Cartesian source densities.
The finite-volume map preserves the captured source flux.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
from beartype.typing import Tuple
from jaxtyping import Array, Bool, Float64, Int64

from diffpes.types import ArpesCube, DetectorCalibration, ExperimentGeometry

from ._detector_geometry import (
    _bin_quadrature,
    _captured_fraction,
    _endpoint_seams,
    _interpolate_cube,
    _inverse_map_abs_jacobian,
    _segmented_quadrature,
    _source_faces,
    _source_flux,
    _source_to_lab_rotation,
    _target_volumes,
)
from .kinematics import detector_angles_to_kpar, final_state_k_inv_ang


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
    flat_bins: Int64[Array, " n_bin"] = jnp.arange(
        u_nodes.shape[0] * n_v * n_e
    )

    def integrate_bin(flat_index: Int64[Array, ""]) -> Float64[Array, ""]:
        """Integrate one explicit native target bin."""
        u_index: Int64[Array, ""] = flat_index // (n_v * n_e)
        remainder: Int64[Array, ""] = flat_index % (n_v * n_e)
        v_index: Int64[Array, ""] = remainder // n_e
        energy_index: Int64[Array, ""] = remainder % n_e
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
    flat_bins: Int64[Array, " n_bin"] = jnp.arange(n_u * n_v * n_e)

    def integrate_bin(flat_index: Int64[Array, ""]) -> Float64[Array, ""]:
        """Integrate one native bin between mapped momentum seams."""
        u_index: Int64[Array, ""] = flat_index // (n_v * n_e)
        remainder: Int64[Array, ""] = flat_index % (n_v * n_e)
        v_index: Int64[Array, ""] = remainder // n_e
        energy_index: Int64[Array, ""] = remainder % n_e
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


__all__: list[str] = []
