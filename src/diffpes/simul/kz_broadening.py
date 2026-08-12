"""Apply wrapped out-of-plane momentum broadening.

Extended Summary
----------------
This module integrates wrapped-Cauchy mass.
It uses surface-fractional momentum bins.

Routine Listings
----------------
:func:`broaden_kz`
    Apply wrapped-Cauchy bin masses to node-resolved bulk intensity.
:func:`kz_fractional_nodes`
    Build static uniform surface-fractional kz bin centres.
:func:`kz_wrapped_lorentzian_bin_weights`
    Integrate wrapped-Lorentzian mass over fractional kz bins.
"""

import math

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from beartype import beartype
from beartype.typing import Tuple
from jaxtyping import Array, Bool, Float64, jaxtyped

from diffpes.maths import safe_arctan2, safe_norm
from diffpes.types import CrystalGeometry, ScalarFloat, SurfaceCell


def _wrapped_cauchy_unwrapped_cdf(
    displacement_frac: Float64[Array, "..."],
    gamma_frac: Float64[Array, ""],
) -> Float64[Array, "..."]:
    r"""PRIVATE: Evaluate a continuous period-unwrapped Cauchy CDF.

    Parameters
    ----------
    displacement_frac : Float64[Array, "..."]
        Fractional displacement from the wrapped distribution centre.
    gamma_frac : Float64[Array, ""]
        Positive Lorentzian HWHM divided by the reciprocal period.

    Returns
    -------
    cdf : Float64[Array, "..."]
        Continuous primitive satisfying ``cdf(x + 1) = cdf(x) + 1``.

    Notes
    -----
    Reducing onto ``[-1/2, 1/2)`` makes the winding number explicit. The
    ``atan2`` form remains finite when the wrapped peak is narrow and avoids
    multiplying ``cotanh(pi * gamma_frac)`` by a zero tangent at the centre.
    """
    winding: Float64[Array, "..."] = jnp.floor(displacement_frac + 0.5)
    reduced: Float64[Array, "..."] = displacement_frac - winding
    angle: Float64[Array, "..."] = math.pi * reduced
    local: Float64[Array, "..."] = (
        safe_arctan2(
            jnp.sin(angle),
            jnp.tanh(math.pi * gamma_frac) * jnp.cos(angle),
        )
        / math.pi
    )
    cdf: Float64[Array, "..."] = winding + local
    return cdf


@jaxtyped(typechecker=beartype)
def _kz_wrapped_lorentzian_bin_weight(  # noqa: DOC502, DOC503
    lower_edge_frac: ScalarFloat,
    upper_edge_frac: ScalarFloat,
    kz_center_folded_frac: Float64[Array, "..."],
    mean_free_path_ang: ScalarFloat,
    period_inv_ang: ScalarFloat,
) -> Float64[Array, "..."]:
    """PRIVATE: Integrate one wrapped-Cauchy fractional bin.

    Parameters
    ----------
    lower_edge_frac : ScalarFloat
        Lower bin edge in the normalized surface-fractional period.
    upper_edge_frac : ScalarFloat
        Upper bin edge in the normalized surface-fractional period.
    kz_center_folded_frac : Float64[Array, "..."]
        Wrapped distribution centres in ``[-1/2, 1/2)``.
    mean_free_path_ang : ScalarFloat
        Positive finite intensity escape length in angstroms.
    period_inv_ang : ScalarFloat
        Positive finite primitive reciprocal period in inverse angstroms.

    Returns
    -------
    weight : Float64[Array, "..."]
        Positive analytic mass in the declared fractional bin.

    Raises
    ------
    EquinoxRuntimeError
        If an edge, centre, physical scale, or computed mass is invalid.

    Notes
    -----
    This scalar-bin primitive is the streaming seam used by the bulk-kz scan.
    It avoids constructing a complete ``K x E x n_kz`` weight carrier.
    """
    lower_bound: float = -0.5
    upper_bound: float = 0.5
    lower: Float64[Array, ""] = jnp.asarray(lower_edge_frac, dtype=jnp.float64)
    upper: Float64[Array, ""] = jnp.asarray(upper_edge_frac, dtype=jnp.float64)
    lower = eqx.error_if(
        lower,
        ~jnp.isfinite(lower)
        | ~jnp.isfinite(upper)
        | (lower < lower_bound)
        | (upper > upper_bound)
        | ~(upper > lower),
        "kz bin edges must be finite, increasing, and inside [-1/2, 1/2]",
    )
    center: Float64[Array, "..."] = jnp.asarray(
        kz_center_folded_frac, dtype=jnp.float64
    )
    center = eqx.error_if(
        center,
        ~jnp.all(jnp.isfinite(center))
        | ~jnp.all((center >= lower_bound) & (center < upper_bound)),
        "folded kz centres must be finite and lie in [-1/2, 1/2)",
    )
    mean_free_path: Float64[Array, ""] = jnp.asarray(
        mean_free_path_ang, dtype=jnp.float64
    )
    mean_free_path = eqx.error_if(
        mean_free_path,
        ~jnp.isfinite(mean_free_path) | ~(mean_free_path > 0.0),
        "mean_free_path_ang must be finite and positive",
    )
    period: Float64[Array, ""] = jnp.asarray(period_inv_ang, dtype=jnp.float64)
    period = eqx.error_if(
        period,
        ~jnp.isfinite(period) | ~(period > 0.0),
        "period_inv_ang must be finite and positive",
    )

    gamma_frac: Float64[Array, ""] = 0.5 / (mean_free_path * period)
    upper_cdf: Float64[Array, "..."] = _wrapped_cauchy_unwrapped_cdf(
        upper - center, gamma_frac
    )
    lower_cdf: Float64[Array, "..."] = _wrapped_cauchy_unwrapped_cdf(
        lower - center, gamma_frac
    )
    weight: Float64[Array, "..."] = upper_cdf - lower_cdf
    checked_weight: Float64[Array, "..."] = eqx.error_if(
        weight,
        ~jnp.all(jnp.isfinite(weight)) | ~jnp.all(weight > 0.0),
        "wrapped kz bin mass must be finite and positive",
    )
    return checked_weight


@jaxtyped(typechecker=beartype)
def _surface_kz_frame(  # noqa: DOC502, DOC503
    surface_cell: SurfaceCell,
    bulk_geometry: CrystalGeometry,
) -> Tuple[
    Float64[Array, "3 3"],
    Float64[Array, "3 3"],
    Float64[Array, " 3"],
    Float64[Array, ""],
]:
    """PRIVATE: Validate and derive one primitive surface reciprocal frame.

    Parameters
    ----------
    surface_cell : SurfaceCell
        Surface vectors, rotation, and exact integer provenance.
    bulk_geometry : CrystalGeometry
        Bulk direct and reciprocal lattices associated with the surface.

    Returns
    -------
    direct_surface : Float64[Array, "3 3"]
        Surface-frame direct vectors ``(v1, v2, v3)`` as rows.
    reciprocal_surface : Float64[Array, "3 3"]
        Surface-frame reciprocal vectors satisfying ``A @ B.T = 2*pi*I``.
    normal_hat : Float64[Array, " 3"]
        Unit surface normal oriented so ``dot(normal_hat, v3) > 0``.
    period_inv_ang : Float64[Array, ""]
        Primitive normal reciprocal period in inverse angstroms.

    Raises
    ------
    ValueError
        If the exact Miller/stacking metadata lacks unit plane advance.
    EquinoxRuntimeError
        If continuous surface fields disagree with the bulk lattice and
        rotation or fail reciprocal, normal, and spacing identities.

    Notes
    -----
    Reconstructing the surface vectors from exact integer coefficients catches
    a numerically doubled stacking vector with stale unit-advance metadata.
    """
    unit_advance: int = sum(
        miller_component * stacking_component
        for miller_component, stacking_component in zip(
            surface_cell.miller,
            surface_cell.stacking_coeffs,
            strict=True,
        )
    )
    if unit_advance != 1:
        raise ValueError(
            "surface Miller and stacking coefficients must have unit advance"
        )

    coefficients: Float64[Array, "3 3"] = jnp.asarray(
        (
            surface_cell.in_plane_coeffs[0],
            surface_cell.in_plane_coeffs[1],
            surface_cell.stacking_coeffs,
        ),
        dtype=jnp.float64,
    )
    direct_surface: Float64[Array, "3 3"] = jnp.concatenate(
        (surface_cell.in_plane_vectors, surface_cell.stacking_vector[None, :]),
        axis=0,
    )
    expected_surface: Float64[Array, "3 3"] = (
        coefficients @ bulk_geometry.lattice @ surface_cell.rotation.T
    )
    scale: Float64[Array, ""] = jnp.maximum(
        1.0, jnp.max(jnp.abs(expected_surface))
    )
    frame_tolerance: float = 1.0e-10
    direct_surface = eqx.error_if(
        direct_surface,
        ~jnp.all(jnp.isfinite(direct_surface))
        | ~jnp.all(jnp.isfinite(expected_surface))
        | (
            jnp.max(jnp.abs(direct_surface - expected_surface))
            > frame_tolerance * scale
        ),
        "surface vectors must match coefficient @ bulk lattice @ rotation.T",
    )

    raw_normal: Float64[Array, " 3"] = jnp.cross(
        direct_surface[0], direct_surface[1]
    )
    normal_length: Float64[Array, ""] = safe_norm(raw_normal)
    normal_length = eqx.error_if(
        normal_length,
        ~jnp.isfinite(normal_length) | ~(normal_length > 0.0),
        "surface in-plane vectors must define a finite nonzero normal",
    )
    unoriented_normal: Float64[Array, " 3"] = raw_normal / normal_length
    projection: Float64[Array, ""] = jnp.dot(
        unoriented_normal, direct_surface[2]
    )
    minimum_projection: float = 1.0e-12
    projection = eqx.error_if(
        projection,
        ~jnp.isfinite(projection)
        | (jnp.abs(projection) <= minimum_projection),
        "surface stacking vector must have a nonzero normal projection",
    )
    normal_hat: Float64[Array, " 3"] = jnp.where(
        projection > 0.0, unoriented_normal, -unoriented_normal
    )
    positive_projection: Float64[Array, ""] = jnp.abs(projection)

    reciprocal_surface: Float64[Array, "3 3"] = (
        2.0 * math.pi * jnp.linalg.inv(direct_surface).T
    )
    period_inv_ang: Float64[Array, ""] = safe_norm(reciprocal_surface[2])
    expected_period: Float64[Array, ""] = 2.0 * math.pi / positive_projection
    reciprocal_error: Float64[Array, ""] = jnp.max(
        jnp.abs(
            direct_surface @ reciprocal_surface.T
            - 2.0 * math.pi * jnp.eye(3, dtype=jnp.float64)
        )
    )
    spacing_error: Float64[Array, ""] = jnp.abs(
        positive_projection - surface_cell.interlayer_spacing_ang
    )
    period_error: Float64[Array, ""] = jnp.abs(
        period_inv_ang - expected_period
    )
    reciprocal_surface = eqx.error_if(
        reciprocal_surface,
        ~jnp.all(jnp.isfinite(reciprocal_surface))
        | ~jnp.isfinite(period_inv_ang)
        | ~(period_inv_ang > 0.0)
        | (reciprocal_error > frame_tolerance)
        | (
            spacing_error
            > frame_tolerance * jnp.maximum(1.0, positive_projection)
        )
        | (period_error > frame_tolerance * jnp.maximum(1.0, expected_period)),
        "surface reciprocal, normal-spacing, and primitive-period "
        "identities must agree",
    )
    frame: Tuple[
        Float64[Array, "3 3"],
        Float64[Array, "3 3"],
        Float64[Array, " 3"],
        Float64[Array, ""],
    ] = (direct_surface, reciprocal_surface, normal_hat, period_inv_ang)
    return frame


@jaxtyped(typechecker=beartype)
def _map_surface_fractional_to_bulk(  # noqa: DOC502, DOC503
    k_parallel_cart_inv_ang: Float64[Array, "... 3"],
    kz_fractional: Float64[Array, "..."],
    surface_cell: SurfaceCell,
    bulk_geometry: CrystalGeometry,
) -> Tuple[
    Float64[Array, "... 3"],
    Float64[Array, "... 3"],
]:
    """PRIVATE: Convert folded surface-kz coordinates into bulk space.

    Parameters
    ----------
    k_parallel_cart_inv_ang : Float64[Array, "... 3"]
        Physical surface-plane Cartesian momentum vectors.
    kz_fractional : Float64[Array, "..."]
        Folded third-surface-fractional coordinates. Their leading shape must
        begin with the complete non-Cartesian shape of ``k_parallel``.
    surface_cell : SurfaceCell
        Validated surface frame.
    bulk_geometry : CrystalGeometry
        Bulk geometry associated with ``surface_cell``.

    Returns
    -------
    surface_cart_folded : Float64[Array, "... 3"]
        Folded surface-frame Cartesian points with the kz-coordinate shape.
    bulk_fractional_folded : Float64[Array, "... 3"]
        The same points in bulk reciprocal-fractional coordinates.

    Raises
    ------
    ValueError
        If the folded-coordinate shape does not begin with the momentum shape.
    EquinoxRuntimeError
        If momentum or folded coordinates are invalid or a reciprocal round
        trip fails.

    Notes
    -----
    The normal coordinate uses ``q(u) = G_perp * (u - u_parallel)``. This
    retains the lateral component of an oblique stacking vector instead of
    appending a scalar kz to unrelated fractional in-plane coordinates. A
    bulk-direct caller may supply ``(K,E)`` centres; the node wrapper below
    supplies the registered ``(K,n_kz)`` grid to the same implementation.
    """
    k_parallel: Float64[Array, "... 3"] = jnp.asarray(
        k_parallel_cart_inv_ang, dtype=jnp.float64
    )
    folded: Float64[Array, "..."] = jnp.asarray(
        kz_fractional, dtype=jnp.float64
    )
    parallel_shape: Tuple[int, ...] = k_parallel.shape[:-1]
    if (
        folded.ndim < len(parallel_shape)
        or folded.shape[: len(parallel_shape)] != parallel_shape
    ):
        raise ValueError(
            "folded kz shape must begin with the k_parallel batch shape"
        )
    lower_bound: float = -0.5
    upper_bound: float = 0.5
    folded = eqx.error_if(
        folded,
        ~jnp.all(jnp.isfinite(folded))
        | ~jnp.all((folded >= lower_bound) & (folded < upper_bound)),
        "folded surface kz coordinates must be finite and lie in [-1/2, 1/2)",
    )
    direct_surface: Float64[Array, "3 3"]
    reciprocal_surface: Float64[Array, "3 3"]
    normal_hat: Float64[Array, " 3"]
    period_inv_ang: Float64[Array, ""]
    (
        direct_surface,
        reciprocal_surface,
        normal_hat,
        period_inv_ang,
    ) = _surface_kz_frame(surface_cell, bulk_geometry)
    plane_component: Float64[Array, "..."] = jnp.einsum(
        "...i,i->...", k_parallel, normal_hat
    )
    momentum_scale: Float64[Array, "..."] = jnp.maximum(
        1.0, safe_norm(k_parallel)
    )
    k_parallel = eqx.error_if(
        k_parallel,
        ~jnp.all(jnp.isfinite(k_parallel))
        | jnp.any(jnp.abs(plane_component) > 1.0e-10 * momentum_scale),
        "k_parallel_cart_inv_ang must be finite and lie in the surface plane",
    )

    u_parallel: Float64[Array, "..."] = jnp.einsum(
        "...i,i->...", k_parallel, direct_surface[2]
    ) / (2.0 * math.pi)
    extra_axes: int = folded.ndim - len(parallel_shape)
    expanded_parallel: Float64[Array, "... 3"] = jnp.reshape(
        k_parallel,
        (*parallel_shape, *(1 for _ in range(extra_axes)), 3),
    )
    expanded_u_parallel: Float64[Array, "..."] = jnp.reshape(
        u_parallel,
        (*parallel_shape, *(1 for _ in range(extra_axes))),
    )
    q_inv_ang: Float64[Array, "..."] = period_inv_ang * (
        folded - expanded_u_parallel
    )
    surface_cart_unfolded: Float64[Array, "... 3"] = (
        expanded_parallel + q_inv_ang[..., None] * normal_hat
    )
    surface_fractional_unfolded: Float64[Array, "... 3"] = (
        surface_cart_unfolded @ direct_surface.T / (2.0 * math.pi)
    )
    winding: Float64[Array, "..."] = jnp.floor(
        surface_fractional_unfolded[..., 2] + 0.5
    )
    third_axis: Float64[Array, " 3"] = jnp.asarray(
        [0.0, 0.0, 1.0], dtype=jnp.float64
    )
    surface_fractional_folded: Float64[Array, "... 3"] = (
        surface_fractional_unfolded - winding[..., None] * third_axis
    )
    surface_cart_folded: Float64[Array, "... 3"] = (
        surface_fractional_folded @ reciprocal_surface
    )
    bulk_cart_folded: Float64[Array, "... 3"] = (
        surface_cart_folded @ surface_cell.rotation
    )
    bulk_fractional_folded: Float64[Array, "... 3"] = (
        bulk_cart_folded @ jnp.linalg.inv(bulk_geometry.reciprocal)
    )

    surface_round_trip: Float64[Array, "... 3"] = (
        bulk_fractional_folded
        @ bulk_geometry.reciprocal
        @ surface_cell.rotation.T
    )
    folded_round_trip: Float64[Array, "... 3"] = (
        surface_cart_folded @ direct_surface.T / (2.0 * math.pi)
    )
    map_tolerance: float = 1.0e-12
    bulk_fractional_folded = eqx.error_if(
        bulk_fractional_folded,
        ~jnp.all(jnp.isfinite(bulk_fractional_folded))
        | (
            jnp.max(jnp.abs(surface_round_trip - surface_cart_folded))
            > map_tolerance
        )
        | (
            jnp.max(jnp.abs(folded_round_trip - surface_fractional_folded))
            > map_tolerance
        )
        | (
            jnp.max(jnp.abs(surface_fractional_folded[..., 2] - folded))
            > map_tolerance
        ),
        "surface-to-bulk reciprocal round trips must agree",
    )
    mapped: Tuple[
        Float64[Array, "... 3"],
        Float64[Array, "... 3"],
    ] = (surface_cart_folded, bulk_fractional_folded)
    return mapped


@jaxtyped(typechecker=beartype)
def _map_surface_kz_nodes_to_bulk_fractional(  # noqa: DOC502, DOC503
    k_parallel_cart_inv_ang: Float64[Array, "... 3"],
    kz_nodes_frac: Float64[Array, " n_kz"],
    surface_cell: SurfaceCell,
    bulk_geometry: CrystalGeometry,
) -> Tuple[
    Float64[Array, "... n_kz 3"],
    Float64[Array, "... n_kz 3"],
]:
    """PRIVATE: Convert registered uniform surface-kz nodes into bulk space.

    Parameters
    ----------
    k_parallel_cart_inv_ang : Float64[Array, "... 3"]
        Physical surface-plane Cartesian momentum vectors.
    kz_nodes_frac : Float64[Array, " n_kz"]
        Registered uniform third-surface-fractional bin centres.
    surface_cell : SurfaceCell
        Validated surface frame.
    bulk_geometry : CrystalGeometry
        Bulk geometry associated with ``surface_cell``.

    Returns
    -------
    surface_cart_folded : Float64[Array, "... n_kz 3"]
        Folded Cartesian momentum points in the surface frame.
    bulk_fractional_folded : Float64[Array, "... n_kz 3"]
        The same points in bulk reciprocal-fractional coordinates.

    Raises
    ------
    ValueError
        If the node array is not a registered static grid with at least two
        centres.
    EquinoxRuntimeError
        If a node differs from the registered uniform centres.

    Notes
    -----
    This wrapper broadcasts one static node vector over every physical
    in-plane momentum and delegates all geometry to the generic mapper.
    """
    nodes: Float64[Array, " n_kz"] = jnp.asarray(
        kz_nodes_frac, dtype=jnp.float64
    )
    if nodes.ndim != 1 or nodes.shape[0] < 2:  # noqa: PLR2004
        raise ValueError("surface kz mapping requires at least two nodes")
    registered_nodes: Float64[Array, " n_kz"] = kz_fractional_nodes(
        nodes.shape[0]
    )
    node_tolerance: float = 1.0e-14
    nodes = eqx.error_if(
        nodes,
        ~jnp.all(jnp.isfinite(nodes))
        | (jnp.max(jnp.abs(nodes - registered_nodes)) > node_tolerance),
        "surface kz nodes must be the registered uniform fractional centres",
    )
    k_parallel: Float64[Array, "... 3"] = jnp.asarray(
        k_parallel_cart_inv_ang, dtype=jnp.float64
    )
    folded_shape: Tuple[int, ...] = (*k_parallel.shape[:-1], nodes.shape[0])
    folded: Float64[Array, "..."] = jnp.broadcast_to(nodes, folded_shape)
    mapped: Tuple[
        Float64[Array, "... n_kz 3"],
        Float64[Array, "... n_kz 3"],
    ] = _map_surface_fractional_to_bulk(
        k_parallel,
        folded,
        surface_cell,
        bulk_geometry,
    )
    return mapped


@jaxtyped(typechecker=beartype)
def kz_fractional_nodes(n_kz: int) -> Float64[Array, " n_kz"]:
    """Build static uniform surface-fractional kz bin centres.

    The centres cover one primitive reciprocal period on ``[-1/2, 1/2)``.
    A finite-width bulk quadrature always has at least two bins.

    :see: :class:`~.test_kz_broadening.TestKzFractionalNodes`

    Parameters
    ----------
    n_kz : int
        Static number of fractional kz bins.

    Returns
    -------
    nodes : Float64[Array, " n_kz"]
        Uniform bin centres in ascending order.

    Raises
    ------
    ValueError
        If ``n_kz`` is not a static integer of at least two.

    Notes
    -----
    The grid divides the primitive period into equal cells and returns their
    midpoints.
    """
    if type(n_kz) is not int or n_kz < 2:  # noqa: PLR2004
        raise ValueError("n_kz must be a static integer of at least two")
    indices: Float64[Array, " n_kz"] = jnp.arange(n_kz, dtype=jnp.float64)
    nodes: Float64[Array, " n_kz"] = (indices + 0.5) / n_kz - 0.5
    return nodes


@jaxtyped(typechecker=beartype)
def kz_wrapped_lorentzian_bin_weights(  # noqa: DOC502, DOC503
    kz_bin_edges_frac: Float64[Array, " n_kz_plus_one"],
    kz_center_folded_frac: Float64[Array, "..."],
    mean_free_path_ang: ScalarFloat,
    period_inv_ang: ScalarFloat,
) -> Float64[Array, "... n_kz"]:
    r"""Integrate wrapped-Lorentzian mass over fractional kz bins.

    The physical HWHM is ``gamma = 1 / (2 * mean_free_path_ang)`` in inverse
    angstroms. This division by ``period_inv_ang`` produces the dimensionless
    period-one width before analytic CDF evaluation. No image cutoff, cropped
    tail, or post-hoc normalization enters the result.

    :see: :class:`~.test_kz_broadening.TestKzWrappedLorentzianBinWeights`

    Parameters
    ----------
    kz_bin_edges_frac : Float64[Array, " n_kz_plus_one"]
        Increasing fractional bin edges spanning exactly ``[-1/2, 1/2]``.
    kz_center_folded_frac : Float64[Array, "..."]
        Wrapped distribution centres in ``[-1/2, 1/2)``.
    mean_free_path_ang : ScalarFloat
        Positive finite intensity escape length in angstroms.
    period_inv_ang : ScalarFloat
        Positive finite primitive normal reciprocal period in inverse
        angstroms.

    Returns
    -------
    weights : Float64[Array, "... n_kz"]
        Strictly positive analytic bin masses summing to one per centre.

    Raises
    ------
    ValueError
        If the edge array is not one-dimensional or has fewer than two bins.
    EquinoxRuntimeError
        If an edge, centre, physical scale, or computed mass is invalid.

    Notes
    -----
    The implementation subtracts branch-unwrapped CDF values at every adjacent
    edge pair.
    """
    edges: Float64[Array, " n_kz_plus_one"] = jnp.asarray(
        kz_bin_edges_frac, dtype=jnp.float64
    )
    if edges.ndim != 1 or edges.shape[0] < 3:  # noqa: PLR2004
        raise ValueError("kz bin edges must define at least two bins")
    edge_tolerance: float = 32.0 * np.finfo(np.float64).eps
    edge_invalid: Bool[Array, ""] = (
        ~jnp.all(jnp.isfinite(edges))
        | ~jnp.all(jnp.diff(edges) > 0.0)
        | (jnp.abs(edges[0] + 0.5) > edge_tolerance)
        | (jnp.abs(edges[-1] - 0.5) > edge_tolerance)
    )
    edges = eqx.error_if(
        edges,
        edge_invalid,
        "kz bin edges must be finite, increasing, and span [-1/2, 1/2]",
    )

    center: Float64[Array, "..."] = jnp.asarray(
        kz_center_folded_frac, dtype=jnp.float64
    )
    node_first: Float64[Array, "n_kz ..."] = jax.vmap(
        lambda lower, upper: _kz_wrapped_lorentzian_bin_weight(
            lower,
            upper,
            center,
            mean_free_path_ang,
            period_inv_ang,
        )
    )(
        edges[:-1],
        edges[1:],
    )
    weights: Float64[Array, "... n_kz"] = jnp.moveaxis(node_first, 0, -1)
    mass_tolerance: float = 1.0e-13
    checked_weights: Float64[Array, "... n_kz"] = eqx.error_if(
        weights,
        ~jnp.all(jnp.isfinite(weights))
        | ~jnp.all(weights > 0.0)
        | jnp.any(jnp.abs(jnp.sum(weights, axis=-1) - 1.0) > mass_tolerance),
        "wrapped kz bin masses must be finite, positive, and sum to one",
    )
    return checked_weights


@jaxtyped(typechecker=beartype)
def broaden_kz(  # noqa: DOC502, DOC503
    intensity_per_kz: Float64[Array, "n_kz ..."],
    weights: Float64[Array, "... n_kz"],
) -> Float64[Array, "..."]:
    """Apply wrapped-Cauchy bin masses to node-resolved bulk intensity.

    This incoherent bulk operation is mutually exclusive with coherent slab
    depth attenuation. The node axis is first on ``intensity_per_kz`` and last
    on ``weights``; every remaining axis must agree exactly.

    :see: :class:`~.test_kz_broadening.TestBroadenKz`

    Parameters
    ----------
    intensity_per_kz : Float64[Array, "n_kz ..."]
        Finite nonnegative intensity evaluated at every registered kz centre.
    weights : Float64[Array, "... n_kz"]
        Finite positive wrapped-Cauchy bin masses with unit trailing sum.

    Returns
    -------
    broadened : Float64[Array, "..."]
        Incoherently averaged intensity with the kz axis removed.

    Raises
    ------
    ValueError
        If either array is scalar, the node count is below two, or the static
        non-kz shapes disagree.
    EquinoxRuntimeError
        If intensity or weights violate their finite physical domains.

    Notes
    -----
    The implementation moves the node axis last, multiplies corresponding
    values and masses, and sums that axis.
    """
    intensity: Float64[Array, "n_kz ..."] = jnp.asarray(
        intensity_per_kz, dtype=jnp.float64
    )
    weight_array: Float64[Array, "... n_kz"] = jnp.asarray(
        weights, dtype=jnp.float64
    )
    if intensity.ndim < 1 or weight_array.ndim < 1:
        raise ValueError("kz intensity and weights must be nonscalar arrays")
    if intensity.shape[0] < 2:  # noqa: PLR2004
        raise ValueError(
            "finite-width kz broadening requires at least two nodes"
        )
    if (
        weight_array.shape[-1] != intensity.shape[0]
        or weight_array.shape[:-1] != intensity.shape[1:]
    ):
        raise ValueError(
            "kz weights must match the first intensity axis and all "
            "remaining static shapes"
        )
    intensity = eqx.error_if(
        intensity,
        ~jnp.all(jnp.isfinite(intensity)) | ~jnp.all(intensity >= 0.0),
        "kz intensity must be finite and nonnegative",
    )
    mass_tolerance: float = 1.0e-13
    weight_array = eqx.error_if(
        weight_array,
        ~jnp.all(jnp.isfinite(weight_array))
        | ~jnp.all(weight_array > 0.0)
        | jnp.any(
            jnp.abs(jnp.sum(weight_array, axis=-1) - 1.0) > mass_tolerance
        ),
        "kz weights must be finite, positive, and sum to one",
    )
    node_last: Float64[Array, "... n_kz"] = jnp.moveaxis(intensity, 0, -1)
    broadened: Float64[Array, "..."] = jnp.sum(
        node_last * weight_array, axis=-1
    )
    return broadened


__all__: list[str] = [
    "broaden_kz",
    "kz_fractional_nodes",
    "kz_wrapped_lorentzian_bin_weights",
]
