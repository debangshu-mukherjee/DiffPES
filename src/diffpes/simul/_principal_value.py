"""PRIVATE: Evaluate static principal-value quadrature primitives.

Extended Summary
----------------
This private module owns the frozen core quadrature.
It also owns the power-law tail quadrature.
"""

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from beartype.typing import Any, List, Tuple
from jaxtyping import Array, Float64, Int64
from numpy.typing import NDArray

from diffpes.types import Power2TailSpec, make_power2_tail_spec


def _power2_spec_from_edges(
    edge_value_left: Float64[Array, ""],
    slope_left: Float64[Array, ""],
    edge_value_right: Float64[Array, ""],
    slope_right: Float64[Array, ""],
    raw_left: Float64[Array, ""],
    raw_right: Float64[Array, ""],
) -> Power2TailSpec:
    """PRIVATE: Construct the C1 ``power2`` tail contract from edge data.

    The amplitudes negate the sampled edge values. The linear
    coefficients match the one-sided edge slopes. The quadratic
    coefficients add a softplus margin above ``alpha**2 / 4`` per one
    squared eV, so the denominator never crosses zero.

    Parameters
    ----------
    edge_value_left : Float64[Array, ""]
        Dynamic imaginary part at the left domain edge in eV.
    slope_left : Float64[Array, ""]
        One-sided interpolant slope at the left edge.
    edge_value_right : Float64[Array, ""]
        Dynamic imaginary part at the right domain edge in eV.
    slope_right : Float64[Array, ""]
        One-sided interpolant slope at the right edge.
    raw_left : Float64[Array, ""]
        Unconstrained left raw delta-beta tail coordinate.
    raw_right : Float64[Array, ""]
        Unconstrained right raw delta-beta tail coordinate.

    Returns
    -------
    spec : Power2TailSpec
        Derived six-parameter tail contract.

    Notes
    -----
    The construction derives every parameter; no coefficient controls
    an edge value or an edge slope independently.
    """
    amplitude_left: Float64[Array, ""] = -edge_value_left
    amplitude_right: Float64[Array, ""] = -edge_value_right
    alpha_left: Float64[Array, ""] = -slope_left / amplitude_left
    alpha_right: Float64[Array, ""] = slope_right / amplitude_right
    beta_left: Float64[Array, ""] = alpha_left**2 / 4.0 + jnp.logaddexp(
        raw_left, 0.0
    )
    beta_right: Float64[Array, ""] = alpha_right**2 / 4.0 + jnp.logaddexp(
        raw_right, 0.0
    )
    spec: Power2TailSpec = make_power2_tail_spec(
        amplitude_left=amplitude_left,
        alpha_left=alpha_left,
        beta_left=beta_left,
        amplitude_right=amplitude_right,
        alpha_right=alpha_right,
        beta_right=beta_right,
    )
    return spec


def _power2_tail_side(
    edge: Float64[Array, ""],
    sign: float,
    amplitude: Float64[Array, ""],
    alpha: Float64[Array, ""],
    beta: Float64[Array, ""],
    queries: Float64[Array, " n_query"],
    positions: Float64[Array, " n_tail"],
    weights: Float64[Array, " n_tail"],
) -> Float64[Array, " n_query"]:
    """PRIVATE: Integrate one semi-infinite ``power2`` tail contribution.

    The rule maps the distance ``t = scale * u / (1 - u)`` onto the unit
    interval with its exact Jacobian. It then integrates the tail
    density against the principal-value kernel for every query.

    Parameters
    ----------
    edge : Float64[Array, ""]
        Core domain edge that anchors the tail in eV.
    sign : float
        Outward direction of the tail, ``-1.0`` left or ``1.0`` right.
    amplitude : Float64[Array, ""]
        Positive edge amplitude of the tail density in eV.
    alpha : Float64[Array, ""]
        Linear tail denominator coefficient.
    beta : Float64[Array, ""]
        Positive quadratic tail denominator coefficient.
    queries : Float64[Array, " n_query"]
        Query energies inside the trusted interval in eV.
    positions : Float64[Array, " n_tail"]
        Gauss--Legendre nodes mapped onto the unit interval.
    weights : Float64[Array, " n_tail"]
        Gauss--Legendre weights scaled onto the unit interval.

    Returns
    -------
    contribution : Float64[Array, " n_query"]
        Signed tail contribution to the unsubtracted real part.

    Notes
    -----
    The tail density reads ``-A / (1 + alpha*t + beta*t**2)`` at
    distance ``t`` outside the edge. The map ``t = scale*u/(1 - u)``
    compactifies the half line onto the unit interval.
    """
    scale: Float64[Array, ""] = beta**-0.5
    one_minus_u: Float64[Array, " n_tail"] = 1.0 - positions
    distance: Float64[Array, " n_tail"] = scale * positions / one_minus_u
    jacobian: Float64[Array, " n_tail"] = scale / one_minus_u**2
    sigma_imag: Float64[Array, " n_tail"] = -amplitude / (
        1.0 + alpha * distance + beta * distance**2
    )
    denominator: Float64[Array, "n_query n_tail"] = (
        edge + sign * distance - queries[..., None]
    )
    integrand: Float64[Array, "n_query n_tail"] = (
        sigma_imag * jacobian / (jnp.pi * denominator)
    )
    contribution: Float64[Array, " n_query"] = jnp.sum(
        weights * integrand, axis=-1
    )
    return contribution


def _power2_tail_pv(
    model_domain: Float64[Array, " 2"],
    tail_spec: Any,
    queries: Float64[Array, " n_query"],
    n_tail: int,
) -> Float64[Array, " n_query"]:
    """PRIVATE: Evaluate both semi-infinite ``power2`` tail quadratures.

    The rule never truncates a tail. Both continuations use the same
    fixed-order Gauss--Legendre rule under the rational map
    ``t = u / (1 - u)`` scaled by ``beta**-0.5``.

    Parameters
    ----------
    model_domain : Float64[Array, " 2"]
        Increasing carrier domain ``[a, b]`` in eV.
    tail_spec : Any
        Six-attribute left-then-right ``power2`` tail contract.
    queries : Float64[Array, " n_query"]
        Query energies inside the trusted interval in eV.
    n_tail : int
        Positive number of tail quadrature nodes per side.

    Returns
    -------
    total : Float64[Array, " n_query"]
        Unsubtracted tail contribution at every query.

    Raises
    ------
    ValueError
        If ``n_tail`` truncates the semi-infinite tail quadrature.

    Notes
    -----
    The rule builds the static Gauss--Legendre nodes with NumPy and
    sums both signed side integrals per query.
    """
    if n_tail <= 0:
        msg: str = (
            "n_tail must stay positive; a zero order truncates the "
            "semi-infinite tail quadrature"
        )
        raise ValueError(msg)
    gauss_nodes: Float64[NDArray, " n_tail"]
    gauss_weights: Float64[NDArray, " n_tail"]
    gauss_nodes, gauss_weights = np.polynomial.legendre.leggauss(n_tail)
    positions: Float64[Array, " n_tail"] = jnp.asarray(
        (gauss_nodes + 1.0) / 2.0, dtype=jnp.float64
    )
    weights: Float64[Array, " n_tail"] = jnp.asarray(
        gauss_weights / 2.0, dtype=jnp.float64
    )
    left: Float64[Array, " n_query"] = _power2_tail_side(
        model_domain[0],
        -1.0,
        tail_spec.amplitude_left,
        tail_spec.alpha_left,
        tail_spec.beta_left,
        queries,
        positions,
        weights,
    )
    right: Float64[Array, " n_query"] = _power2_tail_side(
        model_domain[1],
        1.0,
        tail_spec.amplitude_right,
        tail_spec.alpha_right,
        tail_spec.beta_right,
        queries,
        positions,
        weights,
    )
    total: Float64[Array, " n_query"] = left + right
    return total


def _cubic_core_pv(
    core_grid: Float64[Array, " n_kk"],
    core_values: Float64[Array, " n_kk"],
    queries: Float64[Array, " n_query"],
) -> Float64[Array, " n_query"]:
    """PRIVATE: Evaluate the cell-integrated piecewise-cubic core PV.

    Each cell integrates the cubic through the four-node stencil that
    starts at ``clip(i - 1, 0, n_kk - 4)``. Interior cells use the
    centered stencil. The first and last cells clamp to the one-sided
    edge stencils. After query-value subtraction, each cell has the
    closed-form principal-value integral. The node logarithms regroup by
    grid node, so interior coefficients cancel exactly. A double
    ``where`` guards node-coincident queries and their reverse-mode
    logarithm derivatives.

    Parameters
    ----------
    core_grid : Float64[Array, " n_kk"]
        Uniform core grid on the declared domain in eV.
    core_values : Float64[Array, " n_kk"]
        Sampled dynamic imaginary part on the core grid in eV.
    queries : Float64[Array, " n_query"]
        Query energies in eV.

    Returns
    -------
    core : Float64[Array, " n_query"]
        Unsubtracted core contribution at every query.

    Raises
    ------
    ValueError
        If the core grid has fewer than four nodes.

    Notes
    -----
    For a query ``q`` each cell interpolant reads ``C(u)`` in
    ``u = w - q``. Its exact principal value combines polynomial
    differences with regrouped node logarithms.
    """
    n_kk: int = core_grid.shape[0]
    minimum_nodes: int = 4
    if n_kk < minimum_nodes:
        msg: str = "the piecewise-cubic core grid requires at least four nodes"
        raise ValueError(msg)
    spacing: Float64[Array, ""] = core_grid[1] - core_grid[0]
    cell_indices: Int64[Array, " n_cell"] = jnp.arange(n_kk - 1)
    stencil_starts: Int64[Array, " n_cell"] = jnp.clip(
        cell_indices - 1, 0, n_kk - 4
    )

    y0: Float64[Array, " n_cell"] = core_values[stencil_starts]
    y1: Float64[Array, " n_cell"] = core_values[stencil_starts + 1]
    y2: Float64[Array, " n_cell"] = core_values[stencil_starts + 2]
    y3: Float64[Array, " n_cell"] = core_values[stencil_starts + 3]
    x0: Float64[Array, " n_cell"] = core_grid[stencil_starts]

    linear: Float64[Array, " n_cell"] = (
        -11.0 * y0 + 18.0 * y1 - 9.0 * y2 + 2.0 * y3
    ) / (6.0 * spacing)
    quadratic: Float64[Array, " n_cell"] = (
        2.0 * y0 - 5.0 * y1 + 4.0 * y2 - y3
    ) / (2.0 * spacing**2)
    cubic: Float64[Array, " n_cell"] = (-y0 + 3.0 * y1 - 3.0 * y2 + y3) / (
        6.0 * spacing**3
    )

    offset: Float64[Array, "n_query n_cell"] = queries[:, None] - x0[None, :]
    u_left: Float64[Array, "n_query n_cell"] = (
        core_grid[:-1][None, :] - queries[:, None]
    )
    u_right: Float64[Array, "n_query n_cell"] = (
        core_grid[1:][None, :] - queries[:, None]
    )
    q1: Float64[Array, "n_query n_cell"] = (
        linear[None, :]
        + 2.0 * quadratic[None, :] * offset
        + 3.0 * cubic[None, :] * offset**2
    )
    regular: Float64[Array, "n_query n_cell"] = q1 * (u_right - u_left)
    del q1
    q2: Float64[Array, "n_query n_cell"] = (
        quadratic[None, :] + 3.0 * cubic[None, :] * offset
    )
    regular += 0.5 * q2 * (u_right**2 - u_left**2)
    del q2
    regular += (cubic[None, :] / 3.0) * (u_right**3 - u_left**3)
    del u_left, u_right
    regular_sum: Float64[Array, " n_query"] = jnp.sum(regular, axis=1)
    del regular

    q0: Float64[Array, "n_query n_cell"] = (
        y0[None, :]
        + linear[None, :] * offset
        + quadratic[None, :] * offset**2
        + cubic[None, :] * offset**3
    )
    del offset
    log_coefficients: Float64[Array, "n_query n_kk"] = jnp.concatenate(
        (
            -q0[:, :1],
            q0[:, :-1] - q0[:, 1:],
            q0[:, -1:],
        ),
        axis=1,
    )
    del q0
    distances: Float64[Array, "n_query n_kk"] = (
        core_grid[None, :] - queries[:, None]
    )
    at_node: Any = distances == 0.0
    log_coefficients = jnp.where(at_node, 0.0, log_coefficients)
    safe_distances: Float64[Array, "n_query n_kk"] = jnp.where(
        at_node, 1.0, distances
    )
    del distances, at_node
    logarithmic: Float64[Array, " n_query"] = jnp.sum(
        log_coefficients * jnp.log(jnp.abs(safe_distances)), axis=1
    )
    core: Float64[Array, " n_query"] = (regular_sum + logarithmic) / jnp.pi
    return core


def _hat_core_pv(
    nodes: Float64[Array, " n_nodes"],
    ordinates: Float64[Array, " n_nodes"],
    queries: Float64[Array, " n_query"],
) -> Float64[Array, " n_query"]:
    """PRIVATE: Evaluate the exact cell-integrated piecewise-linear PV.

    Each linear cell has the closed-form principal-value integral. The
    node logarithms regroup by grid node, so hat-interpolant
    coefficients cancel exactly at interior on-node queries. The
    transform is exact for the grid-mode hat carrier class.

    Parameters
    ----------
    nodes : Float64[Array, " n_nodes"]
        Strictly increasing carrier nodes in eV.
    ordinates : Float64[Array, " n_nodes"]
        Negative hat ordinates of the imaginary part in eV.
    queries : Float64[Array, " n_query"]
        Query energies in eV.

    Returns
    -------
    core : Float64[Array, " n_query"]
        Unsubtracted core contribution at every query.

    Notes
    -----
    Each segment contributes ``m*(x1 - x0)`` plus the extended
    interpolant times the segment logarithm. The regrouping sums one
    logarithm per node with exactly cancelling hat coefficients.
    """
    cell_widths: Float64[Array, " n_cell"] = jnp.diff(nodes)
    slopes: Float64[Array, " n_cell"] = jnp.diff(ordinates) / cell_widths
    query_offsets: Float64[Array, "n_query n_nodes"] = (
        queries[:, None] - nodes[None, :]
    )
    left_coefficients: Float64[Array, " n_query"] = -(
        ordinates[0] + slopes[0] * query_offsets[:, 0]
    )
    interior_coefficients: Float64[Array, "n_query n_interior"] = (
        slopes[:-1] - slopes[1:]
    )[None, :] * query_offsets[:, 1:-1]
    right_coefficients: Float64[Array, " n_query"] = (
        ordinates[-1] + slopes[-1] * query_offsets[:, -1]
    )
    log_coefficients: Float64[Array, "n_query n_nodes"] = jnp.concatenate(
        (
            left_coefficients[:, None],
            interior_coefficients,
            right_coefficients[:, None],
        ),
        axis=1,
    )
    distances: Float64[Array, "n_query n_nodes"] = (
        nodes[None, :] - queries[:, None]
    )
    at_node: Any = distances == 0.0
    log_coefficients = jnp.where(at_node, 0.0, log_coefficients)
    safe_distances: Float64[Array, "n_query n_nodes"] = jnp.where(
        at_node, 1.0, distances
    )
    log_terms: Float64[Array, "n_query n_nodes"] = log_coefficients * jnp.log(
        jnp.abs(safe_distances)
    )
    slope_integrals: Float64[Array, ""] = jnp.sum(slopes * cell_widths)
    core: Float64[Array, " n_query"] = (
        slope_integrals + jnp.sum(log_terms, axis=1)
    ) / jnp.pi
    return core


def _cubic_edge_slopes(
    values: Float64[Array, " n_kk"],
    spacing: Float64[Array, ""],
) -> Tuple[Float64[Array, ""], Float64[Array, ""]]:
    """PRIVATE: Evaluate the clamped one-sided cubic edge-stencil slopes.

    The one-sided four-node stencils supply the derivative of the core
    interpolant at both domain edges. The tails reuse these slopes for
    the C1 match.

    Parameters
    ----------
    values : Float64[Array, " n_kk"]
        Sampled dynamic imaginary part on the core grid in eV.
    spacing : Float64[Array, ""]
        Uniform core grid spacing in eV.

    Returns
    -------
    slopes : Tuple[Float64[Array, ""], Float64[Array, ""]]
        Left and right one-sided interpolant slopes.

    Notes
    -----
    The rows read ``(-11, 18, -9, 2) / (6*h)`` forward at the left
    edge and its mirrored negation at the right edge.
    """
    left: Float64[Array, ""] = (
        -11.0 * values[0]
        + 18.0 * values[1]
        - 9.0 * values[2]
        + 2.0 * values[3]
    ) / (6.0 * spacing)
    right: Float64[Array, ""] = (
        11.0 * values[-1]
        - 18.0 * values[-2]
        + 9.0 * values[-3]
        - 2.0 * values[-4]
    ) / (6.0 * spacing)
    slopes: Tuple[Float64[Array, ""], Float64[Array, ""]] = (left, right)
    return slopes


def _derivative_samples_sixth_order(
    values: Float64[Array, " n_kk"],
    spacing: Float64[Array, ""],
) -> Float64[Array, " n_kk"]:
    """PRIVATE: Differentiate uniform-grid samples with sixth-order stencils.

    Interior nodes use the centered seven-node sixth-order first
    derivative. The three nodes at each edge use one-sided seven-node
    rows from an exact Vandermonde solve. The mode-agnostic transform
    seam consumes these samples for its composite query derivative.

    Parameters
    ----------
    values : Float64[Array, " n_kk"]
        Sampled function values on the uniform grid.
    spacing : Float64[Array, ""]
        Uniform grid spacing in eV.

    Returns
    -------
    derivative : Float64[Array, " n_kk"]
        Sixth-order derivative samples at every grid node.

    Raises
    ------
    ValueError
        If the grid has fewer than seven nodes.

    Notes
    -----
    The interior weights read ``(-1, 9, -45, 0, 45, -9, 1) / (60*h)``.
    A seven-node Vandermonde solve supplies each one-sided edge row.
    """
    minimum_nodes: int = 7
    if values.shape[0] < minimum_nodes:
        msg: str = (
            "sixth-order derivative samples require at least seven nodes"
        )
        raise ValueError(msg)
    stencil_offsets: Float64[NDArray, " seven"] = np.arange(
        7, dtype=np.float64
    )
    edge_rows: List[Float64[NDArray, " seven"]] = []
    position: int
    for position in range(3):
        system: Float64[NDArray, "seven seven"] = np.vander(
            stencil_offsets - position, 7, increasing=True
        ).T
        target: Float64[NDArray, " seven"] = np.zeros(7, dtype=np.float64)
        target[1] = 1.0
        edge_rows.append(np.linalg.solve(system, target))
    edge_matrix: Float64[Array, "three seven"] = jnp.asarray(
        np.stack(edge_rows), dtype=jnp.float64
    )
    head: Float64[Array, " three"] = edge_matrix @ values[:7]
    tail_reversed: Float64[Array, " three"] = -(
        edge_matrix[:, ::-1] @ values[-7:]
    )
    interior: Float64[Array, " n_interior"] = (
        -values[:-6]
        + 9.0 * values[1:-5]
        - 45.0 * values[2:-4]
        + 45.0 * values[4:-2]
        - 9.0 * values[5:-1]
        + values[6:]
    ) / 60.0
    derivative: Float64[Array, " n_kk"] = (
        jnp.concatenate([head, interior, tail_reversed[::-1]]) / spacing
    )
    return derivative


def _check_frozen_core_grid(  # noqa: DOC502 -- eqx.error_if raises under JAX execution.
    positions: Float64[Array, " n_kk"],
    model_domain: Float64[Array, " 2"],
) -> Float64[Array, " n_kk"]:
    """PRIVATE: Reject a quadrature grid that departs from the carrier domain.

    The check compares both grid endpoints and the uniform spacing with
    the frozen index construction on the declared domain. A grid that
    spans the query window instead of the carrier domain fails here.

    Parameters
    ----------
    positions : Float64[Array, " n_kk"]
        Candidate core quadrature grid in eV.
    model_domain : Float64[Array, " 2"]
        Increasing carrier domain ``[a, b]`` in eV.

    Returns
    -------
    checked : Float64[Array, " n_kk"]
        Grid with the attached runtime predicate.

    Raises
    ------
    EquinoxRuntimeError
        If the grid does not span the declared model domain.

    Notes
    -----
    The predicate allows one relative part in ``1e9`` of roundoff on
    the endpoints and on the uniform spacing.
    """
    width: Float64[Array, ""] = model_domain[1] - model_domain[0]
    expected_spacing: Float64[Array, ""] = width / (positions.shape[0] - 1)
    edge_tolerance: Float64[Array, ""] = 1.0e-9 * width
    spacing_tolerance: Float64[Array, ""] = 1.0e-9 * expected_spacing
    bad: Any = (
        (jnp.abs(positions[0] - model_domain[0]) > edge_tolerance)
        | (jnp.abs(positions[-1] - model_domain[1]) > edge_tolerance)
        | (
            jnp.max(jnp.abs(jnp.diff(positions) - expected_spacing))
            > spacing_tolerance
        )
    )
    checked: Float64[Array, " n_kk"] = eqx.error_if(
        positions,
        bad,
        "the core grid must span the declared model domain with the "
        "frozen uniform index construction; a query-window grid is "
        "noncompliant",
    )
    return checked


def _check_trusted_interval(  # noqa: DOC502 -- eqx.error_if raises under JAX execution.
    queries: Float64[Array, " n_query"],
    model_domain: Float64[Array, " 2"],
    spacing: Float64[Array, ""],
) -> Float64[Array, " n_query"]:
    """PRIVATE: Reject queries outside the trusted interval.

    The check runs eagerly and inside compiled code. It covers every
    stacked evaluation point, so the subtraction point obeys the same
    trusted-interval contract as the queries.

    Parameters
    ----------
    queries : Float64[Array, " n_query"]
        Stacked query energies in eV.
    model_domain : Float64[Array, " 2"]
        Increasing carrier domain ``[a, b]`` in eV.
    spacing : Float64[Array, ""]
        Frozen base grid spacing ``h`` in eV.

    Returns
    -------
    checked : Float64[Array, " n_query"]
        Queries with the attached runtime predicate.

    Raises
    ------
    EquinoxRuntimeError
        If one point leaves the trusted interval.

    Notes
    -----
    The margin of two spacings keeps every query away from the edge
    stencils and the tail seams.
    """
    lower: Float64[Array, ""] = model_domain[0] + 2.0 * spacing
    upper: Float64[Array, ""] = model_domain[1] - 2.0 * spacing
    bad: Any = (jnp.min(queries) < lower) | (jnp.max(queries) > upper)
    checked: Float64[Array, " n_query"] = eqx.error_if(
        queries,
        bad,
        "queries and the subtraction point must lie inside the trusted "
        "interval [a + 2h, b - 2h] of the declared domain",
    )
    return checked


def _check_tail_spec(  # noqa: DOC502 -- eqx.error_if raises under JAX execution.
    values: Float64[Array, " n_kk"],
    spacing: Float64[Array, ""],
    tail_spec: Any,
) -> Float64[Array, " n_kk"]:
    """PRIVATE: Reject a tail contract that breaks the C1 or positivity rules.

    The check compares both tail amplitudes with the sampled core edge
    values. It compares both linear coefficients with the clamped cubic
    edge-stencil slopes. It finally requires quadratic coefficients that
    keep every tail denominator strictly positive.

    Parameters
    ----------
    values : Float64[Array, " n_kk"]
        Sampled dynamic imaginary part on the core grid in eV.
    spacing : Float64[Array, ""]
        Uniform core grid spacing in eV.
    tail_spec : Any
        Six-attribute left-then-right ``power2`` tail contract.

    Returns
    -------
    checked : Float64[Array, " n_kk"]
        Core samples with the attached runtime predicates.

    Raises
    ------
    EquinoxRuntimeError
        If the tail contract violates continuity or positivity.

    Notes
    -----
    The predicates run through value-threaded ``eqx.error_if`` calls,
    so each check stays active inside compiled code.
    """
    tolerance: float = 1.0e-9
    slope_left: Float64[Array, ""]
    slope_right: Float64[Array, ""]
    slope_left, slope_right = _cubic_edge_slopes(values, spacing)
    edge_gap: Any = (
        jnp.abs(tail_spec.amplitude_left + values[0])
        > tolerance * jnp.maximum(1.0, jnp.abs(values[0]))
    ) | (
        jnp.abs(tail_spec.amplitude_right + values[-1])
        > tolerance * jnp.maximum(1.0, jnp.abs(values[-1]))
    )
    edge_bad: Any = edge_gap | (
        (tail_spec.amplitude_left <= 0.0) | (tail_spec.amplitude_right <= 0.0)
    )
    checked: Float64[Array, " n_kk"] = eqx.error_if(
        values,
        edge_bad,
        "each tail edge amplitude must stay strictly positive and "
        "match the sampled core edge value",
    )
    slope_bad: Any = (
        jnp.abs(tail_spec.alpha_left * tail_spec.amplitude_left + slope_left)
        > tolerance * jnp.maximum(1.0, jnp.abs(slope_left))
    ) | (
        jnp.abs(
            tail_spec.alpha_right * tail_spec.amplitude_right - slope_right
        )
        > tolerance * jnp.maximum(1.0, jnp.abs(slope_right))
    )
    checked = eqx.error_if(
        checked,
        slope_bad,
        "each tail edge slope must match the clamped cubic edge stencil "
        "of the core interpolant",
    )
    beta_bad: Any = (
        (tail_spec.beta_left <= 0.0)
        | (tail_spec.beta_right <= 0.0)
        | (tail_spec.beta_left < tail_spec.alpha_left**2 / 4.0)
        | (tail_spec.beta_right < tail_spec.alpha_right**2 / 4.0)
    )
    validated: Float64[Array, " n_kk"] = eqx.error_if(
        checked,
        beta_bad,
        "each tail denominator must stay strictly positive at every distance",
    )
    return validated


__all__: list[str] = []
