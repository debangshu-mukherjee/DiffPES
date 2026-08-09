r"""Evaluate the complex retarded self-energy through the certified KK map.

Extended Summary
----------------
The module evaluates :math:`\Sigma(E - E_F)` for every causal carrier
mode. The real part follows the once-subtracted retarded
Kramers--Kronig convention

.. math::

    \Sigma'(\omega) - \Sigma'(\omega_s) = \frac{1}{\pi}\,\mathrm{PV}
    \int \Sigma''(w)\left[\frac{1}{w - \omega}
    - \frac{1}{w - \omega_s}\right]\,dw .

The retarded pole :math:`\Sigma'' = -g\Gamma/((\omega - \omega_0)^2 +
\Gamma^2)` maps to :math:`\Sigma' = g(\omega - \omega_0)/((\omega -
\omega_0)^2 + \Gamma^2)` under this convention, which fixes the sign.

The committed operator is the static cell-integrated principal-value
family on the frozen index grid :math:`x_j = a + jh`. Grid mode uses the
exact piecewise-linear hat transform on the carrier nodes. The smooth
modes ``poly`` and ``fermi_liquid`` use the piecewise-cubic transform
with clamped one-sided edge stencils. Both attach C1 ``power2`` tails
under a 256-node semi-infinite Gauss--Legendre rule. The ``constant``
and ``bosonic_kink`` modes evaluate analytic closed forms. Queries and
the subtraction point must stay inside the trusted interval
:math:`[a + 2h,\, b - 2h]`.

Routine Listings
----------------
:func:`evaluate_self_energy`
    Evaluate the complex retarded self-energy for one causal model.

Notes
-----
The public frequency derivative follows the composite route. The rule
applies the same principal-value operator to the analytic
:math:`\partial_\omega\Sigma''`. It then adds the finite-core boundary
terms and the exact semi-infinite tail derivatives.
"""

from functools import partial

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from beartype import beartype
from beartype.typing import Any, Optional, Tuple
from jax.custom_derivatives import SymbolicZero
from jaxtyping import Array, Complex128, Float64, Int, jaxtyped
from numpy.typing import NDArray

from diffpes.types import SelfEnergyModel


def _tangent_is_symbolic_zero(tangent: Any) -> bool:
    """PRIVATE: Report whether one tangent tree carries no perturbation.

    The custom derivative rules use this static predicate to skip the
    linearization of unperturbed argument groups.

    Parameters
    ----------
    tangent : Any
        Tangent tree from a ``symbolic_zeros`` custom derivative rule.

    Returns
    -------
    all_zero : bool
        Whether every tangent leaf is a symbolic zero.

    Notes
    -----
    The check flattens the tree with symbolic zeros as leaves.
    It then tests every leaf for the symbolic zero type.
    """
    leaves: list[Any] = jax.tree_util.tree_leaves(
        tangent, is_leaf=lambda value: isinstance(value, SymbolicZero)
    )
    all_zero: bool = all(isinstance(value, SymbolicZero) for value in leaves)
    return all_zero


def _materialize_tangent(tangent: Any) -> Any:
    """PRIVATE: Replace symbolic zero tangent leaves with explicit zero arrays.

    Parameters
    ----------
    tangent : Any
        Tangent tree from a ``symbolic_zeros`` custom derivative rule.

    Returns
    -------
    materialized : Any
        Tangent tree with every leaf as an ordinary array.

    Notes
    -----
    The map visits every leaf with symbolic zeros as leaves.
    It builds a matching zero array for each symbolic leaf.
    """

    def _leaf(value: Any) -> Any:
        """PRIVATE: Convert one symbolic zero leaf to an explicit zero array.

        Parameters
        ----------
        value : Any
            One tangent tree leaf.

        Returns
        -------
        value : Any
            Unchanged leaf or a matching explicit zero array.

        Notes
        -----
        The symbolic leaf exposes its shape and dtype directly.
        """
        if isinstance(value, SymbolicZero):
            zero_leaf: Float64[Array, "..."] = jnp.zeros(
                value.shape, value.dtype
            )
            return zero_leaf
        return value

    materialized: Any = jax.tree_util.tree_map(
        _leaf, tangent, is_leaf=lambda value: isinstance(value, SymbolicZero)
    )
    return materialized


class _Power2TailSpec(eqx.Module):
    """Store one C1 ``power2`` tail contract in left-then-right order.

    The carrier stores the six derived tail parameters for both
    semi-infinite continuations. The amplitudes stay strictly positive.
    The quadratic coefficients keep every tail denominator positive.

    Attributes
    ----------
    amplitude_left : Float64[Array, ""]
        Positive left edge amplitude ``-Sigma''(a)`` in eV.
    alpha_left : Float64[Array, ""]
        Left linear tail coefficient from the edge slope match.
    beta_left : Float64[Array, ""]
        Left quadratic tail coefficient above ``alpha_left**2 / 4``.
    amplitude_right : Float64[Array, ""]
        Positive right edge amplitude ``-Sigma''(b)`` in eV.
    alpha_right : Float64[Array, ""]
        Right linear tail coefficient from the edge slope match.
    beta_right : Float64[Array, ""]
        Right quadratic tail coefficient above ``alpha_right**2 / 4``.
    """

    amplitude_left: Float64[Array, ""]
    alpha_left: Float64[Array, ""]
    beta_left: Float64[Array, ""]
    amplitude_right: Float64[Array, ""]
    alpha_right: Float64[Array, ""]
    beta_right: Float64[Array, ""]


def _power2_spec_from_edges(
    edge_value_left: Float64[Array, ""],
    slope_left: Float64[Array, ""],
    edge_value_right: Float64[Array, ""],
    slope_right: Float64[Array, ""],
    raw_left: Float64[Array, ""],
    raw_right: Float64[Array, ""],
) -> _Power2TailSpec:
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
    spec : _Power2TailSpec
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
    spec: _Power2TailSpec = _Power2TailSpec(
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
    cell_indices: Int[Array, " n_cell"] = jnp.arange(n_kk - 1)
    stencil_starts: Int[Array, " n_cell"] = jnp.clip(
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
    edge_rows: list[Float64[NDArray, " seven"]] = []
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


def _kk_transform_impl(  # noqa: DOC502 -- eqx.error_if raises under JAX execution.
    core_grid: Tuple[Float64[Array, " n_kk"], Float64[Array, " n_kk"]],
    model_domain: Float64[Array, " 2"],
    tail_spec: Any,
    queries: Float64[Array, " n_query"],
    n_tail: int,
) -> Float64[Array, " n_query"]:
    """PRIVATE: Evaluate the validated transform without a rule.

    The routine validates the grid, the trusted interval, and the tail
    contract. It then adds the piecewise-cubic core principal value and
    both semi-infinite tail quadratures at every query.

    Parameters
    ----------
    core_grid : Tuple[Float64[Array, " n_kk"], Float64[Array, " n_kk"]]
        Pair of frozen node positions and sampled imaginary values.
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
    transformed : Float64[Array, " n_query"]
        Unsubtracted principal-value transform at every query.

    Raises
    ------
    ValueError
        If ``n_tail`` truncates the tails or the grid pair shapes
        disagree.
    EquinoxRuntimeError
        If a traced grid, query, or tail predicate fails.

    Notes
    -----
    The custom-rule wrapper shares this body, so the primal and the
    derivative rule evaluate identical validated values.
    """
    positions: Float64[Array, " n_kk"]
    values: Float64[Array, " n_kk"]
    positions, values = core_grid
    if n_tail <= 0:
        msg: str = (
            "n_tail must stay positive; a zero order truncates the "
            "semi-infinite tail quadrature"
        )
        raise ValueError(msg)
    if positions.ndim != 1 or positions.shape != values.shape:
        msg = (
            "core_grid must pair one grid vector with one matching "
            "value vector"
        )
        raise ValueError(msg)
    checked_positions: Float64[Array, " n_kk"] = _check_frozen_core_grid(
        positions, model_domain
    )
    spacing: Float64[Array, ""] = (model_domain[1] - model_domain[0]) / (
        positions.shape[0] - 1
    )
    checked_values: Float64[Array, " n_kk"] = _check_tail_spec(
        values, spacing, tail_spec
    )
    checked_queries: Float64[Array, " n_query"] = _check_trusted_interval(
        queries, model_domain, spacing
    )
    core: Float64[Array, " n_query"] = _cubic_core_pv(
        checked_positions, checked_values, checked_queries
    )
    tails: Float64[Array, " n_query"] = _power2_tail_pv(
        model_domain, tail_spec, checked_queries, n_tail
    )
    transformed: Float64[Array, " n_query"] = core + tails
    return transformed


def _seam_query_composite(
    core_grid: Tuple[Float64[Array, " n_kk"], Float64[Array, " n_kk"]],
    model_domain: Float64[Array, " 2"],
    tail_spec: Any,
    queries: Float64[Array, " n_query"],
    n_tail: int,
) -> Float64[Array, " n_query"]:
    """PRIVATE: Evaluate the composite seam derivative from grid samples alone.

    The mode-agnostic seam has no analytic derivative source. The rule
    therefore differentiates the samples with sixth-order stencils. It
    applies the same core operator to those samples, adds the boundary
    terms, and adds the exact tail derivatives.

    Parameters
    ----------
    core_grid : Tuple[Float64[Array, " n_kk"], Float64[Array, " n_kk"]]
        Pair of frozen node positions and sampled imaginary values.
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
    composite : Float64[Array, " n_query"]
        Composite query derivative of the unsubtracted transform.

    Notes
    -----
    The derivative of the transform equals the transform of the
    derivative plus boundary terms, by partial integration on the
    finite core.
    """
    positions: Float64[Array, " n_kk"]
    values: Float64[Array, " n_kk"]
    positions, values = core_grid
    spacing: Float64[Array, ""] = (model_domain[1] - model_domain[0]) / (
        positions.shape[0] - 1
    )
    derivative_samples: Float64[Array, " n_kk"] = (
        _derivative_samples_sixth_order(values, spacing)
    )
    core_derivative: Float64[Array, " n_query"] = _cubic_core_pv(
        positions, derivative_samples, queries
    )
    boundary: Float64[Array, " n_query"] = (
        values[0] / (model_domain[0] - queries)
        - values[-1] / (model_domain[1] - queries)
    ) / jnp.pi

    def _tail_only(
        points: Float64[Array, " n_query"],
    ) -> Float64[Array, " n_query"]:
        """PRIVATE: Evaluate both tail quadratures for the derivative closure.

        Parameters
        ----------
        points : Float64[Array, " n_query"]
            Query energies in eV.

        Returns
        -------
        contribution : Float64[Array, " n_query"]
            Unsubtracted tail contribution at every query.

        Notes
        -----
        Forward-mode differentiation of this closure supplies the
        exact tail derivative.
        """
        contribution: Float64[Array, " n_query"] = _power2_tail_pv(
            model_domain, tail_spec, points, n_tail
        )
        return contribution

    tail_derivative: Float64[Array, " n_query"]
    _, tail_derivative = jax.jvp(
        _tail_only, (queries,), (jnp.ones_like(queries),)
    )
    composite: Float64[Array, " n_query"] = (
        core_derivative + boundary + tail_derivative
    )
    return composite


@partial(jax.custom_jvp, nondiff_argnums=(4,))
def _kk_transform(  # noqa: DOC502 -- eqx.error_if raises under JAX execution.
    core_grid: Tuple[Float64[Array, " n_kk"], Float64[Array, " n_kk"]],
    model_domain: Float64[Array, " 2"],
    tail_spec: Any,
    queries: Float64[Array, " n_query"],
    n_tail: int,
) -> Float64[Array, " n_query"]:
    """PRIVATE: Evaluate the cell-integrated principal-value transform seam.

    The seam performs direct per-query evaluation with no kernel matrix
    and no post-transform interpolation. Its custom derivative rule
    routes query tangents through the composite derivative and keeps
    every other tangent on the primal linearization.

    :see: :class:`~.test_spectral.TestKkTransformSeam`
    :see: :class:`~.test_spectral.TestPlantedNoncompliantConstructions`

    Parameters
    ----------
    core_grid : Tuple[Float64[Array, " n_kk"], Float64[Array, " n_kk"]]
        Pair of frozen node positions and sampled imaginary values.
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
    transformed : Float64[Array, " n_query"]
        Unsubtracted principal-value transform at every query.

    Raises
    ------
    ValueError
        If ``n_tail`` truncates the tails or the grid pair shapes
        disagree.
    EquinoxRuntimeError
        If a traced grid, query, or tail predicate fails.

    Notes
    -----
    The wrapper defers to the shared implementation body. Only the
    attached derivative rule distinguishes it from the plain call.
    """
    transformed: Float64[Array, " n_query"] = _kk_transform_impl(
        core_grid, model_domain, tail_spec, queries, n_tail
    )
    return transformed


@partial(_kk_transform.defjvp, symbolic_zeros=True)
def _kk_transform_jvp(
    n_tail: int,
    primals: Any,
    tangents: Any,
) -> Tuple[Float64[Array, " n_query"], Float64[Array, " n_query"]]:
    """PRIVATE: Dispatch seam tangents through the composite contract.

    Query tangents multiply the composite seam derivative. Grid, domain,
    and tail tangents pass through the linearized primal, which stays
    exact because the transform is linear in the samples. Symbolic-zero
    detection skips every unperturbed argument group.

    Parameters
    ----------
    n_tail : int
        Positive number of tail quadrature nodes per side.
    primals : Any
        Seam primal inputs ``(core_grid, model_domain, tail_spec,
        queries)``.
    tangents : Any
        Matching tangent structure for the seam primal inputs.

    Returns
    -------
    pair : Tuple[Float64[Array, " n_query"], Float64[Array, " n_query"]]
        Primal seam output and its tangent.

    Notes
    -----
    The rule stays linear in every tangent, so the transform transpose
    supplies reverse mode without a separate rule.
    """
    core_grid: Tuple[Float64[Array, " n_kk"], Float64[Array, " n_kk"]]
    model_domain: Float64[Array, " 2"]
    tail_spec: Any
    queries: Float64[Array, " n_query"]
    core_grid, model_domain, tail_spec, queries = primals
    core_tangent: Any
    domain_tangent: Any
    spec_tangent: Any
    query_tangent: Any
    core_tangent, domain_tangent, spec_tangent, query_tangent = tangents
    primal_out: Float64[Array, " n_query"] = _kk_transform_impl(
        core_grid, model_domain, tail_spec, queries, n_tail
    )
    tangent_out: Float64[Array, " n_query"] = jnp.zeros_like(primal_out)

    def _fixed_queries(
        grid_pair: Tuple[Float64[Array, " n_kk"], Float64[Array, " n_kk"]],
        domain: Float64[Array, " 2"],
        spec: Any,
    ) -> Float64[Array, " n_query"]:
        """PRIVATE: Evaluate the transform with the query set held fixed.

        Parameters
        ----------
        grid_pair : Tuple[Float64[Array, " n_kk"], Float64[Array, " n_kk"]]
            Pair of node positions and sampled imaginary values.
        domain : Float64[Array, " 2"]
            Increasing carrier domain ``[a, b]`` in eV.
        spec : Any
            Six-attribute ``power2`` tail contract.

        Returns
        -------
        value : Float64[Array, " n_query"]
            Unsubtracted transform at the closed-over queries.

        Notes
        -----
        Linearizing this closure yields the exact sample, domain, and
        tail tangents.
        """
        value: Float64[Array, " n_query"] = _kk_transform_impl(
            grid_pair, domain, spec, queries, n_tail
        )
        return value

    linear_perturbed: bool = not (
        _tangent_is_symbolic_zero(core_tangent)
        and _tangent_is_symbolic_zero(domain_tangent)
        and _tangent_is_symbolic_zero(spec_tangent)
    )
    if linear_perturbed:
        linear_tangent: Float64[Array, " n_query"]
        _, linear_tangent = jax.jvp(
            _fixed_queries,
            (core_grid, model_domain, tail_spec),
            (
                _materialize_tangent(core_tangent),
                _materialize_tangent(domain_tangent),
                _materialize_tangent(spec_tangent),
            ),
        )
        tangent_out = tangent_out + linear_tangent
    if not _tangent_is_symbolic_zero(query_tangent):
        composite: Float64[Array, " n_query"] = _seam_query_composite(
            core_grid, model_domain, tail_spec, queries, n_tail
        )
        tangent_out = tangent_out + composite * query_tangent
    pair: Tuple[Float64[Array, " n_query"], Float64[Array, " n_query"]] = (
        primal_out,
        tangent_out,
    )
    return pair


def _dynamic_imag(
    mode: str,
    coefficients: Float64[Array, " n_coef"],
    points: Float64[Array, " n_points"],
) -> Float64[Array, " n_points"]:
    """PRIVATE: Evaluate the transform-side dynamic imaginary part.

    The Fermi-liquid mode excludes its constant baseline, because a
    constant has a vanishing subtracted transform. The polynomial mode
    keeps its complete strictly negative softplus profile.

    Parameters
    ----------
    mode : str
        Smooth carrier mode, ``poly`` or ``fermi_liquid``.
    coefficients : Float64[Array, " n_coef"]
        Unconstrained raw model coordinates.
    points : Float64[Array, " n_points"]
        Evaluation energies in eV.

    Returns
    -------
    dynamic : Float64[Array, " n_points"]
        Dynamic imaginary part at every point in eV.

    Raises
    ------
    ValueError
        If the mode has no numerical transform contract.

    Notes
    -----
    The softplus map keeps every profile strictly negative, so the
    tail amplitudes stay strictly positive at both edges.
    """
    if mode == "fermi_liquid":
        beta: Float64[Array, ""] = jnp.logaddexp(coefficients[1], 0.0)
        omega_c: Float64[Array, ""] = jnp.logaddexp(coefficients[2], 0.0)
        dynamic: Float64[Array, " n_points"] = (
            -beta * points**2 / (1.0 + (points / omega_c) ** 4)
        )
    elif mode == "poly":
        dynamic = -jnp.logaddexp(jnp.polyval(coefficients, points), 0.0)
    else:
        msg: str = f"mode {mode!r} has no numerical transform contract"
        raise ValueError(msg)
    return dynamic


def _dynamic_imag_derivative(
    mode: str,
    coefficients: Float64[Array, " n_coef"],
    points: Float64[Array, " n_points"],
) -> Float64[Array, " n_points"]:
    """PRIVATE: Differentiate the dynamic imaginary part analytically.

    The composite frequency-derivative route consumes these analytic
    samples instead of differentiating the discrete interpolant.

    Parameters
    ----------
    mode : str
        Smooth carrier mode, ``poly`` or ``fermi_liquid``.
    coefficients : Float64[Array, " n_coef"]
        Unconstrained raw model coordinates.
    points : Float64[Array, " n_points"]
        Evaluation energies in eV.

    Returns
    -------
    derivative : Float64[Array, " n_points"]
        Analytic frequency derivative at every point.

    Raises
    ------
    ValueError
        If the mode has no numerical transform contract.

    Notes
    -----
    The Fermi-liquid branch evaluates
    ``2*beta*w*(q - 1)/(1 + q)**2`` with ``q = (w/omega_c)**4``. The
    polynomial branch chains the sigmoid with the derivative
    polynomial.
    """
    if mode == "fermi_liquid":
        beta: Float64[Array, ""] = jnp.logaddexp(coefficients[1], 0.0)
        omega_c: Float64[Array, ""] = jnp.logaddexp(coefficients[2], 0.0)
        quartic: Float64[Array, " n_points"] = (points / omega_c) ** 4
        derivative: Float64[Array, " n_points"] = (
            2.0 * beta * points * (quartic - 1.0) / (1.0 + quartic) ** 2
        )
    elif mode == "poly":
        profile: Float64[Array, " n_points"] = jnp.polyval(
            coefficients, points
        )
        degree: int = coefficients.shape[0] - 1
        if degree == 0:
            derivative = jnp.zeros_like(points)
        else:
            slope_coefficients: Float64[Array, " n_deriv"] = coefficients[
                :-1
            ] * jnp.arange(degree, 0, -1)
            derivative = -jax.nn.sigmoid(profile) * jnp.polyval(
                slope_coefficients, points
            )
    else:
        msg: str = f"mode {mode!r} has no numerical transform contract"
        raise ValueError(msg)
    return derivative


def _frozen_base_grid(
    model_domain: Float64[Array, " 2"],
    n_kk: int,
) -> Tuple[Float64[Array, " n_kk"], Float64[Array, ""]]:
    """PRIVATE: Construct the frozen uniform base grid on the carrier domain.

    The grid follows the index construction ``x_j = a + j * h`` with
    ``h = (b - a) / (n_kk - 1)``. The construction never reads the
    query window.

    Parameters
    ----------
    model_domain : Float64[Array, " 2"]
        Increasing carrier domain ``[a, b]`` in eV.
    n_kk : int
        Static number of base grid nodes.

    Returns
    -------
    grid_and_spacing : Tuple[Float64[Array, " n_kk"], Float64[Array, ""]]
        Base grid nodes and the uniform spacing ``h``.

    Notes
    -----
    The spacing evaluates first, so refinements embed the base nodes
    through the shared index expression.
    """
    spacing: Float64[Array, ""] = (model_domain[1] - model_domain[0]) / (
        n_kk - 1
    )
    grid: Float64[Array, " n_kk"] = model_domain[0] + spacing * jnp.arange(
        n_kk, dtype=jnp.float64
    )
    grid_and_spacing: Tuple[Float64[Array, " n_kk"], Float64[Array, ""]] = (
        grid,
        spacing,
    )
    return grid_and_spacing


def _smooth_real_impl(
    mode: str,
    n_kk: int,
    omega_rel_fermi_ev: Float64[Array, " n_omega"],
    coefficients: Float64[Array, " n_coef"],
    tail_raw: Float64[Array, " 2"],
    subtraction: Float64[Array, ""],
    model_domain: Float64[Array, " 2"],
) -> Float64[Array, " n_omega"]:
    """PRIVATE: Evaluate the subtracted smooth-mode real part without a rule.

    The routine samples the dynamic imaginary part on the frozen grid.
    It derives the C1 tail contract from the cubic edge stencils and
    evaluates the transform at the stacked queries. The subtraction
    happens after the transform at the carrier subtraction point.

    Parameters
    ----------
    mode : str
        Smooth carrier mode, ``poly`` or ``fermi_liquid``.
    n_kk : int
        Static number of base grid nodes.
    omega_rel_fermi_ev : Float64[Array, " n_omega"]
        Query energies relative to the Fermi level in eV.
    coefficients : Float64[Array, " n_coef"]
        Unconstrained raw model coordinates.
    tail_raw : Float64[Array, " 2"]
        Raw delta-beta tail coordinates, left then right.
    subtraction : Float64[Array, ""]
        Carrier subtraction point in eV.
    model_domain : Float64[Array, " 2"]
        Increasing carrier domain ``[a, b]`` in eV.

    Returns
    -------
    real_subtracted : Float64[Array, " n_omega"]
        Subtracted real part at every query in eV.

    Notes
    -----
    The custom-rule wrapper shares this body, so the primal and the
    derivative rule evaluate identical validated values.
    """
    grid: Float64[Array, " n_kk"]
    spacing: Float64[Array, ""]
    grid, spacing = _frozen_base_grid(model_domain, n_kk)
    values: Float64[Array, " n_kk"] = _dynamic_imag(mode, coefficients, grid)
    slope_left: Float64[Array, ""]
    slope_right: Float64[Array, ""]
    slope_left, slope_right = _cubic_edge_slopes(values, spacing)
    spec: _Power2TailSpec = _power2_spec_from_edges(
        values[0],
        slope_left,
        values[-1],
        slope_right,
        tail_raw[0],
        tail_raw[1],
    )
    stacked: Float64[Array, " n_stacked"] = jnp.concatenate(
        [omega_rel_fermi_ev, subtraction[None]]
    )
    total: Float64[Array, " n_stacked"] = _kk_transform_impl(
        (grid, values), model_domain, spec, stacked, 256
    )
    real_subtracted: Float64[Array, " n_omega"] = total[:-1] - total[-1]
    return real_subtracted


def _smooth_query_composite(
    mode: str,
    n_kk: int,
    points: Float64[Array, " n_points"],
    coefficients: Float64[Array, " n_coef"],
    tail_raw: Float64[Array, " 2"],
    model_domain: Float64[Array, " 2"],
) -> Float64[Array, " n_points"]:
    """PRIVATE: Evaluate the analytic composite frequency derivative.

    The rule applies the core operator to the analytic mode-supplied
    derivative samples. It adds the finite-core boundary terms
    ``(1 / pi) * [Sigma''(a) / (a - w) - Sigma''(b) / (b - w)]``. It
    finally adds the exact forward-mode derivative of both tails.

    Parameters
    ----------
    mode : str
        Smooth carrier mode, ``poly`` or ``fermi_liquid``.
    n_kk : int
        Static number of base grid nodes.
    points : Float64[Array, " n_points"]
        Stacked evaluation energies in eV.
    coefficients : Float64[Array, " n_coef"]
        Unconstrained raw model coordinates.
    tail_raw : Float64[Array, " 2"]
        Raw delta-beta tail coordinates, left then right.
    model_domain : Float64[Array, " 2"]
        Increasing carrier domain ``[a, b]`` in eV.

    Returns
    -------
    composite : Float64[Array, " n_points"]
        Composite frequency derivative of the unsubtracted transform.

    Notes
    -----
    The derivative of the transform equals the transform of the
    derivative plus boundary terms, by partial integration on the
    finite core.
    """
    grid: Float64[Array, " n_kk"]
    spacing: Float64[Array, ""]
    grid, spacing = _frozen_base_grid(model_domain, n_kk)
    values: Float64[Array, " n_kk"] = _dynamic_imag(mode, coefficients, grid)
    derivative_samples: Float64[Array, " n_kk"] = _dynamic_imag_derivative(
        mode, coefficients, grid
    )
    core_derivative: Float64[Array, " n_points"] = _cubic_core_pv(
        grid, derivative_samples, points
    )
    boundary: Float64[Array, " n_points"] = (
        values[0] / (model_domain[0] - points)
        - values[-1] / (model_domain[1] - points)
    ) / jnp.pi
    slope_left: Float64[Array, ""]
    slope_right: Float64[Array, ""]
    slope_left, slope_right = _cubic_edge_slopes(values, spacing)
    spec: _Power2TailSpec = _power2_spec_from_edges(
        values[0],
        slope_left,
        values[-1],
        slope_right,
        tail_raw[0],
        tail_raw[1],
    )

    def _tail_only(
        stacked: Float64[Array, " n_points"],
    ) -> Float64[Array, " n_points"]:
        """PRIVATE: Evaluate both tail quadratures for the derivative closure.

        Parameters
        ----------
        stacked : Float64[Array, " n_points"]
            Stacked evaluation energies in eV.

        Returns
        -------
        contribution : Float64[Array, " n_points"]
            Unsubtracted tail contribution at every point.

        Notes
        -----
        Forward-mode differentiation of this closure supplies the
        exact tail derivative.
        """
        contribution: Float64[Array, " n_points"] = _power2_tail_pv(
            model_domain, spec, stacked, 256
        )
        return contribution

    tail_derivative: Float64[Array, " n_points"]
    _, tail_derivative = jax.jvp(
        _tail_only, (points,), (jnp.ones_like(points),)
    )
    composite: Float64[Array, " n_points"] = (
        core_derivative + boundary + tail_derivative
    )
    return composite


@partial(jax.custom_jvp, nondiff_argnums=(0, 1))
def _smooth_real_subtracted(
    mode: str,
    n_kk: int,
    omega_rel_fermi_ev: Float64[Array, " n_omega"],
    coefficients: Float64[Array, " n_coef"],
    tail_raw: Float64[Array, " 2"],
    subtraction: Float64[Array, ""],
    model_domain: Float64[Array, " 2"],
) -> Float64[Array, " n_omega"]:
    """PRIVATE: Evaluate the subtracted smooth-mode real part with its rule.

    The custom derivative rule binds the public frequency derivative to
    the analytic composite route. Its transpose supplies reverse mode,
    so public ``jax.jvp`` and ``jax.grad`` share one contract.

    Parameters
    ----------
    mode : str
        Smooth carrier mode, ``poly`` or ``fermi_liquid``.
    n_kk : int
        Static number of base grid nodes.
    omega_rel_fermi_ev : Float64[Array, " n_omega"]
        Query energies relative to the Fermi level in eV.
    coefficients : Float64[Array, " n_coef"]
        Unconstrained raw model coordinates.
    tail_raw : Float64[Array, " 2"]
        Raw delta-beta tail coordinates, left then right.
    subtraction : Float64[Array, ""]
        Carrier subtraction point in eV.
    model_domain : Float64[Array, " 2"]
        Increasing carrier domain ``[a, b]`` in eV.

    Returns
    -------
    real_subtracted : Float64[Array, " n_omega"]
        Subtracted real part at every query in eV.

    Notes
    -----
    The wrapper defers to the shared implementation body. Only the
    attached derivative rule distinguishes it from the plain call.
    """
    real_subtracted: Float64[Array, " n_omega"] = _smooth_real_impl(
        mode,
        n_kk,
        omega_rel_fermi_ev,
        coefficients,
        tail_raw,
        subtraction,
        model_domain,
    )
    return real_subtracted


@partial(_smooth_real_subtracted.defjvp, symbolic_zeros=True)
def _smooth_real_subtracted_jvp(
    mode: str,
    n_kk: int,
    primals: Any,
    tangents: Any,
) -> Tuple[Float64[Array, " n_omega"], Float64[Array, " n_omega"]]:
    """PRIVATE: Bind the public frequency tangent to the composite route.

    Frequency and subtraction tangents multiply the analytic composite
    derivative. Coefficient, tail, and domain tangents flow through the
    primal linearization at fixed frequencies. Symbolic-zero detection
    skips every unperturbed argument group.

    Parameters
    ----------
    mode : str
        Smooth carrier mode, ``poly`` or ``fermi_liquid``.
    n_kk : int
        Static number of base grid nodes.
    primals : Any
        Primal inputs ``(omega, coefficients, tail_raw, subtraction,
        domain)``.
    tangents : Any
        Matching tangent structure for the primal inputs.

    Returns
    -------
    pair : Tuple[Float64[Array, " n_omega"], Float64[Array, " n_omega"]]
        Primal subtracted real part and its tangent.

    Notes
    -----
    The composite evaluates once on the stacked queries. The last row
    carries the subtraction-point derivative with a negative sign.
    """
    omega_rel_fermi_ev: Float64[Array, " n_omega"]
    coefficients: Float64[Array, " n_coef"]
    tail_raw: Float64[Array, " 2"]
    subtraction: Float64[Array, ""]
    model_domain: Float64[Array, " 2"]
    (
        omega_rel_fermi_ev,
        coefficients,
        tail_raw,
        subtraction,
        model_domain,
    ) = primals
    omega_tangent: Any
    coefficient_tangent: Any
    tail_tangent: Any
    subtraction_tangent: Any
    domain_tangent: Any
    (
        omega_tangent,
        coefficient_tangent,
        tail_tangent,
        subtraction_tangent,
        domain_tangent,
    ) = tangents
    primal_out: Float64[Array, " n_omega"] = _smooth_real_impl(
        mode,
        n_kk,
        omega_rel_fermi_ev,
        coefficients,
        tail_raw,
        subtraction,
        model_domain,
    )
    tangent_out: Float64[Array, " n_omega"] = jnp.zeros_like(primal_out)

    def _fixed_frequencies(
        raw_coefficients: Float64[Array, " n_coef"],
        raw_tail: Float64[Array, " 2"],
        domain: Float64[Array, " 2"],
    ) -> Float64[Array, " n_omega"]:
        """PRIVATE: Evaluate the real part with every frequency held fixed.

        Parameters
        ----------
        raw_coefficients : Float64[Array, " n_coef"]
            Unconstrained raw model coordinates.
        raw_tail : Float64[Array, " 2"]
            Raw delta-beta tail coordinates, left then right.
        domain : Float64[Array, " 2"]
            Increasing carrier domain ``[a, b]`` in eV.

        Returns
        -------
        value : Float64[Array, " n_omega"]
            Subtracted real part at the closed-over queries.

        Notes
        -----
        Linearizing this closure yields the exact parameter tangents.
        """
        value: Float64[Array, " n_omega"] = _smooth_real_impl(
            mode,
            n_kk,
            omega_rel_fermi_ev,
            raw_coefficients,
            raw_tail,
            subtraction,
            domain,
        )
        return value

    parameter_perturbed: bool = not (
        _tangent_is_symbolic_zero(coefficient_tangent)
        and _tangent_is_symbolic_zero(tail_tangent)
        and _tangent_is_symbolic_zero(domain_tangent)
    )
    if parameter_perturbed:
        parameter_tangent: Float64[Array, " n_omega"]
        _, parameter_tangent = jax.jvp(
            _fixed_frequencies,
            (coefficients, tail_raw, model_domain),
            (
                _materialize_tangent(coefficient_tangent),
                _materialize_tangent(tail_tangent),
                _materialize_tangent(domain_tangent),
            ),
        )
        tangent_out = tangent_out + parameter_tangent
    frequency_perturbed: bool = not (
        _tangent_is_symbolic_zero(omega_tangent)
        and _tangent_is_symbolic_zero(subtraction_tangent)
    )
    if frequency_perturbed:
        stacked: Float64[Array, " n_stacked"] = jnp.concatenate(
            [omega_rel_fermi_ev, subtraction[None]]
        )
        composite: Float64[Array, " n_stacked"] = _smooth_query_composite(
            mode, n_kk, stacked, coefficients, tail_raw, model_domain
        )
        tangent_out = (
            tangent_out
            + composite[:-1] * _materialize_tangent(omega_tangent)
            - composite[-1] * _materialize_tangent(subtraction_tangent)
        )
    pair: Tuple[Float64[Array, " n_omega"], Float64[Array, " n_omega"]] = (
        primal_out,
        tangent_out,
    )
    return pair


def _hat_real_subtracted(
    omega_rel_fermi_ev: Float64[Array, " n_omega"],
    nodes: Float64[Array, " n_nodes"],
    coefficients: Float64[Array, " n_nodes"],
    tail_raw: Float64[Array, " 2"],
    subtraction: Float64[Array, ""],
    model_domain: Float64[Array, " 2"],
    n_kk: int,
) -> Float64[Array, " n_omega"]:
    """PRIVATE: Evaluate the exact subtracted grid-mode real part.

    The hat interpolant owns grid mode. The exact piecewise-linear
    transform runs on the carrier nodes, and the tail slopes come from
    the outer hat segments. The cubic reconstruction never touches this
    carrier class, because it can overshoot between negative samples.

    Parameters
    ----------
    omega_rel_fermi_ev : Float64[Array, " n_omega"]
        Query energies relative to the Fermi level in eV.
    nodes : Float64[Array, " n_nodes"]
        Strictly increasing carrier nodes in eV.
    coefficients : Float64[Array, " n_nodes"]
        Unconstrained raw hat coordinates.
    tail_raw : Float64[Array, " 2"]
        Raw delta-beta tail coordinates, left then right.
    subtraction : Float64[Array, ""]
        Carrier subtraction point in eV.
    model_domain : Float64[Array, " 2"]
        Increasing carrier domain ``[a, b]`` in eV.
    n_kk : int
        Static node count that fixes the trusted-interval margin.

    Returns
    -------
    real_subtracted : Float64[Array, " n_omega"]
        Subtracted real part at every query in eV.

    Notes
    -----
    The trusted-interval margin derives from the frozen base spacing
    ``(b - a) / (n_kk - 1)`` even though the hat core runs on the
    carrier nodes.
    """
    spacing: Float64[Array, ""] = (model_domain[1] - model_domain[0]) / (
        n_kk - 1
    )
    ordinates: Float64[Array, " n_nodes"] = -jnp.logaddexp(coefficients, 0.0)
    slope_left: Float64[Array, ""] = (ordinates[1] - ordinates[0]) / (
        nodes[1] - nodes[0]
    )
    slope_right: Float64[Array, ""] = (ordinates[-1] - ordinates[-2]) / (
        nodes[-1] - nodes[-2]
    )
    spec: _Power2TailSpec = _power2_spec_from_edges(
        ordinates[0],
        slope_left,
        ordinates[-1],
        slope_right,
        tail_raw[0],
        tail_raw[1],
    )
    stacked: Float64[Array, " n_stacked"] = jnp.concatenate(
        [omega_rel_fermi_ev, subtraction[None]]
    )
    checked: Float64[Array, " n_stacked"] = _check_trusted_interval(
        stacked, model_domain, spacing
    )
    core: Float64[Array, " n_stacked"] = _hat_core_pv(
        nodes, ordinates, checked
    )
    tails: Float64[Array, " n_stacked"] = _power2_tail_pv(
        model_domain, spec, checked, 256
    )
    total: Float64[Array, " n_stacked"] = core + tails
    real_subtracted: Float64[Array, " n_omega"] = total[:-1] - total[-1]
    return real_subtracted


def _kink_real_part(
    points: Float64[Array, " n_points"],
    coupling: Float64[Array, ""],
    omega_0: Float64[Array, ""],
    width: Float64[Array, ""],
) -> Float64[Array, " n_points"]:
    """PRIVATE: Evaluate the analytic bosonic-kink real pole pair.

    Parameters
    ----------
    points : Float64[Array, " n_points"]
        Evaluation energies in eV.
    coupling : Float64[Array, ""]
        Positive kink coupling in eV.
    omega_0 : Float64[Array, ""]
        Positive boson energy in eV.
    width : Float64[Array, ""]
        Positive pole width in eV.

    Returns
    -------
    real : Float64[Array, " n_points"]
        Analytic real part at every point in eV.

    Notes
    -----
    The pair reads ``g**2 * Re[1 / (w - w0 + i*W) + 1 / (w + w0 +
    i*W)]`` and vanishes at zero frequency.
    """
    lower: Float64[Array, " n_points"] = points - omega_0
    upper: Float64[Array, " n_points"] = points + omega_0
    real: Float64[Array, " n_points"] = coupling**2 * (
        lower / (lower**2 + width**2) + upper / (upper**2 + width**2)
    )
    return real


@jaxtyped(typechecker=beartype)
def evaluate_self_energy(  # noqa: DOC502 -- eqx.error_if raises under JAX execution.
    omega_rel_fermi_ev: Float64[Array, " n_omega"],
    model: SelfEnergyModel,
    n_kk: int = 4096,
) -> Complex128[Array, " n_omega"]:
    r"""Evaluate the complex retarded self-energy for one causal model.

    The function returns :math:`\Sigma(E - E_F) = \Sigma' + i\Sigma''`
    for every carrier mode. Constant mode returns a purely imaginary
    result with an exactly zero subtracted real part. The bosonic kink
    evaluates its analytic complex pole pair. The numerical modes
    ``poly``, ``grid``, and ``fermi_liquid`` obtain the subtracted real
    part from the certified cell-integrated Kramers--Kronig operator.
    That operator lives on the declared carrier domain and carries C1
    ``power2`` tails.

    :see: :class:`~.test_spectral.TestEvaluateSelfEnergy`

    Implementation Logic
    --------------------
    1. **Dispatch on the static carrier mode**::

           mode = model.mode

       The Python string selects one code path outside tracing.
    2. **Evaluate the analytic modes directly**::

           real = jnp.zeros_like(omega_rel_fermi_ev)
           real = _kink_real_part(omega, g, omega_0, width) - baseline

       Constant mode keeps an exactly zero subtracted real part. The
       kink subtracts its pole pair at the carrier subtraction point.
    3. **Evaluate the numerical modes through the transform**::

           real = _smooth_real_subtracted(mode, n_kk, omega, ...)
           real = _hat_real_subtracted(omega, nodes, ...)

       The frozen grid comes from ``kk_domain_rel_fermi_ev``, never
       from the query extrema. Grid mode uses the exact hat transform.
    4. **Assemble the complex retarded result**::

           result = jax.lax.complex(real, imag)

       The imaginary part evaluates the mode closed form directly.

    Parameters
    ----------
    omega_rel_fermi_ev : Float64[Array, " n_omega"]
        Query energies relative to the Fermi level in eV. Numerical
        modes require every query inside the trusted interval
        ``[a + 2h, b - 2h]``.
    model : SelfEnergyModel
        Validated causal self-energy carrier.
    n_kk : int, optional
        Static internal Kramers--Kronig grid length. Default is 4096.

    Returns
    -------
    sigma : Complex128[Array, " n_omega"]
        Complex retarded self-energy at every query in eV.

    Raises
    ------
    ValueError
        If ``n_kk`` cannot support the certified operator stencils.
    EquinoxRuntimeError
        If one query or the subtraction point leaves the trusted
        interval, eagerly and inside compiled code.

    Notes
    -----
    The public frequency derivative follows the composite route through
    ``jax.custom_jvp``. The rule applies the same principal-value
    operator to the analytic mode-supplied
    :math:`\partial_\omega\Sigma''`. It then adds the boundary terms
    and the exact tail derivatives. The rule transpose supplies reverse
    mode, so ``jax.jvp`` and ``jax.grad`` agree. Parameter tangents
    flow through the primal linearization. Grid mode differentiates its
    exact closed form directly. Its derivative contract holds only away
    from the hat knots, where the hat transform stays smooth. The
    ``TestEvaluateSelfEnergyDerivatives`` and
    ``TestGridModeHatTransform`` classes pin these contracts.
    """
    mode: str = model.mode
    minimum_n_kk: int = 8
    if n_kk < minimum_n_kk:
        msg: str = (
            "n_kk must reach eight nodes so the certified operator "
            "stencils stay defined"
        )
        raise ValueError(msg)
    if mode == "constant":
        real: Float64[Array, " n_omega"] = jnp.zeros_like(omega_rel_fermi_ev)
        imag: Float64[Array, " n_omega"] = jnp.broadcast_to(
            -jnp.logaddexp(model.coefficients[0], 0.0),
            omega_rel_fermi_ev.shape,
        )
    elif mode == "bosonic_kink":
        gamma_0: Float64[Array, ""] = jnp.logaddexp(model.coefficients[0], 0.0)
        coupling: Float64[Array, ""] = jnp.logaddexp(
            model.coefficients[1], 0.0
        )
        omega_0: Float64[Array, ""] = jnp.logaddexp(model.coefficients[2], 0.0)
        width: Float64[Array, ""] = jnp.logaddexp(model.coefficients[3], 0.0)
        baseline: Float64[Array, " one"] = _kink_real_part(
            model.subtraction_point_rel_fermi_ev[None],
            coupling,
            omega_0,
            width,
        )
        real = (
            _kink_real_part(omega_rel_fermi_ev, coupling, omega_0, width)
            - baseline[0]
        )
        lower: Float64[Array, " n_omega"] = omega_rel_fermi_ev - omega_0
        upper: Float64[Array, " n_omega"] = omega_rel_fermi_ev + omega_0
        imag = -gamma_0 - coupling**2 * width * (
            1.0 / (lower**2 + width**2) + 1.0 / (upper**2 + width**2)
        )
    else:
        domain: Optional[Float64[Array, " 2"]] = model.kk_domain_rel_fermi_ev
        tail_raw: Optional[Float64[Array, " 2"]] = model.tail_coefficients
        if domain is None or tail_raw is None:
            msg = (
                "numerical Kramers-Kronig modes require a declared "
                "domain and tail coordinates"
            )
            raise ValueError(msg)
        if mode == "grid":
            nodes: Optional[Float64[Array, " n_nodes"]] = (
                model.energy_nodes_rel_fermi_ev
            )
            if nodes is None:
                msg = "grid mode requires carrier energy nodes"
                raise ValueError(msg)
            real = _hat_real_subtracted(
                omega_rel_fermi_ev,
                nodes,
                model.coefficients,
                tail_raw,
                model.subtraction_point_rel_fermi_ev,
                domain,
                n_kk,
            )
            imag = jnp.interp(
                omega_rel_fermi_ev,
                nodes,
                -jnp.logaddexp(model.coefficients, 0.0),
            )
        else:
            real = _smooth_real_subtracted(
                mode,
                n_kk,
                omega_rel_fermi_ev,
                model.coefficients,
                tail_raw,
                model.subtraction_point_rel_fermi_ev,
                domain,
            )
            if mode == "fermi_liquid":
                imag = -jnp.logaddexp(
                    model.coefficients[0], 0.0
                ) + _dynamic_imag(mode, model.coefficients, omega_rel_fermi_ev)
            else:
                imag = _dynamic_imag(
                    mode, model.coefficients, omega_rel_fermi_ev
                )
    sigma: Complex128[Array, " n_omega"] = jax.lax.complex(real, imag)
    return sigma


__all__: list[str] = ["evaluate_self_energy"]
