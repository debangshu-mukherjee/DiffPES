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

The same module assembles the intrinsic coherent observable
:math:`A(k,\omega)f_\mathrm{FD}(\omega,T)`. The degeneracy-safe path applies a
complex128 Lineax resolvent to the full matrix-element source. The faster
eigen path consumes gauge-invariant band weights. Differentiated eigen calls
require every adjacent gap to be at least :math:`10^3\epsilon_{deg}`. An
explicit value-only mode admits complete invariant weights at degeneracy for
primal compatibility checks. Both paths keep detector resolution,
transmission, backgrounds, and counts outside the intrinsic spectral boundary.

Scalability
-----------
The resolvent route performs one complex128 LU solve per
``(k, omega, n_out)`` and therefore costs
:math:`O(K E n_{out} n_{orb}^3)`. Its checkpointed static scan keeps only a
live ``(k_chunk, omega_chunk)`` transition block; the registered spinless
solve-tape estimate is
``16 * n_k * omega_chunk * n_orb**2`` bytes. It never materializes a complete
``(K, E, n_out, n_orb)`` source carrier. The eigen route performs one
eigendecomposition per k point and amortizes it over sampled energy. Its cost
is :math:`O(K n_{orb}^3 + K E n_{orb})`. Use it for nondegenerate k paths.
Use the resolvent route at degeneracies and for Hamiltonian
gradients; the explicit degenerate eigen exception carries no derivative
claim. The resolver forbids mixed precision. Operator, RHS, LU, and solution
remain complex128.

Routine Listings
----------------
:func:`assemble_spectral_intensity_bands_chunk`
    Assemble occupied intrinsic intensity from eigenvalues and band weights.
:func:`assemble_spectral_intensity_chunk`
    Assemble occupied intrinsic intensity from Hamiltonians and sources.
:func:`evaluate_self_energy`
    Evaluate the complex retarded self-energy for one causal model.
:func:`projected_spectral_density_resolvent`
    Compute the projected Hermitian resolvent spectral density.
:func:`spectral_intensity_eigen`
    Evaluate spectral intensity from eigenvalues and invariant weights.
:func:`spectral_intensity_resolvent`
    Evaluate degeneracy-safe spectral intensity through a linear solve.

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
import lineax as lx
import numpy as np
from beartype import beartype
from beartype.typing import Any, Optional, Tuple
from jax.custom_derivatives import SymbolicZero
from jaxtyping import Array, Bool, Complex128, Float64, Int, jaxtyped
from numpy.typing import NDArray

from diffpes.radial import momentum_inv_ang_to_bohr_inv, radial_bvals
from diffpes.types import (
    EPS,
    EPS_DEG,
    G_PARALLEL_ATOL_INV_ANG,
    FinalStateSpec,
    MatrixElementParams,
    RadialQuadratureSpec,
    RadialSpec,
    ScalarBool,
    ScalarFloat,
    SelfEnergyModel,
)

from .broadening import fermi_dirac
from .matrixel import (
    contract_polarization,
    orbital_transition_channels,
    transition_source,
)


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


def _kk_transform_impl(  # noqa: DOC502, DOC503 -- JAX runtime guards.
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
def evaluate_self_energy(  # noqa: DOC502, DOC503 -- JAX runtime guards.
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


def _checked_spectral_hamiltonian(
    hamiltonian: Complex128[Array, "n_orb n_orb"],
    *,
    context: str,
) -> Complex128[Array, "n_orb n_orb"]:
    """PRIVATE: Validate one finite Hermitian Hamiltonian.

    Parameters
    ----------
    hamiltonian : Complex128[Array, "n_orb n_orb"]
        Candidate Hamiltonian in eV.
    context : str
        Public caller name used in error messages.

    Returns
    -------
    checked : Complex128[Array, "n_orb n_orb"]
        Unchanged Hamiltonian carrying both runtime guards.

    Notes
    -----
    The types-owned ``EPS`` tolerance matches the Hermitian validation
    used by the tight-binding eigensolver. Both checks survive JIT.
    """
    checked: Complex128[Array, "n_orb n_orb"] = eqx.error_if(
        hamiltonian,
        ~jnp.all(jnp.isfinite(hamiltonian)),
        f"{context}: Hamiltonian entries must be finite",
    )
    checked = eqx.error_if(
        checked,
        ~jnp.allclose(checked, checked.conj().T, rtol=EPS, atol=EPS),
        f"{context}: Hamiltonian must be Hermitian",
    )
    return checked  # noqa: RET504 -- the returned value carries both guards.


def _checked_resolvent_scalars(
    omega_rel_fermi_ev: Float64[Array, ""],
    sigma_omega: Complex128[Array, ""],
    eta: ScalarFloat,
    *,
    context: str,
) -> Tuple[
    Float64[Array, ""],
    Complex128[Array, ""],
    Float64[Array, ""],
]:
    """PRIVATE: Validate one retarded resolvent coordinate.

    Parameters
    ----------
    omega_rel_fermi_ev : Float64[Array, ""]
        Sampled energy relative to the Fermi level in eV.
    sigma_omega : Complex128[Array, ""]
        Retarded self-energy at the sampled energy in eV.
    eta : ScalarFloat
        Positive regulator in eV.
    context : str
        Public caller name used in error messages.

    Returns
    -------
    checked : Tuple[Float64[Array, ""], Complex128[Array, ""],
        Float64[Array, ""]]
        Finite sampled energy, retarded self-energy with a positive total
        linewidth, and a positive float64 regulator.

    Notes
    -----
    The physical denominator width is ``eta - imag(sigma)``. Requiring it
    to remain positive rejects an advanced or singular resolvent.
    """
    omega_checked: Float64[Array, ""] = eqx.error_if(
        omega_rel_fermi_ev,
        ~jnp.isfinite(omega_rel_fermi_ev),
        f"{context}: omega must be finite",
    )
    eta_array: Float64[Array, ""] = jnp.asarray(eta, dtype=jnp.float64)
    eta_checked: Float64[Array, ""] = eqx.error_if(
        eta_array,
        ~jnp.isfinite(eta_array) | (eta_array <= 0.0),
        f"{context}: eta must be finite and strictly positive",
    )
    sigma_checked: Complex128[Array, ""] = eqx.error_if(
        sigma_omega,
        ~jnp.isfinite(sigma_omega),
        f"{context}: sigma_omega must be finite",
    )
    sigma_checked = eqx.error_if(
        sigma_checked,
        jnp.imag(sigma_checked) > 0.0,
        f"{context}: retarded sigma_omega must have a nonpositive "
        "imaginary part",
    )
    sigma_checked = eqx.error_if(
        sigma_checked,
        eta_checked - jnp.imag(sigma_checked) <= 0.0,
        f"{context}: eta - imag(sigma_omega) must be strictly positive",
    )
    checked: Tuple[
        Float64[Array, ""],
        Complex128[Array, ""],
        Float64[Array, ""],
    ] = (omega_checked, sigma_checked, eta_checked)
    return checked


def _resolvent_solution(
    hamiltonian_rel_fermi_k: Complex128[Array, "n_orb n_orb"],
    source: Complex128[Array, " n_orb"],
    omega_rel_fermi_ev: Float64[Array, ""],
    sigma_omega: Complex128[Array, ""],
    eta: Float64[Array, ""],
) -> Complex128[Array, " n_orb"]:
    """PRIVATE: Apply the complex128 retarded resolvent to one source.

    Parameters
    ----------
    hamiltonian_rel_fermi_k : Complex128[Array, "n_orb n_orb"]
        Hermitian Hamiltonian relative to the Fermi level in eV.
    source : Complex128[Array, " n_orb"]
        Right-hand side source ket.
    omega_rel_fermi_ev : Float64[Array, ""]
        Relative sampled energy in eV.
    sigma_omega : Complex128[Array, ""]
        Retarded self-energy at that energy in eV.
    eta : Float64[Array, ""]
        Positive regulator in eV.

    Returns
    -------
    solution : Complex128[Array, " n_orb"]
        ``((omega + i*eta - sigma) I - H)^{-1} source``.

    Notes
    -----
    Lineax owns the transpose rule, so reverse mode uses the corresponding
    adjoint solve without a hand-written custom derivative.
    """
    identity: Complex128[Array, "n_orb n_orb"] = jnp.eye(
        hamiltonian_rel_fermi_k.shape[0], dtype=jnp.complex128
    )
    operator_matrix: Complex128[Array, "n_orb n_orb"] = (
        omega_rel_fermi_ev + 1.0j * eta - sigma_omega
    ) * identity - hamiltonian_rel_fermi_k
    solution: Complex128[Array, " n_orb"] = lx.linear_solve(
        lx.MatrixLinearOperator(operator_matrix),
        source,
        lx.LU(),
    ).value
    return solution


def _spectral_intensity_resolvent_unchecked(
    hamiltonian_rel_fermi_k: Complex128[Array, "n_orb n_orb"],
    transition_source: Complex128[Array, " n_orb"],
    omega_rel_fermi_ev: Float64[Array, ""],
    sigma_omega: Complex128[Array, ""],
    eta: Float64[Array, ""],
) -> Float64[Array, ""]:
    """PRIVATE: Evaluate one already-validated resolvent quadratic form.

    Notes
    -----
    The caller owns all domain checks. This helper performs one complex128
    solve and contracts the source with its response.
    """
    solution: Complex128[Array, " n_orb"] = _resolvent_solution(
        hamiltonian_rel_fermi_k,
        transition_source,
        omega_rel_fermi_ev,
        sigma_omega,
        eta,
    )
    intensity: Float64[Array, ""] = (
        -jnp.imag(jnp.vdot(transition_source, solution)) / jnp.pi
    )
    return intensity


def _summed_spectral_intensity_resolvent_unchecked(
    hamiltonian_rel_fermi_k: Complex128[Array, "n_orb n_orb"],
    transition_sources: Complex128[Array, "n_out n_orb"],
    omega_rel_fermi_ev: Float64[Array, ""],
    sigma_omega: Complex128[Array, ""],
    eta: Float64[Array, ""],
) -> Float64[Array, ""]:
    """PRIVATE: Evaluate every outgoing source before incoherent reduction.

    Notes
    -----
    Vectorization applies the scalar resolvent to each source separately.
    The helper sums only after it forms each real quadratic response.
    """
    per_output: Float64[Array, " n_out"] = jax.vmap(
        _spectral_intensity_resolvent_unchecked,
        in_axes=(None, 0, None, None, None),
    )(
        hamiltonian_rel_fermi_k,
        transition_sources,
        omega_rel_fermi_ev,
        sigma_omega,
        eta,
    )
    intensity: Float64[Array, ""] = jnp.sum(per_output)
    return intensity


@jaxtyped(typechecker=beartype)
def spectral_intensity_resolvent(  # noqa: DOC502, DOC503 -- traced guards.
    hamiltonian_rel_fermi_k: Complex128[Array, "n_orb n_orb"],
    transition_sources: Complex128[Array, "n_out n_orb"],
    omega_rel_fermi_ev: Float64[Array, ""],
    sigma_omega: Complex128[Array, ""],
    eta: ScalarFloat,
) -> Float64[Array, ""]:
    r"""Evaluate degeneracy-safe spectral intensity through a linear solve.

    For every outgoing channel :math:`\alpha`, the primitive computes
    :math:`-\operatorname{Im}[s_\alpha^\dagger G(\omega)s_\alpha]/\pi`,
    where :math:`G=[(\omega+i\eta-\Sigma)I-H]^{-1}`, and then sums the real
    responses. It never coherently combines sources before solving and never
    differentiates an eigenvector, so exact band degeneracies remain regular.

    :see: :class:`~.test_spectral.TestSpectralIntensityResolvent`

    Parameters
    ----------
    hamiltonian_rel_fermi_k : Complex128[Array, "n_orb n_orb"]
        Finite Hermitian Hamiltonian relative to the Fermi level in eV.
    transition_sources : Complex128[Array, "n_out n_orb"]
        Nonempty outgoing-channel source kets ``s = d.conj()`` from the
        matrix-element seam. ``n_out=1`` is the spinless case.
    omega_rel_fermi_ev : Float64[Array, ""]
        Sampled energy relative to the Fermi level in eV.
    sigma_omega : Complex128[Array, ""]
        Complex retarded self-energy at the sampled energy in eV.
    eta : ScalarFloat
        Finite, strictly positive resolvent regulator in eV.

    Returns
    -------
    intensity : Float64[Array, ""]
        Intrinsic spectral intensity in inverse eV.

    Raises
    ------
    ValueError
        If the outgoing-channel axis is empty.
    EquinoxRuntimeError
        If an input is non-finite, the Hamiltonian is non-Hermitian, or the
        total linewidth is not strictly positive.

    Notes
    -----
    Each source enters an independent scalar-RHS solve. The contraction uses
    :func:`jax.numpy.vdot`, not ``dot``. The helper reduces only after forming
    the real quadratic responses. Lineax keeps the operator, right-hand side,
    LU factorization, and result in complex128. It supplies exact forward- and
    reverse-mode rules.
    """
    if transition_sources.shape[0] == 0:
        raise ValueError("transition_sources n_out axis must be nonempty")
    checked_hamiltonian: Complex128[Array, "n_orb n_orb"] = (
        _checked_spectral_hamiltonian(
            hamiltonian_rel_fermi_k,
            context="spectral_intensity_resolvent",
        )
    )
    checked_sources: Complex128[Array, "n_out n_orb"] = eqx.error_if(
        transition_sources,
        ~jnp.all(jnp.isfinite(transition_sources)),
        "spectral_intensity_resolvent: transition_sources must be finite",
    )
    omega_checked: Float64[Array, ""]
    sigma_checked: Complex128[Array, ""]
    eta_checked: Float64[Array, ""]
    omega_checked, sigma_checked, eta_checked = _checked_resolvent_scalars(
        omega_rel_fermi_ev,
        sigma_omega,
        eta,
        context="spectral_intensity_resolvent",
    )
    intensity: Float64[Array, ""] = (
        _summed_spectral_intensity_resolvent_unchecked(
            checked_hamiltonian,
            checked_sources,
            omega_checked,
            sigma_checked,
            eta_checked,
        )
    )
    return intensity


@jaxtyped(typechecker=beartype)
def projected_spectral_density_resolvent(  # noqa: DOC502 -- traced guards.
    hamiltonian_rel_fermi_k: Complex128[Array, "n_orb n_orb"],
    transition_operator: Complex128[Array, "n_out n_orb"],
    omega_rel_fermi_ev: Float64[Array, ""],
    sigma_omega: Complex128[Array, ""],
    eta: ScalarFloat,
) -> Complex128[Array, "n_out n_out"]:
    r"""Compute the projected Hermitian resolvent spectral density.

    The returned matrix is
    :math:`D[-(G-G^\dagger)/(2\pi i)]D^\dagger`. This polynomial projector
    form preserves off-diagonal spin and channel coherences at degeneracies.

    :see: :class:`~.test_spectral.TestProjectedSpectralDensityResolvent`

    Parameters
    ----------
    hamiltonian_rel_fermi_k : Complex128[Array, "n_orb n_orb"]
        Finite Hermitian Hamiltonian relative to the Fermi level in eV.
    transition_operator : Complex128[Array, "n_out n_orb"]
        Output-channel rows ``D`` in the orbital basis.
    omega_rel_fermi_ev : Float64[Array, ""]
        Sampled energy relative to the Fermi level in eV.
    sigma_omega : Complex128[Array, ""]
        Complex retarded self-energy at the sampled energy in eV.
    eta : ScalarFloat
        Finite, strictly positive regulator in eV.

    Returns
    -------
    density : Complex128[Array, "n_out n_out"]
        Hermitian positive-semidefinite projected spectral density.

    Raises
    ------
    EquinoxRuntimeError
        If an input is non-finite, the Hamiltonian is non-Hermitian, or the
        total linewidth is not strictly positive.

    Notes
    -----
    A static ``vmap`` applies the same Lineax operator to every column of
    ``D.dagger``. Antisymmetrizing the projected Green function as a matrix
    preserves its off-diagonal coherences. An elementwise imaginary part
    corrupts them.
    """
    checked_hamiltonian: Complex128[Array, "n_orb n_orb"] = (
        _checked_spectral_hamiltonian(
            hamiltonian_rel_fermi_k,
            context="projected_spectral_density_resolvent",
        )
    )
    checked_operator: Complex128[Array, "n_out n_orb"] = eqx.error_if(
        transition_operator,
        ~jnp.all(jnp.isfinite(transition_operator)),
        "projected_spectral_density_resolvent: transition_operator "
        "must be finite",
    )
    omega_checked: Float64[Array, ""]
    sigma_checked: Complex128[Array, ""]
    eta_checked: Float64[Array, ""]
    omega_checked, sigma_checked, eta_checked = _checked_resolvent_scalars(
        omega_rel_fermi_ev,
        sigma_omega,
        eta,
        context="projected_spectral_density_resolvent",
    )
    right_hand_sides: Complex128[Array, "n_orb n_out"] = (
        checked_operator.conj().T
    )
    solved: Complex128[Array, "n_orb n_out"] = jax.vmap(
        lambda source: _resolvent_solution(
            checked_hamiltonian,
            source,
            omega_checked,
            sigma_checked,
            eta_checked,
        ),
        in_axes=1,
        out_axes=1,
    )(right_hand_sides)
    projected_green: Complex128[Array, "n_out n_out"] = (
        checked_operator @ solved
    )
    density: Complex128[Array, "n_out n_out"] = -(
        projected_green - projected_green.conj().T
    ) / (2.0j * jnp.pi)
    return density


def _checked_eigenvalue_domain(
    eigenvalues_ev: Float64[Array, "... n_bands"],
    allow_degenerate_value_only: ScalarBool,
    *,
    context: str,
) -> Float64[Array, "... n_bands"]:
    """PRIVATE: Enforce the differentiated eigen-path gap floor.

    Parameters
    ----------
    eigenvalues_ev : Float64[Array, "... n_bands"]
        Finite eigenvalues in eV, with any leading batch axes.
    allow_degenerate_value_only : ScalarBool
        Whether to admit a degenerate primal with no derivative claim.
    context : str
        Public caller name included in a rejection message.

    Returns
    -------
    checked : Float64[Array, "... n_bands"]
        Eigenvalues carrying the traced nondegenerate-domain guard.
    """
    if eigenvalues_ev.shape[-1] < 2:  # noqa: PLR2004 -- a gap needs a pair.
        return eigenvalues_ev
    minimum_gap_ev: float = 1.0e3 * EPS_DEG
    sorted_eigenvalues: Float64[Array, "... n_bands"] = jnp.sort(
        eigenvalues_ev,
        axis=-1,
    )
    adjacent_gaps: Float64[Array, "... n_gap"] = jnp.diff(
        sorted_eigenvalues,
        axis=-1,
    )
    minimum_gap: Float64[Array, ""] = jnp.min(adjacent_gaps)
    enforce_gap: Bool[Array, ""] = ~jnp.asarray(
        allow_degenerate_value_only,
        dtype=jnp.bool_,
    )
    checked: Float64[Array, "... n_bands"] = eqx.error_if(
        eigenvalues_ev,
        enforce_gap & (minimum_gap < minimum_gap_ev),
        f"{context}: differentiated eigen path requires every adjacent band "
        f"gap to be at least {minimum_gap_ev:.1e} eV; use the "
        "resolvent for gradients or set allow_degenerate_value_only=True "
        "only for primal evaluation",
    )
    return checked


def _spectral_intensity_eigen_unchecked(
    eigenvalues_rel_fermi_ev: Float64[Array, " n_bands"],
    band_weights: Float64[Array, " n_bands"],
    omega_rel_fermi_ev: Float64[Array, ""],
    sigma_omega: Complex128[Array, ""],
    eta: Float64[Array, ""],
) -> Float64[Array, ""]:
    """PRIVATE: Sum already-validated Lorentzian band contributions.

    Notes
    -----
    The caller validates weights, eigenvalues, and linewidth. This helper
    contains only the normalized Lorentzian arithmetic.
    """
    linewidth: Float64[Array, ""] = eta - jnp.imag(sigma_omega)
    displacement: Float64[Array, " n_bands"] = (
        omega_rel_fermi_ev - eigenvalues_rel_fermi_ev - jnp.real(sigma_omega)
    )
    intensity: Float64[Array, ""] = jnp.sum(
        band_weights * linewidth / (jnp.pi * (displacement**2 + linewidth**2))
    )
    return intensity


@jaxtyped(typechecker=beartype)
def spectral_intensity_eigen(  # noqa: DOC502 -- traced domain guards.
    eigenvalues_rel_fermi_ev: Float64[Array, " n_bands"],
    band_weights: Float64[Array, " n_bands"],
    omega_rel_fermi_ev: Float64[Array, ""],
    sigma_omega: Complex128[Array, ""],
    eta: ScalarFloat,
    *,
    allow_degenerate_value_only: ScalarBool = False,
) -> Float64[Array, ""]:
    """Evaluate spectral intensity from eigenvalues and invariant weights.

    This fast path sums one normalized Lorentzian per band. Its inputs are
    gauge-invariant band weights, so raw eigenvector phases never reach the
    observable. The resolvent path remains the certified choice at an exact
    degeneracy.

    :see: :class:`~.test_spectral.TestSpectralIntensityEigen`

    Parameters
    ----------
    eigenvalues_rel_fermi_ev : Float64[Array, " n_bands"]
        Band energies relative to the Fermi level in eV.
    band_weights : Float64[Array, " n_bands"]
        Finite, nonnegative gauge-invariant transition weights.
    omega_rel_fermi_ev : Float64[Array, ""]
        Sampled energy relative to the Fermi level in eV.
    sigma_omega : Complex128[Array, ""]
        Complex retarded self-energy at the sampled energy in eV.
    eta : ScalarFloat
        Finite, strictly positive regulator in eV.
    allow_degenerate_value_only : ScalarBool, optional
        Admit an exact or near-degenerate primal without certifying JVPs,
        VJPs, finite differences, or Hamiltonian-parameter derivatives.
        Default is ``False``.

    Returns
    -------
    intensity : Float64[Array, ""]
        Intrinsic spectral intensity in inverse eV.

    Raises
    ------
    EquinoxRuntimeError
        If an input is non-finite, a band weight is negative, or the total
        linewidth is not strictly positive. Also raised when the minimum band
        gap is below ``1e3 * EPS_DEG`` unless value-only evaluation is
        explicit.

    Notes
    -----
    The linewidth is exactly ``eta - imag(sigma_omega)`` and the pole
    displacement is ``omega - eigenvalue - real(sigma_omega)``. Equal poles
    have a gauge-invariant primal when their supplied weights form a complete
    invariant group. Only the resolvent path owns derivatives at such a
    degeneracy.
    """
    checked_eigenvalues: Float64[Array, " n_bands"] = eqx.error_if(
        eigenvalues_rel_fermi_ev,
        ~jnp.all(jnp.isfinite(eigenvalues_rel_fermi_ev)),
        "spectral_intensity_eigen: eigenvalues must be finite",
    )
    checked_eigenvalues = _checked_eigenvalue_domain(
        checked_eigenvalues,
        allow_degenerate_value_only,
        context="spectral_intensity_eigen",
    )
    checked_weights: Float64[Array, " n_bands"] = eqx.error_if(
        band_weights,
        ~jnp.all(jnp.isfinite(band_weights) & (band_weights >= 0.0)),
        "spectral_intensity_eigen: band_weights must be finite and "
        "nonnegative",
    )
    omega_checked: Float64[Array, ""]
    sigma_checked: Complex128[Array, ""]
    eta_checked: Float64[Array, ""]
    omega_checked, sigma_checked, eta_checked = _checked_resolvent_scalars(
        omega_rel_fermi_ev,
        sigma_omega,
        eta,
        context="spectral_intensity_eigen",
    )
    intensity: Float64[Array, ""] = _spectral_intensity_eigen_unchecked(
        checked_eigenvalues,
        checked_weights,
        omega_checked,
        sigma_checked,
        eta_checked,
    )
    return intensity


def _sampled_fermi_occupation(
    omega_rel_fermi_ev: Float64[Array, " n_chunk"],
    temperature_k: ScalarFloat,
) -> Float64[Array, " n_chunk"]:
    """PRIVATE: Evaluate occupation on the sampled relative-energy axis.

    Notes
    -----
    Vectorization evaluates the shared scalar Fermi primitive at every
    sampled energy and a zero relative chemical potential.
    """
    occupation: Float64[Array, " n_chunk"] = jax.vmap(
        lambda omega: fermi_dirac(omega, 0.0, temperature_k)
    )(omega_rel_fermi_ev)
    return occupation


@jaxtyped(typechecker=beartype)
def assemble_spectral_intensity_chunk(  # noqa: DOC502, DOC503 -- traced guards.
    hamiltonians_ev: Complex128[Array, "n_k n_orb n_orb"],
    transition_sources: Complex128[Array, "n_k n_chunk n_out n_orb"],
    omega_rel_fermi_ev: Float64[Array, " n_chunk"],
    self_energy: SelfEnergyModel,
    fermi_energy_ev: Float64[Array, ""],
    temperature_k: ScalarFloat,
    eta: ScalarFloat = 1.0e-4,
) -> Float64[Array, "n_k n_chunk"]:
    """Assemble occupied intrinsic intensity from Hamiltonians and sources.

    The degeneracy-safe path shifts each absolute Hamiltonian by the Fermi
    energy exactly once. It evaluates the causal self-energy once on the
    sampled relative-energy grid. It multiplies the spectral function by the
    Fermi occupation at those sampled energies.

    :see: :class:`~.test_spectral.TestAssembleSpectralIntensityChunk`

    Parameters
    ----------
    hamiltonians_ev : Complex128[Array, "n_k n_orb n_orb"]
        Absolute-energy Hermitian Hamiltonians in eV.
    transition_sources : Complex128[Array, "n_k n_chunk n_out n_orb"]
        Nonempty outgoing-channel source kets for each ``(k, omega)``.
        The code solves every channel independently; ``n_out=1`` is spinless.
    omega_rel_fermi_ev : Float64[Array, " n_chunk"]
        Sampled energies ``E - E_F`` in eV.
    self_energy : SelfEnergyModel
        Validated causal self-energy carrier on the relative-energy axis.
    fermi_energy_ev : Float64[Array, ""]
        Absolute Fermi energy subtracted from every Hamiltonian once.
    temperature_k : ScalarFloat
        Finite, strictly positive sample temperature in kelvin.
    eta : ScalarFloat, optional
        Positive resolvent regulator in eV. Default is ``1e-4``.

    Returns
    -------
    intensity : Float64[Array, "n_k n_chunk"]
        Intrinsic ``A(k, omega) f_FD(omega, T)`` in inverse eV.

    Raises
    ------
    ValueError
        If the outgoing-channel axis is empty.
    EquinoxRuntimeError
        If any numerical input violates the finite, Hermitian, causal, or
        positive-temperature contract.

    Notes
    -----
    The operation contains no detector convolution, count normalization, or
    background. Peak live solve storage scales as approximately
    ``16 * n_k * n_chunk * n_out * n_orb**2`` bytes in complex128. Scan static
    omega chunks and checkpoint this function. Use the eigen path for long
    nondegenerate paths. Use the resolvent at degeneracies or for Hamiltonian
    gradients.
    """
    if transition_sources.shape[2] == 0:
        raise ValueError("transition_sources n_out axis must be nonempty")
    checked_fermi: Float64[Array, ""] = eqx.error_if(
        fermi_energy_ev,
        ~jnp.isfinite(fermi_energy_ev),
        "assemble_spectral_intensity_chunk: fermi_energy_ev must be finite",
    )
    checked_omega: Float64[Array, " n_chunk"] = eqx.error_if(
        omega_rel_fermi_ev,
        ~jnp.all(jnp.isfinite(omega_rel_fermi_ev)),
        "assemble_spectral_intensity_chunk: omega must be finite",
    )
    checked_sources: Complex128[Array, "n_k n_chunk n_out n_orb"] = (
        eqx.error_if(
            transition_sources,
            ~jnp.all(jnp.isfinite(transition_sources)),
            "assemble_spectral_intensity_chunk: transition_sources must be "
            "finite",
        )
    )
    checked_hamiltonians: Complex128[Array, "n_k n_orb n_orb"] = jax.vmap(
        lambda hamiltonian: _checked_spectral_hamiltonian(
            hamiltonian,
            context="assemble_spectral_intensity_chunk",
        )
    )(hamiltonians_ev)
    identity: Complex128[Array, "n_orb n_orb"] = jnp.eye(
        hamiltonians_ev.shape[-1], dtype=jnp.complex128
    )
    hamiltonians_rel: Complex128[Array, "n_k n_orb n_orb"] = (
        checked_hamiltonians - checked_fermi * identity[None, :, :]
    )
    sigma: Complex128[Array, " n_chunk"] = evaluate_self_energy(
        checked_omega,
        self_energy,
    )
    eta_array: Float64[Array, ""] = jnp.asarray(eta, dtype=jnp.float64)
    eta_checked: Float64[Array, ""] = eqx.error_if(
        eta_array,
        ~jnp.isfinite(eta_array) | (eta_array <= 0.0),
        "assemble_spectral_intensity_chunk: eta must be finite and positive",
    )
    spectral: Float64[Array, "n_k n_chunk"] = jax.vmap(
        lambda hamiltonian, sources: jax.vmap(
            _summed_spectral_intensity_resolvent_unchecked,
            in_axes=(None, 0, 0, 0, None),
        )(hamiltonian, sources, checked_omega, sigma, eta_checked)
    )(hamiltonians_rel, checked_sources)
    occupation: Float64[Array, " n_chunk"] = _sampled_fermi_occupation(
        checked_omega,
        temperature_k,
    )
    intensity: Float64[Array, "n_k n_chunk"] = spectral * occupation[None, :]
    return intensity


@jaxtyped(typechecker=beartype)
def assemble_spectral_intensity_bands_chunk(  # noqa: DOC502 -- traced guards.
    eigenvalues_ev: Float64[Array, "n_k n_bands"],
    band_weights: Float64[Array, "n_k n_chunk n_bands"],
    omega_rel_fermi_ev: Float64[Array, " n_chunk"],
    self_energy: SelfEnergyModel,
    fermi_energy_ev: Float64[Array, ""],
    temperature_k: ScalarFloat,
    eta: ScalarFloat = 1.0e-4,
    *,
    allow_degenerate_value_only: ScalarBool = False,
) -> Float64[Array, "n_k n_chunk"]:
    """Assemble occupied intrinsic intensity from eigenvalues and band weights.

    This nondegenerate fast path shifts absolute eigenvalues by the Fermi
    energy exactly once and sums gauge-invariant Lorentzian band weights.
    The code evaluates occupation at sampled omega, never at a band eigenvalue.

    :see: :class:`~.test_spectral.TestAssembleSpectralIntensityBandsChunk`

    Parameters
    ----------
    eigenvalues_ev : Float64[Array, "n_k n_bands"]
        Absolute band energies in eV.
    band_weights : Float64[Array, "n_k n_chunk n_bands"]
        Explicit finite, nonnegative transition weights for each sample.
    omega_rel_fermi_ev : Float64[Array, " n_chunk"]
        Sampled energies ``E - E_F`` in eV.
    self_energy : SelfEnergyModel
        Validated causal self-energy carrier on the relative-energy axis.
    fermi_energy_ev : Float64[Array, ""]
        Absolute Fermi energy subtracted from every eigenvalue once.
    temperature_k : ScalarFloat
        Finite, strictly positive sample temperature in kelvin.
    eta : ScalarFloat, optional
        Positive regulator in eV. Default is ``1e-4``.
    allow_degenerate_value_only : ScalarBool, optional
        Admit exact or near-degenerate rows only for primal compatibility
        checks with already-formed complete invariant weights. Default is
        ``False``.

    Returns
    -------
    intensity : Float64[Array, "n_k n_chunk"]
        Intrinsic ``A(k, omega) f_FD(omega, T)`` in inverse eV.

    Raises
    ------
    EquinoxRuntimeError
        If an input is non-finite, a weight is negative, or a physical width
        or temperature is not strictly positive. Also raised when any band
        gap is below ``1e3 * EPS_DEG`` unless value-only evaluation is
        explicit.

    Notes
    -----
    The eigen route amortizes one eigendecomposition over all sampled
    energies. Its differentiated domain requires every adjacent band gap to
    be at least ``1e3 * EPS_DEG``. The explicit value-only exception emits no
    derivative claim. The function performs no convolution, normalization,
    or detector response; Plan 08a owns those downstream operations.
    """
    checked_eigenvalues: Float64[Array, "n_k n_bands"] = eqx.error_if(
        eigenvalues_ev,
        ~jnp.all(jnp.isfinite(eigenvalues_ev)),
        "assemble_spectral_intensity_bands_chunk: eigenvalues must be finite",
    )
    checked_eigenvalues = _checked_eigenvalue_domain(
        checked_eigenvalues,
        allow_degenerate_value_only,
        context="assemble_spectral_intensity_bands_chunk",
    )
    checked_weights: Float64[Array, "n_k n_chunk n_bands"] = eqx.error_if(
        band_weights,
        ~jnp.all(jnp.isfinite(band_weights) & (band_weights >= 0.0)),
        "assemble_spectral_intensity_bands_chunk: weights must be finite "
        "and nonnegative",
    )
    checked_fermi: Float64[Array, ""] = eqx.error_if(
        fermi_energy_ev,
        ~jnp.isfinite(fermi_energy_ev),
        "assemble_spectral_intensity_bands_chunk: fermi energy must be finite",
    )
    checked_omega: Float64[Array, " n_chunk"] = eqx.error_if(
        omega_rel_fermi_ev,
        ~jnp.all(jnp.isfinite(omega_rel_fermi_ev)),
        "assemble_spectral_intensity_bands_chunk: omega must be finite",
    )
    eigenvalues_rel: Float64[Array, "n_k n_bands"] = (
        checked_eigenvalues - checked_fermi
    )
    sigma: Complex128[Array, " n_chunk"] = evaluate_self_energy(
        checked_omega,
        self_energy,
    )
    eta_array: Float64[Array, ""] = jnp.asarray(eta, dtype=jnp.float64)
    eta_checked: Float64[Array, ""] = eqx.error_if(
        eta_array,
        ~jnp.isfinite(eta_array) | (eta_array <= 0.0),
        "assemble_spectral_intensity_bands_chunk: eta must be finite and "
        "positive",
    )
    spectral: Float64[Array, "n_k n_chunk"] = jax.vmap(
        lambda eigenvalues, weights: jax.vmap(
            _spectral_intensity_eigen_unchecked,
            in_axes=(None, 0, 0, 0, None),
        )(eigenvalues, weights, checked_omega, sigma, eta_checked)
    )(eigenvalues_rel, checked_weights)
    occupation: Float64[Array, " n_chunk"] = _sampled_fermi_occupation(
        checked_omega,
        temperature_k,
    )
    intensity: Float64[Array, "n_k n_chunk"] = spectral * occupation[None, :]
    return intensity


class _TransitionSourceSchedule(eqx.Module):
    """PRIVATE: Store traced Plan-06 inputs for block-local source assembly.

    The carrier contains compact Plan-03 kinematics and energy-independent
    matrix-element state, never a precomputed ``(K, E, 3)`` final-momentum or
    ``(K, E, B)`` transition tensor. The streamed driver reconstructs final
    momenta and source kets only for the live
    ``(k_chunk, omega_chunk)`` block.

    Attributes
    ----------
    k_i_cart : Float64[Array, "n_k_max 3"]
        Initial sample-frame crystal momenta in inverse Angstrom.
    final_norm : Float64[Array, "n_omega_max"]
        Vacuum final-momentum magnitude for each sampled energy.
    emission_energy_valid : Bool[Array, "n_omega_max"]
        Positive kinetic-energy and final-state-momentum mask.
    positions_cart : Float64[Array, "n_orb 3"]
        Orbital or Wannier centres in Cartesian Angstrom.
    depths : Float64[Array, "n_orb"]
        Orbital depths below the surface in Angstrom.
    polarization_sample_cart : Complex128[Array, "3"]
        Sample-frame Cartesian polarization after the one lab-to-sample map.
    mean_free_path_ang : Float64[Array, ""]
        Photoelectron intensity mean free path in Angstrom.
    radial : RadialSpec
        Shell-shared initial-state radial carrier.
    matrix_element : MatrixElementParams
        Shell scales and phase coordinates.
    quadrature : RadialQuadratureSpec
        Certified fixed radial quadrature.
    final_state : FinalStateSpec
        Direct plane-wave or Coulomb final-state selection.
    """

    k_i_cart: Float64[Array, "n_k_max 3"]
    final_norm: Float64[Array, " n_omega_max"]
    emission_energy_valid: Bool[Array, " n_omega_max"]
    positions_cart: Float64[Array, "n_orb 3"]
    depths: Float64[Array, " n_orb"]
    polarization_sample_cart: Complex128[Array, " 3"]
    mean_free_path_ang: Float64[Array, ""]
    radial: RadialSpec
    matrix_element: MatrixElementParams
    quadrature: RadialQuadratureSpec
    final_state: FinalStateSpec


def _validate_transition_source_schedule(
    schedule: _TransitionSourceSchedule,
    *,
    n_k_max: int,
    n_omega_max: int,
    n_orb: int,
) -> None:
    """PRIVATE: Validate the static axes of one padded source schedule.

    Notes
    -----
    Python shape checks run before tracing. The schedule must also share one
    orbital basis and radial-shell partition across its carriers.
    """
    if (
        schedule.k_i_cart.shape != (n_k_max, 3)
        or schedule.final_norm.shape != (n_omega_max,)
        or schedule.emission_energy_valid.shape != (n_omega_max,)
        or schedule.positions_cart.shape != (n_orb, 3)
        or schedule.depths.shape != (n_orb,)
        or schedule.polarization_sample_cart.shape != (3,)
        or schedule.mean_free_path_ang.ndim != 0
        or len(schedule.radial.basis.n) != n_orb
    ):
        raise ValueError(
            "transition source schedule axes must match the padded spectral "
            "and orbital dimensions"
        )
    if (
        schedule.radial.basis != schedule.matrix_element.basis
        or schedule.radial.radial_shell_index
        != schedule.matrix_element.radial_shell_index
    ):
        raise ValueError(
            "transition source radial and matrix-element carriers must share "
            "one basis and shell partition"
        )


def _transition_sources_for_block(
    schedule: _TransitionSourceSchedule,
    k_i_block: Float64[Array, "k_chunk 3"],
    k_f_block: Float64[Array, "k_chunk omega_chunk 3"],
    valid_block: Bool[Array, "k_chunk omega_chunk"],
) -> Complex128[Array, "k_chunk omega_chunk n_spin n_orb"]:
    """PRIVATE: Build only one live matrix-element source block.

    The helper replaces invalid padding before the Plan-06 primitives run. It
    restores exact zeros afterward. Every physically valid final momentum must
    be finite, nonzero, and on the registered zero-umklapp in-plane seam.

    Notes
    -----
    The energy-axis vectorization keeps only one chunk of radial values,
    transition channels, and outgoing sources live at a time.
    """

    def one_energy(
        final_momentum: Float64[Array, "k_chunk 3"],
        valid: Bool[Array, " k_chunk"],
    ) -> Complex128[Array, "k_chunk n_spin n_orb"]:
        """Construct the outgoing-spin source rows at one omega."""
        safe_initial: Float64[Array, "k_chunk 3"] = jnp.where(
            valid[:, None], k_i_block, 0.0
        )
        filler: Float64[Array, "k_chunk 3"] = jnp.broadcast_to(
            jnp.asarray([0.0, 0.0, 1.0], dtype=jnp.float64),
            final_momentum.shape,
        )
        safe_final: Float64[Array, "k_chunk 3"] = jnp.where(
            valid[:, None], final_momentum, filler
        )
        final_norm: Float64[Array, " k_chunk"] = jnp.linalg.norm(
            safe_final, axis=-1
        )
        invalid_physical: Bool[Array, " k_chunk"] = valid & (
            ~jnp.all(jnp.isfinite(k_i_block), axis=-1)
            | ~jnp.all(jnp.isfinite(final_momentum), axis=-1)
            | (jnp.linalg.norm(final_momentum, axis=-1) <= 0.0)
            | jnp.any(
                jnp.abs(final_momentum[:, :2] - k_i_block[:, :2])
                > G_PARALLEL_ATOL_INV_ANG,
                axis=-1,
            )
        )
        safe_final = eqx.error_if(
            safe_final,
            jnp.any(invalid_physical),
            "valid streamed final momenta must be finite, nonzero, and on "
            "the G_parallel=0 seam",
        )
        momentum_bohr_inv: Float64[Array, " k_chunk"] = (
            momentum_inv_ang_to_bohr_inv(final_norm)
        )
        bvals: Complex128[Array, "k_chunk n_orb 2"] = radial_bvals(
            schedule.radial,
            momentum_bohr_inv,
            schedule.quadrature,
            schedule.final_state,
        )
        channels: Complex128[Array, "k_chunk n_spin n_orb_per_spin 3"] = (
            orbital_transition_channels(
                safe_initial,
                safe_final,
                schedule.positions_cart,
                schedule.depths,
                bvals,
                schedule.matrix_element,
                schedule.mean_free_path_ang,
                schedule.radial.basis,
            )
        )
        rows: Complex128[Array, "k_chunk n_spin n_orb_per_spin"] = (
            contract_polarization(
                channels,
                schedule.polarization_sample_cart,
            )
        )
        sources: Complex128[Array, "k_chunk n_spin n_orb"] = transition_source(
            rows
        )
        masked_sources: Complex128[Array, "k_chunk n_spin n_orb"] = jnp.where(
            valid[:, None, None],
            sources,
            0.0,
        )
        return masked_sources

    sources: Complex128[Array, "k_chunk omega_chunk n_spin n_orb"] = jax.vmap(
        one_energy, in_axes=(1, 1), out_axes=1
    )(
        k_f_block,
        valid_block,
    )
    return sources


def _stream_spectral_intensity(  # noqa: DOC503, PLR0913, PLR0915 -- scan contract.
    hamiltonians_ev: Complex128[Array, "n_k_max n_orb n_orb"],
    omega_rel_fermi_ev: Float64[Array, " n_omega_max"],
    k_valid: Bool[Array, " n_k_max"],
    omega_valid: Bool[Array, " n_omega_max"],
    transition_schedule: _TransitionSourceSchedule,
    self_energy: SelfEnergyModel,
    fermi_energy_ev: Float64[Array, ""],
    temperature_k: ScalarFloat,
    eta: ScalarFloat = 1.0e-4,
    *,
    k_chunk: int = 32,
    omega_chunk: int = 32,
    checkpoint: bool = True,
) -> Float64[Array, "n_k_max n_omega_max"]:
    """PRIVATE: Stream padded chunks without a ``(K,E,B)`` source.

    Parameters
    ----------
    hamiltonians_ev : Complex128[Array, "n_k_max n_orb n_orb"]
        Padded absolute-energy Hermitian Hamiltonians in eV.
    omega_rel_fermi_ev : Float64[Array, " n_omega_max"]
        Padded sampled relative-energy axis in eV.
    k_valid : Bool[Array, " n_k_max"]
        Validity mask for the padded k axis.
    omega_valid : Bool[Array, " n_omega_max"]
        Validity mask for the padded energy axis.
    transition_schedule : _TransitionSourceSchedule
        Plan-03 kinematics and Plan-06 carriers used to construct only the
        current source block.
    self_energy : SelfEnergyModel
        Validated causal self-energy carrier.
    fermi_energy_ev : Float64[Array, ""]
        Absolute Fermi energy in eV.
    temperature_k : ScalarFloat
        Finite, strictly positive temperature in kelvin.
    eta : ScalarFloat, optional
        Positive regulator in eV. Default is ``1e-4``.
    k_chunk : int, optional
        Positive static k chunk size. Default is 32.
    omega_chunk : int, optional
        Positive static energy chunk size. Default is 32.
    checkpoint : bool, optional
        Static selector for rematerializing each two-dimensional chunk.

    Returns
    -------
    intensity : Float64[Array, "n_k_max n_omega_max"]
        Masked intrinsic intensity on the complete padded schedule.

    Raises
    ------
    ValueError
        If padded axes, source carriers, or chunk sizes are inconsistent.
    EquinoxRuntimeError
        If a physically valid final momentum or traced carrier value leaves
        its registered domain.

    Notes
    -----
    Callers keep padded shapes and the chunk schedule fixed across a sweep;
    only masks and physical leaves vary. Each scan step constructs radial
    channels, polarized outgoing-spin source kets, resolvent solutions, and
    the spin-incoherent reduction for one ``(k_chunk, omega_chunk)`` block.
    No complete ``(K, E, B)`` transition tensor exists. Checkpointing bounds
    reverse-mode tape without changing values.
    """
    if type(k_chunk) is not int or k_chunk <= 0:
        raise ValueError("k_chunk must be a positive integer")
    if type(omega_chunk) is not int or omega_chunk <= 0:
        raise ValueError("omega_chunk must be a positive integer")
    n_k_max: int = hamiltonians_ev.shape[0]
    n_omega_max: int = omega_rel_fermi_ev.shape[0]
    n_orb: int = hamiltonians_ev.shape[-1]
    batch_matrix_ndim: int = 3
    if (
        hamiltonians_ev.ndim != batch_matrix_ndim
        or hamiltonians_ev.shape[-2] != n_orb
        or k_valid.shape != (n_k_max,)
        or omega_valid.shape != (n_omega_max,)
    ):
        raise ValueError("streamed spectral padded axes are inconsistent")
    _validate_transition_source_schedule(
        transition_schedule,
        n_k_max=n_k_max,
        n_omega_max=n_omega_max,
        n_orb=n_orb,
    )
    checked_final_norm: Float64[Array, " n_omega_max"] = eqx.error_if(
        transition_schedule.final_norm,
        ~jnp.all(jnp.isfinite(transition_schedule.final_norm))
        | jnp.any(transition_schedule.final_norm < 0.0)
        | jnp.any(
            transition_schedule.emission_energy_valid
            & (transition_schedule.final_norm == 0.0)
        ),
        "streamed final-momentum magnitudes must be finite and nonnegative; "
        "active magnitudes must be strictly positive",
    )
    if n_k_max % k_chunk:
        raise ValueError("k_chunk must divide the padded k axis")
    if n_omega_max % omega_chunk:
        raise ValueError("omega_chunk must divide the padded omega axis")
    n_k_blocks: int = n_k_max // k_chunk
    n_omega_blocks: int = n_omega_max // omega_chunk
    hamiltonian_blocks: Complex128[Array, "n_k_block k_chunk n_orb n_orb"] = (
        jnp.reshape(
            hamiltonians_ev,
            (n_k_blocks, k_chunk, n_orb, n_orb),
        )
    )
    initial_blocks: Float64[Array, "n_k_block k_chunk 3"] = jnp.reshape(
        transition_schedule.k_i_cart,
        (n_k_blocks, k_chunk, 3),
    )
    final_norm_blocks: Float64[Array, "n_omega_block omega_chunk"] = (
        jnp.reshape(
            checked_final_norm,
            (n_omega_blocks, omega_chunk),
        )
    )
    emission_energy_blocks: Bool[Array, "n_omega_block omega_chunk"] = (
        jnp.reshape(
            transition_schedule.emission_energy_valid,
            (n_omega_blocks, omega_chunk),
        )
    )
    omega_blocks: Float64[Array, "n_omega_block omega_chunk"] = jnp.reshape(
        omega_rel_fermi_ev,
        (n_omega_blocks, omega_chunk),
    )
    k_mask_blocks: Bool[Array, "n_k_block k_chunk"] = jnp.reshape(
        k_valid,
        (n_k_blocks, k_chunk),
    )
    omega_mask_blocks: Bool[Array, "n_omega_block omega_chunk"] = jnp.reshape(
        omega_valid,
        (n_omega_blocks, omega_chunk),
    )

    def assemble_block(
        hamiltonian_block: Complex128[Array, "k_chunk n_orb n_orb"],
        k_i_block: Float64[Array, "k_chunk 3"],
        final_norm_block: Float64[Array, " omega_chunk"],
        emission_energy_block: Bool[Array, " omega_chunk"],
        k_mask: Bool[Array, " k_chunk"],
        omega_mask: Bool[Array, " omega_chunk"],
        omega_block: Float64[Array, " omega_chunk"],
    ) -> Float64[Array, "k_chunk omega_chunk"]:
        """Compute one live block from reconstructed kinematics and solves."""
        parallel_sq: Float64[Array, " k_chunk"] = jnp.sum(
            k_i_block[:, :2] * k_i_block[:, :2], axis=-1
        )
        normal_sq: Float64[Array, "k_chunk omega_chunk"] = (
            final_norm_block[None, :] * final_norm_block[None, :]
            - parallel_sq[:, None]
        )
        emission_valid: Bool[Array, "k_chunk omega_chunk"] = (
            emission_energy_block[None, :] & (normal_sq > 0.0)
        )
        valid_block: Bool[Array, "k_chunk omega_chunk"] = (
            k_mask[:, None] & omega_mask[None, :] & emission_valid
        )
        safe_normal_sq: Float64[Array, "k_chunk omega_chunk"] = jnp.where(
            valid_block,
            normal_sq,
            1.0,
        )
        final_kz: Float64[Array, "k_chunk omega_chunk"] = jnp.where(
            valid_block,
            jnp.sqrt(safe_normal_sq),
            0.0,
        )
        final_kx: Float64[Array, "k_chunk omega_chunk"] = jnp.broadcast_to(
            k_i_block[:, 0, None], final_kz.shape
        )
        final_ky: Float64[Array, "k_chunk omega_chunk"] = jnp.broadcast_to(
            k_i_block[:, 1, None], final_kz.shape
        )
        k_f_block: Float64[Array, "k_chunk omega_chunk 3"] = jnp.stack(
            (final_kx, final_ky, final_kz), axis=-1
        )
        sources: Complex128[Array, "k_chunk omega_chunk n_spin n_orb"] = (
            _transition_sources_for_block(
                transition_schedule,
                k_i_block,
                k_f_block,
                valid_block,
            )
        )
        intensity: Float64[Array, "k_chunk omega_chunk"] = (
            assemble_spectral_intensity_chunk(
                hamiltonian_block,
                sources,
                omega_block,
                self_energy,
                fermi_energy_ev,
                temperature_k,
                eta,
            )
        )
        masked_intensity: Float64[Array, "k_chunk omega_chunk"] = jnp.where(
            valid_block,
            intensity,
            0.0,
        )
        return masked_intensity

    block_function: Any = (
        jax.checkpoint(assemble_block) if checkpoint else assemble_block
    )

    def scan_k_block(
        carry: None,
        arguments: Tuple[
            Complex128[Array, "k_chunk n_orb n_orb"],
            Float64[Array, "k_chunk 3"],
            Bool[Array, " k_chunk"],
        ],
    ) -> Tuple[
        None,
        Float64[Array, "n_omega_block k_chunk omega_chunk"],
    ]:
        """Stream every energy block for one k block."""
        hamiltonian_block: Complex128[Array, "k_chunk n_orb n_orb"]
        k_i_block: Float64[Array, "k_chunk 3"]
        k_mask: Bool[Array, " k_chunk"]
        (
            hamiltonian_block,
            k_i_block,
            k_mask,
        ) = arguments

        def scan_omega_block(
            inner_carry: None,
            inner_arguments: Tuple[
                Float64[Array, " omega_chunk"],
                Float64[Array, " omega_chunk"],
                Bool[Array, " omega_chunk"],
                Bool[Array, " omega_chunk"],
            ],
        ) -> Tuple[None, Float64[Array, "k_chunk omega_chunk"]]:
            """Construct, assemble, and mask one omega block."""
            omega_block: Float64[Array, " omega_chunk"]
            final_norm_block: Float64[Array, " omega_chunk"]
            emission_energy_block: Bool[Array, " omega_chunk"]
            omega_mask: Bool[Array, " omega_chunk"]
            (
                omega_block,
                final_norm_block,
                emission_energy_block,
                omega_mask,
            ) = inner_arguments
            values: Float64[Array, "k_chunk omega_chunk"] = block_function(
                hamiltonian_block,
                k_i_block,
                final_norm_block,
                emission_energy_block,
                k_mask,
                omega_mask,
                omega_block,
            )
            result: Tuple[None, Float64[Array, "k_chunk omega_chunk"]] = (
                inner_carry,
                values,
            )
            return result

        outputs: Float64[Array, "n_omega_block k_chunk omega_chunk"]
        _, outputs = jax.lax.scan(
            scan_omega_block,
            None,
            (
                omega_blocks,
                final_norm_blocks,
                emission_energy_blocks,
                omega_mask_blocks,
            ),
        )
        result: Tuple[
            None,
            Float64[Array, "n_omega_block k_chunk omega_chunk"],
        ] = (carry, outputs)
        return result

    scanned: Float64[Array, "n_k_block n_omega_block k_chunk omega_chunk"]
    _, scanned = jax.lax.scan(
        scan_k_block,
        None,
        (
            hamiltonian_blocks,
            initial_blocks,
            k_mask_blocks,
        ),
    )
    intensity: Float64[Array, "n_k_max n_omega_max"] = jnp.reshape(
        jnp.transpose(scanned, (0, 2, 1, 3)),
        (n_k_max, n_omega_max),
    )
    return intensity


__all__: list[str] = [
    "assemble_spectral_intensity_bands_chunk",
    "assemble_spectral_intensity_chunk",
    "evaluate_self_energy",
    "projected_spectral_density_resolvent",
    "spectral_intensity_eigen",
    "spectral_intensity_resolvent",
]
