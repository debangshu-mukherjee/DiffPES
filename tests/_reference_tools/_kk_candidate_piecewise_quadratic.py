"""Provide a cell-integrated piecewise-quadratic core PV transform.

Extended Summary
----------------
This comparison operator integrates adjacent quadratic cell pairs. It contains
no mode dispatch, tail treatment, or carrier objects.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from beartype import beartype
from jaxtyping import Array, Bool, Float64, Int64, jaxtyped

jax.config.update("jax_enable_x64", True)

NAME = "cell_integrated_piecewise_quadratic"
DESCRIPTION = (
    "Analytic cell PV integration of quadratic interpolants on adjacent "
    "cell pairs, "
    "with regrouped node logs and double-where node cancellation."
)
MINIMUM_NODES: int = 3


@jaxtyped(typechecker=beartype)
def core_pv_transform(
    core_grid_ev: Float64[Array, " n_kk"],
    core_imag_values: Float64[Array, " n_kk"],
    queries_ev: Float64[Array, " n_query"],
) -> Float64[Array, " n_query"]:
    """Return the unsubtracted core principal-value contribution.

    Group cells as ``(0, 1), (2, 3), ...``. Both cells in a pair use the
    quadratic through their three bounding nodes. For an odd cell count, use
    the final three nodes for the final cell. This rule integrates every cell
    exactly once.

    For a query ``q``, write a cell's interpolant as
    ``Q(u) = Q0 + Q1*u + Q2*u**2``, ``u = w-q``.  Its exact integral is
    ``Q0*log|u2/u1| + Q1*(u2-u1) + Q2*(u2**2-u1**2)/2``. Log terms
    regroup by grid node before evaluation. This exposes the exact cancellation
    at interior nodes.

    Implementation Logic
    --------------------
    The function assigns one three-node stencil to each cell. It integrates
    the regular polynomial terms and regroups the logarithmic coefficients.

    Parameters
    ----------
    core_grid_ev : Float64[Array, " n_kk"]
        Uniform core energy grid in eV.
    core_imag_values : Float64[Array, " n_kk"]
        Dynamic imaginary self-energy values in eV.
    queries_ev : Float64[Array, " n_query"]
        Query energies in eV.

    Returns
    -------
    result_ev : Float64[Array, " n_query"]
        Unsubtracted core principal-value contribution in eV.

    Raises
    ------
    ValueError
        If the core grid contains fewer than three nodes.
    """
    grid: Float64[Array, " n_kk"] = jnp.asarray(
        core_grid_ev, dtype=jnp.float64
    )
    values: Float64[Array, " n_kk"] = jnp.asarray(
        core_imag_values, dtype=jnp.float64
    )
    queries: Float64[Array, " n_query"] = jnp.asarray(
        queries_ev, dtype=jnp.float64
    )

    n_kk: int = grid.shape[0]
    if n_kk < MINIMUM_NODES:
        msg: str = (
            "piecewise-quadratic PV transform requires at least three grid "
            "nodes"
        )
        raise ValueError(msg)

    h: Float64[Array, ""] = grid[1] - grid[0]
    cell_indices: Int64[Array, " n_cell"] = jnp.arange(n_kk - 1)
    stencil_starts: Int64[Array, " n_cell"] = jnp.minimum(
        2 * (cell_indices // 2), n_kk - 3
    )

    y0: Float64[Array, " n_cell"] = values[stencil_starts]
    y1: Float64[Array, " n_cell"] = values[stencil_starts + 1]
    y2: Float64[Array, " n_cell"] = values[stencil_starts + 2]
    x0: Float64[Array, " n_cell"] = grid[stencil_starts]

    # Q(w) = y0 + linear*(w-x0) + quadratic*(w-x0)^2.
    linear: Float64[Array, " n_cell"] = (-3.0 * y0 + 4.0 * y1 - y2) / (2.0 * h)
    quadratic: Float64[Array, " n_cell"] = (y0 - 2.0 * y1 + y2) / (2.0 * h**2)

    offset: Float64[Array, "n_query n_cell"] = queries[:, None] - x0[None, :]
    q0: Float64[Array, "n_query n_cell"] = (
        y0[None, :] + linear[None, :] * offset + quadratic[None, :] * offset**2
    )
    q1: Float64[Array, "n_query n_cell"] = (
        linear[None, :] + 2.0 * quadratic[None, :] * offset
    )

    u_left: Float64[Array, "n_query n_cell"] = (
        grid[:-1][None, :] - queries[:, None]
    )
    u_right: Float64[Array, "n_query n_cell"] = (
        grid[1:][None, :] - queries[:, None]
    )
    regular: Float64[Array, "n_query n_cell"] = q1 * (u_right - u_left)
    regular += 0.5 * quadratic[None, :] * (u_right**2 - u_left**2)

    # Regroup q0*(log|u_right|-log|u_left|) by node.  At an interior
    # node the coefficient is q0_left-q0_right, which vanishes because both
    # interpolants hit the sampled node value.  The first where makes that
    # cancellation bit-exact; the second guards the log argument, so reverse AD
    # never evaluates log(0) or its derivative.
    log_coefficients: Float64[Array, "n_query n_kk"] = jnp.concatenate(
        (
            -q0[:, :1],
            q0[:, :-1] - q0[:, 1:],
            q0[:, -1:],
        ),
        axis=1,
    )
    distances: Float64[Array, "n_query n_kk"] = (
        grid[None, :] - queries[:, None]
    )
    at_node: Bool[Array, "n_query n_kk"] = distances == 0.0
    log_coefficients = jnp.where(at_node, 0.0, log_coefficients)
    safe_distances: Float64[Array, "n_query n_kk"] = jnp.where(
        at_node, 1.0, distances
    )
    logarithmic: Float64[Array, " n_query"] = jnp.sum(
        log_coefficients * jnp.log(jnp.abs(safe_distances)), axis=1
    )

    result_ev: Float64[Array, " n_query"] = (
        jnp.sum(regular, axis=1) + logarithmic
    ) / jnp.pi
    return result_ev
