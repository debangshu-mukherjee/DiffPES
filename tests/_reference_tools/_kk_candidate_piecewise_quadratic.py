"""Candidate B: cell-integrated piecewise-quadratic core PV transform.

This is a preproduction algorithm-spike implementation.  It deliberately contains no
mode dispatch, tail treatment, or production carrier types.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float64

jax.config.update("jax_enable_x64", True)

NAME = "cell_integrated_piecewise_quadratic"
DESCRIPTION = (
    "Analytic cell PV integration of quadratic interpolants on adjacent cell pairs, "
    "with regrouped node logs and double-where node cancellation."
)


def core_pv_transform(
    core_grid_ev: Float64[Array, " n_kk"],
    core_imag_values: Float64[Array, " n_kk"],
    queries_ev: Float64[Array, " n_query"],
) -> Float64[Array, " n_query"]:
    """Return the unsubtracted core principal-value contribution to Sigma'.

    Cells are grouped as ``(0, 1), (2, 3), ...``.  Both cells in a pair use the
    quadratic through their three bounding nodes.  If the number of cells is odd
    (equivalently, ``n_kk`` is even), the final cell uses the quadratic through the
    final three nodes.  Thus no cell is omitted or integrated twice.

    For a query ``q``, write a cell's interpolant as
    ``Q(u) = Q0 + Q1*u + Q2*u**2``, ``u = w-q``.  Its exact integral is
    ``Q0*log|u2/u1| + Q1*(u2-u1) + Q2*(u2**2-u1**2)/2``.  Log terms are regrouped by
    grid node before evaluation, exposing the exact cancellation at interior nodes.
    """
    grid = jnp.asarray(core_grid_ev, dtype=jnp.float64)
    values = jnp.asarray(core_imag_values, dtype=jnp.float64)
    queries = jnp.asarray(queries_ev, dtype=jnp.float64)

    n_kk = grid.shape[0]
    if n_kk < 3:
        msg = "piecewise-quadratic PV transform requires at least three grid nodes"
        raise ValueError(msg)

    h = grid[1] - grid[0]
    cell_indices = jnp.arange(n_kk - 1)
    stencil_starts = jnp.minimum(2 * (cell_indices // 2), n_kk - 3)

    y0 = values[stencil_starts]
    y1 = values[stencil_starts + 1]
    y2 = values[stencil_starts + 2]
    x0 = grid[stencil_starts]

    # Q(w) = y0 + linear*(w-x0) + quadratic*(w-x0)^2.
    linear = (-3.0 * y0 + 4.0 * y1 - y2) / (2.0 * h)
    quadratic = (y0 - 2.0 * y1 + y2) / (2.0 * h**2)

    offset = queries[:, None] - x0[None, :]
    q0 = (
        y0[None, :] + linear[None, :] * offset + quadratic[None, :] * offset**2
    )
    q1 = linear[None, :] + 2.0 * quadratic[None, :] * offset

    u_left = grid[:-1][None, :] - queries[:, None]
    u_right = grid[1:][None, :] - queries[:, None]
    regular = q1 * (u_right - u_left)
    regular += 0.5 * quadratic[None, :] * (u_right**2 - u_left**2)

    # Regroup q0*(log|u_right|-log|u_left|) by node.  At an interior
    # node the coefficient is q0_left-q0_right, which vanishes because both
    # interpolants hit the sampled node value.  The first where makes that
    # cancellation bit-exact; the second guards the log argument, so reverse AD
    # never evaluates log(0) or its derivative.
    log_coefficients = jnp.concatenate(
        (
            -q0[:, :1],
            q0[:, :-1] - q0[:, 1:],
            q0[:, -1:],
        ),
        axis=1,
    )
    distances = grid[None, :] - queries[:, None]
    at_node = distances == 0.0
    log_coefficients = jnp.where(at_node, 0.0, log_coefficients)
    safe_distances = jnp.where(at_node, 1.0, distances)
    logarithmic = jnp.sum(
        log_coefficients * jnp.log(jnp.abs(safe_distances)), axis=1
    )

    return (jnp.sum(regular, axis=1) + logarithmic) / jnp.pi
