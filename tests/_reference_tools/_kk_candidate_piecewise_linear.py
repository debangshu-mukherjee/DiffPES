"""Provide a piecewise-linear Kramers--Kronig comparison operator.

Extended Summary
----------------
This module integrates linear cell interpolants analytically. It regroups
endpoint logarithms so an on-node query has exact coefficient cancellation.
"""

import jax
import jax.numpy as jnp
from beartype import beartype
from jaxtyping import Array, Float64, jaxtyped

jax.config.update("jax_enable_x64", True)

NAME = "cell-integrated piecewise-linear"
DESCRIPTION = (
    "Analytic linear-cell PV integrals with exactly cancelling node logs."
)


@jaxtyped(typechecker=beartype)
def core_pv_transform(
    core_grid_ev: Float64[Array, " n_kk"],
    core_imag_values: Float64[Array, " n_kk"],
    queries_ev: Float64[Array, " n_query"],
) -> Float64[Array, " n_query"]:
    """Return the unsubtracted core principal-value contribution.

    Integrate each linear cell analytically. Regroup logarithms by grid node.
    Adjacent coefficients then cancel exactly at an on-node query.

    Implementation Logic
    --------------------
    The function forms each cell slope and regroups endpoint logarithms by
    grid node. It guards zero logarithm arguments and adds the slope terms.

    Parameters
    ----------
    core_grid_ev : Float64[Array, " n_kk"]
        Uniform core grid in eV.
    core_imag_values : Float64[Array, " n_kk"]
        Dynamic imaginary self-energy values on the core grid.
    queries_ev : Float64[Array, " n_query"]
        Query energies in eV.

    Returns
    -------
    result_ev : Float64[Array, " n_query"]
        Unsubtracted core contribution at each query.
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

    cell_widths: Float64[Array, " n_cell"] = jnp.diff(grid)
    slopes: Float64[Array, " n_cell"] = jnp.diff(values) / cell_widths
    query_offsets: Float64[Array, "n_query n_kk"] = (
        queries[:, None] - grid[None, :]
    )

    left_coefficients: Float64[Array, " n_query"] = -(
        values[0] + slopes[0] * query_offsets[:, 0]
    )
    interior_coefficients: Float64[Array, "n_query n_interior"] = (
        slopes[:-1] - slopes[1:]
    )[None, :] * query_offsets[:, 1:-1]
    right_coefficients: Float64[Array, " n_query"] = (
        values[-1] + slopes[-1] * query_offsets[:, -1]
    )
    log_coefficients: Float64[Array, "n_query n_kk"] = jnp.concatenate(
        (
            left_coefficients[:, None],
            interior_coefficients,
            right_coefficients[:, None],
        ),
        axis=1,
    )

    distances: Float64[Array, "n_query n_kk"] = (
        grid[None, :] - queries[:, None]
    )
    safe_distances: Float64[Array, "n_query n_kk"] = jnp.where(
        distances == 0.0, 1.0, distances
    )
    log_terms: Float64[Array, "n_query n_kk"] = log_coefficients * jnp.log(
        jnp.abs(safe_distances)
    )
    slope_integrals: Float64[Array, ""] = jnp.sum(slopes * cell_widths)
    result_ev: Float64[Array, " n_query"] = (
        slope_integrals + jnp.sum(log_terms, axis=1)
    ) / jnp.pi
    return result_ev
