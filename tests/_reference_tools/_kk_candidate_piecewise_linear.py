"""Piecewise-linear candidate for the Plan 07 KK algorithm spike."""

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

NAME = "cell-integrated piecewise-linear"
DESCRIPTION = (
    "Analytic linear-cell PV integrals with exactly cancelling node logs."
)


def core_pv_transform(
    core_grid_ev: jax.Array,
    core_imag_values: jax.Array,
    queries_ev: jax.Array,
) -> jax.Array:
    """Return the unsubtracted core principal-value contribution.

    Each linear cell is integrated analytically. Logarithms are regrouped by
    grid node so adjacent coefficients cancel exactly at an on-node query.

    Parameters
    ----------
    core_grid_ev
        Uniform core grid in eV.
    core_imag_values
        Dynamic imaginary self-energy values on the core grid.
    queries_ev
        Query energies in eV.

    Returns
    -------
    jax.Array
        Unsubtracted core contribution at each query.

    Implementation Logic
    --------------------
    1. Construct each cell slope.
    2. Regroup cell endpoint logarithms by grid node.
    3. Guard zero log arguments before evaluating the logarithm.
    4. Add the nonsingular slope integrals and divide by pi.
    """
    grid = jnp.asarray(core_grid_ev, dtype=jnp.float64)
    values = jnp.asarray(core_imag_values, dtype=jnp.float64)
    queries = jnp.asarray(queries_ev, dtype=jnp.float64)

    cell_widths = jnp.diff(grid)
    slopes = jnp.diff(values) / cell_widths
    query_offsets = queries[:, None] - grid[None, :]

    left_coefficients = -(values[0] + slopes[0] * query_offsets[:, 0])
    interior_coefficients = (slopes[:-1] - slopes[1:])[
        None, :
    ] * query_offsets[:, 1:-1]
    right_coefficients = values[-1] + slopes[-1] * query_offsets[:, -1]
    log_coefficients = jnp.concatenate(
        (
            left_coefficients[:, None],
            interior_coefficients,
            right_coefficients[:, None],
        ),
        axis=1,
    )

    distances = grid[None, :] - queries[:, None]
    safe_distances = jnp.where(distances == 0.0, 1.0, distances)
    log_terms = log_coefficients * jnp.log(jnp.abs(safe_distances))
    slope_integrals = jnp.sum(slopes * cell_widths)
    return (slope_integrals + jnp.sum(log_terms, axis=1)) / jnp.pi
