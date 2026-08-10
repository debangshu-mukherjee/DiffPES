"""Rejected opposite-parity Maclaurin control for the Kramers--Kronig study.

This module deliberately reproduces the rejected point-sampled transform
path.  It first transforms point samples at the core-grid nodes and then
uses one cubic-Hermite interpolant for arbitrary queries.  Production code
must not use this post-transform interpolation scheme.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from beartype.typing import Tuple
from jax import Array

jax.config.update("jax_enable_x64", True)

NAME: str = "opposite-parity Maclaurin control"
DESCRIPTION: str = "Rejected point-sampled Maclaurin rule with cubic-Hermite post-interpolation."


def _node_transform(
    core_grid_ev: Array,
    core_imag_values: Array,
) -> Array:
    """PRIVATE: Apply the opposite-parity Maclaurin rule at every grid node.

    Parameters
    ----------
    core_grid_ev : Array
        Uniform core energy grid in eV.
    core_imag_values : Array
        Imaginary self-energy samples in eV at the grid nodes.

    Returns
    -------
    result_ev : Array
        Point-sampled principal-value transform in eV at every node.

    Implementation Logic
    --------------------
    The rule couples only node pairs with opposite index parity. Each
    such pair receives the weight ``(2 h / pi) / (omega_j - omega_i)``
    with grid spacing ``h``; same-parity pairs receive zero. One dense
    matrix-vector product then yields all node values at once.
    """
    spacing_ev: Array = core_grid_ev[1] - core_grid_ev[0]
    row_indices: Array = jnp.arange(core_grid_ev.size)[:, None]
    column_indices: Array = jnp.arange(core_grid_ev.size)[None, :]
    opposite_parity: Array = (row_indices + column_indices) % 2 == 1
    differences_ev: Array = core_grid_ev[None, :] - core_grid_ev[:, None]
    safe_differences_ev: Array = jnp.where(
        opposite_parity,
        differences_ev,
        1.0,
    )
    weights: Array = jnp.where(
        opposite_parity,
        (2.0 * spacing_ev / jnp.pi) / safe_differences_ev,
        0.0,
    )
    result_ev: Array = weights @ core_imag_values
    return result_ev


def _node_slopes(core_grid_ev: Array, node_values: Array) -> Array:
    """PRIVATE: Return second-order centered and one-sided node derivatives.

    Parameters
    ----------
    core_grid_ev : Array
        Uniform core energy grid in eV.
    node_values : Array
        Transformed values in eV at the grid nodes.

    Returns
    -------
    result : Array
        First-derivative estimate at every node, in eV per eV.

    Implementation Logic
    --------------------
    Interior nodes use the centered difference over ``2 h``. The first
    and last nodes use the matching second-order three-point one-sided
    stencils, so every slope keeps the same truncation order.
    """
    spacing_ev: Array = core_grid_ev[1] - core_grid_ev[0]
    interior: Array = (node_values[2:] - node_values[:-2]) / (2.0 * spacing_ev)
    left: Array = (
        -3.0 * node_values[0] + 4.0 * node_values[1] - node_values[2]
    ) / (2.0 * spacing_ev)
    right: Array = (
        3.0 * node_values[-1] - 4.0 * node_values[-2] + node_values[-3]
    ) / (2.0 * spacing_ev)
    result: Array = jnp.concatenate((left[None], interior, right[None]))
    return result


def _cubic_hermite(
    core_grid_ev: Array,
    node_values: Array,
    queries_ev: Array,
) -> Array:
    """PRIVATE: Evaluate the rejected post-transform cubic-Hermite interpolant.

    Parameters
    ----------
    core_grid_ev : Array
        Uniform core energy grid in eV.
    node_values : Array
        Point-sampled transform values in eV at the grid nodes.
    queries_ev : Array
        Query energies in eV inside the core grid.

    Returns
    -------
    result_ev : Array
        Interpolated transform values in eV at the queries.

    Implementation Logic
    --------------------
    A clipped ``searchsorted`` locates the cell of every query. The
    normalized cell coordinate feeds the four Hermite basis cubics.
    Node values and spacing-scaled ``_node_slopes`` derivatives then
    combine into the C1 piecewise-cubic value.
    """
    spacing_ev: Array = core_grid_ev[1] - core_grid_ev[0]
    slopes: Array = _node_slopes(core_grid_ev, node_values)
    left_indices: Array = (
        jnp.searchsorted(
            core_grid_ev,
            queries_ev,
            side="right",
        )
        - 1
    )
    left_indices = jnp.clip(left_indices, 0, core_grid_ev.size - 2)
    coordinate: Array = (queries_ev - core_grid_ev[left_indices]) / spacing_ev
    coordinate_squared: Array = coordinate * coordinate
    coordinate_cubed: Array = coordinate_squared * coordinate
    basis_00: Array = 2.0 * coordinate_cubed - 3.0 * coordinate_squared + 1.0
    basis_10: Array = coordinate_cubed - 2.0 * coordinate_squared + coordinate
    basis_01: Array = -2.0 * coordinate_cubed + 3.0 * coordinate_squared
    basis_11: Array = coordinate_cubed - coordinate_squared
    result_ev: Array = (
        basis_00 * node_values[left_indices]
        + basis_10 * spacing_ev * slopes[left_indices]
        + basis_01 * node_values[left_indices + 1]
        + basis_11 * spacing_ev * slopes[left_indices + 1]
    )
    return result_ev


def core_pv_transform(
    core_grid_ev: Array,
    core_imag_values: Array,
    queries_ev: Array,
) -> Array:
    """Return the unsubtracted core principal-value contribution.

    Implementation Logic
    --------------------
    Apply the opposite-parity Maclaurin matrix on the uniform core grid.
    Evaluate arbitrary queries with one cubic-Hermite interpolant.
    """
    node_values: Array = _node_transform(core_grid_ev, core_imag_values)
    result_ev: Array = _cubic_hermite(core_grid_ev, node_values, queries_ev)
    return result_ev


def _wigner_calibration(node_count: int) -> Tuple[float, float]:
    """PRIVATE: Measure maximum Wigner value and query-derivative errors.

    Parameters
    ----------
    node_count : int
        Number of uniform core-grid nodes on ``[-8, 8]`` eV.

    Returns
    -------
    errors : tuple[float, float]
        Maximum absolute value error in eV and maximum absolute
        query-derivative error over the in-band queries.

    Implementation Logic
    --------------------
    The fixture is the Wigner semicircle with half-width 1.5 eV and
    coupling 0.2 eV^2, whose in-band transform is ``prefactor * omega``.
    The function evaluates the control on 1001 queries in ``[-1, 1]``
    eV, differentiates one scalar wrapper with ``jax.vmap(jax.grad)``,
    and returns both maximum deviations from the analytic line.
    """
    core_grid_ev: Array = jnp.linspace(
        -8.0, 8.0, node_count, dtype=jnp.float64
    )
    band_ev: float = 1.5
    coupling_ev2: float = 0.2
    prefactor: float = 2.0 * coupling_ev2 / band_ev**2
    radicand_ev2: Array = jnp.maximum(band_ev**2 - core_grid_ev**2, 0.0)
    core_imag_values: Array = -prefactor * jnp.sqrt(radicand_ev2)
    queries_ev: Array = jnp.linspace(-1.0, 1.0, 1001, dtype=jnp.float64)

    def scalar_transform(query_ev: Array) -> Array:
        value: Array = core_pv_transform(
            core_grid_ev,
            core_imag_values,
            query_ev[None],
        )[0]
        return value

    values_ev: Array = core_pv_transform(
        core_grid_ev,
        core_imag_values,
        queries_ev,
    )
    derivatives: Array = jax.vmap(jax.grad(scalar_transform))(queries_ev)
    value_error_ev: Array = jnp.max(
        jnp.abs(values_ev - prefactor * queries_ev)
    )
    derivative_error: Array = jnp.max(jnp.abs(derivatives - prefactor))
    return float(value_error_ev), float(derivative_error)


def main() -> None:
    """Print calibration measurements for the rejected control."""
    quoted_value_error: float = 1.106e-5
    quoted_derivative_error: float = 6.376e-3
    value_4096, derivative_4096 = _wigner_calibration(4096)
    value_8192, derivative_8192 = _wigner_calibration(8192)
    print(
        "4096 value error: "
        f"{value_4096:.12e} (quoted {quoted_value_error:.12e}, "
        f"ratio {value_4096 / quoted_value_error:.6f})"
    )
    print(
        "4096 derivative error: "
        f"{derivative_4096:.12e} (quoted {quoted_derivative_error:.12e}, "
        f"ratio {derivative_4096 / quoted_derivative_error:.6f})"
    )
    print(
        "8192 value error: "
        f"{value_8192:.12e} (4096/8192 ratio {value_4096 / value_8192:.6f})"
    )
    print(
        "8192 derivative error: "
        f"{derivative_8192:.12e} "
        f"(4096/8192 ratio {derivative_4096 / derivative_8192:.6f})"
    )


if __name__ == "__main__":
    main()
