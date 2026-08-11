"""Provide the rejected opposite-parity Maclaurin control.

Extended Summary
----------------
This module deliberately reproduces the rejected point-sampled transform
path.  It first transforms point samples at the core-grid nodes and then
uses one cubic-Hermite interpolant for arbitrary queries.  Production code
must not use this post-transform interpolation scheme.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from beartype import beartype
from beartype.typing import Tuple
from jaxtyping import Array, Bool, Float64, Int64, jaxtyped

jax.config.update("jax_enable_x64", True)

NAME: str = "opposite-parity Maclaurin control"
DESCRIPTION: str = (
    "Rejected point-sampled Maclaurin rule with cubic-Hermite "
    "post-interpolation."
)


@jaxtyped(typechecker=beartype)
def _node_transform(
    core_grid_ev: Float64[Array, " n_kk"],
    core_imag_values: Float64[Array, " n_kk"],
) -> Float64[Array, " n_kk"]:
    """PRIVATE: Apply the opposite-parity Maclaurin rule at every grid node.

    Parameters
    ----------
    core_grid_ev : Float64[Array, " n_kk"]
        Uniform core energy grid in eV.
    core_imag_values : Float64[Array, " n_kk"]
        Imaginary self-energy samples in eV at the grid nodes.

    Returns
    -------
    result_ev : Float64[Array, " n_kk"]
        Point-sampled principal-value transform in eV at every node.

    Implementation Logic
    --------------------
    The rule couples only node pairs with opposite index parity. Each
    such pair receives the weight ``(2 h / pi) / (omega_j - omega_i)``
    with grid spacing ``h``; same-parity pairs receive zero. One dense
    matrix-vector product then yields all node values at once.
    """
    spacing_ev: Float64[Array, ""] = core_grid_ev[1] - core_grid_ev[0]
    row_indices: Int64[Array, "n_kk 1"] = jnp.arange(core_grid_ev.size)[
        :, None
    ]
    column_indices: Int64[Array, "1 n_kk"] = jnp.arange(core_grid_ev.size)[
        None, :
    ]
    opposite_parity: Bool[Array, "n_kk n_kk"] = (
        row_indices + column_indices
    ) % 2 == 1
    differences_ev: Float64[Array, "n_kk n_kk"] = (
        core_grid_ev[None, :] - core_grid_ev[:, None]
    )
    safe_differences_ev: Float64[Array, "n_kk n_kk"] = jnp.where(
        opposite_parity,
        differences_ev,
        1.0,
    )
    weights: Float64[Array, "n_kk n_kk"] = jnp.where(
        opposite_parity,
        (2.0 * spacing_ev / jnp.pi) / safe_differences_ev,
        0.0,
    )
    result_ev: Float64[Array, " n_kk"] = weights @ core_imag_values
    return result_ev


@jaxtyped(typechecker=beartype)
def _node_slopes(
    core_grid_ev: Float64[Array, " n_kk"],
    node_values: Float64[Array, " n_kk"],
) -> Float64[Array, " n_kk"]:
    """PRIVATE: Return second-order centered and one-sided node derivatives.

    Parameters
    ----------
    core_grid_ev : Float64[Array, " n_kk"]
        Uniform core energy grid in eV.
    node_values : Float64[Array, " n_kk"]
        Transformed values in eV at the grid nodes.

    Returns
    -------
    result : Float64[Array, " n_kk"]
        First-derivative estimate at every node, in eV per eV.

    Implementation Logic
    --------------------
    Interior nodes use the centered difference over ``2 h``. The first
    and last nodes use the matching second-order three-point one-sided
    stencils, so every slope keeps the same truncation order.
    """
    spacing_ev: Float64[Array, ""] = core_grid_ev[1] - core_grid_ev[0]
    interior: Float64[Array, " n_interior"] = (
        node_values[2:] - node_values[:-2]
    ) / (2.0 * spacing_ev)
    left: Float64[Array, ""] = (
        -3.0 * node_values[0] + 4.0 * node_values[1] - node_values[2]
    ) / (2.0 * spacing_ev)
    right: Float64[Array, ""] = (
        3.0 * node_values[-1] - 4.0 * node_values[-2] + node_values[-3]
    ) / (2.0 * spacing_ev)
    result: Float64[Array, " n_kk"] = jnp.concatenate(
        (left[None], interior, right[None])
    )
    return result


@jaxtyped(typechecker=beartype)
def _cubic_hermite(
    core_grid_ev: Float64[Array, " n_kk"],
    node_values: Float64[Array, " n_kk"],
    queries_ev: Float64[Array, " n_query"],
) -> Float64[Array, " n_query"]:
    """PRIVATE: Evaluate the rejected post-transform cubic-Hermite interpolant.

    Parameters
    ----------
    core_grid_ev : Float64[Array, " n_kk"]
        Uniform core energy grid in eV.
    node_values : Float64[Array, " n_kk"]
        Point-sampled transform values in eV at the grid nodes.
    queries_ev : Float64[Array, " n_query"]
        Query energies in eV inside the core grid.

    Returns
    -------
    result_ev : Float64[Array, " n_query"]
        Interpolated transform values in eV at the queries.

    Implementation Logic
    --------------------
    A clipped ``searchsorted`` locates the cell of every query. The
    normalized cell coordinate feeds the four Hermite basis cubics.
    Node values and spacing-scaled ``_node_slopes`` derivatives then
    combine into the C1 piecewise-cubic value.
    """
    spacing_ev: Float64[Array, ""] = core_grid_ev[1] - core_grid_ev[0]
    slopes: Float64[Array, " n_kk"] = _node_slopes(core_grid_ev, node_values)
    left_indices: Int64[Array, " n_query"] = (
        jnp.searchsorted(
            core_grid_ev,
            queries_ev,
            side="right",
        )
        - 1
    )
    left_indices = jnp.clip(left_indices, 0, core_grid_ev.size - 2)
    coordinate: Float64[Array, " n_query"] = (
        queries_ev - core_grid_ev[left_indices]
    ) / spacing_ev
    coordinate_squared: Float64[Array, " n_query"] = coordinate * coordinate
    coordinate_cubed: Float64[Array, " n_query"] = (
        coordinate_squared * coordinate
    )
    basis_00: Float64[Array, " n_query"] = (
        2.0 * coordinate_cubed - 3.0 * coordinate_squared + 1.0
    )
    basis_10: Float64[Array, " n_query"] = (
        coordinate_cubed - 2.0 * coordinate_squared + coordinate
    )
    basis_01: Float64[Array, " n_query"] = (
        -2.0 * coordinate_cubed + 3.0 * coordinate_squared
    )
    basis_11: Float64[Array, " n_query"] = (
        coordinate_cubed - coordinate_squared
    )
    result_ev: Float64[Array, " n_query"] = (
        basis_00 * node_values[left_indices]
        + basis_10 * spacing_ev * slopes[left_indices]
        + basis_01 * node_values[left_indices + 1]
        + basis_11 * spacing_ev * slopes[left_indices + 1]
    )
    return result_ev


@jaxtyped(typechecker=beartype)
def core_pv_transform(
    core_grid_ev: Float64[Array, " n_kk"],
    core_imag_values: Float64[Array, " n_kk"],
    queries_ev: Float64[Array, " n_query"],
) -> Float64[Array, " n_query"]:
    """Return the unsubtracted core principal-value contribution.

    Parameters
    ----------
    core_grid_ev : Float64[Array, " n_kk"]
        Uniform core energy grid in eV.
    core_imag_values : Float64[Array, " n_kk"]
        Imaginary self-energy values in eV at the grid nodes.
    queries_ev : Float64[Array, " n_query"]
        Query energies in eV inside the core grid.

    Returns
    -------
    result_ev : Float64[Array, " n_query"]
        Interpolated principal-value contribution in eV.

    Implementation Logic
    --------------------
    Apply the opposite-parity Maclaurin matrix on the uniform core grid.
    Evaluate arbitrary queries with one cubic-Hermite interpolant.
    """
    node_values: Float64[Array, " n_kk"] = _node_transform(
        core_grid_ev, core_imag_values
    )
    result_ev: Float64[Array, " n_query"] = _cubic_hermite(
        core_grid_ev, node_values, queries_ev
    )
    return result_ev


def _wigner_calibration(node_count: int) -> Tuple[float, float]:
    """PRIVATE: Measure maximum Wigner value and query-derivative errors.

    Parameters
    ----------
    node_count : int
        Number of uniform core-grid nodes on ``[-8, 8]`` eV.

    Returns
    -------
    errors : Tuple[float, float]
        Maximum absolute value error in eV and maximum absolute
        query-derivative error over the in-band queries.

    Implementation Logic
    --------------------
    The fixture is the Wigner semicircle with half-width 1.5 eV and
    coupling 0.2 eV^2, whose in-band transform is ``prefactor * omega``.
    Evaluate the control on 1001 queries in ``[-1, 1]`` eV. Differentiate
    one scalar wrapper with ``jax.vmap(jax.grad)``. Return both maximum
    deviations from the analytic line.
    """
    core_grid_ev: Float64[Array, " n_kk"] = jnp.linspace(
        -8.0, 8.0, node_count, dtype=jnp.float64
    )
    band_ev: float = 1.5
    coupling_ev2: float = 0.2
    prefactor: float = 2.0 * coupling_ev2 / band_ev**2
    radicand_ev2: Float64[Array, " n_kk"] = jnp.maximum(
        band_ev**2 - core_grid_ev**2, 0.0
    )
    core_imag_values: Float64[Array, " n_kk"] = -prefactor * jnp.sqrt(
        radicand_ev2
    )
    queries_ev: Float64[Array, " 1001"] = jnp.linspace(
        -1.0, 1.0, 1001, dtype=jnp.float64
    )

    def scalar_transform(query_ev: Float64[Array, ""]) -> Float64[Array, ""]:
        """Evaluate the control at one scalar query.

        Parameters
        ----------
        query_ev : Float64[Array, ""]
            Scalar query energy in eV.

        Returns
        -------
        value : Float64[Array, ""]
            Interpolated principal-value contribution in eV.

        Notes
        -----
        The wrapper supplies the singleton query axis that the operator
        expects. JAX differentiates this scalar result.
        """
        value: Float64[Array, ""] = core_pv_transform(
            core_grid_ev,
            core_imag_values,
            query_ev[None],
        )[0]
        return value

    values_ev: Float64[Array, " 1001"] = core_pv_transform(
        core_grid_ev,
        core_imag_values,
        queries_ev,
    )
    derivatives: Float64[Array, " 1001"] = jax.vmap(
        jax.grad(scalar_transform)
    )(queries_ev)
    value_error_ev: Float64[Array, ""] = jnp.max(
        jnp.abs(values_ev - prefactor * queries_ev)
    )
    derivative_error: Float64[Array, ""] = jnp.max(
        jnp.abs(derivatives - prefactor)
    )
    errors: Tuple[float, float] = (
        float(value_error_ev),
        float(derivative_error),
    )
    return errors


def main() -> None:
    """Report calibration measurements for the rejected control.

    Notes
    -----
    The function compares the measured errors at 4096 nodes with published
    values. It then prints the refinement ratio from 4096 to 8192 nodes.
    """
    quoted_value_error: float = 1.106e-5
    quoted_derivative_error: float = 6.376e-3
    value_4096: float
    derivative_4096: float
    value_4096, derivative_4096 = _wigner_calibration(4096)
    value_8192: float
    derivative_8192: float
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
