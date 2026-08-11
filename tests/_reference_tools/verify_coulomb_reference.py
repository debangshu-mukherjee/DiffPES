"""Verify one frozen Coulomb order in an isolated process.

Measure value, derivative, recurrence, symmetry, and branch evidence against
the frozen mpmath authority. Emit one JSON report for the parent verifier.
"""

import json
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from beartype.typing import Any, Dict, List, Tuple, Union
from jaxtyping import Array, Bool, Float64, Int64
from numpy.typing import NDArray

from diffpes.radial import coulomb_fg

ETA_MIN: float = -3.0
ETA_MAX: float = 3.0
RHO_MIN: float = 1.0e-4
RHO_MAX: float = 40.0
ETA_FD_EXPONENTS: Tuple[int, ...] = (12, 14, 16)
ETA_FIVE_POINT_FD_EXPONENTS: Tuple[int, ...] = (8, 10, 12)
ETA_FIVE_POINT_FD_MULTIPLIER: float = 2.0**0.5
RHO_FD_EXPONENTS: Tuple[int, ...] = (8, 10, 12)
REGULAR_RHO_FD_SCALE_FLOOR: float = 2.0**-5
IRREGULAR_RHO_FD_SCALE_FLOOR: float = 2.0**-9
RHO_FD_SCALE_RULE: str = (
    "fixed physics rows [regular,irregular,regular,irregular] with "
    "max(abs(rho), 2**-5 or 2**-9)*2**-exponent"
)


def _mixed_budget_ratio(
    actual: Float64[Array, "..."],
    reference: Float64[NDArray, "..."],
) -> float:
    """PRIVATE: Return the maximum mixed-tolerance consumption.

    Parameters
    ----------
    actual : Float64[Array, "..."]
        Production values under test.
    reference : Float64[NDArray, "..."]
        Frozen 80-digit mpmath truth of the same shape.

    Returns
    -------
    ratio : float
        Worst ``|actual - reference| / (1e-10 + 1e-7 |reference|)`` over all
        entries. A ratio of one exactly consumes the budget.

    Notes
    -----
    The mixed absolute-plus-relative denominator matches the
    registered Coulomb-assembly rule, so the report stays comparable
    across rows of very different magnitude.
    """
    ratio: Float64[Array, "..."] = jnp.abs(actual - jnp.asarray(reference)) / (
        1.0e-10 + 1.0e-7 * jnp.abs(jnp.asarray(reference))
    )
    maximum_ratio: float = float(jnp.max(ratio))
    return maximum_ratio


def main() -> None:  # noqa: PLR0915
    """Check sparse values, both AD modes, and registered FD ladders.

    Raises
    ------
    AssertionError
        If a global five-point Richardson result exceeds the mixed budget.

    Notes
    -----
    The isolated command reads one angular-momentum order from ``sys.argv``.
    It prints the complete derivative and invariant record.
    """
    order: int = int(sys.argv[1])
    path: Path = (
        Path(__file__).parents[2]
        / "tests"
        / "test_diffpes"
        / "test_radial"
        / "data"
        / "coulomb_mpmath_80digit.npz"
    )
    archive: Any
    with np.load(path) as archive:
        reference: Dict[
            str,
            Union[
                Float64[NDArray, "..."],
                Int64[NDArray, "..."],
            ],
        ] = {name: archive[name] for name in archive.files}
    eta_grid_numpy: Float64[NDArray, "n_eta n_rho"]
    rho_grid_numpy: Float64[NDArray, "n_eta n_rho"]
    eta_grid_numpy, rho_grid_numpy = np.meshgrid(
        reference["etas"],
        reference["rhos"],
        indexing="ij",
    )
    eta_grid: Float64[Array, "n_eta n_rho"] = jnp.asarray(eta_grid_numpy)
    rho_grid: Float64[Array, "n_eta n_rho"] = jnp.asarray(rho_grid_numpy)

    @jax.jit
    def values(
        eta: Float64[Array, "n_eta n_rho"],
        rho: Float64[Array, "n_eta n_rho"],
    ) -> Float64[Array, "4 n_eta n_rho"]:
        """Compute all Coulomb rows on the sparse product.

        Parameters
        ----------
        eta : Float64[Array, "n_eta n_rho"]
            Sommerfeld-parameter product grid.
        rho : Float64[Array, "n_eta n_rho"]
            Radial-coordinate product grid.

        Returns
        -------
        rows : Float64[Array, "4 n_eta n_rho"]
            Regular, irregular, and radial-derivative rows.

        Notes
        -----
        The closure fixes one angular-momentum order.
        """
        rows: Float64[Array, "4 n_eta n_rho"] = jnp.stack(
            coulomb_fg(order, eta, rho)
        )
        return rows

    actual: Float64[Array, "4 n_eta n_rho"] = values(eta_grid, rho_grid)
    jax.block_until_ready(actual)
    names: Tuple[str, ...] = ("f", "g", "df_drho", "dg_drho")
    name: str
    row: Float64[Array, "n_eta n_rho"]
    for name, row in zip(names, actual, strict=True):
        np.testing.assert_allclose(
            row,
            reference[name][order],
            rtol=1.0e-10,
            atol=1.0e-12,
        )

    @jax.jit
    def eta_forward(
        eta: Float64[Array, "n_eta n_rho"],
        rho: Float64[Array, "n_eta n_rho"],
    ) -> Float64[Array, "4 n_eta n_rho"]:
        """Compute the unit-eta forward-mode tangent.

        Parameters
        ----------
        eta : Float64[Array, "n_eta n_rho"]
            Sommerfeld-parameter product grid.
        rho : Float64[Array, "n_eta n_rho"]
            Radial-coordinate product grid.

        Returns
        -------
        tangent : Float64[Array, "4 n_eta n_rho"]
            Unit-eta directional derivative of all Coulomb rows.

        Notes
        -----
        ``jax.jvp`` differentiates the complete sparse product.
        """
        tangent: Float64[Array, "4 n_eta n_rho"] = jax.jvp(
            lambda argument: values(argument, rho),
            (eta,),
            (jnp.ones_like(eta),),
        )[1]
        return tangent

    @jax.jit
    def rho_forward(
        eta: Float64[Array, "n_eta n_rho"],
        rho: Float64[Array, "n_eta n_rho"],
    ) -> Float64[Array, "4 n_eta n_rho"]:
        """Compute the unit-rho forward-mode tangent.

        Parameters
        ----------
        eta : Float64[Array, "n_eta n_rho"]
            Sommerfeld-parameter product grid.
        rho : Float64[Array, "n_eta n_rho"]
            Radial-coordinate product grid.

        Returns
        -------
        tangent : Float64[Array, "4 n_eta n_rho"]
            Unit-rho directional derivative of all Coulomb rows.

        Notes
        -----
        ``jax.jvp`` differentiates the complete sparse product.
        """
        tangent: Float64[Array, "4 n_eta n_rho"] = jax.jvp(
            lambda argument: values(eta, argument),
            (rho,),
            (jnp.ones_like(rho),),
        )[1]
        return tangent

    eta_tangent: Float64[Array, "4 n_eta n_rho"] = eta_forward(
        eta_grid, rho_grid
    )
    rho_tangent: Float64[Array, "4 n_eta n_rho"] = rho_forward(
        eta_grid, rho_grid
    )
    jax.block_until_ready((eta_tangent, rho_tangent))
    eta_names: Tuple[str, ...] = (
        "df_deta",
        "dg_deta",
        "d_df_drho_deta",
        "d_dg_drho_deta",
    )
    rho_names: Tuple[str, ...] = (
        "df_drho",
        "dg_drho",
        "d2f_drho2",
        "d2g_drho2",
    )
    for name, row in zip(eta_names, eta_tangent, strict=True):
        np.testing.assert_allclose(
            row,
            reference[name][order],
            rtol=1.0e-7,
            atol=1.0e-10,
        )
    for name, row in zip(rho_names, rho_tangent, strict=True):
        np.testing.assert_allclose(
            row,
            reference[name][order],
            rtol=1.0e-7,
            atol=1.0e-10,
        )

    weights: Float64[Array, "4 n_eta n_rho"] = jnp.sin(
        jnp.arange(actual.size, dtype=jnp.float64).reshape(actual.shape) + 0.37
    )

    @jax.jit
    def eta_objective(
        eta: Float64[Array, "n_eta n_rho"],
        rho: Float64[Array, "n_eta n_rho"],
    ) -> Float64[Array, ""]:
        """Compute one weighted scalar for reverse eta AD.

        Parameters
        ----------
        eta : Float64[Array, "n_eta n_rho"]
            Sommerfeld-parameter product grid.
        rho : Float64[Array, "n_eta n_rho"]
            Radial-coordinate product grid.

        Returns
        -------
        objective : Float64[Array, ""]
            Generic weighted sum of all Coulomb rows.

        Notes
        -----
        Fixed sinusoidal weights prevent accidental cancellation.
        """
        objective: Float64[Array, ""] = jnp.sum(values(eta, rho) * weights)
        return objective

    @jax.jit
    def rho_objective(
        eta: Float64[Array, "n_eta n_rho"],
        rho: Float64[Array, "n_eta n_rho"],
    ) -> Float64[Array, ""]:
        """Compute one weighted scalar for reverse rho AD.

        Parameters
        ----------
        eta : Float64[Array, "n_eta n_rho"]
            Sommerfeld-parameter product grid.
        rho : Float64[Array, "n_eta n_rho"]
            Radial-coordinate product grid.

        Returns
        -------
        objective : Float64[Array, ""]
            Generic weighted sum of all Coulomb rows.

        Notes
        -----
        Fixed sinusoidal weights prevent accidental cancellation.
        """
        objective: Float64[Array, ""] = jnp.sum(values(eta, rho) * weights)
        return objective

    eta_reverse: Float64[Array, "n_eta n_rho"] = jax.grad(
        eta_objective, argnums=0
    )(
        eta_grid,
        rho_grid,
    )
    rho_reverse: Float64[Array, "n_eta n_rho"] = jax.grad(
        rho_objective, argnums=1
    )(
        eta_grid,
        rho_grid,
    )
    jax.block_until_ready((eta_reverse, rho_reverse))
    eta_reference: Float64[Array, "4 n_eta n_rho"] = jnp.stack(
        tuple(jnp.asarray(reference[name][order]) for name in eta_names)
    )
    rho_reference: Float64[Array, "4 n_eta n_rho"] = jnp.stack(
        tuple(jnp.asarray(reference[name][order]) for name in rho_names)
    )
    expected_eta_reverse: Float64[Array, "n_eta n_rho"] = jnp.sum(
        weights * eta_reference,
        axis=0,
    )
    expected_rho_reverse: Float64[Array, "n_eta n_rho"] = jnp.sum(
        weights * rho_reference,
        axis=0,
    )
    np.testing.assert_allclose(
        eta_reverse,
        expected_eta_reverse,
        rtol=1.0e-7,
        atol=1.0e-10,
    )
    np.testing.assert_allclose(
        rho_reverse,
        expected_rho_reverse,
        rtol=1.0e-7,
        atol=1.0e-10,
    )

    eta_fd_ratios: List[float] = []
    eta_fd_rows: List[Float64[Array, "4 n_eta n_rho"]] = []
    eta_scale: Float64[Array, "n_eta n_rho"] = jnp.maximum(
        1.0, jnp.abs(eta_grid)
    )
    eta_exponent: int
    for eta_exponent in ETA_FD_EXPONENTS:
        eta_step: Float64[Array, "n_eta n_rho"] = (
            eta_scale * 2.0**-eta_exponent
        )
        plus_one: Float64[Array, "4 n_eta n_rho"] = values(
            jnp.minimum(eta_grid + eta_step, ETA_MAX),
            rho_grid,
        )
        plus_two: Float64[Array, "4 n_eta n_rho"] = values(
            jnp.minimum(eta_grid + 2.0 * eta_step, ETA_MAX),
            rho_grid,
        )
        minus_one: Float64[Array, "4 n_eta n_rho"] = values(
            jnp.maximum(eta_grid - eta_step, ETA_MIN),
            rho_grid,
        )
        minus_two: Float64[Array, "4 n_eta n_rho"] = values(
            jnp.maximum(eta_grid - 2.0 * eta_step, ETA_MIN),
            rho_grid,
        )
        eta_central: Float64[Array, "4 n_eta n_rho"] = (
            plus_one - minus_one
        ) / (2.0 * eta_step)
        eta_forward_edge: Float64[Array, "4 n_eta n_rho"] = (
            -3.0 * actual + 4.0 * plus_one - plus_two
        ) / (2.0 * eta_step)
        eta_backward_edge: Float64[Array, "4 n_eta n_rho"] = (
            3.0 * actual - 4.0 * minus_one + minus_two
        ) / (2.0 * eta_step)
        eta_fd: Float64[Array, "4 n_eta n_rho"] = jnp.where(
            (eta_grid == ETA_MIN)[None, ...],
            eta_forward_edge,
            jnp.where(
                (eta_grid == ETA_MAX)[None, ...],
                eta_backward_edge,
                eta_central,
            ),
        )
        jax.block_until_ready(eta_fd)
        eta_fd_rows.append(eta_fd)
        eta_fd_ratios.append(
            _mixed_budget_ratio(eta_fd, np.asarray(eta_reference))
        )

    eta_five_point_ratios: List[float] = []
    eta_five_point_rows: List[Float64[Array, "4 n_eta n_rho"]] = []
    for eta_exponent in ETA_FIVE_POINT_FD_EXPONENTS:
        eta_step = (
            ETA_FIVE_POINT_FD_MULTIPLIER * eta_scale * 2.0**-eta_exponent
        )
        plus_one = values(
            jnp.minimum(eta_grid + eta_step, ETA_MAX),
            rho_grid,
        )
        plus_two = values(
            jnp.minimum(eta_grid + 2.0 * eta_step, ETA_MAX),
            rho_grid,
        )
        plus_three: Float64[Array, "4 n_eta n_rho"] = values(
            jnp.minimum(eta_grid + 3.0 * eta_step, ETA_MAX),
            rho_grid,
        )
        plus_four: Float64[Array, "4 n_eta n_rho"] = values(
            jnp.minimum(eta_grid + 4.0 * eta_step, ETA_MAX),
            rho_grid,
        )
        minus_one = values(
            jnp.maximum(eta_grid - eta_step, ETA_MIN),
            rho_grid,
        )
        minus_two = values(
            jnp.maximum(eta_grid - 2.0 * eta_step, ETA_MIN),
            rho_grid,
        )
        minus_three: Float64[Array, "4 n_eta n_rho"] = values(
            jnp.maximum(eta_grid - 3.0 * eta_step, ETA_MIN),
            rho_grid,
        )
        minus_four: Float64[Array, "4 n_eta n_rho"] = values(
            jnp.maximum(eta_grid - 4.0 * eta_step, ETA_MIN),
            rho_grid,
        )
        eta_five_point_central: Float64[Array, "4 n_eta n_rho"] = (
            minus_two - 8.0 * minus_one + 8.0 * plus_one - plus_two
        ) / (12.0 * eta_step)
        eta_five_point_forward: Float64[Array, "4 n_eta n_rho"] = (
            -25.0 * actual
            + 48.0 * plus_one
            - 36.0 * plus_two
            + 16.0 * plus_three
            - 3.0 * plus_four
        ) / (12.0 * eta_step)
        eta_five_point_backward: Float64[Array, "4 n_eta n_rho"] = (
            25.0 * actual
            - 48.0 * minus_one
            + 36.0 * minus_two
            - 16.0 * minus_three
            + 3.0 * minus_four
        ) / (12.0 * eta_step)
        use_forward_stencil: Bool[Array, "n_eta n_rho"] = (
            eta_grid - 2.0 * eta_step < ETA_MIN
        )
        use_backward_stencil: Bool[Array, "n_eta n_rho"] = (
            eta_grid + 2.0 * eta_step > ETA_MAX
        )
        eta_five_point: Float64[Array, "4 n_eta n_rho"] = jnp.where(
            use_forward_stencil[None, ...],
            eta_five_point_forward,
            jnp.where(
                use_backward_stencil[None, ...],
                eta_five_point_backward,
                eta_five_point_central,
            ),
        )
        jax.block_until_ready(eta_five_point)
        eta_five_point_rows.append(eta_five_point)
        eta_five_point_ratios.append(
            _mixed_budget_ratio(
                eta_five_point,
                np.asarray(eta_reference),
            )
        )

    rho_fd_ratios: List[float] = []
    rho_fd_rows: List[Float64[Array, "4 n_eta n_rho"]] = []
    rho_five_point_ratios: List[float] = []
    rho_five_point_rows: List[Float64[Array, "4 n_eta n_rho"]] = []
    regular_regime_five_point_ratios: List[float] = []
    irregular_regime_five_point_ratios: List[float] = []
    regular_row_mask: Bool[Array, "4 1 1"] = jnp.asarray(
        (True, False, True, False),
    )[:, None, None]

    def rho_fd_rows_for_step(
        rho_step: Float64[Array, "n_eta n_rho"],
    ) -> Tuple[
        Float64[Array, "4 n_eta n_rho"],
        Float64[Array, "4 n_eta n_rho"],
    ]:
        """Compute second- and fourth-order rows for one step field.

        Parameters
        ----------
        rho_step : Float64[Array, "n_eta n_rho"]
            Positive radial finite-difference step at each product point.

        Returns
        -------
        result : Tuple[Float64[Array, "4 n_eta n_rho"],
            Float64[Array, "4 n_eta n_rho"]]
            Second-order and five-point radial derivative rows.

        Notes
        -----
        One-sided fourth-order stencils protect both radial boundaries.
        """
        plus_one: Float64[Array, "4 n_eta n_rho"] = values(
            eta_grid,
            jnp.minimum(rho_grid + rho_step, RHO_MAX),
        )
        plus_two: Float64[Array, "4 n_eta n_rho"] = values(
            eta_grid,
            jnp.minimum(rho_grid + 2.0 * rho_step, RHO_MAX),
        )
        minus_one: Float64[Array, "4 n_eta n_rho"] = values(
            eta_grid,
            jnp.maximum(rho_grid - rho_step, RHO_MIN),
        )
        minus_two: Float64[Array, "4 n_eta n_rho"] = values(
            eta_grid,
            jnp.maximum(rho_grid - 2.0 * rho_step, RHO_MIN),
        )
        plus_three: Float64[Array, "4 n_eta n_rho"] = values(
            eta_grid,
            jnp.minimum(rho_grid + 3.0 * rho_step, RHO_MAX),
        )
        plus_four: Float64[Array, "4 n_eta n_rho"] = values(
            eta_grid,
            jnp.minimum(rho_grid + 4.0 * rho_step, RHO_MAX),
        )
        minus_three: Float64[Array, "4 n_eta n_rho"] = values(
            eta_grid,
            jnp.maximum(rho_grid - 3.0 * rho_step, RHO_MIN),
        )
        minus_four: Float64[Array, "4 n_eta n_rho"] = values(
            eta_grid,
            jnp.maximum(rho_grid - 4.0 * rho_step, RHO_MIN),
        )
        use_forward_stencil: Bool[Array, "n_eta n_rho"] = (
            rho_grid - 2.0 * rho_step < RHO_MIN
        )
        use_backward_stencil: Bool[Array, "n_eta n_rho"] = (
            rho_grid + 2.0 * rho_step > RHO_MAX
        )
        rho_central: Float64[Array, "4 n_eta n_rho"] = (
            plus_one - minus_one
        ) / (2.0 * rho_step)
        rho_forward_edge: Float64[Array, "4 n_eta n_rho"] = (
            -3.0 * actual + 4.0 * plus_one - plus_two
        ) / (2.0 * rho_step)
        rho_backward_edge: Float64[Array, "4 n_eta n_rho"] = (
            3.0 * actual - 4.0 * minus_one + minus_two
        ) / (2.0 * rho_step)
        rho_second_order: Float64[Array, "4 n_eta n_rho"] = jnp.where(
            use_forward_stencil[None, ...],
            rho_forward_edge,
            jnp.where(
                use_backward_stencil[None, ...],
                rho_backward_edge,
                rho_central,
            ),
        )
        rho_five_point_central: Float64[Array, "4 n_eta n_rho"] = (
            minus_two - 8.0 * minus_one + 8.0 * plus_one - plus_two
        ) / (12.0 * rho_step)
        rho_five_point_forward: Float64[Array, "4 n_eta n_rho"] = (
            -25.0 * actual
            + 48.0 * plus_one
            - 36.0 * plus_two
            + 16.0 * plus_three
            - 3.0 * plus_four
        ) / (12.0 * rho_step)
        rho_five_point_backward: Float64[Array, "4 n_eta n_rho"] = (
            25.0 * actual
            - 48.0 * minus_one
            + 36.0 * minus_two
            - 16.0 * minus_three
            + 3.0 * minus_four
        ) / (12.0 * rho_step)
        rho_five_point: Float64[Array, "4 n_eta n_rho"] = jnp.where(
            use_forward_stencil[None, ...],
            rho_five_point_forward,
            jnp.where(
                use_backward_stencil[None, ...],
                rho_five_point_backward,
                rho_five_point_central,
            ),
        )
        result: Tuple[
            Float64[Array, "4 n_eta n_rho"],
            Float64[Array, "4 n_eta n_rho"],
        ] = (rho_second_order, rho_five_point)
        return result

    rho_exponent: int
    for rho_exponent in RHO_FD_EXPONENTS:
        regular_rho_step: Float64[Array, "n_eta n_rho"] = (
            jnp.maximum(REGULAR_RHO_FD_SCALE_FLOOR, jnp.abs(rho_grid))
            * 2.0**-rho_exponent
        )
        irregular_rho_step: Float64[Array, "n_eta n_rho"] = (
            jnp.maximum(IRREGULAR_RHO_FD_SCALE_FLOOR, jnp.abs(rho_grid))
            * 2.0**-rho_exponent
        )
        regular_rho_fd: Float64[Array, "4 n_eta n_rho"]
        regular_rho_five_point: Float64[Array, "4 n_eta n_rho"]
        regular_rho_fd, regular_rho_five_point = rho_fd_rows_for_step(
            regular_rho_step
        )
        irregular_rho_fd: Float64[Array, "4 n_eta n_rho"]
        irregular_rho_five_point: Float64[Array, "4 n_eta n_rho"]
        irregular_rho_fd, irregular_rho_five_point = rho_fd_rows_for_step(
            irregular_rho_step
        )
        rho_fd: Float64[Array, "4 n_eta n_rho"] = jnp.where(
            regular_row_mask,
            regular_rho_fd,
            irregular_rho_fd,
        )
        rho_five_point: Float64[Array, "4 n_eta n_rho"] = jnp.where(
            regular_row_mask,
            regular_rho_five_point,
            irregular_rho_five_point,
        )
        jax.block_until_ready(rho_fd)
        rho_fd_rows.append(rho_fd)
        rho_fd_ratios.append(
            _mixed_budget_ratio(rho_fd, np.asarray(rho_reference))
        )
        jax.block_until_ready(rho_five_point)
        rho_five_point_rows.append(rho_five_point)
        rho_five_point_ratios.append(
            _mixed_budget_ratio(
                rho_five_point,
                np.asarray(rho_reference),
            )
        )
        regular_regime_five_point_ratios.append(
            _mixed_budget_ratio(
                regular_rho_five_point,
                np.asarray(rho_reference),
            )
        )
        irregular_regime_five_point_ratios.append(
            _mixed_budget_ratio(
                irregular_rho_five_point,
                np.asarray(rho_reference),
            )
        )
    eta_fd_stack: Float64[Array, "3 4 n_eta n_rho"] = jnp.stack(eta_fd_rows)
    eta_five_point_stack: Float64[Array, "3 4 n_eta n_rho"] = jnp.stack(
        eta_five_point_rows
    )
    rho_fd_stack: Float64[Array, "3 4 n_eta n_rho"] = jnp.stack(rho_fd_rows)
    rho_five_point_stack: Float64[Array, "3 4 n_eta n_rho"] = jnp.stack(
        rho_five_point_rows
    )
    eta_richardson: Float64[Array, "4 n_eta n_rho"] = (
        16.0 * eta_fd_stack[1] - eta_fd_stack[0]
    ) / 15.0
    eta_five_point_richardson: Float64[Array, "4 n_eta n_rho"] = (
        256.0 * eta_five_point_stack[1] - eta_five_point_stack[0]
    ) / 255.0
    eta_five_point_fine_richardson: Float64[Array, "4 n_eta n_rho"] = (
        256.0 * eta_five_point_stack[2] - eta_five_point_stack[1]
    ) / 255.0
    eta_five_point_fine_pair_spread_ratio: float = float(
        jnp.max(
            jnp.abs(eta_five_point_stack[2] - eta_five_point_stack[1])
            / (1.0e-10 + 1.0e-7 * jnp.abs(eta_reference))
        )
    )
    rho_richardson_coarse: Float64[Array, "4 n_eta n_rho"] = (
        16.0 * rho_fd_stack[1] - rho_fd_stack[0]
    ) / 15.0
    rho_richardson_fine: Float64[Array, "4 n_eta n_rho"] = (
        16.0 * rho_fd_stack[2] - rho_fd_stack[1]
    ) / 15.0
    rho_richardson: Float64[Array, "4 n_eta n_rho"] = (
        256.0 * rho_richardson_fine - rho_richardson_coarse
    ) / 255.0
    rho_five_point_richardson: Float64[Array, "4 n_eta n_rho"] = (
        256.0 * rho_five_point_stack[2] - rho_five_point_stack[1]
    ) / 255.0
    rho_five_point_fine_pair_spread_ratio: float = float(
        jnp.max(
            jnp.abs(rho_five_point_stack[2] - rho_five_point_stack[1])
            / (1.0e-10 + 1.0e-7 * jnp.abs(rho_reference))
        )
    )
    eta_plateau_candidates: Float64[Array, "4 4 n_eta n_rho"] = (
        jnp.concatenate(
            (eta_fd_stack, eta_richardson[None, ...]),
            axis=0,
        )
    )
    rho_plateau_candidates: Float64[Array, "6 4 n_eta n_rho"] = (
        jnp.concatenate(
            (
                rho_fd_stack,
                rho_richardson_coarse[None, ...],
                rho_richardson_fine[None, ...],
                rho_richardson[None, ...],
            ),
            axis=0,
        )
    )
    eta_best_ratio: float = float(
        jnp.max(
            jnp.min(
                jnp.abs(eta_plateau_candidates - eta_reference[None, ...])
                / (1.0e-10 + 1.0e-7 * jnp.abs(eta_reference[None, ...])),
                axis=0,
            )
        )
    )
    rho_elementwise_best: Float64[Array, "4 n_eta n_rho"] = jnp.min(
        jnp.abs(rho_plateau_candidates - rho_reference[None, ...])
        / (1.0e-10 + 1.0e-7 * jnp.abs(rho_reference[None, ...])),
        axis=0,
    )
    rho_best_ratio: float = float(jnp.max(rho_elementwise_best))
    rho_worst_index: Tuple[int, ...] = tuple(
        int(value)
        for value in jnp.unravel_index(
            jnp.argmax(rho_elementwise_best),
            rho_elementwise_best.shape,
        )
    )
    eta_richardson_ratio: float = _mixed_budget_ratio(
        eta_richardson,
        np.asarray(eta_reference),
    )
    eta_five_point_richardson_ratio: float = _mixed_budget_ratio(
        eta_five_point_richardson,
        np.asarray(eta_reference),
    )
    eta_five_point_fine_richardson_ratio: float = _mixed_budget_ratio(
        eta_five_point_fine_richardson,
        np.asarray(eta_reference),
    )
    eta_five_point_error_ratio: Float64[Array, "4 n_eta n_rho"] = jnp.abs(
        eta_five_point_richardson - eta_reference
    ) / (1.0e-10 + 1.0e-7 * jnp.abs(eta_reference))
    eta_five_point_worst_index: Tuple[int, ...] = tuple(
        int(value)
        for value in jnp.unravel_index(
            jnp.argmax(eta_five_point_error_ratio),
            eta_five_point_error_ratio.shape,
        )
    )
    rho_richardson_ratio: float = _mixed_budget_ratio(
        rho_richardson,
        np.asarray(rho_reference),
    )
    rho_five_point_richardson_ratio: float = _mixed_budget_ratio(
        rho_five_point_richardson,
        np.asarray(rho_reference),
    )
    rho_five_point_error_ratio: Float64[Array, "4 n_eta n_rho"] = jnp.abs(
        rho_five_point_richardson - rho_reference
    ) / (1.0e-10 + 1.0e-7 * jnp.abs(rho_reference))
    rho_five_point_worst_index: Tuple[int, ...] = tuple(
        int(value)
        for value in jnp.unravel_index(
            jnp.argmax(rho_five_point_error_ratio),
            rho_five_point_error_ratio.shape,
        )
    )

    regular: Float64[Array, "n_eta n_rho"] = actual[0]
    irregular: Float64[Array, "n_eta n_rho"] = actual[1]
    regular_derivative: Float64[Array, "n_eta n_rho"] = actual[2]
    irregular_derivative: Float64[Array, "n_eta n_rho"] = actual[3]
    wronskian: Float64[Array, "n_eta n_rho"] = (
        regular_derivative * irregular - regular * irregular_derivative
    )
    np.testing.assert_allclose(
        wronskian,
        jnp.ones_like(wronskian),
        rtol=0.0,
        atol=1.0e-10,
    )
    metrics: Dict[str, Any] = {
        "order": order,
        "eta_forward_budget_ratio": _mixed_budget_ratio(
            eta_tangent,
            np.asarray(eta_reference),
        ),
        "rho_forward_budget_ratio": _mixed_budget_ratio(
            rho_tangent,
            np.asarray(rho_reference),
        ),
        "eta_reverse_budget_ratio": _mixed_budget_ratio(
            eta_reverse,
            np.asarray(expected_eta_reverse),
        ),
        "rho_reverse_budget_ratio": _mixed_budget_ratio(
            rho_reverse,
            np.asarray(expected_rho_reverse),
        ),
        "eta_fd_ladder_budget_ratios": eta_fd_ratios,
        "eta_five_point_ladder_budget_ratios": eta_five_point_ratios,
        "eta_five_point_exponents": list(ETA_FIVE_POINT_FD_EXPONENTS),
        "eta_five_point_scale_rule": ("sqrt(2)*max(1, abs(eta))*2**-exponent"),
        "eta_five_point_fine_pair_spread_budget_ratio": (
            eta_five_point_fine_pair_spread_ratio
        ),
        "eta_five_point_richardson_budget_ratio": (
            eta_five_point_richardson_ratio
        ),
        "eta_five_point_fine_richardson_diagnostic_budget_ratio": (
            eta_five_point_fine_richardson_ratio
        ),
        "eta_five_point_acceptance_rule": (
            "single global coarse-pair h^4 Richardson from exponents 8 and "
            "10 with sqrt(2) multiplier; exponent 12 and its fine pair are "
            "roundoff diagnostics only"
        ),
        "eta_five_point_worst_index": list(eta_five_point_worst_index),
        "eta_five_point_worst_eta": float(
            eta_grid[eta_five_point_worst_index[1:]]
        ),
        "eta_five_point_worst_rho": float(
            rho_grid[eta_five_point_worst_index[1:]]
        ),
        "eta_five_point_worst_row": eta_names[eta_five_point_worst_index[0]],
        "rho_fd_ladder_budget_ratios": rho_fd_ratios,
        "rho_five_point_ladder_budget_ratios": rho_five_point_ratios,
        "rho_five_point_regular_regime_full_row_budget_ratios": (
            regular_regime_five_point_ratios
        ),
        "rho_five_point_irregular_regime_full_row_budget_ratios": (
            irregular_regime_five_point_ratios
        ),
        "rho_fd_exponents": list(RHO_FD_EXPONENTS),
        "rho_fd_scale_rule": RHO_FD_SCALE_RULE,
        "rho_fd_stencil_rule": (
            "five-point central; fourth-order forward/backward when the "
            "central stencil crosses the registered domain"
        ),
        "eta_richardson_budget_ratio": eta_richardson_ratio,
        "rho_richardson_budget_ratio": rho_richardson_ratio,
        "rho_five_point_richardson_budget_ratio": (
            rho_five_point_richardson_ratio
        ),
        "rho_five_point_fine_pair_spread_budget_ratio": (
            rho_five_point_fine_pair_spread_ratio
        ),
        "rho_five_point_worst_index": list(rho_five_point_worst_index),
        "rho_five_point_worst_eta": float(
            eta_grid[rho_five_point_worst_index[1:]]
        ),
        "rho_five_point_worst_rho": float(
            rho_grid[rho_five_point_worst_index[1:]]
        ),
        "rho_five_point_worst_row": rho_names[rho_five_point_worst_index[0]],
        "eta_fd_elementwise_best_budget_ratio": eta_best_ratio,
        "rho_fd_elementwise_best_budget_ratio": rho_best_ratio,
        "rho_fd_worst_index": list(rho_worst_index),
        "rho_fd_worst_eta": float(eta_grid[rho_worst_index[1:]]),
        "rho_fd_worst_rho": float(rho_grid[rho_worst_index[1:]]),
        "rho_fd_worst_row": rho_names[rho_worst_index[0]],
        "wronskian_absolute_error": float(jnp.max(jnp.abs(wronskian - 1.0))),
    }
    print(json.dumps(metrics, sort_keys=True))
    if eta_five_point_richardson_ratio > 1.0:
        message: str = (
            "eta global five-point Richardson plateau failed mixed budget: "
            f"{eta_five_point_richardson_ratio}"
        )
        raise AssertionError(message)
    if rho_five_point_richardson_ratio > 1.0:
        message: str = (
            "rho global five-point Richardson plateau failed mixed budget: "
            f"{rho_five_point_richardson_ratio}"
        )
        raise AssertionError(message)


if __name__ == "__main__":
    main()
