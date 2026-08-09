"""Verify one frozen Coulomb order in an isolated process."""

import json
import sys
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Float64, Shaped
from numpy.typing import NDArray

from diffpes.radial import coulomb_fg

ETA_MIN: float = -3.0
ETA_MAX: float = 3.0
RHO_MIN: float = 1.0e-4
RHO_MAX: float = 40.0
ETA_FD_EXPONENTS: tuple[int, ...] = (12, 14, 16)
ETA_FIVE_POINT_FD_EXPONENTS: tuple[int, ...] = (8, 10, 12)
ETA_FIVE_POINT_FD_MULTIPLIER: float = 2.0**0.5
RHO_FD_EXPONENTS: tuple[int, ...] = (8, 10, 12)
REGULAR_RHO_FD_SCALE_FLOOR: float = 2.0**-5
IRREGULAR_RHO_FD_SCALE_FLOOR: float = 2.0**-9
RHO_FD_SCALE_RULE: str = (
    "fixed physics rows [regular,irregular,regular,irregular] with "
    "max(abs(rho), 2**-5 or 2**-9)*2**-exponent"
)


def _mixed_budget_ratio(
    actual: jax.Array,
    reference: Float64[NDArray, "..."],
) -> float:
    """PRIVATE: Return the maximum D11 mixed-tolerance consumption.

    Parameters
    ----------
    actual : jax.Array
        Production values under test.
    reference : Float64[NDArray, "..."]
        Frozen 80-digit mpmath truth of the same shape.

    Returns
    -------
    ratio : float
        Worst ``|actual - reference| / (1e-10 + 1e-7 |reference|)``
        over all entries; one means the budget is exactly consumed.

    Notes
    -----
    The mixed absolute-plus-relative denominator matches the
    registered D11 acceptance rule, so the report stays comparable
    across rows of very different magnitude.
    """
    ratio: jax.Array = jnp.abs(actual - jnp.asarray(reference)) / (
        1.0e-10 + 1.0e-7 * jnp.abs(jnp.asarray(reference))
    )
    return float(jnp.max(ratio))


def main() -> None:  # noqa: PLR0915
    """Check sparse values, both AD modes, and registered FD ladders."""
    order: int = int(sys.argv[1])
    path: Path = (
        Path(__file__).parents[2]
        / "tests"
        / "test_diffpes"
        / "test_radial"
        / "data"
        / "coulomb_mpmath_80digit.npz"
    )
    with np.load(path) as archive:
        reference: dict[str, Shaped[NDArray, "..."]] = {
            name: archive[name] for name in archive.files
        }
    eta_grid_numpy: Float64[NDArray, "n_eta n_rho"]
    rho_grid_numpy: Float64[NDArray, "n_eta n_rho"]
    eta_grid_numpy, rho_grid_numpy = np.meshgrid(
        reference["etas"],
        reference["rhos"],
        indexing="ij",
    )
    eta_grid: jax.Array = jnp.asarray(eta_grid_numpy)
    rho_grid: jax.Array = jnp.asarray(rho_grid_numpy)

    @jax.jit
    def values(eta: jax.Array, rho: jax.Array) -> jax.Array:
        """Return all production Coulomb rows on the sparse product."""
        return jnp.stack(coulomb_fg(order, eta, rho))

    actual: jax.Array = values(eta_grid, rho_grid)
    jax.block_until_ready(actual)
    names: tuple[str, ...] = ("f", "g", "df_drho", "dg_drho")
    name: str
    row: jax.Array
    for name, row in zip(names, actual, strict=True):
        np.testing.assert_allclose(
            row,
            reference[name][order],
            rtol=1.0e-10,
            atol=1.0e-12,
        )

    @jax.jit
    def eta_forward(eta: jax.Array, rho: jax.Array) -> jax.Array:
        """Return the unit-eta forward-mode tangent."""
        tangent: jax.Array = jax.jvp(
            lambda argument: values(argument, rho),
            (eta,),
            (jnp.ones_like(eta),),
        )[1]
        return tangent

    @jax.jit
    def rho_forward(eta: jax.Array, rho: jax.Array) -> jax.Array:
        """Return the unit-rho forward-mode tangent."""
        tangent: jax.Array = jax.jvp(
            lambda argument: values(eta, argument),
            (rho,),
            (jnp.ones_like(rho),),
        )[1]
        return tangent

    eta_tangent: jax.Array = eta_forward(eta_grid, rho_grid)
    rho_tangent: jax.Array = rho_forward(eta_grid, rho_grid)
    jax.block_until_ready((eta_tangent, rho_tangent))
    eta_names: tuple[str, ...] = (
        "df_deta",
        "dg_deta",
        "d_df_drho_deta",
        "d_dg_drho_deta",
    )
    rho_names: tuple[str, ...] = (
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

    weights: jax.Array = jnp.sin(
        jnp.arange(actual.size, dtype=jnp.float64).reshape(actual.shape) + 0.37
    )

    @jax.jit
    def eta_objective(eta: jax.Array, rho: jax.Array) -> jax.Array:
        """Return one generic weighted scalar for reverse eta AD."""
        return jnp.sum(values(eta, rho) * weights)

    @jax.jit
    def rho_objective(eta: jax.Array, rho: jax.Array) -> jax.Array:
        """Return one generic weighted scalar for reverse rho AD."""
        return jnp.sum(values(eta, rho) * weights)

    eta_reverse: jax.Array = jax.grad(eta_objective, argnums=0)(
        eta_grid,
        rho_grid,
    )
    rho_reverse: jax.Array = jax.grad(rho_objective, argnums=1)(
        eta_grid,
        rho_grid,
    )
    jax.block_until_ready((eta_reverse, rho_reverse))
    eta_reference: jax.Array = jnp.stack(
        tuple(jnp.asarray(reference[name][order]) for name in eta_names)
    )
    rho_reference: jax.Array = jnp.stack(
        tuple(jnp.asarray(reference[name][order]) for name in rho_names)
    )
    expected_eta_reverse: jax.Array = jnp.sum(
        weights * eta_reference,
        axis=0,
    )
    expected_rho_reverse: jax.Array = jnp.sum(
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

    eta_fd_ratios: list[float] = []
    eta_fd_rows: list[jax.Array] = []
    eta_scale: jax.Array = jnp.maximum(1.0, jnp.abs(eta_grid))
    eta_exponent: int
    for eta_exponent in ETA_FD_EXPONENTS:
        eta_step: jax.Array = eta_scale * 2.0**-eta_exponent
        plus_one: jax.Array = values(
            jnp.minimum(eta_grid + eta_step, ETA_MAX),
            rho_grid,
        )
        plus_two: jax.Array = values(
            jnp.minimum(eta_grid + 2.0 * eta_step, ETA_MAX),
            rho_grid,
        )
        minus_one: jax.Array = values(
            jnp.maximum(eta_grid - eta_step, ETA_MIN),
            rho_grid,
        )
        minus_two: jax.Array = values(
            jnp.maximum(eta_grid - 2.0 * eta_step, ETA_MIN),
            rho_grid,
        )
        eta_central: jax.Array = (plus_one - minus_one) / (2.0 * eta_step)
        eta_forward_edge: jax.Array = (
            -3.0 * actual + 4.0 * plus_one - plus_two
        ) / (2.0 * eta_step)
        eta_backward_edge: jax.Array = (
            3.0 * actual - 4.0 * minus_one + minus_two
        ) / (2.0 * eta_step)
        eta_fd: jax.Array = jnp.where(
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

    eta_five_point_ratios: list[float] = []
    eta_five_point_rows: list[jax.Array] = []
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
        plus_three = values(
            jnp.minimum(eta_grid + 3.0 * eta_step, ETA_MAX),
            rho_grid,
        )
        plus_four = values(
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
        minus_three = values(
            jnp.maximum(eta_grid - 3.0 * eta_step, ETA_MIN),
            rho_grid,
        )
        minus_four = values(
            jnp.maximum(eta_grid - 4.0 * eta_step, ETA_MIN),
            rho_grid,
        )
        eta_five_point_central: jax.Array = (
            minus_two - 8.0 * minus_one + 8.0 * plus_one - plus_two
        ) / (12.0 * eta_step)
        eta_five_point_forward: jax.Array = (
            -25.0 * actual
            + 48.0 * plus_one
            - 36.0 * plus_two
            + 16.0 * plus_three
            - 3.0 * plus_four
        ) / (12.0 * eta_step)
        eta_five_point_backward: jax.Array = (
            25.0 * actual
            - 48.0 * minus_one
            + 36.0 * minus_two
            - 16.0 * minus_three
            + 3.0 * minus_four
        ) / (12.0 * eta_step)
        use_forward_stencil = eta_grid - 2.0 * eta_step < ETA_MIN
        use_backward_stencil = eta_grid + 2.0 * eta_step > ETA_MAX
        eta_five_point: jax.Array = jnp.where(
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

    rho_fd_ratios: list[float] = []
    rho_fd_rows: list[jax.Array] = []
    rho_five_point_ratios: list[float] = []
    rho_five_point_rows: list[jax.Array] = []
    regular_regime_five_point_ratios: list[float] = []
    irregular_regime_five_point_ratios: list[float] = []
    regular_row_mask: jax.Array = jnp.asarray(
        (True, False, True, False),
    )[:, None, None]

    def rho_fd_rows_for_step(
        rho_step: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        """Return second- and fourth-order rows for one global step field."""
        plus_one: jax.Array = values(
            eta_grid,
            jnp.minimum(rho_grid + rho_step, RHO_MAX),
        )
        plus_two: jax.Array = values(
            eta_grid,
            jnp.minimum(rho_grid + 2.0 * rho_step, RHO_MAX),
        )
        minus_one: jax.Array = values(
            eta_grid,
            jnp.maximum(rho_grid - rho_step, RHO_MIN),
        )
        minus_two: jax.Array = values(
            eta_grid,
            jnp.maximum(rho_grid - 2.0 * rho_step, RHO_MIN),
        )
        plus_three: jax.Array = values(
            eta_grid,
            jnp.minimum(rho_grid + 3.0 * rho_step, RHO_MAX),
        )
        plus_four: jax.Array = values(
            eta_grid,
            jnp.minimum(rho_grid + 4.0 * rho_step, RHO_MAX),
        )
        minus_three: jax.Array = values(
            eta_grid,
            jnp.maximum(rho_grid - 3.0 * rho_step, RHO_MIN),
        )
        minus_four: jax.Array = values(
            eta_grid,
            jnp.maximum(rho_grid - 4.0 * rho_step, RHO_MIN),
        )
        use_forward_stencil: jax.Array = rho_grid - 2.0 * rho_step < RHO_MIN
        use_backward_stencil: jax.Array = rho_grid + 2.0 * rho_step > RHO_MAX
        rho_central: jax.Array = (plus_one - minus_one) / (2.0 * rho_step)
        rho_forward_edge: jax.Array = (
            -3.0 * actual + 4.0 * plus_one - plus_two
        ) / (2.0 * rho_step)
        rho_backward_edge: jax.Array = (
            3.0 * actual - 4.0 * minus_one + minus_two
        ) / (2.0 * rho_step)
        rho_second_order: jax.Array = jnp.where(
            use_forward_stencil[None, ...],
            rho_forward_edge,
            jnp.where(
                use_backward_stencil[None, ...],
                rho_backward_edge,
                rho_central,
            ),
        )
        rho_five_point_central: jax.Array = (
            minus_two - 8.0 * minus_one + 8.0 * plus_one - plus_two
        ) / (12.0 * rho_step)
        rho_five_point_forward: jax.Array = (
            -25.0 * actual
            + 48.0 * plus_one
            - 36.0 * plus_two
            + 16.0 * plus_three
            - 3.0 * plus_four
        ) / (12.0 * rho_step)
        rho_five_point_backward: jax.Array = (
            25.0 * actual
            - 48.0 * minus_one
            + 36.0 * minus_two
            - 16.0 * minus_three
            + 3.0 * minus_four
        ) / (12.0 * rho_step)
        rho_five_point: jax.Array = jnp.where(
            use_forward_stencil[None, ...],
            rho_five_point_forward,
            jnp.where(
                use_backward_stencil[None, ...],
                rho_five_point_backward,
                rho_five_point_central,
            ),
        )
        return rho_second_order, rho_five_point

    rho_exponent: int
    for rho_exponent in RHO_FD_EXPONENTS:
        regular_rho_step: jax.Array = (
            jnp.maximum(REGULAR_RHO_FD_SCALE_FLOOR, jnp.abs(rho_grid))
            * 2.0**-rho_exponent
        )
        irregular_rho_step: jax.Array = (
            jnp.maximum(IRREGULAR_RHO_FD_SCALE_FLOOR, jnp.abs(rho_grid))
            * 2.0**-rho_exponent
        )
        regular_rho_fd: jax.Array
        regular_rho_five_point: jax.Array
        regular_rho_fd, regular_rho_five_point = rho_fd_rows_for_step(
            regular_rho_step
        )
        irregular_rho_fd: jax.Array
        irregular_rho_five_point: jax.Array
        irregular_rho_fd, irregular_rho_five_point = rho_fd_rows_for_step(
            irregular_rho_step
        )
        rho_fd: jax.Array = jnp.where(
            regular_row_mask,
            regular_rho_fd,
            irregular_rho_fd,
        )
        rho_five_point: jax.Array = jnp.where(
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
    eta_fd_stack: jax.Array = jnp.stack(eta_fd_rows)
    eta_five_point_stack: jax.Array = jnp.stack(eta_five_point_rows)
    rho_fd_stack: jax.Array = jnp.stack(rho_fd_rows)
    rho_five_point_stack: jax.Array = jnp.stack(rho_five_point_rows)
    eta_richardson: jax.Array = (
        16.0 * eta_fd_stack[1] - eta_fd_stack[0]
    ) / 15.0
    eta_five_point_richardson: jax.Array = (
        256.0 * eta_five_point_stack[1] - eta_five_point_stack[0]
    ) / 255.0
    eta_five_point_fine_richardson: jax.Array = (
        256.0 * eta_five_point_stack[2] - eta_five_point_stack[1]
    ) / 255.0
    eta_five_point_fine_pair_spread_ratio: float = float(
        jnp.max(
            jnp.abs(eta_five_point_stack[2] - eta_five_point_stack[1])
            / (1.0e-10 + 1.0e-7 * jnp.abs(eta_reference))
        )
    )
    rho_richardson_coarse: jax.Array = (
        16.0 * rho_fd_stack[1] - rho_fd_stack[0]
    ) / 15.0
    rho_richardson_fine: jax.Array = (
        16.0 * rho_fd_stack[2] - rho_fd_stack[1]
    ) / 15.0
    rho_richardson: jax.Array = (
        256.0 * rho_richardson_fine - rho_richardson_coarse
    ) / 255.0
    rho_five_point_richardson: jax.Array = (
        256.0 * rho_five_point_stack[2] - rho_five_point_stack[1]
    ) / 255.0
    rho_five_point_fine_pair_spread_ratio: float = float(
        jnp.max(
            jnp.abs(rho_five_point_stack[2] - rho_five_point_stack[1])
            / (1.0e-10 + 1.0e-7 * jnp.abs(rho_reference))
        )
    )
    eta_plateau_candidates: jax.Array = jnp.concatenate(
        (eta_fd_stack, eta_richardson[None, ...]),
        axis=0,
    )
    rho_plateau_candidates: jax.Array = jnp.concatenate(
        (
            rho_fd_stack,
            rho_richardson_coarse[None, ...],
            rho_richardson_fine[None, ...],
            rho_richardson[None, ...],
        ),
        axis=0,
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
    rho_elementwise_best: jax.Array = jnp.min(
        jnp.abs(rho_plateau_candidates - rho_reference[None, ...])
        / (1.0e-10 + 1.0e-7 * jnp.abs(rho_reference[None, ...])),
        axis=0,
    )
    rho_best_ratio: float = float(jnp.max(rho_elementwise_best))
    rho_worst_index: tuple[int, ...] = tuple(
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
    eta_five_point_error_ratio: jax.Array = jnp.abs(
        eta_five_point_richardson - eta_reference
    ) / (1.0e-10 + 1.0e-7 * jnp.abs(eta_reference))
    eta_five_point_worst_index: tuple[int, ...] = tuple(
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
    rho_five_point_error_ratio: jax.Array = jnp.abs(
        rho_five_point_richardson - rho_reference
    ) / (1.0e-10 + 1.0e-7 * jnp.abs(rho_reference))
    rho_five_point_worst_index: tuple[int, ...] = tuple(
        int(value)
        for value in jnp.unravel_index(
            jnp.argmax(rho_five_point_error_ratio),
            rho_five_point_error_ratio.shape,
        )
    )

    regular: jax.Array = actual[0]
    irregular: jax.Array = actual[1]
    regular_derivative: jax.Array = actual[2]
    irregular_derivative: jax.Array = actual[3]
    wronskian: jax.Array = (
        regular_derivative * irregular - regular * irregular_derivative
    )
    np.testing.assert_allclose(
        wronskian,
        jnp.ones_like(wronskian),
        rtol=0.0,
        atol=1.0e-10,
    )
    metrics: dict[str, Any] = {
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
        message = (
            "eta global five-point Richardson plateau failed mixed budget: "
            f"{eta_five_point_richardson_ratio}"
        )
        raise AssertionError(message)
    if rho_five_point_richardson_ratio > 1.0:
        message = (
            "rho global five-point Richardson plateau failed mixed budget: "
            f"{rho_five_point_richardson_ratio}"
        )
        raise AssertionError(message)


if __name__ == "__main__":
    main()
