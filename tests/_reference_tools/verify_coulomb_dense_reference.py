"""Verify one Coulomb order on the dense mpmath-reference domain.

Compare values with frozen high-precision truth. Check differential-equation
residuals, the Wronskian, origin limits, and the neutral spherical-Bessel
reduction.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from beartype.typing import Any, Dict, Tuple
from jaxtyping import Array, Float64
from numpy.typing import NDArray
from scipy.special import loggamma

from diffpes.radial import coulomb_fg, spherical_bessel_jl

ODE_RESIDUAL_TOLERANCE: float = 1.0e-9
WRONSKIAN_TOLERANCE: float = 1.0e-10


def main() -> None:  # noqa: PLR0915
    """Check dense values, ODE residuals, Wronskians, and asymptotics.

    Raises
    ------
    AssertionError
        If an ODE residual or the Wronskian exceeds its registered tolerance.

    Notes
    -----
    The isolated command reads one angular-momentum order from ``sys.argv``.
    It prints the complete error record to standard output.
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
        dense_etas: Float64[NDArray, " n_eta"] = archive["dense_etas"]
        dense_rhos: Float64[NDArray, " n_rho"] = archive["dense_rhos"]
        reference_regular: Float64[NDArray, "n_eta n_rho"] = archive[
            "dense_f"
        ][order]
        reference_irregular: Float64[NDArray, "n_eta n_rho"] = archive[
            "dense_g"
        ][order]

    eta_grid_numpy: Float64[NDArray, "n_eta n_rho"]
    rho_grid_numpy: Float64[NDArray, "n_eta n_rho"]
    eta_grid_numpy, rho_grid_numpy = np.meshgrid(
        dense_etas,
        dense_rhos,
        indexing="ij",
    )
    eta_grid: Float64[Array, "n_eta n_rho"] = jnp.asarray(eta_grid_numpy)
    rho_grid: Float64[Array, "n_eta n_rho"] = jnp.asarray(rho_grid_numpy)

    @jax.jit
    def values(
        eta: Float64[Array, "n_eta n_rho"],
        rho: Float64[Array, "n_eta n_rho"],
    ) -> Float64[Array, "4 n_eta n_rho"]:
        """Compute all four Coulomb rows.

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
    np.testing.assert_allclose(
        actual[0],
        reference_regular,
        rtol=1.0e-10,
        atol=1.0e-12,
    )
    np.testing.assert_allclose(
        actual[1],
        reference_irregular,
        rtol=1.0e-10,
        atol=1.0e-12,
    )

    @jax.jit
    def rho_direction(
        eta: Float64[Array, "n_eta n_rho"],
        rho: Float64[Array, "n_eta n_rho"],
    ) -> Float64[Array, "4 n_eta n_rho"]:
        """Compute the full unit-rho JVP on the dense product.

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
        ``jax.jvp`` differentiates the complete dense product in one call.
        """
        tangent: Float64[Array, "4 n_eta n_rho"] = jax.jvp(
            lambda argument: jnp.stack(coulomb_fg(order, eta, argument)),
            (rho,),
            (jnp.ones_like(rho),),
        )[1]
        return tangent

    rho_tangent: Float64[Array, "4 n_eta n_rho"] = rho_direction(
        eta_grid, rho_grid
    )
    jax.block_until_ready(rho_tangent)
    ode_factor: Float64[Array, "n_eta n_rho"] = (
        1.0 - 2.0 * eta_grid / rho_grid - order * (order + 1) / rho_grid**2
    )
    regular_residual: Float64[Array, "n_eta n_rho"] = (
        rho_tangent[2] + ode_factor * actual[0]
    )
    irregular_residual: Float64[Array, "n_eta n_rho"] = (
        rho_tangent[3] + ode_factor * actual[1]
    )
    regular_scale: Float64[Array, "n_eta n_rho"] = (
        jnp.abs(rho_tangent[2]) + jnp.abs(ode_factor * actual[0]) + 1.0
    )
    irregular_scale: Float64[Array, "n_eta n_rho"] = (
        jnp.abs(rho_tangent[3]) + jnp.abs(ode_factor * actual[1]) + 1.0
    )
    regular_residual_error: float = float(
        jnp.max(jnp.abs(regular_residual) / regular_scale)
    )
    irregular_residual_error: float = float(
        jnp.max(jnp.abs(irregular_residual) / irregular_scale)
    )
    if regular_residual_error > ODE_RESIDUAL_TOLERANCE:
        regular_index: Tuple[int, int] = tuple(
            int(value)
            for value in jnp.unravel_index(
                jnp.argmax(jnp.abs(regular_residual) / regular_scale),
                regular_residual.shape,
            )
        )
        message: str = (
            f"regular dense ODE residual {regular_residual_error} "
            f"at eta={eta_grid_numpy[regular_index]}, "
            f"rho={rho_grid_numpy[regular_index]}"
        )
        raise AssertionError(message)
    if irregular_residual_error > ODE_RESIDUAL_TOLERANCE:
        irregular_index: Tuple[int, int] = tuple(
            int(value)
            for value in jnp.unravel_index(
                jnp.argmax(jnp.abs(irregular_residual) / irregular_scale),
                irregular_residual.shape,
            )
        )
        message: str = (
            f"irregular dense ODE residual {irregular_residual_error} "
            f"at eta={eta_grid_numpy[irregular_index]}, "
            f"rho={rho_grid_numpy[irregular_index]}"
        )
        raise AssertionError(message)

    wronskian: Float64[Array, "n_eta n_rho"] = (
        actual[2] * actual[1] - actual[0] * actual[3]
    )
    wronskian_error: float = float(jnp.max(jnp.abs(wronskian - 1.0)))
    if wronskian_error > WRONSKIAN_TOLERANCE:
        message: str = f"dense Wronskian error {wronskian_error}"
        raise AssertionError(message)

    zero_eta_index: int = int(np.flatnonzero(dense_etas == 0.0)[0])
    plane_reference: Float64[Array, " n_rho"] = rho_grid[
        zero_eta_index
    ] * spherical_bessel_jl(
        order,
        rho_grid[zero_eta_index],
    )
    np.testing.assert_allclose(
        actual[0, zero_eta_index],
        plane_reference,
        rtol=1.0e-10,
        atol=1.0e-12,
    )

    origin_rho: float = float(dense_rhos[0])
    normalization: Float64[NDArray, " n_eta"] = np.exp(
        order * np.log(2.0)
        - np.pi * dense_etas / 2.0
        + np.real(loggamma(order + 1 + 1j * dense_etas))
        - loggamma(2 * order + 2)
    )
    regular_origin_ratio: Float64[NDArray, " n_eta"] = np.asarray(
        actual[0, :, 0]
    ) / (normalization * origin_rho ** (order + 1))
    irregular_origin_ratio: Float64[NDArray, " n_eta"] = (
        np.asarray(actual[1, :, 0])
        * (2 * order + 1)
        * normalization
        * origin_rho**order
    )
    np.testing.assert_allclose(
        regular_origin_ratio,
        np.ones_like(regular_origin_ratio),
        rtol=0.0,
        atol=5.0e-4,
    )
    np.testing.assert_allclose(
        irregular_origin_ratio,
        np.ones_like(irregular_origin_ratio),
        rtol=0.0,
        atol=5.0e-3,
    )

    regular_mixed_error: float = float(
        np.max(
            np.abs(np.asarray(actual[0]) - reference_regular)
            / (1.0e-12 + 1.0e-10 * np.abs(reference_regular))
        )
    )
    irregular_mixed_error: float = float(
        np.max(
            np.abs(np.asarray(actual[1]) - reference_irregular)
            / (1.0e-12 + 1.0e-10 * np.abs(reference_irregular))
        )
    )
    metrics: Dict[str, Any] = {
        "order": order,
        "dense_shape": list(eta_grid.shape),
        "regular_mixed_budget_ratio": regular_mixed_error,
        "irregular_mixed_budget_ratio": irregular_mixed_error,
        "regular_normalized_ode_residual": regular_residual_error,
        "irregular_normalized_ode_residual": irregular_residual_error,
        "wronskian_absolute_error": wronskian_error,
        "regular_origin_ratio_error": float(
            np.max(np.abs(regular_origin_ratio - 1.0))
        ),
        "irregular_origin_ratio_error": float(
            np.max(np.abs(irregular_origin_ratio - 1.0))
        ),
    }
    print(json.dumps(metrics, sort_keys=True))


if __name__ == "__main__":
    main()
