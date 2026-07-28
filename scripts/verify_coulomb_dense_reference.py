"""Verify one Plan-06 Coulomb order on the frozen dense G11 domain."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from scipy.special import loggamma

from diffpes.radial import coulomb_fg, spherical_bessel_jl

ODE_RESIDUAL_TOLERANCE: float = 1.0e-9
WRONSKIAN_TOLERANCE: float = 1.0e-10


def main() -> None:  # noqa: PLR0915
    """Check dense values, ODE residuals, Wronskians, and asymptotics."""
    order: int = int(sys.argv[1])
    path: Path = (
        Path(__file__).parents[1]
        / "tests"
        / "test_diffpes"
        / "test_radial"
        / "data"
        / "coulomb_mpmath_80digit.npz"
    )
    with np.load(path) as archive:
        dense_etas: np.ndarray = archive["dense_etas"]
        dense_rhos: np.ndarray = archive["dense_rhos"]
        reference_regular: np.ndarray = archive["dense_f"][order]
        reference_irregular: np.ndarray = archive["dense_g"][order]

    eta_grid_numpy: np.ndarray
    rho_grid_numpy: np.ndarray
    eta_grid_numpy, rho_grid_numpy = np.meshgrid(
        dense_etas,
        dense_rhos,
        indexing="ij",
    )
    eta_grid = jnp.asarray(eta_grid_numpy)
    rho_grid = jnp.asarray(rho_grid_numpy)

    @jax.jit
    def values(
        eta: jax.Array,
        rho: jax.Array,
    ) -> jax.Array:
        """Return all four production Coulomb rows."""
        return jnp.stack(coulomb_fg(order, eta, rho))

    actual: jax.Array = values(eta_grid, rho_grid)
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
        eta: jax.Array,
        rho: jax.Array,
    ) -> jax.Array:
        """Return the full unit-rho JVP on the dense product."""
        tangent: jax.Array = jax.jvp(
            lambda argument: jnp.stack(coulomb_fg(order, eta, argument)),
            (rho,),
            (jnp.ones_like(rho),),
        )[1]
        return tangent

    rho_tangent: jax.Array = rho_direction(eta_grid, rho_grid)
    jax.block_until_ready(rho_tangent)
    ode_factor: jax.Array = (
        1.0 - 2.0 * eta_grid / rho_grid - order * (order + 1) / rho_grid**2
    )
    regular_residual: jax.Array = rho_tangent[2] + ode_factor * actual[0]
    irregular_residual: jax.Array = rho_tangent[3] + ode_factor * actual[1]
    regular_scale: jax.Array = (
        jnp.abs(rho_tangent[2]) + jnp.abs(ode_factor * actual[0]) + 1.0
    )
    irregular_scale: jax.Array = (
        jnp.abs(rho_tangent[3]) + jnp.abs(ode_factor * actual[1]) + 1.0
    )
    regular_residual_error: float = float(
        jnp.max(jnp.abs(regular_residual) / regular_scale)
    )
    irregular_residual_error: float = float(
        jnp.max(jnp.abs(irregular_residual) / irregular_scale)
    )
    if regular_residual_error > ODE_RESIDUAL_TOLERANCE:
        regular_index = tuple(
            int(value)
            for value in jnp.unravel_index(
                jnp.argmax(jnp.abs(regular_residual) / regular_scale),
                regular_residual.shape,
            )
        )
        message = (
            f"regular dense ODE residual {regular_residual_error} "
            f"at eta={eta_grid_numpy[regular_index]}, "
            f"rho={rho_grid_numpy[regular_index]}"
        )
        raise AssertionError(message)
    if irregular_residual_error > ODE_RESIDUAL_TOLERANCE:
        irregular_index = tuple(
            int(value)
            for value in jnp.unravel_index(
                jnp.argmax(jnp.abs(irregular_residual) / irregular_scale),
                irregular_residual.shape,
            )
        )
        message = (
            f"irregular dense ODE residual {irregular_residual_error} "
            f"at eta={eta_grid_numpy[irregular_index]}, "
            f"rho={rho_grid_numpy[irregular_index]}"
        )
        raise AssertionError(message)

    wronskian: jax.Array = actual[2] * actual[1] - actual[0] * actual[3]
    wronskian_error: float = float(jnp.max(jnp.abs(wronskian - 1.0)))
    if wronskian_error > WRONSKIAN_TOLERANCE:
        message = f"dense Wronskian error {wronskian_error}"
        raise AssertionError(message)

    zero_eta_index: int = int(np.flatnonzero(dense_etas == 0.0)[0])
    plane_reference: jax.Array = rho_grid[
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
    normalization: np.ndarray = np.exp(
        order * np.log(2.0)
        - np.pi * dense_etas / 2.0
        + np.real(loggamma(order + 1 + 1j * dense_etas))
        - loggamma(2 * order + 2)
    )
    regular_origin_ratio: np.ndarray = np.asarray(actual[0, :, 0]) / (
        normalization * origin_rho ** (order + 1)
    )
    irregular_origin_ratio: np.ndarray = (
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
    metrics: dict[str, Any] = {
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
