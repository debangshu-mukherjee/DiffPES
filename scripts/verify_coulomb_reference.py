"""Verify one frozen Plan-06 Coulomb order in an isolated process."""

import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from scipy.special import loggamma

from diffpes.radial import coulomb_fg


def main() -> None:
    """Compare one requested order with the frozen mixed tolerance."""
    order: int = int(sys.argv[1])
    path: Path = (
        Path(__file__).parents[1]
        / "tests"
        / "test_diffpes"
        / "test_radial"
        / "data"
        / "coulomb_mpmath_80digit.npz"
    )
    archive: np.lib.npyio.NpzFile = np.load(path)
    eta_grid: np.ndarray
    rho_grid: np.ndarray
    eta_grid, rho_grid = np.meshgrid(
        archive["etas"],
        archive["rhos"],
        indexing="ij",
    )
    actual: tuple[jnp.ndarray, ...] = coulomb_fg(
        order,
        jnp.asarray(eta_grid),
        jnp.asarray(rho_grid),
    )
    names: tuple[str, ...] = ("f", "g", "df_drho", "dg_drho")
    name: str
    values: jnp.ndarray
    for name, values in zip(names, actual, strict=True):
        np.testing.assert_allclose(
            values,
            archive[name][order],
            rtol=1.0e-10,
            atol=1.0e-12,
        )
    eta_tangent: tuple[jnp.ndarray, ...] = jax.jvp(
        lambda arguments: coulomb_fg(
            order,
            arguments,
            jnp.asarray(rho_grid),
        ),
        (jnp.asarray(eta_grid),),
        (jnp.ones_like(jnp.asarray(eta_grid)),),
    )[1]
    rho_tangent: tuple[jnp.ndarray, ...] = jax.jvp(
        lambda arguments: coulomb_fg(
            order,
            jnp.asarray(eta_grid),
            arguments,
        ),
        (jnp.asarray(rho_grid),),
        (jnp.ones_like(jnp.asarray(rho_grid)),),
    )[1]
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
    for name, values in zip(eta_names, eta_tangent, strict=True):
        np.testing.assert_allclose(
            values,
            archive[name][order],
            rtol=1.0e-7,
            atol=1.0e-10,
        )
    for name, values in zip(rho_names, rho_tangent, strict=True):
        np.testing.assert_allclose(
            values,
            archive[name][order],
            rtol=1.0e-7,
            atol=1.0e-10,
        )
    regular: jnp.ndarray = actual[0]
    irregular: jnp.ndarray = actual[1]
    regular_derivative: jnp.ndarray = actual[2]
    irregular_derivative: jnp.ndarray = actual[3]
    wronskian: jnp.ndarray = (
        regular_derivative * irregular - regular * irregular_derivative
    )
    np.testing.assert_allclose(
        wronskian,
        jnp.ones_like(wronskian),
        rtol=0.0,
        atol=1.0e-10,
    )

    ode_factor: jnp.ndarray = (
        1.0
        - 2.0 * jnp.asarray(eta_grid) / jnp.asarray(rho_grid)
        - order * (order + 1) / jnp.asarray(rho_grid) ** 2
    )
    regular_residual: jnp.ndarray = rho_tangent[2] + ode_factor * regular
    irregular_residual: jnp.ndarray = rho_tangent[3] + ode_factor * irregular
    regular_scale: jnp.ndarray = (
        jnp.abs(rho_tangent[2]) + jnp.abs(ode_factor * regular) + 1.0
    )
    irregular_scale: jnp.ndarray = (
        jnp.abs(rho_tangent[3]) + jnp.abs(ode_factor * irregular) + 1.0
    )
    dense_interior: tuple[slice, slice] = (slice(None), slice(3, -1))
    regular_residual_error: float = float(
        jnp.max(
            jnp.abs(regular_residual[dense_interior])
            / regular_scale[dense_interior]
        )
    )
    irregular_residual_error: float = float(
        jnp.max(
            jnp.abs(irregular_residual[dense_interior])
            / irregular_scale[dense_interior]
        )
    )
    residual_tolerance: float = 1.0e-9
    if regular_residual_error > residual_tolerance:
        message = f"regular ODE residual {regular_residual_error}"
        raise AssertionError(message)
    if irregular_residual_error > residual_tolerance:
        message = f"irregular ODE residual {irregular_residual_error}"
        raise AssertionError(message)

    eta_values: np.ndarray = archive["etas"]
    normalization: np.ndarray = np.exp(
        order * np.log(2.0)
        - np.pi * eta_values / 2.0
        + np.real(loggamma(order + 1 + 1j * eta_values))
        - loggamma(2 * order + 2)
    )
    origin_rho: float = float(archive["rhos"][0])
    regular_origin_ratio: np.ndarray = np.asarray(regular[:, 0]) / (
        normalization * origin_rho ** (order + 1)
    )
    irregular_origin_ratio: np.ndarray = (
        np.asarray(irregular[:, 0])
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


if __name__ == "__main__":
    main()
