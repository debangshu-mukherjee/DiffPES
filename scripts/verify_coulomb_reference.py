"""Verify one frozen Plan-06 Coulomb order in an isolated process."""

import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

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


if __name__ == "__main__":
    main()
