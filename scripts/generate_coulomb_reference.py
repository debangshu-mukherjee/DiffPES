# ruff: noqa: B023
"""Generate the frozen Plan-06 G11/D11 Coulomb reference artifact.

The script is an offline evidence generator. It requires mpmath 1.3.0 and
evaluates every frozen value at 80 decimal digits. Production code never
imports mpmath or this module.
"""

from pathlib import Path

import mpmath as mp
import numpy as np

ORDERS: tuple[int, ...] = tuple(range(5))
ETAS: tuple[float, ...] = (-3.0, -1.0, -0.25, 0.0, 0.25, 1.0, 3.0)
RHOS: tuple[float, ...] = (
    1.0e-4,
    3.0e-4,
    1.0e-3,
    1.0e-2,
    0.1,
    1.0,
    4.0,
    10.0,
    20.0,
    40.0,
)


def coulomb_rows(
    order: int,
    eta: mp.mpf,
    rho: mp.mpf,
) -> tuple[mp.mpf, mp.mpf, mp.mpf, mp.mpf]:
    """Evaluate F, G, and rho derivatives from the order recurrence."""
    regular: mp.mpf = mp.coulombf(order, eta, rho)
    irregular: mp.mpf = mp.coulombg(order, eta, rho)
    regular_next: mp.mpf = mp.coulombf(order + 1, eta, rho)
    irregular_next: mp.mpf = mp.coulombg(order + 1, eta, rho)
    scale: mp.mpf = mp.sqrt((order + 1) ** 2 + eta**2)
    coefficient: mp.mpf = (order + 1) ** 2 / rho + eta
    regular_derivative: mp.mpf = (
        coefficient * regular - scale * regular_next
    ) / (order + 1)
    irregular_derivative: mp.mpf = (
        coefficient * irregular - scale * irregular_next
    ) / (order + 1)
    result: tuple[mp.mpf, mp.mpf, mp.mpf, mp.mpf] = (
        regular,
        irregular,
        regular_derivative,
        irregular_derivative,
    )
    return result


def main() -> None:
    """Write the complete 80-digit-generated reference as float64 arrays."""
    mp.mp.dps = 80
    shape: tuple[int, int, int] = (len(ORDERS), len(ETAS), len(RHOS))
    values: dict[str, np.ndarray] = {
        name: np.empty(shape, dtype=np.float64)
        for name in (
            "f",
            "g",
            "df_drho",
            "dg_drho",
            "d2f_drho2",
            "d2g_drho2",
            "df_deta",
            "dg_deta",
            "d_df_drho_deta",
            "d_dg_drho_deta",
        )
    }
    phase: np.ndarray = np.empty((len(ORDERS), len(ETAS)), dtype=np.float64)
    phase_eta: np.ndarray = np.empty_like(phase)
    eta_step: mp.mpf = mp.mpf("1e-20")

    for order_index, order in enumerate(ORDERS):
        for eta_index, eta_float in enumerate(ETAS):
            eta = mp.mpf(str(eta_float))
            phase[order_index, eta_index] = float(
                mp.im(mp.loggamma(order + 1 + 1j * eta))
            )
            phase_eta[order_index, eta_index] = float(
                mp.re(mp.digamma(order + 1 + 1j * eta))
            )
            for rho_index, rho_float in enumerate(RHOS):
                rho = mp.mpf(str(rho_float))
                rows: tuple[mp.mpf, mp.mpf, mp.mpf, mp.mpf] = coulomb_rows(
                    order,
                    eta,
                    rho,
                )
                rows_plus: tuple[mp.mpf, mp.mpf, mp.mpf, mp.mpf] = (
                    coulomb_rows(order, eta + eta_step, rho)
                )
                rows_minus: tuple[mp.mpf, mp.mpf, mp.mpf, mp.mpf] = (
                    coulomb_rows(order, eta - eta_step, rho)
                )
                eta_rows: tuple[mp.mpf, mp.mpf, mp.mpf, mp.mpf] = tuple(
                    (plus - minus) / (2 * eta_step)
                    for plus, minus in zip(rows_plus, rows_minus, strict=True)
                )
                f_value: mp.mpf = rows[0]
                g_value: mp.mpf = rows[1]
                df_value: mp.mpf = rows[2]
                dg_value: mp.mpf = rows[3]
                ode_factor = 1 - 2 * eta / rho - order * (order + 1) / rho**2
                values["f"][order_index, eta_index, rho_index] = float(f_value)
                values["g"][order_index, eta_index, rho_index] = float(g_value)
                values["df_drho"][order_index, eta_index, rho_index] = float(
                    df_value
                )
                values["dg_drho"][order_index, eta_index, rho_index] = float(
                    dg_value
                )
                values["d2f_drho2"][order_index, eta_index, rho_index] = float(
                    -ode_factor * f_value
                )
                values["d2g_drho2"][order_index, eta_index, rho_index] = float(
                    -ode_factor * g_value
                )
                values["df_deta"][order_index, eta_index, rho_index] = float(
                    eta_rows[0]
                )
                values["dg_deta"][order_index, eta_index, rho_index] = float(
                    eta_rows[1]
                )
                values["d_df_drho_deta"][order_index, eta_index, rho_index] = (
                    float(eta_rows[2])
                )
                values["d_dg_drho_deta"][order_index, eta_index, rho_index] = (
                    float(eta_rows[3])
                )

    target: Path = (
        Path(__file__).parents[1]
        / "tests"
        / "test_diffpes"
        / "test_radial"
        / "data"
        / "coulomb_mpmath_80digit.npz"
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        target,
        orders=np.asarray(ORDERS, dtype=np.int64),
        etas=np.asarray(ETAS, dtype=np.float64),
        rhos=np.asarray(RHOS, dtype=np.float64),
        phase=phase,
        phase_eta=phase_eta,
        generator="mpmath 1.3.0; mp.dps=80; Plan 06 frozen G11/D11 product",
        **values,
    )


if __name__ == "__main__":
    main()
