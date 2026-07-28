# ruff: noqa: B023
"""Generate the frozen Plan-06 G11/D11 Coulomb reference artifacts.

The script is an offline evidence generator. It records the installed mpmath
version and evaluates every frozen value at 80 decimal digits. The dense
residual grid is computed in independent worker processes and the resulting
archive is written with deterministic ZIP metadata. Production code never
imports mpmath or this module.
"""

from __future__ import annotations

import hashlib
import io
import json
import platform
import shutil
import subprocess
import zipfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

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
DENSE_ETAS: tuple[float, ...] = tuple(
    float(value) for value in np.linspace(-3.0, 3.0, 25)
)
DENSE_RHOS: tuple[float, ...] = tuple(
    float(value) for value in np.geomspace(1.0e-4, 40.0, 257)
)
REFERENCE_DPS: int = 80


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


def dense_value_rows(order: int) -> tuple[int, np.ndarray, np.ndarray]:
    """Evaluate dense F and G values for one independent static order."""
    mp.mp.dps = REFERENCE_DPS
    regular = np.empty((len(DENSE_ETAS), len(DENSE_RHOS)), dtype=np.float64)
    irregular = np.empty_like(regular)
    eta_index: int
    eta_float: float
    rho_index: int
    rho_float: float
    for eta_index, eta_float in enumerate(DENSE_ETAS):
        eta = mp.mpf(str(eta_float))
        for rho_index, rho_float in enumerate(DENSE_RHOS):
            rho = mp.mpf(str(rho_float))
            regular[eta_index, rho_index] = float(mp.coulombf(order, eta, rho))
            irregular[eta_index, rho_index] = float(
                mp.coulombg(order, eta, rho)
            )
    return order, regular, irregular


def _array_bytes(array: np.ndarray) -> bytes:
    """Serialize one NumPy array without filesystem timestamp metadata."""
    output = io.BytesIO()
    np.lib.format.write_array(output, np.asarray(array), allow_pickle=False)
    return output.getvalue()


def _write_deterministic_npz(
    path: Path,
    arrays: dict[str, np.ndarray],
) -> None:
    """Write an NPZ whose members have stable order and timestamps."""
    with zipfile.ZipFile(
        path,
        mode="w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=9,
    ) as archive:
        name: str
        array: np.ndarray
        for name, array in sorted(arrays.items()):
            member = zipfile.ZipInfo(
                filename=f"{name}.npy",
                date_time=(1980, 1, 1, 0, 0, 0),
            )
            member.compress_type = zipfile.ZIP_DEFLATED
            member.external_attr = 0o100644 << 16
            archive.writestr(
                member,
                _array_bytes(array),
                compress_type=zipfile.ZIP_DEFLATED,
                compresslevel=9,
            )


def _sha256(path: Path) -> str:
    """Return one frozen artifact checksum."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_head(root: Path) -> str:
    """Return the source revision used by the offline generator."""
    git: str | None = shutil.which("git")
    if git is None:
        message = "git is required to record Coulomb evidence provenance"
        raise RuntimeError(message)
    completed = subprocess.run(  # noqa: S603
        [git, "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def main() -> None:
    """Write dense and sparse 80-digit references plus their provenance."""
    mp.mp.dps = REFERENCE_DPS
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

    dense_regular = np.empty(
        (len(ORDERS), len(DENSE_ETAS), len(DENSE_RHOS)),
        dtype=np.float64,
    )
    dense_irregular = np.empty_like(dense_regular)
    with ProcessPoolExecutor(max_workers=len(ORDERS)) as executor:
        dense_rows: tuple[int, np.ndarray, np.ndarray]
        for dense_rows in executor.map(dense_value_rows, ORDERS):
            order, regular_row, irregular_row = dense_rows
            dense_regular[order] = regular_row
            dense_irregular[order] = irregular_row

    root: Path = Path(__file__).parents[2]
    target_directory: Path = (
        root / "tests" / "test_diffpes" / "test_radial" / "data"
    )
    target_directory.mkdir(parents=True, exist_ok=True)
    target: Path = target_directory / "coulomb_mpmath_80digit.npz"
    arrays: dict[str, np.ndarray] = {
        "orders": np.asarray(ORDERS, dtype=np.int64),
        "etas": np.asarray(ETAS, dtype=np.float64),
        "rhos": np.asarray(RHOS, dtype=np.float64),
        "dense_etas": np.asarray(DENSE_ETAS, dtype=np.float64),
        "dense_rhos": np.asarray(DENSE_RHOS, dtype=np.float64),
        "dense_f": dense_regular,
        "dense_g": dense_irregular,
        "phase": phase,
        "phase_eta": phase_eta,
        **values,
    }
    _write_deterministic_npz(target, arrays)

    generator_path: Path = Path(__file__).resolve()
    manifest_path: Path = (
        target_directory / "coulomb_mpmath_80digit.manifest.json"
    )
    manifest: dict[str, Any] = {
        "schema": "diffpes.plan06.coulomb-reference.v2",
        "gates": ["06.G11", "06.D11"],
        "source_revision": _git_head(root),
        "generator": generator_path.relative_to(root).as_posix(),
        "generator_sha256": _sha256(generator_path),
        "reference_engine": {
            "name": "mpmath",
            "version": mp.__version__,
            "decimal_digits": REFERENCE_DPS,
        },
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "platform": platform.platform(),
        },
        "sparse_value_derivative_product": {
            "orders": list(ORDERS),
            "etas": list(ETAS),
            "rhos": list(RHOS),
        },
        "dense_value_residual_product": {
            "orders": list(ORDERS),
            "eta_count": len(DENSE_ETAS),
            "eta_interval": [-3.0, 3.0],
            "rho_count": len(DENSE_RHOS),
            "rho_interval": [1.0e-4, 40.0],
            "rho_spacing": "geometric",
        },
        "derivative_construction": {
            "rho": "exact Coulomb order recurrence at 80 decimal digits",
            "eta": "80-digit symmetric quotient with step 1e-20",
            "rho_second": "independent Coulomb ODE identity",
        },
        "normalization": "mpmath Coulomb F_l/G_l convention; F'G-FG'=1",
        "archive": target.name,
        "archive_sha256": _sha256(target),
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    checksums_path: Path = target_directory / "coulomb_SHA256SUMS"
    checksum_paths: tuple[Path, ...] = (
        generator_path,
        manifest_path,
        target,
    )
    checksums_path.write_text(
        "".join(
            f"{_sha256(path)}  {path.relative_to(root).as_posix()}\n"
            for path in checksum_paths
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
