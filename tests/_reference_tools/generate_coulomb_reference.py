# ruff: noqa: B023
"""Generate frozen Coulomb value and assembly-derivative reference artifacts.

The script is an offline evidence generator. It records the installed mpmath
version and evaluates every frozen value at 80 decimal digits. Independent
worker processes compute the dense residual grid. Deterministic ZIP metadata
preserves the resulting archive. Production code never imports mpmath or this
module.
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

import mpmath as mp
import numpy as np
from beartype.typing import Any, Dict, Tuple
from jaxtyping import Float64, Shaped
from numpy.typing import NDArray

ORDERS: Tuple[int, ...] = tuple(range(5))
ETAS: Tuple[float, ...] = (-3.0, -1.0, -0.25, 0.0, 0.25, 1.0, 3.0)
RHOS: Tuple[float, ...] = (
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
DENSE_ETAS: Tuple[float, ...] = tuple(
    float(value) for value in np.linspace(-3.0, 3.0, 25)
)
DENSE_RHOS: Tuple[float, ...] = tuple(
    float(value) for value in np.geomspace(1.0e-4, 40.0, 257)
)
REFERENCE_DPS: int = 80


def coulomb_rows(
    order: int,
    eta: mp.mpf,
    rho: mp.mpf,
) -> Tuple[mp.mpf, mp.mpf, mp.mpf, mp.mpf]:
    """Evaluate Coulomb values and radial derivatives from the recurrence.

    Parameters
    ----------
    order : int
        Nonnegative angular-momentum order.
    eta : mp.mpf
        Dimensionless Sommerfeld parameter.
    rho : mp.mpf
        Positive dimensionless radial coordinate.

    Returns
    -------
    result : Tuple[mp.mpf, mp.mpf, mp.mpf, mp.mpf]
        The regular value, irregular value, and their radial derivatives.

    Notes
    -----
    The expression uses the adjacent-order Coulomb recurrence at the current
    mpmath working precision.
    """
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
    result: Tuple[mp.mpf, mp.mpf, mp.mpf, mp.mpf] = (
        regular,
        irregular,
        regular_derivative,
        irregular_derivative,
    )
    return result


def dense_value_rows(
    order: int,
) -> Tuple[
    int,
    Float64[NDArray, "n_eta n_rho"],
    Float64[NDArray, "n_eta n_rho"],
]:
    """Evaluate dense Coulomb values for one independent static order.

    Parameters
    ----------
    order : int
        Nonnegative angular-momentum order.

    Returns
    -------
    result : Tuple[int, Float64[NDArray, "n_eta n_rho"],
        Float64[NDArray, "n_eta n_rho"]]
        The order and its dense regular and irregular value tables.

    Notes
    -----
    The worker evaluates each registered coordinate with 80 decimal digits.
    It converts only the final values to canonical float64 storage.
    """
    mp.mp.dps = REFERENCE_DPS
    regular: Float64[NDArray, "n_eta n_rho"] = np.empty(
        (len(DENSE_ETAS), len(DENSE_RHOS)), dtype=np.float64
    )
    irregular: Float64[NDArray, "n_eta n_rho"] = np.empty_like(regular)
    eta_index: int
    eta_float: float
    rho_index: int
    rho_float: float
    for eta_index, eta_float in enumerate(DENSE_ETAS):
        eta: mp.mpf = mp.mpf(str(eta_float))
        for rho_index, rho_float in enumerate(DENSE_RHOS):
            rho: mp.mpf = mp.mpf(str(rho_float))
            regular[eta_index, rho_index] = float(mp.coulombf(order, eta, rho))
            irregular[eta_index, rho_index] = float(
                mp.coulombg(order, eta, rho)
            )
    result: Tuple[
        int,
        Float64[NDArray, "n_eta n_rho"],
        Float64[NDArray, "n_eta n_rho"],
    ] = (order, regular, irregular)
    return result


def _array_bytes(array: Shaped[NDArray, "..."]) -> bytes:
    """PRIVATE: Serialize one NumPy array without timestamp metadata.

    Parameters
    ----------
    array : Shaped[NDArray, "..."]
        Array to serialize.

    Returns
    -------
    payload : bytes
        The exact ``.npy`` byte stream for the array.

    Notes
    -----
    ``np.lib.format.write_array`` writes into an in-memory buffer with
    pickle disabled, so equal arrays always produce equal bytes.
    """
    output: io.BytesIO = io.BytesIO()
    np.lib.format.write_array(output, np.asarray(array), allow_pickle=False)
    payload: bytes = output.getvalue()
    return payload


def _write_deterministic_npz(
    path: Path,
    arrays: Dict[str, Shaped[NDArray, "..."]],
) -> None:
    """PRIVATE: Write an NPZ whose members have stable order and dates.

    Parameters
    ----------
    path : Path
        Destination NPZ path.
    arrays : Dict[str, Shaped[NDArray, "..."]]
        Named arrays to store.

    Notes
    -----
    Sort members by name. Give each member the fixed 1980-01-01 timestamp and
    file mode. Use DEFLATE level 9. Identical arrays then produce identical
    archive bytes.
    """
    with zipfile.ZipFile(
        path,
        mode="w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=9,
    ) as archive:
        archive: zipfile.ZipFile
        name: str
        array: Shaped[NDArray, "..."]
        for name, array in sorted(arrays.items()):
            member: zipfile.ZipInfo = zipfile.ZipInfo(
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
    """PRIVATE: Return one frozen artifact checksum.

    Parameters
    ----------
    path : Path
        File to digest.

    Returns
    -------
    digest : str
        Lowercase hexadecimal SHA-256 of the file bytes.

    Notes
    -----
    The function reads the complete file into memory before hashing;
    every evidence file stays small enough for that.
    """
    digest: str = hashlib.sha256(path.read_bytes()).hexdigest()
    return digest


def _git_head(root: Path) -> str:
    """PRIVATE: Return the source revision of the offline generator run.

    Parameters
    ----------
    root : Path
        Repository root for the ``git rev-parse`` call.

    Returns
    -------
    revision : str
        Full HEAD commit hash.

    Raises
    ------
    RuntimeError
        If no ``git`` executable exists on the path.

    Notes
    -----
    The revision enters the manifest as evidence provenance only; no
    consumer recomputes it. A nonzero ``git rev-parse HEAD`` command
    propagates :class:`subprocess.CalledProcessError` through ``check=True``.
    """
    git: str | None = shutil.which("git")
    if git is None:
        message: str = "git is required to record Coulomb evidence provenance"
        raise RuntimeError(message)
    completed: subprocess.CompletedProcess[str] = subprocess.run(  # noqa: S603
        [git, "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    revision: str = completed.stdout.strip()
    return revision


def main() -> None:  # noqa: PLR0915 -- one deterministic authority assembly.
    """Write dense and sparse 80-digit references plus their provenance.

    Notes
    -----
    The generator evaluates sparse derivatives and dense values independently.
    It writes deterministic array bytes, provenance, and checksums.
    """
    mp.mp.dps = REFERENCE_DPS
    shape: Tuple[int, int, int] = (len(ORDERS), len(ETAS), len(RHOS))
    values: Dict[str, Float64[NDArray, "n_order n_eta n_rho"]] = {
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
    phase: Float64[NDArray, "n_order n_eta"] = np.empty(
        (len(ORDERS), len(ETAS)), dtype=np.float64
    )
    phase_eta: Float64[NDArray, "n_order n_eta"] = np.empty_like(phase)
    eta_step: mp.mpf = mp.mpf("1e-20")

    order_index: int
    order: int
    eta_index: int
    eta_float: float
    rho_index: int
    rho_float: float
    for order_index, order in enumerate(ORDERS):
        for eta_index, eta_float in enumerate(ETAS):
            eta: mp.mpf = mp.mpf(str(eta_float))
            phase[order_index, eta_index] = float(
                mp.im(mp.loggamma(order + 1 + 1j * eta))
            )
            phase_eta[order_index, eta_index] = float(
                mp.re(mp.digamma(order + 1 + 1j * eta))
            )
            for rho_index, rho_float in enumerate(RHOS):
                rho: mp.mpf = mp.mpf(str(rho_float))
                rows: Tuple[mp.mpf, mp.mpf, mp.mpf, mp.mpf] = coulomb_rows(
                    order,
                    eta,
                    rho,
                )
                rows_plus: Tuple[mp.mpf, mp.mpf, mp.mpf, mp.mpf] = (
                    coulomb_rows(order, eta + eta_step, rho)
                )
                rows_minus: Tuple[mp.mpf, mp.mpf, mp.mpf, mp.mpf] = (
                    coulomb_rows(order, eta - eta_step, rho)
                )
                eta_rows: Tuple[mp.mpf, mp.mpf, mp.mpf, mp.mpf] = tuple(
                    (plus - minus) / (2 * eta_step)
                    for plus, minus in zip(rows_plus, rows_minus, strict=True)
                )
                f_value: mp.mpf = rows[0]
                g_value: mp.mpf = rows[1]
                df_value: mp.mpf = rows[2]
                dg_value: mp.mpf = rows[3]
                ode_factor: mp.mpf = (
                    1 - 2 * eta / rho - order * (order + 1) / rho**2
                )
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

    dense_regular: Float64[NDArray, "n_order n_eta n_rho"] = np.empty(
        (len(ORDERS), len(DENSE_ETAS), len(DENSE_RHOS)),
        dtype=np.float64,
    )
    dense_irregular: Float64[NDArray, "n_order n_eta n_rho"] = np.empty_like(
        dense_regular
    )
    with ProcessPoolExecutor(max_workers=len(ORDERS)) as executor:
        executor: ProcessPoolExecutor
        dense_rows: Tuple[
            int,
            Float64[NDArray, "n_eta n_rho"],
            Float64[NDArray, "n_eta n_rho"],
        ]
        for dense_rows in executor.map(dense_value_rows, ORDERS):
            regular_row: Float64[NDArray, "n_eta n_rho"]
            irregular_row: Float64[NDArray, "n_eta n_rho"]
            order, regular_row, irregular_row = dense_rows
            dense_regular[order] = regular_row
            dense_irregular[order] = irregular_row

    root: Path = Path(__file__).parents[2]
    target_directory: Path = (
        root / "tests" / "test_diffpes" / "test_radial" / "data"
    )
    target_directory.mkdir(parents=True, exist_ok=True)
    target: Path = target_directory / "coulomb_mpmath_80digit.npz"
    arrays: Dict[str, Shaped[NDArray, "..."]] = {
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
    manifest: Dict[str, Any] = {
        "schema": "diffpes.coulomb-mpmath-reference.v2",
        "requirements": [
            "coulomb-mpmath-reference",
            "coulomb-assembly-gradient",
        ],
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
    checksum_paths: Tuple[Path, ...] = (
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
