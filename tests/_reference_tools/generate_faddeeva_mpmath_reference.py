"""Generate the frozen mpmath Faddeeva value and derivative reference artifact.

The script evaluates the preregistered upper-half-plane grid at 100 decimal
digits. It rounds values only after forming the Faddeeva ODE derivative in
arbitrary precision. Production code never imports this module or mpmath.
"""

from __future__ import annotations

import hashlib
import io
import json
import platform
import zipfile
from pathlib import Path

import mpmath as mp
import numpy as np
from beartype.typing import Any, Dict, List, Tuple
from jaxtyping import Complex128, Float64, Shaped
from numpy.typing import NDArray

REFERENCE_DPS: int = 100
MAXIMUM_RADIUS: float = 1.0e8
RADII: Float64[NDArray, " n_radius"] = np.concatenate(
    (np.asarray([0.0]), np.logspace(-12.0, 8.0, 161))
)
ANGLES: Float64[NDArray, " n_angle"] = np.linspace(0.0, np.pi, 33)
EXPLICIT_POINTS: Complex128[NDArray, " n_explicit"] = np.asarray(
    [
        2.5 + 0.0j,
        3.0 + 1.0j,
        6.0 + 0.0j,
        MAXIMUM_RADIUS + 0.0j,
        MAXIMUM_RADIUS * 1.0j,
    ],
    dtype=np.complex128,
)
DIRECTIONS: Complex128[NDArray, " n_direction"] = np.asarray(
    [1.0 + 0.0j, 0.0 + 1.0j, 0.6 + 0.8j],
    dtype=np.complex128,
)


def _reference_points() -> Complex128[NDArray, " n_point"]:
    """PRIVATE: Return the preregistered grid with duplicate zeros removed.

    Returns
    -------
    result : Complex128[NDArray, " n_point"]
        Unique complex128 evaluation points in the closed upper half
        plane with ``|z| <= 1e8``.

    Implementation Logic
    --------------------
    Form the polar product of frozen radii and angles. Clamp ``sin`` to keep
    every imaginary part nonnegative. Rescale points beyond ``|z| = 1e8``
    onto that envelope. Append the explicit spot points. Remove exact
    duplicates with a first-seen filter that preserves order.
    """
    radial_points: Complex128[NDArray, " n_radial"] = (
        RADII[:, None] * np.cos(ANGLES)[None, :]
        + 1j * RADII[:, None] * np.maximum(np.sin(ANGLES), 0.0)[None, :]
    ).reshape(-1)
    magnitudes: Float64[NDArray, " n_radial"] = np.abs(radial_points)
    scales: Float64[NDArray, " n_radial"] = np.ones_like(magnitudes)
    np.divide(
        MAXIMUM_RADIUS,
        magnitudes,
        out=scales,
        where=magnitudes > MAXIMUM_RADIUS,
    )
    radial_points = radial_points * scales
    combined: Complex128[NDArray, " n_combined"] = np.concatenate(
        (radial_points, EXPLICIT_POINTS)
    )
    unique: List[complex] = []
    seen: set[Tuple[float, float]] = set()
    value: complex
    for value in combined:
        key: Tuple[float, float] = (float(value.real), float(value.imag))
        if key not in seen:
            seen.add(key)
            unique.append(value)
    result: Complex128[NDArray, " n_point"] = np.asarray(
        unique, dtype=np.complex128
    )
    return result


def _array_bytes(array: Shaped[NDArray, "..."]) -> bytes:
    """PRIVATE: Serialize one NumPy array without timestamp metadata.

    Parameters
    ----------
    array : Shaped[NDArray, "..."]
        Array to serialize.

    Returns
    -------
    result : bytes
        The exact ``.npy`` byte stream for the array.

    Notes
    -----
    ``np.lib.format.write_array`` writes into an in-memory buffer with
    pickle disabled, so equal arrays always produce equal bytes.
    """
    output: io.BytesIO = io.BytesIO()
    np.lib.format.write_array(output, np.asarray(array), allow_pickle=False)
    result: bytes = output.getvalue()
    return result


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
    archive: zipfile.ZipFile
    with zipfile.ZipFile(
        path,
        mode="w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=9,
    ) as archive:
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
    """PRIVATE: Return the SHA-256 digest of one evidence file.

    Parameters
    ----------
    path : Path
        File to digest.

    Returns
    -------
    result : str
        Lowercase hexadecimal SHA-256 of the file bytes.

    Notes
    -----
    The function reads the complete file into memory before hashing;
    every evidence file stays small enough for that.
    """
    result: str = hashlib.sha256(path.read_bytes()).hexdigest()
    return result


def main() -> None:
    """Write the 100-digit values, derivatives, and provenance manifest.

    Notes
    -----
    The generator evaluates the registered upper-half-plane grid with mpmath.
    It writes a deterministic NPZ archive and its authenticated manifest.
    """
    mp.mp.dps = REFERENCE_DPS
    points: Complex128[NDArray, " n_point"] = _reference_points()
    values: Complex128[NDArray, " n_point"] = np.empty(
        points.shape, dtype=np.complex128
    )
    derivatives: Complex128[NDArray, " n_point"] = np.empty(
        points.shape, dtype=np.complex128
    )
    index: int
    point: complex
    for index, point in enumerate(points):
        argument: mp.mpc = mp.mpc(str(point.real), str(point.imag))
        value: mp.mpc = mp.exp(-(argument**2)) * mp.erfc(-1j * argument)
        derivative: mp.mpc = -2 * argument * value + 2j / mp.sqrt(mp.pi)
        values[index] = complex(value)
        derivatives[index] = complex(derivative)

    root: Path = Path(__file__).resolve().parents[2]
    data_directory: Path = root / "tests" / "test_diffpes" / "_reference_data"
    archive_path: Path = (
        data_directory / "faddeeva_mpmath_100digit_reference.npz"
    )
    manifest_path: Path = data_directory / "faddeeva_mpmath_manifest.json"
    _write_deterministic_npz(
        archive_path,
        {
            "angles": ANGLES,
            "derivatives": derivatives,
            "directions": DIRECTIONS,
            "points": points,
            "radii": RADII,
            "values": values,
        },
    )
    generator_path: Path = Path(__file__).resolve()
    manifest: Dict[str, Any] = {
        "archive": archive_path.name,
        "archive_sha256": _sha256(archive_path),
        "componentwise_value_bound": {
            "absolute": 2.0e-15,
            "relative_to_same_component": 2.0e-12,
        },
        "derivative_bound": (
            "2e-14*(1+abs(z))^-2 + 2e-11*abs(reference_derivative)"
        ),
        "directions": ["1", "1j", "(3+4j)/5"],
        "domain": {"abs_max": MAXIMUM_RADIUS, "imag_min": 0.0},
        "environment": {
            "mpmath": mp.__version__,
            "numpy": np.__version__,
            "platform": platform.platform(),
        },
        "explicit_points": [
            "2.5",
            "3+1j",
            "6",
            "1e8",
            "1e8j",
        ],
        "requirements": [
            "faddeeva-mpmath-reference",
            "spectral-broadening-gradient",
        ],
        "generator": "tests/_reference_tools/"
        "generate_faddeeva_mpmath_reference.py",
        "generator_sha256": _sha256(generator_path),
        "grid": {
            "angles": "linspace(0,pi,33)",
            "radii": "{0} union 10**linspace(-12,8,161)",
        },
        "reference_engine": {
            "decimal_digits": REFERENCE_DPS,
            "formula": "exp(-z*z)*erfc(-1j*z)",
            "name": "mpmath",
        },
        "selected_algorithm": {
            "name": "Weideman rational approximation",
            "order": 40,
            "regions": 1,
            "seam_count": 0,
            "selection_max_derivative_budget_ratio": 4.5632e-4,
            "selection_max_value_budget_ratio": 2.2956e-2,
        },
        "schema": "diffpes.faddeeva-mpmath-reference.v1",
        "seams": [],
        "authority_status": "algorithm-selected-before-production-edit",
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
