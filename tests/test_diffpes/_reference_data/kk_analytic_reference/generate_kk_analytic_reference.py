"""Generate the independent analytic Kramers--Kronig reference artifact.

The generator evaluates closed forms with mpmath and checks their signs by
direct high-precision principal-value integration. It imports no DiffPES code.
"""

from __future__ import annotations

import hashlib
import io
import json
import zipfile
from pathlib import Path

import mpmath as mp
import numpy as np
from beartype.typing import Dict, Tuple
from jaxtyping import Float
from numpy.typing import NDArray

_DECIMAL_DIGITS: int = 80
mp.mp.dps = _DECIMAL_DIGITS
_PV_CHECK_ATOL: mp.mpf = mp.mpf("1e-45")
_GRID_START_EV: mp.mpf = mp.mpf("-1.0")
_GRID_STOP_EV: mp.mpf = mp.mpf("1.0")
_GRID_COUNT: int = 1001
_SUBTRACTION_POINT_EV: mp.mpf = mp.mpf("0.0")
_POLE_OMEGA_0_EV: mp.mpf = mp.mpf("0.35")
_POLE_GAMMA_EV: mp.mpf = mp.mpf("0.20")
_POLE_G_EV2: mp.mpf = mp.mpf("0.12")
_SEMICIRCLE_BAND_EV: mp.mpf = mp.mpf("1.50")
_SEMICIRCLE_G_EV2: mp.mpf = mp.mpf("0.20")
_PV_CHECK_POINTS_EV: Tuple[str, ...] = (
    "-1.0",
    "-0.75",
    "-0.2",
    "0.0",
    "0.35",
    "0.8",
    "1.0",
)
_ZIP_TIMESTAMP: Tuple[int, int, int, int, int, int] = (1980, 1, 1, 0, 0, 0)


def _pole_parts(omega_ev: mp.mpf) -> Tuple[mp.mpf, mp.mpf]:
    """PRIVATE: Evaluate the retarded-pole real and imaginary parts.

    Parameters
    ----------
    omega_ev : mp.mpf
        Evaluation energy in eV.

    Returns
    -------
    real_ev : mp.mpf
        Real part g (omega - omega_0) / ((omega - omega_0)^2 +
        gamma^2) in eV.
    imaginary_ev : mp.mpf
        Imaginary part -g gamma / ((omega - omega_0)^2 + gamma^2) in
        eV, strictly negative.

    Notes
    -----
    Evaluates the retarded-pole closed form at the module working
    precision. The module constants give the strength g in eV^2, the
    center omega_0 in eV, and the width gamma in eV.
    """
    offset_ev: mp.mpf = omega_ev - _POLE_OMEGA_0_EV
    denominator_ev2: mp.mpf = offset_ev**2 + _POLE_GAMMA_EV**2
    real_ev: mp.mpf = _POLE_G_EV2 * offset_ev / denominator_ev2
    imaginary_ev: mp.mpf = -_POLE_G_EV2 * _POLE_GAMMA_EV / denominator_ev2
    return real_ev, imaginary_ev


def _semicircle_parts(omega_ev: mp.mpf) -> Tuple[mp.mpf, mp.mpf]:
    """PRIVATE: Evaluate the in-band Wigner-semicircle analytic pair.

    Parameters
    ----------
    omega_ev : mp.mpf
        Evaluation energy in eV, inside the band of half width W.

    Returns
    -------
    real_ev : mp.mpf
        Real part (2 g / W^2) omega in eV.
    imaginary_ev : mp.mpf
        Imaginary part -(2 g / W^2) sqrt(W^2 - omega^2) in eV,
        non-positive.

    Notes
    -----
    Evaluates the in-band closed form of the semicircle model. The
    module constants give the strength g in eV^2 and the band half
    width W in eV. The caller keeps the grid inside the band, so the
    radicand stays non-negative.
    """
    prefactor: mp.mpf = 2 * _SEMICIRCLE_G_EV2 / _SEMICIRCLE_BAND_EV**2
    real_ev: mp.mpf = prefactor * omega_ev
    radicand_ev2: mp.mpf = _SEMICIRCLE_BAND_EV**2 - omega_ev**2
    imaginary_ev: mp.mpf = -prefactor * mp.sqrt(radicand_ev2)
    return real_ev, imaginary_ev


def _pole_pv_real(omega_ev: mp.mpf) -> mp.mpf:
    """PRIVATE: Integrate the pole KK principal value numerically.

    Parameters
    ----------
    omega_ev : mp.mpf
        Query energy in eV.

    Returns
    -------
    result_ev : mp.mpf
        Kramers--Kronig real part in eV: the principal value of the
        integral of Im / (omega' - omega) over the whole real line,
        divided by pi.

    Notes
    -----
    Maps the infinite domain to (-pi/2, pi/2) with the tangent
    substitution, then subtracts the transformed numerator value at
    the singular angle. Integrates the regularized difference with
    ``mp.quad`` through the singular node and supplies the exact
    derivative limit there. Adds the closed-form principal value of
    the split-off constant term and divides the sum by pi. The
    constant term contributes -pi omega N(theta_0) / (1 + omega^2).
    """
    theta_0: mp.mpf = mp.atan(omega_ev)

    def transformed_numerator(theta: mp.mpf) -> mp.mpf:
        integration_ev: mp.mpf = mp.tan(theta)
        imaginary_ev: mp.mpf = _pole_parts(integration_ev)[1]
        return imaginary_ev / mp.cos(theta) ** 2

    numerator_0: mp.mpf = transformed_numerator(theta_0)

    def regular_integrand(theta: mp.mpf) -> mp.mpf:
        if theta == theta_0:
            derivative: mp.mpf = mp.diff(transformed_numerator, theta_0)
            return derivative / (1 + omega_ev**2)
        return (transformed_numerator(theta) - numerator_0) / (
            mp.tan(theta) - omega_ev
        )

    regular_integral_ev: mp.mpf = mp.quad(
        regular_integrand, [-mp.pi / 2, theta_0, mp.pi / 2]
    )
    singular_integral_ev: mp.mpf = (
        -mp.pi * omega_ev * numerator_0 / (1 + omega_ev**2)
    )
    result_ev: mp.mpf = (regular_integral_ev + singular_integral_ev) / mp.pi
    return result_ev


def _semicircle_pv_real(omega_ev: mp.mpf) -> mp.mpf:
    """PRIVATE: Integrate the semicircle KK principal value numerically.

    Parameters
    ----------
    omega_ev : mp.mpf
        Query energy in eV, strictly inside the band.

    Returns
    -------
    result_ev : mp.mpf
        Kramers--Kronig real part in eV: the principal value of the
        integral of Im / (omega' - omega) over the band [-W, W],
        divided by pi.

    Notes
    -----
    Subtracts the imaginary part at the query point so the remaining
    integrand is regular. Supplies the exact derivative limit at the
    query node and integrates with ``mp.quad`` through it. Adds the
    closed-form principal value of the subtracted constant, Im(omega)
    log((W - omega) / (W + omega)), before it divides by pi.
    """
    imaginary_0_ev: mp.mpf = _semicircle_parts(omega_ev)[1]
    prefactor: mp.mpf = 2 * _SEMICIRCLE_G_EV2 / _SEMICIRCLE_BAND_EV**2

    def regular_integrand(integration_ev: mp.mpf) -> mp.mpf:
        if integration_ev == omega_ev:
            return (
                prefactor
                * omega_ev
                / mp.sqrt(_SEMICIRCLE_BAND_EV**2 - omega_ev**2)
            )
        imaginary_ev: mp.mpf = _semicircle_parts(integration_ev)[1]
        return (imaginary_ev - imaginary_0_ev) / (integration_ev - omega_ev)

    regular_integral_ev: mp.mpf = mp.quad(
        regular_integrand,
        [-_SEMICIRCLE_BAND_EV, omega_ev, _SEMICIRCLE_BAND_EV],
    )
    singular_integral_ev: mp.mpf = imaginary_0_ev * mp.log(
        (_SEMICIRCLE_BAND_EV - omega_ev) / (_SEMICIRCLE_BAND_EV + omega_ev)
    )
    result_ev: mp.mpf = (regular_integral_ev + singular_integral_ev) / mp.pi
    return result_ev


def _verify_principal_values() -> Dict[str, str]:
    """PRIVATE: Verify both closed forms against independent PV quadrature.

    Returns
    -------
    rendered_errors : dict[str, str]
        Observed maximum absolute errors in eV for the pole and the
        semicircle model, rendered to 12 significant digits for the
        manifest.

    Raises
    ------
    RuntimeError
        If any maximum absolute error exceeds the module tolerance of
        1e-45 eV.

    Notes
    -----
    Walks the registered check energies and compares the quadrature
    principal value with the closed-form real part for both models.
    Tracks the per-model maxima before it renders them.
    """
    maximum_errors: Dict[str, mp.mpf] = {
        "pole_max_abs_error_ev": mp.mpf("0"),
        "semicircle_max_abs_error_ev": mp.mpf("0"),
    }
    point_text: str
    for point_text in _PV_CHECK_POINTS_EV:
        omega_ev: mp.mpf = mp.mpf(point_text)
        pole_error_ev: mp.mpf = abs(
            _pole_pv_real(omega_ev) - _pole_parts(omega_ev)[0]
        )
        semicircle_error_ev: mp.mpf = abs(
            _semicircle_pv_real(omega_ev) - _semicircle_parts(omega_ev)[0]
        )
        maximum_errors["pole_max_abs_error_ev"] = max(
            maximum_errors["pole_max_abs_error_ev"], pole_error_ev
        )
        maximum_errors["semicircle_max_abs_error_ev"] = max(
            maximum_errors["semicircle_max_abs_error_ev"],
            semicircle_error_ev,
        )
    if any(error > _PV_CHECK_ATOL for error in maximum_errors.values()):
        raise RuntimeError(
            f"principal-value verification failed: {maximum_errors}"
        )
    rendered_errors: Dict[str, str] = {
        name: mp.nstr(error, 12) for name, error in maximum_errors.items()
    }
    return rendered_errors


def _build_arrays() -> Dict[str, Float[NDArray, "..."]]:
    """PRIVATE: Build the six frozen float64 arrays from mpmath values.

    Returns
    -------
    arrays : dict[str, Float[NDArray, "..."]]
        Archive members ``pole_omega``, ``pole_sigma_real``,
        ``pole_sigma_imag`` and the three semicircle counterparts,
        each of length 1001 in eV.

    Notes
    -----
    Evaluates both closed forms on the uniform grid over [-1, 1] eV
    at working precision. Subtracts the pole real part at the 0 eV
    subtraction point, which freezes the once-subtracted convention.
    Casts every value to float64 only at the end.
    """
    step_ev: mp.mpf = (_GRID_STOP_EV - _GRID_START_EV) / (_GRID_COUNT - 1)
    omega_mp: list[mp.mpf] = [
        _GRID_START_EV + index * step_ev for index in range(_GRID_COUNT)
    ]
    pole_at_subtraction_ev: mp.mpf = _pole_parts(_SUBTRACTION_POINT_EV)[0]
    pole_real: list[float] = []
    pole_imaginary: list[float] = []
    semicircle_real: list[float] = []
    semicircle_imaginary: list[float] = []
    omega_ev: mp.mpf
    pole_real_ev: mp.mpf
    pole_imaginary_ev: mp.mpf
    semicircle_real_ev: mp.mpf
    semicircle_imaginary_ev: mp.mpf
    for omega_ev in omega_mp:
        pole_real_ev, pole_imaginary_ev = _pole_parts(omega_ev)
        semicircle_real_ev, semicircle_imaginary_ev = _semicircle_parts(
            omega_ev
        )
        pole_real.append(float(pole_real_ev - pole_at_subtraction_ev))
        pole_imaginary.append(float(pole_imaginary_ev))
        semicircle_real.append(float(semicircle_real_ev))
        semicircle_imaginary.append(float(semicircle_imaginary_ev))
    omega: Float[NDArray, "..."] = np.asarray(omega_mp, dtype=np.float64)
    arrays: Dict[str, Float[NDArray, "..."]] = {
        "pole_omega": omega,
        "pole_sigma_real": np.asarray(pole_real, dtype=np.float64),
        "pole_sigma_imag": np.asarray(pole_imaginary, dtype=np.float64),
        "semicircle_omega": omega.copy(),
        "semicircle_sigma_real": np.asarray(semicircle_real, dtype=np.float64),
        "semicircle_sigma_imag": np.asarray(
            semicircle_imaginary, dtype=np.float64
        ),
    }
    return arrays


def _write_deterministic_npz(
    archive_path: Path, arrays: Dict[str, Float[NDArray, "..."]]
) -> None:
    """PRIVATE: Write arrays in a deterministic uncompressed NumPy archive.

    Parameters
    ----------
    archive_path : Path
        Destination path of the ``.npz`` archive.
    arrays : dict[str, Float[NDArray, "..."]]
        Named arrays to store, one ``.npy`` member each.

    Notes
    -----
    Serializes each array with ``np.lib.format.write_array`` and no
    pickle support. Stores each member with the fixed 1980 zip
    timestamp, the ``ZIP_STORED`` method, and 0o600 permissions, so
    the archive bytes stay identical across runs.
    """
    archive: zipfile.ZipFile
    name: str
    array: Float[NDArray, "..."]
    with zipfile.ZipFile(archive_path, "w", allowZip64=True) as archive:
        for name, array in arrays.items():
            buffer: io.BytesIO = io.BytesIO()
            np.lib.format.write_array(buffer, array, allow_pickle=False)
            member: zipfile.ZipInfo = zipfile.ZipInfo(
                f"{name}.npy", date_time=_ZIP_TIMESTAMP
            )
            member.compress_type = zipfile.ZIP_STORED
            member.external_attr = 0o600 << 16
            archive.writestr(member, buffer.getvalue())


def main() -> None:
    """Generate the archive and its provenance manifest."""
    mp.mp.dps = _DECIMAL_DIGITS
    pv_errors: Dict[str, str] = _verify_principal_values()
    arrays: Dict[str, Float[NDArray, "..."]] = _build_arrays()
    output_directory: Path = Path(__file__).resolve().parent
    archive_path: Path = output_directory / "kk_reference.npz"
    manifest_path: Path = output_directory / "manifest.json"
    _write_deterministic_npz(archive_path, arrays)
    archive_sha256: str = hashlib.sha256(archive_path.read_bytes()).hexdigest()
    manifest: Dict[str, object] = {
        "schema": "diffpes.kk-analytic-reference.v1",
        "archive": archive_path.name,
        "archive_sha256": archive_sha256,
        "generator": Path(__file__).name,
        "arbiter": {
            "name": "mpmath",
            "decimal_digits": _DECIMAL_DIGITS,
            "method": "closed forms checked by numerical principal-value integration",
            "pv_check_absolute_tolerance_ev": str(_PV_CHECK_ATOL),
            "pv_check_points_ev": list(_PV_CHECK_POINTS_EV),
            "observed_maximum_errors": pv_errors,
        },
        "grid": {
            "start_ev": float(_GRID_START_EV),
            "stop_ev": float(_GRID_STOP_EV),
            "count": _GRID_COUNT,
            "subtraction_point_ev": float(_SUBTRACTION_POINT_EV),
        },
        "fixtures": {
            "pole": {
                "omega_0_ev": float(_POLE_OMEGA_0_EV),
                "gamma_ev": float(_POLE_GAMMA_EV),
                "g_ev2": float(_POLE_G_EV2),
            },
            "semicircle": {
                "band_half_width_ev": float(_SEMICIRCLE_BAND_EV),
                "g_ev2": float(_SEMICIRCLE_G_EV2),
            },
        },
        "arrays": {name: list(array.shape) for name, array in arrays.items()},
        "dtype": "float64",
        "production_imports": [],
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
