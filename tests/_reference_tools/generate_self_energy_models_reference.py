"""Generate frozen Fermi-liquid and bosonic-kink self-energy truth artifacts.

The generator uses only NumPy and SciPy for numerical truth.  It does not
import DiffPES or evaluate any production self-energy code.  It freezes the
Fermi-liquid and bosonic-kink fixture parameters (physical eV values and raw
softplus-inverse coordinates), evaluation and KK grids, the independently
derived causal real parts, asymptotic first-moment identities, and
complex-step parameter-derivative truths.  These tables back the
causal-self-energy-analytic-truth and self-energy-parameter-derivative-truth
requirements.

Truth derivations are doubly independent: the Fermi-liquid subtracted real
part is computed by adaptive principal-value quadrature (QAWC core plus QAGI
infinite tails of the dynamic remainder after subtracting ``Gamma0``) and is
certified against the exact residue closed form; the bosonic-kink pole pair
is analytic, and its Kramers-Kronig self-consistency is spot-checked by the
same principal-value quadrature.  The retarded convention throughout is
``Sigma'(w) = (1/pi) PV integral Sigma''_dyn(x)/(x-w) dx`` with
``Sigma'' <= 0``.
"""

from __future__ import annotations

import hashlib
import io
import json
import platform
import warnings
import zipfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import scipy
from beartype.typing import Dict, Tuple
from jaxtyping import Bool, Complex128, Float64, Shaped
from numpy.typing import NDArray
from scipy import integrate

# Frozen Fermi-liquid fixture parameters (physical, eV-pinned units).
FL_GAMMA0_EV: float = 0.02
FL_BETA_PER_EV: float = 0.5
FL_OMEGA_C_EV: float = 0.8
# Frozen bosonic-kink fixture parameters (physical, eV-pinned units).
KINK_GAMMA0_EV: float = 0.015
KINK_COUPLING_EV: float = 0.35
KINK_OMEGA0_EV: float = 0.07
KINK_WIDTH_EV: float = 0.02

EVAL_GRID_POINTS: int = 2001
EVAL_GRID_HALF_WIDTH_EV: float = 1.5
KK_DOMAIN_EV: Tuple[float, float] = (-8.0, 8.0)
RAW_COORDINATE_BOUND: float = 30.0
SOFTPLUS_ROUNDTRIP_RTOL: float = 1.0e-14

QUAD_EPSABS: float = 1.0e-13
QUAD_EPSREL: float = 1.0e-12
QUAD_LIMIT: int = 800
KINK_CAUCHY_HALF_WIDTH_EV: float = 0.5
FL_QUAD_ABS_BUDGET: float = 1.0e-12
KINK_SPOT_ABS_BUDGET: float = 1.0e-10

COMPLEX_STEP_RELATIVE: float = 1.0e-30
DERIVATIVE_CROSSCHECK_ATOL: float = 1.0e-10
DERIVATIVE_CROSSCHECK_RTOL: float = 1.0e-10
DERIVATIVE_NONZERO_FLOOR: float = 1.0e-12
DERIVATIVE_COLUMN_FLOOR: float = 1.0e-3

# Eleven probe frequencies straddling the Omega_c = 0.8 eV crossover on
# both sides; none coincides with 0 or +-Omega_c where symmetry zeroes a
# beta-sensitivity row.
FL_PROBE_FREQUENCIES_EV: Float64[NDArray, " n_fl_probe"] = np.asarray(
    [
        -1.35,
        -1.05,
        -0.85,
        -0.55,
        -0.25,
        0.05,
        0.35,
        0.65,
        0.95,
        1.15,
        1.35,
    ],
    dtype=np.float64,
)
# Eleven probe frequencies straddling the +-Omega0 = +-0.07 eV kink on
# both sides; none coincides with 0 or the analytic Sigma' zero at
# +-sqrt(Omega0**2 - width**2) ~= +-0.067082 eV.
KINK_PROBE_FREQUENCIES_EV: Float64[NDArray, " n_kink_probe"] = np.asarray(
    [
        -0.21,
        -0.13,
        -0.09,
        -0.075,
        -0.05,
        -0.015,
        0.025,
        0.06,
        0.08,
        0.12,
        0.2,
    ],
    dtype=np.float64,
)
# Twenty-one spot-check frequencies for pole-pair KK self-consistency.
KINK_SPOT_FREQUENCIES_EV: Float64[NDArray, " n_spot"] = np.asarray(
    [
        -0.3,
        -0.2,
        -0.14,
        -0.1,
        -0.085,
        -0.07,
        -0.055,
        -0.04,
        -0.02,
        -0.005,
        0.0,
        0.005,
        0.02,
        0.04,
        0.055,
        0.07,
        0.085,
        0.1,
        0.14,
        0.2,
        0.3,
    ],
    dtype=np.float64,
)
KINK_MOMENT_FREQUENCIES_EV: Float64[NDArray, " n_moment"] = np.asarray(
    [25.0, 50.0, 100.0, 200.0, 400.0],
    dtype=np.float64,
)
MOMENT_RATIO_BOUNDS: Tuple[float, float] = (3.9, 4.1)


def _array_bytes(array: Shaped[NDArray, "..."]) -> bytes:
    """PRIVATE: Serialize one array without pickle or filesystem metadata.

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
    return output.getvalue()


def _write_deterministic_npz(
    path: Path,
    arrays: Dict[str, Shaped[NDArray, "..."]],
) -> None:
    """PRIVATE: Write a sorted float64 NPZ with fixed dates and modes.

    Parameters
    ----------
    path : Path
        Destination NPZ path.
    arrays : dict[str, Shaped[NDArray, "..."]]
        Named arrays to store; each one casts to float64.

    Notes
    -----
    Members enter in sorted name order with the fixed 1980-01-01 ZIP
    timestamp, a fixed file mode, and DEFLATE level 9, so identical
    arrays always produce byte-identical archives.
    """
    with zipfile.ZipFile(
        path,
        mode="w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=9,
    ) as archive:
        name: str
        array: Shaped[NDArray, "..."]
        for name, array in sorted(arrays.items()):
            normalized: Float64[NDArray, "..."] = np.asarray(
                array, dtype=np.float64
            )
            member: zipfile.ZipInfo = zipfile.ZipInfo(
                filename=f"{name}.npy",
                date_time=(1980, 1, 1, 0, 0, 0),
            )
            member.compress_type = zipfile.ZIP_DEFLATED
            member.external_attr = 0o100644 << 16
            archive.writestr(
                member,
                _array_bytes(normalized),
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
    digest : str
        Lowercase hexadecimal SHA-256 of the file bytes.

    Notes
    -----
    The function reads the complete file into memory before hashing;
    every evidence file stays small enough for that.
    """
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _softplus(
    raw: Float64[NDArray, " n_param"],
) -> Float64[NDArray, " n_param"]:
    """PRIVATE: Evaluate the smooth positive map ``log1p(exp(raw))``.

    Parameters
    ----------
    raw : Float64[NDArray, " n_param"]
        Unconstrained raw parameter coordinates.

    Returns
    -------
    positive : Float64[NDArray, " n_param"]
        Softplus values, strictly positive.

    Notes
    -----
    The overflow-safe form ``log1p(exp(-|raw|)) + max(raw, 0)`` equals
    ``log(1 + exp(raw))`` for every finite input.
    """
    return np.log1p(np.exp(-np.abs(raw))) + np.maximum(raw, 0.0)


def _softplus_inverse(
    positive: Float64[NDArray, " n_param"],
) -> Float64[NDArray, " n_param"]:
    """PRIVATE: Invert softplus with the stable ``p + log(-expm1(-p))``.

    Parameters
    ----------
    positive : Float64[NDArray, " n_param"]
        Strictly positive physical parameter values.

    Returns
    -------
    raw : Float64[NDArray, " n_param"]
        Raw coordinates with ``softplus(raw) == positive``.

    Notes
    -----
    The ``expm1`` form keeps full precision for small inputs, where the
    naive ``log(exp(p) - 1)`` cancels catastrophically.
    """
    return positive + np.log(-np.expm1(-positive))


def _raw_coordinates(
    physical: Float64[NDArray, " n_param"],
) -> Float64[NDArray, " n_param"]:
    """PRIVATE: Convert physical parameters to bounded raw coordinates.

    Parameters
    ----------
    physical : Float64[NDArray, " n_param"]
        Strictly positive physical parameter values in eV units.

    Returns
    -------
    raw : Float64[NDArray, " n_param"]
        Softplus-inverse coordinates inside ``[-30, 30]``.

    Raises
    ------
    RuntimeError
        If a raw coordinate leaves the closed ``[-30, 30]`` bound, or
        if the softplus round trip leaves its relative budget.

    Notes
    -----
    The function inverts softplus, checks the identifiability bound,
    and certifies the round trip ``softplus(raw)`` against the inputs
    at relative tolerance ``1e-14``.
    """
    raw: Float64[NDArray, " n_param"] = _softplus_inverse(physical)
    if np.any(np.abs(raw) > RAW_COORDINATE_BOUND):
        raise RuntimeError("raw coordinate leaves the closed [-30, 30] bound")
    roundtrip: Float64[NDArray, " n_param"] = _softplus(raw)
    if np.any(
        np.abs(roundtrip - physical) > SOFTPLUS_ROUNDTRIP_RTOL * physical
    ):
        raise RuntimeError("softplus round trip leaves its relative budget")
    return raw


def _evaluation_grid() -> Float64[NDArray, " n_grid"]:
    """PRIVATE: Return the frozen shared evaluation grid.

    Returns
    -------
    grid : Float64[NDArray, " n_grid"]
        2001 uniform points on ``[-1.5, 1.5]`` eV of the relative
        ``E - E_F`` axis.

    Notes
    -----
    Both fixtures share this grid, so every truth table aligns row by
    row.
    """
    return np.linspace(
        -EVAL_GRID_HALF_WIDTH_EV,
        EVAL_GRID_HALF_WIDTH_EV,
        EVAL_GRID_POINTS,
        dtype=np.float64,
    )


def _fl_sigma_imag(
    omega: Float64[NDArray, " n_omega"] | Complex128[NDArray, " n_omega"],
    gamma0: complex,
    beta: complex,
    omega_c: complex,
) -> Float64[NDArray, " n_omega"] | Complex128[NDArray, " n_omega"]:
    """PRIVATE: Evaluate the Fermi-liquid ``Sigma''`` analytically.

    Parameters
    ----------
    omega : Float64[NDArray, " n_omega"] | Complex128[NDArray, " n_omega"]
        Frequencies in eV; complex inputs serve the complex-step rows.
    gamma0 : complex
        Constant baseline scattering rate in eV.
    beta : complex
        Quadratic coefficient in 1/eV.
    omega_c : complex
        Crossover scale in eV.

    Returns
    -------
    sigma_imag : Float64[NDArray, " n_omega"] | Complex128[NDArray, " n_omega"]
        ``-Gamma0 - beta w^2 / (1 + (w/Omega_c)^4)`` in eV.

    Notes
    -----
    The quartic saturation caps the quadratic Fermi-liquid growth above
    the crossover, which keeps the KK transform convergent.
    """
    return -gamma0 - beta * omega**2 / (1.0 + (omega / omega_c) ** 4)


def _fl_sigma_real(
    omega: Float64[NDArray, " n_omega"] | Complex128[NDArray, " n_omega"],
    beta: complex,
    omega_c: complex,
) -> Float64[NDArray, " n_omega"] | Complex128[NDArray, " n_omega"]:
    """PRIVATE: Evaluate the exact residue form of the Fermi-liquid Sigma'.

    Parameters
    ----------
    omega : Float64[NDArray, " n_omega"] | Complex128[NDArray, " n_omega"]
        Frequencies in eV; complex inputs serve the complex-step rows.
    beta : complex
        Quadratic coefficient in 1/eV.
    omega_c : complex
        Crossover scale in eV.

    Returns
    -------
    sigma_real : Float64[NDArray, " n_omega"] | Complex128[NDArray, " n_omega"]
        Subtracted real part ``Sigma'(w) - Sigma'(0)`` in eV.

    Notes
    -----
    The dynamic remainder ``R(x) = -beta*x**2/(1+(x/Omega_c)**4)`` has
    upper-half-plane poles at ``Omega_c*exp(i*pi/4)`` and
    ``Omega_c*exp(3i*pi/4)``; contour integration of the retarded KK
    transform collapses to this real rational function.
    """
    return (
        (np.sqrt(2.0) / 2.0)
        * beta
        * omega_c**3
        * omega
        * (omega**2 - omega_c**2)
        / (omega**4 + omega_c**4)
    )


def _fl_dynamic_imag(x: float) -> float:
    """PRIVATE: Evaluate the Fermi-liquid dynamic remainder.

    Parameters
    ----------
    x : float
        Frequency in eV.

    Returns
    -------
    remainder : float
        ``Sigma''(x) + Gamma0`` in eV at the frozen fixture parameters.

    Notes
    -----
    The constant ``-Gamma0`` baseline drops out before the KK
    transform; this scalar form feeds the adaptive quadrature.
    """
    return -FL_BETA_PER_EV * x**2 / (1.0 + (x / FL_OMEGA_C_EV) ** 4)


def _kink_sigma_real(
    omega: Float64[NDArray, " n_omega"] | Complex128[NDArray, " n_omega"],
    coupling: complex,
    omega0: complex,
    width: complex,
) -> Float64[NDArray, " n_omega"] | Complex128[NDArray, " n_omega"]:
    """PRIVATE: Evaluate the exact bosonic-kink pole-pair real part.

    Parameters
    ----------
    omega : Float64[NDArray, " n_omega"] | Complex128[NDArray, " n_omega"]
        Frequencies in eV; complex inputs serve the complex-step rows.
    coupling : complex
        Mode coupling ``g`` in eV.
    omega0 : complex
        Boson energy in eV.
    width : complex
        Pole broadening in eV.

    Returns
    -------
    sigma_real : Float64[NDArray, " n_omega"] | Complex128[NDArray, " n_omega"]
        ``g^2 [(w-Omega0)/d_- + (w+Omega0)/d_+]`` in eV with
        ``d_+- = (w -+ Omega0)^2 + width^2``.

    Notes
    -----
    The pair of retarded poles at ``+-Omega0 - i width`` makes this
    real part exactly KK-consistent with ``_kink_sigma_imag``.
    """
    d_minus: Float64[NDArray, " n_omega"] | Complex128[NDArray, " n_omega"] = (
        omega - omega0
    ) ** 2 + width**2
    d_plus: Float64[NDArray, " n_omega"] | Complex128[NDArray, " n_omega"] = (
        omega + omega0
    ) ** 2 + width**2
    return coupling**2 * (
        (omega - omega0) / d_minus + (omega + omega0) / d_plus
    )


def _kink_sigma_imag(
    omega: Float64[NDArray, " n_omega"] | Complex128[NDArray, " n_omega"],
    gamma0: complex,
    coupling: complex,
    omega0: complex,
    width: complex,
) -> Float64[NDArray, " n_omega"] | Complex128[NDArray, " n_omega"]:
    """PRIVATE: Evaluate the exact bosonic-kink imaginary part.

    Parameters
    ----------
    omega : Float64[NDArray, " n_omega"] | Complex128[NDArray, " n_omega"]
        Frequencies in eV; complex inputs serve the complex-step rows.
    gamma0 : complex
        Constant baseline scattering rate in eV.
    coupling : complex
        Mode coupling ``g`` in eV.
    omega0 : complex
        Boson energy in eV.
    width : complex
        Pole broadening in eV.

    Returns
    -------
    sigma_imag : Float64[NDArray, " n_omega"] | Complex128[NDArray, " n_omega"]
        ``-Gamma0 - g^2 width (1/d_- + 1/d_+)`` in eV with
        ``d_+- = (w -+ Omega0)^2 + width^2``.

    Notes
    -----
    The two Lorentzians center on ``+-Omega0``; the constant baseline
    keeps the whole grid strictly negative.
    """
    d_minus: Float64[NDArray, " n_omega"] | Complex128[NDArray, " n_omega"] = (
        omega - omega0
    ) ** 2 + width**2
    d_plus: Float64[NDArray, " n_omega"] | Complex128[NDArray, " n_omega"] = (
        omega + omega0
    ) ** 2 + width**2
    return -gamma0 - coupling**2 * width * (1.0 / d_minus + 1.0 / d_plus)


def _kink_dynamic_imag(x: float) -> float:
    """PRIVATE: Evaluate the kink pole-pair imaginary part without Gamma0.

    Parameters
    ----------
    x : float
        Frequency in eV.

    Returns
    -------
    remainder : float
        Dynamic ``Sigma''`` in eV at the frozen fixture parameters.

    Notes
    -----
    This scalar form feeds the adaptive quadrature for the KK
    spot checks; the constant baseline drops out beforehand.
    """
    d_minus: float = (x - KINK_OMEGA0_EV) ** 2 + KINK_WIDTH_EV**2
    d_plus: float = (x + KINK_OMEGA0_EV) ** 2 + KINK_WIDTH_EV**2
    return (
        -(KINK_COUPLING_EV**2) * KINK_WIDTH_EV * (1.0 / d_minus + 1.0 / d_plus)
    )


def _pv_real_part(
    dynamic_imag: Callable[[float], float],
    omega: float,
    cauchy_half_width: float | None,
) -> Tuple[float, float]:
    """PRIVATE: Evaluate the retarded KK real part by adaptive quadrature.

    Parameters
    ----------
    dynamic_imag : Callable[[float], float]
        Dynamic ``Sigma''`` remainder in eV as a scalar function.
    omega : float
        Query frequency in eV.
    cauchy_half_width : float | None
        Half-width in eV of the Cauchy-weighted core segment; ``None``
        spans the whole finite KK domain with the Cauchy rule.

    Returns
    -------
    result : tuple[float, float]
        The KK real part in eV and the summed quadrature error
        estimate in eV.

    Implementation Logic
    --------------------
    The principal-value core uses QAWC with the Cauchy weight around
    the query, optional smooth remainder segments cover the rest of the
    finite KK domain, and QAGI integrates the true infinite tails of
    the dynamic remainder.  Certification happens later by hard
    comparison against analytic values, so QUADPACK convergence
    warnings stay suppressed here.
    """

    def _shifted(x: float) -> float:
        """PRIVATE: Evaluate the smooth shifted integrand.

        Parameters
        ----------
        x : float
            Integration variable in eV.

        Returns
        -------
        value : float
            ``dynamic_imag(x) / (x - omega)`` in eV per eV.

        Notes
        -----
        This form serves the smooth segments away from the query, where
        the denominator never vanishes.
        """
        return dynamic_imag(x) / (x - omega)

    lower_edge: float
    upper_edge: float
    lower_edge, upper_edge = KK_DOMAIN_EV
    smooth_segments: list[Tuple[float, float]]
    if cauchy_half_width is None:
        cauchy_segment: Tuple[float, float] = (lower_edge, upper_edge)
        smooth_segments = []
    else:
        cauchy_segment = (
            omega - cauchy_half_width,
            omega + cauchy_half_width,
        )
        smooth_segments = [
            (lower_edge, cauchy_segment[0]),
            (cauchy_segment[1], upper_edge),
        ]
    smooth_segments.extend(
        ((upper_edge, np.inf), (-np.inf, lower_edge)),
    )
    total: float = 0.0
    error: float = 0.0
    with warnings.catch_warnings():
        # Certification is by hard comparison against exact analytic
        # values below, not by the QUADPACK convergence heuristics.
        warnings.simplefilter("ignore", integrate.IntegrationWarning)
        value: float
        estimate: float
        value, estimate = integrate.quad(
            dynamic_imag,
            cauchy_segment[0],
            cauchy_segment[1],
            weight="cauchy",
            wvar=omega,
            epsabs=QUAD_EPSABS,
            epsrel=QUAD_EPSREL,
            limit=QUAD_LIMIT,
        )
        total += value
        error += estimate
        segment: Tuple[float, float]
        for segment in smooth_segments:
            value, estimate = integrate.quad(
                _shifted,
                segment[0],
                segment[1],
                epsabs=QUAD_EPSABS,
                epsrel=QUAD_EPSREL,
                limit=QUAD_LIMIT,
            )
            total += value
            error += estimate
    return total / np.pi, error / np.pi


def _parameters_payload() -> Dict[str, Float64[NDArray, "..."]]:
    """PRIVATE: Record physical parameters, raw coordinates, and grids.

    Returns
    -------
    payload : dict[str, Float64[NDArray, "..."]]
        Evaluation grid, KK domain, both physical parameter vectors,
        their raw coordinates, and the softplus Jacobian rows.

    Notes
    -----
    The Jacobian rows store ``d softplus/d raw = -expm1(-p)`` evaluated
    at each physical value, the chain factor between raw and physical
    sensitivities.
    """
    fl_physical: Float64[NDArray, " n_fl_param"] = np.asarray(
        [FL_GAMMA0_EV, FL_BETA_PER_EV, FL_OMEGA_C_EV],
        dtype=np.float64,
    )
    kink_physical: Float64[NDArray, " n_kink_param"] = np.asarray(
        [KINK_GAMMA0_EV, KINK_COUPLING_EV, KINK_OMEGA0_EV, KINK_WIDTH_EV],
        dtype=np.float64,
    )
    fl_raw: Float64[NDArray, " n_fl_param"] = _raw_coordinates(fl_physical)
    kink_raw: Float64[NDArray, " n_kink_param"] = _raw_coordinates(
        kink_physical
    )
    return {
        "eval_grid_ev": _evaluation_grid(),
        "fl_kk_domain_ev": np.asarray(KK_DOMAIN_EV, dtype=np.float64),
        "fl_parameters_physical": fl_physical,
        "fl_parameters_raw": fl_raw,
        "fl_softplus_jacobian": -np.expm1(-fl_physical),
        "kink_parameters_physical": kink_physical,
        "kink_parameters_raw": kink_raw,
        "kink_softplus_jacobian": -np.expm1(-kink_physical),
    }


def _fermi_liquid_payload() -> Dict[str, Float64[NDArray, "..."]]:
    """PRIVATE: Build the Fermi-liquid grid, quadrature, and causality rows.

    Returns
    -------
    payload : dict[str, Float64[NDArray, "..."]]
        Grid ``Sigma''``, quadrature and analytic subtracted real
        parts, quadrature error estimates, the causality ceiling, and
        the achieved truth deviation, all in eV.

    Raises
    ------
    RuntimeError
        If ``Sigma''`` loses strict negativity anywhere on the grid, or
        if the quadrature leaves the 1e-12 truth budget against the
        residue closed form.

    Implementation Logic
    --------------------
    The routine integrates the dynamic remainder at every grid point
    and at zero, subtracts the zero value, and certifies the result
    against the exact residue closed form before it freezes both rows.
    """
    grid: Float64[NDArray, " n_grid"] = _evaluation_grid()
    sigma_imag: Float64[NDArray, " n_grid"] = _fl_sigma_imag(
        grid,
        FL_GAMMA0_EV,
        FL_BETA_PER_EV,
        FL_OMEGA_C_EV,
    )
    if not np.all(sigma_imag < 0.0):
        raise RuntimeError("Fermi-liquid Sigma'' is not strictly negative")
    real_quad: Float64[NDArray, " n_grid"] = np.empty_like(grid)
    quad_errors: Float64[NDArray, " n_grid"] = np.empty_like(grid)
    index: int
    omega: float
    for index, omega in enumerate(grid):
        real_quad[index], quad_errors[index] = _pv_real_part(
            _fl_dynamic_imag,
            float(omega),
            None,
        )
    real_at_zero: float
    zero_error: float
    real_at_zero, zero_error = _pv_real_part(_fl_dynamic_imag, 0.0, None)
    subtracted_quad: Float64[NDArray, " n_grid"] = real_quad - real_at_zero
    subtracted_analytic: Float64[NDArray, " n_grid"] = _fl_sigma_real(
        grid,
        FL_BETA_PER_EV,
        FL_OMEGA_C_EV,
    )
    quad_vs_analytic: float = float(
        np.max(np.abs(subtracted_quad - subtracted_analytic))
    )
    if quad_vs_analytic > FL_QUAD_ABS_BUDGET:
        raise RuntimeError(
            "Fermi-liquid quadrature leaves the 1e-12 truth budget"
        )
    return {
        "fl_causality_ceiling": np.asarray(
            np.max(sigma_imag),
            dtype=np.float64,
        ),
        "fl_quad_error_estimates": quad_errors,
        "fl_quad_vs_analytic_max_abs_error": np.asarray(
            quad_vs_analytic,
            dtype=np.float64,
        ),
        "fl_sigma_imag_grid": sigma_imag,
        "fl_sigma_real_at_zero_quad": np.asarray(
            real_at_zero,
            dtype=np.float64,
        ),
        "fl_sigma_real_at_zero_quad_error": np.asarray(
            zero_error,
            dtype=np.float64,
        ),
        "fl_sigma_real_subtracted_analytic": subtracted_analytic,
        "fl_sigma_real_subtracted_quad": subtracted_quad,
    }


def _bosonic_kink_payload() -> Dict[str, Float64[NDArray, "..."]]:
    """PRIVATE: Build the exact bosonic-kink grid values and causality row.

    Returns
    -------
    payload : dict[str, Float64[NDArray, "..."]]
        Analytic ``Sigma'`` and ``Sigma''`` on the shared grid and the
        causality ceiling, all in eV.

    Raises
    ------
    RuntimeError
        If ``Sigma''`` loses strict negativity anywhere on the grid.

    Notes
    -----
    Both rows come from the closed-form pole pair; no quadrature enters
    this payload.
    """
    grid: Float64[NDArray, " n_grid"] = _evaluation_grid()
    sigma_real: Float64[NDArray, " n_grid"] = _kink_sigma_real(
        grid,
        KINK_COUPLING_EV,
        KINK_OMEGA0_EV,
        KINK_WIDTH_EV,
    )
    sigma_imag: Float64[NDArray, " n_grid"] = _kink_sigma_imag(
        grid,
        KINK_GAMMA0_EV,
        KINK_COUPLING_EV,
        KINK_OMEGA0_EV,
        KINK_WIDTH_EV,
    )
    if not np.all(sigma_imag < 0.0):
        raise RuntimeError("bosonic-kink Sigma'' is not strictly negative")
    return {
        "kink_causality_ceiling": np.asarray(
            np.max(sigma_imag),
            dtype=np.float64,
        ),
        "kink_sigma_imag_grid": sigma_imag,
        "kink_sigma_real_grid": sigma_real,
    }


def _bosonic_kink_kk_payload() -> Dict[str, Float64[NDArray, "..."]]:
    """PRIVATE: Check pole-pair KK self-consistency by PV quadrature.

    Returns
    -------
    payload : dict[str, Float64[NDArray, "..."]]
        Spot frequencies in eV, exact and quadrature real parts in eV,
        absolute residuals, and quadrature error estimates.

    Raises
    ------
    RuntimeError
        If the worst spot residual leaves the 1e-10 budget.

    Notes
    -----
    Twenty-one spot frequencies straddle both kinks; the quadrature
    uses a 0.5 eV Cauchy core around each query.
    """
    exact: Float64[NDArray, " n_spot"] = _kink_sigma_real(
        KINK_SPOT_FREQUENCIES_EV,
        KINK_COUPLING_EV,
        KINK_OMEGA0_EV,
        KINK_WIDTH_EV,
    )
    quad_values: Float64[NDArray, " n_spot"] = np.empty_like(exact)
    quad_errors: Float64[NDArray, " n_spot"] = np.empty_like(exact)
    index: int
    omega: float
    for index, omega in enumerate(KINK_SPOT_FREQUENCIES_EV):
        quad_values[index], quad_errors[index] = _pv_real_part(
            _kink_dynamic_imag,
            float(omega),
            KINK_CAUCHY_HALF_WIDTH_EV,
        )
    residuals: Float64[NDArray, " n_spot"] = np.abs(quad_values - exact)
    if np.max(residuals) > KINK_SPOT_ABS_BUDGET:
        raise RuntimeError(
            "bosonic-kink KK spot check leaves the 1e-10 budget"
        )
    return {
        "kink_spot_abs_residuals": residuals,
        "kink_spot_frequencies_ev": KINK_SPOT_FREQUENCIES_EV,
        "kink_spot_quad_error_estimates": quad_errors,
        "kink_spot_real_exact": exact,
        "kink_spot_real_quad": quad_values,
    }


def _bosonic_kink_moment_payload() -> Dict[str, Float64[NDArray, "..."]]:
    """PRIVATE: Record the first-moment asymptotic identity of the pair.

    Returns
    -------
    payload : dict[str, Float64[NDArray, "..."]]
        Moment frequencies in eV, products ``w * Sigma'(w)`` in eV^2,
        the ``2 g^2`` limit, deviations, and consecutive ratios.

    Raises
    ------
    RuntimeError
        If the deviations stop decreasing, or if a consecutive ratio
        leaves the ``[3.9, 4.1]`` band expected for ``1/w^2`` decay.

    Notes
    -----
    Frequency doubling from 25 to 400 eV multiplies a ``1/w^2`` tail
    deviation by about four; the ratio band certifies that rate.
    """
    products: Float64[NDArray, " n_moment"] = (
        KINK_MOMENT_FREQUENCIES_EV
        * _kink_sigma_real(
            KINK_MOMENT_FREQUENCIES_EV,
            KINK_COUPLING_EV,
            KINK_OMEGA0_EV,
            KINK_WIDTH_EV,
        )
    )
    limit: float = 2.0 * KINK_COUPLING_EV**2
    deviations: Float64[NDArray, " n_moment"] = np.abs(products - limit)
    if not np.all(np.diff(deviations) < 0.0):
        raise RuntimeError("first-moment deviations are not decreasing")
    ratios: Float64[NDArray, " n_moment_ratio"] = (
        deviations[:-1] / deviations[1:]
    )
    if np.any(
        (ratios < MOMENT_RATIO_BOUNDS[0]) | (ratios > MOMENT_RATIO_BOUNDS[1])
    ):
        raise RuntimeError(
            "first-moment convergence is not the expected 1/omega**2"
        )
    return {
        "kink_first_moment_deviations": deviations,
        "kink_first_moment_frequencies_ev": KINK_MOMENT_FREQUENCIES_EV,
        "kink_first_moment_limit": np.asarray(limit, dtype=np.float64),
        "kink_first_moment_products": products,
        "kink_first_moment_ratios": ratios,
    }


def _complex_step_column(
    function: Callable[..., Complex128[NDArray, " n_probe"]],
    frequencies: Float64[NDArray, " n_probe"],
    parameters: Float64[NDArray, " n_param"],
    index: int,
) -> Float64[NDArray, " n_probe"]:
    """PRIVATE: Differentiate one response for one physical parameter.

    Parameters
    ----------
    function : Callable[..., Complex128[NDArray, " n_probe"]]
        Real-form response accepting complex frequencies and complex
        parameters in positional order.
    frequencies : Float64[NDArray, " n_probe"]
        Probe frequencies in eV.
    parameters : Float64[NDArray, " n_param"]
        Physical parameter values in eV units.
    index : int
        Position of the differentiated parameter.

    Returns
    -------
    column : Float64[NDArray, " n_probe"]
        Physical derivative of the response at every probe.

    Notes
    -----
    The complex-step method evaluates the function at ``p + i h`` with
    relative step ``h = 1e-30 p`` and returns ``Im(f)/h``, which is
    subtraction-free and exact to machine precision.
    """
    step: float = COMPLEX_STEP_RELATIVE * float(parameters[index])
    perturbed: list[complex] = [complex(value) for value in parameters]
    perturbed[index] = complex(parameters[index], step)
    values: Complex128[NDArray, " n_probe"] = function(
        frequencies.astype(np.complex128),
        *perturbed,
    )
    return np.imag(values) / step


def _check_derivative_matrix(
    step_matrix: Float64[NDArray, "n_probe n_param"],
    analytic_matrix: Float64[NDArray, "n_probe n_param"],
    structural_zero_columns: Tuple[int, ...],
) -> float:
    """PRIVATE: Certify one dimensionless derivative truth matrix.

    Parameters
    ----------
    step_matrix : Float64[NDArray, "n_probe n_param"]
        Complex-step derivative matrix in dimensionless coordinates.
    analytic_matrix : Float64[NDArray, "n_probe n_param"]
        Hand-derived analytic matrix in the same coordinates.
    structural_zero_columns : tuple[int, ...]
        Columns whose entries must vanish exactly.

    Returns
    -------
    max_deviation : float
        Worst absolute step-versus-analytic deviation.

    Raises
    ------
    RuntimeError
        If the matrices disagree beyond the mixed tolerance, if a
        structural-zero column carries a nonzero entry, if a sensitive
        entry underflows the 1e-12 floor, or if a sensitive column
        lacks one probe above the 1e-3 floor.

    Notes
    -----
    The floors guarantee every recorded sensitivity row stays usable
    as a nonzero regression witness.
    """
    deviation: Float64[NDArray, "n_probe n_param"] = np.abs(
        step_matrix - analytic_matrix
    )
    tolerance: Float64[NDArray, "n_probe n_param"] = (
        DERIVATIVE_CROSSCHECK_ATOL
        + DERIVATIVE_CROSSCHECK_RTOL * np.abs(analytic_matrix)
    )
    if np.any(deviation > tolerance):
        raise RuntimeError(
            "complex-step and analytic derivative truths disagree"
        )
    sensitive: Bool[NDArray, " n_param"] = np.ones(
        step_matrix.shape[1], dtype=bool
    )
    column: int
    for column in structural_zero_columns:
        sensitive[column] = False
        if np.any(step_matrix[:, column] != 0.0):
            raise RuntimeError("structural-zero column is not exactly zero")
    if np.min(np.abs(step_matrix[:, sensitive])) < DERIVATIVE_NONZERO_FLOOR:
        raise RuntimeError("sensitive derivative entry underflows the floor")
    if np.any(
        np.max(np.abs(step_matrix[:, sensitive]), axis=0)
        < DERIVATIVE_COLUMN_FLOOR
    ):
        raise RuntimeError("sensitive derivative column lacks a strong probe")
    return float(np.max(deviation))


def _fermi_liquid_derivative_payload() -> Dict[str, Float64[NDArray, "..."]]:
    """PRIVATE: Build Fermi-liquid dimensionless derivative truths.

    Returns
    -------
    payload : dict[str, Float64[NDArray, "..."]]
        Probe frequencies in eV, both derivative matrices, the
        sensitivity masks, and the achieved cross-check deviation.

    Raises
    ------
    RuntimeError
        If :func:`_check_derivative_matrix` rejects either matrix,
        propagated unchanged.

    Notes
    -----
    Columns follow the frozen coefficient order ``(Gamma0, beta,
    Omega_c)`` in the dimensionless coordinates ``q_p = p /
    p_fixture``, so each column is the physical derivative scaled by
    its fixture value.  ``dSigma'/dGamma0`` is a structural zero.
    """
    frequencies: Float64[NDArray, " n_fl_probe"] = FL_PROBE_FREQUENCIES_EV
    physical: Float64[NDArray, " n_fl_param"] = np.asarray(
        [FL_GAMMA0_EV, FL_BETA_PER_EV, FL_OMEGA_C_EV],
        dtype=np.float64,
    )
    imag_step: Float64[NDArray, "n_fl_probe n_fl_param"] = (
        np.stack(
            [
                _complex_step_column(_fl_sigma_imag, frequencies, physical, k)
                for k in range(physical.size)
            ],
            axis=1,
        )
        * physical[None, :]
    )
    real_pair: Float64[NDArray, " n_fl_real_param"] = physical[1:]
    real_step: Float64[NDArray, "n_fl_probe n_fl_param"] = np.zeros(
        (frequencies.size, physical.size),
        dtype=np.float64,
    )
    real_step[:, 1:] = (
        np.stack(
            [
                _complex_step_column(_fl_sigma_real, frequencies, real_pair, k)
                for k in range(real_pair.size)
            ],
            axis=1,
        )
        * real_pair[None, :]
    )

    ratio4: Float64[NDArray, " n_fl_probe"] = (
        frequencies / FL_OMEGA_C_EV
    ) ** 4
    den: Float64[NDArray, " n_fl_probe"] = 1.0 + ratio4
    imag_analytic: Float64[NDArray, "n_fl_probe n_fl_param"] = np.stack(
        (
            -np.ones_like(frequencies) * FL_GAMMA0_EV,
            -(frequencies**2) / den * FL_BETA_PER_EV,
            -4.0
            * FL_BETA_PER_EV
            * frequencies**2
            * ratio4
            / (FL_OMEGA_C_EV * den**2)
            * FL_OMEGA_C_EV,
        ),
        axis=1,
    )
    pole_den: Float64[NDArray, " n_fl_probe"] = (
        frequencies**4 + FL_OMEGA_C_EV**4
    )
    numerator: Float64[NDArray, " n_fl_probe"] = FL_OMEGA_C_EV**3 * (
        frequencies**2 - FL_OMEGA_C_EV**2
    )
    prefactor: float = np.sqrt(2.0) / 2.0
    d_real_d_omega_c: Float64[NDArray, " n_fl_probe"] = (
        prefactor
        * FL_BETA_PER_EV
        * frequencies
        * (
            (3.0 * FL_OMEGA_C_EV**2 * frequencies**2 - 5.0 * FL_OMEGA_C_EV**4)
            / pole_den
            - numerator * 4.0 * FL_OMEGA_C_EV**3 / pole_den**2
        )
    )
    real_analytic: Float64[NDArray, "n_fl_probe n_fl_param"] = np.stack(
        (
            np.zeros_like(frequencies),
            _fl_sigma_real(frequencies, FL_BETA_PER_EV, FL_OMEGA_C_EV),
            d_real_d_omega_c * FL_OMEGA_C_EV,
        ),
        axis=1,
    )
    crosscheck: float = max(
        _check_derivative_matrix(imag_step, imag_analytic, ()),
        _check_derivative_matrix(real_step, real_analytic, (0,)),
    )
    return {
        "fl_derivative_crosscheck_max_abs_error": np.asarray(
            crosscheck,
            dtype=np.float64,
        ),
        "fl_derivative_sensitivity_imag": np.asarray(
            [1.0, 1.0, 1.0],
            dtype=np.float64,
        ),
        "fl_derivative_sensitivity_real": np.asarray(
            [0.0, 1.0, 1.0],
            dtype=np.float64,
        ),
        "fl_dsigma_imag_dq": imag_step,
        "fl_dsigma_real_dq": real_step,
        "fl_probe_frequencies_ev": frequencies,
    }


def _bosonic_kink_derivative_payload() -> Dict[str, Float64[NDArray, "..."]]:
    """PRIVATE: Build bosonic-kink dimensionless derivative truths.

    Returns
    -------
    payload : dict[str, Float64[NDArray, "..."]]
        Probe frequencies in eV, both derivative matrices, the
        sensitivity masks, and the achieved cross-check deviation.

    Raises
    ------
    RuntimeError
        If :func:`_check_derivative_matrix` rejects either matrix,
        propagated unchanged.

    Notes
    -----
    Columns follow the frozen coefficient order ``(Gamma0, g, Omega0,
    width)`` in the dimensionless coordinates ``q_p = p / p_fixture``.
    ``dSigma'/dGamma0`` is a structural zero; every analytic row comes
    from hand differentiation of the pole-pair closed forms.
    """
    frequencies: Float64[NDArray, " n_kink_probe"] = KINK_PROBE_FREQUENCIES_EV
    physical: Float64[NDArray, " n_kink_param"] = np.asarray(
        [KINK_GAMMA0_EV, KINK_COUPLING_EV, KINK_OMEGA0_EV, KINK_WIDTH_EV],
        dtype=np.float64,
    )
    imag_step: Float64[NDArray, "n_kink_probe n_kink_param"] = (
        np.stack(
            [
                _complex_step_column(
                    _kink_sigma_imag, frequencies, physical, k
                )
                for k in range(physical.size)
            ],
            axis=1,
        )
        * physical[None, :]
    )
    real_triple: Float64[NDArray, " n_kink_real_param"] = physical[1:]
    real_step: Float64[NDArray, "n_kink_probe n_kink_param"] = np.zeros(
        (frequencies.size, physical.size),
        dtype=np.float64,
    )
    real_step[:, 1:] = (
        np.stack(
            [
                _complex_step_column(
                    _kink_sigma_real, frequencies, real_triple, k
                )
                for k in range(real_triple.size)
            ],
            axis=1,
        )
        * real_triple[None, :]
    )

    d_minus: Float64[NDArray, " n_kink_probe"] = (
        frequencies - KINK_OMEGA0_EV
    ) ** 2 + KINK_WIDTH_EV**2
    d_plus: Float64[NDArray, " n_kink_probe"] = (
        frequencies + KINK_OMEGA0_EV
    ) ** 2 + KINK_WIDTH_EV**2
    g2: float = KINK_COUPLING_EV**2
    sigma_real: Float64[NDArray, " n_kink_probe"] = _kink_sigma_real(
        frequencies,
        KINK_COUPLING_EV,
        KINK_OMEGA0_EV,
        KINK_WIDTH_EV,
    )
    pair_imag: Float64[NDArray, " n_kink_probe"] = (
        -g2 * KINK_WIDTH_EV * (1.0 / d_minus + 1.0 / d_plus)
    )
    real_analytic: Float64[NDArray, "n_kink_probe n_kink_param"] = np.stack(
        (
            np.zeros_like(frequencies),
            2.0 * sigma_real,
            g2
            * (
                ((frequencies - KINK_OMEGA0_EV) ** 2 - KINK_WIDTH_EV**2)
                / d_minus**2
                - ((frequencies + KINK_OMEGA0_EV) ** 2 - KINK_WIDTH_EV**2)
                / d_plus**2
            )
            * KINK_OMEGA0_EV,
            -2.0
            * g2
            * KINK_WIDTH_EV
            * (
                (frequencies - KINK_OMEGA0_EV) / d_minus**2
                + (frequencies + KINK_OMEGA0_EV) / d_plus**2
            )
            * KINK_WIDTH_EV,
        ),
        axis=1,
    )
    imag_analytic: Float64[NDArray, "n_kink_probe n_kink_param"] = np.stack(
        (
            -np.ones_like(frequencies) * KINK_GAMMA0_EV,
            2.0 * pair_imag,
            -g2
            * KINK_WIDTH_EV
            * (
                2.0 * (frequencies - KINK_OMEGA0_EV) / d_minus**2
                - 2.0 * (frequencies + KINK_OMEGA0_EV) / d_plus**2
            )
            * KINK_OMEGA0_EV,
            -g2
            * (
                ((frequencies - KINK_OMEGA0_EV) ** 2 - KINK_WIDTH_EV**2)
                / d_minus**2
                + ((frequencies + KINK_OMEGA0_EV) ** 2 - KINK_WIDTH_EV**2)
                / d_plus**2
            )
            * KINK_WIDTH_EV,
        ),
        axis=1,
    )
    crosscheck: float = max(
        _check_derivative_matrix(imag_step, imag_analytic, ()),
        _check_derivative_matrix(real_step, real_analytic, (0,)),
    )
    return {
        "kink_derivative_crosscheck_max_abs_error": np.asarray(
            crosscheck,
            dtype=np.float64,
        ),
        "kink_derivative_sensitivity_imag": np.asarray(
            [1.0, 1.0, 1.0, 1.0],
            dtype=np.float64,
        ),
        "kink_derivative_sensitivity_real": np.asarray(
            [0.0, 1.0, 1.0, 1.0],
            dtype=np.float64,
        ),
        "kink_dsigma_imag_dq": imag_step,
        "kink_dsigma_real_dq": real_step,
        "kink_probe_frequencies_ev": frequencies,
    }


def _reference_payload() -> Dict[str, Float64[NDArray, "..."]]:
    """PRIVATE: Assemble every self-energy truth table into one payload.

    Returns
    -------
    payload : dict[str, Float64[NDArray, "..."]]
        Union of the parameter, Fermi-liquid, derivative, kink, KK
        spot-check, and first-moment payloads.

    Notes
    -----
    Later payloads share no keys with earlier ones, so the update
    sequence never overwrites an entry.
    """
    payload: Dict[str, Float64[NDArray, "..."]] = {}
    payload.update(_parameters_payload())
    payload.update(_fermi_liquid_payload())
    payload.update(_fermi_liquid_derivative_payload())
    payload.update(_bosonic_kink_payload())
    payload.update(_bosonic_kink_kk_payload())
    payload.update(_bosonic_kink_moment_payload())
    payload.update(_bosonic_kink_derivative_payload())
    return payload


def main() -> None:
    """Write the frozen self-energy archive and its provenance manifest."""
    root: Path = Path(__file__).resolve().parents[2]
    data_directory: Path = root / "tests" / "test_diffpes" / "_reference_data"
    data_directory.mkdir(parents=True, exist_ok=True)
    reference_path: Path = data_directory / "self_energy_models_reference.npz"
    manifest_path: Path = data_directory / "self_energy_models_manifest.json"

    payload: Dict[str, Float64[NDArray, "..."]] = _reference_payload()
    _write_deterministic_npz(reference_path, payload)

    generator_path: Path = Path(__file__).resolve()
    manifest: Dict[str, Any] = {
        "archives": {
            "self_energy_models_reference": {
                "filename": reference_path.name,
                "sha256": _sha256(reference_path),
            },
        },
        "causality": {
            "convention": "retarded, Sigma'' <= 0 on the whole grid",
            "fl_ceiling_ev": float(payload["fl_causality_ceiling"]),
            "kink_ceiling_ev": float(payload["kink_causality_ceiling"]),
        },
        "conventions": {
            "energy_axis": "relative E-EF in eV",
            "kk_transform": (
                "Sigma'(w) = (1/pi) PV int Sigma''_dyn(x)/(x-w) dx; the "
                "constant -Gamma0 baseline is subtracted before the "
                "transform and contributes zero real part"
            ),
            "raw_coordinates": (
                "raw = softplus_inverse(p) = p + log(-expm1(-p)), bounded "
                "to the closed interval [-30, 30]"
            ),
        },
        "derivatives": {
            "column_floor": DERIVATIVE_COLUMN_FLOOR,
            "coordinates": "dimensionless q_p = p / p_fixture",
            "crosscheck": {
                "fl_max_abs_error": float(
                    payload["fl_derivative_crosscheck_max_abs_error"]
                ),
                "kink_max_abs_error": float(
                    payload["kink_derivative_crosscheck_max_abs_error"]
                ),
                "tolerance": (
                    f"{DERIVATIVE_CROSSCHECK_ATOL} + "
                    f"{DERIVATIVE_CROSSCHECK_RTOL}*|analytic|"
                ),
            },
            "method": (
                "complex-step (relative step 1e-30) on real-form "
                "responses, certified against hand-derived analytic rows"
            ),
            "nonzero_floor": DERIVATIVE_NONZERO_FLOOR,
            "structural_zeros": [
                "dSigma'/dGamma0 == 0 for both fixtures (baseline is "
                "imaginary-only and KK-subtracted)"
            ],
        },
        "environment": {
            "numpy": np.__version__,
            "platform": platform.platform(),
            "python": platform.python_version(),
            "scipy": scipy.__version__,
        },
        "fixtures": {
            "bosonic_kink": {
                "coefficient_order": ["gamma0", "g", "omega0", "width"],
                "first_moment": {
                    "frequencies_ev": KINK_MOMENT_FREQUENCIES_EV.tolist(),
                    "identity": "w*Sigma'(w) -> 2*g**2 as w -> inf",
                    "limit_ev2": float(payload["kink_first_moment_limit"]),
                    "ratio_bounds": list(MOMENT_RATIO_BOUNDS),
                },
                "formula": (
                    "Sigma(w) = -i*Gamma0 + g**2*(1/(w-Omega0+i*width) + "
                    "1/(w+Omega0+i*width))"
                ),
                "parameters_physical": {
                    "g_ev": KINK_COUPLING_EV,
                    "gamma0_ev": KINK_GAMMA0_EV,
                    "omega0_ev": KINK_OMEGA0_EV,
                    "width_ev": KINK_WIDTH_EV,
                },
                "parameters_raw": payload["kink_parameters_raw"].tolist(),
                "spot_check": {
                    "achieved_max_abs_residual": float(
                        np.max(payload["kink_spot_abs_residuals"])
                    ),
                    "budget": KINK_SPOT_ABS_BUDGET,
                    "frequencies_ev": KINK_SPOT_FREQUENCIES_EV.tolist(),
                },
                "tail_mode": "analytic",
            },
            "fermi_liquid": {
                "coefficient_order": ["gamma0", "beta", "omega_c"],
                "formula": (
                    "Sigma''(w) = -Gamma0 - beta*w**2/(1+(w/Omega_c)**4); "
                    "Sigma'(w)-Sigma'(0) = (sqrt(2)/2)*beta*Omega_c**3*"
                    "w*(w**2-Omega_c**2)/(w**4+Omega_c**4)"
                ),
                "kk_domain_ev": list(KK_DOMAIN_EV),
                "parameters_physical": {
                    "beta_per_ev": FL_BETA_PER_EV,
                    "gamma0_ev": FL_GAMMA0_EV,
                    "omega_c_ev": FL_OMEGA_C_EV,
                },
                "parameters_raw": payload["fl_parameters_raw"].tolist(),
                "quadrature_truth": {
                    "achieved_max_abs_error": float(
                        payload["fl_quad_vs_analytic_max_abs_error"]
                    ),
                    "budget": FL_QUAD_ABS_BUDGET,
                    "certified_by": (
                        "exact residue closed form of the rational "
                        "dynamic remainder"
                    ),
                },
                "tail_mode": "power2",
            },
        },
        "generator": (
            "tests/_reference_tools/generate_self_energy_models_reference.py"
        ),
        "generator_sha256": _sha256(generator_path),
        "grids": {
            "eval_grid": {
                "half_width_ev": EVAL_GRID_HALF_WIDTH_EV,
                "points": EVAL_GRID_POINTS,
                "spacing": "uniform, shared by both fixtures",
            },
            "probe_frequencies_ev": {
                "bosonic_kink": KINK_PROBE_FREQUENCIES_EV.tolist(),
                "fermi_liquid": FL_PROBE_FREQUENCIES_EV.tolist(),
            },
        },
        "quadrature": {
            "cauchy_core": "scipy.integrate.quad weight='cauchy' (QAWC)",
            "epsabs": QUAD_EPSABS,
            "epsrel": QUAD_EPSREL,
            "kink_cauchy_half_width_ev": KINK_CAUCHY_HALF_WIDTH_EV,
            "limit": QUAD_LIMIT,
            "tails": (
                "QAGI on the true infinite tails of the dynamic remainder"
            ),
        },
        "requirements": [
            "causal-self-energy-analytic-truth",
            "self-energy-parameter-derivative-truth",
        ],
        "schema": "diffpes.self-energy-models-reference.v1",
        "stage": "frozen-truth-before-self-energy-production-edit",
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
