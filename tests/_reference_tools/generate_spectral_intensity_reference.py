"""Generate the frozen spectral-intensity truth artifacts.

The generator uses only NumPy for numerical truth and never imports
DiffPES, JAX, or either production spectral primitive.  It freezes the
two-pole closed form (including the exact t=0 degenerate limit), the
Hermitian fixture with an independent dense eigendecomposition and a
hand-derived two-solve adjoint derivative truth, the finite-window
arctangent masses plus the full-line tan-map Gauss-Legendre quadrature
convention, the four-rung eta ladder with the frozen HWHM definition,
the sampled-omega Fermi-occupation counterexample, and the registry of
differentiated coordinates.  This preregistration artifact freezes
every constant below.

The eta ladder is ``{8, 4, 2, 1} x 1e-4`` eV exactly; the final-rung
captured-mass budget (6.5e-5) and the HWHM-change budget (1.01e-4 eV)
are satisfiable only at that scale.
"""

from __future__ import annotations

import hashlib
import io
import json
import platform
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import scipy
from beartype.typing import Dict, Tuple
from jaxtyping import Bool, Complex128, Float64, Shaped
from numpy.typing import NDArray

# --- two-pole fixture --------------------------------------------------------
TWO_POLE_EPSILON0: float = -0.15
TWO_POLE_HOPPING: float = 0.08
TWO_POLE_SOURCE: Complex128[NDArray, " n_orb"] = np.asarray(
    [0.9 + 0.4j, -0.5 + 0.7j],
    dtype=np.complex128,
)
TWO_POLE_GAMMA: float = 0.03
TWO_POLE_ETA: float = 1.0e-4
TWO_POLE_OMEGA: Float64[NDArray, " n_two_pole_omega"] = np.linspace(
    -0.6, 0.4, 2001, dtype=np.float64
)

# --- Hermitian resolvent and adjoint-derivative fixture ----------------------
HERMITIAN_MATRIX_A: Complex128[NDArray, "n_orb n_orb"] = np.asarray(
    [[0.2 + 0.1j, -0.3 + 0.7j], [0.4 - 0.2j, -0.1 + 0.05j]],
    dtype=np.complex128,
)
HERMITIAN_SOURCE: Complex128[NDArray, " n_orb"] = np.asarray(
    [1.0 + 0.3j, -0.4 + 0.8j],
    dtype=np.complex128,
)
HERMITIAN_ETA: float = 1.0e-4
HERMITIAN_GAMMA_SIGMA: float = 0.02
HERMITIAN_OMEGA: Float64[NDArray, " n_hermitian_omega"] = np.linspace(
    -1.0, 1.0, 4001, dtype=np.float64
)
ADJOINT_OMEGAS: Float64[NDArray, " n_adjoint_omega"] = np.asarray(
    [-0.95, -0.855, -0.6, -0.2, 0.1, 0.55, 0.9],
    dtype=np.float64,
)
ADJOINT_STEPS: Float64[NDArray, " n_step"] = np.asarray(
    [2.0**-12, 2.0**-14, 2.0**-16],
    dtype=np.float64,
)
ADJOINT_DIRECTIONS: Complex128[NDArray, "n_direction n_orb n_orb"] = (
    np.asarray(
        [
            [[1.0, 0.0], [0.0, 0.0]],
            [[0.0, 0.0], [0.0, 1.0]],
            [[0.0, 1.0], [1.0, 0.0]],
            [[0.0, 1.0j], [-1.0j, 0.0]],
        ],
        dtype=np.complex128,
    )
)
ADJOINT_COORDINATES: Tuple[str, ...] = (
    "h00_real",
    "h11_real",
    "h01_real",
    "h01_imag",
)

# --- sum-rule window and quadrature ------------------------------------------
SUM_RULE_WINDOW: Tuple[float, float] = (-2.0, 2.0)
SUM_RULE_QUADRATURE_ORDERS: Tuple[int, int] = (256, 512)

# --- regulator-limit eta-ladder fixture --------------------------------------
ETA_LADDER_LEVEL_EV: float = 0.10
ETA_LADDER_GAMMA_PHYSICAL: float = 0.05
ETA_LADDER_ETAS: Float64[NDArray, " n_rung"] = np.asarray(
    [8.0e-4, 4.0e-4, 2.0e-4, 1.0e-4],
    dtype=np.float64,
)
ETA_LADDER_OMEGA: Float64[NDArray, " n_ladder_omega"] = np.linspace(
    -1.0, 1.0, 8001, dtype=np.float64
)

# --- Sampled-omega Fermi counterexample -------------------------------------
FERMI_BAND_EV: float = 0.05
FERMI_GAMMA: float = 0.05
FERMI_TEMPERATURE_K: float = 15.0
KB_EV_PER_K: float = 8.617333e-5
FERMI_OMEGA: Float64[NDArray, " n_fermi_omega"] = np.linspace(
    -0.3, 0.3, 1201, dtype=np.float64
)


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


def _lorentzian(
    omega: Float64[NDArray, " n_omega"],
    center: float,
    gamma_total: float,
) -> Float64[NDArray, " n_omega"]:
    """PRIVATE: Evaluate the unit-weight Lorentzian spectral function.

    Parameters
    ----------
    omega : Float64[NDArray, " n_omega"]
        Energies in eV.
    center : float
        Pole position in eV.
    gamma_total : float
        Total HWHM broadening in eV.

    Returns
    -------
    spectral : Float64[NDArray, " n_omega"]
        ``(gamma/pi) / ((omega-center)^2 + gamma^2)`` in 1/eV.

    Notes
    -----
    The full-line integral of this form is exactly one, which anchors
    every sum-rule row.
    """
    return (gamma_total / np.pi) / ((omega - center) ** 2 + gamma_total**2)


def _resolvent_intensity(
    hamiltonian: Complex128[NDArray, "n_orb n_orb"],
    source: Complex128[NDArray, " n_orb"],
    omega: float,
    gamma_total: float,
) -> float:
    """PRIVATE: Evaluate ``-(1/pi) Im[s^dagger G s]`` via one dense solve.

    Parameters
    ----------
    hamiltonian : Complex128[NDArray, "n_orb n_orb"]
        Hamiltonian matrix in eV.
    source : Complex128[NDArray, " n_orb"]
        Matrix-element source vector.
    omega : float
        Energy in eV.
    gamma_total : float
        Total broadening in eV entering ``omega + i gamma``.

    Returns
    -------
    intensity : float
        Spectral intensity in 1/eV at ``omega``.

    Notes
    -----
    ``np.linalg.solve`` applies the resolvent ``G = ((omega + i
    gamma) I - H)^{-1}`` to the source; the projection onto the same
    source gives the diagonal spectral weight.
    """
    operator: Complex128[NDArray, "n_orb n_orb"] = (
        omega + 1j * gamma_total
    ) * np.eye(hamiltonian.shape[0]) - hamiltonian
    solution: Complex128[NDArray, " n_orb"] = np.linalg.solve(operator, source)
    return float(-np.imag(np.vdot(source, solution)) / np.pi)


def _eigen_intensity(
    hamiltonian: Complex128[NDArray, "n_orb n_orb"],
    source: Complex128[NDArray, " n_orb"],
    omega: float,
    gamma_total: float,
) -> float:
    """PRIVATE: Evaluate the eigendecomposition Lorentzian sum via eigh.

    Parameters
    ----------
    hamiltonian : Complex128[NDArray, "n_orb n_orb"]
        Hermitian Hamiltonian matrix in eV.
    source : Complex128[NDArray, " n_orb"]
        Matrix-element source vector.
    omega : float
        Energy in eV.
    gamma_total : float
        Total HWHM broadening in eV.

    Returns
    -------
    intensity : float
        Spectral intensity in 1/eV at ``omega``.

    Notes
    -----
    ``np.linalg.eigh`` supplies eigenvalues and eigenvectors; the
    projected weights ``|V^dagger s|^2`` scale one unit Lorentzian per
    band and the bands sum.
    """
    eigenvalues: Float64[NDArray, " n_orb"]
    eigenvectors: Complex128[NDArray, "n_orb n_orb"]
    eigenvalues, eigenvectors = np.linalg.eigh(hamiltonian)
    weights: Float64[NDArray, " n_orb"] = (
        np.abs(eigenvectors.conj().T @ source) ** 2
    )
    return float(
        np.sum(
            weights
            * (gamma_total / np.pi)
            / ((omega - eigenvalues) ** 2 + gamma_total**2)
        )
    )


def _window_mass(
    center: float,
    gamma_total: float,
    window: Tuple[float, float],
) -> float:
    """PRIVATE: Evaluate the exact finite-window arctangent mass.

    Parameters
    ----------
    center : float
        Lorentzian center in eV.
    gamma_total : float
        Total HWHM broadening in eV.
    window : tuple[float, float]
        Window edges ``(lower, upper)`` in eV.

    Returns
    -------
    mass : float
        Dimensionless captured mass inside the window.

    Notes
    -----
    The antiderivative of the unit Lorentzian is ``arctan((omega -
    center)/gamma)/pi``, so the window mass is the arctangent
    difference over pi.
    """
    lower: float
    upper: float
    lower, upper = window
    return float(
        (
            np.arctan((upper - center) / gamma_total)
            - np.arctan((lower - center) / gamma_total)
        )
        / np.pi
    )


def _fullline_mass(center: float, gamma_total: float, order: int) -> float:
    """PRIVATE: Integrate the Lorentzian over the line via the tan map.

    Parameters
    ----------
    center : float
        Lorentzian center in eV.
    gamma_total : float
        Total HWHM broadening in eV.
    order : int
        Gauss-Legendre order on the compact variable.

    Returns
    -------
    mass : float
        Dimensionless full-line quadrature mass, analytically one.

    Implementation Logic
    --------------------
    The map ``E = center + gamma tan(pi u / 2)`` sends ``u`` in
    ``(-1, 1)`` to the full line with Jacobian ``gamma (pi/2)
    sec^2(pi u / 2)``; Gauss-Legendre nodes in ``u`` then integrate the
    mapped Lorentzian, mirroring the frozen true-Voigt convention.
    """
    nodes: Float64[NDArray, " n_node"]
    weights: Float64[NDArray, " n_node"]
    nodes, weights = np.polynomial.legendre.leggauss(order)
    angle: Float64[NDArray, " n_node"] = np.pi * nodes / 2.0
    energy: Float64[NDArray, " n_node"] = center + gamma_total * np.tan(angle)
    jacobian: Float64[NDArray, " n_node"] = (
        gamma_total * np.pi / 2.0 / np.cos(angle) ** 2
    )
    return float(
        np.sum(weights * _lorentzian(energy, center, gamma_total) * jacobian)
    )


def _quadratic_peak(
    values: Float64[NDArray, " n_omega"],
    omega: Float64[NDArray, " n_omega"],
) -> Tuple[float, float, int]:
    """PRIVATE: Locate the quadratic-interpolated peak on the frozen grid.

    Parameters
    ----------
    values : Float64[NDArray, " n_omega"]
        Sampled intensity values in 1/eV.
    omega : Float64[NDArray, " n_omega"]
        Uniform energy grid in eV.

    Returns
    -------
    peak : tuple[float, float, int]
        Interpolated peak position in eV, interpolated peak height in
        1/eV, and the grid argmax index.

    Notes
    -----
    A parabola through the argmax sample and its two neighbors gives
    the vertex offset ``0.5 (left - right) / (left - 2 mid + right)``
    in cells and the matching vertex height.
    """
    index: int = int(np.argmax(values))
    left: float = values[index - 1]
    mid: float = values[index]
    right: float = values[index + 1]
    curvature: float = left - 2.0 * mid + right
    offset: float = 0.5 * (left - right) / curvature
    spacing: float = float(omega[1] - omega[0])
    position: float = float(omega[index] + offset * spacing)
    height: float = float(mid - 0.25 * (left - right) * offset)
    return position, height, index


def _measured_hwhm(
    values: Float64[NDArray, " n_omega"],
    omega: Float64[NDArray, " n_omega"],
) -> float:
    """PRIVATE: Measure the frozen-definition HWHM on the frozen grid.

    Parameters
    ----------
    values : Float64[NDArray, " n_omega"]
        Sampled intensity values in 1/eV.
    omega : Float64[NDArray, " n_omega"]
        Uniform energy grid in eV.

    Returns
    -------
    hwhm : float
        Half width at half maximum in eV.

    Implementation Logic
    --------------------
    The frozen definition: the peak position and height come from the
    three-point quadratic vertex around the grid argmax; the half
    level is half of that vertex height; linear interpolation between
    the bracketing grid samples locates each half-level crossing; the
    HWHM is half the distance between the two crossings.
    """
    position: float
    height: float
    index: int
    position, height, index = _quadratic_peak(values, omega)
    del position
    half: float = 0.5 * height
    spacing: float = float(omega[1] - omega[0])
    left_index: int = index
    while values[left_index] >= half:
        left_index -= 1
    left_fraction: float = (half - values[left_index]) / (
        values[left_index + 1] - values[left_index]
    )
    left_crossing: float = float(omega[left_index] + left_fraction * spacing)
    right_index: int = index
    while values[right_index] >= half:
        right_index += 1
    right_fraction: float = (half - values[right_index - 1]) / (
        values[right_index] - values[right_index - 1]
    )
    right_crossing: float = float(
        omega[right_index - 1] + right_fraction * spacing
    )
    return 0.5 * (right_crossing - left_crossing)


def _stable_fermi(
    energy: Float64[NDArray, " n_omega"],
    kt: float,
) -> Float64[NDArray, " n_omega"]:
    """PRIVATE: Evaluate the analytic Fermi function without overflow.

    Parameters
    ----------
    energy : Float64[NDArray, " n_omega"]
        Energies relative to the chemical potential in eV.
    kt : float
        Thermal energy ``k_B T`` in eV.

    Returns
    -------
    occupation : Float64[NDArray, " n_omega"]
        Fermi-Dirac occupation in ``[0, 1]``.

    Notes
    -----
    Positive exponents use the decaying form ``e^{-x}/(1+e^{-x})`` and
    negative exponents use ``1/(1+e^{x})``, so no branch ever
    exponentiates a large positive number.
    """
    exponent: Float64[NDArray, " n_omega"] = (
        np.asarray(energy, dtype=np.float64) / kt
    )
    occupation: Float64[NDArray, " n_omega"] = np.empty_like(exponent)
    positive: Bool[NDArray, " n_omega"] = exponent >= 0.0
    decaying: Float64[NDArray, " n_positive"] = np.exp(-exponent[positive])
    occupation[positive] = decaying / (1.0 + decaying)
    occupation[~positive] = 1.0 / (1.0 + np.exp(exponent[~positive]))
    return occupation


def _two_pole_payload() -> Dict[str, Float64[NDArray, "..."]]:
    """PRIVATE: Build the two-pole closed form and its degenerate limit.

    Returns
    -------
    payload : dict[str, Float64[NDArray, "..."]]
        Fixture constants, pole energies and weights in eV, the omega
        grid, and both closed-form intensity rows in 1/eV.

    Raises
    ------
    RuntimeError
        If the closed form or the exact ``t = 0`` degenerate limit
        disagrees with the dense resolvent solve beyond rtol 1e-12.

    Implementation Logic
    --------------------
    The symmetric 2x2 Hamiltonian diagonalizes into ``eps0 +- t`` with
    weights ``|m1 +- m2|^2 / 2``; each pole carries one unit
    Lorentzian scaled by its weight, and the dense solve certifies
    both rows point by point.
    """
    gamma_total: float = TWO_POLE_GAMMA + TWO_POLE_ETA
    source_plus: complex = TWO_POLE_SOURCE[0] + TWO_POLE_SOURCE[1]
    source_minus: complex = TWO_POLE_SOURCE[0] - TWO_POLE_SOURCE[1]
    pole_weights: Float64[NDArray, " n_pole"] = np.asarray(
        [np.abs(source_plus) ** 2 / 2.0, np.abs(source_minus) ** 2 / 2.0],
        dtype=np.float64,
    )
    pole_energies: Float64[NDArray, " n_pole"] = np.asarray(
        [
            TWO_POLE_EPSILON0 + TWO_POLE_HOPPING,
            TWO_POLE_EPSILON0 - TWO_POLE_HOPPING,
        ],
        dtype=np.float64,
    )
    intensity: Float64[NDArray, " n_two_pole_omega"] = pole_weights[
        0
    ] * _lorentzian(
        TWO_POLE_OMEGA, pole_energies[0], gamma_total
    ) + pole_weights[1] * _lorentzian(
        TWO_POLE_OMEGA, pole_energies[1], gamma_total
    )
    degenerate_weight: float = float(np.sum(np.abs(TWO_POLE_SOURCE) ** 2))
    degenerate: Float64[NDArray, " n_two_pole_omega"] = (
        degenerate_weight
        * _lorentzian(TWO_POLE_OMEGA, TWO_POLE_EPSILON0, gamma_total)
    )
    hamiltonian: Complex128[NDArray, "n_orb n_orb"] = np.asarray(
        [
            [TWO_POLE_EPSILON0, TWO_POLE_HOPPING],
            [TWO_POLE_HOPPING, TWO_POLE_EPSILON0],
        ],
        dtype=np.complex128,
    )
    hamiltonian_degenerate: Complex128[NDArray, "n_orb n_orb"] = (
        TWO_POLE_EPSILON0 * np.eye(2, dtype=np.complex128)
    )
    direct: Float64[NDArray, " n_two_pole_omega"] = np.asarray(
        [
            _resolvent_intensity(
                hamiltonian, TWO_POLE_SOURCE, omega, gamma_total
            )
            for omega in TWO_POLE_OMEGA
        ],
        dtype=np.float64,
    )
    direct_degenerate: Float64[NDArray, " n_two_pole_omega"] = np.asarray(
        [
            _resolvent_intensity(
                hamiltonian_degenerate, TWO_POLE_SOURCE, omega, gamma_total
            )
            for omega in TWO_POLE_OMEGA
        ],
        dtype=np.float64,
    )
    if np.max(np.abs(intensity - direct) / direct) > 1.0e-12:
        raise RuntimeError(
            "two-pole closed form disagrees with the dense solve"
        )
    if np.max(np.abs(degenerate - direct_degenerate) / degenerate) > 1.0e-12:
        raise RuntimeError(
            "two-pole degenerate limit disagrees with the solve"
        )
    return {
        "two_pole_epsilon0": np.asarray(TWO_POLE_EPSILON0),
        "two_pole_eta": np.asarray(TWO_POLE_ETA),
        "two_pole_gamma": np.asarray(TWO_POLE_GAMMA),
        "two_pole_gamma_total": np.asarray(gamma_total),
        "two_pole_hopping": np.asarray(TWO_POLE_HOPPING),
        "two_pole_intensity": intensity,
        "two_pole_intensity_degenerate": degenerate,
        "two_pole_omega": TWO_POLE_OMEGA,
        "two_pole_pole_energies": pole_energies,
        "two_pole_pole_weights": pole_weights,
        "two_pole_source_imag": TWO_POLE_SOURCE.imag,
        "two_pole_source_real": TWO_POLE_SOURCE.real,
    }


def _adjoint_truth(
    hamiltonian: Complex128[NDArray, "n_orb n_orb"],
    gamma_total: float,
) -> Dict[str, Float64[NDArray, "..."]]:
    """PRIVATE: Build the adjoint and Richardson FD derivative truth.

    Parameters
    ----------
    hamiltonian : Complex128[NDArray, "n_orb n_orb"]
        Hermitian fixture Hamiltonian in eV.
    gamma_total : float
        Total broadening in eV.

    Returns
    -------
    payload : dict[str, Float64[NDArray, "..."]]
        Adjoint omegas in eV, intensities, analytic two-solve
        derivatives, raw central differences per step, and the
        two-level Richardson extrapolation.

    Raises
    ------
    RuntimeError
        If the Richardson finite differences leave the 1e-10 adjoint
        bound.

    Implementation Logic
    --------------------
    Two dense solves per omega give ``g = G s`` and ``h = G^dagger
    s``; the derivative for direction ``E`` is ``-(1/pi) Im[h^dagger E
    g]``.  Central differences of the eigendecomposition intensity
    over the frozen step ladder extrapolate twice (orders 2 -> 4 -> 6)
    and certify the analytic rows.
    """
    analytic: Float64[NDArray, "n_adjoint_omega n_direction"] = np.empty(
        (ADJOINT_OMEGAS.size, ADJOINT_DIRECTIONS.shape[0])
    )
    finite_difference: Float64[
        NDArray, "n_adjoint_omega n_step n_direction"
    ] = np.empty(
        (ADJOINT_OMEGAS.size, ADJOINT_STEPS.size, ADJOINT_DIRECTIONS.shape[0])
    )
    intensities: Float64[NDArray, " n_adjoint_omega"] = np.empty(
        ADJOINT_OMEGAS.size
    )
    omega_index: int
    omega: float
    for omega_index, omega in enumerate(ADJOINT_OMEGAS):
        operator: Complex128[NDArray, "n_orb n_orb"] = (
            omega + 1j * gamma_total
        ) * np.eye(2) - hamiltonian
        forward: Complex128[NDArray, " n_orb"] = np.linalg.solve(
            operator, HERMITIAN_SOURCE
        )
        adjoint: Complex128[NDArray, " n_orb"] = np.linalg.solve(
            operator.conj().T, HERMITIAN_SOURCE
        )
        intensities[omega_index] = -np.imag(
            np.vdot(HERMITIAN_SOURCE, forward)
        ) / (np.pi)
        direction_index: int
        direction: Complex128[NDArray, "n_orb n_orb"]
        for direction_index, direction in enumerate(ADJOINT_DIRECTIONS):
            analytic[omega_index, direction_index] = (
                -np.imag(adjoint.conj() @ (direction @ forward)) / np.pi
            )
            step_index: int
            step: float
            for step_index, step in enumerate(ADJOINT_STEPS):
                finite_difference[omega_index, step_index, direction_index] = (
                    _eigen_intensity(
                        hamiltonian + step * direction,
                        HERMITIAN_SOURCE,
                        omega,
                        gamma_total,
                    )
                    - _eigen_intensity(
                        hamiltonian - step * direction,
                        HERMITIAN_SOURCE,
                        omega,
                        gamma_total,
                    )
                ) / (2.0 * step)
    first_level_coarse: Float64[NDArray, "n_adjoint_omega n_direction"] = (
        16.0 * finite_difference[:, 1, :] - finite_difference[:, 0, :]
    ) / 15.0
    first_level_fine: Float64[NDArray, "n_adjoint_omega n_direction"] = (
        16.0 * finite_difference[:, 2, :] - finite_difference[:, 1, :]
    ) / 15.0
    richardson: Float64[NDArray, "n_adjoint_omega n_direction"] = (
        256.0 * first_level_fine - first_level_coarse
    ) / 255.0
    tolerance: Float64[NDArray, "n_adjoint_omega n_direction"] = (
        1.0e-10 * np.maximum(1.0, np.abs(analytic))
    )
    if np.any(np.abs(richardson - analytic) > tolerance):
        raise RuntimeError(
            "Richardson finite differences leave the adjoint bound"
        )
    return {
        "adjoint_analytic": analytic,
        "adjoint_direction_imag": ADJOINT_DIRECTIONS.imag,
        "adjoint_direction_real": ADJOINT_DIRECTIONS.real,
        "adjoint_fd": finite_difference,
        "adjoint_intensities": intensities,
        "adjoint_omegas": ADJOINT_OMEGAS,
        "adjoint_richardson": richardson,
        "adjoint_steps": ADJOINT_STEPS,
    }


def _hermitian_payload() -> Tuple[
    Dict[str, Float64[NDArray, "..."]],
    Float64[NDArray, " n_orb"],
    Float64[NDArray, " n_orb"],
]:
    """PRIVATE: Build the eigendecomposition truth and the adjoint rows.

    Returns
    -------
    result : tuple[dict[str, Float64[NDArray, "..."]], Float64[NDArray, " n_orb"], Float64[NDArray, " n_orb"]]
        The Hermitian payload (fixture matrices, eigenvalues in eV,
        band weights, intensity row, adjoint rows), plus the
        eigenvalues and band weights again for the sum-rule assembly.

    Raises
    ------
    RuntimeError
        If the band weights lose the source norm beyond 1e-13, or if
        the eigendecomposition intensity disagrees with the dense
        solve beyond rtol 1e-12.

    Implementation Logic
    --------------------
    ``H = A + A^dagger`` symmetrizes the frozen seed matrix; ``eigh``
    gives bands and weights; the Lorentzian band sum and the resolvent
    solve certify each other on the whole omega grid.
    """
    hamiltonian: Complex128[NDArray, "n_orb n_orb"] = (
        HERMITIAN_MATRIX_A + HERMITIAN_MATRIX_A.conj().T
    )
    gamma_total: float = HERMITIAN_GAMMA_SIGMA + HERMITIAN_ETA
    eigenvalues: Float64[NDArray, " n_orb"]
    eigenvectors: Complex128[NDArray, "n_orb n_orb"]
    eigenvalues, eigenvectors = np.linalg.eigh(hamiltonian)
    band_weights: Float64[NDArray, " n_orb"] = (
        np.abs(eigenvectors.conj().T @ HERMITIAN_SOURCE) ** 2
    )
    weight_total: float = float(np.sum(np.abs(HERMITIAN_SOURCE) ** 2))
    if abs(float(np.sum(band_weights)) - weight_total) > 1.0e-13:
        raise RuntimeError(
            "Hermitian band weights do not resolve the source norm"
        )
    intensity: Float64[NDArray, " n_hermitian_omega"] = np.sum(
        band_weights[:, None]
        * (gamma_total / np.pi)
        / (
            (HERMITIAN_OMEGA[None, :] - eigenvalues[:, None]) ** 2
            + gamma_total**2
        ),
        axis=0,
    )
    direct: Float64[NDArray, " n_hermitian_omega"] = np.asarray(
        [
            _resolvent_intensity(
                hamiltonian, HERMITIAN_SOURCE, omega, gamma_total
            )
            for omega in HERMITIAN_OMEGA
        ],
        dtype=np.float64,
    )
    if np.max(np.abs(intensity - direct) / intensity) > 1.0e-12:
        raise RuntimeError(
            "Hermitian eigendecomposition disagrees with the solve"
        )
    payload: Dict[str, Float64[NDArray, "..."]] = {
        "hermitian_band_weights": band_weights,
        "hermitian_eigenvalues": eigenvalues,
        "hermitian_eta": np.asarray(HERMITIAN_ETA),
        "hermitian_gamma_sigma": np.asarray(HERMITIAN_GAMMA_SIGMA),
        "hermitian_gamma_total": np.asarray(gamma_total),
        "hermitian_hamiltonian_imag": hamiltonian.imag,
        "hermitian_hamiltonian_real": hamiltonian.real,
        "hermitian_intensity": intensity,
        "hermitian_matrix_a_imag": HERMITIAN_MATRIX_A.imag,
        "hermitian_matrix_a_real": HERMITIAN_MATRIX_A.real,
        "hermitian_omega": HERMITIAN_OMEGA,
        "hermitian_source_imag": HERMITIAN_SOURCE.imag,
        "hermitian_source_real": HERMITIAN_SOURCE.real,
    }
    payload.update(_adjoint_truth(hamiltonian, gamma_total))
    return payload, eigenvalues, band_weights


def _sum_rule_rows(
    hermitian_eigenvalues: Float64[NDArray, " n_orb"],
    hermitian_band_weights: Float64[NDArray, " n_orb"],
) -> Tuple[
    Tuple[str, ...],
    Float64[NDArray, " n_row"],
    Float64[NDArray, " n_row"],
    Float64[NDArray, " n_row"],
]:
    """PRIVATE: Assemble the frozen sum-rule fixture rows.

    Parameters
    ----------
    hermitian_eigenvalues : Float64[NDArray, " n_orb"]
        Hermitian fixture band energies in eV.
    hermitian_band_weights : Float64[NDArray, " n_orb"]
        Projected band weights of the Hermitian fixture.

    Returns
    -------
    rows : tuple[tuple[str, ...], Float64[NDArray, " n_row"], Float64[NDArray, " n_row"], Float64[NDArray, " n_row"]]
        Row labels, centers in eV, total gammas in eV, and weights.

    Notes
    -----
    The rows cover both two-pole branches, the degenerate pole, both
    Hermitian bands, every eta-ladder rung, and the eta-to-zero rung.
    """
    two_pole_gamma_total: float = TWO_POLE_GAMMA + TWO_POLE_ETA
    hermitian_gamma_total: float = HERMITIAN_GAMMA_SIGMA + HERMITIAN_ETA
    labels: list[str] = [
        "two_pole_plus",
        "two_pole_minus",
        "two_pole_degenerate",
        "hermitian_band_0",
        "hermitian_band_1",
    ]
    centers: list[float] = [
        TWO_POLE_EPSILON0 + TWO_POLE_HOPPING,
        TWO_POLE_EPSILON0 - TWO_POLE_HOPPING,
        TWO_POLE_EPSILON0,
        float(hermitian_eigenvalues[0]),
        float(hermitian_eigenvalues[1]),
    ]
    gammas: list[float] = [
        two_pole_gamma_total,
        two_pole_gamma_total,
        two_pole_gamma_total,
        hermitian_gamma_total,
        hermitian_gamma_total,
    ]
    weights: list[float] = [
        float(np.abs(TWO_POLE_SOURCE[0] + TWO_POLE_SOURCE[1]) ** 2 / 2.0),
        float(np.abs(TWO_POLE_SOURCE[0] - TWO_POLE_SOURCE[1]) ** 2 / 2.0),
        float(np.sum(np.abs(TWO_POLE_SOURCE) ** 2)),
        float(hermitian_band_weights[0]),
        float(hermitian_band_weights[1]),
    ]
    eta: float
    for eta in ETA_LADDER_ETAS:
        labels.append(f"eta_ladder_eta_{eta:.0e}")
        centers.append(ETA_LADDER_LEVEL_EV)
        gammas.append(ETA_LADDER_GAMMA_PHYSICAL + float(eta))
        weights.append(1.0)
    labels.append("eta_ladder_eta_0")
    centers.append(ETA_LADDER_LEVEL_EV)
    gammas.append(ETA_LADDER_GAMMA_PHYSICAL)
    weights.append(1.0)
    return (
        tuple(labels),
        np.asarray(centers, dtype=np.float64),
        np.asarray(gammas, dtype=np.float64),
        np.asarray(weights, dtype=np.float64),
    )


def _sum_rule_payload(
    hermitian_eigenvalues: Float64[NDArray, " n_orb"],
    hermitian_band_weights: Float64[NDArray, " n_orb"],
) -> Tuple[Dict[str, Float64[NDArray, "..."]], Tuple[str, ...]]:
    """PRIVATE: Build the window masses and full-line quadrature masses.

    Parameters
    ----------
    hermitian_eigenvalues : Float64[NDArray, " n_orb"]
        Hermitian fixture band energies in eV.
    hermitian_band_weights : Float64[NDArray, " n_orb"]
        Projected band weights of the Hermitian fixture.

    Returns
    -------
    result : tuple[dict[str, Float64[NDArray, "..."]], tuple[str, ...]]
        The sum-rule payload (window and full-line masses, order
        deltas, row constants) and the row labels for the manifest.

    Raises
    ------
    RuntimeError
        If a full-line tan-map mass leaves the 1e-8 analytic bound, or
        if the 256-to-512 order delta exceeds 1e-12.

    Notes
    -----
    Window masses use the exact arctangent form; full-line masses use
    the frozen tan-map Gauss-Legendre rule at both registered orders.
    """
    labels: Tuple[str, ...]
    centers: Float64[NDArray, " n_row"]
    gammas: Float64[NDArray, " n_row"]
    weights: Float64[NDArray, " n_row"]
    labels, centers, gammas, weights = _sum_rule_rows(
        hermitian_eigenvalues, hermitian_band_weights
    )
    window_masses: Float64[NDArray, " n_row"] = np.asarray(
        [
            _window_mass(center, gamma, SUM_RULE_WINDOW)
            for center, gamma in zip(centers, gammas, strict=True)
        ],
        dtype=np.float64,
    )
    fullline: Float64[NDArray, "n_row n_order"] = np.asarray(
        [
            [
                _fullline_mass(center, gamma, order)
                for order in SUM_RULE_QUADRATURE_ORDERS
            ]
            for center, gamma in zip(centers, gammas, strict=True)
        ],
        dtype=np.float64,
    )
    if np.max(np.abs(fullline - 1.0)) > 1.0e-8:
        raise RuntimeError("full-line tan-map mass leaves the analytic bound")
    order_deltas: Float64[NDArray, " n_row"] = np.abs(
        fullline[:, 1] - fullline[:, 0]
    )
    if np.max(order_deltas) > 1.0e-12:
        raise RuntimeError("full-line 256-to-512 order delta is unstable")
    payload: Dict[str, Float64[NDArray, "..."]] = {
        "sum_rule_centers": centers,
        "sum_rule_fullline_masses": fullline,
        "sum_rule_fullline_order_deltas": order_deltas,
        "sum_rule_gamma_totals": gammas,
        "sum_rule_quadrature_orders": np.asarray(
            SUM_RULE_QUADRATURE_ORDERS,
            dtype=np.float64,
        ),
        "sum_rule_weights": weights,
        "sum_rule_window": np.asarray(SUM_RULE_WINDOW, dtype=np.float64),
        "sum_rule_window_masses_unit": window_masses,
        "sum_rule_window_masses_weighted": weights * window_masses,
    }
    return payload, labels


def _eta_ladder_payload() -> Dict[str, Float64[NDArray, "..."]]:
    """PRIVATE: Build the eta ladder, HWHM, ratios, and extrapolation.

    Returns
    -------
    payload : dict[str, Float64[NDArray, "..."]]
        Rung intensities in 1/eV, captured masses, measured and
        analytic HWHM in eV, error ratios, final-rung changes, and the
        eta-to-zero linear extrapolations.

    Raises
    ------
    RuntimeError
        If a rung-to-rung error ratio leaves the ``2 +/- 0.02`` band,
        if a final-rung change exceeds its frozen budget, if the
        interpolated peak drifts beyond 2e-6 eV from the level, or if
        an extrapolation misses its limit beyond 2e-6.

    Implementation Logic
    --------------------
    Each rung is one Lorentzian at the fixed level with ``gamma +
    eta``.  Errors against the eta-to-zero row scale linearly in eta,
    so halving eta halves them; the two smallest rungs extrapolate
    linearly to eta zero.
    """
    window: Tuple[float, float] = (
        float(ETA_LADDER_OMEGA[0]),
        float(ETA_LADDER_OMEGA[-1]),
    )
    gamma_totals: Float64[NDArray, " n_rung"] = (
        ETA_LADDER_GAMMA_PHYSICAL + ETA_LADDER_ETAS
    )
    intensities: Float64[NDArray, "n_rung n_ladder_omega"] = np.stack(
        [
            _lorentzian(ETA_LADDER_OMEGA, ETA_LADDER_LEVEL_EV, gamma)
            for gamma in gamma_totals
        ]
    )
    intensity_eta0: Float64[NDArray, " n_ladder_omega"] = _lorentzian(
        ETA_LADDER_OMEGA, ETA_LADDER_LEVEL_EV, ETA_LADDER_GAMMA_PHYSICAL
    )
    captured: Float64[NDArray, " n_rung"] = np.asarray(
        [
            _window_mass(ETA_LADDER_LEVEL_EV, gamma, window)
            for gamma in gamma_totals
        ],
        dtype=np.float64,
    )
    captured_eta0: float = _window_mass(
        ETA_LADDER_LEVEL_EV, ETA_LADDER_GAMMA_PHYSICAL, window
    )
    peak_positions: Float64[NDArray, " n_rung"] = np.empty(
        ETA_LADDER_ETAS.size
    )
    peak_heights: Float64[NDArray, " n_rung"] = np.empty(ETA_LADDER_ETAS.size)
    hwhm: Float64[NDArray, " n_rung"] = np.empty(ETA_LADDER_ETAS.size)
    rung: int
    for rung in range(ETA_LADDER_ETAS.size):
        peak_positions[rung], peak_heights[rung], _ = _quadratic_peak(
            intensities[rung], ETA_LADDER_OMEGA
        )
        hwhm[rung] = _measured_hwhm(intensities[rung], ETA_LADDER_OMEGA)
    mass_errors: Float64[NDArray, " n_rung"] = np.abs(captured - captured_eta0)
    hwhm_errors: Float64[NDArray, " n_rung"] = np.abs(
        hwhm - ETA_LADDER_GAMMA_PHYSICAL
    )
    mass_ratios: Float64[NDArray, " n_rung_ratio"] = (
        mass_errors[:-1] / mass_errors[1:]
    )
    hwhm_ratios: Float64[NDArray, " n_rung_ratio"] = (
        hwhm_errors[:-1] / hwhm_errors[1:]
    )
    ratios: Float64[NDArray, " n_ratio"] = np.concatenate(
        (mass_ratios, hwhm_ratios)
    )
    if np.any((ratios < 1.98) | (ratios > 2.02)):
        raise RuntimeError(
            "eta-ladder rung-to-rung ratio leaves the 2 +/- 0.02 band"
        )
    final_mass_change: float = float(np.abs(captured[-1] - captured[-2]))
    final_hwhm_change: float = float(np.abs(hwhm[-1] - hwhm[-2]))
    if final_mass_change > 6.5e-5:
        raise RuntimeError(
            "eta-ladder final-rung captured-mass change is too large"
        )
    if final_hwhm_change > 1.01e-4:
        raise RuntimeError("eta-ladder final-rung HWHM change is too large")
    if np.max(np.abs(peak_positions - ETA_LADDER_LEVEL_EV)) > 2.0e-6:
        raise RuntimeError(
            "eta-ladder interpolated peak drifts from the level"
        )
    extrapolated_mass: float = float(2.0 * captured[-1] - captured[-2])
    extrapolated_hwhm: float = float(2.0 * hwhm[-1] - hwhm[-2])
    extrapolation_errors: Float64[NDArray, " n_extrapolation"] = np.asarray(
        [
            abs(extrapolated_mass - captured_eta0),
            abs(extrapolated_hwhm - ETA_LADDER_GAMMA_PHYSICAL),
        ],
        dtype=np.float64,
    )
    if np.any(extrapolation_errors > 2.0e-6):
        raise RuntimeError(
            "eta-ladder eta-to-zero extrapolation misses the limit"
        )
    return {
        "eta_ladder_captured_mass_eta0": np.asarray(captured_eta0),
        "eta_ladder_captured_masses": captured,
        "eta_ladder_etas": ETA_LADDER_ETAS,
        "eta_ladder_extrapolated_hwhm": np.asarray(extrapolated_hwhm),
        "eta_ladder_extrapolated_mass": np.asarray(extrapolated_mass),
        "eta_ladder_extrapolation_errors": extrapolation_errors,
        "eta_ladder_final_hwhm_change": np.asarray(final_hwhm_change),
        "eta_ladder_final_mass_change": np.asarray(final_mass_change),
        "eta_ladder_gamma_physical": np.asarray(ETA_LADDER_GAMMA_PHYSICAL),
        "eta_ladder_gamma_totals": gamma_totals,
        "eta_ladder_hwhm_analytic": gamma_totals.copy(),
        "eta_ladder_hwhm_errors": hwhm_errors,
        "eta_ladder_hwhm_measured": hwhm,
        "eta_ladder_hwhm_ratios": hwhm_ratios,
        "eta_ladder_intensities": intensities,
        "eta_ladder_intensity_eta0": intensity_eta0,
        "eta_ladder_level_energy": np.asarray(ETA_LADDER_LEVEL_EV),
        "eta_ladder_mass_errors": mass_errors,
        "eta_ladder_mass_ratios": mass_ratios,
        "eta_ladder_omega": ETA_LADDER_OMEGA,
        "eta_ladder_peak_heights": peak_heights,
        "eta_ladder_peak_heights_analytic": 1.0 / (np.pi * gamma_totals),
        "eta_ladder_peak_positions": peak_positions,
    }


def _fermi_payload() -> Dict[str, Float64[NDArray, "..."]]:
    """PRIVATE: Build the sampled-omega Fermi-occupation counterexample.

    Returns
    -------
    payload : dict[str, Float64[NDArray, "..."]]
        Spectral row in 1/eV, both intensity conventions, the sampled
        occupation, fixture constants, and the achieved maximum
        relative discrepancy.

    Raises
    ------
    RuntimeError
        If the maximum relative discrepancy drops below the 1e12
        floor, which would mean the occupied-side tail vanished.

    Notes
    -----
    The correct convention multiplies the spectral function by the
    occupation at the sampled omega; the wrong convention uses the
    occupation at the band eigenvalue, which kills the occupied-side
    tail below the chemical potential at 15 K.
    """
    kt: float = KB_EV_PER_K * FERMI_TEMPERATURE_K
    spectral: Float64[NDArray, " n_fermi_omega"] = _lorentzian(
        FERMI_OMEGA, FERMI_BAND_EV, FERMI_GAMMA
    )
    occupation_sampled: Float64[NDArray, " n_fermi_omega"] = _stable_fermi(
        FERMI_OMEGA, kt
    )
    occupation_at_band: float = float(
        _stable_fermi(np.asarray([FERMI_BAND_EV]), kt)[0]
    )
    intensity_correct: Float64[NDArray, " n_fermi_omega"] = (
        spectral * occupation_sampled
    )
    intensity_wrong: Float64[NDArray, " n_fermi_omega"] = (
        spectral * occupation_at_band
    )
    relative: Float64[NDArray, " n_fermi_omega"] = (
        np.abs(intensity_correct - intensity_wrong) / intensity_wrong
    )
    maximum_relative: float = float(np.max(relative))
    if maximum_relative < 1.0e12:
        raise RuntimeError(
            "sampled-omega counterexample lost its occupied-side tail"
        )
    return {
        "fermi_band_energy": np.asarray(FERMI_BAND_EV),
        "fermi_gamma": np.asarray(FERMI_GAMMA),
        "fermi_intensity_correct": intensity_correct,
        "fermi_intensity_wrong": intensity_wrong,
        "fermi_kt_ev": np.asarray(kt),
        "fermi_max_relative_discrepancy": np.asarray(maximum_relative),
        "fermi_occupation_at_band": np.asarray(occupation_at_band),
        "fermi_occupation_sampled": occupation_sampled,
        "fermi_omega": FERMI_OMEGA,
        "fermi_spectral": spectral,
        "fermi_temperature_k": np.asarray(FERMI_TEMPERATURE_K),
    }


def _reference_payload() -> Tuple[
    Dict[str, Float64[NDArray, "..."]], Tuple[str, ...]
]:
    """PRIVATE: Assemble every spectral-intensity table into one payload.

    Returns
    -------
    result : tuple[dict[str, Float64[NDArray, "..."]], tuple[str, ...]]
        Union of the two-pole, Hermitian, sum-rule, eta-ladder, and
        Fermi payloads, plus the sum-rule row labels.

    Notes
    -----
    Later payloads share no keys with earlier ones, so the update
    sequence never overwrites an entry.
    """
    payload: Dict[str, Float64[NDArray, "..."]] = {}
    payload.update(_two_pole_payload())
    hermitian_payload: Dict[str, Float64[NDArray, "..."]]
    hermitian_eigenvalues: Float64[NDArray, " n_orb"]
    hermitian_band_weights: Float64[NDArray, " n_orb"]
    hermitian_payload, hermitian_eigenvalues, hermitian_band_weights = (
        _hermitian_payload()
    )
    payload.update(hermitian_payload)
    sum_rule_payload: Dict[str, Float64[NDArray, "..."]]
    sum_rule_labels: Tuple[str, ...]
    sum_rule_payload, sum_rule_labels = _sum_rule_payload(
        hermitian_eigenvalues, hermitian_band_weights
    )
    payload.update(sum_rule_payload)
    payload.update(_eta_ladder_payload())
    payload.update(_fermi_payload())
    return payload, sum_rule_labels


def _manifest(
    reference_path: Path,
    generator_path: Path,
    sum_rule_labels: Tuple[str, ...],
) -> Dict[str, Any]:
    """PRIVATE: Assemble the provenance manifest with every constant.

    Parameters
    ----------
    reference_path : Path
        Written NPZ archive whose digest enters the manifest.
    generator_path : Path
        This generator file whose digest enters the manifest.
    sum_rule_labels : tuple[str, ...]
        Row labels from the sum-rule assembly.

    Returns
    -------
    manifest : dict[str, Any]
        JSON-ready manifest with fixture constants, conventions,
        budgets, requirement names, and both SHA-256 digests.

    Notes
    -----
    The manifest freezes every tolerance and grid constant next to the
    archive digest, so any regeneration with different constants
    changes the recorded provenance.
    """
    gamma_raw: float = float(np.log(np.expm1(TWO_POLE_GAMMA)))
    return {
        "adjoint_derivative": {
            "adjoint_truth": (
                "two dense solves g=G M and h=G^dagger M with "
                "G=((omega+i*Gamma_total)I-H)^-1; "
                "dI/dtheta = -(1/pi) Im[h^dagger E_theta g]"
            ),
            "coordinates": list(ADJOINT_COORDINATES),
            "direction_matrices": (
                "E_00, E_11, E_01+E_10, i*(E_01-E_10); real coordinate "
                "directions for every independent Hermitian entry"
            ),
            "fd_steps": ADJOINT_STEPS.tolist(),
            "omegas_ev": ADJOINT_OMEGAS.tolist(),
            "richardson": (
                "two-level: R_j=(16*D(h_{j+1})-D(h_j))/15 over the "
                "frozen step ladder, then (256*R_2-R_1)/255"
            ),
            "tolerance": (
                "abs(richardson - analytic) <= 1e-10 * max(1, abs(analytic))"
            ),
        },
        "archives": {
            "spectral_intensity_reference": {
                "filename": reference_path.name,
                "sha256": _sha256(reference_path),
            },
        },
        "differentiated_coordinates": [
            {
                "fixture": "two_pole",
                "name": "two_pole_hopping_t",
                "value": TWO_POLE_HOPPING,
            },
            {
                "fixture": "hermitian_resolvent",
                "name": "h01_real",
                "value": 0.1,
            },
            {
                "fixture": "two_pole",
                "name": "gamma_softplus_raw",
                "note": "Gamma = softplus(raw); raw = log(expm1(0.03))",
                "value": gamma_raw,
            },
            {
                "fixture": "shared",
                "name": "eta",
                "value": TWO_POLE_ETA,
            },
        ],
        "environment": {
            "numpy": np.__version__,
            "platform": platform.platform(),
            "python": platform.python_version(),
            "scipy": scipy.__version__,
        },
        "eta_ladder": {
            "convergence_ratio_band": [1.98, 2.02],
            "eta_ladder_ev": ETA_LADDER_ETAS.tolist(),
            "eta_ladder_note": (
                "frozen at {8,4,2,1}x1e-4 eV; an e-3 ladder violates the "
                "frozen final-rung budgets"
            ),
            "extrapolation": (
                "two smallest rungs, linear in eta: X(0) ~ 2*X(eta_min)"
                " - X(2*eta_min); absolute budget 2e-6"
            ),
            "final_rung_budgets": {
                "captured_mass_change": 6.5e-5,
                "hwhm_change_ev": 1.01e-4,
            },
            "gamma_physical_ev": ETA_LADDER_GAMMA_PHYSICAL,
            "hwhm_definition": (
                "three-point quadratic vertex around the grid argmax gives "
                "peak position and height; the half level is half the "
                "vertex height; each crossing is linearly interpolated "
                "between the bracketing frozen-grid samples; HWHM is half "
                "the crossing separation"
            ),
            "level_energy_ev": ETA_LADDER_LEVEL_EV,
            "omega_window_ev": [
                float(ETA_LADDER_OMEGA[0]),
                float(ETA_LADDER_OMEGA[-1]),
            ],
            "peak_position_budget_ev": 2.0e-6,
            "per_rung_tolerance_rtol": 1.0e-10,
            "points": int(ETA_LADDER_OMEGA.size),
        },
        "fermi_counterexample": {
            "band_energy_ev": FERMI_BAND_EV,
            "boltzmann_ev_per_k": KB_EV_PER_K,
            "definition": (
                "I_correct = A(omega)*f_FD(omega, T) with occupation at "
                "the sampled omega; I_wrong = A(omega)*f_FD(E0, T) with "
                "occupation at the band eigenvalue; the occupied-side "
                "tail survives only in I_correct"
            ),
            "gamma_ev": FERMI_GAMMA,
            "max_relative_discrepancy_floor": 1.0e12,
            "omega_window_ev": [
                float(FERMI_OMEGA[0]),
                float(FERMI_OMEGA[-1]),
            ],
            "points": int(FERMI_OMEGA.size),
            "temperature_k": FERMI_TEMPERATURE_K,
        },
        "generator": (
            "tests/_reference_tools/generate_spectral_intensity_reference.py"
        ),
        "generator_sha256": _sha256(generator_path),
        "hermitian_resolvent": {
            "eta_ev": HERMITIAN_ETA,
            "matrix_a_ev": [
                ["0.2+0.1j", "-0.3+0.7j"],
                ["0.4-0.2j", "-0.1+0.05j"],
            ],
            "omega_window_ev": [
                float(HERMITIAN_OMEGA[0]),
                float(HERMITIAN_OMEGA[-1]),
            ],
            "points": int(HERMITIAN_OMEGA.size),
            "sigma_ev": "-0.02j",
            "source": ["1+0.3j", "-0.4+0.8j"],
            "tolerance_rtol": 1.0e-10,
            "truth_engine": "numpy.linalg.eigh dense eigendecomposition",
        },
        "reference_engine": {
            "name": "numpy",
            "value_functions": [
                "numpy.linalg.eigh",
                "numpy.linalg.solve",
                "numpy.polynomial.legendre.leggauss",
                "analytic arctangent/Lorentzian closed forms",
            ],
        },
        "requirements": [
            "two-pole-closed-form",
            "hermitian-resolvent-eigh-agreement",
            "finite-window-arctangent-mass",
            "lorentzian-linewidth-ladder",
            "resolvent-adjoint-derivative-truth",
            "differentiated-coordinate-registry",
            "sampled-energy-occupation-witness",
        ],
        "schema": "diffpes.spectral-intensity-reference.v1",
        "stage": "preregistered-before-spectral-intensity-production-edit",
        "sum_rule": {
            "fullline_map": (
                "E = center + s*tan(pi*u/2), Jacobian s*pi/2*sec(pi*u/2)^2,"
                " s = Gamma_total, mirroring the frozen true-Voigt "
                "convention"
            ),
            "orders": list(SUM_RULE_QUADRATURE_ORDERS),
            "row_labels": list(sum_rule_labels),
            "tolerance_rtol": 1.0e-8,
            "window_ev": list(SUM_RULE_WINDOW),
            "window_mass": (
                "(1/pi)*(arctan((w2-E)/Gamma) - arctan((w1-E)/Gamma))"
            ),
        },
        "two_pole": {
            "closed_form": (
                "I(omega) = sum_pm |m1 +/- m2|^2/2 * (1/pi) * Gamma_total"
                " / ((omega - (eps0 +/- t))^2 + Gamma_total^2)"
            ),
            "degenerate_limit": "t = 0 exactly; single pole at eps0",
            "epsilon0_ev": TWO_POLE_EPSILON0,
            "eta_ev": TWO_POLE_ETA,
            "gamma_ev": TWO_POLE_GAMMA,
            "hopping_ev": TWO_POLE_HOPPING,
            "omega_window_ev": [
                float(TWO_POLE_OMEGA[0]),
                float(TWO_POLE_OMEGA[-1]),
            ],
            "points": int(TWO_POLE_OMEGA.size),
            "sigma": "constant Sigma = -i*Gamma",
            "source": ["0.9+0.4j", "-0.5+0.7j"],
            "tolerance_rtol": 1.0e-10,
        },
    }


def main() -> None:
    """Write the frozen spectral-intensity archive and provenance manifest."""
    root: Path = Path(__file__).resolve().parents[2]
    data_directory: Path = root / "tests" / "test_diffpes" / "_reference_data"
    data_directory.mkdir(parents=True, exist_ok=True)
    reference_path: Path = data_directory / "spectral_intensity_reference.npz"
    manifest_path: Path = data_directory / "spectral_intensity_manifest.json"

    payload: Dict[str, Float64[NDArray, "..."]]
    sum_rule_labels: Tuple[str, ...]
    payload, sum_rule_labels = _reference_payload()
    _write_deterministic_npz(reference_path, payload)

    generator_path: Path = Path(__file__).resolve()
    manifest: Dict[str, Any] = _manifest(
        reference_path,
        generator_path,
        sum_rule_labels,
    )
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
