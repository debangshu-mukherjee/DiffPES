"""Generate the independent Plan-07 G2/D1 Voigt evidence artifacts.

The generator uses only NumPy and SciPy for numerical truth.  In particular,
it does not import DiffPES or evaluate either production target.  The novice
fixture inputs are frozen from the retained seed-20260713 factory realization
so the true-Voigt migration changes only the registered line shape.
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
from scipy import special

CENTER: float = 0.137
WIDTHS: np.ndarray = np.asarray(
    [1.0e-6, 1.0e-4, 1.0e-2, 1.0e-1, 1.0, 10.0],
    dtype=np.float64,
)
Q_VALUES: np.ndarray = np.asarray(
    [-12.0, -5.0, -2.0, -1.0, -0.25, 0.0, 0.5, 1.5, 5.0, 12.0],
    dtype=np.float64,
)
ANCHORS: np.ndarray = np.asarray([1.0e-6, 0.2, 10.0], dtype=np.float64)
EPSILONS: np.ndarray = np.asarray(
    [2.0**-8, 2.0**-10, 2.0**-12],
    dtype=np.float64,
)
QUADRATURE_ORDERS: tuple[int, int] = (256, 512)
NORMALIZATION_WIDTHS: np.ndarray = np.asarray(
    [
        (1.0e-6, 1.0e-6),
        (1.0e-4, 1.0e-2),
        (1.0e-1, 2.0e-1),
        (10.0, 10.0),
        (1.0e-2, 10.0),
        (10.0, 1.0e-6),
        (1.0e-6, 0.0),
        (10.0, 0.0),
        (0.0, 1.0e-6),
        (0.0, 10.0),
    ],
    dtype=np.float64,
)
ENVELOPE_SIGMA: float = 1.0e-6
ENVELOPE_GAMMA: float = 2.0e-6
ENVELOPE_RADII: np.ndarray = np.asarray(
    [0.99e8, 1.0e8, 1.01e8],
    dtype=np.float64,
)
D1_PROBES: np.ndarray = np.asarray(
    [
        (CENTER, 1.0e-6, 3.0e-6),
        (CENTER, 0.3, 0.02),
        (CENTER, 0.03, 0.2),
        (CENTER, 10.0, 10.0),
    ],
    dtype=np.float64,
)
D1_Q_VALUES: np.ndarray = np.asarray(
    [-2.3, -0.7, -0.1, 0.4, 1.8, 3.1],
    dtype=np.float64,
)
D1_WEIGHTS: np.ndarray = (
    np.asarray([0.7, 1.1, 0.4, 1.6, 0.9, 1.3], dtype=np.float64) / 6.0
)
D1_STEPS: np.ndarray = np.asarray(
    [2.0**-12, 2.0**-14, 2.0**-16],
    dtype=np.float64,
)

# These two compact arrays are the exact seed-20260713 factory realization.
# Freezing the realized inputs avoids importing JAX or any production code.
NOVICE_EIGENVALUES: np.ndarray = np.asarray(
    [
        [
            -1.7431084408649649,
            -1.6279458915048952,
            -0.33372591971692045,
            0.20985390567214862,
        ],
        [
            -2.151169280951573,
            -2.067154900276044,
            -0.8436468645092519,
            -0.23498178372311823,
        ],
        [
            -2.26835980029603,
            -2.1624686520839598,
            -2.033839297375634,
            -1.7738799533890963,
        ],
        [
            -1.2556999045083308,
            -1.1289924318544295,
            -0.6609282873583171,
            0.008383287543690776,
        ],
        [
            -2.1369924519687697,
            -1.75669932933459,
            -0.9498543488916997,
            -0.2093868937434501,
        ],
        [
            -1.755101893442041,
            -1.282871696980493,
            -0.5015744310847319,
            0.12632490012588943,
        ],
        [
            -1.927200156047657,
            -1.401244770556664,
            -1.3253444529370153,
            0.16335537715774517,
        ],
        [
            -1.9155878350508053,
            -0.6084300499139346,
            -0.3444576908169674,
            -0.2796254296460949,
        ],
    ],
    dtype=np.float64,
)
NOVICE_BAND_WEIGHTS: np.ndarray = np.asarray(
    [
        [
            0.817011035192613,
            0.8914761747456353,
            0.9343754276950469,
            0.902568818537731,
        ],
        [
            0.8599828025296482,
            0.8694913830074208,
            0.8393371873020797,
            0.8804653583237264,
        ],
        [
            0.8754279306937602,
            0.9240024529494595,
            0.8822441527339192,
            0.9187390133218558,
        ],
        [
            0.9573036017968908,
            0.937836688250868,
            0.877930680229983,
            0.9296038502535178,
        ],
        [
            0.9409557916768478,
            0.9281170064769995,
            0.8803444010002437,
            0.876344947594615,
        ],
        [
            0.9495673333649742,
            0.8650168849847819,
            0.9022991778711456,
            0.9611516330354464,
        ],
        [
            0.8872139554579251,
            0.8600635990084143,
            0.8751689000342013,
            0.8771844674331915,
        ],
        [
            0.8658948255408949,
            0.8770650318680764,
            0.9239044777669472,
            0.911703584651073,
        ],
    ],
    dtype=np.float64,
)
NOVICE_SEED: int = 20260713
NOVICE_SIGMA: float = 0.04
NOVICE_GAMMA: float = 0.1
NOVICE_TEMPERATURE: float = 15.0
KB_EV_PER_K: float = 8.617333e-5


def _array_bytes(array: np.ndarray) -> bytes:
    """Serialize one array without pickle or filesystem metadata."""
    output: io.BytesIO = io.BytesIO()
    np.lib.format.write_array(output, np.asarray(array), allow_pickle=False)
    return output.getvalue()


def _write_deterministic_npz(
    path: Path,
    arrays: dict[str, np.ndarray],
) -> None:
    """Write a sorted float64 NPZ with fixed timestamps and permissions."""
    with zipfile.ZipFile(
        path,
        mode="w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=9,
    ) as archive:
        name: str
        array: np.ndarray
        for name, array in sorted(arrays.items()):
            normalized: np.ndarray = np.asarray(array, dtype=np.float64)
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
    """Return the SHA-256 digest of one evidence file."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _positive_payload() -> dict[str, np.ndarray]:
    """Build the complete positive-width SciPy value table."""
    width_rows: np.ndarray = np.asarray(
        [(sigma, gamma) for sigma in WIDTHS for gamma in WIDTHS],
        dtype=np.float64,
    )
    scales: np.ndarray = np.max(width_rows, axis=1)
    energies: np.ndarray = CENTER + scales[:, None] * Q_VALUES[None, :]
    values: np.ndarray = np.stack(
        [
            special.voigt_profile(energy - CENTER, sigma, gamma)
            for energy, (sigma, gamma) in zip(
                energies, width_rows, strict=True
            )
        ]
    )
    z_values: np.ndarray = (
        energies - CENTER + 1j * width_rows[:, 1, None]
    ) / (width_rows[:, 0, None] * np.sqrt(2.0))
    if np.max(np.abs(z_values)) > 1.0e8:
        raise RuntimeError("positive-width table leaves the Faddeeva envelope")
    return {
        "positive_energies": energies,
        "positive_values": values,
        "positive_widths": width_rows,
        "positive_z_imag": z_values.imag,
        "positive_z_real": z_values.real,
    }


def _endpoint_payload() -> dict[str, np.ndarray]:
    """Build the exact value-only Gaussian and Cauchy endpoint rows."""
    width_rows: list[tuple[float, float]] = []
    energies: list[np.ndarray] = []
    values: list[np.ndarray] = []
    anchor: float
    for anchor in ANCHORS:
        energy: np.ndarray = CENTER + anchor * Q_VALUES
        q_hat: np.ndarray = (energy - CENTER) / anchor
        gaussian: np.ndarray = np.exp(-(q_hat**2) / 2.0) / (
            anchor * np.sqrt(2.0 * np.pi)
        )
        cauchy: np.ndarray = 1.0 / (np.pi * anchor * (1.0 + q_hat**2))
        width_rows.extend(((anchor, 0.0), (0.0, anchor)))
        energies.extend((energy, energy))
        values.extend((gaussian, cauchy))
    return {
        "endpoint_energies": np.asarray(energies, dtype=np.float64),
        "endpoint_values": np.asarray(values, dtype=np.float64),
        "endpoint_widths": np.asarray(width_rows, dtype=np.float64),
    }


def _one_sided_payload() -> dict[str, np.ndarray]:
    """Build positive rungs and registered endpoint-convergence rates."""
    width_rows: list[tuple[float, float]] = []
    energies: list[np.ndarray] = []
    values: list[np.ndarray] = []
    endpoint_values: list[np.ndarray] = []
    differences: np.ndarray = np.empty((2, ANCHORS.size, EPSILONS.size))
    direction: int
    anchor_index: int
    anchor: float
    for direction in range(2):
        for anchor_index, anchor in enumerate(ANCHORS):
            energy: np.ndarray = CENTER + anchor * Q_VALUES
            q_hat: np.ndarray = (energy - CENTER) / anchor
            endpoint: np.ndarray
            if direction == 0:
                endpoint = 1.0 / (np.pi * anchor * (1.0 + q_hat**2))
            else:
                endpoint = np.exp(-(q_hat**2) / 2.0) / (
                    anchor * np.sqrt(2.0 * np.pi)
                )
            epsilon_index: int
            epsilon: float
            for epsilon_index, epsilon in enumerate(EPSILONS):
                sigma: float
                gamma: float
                if direction == 0:
                    sigma, gamma = anchor * epsilon, anchor
                else:
                    sigma, gamma = anchor, anchor * epsilon
                reference: np.ndarray = special.voigt_profile(
                    energy - CENTER,
                    sigma,
                    gamma,
                )
                width_rows.append((sigma, gamma))
                energies.append(energy)
                values.append(reference)
                endpoint_values.append(endpoint)
                differences[direction, anchor_index, epsilon_index] = (
                    anchor * np.max(np.abs(reference - endpoint))
                )
    ratios: np.ndarray = differences[..., :-1] / differences[..., 1:]
    if not np.all(np.diff(differences, axis=-1) < 0.0):
        raise RuntimeError("one-sided endpoint differences are not decreasing")
    if not np.all((ratios[0] >= 15.5) & (ratios[0] <= 16.5)):
        raise RuntimeError(
            "sigma-to-zero convergence rate is outside its gate"
        )
    if not np.all((ratios[1] >= 3.9) & (ratios[1] <= 4.1)):
        raise RuntimeError(
            "gamma-to-zero convergence rate is outside its gate"
        )
    return {
        "onesided_differences": differences,
        "onesided_energies": np.asarray(energies, dtype=np.float64),
        "onesided_endpoint_values": np.asarray(
            endpoint_values,
            dtype=np.float64,
        ),
        "onesided_epsilons": EPSILONS,
        "onesided_ratios": ratios,
        "onesided_values": np.asarray(values, dtype=np.float64),
        "onesided_widths": np.asarray(width_rows, dtype=np.float64),
    }


def _profile(
    energy: np.ndarray,
    center: float,
    sigma: float,
    gamma: float,
) -> np.ndarray:
    """Evaluate a SciPy Voigt profile, including either analytic endpoint."""
    displacement: np.ndarray = energy - center
    if gamma == 0.0:
        return np.exp(-((displacement / sigma) ** 2) / 2.0) / (
            sigma * np.sqrt(2.0 * np.pi)
        )
    if sigma == 0.0:
        return gamma / (np.pi * (displacement**2 + gamma**2))
    return special.voigt_profile(displacement, sigma, gamma)


def _normalization_payload() -> dict[str, np.ndarray]:
    """Evaluate the frozen scaled full-line quadrature battery."""
    masses: np.ndarray = np.empty(
        (NORMALIZATION_WIDTHS.shape[0], len(QUADRATURE_ORDERS)),
        dtype=np.float64,
    )
    maximum_z: np.ndarray = np.zeros(NORMALIZATION_WIDTHS.shape[0])
    row_index: int
    sigma: float
    gamma: float
    for row_index, (sigma, gamma) in enumerate(NORMALIZATION_WIDTHS):
        scale: float = (
            sigma + gamma if sigma > 0.0 and gamma > 0.0 else sigma + gamma
        )
        order_index: int
        order: int
        for order_index, order in enumerate(QUADRATURE_ORDERS):
            nodes: np.ndarray
            weights: np.ndarray
            nodes, weights = np.polynomial.legendre.leggauss(order)
            angle: np.ndarray = np.pi * nodes / 2.0
            energy: np.ndarray = CENTER + scale * np.tan(angle)
            jacobian: np.ndarray = scale * np.pi / 2.0 / np.cos(angle) ** 2
            if sigma > 0.0 and gamma > 0.0:
                z_values: np.ndarray = (energy - CENTER + 1j * gamma) / (
                    sigma * np.sqrt(2.0)
                )
                maximum_z[row_index] = max(
                    maximum_z[row_index],
                    float(np.max(np.abs(z_values))),
                )
            masses[row_index, order_index] = np.sum(
                weights * _profile(energy, CENTER, sigma, gamma) * jacobian
            )
    if np.any(maximum_z[:6] > 1.0e8):
        raise RuntimeError("normalization nodes leave the Faddeeva envelope")
    if np.any(np.abs(masses - 1.0) > 2.0e-10):
        raise RuntimeError("normalization mass leaves its absolute gate")
    if np.any(np.abs(masses[:, 1] - masses[:, 0]) > 2.0e-10):
        raise RuntimeError("normalization order delta leaves its gate")
    return {
        "normalization_masses": masses,
        "normalization_maximum_z": maximum_z,
        "normalization_orders": np.asarray(
            QUADRATURE_ORDERS,
            dtype=np.float64,
        ),
        "normalization_scales": np.sum(
            NORMALIZATION_WIDTHS,
            axis=1,
        ),
        "normalization_widths": NORMALIZATION_WIDTHS,
    }


def _envelope_payload() -> dict[str, np.ndarray]:
    """Build the three closed-envelope pass/reject boundary rows."""
    deltas: np.ndarray = np.sqrt(
        2.0 * (ENVELOPE_SIGMA * ENVELOPE_RADII) ** 2 - ENVELOPE_GAMMA**2
    )
    energies: np.ndarray = np.stack(
        (
            CENTER - deltas,
            np.full_like(deltas, CENTER),
            CENTER + deltas,
        ),
        axis=1,
    )
    reconstructed: np.ndarray = np.max(
        np.abs(
            (energies - CENTER + 1j * ENVELOPE_GAMMA)
            / (ENVELOPE_SIGMA * np.sqrt(2.0))
        ),
        axis=1,
    )
    error: np.ndarray = np.abs(reconstructed - ENVELOPE_RADII)
    if np.any(error > 4.0 * np.finfo(np.float64).eps * ENVELOPE_RADII):
        raise RuntimeError("shared-envelope radius reconstruction is unstable")
    return {
        "envelope_deltas": deltas,
        "envelope_energies": energies,
        "envelope_radii": ENVELOPE_RADII,
        "envelope_reconstructed_radii": reconstructed,
        "envelope_widths": np.asarray(
            [ENVELOPE_SIGMA, ENVELOPE_GAMMA],
            dtype=np.float64,
        ),
    }


def _d1_loss(
    parameters: np.ndarray,
    probe: np.ndarray,
    energies: np.ndarray,
) -> float:
    """Evaluate the dimensionless contracted SciPy D1 loss."""
    center: float
    sigma: float
    gamma: float
    center, sigma, gamma = probe
    scale: float = max(sigma, gamma)
    shifted_center: float = center + scale * parameters[0]
    shifted_sigma: float = sigma * (1.0 + parameters[1])
    shifted_gamma: float = gamma * (1.0 + parameters[2])
    values: np.ndarray = special.voigt_profile(
        energies - shifted_center,
        shifted_sigma,
        shifted_gamma,
    )
    return float(scale * np.sum(D1_WEIGHTS * values))


def _five_point_derivative(
    probe: np.ndarray,
    energies: np.ndarray,
    coordinate: int,
    step: float,
) -> float:
    """Evaluate one symmetric five-point D1 derivative."""
    delta: np.ndarray = np.zeros(3, dtype=np.float64)
    delta[coordinate] = step
    return (
        _d1_loss(-2.0 * delta, probe, energies)
        - 8.0 * _d1_loss(-delta, probe, energies)
        + 8.0 * _d1_loss(delta, probe, energies)
        - _d1_loss(2.0 * delta, probe, energies)
    ) / (12.0 * step)


def _d1_payload() -> dict[str, np.ndarray]:
    """Build analytic point derivatives and contracted D1 truth."""
    energies: np.ndarray = np.empty((D1_PROBES.shape[0], D1_Q_VALUES.size))
    point_values: np.ndarray = np.empty_like(energies)
    point_derivatives: np.ndarray = np.empty(
        (D1_PROBES.shape[0], D1_Q_VALUES.size, 3)
    )
    contracted: np.ndarray = np.empty((D1_PROBES.shape[0], 3))
    finite_difference: np.ndarray = np.empty(
        (D1_PROBES.shape[0], D1_STEPS.size, 3)
    )
    probe_index: int
    probe: np.ndarray
    for probe_index, probe in enumerate(D1_PROBES):
        center: float
        sigma: float
        gamma: float
        center, sigma, gamma = probe
        scale: float = max(sigma, gamma)
        energy: np.ndarray = center + scale * D1_Q_VALUES
        z_values: np.ndarray = (energy - center + 1j * gamma) / (
            sigma * np.sqrt(2.0)
        )
        w_values: np.ndarray = special.wofz(z_values)
        w_prime: np.ndarray = -2.0 * z_values * w_values + 2j / np.sqrt(np.pi)
        prefactor: float = 1.0 / (sigma * np.sqrt(2.0 * np.pi))
        values: np.ndarray = np.real(w_values) * prefactor
        derivatives: np.ndarray = np.stack(
            (
                prefactor * np.real(w_prime * (-1.0 / (sigma * np.sqrt(2.0)))),
                -values / sigma
                + prefactor * np.real(w_prime * (-z_values / sigma)),
                prefactor * np.real(w_prime * (1j / (sigma * np.sqrt(2.0)))),
            ),
            axis=1,
        )
        energies[probe_index] = energy
        point_values[probe_index] = values
        point_derivatives[probe_index] = derivatives
        contracted[probe_index] = np.asarray(
            [
                scale**2 * np.sum(D1_WEIGHTS * derivatives[:, 0]),
                scale * sigma * np.sum(D1_WEIGHTS * derivatives[:, 1]),
                scale * gamma * np.sum(D1_WEIGHTS * derivatives[:, 2]),
            ]
        )
        step_index: int
        step: float
        for step_index, step in enumerate(D1_STEPS):
            coordinate: int
            for coordinate in range(3):
                finite_difference[probe_index, step_index, coordinate] = (
                    _five_point_derivative(
                        probe,
                        energy,
                        coordinate,
                        step,
                    )
                )
    median: np.ndarray = np.median(finite_difference, axis=1)
    spread: np.ndarray = np.ptp(finite_difference, axis=1)
    tolerance: np.ndarray = 2.0e-10 + 1.0e-6 * np.abs(contracted)
    if np.any(np.abs(median - contracted) > tolerance):
        raise RuntimeError("D1 five-point median leaves the analytic gate")
    if np.any(spread > tolerance):
        raise RuntimeError("D1 five-point spread leaves the analytic gate")
    if np.any(np.abs(contracted) <= 1.0e-4):
        raise RuntimeError("D1 contracted coordinate lacks sensitivity")
    return {
        "d1_analytic_contracted": contracted,
        "d1_energies": energies,
        "d1_fd_estimates": finite_difference,
        "d1_fd_steps": D1_STEPS,
        "d1_point_derivatives": point_derivatives,
        "d1_point_values": point_values,
        "d1_probes": D1_PROBES,
        "d1_q_values": D1_Q_VALUES,
        "d1_weights": D1_WEIGHTS,
    }


def _stable_fermi(energy: np.ndarray) -> np.ndarray:
    """Evaluate the analytic Fermi function without overflow."""
    exponent: np.ndarray = energy / (KB_EV_PER_K * NOVICE_TEMPERATURE)
    occupation: np.ndarray = np.empty_like(exponent)
    positive: np.ndarray = exponent >= 0.0
    decaying: np.ndarray = np.exp(-exponent[positive])
    occupation[positive] = decaying / (1.0 + decaying)
    occupation[~positive] = 1.0 / (1.0 + np.exp(exponent[~positive]))
    return occupation


def _novice_payload() -> dict[str, np.ndarray]:
    """Manually assemble the fixed-seed true-Voigt novice spectrum."""
    energy_axis: np.ndarray = np.linspace(-3.0, 0.5, 512, dtype=np.float64)
    occupations: np.ndarray = _stable_fermi(NOVICE_EIGENVALUES)
    profiles: np.ndarray = special.voigt_profile(
        energy_axis[None, None, :] - NOVICE_EIGENVALUES[..., None],
        NOVICE_SIGMA,
        NOVICE_GAMMA,
    )
    intensity: np.ndarray = np.sum(
        NOVICE_BAND_WEIGHTS[..., None] * occupations[..., None] * profiles,
        axis=1,
    )
    return {
        "leaf_000_intensity": intensity,
        "leaf_001_energy_axis": energy_axis,
    }


def _reference_payload() -> dict[str, np.ndarray]:
    """Assemble every G2/D1 reference table into one float64 payload."""
    payload: dict[str, np.ndarray] = {
        "anchors": ANCHORS,
        "center": np.asarray(CENTER, dtype=np.float64),
        "novice_band_weights": NOVICE_BAND_WEIGHTS,
        "novice_eigenvalues": NOVICE_EIGENVALUES,
        "positive_width_grid": WIDTHS,
        "q_values": Q_VALUES,
    }
    payload.update(_positive_payload())
    payload.update(_endpoint_payload())
    payload.update(_one_sided_payload())
    payload.update(_normalization_payload())
    payload.update(_envelope_payload())
    payload.update(_d1_payload())
    return payload


def main() -> None:
    """Write the independent artifacts, archive, and provenance manifest."""
    root: Path = Path(__file__).resolve().parents[2]
    data_directory: Path = root / "tests" / "test_diffpes" / "_reference_data"
    data_directory.mkdir(parents=True, exist_ok=True)
    reference_path: Path = data_directory / "plan07_voigt_scipy_reference.npz"
    novice_path: Path = data_directory / "novice_toy_plan07_true_voigt.npz"
    active_plan02_path: Path = data_directory / "novice_toy.npz"
    historical_path: Path = (
        data_directory / "novice_toy_plan02_pseudo_voigt.npz"
    )
    manifest_path: Path = data_directory / "plan07_voigt_manifest.json"

    reference_payload: dict[str, np.ndarray] = _reference_payload()
    novice_payload: dict[str, np.ndarray] = _novice_payload()
    _write_deterministic_npz(reference_path, reference_payload)
    _write_deterministic_npz(novice_path, novice_payload)
    historical_path.write_bytes(active_plan02_path.read_bytes())

    generator_path: Path = Path(__file__).resolve()
    manifest: dict[str, Any] = {
        "archives": {
            "historical_plan02": {
                "classification": (
                    "superseded pseudo-Voigt evidence; not a compatibility shim"
                ),
                "filename": historical_path.name,
                "sha256": _sha256(historical_path),
            },
            "novice_plan07": {
                "filename": novice_path.name,
                "sha256": _sha256(novice_path),
            },
            "voigt_reference": {
                "filename": reference_path.name,
                "sha256": _sha256(reference_path),
            },
        },
        "d1": {
            "analytic_source": (
                "scipy.special.wofz and w_prime=-2*z*w+2j/sqrt(pi)"
            ),
            "check_grads_eps": 2.0**-14,
            "fd_steps": D1_STEPS.tolist(),
            "tolerance": {"absolute": 2.0e-10, "relative": 1.0e-6},
        },
        "environment": {
            "numpy": np.__version__,
            "platform": platform.platform(),
            "python": platform.python_version(),
            "scipy": scipy.__version__,
        },
        "gates": ["07.G2", "07.D1"],
        "generator": (
            "tests/_reference_tools/generate_plan07_voigt_reference.py"
        ),
        "generator_sha256": _sha256(generator_path),
        "normalization": {
            "absolute_mass_bound": 2.0e-10,
            "map": "E=c+s*tan(pi*u/2)",
            "orders": list(QUADRATURE_ORDERS),
            "positive_scale": "sigma+gamma",
        },
        "novice": {
            "fermi": (
                "stable analytic reciprocal exponential, KB=8.617333e-5 eV/K"
            ),
            "fidelity": 512,
            "seed": NOVICE_SEED,
            "temperature_kelvin": NOVICE_TEMPERATURE,
            "truth": "scipy.special.voigt_profile",
        },
        "reference_engine": {
            "name": "scipy.special",
            "value_function": "voigt_profile",
        },
        "schema": "diffpes.plan07.voigt-reference.v1",
        "stage": "preregistered-before-WP7.2-production-edit",
        "value_bounds": {
            "endpoint": "5e-15/nonzero_width + 1e-12*abs(reference)",
            "positive": ("2e-15/(sigma*sqrt(2*pi)) + 1e-10*abs(reference)"),
        },
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
