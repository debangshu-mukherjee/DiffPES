"""Provide shared fixtures for the mirrored split test modules.

The helpers preserve the inputs and independent reference calculations.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
from types import ModuleType

import jax
import jax.numpy as jnp
import numpy as np
from beartype.typing import Any, Callable, Dict, Tuple
from jaxtyping import Array, Float64
from numpy.typing import NDArray

from diffpes.types import (
    SelfEnergyModel,
    make_self_energy_model,
)

_REFERENCE_DIRECTORY: Path = (
    Path(__file__).resolve().parents[1] / "_reference_data"
)


_TOOLS_DIRECTORY: Path = (
    Path(__file__).resolve().parents[2] / "_reference_tools"
)


_ANALYTIC_DIRECTORY: Path = _REFERENCE_DIRECTORY / "kk_analytic_reference"


_SELECTION_MANIFEST_PATH: Path = (
    _REFERENCE_DIRECTORY / "kk_operator_selection_manifest.json"
)


_SELECTION_ARCHIVE_PATH: Path = (
    _REFERENCE_DIRECTORY / "kk_operator_selection_reference.npz"
)


_MODELS_MANIFEST_PATH: Path = (
    _REFERENCE_DIRECTORY / "self_energy_models_manifest.json"
)


_MODELS_ARCHIVE_PATH: Path = (
    _REFERENCE_DIRECTORY / "self_energy_models_reference.npz"
)


_SPECTRAL_INTENSITY_MANIFEST_PATH: Path = (
    _REFERENCE_DIRECTORY / "spectral_intensity_manifest.json"
)


_SPECTRAL_INTENSITY_ARCHIVE_PATH: Path = (
    _REFERENCE_DIRECTORY / "spectral_intensity_reference.npz"
)


_CHINOOK_SPECTRAL_MANIFEST_PATH: Path = (
    _REFERENCE_DIRECTORY / "chinook_spectral_manifest.json"
)


_CHINOOK_SPECTRAL_ARCHIVE_PATH: Path = (
    _REFERENCE_DIRECTORY / "chinook_spectral_reference.npz"
)


_DEGENERATE_WITNESS_PATH: Path = (
    _REFERENCE_DIRECTORY / "degenerate_k_gradient_witness.json"
)


_SELECTION_MANIFEST_SHA256: str = (
    "e0188fc51f6f61f7c94c30cbbd460fc784d2f5772544225b64e8015ad86ba7f5"
)


_SELECTION_ARCHIVE_SHA256: str = (
    "e827e91c62e294afc50112af5fe484e5ff002070f106511d07bb813502648430"
)


_ANALYTIC_MANIFEST_SHA256: str = (
    "e91e02d117c0b389e55e9505b3b5affac6780927ef39676a700bb5818078a14a"
)


_ANALYTIC_ARCHIVE_SHA256: str = (
    "ba30a3ee4e65658ace63ed54e65c3ec8ad8ae0868c396653123896265829cba5"
)


_ANALYTIC_GENERATOR_SHA256: str = (
    "563aaf92c94b3962dbdb63a2ce0121b0cf8192beea3fc833e983292fb086cbf8"
)


_MODELS_MANIFEST_SHA256: str = (
    "b72314c2587acbd61ee35cb81eeaa84d411562c82b597b23aae74e217551669b"
)


_MODELS_ARCHIVE_SHA256: str = (
    "59a115c16cbcbd57e7b70ec290380d224d2f2f73c5b6682afbcec047a4fe2830"
)


_MODELS_GENERATOR_SHA256: str = (
    "44d694b431703ae6b9a80af9e2f4a2a8cfab8dd12c660fc8bfd81db8e0b792ec"
)


_SELECTION_GENERATOR_SHA256: str = (
    "9bf4aa378886e3e135787fd50477a8da464ae0bc6a689b3bf6f99485f3e0da64"
)


_COMMON_MODULE_SHA256: str = (
    "2f401f171ac90b1bd1c1f75448aa1f06662d537e4a6301f0560a2c41d10303cb"
)


_CUBIC_MODULE_SHA256: str = (
    "1158e257f4700353ff49e7f170228f43b263c27025499e853002d65add2a1ea4"
)


_LINEAR_MODULE_SHA256: str = (
    "b8979970ed70200e89a0523c7528ddc51f27e25f279a9a86bcdd97a26be07af5"
)


_QUADRATIC_MODULE_SHA256: str = (
    "2dee8a85e8f4f4e24a457fcebf029c470ebc965377557ce6b614851bfa7b8e19"
)


_CONTROL_MODULE_SHA256: str = (
    "3d855cc5afbd84591b83a0d0c7136504d959c4e514ae26a30402f8601fd81920"
)


_SPECTRAL_INTENSITY_MANIFEST_SHA256: str = (
    "8edc1297a10ed66ad6058acf44c05b8512b55f1bd77fa15a97174a848fbd0673"
)


_SPECTRAL_INTENSITY_ARCHIVE_SHA256: str = (
    "67e5f9e18ad39a1b51e8ed5713003b684f35fff60df2637e9dd16e90f6ca547c"
)


_CHINOOK_SPECTRAL_MANIFEST_SHA256: str = (
    "48b3020c51f01b89506923cfcd32cb48093cdfe645650fc69c6859db3b7bfedc"
)


_CHINOOK_SPECTRAL_ARCHIVE_SHA256: str = (
    "5a6163b2566e09de2974873eea1bf6062782b3af41e3225fa1fe26eca2859c56"
)


_DEGENERATE_WITNESS_SHA256: str = (
    "60eed816e8c693c4547e351ae2504c8a0e7e05422c6e6d56d09adba41e81891b"
)


_PAIR_TRUTH_ATOL_EV: float = 2.0e-8


_PAIR_TRUTH_RTOL: float = 1.0e-6


_PAIR_TRUTH_EXPECTED_MAX_ERROR_EV: float = 3.393963743381079e-08


_PAIR_TRUTH_EXPECTED_MAX_RATIO: float = 0.8239596852267533


_COMPOSITE_EXPECTED_MAX_ERROR: float = 9.342723950034326e-07


_COMPOSITE_DERIVATIVE_BOUND: float = 2.0e-6


_TAIL_RULE_BOUND: float = 1.0e-13


_IDENTITY_SCALE_BOUND: float = 1.0e-9


_QUERY_INVARIANCE_ATOL_EV: float = 1.0e-15


_GRID_EXACTNESS_ATOL_EV: float = 1.0e-12


_GRID_EXACTNESS_RTOL: float = 1.0e-12


_FL_VALUE_ATOL_EV: float = 2.0e-8


_FL_VALUE_RTOL: float = 1.0e-6


_FL_REAL_ROW_ATOL: float = 5.0e-8


_DERIVATIVE_ROW_RTOL: float = 1.0e-6


_ANALYTIC_ROW_ATOL: float = 1.0e-10


_DOMAIN_LOW_EV: float = -8.0


_DOMAIN_HIGH_EV: float = 8.0


_N_KK: int = 4096


_N_TAIL: int = 256


_BASE_SPACING_EV: float = (_DOMAIN_HIGH_EV - _DOMAIN_LOW_EV) / (_N_KK - 1)


_POLE_OMEGA0_EV: float = 0.35


_POLE_GAMMA_EV: float = 0.20


_POLE_COUPLING_EV2: float = 0.12


_POLE_TAIL_RAW: Tuple[float, float] = (
    -11.70907030862417,
    -11.359064962331182,
)


_FL_PARAMETERS_PHYSICAL: Tuple[float, float, float] = (0.02, 0.5, 0.8)


_FL_TAIL_RAW: Tuple[float, float] = (
    -12.270842147353243,
    -12.270842147353243,
)


_KINK_PARAMETERS_PHYSICAL: Tuple[float, float, float, float] = (
    0.015,
    0.35,
    0.07,
    0.02,
)


_COMMITTED_MODULE_CACHE: Dict[str, ModuleType] = {}


def _sha256(path: Path) -> str:
    """PRIVATE: Return the SHA-256 digest of one evidence file.

    Notes
    -----
    The digest streams the complete file bytes in one read.
    """
    returned: str = hashlib.sha256(path.read_bytes()).hexdigest()
    return returned


def _load_npz(path: Path) -> Dict[str, Float64[NDArray, "..."]]:
    """PRIVATE: Load one inert NPZ into ordinary arrays without pickle.

    Notes
    -----
    The loader forbids pickle and copies each named array.
    """
    archive: Any
    with np.load(path, allow_pickle=False) as archive:
        returned: Dict[str, Float64[NDArray, "..."]] = {
            name: archive[name] for name in archive.files
        }
        return returned


def _authenticated_json(path: Path, digest: str) -> Dict[str, Any]:
    """PRIVATE: Load one manifest after its SHA-256 digest matches the pin.

    Notes
    -----
    The check compares the digest before any parse.
    """
    assert _sha256(path) == digest, f"digest mismatch for {path.name}"
    returned: Dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return returned


def _authenticated_npz(
    path: Path, digest: str
) -> Dict[str, Float64[NDArray, "..."]]:
    """PRIVATE: Load one archive after its SHA-256 digest matches the pin.

    Notes
    -----
    The check compares the digest before any load.
    """
    assert _sha256(path) == digest, f"digest mismatch for {path.name}"
    returned: Dict[str, Float64[NDArray, "..."]] = _load_npz(path)
    return returned


def _committed_module(filename: str, digest: str) -> ModuleType:
    """PRIVATE: Import one committed instrument module after authentication.

    Notes
    -----
    The cache keys one import per authenticated filename.
    """
    if filename in _COMMITTED_MODULE_CACHE:
        returned: ModuleType = _COMMITTED_MODULE_CACHE[filename]
        return returned
    path: Path = _TOOLS_DIRECTORY / filename
    assert _sha256(path) == digest, f"digest mismatch for {filename}"
    spec: Any = importlib.util.spec_from_file_location(
        f"_kk_lane_{path.stem}", path
    )
    assert spec is not None
    assert spec.loader is not None
    module: ModuleType = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    _COMMITTED_MODULE_CACHE[filename] = module
    return module


def _committed_operator() -> Tuple[ModuleType, ModuleType]:
    """PRIVATE: Return the authenticated common and piecewise-cubic modules.

    Notes
    -----
    Both modules authenticate against their frozen digests.
    """
    common: ModuleType = _committed_module(
        "_kk_candidate_common.py", _COMMON_MODULE_SHA256
    )
    cubic: ModuleType = _committed_module(
        "_kk_candidate_piecewise_cubic.py", _CUBIC_MODULE_SHA256
    )
    returned: Tuple[ModuleType, ModuleType] = common, cubic
    return returned


def _base_grid() -> Float64[Array, " n_kk"]:
    """PRIVATE: Return the frozen uniform base quadrature grid in eV.

    Notes
    -----
    The grid follows the frozen index construction.
    """
    returned: Float64[Array, " n_kk"] = jnp.asarray(
        _DOMAIN_LOW_EV + _BASE_SPACING_EV * np.arange(_N_KK),
        dtype=jnp.float64,
    )
    return returned


def _cubic_edge_slopes(
    values: Float64[Array, " n_kk"],
    spacing: float,
) -> Tuple[Float64[Array, ""], Float64[Array, ""]]:
    """PRIVATE: Return the committed one-sided cubic edge-stencil slopes.

    Notes
    -----
    The one-sided four-node stencils match the production tails.
    """
    left: Float64[Array, ""] = (
        -11.0 * values[0]
        + 18.0 * values[1]
        - 9.0 * values[2]
        + 2.0 * values[3]
    ) / (6.0 * spacing)
    right: Float64[Array, ""] = (
        11.0 * values[-1]
        - 18.0 * values[-2]
        + 9.0 * values[-3]
        - 2.0 * values[-4]
    ) / (6.0 * spacing)
    returned: Tuple[Float64[Array, ""], Float64[Array, ""]] = left, right
    return returned


def _pole_sigma_imag(
    omega_rel_fermi_ev: Float64[Array, " n"],
) -> Float64[Array, " n"]:
    """PRIVATE: Evaluate the analytic retarded-pole imaginary part in eV.

    Notes
    -----
    The pole parameters come from the frozen fixture constants.
    """
    offset: Float64[Array, " n"] = omega_rel_fermi_ev - _POLE_OMEGA0_EV
    returned: Float64[Array, " n"] = (
        -_POLE_COUPLING_EV2 * _POLE_GAMMA_EV / (offset**2 + _POLE_GAMMA_EV**2)
    )
    return returned


def _pole_dsigma_imag(
    omega_rel_fermi_ev: Float64[Array, " n"],
) -> Float64[Array, " n"]:
    """PRIVATE: Evaluate the pole imaginary-part query derivative.

    Notes
    -----
    The formula differentiates the frozen pole closed form.
    """
    offset: Float64[Array, " n"] = omega_rel_fermi_ev - _POLE_OMEGA0_EV
    returned: Float64[Array, " n"] = (
        2.0
        * _POLE_COUPLING_EV2
        * _POLE_GAMMA_EV
        * offset
        / (offset**2 + _POLE_GAMMA_EV**2) ** 2
    )
    return returned


def _fl_sigma_imag_dynamic(
    omega_rel_fermi_ev: Float64[Array, " n"],
) -> Float64[Array, " n"]:
    """PRIVATE: Evaluate the Fermi-liquid dynamic imaginary remainder in eV.

    Notes
    -----
    The remainder excludes the constant baseline.
    """
    beta: float
    omega_c: float
    _, beta, omega_c = _FL_PARAMETERS_PHYSICAL
    returned: Float64[Array, " n"] = (
        -beta
        * omega_rel_fermi_ev**2
        / (1.0 + (omega_rel_fermi_ev / omega_c) ** 4)
    )
    return returned


def _fl_dsigma_imag_dynamic(
    omega_rel_fermi_ev: Float64[Array, " n"],
) -> Float64[Array, " n"]:
    """PRIVATE: Evaluate the query derivative of the dynamic remainder.

    Notes
    -----
    The formula differentiates the dynamic remainder analytically.
    """
    beta: float
    omega_c: float
    _, beta, omega_c = _FL_PARAMETERS_PHYSICAL
    quartic: Float64[Array, " n"] = (omega_rel_fermi_ev / omega_c) ** 4
    returned: Float64[Array, " n"] = (
        2.0
        * beta
        * omega_rel_fermi_ev
        * (quartic - 1.0)
        / (1.0 + quartic) ** 2
    )
    return returned


def _fl_dsigma_real_domega(
    omega_rel_fermi_ev: Float64[NDArray, " n"],
) -> Float64[NDArray, " n"]:
    """PRIVATE: Evaluate the Fermi-liquid real-part derivative.

    Notes
    -----
    The identity differentiates the analytic subtracted real part.
    """
    beta: float
    omega_c: float
    _, beta, omega_c = _FL_PARAMETERS_PHYSICAL
    omega: Float64[NDArray, " n"] = np.asarray(omega_rel_fermi_ev)
    quartic_sum: Float64[NDArray, " n"] = omega**4 + omega_c**4
    numerator: Float64[NDArray, " n"] = (
        3.0 * omega**2 - omega_c**2
    ) * quartic_sum - omega * (omega**2 - omega_c**2) * 4.0 * omega**3
    returned: Float64[NDArray, " n"] = (
        (np.sqrt(2.0) / 2.0) * beta * omega_c**3 * numerator / quartic_sum**2
    )
    return returned


def _pole_tail_spec(common: ModuleType) -> Any:
    """PRIVATE: Return the pole fixture's power2 tail parameters.

    Notes
    -----
    The spec derives from the frozen raw coordinates and stencils.
    """
    grid: Float64[Array, " n_kk"] = _base_grid()
    values: Float64[Array, " n_kk"] = _pole_sigma_imag(grid)
    slope_left: Float64[Array, ""]
    slope_right: Float64[Array, ""]
    slope_left, slope_right = _cubic_edge_slopes(values, _BASE_SPACING_EV)
    returned: Any = common.construct_power2_tail_spec(
        values[0],
        slope_left,
        values[-1],
        slope_right,
        _POLE_TAIL_RAW[0],
        _POLE_TAIL_RAW[1],
    )
    return returned


def _fl_tail_spec(common: ModuleType) -> Any:
    """PRIVATE: Return the Fermi-liquid power2 tail parameters.

    Notes
    -----
    The spec derives from the frozen raw coordinates and stencils.
    """
    grid: Float64[Array, " n_kk"] = _base_grid()
    values: Float64[Array, " n_kk"] = _fl_sigma_imag_dynamic(grid)
    slope_left: Float64[Array, ""]
    slope_right: Float64[Array, ""]
    slope_left, slope_right = _cubic_edge_slopes(values, _BASE_SPACING_EV)
    returned: Any = common.construct_power2_tail_spec(
        values[0],
        slope_left,
        values[-1],
        slope_right,
        _FL_TAIL_RAW[0],
        _FL_TAIL_RAW[1],
    )
    return returned


def _instrument_transform(
    sigma_imag_dynamic: Callable[[Float64[Array, " n"]], Float64[Array, " n"]],
    tail_spec: Any,
    queries_ev: Float64[Array, " n_query"],
) -> Float64[Array, " n_query"]:
    """PRIVATE: Evaluate the unsubtracted operator at arbitrary queries.

    Notes
    -----
    The instrument adds the cubic core and both tail quadratures.
    """
    common: ModuleType
    cubic: ModuleType
    common, cubic = _committed_operator()
    grid: Float64[Array, " n_kk"] = _base_grid()
    core: Float64[Array, " n_query"] = cubic.core_pv_transform(
        grid, sigma_imag_dynamic(grid), queries_ev
    )
    domain: Float64[Array, " 2"] = jnp.asarray(
        [_DOMAIN_LOW_EV, _DOMAIN_HIGH_EV], dtype=jnp.float64
    )
    tail: Float64[Array, " n_query"] = common.semi_infinite_tail_contribution(
        domain, tail_spec, queries_ev, _N_TAIL
    )
    returned: Float64[Array, " n_query"] = core + tail
    return returned


def _instrument_subtracted(
    sigma_imag_dynamic: Callable[[Float64[Array, " n"]], Float64[Array, " n"]],
    tail_spec: Any,
    queries_ev: Float64[Array, " n_query"],
    subtraction_point_ev: float,
) -> Float64[Array, " n_query"]:
    """PRIVATE: Evaluate the subtracted real part at arbitrary queries.

    Notes
    -----
    The subtraction reuses one stacked instrument evaluation.
    """
    stacked: Float64[Array, " n_plus_one"] = jnp.concatenate(
        [
            queries_ev,
            jnp.asarray([subtraction_point_ev], dtype=jnp.float64),
        ]
    )
    total: Float64[Array, " n_plus_one"] = _instrument_transform(
        sigma_imag_dynamic, tail_spec, stacked
    )
    returned: Float64[Array, " n_query"] = total[:-1] - total[-1]
    return returned


def _instrument_composite_derivative(
    sigma_imag_dynamic: Callable[[Float64[Array, " n"]], Float64[Array, " n"]],
    dsigma_imag_dynamic: Callable[
        [Float64[Array, " n"]], Float64[Array, " n"]
    ],
    tail_spec: Any,
    queries_ev: Float64[Array, " n_query"],
) -> Float64[Array, " n_query"]:
    """PRIVATE: Evaluate the committed composite query-derivative route.

    The route applies the committed operator to the analytic derivative
    of the dynamic imaginary part. It then adds the finite-core boundary
    terms ``(1/pi) * [Sigma''(a)/(a-omega) - Sigma''(b)/(b-omega)]``. It
    finally adds the exact forward-mode derivative of both semi-infinite
    tail quadratures.

    Notes
    -----
    The route matches the committed composite contract.
    """
    common: ModuleType
    cubic: ModuleType
    common, cubic = _committed_operator()
    grid: Float64[Array, " n_kk"] = _base_grid()
    core_derivative: Float64[Array, " n_query"] = cubic.core_pv_transform(
        grid, dsigma_imag_dynamic(grid), queries_ev
    )
    edge_low: Float64[Array, " 1"] = sigma_imag_dynamic(
        jnp.asarray([_DOMAIN_LOW_EV], dtype=jnp.float64)
    )
    edge_high: Float64[Array, " 1"] = sigma_imag_dynamic(
        jnp.asarray([_DOMAIN_HIGH_EV], dtype=jnp.float64)
    )
    boundary: Float64[Array, " n_query"] = (
        edge_low[0] / (_DOMAIN_LOW_EV - queries_ev)
        - edge_high[0] / (_DOMAIN_HIGH_EV - queries_ev)
    ) / jnp.pi
    domain: Float64[Array, " 2"] = jnp.asarray(
        [_DOMAIN_LOW_EV, _DOMAIN_HIGH_EV], dtype=jnp.float64
    )

    def tail_only(points_ev: Float64[Array, " n_query"]) -> Any:
        returned: Any = common.semi_infinite_tail_contribution(
            domain, tail_spec, points_ev, _N_TAIL
        )
        return returned

    tail_derivative: Float64[Array, " n_query"]
    _, tail_derivative = jax.jvp(
        tail_only, (queries_ev,), (jnp.ones_like(queries_ev),)
    )
    returned: Float64[Array, " n_query"] = (
        core_derivative + boundary + tail_derivative
    )
    return returned


def _softplus_inverse_np(
    positive: Float64[NDArray, " n"],
) -> Float64[NDArray, " n"]:
    """PRIVATE: Convert positive physical parameters to raw coordinates.

    Notes
    -----
    The map inverts softplus through ``log(expm1(x))``.
    """
    returned: Float64[NDArray, " n"] = np.log(
        np.expm1(np.asarray(positive, dtype=np.float64))
    )
    return returned


def _fermi_liquid_model() -> SelfEnergyModel:
    """PRIVATE: Create the frozen Fermi-liquid carrier and tails.

    Notes
    -----
    The carrier reuses the frozen tail coordinates.
    """
    raw: Float64[NDArray, " three"] = _softplus_inverse_np(
        np.asarray(_FL_PARAMETERS_PHYSICAL)
    )
    returned: SelfEnergyModel = make_self_energy_model(
        coefficients=jnp.asarray(raw),
        mode="fermi_liquid",
        kk_consistent=True,
        kk_domain_rel_fermi_ev=jnp.asarray([_DOMAIN_LOW_EV, _DOMAIN_HIGH_EV]),
        tail_coefficients=jnp.asarray(_FL_TAIL_RAW),
        subtraction_point_rel_fermi_ev=0.0,
        tail_mode="power2",
    )
    return returned


def _scaled_model(
    fixture: str,
    scale: Float64[Array, " n_param"],
) -> SelfEnergyModel:
    """PRIVATE: Create a carrier scaled from the fixture.

    Notes
    -----
    The scale multiplies the physical parameters before the raw map.
    """
    if fixture == "fermi_liquid":
        physical: Float64[Array, " n_param"] = scale * jnp.asarray(
            _FL_PARAMETERS_PHYSICAL
        )
        raw: Float64[Array, " n_param"] = jnp.log(jnp.expm1(physical))
        returned: SelfEnergyModel = make_self_energy_model(
            coefficients=raw,
            mode="fermi_liquid",
            kk_consistent=True,
            kk_domain_rel_fermi_ev=jnp.asarray(
                [_DOMAIN_LOW_EV, _DOMAIN_HIGH_EV]
            ),
            tail_coefficients=jnp.asarray(_FL_TAIL_RAW),
            subtraction_point_rel_fermi_ev=0.0,
            tail_mode="power2",
        )
        return returned
    physical = scale * jnp.asarray(_KINK_PARAMETERS_PHYSICAL)
    raw = jnp.log(jnp.expm1(physical))
    returned: SelfEnergyModel = make_self_energy_model(
        coefficients=raw,
        mode="bosonic_kink",
        kk_consistent=True,
        tail_mode="analytic",
    )
    return returned


def _hat_core_pv(
    nodes_ev: Float64[NDArray, " n_nodes"],
    ordinates_ev: Float64[NDArray, " n_nodes"],
    queries_ev: Float64[NDArray, " n_query"],
) -> Float64[NDArray, " n_query"]:
    """PRIVATE: Evaluate one hat function's exact principal value.

    For a linear segment with slope ``m`` on ``[x0, x1]`` the principal
    value of ``y(x) / (x - q)`` equals ``m * (x1 - x0) + y(q) *
    log|((x1 - q) / (x0 - q))|`` with ``y(q)`` the extended interpolant.
    The queries must avoid the interior nodes.

    Notes
    -----
    The loop evaluates the documented per-segment formula with NumPy only.
    """
    out: Float64[NDArray, " n_query"] = np.zeros_like(
        np.asarray(queries_ev, dtype=np.float64)
    )
    index: int
    query: float
    segment: int
    for index, query in enumerate(np.asarray(queries_ev)):
        total: float = 0.0
        for segment in range(len(nodes_ev) - 1):
            x0: float = float(nodes_ev[segment])
            x1: float = float(nodes_ev[segment + 1])
            y0: float = float(ordinates_ev[segment])
            y1: float = float(ordinates_ev[segment + 1])
            slope: float = (y1 - y0) / (x1 - x0)
            extended: float = y0 + slope * (query - x0)
            total += slope * (x1 - x0) + extended * np.log(
                abs((x1 - query) / (x0 - query))
            )
        out[index] = total / np.pi
    return out


def _hand_power2_tail(
    domain_ev: Tuple[float, float],
    amplitudes: Tuple[float, float],
    alphas: Tuple[float, float],
    betas: Tuple[float, float],
    queries_ev: Float64[NDArray, " n_query"],
) -> Float64[NDArray, " n_query"]:
    """PRIVATE: Evaluate both power2 tails with the frozen 256 rule.

    Notes
    -----
    The rule mirrors the frozen 256-node tail quadrature.
    """
    gauss_nodes: Float64[NDArray, " n_tail"]
    gauss_weights: Float64[NDArray, " n_tail"]
    gauss_nodes, gauss_weights = np.polynomial.legendre.leggauss(_N_TAIL)
    u: Float64[NDArray, " n_tail"] = (gauss_nodes + 1.0) / 2.0
    weights: Float64[NDArray, " n_tail"] = gauss_weights / 2.0
    out: Float64[NDArray, " n_query"] = np.zeros_like(
        np.asarray(queries_ev, dtype=np.float64)
    )
    sides: Tuple[Tuple[float, float, float, float, float], ...] = (
        (amplitudes[0], alphas[0], betas[0], domain_ev[0], -1.0),
        (amplitudes[1], alphas[1], betas[1], domain_ev[1], 1.0),
    )
    index: int
    query: float
    amplitude: float
    alpha: float
    beta: float
    edge: float
    sign: float
    for index, query in enumerate(np.asarray(queries_ev)):
        for amplitude, alpha, beta, edge, sign in sides:
            scale: float = beta**-0.5
            distance: Float64[NDArray, " n_tail"] = scale * u / (1.0 - u)
            jacobian: Float64[NDArray, " n_tail"] = scale / (1.0 - u) ** 2
            sigma_imag: Float64[NDArray, " n_tail"] = -amplitude / (
                1.0 + alpha * distance + beta * distance**2
            )
            denominator: Float64[NDArray, " n_tail"] = (
                edge + sign * distance - query
            )
            out[index] += float(
                np.sum(weights * sigma_imag * jacobian / (np.pi * denominator))
            )
    return out


def _spectral_intensity_reference() -> Dict[str, Float64[NDArray, "..."]]:
    """PRIVATE: Load and authenticate the registered resolvent archive.

    Notes
    -----
    Digest checks precede parsing of both the manifest and numeric archive.
    """
    manifest: Dict[str, Any] = _authenticated_json(
        _SPECTRAL_INTENSITY_MANIFEST_PATH,
        _SPECTRAL_INTENSITY_MANIFEST_SHA256,
    )
    assert manifest["schema"] == "diffpes.spectral-intensity-reference.v1"
    returned: Dict[str, Float64[NDArray, "..."]] = _authenticated_npz(
        _SPECTRAL_INTENSITY_ARCHIVE_PATH,
        _SPECTRAL_INTENSITY_ARCHIVE_SHA256,
    )
    return returned


def _degenerate_gradient_witness() -> Dict[str, Any]:
    """PRIVATE: Load and authenticate two Hamiltonian-gradient witnesses.

    Notes
    -----
    The JSON contains the frozen graphene and Kramers coordinates and their
    independently measured central finite-difference ladders.
    """
    returned: Dict[str, Any] = _authenticated_json(
        _DEGENERATE_WITNESS_PATH,
        _DEGENERATE_WITNESS_SHA256,
    )
    return returned
