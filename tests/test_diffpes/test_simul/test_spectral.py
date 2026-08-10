"""Register the strict-red Kramers-Kronig spectral evaluation lane.

Extended Summary
----------------
The artifact, replication, and composition checks pass before production
changes. Every assertion on the committed :mod:`diffpes.simul.spectral`
contract carries a strict expected-failure mark until the production
module lands. Each red test states its registered acceptance number.
Run one red test raw with ``--runxfail``. Deselect the complete red lane
with ``-m "not xfail"``.

The committed truth set contains three frozen artifacts:

- ``kk_analytic_reference/`` with the 80-digit analytic pole and Wigner
  arbiter archive;
- ``kk_operator_selection_reference.npz`` with the frozen measurements of
  the committed cell-integrated principal-value operator;
- ``self_energy_models_reference.npz`` with the frozen Fermi-liquid and
  bosonic-kink truths and complex-step parameter-derivative rows.

The battery authenticates the committed selection instrument modules in
``tests/_reference_tools`` through SHA-256 digests. It then uses them as
the independent composite-route truth for the autodiff demonstrator.

:see: :func:`diffpes.simul.spectral.evaluate_self_energy`
:see: :func:`diffpes.simul.spectral._kk_transform`
"""

from __future__ import annotations

import hashlib
import importlib.util
import inspect
import json
from pathlib import Path
from types import ModuleType
from typing import Any, Callable

import chex
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from beartype.typing import Dict, Tuple
from jaxtyping import Array, Complex128, Float64
from numpy.typing import NDArray

from diffpes.simul import evaluate_self_energy, voigt
from diffpes.types import SelfEnergyModel, make_self_energy_model
from diffpes.utils import faddeeva
from tests._assertions import assert_rejects

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

# Frozen SHA-256 digests. The selection manifest records the archive and
# instrument digests; the constants below pin the manifests themselves.
_SELECTION_MANIFEST_SHA256: str = (
    "c84b3e274b0ab519cfde1b6898c6ccc6fb5568d05b12e12483096866cb337816"
)
_SELECTION_ARCHIVE_SHA256: str = (
    "e827e91c62e294afc50112af5fe484e5ff002070f106511d07bb813502648430"
)
_ANALYTIC_MANIFEST_SHA256: str = (
    "cf0ddb484727102a65c4bf88d742c11334ca03cec1e010422ea6ffc19aefa7be"
)
_ANALYTIC_ARCHIVE_SHA256: str = (
    "ba30a3ee4e65658ace63ed54e65c3ec8ad8ae0868c396653123896265829cba5"
)
_ANALYTIC_GENERATOR_SHA256: str = (
    "4080e54f69f8c4691ed265c268c681eb508880a9b0db30b80546c2f27e69a554"
)
_MODELS_MANIFEST_SHA256: str = (
    "04e207d345a4c48500f26364a3c6aa03bc6435968b2357d381c7495eeaeb9b27"
)
_MODELS_ARCHIVE_SHA256: str = (
    "59a115c16cbcbd57e7b70ec290380d224d2f2f73c5b6682afbcec047a4fe2830"
)
_MODELS_GENERATOR_SHA256: str = (
    "3530d7c120edfed2cfe7c57c96b9105076c9fca83ef9c905665b14a8a9789ecf"
)
_SELECTION_GENERATOR_SHA256: str = (
    "239543b12c5c3ea46cbc4b579f1fd062c0183328d9dc379cbe67cbcd689307c7"
)
_COMMON_MODULE_SHA256: str = (
    "f6af92d4ea7b16cc020b894637e9ce4feecb7dcde8926c1026c6928dee2e5517"
)
_CUBIC_MODULE_SHA256: str = (
    "a17e6f2c6b4ec3aee76d3ec116cb091bccc85bcc0edb59f908f1128aa2ea7900"
)
_LINEAR_MODULE_SHA256: str = (
    "f01469f8fa44163dc07c5f1e1be76facd4c4a8ad00584deef9b8e210a23d6bfb"
)

# Registered acceptance numbers. Sources: the frozen operator-selection
# manifest (`kk-analytic-pair-truth`, `kk-derivative-composite-route`,
# `kk-refinement-convergence`) and the frozen causal-model manifest
# (`causal-self-energy-analytic-truth`,
# `self-energy-parameter-derivative-truth`).
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
# Fermi-liquid rows, measured by the committed instrument before this file
# froze: subtracted value max error 1.6529e-9 eV (mixed ratio 0.0497),
# composite query-derivative max error 3.0722e-9 (mixed ratio 0.0459),
# parameter-row max error 2.579e-8 against the complex-step truths.
_FL_VALUE_ATOL_EV: float = 2.0e-8
_FL_VALUE_RTOL: float = 1.0e-6
_FL_REAL_ROW_ATOL: float = 5.0e-8
_DERIVATIVE_ROW_RTOL: float = 1.0e-6
_ANALYTIC_ROW_ATOL: float = 1.0e-10

# Frozen evaluation geometry: base domain [-8, 8] eV, n_kk = 4096 nodes,
# 256-node semi-infinite tail rule, subtraction point 0 eV.
_DOMAIN_LOW_EV: float = -8.0
_DOMAIN_HIGH_EV: float = 8.0
_N_KK: int = 4096
_N_TAIL: int = 256
_BASE_SPACING_EV: float = (_DOMAIN_HIGH_EV - _DOMAIN_LOW_EV) / (_N_KK - 1)

# Frozen retarded-pole fixture and its committed raw tail coordinates.
_POLE_OMEGA0_EV: float = 0.35
_POLE_GAMMA_EV: float = 0.20
_POLE_COUPLING_EV2: float = 0.12
_POLE_TAIL_RAW: Tuple[float, float] = (
    -11.70907030862417,
    -11.359064962331182,
)

# Frozen Fermi-liquid fixture (gamma0, beta, omega_c) and the derived raw
# power2 tail coordinates. The tail derivation matches the true 1/omega^2
# asymptote: beta_tail = A_edge / (beta * omega_c**4), delta_beta =
# beta_tail - alpha**2 / 4, raw = log(expm1(delta_beta)).
_FL_PARAMETERS_PHYSICAL: Tuple[float, float, float] = (0.02, 0.5, 0.8)
_FL_TAIL_RAW: Tuple[float, float] = (
    -12.270842147353243,
    -12.270842147353243,
)

# Frozen bosonic-kink fixture (gamma0, g, omega0, width).
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
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_npz(path: Path) -> Dict[str, Float64[NDArray, "..."]]:
    """PRIVATE: Load one inert NPZ into ordinary arrays without pickle.

    Notes
    -----
    The loader forbids pickle and copies each named array.
    """
    archive: Any
    with np.load(path, allow_pickle=False) as archive:
        return {name: archive[name] for name in archive.files}


def _authenticated_json(path: Path, digest: str) -> Dict[str, Any]:
    """PRIVATE: Load one manifest after its SHA-256 digest matches the pin.

    Notes
    -----
    The check compares the digest before any parse.
    """
    assert _sha256(path) == digest, f"digest mismatch for {path.name}"
    return json.loads(path.read_text(encoding="utf-8"))


def _authenticated_npz(
    path: Path, digest: str
) -> Dict[str, Float64[NDArray, "..."]]:
    """PRIVATE: Load one archive after its SHA-256 digest matches the pin.

    Notes
    -----
    The check compares the digest before any load.
    """
    assert _sha256(path) == digest, f"digest mismatch for {path.name}"
    return _load_npz(path)


def _committed_module(filename: str, digest: str) -> ModuleType:
    """PRIVATE: Import one committed instrument module after authentication.

    Notes
    -----
    The cache keys one import per authenticated filename.
    """
    if filename in _COMMITTED_MODULE_CACHE:
        return _COMMITTED_MODULE_CACHE[filename]
    path: Path = _TOOLS_DIRECTORY / filename
    assert _sha256(path) == digest, f"digest mismatch for {filename}"
    spec: Any = importlib.util.spec_from_file_location(
        f"_kk_lane_{path.stem}", path
    )
    assert spec is not None and spec.loader is not None
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
    return common, cubic


def _base_grid() -> Float64[Array, " n_kk"]:
    """PRIVATE: Return the frozen uniform base quadrature grid in eV.

    Notes
    -----
    The grid follows the frozen index construction.
    """
    return jnp.asarray(
        _DOMAIN_LOW_EV + _BASE_SPACING_EV * np.arange(_N_KK),
        dtype=jnp.float64,
    )


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
    return left, right


def _pole_sigma_imag(
    omega_rel_fermi_ev: Float64[Array, " n"],
) -> Float64[Array, " n"]:
    """PRIVATE: Evaluate the analytic retarded-pole imaginary part in eV.

    Notes
    -----
    The pole parameters come from the frozen fixture constants.
    """
    offset: Float64[Array, " n"] = omega_rel_fermi_ev - _POLE_OMEGA0_EV
    return (
        -_POLE_COUPLING_EV2 * _POLE_GAMMA_EV / (offset**2 + _POLE_GAMMA_EV**2)
    )


def _pole_dsigma_imag(
    omega_rel_fermi_ev: Float64[Array, " n"],
) -> Float64[Array, " n"]:
    """PRIVATE: Evaluate the analytic query derivative of the pole imaginary part.

    Notes
    -----
    The formula differentiates the frozen pole closed form.
    """
    offset: Float64[Array, " n"] = omega_rel_fermi_ev - _POLE_OMEGA0_EV
    return (
        2.0
        * _POLE_COUPLING_EV2
        * _POLE_GAMMA_EV
        * offset
        / (offset**2 + _POLE_GAMMA_EV**2) ** 2
    )


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
    return (
        -beta
        * omega_rel_fermi_ev**2
        / (1.0 + (omega_rel_fermi_ev / omega_c) ** 4)
    )


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
    return (
        2.0
        * beta
        * omega_rel_fermi_ev
        * (quartic - 1.0)
        / (1.0 + quartic) ** 2
    )


def _fl_dsigma_real_domega(
    omega_rel_fermi_ev: Float64[NDArray, " n"],
) -> Float64[NDArray, " n"]:
    """PRIVATE: Evaluate the analytic full-line Fermi-liquid real-part derivative.

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
    return (
        (np.sqrt(2.0) / 2.0) * beta * omega_c**3 * numerator / quartic_sum**2
    )


def _pole_tail_spec(common: ModuleType) -> Any:
    """PRIVATE: Return the committed power2 tail parameters for the pole fixture.

    Notes
    -----
    The spec derives from the frozen raw coordinates and stencils.
    """
    grid: Float64[Array, " n_kk"] = _base_grid()
    values: Float64[Array, " n_kk"] = _pole_sigma_imag(grid)
    slope_left: Float64[Array, ""]
    slope_right: Float64[Array, ""]
    slope_left, slope_right = _cubic_edge_slopes(values, _BASE_SPACING_EV)
    return common.construct_power2_tail_spec(
        values[0],
        slope_left,
        values[-1],
        slope_right,
        _POLE_TAIL_RAW[0],
        _POLE_TAIL_RAW[1],
    )


def _fl_tail_spec(common: ModuleType) -> Any:
    """PRIVATE: Return the committed power2 tail parameters for the Fermi liquid.

    Notes
    -----
    The spec derives from the frozen raw coordinates and stencils.
    """
    grid: Float64[Array, " n_kk"] = _base_grid()
    values: Float64[Array, " n_kk"] = _fl_sigma_imag_dynamic(grid)
    slope_left: Float64[Array, ""]
    slope_right: Float64[Array, ""]
    slope_left, slope_right = _cubic_edge_slopes(values, _BASE_SPACING_EV)
    return common.construct_power2_tail_spec(
        values[0],
        slope_left,
        values[-1],
        slope_right,
        _FL_TAIL_RAW[0],
        _FL_TAIL_RAW[1],
    )


def _instrument_transform(
    sigma_imag_dynamic: Callable[[Float64[Array, " n"]], Float64[Array, " n"]],
    tail_spec: Any,
    queries_ev: Float64[Array, " n_query"],
) -> Float64[Array, " n_query"]:
    """PRIVATE: Evaluate the committed unsubtracted operator at arbitrary queries.

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
    return core + tail


def _instrument_subtracted(
    sigma_imag_dynamic: Callable[[Float64[Array, " n"]], Float64[Array, " n"]],
    tail_spec: Any,
    queries_ev: Float64[Array, " n_query"],
    subtraction_point_ev: float,
) -> Float64[Array, " n_query"]:
    """PRIVATE: Evaluate the committed subtracted real part at arbitrary queries.

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
    return total[:-1] - total[-1]


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
        return common.semi_infinite_tail_contribution(
            domain, tail_spec, points_ev, _N_TAIL
        )

    tail_derivative: Float64[Array, " n_query"]
    _, tail_derivative = jax.jvp(
        tail_only, (queries_ev,), (jnp.ones_like(queries_ev),)
    )
    return core_derivative + boundary + tail_derivative


def _softplus_inverse_np(
    positive: Float64[NDArray, " n"],
) -> Float64[NDArray, " n"]:
    """PRIVATE: Convert positive physical parameters to raw coordinates.

    Notes
    -----
    The map inverts softplus through ``log(expm1(x))``.
    """
    return np.log(np.expm1(np.asarray(positive, dtype=np.float64)))


def _fermi_liquid_model() -> SelfEnergyModel:
    """PRIVATE: Create the frozen Fermi-liquid carrier with its committed tails.

    Notes
    -----
    The carrier reuses the frozen tail coordinates.
    """
    raw: Float64[NDArray, " three"] = _softplus_inverse_np(
        np.asarray(_FL_PARAMETERS_PHYSICAL)
    )
    return make_self_energy_model(
        coefficients=jnp.asarray(raw),
        mode="fermi_liquid",
        kk_consistent=True,
        kk_domain_rel_fermi_ev=jnp.asarray([_DOMAIN_LOW_EV, _DOMAIN_HIGH_EV]),
        tail_coefficients=jnp.asarray(_FL_TAIL_RAW),
        subtraction_point_rel_fermi_ev=0.0,
        tail_mode="power2",
    )


def _scaled_model(
    fixture: str,
    scale: Float64[Array, " n_param"],
) -> SelfEnergyModel:
    """PRIVATE: Create a carrier whose parameters equal ``scale`` times the fixture.

    Notes
    -----
    The scale multiplies the physical parameters before the raw map.
    """
    if fixture == "fermi_liquid":
        physical: Float64[Array, " n_param"] = scale * jnp.asarray(
            _FL_PARAMETERS_PHYSICAL
        )
        raw: Float64[Array, " n_param"] = jnp.log(jnp.expm1(physical))
        return make_self_energy_model(
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
    physical = scale * jnp.asarray(_KINK_PARAMETERS_PHYSICAL)
    raw = jnp.log(jnp.expm1(physical))
    return make_self_energy_model(
        coefficients=raw,
        mode="bosonic_kink",
        kk_consistent=True,
        tail_mode="analytic",
    )


def _hat_core_pv(
    nodes_ev: Float64[NDArray, " n_nodes"],
    ordinates_ev: Float64[NDArray, " n_nodes"],
    queries_ev: Float64[NDArray, " n_query"],
) -> Float64[NDArray, " n_query"]:
    """PRIVATE: Evaluate the exact segment-wise principal value of one hat function.

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
    """PRIVATE: Evaluate both semi-infinite power2 tails with the frozen 256 rule.

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


class TestKramersKronigEvidence(chex.TestCase):
    """Validate the frozen independent artifacts before production edits."""

    def test_reference_manifests_and_archives_are_authenticated(
        self,
    ) -> None:
        """Verify every truth manifest, archive, and generator digest.

        The test recomputes each SHA-256 digest and compares it against
        its frozen pin. It also checks each manifest against the digest of
        its own archive.

        Notes
        -----
        The test reads the three manifests, the three archives, both
        generators, and the committed instrument modules. It then compares
        the registered budget constants against the frozen module values.
        """
        selection: Dict[str, Any] = _authenticated_json(
            _SELECTION_MANIFEST_PATH, _SELECTION_MANIFEST_SHA256
        )
        assert selection["schema"] == "diffpes.kk-operator-selection.v1"
        assert selection["archive_sha256"] == _SELECTION_ARCHIVE_SHA256
        assert _sha256(_SELECTION_ARCHIVE_PATH) == _SELECTION_ARCHIVE_SHA256
        executable: Dict[str, Any] = selection["executable_inputs"]
        assert executable["generator"]["sha256"] == _SELECTION_GENERATOR_SHA256
        assert executable["common"]["sha256"] == _COMMON_MODULE_SHA256
        assert executable["piecewise_cubic"]["sha256"] == _CUBIC_MODULE_SHA256
        assert (
            executable["piecewise_linear"]["sha256"] == _LINEAR_MODULE_SHA256
        )
        assert selection["production_imports"] == []

        analytic: Dict[str, Any] = _authenticated_json(
            _ANALYTIC_DIRECTORY / "manifest.json", _ANALYTIC_MANIFEST_SHA256
        )
        assert analytic["schema"] == "diffpes.kk-analytic-reference.v1"
        assert analytic["archive_sha256"] == _ANALYTIC_ARCHIVE_SHA256
        assert (
            _sha256(_ANALYTIC_DIRECTORY / analytic["archive"])
            == _ANALYTIC_ARCHIVE_SHA256
        )
        assert (
            _sha256(_ANALYTIC_DIRECTORY / analytic["generator"])
            == _ANALYTIC_GENERATOR_SHA256
        )
        assert int(analytic["arbiter"]["decimal_digits"]) == 80

        models: Dict[str, Any] = _authenticated_json(
            _MODELS_MANIFEST_PATH, _MODELS_MANIFEST_SHA256
        )
        assert models["schema"] == "diffpes.self-energy-models-reference.v1"
        recorded: str = models["archives"]["self_energy_models_reference"][
            "sha256"
        ]
        assert recorded == _MODELS_ARCHIVE_SHA256
        assert _sha256(_MODELS_ARCHIVE_PATH) == _MODELS_ARCHIVE_SHA256
        assert models["generator_sha256"] == _MODELS_GENERATOR_SHA256
        zeros: list[str] = models["derivatives"]["structural_zeros"]
        assert any("dSigma'/dGamma0" in entry for entry in zeros)

        budgets: Dict[str, float] = selection["registered_budgets"]
        assert budgets["pair_truth_atol_ev"] == _PAIR_TRUTH_ATOL_EV
        assert budgets["pair_truth_rtol"] == _PAIR_TRUTH_RTOL
        assert budgets["pole_tail_only_max_delta_sigma_ev"] == _TAIL_RULE_BOUND
        assert budgets["pole_tail_only_max_delta_dsigma"] == _TAIL_RULE_BOUND

    def test_operator_selection_measurements_replicate_frozen_numbers(
        self,
    ) -> None:
        """Reproduce the committed acceptance measurements from the archive.

        The frozen operator meets the per-row pair-truth mixed criterion
        ``|err| <= 2e-8 + 1e-6 * |truth|``. The recomputed maximum error
        equals ``3.393963743381079e-8 eV`` with worst ratio
        ``0.8239596852267533``. The recomputed composite query-derivative
        error equals ``9.342723950034326e-7``.

        Notes
        -----
        The test loads the frozen arrays, recomputes each measurement, and
        compares it against the recorded manifest number. It also checks
        the 256-to-512 tail rule deltas against the ``1e-13`` budget.
        """
        selection: Dict[str, Float64[NDArray, "..."]] = _authenticated_npz(
            _SELECTION_ARCHIVE_PATH, _SELECTION_ARCHIVE_SHA256
        )
        analytic: Dict[str, Float64[NDArray, "..."]] = _authenticated_npz(
            _ANALYTIC_DIRECTORY / "kk_reference.npz",
            _ANALYTIC_ARCHIVE_SHA256,
        )
        queries: Float64[NDArray, " n"] = selection["queries_ev"]
        truth: Float64[NDArray, " n"] = selection[
            "truth_pole_sigma_real_sub_ev"
        ]
        np.testing.assert_array_equal(queries, analytic["pole_omega"])
        np.testing.assert_array_equal(truth, analytic["pole_sigma_real"])

        offset: Float64[NDArray, " n"] = queries - _POLE_OMEGA0_EV
        closed: Float64[NDArray, " n"] = (
            _POLE_COUPLING_EV2 * offset / (offset**2 + _POLE_GAMMA_EV**2)
        ) - (
            _POLE_COUPLING_EV2
            * (0.0 - _POLE_OMEGA0_EV)
            / ((0.0 - _POLE_OMEGA0_EV) ** 2 + _POLE_GAMMA_EV**2)
        )
        assert np.max(np.abs(closed - truth)) <= 5.0e-16

        error: Float64[NDArray, " n"] = np.abs(
            selection["pwcubic_pole_base_sigma_sub_ev"] - truth
        )
        bound: Float64[NDArray, " n"] = (
            _PAIR_TRUTH_ATOL_EV + _PAIR_TRUTH_RTOL * np.abs(truth)
        )
        assert int(np.sum(error > bound)) == 0
        np.testing.assert_allclose(
            np.max(error), _PAIR_TRUTH_EXPECTED_MAX_ERROR_EV, rtol=1e-12
        )
        np.testing.assert_allclose(
            np.max(error / bound),
            _PAIR_TRUTH_EXPECTED_MAX_RATIO,
            rtol=1e-12,
        )

        composite_error: float = float(
            np.max(
                np.abs(
                    selection["pwcubic_pole_base_dsigma_composite"]
                    - selection["truth_pole_dsigma_domega"]
                )
            )
        )
        np.testing.assert_allclose(
            composite_error, _COMPOSITE_EXPECTED_MAX_ERROR, rtol=1e-12
        )
        assert composite_error <= _COMPOSITE_DERIVATIVE_BOUND

        tail_value_delta: float = float(
            np.max(
                np.abs(
                    selection["pwcubic_pole_tail512_sigma_sub_ev"]
                    - selection["pwcubic_pole_base_sigma_sub_ev"]
                )
            )
        )
        tail_derivative_delta: float = float(
            np.max(
                np.abs(
                    selection["pwcubic_pole_tail512_dsigma_composite"]
                    - selection["pwcubic_pole_base_dsigma_composite"]
                )
            )
        )
        assert tail_value_delta <= _TAIL_RULE_BOUND
        assert tail_derivative_delta <= _TAIL_RULE_BOUND

    def test_causal_model_truth_archive_replicates_closed_forms(
        self,
    ) -> None:
        """Reproduce the frozen causal-model truths from their formulas.

        The raw coordinates reproduce the physical parameters through
        softplus exactly. Both imaginary-part grids replicate their closed
        forms. The Fermi-liquid subtracted real part replicates its
        analytic identity. The documented structural-zero columns
        ``dSigma'/dGamma0`` equal zero exactly for both fixtures.

        Notes
        -----
        The test evaluates each closed form on the frozen grids with
        NumPy. It then compares the results against the archive rows and
        checks the recorded complex-step cross-check scalars.
        """
        models: Dict[str, Float64[NDArray, "..."]] = _authenticated_npz(
            _MODELS_ARCHIVE_PATH, _MODELS_ARCHIVE_SHA256
        )
        fl_raw: Float64[NDArray, " three"] = models["fl_parameters_raw"]
        np.testing.assert_allclose(
            np.logaddexp(fl_raw, 0.0),
            np.asarray(_FL_PARAMETERS_PHYSICAL),
            rtol=1e-15,
        )
        kink_raw: Float64[NDArray, " four"] = models["kink_parameters_raw"]
        np.testing.assert_allclose(
            np.logaddexp(kink_raw, 0.0),
            np.asarray(_KINK_PARAMETERS_PHYSICAL),
            rtol=1e-15,
        )

        grid: Float64[NDArray, " n"] = models["eval_grid_ev"]
        gamma0: float
        beta: float
        omega_c: float
        gamma0, beta, omega_c = _FL_PARAMETERS_PHYSICAL
        fl_imag: Float64[NDArray, " n"] = -gamma0 - beta * grid**2 / (
            1.0 + (grid / omega_c) ** 4
        )
        np.testing.assert_allclose(
            fl_imag, models["fl_sigma_imag_grid"], rtol=1e-13, atol=1e-16
        )
        fl_real: Float64[NDArray, " n"] = (
            (np.sqrt(2.0) / 2.0)
            * beta
            * omega_c**3
            * grid
            * (grid**2 - omega_c**2)
            / (grid**4 + omega_c**4)
        )
        np.testing.assert_allclose(
            fl_real,
            models["fl_sigma_real_subtracted_analytic"],
            rtol=1e-13,
            atol=1e-16,
        )

        k_gamma0: float
        k_g: float
        k_omega0: float
        k_width: float
        k_gamma0, k_g, k_omega0, k_width = _KINK_PARAMETERS_PHYSICAL
        pair: Complex128[NDArray, " n"] = k_g**2 * (
            1.0 / (grid - k_omega0 + 1j * k_width)
            + 1.0 / (grid + k_omega0 + 1j * k_width)
        )
        np.testing.assert_allclose(
            -k_gamma0 + np.imag(pair),
            models["kink_sigma_imag_grid"],
            rtol=1e-13,
            atol=1e-16,
        )
        np.testing.assert_allclose(
            np.real(pair),
            models["kink_sigma_real_grid"],
            rtol=1e-13,
            atol=1e-16,
        )

        np.testing.assert_array_equal(
            models["fl_dsigma_real_dq"][:, 0],
            np.zeros_like(models["fl_probe_frequencies_ev"]),
        )
        np.testing.assert_array_equal(
            models["kink_dsigma_real_dq"][:, 0],
            np.zeros_like(models["kink_probe_frequencies_ev"]),
        )
        assert (
            float(models["fl_derivative_crosscheck_max_abs_error"]) <= 1.0e-10
        )
        assert (
            float(models["kink_derivative_crosscheck_max_abs_error"])
            <= 1.0e-10
        )

    @pytest.mark.big_mem
    @pytest.mark.rss_limit_mb(1200)
    def test_committed_instrument_replicates_frozen_operator_outputs(
        self,
    ) -> None:
        """Reproduce the frozen operator outputs with the instrument.

        The authenticated instrument modules reproduce the frozen
        subtracted values and the frozen composite derivative on the pole
        fixture. This certifies the in-file composite-route truth for the
        red autodiff demonstrator.

        Notes
        -----
        The test rebuilds the tail contract from the recorded raw
        coordinates and the committed edge stencils. It then compares both
        instrument outputs against the frozen arrays at ``1e-15``.
        """
        selection: Dict[str, Float64[NDArray, "..."]] = _authenticated_npz(
            _SELECTION_ARCHIVE_PATH, _SELECTION_ARCHIVE_SHA256
        )
        common: ModuleType
        common, _ = _committed_operator()
        queries: Float64[Array, " n"] = jnp.asarray(
            selection["queries_ev"], dtype=jnp.float64
        )
        spec: Any = _pole_tail_spec(common)
        values: Float64[Array, " n"] = _instrument_subtracted(
            _pole_sigma_imag, spec, queries, 0.0
        )
        np.testing.assert_allclose(
            np.asarray(values),
            selection["pwcubic_pole_base_sigma_sub_ev"],
            rtol=0.0,
            atol=1e-15,
        )
        derivative: Float64[Array, " n"] = _instrument_composite_derivative(
            _pole_sigma_imag, _pole_dsigma_imag, spec, queries
        )
        np.testing.assert_allclose(
            np.asarray(derivative),
            selection["pwcubic_pole_base_dsigma_composite"],
            rtol=0.0,
            atol=1e-15,
        )

    def test_positive_width_voigt_consumes_certified_faddeeva_envelope(
        self,
    ) -> None:
        """Require the positive-width Voigt path to use the Faddeeva map.

        The strictly positive branch must equal
        ``Re w((E - E0 + i*gamma) / (sigma*sqrt(2))) / (sigma*sqrt(2*pi))``
        with the certified :func:`diffpes.utils.faddeeva` envelope. The
        check guards this composition when :mod:`diffpes.simul.spectral`
        lands next to the broadening module.

        Notes
        -----
        The test composes the Faddeeva map directly and compares the
        production Voigt values against the composition at ``1e-13``.
        """
        energy: Float64[Array, " n"] = jnp.linspace(-1.0, 1.0, 41)
        center: float = 0.137
        sigma: float = 0.04
        gamma: float = 0.1
        argument: Complex128[Array, " n"] = (energy - center + 1j * gamma) / (
            sigma * jnp.sqrt(2.0)
        )
        composed: Float64[Array, " n"] = jnp.real(faddeeva(argument)) / (
            sigma * jnp.sqrt(2.0 * jnp.pi)
        )
        produced: Float64[Array, " n"] = voigt(energy, center, sigma, gamma)
        np.testing.assert_allclose(
            np.asarray(produced),
            np.asarray(composed),
            rtol=1e-13,
            atol=1e-16,
        )


class TestEvaluateSelfEnergy(chex.TestCase):
    """Test the public complex retarded evaluation contract.

    :see: :func:`diffpes.simul.spectral.evaluate_self_energy`
    :see: :func:`~diffpes.simul.evaluate_self_energy`
    """

    def test_constant_mode_returns_complex_retarded_sigma(self) -> None:
        """Require ``Sigma = -i * softplus(c0)`` for the constant carrier.

        Acceptance: the output dtype is complex. The subtracted real part
        equals zero exactly. The imaginary part equals ``-gamma`` within
        ``1e-14`` through the stored inverse-softplus coordinate.

        Notes
        -----
        The test builds the ``gamma`` shortcut carrier and evaluates it on
        a small query grid. It then checks dtype, real part, and imaginary
        part separately.
        """
        model: SelfEnergyModel = make_self_energy_model(gamma=0.1)
        omega: Float64[Array, " n"] = jnp.linspace(-1.0, 1.0, 21)
        sigma: Complex128[Array, " n"] = evaluate_self_energy(omega, model)
        assert jnp.iscomplexobj(sigma), (
            "constant-mode evaluation must return the complex retarded "
            "self-energy"
        )
        np.testing.assert_array_equal(np.real(np.asarray(sigma)), np.zeros(21))
        np.testing.assert_allclose(
            np.imag(np.asarray(sigma)),
            np.full(21, -0.1),
            rtol=0.0,
            atol=1e-14,
        )

    @pytest.mark.big_mem
    @pytest.mark.rss_limit_mb(1200)
    def test_fermi_liquid_subtracted_real_part_matches_frozen_truth(
        self,
    ) -> None:
        """Match the frozen full-line Fermi-liquid subtracted real part.

        Acceptance: per query row ``|err| <= 2e-8 + 1e-6 * |truth|``
        against ``fl_sigma_real_subtracted_analytic``. The committed
        instrument measures maximum error ``1.6529e-9 eV`` (worst mixed
        ratio ``0.0497``) here. The imaginary part must reproduce
        ``fl_sigma_imag_grid`` at roundoff.

        Notes
        -----
        The test evaluates the frozen carrier on the frozen grid with the
        default committed geometry. It then applies the registered mixed
        criterion per row.
        """
        models: Dict[str, Float64[NDArray, "..."]] = _authenticated_npz(
            _MODELS_ARCHIVE_PATH, _MODELS_ARCHIVE_SHA256
        )
        queries: Float64[Array, " n"] = jnp.asarray(
            models["eval_grid_ev"], dtype=jnp.float64
        )
        model: SelfEnergyModel = _fermi_liquid_model()
        sigma: Complex128[Array, " n"] = evaluate_self_energy(
            queries, model, n_kk=_N_KK
        )
        assert jnp.iscomplexobj(sigma), (
            "fermi_liquid evaluation must return the complex retarded "
            "self-energy"
        )
        truth: Float64[NDArray, " n"] = models[
            "fl_sigma_real_subtracted_analytic"
        ]
        error: Float64[NDArray, " n"] = np.abs(
            np.real(np.asarray(sigma)) - truth
        )
        assert np.all(
            error <= _FL_VALUE_ATOL_EV + _FL_VALUE_RTOL * np.abs(truth)
        )
        np.testing.assert_allclose(
            np.imag(np.asarray(sigma)),
            models["fl_sigma_imag_grid"],
            rtol=1e-12,
            atol=1e-14,
        )

    def test_bosonic_kink_complex_pole_pair_matches_frozen_truth(
        self,
    ) -> None:
        """Match the frozen analytic bosonic-kink complex pole pair.

        Acceptance: the real part matches ``kink_sigma_real_grid`` at
        rtol ``1e-8``. The imaginary part reproduces the frozen
        ``kink_sigma_imag_grid`` parameter truth at roundoff.

        Notes
        -----
        The test builds the frozen kink carrier from raw coordinates and
        evaluates the analytic pole pair on the frozen grid. It compares
        both parts against the archive rows.
        """
        models: Dict[str, Float64[NDArray, "..."]] = _authenticated_npz(
            _MODELS_ARCHIVE_PATH, _MODELS_ARCHIVE_SHA256
        )
        queries: Float64[Array, " n"] = jnp.asarray(
            models["eval_grid_ev"], dtype=jnp.float64
        )
        model: SelfEnergyModel = _scaled_model(
            "bosonic_kink", jnp.ones(4, dtype=jnp.float64)
        )
        sigma: Complex128[Array, " n"] = evaluate_self_energy(queries, model)
        assert jnp.iscomplexobj(sigma), (
            "bosonic_kink evaluation must return the complex retarded "
            "self-energy"
        )
        np.testing.assert_allclose(
            np.real(np.asarray(sigma)),
            models["kink_sigma_real_grid"],
            rtol=1e-8,
            atol=1e-12,
        )
        np.testing.assert_allclose(
            np.imag(np.asarray(sigma)),
            models["kink_sigma_imag_grid"],
            rtol=1e-12,
            atol=1e-14,
        )

    def test_appending_distant_query_points_preserves_existing_values(
        self,
    ) -> None:
        """Require query-set invariance of the existing outputs.

        Acceptance: appended distant trusted query points change no
        existing output by more than ``1e-15 eV``. The invariance holds on
        the complex retarded contract.

        Notes
        -----
        The test evaluates one base query set and one extended query set.
        It then compares the shared prefix of both outputs.
        """
        model: SelfEnergyModel = _fermi_liquid_model()
        base: Float64[Array, " n"] = jnp.linspace(-1.0, 1.0, 41)
        extended: Float64[Array, " m"] = jnp.concatenate(
            [base, jnp.asarray([5.0, -5.5, 6.5])]
        )
        first: Complex128[Array, " n"] = evaluate_self_energy(base, model)
        second: Complex128[Array, " m"] = evaluate_self_energy(extended, model)
        assert jnp.iscomplexobj(first), (
            "the invariance check runs on the complex retarded contract"
        )
        np.testing.assert_allclose(
            np.asarray(second)[: base.shape[0]],
            np.asarray(first),
            rtol=0.0,
            atol=_QUERY_INVARIANCE_ATOL_EV,
        )

    def test_query_window_never_defines_the_quadrature_grid(self) -> None:
        """Require the quadrature domain to come from the carrier.

        Acceptance: two disjoint query windows agree within ``1e-15 eV``
        at shared points. The grid derives from
        ``kk_domain_rel_fermi_ev`` and never from the query extrema. A
        planted query-window grid fails this bound.

        Notes
        -----
        The test evaluates one narrow and one wide query window with a
        shared prefix. It then compares the shared outputs bitwise-close.
        """
        model: SelfEnergyModel = _fermi_liquid_model()
        shared: Float64[Array, " k"] = jnp.linspace(-0.4, 0.4, 17)
        narrow: Float64[Array, " n"] = jnp.concatenate(
            [shared, jnp.asarray([-0.9, 1.1])]
        )
        wide: Float64[Array, " m"] = jnp.concatenate(
            [shared, jnp.asarray([-6.5, 6.9])]
        )
        narrow_out: Complex128[Array, " n"] = evaluate_self_energy(
            narrow, model
        )
        wide_out: Complex128[Array, " m"] = evaluate_self_energy(wide, model)
        assert jnp.iscomplexobj(narrow_out), (
            "the window check runs on the complex retarded contract"
        )
        np.testing.assert_allclose(
            np.asarray(wide_out)[: shared.shape[0]],
            np.asarray(narrow_out)[: shared.shape[0]],
            rtol=0.0,
            atol=_QUERY_INVARIANCE_ATOL_EV,
        )

    def test_trusted_interval_is_enforced_eagerly_and_under_jit(
        self,
    ) -> None:
        """Reject queries and subtraction points outside the trusted band.

        Acceptance: with base spacing ``h = 16/4095 eV`` the trusted
        interval equals ``[a + 2h, b - 2h]``. A query at ``b - h/2``, a
        query beyond the domain, and an out-of-band subtraction point each
        raise eagerly and under jit.

        Notes
        -----
        The test drives the shared rejection helper, which repeats each
        call through ``eqx.filter_jit`` and matches the error text.
        """
        model: SelfEnergyModel = _fermi_liquid_model()
        outside_query: Float64[Array, " one"] = jnp.asarray(
            [_DOMAIN_HIGH_EV - 0.5 * _BASE_SPACING_EV]
        )
        assert_rejects(
            evaluate_self_energy, outside_query, model, match="trusted"
        )
        beyond_domain: Float64[Array, " one"] = jnp.asarray([9.0])
        assert_rejects(
            evaluate_self_energy, beyond_domain, model, match="trusted"
        )
        raw: Float64[NDArray, " three"] = _softplus_inverse_np(
            np.asarray(_FL_PARAMETERS_PHYSICAL)
        )
        edge_subtraction: SelfEnergyModel = make_self_energy_model(
            coefficients=jnp.asarray(raw),
            mode="fermi_liquid",
            kk_consistent=True,
            kk_domain_rel_fermi_ev=jnp.asarray(
                [_DOMAIN_LOW_EV, _DOMAIN_HIGH_EV]
            ),
            tail_coefficients=jnp.asarray(_FL_TAIL_RAW),
            subtraction_point_rel_fermi_ev=(
                _DOMAIN_HIGH_EV - 0.5 * _BASE_SPACING_EV
            ),
            tail_mode="power2",
        )
        inside_query: Float64[Array, " one"] = jnp.asarray([0.25])
        assert_rejects(
            evaluate_self_energy,
            inside_query,
            edge_subtraction,
            match="trusted",
        )

    def test_fermi_liquid_trusted_boundary_evaluation_is_certified(
        self,
    ) -> None:
        """Certify boundary evaluation for a non-decaying imaginary part.

        The Fermi-liquid imaginary part does not decay inside the core
        domain. No selection-battery fixture covers this boundary class.
        Acceptance: a query just inside ``b - 2h`` evaluates and matches
        the committed operator within ``1e-9 * max(1, |value|)``. Queries
        at ``a + 1.5h`` and ``b - 1.5h`` raise eagerly and under jit.

        Notes
        -----
        The test evaluates production near the trusted boundary and
        compares against the authenticated instrument value. It then
        drives the shared rejection helper for both edges.
        """
        model: SelfEnergyModel = _fermi_liquid_model()
        boundary_query: Float64[Array, " one"] = jnp.asarray(
            [_DOMAIN_HIGH_EV - 2.5 * _BASE_SPACING_EV]
        )
        sigma: Complex128[Array, " one"] = evaluate_self_energy(
            boundary_query, model
        )
        assert jnp.iscomplexobj(sigma), (
            "the boundary check runs on the complex retarded contract"
        )
        common: ModuleType
        common, _ = _committed_operator()
        spec: Any = _fl_tail_spec(common)
        expected: Float64[Array, " one"] = _instrument_subtracted(
            _fl_sigma_imag_dynamic, spec, boundary_query, 0.0
        )
        deviation: float = float(
            np.abs(np.real(np.asarray(sigma))[0] - np.asarray(expected)[0])
        )
        assert deviation <= _IDENTITY_SCALE_BOUND * max(
            1.0, float(np.abs(np.asarray(expected)[0]))
        )
        assert_rejects(
            evaluate_self_energy,
            jnp.asarray([_DOMAIN_HIGH_EV - 1.5 * _BASE_SPACING_EV]),
            model,
            match="trusted",
        )
        assert_rejects(
            evaluate_self_energy,
            jnp.asarray([_DOMAIN_LOW_EV + 1.5 * _BASE_SPACING_EV]),
            model,
            match="trusted",
        )


class TestEvaluateSelfEnergyDerivatives(chex.TestCase):
    """Test the composite derivative route on the public callable.

    :see: :func:`diffpes.simul.spectral.evaluate_self_energy`
    """

    @pytest.mark.big_mem
    @pytest.mark.rss_limit_mb(1200)
    def test_public_forward_and_reverse_ad_follow_the_composite_route(
        self,
    ) -> None:
        """Require the composite route from public ``jvp`` and ``grad``.

        Acceptance: ``jax.jvp`` and ``jax.grad`` of the public callable in
        the query coordinate reproduce the committed composite route
        within ``1e-9 * max(1, |composite|)`` per query. Both also match
        the analytic derivative within ``2e-8 + 1e-6 * |truth|`` per
        query. The committed instrument measures ``3.0722e-9`` (mixed
        ratio ``0.0459``) for the composite route here. Direct
        differentiation of the transform reaches ``7.46e-7`` and fails
        the identity bound.

        Notes
        -----
        The test runs forward mode with a unit query tangent and reverse
        mode through the summed real part. It rebuilds the composite truth
        from the authenticated instrument modules.
        """
        models: Dict[str, Float64[NDArray, "..."]] = _authenticated_npz(
            _MODELS_ARCHIVE_PATH, _MODELS_ARCHIVE_SHA256
        )
        queries: Float64[Array, " n"] = jnp.asarray(
            models["eval_grid_ev"][::10], dtype=jnp.float64
        )
        model: SelfEnergyModel = _fermi_liquid_model()

        def public_map(
            points: Float64[Array, " n"],
        ) -> Complex128[Array, " n"]:
            return evaluate_self_energy(points, model)

        primal: Complex128[Array, " n"]
        tangent: Complex128[Array, " n"]
        primal, tangent = jax.jvp(
            public_map, (queries,), (jnp.ones_like(queries),)
        )
        assert jnp.iscomplexobj(primal), (
            "the public derivative demonstrator requires the complex "
            "retarded contract"
        )
        forward: Float64[NDArray, " n"] = np.real(np.asarray(tangent))

        def real_sum(points: Float64[Array, " n"]) -> Float64[Array, ""]:
            return jnp.sum(jnp.real(evaluate_self_energy(points, model)))

        reverse: Float64[NDArray, " n"] = np.asarray(
            jax.grad(real_sum)(queries)
        )

        common: ModuleType
        common, _ = _committed_operator()
        spec: Any = _fl_tail_spec(common)
        composite: Float64[NDArray, " n"] = np.asarray(
            _instrument_composite_derivative(
                _fl_sigma_imag_dynamic,
                _fl_dsigma_imag_dynamic,
                spec,
                queries,
            )
        )
        identity_bound: Float64[NDArray, " n"] = (
            _IDENTITY_SCALE_BOUND * np.maximum(1.0, np.abs(composite))
        )
        assert np.all(np.abs(forward - composite) <= identity_bound)
        assert np.all(np.abs(reverse - composite) <= identity_bound)
        assert np.all(np.abs(forward - reverse) <= identity_bound)

        analytic: Float64[NDArray, " n"] = _fl_dsigma_real_domega(
            np.asarray(queries)
        )
        mixed_bound: Float64[NDArray, " n"] = (
            _FL_VALUE_ATOL_EV + _FL_VALUE_RTOL * np.abs(analytic)
        )
        assert np.all(np.abs(forward - analytic) <= mixed_bound)
        assert np.all(np.abs(reverse - analytic) <= mixed_bound)

        imag_tangent: Float64[NDArray, " n"] = np.imag(np.asarray(tangent))
        np.testing.assert_allclose(
            imag_tangent,
            np.asarray(_fl_dsigma_imag_dynamic(queries)),
            rtol=1e-12,
            atol=1e-14,
        )

    def test_parameter_derivative_rows_match_frozen_complex_step_truths(
        self,
    ) -> None:
        """Match every frozen parameter-derivative row on the public map.

        Acceptance: for each dimensionless coordinate ``q_p = p / p_fix``
        the public forward-mode derivative matches the frozen complex-step
        rows at rtol ``1e-6``. The numerical Fermi-liquid real rows carry
        atol ``5e-8`` (committed instrument measurement ``2.579e-8``).
        Every analytic row carries atol ``1e-10``. A separate test covers
        the structural-zero columns.

        Notes
        -----
        The test rescales each fixture through its dimensionless
        coordinates and differentiates the public map per column. It then
        compares real and imaginary rows against the archive.
        """
        models: Dict[str, Float64[NDArray, "..."]] = _authenticated_npz(
            _MODELS_ARCHIVE_PATH, _MODELS_ARCHIVE_SHA256
        )
        cases: Tuple[Tuple[str, str, int], ...] = (
            ("fermi_liquid", "fl", 3),
            ("bosonic_kink", "kink", 4),
        )
        fixture: str
        prefix: str
        count: int
        column: int
        for fixture, prefix, count in cases:
            probes: Float64[Array, " n"] = jnp.asarray(
                models[f"{prefix}_probe_frequencies_ev"],
                dtype=jnp.float64,
            )
            real_rows: Float64[NDArray, " n n_param"] = models[
                f"{prefix}_dsigma_real_dq"
            ]
            imag_rows: Float64[NDArray, " n n_param"] = models[
                f"{prefix}_dsigma_imag_dq"
            ]

            def scaled_map(
                scale: Float64[Array, " n_param"],
                fixture_name: str = fixture,
                points: Float64[Array, " n"] = probes,
            ) -> Complex128[Array, " n"]:
                return evaluate_self_energy(
                    points, _scaled_model(fixture_name, scale)
                )

            ones: Float64[Array, " n_param"] = jnp.ones(
                count, dtype=jnp.float64
            )
            primal: Complex128[Array, " n"] = scaled_map(ones)
            assert jnp.iscomplexobj(primal), (
                "the parameter rows run on the complex retarded contract"
            )
            for column in range(count):
                tangent_direction: Float64[Array, " n_param"] = (
                    jnp.zeros(count, dtype=jnp.float64).at[column].set(1.0)
                )
                tangent: Complex128[Array, " n"]
                _, tangent = jax.jvp(scaled_map, (ones,), (tangent_direction,))
                real_atol: float = (
                    _FL_REAL_ROW_ATOL
                    if fixture == "fermi_liquid"
                    else _ANALYTIC_ROW_ATOL
                )
                if column > 0:
                    np.testing.assert_allclose(
                        np.real(np.asarray(tangent)),
                        real_rows[:, column],
                        rtol=_DERIVATIVE_ROW_RTOL,
                        atol=real_atol,
                    )
                np.testing.assert_allclose(
                    np.imag(np.asarray(tangent)),
                    imag_rows[:, column],
                    rtol=_DERIVATIVE_ROW_RTOL,
                    atol=_ANALYTIC_ROW_ATOL,
                )

    def test_structural_zero_gamma0_real_response_is_exactly_zero(
        self,
    ) -> None:
        """Assert the documented structural zeros with exact-zero rows.

        Acceptance: ``dSigma'/dGamma0`` equals zero exactly for both
        fixtures. The constant baseline stays imaginary-only, and the
        subtraction removes it before the Kramers-Kronig map. The rows
        carry exact-zero assertions, never nonzero tripwires.

        Notes
        -----
        The test differentiates the public map along the ``gamma0``
        coordinate for both fixtures. It then requires an exactly zero
        real tangent row.
        """
        models: Dict[str, Float64[NDArray, "..."]] = _authenticated_npz(
            _MODELS_ARCHIVE_PATH, _MODELS_ARCHIVE_SHA256
        )
        fixture: str
        prefix: str
        count: int
        for fixture, prefix, count in (
            ("fermi_liquid", "fl", 3),
            ("bosonic_kink", "kink", 4),
        ):
            probes: Float64[Array, " n"] = jnp.asarray(
                models[f"{prefix}_probe_frequencies_ev"],
                dtype=jnp.float64,
            )

            def scaled_map(
                scale: Float64[Array, " n_param"],
                fixture_name: str = fixture,
                points: Float64[Array, " n"] = probes,
            ) -> Complex128[Array, " n"]:
                return evaluate_self_energy(
                    points, _scaled_model(fixture_name, scale)
                )

            ones: Float64[Array, " n_param"] = jnp.ones(
                count, dtype=jnp.float64
            )
            gamma_direction: Float64[Array, " n_param"] = (
                jnp.zeros(count, dtype=jnp.float64).at[0].set(1.0)
            )
            primal: Complex128[Array, " n"]
            tangent: Complex128[Array, " n"]
            primal, tangent = jax.jvp(scaled_map, (ones,), (gamma_direction,))
            assert jnp.iscomplexobj(primal), (
                "the structural zeros run on the complex retarded contract"
            )
            np.testing.assert_array_equal(
                np.real(np.asarray(tangent)),
                np.zeros(probes.shape[0]),
            )


class TestKkTransformSeam(chex.TestCase):
    """Test the private cell-integrated transform seam.

    :see: :func:`diffpes.simul.spectral._kk_transform`
    """

    def test_private_transform_seam_signature_and_no_kernel_matrix(
        self,
    ) -> None:
        """Require the committed seam and forbid the kernel-matrix API.

        Acceptance: ``diffpes.simul.spectral._kk_transform`` exists with
        the exact parameters ``(core_grid, model_domain, tail_spec,
        queries, n_tail)``. The module defines no ``build_kk_kernel``
        dense ``[n_kk, n_kk]`` constructor.

        Notes
        -----
        The test imports the production module, inspects the seam
        signature, and checks the retired kernel name stays absent.
        """
        import diffpes.simul.spectral as spectral

        seam: Callable[..., Any] = spectral._kk_transform
        parameters: list[str] = list(inspect.signature(seam).parameters)
        assert parameters == [
            "core_grid",
            "model_domain",
            "tail_spec",
            "queries",
            "n_tail",
        ]
        assert not hasattr(spectral, "build_kk_kernel")

    @pytest.mark.big_mem
    @pytest.mark.rss_limit_mb(1200)
    def test_retarded_pole_real_part_matches_analytic_truth(self) -> None:
        """Match the 80-digit pole truth through the production seam.

        Acceptance: on the 1001 frozen queries with the committed base
        configuration, ``|err| <= 2e-8 + 1e-6 * |truth|`` per query row.
        The committed operator measures maximum error ``3.3940e-8 eV``
        with worst mixed ratio ``0.8240`` here. ``core_grid`` carries the
        frozen node positions and the sampled imaginary part.

        Notes
        -----
        The test samples the analytic pole on the frozen grid, attaches
        the committed tail contract, and subtracts at the frozen point. It
        then applies the registered mixed criterion per row.
        """
        import diffpes.simul.spectral as spectral

        selection: Dict[str, Float64[NDArray, "..."]] = _authenticated_npz(
            _SELECTION_ARCHIVE_PATH, _SELECTION_ARCHIVE_SHA256
        )
        common: ModuleType
        common, _ = _committed_operator()
        grid: Float64[Array, " n_kk"] = _base_grid()
        spec: Any = _pole_tail_spec(common)
        domain: Float64[Array, " 2"] = jnp.asarray(
            [_DOMAIN_LOW_EV, _DOMAIN_HIGH_EV], dtype=jnp.float64
        )
        queries: Float64[Array, " n"] = jnp.asarray(
            selection["queries_ev"], dtype=jnp.float64
        )
        stacked: Float64[Array, " m"] = jnp.concatenate(
            [queries, jnp.asarray([0.0], dtype=jnp.float64)]
        )
        transformed: Float64[Array, " m"] = spectral._kk_transform(
            (grid, _pole_sigma_imag(grid)),
            domain,
            spec,
            stacked,
            _N_TAIL,
        )
        subtracted: Float64[NDArray, " n"] = np.asarray(
            transformed[:-1] - transformed[-1]
        )
        truth: Float64[NDArray, " n"] = selection[
            "truth_pole_sigma_real_sub_ev"
        ]
        error: Float64[NDArray, " n"] = np.abs(subtracted - truth)
        assert np.all(
            error <= _PAIR_TRUTH_ATOL_EV + _PAIR_TRUTH_RTOL * np.abs(truth)
        )

    @pytest.mark.big_mem
    @pytest.mark.rss_limit_mb(1200)
    def test_retarded_pole_query_derivative_follows_composite_route(
        self,
    ) -> None:
        """Require the composite-route class from the seam derivative.

        Acceptance: forward-mode differentiation of the seam in the query
        coordinate matches the analytic pole derivative within ``2e-6``.
        The committed composite route measures ``9.3427e-7`` on the base
        configuration. Direct differentiation of the transform measures
        ``8.87e-5`` and fails this bound.

        Notes
        -----
        The test differentiates the seam with a unit query tangent on the
        1001 frozen queries. It then compares against the frozen analytic
        derivative array.
        """
        import diffpes.simul.spectral as spectral

        selection: Dict[str, Float64[NDArray, "..."]] = _authenticated_npz(
            _SELECTION_ARCHIVE_PATH, _SELECTION_ARCHIVE_SHA256
        )
        common: ModuleType
        common, _ = _committed_operator()
        grid: Float64[Array, " n_kk"] = _base_grid()
        spec: Any = _pole_tail_spec(common)
        domain: Float64[Array, " 2"] = jnp.asarray(
            [_DOMAIN_LOW_EV, _DOMAIN_HIGH_EV], dtype=jnp.float64
        )
        queries: Float64[Array, " n"] = jnp.asarray(
            selection["queries_ev"], dtype=jnp.float64
        )

        def seam_map(
            points: Float64[Array, " n"],
        ) -> Float64[Array, " n"]:
            return spectral._kk_transform(
                (grid, _pole_sigma_imag(grid)),
                domain,
                spec,
                points,
                _N_TAIL,
            )

        derivative: Float64[Array, " n"]
        _, derivative = jax.jvp(
            seam_map, (queries,), (jnp.ones_like(queries),)
        )
        error: float = float(
            np.max(
                np.abs(
                    np.asarray(derivative)
                    - selection["truth_pole_dsigma_domega"]
                )
            )
        )
        assert error <= _COMPOSITE_DERIVATIVE_BOUND


class TestSelfEnergyModelFactoryGuards(chex.TestCase):
    """Test the remaining carrier factory guard.

    :see: :func:`diffpes.types.make_self_energy_model`
    """

    def test_gamma_with_explicit_coefficients_is_rejected(self) -> None:
        """Reject a ``gamma`` shortcut next to explicit coefficients.

        Acceptance: the factory accepts the ``gamma`` shortcut only when
        ``coefficients`` stays absent and ``mode='constant'``. An explicit
        ``gamma`` together with ``coefficients`` raises instead of a
        silent drop.

        Notes
        -----
        The test drives the shared rejection helper with both arguments
        supplied. The helper repeats the call under ``eqx.filter_jit``.
        """
        assert_rejects(
            make_self_energy_model,
            coefficients=jnp.asarray([-2.0]),
            gamma=0.2,
            match="gamma",
        )


class TestGridModeHatTransform(chex.TestCase):
    """Test the exact grid-mode hat-interpolant transform.

    :see: :func:`diffpes.simul.spectral.evaluate_self_energy`
    """

    def test_three_node_hat_evaluation_matches_closed_form_transform(
        self,
    ) -> None:
        """Match a hand-computed closed-form hat principal value.

        Acceptance: for a three-node hat carrier the grid-mode real part
        equals the exact segment-wise principal value plus the frozen
        256-node power2 tail quadrature within
        ``1e-12 + 1e-12 * |truth|``. The committed selection evidence
        measures ``3.886e-16 eV`` for this operator class. The hat tail
        slopes come from the hat interpolant's edge segments.

        Notes
        -----
        The test computes the closed-form truth with NumPy only, using the
        documented per-segment formula and the frozen tail rule. It then
        compares the production grid-mode output per query.
        """
        nodes: Float64[NDArray, " three"] = np.asarray([-4.0, 0.0, 4.0])
        raw: Float64[NDArray, " three"] = np.asarray([-1.1, 0.4, -0.7])
        ordinates: Float64[NDArray, " three"] = -np.logaddexp(raw, 0.0)
        tail_raw: Tuple[float, float] = (-2.0, -1.5)
        subtraction_point: float = 0.3
        queries: Float64[NDArray, " four"] = np.asarray(
            [-1.3, -0.45, 0.62, 1.85]
        )

        amplitude_left: float = float(-ordinates[0])
        amplitude_right: float = float(-ordinates[-1])
        slope_left: float = float(
            (ordinates[1] - ordinates[0]) / (nodes[1] - nodes[0])
        )
        slope_right: float = float(
            (ordinates[2] - ordinates[1]) / (nodes[2] - nodes[1])
        )
        alpha_left: float = -slope_left / amplitude_left
        alpha_right: float = slope_right / amplitude_right
        beta_left: float = alpha_left**2 / 4.0 + float(
            np.logaddexp(tail_raw[0], 0.0)
        )
        beta_right: float = alpha_right**2 / 4.0 + float(
            np.logaddexp(tail_raw[1], 0.0)
        )

        evaluation_points: Float64[NDArray, " five"] = np.concatenate(
            [queries, np.asarray([subtraction_point])]
        )
        core: Float64[NDArray, " five"] = _hat_core_pv(
            nodes, ordinates, evaluation_points
        )
        tails: Float64[NDArray, " five"] = _hand_power2_tail(
            (float(nodes[0]), float(nodes[-1])),
            (amplitude_left, amplitude_right),
            (alpha_left, alpha_right),
            (beta_left, beta_right),
            evaluation_points,
        )
        total: Float64[NDArray, " five"] = core + tails
        truth: Float64[NDArray, " four"] = total[:-1] - total[-1]

        model: SelfEnergyModel = make_self_energy_model(
            coefficients=jnp.asarray(raw),
            mode="grid",
            energy_nodes_rel_fermi_ev=jnp.asarray(nodes),
            kk_consistent=True,
            kk_domain_rel_fermi_ev=jnp.asarray([-4.0, 4.0]),
            tail_coefficients=jnp.asarray(tail_raw),
            subtraction_point_rel_fermi_ev=subtraction_point,
            tail_mode="power2",
        )
        sigma: Complex128[Array, " four"] = evaluate_self_energy(
            jnp.asarray(queries), model
        )
        assert jnp.iscomplexobj(sigma), (
            "grid-mode evaluation must return the complex retarded self-energy"
        )
        error: Float64[NDArray, " four"] = np.abs(
            np.real(np.asarray(sigma)) - truth
        )
        assert np.all(
            error
            <= _GRID_EXACTNESS_ATOL_EV + _GRID_EXACTNESS_RTOL * np.abs(truth)
        )
        np.testing.assert_allclose(
            np.imag(np.asarray(sigma)),
            np.interp(queries, nodes, ordinates),
            rtol=0.0,
            atol=1e-14,
        )


class TestPlantedNoncompliantConstructions(chex.TestCase):
    """Test rejection of the planted noncompliant constructions.

    :see: :func:`diffpes.simul.spectral._kk_transform`
    """

    def _seam_arguments(
        self,
    ) -> Tuple[
        Tuple[Float64[Array, " n_kk"], Float64[Array, " n_kk"]],
        Float64[Array, " 2"],
        Any,
        Float64[Array, " n"],
    ]:
        """PRIVATE: Return one compliant seam argument set for mutation.

        Notes
        -----
        The set reuses the pole fixture and the committed tail contract.
        """
        common: ModuleType
        common, _ = _committed_operator()
        grid: Float64[Array, " n_kk"] = _base_grid()
        spec: Any = _pole_tail_spec(common)
        domain: Float64[Array, " 2"] = jnp.asarray(
            [_DOMAIN_LOW_EV, _DOMAIN_HIGH_EV], dtype=jnp.float64
        )
        queries: Float64[Array, " n"] = jnp.linspace(-1.0, 1.0, 11)
        return (grid, _pole_sigma_imag(grid)), domain, spec, queries

    def test_discontinuous_tail_edge_value_is_rejected(self) -> None:
        """Reject a tail whose edge value breaks the C1 match.

        Acceptance: a planted tail amplitude away from the core edge
        sample violates the continuity contract. Production raises on this
        construction.

        Notes
        -----
        The test scales one committed tail amplitude by 1.05 and drives
        the shared rejection helper on the seam.
        """
        import diffpes.simul.spectral as spectral

        core_grid: Tuple[Float64[Array, " n_kk"], Float64[Array, " n_kk"]]
        domain: Float64[Array, " 2"]
        spec: Any
        queries: Float64[Array, " n"]
        core_grid, domain, spec, queries = self._seam_arguments()
        planted: Any = spec._replace(amplitude_left=spec.amplitude_left * 1.05)
        assert_rejects(
            spectral._kk_transform,
            core_grid,
            domain,
            planted,
            queries,
            _N_TAIL,
            match="edge",
        )

    def test_sign_crossing_tail_denominator_is_rejected(self) -> None:
        """Reject a tail denominator that can cross zero.

        Acceptance: a planted negative quadratic tail coefficient lets
        ``1 + alpha*t + beta*t**2`` cross zero. Production raises on this
        construction.

        Notes
        -----
        The test replaces one committed ``beta`` with ``-0.01`` and drives
        the shared rejection helper on the seam.
        """
        import diffpes.simul.spectral as spectral

        core_grid: Tuple[Float64[Array, " n_kk"], Float64[Array, " n_kk"]]
        domain: Float64[Array, " 2"]
        spec: Any
        queries: Float64[Array, " n"]
        core_grid, domain, spec, queries = self._seam_arguments()
        planted: Any = spec._replace(
            beta_left=jnp.asarray(-0.01, dtype=jnp.float64)
        )
        assert_rejects(
            spectral._kk_transform,
            core_grid,
            domain,
            planted,
            queries,
            _N_TAIL,
            match="denominator|positive",
        )

    def test_silently_truncated_tail_is_rejected(self) -> None:
        """Reject a truncated semi-infinite tail quadrature.

        Acceptance: a zero tail order truncates both semi-infinite
        contributions. Production raises instead of evaluating the
        truncated transform.

        Notes
        -----
        The test passes ``n_tail=0`` and drives the shared rejection
        helper on the seam.
        """
        import diffpes.simul.spectral as spectral

        core_grid: Tuple[Float64[Array, " n_kk"], Float64[Array, " n_kk"]]
        domain: Float64[Array, " 2"]
        spec: Any
        queries: Float64[Array, " n"]
        core_grid, domain, spec, queries = self._seam_arguments()
        assert_rejects(
            spectral._kk_transform,
            core_grid,
            domain,
            spec,
            queries,
            0,
            match="n_tail|tail",
        )

    def test_query_window_built_grid_is_rejected(self) -> None:
        """Reject a quadrature grid built from the query extrema.

        Acceptance: a planted core grid that spans only the query window
        contradicts the declared model domain. Production raises because
        the domain always comes from the carrier.

        Notes
        -----
        The test rebuilds the grid from the query extrema and drives the
        shared rejection helper on the seam.
        """
        import diffpes.simul.spectral as spectral

        domain: Float64[Array, " 2"]
        spec: Any
        queries: Float64[Array, " n"]
        _, domain, spec, queries = self._seam_arguments()
        window_grid: Float64[Array, " n_kk"] = jnp.linspace(
            float(jnp.min(queries)), float(jnp.max(queries)), _N_KK
        )
        assert_rejects(
            spectral._kk_transform,
            (window_grid, _pole_sigma_imag(window_grid)),
            domain,
            spec,
            queries,
            _N_TAIL,
            match="domain|grid",
        )
