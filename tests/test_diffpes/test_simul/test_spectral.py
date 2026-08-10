"""Verify the production Kramers--Kronig spectral evaluation lane.

Extended Summary
----------------
The artifact, replication, production, and composition checks pin the
certified :mod:`diffpes.simul.spectral` contract. Every production test
states its registered acceptance number. Memory-intensive refinement rows
carry the ``big_mem`` marker and an explicit RSS-growth ceiling.

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

from diffpes.simul import (
    assemble_spectral_intensity_bands_chunk,
    assemble_spectral_intensity_chunk,
    evaluate_self_energy,
    projected_spectral_density_resolvent,
    spectral_intensity_eigen,
    spectral_intensity_resolvent,
    voigt,
)
from diffpes.tightb import bloch_hamiltonian
from diffpes.types import (
    EPS_DEG,
    KB_EV_PER_K,
    SelfEnergyModel,
    make_final_state_spec,
    make_matrix_element_params,
    make_orbital_basis,
    make_radial_quadrature_spec,
    make_radial_spec,
    make_self_energy_model,
)
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

# Frozen SHA-256 digests. The selection manifest records the archive and
# instrument digests; the constants below pin the manifests themselves.
_SELECTION_MANIFEST_SHA256: str = (
    "80cb05ba62f795ee4ab781b3dd328f673274d1afc44752500d62ce9a2ee43932"
)
_SELECTION_ARCHIVE_SHA256: str = (
    "e827e91c62e294afc50112af5fe484e5ff002070f106511d07bb813502648430"
)
_ANALYTIC_MANIFEST_SHA256: str = (
    "1c08269c503871c367d7610fd927d0a6d8f4b93012458be1e43e611f1d0f81a8"
)
_ANALYTIC_ARCHIVE_SHA256: str = (
    "ba30a3ee4e65658ace63ed54e65c3ec8ad8ae0868c396653123896265829cba5"
)
_ANALYTIC_GENERATOR_SHA256: str = (
    "0ab039848e5d16d4e8a540fdb17467754159cf97e567bde3dc3ba0794d3dcec1"
)
_MODELS_MANIFEST_SHA256: str = (
    "040c5a7bec3f7123b71d64e3314e17f1008c5f8536d984608bc942336e183165"
)
_MODELS_ARCHIVE_SHA256: str = (
    "59a115c16cbcbd57e7b70ec290380d224d2f2f73c5b6682afbcec047a4fe2830"
)
_MODELS_GENERATOR_SHA256: str = (
    "e65ea2c1117cc4fe33ca3bee00645a59547e2cd7cf5fbd04bc7a71556676f811"
)
_SELECTION_GENERATOR_SHA256: str = (
    "336d7e5f04533491f51b6fbed1abb1b1fde8f872403918bea847761b2671a6b8"
)
_COMMON_MODULE_SHA256: str = (
    "df2ecd4cd002e2e2da179ac4c8a0f663ddbf2f91ff3d8c6bd21c02addfc4de6e"
)
_CUBIC_MODULE_SHA256: str = (
    "a17e6f2c6b4ec3aee76d3ec116cb091bccc85bcc0edb59f908f1128aa2ea7900"
)
_LINEAR_MODULE_SHA256: str = (
    "f01469f8fa44163dc07c5f1e1be76facd4c4a8ad00584deef9b8e210a23d6bfb"
)
_QUADRATIC_MODULE_SHA256: str = (
    "83ae1ff85341782bac118447f1d9ab0381c1135a9fb00e4e79c83220c968cf3e"
)
_CONTROL_MODULE_SHA256: str = (
    "a02a72244c6c64a265a97bca05689f9033f80ea10099a1b702e148dc086e6775"
)
_SPECTRAL_INTENSITY_MANIFEST_SHA256: str = (
    "95cc321072627f2babe5d160ff89fea64ddb92a3674e5b0fecd9e8a2d7b2e929"
)
_SPECTRAL_INTENSITY_ARCHIVE_SHA256: str = (
    "67e5f9e18ad39a1b51e8ed5713003b684f35fff60df2637e9dd16e90f6ca547c"
)
_CHINOOK_SPECTRAL_MANIFEST_SHA256: str = (
    "11e3af71899b92bb7a77e818268e44793f89ed0d0ac5c274c20191ead3272fe9"
)
_CHINOOK_SPECTRAL_ARCHIVE_SHA256: str = (
    "5a6163b2566e09de2974873eea1bf6062782b3af41e3225fa1fe26eca2859c56"
)
_DEGENERATE_WITNESS_SHA256: str = (
    "3e9e555d5d037968a558869aa97938161a4af4f33d32c9976c47bdd8f43bdae2"
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
        assert (
            _sha256(_TOOLS_DIRECTORY / "generate_kk_operator_selection.py")
            == _SELECTION_GENERATOR_SHA256
        )
        assert executable["common"]["sha256"] == _COMMON_MODULE_SHA256
        assert (
            _sha256(_TOOLS_DIRECTORY / "_kk_candidate_common.py")
            == _COMMON_MODULE_SHA256
        )
        assert executable["piecewise_cubic"]["sha256"] == _CUBIC_MODULE_SHA256
        assert (
            executable["piecewise_linear"]["sha256"] == _LINEAR_MODULE_SHA256
        )
        assert (
            executable["piecewise_quadratic"]["sha256"]
            == _QUADRATIC_MODULE_SHA256
        )
        assert (
            executable["maclaurin_control"]["sha256"] == _CONTROL_MODULE_SHA256
        )
        name: str
        digest: str
        for name, digest in (
            ("_kk_candidate_piecewise_cubic.py", _CUBIC_MODULE_SHA256),
            ("_kk_candidate_piecewise_linear.py", _LINEAR_MODULE_SHA256),
            ("_kk_candidate_piecewise_quadratic.py", _QUADRATIC_MODULE_SHA256),
            (
                "_kk_control_opposite_parity_maclaurin.py",
                _CONTROL_MODULE_SHA256,
            ),
        ):
            assert _sha256(_TOOLS_DIRECTORY / name) == digest
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
        assert analytic["generator_sha256"] == _ANALYTIC_GENERATOR_SHA256
        assert (
            executable["analytic_arbiter"]["generator_sha256"]
            == _ANALYTIC_GENERATOR_SHA256
        )
        assert (
            executable["analytic_arbiter"]["manifest_sha256"]
            == _ANALYTIC_MANIFEST_SHA256
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
        assert (
            _sha256(
                _TOOLS_DIRECTORY / "generate_self_energy_models_reference.py"
            )
            == _MODELS_GENERATOR_SHA256
        )
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


class TestProductionKkConvergence(chex.TestCase):
    """Run the frozen refinement battery through production operators.

    :see: :func:`diffpes.simul.spectral._kk_transform`
    """

    @staticmethod
    def _uniform_grid(
        low: float, high: float, count: int
    ) -> Float64[NDArray, " n"]:
        """PRIVATE: Construct the frozen index-defined uniform grid.

        Notes
        -----
        The construction matches the selection instrument exactly and avoids
        endpoint redistribution by ``linspace``.
        """
        spacing: float = (high - low) / (count - 1)
        return low + np.arange(count, dtype=np.float64) * spacing

    @staticmethod
    def _production_core_only_subtracted(
        grid_np: Float64[NDArray, " n_kk"],
        queries_np: Float64[NDArray, " n_query"],
    ) -> Float64[NDArray, " n_query"]:
        """PRIVATE: Evaluate the production cubic under a zero-tail contract.

        Notes
        -----
        The Wigner fixture has compact support and exactly zero edge values on
        every registered domain. The plan explicitly routes this
        test-only analytic exception through the production cubic core, not
        through the positive-amplitude ``power2`` tail seam. Query chunks keep
        the matrix-free working set proportional to ``chunk * n_kk``.
        """
        import diffpes.simul.spectral as spectral

        grid: Float64[Array, " n_kk"] = jnp.asarray(grid_np)
        half_width: float = 1.5
        coupling: float = 0.2
        scale: float = 2.0 * coupling / half_width**2
        radicand: Float64[Array, " n_kk"] = jnp.maximum(
            half_width**2 - grid**2, 0.0
        )
        values: Float64[Array, " n_kk"] = jnp.where(
            jnp.abs(grid) < half_width,
            -scale * jnp.sqrt(radicand),
            0.0,
        )
        subtraction: float = float(
            spectral._cubic_core_pv(
                grid,
                values,
                jnp.asarray([0.0], dtype=jnp.float64),
            )[0]
        )
        chunks: list[Float64[NDArray, " chunk"]] = []
        start: int
        chunk_size: int = 64
        for start in range(0, queries_np.shape[0], chunk_size):
            points: Float64[Array, " chunk"] = jnp.asarray(
                queries_np[start : start + chunk_size]
            )
            chunks.append(
                np.asarray(spectral._cubic_core_pv(grid, values, points))
                - subtraction
            )
        return np.concatenate(chunks)

    @staticmethod
    def _pole_tail_raw(
        grid: Float64[Array, " n_kk"],
        *,
        hold_base_raw: bool,
    ) -> Tuple[float, float]:
        """PRIVATE: Return the frozen or domain-recomputed pole tail raws.

        Notes
        -----
        Same-domain refinements retain the base carrier coordinates. The
        phase-aligned domain extension recomputes them from the analytic pole
        before inspecting any production output.
        """
        if hold_base_raw:
            return _POLE_TAIL_RAW
        import diffpes.simul.spectral as spectral

        values: Float64[Array, " n_kk"] = _pole_sigma_imag(grid)
        spacing: Float64[Array, ""] = grid[1] - grid[0]
        slope_left: Float64[Array, ""]
        slope_right: Float64[Array, ""]
        slope_left, slope_right = spectral._cubic_edge_slopes(values, spacing)
        raw: list[float] = []
        edge: float
        amplitude: float
        alpha: float
        for edge, amplitude, alpha in (
            (
                float(grid[0]),
                float(-values[0]),
                float(-slope_left / (-values[0])),
            ),
            (
                float(grid[-1]),
                float(-values[-1]),
                float(slope_right / (-values[-1])),
            ),
        ):
            del amplitude
            beta_target: float = 1.0 / (
                (edge - _POLE_OMEGA0_EV) ** 2 + _POLE_GAMMA_EV**2
            )
            delta_beta: float = beta_target - alpha**2 / 4.0
            raw.append(float(np.log(np.expm1(delta_beta))))
        return raw[0], raw[1]

    @classmethod
    def _production_pole_subtracted(
        cls,
        grid_np: Float64[NDArray, " n_kk"],
        queries_np: Float64[NDArray, " n_query"],
        *,
        n_tail: int,
        hold_base_raw: bool,
    ) -> Float64[NDArray, " n_query"]:
        """PRIVATE: Evaluate one production pole refinement in query chunks.

        Notes
        -----
        The helper constructs the production C1 tail from the registered raw
        carrier coordinates, evaluates the mandatory seam directly, and
        subtracts the independently evaluated zero-frequency value.
        """
        import diffpes.simul.spectral as spectral

        grid: Float64[Array, " n_kk"] = jnp.asarray(grid_np)
        values: Float64[Array, " n_kk"] = _pole_sigma_imag(grid)
        spacing: Float64[Array, ""] = grid[1] - grid[0]
        slope_left: Float64[Array, ""]
        slope_right: Float64[Array, ""]
        slope_left, slope_right = spectral._cubic_edge_slopes(values, spacing)
        raw_left: float
        raw_right: float
        raw_left, raw_right = cls._pole_tail_raw(
            grid, hold_base_raw=hold_base_raw
        )
        spec: Any = spectral._power2_spec_from_edges(
            values[0],
            slope_left,
            values[-1],
            slope_right,
            jnp.asarray(raw_left),
            jnp.asarray(raw_right),
        )
        domain: Float64[Array, " 2"] = jnp.asarray(
            [grid_np[0], grid_np[-1]], dtype=jnp.float64
        )
        subtraction: float = float(
            spectral._kk_transform(
                (grid, values),
                domain,
                spec,
                jnp.asarray([0.0], dtype=jnp.float64),
                n_tail,
            )[0]
        )
        chunks: list[Float64[NDArray, " chunk"]] = []
        start: int
        chunk_size: int = 64
        for start in range(0, queries_np.shape[0], chunk_size):
            points: Float64[Array, " chunk"] = jnp.asarray(
                queries_np[start : start + chunk_size]
            )
            values_chunk: Float64[Array, " chunk"] = spectral._kk_transform(
                (grid, values), domain, spec, points, n_tail
            )
            chunks.append(np.asarray(values_chunk) - subtraction)
        return np.concatenate(chunks)

    @pytest.mark.big_mem
    @pytest.mark.rss_limit_mb(1200)
    def test_wigner_stress_witness_runs_through_production_cubic(self) -> None:
        """Certify Wigner convergence, order, and phase-aligned extension.

        Acceptance: production reproduces every frozen cubic array within
        ``2e-12 eV``. Errors decrease monotonically from 4096 to 8192 to
        16384 nodes, both observed orders reach ``1.4``, and the base error
        stays below ``1e-5 eV``. The phase-aligned extension embeds every
        base node bitwise and leaves the compact-support result unchanged.

        Notes
        -----
        Evaluate the explicit test-only zero-tail exception through
        :func:`diffpes.simul.spectral._cubic_core_pv`. This is the production
        smooth core selected for the singularity witness. Routing a zero edge
        through the positive-amplitude power2 carrier violates the plan.
        """
        selection: Dict[str, Float64[NDArray, "..."]] = _authenticated_npz(
            _SELECTION_ARCHIVE_PATH, _SELECTION_ARCHIVE_SHA256
        )
        queries: Float64[NDArray, " n"] = selection["queries_ev"]
        base_grid: Float64[NDArray, " n_base"] = self._uniform_grid(
            -8.0, 8.0, 4096
        )
        grid_8192: Float64[NDArray, " n_refined"] = self._uniform_grid(
            -8.0, 8.0, 8192
        )
        grid_16384: Float64[NDArray, " n_refined"] = self._uniform_grid(
            -8.0, 8.0, 16384
        )
        spacing: float = 16.0 / 4095.0
        extension_grid: Float64[NDArray, " n_extended"] = (
            -8.0 + (np.arange(8192, dtype=np.float64) - 2048.0) * spacing
        )
        np.testing.assert_array_equal(extension_grid[2048:6144], base_grid)

        produced: Dict[str, Float64[NDArray, " n"]] = {
            "base": self._production_core_only_subtracted(base_grid, queries),
            "grid8192": self._production_core_only_subtracted(
                grid_8192, queries
            ),
            "grid16384": self._production_core_only_subtracted(
                grid_16384, queries
            ),
            "domain_extension": self._production_core_only_subtracted(
                extension_grid, queries
            ),
        }
        configuration: str
        values: Float64[NDArray, " n"]
        for configuration, values in produced.items():
            np.testing.assert_allclose(
                values,
                selection[f"pwcubic_wigner_{configuration}_sigma_sub_ev"],
                rtol=0.0,
                atol=2.0e-12,
            )
        truth: Float64[NDArray, " n"] = selection[
            "truth_wigner_sigma_real_sub_ev"
        ]
        errors: Tuple[float, float, float] = tuple(
            float(np.max(np.abs(produced[name] - truth)))
            for name in ("base", "grid8192", "grid16384")
        )
        assert errors[0] <= 1.0e-5
        assert errors[0] > errors[1] > errors[2]
        assert np.log2(errors[0] / errors[1]) >= 1.4
        assert np.log2(errors[1] / errors[2]) >= 1.4
        np.testing.assert_allclose(
            produced["domain_extension"],
            produced["base"],
            rtol=0.0,
            atol=2.0e-12,
        )

    @pytest.mark.big_mem
    @pytest.mark.rss_limit_mb(1200)
    def test_pole_refinement_domain_and_tail_rules_use_production_seam(
        self,
    ) -> None:
        """Certify the pole value refinements through the production seam.

        Acceptance: each production configuration reproduces its frozen cubic
        array within ``2e-12 eV``. The 4096-to-8192 and phase-aligned-domain
        changes stay below ``2e-6 eV``; the 256-to-512 tail change stays below
        ``1e-13 eV``.

        Notes
        -----
        Same-domain refinement holds raw tail coordinates fixed. Domain
        extension recomputes them from the analytic fixture, exactly as the
        frozen carrier convention requires.
        """
        selection: Dict[str, Float64[NDArray, "..."]] = _authenticated_npz(
            _SELECTION_ARCHIVE_PATH, _SELECTION_ARCHIVE_SHA256
        )
        queries: Float64[NDArray, " n"] = selection["queries_ev"]
        base_grid: Float64[NDArray, " n_base"] = self._uniform_grid(
            -8.0, 8.0, 4096
        )
        grid_8192: Float64[NDArray, " n_refined"] = self._uniform_grid(
            -8.0, 8.0, 8192
        )
        spacing: float = 16.0 / 4095.0
        extension_grid: Float64[NDArray, " n_extended"] = (
            -8.0 + (np.arange(8192, dtype=np.float64) - 2048.0) * spacing
        )
        np.testing.assert_array_equal(extension_grid[2048:6144], base_grid)
        produced: Dict[str, Float64[NDArray, " n"]] = {
            "base": self._production_pole_subtracted(
                base_grid, queries, n_tail=256, hold_base_raw=True
            ),
            "grid8192": self._production_pole_subtracted(
                grid_8192, queries, n_tail=256, hold_base_raw=True
            ),
            "domain_extension": self._production_pole_subtracted(
                extension_grid, queries, n_tail=256, hold_base_raw=False
            ),
            "tail512": self._production_pole_subtracted(
                base_grid, queries, n_tail=512, hold_base_raw=True
            ),
        }
        configuration: str
        values: Float64[NDArray, " n"]
        for configuration, values in produced.items():
            np.testing.assert_allclose(
                values,
                selection[f"pwcubic_pole_{configuration}_sigma_sub_ev"],
                rtol=0.0,
                atol=2.0e-12,
            )
        assert (
            float(np.max(np.abs(produced["grid8192"] - produced["base"])))
            <= 2.0e-6
        )
        assert (
            float(
                np.max(np.abs(produced["domain_extension"] - produced["base"]))
            )
            <= 2.0e-6
        )
        assert (
            float(np.max(np.abs(produced["tail512"] - produced["base"])))
            <= 1.0e-13
        )


class TestProductionKkContinuityAndDerivatives(chex.TestCase):
    """Certify C1 seams, parameter derivatives, JIT, and VMAP success."""

    def test_power2_tails_match_production_edge_values_and_slopes(
        self,
    ) -> None:
        """Match smooth and hat tail values at rtol 1e-12 and slopes at 1e-9.

        The test probes both interpolation families at each domain boundary.

        Notes
        -----
        Construct one smooth pole seam and one grid-hat seam with production
        helpers. Evaluate the analytic tail value and outward-coordinate
        derivative at zero distance. Compare both with the owning core
        interpolant.
        """
        import diffpes.simul.spectral as spectral

        smooth_grid: Float64[Array, " n_kk"] = _base_grid()
        smooth_values: Float64[Array, " n_kk"] = _pole_sigma_imag(smooth_grid)
        smooth_left: Float64[Array, ""]
        smooth_right: Float64[Array, ""]
        smooth_left, smooth_right = spectral._cubic_edge_slopes(
            smooth_values, smooth_grid[1] - smooth_grid[0]
        )

        raw: Float64[Array, " four"] = jnp.asarray([-1.1, 0.4, -0.7, -1.0])
        nodes: Float64[Array, " four"] = jnp.asarray([-4.0, -1.2, 1.1, 4.0])
        hat_values: Float64[Array, " four"] = -jnp.logaddexp(raw, 0.0)
        hat_left: Float64[Array, ""] = (hat_values[1] - hat_values[0]) / (
            nodes[1] - nodes[0]
        )
        hat_right: Float64[Array, ""] = (hat_values[-1] - hat_values[-2]) / (
            nodes[-1] - nodes[-2]
        )
        cases: Tuple[
            Tuple[
                Float64[Array, ""],
                Float64[Array, ""],
                Float64[Array, ""],
                Float64[Array, ""],
            ],
            ...,
        ] = (
            (smooth_values[0], smooth_left, smooth_values[-1], smooth_right),
            (hat_values[0], hat_left, hat_values[-1], hat_right),
        )
        edge_left: Float64[Array, ""]
        slope_left: Float64[Array, ""]
        edge_right: Float64[Array, ""]
        slope_right: Float64[Array, ""]
        for edge_left, slope_left, edge_right, slope_right in cases:
            spec: Any = spectral._power2_spec_from_edges(
                edge_left,
                slope_left,
                edge_right,
                slope_right,
                jnp.asarray(-2.0),
                jnp.asarray(-1.5),
            )
            np.testing.assert_allclose(
                [-float(spec.amplitude_left), -float(spec.amplitude_right)],
                [float(edge_left), float(edge_right)],
                rtol=1.0e-12,
                atol=0.0,
            )
            np.testing.assert_allclose(
                [
                    -float(spec.amplitude_left * spec.alpha_left),
                    float(spec.amplitude_right * spec.alpha_right),
                ],
                [float(slope_left), float(slope_right)],
                rtol=1.0e-9,
                atol=1.0e-14,
            )
            assert float(spec.beta_left) >= float(spec.alpha_left**2 / 4.0)
            assert float(spec.beta_right) >= float(spec.alpha_right**2 / 4.0)

    def _assert_parameter_jacobian_matches_central_fd(self, mode: str) -> None:
        """PRIVATE: Match each coefficient column away from grid knots.

        Acceptance: production ``jacfwd`` matches a central finite-difference
        Jacobian at rtol ``1e-6`` and atol ``2e-8``. Every coefficient has a
        nonzero complex-response column. Grid queries and the subtraction point
        remain at least ``0.1 eV`` from every interpolation knot.

        Notes
        -----
        Rebuild the immutable carrier from each perturbed raw coordinate vector.
        The grid case also plants the knot derivative jump and verifies that the
        certified points lie strictly away from that nonsmooth set.
        """
        if mode == "poly":
            coefficients: Float64[Array, " n_coef"] = jnp.asarray(
                [-0.04, 0.08, -1.4]
            )
            queries: Float64[Array, " n"] = jnp.asarray(
                [-0.77, -0.21, 0.38, 0.93]
            )
            nodes: Float64[Array, " n_nodes"] | None = None
            subtraction: float = 0.13
        else:
            coefficients = jnp.asarray([-1.1, 0.4, -0.7, -1.0])
            queries = jnp.asarray([-0.83, -0.17, 0.46, 0.88])
            nodes = jnp.asarray([-4.0, -1.2, 1.1, 4.0])
            subtraction = 0.2
            distances: Float64[Array, "n n_nodes"] = jnp.abs(
                queries[:, None] - nodes[None, :]
            )
            assert float(jnp.min(distances)) > 0.1
            assert float(jnp.min(jnp.abs(nodes - subtraction))) > 0.1
            ordinates: Float64[Array, " n_nodes"] = -jnp.logaddexp(
                coefficients, 0.0
            )
            left_slope: float = float(
                (ordinates[1] - ordinates[0]) / (nodes[1] - nodes[0])
            )
            right_slope: float = float(
                (ordinates[2] - ordinates[1]) / (nodes[2] - nodes[1])
            )
            assert abs(left_slope - right_slope) > 1.0e-3

        def response(
            raw: Float64[Array, " n_coef"],
        ) -> Complex128[Array, " n"]:
            """Evaluate one coefficient vector on frozen geometry."""
            model: SelfEnergyModel = make_self_energy_model(
                coefficients=raw,
                mode=mode,
                energy_nodes_rel_fermi_ev=nodes,
                kk_consistent=True,
                kk_domain_rel_fermi_ev=jnp.asarray([-4.0, 4.0]),
                tail_coefficients=jnp.asarray([-2.2, -1.7]),
                subtraction_point_rel_fermi_ev=subtraction,
                tail_mode="power2",
            )
            return evaluate_self_energy(queries, model, n_kk=256)

        automatic: Complex128[Array, "n n_coef"] = jax.jacfwd(response)(
            coefficients
        )
        step: float = 2.0**-15
        fd_columns: list[Complex128[Array, " n"]] = []
        column: int
        for column in range(coefficients.shape[0]):
            direction: Float64[Array, " n_coef"] = (
                jnp.zeros_like(coefficients).at[column].set(step)
            )
            fd_columns.append(
                (
                    response(coefficients + direction)
                    - response(coefficients - direction)
                )
                / (2.0 * step)
            )
        finite_difference: Complex128[Array, "n n_coef"] = jnp.stack(
            fd_columns, axis=-1
        )
        np.testing.assert_allclose(
            np.asarray(automatic),
            np.asarray(finite_difference),
            rtol=1.0e-6,
            atol=2.0e-8,
        )
        column_norms: Float64[NDArray, " n_coef"] = np.max(
            np.abs(np.asarray(automatic)), axis=0
        )
        assert np.all(column_norms > 1.0e-8)

    def test_poly_parameter_jacobian_matches_central_fd(self) -> None:
        """Match every smooth polynomial coefficient against central FD.

        The shared helper compares the complete complex response Jacobian.

        Notes
        -----
        Polynomial coordinates remain smooth throughout the frozen domain.
        """
        self._assert_parameter_jacobian_matches_central_fd("poly")

    def test_grid_parameter_jacobian_matches_central_fd_away_from_knots(
        self,
    ) -> None:
        """Match every hat ordinate against central FD away from knots.

        The shared helper compares the complete complex response Jacobian.

        Notes
        -----
        Queries and the subtraction point stay outside every knot neighborhood.
        """
        self._assert_parameter_jacobian_matches_central_fd("grid")

    def test_numerical_mode_succeeds_under_jit_and_vmap(self) -> None:
        """Run a grid-mode success path through nested JIT and VMAP.

        Acceptance: compiled batched values equal the eager per-row reference
        at rtol ``1e-12``; the output stays finite, complex128, and retarded.

        Notes
        -----
        VMAP batches complete query vectors because the public API owns a
        one-dimensional energy axis. JIT wraps the batched numerical-KK path,
        including its traced validation predicates.
        """
        model: SelfEnergyModel = make_self_energy_model(
            coefficients=jnp.asarray([-1.1, 0.4, -0.7, -1.0]),
            mode="grid",
            energy_nodes_rel_fermi_ev=jnp.asarray([-4.0, -1.2, 1.1, 4.0]),
            kk_consistent=True,
            kk_domain_rel_fermi_ev=jnp.asarray([-4.0, 4.0]),
            tail_coefficients=jnp.asarray([-2.2, -1.7]),
            subtraction_point_rel_fermi_ev=0.2,
            tail_mode="power2",
        )
        batches: Float64[Array, "batch n"] = jnp.asarray(
            [[-0.9, -0.2, 0.5], [-0.7, 0.1, 0.8]]
        )

        def batched(
            points: Float64[Array, "batch n"],
        ) -> Complex128[Array, "batch n"]:
            """Batch the complete production evaluator by query row."""
            return jax.vmap(
                lambda row: evaluate_self_energy(row, model, n_kk=64)
            )(points)

        expected: Complex128[Array, "batch n"] = jnp.stack(
            [evaluate_self_energy(row, model, n_kk=64) for row in batches]
        )
        produced: Complex128[Array, "batch n"] = jax.jit(batched)(batches)
        np.testing.assert_allclose(
            np.asarray(produced),
            np.asarray(expected),
            rtol=1.0e-12,
            atol=1.0e-14,
        )
        assert produced.dtype == jnp.complex128
        assert bool(jnp.all(jnp.isfinite(produced)))
        assert bool(jnp.all(jnp.imag(produced) < 0.0))


def _spectral_intensity_reference() -> Dict[str, Float64[NDArray, "..."]]:
    """PRIVATE: Load and authenticate the preregistered WP7.5 archive.

    Notes
    -----
    Digest checks precede parsing of both the manifest and numeric archive.
    """
    manifest: Dict[str, Any] = _authenticated_json(
        _SPECTRAL_INTENSITY_MANIFEST_PATH,
        _SPECTRAL_INTENSITY_MANIFEST_SHA256,
    )
    assert manifest["schema"] == "diffpes.spectral-intensity-reference.v1"
    return _authenticated_npz(
        _SPECTRAL_INTENSITY_ARCHIVE_PATH,
        _SPECTRAL_INTENSITY_ARCHIVE_SHA256,
    )


def _degenerate_gradient_witness() -> Dict[str, Any]:
    """PRIVATE: Load and authenticate the two registered D4 witnesses.

    Notes
    -----
    The JSON contains the frozen graphene and Kramers coordinates and their
    independently measured central finite-difference ladders.
    """
    return _authenticated_json(
        _DEGENERATE_WITNESS_PATH,
        _DEGENERATE_WITNESS_SHA256,
    )


class TestSpectralIntensityResolvent(chex.TestCase):
    """Validate :func:`~diffpes.simul.spectral_intensity_resolvent`."""

    def test_two_pole_closed_form_and_degenerate_limit(self) -> None:
        """Match G4's frozen two-pole truth, including exact degeneracy.

        The test covers a full energy row and the coincident-pole limit.

        Notes
        -----
        One jitted vmap evaluates the full 2001-point energy row. The same
        source then probes the exact ``t=0`` limit without choosing an
        eigenvector basis.
        """
        reference: Dict[str, Float64[NDArray, "..."]] = (
            _spectral_intensity_reference()
        )
        epsilon: float = float(reference["two_pole_epsilon0"])
        hopping: float = float(reference["two_pole_hopping"])
        hamiltonian: Complex128[Array, "2 2"] = jnp.asarray(
            [[epsilon, hopping], [hopping, epsilon]],
            dtype=jnp.complex128,
        )
        degenerate: Complex128[Array, "2 2"] = epsilon * jnp.eye(
            2, dtype=jnp.complex128
        )
        source: Complex128[Array, "1 2"] = jnp.asarray(
            reference["two_pole_source_real"]
            + 1.0j * reference["two_pole_source_imag"],
            dtype=jnp.complex128,
        )[None, :]
        omega: Float64[Array, " n"] = jnp.asarray(reference["two_pole_omega"])
        sigma: Complex128[Array, ""] = jnp.asarray(
            -1.0j * reference["two_pole_gamma"],
            dtype=jnp.complex128,
        )
        eta: Float64[Array, ""] = jnp.asarray(reference["two_pole_eta"])

        def row(
            matrix: Complex128[Array, "2 2"],
        ) -> Float64[Array, " n"]:
            """Vectorize one Hamiltonian over the frozen axis."""
            return jax.vmap(
                spectral_intensity_resolvent,
                in_axes=(None, None, 0, None, None),
            )(matrix, source, omega, sigma, eta)

        actual: Float64[Array, " n"] = jax.jit(row)(hamiltonian)
        actual_degenerate: Float64[Array, " n"] = jax.jit(row)(degenerate)
        np.testing.assert_allclose(
            np.asarray(actual),
            reference["two_pole_intensity"],
            rtol=1.0e-10,
            atol=1.0e-13,
        )
        np.testing.assert_allclose(
            np.asarray(actual_degenerate),
            reference["two_pole_intensity_degenerate"],
            rtol=1.0e-10,
            atol=1.0e-13,
        )

    def test_outgoing_sources_solve_separately_before_sum(self) -> None:
        """Require the amended incoherent outgoing-channel reduction.

        The planted sources distinguish separate solves from coherent addition.

        Notes
        -----
        The two planted sources have a nonzero cross term. Coherently adding
        them before the solve therefore changes the answer and cannot satisfy
        the registered Plan-06 handoff.
        """
        hamiltonian: Complex128[Array, "2 2"] = jnp.asarray(
            [[-0.27, 0.09 + 0.04j], [0.09 - 0.04j, 0.31]]
        )
        sources: Complex128[Array, "2 2"] = jnp.asarray(
            [[0.8 + 0.2j, -0.3 + 0.5j], [0.4 - 0.7j, 0.6 + 0.1j]]
        )
        omega: Float64[Array, ""] = jnp.asarray(-0.08)
        sigma: Complex128[Array, ""] = jnp.asarray(0.01 - 0.05j)
        eta: float = 2.0e-4
        produced: Float64[Array, ""] = spectral_intensity_resolvent(
            hamiltonian,
            sources,
            omega,
            sigma,
            eta,
        )
        separate: Float64[Array, ""] = sum(
            spectral_intensity_resolvent(
                hamiltonian,
                sources[index : index + 1],
                omega,
                sigma,
                eta,
            )
            for index in range(2)
        )
        coherent: Float64[Array, ""] = spectral_intensity_resolvent(
            hamiltonian,
            jnp.sum(sources, axis=0, keepdims=True),
            omega,
            sigma,
            eta,
        )
        np.testing.assert_allclose(
            np.asarray(produced),
            np.asarray(separate),
            rtol=1.0e-13,
            atol=1.0e-14,
        )
        assert float(jnp.abs(produced - coherent)) > 1.0e-3

    def test_outgoing_source_axis_must_be_nonempty(self) -> None:
        """Reject an empty outgoing-channel axis before tracing a solve.

        The public scalar seam must always receive at least one source ket.

        Notes
        -----
        A Python shape guard rejects before tracing the linear solver.
        """
        with pytest.raises(ValueError, match="n_out|nonempty"):
            spectral_intensity_resolvent(
                jnp.eye(2, dtype=jnp.complex128),
                jnp.empty((0, 2), dtype=jnp.complex128),
                jnp.asarray(0.0),
                jnp.asarray(-0.04j),
                1.0e-4,
            )

    def test_generic_complex_adjoint_gradient(self) -> None:
        """Match D5's independent two-solve adjoint derivative truth.

        The test differentiates a generic complex-Hermitian two-level problem.

        Notes
        -----
        Four real coordinates span every independent entry of a generic
        complex-Hermitian two-orbital Hamiltonian. Lineax supplies reverse
        mode without a production custom derivative.
        """
        reference: Dict[str, Float64[NDArray, "..."]] = (
            _spectral_intensity_reference()
        )
        hamiltonian: Complex128[Array, "2 2"] = jnp.asarray(
            reference["hermitian_hamiltonian_real"]
            + 1.0j * reference["hermitian_hamiltonian_imag"]
        )
        source: Complex128[Array, "1 2"] = jnp.asarray(
            reference["hermitian_source_real"]
            + 1.0j * reference["hermitian_source_imag"]
        )[None, :]
        directions: Complex128[Array, "4 2 2"] = jnp.asarray(
            reference["adjoint_direction_real"]
            + 1.0j * reference["adjoint_direction_imag"]
        )
        sigma: Complex128[Array, ""] = jnp.asarray(
            -1.0j * reference["hermitian_gamma_sigma"],
            dtype=jnp.complex128,
        )
        eta: Float64[Array, ""] = jnp.asarray(reference["hermitian_eta"])
        omegas: Float64[Array, " n"] = jnp.asarray(reference["adjoint_omegas"])

        def gradient_at(omega: Float64[Array, ""]) -> Float64[Array, " 4"]:
            """Differentiate all Hermitian coordinates at omega."""

            def intensity(
                coordinates: Float64[Array, " 4"],
            ) -> Float64[Array, ""]:
                candidate: Complex128[Array, "2 2"] = (
                    hamiltonian
                    + jnp.tensordot(
                        coordinates,
                        directions,
                        axes=1,
                    )
                )
                return spectral_intensity_resolvent(
                    candidate,
                    source,
                    omega,
                    sigma,
                    eta,
                )

            return jax.grad(intensity)(jnp.zeros(4, dtype=jnp.float64))

        actual: Float64[Array, "n 4"] = jax.jit(jax.vmap(gradient_at))(omegas)
        np.testing.assert_allclose(
            np.asarray(actual),
            reference["adjoint_analytic"],
            rtol=1.0e-10,
            atol=1.0e-10,
        )

    def test_graphene_exact_degeneracy_parameter_gradient(self) -> None:
        """Match D4 for the one-bond coordinate at graphene K.

        The test differentiates through an exact orbital degeneracy.

        Notes
        -----
        The path varies the registered single bond. Both reverse and forward
        AD match the independently frozen finite-difference truth.
        """
        witness: Dict[str, Any] = _degenerate_gradient_witness()
        graphene: Dict[str, Any] = witness["graphene_one_bond_witness"]
        direction_entry: Dict[str, float] = graphene["measurements"][
            "one_bond_dh_dtheta_offdiag"
        ]
        off_diagonal: complex = complex(
            direction_entry["real"], direction_entry["imag"]
        )
        bond_direction: Complex128[Array, "2 2"] = jnp.asarray(
            [[0.0, off_diagonal], [off_diagonal.conjugate(), 0.0]],
            dtype=jnp.complex128,
        )
        graphene_source: Complex128[Array, "1 2"] = (
            jnp.asarray(graphene["intensity"]["source_real"])[None, :]
            + 1.0j * jnp.asarray(graphene["intensity"]["source_imag"])[None, :]
        )

        def graphene_intensity(
            coordinate: Float64[Array, ""],
        ) -> Float64[Array, ""]:
            """Evaluate the registered one-bond resolvent path."""
            hamiltonian: Complex128[Array, "2 2"] = coordinate * bond_direction
            return spectral_intensity_resolvent(
                hamiltonian,
                graphene_source,
                jnp.asarray(graphene["intensity"]["omega_ev"]),
                jnp.asarray(0.0j, dtype=jnp.complex128),
                graphene["intensity"]["eta_ev"],
            )

        zero: Float64[Array, ""] = jnp.asarray(0.0)
        graphene_reverse: Float64[Array, ""] = jax.grad(graphene_intensity)(
            zero
        )
        graphene_forward: Float64[Array, ""] = jax.jacfwd(graphene_intensity)(
            zero
        )
        graphene_truth: float = graphene["measurements"][
            "one_bond_grad_reverse"
        ]
        np.testing.assert_allclose(
            np.asarray([graphene_reverse, graphene_forward]),
            graphene_truth,
            rtol=1.0e-10,
            atol=1.0e-10,
        )

    def test_kramers_exact_degeneracy_parameter_gradient(self) -> None:
        """Match D4 for a crystal field at a Kramers-degenerate point.

        The test differentiates a perturbation that preserves each Kramers pair.

        Notes
        -----
        The spin-symmetric field preserves every Kramers pair. Both reverse
        and forward resolvent AD match the finest independently frozen
        central finite-difference rung.
        """
        from tests._factories import make_t2g_soc_model

        witness: Dict[str, Any] = _degenerate_gradient_witness()
        kramers: Dict[str, Any] = witness["t2g_soc_kramers_witness"]
        kramers_model: Any = make_t2g_soc_model(coupling=0.4)
        kramers_k: Float64[Array, " 3"] = jnp.asarray(
            kramers["kramers_k_fractional"]
        )
        kramers_hamiltonian: Complex128[Array, "6 6"] = bloch_hamiltonian(
            kramers_model, kramers_k
        )
        field_direction: Complex128[Array, "6 6"] = jnp.diag(
            jnp.asarray([1.0, 0.0, 0.0, 1.0, 0.0, 0.0])
        ).astype(jnp.complex128)
        kramers_source: Complex128[Array, "1 6"] = (
            jnp.asarray(kramers["intensity"]["source_real"])[None, :]
            + 1.0j * jnp.asarray(kramers["intensity"]["source_imag"])[None, :]
        )

        def kramers_intensity(
            coordinate: Float64[Array, ""],
        ) -> Float64[Array, ""]:
            """Evaluate the registered Kramers resolvent path."""
            return spectral_intensity_resolvent(
                kramers_hamiltonian + coordinate * field_direction,
                kramers_source,
                jnp.asarray(kramers["intensity"]["omega_ev"]),
                jnp.asarray(0.0j, dtype=jnp.complex128),
                kramers["intensity"]["gamma_ev"],
            )

        zero: Float64[Array, ""] = jnp.asarray(0.0)
        kramers_reverse: Float64[Array, ""] = jax.grad(kramers_intensity)(zero)
        kramers_forward: Float64[Array, ""] = jax.jacfwd(kramers_intensity)(
            zero
        )
        kramers_truth: float = kramers["measurements"][
            "crystal_field_fd_central"
        ][-1]
        np.testing.assert_allclose(
            np.asarray([kramers_reverse, kramers_forward]),
            kramers_truth,
            rtol=1.0e-5,
            atol=1.0e-8,
        )

    def test_invalid_physical_domains_reject_eager_and_jit(self) -> None:
        """Reject non-Hermitian H, advanced self-energy, and nonpositive eta.

        The test exercises the same physical-domain predicates eagerly and in JIT.

        Notes
        -----
        The shared traced rejection helper evaluates each predicate.
        """
        hamiltonian: Complex128[Array, "2 2"] = jnp.asarray(
            [[0.1, 0.2j], [0.1j, -0.3]],
            dtype=jnp.complex128,
        )
        source: Complex128[Array, "1 2"] = jnp.asarray(
            [[0.4 + 0.2j, -0.1 + 0.7j]]
        )
        omega: Float64[Array, ""] = jnp.asarray(0.05)
        sigma: Complex128[Array, ""] = jnp.asarray(-0.02j)
        assert_rejects(
            spectral_intensity_resolvent,
            hamiltonian,
            source,
            omega,
            sigma,
            1.0e-4,
            match="Hermitian",
        )
        hermitian: Complex128[Array, "2 2"] = (
            hamiltonian + hamiltonian.conj().T
        )
        assert_rejects(
            spectral_intensity_resolvent,
            hermitian,
            source,
            omega,
            jnp.asarray(1.0e-5j),
            1.0e-4,
            match="retarded|nonpositive",
        )
        assert_rejects(
            spectral_intensity_resolvent,
            hermitian,
            source,
            omega,
            sigma,
            0.0,
            match="eta|positive",
        )


class TestProjectedSpectralDensityResolvent(chex.TestCase):
    """Validate :func:`~diffpes.simul.projected_spectral_density_resolvent`."""

    def test_matrix_spectral_density_matches_independent_inverse(self) -> None:
        """Match the full Hermitian density and preserve its coherences.

        The test compares every matrix entry across three sampled energies.

        Notes
        -----
        The truth explicitly inverts the two-orbital matrix and forms the
        matrix anti-Hermitian part before projection. A jitted vmap checks
        three sampled energies.
        """
        hamiltonian: Complex128[Array, "2 2"] = jnp.asarray(
            [[0.2, -0.1 + 0.3j], [-0.1 - 0.3j, -0.4]]
        )
        transition: Complex128[Array, "3 2"] = jnp.asarray(
            [
                [1.0 + 0.2j, -0.3 + 0.7j],
                [0.1 - 0.4j, 0.8 + 0.5j],
                [-0.6 + 0.3j, 0.2 - 0.9j],
            ]
        )
        omegas: Float64[Array, " 3"] = jnp.asarray([-0.5, 0.0, 0.7])
        sigma: Complex128[Array, ""] = jnp.asarray(-0.03 - 0.04j)
        eta: Float64[Array, ""] = jnp.asarray(2.0e-4)

        def production(omega: Float64[Array, ""]) -> Complex128[Array, "3 3"]:
            """Evaluate one projected production density."""
            return projected_spectral_density_resolvent(
                hamiltonian,
                transition,
                omega,
                sigma,
                eta,
            )

        def truth(omega: Float64[Array, ""]) -> Complex128[Array, "3 3"]:
            """Compute the density through an explicit dense inverse."""
            identity: Complex128[Array, "2 2"] = jnp.eye(
                2, dtype=jnp.complex128
            )
            green: Complex128[Array, "2 2"] = jnp.linalg.inv(
                (omega + 1.0j * eta - sigma) * identity - hamiltonian
            )
            spectral: Complex128[Array, "2 2"] = -(green - green.conj().T) / (
                2.0j * jnp.pi
            )
            return transition @ spectral @ transition.conj().T

        actual: Complex128[Array, "3 3 3"] = jax.jit(jax.vmap(production))(
            omegas
        )
        expected: Complex128[Array, "3 3 3"] = jax.vmap(truth)(omegas)
        np.testing.assert_allclose(
            np.asarray(actual),
            np.asarray(expected),
            rtol=1.0e-12,
            atol=1.0e-12,
        )
        np.testing.assert_allclose(
            np.asarray(actual),
            np.asarray(actual.conj().swapaxes(-1, -2)),
            rtol=0.0,
            atol=1.0e-13,
        )
        assert bool(jnp.all(jnp.linalg.eigvalsh(actual) >= -1.0e-12))
        assert bool(jnp.any(jnp.abs(jnp.imag(actual[:, 0, 1])) > 1.0e-4))

    def test_projected_density_gradient_matches_central_difference(
        self,
    ) -> None:
        """Match a generic matrix-density gradient to a central difference.

        The scalar loss retains diagonal and off-diagonal density information.

        Notes
        -----
        The scalar loss retains off-diagonal density entries so the test
        exercises complex cotangents through all multiple right-hand sides.
        """
        base: Complex128[Array, "2 2"] = jnp.asarray(
            [[-0.2, 0.11 + 0.08j], [0.11 - 0.08j, 0.3]]
        )
        transition: Complex128[Array, "2 2"] = jnp.asarray(
            [[0.7 + 0.2j, -0.4 + 0.1j], [0.3 - 0.8j, 0.5 + 0.6j]]
        )

        def loss(coordinate: Float64[Array, ""]) -> Float64[Array, ""]:
            """Compute a real loss from the projected density."""
            candidate: Complex128[Array, "2 2"] = base.at[0, 0].add(
                coordinate + 0.0j
            )
            density: Complex128[Array, "2 2"] = (
                projected_spectral_density_resolvent(
                    candidate,
                    transition,
                    jnp.asarray(0.07),
                    jnp.asarray(-0.025j),
                    1.0e-4,
                )
            )
            return jnp.real(density[0, 0] + 0.3j * density[0, 1])

        zero: Float64[Array, ""] = jnp.asarray(0.0)
        reverse: Float64[Array, ""] = jax.grad(loss)(zero)
        forward: Float64[Array, ""] = jax.jacfwd(loss)(zero)
        step: float = 2.0**-16
        finite_difference: Float64[Array, ""] = (
            loss(zero + step) - loss(zero - step)
        ) / (2.0 * step)
        np.testing.assert_allclose(
            np.asarray([reverse, forward]),
            np.asarray(finite_difference),
            rtol=1.0e-6,
            atol=1.0e-8,
        )


class TestSpectralIntensityEigen(chex.TestCase):
    """Validate :func:`~diffpes.simul.spectral_intensity_eigen`."""

    def test_generic_hermitian_resolvent_equivalence(self) -> None:
        """Match G4b/G6 values on the frozen complex-Hermitian fixture.

        The two public representations consume independently prepared inputs.

        Notes
        -----
        The eigen path consumes only the independently diagonalized values
        and gauge-invariant weights from the immutable archive.
        """
        reference: Dict[str, Float64[NDArray, "..."]] = (
            _spectral_intensity_reference()
        )
        eigenvalues: Float64[Array, " 2"] = jnp.asarray(
            reference["hermitian_eigenvalues"]
        )
        weights: Float64[Array, " 2"] = jnp.asarray(
            reference["hermitian_band_weights"]
        )
        omega: Float64[Array, " n"] = jnp.asarray(reference["hermitian_omega"])
        sigma: Complex128[Array, ""] = jnp.asarray(
            -1.0j * reference["hermitian_gamma_sigma"],
            dtype=jnp.complex128,
        )
        eta: Float64[Array, ""] = jnp.asarray(reference["hermitian_eta"])
        actual: Float64[Array, " n"] = jax.jit(
            jax.vmap(
                spectral_intensity_eigen,
                in_axes=(None, None, 0, None, None),
            )
        )(eigenvalues, weights, omega, sigma, eta)
        np.testing.assert_allclose(
            np.asarray(actual),
            reference["hermitian_intensity"],
            rtol=1.0e-10,
            atol=1.0e-13,
        )

    def test_eta_regulator_ladder(self) -> None:
        """Match every G5b regulator rung and its frozen convergence rows.

        The test checks values and convergence against the independent archive.

        Notes
        -----
        The one-level fixture isolates the rule that eta enters only through
        the total Lorentzian linewidth ``Gamma + eta``.
        """
        reference: Dict[str, Float64[NDArray, "..."]] = (
            _spectral_intensity_reference()
        )
        eigenvalue: Float64[Array, " 1"] = jnp.atleast_1d(
            jnp.asarray(reference["eta_ladder_level_energy"])
        )
        weight: Float64[Array, " 1"] = jnp.ones(1, dtype=jnp.float64)
        omega: Float64[Array, " n"] = jnp.asarray(
            reference["eta_ladder_omega"]
        )
        sigma: Complex128[Array, ""] = jnp.asarray(
            -1.0j * reference["eta_ladder_gamma_physical"],
            dtype=jnp.complex128,
        )

        def row(eta: Float64[Array, ""]) -> Float64[Array, " n"]:
            """Evaluate one regulator rung over all energies."""
            return jax.vmap(
                spectral_intensity_eigen,
                in_axes=(None, None, 0, None, None),
            )(eigenvalue, weight, omega, sigma, eta)

        actual: Float64[Array, "rung n"] = jax.jit(jax.vmap(row))(
            jnp.asarray(reference["eta_ladder_etas"])
        )
        np.testing.assert_allclose(
            np.asarray(actual),
            reference["eta_ladder_intensities"],
            rtol=1.0e-10,
            atol=1.0e-13,
        )
        captured: Float64[Array, " rung"] = jnp.trapezoid(actual, omega)
        np.testing.assert_allclose(
            np.asarray(captured),
            reference["eta_ladder_captured_masses"],
            rtol=1.0e-5,
            atol=2.0e-7,
        )

    def test_off_degenerate_value_and_gradient_equivalence(self) -> None:
        """Match G6 resolvent and eigen values and hopping gradients.

        The fixture remains safely above the differentiated eigen gap floor.

        Notes
        -----
        The two-pole gap remains far above the degeneracy tolerance. Its
        eigenvectors only form the invariant band weights.
        """
        epsilon: float = -0.15
        source: Complex128[Array, " 2"] = jnp.asarray(
            [0.9 + 0.4j, -0.5 + 0.7j]
        )
        omega: Float64[Array, ""] = jnp.asarray(-0.08)
        sigma: Complex128[Array, ""] = jnp.asarray(-0.03j)
        eta: float = 1.0e-4

        def pair(hopping: Float64[Array, ""]) -> Float64[Array, " 2"]:
            """Return resolver and eigen intensity at one hopping."""
            hamiltonian: Complex128[Array, "2 2"] = jnp.asarray(
                [[epsilon, hopping], [hopping, epsilon]],
                dtype=jnp.complex128,
            )
            eigenvalues: Float64[Array, " 2"]
            eigenvectors: Complex128[Array, "2 2"]
            eigenvalues, eigenvectors = jnp.linalg.eigh(hamiltonian)
            weights: Float64[Array, " 2"] = (
                jnp.abs(eigenvectors.conj().T @ source) ** 2
            )
            return jnp.stack(
                [
                    spectral_intensity_resolvent(
                        hamiltonian, source[None, :], omega, sigma, eta
                    ),
                    spectral_intensity_eigen(
                        eigenvalues, weights, omega, sigma, eta
                    ),
                ]
            )

        hopping: Float64[Array, ""] = jnp.asarray(0.08)
        values: Float64[Array, " 2"] = jax.jit(pair)(hopping)
        gradients: Float64[Array, " 2"] = jax.jacfwd(pair)(hopping)
        np.testing.assert_allclose(
            np.asarray(values[0]),
            np.asarray(values[1]),
            rtol=1.0e-10,
            atol=1.0e-12,
        )
        np.testing.assert_allclose(
            np.asarray(gradients[0]),
            np.asarray(gradients[1]),
            rtol=1.0e-10,
            atol=1.0e-10,
        )

    def test_negative_band_weight_rejects(self) -> None:
        """Reject a negative input weight eagerly and inside JIT.

        The public eigen seam enforces nonnegative squared amplitudes.

        Notes
        -----
        Band weights are gauge-invariant squared amplitudes and cannot carry
        a negative signed contribution.
        """
        assert_rejects(
            spectral_intensity_eigen,
            jnp.asarray([-0.2, 0.3]),
            jnp.asarray([1.0, -0.1]),
            jnp.asarray(0.0),
            jnp.asarray(-0.02j),
            1.0e-4,
            match="weights|nonnegative",
        )

    def test_nondegenerate_domain_floor_and_value_only_exception(self) -> None:
        """Enforce the differentiated gap floor eagerly and inside JIT.

        The test probes exact, sub-floor, boundary, and explicit primal cases.

        Notes
        -----
        The registered G6 domain includes its exact lower boundary. Exact and
        sub-floor pairs reject by default. A degenerate primal requires the
        explicit value-only policy and emits no derivative evidence.
        """
        gap_floor: float = 1.0e3 * EPS_DEG
        below_floor: float = float(np.nextafter(gap_floor, 0.0))
        weights: Float64[Array, " 2"] = jnp.asarray([0.7, 0.4])
        omega: Float64[Array, ""] = jnp.asarray(0.03)
        sigma: Complex128[Array, ""] = jnp.asarray(-0.02j)
        eigenvalues: Float64[Array, " 2"]
        for eigenvalues in (
            jnp.asarray([0.0, 0.0]),
            jnp.asarray([0.0, below_floor]),
        ):
            assert_rejects(
                spectral_intensity_eigen,
                eigenvalues,
                weights,
                omega,
                sigma,
                1.0e-4,
                match="gap|resolvent|value_only",
            )

        boundary: Float64[Array, " 2"] = jnp.asarray([0.0, gap_floor])
        accepted: Float64[Array, ""] = jax.jit(spectral_intensity_eigen)(
            boundary,
            weights,
            omega,
            sigma,
            1.0e-4,
        )
        value_only: Float64[Array, ""] = jax.jit(
            lambda eigenvalues: spectral_intensity_eigen(
                eigenvalues,
                weights,
                omega,
                sigma,
                1.0e-4,
                allow_degenerate_value_only=True,
            )
        )(jnp.asarray([0.0, 0.0]))
        assert bool(jnp.all(jnp.isfinite(jnp.asarray([accepted, value_only]))))


class TestAssembleSpectralIntensityChunk(chex.TestCase):
    """Validate :func:`~diffpes.simul.assemble_spectral_intensity_chunk`."""

    @staticmethod
    def _fixture() -> Tuple[
        Complex128[Array, "2 2 2"],
        Complex128[Array, "2 5 2 2"],
        Float64[Array, " 5"],
        SelfEnergyModel,
        Float64[Array, ""],
    ]:
        """PRIVATE: Return one generic two-k coherent assembly fixture.

        Notes
        -----
        The absolute Hamiltonians share a nonzero Fermi offset. Sources vary
        with sampled energy so no early matrix-element reduction can pass.
        """
        fermi_energy: Float64[Array, ""] = jnp.asarray(1.7)
        relative: Complex128[Array, "2 2 2"] = jnp.asarray(
            [
                [[-0.3, 0.08 + 0.03j], [0.08 - 0.03j, 0.2]],
                [[-0.1, -0.05 + 0.07j], [-0.05 - 0.07j, 0.35]],
            ],
            dtype=jnp.complex128,
        )
        hamiltonians: Complex128[Array, "2 2 2"] = relative + (
            fermi_energy * jnp.eye(2, dtype=jnp.complex128)[None, :, :]
        )
        omega: Float64[Array, " 5"] = jnp.asarray(
            [-0.35, -0.12, 0.0, 0.24, 2.0]
        )
        source_base: Complex128[Array, "2 2 2"] = jnp.asarray(
            [
                [[0.8 + 0.2j, -0.3 + 0.5j], [0.1 - 0.4j, 0.6 + 0.2j]],
                [[0.2 - 0.6j, 0.7 + 0.1j], [-0.5 + 0.3j, 0.2 + 0.8j]],
            ]
        )
        scales: Complex128[Array, " 5"] = jnp.asarray(
            [1.0, 0.8 + 0.1j, 1.1 - 0.2j, 0.6 + 0.3j, 0.9 - 0.1j]
        )
        sources: Complex128[Array, "2 5 2 2"] = (
            source_base[:, None, :, :] * scales[None, :, None, None]
        )
        model: SelfEnergyModel = make_self_energy_model(gamma=0.04)
        return hamiltonians, sources, omega, model, fermi_energy

    def test_analytic_composition_and_single_fermi_shift(self) -> None:
        """Match a dense NumPy-style composition and one energy shift.

        The test also proves invariance under a common absolute-energy offset.

        Notes
        -----
        The independent expression uses ``jnp.linalg.solve`` and an analytic
        sigmoid. Shifting both absolute H and E_F by the same constant leaves
        the relative observable unchanged bit for bit within roundoff.
        """
        hamiltonians: Complex128[Array, "2 2 2"]
        sources: Complex128[Array, "2 5 2 2"]
        omega: Float64[Array, " 5"]
        model: SelfEnergyModel
        fermi_energy: Float64[Array, ""]
        hamiltonians, sources, omega, model, fermi_energy = self._fixture()
        temperature: float = 18.0
        eta: float = 2.0e-4
        actual: Float64[Array, "2 5"] = jax.jit(
            assemble_spectral_intensity_chunk
        )(
            hamiltonians,
            sources,
            omega,
            model,
            fermi_energy,
            temperature,
            eta,
        )
        sigma: Complex128[Array, ""] = jnp.asarray(-0.04j)
        relative: Complex128[Array, "2 2 2"] = hamiltonians - (
            fermi_energy * jnp.eye(2, dtype=jnp.complex128)[None, :, :]
        )

        def one(
            hamiltonian: Complex128[Array, "2 2"],
            sources_at_sample: Complex128[Array, "n_out 2"],
            sampled: Float64[Array, ""],
        ) -> Float64[Array, ""]:
            """Sum independent dense solves for one sample."""
            operator: Complex128[Array, "2 2"] = (
                sampled + 1.0j * eta - sigma
            ) * jnp.eye(2, dtype=jnp.complex128) - hamiltonian
            spectral: Float64[Array, ""] = jnp.sum(
                jax.vmap(
                    lambda source: (
                        -jnp.imag(
                            jnp.vdot(
                                source, jnp.linalg.solve(operator, source)
                            )
                        )
                        / jnp.pi
                    )
                )(sources_at_sample)
            )
            occupation: Float64[Array, ""] = jax.nn.sigmoid(
                -sampled / (KB_EV_PER_K * temperature)
            )
            return spectral * occupation

        expected: Float64[Array, "2 5"] = jax.vmap(
            lambda hamiltonian, source_row: jax.vmap(
                one, in_axes=(None, 0, 0)
            )(hamiltonian, source_row, omega)
        )(relative, sources)
        np.testing.assert_allclose(
            np.asarray(actual),
            np.asarray(expected),
            rtol=1.0e-12,
            atol=1.0e-13,
        )
        shifted: Float64[Array, "2 5"] = assemble_spectral_intensity_chunk(
            hamiltonians + 3.25 * jnp.eye(2)[None, :, :],
            sources,
            omega,
            model,
            fermi_energy + 3.25,
            temperature,
            eta,
        )
        np.testing.assert_allclose(
            np.asarray(shifted),
            np.asarray(actual),
            rtol=1.0e-12,
            atol=1.0e-13,
        )

    def test_outgoing_source_axis_must_be_nonempty(self) -> None:
        """Reject an empty output axis on the public chunk boundary.

        The chunk assembler must retain at least one incoherent source channel.

        Notes
        -----
        A Python guard rejects the zero-length axis before tracing a batched solve.
        """
        hamiltonians: Complex128[Array, "2 2 2"]
        omega: Float64[Array, " 5"]
        model: SelfEnergyModel
        fermi_energy: Float64[Array, ""]
        hamiltonians, _, omega, model, fermi_energy = self._fixture()
        with pytest.raises(ValueError, match="n_out|nonempty"):
            assemble_spectral_intensity_chunk(
                hamiltonians,
                jnp.empty((2, 5, 0, 2), dtype=jnp.complex128),
                omega,
                model,
                fermi_energy,
                18.0,
            )

    def test_temperature_eta_gradients_and_vmap(self) -> None:
        """Match D3 gradients to FD and vmap the complete public assembly.

        Temperature and regulator derivatives traverse the full composition.

        Notes
        -----
        The frozen axis includes a ``+2 eV`` sample at 15 K, pinning the
        former overflow-NaN regime while other samples retain nonzero thermal
        sensitivity.
        """
        hamiltonians: Complex128[Array, "2 2 2"]
        sources: Complex128[Array, "2 5 2 2"]
        omega: Float64[Array, " 5"]
        model: SelfEnergyModel
        fermi_energy: Float64[Array, ""]
        hamiltonians, sources, omega, model, fermi_energy = self._fixture()

        def loss_temperature(
            temperature: Float64[Array, ""],
        ) -> Float64[Array, ""]:
            """Sum the assembly at one traced temperature."""
            return jnp.sum(
                assemble_spectral_intensity_chunk(
                    hamiltonians,
                    sources,
                    omega,
                    model,
                    fermi_energy,
                    temperature,
                    0.01,
                )
            )

        temperature: Float64[Array, ""] = jnp.asarray(15.0)
        temperature_step: float = 2.0**-12
        temperature_grad: Float64[Array, ""] = jax.grad(loss_temperature)(
            temperature
        )
        temperature_fd: Float64[Array, ""] = (
            loss_temperature(temperature + temperature_step)
            - loss_temperature(temperature - temperature_step)
        ) / (2.0 * temperature_step)
        np.testing.assert_allclose(
            np.asarray(temperature_grad),
            np.asarray(temperature_fd),
            rtol=1.0e-6,
            atol=1.0e-9,
        )
        assert bool(jnp.isfinite(temperature_grad))

        def loss_eta(eta: Float64[Array, ""]) -> Float64[Array, ""]:
            """Sum the assembly at one traced regulator."""
            return jnp.sum(
                assemble_spectral_intensity_chunk(
                    hamiltonians,
                    sources,
                    omega,
                    model,
                    fermi_energy,
                    15.0,
                    eta,
                )
            )

        eta: Float64[Array, ""] = jnp.asarray(0.01)
        eta_step: float = 2.0**-16
        eta_grad: Float64[Array, ""] = jax.grad(loss_eta)(eta)
        eta_fd: Float64[Array, ""] = (
            loss_eta(eta + eta_step) - loss_eta(eta - eta_step)
        ) / (2.0 * eta_step)
        np.testing.assert_allclose(
            np.asarray(eta_grad),
            np.asarray(eta_fd),
            rtol=1.0e-6,
            atol=1.0e-8,
        )
        temperatures: Float64[Array, " 2"] = jnp.asarray([15.0, 22.0])
        batched: Float64[Array, "2 2 5"] = jax.jit(
            jax.vmap(
                lambda value: assemble_spectral_intensity_chunk(
                    hamiltonians,
                    sources,
                    omega,
                    model,
                    fermi_energy,
                    value,
                    0.01,
                )
            )
        )(temperatures)
        assert batched.shape == (2, 2, 5)
        assert bool(jnp.all(jnp.isfinite(batched)))

    @pytest.mark.big_mem
    @pytest.mark.rss_limit_mb(1200)
    def test_poly_coefficient_gradient_through_kk_and_resolvent(self) -> None:
        """Match D2 through a poly KK map and the complete intensity solve.

        The test differentiates raw self-energy coordinates through all layers.

        Notes
        -----
        Every raw polynomial coordinate remains connected to the scalar loss;
        a central difference checks the generic linear coefficient.
        """
        hamiltonians: Complex128[Array, "2 2 2"]
        sources: Complex128[Array, "2 5 2 2"]
        fermi_energy: Float64[Array, ""]
        hamiltonians, sources, _, _, fermi_energy = self._fixture()
        omega: Float64[Array, " 2"] = jnp.asarray([-0.2, 0.13])
        source_subset: Complex128[Array, "2 2 2 2"] = sources[:, :2, :, :]
        base: Float64[Array, " 3"] = jnp.asarray([-1.4, 0.25, -0.8])

        def loss(coefficients: Float64[Array, " 3"]) -> Float64[Array, ""]:
            """Assemble a scalar loss from one poly model."""
            model: SelfEnergyModel = make_self_energy_model(
                coefficients=coefficients,
                mode="poly",
                kk_consistent=True,
                kk_domain_rel_fermi_ev=jnp.asarray([-4.0, 4.0]),
                tail_coefficients=jnp.asarray([-2.0, -1.7]),
                subtraction_point_rel_fermi_ev=0.0,
                tail_mode="power2",
            )
            return jnp.sum(
                assemble_spectral_intensity_chunk(
                    hamiltonians,
                    source_subset,
                    omega,
                    model,
                    fermi_energy,
                    20.0,
                    1.0e-3,
                )
            )

        gradient: Float64[Array, " 3"] = jax.grad(loss)(base)
        assert bool(jnp.all(jnp.isfinite(gradient)))
        assert bool(jnp.all(jnp.abs(gradient) > 1.0e-8))
        step: float = 2.0**-14
        direction: Float64[Array, " 3"] = jnp.asarray([0.0, 1.0, 0.0])
        finite_difference: Float64[Array, ""] = (
            loss(base + step * direction) - loss(base - step * direction)
        ) / (2.0 * step)
        np.testing.assert_allclose(
            np.asarray(gradient[1]),
            np.asarray(finite_difference),
            rtol=1.0e-6,
            atol=1.0e-8,
        )


class TestAssembleSpectralIntensityBandsChunk(chex.TestCase):
    """Validate :func:`~diffpes.simul.assemble_spectral_intensity_bands_chunk`."""

    def test_sampled_omega_fermi_counterexample(self) -> None:
        """Match the frozen sampled-energy occupation witness.

        The counterexample distinguishes sampled omega from band-energy Fermi use.

        Notes
        -----
        A band above the Fermi level retains its occupied-side Lorentzian tail
        only with sampled-omega occupation. The tiny positive eta approximates
        the preregistered eta-free analytic row.
        """
        reference: Dict[str, Float64[NDArray, "..."]] = (
            _spectral_intensity_reference()
        )
        omega: Float64[Array, " n"] = jnp.asarray(reference["fermi_omega"])
        eigenvalues: Float64[Array, "1 1"] = jnp.asarray(
            [[reference["fermi_band_energy"]]]
        )
        weights: Float64[Array, "1 n 1"] = jnp.ones(
            (1, omega.shape[0], 1), dtype=jnp.float64
        )
        model: SelfEnergyModel = make_self_energy_model(
            gamma=float(reference["fermi_gamma"])
        )
        actual: Float64[Array, "1 n"] = jax.jit(
            assemble_spectral_intensity_bands_chunk
        )(
            eigenvalues,
            weights,
            omega,
            model,
            jnp.asarray(0.0),
            float(reference["fermi_temperature_k"]),
            1.0e-12,
        )
        np.testing.assert_allclose(
            np.asarray(actual[0]),
            reference["fermi_intensity_correct"],
            rtol=1.0e-9,
            atol=1.0e-12,
        )
        occupied_index: int = int(jnp.argmin(jnp.abs(omega + 0.1)))
        assert actual[0, occupied_index] > (
            1.0e12 * reference["fermi_intensity_wrong"][occupied_index]
        )

    def test_resolvent_and_band_chunk_paths_agree(self) -> None:
        """Match both chunk assemblers on generic nondegenerate bands.

        The test compares the coherent resolvent and invariant-weight formulas.

        Notes
        -----
        Explicit eigendecomposition forms invariant source weights at every
        sampled energy; the two public assemblies then share only self-energy
        evaluation and the final Fermi factor.
        """
        hamiltonians: Complex128[Array, "2 2 2"]
        sources: Complex128[Array, "2 5 2 2"]
        omega: Float64[Array, " 5"]
        model: SelfEnergyModel
        fermi_energy: Float64[Array, ""]
        hamiltonians, sources, omega, model, fermi_energy = (
            TestAssembleSpectralIntensityChunk._fixture()
        )
        eigenvalues: Float64[Array, "2 2"]
        eigenvectors: Complex128[Array, "2 2 2"]
        eigenvalues, eigenvectors = jax.vmap(jnp.linalg.eigh)(hamiltonians)
        weights: Float64[Array, "2 5 2"] = jnp.sum(
            jnp.abs(
                jnp.einsum(
                    "kob,keao->keab",
                    eigenvectors.conj(),
                    sources,
                )
            )
            ** 2,
            axis=2,
        )
        resolvent: Float64[Array, "2 5"] = assemble_spectral_intensity_chunk(
            hamiltonians,
            sources,
            omega,
            model,
            fermi_energy,
            19.0,
            2.0e-4,
        )
        bands: Float64[Array, "2 5"] = assemble_spectral_intensity_bands_chunk(
            eigenvalues,
            weights,
            omega,
            model,
            fermi_energy,
            19.0,
            2.0e-4,
        )
        np.testing.assert_allclose(
            np.asarray(bands),
            np.asarray(resolvent),
            rtol=1.0e-10,
            atol=1.0e-12,
        )

    def test_degenerate_rows_require_explicit_value_only_mode(self) -> None:
        """Prevent the chunk assembler from bypassing the eigen gap policy.

        Exact-degenerate rows reject unless the caller selects primal-only use.

        Notes
        -----
        Both the default jitted call and the explicit value-only success path run.
        """
        eigenvalues: Float64[Array, "1 2"] = jnp.zeros((1, 2))
        weights: Float64[Array, "1 1 2"] = jnp.asarray([[[0.7, 0.4]]])
        omega: Float64[Array, " 1"] = jnp.asarray([0.03])
        model: SelfEnergyModel = make_self_energy_model(gamma=0.02)
        assert_rejects(
            assemble_spectral_intensity_bands_chunk,
            eigenvalues,
            weights,
            omega,
            model,
            jnp.asarray(0.0),
            20.0,
            1.0e-4,
            match="gap|resolvent|value_only",
        )
        value_only: Float64[Array, "1 1"] = jax.jit(
            lambda values, candidate_weights: (
                assemble_spectral_intensity_bands_chunk(
                    values,
                    candidate_weights,
                    omega,
                    model,
                    jnp.asarray(0.0),
                    20.0,
                    1.0e-4,
                    allow_degenerate_value_only=True,
                )
            )
        )(eigenvalues, weights)
        assert bool(jnp.all(jnp.isfinite(value_only)))

    def test_frozen_chinook_spectral_comparison(self) -> None:
        """Match G7's frozen Chinook cube after its rounded-kB convention.

        The full imported k-energy cube exercises the value-only Dirac row.

        Notes
        -----
        Chinook 1.1.3 uses the rounded ratio ``1.38e-23 / 1.602e-19`` eV/K.
        Matching thermal energy therefore uses a documented effective Kelvin
        coordinate. The analytic sampled-Fermi test above separately owns
        physical correctness with DiffPES's types-owned Boltzmann constant.
        """
        manifest: Dict[str, Any] = _authenticated_json(
            _CHINOOK_SPECTRAL_MANIFEST_PATH,
            _CHINOOK_SPECTRAL_MANIFEST_SHA256,
        )
        assert manifest["schema"] == "diffpes.chinook-spectral-reference.v1"
        reference: Dict[str, Float64[NDArray, "..."]] = _authenticated_npz(
            _CHINOOK_SPECTRAL_ARCHIVE_PATH,
            _CHINOOK_SPECTRAL_ARCHIVE_SHA256,
        )
        eigenvalues_np: Float64[NDArray, "n_k n_band"] = np.asarray(
            reference["band_energies_k_band_ev"]
        )
        minimum_gap: float = float(
            np.min(np.diff(np.sort(eigenvalues_np, axis=-1), axis=-1))
        )
        assert minimum_gap < 1.0e3 * EPS_DEG
        band_weight_np: Float64[NDArray, "n_k n_band"] = np.zeros_like(
            eigenvalues_np
        )
        matrix_factor: float
        state: Float64[NDArray, " n_state_field"]
        for matrix_factor, state in zip(
            reference["m_factor_state"],
            reference["pks_state"],
            strict=True,
        ):
            row: int = int(state[1])
            column: int = int(state[2])
            flat_k: int = row * 31 + column
            band: int = int(
                np.argmin(np.abs(eigenvalues_np[flat_k] - state[3]))
            )
            assert abs(eigenvalues_np[flat_k, band] - state[3]) < 1.0e-10
            band_weight_np[flat_k, band] += matrix_factor
        omega: Float64[Array, " n_energy"] = jnp.asarray(
            reference["omega_rel_ev"]
        )
        weights_np: Float64[NDArray, "n_k n_energy n_band"] = np.broadcast_to(
            band_weight_np[:, None, :],
            (961, omega.shape[0], 2),
        ).copy()
        chinook_kb_ev_per_k: float = 1.38e-23 / 1.602e-19
        effective_temperature: float = 4.2 * chinook_kb_ev_per_k / KB_EV_PER_K
        actual: Float64[Array, "961 n_energy"] = (
            assemble_spectral_intensity_bands_chunk(
                jnp.asarray(eigenvalues_np),
                jnp.asarray(weights_np),
                omega,
                make_self_energy_model(gamma=0.02),
                jnp.asarray(0.0),
                effective_temperature,
                5.0e-5,
                allow_degenerate_value_only=True,
            )
        )
        expected: Float64[NDArray, "n_k n_energy"] = np.asarray(
            reference["intensity_raw"]
        ).reshape(961, omega.shape[0])
        np.testing.assert_allclose(
            np.asarray(actual),
            expected,
            rtol=1.0e-6,
            atol=0.0,
        )

    def test_jit_vmap_and_weight_gradient(self) -> None:
        """Exercise JIT, VMAP, and a nonzero band-weight gradient.

        The test batches complete weight fields through the public assembler.

        Notes
        -----
        VMAP batches two weight fields while the public function owns its
        native k and energy axes. The scalar loss differentiates every weight.
        """
        eigenvalues: Float64[Array, "2 2"] = jnp.asarray(
            [[-0.2, 0.3], [-0.1, 0.45]]
        )
        omega: Float64[Array, " 3"] = jnp.asarray([-0.25, -0.05, 0.2])
        weights: Float64[Array, "2 3 2"] = jnp.asarray(
            [
                [[0.8, 0.2], [0.7, 0.3], [0.6, 0.4]],
                [[0.3, 0.9], [0.4, 0.8], [0.5, 0.7]],
            ]
        )
        model: SelfEnergyModel = make_self_energy_model(gamma=0.03)

        def assemble(
            candidate: Float64[Array, "2 3 2"],
        ) -> Float64[Array, "2 3"]:
            """Assemble one member of a weight-field batch."""
            return assemble_spectral_intensity_bands_chunk(
                eigenvalues,
                candidate,
                omega,
                model,
                jnp.asarray(0.0),
                25.0,
                1.0e-3,
            )

        batched_weights: Float64[Array, "2 2 3 2"] = jnp.stack(
            [weights, 1.2 * weights]
        )
        batched: Float64[Array, "2 2 3"] = jax.jit(jax.vmap(assemble))(
            batched_weights
        )
        gradient: Float64[Array, "2 3 2"] = jax.grad(
            lambda candidate: jnp.sum(assemble(candidate))
        )(weights)
        assert batched.shape == (2, 2, 3)
        assert bool(jnp.all(jnp.isfinite(batched)))
        assert bool(jnp.all(jnp.isfinite(gradient)))
        assert bool(jnp.all(gradient > 0.0))


class TestStreamSpectralIntensity(chex.TestCase):
    """Validate the private padded WP7.6 spectral scan owner."""

    @staticmethod
    def _fixture() -> Tuple[
        Complex128[Array, "4 2 2"],
        Any,
        Float64[Array, " 8"],
        Array,
        Array,
        SelfEnergyModel,
    ]:
        """PRIVATE: Return one padded two-by-two chunk schedule.

        Notes
        -----
        Masks exclude the final k row and final two omega columns as padding.
        """
        import diffpes.simul.spectral as spectral

        base: Complex128[Array, "2 2"] = jnp.asarray(
            [[-0.2, 0.07 + 0.03j], [0.07 - 0.03j, 0.1]]
        )
        hamiltonians: Complex128[Array, "4 2 2"] = jnp.stack(
            [
                base + 0.02 * index * jnp.eye(2, dtype=jnp.complex128)
                for index in range(4)
            ]
        )
        basis: Any = make_orbital_basis(
            atom_indices=(0, 0),
            n=(1, 1),
            l=(0, 0),
            m=(0, 0),
        )
        radial: Any = make_radial_spec(
            basis,
            (0, 0),
            mode="fixed",
            fixed_integrals_shell=jnp.asarray([[0.0, 1.0]]),
        )
        matrix_element: Any = make_matrix_element_params(basis, (0, 0))
        omega: Float64[Array, " 8"] = jnp.linspace(-0.4, 0.3, 8)
        k_i: Float64[Array, "4 3"] = jnp.stack(
            (
                jnp.linspace(0.1, 0.16, 4),
                jnp.zeros(4),
                jnp.zeros(4),
            ),
            axis=-1,
        )
        final_z: Float64[Array, "4 8"] = (
            1.1
            + 0.01 * jnp.arange(4, dtype=jnp.float64)[:, None]
            + 0.02 * jnp.arange(8, dtype=jnp.float64)[None, :]
        )
        k_f: Float64[Array, "4 8 3"] = jnp.stack(
            (
                jnp.broadcast_to(k_i[:, 0, None], final_z.shape),
                jnp.zeros_like(final_z),
                final_z,
            ),
            axis=-1,
        )
        schedule: Any = spectral._TransitionSourceSchedule(
            k_i_cart=k_i,
            k_f_cart=k_f,
            emission_valid=jnp.ones((4, 8), dtype=jnp.bool_),
            positions_cart=jnp.asarray([[0.0, 0.0, 0.0], [0.23, 0.07, 0.02]]),
            depths=jnp.asarray([0.0, 0.4]),
            polarization_sample_cart=jnp.asarray(
                [1.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j]
            ),
            mean_free_path_ang=jnp.asarray(10.0),
            radial=radial,
            matrix_element=matrix_element,
            quadrature=make_radial_quadrature_spec(),
            final_state=make_final_state_spec(),
        )
        k_valid: Array = jnp.asarray([True, True, True, False])
        omega_valid: Array = jnp.asarray(
            [True, True, True, True, True, True, False, False]
        )
        return (
            hamiltonians,
            schedule,
            omega,
            k_valid,
            omega_valid,
            make_self_energy_model(gamma=0.04),
        )

    @pytest.mark.big_mem
    @pytest.mark.rss_limit_mb(1200)
    def test_checkpointed_values_and_gradients_match_uncheckpointed(
        self,
    ) -> None:
        """Match S1 rematerialized values and gradients to the direct scan.

        The comparison exercises one padded schedule with and without checkpoints.

        Notes
        -----
        The comparison is at rtol ``1e-12`` and verifies that masked padding
        contributes an exact zero gradient.
        """
        import diffpes.simul.spectral as spectral

        hamiltonians: Complex128[Array, "4 2 2"]
        schedule: Any
        omega: Float64[Array, " 8"]
        k_valid: Array
        omega_valid: Array
        model: SelfEnergyModel
        hamiltonians, schedule, omega, k_valid, omega_valid, model = (
            self._fixture()
        )

        def streamed(
            candidate: Complex128[Array, "4 2 2"],
            checkpoint: bool,
        ) -> Float64[Array, "4 8"]:
            """Run one static stream schedule."""
            return spectral._stream_spectral_intensity(
                candidate,
                omega,
                k_valid,
                omega_valid,
                schedule,
                model,
                jnp.asarray(0.03),
                20.0,
                1.0e-4,
                k_chunk=2,
                omega_chunk=4,
                checkpoint=checkpoint,
            )

        checkpointed: Float64[Array, "4 8"] = streamed(hamiltonians, True)
        direct: Float64[Array, "4 8"] = streamed(hamiltonians, False)
        checkpointed_gradient: Complex128[Array, "4 2 2"] = jax.grad(
            lambda candidate: jnp.sum(streamed(candidate, True))
        )(hamiltonians)
        direct_gradient: Complex128[Array, "4 2 2"] = jax.grad(
            lambda candidate: jnp.sum(streamed(candidate, False))
        )(hamiltonians)
        np.testing.assert_allclose(
            np.asarray(checkpointed),
            np.asarray(direct),
            rtol=1.0e-12,
            atol=1.0e-13,
        )
        np.testing.assert_allclose(
            np.asarray(checkpointed_gradient),
            np.asarray(direct_gradient),
            rtol=1.0e-12,
            atol=1.0e-12,
        )
        assert bool(jnp.all(checkpointed[-1] == 0.0))
        assert bool(jnp.all(checkpointed[:, -2:] == 0.0))
        assert bool(jnp.all(checkpointed_gradient[-1] == 0.0))

    def test_one_trace_for_one_padded_schedule(self) -> None:
        """Require one S2 trace across different masks of the same shapes.

        The test varies active extents while retaining all compiled dimensions.

        Notes
        -----
        A Python counter runs only while JAX traces the wrapper. Changing
        validity masks cannot retrace a fixed padded chunk schedule.
        """
        import diffpes.simul.spectral as spectral

        hamiltonians: Complex128[Array, "4 2 2"]
        schedule: Any
        omega: Float64[Array, " 8"]
        k_valid: Array
        omega_valid: Array
        model: SelfEnergyModel
        hamiltonians, schedule, omega, k_valid, omega_valid, model = (
            self._fixture()
        )
        trace_count: list[int] = [0]

        def scheduled(
            matrices: Complex128[Array, "4 2 2"],
            energies: Float64[Array, " 8"],
            valid_k: Array,
            valid_omega: Array,
        ) -> Float64[Array, "4 8"]:
            """Record traces of one fixed stream schedule."""
            trace_count[0] += 1
            return spectral._stream_spectral_intensity(
                matrices,
                energies,
                valid_k,
                valid_omega,
                schedule,
                model,
                jnp.asarray(0.03),
                20.0,
                1.0e-4,
                k_chunk=2,
                omega_chunk=4,
                checkpoint=True,
            )

        compiled: Callable[..., Array] = jax.jit(scheduled)
        first: Array = compiled(hamiltonians, omega, k_valid, omega_valid)
        second: Array = compiled(
            hamiltonians,
            omega,
            jnp.asarray([True, True, False, False]),
            jnp.asarray([True, True, True, True, False, False, False, False]),
        )
        jax.block_until_ready((first, second))
        assert trace_count[0] == 1
