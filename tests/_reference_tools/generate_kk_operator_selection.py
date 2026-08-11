"""Generate the independent Kramers--Kronig operator-selection artifact.

Extended Summary
----------------
Evaluate candidate principal-value operators with the frozen convergence
battery. Write the deterministic evidence archive that selects the immutable
production operator. Compare the quadratic and cubic cell operators on two
analytic fixtures. Retain the rejected opposite-parity Maclaurin control.
Record four evidence families:

- Carrier consistency uses a piecewise-linear hat carrier. Its linear
  transform is exact and sign-preserving. A kinked fixture compares this
  result with a closed-form segment arbiter. It records cubic kink error and
  positivity overshoot as counter-witnesses.
- Query derivatives follow the finite-core composite route. Transform the
  sampled Sigma'' derivative and add both analytic boundary terms. Add the
  exact forward derivative of the tail quadrature.
- Reverse-mode rows differentiate query positions, core samples, and raw tail
  coordinates. Compare them with forward mode and finite differences. Include
  exact grid-node queries to exercise reverse-mode node cancellation.
- Spectral rows measure shape, integrated weight, and peak stability for a
  frozen complex-Hermitian two-band intensity. The retarded-pole self-energy
  supplies analytic Sigma'' and operator Sigma'. Subtract Sigma' at 0 eV.

Use a base domain of [-8, 8] eV with ``n_kk = 4096`` and 256 tail nodes.
Refine the same domain to ``n_kk = 8192``. Use the phase-aligned extension
``x_j = -8 + (j - 2048) h`` with ``h = 16/4095``. Every base node remains
bitwise embedded. Also use 512 tail nodes. Keep raw tail coordinates fixed
when the model domain stays unchanged. Recompute them only after a domain
change. Record both conventions for the extension. Authenticate the 80-digit
mpmath archive with SHA-256 before loading it. Import no DiffPES production
code.

Run the generator with::

    .venv/bin/python tests/_reference_tools/generate_kk_operator_selection.py

Routine Listings
----------------
:func:`main`
    Run the frozen battery, write the deterministic archive and the
    manifest with every measured number and PASS/FAIL verdict.
"""

from __future__ import annotations

import hashlib
import io
import json
import platform
import sys
import zipfile
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import scipy
from beartype.typing import Any, Callable, Dict, List, NamedTuple, Tuple
from jaxtyping import Array, Bool, Complex128, Float64, Int64
from numpy.typing import NDArray

jax.config.update("jax_enable_x64", True)

_TOOLS_DIRECTORY: Path = Path(__file__).resolve().parent
if str(_TOOLS_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(_TOOLS_DIRECTORY))

import _kk_candidate_common  # noqa: E402
import _kk_candidate_piecewise_cubic  # noqa: E402
import _kk_candidate_piecewise_linear  # noqa: E402
import _kk_candidate_piecewise_quadratic  # noqa: E402
import _kk_control_opposite_parity_maclaurin  # noqa: E402


class GridConfig(NamedTuple):
    """Represent one frozen convergence-battery configuration."""

    domain_low_ev: float
    domain_high_ev: float
    n_kk: int
    n_tail: int
    construction: str


OMEGA_S_EV: float = 0.0
POLE_PARAMS: Tuple[float, float, float] = (0.35, 0.20, 0.12)
POLE_PARAM_NAMES: Tuple[str, str, str] = ("omega0", "gamma", "g")
WIGNER_PARAMS: Tuple[float, float] = (1.50, 0.20)
WIGNER_PARAM_NAMES: Tuple[str, str] = ("half_width", "g")
FIXTURE_PARAMS: Dict[str, Tuple[float, ...]] = {
    "pole": POLE_PARAMS,
    "wigner": WIGNER_PARAMS,
}
FIXTURE_PARAM_NAMES: Dict[str, Tuple[str, ...]] = {
    "pole": POLE_PARAM_NAMES,
    "wigner": WIGNER_PARAM_NAMES,
}
BASE_N_KK: int = 4096
EXTENSION_SHIFT_CELLS: int = 2048
CONFIGS: Dict[str, GridConfig] = {
    "base": GridConfig(-8.0, 8.0, 4096, 256, "uniform"),
    "grid8192": GridConfig(-8.0, 8.0, 8192, 256, "uniform"),
    "domain_extension": GridConfig(
        -8.0, 8.0, 8192, 256, "phase_aligned_extension"
    ),
    "tail512": GridConfig(-8.0, 8.0, 4096, 512, "uniform"),
    "grid16384": GridConfig(-8.0, 8.0, 16384, 256, "uniform"),
}
FIXTURE_CONFIG_KEYS: Dict[str, Tuple[str, ...]] = {
    "pole": ("base", "grid8192", "domain_extension", "tail512"),
    "wigner": ("base", "grid8192", "domain_extension", "grid16384"),
}
CANDIDATE_MODULES: Dict[str, Any] = {
    "pwquadratic": _kk_candidate_piecewise_quadratic,
    "pwcubic": _kk_candidate_piecewise_cubic,
}
GRID_CARRIER_MODULE: Any = _kk_candidate_piecewise_linear
CONTROL_MODULE: Any = _kk_control_opposite_parity_maclaurin
BUDGET_DELTA_SIGMA_EV: float = 2.0e-6
BUDGET_DELTA_DSIGMA: float = 2.0e-5
BUDGET_DELTA_JVP_EV: float = 2.0e-5
BUDGET_TAIL_DELTA_SIGMA_EV: float = 1.0e-13
BUDGET_TAIL_DELTA_DSIGMA: float = 1.0e-13
PAIR_TRUTH_ATOL_EV: float = 2.0e-8
PAIR_TRUTH_RTOL: float = 1.0e-6
WITNESS_MIN_VALUE_ORDER: float = 1.4
WITNESS_BASE_ERROR_EV: float = 1.0e-5
SPECTRAL_L1_BUDGET: float = 1.0e-5
SPECTRAL_WEIGHT_BUDGET: float = 1.0e-5
SPECTRAL_PEAK_BUDGET_EV: float = 2.0e-5
RAW_DELTA_BETA_FLOOR: float = -30.0
FD_STEP: float = 2.0**-12
RAW_FD_STEP: float = 2.0**-4
DIRECTIONAL_FD_STEP: float = 2.0**-14
JVP_SPOT_CHECK_INDICES: Tuple[int, int, int] = (100, 500, 900)
IDENTITY_SPOT_RTOL: float = 1.0e-9
CARRIER_BREAK_INDICES: Tuple[int, ...] = (0, 512, 1600, 2100, 2600, 3300, 4095)
CARRIER_BREAK_VALUES_EV: Tuple[float, ...] = (
    -0.05,
    -0.12,
    -0.35,
    -1.0e-6,
    -1.0e-6,
    -0.07,
    -0.04,
)
CARRIER_EXACTNESS_RTOL: float = 1.0e-12
SEAM_SLOPE_RTOL: float = 1.0e-13
REVERSE_FORWARD_RTOL: float = 1.0e-10
REVERSE_FD_RTOL: float = 1.0e-6
REVERSE_RAW_FD_RTOL: float = 1.0e-3
RAW_FD_NOISE_FLOOR: float = 5.0e-12
SPECTRAL_HAMILTONIAN_EV: Complex128[NDArray, "2 2"] = np.array(
    [[-0.25, 0.11 + 0.07j], [0.11 - 0.07j, 0.32]], dtype=np.complex128
)
SPECTRAL_SOURCE: Complex128[NDArray, " 2"] = np.array(
    [1.0 + 0.3j, -0.4 + 0.8j], dtype=np.complex128
)
SPECTRAL_ETA_EV: float = 1.0e-4
SPECTRAL_N_OMEGA: int = 4001


def _core_grid_np(config: GridConfig) -> Float64[NDArray, " n_kk"]:
    """PRIVATE: Construct the frozen uniform core grid for one configuration.

    Parameters
    ----------
    config : GridConfig
        Frozen quadrature configuration with domain endpoints in eV,
        node count, tail node count, and construction rule.

    Returns
    -------
    grid_ev : Float64[NDArray, " n_kk"]
        Core grid node positions in eV.

    Implementation Logic
    --------------------
    Use ``x_j = low + j*h`` for uniform configurations. Set
    ``h = (high-low)/(n_kk-1)``. Keep the base spacing
    ``h = (high-low)/(n_base-1)`` for the phase-aligned extension. Use
    ``x_j = low + (j - 2048)*h`` there. This preserves every base node
    bitwise and extends beyond the doubled interval.
    """
    if config.construction == "uniform":
        spacing: float = (config.domain_high_ev - config.domain_low_ev) / (
            config.n_kk - 1
        )
        grid_ev: Float64[NDArray, " n_kk"] = (
            config.domain_low_ev + np.arange(config.n_kk) * spacing
        )
        return grid_ev
    spacing = (config.domain_high_ev - config.domain_low_ev) / (BASE_N_KK - 1)
    indices: Float64[NDArray, " n_kk"] = np.arange(config.n_kk) - float(
        EXTENSION_SHIFT_CELLS
    )
    grid_ev = config.domain_low_ev + indices * spacing
    return grid_ev


def _sigma_imag(
    fixture_key: str,
    grid_ev: Float64[Array, " n_kk"],
    params: Float64[Array, " n_params"],
) -> Float64[Array, " n_kk"]:
    """PRIVATE: Evaluate the parameterized analytic Sigma'' on the KK grid.

    Parameters
    ----------
    fixture_key : str
        Fixture selector, ``"pole"`` or ``"wigner"``.
    grid_ev : Float64[Array, " n_kk"]
        Core grid node positions in eV.
    params : Float64[Array, " n_params"]
        Fixture parameters: ``(omega0, gamma, g)`` for the pole or
        ``(half_width, g)`` for the Wigner semicircle, in eV units.

    Returns
    -------
    sigma_imag : Float64[Array, " n_kk"]
        Imaginary self-energy samples in eV at the grid nodes.

    Implementation Logic
    --------------------
    Keep the Wigner square root JVP-safe with ``where``. Substitute a positive
    radicand outside the band before taking the square root. Set the outer
    branch exactly to zero.
    """
    if fixture_key == "pole":
        offset: Float64[Array, " n_kk"] = grid_ev - params[0]
        sigma_imag: Float64[Array, " n_kk"] = (
            -params[2] * params[1] / (offset * offset + params[1] ** 2)
        )
        return sigma_imag
    inside: Bool[Array, " n_kk"] = jnp.abs(grid_ev) < params[0]
    radicand: Float64[Array, " n_kk"] = jnp.where(
        inside, params[0] ** 2 - grid_ev**2, 1.0
    )
    scale: Float64[Array, ""] = 2.0 * params[1] / params[0] ** 2
    sigma_imag = jnp.where(inside, -scale * jnp.sqrt(radicand), 0.0)
    return sigma_imag


def _pole_sigma_imag_derivative(
    grid_ev: Float64[Array, " n_kk"],
    params: Float64[Array, " n_params"],
) -> Float64[Array, " n_kk"]:
    """PRIVATE: Evaluate the analytic energy derivative of the pole Sigma''.

    Parameters
    ----------
    grid_ev : Float64[Array, " n_kk"]
        Core grid node positions in eV.
    params : Float64[Array, " n_params"]
        Pole parameters ``(omega0, gamma, g)`` in eV units.

    Returns
    -------
    derivative : Float64[Array, " n_kk"]
        ``d Sigma''/d omega`` at the nodes, dimensionless (eV per eV).

    Notes
    -----
    The closed form is ``2 g gamma (omega - omega0) / ((omega -
    omega0)**2 + gamma**2)**2``, the exact derivative of the Lorentzian
    ``Sigma'' = -g gamma / ((omega - omega0)**2 + gamma**2)``.
    """
    offset: Float64[Array, " n_kk"] = grid_ev - params[0]
    denominator: Float64[Array, " n_kk"] = offset * offset + params[1] ** 2
    derivative: Float64[Array, " n_kk"] = (
        2.0 * params[2] * params[1] * offset / denominator**2
    )
    return derivative


def _edge_slopes(
    candidate_key: str,
    values: Float64[Array, " n_kk"],
    spacing_ev: Float64[Array, ""],
) -> Tuple[Float64[Array, ""], Float64[Array, ""]]:
    """PRIVATE: Return the one-sided edge slopes of the core interpolant.

    Parameters
    ----------
    candidate_key : str
        Operator selector: ``"pwlinear"``, ``"pwquadratic"``, or any
        other key for the piecewise-cubic stencil.
    values : Float64[Array, " n_kk"]
        Sigma'' samples in eV at the grid nodes.
    spacing_ev : Float64[Array, ""]
        Uniform grid spacing in eV.

    Returns
    -------
    slopes : Tuple[Float64[Array, ""], Float64[Array, ""]]
        Left and right edge slopes in eV per eV.

    Notes
    -----
    The piecewise-linear carrier supplies its first/last cell slopes
    (first order).  The piecewise-quadratic operator supplies the edge
    derivative of its edge quadratic stencil (second order, one-sided).
    The piecewise-cubic operator supplies the edge derivative of its
    clamped four-node stencil (third order, one-sided).
    """
    if candidate_key == "pwlinear":
        left: Float64[Array, ""] = (values[1] - values[0]) / spacing_ev
        right: Float64[Array, ""] = (values[-1] - values[-2]) / spacing_ev
        slopes: Tuple[Float64[Array, ""], Float64[Array, ""]] = (left, right)
        return slopes
    if candidate_key == "pwquadratic":
        left = (-3.0 * values[0] + 4.0 * values[1] - values[2]) / (
            2.0 * spacing_ev
        )
        right = (3.0 * values[-1] - 4.0 * values[-2] + values[-3]) / (
            2.0 * spacing_ev
        )
        slopes = (left, right)
        return slopes
    left = (
        -11.0 * values[0]
        + 18.0 * values[1]
        - 9.0 * values[2]
        + 2.0 * values[3]
    ) / (6.0 * spacing_ev)
    right = (
        11.0 * values[-1]
        - 18.0 * values[-2]
        + 9.0 * values[-3]
        - 2.0 * values[-4]
    ) / (6.0 * spacing_ev)
    slopes = (left, right)
    return slopes


def _pole_tail_raw_parameters(
    candidate_key: str,
    grid: Float64[NDArray, " n_kk"],
) -> Dict[str, Any]:
    """PRIVATE: Select frozen ``raw_delta_beta`` values for the pole tails.

    Parameters
    ----------
    candidate_key : str
        Operator selector that fixes the edge-slope stencil.
    grid : Float64[NDArray, " n_kk"]
        Core grid node positions in eV.

    Returns
    -------
    record : Dict[str, Any]
        Per-side amplitude, alpha, beta target, raw coordinate, and clamp
        flag. Units are eV, 1/eV, and 1/eV^2. The record also contains the
        frozen ``raw_left`` and ``raw_right`` pair.

    Implementation Logic
    --------------------
    Derive amplitudes and alphas from the core interpolant at both grid edges.
    Choose ``raw_delta_beta`` through
    ``beta = alpha**2/4 + softplus(raw)``. Match the pole curvature
    ``beta = 1/((edge-omega0)**2 + gamma**2)`` when possible. Otherwise,
    clamp raw to the -30 identifiability floor and record that clamp. Make
    this choice from the analytic fixture before inspecting candidate output.
    Store one frozen raw pair per model domain. Keep that pair fixed across
    refinements with the same domain.
    """
    omega0: float
    gamma: float
    omega0, gamma, _ = POLE_PARAMS
    spacing: float = grid[1] - grid[0]
    offset: Float64[NDArray, " n_kk"] = grid - omega0
    values: Float64[NDArray, " n_kk"] = (
        -POLE_PARAMS[2] * gamma / (offset * offset + gamma * gamma)
    )
    slope_left: float
    slope_right: float
    slope_left, slope_right = (
        float(x)
        for x in _edge_slopes(
            candidate_key, jnp.asarray(values), jnp.float64(spacing)
        )
    )
    record: Dict[str, Any] = {}
    raws: List[float] = []
    side: str
    edge: float
    slope: float
    for side, edge, slope in (
        ("left", float(grid[0]), slope_left),
        ("right", float(grid[-1]), slope_right),
    ):
        amplitude: float = float(-values[0] if side == "left" else -values[-1])
        alpha: float = (-slope if side == "left" else slope) / amplitude
        beta_target: float = 1.0 / ((edge - omega0) ** 2 + gamma**2)
        delta_beta: float = beta_target - alpha**2 / 4.0
        floor: float = float(np.log1p(np.exp(RAW_DELTA_BETA_FLOOR)))
        clamped: bool = bool(delta_beta <= floor)
        raw: float = (
            RAW_DELTA_BETA_FLOOR
            if clamped
            else float(np.log(np.expm1(delta_beta)))
        )
        raws.append(raw)
        record[side] = {
            "amplitude_ev": amplitude,
            "alpha_per_ev": alpha,
            "analytic_beta_target_per_ev2": beta_target,
            "delta_beta_per_ev2": delta_beta,
            "raw_delta_beta": raw,
            "clamped_to_floor": clamped,
        }
    record["raw_left"] = raws[0]
    record["raw_right"] = raws[1]
    return record


def _sigma_prime_unsubtracted(
    candidate_module: Any,
    candidate_key: str,
    fixture_key: str,
    config: GridConfig,
    params: Float64[Array, " n_params"],
    raws: Tuple[float, float] | None,
    queries_ev: Float64[Array, " n_query"],
) -> Float64[Array, " n_query"]:
    """PRIVATE: Evaluate one candidate's unsubtracted Sigma' at queries.

    Parameters
    ----------
    candidate_module : Any
        Candidate operator module exposing ``core_pv_transform``.
    candidate_key : str
        Operator selector that fixes the edge-slope stencil.
    fixture_key : str
        Fixture selector, ``"pole"`` or ``"wigner"``.
    config : GridConfig
        Frozen quadrature configuration for the core grid and tail.
    params : Float64[Array, " n_params"]
        Fixture parameters in eV units.
    raws : Tuple[float, float] | None
        Frozen raw tail coordinates for the pole; ``None`` for Wigner.
    queries_ev : Float64[Array, " n_query"]
        Query energies in eV.

    Returns
    -------
    sigma_prime : Float64[Array, " n_query"]
        Unsubtracted Sigma' values in eV at the queries.

    Raises
    ------
    ValueError
        If a pole scenario arrives without frozen raw tail values.

    Notes
    -----
    1. Sample Sigma'' on the frozen uniform KK grid of ``config``.
    2. Apply the candidate's cell-integrated core PV transform directly
       at the queries (no post-transform interpolation).
    3. For the pole, attach C1 ``power2`` tails from the core edges. Use the
       frozen raw carrier coordinates. Add the declared semi-infinite
       Gauss--Legendre contributions. For Wigner, use the zero-tail contract.
    """
    grid_np: Float64[NDArray, " n_kk"] = _core_grid_np(config)
    grid: Float64[Array, " n_kk"] = jnp.asarray(grid_np)
    values: Float64[Array, " n_kk"] = _sigma_imag(fixture_key, grid, params)
    core: Float64[Array, " n_query"] = candidate_module.core_pv_transform(
        grid, values, queries_ev
    )
    if fixture_key != "pole":
        return core
    if raws is None:
        raise ValueError("pole scenarios require frozen raw tail values")
    spacing: Float64[Array, ""] = grid[1] - grid[0]
    slope_left: Float64[Array, ""]
    slope_right: Float64[Array, ""]
    slope_left, slope_right = _edge_slopes(candidate_key, values, spacing)
    tail_spec: _kk_candidate_common.Power2TailSpec = (
        _kk_candidate_common.construct_power2_tail_spec(
            values[0],
            slope_left,
            values[-1],
            slope_right,
            raws[0],
            raws[1],
        )
    )
    tail: Float64[Array, " n_query"] = (
        _kk_candidate_common.semi_infinite_tail_contribution(
            jnp.asarray([grid_np[0], grid_np[-1]], dtype=jnp.float64),
            tail_spec,
            queries_ev,
            n_tail=config.n_tail,
        )
    )
    sigma_prime: Float64[Array, " n_query"] = core + tail
    return sigma_prime


def _composite_query_derivative(
    candidate_module: Any,
    candidate_key: str,
    config: GridConfig,
    params: Float64[Array, " n_params"],
    raws: Tuple[float, float],
    queries_ev: Float64[Array, " n_query"],
) -> Float64[NDArray, " n_query"]:
    """PRIVATE: Evaluate the pole query derivative via the composite route.

    Parameters
    ----------
    candidate_module : Any
        Candidate operator module exposing ``core_pv_transform``.
    candidate_key : str
        Operator selector that fixes the edge-slope stencil.
    config : GridConfig
        Frozen quadrature configuration for the core grid and tail.
    params : Float64[Array, " n_params"]
        Pole parameters ``(omega0, gamma, g)`` in eV units.
    raws : Tuple[float, float]
        Frozen raw tail coordinates for the pole tails.
    queries_ev : Float64[Array, " n_query"]
        Query energies in eV.

    Returns
    -------
    derivative : Float64[NDArray, " n_query"]
        ``d Sigma'/d omega`` at the queries, dimensionless (eV per eV).

    Implementation Logic
    --------------------
    On the finite core ``[a, b]`` the derivative identity is
    ``d/domega Sigma'_core = (1/pi) [ PV int_a^b dSigma''/dw / (w-omega)
    dw + Sigma''(a)/(a-omega) - Sigma''(b)/(b-omega) ]``; the two
    boundary terms are mandatory. Feed the sampled analytic Sigma''
    derivative to the same cell-integrated operator. Differentiate the tail
    contribution exactly with forward-mode AD. The subtraction constant has
    zero query derivative. Therefore, this also differentiates the subtracted
    output.
    """
    grid_np: Float64[NDArray, " n_kk"] = _core_grid_np(config)
    grid: Float64[Array, " n_kk"] = jnp.asarray(grid_np)
    values: Float64[Array, " n_kk"] = _sigma_imag("pole", grid, params)
    derivative_values: Float64[Array, " n_kk"] = _pole_sigma_imag_derivative(
        grid, params
    )
    core: Float64[Array, " n_query"] = candidate_module.core_pv_transform(
        grid, derivative_values, queries_ev
    )
    boundary: Float64[Array, " n_query"] = (
        values[0] / (grid[0] - queries_ev)
        - values[-1] / (grid[-1] - queries_ev)
    ) / jnp.pi
    spacing: Float64[Array, ""] = grid[1] - grid[0]
    slope_left: Float64[Array, ""]
    slope_right: Float64[Array, ""]
    slope_left, slope_right = _edge_slopes(candidate_key, values, spacing)
    tail_spec: _kk_candidate_common.Power2TailSpec = (
        _kk_candidate_common.construct_power2_tail_spec(
            values[0],
            slope_left,
            values[-1],
            slope_right,
            raws[0],
            raws[1],
        )
    )

    def _tail_only(
        query_values: Float64[Array, " n_any"],
    ) -> Float64[Array, " n_any"]:
        """PRIVATE: Evaluate the tail contribution at arbitrary queries.

        Parameters
        ----------
        query_values : Float64[Array, " n_any"]
            Query energies in eV.

        Returns
        -------
        tail_values : Float64[Array, " n_any"]
            Semi-infinite tail contributions in eV.

        Notes
        -----
        The closure uses the fixed domain, tail parameters, and quadrature
        order. JAX differentiates only the query values.
        """
        tail_values: Float64[Array, " n_any"] = (
            _kk_candidate_common.semi_infinite_tail_contribution(
                jnp.asarray([grid_np[0], grid_np[-1]], dtype=jnp.float64),
                tail_spec,
                query_values,
                n_tail=config.n_tail,
            )
        )
        return tail_values

    tail_derivative: Float64[Array, " n_query"]
    _, tail_derivative = jax.jvp(
        _tail_only, (queries_ev,), (jnp.ones_like(queries_ev),)
    )
    derivative: Float64[NDArray, " n_query"] = np.asarray(
        core + boundary + tail_derivative
    )
    return derivative


def _evaluate_scenario(
    candidate_module: Any,
    candidate_key: str,
    fixture_key: str,
    config: GridConfig,
    raws: Tuple[float, float] | None,
    queries: Float64[NDArray, " n_query"],
    values_only: bool = False,
) -> Dict[str, Float64[NDArray, " n_query"]]:
    """PRIVATE: Compute subtracted values, query derivatives, and JVPs.

    Parameters
    ----------
    candidate_module : Any
        Candidate operator module exposing ``core_pv_transform``.
    candidate_key : str
        Operator selector that fixes the edge-slope stencil.
    fixture_key : str
        Fixture selector, ``"pole"`` or ``"wigner"``.
    config : GridConfig
        Frozen quadrature configuration for the core grid and tail.
    raws : Tuple[float, float] | None
        Frozen raw tail coordinates for the pole; ``None`` for Wigner.
    queries : Float64[NDArray, " n_query"]
        Query energies in eV.
    values_only : bool
        When true, return only the subtracted values row.

    Returns
    -------
    result : Dict[str, Float64[NDArray, " n_query"]]
        Row dictionary with ``sigma_sub_ev`` and, unless
        ``values_only``, the direct and composite query derivatives,
        the identity spot deviation, and one JVP row per parameter.

    Raises
    ------
    RuntimeError
        If the forward JVP and ``jax.grad`` disagree beyond the
        scale-aware tolerance at a spot-check query.

    Notes
    -----
    Use one forward-mode JVP with an all-ones query tangent. Every output row
    depends on exactly one query. Therefore, the tangent equals the per-query
    derivative. Compare this identity with ``jax.grad`` at three spot checks.
    Use a scale-aware tolerance that grows with summation length. For pole
    scenarios, also evaluate the composite derivative with finite-core
    boundary terms. Parameter JVPs vary each dimensionless coordinate
    ``q_p = p/p_fixture`` at ``q = 1``. Keep raw tail coordinates fixed
    because they represent separate carrier coordinates.
    """
    queries_jnp: Float64[Array, " n_query"] = jnp.asarray(
        queries, dtype=jnp.float64
    )
    base_params: Float64[Array, " n_param"] = jnp.asarray(
        FIXTURE_PARAMS[fixture_key], dtype=jnp.float64
    )
    n_params: int = base_params.shape[0]

    def _unsubtracted(
        query_values: Float64[Array, " n_any"],
    ) -> Float64[Array, " n_any"]:
        """PRIVATE: Evaluate the unsubtracted operator at arbitrary queries.

        Parameters
        ----------
        query_values : Float64[Array, " n_any"]
            Query energies in eV.

        Returns
        -------
        sigma_prime : Float64[Array, " n_any"]
            Unsubtracted real self-energy values in eV.

        Notes
        -----
        The closure fixes the candidate, fixture, grid, parameters, and tail
        coordinates. JAX differentiates the query values.
        """
        sigma_prime: Float64[Array, " n_any"] = _sigma_prime_unsubtracted(
            candidate_module,
            candidate_key,
            fixture_key,
            config,
            base_params,
            raws,
            query_values,
        )
        return sigma_prime

    def _subtracted_of_scale(
        scales: Float64[Array, " n_params"],
    ) -> Float64[Array, " n_query"]:
        """PRIVATE: Evaluate the subtracted output for parameter scales.

        Parameters
        ----------
        scales : Float64[Array, " n_params"]
            Dimensionless multipliers for the physical fixture parameters.

        Returns
        -------
        sigma_subtracted : Float64[Array, " n_query"]
            Real self-energy relative to the subtraction point in eV.

        Notes
        -----
        The closure appends the subtraction point to the query vector. It
        subtracts that final value from every physical query value.
        """
        total: Float64[Array, "..."] = _sigma_prime_unsubtracted(
            candidate_module,
            candidate_key,
            fixture_key,
            config,
            scales * base_params,
            raws,
            jnp.concatenate(
                (queries_jnp, jnp.asarray([OMEGA_S_EV], dtype=jnp.float64))
            ),
        )
        sigma_subtracted: Float64[Array, " n_query"] = total[:-1] - total[-1]
        return sigma_subtracted

    sigma_sub: Float64[NDArray, " n_query"] = np.asarray(
        _subtracted_of_scale(jnp.ones(n_params))
    )
    result: Dict[str, Float64[NDArray, " n_query"]] = {
        "sigma_sub_ev": sigma_sub
    }
    if values_only:
        return result

    dsigma: Float64[Array, " n_query"]
    _, dsigma = jax.jvp(
        _unsubtracted, (queries_jnp,), (jnp.ones_like(queries_jnp),)
    )
    dsigma_np: Float64[NDArray, " n_query"] = np.asarray(dsigma)
    identity_deviation: float = 0.0
    index: int
    for index in JVP_SPOT_CHECK_INDICES:
        spot: Float64[Array, ""] = jax.grad(
            lambda q: _unsubtracted(q[None])[0]
        )(queries_jnp[index])
        deviation: float = abs(float(spot) - dsigma_np[index])
        identity_deviation = max(identity_deviation, deviation)
        tolerance: float = (
            IDENTITY_SPOT_RTOL
            * (config.n_kk / float(BASE_N_KK))
            * max(1.0, abs(float(spot)), abs(dsigma_np[index]))
        )
        if deviation > tolerance:
            raise RuntimeError(
                "query-derivative JVP identity violated at index "
                f"{index}: {float(spot)} != {dsigma_np[index]}"
            )
    result["identity_spot_max_abs_dev"] = np.float64(identity_deviation)
    result["dsigma_direct"] = dsigma_np
    if fixture_key == "pole" and raws is not None:
        result["dsigma_composite"] = _composite_query_derivative(
            candidate_module,
            candidate_key,
            config,
            base_params,
            raws,
            queries_jnp,
        )
    param_index: int
    param_name: str
    for param_index, param_name in enumerate(FIXTURE_PARAM_NAMES[fixture_key]):
        direction: Float64[Array, " n_param"] = (
            jnp.zeros(n_params).at[param_index].set(1.0)
        )
        tangent: Float64[Array, " n_query"]
        _, tangent = jax.jvp(
            _subtracted_of_scale, (jnp.ones(n_params),), (direction,)
        )
        result[f"jvp_{param_name}"] = np.asarray(tangent)
    return result


def _five_point_fd(
    candidate_module: Any,
    candidate_key: str,
    fixture_key: str,
    config: GridConfig,
    raws: Tuple[float, float] | None,
    queries: Float64[NDArray, " n_query"],
    param_index: int,
) -> Float64[NDArray, " n_query"]:
    """PRIVATE: Compute a five-point finite difference of the output.

    Parameters
    ----------
    candidate_module : Any
        Candidate operator module exposing ``core_pv_transform``.
    candidate_key : str
        Operator selector that fixes the edge-slope stencil.
    fixture_key : str
        Fixture selector, ``"pole"`` or ``"wigner"``.
    config : GridConfig
        Frozen quadrature configuration for the core grid and tail.
    raws : Tuple[float, float] | None
        Frozen raw tail coordinates for the pole; ``None`` for Wigner.
    queries : Float64[NDArray, " n_query"]
        Query energies in eV.
    param_index : int
        Index of the perturbed fixture parameter.

    Returns
    -------
    difference : Float64[NDArray, " n_query"]
        Finite-difference derivative of the subtracted output with
        respect to the dimensionless parameter coordinate, in eV.

    Notes
    -----
    The stencil acts on the dimensionless coordinate ``q_p`` around 1
    with step :data:`FD_STEP` and truncation order ``FD_STEP**4``.
    """
    queries_jnp: Float64[Array, " n_query"] = jnp.asarray(
        queries, dtype=jnp.float64
    )
    base_params: Float64[NDArray, " n_param"] = np.asarray(
        FIXTURE_PARAMS[fixture_key], dtype=np.float64
    )

    def _subtracted(step_multiple: float) -> Float64[NDArray, " n_query"]:
        """PRIVATE: Evaluate one finite-difference stencil displacement.

        Parameters
        ----------
        step_multiple : float
            Signed multiple of the dimensionless finite-difference step.

        Returns
        -------
        sigma_subtracted : Float64[NDArray, " n_query"]
            Subtracted real self-energy values in eV.

        Notes
        -----
        The closure perturbs one physical parameter through its dimensionless
        scale. It keeps the tail coordinates fixed.
        """
        scales: Float64[NDArray, " n_param"] = np.ones_like(base_params)
        scales[param_index] += step_multiple * FD_STEP
        total: Float64[Array, "..."] = _sigma_prime_unsubtracted(
            candidate_module,
            candidate_key,
            fixture_key,
            config,
            jnp.asarray(scales * base_params),
            raws,
            jnp.concatenate(
                (queries_jnp, jnp.asarray([OMEGA_S_EV], dtype=jnp.float64))
            ),
        )
        sigma_subtracted: Float64[NDArray, " n_query"] = np.asarray(
            total[:-1] - total[-1]
        )
        return sigma_subtracted

    difference: Float64[NDArray, " n_query"] = (
        _subtracted(-2.0)
        - 8.0 * _subtracted(-1.0)
        + 8.0 * _subtracted(1.0)
        - _subtracted(2.0)
    ) / (12.0 * FD_STEP)
    return difference


def _five_point_scalar_fd(
    function: Callable[[float], float], step: float
) -> float:
    """PRIVATE: Compute a five-point central difference of one scalar map.

    Parameters
    ----------
    function : Callable[[float], float]
        Scalar function of the signed perturbation to differentiate at
        zero.
    step : float
        Positive stencil step in the units of the function argument.

    Returns
    -------
    derivative : float
        Central-difference derivative with truncation order
        ``step**4``.

    Notes
    -----
    The stencil is ``(f(-2s) - 8 f(-s) + 8 f(s) - f(2s)) / (12 s)``.
    """
    derivative: float = (
        function(-2.0 * step)
        - 8.0 * function(-step)
        + 8.0 * function(step)
        - function(2.0 * step)
    ) / (12.0 * step)
    return derivative


def _analytic_truth(
    queries: Float64[NDArray, " n_query"],
) -> Dict[str, Float64[NDArray, " n_query"]]:
    """PRIVATE: Return closed-form query-derivative truths for both fixtures.

    Parameters
    ----------
    queries : Float64[NDArray, " n_query"]
        Query energies in eV.

    Returns
    -------
    truths : Dict[str, Float64[NDArray, " n_query"]]
        Arrays ``truth_pole_dsigma_domega`` and
        ``truth_wigner_dsigma_domega`` in eV per eV.

    Notes
    -----
    The pole truth is ``g (gamma^2 - (omega-omega0)^2) / ((omega-
    omega0)^2 + gamma^2)^2``. The in-band Wigner truth is the constant
    ``2 g / half_width^2``.
    """
    omega0: float
    gamma: float
    coupling: float
    omega0, gamma, coupling = POLE_PARAMS
    offset: Float64[NDArray, " n_query"] = queries - omega0
    denominator: Float64[NDArray, " n_query"] = offset * offset + gamma * gamma
    pole_dsigma: Float64[NDArray, " n_query"] = (
        coupling
        * (gamma * gamma - offset * offset)
        / (denominator * denominator)
    )
    half_width: float
    wigner_coupling: float
    half_width, wigner_coupling = WIGNER_PARAMS
    wigner_dsigma: Float64[NDArray, " n_query"] = np.full_like(
        queries, 2.0 * wigner_coupling / half_width**2
    )
    truth: Dict[str, Float64[NDArray, " n_query"]] = {
        "truth_pole_dsigma_domega": pole_dsigma,
        "truth_wigner_dsigma_domega": wigner_dsigma,
    }
    return truth


def _mixed_criterion_statistics(
    observed: Float64[NDArray, " n_query"],
    truth: Float64[NDArray, " n_query"],
    queries: Float64[NDArray, " n_query"],
) -> Dict[str, Any]:
    """PRIVATE: Evaluate the analytic-pair-truth mixed criterion per row.

    Parameters
    ----------
    observed : Float64[NDArray, " n_query"]
        Candidate output in eV at the queries.
    truth : Float64[NDArray, " n_query"]
        Analytic truth in eV at the same queries.
    queries : Float64[NDArray, " n_query"]
        Query energies in eV.

    Returns
    -------
    statistics : Dict[str, Any]
        Maximum absolute error in eV, worst mixed-criterion ratio, the
        worst query in eV, the violating row count, and the verdict.

    Notes
    -----
    Each row's allowance is ``PAIR_TRUTH_ATOL_EV + PAIR_TRUTH_RTOL *
    |truth|``; the criterion passes when every ``|error|/allowance``
    ratio stays at or below one.
    """
    error: Float64[NDArray, " n_query"] = np.abs(observed - truth)
    allowance: Float64[NDArray, " n_query"] = (
        PAIR_TRUTH_ATOL_EV + PAIR_TRUTH_RTOL * np.abs(truth)
    )
    ratio: Float64[NDArray, " n_query"] = error / allowance
    worst: int = int(np.argmax(ratio))
    statistics: Dict[str, Any] = {
        "max_abs_error_ev": float(np.max(error)),
        "mixed_criterion_max_ratio": float(ratio[worst]),
        "worst_query_ev": float(queries[worst]),
        "violating_rows": int(np.sum(ratio > 1.0)),
        "pass": bool(np.all(ratio <= 1.0)),
    }
    return statistics


def _refinement_deltas(
    base: Dict[str, Float64[NDArray, " n_query"]],
    refined: Dict[str, Float64[NDArray, " n_query"]],
    param_names: Tuple[str, ...],
) -> Dict[str, Any]:
    """PRIVATE: Return the raw maximum-absolute refinement deltas.

    Parameters
    ----------
    base : Dict[str, Float64[NDArray, " n_query"]]
        Scenario rows evaluated on the base configuration.
    refined : Dict[str, Float64[NDArray, " n_query"]]
        Scenario rows evaluated on the refined configuration.
    param_names : Tuple[str, ...]
        Fixture parameter names whose JVP rows the delta covers.

    Returns
    -------
    deltas : Dict[str, Any]
        Maximum absolute value delta in eV, derivative deltas when both
        rows exist, and one per-parameter JVP delta mapping.

    Notes
    -----
    Every delta is ``max |refined - base|`` over the shared query grid;
    missing rows on either side drop out silently.
    """
    deltas: Dict[str, Any] = {
        "max_delta_sigma_ev": float(
            np.max(np.abs(refined["sigma_sub_ev"] - base["sigma_sub_ev"]))
        )
    }
    if "dsigma_direct" in refined and "dsigma_direct" in base:
        deltas["max_delta_dsigma_direct"] = float(
            np.max(np.abs(refined["dsigma_direct"] - base["dsigma_direct"]))
        )
    if "dsigma_composite" in refined and "dsigma_composite" in base:
        deltas["max_delta_dsigma_composite"] = float(
            np.max(
                np.abs(refined["dsigma_composite"] - base["dsigma_composite"])
            )
        )
    per_param: Dict[str, Any] = {
        name: float(
            np.max(np.abs(refined[f"jvp_{name}"] - base[f"jvp_{name}"]))
        )
        for name in param_names
        if f"jvp_{name}" in refined and f"jvp_{name}" in base
    }
    if per_param:
        deltas["max_delta_jvp_per_param_ev"] = per_param
    return deltas


def _pole_refinement_metrics(
    deltas: Dict[str, Any], is_tail_refinement: bool
) -> Dict[str, Any]:
    """PRIVATE: Attach PASS/FAIL verdicts to one pole refinement row.

    Parameters
    ----------
    deltas : Dict[str, Any]
        Raw refinement deltas from :func:`_refinement_deltas`.
    is_tail_refinement : bool
        Whether the row compares the 512-node tail against the base.

    Returns
    -------
    metrics : Dict[str, Any]
        The deltas plus boolean verdicts against the registered value,
        composite-derivative, and JVP budgets; tail refinements gain
        the two 1e-13 tail-only verdicts.

    Notes
    -----
    The registered budgets are module constants; the function copies
    the delta dictionary and never mutates its input.
    """
    metrics: Dict[str, Any] = dict(deltas)
    metrics["pass_delta_sigma_2em6"] = bool(
        deltas["max_delta_sigma_ev"] <= BUDGET_DELTA_SIGMA_EV
    )
    metrics["pass_delta_dsigma_composite_2em5"] = bool(
        deltas["max_delta_dsigma_composite"] <= BUDGET_DELTA_DSIGMA
    )
    max_jvp: float = max(deltas["max_delta_jvp_per_param_ev"].values())
    metrics["max_delta_jvp_ev"] = max_jvp
    metrics["pass_delta_jvp_2em5"] = bool(max_jvp <= BUDGET_DELTA_JVP_EV)
    if is_tail_refinement:
        metrics["pass_tail_delta_sigma_1em13"] = bool(
            deltas["max_delta_sigma_ev"] <= BUDGET_TAIL_DELTA_SIGMA_EV
        )
        metrics["pass_tail_delta_dsigma_1em13"] = bool(
            deltas["max_delta_dsigma_composite"] <= BUDGET_TAIL_DELTA_DSIGMA
        )
    return metrics


def _wigner_refinement_metrics(deltas: Dict[str, Any]) -> Dict[str, Any]:
    """PRIVATE: Attach the stress-witness verdicts to one Wigner row.

    Parameters
    ----------
    deltas : Dict[str, Any]
        Raw refinement deltas from :func:`_refinement_deltas`.

    Returns
    -------
    metrics : Dict[str, Any]
        The deltas plus the coupling-JVP verdict and the recorded
        half-width requirement exclusion.

    Notes
    -----
    Measure and record the half-width JVP without using it for acceptance.
    Its band edge is a square-root branch point. The sampled parameter tangent
    lacks uniform integrability on a static grid. Compare the coupling JVP
    with the value budget.
    """
    metrics: Dict[str, Any] = dict(deltas)
    coupling_delta: float = deltas["max_delta_jvp_per_param_ev"]["g"]
    metrics["pass_delta_jvp_g_value_budget_2em6"] = bool(
        coupling_delta <= BUDGET_DELTA_SIGMA_EV
    )
    metrics["jvp_half_width_requirement"] = "excluded (band-edge branch point)"
    return metrics


def _verify_wigner_zero_edges() -> None:
    """PRIVATE: Assert the compact-support zero-tail contract per domain.

    Notes
    -----
    Sample Wigner Sigma'' on every registered configuration grid. Pass both
    edge values to the zero-tail constructor. The constructor rejects any
    nonzero amplitude.
    """
    params: Float64[Array, " n_wigner_param"] = jnp.asarray(
        WIGNER_PARAMS, dtype=jnp.float64
    )
    config_key: str
    for config_key in FIXTURE_CONFIG_KEYS["wigner"]:
        config: GridConfig = CONFIGS[config_key]
        grid: Float64[Array, " n_kk"] = jnp.asarray(_core_grid_np(config))
        values: Float64[Array, " n_kk"] = _sigma_imag("wigner", grid, params)
        _kk_candidate_common.construct_wigner_zero_tail(values[0], values[-1])


def _carrier_consistency(
    queries: Float64[NDArray, " n_query"],
) -> Tuple[Dict[str, Any], Dict[str, Float64[NDArray, "..."]]]:
    """PRIVATE: Measure the grid-mode carrier contract on a hat fixture.

    Parameters
    ----------
    queries : Float64[NDArray, " n_query"]
        Query energies in eV.

    Returns
    -------
    evidence : Tuple[Dict[str, Any], Dict[str, Float64[NDArray, "..."]]]
        Metrics dictionary (exactness, sign preservation, seam
        verdicts) and the arrays for the archive (node values, arbiter,
        linear and cubic transforms, all in eV).

    Implementation Logic
    --------------------
    Use a piecewise-linear hat as the binding grid carrier. Plant interior
    kinks at exact grid nodes. The fine-sample hat then equals the coarse
    kinked function exactly. Sum exact principal-value integrals over six
    coarse segments with the closed-form arbiter. Require the cell-integrated
    linear transform to match at roundoff. Record cubic error and positivity
    overshoot as counter-witnesses. A near-zero plateau follows a steep rise.
    This makes the cubic interpolant cross zero between negative samples. The
    hat reconstruction always stays within the sample range.
    """
    config: GridConfig = CONFIGS["base"]
    grid: Float64[NDArray, " n_kk"] = _core_grid_np(config)
    spacing: float = grid[1] - grid[0]
    node_values: Float64[NDArray, " n_kk"] = np.interp(
        np.arange(config.n_kk),
        np.asarray(CARRIER_BREAK_INDICES, dtype=np.float64),
        np.asarray(CARRIER_BREAK_VALUES_EV, dtype=np.float64),
    )

    arbiter: Float64[NDArray, " n_query"] = np.zeros_like(queries)
    segment: int
    for segment in range(len(CARRIER_BREAK_INDICES) - 1):
        x_start: float = grid[CARRIER_BREAK_INDICES[segment]]
        x_stop: float = grid[CARRIER_BREAK_INDICES[segment + 1]]
        f_start: float = CARRIER_BREAK_VALUES_EV[segment]
        f_stop: float = CARRIER_BREAK_VALUES_EV[segment + 1]
        slope: float = (f_stop - f_start) / (x_stop - x_start)
        extension: Float64[NDArray, " n_query"] = f_start + slope * (
            queries - x_start
        )
        arbiter += slope * (x_stop - x_start) + extension * np.log(
            np.abs((x_stop - queries) / (x_start - queries))
        )
    arbiter /= np.pi

    grid_jnp: Float64[Array, " n_kk"] = jnp.asarray(grid)
    values_jnp: Float64[Array, " n_kk"] = jnp.asarray(node_values)
    queries_jnp: Float64[Array, " n_query"] = jnp.asarray(queries)
    linear_core: Float64[NDArray, " n_query"] = np.asarray(
        GRID_CARRIER_MODULE.core_pv_transform(
            grid_jnp, values_jnp, queries_jnp
        )
    )
    cubic_core: Float64[NDArray, " n_query"] = np.asarray(
        _kk_candidate_piecewise_cubic.core_pv_transform(
            grid_jnp, values_jnp, queries_jnp
        )
    )
    scale: float = max(1.0, float(np.max(np.abs(arbiter))))
    linear_error: float = float(np.max(np.abs(linear_core - arbiter)))
    cubic_error: float = float(np.max(np.abs(cubic_core - arbiter)))

    subdivisions: Float64[NDArray, " n_subdivision"] = np.linspace(
        0.0, 1.0, 9
    )[:-1]
    cells: Int64[NDArray, " n_cell"] = np.arange(config.n_kk - 1)
    starts: Int64[NDArray, " n_cell"] = np.clip(cells - 1, 0, config.n_kk - 4)
    y0: Float64[NDArray, " n_cell"] = node_values[starts]
    y1: Float64[NDArray, " n_cell"] = node_values[starts + 1]
    y2: Float64[NDArray, " n_cell"] = node_values[starts + 2]
    y3: Float64[NDArray, " n_cell"] = node_values[starts + 3]
    linear_c: Float64[NDArray, " n_cell"] = (
        -11.0 * y0 + 18.0 * y1 - 9.0 * y2 + 2.0 * y3
    ) / (6.0 * spacing)
    quadratic_c: Float64[NDArray, " n_cell"] = (
        2.0 * y0 - 5.0 * y1 + 4.0 * y2 - y3
    ) / (2.0 * spacing**2)
    cubic_c: Float64[NDArray, " n_cell"] = (-y0 + 3.0 * y1 - 3.0 * y2 + y3) / (
        6.0 * spacing**3
    )
    dense_offsets: Float64[NDArray, "n_cell n_subdivision"] = (
        grid[cells][:, None]
        + subdivisions[None, :] * spacing
        - grid[starts][:, None]
    )
    dense_cubic: Float64[NDArray, "n_cell n_subdivision"] = (
        y0[:, None]
        + linear_c[:, None] * dense_offsets
        + quadratic_c[:, None] * dense_offsets**2
        + cubic_c[:, None] * dense_offsets**3
    )
    cubic_reconstruction_max: float = float(np.max(dense_cubic))
    hat_reconstruction_max: float = float(np.max(node_values))

    hat_slope_left: float = (node_values[1] - node_values[0]) / spacing
    hat_slope_right: float = (node_values[-1] - node_values[-2]) / spacing
    tail_spec: _kk_candidate_common.Power2TailSpec = (
        _kk_candidate_common.construct_power2_tail_spec(
            node_values[0],
            hat_slope_left,
            node_values[-1],
            hat_slope_right,
            0.0,
            0.0,
        )
    )
    seam_left: float = abs(
        float(-tail_spec.amplitude_left * tail_spec.alpha_left)
        - hat_slope_left
    )
    seam_right: float = abs(
        float(tail_spec.amplitude_right * tail_spec.alpha_right)
        - hat_slope_right
    )
    slope_scale: float = max(1.0, abs(hat_slope_left), abs(hat_slope_right))

    metrics: Dict[str, Any] = {
        "requirement": "kk-carrier-consistency",
        "fixture": {
            "break_node_indices": list(CARRIER_BREAK_INDICES),
            "break_values_ev": list(CARRIER_BREAK_VALUES_EV),
            "interior_kink_count": len(CARRIER_BREAK_INDICES) - 2,
            "all_samples_negative": bool(np.all(node_values < 0.0)),
        },
        "linear_transform_max_abs_error_ev": linear_error,
        "linear_transform_error_scale_ev": scale,
        "pass_linear_exactness_roundoff": bool(
            linear_error <= CARRIER_EXACTNESS_RTOL * scale
        ),
        "cubic_transform_max_abs_error_ev": cubic_error,
        "hat_reconstruction_max_ev": hat_reconstruction_max,
        "hat_sign_preserved": bool(hat_reconstruction_max < 0.0),
        "cubic_reconstruction_max_ev": cubic_reconstruction_max,
        "cubic_positivity_overshoot": bool(cubic_reconstruction_max > 0.0),
        "hat_tail_seam_slope_error_left": seam_left,
        "hat_tail_seam_slope_error_right": seam_right,
        "pass_hat_tail_seam_roundoff": bool(
            max(seam_left, seam_right) <= SEAM_SLOPE_RTOL * slope_scale
        ),
    }
    arrays: Dict[str, Float64[NDArray, "..."]] = {
        "carrier_node_values_ev": node_values,
        "carrier_arbiter_sigma_ev": arbiter,
        "carrier_linear_sigma_ev": linear_core,
        "carrier_cubic_sigma_ev": cubic_core,
    }
    evidence: Tuple[Dict[str, Any], Dict[str, Float64[NDArray, "..."]]] = (
        metrics,
        arrays,
    )
    return evidence


def _smooth_seam_consistency(
    raw_records: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    """PRIVATE: Measure tail edge-slope consistency for the smooth route.

    Parameters
    ----------
    raw_records : Dict[str, Dict[str, Any]]
        Frozen per-candidate raw tail records keyed by domain.

    Returns
    -------
    result : Dict[str, Any]
        Per-candidate edge-slope errors against the analytic pole
        slope, the C1 seam errors, and the roundoff seam verdict.

    Notes
    -----
    The row records the exact C1 seam identity between the ``power2``
    tail derivative and each operator's one-sided edge stencil slope.
    It also records the accuracy of every edge-slope estimator against
    the analytic pole slope at the base edges.
    """
    omega0: float
    gamma: float
    coupling: float
    omega0, gamma, coupling = POLE_PARAMS
    config: GridConfig = CONFIGS["base"]
    grid: Float64[NDArray, " n_kk"] = _core_grid_np(config)
    spacing: float = grid[1] - grid[0]
    offset: Float64[NDArray, " n_kk"] = grid - omega0
    values: Float64[NDArray, " n_kk"] = (
        -coupling * gamma / (offset * offset + gamma * gamma)
    )

    def _analytic_slope(point: float) -> float:
        """PRIVATE: Evaluate the analytic pole slope at one energy.

        Parameters
        ----------
        point : float
            Evaluation energy in eV.

        Returns
        -------
        slope_value : float
            Energy derivative of the imaginary self-energy.

        Notes
        -----
        The closure differentiates the retarded-pole Lorentzian with its fixed
        center, width, and coupling.
        """
        shifted: float = point - omega0
        slope_value: float = (
            2.0
            * coupling
            * gamma
            * shifted
            / (shifted * shifted + gamma * gamma) ** 2
        )
        return slope_value

    result: Dict[str, Any] = {"requirement": "kk-carrier-consistency"}
    candidate_key: str
    for candidate_key in ("pwlinear", "pwquadratic", "pwcubic"):
        slope_left: float
        slope_right: float
        slope_left, slope_right = (
            float(x)
            for x in _edge_slopes(
                candidate_key, jnp.asarray(values), jnp.float64(spacing)
            )
        )
        record: Dict[str, Any] | None = raw_records.get(candidate_key, {}).get(
            "base_domain"
        )
        seam: Dict[str, Any] = {
            "edge_slope_error_left": abs(
                slope_left - _analytic_slope(float(grid[0]))
            ),
            "edge_slope_error_right": abs(
                slope_right - _analytic_slope(float(grid[-1]))
            ),
        }
        if record is not None:
            tail_spec: _kk_candidate_common.Power2TailSpec = (
                _kk_candidate_common.construct_power2_tail_spec(
                    values[0],
                    slope_left,
                    values[-1],
                    slope_right,
                    record["raw_left"],
                    record["raw_right"],
                )
            )
            seam_left: float = abs(
                float(-tail_spec.amplitude_left * tail_spec.alpha_left)
                - slope_left
            )
            seam_right: float = abs(
                float(tail_spec.amplitude_right * tail_spec.alpha_right)
                - slope_right
            )
            slope_scale: float = max(1.0, abs(slope_left), abs(slope_right))
            seam["tail_seam_slope_error_left"] = seam_left
            seam["tail_seam_slope_error_right"] = seam_right
            seam["pass_tail_seam_roundoff"] = bool(
                max(seam_left, seam_right) <= SEAM_SLOPE_RTOL * slope_scale
            )
        result[candidate_key] = seam
    return result


def _reverse_mode_evidence(  # noqa: PLR0915 -- coupled AD evidence record.
    candidate_module: Any,
    candidate_key: str,
    raws: Tuple[float, float],
    queries: Float64[NDArray, " n_query"],
) -> Tuple[Dict[str, Any], Dict[str, Float64[NDArray, "..."]]]:
    """PRIVATE: Measure reverse-mode gradients of a scalar contraction.

    Parameters
    ----------
    candidate_module : Any
        Candidate operator module exposing ``core_pv_transform``.
    candidate_key : str
        Operator selector that fixes the edge-slope stencil.
    raws : Tuple[float, float]
        Frozen raw tail coordinates for the pole tails.
    queries : Float64[NDArray, " n_query"]
        Query energies in eV.

    Returns
    -------
    evidence : Tuple[Dict[str, Any], Dict[str, Float64[NDArray, "..."]]]
        Metrics dictionary (identity, finite-difference, and node-
        coincident verdicts) and the three gradient arrays for the
        archive.

    Implementation Logic
    --------------------
    A fixed positive weight vector contracts the subtracted pole output
    to a scalar.  ``jax.grad`` computes gradients with respect to query
    positions, core Sigma'' samples, and the two raw tail coordinates.
    Compare each gradient with forward-mode JVPs through the dot-product
    identity. Also compare it with five-point central finite differences.
    Place additional queries exactly on grid nodes. These queries exercise
    reverse-mode node cancellation. Their gradients must remain finite.
    """
    config: GridConfig = CONFIGS["base"]
    grid_np: Float64[NDArray, " n_kk"] = _core_grid_np(config)
    grid: Float64[Array, " n_kk"] = jnp.asarray(grid_np)
    spacing: Float64[Array, ""] = grid[1] - grid[0]
    params: Float64[Array, " n_param"] = jnp.asarray(
        POLE_PARAMS, dtype=jnp.float64
    )
    values: Float64[Array, " n_kk"] = _sigma_imag("pole", grid, params)
    queries_jnp: Float64[Array, " n_query"] = jnp.asarray(
        queries, dtype=jnp.float64
    )
    weights: Float64[Array, " n_query"] = 0.75 + 0.5 * jnp.cos(
        3.0 * queries_jnp
    )
    raws_vec: Float64[Array, " 2"] = jnp.asarray(raws, dtype=jnp.float64)

    def _subtracted_from(
        values_in: Float64[Array, " n_kk"],
        raws_in: Float64[Array, " 2"],
        queries_in: Float64[Array, " n_any"],
    ) -> Float64[Array, " n_any"]:
        """PRIVATE: Evaluate the subtracted operator from explicit carriers.

        Parameters
        ----------
        values_in : Float64[Array, " n_kk"]
            Imaginary self-energy samples in eV.
        raws_in : Float64[Array, " 2"]
            Raw coordinates for the left and right tails.
        queries_in : Float64[Array, " n_any"]
            Query energies in eV.

        Returns
        -------
        sigma_subtracted : Float64[Array, " n_any"]
            Real self-energy relative to the subtraction point in eV.

        Notes
        -----
        The closure combines the core transform and both tails. It appends and
        removes the fixed subtraction-point value.
        """
        appended: Float64[Array, "..."] = jnp.concatenate(
            (queries_in, jnp.asarray([OMEGA_S_EV], dtype=jnp.float64))
        )
        core: Float64[Array, "..."] = candidate_module.core_pv_transform(
            grid, values_in, appended
        )
        slope_left: Float64[Array, ""]
        slope_right: Float64[Array, ""]
        slope_left, slope_right = _edge_slopes(
            candidate_key, values_in, spacing
        )
        tail_spec: _kk_candidate_common.Power2TailSpec = (
            _kk_candidate_common.construct_power2_tail_spec(
                values_in[0],
                slope_left,
                values_in[-1],
                slope_right,
                raws_in[0],
                raws_in[1],
            )
        )
        tail: Float64[Array, "..."] = (
            _kk_candidate_common.semi_infinite_tail_contribution(
                jnp.asarray([grid_np[0], grid_np[-1]], dtype=jnp.float64),
                tail_spec,
                appended,
                n_tail=config.n_tail,
            )
        )
        total: Float64[Array, "..."] = core + tail
        sigma_subtracted: Float64[Array, " n_any"] = total[:-1] - total[-1]
        return sigma_subtracted

    def _contract_queries(
        query_values: Float64[Array, " n_query"],
    ) -> Float64[Array, ""]:
        """PRIVATE: Compute the output contraction while varying queries.

        Parameters
        ----------
        query_values : Float64[Array, " n_query"]
            Query energies in eV.

        Returns
        -------
        contracted : Float64[Array, ""]
            Weighted scalar contraction of the real self-energy in eV.

        Notes
        -----
        The closure fixes the core samples and tail coordinates. The weight
        vector gives each query a nonzero reverse-mode sensitivity.
        """
        contracted: Float64[Array, ""] = jnp.sum(
            weights * _subtracted_from(values, raws_vec, query_values)
        )
        return contracted

    def _contract_values(
        core_values: Float64[Array, " n_kk"],
    ) -> Float64[Array, ""]:
        """PRIVATE: Compute the output contraction while varying core samples.

        Parameters
        ----------
        core_values : Float64[Array, " n_kk"]
            Imaginary self-energy samples in eV.

        Returns
        -------
        contracted : Float64[Array, ""]
            Weighted scalar contraction of the real self-energy in eV.

        Notes
        -----
        The closure fixes the queries and tail coordinates. JAX differentiates
        every core sample through the principal-value operator.
        """
        contracted: Float64[Array, ""] = jnp.sum(
            weights * _subtracted_from(core_values, raws_vec, queries_jnp)
        )
        return contracted

    def _contract_raws(
        raw_values: Float64[Array, " 2"],
    ) -> Float64[Array, ""]:
        """PRIVATE: Compute the contraction while varying tail coordinates.

        Parameters
        ----------
        raw_values : Float64[Array, " 2"]
            Raw coordinates for the left and right tails.

        Returns
        -------
        contracted : Float64[Array, ""]
            Weighted scalar contraction of the real self-energy in eV.

        Notes
        -----
        The closure fixes the queries and core samples. JAX differentiates the
        softplus curvature coordinates through both tail integrals.
        """
        contracted: Float64[Array, ""] = jnp.sum(
            weights * _subtracted_from(values, raw_values, queries_jnp)
        )
        return contracted

    metrics: Dict[str, Any] = {
        "requirement": "kk-reverse-mode-consistency",
        "rules": (
            "forward-versus-reverse identities at roundoff scale; "
            "finite-difference directional cross-checks at the "
            "program-wide f64 ladder rtol 1e-6 (the query-direction "
            "difference quotient carries an intrinsic truncation floor "
            "from node-crossing logarithmic kinks of the interpolant, "
            "so roundoff-scale FD agreement is not attainable there)"
        ),
    }

    grad_queries: Float64[Array, " n_query"] = jax.grad(_contract_queries)(
        queries_jnp
    )
    dsigma: Float64[Array, " n_query"]
    _, dsigma = jax.jvp(
        lambda q: _subtracted_from(values, raws_vec, q),
        (queries_jnp,),
        (jnp.ones_like(queries_jnp),),
    )
    per_query_diff: float = float(
        jnp.max(jnp.abs(grad_queries - weights * dsigma))
    )
    query_scale: float = max(1.0, float(jnp.max(jnp.abs(grad_queries))))
    query_tangent: Float64[Array, " n_query"] = (
        jnp.sin(2.0 * queries_jnp) + 0.25
    )
    forward_query: Float64[Array, ""]
    _, forward_query = jax.jvp(
        _contract_queries, (queries_jnp,), (query_tangent,)
    )
    dot_query: float = float(jnp.sum(grad_queries * query_tangent))
    fd_query: float = _five_point_scalar_fd(
        lambda s: float(_contract_queries(queries_jnp + s * query_tangent)),
        DIRECTIONAL_FD_STEP,
    )
    metrics["queries"] = {
        "max_abs_grad": float(jnp.max(jnp.abs(grad_queries))),
        "reverse_vs_forward_per_query_max_abs": per_query_diff,
        "pass_reverse_vs_forward": bool(
            per_query_diff <= 1.0e-12 * query_scale
        ),
        "directional_reverse": dot_query,
        "directional_forward": float(forward_query),
        "directional_fd": fd_query,
        "abs_reverse_minus_forward": abs(dot_query - float(forward_query)),
        "abs_forward_minus_fd": abs(float(forward_query) - fd_query),
        "pass_fd_agreement": bool(
            abs(float(forward_query) - fd_query)
            <= REVERSE_FD_RTOL * max(1.0, abs(float(forward_query)))
        ),
    }

    grad_values: Float64[Array, " n_kk"] = jax.grad(_contract_values)(values)
    # The tangent is relative to the sampled magnitude so the finite
    # difference keeps the derived tail parameters in their smooth
    # regime: an absolute tangent would perturb the tiny edge amplitude
    # by order one across the stencil and invalidate the difference
    # quotient without measuring any AD defect.
    value_tangent: Float64[Array, " n_kk"] = values * (
        0.5 * jnp.cos(1.3 * grid)
    )
    forward_value: Float64[Array, ""]
    _, forward_value = jax.jvp(_contract_values, (values,), (value_tangent,))
    dot_value: float = float(jnp.sum(grad_values * value_tangent))
    fd_value: float = _five_point_scalar_fd(
        lambda s: float(_contract_values(values + s * value_tangent)),
        DIRECTIONAL_FD_STEP,
    )
    metrics["core_samples"] = {
        "max_abs_grad": float(jnp.max(jnp.abs(grad_values))),
        "directional_reverse": dot_value,
        "directional_forward": float(forward_value),
        "directional_fd": fd_value,
        "abs_reverse_minus_forward": abs(dot_value - float(forward_value)),
        "pass_reverse_vs_forward": bool(
            abs(dot_value - float(forward_value))
            <= REVERSE_FORWARD_RTOL * max(1.0, abs(float(forward_value)))
        ),
        "abs_forward_minus_fd": abs(float(forward_value) - fd_value),
        "pass_fd_agreement": bool(
            abs(float(forward_value) - fd_value)
            <= REVERSE_FD_RTOL * max(1.0, abs(float(forward_value)))
        ),
    }

    grad_raws: Float64[Array, " 2"] = jax.grad(_contract_raws)(raws_vec)
    raw_rows: Dict[str, Any] = {}
    raw_pass: bool = True
    coordinate: int
    name: str
    for coordinate, name in enumerate(("raw_left", "raw_right")):
        direction: Float64[Array, " 2"] = jnp.zeros(2).at[coordinate].set(1.0)
        forward_raw: Float64[Array, ""]
        _, forward_raw = jax.jvp(_contract_raws, (raws_vec,), (direction,))
        reverse_raw: float = float(grad_raws[coordinate])
        fd_raw: float = _five_point_scalar_fd(
            lambda s, d=direction: float(_contract_raws(raws_vec + s * d)),
            RAW_FD_STEP,
        )
        forward_reverse: float = abs(reverse_raw - float(forward_raw))
        fd_error: float = abs(reverse_raw - fd_raw)
        # The raw gradients are tiny (softplus-suppressed), so the
        # difference-quotient roundoff floor eps*|f|/(12*step) dominates
        # the relative criterion and is included explicitly.
        row_pass: bool = bool(
            forward_reverse
            <= REVERSE_FORWARD_RTOL * max(1.0, abs(float(forward_raw)))
            and fd_error
            <= max(REVERSE_RAW_FD_RTOL * abs(reverse_raw), RAW_FD_NOISE_FLOOR)
        )
        raw_pass = raw_pass and row_pass
        raw_rows[name] = {
            "reverse": reverse_raw,
            "forward": float(forward_raw),
            "fd": fd_raw,
            "abs_reverse_minus_forward": forward_reverse,
            "abs_reverse_minus_fd": fd_error,
            "pass": row_pass,
        }
    metrics["raw_tail_coordinates"] = {**raw_rows, "pass_all": raw_pass}

    node_indices: Tuple[int, int, int] = (1024, 2048, 3072)
    node_queries: Float64[Array, " 3"] = grid[jnp.asarray(node_indices)]

    def _contract_node_queries(
        query_values: Float64[Array, " 3"],
    ) -> Float64[Array, ""]:
        """PRIVATE: Compute a contraction at node-coincident queries.

        Parameters
        ----------
        query_values : Float64[Array, " 3"]
            Three query energies that equal core-grid nodes, in eV.

        Returns
        -------
        contracted : Float64[Array, ""]
            Sum of the three subtracted real self-energy values in eV.

        Notes
        -----
        The closure exercises exact logarithm cancellation at grid nodes. It
        checks the reverse-mode query seam without an extra weight vector.
        """
        contracted: Float64[Array, ""] = jnp.sum(
            _subtracted_from(values, raws_vec, query_values)
        )
        return contracted

    def _contract_values_at_nodes(
        core_values: Float64[Array, " n_kk"],
    ) -> Float64[Array, ""]:
        """PRIVATE: Compute node-query contractions while varying core samples.

        Parameters
        ----------
        core_values : Float64[Array, " n_kk"]
            Imaginary self-energy samples in eV.

        Returns
        -------
        contracted : Float64[Array, ""]
            Sum of the node-query real self-energy values in eV.

        Notes
        -----
        The closure fixes three node-coincident queries and the tail
        coordinates. JAX differentiates every core sample.
        """
        contracted: Float64[Array, ""] = jnp.sum(
            _subtracted_from(core_values, raws_vec, node_queries)
        )
        return contracted

    grad_node_queries: Float64[Array, " 3"] = jax.grad(_contract_node_queries)(
        node_queries
    )
    grad_values_nodes: Float64[Array, " n_kk"] = jax.grad(
        _contract_values_at_nodes
    )(values)
    forward_nodes: Float64[Array, ""]
    _, forward_nodes = jax.jvp(
        _contract_values_at_nodes, (values,), (value_tangent,)
    )
    dot_nodes: float = float(jnp.sum(grad_values_nodes * value_tangent))
    fd_nodes: float = _five_point_scalar_fd(
        lambda s: float(_contract_values_at_nodes(values + s * value_tangent)),
        DIRECTIONAL_FD_STEP,
    )
    metrics["node_coincident_queries"] = {
        "node_indices": list(node_indices),
        "grad_wrt_queries_finite": bool(
            np.all(np.isfinite(np.asarray(grad_node_queries)))
        ),
        "grad_wrt_core_samples_finite": bool(
            np.all(np.isfinite(np.asarray(grad_values_nodes)))
        ),
        "directional_reverse": dot_nodes,
        "directional_forward": float(forward_nodes),
        "directional_fd": fd_nodes,
        "abs_reverse_minus_forward": abs(dot_nodes - float(forward_nodes)),
        "pass_reverse_vs_forward": bool(
            abs(dot_nodes - float(forward_nodes))
            <= REVERSE_FORWARD_RTOL * max(1.0, abs(float(forward_nodes)))
        ),
        "abs_forward_minus_fd": abs(float(forward_nodes) - fd_nodes),
        "pass_fd_agreement": bool(
            abs(float(forward_nodes) - fd_nodes)
            <= REVERSE_FD_RTOL * max(1.0, abs(float(forward_nodes)))
        ),
    }

    arrays: Dict[str, Float64[NDArray, "..."]] = {
        f"reverse_{candidate_key}_grad_queries": np.asarray(grad_queries),
        f"reverse_{candidate_key}_grad_core_samples": np.asarray(grad_values),
        f"reverse_{candidate_key}_grad_raw_tail": np.asarray(grad_raws),
    }
    evidence: Tuple[Dict[str, Any], Dict[str, Float64[NDArray, "..."]]] = (
        metrics,
        arrays,
    )
    return evidence


def _two_band_intensity(
    sigma_sub: Float64[NDArray, " n_omega"],
    sigma_imag: Float64[NDArray, " n_omega"],
    omega: Float64[NDArray, " n_omega"],
) -> Float64[NDArray, " n_omega"]:
    """PRIVATE: Evaluate the frozen two-band spectral intensity.

    Parameters
    ----------
    sigma_sub : Float64[NDArray, " n_omega"]
        Subtracted real self-energy in eV on the omega grid.
    sigma_imag : Float64[NDArray, " n_omega"]
        Imaginary self-energy in eV on the omega grid.
    omega : Float64[NDArray, " n_omega"]
        Energy grid in eV.

    Returns
    -------
    intensity : Float64[NDArray, " n_omega"]
        Spectral intensity in 1/eV at every omega.

    Implementation Logic
    --------------------
    ``G(omega) = [(omega + i eta - Sigma(omega)) I - H]^{-1}`` with the
    scalar retarded self-energy ``Sigma = Sigma'_sub + i Sigma''``; the
    intensity is ``-Im(M^dagger G M)/pi``.
    """
    sigma: Complex128[NDArray, " n_omega"] = sigma_sub + 1j * sigma_imag
    z_values: Complex128[NDArray, " n_omega"] = (
        omega + 1j * SPECTRAL_ETA_EV - sigma
    )
    identity: Complex128[NDArray, "2 2"] = np.eye(2, dtype=np.complex128)
    matrices: Complex128[NDArray, "n_omega 2 2"] = (
        z_values[:, None, None] * identity[None, :, :]
        - SPECTRAL_HAMILTONIAN_EV[None, :, :]
    )
    sources: Complex128[NDArray, "n_omega 2"] = np.broadcast_to(
        SPECTRAL_SOURCE, (omega.shape[0], 2)
    )
    solved: Complex128[NDArray, "n_omega 2"] = np.linalg.solve(
        matrices, sources[..., None]
    )[..., 0]
    projected: Complex128[NDArray, " n_omega"] = np.einsum(
        "i,ni->n", np.conj(SPECTRAL_SOURCE), solved
    )
    intensity: Float64[NDArray, " n_omega"] = -projected.imag / np.pi
    return intensity


def _quadratic_peak_ev(
    intensity: Float64[NDArray, " n_omega"],
    omega: Float64[NDArray, " n_omega"],
) -> float:
    """PRIVATE: Return the quadratic-interpolated peak position.

    Parameters
    ----------
    intensity : Float64[NDArray, " n_omega"]
        Spectral intensity in 1/eV on the omega grid.
    omega : Float64[NDArray, " n_omega"]
        Uniform energy grid in eV.

    Returns
    -------
    peak_ev : float
        Sub-grid peak position in eV.

    Notes
    -----
    Fit a parabola through the maximum sample and both neighbors. Clamp argmax
    away from the ends. Shift the node by
    ``0.5 (left - right) / (left - 2 center + right)`` cells. Keep the node
    position when the denominator equals zero.
    """
    index: int = int(np.argmax(intensity))
    index = min(max(index, 1), intensity.shape[0] - 2)
    left: float
    center: float
    right: float
    left, center, right = intensity[index - 1 : index + 2]
    denominator: float = left - 2.0 * center + right
    shift: float = (
        0.0 if denominator == 0.0 else 0.5 * (left - right) / denominator
    )
    peak_ev: float = float(omega[index] + shift * (omega[1] - omega[0]))
    return peak_ev


def _spectral_observable_rows(
    raw_records: Dict[str, Dict[str, Any]],
) -> Tuple[Dict[str, Any], Dict[str, Float64[NDArray, "..."]]]:
    """PRIVATE: Measure the two-band spectral observable stability rows.

    Parameters
    ----------
    raw_records : Dict[str, Dict[str, Any]]
        Frozen per-candidate raw tail records keyed by domain.

    Returns
    -------
    evidence : Tuple[Dict[str, Any], Dict[str, Float64[NDArray, "..."]]]
        Metrics dictionary (L1 shape change, weight change, peak
        motion, and verdicts per refinement) and the intensity arrays
        for the archive.

    Notes
    -----
    The self-energy that feeds the two-band fixture is the
    retarded-pole fixture: Sigma'' analytic, Sigma' from the operator
    under test, subtracted at ``omega_s = 0``. Shape changes integrate
    over the frozen 4001-point omega grid on ``[-1, 1]`` eV.
    """
    omega: Float64[NDArray, " n_omega"] = np.linspace(
        -1.0, 1.0, SPECTRAL_N_OMEGA
    )
    omega_jnp: Float64[Array, " n_omega"] = jnp.asarray(omega)
    params: Float64[Array, " n_param"] = jnp.asarray(
        POLE_PARAMS, dtype=jnp.float64
    )
    omega0: float
    gamma: float
    coupling: float
    omega0, gamma, coupling = POLE_PARAMS
    offset: Float64[NDArray, " n_omega"] = omega - omega0
    sigma_imag: Float64[NDArray, " n_omega"] = (
        -coupling * gamma / (offset * offset + gamma * gamma)
    )

    metrics: Dict[str, Any] = {
        "requirement": "kk-spectral-observable-stability",
        "sigma_model": (
            "retarded-pole fixture: Sigma'' analytic, Sigma' from the "
            "operator under test, subtracted at omega_s = 0 eV"
        ),
        "hamiltonian_ev": [
            [str(entry) for entry in row]
            for row in SPECTRAL_HAMILTONIAN_EV.tolist()
        ],
        "source": [str(entry) for entry in SPECTRAL_SOURCE.tolist()],
        "eta_ev": SPECTRAL_ETA_EV,
        "n_omega": SPECTRAL_N_OMEGA,
    }
    arrays: Dict[str, Float64[NDArray, "..."]] = {"spectral_omega_ev": omega}
    spacing: float = omega[1] - omega[0]
    candidate_key: str
    module: Any
    for candidate_key, module in CANDIDATE_MODULES.items():
        intensities: Dict[str, Float64[NDArray, " n_omega"]] = {}
        config_key: str
        for config_key in FIXTURE_CONFIG_KEYS["pole"]:
            config: GridConfig = CONFIGS[config_key]
            domain_key: str = (
                "extended_domain"
                if config.construction == "phase_aligned_extension"
                else "base_domain"
            )
            record: Dict[str, Any] = raw_records[candidate_key][domain_key]
            raws: Tuple[float, float] = (
                record["raw_left"],
                record["raw_right"],
            )
            appended: Float64[Array, " n_appended"] = jnp.concatenate(
                (omega_jnp, jnp.asarray([OMEGA_S_EV], dtype=jnp.float64))
            )
            total: Float64[NDArray, " n_appended"] = np.asarray(
                _sigma_prime_unsubtracted(
                    module,
                    candidate_key,
                    "pole",
                    config,
                    params,
                    raws,
                    appended,
                )
            )
            sigma_sub: Float64[NDArray, " n_omega"] = total[:-1] - total[-1]
            intensity: Float64[NDArray, " n_omega"] = _two_band_intensity(
                sigma_sub, sigma_imag, omega
            )
            intensities[config_key] = intensity
            arrays[f"spectral_{candidate_key}_{config_key}_intensity"] = (
                intensity
            )
        base_intensity: Float64[NDArray, " n_omega"] = intensities["base"]
        base_weight: float = float(np.trapezoid(base_intensity, dx=spacing))
        base_shape: Float64[NDArray, " n_omega"] = base_intensity / base_weight
        base_peak: float = _quadratic_peak_ev(base_intensity, omega)
        candidate_rows: Dict[str, Any] = {"base_peak_ev": base_peak}
        for config_key in FIXTURE_CONFIG_KEYS["pole"][1:]:
            refined: Float64[NDArray, " n_omega"] = intensities[config_key]
            weight: float = float(np.trapezoid(refined, dx=spacing))
            shape: Float64[NDArray, " n_omega"] = refined / weight
            l1_change: float = float(
                np.trapezoid(np.abs(shape - base_shape), dx=spacing)
            )
            weight_change: float = abs(weight - base_weight) / abs(base_weight)
            peak_motion: float = abs(
                _quadratic_peak_ev(refined, omega) - base_peak
            )
            candidate_rows[config_key] = {
                "l1_shape_change": l1_change,
                "pass_l1_shape_1em5": bool(l1_change <= SPECTRAL_L1_BUDGET),
                "weight_relative_change": weight_change,
                "pass_weight_1em5": bool(
                    weight_change <= SPECTRAL_WEIGHT_BUDGET
                ),
                "peak_motion_ev": peak_motion,
                "pass_peak_2em5": bool(peak_motion <= SPECTRAL_PEAK_BUDGET_EV),
            }
        metrics[candidate_key] = candidate_rows
    evidence: Tuple[Dict[str, Any], Dict[str, Float64[NDArray, "..."]]] = (
        metrics,
        arrays,
    )
    return evidence


def _array_bytes(array: Float64[NDArray, "..."]) -> bytes:
    """PRIVATE: Serialize one NumPy array without timestamp metadata.

    Parameters
    ----------
    array : Float64[NDArray, "..."]
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
    arrays: Dict[str, Float64[NDArray, "..."]],
) -> None:
    """PRIVATE: Write an NPZ whose members have stable order and dates.

    Parameters
    ----------
    path : Path
        Destination NPZ path.
    arrays : Dict[str, Float64[NDArray, "..."]]
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
        array: Float64[NDArray, "..."]
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
    digest : str
        Lowercase hexadecimal SHA-256 of the file bytes.

    Notes
    -----
    The function reads the complete file into memory before hashing;
    every evidence file stays small enough for that.
    """
    digest: str = hashlib.sha256(path.read_bytes()).hexdigest()
    return digest


def _executable_input_provenance() -> Dict[str, Any]:
    """PRIVATE: Record the digest of every evidence-shaping executable.

    Returns
    -------
    provenance : Dict[str, Any]
        Path and SHA-256 for the generator, the shared scaffolding, the
        three candidate modules, and the Maclaurin control, plus the
        analytic-arbiter archive, manifest, and generator digests.

    Notes
    -----
    The analytic-arbiter entry also copies the manifest-recorded
    archive digest and arbiter description, so the manifest exposes
    both the observed and the recorded values.
    """
    module_files: Dict[str, Path] = {
        "generator": Path(__file__).resolve(),
        "common": _TOOLS_DIRECTORY / "_kk_candidate_common.py",
        "piecewise_linear": (
            _TOOLS_DIRECTORY / "_kk_candidate_piecewise_linear.py"
        ),
        "piecewise_quadratic": (
            _TOOLS_DIRECTORY / "_kk_candidate_piecewise_quadratic.py"
        ),
        "piecewise_cubic": (
            _TOOLS_DIRECTORY / "_kk_candidate_piecewise_cubic.py"
        ),
        "maclaurin_control": (
            _TOOLS_DIRECTORY / "_kk_control_opposite_parity_maclaurin.py"
        ),
    }
    provenance: Dict[str, Any] = {
        key: {
            "path": f"tests/_reference_tools/{path.name}",
            "sha256": _sha256(path),
        }
        for key, path in module_files.items()
    }
    reference_directory: Path = _kk_candidate_common.REFERENCE_DIRECTORY
    truth_manifest: Dict[str, Any] = json.loads(
        (reference_directory / "manifest.json").read_text(encoding="utf-8")
    )
    provenance["analytic_arbiter"] = {
        "directory": (
            "tests/test_diffpes/_reference_data/kk_analytic_reference"
        ),
        "archive_sha256": _sha256(reference_directory / "kk_reference.npz"),
        "manifest_sha256": _sha256(reference_directory / "manifest.json"),
        "generator_sha256": _sha256(
            reference_directory / "generate_kk_analytic_reference.py"
        ),
        "manifest_recorded_archive_sha256": truth_manifest["archive_sha256"],
        "arbiter": truth_manifest["arbiter"],
    }
    return provenance


def main() -> None:  # noqa: PLR0912, PLR0915 -- frozen evidence battery.
    """Run the operator comparison and write its authority artifact.

    Notes
    -----
    The function evaluates all fixtures, refinements, derivatives, and spectral
    observables. It writes deterministic arrays and complete provenance.
    """
    reference: Dict[str, Float64[NDArray, "..."]] = (
        _kk_candidate_common.load_analytic_reference()
    )
    queries: Float64[NDArray, " n_query"] = reference["pole_omega"]
    if not np.array_equal(queries, reference["semicircle_omega"]):
        raise RuntimeError("truth archive query grids disagree")
    _verify_wigner_zero_edges()

    truth: Dict[str, Float64[NDArray, " n_query"]] = {
        "truth_pole_sigma_real_sub_ev": reference["pole_sigma_real"],
        "truth_wigner_sigma_real_sub_ev": reference["semicircle_sigma_real"],
    }
    truth.update(_analytic_truth(queries))

    base_grid: Float64[NDArray, " n_base"] = _core_grid_np(CONFIGS["base"])
    extension_grid: Float64[NDArray, " n_extension"] = _core_grid_np(
        CONFIGS["domain_extension"]
    )
    embedded: Float64[NDArray, " n_base"] = extension_grid[
        EXTENSION_SHIFT_CELLS : EXTENSION_SHIFT_CELLS + BASE_N_KK
    ]
    embedding_mismatches: int = int(np.sum(embedded != base_grid))
    query_node_collisions: int = int(np.intersect1d(queries, base_grid).size)

    arrays: Dict[str, Float64[NDArray, "..."]] = {
        "queries_ev": queries,
        **truth,
    }
    metrics: Dict[str, Any] = {}
    raw_records: Dict[str, Dict[str, Any]] = {}
    candidate_key: str
    for candidate_key in (*CANDIDATE_MODULES, "pwlinear"):
        raw_records[candidate_key] = {
            "base_domain": _pole_tail_raw_parameters(candidate_key, base_grid),
            "extended_domain": _pole_tail_raw_parameters(
                candidate_key, extension_grid
            ),
        }

    def _pole_raws(candidate_key: str, domain_key: str) -> Tuple[float, float]:
        """PRIVATE: Read the two pole-tail coordinates for one domain.

        Parameters
        ----------
        candidate_key : str
            Principal-value operator key.
        domain_key : str
            Model-domain key for the stored tail record.

        Returns
        -------
        raw_values : Tuple[float, float]
            Left and right raw curvature coordinates.

        Notes
        -----
        The closure reads coordinates that the analytic pole edges define. It
        does not inspect a candidate output.
        """
        record: Dict[str, Any] = raw_records[candidate_key][domain_key]
        raw_values: Tuple[float, float] = (
            record["raw_left"],
            record["raw_right"],
        )
        return raw_values

    scenario_values: Dict[
        Tuple[str, str, str], Dict[str, Float64[NDArray, " n_query"]]
    ] = {}
    module: Any
    for candidate_key, module in CANDIDATE_MODULES.items():
        fixture_key: str
        for fixture_key in ("pole", "wigner"):
            config_key: str
            for config_key in FIXTURE_CONFIG_KEYS[fixture_key]:
                config: GridConfig = CONFIGS[config_key]
                raws: Tuple[float, float] | None = None
                if fixture_key == "pole":
                    domain_key: str = (
                        "extended_domain"
                        if config.construction == "phase_aligned_extension"
                        else "base_domain"
                    )
                    raws = _pole_raws(candidate_key, domain_key)
                values_only: bool = config_key == "grid16384"
                print(
                    f"evaluating {candidate_key}/{fixture_key}/"
                    f"{config_key} ..."
                )
                outputs: Dict[str, Float64[NDArray, " n_query"]] = (
                    _evaluate_scenario(
                        module,
                        candidate_key,
                        fixture_key,
                        config,
                        raws,
                        queries,
                        values_only=values_only,
                    )
                )
                scenario_values[(candidate_key, fixture_key, config_key)] = (
                    outputs
                )
                name: str
                value: Float64[NDArray, " n_query"]
                for name, value in outputs.items():
                    arrays[
                        f"{candidate_key}_{fixture_key}_{config_key}_{name}"
                    ] = value
        print(f"evaluating {candidate_key}/pole/domain_extension_held ...")
        held_outputs: Dict[str, Float64[NDArray, " n_query"]] = (
            _evaluate_scenario(
                module,
                candidate_key,
                "pole",
                CONFIGS["domain_extension"],
                _pole_raws(candidate_key, "base_domain"),
                queries,
            )
        )
        scenario_values[(candidate_key, "pole", "domain_extension_held")] = (
            held_outputs
        )
        for name, value in held_outputs.items():
            arrays[
                f"{candidate_key}_pole_domain_extension_held_raw_{name}"
            ] = value

    metrics["candidates"] = {}
    for candidate_key, module in CANDIDATE_MODULES.items():
        pole_base: Dict[str, Float64[NDArray, " n_query"]] = scenario_values[
            (candidate_key, "pole", "base")
        ]
        pole_metrics: Dict[str, Any] = {
            "analytic_pair_truth": {"requirement": "kk-analytic-pair-truth"},
            "derivative_composite": {
                "requirement": "kk-derivative-composite-route"
            },
            "refinements": {"requirement": "kk-refinement-convergence"},
            "query_derivative_identity_max_abs_dev": {},
        }
        for config_key in FIXTURE_CONFIG_KEYS["pole"]:
            outputs = scenario_values[(candidate_key, "pole", config_key)]
            pole_metrics["query_derivative_identity_max_abs_dev"][
                config_key
            ] = float(outputs["identity_spot_max_abs_dev"])
            pole_metrics["analytic_pair_truth"][config_key] = (
                _mixed_criterion_statistics(
                    outputs["sigma_sub_ev"],
                    truth["truth_pole_sigma_real_sub_ev"],
                    queries,
                )
            )
            pole_metrics["derivative_composite"][config_key] = {
                "max_abs_error": float(
                    np.max(
                        np.abs(
                            outputs["dsigma_composite"]
                            - truth["truth_pole_dsigma_domega"]
                        )
                    )
                ),
                "max_abs_error_direct_ad": float(
                    np.max(
                        np.abs(
                            outputs["dsigma_direct"]
                            - truth["truth_pole_dsigma_domega"]
                        )
                    )
                ),
            }
        for config_key in FIXTURE_CONFIG_KEYS["pole"][1:]:
            deltas: Dict[str, Any] = _refinement_deltas(
                pole_base,
                scenario_values[(candidate_key, "pole", config_key)],
                POLE_PARAM_NAMES,
            )
            row: Dict[str, Any] = _pole_refinement_metrics(
                deltas, is_tail_refinement=(config_key == "tail512")
            )
            if config_key == "domain_extension":
                row["tail_raw_convention"] = "recomputed (domain changed)"
                held_deltas: Dict[str, Any] = _refinement_deltas(
                    pole_base,
                    scenario_values[
                        (candidate_key, "pole", "domain_extension_held")
                    ],
                    POLE_PARAM_NAMES,
                )
                row["held_raw_deltas"] = held_deltas
            pole_metrics["refinements"][config_key] = row

        wigner_base: Dict[str, Float64[NDArray, " n_query"]] = scenario_values[
            (candidate_key, "wigner", "base")
        ]
        value_errors: Dict[str, Any] = {
            config_key: float(
                np.max(
                    np.abs(
                        scenario_values[(candidate_key, "wigner", config_key)][
                            "sigma_sub_ev"
                        ]
                        - truth["truth_wigner_sigma_real_sub_ev"]
                    )
                )
            )
            for config_key in FIXTURE_CONFIG_KEYS["wigner"]
        }
        order_first: float = float(
            np.log2(value_errors["base"] / value_errors["grid8192"])
        )
        order_second: float = float(
            np.log2(value_errors["grid8192"] / value_errors["grid16384"])
        )
        monotone: bool = bool(
            value_errors["base"]
            > value_errors["grid8192"]
            > value_errors["grid16384"]
        )
        wigner_metrics: Dict[str, Any] = {
            "stress_witness": {
                "requirement": "kk-singularity-stress-witness",
                "value_error_ev": value_errors,
                "value_order_4096_to_8192": order_first,
                "value_order_8192_to_16384": order_second,
                "pass_value_orders_min_1p4": bool(
                    order_first >= WITNESS_MIN_VALUE_ORDER
                    and order_second >= WITNESS_MIN_VALUE_ORDER
                ),
                "pass_monotone_decrease": monotone,
                "pass_base_value_error_1em5": bool(
                    value_errors["base"] <= WITNESS_BASE_ERROR_EV
                ),
            },
            "refinements": {"requirement": "kk-refinement-convergence"},
            "query_derivative_identity_max_abs_dev": {
                config_key: float(
                    scenario_values[(candidate_key, "wigner", config_key)][
                        "identity_spot_max_abs_dev"
                    ]
                )
                for config_key in ("base", "grid8192", "domain_extension")
            },
        }
        for config_key in ("grid8192", "domain_extension"):
            deltas = _refinement_deltas(
                wigner_base,
                scenario_values[(candidate_key, "wigner", config_key)],
                WIGNER_PARAM_NAMES,
            )
            wigner_metrics["refinements"][config_key] = (
                _wigner_refinement_metrics(deltas)
            )

        fixture_metrics: Dict[str, Any]
        for fixture_key, fixture_metrics in (
            ("pole", pole_metrics),
            ("wigner", wigner_metrics),
        ):
            fd_agreement: Dict[str, float] = {}
            fixture_raws: Tuple[float, float] | None = (
                _pole_raws(candidate_key, "base_domain")
                if fixture_key == "pole"
                else None
            )
            base_outputs: Dict[str, Float64[NDArray, " n_query"]] = (
                scenario_values[(candidate_key, fixture_key, "base")]
            )
            param_index: int
            param_name: str
            for param_index, param_name in enumerate(
                FIXTURE_PARAM_NAMES[fixture_key]
            ):
                fd: Float64[NDArray, " n_query"] = _five_point_fd(
                    module,
                    candidate_key,
                    fixture_key,
                    CONFIGS["base"],
                    fixture_raws,
                    queries,
                    param_index,
                )
                arrays[
                    f"{candidate_key}_{fixture_key}_base_fd5_{param_name}"
                ] = fd
                fd_agreement[param_name] = float(
                    np.max(np.abs(fd - base_outputs[f"jvp_{param_name}"]))
                )
            fixture_metrics["fd_vs_jvp_max_abs"] = fd_agreement
        metrics["candidates"][candidate_key] = {
            "pole": pole_metrics,
            "wigner": wigner_metrics,
        }

    metrics["reverse_mode"] = {}
    for candidate_key, module in CANDIDATE_MODULES.items():
        print(f"reverse-mode evidence for {candidate_key} ...")
        reverse_metrics: Dict[str, Any]
        reverse_arrays: Dict[str, Float64[NDArray, "..."]]
        reverse_metrics, reverse_arrays = _reverse_mode_evidence(
            module,
            candidate_key,
            _pole_raws(candidate_key, "base_domain"),
            queries,
        )
        metrics["reverse_mode"][candidate_key] = reverse_metrics
        arrays.update(reverse_arrays)

    print("carrier-consistency rows ...")
    carrier_metrics: Dict[str, Any]
    carrier_arrays: Dict[str, Float64[NDArray, "..."]]
    carrier_metrics, carrier_arrays = _carrier_consistency(queries)
    metrics["carrier_consistency"] = carrier_metrics
    metrics["smooth_seam_consistency"] = _smooth_seam_consistency(raw_records)
    arrays.update(carrier_arrays)

    print("two-band spectral observable rows ...")
    spectral_metrics: Dict[str, Any]
    spectral_arrays: Dict[str, Float64[NDArray, "..."]]
    spectral_metrics, spectral_arrays = _spectral_observable_rows(raw_records)
    metrics["spectral_observable"] = spectral_metrics
    arrays.update(spectral_arrays)

    metrics["maclaurin_control"] = {
        "requirement": "kk-rejected-control-reference"
    }
    for fixture_key in ("pole", "wigner"):
        control_raws: Tuple[float, float] | None = (
            _pole_raws("pwlinear", "base_domain")
            if fixture_key == "pole"
            else None
        )
        print(f"evaluating maclaurin_control/{fixture_key}/base ...")
        outputs = _evaluate_scenario(
            CONTROL_MODULE,
            "pwlinear",
            fixture_key,
            CONFIGS["base"],
            control_raws,
            queries,
        )
        arrays[f"control_{fixture_key}_base_sigma_sub_ev"] = outputs[
            "sigma_sub_ev"
        ]
        arrays[f"control_{fixture_key}_base_dsigma_direct"] = outputs[
            "dsigma_direct"
        ]
        truth_key: str = (
            "truth_pole_sigma_real_sub_ev"
            if fixture_key == "pole"
            else "truth_wigner_sigma_real_sub_ev"
        )
        metrics["maclaurin_control"][fixture_key] = {
            **_mixed_criterion_statistics(
                outputs["sigma_sub_ev"], truth[truth_key], queries
            ),
            "max_abs_dsigma_error": float(
                np.max(
                    np.abs(
                        outputs["dsigma_direct"]
                        - truth[f"truth_{fixture_key}_dsigma_domega"]
                    )
                )
            ),
        }

    root: Path = Path(__file__).resolve().parents[2]
    data_directory: Path = root / "tests" / "test_diffpes" / "_reference_data"
    archive_path: Path = data_directory / "kk_operator_selection_reference.npz"
    manifest_path: Path = (
        data_directory / "kk_operator_selection_manifest.json"
    )
    _write_deterministic_npz(archive_path, arrays)

    candidate_provenance: Dict[str, Dict[str, str]] = {
        key: {
            "module": f"tests/_reference_tools/{module.__name__}.py",
            "name": module.NAME,
            "description": module.DESCRIPTION,
        }
        for key, module in {
            **CANDIDATE_MODULES,
            "grid_carrier": GRID_CARRIER_MODULE,
            "maclaurin_control": CONTROL_MODULE,
        }.items()
    }
    manifest: Dict[str, Any] = {
        "schema": "diffpes.kk-operator-selection.v1",
        "purpose": (
            "independent Kramers-Kronig operator selection: candidate "
            "comparison, carrier consistency, derivative route, "
            "reverse-mode consistency, and spectral observable stability"
        ),
        "archive": archive_path.name,
        "archive_sha256": _sha256(archive_path),
        "executable_inputs": _executable_input_provenance(),
        "candidate_modules": candidate_provenance,
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "jax": jax.__version__,
            "platform": platform.platform(),
        },
        "fixtures": {
            "pole": dict(zip(POLE_PARAM_NAMES, POLE_PARAMS, strict=True)),
            "wigner": dict(
                zip(WIGNER_PARAM_NAMES, WIGNER_PARAMS, strict=True)
            ),
            "subtraction_point_ev": OMEGA_S_EV,
            "query_grid": "kk_analytic_reference pole_omega (1001 pts)",
        },
        "configurations": {
            key: config._asdict() for key, config in CONFIGS.items()
        },
        "fixture_configurations": {
            key: list(value) for key, value in FIXTURE_CONFIG_KEYS.items()
        },
        "grid_construction": {
            "uniform": "x_j = low + j*h, h = (high-low)/(n_kk-1)",
            "phase_aligned_extension": (
                "x_j = low + (j - 2048)*h, j = 0..8191, "
                "h = (high-low)/(n_base-1) with n_base = 4096; the "
                "construction expression is frozen, not the endpoints"
            ),
            "extension_endpoints_ev": [
                float(extension_grid[0]),
                float(extension_grid[-1]),
            ],
            "base_nodes_embedded_bitwise": bool(embedding_mismatches == 0),
            "embedding_mismatch_count": embedding_mismatches,
            "n_kk_parity": (
                "even: no node at the subtraction point and no collision "
                "with the frozen query grid"
            ),
            "query_node_collision_count": query_node_collisions,
        },
        "mode_operator_contract": {
            "grid": (
                "cell-integrated piecewise-linear PV transform; exact for "
                "the binding hat-interpolant carrier; tail edge slopes "
                "from the hat interpolant's edge cells"
            ),
            "smooth_analytic": (
                "cell-integrated piecewise-cubic PV transform for smooth "
                "analytic modes (poly, fermi_liquid, analytic fixtures); "
                "tail edge slopes from the one-sided cubic edge stencils"
            ),
        },
        "registered_budgets": {
            "max_delta_sigma_ev": BUDGET_DELTA_SIGMA_EV,
            "max_delta_dsigma": BUDGET_DELTA_DSIGMA,
            "max_delta_jvp_ev": BUDGET_DELTA_JVP_EV,
            "pole_tail_only_max_delta_sigma_ev": BUDGET_TAIL_DELTA_SIGMA_EV,
            "pole_tail_only_max_delta_dsigma": BUDGET_TAIL_DELTA_DSIGMA,
            "pair_truth_atol_ev": PAIR_TRUTH_ATOL_EV,
            "pair_truth_rtol": PAIR_TRUTH_RTOL,
            "wigner_min_value_order": WITNESS_MIN_VALUE_ORDER,
            "wigner_base_value_error_ev": WITNESS_BASE_ERROR_EV,
            "wigner_jvp_g_budget_ev": BUDGET_DELTA_SIGMA_EV,
            "spectral_l1_shape_change": SPECTRAL_L1_BUDGET,
            "spectral_weight_relative_change": SPECTRAL_WEIGHT_BUDGET,
            "spectral_peak_motion_ev": SPECTRAL_PEAK_BUDGET_EV,
        },
        "derivative_route": (
            "query derivatives evaluate the transform of the sampled "
            "analytic derivative of Sigma'' plus the finite-core boundary "
            "terms Sigma''(a)/(a-omega) - Sigma''(b)/(b-omega) and the "
            "exact forward-mode derivative of the tail quadrature; the "
            "subtraction constant has zero query derivative"
        ),
        "tail_raw_convention": (
            "raw tail coordinates are carrier state: frozen per model "
            "domain from the analytic fixture before any candidate output "
            "is inspected, held fixed across refinements that keep the "
            "domain unchanged, recomputed only when the domain changes; "
            "the domain-extension row records both conventions"
        ),
        "query_derivative_identity_check": (
            "forward all-ones JVP versus jax.grad at three spot queries; "
            "scale-aware bound 1e-9 * (n_kk/4096) * max(1, |grad|, |jvp|) "
            "covering f64 summation-order conditioning; the measured "
            "maximum deviation is recorded per scenario"
        ),
        "jvp_method": (
            "exact forward-mode jax.jvp through the full pipeline in the "
            "dimensionless coordinate q_p = p/p_fixture; raw tail "
            "coordinates held fixed; cross-checked by a five-point "
            f"central difference with step {FD_STEP}"
        ),
        "pole_tail_construction": {
            candidate_key: raw_records[candidate_key]
            for candidate_key in raw_records
        },
        "metrics": metrics,
        "production_imports": [],
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {archive_path}")
    print(f"wrote {manifest_path}")
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
