"""Shared numerical scaffolding for the preproduction KK candidate study.

This module is test tooling, not production code.  It contains analytic
fixtures, independently checks the committed reference archive, and implements
the frozen semi-infinite tail quadrature used by the candidate sweep.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from beartype.typing import Dict, Tuple
from jax import core
from jaxtyping import Float64
from numpy.typing import NDArray

jax.config.update("jax_enable_x64", True)


REFERENCE_DIRECTORY = (
    Path(__file__).parents[1]
    / "test_diffpes"
    / "_reference_data"
    / "kk_analytic_reference"
)


class Power2TailSpec(NamedTuple):
    """C1-matched power2 parameters in frozen left-then-right order."""

    amplitude_left: Any
    alpha_left: Any
    beta_left: Any
    amplitude_right: Any
    alpha_right: Any
    beta_right: Any


class ZeroTailSpec(NamedTuple):
    """Explicit compact-support tail contract; deliberately has no parameters."""

    kind: str = "zero"


def retarded_pole_fixture(
    points_ev: Any,
    subtraction_point_ev: Any = 0.0,
) -> Tuple[Any, Any]:
    """Return analytic ``(Sigma'', subtracted Sigma')`` for the frozen pole."""
    points = jnp.asarray(points_ev, dtype=jnp.float64)
    subtraction_point = jnp.asarray(subtraction_point_ev, dtype=jnp.float64)
    omega_0 = jnp.float64(0.35)
    gamma = jnp.float64(0.20)
    coupling = jnp.float64(0.12)

    def real(omega: Any) -> Any:
        offset = omega - omega_0
        return coupling * offset / (offset * offset + gamma * gamma)

    offset = points - omega_0
    denominator = offset * offset + gamma * gamma
    sigma_imag = -coupling * gamma / denominator
    return sigma_imag, real(points) - real(subtraction_point)


def wigner_semicircle_fixture(
    points_ev: Any,
    subtraction_point_ev: Any = 0.0,
) -> Tuple[Any, Any]:
    """Return analytic ``(Sigma'', subtracted Sigma')`` for the frozen Wigner fixture."""
    points = jnp.asarray(points_ev, dtype=jnp.float64)
    subtraction_point = jnp.asarray(subtraction_point_ev, dtype=jnp.float64)
    half_width = jnp.float64(1.50)
    coupling = jnp.float64(0.20)
    scale = 2.0 * coupling / half_width**2
    radicand = jnp.maximum(half_width**2 - points**2, 0.0)
    sigma_imag = jnp.where(
        jnp.abs(points) < half_width, -scale * jnp.sqrt(radicand), 0.0
    )

    def real(omega: Any) -> Any:
        # Only the in-band closed form is needed by the frozen trusted query interval.
        return scale * omega

    return sigma_imag, real(points) - real(subtraction_point)


def load_analytic_reference(
    reference_directory: str | Path = REFERENCE_DIRECTORY,
) -> Dict[str, Float64[NDArray, "..."]]:
    """Load the committed reference after verifying its manifest SHA-256 and schema."""
    directory = Path(reference_directory)
    manifest_path = directory / "manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        archive_path = directory / manifest["archive"]
        expected_digest = manifest["archive_sha256"]
    except (OSError, KeyError, json.JSONDecodeError) as error:
        raise RuntimeError(
            f"Invalid KK reference manifest: {manifest_path}"
        ) from error

    actual_digest = hashlib.sha256(archive_path.read_bytes()).hexdigest()
    if actual_digest != expected_digest:
        raise RuntimeError(
            "KK reference archive SHA-256 mismatch: "
            f"expected {expected_digest}, observed {actual_digest} for {archive_path}"
        )

    expected_arrays = manifest.get("arrays")
    if not isinstance(expected_arrays, dict):
        raise RuntimeError("KK reference manifest has no array inventory")
    with np.load(archive_path, allow_pickle=False) as archive:
        observed_names = set(archive.files)
        expected_names = set(expected_arrays)
        if observed_names != expected_names:
            raise RuntimeError(
                f"KK reference array inventory mismatch: {observed_names} != {expected_names}"
            )
        arrays = {}
        for name, shape in expected_arrays.items():
            value = np.asarray(archive[name])
            if value.dtype != np.float64 or list(value.shape) != shape:
                raise RuntimeError(
                    f"KK reference array {name!r} has dtype/shape "
                    f"{value.dtype}/{value.shape}, expected float64/{tuple(shape)}"
                )
            arrays[name] = value.copy()
    return arrays


def _softplus(raw: Any) -> Any:
    """PRIVATE: Compute the softplus of one raw tail parameter.

    Parameters
    ----------
    raw : Any
        Unconstrained real scalar or array of curvature offsets.

    Returns
    -------
    positive_margin : Any
        ``log(1 + exp(raw))`` as a strictly positive float64 value.

    Notes
    -----
    The function casts ``raw`` to float64 and evaluates
    ``jnp.logaddexp(raw, 0.0)``, which avoids overflow for large
    arguments. The caller adds the result to ``alpha**2 / 4`` so the
    power2 tail denominator stays free of real roots.
    """
    return jnp.logaddexp(jnp.asarray(raw, dtype=jnp.float64), 0.0)


def construct_power2_tail_spec(
    dynamic_imag_left: Any,
    dynamic_slope_left: Any,
    dynamic_imag_right: Any,
    dynamic_slope_right: Any,
    raw_delta_beta_left: Any,
    raw_delta_beta_right: Any,
) -> Power2TailSpec:
    """Construct frozen C1 tail parameters from dynamic-remainder edge data."""
    raw_left = jnp.asarray(raw_delta_beta_left, dtype=jnp.float64)
    raw_right = jnp.asarray(raw_delta_beta_right, dtype=jnp.float64)
    for name, raw in (("left", raw_left), ("right", raw_right)):
        if not isinstance(raw, core.Tracer):
            raw_value = np.asarray(raw)
            if np.any(~np.isfinite(raw_value)) or np.any(
                (raw_value < -30.0) | (raw_value > 30.0)
            ):
                raise ValueError(
                    f"raw_delta_beta_{name} must be finite and in [-30, 30]"
                )

    amplitude_left = -jnp.asarray(dynamic_imag_left, dtype=jnp.float64)
    amplitude_right = -jnp.asarray(dynamic_imag_right, dtype=jnp.float64)
    if not isinstance(amplitude_left, core.Tracer) and np.any(
        np.asarray(amplitude_left) <= 0.0
    ):
        raise ValueError(
            "left dynamic-remainder edge amplitude must be strictly positive"
        )
    if not isinstance(amplitude_right, core.Tracer) and np.any(
        np.asarray(amplitude_right) <= 0.0
    ):
        raise ValueError(
            "right dynamic-remainder edge amplitude must be strictly positive"
        )

    alpha_left = (
        -jnp.asarray(dynamic_slope_left, dtype=jnp.float64) / amplitude_left
    )
    alpha_right = (
        jnp.asarray(dynamic_slope_right, dtype=jnp.float64) / amplitude_right
    )
    beta_left = alpha_left**2 / 4.0 + _softplus(raw_left)
    beta_right = alpha_right**2 / 4.0 + _softplus(raw_right)
    return Power2TailSpec(
        amplitude_left,
        alpha_left,
        beta_left,
        amplitude_right,
        alpha_right,
        beta_right,
    )


def construct_wigner_zero_tail(
    dynamic_imag_left: Any,
    dynamic_imag_right: Any,
) -> ZeroTailSpec:
    """Construct the Wigner-only zero tail, rejecting nonzero edge amplitudes."""
    left = np.asarray(dynamic_imag_left, dtype=np.float64)
    right = np.asarray(dynamic_imag_right, dtype=np.float64)
    if np.any(left != 0.0) or np.any(right != 0.0):
        raise ValueError(
            "Wigner zero-tail contract requires exactly zero edge values"
        )
    return ZeroTailSpec()


def semi_infinite_tail_contribution(
    model_domain_ev: Any,
    tail_spec: Power2TailSpec | ZeroTailSpec,
    queries_ev: Any,
    n_tail: int = 256,
) -> Any:
    """Return both signed semi-infinite tail contributions to unsubtracted Sigma'."""
    queries = jnp.asarray(queries_ev, dtype=jnp.float64)
    if isinstance(tail_spec, ZeroTailSpec):
        return jnp.zeros_like(queries)
    if n_tail <= 0:
        raise ValueError("n_tail must be positive")
    domain = jnp.asarray(model_domain_ev, dtype=jnp.float64)
    if domain.shape != (2,):
        raise ValueError("model_domain_ev must have shape (2,)")

    nodes, weights = np.polynomial.legendre.leggauss(n_tail)
    u = jnp.asarray((nodes + 1.0) / 2.0, dtype=jnp.float64)
    weights_u = jnp.asarray(weights / 2.0, dtype=jnp.float64)

    def side(
        amplitude: Any, alpha: Any, beta: Any, edge: Any, sign: float
    ) -> Any:
        scale = beta**-0.5
        one_minus_u = 1.0 - u
        distance = scale * u / one_minus_u
        jacobian = scale / one_minus_u**2
        sigma_imag = -amplitude / (1.0 + alpha * distance + beta * distance**2)
        denominator = edge + sign * distance - queries[..., None]
        integrand = sigma_imag * jacobian / (jnp.pi * denominator)
        return jnp.sum(weights_u * integrand, axis=-1)

    left = side(
        tail_spec.amplitude_left,
        tail_spec.alpha_left,
        tail_spec.beta_left,
        domain[0],
        -1.0,
    )
    right = side(
        tail_spec.amplitude_right,
        tail_spec.alpha_right,
        tail_spec.beta_right,
        domain[1],
        1.0,
    )
    return left + right


def _jsonable(value: Any) -> Any:
    """PRIVATE: Convert one nested value into JSON-serializable builtins.

    Parameters
    ----------
    value : Any
        Mapping, sequence, NumPy or JAX array, NumPy scalar, or plain
        Python value.

    Returns
    -------
    converted : Any
        The same structure with string keys, lists, and Python scalars.

    Implementation Logic
    --------------------
    The function recurses through mappings and list-like sequences. It
    converts arrays with ``np.asarray(...).tolist()``, unwraps NumPy
    scalars with ``.item()``, and returns every other value unchanged.
    """
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (np.ndarray, jax.Array)):
        return np.asarray(value).tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def write_sweep_report(
    results: Sequence[Mapping[str, Any]],
    json_path: str | Path,
    markdown_path: str | Path,
) -> None:
    """Write sweep rows as lossless JSON and a readable Markdown table."""
    rows = [_jsonable(dict(row)) for row in results]
    json_target = Path(json_path)
    markdown_target = Path(markdown_path)
    json_target.write_text(
        json.dumps(rows, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    columns = list(dict.fromkeys(key for row in rows for key in row))
    lines = ["# KK preproduction candidate sweep", ""]
    if not columns:
        lines.append("No sweep rows were recorded.")
    else:
        lines.extend(
            [
                "| " + " | ".join(columns) + " |",
                "| " + " | ".join("---" for _ in columns) + " |",
            ]
        )
        for row in rows:
            cells = []
            for column in columns:
                value = row.get(column, "")
                rendered = (
                    json.dumps(value, sort_keys=True)
                    if isinstance(value, (dict, list))
                    else str(value)
                )
                cells.append(rendered.replace("|", "\\|"))
            lines.append("| " + " | ".join(cells) + " |")
    markdown_target.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _run_self_tests() -> None:
    """PRIVATE: Run the module self-checks against SciPy quadrature.

    Raises
    ------
    AssertionError
        If a seam value, a seam derivative, or the tail integral moves
        outside its pinned tolerance.

    Implementation Logic
    --------------------
    The check builds one power2 tail spec from symmetric edge data and
    asserts the C1 seam values and derivatives match the inputs. It then
    compares ``semi_infinite_tail_contribution`` at one query against
    adaptive ``scipy.integrate.quad`` on both semi-infinite sides and
    prints the observed errors.
    """
    from scipy.integrate import quad

    amplitude = 0.7
    slope_left = -0.21
    slope_right = 0.14
    raw = -0.4
    spec = construct_power2_tail_spec(
        -amplitude, slope_left, -amplitude, slope_right, raw, raw
    )

    left_value = -spec.amplitude_left
    right_value = -spec.amplitude_right
    left_derivative = -spec.amplitude_left * spec.alpha_left
    right_derivative = spec.amplitude_right * spec.alpha_right
    np.testing.assert_allclose(left_value, -amplitude, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(right_value, -amplitude, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(
        left_derivative, slope_left, rtol=1e-15, atol=1e-15
    )
    np.testing.assert_allclose(
        right_derivative, slope_right, rtol=1e-15, atol=1e-15
    )

    domain = np.array([-2.0, 3.0], dtype=np.float64)
    query = 0.25

    def reference_side(
        edge: float, sign: float, a: float, alpha: float, beta: float
    ) -> float:
        def integrand(distance: float) -> float:
            sigma_imag = -a / (1.0 + alpha * distance + beta * distance**2)
            return sigma_imag / (np.pi * (edge + sign * distance - query))

        return quad(
            integrand, 0.0, np.inf, epsabs=2e-14, epsrel=2e-14, limit=500
        )[0]

    reference = reference_side(domain[0], -1.0, *spec[:3]) + reference_side(
        domain[1], 1.0, *spec[3:]
    )
    observed = float(
        semi_infinite_tail_contribution(
            domain, spec, np.array(query), n_tail=256
        )
    )
    error = abs(observed - reference)
    np.testing.assert_allclose(observed, reference, rtol=2e-13, atol=2e-14)
    print(
        f"C1 seam max error: {max(abs(float(left_derivative) - slope_left), abs(float(right_derivative) - slope_right)):.3e}"
    )
    print(
        f"tail integral: observed={observed:.17e}, reference={reference:.17e}, abs_error={error:.3e}"
    )


if __name__ == "__main__":
    _run_self_tests()
