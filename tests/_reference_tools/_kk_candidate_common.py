"""Provide shared numerical tools for the KK operator comparison.

Extended Summary
----------------
This module contains analytic fixtures and verifies the committed reference
archive independently. It also implements the fixed semi-infinite tail
quadrature for the operator comparison.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from beartype.typing import (
    Any,
    Dict,
    List,
    Mapping,
    NamedTuple,
    Sequence,
    Set,
    Tuple,
)
from jax import core
from jaxtyping import Array, ArrayLike, Float, Float64
from numpy.typing import NDArray

jax.config.update("jax_enable_x64", True)


REFERENCE_DIRECTORY = (
    Path(__file__).parents[1]
    / "test_diffpes"
    / "_reference_data"
    / "kk_analytic_reference"
)
RAW_CURVATURE_BOUND: float = 30.0


class Power2TailSpec(NamedTuple):
    """Store scalar C1-matched power2 parameters in left-to-right order."""

    amplitude_left: Float64[Array, ""]
    alpha_left: Float64[Array, ""]
    beta_left: Float64[Array, ""]
    amplitude_right: Float64[Array, ""]
    alpha_right: Float64[Array, ""]
    beta_right: Float64[Array, ""]


class ZeroTailSpec(NamedTuple):
    """Store the explicit compact-support tail contract."""

    kind: str = "zero"


def retarded_pole_fixture(
    points_ev: Float[ArrayLike, "..."],
    subtraction_point_ev: Float[ArrayLike, "..."] = 0.0,
) -> Tuple[Float64[Array, "..."], Float64[Array, "..."]]:
    """Return the analytic retarded-pole self-energy pair.

    Parameters
    ----------
    points_ev : Float[ArrayLike, "..."]
        Evaluation energies in eV.
    subtraction_point_ev : Float[ArrayLike, "..."]
        Subtraction energy in eV. Default 0.0.

    Returns
    -------
    result : Tuple[Float64[Array, "..."], Float64[Array, "..."]]
        The imaginary part and the subtracted real part in eV.

    Notes
    -----
    The function evaluates the retarded pole with the fixed center, width,
    and coupling values that define the analytic reference.
    """
    points: Float64[Array, "..."] = jnp.asarray(points_ev, dtype=jnp.float64)
    subtraction_point: Float64[Array, "..."] = jnp.asarray(
        subtraction_point_ev, dtype=jnp.float64
    )
    omega_0: Float64[Array, ""] = jnp.float64(0.35)
    gamma: Float64[Array, ""] = jnp.float64(0.20)
    coupling: Float64[Array, ""] = jnp.float64(0.12)

    def real(omega: Float64[Array, "..."]) -> Float64[Array, "..."]:
        """Evaluate the real part of the retarded pole.

        Parameters
        ----------
        omega : Float64[Array, "..."]
            Evaluation energies in eV.

        Returns
        -------
        real_part : Float64[Array, "..."]
            Real self-energy values in eV.

        Notes
        -----
        The function evaluates the rational closed form with the fixed pole
        parameters from the enclosing fixture.
        """
        offset: Float64[Array, "..."] = omega - omega_0
        real_part: Float64[Array, "..."] = (
            coupling * offset / (offset * offset + gamma * gamma)
        )
        return real_part

    offset: Float64[Array, "..."] = points - omega_0
    denominator: Float64[Array, "..."] = offset * offset + gamma * gamma
    sigma_imag: Float64[Array, "..."] = -coupling * gamma / denominator
    sigma_real: Float64[Array, "..."] = real(points) - real(subtraction_point)
    result: Tuple[Float64[Array, "..."], Float64[Array, "..."]] = (
        sigma_imag,
        sigma_real,
    )
    return result


def wigner_semicircle_fixture(
    points_ev: Float[ArrayLike, "..."],
    subtraction_point_ev: Float[ArrayLike, "..."] = 0.0,
) -> Tuple[Float64[Array, "..."], Float64[Array, "..."]]:
    """Return the analytic Wigner-semicircle self-energy pair.

    Parameters
    ----------
    points_ev : Float[ArrayLike, "..."]
        Evaluation energies in eV.
    subtraction_point_ev : Float[ArrayLike, "..."]
        Subtraction energy in eV. Default 0.0.

    Returns
    -------
    result : Tuple[Float64[Array, "..."], Float64[Array, "..."]]
        The imaginary part and the subtracted real part in eV.

    Notes
    -----
    The function evaluates the in-band Wigner closed form with fixed model
    parameters. Values outside the band have a zero imaginary part.
    """
    points: Float64[Array, "..."] = jnp.asarray(points_ev, dtype=jnp.float64)
    subtraction_point: Float64[Array, "..."] = jnp.asarray(
        subtraction_point_ev, dtype=jnp.float64
    )
    half_width: Float64[Array, ""] = jnp.float64(1.50)
    coupling: Float64[Array, ""] = jnp.float64(0.20)
    scale: Float64[Array, ""] = 2.0 * coupling / half_width**2
    radicand: Float64[Array, "..."] = jnp.maximum(
        half_width**2 - points**2, 0.0
    )
    sigma_imag: Float64[Array, "..."] = jnp.where(
        jnp.abs(points) < half_width, -scale * jnp.sqrt(radicand), 0.0
    )

    def real(omega: Float64[Array, "..."]) -> Float64[Array, "..."]:
        """Evaluate the in-band Wigner real part.

        Parameters
        ----------
        omega : Float64[Array, "..."]
            Evaluation energies in eV.

        Returns
        -------
        real_part : Float64[Array, "..."]
            Real self-energy values in eV.

        Notes
        -----
        The trusted query interval stays inside the semicircle band. The
        in-band real part is linear in energy.
        """
        real_part: Float64[Array, "..."] = scale * omega
        return real_part

    sigma_real: Float64[Array, "..."] = real(points) - real(subtraction_point)
    result: Tuple[Float64[Array, "..."], Float64[Array, "..."]] = (
        sigma_imag,
        sigma_real,
    )
    return result


def load_analytic_reference(
    reference_directory: str | Path = REFERENCE_DIRECTORY,
) -> Dict[str, Float64[NDArray, "..."]]:
    """Load and verify the committed analytic reference.

    Parameters
    ----------
    reference_directory : str | Path
        Directory that contains the reference archive and manifest.

    Returns
    -------
    arrays : Dict[str, Float64[NDArray, "..."]]
        Verified float64 reference arrays, keyed by manifest name.

    Raises
    ------
    RuntimeError
        If the manifest, digest, inventory, dtype, or shape is invalid.

    Notes
    -----
    The function reads the manifest and verifies the archive SHA-256 digest.
    It then checks every array name, shape, and dtype before it copies data.
    """
    directory: Path = Path(reference_directory)
    manifest_path: Path = directory / "manifest.json"
    error: OSError | KeyError | json.JSONDecodeError
    try:
        manifest: Dict[str, Any] = json.loads(
            manifest_path.read_text(encoding="utf-8")
        )
        archive_path: Path = directory / manifest["archive"]
        expected_digest: str = manifest["archive_sha256"]
    except (OSError, KeyError, json.JSONDecodeError) as error:
        raise RuntimeError(
            f"Invalid KK reference manifest: {manifest_path}"
        ) from error

    actual_digest: str = hashlib.sha256(archive_path.read_bytes()).hexdigest()
    if actual_digest != expected_digest:
        raise RuntimeError(
            "KK reference archive SHA-256 mismatch: "
            f"expected {expected_digest}, observed {actual_digest} "
            f"for {archive_path}"
        )

    expected_arrays: Any = manifest.get("arrays")
    if not isinstance(expected_arrays, dict):
        raise RuntimeError("KK reference manifest has no array inventory")
    archive: Any
    arrays: Dict[str, Float64[NDArray, "..."]] = {}
    with np.load(archive_path, allow_pickle=False) as archive:
        observed_names: Set[str] = set(archive.files)
        expected_names: Set[str] = set(expected_arrays)
        if observed_names != expected_names:
            raise RuntimeError(
                "KK reference array inventory mismatch: "
                f"{observed_names} != {expected_names}"
            )
        name: str
        shape: List[int]
        for name, shape in expected_arrays.items():
            value: Float64[NDArray, "..."] = np.asarray(archive[name])
            if value.dtype != np.float64 or list(value.shape) != shape:
                raise RuntimeError(
                    f"KK reference array {name!r} has dtype/shape "
                    f"{value.dtype}/{value.shape}, expected "
                    f"float64/{tuple(shape)}"
                )
            arrays[name] = value.copy()
    return arrays


def _softplus(raw: Float[ArrayLike, "..."]) -> Float64[Array, "..."]:
    """PRIVATE: Compute the softplus of one raw tail parameter.

    Parameters
    ----------
    raw : Float[ArrayLike, "..."]
        Unconstrained real scalar or array of curvature offsets.

    Returns
    -------
    positive_margin : Float64[Array, "..."]
        ``log(1 + exp(raw))`` as a strictly positive float64 value.

    Notes
    -----
    The function casts ``raw`` to float64 and evaluates
    ``jnp.logaddexp(raw, 0.0)``, which avoids overflow for large
    arguments. The caller adds the result to ``alpha**2 / 4`` so the
    power2 tail denominator stays free of real roots.
    """
    positive_margin: Float64[Array, "..."] = jnp.logaddexp(
        jnp.asarray(raw, dtype=jnp.float64), 0.0
    )
    return positive_margin


def construct_power2_tail_spec(
    dynamic_imag_left: Float[ArrayLike, ""],
    dynamic_slope_left: Float[ArrayLike, ""],
    dynamic_imag_right: Float[ArrayLike, ""],
    dynamic_slope_right: Float[ArrayLike, ""],
    raw_delta_beta_left: Float[ArrayLike, ""],
    raw_delta_beta_right: Float[ArrayLike, ""],
) -> Power2TailSpec:
    """Construct C1 power2 tails from dynamic edge data.

    Parameters
    ----------
    dynamic_imag_left : Float[ArrayLike, ""]
        Dynamic imaginary part at the left edge in eV.
    dynamic_slope_left : Float[ArrayLike, ""]
        Left-edge energy derivative.
    dynamic_imag_right : Float[ArrayLike, ""]
        Dynamic imaginary part at the right edge in eV.
    dynamic_slope_right : Float[ArrayLike, ""]
        Right-edge energy derivative.
    raw_delta_beta_left : Float[ArrayLike, ""]
        Raw positive-curvature coordinate for the left tail.
    raw_delta_beta_right : Float[ArrayLike, ""]
        Raw positive-curvature coordinate for the right tail.

    Returns
    -------
    tail_spec : Power2TailSpec
        C1-matched parameters in left-to-right order.

    Raises
    ------
    ValueError
        If a raw curvature coordinate is invalid or an amplitude is not
        strictly positive.

    Notes
    -----
    The function derives each amplitude and slope coefficient from the edge
    values. A softplus margin keeps each quadratic denominator positive.
    """
    raw_left: Float64[Array, ""] = jnp.asarray(
        raw_delta_beta_left, dtype=jnp.float64
    )
    raw_right: Float64[Array, ""] = jnp.asarray(
        raw_delta_beta_right, dtype=jnp.float64
    )
    name: str
    raw: Float64[Array, ""]
    for name, raw in (("left", raw_left), ("right", raw_right)):
        if not isinstance(raw, core.Tracer):
            raw_value: Float64[NDArray, ""] = np.asarray(raw)
            if np.any(~np.isfinite(raw_value)) or np.any(
                (raw_value < -RAW_CURVATURE_BOUND)
                | (raw_value > RAW_CURVATURE_BOUND)
            ):
                raise ValueError(
                    f"raw_delta_beta_{name} must be finite and in [-30, 30]"
                )

    amplitude_left: Float64[Array, ""] = -jnp.asarray(
        dynamic_imag_left, dtype=jnp.float64
    )
    amplitude_right: Float64[Array, ""] = -jnp.asarray(
        dynamic_imag_right, dtype=jnp.float64
    )
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

    alpha_left: Float64[Array, ""] = (
        -jnp.asarray(dynamic_slope_left, dtype=jnp.float64) / amplitude_left
    )
    alpha_right: Float64[Array, ""] = (
        jnp.asarray(dynamic_slope_right, dtype=jnp.float64) / amplitude_right
    )
    beta_left: Float64[Array, ""] = alpha_left**2 / 4.0 + _softplus(raw_left)
    beta_right: Float64[Array, ""] = alpha_right**2 / 4.0 + _softplus(
        raw_right
    )
    tail_spec: Power2TailSpec = Power2TailSpec(
        amplitude_left,
        alpha_left,
        beta_left,
        amplitude_right,
        alpha_right,
        beta_right,
    )
    return tail_spec


def construct_wigner_zero_tail(
    dynamic_imag_left: Float[ArrayLike, "..."],
    dynamic_imag_right: Float[ArrayLike, "..."],
) -> ZeroTailSpec:
    """Construct the compact Wigner tail contract.

    Parameters
    ----------
    dynamic_imag_left : Float[ArrayLike, "..."]
        Dynamic imaginary part at the left band edge in eV.
    dynamic_imag_right : Float[ArrayLike, "..."]
        Dynamic imaginary part at the right band edge in eV.

    Returns
    -------
    tail_spec : ZeroTailSpec
        Explicit zero-tail contract.

    Raises
    ------
    ValueError
        If either edge value is not exactly zero.

    Notes
    -----
    Compact support requires a zero continuation beyond both band edges.
    """
    left: Float64[NDArray, "..."] = np.asarray(
        dynamic_imag_left, dtype=np.float64
    )
    right: Float64[NDArray, "..."] = np.asarray(
        dynamic_imag_right, dtype=np.float64
    )
    if np.any(left != 0.0) or np.any(right != 0.0):
        raise ValueError(
            "Wigner zero-tail contract requires exactly zero edge values"
        )
    tail_spec: ZeroTailSpec = ZeroTailSpec()
    return tail_spec


def semi_infinite_tail_contribution(
    model_domain_ev: Float[ArrayLike, " 2"],
    tail_spec: Power2TailSpec | ZeroTailSpec,
    queries_ev: Float[ArrayLike, "..."],
    n_tail: int = 256,
) -> Float64[Array, "..."]:
    """Return both semi-infinite contributions to the unsubtracted real part.

    Parameters
    ----------
    model_domain_ev : Float[ArrayLike, " 2"]
        Left and right boundaries of the finite model domain in eV.
    tail_spec : Power2TailSpec | ZeroTailSpec
        Tail parameters or the compact-support contract.
    queries_ev : Float[ArrayLike, "..."]
        Query energies in eV.
    n_tail : int
        Number of Gauss-Legendre nodes on each transformed tail. Default 256.

    Returns
    -------
    contribution_ev : Float64[Array, "..."]
        Sum of the left and right tail contributions in eV.

    Raises
    ------
    ValueError
        If ``n_tail`` is not positive or the domain does not have two edges.

    Notes
    -----
    The function maps each semi-infinite interval to the unit interval. It
    applies fixed Gauss-Legendre quadrature and sums both signed tails.
    """
    queries: Float64[Array, "..."] = jnp.asarray(queries_ev, dtype=jnp.float64)
    if isinstance(tail_spec, ZeroTailSpec):
        contribution_ev: Float64[Array, "..."] = jnp.zeros_like(queries)
        return contribution_ev
    if n_tail <= 0:
        raise ValueError("n_tail must be positive")
    domain: Float64[Array, " 2"] = jnp.asarray(
        model_domain_ev, dtype=jnp.float64
    )
    if domain.shape != (2,):
        raise ValueError("model_domain_ev must have shape (2,)")

    nodes: Float64[NDArray, " n_tail"]
    weights: Float64[NDArray, " n_tail"]
    nodes, weights = np.polynomial.legendre.leggauss(n_tail)
    u: Float64[Array, " n_tail"] = jnp.asarray(
        (nodes + 1.0) / 2.0, dtype=jnp.float64
    )
    weights_u: Float64[Array, " n_tail"] = jnp.asarray(
        weights / 2.0, dtype=jnp.float64
    )

    def side(
        amplitude: Float64[Array, "..."],
        alpha: Float64[Array, "..."],
        beta: Float64[Array, "..."],
        edge: Float64[Array, ""],
        sign: float,
    ) -> Float64[Array, "..."]:
        """Integrate one transformed semi-infinite tail.

        Parameters
        ----------
        amplitude : Float64[Array, "..."]
            Positive edge amplitude in eV.
        alpha : Float64[Array, "..."]
            Linear denominator coefficient in 1/eV.
        beta : Float64[Array, "..."]
            Quadratic denominator coefficient in 1/eV^2.
        edge : Float64[Array, ""]
            Finite-domain edge energy in eV.
        sign : float
            Direction from the domain edge. Use -1.0 left and 1.0 right.

        Returns
        -------
        contribution_ev : Float64[Array, "..."]
            Tail contribution at every query in eV.

        Notes
        -----
        The rational map sends the unit interval to a semi-infinite distance.
        The fixed nodes make the result deterministic and differentiable.
        """
        scale: Float64[Array, "..."] = beta**-0.5
        one_minus_u: Float64[Array, " n_tail"] = 1.0 - u
        distance: Float64[Array, "... n_tail"] = scale * u / one_minus_u
        jacobian: Float64[Array, "... n_tail"] = scale / one_minus_u**2
        sigma_imag: Float64[Array, "... n_tail"] = -amplitude / (
            1.0 + alpha * distance + beta * distance**2
        )
        denominator: Float64[Array, "... n_tail"] = (
            edge + sign * distance - queries[..., None]
        )
        integrand: Float64[Array, "... n_tail"] = (
            sigma_imag * jacobian / (jnp.pi * denominator)
        )
        contribution_ev: Float64[Array, "..."] = jnp.sum(
            weights_u * integrand, axis=-1
        )
        return contribution_ev

    left: Float64[Array, "..."] = side(
        tail_spec.amplitude_left,
        tail_spec.alpha_left,
        tail_spec.beta_left,
        domain[0],
        -1.0,
    )
    right: Float64[Array, "..."] = side(
        tail_spec.amplitude_right,
        tail_spec.alpha_right,
        tail_spec.beta_right,
        domain[1],
        1.0,
    )
    contribution_ev = left + right
    return contribution_ev


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
        converted: Any = {
            str(key): _jsonable(item) for key, item in value.items()
        }
        return converted
    if isinstance(value, (list, tuple)):
        converted = [_jsonable(item) for item in value]
        return converted
    if isinstance(value, (np.ndarray, jax.Array)):
        converted = np.asarray(value).tolist()
        return converted
    if isinstance(value, np.generic):
        converted = value.item()
        return converted
    converted = value
    return converted


def write_sweep_report(
    results: Sequence[Mapping[str, Any]],
    json_path: str | Path,
    markdown_path: str | Path,
) -> None:
    """Write comparison rows as JSON and a Markdown table.

    Parameters
    ----------
    results : Sequence[Mapping[str, Any]]
        Comparison rows with JSON-compatible or array-like values.
    json_path : str | Path
        Destination path for the lossless JSON report.
    markdown_path : str | Path
        Destination path for the readable Markdown report.

    Notes
    -----
    The function normalizes nested values, preserves all JSON fields, and
    derives the Markdown columns from their first observed order.
    """
    rows: List[Dict[str, Any]] = [_jsonable(dict(row)) for row in results]
    json_target: Path = Path(json_path)
    markdown_target: Path = Path(markdown_path)
    json_target.write_text(
        json.dumps(rows, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    columns: List[str] = list(
        dict.fromkeys(key for row in rows for key in row)
    )
    lines: List[str] = ["# KK operator comparison", ""]
    if not columns:
        lines.append("No sweep rows were recorded.")
    else:
        lines.extend(
            [
                "| " + " | ".join(columns) + " |",
                "| " + " | ".join("---" for _ in columns) + " |",
            ]
        )
        row: Dict[str, Any]
        for row in rows:
            cells: List[str] = []
            column: str
            for column in columns:
                value: Any = row.get(column, "")
                rendered: str = (
                    json.dumps(value, sort_keys=True)
                    if isinstance(value, (dict, list))
                    else str(value)
                )
                cells.append(rendered.replace("|", "\\|"))
            lines.append("| " + " | ".join(cells) + " |")
    markdown_target.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _run_self_tests() -> None:
    """PRIVATE: Run the module self-checks against SciPy quadrature.

    Implementation Logic
    --------------------
    The check builds one power2 tail spec from symmetric edge data and
    asserts the C1 seam values and derivatives match the inputs. It then
    compares ``semi_infinite_tail_contribution`` at one query against
    adaptive ``scipy.integrate.quad`` on both semi-infinite sides and
    prints the observed errors.
    """
    from scipy.integrate import quad  # noqa: PLC0415 -- optional self-test.

    amplitude: float = 0.7
    slope_left: float = -0.21
    slope_right: float = 0.14
    raw: float = -0.4
    spec: Power2TailSpec = construct_power2_tail_spec(
        -amplitude, slope_left, -amplitude, slope_right, raw, raw
    )

    left_value: Float64[Array, ""] = -spec.amplitude_left
    right_value: Float64[Array, ""] = -spec.amplitude_right
    left_derivative: Float64[Array, ""] = (
        -spec.amplitude_left * spec.alpha_left
    )
    right_derivative: Float64[Array, ""] = (
        spec.amplitude_right * spec.alpha_right
    )
    np.testing.assert_allclose(left_value, -amplitude, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(right_value, -amplitude, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(
        left_derivative, slope_left, rtol=1e-15, atol=1e-15
    )
    np.testing.assert_allclose(
        right_derivative, slope_right, rtol=1e-15, atol=1e-15
    )

    domain: Float64[NDArray, " 2"] = np.array([-2.0, 3.0], dtype=np.float64)
    query: float = 0.25

    def reference_side(
        edge: float, sign: float, a: float, alpha: float, beta: float
    ) -> float:
        """Integrate one tail with adaptive SciPy quadrature.

        Parameters
        ----------
        edge : float
            Finite-domain edge energy in eV.
        sign : float
            Direction from the edge.
        a : float
            Positive tail amplitude in eV.
        alpha : float
            Linear denominator coefficient in 1/eV.
        beta : float
            Quadratic denominator coefficient in 1/eV^2.

        Returns
        -------
        contribution_ev : float
            Adaptive tail contribution in eV.

        Notes
        -----
        SciPy integrates the physical distance from zero to infinity with
        tight absolute and relative tolerances.
        """

        def integrand(distance: float) -> float:
            """Evaluate the tail principal-value integrand.

            Parameters
            ----------
            distance : float
                Nonnegative distance from the finite-domain edge in eV.

            Returns
            -------
            value : float
                Principal-value integrand at the distance.

            Notes
            -----
            The rational tail supplies the imaginary part. The pole
            denominator supplies the Kramers--Kronig kernel.
            """
            sigma_imag: float = -a / (
                1.0 + alpha * distance + beta * distance**2
            )
            value: float = sigma_imag / (
                np.pi * (edge + sign * distance - query)
            )
            return value

        contribution_ev: float = quad(
            integrand, 0.0, np.inf, epsabs=2e-14, epsrel=2e-14, limit=500
        )[0]
        return contribution_ev

    left_parameters: Tuple[float, float, float] = (
        float(spec.amplitude_left),
        float(spec.alpha_left),
        float(spec.beta_left),
    )
    right_parameters: Tuple[float, float, float] = (
        float(spec.amplitude_right),
        float(spec.alpha_right),
        float(spec.beta_right),
    )
    reference: float = reference_side(
        float(domain[0]), -1.0, *left_parameters
    ) + reference_side(float(domain[1]), 1.0, *right_parameters)
    observed: float = float(
        semi_infinite_tail_contribution(
            domain, spec, np.array(query), n_tail=256
        )
    )
    error: float = abs(observed - reference)
    np.testing.assert_allclose(observed, reference, rtol=2e-13, atol=2e-14)
    seam_error: float = max(
        abs(float(left_derivative) - slope_left),
        abs(float(right_derivative) - slope_right),
    )
    print(f"C1 seam max error: {seam_error:.3e}")
    print(
        f"tail integral: observed={observed:.17e}, "
        f"reference={reference:.17e}, abs_error={error:.3e}"
    )


if __name__ == "__main__":
    _run_self_tests()
