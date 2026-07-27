"""Evaluate spherical Bessel functions with stable JAX primitives.

Extended Summary
----------------
The implementation combines a three-term origin series with two recurrences.
It uses upward recurrence below the order and downward Miller recurrence
elsewhere. Miller normalization selects the
better-conditioned of the analytic :math:`j_0` and :math:`j_1` anchors.

Routine Listings
----------------
:func:`spherical_bessel_jl`
    Evaluate the spherical Bessel function :math:`j_l(x)`.
:func:`spherical_bessel_jl_derivative`
    Evaluate :math:`d j_l(x)/dx`.
"""

import math

import jax
import jax.numpy as jnp
from beartype import beartype
from jaxtyping import Array, Float, Integer, jaxtyped


def _odd_double_factorial(order: int) -> float:
    """Return an odd positive double factorial as a float."""
    if order < 1 or order % 2 == 0:
        message: str = "order must be a positive odd integer"
        raise ValueError(message)
    result: float = float(math.prod(range(1, order + 1, 2)))
    return result


def _origin_series(
    order: int,
    x: Float[Array, " ..."],
) -> Float[Array, " ..."]:
    """Evaluate the first three nonzero terms of the origin series."""
    denominator: float = _odd_double_factorial(2 * order + 1)
    second_denominator: float = 2.0 * (2 * order + 3)
    fourth_denominator: float = 8.0 * (2 * order + 3) * (2 * order + 5)
    correction: Float[Array, " ..."] = (
        1.0 - x * x / second_denominator + x**4 / fourth_denominator
    )
    values: Float[Array, " ..."] = (x**order / denominator) * correction
    return values


def _origin_series_derivative(
    order: int,
    x: Float[Array, " ..."],
) -> Float[Array, " ..."]:
    """Differentiate the three-term origin series analytically."""
    denominator: float = _odd_double_factorial(2 * order + 1)
    second_denominator: float = 2.0 * (2 * order + 3)
    fourth_denominator: float = 8.0 * (2 * order + 3) * (2 * order + 5)
    correction: Float[Array, " ..."] = (
        1.0 - x * x / second_denominator + x**4 / fourth_denominator
    )
    correction_derivative: Float[Array, " ..."] = (
        -2.0 * x / second_denominator + 4.0 * x**3 / fourth_denominator
    )
    derivatives: Float[Array, " ..."]
    if order == 0:
        derivatives = correction_derivative / denominator
    else:
        derivatives = (
            order * x ** (order - 1) * correction
            + x**order * correction_derivative
        ) / denominator
    return derivatives


def _analytic_anchors(
    x: Float[Array, " ..."],
) -> tuple[Float[Array, " ..."], Float[Array, " ..."]]:
    """Evaluate analytic nonzero-argument anchors j0 and j1."""
    j0: Float[Array, " ..."] = jnp.sin(x) / x
    j1: Float[Array, " ..."] = jnp.sin(x) / (x * x) - jnp.cos(x) / x
    anchors: tuple[Float[Array, " ..."], Float[Array, " ..."]] = (j0, j1)
    return anchors


def _upward_recurrence(
    order: int,
    x: Float[Array, " ..."],
    j0: Float[Array, " ..."],
    j1: Float[Array, " ..."],
) -> Float[Array, " ..."]:
    """Evaluate j_l by upward recurrence."""

    def _step(
        index: Integer[Array, ""],
        state: tuple[Float[Array, " ..."], Float[Array, " ..."]],
    ) -> tuple[Float[Array, " ..."], Float[Array, " ..."]]:
        previous: Float[Array, " ..."] = state[0]
        current: Float[Array, " ..."] = state[1]
        index_float: Float[Array, ""] = jnp.asarray(index, dtype=jnp.float64)
        following: Float[Array, " ..."] = (
            2.0 * index_float + 1.0
        ) * current / x - previous
        next_state: tuple[Float[Array, " ..."], Float[Array, " ..."]] = (
            current,
            following,
        )
        return next_state

    final_state: tuple[Float[Array, " ..."], Float[Array, " ..."]] = (
        jax.lax.fori_loop(1, order, _step, (j0, j1))
    )
    values: Float[Array, " ..."] = final_state[1]
    return values


def _downward_miller(
    order: int,
    x: Float[Array, " ..."],
    j0: Float[Array, " ..."],
    j1: Float[Array, " ..."],
) -> Float[Array, " ..."]:
    """Evaluate j_l by fixed-depth downward Miller recurrence."""
    start_order: int = order + math.ceil(math.sqrt(40.0 * max(order, 1))) + 12
    target_seed: Float[Array, " ..."] = jnp.ones_like(x)

    def _step(
        iteration: Integer[Array, ""],
        state: tuple[
            Float[Array, " ..."],
            Float[Array, " ..."],
            Float[Array, " ..."],
        ],
    ) -> tuple[
        Float[Array, " ..."],
        Float[Array, " ..."],
        Float[Array, " ..."],
    ]:
        following: Float[Array, " ..."] = state[0]
        current: Float[Array, " ..."] = state[1]
        target: Float[Array, " ..."] = state[2]
        ell: Integer[Array, ""] = start_order - iteration
        ell_float: Float[Array, ""] = jnp.asarray(ell, dtype=jnp.float64)
        previous: Float[Array, " ..."] = (
            2.0 * ell_float + 1.0
        ) * current / x - following
        target = jnp.where(ell - 1 == order, previous, target)
        next_state: tuple[
            Float[Array, " ..."],
            Float[Array, " ..."],
            Float[Array, " ..."],
        ] = (current, previous, target)
        return next_state

    initial_state: tuple[
        Float[Array, " ..."],
        Float[Array, " ..."],
        Float[Array, " ..."],
    ] = (jnp.zeros_like(x), jnp.ones_like(x), target_seed)
    miller_state: tuple[
        Float[Array, " ..."],
        Float[Array, " ..."],
        Float[Array, " ..."],
    ] = jax.lax.fori_loop(
        0,
        start_order,
        _step,
        initial_state,
    )
    raw_j1: Float[Array, " ..."] = miller_state[0]
    raw_j0: Float[Array, " ..."] = miller_state[1]
    raw_target: Float[Array, " ..."] = miller_state[2]
    use_j0: Float[Array, " ..."] = jnp.abs(j0) >= jnp.abs(j1)
    safe_raw_j0: Float[Array, " ..."] = jnp.where(
        use_j0, raw_j0, jnp.ones_like(raw_j0)
    )
    safe_raw_j1: Float[Array, " ..."] = jnp.where(
        use_j0, jnp.ones_like(raw_j1), raw_j1
    )
    scale_from_j0: Float[Array, " ..."] = j0 / safe_raw_j0
    scale_from_j1: Float[Array, " ..."] = j1 / safe_raw_j1
    scale: Float[Array, " ..."] = jnp.where(
        use_j0, scale_from_j0, scale_from_j1
    )
    values: Float[Array, " ..."] = raw_target * scale
    return values


@jaxtyped(typechecker=beartype)
def spherical_bessel_jl(
    order: int,
    x: Float[Array, " ..."],
) -> Float[Array, " ..."]:
    """Evaluate the spherical Bessel function :math:`j_l(x)`.

    The kernel selects an origin series or a stable recurrence per argument.

    :see: :class:`~.test_bessel.TestSphericalBesselJl`

    Parameters
    ----------
    order : int
        Nonnegative static angular-momentum order.
    x : Float[Array, " ..."]
        Real arguments.

    Returns
    -------
    values : Float[Array, " ..."]
        Spherical Bessel values with the input shape.

    Raises
    ------
    ValueError
        If ``order`` is negative.

    Notes
    -----
    Values with ``abs(x) < 0.02`` use the three-term origin series.  Other
    values use upward recurrence for ``order <= abs(x)`` and downward Miller
    recurrence otherwise.  Sanitized branch inputs prevent inactive singular
    formulas from contaminating derivatives.
    """
    if order < 0:
        message: str = "order must be non-negative"
        raise ValueError(message)

    x_array: Float[Array, " ..."] = jnp.asarray(x, dtype=jnp.float64)
    series_threshold: float = 2.0e-2
    use_series: Float[Array, " ..."] = jnp.abs(x_array) < series_threshold
    safe_x: Float[Array, " ..."] = jnp.where(use_series, 1.0, x_array)
    series_values: Float[Array, " ..."] = _origin_series(order, x_array)
    anchors: tuple[Float[Array, " ..."], Float[Array, " ..."]] = (
        _analytic_anchors(safe_x)
    )
    j0: Float[Array, " ..."] = anchors[0]
    j1: Float[Array, " ..."] = anchors[1]

    recurrence_values: Float[Array, " ..."]
    if order == 0:
        recurrence_values = j0
    elif order == 1:
        recurrence_values = j1
    else:
        upward_values: Float[Array, " ..."] = _upward_recurrence(
            order, safe_x, j0, j1
        )
        downward_values: Float[Array, " ..."] = _downward_miller(
            order, safe_x, j0, j1
        )
        recurrence_values = jnp.where(
            order <= jnp.abs(safe_x),
            upward_values,
            downward_values,
        )
    values: Float[Array, " ..."] = jnp.where(
        use_series,
        series_values,
        recurrence_values,
    )
    return values


@jaxtyped(typechecker=beartype)
def spherical_bessel_jl_derivative(
    order: int,
    x: Float[Array, " ..."],
) -> Float[Array, " ..."]:
    """Evaluate :math:`d j_l(x)/dx`.

    The kernel differentiates the origin series and uses stable identities
    elsewhere.

    :see: :class:`~.test_bessel.TestSphericalBesselJlDerivative`

    Parameters
    ----------
    order : int
        Nonnegative static angular-momentum order.
    x : Float[Array, " ..."]
        Real arguments.

    Returns
    -------
    derivatives : Float[Array, " ..."]
        First argument derivatives with the input shape.

    Raises
    ------
    ValueError
        If ``order`` is negative.

    Notes
    -----
    The origin branch differentiates the same three-term series as the
    primal.  Away from the origin, order zero uses ``-j_1`` and higher orders
    use ``j_(l-1) - (l+1) j_l / x``.
    """
    if order < 0:
        message: str = "order must be non-negative"
        raise ValueError(message)

    x_array: Float[Array, " ..."] = jnp.asarray(x, dtype=jnp.float64)
    series_threshold: float = 2.0e-2
    use_series: Float[Array, " ..."] = jnp.abs(x_array) < series_threshold
    safe_x: Float[Array, " ..."] = jnp.where(use_series, 1.0, x_array)
    series_derivatives: Float[Array, " ..."] = _origin_series_derivative(
        order, x_array
    )
    if order == 0:
        recurrence_derivatives: Float[Array, " ..."] = -spherical_bessel_jl(
            1, safe_x
        )
    else:
        recurrence_derivatives = spherical_bessel_jl(order - 1, safe_x) - (
            (order + 1.0) * spherical_bessel_jl(order, safe_x) / safe_x
        )
    derivatives: Float[Array, " ..."] = jnp.where(
        use_series,
        series_derivatives,
        recurrence_derivatives,
    )
    return derivatives


__all__: list[str] = [
    "spherical_bessel_jl",
    "spherical_bessel_jl_derivative",
]
