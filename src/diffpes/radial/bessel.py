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
from beartype.typing import Tuple
from jaxtyping import Array, Float64, Integer, jaxtyped


def _odd_double_factorial(order: int) -> float:
    """PRIVATE: Return an odd positive double factorial as a float.

    Parameters
    ----------
    order : int
        Odd positive integer argument of the double factorial.

    Returns
    -------
    result : float
        Product of the odd integers from 1 to ``order``.

    Raises
    ------
    ValueError
        If ``order`` is not a positive odd integer.

    Notes
    -----
    Multiplies the odd integers with :func:`math.prod` and converts the
    exact integer product to a float once.
    """
    if order < 1 or order % 2 == 0:
        message: str = "order must be a positive odd integer"
        raise ValueError(message)
    result: float = float(math.prod(range(1, order + 1, 2)))
    return result


def _origin_series(
    order: int,
    x: Float64[Array, " ..."],
) -> Float64[Array, " ..."]:
    """PRIVATE: Evaluate the first three nonzero terms of the origin series.

    Parameters
    ----------
    order : int
        Nonnegative static angular-momentum order.
    x : Float64[Array, " ..."]
        Real dimensionless arguments near the origin.

    Returns
    -------
    values : Float64[Array, " ..."]
        Truncated Maclaurin values of :math:`j_l(x)` with the input
        shape.

    Notes
    -----
    Evaluates ``x**l / (2l+1)!!`` times the correction polynomial
    ``1 - x**2/(2(2l+3)) + x**4/(8(2l+3)(2l+5))``.  The truncation
    error is of relative order ``x**6``.
    """
    denominator: float = _odd_double_factorial(2 * order + 1)
    second_denominator: float = 2.0 * (2 * order + 3)
    fourth_denominator: float = 8.0 * (2 * order + 3) * (2 * order + 5)
    correction: Float64[Array, " ..."] = (
        1.0 - x * x / second_denominator + x**4 / fourth_denominator
    )
    values: Float64[Array, " ..."] = (x**order / denominator) * correction
    return values


def _origin_series_derivative(
    order: int,
    x: Float64[Array, " ..."],
) -> Float64[Array, " ..."]:
    """PRIVATE: Differentiate the three-term origin series analytically.

    Parameters
    ----------
    order : int
        Nonnegative static angular-momentum order.
    x : Float64[Array, " ..."]
        Real dimensionless arguments near the origin.

    Returns
    -------
    derivatives : Float64[Array, " ..."]
        Exact derivative of the truncated series with the input shape.

    Notes
    -----
    Applies the product rule to ``x**l`` times the correction
    polynomial of :func:`_origin_series`.  Order zero keeps only the
    correction derivative, so the expression stays finite at the
    origin.
    """
    denominator: float = _odd_double_factorial(2 * order + 1)
    second_denominator: float = 2.0 * (2 * order + 3)
    fourth_denominator: float = 8.0 * (2 * order + 3) * (2 * order + 5)
    correction: Float64[Array, " ..."] = (
        1.0 - x * x / second_denominator + x**4 / fourth_denominator
    )
    correction_derivative: Float64[Array, " ..."] = (
        -2.0 * x / second_denominator + 4.0 * x**3 / fourth_denominator
    )
    derivatives: Float64[Array, " ..."]
    if order == 0:
        derivatives = correction_derivative / denominator
    else:
        derivatives = (
            order * x ** (order - 1) * correction
            + x**order * correction_derivative
        ) / denominator
    return derivatives


def _analytic_anchors(
    x: Float64[Array, " ..."],
) -> Tuple[Float64[Array, " ..."], Float64[Array, " ..."]]:
    """PRIVATE: Evaluate analytic nonzero-argument anchors j0 and j1.

    Parameters
    ----------
    x : Float64[Array, " ..."]
        Real dimensionless arguments away from zero.

    Returns
    -------
    anchors : Tuple[Float64[Array, " ..."], Float64[Array, " ..."]]
        Pair ``(j0, j1)`` with ``j0 = sin(x)/x`` and
        ``j1 = sin(x)/x**2 - cos(x)/x``.

    Notes
    -----
    The closed forms divide by ``x``.  Callers pass sanitized nonzero
    arguments so the quotients stay finite.
    """
    j0: Float64[Array, " ..."] = jnp.sin(x) / x
    j1: Float64[Array, " ..."] = jnp.sin(x) / (x * x) - jnp.cos(x) / x
    anchors: Tuple[Float64[Array, " ..."], Float64[Array, " ..."]] = (j0, j1)
    return anchors


def _upward_recurrence(
    order: int,
    x: Float64[Array, " ..."],
    j0: Float64[Array, " ..."],
    j1: Float64[Array, " ..."],
) -> Float64[Array, " ..."]:
    """PRIVATE: Evaluate j_l by upward recurrence.

    Parameters
    ----------
    order : int
        Static target order of at least two.
    x : Float64[Array, " ..."]
        Real nonzero dimensionless arguments.
    j0 : Float64[Array, " ..."]
        Analytic anchor :math:`j_0(x)`.
    j1 : Float64[Array, " ..."]
        Analytic anchor :math:`j_1(x)`.

    Returns
    -------
    values : Float64[Array, " ..."]
        Spherical Bessel values :math:`j_l(x)` with the input shape.

    Implementation Logic
    --------------------
    A :func:`jax.lax.fori_loop` applies the recurrence
    ``j_(l+1) = (2l+1) j_l / x - j_(l-1)`` from the anchor pair
    ``(j0, j1)`` up to ``order``.  Upward recurrence is stable when
    ``order <= abs(x)``.
    """

    def _step(
        index: Integer[Array, ""],
        state: Tuple[Float64[Array, " ..."], Float64[Array, " ..."]],
    ) -> Tuple[Float64[Array, " ..."], Float64[Array, " ..."]]:
        """PRIVATE: Apply one step of the upward recurrence.

        Parameters
        ----------
        index : Integer[Array, ""]
            Order ``l`` of the current leading value.
        state : Tuple[Float64[Array, " ..."], Float64[Array, " ..."]]
            Consecutive pair ``(j_(l-1), j_l)``.

        Returns
        -------
        next_state : Tuple[Float64[Array, " ..."], Float64[Array, " ..."]]
            Shifted pair ``(j_l, j_(l+1))``.

        Notes
        -----
        Casts the loop index to float64 and applies
        ``j_(l+1) = (2l+1) j_l / x - j_(l-1)``.
        """
        previous: Float64[Array, " ..."] = state[0]
        current: Float64[Array, " ..."] = state[1]
        index_float: Float64[Array, ""] = jnp.asarray(index, dtype=jnp.float64)
        following: Float64[Array, " ..."] = (
            2.0 * index_float + 1.0
        ) * current / x - previous
        next_state: Tuple[Float64[Array, " ..."], Float64[Array, " ..."]] = (
            current,
            following,
        )
        return next_state

    final_state: Tuple[Float64[Array, " ..."], Float64[Array, " ..."]] = (
        jax.lax.fori_loop(1, order, _step, (j0, j1))
    )
    values: Float64[Array, " ..."] = final_state[1]
    return values


def _downward_miller(
    order: int,
    x: Float64[Array, " ..."],
    j0: Float64[Array, " ..."],
    j1: Float64[Array, " ..."],
) -> Float64[Array, " ..."]:
    """PRIVATE: Evaluate j_l by fixed-depth downward Miller recurrence.

    Parameters
    ----------
    order : int
        Static target order of at least two.
    x : Float64[Array, " ..."]
        Real nonzero dimensionless arguments.
    j0 : Float64[Array, " ..."]
        Analytic anchor :math:`j_0(x)`.
    j1 : Float64[Array, " ..."]
        Analytic anchor :math:`j_1(x)`.

    Returns
    -------
    values : Float64[Array, " ..."]
        Spherical Bessel values :math:`j_l(x)` with the input shape.

    Implementation Logic
    --------------------
    Starts at order ``order + ceil(sqrt(40 * max(order, 1))) + 12``
    with the arbitrary seed pair ``(0, 1)``.  Iterates the downward
    recurrence ``j_(l-1) = (2l+1) j_l / x - j_(l+1)`` to order zero
    and records the unnormalized value at the target order.  Miller
    normalization then rescales the recorded value with whichever
    analytic anchor, ``j0`` or ``j1``, has the larger magnitude.  The
    inactive ratio divides by one, so it stays finite.
    """
    start_order: int = order + math.ceil(math.sqrt(40.0 * max(order, 1))) + 12
    target_seed: Float64[Array, " ..."] = jnp.ones_like(x)

    def _step(
        iteration: Integer[Array, ""],
        state: Tuple[
            Float64[Array, " ..."],
            Float64[Array, " ..."],
            Float64[Array, " ..."],
        ],
    ) -> Tuple[
        Float64[Array, " ..."],
        Float64[Array, " ..."],
        Float64[Array, " ..."],
    ]:
        """PRIVATE: Apply one step of the downward recurrence.

        Parameters
        ----------
        iteration : Integer[Array, ""]
            Loop counter; the active order is
            ``start_order - iteration``.
        state : Tuple[Float64[Array, " ..."], Float64[Array, " ..."], \
Float64[Array, " ..."]]
            Triple ``(j_(l+1), j_l, target)`` of unnormalized values
            and the recorded target-order value.

        Returns
        -------
        next_state : Tuple[Float64[Array, " ..."], Float64[Array, " ..."], \
Float64[Array, " ..."]]
            Shifted triple ``(j_l, j_(l-1), target)``.

        Notes
        -----
        Applies ``j_(l-1) = (2l+1) j_l / x - j_(l+1)`` and stores the
        new value into ``target`` when ``l - 1`` equals the requested
        order.
        """
        following: Float64[Array, " ..."] = state[0]
        current: Float64[Array, " ..."] = state[1]
        target: Float64[Array, " ..."] = state[2]
        ell: Integer[Array, ""] = start_order - iteration
        ell_float: Float64[Array, ""] = jnp.asarray(ell, dtype=jnp.float64)
        previous: Float64[Array, " ..."] = (
            2.0 * ell_float + 1.0
        ) * current / x - following
        target = jnp.where(ell - 1 == order, previous, target)
        next_state: Tuple[
            Float64[Array, " ..."],
            Float64[Array, " ..."],
            Float64[Array, " ..."],
        ] = (current, previous, target)
        return next_state

    initial_state: Tuple[
        Float64[Array, " ..."],
        Float64[Array, " ..."],
        Float64[Array, " ..."],
    ] = (jnp.zeros_like(x), jnp.ones_like(x), target_seed)
    miller_state: Tuple[
        Float64[Array, " ..."],
        Float64[Array, " ..."],
        Float64[Array, " ..."],
    ] = jax.lax.fori_loop(
        0,
        start_order,
        _step,
        initial_state,
    )
    raw_j1: Float64[Array, " ..."] = miller_state[0]
    raw_j0: Float64[Array, " ..."] = miller_state[1]
    raw_target: Float64[Array, " ..."] = miller_state[2]
    use_j0: Float64[Array, " ..."] = jnp.abs(j0) >= jnp.abs(j1)
    safe_raw_j0: Float64[Array, " ..."] = jnp.where(
        use_j0, raw_j0, jnp.ones_like(raw_j0)
    )
    safe_raw_j1: Float64[Array, " ..."] = jnp.where(
        use_j0, jnp.ones_like(raw_j1), raw_j1
    )
    scale_from_j0: Float64[Array, " ..."] = j0 / safe_raw_j0
    scale_from_j1: Float64[Array, " ..."] = j1 / safe_raw_j1
    scale: Float64[Array, " ..."] = jnp.where(
        use_j0, scale_from_j0, scale_from_j1
    )
    values: Float64[Array, " ..."] = raw_target * scale
    return values


@jaxtyped(typechecker=beartype)
def spherical_bessel_jl(
    order: int,
    x: Float64[Array, " ..."],
) -> Float64[Array, " ..."]:
    """Evaluate the spherical Bessel function :math:`j_l(x)`.

    The kernel selects an origin series or a stable recurrence per argument.

    :see: :class:`~.test_bessel.TestSphericalBesselJl`

    Parameters
    ----------
    order : int
        Nonnegative static angular-momentum order.
    x : Float64[Array, " ..."]
        Real arguments.

    Returns
    -------
    values : Float64[Array, " ..."]
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

    x_array: Float64[Array, " ..."] = jnp.asarray(x, dtype=jnp.float64)
    series_threshold: float = 2.0e-2
    use_series: Float64[Array, " ..."] = jnp.abs(x_array) < series_threshold
    safe_x: Float64[Array, " ..."] = jnp.where(use_series, 1.0, x_array)
    series_values: Float64[Array, " ..."] = _origin_series(order, x_array)
    anchors: Tuple[Float64[Array, " ..."], Float64[Array, " ..."]] = (
        _analytic_anchors(safe_x)
    )
    j0: Float64[Array, " ..."] = anchors[0]
    j1: Float64[Array, " ..."] = anchors[1]

    recurrence_values: Float64[Array, " ..."]
    if order == 0:
        recurrence_values = j0
    elif order == 1:
        recurrence_values = j1
    else:
        upward_values: Float64[Array, " ..."] = _upward_recurrence(
            order, safe_x, j0, j1
        )
        downward_values: Float64[Array, " ..."] = _downward_miller(
            order, safe_x, j0, j1
        )
        recurrence_values = jnp.where(
            order <= jnp.abs(safe_x),
            upward_values,
            downward_values,
        )
    values: Float64[Array, " ..."] = jnp.where(
        use_series,
        series_values,
        recurrence_values,
    )
    return values


@jaxtyped(typechecker=beartype)
def spherical_bessel_jl_derivative(
    order: int,
    x: Float64[Array, " ..."],
) -> Float64[Array, " ..."]:
    """Evaluate :math:`d j_l(x)/dx`.

    The kernel differentiates the origin series and uses stable identities
    elsewhere.

    :see: :class:`~.test_bessel.TestSphericalBesselJlDerivative`

    Parameters
    ----------
    order : int
        Nonnegative static angular-momentum order.
    x : Float64[Array, " ..."]
        Real arguments.

    Returns
    -------
    derivatives : Float64[Array, " ..."]
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

    x_array: Float64[Array, " ..."] = jnp.asarray(x, dtype=jnp.float64)
    series_threshold: float = 2.0e-2
    use_series: Float64[Array, " ..."] = jnp.abs(x_array) < series_threshold
    safe_x: Float64[Array, " ..."] = jnp.where(use_series, 1.0, x_array)
    series_derivatives: Float64[Array, " ..."] = _origin_series_derivative(
        order, x_array
    )
    if order == 0:
        recurrence_derivatives: Float64[Array, " ..."] = -spherical_bessel_jl(
            1, safe_x
        )
    else:
        recurrence_derivatives = spherical_bessel_jl(order - 1, safe_x) - (
            (order + 1.0) * spherical_bessel_jl(order, safe_x) / safe_x
        )
    derivatives: Float64[Array, " ..."] = jnp.where(
        use_series,
        series_derivatives,
        recurrence_derivatives,
    )
    return derivatives


__all__: list[str] = [
    "spherical_bessel_jl",
    "spherical_bessel_jl_derivative",
]
