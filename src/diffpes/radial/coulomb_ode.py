# ruff: noqa: PLR2004
"""Propagate diagnostic Coulomb states with a fixed ODE solve.

Extended Summary
----------------
This module uses a fixed-step Dormand--Prince solve.
It produces diagnostic Coulomb states.
"""

from functools import partial

import jax
import jax.numpy as jnp
from beartype.typing import Tuple
from jax import lax
from jaxtyping import Array, Complex128, Float64, Int64

from .coulomb_asymptotics import (
    _outgoing_asymptotic_state,
    _regular_origin_state,
)


def _coulomb_log_radius_rhs(
    state: Float64[Array, " 2"],
    coordinate: Float64[Array, ""],
    eta: Float64[Array, ""],
    order: int,
    direction: float,
) -> Float64[Array, " 2"]:
    """PRIVATE: Return the first-order Coulomb system in signed log radius.

    Parameters
    ----------
    state : Float64[Array, " 2"]
        Pair of the untransformed value ``u`` and its log-radius
        derivative ``du/d(log rho)``.
    coordinate : Float64[Array, ""]
        Integration coordinate; ``rho = exp(direction * coordinate)``.
    eta : Float64[Array, ""]
        Dimensionless Sommerfeld parameter.
    order : int
        Static angular momentum from 0 through 5.
    direction : float
        Static sign that maps the coordinate to log radius.

    Returns
    -------
    result : Float64[Array, " 2"]
        Coordinate derivative of the state.

    Notes
    -----
    In log radius the Coulomb equation reads
    ``u'' = u' - (rho**2 - 2 eta rho - l(l+1)) u`` with primes taken
    with respect to ``log rho``.  The returned stack multiplies both
    components by ``direction`` so one right-hand side serves the
    outward and the inward integration.
    """
    radius: Float64[Array, ""] = jnp.exp(direction * coordinate)
    value: Float64[Array, ""] = state[0]
    coordinate_derivative: Float64[Array, ""] = state[1]
    second_derivative: Float64[Array, ""] = (
        coordinate_derivative
        - (radius**2 - 2.0 * eta * radius - order * (order + 1)) * value
    )
    result: Float64[Array, " 2"] = direction * jnp.stack(
        (coordinate_derivative, second_derivative)
    )
    return result


def _fixed_dopri5_endpoint(
    state: Float64[Array, " 2"],
    coordinate_start: Float64[Array, ""],
    coordinate_end: Float64[Array, ""],
    eta: Float64[Array, ""],
    order: int,
    direction: float,
) -> Float64[Array, " 2"]:
    """PRIVATE: Propagate the Coulomb system with fixed Dormand--Prince
    fifth order.

    Implementation Logic
    --------------------
    1. **Set the parameter-independent steps**::

           step = (coordinate_end - coordinate_start) / 32768

       The fixed sequence avoids a data-dependent adaptive branch.

    2. **Apply the six-stage tableau**::

           result = lax.fori_loop(0, steps, jax.checkpoint(body), state)

       The loop keeps the fifth-order solution. Checkpointing bounds the
       memory use during reverse-mode differentiation.

    Parameters
    ----------
    state : Float64[Array, " 2"]
        Initial pair of the value and its log-radius derivative.
    coordinate_start : Float64[Array, ""]
        Signed log-radius start coordinate.
    coordinate_end : Float64[Array, ""]
        Signed log-radius end coordinate.
    eta : Float64[Array, ""]
        Dimensionless Sommerfeld parameter.
    order : int
        Static angular momentum from 0 through 5.
    direction : float
        Static sign that maps the coordinate to log radius.

    Returns
    -------
    result : Float64[Array, " 2"]
        Propagated state at the end coordinate.

    """
    steps: int = 32768
    step: Float64[Array, ""] = (coordinate_end - coordinate_start) / steps

    def rhs(
        value: Float64[Array, " 2"],
        coordinate: Float64[Array, ""],
    ) -> Float64[Array, " 2"]:
        result: Float64[Array, " 2"] = _coulomb_log_radius_rhs(
            value,
            coordinate,
            eta,
            order,
            direction,
        )
        return result

    def body(
        index: Int64[Array, ""], current: Float64[Array, " 2"]
    ) -> Float64[Array, " 2"]:
        coordinate: Float64[Array, ""] = coordinate_start + index * step
        first: Float64[Array, " 2"] = rhs(current, coordinate)
        second: Float64[Array, " 2"] = rhs(
            current + step * first / 5.0,
            coordinate + step / 5.0,
        )
        third: Float64[Array, " 2"] = rhs(
            current + step * (3.0 * first + 9.0 * second) / 40.0,
            coordinate + 3.0 * step / 10.0,
        )
        fourth: Float64[Array, " 2"] = rhs(
            current
            + step
            * (
                44.0 * first / 45.0 - 56.0 * second / 15.0 + 32.0 * third / 9.0
            ),
            coordinate + 4.0 * step / 5.0,
        )
        fifth: Float64[Array, " 2"] = rhs(
            current
            + step
            * (
                19372.0 * first / 6561.0
                - 25360.0 * second / 2187.0
                + 64448.0 * third / 6561.0
                - 212.0 * fourth / 729.0
            ),
            coordinate + 8.0 * step / 9.0,
        )
        sixth: Float64[Array, " 2"] = rhs(
            current
            + step
            * (
                9017.0 * first / 3168.0
                - 355.0 * second / 33.0
                + 46732.0 * third / 5247.0
                + 49.0 * fourth / 176.0
                - 5103.0 * fifth / 18656.0
            ),
            coordinate + step,
        )
        result: Float64[Array, " 2"] = current + step * (
            35.0 * first / 384.0
            + 500.0 * third / 1113.0
            + 125.0 * fourth / 192.0
            - 2187.0 * fifth / 6784.0
            + 11.0 * sixth / 84.0
        )
        return result

    result: Float64[Array, " 2"] = lax.fori_loop(
        0,
        steps,
        jax.checkpoint(body),
        state,
    )
    return result


def _accurate_coulomb_scalar(
    order: int,
    eta: Float64[Array, ""],
    rho: Float64[Array, ""],
) -> Float64[Array, " 4"]:
    r"""PRIVATE: Evaluate one high-accuracy value-and-derivative row.

    Implementation Logic
    --------------------
    1. **Propagate the regular solution outward**::

           regular_endpoint = _fixed_dopri5_endpoint(
               regular_state,
               jnp.log(origin_rho),
               jnp.log(regular_target),
               eta,
               order,
               1.0,
           )

       The origin series at ``rho = 1e-4`` supplies the scaled initial state.

    2. **Propagate the irregular solution inward**::

           irregular_endpoint = _fixed_dopri5_endpoint(
               irregular_state,
               -jnp.log(boundary_rho),
               -jnp.log(jnp.minimum(rho, boundary_rho)),
               eta,
               order,
               -1.0,
           )

       The real outgoing solution at ``rho = 20`` supplies the boundary state.

    3. **Restore radial derivatives**::

           irregular_derivative = irregular_endpoint[1] / rho

       Division by the radius converts log-radius derivatives to radial
       derivatives. Exact series or asymptotic values remain at each boundary.

    Parameters
    ----------
    order : int
        Static angular momentum from 0 through 5.
    eta : Float64[Array, ""]
        Dimensionless Sommerfeld parameter.
    rho : Float64[Array, ""]
        Positive dimensionless radius.

    Returns
    -------
    result : Float64[Array, " 4"]
        Stack of :math:`F_l`, :math:`G_l`, and their :math:`\rho`
        derivatives at one scalar argument pair.

    """
    origin_rho: Float64[Array, ""] = jnp.asarray(1.0e-4)
    regular_target: Float64[Array, ""] = jnp.maximum(
        rho,
        origin_rho * (1.0 + 1.0e-12),
    )
    regular_start: Float64[Array, ""]
    regular_start_derivative: Float64[Array, ""]
    regular_start, regular_start_derivative = _regular_origin_state(
        order,
        eta,
        origin_rho,
    )
    regular_scale: Float64[Array, ""] = origin_rho ** (order + 1)
    regular_state: Float64[Array, " 2"] = jnp.stack(
        (
            regular_start / regular_scale,
            origin_rho * regular_start_derivative / regular_scale,
        )
    )
    regular_endpoint: Float64[Array, " 2"] = _fixed_dopri5_endpoint(
        regular_state,
        jnp.log(origin_rho),
        jnp.log(regular_target),
        eta,
        order,
        1.0,
    )
    propagated_regular: Float64[Array, ""] = (
        regular_scale * regular_endpoint[0]
    )
    origin_regular: Float64[Array, ""]
    origin_regular, _ = _regular_origin_state(order, eta, rho)
    regular: Float64[Array, ""] = jnp.where(
        rho == origin_rho,
        origin_regular,
        propagated_regular,
    )
    origin_regular_derivative: Float64[Array, ""]
    _, origin_regular_derivative = _regular_origin_state(
        order,
        eta,
        rho,
    )
    propagated_regular_derivative: Float64[Array, ""] = (
        regular_scale * regular_endpoint[1] / regular_target
    )
    regular_derivative: Float64[Array, ""] = jnp.where(
        rho == origin_rho,
        origin_regular_derivative,
        propagated_regular_derivative,
    )

    boundary_rho: Float64[Array, ""] = jnp.asarray(20.0)
    outgoing: Complex128[Array, ""]
    outgoing_derivative: Complex128[Array, ""]
    outgoing, outgoing_derivative = _outgoing_asymptotic_state(
        order,
        eta,
        boundary_rho,
    )
    irregular_state: Float64[Array, " 2"] = jnp.stack(
        (
            jnp.real(outgoing),
            boundary_rho * jnp.real(outgoing_derivative),
        )
    )
    irregular_endpoint: Float64[Array, " 2"] = _fixed_dopri5_endpoint(
        irregular_state,
        -jnp.log(boundary_rho),
        -jnp.log(jnp.minimum(rho, boundary_rho)),
        eta,
        order,
        -1.0,
    )
    asymptotic_at_rho: Complex128[Array, ""]
    asymptotic_derivative_at_rho: Complex128[Array, ""]
    asymptotic_at_rho, asymptotic_derivative_at_rho = (
        _outgoing_asymptotic_state(
            order,
            eta,
            rho,
        )
    )
    irregular: Float64[Array, ""] = jnp.where(
        rho >= boundary_rho,
        jnp.real(asymptotic_at_rho),
        irregular_endpoint[0],
    )
    irregular_derivative: Float64[Array, ""] = jnp.where(
        rho >= boundary_rho,
        jnp.real(asymptotic_derivative_at_rho),
        irregular_endpoint[1] / rho,
    )
    result: Float64[Array, " 4"] = jnp.stack(
        (
            regular,
            irregular,
            regular_derivative,
            irregular_derivative,
        )
    )
    return result


def _accurate_coulomb_values_impl(
    order: int,
    eta: Float64[Array, " ..."],
    rho: Float64[Array, " ..."],
) -> Float64[Array, "4 ..."]:
    """PRIVATE: Evaluate adaptive values before the custom derivative rule.

    Parameters
    ----------
    order : int
        Static angular momentum from 0 through 5.
    eta : Float64[Array, " ..."]
        Dimensionless Sommerfeld parameter.
    rho : Float64[Array, " ..."]
        Positive dimensionless radius with the same shape as ``eta``.

    Returns
    -------
    values : Float64[Array, "4 ..."]
        Rows ``F``, ``G``, ``dF/drho``, ``dG/drho`` with the input
        shape behind the leading axis.

    Notes
    -----
    Flattens both arguments, maps :func:`_accurate_coulomb_scalar`
    sequentially with :func:`jax.lax.map`, restores the input shape,
    and moves the four-component axis to the front.
    """
    flat_eta: Float64[Array, " n"] = jnp.ravel(eta)
    flat_rho: Float64[Array, " n"] = jnp.ravel(rho)
    flat_values: Float64[Array, "n 4"] = lax.map(
        lambda arguments: _accurate_coulomb_scalar(
            order,
            arguments[0],
            arguments[1],
        ),
        (flat_eta, flat_rho),
    )
    values: Float64[Array, "2 ..."] = jnp.moveaxis(
        flat_values.reshape(eta.shape + (4,)),
        -1,
        0,
    )
    return values


@partial(jax.custom_jvp, nondiff_argnums=(0,))
def _accurate_coulomb_values(
    order: int,
    eta: Float64[Array, " ..."],
    rho: Float64[Array, " ..."],
) -> Float64[Array, "4 ..."]:
    """PRIVATE: Evaluate adaptive values with a fixed-Numerov
    differentiation rule.

    Parameters
    ----------
    order : int
        Static angular momentum from 0 through 5.
    eta : Float64[Array, " ..."]
        Dimensionless Sommerfeld parameter.
    rho : Float64[Array, " ..."]
        Positive dimensionless radius with the same shape as ``eta``.

    Returns
    -------
    result : Float64[Array, "4 ..."]
        Rows ``F``, ``G``, ``dF/drho``, ``dG/drho``.

    Notes
    -----
    This is the primal of a :func:`jax.custom_jvp`.  It wraps
    :func:`_accurate_coulomb_values_impl` in
    :func:`jax.lax.stop_gradient` so all differentiation flows through
    the registered rule :func:`_accurate_coulomb_values_jvp`.
    """
    values: Float64[Array, "4 ..."] = _accurate_coulomb_values_impl(
        order,
        eta,
        rho,
    )
    result: Float64[Array, "4 ..."] = lax.stop_gradient(values)
    return result  # noqa: RET504 -- assign-before-return is required.


@_accurate_coulomb_values.defjvp
def _accurate_coulomb_values_jvp(
    order: int,
    primals: Tuple[Float64[Array, " ..."], Float64[Array, " ..."]],
    tangents: Tuple[Float64[Array, " ..."], Float64[Array, " ..."]],
) -> Tuple[Float64[Array, "4 ..."], Float64[Array, "4 ..."]]:
    """PRIVATE: Differentiate adaptive values through the fixed Numerov
    solver.

    Implementation Logic
    --------------------
    1. **Differentiate the Sommerfeld parameter**::

           eta_derivative = (
               -_accurate_coulomb_values_impl(
                   order, eta + 2.0 * eta_step, rho
               )
               + 8.0 * _accurate_coulomb_values_impl(
                   order, eta + eta_step, rho
               )
               - 8.0 * _accurate_coulomb_values_impl(
                   order, eta - eta_step, rho
               )
               + _accurate_coulomb_values_impl(
                   order, eta - 2.0 * eta_step, rho
               )
           ) / (12.0 * eta_step)

       The fourth-order central difference uses
       ``max(1, abs(eta)) * 2**-9`` as its step.

    2. **Differentiate the radius through the ODE**::

           rho_derivative = jnp.stack(
               (
                   values[2],
                   values[3],
                   -ode_factor * values[0],
                   -ode_factor * values[1],
               )
           )

       The stored first derivatives and the Coulomb equation give exact radial
       tangent rows.

    3. **Combine the input tangents**::

           tangent_values = (
               eta_derivative * eta_tangent
               + rho_derivative * rho_tangent
           )

       ``lax.stop_gradient`` makes both derivative blocks first-order rules.

    Parameters
    ----------
    order : int
        Static angular momentum from 0 through 5.
    primals : Tuple[Float64[Array, " ..."], Float64[Array, " ..."]]
        Pair ``(eta, rho)`` of primal arguments.
    tangents : Tuple[Float64[Array, " ..."], Float64[Array, " ..."]]
        Pair of the matching input tangents.

    Returns
    -------
    result : Tuple[Float64[Array, "4 ..."], Float64[Array, "4 ..."]]
        Primal rows and their joint tangent rows.

    """
    eta: Float64[Array, " ..."]
    rho: Float64[Array, " ..."]
    eta_tangent: Float64[Array, " ..."]
    rho_tangent: Float64[Array, " ..."]
    eta, rho = primals
    eta_tangent, rho_tangent = tangents
    values: Float64[Array, "4 ..."] = _accurate_coulomb_values_impl(
        order,
        eta,
        rho,
    )
    eta_step: Float64[Array, " ..."] = jnp.maximum(1.0, jnp.abs(eta)) * 2.0**-9
    eta_derivative: Float64[Array, "4 ..."] = (
        -_accurate_coulomb_values_impl(order, eta + 2.0 * eta_step, rho)
        + 8.0 * _accurate_coulomb_values_impl(order, eta + eta_step, rho)
        - 8.0 * _accurate_coulomb_values_impl(order, eta - eta_step, rho)
        + _accurate_coulomb_values_impl(order, eta - 2.0 * eta_step, rho)
    ) / (12.0 * eta_step)
    ode_factor: Float64[Array, " ..."] = (
        1.0 - 2.0 * eta / rho - float(order * (order + 1)) / rho**2
    )
    rho_derivative: Float64[Array, "4 ..."] = jnp.stack(
        (
            values[2],
            values[3],
            -ode_factor * values[0],
            -ode_factor * values[1],
        )
    )
    eta_derivative = lax.stop_gradient(eta_derivative)
    rho_derivative = lax.stop_gradient(rho_derivative)
    tangent_values: Float64[Array, "4 ..."] = (
        eta_derivative * eta_tangent + rho_derivative * rho_tangent
    )
    result: Tuple[Float64[Array, "4 ..."], Float64[Array, "4 ..."]] = (
        values,
        tangent_values,
    )
    return result


__all__: list[str] = []
