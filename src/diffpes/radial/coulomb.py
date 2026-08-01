# ruff: noqa: PLR2004
r"""Evaluate regular and irregular Coulomb radial functions.

Extended Summary
----------------
This module implements an original pure-JAX solver for the dimensionless
Coulomb equation

.. math::

    u'' + [1 - 2\eta/\rho - l(l+1)/\rho^2]u = 0.

A fixed-step pure-JAX Dormand--Prince solve propagates the diagnostic Coulomb
functions without parameter-dependent adaptive-step noise.  The production
final-state row uses a bounded static Numerov propagation initialized by
convergent origin and inverse-radius series.  A custom differentiation rule
supplies stable parameter derivatives while the returned radial derivatives
remain components of the solved ODE state.

Routine Listings
----------------
:func:`coulomb_fg`
    Evaluate normalized Coulomb functions and radial derivatives.
:func:`coulomb_phase_shift`
    Evaluate the continuous Coulomb arg-Gamma phase.
:func:`final_state_radial`
    Evaluate a plane-wave or Coulomb final-state radial row.
"""

import math
from functools import partial

import equinox as eqx
import jax
import jax.numpy as jnp
from beartype import beartype
from jax import lax
from jaxtyping import Array, Complex128, Float64, jaxtyped

from diffpes.types import FinalStateSpec

from .bessel import spherical_bessel_jl, spherical_bessel_jl_derivative


def _validate_order(order: int) -> None:
    """Validate one static Coulomb angular momentum."""
    if type(order) is not int or not 0 <= order <= 5:
        message: str = "Coulomb order must be an integer from 0 through 5"
        raise ValueError(message)


def _complex_log_gamma_shifted(
    value: Complex128[Array, " ..."],
) -> Complex128[Array, " ..."]:
    """Evaluate log Gamma by recurrence to a converged Stirling domain."""
    shifted: Complex128[Array, " ..."] = value + 20
    result: Complex128[Array, " ..."] = (
        (shifted - 0.5) * jnp.log(shifted)
        - shifted
        + 0.5 * math.log(2.0 * math.pi)
    )
    bernoulli_values: tuple[float, ...] = (
        1.0 / 6.0,
        -1.0 / 30.0,
        1.0 / 42.0,
        -1.0 / 30.0,
        5.0 / 66.0,
        -691.0 / 2730.0,
        7.0 / 6.0,
        -3617.0 / 510.0,
        43867.0 / 798.0,
        -174611.0 / 330.0,
    )
    term_index: int
    bernoulli: float
    for term_index, bernoulli in enumerate(
        bernoulli_values,
        start=1,
    ):
        result = result + bernoulli / (
            2.0
            * term_index
            * (2.0 * term_index - 1.0)
            * shifted ** (2 * term_index - 1)
        )
    recurrence_index: int
    for recurrence_index in range(20):
        result = result - jnp.log(value + recurrence_index)
    return result


def _log_coulomb_normalization(
    order: int,
    eta: Float64[Array, " ..."],
) -> Float64[Array, " ..."]:
    """Return log C_l(eta) for the regular Coulomb solution."""
    scaled_eta: Float64[Array, " ..."] = math.pi * eta
    small: Array = jnp.abs(scaled_eta) < 1.0e-4
    safe_scaled_eta: Float64[Array, " ..."] = jnp.where(
        small,
        jnp.ones_like(scaled_eta),
        scaled_eta,
    )
    direct_ratio: Float64[Array, " ..."] = safe_scaled_eta / jnp.sinh(
        safe_scaled_eta
    )
    series_ratio: Float64[Array, " ..."] = (
        1.0
        - scaled_eta**2 / 6.0
        + 7.0 * scaled_eta**4 / 360.0
        - 31.0 * scaled_eta**6 / 15120.0
    )
    gamma_one_abs_squared: Float64[Array, " ..."] = jnp.where(
        small,
        series_ratio,
        direct_ratio,
    )
    result: Float64[Array, " ..."] = (
        order * math.log(2.0)
        - 0.5 * scaled_eta
        + 0.5 * jnp.log(gamma_one_abs_squared)
        - math.lgamma(2 * order + 2)
    )
    recurrence_index: int
    for recurrence_index in range(1, order + 1):
        result = result + 0.5 * jnp.log(recurrence_index**2 + eta**2)
    return result


def _coulomb_phase_unchecked(
    order: int,
    eta: Float64[Array, " ..."],
) -> Float64[Array, " ..."]:
    """Evaluate the continuous phase for an internal recurrence order."""
    argument: Complex128[Array, " ..."] = (1.0 + 1j * eta).astype(
        jnp.complex128
    )
    phase: Float64[Array, " ..."] = jnp.imag(
        _complex_log_gamma_shifted(argument)
    )
    recurrence_index: int
    for recurrence_index in range(1, order + 1):
        phase = phase + jnp.arctan2(eta, recurrence_index)
    return phase


def _regular_origin_state(
    order: int,
    eta: Float64[Array, " ..."],
    rho: Float64[Array, " ..."],
) -> tuple[Float64[Array, " ..."], Float64[Array, " ..."]]:
    """Evaluate regular origin series and its rho derivative."""
    coefficients: list[Float64[Array, " ..."]] = [jnp.ones_like(eta)]
    series_index: int
    for series_index in range(1, 64):
        previous_two: Float64[Array, " ..."] = (
            coefficients[series_index - 2]
            if series_index >= 2
            else jnp.zeros_like(eta)
        )
        denominator: float = float(
            series_index * (series_index + 2 * order + 1)
        )
        coefficient: Float64[Array, " ..."] = (
            2.0 * eta * coefficients[series_index - 1] - previous_two
        ) / denominator
        coefficients.append(coefficient)
    unnormalized: Float64[Array, " ..."] = jnp.zeros_like(rho)
    unnormalized_derivative: Float64[Array, " ..."] = jnp.zeros_like(rho)
    coefficient_index: int
    for coefficient_index, coefficient in enumerate(coefficients):
        power: int = order + 1 + coefficient_index
        unnormalized = unnormalized + coefficient * rho**power
        unnormalized_derivative = (
            unnormalized_derivative + power * coefficient * rho ** (power - 1)
        )
    normalization: Float64[Array, " ..."] = jnp.exp(
        _log_coulomb_normalization(order, eta)
    )
    value: Float64[Array, " ..."] = normalization * unnormalized
    derivative: Float64[Array, " ..."] = (
        normalization * unnormalized_derivative
    )
    result: tuple[Float64[Array, " ..."], Float64[Array, " ..."]] = (
        value,
        derivative,
    )
    return result


def _irregular_origin_state(
    order: int,
    eta: Float64[Array, " ..."],
    rho: Float64[Array, " ..."],
    matched_irregular: Float64[Array, " ..."],
) -> tuple[Float64[Array, " ..."], Float64[Array, " ..."]]:
    """Evaluate the logarithmic irregular Frobenius row near the origin."""
    coefficient_count: int = 32
    regular_coefficients: list[Float64[Array, " ..."]] = [jnp.ones_like(eta)]
    coefficient_index: int
    for coefficient_index in range(1, coefficient_count):
        previous_two: Float64[Array, " ..."] = (
            regular_coefficients[coefficient_index - 2]
            if coefficient_index >= 2
            else jnp.zeros_like(eta)
        )
        denominator: float = float(
            coefficient_index * (coefficient_index + 2 * order + 1)
        )
        regular_coefficients.append(
            (
                2.0 * eta * regular_coefficients[coefficient_index - 1]
                - previous_two
            )
            / denominator
        )
    squared_coefficients: list[Float64[Array, " ..."]] = []
    for coefficient_index in range(coefficient_count):
        squared: Float64[Array, " ..."] = jnp.zeros_like(eta)
        left_index: int
        for left_index in range(coefficient_index + 1):
            squared = (
                squared
                + regular_coefficients[left_index]
                * regular_coefficients[coefficient_index - left_index]
            )
        squared_coefficients.append(squared)
    inverse_coefficients: list[Float64[Array, " ..."]] = [jnp.ones_like(eta)]
    for coefficient_index in range(1, coefficient_count):
        inverse: Float64[Array, " ..."] = jnp.zeros_like(eta)
        product_index: int
        for product_index in range(1, coefficient_index + 1):
            inverse = (
                inverse
                + squared_coefficients[product_index]
                * inverse_coefficients[coefficient_index - product_index]
            )
        inverse_coefficients.append(-inverse)

    normalization: Float64[Array, " ..."] = jnp.exp(
        _log_coulomb_normalization(order, eta)
    )

    def primitive(argument: Float64[Array, " ..."]) -> Float64[Array, " ..."]:
        result: Float64[Array, " ..."] = jnp.zeros_like(argument)
        term_index: int
        inverse_coefficient: Float64[Array, " ..."]
        for term_index, inverse_coefficient in enumerate(inverse_coefficients):
            exponent: int = term_index - 2 * order - 1
            if exponent == 0:
                result = result - inverse_coefficient * jnp.log(argument)
            else:
                result = (
                    result
                    - inverse_coefficient * argument**exponent / exponent
                )
        result = result / normalization**2
        return result  # noqa: RET504

    switch_rho: Float64[Array, " ..."] = jnp.full_like(rho, 0.1)
    regular: Float64[Array, " ..."]
    regular_derivative: Float64[Array, " ..."]
    regular_switch: Float64[Array, " ..."]
    regular, regular_derivative = _regular_origin_state(order, eta, rho)
    regular_switch, _ = _regular_origin_state(order, eta, switch_rho)
    primitive_value: Float64[Array, " ..."] = primitive(rho)
    primitive_switch: Float64[Array, " ..."] = primitive(switch_rho)
    matching_constant: Float64[Array, " ..."] = (
        matched_irregular / regular_switch - primitive_switch
    )
    combined_primitive: Float64[Array, " ..."] = (
        primitive_value + matching_constant
    )
    value: Float64[Array, " ..."] = regular * combined_primitive
    inverse_square: Float64[Array, " ..."] = jnp.zeros_like(rho)
    inverse_coefficient: Float64[Array, " ..."]
    for coefficient_index, inverse_coefficient in enumerate(
        inverse_coefficients
    ):
        inverse_square = (
            inverse_square + inverse_coefficient * rho**coefficient_index
        )
    primitive_derivative: Float64[Array, " ..."] = (
        -(rho ** (-2 * order - 2)) * inverse_square / normalization**2
    )
    derivative: Float64[Array, " ..."] = (
        regular_derivative * combined_primitive
        + regular * primitive_derivative
    )
    result: tuple[Float64[Array, " ..."], Float64[Array, " ..."]] = (
        value,
        derivative,
    )
    return result


def _outgoing_asymptotic_state(
    order: int,
    eta: Float64[Array, " ..."],
    rho: Float64[Array, " ..."],
) -> tuple[Complex128[Array, " ..."], Complex128[Array, " ..."]]:
    """Evaluate outgoing H+ and its rho derivative by inverse-radius series."""
    channel_constant: Complex128[Array, " ..."] = (
        1j * eta - eta**2 - order * (order + 1)
    )
    coefficients: list[Complex128[Array, " ..."]] = [
        jnp.ones_like(eta, dtype=jnp.complex128)
    ]
    series_index: int
    for series_index in range(30):
        numerator: Complex128[Array, " ..."] = (
            series_index * (series_index + 1)
            + 2j * eta * series_index
            + channel_constant
        )
        coefficient: Complex128[Array, " ..."] = (
            numerator * coefficients[-1] / (2j * (series_index + 1))
        )
        coefficients.append(coefficient)
    amplitude: Complex128[Array, " ..."] = jnp.zeros_like(
        eta,
        dtype=jnp.complex128,
    )
    amplitude_derivative: Complex128[Array, " ..."] = jnp.zeros_like(
        eta,
        dtype=jnp.complex128,
    )
    coefficient_index: int
    for coefficient_index, coefficient in enumerate(coefficients):
        amplitude = amplitude + coefficient * rho ** (-coefficient_index)
        amplitude_derivative = amplitude_derivative - (
            coefficient_index * coefficient * rho ** (-coefficient_index - 1)
        )
    phase: Float64[Array, " ..."] = (
        rho
        - eta * jnp.log(2.0 * rho)
        - order * math.pi / 2.0
        + _coulomb_phase_unchecked(order, eta)
    )
    phase_derivative: Float64[Array, " ..."] = 1.0 - eta / rho
    phase_factor: Complex128[Array, " ..."] = jnp.exp(1j * phase)
    value: Complex128[Array, " ..."] = phase_factor * amplitude
    derivative: Complex128[Array, " ..."] = phase_factor * (
        1j * phase_derivative * amplitude + amplitude_derivative
    )
    result: tuple[
        Complex128[Array, " ..."],
        Complex128[Array, " ..."],
    ] = (value, derivative)
    return result


def _numerov_endpoint(
    order: int,
    eta: Float64[Array, " ..."],
    coordinate_start: Float64[Array, " ..."],
    coordinate_end: Float64[Array, " ..."],
    value_start: Float64[Array, " ..."],
    value_next: Float64[Array, " ..."],
    direction: float,
    steps: int,
) -> Float64[Array, " ..."]:
    """Propagate one transformed Coulomb value on a uniform log grid."""
    step: Float64[Array, " ..."] = (coordinate_end - coordinate_start) / steps

    def potential(
        coordinate: Float64[Array, " ..."],
    ) -> Float64[Array, " ..."]:
        radius: Float64[Array, " ..."] = jnp.exp(direction * coordinate)
        result: Float64[Array, " ..."] = (
            radius**2 - 2.0 * eta * radius - order * (order + 1) - 0.25
        )
        return result

    potential_previous: Float64[Array, " ..."] = potential(coordinate_start)
    potential_current: Float64[Array, " ..."] = potential(
        coordinate_start + step
    )

    def body(
        index: Array,
        state: tuple[
            Float64[Array, " ..."],
            Float64[Array, " ..."],
            Float64[Array, " ..."],
            Float64[Array, " ..."],
        ],
    ) -> tuple[
        Float64[Array, " ..."],
        Float64[Array, " ..."],
        Float64[Array, " ..."],
        Float64[Array, " ..."],
    ]:
        previous: Float64[Array, " ..."]
        current: Float64[Array, " ..."]
        q_previous: Float64[Array, " ..."]
        q_current: Float64[Array, " ..."]
        previous, current, q_previous, q_current = state
        coordinate_next: Float64[Array, " ..."] = (
            coordinate_start + (index + 1) * step
        )
        q_next: Float64[Array, " ..."] = potential(coordinate_next)
        next_value: Float64[Array, " ..."] = (
            2.0 * (1.0 - 5.0 * step**2 * q_current / 12.0) * current
            - (1.0 + step**2 * q_previous / 12.0) * previous
        ) / (1.0 + step**2 * q_next / 12.0)
        next_state: tuple[
            Float64[Array, " ..."],
            Float64[Array, " ..."],
            Float64[Array, " ..."],
            Float64[Array, " ..."],
        ] = (current, next_value, q_current, q_next)
        return next_state

    initial_state: tuple[
        Float64[Array, " ..."],
        Float64[Array, " ..."],
        Float64[Array, " ..."],
        Float64[Array, " ..."],
    ] = (
        value_start,
        value_next,
        potential_previous,
        potential_current,
    )
    final_state: tuple[
        Float64[Array, " ..."],
        Float64[Array, " ..."],
        Float64[Array, " ..."],
        Float64[Array, " ..."],
    ] = lax.fori_loop(
        1,
        steps,
        jax.checkpoint(body),
        initial_state,
    )
    result: Float64[Array, " ..."] = final_state[1]
    return result


def _regular_value(
    order: int,
    eta: Float64[Array, " ..."],
    rho: Float64[Array, " ..."],
) -> Float64[Array, " ..."]:
    """Evaluate F_l from the origin or its Numerov continuation."""
    series_value: Float64[Array, " ..."]
    series_value, _ = _regular_origin_state(order, eta, rho)
    outgoing: Complex128[Array, " ..."]
    outgoing, _ = _outgoing_asymptotic_state(order, eta, rho)
    propagated_rho: Float64[Array, " ..."] = jnp.clip(
        rho,
        4.0,
        20.0,
    )
    coordinate_start: Float64[Array, " ..."] = jnp.full_like(
        propagated_rho,
        math.log(4.0),
    )
    coordinate_end: Float64[Array, " ..."] = jnp.log(propagated_rho)
    rho_start: Float64[Array, " ..."] = jnp.full_like(rho, 4.0)
    start_value: Float64[Array, " ..."]
    start_value, _ = _regular_origin_state(order, eta, rho_start)
    transformed_start: Float64[Array, " ..."] = start_value / jnp.sqrt(
        rho_start
    )

    def propagate(steps: int) -> Float64[Array, " ..."]:
        step: Float64[Array, " ..."] = (
            coordinate_end - coordinate_start
        ) / steps
        rho_next: Float64[Array, " ..."] = jnp.exp(coordinate_start + step)
        next_value: Float64[Array, " ..."]
        next_value, _ = _regular_origin_state(order, eta, rho_next)
        transformed_next: Float64[Array, " ..."] = next_value / jnp.sqrt(
            rho_next
        )
        result: Float64[Array, " ..."] = _numerov_endpoint(
            order,
            eta,
            coordinate_start,
            coordinate_end,
            transformed_start,
            transformed_next,
            1.0,
            steps,
        )
        return result

    propagated_transformed: Float64[Array, " ..."] = propagate(8192)
    propagated_value: Float64[Array, " ..."] = (
        jnp.sqrt(propagated_rho) * propagated_transformed
    )
    value: Float64[Array, " ..."] = jnp.where(
        rho <= 4.0,
        series_value,
        jnp.where(rho >= 20.0, jnp.imag(outgoing), propagated_value),
    )
    return value


def _irregular_value(
    order: int,
    eta: Float64[Array, " ..."],
    rho: Float64[Array, " ..."],
) -> Float64[Array, " ..."]:
    """Evaluate G_l by backward propagation of the outgoing solution."""
    asymptotic_value: Complex128[Array, " ..."]
    asymptotic_value, _ = _outgoing_asymptotic_state(order, eta, rho)
    propagated_rho: Float64[Array, " ..."] = jnp.clip(
        rho,
        0.1,
        20.0,
    )
    coordinate_start: Float64[Array, " ..."] = jnp.full_like(
        rho,
        -math.log(20.0),
    )
    coordinate_end: Float64[Array, " ..."] = -jnp.log(propagated_rho)
    rho_start: Float64[Array, " ..."] = jnp.full_like(
        rho,
        20.0,
    )
    outgoing_start: Complex128[Array, " ..."]
    outgoing_start, _ = _outgoing_asymptotic_state(
        order,
        eta,
        rho_start,
    )
    transformed_start: Float64[Array, " ..."] = jnp.real(
        outgoing_start
    ) / jnp.sqrt(rho_start)

    def propagate(steps: int) -> Float64[Array, " ..."]:
        step: Float64[Array, " ..."] = (
            coordinate_end - coordinate_start
        ) / steps
        rho_next: Float64[Array, " ..."] = jnp.exp(-(coordinate_start + step))
        outgoing_next: Complex128[Array, " ..."]
        outgoing_next, _ = _outgoing_asymptotic_state(
            order,
            eta,
            rho_next,
        )
        transformed_next: Float64[Array, " ..."] = jnp.real(
            outgoing_next
        ) / jnp.sqrt(rho_next)
        result: Float64[Array, " ..."] = _numerov_endpoint(
            order,
            eta,
            coordinate_start,
            coordinate_end,
            transformed_start,
            transformed_next,
            -1.0,
            steps,
        )
        return result

    propagated_transformed: Float64[Array, " ..."] = propagate(8192)
    propagated_value: Float64[Array, " ..."] = (
        jnp.sqrt(propagated_rho) * propagated_transformed
    )
    local_value: Float64[Array, " ..."]
    local_value, _ = _irregular_origin_state(
        order,
        eta,
        rho,
        propagated_value,
    )
    value: Float64[Array, " ..."] = jnp.where(
        rho >= 20.0,
        jnp.real(asymptotic_value),
        jnp.where(rho <= 0.1, local_value, propagated_value),
    )
    return value


def _coulomb_values(
    order: int,
    eta: Float64[Array, " ..."],
    rho: Float64[Array, " ..."],
) -> Float64[Array, "2 ..."]:
    """Return the normalized regular and irregular Coulomb values."""
    regular: Float64[Array, " ..."] = _regular_value(order, eta, rho)
    irregular: Float64[Array, " ..."] = _irregular_value(order, eta, rho)
    values: Float64[Array, "2 ..."] = jnp.stack((regular, irregular))
    return values


def _coulomb_log_radius_rhs(
    state: Float64[Array, " 2"],
    coordinate: Float64[Array, ""],
    eta: Float64[Array, ""],
    order: int,
    direction: float,
) -> Float64[Array, " 2"]:
    """Return the first-order Coulomb system in signed log radius."""
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
    """Propagate the Coulomb system with fixed Dormand--Prince fifth order."""
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
        index: Array, current: Float64[Array, " 2"]
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
    """Evaluate one high-accuracy value-and-derivative row."""
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
    """Evaluate adaptive values before the custom derivative rule."""
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
    """Evaluate adaptive values with a fixed-Numerov differentiation rule."""
    values: Float64[Array, "4 ..."] = _accurate_coulomb_values_impl(
        order,
        eta,
        rho,
    )
    result: Float64[Array, "4 ..."] = lax.stop_gradient(values)
    return result  # noqa: RET504


@_accurate_coulomb_values.defjvp
def _accurate_coulomb_values_jvp(
    order: int,
    primals: tuple[Float64[Array, " ..."], Float64[Array, " ..."]],
    tangents: tuple[Float64[Array, " ..."], Float64[Array, " ..."]],
) -> tuple[Float64[Array, "4 ..."], Float64[Array, "4 ..."]]:
    """Differentiate adaptive values through the fixed Numerov solver."""
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
    result: tuple[Float64[Array, "4 ..."], Float64[Array, "4 ..."]] = (
        values,
        tangent_values,
    )
    return result


def _plane_coulomb_rows(
    order: int,
    rho: Float64[Array, " ..."],
) -> tuple[
    Float64[Array, " ..."],
    Float64[Array, " ..."],
    Float64[Array, " ..."],
    Float64[Array, " ..."],
]:
    """Return the exact eta-zero F, G, and derivative rows."""
    irregular_zero: Float64[Array, " ..."] = jnp.cos(rho)
    if order == 0:
        regular: Float64[Array, " ..."] = jnp.sin(rho)
        regular_derivative: Float64[Array, " ..."] = jnp.cos(rho)
        irregular: Float64[Array, " ..."] = irregular_zero
        irregular_derivative: Float64[Array, " ..."] = -jnp.sin(rho)
    else:
        regular = rho * spherical_bessel_jl(order, rho)
        regular_derivative = spherical_bessel_jl(
            order,
            rho,
        ) + rho * spherical_bessel_jl_derivative(order, rho)
        irregular_previous: Float64[Array, " ..."] = irregular_zero
        irregular = jnp.cos(rho) / rho + jnp.sin(rho)
        irregular_next: Float64[Array, " ..."]
        recurrence_order: int
        for recurrence_order in range(1, order):
            irregular_next = (
                2 * recurrence_order + 1
            ) * irregular / rho - irregular_previous
            irregular_previous = irregular
            irregular = irregular_next
        irregular_next = (2 * order + 1) * irregular / rho - irregular_previous
        irregular_derivative = (order + 1) * irregular / rho - irregular_next
    result: tuple[
        Float64[Array, " ..."],
        Float64[Array, " ..."],
        Float64[Array, " ..."],
        Float64[Array, " ..."],
    ] = (
        regular,
        irregular,
        regular_derivative,
        irregular_derivative,
    )
    return result


@partial(jax.custom_jvp, nondiff_argnums=(0,))
def _accurate_coulomb_values_with_plane_limit(
    order: int,
    eta: Float64[Array, " ..."],
    rho: Float64[Array, " ..."],
) -> Float64[Array, "4 ..."]:
    """Apply the exact eta-zero plane limit to adaptive Coulomb rows."""
    values: Float64[Array, "4 ..."] = _accurate_coulomb_values(order, eta, rho)
    plane_values: Float64[Array, "4 ..."] = jnp.stack(
        _plane_coulomb_rows(order, rho)
    )
    eta_zero: Array = eta == 0.0
    result: Float64[Array, "4 ..."] = jnp.where(
        eta_zero[None, ...],
        plane_values,
        values,
    )
    return result


@_accurate_coulomb_values_with_plane_limit.defjvp
def _accurate_coulomb_values_with_plane_limit_jvp(
    order: int,
    primals: tuple[Float64[Array, " ..."], Float64[Array, " ..."]],
    tangents: tuple[Float64[Array, " ..."], Float64[Array, " ..."]],
) -> tuple[Float64[Array, "4 ..."], Float64[Array, "4 ..."]]:
    """Preserve eta tangents and exact plane-limit rho tangents."""
    eta: Float64[Array, " ..."]
    rho: Float64[Array, " ..."]
    eta_tangent: Float64[Array, " ..."]
    rho_tangent: Float64[Array, " ..."]
    eta, rho = primals
    eta_tangent, rho_tangent = tangents
    values: Float64[Array, "4 ..."]
    values_tangent: Float64[Array, "4 ..."]
    values, values_tangent = jax.jvp(
        lambda first, second: _accurate_coulomb_values(
            order,
            first,
            second,
        ),
        (eta, rho),
        (eta_tangent, rho_tangent),
    )
    plane_values: Float64[Array, "4 ..."]
    plane_tangent: Float64[Array, "4 ..."]
    plane_values, plane_tangent = jax.jvp(
        lambda argument: jnp.stack(_plane_coulomb_rows(order, argument)),
        (rho,),
        (rho_tangent,),
    )
    ode_factor: Float64[Array, " ..."] = (
        1.0 - 2.0 * eta / rho - float(order * (order + 1)) / rho**2
    )
    solver_rho_derivative: Float64[Array, "4 ..."] = jnp.stack(
        (
            values[2],
            values[3],
            -ode_factor * values[0],
            -ode_factor * values[1],
        )
    )
    plane_limit_tangent: Float64[Array, "4 ..."] = (
        values_tangent - solver_rho_derivative * rho_tangent + plane_tangent
    )
    eta_zero: Array = eta == 0.0
    result_values: Float64[Array, "4 ..."] = jnp.where(
        eta_zero[None, ...],
        plane_values,
        values,
    )
    result_tangent: Float64[Array, "4 ..."] = jnp.where(
        eta_zero[None, ...],
        plane_limit_tangent,
        values_tangent,
    )
    result: tuple[Float64[Array, "4 ..."], Float64[Array, "4 ..."]] = (
        result_values,
        result_tangent,
    )
    return result


def _normalized_coulomb_rows_impl(
    order: int,
    eta: Float64[Array, " ..."],
    rho: Float64[Array, " ..."],
) -> Float64[Array, "4 ..."]:
    """Return Wronskian-normalized Coulomb rows before the public JVP."""
    values: Float64[Array, "4 ..."] = (
        _accurate_coulomb_values_with_plane_limit(
            order,
            eta,
            rho,
        )
    )
    regular: Float64[Array, " ..."] = values[0]
    irregular: Float64[Array, " ..."] = values[1]
    regular_derivative: Float64[Array, " ..."] = values[2]
    irregular_derivative: Float64[Array, " ..."] = values[3]
    wronskian: Float64[Array, " ..."] = (
        regular_derivative * irregular - regular * irregular_derivative
    )
    derivative_norm: Float64[Array, " ..."] = regular**2 + irregular**2
    correction: Float64[Array, " ..."] = (1.0 - wronskian) / derivative_norm
    normalized: Float64[Array, "4 ..."] = jnp.stack(
        (
            regular,
            irregular,
            regular_derivative + correction * irregular,
            irregular_derivative - correction * regular,
        )
    )
    return normalized


@partial(jax.custom_jvp, nondiff_argnums=(0,))
def _normalized_coulomb_rows(
    order: int,
    eta: Float64[Array, " ..."],
    rho: Float64[Array, " ..."],
) -> Float64[Array, "4 ..."]:
    """Preserve both the normalized Wronskian and Coulomb ODE tangent."""
    values: Float64[Array, "4 ..."] = _normalized_coulomb_rows_impl(
        order,
        eta,
        rho,
    )
    return values


@partial(_normalized_coulomb_rows.defjvp, symbolic_zeros=True)
def _normalized_coulomb_rows_jvp(
    order: int,
    primals: tuple[Float64[Array, " ..."], Float64[Array, " ..."]],
    tangents: tuple[
        Float64[Array, " ..."] | jax.custom_derivatives.SymbolicZero,
        Float64[Array, " ..."] | jax.custom_derivatives.SymbolicZero,
    ],
) -> tuple[Float64[Array, "4 ..."], Float64[Array, "4 ..."]]:
    """Differentiate eta numerically and rho through the exact ODE system."""
    eta: Float64[Array, " ..."]
    rho: Float64[Array, " ..."]
    eta_tangent: Float64[Array, " ..."] | jax.custom_derivatives.SymbolicZero
    rho_tangent: Float64[Array, " ..."] | jax.custom_derivatives.SymbolicZero
    eta, rho = primals
    eta_tangent, rho_tangent = tangents
    values: Float64[Array, "4 ..."] = _normalized_coulomb_rows_impl(
        order,
        eta,
        rho,
    )
    eta_contribution: Float64[Array, "4 ..."]
    if isinstance(eta_tangent, jax.custom_derivatives.SymbolicZero):
        eta_contribution = jnp.zeros_like(values)
    else:
        eta_derivative: Float64[Array, "4 ..."] = jax.jvp(
            lambda argument: _normalized_coulomb_rows_impl(
                order,
                argument,
                rho,
            ),
            (eta,),
            (jnp.ones_like(eta),),
        )[1]
        eta_contribution = eta_derivative * eta_tangent
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
    rho_contribution: Float64[Array, "4 ..."] = (
        jnp.zeros_like(values)
        if isinstance(rho_tangent, jax.custom_derivatives.SymbolicZero)
        else rho_derivative * rho_tangent
    )
    tangent_values: Float64[Array, "4 ..."] = (
        eta_contribution + rho_contribution
    )
    result: tuple[Float64[Array, "4 ..."], Float64[Array, "4 ..."]] = (
        values,
        tangent_values,
    )
    return result


@jaxtyped(typechecker=beartype)
def coulomb_phase_shift(  # noqa: DOC502
    order: int,
    eta: Float64[Array, " ..."],
) -> Float64[Array, " ..."]:
    r"""Evaluate the continuous Coulomb arg-Gamma phase.

    The shifted Stirling expansion and downward recurrence remain on one
    continuous branch anchored by ``sigma_l(0) == 0``.

    :see: :class:`~.test_coulomb.TestCoulombPhaseShift`

    Parameters
    ----------
    order : int
        Static angular momentum from zero through five.
    eta : Float64[Array, " ..."]
        Dimensionless Sommerfeld parameter.

    Returns
    -------
    phase : Float64[Array, " ..."]
        Continuous phase in radians.

    Raises
    ------
    ValueError
        If ``order`` is outside the certified range.
    EquinoxRuntimeError
        If ``eta`` is nonfinite or outside ``[-3, 3]``.

    Notes
    -----
    Recurrence moves the argument to a converged Stirling domain and returns
    on the same analytic branch.
    """
    _validate_order(order)
    eta_array: Float64[Array, " ..."] = jnp.asarray(eta, dtype=jnp.float64)
    eta_array = eqx.error_if(
        eta_array,
        ~jnp.all(jnp.isfinite(eta_array)) | jnp.any(jnp.abs(eta_array) > 3.0),
        "eta must be finite and lie in [-3, 3]",
    )
    phase: Float64[Array, " ..."] = _coulomb_phase_unchecked(order, eta_array)
    return phase


@jaxtyped(typechecker=beartype)
def coulomb_fg(  # noqa: DOC503
    order: int,
    eta: Float64[Array, " ..."],
    rho: Float64[Array, " ..."],
) -> tuple[
    Float64[Array, " ..."],
    Float64[Array, " ..."],
    Float64[Array, " ..."],
    Float64[Array, " ..."],
]:
    """Evaluate normalized Coulomb functions and radial derivatives.

    The regular and irregular values share one normalization. Their radial
    derivatives are components of the adaptively propagated ODE state.

    :see: :class:`~.test_coulomb.TestCoulombFg`

    Parameters
    ----------
    order : int
        Static angular momentum from zero through five.
    eta : Float64[Array, " ..."]
        Dimensionless Sommerfeld parameter.
    rho : Float64[Array, " ..."]
        Strictly positive dimensionless radius.

    Returns
    -------
    regular : Float64[Array, " ..."]
        Regular Coulomb function ``F_l``.
    irregular : Float64[Array, " ..."]
        Irregular Coulomb function ``G_l``.
    regular_derivative : Float64[Array, " ..."]
        Derivative ``dF_l / d rho``.
    irregular_derivative : Float64[Array, " ..."]
        Derivative ``dG_l / d rho``.

    Raises
    ------
    ValueError
        If ``order`` is outside the certified range or shapes differ.
    EquinoxRuntimeError
        If the numerical arguments leave the G11 domain.

    Notes
    -----
    The implementation maps the adaptive value solve over every input axis.
    A custom rule provides forward- and reverse-mode differentiation.
    """
    _validate_order(order)
    eta_array: Float64[Array, " ..."] = jnp.asarray(eta, dtype=jnp.float64)
    rho_array: Float64[Array, " ..."] = jnp.asarray(rho, dtype=jnp.float64)
    if eta_array.shape != rho_array.shape:
        message: str = "eta and rho must have identical shapes"
        raise ValueError(message)
    eta_array = eqx.error_if(
        eta_array,
        ~jnp.all(jnp.isfinite(eta_array)) | jnp.any(jnp.abs(eta_array) > 3.0),
        "eta must be finite and lie in [-3, 3]",
    )
    rho_array = eqx.error_if(
        rho_array,
        ~jnp.all(jnp.isfinite(rho_array))
        | jnp.any(rho_array < 1.0e-4)
        | jnp.any(rho_array > 40.0),
        "rho must be finite and lie in [1e-4, 40]",
    )
    values: Float64[Array, "4 ..."] = _normalized_coulomb_rows(
        order,
        eta_array,
        rho_array,
    )
    regular: Float64[Array, " ..."] = values[0]
    irregular: Float64[Array, " ..."] = values[1]
    regular_derivative: Float64[Array, " ..."] = values[2]
    irregular_derivative: Float64[Array, " ..."] = values[3]
    result: tuple[
        Float64[Array, " ..."],
        Float64[Array, " ..."],
        Float64[Array, " ..."],
        Float64[Array, " ..."],
    ] = (
        regular,
        irregular,
        regular_derivative,
        irregular_derivative,
    )
    return result


@jaxtyped(typechecker=beartype)
def final_state_radial(  # noqa: DOC503
    order: int,
    k_bohr_inv: Float64[Array, " ..."],
    r_bohr: Float64[Array, " n_r"],
    spec: FinalStateSpec,
) -> Complex128[Array, "... n_r"]:
    """Evaluate a plane-wave or Coulomb final-state radial row.

    Coulomb mode uses ``eta = -effective_charge / k`` and the normalized
    convention ``exp(i sigma_l) F_l(eta, k r) / (k r)``. The implementation
    uses the analytic regular limit at the origin.

    :see: :class:`~.test_coulomb.TestFinalStateRadial`

    Parameters
    ----------
    order : int
        Static final-state angular momentum.
    k_bohr_inv : Float64[Array, " ..."]
        Momentum in inverse Bohr.
    r_bohr : Float64[Array, " n_r"]
        Nonnegative radii in Bohr.
    spec : FinalStateSpec
        Validated final-state selection.

    Returns
    -------
    values : Complex128[Array, "... n_r"]
        Complex128 radial final-state values.

    Raises
    ------
    ValueError
        For an unsupported static final-state mode.
    EquinoxRuntimeError
        If numerical arguments leave the certified domain.

    Notes
    -----
    Plane-wave mode calls the spherical-Bessel kernel directly. Coulomb mode
    uses the regular origin series below the Numerov domain.
    """
    _validate_order(order)
    momentum: Float64[Array, " ..."] = jnp.asarray(
        k_bohr_inv,
        dtype=jnp.float64,
    )
    radius: Float64[Array, " n_r"] = jnp.asarray(r_bohr, dtype=jnp.float64)
    radius = eqx.error_if(
        radius,
        ~jnp.all(jnp.isfinite(radius)) | jnp.any(radius < 0.0),
        "r_bohr must be finite and nonnegative",
    )
    if spec.mode == "plane_wave":
        rho_plane: Float64[Array, "... n_r"] = momentum[..., None] * radius
        plane_values: Float64[Array, "... n_r"] = spherical_bessel_jl(
            order,
            rho_plane,
        )
        values: Complex128[Array, "... n_r"] = plane_values.astype(
            jnp.complex128
        )
        return values
    if spec.mode != "coulomb":
        message: str = "unsupported final-state mode"
        raise ValueError(message)
    momentum = eqx.error_if(
        momentum,
        ~jnp.all(jnp.isfinite(momentum)) | jnp.any(momentum <= 0.0),
        "Coulomb final states require finite positive momentum",
    )
    eta: Float64[Array, " ..."] = -spec.effective_charge / momentum
    eta = eqx.error_if(
        eta,
        jnp.any(jnp.abs(eta) > 3.0),
        "effective charge and momentum produce eta outside [-3, 3]",
    )
    rho: Float64[Array, "... n_r"] = momentum[..., None] * radius
    eta_grid: Float64[Array, "... n_r"] = jnp.broadcast_to(
        eta[..., None],
        rho.shape,
    )
    safe_rho: Float64[Array, "... n_r"] = jnp.where(
        rho == 0.0,
        jnp.full_like(rho, 1.0e-4),
        rho,
    )
    regular: Float64[Array, "... n_r"] = _regular_value(
        order,
        eta_grid,
        safe_rho,
    )
    normalization: Float64[Array, " ..."] = jnp.exp(
        _log_coulomb_normalization(order, eta)
    )
    origin_limit: Float64[Array, "... n_r"] = jnp.where(
        order == 0,
        jnp.broadcast_to(normalization[..., None], rho.shape),
        jnp.zeros_like(rho),
    )
    radial_real: Float64[Array, "... n_r"] = jnp.where(
        rho == 0.0,
        origin_limit,
        regular / safe_rho,
    )
    plane_limit: Float64[Array, "... n_r"] = spherical_bessel_jl(order, rho)
    radial_real = jnp.where(
        eta_grid == 0.0,
        plane_limit + radial_real - lax.stop_gradient(radial_real),
        radial_real,
    )
    phase: Float64[Array, " ..."] = coulomb_phase_shift(order, eta)
    values: Complex128[Array, "... n_r"] = (
        jnp.exp(1j * phase[..., None]) * radial_real
    )
    return values  # noqa: RET504


__all__: list[str] = [
    "coulomb_fg",
    "coulomb_phase_shift",
    "final_state_radial",
]
