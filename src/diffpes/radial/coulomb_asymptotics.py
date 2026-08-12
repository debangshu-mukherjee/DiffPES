# ruff: noqa: PLR2004
"""Evaluate Coulomb asymptotic states and phase shifts.

Extended Summary
----------------
This module owns the origin series and the outgoing asymptote.
It also owns the continuous Coulomb phase.

Routine Listings
----------------
:func:`coulomb_phase_shift`
    Evaluate the continuous Coulomb arg-Gamma phase.
"""

import math

import equinox as eqx
import jax.numpy as jnp
from beartype import beartype
from beartype.typing import List, Tuple
from jaxtyping import Array, Bool, Complex128, Float64, jaxtyped


def _validate_order(order: int) -> None:
    """PRIVATE: Validate one static Coulomb angular momentum.

    Parameters
    ----------
    order : int
        Static angular momentum to check.

    Raises
    ------
    ValueError
        If ``order`` is not an integer from 0 through 5.

    Notes
    -----
    Rejects booleans and other integer subtypes with the exact
    ``type`` comparison, then checks the certified range.
    """
    if type(order) is not int or not 0 <= order <= 5:
        message: str = "Coulomb order must be an integer from 0 through 5"
        raise ValueError(message)


def _complex_log_gamma_shifted(
    value: Complex128[Array, " ..."],
) -> Complex128[Array, " ..."]:
    """PRIVATE: Evaluate log Gamma by recurrence to a converged Stirling
    domain.

    Implementation Logic
    --------------------
    1. **Evaluate the shifted Stirling series**::

           shifted = value + 20

       Ten Bernoulli terms approximate ``log Gamma`` in the converged domain.

    2. **Remove the shift**::

           result = result - jnp.log(value + recurrence_index)

       Twenty recurrence steps restore the requested argument. They keep the
       imaginary part on one analytic branch.

    Parameters
    ----------
    value : Complex128[Array, " ..."]
        Complex arguments with a positive real part.

    Returns
    -------
    result : Complex128[Array, " ..."]
        Principal-branch ``log Gamma(value)`` on one continuous branch.

    """
    shifted: Complex128[Array, " ..."] = value + 20
    result: Complex128[Array, " ..."] = (
        (shifted - 0.5) * jnp.log(shifted)
        - shifted
        + 0.5 * math.log(2.0 * math.pi)
    )
    bernoulli_values: Tuple[float, ...] = (
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
    r"""PRIVATE: Return log C_l(eta) for the regular Coulomb solution.

    Implementation Logic
    --------------------
    1. **Compute the order-zero Gamma factor**::

           gamma_one_abs_squared = jnp.where(small, series_ratio, direct_ratio)

       The even series avoids the removable ``0/0`` form near zero.

    2. **Assemble the logarithmic prefactors**::

           result = order * log(2) - scaled_eta / 2 + log(gamma_abs) / 2

       The expression also includes the factorial normalization in log space.

    3. **Raise the angular order**::

           result = result + 0.5 * jnp.log(recurrence_index**2 + eta**2)

       The recurrence builds :math:`|\Gamma(l+1+i\eta)|` through ``order``.

    Parameters
    ----------
    order : int
        Static angular momentum from 0 through 5.
    eta : Float64[Array, " ..."]
        Dimensionless Sommerfeld parameter.

    Returns
    -------
    result : Float64[Array, " ..."]
        Natural logarithm of the Gamow normalization
        :math:`C_l(\eta)`.

    """
    scaled_eta: Float64[Array, " ..."] = math.pi * eta
    small: Bool[Array, " ..."] = jnp.abs(scaled_eta) < 1.0e-4
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
    r"""PRIVATE: Evaluate the continuous phase for an internal recurrence
    order.

    Parameters
    ----------
    order : int
        Static angular momentum from 0 through 5.
    eta : Float64[Array, " ..."]
        Dimensionless Sommerfeld parameter.

    Returns
    -------
    phase : Float64[Array, " ..."]
        Coulomb phase :math:`\sigma_l(\eta)` in radians on one
        continuous branch.

    Notes
    -----
    Takes the imaginary part of the shifted log-Gamma evaluation at
    :math:`1 + i\eta` and applies the upward recurrence
    ``sigma_l = sigma_0 + sum(arctan2(eta, j))`` for
    ``j = 1 .. order``.  No domain check runs here; the public wrapper
    validates ``eta``.
    """
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
) -> Tuple[Float64[Array, " ..."], Float64[Array, " ..."]]:
    r"""PRIVATE: Evaluate regular origin series and its rho derivative.

    Implementation Logic
    --------------------
    1. **Build the Frobenius coefficients**::

           coefficient = (2 * eta * previous - previous_two) / denominator

       The 64 coefficients start with ``a_0 = 1``.

    2. **Evaluate the value and derivative series**::

           unnormalized = unnormalized + coefficient * rho**power

       The second accumulation differentiates every power term analytically.

    3. **Apply the Gamow normalization**::

           value = normalization * unnormalized

       The same normalization multiplies the value and its derivative.

    Parameters
    ----------
    order : int
        Static angular momentum from 0 through 5.
    eta : Float64[Array, " ..."]
        Dimensionless Sommerfeld parameter.
    rho : Float64[Array, " ..."]
        Dimensionless radius inside the series convergence region.

    Returns
    -------
    result : Tuple[Float64[Array, " ..."], Float64[Array, " ..."]]
        Pair of :math:`F_l(\eta,\rho)` and its :math:`\rho`
        derivative.

    """
    coefficients: List[Float64[Array, " ..."]] = [jnp.ones_like(eta)]
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
    result: Tuple[Float64[Array, " ..."], Float64[Array, " ..."]] = (
        value,
        derivative,
    )
    return result


def _irregular_origin_state(
    order: int,
    eta: Float64[Array, " ..."],
    rho: Float64[Array, " ..."],
    matched_irregular: Float64[Array, " ..."],
) -> Tuple[Float64[Array, " ..."], Float64[Array, " ..."]]:
    r"""PRIVATE: Evaluate the logarithmic irregular Frobenius row near the
    origin.

    Implementation Logic
    --------------------
    1. **Invert the squared regular series**::

           inverse_coefficients.append(-inverse)

       Cauchy products produce the series for :math:`1/F(\rho)^2`.

    2. **Integrate the inverse series**::

           primitive_value = primitive(rho)

       Each power integrates analytically. The zero exponent uses a logarithm.

    3. **Match and differentiate the second solution**::

           value = regular * combined_primitive

       A constant matches the propagated value at ``rho = 0.1``. The product
       rule and the inverse-square series give the derivative.

    Parameters
    ----------
    order : int
        Static angular momentum from 0 through 5.
    eta : Float64[Array, " ..."]
        Dimensionless Sommerfeld parameter.
    rho : Float64[Array, " ..."]
        Dimensionless radius at or below the 0.1 switch point.
    matched_irregular : Float64[Array, " ..."]
        Numerov-propagated :math:`G_l` value at the switch radius 0.1
        that fixes the free constant.

    Returns
    -------
    result : Tuple[Float64[Array, " ..."], Float64[Array, " ..."]]
        Pair of :math:`G_l(\eta,\rho)` and its :math:`\rho`
        derivative.

    """
    coefficient_count: int = 32
    regular_coefficients: List[Float64[Array, " ..."]] = [jnp.ones_like(eta)]
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
    squared_coefficients: List[Float64[Array, " ..."]] = []
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
    inverse_coefficients: List[Float64[Array, " ..."]] = [jnp.ones_like(eta)]
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
        return result  # noqa: RET504 -- assign-before-return is required.

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
    result: Tuple[Float64[Array, " ..."], Float64[Array, " ..."]] = (
        value,
        derivative,
    )
    return result


def _outgoing_asymptotic_state(
    order: int,
    eta: Float64[Array, " ..."],
    rho: Float64[Array, " ..."],
) -> Tuple[Complex128[Array, " ..."], Complex128[Array, " ..."]]:
    r"""PRIVATE: Evaluate outgoing H+ and its rho derivative by
    inverse-radius series.

    Implementation Logic
    --------------------
    1. **Evaluate the inverse-radius amplitude**::

           amplitude = amplitude + coefficient * rho ** (-coefficient_index)

       A 31-term recurrence supplies the amplitude and its derivative.

    2. **Apply the Coulomb phase**::

           value = phase_factor * amplitude

       The product rule combines the amplitude derivative with
       ``1 - eta / rho`` for the phase derivative.

    Parameters
    ----------
    order : int
        Static angular momentum from 0 through 5.
    eta : Float64[Array, " ..."]
        Dimensionless Sommerfeld parameter.
    rho : Float64[Array, " ..."]
        Dimensionless radius in the asymptotic region.

    Returns
    -------
    result : Tuple[Complex128[Array, " ..."], Complex128[Array, " ..."]]
        Pair of the outgoing Coulomb--Hankel function
        :math:`H^+_l(\eta,\rho)` and its :math:`\rho` derivative.

    """
    channel_constant: Complex128[Array, " ..."] = (
        1j * eta - eta**2 - order * (order + 1)
    )
    coefficients: List[Complex128[Array, " ..."]] = [
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
    result: Tuple[
        Complex128[Array, " ..."],
        Complex128[Array, " ..."],
    ] = (value, derivative)
    return result


@jaxtyped(typechecker=beartype)
def coulomb_phase_shift(  # noqa: DOC502
    order: int,
    eta: Float64[Array, " ..."],
) -> Float64[Array, " ..."]:
    r"""Evaluate the continuous Coulomb arg-Gamma phase.

    The shifted Stirling expansion and downward recurrence remain on one
    continuous branch anchored by ``sigma_l(0) == 0``.

    :see: :class:`~.test_coulomb_asymptotics.TestCoulombPhaseShift`

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


__all__: list[str] = [
    "coulomb_phase_shift",
]
