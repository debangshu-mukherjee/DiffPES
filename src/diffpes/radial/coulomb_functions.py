# ruff: noqa: PLR2004
"""Evaluate normalized Coulomb radial functions.

Extended Summary
----------------
This module normalizes Coulomb rows.
It joins the charged and plane-wave limits.

Routine Listings
----------------
:func:`coulomb_fg`
    Evaluate normalized Coulomb functions and radial derivatives.
"""

from functools import partial

import equinox as eqx
import jax
import jax.numpy as jnp
from beartype import beartype
from beartype.typing import Tuple
from jaxtyping import Array, Bool, Float64, jaxtyped

from .bessel import spherical_bessel_jl, spherical_bessel_jl_derivative
from .coulomb_asymptotics import _validate_order
from .coulomb_ode import _accurate_coulomb_values


def _plane_coulomb_rows(
    order: int,
    rho: Float64[Array, " ..."],
) -> Tuple[
    Float64[Array, " ..."],
    Float64[Array, " ..."],
    Float64[Array, " ..."],
    Float64[Array, " ..."],
]:
    r"""PRIVATE: Return the exact eta-zero F, G, and derivative rows.

    Implementation Logic
    --------------------
    1. **Evaluate the regular Riccati--Bessel rows**::

           regular = rho * spherical_bessel_jl(order, rho)

       At :math:`\eta = 0`, the regular Coulomb function has this exact form.

    2. **Propagate the irregular Riccati--Bessel row**::

           irregular_next = (
               (2 * order + 1) * irregular / rho - irregular_previous
           )

       Closed forms supply the order-zero and order-one anchors.

    3. **Assemble the value and derivative rows**::

           result = (
               regular,
               irregular,
               regular_derivative,
               irregular_derivative,
           )

       Order zero uses the exact sine and cosine forms directly.

    Parameters
    ----------
    order : int
        Static angular momentum from 0 through 5.
    rho : Float64[Array, " ..."]
        Positive dimensionless radius.

    Returns
    -------
    result : tuple of Float64[Array, " ..."]
        Rows ``F``, ``G``, ``dF/drho``, ``dG/drho`` at
        :math:`\eta = 0`.

    """
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
    result: Tuple[
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
    """PRIVATE: Apply the exact eta-zero plane limit to adaptive Coulomb
    rows.

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
        Rows ``F``, ``G``, ``dF/drho``, ``dG/drho`` with exact
        Riccati--Bessel values wherever ``eta == 0``.

    Notes
    -----
    This is the primal of a :func:`jax.custom_jvp`.  It evaluates the
    solver rows and the exact plane rows and selects per element with
    :func:`jnp.where` on the broadcast ``eta == 0`` mask.
    """
    values: Float64[Array, "4 ..."] = _accurate_coulomb_values(order, eta, rho)
    plane_values: Float64[Array, "4 ..."] = jnp.stack(
        _plane_coulomb_rows(order, rho)
    )
    eta_zero: Bool[Array, " ..."] = eta == 0.0
    result: Float64[Array, "4 ..."] = jnp.where(
        eta_zero[None, ...],
        plane_values,
        values,
    )
    return result


@_accurate_coulomb_values_with_plane_limit.defjvp
def _accurate_coulomb_values_with_plane_limit_jvp(
    order: int,
    primals: Tuple[Float64[Array, " ..."], Float64[Array, " ..."]],
    tangents: Tuple[Float64[Array, " ..."], Float64[Array, " ..."]],
) -> Tuple[Float64[Array, "4 ..."], Float64[Array, "4 ..."]]:
    """PRIVATE: Preserve eta tangents and exact plane-limit rho tangents.

    Implementation Logic
    --------------------
    1. **Differentiate both candidate rows**::

           values, values_tangent = jax.jvp(
               lambda first, second: _accurate_coulomb_values(
                   order,
                   first,
                   second,
               ),
               (eta, rho),
               (eta_tangent, rho_tangent),
           )

       A second JVP differentiates the exact plane rows with respect to radius.

    2. **Replace the radial tangent at eta zero**::

           plane_limit_tangent = (
               values_tangent
               - solver_rho_derivative * rho_tangent
               + plane_tangent
           )

       The subtraction retains the solver's Sommerfeld-parameter contribution.

    3. **Select the exact plane limit**::

           result_tangent = jnp.where(
               eta_zero[None, ...],
               plane_limit_tangent,
               values_tangent,
           )

       Away from zero, the adaptive solver tangent passes through unchanged.

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
        Selected primal rows and their selected tangent rows.

    """
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
    eta_zero: Bool[Array, " ..."] = eta == 0.0
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
    result: Tuple[Float64[Array, "4 ..."], Float64[Array, "4 ..."]] = (
        result_values,
        result_tangent,
    )
    return result


def _normalized_coulomb_rows_impl(
    order: int,
    eta: Float64[Array, " ..."],
    rho: Float64[Array, " ..."],
) -> Float64[Array, "4 ..."]:
    """PRIVATE: Return Wronskian-normalized Coulomb rows before the public
    JVP.

    Implementation Logic
    --------------------
    1. **Measure the Wronskian error**::

           correction = (1 - wronskian) / (regular**2 + irregular**2)

       The denominator is the squared norm of the two value rows.

    2. **Correct only the derivative rows**::

           normalized = jnp.stack(
               (
                   regular,
                   irregular,
                   regular_derivative + correction * irregular,
                   irregular_derivative - correction * regular,
               )
           )

       The added terms change the Wronskian by exactly ``1 - wronskian``.

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
    normalized : Float64[Array, "4 ..."]
        Rows ``F``, ``G``, ``dF/drho``, ``dG/drho`` with the exact
        Wronskian ``F' G - F G' = 1``.

    """
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
    """PRIVATE: Preserve both the normalized Wronskian and Coulomb ODE
    tangent.

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
        Wronskian-normalized rows ``F``, ``G``, ``dF/drho``,
        ``dG/drho``.

    Notes
    -----
    This is the primal of a :func:`jax.custom_jvp`.  It forwards to
    :func:`_normalized_coulomb_rows_impl`; the registered rule
    :func:`_normalized_coulomb_rows_jvp` supplies all derivatives.
    """
    values: Float64[Array, "4 ..."] = _normalized_coulomb_rows_impl(
        order,
        eta,
        rho,
    )
    return values


@partial(_normalized_coulomb_rows.defjvp, symbolic_zeros=True)
def _normalized_coulomb_rows_jvp(
    order: int,
    primals: Tuple[Float64[Array, " ..."], Float64[Array, " ..."]],
    tangents: Tuple[
        Float64[Array, " ..."] | jax.custom_derivatives.SymbolicZero,
        Float64[Array, " ..."] | jax.custom_derivatives.SymbolicZero,
    ],
) -> Tuple[Float64[Array, "4 ..."], Float64[Array, "4 ..."]]:
    """PRIVATE: Differentiate eta numerically and rho through the exact ODE
    system.

    Implementation Logic
    --------------------
    1. **Form the Sommerfeld-parameter contribution**::

           eta_contribution = eta_derivative * eta_tangent

       A JVP through the normalized implementation supplies the derivative.
       A symbolic-zero tangent supplies an exact zero block.

    2. **Form the radial contribution**::

           rho_contribution = rho_derivative * rho_tangent

       The stored derivatives and the Coulomb ODE supply the exact derivative.

    3. **Combine both contributions**::

           tangent_values = eta_contribution + rho_contribution

       The returned pair contains the normalized rows and their joint tangent.

    Parameters
    ----------
    order : int
        Static angular momentum from 0 through 5.
    primals : Tuple[Float64[Array, " ..."], Float64[Array, " ..."]]
        Pair ``(eta, rho)`` of primal arguments.
    tangents : Tuple[Float64[Array, " ..."] | \
jax.custom_derivatives.SymbolicZero, Float64[Array, " ..."] | \
jax.custom_derivatives.SymbolicZero]
        Pair of input tangents; symbolic zeros mark inactive
        arguments.

    Returns
    -------
    result : Tuple[Float64[Array, "4 ..."], Float64[Array, "4 ..."]]
        Primal rows and their joint tangent rows.

    """
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
    result: Tuple[Float64[Array, "4 ..."], Float64[Array, "4 ..."]] = (
        values,
        tangent_values,
    )
    return result


@jaxtyped(typechecker=beartype)
def coulomb_fg(  # noqa: DOC503
    order: int,
    eta: Float64[Array, " ..."],
    rho: Float64[Array, " ..."],
) -> Tuple[
    Float64[Array, " ..."],
    Float64[Array, " ..."],
    Float64[Array, " ..."],
    Float64[Array, " ..."],
]:
    """Evaluate normalized Coulomb functions and radial derivatives.

    The regular and irregular values share one normalization. Their radial
    derivatives are components of the adaptively propagated ODE state.

    :see: :class:`~.test_coulomb_functions.TestCoulombFg`

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
        If the numerical arguments leave the declared numerical domain.

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
    result: Tuple[
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


__all__: list[str] = [
    "coulomb_fg",
]
