# ruff: noqa: PLR2004
"""Propagate Coulomb radial states with the Numerov method.

Extended Summary
----------------
This module uses bounded static Numerov propagation.
It produces final-state radial rows.

Routine Listings
----------------
:func:`final_state_radial`
    Evaluate a plane-wave or Coulomb final-state radial row.
"""

import math

import equinox as eqx
import jax
import jax.numpy as jnp
from beartype import beartype
from beartype.typing import Tuple
from jax import lax
from jaxtyping import Array, Complex128, Float64, Int64, jaxtyped

from diffpes.types import FinalStateSpec

from .bessel import spherical_bessel_jl
from .coulomb_asymptotics import (
    _irregular_origin_state,
    _log_coulomb_normalization,
    _outgoing_asymptotic_state,
    _regular_origin_state,
    _validate_order,
    coulomb_phase_shift,
)


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
    """PRIVATE: Propagate one transformed Coulomb value on a uniform log
    grid.

    Implementation Logic
    --------------------
    1. **Form the transformed potential**::

           result = radius**2 - 2 * eta * radius - order * (order + 1) - 0.25

       The Liouville substitution gives ``w'' + q w = 0`` in log radius.

    2. **Apply the Numerov recurrence**::

           final_state = lax.fori_loop(1, steps, jax.checkpoint(body), state)

       Two seed values start the three-point recurrence. Checkpointing bounds
       the memory use during reverse-mode differentiation.

    Parameters
    ----------
    order : int
        Static angular momentum from 0 through 5.
    eta : Float64[Array, " ..."]
        Dimensionless Sommerfeld parameter.
    coordinate_start : Float64[Array, " ..."]
        Signed log-radius start coordinate.
    coordinate_end : Float64[Array, " ..."]
        Signed log-radius end coordinate.
    value_start : Float64[Array, " ..."]
        Transformed value ``u / sqrt(rho)`` at the start coordinate.
    value_next : Float64[Array, " ..."]
        Transformed value at the start coordinate plus one step.
    direction : float
        Static sign of the log map; ``rho = exp(direction *
        coordinate)``.
    steps : int
        Static step count of the uniform grid.

    Returns
    -------
    result : Float64[Array, " ..."]
        Transformed value at the end coordinate.

    """
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
        index: Int64[Array, ""],
        state: Tuple[
            Float64[Array, " ..."],
            Float64[Array, " ..."],
            Float64[Array, " ..."],
            Float64[Array, " ..."],
        ],
    ) -> Tuple[
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
        next_state: Tuple[
            Float64[Array, " ..."],
            Float64[Array, " ..."],
            Float64[Array, " ..."],
            Float64[Array, " ..."],
        ] = (current, next_value, q_current, q_next)
        return next_state

    initial_state: Tuple[
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
    final_state: Tuple[
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
    r"""PRIVATE: Evaluate F_l from the origin or its Numerov continuation.

    Implementation Logic
    --------------------
    1. **Evaluate the three region candidates**::

           propagated_transformed = propagate(8192)

       The origin series covers ``rho <= 4``. Numerov propagation covers the
       middle region, and the outgoing solution covers ``rho >= 20``.

    2. **Select the active region**::

           value = jnp.where(rho <= 4, series_value, continued_value)

       Clipped candidate arguments keep every inactive branch finite.

    Parameters
    ----------
    order : int
        Static angular momentum from 0 through 5.
    eta : Float64[Array, " ..."]
        Dimensionless Sommerfeld parameter.
    rho : Float64[Array, " ..."]
        Positive dimensionless radius.

    Returns
    -------
    value : Float64[Array, " ..."]
        Regular Coulomb function :math:`F_l(\eta,\rho)`.

    """
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
    r"""PRIVATE: Evaluate G_l by backward propagation of the outgoing
    solution.

    Implementation Logic
    --------------------
    1. **Evaluate the three region candidates**::

           propagated_transformed = propagate(8192)

       The outgoing solution covers ``rho >= 20``. Inward Numerov propagation
       covers the middle region, and a matched Frobenius row covers the origin.

    2. **Select the active region**::

           value = jnp.where(rho >= 20, asymptotic_value, continued_value)

       The origin row matches the propagated value at ``rho = 0.1``.

    Parameters
    ----------
    order : int
        Static angular momentum from 0 through 5.
    eta : Float64[Array, " ..."]
        Dimensionless Sommerfeld parameter.
    rho : Float64[Array, " ..."]
        Positive dimensionless radius.

    Returns
    -------
    value : Float64[Array, " ..."]
        Irregular Coulomb function :math:`G_l(\eta,\rho)`.

    """
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
    r"""PRIVATE: Return the normalized regular and irregular Coulomb values.

    Parameters
    ----------
    order : int
        Static angular momentum from 0 through 5.
    eta : Float64[Array, " ..."]
        Dimensionless Sommerfeld parameter.
    rho : Float64[Array, " ..."]
        Positive dimensionless radius.

    Returns
    -------
    values : Float64[Array, "2 ..."]
        Stack of :math:`F_l` and :math:`G_l` along a new leading axis.

    Notes
    -----
    Evaluates :func:`_regular_value` and :func:`_irregular_value` on
    the same arguments and stacks the two rows.
    """
    regular: Float64[Array, " ..."] = _regular_value(order, eta, rho)
    irregular: Float64[Array, " ..."] = _irregular_value(order, eta, rho)
    values: Float64[Array, "2 ..."] = jnp.stack((regular, irregular))
    return values


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

    :see: :class:`~.test_coulomb_numerov.TestFinalStateRadial`

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
    return values  # noqa: RET504 -- assign-before-return is required.


__all__: list[str] = [
    "final_state_radial",
]
