r"""Provide named gradient-safe elementary operations.

Extended Summary
----------------
This module centralizes guarded elementary operations used by differentiable
physics paths. Each helper sanitizes the input to an unsafe branch before an
outer ``jnp.where`` selects the documented guard value. This double-``where``
pattern prevents a NaN or infinity produced by an inactive branch from
polluting reverse-mode gradients, as described in the JAX FAQ.

Routine Listings
----------------
:func:`safe_arccos`
    Evaluate arccos with saturated values and zero boundary gradients.
:func:`safe_arctan2`
    Evaluate arctan2 with a zero value and gradient at the origin.
:func:`safe_divide`
    Divide with a fallback and zero quotient gradients at zero denominators.
:func:`safe_log`
    Evaluate log with a finite floor and zero gradients below it.
:func:`safe_norm`
    Compute a Euclidean norm with a zero gradient at zero vectors.
:func:`safe_power`
    Raise positive inputs to a power and return zero otherwise.
:func:`safe_sqrt`
    Evaluate sqrt on positive inputs and return zero otherwise.

Notes
-----
These helpers are boundary-convention primitives, not domain validators. They
intentionally return finite values on their guarded sets and do not signal
that an input is scientifically invalid. Every caller must validate any
physical domain restriction before invoking them. Use a helper only when its
guarded value and subgradient belong to the caller's contract. Never use a
helper to hide invalid parameters or replace a known nonzero analytic limit.
"""

import jax.numpy as jnp
from beartype import beartype
from jaxtyping import Array, Bool, Float, jaxtyped

from diffpes.types import ScalarFloat


@jaxtyped(typechecker=beartype)
def safe_divide(
    numerator: Float[Array, " ..."],
    denominator: Float[Array, " ..."],
    fallback: ScalarFloat = 0.0,
) -> Float[Array, " ..."]:
    """Divide with a fallback and zero quotient gradients at zero denominators.

    Applies elementwise guarded division to broadcast-compatible real arrays.
    The fallback defines the value on the zero-denominator boundary.

    :see: :class:`~.test_safe.TestSafeDivide`

    Parameters
    ----------
    numerator : Float[Array, " ..."]
        Dividend array.
    denominator : Float[Array, " ..."]
        Divisor array broadcast-compatible with ``numerator``.
    fallback : ScalarFloat
        Value returned wherever ``denominator`` is zero.

    Returns
    -------
    quotient : Float[Array, " ..."]
        Broadcast quotient with ``fallback`` at zero denominators.

    Notes
    -----
    The function replaces a zero denominator with one before it evaluates the
    inactive division branch. At this boundary, both quotient operands have
    zero selected subgradients. A traced ``fallback`` retains its usual
    selected-value gradient. The function does not establish that a zero
    denominator is valid for the calling model.
    """
    nonzero: Bool[Array, " ..."] = denominator != 0.0
    sanitized_denominator: Float[Array, " ..."] = jnp.where(
        nonzero, denominator, 1.0
    )
    divided: Float[Array, " ..."] = numerator / sanitized_denominator
    quotient: Float[Array, " ..."] = jnp.where(nonzero, divided, fallback)
    return quotient


@jaxtyped(typechecker=beartype)
def safe_sqrt(x: Float[Array, " ..."]) -> Float[Array, " ..."]:
    """Evaluate sqrt on positive inputs and return zero otherwise.

    Applies the principal real square root only on its positive domain. The
    guarded branch defines a finite boundary convention for nonpositive
    inputs; it does not validate the mathematical or physical domain.

    :see: :class:`~.test_safe.TestSafeSqrt`

    Parameters
    ----------
    x : Float[Array, " ..."]
        Real input array.

    Returns
    -------
    roots : Float[Array, " ..."]
        Principal square roots, with zero for ``x <= 0``.

    Notes
    -----
    The function replaces nonpositive inputs with one before it computes the
    square root. The selected value and subgradient are zero for ``x <= 0``.
    Callers for which ``x < 0`` is invalid must reject it before this helper.
    """
    positive: Bool[Array, " ..."] = x > 0.0
    sanitized_x: Float[Array, " ..."] = jnp.where(positive, x, 1.0)
    positive_roots: Float[Array, " ..."] = jnp.sqrt(sanitized_x)
    roots: Float[Array, " ..."] = jnp.where(positive, positive_roots, 0.0)
    return roots


@jaxtyped(typechecker=beartype)
def safe_norm(
    x: Float[Array, " ... n"],
    axis: int = -1,
    keepdims: bool = False,
) -> Float[Array, " ..."]:
    """Compute a Euclidean norm with a zero gradient at zero vectors.

    Reduces the selected vector axis with a guarded square root. The operation
    supports batched vectors and an optional retained axis.

    :see: :class:`~.test_safe.TestSafeNorm`

    Parameters
    ----------
    x : Float[Array, " ... n"]
        Real vectors.
    axis : int
        (**static** — a compile-time constant; changing it triggers
        retracing) Axis containing vector components.
    keepdims : bool
        (**static** — a compile-time constant; changing it triggers
        retracing) Whether the reduced axis remains with length one.

    Returns
    -------
    norms : Float[Array, " ..."]
        Euclidean norms reduced along ``axis``.

    Notes
    -----
    The function passes the squared norm to :func:`safe_sqrt`. A zero vector
    has value zero and a zero selected gradient.
    """
    squared_norms: Float[Array, " ..."] = jnp.sum(
        x * x, axis=axis, keepdims=keepdims
    )
    norms: Float[Array, " ..."] = safe_sqrt(squared_norms)
    return norms


@jaxtyped(typechecker=beartype)
def safe_arccos(x: Float[Array, " ..."]) -> Float[Array, " ..."]:
    """Evaluate arccos with saturated values and zero boundary gradients.

    Computes real angles on the closed cosine range. Values outside the range
    use the nearest physical endpoint.

    :see: :class:`~.test_safe.TestSafeArccos`

    Parameters
    ----------
    x : Float[Array, " ..."]
        Real cosine values.

    Returns
    -------
    angles : Float[Array, " ..."]
        Angles in radians, saturated to ``pi`` below -1 and zero above 1.

    Notes
    -----
    Inputs strictly inside ``(-1, 1)`` use the ordinary ``arccos`` operation.
    Constants supply values at or beyond either endpoint. This selection gives
    zero subgradients and avoids the infinite endpoint derivative. Saturation
    is a convention, not proof that an out-of-range cosine is physically
    acceptable.
    """
    interior: Bool[Array, " ..."] = jnp.abs(x) < 1.0
    sanitized_x: Float[Array, " ..."] = jnp.where(interior, x, 0.0)
    interior_angles: Float[Array, " ..."] = jnp.arccos(sanitized_x)
    saturated_angles: Float[Array, " ..."] = jnp.where(x <= -1.0, jnp.pi, 0.0)
    angles: Float[Array, " ..."] = jnp.where(
        interior, interior_angles, saturated_angles
    )
    return angles


@jaxtyped(typechecker=beartype)
def safe_arctan2(
    y: Float[Array, " ..."], x: Float[Array, " ..."]
) -> Float[Array, " ..."]:
    """Evaluate arctan2 with a zero value and gradient at the origin.

    Computes four-quadrant angles for broadcast-compatible Cartesian
    coordinates. The origin uses an explicit boundary convention.

    :see: :class:`~.test_safe.TestSafeArctan2`

    Parameters
    ----------
    y : Float[Array, " ..."]
        Vertical coordinates.
    x : Float[Array, " ..."]
        Horizontal coordinates broadcast-compatible with ``y``.

    Returns
    -------
    angles : Float[Array, " ..."]
        Four-quadrant angles in radians, with zero at ``(0, 0)``.

    Notes
    -----
    At the indeterminate origin, sanitized coordinates ``(0, 1)`` keep the
    inactive branch finite. The selected value and both coordinate
    subgradients at the origin are zero.
    """
    away_from_origin: Bool[Array, " ..."] = (x != 0.0) | (y != 0.0)
    sanitized_x: Float[Array, " ..."] = jnp.where(away_from_origin, x, 1.0)
    sanitized_y: Float[Array, " ..."] = jnp.where(away_from_origin, y, 0.0)
    ordinary_angles: Float[Array, " ..."] = jnp.arctan2(
        sanitized_y, sanitized_x
    )
    angles: Float[Array, " ..."] = jnp.where(
        away_from_origin, ordinary_angles, 0.0
    )
    return angles


@jaxtyped(typechecker=beartype)
def safe_log(
    x: Float[Array, " ..."], floor: ScalarFloat = 1e-300
) -> Float[Array, " ..."]:
    """Evaluate log with a finite floor and zero gradients below it.

    Computes the natural logarithm on a positive guarded domain. The floor
    keeps the returned values finite for small or nonpositive inputs without
    validating that those inputs belong to the caller's domain.

    :see: :class:`~.test_safe.TestSafeLog`

    Parameters
    ----------
    x : Float[Array, " ..."]
        Real input array.
    floor : ScalarFloat
        Positive lower bound used before taking the logarithm.

    Returns
    -------
    logarithms : Float[Array, " ..."]
        Natural logarithms of ``maximum(x, floor)``.

    Notes
    -----
    The function replaces inputs at or below the positive floor before it
    computes the logarithm. Their selected subgradient with respect to ``x``
    is zero. Callers must separately reject nonpositive inputs when positivity
    is a scientific requirement.
    """
    above_floor: Bool[Array, " ..."] = x > floor
    sanitized_x: Float[Array, " ..."] = jnp.where(above_floor, x, floor)
    logarithms: Float[Array, " ..."] = jnp.log(sanitized_x)
    return logarithms


@jaxtyped(typechecker=beartype)
def safe_power(
    x: Float[Array, " ..."], exponent: ScalarFloat
) -> Float[Array, " ..."]:
    """Raise positive inputs to a power and return zero otherwise.

    Computes real powers on positive bases for arbitrary real exponents. A
    guarded branch supplies a finite convention for nonpositive bases; it is
    not a domain check.

    :see: :class:`~.test_safe.TestSafePower`

    Parameters
    ----------
    x : Float[Array, " ..."]
        Real bases.
    exponent : ScalarFloat
        Real exponent, including non-integer values.

    Returns
    -------
    powers : Float[Array, " ..."]
        ``x**exponent`` for positive ``x`` and zero otherwise.

    Notes
    -----
    The function replaces nonpositive bases with one before exponentiation.
    This replacement prevents complex or invalid results from fractional
    powers. Both inputs have zero selected subgradients on the guarded set.
    Callers must reject a negative base first when it indicates invalid
    physics rather than a registered boundary convention.
    """
    positive: Bool[Array, " ..."] = x > 0.0
    sanitized_x: Float[Array, " ..."] = jnp.where(positive, x, 1.0)
    positive_powers: Float[Array, " ..."] = jnp.power(sanitized_x, exponent)
    powers: Float[Array, " ..."] = jnp.where(positive, positive_powers, 0.0)
    return powers


__all__: list[str] = [
    "safe_arccos",
    "safe_arctan2",
    "safe_divide",
    "safe_log",
    "safe_norm",
    "safe_power",
    "safe_sqrt",
]
