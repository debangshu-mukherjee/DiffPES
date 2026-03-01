r"""Real spherical harmonics in JAX.

Extended Summary
----------------
Implements real spherical harmonics :math:`Y_l^m(\theta, \varphi)`
via associated Legendre polynomial recurrence, JIT-compatible and
differentiable in :math:`(\theta, \varphi)`. Supports l = 0..4
(s, p, d, f, g orbitals).

The convention follows the Condon-Shortley phase:

.. math::

    Y_l^m = \sqrt{2} N_l^{|m|} P_l^{|m|}(\cos\theta) \cos(m\varphi)
        \quad (m > 0)

    Y_l^0 = N_l^0 P_l^0(\cos\theta)

    Y_l^m = \sqrt{2} N_l^{|m|} P_l^{|m|}(\cos\theta) \sin(|m|\varphi)
        \quad (m < 0)

where :math:`N_l^m = \sqrt{\frac{2l+1}{4\pi} \frac{(l-m)!}{(l+m)!}}`.
"""

import math

import jax
import jax.numpy as jnp
from beartype import beartype
from jaxtyping import Array, Float, jaxtyped


def _normalization(l: int, m: int) -> float:
    """Normalization factor for associated Legendre / real Y_lm.

    N_l^m = sqrt((2l+1)/(4pi) * (l-|m|)!/(l+|m|)!)
    """
    am = abs(m)
    return math.sqrt(
        (2 * l + 1)
        / (4.0 * math.pi)
        * math.factorial(l - am)
        / math.factorial(l + am)
    )


def _associated_legendre_plm(
    l: int,
    m: int,
    x: Float[Array, " ..."],
) -> Float[Array, " ..."]:
    r"""Evaluate associated Legendre polynomial P_l^m(x).

    Uses upward recurrence starting from P_m^m, which is the
    most numerically stable direction for m >= 0. The Condon-Shortley
    phase (-1)^m is included.

    Parameters
    ----------
    l : int
        Degree (l >= 0).
    m : int
        Order (0 <= m <= l). Caller must pass |m|.
    x : Float[Array, " ..."]
        cos(theta) values.

    Returns
    -------
    plm : Float[Array, " ..."]
        P_l^m(x) evaluated element-wise.
    """
    # P_m^m = (-1)^m (2m-1)!! (1-x^2)^{m/2}
    pmm = jnp.ones_like(x)
    if m > 0:
        sin_theta = jnp.sqrt(jnp.maximum(1.0 - x * x, 0.0))
        double_fact = 1.0
        for i in range(1, m + 1):
            double_fact *= 2.0 * i - 1.0
        pmm = ((-1.0) ** m) * double_fact * (sin_theta**m)

    if l == m:
        return pmm

    # P_{m+1}^m = x (2m+1) P_m^m
    pmm1 = x * (2.0 * m + 1.0) * pmm
    if l == m + 1:
        return pmm1

    # Upward recurrence: (l-m) P_l^m = (2l-1) x P_{l-1}^m - (l+m-1) P_{l-2}^m
    def _step(
        idx: int,
        state: tuple[Float[Array, " ..."], Float[Array, " ..."]],
    ) -> tuple[Float[Array, " ..."], Float[Array, " ..."]]:
        p_prev2, p_prev1 = state
        idx_f = jnp.asarray(idx, dtype=jnp.float64)
        m_f = jnp.asarray(m, dtype=jnp.float64)
        p_curr = (
            (2.0 * idx_f - 1.0) * x * p_prev1 - (idx_f + m_f - 1.0) * p_prev2
        ) / (idx_f - m_f)
        return p_prev1, p_curr

    _, plm = jax.lax.fori_loop(m + 2, l + 1, _step, (pmm, pmm1))
    return plm


@jaxtyped(typechecker=beartype)
def real_spherical_harmonic(
    l: int,
    m: int,
    theta: Float[Array, " ..."],
    phi: Float[Array, " ..."],
) -> Float[Array, " ..."]:
    r"""Evaluate a single real spherical harmonic :math:`Y_l^m(\theta, \varphi)`.

    Parameters
    ----------
    l : int
        Degree (0 <= l).
    m : int
        Order (-l <= m <= l).
    theta : Float[Array, " ..."]
        Polar angle from z-axis in radians.
    phi : Float[Array, " ..."]
        Azimuthal angle in radians.

    Returns
    -------
    ylm : Float[Array, " ..."]
        Real spherical harmonic values.
    """
    if l < 0:
        msg = "l must be non-negative"
        raise ValueError(msg)
    if abs(m) > l:
        msg = f"|m|={abs(m)} must be <= l={l}"
        raise ValueError(msg)

    cos_theta = jnp.cos(theta)
    am = abs(m)
    plm = _associated_legendre_plm(l, am, cos_theta)
    norm = _normalization(l, m)

    if m > 0:
        return jnp.sqrt(2.0) * norm * plm * jnp.cos(m * phi)
    if m == 0:
        return norm * plm
    # Cancel the Condon-Shortley phase (-1)^|m| embedded in P_l^|m|
    # to match the real-to-complex transform used in the Gaunt table.
    return (-1) ** am * jnp.sqrt(2.0) * norm * plm * jnp.sin(am * phi)


@jaxtyped(typechecker=beartype)
def real_spherical_harmonics_all(
    l_max: int,
    theta: Float[Array, " ..."],
    phi: Float[Array, " ..."],
) -> Float[Array, " ... N"]:
    r"""Evaluate all real spherical harmonics up to l_max.

    Returns values for all (l, m) pairs ordered as:
    (0,0), (1,-1), (1,0), (1,1), (2,-2), (2,-1), (2,0), (2,1), (2,2), ...

    The last axis has size (l_max+1)^2.

    Parameters
    ----------
    l_max : int
        Maximum angular momentum.
    theta : Float[Array, " ..."]
        Polar angle from z-axis.
    phi : Float[Array, " ..."]
        Azimuthal angle.

    Returns
    -------
    ylm_all : Float[Array, " ... N"]
        All spherical harmonics stacked along the last axis.
    """
    results = []
    for l in range(l_max + 1):
        for m in range(-l, l + 1):
            results.append(real_spherical_harmonic(l, m, theta, phi))
    return jnp.stack(results, axis=-1)


__all__: list[str] = [
    "real_spherical_harmonic",
    "real_spherical_harmonics_all",
]
