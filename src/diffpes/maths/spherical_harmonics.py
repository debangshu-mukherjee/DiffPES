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

Routine Listings
----------------
:func:`real_spherical_harmonic`
    Evaluate a single real spherical harmonic Y_l^m(theta, phi).
:func:`real_spherical_harmonics_all`
    Evaluate all real spherical harmonics up to l_max.
"""

import math

import jax
import jax.numpy as jnp
from beartype import beartype
from jaxtyping import Array, Float, jaxtyped


def _normalization(l: int, m: int) -> float:
    r"""Normalization factor for associated Legendre / real Y_lm.

    Extended Summary
    ----------------
    Computes the normalization constant that ensures the real spherical
    harmonics satisfy the orthonormality relation:

    .. math::

        \int Y_l^m \, Y_{l'}^{m'} \, d\Omega
        = \delta_{ll'}\delta_{mm'}

    The normalization factor is:

    .. math::

        N_l^m = \sqrt{\frac{2l+1}{4\pi} \cdot \frac{(l - |m|)!}{(l + |m|)!}}

    This factor absorbs the factorial ratio that arises from the
    integral of the squared associated Legendre polynomial
    :math:`[P_l^{|m|}(\cos\theta)]^2` over :math:`\sin\theta \, d\theta`.
    The ``(2l+1)/(4pi)`` prefactor provides the correct solid-angle
    normalization.

    The computation uses Python's ``math.factorial`` (arbitrary precision
    integers) to avoid overflow for large l + |m|, then takes a
    floating-point square root at the end.

    Parameters
    ----------
    l : int
        Degree of the spherical harmonic (l >= 0).
    m : int
        Order of the spherical harmonic (-l <= m <= l).

    Returns
    -------
    norm : float
        The normalization constant :math:`N_l^{|m|}`.
    """
    am: int = abs(m)
    norm: float = math.sqrt(
        (2 * l + 1)
        / (4.0 * math.pi)
        * math.factorial(l - am)
        / math.factorial(l + am)
    )
    return norm


def _associated_legendre_plm(
    l: int,
    m: int,
    x: Float[Array, " ..."],
) -> Float[Array, " ..."]:
    r"""Evaluate associated Legendre polynomial P_l^m(x).

    Extended Summary
    ----------------
    Computes the associated Legendre polynomial :math:`P_l^m(x)` using
    a three-step recurrence that is numerically stable for upward
    iteration in l at fixed m:

    **Step 1 -- Sectoral seed** :math:`P_m^m(x)`:

    .. math::

        P_m^m(x) = (-1)^m \, (2m-1)!! \, (1 - x^2)^{m/2}

    The Condon-Shortley phase :math:`(-1)^m` is included here,
    consistent with the physics convention used by Condon & Shortley
    (1935) and adopted in most quantum mechanics textbooks. The
    double factorial :math:`(2m-1)!!` is computed iteratively to
    avoid large intermediate values.

    The factor :math:`(1 - x^2)^{m/2}` is computed as
    ``sqrt(max(1 - x^2, 0))^m`` to guard against numerical issues
    when :math:`|x| \approx 1` (near the poles).

    **Step 2 -- First recurrence** :math:`P_{m+1}^m(x)`:

    .. math::

        P_{m+1}^m(x) = x \, (2m + 1) \, P_m^m(x)

    **Step 3 -- Upward recurrence** for l = m+2, ..., target l:

    .. math::

        (l - m) \, P_l^m(x) = (2l - 1) \, x \, P_{l-1}^m(x)
                              - (l + m - 1) \, P_{l-2}^m(x)

    This recurrence is implemented via ``jax.lax.fori_loop`` for
    JIT compatibility. The loop carries two consecutive values
    :math:`(P_{l-2}^m, P_{l-1}^m)` and advances one step per
    iteration.

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

    Notes
    -----
    Upward recurrence in l at fixed m is the standard stable direction
    for associated Legendre polynomials. Downward recurrence in l is
    unstable and is not used here. The function does not support
    negative m directly; the caller should pass ``|m|`` and apply
    the appropriate sign/phase correction externally.
    """
    # P_m^m = (-1)^m (2m-1)!! (1-x^2)^{m/2}
    pmm: Float[Array, " ..."] = jnp.ones_like(x)
    if m > 0:
        sin_theta: Float[Array, " ..."] = jnp.sqrt(
            jnp.maximum(1.0 - x * x, 0.0)
        )
        double_fact: float = 1.0
        for i in range(1, m + 1):
            double_fact *= 2.0 * i - 1.0
        pmm = ((-1.0) ** m) * double_fact * (sin_theta**m)

    if l == m:
        return pmm

    # P_{m+1}^m = x (2m+1) P_m^m
    pmm1: Float[Array, " ..."] = x * (2.0 * m + 1.0) * pmm
    if l == m + 1:
        return pmm1

    # Upward recurrence: (l-m) P_l^m = (2l-1) x P_{l-1}^m - (l+m-1) P_{l-2}^m
    def _step(
        idx: int,
        state: tuple[Float[Array, " ..."], Float[Array, " ..."]],
    ) -> tuple[Float[Array, " ..."], Float[Array, " ..."]]:
        p_prev2, p_prev1 = state
        idx_f: Float[Array, ""] = jnp.asarray(idx, dtype=jnp.float64)
        m_f: Float[Array, ""] = jnp.asarray(m, dtype=jnp.float64)
        p_curr: Float[Array, " ..."] = (
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

    Extended Summary
    ----------------
    Computes the real-valued spherical harmonic using the
    Condon-Shortley phase convention:

    .. math::

        Y_l^m(\theta, \varphi) = \sqrt{2} \, N_l^{|m|} \,
            P_l^{|m|}(\cos\theta) \, \cos(m\varphi)
            \quad (m > 0)

        Y_l^0(\theta, \varphi) = N_l^0 \, P_l^0(\cos\theta)

        Y_l^m(\theta, \varphi) = (-1)^{|m|} \sqrt{2} \, N_l^{|m|} \,
            P_l^{|m|}(\cos\theta) \, \sin(|m|\varphi)
            \quad (m < 0)

    The :math:`(-1)^{|m|}` factor for negative m cancels the
    Condon-Shortley phase :math:`(-1)^{|m|}` already embedded in
    :math:`P_l^{|m|}` by `_associated_legendre_plm`, yielding the
    sign convention consistent with the real-to-complex transformation
    used in the Gaunt coefficient table. This ensures that the Gaunt
    coefficients computed via the complex basis match the angular
    integrals evaluated directly with these real harmonics.

    The normalization factor :math:`N_l^{|m|}` is computed by
    `_normalization` using exact integer arithmetic for the
    factorial ratio.

    This function is JIT-compatible and differentiable with respect
    to both ``theta`` and ``phi`` through JAX automatic
    differentiation.

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
        msg: str = "l must be non-negative"
        raise ValueError(msg)
    if abs(m) > l:
        msg: str = f"|m|={abs(m)} must be <= l={l}"
        raise ValueError(msg)

    cos_theta: Float[Array, " ..."] = jnp.cos(theta)
    am: int = abs(m)
    plm: Float[Array, " ..."] = _associated_legendre_plm(l, am, cos_theta)
    norm: float = _normalization(l, m)

    ylm: Float[Array, " ..."]
    if m > 0:
        ylm = jnp.sqrt(2.0) * norm * plm * jnp.cos(m * phi)
        return ylm
    if m == 0:
        ylm = norm * plm
        return ylm
    # Cancel the Condon-Shortley phase (-1)^|m| embedded in P_l^|m|
    # to match the real-to-complex transform used in the Gaunt table.
    ylm = (-1) ** am * jnp.sqrt(2.0) * norm * plm * jnp.sin(am * phi)
    return ylm


@jaxtyped(typechecker=beartype)
def real_spherical_harmonics_all(
    l_max: int,
    theta: Float[Array, " ..."],
    phi: Float[Array, " ..."],
) -> Float[Array, " ... N"]:
    r"""Evaluate all real spherical harmonics up to l_max.

    Extended Summary
    ----------------
    Computes every real spherical harmonic :math:`Y_l^m(\theta, \varphi)`
    for :math:`0 \le l \le l_{\max}` and :math:`-l \le m \le l`,
    returning them stacked along a new trailing axis.

    The ordering is the standard "degree-then-order" layout:

    .. math::

        (0,0),\;(1,-1),(1,0),(1,1),\;(2,-2),\ldots,(2,2),\;\ldots

    so that the index into the last axis for a given (l, m) pair is
    :math:`l^2 + l + m`. The total number of harmonics is
    :math:`(l_{\max}+1)^2`.

    The loop over (l, m) is unrolled at Python trace time. Each
    harmonic is computed independently via `real_spherical_harmonic`,
    and the results are concatenated with ``jnp.stack``. This approach
    is simple and correct but means that each l value triggers its own
    Legendre recurrence chain; no cross-l recurrence sharing is
    exploited.

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
        All spherical harmonics stacked along the last axis,
        where ``N = (l_max + 1)**2``.
    """
    results: list[Float[Array, " ..."]] = []
    for l in range(l_max + 1):
        for m in range(-l, l + 1):
            results.append(real_spherical_harmonic(l, m, theta, phi))
    ylm_all: Float[Array, " ... N"] = jnp.stack(results, axis=-1)
    return ylm_all


__all__: list[str] = [
    "real_spherical_harmonic",
    "real_spherical_harmonics_all",
]
