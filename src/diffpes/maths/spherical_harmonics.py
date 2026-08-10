r"""Compute real spherical harmonics in JAX.

Extended Summary
----------------
The module computes real spherical harmonics with an associated Legendre
polynomial recurrence. JAX can transform and differentiate the computation in
:math:`(\theta, \varphi)`. The module supports l = 0..4.

The convention follows the Condon-Shortley phase:

.. math::

    Y_l^m = (-1)^m \sqrt{2} N_l^{|m|} P_l^{|m|}(\cos\theta) \cos(m\varphi)
        \quad (m > 0)

    Y_l^0 = N_l^0 P_l^0(\cos\theta)

    Y_l^m = \sqrt{2} N_l^{|m|} P_l^{|m|}(\cos\theta) \sin(|m|\varphi)
        \quad (m < 0)

where :math:`N_l^m = \sqrt{\frac{2l+1}{4\pi} \frac{(l-m)!}{(l+m)!}}`.

Routine Listings
----------------
:func:`real_spherical_harmonic`
    Evaluate a single real spherical harmonic.
:func:`real_spherical_harmonics_all`
    Evaluate all real spherical harmonics up to l_max.
"""

import math

import jax
import jax.numpy as jnp
from beartype import beartype
from beartype.typing import Tuple
from jaxtyping import Array, Float64, Int64, jaxtyped


def _normalization(l: int, m: int) -> float:
    r"""PRIVATE: Compute normalization for associated Legendre / real Y_lm.

    Extended Summary
    ----------------
    The function computes the normalization constant that makes the real
    spherical harmonics satisfy the orthonormality relation:

    .. math::

        \int Y_l^m \, Y_{l'}^{m'} \, d\Omega
        = \delta_{ll'}\delta_{mm'}

    The normalization factor is:

    .. math::

        N_l^m = \sqrt{\frac{2l+1}{4\pi} \cdot \frac{(l - |m|)!}{(l + |m|)!}}

    This factor includes the factorial ratio that arises from the
    integral of the squared associated Legendre polynomial
    :math:`[P_l^{|m|}(\cos\theta)]^2` over :math:`\sin\theta \, d\theta`.
    The ``(2l+1)/(4pi)`` prefactor provides the correct solid-angle
    normalization.

    The function uses Python's ``math.factorial`` with arbitrary-precision
    integers. This operation avoids overflow for large l + |m|. The function
    then computes a floating-point square root.

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
    x: Float64[Array, " ..."],
) -> Float64[Array, " ..."]:
    r"""PRIVATE: Evaluate associated Legendre polynomial P_l^m(x).

    Extended Summary
    ----------------
    The function computes :math:`P_l^m(x)` with a three-step recurrence. This
    recurrence remains numerically stable during upward iteration in l at
    fixed m:

    **Step 1 -- Sectoral seed** :math:`P_m^m(x)`:

    .. math::

        P_m^m(x) = (-1)^m \, (2m-1)!! \, (1 - x^2)^{m/2}

    The seed includes the Condon-Shortley phase :math:`(-1)^m`. This phase
    follows the physics convention from Condon and Shortley (1935). The
    function computes :math:`(2m-1)!!` iteratively to avoid large intermediate
    values.

    The function computes :math:`(1 - x^2)^{m/2}` as
    ``sqrt(max(1 - x^2, 0))^m`` to guard against numerical issues
    when :math:`|x| \approx 1` (near the poles).

    **Step 2 -- First recurrence** :math:`P_{m+1}^m(x)`:

    .. math::

        P_{m+1}^m(x) = x \, (2m + 1) \, P_m^m(x)

    **Step 3 -- Upward recurrence** for l = m+2, ..., target l:

    .. math::

        (l - m) \, P_l^m(x) = (2l - 1) \, x \, P_{l-1}^m(x)
                              - (l + m - 1) \, P_{l-2}^m(x)

    The function implements this recurrence with ``jax.lax.fori_loop``. The
    loop carries two consecutive values
    :math:`(P_{l-2}^m, P_{l-1}^m)` and advances one step per
    iteration.

    Parameters
    ----------
    l : int
        Degree (l >= 0).
    m : int
        Order (0 <= m <= l). Caller must pass |m|.
    x : Float64[Array, " ..."]
        cos(theta) values.

    Returns
    -------
    plm : Float64[Array, " ..."]
        P_l^m(x) evaluated element-wise.

    Notes
    -----
    Upward recurrence in l at fixed m gives the stable direction for associated
    Legendre polynomials. The function does not use the unstable downward
    recurrence. Pass ``|m|`` because the function does not accept a negative
    m. Apply the required sign or phase correction separately.
    """
    i: int

    pmm: Float64[Array, " ..."] = jnp.ones_like(x)
    if m > 0:
        sin_theta: Float64[Array, " ..."] = jnp.sqrt(
            jnp.maximum(1.0 - x * x, 0.0)
        )
        double_fact: float = 1.0
        for i in range(1, m + 1):
            double_fact *= 2.0 * i - 1.0
        pmm = ((-1.0) ** m) * double_fact * (sin_theta**m)

    if l == m:
        return pmm

    pmm1: Float64[Array, " ..."] = x * (2.0 * m + 1.0) * pmm
    if l == m + 1:
        return pmm1

    def _step(
        idx: Int64[Array, ""],
        state: Tuple[Float64[Array, " ..."], Float64[Array, " ..."]],
    ) -> Tuple[Float64[Array, " ..."], Float64[Array, " ..."]]:
        """PRIVATE: Compute one upward Legendre recurrence step in degree.

        Parameters
        ----------
        idx : Int64[Array, ""]
            Current target degree ``l`` supplied by ``fori_loop``.
        state : Tuple[Float64[Array, " ..."], Float64[Array, " ..."]]
            Values ``(P(idx - 2), P(idx - 1))`` at fixed order ``m``.

        Returns
        -------
        recurrence_state : tuple
            Shifted carry ``(P(idx - 1), P(idx))``.

        Notes
        -----
        One step evaluates ``P(idx) = ((2 idx - 1) x P(idx - 1)
        - (idx + m - 1) P(idx - 2)) / (idx - m)``, the stable upward
        three-term recurrence in the degree at fixed order.
        """
        p_prev2: Float64[Array, " ..."]
        p_prev1: Float64[Array, " ..."]
        p_prev2, p_prev1 = state
        idx_f: Float64[Array, ""] = jnp.asarray(idx, dtype=jnp.float64)
        m_f: Float64[Array, ""] = jnp.asarray(m, dtype=jnp.float64)
        p_curr: Float64[Array, " ..."] = (
            (2.0 * idx_f - 1.0) * x * p_prev1 - (idx_f + m_f - 1.0) * p_prev2
        ) / (idx_f - m_f)
        recurrence_state: Tuple[
            Float64[Array, " ..."], Float64[Array, " ..."]
        ] = (
            p_prev1,
            p_curr,
        )
        return recurrence_state

    recurrence_result: Tuple[
        Float64[Array, " ..."], Float64[Array, " ..."]
    ] = jax.lax.fori_loop(m + 2, l + 1, _step, (pmm, pmm1))
    plm: Float64[Array, " ..."] = recurrence_result[1]
    return plm


@jaxtyped(typechecker=beartype)
def real_spherical_harmonic(
    l: int,
    m: int,
    theta: Float64[Array, " ..."],
    phi: Float64[Array, " ..."],
) -> Float64[Array, " ..."]:
    r"""Evaluate a single real spherical harmonic.

    The function computes :math:`Y_l^m(\theta, \varphi)`.

    The function computes the real-valued spherical harmonic with the
    Condon-Shortley phase convention:

    .. math::

        Y_l^m(\theta, \varphi) = (-1)^m \sqrt{2} \, N_l^{|m|} \,
            P_l^{|m|}(\cos\theta) \, \cos(m\varphi)
            \quad (m > 0)

        Y_l^0(\theta, \varphi) = N_l^0 \, P_l^0(\cos\theta)

        Y_l^m(\theta, \varphi) = (-1)^{|m|} \sqrt{2} \, N_l^{|m|} \,
            P_l^{|m|}(\cos\theta) \, \sin(|m|\varphi)
            \quad (m < 0)

    For nonzero m, the explicit :math:`(-1)^{|m|}` factor cancels the
    Condon-Shortley phase already included by `_associated_legendre_plm` in
    :math:`P_l^{|m|}`. The result follows the real-to-complex
    convention of the Gaunt table. Consequently, the complex-basis Gaunt
    coefficients match direct integrals of these real harmonics.

    The `_normalization` function computes :math:`N_l^{|m|}`. It uses exact
    integer arithmetic for the factorial ratio.

    JAX can transform this function and differentiate it with respect to
    ``theta`` and ``phi``.

    :see: :class:`~.test_spherical_harmonics.TestRealSphericalHarmonic`

    Implementation Logic
    --------------------
    1. **Compute the associated Legendre values**::

           cos_theta: Float64[Array, " ..."] = jnp.cos(theta)
           am: int = abs(m)
           plm: Float64[Array, " ..."] = _associated_legendre_plm(
               l, am, cos_theta
           )

       These values provide the angular basis for each real branch.

    2. **Select the real harmonic branch**::

           if m > 0:
               ylm = jnp.sqrt(2.0) * norm * plm * jnp.cos(m * phi)
               return ylm

       The sign of ``m`` selects the cosine, constant, or sine basis.

    Parameters
    ----------
    l : int
        Degree (0 <= l).
    m : int
        Order (-l <= m <= l).
    theta : Float64[Array, " ..."]
        Polar angle from z-axis in radians.
    phi : Float64[Array, " ..."]
        Azimuthal angle in radians.

    Returns
    -------
    ylm : Float64[Array, " ..."]
        Real spherical harmonic values.

    Raises
    ------
    ValueError
        If ``l`` is negative or ``abs(m)`` exceeds ``l``.
    """
    if l < 0:
        msg: str = "l must be non-negative"
        raise ValueError(msg)
    if abs(m) > l:
        msg: str = f"|m|={abs(m)} must be <= l={l}"
        raise ValueError(msg)

    cos_theta: Float64[Array, " ..."] = jnp.cos(theta)
    am: int = abs(m)
    plm: Float64[Array, " ..."] = _associated_legendre_plm(l, am, cos_theta)
    norm: float = _normalization(l, m)

    ylm: Float64[Array, " ..."]
    if m > 0:
        ylm = ((-1.0) ** m) * jnp.sqrt(2.0) * norm * plm * jnp.cos(m * phi)
        return ylm
    if m == 0:
        ylm = norm * plm
        return ylm
    ylm = (-1) ** am * jnp.sqrt(2.0) * norm * plm * jnp.sin(am * phi)
    return ylm


@jaxtyped(typechecker=beartype)
def real_spherical_harmonics_all(
    l_max: int,
    theta: Float64[Array, " ..."],
    phi: Float64[Array, " ..."],
) -> Float64[Array, " ... N"]:
    r"""Evaluate all real spherical harmonics up to l_max.

    The function computes every real spherical harmonic
    :math:`Y_l^m(\theta, \varphi)` for :math:`0 \le l \le l_{\max}` and
    :math:`-l \le m \le l`,
    returning them stacked along a new trailing axis.

    The ordering is the standard "degree-then-order" layout:

    .. math::

        (0,0),\;(1,-1),(1,0),(1,1),\;(2,-2),\ldots,(2,2),\;\ldots

    so that the index into the last axis for a given (l, m) pair is
    :math:`l^2 + l + m`. The total number of harmonics is
    :math:`(l_{\max}+1)^2`.

    Python unrolls the loop over (l, m) during tracing. The
    `real_spherical_harmonic` function computes each harmonic independently.
    ``jnp.stack`` joins the results. Each l value therefore starts a separate
    Legendre recurrence chain. The implementation does not share recurrence
    values across l values.

    :see: :class:`~.test_spherical_harmonics.TestRealSphericalHarmonicsAll`

    Implementation Logic
    --------------------
    1. **Collect harmonics in canonical order**::

           for l in range(l_max + 1):
               for m in range(-l, l + 1):
                   results.append(
                       real_spherical_harmonic(l, m, theta, phi)
                   )

       The nested ranges produce the documented degree-then-order sequence.

    2. **Stack the harmonic fields**::

           ylm_all: Float64[Array, " ... N"] = jnp.stack(
               results, axis=-1
           )

       The new trailing axis indexes each ``(l, m)`` pair.

    Parameters
    ----------
    l_max : int
        Maximum angular momentum.
    theta : Float64[Array, " ..."]
        Polar angle from z-axis.
    phi : Float64[Array, " ..."]
        Azimuthal angle.

    Returns
    -------
    ylm_all : Float64[Array, " ... N"]
        All spherical harmonics stacked along the last axis,
        where ``N = (l_max + 1)**2``.
    """
    l: int
    m: int

    results: list[Float64[Array, " ..."]] = []
    for l in range(l_max + 1):
        for m in range(-l, l + 1):
            results.append(real_spherical_harmonic(l, m, theta, phi))
    ylm_all: Float64[Array, " ... N"] = jnp.stack(results, axis=-1)
    return ylm_all


__all__: list[str] = [
    "real_spherical_harmonic",
    "real_spherical_harmonics_all",
]
