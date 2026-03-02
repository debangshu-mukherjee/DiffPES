r"""Gaunt coefficient table for dipole transitions.

Extended Summary
----------------
Precomputes real-valued Gaunt integrals for electric dipole
transitions in real spherical harmonics:

.. math::

    G(l, m, l', m') = \int Y_l^m(\hat{r})\, r_q\, Y_{l'}^{m'}(\hat{r})\,d\Omega

where :math:`r_q` is one of the three components of the position
operator expressed in the real spherical harmonic basis (q = -1, 0, +1).

Selection rules: :math:`l' = l \pm 1` and :math:`|m' - m| \leq 1`.

The table is computed once at module load time using pure Python
(not JAX-traced) and stored as a JAX array for O(1) lookup.

Routine Listings
----------------
:data:`GAUNT_TABLE`
    Module-level precomputed Gaunt coefficient table for l_max=4.
:data:`L_MAX`
    Maximum angular momentum supported by the precomputed table.
:func:`build_gaunt_table`
    Build the dipole Gaunt coefficient lookup table.
:func:`gaunt_lookup`
    Look up a single Gaunt coefficient from the precomputed table.
"""

import math
from functools import cache

import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float
from numpy import ndarray as NDArray  # noqa: N812

_IMAG_TOL: float = 1e-12


def _wigner3j(j1: int, j2: int, j3: int, m1: int, m2: int, m3: int) -> float:
    r"""Evaluate Wigner 3-j symbol using the Racah formula.

    Extended Summary
    ----------------
    Computes the Wigner 3-j symbol

    .. math::

        \begin{pmatrix} j_1 & j_2 & j_3 \\ m_1 & m_2 & m_3 \end{pmatrix}

    using the explicit Racah formula, which expresses the 3-j symbol as a
    finite sum over factorials. The formula is:

    .. math::

        \begin{pmatrix} j_1 & j_2 & j_3 \\ m_1 & m_2 & m_3 \end{pmatrix}
        = (-1)^{j_1 - j_2 - m_3} \Delta(j_1, j_2, j_3) \cdot P \cdot
        \sum_t \frac{(-1)^t}{t! \cdot D(t)}

    where :math:`\Delta(j_1, j_2, j_3)` is the triangle coefficient

    .. math::

        \Delta = \sqrt{
            \frac{(j_1+j_2-j_3)!(j_1-j_2+j_3)!(-j_1+j_2+j_3)!}
                 {(j_1+j_2+j_3+1)!}
        }

    and :math:`P` is the product of square roots of magnetic-number
    factorials:

    .. math::

        P = \sqrt{(j_1+m_1)!(j_1-m_1)!(j_2+m_2)!(j_2-m_2)!(j_3+m_3)!(j_3-m_3)!}

    The summation index *t* runs over all integers for which all
    factorial arguments are non-negative, determined by ``t_min`` and
    ``t_max``.

    **Selection rules** enforced before computation:

    - :math:`m_1 + m_2 + m_3 = 0` (magnetic quantum number conservation)
    - :math:`|m_i| \le j_i` for each i (projection bounds)
    - :math:`|j_1 - j_2| \le j_3 \le j_1 + j_2` (triangle inequality)

    If any selection rule is violated, the function returns 0.0 immediately
    without computing the sum.

    Only needed for small angular momenta (l_max <= 5), so the
    factorial-based formula is efficient and exact (no floating-point
    cancellation issues at this scale).

    Parameters
    ----------
    j1, j2, j3 : int
        Angular momentum quantum numbers (non-negative integers).
    m1, m2, m3 : int
        Magnetic quantum numbers satisfying :math:`|m_i| \le j_i`.

    Returns
    -------
    value : float
        The Wigner 3-j symbol value.
    """
    if m1 + m2 + m3 != 0:
        return 0.0
    if abs(m1) > j1 or abs(m2) > j2 or abs(m3) > j3:
        return 0.0
    if j3 < abs(j1 - j2) or j3 > j1 + j2:
        return 0.0

    t_min: int = max(0, j2 - j3 - m1, j1 - j3 + m2)
    t_max: int = min(j1 + j2 - j3, j1 - m1, j2 + m2)

    prefactor: float = math.sqrt(
        math.factorial(j1 + m1)
        * math.factorial(j1 - m1)
        * math.factorial(j2 + m2)
        * math.factorial(j2 - m2)
        * math.factorial(j3 + m3)
        * math.factorial(j3 - m3)
    )
    triangle: float = math.sqrt(
        math.factorial(j1 + j2 - j3)
        * math.factorial(j1 - j2 + j3)
        * math.factorial(-j1 + j2 + j3)
        / math.factorial(j1 + j2 + j3 + 1)
    )

    total: float = 0.0
    for t in range(t_min, t_max + 1):
        sign: int = (-1) ** t
        denom: int = (
            math.factorial(t)
            * math.factorial(j1 + j2 - j3 - t)
            * math.factorial(j1 - m1 - t)
            * math.factorial(j2 + m2 - t)
            * math.factorial(j3 - j2 + m1 + t)
            * math.factorial(j3 - j1 - m2 + t)
        )
        total += sign / denom

    value: float = float((-1) ** (j1 - j2 - m3) * prefactor * triangle * total)
    return value


@cache
def _complex_gaunt(
    l1: int, m1: int, l2: int, m2: int, l3: int, m3: int
) -> float:
    r"""Complex Gaunt integral for three complex spherical harmonics.

    Extended Summary
    ----------------
    Computes the Gaunt integral (three-Y integral) defined as:

    .. math::

        G(l_1,m_1; l_2,m_2; l_3,m_3)
        = \int Y_{l_1}^{m_1} Y_{l_2}^{m_2} Y_{l_3}^{m_3} \, d\Omega

    Using the standard decomposition into Wigner 3-j symbols:

    .. math::

        G = (-1)^{m_3}
            \sqrt{\frac{(2l_1+1)(2l_2+1)(2l_3+1)}{4\pi}}
            \begin{pmatrix} l_1 & l_2 & l_3 \\ 0 & 0 & 0 \end{pmatrix}
            \begin{pmatrix} l_1 & l_2 & l_3 \\ m_1 & m_2 & -m_3 \end{pmatrix}

    The first 3-j symbol enforces the parity selection rule
    :math:`l_1 + l_2 + l_3 = \text{even}`, and is evaluated first as
    an early-exit optimization. The result is cached via
    ``functools.cache`` since the same (l, m) combinations appear
    repeatedly during Gaunt table construction.

    Parameters
    ----------
    l1, l2, l3 : int
        Angular momentum quantum numbers.
    m1, m2, m3 : int
        Magnetic quantum numbers.

    Returns
    -------
    value : float
        The complex Gaunt integral.
    """
    w3j_000: float = _wigner3j(l1, l2, l3, 0, 0, 0)
    if w3j_000 == 0.0:
        return 0.0
    w3j_mmm: float = _wigner3j(l1, l2, l3, m1, m2, -m3)
    if w3j_mmm == 0.0:
        return 0.0

    prefactor: float = (-1) ** m3 * math.sqrt(
        (2 * l1 + 1) * (2 * l2 + 1) * (2 * l3 + 1) / (4.0 * math.pi)
    )
    value: float = prefactor * w3j_000 * w3j_mmm
    return value


def _real_gaunt_dipole(l: int, m: int, lp: int, mp: int, q: int) -> float:
    r"""Gaunt coefficient for real spherical harmonics with dipole operator.

    Extended Summary
    ----------------
    Computes the integral of :math:`Y_l^m(\text{real}) \cdot r_q \cdot
    Y_{l'}^{m'}(\text{real})` over the unit sphere, where
    :math:`r_q` (q = -1, 0, +1) is a dipole operator component.

    Because the Gaunt integral is natively defined for complex spherical
    harmonics, this function performs a basis transformation: each real
    harmonic is expanded in the complex basis using unitary coefficients,
    and the resulting triple sum of complex Gaunt integrals is accumulated.

    **Real-to-complex transformation:**

    .. math::

        Y_l^m(\text{real}) = \frac{1}{\sqrt{2}}
            \bigl(Y_l^m + (-1)^m Y_l^{-m}\bigr)
            \quad (m > 0)

        Y_l^0(\text{real}) = Y_l^0 \quad (m = 0)

        Y_l^m(\text{real}) = \frac{-i}{\sqrt{2}}
            \bigl((-1)^{|m|} Y_l^{|m|} - Y_l^{-|m|}\bigr)
            \quad (m < 0)

    Each real harmonic thus becomes a sum of at most two complex harmonics
    with known coefficients (computed by the nested helper
    ``_real_to_complex_coeffs``).

    **Dipole operator in complex basis:**

    The position operator component :math:`r_q` corresponds to the l=1
    real spherical harmonic :math:`Y_1^q(\text{real})`, which is itself
    expanded in the complex basis using the same transformation.

    **Assembly:**

    The full real Gaunt coefficient is:

    .. math::

        G_{\text{real}} = \sum_{\mu, \nu, \rho}
            \overline{U_{\text{final}}(m', \rho)} \cdot
            U_{\text{dip}}(q, \nu) \cdot
            U_{\text{init}}(m, \mu) \cdot
            (-1)^{\rho} \cdot
            G_{\text{complex}}(l', -\rho; 1, \nu; l, \mu)

    where :math:`U` matrices are the real-to-complex transformation
    coefficients. The conjugation on the final-state coefficients arises
    because the complex Gaunt integral involves :math:`Y_{l'}^{\rho *}`,
    and :math:`Y_l^{m *} = (-1)^m Y_l^{-m}`.

    The result is guaranteed to be real; an imaginary part exceeding
    :math:`10^{-12}` triggers a ``ValueError`` as a consistency check.

    Parameters
    ----------
    l : int
        Initial state angular momentum.
    m : int
        Initial state magnetic quantum number.
    lp : int
        Final state angular momentum.
    mp : int
        Final state magnetic quantum number.
    q : int
        Dipole component index (-1, 0, or +1).

    Returns
    -------
    value : float
        The real Gaunt coefficient for the specified quantum numbers.
    """
    sqrt2: float = math.sqrt(2.0)

    # Build transformation coefficients for Y_l^m(real)
    # in terms of complex Y_l^mu: Y_l^m(real) = sum_mu U_{m,mu} Y_l^mu
    def _real_to_complex_coeffs(ll: int, mm: int) -> list[tuple[complex, int]]:
        r"""Return real-to-complex expansion coefficients.

        Computes the unitary transformation coefficients :math:`U_{m,\mu}`
        such that :math:`Y_l^m(\text{real}) = \sum_\mu U_{m,\mu} Y_l^\mu`.
        For m > 0, two terms with :math:`\mu = \pm m` contribute with
        coefficients :math:`1/\sqrt{2}` and :math:`(-1)^m/\sqrt{2}`.
        For m = 0, a single term with coefficient 1. For m < 0, two
        terms with imaginary coefficients that isolate the sine component.

        Parameters
        ----------
        ll : int
            Angular momentum (unused in the formula but kept for clarity).
        mm : int
            Magnetic quantum number of the real harmonic.

        Returns
        -------
        coeffs : list of (complex, int)
            List of ``(coefficient, mu)`` pairs.
        """
        if mm > 0:
            return [
                (1.0 / sqrt2, mm),
                ((-1) ** mm / sqrt2, -mm),
            ]
        if mm == 0:
            return [(1.0, 0)]
        am: int = abs(mm)
        return [
            (-1j * (-1) ** am / sqrt2, am),
            (1j / sqrt2, -am),
        ]

    # The dipole operator r_q in terms of complex Y_1^mu:
    # r_q is proportional to Y_1^q(complex)
    # We need the complex m-value for the dipole component
    # Convention: q=-1 -> m_dip=+1 (y), q=0 -> m_dip=0 (z),
    # q=+1 -> m_dip=-1 (x)
    # Actually, using the standard convention:
    #   x = sqrt(4pi/3) * Y_1^{-1}(real)
    #   y = sqrt(4pi/3) * Y_1^{+1}(real) ... but let's use complex:
    #   r_{-1} = sqrt(2pi/3) (x - iy) / sqrt(2) ~ Y_1^{-1}(complex)
    #   r_0    = sqrt(4pi/3) z ~ Y_1^0(complex)
    #   r_{+1} = -sqrt(2pi/3) (x + iy) / sqrt(2) ~ Y_1^{+1}(complex)
    #
    # For the real dipole operator r_q (q index for the 3 Cartesian components
    # mapped to real spherical harmonics of l=1):
    #   r_q(real) corresponds to Y_1^q(real)
    # Transform Y_1^q(real) to complex basis:
    dip_coeffs: list[tuple[complex, int]] = _real_to_complex_coeffs(1, q)

    # Coefficients for initial state Y_l^m(real)
    init_coeffs: list[tuple[complex, int]] = _real_to_complex_coeffs(l, m)
    # Coefficients for final state Y_{l'}^{m'}(real)
    final_coeffs: list[tuple[complex, int]] = _real_to_complex_coeffs(lp, mp)

    # Real Gaunt integral: G_real = sum over (mu, nu, rho) of
    # conj(U_final) * U_dip * U_init * complex Gaunt integral.
    total: complex = 0.0 + 0.0j
    for c_init, mu in init_coeffs:
        for c_dip, nu in dip_coeffs:
            for c_final, rho in final_coeffs:
                # integral of Y_{lp}^{rho*} * Y_1^nu * Y_l^mu
                # = (-1)^rho * integral Y_{lp}^{-rho} Y_1^nu Y_l^mu
                # Using our _complex_gaunt which computes
                # integral Y_{l1}^{m1} Y_{l2}^{m2} Y_{l3}^{m3}:
                cg: float = _complex_gaunt(lp, -rho, 1, nu, l, mu)
                coeff: complex = (
                    complex(c_final).conjugate()
                    * complex(c_dip)
                    * complex(c_init)
                )
                total += coeff * (-1) ** rho * cg

    result: float = total.real
    if abs(total.imag) > _IMAG_TOL:  # pragma: no cover
        msg: str = f"Imaginary part {total.imag} in real Gaunt coefficient"
        raise ValueError(msg)
    return result


def build_gaunt_table(
    l_max: int = 4,
) -> Float[Array, "L_src M_src 3 L_dst M_dst"]:
    r"""Build the dipole Gaunt coefficient lookup table.

    Extended Summary
    ----------------
    Precomputes every non-zero real Gaunt coefficient for electric
    dipole transitions up to angular momentum ``l_max``. The table
    is stored as a dense 5-D NumPy array and then converted to a
    JAX array for O(1) lookup during forward-model evaluation.

    The five axes correspond to:

    - **axis 0** (``l``): initial angular momentum, size ``l_max + 1``.
    - **axis 1** (``m + l_max``): offset initial magnetic quantum number,
      size ``2 * l_max + 1``, so that m = -l_max maps to index 0.
    - **axis 2** (``q + 1``): dipole component index, size 3, for
      q in {-1, 0, +1}.
    - **axis 3** (``lp``): final angular momentum, size ``l_max + 2``
      (because the dipole operator can promote l to l + 1).
    - **axis 4** (``mp + l_max``): offset final magnetic quantum number,
      size ``2 * (l_max + 1) + 1``.

    The construction loops over all valid quantum number combinations
    respecting the dipole selection rules :math:`l' = l \pm 1` and
    calls `_real_gaunt_dipole` for each. Zero entries (forbidden
    transitions) are left at their initialized value.

    The table is indexed as
    ``GAUNT_TABLE[l, m + l_max, q + 1, lp, mp + l_max]``
    where q in {-1, 0, +1} indexes the three dipole components.

    Parameters
    ----------
    l_max : int
        Maximum angular momentum. Default 4 (s through g).

    Returns
    -------
    table : Float[Array, "..."]
        Dense array of Gaunt coefficients.
        Shape: ``(l_max+1, 2*l_max+1, 3, l_max+2, 2*(l_max+1)+1)``.

    Notes
    -----
    This function uses pure Python / NumPy (not JAX-traced) because it
    runs once at module import time. The result is frozen as a JAX
    array constant, so it does not appear as a trainable parameter in
    any gradient computation.
    """
    l_src_dim: int = l_max + 1
    m_src_dim: int = 2 * l_max + 1
    q_dim: int = 3
    l_dst_dim: int = l_max + 2
    m_dst_dim: int = 2 * (l_max + 1) + 1

    table: Float[NDArray, "L1 M1 Q L2 M2"] = np.zeros(
        (l_src_dim, m_src_dim, q_dim, l_dst_dim, m_dst_dim),
        dtype=np.float64,
    )

    for l in range(l_src_dim):
        for m in range(-l, l + 1):
            for q in (-1, 0, 1):
                for lp in (l - 1, l + 1):
                    if lp < 0 or lp >= l_dst_dim:
                        continue
                    for mp in range(-lp, lp + 1):
                        val: float = _real_gaunt_dipole(l, m, lp, mp, q)
                        table[l, m + l_max, q + 1, lp, mp + l_max] = val

    return jnp.asarray(table, dtype=jnp.float64)


GAUNT_TABLE: Float[Array, "..."] = build_gaunt_table(l_max=4)
"""Module-level precomputed Gaunt coefficient table for l_max=4."""

L_MAX: int = 4
"""Maximum angular momentum supported by the precomputed table."""


def gaunt_lookup(l: int, m: int, q: int, lp: int, mp: int) -> float:
    r"""Look up a single Gaunt coefficient from the precomputed table.

    Extended Summary
    ----------------
    Provides a convenience accessor for the module-level ``GAUNT_TABLE``
    array, converting the physical quantum numbers (l, m, q, l', m')
    into the offset indices used by the dense storage layout. The
    index mapping is:

    - ``l`` indexes axis 0 directly.
    - ``m + L_MAX`` offsets the magnetic quantum number to a
      non-negative index on axis 1.
    - ``q + 1`` maps q in {-1, 0, +1} to indices {0, 1, 2} on axis 2.
    - ``lp`` indexes axis 3 directly.
    - ``mp + L_MAX`` offsets m' on axis 4.

    The returned value is cast to a Python float for use in
    non-JAX contexts. For JAX-traced code, direct indexing into
    ``GAUNT_TABLE`` is preferred to avoid Python-level overhead.

    Parameters
    ----------
    l : int
        Initial state angular momentum.
    m : int
        Initial state magnetic quantum number.
    q : int
        Dipole component (-1, 0, or +1).
    lp : int
        Final state angular momentum.
    mp : int
        Final state magnetic quantum number.

    Returns
    -------
    coeff : float
        The Gaunt coefficient.
    """
    return float(GAUNT_TABLE[l, m + L_MAX, q + 1, lp, mp + L_MAX])


__all__: list[str] = [
    "GAUNT_TABLE",
    "L_MAX",
    "build_gaunt_table",
    "gaunt_lookup",
]
