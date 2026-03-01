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
"""

import math
from functools import cache

import jax.numpy as jnp
from jaxtyping import Array, Float


def _wigner3j(j1: int, j2: int, j3: int, m1: int, m2: int, m3: int) -> float:
    """Evaluate Wigner 3-j symbol using the Racah formula.

    Only needed for small angular momenta (l_max <= 5), so the
    factorial-based formula is efficient and exact.
    """
    if m1 + m2 + m3 != 0:
        return 0.0
    if abs(m1) > j1 or abs(m2) > j2 or abs(m3) > j3:
        return 0.0
    if j3 < abs(j1 - j2) or j3 > j1 + j2:
        return 0.0

    t_min = max(0, j2 - j3 - m1, j1 - j3 + m2)
    t_max = min(j1 + j2 - j3, j1 - m1, j2 + m2)

    prefactor = math.sqrt(
        math.factorial(j1 + m1)
        * math.factorial(j1 - m1)
        * math.factorial(j2 + m2)
        * math.factorial(j2 - m2)
        * math.factorial(j3 + m3)
        * math.factorial(j3 - m3)
    )
    triangle = math.sqrt(
        math.factorial(j1 + j2 - j3)
        * math.factorial(j1 - j2 + j3)
        * math.factorial(-j1 + j2 + j3)
        / math.factorial(j1 + j2 + j3 + 1)
    )

    total = 0.0
    for t in range(t_min, t_max + 1):
        sign = (-1) ** t
        denom = (
            math.factorial(t)
            * math.factorial(j1 + j2 - j3 - t)
            * math.factorial(j1 - m1 - t)
            * math.factorial(j2 + m2 - t)
            * math.factorial(j3 - j2 + m1 + t)
            * math.factorial(j3 - j1 - m2 + t)
        )
        total += sign / denom

    return float((-1) ** (j1 - j2 - m3) * prefactor * triangle * total)


@cache
def _complex_gaunt(
    l1: int, m1: int, l2: int, m2: int, l3: int, m3: int
) -> float:
    """Complex Gaunt integral for three complex spherical harmonics.

    G = (-1)^m3 * sqrt((2l1+1)(2l2+1)(2l3+1)/(4pi))
        * W3j(l1,l2,l3,0,0,0) * W3j(l1,l2,l3,m1,m2,-m3)
    """
    w3j_000 = _wigner3j(l1, l2, l3, 0, 0, 0)
    if w3j_000 == 0.0:
        return 0.0
    w3j_mmm = _wigner3j(l1, l2, l3, m1, m2, -m3)
    if w3j_mmm == 0.0:
        return 0.0

    prefactor = (-1) ** m3 * math.sqrt(
        (2 * l1 + 1) * (2 * l2 + 1) * (2 * l3 + 1) / (4.0 * math.pi)
    )
    return prefactor * w3j_000 * w3j_mmm


def _real_gaunt_dipole(l: int, m: int, lp: int, mp: int, q: int) -> float:
    """Gaunt coefficient for real spherical harmonics with dipole operator.

    Computes the integral of Y_l^m(real) * r_q * Y_{l'}^{m'}(real)
    over the unit sphere, where r_q (q=-1,0,+1) is the dipole
    operator component.

    The real spherical harmonics are related to complex ones by:
      Y_l^m(real) = (Y_l^m + (-1)^m Y_l^{-m}) / sqrt(2)   for m > 0
      Y_l^0(real) = Y_l^0                                   for m = 0
      Y_l^m(real) = (Y_l^{|m|} - (-1)^m Y_l^{-|m|}) / (i*sqrt(2))  for m < 0

    And the dipole components r_q correspond to l=1 spherical harmonics:
      r_{+1} ~ Y_1^{-1}(real) ~ x  (with factor sqrt(4pi/3))
      r_0    ~ Y_1^0(real)    ~ z
      r_{-1} ~ Y_1^{+1}(real) ~ y
    """
    sqrt2 = math.sqrt(2.0)

    # Build transformation coefficients for Y_l^m(real)
    # in terms of complex Y_l^mu: Y_l^m(real) = sum_mu U_{m,mu} Y_l^mu
    def _real_to_complex_coeffs(ll: int, mm: int) -> list[tuple[complex, int]]:
        """Return [(coeff, mu), ...] such that Y_l^m(real) = sum coeff*Y_l^mu."""
        if mm > 0:
            return [
                (1.0 / sqrt2, mm),
                ((-1) ** mm / sqrt2, -mm),
            ]
        if mm == 0:
            return [(1.0, 0)]
        am = abs(mm)
        return [
            (-1j * (-1) ** am / sqrt2, am),
            (1j / sqrt2, -am),
        ]

    # The dipole operator r_q in terms of complex Y_1^mu:
    # r_q is proportional to Y_1^q(complex)
    # We need the complex m-value for the dipole component
    # Convention: q=-1 -> m_dip=+1 (y), q=0 -> m_dip=0 (z), q=+1 -> m_dip=-1 (x)
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
    dip_coeffs = _real_to_complex_coeffs(1, q)

    # Coefficients for initial state Y_l^m(real)
    init_coeffs = _real_to_complex_coeffs(l, m)
    # Coefficients for final state Y_{l'}^{m'}(real)
    final_coeffs = _real_to_complex_coeffs(lp, mp)

    # The real Gaunt integral is:
    # G_real = sum_{mu, nu, rho} conj(U_final(mp,rho)) * U_dip(q,nu) * U_init(m,mu)
    #          * integral Y_{lp}^{rho*} Y_1^nu Y_l^mu d Omega
    # where the integral is the complex Gaunt coefficient.
    total = 0.0 + 0.0j
    for c_init, mu in init_coeffs:
        for c_dip, nu in dip_coeffs:
            for c_final, rho in final_coeffs:
                # integral of Y_{lp}^{rho*} * Y_1^nu * Y_l^mu
                # = (-1)^rho * integral Y_{lp}^{-rho} Y_1^nu Y_l^mu
                # Using our _complex_gaunt which computes
                # integral Y_{l1}^{m1} Y_{l2}^{m2} Y_{l3}^{m3}:
                cg = _complex_gaunt(lp, -rho, 1, nu, l, mu)
                coeff = (
                    complex(c_final).conjugate()
                    * complex(c_dip)
                    * complex(c_init)
                )
                total += coeff * (-1) ** rho * cg

    result = total.real
    if abs(total.imag) > 1e-12:
        msg = f"Imaginary part {total.imag} in real Gaunt coefficient"
        raise ValueError(msg)
    return result


def build_gaunt_table(
    l_max: int = 4,
) -> Float[Array, "L_src M_src 3 L_dst M_dst"]:
    """Build the dipole Gaunt coefficient lookup table.

    The table is indexed as ``GAUNT_TABLE[l, m + l_max, q + 1, lp, mp + l_max]``
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
    """
    l_src_dim = l_max + 1
    m_src_dim = 2 * l_max + 1
    q_dim = 3
    l_dst_dim = l_max + 2
    m_dst_dim = 2 * (l_max + 1) + 1

    import numpy as np

    table = np.zeros(
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
                        val = _real_gaunt_dipole(l, m, lp, mp, q)
                        table[l, m + l_max, q + 1, lp, mp + l_max] = val

    return jnp.asarray(table, dtype=jnp.float64)


GAUNT_TABLE: Float[Array, "..."] = build_gaunt_table(l_max=4)
"""Module-level precomputed Gaunt coefficient table for l_max=4."""

L_MAX: int = 4
"""Maximum angular momentum supported by the precomputed table."""


def gaunt_lookup(l: int, m: int, q: int, lp: int, mp: int) -> float:
    """Look up a single Gaunt coefficient from the precomputed table.

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
