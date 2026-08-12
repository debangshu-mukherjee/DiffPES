r"""Construct differentiable three-dimensional rotations.

Extended Summary
----------------
This module provides Cartesian Rodrigues rotations and angular-momentum
representations of active rotations. The Wigner matrices use the package-wide
Condon--Shortley convention and order both magnetic-number axes from
:math:`m=-l` through :math:`m=+l`. The real-harmonic unitary uses the same
ordering as :mod:`diffpes.maths.spherical_harmonics` and
:mod:`diffpes.maths.gaunt`.

For a z--y--z active rotation, the representation is

.. math::

    D^l_{m'm}(\alpha,\beta,\gamma)
    = e^{-i m'\alpha}d^l_{m'm}(\beta)e^{-i m\gamma}.

The finite factorial sum evaluates the small-d matrix. The angular
momenta are static and restricted to :math:`0\leq l\leq4`; the angles remain
traced JAX values.

Routine Listings
----------------
:func:`bond_angles`
    Convert a Cartesian bond to safe polar and azimuthal angles.
:func:`real_harmonic_unitary`
    Construct the complex-to-real harmonic basis-function unitary.
:func:`rodrigues_rotation`
    Construct a rotation matrix with Rodrigues' formula.
:func:`wigner_d`
    Construct a Wigner D matrix for an active z--y--z rotation.
:func:`wigner_small_d`
    Construct a Wigner small-d matrix from its finite factorial sum.

Notes
-----
The Cartesian and Wigner functions both describe active rotations. A zero
Rodrigues axis gives the identity matrix. Bond poles use azimuth zero, which
is a pure-gauge convention for the composed Slater--Koster block.
"""

import math

import jax.numpy as jnp
from beartype import beartype
from beartype.typing import List, Tuple
from jaxtyping import Array, Complex128, Float64, jaxtyped

from diffpes.constants import (
    L_MAX,
)
from diffpes.types import (
    ScalarFloat,
)

from .safe import safe_arccos, safe_arctan2, safe_divide, safe_norm


def _validate_l(l: int) -> None:
    """PRIVATE: Validate one static angular-momentum quantum number.

    Parameters
    ----------
    l : int
        Candidate static orbital angular momentum.

    Raises
    ------
    ValueError
        If ``l`` lies outside ``[0, L_MAX]``.

    Notes
    -----
    The check reads the static Python value before any Wigner table
    construction, so ``l`` participates in the JAX trace signature.
    """
    if l < 0 or l > L_MAX:
        message: str = f"l={l} must satisfy 0 <= l <= {L_MAX}"
        raise ValueError(message)


@jaxtyped(typechecker=beartype)
def wigner_small_d(  # noqa: DOC502 -- validation is shared in _validate_l.
    l: int,
    beta: ScalarFloat,
) -> Float64[Array, "m1 m2"]:
    r"""Construct a Wigner small-d matrix from its finite factorial sum.

    The function evaluates the matrix of an active y-axis rotation in the
    Condon--Shortley basis. Rows index :math:`m'` and columns index
    :math:`m`, both in ascending order from :math:`-l` to :math:`+l`.
    Python unrolls the finite sum because ``l`` is static, while ``beta``
    remains differentiable.

    :see: :class:`~.test_rotations.TestWignerSmallD`

    Parameters
    ----------
    l : int
        (**static** -- a compile-time constant; changing it triggers
        retracing) Angular momentum, restricted to ``0 <= l <= 4``.
    beta : ScalarFloat
        Active y-axis rotation angle in radians.

    Returns
    -------
    small_d : Float64[Array, "m1 m2"]
        Real matrix with shape ``(2*l + 1, 2*l + 1)``.

    Raises
    ------
    ValueError
        If ``l`` lies outside the supported interval from zero through four.

    Notes
    -----
    Each element uses

    .. math::

        d^l_{m'm}(\beta) = A_{m'm}\sum_k
        \frac{(-1)^{m'-m+k}
        c^{2l+m-m'-2k}s^{m'-m+2k}}
        {(l+m-k)!k!(m'-m+k)!(l-m'-k)!},

    where :math:`c=\cos(\beta/2)`, :math:`s=\sin(\beta/2)`, and
    :math:`A_{m'm}` is the square root of the four magnetic-number
    factorials. The integer bounds admit only nonnegative factorial arguments
    and powers.
    """
    _validate_l(l)
    beta_array: Float64[Array, ""] = jnp.asarray(beta)
    cosine_half: Float64[Array, ""] = jnp.cos(0.5 * beta_array)
    sine_half: Float64[Array, ""] = jnp.sin(0.5 * beta_array)
    rows: List[Float64[Array, " m2"]] = []
    m_prime: int
    m: int
    k: int
    for m_prime in range(-l, l + 1):
        entries: List[Float64[Array, ""]] = []
        for m in range(-l, l + 1):
            prefactor: float = math.sqrt(
                math.factorial(l + m)
                * math.factorial(l - m)
                * math.factorial(l + m_prime)
                * math.factorial(l - m_prime)
            )
            k_min: int = max(0, m - m_prime)
            k_max: int = min(l + m, l - m_prime)
            element: Float64[Array, ""] = jnp.zeros_like(beta_array)
            for k in range(k_min, k_max + 1):
                denominator: int = (
                    math.factorial(l + m - k)
                    * math.factorial(k)
                    * math.factorial(m_prime - m + k)
                    * math.factorial(l - m_prime - k)
                )
                cosine_power: int = 2 * l + m - m_prime - 2 * k
                sine_power: int = m_prime - m + 2 * k
                coefficient: float = (
                    prefactor * (-1) ** (m_prime - m + k) / denominator
                )
                element = element + (
                    coefficient
                    * cosine_half**cosine_power
                    * sine_half**sine_power
                )
            entries.append(element)
        rows.append(jnp.stack(entries))
    small_d: Float64[Array, "m1 m2"] = jnp.stack(rows)
    return small_d


@jaxtyped(typechecker=beartype)
def wigner_d(  # noqa: DOC502 -- validation is shared in _validate_l.
    l: int,
    alpha: ScalarFloat,
    beta: ScalarFloat,
    gamma: ScalarFloat,
) -> Complex128[Array, "m1 m2"]:
    r"""Construct a Wigner D matrix for an active z--y--z rotation.

    The function dresses :func:`wigner_small_d` with the two diagonal z-axis
    phase matrices. Rows index :math:`m'` and columns index :math:`m` in
    ascending magnetic-number order.

    :see: :class:`~.test_rotations.TestWignerD`

    Parameters
    ----------
    l : int
        (**static** -- a compile-time constant; changing it triggers
        retracing) Angular momentum, restricted to ``0 <= l <= 4``.
    alpha : ScalarFloat
        First active z-axis angle in radians.
    beta : ScalarFloat
        Active y-axis angle in radians.
    gamma : ScalarFloat
        Final active z-axis angle in radians.

    Returns
    -------
    matrix : Complex128[Array, "m1 m2"]
        Complex Wigner matrix with shape ``(2*l + 1, 2*l + 1)``.

    Raises
    ------
    ValueError
        If ``l`` lies outside the supported interval from zero through four.

    Notes
    -----
    The implemented convention is
    :math:`D^l_{m'm}=e^{-im'\alpha}d^l_{m'm}(\beta)e^{-im\gamma}`.
    This is the active z--y--z convention registered in the ARPES forward
    canon.
    """
    _validate_l(l)
    alpha_array: Float64[Array, ""] = jnp.asarray(alpha)
    beta_array: Float64[Array, ""] = jnp.asarray(beta)
    gamma_array: Float64[Array, ""] = jnp.asarray(gamma)
    magnetic_numbers: Float64[Array, " m"] = jnp.arange(
        -l,
        l + 1,
        dtype=beta_array.dtype,
    )
    alpha_phases: Complex128[Array, " m"] = jnp.exp(
        -1j * magnetic_numbers * alpha_array
    )
    gamma_phases: Complex128[Array, " m"] = jnp.exp(
        -1j * magnetic_numbers * gamma_array
    )
    small_d: Float64[Array, "m1 m2"] = wigner_small_d(l, beta_array)
    matrix: Complex128[Array, "m1 m2"] = (
        alpha_phases[:, None] * small_d * gamma_phases[None, :]
    )
    return matrix


@jaxtyped(typechecker=beartype)
def real_harmonic_unitary(  # noqa: DOC502 -- validation is shared in _validate_l.
    l: int,
) -> Complex128[Array, "m1 m2"]:
    r"""Construct the complex-to-real harmonic basis-function unitary.

    The returned matrix obeys
    :math:`Y_l^{\mathrm{real}}=U^{(l)}Y_l^{\mathrm{complex}}`.
    Rows carry real-harmonic indices and columns carry complex-harmonic
    indices, with both axes ordered from :math:`-l` to :math:`+l`. This gives
    the VASP-compatible shell orders ``(p_y, p_z, p_x)`` and
    ``(d_xy, d_yz, d_z2, d_xz, d_x2-y2)``.

    :see: :class:`~.test_rotations.TestRealHarmonicUnitary`

    Parameters
    ----------
    l : int
        Angular momentum, restricted to ``0 <= l <= 4``.

    Returns
    -------
    unitary : Complex128[Array, "m1 m2"]
        Complex unitary with shape ``(2*l + 1, 2*l + 1)``.

    Raises
    ------
    ValueError
        If ``l`` lies outside the supported interval from zero through four.

    Notes
    -----
    For positive ``m``, the real cosine row has coefficients
    :math:`[Y_l^{-m}+(-1)^mY_l^m]/\sqrt2`. The negative ``m`` sine row has
    :math:`i[Y_l^{-m}-(-1)^mY_l^m]/\sqrt2`. The zero row is unchanged.
    Coefficient vectors therefore transform as
    :math:`c_\mathrm{complex}=U^T c_\mathrm{real}`, and operators transform
    as :math:`O_\mathrm{real}=U^*O_\mathrm{complex}U^T`.
    """
    _validate_l(l)
    size: int = 2 * l + 1
    unitary: Complex128[Array, "m1 m2"] = jnp.zeros(
        (size, size),
        dtype=jnp.complex128,
    )
    unitary = unitary.at[l, l].set(1.0 + 0.0j)
    m: int
    inverse_sqrt_two: float = 1.0 / math.sqrt(2.0)
    for m in range(1, l + 1):
        negative_index: int = l - m
        positive_index: int = l + m
        phase: int = (-1) ** m
        unitary = unitary.at[positive_index, negative_index].set(
            inverse_sqrt_two
        )
        unitary = unitary.at[positive_index, positive_index].set(
            phase * inverse_sqrt_two
        )
        unitary = unitary.at[negative_index, negative_index].set(
            1j * inverse_sqrt_two
        )
        unitary = unitary.at[negative_index, positive_index].set(
            -1j * phase * inverse_sqrt_two
        )
    return unitary


@jaxtyped(typechecker=beartype)
def bond_angles(
    bond_cart: Float64[Array, " 3"],
) -> Tuple[Float64[Array, ""], Float64[Array, ""]]:
    r"""Convert a Cartesian bond to safe polar and azimuthal angles.

    The function returns the polar angle from positive z followed by the
    azimuth from positive x. Exact positive and negative z-axis bonds use
    azimuth zero. A zero bond also maps to ``(0, 0)`` as a finite guard value;
    physical callers must exclude zero-length neighbor bonds.

    :see: :class:`~.test_rotations.TestBondAngles`

    Parameters
    ----------
    bond_cart : Float64[Array, " 3"]
        Cartesian bond vector.

    Returns
    -------
    beta : Float64[Array, ""]
        Polar angle in radians on the closed interval ``[0, pi]``.
    alpha : Float64[Array, ""]
        Azimuthal angle in radians on the interval ``[-pi, pi]``.

    Notes
    -----
    :func:`safe_divide` supplies a cosine of one at zero norm, and
    :func:`safe_arccos` sanitizes its endpoint derivative. The azimuth uses
    :func:`safe_arctan2`, whose double selection evaluates a finite inactive
    branch and selects zero value and gradient at either pole. Only composed
    Production differentiates only composed Slater--Koster blocks and never
    treats these singular Euler coordinates as observables.
    """
    norm: Float64[Array, ""] = safe_norm(bond_cart)
    cosine_beta: Float64[Array, ""] = safe_divide(
        bond_cart[2],
        norm,
        fallback=1.0,
    )
    beta: Float64[Array, ""] = safe_arccos(cosine_beta)
    alpha: Float64[Array, ""] = safe_arctan2(bond_cart[1], bond_cart[0])
    angles: Tuple[Float64[Array, ""], Float64[Array, ""]] = (beta, alpha)
    return angles


@jaxtyped(typechecker=beartype)
def rodrigues_rotation(
    axis: Float64[Array, "3"],
    angle: ScalarFloat,
) -> Float64[Array, "3 3"]:
    r"""Construct a rotation matrix with Rodrigues' formula.

    The function safely normalizes the rotation axis. It then constructs an
    active rotation matrix for column vectors.

    :see: :class:`~.test_rotations.TestRodriguesRotation`

    Parameters
    ----------
    axis : Float64[Array, "3"]
        Rotation axis in Cartesian coordinates.
    angle : ScalarFloat
        Active rotation angle in radians.

    Returns
    -------
    rotation : Float64[Array, "3 3"]
        Active Cartesian rotation matrix.

    Notes
    -----
    The function uses
    :math:`R = I + \sin(\alpha)[\hat n]_\times +
    (1 - \cos(\alpha))[\hat n]_\times^2`.
    The safe normalization gives the identity matrix for a zero axis.
    """
    angle_array: Float64[Array, ""] = jnp.asarray(angle)
    axis_norm: Float64[Array, ""] = safe_norm(axis)
    normalized_axis: Float64[Array, "3"] = safe_divide(axis, axis_norm)
    axis_x: Float64[Array, ""] = normalized_axis[0]
    axis_y: Float64[Array, ""] = normalized_axis[1]
    axis_z: Float64[Array, ""] = normalized_axis[2]
    zero: Float64[Array, ""] = jnp.zeros_like(axis_x)
    skew: Float64[Array, "3 3"] = jnp.stack(
        (
            jnp.stack((zero, -axis_z, axis_y)),
            jnp.stack((axis_z, zero, -axis_x)),
            jnp.stack((-axis_y, axis_x, zero)),
        )
    )
    identity: Float64[Array, "3 3"] = jnp.eye(3, dtype=axis.dtype)
    skew_squared: Float64[Array, "3 3"] = skew @ skew
    rotation: Float64[Array, "3 3"] = (
        identity
        + jnp.sin(angle_array) * skew
        + (1.0 - jnp.cos(angle_array)) * skew_squared
    )
    return rotation


__all__: list[str] = [
    "bond_angles",
    "real_harmonic_unitary",
    "rodrigues_rotation",
    "wigner_d",
    "wigner_small_d",
]
