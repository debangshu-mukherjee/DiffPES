"""Atomic radial wavefunction models in JAX.

Extended Summary
----------------
Provides normalized Slater-type and hydrogenic radial wavefunctions
for use in differentiable ARPES matrix-element calculations.

Routine Listings
----------------
:func:`slater_radial`
    Evaluate normalized Slater-type radial function.
:func:`hydrogenic_radial`
    Evaluate normalized hydrogenic radial function.
"""

import math

import jax
import jax.numpy as jnp
from beartype import beartype
from jaxtyping import Array, Float, jaxtyped

from diffpes.types import ScalarFloat


def _associated_laguerre(
    order: int,
    alpha: int,
    x: Float[Array, " ..."],
) -> Float[Array, " ..."]:
    r"""Evaluate associated Laguerre polynomial.

    Computes :math:`L_n^\alpha(x)`.

    Extended Summary
    ----------------
    Computes the generalized Laguerre polynomial using the standard
    three-term recurrence relation, which is numerically stable for
    upward iteration in the polynomial order.

    **Seed values:**

    .. math::

        L_0^\alpha(x) = 1

        L_1^\alpha(x) = 1 + \alpha - x

    **Upward recurrence** (for n >= 2):

    .. math::

        n \, L_n^\alpha(x) = (2n - 1 + \alpha - x) \, L_{n-1}^\alpha(x)
                            - (n - 1 + \alpha) \, L_{n-2}^\alpha(x)

    This is implemented via ``jax.lax.fori_loop`` for JIT compatibility,
    carrying the pair :math:`(L_{n-2}^\alpha, L_{n-1}^\alpha)` and
    advancing one order per iteration from n=2 up to ``order``.

    The generalized Laguerre polynomials appear in the hydrogenic
    radial wavefunctions as :math:`L_{n-l-1}^{2l+1}(\rho)` where
    :math:`\rho = 2 Z_{\text{eff}} r / n`. They are orthogonal on
    :math:`[0, \infty)` with weight :math:`x^\alpha e^{-x}`:

    .. math::

        \int_0^\infty x^\alpha e^{-x} L_n^\alpha(x) L_m^\alpha(x) \, dx
        = \frac{\Gamma(n + \alpha + 1)}{n!} \, \delta_{nm}

    Parameters
    ----------
    order : int
        Polynomial order (n >= 0).
    alpha : int
        Generalization parameter (alpha >= 0). For hydrogenic
        wavefunctions, alpha = 2*l + 1.
    x : Float[Array, " ..."]
        Evaluation points.

    Returns
    -------
    values : Float[Array, " ..."]
        :math:`L_n^\alpha(x)` evaluated element-wise.

    Raises
    ------
    ValueError
        If ``order`` or ``alpha`` is negative.
    """
    if order < 0:
        msg: str = "order must be non-negative"
        raise ValueError(msg)
    if alpha < 0:
        msg: str = "alpha must be non-negative"
        raise ValueError(msg)

    x_arr: Float[Array, " ..."] = jnp.asarray(x, dtype=jnp.float64)
    laguerre_zero: Float[Array, " ..."] = jnp.ones_like(x_arr)
    if order == 0:
        return laguerre_zero

    alpha_arr: Float[Array, " "] = jnp.asarray(alpha, dtype=jnp.float64)
    laguerre_one: Float[Array, " ..."] = 1.0 + alpha_arr - x_arr
    if order == 1:
        return laguerre_one

    def _recurrence_step(
        current_order: int,
        state: tuple[Float[Array, " ..."], Float[Array, " ..."]],
    ) -> tuple[Float[Array, " ..."], Float[Array, " ..."]]:
        laguerre_prev_prev: Float[Array, " ..."]
        laguerre_prev: Float[Array, " ..."]
        laguerre_prev_prev, laguerre_prev = state
        order_arr: Float[Array, " "] = jnp.asarray(
            current_order, dtype=jnp.float64
        )
        prefactor: Float[Array, " ..."] = (
            2.0 * order_arr - 1.0 + alpha_arr - x_arr
        ) / order_arr
        correction: Float[Array, " ..."] = (
            (order_arr - 1.0 + alpha_arr) / order_arr
        ) * laguerre_prev_prev
        laguerre_curr: Float[Array, " ..."] = (
            prefactor * laguerre_prev - correction
        )
        return laguerre_prev, laguerre_curr

    laguerre_final: Float[Array, " ..."]
    _, laguerre_final = jax.lax.fori_loop(
        2,
        order + 1,
        _recurrence_step,
        (laguerre_zero, laguerre_one),
    )
    return laguerre_final


@jaxtyped(typechecker=beartype)
def slater_radial(
    r: Float[Array, " ..."],
    n: int,
    zeta: ScalarFloat,
) -> Float[Array, " ..."]:
    r"""Evaluate normalized Slater-type radial function.

    Extended Summary
    ----------------
    Computes the Slater-type orbital (STO) radial function:

    .. math::

        R(r) = N \, r^{n-1} \, e^{-\zeta r}

    where the normalization constant :math:`N` is chosen so that
    :math:`\int_0^\infty |R(r)|^2 r^2 dr = 1`:

    .. math::

        N = \frac{(2\zeta)^{n + 1/2}}{\sqrt{(2n)!}}

    **Slater vs. hydrogenic models:**

    Slater-type orbitals are simpler than hydrogenic radial functions
    because they lack the associated Laguerre polynomial factor. They
    have the correct exponential decay and cusp behavior at the
    nucleus, making them popular as basis functions in quantum
    chemistry. However, they do not possess radial nodes (except at
    r = 0 and r = infinity), unlike the exact hydrogenic solutions.

    The Slater exponent :math:`\zeta` encodes the effective nuclear
    charge and screening. It is typically fitted to reproduce
    Hartree-Fock atomic orbitals (e.g., Clementi-Raimondi rules) or
    optimized variationally.

    **Normalization derivation:**

    The radial normalization integral is:

    .. math::

        \int_0^\infty r^{2(n-1)} e^{-2\zeta r} r^2 dr
        = \int_0^\infty r^{2n} e^{-2\zeta r} dr
        = \frac{(2n)!}{(2\zeta)^{2n+1}}

    Setting :math:`N^2 \cdot (2n)! / (2\zeta)^{2n+1} = 1` gives the
    formula above.

    Parameters
    ----------
    r : Float[Array, " ..."]
        Radial coordinate in atomic units.
    n : int
        Principal quantum number (``n >= 1``).
    zeta : Float[Array, " "]
        Slater exponent.

    Returns
    -------
    values : Float[Array, " ..."]
        Normalized radial function
        ``R(r) = N r^(n-1) exp(-zeta * r)``.

    Notes
    -----
    The ``zeta`` parameter is a JAX array (not a Python float) so that
    it can participate in automatic differentiation. This allows
    gradient-based optimization of Slater exponents in inverse-fitting
    workflows.
    """
    if n < 1:
        msg: str = "n must be >= 1"
        raise ValueError(msg)

    r_arr: Float[Array, " ..."] = jnp.asarray(r, dtype=jnp.float64)
    zeta_arr: Float[Array, " "] = jnp.asarray(zeta, dtype=jnp.float64)
    factorial_term: Float[Array, " "] = jnp.asarray(
        math.factorial(2 * n), dtype=jnp.float64
    )
    norm: Float[Array, " "] = ((2.0 * zeta_arr) ** (n + 0.5)) / jnp.sqrt(
        factorial_term
    )
    values: Float[Array, " ..."] = (
        norm * (r_arr ** (n - 1)) * jnp.exp(-zeta_arr * r_arr)
    )
    return values


@jaxtyped(typechecker=beartype)
def hydrogenic_radial(
    r: Float[Array, " ..."],
    n: int,
    angular_momentum: int,
    z_eff: ScalarFloat,
) -> Float[Array, " ..."]:
    r"""Evaluate normalized hydrogenic radial function.

    Extended Summary
    ----------------
    Computes the exact radial wavefunction for a hydrogenic
    (one-electron) atom with effective nuclear charge :math:`Z_{\text{eff}}`:

    .. math::

        R_{n,l}(r) = N_{n,l} \, e^{-\rho/2} \, \rho^l \,
            L_{n-l-1}^{2l+1}(\rho)

    where :math:`\rho = 2 Z_{\text{eff}} r / n` is the scaled radial
    coordinate, and :math:`L_{n-l-1}^{2l+1}` is the generalized
    Laguerre polynomial evaluated by `_associated_laguerre`.

    **Normalization:**

    The normalization constant is:

    .. math::

        N_{n,l} = \left(\frac{2 Z_{\text{eff}}}{n}\right)^{3/2}
            \sqrt{\frac{(n - l - 1)!}{2n \cdot (n + l)!}}

    This ensures :math:`\int_0^\infty |R_{n,l}(r)|^2 r^2 dr = 1`.
    The factorial ratio is computed using Python's ``math.factorial``
    for exact integer arithmetic, then converted to a JAX scalar
    via ``jnp.sqrt``.

    **Hydrogenic vs. Slater model:**

    Unlike Slater-type orbitals (which are node-free exponentials),
    hydrogenic radial functions have :math:`n - l - 1` radial nodes
    encoded by the zeros of the Laguerre polynomial. This makes them
    exact solutions for hydrogen-like atoms but less commonly used as
    basis functions in multi-electron calculations.

    **Laguerre polynomial recurrence:**

    The associated Laguerre polynomial :math:`L_{n-l-1}^{2l+1}(\rho)`
    is computed by `_associated_laguerre` using upward three-term
    recurrence from order 0 to :math:`n - l - 1`. The recurrence
    is stable in the upward direction and is wrapped in
    ``jax.lax.fori_loop`` for JIT compatibility.

    Parameters
    ----------
    r : Float[Array, " ..."]
        Radial coordinate in atomic units.
    n : int
        Principal quantum number.
    angular_momentum : int
        Angular momentum quantum number (``0 <= angular_momentum < n``).
    z_eff : Float[Array, " "]
        Effective nuclear charge.

    Returns
    -------
    values : Float[Array, " ..."]
        ``R_{n,l}(r)`` for hydrogenic orbitals.

    Notes
    -----
    The ``z_eff`` parameter is a JAX array to support automatic
    differentiation. The quantum numbers ``n`` and ``angular_momentum``
    are Python integers that control the Laguerre polynomial order
    and are baked into the traced computation graph.
    """
    if n < 1:
        msg: str = "n must be >= 1"
        raise ValueError(msg)
    if angular_momentum < 0 or angular_momentum >= n:
        msg: str = "angular_momentum must satisfy 0 <= angular_momentum < n"
        raise ValueError(msg)

    r_arr: Float[Array, " ..."] = jnp.asarray(r, dtype=jnp.float64)
    z_arr: Float[Array, " "] = jnp.asarray(z_eff, dtype=jnp.float64)
    n_float: float = float(n)
    rho: Float[Array, " ..."] = 2.0 * z_arr * r_arr / n_float

    laguerre_order: int = n - angular_momentum - 1
    laguerre_alpha: int = 2 * angular_momentum + 1
    laguerre_values: Float[Array, " ..."] = _associated_laguerre(
        laguerre_order, laguerre_alpha, rho
    )

    factorial_ratio: float = math.factorial(laguerre_order) / (
        2.0 * n_float * math.factorial(n + angular_momentum)
    )
    prefactor: Float[Array, " "] = ((2.0 * z_arr) / n_float) ** 1.5
    norm: Float[Array, " "] = prefactor * jnp.sqrt(
        jnp.asarray(factorial_ratio, dtype=jnp.float64)
    )
    values: Float[Array, " ..."] = (
        norm * jnp.exp(-0.5 * rho) * (rho**angular_momentum) * laguerre_values
    )
    return values


__all__: list[str] = ["hydrogenic_radial", "slater_radial"]
