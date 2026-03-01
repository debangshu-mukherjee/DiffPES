r"""Radial-integral evaluation utilities.

Extended Summary
----------------
Implements fixed-grid quadrature for dipole radial integrals

.. math::

    B^{l'}(k) = (i)^{l'} \int_0^\infty R(r) r^3 j_{l'}(k r) dr

using JAX-traceable composite trapezoidal integration.

Routine Listings
----------------
:func:`radial_integral`
    Evaluate dipole radial integral on a fixed radial grid.
"""

import jax.numpy as jnp
from beartype import beartype
from jaxtyping import Array, Complex, Float, jaxtyped

from .bessel import spherical_bessel_jl


@jaxtyped(typechecker=beartype)
def radial_integral(
    k: Float[Array, " ..."],
    r: Float[Array, " R"],
    radial_values: Float[Array, " R"],
    l_prime: int,
) -> Complex[Array, " ..."]:
    r"""Evaluate dipole radial integral on a fixed radial grid.

    Extended Summary
    ----------------
    Computes the dipole radial integral that appears in the partial-wave
    expansion of the photoemission matrix element:

    .. math::

        B^{l'}(k) = i^{l'} \int_0^\infty R(r) \, r^3 \, j_{l'}(k r) \, dr

    The computation proceeds in four steps:

    1. **Outer product**: Form the 2-D array :math:`kr` by broadcasting
       ``k`` (shape ``...``) against ``r`` (shape ``R``) via
       ``jnp.expand_dims``. This produces shape ``(... , R)``.

    2. **Bessel evaluation**: Evaluate :math:`j_{l'}(kr)` element-wise
       using `spherical_bessel_jl`, which handles the small-argument
       limit internally.

    3. **Integrand assembly**: Multiply the Bessel values by the
       radial factor :math:`R(r) \cdot r^3` (the :math:`r^3` weighting
       comes from the :math:`r^2 dr` volume element times the dipole
       operator factor :math:`r`).

    4. **Trapezoidal quadrature**: Integrate over the radial grid
       using ``jnp.trapezoid`` along the last axis. The grid need not
       be uniform; ``jnp.trapezoid`` handles non-uniform spacing via
       the ``x`` argument.

    5. **Phase factor**: Multiply the real integral by the complex
       phase :math:`i^{l'}` to obtain the final complex-valued result.

    The function supports batched evaluation: if ``k`` has shape ``(...)``,
    the output has the same leading shape, with the radial integration
    contracted over the last axis.

    Parameters
    ----------
    k : Float[Array, " ..."]
        Momentum magnitude(s) where the integral is evaluated.
    r : Float[Array, " R"]
        Monotonic radial grid.
    radial_values : Float[Array, " R"]
        Radial wavefunction values sampled on ``r``.
    l_prime : int
        Final-state angular momentum order.

    Returns
    -------
    values : Complex[Array, " ..."]
        Complex radial integral values with the same leading shape as ``k``.

    Notes
    -----
    The accuracy of the result depends on the density and extent of
    the radial grid. For Slater-type orbitals that decay as
    :math:`e^{-\zeta r}`, a grid extending to :math:`r \sim 10/\zeta`
    with ~200--500 points is typically sufficient.

    The trapezoidal rule is chosen for simplicity and JAX compatibility.
    Higher-order quadrature (e.g., Simpson's rule) could improve
    accuracy for smooth integrands on coarse grids but would require
    custom JAX-traceable implementations.
    """
    if l_prime < 0:
        msg = "l_prime must be non-negative"
        raise ValueError(msg)

    k_arr: Float[Array, " ..."] = jnp.asarray(k, dtype=jnp.float64)
    r_arr: Float[Array, " R"] = jnp.asarray(r, dtype=jnp.float64)
    radial_arr: Float[Array, " R"] = jnp.asarray(
        radial_values, dtype=jnp.float64
    )

    kr: Float[Array, " ... R"] = jnp.expand_dims(k_arr, axis=-1) * r_arr
    bessel_vals: Float[Array, " ... R"] = spherical_bessel_jl(l_prime, kr)
    radial_factor: Float[Array, " R"] = radial_arr * (r_arr**3)
    integrand: Float[Array, " ... R"] = bessel_vals * radial_factor
    real_integral: Float[Array, " ..."] = jnp.trapezoid(
        integrand,
        x=r_arr,
        axis=-1,
    )

    phase: Complex[Array, " "] = jnp.asarray(
        (1j) ** l_prime, dtype=jnp.complex128
    )
    values: Complex[Array, " ..."] = phase * real_integral.astype(
        jnp.complex128
    )
    return values


__all__: list[str] = ["radial_integral"]
