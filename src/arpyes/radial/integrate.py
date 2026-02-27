r"""Radial-integral evaluation utilities.

Extended Summary
----------------
Implements fixed-grid quadrature for dipole radial integrals

.. math::

    B^{l'}(k) = (i)^{l'} \int_0^\infty R(r) r^3 j_{l'}(k r) dr

using JAX-traceable composite trapezoidal integration.
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
    """Evaluate dipole radial integral on a fixed radial grid.

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
