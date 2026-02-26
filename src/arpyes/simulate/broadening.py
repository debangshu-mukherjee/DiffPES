"""Energy broadening functions for ARPES simulations.

Extended Summary
----------------
Provides JAX-compatible broadening profiles including Gaussian
(instrumental resolution), Voigt (combined Gaussian-Lorentzian),
and Fermi-Dirac thermal occupation functions.

Routine Listings
----------------
:func:`gaussian`
    Normalized Gaussian broadening profile.
:func:`voigt`
    Normalized Voigt profile via the Faddeeva function.
:func:`fermi_dirac`
    Fermi-Dirac thermal distribution function.

Notes
-----
All functions are JIT-compilable and support ``jax.vmap``
for vectorized evaluation across k-points and bands.
"""

import jax.numpy as jnp
from beartype import beartype
from jaxtyping import Array, Float, jaxtyped

from arpyes.types import ScalarFloat
from arpyes.utils.math import faddeeva

_KB: float = 8.617333e-5


@jaxtyped(typechecker=beartype)
def gaussian(
    energy_range: Float[Array, " E"],
    center: ScalarFloat,
    sigma: ScalarFloat,
) -> Float[Array, " E"]:
    """Compute normalized Gaussian broadening profile.

    Parameters
    ----------
    energy_range : Float[Array, " E"]
        Energy axis values in eV.
    center : ScalarFloat
        Center energy of the peak in eV.
    sigma : ScalarFloat
        Gaussian standard deviation in eV.

    Returns
    -------
    profile : Float[Array, " E"]
        Normalized Gaussian profile values.
    """
    diff: Float[Array, " E"] = energy_range - center
    norm_factor: Float[Array, " "] = (
        jnp.sqrt(2.0 * jnp.pi) * sigma
    )
    profile: Float[Array, " E"] = (
        jnp.exp(-diff**2 / (2.0 * sigma**2)) / norm_factor
    )
    return profile


@jaxtyped(typechecker=beartype)
def voigt(
    energy_range: Float[Array, " E"],
    center: ScalarFloat,
    sigma: ScalarFloat,
    gamma: ScalarFloat,
) -> Float[Array, " E"]:
    """Compute normalized Voigt profile.

    Parameters
    ----------
    energy_range : Float[Array, " E"]
        Energy axis values in eV.
    center : ScalarFloat
        Center energy of the peak in eV.
    sigma : ScalarFloat
        Gaussian standard deviation in eV.
    gamma : ScalarFloat
        Lorentzian half-width at half-maximum in eV.

    Returns
    -------
    profile : Float[Array, " E"]
        Normalized Voigt profile values.

    Notes
    -----
    Uses the Faddeeva function: V(x) = Re[w(z)] / (sigma sqrt(2pi))
    where z = (x - x0 + i*gamma) / (sigma * sqrt(2)).
    """
    z: Float[Array, " E"] = (
        (energy_range - center) + 1j * gamma
    ) / (sigma * jnp.sqrt(2.0))
    w_val: Float[Array, " E"] = faddeeva(z)
    norm_factor: Float[Array, " "] = (
        sigma * jnp.sqrt(2.0 * jnp.pi)
    )
    profile: Float[Array, " E"] = (
        jnp.real(w_val) / norm_factor
    )
    return profile


@jaxtyped(typechecker=beartype)
def fermi_dirac(
    energy: ScalarFloat,
    fermi_energy: ScalarFloat,
    temperature: ScalarFloat,
) -> Float[Array, " "]:
    """Compute Fermi-Dirac distribution value.

    Parameters
    ----------
    energy : ScalarFloat
        Electron energy in eV.
    fermi_energy : ScalarFloat
        Fermi level energy in eV.
    temperature : ScalarFloat
        Temperature in Kelvin.

    Returns
    -------
    occupation : Float[Array, " "]
        Fermi-Dirac occupation (0 to 1).

    Notes
    -----
    f(E) = 1 / (1 + exp((E - Ef) / (kB * T)))
    Uses Boltzmann constant kB = 8.617333e-5 eV/K.
    """
    kt: Float[Array, " "] = jnp.asarray(
        _KB, dtype=jnp.float64
    ) * jnp.asarray(temperature, dtype=jnp.float64)
    safe_kt: Float[Array, " "] = jnp.where(
        kt > 0.0, kt, jnp.float64(1e-10)
    )
    exponent: Float[Array, " "] = (
        jnp.asarray(energy, dtype=jnp.float64)
        - jnp.asarray(fermi_energy, dtype=jnp.float64)
    ) / safe_kt
    occupation: Float[Array, " "] = 1.0 / (
        1.0 + jnp.exp(exponent)
    )
    return occupation


__all__: list[str] = [
    "fermi_dirac",
    "gaussian",
    "voigt",
]
