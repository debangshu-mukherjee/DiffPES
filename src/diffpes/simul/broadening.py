"""Compute energy broadening functions for ARPES simulations.

Extended Summary
----------------
The module provides JAX-compatible broadening profiles, including Gaussian
(instrumental resolution), true Voigt (combined Gaussian-Lorentzian),
and Fermi-Dirac thermal occupation functions.

Routine Listings
----------------
:func:`fermi_dirac`
    Compute Fermi-Dirac distribution value.
:func:`gaussian`
    Compute normalized Gaussian broadening profile.
:func:`voigt`
    Compute a normalized Voigt profile through the Faddeeva function.

Notes
-----
JAX can compile all functions. They support ``jax.vmap``
for vectorized evaluation across k-points and bands.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
from beartype import beartype
from jaxtyping import Array, Complex, Float, jaxtyped

from diffpes.types import KB_EV_PER_K, ScalarFloat
from diffpes.utils import faddeeva


@jaxtyped(typechecker=beartype)
def gaussian(  # noqa: DOC502 -- eqx.error_if raises under JAX execution.
    energy_range: Float[Array, " E"],
    center: ScalarFloat,
    sigma: ScalarFloat,
) -> Float[Array, " E"]:
    """Compute normalized Gaussian broadening profile.

    Evaluates a Gaussian lineshape centered at ``center`` with standard
    deviation ``sigma``, normalized so that the integral over all energies
    equals unity.

    :see: :class:`~.test_broadening.TestGaussian`

    Implementation Logic
    --------------------
    The function evaluates the analytic Gaussian probability density::

        G(E) = exp(-(E - E0)^2 / (2 * sigma^2))
               / (sqrt(2 * pi) * sigma)

    1. **Compute energy differences**::

           diff = energy_range - center

       Shifts the energy axis so the peak is at the origin.

    2. **Compute normalization factor**::

           norm_factor = sqrt(2 * pi) * sigma

       This prefactor ensures the profile integrates to unity over
       (-inf, +inf). Thus, the Gaussian has a unit area.

    3. **Evaluate Gaussian profile**::

           profile = exp(-diff^2 / (2 * sigma^2)) / norm_factor

       Element-wise evaluation of the normalized Gaussian at each
       energy point.

    Parameters
    ----------
    energy_range : Float[Array, " E"]
        Energy axis values in eV.
    center : ScalarFloat
        Center energy of the peak in eV.
    sigma : ScalarFloat
        Strictly positive Gaussian standard deviation in eV.

    Returns
    -------
    profile : Float[Array, " E"]
        Normalized Gaussian profile values.

    Raises
    ------
    EquinoxRuntimeError
        If ``sigma`` is non-finite or not strictly positive.

    Notes
    -----
    No normalized Gaussian density exists at ``sigma = 0``. The function
    therefore validates the physical width rather than fabricating a finite
    profile through a guarded elementary operation.
    """
    sigma_array: Float[Array, ""] = jnp.asarray(sigma, dtype=jnp.float64)
    checked_sigma: Float[Array, ""] = eqx.error_if(
        sigma_array,
        ~jnp.isfinite(sigma_array) | (sigma_array <= 0.0),
        "sigma must be finite and strictly positive",
    )
    diff: Float[Array, " E"] = energy_range - center
    norm_factor: Float[Array, " "] = jnp.sqrt(2.0 * jnp.pi) * checked_sigma
    profile: Float[Array, " E"] = (
        jnp.exp(-(diff**2) / (2.0 * checked_sigma**2)) / norm_factor
    )
    return profile


@jaxtyped(typechecker=beartype)
def voigt(  # noqa: DOC502 -- eqx.error_if raises under JAX execution.
    energy_range: Float[Array, " E"],
    center: ScalarFloat,
    sigma: ScalarFloat,
    gamma: ScalarFloat,
) -> Float[Array, " E"]:
    r"""Compute a normalized Voigt profile through the Faddeeva function.

    The function combines a Gaussian instrument width with a Lorentzian
    lifetime width. It follows the standard Faddeeva definition [1]_ and the
    certified Weideman primal from WP7.1 [2]_.

    :see: :class:`~.test_broadening.TestVoigt`

    For positive Gaussian and Lorentzian widths, the function evaluates

    .. math::

        V(E) = \frac{\operatorname{Re} w(z)}
                    {\sigma\sqrt{2\pi}},
        \qquad
        z = \frac{E-E_0+i\gamma}{\sigma\sqrt{2}},

    where :math:`w` is the Faddeeva function. The exact zero-width rays use
    their analytic Gaussian and Cauchy limits.

    Parameters
    ----------
    energy_range : Float[Array, " E"]
        Energy axis values in eV.
    center : ScalarFloat
        Center energy of the peak in eV.
    sigma : ScalarFloat
        Nonnegative Gaussian standard deviation in eV.
    gamma : ScalarFloat
        Nonnegative Lorentzian half-width at half-maximum in eV.

    Returns
    -------
    profile : Float[Array, " E"]
        Normalized Voigt profile values.

    Raises
    ------
    EquinoxRuntimeError
        If an energy coordinate or ``center`` is non-finite.
        Also raised for invalid widths or an out-of-envelope positive call.

    Notes
    -----
    The positive-width interior differentiates the validated Faddeeva primal
    directly. The exact ``sigma = 0`` and ``gamma = 0`` branches are
    value-only contracts; endpoint JVPs and VJPs are not certified. Analytic
    endpoints bypass the Faddeeva envelope. Finite inactive branches prevent
    zero-width divisions from poisoning JAX transformations.

    References
    ----------
    .. [1] Abramowitz, M. and Stegun, I. A., eds., *Handbook of Mathematical
       Functions*, section 7.1, Dover, 1972.
    .. [2] Weideman, J. A. C., "Computation of the Complex Error Function",
       SIAM J. Numer. Anal. 31, 1497-1518 (1994).
    """
    energy_array: Float[Array, " E"] = jnp.asarray(
        energy_range,
        dtype=jnp.float64,
    )
    center_array: Float[Array, ""] = jnp.asarray(center, dtype=jnp.float64)
    sigma_array: Float[Array, ""] = jnp.asarray(sigma, dtype=jnp.float64)
    gamma_array: Float[Array, ""] = jnp.asarray(gamma, dtype=jnp.float64)
    checked_energy: Float[Array, " E"] = eqx.error_if(
        energy_array,
        ~jnp.all(jnp.isfinite(energy_array)),
        "energy_range must be finite",
    )
    checked_center: Float[Array, ""] = eqx.error_if(
        center_array,
        ~jnp.isfinite(center_array),
        "center must be finite",
    )
    checked_sigma: Float[Array, ""] = eqx.error_if(
        sigma_array,
        ~jnp.isfinite(sigma_array) | (sigma_array < 0.0),
        "sigma must be finite and nonnegative",
    )
    checked_gamma: Float[Array, ""] = eqx.error_if(
        gamma_array,
        ~jnp.isfinite(gamma_array) | (gamma_array < 0.0),
        "gamma must be finite and nonnegative",
    )
    checked_sigma = eqx.error_if(
        checked_sigma,
        (checked_sigma == 0.0) & (checked_gamma == 0.0),
        "sigma and gamma must not both be zero",
    )
    maximum_absolute_value: float = 1.0e8
    interior: Array = (checked_sigma > 0.0) & (checked_gamma > 0.0)
    safe_sigma: Float[Array, ""] = jnp.where(
        checked_sigma > 0.0,
        checked_sigma,
        jnp.float64(1.0),
    )
    safe_gamma: Float[Array, ""] = jnp.where(
        checked_gamma > 0.0,
        checked_gamma,
        jnp.float64(1.0),
    )
    displacement: Float[Array, " E"] = checked_energy - checked_center
    candidate_z: Complex[Array, " E"] = (displacement + 1j * checked_gamma) / (
        safe_sigma * jnp.sqrt(jnp.float64(2.0))
    )
    inactive_z: Complex[Array, " E"] = jnp.zeros_like(candidate_z)
    safe_z: Complex[Array, " E"] = jnp.where(
        interior,
        candidate_z,
        inactive_z,
    )
    invalid_z: Array = interior & jnp.any(
        ~jnp.isfinite(candidate_z)
        | (jnp.abs(candidate_z) > maximum_absolute_value)
    )
    bounded_z: Complex[Array, " E"] = jnp.where(
        invalid_z,
        inactive_z,
        safe_z,
    )
    checked_z: Complex[Array, " E"] = eqx.error_if(
        bounded_z,
        invalid_z,
        "positive-width arguments must remain inside the Faddeeva envelope "
        "with finite abs(z) <= 1e8",
    )
    interior_profile: Float[Array, " E"] = jnp.real(faddeeva(checked_z)) / (
        safe_sigma * jnp.sqrt(2.0 * jnp.pi)
    )
    gaussian_profile: Float[Array, " E"] = gaussian(
        checked_energy,
        checked_center,
        safe_sigma,
    )
    cauchy_profile: Float[Array, " E"] = safe_gamma / (
        jnp.pi * (displacement**2 + safe_gamma**2)
    )
    profile: Float[Array, " E"] = jnp.where(
        checked_sigma == 0.0,
        cauchy_profile,
        jnp.where(
            checked_gamma == 0.0,
            gaussian_profile,
            interior_profile,
        ),
    )
    return profile


@jaxtyped(typechecker=beartype)
def fermi_dirac(  # noqa: DOC502 -- eqx.error_if raises under JAX execution.
    energy: ScalarFloat,
    fermi_energy: ScalarFloat,
    temperature: ScalarFloat,
) -> Float[Array, " "]:
    """Compute Fermi-Dirac distribution value.

    Evaluates the Fermi-Dirac thermal occupation function at a given
    energy, Fermi level, and temperature::

        f(E) = 1 / (1 + exp((E - Ef) / (kB * T)))

    :see: :class:`~.test_broadening.TestFermiDirac`

    Implementation Logic
    --------------------
    1. **Compute thermal energy kT**::

           kt = kB * T

       Multiplies the Boltzmann constant kB = 8.617333e-5 eV/K by the
       temperature in Kelvin to obtain the thermal energy scale. Both
       values are cast to float64 for numerical precision.

    2. **Validate the thermal domain**::

           checked_temperature = error_if(T, ~isfinite(T) or T <= 0)

       Rejects zero, negative, and non-finite temperatures. The finite-
       temperature Fermi-Dirac formula has no classical derivative at its
       zero-temperature limit.

    3. **Evaluate Fermi-Dirac function**::

           exponent = (E - Ef) / kt
           occupation = sigmoid(-exponent)

       Computes the occupation probability. For E << Ef the result
       approaches 1 (filled states); for E >> Ef it approaches 0
       (empty states).

    Parameters
    ----------
    energy : ScalarFloat
        Electron energy in eV.
    fermi_energy : ScalarFloat
        Fermi level energy in eV.
    temperature : ScalarFloat
        Finite, strictly positive temperature in kelvin.

    Returns
    -------
    occupation : Float[Array, " "]
        Fermi-Dirac occupation (0 to 1).

    Raises
    ------
    EquinoxRuntimeError
        If ``temperature`` is non-finite or not strictly positive.

    Notes
    -----
    Uses the Boltzmann constant kB = 8.617333e-5 eV/K, imported as
    :obj:`~diffpes.types.KB_EV_PER_K`. ``jax.nn.sigmoid`` is algebraically
    identical to the reciprocal-exponential expression but has an
    overflow-safe JVP. Values and derivatives therefore underflow to finite
    exact zeros far above the Fermi level instead of becoming NaN. The
    function does not approximate the discontinuous zero-temperature step.
    A separate static zero-temperature model must define that limit and its
    derivative policy.
    """
    temperature_array: Float[Array, ""] = jnp.asarray(
        temperature,
        dtype=jnp.float64,
    )
    checked_temperature: Float[Array, ""] = eqx.error_if(
        temperature_array,
        ~jnp.isfinite(temperature_array) | (temperature_array <= 0.0),
        "temperature must be finite and strictly positive",
    )
    kt: Float[Array, " "] = (
        jnp.asarray(KB_EV_PER_K, dtype=jnp.float64) * checked_temperature
    )
    exponent: Float[Array, " "] = (
        jnp.asarray(energy, dtype=jnp.float64)
        - jnp.asarray(fermi_energy, dtype=jnp.float64)
    ) / kt
    occupation: Float[Array, " "] = jax.nn.sigmoid(-exponent)
    return occupation


__all__: list[str] = [
    "fermi_dirac",
    "gaussian",
    "voigt",
]
