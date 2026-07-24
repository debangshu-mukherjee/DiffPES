"""Compute energy broadening functions for ARPES simulations.

Extended Summary
----------------
The module provides JAX-compatible broadening profiles, including Gaussian
(instrumental resolution), pseudo-Voigt (combined Gaussian-Lorentzian),
and Fermi-Dirac thermal occupation functions.

Routine Listings
----------------
:func:`fermi_dirac`
    Compute Fermi-Dirac distribution value.
:func:`gaussian`
    Compute normalized Gaussian broadening profile.
:func:`voigt`
    Compute a normalized Thompson-Cox-Hastings pseudo-Voigt profile.

Notes
-----
JAX can compile all functions. They support ``jax.vmap``
for vectorized evaluation across k-points and bands.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
from beartype import beartype
from jaxtyping import Array, Float, jaxtyped

from diffpes.maths import safe_divide, safe_power
from diffpes.types import KB_EV_PER_K, ScalarFloat


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
    """Compute a normalized Thompson-Cox-Hastings pseudo-Voigt profile.

    The function uses the pseudo-Voigt method from Thompson, Cox, and Hastings
    (1987) [1]_. This method expresses the Voigt profile as a linear
    combination of Gaussian and Lorentzian components:

        V(E) = eta * L(E) + (1 - eta) * G(E)

    :see: :class:`~.test_broadening.TestVoigt`

    where eta is an empirically determined mixing ratio. This
    approximation is accurate to better than 1% relative error.

    Implementation Logic
    --------------------
    The pseudo-Voigt approximation proceeds in four stages:

    1. **Compute component FWHMs**::

           f_G = 2 * sigma * sqrt(2 * ln2)   (Gaussian FWHM)
           f_L = 2 * gamma                    (Lorentzian FWHM)

       Converts the Gaussian standard deviation and Lorentzian
       half-width to their respective full-width at half-maximum
       values.

    2. **Compute Voigt FWHM via empirical formula**::

           f_V = (f_G^5 + 2.69269 * f_G^4 * f_L
                  + 2.42843 * f_G^3 * f_L^2
                  + 4.47163 * f_G^2 * f_L^3
                  + 0.07842 * f_G * f_L^4
                  + f_L^5)^(1/5)

       The Thompson-Cox-Hastings empirical relation approximates
       the FWHM of the true Voigt convolution from the component
       FWHMs. The function sanitizes the width polynomial before the fractional
       power. Thus, an inactive branch cannot produce invalid gradients.

    3. **Compute mixing ratio eta**::

           ratio = f_L / f_V
           eta = 1.36603 * ratio - 0.47719 * ratio^2
                 + 0.11116 * ratio^3

       The mixing ratio interpolates between pure Gaussian (eta = 0)
       and pure Lorentzian (eta = 1). For non-negative physical widths,
       the Thompson-Cox-Hastings polynomial keeps it in [0, 1].

    4. **Combine Gaussian and Lorentzian components**::

           sigma_V = f_V / (2 * sqrt(2 * ln2))
           gamma_V = f_V / 2
           G = gaussian(energy_range, center, sigma_V)
           L = gamma_V / (pi * (diff^2 + gamma_V^2))
           profile = eta * L + (1 - eta) * G

       Both components use the Voigt FWHM (not the original widths)
       so that the combined profile has the correct total width.

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
        Normalized pseudo-Voigt profile values.

    Raises
    ------
    EquinoxRuntimeError
        If either width is non-finite or negative.
        Also raised when ``sigma`` and ``gamma`` both equal zero because no
        normalized profile or directional derivative exists there.

    Notes
    -----
    The function validates widths before
    :func:`diffpes.maths.safe_power` supplies its boundary convention; that
    helper does not validate domains. Quotients use
    :func:`diffpes.maths.safe_divide`, so inactive zero-width branches cannot
    inject NaNs into reverse-mode gradients. The pure-Gaussian ray
    ``gamma = 0`` and pure-Lorentzian ray ``sigma = 0`` retain finite
    one-sided boundary sensitivities.

    References
    ----------
    .. [1] Thompson, Cox & Hastings, "Rietveld refinement of
       Debye-Scherrer synchrotron X-ray data from Al2O3",
       J. Appl. Cryst. 20, 79-83 (1987).
    """
    sigma_array: Float[Array, ""] = jnp.asarray(sigma, dtype=jnp.float64)
    gamma_array: Float[Array, ""] = jnp.asarray(gamma, dtype=jnp.float64)
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
    ln_two: Float[Array, ""] = jnp.log(jnp.float64(2.0))
    f_g: Float[Array, ""] = 2.0 * checked_sigma * jnp.sqrt(2.0 * ln_two)
    f_l: Float[Array, ""] = 2.0 * checked_gamma
    poly: Float[Array, ""] = (
        f_g**5
        + 2.69269 * f_g**4 * f_l
        + 2.42843 * f_g**3 * f_l**2
        + 4.47163 * f_g**2 * f_l**3
        + 0.07842 * f_g * f_l**4
        + f_l**5
    )
    f_v: Float[Array, ""] = safe_power(poly, 0.2)
    ratio: Float[Array, ""] = safe_divide(f_l, f_v)
    eta: Float[Array, ""] = (
        1.36603 * ratio - 0.47719 * ratio**2 + 0.11116 * ratio**3
    )
    sigma_v: Float[Array, ""] = safe_divide(f_v, 2.0 * jnp.sqrt(2.0 * ln_two))
    g_part: Float[Array, " E"] = gaussian(energy_range, center, sigma_v)
    diff: Float[Array, " E"] = energy_range - center
    gamma_v: Float[Array, ""] = safe_divide(f_v, jnp.float64(2.0))
    l_part: Float[Array, " E"] = safe_divide(
        gamma_v, jnp.pi * (diff**2 + gamma_v**2)
    )
    profile: Float[Array, " E"] = eta * l_part + (1.0 - eta) * g_part
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
