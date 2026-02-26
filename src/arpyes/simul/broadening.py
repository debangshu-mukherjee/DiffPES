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

_KB: float = 8.617333e-5


@jaxtyped(typechecker=beartype)
def gaussian(
    energy_range: Float[Array, " E"],
    center: ScalarFloat,
    sigma: ScalarFloat,
) -> Float[Array, " E"]:
    """Compute normalized Gaussian broadening profile.

    Evaluates a Gaussian lineshape centered at ``center`` with standard
    deviation ``sigma``, normalized so that the integral over all energies
    equals unity.

    Implementation Logic
    --------------------
    The function evaluates the analytic Gaussian probability density:

        G(E) = exp(-(E - E0)^2 / (2 * sigma^2)) / (sqrt(2 * pi) * sigma)

    1. **Compute energy differences**:
       diff = energy_range - center
       - Shifts the energy axis so the peak is at the origin.

    2. **Compute normalization factor**:
       norm_factor = sqrt(2 * pi) * sigma
       - This prefactor ensures the profile integrates to unity over
         (-inf, +inf), i.e. the Gaussian is normalized to unit area.

    3. **Evaluate Gaussian profile**:
       profile = exp(-diff^2 / (2 * sigma^2)) / norm_factor
       - Element-wise evaluation of the normalized Gaussian at each
         energy point.

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
    """Compute normalized Voigt profile via pseudo-Voigt approximation.

    Evaluates a Voigt lineshape using the pseudo-Voigt method of
    Thompson, Cox & Hastings (1987), which expresses the Voigt profile
    as a linear combination of Gaussian and Lorentzian components:

        V(E) = eta * L(E) + (1 - eta) * G(E)

    where eta is an empirically determined mixing ratio. This
    approximation is accurate to better than 1% relative error.

    Implementation Logic
    --------------------
    The pseudo-Voigt approximation proceeds in four stages:

    1. **Compute component FWHMs**:
       f_G = 2 * sigma * sqrt(2 * ln2)   (Gaussian FWHM)
       f_L = 2 * gamma                    (Lorentzian FWHM)
       - Converts the Gaussian standard deviation and Lorentzian
         half-width to their respective full-width at half-maximum
         values.

    2. **Compute Voigt FWHM via empirical formula**:
       f_V = (f_G^5 + 2.69269 * f_G^4 * f_L + 2.42843 * f_G^3 * f_L^2
              + 4.47163 * f_G^2 * f_L^3 + 0.07842 * f_G * f_L^4
              + f_L^5)^(1/5)
       - The Thompson-Cox-Hastings empirical relation approximates
         the FWHM of the true Voigt convolution from the component
         FWHMs. A safety guard replaces f_V = 0 with 1e-30 to avoid
         division by zero.

    3. **Compute mixing ratio eta**:
       ratio = f_L / f_V
       eta = 1.36603 * ratio - 0.47719 * ratio^2 + 0.11116 * ratio^3
       - The mixing ratio interpolates between pure Gaussian (eta = 0)
         and pure Lorentzian (eta = 1). It is clipped to [0, 1].

    4. **Combine Gaussian and Lorentzian components**:
       sigma_V = f_V / (2 * sqrt(2 * ln2))
       gamma_V = f_V / 2
       G = gaussian(energy_range, center, sigma_V)
       L = gamma_V / (pi * (diff^2 + gamma_V^2))
       profile = eta * L + (1 - eta) * G
       - Both components use the Voigt FWHM (not the original widths)
         so that the combined profile has the correct total width.

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

    References
    ----------
    .. [1] Thompson, Cox & Hastings, "Rietveld refinement of
       Debye-Scherrer synchrotron X-ray data from Al2O3",
       J. Appl. Cryst. 20, 79-83 (1987).
    """
    _ln2: Float[Array, " "] = jnp.log(jnp.float64(2.0))
    f_g: Float[Array, " "] = (
        2.0 * sigma * jnp.sqrt(2.0 * _ln2)
    )
    f_l: Float[Array, " "] = 2.0 * gamma
    f_v: Float[Array, " "] = (
        f_g**5
        + 2.69269 * f_g**4 * f_l
        + 2.42843 * f_g**3 * f_l**2
        + 4.47163 * f_g**2 * f_l**3
        + 0.07842 * f_g * f_l**4
        + f_l**5
    ) ** 0.2
    safe_fv: Float[Array, " "] = jnp.where(
        f_v > 0.0, f_v, jnp.float64(1e-30)
    )
    ratio: Float[Array, " "] = f_l / safe_fv
    eta: Float[Array, " "] = (
        1.36603 * ratio
        - 0.47719 * ratio**2
        + 0.11116 * ratio**3
    )
    eta = jnp.clip(eta, 0.0, 1.0)
    sigma_v: Float[Array, " "] = (
        safe_fv / (2.0 * jnp.sqrt(2.0 * _ln2))
    )
    g_part: Float[Array, " E"] = gaussian(
        energy_range, center, sigma_v
    )
    diff: Float[Array, " E"] = energy_range - center
    gamma_v: Float[Array, " "] = safe_fv / 2.0
    l_part: Float[Array, " E"] = (
        gamma_v
        / (jnp.pi * (diff**2 + gamma_v**2))
    )
    profile: Float[Array, " E"] = (
        eta * l_part + (1.0 - eta) * g_part
    )
    return profile


@jaxtyped(typechecker=beartype)
def fermi_dirac(
    energy: ScalarFloat,
    fermi_energy: ScalarFloat,
    temperature: ScalarFloat,
) -> Float[Array, " "]:
    """Compute Fermi-Dirac distribution value.

    Evaluates the Fermi-Dirac thermal occupation function at a given
    energy, Fermi level, and temperature:

        f(E) = 1 / (1 + exp((E - Ef) / (kB * T)))

    Implementation Logic
    --------------------
    1. **Compute thermal energy kT**:
       kt = kB * T
       - Multiplies the Boltzmann constant kB = 8.617333e-5 eV/K by the
         temperature in Kelvin to obtain the thermal energy scale. Both
         values are cast to float64 for numerical precision.

    2. **Guard against T = 0**:
       safe_kt = max(kt, 1e-10)
       - At zero temperature the distribution becomes a step function,
         but the exponential would diverge. Clamping kT to a small
         positive value (1e-10 eV) avoids division by zero while
         preserving the sharp step-function behavior numerically.

    3. **Evaluate Fermi-Dirac function**:
       exponent = (E - Ef) / safe_kt
       occupation = 1 / (1 + exp(exponent))
       - Computes the occupation probability. For E << Ef the result
         approaches 1 (filled states); for E >> Ef it approaches 0
         (empty states).

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
    Uses the Boltzmann constant kB = 8.617333e-5 eV/K, stored in
    the module-level constant ``_KB``.
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
