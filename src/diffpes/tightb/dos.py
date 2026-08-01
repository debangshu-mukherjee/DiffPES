"""Construct broadened tight-binding densities of states and fillings.

Extended Summary
----------------
This module evaluates a normalized Gaussian k-sum density of states and solves
the finite-temperature filling equation for the chemical potential. Both
operations are pure JAX programs. The filling solve uses Optimistix with an
initial bracketing solve followed by an implicitly differentiated Newton solve.

Routine Listings
----------------
:func:`dos_gaussian`
    Evaluate a Gaussian-broadened tight-binding density of states.
:func:`fermi_level_from_filling`
    Compute the finite-temperature Fermi level from the filling equation.

Notes
-----
Require nonnegative k-point weights normalized to one. Integrating the
returned DOS over the full real line then gives the number of bands. A finite
sampled energy window carries the corresponding Gaussian tail truncation
error.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
import optimistix as optx
from beartype import beartype
from jaxtyping import Array, Float, Float64, jaxtyped

from diffpes.simul import fermi_dirac
from diffpes.types import (
    EPS,
    KB_EV_PER_K,
    MINIMUM_AXIS_POINTS,
    SPECTRUM_NDIM,
    DensityOfStates,
    ScalarFloat,
    make_density_of_states,
)


def _validate_spectrum_inputs(
    eigenvalues: Float[Array, "n_k n_bands"],
    k_weights: Float[Array, " n_k"],
    *,
    context: str,
) -> tuple[Float[Array, "n_k n_bands"], Float[Array, " n_k"]]:
    """Normalize arrays and enforce the shared weighted-spectrum contract."""
    energies: Float[Array, "n_k n_bands"] = jnp.asarray(
        eigenvalues,
        dtype=jnp.float64,
    )
    weights: Float[Array, " n_k"] = jnp.asarray(
        k_weights,
        dtype=jnp.float64,
    )
    if energies.ndim != SPECTRUM_NDIM:
        raise ValueError(f"{context}: eigenvalues must be two-dimensional")
    if weights.ndim != 1:
        raise ValueError(f"{context}: k_weights must be one-dimensional")
    if energies.shape[0] != weights.shape[0]:
        raise ValueError(
            f"{context}: eigenvalues and k_weights k-point counts disagree"
        )
    if energies.shape[0] == 0 or energies.shape[1] == 0:
        raise ValueError(
            f"{context}: spectrum must contain at least one state"
        )
    energies = eqx.error_if(
        energies,
        ~jnp.all(jnp.isfinite(energies)),
        f"{context}: eigenvalues must be finite",
    )
    weights = eqx.error_if(
        weights,
        ~jnp.all(jnp.isfinite(weights)),
        f"{context}: k_weights must be finite",
    )
    weights = eqx.error_if(
        weights,
        jnp.any(weights < 0.0),
        f"{context}: k_weights must be nonnegative",
    )
    weight_sum: Float[Array, ""] = jnp.sum(weights)
    weights = eqx.error_if(
        weights,
        jnp.abs(weight_sum - 1.0) > EPS,
        f"{context}: k_weights must sum to one",
    )
    weights = weights / weight_sum
    result: tuple[
        Float[Array, "n_k n_bands"],
        Float[Array, " n_k"],
    ] = (energies, weights)
    return result


@jaxtyped(typechecker=beartype)
def dos_gaussian(  # noqa: DOC502, DOC503 -- traced validation raises under JAX.
    eigenvalues: Float[Array, "n_k n_bands"],
    k_weights: Float[Array, " n_k"],
    energy_axis: Float[Array, " n_e"],
    sigma: ScalarFloat,
) -> DensityOfStates:
    r"""Evaluate a Gaussian-broadened tight-binding density of states.

    The returned values implement

    .. math::

       N(E)=\sum_{kn}w_k\frac{\exp[-(E-\epsilon_{kn})^2/(2\sigma^2)]}
       {\sqrt{2\pi}\sigma}.

    :see: :class:`~.test_dos.TestDosGaussian`

    Parameters
    ----------
    eigenvalues : Float[Array, "n_k n_bands"]
        Band energies in eV.
    k_weights : Float[Array, " n_k"]
        Nonnegative k-point weights normalized to one.
    energy_axis : Float[Array, " n_e"]
        Strictly increasing output energy axis in eV.
    sigma : ScalarFloat
        Positive Gaussian standard deviation in eV.

    Returns
    -------
    dos : DensityOfStates
        Validated DOS carrier. Its reference energy is zero.

    Raises
    ------
    ValueError
        If array ranks or static dimensions are invalid.
    EquinoxRuntimeError
        If numerical inputs are nonfinite, weights are invalid, or ``sigma``
        is not positive.

    Notes
    -----
    Broadcasting constructs every sampled Gaussian profile. Weighted
    reductions over k-points and bands produce the carrier values.
    """
    energies: Float[Array, "n_k n_bands"]
    weights: Float[Array, " n_k"]
    energies, weights = _validate_spectrum_inputs(
        eigenvalues,
        k_weights,
        context="dos_gaussian",
    )
    axis: Float64[Array, " n_e"] = jnp.asarray(
        energy_axis,
        dtype=jnp.float64,
    )
    width: Float64[Array, ""] = jnp.asarray(sigma, dtype=jnp.float64)
    if axis.ndim != 1:
        raise ValueError("dos_gaussian: energy_axis must be one-dimensional")
    if axis.shape[0] < MINIMUM_AXIS_POINTS:
        raise ValueError(
            "dos_gaussian: energy_axis must contain at least two points"
        )
    width = eqx.error_if(
        width,
        ~jnp.isfinite(width) | (width <= 0.0),
        "dos_gaussian: sigma must be finite and positive",
    )
    differences: Float[Array, "n_e n_k n_bands"] = (
        axis[:, None, None] - energies[None, :, :]
    )
    normalization: Float[Array, ""] = jnp.sqrt(2.0 * jnp.pi) * width
    profiles: Float[Array, "n_e n_k n_bands"] = (
        jnp.exp(-0.5 * (differences / width) ** 2) / normalization
    )
    values: Float[Array, " n_e"] = jnp.sum(
        profiles * weights[None, :, None],
        axis=(1, 2),
    )
    dos: DensityOfStates = make_density_of_states(
        energy=axis,
        total_dos=values,
        fermi_energy=0.0,
    )
    return dos


def _filling_residual(
    chemical_potential: Float[Array, ""],
    arguments: tuple[
        Float[Array, "n_k n_bands"],
        Float[Array, " n_k"],
        Float[Array, ""],
        Float[Array, ""],
    ],
) -> Float[Array, ""]:
    """Evaluate the weighted occupation minus the requested filling."""
    eigenvalues: Float[Array, "n_k n_bands"]
    k_weights: Float[Array, " n_k"]
    temperature_k: Float[Array, ""]
    n_electrons: Float[Array, ""]
    eigenvalues, k_weights, temperature_k, n_electrons = arguments
    flat_occupations: Float[Array, " n_states"] = jax.vmap(
        lambda energy: fermi_dirac(
            energy,
            chemical_potential,
            temperature_k,
        )
    )(jnp.ravel(eigenvalues))
    occupations: Float[Array, "n_k n_bands"] = jnp.reshape(
        flat_occupations,
        eigenvalues.shape,
    )
    count: Float[Array, ""] = jnp.sum(k_weights[:, None] * occupations)
    residual: Float[Array, ""] = count - n_electrons
    return residual


def _solve_filling(
    arguments: tuple[
        Float[Array, "n_k n_bands"],
        Float[Array, " n_k"],
        Float[Array, ""],
        Float[Array, ""],
    ],
    lower: Float[Array, ""],
    upper: Float[Array, ""],
    *,
    tolerance: float,
) -> Float[Array, ""]:
    """Compute one validated filling root at a static tolerance."""
    midpoint: Float[Array, ""] = lower + 0.5 * (upper - lower)
    solver: optx.Bisection = optx.Bisection(
        rtol=tolerance,
        atol=tolerance,
        flip=False,
        expand_if_necessary=False,
    )
    solution: optx.Solution = optx.root_find(
        _filling_residual,
        solver,
        midpoint,
        args=arguments,
        options={"lower": lower, "upper": upper},
        max_steps=2048,
        throw=True,
    )
    chemical_potential: Float[Array, ""] = solution.value
    return chemical_potential


@jaxtyped(typechecker=beartype)
def fermi_level_from_filling(  # noqa: DOC502
    eigenvalues: Float[Array, "n_k n_bands"],
    k_weights: Float[Array, " n_k"],
    n_electrons: ScalarFloat,
    temperature_k: ScalarFloat = 300.0,
) -> Float[Array, ""]:
    r"""Compute the finite-temperature Fermi level from the filling equation.

    The root satisfies

    .. math::

       \sum_{kn}w_k f_\mathrm{FD}(\epsilon_{kn};\mu,T)=N_e.

    A bracket-preserving bisection solve supplies the returned root.
    Optimistix applies the implicit-function derivative to the converged
    filling equation rather than differentiating the bisection iterations.

    :see: :class:`~.test_dos.TestFermiLevelFromFilling`

    Parameters
    ----------
    eigenvalues : Float[Array, "n_k n_bands"]
        Band energies in eV.
    k_weights : Float[Array, " n_k"]
        Nonnegative k-point weights normalized to one.
    n_electrons : ScalarFloat
        Electron count per cell, strictly between zero and the band capacity.
    temperature_k : ScalarFloat, optional
        Positive electronic temperature in kelvin. Default is 300 K.

    Returns
    -------
    chemical_potential : Float[Array, ""]
        Fermi level in eV.

    Raises
    ------
    ValueError
        If array ranks or static dimensions are invalid.
    EquinoxRuntimeError
        If numerical inputs, temperature, or filling are outside the declared
        domain, or if the solve does not produce a finite root.

    Notes
    -----
    The discrete band capacity is ``n_bands`` because k-point weights sum to
    one. The domain excludes end-point fillings because their chemical
    potentials occur only in infinite limits at positive temperature.
    """
    energies: Float[Array, "n_k n_bands"]
    weights: Float[Array, " n_k"]
    energies, weights = _validate_spectrum_inputs(
        eigenvalues,
        k_weights,
        context="fermi_level_from_filling",
    )
    filling: Float[Array, ""] = jnp.asarray(
        n_electrons,
        dtype=jnp.float64,
    )
    temperature: Float[Array, ""] = jnp.asarray(
        temperature_k,
        dtype=jnp.float64,
    )
    capacity: Float[Array, ""] = jnp.asarray(
        energies.shape[1],
        dtype=jnp.float64,
    )
    temperature = eqx.error_if(
        temperature,
        ~jnp.isfinite(temperature) | (temperature <= 0.0),
        "fermi_level_from_filling: temperature_k must be finite and positive",
    )
    filling = eqx.error_if(
        filling,
        ~jnp.isfinite(filling) | (filling <= 0.0) | (filling >= capacity),
        "fermi_level_from_filling: n_electrons must lie inside band capacity",
    )
    thermal_energy: Float[Array, ""] = KB_EV_PER_K * temperature
    thermal_energy = eqx.error_if(
        thermal_energy,
        ~jnp.isfinite(thermal_energy),
        "fermi_level_from_filling: thermal energy must be finite",
    )
    filling_fraction: Float[Array, ""] = filling / capacity
    lower_logit: Float[Array, ""] = jnp.log(filling_fraction) - jnp.log(
        2.0 - filling_fraction
    )
    upper_logit: Float[Array, ""] = jnp.log1p(filling_fraction) - jnp.log1p(
        -filling_fraction
    )
    lower: Float[Array, ""] = jnp.min(energies) + thermal_energy * lower_logit
    upper: Float[Array, ""] = jnp.max(energies) + thermal_energy * upper_logit
    arguments: tuple[
        Float[Array, "n_k n_bands"],
        Float[Array, " n_k"],
        Float[Array, ""],
        Float[Array, ""],
    ] = (energies, weights, temperature, filling)
    lower_residual: Float[Array, ""] = _filling_residual(lower, arguments)
    upper_residual: Float[Array, ""] = _filling_residual(upper, arguments)
    lower = eqx.error_if(
        lower,
        (lower_residual > 0.0) | (upper_residual < 0.0),
        "fermi_level_from_filling: analytic root bracket is invalid",
    )
    solved: Float[Array, ""] = _solve_filling(
        arguments,
        lower,
        upper,
        tolerance=1e-12,
    )
    chemical_potential: Float[Array, ""] = eqx.error_if(
        solved,
        ~jnp.isfinite(solved),
        "fermi_level_from_filling: root must be finite",
    )
    return chemical_potential


__all__: list[str] = [
    "dos_gaussian",
    "fermi_level_from_filling",
]
