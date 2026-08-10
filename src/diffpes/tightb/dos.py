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
from beartype.typing import Tuple
from jaxtyping import Array, Float64, jaxtyped

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
    eigenvalues: Float64[Array, "n_k n_bands"],
    k_weights: Float64[Array, " n_k"],
    *,
    context: str,
) -> Tuple[Float64[Array, "n_k n_bands"], Float64[Array, " n_k"]]:
    """PRIVATE: Normalize arrays and enforce the weighted-spectrum contract.

    Parameters
    ----------
    eigenvalues : Float64[Array, "n_k n_bands"]
        Band energies in eV.
    k_weights : Float64[Array, " n_k"]
        Candidate k-point weights.
    context : str
        Caller name used as the error-message prefix.

    Returns
    -------
    result : Tuple[Float64[Array, "n_k n_bands"], Float64[Array, " n_k"]]
        Float64 energies and weights renormalized to an exact unit sum.

    Raises
    ------
    ValueError
        If ``eigenvalues`` is not two-dimensional, ``k_weights`` is not
        one-dimensional, the k-point counts disagree, or the spectrum
        holds no state.

    Notes
    -----
    Static checks fix ranks and shapes before tracing. Chained
    :func:`equinox.error_if` guards then raise ``EquinoxRuntimeError``
    for non-finite energies or weights, a negative weight, or a weight
    sum away from one beyond ``EPS``. Division by the checked sum makes
    the returned weights sum to one exactly.
    """
    energies: Float64[Array, "n_k n_bands"] = jnp.asarray(
        eigenvalues,
        dtype=jnp.float64,
    )
    weights: Float64[Array, " n_k"] = jnp.asarray(
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
    weight_sum: Float64[Array, ""] = jnp.sum(weights)
    weights = eqx.error_if(
        weights,
        jnp.abs(weight_sum - 1.0) > EPS,
        f"{context}: k_weights must sum to one",
    )
    weights = weights / weight_sum
    result: Tuple[
        Float64[Array, "n_k n_bands"],
        Float64[Array, " n_k"],
    ] = (energies, weights)
    return result


@jaxtyped(typechecker=beartype)
def dos_gaussian(  # noqa: DOC502, DOC503 -- traced validation raises under JAX.
    eigenvalues: Float64[Array, "n_k n_bands"],
    k_weights: Float64[Array, " n_k"],
    energy_axis: Float64[Array, " n_e"],
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
    eigenvalues : Float64[Array, "n_k n_bands"]
        Band energies in eV.
    k_weights : Float64[Array, " n_k"]
        Nonnegative k-point weights normalized to one.
    energy_axis : Float64[Array, " n_e"]
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
    energies: Float64[Array, "n_k n_bands"]
    weights: Float64[Array, " n_k"]
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
    differences: Float64[Array, "n_e n_k n_bands"] = (
        axis[:, None, None] - energies[None, :, :]
    )
    normalization: Float64[Array, ""] = jnp.sqrt(2.0 * jnp.pi) * width
    profiles: Float64[Array, "n_e n_k n_bands"] = (
        jnp.exp(-0.5 * (differences / width) ** 2) / normalization
    )
    values: Float64[Array, " n_e"] = jnp.sum(
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
    chemical_potential: Float64[Array, ""],
    arguments: Tuple[
        Float64[Array, "n_k n_bands"],
        Float64[Array, " n_k"],
        Float64[Array, ""],
        Float64[Array, ""],
    ],
) -> Float64[Array, ""]:
    """PRIVATE: Evaluate the weighted occupation minus the requested filling.

    Parameters
    ----------
    chemical_potential : Float64[Array, ""]
        Trial chemical potential in eV.
    arguments : Tuple[Float64[Array, "n_k n_bands"], Float64[Array, \
" n_k"], Float64[Array, ""], Float64[Array, ""]]
        Band energies in eV, unit-sum k-point weights, electronic
        temperature in kelvin, and target electron count per cell.

    Returns
    -------
    residual : Float64[Array, ""]
        Weighted Fermi--Dirac occupation sum minus the electron count.

    Notes
    -----
    :func:`jax.vmap` evaluates :func:`diffpes.simul.fermi_dirac` over the
    flattened spectrum. A k-weighted sum over all states gives the
    occupation. The root of this residual in ``chemical_potential`` is
    the finite-temperature Fermi level.
    """
    eigenvalues: Float64[Array, "n_k n_bands"]
    k_weights: Float64[Array, " n_k"]
    temperature_k: Float64[Array, ""]
    n_electrons: Float64[Array, ""]
    eigenvalues, k_weights, temperature_k, n_electrons = arguments
    flat_occupations: Float64[Array, " n_states"] = jax.vmap(
        lambda energy: fermi_dirac(
            energy,
            chemical_potential,
            temperature_k,
        )
    )(jnp.ravel(eigenvalues))
    occupations: Float64[Array, "n_k n_bands"] = jnp.reshape(
        flat_occupations,
        eigenvalues.shape,
    )
    count: Float64[Array, ""] = jnp.sum(k_weights[:, None] * occupations)
    residual: Float64[Array, ""] = count - n_electrons
    return residual


def _solve_filling(
    arguments: Tuple[
        Float64[Array, "n_k n_bands"],
        Float64[Array, " n_k"],
        Float64[Array, ""],
        Float64[Array, ""],
    ],
    lower: Float64[Array, ""],
    upper: Float64[Array, ""],
    *,
    tolerance: float,
) -> Float64[Array, ""]:
    """PRIVATE: Compute one validated filling root at a static tolerance.

    Parameters
    ----------
    arguments : Tuple[Float64[Array, "n_k n_bands"], Float64[Array, \
" n_k"], Float64[Array, ""], Float64[Array, ""]]
        Band energies in eV, unit-sum k-point weights, electronic
        temperature in kelvin, and target electron count per cell.
    lower : Float64[Array, ""]
        Lower bracket edge for the Fermi level in eV.
    upper : Float64[Array, ""]
        Upper bracket edge for the Fermi level in eV.
    tolerance : float
        Static absolute and relative bisection tolerance in eV.

    Returns
    -------
    chemical_potential : Float64[Array, ""]
        Converged root of ``_filling_residual`` in eV.

    Notes
    -----
    A non-expanding Optimistix ``Bisection`` solve starts from the
    bracket midpoint and keeps every iterate inside ``[lower, upper]``.
    ``throw=True`` raises through Equinox when 2048 steps do not reach
    the tolerance. Optimistix differentiates the converged filling
    equation implicitly instead of unrolling bisection iterations.
    """
    midpoint: Float64[Array, ""] = lower + 0.5 * (upper - lower)
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
    chemical_potential: Float64[Array, ""] = solution.value
    return chemical_potential


@jaxtyped(typechecker=beartype)
def fermi_level_from_filling(  # noqa: DOC502
    eigenvalues: Float64[Array, "n_k n_bands"],
    k_weights: Float64[Array, " n_k"],
    n_electrons: ScalarFloat,
    temperature_k: ScalarFloat = 300.0,
) -> Float64[Array, ""]:
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
    eigenvalues : Float64[Array, "n_k n_bands"]
        Band energies in eV.
    k_weights : Float64[Array, " n_k"]
        Nonnegative k-point weights normalized to one.
    n_electrons : ScalarFloat
        Electron count per cell, strictly between zero and the band capacity.
    temperature_k : ScalarFloat, optional
        Positive electronic temperature in kelvin. Default is 300 K.

    Returns
    -------
    chemical_potential : Float64[Array, ""]
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
    energies: Float64[Array, "n_k n_bands"]
    weights: Float64[Array, " n_k"]
    energies, weights = _validate_spectrum_inputs(
        eigenvalues,
        k_weights,
        context="fermi_level_from_filling",
    )
    filling: Float64[Array, ""] = jnp.asarray(
        n_electrons,
        dtype=jnp.float64,
    )
    temperature: Float64[Array, ""] = jnp.asarray(
        temperature_k,
        dtype=jnp.float64,
    )
    capacity: Float64[Array, ""] = jnp.asarray(
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
    thermal_energy: Float64[Array, ""] = KB_EV_PER_K * temperature
    thermal_energy = eqx.error_if(
        thermal_energy,
        ~jnp.isfinite(thermal_energy),
        "fermi_level_from_filling: thermal energy must be finite",
    )
    filling_fraction: Float64[Array, ""] = filling / capacity
    lower_logit: Float64[Array, ""] = jnp.log(filling_fraction) - jnp.log(
        2.0 - filling_fraction
    )
    upper_logit: Float64[Array, ""] = jnp.log1p(filling_fraction) - jnp.log1p(
        -filling_fraction
    )
    lower: Float64[Array, ""] = (
        jnp.min(energies) + thermal_energy * lower_logit
    )
    upper: Float64[Array, ""] = (
        jnp.max(energies) + thermal_energy * upper_logit
    )
    arguments: Tuple[
        Float64[Array, "n_k n_bands"],
        Float64[Array, " n_k"],
        Float64[Array, ""],
        Float64[Array, ""],
    ] = (energies, weights, temperature, filling)
    lower_residual: Float64[Array, ""] = _filling_residual(lower, arguments)
    upper_residual: Float64[Array, ""] = _filling_residual(upper, arguments)
    lower = eqx.error_if(
        lower,
        (lower_residual > 0.0) | (upper_residual < 0.0),
        "fermi_level_from_filling: analytic root bracket is invalid",
    )
    solved: Float64[Array, ""] = _solve_filling(
        arguments,
        lower,
        upper,
        tolerance=1e-12,
    )
    chemical_potential: Float64[Array, ""] = eqx.error_if(
        solved,
        ~jnp.isfinite(solved),
        "fermi_level_from_filling: root must be finite",
    )
    return chemical_potential


__all__: list[str] = [
    "dos_gaussian",
    "fermi_level_from_filling",
]
