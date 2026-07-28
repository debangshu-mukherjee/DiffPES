"""Define simulation-parameter data structures.

Extended Summary
----------------
This module defines the PyTree used by the retained incoherent spectrum tiers.
Experiment properties such as photon energy, sample temperature, incidence
geometry, and polarization belong to
``ExperimentGeometry``.

Routine Listings
----------------
:class:`SimulationParams`
    Store ARPES simulation parameters in a JAX PyTree.
:func:`make_expanded_simulation_params`
    Build simulation parameters with auto-derived energy window.
:func:`make_simulation_params`
    Create a validated SimulationParams instance.
"""

import equinox as eqx
import jax.numpy as jnp
from beartype import beartype
from jaxtyping import Array, Float, jaxtyped

from .aliases import ScalarFloat, ScalarNumeric

_MIN_FIDELITY: int = 2


class SimulationParams(eqx.Module):
    """Store ARPES simulation parameters in a JAX PyTree.

    The float-valued fields are differentiable array children. ``fidelity``
    is static because it determines the output array shape.

    :see: :class:`~.test_params.TestSimulationParams`

    Attributes
    ----------
    energy_min : Float[Array, " "]
        Lower bound of the energy window in eV.
    energy_max : Float[Array, " "]
        Upper bound of the energy window in eV.
    sigma : Float[Array, " "]
        Gaussian instrumental broadening width in eV.
    gamma : Float[Array, " "]
        Lorentzian lifetime broadening half-width in eV.
    fidelity : int
        Number of energy samples. This field is static and changing it
        retraces compiled functions.

    See Also
    --------
    ExperimentGeometry
        Carrier for photon, temperature, and polarization properties.
    make_simulation_params
        Validated factory for this carrier.
    """

    energy_min: Float[Array, " "]
    energy_max: Float[Array, " "]
    sigma: Float[Array, " "]
    gamma: Float[Array, " "]
    fidelity: int = eqx.field(static=True)


@jaxtyped(typechecker=beartype)
def make_simulation_params(  # noqa: DOC503
    energy_min: ScalarNumeric = -3.0,
    energy_max: ScalarNumeric = 1.0,
    fidelity: int = 25000,
    sigma: ScalarFloat = 0.04,
    gamma: ScalarFloat = 0.1,
) -> SimulationParams:
    """Create a validated SimulationParams instance.

    The factory casts numerical inputs to scalar float64 arrays. It validates
    the static energy-grid size before traced numerical checks.

    :see: :class:`~.test_params.TestMakeSimulationParams`

    Parameters
    ----------
    energy_min : ScalarNumeric, optional
        Lower energy bound in eV. Default is -3.0.
    energy_max : ScalarNumeric, optional
        Upper energy bound in eV. Default is 1.0.
    fidelity : int, optional
        Number of energy samples. This field is static. Default is 25000.
    sigma : ScalarFloat, optional
        Positive Gaussian broadening in eV. Default is 0.04.
    gamma : ScalarFloat, optional
        Positive Lorentzian broadening in eV. Default is 0.1.

    Returns
    -------
    params : SimulationParams
        Validated simulation parameters.

    Raises
    ------
    ValueError
        If ``fidelity`` is less than two.
    EquinoxRuntimeError
        If an input has a non-finite value, the energy window has the wrong
        order, or a broadening width is not positive.

    Notes
    -----
    Static shape validation runs before tracing. Numerical validation uses
    :func:`equinox.error_if` and therefore remains active under JIT.
    """
    if fidelity < _MIN_FIDELITY:
        message: str = "make_simulation_params: fidelity must be at least 2"
        raise ValueError(message)

    minimum: Float[Array, " "] = jnp.asarray(
        energy_min,
        dtype=jnp.float64,
    )
    maximum: Float[Array, " "] = jnp.asarray(
        energy_max,
        dtype=jnp.float64,
    )
    gaussian_width: Float[Array, " "] = jnp.asarray(
        sigma,
        dtype=jnp.float64,
    )
    lorentzian_width: Float[Array, " "] = jnp.asarray(
        gamma,
        dtype=jnp.float64,
    )
    minimum = eqx.error_if(
        minimum,
        ~jnp.isfinite(minimum),
        "make_simulation_params: energy_min must be finite",
    )
    maximum = eqx.error_if(
        maximum,
        ~jnp.isfinite(maximum),
        "make_simulation_params: energy_max must be finite",
    )
    minimum = eqx.error_if(
        minimum,
        ~(minimum < maximum),
        "make_simulation_params: energy_min must be less than energy_max",
    )
    gaussian_width = eqx.error_if(
        gaussian_width,
        ~jnp.isfinite(gaussian_width),
        "make_simulation_params: sigma must be finite",
    )
    gaussian_width = eqx.error_if(
        gaussian_width,
        ~(gaussian_width > 0.0),
        "make_simulation_params: sigma must be positive",
    )
    lorentzian_width = eqx.error_if(
        lorentzian_width,
        ~jnp.isfinite(lorentzian_width),
        "make_simulation_params: gamma must be finite",
    )
    lorentzian_width = eqx.error_if(
        lorentzian_width,
        ~(lorentzian_width > 0.0),
        "make_simulation_params: gamma must be positive",
    )
    params: SimulationParams = SimulationParams(
        energy_min=minimum,
        energy_max=maximum,
        fidelity=fidelity,
        sigma=gaussian_width,
        gamma=lorentzian_width,
    )
    return params


@jaxtyped(typechecker=beartype)
def make_expanded_simulation_params(  # noqa: DOC503
    eigenbands: Float[Array, "K B"],
    fidelity: int = 25000,
    sigma: ScalarFloat = 0.04,
    gamma: ScalarFloat = 0.1,
    energy_padding: ScalarFloat = 1.0,
) -> SimulationParams:
    """Build simulation parameters with auto-derived energy window.

    The factory derives the finite energy window from band extrema and a
    symmetric padding value. It delegates broadening validation to the base
    factory.

    :see: :class:`~.test_params.TestMakeExpandedSimulationParams`

    Parameters
    ----------
    eigenbands : Float[Array, "K B"]
        Band eigenvalues in eV. Their extrema define the energy window.
    fidelity : int, optional
        Number of energy samples. This field is static. Default is 25000.
    sigma : ScalarFloat, optional
        Positive Gaussian broadening in eV. Default is 0.04.
    gamma : ScalarFloat, optional
        Positive Lorentzian broadening in eV. Default is 0.1.
    energy_padding : ScalarFloat, optional
        Nonnegative symmetric padding around the extrema in eV. Default is 1.

    Returns
    -------
    params : SimulationParams
        Parameters spanning the padded band-energy range.

    Raises
    ------
    ValueError
        If ``eigenbands`` is empty.
    EquinoxRuntimeError
        If the bands or padding are non-finite, or the padding is negative.

    Notes
    -----
    Experiment properties are intentionally not accepted here. Supply
    temperature and photon energy at the consuming physics boundary or use
    :class:`~diffpes.types.ExperimentGeometry`.
    """
    bands: Float[Array, "K B"] = jnp.asarray(
        eigenbands,
        dtype=jnp.float64,
    )
    padding: Float[Array, " "] = jnp.asarray(
        energy_padding,
        dtype=jnp.float64,
    )
    if bands.size == 0:
        message: str = "eigenbands must contain at least one value"
        raise ValueError(message)
    bands = eqx.error_if(
        bands,
        ~jnp.all(jnp.isfinite(bands)),
        "make_expanded_simulation_params: eigenbands finite",
    )
    padding = eqx.error_if(
        padding,
        ~jnp.isfinite(padding) | (padding < 0.0),
        "make_expanded_simulation_params: padding finite and nonnegative",
    )
    params: SimulationParams = make_simulation_params(
        energy_min=jnp.min(bands) - padding,
        energy_max=jnp.max(bands) + padding,
        fidelity=fidelity,
        sigma=sigma,
        gamma=gamma,
    )
    return params


__all__: list[str] = [
    "SimulationParams",
    "make_expanded_simulation_params",
    "make_simulation_params",
]
