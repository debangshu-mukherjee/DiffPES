"""Simulation parameter data structures.

Extended Summary
----------------
Defines PyTree types for ARPES simulation parameters including
energy resolution, broadening widths, temperature, photon energy,
and light polarization configuration.

Routine Listings
----------------
:class:`SimulationParams`
    PyTree for core simulation parameters.
:class:`PolarizationConfig`
    PyTree for photon polarization geometry.
:func:`make_simulation_params`
    Factory for SimulationParams.
:func:`make_polarization_config`
    Factory for PolarizationConfig.

Notes
-----
Polarization types follow standard optics conventions:
``"LVP"`` (s-pol), ``"LHP"`` (p-pol), ``"RCP"``, ``"LCP"``,
``"LAP"`` (linear arbitrary), ``"unpolarized"``.
"""

import jax.numpy as jnp
from beartype import beartype
from beartype.typing import NamedTuple, Tuple
from jax.tree_util import register_pytree_node_class
from jaxtyping import Array, Float, jaxtyped

from .aliases import ScalarFloat, ScalarNumeric


@register_pytree_node_class
class SimulationParams(NamedTuple):
    """PyTree for ARPES simulation parameters.

    Attributes
    ----------
    energy_min : Float[Array, " "]
        Lower bound of energy window in eV.
    energy_max : Float[Array, " "]
        Upper bound of energy window in eV.
    fidelity : int
        Number of points along the energy axis.
    sigma : Float[Array, " "]
        Gaussian instrumental broadening width in eV.
    gamma : Float[Array, " "]
        Lorentzian lifetime broadening half-width in eV.
    temperature : Float[Array, " "]
        Sample temperature in Kelvin.
    photon_energy : Float[Array, " "]
        Incident photon energy in eV.
    """

    energy_min: Float[Array, " "]
    energy_max: Float[Array, " "]
    fidelity: int
    sigma: Float[Array, " "]
    gamma: Float[Array, " "]
    temperature: Float[Array, " "]
    photon_energy: Float[Array, " "]

    def tree_flatten(
        self,
    ) -> Tuple[
        Tuple[
            Float[Array, " "],
            Float[Array, " "],
            Float[Array, " "],
            Float[Array, " "],
            Float[Array, " "],
            Float[Array, " "],
        ],
        int,
    ]:
        """Flatten into JAX children and auxiliary data."""
        return (
            (
                self.energy_min,
                self.energy_max,
                self.sigma,
                self.gamma,
                self.temperature,
                self.photon_energy,
            ),
            self.fidelity,
        )

    @classmethod
    def tree_unflatten(
        cls,
        aux_data: int,
        children: Tuple[
            Float[Array, " "],
            Float[Array, " "],
            Float[Array, " "],
            Float[Array, " "],
            Float[Array, " "],
            Float[Array, " "],
        ],
    ) -> "SimulationParams":
        """Reconstruct from flattened components."""
        (
            energy_min,
            energy_max,
            sigma,
            gamma,
            temperature,
            photon_energy,
        ) = children
        return cls(
            energy_min=energy_min,
            energy_max=energy_max,
            fidelity=aux_data,
            sigma=sigma,
            gamma=gamma,
            temperature=temperature,
            photon_energy=photon_energy,
        )


@register_pytree_node_class
class PolarizationConfig(NamedTuple):
    """PyTree for photon polarization geometry.

    Attributes
    ----------
    theta : Float[Array, " "]
        Incident angle from surface normal in radians.
    phi : Float[Array, " "]
        In-plane azimuthal angle in radians.
    polarization_angle : Float[Array, " "]
        Arbitrary linear polarization angle in radians.
    polarization_type : str
        One of LVP, LHP, RCP, LCP, LAP, unpolarized.
    """

    theta: Float[Array, " "]
    phi: Float[Array, " "]
    polarization_angle: Float[Array, " "]
    polarization_type: str

    def tree_flatten(
        self,
    ) -> Tuple[
        Tuple[
            Float[Array, " "],
            Float[Array, " "],
            Float[Array, " "],
        ],
        str,
    ]:
        """Flatten into JAX children and auxiliary data."""
        return (
            (self.theta, self.phi, self.polarization_angle),
            self.polarization_type,
        )

    @classmethod
    def tree_unflatten(
        cls,
        aux_data: str,
        children: Tuple[
            Float[Array, " "],
            Float[Array, " "],
            Float[Array, " "],
        ],
    ) -> "PolarizationConfig":
        """Reconstruct from flattened components."""
        theta, phi, polarization_angle = children
        return cls(
            theta=theta,
            phi=phi,
            polarization_angle=polarization_angle,
            polarization_type=aux_data,
        )


@jaxtyped(typechecker=beartype)
def make_simulation_params(
    energy_min: ScalarNumeric = -3.0,
    energy_max: ScalarNumeric = 1.0,
    fidelity: int = 25000,
    sigma: ScalarFloat = 0.04,
    gamma: ScalarFloat = 0.1,
    temperature: ScalarFloat = 15.0,
    photon_energy: ScalarFloat = 11.0,
) -> SimulationParams:
    """Create a validated SimulationParams instance.

    Parameters
    ----------
    energy_min : ScalarNumeric, optional
        Lower energy bound in eV. Default is -3.0.
    energy_max : ScalarNumeric, optional
        Upper energy bound in eV. Default is 1.0.
    fidelity : int, optional
        Energy axis resolution. Default is 25000.
    sigma : ScalarFloat, optional
        Gaussian broadening in eV. Default is 0.04.
    gamma : ScalarFloat, optional
        Lorentzian broadening in eV. Default is 0.1.
    temperature : ScalarFloat, optional
        Temperature in Kelvin. Default is 15.0.
    photon_energy : ScalarFloat, optional
        Photon energy in eV. Default is 11.0.

    Returns
    -------
    params : SimulationParams
        Validated simulation parameters.
    """
    params: SimulationParams = SimulationParams(
        energy_min=jnp.asarray(energy_min, dtype=jnp.float64),
        energy_max=jnp.asarray(energy_max, dtype=jnp.float64),
        fidelity=fidelity,
        sigma=jnp.asarray(sigma, dtype=jnp.float64),
        gamma=jnp.asarray(gamma, dtype=jnp.float64),
        temperature=jnp.asarray(
            temperature, dtype=jnp.float64
        ),
        photon_energy=jnp.asarray(
            photon_energy, dtype=jnp.float64
        ),
    )
    return params


@jaxtyped(typechecker=beartype)
def make_polarization_config(
    theta: ScalarFloat = 0.7854,
    phi: ScalarFloat = 0.0,
    polarization_angle: ScalarFloat = 0.0,
    polarization_type: str = "unpolarized",
) -> PolarizationConfig:
    """Create a validated PolarizationConfig instance.

    Parameters
    ----------
    theta : ScalarFloat, optional
        Incident angle in radians. Default is pi/4.
    phi : ScalarFloat, optional
        Azimuthal angle in radians. Default is 0.
    polarization_angle : ScalarFloat, optional
        Linear polarization angle in radians. Default is 0.
    polarization_type : str, optional
        Polarization type. Default is ``"unpolarized"``.

    Returns
    -------
    config : PolarizationConfig
        Validated polarization configuration.
    """
    config: PolarizationConfig = PolarizationConfig(
        theta=jnp.asarray(theta, dtype=jnp.float64),
        phi=jnp.asarray(phi, dtype=jnp.float64),
        polarization_angle=jnp.asarray(
            polarization_angle, dtype=jnp.float64
        ),
        polarization_type=polarization_type,
    )
    return config


__all__: list[str] = [
    "PolarizationConfig",
    "SimulationParams",
    "make_polarization_config",
    "make_simulation_params",
]
