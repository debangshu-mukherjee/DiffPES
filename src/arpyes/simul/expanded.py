"""Expanded-input workflows for ARPES simulation.

Extended Summary
----------------
Provides convenience wrappers that accept plain arrays and scalars
so that the user does not need to know about the PyTree structures
used internally by arpyes.

Routine Listings
----------------
:func:`_build_inputs`
    Build internal PyTrees from plain arrays.
    Internal helper function.
:func:`simulate_novice_expanded`
    Novice-level simulation with expanded inputs.
:func:`simulate_basic_expanded`
    Basic-level simulation with expanded inputs.
:func:`simulate_basicplus_expanded`
    Basicplus-level simulation with expanded inputs.
:func:`simulate_advanced_expanded`
    Advanced-level simulation with expanded inputs.
:func:`simulate_expert_expanded`
    Expert-level simulation with expanded inputs.
:func:`simulate_expanded`
    Dispatch an expanded simulation by complexity level.

Notes
-----
Energy axes are built as
``linspace(min(eigenbands)-1, max(eigenbands)+1, fidelity)``.
"""

import jax.numpy as jnp
from beartype import beartype
from beartype.typing import Tuple
from jaxtyping import Array, Float, jaxtyped

from arpyes.types import (
    ArpesSpectrum,
    BandStructure,
    OrbitalProjection,
    PolarizationConfig,
    ScalarFloat,
    SimulationParams,
    make_band_structure,
    make_orbital_projection,
    make_polarization_config,
    make_simulation_params,
)

from .spectrum import (
    simulate_advanced,
    simulate_basic,
    simulate_basicplus,
    simulate_expert,
    simulate_novice,
)


@jaxtyped(typechecker=beartype)
def _build_inputs(
    eigenbands: Float[Array, "K B"],
    surface_orb: Float[Array, "K B A 9"],
    ef: ScalarFloat,
) -> Tuple[BandStructure, OrbitalProjection]:
    """Convert plain arrays into core ARPES input PyTrees."""
    bands_arr: Float[Array, "K B"] = jnp.asarray(eigenbands, dtype=jnp.float64)
    proj_arr: Float[Array, "K B A 9"] = jnp.asarray(
        surface_orb, dtype=jnp.float64
    )
    nkpts: int = bands_arr.shape[0]
    kpoints: Float[Array, "K 3"] = jnp.zeros((nkpts, 3), dtype=jnp.float64)
    bands: BandStructure = make_band_structure(
        eigenvalues=bands_arr,
        kpoints=kpoints,
        fermi_energy=ef,
    )
    orb_proj: OrbitalProjection = make_orbital_projection(projections=proj_arr)
    return bands, orb_proj


@jaxtyped(typechecker=beartype)
def simulate_novice_expanded(
    eigenbands: Float[Array, "K B"],
    surface_orb: Float[Array, "K B A 9"],
    ef: ScalarFloat,
    sigma: ScalarFloat,
    gamma: ScalarFloat,
    fidelity: int,
    temperature: ScalarFloat,
    photon_energy: ScalarFloat,
) -> ArpesSpectrum:
    """Run novice simulation with expanded inputs."""
    bands: BandStructure
    orb_proj: OrbitalProjection
    bands, orb_proj = _build_inputs(
        eigenbands=eigenbands,
        surface_orb=surface_orb,
        ef=ef,
    )
    params: SimulationParams = make_simulation_params(
        eigenbands=eigenbands,
        fidelity=fidelity,
        sigma=sigma,
        gamma=gamma,
        temperature=temperature,
        photon_energy=photon_energy,
    )
    spectrum: ArpesSpectrum = simulate_novice(bands, orb_proj, params)
    return spectrum


@jaxtyped(typechecker=beartype)
def simulate_basic_expanded(
    eigenbands: Float[Array, "K B"],
    surface_orb: Float[Array, "K B A 9"],
    ef: ScalarFloat,
    sigma: ScalarFloat,
    fidelity: int,
    temperature: ScalarFloat,
    photon_energy: ScalarFloat,
) -> ArpesSpectrum:
    """Run basic simulation with expanded inputs."""
    bands: BandStructure
    orb_proj: OrbitalProjection
    bands, orb_proj = _build_inputs(
        eigenbands=eigenbands,
        surface_orb=surface_orb,
        ef=ef,
    )
    params: SimulationParams = make_simulation_params(
        eigenbands=eigenbands,
        fidelity=fidelity,
        sigma=sigma,
        temperature=temperature,
        photon_energy=photon_energy,
    )
    spectrum: ArpesSpectrum = simulate_basic(bands, orb_proj, params)
    return spectrum


@jaxtyped(typechecker=beartype)
def simulate_basicplus_expanded(
    eigenbands: Float[Array, "K B"],
    surface_orb: Float[Array, "K B A 9"],
    ef: ScalarFloat,
    sigma: ScalarFloat,
    fidelity: int,
    temperature: ScalarFloat,
    photon_energy: ScalarFloat,
) -> ArpesSpectrum:
    """Run basicplus simulation with expanded inputs."""
    bands: BandStructure
    orb_proj: OrbitalProjection
    bands, orb_proj = _build_inputs(
        eigenbands=eigenbands,
        surface_orb=surface_orb,
        ef=ef,
    )
    params: SimulationParams = make_simulation_params(
        eigenbands=eigenbands,
        fidelity=fidelity,
        sigma=sigma,
        temperature=temperature,
        photon_energy=photon_energy,
    )
    spectrum: ArpesSpectrum = simulate_basicplus(bands, orb_proj, params)
    return spectrum


@jaxtyped(typechecker=beartype)
def simulate_advanced_expanded(  # noqa: PLR0913
    eigenbands: Float[Array, "K B"],
    surface_orb: Float[Array, "K B A 9"],
    ef: ScalarFloat,
    sigma: ScalarFloat,
    fidelity: int,
    temperature: ScalarFloat,
    photon_energy: ScalarFloat,
    polarization: str = "unpolarized",
    incident_theta: ScalarFloat = 45.0,
    incident_phi: ScalarFloat = 0.0,
    polarization_angle: ScalarFloat = 0.0,
) -> ArpesSpectrum:
    """Run advanced simulation with expanded inputs."""
    bands: BandStructure
    orb_proj: OrbitalProjection
    bands, orb_proj = _build_inputs(
        eigenbands=eigenbands,
        surface_orb=surface_orb,
        ef=ef,
    )
    params: SimulationParams = make_simulation_params(
        eigenbands=eigenbands,
        fidelity=fidelity,
        sigma=sigma,
        temperature=temperature,
        photon_energy=photon_energy,
    )
    pol: PolarizationConfig = make_polarization_config(
        polarization=polarization,
        incident_theta=incident_theta,
        incident_phi=incident_phi,
        polarization_angle=polarization_angle,
    )
    spectrum: ArpesSpectrum = simulate_advanced(bands, orb_proj, params, pol)
    return spectrum


@jaxtyped(typechecker=beartype)
def simulate_expert_expanded(  # noqa: PLR0913
    eigenbands: Float[Array, "K B"],
    surface_orb: Float[Array, "K B A 9"],
    ef: ScalarFloat,
    sigma: ScalarFloat,
    gamma: ScalarFloat,
    fidelity: int,
    temperature: ScalarFloat,
    photon_energy: ScalarFloat,
    polarization: str = "unpolarized",
    incident_theta: ScalarFloat = 45.0,
    incident_phi: ScalarFloat = 0.0,
    polarization_angle: ScalarFloat = 0.0,
) -> ArpesSpectrum:
    """Run expert simulation with expanded inputs."""
    bands: BandStructure
    orb_proj: OrbitalProjection
    bands, orb_proj = _build_inputs(
        eigenbands=eigenbands,
        surface_orb=surface_orb,
        ef=ef,
    )
    params: SimulationParams = make_simulation_params(
        eigenbands=eigenbands,
        fidelity=fidelity,
        sigma=sigma,
        gamma=gamma,
        temperature=temperature,
        photon_energy=photon_energy,
    )
    pol: PolarizationConfig = make_polarization_config(
        polarization=polarization,
        incident_theta=incident_theta,
        incident_phi=incident_phi,
        polarization_angle=polarization_angle,
    )
    spectrum: ArpesSpectrum = simulate_expert(bands, orb_proj, params, pol)
    return spectrum


@jaxtyped(typechecker=beartype)
def simulate_expanded(  # noqa: PLR0913
    level: str,
    eigenbands: Float[Array, "K B"],
    surface_orb: Float[Array, "K B A 9"],
    ef: ScalarFloat = 0.0,
    sigma: ScalarFloat = 0.04,
    gamma: ScalarFloat = 0.1,
    fidelity: int = 25000,
    temperature: ScalarFloat = 15.0,
    photon_energy: ScalarFloat = 11.0,
    polarization: str = "unpolarized",
    incident_theta: ScalarFloat = 45.0,
    incident_phi: ScalarFloat = 0.0,
    polarization_angle: ScalarFloat = 0.0,
) -> ArpesSpectrum:
    """Dispatch an expanded-input simulation by complexity level.

    Parameters
    ----------
    level : str
        One of ``"novice"``, ``"basic"``, ``"basicplus"``,
        ``"advanced"``, or ``"expert"`` (case-insensitive).
    eigenbands : Float[Array, "K B"]
        Band energies.
    surface_orb : Float[Array, "K B A 9"]
        Orbital projections.
    ef : ScalarFloat, optional
        Fermi level in eV. Default is 0.
    sigma : ScalarFloat, optional
        Gaussian broadening in eV. Default is 0.04.
    gamma : ScalarFloat, optional
        Lorentzian broadening in eV. Used by novice/expert.
    fidelity : int, optional
        Energy-axis size. Default is 25000.
    temperature : ScalarFloat, optional
        Temperature in Kelvin. Default is 15.
    photon_energy : ScalarFloat, optional
        Photon energy in eV. Default is 11.
    polarization : str, optional
        Polarization type for advanced/expert.
    incident_theta : ScalarFloat, optional
        Incident theta in degrees.
    incident_phi : ScalarFloat, optional
        Incident phi in degrees.
    polarization_angle : ScalarFloat, optional
        Arbitrary linear polarization angle in radians.

    Returns
    -------
    spectrum : ArpesSpectrum
        Simulated ARPES spectrum.
    """
    level_key: str = level.lower()
    if level_key == "novice":
        return simulate_novice_expanded(
            eigenbands=eigenbands,
            surface_orb=surface_orb,
            ef=ef,
            sigma=sigma,
            gamma=gamma,
            fidelity=fidelity,
            temperature=temperature,
            photon_energy=photon_energy,
        )
    if level_key == "basic":
        return simulate_basic_expanded(
            eigenbands=eigenbands,
            surface_orb=surface_orb,
            ef=ef,
            sigma=sigma,
            fidelity=fidelity,
            temperature=temperature,
            photon_energy=photon_energy,
        )
    if level_key == "basicplus":
        return simulate_basicplus_expanded(
            eigenbands=eigenbands,
            surface_orb=surface_orb,
            ef=ef,
            sigma=sigma,
            fidelity=fidelity,
            temperature=temperature,
            photon_energy=photon_energy,
        )
    if level_key == "advanced":
        return simulate_advanced_expanded(
            eigenbands=eigenbands,
            surface_orb=surface_orb,
            ef=ef,
            sigma=sigma,
            fidelity=fidelity,
            temperature=temperature,
            photon_energy=photon_energy,
            polarization=polarization,
            incident_theta=incident_theta,
            incident_phi=incident_phi,
            polarization_angle=polarization_angle,
        )
    if level_key == "expert":
        return simulate_expert_expanded(
            eigenbands=eigenbands,
            surface_orb=surface_orb,
            ef=ef,
            sigma=sigma,
            gamma=gamma,
            fidelity=fidelity,
            temperature=temperature,
            photon_energy=photon_energy,
            polarization=polarization,
            incident_theta=incident_theta,
            incident_phi=incident_phi,
            polarization_angle=polarization_angle,
        )
    msg: str = (
        "Unknown simulation level. "
        "Expected one of: novice, basic, basicplus, advanced, expert."
    )
    raise ValueError(msg)


__all__: list[str] = [
    "_build_inputs",
    "simulate_advanced_expanded",
    "simulate_basic_expanded",
    "simulate_basicplus_expanded",
    "simulate_novice_expanded",
    "simulate_expert_expanded",
    "simulate_expanded",
]
