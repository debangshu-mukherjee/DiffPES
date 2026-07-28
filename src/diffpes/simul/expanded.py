"""Run the incoherent ARPES tiers from plain band/projection arrays.

Extended Summary
----------------
The wrappers construct validated carriers around plain arrays.  Only the
``novice`` and ``basic`` incoherent projection tiers remain.  Coherent
matrix-element calculations use :mod:`diffpes.simul.matrixel` directly.

Routine Listings
----------------
:func:`simulate_basic_expanded`
    Run the Yeh--Lindau-weighted incoherent tier from plain arrays.
:func:`simulate_expanded`
    Dispatch one of the two retained incoherent simulation tiers.
:func:`simulate_novice_expanded`
    Run the uniformly weighted incoherent tier from plain arrays.
"""

import jax.numpy as jnp
from beartype import beartype
from beartype.typing import Optional
from jaxtyping import Array, Float, jaxtyped

from diffpes.types import (
    ArpesSpectrum,
    BandStructure,
    OrbitalBasis,
    OrbitalProjection,
    ScalarFloat,
    SimulationParams,
    make_band_structure,
    make_expanded_simulation_params,
    make_orbital_projection,
)

from .spectrum import simulate_basic, simulate_novice


@jaxtyped(typechecker=beartype)
def _build_inputs(
    eigenbands: Float[Array, "K B"],
    surface_orb: Float[Array, "K B A 9"],
    ef: ScalarFloat,
) -> tuple[BandStructure, OrbitalProjection]:
    """Convert plain arrays into validated spectrum input carriers."""
    bands_array: Float[Array, "K B"] = jnp.asarray(
        eigenbands,
        dtype=jnp.float64,
    )
    projection_array: Float[Array, "K B A 9"] = jnp.asarray(
        surface_orb,
        dtype=jnp.float64,
    )
    kpoints: Float[Array, "K 3"] = jnp.zeros(
        (bands_array.shape[0], 3),
        dtype=jnp.float64,
    )
    bands: BandStructure = make_band_structure(
        eigenvalues=bands_array,
        kpoints=kpoints,
        fermi_energy=ef,
    )
    projection: OrbitalProjection = make_orbital_projection(
        projections=projection_array,
    )
    inputs: tuple[BandStructure, OrbitalProjection] = (bands, projection)
    return inputs


@jaxtyped(typechecker=beartype)
def simulate_novice_expanded(
    eigenbands: Float[Array, "K B"],
    surface_orb: Float[Array, "K B A 9"],
    ef: ScalarFloat,
    sigma: ScalarFloat,
    gamma: ScalarFloat,
    fidelity: int,
    temperature: ScalarFloat,
) -> ArpesSpectrum:
    """Run the uniformly weighted incoherent tier from plain arrays.

    **Incoherent approximation tier.** This wrapper does not recover phases
    discarded by projection data.

    :see: :class:`~.test_expanded.TestSimulateNoviceExpanded`

    Parameters
    ----------
    eigenbands : Float[Array, "K B"]
        Band energies in eV.
    surface_orb : Float[Array, "K B A 9"]
        VASP-order projection probabilities.
    ef : ScalarFloat
        Fermi energy in eV.
    sigma : ScalarFloat
        Gaussian component of the Voigt width in eV.
    gamma : ScalarFloat
        Lorentzian component of the Voigt width in eV.
    fidelity : int
        Number of energy samples.
    temperature : ScalarFloat
        Electronic temperature in Kelvin.

    Returns
    -------
    spectrum : ArpesSpectrum
        Uniformly weighted incoherent spectrum.

    Notes
    -----
    The wrapper builds validated carriers and delegates numerical work to
    :func:`simulate_novice`.
    """
    bands: BandStructure
    projection: OrbitalProjection
    bands, projection = _build_inputs(eigenbands, surface_orb, ef)
    params: SimulationParams = make_expanded_simulation_params(
        eigenbands=eigenbands,
        sigma=sigma,
        gamma=gamma,
        fidelity=fidelity,
    )
    spectrum: ArpesSpectrum = simulate_novice(
        bands,
        projection,
        params,
        temperature,
    )
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
    basis: OrbitalBasis,
    atomic_numbers: tuple[int, ...],
) -> ArpesSpectrum:
    """Run the Yeh--Lindau-weighted incoherent tier from plain arrays.

    **Incoherent approximation tier.** Authenticated subshell data supplies
    cross sections. The VASP projections remain probability-only input.

    :see: :class:`~.test_expanded.TestSimulateBasicExpanded`

    Parameters
    ----------
    eigenbands : Float[Array, "K B"]
        Band energies in eV.
    surface_orb : Float[Array, "K B A 9"]
        VASP-order projection probabilities.
    ef : ScalarFloat
        Fermi energy in eV.
    sigma : ScalarFloat
        Gaussian broadening width in eV.
    fidelity : int
        Number of energy samples.
    temperature : ScalarFloat
        Electronic temperature in Kelvin.
    photon_energy : ScalarFloat
        Photon energy in eV for Yeh--Lindau interpolation.
    basis : OrbitalBasis
        Atom-major subshell identity for every projection channel.
    atomic_numbers : tuple[int, ...]
        Atomic number for every projection atom.

    Returns
    -------
    spectrum : ArpesSpectrum
        Cross-section-weighted incoherent spectrum.

    Notes
    -----
    The wrapper builds validated carriers and delegates numerical work to
    :func:`simulate_basic`.
    """
    bands: BandStructure
    projection: OrbitalProjection
    bands, projection = _build_inputs(eigenbands, surface_orb, ef)
    params: SimulationParams = make_expanded_simulation_params(
        eigenbands=eigenbands,
        sigma=sigma,
        fidelity=fidelity,
    )
    spectrum: ArpesSpectrum = simulate_basic(
        bands,
        projection,
        params,
        basis,
        atomic_numbers,
        temperature,
        photon_energy,
    )
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
    photon_energy: ScalarFloat = 21.2,
    basis: Optional[OrbitalBasis] = None,
    atomic_numbers: Optional[tuple[int, ...]] = None,
) -> ArpesSpectrum:
    """Dispatch one of the two retained incoherent simulation tiers.

    The static level selector routes raw arrays through the corresponding
    retained wrapper. The basic route requires explicit atomic metadata.

    :see: :class:`~.test_expanded.TestSimulateExpanded`

    Parameters
    ----------
    level : str
        ``"novice"`` or ``"basic"`` (case-insensitive).
    eigenbands : Float[Array, "K B"]
        Band energies in eV.
    surface_orb : Float[Array, "K B A 9"]
        VASP-order projection probabilities.
    ef : ScalarFloat, optional
        Fermi energy in eV.
    sigma : ScalarFloat, optional
        Gaussian broadening width in eV.
    gamma : ScalarFloat, optional
        Lorentzian width used by the novice tier.
    fidelity : int, optional
        Number of energy samples.
    temperature : ScalarFloat, optional
        Electronic temperature in Kelvin.
    photon_energy : ScalarFloat, optional
        Photon energy in eV.
    basis : Optional[OrbitalBasis], optional
        Required for the basic tier.
    atomic_numbers : Optional[tuple[int, ...]], optional
        Required for the basic tier.

    Returns
    -------
    spectrum : ArpesSpectrum
        Selected incoherent spectrum.

    Raises
    ------
    ValueError
        If the caller selects another level or omits basic atomic metadata.

    Notes
    -----
    The function normalizes the level key. It forwards only arguments owned by
    the selected incoherent tier.
    """
    level_key: str = level.lower()
    if level_key == "novice":
        spectrum: ArpesSpectrum = simulate_novice_expanded(
            eigenbands=eigenbands,
            surface_orb=surface_orb,
            ef=ef,
            sigma=sigma,
            gamma=gamma,
            fidelity=fidelity,
            temperature=temperature,
        )
    elif level_key == "basic":
        if basis is None or atomic_numbers is None:
            message: str = (
                "simulate_expanded(level='basic') requires basis and "
                "atomic_numbers"
            )
            raise ValueError(message)
        spectrum = simulate_basic_expanded(
            eigenbands=eigenbands,
            surface_orb=surface_orb,
            ef=ef,
            sigma=sigma,
            fidelity=fidelity,
            temperature=temperature,
            photon_energy=photon_energy,
            basis=basis,
            atomic_numbers=atomic_numbers,
        )
    else:
        message = "unknown simulation level; expected 'novice' or 'basic'"
        raise ValueError(message)
    return spectrum


__all__: list[str] = [
    "simulate_basic_expanded",
    "simulate_expanded",
    "simulate_novice_expanded",
]
