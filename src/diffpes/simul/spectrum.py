r"""Simulate deliberately incoherent ARPES projection spectra.

Extended Summary
----------------
This module retains two lightweight VASP-projection tiers.  Both form one
incoherent orbital reduction, :math:`\sum_o |c_o|^2 w_o`, before applying
occupation and energy broadening.  They do not preserve orbital phases or
inter-centre interference.  Quantitative coherent photoemission amplitudes
belong to :mod:`diffpes.simul.matrixel`.

Routine Listings
----------------
:func:`simulate_basic`
    Simulate an incoherent spectrum with Yeh--Lindau weights.
:func:`simulate_novice`
    Simulate an incoherent spectrum with uniform orbital weights.
"""

import jax
import jax.numpy as jnp
from beartype import beartype
from jaxtyping import Array, Float, jaxtyped

from diffpes.types import (
    NON_S_ORBITAL_SLICE,
    ArpesSpectrum,
    BandStructure,
    OrbitalBasis,
    OrbitalProjection,
    ScalarFloat,
    SimulationParams,
    make_arpes_spectrum,
)

from .broadening import fermi_dirac, gaussian, voigt
from .crosssections import yeh_lindau_orbital_weights


def _validate_projection_basis(
    projections: Float[Array, "K B A nine"],
    basis: OrbitalBasis,
) -> None:
    """Validate atom-major VASP projection and basis alignment."""
    atom_count: int = projections.shape[2]
    orbital_count: int = projections.shape[2] * projections.shape[3]
    if len(basis.n) != orbital_count:
        message: str = (
            "basis must contain one row for every atom-major VASP "
            f"projection channel ({orbital_count} rows required)"
        )
        raise ValueError(message)
    expected_atoms: tuple[int, ...] = tuple(
        atom_index
        for atom_index in range(atom_count)
        for _ in range(projections.shape[3])
    )
    if basis.atom_indices != expected_atoms:
        message = (
            "basis.atom_indices must follow the atom-major projection order"
        )
        raise ValueError(message)
    vasp_angular_order: tuple[int, ...] = (0, 1, 1, 1, 2, 2, 2, 2, 2)
    expected_angular: tuple[int, ...] = vasp_angular_order * atom_count
    if basis.l != expected_angular:
        message = "basis.l must follow the atom-major VASP orbital order"
        raise ValueError(message)
    atom_index: int
    channel_offset: int
    for atom_index in range(atom_count):
        channel_offset = atom_index * projections.shape[3]
        p_principals: tuple[int, ...] = basis.n[
            channel_offset + 1 : channel_offset + 4
        ]
        d_principals: tuple[int, ...] = basis.n[
            channel_offset + 4 : channel_offset + 9
        ]
        if len(set(p_principals)) != 1 or len(set(d_principals)) != 1:
            message = "each atom-major VASP l shell must share one n value"
            raise ValueError(message)


@jaxtyped(typechecker=beartype)
def simulate_novice(
    bands: BandStructure,
    orb_proj: OrbitalProjection,
    params: SimulationParams,
    temperature: ScalarFloat,
) -> ArpesSpectrum:
    """Simulate an incoherent spectrum with uniform orbital weights.

    **Incoherent approximation tier.** This function consumes VASP projection
    probabilities and therefore cannot reproduce phase-sensitive matrix
    elements or inter-centre interference.  Use
    :mod:`diffpes.simul.matrixel` for coherent amplitudes.

    :see: :class:`~.test_spectrum.TestSimulateNovice`

    Parameters
    ----------
    bands : BandStructure
        Band eigenvalues and Fermi energy.
    orb_proj : OrbitalProjection
        VASP-order projection probabilities with shape ``(K, B, A, 9)``.
    params : SimulationParams
        Energy window, Voigt widths, and fidelity.
    temperature : ScalarFloat
        Positive sample temperature in kelvin.

    Returns
    -------
    spectrum : ArpesSpectrum
        Uniformly weighted incoherent intensity on the requested energy grid.

    Notes
    -----
    The function sums non-s projection probabilities once. It then broadens
    each band with a Voigt profile and sums band contributions.
    """
    energy_axis: Float[Array, " E"] = jnp.linspace(
        params.energy_min,
        params.energy_max,
        params.fidelity,
    )
    band_weights: Float[Array, "K B"] = jnp.sum(
        orb_proj.projections[..., NON_S_ORBITAL_SLICE],
        axis=(-2, -1),
    )

    def broaden_band(
        energy: Float[Array, ""],
        weight: Float[Array, ""],
    ) -> Float[Array, " E"]:
        occupation: Float[Array, ""] = fermi_dirac(
            energy,
            bands.fermi_energy,
            temperature,
        )
        profile: Float[Array, " E"] = voigt(
            energy_axis,
            energy,
            params.sigma,
            params.gamma,
        )
        contribution: Float[Array, " E"] = weight * occupation * profile
        return contribution

    def broaden_kpoint(
        energies: Float[Array, " B"],
        weights: Float[Array, " B"],
    ) -> Float[Array, " E"]:
        contributions: Float[Array, "B E"] = jax.vmap(broaden_band)(
            energies,
            weights,
        )
        intensity_row: Float[Array, " E"] = jnp.sum(contributions, axis=0)
        return intensity_row

    intensity: Float[Array, "K E"] = jax.vmap(broaden_kpoint)(
        bands.eigenvalues,
        band_weights,
    )
    spectrum: ArpesSpectrum = make_arpes_spectrum(
        intensity=intensity,
        energy_axis=energy_axis,
    )
    return spectrum


@jaxtyped(typechecker=beartype)
def simulate_basic(
    bands: BandStructure,
    orb_proj: OrbitalProjection,
    params: SimulationParams,
    basis: OrbitalBasis,
    atomic_numbers: tuple[int, ...],
    temperature: ScalarFloat,
    photon_energy: ScalarFloat,
) -> ArpesSpectrum:
    """Simulate an incoherent spectrum with Yeh--Lindau weights.

    **Incoherent approximation tier.** The element- and subshell-resolved
    Yeh--Lindau weights are physical isolated-atom cross sections, but the
    input projections contain no orbital phases.  The function consequently
    performs one probability-level orbital reduction.  Use
    :mod:`diffpes.simul.matrixel` for coherent amplitudes.

    :see: :class:`~.test_spectrum.TestSimulateBasic`

    Parameters
    ----------
    bands : BandStructure
        Band eigenvalues and Fermi energy.
    orb_proj : OrbitalProjection
        VASP-order projection probabilities with shape ``(K, B, A, 9)``.
    params : SimulationParams
        Energy window, Gaussian width, and fidelity.
    basis : OrbitalBasis
        One atom-major row per projection channel, carrying the subshell
        ``(n, l)`` identity needed by the Yeh--Lindau table.
    atomic_numbers : tuple[int, ...]
        Atomic number for every atom axis in ``orb_proj``.
    temperature : ScalarFloat
        Positive sample temperature in kelvin.
    photon_energy : ScalarFloat
        Photon energy in eV for Yeh--Lindau interpolation.

    Returns
    -------
    spectrum : ArpesSpectrum
        Cross-section-weighted incoherent intensity.

    Raises
    ------
    ValueError
        If ``basis`` is not aligned with the atom-major projection layout or
        ``atomic_numbers`` does not cover every atom.

    Notes
    -----
    The function gathers one tabulated subshell weight per basis row. It
    multiplies projection probabilities before one orbital reduction.
    """
    projections: Float[Array, "K B A nine"] = orb_proj.projections
    _validate_projection_basis(projections, basis)
    if len(atomic_numbers) != projections.shape[2]:
        message: str = (
            "atomic_numbers must contain one entry per projection atom"
        )
        raise ValueError(message)
    orbital_weights: Float[Array, " n_orb"] = yeh_lindau_orbital_weights(
        photon_energy,
        basis,
        atomic_numbers,
    )
    flattened: Float[Array, "K B n_orb"] = projections.reshape(
        projections.shape[0],
        projections.shape[1],
        -1,
    )
    band_weights: Float[Array, "K B"] = jnp.sum(
        flattened * orbital_weights,
        axis=-1,
    )
    energy_axis: Float[Array, " E"] = jnp.linspace(
        params.energy_min,
        params.energy_max,
        params.fidelity,
    )

    def broaden_band(
        energy: Float[Array, ""],
        weight: Float[Array, ""],
    ) -> Float[Array, " E"]:
        occupation: Float[Array, ""] = fermi_dirac(
            energy,
            bands.fermi_energy,
            temperature,
        )
        profile: Float[Array, " E"] = gaussian(
            energy_axis,
            energy,
            params.sigma,
        )
        contribution: Float[Array, " E"] = weight * occupation * profile
        return contribution

    def broaden_kpoint(
        energies: Float[Array, " B"],
        weights: Float[Array, " B"],
    ) -> Float[Array, " E"]:
        contributions: Float[Array, "B E"] = jax.vmap(broaden_band)(
            energies,
            weights,
        )
        intensity_row: Float[Array, " E"] = jnp.sum(contributions, axis=0)
        return intensity_row

    intensity: Float[Array, "K E"] = jax.vmap(broaden_kpoint)(
        bands.eigenvalues,
        band_weights,
    )
    spectrum: ArpesSpectrum = make_arpes_spectrum(
        intensity=intensity,
        energy_axis=energy_axis,
    )
    return spectrum


__all__: list[str] = ["simulate_basic", "simulate_novice"]
