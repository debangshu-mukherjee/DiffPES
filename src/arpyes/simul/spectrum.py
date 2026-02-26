"""ARPES spectrum simulation functions at five complexity levels.

Extended Summary
----------------
Provides five simulation functions of increasing physical
sophistication, from basic Voigt convolution (novice) to full
polarization-dependent dipole matrix element calculations
(expert). All functions are vectorized with ``jax.vmap`` for
efficient GPU execution.

Routine Listings
----------------
:func:`simulate_novice`
    Voigt broadening with uniform orbital weights.
:func:`simulate_basic`
    Gaussian broadening with heuristic orbital weights.
:func:`simulate_basicplus`
    Gaussian broadening with Yeh-Lindau cross-sections.
:func:`simulate_advanced`
    Gaussian with Yeh-Lindau and polarization selection rules.
:func:`simulate_expert`
    Voigt with Yeh-Lindau, polarization, and dipole elements.

Notes
-----
All functions accept :class:`~arpyes.types.BandStructure`,
:class:`~arpyes.types.OrbitalProjection`, and
:class:`~arpyes.types.SimulationParams` PyTrees and return an
:class:`~arpyes.types.ArpesSpectrum` PyTree.
"""

import jax
import jax.numpy as jnp
from beartype import beartype
from jaxtyping import Array, Float, jaxtyped

from arpyes.types import (
    ArpesSpectrum,
    BandStructure,
    OrbitalProjection,
    PolarizationConfig,
    SimulationParams,
    make_arpes_spectrum,
)

from .broadening import fermi_dirac, gaussian, voigt
from .crosssections import heuristic_weights, yeh_lindau_weights
from .polarization import (
    build_efield,
    build_polarization_vectors,
    dipole_matrix_elements,
)


@jaxtyped(typechecker=beartype)
def simulate_novice(
    bands: BandStructure,
    orb_proj: OrbitalProjection,
    params: SimulationParams,
) -> ArpesSpectrum:
    """Simulate ARPES spectrum with Voigt broadening.

    Parameters
    ----------
    bands : BandStructure
        Electronic band structure.
    orb_proj : OrbitalProjection
        Orbital projections (K, B, A, 9).
    params : SimulationParams
        Simulation parameters.

    Returns
    -------
    spectrum : ArpesSpectrum
        Simulated ARPES intensity map.

    Notes
    -----
    Uses Voigt profile (combined Gaussian-Lorentzian) and
    sums all non-s orbital contributions with equal weight.
    """
    energy_axis: Float[Array, " E"] = jnp.linspace(
        params.energy_min,
        params.energy_max,
        params.fidelity,
    )
    proj: Float[Array, "K B A 9"] = orb_proj.projections
    weights: Float[Array, "K B"] = jnp.sum(
        jnp.sum(proj[..., 1:9], axis=-1), axis=-1
    )

    def _single_band(
        energy: Float[Array, " "],
        weight: Float[Array, " "],
    ) -> Float[Array, " E"]:
        fd: Float[Array, " "] = fermi_dirac(
            energy, bands.fermi_energy, params.temperature
        )
        profile: Float[Array, " E"] = voigt(
            energy_axis, energy, params.sigma, params.gamma
        )
        contribution: Float[Array, " E"] = (
            weight * fd * profile
        )
        return contribution

    def _single_kpoint(
        energies: Float[Array, " B"],
        kweights: Float[Array, " B"],
    ) -> Float[Array, " E"]:
        contributions: Float[Array, "B E"] = jax.vmap(
            _single_band
        )(energies, kweights)
        total: Float[Array, " E"] = jnp.sum(
            contributions, axis=0
        )
        return total

    intensity: Float[Array, "K E"] = jax.vmap(
        _single_kpoint
    )(bands.eigenvalues, weights)
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
) -> ArpesSpectrum:
    """Simulate ARPES spectrum with Gaussian broadening.

    Parameters
    ----------
    bands : BandStructure
        Electronic band structure.
    orb_proj : OrbitalProjection
        Orbital projections (K, B, A, 9).
    params : SimulationParams
        Simulation parameters.

    Returns
    -------
    spectrum : ArpesSpectrum
        Simulated ARPES intensity map.

    Notes
    -----
    Uses Gaussian broadening with energy-dependent heuristic
    orbital weights (p-enhanced below 50 eV, d-enhanced above).
    """
    energy_axis: Float[Array, " E"] = jnp.linspace(
        params.energy_min,
        params.energy_max,
        params.fidelity,
    )
    orb_w: Float[Array, " 9"] = heuristic_weights(
        params.photon_energy
    )
    proj: Float[Array, "K B A 9"] = orb_proj.projections
    weighted_proj: Float[Array, "K B A 9"] = (
        proj[..., 1:9] * orb_w[1:9]
    )
    weights: Float[Array, "K B"] = jnp.sum(
        jnp.sum(weighted_proj, axis=-1), axis=-1
    )

    def _single_band(
        energy: Float[Array, " "],
        weight: Float[Array, " "],
    ) -> Float[Array, " E"]:
        fd: Float[Array, " "] = fermi_dirac(
            energy, bands.fermi_energy, params.temperature
        )
        profile: Float[Array, " E"] = gaussian(
            energy_axis, energy, params.sigma
        )
        contribution: Float[Array, " E"] = (
            weight * fd * profile
        )
        return contribution

    def _single_kpoint(
        energies: Float[Array, " B"],
        kweights: Float[Array, " B"],
    ) -> Float[Array, " E"]:
        contributions: Float[Array, "B E"] = jax.vmap(
            _single_band
        )(energies, kweights)
        total: Float[Array, " E"] = jnp.sum(
            contributions, axis=0
        )
        return total

    intensity: Float[Array, "K E"] = jax.vmap(
        _single_kpoint
    )(bands.eigenvalues, weights)
    spectrum: ArpesSpectrum = make_arpes_spectrum(
        intensity=intensity,
        energy_axis=energy_axis,
    )
    return spectrum


@jaxtyped(typechecker=beartype)
def simulate_basicplus(
    bands: BandStructure,
    orb_proj: OrbitalProjection,
    params: SimulationParams,
) -> ArpesSpectrum:
    """Simulate ARPES with Yeh-Lindau cross-sections.

    Parameters
    ----------
    bands : BandStructure
        Electronic band structure.
    orb_proj : OrbitalProjection
        Orbital projections (K, B, A, 9).
    params : SimulationParams
        Simulation parameters.

    Returns
    -------
    spectrum : ArpesSpectrum
        Simulated ARPES intensity map.

    Notes
    -----
    Uses Gaussian broadening with interpolated Yeh-Lindau
    photoionization cross-section weights per orbital type.
    """
    energy_axis: Float[Array, " E"] = jnp.linspace(
        params.energy_min,
        params.energy_max,
        params.fidelity,
    )
    orb_w: Float[Array, " 9"] = yeh_lindau_weights(
        params.photon_energy
    )
    proj: Float[Array, "K B A 9"] = orb_proj.projections
    weighted_proj: Float[Array, "K B A 9"] = (
        proj * orb_w
    )
    weights: Float[Array, "K B"] = jnp.sum(
        jnp.sum(weighted_proj[..., 1:9], axis=-1), axis=-1
    )

    def _single_band(
        energy: Float[Array, " "],
        weight: Float[Array, " "],
    ) -> Float[Array, " E"]:
        fd: Float[Array, " "] = fermi_dirac(
            energy, bands.fermi_energy, params.temperature
        )
        profile: Float[Array, " E"] = gaussian(
            energy_axis, energy, params.sigma
        )
        contribution: Float[Array, " E"] = (
            weight * fd * profile
        )
        return contribution

    def _single_kpoint(
        energies: Float[Array, " B"],
        kweights: Float[Array, " B"],
    ) -> Float[Array, " E"]:
        contributions: Float[Array, "B E"] = jax.vmap(
            _single_band
        )(energies, kweights)
        total: Float[Array, " E"] = jnp.sum(
            contributions, axis=0
        )
        return total

    intensity: Float[Array, "K E"] = jax.vmap(
        _single_kpoint
    )(bands.eigenvalues, weights)
    spectrum: ArpesSpectrum = make_arpes_spectrum(
        intensity=intensity,
        energy_axis=energy_axis,
    )
    return spectrum


@jaxtyped(typechecker=beartype)
def simulate_advanced(
    bands: BandStructure,
    orb_proj: OrbitalProjection,
    params: SimulationParams,
    pol_config: PolarizationConfig,
) -> ArpesSpectrum:
    """Simulate ARPES with polarization selection rules.

    Parameters
    ----------
    bands : BandStructure
        Electronic band structure.
    orb_proj : OrbitalProjection
        Orbital projections (K, B, A, 9).
    params : SimulationParams
        Simulation parameters.
    pol_config : PolarizationConfig
        Light polarization configuration.

    Returns
    -------
    spectrum : ArpesSpectrum
        Simulated ARPES intensity map.

    Notes
    -----
    Uses Gaussian broadening with Yeh-Lindau cross-sections
    and polarization-dependent orbital selection via
    |e_field dot d_orbital|^2 weighting.
    For unpolarized light, averages s and p contributions.
    """
    energy_axis: Float[Array, " E"] = jnp.linspace(
        params.energy_min,
        params.energy_max,
        params.fidelity,
    )
    orb_w: Float[Array, " 9"] = yeh_lindau_weights(
        params.photon_energy
    )
    proj: Float[Array, "K B A 9"] = orb_proj.projections
    is_unpolarized: bool = (
        pol_config.polarization_type.lower() == "unpolarized"
    )
    if is_unpolarized:
        e_s, e_p = build_polarization_vectors(
            pol_config.theta, pol_config.phi
        )
        m_s: Float[Array, " 9"] = dipole_matrix_elements(
            e_s.astype(jnp.complex128)
        )
        m_p: Float[Array, " 9"] = dipole_matrix_elements(
            e_p.astype(jnp.complex128)
        )
        w_s: Float[Array, "K B A 9"] = (
            proj * orb_w * m_s
        )
        w_p: Float[Array, "K B A 9"] = (
            proj * orb_w * m_p
        )
        ws_sum: Float[Array, "K B"] = jnp.sum(
            jnp.sum(w_s[..., 1:9], axis=-1), axis=-1
        )
        wp_sum: Float[Array, "K B"] = jnp.sum(
            jnp.sum(w_p[..., 1:9], axis=-1), axis=-1
        )
        i_s: Float[Array, "K B"] = jnp.abs(ws_sum) ** 2
        i_p: Float[Array, "K B"] = jnp.abs(wp_sum) ** 2
        band_intensity: Float[Array, "K B"] = (
            i_s + i_p
        ) / 2.0
    else:
        efield = build_efield(pol_config)
        m_elem: Float[Array, " 9"] = (
            dipole_matrix_elements(efield)
        )
        weighted: Float[Array, "K B A 9"] = (
            proj * orb_w * m_elem
        )
        w_sum: Float[Array, "K B"] = jnp.sum(
            jnp.sum(weighted[..., 1:9], axis=-1), axis=-1
        )
        band_intensity = jnp.abs(w_sum) ** 2

    def _single_band(
        energy: Float[Array, " "],
        bi: Float[Array, " "],
    ) -> Float[Array, " E"]:
        fd: Float[Array, " "] = fermi_dirac(
            energy, bands.fermi_energy, params.temperature
        )
        profile: Float[Array, " E"] = gaussian(
            energy_axis, energy, params.sigma
        )
        contribution: Float[Array, " E"] = (
            bi * fd * profile
        )
        return contribution

    def _single_kpoint(
        energies: Float[Array, " B"],
        bi_k: Float[Array, " B"],
    ) -> Float[Array, " E"]:
        contributions: Float[Array, "B E"] = jax.vmap(
            _single_band
        )(energies, bi_k)
        total: Float[Array, " E"] = jnp.sum(
            contributions, axis=0
        )
        return total

    intensity: Float[Array, "K E"] = jax.vmap(
        _single_kpoint
    )(bands.eigenvalues, band_intensity)
    spectrum: ArpesSpectrum = make_arpes_spectrum(
        intensity=intensity,
        energy_axis=energy_axis,
    )
    return spectrum


@jaxtyped(typechecker=beartype)
def simulate_expert(
    bands: BandStructure,
    orb_proj: OrbitalProjection,
    params: SimulationParams,
    pol_config: PolarizationConfig,
) -> ArpesSpectrum:
    """Simulate ARPES with full dipole matrix elements.

    Parameters
    ----------
    bands : BandStructure
        Electronic band structure.
    orb_proj : OrbitalProjection
        Orbital projections (K, B, A, 9).
    params : SimulationParams
        Simulation parameters.
    pol_config : PolarizationConfig
        Light polarization configuration.

    Returns
    -------
    spectrum : ArpesSpectrum
        Simulated ARPES intensity map.

    Notes
    -----
    Uses Voigt broadening with Yeh-Lindau cross-sections,
    polarization selection rules, and dipole matrix element
    weighting. This is the most physically complete model.
    For unpolarized light, averages s and p contributions.
    """
    energy_axis: Float[Array, " E"] = jnp.linspace(
        params.energy_min,
        params.energy_max,
        params.fidelity,
    )
    orb_w: Float[Array, " 9"] = yeh_lindau_weights(
        params.photon_energy
    )
    proj: Float[Array, "K B A 9"] = orb_proj.projections
    is_unpolarized: bool = (
        pol_config.polarization_type.lower() == "unpolarized"
    )
    if is_unpolarized:
        e_s, e_p = build_polarization_vectors(
            pol_config.theta, pol_config.phi
        )
        m_s: Float[Array, " 9"] = dipole_matrix_elements(
            e_s.astype(jnp.complex128)
        )
        m_p: Float[Array, " 9"] = dipole_matrix_elements(
            e_p.astype(jnp.complex128)
        )
        w_s: Float[Array, "K B A 9"] = (
            proj * orb_w * m_s
        )
        w_p: Float[Array, "K B A 9"] = (
            proj * orb_w * m_p
        )
        ws_sum: Float[Array, "K B"] = jnp.sum(
            jnp.sum(w_s[..., 1:9], axis=-1), axis=-1
        )
        wp_sum: Float[Array, "K B"] = jnp.sum(
            jnp.sum(w_p[..., 1:9], axis=-1), axis=-1
        )
        i_s: Float[Array, "K B"] = jnp.abs(ws_sum) ** 2
        i_p: Float[Array, "K B"] = jnp.abs(wp_sum) ** 2
        band_intensity: Float[Array, "K B"] = (
            i_s + i_p
        ) / 2.0
    else:
        efield = build_efield(pol_config)
        m_elem: Float[Array, " 9"] = (
            dipole_matrix_elements(efield)
        )
        weighted: Float[Array, "K B A 9"] = (
            proj * orb_w * m_elem
        )
        w_sum: Float[Array, "K B"] = jnp.sum(
            jnp.sum(weighted[..., 1:9], axis=-1), axis=-1
        )
        band_intensity = jnp.abs(w_sum) ** 2

    def _single_band(
        energy: Float[Array, " "],
        bi: Float[Array, " "],
    ) -> Float[Array, " E"]:
        fd: Float[Array, " "] = fermi_dirac(
            energy, bands.fermi_energy, params.temperature
        )
        profile: Float[Array, " E"] = voigt(
            energy_axis, energy, params.sigma, params.gamma
        )
        contribution: Float[Array, " E"] = (
            bi * fd * profile
        )
        return contribution

    def _single_kpoint(
        energies: Float[Array, " B"],
        bi_k: Float[Array, " B"],
    ) -> Float[Array, " E"]:
        contributions: Float[Array, "B E"] = jax.vmap(
            _single_band
        )(energies, bi_k)
        total: Float[Array, " E"] = jnp.sum(
            contributions, axis=0
        )
        return total

    intensity: Float[Array, "K E"] = jax.vmap(
        _single_kpoint
    )(bands.eigenvalues, band_intensity)
    spectrum: ArpesSpectrum = make_arpes_spectrum(
        intensity=intensity,
        energy_axis=energy_axis,
    )
    return spectrum


__all__: list[str] = [
    "simulate_advanced",
    "simulate_basic",
    "simulate_basicplus",
    "simulate_expert",
    "simulate_novice",
]
