"""ARPES spectrum simulation at six complexity levels (incl. spin-orbit).

Extended Summary
----------------
Provides five simulation functions of increasing physical
sophistication, from basic Voigt convolution (novice) to full
polarization-dependent dipole matrix element calculations
(expert) and spin-orbit (soc). All functions are vectorized with
``jax.vmap`` for efficient GPU execution.

Routine Listings
----------------
:func:`simulate_advanced`
    Gaussian with Yeh-Lindau and polarization selection rules.
:func:`simulate_basic`
    Gaussian broadening with heuristic orbital weights.
:func:`simulate_basicplus`
    Gaussian broadening with Yeh-Lindau cross-sections.
:func:`simulate_expert`
    Voigt with Yeh-Lindau, polarization, and dipole elements.
:func:`simulate_novice`
    Voigt broadening with uniform orbital weights.
:func:`simulate_soc`
    Expert model plus spin-orbit (S·k_photon) correction.

Notes
-----
All functions accept :class:`~arpyes.types.BandStructure`,
:class:`~arpyes.types.OrbitalProjection`, and
:class:`~arpyes.types.SimulationParams` PyTrees and return an
:class:`~arpyes.types.ArpesSpectrum` PyTree.

The ``if is_unpolarized`` branches are intentional Python-side dispatch
(config choice); they are not traced, so only one path is compiled per call.
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
    ScalarFloat,
    SimulationParams,
    SpinOrbitalProjection,
    make_arpes_spectrum,
)

from .broadening import fermi_dirac, gaussian, voigt
from .crosssections import heuristic_weights, yeh_lindau_weights
from .polarization import (
    build_efield,
    build_polarization_vectors,
    dipole_matrix_elements,
    photon_wavevector,
)

_NON_S_ORBITAL_SLICE: slice = slice(1, 9)


@jaxtyped(typechecker=beartype)
def simulate_novice(
    bands: BandStructure,
    orb_proj: OrbitalProjection,
    params: SimulationParams,
) -> ArpesSpectrum:
    """Simulate ARPES spectrum with Voigt broadening and uniform weights.

    Entry-level simulation that convolves each band with a Voigt profile
    (combined Gaussian + Lorentzian) and applies Fermi-Dirac occupation.
    All non-s orbital projections (p, d) are summed with equal weight,
    making this the simplest model that still captures lifetime and
    instrumental broadening simultaneously.

    Implementation Logic
    --------------------
    1. **Build energy axis** via ``jnp.linspace(energy_min, energy_max,
       fidelity)`` to define the output energy grid of shape ``(E,)``.

    2. **Sum all non-s orbital projections**
       (``slice(1, 9)``, i.e., indices 1 through 8) across
       both orbital type and atom axes to produce uniform per-band
       weights of shape ``(K, B)``. The s-orbital (index 0) is excluded
       because it contributes negligible photoemission intensity at
       typical photon energies.

    3. **Define ``_single_band``**: for one band at one k-point, compute
       ``Fermi-Dirac(E_band) * Voigt(energy_axis, E_band, sigma, gamma)
       * weight`` to yield a spectral contribution of shape ``(E,)``.

    4. **Define ``_single_kpoint``**: ``jax.vmap`` ``_single_band`` over
       the band index B, then sum contributions to produce the total
       intensity at one k-point with shape ``(E,)``.

    5. **Outer vmap** ``_single_kpoint`` over k-points K to produce the
       full intensity array of shape ``(K, E)``.

    Parameters
    ----------
    bands : BandStructure
        Electronic band structure containing eigenvalues of shape
        ``(K, B)`` and the Fermi energy.
    orb_proj : OrbitalProjection
        Orbital projections of shape ``(K, B, A, 9)`` where A is the
        number of atoms and 9 is the number of orbital channels
        (s, p_y, p_z, p_x, d_xy, d_yz, d_z2, d_xz, d_x2-y2).
    params : SimulationParams
        Simulation parameters including ``sigma``, ``gamma``,
        ``temperature``, ``energy_min``, ``energy_max``, and
        ``fidelity``.

    Returns
    -------
    spectrum : ArpesSpectrum
        Simulated ARPES intensity map with ``intensity`` of shape
        ``(K, E)`` and ``energy_axis`` of shape ``(E,)``.

    See Also
    --------
    simulate_novice_expanded : Expanded variant that returns per-band
        contributions before summation.

    Notes
    -----
    Uses Voigt profile (combined Gaussian-Lorentzian) and sums all
    non-s orbital contributions with equal weight. This is appropriate
    when orbital cross-section data is unavailable or when a quick
    qualitative comparison with experiment is sufficient.
    """
    energy_axis: Float[Array, " E"] = jnp.linspace(
        params.energy_min,
        params.energy_max,
        params.fidelity,
    )
    proj: Float[Array, "K B A 9"] = orb_proj.projections
    weights: Float[Array, "K B"] = jnp.sum(
        jnp.sum(proj[..., _NON_S_ORBITAL_SLICE], axis=-1),
        axis=-1,
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
        contribution: Float[Array, " E"] = weight * fd * profile
        return contribution

    def _single_kpoint(
        energies: Float[Array, " B"],
        kweights: Float[Array, " B"],
    ) -> Float[Array, " E"]:
        contributions: Float[Array, "B E"] = jax.vmap(_single_band)(
            energies, kweights
        )
        total: Float[Array, " E"] = jnp.sum(contributions, axis=0)
        return total

    intensity: Float[Array, "K E"] = jax.vmap(_single_kpoint)(
        bands.eigenvalues, weights
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
) -> ArpesSpectrum:
    """Simulate ARPES spectrum with Gaussian broadening and heuristic weights.

    Intermediate simulation that replaces the Voigt profile with a pure
    Gaussian and introduces energy-dependent heuristic orbital weights.
    The heuristic weights enhance p-orbital contributions below ~50 eV
    photon energy and d-orbital contributions above, providing a
    first-order approximation to photoionization cross-sections without
    requiring tabulated atomic data.

    Implementation Logic
    --------------------
    1. **Build energy axis** via ``jnp.linspace(energy_min, energy_max,
       fidelity)`` to define the output energy grid of shape ``(E,)``.

    2. **Compute heuristic weights** from ``photon_energy`` via
       :func:`~arpyes.simul.crosssections.heuristic_weights`, returning
       a weight vector of shape ``(9,)`` with empirical cross-section
       approximations per orbital channel.

    3. **Weight projections by heuristic weights** (skipping the
       s-orbital at index 0): multiply
       ``proj[..., slice(1, 9)]`` element-wise by
       ``orb_w[slice(1, 9)]``, then sum over orbital and atom axes to yield
       per-band weights of shape ``(K, B)``.

    4. **Define ``_single_band``**: for one band at one k-point, compute
       ``Fermi-Dirac(E_band) * Gaussian(energy_axis, E_band, sigma)
       * weight`` to yield a spectral contribution of shape ``(E,)``.

    5. **Double vmap over k-points and bands**: ``jax.vmap``
       ``_single_band`` over B inside ``_single_kpoint``, then
       ``jax.vmap`` ``_single_kpoint`` over K to produce the full
       intensity array of shape ``(K, E)``.

    Parameters
    ----------
    bands : BandStructure
        Electronic band structure containing eigenvalues of shape
        ``(K, B)`` and the Fermi energy.
    orb_proj : OrbitalProjection
        Orbital projections of shape ``(K, B, A, 9)`` where A is the
        number of atoms and 9 is the number of orbital channels
        (s, p_y, p_z, p_x, d_xy, d_yz, d_z2, d_xz, d_x2-y2).
    params : SimulationParams
        Simulation parameters including ``sigma``, ``photon_energy``,
        ``temperature``, ``energy_min``, ``energy_max``, and
        ``fidelity``.

    Returns
    -------
    spectrum : ArpesSpectrum
        Simulated ARPES intensity map with ``intensity`` of shape
        ``(K, E)`` and ``energy_axis`` of shape ``(E,)``.

    Notes
    -----
    Uses Gaussian broadening with energy-dependent heuristic orbital
    weights (p-enhanced below 50 eV, d-enhanced above). This level is
    suitable when Yeh-Lindau cross-section tables are not available but
    some orbital selectivity is desired.
    """
    energy_axis: Float[Array, " E"] = jnp.linspace(
        params.energy_min,
        params.energy_max,
        params.fidelity,
    )
    orb_w: Float[Array, " 9"] = heuristic_weights(params.photon_energy)
    proj: Float[Array, "K B A 9"] = orb_proj.projections
    weighted_proj: Float[Array, "K B A 9"] = (
        proj[..., _NON_S_ORBITAL_SLICE] * orb_w[_NON_S_ORBITAL_SLICE]
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
        contribution: Float[Array, " E"] = weight * fd * profile
        return contribution

    def _single_kpoint(
        energies: Float[Array, " B"],
        kweights: Float[Array, " B"],
    ) -> Float[Array, " E"]:
        contributions: Float[Array, "B E"] = jax.vmap(_single_band)(
            energies, kweights
        )
        total: Float[Array, " E"] = jnp.sum(contributions, axis=0)
        return total

    intensity: Float[Array, "K E"] = jax.vmap(_single_kpoint)(
        bands.eigenvalues, weights
    )
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
    """Simulate ARPES with Gaussian broadening and Yeh-Lindau cross-sections.

    Upgrades the heuristic weights of ``simulate_basic`` to physically
    motivated Yeh-Lindau photoionization cross-sections interpolated at
    the experimental photon energy. Unlike the basic level, ALL orbital
    projections (including s) are first weighted by their respective
    cross-sections before summing the non-s channels, ensuring that
    cross-section magnitudes are correctly applied before orbital
    selection.

    Implementation Logic
    --------------------
    1. **Build energy axis** via ``jnp.linspace(energy_min, energy_max,
       fidelity)`` to define the output energy grid of shape ``(E,)``.

    2. **Compute Yeh-Lindau weights** from ``photon_energy`` via
       :func:`~arpyes.simul.crosssections.yeh_lindau_weights`, returning
       interpolated photoionization cross-sections of shape ``(9,)``.

    3. **Weight ALL projections by cross-sections, then sum non-s**:
       multiply the full ``proj`` array by ``orb_w`` across all 9
       orbital channels, then apply ``slice(1, 9)`` (indices 1..8) and
       sum over orbital and atom axes to yield per-band weights of
       shape ``(K, B)``.
       This order matters: weighting before slicing ensures correct
       normalization.

    4. **Define ``_single_band``**: for one band at one k-point, compute
       ``Fermi-Dirac(E_band) * Gaussian(energy_axis, E_band, sigma)
       * weight`` to yield a spectral contribution of shape ``(E,)``.

    5. **Double vmap over k-points and bands**: same ``_single_band``
       inside ``_single_kpoint`` pattern as ``simulate_basic``, with
       ``jax.vmap`` over B then K to produce shape ``(K, E)``.

    Parameters
    ----------
    bands : BandStructure
        Electronic band structure containing eigenvalues of shape
        ``(K, B)`` and the Fermi energy.
    orb_proj : OrbitalProjection
        Orbital projections of shape ``(K, B, A, 9)`` where A is the
        number of atoms and 9 is the number of orbital channels
        (s, p_y, p_z, p_x, d_xy, d_yz, d_z2, d_xz, d_x2-y2).
    params : SimulationParams
        Simulation parameters including ``sigma``, ``photon_energy``,
        ``temperature``, ``energy_min``, ``energy_max``, and
        ``fidelity``.

    Returns
    -------
    spectrum : ArpesSpectrum
        Simulated ARPES intensity map with ``intensity`` of shape
        ``(K, E)`` and ``energy_axis`` of shape ``(E,)``.

    Notes
    -----
    Uses Gaussian broadening with interpolated Yeh-Lindau
    photoionization cross-section weights per orbital type. The
    cross-sections are computed from tabulated atomic data and
    interpolated to the specified photon energy.
    """
    energy_axis: Float[Array, " E"] = jnp.linspace(
        params.energy_min,
        params.energy_max,
        params.fidelity,
    )
    orb_w: Float[Array, " 9"] = yeh_lindau_weights(params.photon_energy)
    proj: Float[Array, "K B A 9"] = orb_proj.projections
    weighted_proj: Float[Array, "K B A 9"] = proj * orb_w
    weights: Float[Array, "K B"] = jnp.sum(
        jnp.sum(
            weighted_proj[..., _NON_S_ORBITAL_SLICE],
            axis=-1,
        ),
        axis=-1,
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
        contribution: Float[Array, " E"] = weight * fd * profile
        return contribution

    def _single_kpoint(
        energies: Float[Array, " B"],
        kweights: Float[Array, " B"],
    ) -> Float[Array, " E"]:
        contributions: Float[Array, "B E"] = jax.vmap(_single_band)(
            energies, kweights
        )
        total: Float[Array, " E"] = jnp.sum(contributions, axis=0)
        return total

    intensity: Float[Array, "K E"] = jax.vmap(_single_kpoint)(
        bands.eigenvalues, weights
    )
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
    """Simulate ARPES with Gaussian broadening and polarization rules.

    Extends ``simulate_basicplus`` by incorporating light polarization
    dependence through dipole matrix elements. The intensity for each
    orbital channel is weighted by ``|E . d_orbital|^2`` where E is the
    electric-field polarization vector and d_orbital is the dipole
    selection vector. Supports both polarized (linear, circular) and
    unpolarized light configurations.

    Implementation Logic
    --------------------
    1. **Build energy axis** via ``jnp.linspace(energy_min, energy_max,
       fidelity)`` and **compute Yeh-Lindau weights** from
       ``photon_energy``, same as ``simulate_basicplus``.

    2. **Branch on unpolarized vs polarized**:

       a. **Unpolarized**: build orthogonal polarization vectors
          ``e_s``, ``e_p`` from ``(theta, phi)`` via
          :func:`~arpyes.simul.polarization.build_polarization_vectors`.
          For each polarization vector, compute dipole matrix elements
          of shape ``(9,)`` via
          :func:`~arpyes.simul.polarization.dipole_matrix_elements`.
          Weight projections by ``orb_w * m_s`` and ``orb_w * m_p``
          respectively, sum non-s channels over orbital and atom axes
          to get ``ws_sum`` and ``wp_sum`` of shape ``(K, B)``, compute
          ``|ws_sum|^2`` and ``|wp_sum|^2``, then average:
          ``band_intensity = (i_s + i_p) / 2``.

       b. **Polarized**: build the electric field vector E via
          :func:`~arpyes.simul.polarization.build_efield`, compute
          dipole matrix elements ``m_elem`` of shape ``(9,)``, weight
          projections by ``orb_w * m_elem``, sum non-s channels, and
          compute ``band_intensity = |w_sum|^2``.

    3. **Double vmap with ``band_intensity``**: define ``_single_band``
       as ``Fermi-Dirac * Gaussian * band_intensity``, vmap over bands
       B inside ``_single_kpoint``, then vmap over k-points K to
       produce shape ``(K, E)``.

    Parameters
    ----------
    bands : BandStructure
        Electronic band structure containing eigenvalues of shape
        ``(K, B)`` and the Fermi energy.
    orb_proj : OrbitalProjection
        Orbital projections of shape ``(K, B, A, 9)`` where A is the
        number of atoms and 9 is the number of orbital channels
        (s, p_y, p_z, p_x, d_xy, d_yz, d_z2, d_xz, d_x2-y2).
    params : SimulationParams
        Simulation parameters including ``sigma``, ``photon_energy``,
        ``temperature``, ``energy_min``, ``energy_max``, and
        ``fidelity``.
    pol_config : PolarizationConfig
        Light polarization configuration specifying
        ``polarization_type`` (``"unpolarized"``, ``"linear"``, etc.),
        incidence angles ``theta`` and ``phi``, and any additional
        polarization parameters.

    Returns
    -------
    spectrum : ArpesSpectrum
        Simulated ARPES intensity map with ``intensity`` of shape
        ``(K, E)`` and ``energy_axis`` of shape ``(E,)``.

    Notes
    -----
    Uses Gaussian broadening with Yeh-Lindau cross-sections and
    polarization-dependent orbital selection via
    ``|E . d_orbital|^2`` weighting. For unpolarized light, the s- and
    p-polarization intensities are averaged, which is exact for
    incoherent superposition of orthogonal polarization states.
    """
    energy_axis: Float[Array, " E"] = jnp.linspace(
        params.energy_min,
        params.energy_max,
        params.fidelity,
    )
    orb_w: Float[Array, " 9"] = yeh_lindau_weights(params.photon_energy)
    proj: Float[Array, "K B A 9"] = orb_proj.projections
    is_unpolarized: bool = (
        pol_config.polarization_type.lower() == "unpolarized"
    )
    if is_unpolarized:
        e_s, e_p = build_polarization_vectors(pol_config.theta, pol_config.phi)
        m_s: Float[Array, " 9"] = dipole_matrix_elements(
            e_s.astype(jnp.complex128)
        )
        m_p: Float[Array, " 9"] = dipole_matrix_elements(
            e_p.astype(jnp.complex128)
        )
        w_s: Float[Array, "K B A 9"] = proj * orb_w * m_s
        w_p: Float[Array, "K B A 9"] = proj * orb_w * m_p
        ws_sum: Float[Array, "K B"] = jnp.sum(
            jnp.sum(w_s[..., _NON_S_ORBITAL_SLICE], axis=-1),
            axis=-1,
        )
        wp_sum: Float[Array, "K B"] = jnp.sum(
            jnp.sum(w_p[..., _NON_S_ORBITAL_SLICE], axis=-1),
            axis=-1,
        )
        i_s: Float[Array, "K B"] = jnp.abs(ws_sum) ** 2
        i_p: Float[Array, "K B"] = jnp.abs(wp_sum) ** 2
        band_intensity: Float[Array, "K B"] = (i_s + i_p) / 2.0
    else:
        efield = build_efield(pol_config)
        m_elem: Float[Array, " 9"] = dipole_matrix_elements(efield)
        weighted: Float[Array, "K B A 9"] = proj * orb_w * m_elem
        w_sum: Float[Array, "K B"] = jnp.sum(
            jnp.sum(
                weighted[..., _NON_S_ORBITAL_SLICE],
                axis=-1,
            ),
            axis=-1,
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
        contribution: Float[Array, " E"] = bi * fd * profile
        return contribution

    def _single_kpoint(
        energies: Float[Array, " B"],
        bi_k: Float[Array, " B"],
    ) -> Float[Array, " E"]:
        contributions: Float[Array, "B E"] = jax.vmap(_single_band)(
            energies, bi_k
        )
        total: Float[Array, " E"] = jnp.sum(contributions, axis=0)
        return total

    intensity: Float[Array, "K E"] = jax.vmap(_single_kpoint)(
        bands.eigenvalues, band_intensity
    )
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
    """Simulate ARPES with Voigt broadening and dipole matrix elements.

    The most physically complete simulation model. Combines Voigt
    broadening (capturing both instrumental Gaussian and lifetime
    Lorentzian contributions via ``sigma`` and ``gamma``), Yeh-Lindau
    photoionization cross-sections, and full polarization-dependent
    dipole matrix element weighting. This level should be used when
    quantitative comparison with experimental spectra is required.

    Implementation Logic
    --------------------
    1. **Build energy axis** and **compute Yeh-Lindau weights**, same
       as ``simulate_advanced``.

    2. **Branch on unpolarized vs polarized**: identical logic to
       ``simulate_advanced`` for computing ``band_intensity`` of shape
       ``(K, B)`` from dipole matrix elements and cross-sections.

       a. **Unpolarized**: build ``e_s``, ``e_p`` polarization vectors,
          compute dipole elements for each, weight projections by
          ``orb_w * m_s`` and ``orb_w * m_p``, sum non-s channels,
          compute ``|sum|^2`` for each, average to get
          ``band_intensity = (i_s + i_p) / 2``.

       b. **Polarized**: build E-field vector, compute dipole elements,
          weight projections, sum non-s, compute
          ``band_intensity = |w_sum|^2``.

    3. **Double vmap with ``band_intensity``**: define ``_single_band``
       using **Voigt(sigma, gamma)** instead of Gaussian. This is the
       key distinction from ``simulate_advanced``: the Voigt profile
       ``V(E; E_band, sigma, gamma)`` accounts for both instrumental
       resolution (Gaussian, width ``sigma``) and quasiparticle
       lifetime broadening (Lorentzian, width ``gamma``)
       simultaneously. Then vmap over bands B and k-points K to produce
       the full intensity of shape ``(K, E)``.

    Parameters
    ----------
    bands : BandStructure
        Electronic band structure containing eigenvalues of shape
        ``(K, B)`` and the Fermi energy.
    orb_proj : OrbitalProjection
        Orbital projections of shape ``(K, B, A, 9)`` where A is the
        number of atoms and 9 is the number of orbital channels
        (s, p_y, p_z, p_x, d_xy, d_yz, d_z2, d_xz, d_x2-y2).
    params : SimulationParams
        Simulation parameters including ``sigma``, ``gamma``,
        ``photon_energy``, ``temperature``, ``energy_min``,
        ``energy_max``, and ``fidelity``.
    pol_config : PolarizationConfig
        Light polarization configuration specifying
        ``polarization_type`` (``"unpolarized"``, ``"linear"``, etc.),
        incidence angles ``theta`` and ``phi``, and any additional
        polarization parameters.

    Returns
    -------
    spectrum : ArpesSpectrum
        Simulated ARPES intensity map with ``intensity`` of shape
        ``(K, E)`` and ``energy_axis`` of shape ``(E,)``.

    Notes
    -----
    Uses Voigt broadening with Yeh-Lindau cross-sections, polarization
    selection rules, and dipole matrix element weighting. This is the
    most physically complete model. For unpolarized light, averages s
    and p contributions. The Voigt profile is essential for accurate
    lineshape fitting where both instrumental and intrinsic broadening
    must be deconvolved.
    """
    energy_axis: Float[Array, " E"] = jnp.linspace(
        params.energy_min,
        params.energy_max,
        params.fidelity,
    )
    orb_w: Float[Array, " 9"] = yeh_lindau_weights(params.photon_energy)
    proj: Float[Array, "K B A 9"] = orb_proj.projections
    is_unpolarized: bool = (
        pol_config.polarization_type.lower() == "unpolarized"
    )
    if is_unpolarized:
        e_s, e_p = build_polarization_vectors(pol_config.theta, pol_config.phi)
        m_s: Float[Array, " 9"] = dipole_matrix_elements(
            e_s.astype(jnp.complex128)
        )
        m_p: Float[Array, " 9"] = dipole_matrix_elements(
            e_p.astype(jnp.complex128)
        )
        w_s: Float[Array, "K B A 9"] = proj * orb_w * m_s
        w_p: Float[Array, "K B A 9"] = proj * orb_w * m_p
        ws_sum: Float[Array, "K B"] = jnp.sum(
            jnp.sum(w_s[..., _NON_S_ORBITAL_SLICE], axis=-1),
            axis=-1,
        )
        wp_sum: Float[Array, "K B"] = jnp.sum(
            jnp.sum(w_p[..., _NON_S_ORBITAL_SLICE], axis=-1),
            axis=-1,
        )
        i_s: Float[Array, "K B"] = jnp.abs(ws_sum) ** 2
        i_p: Float[Array, "K B"] = jnp.abs(wp_sum) ** 2
        band_intensity: Float[Array, "K B"] = (i_s + i_p) / 2.0
    else:
        efield = build_efield(pol_config)
        m_elem: Float[Array, " 9"] = dipole_matrix_elements(efield)
        weighted: Float[Array, "K B A 9"] = proj * orb_w * m_elem
        w_sum: Float[Array, "K B"] = jnp.sum(
            jnp.sum(
                weighted[..., _NON_S_ORBITAL_SLICE],
                axis=-1,
            ),
            axis=-1,
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
        contribution: Float[Array, " E"] = bi * fd * profile
        return contribution

    def _single_kpoint(
        energies: Float[Array, " B"],
        bi_k: Float[Array, " B"],
    ) -> Float[Array, " E"]:
        contributions: Float[Array, "B E"] = jax.vmap(_single_band)(
            energies, bi_k
        )
        total: Float[Array, " E"] = jnp.sum(contributions, axis=0)
        return total

    intensity: Float[Array, "K E"] = jax.vmap(_single_kpoint)(
        bands.eigenvalues, band_intensity
    )
    spectrum: ArpesSpectrum = make_arpes_spectrum(
        intensity=intensity,
        energy_axis=energy_axis,
    )
    return spectrum


@jaxtyped(typechecker=beartype)
def simulate_soc(
    bands: BandStructure,
    orb_proj: SpinOrbitalProjection,
    params: SimulationParams,
    pol_config: PolarizationConfig,
    ls_scale: ScalarFloat = 0.01,
) -> ArpesSpectrum:
    """Simulate ARPES with spin-orbit coupling (spin-dependent intensity).

    Extends the expert model with a spin-orbit correction: the
    orbital-derived band intensity is modulated by the spin projection
    along the photon wavevector, enabling spin-ARPES and circular
    dichroism. Requires ``orb_proj.spin`` of shape ``(K, B, A, 6)``
    (spin up/down for x, y, z). Uses Voigt broadening and the same
    Yeh-Lindau and polarization logic as ``simulate_expert``.

    Implementation Logic
    --------------------
    1. **Orbital intensity**: Compute ``band_intensity`` exactly as in
       ``simulate_expert`` (Yeh-Lindau, dipole matrix elements,
       unpolarized or polarized branch).
    2. **Spin vector**: From ``spin`` (K, B, A, 6) form (Sx, Sy, Sz) per
       (k, band, atom): Sx = spin[...,0]+spin[...,1], Sy = spin[...,2]+
       spin[...,3], Sz = spin[...,4]+spin[...,5]. Sum over atoms to get
       spin per (k, band) of shape (K, B, 3).
    3. **SOC correction**: k_photon = photon_wavevector(theta, phi).
       spin_dot_k = (K, B). band_intensity_soc = band_intensity *
       (1 + ls_scale * spin_dot_k). Then Voigt convolution and vmap
       as in expert.

    Parameters
    ----------
    bands : BandStructure
        Band structure with eigenvalues and Fermi energy.
    orb_proj : SpinOrbitalProjection
        Orbital projections with mandatory spin data
        (shape ``(K, B, A, 6)``).
    params : SimulationParams
        Simulation parameters (sigma, gamma, fidelity, etc.).
    pol_config : PolarizationConfig
        Light polarization and incidence angles.
    ls_scale : ScalarFloat, optional
        Spin-orbit coupling strength for the S·k_photon correction.
        Default is 0.01.

    Returns
    -------
    spectrum : ArpesSpectrum
        Simulated ARPES intensity and energy axis.

    Notes
    -----
    The spin array has six channels (up/down for x, y, z). The net
    spin vector (Sx, Sy, Sz) per (k, band, atom) is formed by
    summing the two components for each axis, then summed over
    atoms to get a per-band spin used in the modulation
    (1 + ls_scale * S·k_photon). With ``ls_scale=0`` the result
    coincides with ``simulate_expert``.

    See Also
    --------
    simulate_expert : Same physics without spin correction.
    photon_wavevector : Builds k_photon from incidence angles.
    """
    energy_axis: Float[Array, " E"] = jnp.linspace(
        params.energy_min,
        params.energy_max,
        params.fidelity,
    )
    orb_w: Float[Array, " 9"] = yeh_lindau_weights(params.photon_energy)
    proj: Float[Array, "K B A 9"] = orb_proj.projections
    spin_raw: Float[Array, "K B A 6"] = orb_proj.spin
    is_unpolarized: bool = (
        pol_config.polarization_type.lower() == "unpolarized"
    )
    if is_unpolarized:
        e_s, e_p = build_polarization_vectors(pol_config.theta, pol_config.phi)
        m_s: Float[Array, " 9"] = dipole_matrix_elements(
            e_s.astype(jnp.complex128)
        )
        m_p: Float[Array, " 9"] = dipole_matrix_elements(
            e_p.astype(jnp.complex128)
        )
        w_s: Float[Array, "K B A 9"] = proj * orb_w * m_s
        w_p: Float[Array, "K B A 9"] = proj * orb_w * m_p
        ws_sum: Float[Array, "K B"] = jnp.sum(
            jnp.sum(w_s[..., _NON_S_ORBITAL_SLICE], axis=-1),
            axis=-1,
        )
        wp_sum: Float[Array, "K B"] = jnp.sum(
            jnp.sum(w_p[..., _NON_S_ORBITAL_SLICE], axis=-1),
            axis=-1,
        )
        i_s: Float[Array, "K B"] = jnp.abs(ws_sum) ** 2
        i_p: Float[Array, "K B"] = jnp.abs(wp_sum) ** 2
        band_intensity: Float[Array, "K B"] = (i_s + i_p) / 2.0
    else:
        efield = build_efield(pol_config)
        m_elem: Float[Array, " 9"] = dipole_matrix_elements(efield)
        weighted: Float[Array, "K B A 9"] = proj * orb_w * m_elem
        w_sum: Float[Array, "K B"] = jnp.sum(
            jnp.sum(
                weighted[..., _NON_S_ORBITAL_SLICE],
                axis=-1,
            ),
            axis=-1,
        )
        band_intensity = jnp.abs(w_sum) ** 2

    k_photon: Float[Array, " 3"] = photon_wavevector(
        pol_config.theta, pol_config.phi
    )
    spin_vec: Float[Array, "K B A 3"] = jnp.stack(
        [
            spin_raw[..., 0] + spin_raw[..., 1],
            spin_raw[..., 2] + spin_raw[..., 3],
            spin_raw[..., 4] + spin_raw[..., 5],
        ],
        axis=-1,
    )
    spin_per_band: Float[Array, "K B 3"] = jnp.sum(spin_vec, axis=-2)
    spin_dot_k: Float[Array, "K B"] = jnp.dot(spin_per_band, k_photon)
    ls_arr: Float[Array, " "] = jnp.asarray(ls_scale, dtype=jnp.float64)
    band_intensity_soc: Float[Array, "K B"] = band_intensity * (
        1.0 + ls_arr * spin_dot_k
    )

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
        contribution: Float[Array, " E"] = bi * fd * profile
        return contribution

    def _single_kpoint(
        energies: Float[Array, " B"],
        bi_k: Float[Array, " B"],
    ) -> Float[Array, " E"]:
        contributions: Float[Array, "B E"] = jax.vmap(_single_band)(
            energies, bi_k
        )
        total: Float[Array, " E"] = jnp.sum(contributions, axis=0)
        return total

    intensity: Float[Array, "K E"] = jax.vmap(_single_kpoint)(
        bands.eigenvalues, band_intensity_soc
    )
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
    "simulate_soc",
]
