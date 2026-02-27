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
from beartype.typing import Optional, Tuple
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
    simulate_soc,
)


@jaxtyped(typechecker=beartype)
def make_expanded_simulation_params(
    eigenbands: Float[Array, "K B"],
    fidelity: int = 25000,
    sigma: ScalarFloat = 0.04,
    gamma: ScalarFloat = 0.1,
    temperature: ScalarFloat = 15.0,
    photon_energy: ScalarFloat = 11.0,
    energy_padding: ScalarFloat = 1.0,
) -> SimulationParams:
    """Build simulation parameters with auto-derived energy window.

    Constructs a :class:`~arpyes.types.SimulationParams` PyTree whose
    energy window is derived from the actual band energies rather than
    from fixed defaults.  The window spans
    ``[min(eigenbands) - energy_padding, max(eigenbands) + energy_padding]``,
    ensuring every band falls within the simulated range.

    Implementation Logic
    --------------------
    1. **Cast to float64**: ``eigenbands`` is promoted to
       ``jnp.float64`` via ``jnp.asarray``.
    2. **Derive energy bounds**: ``energy_min`` and ``energy_max``
       are computed from the global min/max of the band array plus
       symmetric padding controlled by ``energy_padding``.
    3. **Delegate**: Passes all values to
       :func:`~arpyes.types.make_simulation_params` for final
       construction and type validation.

    Parameters
    ----------
    eigenbands : Float[Array, "K B"]
        Band eigenvalues in eV, shape ``(n_kpoints, n_bands)``.
        Used only to derive ``energy_min`` and ``energy_max``.
    fidelity : int, optional
        Number of points in the energy axis. Default is 25000.
    sigma : ScalarFloat, optional
        Gaussian broadening width in eV. Default is 0.04.
    gamma : ScalarFloat, optional
        Lorentzian broadening width in eV. Default is 0.1.
    temperature : ScalarFloat, optional
        Electronic temperature in Kelvin. Default is 15.
    photon_energy : ScalarFloat, optional
        Incident photon energy in eV. Default is 11.
    energy_padding : ScalarFloat, optional
        Symmetric padding around band extrema in eV. Default is 1.

    Returns
    -------
    params : SimulationParams
        Simulation parameters with data-derived energy window.

    See Also
    --------
    make_simulation_params : Lower-level factory with explicit
        energy bounds.
    """
    bands_arr: Float[Array, "K B"] = jnp.asarray(
        eigenbands, dtype=jnp.float64
    )
    pad: Float[Array, " "] = jnp.asarray(
        energy_padding, dtype=jnp.float64
    )
    energy_min: Float[Array, " "] = jnp.min(bands_arr) - pad
    energy_max: Float[Array, " "] = jnp.max(bands_arr) + pad
    params: SimulationParams = make_simulation_params(
        energy_min=energy_min,
        energy_max=energy_max,
        fidelity=fidelity,
        sigma=sigma,
        gamma=gamma,
        temperature=temperature,
        photon_energy=photon_energy,
    )
    return params


@jaxtyped(typechecker=beartype)
def _build_inputs(
    eigenbands: Float[Array, "K B"],
    surface_orb: Float[Array, "K B A 9"],
    ef: ScalarFloat,
    surface_spin: Optional[Float[Array, "K B A 6"]] = None,
) -> Tuple[BandStructure, OrbitalProjection]:
    """Convert plain arrays into core ARPES input PyTrees.

    Wraps raw eigenenergy and orbital-projection arrays into the
    :class:`~arpyes.types.BandStructure` and
    :class:`~arpyes.types.OrbitalProjection` PyTrees expected by
    the low-level simulation functions.

    Implementation Logic
    --------------------
    1. **Cast to float64**: ``eigenbands`` and ``surface_orb`` are
       promoted to ``jnp.float64`` via ``jnp.asarray`` so that all
       downstream arithmetic operates at double precision.

    2. **Synthesize k-point coordinates**: A placeholder array of
       zeros with shape ``(K, 3)`` is created. The expanded-input
       workflow does not require physical k-point coordinates because
       the simulation kernels only use the eigenvalues; the k-point
       axis merely indexes the bands.

    3. **Build PyTrees**: ``make_band_structure`` and
       ``make_orbital_projection`` assemble the validated PyTrees.
       If ``surface_spin`` is provided (shape ``(K, B, A, 6)``), it
       is passed to ``make_orbital_projection`` for SOC simulations.

    Parameters
    ----------
    eigenbands : Float[Array, "K B"]
        Band eigenvalues in eV, with shape ``(n_kpoints, n_bands)``.
    surface_orb : Float[Array, "K B A 9"]
        Orbital projection coefficients, with shape
        ``(n_kpoints, n_bands, n_atoms, 9)``. The last axis follows
        the order ``(s, py, pz, px, dxy, dyz, dz2, dxz, dx2-y2)``.
    ef : ScalarFloat
        Fermi energy in eV.
    surface_spin : Optional[Float[Array, "K B A 6"]], optional
        Spin projections (up/down for x, y, z). Required for SOC
        level; omit for other levels.

    Returns
    -------
    bands : BandStructure
        Band-structure PyTree containing eigenvalues, placeholder
        k-points, and the Fermi energy.
    orb_proj : OrbitalProjection
        Orbital-projection PyTree wrapping ``surface_orb`` (and
        ``surface_spin`` when provided).
    """
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
    orb_proj: OrbitalProjection = make_orbital_projection(
        projections=proj_arr,
        spin=surface_spin,
    )
    return bands, orb_proj


@jaxtyped(typechecker=beartype)
def _build_polarization(
    polarization: str = "unpolarized",
    incident_theta: ScalarFloat = 45.0,
    incident_phi: ScalarFloat = 0.0,
    polarization_angle: ScalarFloat = 0.0,
) -> PolarizationConfig:
    """Create polarization config with degree-to-radian conversion.

    The expanded API accepts incident angles in **degrees** (more
    intuitive for users) while the internal
    :class:`~arpyes.types.PolarizationConfig` stores them in
    radians.  This helper performs the conversion.

    Implementation Logic
    --------------------
    1. **Convert angles**: ``incident_theta`` and ``incident_phi``
       are converted from degrees to radians via ``jnp.deg2rad``.
    2. **Pass through**: ``polarization_angle`` is already in
       radians and is forwarded without conversion.
    3. **Delegate**: Calls :func:`~arpyes.types.make_polarization_config`
       to construct the validated PyTree.

    Parameters
    ----------
    polarization : str, optional
        Polarization type: ``"s"``, ``"p"``, ``"linear"``, or
        ``"unpolarized"`` (default).
    incident_theta : ScalarFloat, optional
        Polar angle of the incident beam in **degrees**.
        Default is 45.
    incident_phi : ScalarFloat, optional
        Azimuthal angle of the incident beam in **degrees**.
        Default is 0.
    polarization_angle : ScalarFloat, optional
        Rotation angle for arbitrary linear polarization in
        **radians**. Default is 0.

    Returns
    -------
    config : PolarizationConfig
        Polarization configuration with angles in radians.

    See Also
    --------
    make_polarization_config : Lower-level factory accepting
        radians directly.
    """
    theta: Float[Array, " "] = jnp.deg2rad(
        jnp.asarray(incident_theta, dtype=jnp.float64)
    )
    phi: Float[Array, " "] = jnp.deg2rad(
        jnp.asarray(incident_phi, dtype=jnp.float64)
    )
    pol_ang: Float[Array, " "] = jnp.asarray(
        polarization_angle, dtype=jnp.float64
    )
    config: PolarizationConfig = make_polarization_config(
        theta=theta,
        phi=phi,
        polarization_angle=pol_ang,
        polarization_type=polarization,
    )
    return config


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
    """Run novice-level ARPES simulation from plain arrays.

    Simplest physical model: applies Voigt broadening (combined
    Gaussian + Lorentzian) with uniform orbital weights. All
    non-s orbitals contribute equally to the photoemission
    intensity.

    Implementation Logic
    --------------------
    1. **Build PyTrees**: Calls :func:`_build_inputs` to wrap the
       raw arrays into ``BandStructure`` and ``OrbitalProjection``.
    2. **Build parameters**: Calls ``make_simulation_params`` to
       construct a ``SimulationParams`` PyTree from the scalar
       broadening / fidelity / temperature / photon-energy values.
    3. **Simulate**: Delegates to :func:`~arpyes.simul.simulate_novice`,
       which vectorizes the Voigt convolution over all k-points and
       bands with ``jax.vmap``.

    Parameters
    ----------
    eigenbands : Float[Array, "K B"]
        Band eigenvalues in eV, shape ``(n_kpoints, n_bands)``.
    surface_orb : Float[Array, "K B A 9"]
        Orbital projection coefficients, shape
        ``(n_kpoints, n_bands, n_atoms, 9)``.
    ef : ScalarFloat
        Fermi energy in eV.
    sigma : ScalarFloat
        Gaussian broadening width in eV.
    gamma : ScalarFloat
        Lorentzian broadening width in eV.
    fidelity : int
        Number of points in the energy axis.
    temperature : ScalarFloat
        Electronic temperature in Kelvin for the Fermi-Dirac
        distribution.
    photon_energy : ScalarFloat
        Incident photon energy in eV.

    Returns
    -------
    spectrum : ArpesSpectrum
        Simulated ARPES spectrum containing the intensity map and
        energy axis.

    See Also
    --------
    simulate_novice : Low-level implementation accepting PyTrees.
    """
    bands: BandStructure
    orb_proj: OrbitalProjection
    bands, orb_proj = _build_inputs(
        eigenbands=eigenbands,
        surface_orb=surface_orb,
        ef=ef,
    )
    params: SimulationParams = make_expanded_simulation_params(
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
    """Run basic-level ARPES simulation from plain arrays.

    Applies Gaussian broadening with energy-dependent heuristic
    orbital weights. The heuristic enhances p-orbital contributions
    below 50 eV photon energy and d-orbital contributions above,
    providing a rough approximation of photoionization cross-section
    effects without tabulated data.

    Implementation Logic
    --------------------
    1. **Build PyTrees**: Calls :func:`_build_inputs` to wrap the
       raw arrays into ``BandStructure`` and ``OrbitalProjection``.
    2. **Build parameters**: Calls ``make_simulation_params`` to
       construct a ``SimulationParams`` PyTree. No ``gamma`` is
       needed because this level uses pure Gaussian broadening.
    3. **Simulate**: Delegates to :func:`~arpyes.simul.simulate_basic`,
       which computes heuristic weights from ``photon_energy`` and
       vectorizes the Gaussian convolution with ``jax.vmap``.

    Parameters
    ----------
    eigenbands : Float[Array, "K B"]
        Band eigenvalues in eV, shape ``(n_kpoints, n_bands)``.
    surface_orb : Float[Array, "K B A 9"]
        Orbital projection coefficients, shape
        ``(n_kpoints, n_bands, n_atoms, 9)``.
    ef : ScalarFloat
        Fermi energy in eV.
    sigma : ScalarFloat
        Gaussian broadening width in eV.
    fidelity : int
        Number of points in the energy axis.
    temperature : ScalarFloat
        Electronic temperature in Kelvin for the Fermi-Dirac
        distribution.
    photon_energy : ScalarFloat
        Incident photon energy in eV. Determines the heuristic
        orbital weighting regime.

    Returns
    -------
    spectrum : ArpesSpectrum
        Simulated ARPES spectrum containing the intensity map and
        energy axis.

    See Also
    --------
    simulate_basic : Low-level implementation accepting PyTrees.
    """
    bands: BandStructure
    orb_proj: OrbitalProjection
    bands, orb_proj = _build_inputs(
        eigenbands=eigenbands,
        surface_orb=surface_orb,
        ef=ef,
    )
    params: SimulationParams = make_expanded_simulation_params(
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
    """Run basicplus-level ARPES simulation from plain arrays.

    Applies Gaussian broadening with interpolated Yeh-Lindau
    photoionization cross-sections. Unlike the heuristic weights
    used at the basic level, Yeh-Lindau cross-sections are derived
    from tabulated atomic data and provide physically accurate
    orbital-dependent intensity scaling at each photon energy.

    Implementation Logic
    --------------------
    1. **Build PyTrees**: Calls :func:`_build_inputs` to wrap the
       raw arrays into ``BandStructure`` and ``OrbitalProjection``.
    2. **Build parameters**: Calls ``make_simulation_params`` to
       construct a ``SimulationParams`` PyTree. No ``gamma`` is
       needed because this level uses pure Gaussian broadening.
    3. **Simulate**: Delegates to
       :func:`~arpyes.simul.simulate_basicplus`, which interpolates
       Yeh-Lindau cross-section weights from ``photon_energy`` and
       vectorizes the Gaussian convolution with ``jax.vmap``.

    Parameters
    ----------
    eigenbands : Float[Array, "K B"]
        Band eigenvalues in eV, shape ``(n_kpoints, n_bands)``.
    surface_orb : Float[Array, "K B A 9"]
        Orbital projection coefficients, shape
        ``(n_kpoints, n_bands, n_atoms, 9)``.
    ef : ScalarFloat
        Fermi energy in eV.
    sigma : ScalarFloat
        Gaussian broadening width in eV.
    fidelity : int
        Number of points in the energy axis.
    temperature : ScalarFloat
        Electronic temperature in Kelvin for the Fermi-Dirac
        distribution.
    photon_energy : ScalarFloat
        Incident photon energy in eV. Used to interpolate
        Yeh-Lindau cross-section tables.

    Returns
    -------
    spectrum : ArpesSpectrum
        Simulated ARPES spectrum containing the intensity map and
        energy axis.

    See Also
    --------
    simulate_basicplus : Low-level implementation accepting PyTrees.
    """
    bands: BandStructure
    orb_proj: OrbitalProjection
    bands, orb_proj = _build_inputs(
        eigenbands=eigenbands,
        surface_orb=surface_orb,
        ef=ef,
    )
    params: SimulationParams = make_expanded_simulation_params(
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
    """Run advanced-level ARPES simulation from plain arrays.

    Builds on the basicplus level by adding polarization-dependent
    selection rules. The photoemission intensity is weighted by
    |e . d|^2, where e is the light electric-field vector and d is
    the dipole matrix element for each orbital channel. For
    unpolarized light the s- and p-polarization contributions are
    averaged.

    Implementation Logic
    --------------------
    1. **Build PyTrees**: Calls :func:`_build_inputs` to wrap the
       raw arrays into ``BandStructure`` and ``OrbitalProjection``.
    2. **Build parameters**: Calls ``make_simulation_params`` to
       construct a ``SimulationParams`` PyTree. No ``gamma`` is
       needed because this level uses pure Gaussian broadening.
    3. **Build polarization**: Calls ``make_polarization_config`` to
       construct a ``PolarizationConfig`` PyTree from the incident
       angles and polarization type.
    4. **Simulate**: Delegates to
       :func:`~arpyes.simul.simulate_advanced`, which computes
       Yeh-Lindau weights, builds the electric-field vector, and
       evaluates polarization-weighted Gaussian spectra via
       ``jax.vmap``.

    Parameters
    ----------
    eigenbands : Float[Array, "K B"]
        Band eigenvalues in eV, shape ``(n_kpoints, n_bands)``.
    surface_orb : Float[Array, "K B A 9"]
        Orbital projection coefficients, shape
        ``(n_kpoints, n_bands, n_atoms, 9)``.
    ef : ScalarFloat
        Fermi energy in eV.
    sigma : ScalarFloat
        Gaussian broadening width in eV.
    fidelity : int
        Number of points in the energy axis.
    temperature : ScalarFloat
        Electronic temperature in Kelvin for the Fermi-Dirac
        distribution.
    photon_energy : ScalarFloat
        Incident photon energy in eV.
    polarization : str, optional
        Polarization type: ``"s"``, ``"p"``, ``"linear"``, or
        ``"unpolarized"`` (default).
    incident_theta : ScalarFloat, optional
        Polar angle of the incident beam in degrees. Default 45.
    incident_phi : ScalarFloat, optional
        Azimuthal angle of the incident beam in degrees. Default 0.
    polarization_angle : ScalarFloat, optional
        Rotation angle for arbitrary linear polarization in
        radians. Default 0.

    Returns
    -------
    spectrum : ArpesSpectrum
        Simulated ARPES spectrum containing the intensity map and
        energy axis.

    See Also
    --------
    simulate_advanced : Low-level implementation accepting PyTrees.
    """
    bands: BandStructure
    orb_proj: OrbitalProjection
    bands, orb_proj = _build_inputs(
        eigenbands=eigenbands,
        surface_orb=surface_orb,
        ef=ef,
    )
    params: SimulationParams = make_expanded_simulation_params(
        eigenbands=eigenbands,
        fidelity=fidelity,
        sigma=sigma,
        temperature=temperature,
        photon_energy=photon_energy,
    )
    pol: PolarizationConfig = _build_polarization(
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
    """Run expert-level ARPES simulation from plain arrays.

    Most physically complete model. Combines Voigt broadening
    (Gaussian + Lorentzian), Yeh-Lindau photoionization
    cross-sections, polarization selection rules, and full dipole
    matrix element weighting. For unpolarized light the s- and
    p-polarization contributions are averaged.

    Implementation Logic
    --------------------
    1. **Build PyTrees**: Calls :func:`_build_inputs` to wrap the
       raw arrays into ``BandStructure`` and ``OrbitalProjection``.
    2. **Build parameters**: Calls ``make_simulation_params`` to
       construct a ``SimulationParams`` PyTree. Both ``sigma`` and
       ``gamma`` are required because this level uses Voigt
       (Gaussian + Lorentzian) broadening.
    3. **Build polarization**: Calls ``make_polarization_config`` to
       construct a ``PolarizationConfig`` PyTree from the incident
       angles and polarization type.
    4. **Simulate**: Delegates to
       :func:`~arpyes.simul.simulate_expert`, which computes
       Yeh-Lindau weights, builds the electric-field vector,
       evaluates dipole matrix elements, and produces
       polarization-weighted Voigt spectra via ``jax.vmap``.

    Parameters
    ----------
    eigenbands : Float[Array, "K B"]
        Band eigenvalues in eV, shape ``(n_kpoints, n_bands)``.
    surface_orb : Float[Array, "K B A 9"]
        Orbital projection coefficients, shape
        ``(n_kpoints, n_bands, n_atoms, 9)``.
    ef : ScalarFloat
        Fermi energy in eV.
    sigma : ScalarFloat
        Gaussian broadening width in eV.
    gamma : ScalarFloat
        Lorentzian broadening width in eV.
    fidelity : int
        Number of points in the energy axis.
    temperature : ScalarFloat
        Electronic temperature in Kelvin for the Fermi-Dirac
        distribution.
    photon_energy : ScalarFloat
        Incident photon energy in eV.
    polarization : str, optional
        Polarization type: ``"s"``, ``"p"``, ``"linear"``, or
        ``"unpolarized"`` (default).
    incident_theta : ScalarFloat, optional
        Polar angle of the incident beam in degrees. Default 45.
    incident_phi : ScalarFloat, optional
        Azimuthal angle of the incident beam in degrees. Default 0.
    polarization_angle : ScalarFloat, optional
        Rotation angle for arbitrary linear polarization in
        radians. Default 0.

    Returns
    -------
    spectrum : ArpesSpectrum
        Simulated ARPES spectrum containing the intensity map and
        energy axis.

    See Also
    --------
    simulate_expert : Low-level implementation accepting PyTrees.
    """
    bands: BandStructure
    orb_proj: OrbitalProjection
    bands, orb_proj = _build_inputs(
        eigenbands=eigenbands,
        surface_orb=surface_orb,
        ef=ef,
    )
    params: SimulationParams = make_expanded_simulation_params(
        eigenbands=eigenbands,
        fidelity=fidelity,
        sigma=sigma,
        gamma=gamma,
        temperature=temperature,
        photon_energy=photon_energy,
    )
    pol: PolarizationConfig = _build_polarization(
        polarization=polarization,
        incident_theta=incident_theta,
        incident_phi=incident_phi,
        polarization_angle=polarization_angle,
    )
    spectrum: ArpesSpectrum = simulate_expert(bands, orb_proj, params, pol)
    return spectrum


@jaxtyped(typechecker=beartype)
def simulate_soc_expanded(  # noqa: PLR0913
    eigenbands: Float[Array, "K B"],
    surface_orb: Float[Array, "K B A 9"],
    surface_spin: Float[Array, "K B A 6"],
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
    ls_scale: ScalarFloat = 0.01,
) -> ArpesSpectrum:
    """Run SOC (spin-orbit coupling) ARPES simulation from plain arrays.

    Expert model plus spin-dependent intensity correction
    (S·k_photon). Requires spin projections of shape
    ``(n_kpoints, n_bands, n_atoms, 6)`` (up/down for x, y, z).
    Incident angles are interpreted in degrees.

    Implementation Logic
    --------------------
    1. **Build PyTrees**: Calls :func:`_build_inputs` with
       ``surface_spin`` so that ``OrbitalProjection`` has non-None
       spin.
    2. **Build params and polarization**: Same as
       :func:`simulate_expert_expanded` (auto-derived energy window,
       degree-to-radian conversion for angles).
    3. **Simulate**: Delegates to :func:`~arpyes.simul.simulate_soc`
       with the given ``ls_scale``.

    Parameters
    ----------
    eigenbands : Float[Array, "K B"]
        Band eigenvalues in eV, shape ``(n_kpoints, n_bands)``.
    surface_orb : Float[Array, "K B A 9"]
        Orbital projection coefficients, shape
        ``(n_kpoints, n_bands, n_atoms, 9)``.
    surface_spin : Float[Array, "K B A 6"]
        Spin projections (required for SOC), shape
        ``(n_kpoints, n_bands, n_atoms, 6)`` with channels
        (x up, x down, y up, y down, z up, z down).
    ef : ScalarFloat
        Fermi energy in eV.
    sigma : ScalarFloat
        Gaussian broadening width in eV.
    gamma : ScalarFloat
        Lorentzian broadening width in eV.
    fidelity : int
        Number of points on the energy axis.
    temperature : ScalarFloat
        Electronic temperature in Kelvin.
    photon_energy : ScalarFloat
        Incident photon energy in eV.
    polarization : str, optional
        Polarization type (e.g. ``"unpolarized"``, ``"LHP"``).
        Default is ``"unpolarized"``.
    incident_theta : ScalarFloat, optional
        Polar angle of the incident beam in degrees. Default 45.
    incident_phi : ScalarFloat, optional
        Azimuthal angle of the incident beam in degrees. Default 0.
    polarization_angle : ScalarFloat, optional
        Rotation angle for arbitrary linear polarization in
        radians. Default 0.
    ls_scale : ScalarFloat, optional
        Spin-orbit coupling strength for the S·k_photon
        correction. Default 0.01.

    Returns
    -------
    spectrum : ArpesSpectrum
        Simulated ARPES spectrum with spin-orbit correction;
        ``intensity`` shape ``(K, E)``, ``energy_axis`` shape ``(E,)``.

    See Also
    --------
    simulate_soc : Low-level implementation accepting PyTrees.
    simulate_expert_expanded : Same pipeline without spin data.
    """
    bands: BandStructure
    orb_proj: OrbitalProjection
    bands, orb_proj = _build_inputs(
        eigenbands=eigenbands,
        surface_orb=surface_orb,
        ef=ef,
        surface_spin=surface_spin,
    )
    params: SimulationParams = make_expanded_simulation_params(
        eigenbands=eigenbands,
        fidelity=fidelity,
        sigma=sigma,
        gamma=gamma,
        temperature=temperature,
        photon_energy=photon_energy,
    )
    pol: PolarizationConfig = _build_polarization(
        polarization=polarization,
        incident_theta=incident_theta,
        incident_phi=incident_phi,
        polarization_angle=polarization_angle,
    )
    spectrum: ArpesSpectrum = simulate_soc(
        bands, orb_proj, params, pol, ls_scale=ls_scale
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
    photon_energy: ScalarFloat = 11.0,
    polarization: str = "unpolarized",
    incident_theta: ScalarFloat = 45.0,
    incident_phi: ScalarFloat = 0.0,
    polarization_angle: ScalarFloat = 0.0,
    surface_spin: Optional[Float[Array, "K B A 6"]] = None,
    ls_scale: ScalarFloat = 0.01,
) -> ArpesSpectrum:
    """Dispatch an expanded-input simulation by complexity level.

    Single entry-point that routes to one of the six simulation
    functions based on ``level``. All parameters have sensible
    defaults, so only ``level``, ``eigenbands``, and ``surface_orb``
    are required. Parameters that are unused by the selected level
    (e.g. ``gamma`` for basic/basicplus/advanced, or polarization
    settings for novice/basic/basicplus) are silently ignored.

    Implementation Logic
    --------------------
    1. **Normalize level**: ``level`` is lowered to a canonical key.
    2. **Dispatch**: An ``if``-chain selects the matching
       ``simulate_*_expanded`` function and forwards only the
       parameters that function accepts.
    3. **Error on unknown level**: Raises ``ValueError`` with the
       list of valid levels.

    Parameters
    ----------
    level : str
        One of ``"novice"``, ``"basic"``, ``"basicplus"``,
        ``"advanced"``, ``"expert"``, or ``"soc"`` (case-insensitive).
    eigenbands : Float[Array, "K B"]
        Band eigenvalues in eV, shape ``(n_kpoints, n_bands)``.
    surface_orb : Float[Array, "K B A 9"]
        Orbital projection coefficients, shape
        ``(n_kpoints, n_bands, n_atoms, 9)``.
    ef : ScalarFloat, optional
        Fermi energy in eV. Default is 0.
    sigma : ScalarFloat, optional
        Gaussian broadening width in eV. Default is 0.04.
    gamma : ScalarFloat, optional
        Lorentzian broadening width in eV. Only used by novice
        and expert levels. Default is 0.1.
    fidelity : int, optional
        Number of points in the energy axis. Default is 25000.
    temperature : ScalarFloat, optional
        Electronic temperature in Kelvin. Default is 15.
    photon_energy : ScalarFloat, optional
        Incident photon energy in eV. Default is 11.
    polarization : str, optional
        Polarization type: ``"s"``, ``"p"``, ``"linear"``, or
        ``"unpolarized"``. Only used by advanced and expert.
    incident_theta : ScalarFloat, optional
        Polar angle of the incident beam in degrees. Only used
        by advanced and expert. Default is 45.
    incident_phi : ScalarFloat, optional
        Azimuthal angle of the incident beam in degrees. Only
        used by advanced and expert. Default is 0.
    polarization_angle : ScalarFloat, optional
        Rotation angle for arbitrary linear polarization in
        radians. Only used by advanced and expert. Default is 0.
    surface_spin : Optional[Float[Array, "K B A 6"]], optional
        Spin projections; required when ``level='soc'``. Default None.
    ls_scale : ScalarFloat, optional
        Spin-orbit coupling strength when ``level='soc'``.
        Default 0.01.

    Returns
    -------
    spectrum : ArpesSpectrum
        Simulated ARPES spectrum containing the intensity map and
        energy axis.

    Raises
    ------
    ValueError
        If ``level`` is not one of the six recognized levels, or
        if ``level='soc'`` and ``surface_spin`` is None.

    See Also
    --------
    simulate_novice_expanded : Voigt broadening, uniform weights.
    simulate_basic_expanded : Gaussian, heuristic weights.
    simulate_basicplus_expanded : Gaussian, Yeh-Lindau weights.
    simulate_advanced_expanded : Gaussian, Yeh-Lindau, polarization.
    simulate_expert_expanded : Voigt, Yeh-Lindau, polarization,
        dipole matrix elements.
    simulate_soc_expanded : Expert plus spin-orbit (S·k_photon) correction.
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
    if level_key == "soc":
        if surface_spin is None:
            raise ValueError(
                "simulate_expanded(level='soc', ...) requires surface_spin."
            )
        return simulate_soc_expanded(
            eigenbands=eigenbands,
            surface_orb=surface_orb,
            surface_spin=surface_spin,
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
            ls_scale=ls_scale,
        )
    msg: str = (
        "Unknown simulation level. "
        "Expected one of: novice, basic, basicplus, advanced, expert, soc."
    )
    raise ValueError(msg)


__all__: list[str] = [
    "make_expanded_simulation_params",
    "_build_polarization",
    "_build_inputs",
    "simulate_advanced_expanded",
    "simulate_basic_expanded",
    "simulate_basicplus_expanded",
    "simulate_novice_expanded",
    "simulate_expert_expanded",
    "simulate_soc_expanded",
    "simulate_expanded",
]
