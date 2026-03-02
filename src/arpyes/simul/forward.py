r"""End-to-end differentiable ARPES forward model.

Extended Summary
----------------
Implements ``simulate_tb_radial``, the fully differentiable ARPES
simulation function that computes dipole matrix elements from first
principles using radial integrals, Gaunt coefficients, and real
spherical harmonics. This is the "Chinook-style" tight-binding
forward model, distinct from the VASP-projection-based ``simulate_*``
functions in ``spectrum.py``.

The full simulation pipeline is:

    diag_bands --> eigenvectors --> per-orbital M(k) -->
    total |M|^2 x Fermi-Dirac x Voigt --> I(k, E)

Every stage is JAX-traceable, enabling ``jax.grad`` with respect to
Slater exponents (controlling radial wavefunction extent),
eigenvectors/eigenvalues, simulation parameters (sigma, gamma,
temperature), polarization angles, and work function. Optional
energy-dependent self-energy and momentum broadening are supported.

Routine Listings
----------------
:func:`simulate_tb_radial`
    End-to-end differentiable ARPES forward model.

Notes
-----
Physical constants used in this module:

- ``_HBAR_EV_S``: hbar in eV*s (6.582e-16)
- ``_ME_EV``: electron mass in eV/c^2 (0.511e6)
- ``_HBAR_C_EV_A``: hbar*c in eV*Angstrom (1973.27)
- ``_BOHR_TO_ANGSTROM``: Bohr radius in Angstroms (0.5292)

The private helper ``_ekin_to_k_magnitude`` converts photon
energy and binding energy to a photoelectron wavevector magnitude
using the free-electron final-state approximation.
"""

import jax
import jax.numpy as jnp
from beartype import beartype
from beartype.typing import Optional
from jaxtyping import Array, Complex, Float, jaxtyped

from arpyes.maths.dipole import dipole_matrix_element_single
from arpyes.radial import slater_radial
from arpyes.types import (
    ArpesSpectrum,
    DiagonalizedBands,
    OrbitalBasis,
    PolarizationConfig,
    SelfEnergyConfig,
    SimulationParams,
    SlaterParams,
    make_arpes_spectrum,
)
from arpyes.types.aliases import ScalarFloat

from .broadening import fermi_dirac, voigt
from .polarization import build_efield, build_polarization_vectors
from .self_energy import evaluate_self_energy

# Physical constants
_HBAR_EV_S: float = 6.582119569e-16  # eV·s
_ME_EV: float = 0.51099895e6  # electron mass in eV/c^2
_HBAR_C_EV_A: float = 1973.269804  # hbar*c in eV·Å
_BOHR_TO_ANGSTROM: float = 0.529177


def _ekin_to_k_magnitude(
    photon_energy: Float[Array, " "],
    work_function: Float[Array, " "],
    binding_energy: Float[Array, " "],
) -> Float[Array, " "]:
    r"""Compute photoelectron momentum magnitude from kinematics.

    Converts photon energy, work function, and binding energy into the
    magnitude of the photoelectron wavevector using the free-electron
    final-state approximation.

    Extended Summary
    ----------------
    In the three-step model of photoemission, the photoelectron kinetic
    energy is:

    .. math::

        E_{\mathrm{kin}} = h\nu - W - |E_b|

    where :math:`h\nu` is the photon energy, :math:`W` is the work
    function, and :math:`E_b` is the binding energy (negative by
    convention). The free-electron wavevector magnitude is then:

    .. math::

        |k| = \frac{\sqrt{2\,m_e\,E_{\mathrm{kin}}}}{\hbar}

    Implementation Logic
    --------------------
    1. **Compute kinetic energy**:
       ``e_kin = photon_energy - work_function - |binding_energy|``
       The absolute value of ``binding_energy`` is used so that the
       sign convention does not matter.

    2. **Guard against negative kinetic energy**:
       ``safe_ekin = max(e_kin, 0)``
       Negative kinetic energy is unphysical (the photon cannot eject
       this electron); clamping to zero yields ``|k| = 0`` for such
       bands, which is correct and gradient-safe.

    3. **Convert to wavevector magnitude**:
       ``k_mag = sqrt(2 * m_e * safe_ekin) / (hbar * c)``
       Using natural units with ``_ME_EV = 0.511e6 eV/c^2`` and
       ``_HBAR_C_EV_A = 1973.27 eV*Angstrom``, the result is
       directly in inverse Angstroms. The approximate relation is
       ``|k| ~ 0.5123 * sqrt(E_kin [eV])`` in Angstrom^-1.

    Parameters
    ----------
    photon_energy : Float[Array, " "]
        Incident photon energy in eV.
    work_function : Float[Array, " "]
        Material work function in eV.
    binding_energy : Float[Array, " "]
        Electron binding energy in eV (sign-agnostic: absolute
        value is taken internally).

    Returns
    -------
    k_mag : Float[Array, " "]
        Photoelectron wavevector magnitude in inverse Angstroms.

    Notes
    -----
    Uses physical constants defined at module level:
    ``_ME_EV`` (electron mass in eV/c^2) and ``_HBAR_C_EV_A``
    (hbar*c in eV*Angstrom). The function is JAX-traceable and
    supports ``jax.grad`` through the ``jnp.maximum`` guard.
    """
    e_kin: Float[Array, " "] = photon_energy - work_function - jnp.abs(binding_energy)
    safe_ekin: Float[Array, " "] = jnp.maximum(e_kin, 0.0)
    k_mag: Float[Array, " "] = jnp.sqrt(2.0 * _ME_EV * safe_ekin) / _HBAR_C_EV_A
    return k_mag


@jaxtyped(typechecker=beartype)
def simulate_tb_radial(
    diag_bands: DiagonalizedBands,
    slater_params: SlaterParams,
    params: SimulationParams,
    pol_config: PolarizationConfig,
    work_function: ScalarFloat = 4.5,
    self_energy: Optional[SelfEnergyConfig] = None,
    r_grid: Optional[Float[Array, " R"]] = None,
    dk: Optional[ScalarFloat] = None,
) -> ArpesSpectrum:
    r"""End-to-end differentiable ARPES forward model.

    Computes dipole matrix elements from first principles using
    radial integrals, Gaunt coefficients, and real spherical
    harmonics, then produces a simulated ARPES spectrum. The entire
    pipeline is JAX-traceable and supports ``jax.grad`` with respect
    to all continuous parameters.

    Extended Summary
    ----------------
    This is the "Chinook-style" tight-binding ARPES forward model.
    Unlike the six-level ``simulate_*`` functions in ``spectrum.py``
    (which use VASP orbital projections as pre-computed weights),
    this function computes dipole matrix elements ab initio from
    Slater-type radial wavefunctions and the diagonalized
    tight-binding eigenvectors. This makes the entire simulation
    differentiable with respect to:

    - ``slater_params.zeta`` (Slater exponents controlling radial
      extent)
    - ``diag_bands.eigenvalues`` and ``diag_bands.eigenvectors``
    - ``params.sigma``, ``params.gamma``, ``params.temperature``
    - ``pol_config.theta``, ``pol_config.phi``
    - ``work_function``
    - ``self_energy.coefficients`` (if provided)

    Implementation Logic
    --------------------
    The simulation proceeds in five stages:

    1. **Energy axis and radial grid setup**:
       A linear energy axis of ``fidelity`` points spanning
       ``[energy_min, energy_max]`` is created. If no radial grid is
       provided, a default grid of 10000 points on ``[1e-6, 50.0]``
       Bohr is used for the radial integrals.

    2. **Precompute radial wavefunctions**:
       For each orbital in ``slater_params.orbital_basis``, the
       Slater radial function ``R_nl(r; zeta)`` is evaluated on the
       grid and scaled by the leading coefficient. This is a
       Python-level loop that JAX unrolls during tracing.

    3. **Compute band intensities |M(k,b)|^2**:
       For each (k-point, band) pair, the total dipole matrix element
       is:

       .. math::

           M_{k,b} = \sum_o c_{k,b,o} \cdot M_o(k, \hat{\epsilon})

       where :math:`c_{k,b,o}` are eigenvector coefficients and
       :math:`M_o` is the single-orbital dipole matrix element
       computed by ``dipole_matrix_element_single``. The photoelectron
       momentum ``k`` is derived from kinematics via
       ``_ekin_to_k_magnitude``, scaled along the crystal k-direction.
       For unpolarized light, s- and p-polarization intensities are
       averaged: ``I = (|M_s|^2 + |M_p|^2) / 2``. The computation
       is vectorized with nested ``jax.vmap`` over bands and k-points.

    4. **Spectral broadening**:
       Each band contributes a Voigt profile weighted by the
       Fermi-Dirac occupation and the band intensity. If a
       ``SelfEnergyConfig`` is provided, the Lorentzian width
       ``gamma`` becomes energy-dependent via ``evaluate_self_energy``,
       and a per-energy-point Voigt is computed. Otherwise, a
       constant ``params.gamma`` is used. Contributions from all
       bands at each k-point are summed.

    5. **Optional momentum broadening**:
       If ``dk`` is specified, a Gaussian convolution along the
       k-axis is applied via ``apply_momentum_broadening`` to
       simulate finite angular acceptance. Cumulative k-distances
       are computed from the k-point path.

    Parameters
    ----------
    diag_bands : DiagonalizedBands
        Diagonalized electronic structure containing eigenvalues of
        shape ``(K, B)``, eigenvectors of shape ``(K, B, O)`` where
        O is the number of orbitals, k-points of shape ``(K, 3)``,
        and the Fermi energy.
    slater_params : SlaterParams
        Slater radial wavefunction parameters including Slater
        exponents ``zeta`` of shape ``(O,)``, expansion coefficients,
        and the ``orbital_basis`` specifying (n, l, m) quantum
        numbers for each orbital.
    params : SimulationParams
        Simulation parameters including ``energy_min``, ``energy_max``,
        ``fidelity``, ``sigma`` (Gaussian width), ``gamma``
        (Lorentzian width), ``temperature``, and ``photon_energy``.
    pol_config : PolarizationConfig
        Photon polarization configuration specifying
        ``polarization_type``, incidence angles ``theta`` and ``phi``,
        and ``polarization_angle``.
    work_function : ScalarFloat, optional
        Material work function in eV. Default is 4.5.
    self_energy : SelfEnergyConfig, optional
        Energy-dependent self-energy model for the Lorentzian
        broadening. If ``None``, the constant ``params.gamma`` is
        used for all energies.
    r_grid : Float[Array, " R"], optional
        Radial integration grid in Bohr. Default is 10000 points
        linearly spaced on ``[1e-6, 50.0]``.
    dk : ScalarFloat, optional
        Momentum broadening width in inverse Angstroms. If ``None``,
        no k-space convolution is applied.

    Returns
    -------
    spectrum : ArpesSpectrum
        Simulated ARPES spectrum with ``intensity`` of shape
        ``(K, E)`` and ``energy_axis`` of shape ``(E,)``.

    See Also
    --------
    simulate_expert : Projection-based simulation (uses VASP weights
        rather than ab-initio dipole matrix elements).
    dipole_matrix_element_single : Core dipole integral computation.
    slater_radial : Slater-type radial wavefunction evaluation.
    evaluate_self_energy : Energy-dependent broadening model.
    apply_momentum_broadening : k-space Gaussian convolution.

    Notes
    -----
    The inner function ``_single_k_band`` adds a small epsilon
    (1e-30) inside the k-vector norm to avoid NaN gradients at the
    Gamma point (k = 0). The Python-level ``for`` loop over orbitals
    is unrolled by the JAX tracer and does not affect runtime
    performance after JIT compilation.
    """
    # Energy axis
    energy_axis: Float[Array, " E"] = jnp.linspace(
        params.energy_min, params.energy_max, params.fidelity
    )

    # Radial grid
    if r_grid is None:
        r_grid = jnp.linspace(1e-6, 50.0, 10000)

    # Work function as JAX scalar
    W: Float[Array, " "] = jnp.asarray(work_function, dtype=jnp.float64)

    # Build polarization E-field
    is_unpolarized: bool = (
        pol_config.polarization_type.lower() == "unpolarized"
    )

    basis: OrbitalBasis = slater_params.orbital_basis
    n_orbitals: int = len(basis.n_values)

    # Precompute radial wavefunctions on the grid for each orbital
    # (Python-level loop, unrolled by JAX tracer)
    radial_on_grid: list[Float[Array, " R"]] = []
    for o in range(n_orbitals):
        R_vals: Float[Array, " R"] = slater_radial(
            r_grid, basis.n_values[o], slater_params.zeta[o]
        )
        R_vals = R_vals * slater_params.coefficients[o, 0]
        radial_on_grid.append(R_vals)

    def _compute_band_intensity_single_efield(
        efield: Complex[Array, " 3"],
    ) -> Float[Array, "K B"]:
        """Compute squared dipole matrix element for all (k, band) pairs.

        For a fixed electric-field polarization vector, evaluates the
        total dipole matrix element ``M = sum_o c_{k,b,o} * M_o`` for
        every (k-point, band) pair and returns ``|M|^2``. The
        computation is vectorized with nested ``jax.vmap``: the inner
        vmap maps over bands (B) and the outer vmap maps over k-points
        (K), yielding the full intensity array of shape ``(K, B)``.

        Parameters
        ----------
        efield : Complex[Array, " 3"]
            Complex electric-field polarization vector.

        Returns
        -------
        Float[Array, "K B"]
            Squared modulus of the total dipole matrix element for
            each (k-point, band) pair.
        """

        def _single_k_band(
            k_crystal: Float[Array, " 3"],
            eigvec: Complex[Array, " O"],
            eigenval: Float[Array, " "],
        ) -> Float[Array, " "]:
            """Compute |M|^2 for a single (k-point, band) pair.

            Converts the crystal k-vector to the photoelectron
            wavevector using free-electron kinematics, then sums the
            orbital-weighted dipole matrix elements and returns the
            squared modulus. A small epsilon (1e-30) is added to the
            k-vector norm to ensure gradient safety at the Gamma point.

            Parameters
            ----------
            k_crystal : Float[Array, " 3"]
                Crystal momentum vector for this k-point.
            eigvec : Complex[Array, " O"]
                Eigenvector coefficients for this band at this k-point.
            eigenval : Float[Array, " "]
                Band eigenvalue (binding energy) in eV.

            Returns
            -------
            Float[Array, " "]
                Squared modulus of the total dipole matrix element.
            """
            # Compute photoelectron k magnitude from kinematics
            k_mag: Float[Array, " "] = _ekin_to_k_magnitude(
                params.photon_energy, W, eigenval
            )
            # Use crystal k-direction, scale to photoelectron magnitude
            # Gradient-safe norm: eps avoids NaN grad at Gamma point (k=0)
            k_norm: Float[Array, " "] = jnp.sqrt(
                jnp.dot(k_crystal, k_crystal) + 1e-30
            )
            k_hat: Float[Array, " 3"] = k_crystal / k_norm
            k_vec: Float[Array, " 3"] = k_hat * k_mag

            # Compute total M = sum_o c_{k,b,o} * M_o
            M_total: Complex[Array, " "] = jnp.zeros((), dtype=jnp.complex128)
            for o in range(n_orbitals):
                M_o: Complex[Array, " "] = dipole_matrix_element_single(
                    k_vec,
                    r_grid,
                    radial_on_grid[o],
                    basis.l_values[o],
                    basis.m_values[o],
                    efield,
                )
                M_total = M_total + eigvec[o] * M_o

            intensity_kb: Float[Array, " "] = jnp.abs(M_total) ** 2
            return intensity_kb

        # vmap over bands (B), then over k-points (K)
        _vmap_bands = jax.vmap(
            _single_k_band,
            in_axes=(None, 0, 0),
        )
        _vmap_k = jax.vmap(
            _vmap_bands,
            in_axes=(0, 0, 0),
        )
        result: Float[Array, "K B"] = _vmap_k(
            diag_bands.kpoints,
            diag_bands.eigenvectors,
            diag_bands.eigenvalues,
        )
        return result

    # Compute band intensities
    if is_unpolarized:
        e_s: Float[Array, " 3"]
        e_p: Float[Array, " 3"]
        e_s, e_p = build_polarization_vectors(pol_config.theta, pol_config.phi)
        e_s_c: Complex[Array, " 3"] = e_s.astype(jnp.complex128)
        e_p_c: Complex[Array, " 3"] = e_p.astype(jnp.complex128)
        i_s: Float[Array, "K B"] = _compute_band_intensity_single_efield(e_s_c)
        i_p: Float[Array, "K B"] = _compute_band_intensity_single_efield(e_p_c)
        band_intensity: Float[Array, "K B"] = (i_s + i_p) / 2.0
    else:
        efield: Complex[Array, " 3"] = build_efield(pol_config)
        band_intensity = _compute_band_intensity_single_efield(efield)

    # Broadening: Voigt profile with optional energy-dependent gamma
    if self_energy is not None:
        gamma_E: Float[Array, " E"] = evaluate_self_energy(
            energy_axis, self_energy
        )

        def _single_band_se(
            energy: Float[Array, " "],
            bi: Float[Array, " "],
        ) -> Float[Array, " E"]:
            """Spectral contribution of one band with energy-dependent gamma.

            Computes the Fermi-Dirac occupation at the band energy,
            then evaluates a Voigt profile at each energy-axis point
            with a different Lorentzian width ``gamma(E)`` from the
            self-energy model. The result is scaled by the band
            intensity.

            Parameters
            ----------
            energy : Float[Array, " "]
                Band eigenvalue in eV.
            bi : Float[Array, " "]
                Band intensity weight (|M|^2).

            Returns
            -------
            Float[Array, " E"]
                Weighted spectral contribution of this band.
            """
            fd: Float[Array, " "] = fermi_dirac(
                energy, diag_bands.fermi_energy, params.temperature
            )
            # Per-energy-point Voigt with varying gamma
            profile: Float[Array, " E"] = jax.vmap(
                lambda e_pt, g: voigt(
                    jnp.expand_dims(e_pt, 0), energy, params.sigma, g
                ).squeeze(),
            )(energy_axis, gamma_E)
            contribution: Float[Array, " E"] = bi * fd * profile
            return contribution

        def _single_kpoint_se(
            energies: Float[Array, " B"],
            bi_k: Float[Array, " B"],
        ) -> Float[Array, " E"]:
            """Sum spectral contributions over all bands at one k-point.

            Vmaps ``_single_band_se`` over the band axis and sums the
            resulting (B, E) array along the band dimension to produce
            the total spectral intensity at this k-point.

            Parameters
            ----------
            energies : Float[Array, " B"]
                Band eigenvalues for all bands at this k-point.
            bi_k : Float[Array, " B"]
                Band intensity weights for all bands at this k-point.

            Returns
            -------
            Float[Array, " E"]
                Total spectral intensity at this k-point.
            """
            contributions: Float[Array, "B E"] = jax.vmap(_single_band_se)(
                energies, bi_k
            )
            total: Float[Array, " E"] = jnp.sum(contributions, axis=0)
            return total

        intensity: Float[Array, "K E"] = jax.vmap(_single_kpoint_se)(
            diag_bands.eigenvalues, band_intensity
        )
    else:

        def _single_band(
            energy: Float[Array, " "],
            bi: Float[Array, " "],
        ) -> Float[Array, " E"]:
            """Spectral contribution of one band with constant gamma.

            Computes the Fermi-Dirac occupation at the band energy,
            evaluates a Voigt profile with constant ``sigma`` and
            ``gamma`` from ``params``, and scales by the band
            intensity.

            Parameters
            ----------
            energy : Float[Array, " "]
                Band eigenvalue in eV.
            bi : Float[Array, " "]
                Band intensity weight (|M|^2).

            Returns
            -------
            Float[Array, " E"]
                Weighted spectral contribution of this band.
            """
            fd: Float[Array, " "] = fermi_dirac(
                energy, diag_bands.fermi_energy, params.temperature
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
            """Sum spectral contributions over all bands at one k-point.

            Vmaps ``_single_band`` over the band axis and sums the
            resulting (B, E) array along the band dimension to produce
            the total spectral intensity at this k-point.

            Parameters
            ----------
            energies : Float[Array, " B"]
                Band eigenvalues for all bands at this k-point.
            bi_k : Float[Array, " B"]
                Band intensity weights for all bands at this k-point.

            Returns
            -------
            Float[Array, " E"]
                Total spectral intensity at this k-point.
            """
            contributions: Float[Array, "B E"] = jax.vmap(_single_band)(
                energies, bi_k
            )
            total: Float[Array, " E"] = jnp.sum(contributions, axis=0)
            return total

        intensity: Float[Array, "K E"] = jax.vmap(_single_kpoint)(
            diag_bands.eigenvalues, band_intensity
        )

    # Optional momentum broadening
    if dk is not None:
        from .resolution import apply_momentum_broadening

        # Compute cumulative k-distances
        dk_vecs: Float[Array, "Km1 3"] = jnp.diff(diag_bands.kpoints, axis=0)
        dk_norms: Float[Array, " Km1"] = jnp.linalg.norm(dk_vecs, axis=1)
        k_distances: Float[Array, " K"] = jnp.concatenate(
            [jnp.zeros(1), jnp.cumsum(dk_norms)]
        )
        intensity = apply_momentum_broadening(intensity, k_distances, dk)

    spectrum: ArpesSpectrum = make_arpes_spectrum(
        intensity=intensity, energy_axis=energy_axis
    )
    return spectrum


__all__: list[str] = ["simulate_tb_radial"]
