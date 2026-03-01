r"""End-to-end differentiable ARPES forward model.

Extended Summary
----------------
Implements ``simulate_tb_radial``, the fully differentiable ARPES
simulation function that computes dipole matrix elements from first
principles using radial integrals, Gaunt coefficients, and real
spherical harmonics. Supports ``jax.grad`` with respect to Slater
exponents, eigenvectors/eigenvalues, simulation parameters, and
work function.

Pipeline:
    diag_bands → eigenvectors → per-orbital M(k) →
    total |M|^2 × Fermi-Dirac × Voigt → I(k, E)
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
    """Compute photoelectron momentum magnitude.

    E_kin = hv - W - |E_b|
    |k| = sqrt(2 m_e E_kin) / hbar  (in Å^-1)

    Using: |k| [Å^-1] = sqrt(2 * m_e * E_kin) / (hbar*c) * c
         = sqrt(2 * 0.511e6 eV * E_kin) / 1973.27 eV·Å
         = 0.5123 * sqrt(E_kin [eV])  (approximate)
    """
    e_kin = photon_energy - work_function - jnp.abs(binding_energy)
    safe_ekin = jnp.maximum(e_kin, 0.0)
    k_mag = jnp.sqrt(2.0 * _ME_EV * safe_ekin) / _HBAR_C_EV_A
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

    Computes dipole matrix elements from first principles and
    produces a simulated ARPES spectrum. The entire pipeline is
    JAX-traceable and supports ``jax.grad`` with respect to:

    - ``slater_params.zeta`` (Slater exponents)
    - ``diag_bands.eigenvalues``, ``diag_bands.eigenvectors``
    - ``params.sigma``, ``params.gamma``, ``params.temperature``
    - ``pol_config.theta``, ``pol_config.phi``
    - ``work_function``
    - ``self_energy.coefficients`` (if provided)

    Parameters
    ----------
    diag_bands : DiagonalizedBands
        Diagonalized electronic structure.
    slater_params : SlaterParams
        Slater radial wavefunction parameters.
    params : SimulationParams
        Simulation parameters (energy window, broadening, etc.).
    pol_config : PolarizationConfig
        Photon polarization configuration.
    work_function : ScalarFloat, optional
        Work function in eV. Default 4.5.
    self_energy : SelfEnergyConfig, optional
        Energy-dependent broadening. If None, uses ``params.gamma``.
    r_grid : Float[Array, " R"], optional
        Radial grid for integration. Default: 10000 points on [0, 50].
    dk : ScalarFloat, optional
        Momentum broadening in Å^-1. If None, no k-convolution.

    Returns
    -------
    spectrum : ArpesSpectrum
        Simulated ARPES intensity map.
    """
    # Energy axis
    energy_axis: Float[Array, " E"] = jnp.linspace(
        params.energy_min, params.energy_max, params.fidelity
    )

    # Radial grid
    if r_grid is None:
        r_grid = jnp.linspace(1e-6, 50.0, 10000)

    # Work function as JAX scalar
    W = jnp.asarray(work_function, dtype=jnp.float64)

    # Build polarization E-field
    is_unpolarized: bool = (
        pol_config.polarization_type.lower() == "unpolarized"
    )

    basis = slater_params.orbital_basis
    n_orbitals = len(basis.n_values)

    # Precompute radial wavefunctions on the grid for each orbital
    # (Python-level loop, unrolled by JAX tracer)
    radial_on_grid = []
    for o in range(n_orbitals):
        R_vals = slater_radial(
            r_grid, basis.n_values[o], slater_params.zeta[o]
        )
        R_vals = R_vals * slater_params.coefficients[o, 0]
        radial_on_grid.append(R_vals)

    def _compute_band_intensity_single_efield(
        efield: Complex[Array, " 3"],
    ) -> Float[Array, "K B"]:
        """Compute |M|^2 for all (k, band) with a given E-field."""

        def _single_k_band(
            k_crystal: Float[Array, " 3"],
            eigvec: Complex[Array, " O"],
            eigenval: Float[Array, " "],
        ) -> Float[Array, " "]:
            """Intensity for one (k, band) pair."""
            # Compute photoelectron k magnitude from kinematics
            k_mag = _ekin_to_k_magnitude(params.photon_energy, W, eigenval)
            # Use crystal k-direction, scale to photoelectron magnitude
            # Gradient-safe norm: eps avoids NaN grad at Gamma point (k=0)
            k_norm = jnp.sqrt(jnp.dot(k_crystal, k_crystal) + 1e-30)
            k_hat = k_crystal / k_norm
            k_vec = k_hat * k_mag

            # Compute total M = sum_o c_{k,b,o} * M_o
            M_total = jnp.zeros((), dtype=jnp.complex128)
            for o in range(n_orbitals):
                M_o = dipole_matrix_element_single(
                    k_vec,
                    r_grid,
                    radial_on_grid[o],
                    basis.l_values[o],
                    basis.m_values[o],
                    efield,
                )
                M_total = M_total + eigvec[o] * M_o

            return jnp.abs(M_total) ** 2

        # vmap over bands (B), then over k-points (K)
        _vmap_bands = jax.vmap(
            _single_k_band,
            in_axes=(None, 0, 0),
        )
        _vmap_k = jax.vmap(
            _vmap_bands,
            in_axes=(0, 0, 0),
        )
        return _vmap_k(
            diag_bands.kpoints,
            diag_bands.eigenvectors,
            diag_bands.eigenvalues,
        )

    # Compute band intensities
    if is_unpolarized:
        e_s, e_p = build_polarization_vectors(pol_config.theta, pol_config.phi)
        e_s_c = e_s.astype(jnp.complex128)
        e_p_c = e_p.astype(jnp.complex128)
        i_s = _compute_band_intensity_single_efield(e_s_c)
        i_p = _compute_band_intensity_single_efield(e_p_c)
        band_intensity: Float[Array, "K B"] = (i_s + i_p) / 2.0
    else:
        efield = build_efield(pol_config)
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
            fd = fermi_dirac(
                energy, diag_bands.fermi_energy, params.temperature
            )
            # Per-energy-point Voigt with varying gamma
            profile = jax.vmap(
                lambda e_pt, g: voigt(
                    jnp.expand_dims(e_pt, 0), energy, params.sigma, g
                ).squeeze(),
            )(energy_axis, gamma_E)
            return bi * fd * profile

        def _single_kpoint_se(
            energies: Float[Array, " B"],
            bi_k: Float[Array, " B"],
        ) -> Float[Array, " E"]:
            contributions = jax.vmap(_single_band_se)(energies, bi_k)
            return jnp.sum(contributions, axis=0)

        intensity: Float[Array, "K E"] = jax.vmap(_single_kpoint_se)(
            diag_bands.eigenvalues, band_intensity
        )
    else:

        def _single_band(
            energy: Float[Array, " "],
            bi: Float[Array, " "],
        ) -> Float[Array, " E"]:
            fd = fermi_dirac(
                energy, diag_bands.fermi_energy, params.temperature
            )
            profile = voigt(energy_axis, energy, params.sigma, params.gamma)
            return bi * fd * profile

        def _single_kpoint(
            energies: Float[Array, " B"],
            bi_k: Float[Array, " B"],
        ) -> Float[Array, " E"]:
            contributions = jax.vmap(_single_band)(energies, bi_k)
            return jnp.sum(contributions, axis=0)

        intensity = jax.vmap(_single_kpoint)(
            diag_bands.eigenvalues, band_intensity
        )

    # Optional momentum broadening
    if dk is not None:
        from .resolution import apply_momentum_broadening

        # Compute cumulative k-distances
        dk_vecs = jnp.diff(diag_bands.kpoints, axis=0)
        dk_norms = jnp.linalg.norm(dk_vecs, axis=1)
        k_distances = jnp.concatenate([jnp.zeros(1), jnp.cumsum(dk_norms)])
        intensity = apply_momentum_broadening(intensity, k_distances, dk)

    return make_arpes_spectrum(intensity=intensity, energy_axis=energy_axis)


__all__: list[str] = ["simulate_tb_radial"]
