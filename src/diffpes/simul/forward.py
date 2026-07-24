r"""Run an end-to-end differentiable ARPES forward model.

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

JAX can trace every stage. ``jax.grad`` can differentiate the Slater exponents,
eigenvectors, eigenvalues, simulation parameters, polarization angles, and
work function. The model also supports an optional energy-dependent self-energy
and momentum broadening.

Routine Listings
----------------
:func:`simulate_tb_radial`
    Run the end-to-end differentiable ARPES forward model.

Notes
-----
Physical constants used in this module:

- ``HBAR_EV_S``: hbar in eV*s (6.582e-16)
- ``ME_EV``: electron mass in eV/c^2 (0.511e6)
- ``HBAR_C_EV_A``: hbar*c in eV*Angstrom (1973.27)
- ``BOHR_TO_ANGSTROM``: Bohr radius in Angstroms (0.5292)

The private helper ``_ekin_to_k_magnitude`` converts photon energy and signed
Fermi-relative electron energy to a photoelectron wavevector magnitude and
an explicit emission-validity mask.
"""

from functools import partial

import jax
import jax.numpy as jnp
from beartype import beartype
from beartype.typing import Callable, Optional
from jaxtyping import Array, Bool, Complex, Float, Integer, jaxtyped

from diffpes.maths import dipole_matrix_element_single, safe_divide, safe_norm
from diffpes.radial import slater_radial
from diffpes.simul.kinematics import final_state_k_inv_ang, kinetic_energy_ev
from diffpes.types import (
    ArpesSpectrum,
    DiagonalizedBands,
    OrbitalBasis,
    PolarizationConfig,
    ScalarFloat,
    SelfEnergyConfig,
    SimulationParams,
    SlaterParams,
    make_arpes_spectrum,
)

from .broadening import fermi_dirac, voigt
from .polarization import build_efield, build_polarization_vectors
from .resolution import apply_momentum_broadening
from .self_energy import evaluate_self_energy


def _ekin_to_k_magnitude(
    photon_energy: Float[Array, " "],
    work_function: Float[Array, " "],
    omega_rel_fermi: Float[Array, " "],
) -> tuple[Float[Array, " "], Bool[Array, " "]]:
    r"""Compute photoelectron momentum and its emission-validity mask.

    Converts photon energy, work function, and Fermi-relative electron energy
    into the magnitude of the photoelectron wavevector using the free-electron
    final-state approximation.

    In the three-step model of photoemission, the photoelectron kinetic
    energy is:

    .. math::

        E_{\mathrm{kin}} = h\nu - W + (E-E_F)

    where :math:`h\nu` is the photon energy, :math:`W` is the work function,
    and :math:`E-E_F` is the signed electron energy.

    .. math::

        |k| = \frac{\sqrt{2\,m_e\,E_{\mathrm{kin}}}}{\hbar}

    Implementation Logic
    --------------------
    1. **Compute the kinetic energy**:
       ``e_kin = photon_energy - work_function + omega_rel_fermi``

    2. **Propagate forbidden emission**:
       The helpers return raw energy, zero momentum, and a false validity mask
       for nonpositive energy.

    3. **Convert to wavevector magnitude**:
       ``k_mag = safe_sqrt(2 * m_e * e_kin) / (hbar * c)``
       Using natural units with ``ME_EV = 0.511e6 eV/c^2`` and
       ``HBAR_C_EV_A = 1973.27 eV*Angstrom``, the result is
       directly in inverse Angstroms. The approximate relation is
       ``|k| ~ 0.5123 * sqrt(E_kin [eV])`` in Angstrom^-1.

    Parameters
    ----------
    photon_energy : Float[Array, " "]
        Incident photon energy in eV.
    work_function : Float[Array, " "]
        Material work function in eV.
    omega_rel_fermi : Float[Array, " "]
        Signed electron energy ``E - E_F`` in eV.

    Returns
    -------
    k_mag : Float[Array, " "]
        Photoelectron wavevector magnitude in inverse Angstroms.
    emission_valid : Bool[Array, " "]
        Mask selecting strictly positive kinetic energy.

    Notes
    -----
    The function supports JAX transformations. Differentiate only away from
    the emission threshold.
    """
    e_kin: Float[Array, " "]
    emission_valid: Bool[Array, " "]
    e_kin, emission_valid = kinetic_energy_ev(
        photon_energy,
        work_function,
        omega_rel_fermi,
    )
    k_mag: Float[Array, " "]
    k_mag, _ = final_state_k_inv_ang(e_kin)
    result: tuple[Float[Array, " "], Bool[Array, " "]] = (
        k_mag,
        emission_valid,
    )
    return result


@jaxtyped(typechecker=beartype)
def simulate_tb_radial(  # noqa: PLR0915
    diag_bands: DiagonalizedBands,
    slater_params: SlaterParams,
    params: SimulationParams,
    pol_config: PolarizationConfig,
    work_function: ScalarFloat = 4.5,
    self_energy: Optional[SelfEnergyConfig] = None,
    r_grid: Optional[Float[Array, " R"]] = None,
    dk: Optional[ScalarFloat] = None,
) -> ArpesSpectrum:
    r"""Run the end-to-end differentiable ARPES forward model.

    Computes dipole matrix elements from first principles using
    radial integrals, Gaunt coefficients, and real spherical
    harmonics, then produces a simulated ARPES spectrum. The entire
    pipeline is JAX-traceable and supports ``jax.grad`` with respect
    to all continuous parameters.

    This function provides the Chinook-style tight-binding ARPES forward model.
    The six-level functions in ``spectrum.py`` use VASP orbital projections as
    weights. In contrast, this function computes dipole matrix elements from
    Slater-type radial wavefunctions and tight-binding eigenvectors. JAX can
    therefore differentiate the complete simulation with respect to:

    - ``slater_params.zeta`` (Slater exponents controlling radial
      extent)
    - ``diag_bands.eigenvalues`` and ``diag_bands.eigenvectors``
    - ``params.sigma``, ``params.gamma``, ``params.temperature``
    - ``pol_config.theta``, ``pol_config.phi``
    - ``work_function``
    - ``self_energy.coefficients`` (if provided)

    :see: :class:`~.test_forward.TestSimulateTBRadial`

    Implementation Logic
    --------------------
    Before the numerical stages, the function requires exact agreement
    between the static Slater and diagonalized orbital bases. This keeps each
    radial row paired with coefficients for the same orbital.

    The simulation proceeds in five stages:

    1. **Create the numerical grids**::

           energy_axis: Float[Array, " E"] = jnp.linspace(
               params.energy_min, params.energy_max, params.fidelity
           )

       The energy axis fixes the output sampling. The optional radial grid
       controls the integration sampling in Bohr.

    2. **Evaluate the radial wavefunctions**::

           radial_scan: tuple[None, Float[Array, "O R"]] = jax.lax.scan(
               _scan_radial,
               None,
               orbital_indices,
           )

       The scan traverses the differentiable Slater parameters.
       ``jax.lax.switch`` selects each static principal quantum number.

    3. **Compute the coherent band intensities**::

           band_intensity: Float[Array, "K B"] = (i_s + i_p) / 2.0

       Nested ``jax.vmap`` calls evaluate all k-points and bands. Each
       evaluation forms the coherent orbital sum before the squared modulus.

       The coherent matrix element is:

       .. math::

           M_{k,b} = \sum_o c_{k,b,o} \cdot M_o(k, \hat{\epsilon})

       Here, :math:`c_{k,b,o}` is an eigenvector coefficient and
       :math:`M_o` is one orbital matrix element. Unpolarized light averages
       the s-polarized and p-polarized intensities.

    4. **Broaden the band contributions**::

           intensity: Float[Array, "K E"] = jax.vmap(_single_kpoint)(
               diag_bands.eigenvalues,
               band_intensity,
           )

       Each band contributes a Fermi-weighted Voigt profile. The optional
       self-energy makes the Lorentzian width depend on energy.

    5. **Apply the momentum response**::

           intensity = apply_momentum_broadening(
               intensity, k_distances, dk
           )

       The Gaussian response represents finite angular acceptance. The
       operation remains differentiable with respect to the spectrum and the
       response width.

    Parameters
    ----------
    diag_bands : DiagonalizedBands
        Diagonalized electronic structure with eigenvalues of shape ``(K, B)``
        and eigenvectors of shape ``(K, B, O)``. O is the orbital count. The
        carrier also contains k-points and the Fermi energy.
    slater_params : SlaterParams
        Slater radial parameters with exponents ``zeta`` of shape ``(O,)``
        and expansion coefficients. Its ``orbital_basis`` specifies atom
        assignments and ``(n, l, m)`` quantum numbers. This basis must equal
        ``diag_bands.basis``.
    params : SimulationParams
        Simulation parameters including ``energy_min``, ``energy_max``,
        ``fidelity``, ``sigma`` (Gaussian width), ``gamma``
        (Lorentzian width), ``temperature``, and ``photon_energy``.
    pol_config : PolarizationConfig
        Legacy sample-frame photon polarization configuration specifying
        ``polarization_type``, incidence angles ``theta`` and ``phi``,
        and ``polarization_angle``. This path applies no laboratory-to-sample
        or detector-to-sample frame transform.
    work_function : ScalarFloat, optional
        Material work function in eV. Default is 4.5.
    self_energy : Optional[SelfEnergyConfig]
        Energy-dependent self-energy model for the Lorentzian
        broadening. If ``None``, the function uses ``params.gamma`` for all
        energies.
    r_grid : Optional[Float[Array, " R"]]
        Radial integration grid in Bohr. Default is 10000 points
        linearly spaced on ``[1e-6, 50.0]``.
    dk : Optional[ScalarFloat]
        Momentum broadening width in inverse Angstroms. If ``None``, the
        function does not apply k-space convolution.

    Returns
    -------
    spectrum : ArpesSpectrum
        Simulated ARPES spectrum with ``intensity`` of shape
        ``(K, E)`` and ``energy_axis`` of shape ``(E,)``.

    Raises
    ------
    ValueError
        If the orbital metadata in ``slater_params.orbital_basis`` differs
        from ``diag_bands.basis``.

    Notes
    -----
    The inner function ``_single_k_band`` uses :func:`safe_norm` and
    :func:`safe_divide` to select a zero direction and zero gradient at the
    Gamma point. The outer path first maps fractional k-points through the
    reciprocal lattice. It then preserves the Cartesian direction when it
    forms photoelectron momentum. Both radial construction and coherent
    orbital summation use ``jax.lax.scan``. Static quantum numbers select
    specialized branches with ``jax.lax.switch``.

    ``pol_config`` follows the legacy sample-frame convention of this radial
    path. The function performs no detector-pixel or sample-azimuth transform.

    See Also
    --------
    simulate_expert : Projection-based simulation (uses VASP weights
        rather than ab-initio dipole matrix elements).
    dipole_matrix_element_single : Core dipole integral computation.
    slater_radial : Slater-type radial wavefunction evaluation.
    evaluate_self_energy : Energy-dependent broadening model.
    apply_momentum_broadening : k-space Gaussian convolution.
    """
    basis: OrbitalBasis = slater_params.orbital_basis
    bands_basis: OrbitalBasis = diag_bands.basis
    if (
        basis.atom_indices != bands_basis.atom_indices
        or basis.n != bands_basis.n
        or basis.l != bands_basis.l
        or basis.m != bands_basis.m
        or basis.spin != bands_basis.spin
        or basis.labels != bands_basis.labels
    ):
        message: str = (
            "slater_params.orbital_basis must equal diag_bands.basis"
        )
        raise ValueError(message)
    n_orbitals: int = len(basis.n)

    energy_axis: Float[Array, " E"] = jnp.linspace(
        params.energy_min, params.energy_max, params.fidelity
    )

    if r_grid is None:
        r_grid = jnp.linspace(1e-6, 50.0, 10000)
    r_grid: Float[Array, " R"]

    W: Float[Array, " "] = jnp.asarray(work_function, dtype=jnp.float64)
    cartesian_kpoints: Float[Array, "K 3"] = (
        diag_bands.kpoints @ diag_bands.geometry.reciprocal
    )

    is_unpolarized: bool = (
        pol_config.polarization_type.lower() == "unpolarized"
    )

    def _evaluate_radial(
        operand: tuple[Float[Array, ""], Float[Array, ""]],
        *,
        n: int,
    ) -> Float[Array, " R"]:
        """Evaluate one radial branch specialized to static ``n``."""
        zeta_value: Float[Array, ""] = operand[0]
        coefficient: Float[Array, ""] = operand[1]
        radial_values: Float[Array, " R"] = (
            slater_radial(r_grid, n, zeta_value) * coefficient
        )
        return radial_values

    radial_branches: tuple[
        Callable[
            [tuple[Float[Array, ""], Float[Array, ""]]],
            Float[Array, " R"],
        ],
        ...,
    ] = tuple(partial(_evaluate_radial, n=n_value) for n_value in basis.n)

    def _scan_radial(
        carry: None,
        orbital_index: Integer[Array, ""],
    ) -> tuple[None, Float[Array, " R"]]:
        """Construct one radial row from traced Slater parameters."""
        operand: tuple[Float[Array, ""], Float[Array, ""]] = (
            slater_params.zeta[orbital_index],
            slater_params.coefficients[orbital_index, 0],
        )
        radial_values: Float[Array, " R"] = jax.lax.switch(
            orbital_index,
            radial_branches,
            operand,
        )
        scan_output: tuple[None, Float[Array, " R"]] = (
            carry,
            radial_values,
        )
        return scan_output

    orbital_indices: Integer[Array, " O"] = jnp.arange(n_orbitals)
    radial_scan: tuple[None, Float[Array, "O R"]] = jax.lax.scan(
        _scan_radial,
        None,
        orbital_indices,
    )
    radial_on_grid: Float[Array, "O R"] = radial_scan[1]

    def _evaluate_orbital_contribution(
        operand: tuple[
            Float[Array, " 3"],
            Float[Array, " R"],
            Complex[Array, ""],
            Complex[Array, " 3"],
        ],
        *,
        l: int,
        m: int,
    ) -> Complex[Array, ""]:
        """Evaluate one coherent orbital branch with static ``(l, m)``."""
        k_vector: Float[Array, " 3"] = operand[0]
        radial_values: Float[Array, " R"] = operand[1]
        eigenvector_coefficient: Complex[Array, ""] = operand[2]
        electric_field: Complex[Array, " 3"] = operand[3]
        matrix_element: Complex[Array, ""] = dipole_matrix_element_single(
            k_vector,
            r_grid,
            radial_values,
            l,
            m,
            electric_field,
        )
        contribution: Complex[Array, ""] = (
            eigenvector_coefficient * matrix_element
        )
        return contribution

    orbital_branches: tuple[
        Callable[
            [
                tuple[
                    Float[Array, " 3"],
                    Float[Array, " R"],
                    Complex[Array, ""],
                    Complex[Array, " 3"],
                ]
            ],
            Complex[Array, ""],
        ],
        ...,
    ] = tuple(
        partial(_evaluate_orbital_contribution, l=l_value, m=m_value)
        for l_value, m_value in zip(
            basis.l,
            basis.m,
            strict=True,
        )
    )

    def _compute_band_intensity_single_efield(
        efield: Complex[Array, " 3"],
    ) -> Float[Array, "K B"]:
        """Compute squared dipole matrix element for all (k, band) pairs.

        For a fixed electric-field polarization vector, the function computes
        the total dipole matrix element ``M = sum_o c_{k,b,o} * M_o`` for
        every (k-point, band) pair and returns ``|M|^2``. The
        Nested ``jax.vmap`` operations vectorize the computation. The inner
        operation maps the bands, and the outer operation maps the k-points.
        The result has shape ``(K, B)``.

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
            k_cartesian: Float[Array, " 3"],
            eigvec: Complex[Array, " O"],
            eigenval: Float[Array, " "],
        ) -> Float[Array, " "]:
            """Compute ``|M|^2`` for a single (k-point, band) pair.

            Converts the crystal k-vector to the photoelectron
            wavevector using free-electron kinematics, then sums the
            orbital-weighted dipole matrix elements and returns the
            squared modulus. Named safe norm and division primitives select
            a zero direction and gradient at the Gamma point.

            Parameters
            ----------
            k_cartesian : Float[Array, " 3"]
                Cartesian crystal momentum in inverse Angstroms.
            eigvec : Complex[Array, " O"]
                Eigenvector coefficients for this band at this k-point.
            eigenval : Float[Array, " "]
                Absolute band eigenvalue in eV.

            Returns
            -------
            Float[Array, " "]
                Squared modulus of the total dipole matrix element.
            """
            k_mag: Float[Array, " "]
            emission_valid: Bool[Array, " "]
            omega_rel_fermi: Float[Array, " "] = (
                eigenval - diag_bands.fermi_energy
            )
            k_mag, emission_valid = _ekin_to_k_magnitude(
                params.photon_energy,
                W,
                omega_rel_fermi,
            )
            k_norm: Float[Array, " "] = safe_norm(k_cartesian)
            k_hat: Float[Array, " 3"] = safe_divide(k_cartesian, k_norm)
            k_vec: Float[Array, " 3"] = k_hat * k_mag

            def _scan_orbital(
                matrix_element_sum: Complex[Array, ""],
                orbital_index: Integer[Array, ""],
            ) -> tuple[Complex[Array, ""], None]:
                """Accumulate one coherent orbital contribution."""
                operand: tuple[
                    Float[Array, " 3"],
                    Float[Array, " R"],
                    Complex[Array, ""],
                    Complex[Array, " 3"],
                ] = (
                    k_vec,
                    radial_on_grid[orbital_index],
                    eigvec[orbital_index],
                    efield,
                )
                contribution: Complex[Array, ""] = jax.lax.switch(
                    orbital_index,
                    orbital_branches,
                    operand,
                )
                next_sum: Complex[Array, ""] = (
                    matrix_element_sum + contribution
                )
                scan_output: tuple[Complex[Array, ""], None] = (
                    next_sum,
                    None,
                )
                return scan_output

            initial_sum: Complex[Array, ""] = jnp.zeros(
                (), dtype=jnp.complex128
            )
            orbital_scan: tuple[Complex[Array, ""], None] = jax.lax.scan(
                _scan_orbital,
                initial_sum,
                orbital_indices,
            )
            M_total: Complex[Array, ""] = orbital_scan[0]

            intensity_kb: Float[Array, " "] = jnp.where(
                emission_valid,
                jnp.abs(M_total) ** 2,
                0.0,
            )
            return intensity_kb

        _vmap_bands: Callable[..., Float[Array, " B"]] = jax.vmap(
            _single_k_band,
            in_axes=(None, 0, 0),
        )
        _vmap_k: Callable[..., Float[Array, "K B"]] = jax.vmap(
            _vmap_bands,
            in_axes=(0, 0, 0),
        )
        result: Float[Array, "K B"] = _vmap_k(
            cartesian_kpoints,
            diag_bands.eigenvectors,
            diag_bands.eigenvalues,
        )
        return result

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

    if self_energy is not None:
        gamma_E: Float[Array, " E"] = evaluate_self_energy(
            energy_axis, self_energy
        )

        def _single_band_se(
            energy: Float[Array, " "],
            bi: Float[Array, " "],
        ) -> Float[Array, " E"]:
            """Compute one band contribution with energy-dependent gamma.

            The function computes the Fermi-Dirac occupation at the band
            energy. It then computes a Voigt profile at each energy point with
            ``gamma(E)`` from the self-energy model. The band intensity scales
            the result.

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

            The function applies ``jax.vmap`` to ``_single_band_se`` along the
            band axis. It sums the resulting (B, E) array to produce the total
            intensity at this k-point.

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
            """Compute one band contribution with constant gamma.

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

            The function applies ``jax.vmap`` to ``_single_band`` along the
            band axis. It sums the resulting (B, E) array to produce the total
            intensity at this k-point.

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

    if dk is not None:
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
