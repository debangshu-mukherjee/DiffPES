"""Assemble chunked occupied intrinsic spectral intensity.

Extended Summary
----------------
This module streams matrix-element sources through bounded spectral chunks.

Routine Listings
----------------
:func:`assemble_spectral_intensity_bands_chunk`
    Assemble occupied intrinsic intensity from eigenvalues and band weights.
:func:`assemble_spectral_intensity_chunk`
    Assemble occupied intrinsic intensity from Hamiltonians and sources.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
from beartype import beartype
from beartype.typing import Any, Tuple
from jaxtyping import Array, Bool, Complex128, Float64, jaxtyped

from diffpes.constants import G_PARALLEL_ATOL_INV_ANG
from diffpes.matrixel import (
    contract_polarization,
    orbital_transition_channels,
    transition_source,
)
from diffpes.radial import momentum_inv_ang_to_bohr_inv, radial_bvals
from diffpes.types import (
    ScalarBool,
    ScalarFloat,
    SelfEnergyModel,
    TransitionSourceSchedule,
)

from .broadening import fermi_dirac
from .kinematics import kz_from_inner_potential
from .retarded_self_energy import evaluate_self_energy
from .spectral_eigen import (
    _checked_eigenvalue_domain,
    _spectral_intensity_eigen_unchecked,
)
from .spectral_resolvent import (
    _checked_spectral_hamiltonian,
    _summed_spectral_intensity_resolvent_unchecked,
)


def _sampled_fermi_occupation(
    omega_rel_fermi_ev: Float64[Array, " n_chunk"],
    temperature_k: ScalarFloat,
) -> Float64[Array, " n_chunk"]:
    """PRIVATE: Evaluate occupation on the sampled relative-energy axis.

    Notes
    -----
    Vectorization evaluates the shared scalar Fermi primitive at every
    sampled energy and a zero relative chemical potential.
    """
    occupation: Float64[Array, " n_chunk"] = jax.vmap(
        lambda omega: fermi_dirac(omega, 0.0, temperature_k)
    )(omega_rel_fermi_ev)
    return occupation


@jaxtyped(typechecker=beartype)
def assemble_spectral_intensity_chunk(  # noqa: DOC502, DOC503 -- traced guards.
    hamiltonians_ev: Complex128[Array, "n_k n_orb n_orb"],
    transition_sources: Complex128[Array, "n_k n_chunk n_out n_orb"],
    omega_rel_fermi_ev: Float64[Array, " n_chunk"],
    self_energy: SelfEnergyModel,
    fermi_energy_ev: Float64[Array, ""],
    temperature_k: ScalarFloat,
    eta: ScalarFloat = 1.0e-4,
) -> Float64[Array, "n_k n_chunk"]:
    """Assemble occupied intrinsic intensity from Hamiltonians and sources.

    The degeneracy-safe path shifts each absolute Hamiltonian by the Fermi
    energy exactly once. It evaluates the causal self-energy once on the
    sampled relative-energy grid. It multiplies the spectral function by the
    Fermi occupation at those sampled energies.

    :see: :class:`~.test_spectral.TestAssembleSpectralIntensityChunk`

    Parameters
    ----------
    hamiltonians_ev : Complex128[Array, "n_k n_orb n_orb"]
        Absolute-energy Hermitian Hamiltonians in eV.
    transition_sources : Complex128[Array, "n_k n_chunk n_out n_orb"]
        Nonempty outgoing-channel source kets for each ``(k, omega)``.
        The code solves every channel independently; ``n_out=1`` is spinless.
    omega_rel_fermi_ev : Float64[Array, " n_chunk"]
        Sampled energies ``E - E_F`` in eV.
    self_energy : SelfEnergyModel
        Validated causal self-energy carrier on the relative-energy axis.
    fermi_energy_ev : Float64[Array, ""]
        Absolute Fermi energy subtracted from every Hamiltonian once.
    temperature_k : ScalarFloat
        Finite, strictly positive sample temperature in kelvin.
    eta : ScalarFloat, optional
        Positive resolvent regulator in eV. Default is ``1e-4``.

    Returns
    -------
    intensity : Float64[Array, "n_k n_chunk"]
        Intrinsic ``A(k, omega) f_FD(omega, T)`` in inverse eV.

    Raises
    ------
    ValueError
        If the outgoing-channel axis is empty.
    EquinoxRuntimeError
        If any numerical input violates the finite, Hermitian, causal, or
        positive-temperature contract.

    Notes
    -----
    The operation contains no detector convolution, count normalization, or
    background. Peak live solve storage scales as approximately
    ``16 * n_k * n_chunk * n_out * n_orb**2`` bytes in complex128. Scan static
    omega chunks and checkpoint this function. Use the eigen path for long
    nondegenerate paths. Use the resolvent at degeneracies or for Hamiltonian
    gradients.
    """
    if transition_sources.shape[2] == 0:
        raise ValueError("transition_sources n_out axis must be nonempty")
    checked_fermi: Float64[Array, ""] = eqx.error_if(
        fermi_energy_ev,
        ~jnp.isfinite(fermi_energy_ev),
        "assemble_spectral_intensity_chunk: fermi_energy_ev must be finite",
    )
    checked_omega: Float64[Array, " n_chunk"] = eqx.error_if(
        omega_rel_fermi_ev,
        ~jnp.all(jnp.isfinite(omega_rel_fermi_ev)),
        "assemble_spectral_intensity_chunk: omega must be finite",
    )
    checked_sources: Complex128[Array, "n_k n_chunk n_out n_orb"] = (
        eqx.error_if(
            transition_sources,
            ~jnp.all(jnp.isfinite(transition_sources)),
            "assemble_spectral_intensity_chunk: transition_sources must be "
            "finite",
        )
    )
    checked_hamiltonians: Complex128[Array, "n_k n_orb n_orb"] = jax.vmap(
        lambda hamiltonian: _checked_spectral_hamiltonian(
            hamiltonian,
            context="assemble_spectral_intensity_chunk",
        )
    )(hamiltonians_ev)
    identity: Complex128[Array, "n_orb n_orb"] = jnp.eye(
        hamiltonians_ev.shape[-1], dtype=jnp.complex128
    )
    hamiltonians_rel: Complex128[Array, "n_k n_orb n_orb"] = (
        checked_hamiltonians - checked_fermi * identity[None, :, :]
    )
    sigma: Complex128[Array, " n_chunk"] = evaluate_self_energy(
        checked_omega,
        self_energy,
    )
    eta_array: Float64[Array, ""] = jnp.asarray(eta, dtype=jnp.float64)
    eta_checked: Float64[Array, ""] = eqx.error_if(
        eta_array,
        ~jnp.isfinite(eta_array) | (eta_array <= 0.0),
        "assemble_spectral_intensity_chunk: eta must be finite and positive",
    )
    spectral: Float64[Array, "n_k n_chunk"] = jax.vmap(
        lambda hamiltonian, sources: jax.vmap(
            _summed_spectral_intensity_resolvent_unchecked,
            in_axes=(None, 0, 0, 0, None),
        )(hamiltonian, sources, checked_omega, sigma, eta_checked)
    )(hamiltonians_rel, checked_sources)
    occupation: Float64[Array, " n_chunk"] = _sampled_fermi_occupation(
        checked_omega,
        temperature_k,
    )
    intensity: Float64[Array, "n_k n_chunk"] = spectral * occupation[None, :]
    return intensity


@jaxtyped(typechecker=beartype)
def assemble_spectral_intensity_bands_chunk(  # noqa: DOC502 -- traced guards.
    eigenvalues_ev: Float64[Array, "n_k n_bands"],
    band_weights: Float64[Array, "n_k n_chunk n_bands"],
    omega_rel_fermi_ev: Float64[Array, " n_chunk"],
    self_energy: SelfEnergyModel,
    fermi_energy_ev: Float64[Array, ""],
    temperature_k: ScalarFloat,
    eta: ScalarFloat = 1.0e-4,
    *,
    allow_degenerate_value_only: ScalarBool = False,
) -> Float64[Array, "n_k n_chunk"]:
    """Assemble occupied intrinsic intensity from eigenvalues and band weights.

    This nondegenerate fast path shifts absolute eigenvalues by the Fermi
    energy exactly once and sums gauge-invariant Lorentzian band weights.
    The code evaluates occupation at sampled omega, never at a band eigenvalue.

    :see: :class:`~.test_spectral.TestAssembleSpectralIntensityBandsChunk`

    Parameters
    ----------
    eigenvalues_ev : Float64[Array, "n_k n_bands"]
        Absolute band energies in eV.
    band_weights : Float64[Array, "n_k n_chunk n_bands"]
        Explicit finite, nonnegative transition weights for each sample.
    omega_rel_fermi_ev : Float64[Array, " n_chunk"]
        Sampled energies ``E - E_F`` in eV.
    self_energy : SelfEnergyModel
        Validated causal self-energy carrier on the relative-energy axis.
    fermi_energy_ev : Float64[Array, ""]
        Absolute Fermi energy subtracted from every eigenvalue once.
    temperature_k : ScalarFloat
        Finite, strictly positive sample temperature in kelvin.
    eta : ScalarFloat, optional
        Positive regulator in eV. Default is ``1e-4``.
    allow_degenerate_value_only : ScalarBool, optional
        Admit exact or near-degenerate rows only for primal compatibility
        checks with already-formed complete invariant weights. Default is
        ``False``.

    Returns
    -------
    intensity : Float64[Array, "n_k n_chunk"]
        Intrinsic ``A(k, omega) f_FD(omega, T)`` in inverse eV.

    Raises
    ------
    EquinoxRuntimeError
        If an input is non-finite, a weight is negative, or a physical width
        or temperature is not strictly positive. Also raised when any band
        gap is below ``1e3 * EPS_DEG`` unless value-only evaluation is
        explicit.

    Notes
    -----
    The eigen route amortizes one eigendecomposition over all sampled
    energies. Its differentiated domain requires every adjacent band gap to
    be at least ``1e3 * EPS_DEG``. The explicit value-only exception emits no
    derivative claim. The function performs no convolution, normalization,
    or detector response; the canonical detector driver owns those operations.
    """
    checked_eigenvalues: Float64[Array, "n_k n_bands"] = eqx.error_if(
        eigenvalues_ev,
        ~jnp.all(jnp.isfinite(eigenvalues_ev)),
        "assemble_spectral_intensity_bands_chunk: eigenvalues must be finite",
    )
    checked_eigenvalues = _checked_eigenvalue_domain(
        checked_eigenvalues,
        allow_degenerate_value_only,
        context="assemble_spectral_intensity_bands_chunk",
    )
    checked_weights: Float64[Array, "n_k n_chunk n_bands"] = eqx.error_if(
        band_weights,
        ~jnp.all(jnp.isfinite(band_weights) & (band_weights >= 0.0)),
        "assemble_spectral_intensity_bands_chunk: weights must be finite "
        "and nonnegative",
    )
    checked_fermi: Float64[Array, ""] = eqx.error_if(
        fermi_energy_ev,
        ~jnp.isfinite(fermi_energy_ev),
        "assemble_spectral_intensity_bands_chunk: fermi energy must be finite",
    )
    checked_omega: Float64[Array, " n_chunk"] = eqx.error_if(
        omega_rel_fermi_ev,
        ~jnp.all(jnp.isfinite(omega_rel_fermi_ev)),
        "assemble_spectral_intensity_bands_chunk: omega must be finite",
    )
    eigenvalues_rel: Float64[Array, "n_k n_bands"] = (
        checked_eigenvalues - checked_fermi
    )
    sigma: Complex128[Array, " n_chunk"] = evaluate_self_energy(
        checked_omega,
        self_energy,
    )
    eta_array: Float64[Array, ""] = jnp.asarray(eta, dtype=jnp.float64)
    eta_checked: Float64[Array, ""] = eqx.error_if(
        eta_array,
        ~jnp.isfinite(eta_array) | (eta_array <= 0.0),
        "assemble_spectral_intensity_bands_chunk: eta must be finite and "
        "positive",
    )
    spectral: Float64[Array, "n_k n_chunk"] = jax.vmap(
        lambda eigenvalues, weights: jax.vmap(
            _spectral_intensity_eigen_unchecked,
            in_axes=(None, 0, 0, 0, None),
        )(eigenvalues, weights, checked_omega, sigma, eta_checked)
    )(eigenvalues_rel, checked_weights)
    occupation: Float64[Array, " n_chunk"] = _sampled_fermi_occupation(
        checked_omega,
        temperature_k,
    )
    intensity: Float64[Array, "n_k n_chunk"] = spectral * occupation[None, :]
    return intensity


def _validate_transition_source_schedule(
    schedule: TransitionSourceSchedule,
    *,
    n_k_max: int,
    n_omega_max: int,
    n_orb: int,
) -> None:
    """PRIVATE: Validate the static axes of one padded source schedule.

    Notes
    -----
    Python shape checks run before tracing. The schedule must also share one
    orbital basis and radial-shell partition across its carriers.
    """
    if (
        schedule.k_i_cart.shape != (n_k_max, 3)
        or schedule.final_norm.shape != (n_omega_max,)
        or schedule.emission_energy_valid.shape != (n_omega_max,)
        or schedule.positions_cart.shape != (n_orb, 3)
        or schedule.depths.shape != (n_orb,)
        or schedule.polarization_sample_cart.shape != (3,)
        or schedule.mean_free_path_ang.ndim != 0
        or len(schedule.radial.basis.n) != n_orb
    ):
        raise ValueError(
            "transition source schedule axes must match the padded spectral "
            "and orbital dimensions"
        )
    if (
        schedule.radial.basis != schedule.matrix_element.basis
        or schedule.radial.radial_shell_index
        != schedule.matrix_element.radial_shell_index
    ):
        raise ValueError(
            "transition source radial and matrix-element carriers must share "
            "one basis and shell partition"
        )


def _transition_sources_for_block(
    schedule: TransitionSourceSchedule,
    k_i_block: Float64[Array, "k_chunk 3"],
    k_f_block: Float64[Array, "k_chunk omega_chunk 3"],
    valid_block: Bool[Array, "k_chunk omega_chunk"],
) -> Complex128[Array, "k_chunk omega_chunk n_spin n_orb"]:
    """PRIVATE: Build only one live matrix-element source block.

    The helper replaces invalid padding before the source primitives run. It
    restores exact zeros afterward. Every physically valid final momentum must
    be finite, nonzero, and on the registered zero-umklapp in-plane seam.

    Notes
    -----
    The energy-axis vectorization keeps only one chunk of radial values,
    transition channels, and outgoing sources live at a time.
    """

    def one_energy(
        final_momentum: Float64[Array, "k_chunk 3"],
        valid: Bool[Array, " k_chunk"],
    ) -> Complex128[Array, "k_chunk n_spin n_orb"]:
        """Construct the outgoing-spin source rows at one omega."""
        safe_initial: Float64[Array, "k_chunk 3"] = jnp.where(
            valid[:, None], k_i_block, 0.0
        )
        filler: Float64[Array, "k_chunk 3"] = jnp.broadcast_to(
            jnp.asarray([0.0, 0.0, 1.0], dtype=jnp.float64),
            final_momentum.shape,
        )
        safe_final: Float64[Array, "k_chunk 3"] = jnp.where(
            valid[:, None], final_momentum, filler
        )
        final_norm: Float64[Array, " k_chunk"] = jnp.linalg.norm(
            safe_final, axis=-1
        )
        invalid_physical: Bool[Array, " k_chunk"] = valid & (
            ~jnp.all(jnp.isfinite(k_i_block), axis=-1)
            | ~jnp.all(jnp.isfinite(final_momentum), axis=-1)
            | (jnp.linalg.norm(final_momentum, axis=-1) <= 0.0)
            | jnp.any(
                jnp.abs(final_momentum[:, :2] - k_i_block[:, :2])
                > G_PARALLEL_ATOL_INV_ANG,
                axis=-1,
            )
        )
        safe_final = eqx.error_if(
            safe_final,
            jnp.any(invalid_physical),
            "valid streamed final momenta must be finite, nonzero, and on "
            "the G_parallel=0 seam",
        )
        momentum_bohr_inv: Float64[Array, " k_chunk"] = (
            momentum_inv_ang_to_bohr_inv(final_norm)
        )
        bvals: Complex128[Array, "k_chunk n_orb 2"] = radial_bvals(
            schedule.radial,
            momentum_bohr_inv,
            schedule.quadrature,
            schedule.final_state,
        )
        channels: Complex128[Array, "k_chunk n_spin n_orb_per_spin 3"] = (
            orbital_transition_channels(
                safe_initial,
                safe_final,
                schedule.positions_cart,
                schedule.depths,
                bvals,
                schedule.matrix_element,
                schedule.mean_free_path_ang,
                schedule.radial.basis,
            )
        )
        rows: Complex128[Array, "k_chunk n_spin n_orb_per_spin"] = (
            contract_polarization(
                channels,
                schedule.polarization_sample_cart,
            )
        )
        sources: Complex128[Array, "k_chunk n_spin n_orb"] = transition_source(
            rows
        )
        masked_sources: Complex128[Array, "k_chunk n_spin n_orb"] = jnp.where(
            valid[:, None, None],
            sources,
            0.0,
        )
        return masked_sources

    sources: Complex128[Array, "k_chunk omega_chunk n_spin n_orb"] = jax.vmap(
        one_energy, in_axes=(1, 1), out_axes=1
    )(
        k_f_block,
        valid_block,
    )
    return sources


def _stream_spectral_intensity(  # noqa: DOC503, PLR0913, PLR0915 -- scan contract.
    hamiltonians_ev: Complex128[Array, "n_k_max n_orb n_orb"],
    omega_rel_fermi_ev: Float64[Array, " n_omega_max"],
    k_valid: Bool[Array, " n_k_max"],
    omega_valid: Bool[Array, " n_omega_max"],
    transition_schedule: TransitionSourceSchedule,
    self_energy: SelfEnergyModel,
    fermi_energy_ev: Float64[Array, ""],
    temperature_k: ScalarFloat,
    eta: ScalarFloat = 1.0e-4,
    *,
    k_chunk: int = 32,
    omega_chunk: int = 32,
    checkpoint: bool = True,
) -> Float64[Array, "n_k_max n_omega_max"]:
    """PRIVATE: Stream padded chunks without a ``(K,E,B)`` source.

    Parameters
    ----------
    hamiltonians_ev : Complex128[Array, "n_k_max n_orb n_orb"]
        Padded absolute-energy Hermitian Hamiltonians in eV.
    omega_rel_fermi_ev : Float64[Array, " n_omega_max"]
        Padded sampled relative-energy axis in eV.
    k_valid : Bool[Array, " n_k_max"]
        Validity mask for the padded k axis.
    omega_valid : Bool[Array, " n_omega_max"]
        Validity mask for the padded energy axis.
    transition_schedule : TransitionSourceSchedule
        Detector kinematics and source carriers used to construct only the
        current source block.
    self_energy : SelfEnergyModel
        Validated causal self-energy carrier.
    fermi_energy_ev : Float64[Array, ""]
        Absolute Fermi energy in eV.
    temperature_k : ScalarFloat
        Finite, strictly positive temperature in kelvin.
    eta : ScalarFloat, optional
        Positive regulator in eV. Default is ``1e-4``.
    k_chunk : int, optional
        Positive static k chunk size. Default is 32.
    omega_chunk : int, optional
        Positive static energy chunk size. Default is 32.
    checkpoint : bool, optional
        Static selector for rematerializing each two-dimensional chunk.

    Returns
    -------
    intensity : Float64[Array, "n_k_max n_omega_max"]
        Masked intrinsic intensity on the complete padded schedule.

    Raises
    ------
    ValueError
        If padded axes, source carriers, or chunk sizes are inconsistent.
    EquinoxRuntimeError
        If a physically valid final momentum or traced carrier value leaves
        its registered domain.

    Notes
    -----
    Callers keep padded shapes and the chunk schedule fixed across a sweep;
    only masks and physical leaves vary. Each scan step constructs radial
    channels, polarized outgoing-spin source kets, resolvent solutions, and
    the spin-incoherent reduction for one ``(k_chunk, omega_chunk)`` block.
    No complete ``(K, E, B)`` transition tensor exists. Checkpointing bounds
    reverse-mode tape without changing values.
    """
    if type(k_chunk) is not int or k_chunk <= 0:
        raise ValueError("k_chunk must be a positive integer")
    if type(omega_chunk) is not int or omega_chunk <= 0:
        raise ValueError("omega_chunk must be a positive integer")
    n_k_max: int = hamiltonians_ev.shape[0]
    n_omega_max: int = omega_rel_fermi_ev.shape[0]
    n_orb: int = hamiltonians_ev.shape[-1]
    batch_matrix_ndim: int = 3
    if (
        hamiltonians_ev.ndim != batch_matrix_ndim
        or hamiltonians_ev.shape[-2] != n_orb
        or k_valid.shape != (n_k_max,)
        or omega_valid.shape != (n_omega_max,)
    ):
        raise ValueError("streamed spectral padded axes are inconsistent")
    _validate_transition_source_schedule(
        transition_schedule,
        n_k_max=n_k_max,
        n_omega_max=n_omega_max,
        n_orb=n_orb,
    )
    checked_final_norm: Float64[Array, " n_omega_max"] = eqx.error_if(
        transition_schedule.final_norm,
        ~jnp.all(jnp.isfinite(transition_schedule.final_norm))
        | jnp.any(transition_schedule.final_norm < 0.0)
        | jnp.any(
            transition_schedule.emission_energy_valid
            & (transition_schedule.final_norm == 0.0)
        ),
        "streamed final-momentum magnitudes must be finite and nonnegative; "
        "active magnitudes must be strictly positive",
    )
    if n_k_max % k_chunk:
        raise ValueError("k_chunk must divide the padded k axis")
    if n_omega_max % omega_chunk:
        raise ValueError("omega_chunk must divide the padded omega axis")
    n_k_blocks: int = n_k_max // k_chunk
    n_omega_blocks: int = n_omega_max // omega_chunk
    hamiltonian_blocks: Complex128[Array, "n_k_block k_chunk n_orb n_orb"] = (
        jnp.reshape(
            hamiltonians_ev,
            (n_k_blocks, k_chunk, n_orb, n_orb),
        )
    )
    initial_blocks: Float64[Array, "n_k_block k_chunk 3"] = jnp.reshape(
        transition_schedule.k_i_cart,
        (n_k_blocks, k_chunk, 3),
    )
    final_norm_blocks: Float64[Array, "n_omega_block omega_chunk"] = (
        jnp.reshape(
            checked_final_norm,
            (n_omega_blocks, omega_chunk),
        )
    )
    emission_energy_blocks: Bool[Array, "n_omega_block omega_chunk"] = (
        jnp.reshape(
            transition_schedule.emission_energy_valid,
            (n_omega_blocks, omega_chunk),
        )
    )
    omega_blocks: Float64[Array, "n_omega_block omega_chunk"] = jnp.reshape(
        omega_rel_fermi_ev,
        (n_omega_blocks, omega_chunk),
    )
    k_mask_blocks: Bool[Array, "n_k_block k_chunk"] = jnp.reshape(
        k_valid,
        (n_k_blocks, k_chunk),
    )
    omega_mask_blocks: Bool[Array, "n_omega_block omega_chunk"] = jnp.reshape(
        omega_valid,
        (n_omega_blocks, omega_chunk),
    )

    def assemble_block(
        hamiltonian_block: Complex128[Array, "k_chunk n_orb n_orb"],
        k_i_block: Float64[Array, "k_chunk 3"],
        final_norm_block: Float64[Array, " omega_chunk"],
        emission_energy_block: Bool[Array, " omega_chunk"],
        k_mask: Bool[Array, " k_chunk"],
        omega_mask: Bool[Array, " omega_chunk"],
        omega_block: Float64[Array, " omega_chunk"],
    ) -> Float64[Array, "k_chunk omega_chunk"]:
        """Compute one live block from reconstructed kinematics and solves."""
        parallel_sq: Float64[Array, " k_chunk"] = jnp.sum(
            k_i_block[:, :2] * k_i_block[:, :2], axis=-1
        )
        final_kz: Float64[Array, "k_chunk omega_chunk"]
        emission_valid: Bool[Array, "k_chunk omega_chunk"]
        if transition_schedule.inner_potential_geometry is None:
            normal_sq: Float64[Array, "k_chunk omega_chunk"] = (
                final_norm_block[None, :] * final_norm_block[None, :]
                - parallel_sq[:, None]
            )
            emission_valid = emission_energy_block[None, :] & (normal_sq > 0.0)
            provisional_valid: Bool[Array, "k_chunk omega_chunk"] = (
                k_mask[:, None] & omega_mask[None, :] & emission_valid
            )
            safe_normal_sq: Float64[Array, "k_chunk omega_chunk"] = jnp.where(
                provisional_valid,
                normal_sq,
                1.0,
            )
            final_kz = jnp.where(
                provisional_valid,
                jnp.sqrt(safe_normal_sq),
                0.0,
            )
        else:
            internal_kz: Complex128[Array, "k_chunk omega_chunk"]
            propagating: Bool[Array, "k_chunk omega_chunk"]
            internal_kz, propagating = kz_from_inner_potential(
                transition_schedule.inner_potential_geometry.photon_energy_ev,
                transition_schedule.inner_potential_geometry.work_function_ev,
                transition_schedule.inner_potential_geometry.inner_potential_ev,
                omega_block[None, :],
                jnp.sqrt(parallel_sq)[:, None],
            )
            emission_valid = emission_energy_block[None, :] & propagating
            provisional_valid = (
                k_mask[:, None] & omega_mask[None, :] & emission_valid
            )
            final_kz = jnp.where(
                provisional_valid,
                jnp.real(internal_kz),
                0.0,
            )
        valid_block: Bool[Array, "k_chunk omega_chunk"] = (
            k_mask[:, None] & omega_mask[None, :] & emission_valid
        )
        final_kx: Float64[Array, "k_chunk omega_chunk"] = jnp.broadcast_to(
            k_i_block[:, 0, None], final_kz.shape
        )
        final_ky: Float64[Array, "k_chunk omega_chunk"] = jnp.broadcast_to(
            k_i_block[:, 1, None], final_kz.shape
        )
        k_f_block: Float64[Array, "k_chunk omega_chunk 3"] = jnp.stack(
            (final_kx, final_ky, final_kz), axis=-1
        )
        sources: Complex128[Array, "k_chunk omega_chunk n_spin n_orb"] = (
            _transition_sources_for_block(
                transition_schedule,
                k_i_block,
                k_f_block,
                valid_block,
            )
        )
        intensity: Float64[Array, "k_chunk omega_chunk"] = (
            assemble_spectral_intensity_chunk(
                hamiltonian_block,
                sources,
                omega_block,
                self_energy,
                fermi_energy_ev,
                temperature_k,
                eta,
            )
        )
        masked_intensity: Float64[Array, "k_chunk omega_chunk"] = jnp.where(
            valid_block,
            intensity,
            0.0,
        )
        return masked_intensity

    block_function: Any = (
        jax.checkpoint(assemble_block) if checkpoint else assemble_block
    )

    def scan_k_block(
        carry: None,
        arguments: Tuple[
            Complex128[Array, "k_chunk n_orb n_orb"],
            Float64[Array, "k_chunk 3"],
            Bool[Array, " k_chunk"],
        ],
    ) -> Tuple[
        None,
        Float64[Array, "n_omega_block k_chunk omega_chunk"],
    ]:
        """Stream every energy block for one k block."""
        hamiltonian_block: Complex128[Array, "k_chunk n_orb n_orb"]
        k_i_block: Float64[Array, "k_chunk 3"]
        k_mask: Bool[Array, " k_chunk"]
        (
            hamiltonian_block,
            k_i_block,
            k_mask,
        ) = arguments

        def scan_omega_block(
            inner_carry: None,
            inner_arguments: Tuple[
                Float64[Array, " omega_chunk"],
                Float64[Array, " omega_chunk"],
                Bool[Array, " omega_chunk"],
                Bool[Array, " omega_chunk"],
            ],
        ) -> Tuple[None, Float64[Array, "k_chunk omega_chunk"]]:
            """Construct, assemble, and mask one omega block."""
            omega_block: Float64[Array, " omega_chunk"]
            final_norm_block: Float64[Array, " omega_chunk"]
            emission_energy_block: Bool[Array, " omega_chunk"]
            omega_mask: Bool[Array, " omega_chunk"]
            (
                omega_block,
                final_norm_block,
                emission_energy_block,
                omega_mask,
            ) = inner_arguments
            values: Float64[Array, "k_chunk omega_chunk"] = block_function(
                hamiltonian_block,
                k_i_block,
                final_norm_block,
                emission_energy_block,
                k_mask,
                omega_mask,
                omega_block,
            )
            result: Tuple[None, Float64[Array, "k_chunk omega_chunk"]] = (
                inner_carry,
                values,
            )
            return result

        outputs: Float64[Array, "n_omega_block k_chunk omega_chunk"]
        _, outputs = jax.lax.scan(
            scan_omega_block,
            None,
            (
                omega_blocks,
                final_norm_blocks,
                emission_energy_blocks,
                omega_mask_blocks,
            ),
        )
        result: Tuple[
            None,
            Float64[Array, "n_omega_block k_chunk omega_chunk"],
        ] = (carry, outputs)
        return result

    scanned: Float64[Array, "n_k_block n_omega_block k_chunk omega_chunk"]
    _, scanned = jax.lax.scan(
        scan_k_block,
        None,
        (
            hamiltonian_blocks,
            initial_blocks,
            k_mask_blocks,
        ),
    )
    intensity: Float64[Array, "n_k_max n_omega_max"] = jnp.reshape(
        jnp.transpose(scanned, (0, 2, 1, 3)),
        (n_k_max, n_omega_max),
    )
    return intensity


__all__: list[str] = [
    "assemble_spectral_intensity_bands_chunk",
    "assemble_spectral_intensity_chunk",
]
