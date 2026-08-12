r"""Assemble coherent orbital and band photoemission matrix elements.

Extended Summary
----------------
The module keeps the photoemission transition in orbital space until the
last possible stage. For outgoing-spin channel :math:`\sigma`, real
orbital :math:`a`, and real dipole channel :math:`d=(y,z,x)`, it evaluates

.. math::

   d^\sigma_{ad} =
   \sigma_a e^{-z_a/(2\lambda)}
   e^{i(\mathbf k_i-\mathbf k_f)\cdot\mathbf R_a}
   \sum_{c,y} B_{ac}e^{i\delta_{ac}}G_{acdy}Y_y(\hat{\mathbf k}_f).

Callers supply the vacuum final momentum explicitly. The module never
reconstructs it from the inner potential. Orbital positions use explicit
Wannier centres before atom-derived centres. Attenuation acts on amplitude
with a half exponent. Callers supply polarization in the sample frame for
the late linear contraction. Band projection uses the stored
basis-position-gauge coefficients without conjugating them. A modulus square
appears only in :func:`matrix_element_intensity`. Unresolved outgoing spins
sum incoherently there.

Cartesian recurrences evaluate the final harmonics as normalized solid
harmonics. The algorithm avoids azimuth construction and defines both pole
values and transverse directional derivatives. Spectral and detector-count
semantics are outside this module.

Routine Listings
----------------
:func:`assemble_orbital_transition_channels`
    Assemble the validated orbital transition tensor.
:func:`contract_polarization`
    Compute the sample-frame polarization contraction.
:func:`matrix_element_intensity`
    Sum outgoing-spin modulus squares exactly once.
:func:`orbital_transition_channels`
    Assemble coherent orbital transition channels.
:func:`project_band_channels`
    Compute band channels without conjugating orbital coefficients.
:func:`real_spherical_harmonics_cartesian_all`
    Evaluate all real spherical harmonics from Cartesian directions.
:func:`resolve_orbital_positions_cart`
    Resolve orbital centres in Cartesian Angstrom coordinates.
:func:`transition_source`
    Build conjugated outgoing-spin rows as full source kets.
"""

import math

import equinox as eqx
import jax.numpy as jnp
from beartype import beartype
from beartype.typing import Dict, List, Tuple
from jaxtyping import Array, Bool, Complex128, Float64, Int32, jaxtyped

from diffpes.constants import (
    CARTESIAN_COMPONENTS,
    EPS,
    G_PARALLEL_ATOL_INV_ANG,
    ISPIN2_BLOCKS,
    L_MAX,
    MATRIX_NDIM,
)
from diffpes.maths import channel_tables, polarization_cart_to_real
from diffpes.radial import momentum_inv_ang_to_bohr_inv, radial_bvals
from diffpes.types import (
    DiagonalizedBands,
    ExperimentGeometry,
    FinalStateSpec,
    MatrixElementParams,
    OrbitalBasis,
    RadialQuadratureSpec,
    RadialSpec,
)


def _basis_key(basis: OrbitalBasis) -> Tuple[Tuple[object, ...], ...]:
    """PRIVATE: Return the exact static identity of an orbital basis.

    Notes
    -----
    The identity hashes only static basis structure, never values.
    """
    key: Tuple[Tuple[object, ...], ...] = (
        basis.atom_indices,
        basis.n,
        basis.l,
        basis.m,
        basis.spin,
        basis.labels,
    )
    return key


def _spin_layout(basis: OrbitalBasis) -> Tuple[int, int]:
    """PRIVATE: Validate and return ``(n_spin, n_orbitals_per_spin)``.

    Notes
    -----
    The layout derives from the basis length and the spin flag.
    """
    n_orbitals: int = len(basis.n)
    if not basis.spin:
        layout: Tuple[int, int] = (1, n_orbitals)
        return layout
    if n_orbitals % ISPIN2_BLOCKS != 0:
        message: str = "a spinor basis must contain two equal spin blocks"
        raise ValueError(message)
    n_spatial: int = n_orbitals // ISPIN2_BLOCKS
    expected_spin: Tuple[int, ...] = (-1,) * n_spatial + (1,) * n_spatial
    if basis.spin != expected_spin:
        message = "spinor basis must use block-down then block-up ordering"
        raise ValueError(message)
    paired_fields: Tuple[Tuple[object, ...], ...] = (
        basis.atom_indices,
        basis.n,
        basis.l,
        basis.m,
    )
    if any(field[:n_spatial] != field[n_spatial:] for field in paired_fields):
        message = "spin blocks must contain paired spatial orbitals"
        raise ValueError(message)
    layout = (ISPIN2_BLOCKS, n_spatial)
    return layout  # noqa: RET504 -- assign-before-return is required.


def _solid_harmonic_component(
    direction_unit: Float64[Array, "... 3"],
    degree: int,
    order: int,
) -> Float64[Array, " ..."]:
    """PRIVATE: Evaluate one normalized real solid harmonic on the unit sphere.

    Notes
    -----
    The component follows the fixed real-harmonic convention.
    """
    absolute_order: int = abs(order)
    x_component: Float64[Array, " ..."] = direction_unit[..., 0]
    y_component: Float64[Array, " ..."] = direction_unit[..., 1]
    z_component: Float64[Array, " ..."] = direction_unit[..., 2]
    transverse: Complex128[Array, " ..."] = x_component.astype(
        jnp.complex128
    ) + 1j * y_component.astype(jnp.complex128)
    double_factorial: float = 1.0
    factor_index: int
    for factor_index in range(1, absolute_order + 1):
        double_factorial *= 2.0 * factor_index - 1.0
    sectoral_complex: Complex128[Array, " ..."] = transverse**absolute_order
    if order > 0:
        sectoral: Float64[Array, " ..."] = double_factorial * jnp.real(
            sectoral_complex
        )
    elif order < 0:
        sectoral = double_factorial * jnp.imag(sectoral_complex)
    else:
        sectoral = jnp.ones_like(z_component)

    polynomial: Float64[Array, " ..."] = sectoral
    if degree > absolute_order:
        previous_two: Float64[Array, " ..."] = sectoral
        previous_one: Float64[Array, " ..."] = (
            (2.0 * absolute_order + 1.0) * z_component * sectoral
        )
        if degree == absolute_order + 1:
            polynomial = previous_one
        else:
            recurrence_degree: int
            for recurrence_degree in range(absolute_order + 2, degree + 1):
                current: Float64[Array, " ..."] = (
                    (2.0 * recurrence_degree - 1.0)
                    * z_component
                    * previous_one
                    - (recurrence_degree + absolute_order - 1.0) * previous_two
                ) / (recurrence_degree - absolute_order)
                previous_two, previous_one = previous_one, current
            polynomial = previous_one

    normalization: float = math.sqrt(
        (2 * degree + 1)
        / (4.0 * math.pi)
        * math.factorial(degree - absolute_order)
        / math.factorial(degree + absolute_order)
    )
    if order != 0:
        normalization *= math.sqrt(2.0)
    value: Float64[Array, " ..."] = normalization * polynomial
    return value


def _orbital_phase_indices(
    me_params: MatrixElementParams,
) -> Tuple[Tuple[int, int], ...]:
    """PRIVATE: Return each branch phase index or its zero sentinel.

    Notes
    -----
    A zero sentinel marks the branch without a free phase.
    """
    compact_index: Dict[Tuple[int, int], int] = {
        key: index for index, key in enumerate(me_params.phase_channel_keys)
    }
    zero_sentinel: int = len(me_params.phase_channel_keys)
    result: Tuple[Tuple[int, int], ...] = tuple(
        (
            compact_index.get(
                (shell, angular - 1),
                zero_sentinel,
            ),
            compact_index[(shell, angular + 1)],
        )
        for shell, angular in zip(
            me_params.radial_shell_index,
            me_params.basis.l,
            strict=True,
        )
    )
    return result


@jaxtyped(typechecker=beartype)
def resolve_orbital_positions_cart(
    bands: DiagonalizedBands,
) -> Float64[Array, "n_orb 3"]:
    """Resolve orbital centres in Cartesian Angstrom coordinates.

    Preserve the basis-position gauge at the structural boundary.

    Explicit fractional Wannier centres remain authoritative.  Otherwise,
    ``basis.atom_indices`` gathers fractional atom positions.  One final
    matrix product maps the selected fractional array through the lattice.

    :see: :class:`~.test_transition.TestResolveOrbitalPositionsCart`

    Parameters
    ----------
    bands : DiagonalizedBands
        Eigensystem with geometry, basis, and optional orbital centres.

    Returns
    -------
    positions_cart : Float64[Array, "n_orb 3"]
        Orbital centres in Cartesian Angstrom coordinates.

    Notes
    -----
    Select one fractional source before applying the lattice multiplication.
    """
    positions_fractional: Float64[Array, "n_orb 3"]
    if bands.orbital_positions is not None:
        positions_fractional = bands.orbital_positions
    else:
        atom_indices: Int32[Array, " n_orb"] = jnp.asarray(
            bands.basis.atom_indices,
            dtype=jnp.int32,
        )
        positions_fractional = bands.geometry.positions[atom_indices]
    positions_cart: Float64[Array, "n_orb 3"] = (
        positions_fractional @ bands.geometry.lattice
    )
    return positions_cart


@jaxtyped(typechecker=beartype)
def real_spherical_harmonics_cartesian_all(  # noqa: DOC503
    direction_cart: Float64[Array, "... 3"],
    l_max: int,
) -> Float64[Array, "... n_y"]:
    """Evaluate all real spherical harmonics from Cartesian directions.

    Avoid the azimuthal coordinate singularity at normal emission.

    The degree-then-order index is ``l*l + l + m``.  The implementation uses
    homogeneous Cartesian polynomials and therefore never reconstructs an
    azimuth at the poles.

    :see: :class:`~.test_transition.TestRealSphericalHarmonicsCartesianAll`

    Parameters
    ----------
    direction_cart : Float64[Array, "... 3"]
        Nonzero Cartesian direction vectors.
    l_max : int
        Static maximum degree in the inclusive range zero through five.

    Returns
    -------
    harmonics : Float64[Array, "... n_y"]
        Normalized real harmonics with ``n_y = (l_max + 1)**2``.

    Raises
    ------
    ValueError
        If the static degree or trailing Cartesian axis is invalid.
    EquinoxRuntimeError
        If a direction is non-finite or zero.

    Notes
    -----
    Normalize each vector and evaluate finite Cartesian recurrences.
    """
    if type(l_max) is not int or not 0 <= l_max <= L_MAX + 1:
        message: str = "l_max must be an integer in [0, 5]"
        raise ValueError(message)
    if (
        direction_cart.ndim < 1
        or direction_cart.shape[-1] != CARTESIAN_COMPONENTS
    ):
        message = "direction_cart must have a trailing axis of length 3"
        raise ValueError(message)
    directions: Float64[Array, "... 3"] = jnp.asarray(
        direction_cart,
        dtype=jnp.float64,
    )
    norms: Float64[Array, " ..."] = jnp.linalg.norm(directions, axis=-1)
    directions = eqx.error_if(
        directions,
        ~jnp.all(jnp.isfinite(directions)) | jnp.any(norms <= 0.0),
        "Cartesian harmonic directions must be finite and nonzero",
    )
    safe_norms: Float64[Array, " ..."] = jnp.where(norms > 0.0, norms, 1.0)
    direction_unit: Float64[Array, "... 3"] = (
        directions / safe_norms[..., None]
    )
    values: List[Float64[Array, " ..."]] = []
    degree: int
    order: int
    for degree in range(l_max + 1):
        for order in range(-degree, degree + 1):
            values.append(
                _solid_harmonic_component(direction_unit, degree, order)
            )
    harmonics: Float64[Array, "... n_y"] = jnp.stack(values, axis=-1)
    return harmonics


@jaxtyped(typechecker=beartype)
def orbital_transition_channels(  # noqa: DOC503
    k_i_cart: Float64[Array, "n_k 3"],
    k_f_cart: Float64[Array, "n_k 3"],
    positions_cart: Float64[Array, "n_orb 3"],
    depths: Float64[Array, " n_orb"],
    bvals: Complex128[Array, "n_k n_orb 2"],
    me_params: MatrixElementParams,
    mean_free_path_ang: Float64[Array, ""],
    basis: OrbitalBasis,
) -> Complex128[Array, "n_k n_spin n_orb_per_spin 3"]:
    r"""Assemble coherent orbital transition channels.

    Keep every orbital factor at amplitude level.

    Position, attenuation, radial, atomic-phase, angular, and shell-scale
    factors remain complex amplitudes.  The function performs no modulus
    square and leaves outgoing spin as an explicit axis.

    :see: :class:`~.test_transition.TestOrbitalTransitionChannels`

    Parameters
    ----------
    k_i_cart : Float64[Array, "n_k 3"]
        Initial crystal momenta in sample Cartesian inverse Angstrom.
    k_f_cart : Float64[Array, "n_k 3"]
        Explicit vacuum final momenta in the same frame and units.
    positions_cart : Float64[Array, "n_orb 3"]
        Cartesian orbital centres in Angstrom.
    depths : Float64[Array, "n_orb"]
        Orbital depths below the surface in Angstrom.
    bvals : Complex128[Array, "n_k n_orb 2"]
        Radial channels in static order ``(l-1, l+1)``.
    me_params : MatrixElementParams
        Shell-shared amplitude scales and channel phase angles.
    mean_free_path_ang : Float64[Array, ""]
        Positive photoelectron intensity mean free path in Angstrom.
    basis : OrbitalBasis
        Static real-orbital and spin layout.

    Returns
    -------
    transition_channels : Complex128[Array, "n_k n_spin n_orb_per_spin 3"]
        Real dipole channels in order ``(y,z,x)``.

    Raises
    ------
    ValueError
        If static carriers or array axes disagree.
    EquinoxRuntimeError
        If traced physical values are invalid.

    Notes
    -----
    Derive a dense orbital branch view from compact physical phase coordinates.
    Contract both radial branches with one tensor sum.
    """
    n_spin: int
    n_spatial: int
    n_spin, n_spatial = _spin_layout(basis)
    n_orbitals: int = len(basis.n)
    n_kpoints: int = k_i_cart.shape[0]
    if _basis_key(me_params.basis) != _basis_key(basis):
        message: str = "matrix-element parameters and basis must agree"
        raise ValueError(message)
    if len(me_params.radial_shell_index) != n_orbitals:
        message = "matrix-element shell map must match the orbital basis"
        raise ValueError(message)
    if (
        k_i_cart.ndim != MATRIX_NDIM
        or k_f_cart.shape != k_i_cart.shape
        or k_i_cart.shape[-1] != CARTESIAN_COMPONENTS
        or positions_cart.shape != (n_orbitals, CARTESIAN_COMPONENTS)
        or depths.shape != (n_orbitals,)
        or bvals.shape != (n_kpoints, n_orbitals, 2)
        or mean_free_path_ang.ndim != 0
    ):
        message = "matrix-element array axes are inconsistent"
        raise ValueError(message)
    if (
        n_spin == ISPIN2_BLOCKS
        and me_params.radial_shell_index[:n_spatial]
        != me_params.radial_shell_index[n_spatial:]
    ):
        message = "spin blocks must share paired radial shells"
        raise ValueError(message)

    initial_momentum: Float64[Array, "n_k 3"] = jnp.asarray(
        k_i_cart,
        dtype=jnp.float64,
    )
    final_momentum: Float64[Array, "n_k 3"] = jnp.asarray(
        k_f_cart,
        dtype=jnp.float64,
    )
    positions: Float64[Array, "n_orb 3"] = jnp.asarray(
        positions_cart,
        dtype=jnp.float64,
    )
    depth_values: Float64[Array, " n_orb"] = jnp.asarray(
        depths,
        dtype=jnp.float64,
    )
    radial_values: Complex128[Array, "n_k n_orb 2"] = jnp.asarray(
        bvals,
        dtype=jnp.complex128,
    )
    mean_free_path: Float64[Array, ""] = jnp.asarray(
        mean_free_path_ang,
        dtype=jnp.float64,
    )
    checked_arrays: Float64[Array, ""] = jnp.asarray(0.0, dtype=jnp.float64)
    checked_arrays = eqx.error_if(
        checked_arrays,
        ~jnp.all(jnp.isfinite(initial_momentum))
        | ~jnp.all(jnp.isfinite(final_momentum))
        | ~jnp.all(jnp.isfinite(positions))
        | ~jnp.all(jnp.isfinite(depth_values))
        | ~jnp.all(jnp.isfinite(radial_values))
        | ~jnp.isfinite(mean_free_path)
        | (mean_free_path <= 0.0)
        | jnp.any(depth_values < -EPS),
        "matrix-element inputs must be finite with physical depth and mfp",
    )
    initial_momentum = initial_momentum + checked_arrays
    depth_nonnegative: Float64[Array, " n_orb"] = jnp.maximum(
        depth_values, 0.0
    )
    shell_indices: Int32[Array, " n_orb"] = jnp.asarray(
        me_params.radial_shell_index,
        dtype=jnp.int32,
    )
    sigma_shell: Float64[Array, " n_shell"] = eqx.error_if(
        me_params.sigma_shell,
        ~jnp.all(jnp.isfinite(me_params.sigma_shell)),
        "matrix-element shell scales must be finite",
    )
    compact_phase_angles: Float64[Array, " n_valid_phase"] = eqx.error_if(
        me_params.phase_shift_angles_shell,
        ~jnp.all(jnp.isfinite(me_params.phase_shift_angles_shell)),
        "matrix-element phase angles must be finite",
    )
    phase_indices: Int32[Array, " n_orb"] = jnp.asarray(
        _orbital_phase_indices(me_params),
        dtype=jnp.int32,
    )
    phase_angles: Float64[Array, "n_valid_phase_plus_zero"] = jnp.concatenate(
        (
            compact_phase_angles,
            jnp.zeros((1,), dtype=jnp.float64),
        )
    )
    orbital_scales: Float64[Array, " n_orb"] = sigma_shell[shell_indices]
    channel_phases: Complex128[Array, "n_orb 2"] = jnp.exp(
        1j * phase_angles[phase_indices]
    )
    attenuation: Float64[Array, " n_orb"] = jnp.exp(
        -depth_nonnegative / (2.0 * mean_free_path)
    )
    momentum_difference: Float64[Array, "n_k 3"] = (
        initial_momentum - final_momentum
    )
    position_phase: Complex128[Array, "n_k n_orb"] = jnp.exp(
        1j * jnp.einsum("kd,od->ko", momentum_difference, positions)
    )
    harmonics: Float64[Array, "n_k 36"] = (
        real_spherical_harmonics_cartesian_all(
            final_momentum,
            L_MAX + 1,
        )
    )
    coupling_coefficients: Float64[Array, "n_orb 2 3 36"]
    channel_valid: Float64[Array, "n_orb 2 3 36"]
    coupling_coefficients, channel_valid = channel_tables(basis)
    angular_channels: Complex128[Array, "n_k n_orb 3"] = jnp.einsum(
        "koc,oc,ocdy,ky->kod",
        radial_values,
        channel_phases,
        coupling_coefficients * channel_valid,
        harmonics,
    )
    prefactor: Complex128[Array, "n_k n_orb"] = (
        orbital_scales[None, :] * attenuation[None, :] * position_phase
    )
    flat_channels: Complex128[Array, "n_k n_orb 3"] = (
        prefactor[..., None] * angular_channels
    )
    transition_channels: Complex128[Array, "n_k n_spin n_orb_per_spin 3"] = (
        flat_channels.reshape(
            n_kpoints,
            n_spin,
            n_spatial,
            CARTESIAN_COMPONENTS,
        )
    )
    return transition_channels


@jaxtyped(typechecker=beartype)
def contract_polarization(
    transition_channels: Complex128[Array, "... 3"],
    polarization_sample_cart: Complex128[Array, " 3"],
) -> Complex128[Array, " ..."]:
    """Compute the sample-frame polarization contraction.

    Preserve generic complex optical phases through the late linear step.

    :see: :class:`~.test_transition.TestContractPolarization`

    Parameters
    ----------
    transition_channels : Complex128[Array, "... 3"]
        Real dipole channels in order ``(y,z,x)``.
    polarization_sample_cart : Complex128[Array, "3"]
        Cartesian complex polarization in order ``(x,y,z)``.

    Returns
    -------
    polarized_transition : Complex128[Array, "..."]
        Complex amplitude with the channel axis contracted once.

    Notes
    -----
    Permute Cartesian components into real-harmonic order before summing.
    """
    polarization: Complex128[Array, " 3"] = jnp.asarray(
        polarization_sample_cart,
        dtype=jnp.complex128,
    )
    polarization_real: Complex128[Array, " 3"] = polarization_cart_to_real(
        polarization
    )
    polarized_transition: Complex128[Array, " ..."] = jnp.sum(
        transition_channels * polarization_real,
        axis=-1,
    )
    return polarized_transition


@jaxtyped(typechecker=beartype)
def transition_source(
    transition_row: Complex128[Array, "... n_spin n_orb_per_spin"],
) -> Complex128[Array, "... n_spin n_orb"]:
    """Build conjugated outgoing-spin rows as full source kets.

    Retain separate sources for each outgoing-spin measurement channel.

    :see: :class:`~.test_transition.TestTransitionSource`

    Parameters
    ----------
    transition_row : Complex128[Array, "... n_spin n_orb_per_spin"]
        Polarization-contracted outgoing-spin orbital rows.

    Returns
    -------
    source : Complex128[Array, "... n_spin n_orb"]
        Conjugated source kets, with exact zero in the opposite spin block.

    Raises
    ------
    ValueError
        If the outgoing-spin axis is not one or two.

    Notes
    -----
    Multiply conjugated rows by a spin identity before flattening the blocks.
    """
    n_spin: int = transition_row.shape[-2]
    if n_spin not in (1, ISPIN2_BLOCKS):
        message: str = "transition rows must contain one or two spin channels"
        raise ValueError(message)
    n_spatial: int = transition_row.shape[-1]
    spin_identity: Float64[Array, "n_spin n_spin"] = jnp.eye(
        n_spin,
        dtype=jnp.float64,
    )
    blocked_source: Complex128[Array, "... n_spin n_spin n_orb_per_spin"] = (
        jnp.einsum(
            "...sa,st->...sta",
            jnp.conj(transition_row),
            spin_identity,
        )
    )
    source: Complex128[Array, "... n_spin n_orb"] = blocked_source.reshape(
        transition_row.shape[:-2] + (n_spin, n_spin * n_spatial)
    )
    return source


@jaxtyped(typechecker=beartype)
def project_band_channels(
    transition_channels: Complex128[Array, "n_k n_spin n_orb_per_spin 3"],
    eigenvectors: Complex128[Array, "n_k n_bands n_orb"],
) -> Complex128[Array, "n_k n_bands n_spin 3"]:
    """Compute band channels without conjugating orbital coefficients.

    Follow the stored ket-coefficient convention for generic complex bands.

    :see: :class:`~.test_transition.TestProjectBandChannels`

    Parameters
    ----------
    transition_channels : Complex128[Array, "n_k n_spin n_orb_per_spin 3"]
        Orbital-space transition channels.
    eigenvectors : Complex128[Array, "n_k n_bands n_orb"]
        Basis-position-gauge band coefficients.

    Returns
    -------
    band_channels : Complex128[Array, "n_k n_bands n_spin 3"]
        Band amplitudes with outgoing spin retained.

    Raises
    ------
    ValueError
        If the k-point or orbital axes disagree.

    Notes
    -----
    Reshape coefficients into spin blocks and apply one direct complex sum.
    """
    n_kpoints: int = transition_channels.shape[0]
    n_spin: int = transition_channels.shape[1]
    n_spatial: int = transition_channels.shape[2]
    if n_spin not in (1, ISPIN2_BLOCKS):
        message: str = "transition channels must contain one or two spins"
        raise ValueError(message)
    if (
        eigenvectors.ndim != CARTESIAN_COMPONENTS
        or eigenvectors.shape[0] != n_kpoints
        or eigenvectors.shape[2] != n_spin * n_spatial
    ):
        message = "eigenvector axes must match transition channels"
        raise ValueError(message)
    eigenvector_blocks: Complex128[
        Array, "n_k n_bands n_spin n_orb_per_spin"
    ] = eigenvectors.reshape(
        eigenvectors.shape[0],
        eigenvectors.shape[1],
        n_spin,
        n_spatial,
    )
    band_channels: Complex128[Array, "n_k n_bands n_spin 3"] = jnp.einsum(
        "ksad,kbsa->kbsd",
        transition_channels,
        eigenvector_blocks,
    )
    return band_channels


@jaxtyped(typechecker=beartype)
def matrix_element_intensity(
    spin_amplitude: Complex128[Array, "... n_spin"],
) -> Float64[Array, " ..."]:
    """Sum outgoing-spin modulus squares exactly once.

    Preserve interference within spins and exclude interference across spins.

    :see: :class:`~.test_transition.TestMatrixElementIntensity`

    Parameters
    ----------
    spin_amplitude : Complex128[Array, "... n_spin"]
        Polarization-contracted amplitude for each outgoing spin.

    Returns
    -------
    intensity : Float64[Array, "..."]
        Incoherent unresolved-spin intensity.

    Notes
    -----
    Form one modulus square and reduce only the final outgoing-spin axis.
    """
    intensity: Float64[Array, " ..."] = jnp.sum(
        jnp.real(jnp.conj(spin_amplitude) * spin_amplitude),
        axis=-1,
    )
    return intensity


@jaxtyped(typechecker=beartype)
def assemble_orbital_transition_channels(  # noqa: DOC503
    bands: DiagonalizedBands,
    radial: RadialSpec,
    me_params: MatrixElementParams,
    quadrature: RadialQuadratureSpec,
    final_state: FinalStateSpec,
    experiment: ExperimentGeometry,
    k_f_cart: Float64[Array, "n_k 3"],
    emission_valid: Bool[Array, " n_k"],
) -> Complex128[Array, "n_k n_spin n_orb_per_spin 3"]:
    """Assemble the validated orbital transition tensor.

    Bind carrier data to the scalar-energy orbital transition primitive.

    The function consumes explicit valid vacuum momentum in the sample frame.
    It accepts only the registered zero in-plane reciprocal shift and never
    reads ``experiment.inner_potential_ev``.

    :see: :class:`~.test_transition.TestAssembleOrbitalTransitionChannels`

    Parameters
    ----------
    bands : DiagonalizedBands
        Eigensystem, geometry, basis, orbital centres, and optional depths.
    radial : RadialSpec
        Shell-shared radial wavefunctions.
    me_params : MatrixElementParams
        Shell scales and channel phase angles.
    quadrature : RadialQuadratureSpec
        Certified fixed radial quadrature.
    final_state : FinalStateSpec
        Explicit radial final-state model.
    experiment : ExperimentGeometry
        Supplies the traced intensity mean free path.
    k_f_cart : Float64[Array, "n_k 3"]
        Explicit vacuum final momentum in sample inverse Angstrom.
    emission_valid : Bool[Array, "n_k"]
        Explicit emission validity mask from vacuum kinematics.

    Returns
    -------
    transition_channels : Complex128[Array, "n_k n_spin n_orb_per_spin 3"]
        Coherent orbital transition channels in real order ``(y,z,x)``.

    Raises
    ------
    ValueError
        If static carriers or array axes disagree.
    EquinoxRuntimeError
        If momentum is invalid, zero, or outside the zero-umklapp seam.

    Notes
    -----
    Validate the vacuum seam before evaluating radial and angular channels.
    """
    if _basis_key(radial.basis) != _basis_key(bands.basis) or _basis_key(
        me_params.basis
    ) != _basis_key(bands.basis):
        message: str = "bands, radial, and matrix-element bases must agree"
        raise ValueError(message)
    if radial.radial_shell_index != me_params.radial_shell_index:
        message = "radial and matrix-element shell maps must agree"
        raise ValueError(message)
    if (
        k_f_cart.ndim != MATRIX_NDIM
        or k_f_cart.shape != bands.kpoints.shape
        or emission_valid.shape != (bands.kpoints.shape[0],)
    ):
        message = "vacuum momentum and validity axes must match band kpoints"
        raise ValueError(message)
    final_momentum: Float64[Array, "n_k 3"] = jnp.asarray(
        k_f_cart,
        dtype=jnp.float64,
    )
    validity: Bool[Array, " n_k"] = jnp.asarray(
        emission_valid,
        dtype=jnp.bool_,
    )
    initial_momentum: Float64[Array, "n_k 3"] = (
        bands.kpoints @ bands.geometry.reciprocal
    )
    final_norm: Float64[Array, " n_k"] = jnp.linalg.norm(
        final_momentum,
        axis=-1,
    )
    final_momentum = eqx.error_if(
        final_momentum,
        ~jnp.all(jnp.isfinite(final_momentum))
        | ~jnp.all(validity)
        | jnp.any(final_norm <= 0.0)
        | jnp.any(
            jnp.abs(final_momentum[:, :2] - initial_momentum[:, :2])
            > G_PARALLEL_ATOL_INV_ANG
        ),
        (
            "matrix-element assembly requires valid nonzero vacuum momentum "
            "in the registered G_parallel=0 channel"
        ),
    )
    positions_cart: Float64[Array, "n_orb 3"] = resolve_orbital_positions_cart(
        bands
    )
    n_orbitals: int = len(bands.basis.n)
    depths: Float64[Array, " n_orb"] = (
        jnp.zeros((n_orbitals,), dtype=jnp.float64)
        if bands.depths is None
        else bands.depths
    )
    momentum_bohr_inv: Float64[Array, " n_k"] = momentum_inv_ang_to_bohr_inv(
        final_norm
    )
    bvals: Complex128[Array, "n_k n_orb 2"] = radial_bvals(
        radial,
        momentum_bohr_inv,
        quadrature,
        final_state,
    )
    transition_channels: Complex128[Array, "n_k n_spin n_orb_per_spin 3"] = (
        orbital_transition_channels(
            initial_momentum,
            final_momentum,
            positions_cart,
            depths,
            bvals,
            me_params,
            experiment.mean_free_path_ang,
            bands.basis,
        )
    )
    return transition_channels


__all__: list[str] = [
    "assemble_orbital_transition_channels",
    "contract_polarization",
    "matrix_element_intensity",
    "orbital_transition_channels",
    "project_band_channels",
    "real_spherical_harmonics_cartesian_all",
    "resolve_orbital_positions_cart",
    "transition_source",
]
