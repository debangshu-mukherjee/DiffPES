"""Certify Plan 06 attenuation and displaced-centre interference gates.

Extended Summary
----------------
The tests distinguish amplitude and intensity attenuation exponents. They
also verify coherent depth and displaced-centre phase interference.
"""

import chex
import equinox as eqx
import jax
import jax.numpy as jnp
import pytest
from jaxtyping import Array, Complex, Float

from diffpes.simul import (
    contract_polarization,
    orbital_transition_channels,
    resolve_orbital_positions_cart,
)
from diffpes.types import (
    CrystalGeometry,
    DiagonalizedBands,
    MatrixElementParams,
    OrbitalBasis,
    make_crystal_geometry,
    make_diagonalized_bands,
    make_matrix_element_params,
    make_orbital_basis,
)


def _two_s_basis() -> OrbitalBasis:
    """Return two independent real s orbitals."""
    basis: OrbitalBasis = make_orbital_basis(
        atom_indices=(0, 1),
        n=(1, 1),
        l=(0, 0),
        m=(0, 0),
    )
    return basis


def _two_s_channels(
    positions_cart: Float[Array, "2 3"],
    depths: Float[Array, " 2"],
    mean_free_path: Float[Array, ""],
    initial_momentum: Float[Array, "1 3"] | None = None,
    final_momentum: Float[Array, "1 3"] | None = None,
) -> Complex[Array, "1 1 2 3"]:
    """Evaluate identical s-orbital transition rows."""
    basis: OrbitalBasis = _two_s_basis()
    params: MatrixElementParams = make_matrix_element_params(basis, (0, 1))
    initial: Float[Array, "1 3"] = (
        jnp.zeros((1, 3), dtype=jnp.float64)
        if initial_momentum is None
        else initial_momentum
    )
    final: Float[Array, "1 3"] = (
        jnp.asarray(((0.0, 0.0, 1.1),), dtype=jnp.float64)
        if final_momentum is None
        else final_momentum
    )
    radial: Complex[Array, "1 2 2"] = jnp.asarray(
        (((0.0, 0.37 + 0.91j), (0.0, 0.37 + 0.91j)),),
        dtype=jnp.complex128,
    )
    channels: Complex[Array, "1 1 2 3"] = orbital_transition_channels(
        initial,
        final,
        positions_cart,
        depths,
        radial,
        params,
        mean_free_path,
        basis,
    )
    return channels


def test_g9_isolated_amplitude_and_intensity_ratios() -> None:
    """Match both attenuation exponents on isolated depth contributions.

    Two otherwise identical s orbitals isolate the relative depth factor in
    both the amplitude norm and its squared intensity.

    Notes
    -----
    Compare the measured ratios with the two analytic attenuation exponents.
    """
    depth_difference: float = 4.7
    mean_free_path: Float[Array, ""] = jnp.asarray(8.3)
    channels: Complex[Array, "1 1 2 3"] = _two_s_channels(
        jnp.zeros((2, 3)),
        jnp.asarray((0.0, depth_difference)),
        mean_free_path,
    )
    shallow_norm: Float[Array, ""] = jnp.linalg.norm(channels[0, 0, 0])
    deep_norm: Float[Array, ""] = jnp.linalg.norm(channels[0, 0, 1])
    amplitude_ratio: Float[Array, ""] = deep_norm / shallow_norm
    intensity_ratio: Float[Array, ""] = amplitude_ratio**2
    chex.assert_trees_all_close(
        amplitude_ratio,
        jnp.exp(-depth_difference / (2.0 * mean_free_path)),
        rtol=1.0e-14,
        atol=1.0e-14,
    )
    chex.assert_trees_all_close(
        intensity_ratio,
        jnp.exp(-depth_difference / mean_free_path),
        rtol=1.0e-14,
        atol=1.0e-14,
    )


def test_g9_coherent_depth_interference_is_not_an_incoherent_sum() -> None:
    """Match the two-depth coherent amplitude and reject early squaring.

    Identical orbital amplitudes at different depths provide a direct
    interference term after late coherent summation.

    Notes
    -----
    Compare coherent summation with its analytic result and an incoherent false control.
    """
    depth: float = 5.1
    mean_free_path: Float[Array, ""] = jnp.asarray(7.4)
    channels: Complex[Array, "1 1 2 3"] = _two_s_channels(
        jnp.zeros((2, 3)),
        jnp.asarray((0.0, depth)),
        mean_free_path,
    )
    polarization: Complex[Array, " 3"] = jnp.asarray(
        (0.0, 0.0, 1.0),
        dtype=jnp.complex128,
    )
    orbital_amplitudes: Complex[Array, " 2"] = contract_polarization(
        channels[0, 0],
        polarization,
    )
    attenuation: Float[Array, ""] = jnp.exp(-depth / (2.0 * mean_free_path))
    atomic_intensity: Float[Array, ""] = jnp.abs(orbital_amplitudes[0]) ** 2
    coherent: Float[Array, ""] = jnp.abs(jnp.sum(orbital_amplitudes)) ** 2
    expected: Float[Array, ""] = atomic_intensity * (1.0 + attenuation) ** 2
    planted_incoherent: Float[Array, ""] = atomic_intensity * (
        1.0 + attenuation**2
    )
    chex.assert_trees_all_close(
        coherent,
        expected,
        rtol=1.0e-14,
        atol=1.0e-14,
    )
    assert not bool(
        jnp.isclose(
            coherent,
            planted_incoherent,
            rtol=1.0e-12,
            atol=1.0e-12,
        )
    )


def test_g9_negative_depth_clamp_rejection_and_abs_false_control() -> None:
    """Verify noise clamping, negative-depth rejection, and the ``abs`` control.

    Sub-tolerance negative noise maps to the surface while a material negative
    depth triggers the physical-domain guard.

    Notes
    -----
    Contrast both production outcomes with planted absolute-value attenuation.
    """
    mean_free_path: Float[Array, ""] = jnp.asarray(8.0)
    tolerance_noise: float = -0.5e-12
    clamped: Complex[Array, "1 1 2 3"] = _two_s_channels(
        jnp.zeros((2, 3)),
        jnp.asarray((tolerance_noise, 0.0)),
        mean_free_path,
    )
    zero: Complex[Array, "1 1 2 3"] = _two_s_channels(
        jnp.zeros((2, 3)),
        jnp.zeros((2,)),
        mean_free_path,
    )
    chex.assert_trees_all_equal(clamped, zero)
    planted_abs_factor: Float[Array, ""] = jnp.exp(
        -abs(tolerance_noise) / (2.0 * mean_free_path)
    )
    assert float(planted_abs_factor) != 1.0

    material_negative: float = -0.02
    with pytest.raises(
        eqx.EquinoxRuntimeError,
        match="physical depth",
    ):
        _two_s_channels(
            jnp.zeros((2, 3)),
            jnp.asarray((material_negative, 0.0)),
            mean_free_path,
        )
    material_abs_factor: Float[Array, ""] = jnp.exp(
        -abs(material_negative) / (2.0 * mean_free_path)
    )
    assert bool(jnp.isfinite(material_abs_factor))
    assert float(material_abs_factor) < 1.0


def _displaced_bands() -> tuple[
    DiagonalizedBands,
    Float[Array, "2 3"],
]:
    """Return two atoms with one explicitly displaced Wannier centre."""
    basis: OrbitalBasis = _two_s_basis()
    geometry: CrystalGeometry = make_crystal_geometry(
        jnp.asarray(
            (
                (2.3, 0.2, 0.0),
                (0.1, 2.8, 0.3),
                (0.0, 0.2, 3.7),
            )
        ),
        jnp.asarray(((0.08, 0.16, 0.12), (0.57, 0.38, 0.24))),
        ("A", "B"),
    )
    displacement_fractional: Float[Array, " 3"] = jnp.asarray(
        (0.17, 0.11, 0.07)
    )
    explicit_fractional: Float[Array, "2 3"] = geometry.positions.at[1].add(
        displacement_fractional
    )
    bands: DiagonalizedBands = make_diagonalized_bands(
        jnp.zeros((1, 2)),
        jnp.eye(2, dtype=jnp.complex128)[None, :, :],
        jnp.zeros((1, 3)),
        geometry,
        basis,
        orbital_positions=explicit_fractional,
    )
    return bands, displacement_fractional


def test_g18_displaced_centre_interference_and_derivative_controls() -> None:
    """Match the explicit-centre phase, interference, and displacement slope.

    One displaced Wannier centre changes the relative plane-wave phase and the
    coherent two-orbital intensity.

    Notes
    -----
    Compare the phase, intensity, and autodiff slope with their analytic forms.
    """
    bands: DiagonalizedBands
    displacement_fractional: Float[Array, " 3"]
    bands, displacement_fractional = _displaced_bands()
    explicit_cart: Float[Array, "2 3"] = resolve_orbital_positions_cart(bands)
    host_cart: Float[Array, "2 3"] = (
        bands.geometry.positions[jnp.asarray(bands.basis.atom_indices)]
        @ bands.geometry.lattice
    )
    chex.assert_trees_all_close(
        explicit_cart[1] - host_cart[1],
        displacement_fractional @ bands.geometry.lattice,
        rtol=0.0,
        atol=1.0e-15,
    )

    initial: Float[Array, "1 3"] = jnp.asarray(((0.52, -0.21, 0.16),))
    final: Float[Array, "1 3"] = jnp.asarray(((-0.14, 0.31, 1.27),))
    momentum_difference: Float[Array, " 3"] = initial[0] - final[0]
    polarization: Complex[Array, " 3"] = jnp.asarray(
        (0.23 + 0.17j, -0.39 + 0.11j, 0.58 - 0.09j),
        dtype=jnp.complex128,
    )

    def coherent_intensity(scale: Float[Array, ""]) -> Float[Array, ""]:
        """Return intensity after scaling only the explicit displacement."""
        positions: Float[Array, "2 3"] = host_cart.at[1].add(
            scale * (explicit_cart[1] - host_cart[1])
        )
        channels: Complex[Array, "1 1 2 3"] = _two_s_channels(
            positions,
            jnp.zeros((2,)),
            jnp.asarray(9.0),
            initial,
            final,
        )
        amplitudes: Complex[Array, " 2"] = contract_polarization(
            channels[0, 0],
            polarization,
        )
        return jnp.abs(jnp.sum(amplitudes)) ** 2

    explicit_channels: Complex[Array, "1 1 2 3"] = _two_s_channels(
        explicit_cart,
        jnp.zeros((2,)),
        jnp.asarray(9.0),
        initial,
        final,
    )
    explicit_amplitudes: Complex[Array, " 2"] = contract_polarization(
        explicit_channels[0, 0],
        polarization,
    )
    relative_position: Float[Array, " 3"] = explicit_cart[1] - explicit_cart[0]
    relative_phase: Complex[Array, ""] = jnp.exp(
        1j * jnp.dot(momentum_difference, relative_position)
    )
    chex.assert_trees_all_close(
        explicit_amplitudes[1] / explicit_amplitudes[0],
        relative_phase,
        rtol=1.0e-13,
        atol=1.0e-14,
    )
    expected_intensity: Float[Array, ""] = (
        jnp.abs(explicit_amplitudes[0]) ** 2
        * jnp.abs(1.0 + relative_phase) ** 2
    )
    chex.assert_trees_all_close(
        coherent_intensity(jnp.asarray(1.0)),
        expected_intensity,
        rtol=1.0e-13,
        atol=1.0e-14,
    )

    phase_argument: Float[Array, ""] = jnp.dot(
        momentum_difference,
        relative_position,
    )
    displacement_cart: Float[Array, " 3"] = explicit_cart[1] - host_cart[1]
    phase_slope: Float[Array, ""] = jnp.dot(
        momentum_difference,
        displacement_cart,
    )
    expected_slope: Float[Array, ""] = (
        -2.0
        * jnp.abs(explicit_amplitudes[0]) ** 2
        * jnp.sin(phase_argument)
        * phase_slope
    )
    actual_slope: Float[Array, ""] = jax.grad(coherent_intensity)(
        jnp.asarray(1.0)
    )
    chex.assert_trees_all_close(
        actual_slope,
        expected_slope,
        rtol=1.0e-12,
        atol=1.0e-13,
    )
    assert float(jnp.abs(actual_slope)) > 1.0e-5

    host_channels: Complex[Array, "1 1 2 3"] = _two_s_channels(
        host_cart,
        jnp.zeros((2,)),
        jnp.asarray(9.0),
        initial,
        final,
    )
    host_amplitude: Complex[Array, ""] = jnp.sum(
        contract_polarization(host_channels[0, 0], polarization)
    )
    assert not bool(
        jnp.isclose(
            jnp.abs(host_amplitude) ** 2,
            expected_intensity,
            rtol=1.0e-12,
            atol=1.0e-12,
        )
    )
    planted_fallback_slope: Float[Array, ""] = jax.grad(
        lambda scale: jnp.abs(host_amplitude + 0.0 * scale) ** 2
    )(jnp.asarray(1.0))
    assert planted_fallback_slope == 0.0
