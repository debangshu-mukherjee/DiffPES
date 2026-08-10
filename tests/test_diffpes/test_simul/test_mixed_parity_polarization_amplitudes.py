"""Certify mixed-parity and polarization-basis amplitudes.

Extended Summary
----------------
The independent oracle evaluates the complex-spherical formula with SciPy
harmonics and product angular quadrature. It never reads the production Gaunt
table or real-harmonic channel table. The tests cover every supported initial
real harmonic, mixed-parity plane-wave phases, all three polarization bases,
and the complete-shell single-pz projection.
"""

import math

import chex
import jax.numpy as jnp
import numpy as np
from beartype.typing import Tuple
from jaxtyping import Array, Complex128, Float64
from numpy.typing import NDArray
from scipy.special import sph_harm_y

from diffpes.maths import (
    polarization_cart_to_complex,
    polarization_cart_to_real,
    polarization_real_to_cart,
)
from diffpes.simul import (
    contract_polarization,
    orbital_transition_channels,
    project_band_channels,
)
from diffpes.types import (
    L_MAX,
    MatrixElementParams,
    OrbitalBasis,
    make_matrix_element_params,
    make_orbital_basis,
)

_COSINE_NODES, _COSINE_WEIGHTS = np.polynomial.legendre.leggauss(32)
_PHI_VALUES = np.arange(64, dtype=np.float64) * (2.0 * np.pi / 64.0)
_THETA_GRID = np.arccos(_COSINE_NODES)[:, None]
_PHI_GRID = _PHI_VALUES[None, :]
_PHI_WEIGHT = 2.0 * np.pi / 64.0


def _complex_harmonic(
    degree: int,
    order: int,
    theta: Float64[NDArray, "..."] | float,
    phi: Float64[NDArray, "..."] | float,
) -> Complex128[NDArray, "..."]:
    """PRIVATE: Evaluate one Condon--Shortley complex spherical harmonic.

    Parameters
    ----------
    degree : int
        Spherical-harmonic degree l.
    order : int
        Signed spherical-harmonic order m.
    theta : Float64[NDArray, "..."] | float
        Polar angles in radians.
    phi : Float64[NDArray, "..."] | float
        Azimuthal angles in radians.

    Returns
    -------
    value : Complex128[NDArray, "..."]
        Complex harmonic samples from SciPy.

    Notes
    -----
    Wraps scipy.special.sph_harm_y and converts the result to a
    complex128 array.
    """
    value: Complex128[NDArray, "..."] = np.asarray(
        sph_harm_y(degree, order, theta, phi),
        dtype=np.complex128,
    )
    return value


def _real_harmonic(
    degree: int,
    order: int,
    theta: Float64[NDArray, "..."] | float,
    phi: Float64[NDArray, "..."] | float,
) -> Float64[NDArray, "..."]:
    """PRIVATE: Evaluate one real harmonic from independent SciPy values.

    Parameters
    ----------
    degree : int
        Spherical-harmonic degree l.
    order : int
        Signed real-harmonic order m.
    theta : Float64[NDArray, "..."] | float
        Polar angles in radians.
    phi : Float64[NDArray, "..."] | float
        Azimuthal angles in radians.

    Returns
    -------
    value : Float64[NDArray, "..."]
        Real harmonic samples.

    Notes
    -----
    Evaluates the complex harmonic at order abs(m). For positive m,
    scales its real part. For negative m, scales its imaginary part.
    For m = 0, returns the real part.
    """
    complex_value: Complex128[NDArray, "..."] = _complex_harmonic(
        degree,
        abs(order),
        theta,
        phi,
    )
    if order > 0:
        value: Float64[NDArray, "..."] = (
            np.sqrt(2.0) * (-1) ** order * complex_value.real
        )
    elif order < 0:
        value = np.sqrt(2.0) * (-1) ** abs(order) * complex_value.imag
    else:
        value = complex_value.real
    return value


def _cart_to_complex_independent(
    polarization_cart: Complex128[NDArray, " 3"],
) -> Complex128[NDArray, " 3"]:
    """PRIVATE: Apply the Cartesian-to-spherical map without package code.

    Parameters
    ----------
    polarization_cart : Complex128[NDArray, " 3"]
        Cartesian complex polarization.

    Returns
    -------
    result : Complex128[NDArray, " 3"]
        Spherical components in ascending order q = -1, 0, +1.

    Notes
    -----
    Writes the Condon--Shortley components explicitly:
    (ex - i ey) / sqrt(2), ez, and -(ex + i ey) / sqrt(2).
    """
    inverse_sqrt_two: float = 1.0 / math.sqrt(2.0)
    ex: complex = complex(polarization_cart[0])
    ey: complex = complex(polarization_cart[1])
    ez: complex = complex(polarization_cart[2])
    result: Complex128[NDArray, " 3"] = np.asarray(
        (
            inverse_sqrt_two * (ex - 1j * ey),
            ez,
            -inverse_sqrt_two * (ex + 1j * ey),
        ),
        dtype=np.complex128,
    )
    return result


def _complex_formula_amplitude(
    degree: int,
    order: int,
    direction_cart: Float64[NDArray, " 3"],
    radial_channels: Complex128[NDArray, " 2"],
    polarization_cart: Complex128[NDArray, " 3"],
) -> complex:
    r"""PRIVATE: Evaluate the independent complex-Ylm amplitude.

    The oracle contracts
    ``sum_q (-1)^q epsilon_q Y_1^{-q}`` and
    ``sum_m' Y_l'^m'*(khat) Y_l'^m'(rhat)`` by angular quadrature.

    Parameters
    ----------
    degree : int
        Initial orbital degree l.
    order : int
        Initial real-harmonic order m.
    direction_cart : Float64[NDArray, " 3"]
        Detector direction; normalized inside.
    radial_channels : Complex128[NDArray, " 2"]
        Complex radial values for the l-1 and l+1 branches.
    polarization_cart : Complex128[NDArray, " 3"]
        Cartesian complex polarization.

    Returns
    -------
    amplitude : complex
        Oracle transition amplitude for the single orbital.

    Implementation Logic
    --------------------
    Builds the dipole operator on a 32-by-64 product grid. Integrates the
    initial harmonic, dipole, and final harmonic with product weights.
    Multiplies each final order by the conjugated detector-direction
    harmonic. Sums both allowed branches with their radial channels.
    """
    direction: Float64[NDArray, " 3"] = np.asarray(
        direction_cart, dtype=np.float64
    )
    direction = direction / np.linalg.norm(direction)
    theta_direction: float = math.acos(float(direction[2]))
    phi_direction: float = math.atan2(
        float(direction[1]),
        float(direction[0]),
    )
    initial: Float64[NDArray, "n_theta n_phi"] = _real_harmonic(
        degree,
        order,
        _THETA_GRID,
        _PHI_GRID,
    )
    polarization_complex: Complex128[NDArray, " 3"] = (
        _cart_to_complex_independent(polarization_cart)
    )
    dipole: Complex128[NDArray, "n_theta n_phi"] = np.zeros_like(
        _THETA_GRID + _PHI_GRID,
        dtype=np.complex128,
    )
    photon_order: int
    for photon_order in (-1, 0, 1):
        dipole += (
            (-1) ** photon_order
            * polarization_complex[photon_order + 1]
            * _complex_harmonic(
                1,
                -photon_order,
                _THETA_GRID,
                _PHI_GRID,
            )
        )

    amplitude: complex = 0.0j
    branch: int
    final_degree: int
    for branch, final_degree in enumerate((degree - 1, degree + 1)):
        if final_degree < 0:
            continue
        branch_amplitude: complex = 0.0j
        final_order: int
        for final_order in range(-final_degree, final_degree + 1):
            final_grid: Complex128[NDArray, "n_theta n_phi"] = (
                _complex_harmonic(
                    final_degree,
                    final_order,
                    _THETA_GRID,
                    _PHI_GRID,
                )
            )
            angular_integral: complex = complex(
                np.sum(
                    _COSINE_WEIGHTS[:, None] * final_grid * dipole * initial
                )
                * _PHI_WEIGHT
            )
            final_direction: complex = complex(
                _complex_harmonic(
                    final_degree,
                    final_order,
                    theta_direction,
                    phi_direction,
                )
            )
            branch_amplitude += np.conj(final_direction) * angular_integral
        amplitude += complex(radial_channels[branch]) * branch_amplitude
    return amplitude


def _single_orbital_channels(
    degree: int,
    order: int,
    direction_cart: Float64[NDArray, " 3"],
    radial_channels: Complex128[NDArray, " 2"],
) -> Tuple[Complex128[Array, " 3"], OrbitalBasis]:
    """PRIVATE: Evaluate one production real-orbital transition row.

    Parameters
    ----------
    degree : int
        Orbital degree l; the principal number is l + 1.
    order : int
        Real-harmonic order m.
    direction_cart : Float64[NDArray, " 3"]
        Final momentum direction for the single detector point.
    radial_channels : Complex128[NDArray, " 2"]
        Radial branch values of the single orbital.

    Returns
    -------
    row : Complex128[Array, " 3"]
        Cartesian transition row of the single orbital.
    basis : OrbitalBasis
        One-orbital basis metadata used for the row.

    Notes
    -----
    Places the orbital at the origin with zero depth and initial
    momentum. Uses an 8 Angstrom mean free path. Slices the single
    k-point, spin, and orbital entry.
    """
    basis: OrbitalBasis = make_orbital_basis(
        atom_indices=(0,),
        n=(degree + 1,),
        l=(degree,),
        m=(order,),
    )
    params: MatrixElementParams = make_matrix_element_params(basis, (0,))
    direction: Float64[Array, "1 3"] = jnp.asarray(
        direction_cart[None, :],
        dtype=jnp.float64,
    )
    channels: Complex128[Array, "1 1 1 3"] = orbital_transition_channels(
        jnp.zeros((1, 3), dtype=jnp.float64),
        direction,
        jnp.zeros((1, 3), dtype=jnp.float64),
        jnp.zeros((1,), dtype=jnp.float64),
        jnp.asarray(radial_channels[None, None, :]),
        params,
        jnp.asarray(8.0, dtype=jnp.float64),
        basis,
    )
    return channels[0, 0, 0], basis


def _complex_metric(
    first: Complex128[Array, " 3"],
    second: Complex128[Array, " 3"],
) -> Complex128[Array, ""]:
    """PRIVATE: Return the metric contraction of two spherical vectors.

    Parameters
    ----------
    first : Complex128[Array, " 3"]
        Spherical vector in ascending order q = -1, 0, +1.
    second : Complex128[Array, " 3"]
        Spherical vector in the same component order.

    Returns
    -------
    result : Complex128[Array, ""]
        Rank-one scalar contraction of the two vectors.

    Notes
    -----
    Expands sum_q (-1)**q a_q b_(-q) as -a[0] b[2] + a[1] b[1] -
    a[2] b[0] in the ascending component order.
    """
    result: Complex128[Array, ""] = (
        -first[0] * second[2] + first[1] * second[1] - first[2] * second[0]
    )
    return result


def test_g8_all_real_orbitals_match_independent_complex_formula() -> None:
    """Match every supported ``(l,m)`` with the complex-Ylm oracle.

    The comparison covers every production real harmonic through the
    independent complex-spherical quadrature path.

    Notes
    -----
    Generic complex polarization and unrelated complex radial branches expose
    conjugation, spherical-metric, real-basis, and magnetic-index errors.
    """
    direction: Float64[NDArray, " 3"] = np.asarray((0.37, -0.51, 0.78))
    polarization: Complex128[NDArray, " 3"] = np.asarray(
        (0.31 + 0.27j, -0.42 + 0.19j, 0.53 - 0.11j),
        dtype=np.complex128,
    )
    radial_channels: Complex128[NDArray, " 2"] = np.asarray(
        (0.29 - 0.33j, -0.47 + 0.21j),
        dtype=np.complex128,
    )
    degree: int
    order: int
    for degree in range(L_MAX + 1):
        for order in range(-degree, degree + 1):
            transition: Complex128[Array, " 3"]
            transition, _ = _single_orbital_channels(
                degree,
                order,
                direction,
                radial_channels,
            )
            actual: complex = complex(
                contract_polarization(
                    transition,
                    jnp.asarray(polarization),
                )
            )
            expected: complex = _complex_formula_amplitude(
                degree,
                order,
                direction,
                radial_channels,
                polarization,
            )
            np.testing.assert_allclose(
                actual,
                expected,
                rtol=1.0e-12,
                atol=1.0e-13,
                err_msg=f"complex amplitude mismatch for l={degree}, m={order}",
            )


def test_mixed_parity_pins_plane_wave_phase_and_helicity() -> None:
    """Pin the complete s+p amplitude and reject three phase false controls.

    Generic and helicity polarizations expose the relative partial-wave phase
    between the even and odd initial orbitals.

    Notes
    -----
    Correct, omitted, conjugated, and double-applied ``i**l_prime`` factors
    differ only in the radial inputs supplied to the same production seam.
    """
    basis: OrbitalBasis = make_orbital_basis(
        atom_indices=(0, 0),
        n=(1, 2),
        l=(0, 1),
        m=(0, 0),
    )
    params: MatrixElementParams = make_matrix_element_params(basis, (0, 1))
    direction: Float64[NDArray, " 3"] = np.asarray((0.41, 0.36, 0.84))
    phase_free: Complex128[NDArray, "2 2"] = np.asarray(
        ((0.0, 0.73), (0.41, -0.52)),
        dtype=np.complex128,
    )
    coefficients: Complex128[NDArray, " 2"] = np.asarray(
        (0.8 + 0.2j, -0.35 + 0.6j),
        dtype=np.complex128,
    )

    def phased_radial(mode: str) -> Complex128[NDArray, "2 2"]:
        """Return one planted partial-wave phase convention."""
        values: Complex128[NDArray, "2 2"] = np.zeros_like(phase_free)
        orbital: int
        degree: int
        branch: int
        final_degree: int
        for orbital, degree in enumerate((0, 1)):
            for branch, final_degree in enumerate((degree - 1, degree + 1)):
                if final_degree < 0:
                    continue
                if mode == "correct":
                    factor: complex = 1j**final_degree
                elif mode == "omitted":
                    factor = 1.0
                elif mode == "flipped":
                    factor = (-1j) ** final_degree
                else:
                    factor = (1j**final_degree) ** 2
                values[orbital, branch] = factor * phase_free[orbital, branch]
        return values

    def production_amplitude(
        radial_values: Complex128[NDArray, "2 2"],
        polarization: Complex128[NDArray, " 3"],
    ) -> complex:
        """Return the coherent two-orbital production amplitude."""
        transition: Complex128[Array, "1 1 2 3"] = orbital_transition_channels(
            jnp.zeros((1, 3)),
            jnp.asarray(direction[None, :]),
            jnp.zeros((2, 3)),
            jnp.zeros((2,)),
            jnp.asarray(radial_values[None, :, :]),
            params,
            jnp.asarray(9.0),
            basis,
        )
        polarized: Complex128[Array, " 2"] = contract_polarization(
            transition[0, 0],
            jnp.asarray(polarization),
        )
        return complex(jnp.sum(polarized * jnp.asarray(coefficients)))

    inverse_sqrt_two: float = 1.0 / math.sqrt(2.0)
    polarizations: Tuple[Complex128[NDArray, " 3"], ...] = (
        np.asarray((0.23 + 0.17j, -0.49 + 0.31j, 0.61 - 0.09j)),
        inverse_sqrt_two * np.asarray((1.0, 1j, 0.0)),
        inverse_sqrt_two * np.asarray((1.0, -1j, 0.0)),
    )
    correct_radial: Complex128[NDArray, "2 2"] = phased_radial("correct")
    polarization: Complex128[NDArray, " 3"]
    for polarization in polarizations:
        expected: complex = sum(
            coefficients[orbital]
            * _complex_formula_amplitude(
                degree,
                0,
                direction,
                correct_radial[orbital],
                polarization,
            )
            for orbital, degree in enumerate((0, 1))
        )
        actual: complex = production_amplitude(
            correct_radial,
            polarization,
        )
        np.testing.assert_allclose(
            actual,
            expected,
            rtol=1.0e-12,
            atol=1.0e-13,
        )

    generic: Complex128[NDArray, " 3"] = polarizations[0]
    correct: complex = production_amplitude(correct_radial, generic)
    wrong_mode: str
    for wrong_mode in ("omitted", "flipped", "doubled"):
        wrong: complex = production_amplitude(
            phased_radial(wrong_mode),
            generic,
        )
        assert abs(wrong - correct) > 1.0e-3


def test_g14_actual_amplitude_agrees_in_all_polarization_bases() -> None:
    """Match a production transition amplitude in Cartesian, real, and complex bases.

    Cartesian basis vectors, generic elliptic polarization, and both
    helicities cover the three equivalent contraction routes.

    Notes
    -----
    Basis vectors, a generic elliptic vector, and both helicities exercise the
    two unitary transforms; a real-order-as-complex control must disagree.
    """
    direction: Float64[NDArray, " 3"] = np.asarray((-0.32, 0.58, 0.75))
    radial: Complex128[NDArray, " 2"] = np.asarray(
        (0.38 + 0.22j, -0.29 + 0.47j)
    )
    transition: Complex128[Array, " 3"]
    transition, _ = _single_orbital_channels(2, -1, direction, radial)
    dipole_cart: Complex128[Array, " 3"] = polarization_real_to_cart(
        transition
    )
    dipole_complex: Complex128[Array, " 3"] = polarization_cart_to_complex(
        dipole_cart
    )
    inverse_sqrt_two: float = 1.0 / math.sqrt(2.0)
    polarizations: Tuple[Complex128[Array, " 3"], ...] = (
        jnp.asarray((1.0, 0.0, 0.0), dtype=jnp.complex128),
        jnp.asarray((0.0, 1.0, 0.0), dtype=jnp.complex128),
        jnp.asarray((0.0, 0.0, 1.0), dtype=jnp.complex128),
        jnp.asarray(
            (0.27 + 0.19j, -0.43 + 0.31j, 0.52 - 0.17j),
            dtype=jnp.complex128,
        ),
        inverse_sqrt_two * jnp.asarray((1.0, 1j, 0.0), dtype=jnp.complex128),
        inverse_sqrt_two * jnp.asarray((1.0, -1j, 0.0), dtype=jnp.complex128),
    )
    polarization: Complex128[Array, " 3"]
    for polarization in polarizations:
        cartesian: Complex128[Array, ""] = jnp.dot(
            polarization,
            dipole_cart,
        )
        real: Complex128[Array, ""] = jnp.dot(
            polarization_cart_to_real(polarization),
            transition,
        )
        complex_value: Complex128[Array, ""] = _complex_metric(
            polarization_cart_to_complex(polarization),
            dipole_complex,
        )
        chex.assert_trees_all_close(
            cartesian,
            real,
            rtol=1.0e-14,
            atol=1.0e-14,
        )
        chex.assert_trees_all_close(
            cartesian,
            complex_value,
            rtol=1.0e-14,
            atol=1.0e-14,
        )

    generic: Complex128[Array, " 3"] = polarizations[3]
    correct: Complex128[Array, ""] = _complex_metric(
        polarization_cart_to_complex(generic),
        dipole_complex,
    )
    planted_wrong: Complex128[Array, ""] = _complex_metric(
        polarization_cart_to_real(generic),
        dipole_complex,
    )
    assert not bool(
        jnp.allclose(correct, planted_wrong, rtol=1e-12, atol=1e-12)
    )


def test_g8_g14_complete_p_shell_single_pz_projection() -> None:
    """Keep only the pz coefficient and reject an unweighted p-shell sum.

    A complete p shell supplies the projection source while one nonzero
    eigenvector coefficient isolates its middle pz orbital.

    Notes
    -----
    Use the complete real p-shell order ``(p_y,p_z,p_x)``. Select its middle
    row before the same all-basis amplitude comparison.
    """
    basis: OrbitalBasis = make_orbital_basis(
        atom_indices=(0, 0, 0),
        n=(2, 2, 2),
        l=(1, 1, 1),
        m=(-1, 0, 1),
    )
    params: MatrixElementParams = make_matrix_element_params(
        basis,
        (0, 0, 0),
    )
    direction: Float64[Array, "1 3"] = jnp.asarray(((0.35, -0.44, 0.83),))
    radial: Complex128[Array, "1 3 2"] = jnp.asarray(
        [[[0.37 + 0.12j, -0.28 + 0.41j]] * 3]
    )
    transition: Complex128[Array, "1 1 3 3"] = orbital_transition_channels(
        jnp.zeros((1, 3)),
        direction,
        jnp.zeros((3, 3)),
        jnp.zeros((3,)),
        radial,
        params,
        jnp.asarray(7.5),
        basis,
    )
    pz_coefficient: complex = 0.6 + 0.8j
    eigenvectors: Complex128[Array, "1 1 3"] = jnp.asarray(
        (((0.0, pz_coefficient, 0.0),),),
        dtype=jnp.complex128,
    )
    projected: Complex128[Array, " 3"] = project_band_channels(
        transition,
        eigenvectors,
    )[0, 0, 0]
    expected: Complex128[Array, " 3"] = pz_coefficient * transition[0, 0, 1]
    chex.assert_trees_all_close(
        projected,
        expected,
        rtol=0.0,
        atol=0.0,
    )
    planted_unweighted: Complex128[Array, " 3"] = pz_coefficient * jnp.sum(
        transition[0, 0], axis=0
    )
    assert not bool(
        jnp.allclose(
            projected,
            planted_unweighted,
            rtol=1.0e-12,
            atol=1.0e-12,
        )
    )

    polarization: Complex128[Array, " 3"] = jnp.asarray(
        (0.31 + 0.2j, -0.17 + 0.43j, 0.56 - 0.09j),
        dtype=jnp.complex128,
    )
    dipole_cart: Complex128[Array, " 3"] = polarization_real_to_cart(projected)
    cartesian: Complex128[Array, ""] = jnp.dot(
        polarization,
        dipole_cart,
    )
    real: Complex128[Array, ""] = jnp.dot(
        polarization_cart_to_real(polarization),
        projected,
    )
    complex_value: Complex128[Array, ""] = _complex_metric(
        polarization_cart_to_complex(polarization),
        polarization_cart_to_complex(dipole_cart),
    )
    chex.assert_trees_all_close(
        cartesian,
        real,
        rtol=1.0e-14,
        atol=1.0e-14,
    )
    chex.assert_trees_all_close(
        cartesian,
        complex_value,
        rtol=1.0e-14,
        atol=1.0e-14,
    )
