"""Validate dipole basis transforms, channel tables, and gauge contractions.

The tests exercise complex phases, static padding, and independent direct
Cartesian sums.
"""

import math
from functools import lru_cache

import chex
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from beartype.typing import Any, Dict, Tuple
from jaxtyping import Array, Complex128, Float64, Int64
from numpy.typing import NDArray
from scipy.integrate import lebedev_rule
from scipy.sparse import csr_matrix, diags, lil_matrix
from scipy.sparse.linalg import eigsh

from diffpes.maths import (
    GAUNT_TABLE,
    channel_tables,
    dipole_length_cartesian,
    dipole_momentum_cartesian,
    polarization_cart_to_complex,
    polarization_cart_to_real,
    polarization_complex_to_cart,
    polarization_real_to_cart,
)
from diffpes.types import L_MAX, OrbitalBasis, make_orbital_basis


def _generic_polarization() -> Array:
    """PRIVATE: Return a generic complex elliptic polarization.

    Returns
    -------
    polarization : Array
        Complex128 Cartesian polarization with three unequal elliptic
        components.

    Notes
    -----
    Fixes the components so no Cartesian axis or phase is special.
    """
    polarization: Array = jnp.asarray(
        (0.31 + 0.17j, -0.23 + 0.41j, 0.53 - 0.29j),
        dtype=jnp.complex128,
    )
    return polarization


def _boole_weights(
    node_count: int, radius: float
) -> Float64[NDArray, " n_node"]:
    """PRIVATE: Return composite closed-Boole weights on a uniform grid.

    Parameters
    ----------
    node_count : int
        Number of inclusive grid nodes; the interval count must be a
        multiple of four.
    radius : float
        Upper limit of the [0, radius] radial grid in Bohr.

    Returns
    -------
    weights : Float64[NDArray, " n_node"]
        Quadrature weights for the uniform inclusive grid.

    Notes
    -----
    Adds the five-point closed Boole block 2h/45 * (7, 32, 12, 32, 7)
    over each group of four intervals; interior block edges accumulate
    contributions from both neighbor blocks.
    """
    interval_count: int = node_count - 1
    assert interval_count % 4 == 0
    spacing: float = radius / interval_count
    weights: Float64[NDArray, " n_node"] = np.zeros(
        node_count, dtype=np.float64
    )
    block: Float64[NDArray, " 5"] = (2.0 * spacing / 45.0) * np.asarray(
        (7.0, 32.0, 12.0, 32.0, 7.0)
    )
    start: int
    for start in range(0, interval_count, 4):
        weights[start : start + 5] += block
    return weights


def _tensor_product_gauges(
    radial_grid: Float64[NDArray, " n_node"],
    radial_weights: Float64[NDArray, " n_node"],
    radial_initial: Float64[NDArray, " n_node"],
    radial_initial_derivative: Float64[NDArray, " n_node"],
    radial_final: Float64[NDArray, " n_node"],
    lebedev_degree: int,
) -> Tuple[Array, Array]:
    """PRIVATE: Evaluate both public gauges on a radial-Lebedev product grid.

    Parameters
    ----------
    radial_grid : Float64[NDArray, " n_node"]
        Radial nodes in Bohr.
    radial_weights : Float64[NDArray, " n_node"]
        One-dimensional radial quadrature weights in Bohr.
    radial_initial : Float64[NDArray, " n_node"]
        Initial s-state radial function samples.
    radial_initial_derivative : Float64[NDArray, " n_node"]
        Radial derivative samples of the initial radial function.
    radial_final : Float64[NDArray, " n_node"]
        Final p-state radial function samples.
    lebedev_degree : int
        Degree of the Lebedev angular rule.

    Returns
    -------
    length : Array
        Length-gauge amplitude for z polarization.
    momentum : Array
        Momentum-gauge amplitude for z polarization.

    Implementation Logic
    --------------------
    Attaches the s harmonic to the initial state and the p_z harmonic
    to the final state, forms flattened radius-times-direction sample
    points in chunks of 512 radial nodes, weights each sample with the
    radial weight times radius squared times the angular weight, and
    accumulates both public Cartesian contractions over the chunks. The
    initial-state gradient is the radial derivative times the unit
    direction.
    """
    angular_points: Float64[NDArray, "3 n_angular"]
    angular_weights: Float64[NDArray, " n_angular"]
    angular_points, angular_weights = lebedev_rule(lebedev_degree)
    directions: Float64[NDArray, "n_angular 3"] = angular_points.T
    angular_count: int = directions.shape[0]
    polarization: Array = jnp.asarray((0.0, 0.0, 1.0), dtype=jnp.complex128)
    length: Array = jnp.asarray(0.0j)
    momentum: Array = jnp.asarray(0.0j)
    chunk_size: int = 512
    chunk_start: int
    for chunk_start in range(0, radial_grid.size, chunk_size):
        chunk_stop: int = min(chunk_start + chunk_size, radial_grid.size)
        radial_chunk: Float64[NDArray, " n_chunk"] = radial_grid[
            chunk_start:chunk_stop
        ]
        radial_weight_chunk: Float64[NDArray, " n_chunk"] = radial_weights[
            chunk_start:chunk_stop
        ]
        initial_chunk: Float64[NDArray, " n_chunk"] = radial_initial[
            chunk_start:chunk_stop
        ]
        derivative_chunk: Float64[NDArray, " n_chunk"] = (
            radial_initial_derivative[chunk_start:chunk_stop]
        )
        final_chunk: Float64[NDArray, " n_chunk"] = radial_final[
            chunk_start:chunk_stop
        ]
        chunk_count: int = chunk_stop - chunk_start
        radius_flat: Float64[NDArray, " n_flat"] = np.repeat(
            radial_chunk, angular_count
        )
        directions_flat: Float64[NDArray, "n_flat 3"] = np.tile(
            directions, (chunk_count, 1)
        )
        position_flat: Float64[NDArray, "n_flat 3"] = (
            radius_flat[:, None] * directions_flat
        )
        volume_weights: Float64[NDArray, " n_flat"] = np.repeat(
            radial_weight_chunk * radial_chunk**2, angular_count
        ) * np.tile(angular_weights, chunk_count)
        initial_flat: Complex128[NDArray, " n_flat"] = np.repeat(
            initial_chunk / math.sqrt(4.0 * math.pi),
            angular_count,
        ).astype(np.complex128)
        final_flat: Complex128[NDArray, " n_flat"] = (
            np.repeat(
                final_chunk * math.sqrt(3.0 / (4.0 * math.pi)),
                angular_count,
            )
            * directions_flat[:, 2]
        ).astype(np.complex128)
        initial_derivative_flat: Float64[NDArray, " n_flat"] = np.repeat(
            derivative_chunk / math.sqrt(4.0 * math.pi),
            angular_count,
        )
        gradient_flat: Complex128[NDArray, "n_flat 3"] = (
            initial_derivative_flat[:, None] * directions_flat
        ).astype(np.complex128)
        length = length + dipole_length_cartesian(
            jnp.asarray(final_flat),
            jnp.asarray(initial_flat),
            jnp.asarray(position_flat),
            jnp.asarray(volume_weights),
            polarization,
        )
        momentum = momentum + dipole_momentum_cartesian(
            jnp.asarray(final_flat),
            jnp.asarray(gradient_flat),
            jnp.asarray(volume_weights),
            polarization,
        )
    return length, momentum


def _hydrogenic_gauges(
    charge: Array,
    node_count: int,
    lebedev_degree: int,
) -> Tuple[Array, Array]:
    """PRIVATE: Evaluate the normalized hydrogenic 1s-to-2p gauge pair.

    Parameters
    ----------
    charge : Array
        Nuclear charge that scales both hydrogenic orbitals; the JAX
        scalar stays differentiable.
    node_count : int
        Number of radial nodes on the [0, 43] Bohr grid.
    lebedev_degree : int
        Degree of the Lebedev angular rule.

    Returns
    -------
    length : Array
        Length-gauge 1s-to-2p amplitude for z polarization.
    momentum : Array
        Momentum-gauge 1s-to-2p amplitude for z polarization.

    Implementation Logic
    --------------------
    Samples the closed-form normalized 1s and 2p radial functions and
    the analytic 1s radial derivative on a Boole-weighted uniform grid,
    attaches the s and p_z harmonics, and accumulates both public
    Cartesian contractions over radius-times-direction chunks of 512
    radial nodes.
    """
    radius: float = 43.0
    radial_grid_numpy: Float64[NDArray, " n_node"] = np.linspace(
        0.0, radius, node_count
    )
    radial_weights_numpy: Float64[NDArray, " n_node"] = _boole_weights(
        node_count, radius
    )
    radial_grid: Array = jnp.asarray(radial_grid_numpy)
    radial_initial: Array = 2.0 * charge**1.5 * jnp.exp(-charge * radial_grid)
    radial_final: Array = (
        charge**1.5
        * (charge * radial_grid)
        * jnp.exp(-charge * radial_grid / 2.0)
        / (2.0 * math.sqrt(6.0))
    )
    radial_derivative: Array = -charge * radial_initial
    angular_points_numpy: Float64[NDArray, "3 n_angular"]
    angular_weights_numpy: Float64[NDArray, " n_angular"]
    angular_points_numpy, angular_weights_numpy = lebedev_rule(lebedev_degree)
    directions: Array = jnp.asarray(angular_points_numpy.T)
    angular_weights: Array = jnp.asarray(angular_weights_numpy)
    angular_count: int = directions.shape[0]
    polarization: Array = jnp.asarray((0.0, 0.0, 1.0), dtype=jnp.complex128)
    length: Array = jnp.asarray(0.0j)
    momentum: Array = jnp.asarray(0.0j)
    chunk_size: int = 512
    chunk_start: int
    for chunk_start in range(0, node_count, chunk_size):
        chunk_stop: int = min(chunk_start + chunk_size, node_count)
        radial_chunk: Array = radial_grid[chunk_start:chunk_stop]
        radial_weight_chunk: Array = jnp.asarray(radial_weights_numpy)[
            chunk_start:chunk_stop
        ]
        initial_chunk: Array = radial_initial[chunk_start:chunk_stop]
        final_chunk: Array = radial_final[chunk_start:chunk_stop]
        derivative_chunk: Array = radial_derivative[chunk_start:chunk_stop]
        chunk_count: int = chunk_stop - chunk_start
        radius_flat: Array = jnp.repeat(radial_chunk, angular_count)
        directions_flat: Array = jnp.tile(directions, (chunk_count, 1))
        position_flat: Array = radius_flat[:, None] * directions_flat
        volume_weights: Array = jnp.repeat(
            radial_weight_chunk * radial_chunk**2,
            angular_count,
        ) * jnp.tile(angular_weights, chunk_count)
        initial_flat: Array = jnp.repeat(
            initial_chunk / math.sqrt(4.0 * math.pi),
            angular_count,
        ).astype(jnp.complex128)
        final_flat: Array = (
            jnp.repeat(
                final_chunk * math.sqrt(3.0 / (4.0 * math.pi)),
                angular_count,
            )
            * directions_flat[:, 2]
        ).astype(jnp.complex128)
        derivative_flat: Array = jnp.repeat(
            derivative_chunk / math.sqrt(4.0 * math.pi),
            angular_count,
        )
        gradient_flat: Array = (
            derivative_flat[:, None] * directions_flat
        ).astype(jnp.complex128)
        length = length + dipole_length_cartesian(
            final_flat,
            initial_flat,
            position_flat,
            volume_weights,
            polarization,
        )
        momentum = momentum + dipole_momentum_cartesian(
            final_flat,
            gradient_flat,
            volume_weights,
            polarization,
        )
    return length, momentum


def _hydrogenic_reduced_public_gauges(
    charge: Array,
    node_count: int,
) -> Tuple[Array, Array]:
    """PRIVATE: Evaluate the analytic angular reduction through public APIs.

    Parameters
    ----------
    charge : Array
        Nuclear charge that scales both hydrogenic orbitals.
    node_count : int
        Number of radial nodes on the [0, 43] Bohr grid.

    Returns
    -------
    length : Array
        Length-gauge 1s-to-2p amplitude for z polarization.
    momentum : Array
        Momentum-gauge 1s-to-2p amplitude for z polarization.

    Implementation Logic
    --------------------
    Folds the exact angular integral 1/sqrt(3) into the position and
    gradient z components, so each public Cartesian contraction over
    the Boole-weighted radial grid equals the reduced one-dimensional
    radial integral. No angular quadrature remains.
    """
    radius: float = 43.0
    radial_grid: Array = jnp.linspace(0.0, radius, node_count)
    radial_weights: Array = jnp.asarray(_boole_weights(node_count, radius))
    radial_initial: Array = 2.0 * charge**1.5 * jnp.exp(-charge * radial_grid)
    radial_final: Array = (
        charge**1.5
        * (charge * radial_grid)
        * jnp.exp(-charge * radial_grid / 2.0)
        / (2.0 * math.sqrt(6.0))
    )
    positions: Array = jnp.stack(
        (
            jnp.zeros_like(radial_grid),
            jnp.zeros_like(radial_grid),
            radial_grid / math.sqrt(3.0),
        ),
        axis=-1,
    )
    gradient: Array = jnp.stack(
        (
            jnp.zeros_like(radial_grid),
            jnp.zeros_like(radial_grid),
            -charge * radial_initial / math.sqrt(3.0),
        ),
        axis=-1,
    ).astype(jnp.complex128)
    volume_weights: Array = radial_weights * radial_grid**2
    polarization: Array = jnp.asarray((0.0, 0.0, 1.0), dtype=jnp.complex128)
    length: Array = dipole_length_cartesian(
        radial_final.astype(jnp.complex128),
        radial_initial.astype(jnp.complex128),
        positions,
        volume_weights,
        polarization,
    )
    momentum: Array = dipole_momentum_cartesian(
        radial_final.astype(jnp.complex128),
        gradient,
        volume_weights,
        polarization,
    )
    return length, momentum


def _radial_second_derivative(
    node_count: int,
    radius: float,
    angular_momentum: int,
) -> csr_matrix:
    """PRIVATE: Build a centered sixth-order radial Dirichlet operator.

    Parameters
    ----------
    node_count : int
        Number of radial nodes, both boundary nodes included.
    radius : float
        Upper limit of the [0, radius] radial grid in Bohr.
    angular_momentum : int
        Angular momentum l; sets the reflection parity of the reduced
        radial function at the origin.

    Returns
    -------
    result : csr_matrix
        Sparse second-derivative operator on the interior nodes.

    Implementation Logic
    --------------------
    Assembles the seven-point sixth-order centered stencil divided by
    the spacing squared. Out-of-range samples reflect through the
    origin with sign (-1)**(l + 1) and through the outer wall with sign
    -1; contributions on the two Dirichlet boundary nodes are dropped.
    """
    spacing: float = radius / (node_count - 1)
    interior_count: int = node_count - 2
    operator: lil_matrix = lil_matrix(
        (interior_count, interior_count), dtype=np.float64
    )
    coefficients: Dict[int, float] = {
        -3: 1.0 / 90.0,
        -2: -3.0 / 20.0,
        -1: 3.0 / 2.0,
        0: -49.0 / 18.0,
        1: 3.0 / 2.0,
        2: -3.0 / 20.0,
        3: 1.0 / 90.0,
    }
    radial_index: int
    offset: int
    coefficient: float
    for radial_index in range(1, node_count - 1):
        for offset, coefficient in coefficients.items():
            sample_index: int = radial_index + offset
            sign: float = 1.0
            if sample_index < 0:
                sample_index = -sample_index
                sign = float((-1) ** (angular_momentum + 1))
            if sample_index > node_count - 1:
                sample_index = 2 * (node_count - 1) - sample_index
                sign = -1.0
            if sample_index not in (0, node_count - 1):
                operator[radial_index - 1, sample_index - 1] += (
                    sign * coefficient / spacing**2
                )
    result: csr_matrix = operator.tocsr()
    return result


@lru_cache(maxsize=4)
def _local_box_states(
    node_count: int,
    quadratic_coefficient: float,
) -> Tuple[
    Float64[NDArray, " n_node"],
    Float64[NDArray, " n_node"],
    Float64[NDArray, " 2"],
    Float64[NDArray, "2 n_node"],
]:
    """PRIVATE: Compute phase-pinned local anharmonic s and p radial states.

    Parameters
    ----------
    node_count : int
        Number of radial nodes on the [0, 40] Bohr grid.
    quadratic_coefficient : float
        Coefficient of the quadratic confinement term in Hartree per
        Bohr squared.

    Returns
    -------
    radial_grid : Float64[NDArray, " n_node"]
        Radial nodes in Bohr.
    radial_weights : Float64[NDArray, " n_node"]
        Boole quadrature weights in Bohr.
    energies : Float64[NDArray, " 2"]
        Ground s and p eigenvalues in Hartree.
    states : Float64[NDArray, "2 n_node"]
        Normalized reduced radial states with zero boundary values.

    Implementation Logic
    --------------------
    For l = 0 and l = 1, builds the sixth-order Dirichlet Hamiltonian
    with centrifugal, quadratic, and fixed quartic (0.001) terms,
    solves for the lowest eigenpair with shift-invert Lanczos at sigma
    0, embeds the interior eigenvector with zero boundaries, normalizes
    against the Boole weights, and pins the sign so the first interior
    sample is positive. The lru_cache reuses up to four grids.
    """
    radius: float = 40.0
    radial_grid: Float64[NDArray, " n_node"] = np.linspace(
        0.0, radius, node_count
    )
    radial_weights: Float64[NDArray, " n_node"] = _boole_weights(
        node_count, radius
    )
    interior_radius: Float64[NDArray, " n_interior"] = radial_grid[1:-1]
    energies: list[float] = []
    states: list[Float64[NDArray, " n_node"]] = []
    angular_momentum: int
    for angular_momentum in (0, 1):
        second_derivative: csr_matrix = _radial_second_derivative(
            node_count, radius, angular_momentum
        )
        potential: Float64[NDArray, " n_interior"] = (
            angular_momentum
            * (angular_momentum + 1)
            / (2.0 * interior_radius**2)
            + quadratic_coefficient * interior_radius**2
            + 0.001 * interior_radius**4
        )
        hamiltonian: csr_matrix = -0.5 * second_derivative + diags(potential)
        eigenvalues: Float64[NDArray, " 1"]
        eigenvectors: Float64[NDArray, "n_interior 1"]
        eigenvalues, eigenvectors = eigsh(
            hamiltonian,
            k=1,
            sigma=0.0,
            which="LM",
            tol=1e-14,
            maxiter=100_000,
        )
        state: Float64[NDArray, " n_node"] = np.zeros(node_count)
        state[1:-1] = eigenvectors[:, 0]
        state /= np.sqrt(np.sum(radial_weights * state * state))
        if state[1] < 0.0:
            state *= -1.0
        energies.append(float(eigenvalues[0]))
        states.append(state)
    return (
        radial_grid,
        radial_weights,
        np.asarray(energies),
        np.asarray(states),
    )


def _derivative_sixth(
    values: Float64[NDArray, " n_node"], spacing: float
) -> Float64[NDArray, " n_node"]:
    """PRIVATE: Differentiate an array with sixth-order seven-point stencils.

    Parameters
    ----------
    values : Float64[NDArray, " n_node"]
        Samples on a uniform grid.
    spacing : float
        Uniform grid spacing in Bohr.

    Returns
    -------
    derivative : Float64[NDArray, " n_node"]
        First-derivative samples at every node.

    Notes
    -----
    For each node, clamps a seven-point window inside the grid, solves
    the Vandermonde moment system for the first-derivative weights at
    that offset, and applies the weights to the windowed samples.
    """
    node_count: int = values.size
    derivative: Float64[NDArray, " n_node"] = np.empty_like(values)
    index: int
    for index in range(node_count):
        start: int = min(max(index - 3, 0), node_count - 7)
        stencil_indices: Int64[NDArray, " 7"] = np.arange(start, start + 7)
        offsets: Float64[NDArray, " 7"] = (stencil_indices - index) * spacing
        moment_matrix: Float64[NDArray, "7 7"] = np.vander(
            offsets, 7, increasing=True
        ).T
        target: Float64[NDArray, " 7"] = np.zeros(7)
        target[1] = 1.0
        coefficients: Float64[NDArray, " 7"] = np.linalg.solve(
            moment_matrix, target
        )
        derivative[index] = coefficients @ values[stencil_indices]
    return derivative


class TestPolarizationCartToComplex:
    """Validate the Cartesian-to-complex spherical polarization map.

    :see: :func:`~diffpes.maths.polarization_cart_to_complex`
    """

    def test_basis_vectors_and_helicities(self) -> None:
        """Match the pinned Condon--Shortley component formulas.

        The test covers Cartesian basis vectors and both complex helicities.

        Notes
        -----
        The test compares each transformed vector with its analytic components.
        """
        inverse_sqrt_two: float = 1.0 / math.sqrt(2.0)
        x_result: Array = polarization_cart_to_complex(
            jnp.asarray((1.0, 0.0, 0.0), dtype=jnp.complex128)
        )
        y_result: Array = polarization_cart_to_complex(
            jnp.asarray((0.0, 1.0, 0.0), dtype=jnp.complex128)
        )
        z_result: Array = polarization_cart_to_complex(
            jnp.asarray((0.0, 0.0, 1.0), dtype=jnp.complex128)
        )
        chex.assert_trees_all_close(
            x_result,
            jnp.asarray(
                (inverse_sqrt_two, 0.0, -inverse_sqrt_two),
                dtype=jnp.complex128,
            ),
            rtol=1e-14,
            atol=1e-14,
        )
        chex.assert_trees_all_close(
            y_result,
            jnp.asarray(
                (-1j * inverse_sqrt_two, 0.0, -1j * inverse_sqrt_two),
                dtype=jnp.complex128,
            ),
            rtol=1e-14,
            atol=1e-14,
        )
        chex.assert_trees_all_close(
            z_result,
            jnp.asarray((0.0, 1.0, 0.0), dtype=jnp.complex128),
            rtol=1e-14,
            atol=1e-14,
        )
        positive_helicity: Array = (
            jnp.asarray((1.0, 1j, 0.0), dtype=jnp.complex128)
            * inverse_sqrt_two
        )
        negative_helicity: Array = jnp.conj(positive_helicity)
        positive_result: Array = polarization_cart_to_complex(
            positive_helicity
        )
        negative_result: Array = polarization_cart_to_complex(
            negative_helicity
        )
        chex.assert_trees_all_close(
            positive_result,
            jnp.asarray((1.0, 0.0, 0.0), dtype=jnp.complex128),
            rtol=1e-14,
            atol=1e-14,
        )
        chex.assert_trees_all_close(
            negative_result,
            jnp.asarray((0.0, 0.0, -1.0), dtype=jnp.complex128),
            rtol=1e-14,
            atol=1e-14,
        )

    def test_wrong_real_permutation_is_not_complex_map(self) -> None:
        """Reject interpreting ``(y,z,x)`` as complex spherical channels.

        A generic elliptic vector distinguishes a permutation from a unitary map.

        Notes
        -----
        The test compares both mappings at a strict numerical tolerance.
        """
        polarization: Array = _generic_polarization()
        correct: Array = polarization_cart_to_complex(polarization)
        wrong: Array = polarization_cart_to_real(polarization)
        assert not bool(jnp.allclose(correct, wrong, rtol=1e-12, atol=1e-12))

    def test_cartesian_real_and_complex_amplitudes_agree(self) -> None:
        """Match one bilinear amplitude in all three photon bases.

        The complex basis uses the spherical metric from the canonical contraction.

        Notes
        -----
        The test transforms two generic vectors and evaluates each direct formula.
        """
        polarization: Array = _generic_polarization()
        dipole_cart: Array = jnp.asarray(
            (-0.19 + 0.27j, 0.43 - 0.11j, 0.37 + 0.52j),
            dtype=jnp.complex128,
        )
        polarization_real: Array = polarization_cart_to_real(polarization)
        dipole_real: Array = polarization_cart_to_real(dipole_cart)
        polarization_complex: Array = polarization_cart_to_complex(
            polarization
        )
        dipole_complex: Array = polarization_cart_to_complex(dipole_cart)
        cartesian_amplitude: Array = polarization @ dipole_cart
        real_amplitude: Array = polarization_real @ dipole_real
        complex_amplitude: Array = (
            -polarization_complex[0] * dipole_complex[2]
            + polarization_complex[1] * dipole_complex[1]
            - polarization_complex[2] * dipole_complex[0]
        )
        chex.assert_trees_all_close(
            real_amplitude,
            cartesian_amplitude,
            rtol=1e-14,
            atol=1e-14,
        )
        chex.assert_trees_all_close(
            complex_amplitude,
            cartesian_amplitude,
            rtol=1e-14,
            atol=1e-14,
        )


class TestPolarizationComplexToCart:
    """Validate the inverse complex-spherical polarization map.

    :see: :func:`~diffpes.maths.polarization_complex_to_cart`
    """

    def test_generic_complex_round_trip_and_jit(self) -> None:
        """Round-trip a generic elliptic vector through a jitted transform.

        The same fixture also checks norm preservation under the unitary map.

        Notes
        -----
        The test applies both compiled functions and compares their outputs.
        """
        polarization: Array = _generic_polarization()
        transformed: Array = jax.jit(polarization_cart_to_complex)(
            polarization
        )
        recovered: Array = jax.jit(polarization_complex_to_cart)(transformed)
        chex.assert_trees_all_close(
            recovered,
            polarization,
            rtol=1e-14,
            atol=1e-14,
        )
        chex.assert_trees_all_close(
            jnp.linalg.norm(transformed),
            jnp.linalg.norm(polarization),
            rtol=1e-14,
            atol=1e-14,
        )


class TestPolarizationCartToReal:
    """Validate the Cartesian-to-real harmonic polarization permutation.

    :see: :func:`~diffpes.maths.polarization_cart_to_real`
    """

    def test_real_channel_order(self) -> None:
        """Verify the pinned ``(x,y,z)`` to ``(y,z,x)`` mapping.

        A generic complex vector makes every channel independently visible.

        Notes
        -----
        The test compares production with direct integer indexing.
        """
        polarization: Array = _generic_polarization()
        transformed: Array = polarization_cart_to_real(polarization)
        expected: Array = polarization[jnp.asarray((1, 2, 0))]
        chex.assert_trees_all_close(
            transformed,
            expected,
            rtol=0.0,
            atol=0.0,
        )


class TestPolarizationRealToCart:
    """Validate the inverse real-harmonic polarization permutation.

    :see: :func:`~diffpes.maths.polarization_real_to_cart`
    """

    def test_generic_complex_round_trip(self) -> None:
        """Round-trip a generic elliptic vector exactly.

        The permutation and its inverse preserve every complex component.

        Notes
        -----
        The test applies both public maps and requests exact equality.
        """
        polarization: Array = _generic_polarization()
        recovered: Array = polarization_real_to_cart(
            polarization_cart_to_real(polarization)
        )
        chex.assert_trees_all_close(
            recovered,
            polarization,
            rtol=0.0,
            atol=0.0,
        )


class TestChannelTables:
    """Validate the padded real-basis dipole coupling tensors.

    :see: :func:`~diffpes.maths.channel_tables`
    """

    def test_complete_blocks_and_exact_coefficients(self) -> None:
        """Retain every final real harmonic and match the Gaunt table.

        The fixture spans initial angular momenta from zero through four.

        Notes
        -----
        The test walks every valid block and checks its flattened index.
        """
        basis: OrbitalBasis = make_orbital_basis(
            atom_indices=(0, 0, 0, 0),
            n=(1, 2, 3, 5),
            l=(0, 1, 2, 4),
            m=(0, -1, 1, -4),
            labels=("s", "py", "dxz", "g"),
        )
        coupling: Array
        valid: Array
        coupling, valid = channel_tables(basis)
        assert coupling.shape == (4, 2, 3, 36)
        assert valid.shape == coupling.shape
        assert bool(jnp.all((valid == 0.0) | (valid == 1.0)))

        orbital_index: int
        l_initial: int
        m_initial: int
        branch_index: int
        l_final: int
        q_index: int
        m_final: int
        harmonic_index: int
        expected: Array
        for orbital_index, (l_initial, m_initial) in enumerate(
            zip(basis.l, basis.m, strict=True)
        ):
            for branch_index, l_final in enumerate(
                (l_initial - 1, l_initial + 1)
            ):
                for q_index in range(3):
                    for m_final in range(-5, 6):
                        if l_final >= 0 and abs(m_final) <= l_final:
                            harmonic_index = (
                                l_final * l_final + m_final + l_final
                            )
                            expected = GAUNT_TABLE[
                                l_initial,
                                m_initial + L_MAX,
                                q_index,
                                l_final,
                                m_final + L_MAX + 1,
                            ]
                            assert (
                                valid[
                                    orbital_index,
                                    branch_index,
                                    q_index,
                                    harmonic_index,
                                ]
                                == 1.0
                            )
                            chex.assert_trees_all_close(
                                coupling[
                                    orbital_index,
                                    branch_index,
                                    q_index,
                                    harmonic_index,
                                ],
                                expected,
                                rtol=0.0,
                                atol=0.0,
                            )

        assert int(jnp.sum(valid[0, 0])) == 0
        assert int(jnp.sum(valid[0, 1])) == 9
        assert int(jnp.sum(valid[3, 0])) == 21
        assert int(jnp.sum(valid[3, 1])) == 33

    def test_real_basis_does_not_apply_complex_delta_m_shortcut(self) -> None:
        """Keep real couplings beyond the complex-label magnetic shortcut.

        A real ``p_y`` row mixes complex magnetic components by construction.

        Notes
        -----
        The test finds a nonzero entry that violates the false label equation.
        """
        basis: OrbitalBasis = make_orbital_basis(
            atom_indices=(0,),
            n=(2,),
            l=(1,),
            m=(-1,),
            labels=("py",),
        )
        coupling: Array
        valid: Array
        coupling, valid = channel_tables(basis)
        del valid
        nonzero_indices: Int64[NDArray, "n_nonzero 2"] = np.argwhere(
            np.abs(np.asarray(coupling[0, 1])) > 1e-14
        )
        violates_complex_shortcut: bool = any(
            (int(index[1]) - 4) != (-1 + (int(index[0]) - 1))
            for index in nonzero_indices
        )
        assert violates_complex_shortcut

    def test_no_plane_wave_phase_is_hidden_in_table(self) -> None:
        """Keep every static channel coefficient exactly real.

        The radial layer owns the complete plane-wave partial-wave phase.

        Notes
        -----
        The test inspects the dtype of a nontrivial angular table.
        """
        basis: OrbitalBasis = make_orbital_basis(
            atom_indices=(0,),
            n=(3,),
            l=(2,),
            m=(0,),
            labels=("dz2",),
        )
        coupling: Array
        valid: Array
        coupling, valid = channel_tables(basis)
        del valid
        assert coupling.dtype == jnp.float64

    def test_host_static_gaunt_lookup_survives_outer_tracing(self) -> None:
        """Build channel tables inside JAX tracing without concretization.

        A compiled zero-argument callable returns both arrays while retaining
        the closed-over static orbital basis.

        Notes
        -----
        The test constructs a JAX expression and executes its JIT-compiled
        form, catching accidental scalar conversion of traced Gaunt entries.
        """
        basis: OrbitalBasis = make_orbital_basis(
            atom_indices=(0, 0),
            n=(2, 3),
            l=(1, 2),
            m=(-1, 1),
            labels=("py", "dxz"),
        )

        def build_tables() -> Tuple[Array, Array]:
            result: Tuple[Array, Array] = channel_tables(basis)
            return result

        expression: Any = jax.make_jaxpr(build_tables)()
        assert expression.out_avals[0].shape == (2, 2, 3, 36)
        coupling: Array
        valid: Array
        coupling, valid = jax.jit(build_tables)()
        assert coupling.shape == (2, 2, 3, 36)
        assert valid.shape == coupling.shape


class TestDipoleLengthCartesian:
    """Validate the sampled Cartesian length-gauge contraction.

    :see: :func:`~diffpes.maths.dipole_length_cartesian`
    """

    def test_independent_direct_sum_and_phase_covariance(self) -> None:
        """Match NumPy and retain the final-bra and initial-ket phase.

        Generic complex samples expose missing conjugation or an extra conjugation.

        Notes
        -----
        The test evaluates an explicit sum and then rotates both ket phases.
        """
        psi_final: Array = jnp.asarray(
            (0.2 + 0.7j, -0.4 + 0.1j, 0.8 - 0.3j),
            dtype=jnp.complex128,
        )
        psi_initial: Array = jnp.asarray(
            (-0.5 + 0.2j, 0.3 + 0.9j, -0.1 - 0.6j),
            dtype=jnp.complex128,
        )
        positions: Array = jnp.asarray(
            ((0.1, -0.2, 0.4), (0.7, 0.3, -0.1), (-0.5, 0.8, 0.2)),
            dtype=jnp.float64,
        )
        weights: Array = jnp.asarray((0.2, 0.5, 0.3), dtype=jnp.float64)
        polarization: Array = _generic_polarization()
        actual: Array = dipole_length_cartesian(
            psi_final,
            psi_initial,
            positions,
            weights,
            polarization,
        )
        expected: complex = np.sum(
            np.asarray(weights)
            * np.conj(np.asarray(psi_final))
            * (np.asarray(positions) @ np.asarray(polarization))
            * np.asarray(psi_initial)
        )
        chex.assert_trees_all_close(
            actual,
            expected,
            rtol=1e-14,
            atol=1e-14,
        )

        alpha: float = 0.37
        beta: float = -0.61
        transformed: Array = dipole_length_cartesian(
            jnp.exp(1j * alpha) * psi_final,
            jnp.exp(1j * beta) * psi_initial,
            positions,
            weights,
            polarization,
        )
        chex.assert_trees_all_close(
            transformed,
            jnp.exp(1j * (beta - alpha)) * actual,
            rtol=1e-14,
            atol=1e-14,
        )

    def test_generic_complex_directional_derivative(self) -> None:
        """Match a central directional quotient for initial-state samples.

        The direction contains independent real and imaginary components.

        Notes
        -----
        The test compares a JAX tangent with a symmetric finite difference.
        """
        psi_final: Array = jnp.asarray(
            (0.2 + 0.7j, -0.4 + 0.1j), dtype=jnp.complex128
        )
        psi_initial: Array = jnp.asarray(
            (-0.5 + 0.2j, 0.3 + 0.9j), dtype=jnp.complex128
        )
        direction: Array = jnp.asarray(
            (0.1 - 0.3j, -0.2 + 0.4j), dtype=jnp.complex128
        )
        positions: Array = jnp.asarray(
            ((0.1, -0.2, 0.4), (0.7, 0.3, -0.1)), dtype=jnp.float64
        )
        weights: Array = jnp.asarray((0.2, 0.5), dtype=jnp.float64)
        polarization: Array = _generic_polarization()

        def amplitude(initial: Array) -> Array:
            result: Array = dipole_length_cartesian(
                psi_final,
                initial,
                positions,
                weights,
                polarization,
            )
            return result

        jvp_result: Tuple[Array, Array] = jax.jvp(
            amplitude, (psi_initial,), (direction,)
        )
        tangent: Array = jvp_result[1]
        step: float = 1e-5
        finite_difference: Array = (
            amplitude(psi_initial + step * direction)
            - amplitude(psi_initial - step * direction)
        ) / (2.0 * step)
        chex.assert_trees_all_close(
            tangent,
            finite_difference,
            rtol=1e-10,
            atol=1e-11,
        )


class TestDipoleMomentumCartesian:
    """Validate the sampled Cartesian momentum-gauge contraction.

    :see: :func:`~diffpes.maths.dipole_momentum_cartesian`
    """

    def test_independent_direct_sum_and_phase_covariance(self) -> None:
        """Match NumPy including ``-i`` and preserve both ket phases.

        Generic complex gradient rows expose sign and conjugation mistakes.

        Notes
        -----
        The test evaluates an explicit sum and then rotates both ket phases.
        """
        psi_final: Array = jnp.asarray(
            (0.2 + 0.7j, -0.4 + 0.1j, 0.8 - 0.3j),
            dtype=jnp.complex128,
        )
        gradient: Array = jnp.asarray(
            (
                (0.1 + 0.3j, -0.2 + 0.4j, 0.5 - 0.1j),
                (-0.7 + 0.2j, 0.4 + 0.8j, 0.3 - 0.5j),
                (0.6 - 0.9j, -0.1 + 0.2j, 0.7 + 0.4j),
            ),
            dtype=jnp.complex128,
        )
        weights: Array = jnp.asarray((0.2, 0.5, 0.3), dtype=jnp.float64)
        polarization: Array = _generic_polarization()
        actual: Array = dipole_momentum_cartesian(
            psi_final,
            gradient,
            weights,
            polarization,
        )
        expected: complex = -1j * np.sum(
            np.asarray(weights)
            * np.conj(np.asarray(psi_final))
            * (np.asarray(gradient) @ np.asarray(polarization))
        )
        chex.assert_trees_all_close(
            actual,
            expected,
            rtol=1e-14,
            atol=1e-14,
        )

        alpha: float = 0.37
        beta: float = -0.61
        transformed: Array = dipole_momentum_cartesian(
            jnp.exp(1j * alpha) * psi_final,
            jnp.exp(1j * beta) * gradient,
            weights,
            polarization,
        )
        chex.assert_trees_all_close(
            transformed,
            jnp.exp(1j * (beta - alpha)) * actual,
            rtol=1e-14,
            atol=1e-14,
        )

    def test_length_commutator_is_not_used_internally(self) -> None:
        """Show the momentum contraction accepts an unrelated gradient.

        A zero gradient must return zero without consulting length-gauge data.

        Notes
        -----
        The test supplies no position or initial-wavefunction samples.
        """
        psi_final: Array = jnp.asarray((1.0 + 0.2j,), dtype=jnp.complex128)
        zero_gradient: Array = jnp.zeros((1, 3), dtype=jnp.complex128)
        weights: Array = jnp.ones((1,), dtype=jnp.float64)
        polarization: Array = _generic_polarization()
        amplitude: Array = dipole_momentum_cartesian(
            psi_final,
            zero_gradient,
            weights,
            polarization,
        )
        chex.assert_trees_all_close(amplitude, 0.0j, rtol=0.0, atol=0.0)


class TestGaugeEquivalenceBattery:
    """Validate the frozen G12/D12 local-potential gauge battery."""

    def test_generic_complex_phase_directional_covariance(self) -> None:
        """Differentiate both phase-covariance laws on unrelated samples.

        The final ket carries the conjugated phase while the initial ket and
        its gradient carry the unconjugated phase.

        Notes
        -----
        The test compares each JAX directional derivative with the analytic
        derivative and an independent centered quotient.
        """
        psi_final: Array = jnp.asarray(
            (0.2 + 0.7j, -0.4 + 0.1j, 0.8 - 0.3j),
            dtype=jnp.complex128,
        )
        psi_initial: Array = jnp.asarray(
            (-0.5 + 0.2j, 0.3 + 0.9j, -0.1 - 0.6j),
            dtype=jnp.complex128,
        )
        gradient: Array = jnp.asarray(
            (
                (0.1 + 0.3j, -0.2 + 0.4j, 0.5 - 0.1j),
                (-0.7 + 0.2j, 0.4 + 0.8j, 0.3 - 0.5j),
                (0.6 - 0.9j, -0.1 + 0.2j, 0.7 + 0.4j),
            ),
            dtype=jnp.complex128,
        )
        positions: Array = jnp.asarray(
            ((0.1, -0.2, 0.4), (0.7, 0.3, -0.1), (-0.5, 0.8, 0.2)),
            dtype=jnp.float64,
        )
        weights: Array = jnp.asarray((0.2, 0.5, 0.3), dtype=jnp.float64)
        polarization: Array = _generic_polarization()

        def amplitudes(phases: Array) -> Array:
            final_phase: Array = jnp.exp(1j * phases[0])
            initial_phase: Array = jnp.exp(1j * phases[1])
            length: Array = dipole_length_cartesian(
                final_phase * psi_final,
                initial_phase * psi_initial,
                positions,
                weights,
                polarization,
            )
            momentum: Array = dipole_momentum_cartesian(
                final_phase * psi_final,
                initial_phase * gradient,
                weights,
                polarization,
            )
            result: Array = jnp.stack((length, momentum))
            return result

        phases: Array = jnp.asarray((0.37, -0.61))
        direction: Array = jnp.asarray((0.19, -0.31))
        values: Array
        tangent: Array
        values, tangent = jax.jvp(amplitudes, (phases,), (direction,))
        expected_tangent: Array = 1j * (direction[1] - direction[0]) * values
        step: float = 2.0**-16
        finite_difference: Array = (
            amplitudes(phases + step * direction)
            - amplitudes(phases - step * direction)
        ) / (2.0 * step)
        chex.assert_trees_all_close(
            tangent,
            expected_tangent,
            rtol=1e-12,
            atol=1e-12,
        )
        chex.assert_trees_all_close(
            tangent,
            finite_difference,
            rtol=1e-10,
            atol=1e-11,
        )

    def test_hydrogenic_values_and_charge_derivatives(self) -> None:
        """Pass exact local gauge equality and Z derivatives at four nodes.

        The amended diffuse-state profile uses 8193 Boole radial nodes on
        ``[0,43]`` and an independently derived analytic angular reduction.

        Notes
        -----
        The test passes reduced arrays through the public Cartesian APIs. It
        compares exact values, analytic derivatives, JAX Jacobians, and
        centered finite differences. The separate refinement test exercises
        the full Lebedev product.
        """
        analytic_length_scale: float = 128.0 * math.sqrt(2.0) / 243.0
        analytic_momentum_scale: float = 16.0 * math.sqrt(2.0) / 81.0
        charge_value: float
        for charge_value in (0.5, 1.0, 2.0, 3.0):
            charge: Array = jnp.asarray(charge_value)

            def real_gauges(variable_charge: Array) -> Array:
                length: Array
                momentum: Array
                length, momentum = _hydrogenic_reduced_public_gauges(
                    variable_charge,
                    8193,
                )
                result: Array = jnp.stack(
                    (
                        length.real,
                        length.imag,
                        momentum.real,
                        momentum.imag,
                    )
                )
                return result

            actual: Array = real_gauges(charge)
            exact: Array = jnp.asarray(
                (
                    analytic_length_scale / charge_value,
                    0.0,
                    0.0,
                    analytic_momentum_scale * charge_value,
                )
            )
            chex.assert_trees_all_close(
                actual,
                exact,
                rtol=1e-9,
                atol=1e-12,
            )
            commutator_error: Array = jnp.abs(
                actual[3] - (3.0 * charge**2 / 8.0) * actual[0]
            )
            assert float(commutator_error) <= 1e-10

            expected_derivative: Array = jnp.asarray(
                (
                    -analytic_length_scale / charge_value**2,
                    0.0,
                    0.0,
                    analytic_momentum_scale,
                )
            )
            derivative_forward: Array = jax.jacfwd(real_gauges)(charge)
            derivative_reverse: Array = jax.jacrev(real_gauges)(charge)
            step: float = max(1.0, charge_value) * 2.0**-16
            derivative_finite: Array = (
                real_gauges(charge + step) - real_gauges(charge - step)
            ) / (2.0 * step)
            chex.assert_trees_all_close(
                derivative_forward,
                expected_derivative,
                rtol=1e-7,
                atol=1e-10,
            )
            chex.assert_trees_all_close(
                derivative_reverse,
                expected_derivative,
                rtol=1e-7,
                atol=1e-10,
            )
            chex.assert_trees_all_close(
                derivative_forward,
                derivative_finite,
                rtol=1e-7,
                atol=1e-10,
            )

    def test_hydrogenic_product_grid_refinement(self) -> None:
        """Compare diffuse and sharp endpoint rows under both refinements.

        Compare the coarse 8193-by-110 Cartesian product with the independent
        16385-by-194 product for Z equal to 0.5 and 3.

        Notes
        -----
        Bound the mixed residual separately for both public gauge amplitudes.
        """
        charge_value: float
        for charge_value in (0.5, 3.0):
            coarse_length: Array
            coarse_momentum: Array
            coarse_length, coarse_momentum = _hydrogenic_gauges(
                jnp.asarray(charge_value),
                8193,
                17,
            )
            fine_length: Array
            fine_momentum: Array
            fine_length, fine_momentum = _hydrogenic_gauges(
                jnp.asarray(charge_value),
                16385,
                23,
            )
            coarse: Array = jnp.stack((coarse_length, coarse_momentum))
            fine: Array = jnp.stack((fine_length, fine_momentum))
            mixed_error: Array = jnp.abs(coarse - fine) / (
                1.0 + jnp.maximum(jnp.abs(coarse), jnp.abs(fine))
            )
            assert float(jnp.max(mixed_error)) <= 1e-10

    def test_local_anharmonic_box_cartesian_and_reduced_gauges(self) -> None:
        """Pass local gauge equality on the frozen numerical box fixture.

        The lowest l=0 and l=1 states come from an independent centered
        sixth-order radial Hamiltonian on 4097 Dirichlet nodes.

        Notes
        -----
        Compare the full 4097-by-110 public contractions with an analytic
        angular reduction before applying the local commutator identity.
        """
        radial_grid: Float64[NDArray, " n_node"]
        radial_weights: Float64[NDArray, " n_node"]
        energies: Float64[NDArray, " 2"]
        states: Float64[NDArray, "2 n_node"]
        radial_grid, radial_weights, energies, states = _local_box_states(
            4097, 0.02
        )
        state_s: Float64[NDArray, " n_node"] = states[0]
        state_p: Float64[NDArray, " n_node"] = states[1]
        spacing: float = radial_grid[1] - radial_grid[0]
        derivative_s: Float64[NDArray, " n_node"] = _derivative_sixth(
            state_s, spacing
        )
        radial_initial: Float64[NDArray, " n_node"] = np.divide(
            state_s,
            radial_grid,
            out=np.zeros_like(state_s),
            where=radial_grid > 0.0,
        )
        radial_final: Float64[NDArray, " n_node"] = np.divide(
            state_p,
            radial_grid,
            out=np.zeros_like(state_p),
            where=radial_grid > 0.0,
        )
        radial_initial_derivative: Float64[NDArray, " n_node"] = np.divide(
            derivative_s * radial_grid - state_s,
            radial_grid**2,
            out=np.zeros_like(state_s),
            where=radial_grid > 0.0,
        )
        length: Array
        momentum: Array
        length, momentum = _tensor_product_gauges(
            radial_grid,
            radial_weights,
            radial_initial,
            radial_initial_derivative,
            radial_final,
            17,
        )
        safe_ratio: Float64[NDArray, " n_node"] = np.divide(
            state_s,
            radial_grid,
            out=np.zeros_like(state_s),
            where=radial_grid > 0.0,
        )
        reduced_length: complex = complex(
            np.sum(radial_weights * state_p * state_s * radial_grid)
            / math.sqrt(3.0)
        )
        reduced_momentum: complex = complex(
            -1j
            * np.sum(radial_weights * state_p * (derivative_s - safe_ratio))
            / math.sqrt(3.0)
        )
        chex.assert_trees_all_close(
            length,
            reduced_length,
            rtol=1e-10,
            atol=1e-10,
        )
        chex.assert_trees_all_close(
            momentum,
            reduced_momentum,
            rtol=1e-10,
            atol=1e-10,
        )
        commutator: complex = 1j * (energies[1] - energies[0]) * reduced_length
        chex.assert_trees_all_close(
            reduced_momentum,
            commutator,
            rtol=1e-9,
            atol=1e-12,
        )

        fine_grid: Float64[NDArray, " n_fine"]
        fine_weights: Float64[NDArray, " n_fine"]
        fine_energies: Float64[NDArray, " 2"]
        fine_states: Float64[NDArray, "2 n_fine"]
        fine_grid, fine_weights, fine_energies, fine_states = (
            _local_box_states(8193, 0.02)
        )
        del fine_energies
        fine_state_s: Float64[NDArray, " n_fine"] = fine_states[0]
        fine_state_p: Float64[NDArray, " n_fine"] = fine_states[1]
        fine_derivative_s: Float64[NDArray, " n_fine"] = _derivative_sixth(
            fine_state_s, fine_grid[1] - fine_grid[0]
        )
        fine_radial_initial: Float64[NDArray, " n_fine"] = np.divide(
            fine_state_s,
            fine_grid,
            out=np.zeros_like(fine_state_s),
            where=fine_grid > 0.0,
        )
        fine_radial_final: Float64[NDArray, " n_fine"] = np.divide(
            fine_state_p,
            fine_grid,
            out=np.zeros_like(fine_state_p),
            where=fine_grid > 0.0,
        )
        fine_radial_derivative: Float64[NDArray, " n_fine"] = np.divide(
            fine_derivative_s * fine_grid - fine_state_s,
            fine_grid**2,
            out=np.zeros_like(fine_state_s),
            where=fine_grid > 0.0,
        )
        fine_length: Array
        fine_momentum: Array
        fine_length, fine_momentum = _tensor_product_gauges(
            fine_grid,
            fine_weights,
            fine_radial_initial,
            fine_radial_derivative,
            fine_radial_final,
            23,
        )
        coarse_gauges: Array = jnp.stack((length, momentum))
        fine_gauges: Array = jnp.stack((fine_length, fine_momentum))
        mixed_refinement: Array = jnp.abs(coarse_gauges - fine_gauges) / (
            1.0
            + jnp.maximum(
                jnp.abs(coarse_gauges),
                jnp.abs(fine_gauges),
            )
        )
        assert float(jnp.max(mixed_refinement)) <= 1e-10

    @pytest.mark.rss_limit_mb(768)
    def test_local_quadratic_coefficient_derivatives(self) -> None:
        """Match JAX gauge derivatives for the local box coefficient.

        Pass independent central-difference eigenstate tangents through the
        public Cartesian contractions at the frozen box point.

        Notes
        -----
        Forward mode, reverse mode, a second centered quotient, and the
        differentiated local commutator identity must all agree.
        """
        coefficient: float = 0.02
        coefficient_step: float = 2.0e-5
        radial_grid: Float64[NDArray, " n_node"]
        radial_weights: Float64[NDArray, " n_node"]
        energies: Float64[NDArray, " 2"]
        states: Float64[NDArray, "2 n_node"]
        radial_grid, radial_weights, energies, states = _local_box_states(
            4097, coefficient
        )
        minus_grid: Float64[NDArray, " n_node"]
        minus_weights: Float64[NDArray, " n_node"]
        minus_energies: Float64[NDArray, " 2"]
        minus_states: Float64[NDArray, "2 n_node"]
        minus_grid, minus_weights, minus_energies, minus_states = (
            _local_box_states(4097, coefficient - coefficient_step)
        )
        plus_grid: Float64[NDArray, " n_node"]
        plus_weights: Float64[NDArray, " n_node"]
        plus_energies: Float64[NDArray, " 2"]
        plus_states: Float64[NDArray, "2 n_node"]
        plus_grid, plus_weights, plus_energies, plus_states = (
            _local_box_states(4097, coefficient + coefficient_step)
        )
        np.testing.assert_array_equal(minus_grid, radial_grid)
        np.testing.assert_array_equal(plus_grid, radial_grid)
        np.testing.assert_array_equal(minus_weights, radial_weights)
        np.testing.assert_array_equal(plus_weights, radial_weights)
        state_tangents: Float64[NDArray, "2 n_node"] = (
            plus_states - minus_states
        ) / (2.0 * coefficient_step)
        spacing: float = radial_grid[1] - radial_grid[0]

        def radial_function(
            state: Float64[NDArray, " n_node"],
        ) -> Float64[NDArray, " n_node"]:
            result: Float64[NDArray, " n_node"] = np.divide(
                state,
                radial_grid,
                out=np.zeros_like(state),
                where=radial_grid > 0.0,
            )
            return result

        def radial_function_derivative(
            state: Float64[NDArray, " n_node"],
        ) -> Float64[NDArray, " n_node"]:
            derivative: Float64[NDArray, " n_node"] = _derivative_sixth(
                state, spacing
            )
            result: Float64[NDArray, " n_node"] = np.divide(
                derivative * radial_grid - state,
                radial_grid**2,
                out=np.zeros_like(state),
                where=radial_grid > 0.0,
            )
            return result

        angular_points: Float64[NDArray, "3 n_angular"]
        angular_weights: Float64[NDArray, " n_angular"]
        angular_points, angular_weights = lebedev_rule(17)
        directions_numpy: Float64[NDArray, "n_angular 3"] = angular_points.T
        angular_count: int = directions_numpy.shape[0]
        directions: Array = jnp.asarray(
            np.tile(directions_numpy, (radial_grid.size, 1))
        )
        radius_flat: Array = jnp.asarray(np.repeat(radial_grid, angular_count))
        positions: Array = radius_flat[:, None] * directions
        weights: Array = jnp.asarray(
            np.repeat(
                radial_weights * radial_grid**2,
                angular_count,
            )
            * np.tile(angular_weights, radial_grid.size)
        )
        y00: float = 1.0 / math.sqrt(4.0 * math.pi)
        y10_scale: float = math.sqrt(3.0 / (4.0 * math.pi))
        initial_base: Array = jnp.asarray(
            np.repeat(radial_function(states[0]) * y00, angular_count),
            dtype=jnp.complex128,
        )
        initial_tangent: Array = jnp.asarray(
            np.repeat(
                radial_function(state_tangents[0]) * y00,
                angular_count,
            ),
            dtype=jnp.complex128,
        )
        final_base: Array = jnp.asarray(
            np.repeat(
                radial_function(states[1]) * y10_scale,
                angular_count,
            )
            * np.asarray(directions[:, 2]),
            dtype=jnp.complex128,
        )
        final_tangent: Array = jnp.asarray(
            np.repeat(
                radial_function(state_tangents[1]) * y10_scale,
                angular_count,
            )
            * np.asarray(directions[:, 2]),
            dtype=jnp.complex128,
        )
        gradient_base: Array = jnp.asarray(
            np.repeat(
                radial_function_derivative(states[0]) * y00,
                angular_count,
            )[:, None]
            * np.asarray(directions),
            dtype=jnp.complex128,
        )
        gradient_tangent: Array = jnp.asarray(
            np.repeat(
                radial_function_derivative(state_tangents[0]) * y00,
                angular_count,
            )[:, None]
            * np.asarray(directions),
            dtype=jnp.complex128,
        )
        polarization: Array = jnp.asarray(
            (0.0, 0.0, 1.0), dtype=jnp.complex128
        )

        def real_gauges(offset: Array) -> Array:
            length: Array = dipole_length_cartesian(
                final_base + offset * final_tangent,
                initial_base + offset * initial_tangent,
                positions,
                weights,
                polarization,
            )
            momentum: Array = dipole_momentum_cartesian(
                final_base + offset * final_tangent,
                gradient_base + offset * gradient_tangent,
                weights,
                polarization,
            )
            result: Array = jnp.stack(
                (
                    length.real,
                    length.imag,
                    momentum.real,
                    momentum.imag,
                )
            )
            return result

        zero: Array = jnp.asarray(0.0)
        derivative_forward: Array = jax.jacfwd(real_gauges)(zero)
        derivative_reverse: Array = jax.jacrev(real_gauges)(zero)
        quotient_step: float = 2.0**-16
        derivative_finite: Array = (
            real_gauges(zero + quotient_step)
            - real_gauges(zero - quotient_step)
        ) / (2.0 * quotient_step)
        chex.assert_trees_all_close(
            derivative_forward,
            derivative_reverse,
            rtol=1e-10,
            atol=1e-11,
        )
        chex.assert_trees_all_close(
            derivative_forward,
            derivative_finite,
            rtol=1e-7,
            atol=1e-10,
        )

        def reduced_gauges(
            selected_states: Float64[NDArray, "2 n_node"],
        ) -> Float64[NDArray, " 4"]:
            state_s: Float64[NDArray, " n_node"] = selected_states[0]
            state_p: Float64[NDArray, " n_node"] = selected_states[1]
            derivative_s: Float64[NDArray, " n_node"] = _derivative_sixth(
                state_s, spacing
            )
            safe_ratio: Float64[NDArray, " n_node"] = np.divide(
                state_s,
                radial_grid,
                out=np.zeros_like(state_s),
                where=radial_grid > 0.0,
            )
            length: complex = complex(
                np.sum(radial_weights * state_p * state_s * radial_grid)
                / math.sqrt(3.0)
            )
            momentum: complex = complex(
                -1j
                * np.sum(
                    radial_weights * state_p * (derivative_s - safe_ratio)
                )
                / math.sqrt(3.0)
            )
            result: Float64[NDArray, " 4"] = np.asarray(
                (length.real, length.imag, momentum.real, momentum.imag)
            )
            return result

        reduced_derivative: Float64[NDArray, " 4"] = (
            reduced_gauges(plus_states) - reduced_gauges(minus_states)
        ) / (2.0 * coefficient_step)
        chex.assert_trees_all_close(
            derivative_forward,
            reduced_derivative,
            rtol=1e-7,
            atol=1e-10,
        )
        base_gauges: Float64[NDArray, " 4"] = reduced_gauges(states)
        energy_gap: float = energies[1] - energies[0]
        energy_gap_derivative: float = (
            (plus_energies[1] - plus_energies[0])
            - (minus_energies[1] - minus_energies[0])
        ) / (2.0 * coefficient_step)
        commutator_derivative: float = (
            energy_gap_derivative * base_gauges[0]
            + energy_gap * reduced_derivative[0]
        )
        np.testing.assert_allclose(
            reduced_derivative[3],
            commutator_derivative,
            rtol=1e-7,
            atol=1e-10,
        )
