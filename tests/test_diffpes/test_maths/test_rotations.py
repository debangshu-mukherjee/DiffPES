"""Validate differentiable Cartesian rotations.

Extended Summary
----------------
The tests compare Rodrigues matrices with SciPy rotations and Wigner matrices
with independently exponentiated angular-momentum generators. They check the
canonical real-harmonic signs, rotation invariants, pole guards, JAX
transforms, and gradients.
"""

import chex
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Complex128, Float64
from numpy.typing import NDArray
from scipy import linalg, special
from scipy.spatial.transform import Rotation

from diffpes.maths.rotations import (
    bond_angles,
    real_harmonic_unitary,
    rodrigues_rotation,
    wigner_d,
    wigner_small_d,
)
from diffpes.maths.spherical_harmonics import real_spherical_harmonic
from tests._gradients import gradient_gate


def _angular_momentum_matrices(
    l: int,
) -> tuple[Complex128[NDArray, "dim dim"], Complex128[NDArray, "dim dim"]]:
    """Construct independent ascending-order Jz and Jy matrices."""
    magnetic_numbers: Float64[NDArray, " dim"] = np.arange(
        -l, l + 1, dtype=np.float64
    )
    size: int = 2 * l + 1
    raising: Complex128[NDArray, "dim dim"] = np.zeros(
        (size, size), dtype=np.complex128
    )
    m: int
    for m in range(-l, l):
        column: int = m + l
        row: int = column + 1
        raising[row, column] = np.sqrt(l * (l + 1) - m * (m + 1))
    lowering: Complex128[NDArray, "dim dim"] = raising.conj().T
    angular_y: Complex128[NDArray, "dim dim"] = (raising - lowering) / (2.0j)
    angular_z: Complex128[NDArray, "dim dim"] = np.diag(
        magnetic_numbers
    ).astype(np.complex128)
    return angular_z, angular_y


def _external_wigner_d(
    l: int,
    alpha: float,
    beta: float,
    gamma: float,
) -> Complex128[NDArray, "dim dim"]:
    """Construct an external Wigner matrix from independent generators."""
    angular_z: Complex128[NDArray, "dim dim"]
    angular_y: Complex128[NDArray, "dim dim"]
    angular_z, angular_y = _angular_momentum_matrices(l)
    matrix: Complex128[NDArray, "dim dim"] = (
        linalg.expm(-1j * alpha * angular_z)
        @ linalg.expm(-1j * beta * angular_y)
        @ linalg.expm(-1j * gamma * angular_z)
    )
    return matrix


class TestRodriguesRotation(chex.TestCase):
    """Validate :func:`~diffpes.maths.rodrigues_rotation`.

    The tests cover external values, rotation invariants, the zero-axis guard,
    JIT, vmap, and gradients in the axis and angle.

    :see: :func:`~diffpes.maths.rodrigues_rotation`
    """

    def test_matches_scipy_rotation(self) -> None:
        """Match SciPy rotation matrices to relative tolerance 1e-12.

        The test normalizes three external axes. SciPy then constructs each
        active matrix from the corresponding rotation vector.

        Notes
        -----
        Compare each float64 matrix with ``Rotation.from_rotvec``. Use
        ``rtol=1e-12`` and ``atol=1e-12``.
        """
        cases: tuple[tuple[list[float], float], ...] = (
            ([1.0, 0.0, 0.0], 0.37),
            ([0.0, -2.0, 0.0], -0.82),
            ([1.2, -0.7, 2.1], 1.13),
        )
        axis_values: list[float]
        angle_value: float
        for axis_values, angle_value in cases:
            with self.subTest(axis=axis_values, angle=angle_value):
                axis: Float64[Array, "3"] = jnp.asarray(
                    axis_values,
                    dtype=jnp.float64,
                )
                axis_numpy: Float64[NDArray, " 3"] = np.asarray(
                    axis_values,
                    dtype=np.float64,
                )
                rotation_vector: Float64[NDArray, " 3"] = (
                    angle_value * axis_numpy / np.linalg.norm(axis_numpy)
                )
                expected: Float64[Array, "3 3"] = jnp.asarray(
                    Rotation.from_rotvec(rotation_vector).as_matrix()
                )
                actual: Float64[Array, "3 3"] = rodrigues_rotation(
                    axis,
                    angle_value,
                )
                chex.assert_trees_all_close(
                    actual,
                    expected,
                    rtol=1e-12,
                    atol=1e-12,
                )

    def test_preserves_rotation_invariants(self) -> None:
        """Verify orthogonality, orientation, and the fixed-axis property.

        A proper rotation has ``R @ R.T = I`` and determinant one. It also
        leaves its normalized axis unchanged.

        Notes
        -----
        Use a generic axis and a 0.73 radian angle. Check each invariant at
        absolute tolerance ``1e-14``.
        """
        axis: Float64[Array, "3"] = jnp.array([1.4, -0.2, 0.9])
        normalized_axis: Float64[Array, "3"] = axis / jnp.linalg.norm(axis)
        rotation: Float64[Array, "3 3"] = rodrigues_rotation(axis, 0.73)
        chex.assert_trees_all_close(
            rotation @ rotation.T,
            jnp.eye(3),
            rtol=0.0,
            atol=1e-14,
        )
        chex.assert_trees_all_close(
            jnp.linalg.det(rotation),
            jnp.array(1.0),
            rtol=0.0,
            atol=1e-14,
        )
        chex.assert_trees_all_close(
            rotation @ normalized_axis,
            normalized_axis,
            rtol=0.0,
            atol=1e-14,
        )

    def test_zero_axis_returns_identity_with_finite_gradient(self) -> None:
        """Verify the identity value and finite gradient at a zero axis.

        The safe axis normalization selects a zero subgradient at the origin.
        Thus, the matrix remains finite for this boundary input.

        Notes
        -----
        JIT the matrix calculation. Differentiate its sum with respect to the
        three axis components.
        """
        axis: Float64[Array, "3"] = jnp.zeros(3)
        rotation: Float64[Array, "3 3"] = jax.jit(rodrigues_rotation)(
            axis,
            0.4,
        )
        gradient: Float64[Array, "3"] = jax.grad(
            lambda candidate: jnp.sum(rodrigues_rotation(candidate, 0.4))
        )(axis)
        chex.assert_trees_all_close(
            rotation,
            jnp.eye(3),
            rtol=0.0,
            atol=0.0,
        )
        chex.assert_trees_all_close(
            gradient,
            jnp.zeros(3),
            rtol=0.0,
            atol=0.0,
        )
        chex.assert_tree_all_finite((rotation, gradient))

    def test_jit_and_vmap_match_eager_results(self) -> None:
        """Verify consistent matrices under JIT and vmap.

        The test uses one generic axis and five traced angles. It compares the
        batched matrices with an eager stack.

        Notes
        -----
        JIT a vmapped Rodrigues call over the angle axis. Compare all entries
        at ``rtol=1e-14``.
        """
        axis: Float64[Array, "3"] = jnp.array([0.4, -0.8, 1.3])
        angles: Float64[Array, " 5"] = jnp.linspace(-0.7, 0.9, 5)
        vmapped: Float64[Array, "5 3 3"] = jax.jit(
            jax.vmap(lambda angle: rodrigues_rotation(axis, angle))
        )(angles)
        eager: Float64[Array, "5 3 3"] = jnp.stack(
            tuple(rodrigues_rotation(axis, angle) for angle in angles)
        )
        chex.assert_trees_all_close(
            vmapped,
            eager,
            rtol=1e-14,
            atol=1e-14,
        )

    def test_axis_and_angle_gradients_match_finite_differences(self) -> None:
        """Match axis and angle gradients with central finite differences.

        A generic matrix contraction depends on both traced inputs. The shared
        harness checks forward mode, reverse mode, and central differences.

        Notes
        -----
        Contract the matrix with asymmetric weights. Require nonzero gradient
        norms for the axis leaf and the angle leaf.
        """
        axis: Float64[Array, "3"] = jnp.array([0.7, -1.1, 0.3])
        angle: Float64[Array, ""] = jnp.array(0.61)
        weights: Float64[Array, "3 3"] = jnp.array(
            [
                [0.2, -0.7, 1.1],
                [1.4, 0.5, -0.3],
                [-0.9, 0.8, 0.6],
            ]
        )

        def loss(
            parameters: tuple[Float64[Array, "3"], Float64[Array, ""]],
        ) -> Float64[Array, ""]:
            candidate_axis: Float64[Array, "3"]
            candidate_angle: Float64[Array, ""]
            candidate_axis, candidate_angle = parameters
            value: Float64[Array, ""] = jnp.sum(
                rodrigues_rotation(candidate_axis, candidate_angle) * weights
            )
            return value

        gradient_gate(loss, (axis, angle), regime="smooth")


class TestWignerSmallD(chex.TestCase):
    """Validate :func:`~diffpes.maths.wigner_small_d`.

    The tests cover external matrix exponentials through l=4, a closed l=1
    form, static-domain validation, JIT, and angle gradients.

    :see: :func:`~diffpes.maths.wigner_small_d`
    """

    def test_matches_independent_generator_exponentials(self) -> None:
        """Match independently exponentiated y rotations through l=4.

        SciPy exponentiates ladder-operator matrices assembled without using
        the production factorial formula. The angle grid includes both poles
        and a generic interior value.

        Notes
        -----
        Compare every float64 entry for ``l=0..4`` at ``0``, ``0.37``,
        ``pi/2``, and ``pi`` with relative and absolute tolerance ``1e-13``.
        """
        beta_values: tuple[float, ...] = (0.0, 0.37, np.pi / 2.0, np.pi)
        l: int
        beta: float
        for l in range(5):
            for beta in beta_values:
                with self.subTest(l=l, beta=beta):
                    expected: Float64[Array, "m1 m2"] = jnp.asarray(
                        _external_wigner_d(l, 0.0, beta, 0.0).real
                    )
                    actual: Float64[Array, "m1 m2"] = wigner_small_d(
                        l,
                        beta,
                    )
                    chex.assert_trees_all_close(
                        actual,
                        expected,
                        rtol=1e-13,
                        atol=1e-13,
                    )

    def test_l_one_closed_form_under_jit(self) -> None:
        """Match the ascending-order l=1 closed form under JIT.

        The explicit matrix pins the row/column orientation and the signs of
        the off-diagonal sine terms independently of SciPy.

        Notes
        -----
        JIT a closure over static ``l=1`` at a generic traced angle. Compare
        the result with the analytic half-angle matrix at ``1e-14``.
        """
        beta: Float64[Array, ""] = jnp.array(0.61)
        cosine_half: Float64[Array, ""] = jnp.cos(beta / 2.0)
        sine_half: Float64[Array, ""] = jnp.sin(beta / 2.0)
        sine_beta: Float64[Array, ""] = jnp.sin(beta)
        expected: Float64[Array, "3 3"] = jnp.array(
            [
                [
                    cosine_half**2,
                    sine_beta / jnp.sqrt(2.0),
                    sine_half**2,
                ],
                [
                    -sine_beta / jnp.sqrt(2.0),
                    jnp.cos(beta),
                    sine_beta / jnp.sqrt(2.0),
                ],
                [
                    sine_half**2,
                    -sine_beta / jnp.sqrt(2.0),
                    cosine_half**2,
                ],
            ]
        )
        actual: Float64[Array, "3 3"] = jax.jit(
            lambda angle: wigner_small_d(1, angle)
        )(beta)
        chex.assert_trees_all_close(
            actual,
            expected,
            rtol=1e-14,
            atol=1e-14,
        )

    def test_angle_gradient_matches_finite_differences(self) -> None:
        """Match the l=4 angle gradient with finite differences.

        An asymmetric contraction samples every row and column without the
        cancellation that occurs when summing an orthogonal matrix uniformly.

        Notes
        -----
        Run the shared forward-mode, reverse-mode, and central-FD gate at one
        smooth interior angle and require a nonzero derivative.
        """
        beta: Float64[Array, ""] = jnp.array(0.73)
        weights: Float64[Array, "9 9"] = jnp.reshape(
            jnp.linspace(-0.8, 1.3, 81),
            (9, 9),
        )

        def loss(candidate: Float64[Array, ""]) -> Float64[Array, ""]:
            value: Float64[Array, ""] = jnp.sum(
                weights * wigner_small_d(4, candidate)
            )
            return value

        gradient_gate(loss, beta, regime="smooth")

    def test_rejects_unsupported_angular_momentum(self) -> None:
        """Reject angular momenta outside the implemented static range.

        The factorial implementation deliberately supports only the l values
        required by the package's s-through-g angular convention.

        Notes
        -----
        Call the function with ``l=-1`` and ``l=5`` and require ``ValueError``
        before any traced calculation begins.
        """
        invalid_l: int
        for invalid_l in (-1, 5):
            with self.subTest(l=invalid_l):
                with self.assertRaises(ValueError):
                    wigner_small_d(invalid_l, 0.2)


class TestWignerD(chex.TestCase):
    """Validate :func:`~diffpes.maths.wigner_d`.

    The tests cover independent z--y--z exponentials, unitarity, active group
    composition, and JIT transformation.

    :see: :func:`~diffpes.maths.wigner_d`
    """

    def test_matches_independent_exponentials_and_is_unitary(self) -> None:
        """Match generator exponentials and preserve norms through l=4.

        Independently built Jz and Jy matrices define the external active
        representation. Their exponentials exercise all three Euler phases.

        Notes
        -----
        JIT each static-l closure on a generic angle triple. Compare values and
        ``D.conj().T @ D`` at relative and absolute tolerance ``1e-13``.
        """
        angles: Float64[Array, "3"] = jnp.array([0.31, 0.82, -0.47])
        l: int
        for l in range(5):
            with self.subTest(l=l):
                expected: Complex128[Array, "m1 m2"] = jnp.asarray(
                    _external_wigner_d(
                        l,
                        float(angles[0]),
                        float(angles[1]),
                        float(angles[2]),
                    )
                )
                actual: Complex128[Array, "m1 m2"] = jax.jit(
                    lambda euler: wigner_d(
                        l,
                        euler[0],
                        euler[1],
                        euler[2],
                    )
                )(angles)
                identity: Complex128[Array, "m1 m2"] = jnp.eye(
                    2 * l + 1,
                    dtype=jnp.complex128,
                )
                chex.assert_trees_all_close(
                    actual,
                    expected,
                    rtol=1e-13,
                    atol=1e-13,
                )
                chex.assert_trees_all_close(
                    actual.conj().T @ actual,
                    identity,
                    rtol=1e-13,
                    atol=1e-13,
                )

    def test_obeys_active_rotation_composition(self) -> None:
        """Represent products of two generic active rotations.

        SciPy composes the corresponding Cartesian z--y--z rotations and
        extracts one nonsingular Euler triple for the product. The Wigner
        representation must turn that product into matrix multiplication.

        Notes
        -----
        Use two generic rotations away from Euler singularities and compare
        ``D(R1) @ D(R2)`` with ``D(R1 @ R2)`` for ``l=1..4`` at ``1e-13``.
        """
        first_angles: Float64[NDArray, " 3"] = np.array([0.27, 0.64, -0.19])
        second_angles: Float64[NDArray, " 3"] = np.array([-0.38, 0.91, 0.44])
        first_rotation: Rotation = Rotation.from_euler("ZYZ", first_angles)
        second_rotation: Rotation = Rotation.from_euler(
            "ZYZ",
            second_angles,
        )
        product_angles: Float64[NDArray, " 3"] = (
            first_rotation * second_rotation
        ).as_euler("ZYZ")
        l: int
        for l in range(1, 5):
            with self.subTest(l=l):
                first_matrix: Complex128[Array, "m1 m2"] = wigner_d(
                    l,
                    first_angles[0],
                    first_angles[1],
                    first_angles[2],
                )
                second_matrix: Complex128[Array, "m1 m2"] = wigner_d(
                    l,
                    second_angles[0],
                    second_angles[1],
                    second_angles[2],
                )
                expected: Complex128[Array, "m1 m2"] = wigner_d(
                    l,
                    product_angles[0],
                    product_angles[1],
                    product_angles[2],
                )
                actual: Complex128[Array, "m1 m2"] = (
                    first_matrix @ second_matrix
                )
                chex.assert_trees_all_close(
                    actual,
                    expected,
                    rtol=1e-13,
                    atol=1e-13,
                )


class TestRealHarmonicUnitary(chex.TestCase):
    """Validate :func:`~diffpes.maths.real_harmonic_unitary`.

    The tests cover the canon's p-orbital signs, unitary invariants through
    l=4, complex-harmonic reconstruction, and an Lz eigenstate probe.

    :see: :func:`~diffpes.maths.real_harmonic_unitary`
    """

    def test_p_shell_has_canonical_px_and_py_signs(self) -> None:
        """Recover positive p_x and p_y from complex harmonics.

        The explicit l=1 rows pin the Chinook/VASP real-orbital convention:
        ascending real order is ``(p_y, p_z, p_x)`` and ascending complex
        order is ``(-1, 0, +1)``.

        Notes
        -----
        Compare the returned matrix with the closed Condon--Shortley
        combinations from canon equations 5b and 5c at zero tolerance.
        """
        inverse_sqrt_two: float = 1.0 / np.sqrt(2.0)
        expected: Complex128[Array, "3 3"] = jnp.array(
            [
                [
                    1j * inverse_sqrt_two,
                    0.0,
                    1j * inverse_sqrt_two,
                ],
                [0.0, 1.0, 0.0],
                [
                    inverse_sqrt_two,
                    0.0,
                    -inverse_sqrt_two,
                ],
            ],
            dtype=jnp.complex128,
        )
        actual: Complex128[Array, "3 3"] = real_harmonic_unitary(1)
        chex.assert_trees_all_close(
            actual,
            expected,
            rtol=0.0,
            atol=0.0,
        )

        theta: float = 0.83
        phi: float = -0.41
        complex_values: Complex128[NDArray, " 3"] = special.sph_harm_y(
            1,
            np.arange(-1, 2),
            theta,
            phi,
        )
        real_values: Complex128[Array, "3"] = actual @ jnp.asarray(
            complex_values
        )
        normalization: float = np.sqrt(3.0 / (4.0 * np.pi))
        px: float = normalization * np.sin(theta) * np.cos(phi)
        py: float = normalization * np.sin(theta) * np.sin(phi)
        chex.assert_trees_all_close(
            real_values,
            jnp.array([py, normalization * np.cos(theta), px]),
            rtol=1e-14,
            atol=1e-14,
        )

    def test_is_unitary_and_matches_real_harmonics(self) -> None:
        """Preserve norms and reconstruct package real harmonics through l=4.

        SciPy supplies independently normalized complex Condon--Shortley
        harmonics. Applying the unitary must reproduce the package's canonical
        real values after the positive-m sign correction owned by that module.

        Notes
        -----
        Check ``U @ U.conj().T`` and reconstructed values for one generic
        angular point at relative and absolute tolerance ``1e-13``.
        """
        theta: float = 1.07
        phi: float = -0.63
        l: int
        for l in range(5):
            with self.subTest(l=l):
                unitary: Complex128[Array, "m1 m2"] = real_harmonic_unitary(l)
                identity: Complex128[Array, "m1 m2"] = jnp.eye(
                    2 * l + 1,
                    dtype=jnp.complex128,
                )
                complex_values: Complex128[NDArray, " dim"] = (
                    special.sph_harm_y(
                        l,
                        np.arange(-l, l + 1),
                        theta,
                        phi,
                    )
                )
                reconstructed: Complex128[Array, " m"] = unitary @ jnp.asarray(
                    complex_values
                )
                expected: Float64[Array, " m"] = jnp.stack(
                    tuple(
                        real_spherical_harmonic(
                            l,
                            order,
                            jnp.asarray(theta),
                            jnp.asarray(phi),
                        )
                        for order in range(-l, l + 1)
                    )
                )
                chex.assert_trees_all_close(
                    unitary @ unitary.conj().T,
                    identity,
                    rtol=1e-13,
                    atol=1e-13,
                )
                chex.assert_trees_all_close(
                    reconstructed,
                    expected,
                    rtol=1e-13,
                    atol=1e-13,
                )

    def test_px_plus_or_minus_i_py_are_lz_eigenstates(self) -> None:
        """Give p_x plus or minus i p_y the expected Lz eigenvalues.

        The operator transform follows the canon coefficient map
        ``L_real = U.conj() @ L_complex @ U.T``. Its circular p states must
        carry magnetic quantum numbers plus and minus one.

        Notes
        -----
        Build both normalized states in real order ``(p_y, p_z, p_x)`` and
        compare their expectation values with ``+1`` and ``-1`` at ``1e-14``.
        """
        unitary: Complex128[Array, "3 3"] = real_harmonic_unitary(1)
        lz_complex: Complex128[Array, "3 3"] = jnp.diag(
            jnp.array([-1.0, 0.0, 1.0], dtype=jnp.complex128)
        )
        lz_real: Complex128[Array, "3 3"] = (
            unitary.conj() @ lz_complex @ unitary.T
        )
        inverse_sqrt_two: float = 1.0 / np.sqrt(2.0)
        plus_state: Complex128[Array, "3"] = (
            jnp.array(
                [1j, 0.0, 1.0],
                dtype=jnp.complex128,
            )
            * inverse_sqrt_two
        )
        minus_state: Complex128[Array, "3"] = (
            jnp.array(
                [-1j, 0.0, 1.0],
                dtype=jnp.complex128,
            )
            * inverse_sqrt_two
        )
        plus_expectation: Complex128[Array, ""] = (
            plus_state.conj() @ lz_real @ plus_state
        )
        minus_expectation: Complex128[Array, ""] = (
            minus_state.conj() @ lz_real @ minus_state
        )
        chex.assert_trees_all_close(
            plus_expectation,
            jnp.array(1.0 + 0.0j),
            rtol=0.0,
            atol=1e-14,
        )
        chex.assert_trees_all_close(
            minus_expectation,
            jnp.array(-1.0 + 0.0j),
            rtol=0.0,
            atol=1e-14,
        )

    def test_rejects_unsupported_angular_momentum(self) -> None:
        """Reject unitary requests outside the shared l range.

        The package intentionally freezes real-harmonic order only for angular
        momenta from zero through four.

        Notes
        -----
        Call both neighboring invalid integers and require ``ValueError``.
        """
        invalid_l: int
        for invalid_l in (-1, 5):
            with self.subTest(l=invalid_l):
                with self.assertRaises(ValueError):
                    real_harmonic_unitary(invalid_l)


class TestBondAngles(chex.TestCase):
    """Validate :func:`~diffpes.maths.bond_angles`.

    The tests cover generic angle reconstruction, scale invariance, exact
    poles, the zero-vector guard, JIT, and smooth gradients.

    :see: :func:`~diffpes.maths.bond_angles`
    """

    def test_reconstructs_generic_direction_under_jit(self) -> None:
        """Verify reconstruction of a normalized bond from returned angles.

        Polar and azimuthal angles must retain all directional information and
        remain invariant under positive rescaling of the input vector.

        Notes
        -----
        JIT the conversion for a generic vector and its sevenfold rescaling.
        Reconstruct Cartesian unit vectors and compare at ``1e-14``.
        """
        bond: Float64[Array, "3"] = jnp.array([0.7, -1.1, 0.4])
        angles: Float64[Array, "2"] = jax.jit(
            lambda vector: jnp.stack(bond_angles(vector))
        )(bond)
        scaled_angles: Float64[Array, "2"] = jax.jit(
            lambda vector: jnp.stack(bond_angles(vector))
        )(7.0 * bond)
        beta: Float64[Array, ""] = angles[0]
        alpha: Float64[Array, ""] = angles[1]
        reconstructed: Float64[Array, "3"] = jnp.array(
            [
                jnp.sin(beta) * jnp.cos(alpha),
                jnp.sin(beta) * jnp.sin(alpha),
                jnp.cos(beta),
            ]
        )
        expected: Float64[Array, "3"] = bond / jnp.linalg.norm(bond)
        chex.assert_trees_all_close(
            reconstructed,
            expected,
            rtol=1e-14,
            atol=1e-14,
        )
        chex.assert_trees_all_close(
            scaled_angles,
            angles,
            rtol=1e-14,
            atol=1e-14,
        )

    def test_poles_and_zero_vector_are_finite(self) -> None:
        """Select stable angles and Jacobians at both poles and the origin.

        The positive pole and zero-vector guard map to ``(0, 0)``. The
        negative pole maps to ``(pi, 0)``. Azimuth is pure gauge in all three
        cases.

        Notes
        -----
        Evaluate under JIT and differentiate the two-angle vector with respect
        to each bond component. Require exact values and finite Jacobians.
        """
        cases: tuple[tuple[list[float], list[float]], ...] = (
            ([0.0, 0.0, 2.0], [0.0, 0.0]),
            ([0.0, 0.0, -3.0], [np.pi, 0.0]),
            ([0.0, 0.0, 0.0], [0.0, 0.0]),
        )
        bond_values: list[float]
        expected_values: list[float]
        for bond_values, expected_values in cases:
            with self.subTest(bond=bond_values):
                bond: Float64[Array, "3"] = jnp.asarray(bond_values)
                angles: Float64[Array, "2"] = jax.jit(
                    lambda vector: jnp.stack(bond_angles(vector))
                )(bond)
                jacobian: Float64[Array, "2 3"] = jax.jacrev(
                    lambda vector: jnp.stack(bond_angles(vector))
                )(bond)
                chex.assert_trees_all_close(
                    angles,
                    jnp.asarray(expected_values),
                    rtol=0.0,
                    atol=0.0,
                )
                chex.assert_tree_all_finite((angles, jacobian))

    def test_generic_gradient_matches_finite_differences(self) -> None:
        """Match bond-coordinate gradients away from Euler singularities.

        A weighted angle loss depends on all three Cartesian components of a
        generic bond. This probes normalization, arccos, and arctan2 together.

        Notes
        -----
        Run the shared forward-mode, reverse-mode, and central-FD gate in the
        smooth regime. Require a nonzero gradient for the bond leaf.
        """
        bond: Float64[Array, "3"] = jnp.array([0.8, -0.5, 1.2])

        def loss(candidate: Float64[Array, "3"]) -> Float64[Array, ""]:
            beta: Float64[Array, ""]
            alpha: Float64[Array, ""]
            beta, alpha = bond_angles(candidate)
            value: Float64[Array, ""] = 0.7 * beta - 0.3 * alpha
            return value

        gradient_gate(loss, bond, regime="smooth")
