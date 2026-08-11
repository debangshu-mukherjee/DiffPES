"""Validate named gradient-safe elementary operations.

Extended Summary
----------------
The tests compare all seven safe-math helpers with NumPy away from their
guards. They compare both autodiff modes with central finite differences.
They also verify each documented guard value and subgradient. Property tests
cover broadcasting and sign symmetry on generated finite inputs.
"""

import chex
import jax
import jax.numpy as jnp
import numpy as np
from beartype import beartype
from beartype.typing import List, Tuple
from hypothesis import given, settings
from hypothesis import strategies as st
from jaxtyping import Array, Float64, jaxtyped

from diffpes.maths import (
    safe_arccos,
    safe_arctan2,
    safe_divide,
    safe_log,
    safe_norm,
    safe_power,
    safe_sqrt,
)
from tests._gradients import assert_grad_matches_fd


class TestSafeDivide:
    """Validate :func:`~diffpes.maths.safe_divide`.

    Covers ordinary broadcast division, both autodiff modes, and the finite
    fallback with zero operand subgradients at a zero denominator.

    :see: :func:`~diffpes.maths.safe_divide`
    """

    def test_values_gradients_and_guard(self) -> None:
        """Match division and pin zero-denominator guard behavior.

        Extended Summary
        ----------------
        Ordinary values must match NumPy to ``rtol=1e-15`` and derivatives
        must pass the smooth program-wide gradient harness. At a zero divisor,
        the function returns the configured fallback. Both operand gradients
        equal zero and contain no NaNs.

        Notes
        -----
        The test uses a length-three numerator and a nonzero divisor.
        It differentiates a scalar guarded call for both operands with
        JIT-backed autodiff.
        """
        numerator: Float64[Array, " 3"] = jnp.array([1.5, -2.0, 4.0])
        denominator: Float64[Array, " 3"] = jnp.array([0.5, 4.0, -2.0])
        values: Float64[Array, " 3"] = safe_divide(numerator, denominator)
        expected: Float64[Array, " 3"] = jnp.asarray(
            np.divide(np.asarray(numerator), np.asarray(denominator))
        )
        chex.assert_trees_all_close(values, expected, rtol=1e-15, atol=1e-15)
        assert_grad_matches_fd(
            lambda value: jnp.sum(
                safe_divide(jnp.asarray(value), denominator)
            ),
            numerator,
        )

        guarded: Float64[Array, ""] = jax.jit(safe_divide)(
            jnp.array(3.0), jnp.array(0.0), -7.0
        )
        gradients: Tuple[Float64[Array, ""], Float64[Array, ""]] = jax.grad(
            lambda n, d: safe_divide(n, d, -7.0), argnums=(0, 1)
        )(jnp.array(3.0), jnp.array(0.0))
        chex.assert_trees_all_close(guarded, -7.0, rtol=0.0, atol=0.0)
        chex.assert_trees_all_close(gradients, (0.0, 0.0), rtol=0.0, atol=0.0)
        chex.assert_tree_all_finite((guarded, gradients))

    @given(
        values=st.lists(
            st.floats(
                min_value=-100.0,
                max_value=100.0,
                allow_nan=False,
                allow_infinity=False,
            ),
            min_size=3,
            max_size=3,
        ),
        denominator=st.floats(
            min_value=0.25,
            max_value=10.0,
            allow_nan=False,
            allow_infinity=False,
        ),
    )
    @settings(max_examples=20, deadline=None)
    def test_broadcasting_and_sign_symmetry(
        self, values: List[float], denominator: float
    ) -> None:
        """Preserve broadcasting and odd symmetry in the numerator.

        Extended Summary
        ----------------
        For finite vectors and a positive scalar divisor, ``safe_divide``
        must broadcast the scalar. It must also preserve odd symmetry within
        floating-point tolerance.

        Notes
        -----
        Hypothesis generates twenty three-component vectors and divisors away
        from zero; Chex checks shape and sign symmetry at ``rtol=1e-15``.
        """
        numerator: Float64[Array, " 3"] = jnp.asarray(
            values, dtype=jnp.float64
        )
        divisor: Float64[Array, ""] = jnp.asarray(
            denominator, dtype=jnp.float64
        )
        positive: Float64[Array, " 3"] = safe_divide(numerator, divisor)
        negative: Float64[Array, " 3"] = safe_divide(-numerator, divisor)
        chex.assert_shape(positive, (3,))
        chex.assert_trees_all_close(
            negative, -positive, rtol=1e-15, atol=1e-15
        )


class TestSafeSqrt:
    """Validate :func:`~diffpes.maths.safe_sqrt`.

    Covers NumPy agreement, both autodiff modes, and the selected zero value
    and subgradient on non-positive inputs.

    :see: :func:`~diffpes.maths.safe_sqrt`
    """

    def test_values_gradients_and_guard(self) -> None:
        """Match square roots and pin the non-positive guard.

        Extended Summary
        ----------------
        Positive values must match NumPy to ``rtol=1e-15`` and pass the smooth
        gradient harness. Zero and negative inputs must both return zero with
        exact zero gradients and finite traced values.

        Notes
        -----
        The test evaluates three positive values away from the branch boundary
        and a compiled two-value guard probe before differentiating each guard.
        """
        x: Float64[Array, " 3"] = jnp.array([0.25, 2.0, 9.0])
        values: Float64[Array, " 3"] = safe_sqrt(x)
        expected: Float64[Array, " 3"] = jnp.asarray(np.sqrt(np.asarray(x)))
        chex.assert_trees_all_close(values, expected, rtol=1e-15, atol=1e-15)
        assert_grad_matches_fd(
            lambda value: jnp.sum(safe_sqrt(jnp.asarray(value))), x
        )

        guarded_x: Float64[Array, " 2"] = jnp.array([0.0, -4.0])
        guarded: Float64[Array, " 2"] = jax.jit(safe_sqrt)(guarded_x)
        gradients: Float64[Array, " 2"] = jax.grad(
            lambda value: jnp.sum(safe_sqrt(value))
        )(guarded_x)
        chex.assert_trees_all_close(guarded, jnp.zeros(2), rtol=0.0, atol=0.0)
        chex.assert_trees_all_close(
            gradients, jnp.zeros(2), rtol=0.0, atol=0.0
        )
        chex.assert_tree_all_finite((guarded, gradients))


class TestSafeNorm:
    """Validate :func:`~diffpes.maths.safe_norm`.

    Covers Euclidean values, axis handling, sign symmetry, both derivative
    modes, and the zero-vector subgradient convention.

    :see: :func:`~diffpes.maths.safe_norm`
    """

    def test_values_gradients_and_guard(self) -> None:
        """Match Euclidean norms and pin the zero-vector gradient.

        Extended Summary
        ----------------
        Batched vector norms must match NumPy to ``rtol=1e-15`` with retained
        dimensions and pass the gradient harness. An all-zero vector must
        return zero and have an exactly zero, finite gradient.

        Notes
        -----
        The test uses two generic three-vectors for the ordinary path and an
        independent JIT and reverse-mode probe at the origin.
        """
        x: Float64[Array, "2 3"] = jnp.array(
            [[3.0, 4.0, 1.0], [-2.0, 5.0, 7.0]]
        )
        values: Float64[Array, "2 1"] = safe_norm(x, axis=-1, keepdims=True)
        expected: Float64[Array, "2 1"] = jnp.asarray(
            np.linalg.norm(np.asarray(x), axis=-1, keepdims=True)
        )
        chex.assert_trees_all_close(values, expected, rtol=1e-15, atol=1e-15)
        assert_grad_matches_fd(
            lambda value: jnp.sum(safe_norm(jnp.asarray(value))), x
        )

        zero: Float64[Array, " 3"] = jnp.zeros(3)
        guarded: Float64[Array, ""] = jax.jit(safe_norm)(zero)
        gradient: Float64[Array, " 3"] = jax.grad(safe_norm)(zero)
        chex.assert_trees_all_close(guarded, 0.0, rtol=0.0, atol=0.0)
        chex.assert_trees_all_close(gradient, zero, rtol=0.0, atol=0.0)
        chex.assert_tree_all_finite((guarded, gradient))

    @given(
        values=st.lists(
            st.floats(
                min_value=-100.0,
                max_value=100.0,
                allow_nan=False,
                allow_infinity=False,
            ),
            min_size=3,
            max_size=3,
        )
    )
    @settings(max_examples=20, deadline=None)
    def test_sign_symmetry(self, values: List[float]) -> None:
        """Preserve norm symmetry under vector sign reversal.

        Extended Summary
        ----------------
        The safe guard must not change the Euclidean invariant
        ``norm(-x) = norm(x)`` for generated finite three-vectors.

        Notes
        -----
        Hypothesis generates twenty vectors across both signs and Chex checks
        the two scalar norms at ``rtol=1e-15``.
        """
        x: Float64[Array, " 3"] = jnp.asarray(values, dtype=jnp.float64)
        positive: Float64[Array, ""] = safe_norm(x)
        negative: Float64[Array, ""] = safe_norm(-x)
        chex.assert_trees_all_close(negative, positive, rtol=1e-15, atol=1e-15)


class TestSafeArccos:
    """Validate :func:`~diffpes.maths.safe_arccos`.

    Covers NumPy agreement, both derivative modes, saturation outside the
    cosine domain, and zero subgradients at both endpoints.

    :see: :func:`~diffpes.maths.safe_arccos`
    """

    def test_values_gradients_and_guard(self) -> None:
        """Match arccos and pin saturated endpoint gradients.

        Extended Summary
        ----------------
        Interior values must match NumPy to ``rtol=1e-15`` and pass the smooth
        gradient harness. Inputs at and beyond ``-1`` and ``1`` must saturate
        to ``pi`` and zero with exact zero, finite gradients.

        Notes
        -----
        The test uses three generic interior cosines and a four-value JIT guard
        probe spanning both endpoints and both out-of-domain sides.
        """
        x: Float64[Array, " 3"] = jnp.array([-0.6, 0.2, 0.7])
        values: Float64[Array, " 3"] = safe_arccos(x)
        expected: Float64[Array, " 3"] = jnp.asarray(np.arccos(np.asarray(x)))
        chex.assert_trees_all_close(values, expected, rtol=1e-15, atol=1e-15)
        assert_grad_matches_fd(
            lambda value: jnp.sum(safe_arccos(jnp.asarray(value))), x
        )

        guarded_x: Float64[Array, " 4"] = jnp.array([-2.0, -1.0, 1.0, 2.0])
        guarded: Float64[Array, " 4"] = jax.jit(safe_arccos)(guarded_x)
        gradient: Float64[Array, " 4"] = jax.grad(
            lambda value: jnp.sum(safe_arccos(value))
        )(guarded_x)
        expected_guard: Float64[Array, " 4"] = jnp.array(
            [jnp.pi, jnp.pi, 0.0, 0.0]
        )
        chex.assert_trees_all_close(
            guarded, expected_guard, rtol=0.0, atol=0.0
        )
        chex.assert_trees_all_close(gradient, jnp.zeros(4), rtol=0.0, atol=0.0)
        chex.assert_tree_all_finite((guarded, gradient))


class TestSafeArctan2:
    """Validate :func:`~diffpes.maths.safe_arctan2`.

    Covers NumPy agreement in all quadrants, broadcasting, both derivative
    modes, and the zero value and coordinate gradients at the origin.

    :see: :func:`~diffpes.maths.safe_arctan2`
    """

    def test_values_gradients_and_guard(self) -> None:
        """Match arctan2 and pin the origin convention.

        Extended Summary
        ----------------
        Away from the origin, values must match NumPy to ``rtol=1e-15`` and
        derivatives must pass the gradient harness. At ``(0, 0)``, the value
        and both coordinate gradients must be exactly zero and finite.

        Notes
        -----
        The test uses points in three quadrants for ordinary behavior, then
        runs the scalar origin through JIT and differentiates both arguments.
        """
        coordinates: Float64[Array, "3 2"] = jnp.array(
            [[0.5, 1.0], [2.0, -3.0], [-4.0, -2.0]]
        )

        @jaxtyped(typechecker=beartype)
        def _summed_angles(
            value: Float64[Array, "3 2"],
        ) -> Float64[Array, ""]:
            """PRIVATE: Sum safe polar angles for three coordinates.

            Parameters
            ----------
            value : Float64[Array, "3 2"]
                Three ordered ``(y, x)`` coordinate pairs.

            Returns
            -------
            result : Float64[Array, ""]
                Sum of the three guarded polar angles in radians.

            Notes
            -----
            Converts the candidate to a JAX array before evaluating each row.
            """
            value_array: Float64[Array, "3 2"] = jnp.asarray(value)
            result: Float64[Array, ""] = jnp.sum(
                safe_arctan2(value_array[:, 0], value_array[:, 1])
            )
            return result

        values: Float64[Array, " 3"] = safe_arctan2(
            coordinates[:, 0], coordinates[:, 1]
        )
        expected: Float64[Array, " 3"] = jnp.asarray(
            np.arctan2(
                np.asarray(coordinates[:, 0]), np.asarray(coordinates[:, 1])
            )
        )
        chex.assert_trees_all_close(values, expected, rtol=1e-15, atol=1e-15)
        assert_grad_matches_fd(_summed_angles, coordinates)

        guarded: Float64[Array, ""] = jax.jit(safe_arctan2)(
            jnp.array(0.0), jnp.array(0.0)
        )
        gradients: Tuple[Float64[Array, ""], Float64[Array, ""]] = jax.grad(
            safe_arctan2, argnums=(0, 1)
        )(jnp.array(0.0), jnp.array(0.0))
        chex.assert_trees_all_close(guarded, 0.0, rtol=0.0, atol=0.0)
        chex.assert_trees_all_close(gradients, (0.0, 0.0), rtol=0.0, atol=0.0)
        chex.assert_tree_all_finite((guarded, gradients))


class TestSafeLog:
    """Validate :func:`~diffpes.maths.safe_log`.

    Covers NumPy agreement, both derivative modes, and the finite floor value
    with a zero input subgradient at and below the floor.

    :see: :func:`~diffpes.maths.safe_log`
    """

    def test_values_gradients_and_guard(self) -> None:
        """Match logarithms and pin floor-clamped gradients.

        Extended Summary
        ----------------
        Positive values above the floor must match NumPy to ``rtol=1e-15``
        and pass the gradient harness. Inputs at and below a ``1e-3`` floor
        must return ``log(1e-3)`` with exact zero, finite gradients.

        Notes
        -----
        The test uses three ordinary positive values and a three-value JIT
        probe containing the floor, zero, and a negative input.
        """
        x: Float64[Array, " 3"] = jnp.array([0.25, 2.0, 10.0])
        values: Float64[Array, " 3"] = safe_log(x)
        expected: Float64[Array, " 3"] = jnp.asarray(np.log(np.asarray(x)))
        chex.assert_trees_all_close(values, expected, rtol=1e-15, atol=1e-15)
        assert_grad_matches_fd(
            lambda value: jnp.sum(safe_log(jnp.asarray(value))), x
        )

        guarded_x: Float64[Array, " 3"] = jnp.array([1e-3, 0.0, -2.0])
        guarded: Float64[Array, " 3"] = jax.jit(
            lambda value: safe_log(value, 1e-3)
        )(guarded_x)
        gradient: Float64[Array, " 3"] = jax.grad(
            lambda value: jnp.sum(safe_log(value, 1e-3))
        )(guarded_x)
        expected_guard: Float64[Array, " 3"] = jnp.full(3, jnp.log(1e-3))
        chex.assert_trees_all_close(
            guarded, expected_guard, rtol=0.0, atol=0.0
        )
        chex.assert_trees_all_close(gradient, jnp.zeros(3), rtol=0.0, atol=0.0)
        chex.assert_tree_all_finite((guarded, gradient))


class TestSafePower:
    """Validate :func:`~diffpes.maths.safe_power`.

    Covers fractional powers, both derivative modes, and the zero value and
    base/exponent subgradients on non-positive bases.

    :see: :func:`~diffpes.maths.safe_power`
    """

    def test_values_gradients_and_guard(self) -> None:
        """Match fractional powers and pin non-positive gradients.

        Extended Summary
        ----------------
        Positive fractional powers must match NumPy to ``rtol=1e-15`` and
        pass the gradient harness. Zero and negative bases must return zero
        with exact zero gradients for both the base and exponent.

        Notes
        -----
        The test uses exponent ``1.7`` on three positive bases, then evaluates
        and differentiates a guarded two-base sum with JIT-safe primitives.
        """
        exponent: Float64[Array, ""] = jnp.array(1.7)
        x: Float64[Array, " 3"] = jnp.array([0.25, 2.0, 7.0])
        values: Float64[Array, " 3"] = safe_power(x, exponent)
        expected: Float64[Array, " 3"] = jnp.asarray(
            np.power(np.asarray(x), float(exponent))
        )
        chex.assert_trees_all_close(values, expected, rtol=1e-15, atol=1e-15)
        assert_grad_matches_fd(
            lambda value: jnp.sum(safe_power(jnp.asarray(value), exponent)),
            x,
        )

        guarded_x: Float64[Array, " 2"] = jnp.array([0.0, -3.0])
        guarded: Float64[Array, " 2"] = jax.jit(safe_power)(
            guarded_x, exponent
        )
        gradients: Tuple[Float64[Array, " 2"], Float64[Array, ""]] = jax.grad(
            lambda bases, power: jnp.sum(safe_power(bases, power)),
            argnums=(0, 1),
        )(guarded_x, exponent)
        chex.assert_trees_all_close(guarded, jnp.zeros(2), rtol=0.0, atol=0.0)
        chex.assert_trees_all_close(
            gradients, (jnp.zeros(2), jnp.array(0.0)), rtol=0.0, atol=0.0
        )
        chex.assert_tree_all_finite((guarded, gradients))
