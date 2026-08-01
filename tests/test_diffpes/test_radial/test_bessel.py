"""Validate spherical Bessel functions.

Extended Summary
----------------
The tests compare ``spherical_bessel_jl`` for orders 0, 1, and 2 with
closed-form expressions. They verify the singular ``k=0`` limit and the
autodiff gradient of ``j_0``. ``chex.variants`` runs each closed-form test
with and without JIT compilation.

"""

import chex
import jax
import jax.numpy as jnp
import numpy as np
from beartype.typing import Any, Callable
from jax.test_util import check_grads
from jaxtyping import Array, Float
from numpy.typing import NDArray
from scipy.special import spherical_jn

from diffpes.radial import spherical_bessel_jl
from diffpes.radial.bessel import spherical_bessel_jl_derivative


class TestSphericalBesselJl(chex.TestCase):
    """Validate low-order spherical Bessel j_l(x) behavior and derivatives.

    The tests compare the three lowest spherical Bessel functions with their
    closed-form expressions. They verify the ``k=0`` boundary condition and
    compare the ``j_0`` autodiff gradient with its analytical derivative.
    Each variant runs with and without JAX JIT.

    :see: :func:`~diffpes.radial.spherical_bessel_jl`
    """

    @chex.variants(with_jit=True, without_jit=True)
    def test_j0_and_j1_match_closed_form(self) -> None:
        """Verify j_0 and j_1 match their closed-form expressions.

        The test uses test points x = [0.2, 0.7, 1.5] (avoiding x=0 singularity).
        j_0(x) = sin(x)/x and j_1(x) = sin(x)/x^2 - cos(x)/x are the
        standard analytical forms.  Asserts element-wise agreement to
        within 1e-10, run under both JIT and eager modes via
        ``chex.variants``.

        Notes
        -----
        The test builds the inputs in the test body and checks the stated property with the documented numerical or structural assertions."""
        x: Array
        j0_fn: Callable[..., Any]
        j1_fn: Callable[..., Any]
        expected_j0: Array
        expected_j1: Array

        x = jnp.array([0.2, 0.7, 1.5], dtype=jnp.float64)
        j0_fn = self.variant(lambda values: spherical_bessel_jl(0, values))
        j1_fn = self.variant(lambda values: spherical_bessel_jl(1, values))

        expected_j0 = jnp.sin(x) / x
        expected_j1 = jnp.sin(x) / (x * x) - jnp.cos(x) / x
        chex.assert_trees_all_close(j0_fn(x), expected_j0, atol=1.0e-10)
        chex.assert_trees_all_close(j1_fn(x), expected_j1, atol=1.0e-10)

    @chex.variants(with_jit=True, without_jit=True)
    def test_j2_matches_closed_form(self) -> None:
        """Verify j_2 matches its closed-form expression.

        The test uses test points x = [0.4, 1.1, 2.4].  The analytical form is
        j_2(x) = (3/x^3 - 1/x)*sin(x) - (3/x^2)*cos(x).  Asserts
        element-wise agreement to within 1e-10, confirming the recursion
        or series implementation is accurate for the l=2 case.

        Notes
        -----
        The test builds the inputs in the test body and checks the stated property with the documented numerical or structural assertions."""
        x: Array
        fn: Callable[..., Any]
        expected: Array

        x = jnp.array([0.4, 1.1, 2.4], dtype=jnp.float64)
        fn = self.variant(lambda values: spherical_bessel_jl(2, values))
        expected = ((3.0 / (x**3)) - (1.0 / x)) * jnp.sin(x) - (
            3.0 / (x * x)
        ) * jnp.cos(x)
        chex.assert_trees_all_close(fn(x), expected, atol=1.0e-10)

    @chex.variants(with_jit=True, without_jit=True)
    def test_zero_argument_limits(self) -> None:
        """Verify the x=0 boundary conditions: j_0(0)=1, j_l(0)=0 for l>0.

        The test evaluates j_0, j_1, and j_3 at x=0.0.  The mathematical limits
        are j_0(0) = 1 and j_l(0) = 0 for all l >= 1.  This is a critical
        case because the direct ``sin(x)/x`` formula has no value at zero.
        The implementation handles this removable singularity. The test asserts
        agreement to within 1e-12.  The l=3 case also confirms higher-
        order terms beyond the three tested in the closed-form tests.

        Notes
        -----
        The test builds the inputs in the test body and checks the stated property with the documented numerical or structural assertions."""
        zero: Array
        j0_fn: Callable[..., Any]
        j1_fn: Callable[..., Any]
        j3_fn: Callable[..., Any]

        zero = jnp.array([0.0], dtype=jnp.float64)
        j0_fn = self.variant(lambda values: spherical_bessel_jl(0, values))
        j1_fn = self.variant(lambda values: spherical_bessel_jl(1, values))
        j3_fn = self.variant(lambda values: spherical_bessel_jl(3, values))
        chex.assert_trees_all_close(
            j0_fn(zero), jnp.array([1.0]), atol=1.0e-12
        )
        chex.assert_trees_all_close(
            j1_fn(zero), jnp.array([0.0]), atol=1.0e-12
        )
        chex.assert_trees_all_close(
            j3_fn(zero), jnp.array([0.0]), atol=1.0e-12
        )

    def test_j0_gradient_matches_analytic_derivative(self) -> None:
        """Verify autodiff gradient of j_0 matches the analytical derivative.

        The test differentiates ``j_0(x)`` at ``x=1.3`` with ``jax.grad``.
        It compares the result with the closed-form derivative. The values
        agree within ``1e-10``. This agreement confirms that the Bessel
        implementation supports reverse-mode autodiff for radial integrals.

        Notes
        -----
        The test builds the inputs in the test body and checks the stated property with the documented numerical or structural assertions."""
        x0: Array
        grad_fn: Callable[..., Any]
        grad_val: Array
        expected_grad: Array

        x0 = jnp.asarray(1.3, dtype=jnp.float64)

        def objective(x: chex.Numeric) -> chex.Array:
            return spherical_bessel_jl(0, jnp.asarray(x))

        grad_fn = jax.grad(objective)
        grad_val = grad_fn(x0)
        expected_grad = (x0 * jnp.cos(x0) - jnp.sin(x0)) / (x0 * x0)
        chex.assert_trees_all_close(grad_val, expected_grad, atol=1.0e-10)


class TestBesselErrors:
    """Validate invalid input handling in the Bessel module.

    Validates that ``spherical_bessel_jl`` and the private helper
    ``_odd_double_factorial`` raise ``ValueError`` for out-of-range inputs.

    :see: :func:`~diffpes.radial.spherical_bessel_jl`
    """

    def test_negative_order_raises(self) -> None:
        """Verify that a negative order raises ValueError.

        The test calls ``spherical_bessel_jl`` with ``order=-1`` and expects a
        ``ValueError``. This input covers the guard at the top of the
        function.

        Notes
        -----
        The test builds the inputs in the test body and checks the stated property with the documented numerical or structural assertions."""
        x: Array

        import jax.numpy as jnp
        import pytest

        from diffpes.radial import spherical_bessel_jl

        x = jnp.array([1.0], dtype=jnp.float64)
        with pytest.raises(ValueError, match="non-negative"):
            spherical_bessel_jl(-1, x)

    def test_odd_double_factorial_even_input_raises(self) -> None:
        """Verify that an even input to _odd_double_factorial raises ValueError.

        The implementation uses ``_odd_double_factorial`` internally to compute
        the small-argument Taylor coefficient. It requires a positive odd
        integer; even inputs are invalid.

        Notes
        -----
        The test builds the inputs in the test body and checks the stated property with the documented numerical or structural assertions."""
        import pytest

        from diffpes.radial.bessel import _odd_double_factorial

        with pytest.raises(ValueError, match="positive odd integer"):
            _odd_double_factorial(0)

    def test_odd_double_factorial_even_positive_raises(self) -> None:
        """Verify that a positive even input to _odd_double_factorial raises ValueError.

        The test establishes the odd double factorial even positive raises contract for
        bessel errors with the concrete values and array shapes described below.

        Notes
        -----
        The test builds the inputs in the test body and checks the stated property with the documented numerical or structural assertions."""
        import pytest

        from diffpes.radial.bessel import _odd_double_factorial

        with pytest.raises(ValueError, match="positive odd integer"):
            _odd_double_factorial(4)


class TestSphericalBesselJlDerivative(chex.TestCase):
    """Validate the stable value and derivative branches against SciPy.

    :see: :func:`~diffpes.radial.spherical_bessel_jl_derivative`
    """

    def test_values_match_scipy_over_certified_domain(self) -> None:
        """Compare all branches with SciPy on values including l greater than x.

        The assertions pin the documented numerical contract.

        Notes
        -----
        The test builds the inputs in the test body and checks the stated
        property with the documented numerical or structural assertions.
        """
        positive: Float[NDArray, " n_positive"] = np.concatenate(
            (
                np.array(
                    [
                        0.0,
                        1.0e-8,
                        1.0e-4,
                        1.9e-2,
                        2.1e-2,
                        np.pi - 1.0e-10,
                        np.pi + 1.0e-10,
                        4.493409457909064 - 1.0e-10,
                        4.493409457909064 + 1.0e-10,
                    ]
                ),
                np.geomspace(3.0e-2, 100.0, 81),
            )
        )
        arguments: Float[NDArray, " n_arg"] = np.concatenate(
            (-positive[:0:-1], positive)
        )
        x: Array = jnp.asarray(arguments, dtype=jnp.float64)
        order: int
        for order in range(9):
            expected_positive: Float[NDArray, " n_arg"] = spherical_jn(
                order,
                np.abs(arguments),
            )
            expected: Float[NDArray, " n_arg"] = np.where(
                arguments < 0.0,
                (-1) ** order * expected_positive,
                expected_positive,
            )
            actual: Array = spherical_bessel_jl(order, x)
            np.testing.assert_allclose(
                np.asarray(actual),
                expected,
                rtol=1.0e-12,
                atol=2.0e-14,
            )

    def test_derivative_matches_scipy_and_autodiff(self) -> None:
        """Compare the derivative API and autodiff with SciPy.

        The assertions pin the documented numerical contract.

        Notes
        -----
        The test builds the inputs in the test body and checks the stated
        property with the documented numerical or structural assertions.
        """
        arguments: Float[NDArray, " n_arg"] = np.array(
            [
                -20.0,
                -4.493409457909064,
                -0.1,
                -0.019,
                0.0,
                0.019,
                0.1,
                np.pi,
                4.493409457909064,
                20.0,
            ]
        )
        x: Array = jnp.asarray(arguments, dtype=jnp.float64)
        order: int
        for order in range(9):
            positive_derivative: Float[NDArray, " n_arg"] = spherical_jn(
                order, np.abs(arguments), derivative=True
            )
            expected: Float[NDArray, " n_arg"] = np.where(
                arguments < 0.0,
                (-1) ** (order + 1) * positive_derivative,
                positive_derivative,
            )
            actual: Array = spherical_bessel_jl_derivative(order, x)
            autodiff: Array = jax.vmap(
                jax.grad(lambda value: spherical_bessel_jl(order, value))
            )(x)
            np.testing.assert_allclose(
                np.asarray(actual),
                expected,
                rtol=1.0e-10,
                atol=5.0e-12,
            )
            np.testing.assert_allclose(
                np.asarray(autodiff),
                np.asarray(actual),
                rtol=2.0e-10,
                atol=5.0e-12,
            )

    @chex.variants(with_jit=True, without_jit=True)
    def test_origin_derivatives_are_analytic(self) -> None:
        """Check the exact origin derivatives under eager and JIT execution.

        The assertions pin the documented numerical contract.

        Notes
        -----
        The test builds the inputs in the test body and checks the stated
        property with the documented numerical or structural assertions.
        """
        function: Callable[..., Any] = self.variant(
            lambda value: jnp.stack(
                tuple(
                    spherical_bessel_jl_derivative(order, value)
                    for order in range(3)
                )
            )
        )
        actual: Array = function(jnp.asarray(0.0, dtype=jnp.float64))
        expected: Array = jnp.asarray([0.0, 1.0 / 3.0, 0.0])
        chex.assert_trees_all_close(actual, expected, atol=1.0e-15)

    def test_fwd_and_rev_gradients_agree(self) -> None:
        """Exercise both autodiff modes on upward and Miller branches.

        The assertions pin the documented numerical contract.

        Notes
        -----
        The test builds the inputs in the test body and checks the stated
        property with the documented numerical or structural assertions.
        """
        order: int
        argument: float
        for order, argument in ((1, 0.0), (4, 0.7), (4, 8.0)):
            check_grads(
                lambda value: spherical_bessel_jl(order, jnp.asarray(value)),
                (jnp.asarray(argument, dtype=jnp.float64),),
                order=1,
                modes=("fwd", "rev"),
                rtol=2.0e-6,
                atol=2.0e-8,
            )
