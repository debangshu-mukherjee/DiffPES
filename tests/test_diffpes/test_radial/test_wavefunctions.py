"""Validate radial wavefunction models.

Extended Summary
----------------
The tests validate the ``slater_radial`` and ``hydrogenic_radial``
constructors. The Slater tests verify normalization and compare autodiff
gradients with finite differences. The hydrogenic tests compare the 1s and
2p radial functions with analytical expressions. They also verify the
boundary condition ``R_{2p}(0) = 0``.

"""

import chex
import equinox as eqx
import jax
import jax.numpy as jnp
import pytest
from beartype.typing import Any, Callable
from jaxtyping import Array

from diffpes.radial import hydrogenic_radial, slater_radial
from diffpes.radial.wavefunctions import evaluate_radial
from diffpes.types import make_orbital_basis
from diffpes.types.radial_params import RadialSpec, make_radial_spec


class TestSlaterRadial(chex.TestCase):
    """Validate Slater radial normalization and autodiff gradients.

    The tests verify the normalization of the Slater-type orbital.
    They also compare the ``jax.grad`` result for ``zeta`` with central
    finite differences.

    :see: :func:`~diffpes.radial.slater_radial`
    """

    @chex.variants(with_jit=True, without_jit=True)
    def test_normalization(self) -> None:
        """Verify normalization of the Slater 2s orbital to unity.

        The test constructs ``R(r)`` for ``n=2`` and ``zeta=1.3`` on a
        20000-point grid through 30 Bohr. It integrates ``|R(r)|^2 * r^2`` with
        trapezoidal rule.  Asserts the integral is within 2e-3 of 1.0.
        The dense grid and large cutoff ensure the exponential tail
        contributes negligibly.  Run under both JIT and eager modes.

        Notes
        -----
        The test builds the inputs in the test body and checks the stated property with the documented numerical or structural assertions."""
        r: Array
        fn: Callable[..., Any]
        radial: Array
        norm: Array

        r = jnp.linspace(0.0, 30.0, 20000, dtype=jnp.float64)
        fn = self.variant(lambda radius: slater_radial(radius, n=2, zeta=1.3))
        radial = fn(r)
        norm = jnp.trapezoid((radial**2) * (r**2), x=r)
        chex.assert_trees_all_close(norm, jnp.asarray(1.0), atol=2.0e-3)

    def test_gradient_wrt_zeta_matches_finite_difference(self) -> None:
        """Verify autodiff gradient of Slater sum w.r.t. zeta matches FD.

        The test defines a scalar objective = sum(R(r; zeta)) for n=2, zeta=1.15
        on a 500-point grid up to r=8 Bohr.  Differentiates with
        ``jax.grad`` and compares against a central finite-difference
        estimate with step eps=1e-4.  Asserts agreement to within 2e-4
        (atol and rtol), confirming the normalization constant, power-law
        prefactor, and exponential are all smoothly differentiable.

        Notes
        -----
        The test builds the inputs in the test body and checks the stated property with the documented numerical or structural assertions."""
        r: Array
        zeta0: Array
        eps: Array
        grad_auto: Array
        fd: Array

        r = jnp.linspace(0.0, 8.0, 500, dtype=jnp.float64)
        zeta0 = jnp.asarray(1.15, dtype=jnp.float64)
        eps = jnp.asarray(1.0e-4, dtype=jnp.float64)

        def objective(zeta: chex.Numeric) -> chex.Array:
            return jnp.sum(slater_radial(r, n=2, zeta=jnp.asarray(zeta)))

        grad_auto = jax.grad(objective)(zeta0)
        fd = (objective(zeta0 + eps) - objective(zeta0 - eps)) / (2.0 * eps)
        chex.assert_trees_all_close(grad_auto, fd, atol=2.0e-4, rtol=2.0e-4)


class TestHydrogenicRadial(chex.TestCase):
    """Validate hydrogenic radial wavefunctions against analytical expressions.

    The tests compare ``hydrogenic_radial`` for hydrogen with closed-form
    expressions. They verify the 1s ground state and the 2p boundary condition.
    All radial functions with ``l > 0`` vanish at the origin.

    :see: :func:`~diffpes.radial.hydrogenic_radial`
    """

    @chex.variants(with_jit=True, without_jit=True)
    def test_1s_matches_analytic_expression(self) -> None:
        """Verify R_{10}(r) = 2*exp(-r) for the hydrogen 1s orbital.

        The test evaluates the hydrogenic radial function for ``n=1``,
        ``l=0``, and ``Z_eff=1``. It uses four radii from 0.0 to 2.5 Bohr.
        It compares the result with the analytical 1s expression.
        The test asserts element-wise agreement to within 1e-10.  The r=0 point
        tests the boundary condition R_{10}(0) = 2, and the larger r
        values test the exponential decay.

        Notes
        -----
        The test builds the inputs in the test body and checks the stated property with the documented numerical or structural assertions."""
        r: Array
        fn: Callable[..., Any]
        expected: Array

        r = jnp.array([0.0, 0.3, 1.0, 2.5], dtype=jnp.float64)
        fn = self.variant(
            lambda radius: hydrogenic_radial(
                radius,
                n=1,
                angular_momentum=0,
                z_eff=1.0,
            )
        )
        expected = 2.0 * jnp.exp(-r)
        chex.assert_trees_all_close(fn(r), expected, atol=1.0e-10)

    @chex.variants(with_jit=True, without_jit=True)
    def test_2p_is_zero_at_origin(self) -> None:
        """Verify the 2p (n=2, l=1) radial function vanishes at r=0.

        The test evaluates R_{21}(r=0) for Z_eff=1.  All hydrogenic radial
        functions with l > 0 contain a factor r^l and therefore must
        vanish at the origin.  Asserts the output is zero to within
        1e-12, testing this critical boundary condition / edge case.

        Notes
        -----
        The test builds the inputs in the test body and checks the stated property with the documented numerical or structural assertions."""
        fn: Callable[..., Any]
        value_at_origin: Array

        fn = self.variant(
            lambda radius: hydrogenic_radial(
                radius,
                n=2,
                angular_momentum=1,
                z_eff=1.0,
            )
        )
        value_at_origin = fn(jnp.asarray([0.0], dtype=jnp.float64))
        chex.assert_trees_all_close(
            value_at_origin,
            jnp.asarray([0.0], dtype=jnp.float64),
            atol=1.0e-12,
        )


class TestSlaterRadialErrors:
    """Validate invalid input handling in slater_radial.

    Validates that ``slater_radial`` raises ``ValueError`` when the
    principal quantum number ``n`` is less than 1.

    :see: :func:`~diffpes.radial.slater_radial`
    """

    def test_n_zero_raises(self) -> None:
        """Verify that n=0 raises ValueError.

        The test calls ``slater_radial`` with ``n=0`` and expects a
        ``ValueError`` that matches "n must be >= 1".

        Notes
        -----
        The test builds the inputs in the test body and checks the stated property with the documented numerical or structural assertions."""
        r: Array

        r = jnp.array([1.0], dtype=jnp.float64)
        with pytest.raises(ValueError, match="n must be >= 1"):
            slater_radial(r, n=0, zeta=1.0)


class TestHydrogenicRadialErrors:
    """Validate invalid input handling in hydrogenic_radial.

    Validates that ``hydrogenic_radial`` raises ``ValueError`` for
    invalid principal quantum numbers (n < 1) and for angular momentum
    that violates 0 <= l < n.

    :see: :func:`~diffpes.radial.hydrogenic_radial`
    """

    def test_n_zero_raises(self) -> None:
        """Verify that n=0 raises ValueError for hydrogenic_radial.

        The test calls ``hydrogenic_radial`` with ``n=0`` and expects a
        ``ValueError`` that matches "n must be >= 1".

        Notes
        -----
        The test builds the inputs in the test body and checks the stated property with the documented numerical or structural assertions."""
        r: Array

        r = jnp.array([1.0], dtype=jnp.float64)
        with pytest.raises(ValueError, match="n must be >= 1"):
            hydrogenic_radial(r, n=0, angular_momentum=0, z_eff=1.0)

    def test_angular_momentum_equals_n_raises(self) -> None:
        """Verify that angular_momentum >= n raises ValueError.

        The test calls ``hydrogenic_radial`` with ``n=2`` and
        ``angular_momentum=2``. This input violates the angular-momentum
        constraint. The test expects a matching ``ValueError``.

        Notes
        -----
        The test builds the inputs in the test body and checks the stated property with the documented numerical or structural assertions."""
        r: Array

        r = jnp.array([1.0], dtype=jnp.float64)
        with pytest.raises(ValueError, match="angular_momentum"):
            hydrogenic_radial(r, n=2, angular_momentum=2, z_eff=1.0)


class TestLaguerreRecurrence:
    """Validate the Laguerre polynomial recurrence path.

    Exercises the ``order >= 2`` branch of ``_associated_laguerre``
    that uses ``jax.lax.fori_loop`` for the recurrence, and validates
    the error paths for negative order or alpha.

    :see: :func:`~diffpes.radial.hydrogenic_radial`
    """

    def test_negative_order_raises(self) -> None:
        """Verify that order < 0 raises ValueError in _associated_laguerre.

        The test establishes the negative order raises contract for laguerre recurrence
        with the concrete values and array shapes described below.

        Notes
        -----
        The test builds the inputs in the test body and checks the stated property with the documented numerical or structural assertions."""
        r: Array

        from diffpes.radial.wavefunctions import _associated_laguerre

        r = jnp.array([1.0], dtype=jnp.float64)
        with pytest.raises(ValueError, match="non-negative"):
            _associated_laguerre(-1, 0.5, r)

    def test_negative_alpha_raises(self) -> None:
        """Verify that alpha < 0 raises ValueError in _associated_laguerre.

        The test establishes the negative alpha raises contract for laguerre recurrence
        with the concrete values and array shapes described below.

        Notes
        -----
        The test builds the inputs in the test body and checks the stated property with the documented numerical or structural assertions."""
        r: Array

        from diffpes.radial.wavefunctions import _associated_laguerre

        r = jnp.array([1.0], dtype=jnp.float64)
        with pytest.raises(ValueError, match="non-negative"):
            _associated_laguerre(2, -0.1, r)

    def test_order_one_early_return(self) -> None:
        """Verify that order=1 takes the early-return branch.

        L_1^0(x) = 1 - x, so at x=0 the value is 1.0 and at x=1 it is 0.0.
        This exercises the ``if order == 1: return laguerre_one`` path.

        Notes
        -----
        The test builds the inputs in the test body and checks the stated property with the documented numerical or structural assertions."""
        x: Array
        result: Array
        expected: Array

        from diffpes.radial.wavefunctions import _associated_laguerre

        x = jnp.array([0.0, 1.0], dtype=jnp.float64)
        result = _associated_laguerre(1, 0.0, x)
        expected = jnp.array([1.0, 0.0], dtype=jnp.float64)
        chex.assert_trees_all_close(result, expected, atol=1e-10)

    def test_order_two_uses_recurrence(self) -> None:
        """Verify that order=2 executes the fori_loop recurrence branch.

        The order=0 and order=1 branches return early; order >= 2
        uses the upward recurrence. Checks the known value
        L_2^0(0) = 1 - 0 + 0 = 1.

        Notes
        -----
        The test builds the inputs in the test body and checks the stated property with the documented numerical or structural assertions."""
        x: Array
        result: Array

        from diffpes.radial.wavefunctions import _associated_laguerre

        x = jnp.array([0.0], dtype=jnp.float64)

        result = _associated_laguerre(2, 0.0, x)
        chex.assert_trees_all_close(result, jnp.array([1.0]), atol=1e-10)


class TestEvaluateRadial(chex.TestCase):
    """Validate :func:`diffpes.radial.evaluate_radial`."""

    @staticmethod
    def _basis() -> Any:
        """PRIVATE: Return one complete p shell for shell-sharing checks.

        Returns
        -------
        basis : Any
            Orbital basis with the three ``n=4``, ``l=1`` partners
            ``py``, ``pz``, and ``px`` on one atom.

        Notes
        -----
        All three orbitals share one radial shell, so gathered radial
        rows must be identical across the magnetic partners.
        """
        return make_orbital_basis(
            atom_indices=(0, 0, 0),
            n=(4, 4, 4),
            l=(1, 1, 1),
            m=(-1, 0, 1),
            labels=("py", "pz", "px"),
        )

    @chex.variants(with_jit=True, without_jit=True)
    def test_slater_uses_noninteger_n_star_and_shell_gather(self) -> None:
        """Evaluate one normalized noninteger-n-star row for all p partners.

        The shell uses Slater's effective principal value 3.7.

        Notes
        -----
        Require equal gathered rows, finite origin behavior, and unit norm.
        """
        radial_grid: Array = jnp.linspace(0.0, 120.0, 6001)
        spec: RadialSpec = make_radial_spec(
            self._basis(),
            (0, 0, 0),
            zeta_shell=jnp.asarray(((0.8, 1.4),)),
            coefficients_shell=jnp.asarray(((0.7, -0.2),)),
            n_star_shell=(3.7,),
        )
        evaluator: Callable[..., Any] = self.variant(evaluate_radial)
        values: Array = evaluator(spec, radial_grid)
        norm: Array = jnp.trapezoid(
            values[0] ** 2 * radial_grid**2,
            x=radial_grid,
        )

        chex.assert_shape(values, (3, radial_grid.shape[0]))
        chex.assert_trees_all_close(values[0], values[1])
        chex.assert_trees_all_close(values[1], values[2])
        chex.assert_trees_all_close(norm, jnp.asarray(1.0), atol=2.0e-7)
        assert bool(jnp.isfinite(values[0, 0]))

    def test_coefficient_scale_is_null_but_shape_tangent_is_not(self) -> None:
        """Distinguish the exact coefficient gauge from a physical tangent.

        A common scale leaves the normalized contraction unchanged, while a
        linearly independent coefficient direction changes its shape.

        Notes
        -----
        Compare two JVPs through the public mode-dispatch function.
        """
        radial_grid: Array = jnp.linspace(0.0, 20.0, 301)
        coefficients: Array = jnp.asarray(((0.8, 0.4),))

        def radial_from_coefficients(candidate: Array) -> Array:
            spec: RadialSpec = make_radial_spec(
                self._basis(),
                (0, 0, 0),
                zeta_shell=jnp.asarray(((0.7, 1.5),)),
                coefficients_shell=candidate,
                n_star_shell=(3.7,),
            )
            return evaluate_radial(spec, radial_grid)[0]

        gauge_jvp: Array = jax.jvp(
            radial_from_coefficients,
            (coefficients,),
            (coefficients,),
        )[1]
        shape_jvp: Array = jax.jvp(
            radial_from_coefficients,
            (coefficients,),
            (jnp.asarray(((-0.4, 0.8),)),),
        )[1]

        chex.assert_trees_all_close(
            gauge_jvp,
            jnp.zeros_like(gauge_jvp),
            atol=2.0e-12,
            rtol=2.0e-12,
        )
        assert float(jnp.linalg.norm(shape_jvp)) > 1.0e-3

    def test_zeta_and_hydrogenic_charge_jvps_match_finite_difference(
        self,
    ) -> None:
        """Match autodiff and central differences for both decay parameters.

        The scalar objectives use Slater and hydrogenic mode dispatch.

        Notes
        -----
        Compare centered finite differences at a stable step.
        """
        radial_grid: Array = jnp.linspace(0.0, 10.0, 251)
        epsilon: Array = jnp.asarray(1.0e-5)

        def slater_objective(zeta: Array) -> Array:
            spec: RadialSpec = make_radial_spec(
                self._basis(),
                (0, 0, 0),
                zeta_shell=zeta.reshape((1, 1)),
                n_star_shell=(3.7,),
            )
            return jnp.sum(evaluate_radial(spec, radial_grid)[0])

        hydrogen_basis: Any = make_orbital_basis(
            atom_indices=(0,),
            n=(2,),
            l=(1,),
            m=(0,),
        )

        def hydrogenic_objective(charge: Array) -> Array:
            spec: RadialSpec = make_radial_spec(
                hydrogen_basis,
                (0,),
                mode="hydrogenic",
                effective_charge_shell=charge.reshape((1,)),
            )
            return jnp.sum(evaluate_radial(spec, radial_grid)[0])

        objective: Callable[[Array], Array]
        point: Array
        for objective, point in (
            (slater_objective, jnp.asarray(0.9)),
            (hydrogenic_objective, jnp.asarray(1.4)),
        ):
            automatic: Array = jax.grad(objective)(point)
            finite_difference: Array = (
                objective(point + epsilon) - objective(point - epsilon)
            ) / (2.0 * epsilon)
            chex.assert_trees_all_close(
                automatic,
                finite_difference,
                atol=2.0e-6,
                rtol=2.0e-6,
            )

    def test_rejects_traced_tail_and_coefficient_condition_updates(
        self,
    ) -> None:
        """Reject two post-construction updates outside the certified envelope.

        One update lowers the decay exponent and one creates a nearly
        cancelling contraction with condition above 32.

        Notes
        -----
        Exercise both checks through a compiled public evaluation.
        """
        radial_grid: Array = jnp.linspace(0.0, 12.0, 101)
        spec: RadialSpec = make_radial_spec(
            self._basis(),
            (0, 0, 0),
            zeta_shell=jnp.asarray(((0.8, 0.801),)),
            coefficients_shell=jnp.asarray(((1.0, 0.2),)),
            n_star_shell=(3.7,),
        )

        def update_zeta(candidate: Array) -> Array:
            updated: RadialSpec = eqx.tree_at(
                lambda item: item.zeta_shell,
                spec,
                candidate,
            )
            return evaluate_radial(updated, radial_grid)

        def update_coefficients(candidate: Array) -> Array:
            updated: RadialSpec = eqx.tree_at(
                lambda item: item.coefficients_shell,
                spec,
                candidate,
            )
            return evaluate_radial(updated, radial_grid)

        with pytest.raises(
            eqx.EquinoxRuntimeError,
            match="certified tail envelope",
        ):
            eqx.filter_jit(update_zeta)(
                jnp.asarray(((0.49, 0.801),))
            ).block_until_ready()
        with pytest.raises(
            eqx.EquinoxRuntimeError,
            match="certified tail envelope",
        ):
            eqx.filter_jit(update_zeta)(
                jnp.asarray(((4.01, 0.801),))
            ).block_until_ready()
        with pytest.raises(
            eqx.EquinoxRuntimeError,
            match="coefficient condition",
        ):
            eqx.filter_jit(update_coefficients)(
                jnp.asarray(((1.0, -0.999999),))
            ).block_until_ready()

    def test_grid_is_exact_and_fixed_mode_has_no_radial_function(self) -> None:
        """Enforce exact-grid semantics and the fixed-mode dispatch boundary.

        The evaluator accepts the compact sampled row only at stored coordinates.

        Notes
        -----
        Use shifted-grid and fixed-mode calls as planted false controls.
        """
        basis: Any = make_orbital_basis(
            atom_indices=(0,),
            n=(1,),
            l=(0,),
            m=(0,),
        )
        radial_grid: Array = jnp.linspace(0.0, 8.0, 81)
        samples: Array = jnp.exp(-radial_grid).at[-1].set(0.0)[None, :]
        grid_spec: RadialSpec = make_radial_spec(
            basis,
            (0,),
            mode="grid",
            r_grid=radial_grid,
            grid_values_shell=samples,
        )
        fixed_spec: RadialSpec = make_radial_spec(
            basis,
            (0,),
            mode="fixed",
            fixed_integrals_shell=jnp.asarray(((3.0, 4.0),)),
        )

        values: Array = evaluate_radial(grid_spec, radial_grid)
        chex.assert_shape(values, (1, radial_grid.shape[0]))
        with pytest.raises(eqx.EquinoxRuntimeError, match="no interpolation"):
            evaluate_radial(grid_spec, radial_grid + 1.0e-6)
        with pytest.raises(ValueError, match="no radial function"):
            evaluate_radial(fixed_spec, radial_grid)
