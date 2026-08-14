"""Validate fixed-quadrature dipole radial integrals.

Extended Summary
----------------
The tests use analytic transforms and derivatives of the public
:math:`R(r)r^3` measure.  They separately exercise Gauss--Legendre setup,
uniform-grid Simpson quadrature, the inverse-Angstrom conversion seam, and
the single partial-wave phase.
"""

import chex
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from beartype import beartype
from beartype.typing import Any, Callable, List
from jaxtyping import Array, Complex128, Float64, TypeCheckError, jaxtyped

from diffpes.radial import (
    gauss_legendre_nodes,
    momentum_inv_ang_to_bohr_inv,
    radial_bvals,
    radial_integral,
    radial_integral_simpson,
    spherical_bessel_jl,
)
from diffpes.types import (
    FinalStateSpec,
    OrbitalBasis,
    RadialQuadratureSpec,
    RadialSpec,
    make_final_state_spec,
    make_orbital_basis,
    make_radial_quadrature_spec,
    make_radial_spec,
)


@jaxtyped(typechecker=beartype)
def _exp_r3_transform(
    decay: Float64[Array, " ..."],
    momentum: Float64[Array, " ..."],
) -> Float64[Array, " ..."]:
    """PRIVATE: Return the analytic exponential r-cubed Bessel transform.

    Parameters
    ----------
    decay : Float64[Array, " ..."]
        Exponential decay rate ``a`` in inverse Bohr.
    momentum : Float64[Array, " ..."]
        Final-state momentum ``k`` in inverse Bohr.

    Returns
    -------
    values : Float64[Array, " ..."]
        The closed form ``2 * (3*a**2 - k**2) / (a**2 + k**2)**3`` in
        Bohr**4.

    Notes
    -----
    This is the exact infinite integral of ``exp(-a*r) * j_0(k*r) *
    r**3`` over ``r`` from zero to infinity. The quadrature tests
    compare :func:`radial_integral` on a truncated grid against it.
    """
    denominator: Float64[Array, " ..."] = decay * decay + momentum * momentum
    values: Float64[Array, " ..."] = (
        2.0 * (3.0 * decay * decay - momentum * momentum) / denominator**3
    )
    return values


class TestGaussLegendreNodes:
    """Validate host-side Gauss--Legendre setup.

    :see: :func:`~diffpes.radial.gauss_legendre_nodes`
    """

    def test_polynomial_moments_are_exact(self) -> None:
        """Check transformed nodes, positivity, and polynomial moments.

        The assertions pin the documented numerical contract.

        Notes
        -----
        The test builds the inputs in the test body and checks the stated
        property with the documented numerical or structural assertions.
        """
        nodes: Float64[Array, " 12"]
        weights: Float64[Array, " 12"]
        nodes, weights = gauss_legendre_nodes(12, 3.0)
        assert np.all(np.diff(np.asarray(nodes)) > 0.0)
        assert np.all(np.asarray(weights) > 0.0)
        power: int
        numeric: Float64[Array, ""]
        expected: float
        for power in range(12):
            numeric = jnp.sum(weights * nodes**power)
            expected = 3.0 ** (power + 1) / (power + 1)
            np.testing.assert_allclose(
                np.asarray(numeric),
                expected,
                rtol=2.0e-14,
                atol=2.0e-14,
            )

    def test_invalid_setup_raises(self) -> None:
        """Reject nonpositive node counts and radial bounds.

        The assertions pin the documented numerical contract.

        Notes
        -----
        The test builds the inputs in the test body and checks the stated
        property with the documented numerical or structural assertions.
        """
        with pytest.raises(ValueError, match="positive integer"):
            gauss_legendre_nodes(0, 3.0)
        with pytest.raises(ValueError, match="finite and positive"):
            gauss_legendre_nodes(8, 0.0)


class TestRadialIntegral(chex.TestCase):
    """Validate weighted radial-integral values and derivatives.

    :see: :func:`~diffpes.radial.radial_integral`
    """

    @chex.variants(with_jit=True, without_jit=True)
    def test_public_r3_measure_matches_analytic_transform(self) -> None:
        """Compare the public weighted API with the exact infinite integral.

        The assertions pin the documented numerical contract.

        Notes
        -----
        The decay ``a=2`` makes the omitted tail beyond 30 Bohr less than
        ``2e-21`` relative, well below the stated numerical tolerance.
        """
        nodes: Float64[Array, " 256"]
        weights: Float64[Array, " 256"]
        nodes, weights = gauss_legendre_nodes(256, 30.0)
        decay: Float64[Array, ""] = jnp.asarray(2.0, dtype=jnp.float64)
        momenta: Float64[Array, " 4"] = jnp.asarray(
            [0.0, 0.2, 0.8, 1.4],
            dtype=jnp.float64,
        )
        radial: Float64[Array, " 256"] = jnp.exp(-decay * nodes)
        function: Callable[..., Any] = self.variant(
            lambda values: radial_integral(
                values,
                nodes,
                weights,
                radial,
                0,
            )
        )
        actual: Float64[Array, " 4"] = jnp.real(function(momenta))
        expected: Float64[Array, " 4"] = _exp_r3_transform(decay, momenta)
        chex.assert_trees_all_close(
            actual,
            expected,
            rtol=1.0e-10,
            atol=1.0e-12,
        )

    def test_analytic_decay_and_momentum_derivatives(self) -> None:
        """Match autodiff with both closed-form parameter derivatives.

        The assertions pin the documented numerical contract.

        Notes
        -----
        The test calls the production public integral rather than a private
        integrand or an r-squared surrogate.
        """
        nodes: Float64[Array, " 384"]
        weights: Float64[Array, " 384"]
        nodes, weights = gauss_legendre_nodes(384, 30.0)
        decay: Float64[Array, ""] = jnp.asarray(2.0, dtype=jnp.float64)
        momentum: Float64[Array, ""] = jnp.asarray(0.7, dtype=jnp.float64)

        @jaxtyped(typechecker=beartype)
        def _objective_decay(
            value: Float64[Array, ""],
        ) -> Float64[Array, ""]:
            """PRIVATE: Evaluate the real integral for one decay constant.

            Parameters
            ----------
            value : Float64[Array, ""]
                Exponential decay constant in inverse Bohr.

            Returns
            -------
            result : Float64[Array, ""]
                Real s-wave radial integral.

            Notes
            -----
            Holds the momentum and quadrature fixed while varying the
            exponential samples.
            """
            result: Float64[Array, ""] = jnp.real(
                radial_integral(
                    momentum,
                    nodes,
                    weights,
                    jnp.exp(-value * nodes),
                    0,
                )
            )
            return result

        @jaxtyped(typechecker=beartype)
        def _objective_momentum(
            value: Float64[Array, ""],
        ) -> Float64[Array, ""]:
            """PRIVATE: Evaluate the real integral for one momentum.

            Parameters
            ----------
            value : Float64[Array, ""]
                Momentum in inverse Bohr.

            Returns
            -------
            result : Float64[Array, ""]
                Real s-wave radial integral.

            Notes
            -----
            Holds the decay constant and quadrature fixed while varying the
            Bessel-function argument.
            """
            result: Float64[Array, ""] = jnp.real(
                radial_integral(
                    value,
                    nodes,
                    weights,
                    jnp.exp(-decay * nodes),
                    0,
                )
            )
            return result

        denominator: Float64[Array, ""] = decay * decay + momentum * momentum
        expected_decay: Float64[Array, ""] = (
            24.0
            * decay
            * (momentum * momentum - decay * decay)
            / denominator**4
        )
        expected_momentum: Float64[Array, ""] = (
            8.0
            * momentum
            * (momentum * momentum - 5.0 * decay * decay)
            / denominator**4
        )
        chex.assert_trees_all_close(
            jax.grad(_objective_decay)(decay),
            expected_decay,
            rtol=1.0e-10,
            atol=1.0e-12,
        )
        chex.assert_trees_all_close(
            jax.grad(_objective_momentum)(momentum),
            expected_momentum,
            rtol=1.0e-10,
            atol=1.0e-12,
        )

    def test_node_doubling_is_below_registered_value_budget(self) -> None:
        """Require 256-to-512 convergence on the analytic radial fixture.

        The assertions pin the documented numerical contract.

        Notes
        -----
        This is a quadrature-only convergence check.  The separately
        registered production envelope owns its missing-tail certification.
        """
        momentum: Float64[Array, " 41"] = jnp.linspace(
            0.0, 4.0, 41, dtype=jnp.float64
        )
        values: List[Complex128[Array, " 41"]] = []
        n_nodes: int
        nodes: Float64[Array, " n_node"]
        weights: Float64[Array, " n_node"]
        for n_nodes in (256, 512):
            nodes, weights = gauss_legendre_nodes(n_nodes, 30.0)
            values.append(
                radial_integral(
                    momentum,
                    nodes,
                    weights,
                    jnp.exp(-2.0 * nodes),
                    0,
                )
            )
        chex.assert_trees_all_close(
            values[0],
            values[1],
            rtol=1.0e-10,
            atol=1.0e-12,
        )

    def test_partial_wave_phase_is_applied_once(self) -> None:
        """Pin the i-to-the-l phase and reject its omission or duplication.

        The assertions pin the documented numerical contract.

        Notes
        -----
        A direct real quadrature supplies the independent magnitude.  The
        public result must equal that magnitude times exactly one phase.
        """
        nodes: Float64[Array, " 256"]
        weights: Float64[Array, " 256"]
        nodes, weights = gauss_legendre_nodes(256, 30.0)
        momentum: Float64[Array, ""] = jnp.asarray(0.7, dtype=jnp.float64)
        radial: Float64[Array, " 256"] = jnp.exp(-2.0 * nodes)
        order: int
        kr: Float64[Array, " 256"]
        real_reference: Float64[Array, ""]
        actual: Complex128[Array, ""]
        expected: Complex128[Array, ""]
        for order in range(5):
            kr = momentum * nodes
            real_reference = jnp.sum(
                weights * radial * nodes**3 * spherical_bessel_jl(order, kr)
            )
            actual = radial_integral(momentum, nodes, weights, radial, order)
            expected = (1j) ** order * real_reference
            chex.assert_trees_all_close(actual, expected, atol=1.0e-14)
            if order in (1, 2, 3):
                assert not bool(
                    jnp.isclose(
                        actual,
                        (1j) ** (2 * order) * real_reference,
                        rtol=1.0e-8,
                        atol=1.0e-12,
                    )
                )

    def test_vmap_matches_batched_evaluation(self) -> None:
        """Compare explicit vectorization with native leading-axis batching.

        The assertions pin the documented numerical contract.

        Notes
        -----
        The test builds the inputs in the test body and checks the stated
        property with the documented numerical or structural assertions.
        """
        nodes: Float64[Array, " 128"]
        weights: Float64[Array, " 128"]
        nodes, weights = gauss_legendre_nodes(128, 20.0)
        momenta: Float64[Array, " 3"] = jnp.asarray(
            [0.1, 0.5, 1.3],
            dtype=jnp.float64,
        )
        radial: Float64[Array, " 128"] = jnp.exp(-1.5 * nodes)
        batched: Complex128[Array, " 3"] = radial_integral(
            momenta, nodes, weights, radial, 1
        )
        mapped: Complex128[Array, " 3"] = jax.vmap(
            lambda value: radial_integral(value, nodes, weights, radial, 1)
        )(momenta)
        chex.assert_trees_all_close(batched, mapped, atol=1.0e-14)


class TestRadialIntegralSimpson:
    """Validate the uniform-grid composite Simpson path.

    :see: :func:`~diffpes.radial.radial_integral_simpson`
    """

    def test_cubic_measure_is_exact_at_zero_momentum(self) -> None:
        """Use Simpson's exact cubic integration property.

        The assertions pin the documented numerical contract.

        Notes
        -----
        At zero momentum and order zero, a constant radial row leaves the
        exact integral of r cubed on [0,2], which equals four.
        """
        radial_grid: Float64[Array, " 101"] = jnp.linspace(
            0.0,
            2.0,
            101,
            dtype=jnp.float64,
        )
        radial: Float64[Array, " 101"] = jnp.ones_like(radial_grid)
        actual: Complex128[Array, ""] = radial_integral_simpson(
            jnp.asarray(0.0),
            radial_grid,
            radial,
            0,
        )
        np.testing.assert_allclose(
            np.asarray(actual),
            4.0 + 0.0j,
            rtol=0.0,
            atol=2.0e-14,
        )

    def test_invalid_point_count_and_nonuniform_grid_raise(self) -> None:
        """Reject grids outside the composite Simpson contract.

        The assertions pin the documented numerical contract.

        Notes
        -----
        The test builds the inputs in the test body and checks the stated
        property with the documented numerical or structural assertions.
        """
        even_grid: Float64[Array, " 100"] = jnp.linspace(
            0.0,
            2.0,
            100,
            dtype=jnp.float64,
        )
        with pytest.raises(ValueError, match="odd point count"):
            radial_integral_simpson(
                jnp.asarray(0.5),
                even_grid,
                jnp.ones_like(even_grid),
                0,
            )
        nonuniform: Float64[Array, " 5"] = jnp.asarray(
            [0.0, 0.2, 0.5, 1.0, 2.0]
        )
        with pytest.raises(
            (eqx.EquinoxRuntimeError, jax.errors.JaxRuntimeError),
            match="uniform",
        ):
            radial_integral_simpson(
                jnp.asarray(0.5),
                nonuniform,
                jnp.ones_like(nonuniform),
                0,
            )


class TestMomentumInvAngToBohrInv:
    """Validate the inverse-Angstrom to inverse-Bohr conversion.

    :see: :func:`~diffpes.radial.momentum_inv_ang_to_bohr_inv`
    """

    def test_conversion_and_reciprocal_false_control(self) -> None:
        """Pin multiplication by the Bohr radius and reject its reciprocal.

        The assertions pin the documented numerical contract.

        Notes
        -----
        The converted momentum enters an analytic dimensionless ``kr``
        reference.  The reciprocal conversion is a deliberately planted
        failure.
        """
        momentum_ang: Float64[Array, ""] = jnp.asarray(2.3, dtype=jnp.float64)
        converted: Float64[Array, ""] = momentum_inv_ang_to_bohr_inv(
            momentum_ang
        )
        expected: Float64[Array, ""] = momentum_ang * 0.529177210903
        chex.assert_trees_all_close(converted, expected, atol=0.0, rtol=0.0)
        reciprocal: Float64[Array, ""] = momentum_ang / 0.529177210903
        assert not bool(jnp.isclose(converted, reciprocal, rtol=1.0e-6))


class TestRadialBvals:
    """Validate direct shell assembly and certified-domain guards.

    :see: :func:`~diffpes.radial.radial_bvals`
    """

    def test_fixed_rows_are_phase_free_calibration_shapes(self) -> None:
        """Normalize fixed real rows, gather shells, and apply one phase.

        The assertions pin the documented numerical contract.

        Notes
        -----
        The s-shell nonexistent lower channel is exactly zero. The p-shell
        fixture distinguishes the l-prime zero and two phases.
        """
        basis: OrbitalBasis = make_orbital_basis(
            atom_indices=(0, 1),
            n=(1, 2),
            l=(0, 1),
            m=(0, 0),
        )
        spec: RadialSpec = make_radial_spec(
            basis,
            (0, 1),
            mode="fixed",
            fixed_integrals_shell=jnp.asarray(
                [[0.0, 2.0], [3.0, 4.0]], dtype=jnp.float64
            ),
        )
        values: Complex128[Array, "2 2 2"] = radial_bvals(
            spec,
            jnp.asarray([0.2, 1.1], dtype=jnp.float64),
            make_radial_quadrature_spec(),
            make_final_state_spec(),
        )
        expected_orbitals: Complex128[Array, "2 2"] = jnp.asarray(
            [[0.0 + 0.0j, 0.0 + 1.0j], [0.6 + 0.0j, -0.8 + 0.0j]]
        )
        expected: Complex128[Array, "2 2 2"] = jnp.broadcast_to(
            expected_orbitals, (2, 2, 2)
        )
        chex.assert_trees_all_close(values, expected, atol=1.0e-14)

    def test_normalized_slater_s_to_p_matches_closed_form(self) -> None:
        """Compare shell assembly with the normalized analytic 1s transform.

        The assertions pin the documented numerical contract.

        Notes
        -----
        The radial profile uses R=120 Bohr, making the zeta-one missing tail
        negligible for this fixture.
        """
        basis: OrbitalBasis = make_orbital_basis(
            atom_indices=(0,),
            n=(1,),
            l=(0,),
            m=(0,),
        )
        zeta: Float64[Array, "1 1"] = jnp.asarray([[1.0]], dtype=jnp.float64)
        spec: RadialSpec = make_radial_spec(
            basis,
            (0,),
            mode="slater",
            zeta_shell=zeta,
        )
        momenta: Float64[Array, " 3"] = jnp.asarray(
            [0.1, 0.7, 1.4],
            dtype=jnp.float64,
        )
        values: Complex128[Array, "3 1 2"] = radial_bvals(
            spec,
            momenta,
            make_radial_quadrature_spec(),
            make_final_state_spec(),
        )
        normalization: Float64[Array, ""] = (
            2.0 * zeta[0, 0]
        ) ** 1.5 / jnp.sqrt(2.0)
        expected_upper: Complex128[Array, " 3"] = (
            1j
            * normalization
            * 8.0
            * zeta[0, 0]
            * momenta
            / (zeta[0, 0] ** 2 + momenta**2) ** 3
        )
        chex.assert_trees_all_close(values[..., 0, 0], 0.0, atol=0.0)
        chex.assert_trees_all_close(
            values[..., 0, 1],
            expected_upper,
            rtol=1.0e-10,
            atol=1.0e-12,
        )

    @pytest.mark.slow
    def test_slater_gradient_and_coefficient_scale_gauge(self) -> None:
        """Check a physical exponent gradient and the normalized scale gauge.

        The assertions pin the documented numerical contract.

        Notes
        -----
        Normalizing a contracted radial removes its common coefficient scale.
        A tangent changing the contraction shape remains nonzero.
        """
        basis: OrbitalBasis = make_orbital_basis(
            atom_indices=(0,),
            n=(2,),
            l=(1,),
            m=(0,),
        )
        base: RadialSpec = make_radial_spec(
            basis,
            (0,),
            mode="slater",
            zeta_shell=jnp.asarray([[0.8, 1.6]]),
            coefficients_shell=jnp.asarray([[0.6, -0.8]]),
        )
        quadrature: RadialQuadratureSpec = make_radial_quadrature_spec()
        final_state: FinalStateSpec = make_final_state_spec()
        momentum: Float64[Array, ""] = jnp.asarray(0.9, dtype=jnp.float64)

        @jaxtyped(typechecker=beartype)
        def _with_zeta(
            value: Float64[Array, ""],
        ) -> Float64[Array, ""]:
            """PRIVATE: Evaluate one radial row with a varied exponent.

            Parameters
            ----------
            value : Float64[Array, ""]
                First Slater exponent in inverse Bohr.

            Returns
            -------
            result : Float64[Array, ""]
                Real lower-channel radial value.

            Notes
            -----
            Replaces only the first exponent in the closed-over radial
            specification.
            """
            candidate: RadialSpec = eqx.tree_at(
                lambda item: item.zeta_shell,
                base,
                base.zeta_shell.at[0, 0].set(value),
            )
            result: Float64[Array, ""] = jnp.real(
                radial_bvals(candidate, momentum, quadrature, final_state)[
                    0, 0
                ]
            )
            return result

        @jaxtyped(typechecker=beartype)
        def _with_scale(
            value: Float64[Array, ""],
        ) -> Float64[Array, ""]:
            """PRIVATE: Evaluate one radial row after coefficient scaling.

            Parameters
            ----------
            value : Float64[Array, ""]
                Common dimensionless coefficient scale.

            Returns
            -------
            result : Float64[Array, ""]
                Real lower-channel radial value.

            Notes
            -----
            Multiplies every contraction coefficient by the same scale.
            """
            candidate: RadialSpec = eqx.tree_at(
                lambda item: item.coefficients_shell,
                base,
                value * base.coefficients_shell,
            )
            result: Float64[Array, ""] = jnp.real(
                radial_bvals(candidate, momentum, quadrature, final_state)[
                    0, 0
                ]
            )
            return result

        zeta_gradient: Float64[Array, ""] = jax.grad(_with_zeta)(
            base.zeta_shell[0, 0]
        )
        epsilon: Float64[Array, ""] = jnp.asarray(2.0e-5)
        zeta_fd: Float64[Array, ""] = (
            _with_zeta(base.zeta_shell[0, 0] + epsilon)
            - _with_zeta(base.zeta_shell[0, 0] - epsilon)
        ) / (2.0 * epsilon)
        chex.assert_trees_all_close(
            zeta_gradient, zeta_fd, rtol=2.0e-6, atol=2.0e-8
        )
        assert bool(jnp.abs(zeta_gradient) > 1.0e-6)
        chex.assert_trees_all_close(
            jax.grad(_with_scale)(jnp.asarray(1.0)),
            0.0,
            atol=2.0e-12,
        )

    @pytest.mark.slow
    def test_registered_profile_node_doubling_values_and_gradient(
        self,
    ) -> None:
        """Measure production-to-reference agreement at the hard envelope.

        The assertions pin the documented numerical contract.

        Notes
        -----
        The fixture combines the slowest certified decay, the largest
        supported momentum, and nontrivial d/f partial-wave channels.
        Value and exponent-gradient differences must remain within the
        production profile's registered budgets.
        """
        basis: OrbitalBasis = make_orbital_basis(
            atom_indices=(0,),
            n=(4,),
            l=(3,),
            m=(0,),
        )
        base: RadialSpec = make_radial_spec(
            basis,
            (0,),
            mode="slater",
            zeta_shell=jnp.asarray([[0.5]], dtype=jnp.float64),
        )
        production: RadialQuadratureSpec = make_radial_quadrature_spec()
        reference: RadialQuadratureSpec = make_radial_quadrature_spec(
            "gl2048-r120-k4-l9-reference-v1"
        )
        final_state: FinalStateSpec = make_final_state_spec()
        momenta: Float64[Array, " 4"] = jnp.asarray(
            [0.0, 0.4, 1.7, 4.0],
            dtype=jnp.float64,
        )

        @jaxtyped(typechecker=beartype)
        def _evaluated(
            exponent: Float64[Array, ""],
            quadrature: RadialQuadratureSpec,
        ) -> Complex128[Array, "4 1 2"]:
            """PRIVATE: Evaluate the registered radial profile.

            Parameters
            ----------
            exponent : Float64[Array, ""]
                Slater exponent in inverse Bohr.
            quadrature : RadialQuadratureSpec
                Fixed quadrature calibration to evaluate.

            Returns
            -------
            result : Complex128[Array, "4 1 2"]
                Radial channels for four momenta and one orbital.

            Notes
            -----
            Replaces the sole exponent before calling the public assembler.
            """
            candidate: RadialSpec = eqx.tree_at(
                lambda item: item.zeta_shell,
                base,
                base.zeta_shell.at[0, 0].set(exponent),
            )
            result: Complex128[Array, "4 1 2"] = radial_bvals(
                candidate,
                momenta,
                quadrature,
                final_state,
            )
            return result

        exponent: Float64[Array, ""] = base.zeta_shell[0, 0]
        production_values: Complex128[Array, "4 1 2"] = _evaluated(
            exponent, production
        )
        reference_values: Complex128[Array, "4 1 2"] = _evaluated(
            exponent, reference
        )
        value_scale: Float64[Array, ""] = jnp.maximum(
            1.0,
            jnp.max(jnp.abs(reference_values)),
        )
        assert bool(
            jnp.max(jnp.abs(production_values - reference_values))
            <= production.value_rtol * value_scale
        )

        @jaxtyped(typechecker=beartype)
        def _scalar_objective(
            exponent: Float64[Array, ""],
            quadrature: RadialQuadratureSpec,
        ) -> Float64[Array, ""]:
            """PRIVATE: Sum squared radial-channel magnitudes.

            Parameters
            ----------
            exponent : Float64[Array, ""]
                Slater exponent in inverse Bohr.
            quadrature : RadialQuadratureSpec
                Fixed quadrature calibration to evaluate.

            Returns
            -------
            result : Float64[Array, ""]
                Sum of squared complex radial-channel magnitudes.

            Notes
            -----
            Reduces every momentum, orbital, and branch component.
            """
            values: Complex128[Array, "4 1 2"] = _evaluated(
                exponent, quadrature
            )
            result: Float64[Array, ""] = jnp.sum(
                jnp.real(values) ** 2 + jnp.imag(values) ** 2
            )
            return result

        production_gradient: Float64[Array, ""] = jax.grad(
            _scalar_objective, argnums=0
        )(
            exponent,
            production,
        )
        reference_gradient: Float64[Array, ""] = jax.grad(
            _scalar_objective, argnums=0
        )(
            exponent,
            reference,
        )
        gradient_scale: Float64[Array, ""] = jnp.maximum(
            1.0,
            jnp.abs(reference_gradient),
        )
        assert bool(
            jnp.abs(production_gradient - reference_gradient)
            <= production.gradient_rtol * gradient_scale
        )

    def test_grid_mode_uses_compact_support_and_simpson(self) -> None:
        """Match grid-mode shell assembly with direct Simpson quadrature.

        The assertions pin the documented numerical contract.

        Notes
        -----
        The stored row is exactly zero at its finite endpoint. The reference
        independently applies Simpson quadrature to the carrier-normalized
        compact-support row.
        """
        basis: OrbitalBasis = make_orbital_basis(
            atom_indices=(0,),
            n=(1,),
            l=(0,),
            m=(0,),
        )
        radial_grid: Float64[Array, " 1001"] = jnp.linspace(
            0.0,
            20.0,
            1001,
            dtype=jnp.float64,
        )
        raw_values: Float64[Array, " 1001"] = (
            jnp.exp(-radial_grid).at[-1].set(0.0)
        )
        spec: RadialSpec = make_radial_spec(
            basis,
            (0,),
            mode="grid",
            r_grid=radial_grid,
            grid_values_shell=raw_values[None, :],
        )
        momentum: Float64[Array, ""] = jnp.asarray(0.8, dtype=jnp.float64)
        actual: Complex128[Array, ""] = radial_bvals(
            spec,
            momentum,
            make_radial_quadrature_spec(),
            make_final_state_spec(),
        )[0, 1]
        stored: Float64[Array, "n_shell n_r"] | None = spec.grid_values_shell
        assert stored is not None
        expected: Complex128[Array, ""] = radial_integral_simpson(
            momentum,
            radial_grid,
            stored[0],
            1,
        )
        chex.assert_trees_all_close(
            actual, expected, rtol=1.0e-12, atol=1.0e-14
        )

    @pytest.mark.slow
    def test_domain_and_rejected_accelerator_guards(self) -> None:
        """Reject out-of-profile momentum and uncertified acceleration.

        The assertions pin the documented numerical contract.

        Notes
        -----
        Direct Coulomb evaluation is a distinct complex path. The failed
        fixed convergence check makes Hermite acceleration unavailable.
        """
        basis: OrbitalBasis = make_orbital_basis(
            atom_indices=(0,),
            n=(1,),
            l=(0,),
            m=(0,),
        )
        spec: RadialSpec = make_radial_spec(basis, (0,), mode="slater")
        quadrature: RadialQuadratureSpec = make_radial_quadrature_spec()
        with pytest.raises(
            (eqx.EquinoxRuntimeError, jax.errors.JaxRuntimeError),
            match="certified quadrature domain",
        ):
            radial_bvals(
                spec,
                jnp.asarray(4.01),
                quadrature,
                make_final_state_spec(),
            )
        coulomb_values: Complex128[Array, "1 2"] = radial_bvals(
            spec,
            jnp.asarray(1.0),
            quadrature,
            make_final_state_spec(mode="coulomb", effective_charge=1.0),
        )
        chex.assert_tree_all_finite(coulomb_values)
        assert bool(jnp.any(jnp.abs(jnp.imag(coulomb_values)) > 1.0e-8))
        with pytest.raises(
            ValueError, match="failed the frozen radial accelerator"
        ):
            make_final_state_spec(radial_accelerator="hermite")


class TestRadialIntegrateErrors:
    """Validate invalid radial-integral inputs.

    :see: :func:`~diffpes.radial.radial_integral`
    """

    def test_negative_l_prime_and_axis_mismatch_raise(self) -> None:
        """Reject negative orders and inconsistent radial vectors.

        The assertions pin the documented numerical contract.

        Notes
        -----
        The test builds the inputs in the test body and checks the stated
        property with the documented numerical or structural assertions.
        """
        nodes: Float64[Array, " 8"]
        weights: Float64[Array, " 8"]
        nodes, weights = gauss_legendre_nodes(8, 3.0)
        with pytest.raises(ValueError, match="non-negative"):
            radial_integral(
                jnp.asarray(1.0),
                nodes,
                weights,
                jnp.ones_like(nodes),
                -1,
            )
        with pytest.raises(TypeCheckError, match="weights_bohr"):
            radial_integral(
                jnp.asarray(1.0),
                nodes,
                weights[:-1],
                jnp.ones_like(nodes),
                0,
            )
