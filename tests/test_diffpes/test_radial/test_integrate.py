"""Validate fixed-quadrature dipole radial integrals.

Extended Summary
----------------
The tests use analytic transforms and derivatives of the public
:math:`R(r)r^3` measure.  They separately exercise Gauss--Legendre setup,
uniform-grid Simpson quadrature, the inverse-Angstrom conversion seam, and
the single partial-wave phase.
"""

from collections.abc import Callable
from typing import Any

import chex
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jaxtyping import Array

from diffpes.radial.bessel import spherical_bessel_jl
from diffpes.radial.integrate import (
    gauss_legendre_nodes,
    momentum_inv_ang_to_bohr_inv,
    radial_bvals,
    radial_integral,
    radial_integral_simpson,
)
from diffpes.types.radial_params import (
    FinalStateSpec,
    OrbitalBasis,
    RadialQuadratureSpec,
    RadialSpec,
    make_final_state_spec,
    make_orbital_basis,
    make_radial_quadrature_spec,
    make_radial_spec,
)


def _exp_r3_transform(
    decay: Array,
    momentum: Array,
) -> Array:
    """Return the analytic exponential r-cubed Bessel transform."""
    denominator: Array = decay * decay + momentum * momentum
    values: Array = (
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
        nodes: Array
        weights: Array
        nodes, weights = gauss_legendre_nodes(12, 3.0)
        assert np.all(np.diff(np.asarray(nodes)) > 0.0)
        assert np.all(np.asarray(weights) > 0.0)
        power: int
        numeric: Array
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
        nodes: Array
        weights: Array
        nodes, weights = gauss_legendre_nodes(256, 30.0)
        decay: Array = jnp.asarray(2.0, dtype=jnp.float64)
        momenta: Array = jnp.asarray(
            [0.0, 0.2, 0.8, 1.4],
            dtype=jnp.float64,
        )
        radial: Array = jnp.exp(-decay * nodes)
        function: Callable[..., Any] = self.variant(
            lambda values: radial_integral(
                values,
                nodes,
                weights,
                radial,
                0,
            )
        )
        actual: Array = jnp.real(function(momenta))
        expected: Array = _exp_r3_transform(decay, momenta)
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
        nodes: Array
        weights: Array
        nodes, weights = gauss_legendre_nodes(384, 30.0)
        decay: Array = jnp.asarray(2.0, dtype=jnp.float64)
        momentum: Array = jnp.asarray(0.7, dtype=jnp.float64)

        def objective_decay(value: Array) -> Array:
            return jnp.real(
                radial_integral(
                    momentum,
                    nodes,
                    weights,
                    jnp.exp(-value * nodes),
                    0,
                )
            )

        def objective_momentum(value: Array) -> Array:
            return jnp.real(
                radial_integral(
                    value,
                    nodes,
                    weights,
                    jnp.exp(-decay * nodes),
                    0,
                )
            )

        denominator: Array = decay * decay + momentum * momentum
        expected_decay: Array = (
            24.0
            * decay
            * (momentum * momentum - decay * decay)
            / denominator**4
        )
        expected_momentum: Array = (
            8.0
            * momentum
            * (momentum * momentum - 5.0 * decay * decay)
            / denominator**4
        )
        chex.assert_trees_all_close(
            jax.grad(objective_decay)(decay),
            expected_decay,
            rtol=1.0e-10,
            atol=1.0e-12,
        )
        chex.assert_trees_all_close(
            jax.grad(objective_momentum)(momentum),
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
        momentum: Array = jnp.linspace(0.0, 4.0, 41, dtype=jnp.float64)
        values: list[Array] = []
        n_nodes: int
        nodes: Array
        weights: Array
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
        nodes: Array
        weights: Array
        nodes, weights = gauss_legendre_nodes(256, 30.0)
        momentum: Array = jnp.asarray(0.7, dtype=jnp.float64)
        radial: Array = jnp.exp(-2.0 * nodes)
        order: int
        kr: Array
        real_reference: Array
        actual: Array
        expected: Array
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
        nodes: Array
        weights: Array
        nodes, weights = gauss_legendre_nodes(128, 20.0)
        momenta: Array = jnp.asarray(
            [0.1, 0.5, 1.3],
            dtype=jnp.float64,
        )
        radial: Array = jnp.exp(-1.5 * nodes)
        batched: Array = radial_integral(momenta, nodes, weights, radial, 1)
        mapped: Array = jax.vmap(
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
        radial_grid: Array = jnp.linspace(
            0.0,
            2.0,
            101,
            dtype=jnp.float64,
        )
        radial: Array = jnp.ones_like(radial_grid)
        actual: Array = radial_integral_simpson(
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
        even_grid: Array = jnp.linspace(
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
        nonuniform: Array = jnp.asarray([0.0, 0.2, 0.5, 1.0, 2.0])
        with pytest.raises(Exception, match="uniform"):
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
        momentum_ang: Array = jnp.asarray(2.3, dtype=jnp.float64)
        converted: Array = momentum_inv_ang_to_bohr_inv(momentum_ang)
        expected: Array = momentum_ang * 0.529177210903
        chex.assert_trees_all_close(converted, expected, atol=0.0, rtol=0.0)
        reciprocal: Array = momentum_ang / 0.529177210903
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
        values: Array = radial_bvals(
            spec,
            jnp.asarray([0.2, 1.1], dtype=jnp.float64),
            make_radial_quadrature_spec(),
            make_final_state_spec(),
        )
        expected_orbitals: Array = jnp.asarray(
            [[0.0 + 0.0j, 0.0 + 1.0j], [0.6 + 0.0j, -0.8 + 0.0j]]
        )
        expected: Array = jnp.broadcast_to(expected_orbitals, (2, 2, 2))
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
        zeta: Array = jnp.asarray([[1.0]], dtype=jnp.float64)
        spec: RadialSpec = make_radial_spec(
            basis,
            (0,),
            mode="slater",
            zeta_shell=zeta,
        )
        momenta: Array = jnp.asarray(
            [0.1, 0.7, 1.4],
            dtype=jnp.float64,
        )
        values: Array = radial_bvals(
            spec,
            momenta,
            make_radial_quadrature_spec(),
            make_final_state_spec(),
        )
        normalization: Array = (2.0 * zeta[0, 0]) ** 1.5 / jnp.sqrt(2.0)
        expected_upper: Array = (
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
        momentum: Array = jnp.asarray(0.9, dtype=jnp.float64)

        def with_zeta(value: Array) -> Array:
            candidate: RadialSpec = eqx.tree_at(
                lambda item: item.zeta_shell,
                base,
                base.zeta_shell.at[0, 0].set(value),
            )
            return jnp.real(
                radial_bvals(candidate, momentum, quadrature, final_state)[
                    0, 0
                ]
            )

        def with_scale(value: Array) -> Array:
            candidate: RadialSpec = eqx.tree_at(
                lambda item: item.coefficients_shell,
                base,
                value * base.coefficients_shell,
            )
            return jnp.real(
                radial_bvals(candidate, momentum, quadrature, final_state)[
                    0, 0
                ]
            )

        zeta_gradient: Array = jax.grad(with_zeta)(base.zeta_shell[0, 0])
        epsilon: Array = jnp.asarray(2.0e-5)
        zeta_fd: Array = (
            with_zeta(base.zeta_shell[0, 0] + epsilon)
            - with_zeta(base.zeta_shell[0, 0] - epsilon)
        ) / (2.0 * epsilon)
        chex.assert_trees_all_close(
            zeta_gradient, zeta_fd, rtol=2.0e-6, atol=2.0e-8
        )
        assert bool(jnp.abs(zeta_gradient) > 1.0e-6)
        chex.assert_trees_all_close(
            jax.grad(with_scale)(jnp.asarray(1.0)),
            0.0,
            atol=2.0e-12,
        )

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
        momenta: Array = jnp.asarray(
            [0.0, 0.4, 1.7, 4.0],
            dtype=jnp.float64,
        )

        def evaluated(
            exponent: Array,
            quadrature: RadialQuadratureSpec,
        ) -> Array:
            candidate: RadialSpec = eqx.tree_at(
                lambda item: item.zeta_shell,
                base,
                base.zeta_shell.at[0, 0].set(exponent),
            )
            return radial_bvals(
                candidate,
                momenta,
                quadrature,
                final_state,
            )

        exponent: Array = base.zeta_shell[0, 0]
        production_values: Array = evaluated(exponent, production)
        reference_values: Array = evaluated(exponent, reference)
        value_scale: Array = jnp.maximum(
            1.0,
            jnp.max(jnp.abs(reference_values)),
        )
        assert bool(
            jnp.max(jnp.abs(production_values - reference_values))
            <= production.value_rtol * value_scale
        )

        def scalar_objective(
            exponent: Array,
            quadrature: RadialQuadratureSpec,
        ) -> Array:
            values: Array = evaluated(exponent, quadrature)
            return jnp.sum(jnp.real(values) ** 2 + jnp.imag(values) ** 2)

        production_gradient: Array = jax.grad(scalar_objective, argnums=0)(
            exponent,
            production,
        )
        reference_gradient: Array = jax.grad(scalar_objective, argnums=0)(
            exponent,
            reference,
        )
        gradient_scale: Array = jnp.maximum(
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
        radial_grid: Array = jnp.linspace(
            0.0,
            20.0,
            1001,
            dtype=jnp.float64,
        )
        raw_values: Array = jnp.exp(-radial_grid).at[-1].set(0.0)
        spec: RadialSpec = make_radial_spec(
            basis,
            (0,),
            mode="grid",
            r_grid=radial_grid,
            grid_values_shell=raw_values[None, :],
        )
        momentum: Array = jnp.asarray(0.8, dtype=jnp.float64)
        actual: Array = radial_bvals(
            spec,
            momentum,
            make_radial_quadrature_spec(),
            make_final_state_spec(),
        )[0, 1]
        stored: Array | None = spec.grid_values_shell
        assert stored is not None
        expected: Array = radial_integral_simpson(
            momentum,
            radial_grid,
            stored[0],
            1,
        )
        chex.assert_trees_all_close(
            actual, expected, rtol=1.0e-12, atol=1.0e-14
        )

    def test_domain_and_rejected_accelerator_guards(self) -> None:
        """Reject out-of-profile momentum and uncertified acceleration.

        The assertions pin the documented numerical contract.

        Notes
        -----
        Direct Coulomb evaluation is a distinct complex path. The failed
        frozen convergence gate makes Hermite acceleration unavailable.
        """
        basis: OrbitalBasis = make_orbital_basis(
            atom_indices=(0,),
            n=(1,),
            l=(0,),
            m=(0,),
        )
        spec: RadialSpec = make_radial_spec(basis, (0,), mode="slater")
        quadrature: RadialQuadratureSpec = make_radial_quadrature_spec()
        with pytest.raises(Exception, match="certified quadrature domain"):
            radial_bvals(
                spec,
                jnp.asarray(4.01),
                quadrature,
                make_final_state_spec(),
            )
        coulomb_values: Array = radial_bvals(
            spec,
            jnp.asarray(1.0),
            quadrature,
            make_final_state_spec(mode="coulomb", effective_charge=1.0),
        )
        chex.assert_tree_all_finite(coulomb_values)
        assert bool(jnp.any(jnp.abs(jnp.imag(coulomb_values)) > 1.0e-8))
        with pytest.raises(ValueError, match="failed the frozen G13"):
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
        nodes: Array
        weights: Array
        nodes, weights = gauss_legendre_nodes(8, 3.0)
        with pytest.raises(ValueError, match="non-negative"):
            radial_integral(
                jnp.asarray(1.0),
                nodes,
                weights,
                jnp.ones_like(nodes),
                -1,
            )
        with pytest.raises(Exception, match="weights_bohr"):
            radial_integral(
                jnp.asarray(1.0),
                nodes,
                weights[:-1],
                jnp.ones_like(nodes),
                0,
            )
