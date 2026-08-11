"""Validate free-electron photoemission kinematics.

Extended Summary
----------------
The tests cover thresholds, momentum values, complex out-of-plane roots,
emission angles, detector maps, JAX transforms, and certified gradients.
"""

import json
from pathlib import Path

import chex
import jax
import jax.numpy as jnp
import pytest
from beartype.typing import Any, Callable, Dict, List, Tuple
from hypothesis import given, settings
from hypothesis import strategies as st
from jaxtyping import Array, Bool, Complex128, Float64

from diffpes.simul import (
    detector_angles_to_kpar,
    emission_angles,
    final_state_k_inv_ang,
    kinetic_energy_ev,
    kpar_to_detector_angles,
    kz_from_inner_potential,
    kz_from_inner_potential_at_fermi,
)
from diffpes.types import (
    K_PREFACTOR_INV_ANG_SQRT_EV,
    TWO_ME_OVER_HBAR_SQ_INV_EV_ANG2,
)
from tests._gradients import (
    assert_grad_matches_fd,
    assert_gradients_match_finite_differences,
)


class TestKineticEnergyEv(chex.TestCase):
    """Validate :func:`~diffpes.simul.kinetic_energy_ev`.

    The tests cover the signed energy convention, threshold mask, JIT, and
    derivatives on each side of the threshold.

    :see: :func:`~diffpes.simul.kinetic_energy_ev`
    """

    def test_signed_energy_and_validity_under_jit(self) -> None:
        """Match signed energy conservation and preserve forbidden values.

        Positive Fermi-relative energy increases kinetic energy. The last
        value is below threshold and remains negative with a false mask.

        Notes
        -----
        Use 21.2 eV photons and a 4.3 eV work function. Compare four signed
        Fermi-relative electron energies with the closed-form values.
        """
        omega_rel_fermi: Float64[Array, " 4"] = jnp.array(
            [-0.4, 0.0, 0.7, -40.0]
        )
        expected: Float64[Array, " 4"] = jnp.array([16.5, 16.9, 17.6, -23.1])
        actual: Float64[Array, " 4"]
        valid: Bool[Array, " 4"]
        actual, valid = jax.jit(kinetic_energy_ev)(
            21.2,
            4.3,
            omega_rel_fermi,
        )
        chex.assert_shape(actual, (4,))
        self.assertEqual(actual.dtype, jnp.dtype("float64"))
        chex.assert_trees_all_close(
            actual,
            expected,
            rtol=0.0,
            atol=1e-14,
        )
        chex.assert_trees_all_equal(
            valid,
            jnp.array([True, True, True, False]),
        )

    def test_raw_energy_has_unit_gradient_on_both_sides(self) -> None:
        """Verify exact raw-energy gradients above and below threshold.

        Photon energy has unit sensitivity on both sides because the function
        does not floor or otherwise fabricate forbidden energies.

        Notes
        -----
        Differentiate one allowed point and one rejected point, staying away
        from the nondifferentiable Boolean threshold itself.
        """
        allowed_gradient: Float64[Array, ""] = jax.grad(
            lambda photon: kinetic_energy_ev(
                photon,
                4.3,
                jnp.array(0.7),
            )[0]
        )(jnp.array(21.2))
        forbidden_gradient: Float64[Array, ""] = jax.grad(
            lambda photon: kinetic_energy_ev(
                photon,
                4.3,
                jnp.array(-20.0),
            )[0]
        )(jnp.array(10.0))
        chex.assert_trees_all_close(
            allowed_gradient,
            jnp.array(1.0),
            rtol=0.0,
            atol=0.0,
        )
        chex.assert_trees_all_close(
            forbidden_gradient,
            jnp.array(1.0),
            rtol=0.0,
            atol=0.0,
        )
        chex.assert_tree_all_finite((allowed_gradient, forbidden_gradient))


class TestFinalStateKInvAng(chex.TestCase):
    """Validate :func:`~diffpes.simul.final_state_k_inv_ang`.

    The tests compare the free-electron formula with closed-form values. They
    also verify the forbidden-emission sentinel and momentum gradient.

    :see: :func:`~diffpes.simul.final_state_k_inv_ang`
    """

    def test_values_and_shape_match_free_electron_formula(self) -> None:
        """Match the free-electron momentum formula and preserve shape.

        The expected values use the CODATA-derived prefactor. The
        function must return one float64 momentum for each energy.

        Notes
        -----
        Evaluate energies of 1, 16, and 100 eV under JIT. Compare at
        ``rtol=1e-14``.
        """
        energies: Float64[Array, " 3"] = jnp.array([1.0, 16.0, 100.0])
        expected: Float64[Array, " 3"] = (
            K_PREFACTOR_INV_ANG_SQRT_EV * jnp.sqrt(energies)
        )
        actual: Float64[Array, " 3"]
        valid: Bool[Array, " 3"]
        actual, valid = jax.jit(final_state_k_inv_ang)(energies)
        chex.assert_shape(actual, (3,))
        self.assertEqual(actual.dtype, jnp.dtype("float64"))
        chex.assert_trees_all_close(
            actual,
            expected,
            rtol=1e-14,
            atol=1e-14,
        )
        chex.assert_trees_all_equal(valid, jnp.ones(3, dtype=jnp.bool_))

    def test_gradient_matches_formula_and_forbidden_sentinel(self) -> None:
        """Match the analytic derivative and forbidden-emission sentinel.

        The derivative above the floor equals ``C/(2*sqrt(E))``. A negative
        direct input returns zero with a false validity mask.

        Notes
        -----
        Apply the shared finite-difference harness at 24 eV. Differentiate a
        negative input away from the threshold.
        """
        energy: Float64[Array, ""] = jnp.array(24.0)

        def momentum(candidate: Float64[Array, ""]) -> Float64[Array, ""]:
            """Return only the differentiable momentum value."""
            value: Float64[Array, ""] = final_state_k_inv_ang(candidate)[0]
            return value

        assert_grad_matches_fd(momentum, energy)
        actual_gradient: Float64[Array, ""] = jax.grad(
            final_state_k_inv_ang,
            has_aux=True,
        )(energy)[0]
        expected_gradient: Float64[Array, ""] = K_PREFACTOR_INV_ANG_SQRT_EV / (
            2.0 * jnp.sqrt(energy)
        )
        forbidden_momentum: Float64[Array, ""]
        forbidden_valid: Bool[Array, ""]
        forbidden_momentum, forbidden_valid = final_state_k_inv_ang(
            jnp.array(-1.0)
        )
        forbidden_gradient: Float64[Array, ""] = jax.grad(momentum)(
            jnp.array(-1.0)
        )
        chex.assert_trees_all_close(
            actual_gradient,
            expected_gradient,
            rtol=1e-14,
            atol=1e-14,
        )
        chex.assert_trees_all_close(
            forbidden_momentum,
            jnp.array(0.0),
            rtol=0.0,
            atol=0.0,
        )
        chex.assert_trees_all_close(
            forbidden_gradient,
            jnp.array(0.0),
            rtol=0.0,
            atol=0.0,
        )
        self.assertFalse(bool(forbidden_valid))
        chex.assert_tree_all_finite((actual_gradient, forbidden_gradient))


class TestKzFromInnerPotential(chex.TestCase):
    """Validate :func:`diffpes.simul.kz_from_inner_potential`.

    The tests cover threshold validity, the vacuum aperture, exact signed
    energy dependence, the at-Fermi error, and smooth-domain derivatives.
    """

    def test_exact_threshold_and_aperture_boundary_are_invalid(self) -> None:
        """Reject equality at both open physical boundaries.

        The function forbids exactly zero kinetic energy. At positive kinetic
        energy,
        a parallel momentum equal to the vacuum final-state magnitude is also
        outside the open emission aperture.

        Notes
        -----
        Check raw kinetic energy, final-state momentum, and both ``kz``
        boundaries under eager and compiled execution. Require exact zero
        sentinels and false masks.
        """
        raw_energy: Float64[Array, ""]
        energy_valid: Bool[Array, ""]
        raw_energy, energy_valid = kinetic_energy_ev(
            4.5,
            4.5,
            jnp.array(0.0),
        )
        final_momentum: Float64[Array, ""]
        momentum_valid: Bool[Array, ""]
        final_momentum, momentum_valid = final_state_k_inv_ang(raw_energy)
        chex.assert_trees_all_equal(raw_energy, jnp.array(0.0))
        chex.assert_trees_all_equal(final_momentum, jnp.array(0.0))
        self.assertFalse(bool(energy_valid))
        self.assertFalse(bool(momentum_valid))

        operation: Callable[
            ..., Tuple[Float64[Array, "..."], Float64[Array, "..."]]
        ]
        compiled: bool
        for compiled in (False, True):
            operation = kz_from_inner_potential
            if compiled:
                operation = jax.jit(operation)
            threshold_value: Complex128[Array, ""]
            threshold_valid: Bool[Array, ""]
            threshold_value, threshold_valid = operation(
                4.5,
                4.5,
                10.0,
                jnp.array(0.0),
                jnp.array(0.0),
            )
            aperture_value: Complex128[Array, ""]
            aperture_valid: Bool[Array, ""]
            aperture_value, aperture_valid = operation(
                5.5,
                4.5,
                10.0,
                jnp.array(0.0),
                jnp.asarray(K_PREFACTOR_INV_ANG_SQRT_EV),
            )
            with self.subTest(compiled=compiled):
                chex.assert_trees_all_equal(
                    threshold_value,
                    jnp.array(0.0 + 0.0j),
                )
                chex.assert_trees_all_equal(
                    aperture_value,
                    jnp.array(0.0 + 0.0j),
                )
                self.assertFalse(bool(threshold_valid))
                self.assertFalse(bool(aperture_valid))

    def test_forbidden_emission_returns_zero_and_false_mask(self) -> None:
        """Do not let a positive inner potential fabricate photoemission.

        The surface kinetic energy is negative although the inner-potential
        radicand is positive. The returned sentinel must be exactly zero and
        the propagation mask false under eager and JIT execution.

        Notes
        -----
        Evaluate one forbidden point twice and differentiate its real sentinel
        with respect to signed energy.
        """
        eager: Tuple[Complex128[Array, ""], Bool[Array, ""]] = (
            kz_from_inner_potential(
                4.0,
                4.5,
                20.0,
                jnp.array(-1.0),
                jnp.array(0.0),
            )
        )
        compiled: Tuple[Complex128[Array, ""], Bool[Array, ""]] = jax.jit(
            kz_from_inner_potential
        )(
            4.0,
            4.5,
            20.0,
            jnp.array(-1.0),
            jnp.array(0.0),
        )
        value: Complex128[Array, ""]
        valid: Bool[Array, ""]
        for value, valid in (eager, compiled):
            chex.assert_trees_all_equal(value, jnp.array(0.0 + 0.0j))
            self.assertFalse(bool(valid))
        forbidden_gradient: Float64[Array, ""] = jax.grad(
            lambda omega: jnp.real(
                kz_from_inner_potential(
                    4.0,
                    4.5,
                    20.0,
                    omega,
                    jnp.array(0.0),
                )[0]
            )
        )(jnp.array(-1.0))
        chex.assert_trees_all_equal(forbidden_gradient, jnp.array(0.0))
        chex.assert_tree_all_finite(forbidden_gradient)

    def test_super_aperture_returns_zero_and_false_mask(self) -> None:
        """Reject vacuum momenta that a positive inner potential could hide.

        A 1 eV photoelectron has magnitude about 0.512 inverse Angstrom.
        Therefore, ``k_parallel = 1`` exceeds the vacuum aperture. A positive
        inner potential still makes the internal radicand positive.

        Notes
        -----
        Evaluate eager and compiled paths, then differentiate the rejected
        sentinel away from the aperture boundary.
        """
        operation: Callable[
            ..., Tuple[Float64[Array, "..."], Float64[Array, "..."]]
        ]
        compiled: bool
        for compiled in (False, True):
            operation = kz_from_inner_potential
            if compiled:
                operation = jax.jit(operation)
            value: Complex128[Array, ""]
            valid: Bool[Array, ""]
            value, valid = operation(
                5.5,
                4.5,
                10.0,
                jnp.array(0.0),
                jnp.array(1.0),
            )
            with self.subTest(compiled=compiled):
                chex.assert_trees_all_equal(value, jnp.array(0.0 + 0.0j))
                self.assertFalse(bool(valid))

        def rejected_value(
            k_parallel: Float64[Array, ""],
        ) -> Float64[Array, ""]:
            """Return the zero sentinel outside the aperture."""
            value: Complex128[Array, ""] = kz_from_inner_potential(
                5.5,
                4.5,
                10.0,
                jnp.array(0.0),
                k_parallel,
            )[0]
            returned: Float64[Array, ""] = jnp.real(value)
            return returned

        gradient: Float64[Array, ""] = jax.grad(rejected_value)(jnp.array(1.0))
        chex.assert_trees_all_equal(gradient, jnp.array(0.0))
        chex.assert_tree_all_finite(gradient)

    def test_exact_energy_axis_and_at_fermi_error(self) -> None:
        """Verify the finite-window error of the at-Fermi approximation.

        The exact path follows the closed form at five signed energies. The
        named approximation is exact only at zero; its relative error is
        nonzero elsewhere and remains below two percent on this window.

        Notes
        -----
        Replay the committed five-point artifact and compare both paths with
        the independent closed form.
        """
        reference_path: Path = (
            Path(__file__).resolve().parents[2]
            / "data"
            / "kspace"
            / "kz_energy_dependence_reference.json"
        )
        reference: Dict[str, Any] = json.loads(reference_path.read_text())
        inputs: Dict[str, Any] = reference["inputs"]
        omega: Float64[Array, " 5"] = jnp.asarray(inputs["omega_rel_fermi_ev"])
        k_parallel: Float64[Array, " 5"] = jnp.full(
            (5,),
            inputs["k_parallel_inv_ang"],
        )
        exact: Complex128[Array, " 5"]
        propagating: Bool[Array, " 5"]
        exact, propagating = kz_from_inner_potential(
            inputs["photon_energy_ev"],
            inputs["work_function_ev"],
            inputs["inner_potential_ev"],
            omega,
            k_parallel,
        )
        approximate: Complex128[Array, " 5"] = (
            kz_from_inner_potential_at_fermi(
                inputs["photon_energy_ev"],
                inputs["work_function_ev"],
                inputs["inner_potential_ev"],
                k_parallel,
            )[0]
        )
        expected: Complex128[Array, " 5"] = jnp.sqrt(
            (
                TWO_ME_OVER_HBAR_SQ_INV_EV_ANG2
                * (
                    inputs["photon_energy_ev"]
                    - inputs["work_function_ev"]
                    + omega
                    + inputs["inner_potential_ev"]
                )
                - k_parallel**2
            ).astype(jnp.complex128)
        )
        relative_error: Float64[Array, " 5"] = jnp.abs(
            approximate - exact
        ) / jnp.abs(exact)
        chex.assert_trees_all_close(exact, expected, rtol=1e-14, atol=1e-14)
        chex.assert_trees_all_close(
            jnp.real(exact),
            jnp.asarray(reference["exact_kz_inv_ang"]),
            rtol=1e-14,
            atol=1e-14,
        )
        chex.assert_trees_all_close(
            relative_error,
            jnp.asarray(reference["at_fermi_relative_error"]),
            rtol=1e-14,
            atol=1e-14,
        )
        self.assertEqual(
            reference["requirement"], "kz-energy-dependence-reference"
        )
        chex.assert_trees_all_equal(propagating, jnp.ones(5, dtype=jnp.bool_))
        self.assertEqual(float(relative_error[2]), 0.0)
        self.assertGreater(float(relative_error[0]), 0.0)
        self.assertGreater(float(relative_error[-1]), 0.0)
        self.assertLess(float(jnp.max(relative_error)), 0.02)

    def test_omega_derivative_matches_closed_form_and_fd(self) -> None:
        """Match the exact energy derivative away from both branch points.

        The energy derivative equals the inner-potential derivative because
        both quantities enter the radicand additively.

        Notes
        -----
        Compare autodiff with central differences and the analytic derivative
        at one smooth propagating point.
        """
        omega: Float64[Array, ""] = jnp.array(-0.8)

        def real_kz(candidate: Float64[Array, ""]) -> Float64[Array, ""]:
            """Return one propagating real root."""
            value: Complex128[Array, ""] = kz_from_inner_potential(
                50.0,
                4.5,
                12.0,
                candidate,
                jnp.array(0.7),
            )[0]
            returned: Float64[Array, ""] = jnp.real(value)
            return returned

        assert_grad_matches_fd(real_kz, omega)
        kz_value: Float64[Array, ""] = real_kz(omega)
        actual: Float64[Array, ""] = jax.grad(real_kz)(omega)
        expected: Float64[Array, ""] = TWO_ME_OVER_HBAR_SQ_INV_EV_ANG2 / (
            2.0 * kz_value
        )
        chex.assert_trees_all_close(actual, expected, rtol=1e-12, atol=1e-12)

    def test_work_function_and_energy_jacobians_are_exact_opposites(
        self,
    ) -> None:
        """Verify the exact work-function and energy-reference gauge.

        Kinematics depends on work function and Fermi-relative energy only
        through their signed difference. A photon scan cannot lift this gauge.

        Notes
        -----
        Differentiate one smooth real root with respect to both variables and
        compare their Jacobian columns exactly.
        """

        def real_kz(
            work_function: Float64[Array, ""],
            omega: Float64[Array, ""],
        ) -> Float64[Array, ""]:
            """Return one propagating real root."""
            value: Complex128[Array, ""] = kz_from_inner_potential(
                50.0,
                work_function,
                12.0,
                omega,
                jnp.array(0.7),
            )[0]
            returned: Float64[Array, ""] = jnp.real(value)
            return returned

        work_gradient: Float64[Array, ""]
        omega_gradient: Float64[Array, ""]
        work_gradient, omega_gradient = jax.grad(
            real_kz,
            argnums=(0, 1),
        )(jnp.array(4.5), jnp.array(-0.8))
        chex.assert_trees_all_equal(work_gradient, -omega_gradient)
        self.assertGreater(float(jnp.abs(work_gradient)), 1e-12)


class TestKzFromInnerPotentialAtFermi(chex.TestCase):
    """Validate the named at-Fermi compatibility kinematics.

    These parity tests cover Damascelli values, evanescent channels, analytic
    and finite-difference gradients, vmap consistency, and a large JIT raster.

    :see: :func:`~diffpes.simul.kz_from_inner_potential_at_fermi`
    """

    def test_matches_damascelli_grid(self) -> None:
        """Match the Damascelli closed form on a photon-energy grid.

        The detector map supplies the parallel momentum. The expected
        out-of-plane momentum uses the independent angle formula.

        Notes
        -----
        Use horizontal-slit angles ``(theta, 0)``. Compare the composed result
        with ``C*sqrt(Ekin*cos(theta)^2+V0)`` at ``rtol=1e-10``.
        """
        cases: Tuple[Tuple[float, float, float, float], ...] = (
            (21.2, 4.0, 8.0, 0.0),
            (50.0, 4.5, 12.0, 0.17),
            (100.0, 4.0, 15.0, -0.31),
            (150.0, 4.5, 8.0, 0.42),
        )
        photon_energy: float
        work_function: float
        inner_potential: float
        theta_value: float
        for (
            photon_energy,
            work_function,
            inner_potential,
            theta_value,
        ) in cases:
            with self.subTest(photon_energy=photon_energy, theta=theta_value):
                theta: Float64[Array, ""] = jnp.asarray(theta_value)
                surface_energy: Float64[Array, ""] = jnp.asarray(
                    photon_energy - work_function
                )
                k_parallel_vector: Float64[Array, "2"] = (
                    detector_angles_to_kpar(
                        theta,
                        jnp.array(0.0),
                        surface_energy,
                        "H",
                    )
                )
                k_parallel: Float64[Array, ""] = jnp.linalg.norm(
                    k_parallel_vector
                )
                kz_value: Complex128[Array, ""]
                propagating: Bool[Array, ""]
                kz_value, propagating = kz_from_inner_potential_at_fermi(
                    photon_energy,
                    work_function,
                    inner_potential,
                    k_parallel,
                )
                expected: Float64[Array, ""] = (
                    K_PREFACTOR_INV_ANG_SQRT_EV
                    * jnp.sqrt(
                        surface_energy * jnp.cos(theta) ** 2 + inner_potential
                    )
                )
                chex.assert_trees_all_close(
                    jnp.real(kz_value),
                    expected,
                    rtol=1e-10,
                    atol=1e-12,
                )
                chex.assert_trees_all_close(
                    jnp.imag(kz_value),
                    jnp.array(0.0),
                    rtol=0.0,
                    atol=0.0,
                )
                self.assertTrue(bool(propagating))

    def test_matches_pinned_chinook_reference(self) -> None:
        """Match the pinned Chinook kinematics table and its constant mapping.

        The injected Chinook constants must reproduce all 168 source values.
        The production constants must reproduce the separately recorded
        accuracy-improved values and retain the measured constant delta.

        Notes
        -----
        Read the committed offline artifact for the kz kinematics reference.
        Vmap the
        production function across its rows and compare both formulations at
        the artifact tolerance.
        """
        reference_path: Path = (
            Path(__file__).resolve().parents[2]
            / "data"
            / "kspace"
            / "kz_kpt_reference.json"
        )
        document: Dict[str, Any] = json.loads(reference_path.read_text())
        records: List[Dict[str, float]] = document["records"]
        self.assertEqual(document["requirement"], "kz-kinematics-reference")
        self.assertEqual(
            document["metadata"]["chinook_commit"],
            "24913de8cc5b8c162f7c1b4acc64bd1b54dd548b",
        )
        self.assertEqual(len(records), 168)

        photon_energies: Float64[Array, " 168"] = jnp.asarray(
            [record["photon_energy_ev"] for record in records]
        )
        work_functions: Float64[Array, " 168"] = jnp.asarray(
            [record["work_function_ev"] for record in records]
        )
        inner_potentials: Float64[Array, " 168"] = jnp.asarray(
            [record["inner_potential_ev"] for record in records]
        )
        k_parallel: Float64[Array, " 168"] = jnp.asarray(
            [record["k_parallel_inv_ang"] for record in records]
        )
        chinook_values: Float64[Array, " 168"] = jnp.asarray(
            [record["kz_chinook_inv_ang"] for record in records]
        )
        recorded_production: Float64[Array, " 168"] = jnp.asarray(
            [record["kz_production_constants_inv_ang"] for record in records]
        )
        chinook_prefactor: Float64[Array, ""] = jnp.asarray(
            document["chinook_constants"]["momentum_prefactor_inv_ang_sqrt_ev"]
        )
        injected_values: Float64[Array, " 168"] = jnp.sqrt(
            chinook_prefactor**2
            * (photon_energies - work_functions + inner_potentials)
            - k_parallel**2
        )

        def production_value(
            photon_energy: Float64[Array, ""],
            work_function: Float64[Array, ""],
            inner_potential: Float64[Array, ""],
            parallel_momentum: Float64[Array, ""],
        ) -> Float64[Array, ""]:
            """Return one real production out-of-plane momentum."""
            value: Complex128[Array, ""] = kz_from_inner_potential_at_fermi(
                photon_energy,
                work_function,
                inner_potential,
                parallel_momentum,
            )[0]
            returned: Float64[Array, ""] = jnp.real(value)
            return returned

        production_values: Float64[Array, " 168"] = jax.vmap(production_value)(
            photon_energies,
            work_functions,
            inner_potentials,
            k_parallel,
        )
        tolerance: float = document["rtol"]
        chex.assert_trees_all_close(
            injected_values,
            chinook_values,
            rtol=tolerance,
            atol=1e-12,
        )
        chex.assert_trees_all_close(
            production_values,
            recorded_production,
            rtol=1e-13,
            atol=1e-13,
        )
        maximum_delta: float = document["maximum_production_relative_delta"]
        self.assertGreater(maximum_delta, 0.0)
        self.assertLess(maximum_delta, 2e-5)

    def test_preserves_evanescent_channels(self) -> None:
        """Verify principal complex roots and the propagation mask.

        Small parallel momentum gives a positive real root. A negative inner
        potential gives an imaginary internal root inside the vacuum aperture.

        Notes
        -----
        Evaluate parallel momenta 0 and 1.5 in 1/Angstrom at a -10 eV inner
        potential. Compare both values with the direct complex square root.
        """
        k_parallel: Float64[Array, " 2"] = jnp.array([0.0, 1.5])
        kz_values: Complex128[Array, " 2"]
        propagating: Bool[Array, " 2"]
        kz_values, propagating = jax.jit(kz_from_inner_potential_at_fermi)(
            21.2,
            4.5,
            -10.0,
            k_parallel,
        )
        surface_energy: float = 21.2 - 4.5
        radicands: Float64[Array, " 2"] = (
            TWO_ME_OVER_HBAR_SQ_INV_EV_ANG2 * (surface_energy - 10.0)
            - k_parallel**2
        )
        expected: Complex128[Array, " 2"] = jnp.sqrt(
            radicands.astype(jnp.complex128)
        )
        chex.assert_trees_all_close(
            kz_values,
            expected,
            rtol=1e-14,
            atol=1e-14,
        )
        chex.assert_trees_all_equal(
            propagating,
            jnp.array([True, False]),
        )
        self.assertGreater(float(jnp.imag(kz_values[1])), 0.0)

    def test_gradients_match_fd_at_twelve_registered_points(self) -> None:
        """Match three parameter gradients at twelve propagating points.

        The points span ultraviolet through soft-X-ray photon energies. Each
        parameter leaf retains a nonzero gradient norm.

        Notes
        -----
        Vmap the scalar kinematics over twelve tuples. The shared harness
        checks both autodiff modes and central finite differences.
        """
        photon_energies: Float64[Array, " 12"] = jnp.array(
            [
                21.2,
                25.0,
                35.0,
                50.0,
                65.0,
                80.0,
                100.0,
                120.0,
                150.0,
                200.0,
                300.0,
                500.0,
            ]
        )
        work_functions: Float64[Array, " 12"] = jnp.linspace(4.0, 4.5, 12)
        inner_potentials: Float64[Array, " 12"] = jnp.linspace(8.0, 15.0, 12)
        k_parallel: Float64[Array, " 12"] = jnp.linspace(0.1, 1.2, 12)

        def loss(
            parameters: Tuple[
                Float64[Array, " 12"],
                Float64[Array, " 12"],
                Float64[Array, " 12"],
            ],
        ) -> Float64[Array, ""]:
            photon_values: Float64[Array, " 12"]
            work_values: Float64[Array, " 12"]
            inner_values: Float64[Array, " 12"]
            photon_values, work_values, inner_values = parameters

            def one_kz(
                photon: Float64[Array, ""],
                work: Float64[Array, ""],
                inner: Float64[Array, ""],
                parallel: Float64[Array, ""],
            ) -> Float64[Array, ""]:
                value: Complex128[Array, ""] = (
                    kz_from_inner_potential_at_fermi(
                        photon,
                        work,
                        inner,
                        parallel,
                    )[0]
                )
                real_value: Float64[Array, ""] = jnp.real(value)
                return real_value

            values: Float64[Array, " 12"] = jax.vmap(one_kz)(
                photon_values,
                work_values,
                inner_values,
                k_parallel,
            )
            total: Float64[Array, ""] = jnp.sum(values)
            return total

        assert_gradients_match_finite_differences(
            loss,
            (photon_energies, work_functions, inner_potentials),
            regime="smooth",
        )

    def test_inner_potential_gradient_matches_closed_form(self) -> None:
        """Match the inner-potential derivative with its analytic value.

        For a real channel, the derivative equals
        ``TWO_ME_OVER_HBAR_SQ_INV_EV_ANG2 / (2*kz)``.

        Notes
        -----
        Differentiate at 50 eV photon energy and 0.7 1/Angstrom. Compare at
        ``rtol=1e-10`` and require nonzero sensitivity.
        """
        inner_potential: Float64[Array, ""] = jnp.array(12.0)
        k_parallel: Float64[Array, ""] = jnp.array(0.7)

        def real_kz(candidate: Float64[Array, ""]) -> Float64[Array, ""]:
            value: Complex128[Array, ""] = kz_from_inner_potential_at_fermi(
                50.0,
                4.5,
                candidate,
                k_parallel,
            )[0]
            real_value: Float64[Array, ""] = jnp.real(value)
            return real_value

        kz_value: Float64[Array, ""] = real_kz(inner_potential)
        actual: Float64[Array, ""] = jax.grad(real_kz)(inner_potential)
        expected: Float64[Array, ""] = TWO_ME_OVER_HBAR_SQ_INV_EV_ANG2 / (
            2.0 * kz_value
        )
        chex.assert_trees_all_close(
            actual,
            expected,
            rtol=1e-10,
            atol=1e-12,
        )
        self.assertGreater(float(jnp.abs(actual)), 1e-12)

    @given(
        photon_energy=st.floats(
            min_value=20.0,
            max_value=200.0,
            allow_nan=False,
            allow_infinity=False,
        ),
        lower_potential=st.floats(
            min_value=0.0,
            max_value=15.0,
            allow_nan=False,
            allow_infinity=False,
        ),
        potential_step=st.floats(
            min_value=1e-3,
            max_value=10.0,
            allow_nan=False,
            allow_infinity=False,
        ),
    )
    @settings(max_examples=20, deadline=None)
    def test_real_kz_increases_with_inner_potential(
        self,
        photon_energy: float,
        lower_potential: float,
        potential_step: float,
    ) -> None:
        """Verify monotonic out-of-plane momentum in the inner potential.

        The free-electron radicand increases linearly with the inner potential.
        Thus, each positive real root must increase.

        Notes
        -----
        Hypothesis generates twenty photon energies and ordered potentials.
        Compare real roots at fixed work function and parallel momentum.
        """
        lower_kz: Complex128[Array, ""] = kz_from_inner_potential_at_fermi(
            photon_energy,
            4.5,
            lower_potential,
            jnp.array(0.4),
        )[0]
        upper_kz: Complex128[Array, ""] = kz_from_inner_potential_at_fermi(
            photon_energy,
            4.5,
            lower_potential + potential_step,
            jnp.array(0.4),
        )[0]
        self.assertGreater(float(jnp.real(upper_kz - lower_kz)), 0.0)

    def test_vmap_gradient_matches_elementwise_gradients(self) -> None:
        """Match vmapped and elementwise photon-energy gradients.

        Vmap must not change the gradient of the out-of-plane momentum. The
        comparison uses four photon energies and relative tolerance 1e-14.

        Notes
        -----
        Differentiate one scalar real root. Apply JIT and vmap to the gradient
        and compare with an eager stack.
        """
        photon_energies: Float64[Array, " 4"] = jnp.array(
            [21.2, 50.0, 100.0, 150.0]
        )

        def real_kz(photon: Float64[Array, ""]) -> Float64[Array, ""]:
            value: Complex128[Array, ""] = kz_from_inner_potential_at_fermi(
                photon,
                4.3,
                12.0,
                jnp.array(0.6),
            )[0]
            real_value: Float64[Array, ""] = jnp.real(value)
            return real_value

        vmapped: Float64[Array, " 4"] = jax.jit(jax.vmap(jax.grad(real_kz)))(
            photon_energies
        )
        elementwise: Float64[Array, " 4"] = jnp.stack(
            tuple(jax.grad(real_kz)(photon) for photon in photon_energies)
        )
        chex.assert_trees_all_close(
            vmapped,
            elementwise,
            rtol=1e-14,
            atol=1e-14,
        )

    @pytest.mark.big_mem
    @pytest.mark.rss_limit_mb(700)
    def test_large_photon_energy_vmap_has_static_shape(self) -> None:
        """Verify a large JIT-vmap raster has the required static shape.

        The raster contains 64 photon energies and 256 squared parallel
        momenta. The operation must not construct a quadratic point matrix.

        Notes
        -----
        JIT one vmap over photon energy. Check the two output shapes and all
        finite complex values.
        """
        photon_energies: Float64[Array, " 64"] = jnp.linspace(20.0, 150.0, 64)
        k_parallel: Float64[Array, " 65536"] = jnp.linspace(
            0.0,
            1.5,
            256 * 256,
        )

        def one_row(
            photon: Float64[Array, ""],
        ) -> Tuple[Complex128[Array, " 65536"], Bool[Array, " 65536"]]:
            row: Tuple[Complex128[Array, " 65536"], Bool[Array, " 65536"]] = (
                kz_from_inner_potential_at_fermi(
                    photon,
                    4.5,
                    12.0,
                    k_parallel,
                )
            )
            return row

        kz_values: Complex128[Array, "64 65536"]
        propagating: Bool[Array, "64 65536"]
        kz_values, propagating = jax.jit(jax.vmap(one_row))(photon_energies)
        chex.assert_shape(kz_values, (64, 256 * 256))
        chex.assert_shape(propagating, (64, 256 * 256))
        chex.assert_tree_all_finite(kz_values)


class TestEmissionAngles(chex.TestCase):
    """Validate :func:`~diffpes.simul.emission_angles`.

    The tests cover Cartesian directions, the normal-emission gauge, batched
    JIT execution, and finite-difference gradients away from the pole.

    :see: :func:`~diffpes.simul.emission_angles`
    """

    def test_cardinal_directions_and_normal_emission(self) -> None:
        """Match cardinal angles and the normal-emission gauge convention.

        Positive x has polar angle pi/2 and zero azimuth. Positive y has
        azimuth pi/2. Positive z selects two zero angles.

        Notes
        -----
        Pass all three directions as one batch through JIT. Compare each angle
        at absolute tolerance ``1e-14``.
        """
        vectors: Float64[Array, "3 3"] = jnp.eye(3)
        theta: Float64[Array, " 3"]
        phi: Float64[Array, " 3"]
        theta, phi = jax.jit(emission_angles)(vectors)
        chex.assert_trees_all_close(
            theta,
            jnp.array([jnp.pi / 2.0, jnp.pi / 2.0, 0.0]),
            rtol=0.0,
            atol=1e-14,
        )
        chex.assert_trees_all_close(
            phi,
            jnp.array([0.0, jnp.pi / 2.0, 0.0]),
            rtol=0.0,
            atol=1e-14,
        )

    def test_generic_gradients_match_finite_differences(self) -> None:
        """Match angle gradients with central finite differences.

        The generic vector stays outside the punctured normal-emission
        neighborhood. A weighted angle sum depends on all vector components.

        Notes
        -----
        Use ``[0.7, -0.4, 1.2]`` in 1/Angstrom. Apply the shared gradient
        harness in both autodiff modes.
        """
        momentum: Float64[Array, "3"] = jnp.array([0.7, -0.4, 1.2])

        def loss(candidate: Float64[Array, "3"]) -> Float64[Array, ""]:
            theta: Float64[Array, ""]
            phi: Float64[Array, ""]
            theta, phi = emission_angles(candidate)
            value: Float64[Array, ""] = theta + 0.37 * phi
            return value

        assert_gradients_match_finite_differences(
            loss, momentum, regime="smooth"
        )

    def test_normal_emission_selects_zero_angle_gradients(self) -> None:
        """Verify finite zero angle gradients at normal emission.

        Azimuth has no physical value at the pole. The safe coordinate
        primitives assign no derivative to either angle there.

        Notes
        -----
        Compute the Jacobian of both angles at the positive z direction.
        Compare it with a zero matrix exactly.
        """
        momentum: Float64[Array, "3"] = jnp.array([0.0, 0.0, 2.0])

        def both_angles(
            candidate: Float64[Array, "3"],
        ) -> Float64[Array, "2"]:
            theta: Float64[Array, ""]
            phi: Float64[Array, ""]
            theta, phi = emission_angles(candidate)
            angles: Float64[Array, "2"] = jnp.stack((theta, phi))
            return angles

        jacobian: Float64[Array, "2 3"] = jax.jacfwd(both_angles)(momentum)
        chex.assert_trees_all_close(
            jacobian,
            jnp.zeros((2, 3)),
            rtol=0.0,
            atol=0.0,
        )
        chex.assert_tree_all_finite(jacobian)


class TestDetectorAnglesToKpar(chex.TestCase):
    """Validate :func:`~diffpes.simul.detector_angles_to_kpar`.

    The tests cover both slit conventions, the Rodrigues frame, JIT, and
    gradients in both angles and kinetic energy.

    :see: :func:`~diffpes.simul.detector_angles_to_kpar`
    """

    def test_matches_closed_form_rotation(self) -> None:
        """Match the closed-form detector rotation for both slits.

        The expected components follow the registered matrix products. The test
        also verifies static-slit JIT compilation.

        Notes
        -----
        Use angles 0.23 and -0.17 radians with 35 eV kinetic energy. Compare
        at ``rtol=1e-14``.
        """
        slit: str
        for slit in ("H", "V"):
            with self.subTest(slit=slit):
                tx: Float64[Array, ""] = jnp.array(0.23)
                ty: Float64[Array, ""] = jnp.array(-0.17)
                energy: Float64[Array, ""] = jnp.array(35.0)
                momentum: Float64[Array, ""] = final_state_k_inv_ang(energy)[0]
                if slit == "H":
                    expected: Float64[Array, "2"] = momentum * jnp.array(
                        [jnp.sin(tx), -jnp.cos(tx) * jnp.sin(ty)]
                    )
                else:
                    expected = momentum * jnp.array(
                        [jnp.sin(ty), -jnp.sin(tx) * jnp.cos(ty)]
                    )
                actual: Float64[Array, "2"] = jax.jit(
                    detector_angles_to_kpar,
                    static_argnames=("slit",),
                )(tx, ty, energy, slit)
                chex.assert_trees_all_close(
                    actual,
                    expected,
                    rtol=1e-14,
                    atol=1e-14,
                )

    def test_gradients_match_finite_differences(self) -> None:
        """Match angle and energy gradients with finite differences.

        A generic weighted momentum sum depends on all three traced inputs.
        Both slit branches must retain this sensitivity.

        Notes
        -----
        Use nonzero angles and 42 eV kinetic energy. Apply the shared harness
        in forward and reverse autodiff modes.
        """
        tx: Float64[Array, ""] = jnp.array(0.21)
        ty: Float64[Array, ""] = jnp.array(-0.13)
        energy: Float64[Array, ""] = jnp.array(42.0)
        weights: Float64[Array, "2"] = jnp.array([0.7, -1.3])

        slit: str
        for slit in ("H", "V"):
            with self.subTest(slit=slit):

                def loss(
                    parameters: Tuple[
                        Float64[Array, ""],
                        Float64[Array, ""],
                        Float64[Array, ""],
                    ],
                    slit_value: str = slit,
                ) -> Float64[Array, ""]:
                    """Return one weighted forward-map scalar."""
                    candidate_tx: Float64[Array, ""]
                    candidate_ty: Float64[Array, ""]
                    candidate_energy: Float64[Array, ""]
                    candidate_tx, candidate_ty, candidate_energy = parameters
                    k_parallel: Float64[Array, "2"] = detector_angles_to_kpar(
                        candidate_tx,
                        candidate_ty,
                        candidate_energy,
                        slit_value,
                    )
                    value: Float64[Array, ""] = jnp.sum(weights * k_parallel)
                    return value

                assert_gradients_match_finite_differences(
                    loss, (tx, ty, energy), regime="smooth"
                )

    def test_normal_emission_jacobian_matches_frame(self) -> None:
        """Match the Cartesian detector Jacobian at normal emission.

        The direction vector remains smooth when both detector angles vanish.
        Each slit gives a different signed permutation of the tangent axes.

        Notes
        -----
        Differentiate the two parallel components with respect to both angles.
        Compare each Jacobian with its closed form exactly.
        """
        energy: Float64[Array, ""] = jnp.array(30.0)
        momentum: Float64[Array, ""] = final_state_k_inv_ang(energy)[0]
        angles: Float64[Array, "2"] = jnp.zeros(2)
        slit: str
        for slit in ("H", "V"):
            with self.subTest(slit=slit):

                def angle_map(
                    candidate: Float64[Array, "2"],
                    slit_value: str = slit,
                ) -> Float64[Array, "2"]:
                    """Return parallel momentum for one angle pair."""
                    result: Float64[Array, "2"] = detector_angles_to_kpar(
                        candidate[0],
                        candidate[1],
                        energy,
                        slit_value,
                    )
                    return result

                jacobian: Float64[Array, "2 2"] = jax.jacfwd(angle_map)(angles)
                if slit == "H":
                    expected: Float64[Array, "2 2"] = momentum * jnp.array(
                        [[1.0, 0.0], [0.0, -1.0]]
                    )
                else:
                    expected = momentum * jnp.array([[0.0, 1.0], [-1.0, 0.0]])
                chex.assert_trees_all_close(
                    jacobian,
                    expected,
                    rtol=0.0,
                    atol=0.0,
                )

    def test_rejects_unknown_slit(self) -> None:
        """Verify that the detector map rejects an unknown slit.

        The slit controls a static Python branch. Only horizontal and vertical
        orientations define a detector convention.

        Notes
        -----
        Pass the value ``"bad"`` with scalar arrays. Require ``ValueError``
        with the slit validation message.
        """
        with pytest.raises(ValueError, match="slit must be"):
            detector_angles_to_kpar(
                jnp.array(0.1),
                jnp.array(0.2),
                jnp.array(30.0),
                "bad",
            )

    def test_rejects_outside_principal_chart(self) -> None:
        """Reject forbidden energy and angles outside the open chart.

        The detector map accepts only positive kinetic energy and both angles
        strictly between negative and positive pi over two.

        Notes
        -----
        Exercise both slits and all three invalid cases under eager and
        compiled execution.
        """
        invalid_cases: Tuple[Tuple[float, float, float], ...] = (
            (0.0, 0.0, 0.0),
            (jnp.pi / 2.0, 0.0, 30.0),
            (0.0, -jnp.pi / 2.0, 30.0),
        )
        tx: float
        ty: float
        energy: float
        compiled: bool
        slit: str
        for compiled in (False, True):
            operation: Callable[..., Float64[Array, "..."]] = (
                detector_angles_to_kpar
            )
            if compiled:
                operation = jax.jit(operation, static_argnames=("slit",))
            for slit in ("H", "V"):
                for tx, ty, energy in invalid_cases:
                    with (
                        self.subTest(
                            compiled=compiled,
                            slit=slit,
                            tx=tx,
                            ty=ty,
                            energy=energy,
                        ),
                        pytest.raises(RuntimeError, match="requires Ekin > 0"),
                    ):
                        operation(
                            jnp.asarray(tx),
                            jnp.asarray(ty),
                            jnp.asarray(energy),
                            slit,
                        )


class TestKparToDetectorAngles(chex.TestCase):
    """Validate :func:`~diffpes.simul.kpar_to_detector_angles`.

    Property tests cover both exact detector-map compositions. Additional
    checks cover JIT, vmap, and static slit validation.

    :see: :func:`~diffpes.simul.kpar_to_detector_angles`
    """

    @given(
        tx_value=st.floats(
            min_value=-0.8,
            max_value=0.8,
            allow_nan=False,
            allow_infinity=False,
        ),
        ty_value=st.floats(
            min_value=-0.8,
            max_value=0.8,
            allow_nan=False,
            allow_infinity=False,
        ),
        energy_value=st.floats(
            min_value=5.0,
            max_value=150.0,
            allow_nan=False,
            allow_infinity=False,
        ),
    )
    @settings(max_examples=20, deadline=None)
    def test_angle_round_trip(
        self,
        tx_value: float,
        ty_value: float,
        energy_value: float,
    ) -> None:
        """Round-trip random detector angles for both slit conventions.

        The generated angles stay on the positive-normal branch. Their
        parallel momenta therefore have magnitude below ``0.95*k_f``.

        Notes
        -----
        Compose the forward and inverse maps for each slit. Compare both
        angles at absolute tolerance ``1e-12``.
        """
        tx: Float64[Array, ""] = jnp.asarray(tx_value)
        ty: Float64[Array, ""] = jnp.asarray(ty_value)
        energy: Float64[Array, ""] = jnp.asarray(energy_value)
        slit: str
        for slit in ("H", "V"):
            k_parallel: Float64[Array, "2"] = detector_angles_to_kpar(
                tx,
                ty,
                energy,
                slit,
            )
            recovered_tx: Float64[Array, ""]
            recovered_ty: Float64[Array, ""]
            recovered_tx, recovered_ty = kpar_to_detector_angles(
                k_parallel,
                energy,
                slit,
            )
            chex.assert_trees_all_close(
                (recovered_tx, recovered_ty),
                (tx, ty),
                rtol=0.0,
                atol=1e-12,
            )

    @given(
        normalized_kx=st.floats(
            min_value=-0.6,
            max_value=0.6,
            allow_nan=False,
            allow_infinity=False,
        ),
        normalized_ky=st.floats(
            min_value=-0.6,
            max_value=0.6,
            allow_nan=False,
            allow_infinity=False,
        ),
    )
    @settings(max_examples=20, deadline=None)
    def test_parallel_momentum_round_trip(
        self,
        normalized_kx: float,
        normalized_ky: float,
    ) -> None:
        """Round-trip random parallel momenta for both slit conventions.

        The generated normalized magnitude stays below 0.85. This bound lies
        inside the open physical domain required by the inverse.

        Notes
        -----
        Scale the normalized components by ``k_f`` at 40 eV. Compose inverse
        and forward maps at absolute tolerance ``1e-12``.
        """
        energy: Float64[Array, ""] = jnp.array(40.0)
        momentum: Float64[Array, ""] = final_state_k_inv_ang(energy)[0]
        k_parallel: Float64[Array, "2"] = momentum * jnp.asarray(
            [normalized_kx, normalized_ky]
        )
        slit: str
        for slit in ("H", "V"):
            tx: Float64[Array, ""]
            ty: Float64[Array, ""]
            tx, ty = kpar_to_detector_angles(k_parallel, energy, slit)
            recovered: Float64[Array, "2"] = detector_angles_to_kpar(
                tx,
                ty,
                energy,
                slit,
            )
            chex.assert_trees_all_close(
                recovered,
                k_parallel,
                rtol=0.0,
                atol=1e-12,
            )

    def test_jit_and_vmap_preserve_round_trip(self) -> None:
        """Verify batched round trips under static-slit JIT compilation.

        Five detector-angle pairs exercise the broadcast and vmap paths. The
        recovered angles must equal their inputs.

        Notes
        -----
        Vmap both public maps over scalar angle pairs. JIT each vmap and
        compare at ``rtol=1e-13``.
        """
        tx: Float64[Array, " 5"] = jnp.linspace(-0.3, 0.4, 5)
        ty: Float64[Array, " 5"] = jnp.linspace(0.2, -0.25, 5)
        energies: Float64[Array, " 5"] = jnp.linspace(20.0, 60.0, 5)
        slit: str
        for slit in ("H", "V"):
            with self.subTest(slit=slit):

                def forward_scalar(
                    one_tx: Float64[Array, ""],
                    one_ty: Float64[Array, ""],
                    energy: Float64[Array, ""],
                    slit_value: str = slit,
                ) -> Float64[Array, "2"]:
                    """Map one detector-angle pair to parallel momentum."""
                    result: Float64[Array, "2"] = detector_angles_to_kpar(
                        one_tx,
                        one_ty,
                        energy,
                        slit_value,
                    )
                    return result

                def inverse_scalar(
                    one_k_parallel: Float64[Array, "2"],
                    energy: Float64[Array, ""],
                    slit_value: str = slit,
                ) -> Tuple[Float64[Array, ""], Float64[Array, ""]]:
                    """Map one parallel momentum to detector angles."""
                    result: Tuple[
                        Float64[Array, ""],
                        Float64[Array, ""],
                    ] = kpar_to_detector_angles(
                        one_k_parallel,
                        energy,
                        slit_value,
                    )
                    return result

                forward: Callable[..., Float64[Array, "..."]] = jax.jit(
                    jax.vmap(forward_scalar)
                )
                inverse: Callable[
                    ..., Tuple[Float64[Array, "..."], Float64[Array, "..."]]
                ] = jax.jit(jax.vmap(inverse_scalar))
                k_parallel: Float64[Array, "5 2"] = forward(tx, ty, energies)
                recovered_tx: Float64[Array, " 5"]
                recovered_ty: Float64[Array, " 5"]
                recovered_tx, recovered_ty = inverse(k_parallel, energies)
                chex.assert_trees_all_close(
                    (recovered_tx, recovered_ty),
                    (tx, ty),
                    rtol=1e-13,
                    atol=1e-13,
                )

    def test_inverse_gradients_match_finite_differences(self) -> None:
        """Match inverse-map gradients with central finite differences.

        A generic weighted angle sum depends on parallel momentum and kinetic
        energy. Both slit branches must retain these sensitivities.

        Notes
        -----
        Use an interior parallel momentum at 38 eV. Apply the shared harness
        in forward and reverse autodiff modes.
        """
        k_parallel: Float64[Array, "2"] = jnp.array([0.3, -0.4])
        energy: Float64[Array, ""] = jnp.array(38.0)
        slit: str
        for slit in ("H", "V"):
            with self.subTest(slit=slit):

                def loss(
                    parameters: Tuple[
                        Float64[Array, "2"],
                        Float64[Array, ""],
                    ],
                    slit_value: str = slit,
                ) -> Float64[Array, ""]:
                    """Return one weighted inverse-map scalar."""
                    candidate_k: Float64[Array, "2"]
                    candidate_energy: Float64[Array, ""]
                    candidate_k, candidate_energy = parameters
                    tx: Float64[Array, ""]
                    ty: Float64[Array, ""]
                    tx, ty = kpar_to_detector_angles(
                        candidate_k,
                        candidate_energy,
                        slit_value,
                    )
                    value: Float64[Array, ""] = tx + 0.37 * ty
                    return value

                assert_gradients_match_finite_differences(
                    loss, (k_parallel, energy), regime="smooth"
                )

    def test_rejects_invalid_aperture_and_energy(self) -> None:
        """Reject the threshold and closed detector-aperture boundary.

        The inverse requires positive kinetic energy and parallel momentum
        strictly smaller than the final-state magnitude.

        Notes
        -----
        Exercise the energy threshold, aperture boundary, and exterior under
        eager and compiled execution for both slits.
        """
        momentum: Float64[Array, ""] = final_state_k_inv_ang(jnp.array(30.0))[
            0
        ]
        invalid_cases: Tuple[
            Tuple[Float64[Array, "2"], Float64[Array, ""]], ...
        ] = (
            (jnp.zeros(2), jnp.array(0.0)),
            (jnp.array([momentum, 0.0]), jnp.array(30.0)),
            (jnp.array([1.01 * momentum, 0.0]), jnp.array(30.0)),
        )
        k_parallel: Float64[Array, "2"]
        energy: Float64[Array, ""]
        case_index: int
        compiled: bool
        slit: str
        for compiled in (False, True):
            operation: Callable[
                ..., Tuple[Float64[Array, "..."], Float64[Array, "..."]]
            ] = kpar_to_detector_angles
            if compiled:
                operation = jax.jit(operation, static_argnames=("slit",))
            for slit in ("H", "V"):
                for case_index, (k_parallel, energy) in enumerate(
                    invalid_cases
                ):
                    with (
                        self.subTest(
                            compiled=compiled,
                            slit=slit,
                            case_index=case_index,
                        ),
                        pytest.raises(RuntimeError, match="requires Ekin > 0"),
                    ):
                        operation(k_parallel, energy, slit)

    def test_rejects_unknown_slit(self) -> None:
        """Verify that the inverse map rejects an unknown slit.

        The inverse shares the static slit contract with the forward detector
        map. An unsupported string must fail before numerical work.

        Notes
        -----
        Pass the value ``"bad"`` with one parallel vector. Require
        ``ValueError`` with the slit validation message.
        """
        with pytest.raises(ValueError, match="slit must be"):
            kpar_to_detector_angles(
                jnp.array([0.1, 0.2]),
                jnp.array(30.0),
                "bad",
            )
