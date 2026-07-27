"""Validate Coulomb radial functions and final-state dispatch.

The tests exercise frozen references, differential identities, and transforms.
"""

import subprocess
import sys
from pathlib import Path

import chex
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jaxtyping import Array, Complex, Float

from diffpes.radial import (
    coulomb_fg,
    coulomb_phase_shift,
    final_state_radial,
    spherical_bessel_jl,
)
from diffpes.types import FinalStateSpec, make_final_state_spec


def _reference() -> dict[str, np.ndarray]:
    """Load the frozen 80-digit-generated G11/D11 artifact."""
    path: Path = (
        Path(__file__).with_name("data") / "coulomb_mpmath_80digit.npz"
    )
    archive: np.lib.npyio.NpzFile = np.load(path)
    result: dict[str, np.ndarray] = {
        name: archive[name] for name in archive.files
    }
    return result


class TestCoulombPhaseShift:
    """Validate :func:`diffpes.radial.coulomb_phase_shift`."""

    def test_values_derivatives_and_continuous_branch(self) -> None:
        """Match phase values and derivatives on a continuous branch.

        The test covers every frozen order and a dense no-wrap sweep.

        Notes
        -----
        It compares production outputs with stored arbitrary-precision values.
        """
        reference: dict[str, np.ndarray] = _reference()
        etas: Float[Array, " n_eta"] = jnp.asarray(reference["etas"])
        order: int
        for order in range(5):
            values: Float[Array, " n_eta"] = coulomb_phase_shift(order, etas)
            derivatives: Float[Array, " n_eta"] = jax.jvp(
                lambda arguments: coulomb_phase_shift(order, arguments),
                (etas,),
                (jnp.ones_like(etas),),
            )[1]
            np.testing.assert_allclose(
                values,
                reference["phase"][order],
                rtol=1.0e-10,
                atol=1.0e-12,
            )
            np.testing.assert_allclose(
                derivatives,
                reference["phase_eta"][order],
                rtol=1.0e-10,
                atol=1.0e-12,
            )
        dense_eta: Float[Array, " n_dense"] = jnp.linspace(-3.0, 3.0, 601)
        dense_phase: Float[Array, " n_dense"] = coulomb_phase_shift(
            4,
            dense_eta,
        )
        assert float(jnp.max(jnp.abs(jnp.diff(dense_phase)))) < 0.1
        assert float(coulomb_phase_shift(4, jnp.asarray(0.0))) == 0.0


class TestCoulombFg:
    """Validate :func:`diffpes.radial.coulomb_fg`."""

    def test_frozen_product_in_isolated_processes(self) -> None:
        """Match every frozen Coulomb product row in isolated processes.

        The test covers values and both parameter-derivative directions.

        Notes
        -----
        Separate processes release compiled executables between static orders.
        """
        root: Path = Path(__file__).parents[3]
        script: Path = root / "scripts" / "verify_coulomb_reference.py"
        order: int
        for order in range(5):
            completed: subprocess.CompletedProcess[str] = subprocess.run(
                [sys.executable, str(script), str(order)],
                cwd=root,
                check=False,
                capture_output=True,
                text=True,
            )
            assert completed.returncode == 0, (
                completed.stdout + completed.stderr
            )

    @pytest.mark.rss_limit_mb(1200)
    def test_plane_wave_identity_ode_and_parameter_gradients(self) -> None:
        """Check the plane limit, ODE residual, and parameter gradients.

        The test probes neutral and charged rows across the certified domain.

        Notes
        -----
        JAX directional derivatives supply the ODE and sensitivity witnesses.
        """
        rho: Float[Array, " n_rho"] = jnp.asarray(
            [3.0e-4, 0.01, 0.3, 2.0, 10.0, 40.0]
        )
        eta_zero: Float[Array, " n_rho"] = jnp.zeros_like(rho)
        regular: Float[Array, " n_rho"]
        irregular: Float[Array, " n_rho"]
        regular_derivative: Float[Array, " n_rho"]
        irregular_derivative: Float[Array, " n_rho"]
        (
            regular,
            irregular,
            regular_derivative,
            irregular_derivative,
        ) = coulomb_fg(2, eta_zero, rho)
        expected: Float[Array, " n_rho"] = rho * spherical_bessel_jl(2, rho)
        np.testing.assert_allclose(
            regular,
            expected,
            rtol=1.0e-10,
            atol=1.0e-12,
        )

        eta: Float[Array, " n_probe"] = jnp.asarray([-1.0, 0.25, 2.0])
        rho_probe: Float[Array, " n_probe"] = jnp.asarray([0.2, 1.3, 7.0])

        def returned_values(
            eta_argument: Float[Array, " n_probe"],
            rho_argument: Float[Array, " n_probe"],
        ) -> Float[Array, "4 n_probe"]:
            rows: tuple[Float[Array, " n_probe"], ...] = coulomb_fg(
                2,
                eta_argument,
                rho_argument,
            )
            result: Float[Array, "4 n_probe"] = jnp.stack(rows)
            return result

        values: Float[Array, "4 n_probe"]
        eta_tangent: Float[Array, "4 n_probe"]
        values, eta_tangent = jax.jvp(
            lambda argument: returned_values(argument, rho_probe),
            (eta,),
            (jnp.ones_like(eta),),
        )
        rho_tangent: Float[Array, "4 n_probe"] = jax.jvp(
            lambda argument: returned_values(eta, argument),
            (rho_probe,),
            (jnp.ones_like(rho_probe),),
        )[1]
        chex.assert_tree_all_finite((values, eta_tangent, rho_tangent))
        assert float(jnp.linalg.norm(eta_tangent)) > 1.0e-4
        second_regular: Float[Array, " n_probe"] = rho_tangent[2]
        ode_factor: Float[Array, " n_probe"] = (
            1.0 - 2.0 * eta / rho_probe - 2.0 * 3.0 / rho_probe**2
        )
        residual: Float[Array, " n_probe"] = (
            second_regular + ode_factor * values[0]
        )
        scale: Float[Array, " n_probe"] = (
            jnp.abs(second_regular) + jnp.abs(ode_factor * values[0]) + 1.0
        )
        assert float(jnp.max(jnp.abs(residual) / scale)) < 1.0e-9

    def test_jit_vmap_and_domain_rejections(self) -> None:
        """Check transformations and reject arguments outside the domain.

        The test compares eager, compiled, and mapped Coulomb evaluations.

        Notes
        -----
        Explicit invalid inputs exercise radius, charge, and order guards.
        """
        eta: Float[Array, " n"] = jnp.asarray([-0.5, 0.0, 0.5])
        rho: Float[Array, " n"] = jnp.asarray([0.2, 1.0, 3.0])
        eager: tuple[Float[Array, " n"], ...] = coulomb_fg(1, eta, rho)
        compiled: tuple[Float[Array, " n"], ...] = jax.jit(
            lambda first, second: coulomb_fg(1, first, second)
        )(eta, rho)
        chex.assert_trees_all_close(
            eager,
            compiled,
            rtol=1.0e-10,
            atol=1.0e-12,
        )
        mapped: Float[Array, " n"] = jax.vmap(
            lambda first, second: coulomb_fg(1, first, second)[0]
        )(eta, rho)
        chex.assert_trees_all_close(
            mapped,
            eager[0],
            rtol=1.0e-10,
            atol=1.0e-12,
        )
        with pytest.raises(Exception, match="rho"):
            coulomb_fg(0, jnp.asarray(0.0), jnp.asarray(0.0))
        with pytest.raises(Exception, match="eta"):
            coulomb_fg(0, jnp.asarray(3.1), jnp.asarray(1.0))
        with pytest.raises(ValueError, match="order"):
            coulomb_fg(6, jnp.asarray(0.0), jnp.asarray(1.0))


class TestFinalStateRadial:
    """Validate :func:`diffpes.radial.final_state_radial`."""

    def test_plane_wave_limit_origin_and_charge_gradient(self) -> None:
        """Match the plane limit and retain a nonzero charge gradient.

        The test covers the origin and every supported final-state order.

        Notes
        -----
        Direct comparisons and reverse-mode differentiation verify continuity.
        """
        plane_spec: FinalStateSpec = make_final_state_spec()
        coulomb_zero_spec: FinalStateSpec = make_final_state_spec(
            mode="coulomb",
            effective_charge=0.0,
        )
        momentum: Float[Array, ""] = jnp.asarray(1.2)
        radius: Float[Array, " n_r"] = jnp.asarray([0.0, 1.0e-5, 0.1, 2.0])
        order: int
        for order in range(6):
            plane: Complex[Array, " n_r"] = final_state_radial(
                order,
                momentum,
                radius,
                plane_spec,
            )
            coulomb_zero: Complex[Array, " n_r"] = final_state_radial(
                order,
                momentum,
                radius,
                coulomb_zero_spec,
            )
            np.testing.assert_allclose(
                coulomb_zero,
                plane,
                rtol=1.0e-10,
                atol=1.0e-12,
            )
            assert bool(jnp.all(jnp.isfinite(coulomb_zero)))

        def charged_value(charge: Float[Array, ""]) -> Float[Array, ""]:
            spec: FinalStateSpec = make_final_state_spec(
                mode="coulomb",
                effective_charge=charge,
            )
            radial: Complex[Array, " n_r"] = final_state_radial(
                1,
                momentum,
                radius,
                spec,
            )
            result: Float[Array, ""] = jnp.real(radial[-1])
            return result

        charge_gradient: Float[Array, ""] = jax.grad(charged_value)(
            jnp.asarray(0.3)
        )
        assert bool(jnp.isfinite(charge_gradient))
        assert float(jnp.abs(charge_gradient)) > 1.0e-4

    def test_coulomb_rejects_zero_momentum(self) -> None:
        """Reject zero momentum for a charged Coulomb final state.

        The test exercises the singular Sommerfeld-parameter boundary.

        Notes
        -----
        The public runtime guard supplies the expected diagnostic.
        """
        spec: FinalStateSpec = make_final_state_spec(
            mode="coulomb",
            effective_charge=0.2,
        )
        with pytest.raises(Exception, match="positive momentum"):
            final_state_radial(
                0,
                jnp.asarray(0.0),
                jnp.asarray([0.0, 1.0]),
                spec,
            )
