"""Validate the coulomb functions module.

The cases use analytic values, invariants, and finite differences.
"""

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import chex
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from beartype import beartype
from beartype.typing import Any, Dict, Tuple
from jaxtyping import Array, Float64, jaxtyped

from diffpes.radial import (
    coulomb_fg,
    spherical_bessel_jl,
)


class TestCoulombFg:
    """Validate :func:`diffpes.radial.coulomb_fg`.

    The cases cover artifact authentication, dense values, derivatives, the
    Wronskian, the Coulomb ODE, and the plane-wave limit. They use frozen
    arbitrary-precision rows, JVPs, JIT, VMAP, and explicit domain controls.

    :see: :func:`~diffpes.radial.coulomb_fg`
    """

    def test_frozen_artifact_provenance_and_checksums(self) -> None:
        """Bind the independent dense artifact to its generator and manifest.

        The evidence package records the exact sparse and dense domains,
        arbitrary-precision engine, source revision, and immutable digests.

        Notes
        -----
        Recompute every listed SHA-256 from repository-root-relative paths.
        """
        root: Path = Path(__file__).parents[3]
        data_directory: Path = Path(__file__).with_name("data")
        manifest_path: Path = (
            data_directory / "coulomb_mpmath_80digit.manifest.json"
        )
        manifest: Dict[str, Any] = json.loads(
            manifest_path.read_text(encoding="utf-8")
        )
        assert manifest["schema"] == "diffpes.coulomb-mpmath-reference.v2"
        assert manifest["reference_engine"]["decimal_digits"] == 80
        assert manifest["dense_value_residual_product"] == {
            "eta_count": 25,
            "eta_interval": [-3.0, 3.0],
            "orders": [0, 1, 2, 3, 4],
            "rho_count": 257,
            "rho_interval": [1.0e-4, 40.0],
            "rho_spacing": "geometric",
        }
        archive_path: Path = data_directory / manifest["archive"]
        assert (
            hashlib.sha256(archive_path.read_bytes()).hexdigest()
            == manifest["archive_sha256"]
        )
        checksum_path: Path = data_directory / "coulomb_SHA256SUMS"
        line: str
        for line in checksum_path.read_text(encoding="utf-8").splitlines():
            expected: str
            relative: str
            expected, relative = line.split("  ", maxsplit=1)
            target: Path = root / relative
            assert hashlib.sha256(target.read_bytes()).hexdigest() == expected

    @pytest.mark.parametrize("order", range(5))
    @pytest.mark.rss_limit_mb(1500)
    def test_dense_reference_domain_in_isolated_processes(
        self,
        order: int,
    ) -> None:
        """Match all 32,125 independent dense Coulomb value nodes.

        Every static order covers the boundary-inclusive 25-by-257 domain,
        ODE residual, Wronskian, plane row, and origin asymptotics.

        Notes
        -----
        Separate processes release each order's adaptive-solver executables.
        """
        root: Path = Path(__file__).parents[3]
        script: Path = (
            root
            / "tests"
            / "_reference_tools"
            / "verify_coulomb_dense_reference.py"
        )
        completed: subprocess.CompletedProcess[str] = subprocess.run(  # noqa: S603
            [sys.executable, str(script), str(order)],
            cwd=root,
            check=False,
            capture_output=True,
            text=True,
        )
        assert completed.returncode == 0, completed.stdout + completed.stderr

    @pytest.mark.parametrize("order", range(5))
    @pytest.mark.rss_limit_mb(1600)
    def test_frozen_product_in_isolated_processes(
        self,
        order: int,
    ) -> None:
        """Match sparse derivatives in both AD modes and at each FD step.

        All four rows cover both parameter directions on the registered
        7-by-10 value/derivative product, including one-sided boundaries.

        Notes
        -----
        Separate processes release compiled executables between static orders.
        """
        root: Path = Path(__file__).parents[3]
        script: Path = (
            root / "tests" / "_reference_tools" / "verify_coulomb_reference.py"
        )
        completed: subprocess.CompletedProcess[str] = subprocess.run(  # noqa: S603
            [sys.executable, str(script), str(order)],
            cwd=root,
            check=False,
            capture_output=True,
            text=True,
        )
        assert completed.returncode == 0, completed.stdout + completed.stderr

    @pytest.mark.rss_limit_mb(1200)
    def test_plane_wave_identity_ode_and_parameter_gradients(self) -> None:
        """Check the plane limit, ODE residual, and parameter gradients.

        The test probes neutral and charged rows across the certified domain.

        Notes
        -----
        JAX directional derivatives supply the ODE and sensitivity witnesses.
        """
        rho: Float64[Array, " n_rho"] = jnp.asarray(
            [3.0e-4, 0.01, 0.3, 2.0, 10.0, 40.0]
        )
        eta_zero: Float64[Array, " n_rho"] = jnp.zeros_like(rho)
        regular: Float64[Array, " n_rho"]
        irregular: Float64[Array, " n_rho"]
        regular_derivative: Float64[Array, " n_rho"]
        irregular_derivative: Float64[Array, " n_rho"]
        (
            regular,
            irregular,
            regular_derivative,
            irregular_derivative,
        ) = coulomb_fg(2, eta_zero, rho)
        expected: Float64[Array, " n_rho"] = rho * spherical_bessel_jl(2, rho)
        np.testing.assert_allclose(
            regular,
            expected,
            rtol=1.0e-10,
            atol=1.0e-12,
        )

        eta: Float64[Array, " n_probe"] = jnp.asarray([-1.0, 0.25, 2.0])
        rho_probe: Float64[Array, " n_probe"] = jnp.asarray([0.2, 1.3, 7.0])

        @jaxtyped(typechecker=beartype)
        def _returned_values(
            eta_argument: Float64[Array, " n_probe"],
            rho_argument: Float64[Array, " n_probe"],
        ) -> Float64[Array, "4 n_probe"]:
            """PRIVATE: Return every Coulomb row for differentiation.

            Parameters
            ----------
            eta_argument : Float64[Array, " n_probe"]
                Dimensionless Sommerfeld parameters.
            rho_argument : Float64[Array, " n_probe"]
                Dimensionless radial coordinates.

            Returns
            -------
            result : Float64[Array, "4 n_probe"]
                Regular, irregular, and radial-derivative rows.

            Notes
            -----
            Evaluates order two and stacks the public tuple result.
            """
            rows: Tuple[Float64[Array, " n_probe"], ...] = coulomb_fg(
                2,
                eta_argument,
                rho_argument,
            )
            result: Float64[Array, "4 n_probe"] = jnp.stack(rows)
            return result

        values: Float64[Array, "4 n_probe"]
        eta_tangent: Float64[Array, "4 n_probe"]
        values, eta_tangent = jax.jvp(
            lambda argument: _returned_values(argument, rho_probe),
            (eta,),
            (jnp.ones_like(eta),),
        )
        rho_tangent: Float64[Array, "4 n_probe"] = jax.jvp(
            lambda argument: _returned_values(eta, argument),
            (rho_probe,),
            (jnp.ones_like(rho_probe),),
        )[1]
        chex.assert_tree_all_finite((values, eta_tangent, rho_tangent))
        assert float(jnp.linalg.norm(eta_tangent)) > 1.0e-4
        second_regular: Float64[Array, " n_probe"] = rho_tangent[2]
        ode_factor: Float64[Array, " n_probe"] = (
            1.0 - 2.0 * eta / rho_probe - 2.0 * 3.0 / rho_probe**2
        )
        residual: Float64[Array, " n_probe"] = (
            second_regular + ode_factor * values[0]
        )
        scale: Float64[Array, " n_probe"] = (
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
        eta: Float64[Array, " n"] = jnp.asarray([-0.5, 0.0, 0.5])
        rho: Float64[Array, " n"] = jnp.asarray([0.2, 1.0, 3.0])
        eager: Tuple[Float64[Array, " n"], ...] = coulomb_fg(1, eta, rho)
        compiled: Tuple[Float64[Array, " n"], ...] = jax.jit(
            lambda first, second: coulomb_fg(1, first, second)
        )(eta, rho)
        chex.assert_trees_all_close(
            eager,
            compiled,
            rtol=1.0e-10,
            atol=1.0e-12,
        )
        mapped: Float64[Array, " n"] = jax.vmap(
            lambda first, second: coulomb_fg(1, first, second)[0]
        )(eta, rho)
        chex.assert_trees_all_close(
            mapped,
            eager[0],
            rtol=1.0e-10,
            atol=1.0e-12,
        )
        with pytest.raises(
            (eqx.EquinoxRuntimeError, jax.errors.JaxRuntimeError),
            match="rho",
        ):
            coulomb_fg(0, jnp.asarray(0.0), jnp.asarray(0.0))
        with pytest.raises(
            (eqx.EquinoxRuntimeError, jax.errors.JaxRuntimeError),
            match="eta",
        ):
            coulomb_fg(0, jnp.asarray(3.1), jnp.asarray(1.0))
        with pytest.raises(ValueError, match="order"):
            coulomb_fg(6, jnp.asarray(0.0), jnp.asarray(1.0))
