"""Validate the spectral eigen module.

The cases use analytic values, invariants, and finite differences.
"""

from __future__ import annotations

import chex
import jax
import jax.numpy as jnp
import numpy as np
from beartype.typing import Dict
from jaxtyping import Array, Bool, Complex128, Float64
from numpy.typing import NDArray

from diffpes.constants import (
    EPS_DEG,
)
from diffpes.simul import (
    spectral_intensity_eigen,
    spectral_intensity_resolvent,
)
from tests._assertions import assert_rejects

from ._spectral_helpers import (
    _spectral_intensity_reference,
)


class TestSpectralIntensityEigen(chex.TestCase):
    """Validate :func:`~diffpes.simul.spectral_intensity_eigen`.

    The cases compare eigenstate and resolvent results away from degeneracy,
    then check regulator limits, gradients, and domain rejection.
    """

    def test_generic_hermitian_resolvent_equivalence(self) -> None:
        """Match reference values on the frozen complex-Hermitian fixture.

        The two public representations consume independently prepared inputs.

        Notes
        -----
        The eigen path consumes only the independently diagonalized values
        and gauge-invariant weights from the immutable archive.
        """
        reference: Dict[str, Float64[NDArray, "..."]] = (
            _spectral_intensity_reference()
        )
        eigenvalues: Float64[Array, " 2"] = jnp.asarray(
            reference["hermitian_eigenvalues"]
        )
        weights: Float64[Array, " 2"] = jnp.asarray(
            reference["hermitian_band_weights"]
        )
        omega: Float64[Array, " n"] = jnp.asarray(reference["hermitian_omega"])
        sigma: Complex128[Array, ""] = jnp.asarray(
            -1.0j * reference["hermitian_gamma_sigma"],
            dtype=jnp.complex128,
        )
        eta: Float64[Array, ""] = jnp.asarray(reference["hermitian_eta"])
        actual: Float64[Array, " n"] = jax.jit(
            jax.vmap(
                spectral_intensity_eigen,
                in_axes=(None, None, 0, None, None),
            )
        )(eigenvalues, weights, omega, sigma, eta)
        np.testing.assert_allclose(
            np.asarray(actual),
            reference["hermitian_intensity"],
            rtol=1.0e-10,
            atol=1.0e-13,
        )

    def test_eta_regulator_ladder(self) -> None:
        """Match every regulator rung and its frozen convergence rows.

        The test checks values and convergence against the independent archive.

        Notes
        -----
        The one-level fixture isolates the rule that eta enters only through
        the total Lorentzian linewidth ``Gamma + eta``.
        """
        reference: Dict[str, Float64[NDArray, "..."]] = (
            _spectral_intensity_reference()
        )
        eigenvalue: Float64[Array, " 1"] = jnp.atleast_1d(
            jnp.asarray(reference["eta_ladder_level_energy"])
        )
        weight: Float64[Array, " 1"] = jnp.ones(1, dtype=jnp.float64)
        omega: Float64[Array, " n"] = jnp.asarray(
            reference["eta_ladder_omega"]
        )
        sigma: Complex128[Array, ""] = jnp.asarray(
            -1.0j * reference["eta_ladder_gamma_physical"],
            dtype=jnp.complex128,
        )

        def row(eta: Float64[Array, ""]) -> Float64[Array, " n"]:
            """Evaluate one regulator rung over all energies."""
            returned: Float64[Array, " n"] = jax.vmap(
                spectral_intensity_eigen,
                in_axes=(None, None, 0, None, None),
            )(eigenvalue, weight, omega, sigma, eta)
            return returned

        actual: Float64[Array, "rung n"] = jax.jit(jax.vmap(row))(
            jnp.asarray(reference["eta_ladder_etas"])
        )
        np.testing.assert_allclose(
            np.asarray(actual),
            reference["eta_ladder_intensities"],
            rtol=1.0e-10,
            atol=1.0e-13,
        )
        captured: Float64[Array, " rung"] = jnp.trapezoid(actual, omega)
        np.testing.assert_allclose(
            np.asarray(captured),
            reference["eta_ladder_captured_masses"],
            rtol=1.0e-5,
            atol=2.0e-7,
        )

    def test_off_degenerate_value_and_gradient_equivalence(self) -> None:
        """Match resolvent and eigen values and hopping gradients.

        The fixture remains safely above the differentiated eigen gap floor.

        Notes
        -----
        The two-pole gap remains far above the degeneracy tolerance. Its
        eigenvectors only form the invariant band weights.
        """
        epsilon: float = -0.15
        source: Complex128[Array, " 2"] = jnp.asarray(
            [0.9 + 0.4j, -0.5 + 0.7j]
        )
        omega: Float64[Array, ""] = jnp.asarray(-0.08)
        sigma: Complex128[Array, ""] = jnp.asarray(-0.03j)
        eta: float = 1.0e-4

        def pair(hopping: Float64[Array, ""]) -> Float64[Array, " 2"]:
            """Return resolver and eigen intensity at one hopping."""
            hamiltonian: Complex128[Array, "2 2"] = jnp.asarray(
                [[epsilon, hopping], [hopping, epsilon]],
                dtype=jnp.complex128,
            )
            eigenvalues: Float64[Array, " 2"]
            eigenvectors: Complex128[Array, "2 2"]
            eigenvalues, eigenvectors = jnp.linalg.eigh(hamiltonian)
            weights: Float64[Array, " 2"] = (
                jnp.abs(eigenvectors.conj().T @ source) ** 2
            )
            returned: Float64[Array, " 2"] = jnp.stack(
                [
                    spectral_intensity_resolvent(
                        hamiltonian, source[None, :], omega, sigma, eta
                    ),
                    spectral_intensity_eigen(
                        eigenvalues, weights, omega, sigma, eta
                    ),
                ]
            )
            return returned

        hopping: Float64[Array, ""] = jnp.asarray(0.08)
        values: Float64[Array, " 2"] = jax.jit(pair)(hopping)
        gradients: Float64[Array, " 2"] = jax.jacfwd(pair)(hopping)
        np.testing.assert_allclose(
            np.asarray(values[0]),
            np.asarray(values[1]),
            rtol=1.0e-10,
            atol=1.0e-12,
        )
        np.testing.assert_allclose(
            np.asarray(gradients[0]),
            np.asarray(gradients[1]),
            rtol=1.0e-10,
            atol=1.0e-10,
        )

    def test_negative_band_weight_rejects(self) -> None:
        """Reject a negative input weight eagerly and inside JIT.

        The public eigen seam enforces nonnegative squared amplitudes.

        Notes
        -----
        Band weights are gauge-invariant squared amplitudes and cannot carry
        a negative signed contribution.
        """
        assert_rejects(
            spectral_intensity_eigen,
            jnp.asarray([-0.2, 0.3]),
            jnp.asarray([1.0, -0.1]),
            jnp.asarray(0.0),
            jnp.asarray(-0.02j),
            1.0e-4,
            match="weights|nonnegative",
        )

    def test_nondegenerate_domain_floor_and_value_only_exception(self) -> None:
        """Enforce the differentiated gap floor eagerly and inside JIT.

        The test probes exact, sub-floor, boundary, and explicit primal cases.

        Notes
        -----
        The registered comparison domain includes its exact lower boundary.
        Exact and
        sub-floor pairs reject by default. A degenerate primal requires the
        explicit value-only policy and emits no derivative evidence.
        """
        gap_floor: float = 1.0e3 * EPS_DEG
        below_floor: float = float(np.nextafter(gap_floor, 0.0))
        weights: Float64[Array, " 2"] = jnp.asarray([0.7, 0.4])
        omega: Float64[Array, ""] = jnp.asarray(0.03)
        sigma: Complex128[Array, ""] = jnp.asarray(-0.02j)
        eigenvalues: Float64[Array, " 2"]
        for eigenvalues in (
            jnp.asarray([0.0, 0.0]),
            jnp.asarray([0.0, below_floor]),
        ):
            assert_rejects(
                spectral_intensity_eigen,
                eigenvalues,
                weights,
                omega,
                sigma,
                1.0e-4,
                match="gap|resolvent|value_only",
            )

        boundary: Float64[Array, " 2"] = jnp.asarray([0.0, gap_floor])
        accepted: Bool[Array, ""] = jax.jit(spectral_intensity_eigen)(
            boundary,
            weights,
            omega,
            sigma,
            1.0e-4,
        )
        value_only: Float64[Array, ""] = jax.jit(
            lambda eigenvalues: spectral_intensity_eigen(
                eigenvalues,
                weights,
                omega,
                sigma,
                1.0e-4,
                allow_degenerate_value_only=True,
            )
        )(jnp.asarray([0.0, 0.0]))
        assert bool(jnp.all(jnp.isfinite(jnp.asarray([accepted, value_only]))))
