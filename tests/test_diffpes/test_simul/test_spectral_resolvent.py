"""Validate the spectral resolvent module.

The cases use analytic values, invariants, and finite differences.
"""

from __future__ import annotations

import chex
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from beartype.typing import Any, Dict
from jaxtyping import Array, Complex128, Float64
from numpy.typing import NDArray

from diffpes.simul import (
    projected_spectral_density_resolvent,
    spectral_intensity_resolvent,
)
from diffpes.tightb import bloch_hamiltonian
from tests._assertions import assert_rejects
from tests._factories import make_t2g_soc_model

from ._spectral_helpers import (
    _degenerate_gradient_witness,
    _spectral_intensity_reference,
)


class TestSpectralIntensityResolvent(chex.TestCase):
    """Validate :func:`~diffpes.simul.spectral_intensity_resolvent`.

    The cases cover analytic poles, separate outgoing sources, complex
    gradients, exact degeneracies, and invalid physical domains.
    """

    def test_two_pole_closed_form_and_degenerate_limit(self) -> None:
        """Match the frozen two-pole truth, including exact degeneracy.

        The test covers a full energy row and the coincident-pole limit.

        Notes
        -----
        One jitted vmap evaluates the full 2001-point energy row. The same
        source then probes the exact ``t=0`` limit without choosing an
        eigenvector basis.
        """
        reference: Dict[str, Float64[NDArray, "..."]] = (
            _spectral_intensity_reference()
        )
        epsilon: float = float(reference["two_pole_epsilon0"])
        hopping: float = float(reference["two_pole_hopping"])
        hamiltonian: Complex128[Array, "2 2"] = jnp.asarray(
            [[epsilon, hopping], [hopping, epsilon]],
            dtype=jnp.complex128,
        )
        degenerate: Complex128[Array, "2 2"] = epsilon * jnp.eye(
            2, dtype=jnp.complex128
        )
        source: Complex128[Array, "1 2"] = jnp.asarray(
            reference["two_pole_source_real"]
            + 1.0j * reference["two_pole_source_imag"],
            dtype=jnp.complex128,
        )[None, :]
        omega: Float64[Array, " n"] = jnp.asarray(reference["two_pole_omega"])
        sigma: Complex128[Array, ""] = jnp.asarray(
            -1.0j * reference["two_pole_gamma"],
            dtype=jnp.complex128,
        )
        eta: Float64[Array, ""] = jnp.asarray(reference["two_pole_eta"])

        def row(
            matrix: Complex128[Array, "2 2"],
        ) -> Float64[Array, " n"]:
            """Vectorize one Hamiltonian over the frozen axis."""
            returned: Float64[Array, " n"] = jax.vmap(
                spectral_intensity_resolvent,
                in_axes=(None, None, 0, None, None),
            )(matrix, source, omega, sigma, eta)
            return returned

        actual: Float64[Array, " n"] = jax.jit(row)(hamiltonian)
        actual_degenerate: Float64[Array, " n"] = jax.jit(row)(degenerate)
        np.testing.assert_allclose(
            np.asarray(actual),
            reference["two_pole_intensity"],
            rtol=1.0e-10,
            atol=1.0e-13,
        )
        np.testing.assert_allclose(
            np.asarray(actual_degenerate),
            reference["two_pole_intensity_degenerate"],
            rtol=1.0e-10,
            atol=1.0e-13,
        )

    def test_outgoing_sources_solve_separately_before_sum(self) -> None:
        """Require the amended incoherent outgoing-channel reduction.

        The planted sources distinguish separate solves from coherent addition.

        Notes
        -----
        The two planted sources have a nonzero cross term. Coherently adding
        them before the solve therefore changes the answer and cannot satisfy
        the registered matrix-element handoff.
        """
        hamiltonian: Complex128[Array, "2 2"] = jnp.asarray(
            [[-0.27, 0.09 + 0.04j], [0.09 - 0.04j, 0.31]]
        )
        sources: Complex128[Array, "2 2"] = jnp.asarray(
            [[0.8 + 0.2j, -0.3 + 0.5j], [0.4 - 0.7j, 0.6 + 0.1j]]
        )
        omega: Float64[Array, ""] = jnp.asarray(-0.08)
        sigma: Complex128[Array, ""] = jnp.asarray(0.01 - 0.05j)
        eta: float = 2.0e-4
        produced: Float64[Array, ""] = spectral_intensity_resolvent(
            hamiltonian,
            sources,
            omega,
            sigma,
            eta,
        )
        separate: Float64[Array, ""] = sum(
            spectral_intensity_resolvent(
                hamiltonian,
                sources[index : index + 1],
                omega,
                sigma,
                eta,
            )
            for index in range(2)
        )
        coherent: Float64[Array, ""] = spectral_intensity_resolvent(
            hamiltonian,
            jnp.sum(sources, axis=0, keepdims=True),
            omega,
            sigma,
            eta,
        )
        np.testing.assert_allclose(
            np.asarray(produced),
            np.asarray(separate),
            rtol=1.0e-13,
            atol=1.0e-14,
        )
        assert float(jnp.abs(produced - coherent)) > 1.0e-3

    def test_outgoing_source_axis_must_be_nonempty(self) -> None:
        """Reject an empty outgoing-channel axis before tracing a solve.

        The public scalar seam must always receive at least one source ket.

        Notes
        -----
        A Python shape guard rejects before tracing the linear solver.
        """
        with pytest.raises(ValueError, match="n_out|nonempty"):
            spectral_intensity_resolvent(
                jnp.eye(2, dtype=jnp.complex128),
                jnp.empty((0, 2), dtype=jnp.complex128),
                jnp.asarray(0.0),
                jnp.asarray(-0.04j),
                1.0e-4,
            )

    def test_generic_complex_adjoint_gradient(self) -> None:
        """Match independent two-solve adjoint derivative truth.

        The test differentiates a generic complex-Hermitian two-level problem.

        Notes
        -----
        Four real coordinates span every independent entry of a generic
        complex-Hermitian two-orbital Hamiltonian. Lineax supplies reverse
        mode without a production custom derivative.
        """
        reference: Dict[str, Float64[NDArray, "..."]] = (
            _spectral_intensity_reference()
        )
        hamiltonian: Complex128[Array, "2 2"] = jnp.asarray(
            reference["hermitian_hamiltonian_real"]
            + 1.0j * reference["hermitian_hamiltonian_imag"]
        )
        source: Complex128[Array, "1 2"] = jnp.asarray(
            reference["hermitian_source_real"]
            + 1.0j * reference["hermitian_source_imag"]
        )[None, :]
        directions: Complex128[Array, "4 2 2"] = jnp.asarray(
            reference["adjoint_direction_real"]
            + 1.0j * reference["adjoint_direction_imag"]
        )
        sigma: Complex128[Array, ""] = jnp.asarray(
            -1.0j * reference["hermitian_gamma_sigma"],
            dtype=jnp.complex128,
        )
        eta: Float64[Array, ""] = jnp.asarray(reference["hermitian_eta"])
        omegas: Float64[Array, " n"] = jnp.asarray(reference["adjoint_omegas"])

        def gradient_at(omega: Float64[Array, ""]) -> Float64[Array, " 4"]:
            """Differentiate all Hermitian coordinates at omega."""

            def intensity(
                coordinates: Float64[Array, " 4"],
            ) -> Float64[Array, ""]:
                candidate: Complex128[Array, "2 2"] = (
                    hamiltonian
                    + jnp.tensordot(
                        coordinates,
                        directions,
                        axes=1,
                    )
                )
                returned: Float64[Array, ""] = spectral_intensity_resolvent(
                    candidate,
                    source,
                    omega,
                    sigma,
                    eta,
                )
                return returned

            returned: Float64[Array, " 4"] = jax.grad(intensity)(
                jnp.zeros(4, dtype=jnp.float64)
            )
            return returned

        actual: Float64[Array, "n 4"] = jax.jit(jax.vmap(gradient_at))(omegas)
        np.testing.assert_allclose(
            np.asarray(actual),
            reference["adjoint_analytic"],
            rtol=1.0e-10,
            atol=1.0e-10,
        )

    def test_graphene_exact_degeneracy_parameter_gradient(self) -> None:
        """Match the one-bond derivative at graphene K.

        The test differentiates through an exact orbital degeneracy.

        Notes
        -----
        The path varies the registered single bond. Both reverse and forward
        AD match the independently frozen finite-difference truth.
        """
        witness: Dict[str, Any] = _degenerate_gradient_witness()
        graphene: Dict[str, Any] = witness["graphene_one_bond_witness"]
        direction_entry: Dict[str, float] = graphene["measurements"][
            "one_bond_dh_dtheta_offdiag"
        ]
        off_diagonal: complex = complex(
            direction_entry["real"], direction_entry["imag"]
        )
        bond_direction: Complex128[Array, "2 2"] = jnp.asarray(
            [[0.0, off_diagonal], [off_diagonal.conjugate(), 0.0]],
            dtype=jnp.complex128,
        )
        graphene_source: Complex128[Array, "1 2"] = (
            jnp.asarray(graphene["intensity"]["source_real"])[None, :]
            + 1.0j * jnp.asarray(graphene["intensity"]["source_imag"])[None, :]
        )

        def graphene_intensity(
            coordinate: Float64[Array, ""],
        ) -> Float64[Array, ""]:
            """Evaluate the registered one-bond resolvent path."""
            hamiltonian: Complex128[Array, "2 2"] = coordinate * bond_direction
            returned: Float64[Array, ""] = spectral_intensity_resolvent(
                hamiltonian,
                graphene_source,
                jnp.asarray(graphene["intensity"]["omega_ev"]),
                jnp.asarray(0.0j, dtype=jnp.complex128),
                graphene["intensity"]["eta_ev"],
            )
            return returned

        zero: Float64[Array, ""] = jnp.asarray(0.0)
        graphene_reverse: Float64[Array, ""] = jax.grad(graphene_intensity)(
            zero
        )
        graphene_forward: Float64[Array, ""] = jax.jacfwd(graphene_intensity)(
            zero
        )
        graphene_truth: float = graphene["measurements"][
            "one_bond_grad_reverse"
        ]
        np.testing.assert_allclose(
            np.asarray([graphene_reverse, graphene_forward]),
            graphene_truth,
            rtol=1.0e-10,
            atol=1.0e-10,
        )

    def test_kramers_exact_degeneracy_parameter_gradient(self) -> None:
        """Match a crystal-field derivative at a Kramers-degenerate point.

        The test differentiates a perturbation that preserves each Kramers
        pair.

        Notes
        -----
        The spin-symmetric field preserves every Kramers pair. Both reverse
        and forward resolvent AD match the finest independently frozen
        central finite-difference rung.
        """
        witness: Dict[str, Any] = _degenerate_gradient_witness()
        kramers: Dict[str, Any] = witness["t2g_soc_kramers_witness"]
        kramers_model: Any = make_t2g_soc_model(coupling=0.4)
        kramers_k: Float64[Array, " 3"] = jnp.asarray(
            kramers["kramers_k_fractional"]
        )
        kramers_hamiltonian: Complex128[Array, "6 6"] = bloch_hamiltonian(
            kramers_model, kramers_k
        )
        field_direction: Complex128[Array, "6 6"] = jnp.diag(
            jnp.asarray([1.0, 0.0, 0.0, 1.0, 0.0, 0.0])
        ).astype(jnp.complex128)
        kramers_source: Complex128[Array, "1 6"] = (
            jnp.asarray(kramers["intensity"]["source_real"])[None, :]
            + 1.0j * jnp.asarray(kramers["intensity"]["source_imag"])[None, :]
        )

        def kramers_intensity(
            coordinate: Float64[Array, ""],
        ) -> Float64[Array, ""]:
            """Evaluate the registered Kramers resolvent path."""
            returned: Float64[Array, ""] = spectral_intensity_resolvent(
                kramers_hamiltonian + coordinate * field_direction,
                kramers_source,
                jnp.asarray(kramers["intensity"]["omega_ev"]),
                jnp.asarray(0.0j, dtype=jnp.complex128),
                kramers["intensity"]["gamma_ev"],
            )
            return returned

        zero: Float64[Array, ""] = jnp.asarray(0.0)
        kramers_reverse: Float64[Array, ""] = jax.grad(kramers_intensity)(zero)
        kramers_forward: Float64[Array, ""] = jax.jacfwd(kramers_intensity)(
            zero
        )
        kramers_truth: float = kramers["measurements"][
            "crystal_field_fd_central"
        ][-1]
        np.testing.assert_allclose(
            np.asarray([kramers_reverse, kramers_forward]),
            kramers_truth,
            rtol=1.0e-5,
            atol=1.0e-8,
        )

    def test_invalid_physical_domains_reject_eager_and_jit(self) -> None:
        """Reject non-Hermitian H, advanced self-energy, and nonpositive eta.

        The test exercises the same physical-domain predicates eagerly and in
        JIT.

        Notes
        -----
        The shared traced rejection helper evaluates each predicate.
        """
        hamiltonian: Complex128[Array, "2 2"] = jnp.asarray(
            [[0.1, 0.2j], [0.1j, -0.3]],
            dtype=jnp.complex128,
        )
        source: Complex128[Array, "1 2"] = jnp.asarray(
            [[0.4 + 0.2j, -0.1 + 0.7j]]
        )
        omega: Float64[Array, ""] = jnp.asarray(0.05)
        sigma: Complex128[Array, ""] = jnp.asarray(-0.02j)
        assert_rejects(
            spectral_intensity_resolvent,
            hamiltonian,
            source,
            omega,
            sigma,
            1.0e-4,
            match="Hermitian",
        )
        hermitian: Complex128[Array, "2 2"] = (
            hamiltonian + hamiltonian.conj().T
        )
        assert_rejects(
            spectral_intensity_resolvent,
            hermitian,
            source,
            omega,
            jnp.asarray(1.0e-5j),
            1.0e-4,
            match="retarded|nonpositive",
        )
        assert_rejects(
            spectral_intensity_resolvent,
            hermitian,
            source,
            omega,
            sigma,
            0.0,
            match="eta|positive",
        )


class TestProjectedSpectralDensityResolvent(chex.TestCase):
    """Validate :func:`~diffpes.simul.projected_spectral_density_resolvent`.

    The cases compare the matrix density with an independent inverse and its
    parameter gradient with a central finite difference.
    """

    def test_matrix_spectral_density_matches_independent_inverse(self) -> None:
        """Match the full Hermitian density and preserve its coherences.

        The test compares every matrix entry across three sampled energies.

        Notes
        -----
        The truth explicitly inverts the two-orbital matrix and forms the
        matrix anti-Hermitian part before projection. A jitted vmap checks
        three sampled energies.
        """
        hamiltonian: Complex128[Array, "2 2"] = jnp.asarray(
            [[0.2, -0.1 + 0.3j], [-0.1 - 0.3j, -0.4]]
        )
        transition: Complex128[Array, "3 2"] = jnp.asarray(
            [
                [1.0 + 0.2j, -0.3 + 0.7j],
                [0.1 - 0.4j, 0.8 + 0.5j],
                [-0.6 + 0.3j, 0.2 - 0.9j],
            ]
        )
        omegas: Float64[Array, " 3"] = jnp.asarray([-0.5, 0.0, 0.7])
        sigma: Complex128[Array, ""] = jnp.asarray(-0.03 - 0.04j)
        eta: Float64[Array, ""] = jnp.asarray(2.0e-4)

        def production(omega: Float64[Array, ""]) -> Complex128[Array, "3 3"]:
            """Evaluate one projected production density."""
            returned: Complex128[Array, "3 3"] = (
                projected_spectral_density_resolvent(
                    hamiltonian,
                    transition,
                    omega,
                    sigma,
                    eta,
                )
            )
            return returned

        def truth(omega: Float64[Array, ""]) -> Complex128[Array, "3 3"]:
            """Compute the density through an explicit dense inverse."""
            identity: Complex128[Array, "2 2"] = jnp.eye(
                2, dtype=jnp.complex128
            )
            green: Complex128[Array, "2 2"] = jnp.linalg.inv(
                (omega + 1.0j * eta - sigma) * identity - hamiltonian
            )
            spectral: Complex128[Array, "2 2"] = -(green - green.conj().T) / (
                2.0j * jnp.pi
            )
            returned: Complex128[Array, "3 3"] = (
                transition @ spectral @ transition.conj().T
            )
            return returned

        actual: Complex128[Array, "3 3 3"] = jax.jit(jax.vmap(production))(
            omegas
        )
        expected: Complex128[Array, "3 3 3"] = jax.vmap(truth)(omegas)
        np.testing.assert_allclose(
            np.asarray(actual),
            np.asarray(expected),
            rtol=1.0e-12,
            atol=1.0e-12,
        )
        np.testing.assert_allclose(
            np.asarray(actual),
            np.asarray(actual.conj().swapaxes(-1, -2)),
            rtol=0.0,
            atol=1.0e-13,
        )
        assert bool(jnp.all(jnp.linalg.eigvalsh(actual) >= -1.0e-12))
        assert bool(jnp.any(jnp.abs(jnp.imag(actual[:, 0, 1])) > 1.0e-4))

    def test_projected_density_gradient_matches_central_difference(
        self,
    ) -> None:
        """Match a generic matrix-density gradient to a central difference.

        The scalar loss retains diagonal and off-diagonal density information.

        Notes
        -----
        The scalar loss retains off-diagonal density entries so the test
        exercises complex cotangents through all multiple right-hand sides.
        """
        base: Complex128[Array, "2 2"] = jnp.asarray(
            [[-0.2, 0.11 + 0.08j], [0.11 - 0.08j, 0.3]]
        )
        transition: Complex128[Array, "2 2"] = jnp.asarray(
            [[0.7 + 0.2j, -0.4 + 0.1j], [0.3 - 0.8j, 0.5 + 0.6j]]
        )

        def loss(coordinate: Float64[Array, ""]) -> Float64[Array, ""]:
            """Compute a real loss from the projected density."""
            candidate: Complex128[Array, "2 2"] = base.at[0, 0].add(
                coordinate + 0.0j
            )
            density: Complex128[Array, "2 2"] = (
                projected_spectral_density_resolvent(
                    candidate,
                    transition,
                    jnp.asarray(0.07),
                    jnp.asarray(-0.025j),
                    1.0e-4,
                )
            )
            returned: Float64[Array, ""] = jnp.real(
                density[0, 0] + 0.3j * density[0, 1]
            )
            return returned

        zero: Float64[Array, ""] = jnp.asarray(0.0)
        reverse: Float64[Array, ""] = jax.grad(loss)(zero)
        forward: Float64[Array, ""] = jax.jacfwd(loss)(zero)
        step: float = 2.0**-16
        finite_difference: Float64[Array, ""] = (
            loss(zero + step) - loss(zero - step)
        ) / (2.0 * step)
        np.testing.assert_allclose(
            np.asarray([reverse, forward]),
            np.asarray(finite_difference),
            rtol=1.0e-6,
            atol=1.0e-8,
        )
