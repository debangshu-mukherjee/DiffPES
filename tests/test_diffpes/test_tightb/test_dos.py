"""Validate broadened tight-binding DOS and finite-temperature filling.

The tests cover analytic convolution, normalization, compilation, gradients,
particle-hole symmetry, implicit-solve stability, and domain validation.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from beartype.typing import List
from jaxtyping import Array, Float64
from scipy import integrate

from diffpes.tightb import (
    dos_gaussian,
    eigvalsh_bands,
    fermi_level_from_filling,
)
from diffpes.tightb.dos import _solve_filling
from tests._factories import make_graphene_model
from tests._gradients import assert_gradients_match_finite_differences


class TestDosGaussian:
    """Validate :func:`diffpes.tightb.dos_gaussian`.

    The cases check analytic convolution, normalization, differentiation,
    compilation, and invalid weight measures.
    """

    def test_chain_matches_independent_closed_form_convolution(self) -> None:
        """Compare a dense k-sum with quadrature of the analytic chain DOS.

        Independent angular integration supplies the broadened closed-form
        reference at five energies.

        Notes
        -----
        Evaluate SciPy quadrature outside JAX and compare the sampled result.
        """
        hopping: float = 1.0
        n_k: int = 4096
        momenta: Float64[Array, " n_k"] = (
            jnp.arange(n_k, dtype=jnp.float64) / n_k - 0.5
        )
        eigenvalues: Float64[Array, "n_k 1"] = (
            2.0 * hopping * jnp.cos(2.0 * jnp.pi * momenta)
        )[:, None]
        weights: Float64[Array, " n_k"] = jnp.full(
            n_k, 1.0 / n_k, dtype=jnp.float64
        )
        energy_axis: Float64[Array, " 5"] = jnp.asarray(
            [-1.5, -0.7, 0.0, 0.8, 1.6],
            dtype=jnp.float64,
        )
        sigma: float = 0.2
        actual: Float64[Array, " 5"] = dos_gaussian(
            eigenvalues,
            weights,
            energy_axis,
            sigma,
        ).total_dos

        def convolved_density(energy: float) -> float:
            def integrand(angle: float) -> float:
                center: float = 2.0 * hopping * np.cos(angle)
                difference: float = energy - center
                gaussian: float = np.exp(-0.5 * (difference / sigma) ** 2) / (
                    np.sqrt(2.0 * np.pi) * sigma
                )
                density: float = gaussian / np.pi
                return density

            value: float = integrate.quad(
                integrand,
                0.0,
                np.pi,
                epsabs=1e-12,
                epsrel=1e-12,
            )[0]
            return value

        expected: Float64[Array, " 5"] = jnp.asarray(
            [convolved_density(float(energy)) for energy in energy_axis],
            dtype=jnp.float64,
        )
        assert jnp.allclose(actual, expected, rtol=1e-6, atol=1e-10)

    def test_integrated_weight_equals_number_of_bands(self) -> None:
        """Recover the Gaussian DOS sum rule on a tail-complete window.

        Two bands must contribute exactly two states after k-weight
        normalization.

        Notes
        -----
        Integrate a dense energy grid with the trapezoidal rule.
        """
        n_k: int = 512
        momenta: Float64[Array, " n_k"] = (
            jnp.arange(n_k, dtype=jnp.float64) / n_k - 0.5
        )
        first: Float64[Array, " n_k"] = 2.0 * jnp.cos(2.0 * jnp.pi * momenta)
        eigenvalues: Float64[Array, "n_k 2"] = jnp.stack(
            (first, -0.7 * first), axis=-1
        )
        weights: Float64[Array, " n_k"] = jnp.full(
            n_k, 1.0 / n_k, dtype=jnp.float64
        )
        energy_axis: Float64[Array, " 8001"] = jnp.linspace(-4.0, 4.0, 8001)
        dos: Float64[Array, " 8001"] = dos_gaussian(
            eigenvalues,
            weights,
            energy_axis,
            0.1,
        ).total_dos
        integrated: Float64[Array, ""] = jnp.trapezoid(dos, energy_axis)
        assert float(integrated) == pytest.approx(2.0, rel=1e-8)

    def test_jit_and_parameter_gradient(self) -> None:
        """Preserve values under JIT and differentiate through band energies.

        A scaled asymmetric spectrum exercises every Gaussian displacement.

        Notes
        -----
        Apply the shared gradient check and compare eager and compiled arrays.
        """
        base: Float64[Array, "2 2"] = jnp.asarray(
            [[-1.2, 0.3], [-0.4, 1.1]],
            dtype=jnp.float64,
        )
        weights: Float64[Array, " 2"] = jnp.asarray(
            [0.4, 0.6], dtype=jnp.float64
        )
        axis: Float64[Array, " 101"] = jnp.linspace(-2.0, 2.0, 101)

        def loss(scale: Float64[Array, ""]) -> Float64[Array, ""]:
            result: Float64[Array, " 101"] = dos_gaussian(
                scale * base,
                weights,
                axis,
                0.18,
            ).total_dos
            total: Float64[Array, ""] = jnp.sum(
                result * jnp.linspace(0.2, 1.1, result.shape[0])
            )
            return total

        scale: Float64[Array, ""] = jnp.asarray(0.9, dtype=jnp.float64)
        assert_gradients_match_finite_differences(loss, scale, regime="smooth")
        eager: Float64[Array, " 101"] = dos_gaussian(
            base, weights, axis, 0.18
        ).total_dos
        compiled: Float64[Array, " 101"] = eqx.filter_jit(dos_gaussian)(
            base,
            weights,
            axis,
            0.18,
        ).total_dos
        assert jnp.allclose(compiled, eager, rtol=1e-14, atol=1e-14)

    @pytest.mark.parametrize(
        ("weights", "diagnostic"),
        [
            ([0.0, 0.0], "sum to one"),
            ([-0.1, 1.1], "nonnegative"),
            ([0.4, 0.4], "sum to one"),
            ([0.5, np.inf], "finite"),
        ],
    )
    def test_rejects_invalid_weight_measures(
        self,
        weights: List[float],
        diagnostic: str,
    ) -> None:
        """Reject zero, negative, unnormalized, and non-finite weights.

        Exercise each invalid measure through the public DOS function.

        Notes
        -----
        Match each input with its declared validation diagnostic.
        """
        eigenvalues: Float64[Array, "2 2"] = jnp.asarray(
            [[-1.0, 1.0], [-0.5, 0.5]],
            dtype=jnp.float64,
        )
        axis: Float64[Array, " 21"] = jnp.linspace(-2.0, 2.0, 21)

        with pytest.raises(RuntimeError, match=diagnostic):
            dos_gaussian(
                eigenvalues,
                jnp.asarray(weights, dtype=jnp.float64),
                axis,
                0.1,
            )


class TestFermiLevelFromFilling:
    """Validate :func:`diffpes.tightb.fermi_level_from_filling`.

    The cases check analytic fillings, implicit gradients, endpoint
    convergence, normalization, and validation.
    """

    def test_particle_hole_symmetric_chain_and_graphene(self) -> None:
        """Place the half-filled chemical potential at zero for both models.

        Chain and graphene spectra provide independent particle-hole symmetric
        inputs.

        Notes
        -----
        Solve both weighted fillings and compare their roots with zero.
        """
        n_k: int = 256
        momenta: Float64[Array, " 256"] = (
            jnp.arange(n_k, dtype=jnp.float64) / n_k - 0.5
        )
        chain: Float64[Array, "256 1"] = (
            -2.0 * jnp.cos(2.0 * jnp.pi * momenta)
        )[:, None]
        chain_weights: Float64[Array, " 256"] = jnp.full(
            n_k,
            1.0 / n_k,
            dtype=jnp.float64,
        )
        chain_mu: Float64[Array, ""] = fermi_level_from_filling(
            chain,
            chain_weights,
            0.5,
            300.0,
        )

        # A multiple-of-three mesh contains graphene's Dirac points. This
        # avoids the finite-temperature filling residual becoming numerically
        # flat throughout a discretization-induced gap around zero.
        grid_axis: Float64[Array, " 18"] = (
            jnp.arange(18, dtype=jnp.float64) / 18.0
        )
        kx: Float64[Array, "18 18"]
        ky: Float64[Array, "18 18"]
        kx, ky = jnp.meshgrid(grid_axis, grid_axis, indexing="ij")
        graphene_kpoints: Float64[Array, "324 3"] = jnp.stack(
            (
                jnp.ravel(kx),
                jnp.ravel(ky),
                jnp.zeros(kx.size, dtype=jnp.float64),
            ),
            axis=-1,
        )
        graphene: Float64[Array, "324 2"] = eigvalsh_bands(
            make_graphene_model(),
            graphene_kpoints,
        )
        graphene_weights: Float64[Array, " 324"] = jnp.full(
            graphene.shape[0],
            1.0 / graphene.shape[0],
            dtype=jnp.float64,
        )
        graphene_mu: Float64[Array, ""] = fermi_level_from_filling(
            graphene,
            graphene_weights,
            1.0,
            300.0,
        )

        assert float(chain_mu) == pytest.approx(0.0, abs=1e-8)
        assert float(graphene_mu) == pytest.approx(0.0, abs=1e-8)

    def test_hopping_gradient_matches_finite_difference(self) -> None:
        """Differentiate a non-half-filled chain chemical potential.

        Moving away from half filling gives a nontrivial hopping derivative.

        Notes
        -----
        Apply the shared smooth gradient check to the implicit root.
        """
        momenta: Float64[Array, " 65"] = jnp.linspace(
            -0.5, 0.5, 65, endpoint=False
        )
        weights: Float64[Array, " 65"] = jnp.full(
            momenta.shape[0],
            1.0 / momenta.shape[0],
            dtype=jnp.float64,
        )

        def chemical_potential(
            hopping: Float64[Array, ""],
        ) -> Float64[Array, ""]:
            eigenvalues: Float64[Array, "65 1"] = (
                2.0 * hopping * jnp.cos(2.0 * jnp.pi * momenta)
            )[:, None]
            value: Float64[Array, ""] = fermi_level_from_filling(
                eigenvalues,
                weights,
                0.7,
                300.0,
            )
            return value

        assert_gradients_match_finite_differences(
            chemical_potential,
            jnp.asarray(-1.1, dtype=jnp.float64),
            regime="smooth",
            atol=2e-7,
        )

    def test_implicit_gradient_is_stable_to_tighter_solver_tolerance(
        self,
    ) -> None:
        """Keep the implicit derivative fixed when tolerance tightens tenfold.

        The same validated filling problem must produce stable implicit
        sensitivities.

        Notes
        -----
        Differentiate roots from two tolerances and compare them directly.
        """
        momenta: Float64[Array, " 65"] = jnp.linspace(
            -0.5, 0.5, 65, endpoint=False
        )
        weights: Float64[Array, " 65"] = jnp.full(
            momenta.shape[0],
            1.0 / momenta.shape[0],
            dtype=jnp.float64,
        )
        temperature: Float64[Array, ""] = jnp.asarray(300.0, dtype=jnp.float64)
        filling: Float64[Array, ""] = jnp.asarray(0.7, dtype=jnp.float64)

        def solve(
            hopping: Float64[Array, ""], tolerance: float
        ) -> Float64[Array, ""]:
            eigenvalues: Float64[Array, "65 1"] = (
                2.0 * hopping * jnp.cos(2.0 * jnp.pi * momenta)
            )[:, None]
            padding: Float64[Array, ""] = jnp.asarray(2.0, dtype=jnp.float64)
            chemical_potential: Float64[Array, ""] = _solve_filling(
                (eigenvalues, weights, temperature, filling),
                jnp.min(eigenvalues) - padding,
                jnp.max(eigenvalues) + padding,
                tolerance=tolerance,
            )
            return chemical_potential

        hopping: Float64[Array, ""] = jnp.asarray(-1.1, dtype=jnp.float64)
        default_gradient: Float64[Array, ""] = jax.grad(
            lambda value: solve(value, 1e-12)
        )(hopping)
        tighter_gradient: Float64[Array, ""] = jax.grad(
            lambda value: solve(value, 1e-13)
        )(hopping)
        assert jnp.allclose(
            default_gradient,
            tighter_gradient,
            rtol=1e-10,
            atol=1e-12,
        )

    def test_rejects_invalid_domain_eager_and_jit(self) -> None:
        """Reject unnormalized weights, invalid filling, and temperature.

        Both eager and compiled paths must preserve numerical domain checks.

        Notes
        -----
        Match separate diagnostics for capacity, temperature, and weight sum.
        """
        eigenvalues: Float64[Array, "2 2"] = jnp.asarray(
            [[-1.0, 1.0], [-0.5, 0.5]],
            dtype=jnp.float64,
        )
        weights: Float64[Array, " 2"] = jnp.asarray(
            [0.5, 0.5], dtype=jnp.float64
        )
        with pytest.raises(RuntimeError, match="band capacity"):
            fermi_level_from_filling(eigenvalues, weights, 2.0, 300.0)
        with pytest.raises(RuntimeError, match="temperature_k"):
            eqx.filter_jit(fermi_level_from_filling)(
                eigenvalues,
                weights,
                1.0,
                0.0,
            )
        with pytest.raises(RuntimeError, match="sum to one"):
            fermi_level_from_filling(
                eigenvalues,
                jnp.asarray([0.3, 0.3], dtype=jnp.float64),
                1.0,
                300.0,
            )

    @pytest.mark.parametrize("filling", [-0.1, 0.0, 2.0, 2.1, np.inf])
    def test_rejects_out_of_range_counts(self, filling: float) -> None:
        """Reject non-finite counts and both closed capacity endpoints.

        Exercise invalid fillings against one normalized k-point measure.

        Notes
        -----
        Require the band-capacity diagnostic for every invalid count.
        """
        eigenvalues: Float64[Array, "2 2"] = jnp.asarray(
            [[-1.0, 1.0], [-0.5, 0.5]],
            dtype=jnp.float64,
        )
        weights: Float64[Array, " 2"] = jnp.asarray(
            [0.25, 0.75], dtype=jnp.float64
        )

        with pytest.raises(RuntimeError, match="band capacity"):
            fermi_level_from_filling(
                eigenvalues,
                weights,
                filling,
                300.0,
            )

    def test_analytic_bracket_handles_near_endpoint_filling(self) -> None:
        """Verify a small positive filling without heuristic expansion.

        The analytic logits bracket the monotone count while holding the
        normalized weights, positive temperature, and spectrum fixed.

        Notes
        -----
        Reconstruct the count and compare it with the requested filling.
        """
        eigenvalues: Float64[Array, "2 2"] = jnp.asarray(
            [[-2.0, 0.5], [-0.2, 1.7]],
            dtype=jnp.float64,
        )
        weights: Float64[Array, " 2"] = jnp.asarray(
            [0.3, 0.7], dtype=jnp.float64
        )
        filling: float = 1e-12
        temperature: float = 120.0
        chemical_potential: Float64[Array, ""] = fermi_level_from_filling(
            eigenvalues,
            weights,
            filling,
            temperature,
        )
        occupations: Float64[Array, "2 2"] = jax.nn.sigmoid(
            -(eigenvalues - chemical_potential)
            / (8.617333262145e-5 * temperature)
        )
        count: Float64[Array, ""] = jnp.sum(weights[:, None] * occupations)

        assert float(count) == pytest.approx(filling, rel=2e-6, abs=1e-20)

    def test_open_endpoint_grid_converges_at_both_capacity_edges(self) -> None:
        """Verify convergence at both open capacity edges.

        A logarithmic deficit grid includes the former high-filling Newton
        failure. Normalized weights, spectrum, and temperature are held fixed.

        Notes
        -----
        Compare reconstructed counts across both open capacity edges.
        """
        eigenvalues: Float64[Array, "2 2"] = jnp.asarray(
            [[-2.0, 0.5], [-0.2, 1.7]],
            dtype=jnp.float64,
        )
        weights: Float64[Array, " 2"] = jnp.asarray(
            [0.3, 0.7], dtype=jnp.float64
        )
        deficits: Float64[Array, " 4"] = jnp.asarray(
            [1e-5, 1e-8, 1e-11, 1e-13],
            dtype=jnp.float64,
        )
        fillings: Float64[Array, " 8"] = jnp.concatenate(
            (deficits, 2.0 - deficits)
        )
        temperature: float = 120.0
        chemical_potentials: Float64[Array, " 8"] = jax.vmap(
            lambda filling: fermi_level_from_filling(
                eigenvalues,
                weights,
                filling,
                temperature,
            )
        )(fillings)
        occupations: Float64[Array, "8 2 2"] = jax.nn.sigmoid(
            -(eigenvalues[None, :, :] - chemical_potentials[:, None, None])
            / (8.617333262145e-5 * temperature)
        )
        counts: Float64[Array, " 8"] = jnp.sum(
            weights[None, :, None] * occupations,
            axis=(1, 2),
        )

        assert jnp.all(jnp.isfinite(chemical_potentials))
        assert jnp.allclose(counts, fillings, rtol=2e-6, atol=2e-14)

    def test_near_normalized_weights_are_normalized_before_capacity(
        self,
    ) -> None:
        """Normalize an accepted near-one measure before solving near capacity.

        The raw weight sum is five times ``1e-13`` below one, within the
        declared validation tolerance. The requested filling exceeds the raw
        measure capacity. Successful solution confirms normalization before
        the function defines the two-band capacity.

        Notes
        -----
        Compare the solved count with the near-capacity requested filling.
        """
        eigenvalues: Float64[Array, "2 2"] = jnp.asarray(
            [[-1.0, 0.4], [-0.3, 1.2]],
            dtype=jnp.float64,
        )
        weights: Float64[Array, " 2"] = jnp.asarray(
            [0.5, 0.5 - 5e-13],
            dtype=jnp.float64,
        )
        filling: float = 2.0 - 1e-13
        temperature: float = 200.0
        chemical_potential: Float64[Array, ""] = fermi_level_from_filling(
            eigenvalues,
            weights,
            filling,
            temperature,
        )
        normalized_weights: Float64[Array, " 2"] = weights / jnp.sum(weights)
        occupations: Float64[Array, "2 2"] = jax.nn.sigmoid(
            -(eigenvalues - chemical_potential)
            / (8.617333262145e-5 * temperature)
        )
        count: Float64[Array, ""] = jnp.sum(
            normalized_weights[:, None] * occupations
        )

        assert jnp.isfinite(chemical_potential)
        assert float(count) == pytest.approx(filling, rel=0.0, abs=2e-14)
