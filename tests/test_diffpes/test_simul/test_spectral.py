"""Validate the spectral module.

The cases use analytic values, invariants, and finite differences.
"""

from __future__ import annotations

import chex
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from absl.testing import parameterized
from beartype.typing import Any, Callable, Dict, List, Tuple
from jaxtyping import Array, Bool, Complex128, Float64
from numpy.typing import NDArray

from diffpes.constants import (
    EPS_DEG,
    KB_EV_PER_K,
)
from diffpes.simul import (
    assemble_spectral_intensity_bands_chunk,
    assemble_spectral_intensity_chunk,
)
from diffpes.simul.spectral import _stream_spectral_intensity
from diffpes.types import (
    SelfEnergyModel,
    TransitionSourceSchedule,
    make_final_state_spec,
    make_matrix_element_params,
    make_orbital_basis,
    make_radial_quadrature_spec,
    make_radial_spec,
    make_self_energy_model,
    make_transition_source_schedule,
)
from tests._assertions import assert_rejects

from ._spectral_helpers import (
    _CHINOOK_SPECTRAL_ARCHIVE_PATH,
    _CHINOOK_SPECTRAL_ARCHIVE_SHA256,
    _CHINOOK_SPECTRAL_MANIFEST_PATH,
    _CHINOOK_SPECTRAL_MANIFEST_SHA256,
    _authenticated_json,
    _authenticated_npz,
    _spectral_intensity_reference,
)


class TestAssembleSpectralIntensityChunk(chex.TestCase):
    """Validate :func:`~diffpes.simul.assemble_spectral_intensity_chunk`.

    The cases check analytic composition, one Fermi shift, source validation,
    vectorization, and gradients through the self-energy and resolvent.
    """

    @staticmethod
    def _fixture() -> Tuple[
        Complex128[Array, "2 2 2"],
        Complex128[Array, "2 5 2 2"],
        Float64[Array, " 5"],
        SelfEnergyModel,
        Float64[Array, ""],
    ]:
        """PRIVATE: Return one generic two-k coherent assembly fixture.

        Notes
        -----
        The absolute Hamiltonians share a nonzero Fermi offset. Sources vary
        with sampled energy so no early matrix-element reduction can pass.
        """
        fermi_energy: Float64[Array, ""] = jnp.asarray(1.7)
        relative: Complex128[Array, "2 2 2"] = jnp.asarray(
            [
                [[-0.3, 0.08 + 0.03j], [0.08 - 0.03j, 0.2]],
                [[-0.1, -0.05 + 0.07j], [-0.05 - 0.07j, 0.35]],
            ],
            dtype=jnp.complex128,
        )
        hamiltonians: Complex128[Array, "2 2 2"] = relative + (
            fermi_energy * jnp.eye(2, dtype=jnp.complex128)[None, :, :]
        )
        omega: Float64[Array, " 5"] = jnp.asarray(
            [-0.35, -0.12, 0.0, 0.24, 2.0]
        )
        source_base: Complex128[Array, "2 2 2"] = jnp.asarray(
            [
                [[0.8 + 0.2j, -0.3 + 0.5j], [0.1 - 0.4j, 0.6 + 0.2j]],
                [[0.2 - 0.6j, 0.7 + 0.1j], [-0.5 + 0.3j, 0.2 + 0.8j]],
            ]
        )
        scales: Complex128[Array, " 5"] = jnp.asarray(
            [1.0, 0.8 + 0.1j, 1.1 - 0.2j, 0.6 + 0.3j, 0.9 - 0.1j]
        )
        sources: Complex128[Array, "2 5 2 2"] = (
            source_base[:, None, :, :] * scales[None, :, None, None]
        )
        model: SelfEnergyModel = make_self_energy_model(gamma=0.04)
        returned: Tuple[
            Complex128[Array, "2 2 2"],
            Complex128[Array, "2 5 2 2"],
            Float64[Array, " 5"],
            SelfEnergyModel,
            Float64[Array, ""],
        ] = hamiltonians, sources, omega, model, fermi_energy
        return returned

    def test_analytic_composition_and_single_fermi_shift(self) -> None:
        """Match a dense NumPy-style composition and one energy shift.

        The test also proves invariance under a common absolute-energy offset.

        Notes
        -----
        The independent expression uses ``jnp.linalg.solve`` and an analytic
        sigmoid. Shifting both absolute H and E_F by the same constant leaves
        the relative observable unchanged bit for bit within roundoff.
        """
        hamiltonians: Complex128[Array, "2 2 2"]
        sources: Complex128[Array, "2 5 2 2"]
        omega: Float64[Array, " 5"]
        model: SelfEnergyModel
        fermi_energy: Float64[Array, ""]
        hamiltonians, sources, omega, model, fermi_energy = self._fixture()
        temperature: float = 18.0
        eta: float = 2.0e-4
        actual: Float64[Array, "2 5"] = jax.jit(
            assemble_spectral_intensity_chunk
        )(
            hamiltonians,
            sources,
            omega,
            model,
            fermi_energy,
            temperature,
            eta,
        )
        sigma: Complex128[Array, ""] = jnp.asarray(-0.04j)
        relative: Complex128[Array, "2 2 2"] = hamiltonians - (
            fermi_energy * jnp.eye(2, dtype=jnp.complex128)[None, :, :]
        )

        def one(
            hamiltonian: Complex128[Array, "2 2"],
            sources_at_sample: Complex128[Array, "n_out 2"],
            sampled: Float64[Array, ""],
        ) -> Float64[Array, ""]:
            """Sum independent dense solves for one sample."""
            operator: Complex128[Array, "2 2"] = (
                sampled + 1.0j * eta - sigma
            ) * jnp.eye(2, dtype=jnp.complex128) - hamiltonian
            spectral: Float64[Array, ""] = jnp.sum(
                jax.vmap(
                    lambda source: (
                        -jnp.imag(
                            jnp.vdot(
                                source, jnp.linalg.solve(operator, source)
                            )
                        )
                        / jnp.pi
                    )
                )(sources_at_sample)
            )
            occupation: Float64[Array, ""] = jax.nn.sigmoid(
                -sampled / (KB_EV_PER_K * temperature)
            )
            returned: Float64[Array, ""] = spectral * occupation
            return returned

        expected: Float64[Array, "2 5"] = jax.vmap(
            lambda hamiltonian, source_row: jax.vmap(
                one, in_axes=(None, 0, 0)
            )(hamiltonian, source_row, omega)
        )(relative, sources)
        np.testing.assert_allclose(
            np.asarray(actual),
            np.asarray(expected),
            rtol=1.0e-12,
            atol=1.0e-13,
        )
        shifted: Float64[Array, "2 5"] = assemble_spectral_intensity_chunk(
            hamiltonians + 3.25 * jnp.eye(2)[None, :, :],
            sources,
            omega,
            model,
            fermi_energy + 3.25,
            temperature,
            eta,
        )
        np.testing.assert_allclose(
            np.asarray(shifted),
            np.asarray(actual),
            rtol=1.0e-12,
            atol=1.0e-13,
        )

    def test_outgoing_source_axis_must_be_nonempty(self) -> None:
        """Reject an empty output axis on the public chunk boundary.

        The chunk assembler must retain at least one incoherent source channel.

        Notes
        -----
        A Python guard rejects the zero-length axis before tracing a batched
        solve.
        """
        hamiltonians: Complex128[Array, "2 2 2"]
        omega: Float64[Array, " 5"]
        model: SelfEnergyModel
        fermi_energy: Float64[Array, ""]
        hamiltonians, _, omega, model, fermi_energy = self._fixture()
        with pytest.raises(ValueError, match="n_out|nonempty"):
            assemble_spectral_intensity_chunk(
                hamiltonians,
                jnp.empty((2, 5, 0, 2), dtype=jnp.complex128),
                omega,
                model,
                fermi_energy,
                18.0,
            )

    def test_temperature_eta_gradients_and_vmap(self) -> None:
        """Match complete-assembly gradients to finite differences and vmap.

        Temperature and regulator derivatives traverse the full composition.

        Notes
        -----
        The frozen axis includes a ``+2 eV`` sample at 15 K, pinning the
        former overflow-NaN regime while other samples retain nonzero thermal
        sensitivity.
        """
        hamiltonians: Complex128[Array, "2 2 2"]
        sources: Complex128[Array, "2 5 2 2"]
        omega: Float64[Array, " 5"]
        model: SelfEnergyModel
        fermi_energy: Float64[Array, ""]
        hamiltonians, sources, omega, model, fermi_energy = self._fixture()

        def loss_temperature(
            temperature: Float64[Array, ""],
        ) -> Float64[Array, ""]:
            """Sum the assembly at one traced temperature."""
            returned: Float64[Array, ""] = jnp.sum(
                assemble_spectral_intensity_chunk(
                    hamiltonians,
                    sources,
                    omega,
                    model,
                    fermi_energy,
                    temperature,
                    0.01,
                )
            )
            return returned

        temperature: Float64[Array, ""] = jnp.asarray(15.0)
        temperature_step: float = 2.0**-12
        temperature_grad: Float64[Array, ""] = jax.grad(loss_temperature)(
            temperature
        )
        temperature_fd: Float64[Array, ""] = (
            loss_temperature(temperature + temperature_step)
            - loss_temperature(temperature - temperature_step)
        ) / (2.0 * temperature_step)
        np.testing.assert_allclose(
            np.asarray(temperature_grad),
            np.asarray(temperature_fd),
            rtol=1.0e-6,
            atol=1.0e-9,
        )
        assert bool(jnp.isfinite(temperature_grad))

        def loss_eta(eta: Float64[Array, ""]) -> Float64[Array, ""]:
            """Sum the assembly at one traced regulator."""
            returned: Float64[Array, ""] = jnp.sum(
                assemble_spectral_intensity_chunk(
                    hamiltonians,
                    sources,
                    omega,
                    model,
                    fermi_energy,
                    15.0,
                    eta,
                )
            )
            return returned

        eta: Float64[Array, ""] = jnp.asarray(0.01)
        eta_step: float = 2.0**-16
        eta_grad: Float64[Array, ""] = jax.grad(loss_eta)(eta)
        eta_fd: Float64[Array, ""] = (
            loss_eta(eta + eta_step) - loss_eta(eta - eta_step)
        ) / (2.0 * eta_step)
        np.testing.assert_allclose(
            np.asarray(eta_grad),
            np.asarray(eta_fd),
            rtol=1.0e-6,
            atol=1.0e-8,
        )
        temperatures: Float64[Array, " 2"] = jnp.asarray([15.0, 22.0])
        batched: Float64[Array, "2 2 5"] = jax.jit(
            jax.vmap(
                lambda value: assemble_spectral_intensity_chunk(
                    hamiltonians,
                    sources,
                    omega,
                    model,
                    fermi_energy,
                    value,
                    0.01,
                )
            )
        )(temperatures)
        assert batched.shape == (2, 2, 5)
        assert bool(jnp.all(jnp.isfinite(batched)))

    @pytest.mark.big_mem
    @pytest.mark.rss_limit_mb(1200)
    def test_poly_coefficient_gradient_through_kk_and_resolvent(self) -> None:
        """Match self-energy gradients through a polynomial KK map.

        The test differentiates raw self-energy coordinates through all layers.

        Notes
        -----
        Every raw polynomial coordinate remains connected to the scalar loss;
        a central difference checks the generic linear coefficient.
        """
        hamiltonians: Complex128[Array, "2 2 2"]
        sources: Complex128[Array, "2 5 2 2"]
        fermi_energy: Float64[Array, ""]
        hamiltonians, sources, _, _, fermi_energy = self._fixture()
        omega: Float64[Array, " 2"] = jnp.asarray([-0.2, 0.13])
        source_subset: Complex128[Array, "2 2 2 2"] = sources[:, :2, :, :]
        base: Float64[Array, " 3"] = jnp.asarray([-1.4, 0.25, -0.8])

        def loss(coefficients: Float64[Array, " 3"]) -> Float64[Array, ""]:
            """Assemble a scalar loss from one poly model."""
            model: SelfEnergyModel = make_self_energy_model(
                coefficients=coefficients,
                mode="poly",
                kk_consistent=True,
                kk_domain_rel_fermi_ev=jnp.asarray([-4.0, 4.0]),
                tail_coefficients=jnp.asarray([-2.0, -1.7]),
                subtraction_point_rel_fermi_ev=0.0,
                tail_mode="power2",
            )
            returned: Float64[Array, ""] = jnp.sum(
                assemble_spectral_intensity_chunk(
                    hamiltonians,
                    source_subset,
                    omega,
                    model,
                    fermi_energy,
                    20.0,
                    1.0e-3,
                )
            )
            return returned

        gradient: Float64[Array, " 3"] = jax.grad(loss)(base)
        assert bool(jnp.all(jnp.isfinite(gradient)))
        assert bool(jnp.all(jnp.abs(gradient) > 1.0e-8))
        step: float = 2.0**-14
        direction: Float64[Array, " 3"] = jnp.asarray([0.0, 1.0, 0.0])
        finite_difference: Float64[Array, ""] = (
            loss(base + step * direction) - loss(base - step * direction)
        ) / (2.0 * step)
        np.testing.assert_allclose(
            np.asarray(gradient[1]),
            np.asarray(finite_difference),
            rtol=1.0e-6,
            atol=1.0e-8,
        )


class TestAssembleSpectralIntensityBandsChunk(chex.TestCase):
    """Validate chunked spectral-intensity assembly.

    The class covers sampled-energy occupation, gradients, and batching.

    :see: :func:`~diffpes.simul.assemble_spectral_intensity_bands_chunk`
    """

    def test_sampled_omega_fermi_counterexample(self) -> None:
        """Match the frozen sampled-energy occupation witness.

        The counterexample distinguishes sampled omega from band-energy Fermi
        use.

        Notes
        -----
        A band above the Fermi level retains its occupied-side Lorentzian tail
        only with sampled-omega occupation. The tiny positive eta approximates
        the preregistered eta-free analytic row.
        """
        reference: Dict[str, Float64[NDArray, "..."]] = (
            _spectral_intensity_reference()
        )
        omega: Float64[Array, " n"] = jnp.asarray(reference["fermi_omega"])
        eigenvalues: Float64[Array, "1 1"] = jnp.asarray(
            [[reference["fermi_band_energy"]]]
        )
        weights: Float64[Array, "1 n 1"] = jnp.ones(
            (1, omega.shape[0], 1), dtype=jnp.float64
        )
        model: SelfEnergyModel = make_self_energy_model(
            gamma=float(reference["fermi_gamma"])
        )
        actual: Float64[Array, "1 n"] = jax.jit(
            assemble_spectral_intensity_bands_chunk
        )(
            eigenvalues,
            weights,
            omega,
            model,
            jnp.asarray(0.0),
            float(reference["fermi_temperature_k"]),
            1.0e-12,
        )
        np.testing.assert_allclose(
            np.asarray(actual[0]),
            reference["fermi_intensity_correct"],
            rtol=1.0e-9,
            atol=1.0e-12,
        )
        occupied_index: int = int(jnp.argmin(jnp.abs(omega + 0.1)))
        assert actual[0, occupied_index] > (
            1.0e12 * reference["fermi_intensity_wrong"][occupied_index]
        )

    def test_resolvent_and_band_chunk_paths_agree(self) -> None:
        """Match both chunk assemblers on generic nondegenerate bands.

        The test compares the coherent resolvent and invariant-weight formulas.

        Notes
        -----
        Explicit eigendecomposition forms invariant source weights at every
        sampled energy; the two public assemblies then share only self-energy
        evaluation and the final Fermi factor.
        """
        hamiltonians: Complex128[Array, "2 2 2"]
        sources: Complex128[Array, "2 5 2 2"]
        omega: Float64[Array, " 5"]
        model: SelfEnergyModel
        fermi_energy: Float64[Array, ""]
        hamiltonians, sources, omega, model, fermi_energy = self._fixture()
        eigenvalues: Float64[Array, "2 2"]
        eigenvectors: Complex128[Array, "2 2 2"]
        eigenvalues, eigenvectors = jax.vmap(jnp.linalg.eigh)(hamiltonians)
        weights: Float64[Array, "2 5 2"] = jnp.sum(
            jnp.abs(
                jnp.einsum(
                    "kob,keao->keab",
                    eigenvectors.conj(),
                    sources,
                )
            )
            ** 2,
            axis=2,
        )
        resolvent: Float64[Array, "2 5"] = assemble_spectral_intensity_chunk(
            hamiltonians,
            sources,
            omega,
            model,
            fermi_energy,
            19.0,
            2.0e-4,
        )
        bands: Float64[Array, "2 5"] = assemble_spectral_intensity_bands_chunk(
            eigenvalues,
            weights,
            omega,
            model,
            fermi_energy,
            19.0,
            2.0e-4,
        )
        np.testing.assert_allclose(
            np.asarray(bands),
            np.asarray(resolvent),
            rtol=1.0e-10,
            atol=1.0e-12,
        )

    def test_degenerate_rows_require_explicit_value_only_mode(self) -> None:
        """Prevent the chunk assembler from bypassing the eigen gap policy.

        Exact-degenerate rows reject unless the caller selects primal-only use.

        Notes
        -----
        Both the default jitted call and the explicit value-only success path
        run.
        """
        eigenvalues: Float64[Array, "1 2"] = jnp.zeros((1, 2))
        weights: Float64[Array, "1 1 2"] = jnp.asarray([[[0.7, 0.4]]])
        omega: Float64[Array, " 1"] = jnp.asarray([0.03])
        model: SelfEnergyModel = make_self_energy_model(gamma=0.02)
        assert_rejects(
            assemble_spectral_intensity_bands_chunk,
            eigenvalues,
            weights,
            omega,
            model,
            jnp.asarray(0.0),
            20.0,
            1.0e-4,
            match="gap|resolvent|value_only",
        )
        value_only: Float64[Array, "1 1"] = jax.jit(
            lambda values, candidate_weights: (
                assemble_spectral_intensity_bands_chunk(
                    values,
                    candidate_weights,
                    omega,
                    model,
                    jnp.asarray(0.0),
                    20.0,
                    1.0e-4,
                    allow_degenerate_value_only=True,
                )
            )
        )(eigenvalues, weights)
        assert bool(jnp.all(jnp.isfinite(value_only)))

    def test_frozen_chinook_spectral_comparison(self) -> None:
        """Match the frozen Chinook cube after its rounded-kB convention.

        The full imported k-energy cube exercises the value-only Dirac row.

        Notes
        -----
        Chinook 1.1.3 uses the rounded ratio ``1.38e-23 / 1.602e-19`` eV/K.
        Matching thermal energy therefore uses a documented effective Kelvin
        coordinate. The analytic sampled-Fermi test above separately owns
        physical correctness with DiffPES's types-owned Boltzmann constant.
        """
        manifest: Dict[str, Any] = _authenticated_json(
            _CHINOOK_SPECTRAL_MANIFEST_PATH,
            _CHINOOK_SPECTRAL_MANIFEST_SHA256,
        )
        assert manifest["schema"] == "diffpes.chinook-spectral-reference.v1"
        reference: Dict[str, Float64[NDArray, "..."]] = _authenticated_npz(
            _CHINOOK_SPECTRAL_ARCHIVE_PATH,
            _CHINOOK_SPECTRAL_ARCHIVE_SHA256,
        )
        eigenvalues_np: Float64[NDArray, "n_k n_band"] = np.asarray(
            reference["band_energies_k_band_ev"]
        )
        minimum_gap: float = float(
            np.min(np.diff(np.sort(eigenvalues_np, axis=-1), axis=-1))
        )
        assert minimum_gap < 1.0e3 * EPS_DEG
        band_weight_np: Float64[NDArray, "n_k n_band"] = np.zeros_like(
            eigenvalues_np
        )
        matrix_factor: float
        state: Float64[NDArray, " n_state_field"]
        for matrix_factor, state in zip(
            reference["m_factor_state"],
            reference["pks_state"],
            strict=True,
        ):
            row: int = int(state[1])
            column: int = int(state[2])
            flat_k: int = row * 31 + column
            band: int = int(
                np.argmin(np.abs(eigenvalues_np[flat_k] - state[3]))
            )
            assert abs(eigenvalues_np[flat_k, band] - state[3]) < 1.0e-10
            band_weight_np[flat_k, band] += matrix_factor
        omega: Float64[Array, " n_energy"] = jnp.asarray(
            reference["omega_rel_ev"]
        )
        weights_np: Float64[NDArray, "n_k n_energy n_band"] = np.broadcast_to(
            band_weight_np[:, None, :],
            (961, omega.shape[0], 2),
        ).copy()
        chinook_kb_ev_per_k: float = 1.38e-23 / 1.602e-19
        effective_temperature: float = 4.2 * chinook_kb_ev_per_k / KB_EV_PER_K
        actual: Float64[Array, "961 n_energy"] = (
            assemble_spectral_intensity_bands_chunk(
                jnp.asarray(eigenvalues_np),
                jnp.asarray(weights_np),
                omega,
                make_self_energy_model(gamma=0.02),
                jnp.asarray(0.0),
                effective_temperature,
                5.0e-5,
                allow_degenerate_value_only=True,
            )
        )
        expected: Float64[NDArray, "n_k n_energy"] = np.asarray(
            reference["intensity_raw"]
        ).reshape(961, omega.shape[0])
        np.testing.assert_allclose(
            np.asarray(actual),
            expected,
            rtol=1.0e-6,
            atol=0.0,
        )

    def test_jit_vmap_and_weight_gradient(self) -> None:
        """Exercise JIT, VMAP, and a nonzero band-weight gradient.

        The test batches complete weight fields through the public assembler.

        Notes
        -----
        VMAP batches two weight fields while the public function owns its
        native k and energy axes. The scalar loss differentiates every weight.
        """
        eigenvalues: Float64[Array, "2 2"] = jnp.asarray(
            [[-0.2, 0.3], [-0.1, 0.45]]
        )
        omega: Float64[Array, " 3"] = jnp.asarray([-0.25, -0.05, 0.2])
        weights: Float64[Array, "2 3 2"] = jnp.asarray(
            [
                [[0.8, 0.2], [0.7, 0.3], [0.6, 0.4]],
                [[0.3, 0.9], [0.4, 0.8], [0.5, 0.7]],
            ]
        )
        model: SelfEnergyModel = make_self_energy_model(gamma=0.03)

        def assemble(
            candidate: Float64[Array, "2 3 2"],
        ) -> Float64[Array, "2 3"]:
            """Assemble one member of a weight-field batch."""
            returned: Float64[Array, "2 3"] = (
                assemble_spectral_intensity_bands_chunk(
                    eigenvalues,
                    candidate,
                    omega,
                    model,
                    jnp.asarray(0.0),
                    25.0,
                    1.0e-3,
                )
            )
            return returned

        batched_weights: Float64[Array, "2 2 3 2"] = jnp.stack(
            [weights, 1.2 * weights]
        )
        batched: Float64[Array, "2 2 3"] = jax.jit(jax.vmap(assemble))(
            batched_weights
        )
        gradient: Float64[Array, "2 3 2"] = jax.grad(
            lambda candidate: jnp.sum(assemble(candidate))
        )(weights)
        assert batched.shape == (2, 2, 3)
        assert bool(jnp.all(jnp.isfinite(batched)))
        assert bool(jnp.all(jnp.isfinite(gradient)))
        assert bool(jnp.all(gradient > 0.0))


class TestStreamSpectralIntensity(chex.TestCase):
    """Validate the private padded spectral scan owner.

    The cases compare checkpointed values and gradients, count traces, apply
    block masks, and reject invalid active final momenta.
    """

    @staticmethod
    def _fixture() -> Tuple[
        Complex128[Array, "4 2 2"],
        TransitionSourceSchedule,
        Float64[Array, " 8"],
        Bool[Array, " 4"],
        Bool[Array, " 8"],
        SelfEnergyModel,
    ]:
        """PRIVATE: Return one padded two-by-two chunk schedule.

        Notes
        -----
        Masks exclude the final k row and final two omega columns as padding.
        """
        base: Complex128[Array, "2 2"] = jnp.asarray(
            [[-0.2, 0.07 + 0.03j], [0.07 - 0.03j, 0.1]]
        )
        hamiltonians: Complex128[Array, "4 2 2"] = jnp.stack(
            [
                base + 0.02 * index * jnp.eye(2, dtype=jnp.complex128)
                for index in range(4)
            ]
        )
        basis: Any = make_orbital_basis(
            atom_indices=(0, 0),
            n=(1, 1),
            l=(0, 0),
            m=(0, 0),
        )
        radial: Any = make_radial_spec(
            basis,
            (0, 0),
            mode="fixed",
            fixed_integrals_shell=jnp.asarray([[0.0, 1.0]]),
        )
        matrix_element: Any = make_matrix_element_params(basis, (0, 0))
        omega: Float64[Array, " 8"] = jnp.linspace(-0.4, 0.3, 8)
        k_i: Float64[Array, "4 3"] = jnp.stack(
            (
                jnp.linspace(0.1, 0.16, 4),
                jnp.zeros(4),
                jnp.zeros(4),
            ),
            axis=-1,
        )
        final_norm: Float64[Array, " 8"] = 1.1 + 0.02 * jnp.arange(
            8, dtype=jnp.float64
        )
        schedule: TransitionSourceSchedule = make_transition_source_schedule(
            k_i_cart=k_i,
            final_norm=final_norm,
            emission_energy_valid=jnp.ones(8, dtype=jnp.bool_),
            positions_cart=jnp.asarray([[0.0, 0.0, 0.0], [0.23, 0.07, 0.02]]),
            depths=jnp.asarray([0.0, 0.4]),
            polarization_sample_cart=jnp.asarray(
                [1.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j]
            ),
            mean_free_path_ang=jnp.asarray(10.0),
            radial=radial,
            matrix_element=matrix_element,
            quadrature=make_radial_quadrature_spec(),
            final_state=make_final_state_spec(),
        )
        k_valid: Bool[Array, "4"] = jnp.asarray([True, True, True, False])
        omega_valid: Bool[Array, "8"] = jnp.asarray(
            [True, True, True, True, True, True, False, False]
        )
        returned: Tuple[
            Complex128[Array, "4 2 2"],
            TransitionSourceSchedule,
            Float64[Array, " 8"],
            Bool[Array, " 4"],
            Bool[Array, " 8"],
            SelfEnergyModel,
        ] = (
            hamiltonians,
            schedule,
            omega,
            k_valid,
            omega_valid,
            make_self_energy_model(gamma=0.04),
        )
        return returned

    @pytest.mark.big_mem
    @pytest.mark.rss_limit_mb(1200)
    def test_checkpointed_values_and_gradients_match_uncheckpointed(
        self,
    ) -> None:
        """Match rematerialized values and gradients to the direct scan.

        The comparison exercises one padded schedule with and without
        checkpoints.

        Notes
        -----
        The comparison is at rtol ``1e-12`` and verifies that masked padding
        contributes an exact zero gradient.
        """
        hamiltonians: Complex128[Array, "4 2 2"]
        schedule: TransitionSourceSchedule
        omega: Float64[Array, " 8"]
        k_valid: Bool[Array, "..."]
        omega_valid: Bool[Array, "..."]
        model: SelfEnergyModel
        hamiltonians, schedule, omega, k_valid, omega_valid, model = (
            self._fixture()
        )

        def streamed(
            candidate: Complex128[Array, "4 2 2"],
            checkpoint: bool,
        ) -> Float64[Array, "4 8"]:
            """Run one static stream schedule."""
            returned: Float64[Array, "4 8"] = _stream_spectral_intensity(
                candidate,
                omega,
                k_valid,
                omega_valid,
                schedule,
                model,
                jnp.asarray(0.03),
                20.0,
                1.0e-4,
                k_chunk=2,
                omega_chunk=4,
                checkpoint=checkpoint,
            )
            return returned

        checkpointed: Float64[Array, "4 8"] = streamed(hamiltonians, True)
        direct: Float64[Array, "4 8"] = streamed(hamiltonians, False)
        checkpointed_gradient: Complex128[Array, "4 2 2"] = jax.grad(
            lambda candidate: jnp.sum(streamed(candidate, True))
        )(hamiltonians)
        direct_gradient: Complex128[Array, "4 2 2"] = jax.grad(
            lambda candidate: jnp.sum(streamed(candidate, False))
        )(hamiltonians)
        np.testing.assert_allclose(
            np.asarray(checkpointed),
            np.asarray(direct),
            rtol=1.0e-12,
            atol=1.0e-13,
        )
        np.testing.assert_allclose(
            np.asarray(checkpointed_gradient),
            np.asarray(direct_gradient),
            rtol=1.0e-12,
            atol=1.0e-12,
        )
        assert bool(jnp.all(checkpointed[-1] == 0.0))
        assert bool(jnp.all(checkpointed[:, -2:] == 0.0))
        assert bool(jnp.all(checkpointed_gradient[-1] == 0.0))

    def test_one_trace_for_one_padded_schedule(self) -> None:
        """Require one trace across different masks of the same shapes.

        The test varies active extents while retaining all compiled dimensions.

        Notes
        -----
        A Python counter runs only while JAX traces the wrapper. Changing
        validity masks cannot retrace a fixed padded chunk schedule.
        """
        hamiltonians: Complex128[Array, "4 2 2"]
        schedule: TransitionSourceSchedule
        omega: Float64[Array, " 8"]
        k_valid: Bool[Array, "..."]
        omega_valid: Bool[Array, "..."]
        model: SelfEnergyModel
        hamiltonians, schedule, omega, k_valid, omega_valid, model = (
            self._fixture()
        )
        trace_count: List[int] = [0]

        def scheduled(
            matrices: Complex128[Array, "4 2 2"],
            energies: Float64[Array, " 8"],
            valid_k: Float64[Array, "..."],
            valid_omega: Float64[Array, "..."],
        ) -> Float64[Array, "4 8"]:
            """Record traces of one fixed stream schedule."""
            trace_count[0] += 1
            returned: Float64[Array, "4 8"] = _stream_spectral_intensity(
                matrices,
                energies,
                valid_k,
                valid_omega,
                schedule,
                model,
                jnp.asarray(0.03),
                20.0,
                1.0e-4,
                k_chunk=2,
                omega_chunk=4,
                checkpoint=True,
            )
            return returned

        compiled: Callable[..., Float64[Array, "..."]] = jax.jit(scheduled)
        first: Float64[Array, "..."] = compiled(
            hamiltonians, omega, k_valid, omega_valid
        )
        second: Float64[Array, "..."] = compiled(
            hamiltonians,
            omega,
            jnp.asarray([True, True, False, False]),
            jnp.asarray([True, True, True, True, False, False, False, False]),
        )
        jax.block_until_ready((first, second))
        assert trace_count[0] == 1

    def test_block_local_aperture_masks_column_and_gradients(self) -> None:
        """Verify exact masking outside the vacuum aperture.

        The check targets one energy-valid physical column.

        Notes
        -----
        The planted final-state magnitude is smaller than every live
        in-plane momentum. The streamed block derives the aperture mask
        locally. It returns exact zeros for the physical column and both
        complete gradients.
        """
        hamiltonians: Complex128[Array, "4 2 2"]
        schedule: TransitionSourceSchedule
        omega: Float64[Array, " 8"]
        k_valid: Bool[Array, "..."]
        omega_valid: Bool[Array, "..."]
        model: SelfEnergyModel
        hamiltonians, schedule, omega, k_valid, omega_valid, model = (
            self._fixture()
        )
        aperture_column: int = 2
        planted_norms: Float64[Array, " 8"] = schedule.final_norm.at[
            aperture_column
        ].set(0.05)
        planted_schedule: TransitionSourceSchedule = eqx.tree_at(
            lambda item: item.final_norm,
            schedule,
            planted_norms,
        )
        assert bool(planted_schedule.emission_energy_valid[aperture_column])
        assert bool(
            jnp.all(
                planted_norms[aperture_column]
                < jnp.linalg.norm(
                    planted_schedule.k_i_cart[k_valid, :2], axis=-1
                )
            )
        )

        def streamed(
            candidate_hamiltonians: Complex128[Array, "4 2 2"],
            candidate_norms: Float64[Array, " 8"],
        ) -> Float64[Array, "4 8"]:
            """Evaluate one compact schedule with dynamic final norms."""
            candidate_schedule: TransitionSourceSchedule = eqx.tree_at(
                lambda item: item.final_norm,
                planted_schedule,
                candidate_norms,
            )
            returned: Float64[Array, "4 8"] = _stream_spectral_intensity(
                candidate_hamiltonians,
                omega,
                k_valid,
                omega_valid,
                candidate_schedule,
                model,
                jnp.asarray(0.03),
                20.0,
                1.0e-4,
                k_chunk=2,
                omega_chunk=4,
                checkpoint=True,
            )
            return returned

        values: Float64[Array, "4 8"] = streamed(hamiltonians, planted_norms)

        def column_loss(
            candidate_hamiltonians: Complex128[Array, "4 2 2"],
            candidate_norms: Float64[Array, " 8"],
        ) -> Float64[Array, ""]:
            """Reduce only the planted outside-aperture column."""
            returned: Float64[Array, ""] = jnp.sum(
                streamed(candidate_hamiltonians, candidate_norms)[
                    :, aperture_column
                ]
            )
            return returned

        hamiltonian_gradient: Complex128[Array, "4 2 2"]
        norm_gradient: Float64[Array, " 8"]
        hamiltonian_gradient, norm_gradient = jax.grad(
            column_loss,
            argnums=(0, 1),
        )(hamiltonians, planted_norms)
        assert bool(jnp.all(values[:, aperture_column] == 0.0))
        assert bool(jnp.all(hamiltonian_gradient == 0.0))
        assert bool(jnp.all(norm_gradient == 0.0))

    @parameterized.named_parameters(
        ("nan", np.nan),
        ("zero", 0.0),
        ("negative", -0.1),
    )
    def test_active_final_momentum_magnitude_rejects(
        self,
        invalid_norm: float,
    ) -> None:
        """Reject invalid active final-state magnitudes eagerly and in JIT.

        The case covers nonfinite, zero, and negative active magnitudes.

        Notes
        -----
        The compact carrier permits a zero sentinel only when its paired
        physical-energy mask is false. Validation rejects nonfinite, negative,
        and active-zero magnitudes instead of treating them as absent emission.
        """
        hamiltonians: Complex128[Array, "4 2 2"]
        schedule: TransitionSourceSchedule
        omega: Float64[Array, " 8"]
        k_valid: Bool[Array, "..."]
        omega_valid: Bool[Array, "..."]
        model: SelfEnergyModel
        hamiltonians, schedule, omega, k_valid, omega_valid, model = (
            self._fixture()
        )

        def streamed(
            candidate_norms: Float64[Array, " 8"],
        ) -> Float64[Array, "4 8"]:
            """Evaluate one schedule with candidate final magnitudes."""
            candidate_schedule: TransitionSourceSchedule = eqx.tree_at(
                lambda item: item.final_norm,
                schedule,
                candidate_norms,
            )
            returned: Float64[Array, "4 8"] = _stream_spectral_intensity(
                hamiltonians,
                omega,
                k_valid,
                omega_valid,
                candidate_schedule,
                model,
                jnp.asarray(0.03),
                20.0,
                1.0e-4,
                k_chunk=2,
                omega_chunk=4,
                checkpoint=True,
            )
            return returned

        planted_norms: Float64[Array, " 8"] = schedule.final_norm.at[0].set(
            invalid_norm
        )
        assert_rejects(
            streamed,
            planted_norms,
            match="final-momentum|finite|nonnegative|strictly positive",
        )
