"""Validate the WP7.6 streamed spectral scaling evidence and dtype gate.

The isolated benchmark artifact owns literal-target compiler-live allocation,
compile-count, and complex128 solve evidence. Small executable tests
independently compare the checkpointed scan with the unchunked assembler and
reject low-precision solve inputs.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from beartype.typing import Dict, Tuple
from jaxtyping import Array, Complex128, Float64, TypeCheckError

from diffpes.simul import (
    assemble_spectral_intensity_chunk,
    spectral_intensity_resolvent,
)
from diffpes.types import (
    SelfEnergyModel,
    make_final_state_spec,
    make_matrix_element_params,
    make_orbital_basis,
    make_radial_quadrature_spec,
    make_radial_spec,
    make_self_energy_model,
)

ARTIFACT_DIRECTORY: Path = (
    Path(__file__).resolve().parents[1]
    / "_reference_data"
    / "spectral_scalability"
)
ARTIFACT_PATH: Path = ARTIFACT_DIRECTORY / "cpu_benchmark.json"
ARTIFACT_SHA256: str = (
    "3b4248a9498281f09fe9152f4fcc2db42ed92e2d82dedf93cf12cf229e352bca"
)
REPOSITORY_ROOT: Path = Path(__file__).resolve().parents[3]


def _sha256(path: Path) -> str:
    """PRIVATE: Return the SHA-256 digest of one complete file.

    Notes
    -----
    The artifact handshake reads the file as bytes so newline handling cannot
    alter the authenticated identity.
    """
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _artifact() -> Dict[str, Any]:
    """PRIVATE: Load and authenticate the isolated CPU evidence.

    Notes
    -----
    Authentication precedes JSON parsing, so tests never consume an
    unrecognized benchmark record.
    """
    assert _sha256(ARTIFACT_PATH) == ARTIFACT_SHA256
    return json.loads(ARTIFACT_PATH.read_text(encoding="utf-8"))


class TestSpectralScalabilityEvidence:
    """Validate the reproducible S1--S3 benchmark record."""

    def test_s1_literal_target_compiler_memory(self) -> None:
        """Verify the literal target and its measured 1.5x gate.

        The test binds every allocation identity to the registered target.

        Notes
        -----
        XLA compiled the exact registered ``(256, 512, 32)`` value-and-gradient
        program. Compiler allocation analysis provides the authority. The
        artifact marks the large executable as compile-only and process RSS as
        diagnostic.
        """
        artifact: Dict[str, Any] = _artifact()
        assert artifact["schema"] == "diffpes.spectral-scalability.v1"
        assert artifact["gate_ids"] == ["07.S1", "07.S2", "07.S3"]
        assert artifact["backend"] == "cpu"
        assert artifact["x64_enabled"] is True
        relative_path: str
        digest: str
        for relative_path, digest in artifact["source_sha256"].items():
            assert _sha256(REPOSITORY_ROOT / relative_path) == digest

        measurement: Dict[str, Any] = artifact["s1_literal_target"]
        assert (
            measurement["n_k_max"],
            measurement["n_omega_max"],
            measurement["n_out"],
            measurement["n_orb"],
            measurement["k_chunk"],
            measurement["omega_chunk"],
            measurement["n_kk"],
            measurement["n_tail"],
        ) == (256, 512, 1, 32, 32, 32, 4096, 256)
        assert measurement["checkpoint"] is True
        assert measurement["program"] == "value_and_hamiltonian_gradient"
        assert measurement["compilation_seconds"] > 0.0
        assert measurement["executed"] is False
        assert measurement["execution_seconds"] is None
        memory: Dict[str, Any] = measurement["memory_analysis"]
        assert memory["authority_available"] is True
        assert memory["result"] == "measured"
        assert (
            memory["argument_size_bytes"],
            memory["output_size_bytes"],
            memory["temporary_size_bytes"],
            memory["alias_size_bytes"],
            memory["compiler_live_allocation_bytes"],
        ) == (7_483_224, 4_194_328, 48_595_264, 0, 60_272_816)
        live: int = (
            memory["argument_size_bytes"]
            + memory["output_size_bytes"]
            + memory["temporary_size_bytes"]
            - memory["alias_size_bytes"]
        )
        assert live == memory["compiler_live_allocation_bytes"]
        measured_model: Dict[str, Any] = measurement["allocation_model"]
        registered_bound: int = 16 * 256 * 32 * 32**2
        assert (
            measured_model["registered_solve_tape_bytes"] == registered_bound
        )
        assert measured_model["registered_factor"] == 1.5
        assert measured_model["registered_ceiling_bytes"] == int(
            1.5 * registered_bound
        )
        assert measurement["passes_registered_1p5x_bound"] is True
        assert live <= measured_model["registered_ceiling_bytes"]
        assert (
            measurement["process_peak_rss_before_bytes_non_authoritative"] > 0
        )
        assert (
            measurement["process_peak_rss_before_bytes_non_authoritative"],
            measurement["process_peak_rss_after_bytes_non_authoritative"],
        ) == (478_945_280, 693_555_200)
        assert (
            measurement["process_peak_rss_after_bytes_non_authoritative"]
            >= measurement["process_peak_rss_before_bytes_non_authoritative"]
        )

    def test_s1_reference_s2_compile_and_s3_dtype_records(self) -> None:
        """Verify every non-memory S1--S3 artifact verdict.

        The test authenticates reference accuracy, trace reuse, and dtype rows.

        Notes
        -----
        Exact frozen measurements supplement boolean verdicts so an artifact
        cannot pass by changing only its summary field.
        """
        artifact: Dict[str, Any] = _artifact()
        reference: Dict[str, Any] = artifact["s1_reference_comparison"]
        name: str
        for name in (
            "value_passes_rtol_1e_12",
            "gradient_passes_rtol_1e_12",
            "nonzero_gradient",
        ):
            assert reference[name] is True
        assert reference["maximum_reference_gradient"] > 1.0e-8
        assert reference["maximum_value_absolute_error"] == pytest.approx(
            4.336808689942018e-19,
            rel=0.0,
            abs=0.0,
        )
        assert reference["maximum_gradient_absolute_error"] == 0.0
        compile_count: Dict[str, Any] = artifact["s2_compile_count"]
        assert compile_count["n_out"] == 1
        assert compile_count["trace_count"] == 1
        assert compile_count["compile_cache_sizes"] == [0, 1, 1, 1]
        assert compile_count["result"] == "pass"
        dtype: Dict[str, Any] = artifact["s3_dtype"]
        assert dtype["operator_input_dtype"] == "complex128"
        assert dtype["rhs_input_dtype"] == "complex128"
        assert dtype["solution_dtype"] == "complex128"
        assert dtype["compiler_ir_contains_complex_f64"] is True
        assert dtype["compiler_ir_contains_complex_f32"] is False
        assert dtype["complex64_public_call_rejected"] is True
        assert dtype["complex64_rejection_exception"] == "TypeCheckError"
        assert dtype["result"] == "pass"
        assert artifact["result"] == "pass"


class TestSpectralStreamRuntimeScaling:
    """Run small direct S1 and S3 production checks on every test host."""

    @staticmethod
    def _fixture() -> Tuple[
        Complex128[Array, "4 2 2"],
        Any,
        Float64[Array, " 8"],
        Array,
        Array,
        SelfEnergyModel,
    ]:
        """PRIVATE: Return one fixed padded Plan-06 source schedule.

        Notes
        -----
        The fixture includes invalid padding on both axes and deterministic
        transition geometry for value and gradient comparisons.
        """
        import diffpes.simul.spectral as spectral

        base: Complex128[Array, "2 2"] = jnp.asarray(
            [[-0.2, 0.07 + 0.03j], [0.07 - 0.03j, 0.1]]
        )
        identity: Complex128[Array, "2 2"] = jnp.eye(2, dtype=jnp.complex128)
        hamiltonians: Complex128[Array, "4 2 2"] = jnp.stack(
            [base + 0.02 * index * identity for index in range(4)]
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
        omega: Float64[Array, " 8"] = jnp.linspace(-0.4, 0.3, 8)
        k_i: Float64[Array, "4 3"] = jnp.stack(
            (
                jnp.linspace(0.1, 0.16, 4),
                jnp.zeros(4),
                jnp.zeros(4),
            ),
            axis=-1,
        )
        final_z: Float64[Array, "4 8"] = (
            1.1
            + 0.01 * jnp.arange(4, dtype=jnp.float64)[:, None]
            + 0.02 * jnp.arange(8, dtype=jnp.float64)[None, :]
        )
        k_f: Float64[Array, "4 8 3"] = jnp.stack(
            (
                jnp.broadcast_to(k_i[:, 0, None], final_z.shape),
                jnp.zeros_like(final_z),
                final_z,
            ),
            axis=-1,
        )
        schedule: Any = spectral._TransitionSourceSchedule(
            k_i_cart=k_i,
            k_f_cart=k_f,
            emission_valid=jnp.ones((4, 8), dtype=jnp.bool_),
            positions_cart=jnp.asarray([[0.0, 0.0, 0.0], [0.23, 0.07, 0.02]]),
            depths=jnp.asarray([0.0, 0.4]),
            polarization_sample_cart=jnp.asarray(
                [1.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j]
            ),
            mean_free_path_ang=jnp.asarray(10.0),
            radial=radial,
            matrix_element=make_matrix_element_params(basis, (0, 0)),
            quadrature=make_radial_quadrature_spec(),
            final_state=make_final_state_spec(),
        )
        k_valid: Array = jnp.asarray([True, True, True, False])
        omega_valid: Array = jnp.asarray(
            [True, True, True, True, True, True, False, False]
        )
        return (
            hamiltonians,
            schedule,
            omega,
            k_valid,
            omega_valid,
            make_self_energy_model(gamma=0.04),
        )

    @pytest.mark.big_mem
    @pytest.mark.rss_limit_mb(1200)
    def test_checkpointed_stream_matches_unchunked_values_and_gradients(
        self,
    ) -> None:
        """Match rematerialized scan values and gradients to full assembly.

        The test compares both the complete observable and Hamiltonian VJP.

        Notes
        -----
        The direct assembler uses sources built once for the whole padded
        carrier. The streamed path rebuilds only each live chunk.
        """
        import diffpes.simul.spectral as spectral

        hamiltonians: Complex128[Array, "4 2 2"]
        schedule: Any
        omega: Float64[Array, " 8"]
        k_valid: Array
        omega_valid: Array
        model: SelfEnergyModel
        hamiltonians, schedule, omega, k_valid, omega_valid, model = (
            self._fixture()
        )
        mask: Array = (
            k_valid[:, None] & omega_valid[None, :] & schedule.emission_valid
        )
        sources: Complex128[Array, "4 8 n_out 2"] = (
            spectral._transition_sources_for_block(
                schedule,
                schedule.k_i_cart,
                schedule.k_f_cart,
                mask,
            )
        )

        def streamed(
            candidate: Complex128[Array, "4 2 2"],
        ) -> Float64[Array, "4 8"]:
            """Evaluate the checkpointed static schedule."""
            return spectral._stream_spectral_intensity(
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
                checkpoint=True,
            )

        def unchunked(
            candidate: Complex128[Array, "4 2 2"],
        ) -> Float64[Array, "4 8"]:
            """Evaluate and mask the complete reference carrier."""
            values: Float64[Array, "4 8"] = assemble_spectral_intensity_chunk(
                candidate,
                sources,
                omega,
                model,
                jnp.asarray(0.03),
                20.0,
                1.0e-4,
            )
            return jnp.where(mask, values, 0.0)

        produced: Float64[Array, "4 8"] = streamed(hamiltonians)
        expected: Float64[Array, "4 8"] = unchunked(hamiltonians)
        produced_gradient: Complex128[Array, "4 2 2"] = jax.grad(
            lambda candidate: jnp.sum(streamed(candidate))
        )(hamiltonians)
        expected_gradient: Complex128[Array, "4 2 2"] = jax.grad(
            lambda candidate: jnp.sum(unchunked(candidate))
        )(hamiltonians)
        np.testing.assert_allclose(
            np.asarray(produced),
            np.asarray(expected),
            rtol=1.0e-12,
            atol=1.0e-13,
        )
        np.testing.assert_allclose(
            np.asarray(produced_gradient),
            np.asarray(expected_gradient),
            rtol=1.0e-12,
            atol=1.0e-12,
        )
        assert float(jnp.max(jnp.abs(expected_gradient))) > 1.0e-8
        assert bool(jnp.all(produced_gradient[~k_valid] == 0.0))

    def test_resolvent_solve_is_complex128_and_rejects_complex64(self) -> None:
        """Assert S3 inside the solve and at the public typed boundary.

        The test inspects lowered IR and calls the typed API with complex64.

        Notes
        -----
        Compiler text must contain complex-f64 operations and no complex-f32
        operation before the low-precision public call rejects.
        """
        import diffpes.simul.spectral as spectral

        hamiltonian: Complex128[Array, "2 2"] = jnp.asarray(
            [[-0.2, 0.04 + 0.03j], [0.04 - 0.03j, 0.15]],
            dtype=jnp.complex128,
        )
        source: Complex128[Array, " 2"] = jnp.asarray(
            [0.7 + 0.2j, -0.4 + 0.1j], dtype=jnp.complex128
        )
        omega: Float64[Array, ""] = jnp.asarray(0.05, dtype=jnp.float64)
        sigma: Complex128[Array, ""] = jnp.asarray(
            0.01 - 0.04j, dtype=jnp.complex128
        )
        eta: Float64[Array, ""] = jnp.asarray(1.0e-4, dtype=jnp.float64)
        solution: Array = spectral._resolvent_solution(
            hamiltonian, source, omega, sigma, eta
        )
        assert solution.dtype == jnp.complex128
        compiler_text: str = (
            jax.jit(spectral._resolvent_solution)
            .lower(hamiltonian, source, omega, sigma, eta)
            .as_text()
        )
        assert "complex<f64>" in compiler_text
        assert "complex<f32>" not in compiler_text
        with pytest.raises(TypeCheckError, match="Complex128|complex128"):
            spectral_intensity_resolvent(
                hamiltonian.astype(jnp.complex64),
                source[None, :].astype(jnp.complex64),
                omega.astype(jnp.float32),
                sigma.astype(jnp.complex64),
                eta.astype(jnp.float32),
            )
