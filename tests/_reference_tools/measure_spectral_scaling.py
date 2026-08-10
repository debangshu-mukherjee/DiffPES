"""Measure the literal Plan-07 streamed spectral scalability gates on CPU.

The isolated harness lowers and compiles the preregistered
``(256 k, 512 omega, 32 orbital)`` checkpointed value-and-Hamiltonian-gradient
program. XLA's compiler allocation record is authoritative; host RSS is a
diagnostic. Small executable companions check the unchunked reference,
compile reuse, and complex128 solve contract without duplicating the target
allocation in routine CI.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import resource
import sys
import time
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from beartype.typing import Dict, Tuple
from jaxtyping import Array, Complex128, Float64

from diffpes.simul.spectral import (
    _resolvent_solution,
    _stream_spectral_intensity,
    _transition_sources_for_block,
    _TransitionSourceSchedule,
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

REPOSITORY_ROOT: Path = Path(__file__).resolve().parents[2]
OUTPUT_DIRECTORY: Path = (
    REPOSITORY_ROOT
    / "tests"
    / "test_diffpes"
    / "_reference_data"
    / "spectral_scalability"
)
OUTPUT_PATH: Path = OUTPUT_DIRECTORY / "cpu_benchmark.json"

TARGET_N_K: int = 256
TARGET_N_OMEGA: int = 512
TARGET_N_ORB: int = 32
TARGET_N_OUT: int = 1
TARGET_K_CHUNK: int = 32
TARGET_OMEGA_CHUNK: int = 32
REFERENCE_N_K: int = 4
REFERENCE_N_OMEGA: int = 8
REFERENCE_N_ORB: int = 2
REFERENCE_K_CHUNK: int = 2
REFERENCE_OMEGA_CHUNK: int = 4
N_KK: int = 4096
N_TAIL: int = 256
COMPLEX128_BYTES: int = 16
FLOAT64_BYTES: int = 8
BOOL_BYTES: int = 1
BOUND_FACTOR: float = 1.5


def _sha256(path: Path) -> str:
    """PRIVATE: Return the SHA-256 digest of one complete file.

    Notes
    -----
    Binary reads make the recorded identity independent of text decoding and
    newline normalization.
    """
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _maximum_rss_bytes() -> int:
    """PRIVATE: Return the Linux process high-water RSS in bytes.

    Notes
    -----
    Linux reports ``ru_maxrss`` in kibibytes. The benchmark converts that
    diagnostic value to bytes without treating it as allocation authority.
    """
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024


def _memory_record(compiled: Any) -> Dict[str, int | bool | str]:
    """PRIVATE: Normalize XLA compiler allocation counters.

    Parameters
    ----------
    compiled : Any
        Ahead-of-time compiled JAX executable.

    Returns
    -------
    record : dict[str, int | bool | str]
        Backend counters and the derived compiler-live allocation.
    """
    analysis: Any = compiled.memory_analysis()
    required: Tuple[str, ...] = (
        "argument_size_in_bytes",
        "output_size_in_bytes",
        "temp_size_in_bytes",
        "alias_size_in_bytes",
    )
    if analysis is None or any(
        getattr(analysis, name, None) is None for name in required
    ):
        return {
            "authority_available": False,
            "result": "residual: XLA memory_analysis unavailable",
        }
    arguments: int = int(analysis.argument_size_in_bytes)
    outputs: int = int(analysis.output_size_in_bytes)
    temporaries: int = int(analysis.temp_size_in_bytes)
    aliases: int = int(analysis.alias_size_in_bytes)
    live: int = arguments + outputs + temporaries - aliases
    return {
        "authority_available": True,
        "argument_size_bytes": arguments,
        "output_size_bytes": outputs,
        "temporary_size_bytes": temporaries,
        "alias_size_bytes": aliases,
        "compiler_live_allocation_bytes": live,
        "result": "measured",
    }


def _allocation_model(
    n_k: int,
    n_omega: int,
    n_orb: int,
    k_chunk: int,
    omega_chunk: int,
) -> Dict[str, int | float]:
    """PRIVATE: Derive registered and diagnostic allocation terms.

    Parameters
    ----------
    n_k : int
        Padded momentum count.
    n_omega : int
        Padded sampled-energy count.
    n_orb : int
        Orbital count.
    k_chunk : int
        Static momentum chunk length.
    omega_chunk : int
        Static sampled-energy chunk length.

    Returns
    -------
    model : dict[str, int | float]
        The registered solve-tape estimate and 1.5-times ceiling plus
        diagnostic kinematic, block-source, and matrix-free-KK byte terms.

    Notes
    -----
    The blocking S1 criterion remains exactly
    ``16 * n_k * omega_chunk * n_orb**2`` and its registered ``1.5x``
    ceiling. Other terms explain the measured executable but do not enlarge
    that acceptance ceiling after observation.
    """
    solve_tape: int = COMPLEX128_BYTES * n_k * omega_chunk * n_orb**2
    ceiling: int = int(BOUND_FACTOR * solve_tape)
    kinematics: int = (
        FLOAT64_BYTES * n_k * 3
        + FLOAT64_BYTES * n_k * n_omega * 3
        + BOOL_BYTES * n_k * n_omega
    )
    block_bvals: int = COMPLEX128_BYTES * k_chunk * omega_chunk * n_orb * 2
    block_channels: int = COMPLEX128_BYTES * k_chunk * omega_chunk * n_orb * 3
    block_sources: int = COMPLEX128_BYTES * k_chunk * omega_chunk * n_orb
    kk_work: int = FLOAT64_BYTES * omega_chunk * N_KK
    required_axes: int = (
        COMPLEX128_BYTES * n_k * n_orb**2
        + FLOAT64_BYTES * n_omega
        + BOOL_BYTES * (n_k + n_omega)
        + FLOAT64_BYTES * n_k * n_omega
    )
    return {
        "registered_solve_tape_bytes": solve_tape,
        "registered_factor": BOUND_FACTOR,
        "registered_ceiling_bytes": ceiling,
        "padded_kinematics_bytes_diagnostic": kinematics,
        "padded_hamiltonian_axis_mask_output_bytes_diagnostic": required_axes,
        "block_radial_bvals_bytes_diagnostic": block_bvals,
        "block_transition_channels_bytes_diagnostic": block_channels,
        "block_transition_sources_bytes_diagnostic": block_sources,
        "matrix_free_kk_work_bytes_diagnostic": kk_work,
    }


def _self_energy(*, numerical_kk: bool) -> SelfEnergyModel:
    """PRIVATE: Build a constant or registered numerical-KK carrier.

    Notes
    -----
    Literal-target compilation uses the registered numerical transform while
    small comparison and trace checks use the constant-width carrier.
    """
    if not numerical_kk:
        return make_self_energy_model(gamma=0.04)
    physical: Float64[Array, " three"] = jnp.asarray([0.02, 0.5, 0.8])
    raw: Float64[Array, " three"] = jnp.log(jnp.expm1(physical))
    return make_self_energy_model(
        coefficients=raw,
        mode="fermi_liquid",
        kk_consistent=True,
        kk_domain_rel_fermi_ev=jnp.asarray([-8.0, 8.0]),
        tail_coefficients=jnp.asarray(
            [-12.270842147353243, -12.270842147353243]
        ),
        subtraction_point_rel_fermi_ev=0.0,
        tail_mode="power2",
    )


def _fixture(
    n_k: int,
    n_omega: int,
    n_orb: int,
    *,
    numerical_kk: bool,
) -> Tuple[
    Complex128[Array, "n_k n_orb n_orb"],
    Float64[Array, " n_omega"],
    Array,
    Array,
    _TransitionSourceSchedule,
    SelfEnergyModel,
]:
    """PRIVATE: Construct one deterministic padded source schedule.

    Notes
    -----
    The carrier has fixed geometry, static axis lengths, one spinless outgoing
    channel, and a selectable self-energy cost profile.
    """
    diagonal: Float64[Array, " n_orb"] = jnp.linspace(-0.35, 0.4, n_orb)
    row: Array = jnp.arange(n_orb)[:, None]
    column: Array = jnp.arange(n_orb)[None, :]
    off_diagonal: Complex128[Array, "n_orb n_orb"] = jnp.where(
        jnp.abs(row - column) == 1,
        0.012 + 0.007j * jnp.sign(column - row),
        0.0 + 0.0j,
    )
    base: Complex128[Array, "n_orb n_orb"] = (
        jnp.diag(diagonal).astype(jnp.complex128) + off_diagonal
    )
    identity: Complex128[Array, "n_orb n_orb"] = jnp.eye(
        n_orb, dtype=jnp.complex128
    )
    hamiltonians: Complex128[Array, "n_k n_orb n_orb"] = jnp.stack(
        [base + (0.0007 * index) * identity for index in range(n_k)]
    )
    omega: Float64[Array, " n_omega"] = jnp.linspace(-0.7, 0.6, n_omega)
    k_i: Float64[Array, "n_k 3"] = jnp.stack(
        (
            jnp.linspace(0.05, 0.25, n_k),
            jnp.zeros(n_k),
            jnp.zeros(n_k),
        ),
        axis=-1,
    )
    final_z: Float64[Array, "n_k n_omega"] = (
        1.0
        + 5.0e-4 * jnp.arange(n_k, dtype=jnp.float64)[:, None]
        + 2.0e-4 * jnp.arange(n_omega, dtype=jnp.float64)[None, :]
    )
    k_f: Float64[Array, "n_k n_omega 3"] = jnp.stack(
        (
            jnp.broadcast_to(k_i[:, 0, None], final_z.shape),
            jnp.zeros_like(final_z),
            final_z,
        ),
        axis=-1,
    )
    basis: Any = make_orbital_basis(
        atom_indices=(0,) * n_orb,
        n=(1,) * n_orb,
        l=(0,) * n_orb,
        m=(0,) * n_orb,
    )
    shell_index: Tuple[int, ...] = (0,) * n_orb
    radial: Any = make_radial_spec(
        basis,
        shell_index,
        mode="fixed",
        fixed_integrals_shell=jnp.asarray([[0.0, 1.0]]),
    )
    positions: Float64[Array, "n_orb 3"] = jnp.stack(
        (
            jnp.linspace(0.0, 3.0, n_orb),
            jnp.linspace(0.0, 0.4, n_orb),
            jnp.linspace(0.0, 1.0, n_orb),
        ),
        axis=-1,
    )
    schedule: _TransitionSourceSchedule = _TransitionSourceSchedule(
        k_i_cart=k_i,
        k_f_cart=k_f,
        emission_valid=jnp.ones((n_k, n_omega), dtype=jnp.bool_),
        positions_cart=positions,
        depths=jnp.linspace(0.0, 4.0, n_orb),
        polarization_sample_cart=jnp.asarray(
            [1.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j]
        ),
        mean_free_path_ang=jnp.asarray(10.0),
        radial=radial,
        matrix_element=make_matrix_element_params(basis, shell_index),
        quadrature=make_radial_quadrature_spec(),
        final_state=make_final_state_spec(),
    )
    return (
        hamiltonians,
        omega,
        jnp.ones(n_k, dtype=jnp.bool_),
        jnp.ones(n_omega, dtype=jnp.bool_),
        schedule,
        _self_energy(numerical_kk=numerical_kk),
    )


def _stream_call(
    hamiltonians: Complex128[Array, "n_k n_orb n_orb"],
    omega: Float64[Array, " n_omega"],
    k_valid: Array,
    omega_valid: Array,
    schedule: _TransitionSourceSchedule,
    model: SelfEnergyModel,
    *,
    k_chunk: int,
    omega_chunk: int,
    checkpoint: bool,
) -> Float64[Array, "n_k n_omega"]:
    """PRIVATE: Evaluate one static streamed schedule.

    Notes
    -----
    Shared thermodynamic coordinates keep the benchmark focused on chunking,
    rematerialization, and the Hamiltonian derivative tape.
    """
    return _stream_spectral_intensity(
        hamiltonians,
        omega,
        k_valid,
        omega_valid,
        schedule,
        model,
        jnp.asarray(0.025),
        18.0,
        1.0e-4,
        k_chunk=k_chunk,
        omega_chunk=omega_chunk,
        checkpoint=checkpoint,
    )


def _reference_comparison() -> Dict[str, float | bool]:
    """PRIVATE: Compare checkpointed streaming with unchunked assembly.

    Notes
    -----
    The record includes maximum value and Hamiltonian-gradient errors plus a
    nonzero-gradient witness for the small executable companion.
    """
    hamiltonians, omega, k_valid, omega_valid, schedule, model = _fixture(
        REFERENCE_N_K,
        REFERENCE_N_OMEGA,
        REFERENCE_N_ORB,
        numerical_kk=False,
    )
    mask: Array = (
        k_valid[:, None] & omega_valid[None, :] & schedule.emission_valid
    )

    def stream_values(
        candidate: Complex128[Array, "n_k n_orb n_orb"],
    ) -> Float64[Array, "n_k n_omega"]:
        return _stream_call(
            candidate,
            omega,
            k_valid,
            omega_valid,
            schedule,
            model,
            k_chunk=REFERENCE_K_CHUNK,
            omega_chunk=REFERENCE_OMEGA_CHUNK,
            checkpoint=True,
        )

    sources: Complex128[Array, "n_k n_omega one n_orb"] = (
        _transition_sources_for_block(
            schedule,
            schedule.k_i_cart,
            schedule.k_f_cart,
            mask,
        )
    )

    def direct_values(
        candidate: Complex128[Array, "n_k n_orb n_orb"],
    ) -> Float64[Array, "n_k n_omega"]:
        values: Float64[Array, "n_k n_omega"] = (
            assemble_spectral_intensity_chunk(
                candidate,
                sources,
                omega,
                model,
                jnp.asarray(0.025),
                18.0,
                1.0e-4,
            )
        )
        return jnp.where(mask, values, 0.0)

    streamed: Float64[Array, "n_k n_omega"] = stream_values(hamiltonians)
    direct: Float64[Array, "n_k n_omega"] = direct_values(hamiltonians)
    stream_gradient: Complex128[Array, "n_k n_orb n_orb"] = jax.grad(
        lambda candidate: jnp.sum(stream_values(candidate))
    )(hamiltonians)
    direct_gradient: Complex128[Array, "n_k n_orb n_orb"] = jax.grad(
        lambda candidate: jnp.sum(direct_values(candidate))
    )(hamiltonians)
    value_error: float = float(jnp.max(jnp.abs(streamed - direct)))
    gradient_error: float = float(
        jnp.max(jnp.abs(stream_gradient - direct_gradient))
    )
    value_scale: float = float(jnp.max(jnp.abs(direct)))
    gradient_scale: float = float(jnp.max(jnp.abs(direct_gradient)))
    return {
        "maximum_value_absolute_error": value_error,
        "maximum_gradient_absolute_error": gradient_error,
        "maximum_reference_value": value_scale,
        "maximum_reference_gradient": gradient_scale,
        "value_passes_rtol_1e_12": value_error
        <= 1.0e-12 * max(1.0, value_scale),
        "gradient_passes_rtol_1e_12": gradient_error
        <= 1.0e-12 * max(1.0, gradient_scale),
        "nonzero_gradient": gradient_scale > 1.0e-8,
    }


def _compile_count() -> Dict[str, Any]:
    """PRIVATE: Count traces across active sizes on one padded schedule.

    Notes
    -----
    Only validity masks change between calls. Stable shapes and static chunk
    sizes must retain one compiled executable.
    """
    hamiltonians, omega, _, _, schedule, model = _fixture(
        REFERENCE_N_K,
        REFERENCE_N_OMEGA,
        REFERENCE_N_ORB,
        numerical_kk=False,
    )
    traces: list[int] = [0]

    def scheduled(
        matrices: Complex128[Array, "n_k n_orb n_orb"],
        energies: Float64[Array, " n_omega"],
        valid_k: Array,
        valid_omega: Array,
    ) -> Float64[Array, "n_k n_omega"]:
        traces[0] += 1
        return _stream_call(
            matrices,
            energies,
            valid_k,
            valid_omega,
            schedule,
            model,
            k_chunk=REFERENCE_K_CHUNK,
            omega_chunk=REFERENCE_OMEGA_CHUNK,
            checkpoint=True,
        )

    compiled: Any = jax.jit(scheduled)
    active_sizes: Tuple[Tuple[int, int], ...] = ((2, 4), (3, 6), (4, 8))
    cache_sizes: list[int] = [int(compiled._cache_size())]
    active_k: int
    active_omega: int
    for active_k, active_omega in active_sizes:
        result: Array = compiled(
            hamiltonians,
            omega,
            jnp.arange(REFERENCE_N_K) < active_k,
            jnp.arange(REFERENCE_N_OMEGA) < active_omega,
        )
        jax.block_until_ready(result)
        cache_sizes.append(int(compiled._cache_size()))
    return {
        "padded_shape": [REFERENCE_N_K, REFERENCE_N_OMEGA, REFERENCE_N_ORB],
        "n_out": 1,
        "chunk_schedule": [REFERENCE_K_CHUNK, REFERENCE_OMEGA_CHUNK],
        "active_sizes": [list(item) for item in active_sizes],
        "trace_count": traces[0],
        "compile_cache_sizes": cache_sizes,
        "result": "pass"
        if traces[0] == 1 and cache_sizes == [0, 1, 1, 1]
        else "fail",
    }


def _dtype_record() -> Dict[str, Any]:
    """PRIVATE: Record complex128 solve IR and low-precision rejection.

    Notes
    -----
    Lowered compiler text certifies the internal solve precision. A typed
    public call independently witnesses complex64 rejection.
    """
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
    lowered: Any = jax.jit(_resolvent_solution).lower(
        hamiltonian, source, omega, sigma, eta
    )
    compiler_text: str = lowered.as_text()
    solution: Array = lowered.compile()(hamiltonian, source, omega, sigma, eta)
    jax.block_until_ready(solution)
    rejected: bool = False
    exception_type: str = ""
    try:
        spectral_intensity_resolvent(
            hamiltonian.astype(jnp.complex64),
            source[None, :].astype(jnp.complex64),
            omega.astype(jnp.float32),
            sigma.astype(jnp.complex64),
            eta.astype(jnp.float32),
        )
    except Exception as error:  # noqa: BLE001 -- record the guard owner.
        rejected = True
        exception_type = type(error).__name__
    return {
        "operator_input_dtype": str(hamiltonian.dtype),
        "rhs_input_dtype": str(source.dtype),
        "solution_dtype": str(solution.dtype),
        "compiler_ir_contains_complex_f64": "complex<f64>" in compiler_text,
        "compiler_ir_contains_complex_f32": "complex<f32>" in compiler_text,
        "complex64_public_call_rejected": rejected,
        "complex64_rejection_exception": exception_type,
        "result": "pass"
        if solution.dtype == jnp.complex128
        and "complex<f64>" in compiler_text
        and "complex<f32>" not in compiler_text
        and rejected
        else "fail",
    }


def main() -> None:
    """Compile the literal target and write the complete S1--S3 record."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--execute-target",
        action="store_true",
        help="also execute the 256x512x32 value-and-gradient program once",
    )
    arguments = parser.parse_args()
    hamiltonians, omega, k_valid, omega_valid, schedule, model = _fixture(
        TARGET_N_K,
        TARGET_N_OMEGA,
        TARGET_N_ORB,
        numerical_kk=True,
    )

    def loss(
        candidate: Complex128[Array, "n_k n_orb n_orb"],
        energies: Float64[Array, " n_omega"],
        valid_k: Array,
        valid_omega: Array,
        source_schedule: _TransitionSourceSchedule,
    ) -> Float64[Array, ""]:
        return jnp.sum(
            _stream_call(
                candidate,
                energies,
                valid_k,
                valid_omega,
                source_schedule,
                model,
                k_chunk=TARGET_K_CHUNK,
                omega_chunk=TARGET_OMEGA_CHUNK,
                checkpoint=True,
            )
        )

    compiled_function: Any = jax.jit(jax.value_and_grad(loss))
    rss_before: int = _maximum_rss_bytes()
    compilation_start: float = time.perf_counter()
    compiled: Any = compiled_function.lower(
        hamiltonians,
        omega,
        k_valid,
        omega_valid,
        schedule,
    ).compile()
    compilation_seconds: float = time.perf_counter() - compilation_start
    memory: Dict[str, int | bool | str] = _memory_record(compiled)
    executed: bool = bool(arguments.execute_target)
    execution_seconds: float | None = None
    if executed:
        execution_start: float = time.perf_counter()
        result: Any = compiled(
            hamiltonians,
            omega,
            k_valid,
            omega_valid,
            schedule,
        )
        jax.block_until_ready(result)
        execution_seconds = time.perf_counter() - execution_start
    rss_after: int = _maximum_rss_bytes()
    allocation: Dict[str, int | float] = _allocation_model(
        TARGET_N_K,
        TARGET_N_OMEGA,
        TARGET_N_ORB,
        TARGET_K_CHUNK,
        TARGET_OMEGA_CHUNK,
    )
    live_bytes: int = int(memory.get("compiler_live_allocation_bytes", 0))
    memory_passes: bool = bool(memory.get("authority_available")) and (
        live_bytes <= int(allocation["registered_ceiling_bytes"])
    )
    source_paths: Tuple[str, ...] = (
        "tests/_reference_tools/measure_spectral_scaling.py",
        "src/diffpes/simul/spectral.py",
        "src/diffpes/simul/matrixel.py",
        "src/diffpes/radial/integrate.py",
        "src/diffpes/types/radial_params.py",
        "src/diffpes/types/self_energy.py",
        "pyproject.toml",
        "uv.lock",
    )
    reference: Dict[str, float | bool] = _reference_comparison()
    compile_count: Dict[str, Any] = _compile_count()
    dtype: Dict[str, Any] = _dtype_record()
    record: Dict[str, Any] = {
        "schema": "diffpes.spectral-scalability.v1",
        "gate_ids": ["07.S1", "07.S2", "07.S3"],
        "backend": jax.default_backend(),
        "device": str(jax.devices()[0]),
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "jax_version": jax.__version__,
        "numpy_version": np.__version__,
        "x64_enabled": bool(jax.config.jax_enable_x64),
        "preallocate_environment": os.environ.get(
            "XLA_PYTHON_CLIENT_PREALLOCATE", "unset"
        ),
        "source_sha256": {
            relative: _sha256(REPOSITORY_ROOT / relative)
            for relative in source_paths
        },
        "s1_literal_target": {
            "n_k_max": TARGET_N_K,
            "n_omega_max": TARGET_N_OMEGA,
            "n_orb": TARGET_N_ORB,
            "n_out": TARGET_N_OUT,
            "k_chunk": TARGET_K_CHUNK,
            "omega_chunk": TARGET_OMEGA_CHUNK,
            "n_kk": N_KK,
            "n_tail": N_TAIL,
            "checkpoint": True,
            "program": "value_and_hamiltonian_gradient",
            "compilation_seconds": compilation_seconds,
            "executed": executed,
            "execution_seconds": execution_seconds,
            "memory_analysis": memory,
            "allocation_model": allocation,
            "passes_registered_1p5x_bound": memory_passes,
            "process_peak_rss_before_bytes_non_authoritative": rss_before,
            "process_peak_rss_after_bytes_non_authoritative": rss_after,
        },
        "s1_reference_comparison": reference,
        "s2_compile_count": compile_count,
        "s3_dtype": dtype,
        "result": "pass"
        if memory_passes
        and all(
            bool(reference[name])
            for name in (
                "value_passes_rtol_1e_12",
                "gradient_passes_rtol_1e_12",
                "nonzero_gradient",
            )
        )
        and compile_count["result"] == "pass"
        and dtype["result"] == "pass"
        else "fail",
    }
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    json.dump(record, sys.stdout, indent=2, sort_keys=True)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
