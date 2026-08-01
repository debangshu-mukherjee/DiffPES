"""Measure reproducible Plan 06 S1--S3 matrix-element scalability evidence.

The harness keeps all large numerical arrays as dynamic JAX arguments. It
records raw synchronized timings, compiler memory analysis, recursive JAXPR
equation counts, optimized HLO, and the complete input checksum. Timing
thresholds are evidence for the measured host, not CI performance contracts.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import platform
import re
import resource
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NamedTuple, cast

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax.extend.core import ClosedJaxpr, Jaxpr
from jaxtyping import Array, Complex, Float, Shaped
from numpy.typing import NDArray

from diffpes.simul.matrixel import (
    contract_polarization,
    orbital_transition_channels,
    project_band_channels,
)
from diffpes.types import (
    MatrixElementParams,
    OrbitalBasis,
    make_matrix_element_params,
    make_orbital_basis,
)

N_K: int = 4096
N_ORBITALS: int = 18
N_ENERGY: int = 8
N_GROUPS: int = 6
N_POLARIZATIONS: int = 6
REPETITIONS: int = 7
WARMUPS: int = 2
LIMIT_BYTES: int = 2 * 1024**3
MIN_DYNAMIC_ARGUMENT_BYTES: int = 1_000_000
MAX_CONTRACTION_RATIO: float = 1.5


class DynamicInputs(NamedTuple):
    """Contain every dynamic numerical argument to the channel primitive."""

    initial_momentum: Float[Array, "n_k 3"]
    final_momentum: Float[Array, "n_k 3"]
    positions: Float[Array, "n_orb 3"]
    depths: Float[Array, " n_orb"]
    bvals: Complex[Array, "n_k n_orb 2"]
    matrix_params: MatrixElementParams
    mean_free_path: Float[Array, ""]


@dataclass(frozen=True)
class Fixture:
    """Hold static metadata and dynamic arrays for one benchmark shape."""

    basis: OrbitalBasis
    dynamic: DynamicInputs
    eigenvectors: Complex[Array, "n_k n_band n_orb"]
    energy_scales: Float[Array, " n_energy"]
    polarizations: Complex[Array, "n_pol 3"]


def _recursive_equation_count(value: object) -> int:
    """Count equations recursively through nested JAXPR parameters."""
    if isinstance(value, ClosedJaxpr):
        return _recursive_equation_count(value.jaxpr)
    if isinstance(value, Jaxpr):
        return len(value.eqns) + sum(
            _recursive_equation_count(parameter)
            for equation in value.eqns
            for parameter in equation.params.values()
        )
    if isinstance(value, (tuple, list)):
        return sum(_recursive_equation_count(item) for item in value)
    if isinstance(value, dict):
        return sum(_recursive_equation_count(item) for item in value.values())
    return 0


def _basis(n_orbitals: int) -> OrbitalBasis:
    """Build independent complete s shells without rotational padding."""
    basis: OrbitalBasis = make_orbital_basis(
        atom_indices=tuple(range(n_orbitals)),
        n=(1,) * n_orbitals,
        l=(0,) * n_orbitals,
        m=(0,) * n_orbitals,
        labels=tuple(f"s_{index}" for index in range(n_orbitals)),
    )
    return basis


def _fixture(n_k: int = N_K, n_orbitals: int = N_ORBITALS) -> Fixture:
    """Construct deterministic literal-shape dynamic benchmark arguments."""
    basis: OrbitalBasis = _basis(n_orbitals)
    shell_index: tuple[int, ...] = tuple(range(n_orbitals))
    matrix_params: MatrixElementParams = make_matrix_element_params(
        basis,
        shell_index,
        sigma_shell=jnp.linspace(
            0.8,
            1.2,
            n_orbitals,
            dtype=jnp.float64,
        ),
        phase_shift_angles_shell=jnp.linspace(
            -0.3,
            0.4,
            n_orbitals,
            dtype=jnp.float64,
        ),
    )
    coordinate: Float[Array, " n_k"] = jnp.linspace(
        -1.0,
        1.0,
        n_k,
        dtype=jnp.float64,
    )
    initial_momentum: Float[Array, "n_k 3"] = jnp.stack(
        (
            0.18 * coordinate,
            0.11 * jnp.sin(1.7 * coordinate),
            jnp.zeros_like(coordinate),
        ),
        axis=-1,
    )
    final_momentum: Float[Array, "n_k 3"] = initial_momentum.at[:, 2].set(
        1.4 + 0.07 * jnp.cos(2.1 * coordinate)
    )
    orbital_coordinate: Float[Array, " n_orb"] = jnp.linspace(
        -1.0,
        1.0,
        n_orbitals,
        dtype=jnp.float64,
    )
    positions: Float[Array, "n_orb 3"] = jnp.stack(
        (
            0.7 * orbital_coordinate,
            0.3 * orbital_coordinate**2,
            0.2 * jnp.sin(orbital_coordinate),
        ),
        axis=-1,
    )
    depths: Float[Array, " n_orb"] = jnp.linspace(
        0.0,
        12.0,
        n_orbitals,
        dtype=jnp.float64,
    )
    radial_amplitude: Float[Array, "n_k n_orb"] = 0.9 + 0.08 * jnp.cos(
        coordinate[:, None] + orbital_coordinate[None, :]
    )
    bvals: Complex[Array, "n_k n_orb 2"] = jnp.stack(
        (
            jnp.zeros_like(radial_amplitude, dtype=jnp.complex128),
            1j * radial_amplitude.astype(jnp.complex128),
        ),
        axis=-1,
    )
    eigenvectors: Complex[Array, "n_k n_band n_orb"] = jnp.broadcast_to(
        jnp.eye(n_orbitals, dtype=jnp.complex128),
        (n_k, n_orbitals, n_orbitals),
    )
    energy_scales: Float[Array, " n_energy"] = jnp.linspace(
        0.94,
        1.08,
        N_ENERGY,
        dtype=jnp.float64,
    )
    polarizations: Complex[Array, "n_pol 3"] = jnp.asarray(
        (
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
            (2.0**-0.5, 2.0**-0.5, 0.0),
            (2.0**-0.5, 1j * 2.0**-0.5, 0.0),
            (0.3, 0.4j, (0.75) ** 0.5),
        ),
        dtype=jnp.complex128,
    )
    dynamic = DynamicInputs(
        initial_momentum,
        final_momentum,
        positions,
        depths,
        bvals,
        matrix_params,
        jnp.asarray(9.0, dtype=jnp.float64),
    )
    return Fixture(
        basis,
        dynamic,
        eigenvectors,
        energy_scales,
        polarizations,
    )


def _channel_function(basis: OrbitalBasis) -> Any:
    """Return the scalar-energy channel primitive with dynamic arrays."""

    def channel(dynamic: DynamicInputs) -> Complex[Array, "n_k 1 n_orb 3"]:
        return orbital_transition_channels(
            dynamic.initial_momentum,
            dynamic.final_momentum,
            dynamic.positions,
            dynamic.depths,
            dynamic.bvals,
            dynamic.matrix_params,
            dynamic.mean_free_path,
            basis,
        )

    return channel


def _group_weights(
    channels: Complex[Array, "n_k 1 n_orb 3"],
    eigenvectors: Complex[Array, "n_k n_band n_orb"],
    polarization: Complex[Array, " 3"],
) -> Float[Array, "n_k n_group"]:
    """Reduce one scalar-energy channel tensor to six complete groups."""
    band_channels: Complex[Array, "n_k n_band 1 3"] = project_band_channels(
        channels,
        eigenvectors,
    )
    amplitudes: Complex[Array, "n_k n_band 1"] = contract_polarization(
        band_channels,
        polarization,
    )
    weights: Float[Array, "n_k n_band"] = jnp.sum(
        jnp.real(jnp.conj(amplitudes) * amplitudes),
        axis=-1,
    )
    groups: Float[Array, "n_k n_group"] = jnp.sum(
        weights.reshape(weights.shape[0], N_GROUPS, -1),
        axis=-1,
    )
    return groups


def _scan_function(basis: OrbitalBasis) -> Any:
    """Return the eight-energy reduced-output scan executable."""
    channel = _channel_function(basis)

    def scan_weights(
        dynamic: DynamicInputs,
        eigenvectors: Complex[Array, "n_k n_band n_orb"],
        energy_scales: Float[Array, " n_energy"],
        polarization: Complex[Array, " 3"],
    ) -> Float[Array, "n_energy n_k n_group"]:
        def body(
            carry: Float[Array, ""],
            scale: Float[Array, ""],
        ) -> tuple[Float[Array, ""], Float[Array, "n_k n_group"]]:
            scaled_final: Float[Array, "n_k 3"] = dynamic.final_momentum.at[
                :, 2
            ].multiply(scale)
            phase: Complex[Array, ""] = jnp.exp(0.17j * (scale - 1.0))
            scaled_bvals: Complex[Array, "n_k n_orb 2"] = dynamic.bvals * phase
            scaled_dynamic: DynamicInputs = dynamic._replace(
                final_momentum=scaled_final,
                bvals=scaled_bvals,
            )
            groups: Float[Array, "n_k n_group"] = _group_weights(
                channel(scaled_dynamic),
                eigenvectors,
                polarization,
            )
            next_carry: Float[Array, ""] = carry + jnp.sum(groups)
            return next_carry, groups

        _, groups = jax.lax.scan(
            body,
            jnp.asarray(0.0, dtype=jnp.float64),
            energy_scales,
        )
        return groups

    return scan_weights


def _scalar_gradient_function(basis: OrbitalBasis) -> Any:
    """Return one scalar-energy primitive plus its sigma gradient."""
    channel = _channel_function(basis)

    def scalar_with_gradient(
        dynamic: DynamicInputs,
        eigenvectors: Complex[Array, "n_k n_band n_orb"],
        polarization: Complex[Array, " 3"],
    ) -> tuple[
        Float[Array, "n_k n_group"],
        Float[Array, " n_shell"],
    ]:
        groups: Float[Array, "n_k n_group"] = _group_weights(
            channel(dynamic),
            eigenvectors,
            polarization,
        )

        def loss(sigma: Float[Array, " n_shell"]) -> Float[Array, ""]:
            changed_params: MatrixElementParams = eqx.tree_at(
                lambda item: item.sigma_shell,
                dynamic.matrix_params,
                sigma,
            )
            changed_dynamic: DynamicInputs = dynamic._replace(
                matrix_params=changed_params
            )
            values: Float[Array, "n_k n_group"] = _group_weights(
                channel(changed_dynamic),
                eigenvectors,
                polarization,
            )
            return jnp.sum(values)

        sigma_gradient: Float[Array, " n_shell"] = jax.grad(loss)(
            dynamic.matrix_params.sigma_shell
        )
        return groups, sigma_gradient

    return scalar_with_gradient


def _checksum_fixture(fixture: Fixture) -> str:
    """Hash all dynamic numerical leaves in deterministic tree order."""
    digest = hashlib.sha256()
    leaf: object
    for leaf in jax.tree.leaves(
        (
            fixture.dynamic,
            fixture.eigenvectors,
            fixture.energy_scales,
            fixture.polarizations,
        )
    ):
        array: Shaped[NDArray, "..."] = np.asarray(jax.device_get(leaf))
        digest.update(str(array.shape).encode())
        digest.update(str(array.dtype).encode())
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _time_call(function: Any, *arguments: object) -> float:
    """Time one synchronized compiled call."""
    start: float = time.perf_counter()
    result: object = function(*arguments)
    jax.block_until_ready(result)
    elapsed: float = time.perf_counter() - start
    return elapsed


def _time_six(
    function: Any,
    arguments: tuple[object, ...],
    pols: Array,
) -> float:
    """Time six independently synchronized polarization calls."""
    start: float = time.perf_counter()
    index: int
    for index in range(N_POLARIZATIONS):
        result: object = function(*arguments, pols[index])
        jax.block_until_ready(result)
    elapsed: float = time.perf_counter() - start
    return elapsed


def _compile(function: Any, *arguments: object) -> tuple[Any, float]:
    """Lower and compile while recording compilation wall time."""
    start: float = time.perf_counter()
    compiled: Any = jax.jit(function).lower(*arguments).compile()
    elapsed: float = time.perf_counter() - start
    return compiled, elapsed


def _memory_record(compiled: Any) -> dict[str, int | bool | str]:
    """Extract the Plan-05 compiler-live allocation authority."""
    analysis: Any = compiled.memory_analysis()
    required: tuple[str, ...] = (
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
    argument_bytes: int = int(analysis.argument_size_in_bytes)
    output_bytes: int = int(analysis.output_size_in_bytes)
    temporary_bytes: int = int(analysis.temp_size_in_bytes)
    alias_bytes: int = int(analysis.alias_size_in_bytes)
    live_bytes: int = (
        argument_bytes + output_bytes + temporary_bytes - alias_bytes
    )
    record: dict[str, int | bool | str] = {
        "authority_available": True,
        "argument_size_bytes": argument_bytes,
        "output_size_bytes": output_bytes,
        "temporary_size_bytes": temporary_bytes,
        "alias_size_bytes": alias_bytes,
        "compiler_live_allocation_bytes": live_bytes,
        "limit_bytes": LIMIT_BYTES,
        "result": "pass" if live_bytes < LIMIT_BYTES else "fail",
    }
    return record


def _array_shapes(ir_text: str) -> set[tuple[int, ...]]:
    """Extract numeric array dimensions from retained JAXPR or HLO text."""
    shapes: set[tuple[int, ...]] = set()
    match: re.Match[str]
    for match in re.finditer(r"\[([0-9,\s]+)\]", ir_text):
        dimensions: tuple[int, ...] = tuple(
            int(value) for value in match.group(1).split(",") if value.strip()
        )
        if dimensions:
            shapes.add(dimensions)
    return shapes


def _s1() -> dict[str, object]:
    """Measure equation-count scaling and fixed-shape compile reuse."""
    orbital_counts: tuple[int, ...] = (9, 18, 36)
    equation_counts: list[int] = []
    n_orbitals: int
    for n_orbitals in orbital_counts:
        fixture: Fixture = _fixture(n_k=2, n_orbitals=n_orbitals)
        channel = _channel_function(fixture.basis)
        jaxpr: ClosedJaxpr = jax.make_jaxpr(channel)(fixture.dynamic)
        equation_counts.append(_recursive_equation_count(jaxpr))

    fixture = _fixture(n_k=8, n_orbitals=N_ORBITALS)
    channel_jit: Any = cast(
        Any,
        jax.jit(_channel_function(fixture.basis)),
    )
    cache_before: int = channel_jit._cache_size()
    first: Array = channel_jit(fixture.dynamic)
    jax.block_until_ready(first)
    cache_after_first: int = channel_jit._cache_size()
    changed_dynamic: DynamicInputs = fixture.dynamic._replace(
        bvals=fixture.dynamic.bvals * (1.0 + 0.01j)
    )
    second: Array = channel_jit(changed_dynamic)
    jax.block_until_ready(second)
    cache_after_second: int = channel_jit._cache_size()

    trace_count: int = 0

    def composed_sweep(
        dynamic: DynamicInputs,
        polarization: Array,
    ) -> Array:
        """Build channels and contract one fixed-shape polarization."""
        nonlocal trace_count
        trace_count += 1
        channels: Array = _channel_function(fixture.basis)(dynamic)
        contracted: Array = contract_polarization(channels, polarization)
        return contracted

    composed_jit: Any = cast(Any, jax.jit(composed_sweep))
    composed_cache_sizes: list[int] = [composed_jit._cache_size()]
    composed_trace_counts: list[int] = [trace_count]
    polarization: Array
    for polarization in fixture.polarizations:
        contracted: Array = composed_jit(changed_dynamic, polarization)
        jax.block_until_ready(contracted)
        composed_cache_sizes.append(composed_jit._cache_size())
        composed_trace_counts.append(trace_count)
    count_growth: int = max(equation_counts) - min(equation_counts)
    result: str = (
        "pass"
        if count_growth < orbital_counts[-1] - orbital_counts[0]
        and (cache_before, cache_after_first, cache_after_second) == (0, 1, 1)
        and composed_cache_sizes == [0, 1, 1, 1, 1, 1, 1]
        and composed_trace_counts == [0, 1, 1, 1, 1, 1, 1]
        else "fail"
    )
    return {
        "orbital_counts": list(orbital_counts),
        "recursive_jaxpr_equation_counts": equation_counts,
        "equation_count_growth": count_growth,
        "compile_cache_sizes": [
            cache_before,
            cache_after_first,
            cache_after_second,
        ],
        "composed_sweep_compile_cache_sizes": composed_cache_sizes,
        "composed_sweep_trace_counts": composed_trace_counts,
        "result": result,
    }


def _s2(
    fixture: Fixture,
    artifact_directory: Path,
) -> tuple[dict[str, object], Any]:
    """Compile the literal scan, retain IR, and record live allocation."""
    scan_function = _scan_function(fixture.basis)
    scan_arguments: tuple[object, ...] = (
        fixture.dynamic,
        fixture.eigenvectors,
        fixture.energy_scales,
        fixture.polarizations[0],
    )
    scalar_function = _scalar_gradient_function(fixture.basis)
    scalar_arguments: tuple[object, ...] = (
        fixture.dynamic,
        fixture.eigenvectors,
        fixture.polarizations[0],
    )
    scan_jaxpr: ClosedJaxpr = jax.make_jaxpr(scan_function)(*scan_arguments)
    scalar_jaxpr: ClosedJaxpr = jax.make_jaxpr(scalar_function)(
        *scalar_arguments
    )
    jaxpr_text: str = (
        "SCALAR-ENERGY VALUE+GRADIENT\n"
        f"{scalar_jaxpr}\n\nEIGHT-ENERGY REDUCED SCAN\n{scan_jaxpr}\n"
    )
    jaxpr_text = re.sub(r"0x[0-9a-fA-F]+", "0xADDR", jaxpr_text)
    scalar_compiled, scalar_compilation_seconds = _compile(
        scalar_function,
        *scalar_arguments,
    )
    scan_compiled, scan_compilation_seconds = _compile(
        scan_function,
        *scan_arguments,
    )
    scalar_hlo: str = scalar_compiled.as_text()
    scan_hlo: str = scan_compiled.as_text()
    hlo_text: str = (
        "SCALAR-ENERGY VALUE+GRADIENT\n"
        f"{scalar_hlo}\n\nEIGHT-ENERGY REDUCED SCAN\n{scan_hlo}\n"
    )
    artifact_directory.mkdir(parents=True, exist_ok=True)
    jaxpr_path: Path = artifact_directory / "plan06_s2_jaxpr.txt.gz"
    hlo_path: Path = artifact_directory / "plan06_s2_hlo.txt.gz"
    jaxpr_path.write_bytes(
        gzip.compress(jaxpr_text.encode(), compresslevel=9, mtime=0)
    )
    hlo_path.write_bytes(
        gzip.compress(hlo_text.encode(), compresslevel=9, mtime=0)
    )
    scalar_output: object = scalar_compiled(*scalar_arguments)
    scan_output: object = scan_compiled(*scan_arguments)
    jax.block_until_ready((scalar_output, scan_output))
    scalar_groups: Array = scalar_output[0]
    sigma_gradient: Array = scalar_output[1]
    groups: Array = scan_output
    parsed_shapes: set[tuple[int, ...]] = _array_shapes(
        f"{jaxpr_text}\n{hlo_text}"
    )
    forbidden_dimensions: tuple[int, int, int] = (
        N_ENERGY,
        N_K,
        N_ORBITALS,
    )
    forbidden_shapes: list[list[int]] = sorted(
        [
            list(shape)
            for shape in parsed_shapes
            if len(shape) == len(forbidden_dimensions)
            and sorted(shape) == sorted(forbidden_dimensions)
        ]
    )
    forbidden_present: bool = bool(forbidden_shapes)
    scalar_memory: dict[str, int | bool | str] = _memory_record(
        scalar_compiled
    )
    scan_memory: dict[str, int | bool | str] = _memory_record(scan_compiled)
    authority_available: bool = bool(
        scalar_memory.get("authority_available")
        and scan_memory.get("authority_available")
    )
    authoritative_live_bytes: int = max(
        int(scalar_memory.get("compiler_live_allocation_bytes", 0)),
        int(scan_memory.get("compiler_live_allocation_bytes", 0)),
    )
    minimum_argument_bytes: int = min(
        int(scalar_memory.get("argument_size_bytes", 0)),
        int(scan_memory.get("argument_size_bytes", 0)),
    )
    record: dict[str, object] = {
        "n_k": N_K,
        "n_orb": N_ORBITALS,
        "n_energy": N_ENERGY,
        "n_complete_groups": N_GROUPS,
        "output_shape": list(groups.shape),
        "scalar_output_shape": list(scalar_groups.shape),
        "gradient_shape": list(sigma_gradient.shape),
        "compilation_seconds": {
            "scalar_value_and_gradient": scalar_compilation_seconds,
            "reduced_scan": scan_compilation_seconds,
        },
        "recursive_jaxpr_equation_counts": {
            "scalar_value_and_gradient": _recursive_equation_count(
                scalar_jaxpr
            ),
            "reduced_scan": _recursive_equation_count(scan_jaxpr),
        },
        "forbidden_k_e_b_shape_present": forbidden_present,
        "forbidden_k_e_b_shapes": forbidden_shapes,
        "parsed_array_shape_count": len(parsed_shapes),
        "jaxpr_gzip": jaxpr_path.name,
        "jaxpr_gzip_sha256": hashlib.sha256(
            jaxpr_path.read_bytes()
        ).hexdigest(),
        "hlo_gzip": hlo_path.name,
        "hlo_gzip_sha256": hashlib.sha256(hlo_path.read_bytes()).hexdigest(),
        "memory_analysis": {
            "authority_available": authority_available,
            "scalar_value_and_gradient": scalar_memory,
            "reduced_scan": scan_memory,
            "authoritative_maximum_live_allocation_bytes": (
                authoritative_live_bytes
            ),
            "limit_bytes": LIMIT_BYTES,
            "result": (
                "pass"
                if authority_available
                and authoritative_live_bytes < LIMIT_BYTES
                else "residual"
            ),
        },
        "result": (
            "pass"
            if groups.shape == (N_ENERGY, N_K, N_GROUPS)
            and scalar_groups.shape == (N_K, N_GROUPS)
            and not forbidden_present
            and authority_available
            and authoritative_live_bytes < LIMIT_BYTES
            and minimum_argument_bytes > MIN_DYNAMIC_ARGUMENT_BYTES
            else "residual"
        ),
    }
    return record, scan_compiled


def _s3(fixture: Fixture) -> dict[str, object]:
    """Record synchronized seven-repetition late-polarization timings."""
    channel = _channel_function(fixture.basis)

    def batch_contract(
        channels: Complex[Array, "n_k 1 n_orb 3"],
        polarizations: Complex[Array, "n_pol 3"],
    ) -> Complex[Array, "n_pol n_k 1 n_orb"]:
        return jax.vmap(
            lambda polarization: contract_polarization(
                channels,
                polarization,
            )
        )(polarizations)

    def single_contract(
        channels: Complex[Array, "n_k 1 n_orb 3"],
        polarization: Complex[Array, " 3"],
    ) -> Complex[Array, "n_k 1 n_orb"]:
        return contract_polarization(channels, polarization)

    def late_pipeline(
        dynamic: DynamicInputs,
        polarizations: Complex[Array, "n_pol 3"],
    ) -> Complex[Array, "n_pol n_k 1 n_orb"]:
        return batch_contract(channel(dynamic), polarizations)

    def rebuild_pipeline(
        dynamic: DynamicInputs,
        polarization: Complex[Array, " 3"],
    ) -> Complex[Array, "n_k 1 n_orb"]:
        return single_contract(channel(dynamic), polarization)

    channel_compiled, channel_compile = _compile(channel, fixture.dynamic)
    channels: Array = channel_compiled(fixture.dynamic)
    jax.block_until_ready(channels)
    batch_compiled, batch_compile = _compile(
        batch_contract,
        channels,
        fixture.polarizations,
    )
    single_compiled, single_compile = _compile(
        single_contract,
        channels,
        fixture.polarizations[0],
    )
    late_compiled, late_compile = _compile(
        late_pipeline,
        fixture.dynamic,
        fixture.polarizations,
    )
    rebuild_compiled, rebuild_compile = _compile(
        rebuild_pipeline,
        fixture.dynamic,
        fixture.polarizations[0],
    )
    for _ in range(WARMUPS):
        _time_call(batch_compiled, channels, fixture.polarizations)
        _time_six(single_compiled, (channels,), fixture.polarizations)
        _time_call(late_compiled, fixture.dynamic, fixture.polarizations)
        _time_six(
            rebuild_compiled,
            (fixture.dynamic,),
            fixture.polarizations,
        )

    batch_raw: list[float] = []
    sequential_raw: list[float] = []
    late_raw: list[float] = []
    rebuild_raw: list[float] = []
    for _ in range(REPETITIONS):
        batch_raw.append(
            _time_call(batch_compiled, channels, fixture.polarizations)
        )
        sequential_raw.append(
            _time_six(single_compiled, (channels,), fixture.polarizations)
        )
        late_raw.append(
            _time_call(late_compiled, fixture.dynamic, fixture.polarizations)
        )
        rebuild_raw.append(
            _time_six(
                rebuild_compiled,
                (fixture.dynamic,),
                fixture.polarizations,
            )
        )
    batch_median: float = statistics.median(batch_raw)
    sequential_median: float = statistics.median(sequential_raw)
    late_median: float = statistics.median(late_raw)
    rebuild_median: float = statistics.median(rebuild_raw)
    contraction_ratio: float = batch_median / sequential_median
    pipeline_ratio: float = late_median / rebuild_median
    return {
        "n_k": N_K,
        "n_orb": N_ORBITALS,
        "n_polarization": N_POLARIZATIONS,
        "warmups": WARMUPS,
        "repetitions": REPETITIONS,
        "synchronized": True,
        "compilation_seconds": {
            "channel": channel_compile,
            "batch_contraction": batch_compile,
            "single_contraction": single_compile,
            "late_reuse_pipeline": late_compile,
            "single_rebuild_pipeline": rebuild_compile,
        },
        "raw_seconds": {
            "batched_contraction": batch_raw,
            "six_sequential_contractions": sequential_raw,
            "late_reuse_pipeline": late_raw,
            "six_rebuild_pipelines": rebuild_raw,
        },
        "median_seconds": {
            "batched_contraction": batch_median,
            "six_sequential_contractions": sequential_median,
            "late_reuse_pipeline": late_median,
            "six_rebuild_pipelines": rebuild_median,
        },
        "contraction_ratio": contraction_ratio,
        "pipeline_ratio": pipeline_ratio,
        "result": (
            "pass"
            if contraction_ratio < MAX_CONTRACTION_RATIO
            and pipeline_ratio < 1.0
            else "environment-sensitive failure"
        ),
    }


def main() -> None:
    """Run all literal gates and write JSON plus retained compressed IR."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifact-directory",
        type=Path,
        default=Path("tests/test_diffpes/_reference_data/plan06_scalability"),
    )
    arguments: argparse.Namespace = parser.parse_args()
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    jax.config.update("jax_enable_x64", True)
    host_setup_start: float = time.perf_counter()
    fixture: Fixture = _fixture()
    jax.block_until_ready(
        (
            fixture.dynamic,
            fixture.eigenvectors,
            fixture.energy_scales,
            fixture.polarizations,
        )
    )
    host_setup_seconds: float = time.perf_counter() - host_setup_start
    s1: dict[str, object] = _s1()
    s2: dict[str, object]
    s2, _ = _s2(fixture, arguments.artifact_directory)
    s3: dict[str, object] = _s3(fixture)
    peak_rss_raw: int = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    peak_rss_bytes: int = (
        peak_rss_raw if sys.platform == "darwin" else peak_rss_raw * 1024
    )
    repository_root: Path = Path(__file__).resolve().parents[2]
    bound_source_paths: tuple[Path, ...] = (
        Path(__file__).resolve(),
        repository_root / "src/diffpes/simul/matrixel.py",
        repository_root / "src/diffpes/types/radial_params.py",
    )
    source_sha256: dict[str, str] = {
        str(path.relative_to(repository_root)): hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
        for path in bound_source_paths
    }
    artifact: dict[str, object] = {
        "schema": "diffpes.plan06.scalability.v2",
        "gates": ["06.S1", "06.S2", "06.S3"],
        "device": str(jax.devices()[0]),
        "cpu": platform.processor(),
        "platform": platform.platform(),
        "jax_version": jax.__version__,
        "numpy_version": np.__version__,
        "dtype_policy": "JAX x64",
        "host_setup_seconds": host_setup_seconds,
        "process_peak_rss_bytes_non_authoritative": peak_rss_bytes,
        "dynamic_input_sha256": _checksum_fixture(fixture),
        "source_sha256": source_sha256,
        "s1": s1,
        "s2": s2,
        "s3": s3,
    }
    output_path: Path = arguments.artifact_directory / "plan06_s1_s3_cpu.json"
    output_path.write_text(
        json.dumps(artifact, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
