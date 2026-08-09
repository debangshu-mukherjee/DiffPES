#!/usr/bin/env python
"""Measure chunked slab diagonalization memory and retracing.

Run this script explicitly on a CPU worker; routine pytest uses bounded
fixtures and structural JAXPR checks. The defaults execute the registered
80-layer, four-orbital-per-layer slab (320 orbitals), 256-k-point, chunk-32
S1 forward case. The same real slab pipeline can execute the 640-orbital
spinor stretch. A smaller 64-orbital case checks the rematerialized band-loss
gradient against the non-chunked path without intentionally constructing the
production ``(K, O, O)`` batch.
"""

from __future__ import annotations

import argparse
import json
import resource
import time
from collections.abc import Callable
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp

from diffpes.tightb import (
    eigvalsh_bands,
    eigvalsh_bands_chunked,
    gen_slab,
    spin_double_model,
)
from diffpes.types import (
    CrystalGeometry,
    OrbitalBasis,
    SlabSpec,
    TBModel,
    make_crystal_geometry,
    make_orbital_basis,
    make_tb_model,
)

_COMPLEX128_BYTES: int = 16
_GIB: int = 2**30
_GRADIENT_RTOL: float = 1e-12
_NONZERO_GRADIENT_MIN: float = 1e-9
_THICKNESS_TRACE_COUNT: int = 2
_TERMINATION_TRACE_COUNT: int = 3


def _bulk_model() -> TBModel:
    """PRIVATE: Build a dense four-orbital bulk model for slab extrusion.

    Returns
    -------
    model : TBModel
        Single-site orthorhombic model with four s-like orbitals,
        dense complex 4x4 hopping blocks along all six first-neighbor
        cells, and fixed onsite energies in eV.

    Notes
    -----
    Deterministic sine/cosine seeds fill the three directed blocks;
    conjugate transposes close the Hermitian partners, so the Bloch
    Hamiltonian stays Hermitian at every k.
    """
    geometry: CrystalGeometry = make_crystal_geometry(
        lattice=jnp.diag(jnp.asarray((2.2, 2.5, 1.3))),
        positions=jnp.zeros((1, 3), dtype=jnp.float64),
        species=("X",),
    )
    basis: OrbitalBasis = make_orbital_basis(
        atom_indices=(0,) * 4,
        n=(1, 2, 3, 4),
        l=(0,) * 4,
        m=(0,) * 4,
        labels=("s1", "s2", "s3", "s4"),
    )
    pairs: tuple[tuple[int, int], ...] = tuple(
        (row, column) for row in range(4) for column in range(4)
    )
    seed: jax.Array = jnp.arange(16, dtype=jnp.float64).reshape(4, 4)
    blocks: tuple[jax.Array, ...] = tuple(
        scale
        * (jnp.sin(seed + phase) + 1j * jnp.cos(0.7 * seed + 0.3 * phase))
        for scale, phase in ((0.11, 0.2), (0.08, 0.7), (0.19, 1.1))
    )
    directed_blocks: tuple[jax.Array, ...] = (
        blocks[0],
        blocks[0].conj().T,
        blocks[1],
        blocks[1].conj().T,
        blocks[2],
        blocks[2].conj().T,
    )
    cells: tuple[tuple[int, int, int], ...] = (
        (1, 0, 0),
        (-1, 0, 0),
        (0, 1, 0),
        (0, -1, 0),
        (0, 0, 1),
        (0, 0, -1),
    )
    return make_tb_model(
        hopping_amplitudes=jnp.concatenate(
            tuple(block.reshape(-1) for block in directed_blocks)
        ),
        onsite_energies=jnp.asarray((-1.3, -0.31, 0.48, 1.7)),
        soc_lambdas=jnp.zeros((0,)),
        geometry=geometry,
        basis=basis,
        hopping_pairs=pairs * len(cells),
        hopping_cells=tuple(cell for cell in cells for _ in range(len(pairs))),
        shell_index=(-1,) * 4,
    )


def _slab_model(
    n_layers: int,
    *,
    spinor: bool = False,
) -> tuple[TBModel, SlabSpec]:
    """PRIVATE: Extrude one actual four-orbital-per-layer slab design.

    Parameters
    ----------
    n_layers : int
        Target layer count of the (001) slab.
    spinor : bool
        Whether to spin-double the bulk before extrusion.

    Returns
    -------
    slab : tuple[TBModel, SlabSpec]
        Extruded slab model and its specification.

    Raises
    ------
    RuntimeError
        If the extrusion yields an unexpected orbital count.

    Notes
    -----
    The thickness ``(n_layers - 1) * 1.3`` Angstrom matches the bulk
    c-axis spacing, so ``gen_slab`` produces exactly ``n_layers``
    layers with 8 Angstrom of vacuum.
    """
    bulk: TBModel = _bulk_model()
    if spinor:
        bulk = spin_double_model(bulk)
    model: TBModel
    specification: SlabSpec
    model, specification = gen_slab(
        bulk,
        miller=(0, 0, 1),
        thickness_ang=(n_layers - 1) * 1.3,
        vacuum_ang=8.0,
    )
    expected_orbitals: int = n_layers * (8 if spinor else 4)
    if model.onsite_energies.shape != (expected_orbitals,):
        raise RuntimeError(
            "real slab extrusion produced the wrong orbital count"
        )
    return model, specification


def _termination_bulk_model() -> TBModel:
    """PRIVATE: Build the same Hamiltonian on an alternating X/Y basis.

    Returns
    -------
    model : TBModel
        The :func:`_bulk_model` couplings on a two-species cell with
        two orbitals per atom.

    Notes
    -----
    Species alternation along c makes the two (001) terminations
    distinguishable, which the retracing measurement needs.
    """
    reference: TBModel = _bulk_model()
    geometry: CrystalGeometry = make_crystal_geometry(
        lattice=reference.geometry.lattice,
        positions=jnp.asarray(
            ((0.0, 0.0, 0.0), (0.0, 0.0, 0.5)),
            dtype=jnp.float64,
        ),
        species=("X", "Y"),
    )
    basis: OrbitalBasis = make_orbital_basis(
        atom_indices=(0, 0, 1, 1),
        n=(1, 2, 1, 2),
        l=(0,) * 4,
        m=(0,) * 4,
        labels=("X-s1", "X-s2", "Y-s1", "Y-s2"),
    )
    return make_tb_model(
        hopping_amplitudes=reference.hopping_amplitudes,
        onsite_energies=reference.onsite_energies,
        soc_lambdas=reference.soc_lambdas,
        geometry=geometry,
        basis=basis,
        hopping_pairs=reference.hopping_pairs,
        hopping_cells=reference.hopping_cells,
        shell_index=reference.shell_index,
    )


def _termination_slab(
    thickness_ang: float,
    termination: tuple[str, str],
) -> tuple[TBModel, SlabSpec]:
    """PRIVATE: Extrude one actual alternating-species termination design.

    Parameters
    ----------
    thickness_ang : float
        Slab thickness in Angstrom.
    termination : tuple[str, str]
        Requested bottom and top species.

    Returns
    -------
    slab : tuple[TBModel, SlabSpec]
        Extruded slab model and its specification.

    Notes
    -----
    The call fixes the (001) direction and 8 Angstrom of vacuum and
    forwards the termination request to ``gen_slab``.
    """
    return gen_slab(
        _termination_bulk_model(),
        miller=(0, 0, 1),
        thickness_ang=thickness_ang,
        vacuum_ang=8.0,
        termination=termination,
    )


def _kpoints(n_kpoints: int) -> jax.Array:
    """PRIVATE: Build one generic padded path.

    Parameters
    ----------
    n_kpoints : int
        Number of path points.

    Returns
    -------
    kpoints : jax.Array
        ``(n_kpoints, 3)`` fractional coordinates on a slanted
        in-plane line with zero third component.

    Notes
    -----
    The incommensurate slope keeps the path free of symmetry
    coincidences, so no eigenvalue degeneracy is accidental.
    """
    k_x = jnp.linspace(-0.47, 0.43, n_kpoints)
    return jnp.stack(
        (k_x, 0.17 * k_x + 0.03, jnp.zeros_like(k_x)),
        axis=-1,
    )


def _statistics_dict(statistics: Any) -> dict[str, int]:
    """PRIVATE: Normalize the backend's compiler-memory record.

    Parameters
    ----------
    statistics : Any
        Backend ``memory_analysis`` record.

    Returns
    -------
    memory : dict[str, int]
        The eight argument/output/alias/temp byte counters, with
        absent or ``None`` fields as zero.

    Notes
    -----
    ``getattr`` with a zero default absorbs backend differences in
    which counters exist.
    """
    fields: tuple[str, ...] = (
        "argument_size_in_bytes",
        "output_size_in_bytes",
        "alias_size_in_bytes",
        "temp_size_in_bytes",
        "host_argument_size_in_bytes",
        "host_output_size_in_bytes",
        "host_alias_size_in_bytes",
        "host_temp_size_in_bytes",
    )
    return {field: int(getattr(statistics, field, 0) or 0) for field in fields}


def _maximum_rss_bytes() -> int:
    """PRIVATE: Return Linux maximum resident-set size in bytes.

    Returns
    -------
    rss_bytes : int
        Process high-water RSS in bytes.

    Notes
    -----
    Linux reports ``ru_maxrss`` in KiB, so the value scales by 1024.
    """
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024


def _gradient_check(n_layers: int, n_kpoints: int, chunk_size: int) -> dict:
    """PRIVATE: Compare rematerialized and ordinary band-loss derivatives.

    Parameters
    ----------
    n_layers : int
        Layer count of the bounded gradient slab.
    n_kpoints : int
        Number of path k-points.
    chunk_size : int
        Chunk length of the rematerialized path.

    Returns
    -------
    record : dict
        Both scalar gradients, their relative error, the nonzero
        witness, and the 1e-12 relative-tolerance verdict.

    Implementation Logic
    --------------------
    A scalar loss sums a smooth function of the bands of a globally
    scaled model.  ``jax.grad`` with respect to the scale runs once
    through ``eigvalsh_bands_chunked`` and once through the ordinary
    path; both must agree to 1e-12 relative on a nonzero gradient.
    """
    model, _ = _slab_model(n_layers)
    n_orbitals: int = model.onsite_energies.shape[0]
    kpoints: jax.Array = _kpoints(n_kpoints)

    def loss(scale: jax.Array, chunked: bool) -> jax.Array:
        changed: TBModel = eqx.tree_at(
            lambda item: item.hopping_amplitudes,
            model,
            scale * model.hopping_amplitudes,
        )
        bands: jax.Array = (
            eigvalsh_bands_chunked(changed, kpoints, chunk_size)
            if chunked
            else eigvalsh_bands(changed, kpoints)
        )
        return jnp.sum(jnp.sin(0.7 * bands) + 0.13 * bands**2)

    chunked_gradient: jax.Array = jax.grad(loss, argnums=0)(1.1, True)
    ordinary_gradient: jax.Array = jax.grad(loss, argnums=0)(1.1, False)
    relative_error: jax.Array = jnp.abs(
        (chunked_gradient - ordinary_gradient) / ordinary_gradient
    )
    nonzero_gradient: bool = bool(
        jnp.abs(ordinary_gradient) > _NONZERO_GRADIENT_MIN
    )
    return {
        "orbitals": n_orbitals,
        "kpoints": n_kpoints,
        "chunked_gradient": float(chunked_gradient),
        "ordinary_gradient": float(ordinary_gradient),
        "relative_error": float(relative_error),
        "nonzero_gradient": nonzero_gradient,
        "passes_rtol_1e-12": bool(
            nonzero_gradient
            and jnp.isfinite(relative_error)
            and relative_error <= _GRADIENT_RTOL
        ),
    }


def _compile_count(
    kpoints: jax.Array,
    chunk_size: int,
) -> dict:
    """PRIVATE: Count traces across padded lengths and design changes.

    Parameters
    ----------
    kpoints : jax.Array
        Full padded k-point path.
    chunk_size : int
        Chunk length of the rematerialized path.

    Returns
    -------
    record : dict
        The three design descriptions, the padded lengths, the trace
        count after each stage, and the pass verdict.

    Implementation Logic
    --------------------
    A counting wrapper under ``eqx.filter_jit`` runs three padded
    active lengths on one fixed design (one trace expected, since the
    mask changes only data), then a thickness change and a
    termination change (one new trace each, since both change the
    design structure).
    """
    trace_count: list[int] = [0]

    def counted(
        candidate: TBModel,
        points: jax.Array,
        active_mask: jax.Array,
    ) -> jax.Array:
        trace_count[0] += 1
        values = eigvalsh_bands_chunked(candidate, points, chunk_size)
        return jnp.sum(values * active_mask[:, None])

    compiled: Callable[..., jax.Array] = eqx.filter_jit(counted)
    fixed_model, fixed_specification = _termination_slab(
        79 * 1.3,
        ("X", "Y"),
    )
    thickness_model, thickness_specification = _termination_slab(
        80 * 1.3,
        ("X", "Y"),
    )
    termination_model, termination_specification = _termination_slab(
        80 * 1.3,
        ("Y", "X"),
    )
    lengths: tuple[int, ...] = (
        kpoints.shape[0] // 4,
        kpoints.shape[0] // 2,
        kpoints.shape[0],
    )
    for active_length in lengths:
        mask = jnp.arange(kpoints.shape[0]) < active_length
        compiled(fixed_model, kpoints, mask).block_until_ready()
    fixed_design_traces: int = trace_count[0]
    compiled(
        thickness_model,
        kpoints,
        jnp.ones((kpoints.shape[0],), dtype=bool),
    ).block_until_ready()
    thickness_traces: int = trace_count[0]
    compiled(
        termination_model,
        kpoints,
        jnp.ones((kpoints.shape[0],), dtype=bool),
    ).block_until_ready()
    termination_traces: int = trace_count[0]
    return {
        "fixed_design": {
            "layers": fixed_specification.n_layers,
            "termination": fixed_specification.termination,
        },
        "thickness_design": {
            "layers": thickness_specification.n_layers,
            "termination": thickness_specification.termination,
        },
        "termination_design": {
            "layers": termination_specification.n_layers,
            "termination": termination_specification.termination,
        },
        "padded_lengths": lengths,
        "fixed_design_trace_count": fixed_design_traces,
        "after_thickness_change": thickness_traces,
        "after_termination_change": termination_traces,
        "passes": (
            fixed_design_traces == 1
            and thickness_traces == _THICKNESS_TRACE_COUNT
            and termination_traces == _TERMINATION_TRACE_COUNT
        ),
    }


def main() -> None:
    """Execute the production forward measurement and bounded gradient check."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--layers", type=int, default=80)
    parser.add_argument("--spinor", action="store_true")
    parser.add_argument("--kpoints", type=int, default=256)
    parser.add_argument("--chunk-size", type=int, default=32)
    parser.add_argument("--gradient-layers", type=int, default=16)
    parser.add_argument("--gradient-kpoints", type=int, default=64)
    arguments = parser.parse_args()
    if arguments.kpoints % arguments.chunk_size:
        parser.error("--kpoints must be divisible by --chunk-size")
    if arguments.gradient_kpoints % arguments.chunk_size:
        parser.error("--gradient-kpoints must be divisible by --chunk-size")

    model: TBModel
    specification: SlabSpec
    model, specification = _slab_model(
        arguments.layers,
        spinor=arguments.spinor,
    )
    n_orbitals: int = model.onsite_energies.shape[0]
    kpoints: jax.Array = _kpoints(arguments.kpoints)
    forward = jax.jit(
        lambda points: eigvalsh_bands_chunked(
            model,
            points,
            arguments.chunk_size,
        )
    )
    rss_before: int = _maximum_rss_bytes()
    compile_start: float = time.perf_counter()
    executable = forward.lower(kpoints).compile()
    compile_seconds: float = time.perf_counter() - compile_start
    statistics = executable.memory_analysis()
    if statistics is None:
        raise RuntimeError(
            "active JAX backend did not report memory statistics"
        )
    memory: dict[str, int] = _statistics_dict(statistics)
    run_start: float = time.perf_counter()
    values: jax.Array = executable(kpoints)
    values.block_until_ready()
    run_seconds: float = time.perf_counter() - run_start
    rss_after: int = _maximum_rss_bytes()
    compiler_live_bytes: int = (
        memory["argument_size_in_bytes"]
        + memory["output_size_in_bytes"]
        + memory["temp_size_in_bytes"]
        - memory["alias_size_in_bytes"]
    )
    chunk_hamiltonian_bytes: int = (
        arguments.chunk_size * n_orbitals * n_orbitals * _COMPLEX128_BYTES
    )
    result: dict[str, Any] = {
        "gate": "chunked-slab-forward-memory and chunked-slab-gradient-retracing",
        "backend": jax.default_backend(),
        "jax_version": jax.__version__,
        "layers": specification.n_layers,
        "orbitals_per_layer": 8 if arguments.spinor else 4,
        "orbitals": n_orbitals,
        "spinor": model.spinor,
        "termination": specification.termination,
        "kpoints": arguments.kpoints,
        "chunk_size": arguments.chunk_size,
        "compile_seconds": compile_seconds,
        "run_seconds": run_seconds,
        "compiler_memory": memory,
        "compiler_live_bytes": compiler_live_bytes,
        "chunk_hamiltonian_bytes": chunk_hamiltonian_bytes,
        "max_rss_before_bytes": rss_before,
        "max_rss_after_bytes": rss_after,
        "under_one_gib_compiler_budget": compiler_live_bytes < _GIB,
        "under_one_gib_process_high_water": rss_after < _GIB,
        "gradient": _gradient_check(
            arguments.gradient_layers,
            arguments.gradient_kpoints,
            arguments.chunk_size,
        ),
        "compile_count": _compile_count(
            kpoints,
            arguments.chunk_size,
        ),
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
