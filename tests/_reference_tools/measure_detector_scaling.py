"""Measure the literal Plan-08a detector-driver scalability gates on CPU.

The isolated harness ahead-of-time compiles the exact
``256 x 256 x 400``, 20-band coherent driver.  XLA compiler allocation
analysis is the peak-live device-memory authority; host RSS is recorded only
as a diagnostic.  Small executable companions compare checkpointed and
non-rematerialized values and Hamiltonian gradients, count compilations while
native widths and fixed-length photon-energy batches vary, and exercise a
``vmap`` over complete ``ExperimentGeometry`` leaves.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import os
import platform
import resource
import sys
import time
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from beartype.typing import Any, Dict, Iterable, Tuple
from jaxtyping import Array, Complex128, Float64

from diffpes.simul import simulate_arpes
from diffpes.types import (
    DetectorCalibration,
    DetectorEffects,
    DiagonalizedBands,
    ExperimentGeometry,
    FinalStateSpec,
    KGrid,
    MatrixElementParams,
    RadialQuadratureSpec,
    RadialSpec,
    SelfEnergyModel,
    make_crystal_geometry,
    make_detector_calibration,
    make_detector_effects,
    make_diagonalized_bands,
    make_experiment_geometry,
    make_final_state_spec,
    make_kgrid,
    make_matrix_element_params,
    make_orbital_basis,
    make_radial_quadrature_spec,
    make_radial_spec,
    make_self_energy_model,
)


def _repository_root() -> Path:
    """PRIVATE: Return the repository root containing this harness.

    Returns
    -------
    root : Path
        Absolute repository root.
    """
    root: Path = Path(__file__).resolve().parents[2]
    return root


def _output_path() -> Path:
    """PRIVATE: Return the committed detector-scaling artifact path.

    Returns
    -------
    path : Path
        JSON artifact path under the test reference-data tree.
    """
    path: Path = (
        _repository_root()
        / "tests"
        / "test_diffpes"
        / "_reference_data"
        / "detector_scalability"
        / "cpu_benchmark.json"
    )
    return path


def _literal_dimensions() -> Tuple[int, int, int, int, int, int]:
    """PRIVATE: Return the frozen cube, band, and chunk dimensions.

    Returns
    -------
    dimensions : Tuple[int, int, int, int, int, int]
        ``n_kx``, ``n_ky``, ``n_energy``, ``n_band``, k chunk, and energy
        chunk.
    """
    dimensions: Tuple[int, int, int, int, int, int] = (
        256,
        256,
        400,
        20,
        32,
        16,
    )
    return dimensions


def _memory_budgets() -> Tuple[int, int]:
    """PRIVATE: Return the frozen forward and gradient byte budgets.

    Returns
    -------
    budgets : Tuple[int, int]
        Two and twelve decimal gigabytes, respectively.
    """
    budgets: Tuple[int, int] = (2_000_000_000, 12_000_000_000)
    return budgets


def _sha256(path: Path) -> str:
    """PRIVATE: Return the SHA-256 digest of one complete file.

    Parameters
    ----------
    path : Path
        File whose byte identity is required.

    Returns
    -------
    digest : str
        Lowercase hexadecimal SHA-256 digest.
    """
    digest: str = hashlib.sha256(path.read_bytes()).hexdigest()
    return digest


def _maximum_rss_bytes() -> int:
    """PRIVATE: Return the Linux process high-water RSS in bytes.

    Returns
    -------
    maximum_rss : int
        Process high-water resident allocation in bytes.

    Notes
    -----
    Linux reports ``ru_maxrss`` in kibibytes.  This whole-process quantity
    includes Python and compiler caches and is not the device-memory gate.
    """
    maximum_rss: int = (
        int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024
    )
    return maximum_rss


def _memory_record(compiled: Any) -> Dict[str, int | bool | str]:
    """PRIVATE: Normalize XLA compiler allocation counters.

    Parameters
    ----------
    compiled : Any
        Ahead-of-time compiled JAX executable.

    Returns
    -------
    record : Dict[str, int | bool | str]
        Backend counters and the derived compiler-live allocation.

    Notes
    -----
    The live allocation is arguments plus outputs plus temporaries minus
    aliases.  Unlike process RSS, these counters describe the compiled device
    program itself.
    """
    analysis: Any = compiled.memory_analysis()
    names: Tuple[str, ...] = (
        "argument_size_in_bytes",
        "output_size_in_bytes",
        "temp_size_in_bytes",
        "alias_size_in_bytes",
    )
    if analysis is None or any(
        getattr(analysis, name, None) is None for name in names
    ):
        return {
            "authority_available": False,
            "result": "residual: XLA memory_analysis unavailable",
        }
    arguments: int = int(analysis.argument_size_in_bytes)
    outputs: int = int(analysis.output_size_in_bytes)
    temporaries: int = int(analysis.temp_size_in_bytes)
    aliases: int = int(analysis.alias_size_in_bytes)
    return {
        "authority_available": True,
        "argument_size_bytes": arguments,
        "output_size_bytes": outputs,
        "temporary_size_bytes": temporaries,
        "alias_size_bytes": aliases,
        "compiler_live_allocation_bytes": (
            arguments + outputs + temporaries - aliases
        ),
        "result": "measured",
    }


def _small_fixture() -> Dict[str, Any]:
    """PRIVATE: Build a tiny but complete coherent detector fixture.

    Returns
    -------
    fixture : Dict[str, Any]
        Every canonical-driver input except the checkpoint selector.

    Notes
    -----
    The asymmetric source, nonuniform sampled energies, finite depth, and
    generic complex polarization keep the Hamiltonian and geometry paths
    live.  The detector target is independently calibrated rather than
    inferred from source extrema.
    """
    crystal: Any = make_crystal_geometry(
        2.0 * jnp.pi * jnp.eye(3, dtype=jnp.float64),
        jnp.zeros((1, 3), dtype=jnp.float64),
        ("X",),
    )
    basis: Any = make_orbital_basis(
        atom_indices=(0,),
        n=(1,),
        l=(0,),
        m=(0,),
        labels=("1s",),
    )
    kx: Float64[Array, " two"] = jnp.asarray([0.025, 0.13])
    ky: Float64[Array, " two"] = jnp.asarray([-0.04, 0.075])
    mesh_x: Float64[Array, "2 2"]
    mesh_y: Float64[Array, "2 2"]
    mesh_x, mesh_y = jnp.meshgrid(kx, ky, indexing="xy")
    kpoints: Float64[Array, "4 3"] = jnp.stack(
        (mesh_x, mesh_y, jnp.zeros_like(mesh_x)), axis=-1
    ).reshape((-1, 3))
    eigenvalues: Float64[Array, " 4"] = -0.08 + 0.7 * jnp.sum(
        kpoints[:, :2] ** 2, axis=-1
    )
    bands: DiagonalizedBands = make_diagonalized_bands(
        eigenvalues[:, None],
        jnp.ones((4, 1, 1), dtype=jnp.complex128),
        kpoints,
        crystal,
        basis,
        fermi_energy=0.0,
        depths=jnp.asarray([0.65]),
    )
    hamiltonians: Complex128[Array, "4 1 1"] = eigenvalues[
        :, None, None
    ].astype(jnp.complex128)
    radial: RadialSpec = make_radial_spec(
        basis,
        (0,),
        mode="fixed",
        fixed_integrals_shell=jnp.asarray([[0.0, 1.0]]),
    )
    matrix_params: MatrixElementParams = make_matrix_element_params(
        basis,
        (0,),
        sigma_shell=jnp.asarray([1.17]),
        phase_shift_angles_shell=jnp.asarray([0.23]),
    )
    geometry: ExperimentGeometry = make_experiment_geometry(
        photon_energy_ev=50.0,
        polarization=jnp.asarray([1.0, 0.4j, 0.0]),
        sample_azimuth=0.17,
        work_function_ev=4.5,
        temperature_k=30.0,
        mean_free_path_ang=8.0,
    )
    energy_axis: Float64[Array, " five"] = jnp.asarray(
        [-0.22, -0.09, -0.015, 0.055, 0.18]
    )
    calibration: DetectorCalibration = make_detector_calibration(
        u_bin_edges=jnp.asarray([0.002, 0.012, 0.022]),
        v_bin_edges=jnp.asarray([-0.006, 0.0, 0.006]),
        energy_bin_edges_ev=jnp.asarray([-0.25, -0.05, 0.2]),
        psf_fwhm_u=0.01,
        psf_fwhm_v=0.01,
        psf_fwhm_energy_ev=0.02,
        transmission_reference_domain_ev=jnp.asarray([44.0, 47.0]),
    )
    detector_effects: DetectorEffects = make_detector_effects(
        domain_logits=jnp.asarray([0.0]),
        domain_euler_angles_rad=jnp.zeros((1, 3)),
        transmission_raw_slopes=jnp.asarray([-0.4, 0.2]),
        background_coefficients=jnp.asarray([-2.0]),
        sensitivity_coefficients=jnp.asarray([]),
        exposure=1.0,
        background_mode="flat",
        sensitivity_mode="constant",
        domain_frame_ids=("org.diffpes.frame.sample_cartesian",),
    )
    return {
        "hamiltonians": hamiltonians,
        "bands": bands,
        "radial": radial,
        "matrix_params": matrix_params,
        "quadrature": make_radial_quadrature_spec(),
        "final_state": make_final_state_spec(),
        "geometry": geometry,
        "self_energy": make_self_energy_model(gamma=0.04),
        "kgrid": make_kgrid(kpoints, mesh_shape=(2, 2), kz=0.0),
        "energy_axis": energy_axis,
        "calibration": calibration,
        "detector_effects": detector_effects,
    }


def _driver_counts(
    fixture: Dict[str, Any],
    hamiltonians: Complex128[Array, "n_k n_orb n_orb"],
    geometry: ExperimentGeometry,
    calibration: DetectorCalibration,
    *,
    checkpoint: bool,
    k_chunk: int = 2,
    energy_chunk: int = 3,
) -> Float64[Array, "1 u v e"]:
    """PRIVATE: Evaluate expected counts for one fixture state.

    Parameters
    ----------
    fixture : Dict[str, Any]
        Fixed canonical-driver carriers.
    hamiltonians : Complex128[Array, "n_k n_orb n_orb"]
        Explicit Hamiltonian raster.
    geometry : ExperimentGeometry
        One traced acquisition geometry.
    calibration : DetectorCalibration
        Native detector target and point-spread state.
    checkpoint : bool
        Static rematerialization selector.
    k_chunk : int, optional
        Static momentum chunk. Default is two.
    energy_chunk : int, optional
        Static sampled-energy chunk. Default is three.

    Returns
    -------
    counts : Float64[Array, "1 U V E"]
        Complete deterministic expected-count raster.
    """
    counts: Float64[Array, "1 u v e"] = simulate_arpes(
        (hamiltonians,),
        (fixture["bands"],),
        fixture["radial"],
        fixture["matrix_params"],
        fixture["quadrature"],
        fixture["final_state"],
        geometry,
        fixture["self_energy"],
        fixture["kgrid"],
        fixture["energy_axis"],
        calibration,
        fixture["detector_effects"],
        k_chunk=k_chunk,
        energy_chunk=energy_chunk,
        checkpoint=checkpoint,
    ).expected_counts
    return counts


def _remat_record() -> Dict[str, float | bool | str]:
    """PRIVATE: Compare checkpointed and non-rematerialized full drivers.

    Returns
    -------
    record : Dict[str, float | bool | str]
        Value/gradient errors, nonzero-gradient check, and verdict.
    """
    fixture: Dict[str, Any] = _small_fixture()
    hamiltonians: Complex128[Array, "4 1 1"] = fixture["hamiltonians"]
    weights: Float64[Array, "1 2 2 2"] = jnp.asarray(
        [[[[0.7, 1.1], [0.8, 1.2]], [[1.4, 0.9], [1.3, 0.6]]]]
    )

    def loss(
        candidate: Complex128[Array, "4 1 1"], *, checkpoint: bool
    ) -> Float64[Array, ""]:
        """Return an asymmetric scalar expected-count loss."""
        counts: Float64[Array, "1 2 2 2"] = _driver_counts(
            fixture,
            candidate,
            fixture["geometry"],
            fixture["calibration"],
            checkpoint=checkpoint,
        )
        scalar: Float64[Array, ""] = jnp.sum(counts * weights)
        return scalar

    checkpointed_value: Float64[Array, ""]
    checkpointed_gradient: Complex128[Array, "4 1 1"]
    checkpointed_value, checkpointed_gradient = jax.value_and_grad(
        lambda candidate: loss(candidate, checkpoint=True)
    )(hamiltonians)
    plain_value: Float64[Array, ""]
    plain_gradient: Complex128[Array, "4 1 1"]
    plain_value, plain_gradient = jax.value_and_grad(
        lambda candidate: loss(candidate, checkpoint=False)
    )(hamiltonians)
    value_error: float = float(jnp.abs(checkpointed_value - plain_value))
    gradient_error: float = float(
        jnp.max(jnp.abs(checkpointed_gradient - plain_gradient))
    )
    maximum_gradient: float = float(jnp.max(jnp.abs(plain_gradient)))
    value_passes: bool = bool(
        jnp.allclose(
            checkpointed_value, plain_value, rtol=1.0e-12, atol=1.0e-14
        )
    )
    gradient_passes: bool = bool(
        jnp.allclose(
            checkpointed_gradient,
            plain_gradient,
            rtol=1.0e-12,
            atol=1.0e-13,
        )
    )
    nonzero_gradient: bool = maximum_gradient > 1.0e-10
    return {
        "program": "full_driver_value_and_hamiltonian_gradient",
        "mapping_chart": "general rotation with strict target enclosure",
        "checkpointed_value": float(checkpointed_value),
        "non_rematerialized_value": float(plain_value),
        "maximum_value_absolute_error": value_error,
        "maximum_gradient_absolute_error": gradient_error,
        "maximum_reference_gradient": maximum_gradient,
        "value_passes_rtol_1e_12": value_passes,
        "gradient_passes_rtol_1e_12": gradient_passes,
        "nonzero_gradient": nonzero_gradient,
        "result": (
            "pass"
            if value_passes and gradient_passes and nonzero_gradient
            else "fail"
        ),
    }


def _geometry_batch(
    photon_energies: Float64[Array, " batch"],
) -> ExperimentGeometry:
    """PRIVATE: Build one stacked geometry PyTree with a leading batch axis.

    Parameters
    ----------
    photon_energies : Float64[Array, " batch"]
        Fixed-length photon-energy sweep.

    Returns
    -------
    geometry_batch : ExperimentGeometry
        Every numerical geometry leaf stacked on the same leading axis.
    """
    geometries: Tuple[ExperimentGeometry, ...] = tuple(
        make_experiment_geometry(
            photon_energy_ev=photon_energy,
            polarization=jnp.asarray(
                [1.0, 0.2j * (index + 1), 0.0], dtype=jnp.complex128
            ),
            sample_azimuth=0.11 + 0.03 * index,
            work_function_ev=4.5,
            temperature_k=24.0 + 7.0 * index,
            mean_free_path_ang=8.0,
        )
        for index, photon_energy in enumerate(photon_energies)
    )
    geometry_batch: ExperimentGeometry = jax.tree_util.tree_map(
        lambda *leaves: jnp.stack(leaves), *geometries
    )
    return geometry_batch


def _compile_and_vmap_record() -> Dict[str, Any]:
    """PRIVATE: Measure fixed-shape compile reuse and geometry vmapability.

    Returns
    -------
    record : Dict[str, Any]
        Trace/cache counts, sweep values, vmap comparison, and verdict.
    """
    fixture: Dict[str, Any] = _small_fixture()
    trace_count: list[int] = [0]

    def batched_driver(
        geometry_batch: ExperimentGeometry,
        calibration: DetectorCalibration,
    ) -> Float64[Array, "batch 1 u v e"]:
        """Evaluate the same driver over complete geometry leaves."""
        trace_count[0] += 1
        values: Float64[Array, "batch 1 u v e"] = jax.vmap(
            lambda geometry: _driver_counts(
                fixture,
                fixture["hamiltonians"],
                geometry,
                calibration,
                checkpoint=True,
            )
        )(geometry_batch)
        return values

    compiled: Any = jax.jit(batched_driver)
    photon_sweeps: Tuple[Float64[Array, " two"], ...] = (
        jnp.asarray([49.8, 50.2]),
        jnp.asarray([50.0, 50.4]),
        jnp.asarray([49.7, 50.3]),
    )
    width_sweeps: Tuple[Tuple[float, float, float], ...] = (
        (0.010, 0.011, 0.020),
        (0.013, 0.009, 0.026),
        (0.008, 0.014, 0.018),
    )
    cache_sizes: list[int] = [int(compiled._cache_size())]  # noqa: SLF001
    outputs: list[Float64[Array, "2 1 2 2 2"]] = []
    geometry_batches: list[ExperimentGeometry] = []
    calibrations: list[DetectorCalibration] = []
    photons: Float64[Array, " two"]
    widths: Tuple[float, float, float]
    for photons, widths in zip(photon_sweeps, width_sweeps, strict=True):
        geometry_batch: ExperimentGeometry = _geometry_batch(photons)
        calibration: DetectorCalibration = eqx.tree_at(
            lambda item: (
                item.psf_fwhm_u,
                item.psf_fwhm_v,
                item.psf_fwhm_energy_ev,
            ),
            fixture["calibration"],
            tuple(jnp.asarray(width) for width in widths),
        )
        output: Float64[Array, "2 1 2 2 2"] = compiled(
            geometry_batch, calibration
        )
        jax.block_until_ready(output)
        outputs.append(output)
        geometry_batches.append(geometry_batch)
        calibrations.append(calibration)
        cache_sizes.append(int(compiled._cache_size()))  # noqa: SLF001

    first_batch: ExperimentGeometry = geometry_batches[0]
    first_calibration: DetectorCalibration = calibrations[0]
    direct_rows: list[Float64[Array, "1 2 2 2"]] = []
    batch_size: int = photon_sweeps[0].shape[0]
    for index in range(batch_size):
        geometry: ExperimentGeometry = jax.tree_util.tree_map(
            lambda leaf: leaf[index], first_batch
        )
        direct_rows.append(
            _driver_counts(
                fixture,
                fixture["hamiltonians"],
                geometry,
                first_calibration,
                checkpoint=True,
            )
        )
    direct: Float64[Array, "2 1 2 2 2"] = jnp.stack(direct_rows)
    vmap_error: float = float(jnp.max(jnp.abs(outputs[0] - direct)))
    vmap_passes: bool = bool(
        jnp.allclose(outputs[0], direct, rtol=1.0e-12, atol=1.0e-13)
    )
    one_trace: bool = trace_count[0] == 1 and cache_sizes == [0, 1, 1, 1]
    return {
        "program": "jit_vmap_simulate_arpes_over_experiment_geometry",
        "mapping_chart": "general rotation with strict target enclosure",
        "batch_length": batch_size,
        "photon_energy_sweeps_ev": [
            np.asarray(values).tolist() for values in photon_sweeps
        ],
        "native_fwhm_sweeps": [list(values) for values in width_sweeps],
        "fixed_output_shape": list(outputs[0].shape),
        "trace_count": trace_count[0],
        "compile_cache_sizes": cache_sizes,
        "one_compilation": one_trace,
        "maximum_vmap_absolute_error": vmap_error,
        "vmap_matches_direct_rtol_1e_12": vmap_passes,
        "result": "pass" if one_trace and vmap_passes else "fail",
    }


def _literal_fixture() -> Dict[str, Any]:
    """PRIVATE: Build the exact 256x256x400, 20-band target.

    Returns
    -------
    fixture : Dict[str, Any]
        Valid full-shape canonical-driver inputs.

    Notes
    -----
    The 20-band metadata and explicit 20-orbital Hamiltonian share one basis.
    The fixture is used for ahead-of-time compilation and allocation analysis;
    the 26,214,400-bin detector program is deliberately not executed on CPU.
    """
    n_kx: int
    n_ky: int
    n_energy: int
    n_band: int
    k_chunk: int
    energy_chunk: int
    n_kx, n_ky, n_energy, n_band, k_chunk, energy_chunk = _literal_dimensions()
    del energy_chunk, k_chunk
    n_k: int = n_kx * n_ky
    crystal: Any = make_crystal_geometry(
        2.0 * jnp.pi * jnp.eye(3, dtype=jnp.float64),
        jnp.zeros((1, 3), dtype=jnp.float64),
        ("X",),
    )
    repeated_zero: Tuple[int, ...] = (0,) * n_band
    basis: Any = make_orbital_basis(
        atom_indices=repeated_zero,
        n=(1,) * n_band,
        l=repeated_zero,
        m=repeated_zero,
        labels=tuple(f"orbital_{index}" for index in range(n_band)),
    )
    kx: Float64[Array, " kx"] = jnp.linspace(-0.11, 0.11, n_kx)
    ky: Float64[Array, " ky"] = jnp.linspace(-0.105, 0.105, n_ky)
    mesh_x: Float64[Array, "ky kx"]
    mesh_y: Float64[Array, "ky kx"]
    mesh_x, mesh_y = jnp.meshgrid(kx, ky, indexing="xy")
    kpoints: Float64[Array, "n_k 3"] = jnp.stack(
        (mesh_x, mesh_y, jnp.zeros_like(mesh_x)), axis=-1
    ).reshape((n_k, 3))
    offsets: Float64[Array, " band"] = jnp.linspace(-0.9, 0.7, n_band)
    dispersion: Float64[Array, " n_k"] = 0.8 * jnp.sum(
        kpoints[:, :2] ** 2, axis=-1
    )
    eigenvalues: Float64[Array, "n_k band"] = (
        dispersion[:, None] + offsets[None, :]
    )
    identity: Complex128[Array, "band band"] = jnp.eye(
        n_band, dtype=jnp.complex128
    )
    eigenvectors: Complex128[Array, "n_k band band"] = jnp.broadcast_to(
        identity, (n_k, n_band, n_band)
    )
    bands: DiagonalizedBands = make_diagonalized_bands(
        eigenvalues,
        eigenvectors,
        kpoints,
        crystal,
        basis,
        fermi_energy=0.0,
        depths=jnp.linspace(0.0, 2.0, n_band),
    )
    hamiltonians: Complex128[Array, "n_k band band"] = (
        eigenvalues[:, :, None] * identity[None, :, :]
    )
    shell_indices: Tuple[int, ...] = repeated_zero
    radial: RadialSpec = make_radial_spec(
        basis,
        shell_indices,
        mode="fixed",
        fixed_integrals_shell=jnp.asarray([[0.0, 1.0]]),
    )
    energy_axis: Float64[Array, " energy"] = jnp.linspace(
        -0.55, 0.25, n_energy
    )
    energy_step: Float64[Array, ""] = energy_axis[1] - energy_axis[0]
    energy_edges: Float64[Array, " energy_edges"] = jnp.concatenate(
        (
            energy_axis[:1] - 0.5 * energy_step,
            0.5 * (energy_axis[:-1] + energy_axis[1:]),
            energy_axis[-1:] + 0.5 * energy_step,
        )
    )
    geometry: ExperimentGeometry = make_experiment_geometry(
        photon_energy_ev=50.0,
        polarization=jnp.asarray([1.0, 0.35j, 0.0]),
        sample_azimuth=0.0,
        work_function_ev=4.5,
        temperature_k=25.0,
        mean_free_path_ang=8.0,
    )
    calibration: DetectorCalibration = make_detector_calibration(
        u_bin_edges=jnp.linspace(-0.04, 0.04, n_kx + 1),
        v_bin_edges=jnp.linspace(-0.04, 0.04, n_ky + 1),
        energy_bin_edges_ev=energy_edges,
        psf_fwhm_u=0.0012,
        psf_fwhm_v=0.0012,
        psf_fwhm_energy_ev=0.012,
        transmission_reference_domain_ev=jnp.asarray([44.0, 46.0]),
    )
    effects: DetectorEffects = make_detector_effects(
        domain_logits=jnp.asarray([0.0]),
        domain_euler_angles_rad=jnp.zeros((1, 3)),
        transmission_raw_slopes=jnp.asarray([-0.35, 0.18]),
        background_coefficients=jnp.asarray([-3.0]),
        sensitivity_coefficients=jnp.asarray([]),
        exposure=1.0,
        background_mode="flat",
        sensitivity_mode="constant",
        domain_frame_ids=("org.diffpes.frame.sample_cartesian",),
    )
    return {
        "hamiltonians": hamiltonians,
        "bands": bands,
        "radial": radial,
        "matrix_params": make_matrix_element_params(
            basis,
            shell_indices,
            sigma_shell=jnp.asarray([1.0]),
            phase_shift_angles_shell=jnp.asarray([0.2]),
        ),
        "quadrature": make_radial_quadrature_spec(),
        "final_state": make_final_state_spec(),
        "geometry": geometry,
        "self_energy": make_self_energy_model(gamma=0.045),
        "kgrid": make_kgrid(kpoints, mesh_shape=(n_ky, n_kx), kz=0.0),
        "energy_axis": energy_axis,
        "calibration": calibration,
        "detector_effects": effects,
    }


def _iter_nested_jaxprs(value: Any) -> Iterable[Any]:
    """PRIVATE: Yield every JAXPR recursively stored in one value.

    Parameters
    ----------
    value : Any
        JAXPR, closed JAXPR, mapping, or nested parameter container.

    Yields
    ------
    jaxpr : Any
        Every object exposing JAXPR ``eqns``, ``invars``, and ``outvars``.
    """
    if hasattr(value, "jaxpr"):
        yield from _iter_nested_jaxprs(value.jaxpr)
    elif hasattr(value, "eqns") and hasattr(value, "invars"):
        yield value
        for equation in value.eqns:
            yield from _iter_nested_jaxprs(equation.params)
    elif isinstance(value, dict):
        for nested in value.values():
            yield from _iter_nested_jaxprs(nested)
    elif isinstance(value, (tuple, list)):
        for nested in value:
            yield from _iter_nested_jaxprs(nested)


def _jaxpr_shape_record(closed_jaxpr: Any) -> Dict[str, Any]:
    """PRIVATE: Audit recursive JAXPR shapes for forbidden full carriers.

    Parameters
    ----------
    closed_jaxpr : Any
        Result of ``jax.make_jaxpr`` on the literal forward driver.

    Returns
    -------
    record : Dict[str, Any]
        Equation count, forbidden shapes, matches, and verdict.
    """
    n_kx: int
    n_ky: int
    n_energy: int
    n_band: int
    k_chunk: int
    energy_chunk: int
    n_kx, n_ky, n_energy, n_band, k_chunk, energy_chunk = _literal_dimensions()
    del energy_chunk, k_chunk
    n_k: int = n_kx * n_ky
    forbidden_kbe: set[Tuple[int, ...]] = set(
        itertools.permutations((n_k, n_band, n_energy))
    )
    forbidden_kbe.update(
        itertools.permutations(
            (
                n_kx,
                n_ky,
                n_energy,
                n_band,
            )
        )
    )
    forbidden_kinematics: set[Tuple[int, ...]] = set(
        itertools.permutations((n_k, n_energy, 3))
    )
    forbidden_kinematics.update(
        itertools.permutations((n_kx, n_ky, n_energy, 3))
    )
    full_kbe_element_count: int = n_k * n_band * n_energy
    full_kinematics_element_count: int = n_k * n_energy * 3
    kbe_matches: set[Tuple[int, ...]] = set()
    kinematics_matches: set[Tuple[int, ...]] = set()
    shapes: set[Tuple[int, ...]] = set()
    equation_count: int = 0
    for jaxpr in _iter_nested_jaxprs(closed_jaxpr):
        equation_count += len(jaxpr.eqns)
        variables: Tuple[Any, ...] = (
            tuple(jaxpr.invars)
            + tuple(jaxpr.constvars)
            + tuple(jaxpr.outvars)
            + tuple(
                variable
                for equation in jaxpr.eqns
                for variable in equation.outvars
            )
        )
        for variable in variables:
            aval: Any = getattr(variable, "aval", None)
            shape: Tuple[int, ...] | None = getattr(aval, "shape", None)
            if shape is not None:
                normalized: Tuple[int, ...] = tuple(
                    int(size) for size in shape
                )
                shapes.add(normalized)
                element_count: int = int(np.prod(normalized, dtype=np.int64))
                if (
                    normalized in forbidden_kbe
                    or element_count == full_kbe_element_count
                ):
                    kbe_matches.add(normalized)
                if (
                    normalized in forbidden_kinematics
                    or element_count == full_kinematics_element_count
                ):
                    kinematics_matches.add(normalized)
    matches: set[Tuple[int, ...]] = kbe_matches | kinematics_matches
    return {
        "recursive_equation_count": equation_count,
        "distinct_array_shape_count": len(shapes),
        "forbidden_full_kbe_shapes": [
            list(shape) for shape in sorted(forbidden_kbe)
        ],
        "forbidden_full_kinematics_shapes": [
            list(shape) for shape in sorted(forbidden_kinematics)
        ],
        "full_kbe_element_count": full_kbe_element_count,
        "full_kinematics_element_count": full_kinematics_element_count,
        "full_kbe_shape_matches": [
            list(shape) for shape in sorted(kbe_matches)
        ],
        "full_kinematics_shape_matches": [
            list(shape) for shape in sorted(kinematics_matches)
        ],
        "forbidden_shape_matches": [list(shape) for shape in sorted(matches)],
        "contains_full_kbe_materialization": bool(kbe_matches),
        "contains_full_kinematics_materialization": bool(kinematics_matches),
        "compact_kinematics_invariant": (
            "no canonical, flattened, or factored shape with K*E*3 elements"
        ),
        "result": "pass" if not matches else "fail",
    }


def _literal_record() -> Dict[str, Any]:
    """PRIVATE: Compile and measure the exact S1 forward and gradient target.

    Returns
    -------
    record : Dict[str, Any]
        Literal shapes, recursive JAXPR audit, allocation records, and verdict.

    Notes
    -----
    Both programs call the public canonical driver.  The gradient program
    differentiates a scalar sum with respect to the complete explicit
    Hamiltonian raster, so its output includes the full complex128 H gradient.
    """
    n_kx: int
    n_ky: int
    n_energy: int
    n_band: int
    k_chunk: int
    energy_chunk: int
    n_kx, n_ky, n_energy, n_band, k_chunk, energy_chunk = _literal_dimensions()
    forward_budget: int
    gradient_budget: int
    forward_budget, gradient_budget = _memory_budgets()
    fixture: Dict[str, Any] = _literal_fixture()
    hamiltonians: Complex128[Array, "n_k band band"] = fixture["hamiltonians"]

    def forward(
        candidate: Complex128[Array, "n_k band band"],
    ) -> Float64[Array, "1 kx ky energy"]:
        """Return the literal expected-count cube."""
        values: Float64[Array, "1 kx ky energy"] = _driver_counts(
            fixture,
            candidate,
            fixture["geometry"],
            fixture["calibration"],
            checkpoint=True,
            k_chunk=k_chunk,
            energy_chunk=energy_chunk,
        )
        return values

    def loss(
        candidate: Complex128[Array, "n_k band band"],
    ) -> Float64[Array, ""]:
        """Reduce the literal expected-count cube to one scalar."""
        scalar: Float64[Array, ""] = jnp.sum(forward(candidate))
        return scalar

    rss_before: int = _maximum_rss_bytes()
    jaxpr_start: float = time.perf_counter()
    closed_jaxpr: Any = jax.make_jaxpr(forward)(hamiltonians)
    jaxpr_seconds: float = time.perf_counter() - jaxpr_start
    shape_record: Dict[str, Any] = _jaxpr_shape_record(closed_jaxpr)

    forward_start: float = time.perf_counter()
    forward_compiled: Any = jax.jit(forward).lower(hamiltonians).compile()
    forward_seconds: float = time.perf_counter() - forward_start
    forward_memory: Dict[str, int | bool | str] = _memory_record(
        forward_compiled
    )

    gradient_start: float = time.perf_counter()
    gradient_compiled: Any = (
        jax.jit(jax.value_and_grad(loss)).lower(hamiltonians).compile()
    )
    gradient_seconds: float = time.perf_counter() - gradient_start
    gradient_memory: Dict[str, int | bool | str] = _memory_record(
        gradient_compiled
    )
    rss_after: int = _maximum_rss_bytes()
    forward_live: int = int(
        forward_memory.get("compiler_live_allocation_bytes", 0)
    )
    gradient_live: int = int(
        gradient_memory.get("compiler_live_allocation_bytes", 0)
    )
    forward_passes: bool = (
        bool(forward_memory.get("authority_available"))
        and forward_live <= forward_budget
    )
    gradient_passes: bool = (
        bool(gradient_memory.get("authority_available"))
        and gradient_live <= gradient_budget
    )
    no_kbe: bool = shape_record["result"] == "pass"
    output_bytes: int = n_kx * n_ky * n_energy * 8
    return {
        "n_kx": n_kx,
        "n_ky": n_ky,
        "n_k": n_kx * n_ky,
        "n_energy": n_energy,
        "n_band": n_band,
        "n_orbital": n_band,
        "k_chunk": k_chunk,
        "energy_chunk": energy_chunk,
        "checkpoint": True,
        "expected_count_cube_shape": [
            1,
            n_kx,
            n_ky,
            n_energy,
        ],
        "expected_count_cube_bytes": output_bytes,
        "expected_count_cube_decimal_mb": output_bytes / 1_000_000.0,
        "programs_executed": False,
        "mapping_chart": "signed-diagonal boundary-aware cubature",
        "measurement_authority": (
            "XLA executable memory_analysis peak-live allocation"
        ),
        "jax_profiler_execution_trace": (
            "not produced: literal CPU programs are compile-only; compiler "
            "allocation is the registered device-program authority"
        ),
        "jaxpr_inspection_seconds": jaxpr_seconds,
        "jaxpr_shape_audit": shape_record,
        "forward": {
            "program": "simulate_arpes_expected_counts",
            "compilation_seconds": forward_seconds,
            "budget_bytes": forward_budget,
            "memory_analysis": forward_memory,
            "passes_budget": forward_passes,
        },
        "forward_and_gradient": {
            "program": "value_and_complete_hamiltonian_gradient",
            "compilation_seconds": gradient_seconds,
            "budget_bytes": gradient_budget,
            "memory_analysis": gradient_memory,
            "passes_budget": gradient_passes,
        },
        "process_peak_rss_before_bytes_non_authoritative": rss_before,
        "process_peak_rss_after_bytes_non_authoritative": rss_after,
        "result": (
            "pass" if no_kbe and forward_passes and gradient_passes else "fail"
        ),
    }


def main() -> None:
    """Measure S1--S4 and write the authenticated JSON record."""
    repository_root: Path = _repository_root()
    output_path: Path = _output_path()
    literal: Dict[str, Any] = _literal_record()
    remat: Dict[str, Any] = _remat_record()
    compile_and_vmap: Dict[str, Any] = _compile_and_vmap_record()
    source_paths: Tuple[str, ...] = (
        "tests/_reference_tools/measure_detector_scaling.py",
        "src/diffpes/simul/__init__.py",
        "src/diffpes/simul/spectrum.py",
        "src/diffpes/simul/spectral.py",
        "src/diffpes/simul/matrixel.py",
        "src/diffpes/simul/effects.py",
        "src/diffpes/simul/_detector_map.py",
        "src/diffpes/simul/broadening.py",
        "src/diffpes/simul/kinematics.py",
        "src/diffpes/simul/polarization.py",
        "src/diffpes/maths/__init__.py",
        "src/diffpes/maths/dipole.py",
        "src/diffpes/maths/gaunt.py",
        "src/diffpes/maths/rotations.py",
        "src/diffpes/maths/safe.py",
        "src/diffpes/maths/spherical_harmonics.py",
        "src/diffpes/radial/__init__.py",
        "src/diffpes/radial/bessel.py",
        "src/diffpes/radial/coulomb.py",
        "src/diffpes/radial/integrate.py",
        "src/diffpes/radial/screening.py",
        "src/diffpes/radial/wavefunctions.py",
        "src/diffpes/types/__init__.py",
        "src/diffpes/types/aliases.py",
        "src/diffpes/types/constants.py",
        "src/diffpes/types/experiment.py",
        "src/diffpes/types/bands.py",
        "src/diffpes/types/detector_effects.py",
        "src/diffpes/types/geometry.py",
        "src/diffpes/types/kpath.py",
        "src/diffpes/types/radial_params.py",
        "src/diffpes/types/self_energy.py",
        "src/diffpes/types/tb_model.py",
        "src/diffpes/utils/__init__.py",
        "src/diffpes/utils/math.py",
        "pyproject.toml",
        "uv.lock",
    )
    record: Dict[str, Any] = {
        "schema": "diffpes.detector-scalability.v1",
        "gate_ids": ["08a.S1", "08a.S2", "08a.S3", "08a.S4"],
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
            relative: _sha256(repository_root / relative)
            for relative in source_paths
        },
        "s1_literal_target": literal,
        "s2_rematerialization": remat,
        "s3_compile_count_s4_vmap": compile_and_vmap,
        "result": (
            "pass"
            if literal["result"] == "pass"
            and remat["result"] == "pass"
            and compile_and_vmap["result"] == "pass"
            else "fail"
        ),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    json.dump(record, sys.stdout, indent=2, sort_keys=True)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
