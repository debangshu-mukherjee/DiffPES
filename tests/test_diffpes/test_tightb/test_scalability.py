r"""Validate tight-binding differentiation and scalability properties.

Extended Summary
----------------
This module supplies bounded CI evidence for holomorphic-phase-gradient, hamiltonian-compile-bounded, diagonalization-shapes-bounded, and eigvalsh-reverse-mode-bounded. It
checks hopping-count-independent Bloch JAXPRs and one trace per static shape.
It also runs bounded diagonalization and reverse-mode cases. Shape analysis
derives the production memory floor without large allocations.

Notes
-----
The production Hamiltonian scaling output has shape ``(10000, 64, 64)`` and
therefore contains 625 MiB of complex128 data. Tests use
:func:`jax.eval_shape` for that case and execute smaller arrays in CI to avoid
an intentional out-of-memory hazard. The reverse-mode check compares compiler temporary memory
across two batch sizes. This comparison detects superlinear reverse-tape
growth without fragile timing limits.
"""

import math

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest
from beartype.typing import Any, Callable
from jax.extend.core import ClosedJaxpr, Jaxpr, Literal
from jaxtyping import Array, Complex128, Float64

from diffpes.tightb.diagonalize import diagonalize_tb, eigvalsh_bands
from diffpes.tightb.hamiltonian import (
    bloch_hamiltonian,
    bloch_hamiltonian_batch,
)
from diffpes.types import (
    CrystalGeometry,
    OrbitalBasis,
    TBModel,
    make_crystal_geometry,
    make_orbital_basis,
    make_tb_model,
)
from tests._gradients import complex_step_derivative

_HOPPING_SWEEP: tuple[int, ...] = (10, 100, 1_000, 10_000)
_PRODUCTION_BATCH_N_K: int = 10_000
_PRODUCTION_BATCH_N_SO: int = 64
_PRODUCTION_DIAGONALIZE_N_K: int = 4_096
_PRODUCTION_DIAGONALIZE_N_SO: int = 32
_COMPLEX128_BYTES: int = 16
_FLOAT64_BYTES: int = 8
_MIB: int = 2**20


def _one_atom_basis(n_orbitals: int) -> tuple[CrystalGeometry, OrbitalBasis]:
    """Build static one-atom metadata with ``n_orbitals`` scalar channels."""
    geometry: CrystalGeometry = make_crystal_geometry(
        lattice=jnp.eye(3, dtype=jnp.float64),
        positions=jnp.zeros((1, 3), dtype=jnp.float64),
        species=("X",),
    )
    basis: OrbitalBasis = make_orbital_basis(
        atom_indices=(0,) * n_orbitals,
        n=(1,) * n_orbitals,
        l=(0,) * n_orbitals,
        m=(0,) * n_orbitals,
        labels=tuple(f"s_{index}" for index in range(n_orbitals)),
    )
    return geometry, basis


def _make_hopping_count_model(n_hoppings: int) -> TBModel:
    """Build a one-orbital model with an exact closed hopping count."""
    if n_hoppings <= 0 or n_hoppings % 2:
        message: str = "n_hoppings must be a positive even integer"
        raise ValueError(message)
    geometry: CrystalGeometry
    basis: OrbitalBasis
    geometry, basis = _one_atom_basis(1)
    half_count: int = n_hoppings // 2
    forward_cells: tuple[tuple[int, int, int], ...] = tuple(
        (cell, 0, 0) for cell in range(1, half_count + 1)
    )
    reverse_cells: tuple[tuple[int, int, int], ...] = tuple(
        (-cell, 0, 0) for cell in range(1, half_count + 1)
    )
    forward: Float64[Array, " n_half"] = -1.0 / jnp.arange(
        2,
        half_count + 2,
        dtype=jnp.float64,
    )
    amplitudes: Complex128[Array, " n_hop"] = jnp.concatenate(
        (forward, forward)
    ).astype(jnp.complex128)
    model: TBModel = make_tb_model(
        hopping_amplitudes=amplitudes,
        onsite_energies=jnp.zeros((1,), dtype=jnp.float64),
        soc_lambdas=jnp.zeros((0,), dtype=jnp.float64),
        geometry=geometry,
        basis=basis,
        hopping_pairs=((0, 0),) * n_hoppings,
        hopping_cells=forward_cells + reverse_cells,
        shell_index=(-1,),
    )
    return model


def _make_dispersive_diagonal_model(n_orbitals: int) -> TBModel:
    """Build a closed diagonal model that still consumes every k-point."""
    geometry: CrystalGeometry
    basis: OrbitalBasis
    geometry, basis = _one_atom_basis(n_orbitals)
    pairs: tuple[tuple[int, int], ...] = tuple(
        (orbital, orbital) for orbital in range(n_orbitals)
    )
    cells: tuple[tuple[int, int, int], ...] = ((1, 0, 0),) * n_orbitals + (
        (-1, 0, 0),
    ) * n_orbitals
    forward: Float64[Array, " n_orb"] = jnp.linspace(
        -0.2,
        -0.5,
        n_orbitals,
        dtype=jnp.float64,
    )
    amplitudes: Complex128[Array, " n_hop"] = jnp.concatenate(
        (forward, forward)
    ).astype(jnp.complex128)
    model: TBModel = make_tb_model(
        hopping_amplitudes=amplitudes,
        onsite_energies=jnp.linspace(
            -1.0,
            1.0,
            n_orbitals,
            dtype=jnp.float64,
        ),
        soc_lambdas=jnp.zeros((0,), dtype=jnp.float64),
        geometry=geometry,
        basis=basis,
        hopping_pairs=pairs + pairs,
        hopping_cells=cells,
        shell_index=(-1,) * n_orbitals,
    )
    return model


def _kpoints(n_kpoints: int) -> Float64[Array, "n_k 3"]:
    """Build a deterministic generic one-dimensional k-point batch."""
    k_x: Float64[Array, " n_k"] = jnp.linspace(
        -0.47,
        0.43,
        n_kpoints,
        dtype=jnp.float64,
    )
    points: Float64[Array, "n_k 3"] = jnp.stack(
        (
            k_x,
            0.17 * k_x + 0.03,
            -0.11 * k_x + 0.02,
        ),
        axis=-1,
    )
    return points


def _count_jaxpr_equations(value: object) -> int:
    """Return the recursive equation count across nested JAXPR parameters."""
    if isinstance(value, ClosedJaxpr):
        return _count_jaxpr_equations(value.jaxpr)
    if isinstance(value, Jaxpr):
        nested: int = sum(
            _count_jaxpr_equations(parameter)
            for equation in value.eqns
            for parameter in equation.params.values()
        )
        return len(value.eqns) + nested
    if isinstance(value, (tuple, list)):
        return sum(_count_jaxpr_equations(item) for item in value)
    if isinstance(value, dict):
        return sum(_count_jaxpr_equations(item) for item in value.values())
    return 0


def _collect_jaxpr_shapes(
    value: object,
    shapes: list[tuple[int, ...]],
) -> None:
    """Collect every shaped array variable from nested JAXPRs."""
    if isinstance(value, ClosedJaxpr):
        _collect_jaxpr_shapes(value.jaxpr, shapes)
        return
    if isinstance(value, Jaxpr):
        variables: list[object] = [
            *value.constvars,
            *value.invars,
            *value.outvars,
        ]
        equation: Any
        parameter: object
        for equation in value.eqns:
            variables.extend(equation.invars)
            variables.extend(equation.outvars)
            for parameter in equation.params.values():
                _collect_jaxpr_shapes(parameter, shapes)
        variable: object
        for variable in variables:
            if isinstance(variable, Literal):
                continue
            shape: object = getattr(
                getattr(variable, "aval", None), "shape", None
            )
            if shape is not None:
                shapes.append(tuple(int(axis) for axis in shape))
        return
    if isinstance(value, (tuple, list)):
        item: object
        for item in value:
            _collect_jaxpr_shapes(item, shapes)
        return
    if isinstance(value, dict):
        item: object
        for item in value.values():
            _collect_jaxpr_shapes(item, shapes)


def _memory_analysis(executable: Any) -> Any:
    """Return compiler memory statistics, requiring backend support."""
    statistics: Any = executable.memory_analysis()
    if statistics is None:
        message: str = (
            "the active JAX backend did not report memory statistics"
        )
        raise AssertionError(message)
    return statistics


@pytest.fixture(scope="module")
def hopping_sweep_models() -> tuple[TBModel, ...]:
    """Build the four exact hopping-shape cases once per module."""
    models: tuple[TBModel, ...] = tuple(
        _make_hopping_count_model(n_hoppings) for n_hoppings in _HOPPING_SWEEP
    )
    return models


class TestBlochAssemblyScalability:
    """Validate the no-unroll and compile-count requirements of hamiltonian-compile-bounded."""

    def test_jaxpr_op_count_is_constant_across_hopping_sweep(
        self,
        hopping_sweep_models: tuple[TBModel, ...],
    ) -> None:
        """Keep the recursive JAXPR equation count exactly shape-independent.

        The case sweeps four closed hopping-list sizes.

        Notes
        -----
        Trace the production ``bloch_hamiltonian`` at 10, 100, 1,000, and
        10,000 closed hopping records. Counting nested JAXPRs catches hidden
        unrolling behind a call primitive as well as top-level Python loops.
        """
        kpoint: Float64[Array, " 3"] = jnp.asarray(
            [0.17, -0.08, 0.03],
            dtype=jnp.float64,
        )
        counts: list[int] = []
        model: TBModel
        for model in hopping_sweep_models:
            jaxpr: ClosedJaxpr = jax.make_jaxpr(bloch_hamiltonian)(
                model,
                kpoint,
            )
            counts.append(_count_jaxpr_equations(jaxpr))

        assert counts[0] > 0
        assert len(set(counts)) == 1, dict(
            zip(_HOPPING_SWEEP, counts, strict=True)
        )

    def test_one_trace_per_static_hopping_shape(
        self,
        hopping_sweep_models: tuple[TBModel, ...],
    ) -> None:
        """Compile once for each shape and reuse it for dynamic leaf changes.

        The case changes numerical leaves without changing static topology.

        Notes
        -----
        Count Python traces after each original and modified model call.
        """
        trace_count: list[int] = [0]

        def counted(
            model: TBModel,
            kpoint: Float64[Array, " 3"],
        ) -> Complex128[Array, "1 1"]:
            trace_count[0] += 1
            hamiltonian: Complex128[Array, "1 1"] = bloch_hamiltonian(
                model,
                kpoint,
            )
            return hamiltonian

        compiled: Callable[..., Any] = eqx.filter_jit(counted)
        index: int
        model: TBModel
        for index, model in enumerate(hopping_sweep_models):
            kpoint: Float64[Array, " 3"] = jnp.asarray(
                [0.01 * (index + 1), 0.0, 0.0],
                dtype=jnp.float64,
            )
            compiled(model, kpoint).block_until_ready()
            changed: TBModel = eqx.tree_at(
                lambda item: item.hopping_amplitudes,
                model,
                0.9 * model.hopping_amplitudes,
            )
            compiled(changed, 1.7 * kpoint).block_until_ready()
            assert trace_count[0] == index + 1

        assert trace_count[0] == len(_HOPPING_SWEEP)


class TestBatchScalability:
    """Validate static shapes and bounded execution for diagonalization-shapes-bounded."""

    def test_production_shapes_and_memory_are_derived_without_allocation(
        self,
    ) -> None:
        """Verify the production output shapes and dominant byte bounds.

        The case covers Hamiltonians, eigenvalues, and full eigensystems.

        Notes
        -----
        ``jax.eval_shape`` traces the exact production dimensions but creates
        no 625 MiB Hamiltonian buffer. The persistent batch outputs total
        less than 630 MiB: 625 MiB of complex Hamiltonians plus 4.883 MiB of
        float64 eigenvalues. The full-diagonalization production carrier is
        65 MiB for eigenvalues and eigenvectors.
        """
        batch_model: TBModel = _make_dispersive_diagonal_model(
            _PRODUCTION_BATCH_N_SO
        )
        production_kpoints: jax.ShapeDtypeStruct = jax.ShapeDtypeStruct(
            (_PRODUCTION_BATCH_N_K, 3),
            jnp.float64,
        )
        hamiltonian_shape: jax.ShapeDtypeStruct = jax.eval_shape(
            lambda points: bloch_hamiltonian_batch(batch_model, points),
            production_kpoints,
        )
        eigenvalue_shape: jax.ShapeDtypeStruct = jax.eval_shape(
            lambda points: eigvalsh_bands(batch_model, points),
            production_kpoints,
        )

        assert hamiltonian_shape.shape == (
            _PRODUCTION_BATCH_N_K,
            _PRODUCTION_BATCH_N_SO,
            _PRODUCTION_BATCH_N_SO,
        )
        assert hamiltonian_shape.dtype == jnp.complex128
        assert eigenvalue_shape.shape == (
            _PRODUCTION_BATCH_N_K,
            _PRODUCTION_BATCH_N_SO,
        )
        assert eigenvalue_shape.dtype == jnp.float64

        hamiltonian_bytes: int = (
            _PRODUCTION_BATCH_N_K
            * _PRODUCTION_BATCH_N_SO**2
            * _COMPLEX128_BYTES
        )
        eigenvalue_bytes: int = (
            _PRODUCTION_BATCH_N_K * _PRODUCTION_BATCH_N_SO * _FLOAT64_BYTES
        )
        assert hamiltonian_bytes == 625 * _MIB
        assert eigenvalue_bytes * (2 * _PRODUCTION_BATCH_N_SO) == (
            hamiltonian_bytes
        )
        assert hamiltonian_bytes + eigenvalue_bytes < 630 * _MIB

        diagonalize_model: TBModel = _make_dispersive_diagonal_model(
            _PRODUCTION_DIAGONALIZE_N_SO
        )
        diagonalize_shape: Any = jax.eval_shape(
            lambda points: diagonalize_tb(diagonalize_model, points),
            jax.ShapeDtypeStruct(
                (_PRODUCTION_DIAGONALIZE_N_K, 3),
                jnp.float64,
            ),
        )
        assert diagonalize_shape.eigenvalues.shape == (
            _PRODUCTION_DIAGONALIZE_N_K,
            _PRODUCTION_DIAGONALIZE_N_SO,
        )
        assert diagonalize_shape.eigenvectors.shape == (
            _PRODUCTION_DIAGONALIZE_N_K,
            _PRODUCTION_DIAGONALIZE_N_SO,
            _PRODUCTION_DIAGONALIZE_N_SO,
        )
        diagonalize_bytes: int = (
            _PRODUCTION_DIAGONALIZE_N_K
            * _PRODUCTION_DIAGONALIZE_N_SO
            * _FLOAT64_BYTES
            + _PRODUCTION_DIAGONALIZE_N_K
            * _PRODUCTION_DIAGONALIZE_N_SO**2
            * _COMPLEX128_BYTES
        )
        assert diagonalize_bytes == 65 * _MIB

    def test_ci_sized_batch_and_full_diagonalization_execute(self) -> None:
        """Execute shape-faithful proxies and record compiled output bytes.

        The case runs batched assembly and both eigensystem paths.

        Notes
        -----
        Compare compiled output sizes and diagonal-model analytic eigenvalues.
        """
        n_kpoints: int = 32
        n_orbitals: int = 64
        model: TBModel = _make_dispersive_diagonal_model(n_orbitals)
        kpoints: Float64[Array, "n_k 3"] = _kpoints(n_kpoints)

        hamiltonian_executable: Any = (
            jax.jit(lambda points: bloch_hamiltonian_batch(model, points))
            .lower(kpoints)
            .compile()
        )
        hamiltonians: Complex128[Array, "n_k n n"] = hamiltonian_executable(
            kpoints
        )
        hamiltonians.block_until_ready()
        hamiltonian_statistics: Any = _memory_analysis(hamiltonian_executable)
        expected_hamiltonian_bytes: int = (
            n_kpoints * n_orbitals**2 * _COMPLEX128_BYTES
        )
        assert hamiltonians.shape == (n_kpoints, n_orbitals, n_orbitals)
        assert hamiltonians.dtype == jnp.complex128
        assert hamiltonians.nbytes == expected_hamiltonian_bytes
        assert (
            hamiltonian_statistics.output_size_in_bytes
            == expected_hamiltonian_bytes
        )

        eigenvalue_executable: Any = (
            jax.jit(lambda points: eigvalsh_bands(model, points))
            .lower(kpoints)
            .compile()
        )
        eigenvalues: Float64[Array, "n_k n"] = eigenvalue_executable(kpoints)
        eigenvalues.block_until_ready()
        eigenvalue_statistics: Any = _memory_analysis(eigenvalue_executable)
        expected_eigenvalue_bytes: int = (
            n_kpoints * n_orbitals * _FLOAT64_BYTES
        )
        forward: Float64[Array, " n"] = jnp.real(
            model.hopping_amplitudes[:n_orbitals]
        )
        analytic: Float64[Array, "n_k n"] = (
            model.onsite_energies[None, :]
            + 2.0 * jnp.cos(2.0 * jnp.pi * kpoints[:, :1]) * forward[None, :]
        )
        assert eigenvalues.shape == (n_kpoints, n_orbitals)
        assert eigenvalues.dtype == jnp.float64
        assert eigenvalues.nbytes == expected_eigenvalue_bytes
        assert (
            eigenvalue_statistics.output_size_in_bytes
            == expected_eigenvalue_bytes
        )
        assert jnp.allclose(
            eigenvalues,
            jnp.sort(analytic, axis=-1),
            rtol=0.0,
            atol=1e-12,
        )

        diagonalize_model: TBModel = _make_dispersive_diagonal_model(32)
        diagonalize_kpoints: Float64[Array, "n_k 3"] = _kpoints(32)
        compiled_diagonalize: Callable[..., Any] = eqx.filter_jit(
            lambda points: diagonalize_tb(diagonalize_model, points)
        )
        bands: Any = compiled_diagonalize(diagonalize_kpoints)
        bands.eigenvalues.block_until_ready()
        assert bands.eigenvalues.shape == (32, 32)
        assert bands.eigenvectors.shape == (32, 32, 32)
        assert jnp.all(jnp.isfinite(bands.eigenvalues))
        assert jnp.all(jnp.isfinite(bands.eigenvectors))


class TestEigenvalueReverseScalability:
    """Validate the bounded reverse-mode eigvalsh path of eigvalsh-reverse-mode-bounded."""

    def test_reverse_mode_executes_with_linear_batch_memory(self) -> None:
        """Reject a superlinear tape while matching an analytic gradient.

        The case compares two batch sizes for the same static orbital count.

        Notes
        -----
        Compile the same 16-band reverse loss at 8 and 32 k-points. The
        compiler temporary-memory report may include a fixed backend
        workspace, but growth may not exceed the fourfold batch increase plus
        64 KiB. The reverse JAXPR may materialize the required
        ``(n_k,n_so,n_so)`` eigensystem but no larger array.
        """
        n_orbitals: int = 16
        small_n_k: int = 8
        large_n_k: int = 32
        model: TBModel = _make_dispersive_diagonal_model(n_orbitals)
        initial: Float64[Array, " n"] = jnp.real(
            model.hopping_amplitudes[:n_orbitals]
        )

        def loss(
            hopping: Float64[Array, " n"],
            kpoints: Float64[Array, "n_k 3"],
        ) -> Float64[Array, ""]:
            closed: Complex128[Array, " n_hop"] = jnp.concatenate(
                (hopping, hopping)
            ).astype(jnp.complex128)
            candidate: TBModel = eqx.tree_at(
                lambda item: item.hopping_amplitudes,
                model,
                closed,
            )
            eigenvalues: Float64[Array, "n_k n"] = eigvalsh_bands(
                candidate,
                kpoints,
            )
            value: Float64[Array, ""] = jnp.mean(eigenvalues**2)
            return value

        reverse: Any = jax.jit(jax.value_and_grad(loss))
        small_kpoints: Float64[Array, "n_k 3"] = _kpoints(small_n_k)
        large_kpoints: Float64[Array, "n_k 3"] = _kpoints(large_n_k)
        small_executable: Any = reverse.lower(
            initial,
            small_kpoints,
        ).compile()
        large_executable: Any = reverse.lower(
            initial,
            large_kpoints,
        ).compile()
        small_statistics: Any = _memory_analysis(small_executable)
        large_statistics: Any = _memory_analysis(large_executable)
        scale: int = large_n_k // small_n_k
        assert large_statistics.temp_size_in_bytes <= (
            scale * small_statistics.temp_size_in_bytes + 64 * 1024
        )

        value: Float64[Array, ""]
        gradient: Float64[Array, " n"]
        value, gradient = large_executable(initial, large_kpoints)
        value.block_until_ready()
        gradient.block_until_ready()
        cosine: Float64[Array, " n_k"] = jnp.cos(
            2.0 * jnp.pi * large_kpoints[:, 0]
        )
        unsorted_energies: Float64[Array, "n_k n"] = (
            model.onsite_energies[None, :]
            + 2.0 * cosine[:, None] * initial[None, :]
        )
        expected_gradient: Float64[Array, " n"] = (
            4.0
            * jnp.sum(unsorted_energies * cosine[:, None], axis=0)
            / (large_n_k * n_orbitals)
        )
        assert jnp.isfinite(value)
        assert jnp.all(jnp.isfinite(gradient))
        assert jnp.linalg.norm(gradient) > 0.0
        assert jnp.allclose(
            gradient,
            expected_gradient,
            rtol=1e-11,
            atol=1e-12,
        )

        reverse_jaxpr: ClosedJaxpr = jax.make_jaxpr(jax.grad(loss))(
            initial,
            large_kpoints,
        )
        shapes: list[tuple[int, ...]] = []
        _collect_jaxpr_shapes(reverse_jaxpr, shapes)
        largest_array: int = max(math.prod(shape) for shape in shapes)
        assert largest_array <= large_n_k * n_orbitals**2


class TestBlochPhaseDifferentiability:
    """Validate the holomorphic phase sub-block required by holomorphic-phase-gradient."""

    def test_phase_direction_matches_complex_step_at_machine_precision(
        self,
    ) -> None:
        r"""Verify real-channel complex steps for :math:`e^{2\pi i k\cdot d}`.

        The case also compares the recombined derivative with a direct JVP.

        Notes
        -----
        The elementary complex-step formula applies to real-valued
        holomorphic extensions. The phase is therefore represented by its
        independently holomorphic cosine and sine channels, differentiated
        with ``h=1e-20``, and recombined. This avoids applying the real-output
        formula incorrectly to a complex value with a nonzero imaginary
        baseline.
        """
        kpoint: Float64[Array, " 3"] = jnp.asarray(
            [0.31, -0.27, 0.19],
            dtype=jnp.float64,
        )
        displacement: Float64[Array, " 3"] = jnp.asarray(
            [1.13, -0.42, 0.28],
            dtype=jnp.float64,
        )
        direction: Float64[Array, " 3"] = jnp.asarray(
            [-0.37, 0.51, 0.23],
            dtype=jnp.float64,
        )

        def phase_channels(step: Array) -> Array:
            angle: Array = (
                2.0
                * jnp.pi
                * jnp.dot(
                    kpoint,
                    displacement + step * direction,
                )
            )
            channels: Array = jnp.stack((jnp.cos(angle), jnp.sin(angle)))
            return channels

        origin: Float64[Array, ""] = jnp.asarray(0.0, dtype=jnp.float64)
        derivative_channels: Float64[Array, " 2"] = complex_step_derivative(
            phase_channels,
            origin,
            h=1e-20,
        )
        complex_step: Complex128[Array, ""] = (
            derivative_channels[0] + 1j * derivative_channels[1]
        )
        angle: Float64[Array, ""] = (
            2.0
            * jnp.pi
            * jnp.dot(
                kpoint,
                displacement,
            )
        )
        phase: Complex128[Array, ""] = jnp.exp(1j * angle)
        angle_rate: Float64[Array, ""] = (
            2.0
            * jnp.pi
            * jnp.dot(
                kpoint,
                direction,
            )
        )
        expected: Complex128[Array, ""] = 1j * angle_rate * phase

        def complex_phase(step: Array) -> Array:
            shifted: Array = displacement + step * direction
            return jnp.exp(2j * jnp.pi * jnp.dot(kpoint, shifted))

        _: Complex128[Array, ""]
        jvp: Complex128[Array, ""]
        _, jvp = jax.jvp(
            complex_phase,
            (origin,),
            (jnp.ones_like(origin),),
        )
        baseline_channels: Array = phase_channels(origin)
        assert jnp.allclose(
            phase,
            baseline_channels[0] + 1j * baseline_channels[1],
            rtol=0.0,
            atol=1e-15,
        )
        assert jnp.allclose(
            complex_step,
            expected,
            rtol=2e-15,
            atol=2e-15,
        )
        assert jnp.allclose(jvp, expected, rtol=2e-15, atol=2e-15)
