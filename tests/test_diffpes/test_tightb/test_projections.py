r"""Validate gauge-invariant tight-binding projection utilities.

The tests cover orbital normalization, U(1) phase invariance, fixed-group
unitary invariance, exact-degeneracy averaging, analytic graphene weights,
and differentiation of a rotating degenerate projector.
"""

import jax
import jax.numpy as jnp
import pytest
from hypothesis import assume, given, settings, strategies
from jaxtyping import Array

from diffpes.tightb import (
    band_projectors,
    diagonalize_tb,
    expectation_path,
    fat_bands,
    group_projector,
    group_trace,
    orbital_weights,
)
from diffpes.tightb.diagonalize import eigh_safe
from diffpes.types import (
    DiagonalizedBands,
    OrbitalBasis,
    make_crystal_geometry,
    make_diagonalized_bands,
    make_orbital_basis,
)
from tests._factories import make_graphene_model
from tests._gradients import gradient_gate


def _basis(n_orbitals: int) -> OrbitalBasis:
    """Build static s-orbital metadata for a test eigensystem."""
    basis: OrbitalBasis = make_orbital_basis(
        atom_indices=(0,) * n_orbitals,
        n=(1,) * n_orbitals,
        l=(0,) * n_orbitals,
        m=(0,) * n_orbitals,
        labels=tuple(f"o{index}" for index in range(n_orbitals)),
    )
    return basis


def _bands(
    eigenvalues: Array,
    eigenvectors: Array,
    basis: OrbitalBasis | None = None,
) -> DiagonalizedBands:
    """Attach minimal geometry to a supplied band-major eigensystem."""
    resolved_basis: OrbitalBasis = (
        _basis(eigenvectors.shape[-1]) if basis is None else basis
    )
    bands: DiagonalizedBands = make_diagonalized_bands(
        eigenvalues=jnp.asarray(eigenvalues, dtype=jnp.float64),
        eigenvectors=jnp.asarray(eigenvectors, dtype=jnp.complex128),
        kpoints=jnp.zeros((eigenvalues.shape[0], 3), dtype=jnp.float64),
        geometry=make_crystal_geometry(
            lattice=jnp.eye(3, dtype=jnp.float64),
            positions=jnp.zeros((1, 3), dtype=jnp.float64),
            species=("X",),
        ),
        basis=resolved_basis,
    )
    return bands


class TestOrbitalWeights:
    """Validate :func:`diffpes.tightb.orbital_weights`."""

    def test_complex_values_are_modulus_squared(self) -> None:
        """Distinguish complex modulus squared from a component square.

        A normalized three-four-five coefficient exposes incorrect complex
        multiplication.

        Notes
        -----
        Compare every orbital weight with a literal diagonal result.
        """
        coefficient: complex = (3.0 + 4.0j) / 5.0
        eigenvectors: Array = jnp.asarray(
            [[[coefficient, 0.0], [0.0, coefficient]]],
            dtype=jnp.complex128,
        )
        actual: Array = orbital_weights(eigenvectors)
        expected: Array = jnp.asarray(
            [[[1.0, 0.0], [0.0, 1.0]]],
            dtype=jnp.float64,
        )

        assert jnp.allclose(actual, expected, rtol=0.0, atol=1e-15)

    @given(
        strategies.lists(
            strategies.floats(-1.0, 1.0),
            min_size=16,
            max_size=16,
        )
    )
    @settings(max_examples=24, deadline=None)
    def test_normalized_complex_states_sum_to_one(
        self,
        components: list[float],
    ) -> None:
        """Sum to one for generated normalized complex eigenvectors.

        Property samples exercise unrelated real and imaginary components.

        Notes
        -----
        Normalize each state before reducing its returned orbital weights.
        """
        raw: Array = jnp.asarray(components[:8]) + 1j * jnp.asarray(
            components[8:]
        )
        raw = raw.reshape(2, 4)
        norms: Array = jnp.linalg.norm(raw, axis=-1, keepdims=True)
        assume(bool(jnp.all(norms > 1e-8)))
        normalized: Array = (raw / norms)[None, :, :]
        sums: Array = jnp.sum(orbital_weights(normalized), axis=-1)

        assert jnp.allclose(sums, 1.0, rtol=0.0, atol=1e-12)


class TestBandProjectors:
    """Validate :func:`diffpes.tightb.band_projectors`."""

    def test_band_projectors_ignore_independent_u1_phases(self) -> None:
        """Remove one arbitrary complex phase from every band projector.

        Independent phases at each k-point and band must leave all entries
        unchanged.

        Notes
        -----
        Compare projectors before and after multiplying normalized vectors.
        """
        key: Array = jax.random.key(14)
        real: Array = jax.random.normal(key, (3, 4, 4))
        imaginary: Array = jax.random.normal(
            jax.random.fold_in(key, 1),
            (3, 4, 4),
        )
        vectors: Array = real + 1j * imaginary
        vectors = vectors / jnp.linalg.norm(vectors, axis=-1, keepdims=True)
        phases: Array = jnp.exp(
            1j
            * jnp.asarray(
                [[0.2, -1.1, 2.4, 0.7], [0.9, 0.3, -2.0, 1.6], [2.2] * 4]
            )
        )

        assert jnp.allclose(
            band_projectors(vectors * phases[:, :, None]),
            band_projectors(vectors),
            rtol=1e-13,
            atol=1e-13,
        )


class TestGroupProjector:
    """Validate :func:`diffpes.tightb.group_projector`."""

    def test_fixed_groups_ignore_random_two_by_two_rotations(self) -> None:
        """Preserve group projectors, traces, and averaged diagnostics.

        A random unitary rotates one exact pair while phases rotate isolated
        bands.

        Notes
        -----
        Compare every invariant reduction before and after the basis change.
        """
        key: Array = jax.random.key(27)
        raw: Array = jax.random.normal(key, (4, 4)) + 1j * jax.random.normal(
            jax.random.fold_in(key, 1),
            (4, 4),
        )
        columns: Array
        columns, _ = jnp.linalg.qr(raw)
        vectors: Array = columns.conj().T[None, :, :]
        values: Array = jnp.asarray([[0.0, 0.0, 2.0, 2.0]])
        original: DiagonalizedBands = _bands(values, vectors)

        rotation_raw: Array = jax.random.normal(
            jax.random.fold_in(key, 2),
            (2, 2),
        ) + 1j * jax.random.normal(jax.random.fold_in(key, 3), (2, 2))
        rotation: Array
        rotation, _ = jnp.linalg.qr(rotation_raw)
        rotated_vectors: Array = vectors.at[:, :2, :].set(
            jnp.einsum("ab,kbo->kao", rotation, vectors[:, :2, :])
        )
        phases: Array = jnp.exp(1j * jnp.asarray([0.3, -0.7, 1.1, 2.0]))
        rotated_vectors = rotated_vectors.at[:, 2:, :].set(
            vectors[:, 2:, :] * phases[None, 2:, None]
        )
        rotated: DiagonalizedBands = _bands(values, rotated_vectors)
        operator_raw: Array = jnp.asarray(
            [
                [0.2, 0.1 + 0.3j, -0.2j, 0.4],
                [0.1 - 0.3j, -0.7, 0.2, 0.1j],
                [0.2j, 0.2, 1.1, -0.5j],
                [0.4, -0.1j, 0.5j, 0.3],
            ],
            dtype=jnp.complex128,
        )

        assert jnp.allclose(
            group_projector(rotated, (0, 1)),
            group_projector(original, (0, 1)),
            rtol=1e-12,
            atol=1e-12,
        )
        assert jnp.allclose(
            group_trace(rotated, operator_raw, (0, 1)),
            group_trace(original, operator_raw, (0, 1)),
            rtol=1e-12,
            atol=1e-12,
        )
        assert jnp.allclose(
            expectation_path(rotated, operator_raw),
            expectation_path(original, operator_raw),
            rtol=1e-12,
            atol=1e-12,
        )
        assert jnp.allclose(
            fat_bands(rotated, (0, 2)),
            fat_bands(original, (0, 2)),
            rtol=1e-12,
            atol=1e-12,
        )

    def test_group_projector_is_hermitian_and_idempotent(self) -> None:
        """Construct a rank-two orthogonal projector.

        Selecting two canonical basis states provides an exact matrix
        identity.

        Notes
        -----
        Check adjoint symmetry, idempotency, and trace without tolerance.
        """
        vectors: Array = jnp.eye(3, dtype=jnp.complex128)[None, :, :]
        bands: DiagonalizedBands = _bands(
            jnp.asarray([[0.0, 1.0, 2.0]]),
            vectors,
        )
        projector: Array = group_projector(bands, (0, 2))[0]

        assert jnp.allclose(
            projector,
            projector.conj().T,
            rtol=0.0,
            atol=0.0,
        )
        assert jnp.allclose(
            projector @ projector,
            projector,
            rtol=0.0,
            atol=0.0,
        )
        assert jnp.trace(projector) == pytest.approx(2.0)

    @pytest.mark.parametrize("group", ((), (0, 0), (-1,), (3,)))
    def test_rejects_invalid_fixed_groups(
        self,
        group: tuple[int, ...],
    ) -> None:
        """Reject empty, duplicated, and out-of-range group metadata.

        Static band groups must satisfy every selection invariant.

        Notes
        -----
        Parameterize malformed tuples and match the group diagnostic.
        """
        bands: DiagonalizedBands = _bands(
            jnp.asarray([[0.0, 1.0, 2.0]]),
            jnp.eye(3, dtype=jnp.complex128)[None, :, :],
        )

        with pytest.raises(ValueError, match="group"):
            group_projector(bands, group)


class TestFatBands:
    """Validate :func:`diffpes.tightb.fat_bands`."""

    def test_graphene_sublattice_fat_bands_are_half_at_dirac_k(self) -> None:
        """Recover one-half weight on each sublattice at exact degeneracy.

        The registered graphene Dirac point supplies equal two-orbital
        character.

        Notes
        -----
        Diagonalize the exact K point and compare both sublattice selections.
        """
        bands: DiagonalizedBands = diagonalize_tb(
            make_graphene_model(),
            jnp.asarray([[2.0 / 3.0, 1.0 / 3.0, 0.0]]),
        )

        assert jnp.allclose(
            fat_bands(bands, (0,)),
            0.5,
            rtol=0.0,
            atol=1e-12,
        )
        assert jnp.allclose(
            fat_bands(bands, (1,)),
            0.5,
            rtol=0.0,
            atol=1e-12,
        )


class TestExpectationPath:
    """Validate :func:`diffpes.tightb.expectation_path`."""

    def test_expectation_path_averages_only_registered_energy_blocks(
        self,
    ) -> None:
        """Apply pair averaging while preserving an isolated band.

        A near-degenerate pair lies inside the threshold and a third state
        remains outside.

        Notes
        -----
        Compare the three diagonal expectations with their analytic averages.
        """
        values: Array = jnp.asarray([[0.0, 5e-12, 1.0]])
        vectors: Array = jnp.eye(3, dtype=jnp.complex128)[None, :, :]
        bands: DiagonalizedBands = _bands(values, vectors)
        operator: Array = jnp.diag(
            jnp.asarray([1.0, 3.0, 8.0], dtype=jnp.complex128)
        )

        actual: Array = expectation_path(bands, operator, degen_tol=1e-10)

        assert jnp.allclose(
            actual,
            jnp.asarray([[2.0, 2.0, 8.0]]),
            rtol=0.0,
            atol=1e-14,
        )


class TestGroupTrace:
    """Validate :func:`diffpes.tightb.group_trace`."""

    def test_fixed_group_trace_gradient_at_exact_degeneracy(self) -> None:
        """Match autodiff and FD as a degenerate subspace rotates.

        A fixed two-band projector changes smoothly without choosing internal
        eigenvectors.

        Notes
        -----
        Differentiate the operator trace through a parameterized Hamiltonian.
        """
        basis: OrbitalBasis = _basis(3)
        geometry = make_crystal_geometry(
            lattice=jnp.eye(3, dtype=jnp.float64),
            positions=jnp.zeros((1, 3), dtype=jnp.float64),
            species=("X",),
        )
        operator: Array = jnp.diag(
            jnp.asarray([0.2, -0.6, 1.4], dtype=jnp.complex128)
        )

        def objective(theta: Array) -> Array:
            cosine: Array = jnp.cos(theta)
            sine: Array = jnp.sin(theta)
            rotation: Array = jnp.asarray(
                [
                    [cosine, 0.0, sine],
                    [0.0, 1.0, 0.0],
                    [-sine, 0.0, cosine],
                ],
                dtype=jnp.complex128,
            )
            hamiltonian: Array = (
                rotation
                @ jnp.diag(jnp.asarray([0.0, 0.0, 2.0]))
                @ rotation.conj().T
            )
            values: Array
            columns: Array
            values, columns = eigh_safe(hamiltonian)
            bands: DiagonalizedBands = make_diagonalized_bands(
                eigenvalues=values[None, :],
                eigenvectors=columns.T[None, :, :],
                kpoints=jnp.zeros((1, 3)),
                geometry=geometry,
                basis=basis,
            )
            result: Array = group_trace(bands, operator, (0, 1))[0]
            return result

        gradient_gate(objective, jnp.asarray(0.31), atol=3e-7)


__all__: list[str] = []
