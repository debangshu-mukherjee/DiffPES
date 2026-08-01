"""Validate surface and layer-resolved tight-binding operators.

The tests exercise slab numerical and structural contracts.
"""

import chex
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from beartype.typing import Any
from jaxtyping import Array

from diffpes.tightb import (
    layer_resolved_group_traces,
    layer_resolved_weights,
    surface_projector,
)
from diffpes.types import (
    DEGENERACY_GROUP_TOL_EV,
    GROUP_COMPLEMENT_GAP_MIN_EV,
    DiagonalizedBands,
    make_crystal_geometry,
    make_diagonalized_bands,
    make_orbital_basis,
)


def _bands(
    eigenvalues: Array,
    eigenvectors: Array,
    depths: Array | None,
) -> DiagonalizedBands:
    """Attach a minimal geometry and depth carrier to an eigensystem."""
    n_orbitals: int = eigenvectors.shape[-1]
    return make_diagonalized_bands(
        eigenvalues=jnp.asarray(eigenvalues, dtype=jnp.float64),
        eigenvectors=jnp.asarray(eigenvectors, dtype=jnp.complex128),
        kpoints=jnp.zeros((eigenvalues.shape[0], 3), dtype=jnp.float64),
        geometry=make_crystal_geometry(
            lattice=jnp.eye(3, dtype=jnp.float64),
            positions=jnp.zeros((1, 3), dtype=jnp.float64),
            species=("X",),
        ),
        basis=make_orbital_basis(
            atom_indices=(0,) * n_orbitals,
            n=(1,) * n_orbitals,
            l=(0,) * n_orbitals,
            m=(0,) * n_orbitals,
            labels=tuple(f"o{index}" for index in range(n_orbitals)),
        ),
        depths=depths,
    )


def _weighted_identity_bands(
    weights: Array,
    eigenvalues: Array,
) -> DiagonalizedBands:
    """Build canonical eigenvectors from declared surface weights."""
    depths: Array = -jnp.log(weights)
    vectors: Array = jnp.eye(weights.shape[0], dtype=jnp.complex128)[
        None, :, :
    ]
    return _bands(eigenvalues[None, :], vectors, depths)


class TestSurfaceProjector(chex.TestCase):
    """Validate the diagonal surface-probability operator.

    :see: :func:`~diffpes.tightb.surface_projector`
    """

    def test_probability_law_uses_intensity_escape_length(self) -> None:
        """Match exp(-depth/lambda), including the top-surface value.

        Exercise this slab condition with fixed fixtures.

        Notes
        -----
        Compare outputs with declared numerical or structural references.
        """
        depths: Array = jnp.asarray([0.0, 1.5, 4.0], dtype=jnp.float64)
        escape_length: float = 2.5

        actual: Array = surface_projector(depths, escape_length)
        expected: Array = jnp.exp(-depths / escape_length)

        assert jnp.allclose(actual, expected, rtol=0.0, atol=1e-15)
        assert actual[0] == 1.0

    def test_rejects_invalid_escape_length(self) -> None:
        """Reject nonpositive and non-finite physical escape lengths.

        Exercise this slab condition with fixed fixtures.

        Notes
        -----
        Compare outputs with declared numerical or structural references.
        """
        escape_length: Any
        for escape_length in (0.0, -1.0, float("nan"), float("inf")):
            with pytest.raises(RuntimeError, match="finite and positive"):
                surface_projector(jnp.asarray([0.0, 1.0]), escape_length)

    def test_rejects_invalid_depths(self) -> None:
        """Reject negative and non-finite depths under eager evaluation.

        Exercise this slab condition with fixed fixtures.

        Notes
        -----
        Compare outputs with declared numerical or structural references.
        """
        with pytest.raises(RuntimeError, match="finite and nonnegative"):
            surface_projector(jnp.asarray([0.0, -1e-3]), 1.0)
        with pytest.raises(RuntimeError, match="finite and nonnegative"):
            surface_projector(jnp.asarray([0.0, jnp.nan]), 1.0)

    def test_gradients_match_analytic_probability_derivatives(self) -> None:
        """Differentiate through escape length and scaled depth tags.

        Exercise this slab condition with fixed fixtures.

        Notes
        -----
        Compare outputs with declared numerical or structural references.
        """
        base_depths: Array = jnp.asarray([0.0, 0.7, 2.1])
        escape_length: float = 1.9
        value_grad: Array = jax.grad(
            lambda length: jnp.sum(surface_projector(base_depths, length))
        )(escape_length)
        expected_value_grad: Array = jnp.sum(
            base_depths
            / escape_length**2
            * jnp.exp(-base_depths / escape_length)
        )
        depth_scale_grad: Array = jax.grad(
            lambda scale: jnp.sum(
                surface_projector(scale * base_depths, escape_length)
            )
        )(1.3)
        expected_depth_scale_grad: Array = jnp.sum(
            -base_depths
            / escape_length
            * jnp.exp(-1.3 * base_depths / escape_length)
        )

        assert jnp.allclose(
            value_grad,
            expected_value_grad,
            rtol=1e-13,
            atol=1e-13,
        )
        assert jnp.allclose(
            depth_scale_grad,
            expected_depth_scale_grad,
            rtol=1e-13,
            atol=1e-13,
        )

    def test_small_length_guard_has_finite_value_and_gradient(self) -> None:
        """Prevent the inactive exponential derivative from producing NaN.

        Exercise this slab condition with fixed fixtures.

        Notes
        -----
        Compare outputs with declared numerical or structural references.
        """
        escape_length: Any
        depths: Array = jnp.asarray([0.0, 0.2, 3.0])
        for escape_length in (1e-8, 1e-16):
            value: Array = jnp.sum(surface_projector(depths, escape_length))
            derivative: Array = jax.grad(
                lambda length: jnp.sum(surface_projector(depths, length))
            )(escape_length)

            assert jnp.isfinite(value)
            assert jnp.isfinite(derivative)

    @chex.variants(with_jit=True, without_jit=True)
    def test_variants_and_vmap_match_probability_law(self) -> None:
        """Preserve batched values under eager and compiled execution.

        Exercise this slab condition with fixed fixtures.

        Notes
        -----
        Compare outputs with declared numerical or structural references.
        """
        evaluate: Any
        depth_batches: Array = jnp.asarray([[0.0, 1.0, 2.0], [0.0, 0.5, 1.5]])
        expected: Array = jnp.exp(-depth_batches / 2.0)
        evaluate = self.variant(jax.vmap(surface_projector, in_axes=(0, None)))
        actual: Array = evaluate(
            depth_batches,
            2.0,
        )

        assert actual.shape == (2, 3)
        assert jnp.allclose(actual, expected, rtol=0.0, atol=0.0)


class TestLayerResolvedWeights(chex.TestCase):
    """Validate per-band surface-probability diagnostics.

    :see: :func:`~diffpes.tightb.layer_resolved_weights`
    """

    def test_finite_chain_matches_standing_wave_truth(self) -> None:
        """Recover the geometric-weighted analytic sin-squared sum.

        Exercise this slab condition with fixed fixtures.

        Notes
        -----
        Compare outputs with declared numerical or structural references.
        """
        n_sites: int = 5
        sites: Array = jnp.arange(1, n_sites + 1, dtype=jnp.float64)
        modes: Array = jnp.arange(1, n_sites + 1, dtype=jnp.float64)
        standing_waves: Array = jnp.sqrt(2.0 / (n_sites + 1)) * jnp.sin(
            jnp.pi * modes[:, None] * sites[None, :] / (n_sites + 1)
        )
        depths: Array = jnp.arange(n_sites, dtype=jnp.float64) * 0.8
        bands: DiagonalizedBands = _bands(
            jnp.arange(n_sites, dtype=jnp.float64)[None, :] * 1e-2,
            standing_waves[None, :, :],
            depths,
        )

        actual: Array = layer_resolved_weights(bands, 2.3)[0]
        expected: Array = jnp.sum(
            standing_waves**2 * jnp.exp(-depths / 2.3)[None, :],
            axis=-1,
        )
        trace: Array = layer_resolved_group_traces(
            bands,
            ((0, 1),),
            2.3,
        )[0, 0]

        assert jnp.allclose(actual, expected, rtol=1e-12, atol=1e-12)
        assert jnp.allclose(
            trace,
            jnp.sum(expected[:2]),
            rtol=1e-12,
            atol=1e-12,
        )

    def test_absent_depth_carrier_is_rejected(self) -> None:
        """Keep bulk bands outside the layer-resolved operator API.

        Exercise this slab condition with fixed fixtures.

        Notes
        -----
        Compare outputs with declared numerical or structural references.
        """
        bands: DiagonalizedBands = _bands(
            jnp.asarray([[0.0, 1.0]]),
            jnp.eye(2, dtype=jnp.complex128)[None, :, :],
            None,
        )

        with pytest.raises(ValueError, match="bands.depths"):
            layer_resolved_weights(bands, 1.0)

    @chex.variants(with_jit=True, without_jit=True)
    def test_variants_match_vmap_over_single_k(self) -> None:
        """Match eager and compiled contraction to explicit vmap-over-k.

        Exercise this slab condition with fixed fixtures.

        Notes
        -----
        Compare outputs with declared numerical or structural references.
        """
        evaluate: Any
        key: Array = jax.random.key(55)
        raw: Array = jax.random.normal(key, (3, 3, 3))
        columns: Array
        columns, _ = jnp.linalg.qr(raw)
        vectors: Array = jnp.swapaxes(columns, -1, -2).astype(jnp.complex128)
        bands: DiagonalizedBands = _bands(
            jnp.asarray(
                [[0.0, 0.01, 0.02], [0.03, 0.04, 0.05], [0.06, 0.07, 0.08]]
            ),
            vectors,
            jnp.asarray([0.0, 0.4, 1.1]),
        )

        def one_k(vector: Array) -> Array:
            single: DiagonalizedBands = eqx.tree_at(
                lambda item: item.eigenvectors,
                bands,
                vector[None, :, :],
            )
            return layer_resolved_weights(single, 1.7)[0]

        expected: Array = jax.vmap(one_k)(vectors)
        evaluate = self.variant(lambda item: layer_resolved_weights(item, 1.7))
        actual: Array = evaluate(bands)

        assert actual.shape == (3, 3)
        assert jnp.allclose(actual, expected, rtol=1e-13, atol=1e-13)


class TestLayerResolvedGroupTraces(chex.TestCase):
    """Validate gauge-invariant fixed-group surface traces.

    :see: :func:`~diffpes.tightb.layer_resolved_group_traces`
    """

    def test_u2_mixing_changes_individuals_but_preserves_trace(self) -> None:
        """Pin the unequal-weight Hadamard counterexample from layer-group-trace-basis-invariance.

        Exercise this slab condition with fixed fixtures.

        Notes
        -----
        Compare outputs with declared numerical or structural references.
        """
        bands: DiagonalizedBands = _weighted_identity_bands(
            jnp.asarray([0.2, 0.8, 0.4]),
            jnp.asarray([0.0, 0.0, 2e-3]),
        )
        hadamard: Array = jnp.asarray([[1.0, 1.0], [1.0, -1.0]]) / jnp.sqrt(
            2.0
        )
        rotated_vectors: Array = bands.eigenvectors.at[:, :2, :].set(
            jnp.einsum(
                "ab,kbo->kao",
                hadamard,
                bands.eigenvectors[:, :2, :],
            )
        )
        rotated: DiagonalizedBands = eqx.tree_at(
            lambda item: item.eigenvectors,
            bands,
            rotated_vectors,
        )

        original_weights: Array = layer_resolved_weights(bands, 1.0)[0]
        rotated_weights: Array = layer_resolved_weights(rotated, 1.0)[0]
        original_trace: Array = layer_resolved_group_traces(
            bands,
            ((0, 1),),
            1.0,
        )
        rotated_trace: Array = layer_resolved_group_traces(
            rotated,
            ((0, 1),),
            1.0,
        )

        assert jnp.allclose(
            original_weights,
            jnp.asarray([0.2, 0.8, 0.4]),
            rtol=0.0,
            atol=1e-14,
        )
        assert jnp.allclose(
            rotated_weights[:2],
            jnp.asarray([0.5, 0.5]),
            rtol=0.0,
            atol=1e-14,
        )
        assert jnp.allclose(
            original_trace,
            1.0,
            rtol=5e-13,
            atol=5e-13,
        )
        assert jnp.allclose(
            rotated_trace,
            original_trace,
            rtol=5e-13,
            atol=5e-13,
        )
        np.testing.assert_array_max_ulp(
            np.asarray(rotated_trace),
            np.asarray(original_trace),
            maxulp=32,
        )

    def test_u3_partial_group_is_rejected_but_full_trace_is_invariant(
        self,
    ) -> None:
        """Reject a boundary cut whose apparent trace changes 0.5 to 0.75.

        Exercise this slab condition with fixed fixtures.

        Notes
        -----
        Compare outputs with declared numerical or structural references.
        """
        bands: DiagonalizedBands = _weighted_identity_bands(
            jnp.asarray([0.1, 0.4, 0.9]),
            jnp.zeros((3,), dtype=jnp.float64),
        )
        rotation: Array = (
            jnp.asarray(
                [
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 1.0],
                    [0.0, 1.0, -1.0],
                ]
            )
            / jnp.asarray([1.0, jnp.sqrt(2.0), jnp.sqrt(2.0)])[:, None]
        )
        rotated: DiagonalizedBands = eqx.tree_at(
            lambda item: item.eigenvectors,
            bands,
            jnp.einsum("ab,kbo->kao", rotation, bands.eigenvectors),
        )
        original_partial: Array = jnp.sum(
            layer_resolved_weights(bands, 1.0)[0, :2]
        )
        rotated_partial: Array = jnp.sum(
            layer_resolved_weights(rotated, 1.0)[0, :2]
        )

        assert jnp.allclose(original_partial, 0.5, atol=1e-14)
        assert jnp.allclose(rotated_partial, 0.75, atol=1e-14)
        with pytest.raises(RuntimeError, match="cuts a degenerate multiplet"):
            layer_resolved_group_traces(bands, ((0, 1),), 1.0)
        original_full: Array = layer_resolved_group_traces(
            bands,
            ((0, 1, 2),),
            1.0,
        )
        rotated_full: Array = layer_resolved_group_traces(
            rotated,
            ((0, 1, 2),),
            1.0,
        )
        assert jnp.allclose(original_full, 1.4, atol=1e-14)
        assert jnp.allclose(rotated_full, original_full, atol=1e-14)

    def test_unisolated_complement_and_overlaps_are_rejected(self) -> None:
        """Enforce both runtime spectral and static partition contracts.

        Exercise this slab condition with fixed fixtures.

        Notes
        -----
        Compare outputs with declared numerical or structural references.
        """
        bands: DiagonalizedBands = _weighted_identity_bands(
            jnp.asarray([0.2, 0.8, 0.4]),
            jnp.asarray([0.0, 0.0, 5e-7]),
        )

        with pytest.raises(RuntimeError, match="not complement-isolated"):
            layer_resolved_group_traces(bands, ((0, 1),), 1.0)
        with pytest.raises(ValueError, match="disjoint"):
            layer_resolved_group_traces(
                bands,
                ((0, 1), (1, 2)),
                1.0,
            )

    def test_pinned_gap_boundaries_have_declared_inclusivity(self) -> None:
        """Accept the minimum isolation and reject the degeneracy boundary.

        Exercise this slab condition with fixed fixtures.

        Notes
        -----
        Compare outputs with declared numerical or structural references.
        """
        weights: Array = jnp.asarray([0.2, 0.8, 0.4])
        isolated: DiagonalizedBands = _weighted_identity_bands(
            weights,
            jnp.asarray([0.0, 0.0, GROUP_COMPLEMENT_GAP_MIN_EV]),
        )
        cuts_multiplet: DiagonalizedBands = _weighted_identity_bands(
            weights,
            jnp.asarray([0.0, 0.0, DEGENERACY_GROUP_TOL_EV]),
        )

        trace: Array = layer_resolved_group_traces(
            isolated,
            ((0, 1),),
            1.0,
        )
        assert jnp.allclose(trace, 1.0, atol=1e-14)
        with pytest.raises(RuntimeError, match="cuts a degenerate multiplet"):
            layer_resolved_group_traces(
                cuts_multiplet,
                ((0, 1),),
                1.0,
            )

    def test_malformed_fixed_groups_are_rejected(self) -> None:
        """Reject empty, duplicated, and out-of-range static declarations.

        Exercise this slab condition with fixed fixtures.

        Notes
        -----
        Compare outputs with declared numerical or structural references.
        """
        groups: Any
        bands: DiagonalizedBands = _weighted_identity_bands(
            jnp.asarray([0.2, 0.8, 0.4]),
            jnp.asarray([0.0, 0.0, 2e-3]),
        )

        for groups in ((), ((),), ((0, 0),), ((-1,),), ((3,),)):
            with pytest.raises(ValueError, match="fixed_groups"):
                layer_resolved_group_traces(bands, groups, 1.0)

    @chex.variants(with_jit=True, without_jit=True)
    def test_variants_and_gradient_match_analytic_truth(self) -> None:
        """Differentiate a valid U2 trace under eager and compiled execution.

        Exercise this slab condition with fixed fixtures.

        Notes
        -----
        Compare outputs with declared numerical or structural references.
        """
        evaluate: Any
        bands: DiagonalizedBands = _weighted_identity_bands(
            jnp.asarray([0.2, 0.8, 0.4]),
            jnp.asarray([0.0, 0.0, 2e-3]),
        )
        depths: Array = bands.depths
        escape_length: float = 1.7
        derivative: Array = jax.grad(
            lambda length: jnp.sum(
                layer_resolved_group_traces(
                    bands,
                    ((0, 1),),
                    length,
                )
            )
        )(escape_length)
        expected: Array = jnp.sum(
            depths[:2]
            / escape_length**2
            * jnp.exp(-depths[:2] / escape_length)
        )
        evaluate = self.variant(
            lambda length: layer_resolved_group_traces(
                bands,
                ((0, 1),),
                length,
            )
        )
        actual: Array = evaluate(escape_length)
        expected_trace: Array = layer_resolved_group_traces(
            bands, ((0, 1),), escape_length
        )

        assert jnp.abs(derivative) > 0.0
        assert jnp.allclose(derivative, expected, rtol=1e-13, atol=1e-13)
        assert jnp.allclose(actual, expected_trace, rtol=0.0, atol=0.0)


__all__: list[str] = []
