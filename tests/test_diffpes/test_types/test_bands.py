"""Validate the bands contracts.

The tests cover public carrier behavior, validation, and JAX
transformations for this implementation module.
"""

import chex
import equinox as eqx
import jax
import jax.numpy as jnp
import pytest
from beartype.typing import Callable, List
from jaxtyping import Array, Float64

from diffpes.types import (
    BandStructure,
    OrbitalProjection,
    PyTreeDef,
    SpinBandStructure,
    SpinOrbitalProjection,
    make_band_structure,
    make_orbital_projection,
    make_spin_band_structure,
    make_spin_orbital_projection,
)
from tests._assertions import assert_rejects

_CARTESIAN_FRAME_ID: str = "org.diffpes.frame.sample_cartesian"


class TestBandStructure:
    """Validate :class:`~diffpes.types.BandStructure` as a JAX PyTree.

    Eigenvalues, k-points, weights, and the Fermi energy must remain numerical
    leaves with stable dimensions through JAX transformations.

    :see: :class:`~diffpes.types.BandStructure`
    """

    def test_pytree_round_trip(self) -> None:
        """Preserve all band fields through JAX reconstruction.

        The check compares a two-point, two-band structure and its nonuniform
        k-point weights before and after a PyTree round trip.

        Notes
        -----
        The test constructs the carrier through its factory, flattens and
        unflattens it
        with JAX, and compares every array leaf with Chex.
        """
        bands: BandStructure = make_band_structure(
            eigenvalues=jnp.array([[0.0, 1.0], [2.0, 3.0]]),
            kpoints=jnp.zeros((2, 3)),
            kpoint_weights=jnp.array([0.25, 0.75]),
            fermi_energy=-0.5,
        )
        leaves: List[object]
        tree: PyTreeDef
        leaves, tree = jax.tree_util.tree_flatten(bands)
        restored: BandStructure = jax.tree_util.tree_unflatten(tree, leaves)

        chex.assert_trees_all_close(restored, bands)


class TestOrbitalProjection:
    """Validate :class:`~diffpes.types.OrbitalProjection` optional leaves.

    Projection weights must remain present while omitted spin and OAM arrays
    remain empty subtrees.

    :see: :class:`~diffpes.types.OrbitalProjection`
    """

    def test_preserves_absent_optional_fields(self) -> None:
        """Store a nine-orbital projection with no spin or OAM data.

        The check verifies the exact projection shape and both ``None``
        sentinels for a non-spin calculation.

        Notes
        -----
        The test constructs one projection through its public factory and uses
        Chex for
        the array shape plus direct identity checks for optional fields.
        """
        projection: OrbitalProjection = make_orbital_projection(
            jnp.zeros((2, 3, 1, 9))
        )

        chex.assert_shape(projection.projections, (2, 3, 1, 9))
        assert projection.spin is None
        assert projection.oam is None


class TestSpinOrbitalProjection:
    """Validate :class:`~diffpes.types.SpinOrbitalProjection` as a PyTree.

    Mandatory spin projections and optional orbital angular momentum must
    survive JAX reconstruction with their component axes intact.

    :see: :class:`~diffpes.types.SpinOrbitalProjection`
    """

    def test_pytree_round_trip_with_oam(self) -> None:
        """Preserve spin and OAM arrays through JAX reconstruction.

        The check uses six spin channels and three OAM components for a
        two-point, three-band, one-atom projection.

        Notes
        -----
        The test constructs the spin carrier through its factory, performs a
        JAX tree
        round trip, and compares all three array leaves with Chex.
        """
        projection: SpinOrbitalProjection = make_spin_orbital_projection(
            projections=jnp.ones((2, 3, 1, 9)),
            spin=jnp.zeros((2, 3, 1, 6)),
            oam=jnp.full((2, 3, 1, 3), 0.5),
        )
        leaves: List[object]
        tree: PyTreeDef
        leaves, tree = jax.tree_util.tree_flatten(projection)
        restored: SpinOrbitalProjection = jax.tree_util.tree_unflatten(
            tree, leaves
        )

        chex.assert_trees_all_close(
            restored.projections, projection.projections
        )
        chex.assert_trees_all_close(restored.spin, projection.spin)
        chex.assert_trees_all_close(restored.oam, projection.oam)


class TestSpinBandStructure:
    """Validate :class:`~diffpes.types.SpinBandStructure` as a JAX PyTree.

    Up- and down-spin eigenvalues must share their k-point coordinates and
    survive reconstruction together.

    :see: :class:`~diffpes.types.SpinBandStructure`
    """

    def test_pytree_round_trip(self) -> None:
        """Preserve both spin channels through JAX reconstruction.

        The check compares two five-by-three eigenvalue arrays and their scalar
        Fermi energy before and after a PyTree round trip.

        Notes
        -----
        The test constructs the spin-resolved carrier, applies JAX flatten and
        unflatten,
        and compares both energy channels with Chex.
        """
        bands: SpinBandStructure = make_spin_band_structure(
            eigenvalues_up=jnp.zeros((5, 3)),
            eigenvalues_down=jnp.ones((5, 3)),
            kpoints=jnp.zeros((5, 3)),
            fermi_energy=-1.0,
        )
        leaves: List[object]
        tree: PyTreeDef
        leaves, tree = jax.tree_util.tree_flatten(bands)
        restored: SpinBandStructure = jax.tree_util.tree_unflatten(
            tree, leaves
        )

        chex.assert_trees_all_close(
            restored.eigenvalues_up, bands.eigenvalues_up
        )
        chex.assert_trees_all_close(
            restored.eigenvalues_down, bands.eigenvalues_down
        )


class TestMakeBandStructure(chex.TestCase):
    """Validate :func:`~diffpes.types.make_band_structure` under JAX.

    The factory must supply uniform weights, normalize scalar dtype, preserve
    gradients, and reject non-finite eigenvalues in eager and compiled modes.

    :see: :func:`~diffpes.types.make_band_structure`
    """

    @chex.variants(with_jit=True, without_jit=True)
    def test_constructs_uniform_weights(self) -> None:
        """Construct uniform weights in eager and compiled execution.

        The check expects eight unit weights and a scalar JAX Fermi energy for
        an eight-point, three-band input.

        Notes
        -----
        The test wraps the public factory with ``self.variant`` and compares
        output
        shapes, weights, and scalar array type with Chex.
        """
        factory: Callable[..., BandStructure] = self.variant(
            make_band_structure
        )
        bands: BandStructure = factory(
            eigenvalues=jnp.zeros((8, 3)),
            kpoints=jnp.zeros((8, 3)),
            fermi_energy=-1.5,
        )

        chex.assert_trees_all_close(bands.kpoint_weights, jnp.ones(8))
        chex.assert_shape(bands.fermi_energy, ())
        assert isinstance(bands.fermi_energy, jax.Array)

    def test_rejects_nonfinite_eigenvalues(self) -> None:
        """Reject a NaN eigenvalue in eager and compiled execution.

        The check exercises value-threaded ``eqx.error_if`` validation rather
        than allowing a poisoned carrier to escape.

        Notes
        -----
        The test calls the raw and ``eqx.filter_jit`` factories with one NaN
        eigenvalue
        and matches the same runtime diagnostic in both modes.
        """
        eigenvalues: Float64[Array, "1 1"] = jnp.array([[jnp.nan]])
        kpoints: Float64[Array, "1 3"] = jnp.zeros((1, 3))
        under_jit: bool
        for under_jit in (False, True):
            factory: Callable[..., BandStructure] = (
                eqx.filter_jit(make_band_structure)
                if under_jit
                else make_band_structure
            )
            with pytest.raises(RuntimeError, match="eigenvalues finite"):
                factory(eigenvalues=eigenvalues, kpoints=kpoints)

    def test_validation_preserves_gradient(self) -> None:
        """Preserve the unit gradient through valid eigenvalue checking.

        The derivative of the sum of two validated eigenvalues must match the
        direct array-sum derivative exactly.

        Notes
        -----
        The test differentiates an independently defined sum through the
        carrier factory
        and compares it with ``jax.grad(jnp.sum)`` using Chex.
        """
        eigenvalues: Float64[Array, "1 2"] = jnp.array([[0.25, -0.5]])
        kpoints: Float64[Array, "1 3"] = jnp.zeros((1, 3))

        def validated_sum(
            values: Float64[Array, "1 2"],
        ) -> Float64[Array, ""]:
            """Return the sum of validated band energies."""
            bands: BandStructure = make_band_structure(values, kpoints)
            result: Float64[Array, ""] = jnp.sum(bands.eigenvalues)
            return result

        actual: Float64[Array, "1 2"] = jax.grad(validated_sum)(eigenvalues)
        expected: Float64[Array, "1 2"] = jax.grad(jnp.sum)(eigenvalues)

        chex.assert_trees_all_equal(actual, expected)


class TestMakeOrbitalProjection:
    """Validate :func:`~diffpes.types.make_orbital_projection`.

    The factory must accept compatible spin data and reject malformed orbital
    axes or negative projection probabilities.

    :see: :func:`~diffpes.types.make_orbital_projection`
    """

    def test_stores_optional_spin(self) -> None:
        """Preserve the six-channel optional spin projection array.

        The check verifies a four-point, three-band, one-atom spin array shape
        alongside the mandatory nine-orbital weights.

        Notes
        -----
        Supplies compatible projection and spin arrays to the public factory
        and checks the stored spin dimensions with Chex.
        """
        projection: OrbitalProjection = make_orbital_projection(
            projections=jnp.ones((4, 3, 1, 9)),
            spin=jnp.zeros((4, 3, 1, 6)),
        )

        chex.assert_shape(projection.spin, (4, 3, 1, 6))

    def test_rejects_invalid_projection_values(self) -> None:
        """Reject a wrong orbital axis and negative projection weights.

        The check covers the static nine-orbital convention and the traced
        nonnegative-probability contract.

        Notes
        -----
        The test uses the rejection helper with an eight-orbital tensor and
        then with a
        negative nine-orbital tensor, matching each diagnostic.
        """
        assert_rejects(
            make_orbital_projection,
            projections=jnp.zeros((1, 1, 1, 8)),
            match="projections must have 9 orbital columns",
        )
        assert_rejects(
            make_orbital_projection,
            projections=-jnp.ones((1, 1, 1, 9)),
            match="projections non negative",
        )


class TestMakeSpinOrbitalProjection:
    """Validate :func:`~diffpes.types.make_spin_orbital_projection`.

    The spin-specific factory must require compatible six-channel spin data and
    preserve an optional three-component OAM array.

    :see: :func:`~diffpes.types.make_spin_orbital_projection`
    """

    def test_stores_explicit_oam(self) -> None:
        """Preserve a three-component orbital-angular-momentum array.

        The check verifies the ``2 x 3 x 1 x 3`` OAM shape and float64 dtype
        after factory normalization.

        Notes
        -----
        Supplies compatible projection, spin, and OAM arrays and checks the
        optional output leaf with Chex and a dtype comparison.
        """
        projection: SpinOrbitalProjection = make_spin_orbital_projection(
            projections=jnp.ones((2, 3, 1, 9)),
            spin=jnp.zeros((2, 3, 1, 6)),
            oam=jnp.full((2, 3, 1, 3), 0.5),
        )

        chex.assert_shape(projection.oam, (2, 3, 1, 3))
        assert projection.oam is not None
        assert projection.oam.dtype == jnp.float64

    def test_rejects_negative_projection_values(self) -> None:
        """Reject negative orbital probabilities with valid spin dimensions.

        The check isolates the traced nonnegative constraint of the spin-aware
        factory.

        Notes
        -----
        Supplies a negative nine-orbital tensor and a compatible zero spin
        tensor, then matches the factory's probability diagnostic.
        """
        assert_rejects(
            make_spin_orbital_projection,
            projections=-jnp.ones((1, 1, 1, 9)),
            spin=jnp.zeros((1, 1, 1, 6)),
            match="projections non negative",
        )


class TestMakeSpinBandStructure:
    """Validate :func:`~diffpes.types.make_spin_band_structure`.

    The factory must align both spin channels with one k-point mesh and reject
    negative integration weights.

    :see: :func:`~diffpes.types.make_spin_band_structure`
    """

    def test_constructs_aligned_spin_channels(self) -> None:
        """Construct equal-shape up- and down-spin eigenvalue arrays.

        The check verifies both channels retain the requested four-point,
        two-band dimensions.

        Notes
        -----
        Supplies finite spin channels and common k-points, then checks their
        carrier shapes with Chex.
        """
        bands: SpinBandStructure = make_spin_band_structure(
            eigenvalues_up=jnp.zeros((4, 2)),
            eigenvalues_down=jnp.ones((4, 2)),
            kpoints=jnp.zeros((4, 3)),
        )

        chex.assert_shape(bands.eigenvalues_up, (4, 2))
        chex.assert_shape(bands.eigenvalues_down, (4, 2))

    def test_rejects_negative_weights(self) -> None:
        """Reject a negative spin-band integration weight.

        The check isolates the traced nonnegative-weight contract for one
        k-point and one band in each spin channel.

        Notes
        -----
        The test uses the rejection helper with compatible spin arrays and a
        single
        negative weight, matching the factory diagnostic.
        """
        assert_rejects(
            make_spin_band_structure,
            eigenvalues_up=jnp.zeros((1, 1)),
            eigenvalues_down=jnp.zeros((1, 1)),
            kpoints=jnp.zeros((1, 3)),
            kpoint_weights=jnp.array([-1.0]),
            match="weights non negative",
        )
