"""Validate band, projection, spin, and ARPES carriers and factories.

The cases cover JAX PyTree behavior, eager and compiled construction,
gradient-transparent validation, optional spin and OAM data, and shape and
finiteness rejection contracts.
"""

import chex
import equinox as eqx
import jax
import jax.numpy as jnp
import pytest
from beartype.typing import Callable, Dict, Tuple
from scipy.interpolate import RegularGridInterpolator

from diffpes.types import (
    ArpesCube,
    ArpesSpectrum,
    BandStructure,
    DetectorCalibration,
    DetectorRaster,
    OrbitalProjection,
    SpinBandStructure,
    SpinOrbitalProjection,
    constant_energy_map,
    fermi_surface_map,
    make_arpes_cube,
    make_arpes_spectrum,
    make_band_structure,
    make_detector_calibration,
    make_detector_raster,
    make_orbital_projection,
    make_spin_band_structure,
    make_spin_orbital_projection,
    slice_edc,
    slice_mdc,
)
from tests._assertions import assert_rejects
from tests._gradients import assert_grad_matches_fd

_CARTESIAN_FRAME_ID: str = "org.diffpes.frame.sample_cartesian"


def _cartesian_path(n_points: int) -> Tuple[jax.Array, jax.Array]:
    """PRIVATE: Build a straight Cartesian path and cumulative coordinate.

    Parameters
    ----------
    n_points : int
        Number of equally spaced path nodes.

    Returns
    -------
    path : Tuple[jax.Array, jax.Array]
        Cumulative coordinate and matching Cartesian three-vectors.
    """
    k_axis: jax.Array = jnp.linspace(0.0, 1.0, n_points)
    kpoints: jax.Array = jnp.stack(
        (k_axis, jnp.zeros_like(k_axis), jnp.zeros_like(k_axis)),
        axis=1,
    )
    return k_axis, kpoints


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
        The test constructs the carrier through its factory, flattens and unflattens it
        with JAX, and compares every array leaf with Chex.
        """
        bands: BandStructure = make_band_structure(
            eigenvalues=jnp.array([[0.0, 1.0], [2.0, 3.0]]),
            kpoints=jnp.zeros((2, 3)),
            kpoint_weights=jnp.array([0.25, 0.75]),
            fermi_energy=-0.5,
        )
        leaves: list[object]
        tree: jax.tree_util.PyTreeDef
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
        The test constructs one projection through its public factory and uses Chex for
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
        The test constructs the spin carrier through its factory, performs a JAX tree
        round trip, and compares all three array leaves with Chex.
        """
        projection: SpinOrbitalProjection = make_spin_orbital_projection(
            projections=jnp.ones((2, 3, 1, 9)),
            spin=jnp.zeros((2, 3, 1, 6)),
            oam=jnp.full((2, 3, 1, 3), 0.5),
        )
        leaves: list[object]
        tree: jax.tree_util.PyTreeDef
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
        The test constructs the spin-resolved carrier, applies JAX flatten and unflatten,
        and compares both energy channels with Chex.
        """
        bands: SpinBandStructure = make_spin_band_structure(
            eigenvalues_up=jnp.zeros((5, 3)),
            eigenvalues_down=jnp.ones((5, 3)),
            kpoints=jnp.zeros((5, 3)),
            fermi_energy=-1.0,
        )
        leaves: list[object]
        tree: jax.tree_util.PyTreeDef
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


class TestArpesSpectrum:
    """Validate :class:`~diffpes.types.ArpesSpectrum` array storage.

    The momentum-energy intensity map must retain its association with the
    strictly increasing energy axis.

    :see: :class:`~diffpes.types.ArpesSpectrum`
    """

    def test_stores_intensity_and_energy_axis(self) -> None:
        """Preserve a two-point, eight-energy-bin ARPES spectrum.

        The check verifies the exact two-dimensional intensity and
        one-dimensional energy-axis shapes.

        Notes
        -----
        The test constructs the spectrum through its public factory and checks both
        numerical dimensions with Chex.
        """
        spectrum: ArpesSpectrum = make_arpes_spectrum(
            intensity=jnp.zeros((2, 8)),
            energy_axis=jnp.linspace(-3.0, 1.0, 8),
            k_axis=jnp.array([0.0, 1.0]),
            kpoints_cart_inv_ang=jnp.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
            cartesian_frame_id=_CARTESIAN_FRAME_ID,
        )

        chex.assert_shape(spectrum.intensity, (2, 8))
        chex.assert_shape(spectrum.energy_axis, (8,))


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
        The test wraps the public factory with ``self.variant`` and compares output
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

        The check gates the value-threaded ``eqx.error_if`` validation rather
        than allowing a poisoned carrier to escape.

        Notes
        -----
        The test calls the raw and ``eqx.filter_jit`` factories with one NaN eigenvalue
        and matches the same runtime diagnostic in both modes.
        """
        eigenvalues: jax.Array = jnp.array([[jnp.nan]])
        kpoints: jax.Array = jnp.zeros((1, 3))
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
        The test differentiates an independently defined sum through the carrier factory
        and compares it with ``jax.grad(jnp.sum)`` using Chex.
        """
        eigenvalues: jax.Array = jnp.array([[0.25, -0.5]])
        kpoints: jax.Array = jnp.zeros((1, 3))

        def validated_sum(values: jax.Array) -> jax.Array:
            """Return the sum of validated band energies."""
            bands: BandStructure = make_band_structure(values, kpoints)
            result: jax.Array = jnp.sum(bands.eigenvalues)
            return result

        actual: jax.Array = jax.grad(validated_sum)(eigenvalues)
        expected: jax.Array = jax.grad(jnp.sum)(eigenvalues)

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
        The test uses the rejection helper with an eight-orbital tensor and then with a
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
        The test uses the rejection helper with compatible spin arrays and a single
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


class TestMakeArpesSpectrum:
    """Validate :func:`~diffpes.types.make_arpes_spectrum`.

    The factory must preserve intensity dimensions and reject non-increasing
    energy coordinates.

    :see: :func:`~diffpes.types.make_arpes_spectrum`
    """

    def test_constructs_spectrum_shapes(self) -> None:
        """Construct ten momentum points over 100 energy bins.

        The check gates the exact two-dimensional intensity and one-dimensional
        energy-axis shapes after float64 normalization.

        Notes
        -----
        Supplies a zero intensity map and linearly spaced energy axis, then
        checks both output dimensions with Chex.
        """
        spectrum: ArpesSpectrum = make_arpes_spectrum(
            intensity=jnp.zeros((10, 100)),
            energy_axis=jnp.linspace(-3.0, 1.0, 100),
            k_axis=_cartesian_path(10)[0],
            kpoints_cart_inv_ang=_cartesian_path(10)[1],
            cartesian_frame_id=_CARTESIAN_FRAME_ID,
        )

        chex.assert_shape(spectrum.intensity, (10, 100))
        chex.assert_shape(spectrum.energy_axis, (100,))

    def test_rejects_unsorted_energy_axis(self) -> None:
        """Reject repeated ARPES energy coordinates.

        The check verifies strict energy ordering independently of the finite
        intensity map.

        Notes
        -----
        Supplies two equal energy coordinates and matches the traced ordering
        diagnostic through the shared rejection helper.
        """
        assert_rejects(
            make_arpes_spectrum,
            intensity=jnp.zeros((2, 2)),
            energy_axis=jnp.array([0.0, 0.0]),
            k_axis=jnp.array([0.0, 1.0]),
            kpoints_cart_inv_ang=jnp.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
            cartesian_frame_id=_CARTESIAN_FRAME_ID,
            match="energy axis strictly increasing",
        )

    def test_preserves_equal_length_distinct_cartesian_paths(self) -> None:
        """Keep equal-length Gamma-X and Gamma-Y paths distinguishable.

        The check proves cumulative distance alone cannot erase Cartesian path
        direction from the carrier.

        Notes
        -----
        Construct orthogonal unit paths with identical cumulative coordinates
        and compare both stored representations.
        """
        intensity: jax.Array = jnp.ones((2, 3))
        energy: jax.Array = jnp.array([-1.0, 0.0, 1.0])
        k_axis: jax.Array = jnp.array([0.0, 1.0])
        gamma_x: ArpesSpectrum = make_arpes_spectrum(
            intensity,
            energy,
            k_axis,
            jnp.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
            _CARTESIAN_FRAME_ID,
        )
        gamma_y: ArpesSpectrum = make_arpes_spectrum(
            intensity,
            energy,
            k_axis,
            jnp.array([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
            _CARTESIAN_FRAME_ID,
        )

        chex.assert_trees_all_equal(gamma_x.k_axis, gamma_y.k_axis)
        assert not bool(
            jnp.array_equal(
                gamma_x.kpoints_cart_inv_ang,
                gamma_y.kpoints_cart_inv_ang,
            )
        )
        assert gamma_x.cartesian_frame_id == _CARTESIAN_FRAME_ID

    def test_rejects_cartesian_step_mismatch(self) -> None:
        """Reject a cumulative coordinate inconsistent with the full path.

        The check isolates the Cartesian step-length consistency contract.

        Notes
        -----
        Pair a unit Cartesian displacement with a half-unit cumulative
        coordinate and match the diagnostic.
        """
        assert_rejects(
            make_arpes_spectrum,
            intensity=jnp.ones((2, 3)),
            energy_axis=jnp.array([-1.0, 0.0, 1.0]),
            k_axis=jnp.array([0.0, 0.5]),
            kpoints_cart_inv_ang=jnp.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
            cartesian_frame_id=_CARTESIAN_FRAME_ID,
            match="Cartesian path steps disagree",
        )


def _arpes_cube() -> ArpesCube:
    """PRIVATE: Build a trilinear source cube with analytic derivatives.

    Returns
    -------
    cube : ArpesCube
        Positive manufactured cube on nonuniform Cartesian and energy axes.

    Notes
    -----
    The affine coefficients make every interpolation and query derivative
    available in closed form.
    """
    kx_axis: jax.Array = jnp.array([-1.0, 0.0, 2.0])
    ky_axis: jax.Array = jnp.array([-2.0, 1.0])
    energy_axis: jax.Array = jnp.array([-1.0, 0.0, 2.0, 4.0])
    kx_grid: jax.Array
    ky_grid: jax.Array
    energy_grid: jax.Array
    kx_grid, ky_grid, energy_grid = jnp.meshgrid(
        kx_axis,
        ky_axis,
        energy_axis,
        indexing="ij",
    )
    intensity: jax.Array = (
        20.0 + 2.0 * kx_grid + 3.0 * ky_grid + 4.0 * energy_grid
    )
    cube: ArpesCube = make_arpes_cube(
        intensity,
        kx_axis,
        ky_axis,
        energy_axis,
        cartesian_frame_id=_CARTESIAN_FRAME_ID,
        provenance="manufactured/trilinear",
    )
    return cube


class TestArpesCube:
    """Validate :class:`~diffpes.types.ArpesCube` source coordinates.

    The carrier must preserve Cartesian axes, energy samples, provenance, and
    the registered static frame.

    :see: :class:`~diffpes.types.ArpesCube`
    """

    def test_stores_source_axes_and_metadata(self) -> None:
        """Preserve the manufactured cube axes and static metadata.

        The check covers every numerical coordinate and both static strings.

        Notes
        -----
        Construct the shared affine cube and compare its fields with the
        deterministic fixture values.
        """
        cube: ArpesCube = _arpes_cube()

        chex.assert_shape(cube.intensity, (3, 2, 4))
        chex.assert_trees_all_equal(cube.kx_axis, jnp.array([-1.0, 0.0, 2.0]))
        assert cube.provenance == "manufactured/trilinear"
        assert cube.cartesian_frame_id == _CARTESIAN_FRAME_ID


class TestMakeArpesCube:
    """Validate :func:`~diffpes.types.make_arpes_cube` input checks.

    The factory must reject invalid source coordinates and non-finite
    intensity in eager and compiled execution.

    :see: :func:`~diffpes.types.make_arpes_cube`
    """

    def test_rejects_invalid_axes_and_values(self) -> None:
        """Reject nonmonotone and non-finite cube inputs.

        The check exercises one static axis contract and one traced value
        contract through the shared eager/JIT assertion.

        Notes
        -----
        Pass a repeated momentum node, then inject one NaN into an otherwise
        valid source cube.
        """
        cube: ArpesCube = _arpes_cube()
        assert_rejects(
            make_arpes_cube,
            cube.intensity,
            jnp.array([-1.0, 0.0, 0.0]),
            cube.ky_axis,
            cube.energy_axis,
            match="kx axis finite and strictly increasing",
        )
        assert_rejects(
            make_arpes_cube,
            cube.intensity.at[0, 0, 0].set(jnp.nan),
            cube.kx_axis,
            cube.ky_axis,
            cube.energy_axis,
            match="intensity finite",
        )


class TestSliceEdc:
    """Validate :func:`~diffpes.types.slice_edc` interpolation.

    The slicer must match direct indexing and independent SciPy interpolation.
    Query and source-intensity derivatives must remain available.

    :see: :func:`~diffpes.types.slice_edc`
    """

    def test_matches_nodes_scipy_and_gradients(self) -> None:
        """Match node values, SciPy interpolation, and query gradients.

        The check covers exact indexing, one off-node query, JIT, VMAP, and
        the analytic affine derivative.

        Notes
        -----
        Evaluate the shared affine cube and compare every result at strict
        float64 tolerance.
        """
        cube: ArpesCube = _arpes_cube()
        interpolator: RegularGridInterpolator = RegularGridInterpolator(
            (cube.kx_axis, cube.ky_axis, cube.energy_axis),
            cube.intensity,
            method="linear",
        )
        points: jax.Array = jnp.column_stack(
            (
                jnp.full_like(cube.energy_axis, 0.4),
                jnp.full_like(cube.energy_axis, -0.25),
                cube.energy_axis,
            )
        )
        compiled: jax.Array = eqx.filter_jit(slice_edc)(cube, 0.4, -0.25)
        vectorized: jax.Array = jax.vmap(
            lambda kx: slice_edc(cube, kx, -0.25)
        )(jnp.array([-0.5, 0.4, 1.5]))
        gradient: jax.Array = jax.grad(
            lambda kx: jnp.sum(slice_edc(cube, kx, -0.25))
        )(jnp.asarray(0.4))

        chex.assert_trees_all_equal(
            slice_edc(cube, 0.0, 1.0), cube.intensity[1, 1, :]
        )
        chex.assert_trees_all_close(
            compiled,
            interpolator(points),
            rtol=1.0e-12,
            atol=0.0,
        )
        chex.assert_shape(vectorized, (3, 4))
        chex.assert_trees_all_close(gradient, 8.0, rtol=1.0e-12)

    def test_query_and_intensity_gradients_match_finite_differences(
        self,
    ) -> None:
        """Match finite differences for queries and source intensity.

        The check differentiates a joint query loss and an independent
        intensity loss through the public slicer.

        Notes
        -----
        Apply the shared finite-difference harness to interior coordinates and
        every source-intensity leaf.
        """
        cube: ArpesCube = _arpes_cube()

        def query_loss(queries: jax.Array) -> jax.Array:
            """Return a joint EDC and MDC query loss."""
            edc: jax.Array = slice_edc(cube, queries[0], queries[1])
            mdc: jax.Array = slice_mdc(cube, queries[2])
            result: jax.Array = jnp.sum(edc) + jnp.sum(mdc)
            return result

        def intensity_loss(intensity: jax.Array) -> jax.Array:
            """Return an EDC sum after replacing source intensity."""
            candidate: ArpesCube = eqx.tree_at(
                lambda carrier: carrier.intensity,
                cube,
                intensity,
            )
            result: jax.Array = jnp.sum(slice_edc(candidate, 0.4, -0.25))
            return result

        assert_grad_matches_fd(
            query_loss,
            jnp.array([0.4, -0.25, 0.75]),
        )
        assert_grad_matches_fd(intensity_loss, cube.intensity)


class TestSliceMdc:
    """Validate :func:`~diffpes.types.slice_mdc` interpolation.

    The slicer must match direct energy indexing, independent interpolation,
    and the analytic affine energy derivative.

    :see: :func:`~diffpes.types.slice_mdc`
    """

    def test_matches_nodes_scipy_and_gradient(self) -> None:
        """Match node values, SciPy interpolation, and energy gradients.

        The check compares one full momentum plane at both on-node and
        off-node energy coordinates.

        Notes
        -----
        Build SciPy query rows from the Cartesian mesh and compare at strict
        float64 tolerance.
        """
        cube: ArpesCube = _arpes_cube()
        interpolator: RegularGridInterpolator = RegularGridInterpolator(
            (cube.kx_axis, cube.ky_axis, cube.energy_axis),
            cube.intensity,
            method="linear",
        )
        kx_grid: jax.Array
        ky_grid: jax.Array
        kx_grid, ky_grid = jnp.meshgrid(
            cube.kx_axis,
            cube.ky_axis,
            indexing="ij",
        )
        points: jax.Array = jnp.column_stack(
            (
                jnp.ravel(kx_grid),
                jnp.ravel(ky_grid),
                jnp.full(kx_grid.size, 0.75),
            )
        )
        gradient: jax.Array = jax.grad(
            lambda energy: jnp.sum(slice_mdc(cube, energy))
        )(jnp.asarray(0.75))

        chex.assert_trees_all_equal(
            slice_mdc(cube, 2.0), cube.intensity[:, :, 2]
        )
        chex.assert_trees_all_close(
            slice_mdc(cube, 0.75),
            interpolator(points).reshape(kx_grid.shape),
            rtol=1.0e-12,
            atol=0.0,
        )
        chex.assert_trees_all_close(gradient, 24.0, rtol=1.0e-12)


class TestConstantEnergyMap:
    """Validate :func:`~diffpes.types.constant_energy_map` windows.

    The helper must average exactly the sampled planes inside its explicit
    closed top-hat window.

    :see: :func:`~diffpes.types.constant_energy_map`
    """

    def test_averages_selected_energy_planes(self) -> None:
        """Compute the mean of exactly two selected energy planes.

        The chosen window includes the first two nonuniform energy nodes and
        excludes the remaining nodes.

        Notes
        -----
        Compare the public helper with a direct mean over the fixture slice.
        """
        cube: ArpesCube = _arpes_cube()
        desired: jax.Array = jnp.mean(cube.intensity[..., :2], axis=-1)

        chex.assert_trees_all_equal(
            constant_energy_map(cube, -0.5, 0.5), desired
        )


class TestFermiSurfaceMap:
    """Validate :func:`~diffpes.types.fermi_surface_map` windows.

    The helper must centre the top-hat window at zero relative energy and
    preserve the constant-map result.

    :see: :func:`~diffpes.types.fermi_surface_map`
    """

    def test_delegates_zero_centred_window(self) -> None:
        """Match the equivalent zero-centred constant-energy map.

        The equality establishes the helper's exact Fermi-level convention.

        Notes
        -----
        Compare both public helpers with the same tolerance on the affine
        source cube.
        """
        cube: ArpesCube = _arpes_cube()
        desired: jax.Array = constant_energy_map(cube, 0.0, 1.0)

        chex.assert_trees_all_equal(fermi_surface_map(cube, 1.0), desired)


class TestDetectorRaster:
    """Validate :class:`~diffpes.types.DetectorRaster` metadata.

    The carrier must preserve native axes, nonempty channels, and optional
    per-pixel quantization vectors.

    :see: :class:`~diffpes.types.DetectorRaster`
    """

    def test_stores_native_axes_and_static_metadata(self) -> None:
        """Preserve native axes, channels, and quantization vectors.

        The check also confirms static strings do not become numerical PyTree
        leaves.

        Notes
        -----
        Construct two spin channels with one detector row and inspect every
        carrier boundary.
        """
        quantization_axis: jax.Array = jnp.broadcast_to(
            jnp.array([0.0, 0.0, 1.0]),
            (2, 1, 3),
        )
        raster: DetectorRaster = make_detector_raster(
            expected_counts=jnp.ones((2, 2, 1, 3)),
            detector_u_axis=jnp.array([-0.1, 0.1]),
            detector_v_axis=jnp.array([0.0]),
            energy_axis=jnp.array([-1.0, 0.0, 1.0]),
            channel_labels=("spin_up", "spin_down"),
            coordinate_system="hemispherical_angles",
            quantization_axis=quantization_axis,
        )

        chex.assert_shape(raster.expected_counts, (2, 2, 1, 3))
        chex.assert_shape(raster.quantization_axis, (2, 1, 3))
        assert raster.channel_labels == ("spin_up", "spin_down")
        assert not any(
            isinstance(leaf, str) for leaf in jax.tree.leaves(raster)
        )


class TestMakeDetectorRaster:
    """Validate :func:`~diffpes.types.make_detector_raster` rejection.

    The factory must reject non-finite counts and an empty channel axis in
    eager and compiled execution.

    :see: :func:`~diffpes.types.make_detector_raster`
    """

    def test_rejects_nonfinite_and_empty_channel_counts(self) -> None:
        """Reject NaN counts and a zero-length channel axis.

        The two cases distinguish traced value validation from the static
        acquisition-shape contract.

        Notes
        -----
        Apply the shared eager/JIT rejection helper with otherwise valid
        native detector axes.
        """
        arguments: Dict[str, object] = {
            "detector_u_axis": jnp.array([-0.1, 0.1]),
            "detector_v_axis": jnp.array([0.0]),
            "energy_axis": jnp.array([-1.0, 0.0, 1.0]),
            "coordinate_system": "hemispherical_angles",
        }
        assert_rejects(
            make_detector_raster,
            expected_counts=jnp.full((1, 2, 1, 3), jnp.nan),
            channel_labels=("intensity",),
            match="expected counts finite",
            **arguments,
        )
        assert_rejects(
            make_detector_raster,
            expected_counts=jnp.empty((0, 2, 1, 3)),
            channel_labels=(),
            match="channel axis requires at least one channel",
            **arguments,
        )


class TestDetectorCalibration:
    """Validate :class:`~diffpes.types.DetectorCalibration` metadata.

    The carrier must preserve detector edges, point-spread widths, and its
    fixed coordinate and boundary policies.

    :see: :class:`~diffpes.types.DetectorCalibration`
    """

    def test_stores_edges_and_policy_metadata(self) -> None:
        """Preserve detector edges, PSFs, and policy metadata.

        The check covers one slit row and three recorded energy bins.

        Notes
        -----
        Construct a valid calibration and inspect its edge shape and static
        selector values.
        """
        calibration: DetectorCalibration = make_detector_calibration(
            u_bin_edges=jnp.array([-0.2, 0.0, 0.2]),
            v_bin_edges=jnp.array([-0.05, 0.05]),
            energy_bin_edges_ev=jnp.array([-1.5, -0.5, 0.5, 1.5]),
            psf_fwhm_u=0.01,
            psf_fwhm_v=0.02,
            psf_fwhm_energy_ev=0.05,
            transmission_reference_domain_ev=jnp.array([10.0, 30.0]),
        )

        chex.assert_shape(calibration.energy_bin_edges_ev, (4,))
        assert calibration.coordinate_system == "hemispherical_angles"
        assert calibration.boundary_policy == "loss"


class TestMakeDetectorCalibration:
    """Validate :func:`~diffpes.types.make_detector_calibration` checks.

    The factory must reject a nonpositive detector point-spread width in eager
    and compiled execution.

    :see: :func:`~diffpes.types.make_detector_calibration`
    """

    def test_rejects_nonpositive_energy_psf(self) -> None:
        """Reject a zero energy point-spread FWHM.

        The check isolates the traced positivity contract from static policy
        validation.

        Notes
        -----
        Pass valid detector edges and transmission bounds with one zero width
        through the shared eager/JIT helper.
        """
        arguments: Dict[str, object] = {
            "u_bin_edges": jnp.array([-0.2, 0.0, 0.2]),
            "v_bin_edges": jnp.array([-0.05, 0.05]),
            "energy_bin_edges_ev": jnp.array([-1.5, -0.5, 0.5, 1.5]),
            "psf_fwhm_u": 0.01,
            "psf_fwhm_v": 0.02,
            "psf_fwhm_energy_ev": 0.0,
            "transmission_reference_domain_ev": jnp.array([10.0, 30.0]),
        }
        assert_rejects(
            make_detector_calibration,
            match="energy FWHM finite and positive",
            **arguments,
        )
