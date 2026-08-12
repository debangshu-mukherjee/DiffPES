"""Validate the arpes contracts.

The tests cover public carrier behavior, validation, and JAX
transformations for this implementation module.
"""

import chex
import equinox as eqx
import jax
import jax.numpy as jnp
from beartype.typing import Tuple
from jaxtyping import Array, Float64
from scipy.interpolate import RegularGridInterpolator

from diffpes.types import (
    ArpesCube,
    ArpesSpectrum,
    constant_energy_map,
    fermi_surface_map,
    make_arpes_cube,
    make_arpes_spectrum,
    slice_edc,
    slice_mdc,
)
from tests._assertions import assert_rejects
from tests._gradients import assert_grad_matches_fd

_CARTESIAN_FRAME_ID: str = "org.diffpes.frame.sample_cartesian"


def _cartesian_path(
    n_points: int,
) -> Tuple[Float64[Array, " n_points"], Float64[Array, "n_points 3"]]:
    """PRIVATE: Build a straight Cartesian path and cumulative coordinate.

    Parameters
    ----------
    n_points : int
        Number of equally spaced path nodes.

    Returns
    -------
    path : Tuple[Float64[Array, " n_points"], Float64[Array, "n_points 3"]]
        Cumulative coordinate and matching Cartesian three-vectors.
    """
    k_axis: Float64[Array, " n_points"] = jnp.linspace(0.0, 1.0, n_points)
    kpoints: Float64[Array, "n_points 3"] = jnp.stack(
        (k_axis, jnp.zeros_like(k_axis), jnp.zeros_like(k_axis)),
        axis=1,
    )
    path: Tuple[Float64[Array, " n_points"], Float64[Array, "n_points 3"]] = (
        k_axis,
        kpoints,
    )
    return path


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
        The test constructs the spectrum through its public factory and checks
        both
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


class TestMakeArpesSpectrum:
    """Validate :func:`~diffpes.types.make_arpes_spectrum`.

    The factory must preserve intensity dimensions and reject non-increasing
    energy coordinates.

    :see: :func:`~diffpes.types.make_arpes_spectrum`
    """

    def test_constructs_spectrum_shapes(self) -> None:
        """Construct ten momentum points over 100 energy bins.

        The check verifies the two-dimensional intensity and one-dimensional
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
        intensity: Float64[Array, "2 3"] = jnp.ones((2, 3))
        energy: Float64[Array, " 3"] = jnp.array([-1.0, 0.0, 1.0])
        k_axis: Float64[Array, " 2"] = jnp.array([0.0, 1.0])
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
    kx_axis: Float64[Array, " 3"] = jnp.array([-1.0, 0.0, 2.0])
    ky_axis: Float64[Array, " 2"] = jnp.array([-2.0, 1.0])
    energy_axis: Float64[Array, " 4"] = jnp.array([-1.0, 0.0, 2.0, 4.0])
    kx_grid: Float64[Array, "3 2 4"]
    ky_grid: Float64[Array, "3 2 4"]
    energy_grid: Float64[Array, "3 2 4"]
    kx_grid, ky_grid, energy_grid = jnp.meshgrid(
        kx_axis,
        ky_axis,
        energy_axis,
        indexing="ij",
    )
    intensity: Float64[Array, "3 2 4"] = (
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
        points: Float64[Array, "4 3"] = jnp.column_stack(
            (
                jnp.full_like(cube.energy_axis, 0.4),
                jnp.full_like(cube.energy_axis, -0.25),
                cube.energy_axis,
            )
        )
        compiled: Float64[Array, " 4"] = eqx.filter_jit(slice_edc)(
            cube, 0.4, -0.25
        )
        vectorized: Float64[Array, "3 4"] = jax.vmap(
            lambda kx: slice_edc(cube, kx, -0.25)
        )(jnp.array([-0.5, 0.4, 1.5]))
        gradient: Float64[Array, ""] = jax.grad(
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

        def query_loss(queries: Float64[Array, " 3"]) -> Float64[Array, ""]:
            """Return a joint EDC and MDC query loss."""
            edc: Float64[Array, " 4"] = slice_edc(cube, queries[0], queries[1])
            mdc: Float64[Array, "3 2"] = slice_mdc(cube, queries[2])
            result: Float64[Array, ""] = jnp.sum(edc) + jnp.sum(mdc)
            return result

        def intensity_loss(
            intensity: Float64[Array, "3 2 4"],
        ) -> Float64[Array, ""]:
            """Return an EDC sum after replacing source intensity."""
            candidate: ArpesCube = eqx.tree_at(
                lambda carrier: carrier.intensity,
                cube,
                intensity,
            )
            result: Float64[Array, ""] = jnp.sum(
                slice_edc(candidate, 0.4, -0.25)
            )
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
        kx_grid: Float64[Array, "3 2"]
        ky_grid: Float64[Array, "3 2"]
        kx_grid, ky_grid = jnp.meshgrid(
            cube.kx_axis,
            cube.ky_axis,
            indexing="ij",
        )
        points: Float64[Array, "6 3"] = jnp.column_stack(
            (
                jnp.ravel(kx_grid),
                jnp.ravel(ky_grid),
                jnp.full(kx_grid.size, 0.75),
            )
        )
        gradient: Float64[Array, ""] = jax.grad(
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
        desired: Float64[Array, "3 2"] = jnp.mean(
            cube.intensity[..., :2], axis=-1
        )

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
        desired: Float64[Array, "3 2"] = constant_energy_map(cube, 0.0, 1.0)

        chex.assert_trees_all_equal(fermi_surface_map(cube, 1.0), desired)
