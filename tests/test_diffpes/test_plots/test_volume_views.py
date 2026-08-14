"""Validate three-dimensional ARPES cube and band-surface plotting.

The tests cover point-cloud thresholds, jitter determinism, floor-slice
contours, zero-intensity rejection, outer-face rendering, band-surface
counts, mesh-size validation, and reuse of caller-created axes.
"""

import chex
import jax.numpy as jnp
import matplotlib
import numpy as np
import pytest
from beartype import beartype
from beartype.typing import List, Tuple, Union
from jaxtyping import Array, Float64, jaxtyped
from numpy.typing import NDArray

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.figure import Figure, SubFigure
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Path3DCollection, Poly3DCollection

from diffpes.plots.volume_views import (
    plot_band_surface,
    plot_cube_faces,
    plot_cube_scatter,
)
from diffpes.types import ArpesCube, make_arpes_cube


@beartype
def _make_cube(n_kx: int = 6, n_ky: int = 6, n_e: int = 5) -> ArpesCube:
    """PRIVATE: Build a small ArpesCube with one smooth intensity peak.

    Parameters
    ----------
    n_kx : int, optional
        Number of ``k_x`` samples. Default 6.
    n_ky : int, optional
        Number of ``k_y`` samples. Default 6.
    n_e : int, optional
        Number of energy samples. Default 5.

    Returns
    -------
    cube : ArpesCube
        Validated cube with a Gaussian intensity peak near the grid
        center.

    Notes
    -----
    Builds strictly increasing axes with ``jnp.linspace`` and one
    separable Gaussian intensity, so thresholded voxel counts are
    deterministic.
    """
    kx_axis: Float64[Array, " n_kx"] = jnp.linspace(-0.5, 0.5, n_kx)
    ky_axis: Float64[Array, " n_ky"] = jnp.linspace(-0.4, 0.4, n_ky)
    energy_axis: Float64[Array, " n_e"] = jnp.linspace(-1.0, 0.2, n_e)
    kx_mesh: Float64[Array, "n_kx n_ky n_e"]
    ky_mesh: Float64[Array, "n_kx n_ky n_e"]
    energy_mesh: Float64[Array, "n_kx n_ky n_e"]
    kx_mesh, ky_mesh, energy_mesh = jnp.meshgrid(
        kx_axis, ky_axis, energy_axis, indexing="ij"
    )
    intensity: Float64[Array, "n_kx n_ky n_e"] = jnp.exp(
        -(
            (kx_mesh / 0.35) ** 2
            + (ky_mesh / 0.3) ** 2
            + ((energy_mesh + 0.4) / 0.5) ** 2
        )
    )
    cube: ArpesCube = make_arpes_cube(
        intensity=intensity,
        kx_axis=kx_axis,
        ky_axis=ky_axis,
        energy_axis=energy_axis,
    )
    return cube


@beartype
def _make_zero_cube() -> ArpesCube:
    """PRIVATE: Build a small ArpesCube whose intensity is all zero.

    Returns
    -------
    cube : ArpesCube
        Validated cube with zero intensity on a three-point grid.

    Notes
    -----
    Zero intensity passes the nonnegativity check of
    ``make_arpes_cube`` but leaves nothing to normalize, so
    ``plot_cube_scatter`` must reject the cube.
    """
    axis: Float64[Array, " 3"] = jnp.linspace(-0.1, 0.1, 3)
    energy_axis: Float64[Array, " 3"] = jnp.linspace(-1.0, 0.0, 3)
    cube: ArpesCube = make_arpes_cube(
        intensity=jnp.zeros((3, 3, 3), dtype=jnp.float64),
        kx_axis=axis,
        ky_axis=axis,
        energy_axis=energy_axis,
    )
    return cube


@jaxtyped(typechecker=beartype)
def _scatter_offsets(
    scatter: Path3DCollection,
) -> Tuple[
    Float64[NDArray, " n_points"],
    Float64[NDArray, " n_points"],
    Float64[NDArray, " n_points"],
]:
    """PRIVATE: Read the three-dimensional offsets of a scatter artist.

    Parameters
    ----------
    scatter : Path3DCollection
        Scatter artist returned by ``plot_cube_scatter``.

    Returns
    -------
    xs : Float64[NDArray, " n_points"]
        Scatter x coordinates in inverse angstroms.
    ys : Float64[NDArray, " n_points"]
        Scatter y coordinates in inverse angstroms.
    zs : Float64[NDArray, " n_points"]
        Scatter z coordinates in eV.

    Notes
    -----
    Reads the private ``_offsets3d`` storage because Matplotlib exposes
    no public accessor for three-dimensional scatter offsets.
    """
    xs: Float64[NDArray, " n_points"] = np.asarray(
        scatter._offsets3d[0],  # noqa: SLF001
        dtype=np.float64,
    )
    ys: Float64[NDArray, " n_points"] = np.asarray(
        scatter._offsets3d[1],  # noqa: SLF001
        dtype=np.float64,
    )
    zs: Float64[NDArray, " n_points"] = np.asarray(
        scatter._offsets3d[2],  # noqa: SLF001
        dtype=np.float64,
    )
    offsets: Tuple[
        Float64[NDArray, " n_points"],
        Float64[NDArray, " n_points"],
        Float64[NDArray, " n_points"],
    ] = (xs, ys, zs)
    return offsets


@beartype
def _make_band_inputs(
    n_kx: int = 4, n_ky: int = 3
) -> Tuple[
    Float64[Array, "nk 2"],
    Float64[Array, " nkx"],
    Float64[Array, " nky"],
]:
    """PRIVATE: Build two paraboloid bands on a flattened Cartesian mesh.

    Parameters
    ----------
    n_kx : int, optional
        Number of ``k_x`` samples. Default 4.
    n_ky : int, optional
        Number of ``k_y`` samples. Default 3.

    Returns
    -------
    eigenvalues : Float64[Array, "nk 2"]
        Two bands in eV on the flattened mesh with ``indexing="ij"``
        flattening.
    kx_axis : Float64[Array, " nkx"]
        Cartesian ``k_x`` axis in inverse angstroms.
    ky_axis : Float64[Array, " nky"]
        Cartesian ``k_y`` axis in inverse angstroms.

    Notes
    -----
    Builds one electron-like and one hole-like paraboloid, flattens each
    in row-major order, and stacks the bands as columns.
    """
    kx_axis: Float64[Array, " nkx"] = jnp.linspace(-0.3, 0.3, n_kx)
    ky_axis: Float64[Array, " nky"] = jnp.linspace(-0.2, 0.2, n_ky)
    kx_mesh: Float64[Array, "nkx nky"]
    ky_mesh: Float64[Array, "nkx nky"]
    kx_mesh, ky_mesh = jnp.meshgrid(kx_axis, ky_axis, indexing="ij")
    lower_band: Float64[Array, "nkx nky"] = -1.0 + kx_mesh**2 + ky_mesh**2
    upper_band: Float64[Array, "nkx nky"] = 0.5 - kx_mesh**2 - ky_mesh**2
    eigenvalues: Float64[Array, "nk 2"] = jnp.stack(
        (lower_band.reshape(-1), upper_band.reshape(-1)), axis=1
    )
    band_inputs: Tuple[
        Float64[Array, "nk 2"],
        Float64[Array, " nkx"],
        Float64[Array, " nky"],
    ] = (eigenvalues, kx_axis, ky_axis)
    return band_inputs


class TestPlotCubeScatter(chex.TestCase):
    """Validate :func:`diffpes.plots.plot_cube_scatter`.

    Covers artist and axis types, the intensity floor, jitter
    determinism, grid-node positions without jitter, the floor-slice
    contour map, zero-intensity rejection, and axis reuse.

    :see: :func:`~diffpes.plots.plot_cube_scatter`
    """

    def test_returns_expected_objects(self) -> None:
        """Return a labelled three-dimensional scatter plot.

        The returned axis is an ``Axes3D`` and the artist is a
        ``Path3DCollection``. The default labels name the momentum axes
        and the relative energy.

        Notes
        -----
        The test builds a 6x6x5 cube and calls ``plot_cube_scatter``
        with the defaults. It checks the axis and artist types with
        ``isinstance`` and compares the three axis labels with the
        documented defaults.
        """
        cube: ArpesCube
        fig: Union[Figure, SubFigure]
        ax: Axes3D
        scatter: Path3DCollection

        cube = _make_cube()
        fig, ax, scatter = plot_cube_scatter(cube)
        assert isinstance(ax, Axes3D)
        assert isinstance(scatter, Path3DCollection)
        chex.assert_equal(ax.get_xlabel(), r"$k_x$ ($\mathrm{\AA}^{-1}$)")
        chex.assert_equal(ax.get_ylabel(), r"$k_y$ ($\mathrm{\AA}^{-1}$)")
        chex.assert_equal(ax.get_zlabel(), r"$E - E_F$ (eV)")
        plt.close(fig)

    def test_intensity_floor_controls_point_count(self) -> None:
        """Check that the intensity floor selects the scattered voxel count.

        A scatter with a low floor carries every voxel above that floor,
        and a higher floor keeps strictly fewer voxels. Both counts
        equal an independent NumPy count of the thresholded normalized
        intensity.

        Notes
        -----
        The test normalizes the cube intensity with NumPy and counts
        the voxels above floors 0.12 and 0.5. It compares each count
        with the scatter offset length from the matching call.
        """
        cube: ArpesCube
        normalized: Float64[NDArray, "6 6 5"]
        expected_low: int
        expected_high: int
        fig_low: Union[Figure, SubFigure]
        fig_high: Union[Figure, SubFigure]
        scatter_low: Path3DCollection
        scatter_high: Path3DCollection

        cube = _make_cube()
        normalized = np.asarray(cube.intensity, dtype=np.float64)
        normalized = normalized / float(np.max(normalized))
        expected_low = int(np.sum(normalized > 0.12))
        expected_high = int(np.sum(normalized > 0.5))
        fig_low, _, scatter_low = plot_cube_scatter(cube, intensity_floor=0.12)
        fig_high, _, scatter_high = plot_cube_scatter(
            cube, intensity_floor=0.5
        )
        chex.assert_equal(
            _scatter_offsets(scatter_low)[0].shape[0], expected_low
        )
        chex.assert_equal(
            _scatter_offsets(scatter_high)[0].shape[0], expected_high
        )
        assert expected_high < expected_low
        plt.close(fig_low)
        plt.close(fig_high)

    def test_jitter_seed_determinism(self) -> None:
        """Verify that equal seeds give equal jitter and other seeds differ.

        The jitter offsets come from ``np.random.default_rng(seed)``,
        so two calls with one seed produce identical point positions,
        and a different seed produces different positions.

        Notes
        -----
        The test calls ``plot_cube_scatter`` twice with ``seed=3`` and
        once with ``seed=4``. It compares the offset triples with
        ``chex.assert_trees_all_equal`` and asserts that the third call
        disagrees with the first through ``np.allclose``.
        """
        cube: ArpesCube
        fig_a: Union[Figure, SubFigure]
        fig_b: Union[Figure, SubFigure]
        fig_c: Union[Figure, SubFigure]
        scatter_a: Path3DCollection
        scatter_b: Path3DCollection
        scatter_c: Path3DCollection

        cube = _make_cube()
        fig_a, _, scatter_a = plot_cube_scatter(cube, seed=3)
        fig_b, _, scatter_b = plot_cube_scatter(cube, seed=3)
        fig_c, _, scatter_c = plot_cube_scatter(cube, seed=4)
        chex.assert_trees_all_equal(
            _scatter_offsets(scatter_a), _scatter_offsets(scatter_b)
        )
        assert not np.allclose(
            _scatter_offsets(scatter_a)[0], _scatter_offsets(scatter_c)[0]
        )
        plt.close(fig_a)
        plt.close(fig_b)
        plt.close(fig_c)

    def test_no_jitter_keeps_grid_nodes(self) -> None:
        """Check that without jitter every point sits on a grid node.

        With ``jitter=False`` each scatter coordinate equals one axis
        value of the cube, so every point sits exactly on the momentum
        and energy raster.

        Notes
        -----
        The test calls ``plot_cube_scatter`` with ``jitter=False`` and
        checks each offset array against the matching cube axis with
        ``np.isin``.
        """
        cube: ArpesCube
        fig: Union[Figure, SubFigure]
        scatter: Path3DCollection
        xs: Float64[NDArray, " n_points"]
        ys: Float64[NDArray, " n_points"]
        zs: Float64[NDArray, " n_points"]

        cube = _make_cube()
        fig, _, scatter = plot_cube_scatter(cube, jitter=False)
        xs, ys, zs = _scatter_offsets(scatter)
        assert bool(np.all(np.isin(xs, np.asarray(cube.kx_axis))))
        assert bool(np.all(np.isin(ys, np.asarray(cube.ky_axis))))
        assert bool(np.all(np.isin(zs, np.asarray(cube.energy_axis))))
        plt.close(fig)

    def test_floor_slice_adds_contours(self) -> None:
        """Verify that a floor slice index adds contour collections.

        With ``floor_slice_index`` the axis carries the scatter artist
        plus at least one filled-contour collection. The collection
        count therefore exceeds the count of a plain scatter call.

        Notes
        -----
        The test renders the cube once without and once with
        ``floor_slice_index=2``. It compares ``len(ax.collections)``
        between the two axes.
        """
        cube: ArpesCube
        fig_plain: Union[Figure, SubFigure]
        fig_floor: Union[Figure, SubFigure]
        ax_plain: Axes3D
        ax_floor: Axes3D

        cube = _make_cube()
        fig_plain, ax_plain, _ = plot_cube_scatter(cube)
        fig_floor, ax_floor, _ = plot_cube_scatter(cube, floor_slice_index=2)
        assert len(ax_floor.collections) > len(ax_plain.collections)
        plt.close(fig_plain)
        plt.close(fig_floor)

    def test_zero_cube_raises(self) -> None:
        """Reject a cube whose intensity maximum is zero.

        A zero maximum leaves nothing to normalize, so the function
        raises a ``ValueError`` before it creates any artist.

        Notes
        -----
        The test builds an all-zero cube and expects a ``ValueError``
        whose message states that the cube intensity maximum is zero.
        """
        cube: ArpesCube = _make_zero_cube()
        with pytest.raises(ValueError, match="maximum is zero"):
            plot_cube_scatter(cube)

    def test_reuses_supplied_axis(self) -> None:
        """Render the point cloud on a caller-created 3D axis.

        The function keeps the supplied ``Axes3D`` and its parent
        figure instead of creating new ones.

        Notes
        -----
        The test creates a figure with one ``projection="3d"`` subplot,
        passes the axis to ``plot_cube_scatter``, and verifies the
        identities of the returned figure and axis.
        """
        cube: ArpesCube
        fig: Figure
        ax: Axes3D
        out_fig: Union[Figure, SubFigure]
        out_ax: Axes3D

        cube = _make_cube()
        fig = plt.figure()
        ax = fig.add_subplot(projection="3d")
        out_fig, out_ax, _ = plot_cube_scatter(cube, ax=ax)
        assert out_fig is fig
        assert out_ax is ax
        plt.close(fig)


class TestPlotCubeFaces(chex.TestCase):
    """Validate :func:`diffpes.plots.plot_cube_faces`.

    Covers the three returned surface artists, the zero-face
    normalization fallback, and axis reuse.

    :see: :func:`~diffpes.plots.plot_cube_faces`
    """

    def test_returns_three_surfaces(self) -> None:
        """Return three opaque face surfaces on an Axes3D.

        The function draws the last-``k_x`` face, the last-``k_y``
        face, and the top-energy face, so the returned list carries
        exactly three ``Poly3DCollection`` artists.

        Notes
        -----
        The test renders a 6x6x5 cube and checks the axis type, the
        surface count, and the type of every surface artist.
        """
        cube: ArpesCube
        fig: Union[Figure, SubFigure]
        ax: Axes3D
        surfaces: List[Poly3DCollection]
        surface: Poly3DCollection

        cube = _make_cube()
        fig, ax, surfaces = plot_cube_faces(cube)
        assert isinstance(ax, Axes3D)
        chex.assert_equal(len(surfaces), 3)
        for surface in surfaces:
            assert isinstance(surface, Poly3DCollection)
        plt.close(fig)

    def test_zero_face_uses_fallback(self) -> None:
        """Render a cube whose top-energy face is all zero.

        A zero face maximum falls back to a divisor of 1.0, so the
        function still returns three surfaces without a division error.

        Notes
        -----
        The test builds a cube whose intensity lives only in the lowest
        energy slice. The top-energy face therefore holds only zeros.
        The test checks that ``plot_cube_faces`` returns three surface
        artists.
        """
        axis: Float64[Array, " 4"]
        energy_axis: Float64[Array, " 3"]
        intensity: Float64[Array, "4 4 3"]
        cube: ArpesCube
        fig: Union[Figure, SubFigure]
        surfaces: List[Poly3DCollection]

        axis = jnp.linspace(-0.2, 0.2, 4)
        energy_axis = jnp.linspace(-1.0, 0.0, 3)
        intensity = (
            jnp.zeros((4, 4, 3), dtype=jnp.float64).at[:, :, 0].set(1.0)
        )
        cube = make_arpes_cube(
            intensity=intensity,
            kx_axis=axis,
            ky_axis=axis,
            energy_axis=energy_axis,
        )
        fig, _, surfaces = plot_cube_faces(cube)
        chex.assert_equal(len(surfaces), 3)
        plt.close(fig)

    def test_reuses_supplied_axis(self) -> None:
        """Render the faces on a caller-created 3D axis.

        The function keeps the supplied ``Axes3D`` and its parent
        figure instead of creating new ones.

        Notes
        -----
        The test creates a figure with one ``projection="3d"`` subplot,
        passes the axis to ``plot_cube_faces``, and verifies the
        identities of the returned figure and axis.
        """
        cube: ArpesCube
        fig: Figure
        ax: Axes3D
        out_fig: Union[Figure, SubFigure]
        out_ax: Axes3D

        cube = _make_cube()
        fig = plt.figure()
        ax = fig.add_subplot(projection="3d")
        out_fig, out_ax, _ = plot_cube_faces(cube, ax=ax)
        assert out_fig is fig
        assert out_ax is ax
        plt.close(fig)


class TestPlotBandSurface(chex.TestCase):
    """Validate :func:`diffpes.plots.plot_band_surface`.

    Covers the surface count for all and for selected bands, the
    mesh-size validation, and axis reuse.

    :see: :func:`~diffpes.plots.plot_band_surface`
    """

    def test_surface_count_matches_all_bands(self) -> None:
        """Render one surface for every band by default.

        With ``band_indices=None`` the function draws every eigenvalue
        column, so two bands give exactly two ``Poly3DCollection``
        artists on an ``Axes3D``.

        Notes
        -----
        The test builds two paraboloid bands on a 4x3 mesh and calls
        ``plot_band_surface`` with the defaults. It checks the axis
        type, the surface count, and the type of every artist.
        """
        eigenvalues: Float64[Array, "12 2"]
        kx_axis: Float64[Array, " 4"]
        ky_axis: Float64[Array, " 3"]
        fig: Union[Figure, SubFigure]
        ax: Axes3D
        surfaces: List[Poly3DCollection]
        surface: Poly3DCollection

        eigenvalues, kx_axis, ky_axis = _make_band_inputs()
        fig, ax, surfaces = plot_band_surface(eigenvalues, kx_axis, ky_axis)
        assert isinstance(ax, Axes3D)
        chex.assert_equal(len(surfaces), 2)
        for surface in surfaces:
            assert isinstance(surface, Poly3DCollection)
        plt.close(fig)

    def test_band_indices_selects_bands(self) -> None:
        """Limit the drawn surfaces to the given band indices.

        With ``band_indices=(1,)`` the function draws only the second
        eigenvalue column, so the returned list carries one surface.

        Notes
        -----
        The test builds two paraboloid bands, calls
        ``plot_band_surface`` with ``band_indices=(1,)``, and compares
        the surface count with one.
        """
        eigenvalues: Float64[Array, "12 2"]
        kx_axis: Float64[Array, " 4"]
        ky_axis: Float64[Array, " 3"]
        fig: Union[Figure, SubFigure]
        surfaces: List[Poly3DCollection]

        eigenvalues, kx_axis, ky_axis = _make_band_inputs()
        fig, _, surfaces = plot_band_surface(
            eigenvalues, kx_axis, ky_axis, band_indices=(1,)
        )
        chex.assert_equal(len(surfaces), 1)
        plt.close(fig)

    def test_mesh_size_mismatch_raises(self) -> None:
        """Reject eigenvalues that do not cover the Cartesian mesh.

        The eigenvalue rows must equal the product of the two axis
        lengths, so a truncated eigenvalue array raises a
        ``ValueError``.

        Notes
        -----
        The test drops the last row of a valid 12-row eigenvalue array,
        so 11 rows face a 4x3 mesh. It expects a ``ValueError`` whose
        message names the required row count.
        """
        eigenvalues: Float64[Array, "12 2"]
        kx_axis: Float64[Array, " 4"]
        ky_axis: Float64[Array, " 3"]

        eigenvalues, kx_axis, ky_axis = _make_band_inputs()
        with pytest.raises(ValueError, match="eigenvalues rows must equal"):
            plot_band_surface(eigenvalues[:-1, :], kx_axis, ky_axis)

    def test_reuses_supplied_axis(self) -> None:
        """Render the band surfaces on a caller-created 3D axis.

        The function keeps the supplied ``Axes3D`` and its parent
        figure instead of creating new ones.

        Notes
        -----
        The test creates a figure with one ``projection="3d"`` subplot,
        passes the axis to ``plot_band_surface``, and verifies the
        identities of the returned figure and axis.
        """
        eigenvalues: Float64[Array, "12 2"]
        kx_axis: Float64[Array, " 4"]
        ky_axis: Float64[Array, " 3"]
        fig: Figure
        ax: Axes3D
        out_fig: Union[Figure, SubFigure]
        out_ax: Axes3D

        eigenvalues, kx_axis, ky_axis = _make_band_inputs()
        fig = plt.figure()
        ax = fig.add_subplot(projection="3d")
        out_fig, out_ax, _ = plot_band_surface(
            eigenvalues, kx_axis, ky_axis, ax=ax
        )
        assert out_fig is fig
        assert out_ax is ax
        plt.close(fig)
