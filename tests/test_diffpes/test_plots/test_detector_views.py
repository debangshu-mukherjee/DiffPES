"""Validate detector raster plotting utilities.

The tests cover carrier and raw-block inputs, physical and bin-index
extents, energy sums, logarithmic scaling, channel selection, energy
cuts, comparison panels, and standardized Poisson residuals.
"""

import chex
import jax.numpy as jnp
import matplotlib
import numpy as np
import pytest
from beartype.typing import List, Tuple
from jaxtyping import Array, Float64
from numpy.typing import NDArray

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.image import AxesImage

from diffpes.plots.detector_views import (
    plot_detector_comparison,
    plot_detector_energy_cut,
    plot_detector_image,
    plot_detector_residual,
)
from diffpes.types import DetectorRaster, make_detector_raster


def _make_raster() -> DetectorRaster:
    """PRIVATE: Build a minimal two-channel DetectorRaster for plot tests.

    Creates a valid carrier with deterministic expected counts of shape
    ``(2, 5, 4, 6)`` and strictly increasing native axes. Plot functions
    therefore receive consistent test data without reading files.

    Returns
    -------
    raster : DetectorRaster
        Carrier with counts ``arange(240)`` reshaped to ``(2, 5, 4, 6)``,
        ``u`` axis on ``[-0.2, 0.2]`` rad, ``v`` axis on ``[-0.1, 0.1]``
        rad, and energy axis on ``[-0.5, 0.0]`` eV.

    Notes
    -----
    Uses ``make_detector_raster`` with the ``"hemispherical_angles"``
    coordinate system and channel labels ``("up", "down")``.
    """
    expected_counts: Float64[Array, "2 5 4 6"] = jnp.arange(
        240, dtype=jnp.float64
    ).reshape(2, 5, 4, 6)
    raster: DetectorRaster = make_detector_raster(
        expected_counts=expected_counts,
        detector_u_axis=jnp.linspace(-0.2, 0.2, 5),
        detector_v_axis=jnp.linspace(-0.1, 0.1, 4),
        energy_axis=jnp.linspace(-0.5, 0.0, 6),
        channel_labels=("up", "down"),
        coordinate_system="hemispherical_angles",
    )
    return raster


def _raw_block() -> Float64[NDArray, "5 4 6"]:
    """PRIVATE: Build the raw counts block that matches raster channel 0.

    Returns
    -------
    block : Float64[NDArray, "5 4 6"]
        NumPy counts block ``arange(120)`` reshaped to ``(5, 4, 6)``.

    Notes
    -----
    The values equal ``_make_raster().expected_counts[0]``, so carrier
    and raw inputs give identical images in cross-checks.
    """
    block: Float64[NDArray, "5 4 6"] = np.arange(
        120, dtype=np.float64
    ).reshape(5, 4, 6)
    return block


class TestPlotDetectorImage(chex.TestCase):
    """Validate :func:`diffpes.plots.plot_detector_image`.

    Covers the energy sum, physical and bin-index extents, logarithmic
    scaling, channel selection, and the bad-channel rejection.

    :see: :func:`~diffpes.plots.plot_detector_image`
    """

    def test_carrier_energy_sum_and_extent(self) -> None:
        """Sum a carrier over energy and use the physical extent.

        The image array equals ``counts.sum(axis=-1).T`` for channel 0.
        The extent spans the first and last ``u`` and ``v`` bin centres.
        The default labels name the detector axes in radians.

        Notes
        -----
        Builds the two-channel raster, plots without a colorbar, and
        compares the image array, extent, and labels. Closes the figure.
        """
        raster: DetectorRaster
        fig: Figure
        ax: Axes
        image: AxesImage
        expected_image: Float64[NDArray, "5 4"]
        extent: Tuple[float, float, float, float]

        raster = _make_raster()
        fig, ax, image = plot_detector_image(raster, colorbar=False)
        expected_image = np.asarray(raster.expected_counts[0]).sum(axis=-1)
        chex.assert_trees_all_close(
            np.asarray(image.get_array()), expected_image.T
        )
        extent = tuple(float(value) for value in image.get_extent())
        chex.assert_equal(extent, (-0.2, 0.2, -0.1, 0.1))
        chex.assert_equal(ax.get_xlabel(), "detector u (rad)")
        chex.assert_equal(ax.get_ylabel(), "detector v (rad)")
        plt.close(fig)

    def test_raw_block_uses_bin_indices(self) -> None:
        """Plot a raw block on bin indices without a physical extent.

        A raw block carries no axes, so the image keeps the default
        pixel extent and the labels name detector bins.

        Notes
        -----
        Plots the raw ``(5, 4, 6)`` block and compares the extent with
        the Matplotlib pixel default ``(-0.5, 4.5, -0.5, 3.5)``. Closes
        the figure.
        """
        fig: Figure
        ax: Axes
        image: AxesImage
        extent: Tuple[float, float, float, float]

        fig, ax, image = plot_detector_image(_raw_block(), colorbar=False)
        extent = tuple(float(value) for value in image.get_extent())
        chex.assert_equal(extent, (-0.5, 4.5, -0.5, 3.5))
        chex.assert_equal(ax.get_xlabel(), "detector u bin")
        chex.assert_equal(ax.get_ylabel(), "detector v bin")
        plt.close(fig)

    def test_log_counts_after_energy_sum(self) -> None:
        """Apply the logarithm after the energy sum.

        The image array equals ``np.log1p(counts.sum(axis=-1)).T``, not
        the sum of per-bin logarithms.

        Notes
        -----
        Plots the raster with ``log_counts=True`` and compares the image
        array with the direct NumPy computation. Closes the figure.
        """
        raster: DetectorRaster
        fig: Figure
        image: AxesImage
        expected_image: Float64[NDArray, "5 4"]

        raster = _make_raster()
        fig, _, image = plot_detector_image(
            raster, log_counts=True, colorbar=False
        )
        expected_image = np.log1p(
            np.asarray(raster.expected_counts[0]).sum(axis=-1)
        )
        chex.assert_trees_all_close(
            np.asarray(image.get_array()), expected_image.T
        )
        plt.close(fig)

    def test_channel_selection(self) -> None:
        """Select the requested polarization channel of the carrier.

        With ``channel=1`` the image array equals the energy sum of the
        second channel, not the first.

        Notes
        -----
        Plots channel 1 of the two-channel raster and compares the image
        array with ``counts[1].sum(axis=-1).T``. Closes the figure.
        """
        raster: DetectorRaster
        fig: Figure
        image: AxesImage
        expected_image: Float64[NDArray, "5 4"]

        raster = _make_raster()
        fig, _, image = plot_detector_image(raster, channel=1, colorbar=False)
        expected_image = np.asarray(raster.expected_counts[1]).sum(axis=-1)
        chex.assert_trees_all_close(
            np.asarray(image.get_array()), expected_image.T
        )
        plt.close(fig)

    def test_rejects_bad_channel(self) -> None:
        """Reject a channel index outside the carrier channel axis.

        A channel index beyond the two available channels raises a
        ``ValueError`` before any figure exists.

        Notes
        -----
        Calls the plot with ``channel=7`` on the two-channel raster and
        expects a ``ValueError`` that names the channel index.
        """
        raster: DetectorRaster = _make_raster()
        with pytest.raises(ValueError, match="Channel index"):
            plot_detector_image(raster, channel=7)


class TestPlotDetectorEnergyCut(chex.TestCase):
    """Validate :func:`diffpes.plots.plot_detector_energy_cut`.

    Covers the ``u`` and ``v`` cut orientations, the central and
    nearest-index slice selection, bin-index axes for raw blocks, and
    the colorbar label that follows ``log_counts``.

    :see: :func:`~diffpes.plots.plot_detector_energy_cut`
    """

    def test_default_u_cut_central_index(self) -> None:
        """Plot the cut at the central ``v`` index with energy vertical.

        The default cut keeps ``u`` and slices ``v`` at its central
        index 2 of 4. The image array equals
        ``np.log1p(counts[:, 2, :]).T`` and the extent spans the ``u``
        and energy axes.

        Notes
        -----
        Plots the raster with defaults (``cut_axis="u"``, logarithmic
        counts) and compares the image array, shape ``(6, 5)``, and
        extent. Closes the figure.
        """
        raster: DetectorRaster
        fig: Figure
        ax: Axes
        image: AxesImage
        expected_cut: Float64[NDArray, "5 6"]
        extent: Tuple[float, float, float, float]

        raster = _make_raster()
        fig, ax, image = plot_detector_energy_cut(raster, colorbar=False)
        expected_cut = np.log1p(np.asarray(raster.expected_counts[0])[:, 2, :])
        chex.assert_equal(image.get_array().shape, (6, 5))
        chex.assert_trees_all_close(
            np.asarray(image.get_array()), expected_cut.T
        )
        extent = tuple(float(value) for value in image.get_extent())
        chex.assert_equal(extent, (-0.2, 0.2, -0.5, 0.0))
        chex.assert_equal(ax.get_xlabel(), "detector u (rad)")
        chex.assert_equal(ax.get_ylabel(), "energy (eV)")
        plt.close(fig)

    def test_v_cut_shape_and_extent(self) -> None:
        """Plot the cut at the central ``u`` index when ``v`` survives.

        With ``cut_axis="v"`` the image array equals
        ``counts[2, :, :].T`` of shape ``(6, 4)`` and the extent spans
        the ``v`` and energy axes.

        Notes
        -----
        Plots the raster with ``cut_axis="v"`` and ``log_counts=False``
        and compares the image array, shape, and extent. Closes the
        figure.
        """
        raster: DetectorRaster
        fig: Figure
        ax: Axes
        image: AxesImage
        expected_cut: Float64[NDArray, "4 6"]
        extent: Tuple[float, float, float, float]

        raster = _make_raster()
        fig, ax, image = plot_detector_energy_cut(
            raster, cut_axis="v", log_counts=False, colorbar=False
        )
        expected_cut = np.asarray(raster.expected_counts[0])[2, :, :]
        chex.assert_equal(image.get_array().shape, (6, 4))
        chex.assert_trees_all_close(
            np.asarray(image.get_array()), expected_cut.T
        )
        extent = tuple(float(value) for value in image.get_extent())
        chex.assert_equal(extent, (-0.1, 0.1, -0.5, 0.0))
        chex.assert_equal(ax.get_xlabel(), "detector v (rad)")
        plt.close(fig)

    def test_position_selects_nearest_index(self) -> None:
        """Select the bin centre nearest to the physical position.

        The ``v`` axis holds the centres ``[-0.1, -1/30, 1/30, 0.1]``
        rad, so ``position=0.09`` selects index 3.

        Notes
        -----
        Plots the raster with ``position=0.09`` and ``log_counts=False``
        and compares the image array with ``counts[:, 3, :].T``. Closes
        the figure.
        """
        raster: DetectorRaster
        fig: Figure
        image: AxesImage
        expected_cut: Float64[NDArray, "5 6"]

        raster = _make_raster()
        fig, _, image = plot_detector_energy_cut(
            raster, position=0.09, log_counts=False, colorbar=False
        )
        expected_cut = np.asarray(raster.expected_counts[0])[:, 3, :]
        chex.assert_trees_all_close(
            np.asarray(image.get_array()), expected_cut.T
        )
        plt.close(fig)

    def test_raw_block_uses_bin_indices(self) -> None:
        """Plot a raw block on bin indices without a physical extent.

        A raw block carries no axes, so the cut keeps the default pixel
        extent ``(-0.5, 4.5, -0.5, 5.5)`` and the labels name bins.

        Notes
        -----
        Plots the raw ``(5, 4, 6)`` block with ``log_counts=False`` and
        compares the extent and both axis labels. Closes the figure.
        """
        fig: Figure
        ax: Axes
        image: AxesImage
        extent: Tuple[float, float, float, float]

        fig, ax, image = plot_detector_energy_cut(
            _raw_block(), log_counts=False, colorbar=False
        )
        extent = tuple(float(value) for value in image.get_extent())
        chex.assert_equal(extent, (-0.5, 4.5, -0.5, 5.5))
        chex.assert_equal(ax.get_xlabel(), "detector u bin")
        chex.assert_equal(ax.get_ylabel(), "energy bin")
        plt.close(fig)

    def test_colorbar_label_follows_log_counts(self) -> None:
        """Resolve the default colorbar label from ``log_counts``.

        The default label reads ``"log(1 + counts)"`` with logarithmic
        counts and ``"expected counts"`` with linear counts.

        Notes
        -----
        Plots the raster twice with a colorbar, once per ``log_counts``
        value, and reads each label from the colorbar axis. Closes both
        figures.
        """
        raster: DetectorRaster
        log_fig: Figure
        log_image: AxesImage
        linear_fig: Figure
        linear_image: AxesImage

        raster = _make_raster()
        log_fig, _, log_image = plot_detector_energy_cut(raster)
        chex.assert_equal(
            log_image.colorbar.ax.get_ylabel(), "log(1 + counts)"
        )
        plt.close(log_fig)
        linear_fig, _, linear_image = plot_detector_energy_cut(
            raster, log_counts=False
        )
        chex.assert_equal(
            linear_image.colorbar.ax.get_ylabel(), "expected counts"
        )
        plt.close(linear_fig)


class TestPlotDetectorComparison(chex.TestCase):
    """Validate :func:`diffpes.plots.plot_detector_comparison`.

    Covers the two-panel layout, the shared color scale, the energy
    view, and the rejection of mismatched count-block shapes.

    :see: :func:`~diffpes.plots.plot_detector_comparison`
    """

    def test_two_panels_share_scale(self) -> None:
        """Render two titled panels with one shared color scale.

        The comparison of the raster with an offset raw block yields
        two images. Their shared color limits span the joint value
        range of both energy-summed panels.

        Notes
        -----
        Compares the raster channel 0 with ``raw + 3.0`` counts. Checks
        the panel count, titles, and identical color limits equal to
        the joint minimum and maximum. Closes the figure.
        """
        raster: DetectorRaster
        observed: Float64[NDArray, "5 4 6"]
        fig: Figure
        axes_pair: Tuple[Axes, Axes]
        images: List[AxesImage]
        expected_sum: Float64[NDArray, "5 4"]
        observed_sum: Float64[NDArray, "5 4"]

        raster = _make_raster()
        observed = _raw_block() + 3.0
        fig, axes_pair, images = plot_detector_comparison(
            raster, observed, colorbar=False
        )
        chex.assert_equal(len(axes_pair), 2)
        chex.assert_equal(len(images), 2)
        chex.assert_equal(axes_pair[0].get_title(), "expected")
        chex.assert_equal(axes_pair[1].get_title(), "observed")
        expected_sum = np.asarray(raster.expected_counts[0]).sum(axis=-1)
        observed_sum = observed.sum(axis=-1)
        chex.assert_equal(images[0].get_clim(), images[1].get_clim())
        chex.assert_trees_all_close(
            np.asarray(images[0].get_clim()),
            np.asarray((float(expected_sum.min()), float(observed_sum.max()))),
        )
        plt.close(fig)

    def test_energy_view_central_cut(self) -> None:
        """Show central-``v`` energy cuts in the energy view.

        With ``view="energy"`` each panel holds the energy-versus-``u``
        cut at the central ``v`` index 2, so the image arrays have shape
        ``(6, 5)``.

        Notes
        -----
        Compares the observed panel array with ``observed[:, 2, :].T``
        computed directly in NumPy. Closes the figure.
        """
        raster: DetectorRaster
        observed: Float64[NDArray, "5 4 6"]
        fig: Figure
        images: List[AxesImage]

        raster = _make_raster()
        observed = _raw_block() + 1.0
        fig, _, images = plot_detector_comparison(
            raster, observed, view="energy", colorbar=False
        )
        chex.assert_equal(images[0].get_array().shape, (6, 5))
        chex.assert_trees_all_close(
            np.asarray(images[1].get_array()), observed[:, 2, :].T
        )
        plt.close(fig)

    def test_rejects_shape_mismatch(self) -> None:
        """Reject count blocks that disagree in shape.

        The carrier block has shape ``(5, 4, 6)`` and the observed block
        has shape ``(4, 4, 6)``, so the static check raises a
        ``ValueError`` before any figure exists.

        Notes
        -----
        Expects a ``ValueError`` whose message states that the blocks
        disagree in shape.
        """
        raster: DetectorRaster = _make_raster()
        observed: Float64[NDArray, "4 4 6"] = np.ones(
            (4, 4, 6), dtype=np.float64
        )
        with pytest.raises(ValueError, match="disagree in shape"):
            plot_detector_comparison(raster, observed)


class TestPlotDetectorResidual(chex.TestCase):
    """Validate :func:`diffpes.plots.plot_detector_residual`.

    Covers the standardized Poisson residual formula, the symmetric
    color limits with their zero-residual fallback, and the rejection
    of mismatched count-block shapes.

    :see: :func:`~diffpes.plots.plot_detector_residual`
    """

    def test_residual_formula_and_symmetric_clim(self) -> None:
        """Normalize the energy-summed difference with Poisson noise.

        The image array equals
        ``(obs_sum - exp_sum) / sqrt(maximum(exp_sum, 1.0))`` transposed.
        The color limits are symmetric at the maximum absolute residual.

        Notes
        -----
        Uses ``observed = raw + 2.0`` so every angular bin gains 12
        counts. Compares the image with the direct NumPy residual.
        Checks ``clim == (-scale, scale)``. Closes the figure.
        """
        raster: DetectorRaster
        observed: Float64[NDArray, "5 4 6"]
        fig: Figure
        image: AxesImage
        expected_sum: Float64[NDArray, "5 4"]
        observed_sum: Float64[NDArray, "5 4"]
        residual: Float64[NDArray, "5 4"]
        scale: float

        raster = _make_raster()
        observed = _raw_block() + 2.0
        fig, _, image = plot_detector_residual(
            raster, observed, colorbar=False
        )
        expected_sum = np.asarray(raster.expected_counts[0]).sum(axis=-1)
        observed_sum = observed.sum(axis=-1)
        residual = (observed_sum - expected_sum) / np.sqrt(
            np.maximum(expected_sum, 1.0)
        )
        chex.assert_trees_all_close(np.asarray(image.get_array()), residual.T)
        scale = float(np.max(np.abs(residual)))
        chex.assert_trees_all_close(
            np.asarray(image.get_clim()), np.asarray((-scale, scale))
        )
        plt.close(fig)

    def test_zero_residual_fallback_clim(self) -> None:
        """Use the unit color-limit fallback for a zero residual.

        Identical expected and observed counts give an identically zero
        residual, and the color limits fall back to ``(-1.0, 1.0)``.

        Notes
        -----
        Compares the raster channel 0 with the equal raw block and
        checks the fallback color limits. Closes the figure.
        """
        raster: DetectorRaster
        fig: Figure
        image: AxesImage

        raster = _make_raster()
        fig, _, image = plot_detector_residual(
            raster, _raw_block(), colorbar=False
        )
        chex.assert_equal(image.get_clim(), (-1.0, 1.0))
        plt.close(fig)

    def test_rejects_shape_mismatch(self) -> None:
        """Reject count blocks that disagree in shape.

        The carrier block has shape ``(5, 4, 6)`` and the observed block
        has shape ``(5, 4, 5)``, so the static check raises a
        ``ValueError`` before any figure exists.

        Notes
        -----
        Expects a ``ValueError`` whose message states that the blocks
        disagree in shape.
        """
        raster: DetectorRaster = _make_raster()
        observed: Float64[NDArray, "5 4 5"] = np.ones(
            (5, 4, 5), dtype=np.float64
        )
        with pytest.raises(ValueError, match="disagree in shape"):
            plot_detector_residual(raster, observed)
