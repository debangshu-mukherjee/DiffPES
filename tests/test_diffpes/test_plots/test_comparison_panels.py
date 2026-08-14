"""Validate comparison-panel plotting utilities.

The tests check panel rows for spectral cuts and momentum maps with a
shared color scale. They exercise title handling and crosshair guides.
They also check symmetric difference maps with percentile scaling and
zero-scale handling.
"""

import chex
import matplotlib
import numpy as np
import pytest
from beartype import beartype
from beartype.typing import List, Tuple
from jaxtyping import Float64, jaxtyped
from numpy.typing import NDArray

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.image import AxesImage

from diffpes.plots.comparison_panels import (
    plot_difference_map,
    plot_momentum_map_grid,
    plot_spectral_cut_series,
)


@jaxtyped(typechecker=beartype)
def _ramp_map(
    n_rows: int, n_cols: int, peak: float
) -> Float64[NDArray, "n_rows n_cols"]:
    """PRIVATE: Build one deterministic ramp map with a known maximum.

    Parameters
    ----------
    n_rows : int
        Number of rows of the map.
    n_cols : int
        Number of columns of the map.
    peak : float
        Maximum value of the ramp.

    Returns
    -------
    ramp : Float64[NDArray, "n_rows n_cols"]
        Linear ramp from ``0.0`` to ``peak`` in row-major order.

    Notes
    -----
    Uses ``np.linspace`` with a reshape, so the map minimum is ``0.0``
    and the map maximum is ``peak`` exactly.
    """
    ramp: Float64[NDArray, "n_rows n_cols"] = np.linspace(
        0.0, peak, n_rows * n_cols, dtype=np.float64
    ).reshape(n_rows, n_cols)
    return ramp


class TestPlotSpectralCutSeries(chex.TestCase):
    """Validate :func:`~diffpes.plots.plot_spectral_cut_series`.

    The tests check the panel count, container types, and shared color
    limits. They check y-axis label retention on the first panel and
    per-panel titles. They also exercise the title-count validation
    and reuse of caller-supplied axes.

    :see: :func:`~diffpes.plots.plot_spectral_cut_series`
    """

    def test_panel_count_and_return_types(self) -> None:
        """Return one axis and one image per intensity map.

        The test renders three maps and checks the returned container
        types and lengths.

        Notes
        -----
        Builds three ``(6, 8)`` ramp maps and asserts a ``tuple`` of
        three ``Axes`` and a ``list`` of three ``AxesImage`` artists.
        """
        momentum_axis: Float64[NDArray, " 6"] = np.linspace(-0.5, 0.5, 6)
        energy_axis: Float64[NDArray, " 8"] = np.linspace(-2.0, 0.5, 8)
        intensities: Tuple[Float64[NDArray, "6 8"], ...] = (
            _ramp_map(6, 8, 1.0),
            _ramp_map(6, 8, 2.0),
            _ramp_map(6, 8, 3.0),
        )
        fig: Figure
        axes_row: Tuple[Axes, ...]
        images: List[AxesImage]
        fig, axes_row, images = plot_spectral_cut_series(
            intensities, momentum_axis, energy_axis
        )
        assert isinstance(axes_row, tuple)
        assert isinstance(images, list)
        chex.assert_equal(len(axes_row), 3)
        chex.assert_equal(len(images), 3)
        panel_axis: Axes
        for panel_axis in axes_row:
            assert isinstance(panel_axis, Axes)
        image: AxesImage
        for image in images:
            assert isinstance(image, AxesImage)
        plt.close(fig)

    def test_shared_scale_equalizes_clims(self) -> None:
        """Apply one global color scale to every panel.

        The test renders maps with different maxima under
        ``share_scale=True`` and compares every image's color limits.

        Notes
        -----
        The ramps span ``[0, 1]``, ``[0, 2]``, and ``[0, 3]``, so the
        expected shared limits are ``(0.0, 3.0)`` through ``get_clim``.
        """
        momentum_axis: Float64[NDArray, " 6"] = np.linspace(-0.5, 0.5, 6)
        energy_axis: Float64[NDArray, " 8"] = np.linspace(-2.0, 0.5, 8)
        intensities: Tuple[Float64[NDArray, "6 8"], ...] = (
            _ramp_map(6, 8, 1.0),
            _ramp_map(6, 8, 2.0),
            _ramp_map(6, 8, 3.0),
        )
        fig: Figure
        images: List[AxesImage]
        fig, _, images = plot_spectral_cut_series(
            intensities, momentum_axis, energy_axis, share_scale=True
        )
        image: AxesImage
        for image in images:
            clim: Tuple[float, float] = tuple(
                float(value) for value in image.get_clim()
            )
            chex.assert_trees_all_close(clim, (0.0, 3.0))
        plt.close(fig)

    def test_first_panel_keeps_ylabel(self) -> None:
        """Keep the y-axis label only on the first panel.

        The test renders two panels and compares the y-axis labels of
        the first panel and the second panel.

        Notes
        -----
        The first panel keeps the default label ``$E - E_F$ (eV)``.
        The second panel carries the empty label.
        """
        momentum_axis: Float64[NDArray, " 6"] = np.linspace(-0.5, 0.5, 6)
        energy_axis: Float64[NDArray, " 8"] = np.linspace(-2.0, 0.5, 8)
        intensities: Tuple[Float64[NDArray, "6 8"], ...] = (
            _ramp_map(6, 8, 1.0),
            _ramp_map(6, 8, 2.0),
        )
        fig: Figure
        axes_row: Tuple[Axes, ...]
        fig, axes_row, _ = plot_spectral_cut_series(
            intensities, momentum_axis, energy_axis
        )
        chex.assert_equal(axes_row[0].get_ylabel(), r"$E - E_F$ (eV)")
        chex.assert_equal(axes_row[1].get_ylabel(), "")
        plt.close(fig)

    def test_titles_applied_per_panel(self) -> None:
        """Apply one title to each panel in order.

        The test passes two titles and compares each panel title with
        the given text.

        Notes
        -----
        Uses the titles ``("raw", "fit")`` and reads each panel title
        through ``get_title``.
        """
        momentum_axis: Float64[NDArray, " 6"] = np.linspace(-0.5, 0.5, 6)
        energy_axis: Float64[NDArray, " 8"] = np.linspace(-2.0, 0.5, 8)
        intensities: Tuple[Float64[NDArray, "6 8"], ...] = (
            _ramp_map(6, 8, 1.0),
            _ramp_map(6, 8, 2.0),
        )
        fig: Figure
        axes_row: Tuple[Axes, ...]
        fig, axes_row, _ = plot_spectral_cut_series(
            intensities,
            momentum_axis,
            energy_axis,
            titles=("raw", "fit"),
        )
        chex.assert_equal(axes_row[0].get_title(), "raw")
        chex.assert_equal(axes_row[1].get_title(), "fit")
        plt.close(fig)

    def test_titles_length_mismatch_raises(self) -> None:
        """Reject a title count that differs from the map count.

        The test passes three titles for two maps and expects a
        ``ValueError``.

        Notes
        -----
        Matches the message about one title per intensity map. The
        check is static, so no figure exists to close.
        """
        momentum_axis: Float64[NDArray, " 6"] = np.linspace(-0.5, 0.5, 6)
        energy_axis: Float64[NDArray, " 8"] = np.linspace(-2.0, 0.5, 8)
        intensities: Tuple[Float64[NDArray, "6 8"], ...] = (
            _ramp_map(6, 8, 1.0),
            _ramp_map(6, 8, 2.0),
        )
        with pytest.raises(ValueError, match="one title per intensity map"):
            plot_spectral_cut_series(
                intensities,
                momentum_axis,
                energy_axis,
                titles=("a", "b", "c"),
            )

    def test_reuses_supplied_axes(self) -> None:
        """Render the panels on caller-supplied axes.

        The test creates two axes, passes them as a tuple, and checks
        the identities of the returned figure and axes.

        Notes
        -----
        The returned figure is the parent of the supplied axes, and
        each returned axis is the supplied axis at the same position.
        """
        momentum_axis: Float64[NDArray, " 6"] = np.linspace(-0.5, 0.5, 6)
        energy_axis: Float64[NDArray, " 8"] = np.linspace(-2.0, 0.5, 8)
        intensities: Tuple[Float64[NDArray, "6 8"], ...] = (
            _ramp_map(6, 8, 1.0),
            _ramp_map(6, 8, 2.0),
        )
        fig: Figure
        left_axis: Axes
        right_axis: Axes
        fig, (left_axis, right_axis) = plt.subplots(1, 2)
        supplied: Tuple[Axes, ...] = (left_axis, right_axis)
        out_fig: Figure
        axes_row: Tuple[Axes, ...]
        out_fig, axes_row, _ = plot_spectral_cut_series(
            intensities,
            momentum_axis,
            energy_axis,
            axes=supplied,
        )
        assert out_fig is fig
        assert axes_row[0] is supplied[0]
        assert axes_row[1] is supplied[1]
        plt.close(fig)


class TestPlotMomentumMapGrid(chex.TestCase):
    """Validate :func:`~diffpes.plots.plot_momentum_map_grid`.

    The tests check the panel count and the shared color limits across
    panels. They check the crosshair guide lines and the y tick removal
    on later panels. They also exercise the title-count validation.

    :see: :func:`~diffpes.plots.plot_momentum_map_grid`
    """

    def test_panel_count_and_shared_clim(self) -> None:
        """Return one panel per map and one shared color scale.

        The test renders two maps with different maxima and compares
        the color limits of both images.

        Notes
        -----
        The ramps span ``[0, 2]`` and ``[0, 4]``, so both images carry
        the shared limits ``(0.0, 4.0)`` through ``get_clim``.
        """
        kx_axis: Float64[NDArray, " 5"] = np.linspace(-0.2, 0.2, 5)
        ky_axis: Float64[NDArray, " 4"] = np.linspace(-0.1, 0.1, 4)
        maps: Tuple[Float64[NDArray, "5 4"], ...] = (
            _ramp_map(5, 4, 2.0),
            _ramp_map(5, 4, 4.0),
        )
        fig: Figure
        axes_row: Tuple[Axes, ...]
        images: List[AxesImage]
        fig, axes_row, images = plot_momentum_map_grid(maps, kx_axis, ky_axis)
        chex.assert_equal(len(axes_row), 2)
        chex.assert_equal(len(images), 2)
        image: AxesImage
        for image in images:
            clim: Tuple[float, float] = tuple(
                float(value) for value in image.get_clim()
            )
            chex.assert_trees_all_close(clim, (0.0, 4.0))
        plt.close(fig)

    def test_crosshair_lines_on_every_panel(self) -> None:
        """Render the crosshair through the same point in every panel.

        The test passes a crosshair position and inspects the guide
        lines of each panel.

        Notes
        -----
        Each panel carries exactly two lines. The vertical line sits at
        ``k_x = 0.1`` and the horizontal line at ``k_y = -0.05``.
        """
        kx_axis: Float64[NDArray, " 5"] = np.linspace(-0.2, 0.2, 5)
        ky_axis: Float64[NDArray, " 4"] = np.linspace(-0.1, 0.1, 4)
        maps: Tuple[Float64[NDArray, "5 4"], ...] = (
            _ramp_map(5, 4, 1.0),
            _ramp_map(5, 4, 2.0),
            _ramp_map(5, 4, 3.0),
        )
        fig: Figure
        axes_row: Tuple[Axes, ...]
        fig, axes_row, _ = plot_momentum_map_grid(
            maps, kx_axis, ky_axis, crosshair=(0.1, -0.05)
        )
        panel_axis: Axes
        for panel_axis in axes_row:
            chex.assert_equal(len(panel_axis.lines), 2)
            x_data: Tuple[float, ...] = tuple(
                float(value)
                for value in np.asarray(panel_axis.lines[0].get_xdata())
            )
            y_data: Tuple[float, ...] = tuple(
                float(value)
                for value in np.asarray(panel_axis.lines[1].get_ydata())
            )
            chex.assert_trees_all_close(x_data, (0.1, 0.1))
            chex.assert_trees_all_close(y_data, (-0.05, -0.05))
        plt.close(fig)

    def test_later_panels_drop_ytick_labels(self) -> None:
        """Remove the y tick labels and y label on later panels.

        The test renders two panels and inspects the second panel's
        tick label texts and y-axis label.

        Notes
        -----
        Every tick label text on the second panel is empty, and the
        second panel's y-axis label is empty.
        """
        kx_axis: Float64[NDArray, " 5"] = np.linspace(-0.2, 0.2, 5)
        ky_axis: Float64[NDArray, " 4"] = np.linspace(-0.1, 0.1, 4)
        maps: Tuple[Float64[NDArray, "5 4"], ...] = (
            _ramp_map(5, 4, 1.0),
            _ramp_map(5, 4, 2.0),
        )
        fig: Figure
        axes_row: Tuple[Axes, ...]
        fig, axes_row, _ = plot_momentum_map_grid(maps, kx_axis, ky_axis)
        tick_texts: List[str] = [
            tick.get_text() for tick in axes_row[1].get_yticklabels()
        ]
        chex.assert_equal(all(text == "" for text in tick_texts), True)
        chex.assert_equal(axes_row[1].get_ylabel(), "")
        plt.close(fig)

    def test_titles_length_mismatch_raises(self) -> None:
        """Reject a title count that differs from the map count.

        The test passes one title for two maps and expects a
        ``ValueError``.

        Notes
        -----
        Matches the message about one title per momentum map. The
        check is static, so no figure exists to close.
        """
        kx_axis: Float64[NDArray, " 5"] = np.linspace(-0.2, 0.2, 5)
        ky_axis: Float64[NDArray, " 4"] = np.linspace(-0.1, 0.1, 4)
        maps: Tuple[Float64[NDArray, "5 4"], ...] = (
            _ramp_map(5, 4, 1.0),
            _ramp_map(5, 4, 2.0),
        )
        with pytest.raises(ValueError, match="one title per momentum map"):
            plot_momentum_map_grid(
                maps, kx_axis, ky_axis, titles=("only one",)
            )


class TestPlotDifferenceMap(chex.TestCase):
    """Validate :func:`~diffpes.plots.plot_difference_map`.

    The tests check the symmetric color limits and the percentile
    scale on outlier data. They check the zero-scale fallback and the
    optional zero lines. They also exercise reuse of a caller-supplied
    axis.

    :see: :func:`~diffpes.plots.plot_difference_map`
    """

    def test_symmetric_clim(self) -> None:
        """Check the color limits stay symmetric around zero.

        The test renders an asymmetric signed map and compares the two
        color limits.

        Notes
        -----
        The data spans ``[-2, 5]``, so the expected limits are
        ``(-5.0, 5.0)`` with ``vmin == -vmax``.
        """
        x_axis: Float64[NDArray, " 4"] = np.linspace(-1.0, 1.0, 4)
        y_axis: Float64[NDArray, " 4"] = np.linspace(0.0, 3.0, 4)
        difference: Float64[NDArray, "4 4"] = np.linspace(
            -2.0, 5.0, 16, dtype=np.float64
        ).reshape(4, 4)
        fig: Figure
        image: AxesImage
        fig, _, image = plot_difference_map(difference, x_axis, y_axis)
        clim: Tuple[float, float] = tuple(
            float(value) for value in image.get_clim()
        )
        chex.assert_trees_all_close(clim, (-5.0, 5.0))
        chex.assert_trees_all_close(clim[0], -clim[1])
        plt.close(fig)

    def test_percentile_scale_differs_from_abs_max(self) -> None:
        """Verify the percentile scale suppresses an outlier.

        The test renders one map twice: once with the absolute-maximum
        scale and once with a percentile scale. It compares the two
        color ranges.

        Notes
        -----
        The map holds values near ``0.5`` and one outlier ``50.0``. The
        absolute-maximum limits are ``(-50.0, 50.0)``. The 75th
        percentile of ``|difference|`` is ``0.5``, so the percentile
        limits are ``(-0.5, 0.5)`` and remain symmetric.
        """
        x_axis: Float64[NDArray, " 4"] = np.linspace(-1.0, 1.0, 4)
        y_axis: Float64[NDArray, " 4"] = np.linspace(0.0, 3.0, 4)
        difference: Float64[NDArray, "4 4"] = np.full(
            (4, 4), 0.5, dtype=np.float64
        )
        difference[1, 2] = -0.25
        difference[0, 0] = 50.0
        fig_max: Figure
        image_max: AxesImage
        fig_max, _, image_max = plot_difference_map(difference, x_axis, y_axis)
        fig_pct: Figure
        image_pct: AxesImage
        fig_pct, _, image_pct = plot_difference_map(
            difference, x_axis, y_axis, scale_percentile=75.0
        )
        clim_max: Tuple[float, float] = tuple(
            float(value) for value in image_max.get_clim()
        )
        clim_pct: Tuple[float, float] = tuple(
            float(value) for value in image_pct.get_clim()
        )
        chex.assert_trees_all_close(clim_max, (-50.0, 50.0))
        chex.assert_trees_all_close(clim_pct, (-0.5, 0.5))
        assert clim_pct[1] < clim_max[1]
        chex.assert_trees_all_close(clim_pct[0], -clim_pct[1])
        plt.close(fig_max)
        plt.close(fig_pct)

    def test_zero_scale_falls_back_to_unit_range(self) -> None:
        """Verify the fallback to the unit range for an all-zero map.

        The test renders a map of zeros and checks the color limits.

        Notes
        -----
        The absolute maximum of the zero map is ``0.0``. The fallback
        scale ``1.0`` gives the valid limits ``(-1.0, 1.0)``, so the
        call does not fail.
        """
        x_axis: Float64[NDArray, " 3"] = np.linspace(-1.0, 1.0, 3)
        y_axis: Float64[NDArray, " 3"] = np.linspace(-1.0, 1.0, 3)
        difference: Float64[NDArray, "3 3"] = np.zeros(
            (3, 3), dtype=np.float64
        )
        fig: Figure
        image: AxesImage
        fig, _, image = plot_difference_map(difference, x_axis, y_axis)
        clim: Tuple[float, float] = tuple(
            float(value) for value in image.get_clim()
        )
        chex.assert_trees_all_close(clim, (-1.0, 1.0))
        plt.close(fig)

    def test_zero_lines_drawn(self) -> None:
        """Render the two origin guide lines on request.

        The test renders one map with ``zero_lines=True`` and inspects
        the axis lines.

        Notes
        -----
        The axis carries exactly two lines: one horizontal line at
        ``y = 0`` and one vertical line at ``x = 0``.
        """
        x_axis: Float64[NDArray, " 3"] = np.linspace(-1.0, 1.0, 3)
        y_axis: Float64[NDArray, " 3"] = np.linspace(-1.0, 1.0, 3)
        difference: Float64[NDArray, "3 3"] = np.linspace(
            -1.0, 1.0, 9, dtype=np.float64
        ).reshape(3, 3)
        fig: Figure
        ax: Axes
        fig, ax, _ = plot_difference_map(
            difference, x_axis, y_axis, zero_lines=True
        )
        chex.assert_equal(len(ax.lines), 2)
        y_data: Tuple[float, ...] = tuple(
            float(value) for value in np.asarray(ax.lines[0].get_ydata())
        )
        x_data: Tuple[float, ...] = tuple(
            float(value) for value in np.asarray(ax.lines[1].get_xdata())
        )
        chex.assert_trees_all_close(y_data, (0.0, 0.0))
        chex.assert_trees_all_close(x_data, (0.0, 0.0))
        plt.close(fig)

    def test_reuses_supplied_axis(self) -> None:
        """Render the difference map on a caller-supplied axis.

        The test creates an axis, passes it to the function, and checks
        the identities of the returned figure and axis.

        Notes
        -----
        The returned figure is the parent of the supplied axis, and the
        returned axis is the supplied axis.
        """
        x_axis: Float64[NDArray, " 3"] = np.linspace(-1.0, 1.0, 3)
        y_axis: Float64[NDArray, " 3"] = np.linspace(-1.0, 1.0, 3)
        difference: Float64[NDArray, "3 3"] = np.linspace(
            -1.0, 1.0, 9, dtype=np.float64
        ).reshape(3, 3)
        fig: Figure
        ax: Axes
        fig, ax = plt.subplots()
        out_fig: Figure
        out_ax: Axes
        out_fig, out_ax, _ = plot_difference_map(
            difference, x_axis, y_axis, ax=ax
        )
        assert out_fig is fig
        assert out_ax is ax
        plt.close(fig)
