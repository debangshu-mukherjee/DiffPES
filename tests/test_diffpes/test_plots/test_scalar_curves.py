"""Validate scalar-curve plotting utilities.

The tests check labeled curve families, colors, and logarithmic
scales. They check density-of-states rendering with Fermi shifts,
windows, normalization, and shading. They also check overlay
rendering and planar-average profiles.
"""

import jax.numpy as jnp
import matplotlib
import numpy as np

matplotlib.use("Agg")

import chex
import matplotlib.pyplot as plt
import pytest
from beartype.typing import List, Tuple
from jaxtyping import Array, Float64
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from numpy.typing import NDArray

from diffpes.plots.scalar_curves import (
    plot_curve_family,
    plot_dos,
    plot_dos_overlay,
    plot_planar_average,
)
from diffpes.types import DensityOfStates, make_density_of_states


def _make_dos(
    fermi_energy: float = 0.5, scale: float = 1.0
) -> DensityOfStates:
    """PRIVATE: Build a small deterministic DensityOfStates for tests.

    Creates an eleven-point density of states on a strictly increasing
    energy axis from -2 eV to 3 eV. The DOS values increase linearly
    from ``0.2 * scale`` to ``1.0 * scale``, so the maximum sits at the
    last sample.

    Parameters
    ----------
    fermi_energy : float, optional
        Fermi level in eV. Default 0.5.
    scale : float, optional
        Multiplier of the DOS values. Default 1.0.

    Returns
    -------
    dos : DensityOfStates
        Validated carrier with energy shape ``(11,)`` and matching
        total DOS.

    Notes
    -----
    Uses ``jnp.linspace`` for both arrays. The interpolation can round
    a grid value by one unit in the last place, so window tests place
    their bounds between grid samples. The validated factory casts the
    inputs to float64.
    """
    energy: Float64[Array, " 11"] = jnp.linspace(
        -2.0, 3.0, 11, dtype=jnp.float64
    )
    total_dos: Float64[Array, " 11"] = scale * jnp.linspace(
        0.2, 1.0, 11, dtype=jnp.float64
    )
    dos: DensityOfStates = make_density_of_states(
        energy=energy,
        total_dos=total_dos,
        fermi_energy=fermi_energy,
    )
    return dos


class TestPlotCurveFamily(chex.TestCase):
    """Validate :func:`~diffpes.plots.plot_curve_family`.

    Covers the line count, the label round trip, the label-length
    validation, the logarithmic scales, explicit and colormap colors,
    and the reuse of a caller-supplied axis.

    :see: :func:`~diffpes.plots.plot_curve_family`
    """

    def test_line_count_and_label_round_trip(self) -> None:
        """Render one line per curve and round-trip the labels.

        The test plots three curves with three labels. It expects three
        line artists, the given labels on the lines, and the same
        labels in the legend.

        Notes
        -----
        The test builds the inputs in the test body and checks the
        stated property with the documented numerical or structural
        assertions.
        """
        x_axis: Float64[Array, " 5"]
        curves: Tuple[Float64[Array, " 5"], ...]
        labels: Tuple[str, str, str]
        fig: Figure
        ax: Axes
        lines: List[Line2D]
        legend_texts: List[str]

        x_axis = jnp.linspace(0.0, 1.0, 5, dtype=jnp.float64)
        curves = (x_axis, 2.0 * x_axis, 3.0 * x_axis)
        labels = ("a", "b", "c")
        fig, ax, lines = plot_curve_family(x_axis, curves, labels=labels)
        chex.assert_equal(len(lines), 3)
        chex.assert_equal([line.get_label() for line in lines], list(labels))
        legend_texts = [
            text.get_text() for text in ax.get_legend().get_texts()
        ]
        chex.assert_equal(legend_texts, list(labels))
        plt.close(fig)

    def test_label_mismatch_raises(self) -> None:
        """Reject a labels tuple whose length differs from the curves.

        The test passes two curves with three labels. It expects a
        ValueError about the labels length.

        Notes
        -----
        The test builds the inputs in the test body and checks the
        stated property with the documented numerical or structural
        assertions.
        """
        x_axis: Float64[Array, " 5"] = jnp.linspace(
            0.0, 1.0, 5, dtype=jnp.float64
        )
        with pytest.raises(ValueError, match="labels length"):
            plot_curve_family(
                x_axis,
                (x_axis, 2.0 * x_axis),
                labels=("a", "b", "c"),
            )

    def test_log_scales_applied(self) -> None:
        """Apply logarithmic scales on both axes when requested.

        The test plots one positive curve with ``log_x`` and ``log_y``.
        It expects ``"log"`` from ``get_xscale`` and ``get_yscale``.

        Notes
        -----
        The test builds the inputs in the test body and checks the
        stated property with the documented numerical or structural
        assertions.
        """
        x_axis: Float64[Array, " 5"]
        fig: Figure
        ax: Axes
        lines: List[Line2D]

        x_axis = jnp.linspace(1.0, 10.0, 5, dtype=jnp.float64)
        fig, ax, lines = plot_curve_family(
            x_axis,
            (x_axis,),
            log_x=True,
            log_y=True,
        )
        chex.assert_equal(ax.get_xscale(), "log")
        chex.assert_equal(ax.get_yscale(), "log")
        plt.close(fig)

    def test_explicit_colors_cycled(self) -> None:
        """Repeat the explicit colors over a longer curve family.

        The test plots three curves with two explicit colors. It
        expects the color order red, green, red on the lines.

        Notes
        -----
        The test builds the inputs in the test body and checks the
        stated property with the documented numerical or structural
        assertions.
        """
        x_axis: Float64[Array, " 5"]
        fig: Figure
        ax: Axes
        lines: List[Line2D]

        x_axis = jnp.linspace(0.0, 1.0, 5, dtype=jnp.float64)
        fig, ax, lines = plot_curve_family(
            x_axis,
            (x_axis, 2.0 * x_axis, 3.0 * x_axis),
            colors=("red", "green"),
            legend=False,
        )
        chex.assert_equal(
            [line.get_color() for line in lines],
            ["red", "green", "red"],
        )
        plt.close(fig)

    def test_cmap_colors_sampled_uniformly(self) -> None:
        """Sample a named colormap uniformly over the curve family.

        The test plots three curves with ``cmap="viridis"``. It expects
        each line color close to the colormap value at the uniform
        positions 0, 1/2, and 1, at ``rtol=1e-12``.

        Notes
        -----
        The test builds the inputs in the test body and checks the
        stated property with the documented numerical or structural
        assertions.
        """
        x_axis: Float64[Array, " 5"]
        fig: Figure
        ax: Axes
        lines: List[Line2D]
        positions: Float64[NDArray, " 3"]
        index: int

        x_axis = jnp.linspace(0.0, 1.0, 5, dtype=jnp.float64)
        fig, ax, lines = plot_curve_family(
            x_axis,
            (x_axis, 2.0 * x_axis, 3.0 * x_axis),
            cmap="viridis",
            legend=False,
        )
        positions = np.linspace(0.0, 1.0, 3)
        for index in range(3):
            expected: Tuple[float, ...] = tuple(
                float(channel)
                for channel in plt.get_cmap("viridis")(positions[index])
            )
            chex.assert_trees_all_close(
                np.asarray(lines[index].get_color(), dtype=np.float64),
                np.asarray(expected, dtype=np.float64),
                rtol=1e-12,
            )
        plt.close(fig)

    def test_reuses_supplied_axis(self) -> None:
        """Plot the curve family on a caller-supplied axis.

        The function keeps the supplied axis and its parent figure.

        Notes
        -----
        The test builds the inputs in the test body and checks the
        stated property with the documented numerical or structural
        assertions.
        """
        x_axis: Float64[Array, " 5"]
        fig: Figure
        ax: Axes
        out_fig: Figure
        out_ax: Axes
        _lines: List[Line2D]

        x_axis = jnp.linspace(0.0, 1.0, 5, dtype=jnp.float64)
        fig, ax = plt.subplots()
        out_fig, out_ax, _lines = plot_curve_family(x_axis, (x_axis,), ax=ax)
        assert out_ax is ax
        assert out_fig is fig
        plt.close(fig)


class TestPlotDos(chex.TestCase):
    """Validate :func:`~diffpes.plots.plot_dos`.

    Covers the Fermi-shift arithmetic, the unshifted axis, the energy
    window mask, the empty-window rejection, the normalization, and the
    occupied-shading artist.

    :see: :func:`~diffpes.plots.plot_dos`
    """

    def test_fermi_shift_arithmetic(self) -> None:
        """Verify the Fermi shift of the plotted x data.

        The test plots a DOS with Fermi energy 0.5 eV and the default
        shift. It expects the line x data equal to
        ``energy - fermi_energy`` at ``rtol=1e-12``.

        Notes
        -----
        The test builds the inputs in the test body and checks the
        stated property with the documented numerical or structural
        assertions.
        """
        dos: DensityOfStates
        fig: Figure
        ax: Axes
        line: Line2D
        expected: Float64[NDArray, " 11"]

        dos = _make_dos(fermi_energy=0.5)
        fig, ax, line = plot_dos(dos)
        expected = np.asarray(dos.energy, dtype=np.float64) - 0.5
        chex.assert_trees_all_close(
            np.asarray(line.get_xdata(), dtype=np.float64),
            expected,
            rtol=1e-12,
        )
        plt.close(fig)

    def test_unshifted_axis_keeps_absolute_energy(self) -> None:
        """Keep the absolute energy axis without the Fermi shift.

        The test plots a DOS with ``shift_fermi=False``. It expects the
        line x data equal to the energy field at ``rtol=1e-12``.

        Notes
        -----
        The test builds the inputs in the test body and checks the
        stated property with the documented numerical or structural
        assertions.
        """
        dos: DensityOfStates
        fig: Figure
        ax: Axes
        line: Line2D

        dos = _make_dos(fermi_energy=0.5)
        fig, ax, line = plot_dos(dos, shift_fermi=False)
        chex.assert_trees_all_close(
            np.asarray(line.get_xdata(), dtype=np.float64),
            np.asarray(dos.energy, dtype=np.float64),
            rtol=1e-12,
        )
        plt.close(fig)

    def test_energy_window_masks_data(self) -> None:
        """Limit the plotted samples to the energy window.

        The test plots a DOS with the window ``(-1.25, 1.25)`` in
        shifted coordinates. The shifted grid runs from -2.5 eV to
        2.5 eV in 0.5 eV steps, so the window keeps exactly five
        samples. The window bounds sit half a step away from the
        nearest samples, so floating-point rounding of the grid cannot
        move a sample across a bound.

        Notes
        -----
        The test builds the inputs in the test body and checks the
        stated property with the documented numerical or structural
        assertions.
        """
        dos: DensityOfStates
        fig: Figure
        ax: Axes
        line: Line2D
        x_data: Float64[NDArray, " 5"]

        dos = _make_dos(fermi_energy=0.5)
        fig, ax, line = plot_dos(dos, energy_window=(-1.25, 1.25))
        x_data = np.asarray(line.get_xdata(), dtype=np.float64)
        chex.assert_shape(x_data, (5,))
        chex.assert_equal(bool(np.min(x_data) >= -1.25), True)
        chex.assert_equal(bool(np.max(x_data) <= 1.25), True)
        plt.close(fig)

    def test_empty_energy_window_raises(self) -> None:
        """Reject an energy window that selects no samples.

        The test uses the window ``(10.0, 11.0)``, which lies above the
        shifted grid. It expects a ValueError about the empty
        selection.

        Notes
        -----
        The test builds the inputs in the test body and checks the
        stated property with the documented numerical or structural
        assertions.
        """
        dos: DensityOfStates = _make_dos(fermi_energy=0.5)
        with pytest.raises(ValueError, match="selects no"):
            plot_dos(dos, energy_window=(10.0, 11.0))

    def test_normalized_maximum_is_one(self) -> None:
        """Normalize the plotted values to a maximum of one.

        The test plots a scaled DOS with ``normalized=True``. It
        expects the maximum of the line y data equal to 1.0 at
        ``rtol=1e-12``.

        Notes
        -----
        The test builds the inputs in the test body and checks the
        stated property with the documented numerical or structural
        assertions.
        """
        dos: DensityOfStates
        fig: Figure
        ax: Axes
        line: Line2D
        y_maximum: float

        dos = _make_dos(fermi_energy=0.5, scale=7.0)
        fig, ax, line = plot_dos(dos, normalized=True)
        y_maximum = float(
            np.max(np.asarray(line.get_ydata(), dtype=np.float64))
        )
        chex.assert_trees_all_close(y_maximum, 1.0, rtol=1e-12)
        plt.close(fig)

    def test_shading_artist_presence(self) -> None:
        """Add one fill collection only when the caller requests shading.

        The test plots one DOS with shading and one without. It expects
        one collection on the shaded axis and none on the plain axis.

        Notes
        -----
        The test builds the inputs in the test body and checks the
        stated property with the documented numerical or structural
        assertions.
        """
        dos: DensityOfStates
        fig: Figure
        ax: Axes
        line: Line2D
        plain_fig: Figure
        plain_ax: Axes
        _line: Line2D

        dos = _make_dos(fermi_energy=0.5)
        fig, ax, line = plot_dos(dos, shade_occupied=True)
        chex.assert_equal(len(ax.collections), 1)
        plain_fig, plain_ax, _line = plot_dos(dos, shade_occupied=False)
        chex.assert_equal(len(plain_ax.collections), 0)
        plt.close(fig)
        plt.close(plain_fig)


class TestPlotDosOverlay(chex.TestCase):
    """Validate :func:`~diffpes.plots.plot_dos_overlay`.

    Covers the line count, the label round trip, the label-length
    validation, the per-curve normalization, and explicit overlay
    colors.

    :see: :func:`~diffpes.plots.plot_dos_overlay`
    """

    def test_line_count_and_label_round_trip(self) -> None:
        """Render one line per carrier and round-trip the labels.

        The test overlays two DOS carriers with two labels. It expects
        two line artists and the given labels in the legend.

        Notes
        -----
        The test builds the inputs in the test body and checks the
        stated property with the documented numerical or structural
        assertions.
        """
        dos_curves: Tuple[DensityOfStates, DensityOfStates]
        labels: Tuple[str, str]
        fig: Figure
        ax: Axes
        lines: List[Line2D]
        legend_texts: List[str]

        dos_curves = (
            _make_dos(fermi_energy=0.5),
            _make_dos(fermi_energy=0.5, scale=2.0),
        )
        labels = ("bulk", "slab")
        fig, ax, lines = plot_dos_overlay(dos_curves, labels=labels)
        chex.assert_equal(len(lines), 2)
        chex.assert_equal([line.get_label() for line in lines], list(labels))
        legend_texts = [
            text.get_text() for text in ax.get_legend().get_texts()
        ]
        chex.assert_equal(legend_texts, list(labels))
        plt.close(fig)

    def test_label_mismatch_raises(self) -> None:
        """Reject a labels tuple whose length differs from the curves.

        The test passes one carrier with two labels. It expects a
        ValueError about the labels length.

        Notes
        -----
        The test builds the inputs in the test body and checks the
        stated property with the documented numerical or structural
        assertions.
        """
        dos: DensityOfStates = _make_dos(fermi_energy=0.5)
        with pytest.raises(ValueError, match="labels length"):
            plot_dos_overlay((dos,), labels=("a", "b"))

    def test_per_curve_normalization(self) -> None:
        """Normalize every overlay curve by its own maximum.

        The test overlays two carriers with maxima 1.0 and 5.0 under
        the default ``normalized=True``. It expects the maximum of each
        line y data equal to 1.0 at ``rtol=1e-12``.

        Notes
        -----
        The test builds the inputs in the test body and checks the
        stated property with the documented numerical or structural
        assertions.
        """
        dos_curves: Tuple[DensityOfStates, DensityOfStates]
        fig: Figure
        ax: Axes
        lines: List[Line2D]
        line: Line2D

        dos_curves = (
            _make_dos(fermi_energy=0.5),
            _make_dos(fermi_energy=0.5, scale=5.0),
        )
        fig, ax, lines = plot_dos_overlay(dos_curves)
        for line in lines:
            y_maximum: float = float(
                np.max(np.asarray(line.get_ydata(), dtype=np.float64))
            )
            chex.assert_trees_all_close(y_maximum, 1.0, rtol=1e-12)
        plt.close(fig)

    def test_explicit_colors_honored(self) -> None:
        """Apply the explicit overlay colors in carrier order.

        The test overlays two carriers with two explicit colors. It
        expects the given colors on the lines in order.

        Notes
        -----
        The test builds the inputs in the test body and checks the
        stated property with the documented numerical or structural
        assertions.
        """
        dos_curves: Tuple[DensityOfStates, DensityOfStates]
        fig: Figure
        ax: Axes
        lines: List[Line2D]

        dos_curves = (
            _make_dos(fermi_energy=0.5),
            _make_dos(fermi_energy=0.5, scale=2.0),
        )
        fig, ax, lines = plot_dos_overlay(
            dos_curves, colors=("tab:orange", "tab:green")
        )
        chex.assert_equal(
            [line.get_color() for line in lines],
            ["tab:orange", "tab:green"],
        )
        plt.close(fig)


class TestPlotPlanarAverage(chex.TestCase):
    """Validate :func:`~diffpes.plots.plot_planar_average`.

    Covers the zero y floor, the pinned x limits, the fill artist, the
    plain rendering without a fill, and the reuse of a caller-supplied
    axis.

    :see: :func:`~diffpes.plots.plot_planar_average`
    """

    def test_y_floor_limits_and_fill(self) -> None:
        """Pin the axis limits and add the fill collection.

        The test plots a positive profile over positions from 0 to 12
        Angstrom. It expects a lower y limit of 0.0, x limits equal to
        the position range, and one fill collection.

        Notes
        -----
        The test builds the inputs in the test body and checks the
        stated property with the documented numerical or structural
        assertions.
        """
        positions: Float64[Array, " 13"]
        profile: Float64[Array, " 13"]
        fig: Figure
        ax: Axes
        line: Line2D
        x_limits: Tuple[float, float]

        positions = jnp.linspace(0.0, 12.0, 13, dtype=jnp.float64)
        profile = jnp.linspace(0.1, 0.5, 13, dtype=jnp.float64)
        fig, ax, line = plot_planar_average(positions, profile)
        chex.assert_equal(float(ax.get_ylim()[0]), 0.0)
        x_limits = tuple(float(value) for value in ax.get_xlim())
        chex.assert_equal(x_limits, (0.0, 12.0))
        chex.assert_equal(len(ax.collections), 1)
        chex.assert_trees_all_close(
            np.asarray(line.get_xdata(), dtype=np.float64),
            np.asarray(positions, dtype=np.float64),
            rtol=1e-12,
        )
        plt.close(fig)

    def test_no_fill_leaves_no_collections(self) -> None:
        """Render the profile without a fill collection.

        The test plots the profile with ``fill=False``. It expects no
        collections on the axis.

        Notes
        -----
        The test builds the inputs in the test body and checks the
        stated property with the documented numerical or structural
        assertions.
        """
        positions: Float64[Array, " 13"]
        profile: Float64[Array, " 13"]
        fig: Figure
        ax: Axes
        _line: Line2D

        positions = jnp.linspace(0.0, 12.0, 13, dtype=jnp.float64)
        profile = jnp.linspace(0.1, 0.5, 13, dtype=jnp.float64)
        fig, ax, _line = plot_planar_average(positions, profile, fill=False)
        chex.assert_equal(len(ax.collections), 0)
        plt.close(fig)

    def test_reuses_supplied_axis(self) -> None:
        """Plot the profile on a caller-supplied axis.

        The function keeps the supplied axis and its parent figure.

        Notes
        -----
        The test builds the inputs in the test body and checks the
        stated property with the documented numerical or structural
        assertions.
        """
        positions: Float64[Array, " 13"]
        profile: Float64[Array, " 13"]
        fig: Figure
        ax: Axes
        out_fig: Figure
        out_ax: Axes
        _line: Line2D

        positions = jnp.linspace(0.0, 12.0, 13, dtype=jnp.float64)
        profile = jnp.linspace(0.1, 0.5, 13, dtype=jnp.float64)
        fig, ax = plt.subplots()
        out_fig, out_ax, _line = plot_planar_average(positions, profile, ax=ax)
        assert out_ax is ax
        assert out_fig is fig
        plt.close(fig)
