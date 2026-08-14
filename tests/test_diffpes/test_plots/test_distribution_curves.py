"""Validate distribution-curve plotting utilities.

The tests check nearest-sample EDC and MDC selection, waterfall
offsets, and the intensity transformations. They also check curve
colors, legend labels, Fermi guides, the twin panels, and the
energy-integrated momentum profile.
"""

import chex
import jax.numpy as jnp
import matplotlib
import numpy as np
import pytest
from beartype.typing import List, Tuple
from jaxtyping import Array, Bool, Float64
from numpy.typing import NDArray

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

from diffpes.plots.distribution_curves import (
    plot_distribution_curves,
    plot_edc_mdc_panels,
    plot_momentum_profile,
)


def _make_map() -> Tuple[
    Float64[Array, "7 9"], Float64[Array, " 7"], Float64[Array, " 9"]
]:
    """PRIVATE: Build a deterministic 7 x 9 intensity map with axes.

    Creates a small energy-momentum map whose rows and columns follow
    known closed forms, so curve-selection tests can compare plotted
    data against an exact expectation.

    Returns
    -------
    intensity : Float64[Array, "7 9"]
        Intensity map whose row ``i`` equals ``(i + 1)`` times a linear
        ramp from 0 to 1 over the energy axis.
    momentum_axis : Float64[Array, " 7"]
        Momentum axis from -0.3 to 0.3 in 1/Angstrom.
    energy_axis : Float64[Array, " 9"]
        Energy axis from -2.0 to 0.5 eV relative to the Fermi level.

    Notes
    -----
    Builds the map as the outer product of the row weights
    ``jnp.arange(1.0, 8.0)`` and the ramp ``jnp.linspace(0.0, 1.0, 9)``.
    Row 4 is therefore the ramp scaled by 5, and column 8 equals the
    row weights.
    """
    row_weights: Float64[Array, " 7"] = jnp.arange(1.0, 8.0)
    ramp: Float64[Array, " 9"] = jnp.linspace(0.0, 1.0, 9)
    intensity: Float64[Array, "7 9"] = jnp.outer(row_weights, ramp)
    momentum_axis: Float64[Array, " 7"] = jnp.linspace(-0.3, 0.3, 7)
    energy_axis: Float64[Array, " 9"] = jnp.linspace(-2.0, 0.5, 9)
    spectral_map: Tuple[
        Float64[Array, "7 9"], Float64[Array, " 7"], Float64[Array, " 9"]
    ] = (intensity, momentum_axis, energy_axis)
    return spectral_map


class TestPlotDistributionCurves(chex.TestCase):
    """Validate :func:`diffpes.plots.plot_distribution_curves`.

    The cases check the returned Matplotlib objects, nearest-sample
    EDC and MDC selection, waterfall offsets, and normalization. They
    also check log scaling, explicit colors, legend labels, and the
    EDC Fermi-level guide.

    :see: :func:`~diffpes.plots.plot_distribution_curves`
    """

    def test_returns_figure_axis_and_lines(self) -> None:
        """Return a figure, an axis, and one line per position.

        The test plots three EDCs and checks the returned object types.
        The number of returned ``Line2D`` artists equals the number of
        requested positions.

        Notes
        -----
        The test builds the inputs in the test body and checks the
        stated property with the documented numerical or structural
        assertions.
        """
        intensity: Float64[Array, "7 9"]
        momentum_axis: Float64[Array, " 7"]
        energy_axis: Float64[Array, " 9"]
        fig: Figure
        ax: Axes
        lines: List[Line2D]
        line: Line2D

        intensity, momentum_axis, energy_axis = _make_map()
        fig, ax, lines = plot_distribution_curves(
            intensity,
            momentum_axis,
            energy_axis,
            kind="edc",
            positions=(-0.3, 0.0, 0.3),
            legend=False,
        )
        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)
        chex.assert_equal(len(lines), 3)
        for line in lines:
            assert isinstance(line, Line2D)
        plt.close(fig)

    def test_edc_selects_nearest_momentum_row(self) -> None:
        """Select the intensity row nearest to the requested momentum.

        The requested momentum 0.11 lies nearest to the grid value 0.10
        at row index 4. That row equals the known ramp times 5. The
        plotted y-data equals that row and the x-data equals the energy
        axis.

        Notes
        -----
        The test builds the inputs in the test body and checks the
        stated property with the documented numerical or structural
        assertions at ``rtol=1e-12``.
        """
        intensity: Float64[Array, "7 9"]
        momentum_axis: Float64[Array, " 7"]
        energy_axis: Float64[Array, " 9"]
        fig: Figure
        lines: List[Line2D]

        intensity, momentum_axis, energy_axis = _make_map()
        fig, _, lines = plot_distribution_curves(
            intensity,
            momentum_axis,
            energy_axis,
            kind="edc",
            positions=(0.11,),
            legend=False,
            fermi_line=False,
        )
        ydata: Float64[NDArray, " 9"] = np.asarray(
            lines[0].get_ydata(), dtype=np.float64
        )
        xdata: Float64[NDArray, " 9"] = np.asarray(
            lines[0].get_xdata(), dtype=np.float64
        )
        expected_row: Float64[NDArray, " 9"] = np.asarray(
            intensity, dtype=np.float64
        )[4, :]
        chex.assert_trees_all_close(ydata, expected_row, rtol=1e-12)
        chex.assert_trees_all_close(
            xdata, np.asarray(energy_axis, dtype=np.float64), rtol=1e-12
        )
        plt.close(fig)

    def test_mdc_selects_nearest_energy_column(self) -> None:
        """Select the intensity column nearest to the requested energy.

        The requested energy 0.4 eV lies nearest to the grid value
        0.5 eV at column index 8. That column equals the row weights 1
        through 7. The plotted y-data equals that column and the x-data
        equals the momentum axis.

        Notes
        -----
        The test builds the inputs in the test body and checks the
        stated property with the documented numerical or structural
        assertions at ``rtol=1e-12``.
        """
        intensity: Float64[Array, "7 9"]
        momentum_axis: Float64[Array, " 7"]
        energy_axis: Float64[Array, " 9"]
        fig: Figure
        lines: List[Line2D]

        intensity, momentum_axis, energy_axis = _make_map()
        fig, _, lines = plot_distribution_curves(
            intensity,
            momentum_axis,
            energy_axis,
            kind="mdc",
            positions=(0.4,),
            legend=False,
        )
        ydata: Float64[NDArray, " 7"] = np.asarray(
            lines[0].get_ydata(), dtype=np.float64
        )
        xdata: Float64[NDArray, " 7"] = np.asarray(
            lines[0].get_xdata(), dtype=np.float64
        )
        expected_column: Float64[NDArray, " 7"] = np.asarray(
            intensity, dtype=np.float64
        )[:, 8]
        chex.assert_trees_all_close(ydata, expected_column, rtol=1e-12)
        chex.assert_trees_all_close(
            xdata, np.asarray(momentum_axis, dtype=np.float64), rtol=1e-12
        )
        plt.close(fig)

    def test_offset_stacks_successive_curves(self) -> None:
        """Add the waterfall offset once per successive curve.

        With ``offset=2.5`` the first curve keeps the raw data and the
        second curve equals its raw data plus 2.5.

        Notes
        -----
        The test builds the inputs in the test body and checks the
        stated property with the documented numerical or structural
        assertions at ``rtol=1e-12``.
        """
        intensity: Float64[Array, "7 9"]
        momentum_axis: Float64[Array, " 7"]
        energy_axis: Float64[Array, " 9"]
        fig: Figure
        lines: List[Line2D]

        intensity, momentum_axis, energy_axis = _make_map()
        fig, _, lines = plot_distribution_curves(
            intensity,
            momentum_axis,
            energy_axis,
            kind="edc",
            positions=(-0.3, 0.0),
            offset=2.5,
            legend=False,
            fermi_line=False,
        )
        raw: Float64[NDArray, "7 9"] = np.asarray(intensity, dtype=np.float64)
        first: Float64[NDArray, " 9"] = np.asarray(
            lines[0].get_ydata(), dtype=np.float64
        )
        second: Float64[NDArray, " 9"] = np.asarray(
            lines[1].get_ydata(), dtype=np.float64
        )
        chex.assert_trees_all_close(first, raw[0, :], rtol=1e-12)
        chex.assert_trees_all_close(second, raw[3, :] + 2.5, rtol=1e-12)
        plt.close(fig)

    def test_normalized_curves_peak_at_one(self) -> None:
        """Normalize each curve to a maximum of one.

        With ``normalized=True`` and zero offset, every plotted curve
        has a maximum of exactly 1.0.

        Notes
        -----
        The test builds the inputs in the test body and checks the
        stated property with the documented numerical or structural
        assertions at ``rtol=1e-12``.
        """
        intensity: Float64[Array, "7 9"]
        momentum_axis: Float64[Array, " 7"]
        energy_axis: Float64[Array, " 9"]
        fig: Figure
        lines: List[Line2D]
        line: Line2D

        intensity, momentum_axis, energy_axis = _make_map()
        fig, _, lines = plot_distribution_curves(
            intensity,
            momentum_axis,
            energy_axis,
            kind="edc",
            positions=(-0.2, 0.0, 0.2),
            normalized=True,
            legend=False,
            fermi_line=False,
        )
        for line in lines:
            maximum: float = float(
                np.max(np.asarray(line.get_ydata(), dtype=np.float64))
            )
            chex.assert_trees_all_close(maximum, 1.0, rtol=1e-12)
        plt.close(fig)

    def test_log_counts_applies_log1p(self) -> None:
        """Apply log1p to the counts before plotting.

        With ``log_counts=True`` the plotted EDC equals ``np.log1p`` of
        the selected raw intensity row.

        Notes
        -----
        The test builds the inputs in the test body and checks the
        stated property with the documented numerical or structural
        assertions at ``rtol=1e-12``.
        """
        intensity: Float64[Array, "7 9"]
        momentum_axis: Float64[Array, " 7"]
        energy_axis: Float64[Array, " 9"]
        fig: Figure
        lines: List[Line2D]

        intensity, momentum_axis, energy_axis = _make_map()
        fig, _, lines = plot_distribution_curves(
            intensity,
            momentum_axis,
            energy_axis,
            kind="edc",
            positions=(0.11,),
            log_counts=True,
            legend=False,
            fermi_line=False,
        )
        ydata: Float64[NDArray, " 9"] = np.asarray(
            lines[0].get_ydata(), dtype=np.float64
        )
        expected: Float64[NDArray, " 9"] = np.log1p(
            np.asarray(intensity, dtype=np.float64)[4, :]
        )
        chex.assert_trees_all_close(ydata, expected, rtol=1e-12)
        plt.close(fig)

    def test_explicit_colors_cycle(self) -> None:
        """Apply explicit colors cyclically over the curve stack.

        With two colors and three curves, the third curve reuses the
        first color.

        Notes
        -----
        The test builds the inputs in the test body and checks the
        stated property with the documented numerical or structural
        assertions.
        """
        intensity: Float64[Array, "7 9"]
        momentum_axis: Float64[Array, " 7"]
        energy_axis: Float64[Array, " 9"]
        fig: Figure
        lines: List[Line2D]

        intensity, momentum_axis, energy_axis = _make_map()
        fig, _, lines = plot_distribution_curves(
            intensity,
            momentum_axis,
            energy_axis,
            kind="edc",
            positions=(-0.2, 0.0, 0.2),
            colors=("#ff0000", "#00ff00"),
            legend=False,
            fermi_line=False,
        )
        chex.assert_equal(lines[0].get_color(), "#ff0000")
        chex.assert_equal(lines[1].get_color(), "#00ff00")
        chex.assert_equal(lines[2].get_color(), "#ff0000")
        plt.close(fig)

    def test_legend_labels_state_selected_values(self) -> None:
        """Format legend labels from the selected axis values.

        The EDC legend entry states the selected momentum ``+0.10`` with
        the inverse-Angstrom unit. The MDC legend entry states the
        selected energy ``+0.50`` in eV.

        Notes
        -----
        The test builds the inputs in the test body and checks the
        stated property with the documented numerical or structural
        assertions.
        """
        intensity: Float64[Array, "7 9"]
        momentum_axis: Float64[Array, " 7"]
        energy_axis: Float64[Array, " 9"]
        edc_fig: Figure
        edc_ax: Axes
        mdc_fig: Figure
        mdc_ax: Axes

        intensity, momentum_axis, energy_axis = _make_map()
        edc_fig, edc_ax, _ = plot_distribution_curves(
            intensity,
            momentum_axis,
            energy_axis,
            kind="edc",
            positions=(0.11,),
            legend=True,
        )
        edc_labels: List[str] = [
            text.get_text() for text in edc_ax.get_legend().get_texts()
        ]
        assert "$k = +0.10$" in edc_labels[0]
        assert r"\AA" in edc_labels[0]
        plt.close(edc_fig)

        mdc_fig, mdc_ax, _ = plot_distribution_curves(
            intensity,
            momentum_axis,
            energy_axis,
            kind="mdc",
            positions=(0.4,),
            legend=True,
        )
        mdc_labels: List[str] = [
            text.get_text() for text in mdc_ax.get_legend().get_texts()
        ]
        chex.assert_equal(mdc_labels[0], "$E - E_F = +0.50$ eV")
        plt.close(mdc_fig)

    def test_fermi_line_added_for_edc(self) -> None:
        """Render the dashed vertical Fermi guide on an EDC plot.

        With ``fermi_line=True`` and one EDC, the axis carries two line
        artists. The extra artist is vertical at zero energy with a
        dashed style.

        Notes
        -----
        The test builds the inputs in the test body and checks the
        stated property with the documented numerical or structural
        assertions.
        """
        intensity: Float64[Array, "7 9"]
        momentum_axis: Float64[Array, " 7"]
        energy_axis: Float64[Array, " 9"]
        fig: Figure
        ax: Axes

        intensity, momentum_axis, energy_axis = _make_map()
        fig, ax, _ = plot_distribution_curves(
            intensity,
            momentum_axis,
            energy_axis,
            kind="edc",
            positions=(0.0,),
            legend=False,
            fermi_line=True,
        )
        chex.assert_equal(len(ax.lines), 2)
        guide: Line2D = ax.lines[-1]
        guide_xdata: Float64[NDArray, " 2"] = np.asarray(
            guide.get_xdata(), dtype=np.float64
        )
        chex.assert_trees_all_close(guide_xdata, np.zeros(2), atol=1e-15)
        chex.assert_equal(guide.get_linestyle(), "--")
        plt.close(fig)

    def test_fermi_line_ignored_for_mdc(self) -> None:
        """Skip the Fermi guide option on an MDC plot.

        With ``fermi_line=True`` and one MDC, the axis carries exactly
        one line artist, so the guide is absent.

        Notes
        -----
        The test builds the inputs in the test body and checks the
        stated property with the documented numerical or structural
        assertions.
        """
        intensity: Float64[Array, "7 9"]
        momentum_axis: Float64[Array, " 7"]
        energy_axis: Float64[Array, " 9"]
        fig: Figure
        ax: Axes

        intensity, momentum_axis, energy_axis = _make_map()
        fig, ax, _ = plot_distribution_curves(
            intensity,
            momentum_axis,
            energy_axis,
            kind="mdc",
            positions=(0.0,),
            legend=False,
            fermi_line=True,
        )
        chex.assert_equal(len(ax.lines), 1)
        plt.close(fig)


class TestPlotEdcMdcPanels(chex.TestCase):
    """Validate :func:`diffpes.plots.plot_edc_mdc_panels`.

    Covers the twin-panel layout, the nearest-sample cut selection
    behind both panels, the panel titles, and the reuse of
    caller-supplied axes.

    :see: :func:`~diffpes.plots.plot_edc_mdc_panels`
    """

    def test_returns_two_axes_and_two_lines(self) -> None:
        """Return two titled panels with one curve artist each.

        The EDC panel shows row 4 (momentum +0.10) against energy and
        the MDC panel shows column 8 (energy +0.50 eV) against
        momentum. Each panel title states its selected value.

        Notes
        -----
        The test builds the inputs in the test body and checks the
        stated property with the documented numerical or structural
        assertions at ``rtol=1e-12``.
        """
        intensity: Float64[Array, "7 9"]
        momentum_axis: Float64[Array, " 7"]
        energy_axis: Float64[Array, " 9"]
        fig: Figure
        edc_ax: Axes
        mdc_ax: Axes
        edc_line: Line2D
        mdc_line: Line2D

        intensity, momentum_axis, energy_axis = _make_map()
        fig, (edc_ax, mdc_ax), (edc_line, mdc_line) = plot_edc_mdc_panels(
            intensity,
            momentum_axis,
            energy_axis,
            k_value=0.11,
            energy_value=0.4,
        )
        assert isinstance(fig, Figure)
        assert isinstance(edc_ax, Axes)
        assert isinstance(mdc_ax, Axes)
        assert isinstance(edc_line, Line2D)
        assert isinstance(mdc_line, Line2D)
        assert "EDC at " in edc_ax.get_title()
        assert "$k = +0.10$" in edc_ax.get_title()
        assert "MDC at " in mdc_ax.get_title()
        assert "$E - E_F = +0.50$ eV" in mdc_ax.get_title()
        raw: Float64[NDArray, "7 9"] = np.asarray(intensity, dtype=np.float64)
        chex.assert_trees_all_close(
            np.asarray(edc_line.get_ydata(), dtype=np.float64),
            raw[4, :],
            rtol=1e-12,
        )
        chex.assert_trees_all_close(
            np.asarray(mdc_line.get_ydata(), dtype=np.float64),
            raw[:, 8],
            rtol=1e-12,
        )
        plt.close(fig)

    def test_reuses_provided_axes(self) -> None:
        """Render the panels on caller-supplied axes.

        The function keeps the two supplied axes and their parent
        figure instead of creating a new figure.

        Notes
        -----
        The test builds the inputs in the test body and checks the
        stated property with the documented numerical or structural
        assertions.
        """
        intensity: Float64[Array, "7 9"]
        momentum_axis: Float64[Array, " 7"]
        energy_axis: Float64[Array, " 9"]
        fig: Figure
        left_ax: Axes
        right_ax: Axes
        out_fig: Figure
        out_edc_ax: Axes
        out_mdc_ax: Axes

        intensity, momentum_axis, energy_axis = _make_map()
        fig, (left_ax, right_ax) = plt.subplots(1, 2)
        out_fig, (out_edc_ax, out_mdc_ax), _ = plot_edc_mdc_panels(
            intensity,
            momentum_axis,
            energy_axis,
            k_value=0.0,
            energy_value=0.0,
            axes=(left_ax, right_ax),
        )
        assert out_fig is fig
        assert out_edc_ax is left_ax
        assert out_mdc_ax is right_ax
        plt.close(fig)


class TestPlotMomentumProfile(chex.TestCase):
    """Validate :func:`diffpes.plots.plot_momentum_profile`.

    Covers the trapezoid integration over the energy window, the
    returned Matplotlib objects, and the rejection of a window with
    fewer than two energy samples.

    :see: :func:`~diffpes.plots.plot_momentum_profile`
    """

    def test_profile_matches_trapezoid_integration(self) -> None:
        """Integrate the masked energy window with the trapezoid rule.

        The plotted profile equals a direct ``np.trapezoid`` evaluation
        over the energy samples inside the window ``(-1.5, 0.0)`` eV,
        and the x-data equals the momentum axis.

        Notes
        -----
        The test builds the inputs in the test body and checks the
        stated property with the documented numerical or structural
        assertions at ``rtol=1e-12``.
        """
        intensity: Float64[Array, "7 9"]
        momentum_axis: Float64[Array, " 7"]
        energy_axis: Float64[Array, " 9"]
        fig: Figure
        ax: Axes
        line: Line2D

        intensity, momentum_axis, energy_axis = _make_map()
        fig, ax, line = plot_momentum_profile(
            intensity,
            momentum_axis,
            energy_axis,
            energy_window=(-1.5, 0.0),
            fill=False,
        )
        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)
        assert isinstance(line, Line2D)
        raw: Float64[NDArray, "7 9"] = np.asarray(intensity, dtype=np.float64)
        energies: Float64[NDArray, " 9"] = np.asarray(
            energy_axis, dtype=np.float64
        )
        mask: Bool[NDArray, " 9"] = (energies >= -1.5) & (energies <= 0.0)
        expected: Float64[NDArray, " 7"] = np.trapezoid(
            raw[:, mask], x=energies[mask], axis=1
        )
        chex.assert_trees_all_close(
            np.asarray(line.get_ydata(), dtype=np.float64),
            expected,
            rtol=1e-12,
        )
        chex.assert_trees_all_close(
            np.asarray(line.get_xdata(), dtype=np.float64),
            np.asarray(momentum_axis, dtype=np.float64),
            rtol=1e-12,
        )
        plt.close(fig)

    def test_narrow_window_raises_value_error(self) -> None:
        """Reject an energy window with fewer than two samples.

        The window ``(0.45, 0.55)`` eV contains only the energy sample
        at 0.5 eV, so the function raises a ``ValueError``.

        Notes
        -----
        The test builds the inputs in the test body and checks the
        stated property with ``pytest.raises`` on the documented
        message.
        """
        intensity: Float64[Array, "7 9"]
        momentum_axis: Float64[Array, " 7"]
        energy_axis: Float64[Array, " 9"]

        intensity, momentum_axis, energy_axis = _make_map()
        with pytest.raises(ValueError, match="at least two energy samples"):
            plot_momentum_profile(
                intensity,
                momentum_axis,
                energy_axis,
                energy_window=(0.45, 0.55),
            )
