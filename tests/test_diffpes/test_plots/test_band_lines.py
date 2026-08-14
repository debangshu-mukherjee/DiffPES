r"""Validate band-line plotting utilities.

The tests cover dispersion line plots, band overlays on spectral
images, and weight-encoded band scatters. The cases check the returned
artist types, axis reuse, and carrier versus raw-array equivalence.
Further cases check the Fermi shift, the momentum-axis defaults, the
scatter encodings, the backdrop lines, and the weight-shape validation.
"""

import chex
import jax.numpy as jnp
import matplotlib
import numpy as np
import pytest
from beartype import beartype
from beartype.typing import List
from jaxtyping import Array, Float64, jaxtyped
from numpy.typing import NDArray

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.collections import PathCollection
from matplotlib.figure import Figure
from matplotlib.image import AxesImage
from matplotlib.lines import Line2D

from diffpes.plots.band_lines import (
    plot_band_dispersion,
    plot_band_scatter_weights,
    plot_bands_over_spectrum,
)
from diffpes.types import BandStructure, make_band_structure


@jaxtyped(typechecker=beartype)
def _make_bands(
    n_kpoints: int = 12,
    n_bands: int = 3,
    fermi_energy: float = 0.25,
) -> BandStructure:
    """PRIVATE: Build a small validated BandStructure for plotting tests.

    Creates cosine dispersions with one constant offset per band, so
    the plotting functions receive deterministic eigenvalues without
    reading files.

    Parameters
    ----------
    n_kpoints : int, optional
        Number of k-points along the path. Default 12.
    n_bands : int, optional
        Number of bands. Default 3.
    fermi_energy : float, optional
        Fermi energy in eV. Default 0.25.

    Returns
    -------
    bands : BandStructure
        Carrier with ``eigenvalues`` of shape ``(n_kpoints, n_bands)``
        and the given Fermi energy.

    Notes
    -----
    Builds ``cos(pi * k) + offset_b`` on a unit k-axis and stacks the
    k-axis into ``(n_kpoints, 3)`` k-point coordinates.
    """
    band_offsets: Float64[Array, " n_bands"] = jnp.linspace(-1.0, 1.0, n_bands)
    k_axis: Float64[Array, " n_kpoints"] = jnp.linspace(0.0, 1.0, n_kpoints)
    eigenvalues: Float64[Array, "n_kpoints n_bands"] = (
        jnp.cos(jnp.pi * k_axis)[:, jnp.newaxis] + band_offsets[jnp.newaxis, :]
    )
    kpoints: Float64[Array, "n_kpoints 3"] = jnp.stack(
        (k_axis, jnp.zeros_like(k_axis), jnp.zeros_like(k_axis)),
        axis=1,
    )
    bands: BandStructure = make_band_structure(
        eigenvalues=eigenvalues,
        kpoints=kpoints,
        fermi_energy=fermi_energy,
    )
    return bands


class TestPlotBandDispersion(chex.TestCase):
    """Validate :func:`diffpes.plots.plot_band_dispersion`.

    Covers the returned figure, axis, and line artists, the Fermi guide
    line, and the reuse of an existing axis. Covers the equivalence of
    carrier and raw-array inputs, the Fermi shift, and the momentum-axis
    defaults.

    :see: :func:`~diffpes.plots.plot_band_dispersion`
    """

    def test_returns_expected_objects(self) -> None:
        """Return one line per band with the default index labels.

        Confirms the returned types, one Line2D per band, the Fermi
        guide line, and the default axis labels (the *what*).

        Notes
        -----
        Plots a 12-point, 3-band carrier. Asserts a ``Figure``, an
        ``Axes``, three ``Line2D`` artists, and four axis lines in
        total (three bands plus the Fermi guide). Asserts the x-label
        ``"k-point index"`` and the y-label ``"$E - E_F$ (eV)"``
        (the *how*).
        """
        fig: Figure
        ax: Axes
        lines: List[Line2D]

        fig, ax, lines = plot_band_dispersion(_make_bands())
        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)
        chex.assert_equal(len(lines), 3)
        chex.assert_equal(all(isinstance(ln, Line2D) for ln in lines), True)
        chex.assert_equal(len(ax.get_lines()), 4)
        chex.assert_equal(ax.get_xlabel(), "k-point index")
        chex.assert_equal(ax.get_ylabel(), r"$E - E_F$ (eV)")
        plt.close(fig)

    def test_fermi_line_disabled(self) -> None:
        """Plot only the band lines when the Fermi guide is off.

        Confirms ``fermi_line=False`` removes the horizontal guide line
        (the *what*).

        Notes
        -----
        Plots the 3-band carrier with ``fermi_line=False`` and asserts
        exactly three axis lines (the *how*).
        """
        fig: Figure
        ax: Axes
        lines: List[Line2D]

        fig, ax, lines = plot_band_dispersion(_make_bands(), fermi_line=False)
        chex.assert_equal(len(ax.get_lines()), 3)
        plt.close(fig)

    def test_reuses_supplied_axis(self) -> None:
        """Plot the dispersions on a caller-supplied axis.

        Confirms the function keeps the supplied axis and its parent
        figure (the *what*).

        Notes
        -----
        Creates a figure and axis with ``plt.subplots()``, passes the
        axis, and asserts the identity of the returned figure and axis
        (the *how*).
        """
        fig: Figure
        ax: Axes
        out_fig: Figure
        out_ax: Axes

        fig, ax = plt.subplots()
        out_fig, out_ax, _ = plot_band_dispersion(_make_bands(), ax=ax)
        chex.assert_equal(out_fig is fig, True)
        chex.assert_equal(out_ax is ax, True)
        plt.close(fig)

    def test_carrier_matches_shifted_raw_array(self) -> None:
        """Match the carrier plot with a pre-shifted raw-array plot.

        Confirms a carrier with a nonzero Fermi energy and a raw array
        of ``eigenvalues - fermi_energy`` produce identical line data
        (the *what*).

        Notes
        -----
        Builds the carrier with ``fermi_energy=0.25``, forms the raw
        array ``eigenvalues - 0.25``, plots both, and compares the
        y-data of every returned line at ``rtol=1e-12`` (the *how*).
        """
        bands: BandStructure
        raw_values: Float64[Array, "n_kpoints n_bands"]
        carrier_fig: Figure
        carrier_lines: List[Line2D]
        raw_fig: Figure
        raw_lines: List[Line2D]
        carrier_line: Line2D
        raw_line: Line2D

        bands = _make_bands(fermi_energy=0.25)
        raw_values = bands.eigenvalues - bands.fermi_energy
        carrier_fig, _, carrier_lines = plot_band_dispersion(bands)
        raw_fig, _, raw_lines = plot_band_dispersion(raw_values)
        for carrier_line, raw_line in zip(
            carrier_lines, raw_lines, strict=True
        ):
            chex.assert_trees_all_close(
                np.asarray(carrier_line.get_ydata()),
                np.asarray(raw_line.get_ydata()),
                rtol=1e-12,
            )
        plt.close(carrier_fig)
        plt.close(raw_fig)

    def test_shift_fermi_moves_line_data(self) -> None:
        """Apply the Fermi shift to the carrier eigenvalues.

        Confirms ``shift_fermi=True`` subtracts the Fermi energy from
        the plotted y-data and ``shift_fermi=False`` keeps the absolute
        energies (the *what*).

        Notes
        -----
        Plots the carrier with both flag values and compares the first
        line's y-data with ``eigenvalues[:, 0] - 0.25`` and with
        ``eigenvalues[:, 0]`` at ``rtol=1e-12`` (the *how*).
        """
        bands: BandStructure
        shifted_fig: Figure
        shifted_lines: List[Line2D]
        absolute_fig: Figure
        absolute_lines: List[Line2D]
        eigenvalues: Float64[NDArray, "n_kpoints n_bands"]

        bands = _make_bands(fermi_energy=0.25)
        eigenvalues = np.asarray(bands.eigenvalues, dtype=np.float64)
        shifted_fig, _, shifted_lines = plot_band_dispersion(
            bands, shift_fermi=True
        )
        absolute_fig, _, absolute_lines = plot_band_dispersion(
            bands, shift_fermi=False
        )
        chex.assert_trees_all_close(
            np.asarray(shifted_lines[0].get_ydata()),
            eigenvalues[:, 0] - 0.25,
            rtol=1e-12,
        )
        chex.assert_trees_all_close(
            np.asarray(absolute_lines[0].get_ydata()),
            eigenvalues[:, 0],
            rtol=1e-12,
        )
        plt.close(shifted_fig)
        plt.close(absolute_fig)

    def test_momentum_axis_sets_xdata_and_xlabel(self) -> None:
        r"""Use the physical momentum axis for x-data and x-label.

        Confirms the default x-data is the k-point index and an
        explicit momentum axis replaces both the x-data and the default
        x-label (the *what*).

        Notes
        -----
        Plots the carrier once without a momentum axis and once with a
        12-point axis on ``[-0.4, 0.4]`` 1/Angstrom. Asserts the index
        x-data ``arange(12)``, the momentum x-data, and the momentum
        x-label ``"$k$ ($\mathrm{\AA}^{-1}$)"`` (the *how*).
        """
        momentum_axis: Float64[Array, " n_kpoints"]
        index_fig: Figure
        index_lines: List[Line2D]
        momentum_fig: Figure
        momentum_ax: Axes
        momentum_lines: List[Line2D]

        momentum_axis = jnp.linspace(-0.4, 0.4, 12)
        index_fig, _, index_lines = plot_band_dispersion(_make_bands())
        momentum_fig, momentum_ax, momentum_lines = plot_band_dispersion(
            _make_bands(), momentum_axis=momentum_axis
        )
        chex.assert_trees_all_close(
            np.asarray(index_lines[0].get_xdata()),
            np.arange(12, dtype=np.float64),
            rtol=1e-12,
        )
        chex.assert_trees_all_close(
            np.asarray(momentum_lines[0].get_xdata()),
            np.asarray(momentum_axis, dtype=np.float64),
            rtol=1e-12,
        )
        chex.assert_equal(
            momentum_ax.get_xlabel(), r"$k$ ($\mathrm{\AA}^{-1}$)"
        )
        plt.close(index_fig)
        plt.close(momentum_fig)


class TestPlotBandsOverSpectrum(chex.TestCase):
    """Validate :func:`diffpes.plots.plot_bands_over_spectrum`.

    Covers the returned image artist, the number and color of the
    overlaid band lines, and the reuse of an existing axis.

    :see: :func:`~diffpes.plots.plot_bands_over_spectrum`
    """

    def test_overlays_bands_on_image(self) -> None:
        """Render the spectral image with one overlay line per band.

        Confirms the function returns an ``AxesImage`` and overlays
        every band as one line in ``band_color`` on the image axis
        (the *what*).

        Notes
        -----
        Builds a ``(12, 40)`` intensity grid with matching momentum and
        energy axes and a 3-band carrier. Calls the function with
        ``colorbar=False`` and ``band_color="cyan"``. Asserts the image
        type, four axis lines in total (the Fermi guide plus three
        bands), and exactly three lines in ``"cyan"`` (the *how*).
        """
        intensity: Float64[Array, "n_kpoints n_energies"]
        momentum_axis: Float64[Array, " n_kpoints"]
        energy_axis: Float64[Array, " n_energies"]
        fig: Figure
        ax: Axes
        image: AxesImage
        band_lines: List[Line2D]

        intensity = jnp.linspace(0.0, 1.0, 12 * 40).reshape(12, 40)
        momentum_axis = jnp.linspace(-0.4, 0.4, 12)
        energy_axis = jnp.linspace(-2.5, 0.5, 40)
        fig, ax, image = plot_bands_over_spectrum(
            intensity=intensity,
            momentum_axis=momentum_axis,
            energy_axis=energy_axis,
            bands=_make_bands(),
            colorbar=False,
            band_color="cyan",
        )
        assert isinstance(image, AxesImage)
        chex.assert_equal(len(ax.get_lines()), 4)
        band_lines = [
            line for line in ax.get_lines() if line.get_color() == "cyan"
        ]
        chex.assert_equal(len(band_lines), 3)
        plt.close(fig)

    def test_reuses_supplied_axis(self) -> None:
        """Render the overlay on a caller-supplied axis.

        Confirms the function keeps the supplied axis and its parent
        figure (the *what*).

        Notes
        -----
        Creates a figure and axis with ``plt.subplots()``, passes the
        axis, and asserts the identity of the returned figure and axis
        (the *how*).
        """
        intensity: Float64[Array, "n_kpoints n_energies"]
        momentum_axis: Float64[Array, " n_kpoints"]
        energy_axis: Float64[Array, " n_energies"]
        fig: Figure
        ax: Axes
        out_fig: Figure
        out_ax: Axes

        intensity = jnp.ones((12, 20), dtype=jnp.float64)
        momentum_axis = jnp.linspace(-0.4, 0.4, 12)
        energy_axis = jnp.linspace(-2.5, 0.5, 20)
        fig, ax = plt.subplots()
        out_fig, out_ax, _ = plot_bands_over_spectrum(
            intensity=intensity,
            momentum_axis=momentum_axis,
            energy_axis=energy_axis,
            bands=_make_bands(),
            ax=ax,
            colorbar=False,
        )
        chex.assert_equal(out_fig is fig, True)
        chex.assert_equal(out_ax is ax, True)
        plt.close(fig)


class TestPlotBandScatterWeights(chex.TestCase):
    """Validate :func:`diffpes.plots.plot_band_scatter_weights`.

    Covers the size and color encodings, the shifted scatter y-data,
    and the backdrop line count. Covers the reuse of an existing axis
    and the rejection of mismatched weight shapes.

    :see: :func:`~diffpes.plots.plot_band_scatter_weights`
    """

    def test_size_mode_encodes_weight_magnitude(self) -> None:
        """Encode the weight magnitude as the marker area.

        Confirms the ``"size"`` mode sets the marker areas to
        ``size_scale * |weights|`` and places the points at the shifted
        band energies (the *what*).

        Notes
        -----
        Uses signed weights on the 12-point, 3-band carrier with
        ``size_scale=10.0``. Asserts a ``PathCollection``, the sizes
        ``10.0 * |weights|`` raveled in k-major order, and the offset
        y-values ``(eigenvalues - 0.25).ravel()`` at ``rtol=1e-12``
        (the *how*).
        """
        bands: BandStructure
        weights: Float64[Array, "n_kpoints n_bands"]
        fig: Figure
        ax: Axes
        scatter: PathCollection
        expected_sizes: Float64[NDArray, " n_points"]
        expected_energy: Float64[NDArray, " n_points"]

        bands = _make_bands(fermi_energy=0.25)
        weights = jnp.linspace(-1.0, 1.0, 12 * 3).reshape(12, 3)
        fig, ax, scatter = plot_band_scatter_weights(
            bands, weights, size_scale=10.0
        )
        assert isinstance(scatter, PathCollection)
        expected_sizes = (
            10.0 * np.abs(np.asarray(weights, dtype=np.float64)).ravel()
        )
        chex.assert_trees_all_close(
            np.asarray(scatter.get_sizes()),
            expected_sizes,
            rtol=1e-12,
        )
        expected_energy = (
            np.asarray(bands.eigenvalues, dtype=np.float64) - 0.25
        ).ravel()
        chex.assert_trees_all_close(
            np.asarray(scatter.get_offsets())[:, 1],
            expected_energy,
            rtol=1e-12,
        )
        plt.close(fig)

    def test_color_mode_encodes_weight_values(self) -> None:
        """Encode the weight values as the marker colors.

        Confirms the ``"color"`` mode maps the raveled weights to the
        scatter color array with a fixed marker size. Confirms the
        marker placement on the explicit momentum axis and the
        requested colorbar (the *what*).

        Notes
        -----
        Calls the function with ``mode="color"``, a 12-point momentum
        axis, and ``colorbar=True``. Asserts the color array equals the
        raveled weights and the sizes equal ``[4.0]``. Asserts the
        offset x-values repeat the momentum axis per band and the
        figure holds two axes (the *how*).
        """
        weights: Float64[Array, "n_kpoints n_bands"]
        momentum_axis: Float64[Array, " n_kpoints"]
        fig: Figure
        ax: Axes
        scatter: PathCollection
        expected_x: Float64[NDArray, " n_points"]

        weights = jnp.linspace(0.0, 1.0, 12 * 3).reshape(12, 3)
        momentum_axis = jnp.linspace(-0.4, 0.4, 12)
        fig, ax, scatter = plot_band_scatter_weights(
            _make_bands(),
            weights,
            momentum_axis=momentum_axis,
            mode="color",
            colorbar=True,
        )
        chex.assert_trees_all_close(
            np.asarray(scatter.get_array()),
            np.asarray(weights, dtype=np.float64).ravel(),
            rtol=1e-12,
        )
        chex.assert_trees_all_close(
            np.asarray(scatter.get_sizes()),
            np.asarray([4.0]),
            rtol=1e-12,
        )
        expected_x = np.repeat(np.asarray(momentum_axis, dtype=np.float64), 3)
        chex.assert_trees_all_close(
            np.asarray(scatter.get_offsets())[:, 0],
            expected_x,
            rtol=1e-12,
        )
        chex.assert_equal(len(fig.axes), 2)
        plt.close(fig)

    def test_backdrop_controls_line_count(self) -> None:
        """Plot one thin backdrop line per band only on request.

        Confirms ``backdrop=True`` adds one grey line per band and
        ``backdrop=False`` draws no lines (the *what*).

        Notes
        -----
        Plots the 3-band carrier with both flag values and asserts
        three axis lines with the backdrop and zero lines without it
        (the *how*).
        """
        weights: Float64[Array, "n_kpoints n_bands"]
        backdrop_fig: Figure
        backdrop_ax: Axes
        bare_fig: Figure
        bare_ax: Axes

        weights = jnp.ones((12, 3), dtype=jnp.float64)
        backdrop_fig, backdrop_ax, _ = plot_band_scatter_weights(
            _make_bands(), weights, backdrop=True
        )
        bare_fig, bare_ax, _ = plot_band_scatter_weights(
            _make_bands(), weights, backdrop=False
        )
        chex.assert_equal(len(backdrop_ax.get_lines()), 3)
        chex.assert_equal(len(bare_ax.get_lines()), 0)
        plt.close(backdrop_fig)
        plt.close(bare_fig)

    def test_reuses_supplied_axis(self) -> None:
        """Render the scatter on a caller-supplied axis.

        Confirms the function keeps the supplied axis and its parent
        figure (the *what*).

        Notes
        -----
        Creates a figure and axis with ``plt.subplots()``, passes the
        axis, and asserts the identity of the returned figure and axis
        (the *how*).
        """
        weights: Float64[Array, "n_kpoints n_bands"]
        fig: Figure
        ax: Axes
        out_fig: Figure
        out_ax: Axes

        weights = jnp.ones((12, 3), dtype=jnp.float64)
        fig, ax = plt.subplots()
        out_fig, out_ax, _ = plot_band_scatter_weights(
            _make_bands(), weights, ax=ax
        )
        chex.assert_equal(out_fig is fig, True)
        chex.assert_equal(out_ax is ax, True)
        plt.close(fig)

    def test_rejects_mismatched_weight_shape(self) -> None:
        """Reject weights whose shape disagrees with the eigenvalues.

        Confirms the static ``ValueError`` when the weight matrix has
        one band column more than the carrier eigenvalues (the *what*).

        Notes
        -----
        Passes a ``(12, 4)`` weight matrix with the 3-band carrier and
        expects a ``ValueError`` whose message states the
        ``(n_k, n_bands)`` shape (the *how*).
        """
        weights: Float64[Array, "n_kpoints n_extra"]

        weights = jnp.ones((12, 4), dtype=jnp.float64)
        with pytest.raises(
            ValueError, match="Weights must have shape matching"
        ):
            plot_band_scatter_weights(_make_bands(), weights)
