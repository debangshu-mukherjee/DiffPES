"""Tests for ARPES plotting utilities.

Extended Summary
----------------
Exercises the plotting module's public API: plot_arpes_spectrum,
apply_kpath_ticks, and plot_arpes_with_kpath. Tests cover successful
rendering with default and custom options, validation of spectrum array
shapes and compatibility, reuse of an existing axis, color limits and
colorbar, and edge cases such as empty k-path labels. All logic is
documented in the docstrings of each test class and test method.

Routine Listings
----------------
:class:`TestApplyKpathTicks`
    Tests for apply_kpath_ticks.
:class:`TestPlotArpesSpectrum`
    Tests for plot_arpes_spectrum.
:class:`TestPlotArpesWithKpath`
    Tests for plot_arpes_with_kpath.
:func:`_make_spectrum`
    Helper to build a minimal ArpesSpectrum for plotting tests.
"""

import pytest

import chex
import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from arpyes.inout import (
    apply_kpath_ticks,
    plot_arpes_spectrum,
    plot_arpes_with_kpath,
)
from arpyes.types import ArpesSpectrum, make_arpes_spectrum, make_kpath_info


def _make_spectrum(nk=20, ne=120):
    """Build a minimal ArpesSpectrum for plotting tests.

    Creates a valid ArpesSpectrum with intensity shape (nk, ne) and
    energy_axis length ne, so that plot functions receive consistent
    test data without reading files.

    Parameters
    ----------
    nk : int, optional
        Number of k-points (first dimension of intensity). Default 20.
    ne : int, optional
        Number of energy points (second dimension of intensity and
        length of energy_axis). Default 120.

    Returns
    -------
    spectrum : ArpesSpectrum
        PyTree with intensity (nk, ne) and energy_axis (ne,).
    """
    intensity = jnp.linspace(0.0, 1.0, nk * ne, dtype=jnp.float64).reshape(
        nk, ne
    )
    energy_axis = jnp.linspace(-2.0, 0.5, ne)
    return make_arpes_spectrum(
        intensity=intensity,
        energy_axis=energy_axis,
    )


class TestPlotArpesSpectrum(chex.TestCase):
    """Tests for :func:`arpyes.inout.plot_arpes_spectrum`.

    Covers default plotting, optional color limits and colorbar,
    validation of spectrum array dimensions and shape compatibility,
    and reuse of a user-provided matplotlib axis.
    """

    def test_returns_expected_objects(self):
        """Plot with default options returns figure, axis, and image with correct shape and labels.

        Builds a spectrum and calls plot_arpes_spectrum with colorbar=False.
        Asserts that the returned image array has shape (E, K) i.e. (120, 20)
        after transpose, and that the axis has the default xlabel, ylabel,
        and title. The figure is closed after assertions to avoid leaking
        resources.
        """
        spectrum = _make_spectrum()
        fig, ax, image = plot_arpes_spectrum(spectrum, colorbar=False)
        chex.assert_equal(image.get_array().shape, (120, 20))
        chex.assert_equal(ax.get_xlabel(), "k-point index")
        chex.assert_equal(ax.get_ylabel(), "Energy (eV)")
        chex.assert_equal(ax.get_title(), "Simulated ARPES Spectrum")
        plt.close(fig)

    def test_with_clim_and_colorbar(self):
        """Passing clim and colorbar=True applies color limits and adds a colorbar.

        Calls plot_arpes_spectrum with colorbar=True and clim=(0.0, 0.5).
        Asserts that the image's color limits are set to (0.0, 0.5) via
        get_clim(), ensuring the clim and colorbar code paths are exercised.
        """
        spectrum = _make_spectrum()
        fig, ax, image = plot_arpes_spectrum(
            spectrum, colorbar=True, clim=(0.0, 0.5)
        )
        chex.assert_equal(image.get_clim(), (0.0, 0.5))
        plt.close(fig)

    def test_validation_rejects_wrong_intensity_ndim(self):
        """_prepare_plot_arrays raises ValueError when intensity is not 2D.

        Constructs an ArpesSpectrum with 1D intensity (bypassing the
        factory's type checks) and calls plot_arpes_spectrum. Expects
        a ValueError whose message indicates that spectrum.intensity
        must have shape (K, E). This validates the dimension check
        inside the plotting pipeline.
        """
        spectrum = ArpesSpectrum(
            intensity=jnp.linspace(0.0, 1.0, 10),
            energy_axis=jnp.linspace(-1.0, 1.0, 10),
        )
        with pytest.raises(
            ValueError, match="Expected spectrum.intensity to have shape"
        ):
            plot_arpes_spectrum(spectrum, colorbar=False)

    def test_validation_rejects_wrong_energy_axis_ndim(self):
        """_prepare_plot_arrays raises ValueError when energy_axis is not 1D.

        Constructs an ArpesSpectrum with 2D energy_axis and calls
        plot_arpes_spectrum. Expects a ValueError whose message
        indicates that spectrum.energy_axis must have shape (E,).
        """
        spectrum = ArpesSpectrum(
            intensity=jnp.zeros((5, 10)),
            energy_axis=jnp.zeros((10, 2)),
        )
        with pytest.raises(
            ValueError, match="Expected spectrum.energy_axis to have shape"
        ):
            plot_arpes_spectrum(spectrum, colorbar=False)

    def test_validation_rejects_shape_mismatch(self):
        """_prepare_plot_arrays raises ValueError when intensity and energy_axis lengths disagree.

        Uses intensity of shape (5, 10) and energy_axis of length 7, so
        intensity.shape[1] != energy_axis.shape[0]. Expects a ValueError
        with a message about incompatible shapes.
        """
        spectrum = ArpesSpectrum(
            intensity=jnp.zeros((5, 10)),
            energy_axis=jnp.linspace(-1.0, 1.0, 7),
        )
        with pytest.raises(ValueError, match="Incompatible shapes"):
            plot_arpes_spectrum(spectrum, colorbar=False)

    def test_uses_existing_axis(self):
        """When ax is provided, the same figure and axis are returned and used for plotting.

        Creates a figure and axis with plt.subplots(), then passes ax to
        plot_arpes_spectrum. Asserts that the returned figure and axis
        are the same objects as those passed in, and that the image was
        drawn on the provided axis (no new figure created).
        """
        spectrum = _make_spectrum(nk=10, ne=40)
        fig, ax = plt.subplots()
        out_fig, out_ax, _ = plot_arpes_spectrum(
            spectrum, ax=ax, colorbar=False
        )
        chex.assert_equal(out_fig is fig, True)
        chex.assert_equal(out_ax is ax, True)
        plt.close(fig)


class TestApplyKpathTicks(chex.TestCase):
    """Tests for :func:`arpyes.inout.apply_kpath_ticks`.

    Covers application of symmetry-point ticks and labels to an axis,
    behaviour when the number of labels is less than the number of
    label indices, and the early-return path when there are no labels.
    """

    def test_sets_ticks_and_labels(self):
        """apply_kpath_ticks sets x-axis ticks and labels from KPathInfo.

        Builds a KPathInfo with four label indices and four labels
        (G, M, K, G). Applies apply_kpath_ticks to a fresh axis and
        asserts that the x-tick labels after the call are exactly
        ["G", "M", "K", "G"], confirming that indices and labels
        are applied in order.
        """
        fig, ax = plt.subplots()
        kpath = make_kpath_info(
            num_kpoints=60,
            label_indices=[0, 19, 39, 59],
            labels=("G", "M", "K", "G"),
        )
        apply_kpath_ticks(ax, kpath)
        labels = [tick.get_text() for tick in ax.get_xticklabels()]
        chex.assert_equal(labels, ["G", "M", "K", "G"])
        plt.close(fig)

    def test_handles_label_index_mismatch(self):
        """When labels are fewer than label_indices, only the available labels are used.

        Provides four label indices but only two labels (G, M). Asserts
        that the axis ends up with exactly two tick labels, ["G", "M"],
        so that the truncation logic (min(len(indices), len(labels)))
        is exercised and no index error occurs.
        """
        fig, ax = plt.subplots()
        kpath = make_kpath_info(
            num_kpoints=60,
            label_indices=[0, 19, 39, 59],
            labels=("G", "M"),
        )
        apply_kpath_ticks(ax, kpath)
        labels = [tick.get_text() for tick in ax.get_xticklabels()]
        chex.assert_equal(labels, ["G", "M"])
        plt.close(fig)

    def test_empty_labels_returns_ax_unchanged(self):
        """When KPathInfo has no labels, apply_kpath_ticks returns the axis without setting ticks.

        Builds a KPathInfo with empty label_indices and empty labels so
        that n_labels is zero. Asserts that the return value is the
        same axis object and that the early-return path (no tick/label
        setting) is taken without error.
        """
        fig, ax = plt.subplots()
        kpath = make_kpath_info(
            num_kpoints=20,
            label_indices=[],
            labels=(),
        )
        out = apply_kpath_ticks(ax, kpath)
        chex.assert_equal(out is ax, True)
        plt.close(fig)


class TestPlotArpesWithKpath(chex.TestCase):
    """Tests for :func:`arpyes.inout.plot_arpes_with_kpath`.

    Covers the combined workflow of plotting an ARPES spectrum and
    annotating the k-axis with symmetry labels from KPathInfo.
    """

    def test_combined_plot(self):
        """plot_arpes_with_kpath produces a spectrum image and applies k-path ticks and labels.

        Builds a spectrum and a KPathInfo with three symmetry points.
        Calls plot_arpes_with_kpath and asserts that the image array
        has the expected shape (120, 20), that the x-tick labels are
        ("G", "M", "K"), and that the x-axis label is "Momentum (k)",
        confirming that both the spectrum plot and the k-path
        annotation are applied correctly.
        """
        spectrum = _make_spectrum()
        kpath = make_kpath_info(
            num_kpoints=20,
            label_indices=[0, 9, 19],
            labels=("G", "M", "K"),
        )
        fig, ax, image = plot_arpes_with_kpath(
            spectrum=spectrum,
            kpath=kpath,
            colorbar=False,
        )
        chex.assert_equal(image.get_array().shape, (120, 20))
        labels = [tick.get_text() for tick in ax.get_xticklabels()]
        chex.assert_equal(labels, ["G", "M", "K"])
        chex.assert_equal(ax.get_xlabel(), "Momentum (k)")
        plt.close(fig)
