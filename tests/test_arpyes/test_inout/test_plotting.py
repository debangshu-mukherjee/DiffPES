"""Tests for ARPES plotting utilities."""

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
    intensity = jnp.linspace(
        0.0, 1.0, nk * ne, dtype=jnp.float64
    ).reshape(nk, ne)
    energy_axis = jnp.linspace(-2.0, 0.5, ne)
    return make_arpes_spectrum(
        intensity=intensity,
        energy_axis=energy_axis,
    )


class TestPlotArpesSpectrum(chex.TestCase):

    def test_returns_expected_objects(self):
        spectrum = _make_spectrum()
        fig, ax, image = plot_arpes_spectrum(
            spectrum, colorbar=False
        )
        chex.assert_equal(image.get_array().shape, (120, 20))
        chex.assert_equal(ax.get_xlabel(), "k-point index")
        chex.assert_equal(ax.get_ylabel(), "Energy (eV)")
        chex.assert_equal(
            ax.get_title(), "Simulated ARPES Spectrum"
        )
        plt.close(fig)

    def test_with_clim_and_colorbar(self):
        """Cover clim and colorbar branches in plot_arpes_spectrum."""
        spectrum = _make_spectrum()
        fig, ax, image = plot_arpes_spectrum(
            spectrum, colorbar=True, clim=(0.0, 0.5)
        )
        chex.assert_equal(image.get_clim(), (0.0, 0.5))
        plt.close(fig)

    def test_validation_rejects_wrong_intensity_ndim(self):
        """_prepare_plot_arrays raises when intensity is not 2D."""
        spectrum = ArpesSpectrum(
            intensity=jnp.linspace(0.0, 1.0, 10),
            energy_axis=jnp.linspace(-1.0, 1.0, 10),
        )
        with pytest.raises(ValueError, match="Expected spectrum.intensity to have shape"):
            plot_arpes_spectrum(spectrum, colorbar=False)

    def test_validation_rejects_wrong_energy_axis_ndim(self):
        """_prepare_plot_arrays raises when energy_axis is not 1D."""
        spectrum = ArpesSpectrum(
            intensity=jnp.zeros((5, 10)),
            energy_axis=jnp.zeros((10, 2)),
        )
        with pytest.raises(ValueError, match="Expected spectrum.energy_axis to have shape"):
            plot_arpes_spectrum(spectrum, colorbar=False)

    def test_validation_rejects_shape_mismatch(self):
        """_prepare_plot_arrays raises when intensity.shape[1] != energy_axis.shape[0]."""
        spectrum = ArpesSpectrum(
            intensity=jnp.zeros((5, 10)),
            energy_axis=jnp.linspace(-1.0, 1.0, 7),
        )
        with pytest.raises(ValueError, match="Incompatible shapes"):
            plot_arpes_spectrum(spectrum, colorbar=False)

    def test_uses_existing_axis(self):
        spectrum = _make_spectrum(nk=10, ne=40)
        fig, ax = plt.subplots()
        out_fig, out_ax, _ = plot_arpes_spectrum(
            spectrum, ax=ax, colorbar=False
        )
        chex.assert_equal(out_fig is fig, True)
        chex.assert_equal(out_ax is ax, True)
        plt.close(fig)


class TestApplyKpathTicks(chex.TestCase):

    def test_sets_ticks_and_labels(self):
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
        """apply_kpath_ticks with no labels returns ax without setting ticks (n_labels==0)."""
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

    def test_combined_plot(self):
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
