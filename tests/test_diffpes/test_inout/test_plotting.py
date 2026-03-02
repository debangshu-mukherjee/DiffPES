"""Tests for ARPES plotting utilities.

Extended Summary
----------------
Exercises the plotting module's public API: plot_arpes_spectrum,
apply_kpath_ticks, plot_arpes_with_kpath, and projected-band scatter
helpers. Tests cover successful
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
:class:`TestPlotBandScatter`
    Tests for projected-band scatter plotting presets.
:func:`_make_spectrum`
    Helper to build a minimal ArpesSpectrum for plotting tests.
"""

import pytest

import chex
import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from diffpes.inout import (
    apply_kpath_ticks,
    list_band_scatter_presets,
    plot_arpes_spectrum,
    plot_band_scatter_preset,
    plot_band_scatter_with_kpath,
    plot_arpes_with_kpath,
)
from diffpes.types import (
    ArpesSpectrum,
    BandStructure,
    make_arpes_spectrum,
    make_band_structure,
    make_kpath_info,
    make_orbital_projection,
    make_spin_orbital_projection,
)


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
    """Tests for :func:`diffpes.inout.plot_arpes_spectrum`.

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
    """Tests for :func:`diffpes.inout.apply_kpath_ticks`.

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
    """Tests for :func:`diffpes.inout.plot_arpes_with_kpath`.

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


def _make_band_and_projection(nk=12, nb=3, na=2):
    """Build minimal band/projection inputs for band-scatter tests."""
    eigen = jnp.linspace(-1.2, 0.8, nk * nb, dtype=jnp.float64).reshape(nk, nb)
    kx = jnp.linspace(0.0, 1.0, nk, dtype=jnp.float64)
    kpoints = jnp.stack(
        [kx, jnp.zeros_like(kx), jnp.zeros_like(kx)],
        axis=1,
    )
    bands = make_band_structure(
        eigenvalues=eigen,
        kpoints=kpoints,
        fermi_energy=0.15,
    )

    projections = jnp.ones((nk, nb, na, 9), dtype=jnp.float64) * 0.05
    projections = projections.at[..., 1:4].set(0.2)
    spin = jnp.zeros((nk, nb, na, 6), dtype=jnp.float64)
    spin = spin.at[: nk // 2, ..., 4].set(0.2)
    spin = spin.at[nk // 2 :, ..., 5].set(0.3)
    orb = make_orbital_projection(projections=projections, spin=spin)
    return bands, orb


class TestPlotBandScatter(chex.TestCase):
    """Tests for projected-band scatter plotting presets."""

    def test_lists_presets(self):
        """list_band_scatter_presets returns known keys."""
        presets = list_band_scatter_presets()
        assert "p" in presets
        assert "d" in presets
        assert "spin_z" in presets
        assert "oam_total" in presets

    def test_orbital_preset_plot(self):
        """Orbital preset renders a scatter with one point per (k, band)."""
        bands, orb = _make_band_and_projection(nk=10, nb=4, na=2)
        fig, ax, scatter = plot_band_scatter_preset(
            bands=bands,
            orb_proj=orb,
            preset="p",
            colorbar=False,
        )
        chex.assert_equal(scatter.get_offsets().shape[0], 40)
        chex.assert_equal(ax.get_xlabel(), "Momentum (k)")
        chex.assert_equal(ax.get_ylabel(), "Energy (eV)")
        plt.close(fig)

    def test_signed_spin_preset_with_colorbar(self):
        """Signed spin preset can render with colorbar."""
        bands, orb = _make_band_and_projection(nk=8, nb=2, na=2)
        fig, _, _ = plot_band_scatter_preset(
            bands=bands,
            orb_proj=orb,
            preset="spin_z",
            colorbar=True,
        )
        # main axis + colorbar axis
        chex.assert_equal(len(fig.axes), 2)
        plt.close(fig)

    def test_spin_preset_requires_spin_data(self):
        """Spin presets raise when projection has no spin field."""
        bands, orb = _make_band_and_projection(nk=8, nb=2, na=1)
        no_spin = make_orbital_projection(
            projections=orb.projections, spin=None
        )
        with pytest.raises(ValueError, match="requires spin data"):
            plot_band_scatter_preset(
                bands=bands,
                orb_proj=no_spin,
                preset="spin_z",
            )

    def test_band_scatter_with_kpath(self):
        """K-path wrapper applies symmetry labels on projected-band scatter."""
        bands, orb = _make_band_and_projection(nk=10, nb=3, na=2)
        kpath = make_kpath_info(
            num_kpoints=10,
            label_indices=[0, 4, 9],
            labels=("G", "M", "K"),
        )
        fig, ax, scatter = plot_band_scatter_with_kpath(
            bands=bands,
            orb_proj=orb,
            kpath=kpath,
            preset="d",
            colorbar=False,
        )
        chex.assert_equal(scatter.get_offsets().shape[0], 30)
        labels = [tick.get_text() for tick in ax.get_xticklabels()]
        chex.assert_equal(labels, ["G", "M", "K"])
        plt.close(fig)


class TestPlotBandScatterEdgeCases(chex.TestCase):
    """Edge-case tests for the band-scatter plotting helpers.

    Covers remaining uncovered lines in plotting.py:
    - Lines 449-450: ``_prepare_band_arrays`` ValueError for wrong ndim
    - Lines 463-464: ``_subset_atom_axis`` with explicit atom_indices
    - Lines 497-498: orbital-index preset ("s")
    - Line 502: "total" preset
    - Line 528: spin_channel preset ("spin_z_up")
    - Line 541: OAM array subsetting when oam is not None
    - Lines 551-553: OAM preset requires OAM data (oam is None → ValueError)
    - Lines 554-556: oam_component preset branch
    - Lines 562-564: unknown preset → ValueError
    - Lines 644-648: weight shape mismatch → ValueError
    - Line 667: ``fig = ax.figure`` when ax is provided
    """

    def _make_bands_1d(self, nk=4, nb=2):
        """Build BandStructure with 1D eigenvalues (bypassing factory)."""
        return BandStructure(
            eigenvalues=jnp.zeros(nk * nb, dtype=jnp.float64),  # 1D -> wrong ndim
            kpoints=jnp.zeros((nk, 3), dtype=jnp.float64),
            kpoint_weights=jnp.zeros(nk, dtype=jnp.float64),
            fermi_energy=jnp.float64(0.0),
        )

    def _make_orb_with_spin_and_oam(self, nk=4, nb=2, na=1):
        """Build OrbitalProjection with spin and OAM attached."""
        proj = jnp.ones((nk, nb, na, 9), dtype=jnp.float64) * 0.1
        spin = jnp.zeros((nk, nb, na, 6), dtype=jnp.float64)
        spin = spin.at[..., 0].set(0.3)   # spin_x_up non-zero
        spin = spin.at[..., 4].set(0.2)   # spin_z_up non-zero
        oam = jnp.ones((nk, nb, na, 3), dtype=jnp.float64) * 0.05
        return make_orbital_projection(projections=proj, spin=spin, oam=oam)

    def test_prepare_band_arrays_wrong_ndim_raises(self):
        """``_prepare_band_arrays`` with 1D eigenvalues raises ValueError (lines 449-450).

        Constructs a BandStructure with 1D eigenvalues (bypassing the
        factory), then calls ``plot_band_scatter_preset``. Asserts a
        ``ValueError`` matching ``"shape (K, B)"``.
        """
        bands = self._make_bands_1d(nk=4, nb=2)
        proj = jnp.ones((4, 2, 1, 9), dtype=jnp.float64) * 0.1
        orb = make_orbital_projection(projections=proj)
        with pytest.raises(ValueError, match="shape"):
            plot_band_scatter_preset(
                bands=bands, orb_proj=orb, preset="p"
            )

    def test_subset_atom_axis_with_indices(self):
        """Passing atom_indices calls ``_subset_atom_axis`` (lines 463-464).

        Plots the "p" preset with ``atom_indices=[0]`` so that
        ``_subset_atom_axis`` is called with a non-None index array.
        Asserts the scatter point count equals nk * nb.
        """
        nk, nb, na = 6, 2, 2
        eigen = jnp.linspace(-1.0, 0.5, nk * nb, dtype=jnp.float64).reshape(
            nk, nb
        )
        kpoints = jnp.zeros((nk, 3), dtype=jnp.float64)
        bands = make_band_structure(eigenvalues=eigen, kpoints=kpoints)
        proj = jnp.ones((nk, nb, na, 9), dtype=jnp.float64) * 0.1
        orb = make_orbital_projection(projections=proj)
        fig, ax, scatter = plot_band_scatter_preset(
            bands=bands,
            orb_proj=orb,
            preset="p",
            atom_indices=[0],  # triggers _subset_atom_axis
            colorbar=False,
        )
        chex.assert_equal(scatter.get_offsets().shape[0], nk * nb)
        plt.close(fig)

    def test_s_orbital_preset(self):
        """'s' preset uses _ORBITAL_INDEX branch (lines 497-498).

        Calls ``plot_band_scatter_preset`` with ``preset='s'``. Asserts
        that the scatter renders without error and the point count is
        correct (exercises the ``_ORBITAL_INDEX[key]`` branch).
        """
        nk, nb = 6, 2
        eigen = jnp.linspace(-1.0, 0.5, nk * nb, dtype=jnp.float64).reshape(
            nk, nb
        )
        bands = make_band_structure(
            eigenvalues=eigen,
            kpoints=jnp.zeros((nk, 3), dtype=jnp.float64),
        )
        proj = jnp.ones((nk, nb, 1, 9), dtype=jnp.float64) * 0.1
        orb = make_orbital_projection(projections=proj)
        fig, ax, scatter = plot_band_scatter_preset(
            bands=bands, orb_proj=orb, preset="s", colorbar=False
        )
        chex.assert_equal(scatter.get_offsets().shape[0], nk * nb)
        plt.close(fig)

    def test_total_preset(self):
        """'total' preset sums all orbital channels (line 502).

        Calls ``plot_band_scatter_preset`` with ``preset='total'``.
        Asserts the scatter renders and the point count is correct
        (exercises the ``elif key == 'total'`` branch at line 501-502).
        """
        nk, nb = 4, 3
        eigen = jnp.linspace(-1.0, 0.5, nk * nb, dtype=jnp.float64).reshape(
            nk, nb
        )
        bands = make_band_structure(
            eigenvalues=eigen,
            kpoints=jnp.zeros((nk, 3), dtype=jnp.float64),
        )
        proj = jnp.ones((nk, nb, 1, 9), dtype=jnp.float64) * 0.05
        orb = make_orbital_projection(projections=proj)
        fig, ax, scatter = plot_band_scatter_preset(
            bands=bands, orb_proj=orb, preset="total", colorbar=False
        )
        chex.assert_equal(scatter.get_offsets().shape[0], nk * nb)
        plt.close(fig)

    def test_spin_channel_preset(self):
        """'spin_z_up' preset uses the spin_channel branch (line 528).

        Calls ``plot_band_scatter_preset`` with ``preset='spin_z_up'``.
        This exercises the ``if key in spin_channel`` branch (line 527-528).
        Asserts the scatter renders without error.
        """
        nk, nb = 4, 2
        eigen = jnp.linspace(-1.0, 0.5, nk * nb, dtype=jnp.float64).reshape(
            nk, nb
        )
        bands = make_band_structure(
            eigenvalues=eigen,
            kpoints=jnp.zeros((nk, 3), dtype=jnp.float64),
        )
        proj = jnp.ones((nk, nb, 1, 9), dtype=jnp.float64) * 0.1
        spin = jnp.ones((nk, nb, 1, 6), dtype=jnp.float64) * 0.1
        orb = make_orbital_projection(projections=proj, spin=spin)
        fig, ax, scatter = plot_band_scatter_preset(
            bands=bands, orb_proj=orb, preset="spin_z_up", colorbar=False
        )
        chex.assert_equal(scatter.get_offsets().shape[0], nk * nb)
        plt.close(fig)

    def test_oam_preset_with_oam_data(self):
        """OAM preset with OAM data exercises oam_arr subsetting and component branch.

        Calls ``plot_band_scatter_preset`` with ``preset='oam_total'``
        and an OrbitalProjection that has ``oam`` data. This exercises
        line 541 (``oam_arr = _subset_atom_axis(...)``) and lines
        554-556 (``if key in oam_component: weights = ...``).
        """
        nk, nb = 4, 2
        eigen = jnp.linspace(-1.0, 0.5, nk * nb, dtype=jnp.float64).reshape(
            nk, nb
        )
        bands = make_band_structure(
            eigenvalues=eigen,
            kpoints=jnp.zeros((nk, 3), dtype=jnp.float64),
        )
        orb = self._make_orb_with_spin_and_oam(nk=nk, nb=nb, na=1)
        fig, ax, scatter = plot_band_scatter_preset(
            bands=bands, orb_proj=orb, preset="oam_total", colorbar=False
        )
        chex.assert_equal(scatter.get_offsets().shape[0], nk * nb)
        plt.close(fig)

    def test_oam_abs_total_preset(self):
        """'oam_abs_total' preset uses ``np.abs`` on OAM component 2 (lines 557-559).

        Calls ``plot_band_scatter_preset`` with ``preset='oam_abs_total'``
        and an OrbitalProjection with OAM data present. This exercises
        the ``elif key == 'oam_abs_total'`` branch at lines 557-559,
        which computes ``np.sum(np.abs(oam_arr[..., 2]), axis=2)`` and
        sets ``signed = False``. Asserts the scatter renders without error.
        """
        nk, nb = 4, 2
        eigen = jnp.linspace(-1.0, 0.5, nk * nb, dtype=jnp.float64).reshape(
            nk, nb
        )
        bands = make_band_structure(
            eigenvalues=eigen,
            kpoints=jnp.zeros((nk, 3), dtype=jnp.float64),
        )
        orb = self._make_orb_with_spin_and_oam(nk=nk, nb=nb, na=1)
        fig, ax, scatter = plot_band_scatter_preset(
            bands=bands, orb_proj=orb, preset="oam_abs_total", colorbar=False
        )
        chex.assert_equal(scatter.get_offsets().shape[0], nk * nb)
        plt.close(fig)

    def test_oam_preset_without_oam_data_raises(self):
        """OAM preset without OAM data raises ValueError (lines 551-553).

        Calls ``plot_band_scatter_preset`` with ``preset='oam_p'`` but
        provides an OrbitalProjection with ``oam=None``. Asserts
        ``ValueError`` matching ``"requires OAM data"``.
        """
        nk, nb = 4, 2
        eigen = jnp.linspace(-1.0, 0.5, nk * nb, dtype=jnp.float64).reshape(
            nk, nb
        )
        bands = make_band_structure(
            eigenvalues=eigen,
            kpoints=jnp.zeros((nk, 3), dtype=jnp.float64),
        )
        proj = jnp.ones((nk, nb, 1, 9), dtype=jnp.float64) * 0.1
        orb = make_orbital_projection(projections=proj)  # oam=None
        with pytest.raises(ValueError, match="requires OAM data"):
            plot_band_scatter_preset(
                bands=bands, orb_proj=orb, preset="oam_p"
            )

    def test_unknown_preset_raises(self):
        """Unknown preset string raises ValueError (lines 562-564).

        Calls ``plot_band_scatter_preset`` with an unrecognized preset
        name. Asserts ``ValueError`` matching ``"Unknown preset"``.
        """
        nk, nb = 4, 2
        eigen = jnp.linspace(-1.0, 0.5, nk * nb, dtype=jnp.float64).reshape(
            nk, nb
        )
        bands = make_band_structure(
            eigenvalues=eigen,
            kpoints=jnp.zeros((nk, 3), dtype=jnp.float64),
        )
        proj = jnp.ones((nk, nb, 1, 9), dtype=jnp.float64) * 0.1
        orb = make_orbital_projection(projections=proj)
        with pytest.raises(ValueError, match="Unknown preset"):
            plot_band_scatter_preset(
                bands=bands, orb_proj=orb, preset="not_a_real_preset"
            )

    def test_weight_shape_mismatch_raises(self):
        """Weights shape != eigenvalues shape raises ValueError (lines 644-648).

        Creates a BandStructure with shape (4, 2) but an OrbitalProjection
        with shape (4, 3, 1, 9), so the "p" preset produces weights of
        shape (4, 3) != (4, 2). Asserts ``ValueError`` matching
        ``"Preset weights must have shape"``.
        """
        nk = 4
        nb_bands = 2
        nb_proj = 3  # different number of bands
        eigen = jnp.linspace(-1.0, 0.5, nk * nb_bands, dtype=jnp.float64).reshape(
            nk, nb_bands
        )
        bands = make_band_structure(
            eigenvalues=eigen,
            kpoints=jnp.zeros((nk, 3), dtype=jnp.float64),
        )
        proj = jnp.ones((nk, nb_proj, 1, 9), dtype=jnp.float64) * 0.1
        orb = make_orbital_projection(projections=proj)
        with pytest.raises(ValueError, match="Preset weights must have shape"):
            plot_band_scatter_preset(
                bands=bands, orb_proj=orb, preset="p"
            )

    def test_uses_provided_ax(self):
        """When ax is provided, ``fig = ax.figure`` is used (line 667).

        Creates a figure and axis, passes them to ``plot_band_scatter_preset``,
        and asserts the returned figure is the same object, confirming
        that line 667 (``fig = ax.figure``) is executed instead of
        ``fig, ax = plt.subplots()``.
        """
        nk, nb = 6, 2
        eigen = jnp.linspace(-1.0, 0.5, nk * nb, dtype=jnp.float64).reshape(
            nk, nb
        )
        bands = make_band_structure(
            eigenvalues=eigen,
            kpoints=jnp.zeros((nk, 3), dtype=jnp.float64),
        )
        proj = jnp.ones((nk, nb, 1, 9), dtype=jnp.float64) * 0.1
        orb = make_orbital_projection(projections=proj)
        fig0, ax0 = plt.subplots()
        out_fig, out_ax, _ = plot_band_scatter_preset(
            bands=bands, orb_proj=orb, preset="p", ax=ax0, colorbar=False
        )
        chex.assert_equal(out_fig is fig0, True)
        plt.close(fig0)
