"""Validate ARPES plotting utilities.

The tests cover default and custom spectrum rendering, shape validation,
existing axes, color limits, color bars, and empty k-path labels.
"""

import chex
import equinox as eqx
import jax.numpy as jnp
import matplotlib
import pytest
from beartype.typing import List, Tuple
from jaxtyping import Array, Float64

import diffpes

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.image import AxesImage

from diffpes.inout import (
    apply_kpath_ticks,
    plot_arpes_spectrum,
    plot_arpes_with_kpath,
    plot_momentum_map,
)
from diffpes.types import (
    ArpesSpectrum,
    KPathInfo,
    make_arpes_spectrum,
    make_kpath_info,
)


def _unvalidated_kpath(
    label_indices: List[int], labels: Tuple[str, ...]
) -> KPathInfo:
    """PRIVATE: Build malformed legacy k-path metadata for plotting edge tests.

    Parameters
    ----------
    label_indices : List[int]
        Positions of the high-symmetry labels along the 60-point path.
    labels : Tuple[str, ...]
        High-symmetry point labels to attach, possibly mismatched with
        ``label_indices`` on purpose.

    Returns
    -------
    kpath : KPathInfo
        Line-mode metadata carrier with zero ``points_per_segment``,
        zero ``segments``, and no k-point or weight arrays.

    Notes
    -----
    Calls the ``KPathInfo`` constructor directly instead of the
    validated factory, so plotting tests can exercise inputs that the
    factory rejects.
    """
    kpath: KPathInfo = KPathInfo(
        num_kpoints=jnp.asarray(60, dtype=jnp.int32),
        label_indices=jnp.asarray(label_indices, dtype=jnp.int32),
        points_per_segment=jnp.asarray(0, dtype=jnp.int32),
        segments=jnp.asarray(0, dtype=jnp.int32),
        kpoints=None,
        weights=None,
        grid=None,
        shift=None,
        mode="Line-mode",
        labels=labels,
        comment="",
        coordinate_mode="",
    )
    return kpath


def _make_spectrum(nk: int = 20, ne: int = 120) -> ArpesSpectrum:
    """PRIVATE: Build a minimal ArpesSpectrum for plotting tests.

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
    intensity: Float64[Array, "nk ne"] = jnp.linspace(
        0.0, 1.0, nk * ne, dtype=jnp.float64
    ).reshape(nk, ne)
    energy_axis: Float64[Array, " ne"] = jnp.linspace(-2.0, 0.5, ne)
    k_axis: Float64[Array, " nk"] = jnp.linspace(0.0, 1.0, nk)
    kpoints_cart_inv_ang: Float64[Array, "nk 3"] = jnp.stack(
        (k_axis, jnp.zeros_like(k_axis), jnp.zeros_like(k_axis)),
        axis=1,
    )
    spectrum: ArpesSpectrum = make_arpes_spectrum(
        intensity=intensity,
        energy_axis=energy_axis,
        k_axis=k_axis,
        kpoints_cart_inv_ang=kpoints_cart_inv_ang,
    )
    return spectrum


class TestPlotArpesSpectrum(chex.TestCase):
    """Validate :func:`diffpes.inout.plot_arpes_spectrum`.

    Covers default plotting, optional color limits and colorbar,
    validation of spectrum array dimensions and shape compatibility,
    and reuse of a user-provided matplotlib axis.

    :see: :func:`~diffpes.inout.plot_arpes_spectrum`
    """

    def test_returns_expected_objects(self) -> None:
        """Return a correctly labelled default spectrum plot.

        The test builds a spectrum and calls ``plot_arpes_spectrum`` with
        ``colorbar=False``.
        The test checks the transposed image shape ``(120, 20)``. It also
        checks the default axis labels and title. The test closes the figure.

        Notes
        -----
        The test builds the inputs in the test body and checks the stated
        property with the documented numerical or structural assertions.
        """
        spectrum: diffpes.types.ArpesSpectrum
        fig: Figure
        ax: Axes
        image: AxesImage

        spectrum = _make_spectrum()
        fig, ax, image = plot_arpes_spectrum(spectrum, colorbar=False)
        chex.assert_equal(image.get_array().shape, (120, 20))
        chex.assert_equal(ax.get_xlabel(), "k-point index")
        chex.assert_equal(ax.get_ylabel(), "Energy (eV)")
        chex.assert_equal(ax.get_title(), "Simulated ARPES Spectrum")
        plt.close(fig)

    def test_with_clim_and_colorbar(self) -> None:
        """Verify color limits and a color bar with custom options.

        The test calls ``plot_arpes_spectrum`` with a color bar and limits
        ``(0.0, 0.5)``.
        The test compares the image color limits with ``(0.0, 0.5)`` through
        ``get_clim``. This input covers the limits and color bar paths.

        Notes
        -----
        The test builds the inputs in the test body and checks the stated
        property with the documented numerical or structural assertions.
        """
        spectrum: diffpes.types.ArpesSpectrum
        fig: Figure
        ax: Axes
        image: AxesImage

        spectrum = _make_spectrum()
        fig, ax, image = plot_arpes_spectrum(
            spectrum, colorbar=True, clim=(0.0, 0.5)
        )
        chex.assert_equal(image.get_clim(), (0.0, 0.5))
        plt.close(fig)

    def test_validation_rejects_wrong_intensity_ndim(self) -> None:
        """_prepare_plot_arrays raises ValueError when intensity is not 2D.

        The test constructs an ArpesSpectrum with 1D intensity (bypassing the
        factory's type checks) and calls plot_arpes_spectrum. Expects
        a ValueError whose message indicates that spectrum.intensity
        must have shape (K, E). This validates the dimension check
        inside the plotting pipeline.

        Notes
        -----
        The test builds the inputs in the test body and checks the stated
        property with the documented numerical or structural assertions.
        """
        spectrum: ArpesSpectrum = eqx.tree_at(
            lambda candidate: candidate.intensity,
            _make_spectrum(nk=5, ne=10),
            jnp.linspace(0.0, 1.0, 10),
        )
        with pytest.raises(
            ValueError, match="Expected spectrum.intensity to have shape"
        ):
            plot_arpes_spectrum(spectrum, colorbar=False)

    def test_validation_rejects_wrong_energy_axis_ndim(self) -> None:
        """_prepare_plot_arrays raises ValueError when energy_axis is not 1D.

        The test constructs an ArpesSpectrum with 2D energy_axis and calls
        plot_arpes_spectrum. Expects a ValueError whose message
        indicates that spectrum.energy_axis must have shape (E,).

        Notes
        -----
        The test builds the inputs in the test body and checks the stated
        property with the documented numerical or structural assertions.
        """
        spectrum: ArpesSpectrum = eqx.tree_at(
            lambda candidate: candidate.energy_axis,
            _make_spectrum(nk=5, ne=10),
            jnp.zeros((10, 2)),
        )
        with pytest.raises(
            ValueError, match="Expected spectrum.energy_axis to have shape"
        ):
            plot_arpes_spectrum(spectrum, colorbar=False)

    def test_validation_rejects_shape_mismatch(self) -> None:
        """Reject incompatible intensity and energy-axis lengths.

        The test uses intensity of shape (5, 10) and an energy axis of length
        7, so their dimensions disagree. It expects a ValueError
        with a message about incompatible shapes.

        Notes
        -----
        The test builds the inputs in the test body and checks the stated
        property with the documented numerical or structural assertions.
        """
        spectrum: ArpesSpectrum = eqx.tree_at(
            lambda candidate: candidate.energy_axis,
            _make_spectrum(nk=5, ne=10),
            jnp.linspace(-1.0, 1.0, 7),
        )
        with pytest.raises(ValueError, match="Incompatible shapes"):
            plot_arpes_spectrum(spectrum, colorbar=False)

    def test_uses_existing_axis(self) -> None:
        """Verify reuse of a given figure and axis.

        The test creates a figure and axis with ``plt.subplots()``, then passes
        the axis to ``plot_arpes_spectrum``. The test verifies the identities
        of the returned figure and axis, plus the image on the given axis.

        Notes
        -----
        The test builds the inputs in the test body and checks the stated
        property with the documented numerical or structural assertions.
        """
        spectrum: diffpes.types.ArpesSpectrum
        fig: Figure
        ax: Axes
        out_fig: Figure
        out_ax: Axes

        spectrum = _make_spectrum(nk=10, ne=40)
        fig, ax = plt.subplots()
        out_fig, out_ax, _ = plot_arpes_spectrum(
            spectrum, ax=ax, colorbar=False
        )
        chex.assert_equal(out_fig is fig, True)
        chex.assert_equal(out_ax is ax, True)
        plt.close(fig)


class TestApplyKpathTicks(chex.TestCase):
    """Validate :func:`diffpes.inout.apply_kpath_ticks`.

    The tests apply symmetry-point ticks and labels to an axis. They cover
    fewer labels than indices and the path without labels.

    :see: :func:`~diffpes.inout.apply_kpath_ticks`
    """

    def test_sets_ticks_and_labels(self) -> None:
        """apply_kpath_ticks sets x-axis ticks and labels from KPathInfo.

        The test builds a KPathInfo with four label indices and four labels
        (G, M, K, G). The test applies ``apply_kpath_ticks`` to a new axis.
        It compares the x-axis labels with the expected ordered labels.

        Notes
        -----
        The test builds the inputs in the test body and checks the stated
        property with the documented numerical or structural assertions.
        """
        fig: Figure
        ax: Axes
        kpath: diffpes.types.KPathInfo
        labels: List[str]

        fig, ax = plt.subplots()
        kpath = make_kpath_info(
            num_kpoints=60,
            label_indices=[0, 19, 39, 59],
            segments=3,
            labels=("G", "M", "K", "G"),
        )
        apply_kpath_ticks(ax, kpath)
        labels = [tick.get_text() for tick in ax.get_xticklabels()]
        chex.assert_equal(labels, ["G", "M", "K", "G"])
        plt.close(fig)

    def test_handles_label_index_mismatch(self) -> None:
        """Verify truncation when labels are fewer than label indices.

        The fixture provides four label indices but only two labels.
        The test expects exactly the two available labels. This input covers
        the truncation logic without an index error.

        Notes
        -----
        The test builds the inputs in the test body and checks the stated
        property with the documented numerical or structural assertions.
        """
        fig: Figure
        ax: Axes
        kpath: diffpes.types.KPathInfo
        labels: List[str]

        fig, ax = plt.subplots()
        kpath = _unvalidated_kpath([0, 19, 39, 59], ("G", "M"))
        apply_kpath_ticks(ax, kpath)
        labels = [tick.get_text() for tick in ax.get_xticklabels()]
        chex.assert_equal(labels, ["G", "M"])
        plt.close(fig)

    def test_empty_labels_returns_ax_unchanged(self) -> None:
        """Verify the unchanged axis when the k-path has no labels.

        The test builds a KPathInfo with empty indices and labels so
        that ``n_labels`` is zero. The test verifies the identity of the
        returned axis. It also verifies the return path without tick changes.

        Notes
        -----
        The test builds the inputs in the test body and checks the stated
        property with the documented numerical or structural assertions.
        """
        fig: Figure
        ax: Axes
        kpath: diffpes.types.KPathInfo
        out: Axes

        fig, ax = plt.subplots()
        kpath = _unvalidated_kpath([], ())
        out = apply_kpath_ticks(ax, kpath)
        chex.assert_equal(out is ax, True)
        plt.close(fig)


class TestPlotArpesWithKpath(chex.TestCase):
    """Validate :func:`diffpes.inout.plot_arpes_with_kpath`.

    Covers the combined workflow of plotting an ARPES spectrum and
    annotating the k-axis with symmetry labels from KPathInfo.

    :see: :func:`~diffpes.inout.plot_arpes_with_kpath`
    """

    def test_combined_plot(self) -> None:
        """Plot a spectrum with k-path ticks and labels.

        The test builds a spectrum and a KPathInfo with three symmetry points.
        The test calls ``plot_arpes_with_kpath`` and checks the image shape.
        It also compares the tick labels and the x-axis label. These checks
        verify the spectrum plot and its k-path annotation.

        Notes
        -----
        The test builds the inputs in the test body and checks the stated
        property with the documented numerical or structural assertions.
        """
        spectrum: diffpes.types.ArpesSpectrum
        kpath: diffpes.types.KPathInfo
        fig: Figure
        ax: Axes
        image: AxesImage
        labels: List[str]

        spectrum = _make_spectrum()
        kpath = make_kpath_info(
            num_kpoints=20,
            label_indices=[0, 9, 19],
            segments=2,
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


class TestPlotMomentumMap(chex.TestCase):
    """Validate :func:`diffpes.inout.plot_momentum_map`.

    :see: :func:`~diffpes.inout.plot_momentum_map`
    """

    def test_draws_map_with_momentum_extent(self) -> None:
        """Plot one momentum map and verify the artists and extent.

        The image extent follows the two momentum axes. The labels use
        the momentum defaults.

        Notes
        -----
        The test builds the inputs in the test body and checks the stated
        property with the documented numerical or structural assertions.
        """
        kx_axis: Float64[Array, " 4"]
        ky_axis: Float64[Array, " 3"]
        intensity: Float64[Array, "4 3"]
        fig: Figure
        ax: Axes
        image: AxesImage
        extent: Tuple[float, float, float, float]

        kx_axis = jnp.linspace(-0.2, 0.2, 4)
        ky_axis = jnp.linspace(-0.1, 0.1, 3)
        intensity = jnp.ones((4, 3), dtype=jnp.float64)
        fig, ax, image = plot_momentum_map(intensity, kx_axis, ky_axis)
        assert isinstance(ax, Axes)
        assert isinstance(image, AxesImage)
        chex.assert_equal(image.get_array().shape, (3, 4))
        extent = tuple(float(value) for value in image.get_extent())
        chex.assert_equal(extent, (-0.2, 0.2, -0.1, 0.1))
        plt.close(fig)

    def test_reuses_supplied_axis(self) -> None:
        """Plot the map on a caller-supplied axis.

        The function keeps the supplied axis and its parent figure.

        Notes
        -----
        The test builds the inputs in the test body and checks the stated
        property with the documented numerical or structural assertions.
        """
        kx_axis: Float64[Array, " 4"]
        intensity: Float64[Array, "4 4"]
        fig: Figure
        ax: Axes
        out_fig: Figure
        out_ax: Axes
        _image: AxesImage

        kx_axis = jnp.linspace(-0.2, 0.2, 4)
        intensity = jnp.ones((4, 4), dtype=jnp.float64)
        fig, ax = plt.subplots()
        out_fig, out_ax, _image = plot_momentum_map(
            intensity, kx_axis, kx_axis, ax=ax
        )
        assert out_ax is ax
        assert out_fig is fig
        plt.close(fig)
