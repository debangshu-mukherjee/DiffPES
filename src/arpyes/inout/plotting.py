"""Plotting utilities for ARPES spectra.

Extended Summary
----------------
Provides Matplotlib helper functions that consume an
:class:`~arpyes.types.ArpesSpectrum` PyTree directly and render
publication-style ARPES intensity maps.

Routine Listings
----------------
:func:`apply_kpath_ticks`
    Apply symmetry-point tick labels from KPathInfo.
:func:`plot_arpes_spectrum`
    Plot an ARPES intensity map from an ArpesSpectrum.
:func:`plot_arpes_with_kpath`
    Plot spectrum and annotate the k-axis with symmetry labels.

Notes
-----
These functions operate on host-side NumPy arrays and Matplotlib
objects (not JAX-traced arrays). They are intended for visualization
at analysis time, not for inclusion inside ``jax.jit``-compiled
functions.
"""

import numpy as np
from beartype import beartype
from beartype.typing import Literal, Optional, Tuple
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure, SubFigure
from matplotlib.image import AxesImage

from arpyes.types import ArpesSpectrum, KPathInfo

_INTENSITY_NDIM: int = 2
_ENERGY_AXIS_NDIM: int = 1


@beartype
def _prepare_plot_arrays(
    spectrum: ArpesSpectrum,
) -> Tuple[np.ndarray, np.ndarray]:
    """Convert and validate spectrum arrays for plotting.

    Extended Summary
    ----------------
    Internal helper that normalizes an :class:`ArpesSpectrum` PyTree
    into plain NumPy arrays and verifies that array ranks and lengths
    are consistent with a 2D ARPES image.

    Implementation Logic
    --------------------
    1. Convert ``spectrum.intensity`` and ``spectrum.energy_axis`` to
       ``np.float64`` arrays using ``np.asarray``.
    2. Validate dimensions:
       ``intensity.ndim == 2`` and ``energy_axis.ndim == 1``.
    3. Validate shape compatibility:
       ``intensity.shape[1] == energy_axis.shape[0]``.
    4. Return normalized arrays for downstream plotting.

    Parameters
    ----------
    spectrum : ArpesSpectrum
        Input spectrum containing ``intensity`` and ``energy_axis``.

    Returns
    -------
    intensity : np.ndarray
        2D intensity array of shape ``(K, E)``.
    energy_axis : np.ndarray
        1D energy axis array of shape ``(E,)``.

    Raises
    ------
    ValueError
        If array ranks are invalid or if intensity/energy sizes are
        incompatible.
    """
    intensity: np.ndarray = np.asarray(spectrum.intensity, dtype=np.float64)
    energy_axis: np.ndarray = np.asarray(
        spectrum.energy_axis, dtype=np.float64
    )
    if intensity.ndim != _INTENSITY_NDIM:
        msg: str = "Expected spectrum.intensity to have shape (K, E)."
        raise ValueError(msg)
    if energy_axis.ndim != _ENERGY_AXIS_NDIM:
        msg = "Expected spectrum.energy_axis to have shape (E,)."
        raise ValueError(msg)
    if intensity.shape[1] != energy_axis.shape[0]:
        msg = (
            "Incompatible shapes: intensity.shape[1] must equal "
            "energy_axis.shape[0]."
        )
        raise ValueError(msg)
    return intensity, energy_axis


@beartype
def plot_arpes_spectrum(
    spectrum: ArpesSpectrum,
    ax: Optional[Axes] = None,
    cmap: str = "gray",
    colorbar: bool = True,
    clim: Optional[tuple[float, float]] = None,
    interpolation: str = "nearest",
    aspect: Literal["equal", "auto"] = "auto",
    xlabel: str = "k-point index",
    ylabel: str = "Energy (eV)",
    title: str = "Simulated ARPES Spectrum",
) -> Tuple[Figure | SubFigure, Axes, AxesImage]:
    """Plot an ARPES intensity map from an ArpesSpectrum PyTree.

    Extended Summary
    ----------------
    Renders a 2D ARPES map using ``matplotlib.axes.Axes.imshow`` with
    energy on the vertical axis and k-point index on the horizontal
    axis. The function accepts an existing axis or creates a new figure
    and axis when none is supplied.

    Implementation Logic
    --------------------
    1. Normalize and validate arrays via :func:`_prepare_plot_arrays`.
    2. Create a new figure/axis pair if ``ax`` is ``None``; otherwise
       reuse the provided axis and its parent figure.
    3. Compute plotting bounds from data:
       x-range ``[0, K-1]`` and y-range ``[min(E), max(E)]``.
    4. Draw ``intensity.T`` with ``origin="lower"`` so lower energies
       appear at the bottom and k-index increases left to right.
    5. Optionally apply color limits and add a labeled colorbar.
    6. Set axis labels and title, then return figure, axis, and image.

    Parameters
    ----------
    spectrum : ArpesSpectrum
        Spectrum containing ``intensity`` of shape ``(K, E)`` and
        ``energy_axis`` of shape ``(E,)``.
    ax : Optional[Axes], optional
        Existing axis to draw on. If None, a new figure/axis is created.
    cmap : str, optional
        Matplotlib colormap name. Default is ``"gray"``.
    colorbar : bool, optional
        If True, add a colorbar labeled ``"Intensity (a.u.)"``.
    clim : Optional[tuple[float, float]], optional
        Optional ``(vmin, vmax)`` color limits.
    interpolation : str, optional
        Image interpolation mode. Default is ``"nearest"``.
    aspect : str, optional
        Image aspect ratio passed to ``imshow``. Default is ``"auto"``.
    xlabel : str, optional
        x-axis label text. Default is ``"k-point index"``.
    ylabel : str, optional
        y-axis label text. Default is ``"Energy (eV)"``.
    title : str, optional
        Axis title text. Default is ``"Simulated ARPES Spectrum"``.

    Returns
    -------
    fig : Figure
        Matplotlib figure object.
    ax : Axes
        Axis used for plotting.
    image : AxesImage
        Image artist created by ``imshow``.

    Notes
    -----
    The transpose ``intensity.T`` is intentional:
    input data is ``(K, E)``, while ``imshow`` expects rows to map to
    the y-axis. Transposition maps energy to y and k-index to x.
    """
    intensity: np.ndarray
    energy_axis: np.ndarray
    intensity, energy_axis = _prepare_plot_arrays(spectrum)

    fig: Figure | SubFigure
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    nkpoints: int = intensity.shape[0]
    x_max: float = float(max(nkpoints - 1, 0))
    e_min: float = float(np.min(energy_axis))
    e_max: float = float(np.max(energy_axis))

    image: AxesImage = ax.imshow(
        intensity.T,
        origin="lower",
        aspect=aspect,
        cmap=cmap,
        interpolation=interpolation,
        extent=(0.0, x_max, e_min, e_max),
    )
    if clim is not None:
        image.set_clim(clim[0], clim[1])
    if colorbar:
        fig.colorbar(image, ax=ax, label="Intensity (a.u.)")

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    return fig, ax, image


@beartype
def apply_kpath_ticks(
    ax: Axes,
    kpath: KPathInfo,
    draw_symmetry_lines: bool = True,
    line_color: str = "white",
    line_width: float = 0.5,
    line_alpha: float = 0.35,
) -> Axes:
    """Apply symmetry-point ticks/labels from KPathInfo to an axis.

    Extended Summary
    ----------------
    Adds k-path symmetry labels (e.g., G, M, K) to an existing axis
    using :class:`KPathInfo`. Optionally draws vertical guide lines at
    interior symmetry points to visually separate path segments.

    Implementation Logic
    --------------------
    1. Convert ``kpath.label_indices`` to a Python list of ints.
    2. Convert ``kpath.labels`` to a Python list of strings.
    3. Truncate to the shorter list length to tolerate minor metadata
       mismatches without raising.
    4. Apply ticks and tick labels to the x-axis.
    5. Optionally draw interior vertical lines at ticks excluding the
       first and last symmetry points.

    Parameters
    ----------
    ax : Axes
        Target axis.
    kpath : KPathInfo
        K-path metadata containing symmetry labels and their indices.
    draw_symmetry_lines : bool, optional
        If True, draw vertical guide lines at interior symmetry points.
    line_color : str, optional
        Color of symmetry guide lines.
    line_width : float, optional
        Width of symmetry guide lines.
    line_alpha : float, optional
        Alpha of symmetry guide lines.

    Returns
    -------
    ax : Axes
        The same axis, modified in place.

    Notes
    -----
    This function mutates ``ax`` and returns it for convenient chaining.
    """
    indices: list[int] = np.asarray(
        kpath.label_indices, dtype=np.int32
    ).tolist()
    labels: list[str] = list(kpath.labels)
    n_labels: int = min(len(indices), len(labels))
    if n_labels == 0:
        return ax

    ticks: list[float] = [float(idx) for idx in indices[:n_labels]]
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels[:n_labels])

    if draw_symmetry_lines:
        for tick in ticks[1:-1]:
            ax.axvline(
                tick,
                color=line_color,
                linewidth=line_width,
                alpha=line_alpha,
            )
    return ax


@beartype
def plot_arpes_with_kpath(  # noqa: PLR0913
    spectrum: ArpesSpectrum,
    kpath: KPathInfo,
    ax: Optional[Axes] = None,
    cmap: str = "gray",
    colorbar: bool = True,
    clim: Optional[tuple[float, float]] = None,
    interpolation: str = "nearest",
    aspect: Literal["equal", "auto"] = "auto",
    xlabel: str = "Momentum (k)",
    ylabel: str = "Energy (eV)",
    title: str = "Simulated ARPES Spectrum",
    draw_symmetry_lines: bool = True,
) -> Tuple[Figure | SubFigure, Axes, AxesImage]:
    """Plot ARPES spectrum and annotate k-axis using KPathInfo.

    Extended Summary
    ----------------
    Convenience wrapper combining :func:`plot_arpes_spectrum` and
    :func:`apply_kpath_ticks` in one call. Useful for line-mode band
    paths where symmetry labels should be shown directly on the ARPES
    image.

    Implementation Logic
    --------------------
    1. Delegate base image rendering to :func:`plot_arpes_spectrum`
       using the provided visual styling arguments.
    2. Apply k-path ticks/labels (and optional symmetry guide lines)
       via :func:`apply_kpath_ticks`.
    3. Return the same figure/axis/image triple from the base plot.

    Parameters
    ----------
    spectrum : ArpesSpectrum
        Spectrum to plot.
    kpath : KPathInfo
        Symmetry-point metadata used for x-axis annotation.
    ax : Optional[Axes], optional
        Existing axis to draw on. If None, create a new one.
    cmap : str, optional
        Matplotlib colormap name. Default is ``"gray"``.
    colorbar : bool, optional
        If True, add a colorbar.
    clim : Optional[tuple[float, float]], optional
        Optional ``(vmin, vmax)`` color limits.
    interpolation : str, optional
        Image interpolation mode for ``imshow``.
    aspect : str, optional
        Image aspect ratio for ``imshow``.
    xlabel : str, optional
        x-axis label. Default is ``"Momentum (k)"``.
    ylabel : str, optional
        y-axis label. Default is ``"Energy (eV)"``.
    title : str, optional
        Plot title.
    draw_symmetry_lines : bool, optional
        If True, draw vertical guide lines at interior symmetry points.

    Returns
    -------
    fig : Figure
        Matplotlib figure object.
    ax : Axes
        Axis used for plotting.
    image : AxesImage
        Image artist created by ``imshow``.

    See Also
    --------
    plot_arpes_spectrum : Base ARPES heatmap renderer.
    apply_kpath_ticks : K-path label and guide-line utility.
    """
    fig: Figure | SubFigure
    image: AxesImage
    fig, ax, image = plot_arpes_spectrum(
        spectrum=spectrum,
        ax=ax,
        cmap=cmap,
        colorbar=colorbar,
        clim=clim,
        interpolation=interpolation,
        aspect=aspect,
        xlabel=xlabel,
        ylabel=ylabel,
        title=title,
    )
    apply_kpath_ticks(
        ax=ax,
        kpath=kpath,
        draw_symmetry_lines=draw_symmetry_lines,
    )
    return fig, ax, image


__all__: list[str] = [
    "apply_kpath_ticks",
    "plot_arpes_spectrum",
    "plot_arpes_with_kpath",
]
