r"""Plot ARPES spectra with analysis utilities.

Extended Summary
----------------
The module provides Matplotlib functions that render publication-style ARPES
intensity maps. Inputs use the ArpesSpectrum carrier directly.

Routine Listings
----------------
:func:`apply_kpath_ticks`
    Apply symmetry-point ticks/labels from KPathInfo to an axis.
:func:`plot_arpes_spectrum`
    Plot an ARPES intensity map from an ArpesSpectrum PyTree.
:func:`plot_arpes_with_kpath`
    Plot ARPES spectrum and annotate k-axis using KPathInfo.
:func:`plot_momentum_map`
    Plot a momentum-momentum intensity map with Cartesian axes.

Notes
-----
These functions operate on host-side NumPy arrays and Matplotlib objects.
Use them for analysis visualizations outside JAX-compiled functions.
"""

import numpy as np
from beartype import beartype
from beartype.typing import List, Literal, Optional, Tuple, Union
from jaxtyping import Array, Float64, jaxtyped
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure, SubFigure
from matplotlib.image import AxesImage
from numpy.typing import NDArray

from diffpes.constants import ENERGY_AXIS_NDIM, INTENSITY_NDIM
from diffpes.types import ArpesSpectrum, KPathInfo


@beartype
def _prepare_plot_arrays(
    spectrum: ArpesSpectrum,
) -> Tuple[Float64[NDArray, "K E"], Float64[NDArray, " E"]]:
    """PRIVATE: Convert and validate spectrum arrays for plotting.

    The helper converts an :class:`ArpesSpectrum` PyTree to NumPy arrays. It
    verifies the array ranks and lengths for a two-dimensional ARPES image.

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
    intensity : Float64[NDArray, "K E"]
        2D intensity array of shape ``(K, E)``.
    energy_axis : Float64[NDArray, " E"]
        1D energy axis array of shape ``(E,)``.

    Raises
    ------
    ValueError
        If array ranks are invalid or if intensity/energy sizes are
        incompatible.
    """
    intensity: Float64[NDArray, "K E"] = np.asarray(
        spectrum.intensity, dtype=np.float64
    )
    energy_axis: Float64[NDArray, " E"] = np.asarray(
        spectrum.energy_axis, dtype=np.float64
    )
    if intensity.ndim != INTENSITY_NDIM:
        msg: str = "Expected spectrum.intensity to have shape (K, E)."
        raise ValueError(msg)
    if energy_axis.ndim != ENERGY_AXIS_NDIM:
        msg = "Expected spectrum.energy_axis to have shape (E,)."
        raise ValueError(msg)
    if intensity.shape[1] != energy_axis.shape[0]:
        msg = (
            "Incompatible shapes: intensity.shape[1] must equal "
            "energy_axis.shape[0]."
        )
        raise ValueError(msg)
    plot_arrays: Tuple[Float64[NDArray, "K E"], Float64[NDArray, " E"]] = (
        intensity,
        energy_axis,
    )
    return plot_arrays


@jaxtyped(typechecker=beartype)
def plot_arpes_spectrum(
    spectrum: ArpesSpectrum,
    ax: Optional[Axes] = None,
    cmap: str = "gray",
    colorbar: bool = True,
    clim: Optional[Tuple[float, float]] = None,
    interpolation: str = "nearest",
    aspect: Literal["equal", "auto"] = "auto",
    xlabel: str = "k-point index",
    ylabel: str = "Energy (eV)",
    title: str = "Simulated ARPES Spectrum",
) -> Tuple[Union[Figure, SubFigure], Axes, AxesImage]:
    """Plot an ARPES intensity map from an ArpesSpectrum PyTree.

    The function renders a two-dimensional ARPES map with
    ``matplotlib.axes.Axes.imshow``. The vertical axis contains energy, and the
    horizontal axis contains the k-point index. The function accepts an
    existing axis. It creates a figure and axis when the caller supplies none.

    :see: :class:`~.test_plotting.TestPlotArpesSpectrum`

    Implementation Logic
    --------------------
    1. **Normalize plotting arrays**::

           intensity, energy_axis = _prepare_plot_arrays(spectrum)

       This validates the image rank and energy-axis length before plotting.

    2. **Draw the transposed intensity image**::

           image: AxesImage = ax.imshow(
               intensity.T,
               origin="lower",
               aspect=aspect,
               extent=extent,
               cmap=cmap,
           )

       The transpose places energy vertically and k-point index horizontally.

    3. **Return the Matplotlib objects**::

           return plot_result

       This binding keeps axis reuse and colorbar state explicit.

    Parameters
    ----------
    spectrum : ArpesSpectrum
        Spectrum containing ``intensity`` of shape ``(K, E)`` and
        ``energy_axis`` of shape ``(E,)``.
    ax : Optional[Axes], optional
        Existing axis for the image. If ``None``, the function creates a figure
        and axis.
    cmap : str, optional
        Matplotlib colormap name. Default is ``"gray"``.
    colorbar : bool, optional
        If True, add a colorbar labeled ``"Intensity (a.u.)"``.
    clim : Optional[Tuple[float, float]], optional
        Optional ``(vmin, vmax)`` color limits.
    interpolation : str, optional
        Image interpolation mode. Default is ``"nearest"``.
    aspect : Literal["equal", "auto"], optional
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

    """
    intensity: Float64[NDArray, "K E"]
    energy_axis: Float64[NDArray, " E"]
    intensity, energy_axis = _prepare_plot_arrays(spectrum)

    fig: Union[Figure, SubFigure]
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure
    ax: Axes

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
    plot_result: Tuple[Union[Figure, SubFigure], Axes, AxesImage] = (
        fig,
        ax,
        image,
    )
    return plot_result


@jaxtyped(typechecker=beartype)
def apply_kpath_ticks(
    ax: Axes,
    kpath: KPathInfo,
    draw_symmetry_lines: bool = True,
    line_color: str = "white",
    line_width: float = 0.5,
    line_alpha: float = 0.35,
) -> Axes:
    """Apply symmetry-point ticks/labels from KPathInfo to an axis.

    The function adds k-path symmetry labels, such as G, M, and K, to an axis
    using :class:`KPathInfo`. Optionally draws vertical guide lines at
    interior symmetry points to visually separate path segments.

    :see: :class:`~.test_plotting.TestApplyKpathTicks`

    Implementation Logic
    --------------------
    1. **Normalize label positions and text**::

           indices: List[int] = np.asarray(
               kpath.label_indices, dtype=np.int32
           ).tolist()
           labels: List[str] = list(kpath.labels)

       Host-side lists match the Matplotlib tick API and preserve label order.

    2. **Apply the shared label count**::

           n_labels: int = min(len(indices), len(labels))
           ticks: List[float] = [float(idx) for idx in indices[:n_labels]]

       Truncation tolerates incomplete legacy metadata without shifting labels.

    3. **Return the annotated axis**::

           return ax

       The caller receives the same axis after optional guide-line drawing.

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
    tick: float

    indices: List[int] = np.asarray(
        kpath.label_indices, dtype=np.int32
    ).tolist()
    labels: List[str] = list(kpath.labels)
    n_labels: int = min(len(indices), len(labels))
    if n_labels == 0:
        return ax

    ticks: List[float] = [float(idx) for idx in indices[:n_labels]]
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


@jaxtyped(typechecker=beartype)
def plot_arpes_with_kpath(  # noqa: PLR0913, PLR0917
    spectrum: ArpesSpectrum,
    kpath: KPathInfo,
    ax: Optional[Axes] = None,
    cmap: str = "gray",
    colorbar: bool = True,
    clim: Optional[Tuple[float, float]] = None,
    interpolation: str = "nearest",
    aspect: Literal["equal", "auto"] = "auto",
    xlabel: str = "Momentum (k)",
    ylabel: str = "Energy (eV)",
    title: str = "Simulated ARPES Spectrum",
    draw_symmetry_lines: bool = True,
) -> Tuple[Union[Figure, SubFigure], Axes, AxesImage]:
    """Plot ARPES spectrum and annotate k-axis using KPathInfo.

    This wrapper combines :func:`plot_arpes_spectrum` and
    :func:`apply_kpath_ticks`. Use it to show symmetry labels on a line-mode
    ARPES image.

    :see: :class:`~.test_plotting.TestPlotArpesWithKpath`

    Implementation Logic
    --------------------
    1. **Render the base spectrum**::

           fig, ax, image = plot_arpes_spectrum(
               spectrum=spectrum,
               ax=ax,
               cmap=cmap,
               aspect=aspect,
               clim=clim,
               colorbar=colorbar,
               xlabel=xlabel,
               ylabel=ylabel,
               title=title,
           )

       This centralizes image validation and styling in the base plotter.

    2. **Apply symmetry labels**::

           apply_kpath_ticks(
               ax=ax,
               kpath=kpath,
               draw_symmetry_lines=draw_symmetry_lines,
           )

       This adds line-mode metadata without recreating the image artist.

    3. **Return the composed plot**::

           return plot_result

       The result contains the figure, axis, and image from the base call.

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
    clim : Optional[Tuple[float, float]], optional
        Optional ``(vmin, vmax)`` color limits.
    interpolation : str, optional
        Image interpolation mode for ``imshow``.
    aspect : Literal["equal", "auto"], optional
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
    fig: Union[Figure, SubFigure]
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
    ax: Axes
    apply_kpath_ticks(
        ax=ax,
        kpath=kpath,
        draw_symmetry_lines=draw_symmetry_lines,
    )
    plot_result: Tuple[Union[Figure, SubFigure], Axes, AxesImage] = (
        fig,
        ax,
        image,
    )
    return plot_result


@jaxtyped(typechecker=beartype)
def plot_momentum_map(  # noqa: PLR0913, PLR0917
    intensity: Float64[Array, "n_kx n_ky"] | Float64[NDArray, "n_kx n_ky"],
    kx_axis: Float64[Array, " n_kx"] | Float64[NDArray, " n_kx"],
    ky_axis: Float64[Array, " n_ky"] | Float64[NDArray, " n_ky"],
    ax: Optional[Axes] = None,
    cmap: str = "inferno",
    colorbar: bool = True,
    clim: Optional[Tuple[float, float]] = None,
    interpolation: str = "nearest",
    aspect: Literal["equal", "auto"] = "equal",
    xlabel: str = r"$k_x$ ($\mathrm{\AA}^{-1}$)",
    ylabel: str = r"$k_y$ ($\mathrm{\AA}^{-1}$)",
    title: str = "Constant-Energy Map",
) -> Tuple[Union[Figure, SubFigure], Axes, AxesImage]:
    """Plot a momentum-momentum intensity map with Cartesian axes.

    The function renders one constant-energy or energy-integrated map
    with ``matplotlib.axes.Axes.imshow``. The extent follows the two
    momentum axes. The function accepts an existing axis. It creates a
    figure and axis when the caller supplies none.

    :see: :class:`~.test_plotting.TestPlotMomentumMap`

    Implementation Logic
    --------------------
    1. **Normalize the plotting arrays**::

           image_values = np.asarray(intensity, dtype=np.float64)

       The conversion accepts JAX and NumPy inputs alike.

    2. **Draw the transposed intensity image**::

           image = ax.imshow(image_values.T, origin="lower", ...)

       The transpose places ``ky`` vertically and ``kx``
       horizontally.

    3. **Return the Matplotlib objects**::

           return plot_result

       This binding keeps axis reuse and colorbar state explicit.

    Parameters
    ----------
    intensity : Float64[Array, "n_kx n_ky"] | Float64[NDArray, "n_kx n_ky"]
        Momentum-momentum intensity map.
    kx_axis : Float64[Array, " n_kx"] | Float64[NDArray, " n_kx"]
        Cartesian ``k_x`` axis in inverse Angstroms.
    ky_axis : Float64[Array, " n_ky"] | Float64[NDArray, " n_ky"]
        Cartesian ``k_y`` axis in inverse Angstroms.
    ax : Optional[Axes], optional
        Existing axis for the image. If ``None``, the function creates
        a figure and axis.
    cmap : str, optional
        Matplotlib colormap name. Default is ``"inferno"``.
    colorbar : bool, optional
        If True, add a colorbar labeled ``"Intensity (a.u.)"``.
    clim : Optional[Tuple[float, float]], optional
        Optional ``(vmin, vmax)`` color limits.
    interpolation : str, optional
        Image interpolation mode. Default is ``"nearest"``.
    aspect : Literal["equal", "auto"], optional
        Image aspect ratio passed to ``imshow``. Default is
        ``"equal"``.
    xlabel : str, optional
        x-axis label text. Default names ``k_x`` in inverse Angstroms.
    ylabel : str, optional
        y-axis label text. Default names ``k_y`` in inverse Angstroms.
    title : str, optional
        Axis title text. Default is ``"Constant-Energy Map"``.

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
    The runtime type checks reject an intensity shape that disagrees
    with the two momentum axes.
    """
    image_values: Float64[NDArray, "n_kx n_ky"] = np.asarray(
        intensity, dtype=np.float64
    )
    kx_values: Float64[NDArray, " n_kx"] = np.asarray(
        kx_axis, dtype=np.float64
    )
    ky_values: Float64[NDArray, " n_ky"] = np.asarray(
        ky_axis, dtype=np.float64
    )

    fig: Union[Figure, SubFigure]
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure
    ax: Axes

    image: AxesImage = ax.imshow(
        image_values.T,
        origin="lower",
        aspect=aspect,
        cmap=cmap,
        interpolation=interpolation,
        extent=(
            float(kx_values[0]),
            float(kx_values[-1]),
            float(ky_values[0]),
            float(ky_values[-1]),
        ),
    )
    if clim is not None:
        image.set_clim(clim[0], clim[1])
    if colorbar:
        fig.colorbar(image, ax=ax, label="Intensity (a.u.)")

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    plot_result: Tuple[Union[Figure, SubFigure], Axes, AxesImage] = (
        fig,
        ax,
        image,
    )
    return plot_result


__all__: list[str] = [
    "apply_kpath_ticks",
    "plot_arpes_spectrum",
    "plot_arpes_with_kpath",
    "plot_momentum_map",
]
