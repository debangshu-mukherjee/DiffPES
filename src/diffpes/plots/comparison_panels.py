r"""Compare spectral maps in shared-scale panel rows and difference maps.

Extended Summary
----------------
The module arranges related ARPES maps for visual comparison. The row
builders render each map through the base plotters from
:mod:`diffpes.plots.arpes_maps` and apply one shared color scale with
one spanning colorbar. The difference plotter renders signed data, such
as orbital contrasts or Poisson residuals, on a symmetric diverging
scale.

Routine Listings
----------------
:func:`plot_difference_map`
    Plot a signed difference map with a symmetric diverging scale.
:func:`plot_momentum_map_grid`
    Plot momentum-momentum maps as one row of shared-scale panels.
:func:`plot_spectral_cut_series`
    Plot energy-momentum cuts as one row of shared-scale panels.

Notes
-----
These functions operate on host-side NumPy arrays and Matplotlib
objects. Use them for analysis visualizations outside JAX-compiled
functions.
"""

# Exact pydoclint parameter types cannot split across physical lines.
# ruff: noqa: E501

import numpy as np
from beartype import beartype
from beartype.typing import List, Optional, Tuple, Union
from jaxtyping import Array, Float64, Shaped, jaxtyped
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure, SubFigure
from matplotlib.image import AxesImage
from numpy.typing import NDArray

from .arpes_maps import plot_momentum_map, plot_spectral_cut, spectrum_extent


@jaxtyped(typechecker=beartype)
def plot_spectral_cut_series(  # noqa: PLR0913, PLR0917
    intensities: Tuple[
        Float64[Array, "n_k n_e"] | Float64[NDArray, "n_k n_e"], ...
    ],
    momentum_axis: Float64[Array, " n_k"] | Float64[NDArray, " n_k"],
    energy_axis: Float64[Array, " n_e"] | Float64[NDArray, " n_e"],
    titles: Optional[Tuple[str, ...]] = None,
    axes: Optional[Tuple[Axes, ...]] = None,
    cmap: str = "magma",
    share_scale: bool = True,
    colorbar: bool = True,
    colorbar_label: str = "intensity (1/eV)",
    log_counts: bool = False,
    fermi_line: bool = True,
    xlabel: str = r"$k$ ($\mathrm{\AA}^{-1}$)",
    ylabel: str = r"$E - E_F$ (eV)",
) -> Tuple[Union[Figure, SubFigure], Tuple[Axes, ...], List[AxesImage]]:
    r"""Plot energy-momentum cuts as one row of shared-scale panels.

    The function renders several same-shaped spectral cuts side by
    side. Each panel uses :func:`plot_spectral_cut` on one shared
    momentum axis and one shared energy axis. With ``share_scale`` the
    panels share one color scale, so the eye can compare intensities
    across panels directly. One spanning colorbar then describes every
    panel.

    :see: :class:`~.test_comparison_panels.TestPlotSpectralCutSeries`

    Implementation Logic
    --------------------
    1. **Validate the title count**::

           if titles is not None and len(titles) != n_panels:
               raise ValueError(msg)

       Each panel needs exactly one title when the caller gives titles.

    2. **Compute the shared color limits**::

           vmin = min(0.0, float(np.min(stacked)))
           vmax = float(np.max(stacked))

       The stack contains every map after the optional ``np.log1p``.
       The floor at zero anchors a nonnegative intensity scale at zero.

    3. **Create the panel row when the caller gives no axes**::

           fig, axis_grid = plt.subplots(
               1, n_panels, sharey=True, constrained_layout=True
           )

       The shared y axis aligns the energy coordinate across panels.

    4. **Render each panel through the base plotter**::

           plot_spectral_cut(..., ax=panel_axis, colorbar=False, ...)

       Each call passes ``clim`` under ``share_scale``. Only the first
       panel keeps the y-axis label.

    5. **Attach one spanning colorbar**::

           fig.colorbar(images[-1], ax=axes_row, shrink=0.86, ...)

       The single colorbar spans the complete panel row.

    Parameters
    ----------
    intensities : Tuple[Float64[Array, "n_k n_e"] | Float64[NDArray, "n_k n_e"], ...]
        Same-shaped intensity maps on the momentum-energy grid in 1/eV.
    momentum_axis : Float64[Array, " n_k"] | Float64[NDArray, " n_k"]
        Physical momentum axis in 1/Angstrom, shared by every panel.
    energy_axis : Float64[Array, " n_e"] | Float64[NDArray, " n_e"]
        Energy axis relative to the Fermi level in eV, shared by every
        panel.
    titles : Optional[Tuple[str, ...]], optional
        One title per panel. If ``None``, each panel title stays empty.
    axes : Optional[Tuple[Axes, ...]], optional
        Existing axes for the panels, one axis per intensity map. If
        ``None``, the function creates a figure with one panel row.
    cmap : str, optional
        Matplotlib colormap name. Default is ``"magma"``.
    share_scale : bool, optional
        If True, apply one global ``(vmin, vmax)`` to every panel.
        Default True.
    colorbar : bool, optional
        If True, add one colorbar that spans the panel row. Default
        True.
    colorbar_label : str, optional
        Colorbar label text. Default is ``"intensity (1/eV)"``.
    log_counts : bool, optional
        If True, plot ``log(1 + intensity)`` instead of the raw values.
        Default False.
    fermi_line : bool, optional
        If True, draw a horizontal guide line at zero energy in every
        panel. Default True.
    xlabel : str, optional
        x-axis label text for every panel. Default names the momentum
        in 1/Angstrom.
    ylabel : str, optional
        y-axis label text for the first panel. Default is
        :math:`E - E_F` in eV.

    Returns
    -------
    fig : Figure
        Matplotlib figure object.
    axes_row : Tuple[Axes, ...]
        Panel axes in left-to-right order.
    images : List[AxesImage]
        Image artists in the same order as the intensity maps.

    Raises
    ------
    ValueError
        If the length of ``titles`` differs from the number of
        intensity maps.

    Notes
    -----
    Without ``share_scale`` each panel scales its own color range, and
    the colorbar describes only the last panel.

    See Also
    --------
    plot_spectral_cut : Plot an energy-momentum intensity map on
        physical axes.
    plot_momentum_map_grid : Plot momentum-momentum maps as one row of
        shared-scale panels.
    """
    n_panels: int = len(intensities)
    if titles is not None and len(titles) != n_panels:
        msg: str = (
            f"Expected one title per intensity map: got {len(titles)} "
            f"titles for {n_panels} maps."
        )
        raise ValueError(msg)

    panel_values: List[Float64[NDArray, "n_k n_e"]] = [
        np.asarray(item, dtype=np.float64) for item in intensities
    ]

    clim: Optional[Tuple[float, float]] = None
    if share_scale:
        stacked: Float64[NDArray, "n_maps n_k n_e"] = np.stack(panel_values)
        if log_counts:
            stacked = np.log1p(stacked)
        vmin: float = min(0.0, float(np.min(stacked)))
        vmax: float = float(np.max(stacked))
        clim = (vmin, vmax)

    fig: Union[Figure, SubFigure]
    axes_row: Tuple[Axes, ...]
    axis_grid: Union[Axes, Shaped[NDArray, " n_panels"]]
    if axes is None:
        fig, axis_grid = plt.subplots(
            1, n_panels, sharey=True, constrained_layout=True
        )
        axes_row = tuple(np.atleast_1d(axis_grid))
    else:
        axes_row = tuple(axes)
        fig = axes_row[0].figure

    images: List[AxesImage] = []
    panel_index: int
    panel_intensity: Float64[NDArray, "n_k n_e"]
    for panel_index, panel_intensity in enumerate(panel_values):
        panel_axis: Axes = axes_row[panel_index]
        panel_title: str = "" if titles is None else titles[panel_index]
        image: AxesImage
        _, _, image = plot_spectral_cut(
            panel_intensity,
            momentum_axis,
            energy_axis,
            ax=panel_axis,
            cmap=cmap,
            colorbar=False,
            clim=clim,
            log_counts=log_counts,
            fermi_line=fermi_line,
            xlabel=xlabel,
            ylabel=ylabel,
            title=panel_title,
        )
        images.append(image)
        if panel_index > 0:
            panel_axis.set_ylabel("")

    if colorbar:
        fig.colorbar(
            images[-1], ax=axes_row, shrink=0.86, label=colorbar_label
        )

    plot_result: Tuple[
        Union[Figure, SubFigure], Tuple[Axes, ...], List[AxesImage]
    ] = (fig, axes_row, images)
    return plot_result


@jaxtyped(typechecker=beartype)
def plot_momentum_map_grid(  # noqa: PLR0913, PLR0917
    maps: Tuple[
        Float64[Array, "n_kx n_ky"] | Float64[NDArray, "n_kx n_ky"], ...
    ],
    kx_axis: Float64[Array, " n_kx"] | Float64[NDArray, " n_kx"],
    ky_axis: Float64[Array, " n_ky"] | Float64[NDArray, " n_ky"],
    titles: Optional[Tuple[str, ...]] = None,
    axes: Optional[Tuple[Axes, ...]] = None,
    cmap: str = "inferno",
    share_scale: bool = True,
    colorbar: bool = True,
    colorbar_label: str = "intensity (a.u.)",
    crosshair: Optional[Tuple[float, float]] = None,
    crosshair_color: str = "white",
) -> Tuple[Union[Figure, SubFigure], Tuple[Axes, ...], List[AxesImage]]:
    r"""Plot momentum-momentum maps as one row of shared-scale panels.

    The function renders several same-shaped constant-energy maps side
    by side. Each panel uses :func:`plot_momentum_map` on one shared
    ``k_x`` axis and one shared ``k_y`` axis with an equal aspect
    ratio. With ``share_scale`` the panels share one color scale and
    one spanning colorbar. An optional crosshair marks one
    ``(k_x, k_y)`` point in every panel.

    :see: :class:`~.test_comparison_panels.TestPlotMomentumMapGrid`

    Implementation Logic
    --------------------
    1. **Validate the title count**::

           if titles is not None and len(titles) != n_panels:
               raise ValueError(msg)

       Each panel needs exactly one title when the caller gives titles.

    2. **Compute the shared color limits**::

           vmin = min(0.0, float(np.min(stacked)))
           vmax = float(np.max(stacked))

       The floor at zero anchors a nonnegative intensity scale at zero.

    3. **Create the panel row when the caller gives no axes**::

           fig, axis_grid = plt.subplots(
               1, n_panels, sharey=True, constrained_layout=True
           )

       The shared y axis aligns the ``k_y`` coordinate across panels.

    4. **Render each panel through the base plotter**::

           plot_momentum_map(..., ax=panel_axis, colorbar=False, ...)

       Each call passes ``clim`` under ``share_scale``. Later panels
       drop the y tick labels and the y-axis label.

    5. **Draw the crosshair and attach one spanning colorbar**::

           panel_axis.axvline(crosshair[0], ...)
           panel_axis.axhline(crosshair[1], ...)
           fig.colorbar(images[-1], ax=axes_row, shrink=0.86, ...)

       The crosshair marks the same momentum point in every panel.

    Parameters
    ----------
    maps : Tuple[Float64[Array, "n_kx n_ky"] | Float64[NDArray, "n_kx n_ky"], ...]
        Same-shaped momentum-momentum intensity maps.
    kx_axis : Float64[Array, " n_kx"] | Float64[NDArray, " n_kx"]
        Cartesian ``k_x`` axis in 1/Angstrom, shared by every panel.
    ky_axis : Float64[Array, " n_ky"] | Float64[NDArray, " n_ky"]
        Cartesian ``k_y`` axis in 1/Angstrom, shared by every panel.
    titles : Optional[Tuple[str, ...]], optional
        One title per panel. If ``None``, each panel title stays empty.
    axes : Optional[Tuple[Axes, ...]], optional
        Existing axes for the panels, one axis per map. If ``None``,
        the function creates a figure with one panel row.
    cmap : str, optional
        Matplotlib colormap name. Default is ``"inferno"``.
    share_scale : bool, optional
        If True, apply one global ``(vmin, vmax)`` to every panel.
        Default True.
    colorbar : bool, optional
        If True, add one colorbar that spans the panel row. Default
        True.
    colorbar_label : str, optional
        Colorbar label text. Default is ``"intensity (a.u.)"``.
    crosshair : Optional[Tuple[float, float]], optional
        The ``(k_x, k_y)`` position of a crosshair in 1/Angstrom. If
        not ``None``, every panel shows one vertical and one horizontal
        guide line through this point.
    crosshair_color : str, optional
        Color of the crosshair lines. Default ``"white"``.

    Returns
    -------
    fig : Figure
        Matplotlib figure object.
    axes_row : Tuple[Axes, ...]
        Panel axes in left-to-right order.
    images : List[AxesImage]
        Image artists in the same order as the maps.

    Raises
    ------
    ValueError
        If the length of ``titles`` differs from the number of maps.

    Notes
    -----
    Without ``share_scale`` each panel scales its own color range, and
    the colorbar describes only the last panel.

    See Also
    --------
    plot_momentum_map : Plot a momentum-momentum intensity map with
        Cartesian axes.
    plot_spectral_cut_series : Plot energy-momentum cuts as one row of
        shared-scale panels.
    """
    n_panels: int = len(maps)
    if titles is not None and len(titles) != n_panels:
        msg: str = (
            f"Expected one title per momentum map: got {len(titles)} "
            f"titles for {n_panels} maps."
        )
        raise ValueError(msg)

    panel_values: List[Float64[NDArray, "n_kx n_ky"]] = [
        np.asarray(item, dtype=np.float64) for item in maps
    ]

    clim: Optional[Tuple[float, float]] = None
    if share_scale:
        stacked: Float64[NDArray, "n_maps n_kx n_ky"] = np.stack(panel_values)
        vmin: float = min(0.0, float(np.min(stacked)))
        vmax: float = float(np.max(stacked))
        clim = (vmin, vmax)

    fig: Union[Figure, SubFigure]
    axes_row: Tuple[Axes, ...]
    axis_grid: Union[Axes, Shaped[NDArray, " n_panels"]]
    if axes is None:
        fig, axis_grid = plt.subplots(
            1, n_panels, sharey=True, constrained_layout=True
        )
        axes_row = tuple(np.atleast_1d(axis_grid))
    else:
        axes_row = tuple(axes)
        fig = axes_row[0].figure

    images: List[AxesImage] = []
    panel_index: int
    panel_intensity: Float64[NDArray, "n_kx n_ky"]
    for panel_index, panel_intensity in enumerate(panel_values):
        panel_axis: Axes = axes_row[panel_index]
        panel_title: str = "" if titles is None else titles[panel_index]
        image: AxesImage
        _, _, image = plot_momentum_map(
            panel_intensity,
            kx_axis,
            ky_axis,
            ax=panel_axis,
            cmap=cmap,
            colorbar=False,
            clim=clim,
            aspect="equal",
            title=panel_title,
        )
        images.append(image)
        if crosshair is not None:
            panel_axis.axvline(
                crosshair[0],
                color=crosshair_color,
                linewidth=0.8,
                alpha=0.8,
            )
            panel_axis.axhline(
                crosshair[1],
                color=crosshair_color,
                linewidth=0.8,
                alpha=0.8,
            )
        if panel_index > 0:
            panel_axis.set_yticklabels([])
            panel_axis.set_ylabel("")

    if colorbar:
        fig.colorbar(
            images[-1], ax=axes_row, shrink=0.86, label=colorbar_label
        )

    plot_result: Tuple[
        Union[Figure, SubFigure], Tuple[Axes, ...], List[AxesImage]
    ] = (fig, axes_row, images)
    return plot_result


@jaxtyped(typechecker=beartype)
def plot_difference_map(  # noqa: PLR0913, PLR0917
    difference: Float64[Array, "n_x n_y"] | Float64[NDArray, "n_x n_y"],
    x_axis: Float64[Array, " n_x"] | Float64[NDArray, " n_x"],
    y_axis: Float64[Array, " n_y"] | Float64[NDArray, " n_y"],
    ax: Optional[Axes] = None,
    cmap: str = "RdBu_r",
    scale_percentile: Optional[float] = None,
    colorbar: bool = True,
    colorbar_label: str = "difference",
    zero_lines: bool = False,
    xlabel: str = "",
    ylabel: str = "",
    title: str = "",
) -> Tuple[Union[Figure, SubFigure], Axes, AxesImage]:
    r"""Plot a signed difference map with a symmetric diverging scale.

    The function renders signed data, such as an orbital contrast, a
    Poisson residual, or a gradient sensitivity map, with
    ``matplotlib.axes.Axes.imshow``. The color limits stay symmetric
    around zero, so the neutral colormap center marks the zero level.
    A percentile scale suppresses the influence of isolated outliers.

    :see: :class:`~.test_comparison_panels.TestPlotDifferenceMap`

    Implementation Logic
    --------------------
    1. **Compute the symmetric scale**::

           scale = float(np.max(np.abs(difference_values)))

       With ``scale_percentile`` the scale becomes
       ``np.percentile(np.abs(difference_values), scale_percentile)``.
       A zero scale falls back to ``1.0``.

    2. **Draw the transposed difference image**::

           image = ax.imshow(
               difference_values.T,
               origin="lower",
               extent=spectrum_extent(x_axis, y_axis),
               vmin=-scale,
               vmax=scale,
           )

       The transpose places the y axis vertically and the x axis
       horizontally.

    3. **Draw the optional zero lines and return the objects**::

           ax.axhline(0.0, ...)
           ax.axvline(0.0, ...)

       The thin black lines mark the coordinate origin.

    Parameters
    ----------
    difference : Float64[Array, "n_x n_y"] | Float64[NDArray, "n_x n_y"]
        Signed difference values on the ``(x, y)`` grid.
    x_axis : Float64[Array, " n_x"] | Float64[NDArray, " n_x"]
        Physical x axis of the grid.
    y_axis : Float64[Array, " n_y"] | Float64[NDArray, " n_y"]
        Physical y axis of the grid.
    ax : Optional[Axes], optional
        Existing axis for the image. If ``None``, the function creates
        a figure and axis.
    cmap : str, optional
        Diverging Matplotlib colormap name. Default is ``"RdBu_r"``.
    scale_percentile : Optional[float], optional
        Percentile of ``|difference|`` in the range 0 to 100 that sets
        the color scale. If ``None``, the maximum of ``|difference|``
        sets the scale.
    colorbar : bool, optional
        If True, add a colorbar with ``colorbar_label``. Default True.
    colorbar_label : str, optional
        Colorbar label text. Default is ``"difference"``.
    zero_lines : bool, optional
        If True, draw thin black guide lines through the coordinate
        origin. Default False.
    xlabel : str, optional
        x-axis label text. Default is empty.
    ylabel : str, optional
        y-axis label text. Default is empty.
    title : str, optional
        Axis title text. Default is empty.

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
    An all-zero difference map gives a zero scale. The function then
    falls back to the scale ``1.0``, so ``imshow`` receives the valid
    color range ``(-1.0, 1.0)``.

    See Also
    --------
    spectrum_extent : Compute the imshow extent of a momentum and an
        energy axis.
    plot_spectral_cut_series : Plot energy-momentum cuts as one row of
        shared-scale panels.
    """
    difference_values: Float64[NDArray, "n_x n_y"] = np.asarray(
        difference, dtype=np.float64
    )
    magnitude: Float64[NDArray, "n_x n_y"] = np.abs(difference_values)
    scale: float
    if scale_percentile is None:
        scale = float(np.max(magnitude))
    else:
        scale = float(np.percentile(magnitude, scale_percentile))
    if scale == 0.0:
        scale = 1.0

    fig: Union[Figure, SubFigure]
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure
    ax: Axes

    extent: Tuple[float, float, float, float] = spectrum_extent(x_axis, y_axis)
    image: AxesImage = ax.imshow(
        difference_values.T,
        origin="lower",
        aspect="auto",
        cmap=cmap,
        extent=extent,
        vmin=-scale,
        vmax=scale,
    )
    if colorbar:
        fig.colorbar(image, ax=ax, label=colorbar_label)
    if zero_lines:
        ax.axhline(0.0, color="black", linewidth=0.6)
        ax.axvline(0.0, color="black", linewidth=0.6)

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
    "plot_difference_map",
    "plot_momentum_map_grid",
    "plot_spectral_cut_series",
]
