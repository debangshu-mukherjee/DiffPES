r"""Plot band dispersions and weighted band scatters along a path.

Extended Summary
----------------
This module renders band eigenvalues along a one-dimensional
momentum path. The functions accept a :class:`BandStructure` or
a :class:`DiagonalizedBands` carrier, or a raw eigenvalue array.
Line plots show the dispersions, an overlay places the dispersions
on a spectral image, and a scatter plot encodes raw per-band
weights.

Routine Listings
----------------
:func:`plot_band_dispersion`
    Plot band dispersions as lines along a momentum path.
:func:`plot_band_scatter_weights`
    Plot bands as weight-encoded scatter points along a path.
:func:`plot_bands_over_spectrum`
    Plot band dispersions over an energy-momentum intensity image.

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
from matplotlib.collections import PathCollection
from matplotlib.figure import Figure, SubFigure
from matplotlib.image import AxesImage
from matplotlib.lines import Line2D
from numpy.typing import NDArray

from diffpes.types import BandStructure, DiagonalizedBands

from .arpes_maps import plot_spectral_cut


@jaxtyped(typechecker=beartype)
def _band_values(  # noqa: DOC105
    bands: Union[
        BandStructure,
        DiagonalizedBands,
        Float64[Array, "n_k n_bands"],
        Float64[NDArray, "n_k n_bands"],
    ],
    shift_fermi: bool,
) -> Float64[NDArray, "n_k n_bands"]:
    """PRIVATE: Convert band input to a NumPy eigenvalue matrix.

    The helper reads the eigenvalues from a band carrier and applies the
    optional Fermi shift. A raw array passes through unchanged, so a raw
    array must already contain energies relative to the Fermi level.

    Parameters
    ----------
    bands : BandStructure | DiagonalizedBands | Float64 2D array
        Band carrier with ``eigenvalues`` of shape ``(n_k, n_bands)`` in
        eV and a scalar ``fermi_energy`` in eV, or a raw eigenvalue
        array of the same shape.
    shift_fermi : bool
        If True, subtract the carrier ``fermi_energy`` from the carrier
        eigenvalues. The flag has no effect on a raw array.

    Returns
    -------
    band_values : Float64[NDArray, "n_k n_bands"]
        Band energies in eV as a NumPy array of shape
        ``(n_k, n_bands)``.

    Notes
    -----
    Converts the eigenvalues with :func:`np.asarray` to ``float64``. The
    carriers validate the two-dimensional eigenvalue layout at
    construction, and the runtime type checks validate a raw array.
    """
    band_values: Float64[NDArray, "n_k n_bands"]
    if isinstance(bands, (BandStructure, DiagonalizedBands)):
        eigenvalues: Float64[NDArray, "n_k n_bands"] = np.asarray(
            bands.eigenvalues, dtype=np.float64
        )
        fermi: float = float(np.asarray(bands.fermi_energy, dtype=np.float64))
        band_values = eigenvalues - fermi if shift_fermi else eigenvalues
    else:
        band_values = np.asarray(bands, dtype=np.float64)
    return band_values


@jaxtyped(typechecker=beartype)
def _resolve_momentum(  # noqa: DOC105
    momentum_axis: Optional[Float64[Array, " n_k"] | Float64[NDArray, " n_k"]],
    n_kpoints: int,
) -> Tuple[Float64[NDArray, " n_k"], str]:
    r"""PRIVATE: Resolve the horizontal axis values and default label.

    The helper converts an explicit momentum axis to a NumPy array and
    selects the matching default x-axis label. A missing axis falls back
    to the k-point index.

    Parameters
    ----------
    momentum_axis : Optional[Float64 1D array]
        Physical arc-length momentum axis in 1/Angstrom, or ``None``
        for the k-point index.
    n_kpoints : int
        Number of k-points that the band values contain.

    Returns
    -------
    momentum_values : Float64[NDArray, " n_k"]
        Horizontal axis values of shape ``(n_k,)``.
    default_xlabel : str
        The label ``r"$k$ ($\mathrm{\AA}^{-1}$)"`` for a physical axis,
        or ``"k-point index"`` for the index fallback.

    Raises
    ------
    ValueError
        If the explicit momentum axis length differs from
        ``n_kpoints``.

    Notes
    -----
    The index fallback uses ``np.arange(n_kpoints)`` in ``float64``.
    """
    momentum_values: Float64[NDArray, " n_k"]
    default_xlabel: str
    if momentum_axis is None:
        momentum_values = np.arange(n_kpoints, dtype=np.float64)
        default_xlabel = "k-point index"
    else:
        momentum_values = np.asarray(momentum_axis, dtype=np.float64)
        default_xlabel = r"$k$ ($\mathrm{\AA}^{-1}$)"
        if momentum_values.shape[0] != n_kpoints:
            msg: str = (
                "momentum_axis length must equal the band k-point count."
            )
            raise ValueError(msg)
    resolved: Tuple[Float64[NDArray, " n_k"], str] = (
        momentum_values,
        default_xlabel,
    )
    return resolved


@jaxtyped(typechecker=beartype)
def plot_band_dispersion(  # noqa: DOC105, PLR0913, PLR0917
    bands: Union[
        BandStructure,
        DiagonalizedBands,
        Float64[Array, "n_k n_bands"],
        Float64[NDArray, "n_k n_bands"],
    ],
    momentum_axis: Optional[
        Float64[Array, " n_k"] | Float64[NDArray, " n_k"]
    ] = None,
    ax: Optional[Axes] = None,
    shift_fermi: bool = True,
    color: str = "0.25",
    linewidth: float = 1.0,
    alpha: float = 1.0,
    fermi_line: bool = True,
    xlabel: Optional[str] = None,
    ylabel: str = r"$E - E_F$ (eV)",
    title: str = "",
) -> Tuple[Union[Figure, SubFigure], Axes, List[Line2D]]:
    r"""Plot band dispersions as lines along a momentum path.

    The function draws every band as one line over the momentum path.
    The vertical axis carries the energy relative to the Fermi level.
    The function accepts an existing axis. It creates a figure and axis
    when the caller supplies none.

    :see: :class:`~.test_band_lines.TestPlotBandDispersion`

    Implementation Logic
    --------------------
    1. **Normalize the band energies**::

           band_values = _band_values(bands, shift_fermi)

       A carrier contributes its eigenvalues minus the optional Fermi
       shift. A raw array passes through unchanged.

    2. **Resolve the horizontal axis**::

           momentum_values, default_xlabel = _resolve_momentum(
               momentum_axis, band_values.shape[0]
           )

       The default label follows the axis choice, and an explicit
       ``xlabel`` overrides it.

    3. **Draw all bands with one plot call**::

           lines: List[Line2D] = ax.plot(
               momentum_values, band_values, ...
           )

       Matplotlib broadcasts the eigenvalue columns into one line per
       band.

    4. **Return the Matplotlib objects**::

           return plot_result

       This binding keeps axis reuse and line styling explicit.

    Parameters
    ----------
    bands : BandStructure | DiagonalizedBands | Float64 2D array
        Band carrier with ``eigenvalues`` of shape ``(n_k, n_bands)`` in
        eV, or a raw eigenvalue array of the same shape. A raw array
        must already contain energies relative to the Fermi level.
    momentum_axis : Optional[Float64 1D array], optional
        Physical arc-length momentum axis in 1/Angstrom. Default
        ``None`` uses the k-point index.
    ax : Optional[Axes], optional
        Existing axis for the lines. If ``None``, the function creates
        a figure and axis.
    shift_fermi : bool, optional
        If True, subtract the carrier Fermi energy from the carrier
        eigenvalues. Default True.
    color : str, optional
        Line color for every band. Default ``"0.25"``.
    linewidth : float, optional
        Line width for every band. Default 1.0.
    alpha : float, optional
        Line alpha for every band. Default 1.0.
    fermi_line : bool, optional
        If True, draw a dashed horizontal guide line at zero energy.
        Default True.
    xlabel : Optional[str], optional
        x-axis label text. Default ``None`` selects
        ``r"$k$ ($\mathrm{\AA}^{-1}$)"`` for a physical momentum axis
        and ``"k-point index"`` otherwise.
    ylabel : str, optional
        y-axis label text. Default is :math:`E - E_F` in eV.
    title : str, optional
        Axis title text. Default is empty.

    Returns
    -------
    fig : Figure
        Matplotlib figure object.
    ax : Axes
        Axis used for plotting.
    lines : List[Line2D]
        Line artists, one per band, from the plot call.

    See Also
    --------
    plot_bands_over_spectrum : Plot band dispersions over an
        energy-momentum intensity image.
    """
    band_values: Float64[NDArray, "n_k n_bands"] = _band_values(
        bands, shift_fermi
    )
    momentum_values: Float64[NDArray, " n_k"]
    default_xlabel: str
    momentum_values, default_xlabel = _resolve_momentum(
        momentum_axis, band_values.shape[0]
    )
    resolved_xlabel: str = default_xlabel if xlabel is None else xlabel

    fig: Union[Figure, SubFigure]
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure
    ax: Axes

    lines: List[Line2D] = ax.plot(
        momentum_values,
        band_values,
        color=color,
        linewidth=linewidth,
        alpha=alpha,
    )
    if fermi_line:
        ax.axhline(0.0, color="0.45", linewidth=0.9, linestyle="--")

    ax.set_xlabel(resolved_xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    plot_result: Tuple[Union[Figure, SubFigure], Axes, List[Line2D]] = (
        fig,
        ax,
        lines,
    )
    return plot_result


@jaxtyped(typechecker=beartype)
def plot_bands_over_spectrum(  # noqa: DOC105, PLR0913, PLR0917
    intensity: Float64[Array, "n_k n_e"] | Float64[NDArray, "n_k n_e"],
    momentum_axis: Float64[Array, " n_k"] | Float64[NDArray, " n_k"],
    energy_axis: Float64[Array, " n_e"] | Float64[NDArray, " n_e"],
    bands: Union[
        BandStructure,
        DiagonalizedBands,
        Float64[Array, "n_k n_bands"],
        Float64[NDArray, "n_k n_bands"],
    ],
    ax: Optional[Axes] = None,
    cmap: str = "magma",
    colorbar: bool = True,
    band_color: str = "white",
    band_linewidth: float = 0.7,
    band_alpha: float = 0.65,
    shift_fermi: bool = True,
    xlabel: str = r"$k$ ($\mathrm{\AA}^{-1}$)",
    ylabel: str = r"$E - E_F$ (eV)",
    title: str = "",
) -> Tuple[Union[Figure, SubFigure], Axes, AxesImage]:
    r"""Plot band dispersions over an energy-momentum intensity image.

    The function renders one spectral cut with
    :func:`plot_spectral_cut` and overlays every band as one thin line
    on the same axis. The overlay compares the bare dispersions with the
    simulated or measured intensity.

    :see: :class:`~.test_band_lines.TestPlotBandsOverSpectrum`

    Implementation Logic
    --------------------
    1. **Render the spectral image**::

           fig, ax, image = plot_spectral_cut(
               intensity=intensity,
               momentum_axis=momentum_axis,
               energy_axis=energy_axis,
               ...
           )

       The base plotter owns the image extent, colorbar, and axis
       styling.

    2. **Normalize the band energies**::

           band_values = _band_values(bands, shift_fermi)

       A carrier contributes its eigenvalues minus the optional Fermi
       shift. A raw array passes through unchanged.

    3. **Overlay all bands with one plot call**::

           ax.plot(momentum_values, band_values, color=band_color, ...)

       Matplotlib broadcasts the eigenvalue columns into one thin line
       per band on top of the image.

    4. **Return the Matplotlib objects**::

           return plot_result

       The result contains the figure, axis, and image from the base
       call.

    Parameters
    ----------
    intensity : Float64[Array, "n_k n_e"] | Float64[NDArray, "n_k n_e"]
        Intensity on the momentum-energy grid in 1/eV.
    momentum_axis : Float64[Array, " n_k"] | Float64[NDArray, " n_k"]
        Physical momentum axis in 1/Angstrom.
    energy_axis : Float64[Array, " n_e"] | Float64[NDArray, " n_e"]
        Energy axis relative to the Fermi level in eV.
    bands : BandStructure | DiagonalizedBands | Float64 2D array
        Band carrier with ``eigenvalues`` of shape ``(n_k, n_bands)`` in
        eV on the same ``n_k`` k-points, or a raw eigenvalue array of
        the same shape. A raw array must already contain energies
        relative to the Fermi level.
    ax : Optional[Axes], optional
        Existing axis for the image. If ``None``, the function creates
        a figure and axis.
    cmap : str, optional
        Matplotlib colormap name. Default is ``"magma"``.
    colorbar : bool, optional
        If True, add a colorbar to the spectral image. Default True.
    band_color : str, optional
        Line color for every overlaid band. Default ``"white"``.
    band_linewidth : float, optional
        Line width for every overlaid band. Default 0.7.
    band_alpha : float, optional
        Line alpha for every overlaid band. Default 0.65.
    shift_fermi : bool, optional
        If True, subtract the carrier Fermi energy from the carrier
        eigenvalues. Default True.
    xlabel : str, optional
        x-axis label text. Default names the momentum in 1/Angstrom.
    ylabel : str, optional
        y-axis label text. Default is :math:`E - E_F` in eV.
    title : str, optional
        Axis title text. Default is empty.

    Returns
    -------
    fig : Figure
        Matplotlib figure object.
    ax : Axes
        Axis used for plotting.
    image : AxesImage
        Image artist created by the spectral-cut renderer.

    Raises
    ------
    ValueError
        If the band k-point count differs from the momentum axis
        length.

    See Also
    --------
    plot_band_dispersion : Plot band dispersions as lines along a
        momentum path.
    plot_spectral_cut : Plot an energy-momentum intensity map on
        physical axes.
    """
    fig: Union[Figure, SubFigure]
    image: AxesImage
    fig, ax, image = plot_spectral_cut(
        intensity=intensity,
        momentum_axis=momentum_axis,
        energy_axis=energy_axis,
        ax=ax,
        cmap=cmap,
        colorbar=colorbar,
        xlabel=xlabel,
        ylabel=ylabel,
        title=title,
    )
    ax: Axes

    band_values: Float64[NDArray, "n_k n_bands"] = _band_values(
        bands, shift_fermi
    )
    momentum_values: Float64[NDArray, " n_k"] = np.asarray(
        momentum_axis, dtype=np.float64
    )
    if band_values.shape[0] != momentum_values.shape[0]:
        msg: str = "bands must match momentum_axis on the k-point count."
        raise ValueError(msg)
    ax.plot(
        momentum_values,
        band_values,
        color=band_color,
        linewidth=band_linewidth,
        alpha=band_alpha,
    )
    plot_result: Tuple[Union[Figure, SubFigure], Axes, AxesImage] = (
        fig,
        ax,
        image,
    )
    return plot_result


@jaxtyped(typechecker=beartype)
def plot_band_scatter_weights(  # noqa: DOC105, PLR0913, PLR0917
    bands: Union[
        BandStructure,
        DiagonalizedBands,
        Float64[Array, "n_k n_bands"],
        Float64[NDArray, "n_k n_bands"],
    ],
    weights: Float64[Array, "n_k n_bands"] | Float64[NDArray, "n_k n_bands"],
    momentum_axis: Optional[
        Float64[Array, " n_k"] | Float64[NDArray, " n_k"]
    ] = None,
    mode: Literal["size", "color"] = "size",
    ax: Optional[Axes] = None,
    size_scale: float = 34.0,
    cmap: str = "cividis",
    color: str = "tab:blue",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    backdrop: bool = True,
    backdrop_color: str = "0.82",
    colorbar: bool = False,
    colorbar_label: str = "weight",
    alpha: float = 0.75,
    shift_fermi: bool = True,
    xlabel: Optional[str] = None,
    ylabel: str = r"$E - E_F$ (eV)",
    title: str = "",
) -> Tuple[Union[Figure, SubFigure], Axes, PathCollection]:
    r"""Plot bands as weight-encoded scatter points along a path.

    The function encodes one raw weight per band and k-point, such as an
    occupation number or an orbital fraction. The ``"size"`` mode
    encodes the weight magnitude as the marker area with one solid
    color. The ``"color"`` mode encodes the weight value as the marker
    color through a colormap with a fixed small marker size.

    :see: :class:`~.test_band_lines.TestPlotBandScatterWeights`

    Implementation Logic
    --------------------
    1. **Normalize the band energies and weights**::

           band_values = _band_values(bands, shift_fermi)
           weight_values = np.asarray(weights, dtype=np.float64)

       A shape comparison rejects weights that disagree with the
       eigenvalues.

    2. **Draw the optional backdrop**::

           ax.plot(momentum_values, band_values,
                   color=backdrop_color, linewidth=0.3)

       The thin grey band lines keep faint weights legible.

    3. **Draw the weight-encoded scatter**::

           scatter = ax.scatter(xvals.ravel(), band_values.ravel(), ...)

       The ``"size"`` mode sets ``s = size_scale * |w|`` with a solid
       color. The ``"color"`` mode sets ``c = w`` with the colormap and
       a fixed size of 4.0.

    4. **Return the Matplotlib objects**::

           return plot_result

       The caller receives the figure, axis, and scatter artist.

    Parameters
    ----------
    bands : BandStructure | DiagonalizedBands | Float64 2D array
        Band carrier with ``eigenvalues`` of shape ``(n_k, n_bands)`` in
        eV, or a raw eigenvalue array of the same shape. A raw array
        must already contain energies relative to the Fermi level.
    weights : Float64 2D array
        Raw per-band weights of shape ``(n_k, n_bands)``, such as
        occupation numbers or orbital fractions.
    momentum_axis : Optional[Float64 1D array], optional
        Physical arc-length momentum axis in 1/Angstrom. Default
        ``None`` uses the k-point index.
    mode : Literal["size", "color"], optional
        Weight encoding. ``"size"`` encodes the weight magnitude as the
        marker area. ``"color"`` encodes the weight value as the marker
        color. Default ``"size"``.
    ax : Optional[Axes], optional
        Existing axis for the scatter. If ``None``, the function
        creates a figure and axis.
    size_scale : float, optional
        Marker area per unit weight magnitude in points squared for the
        ``"size"`` mode. Default 34.0.
    cmap : str, optional
        Colormap name for the ``"color"`` mode. Default ``"cividis"``.
    color : str, optional
        Solid marker color for the ``"size"`` mode. Default
        ``"tab:blue"``.
    vmin : Optional[float], optional
        Lower color limit for the ``"color"`` mode. ``None`` defers to
        Matplotlib. Default ``None``.
    vmax : Optional[float], optional
        Upper color limit for the ``"color"`` mode. ``None`` defers to
        Matplotlib. Default ``None``.
    backdrop : bool, optional
        If True, first draw all bands as thin grey lines. Default True.
    backdrop_color : str, optional
        Line color of the backdrop bands. Default ``"0.82"``.
    colorbar : bool, optional
        If True, add a colorbar labeled ``colorbar_label`` in the
        ``"color"`` mode. The ``"size"`` mode draws no colorbar.
        Default False.
    colorbar_label : str, optional
        Colorbar label text. Default ``"weight"``.
    alpha : float, optional
        Marker alpha. Default 0.75.
    shift_fermi : bool, optional
        If True, subtract the carrier Fermi energy from the carrier
        eigenvalues. Default True.
    xlabel : Optional[str], optional
        x-axis label text. Default ``None`` selects
        ``r"$k$ ($\mathrm{\AA}^{-1}$)"`` for a physical momentum axis
        and ``"k-point index"`` otherwise.
    ylabel : str, optional
        y-axis label text. Default is :math:`E - E_F` in eV.
    title : str, optional
        Axis title text. Default is empty.

    Returns
    -------
    fig : Figure
        Matplotlib figure object.
    ax : Axes
        Axis used for plotting.
    scatter : PathCollection
        Scatter artist returned by Matplotlib.

    Raises
    ------
    ValueError
        If the weights and the band eigenvalues have different shapes.

    See Also
    --------
    plot_band_scatter_preset : Plot projected bands as
        marker-size-weighted scatter points from a named preset in
        :mod:`diffpes.plots.band_scatter`.
    """
    band_values: Float64[NDArray, "n_k n_bands"] = _band_values(
        bands, shift_fermi
    )
    weight_values: Float64[NDArray, "n_k n_bands"] = np.asarray(
        weights, dtype=np.float64
    )
    if weight_values.shape != band_values.shape:
        msg: str = (
            "Weights must have shape matching the band eigenvalues "
            "(n_k, n_bands)."
        )
        raise ValueError(msg)
    momentum_values: Float64[NDArray, " n_k"]
    default_xlabel: str
    momentum_values, default_xlabel = _resolve_momentum(
        momentum_axis, band_values.shape[0]
    )
    resolved_xlabel: str = default_xlabel if xlabel is None else xlabel

    fig: Union[Figure, SubFigure]
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure
    ax: Axes

    if backdrop:
        ax.plot(
            momentum_values,
            band_values,
            color=backdrop_color,
            linewidth=0.3,
        )

    xvals: Float64[NDArray, "n_k n_bands"] = np.broadcast_to(
        momentum_values[:, np.newaxis],
        band_values.shape,
    )
    scatter: PathCollection
    if mode == "size":
        scatter = ax.scatter(
            xvals.ravel(),
            band_values.ravel(),
            s=size_scale * np.abs(weight_values).ravel(),
            color=color,
            alpha=alpha,
            edgecolors="none",
        )
    else:
        scatter = ax.scatter(
            xvals.ravel(),
            band_values.ravel(),
            s=4.0,
            c=weight_values.ravel(),
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            alpha=alpha,
            edgecolors="none",
        )
        if colorbar:
            fig.colorbar(scatter, ax=ax, label=colorbar_label)

    ax.set_xlabel(resolved_xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    plot_result: Tuple[Union[Figure, SubFigure], Axes, PathCollection] = (
        fig,
        ax,
        scatter,
    )
    return plot_result


__all__: list[str] = [
    "plot_band_dispersion",
    "plot_band_scatter_weights",
    "plot_bands_over_spectrum",
]
