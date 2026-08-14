r"""Plot scalar curve families, densities of states, and axis profiles.

Extended Summary
----------------
The module provides Matplotlib functions that render one-dimensional
scalar data. The curve-family plotter draws labeled curves over one
shared axis. The density-of-states plotters draw one curve or an
overlay of curves with Fermi-level context. The planar-average plotter
draws a charge profile along the stacking axis.

Routine Listings
----------------
:func:`plot_curve_family`
    Plot a labeled family of curves over one shared axis.
:func:`plot_dos`
    Plot one total density of states with Fermi-level context.
:func:`plot_dos_overlay`
    Plot several density-of-states curves on one axis.
:func:`plot_planar_average`
    Plot a planar-averaged profile along the stacking axis.

Notes
-----
These functions operate on host-side NumPy arrays and Matplotlib
objects. Use them for analysis visualizations outside JAX-compiled
functions.
"""

import numpy as np
from beartype import beartype
from beartype.typing import List, Optional, Tuple, Union
from jaxtyping import Array, Bool, Float64, jaxtyped
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.colors import Colormap
from matplotlib.figure import Figure, SubFigure
from matplotlib.lines import Line2D
from numpy.typing import NDArray

from diffpes.types import DensityOfStates


@beartype
def _resolve_curve_colors(
    n_curves: int,
    colors: Optional[Tuple[str, ...]],
    cmap: Optional[str],
) -> List[Optional[Union[str, Tuple[float, float, float, float]]]]:
    """PRIVATE: Resolve one plot color for each curve in a family.

    The helper turns the user color options into one per-curve color
    list. Explicit colors take precedence over a colormap name. Without
    both options, each entry becomes ``None``, and Matplotlib uses the
    default property cycle.

    Parameters
    ----------
    n_curves : int
        Number of curves in the family.
    colors : Optional[Tuple[str, ...]]
        Explicit color names. The helper cycles them over the curves.
    cmap : Optional[str]
        Matplotlib colormap name. The helper samples the colormap
        uniformly on ``[0, 1]`` when ``colors`` is ``None``.

    Returns
    -------
    curve_colors : List[Optional[Union[str, Tuple[float, ...]]]]
        One color entry per curve. An entry is a color name, an RGBA
        tuple of four host floats, or ``None``. A ``None`` entry
        selects the default Matplotlib property cycle for that curve.

    Notes
    -----
    With explicit colors, the entry for curve ``i`` is
    ``colors[i % len(colors)]``. With a colormap, the entry is the
    colormap value at position ``i / max(n_curves - 1, 1)``, cast to a
    tuple of host floats.
    """
    curve_colors: List[Optional[Union[str, Tuple[float, float, float, float]]]]
    if colors is not None:
        curve_colors = [
            colors[index % len(colors)] for index in range(n_curves)
        ]
    elif cmap is not None:
        cmap_object: Colormap = plt.get_cmap(cmap)
        positions: Float64[NDArray, " n_curves"] = np.linspace(
            0.0, 1.0, max(n_curves, 1)
        )
        curve_colors = [
            tuple(float(channel) for channel in cmap_object(position))
            for position in positions
        ]
    else:
        curve_colors = [None] * n_curves
    return curve_colors


@jaxtyped(typechecker=beartype)
def _dos_selection(
    dos: DensityOfStates,
    shift_fermi: bool,
    energy_window: Optional[Tuple[float, float]],
    normalized: bool,
) -> Tuple[Float64[NDArray, " n_sel"], Float64[NDArray, " n_sel"], float]:
    """PRIVATE: Select the plotted x and y values of one DOS curve.

    The helper converts one :class:`DensityOfStates` carrier to NumPy
    arrays. It applies the Fermi shift, the optional energy window, and
    the optional normalization. It also reports the Fermi boundary in
    the plotted x coordinates.

    Implementation Logic
    --------------------
    1. **Build the x axis and the boundary**::

           x_values = energy_values - fermi_value

       With ``shift_fermi`` the boundary is ``0.0``. Without the shift,
       the x axis stays absolute and the boundary is the Fermi energy.

    2. **Apply the window mask**::

           mask = (x_values >= low) & (x_values <= high)

       The mask acts on the plotted x coordinates. An empty selection
       raises ``ValueError`` before any drawing happens.

    3. **Normalize and return the selection**::

           y_selected = y_selected / np.max(y_selected)

       The division uses the maximum of the selected values only. The
       named tuple binding keeps the return type explicit.

    Parameters
    ----------
    dos : DensityOfStates
        Density-of-states carrier with the shared energy axis.
    shift_fermi : bool
        If True, plot ``energy - fermi_energy`` on the x axis.
    energy_window : Optional[Tuple[float, float]]
        Inclusive ``(low, high)`` window in the plotted x coordinates,
        in eV.
    normalized : bool
        If True, divide the selected values by their maximum.

    Returns
    -------
    x_selected : Float64[NDArray, " n_sel"]
        Selected x values in eV.
    y_selected : Float64[NDArray, " n_sel"]
        Selected density-of-states values.
    boundary : float
        Fermi boundary in the plotted x coordinates, in eV.

    Raises
    ------
    ValueError
        If ``energy_window`` selects no samples.
    """
    energy_values: Float64[NDArray, " E"] = np.asarray(
        dos.energy, dtype=np.float64
    )
    dos_values: Float64[NDArray, " E"] = np.asarray(
        dos.total_dos, dtype=np.float64
    )
    fermi_value: float = float(np.asarray(dos.fermi_energy, dtype=np.float64))

    x_values: Float64[NDArray, " E"]
    boundary: float
    if shift_fermi:
        x_values = energy_values - fermi_value
        boundary = 0.0
    else:
        x_values = energy_values
        boundary = fermi_value

    mask: Bool[NDArray, " E"]
    if energy_window is None:
        mask = np.ones(x_values.shape[0], dtype=bool)
    else:
        mask = (x_values >= energy_window[0]) & (x_values <= energy_window[1])
        if not bool(np.any(mask)):
            msg: str = "energy_window selects no density-of-states samples."
            raise ValueError(msg)

    x_selected: Float64[NDArray, " n_sel"] = x_values[mask]
    y_selected: Float64[NDArray, " n_sel"] = dos_values[mask]
    if normalized:
        y_selected = y_selected / np.max(y_selected)
    selection: Tuple[
        Float64[NDArray, " n_sel"], Float64[NDArray, " n_sel"], float
    ] = (x_selected, y_selected, boundary)
    return selection


@jaxtyped(typechecker=beartype)
def plot_curve_family(  # noqa: PLR0913, PLR0917
    x_axis: Float64[Array, " n_x"] | Float64[NDArray, " n_x"],
    curves: Tuple[Float64[Array, " n_x"] | Float64[NDArray, " n_x"], ...],
    labels: Optional[Tuple[str, ...]] = None,
    ax: Optional[Axes] = None,
    colors: Optional[Tuple[str, ...]] = None,
    cmap: Optional[str] = None,
    log_x: bool = False,
    log_y: bool = False,
    linewidth: float = 1.8,
    legend: bool = True,
    legend_title: Optional[str] = None,
    xlabel: str = "",
    ylabel: str = "",
    title: str = "",
) -> Tuple[Union[Figure, SubFigure], Axes, List[Line2D]]:
    """Plot a labeled family of curves over one shared axis.

    The function draws one line per curve against the shared x axis.
    Typical inputs are kinematic :math:`k_z` families over photon
    energy, :math:`k_z` weight curves, lineshape profiles, log-log
    cross sections, and Fermi-edge temperature series. The function
    accepts an existing axis. It creates a figure and axis when the
    caller supplies none.

    :see: :class:`~.test_scalar_curves.TestPlotCurveFamily`

    Implementation Logic
    --------------------
    1. **Validate the labels and resolve the colors**::

           line_colors = _resolve_curve_colors(len(curves), colors, cmap)

       A labels tuple with the wrong length raises ``ValueError``
       before any drawing happens. Explicit colors cycle over the
       curves. A colormap name samples uniform positions. Without both,
       the default property cycle applies.

    2. **Draw one line per curve**::

           line = ax.plot(
               x_values,
               curve_values,
               color=line_colors[index],
               linewidth=linewidth,
               label=label_text,
           )[0]

       Every curve shares the converted x axis. A ``None`` color
       selects the next property-cycle color.

    3. **Apply the scales, the legend, and the labels**::

           ax.set_xscale("log")

       Each requested logarithmic scale applies after drawing. The
       legend appears only when labels exist. The named tuple binding
       keeps the return type explicit.

    Parameters
    ----------
    x_axis : Float64[Array, " n_x"] | Float64[NDArray, " n_x"]
        Shared x axis of the curve family.
    curves : Tuple[Float64[Array, " n_x"] | Float64[NDArray, " n_x"], ...]
        One tuple entry per curve, each on the shared x axis.
    labels : Optional[Tuple[str, ...]], optional
        One legend label per curve. Default is ``None``.
    ax : Optional[Axes], optional
        Existing axis for the lines. If ``None``, the function creates
        a figure and axis.
    colors : Optional[Tuple[str, ...]], optional
        Explicit line colors, cycled over the curves. Default is
        ``None``.
    cmap : Optional[str], optional
        Matplotlib colormap name, sampled uniformly when ``colors`` is
        ``None``. Default is ``None``.
    log_x : bool, optional
        If True, apply ``set_xscale("log")``. Default False.
    log_y : bool, optional
        If True, apply ``set_yscale("log")``. Default False.
    linewidth : float, optional
        Line width of each curve. Default is 1.8.
    legend : bool, optional
        If True, draw a legend when labels exist. Default True.
    legend_title : Optional[str], optional
        Legend title text. Default is ``None``.
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
    lines : List[Line2D]
        Line artists in curve order.

    Raises
    ------
    ValueError
        If ``labels`` is not ``None`` and its length differs from the
        length of ``curves``.

    Notes
    -----
    Matplotlib skips non-finite points. A NaN value therefore leaves a
    gap in its curve. The function does not filter NaN values.
    """
    index: int

    if labels is not None and len(labels) != len(curves):
        msg: str = "labels length must equal the curves length."
        raise ValueError(msg)

    x_values: Float64[NDArray, " n_x"] = np.asarray(x_axis, dtype=np.float64)
    line_colors: List[
        Optional[Union[str, Tuple[float, float, float, float]]]
    ] = _resolve_curve_colors(len(curves), colors, cmap)

    fig: Union[Figure, SubFigure]
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure
    ax: Axes

    lines: List[Line2D] = []
    for index in range(len(curves)):
        curve_values: Float64[NDArray, " n_x"] = np.asarray(
            curves[index], dtype=np.float64
        )
        label_text: Optional[str] = (
            labels[index] if labels is not None else None
        )
        line: Line2D = ax.plot(
            x_values,
            curve_values,
            color=line_colors[index],
            linewidth=linewidth,
            label=label_text,
        )[0]
        lines.append(line)

    if log_x:
        ax.set_xscale("log")
    if log_y:
        ax.set_yscale("log")
    if legend and labels is not None:
        ax.legend(title=legend_title)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    plot_result: Tuple[Union[Figure, SubFigure], Axes, List[Line2D]] = (
        fig,
        ax,
        lines,
    )
    return plot_result


@jaxtyped(typechecker=beartype)
def plot_dos(  # noqa: PLR0913, PLR0917
    dos: DensityOfStates,
    ax: Optional[Axes] = None,
    shift_fermi: bool = True,
    shade_occupied: bool = True,
    energy_window: Optional[Tuple[float, float]] = None,
    normalized: bool = False,
    color: str = "tab:blue",
    linewidth: float = 1.8,
    fermi_line: bool = True,
    xlabel: str = r"$E - E_F$ (eV)",
    ylabel: str = "density of states (1/eV)",
    title: str = "",
) -> Tuple[Union[Figure, SubFigure], Axes, Line2D]:
    r"""Plot one total density of states with Fermi-level context.

    The function draws the total density of states against energy. With
    the Fermi shift, the x axis carries :math:`E - E_F`, and the
    occupied side sits at :math:`x \le 0`. The function accepts an
    existing axis. It creates a figure and axis when the caller
    supplies none.

    :see: :class:`~.test_scalar_curves.TestPlotDos`

    Implementation Logic
    --------------------
    1. **Select the plotted values**::

           x_selected, y_selected, boundary = _dos_selection(
               dos, shift_fermi, energy_window, normalized
           )

       The helper applies the Fermi shift, the window mask, and the
       normalization. The boundary is ``0.0`` in shifted coordinates
       and the Fermi energy otherwise.

    2. **Draw the curve and the occupied shading**::

           ax.fill_between(
               x_selected,
               y_selected,
               0.0,
               where=occupied_mask,
               color=color,
               alpha=0.25,
               linewidth=0.0,
           )

       The shading covers the plotted samples with
       ``x_selected <= boundary``, which is the occupied side.

    3. **Draw the Fermi line and the labels**::

           ax.axvline(boundary, linestyle="--", ...)

       The dashed vertical line marks the Fermi boundary. The named
       tuple binding keeps the return type explicit.

    Parameters
    ----------
    dos : DensityOfStates
        Density-of-states carrier to plot.
    ax : Optional[Axes], optional
        Existing axis for the curve. If ``None``, the function creates
        a figure and axis.
    shift_fermi : bool, optional
        If True, plot ``energy - fermi_energy`` on the x axis. Default
        True.
    shade_occupied : bool, optional
        If True, fill the area under the curve on the occupied side
        with alpha 0.25. Default True.
    energy_window : Optional[Tuple[float, float]], optional
        Inclusive ``(low, high)`` window in the plotted x coordinates,
        in eV. Default is ``None``.
    normalized : bool, optional
        If True, divide the selected values by their maximum. Default
        False.
    color : str, optional
        Curve and shading color. Default is ``"tab:blue"``.
    linewidth : float, optional
        Line width of the curve. Default is 1.8.
    fermi_line : bool, optional
        If True, draw a dashed vertical line at the Fermi boundary.
        Default True.
    xlabel : str, optional
        x-axis label text. Default is :math:`E - E_F` in eV.
    ylabel : str, optional
        y-axis label text. Default is ``"density of states (1/eV)"``.
    title : str, optional
        Axis title text. Default is empty.

    Returns
    -------
    fig : Figure
        Matplotlib figure object.
    ax : Axes
        Axis used for plotting.
    line : Line2D
        Line artist of the density-of-states curve.

    Notes
    -----
    An ``energy_window`` that selects no samples raises ``ValueError``
    inside the selection helper. When ``shift_fermi`` is False, the
    occupied boundary is the Fermi energy itself.

    See Also
    --------
    plot_dos_overlay : Plot several density-of-states curves on one axis.
    """
    x_selected: Float64[NDArray, " n_sel"]
    y_selected: Float64[NDArray, " n_sel"]
    boundary: float
    x_selected, y_selected, boundary = _dos_selection(
        dos, shift_fermi, energy_window, normalized
    )

    fig: Union[Figure, SubFigure]
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure
    ax: Axes

    line: Line2D = ax.plot(
        x_selected,
        y_selected,
        color=color,
        linewidth=linewidth,
    )[0]
    if shade_occupied:
        occupied_mask: Bool[NDArray, " n_sel"] = x_selected <= boundary
        ax.fill_between(
            x_selected,
            y_selected,
            0.0,
            where=occupied_mask,
            color=color,
            alpha=0.25,
            linewidth=0.0,
        )
    if fermi_line:
        ax.axvline(
            boundary,
            color="black",
            linestyle="--",
            linewidth=0.9,
            alpha=0.7,
        )

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    plot_result: Tuple[Union[Figure, SubFigure], Axes, Line2D] = (
        fig,
        ax,
        line,
    )
    return plot_result


@jaxtyped(typechecker=beartype)
def plot_dos_overlay(  # noqa: PLR0913, PLR0917
    dos_curves: Tuple[DensityOfStates, ...],
    labels: Optional[Tuple[str, ...]] = None,
    ax: Optional[Axes] = None,
    shift_fermi: bool = True,
    normalized: bool = True,
    energy_window: Optional[Tuple[float, float]] = None,
    colors: Optional[Tuple[str, ...]] = None,
    linewidth: float = 1.7,
    legend: bool = True,
    fermi_line: bool = True,
    xlabel: str = r"$E - E_F$ (eV)",
    ylabel: str = "normalized density of states",
    title: str = "",
) -> Tuple[Union[Figure, SubFigure], Axes, List[Line2D]]:
    """Plot several density-of-states curves on one axis.

    The function draws one line per density-of-states carrier. Each
    curve receives its own Fermi shift, window mask, and normalization.
    The overlay draws no occupied shading. The function accepts an
    existing axis. It creates a figure and axis when the caller
    supplies none.

    :see: :class:`~.test_scalar_curves.TestPlotDosOverlay`

    Implementation Logic
    --------------------
    1. **Validate the labels**::

           len(labels) != len(dos_curves)

       A labels tuple with the wrong length raises ``ValueError``
       before any drawing happens.

    2. **Draw one selected curve per carrier**::

           x_selected, y_selected, boundary = _dos_selection(
               dos_curves[index], shift_fermi, energy_window, normalized
           )

       The per-curve selection normalizes each curve by its own
       maximum. Explicit colors cycle over the curves.

    3. **Draw the Fermi lines, the legend, and the labels**::

           ax.axvline(boundary_value, linestyle="--", ...)

       One dashed line appears per distinct boundary. With the Fermi
       shift, all boundaries collapse to ``0.0``, and one line appears.
       The named tuple binding keeps the return type explicit.

    Parameters
    ----------
    dos_curves : Tuple[DensityOfStates, ...]
        Density-of-states carriers to overlay.
    labels : Optional[Tuple[str, ...]], optional
        One legend label per carrier. Default is ``None``.
    ax : Optional[Axes], optional
        Existing axis for the curves. If ``None``, the function creates
        a figure and axis.
    shift_fermi : bool, optional
        If True, plot ``energy - fermi_energy`` per curve. Default
        True.
    normalized : bool, optional
        If True, divide each curve by its own selected maximum. Default
        True.
    energy_window : Optional[Tuple[float, float]], optional
        Inclusive ``(low, high)`` window in the plotted x coordinates,
        in eV. Default is ``None``.
    colors : Optional[Tuple[str, ...]], optional
        Explicit line colors, cycled over the curves. Default is
        ``None``.
    linewidth : float, optional
        Line width of each curve. Default is 1.7.
    legend : bool, optional
        If True, draw a legend when labels exist. Default True.
    fermi_line : bool, optional
        If True, draw a dashed vertical line per distinct Fermi
        boundary. Default True.
    xlabel : str, optional
        x-axis label text. Default is :math:`E - E_F` in eV.
    ylabel : str, optional
        y-axis label text. Default is
        ``"normalized density of states"``.
    title : str, optional
        Axis title text. Default is empty.

    Returns
    -------
    fig : Figure
        Matplotlib figure object.
    ax : Axes
        Axis used for plotting.
    lines : List[Line2D]
        Line artists in carrier order.

    Raises
    ------
    ValueError
        If ``labels`` is not ``None`` and its length differs from the
        length of ``dos_curves``.

    Notes
    -----
    An ``energy_window`` that selects no samples of a curve raises
    ``ValueError`` inside the selection helper.

    See Also
    --------
    plot_dos : Plot one total density of states with Fermi-level
        context.
    """
    index: int
    boundary_value: float

    if labels is not None and len(labels) != len(dos_curves):
        msg: str = "labels length must equal the dos_curves length."
        raise ValueError(msg)

    fig: Union[Figure, SubFigure]
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure
    ax: Axes

    lines: List[Line2D] = []
    boundaries: List[float] = []
    for index in range(len(dos_curves)):
        x_selected: Float64[NDArray, " n_sel"]
        y_selected: Float64[NDArray, " n_sel"]
        boundary: float
        x_selected, y_selected, boundary = _dos_selection(
            dos_curves[index], shift_fermi, energy_window, normalized
        )
        line_color: Optional[str] = (
            colors[index % len(colors)] if colors is not None else None
        )
        label_text: Optional[str] = (
            labels[index] if labels is not None else None
        )
        line: Line2D = ax.plot(
            x_selected,
            y_selected,
            color=line_color,
            linewidth=linewidth,
            label=label_text,
        )[0]
        lines.append(line)
        boundaries.append(boundary)

    if fermi_line:
        for boundary_value in sorted(set(boundaries)):
            ax.axvline(
                boundary_value,
                color="black",
                linestyle="--",
                linewidth=0.9,
                alpha=0.7,
            )
    if legend and labels is not None:
        ax.legend()

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    plot_result: Tuple[Union[Figure, SubFigure], Axes, List[Line2D]] = (
        fig,
        ax,
        lines,
    )
    return plot_result


@jaxtyped(typechecker=beartype)
def plot_planar_average(  # noqa: PLR0913, PLR0917
    positions: Float64[Array, " n_z"] | Float64[NDArray, " n_z"],
    profile: Float64[Array, " n_z"] | Float64[NDArray, " n_z"],
    ax: Optional[Axes] = None,
    fill: bool = True,
    color: str = "tab:blue",
    linewidth: float = 1.4,
    xlabel: str = r"$z$ ($\mathrm{\AA}$)",
    ylabel: str = r"planar-averaged charge (e/$\mathrm{\AA}^{3}$)",
    title: str = "",
) -> Tuple[Union[Figure, SubFigure], Axes, Line2D]:
    r"""Plot a planar-averaged profile along the stacking axis.

    The function draws one profile against the stacking-axis positions.
    Typical inputs are the outputs of
    :func:`diffpes.inout.planar_average`. The x limits pin to the
    position range, and the y axis starts at zero. The function accepts
    an existing axis. It creates a figure and axis when the caller
    supplies none.

    :see: :class:`~.test_scalar_curves.TestPlotPlanarAverage`

    Implementation Logic
    --------------------
    1. **Normalize the plotting arrays**::

           position_values = np.asarray(positions, dtype=np.float64)

       The conversion accepts JAX and NumPy inputs alike.

    2. **Draw the profile and the optional fill**::

           ax.fill_between(
               position_values,
               profile_values,
               0.0,
               color=color,
               alpha=0.25,
               linewidth=0.0,
           )

       The fill shades the full area under the curve.

    3. **Pin the axis limits**::

           ax.set_xlim(position_min, position_max)
           ax.set_ylim(0.0, None)

       The x limits follow the position range. The y floor stays at
       zero. The named tuple binding keeps the return type explicit.

    Parameters
    ----------
    positions : Float64[Array, " n_z"] | Float64[NDArray, " n_z"]
        Stacking-axis positions in Angstrom.
    profile : Float64[Array, " n_z"] | Float64[NDArray, " n_z"]
        Planar-averaged charge density in electrons per cubic Angstrom.
    ax : Optional[Axes], optional
        Existing axis for the profile. If ``None``, the function
        creates a figure and axis.
    fill : bool, optional
        If True, shade the area under the curve with alpha 0.25.
        Default True.
    color : str, optional
        Curve and fill color. Default is ``"tab:blue"``.
    linewidth : float, optional
        Line width of the profile. Default is 1.4.
    xlabel : str, optional
        x-axis label text. Default names :math:`z` in Angstrom.
    ylabel : str, optional
        y-axis label text. Default names the planar-averaged charge in
        electrons per cubic Angstrom.
    title : str, optional
        Axis title text. Default is empty.

    Returns
    -------
    fig : Figure
        Matplotlib figure object.
    ax : Axes
        Axis used for plotting.
    line : Line2D
        Line artist of the profile curve.
    """
    position_values: Float64[NDArray, " n_z"] = np.asarray(
        positions, dtype=np.float64
    )
    profile_values: Float64[NDArray, " n_z"] = np.asarray(
        profile, dtype=np.float64
    )

    fig: Union[Figure, SubFigure]
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure
    ax: Axes

    line: Line2D = ax.plot(
        position_values,
        profile_values,
        color=color,
        linewidth=linewidth,
    )[0]
    if fill:
        ax.fill_between(
            position_values,
            profile_values,
            0.0,
            color=color,
            alpha=0.25,
            linewidth=0.0,
        )

    ax.set_xlim(float(np.min(position_values)), float(np.max(position_values)))
    ax.set_ylim(0.0, None)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    plot_result: Tuple[Union[Figure, SubFigure], Axes, Line2D] = (
        fig,
        ax,
        line,
    )
    return plot_result


__all__: list[str] = [
    "plot_curve_family",
    "plot_dos",
    "plot_dos_overlay",
    "plot_planar_average",
]
