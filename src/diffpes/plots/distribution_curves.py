r"""Plot energy and momentum distribution curves from spectral maps.

Extended Summary
----------------
The module provides Matplotlib functions that render one-dimensional
cuts through an energy-momentum intensity map. The cuts include energy
distribution curves (EDCs), momentum distribution curves (MDCs), and
momentum profiles integrated over an energy window. Each function
selects the map row or column nearest to a requested physical value.

Routine Listings
----------------
:func:`plot_distribution_curves`
    Plot a stack of EDCs or MDCs from an intensity map.
:func:`plot_edc_mdc_panels`
    Plot one EDC panel beside one MDC panel from an intensity map.
:func:`plot_momentum_profile`
    Plot the momentum profile integrated over an energy window.

Notes
-----
These functions operate on host-side NumPy arrays and Matplotlib
objects. Use them for analysis visualizations outside JAX-compiled
functions.
"""

import numpy as np
from beartype import beartype
from beartype.typing import List, Literal, Optional, Tuple, Union
from jaxtyping import Array, Bool, Float64, jaxtyped
from matplotlib import colormaps
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure, SubFigure
from matplotlib.lines import Line2D
from numpy.typing import NDArray


@jaxtyped(typechecker=beartype)
def _select_curve(
    values: Float64[NDArray, "n_k n_e"],
    momentum_values: Float64[NDArray, " n_k"],
    energy_values: Float64[NDArray, " n_e"],
    kind: Literal["edc", "mdc"],
    position: float,
    momentum_unit: str = r"$\mathrm{\AA}^{-1}$",
) -> Tuple[Float64[NDArray, " n_x"], Float64[NDArray, " n_x"], str]:
    r"""PRIVATE: Select one EDC or MDC and format its curve label.

    The helper resolves one requested position to the nearest axis
    sample. For an EDC the position is a momentum value and the helper
    returns the intensity row against the energy axis. For an MDC the
    position is an energy value and the helper returns the intensity
    column against the momentum axis.

    Implementation Logic
    --------------------
    1. **Resolve the nearest sample index**::

           row: int = int(np.argmin(np.abs(momentum_values - position)))

       The nearest-sample rule tolerates positions between grid points.

    2. **Slice the intensity map along the fixed index**::

           curve = values[row, :]

       An EDC fixes the momentum row; an MDC fixes the energy column.

    3. **Format the label from the selected axis value**::

           label = f"$k = {selected:+.2f}$ " + momentum_unit

       The label states the selected grid value, not the request.

    Parameters
    ----------
    values : Float64[NDArray, "n_k n_e"]
        Intensity on the momentum-energy grid in 1/eV.
    momentum_values : Float64[NDArray, " n_k"]
        Physical momentum axis in 1/Angstrom.
    energy_values : Float64[NDArray, " n_e"]
        Energy axis relative to the Fermi level in eV.
    kind : Literal["edc", "mdc"]
        Cut direction selector. ``"edc"`` selects a momentum row and
        ``"mdc"`` selects an energy column.
    position : float
        Requested momentum in 1/Angstrom for ``"edc"``, or requested
        energy in eV for ``"mdc"``.
    momentum_unit : str, optional
        Unit string appended to EDC momentum labels. Default is the
        inverse-Angstrom unit; pass ``"rad"`` for detector-angle cuts.

    Returns
    -------
    abscissa : Float64[NDArray, " n_x"]
        Horizontal coordinate of the cut. The energy axis for an EDC
        and the momentum axis for an MDC.
    curve : Float64[NDArray, " n_x"]
        Intensity values along the cut in 1/eV.
    label : str
        Curve label with the selected axis value and its unit.
    """
    if kind == "edc":
        row: int = int(np.argmin(np.abs(momentum_values - position)))
        abscissa: Float64[NDArray, " n_x"] = energy_values
        curve: Float64[NDArray, " n_x"] = values[row, :]
        selected: float = float(momentum_values[row])
        label: str = f"$k = {selected:+.2f}$ " + momentum_unit
    else:
        column: int = int(np.argmin(np.abs(energy_values - position)))
        abscissa = momentum_values
        curve = values[:, column]
        selected = float(energy_values[column])
        label = f"$E - E_F = {selected:+.2f}$ eV"
    selection: Tuple[
        Float64[NDArray, " n_x"], Float64[NDArray, " n_x"], str
    ] = (abscissa, curve, label)
    return selection


@beartype
def _curve_color(
    index: int,
    n_curves: int,
    colors: Optional[Tuple[str, ...]],
    cmap: Optional[str],
) -> Optional[Union[str, Tuple[float, float, float, float]]]:
    """PRIVATE: Resolve the color of one curve in a stack.

    The helper applies the color precedence of
    :func:`plot_distribution_curves`. Explicit colors win over a
    colormap, and a colormap wins over the Matplotlib property cycle.

    Parameters
    ----------
    index : int
        Zero-based index of the curve in the stack.
    n_curves : int
        Total number of curves in the stack.
    colors : Optional[Tuple[str, ...]]
        Explicit color sequence. The helper cycles through it with the
        curve index modulo the sequence length.
    cmap : Optional[str]
        Matplotlib colormap name. The helper samples the colormap at
        ``index / max(n_curves - 1, 1)``.

    Returns
    -------
    line_color : Optional[Union[str, Tuple[float, float, float, float]]]
        Explicit color string, sampled RGBA tuple, or ``None`` when the
        caller uses the default Matplotlib property cycle.

    Notes
    -----
    The ``max(n_curves - 1, 1)`` denominator keeps the sample fraction
    finite for a single-curve stack.
    """
    line_color: Optional[Union[str, Tuple[float, float, float, float]]]
    if colors is not None:
        cycled: str = colors[index % len(colors)]
        line_color = cycled
    elif cmap is not None:
        fraction: float = index / max(n_curves - 1, 1)
        sampled: Tuple[float, float, float, float] = tuple(
            float(channel) for channel in colormaps[cmap](fraction)
        )
        line_color = sampled
    else:
        line_color = None
    return line_color


@beartype
def _default_axis_labels(
    kind: Literal["edc", "mdc"],
    normalized: bool,
    log_counts: bool,
    offset: float,
    momentum_unit: str = r"$\mathrm{\AA}^{-1}$",
) -> Tuple[str, str]:
    r"""PRIVATE: Resolve the default axis labels of a curve stack.

    The helper maps the cut direction to the horizontal label and the
    intensity transformations to the vertical label.

    Parameters
    ----------
    kind : Literal["edc", "mdc"]
        Cut direction selector. ``"edc"`` labels the horizontal axis
        with energy and ``"mdc"`` labels it with momentum.
    normalized : bool
        If True, the vertical label reads ``"normalized intensity"``.
        This transformation wins over ``log_counts`` because the
        normalization acts last on the plotted values.
    log_counts : bool
        If True and ``normalized`` is False, the vertical label reads
        ``"log(1 + intensity)"``.
    offset : float
        Waterfall offset per curve. A non-zero value appends
        ``", offset per curve"`` to the vertical label.
    momentum_unit : str, optional
        Unit string in the MDC momentum label. Default is the
        inverse-Angstrom unit; pass ``"rad"`` for detector-angle cuts.

    Returns
    -------
    xlabel_text : str
        Horizontal axis label with units.
    ylabel_text : str
        Vertical axis label that names the plotted intensity quantity.

    Notes
    -----
    Without a transformation the vertical label reads
    ``"intensity (1/eV)"``.
    """
    xlabel_text: str = (
        r"$E - E_F$ (eV)" if kind == "edc" else rf"$k$ ({momentum_unit})"
    )
    if normalized:
        ylabel_text: str = "normalized intensity"
    elif log_counts:
        ylabel_text = "log(1 + intensity)"
    else:
        ylabel_text = "intensity (1/eV)"
    if offset != 0.0:
        ylabel_text = ylabel_text + ", offset per curve"
    labels: Tuple[str, str] = (xlabel_text, ylabel_text)
    return labels


@jaxtyped(typechecker=beartype)
def plot_distribution_curves(  # noqa: PLR0913, PLR0917
    intensity: Float64[Array, "n_k n_e"] | Float64[NDArray, "n_k n_e"],
    momentum_axis: Float64[Array, " n_k"] | Float64[NDArray, " n_k"],
    energy_axis: Float64[Array, " n_e"] | Float64[NDArray, " n_e"],
    kind: Literal["edc", "mdc"],
    positions: Tuple[float, ...],
    ax: Optional[Axes] = None,
    offset: float = 0.0,
    normalized: bool = False,
    log_counts: bool = False,
    cmap: Optional[str] = None,
    colors: Optional[Tuple[str, ...]] = None,
    linewidth: float = 1.4,
    legend: bool = True,
    legend_title: Optional[str] = None,
    fermi_line: bool = True,
    momentum_unit: str = r"$\mathrm{\AA}^{-1}$",
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    title: str = "",
) -> Tuple[Union[Figure, SubFigure], Axes, List[Line2D]]:
    r"""Plot a stack of EDCs or MDCs from an intensity map.

    The function draws one curve for each requested position. For
    ``kind="edc"`` each position is a momentum value in 1/Angstrom, and
    each curve shows intensity against energy at the nearest momentum
    row. For ``kind="mdc"`` each position is an energy value in eV, and
    each curve shows intensity against momentum at the nearest energy
    column. A non-zero offset stacks the curves as a waterfall. The
    function accepts an existing axis. It creates a figure and axis
    when the caller supplies none.

    :see: :class:`~.test_distribution_curves.TestPlotDistributionCurves`

    Implementation Logic
    --------------------
    1. **Normalize the plotting arrays**::

           values = np.asarray(intensity, dtype=np.float64)

       The conversion accepts JAX and NumPy inputs alike. With
       ``log_counts`` the values become ``np.log1p(values)``.

    2. **Select and transform each curve**::

           abscissa, curve, label = _select_curve(
               values, momentum_values, energy_values, kind, position
           )

       With ``normalized`` the function divides each curve by its own
       maximum. It adds the waterfall term ``offset * index`` last.

    3. **Draw each curve with its resolved color**::

           (line,) = ax.plot(abscissa, curve, label=label, ...)

       Explicit ``colors`` cycle first, then a sampled ``cmap``, then
       the default Matplotlib property cycle.

    4. **Draw the decorations and return the Matplotlib objects**::

           ax.axvline(0.0, color="0.45", linewidth=0.9, linestyle="--")

       The Fermi guide applies only to EDCs. The legend, labels, and
       title complete the axis before the return.

    Parameters
    ----------
    intensity : Float64[Array, "n_k n_e"] | Float64[NDArray, "n_k n_e"]
        Intensity on the momentum-energy grid in 1/eV.
    momentum_axis : Float64[Array, " n_k"] | Float64[NDArray, " n_k"]
        Physical momentum axis in 1/Angstrom.
    energy_axis : Float64[Array, " n_e"] | Float64[NDArray, " n_e"]
        Energy axis relative to the Fermi level in eV.
    kind : Literal["edc", "mdc"]
        Cut direction selector. ``"edc"`` plots intensity against
        energy and ``"mdc"`` plots intensity against momentum.
    positions : Tuple[float, ...]
        Requested cut positions. Momentum values in 1/Angstrom for
        ``"edc"``, energy values in eV for ``"mdc"``. Each position
        selects the nearest axis sample.
    ax : Optional[Axes], optional
        Existing axis for the curves. If ``None``, the function creates
        a figure and axis.
    offset : float, optional
        Vertical offset added per successive curve. Default 0.0, which
        overlays the curves.
    normalized : bool, optional
        If True, divide each curve by its own maximum before the
        offset. Default False.
    log_counts : bool, optional
        If True, apply ``np.log1p`` to the intensity before the
        normalization and the offset. Use this option for detector
        counts. Default False.
    cmap : Optional[str], optional
        Matplotlib colormap name sampled evenly across the stack. Used
        only when ``colors`` is ``None``.
    colors : Optional[Tuple[str, ...]], optional
        Explicit color sequence applied cyclically. Wins over ``cmap``.
    linewidth : float, optional
        Line width of each curve in points. Default 1.4.
    legend : bool, optional
        If True, draw the legend with the auto-formatted curve labels.
        Default True.
    legend_title : Optional[str], optional
        Legend title text. Default ``None``.
    fermi_line : bool, optional
        If True, draw a dashed vertical guide at zero energy on an EDC
        plot. The ``"mdc"`` kind ignores this option. Default True.
    momentum_unit : str, optional
        Unit string for momentum labels in the legend and on the MDC
        axis. Default is the inverse-Angstrom unit; pass ``"rad"`` when
        the momentum axis holds detector angles.
    xlabel : Optional[str], optional
        x-axis label text. If ``None``, the function derives the label
        from ``kind``.
    ylabel : Optional[str], optional
        y-axis label text. If ``None``, the function derives the label
        from the intensity transformations.
    title : str, optional
        Axis title text. Default is empty.

    Returns
    -------
    fig : Figure
        Matplotlib figure object.
    ax : Axes
        Axis used for plotting.
    lines : List[Line2D]
        Curve artists in the order of ``positions``.

    Notes
    -----
    The curve labels state the selected grid values, not the requested
    positions. EDC labels read like :math:`k = +0.10` in the supplied
    momentum unit and MDC labels read like :math:`E - E_F = -0.25` eV.
    """
    values: Float64[NDArray, "n_k n_e"] = np.asarray(
        intensity, dtype=np.float64
    )
    momentum_values: Float64[NDArray, " n_k"] = np.asarray(
        momentum_axis, dtype=np.float64
    )
    energy_values: Float64[NDArray, " n_e"] = np.asarray(
        energy_axis, dtype=np.float64
    )
    if log_counts:
        values = np.log1p(values)

    fig: Union[Figure, SubFigure]
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure
    ax: Axes

    lines: List[Line2D] = []
    index: int
    position: float
    for index, position in enumerate(positions):
        abscissa: Float64[NDArray, " n_x"]
        curve: Float64[NDArray, " n_x"]
        label: str
        abscissa, curve, label = _select_curve(
            values,
            momentum_values,
            energy_values,
            kind,
            float(position),
            momentum_unit,
        )
        if normalized:
            curve = curve / np.max(curve)
        curve = curve + offset * index
        line_color: Optional[Union[str, Tuple[float, float, float, float]]] = (
            _curve_color(index, len(positions), colors, cmap)
        )
        line: Line2D
        if line_color is None:
            (line,) = ax.plot(
                abscissa, curve, linewidth=linewidth, label=label
            )
        else:
            (line,) = ax.plot(
                abscissa,
                curve,
                color=line_color,
                linewidth=linewidth,
                label=label,
            )
        lines.append(line)

    if fermi_line and kind == "edc":
        ax.axvline(0.0, color="0.45", linewidth=0.9, linestyle="--")
    if legend:
        ax.legend(title=legend_title)

    xlabel_default: str
    ylabel_default: str
    xlabel_default, ylabel_default = _default_axis_labels(
        kind, normalized, log_counts, offset, momentum_unit
    )
    ax.set_xlabel(xlabel if xlabel is not None else xlabel_default)
    ax.set_ylabel(ylabel if ylabel is not None else ylabel_default)
    ax.set_title(title)
    plot_result: Tuple[Union[Figure, SubFigure], Axes, List[Line2D]] = (
        fig,
        ax,
        lines,
    )
    return plot_result


@jaxtyped(typechecker=beartype)
def plot_edc_mdc_panels(  # noqa: PLR0913, PLR0917
    intensity: Float64[Array, "n_k n_e"] | Float64[NDArray, "n_k n_e"],
    momentum_axis: Float64[Array, " n_k"] | Float64[NDArray, " n_k"],
    energy_axis: Float64[Array, " n_e"] | Float64[NDArray, " n_e"],
    k_value: float,
    energy_value: float,
    axes: Optional[Tuple[Axes, Axes]] = None,
    color: str = "tab:blue",
    linewidth: float = 1.4,
    fermi_line: bool = True,
    momentum_unit: str = r"$\mathrm{\AA}^{-1}$",
    suptitle: str = "",
) -> Tuple[Union[Figure, SubFigure], Tuple[Axes, Axes], Tuple[Line2D, Line2D]]:
    r"""Plot one EDC panel beside one MDC panel from an intensity map.

    The function draws a twin-panel summary of one spectral cut. The
    left panel shows the EDC at the momentum nearest ``k_value``. The
    right panel shows the MDC at the energy nearest ``energy_value``.
    Each panel title states the selected physical value. The function
    accepts two existing axes. It creates a two-panel figure when the
    caller supplies none.

    :see: :class:`~.test_distribution_curves.TestPlotEdcMdcPanels`

    Implementation Logic
    --------------------
    1. **Normalize the plotting arrays**::

           values = np.asarray(intensity, dtype=np.float64)

       The conversion accepts JAX and NumPy inputs alike.

    2. **Create or reuse the two panels**::

           fig, (edc_ax, mdc_ax) = plt.subplots(
               1, 2, constrained_layout=True
           )

       Caller-supplied axes keep their parent figure.

    3. **Select and draw both cuts**::

           abscissa, curve, label = _select_curve(
               values, momentum_values, energy_values, kind, position
           )

       The EDC panel fixes the momentum row and the MDC panel fixes
       the energy column. Each panel title prefixes the cut label.

    4. **Return the Matplotlib objects**::

           return plot_result

       The result pairs the two axes and the two curve artists.

    Parameters
    ----------
    intensity : Float64[Array, "n_k n_e"] | Float64[NDArray, "n_k n_e"]
        Intensity on the momentum-energy grid in 1/eV.
    momentum_axis : Float64[Array, " n_k"] | Float64[NDArray, " n_k"]
        Physical momentum axis in 1/Angstrom.
    energy_axis : Float64[Array, " n_e"] | Float64[NDArray, " n_e"]
        Energy axis relative to the Fermi level in eV.
    k_value : float
        Requested EDC momentum in 1/Angstrom. The panel uses the
        nearest momentum sample.
    energy_value : float
        Requested MDC energy in eV. The panel uses the nearest energy
        sample.
    axes : Optional[Tuple[Axes, Axes]], optional
        Existing ``(edc_ax, mdc_ax)`` pair. If ``None``, the function
        creates a figure with two panels and constrained layout.
    color : str, optional
        Line color of both curves. Default ``"tab:blue"``.
    linewidth : float, optional
        Line width of both curves in points. Default 1.4.
    fermi_line : bool, optional
        If True, draw a dashed vertical guide at zero energy on the
        EDC panel. Default True.
    momentum_unit : str, optional
        Unit string for the EDC title and the MDC momentum axis.
        Default is the inverse-Angstrom unit; pass ``"rad"`` when the
        momentum axis holds detector angles.
    suptitle : str, optional
        Figure title above both panels. Default is empty, which adds
        no title.

    Returns
    -------
    fig : Figure
        Matplotlib figure object.
    axes_pair : Tuple[Axes, Axes]
        The EDC axis and the MDC axis, in that order.
    lines_pair : Tuple[Line2D, Line2D]
        The EDC curve artist and the MDC curve artist, in that order.
    """
    values: Float64[NDArray, "n_k n_e"] = np.asarray(
        intensity, dtype=np.float64
    )
    momentum_values: Float64[NDArray, " n_k"] = np.asarray(
        momentum_axis, dtype=np.float64
    )
    energy_values: Float64[NDArray, " n_e"] = np.asarray(
        energy_axis, dtype=np.float64
    )

    fig: Union[Figure, SubFigure]
    edc_ax: Axes
    mdc_ax: Axes
    if axes is None:
        fig, (edc_ax, mdc_ax) = plt.subplots(1, 2, constrained_layout=True)
    else:
        edc_ax, mdc_ax = axes
        fig = edc_ax.figure

    edc_abscissa: Float64[NDArray, " n_e"]
    edc_curve: Float64[NDArray, " n_e"]
    edc_label: str
    edc_abscissa, edc_curve, edc_label = _select_curve(
        values, momentum_values, energy_values, "edc", k_value, momentum_unit
    )
    edc_line: Line2D
    (edc_line,) = edc_ax.plot(
        edc_abscissa, edc_curve, color=color, linewidth=linewidth
    )
    if fermi_line:
        edc_ax.axvline(0.0, color="0.45", linewidth=0.9, linestyle="--")
    edc_ax.set_xlabel(r"$E - E_F$ (eV)")
    edc_ax.set_ylabel("intensity (1/eV)")
    edc_ax.set_title("EDC at " + edc_label)

    mdc_abscissa: Float64[NDArray, " n_k"]
    mdc_curve: Float64[NDArray, " n_k"]
    mdc_label: str
    mdc_abscissa, mdc_curve, mdc_label = _select_curve(
        values,
        momentum_values,
        energy_values,
        "mdc",
        energy_value,
        momentum_unit,
    )
    mdc_line: Line2D
    (mdc_line,) = mdc_ax.plot(
        mdc_abscissa, mdc_curve, color=color, linewidth=linewidth
    )
    mdc_ax.set_xlabel(rf"$k$ ({momentum_unit})")
    mdc_ax.set_ylabel("intensity (1/eV)")
    mdc_ax.set_title("MDC at " + mdc_label)

    if suptitle:
        fig.suptitle(suptitle)
    plot_result: Tuple[
        Union[Figure, SubFigure], Tuple[Axes, Axes], Tuple[Line2D, Line2D]
    ] = (fig, (edc_ax, mdc_ax), (edc_line, mdc_line))
    return plot_result


@jaxtyped(typechecker=beartype)
def plot_momentum_profile(  # noqa: PLR0913, PLR0917
    intensity: Float64[Array, "n_k n_e"] | Float64[NDArray, "n_k n_e"],
    momentum_axis: Float64[Array, " n_k"] | Float64[NDArray, " n_k"],
    energy_axis: Float64[Array, " n_e"] | Float64[NDArray, " n_e"],
    energy_window: Tuple[float, float],
    ax: Optional[Axes] = None,
    fill: bool = True,
    color: str = "tab:blue",
    normalized: bool = False,
    linewidth: float = 1.8,
    xlabel: str = r"$k$ ($\mathrm{\AA}^{-1}$)",
    ylabel: str = "integrated intensity",
    title: str = "",
) -> Tuple[Union[Figure, SubFigure], Axes, Line2D]:
    r"""Plot the momentum profile integrated over an energy window.

    The function integrates the intensity map over one energy window
    and draws the result against momentum. Use it to summarize the
    momentum weight near the Fermi level or inside a band feature. The
    function accepts an existing axis. It creates a figure and axis
    when the caller supplies none.

    :see: :class:`~.test_distribution_curves.TestPlotMomentumProfile`

    Implementation Logic
    --------------------
    1. **Mask the energy window**::

           window_mask = (energy_values >= energy_window[0]) & (
               energy_values <= energy_window[1]
           )

       A window with fewer than two energy samples raises a
       ``ValueError`` because the trapezoid rule needs an interval.

    2. **Integrate the masked intensity along energy**::

           profile = np.trapezoid(
               window_values, x=window_energies, axis=1
           )

       With ``normalized`` the profile is divided by its maximum.

    3. **Draw the profile and the optional fill**::

           ax.fill_between(
               momentum_values, profile, alpha=0.28, linewidth=0.0
           )

       The fill uses the line color under the curve.

    4. **Return the Matplotlib objects**::

           return plot_result

       This binding keeps axis reuse explicit.

    Parameters
    ----------
    intensity : Float64[Array, "n_k n_e"] | Float64[NDArray, "n_k n_e"]
        Intensity on the momentum-energy grid in 1/eV.
    momentum_axis : Float64[Array, " n_k"] | Float64[NDArray, " n_k"]
        Physical momentum axis in 1/Angstrom.
    energy_axis : Float64[Array, " n_e"] | Float64[NDArray, " n_e"]
        Energy axis relative to the Fermi level in eV.
    energy_window : Tuple[float, float]
        Inclusive ``(low, high)`` energy bounds of the integration
        window in eV.
    ax : Optional[Axes], optional
        Existing axis for the profile. If ``None``, the function
        creates a figure and axis.
    fill : bool, optional
        If True, shade the area under the profile with the line color
        at alpha 0.28. Default True.
    color : str, optional
        Line and fill color. Default ``"tab:blue"``.
    normalized : bool, optional
        If True, divide the profile by its maximum. Default False.
    linewidth : float, optional
        Line width of the profile in points. Default 1.8.
    xlabel : str, optional
        x-axis label text. Default names the momentum in 1/Angstrom.
    ylabel : str, optional
        y-axis label text. Default is ``"integrated intensity"``.
    title : str, optional
        Axis title text. Default is empty.

    Returns
    -------
    fig : Figure
        Matplotlib figure object.
    ax : Axes
        Axis used for plotting.
    line : Line2D
        Profile curve artist.

    Raises
    ------
    ValueError
        If ``energy_window`` selects fewer than two energy samples.
    """
    values: Float64[NDArray, "n_k n_e"] = np.asarray(
        intensity, dtype=np.float64
    )
    momentum_values: Float64[NDArray, " n_k"] = np.asarray(
        momentum_axis, dtype=np.float64
    )
    energy_values: Float64[NDArray, " n_e"] = np.asarray(
        energy_axis, dtype=np.float64
    )
    window_mask: Bool[NDArray, " n_e"] = (
        energy_values >= energy_window[0]
    ) & (energy_values <= energy_window[1])
    window_energies: Float64[NDArray, " n_w"] = energy_values[window_mask]
    if window_energies.shape[0] < 2:  # noqa: PLR2004
        msg: str = (
            "energy_window must select at least two energy samples "
            "for the trapezoid integration."
        )
        raise ValueError(msg)

    window_values: Float64[NDArray, "n_k n_w"] = values[:, window_mask]
    profile: Float64[NDArray, " n_k"] = np.trapezoid(
        window_values, x=window_energies, axis=1
    )
    if normalized:
        profile = profile / np.max(profile)

    fig: Union[Figure, SubFigure]
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure
    ax: Axes

    line: Line2D
    (line,) = ax.plot(
        momentum_values, profile, color=color, linewidth=linewidth
    )
    if fill:
        ax.fill_between(
            momentum_values,
            profile,
            color=color,
            alpha=0.28,
            linewidth=0.0,
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


__all__: list[str] = [
    "plot_distribution_curves",
    "plot_edc_mdc_panels",
    "plot_momentum_profile",
]
