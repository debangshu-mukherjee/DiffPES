"""Plot calibrated detector rasters and count comparisons.

Extended Summary
----------------
The module provides Matplotlib functions that render native detector
rasters. Each function accepts the :class:`DetectorRaster` carrier or
one raw counts block. A carrier supplies native detector and energy
axes, so the images use physical extents. A raw block carries no axes,
so the images use bin indices.

Routine Listings
----------------
:func:`plot_detector_comparison`
    Plot expected and observed detector counts side by side.
:func:`plot_detector_energy_cut`
    Plot an energy-versus-angle detector cut image.
:func:`plot_detector_image`
    Plot an energy-summed angular detector image.
:func:`plot_detector_residual`
    Plot the standardized Poisson residual image of detector counts.

Notes
-----
These functions operate on host-side NumPy arrays and Matplotlib
objects. Use them for analysis visualizations outside JAX-compiled
functions.
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

from diffpes.types import DetectorRaster


@jaxtyped(typechecker=beartype)
def _resolve_counts(  # noqa: DOC105
    source: Union[
        DetectorRaster,
        Float64[Array, "n_u n_v n_e"],
        Float64[NDArray, "n_u n_v n_e"],
    ],
    channel: int,
) -> Tuple[
    Float64[NDArray, "n_u n_v n_e"],
    Optional[Float64[NDArray, " n_u"]],
    Optional[Float64[NDArray, " n_v"]],
    Optional[Float64[NDArray, " n_e"]],
]:
    """PRIVATE: Extract one counts block and its native axes for plotting.

    The helper accepts a :class:`DetectorRaster` carrier or one raw
    counts block. For a carrier it selects one polarization channel and
    returns the native detector and energy axes. For a raw block it
    returns ``None`` for each axis, and the plot functions then use bin
    indices.

    Implementation Logic
    --------------------
    1. **Select the carrier channel**::

           counts = np.asarray(
               source.expected_counts[channel], dtype=np.float64
           )

       A static range check raises ``ValueError`` before the indexing.

    2. **Convert the native carrier axes**::

           u_axis = np.asarray(source.detector_u_axis, dtype=np.float64)

       The conversion moves each axis to the host for Matplotlib.

    3. **Pass one raw block through**::

           counts = np.asarray(source, dtype=np.float64)

       A raw block carries no axes, so each axis entry is ``None``.

    Parameters
    ----------
    source : DetectorRaster | Float64[Array | NDArray, "n_u n_v n_e"]
        Expected-count carrier or one raw counts block.
    channel : int
        Polarization-channel index into the carrier channel axis. A raw
        block ignores this value.

    Returns
    -------
    counts : Float64[NDArray, "n_u n_v n_e"]
        Counts block for one channel.
    u_axis : Optional[Float64[NDArray, " n_u"]]
        Native detector ``T_x`` bin centres in radians, or ``None``.
    v_axis : Optional[Float64[NDArray, " n_v"]]
        Native detector ``T_y`` bin centres in radians, or ``None``.
    energy_axis : Optional[Float64[NDArray, " n_e"]]
        Recorded energy-bin centres in eV, or ``None``.

    Raises
    ------
    ValueError
        If ``channel`` lies outside the carrier channel axis.
    """
    resolved: Tuple[
        Float64[NDArray, "n_u n_v n_e"],
        Optional[Float64[NDArray, " n_u"]],
        Optional[Float64[NDArray, " n_v"]],
        Optional[Float64[NDArray, " n_e"]],
    ]
    if isinstance(source, DetectorRaster):
        n_channel: int = source.expected_counts.shape[0]
        if channel < 0 or channel >= n_channel:
            msg: str = (
                f"Channel index {channel} lies outside the carrier "
                f"channel axis of length {n_channel}."
            )
            raise ValueError(msg)
        resolved = (
            np.asarray(source.expected_counts[channel], dtype=np.float64),
            np.asarray(source.detector_u_axis, dtype=np.float64),
            np.asarray(source.detector_v_axis, dtype=np.float64),
            np.asarray(source.energy_axis, dtype=np.float64),
        )
        return resolved
    resolved = (np.asarray(source, dtype=np.float64), None, None, None)
    return resolved


@jaxtyped(typechecker=beartype)
def plot_detector_image(  # noqa: DOC105, DOC502
    source: Union[
        DetectorRaster,
        Float64[Array, "n_u n_v n_e"],
        Float64[NDArray, "n_u n_v n_e"],
    ],
    channel: int = 0,
    ax: Optional[Axes] = None,
    cmap: str = "magma",
    log_counts: bool = False,
    colorbar: bool = True,
    colorbar_label: str = "expected counts",
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    title: str = "",
) -> Tuple[Union[Figure, SubFigure], Axes, AxesImage]:
    """Plot an energy-summed angular detector image.

    The function sums the counts over the energy axis. It renders the
    angular image with ``matplotlib.axes.Axes.imshow``. The horizontal
    axis carries the detector ``u`` coordinate, and the vertical axis
    carries the detector ``v`` coordinate. The function accepts an
    existing axis. It creates a figure and axis when the caller supplies
    none.

    :see: :class:`~.test_detector_views.TestPlotDetectorImage`

    Implementation Logic
    --------------------
    1. **Resolve the counts block and axes**::

           counts, u_axis, v_axis, energy_axis = _resolve_counts(
               source, channel
           )

       A carrier supplies native axes. A raw block supplies ``None``.

    2. **Sum the counts over the energy axis**::

           image_values = np.sum(counts, axis=-1)

       With ``log_counts`` the values become
       ``np.log1p(image_values)`` after this sum.

    3. **Draw the transposed angular image**::

           image = ax.imshow(
               image_values.T,
               origin="lower",
               aspect="equal",
               extent=extent,
           )

       The transpose places ``v`` vertically and ``u`` horizontally.
       With carrier axes the extent spans the first and last bin
       centres. Without axes the image uses bin indices.

    Parameters
    ----------
    source : DetectorRaster | Float64[Array | NDArray, "n_u n_v n_e"]
        Expected-count carrier or one raw counts block.
    channel : int, optional
        Polarization-channel index for a carrier. A raw block ignores
        this value. Default 0.
    ax : Optional[Axes], optional
        Existing axis for the image. If ``None``, the function creates
        a figure and axis.
    cmap : str, optional
        Matplotlib colormap name. Default is ``"magma"``.
    log_counts : bool, optional
        If True, plot ``log(1 + counts)`` after the energy sum. Default
        False.
    colorbar : bool, optional
        If True, add a colorbar with ``colorbar_label``.
    colorbar_label : str, optional
        Colorbar label text. Default is ``"expected counts"``.
    xlabel : Optional[str], optional
        x-axis label text. ``None`` resolves to ``"detector u (rad)"``
        with carrier axes and to ``"detector u bin"`` without them.
    ylabel : Optional[str], optional
        y-axis label text. ``None`` resolves to ``"detector v (rad)"``
        with carrier axes and to ``"detector v bin"`` without them.
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

    Raises
    ------
    ValueError
        If ``channel`` lies outside the carrier channel axis.
    """
    counts: Float64[NDArray, "n_u n_v n_e"]
    u_axis: Optional[Float64[NDArray, " n_u"]]
    v_axis: Optional[Float64[NDArray, " n_v"]]
    counts, u_axis, v_axis, _ = _resolve_counts(source, channel)

    image_values: Float64[NDArray, "n_u n_v"] = np.sum(counts, axis=-1)
    if log_counts:
        image_values = np.log1p(image_values)

    fig: Union[Figure, SubFigure]
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure
    ax: Axes

    extent: Optional[Tuple[float, float, float, float]] = None
    if u_axis is not None and v_axis is not None:
        extent = (
            float(u_axis[0]),
            float(u_axis[-1]),
            float(v_axis[0]),
            float(v_axis[-1]),
        )
    image: AxesImage = ax.imshow(
        image_values.T,
        origin="lower",
        aspect="equal",
        cmap=cmap,
        extent=extent,
    )
    if colorbar:
        fig.colorbar(image, ax=ax, label=colorbar_label)

    resolved_xlabel: str = (
        xlabel
        if xlabel is not None
        else ("detector u (rad)" if u_axis is not None else "detector u bin")
    )
    resolved_ylabel: str = (
        ylabel
        if ylabel is not None
        else ("detector v (rad)" if v_axis is not None else "detector v bin")
    )
    ax.set_xlabel(resolved_xlabel)
    ax.set_ylabel(resolved_ylabel)
    ax.set_title(title)
    plot_result: Tuple[Union[Figure, SubFigure], Axes, AxesImage] = (
        fig,
        ax,
        image,
    )
    return plot_result


@jaxtyped(typechecker=beartype)
def plot_detector_energy_cut(  # noqa: DOC105, DOC502, PLR0913, PLR0917
    source: Union[
        DetectorRaster,
        Float64[Array, "n_u n_v n_e"],
        Float64[NDArray, "n_u n_v n_e"],
    ],
    channel: int = 0,
    cut_axis: Literal["u", "v"] = "u",
    position: Optional[float] = None,
    ax: Optional[Axes] = None,
    cmap: str = "magma",
    log_counts: bool = True,
    colorbar: bool = True,
    colorbar_label: str = "log(1 + counts)",
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    title: str = "",
) -> Tuple[Union[Figure, SubFigure], Axes, AxesImage]:
    """Plot an energy-versus-angle detector cut image.

    The function slices the counts block at one position on one angular
    axis. It renders the surviving angular coordinate horizontally and
    the energy vertically. The function accepts an existing axis. It
    creates a figure and axis when the caller supplies none.

    :see: :class:`~.test_detector_views.TestPlotDetectorEnergyCut`

    Implementation Logic
    --------------------
    1. **Resolve the counts block and axes**::

           counts, u_axis, v_axis, energy_axis = _resolve_counts(
               source, channel
           )

       A carrier supplies native axes. A raw block supplies ``None``.

    2. **Select the slice index on the other angular axis**::

           slice_index = int(
               np.argmin(np.abs(slice_coordinate - position))
           )

       ``cut_axis`` names the surviving axis. ``position`` is a
       physical value on the other axis, and the nearest bin centre
       wins. Without carrier axes the coordinate is the bin index. A
       ``None`` position selects the central index.

    3. **Draw the transposed cut image**::

           image = ax.imshow(
               cut_values.T,
               origin="lower",
               aspect="auto",
               extent=extent,
           )

       The transpose places energy vertically. With ``log_counts`` the
       values become ``np.log1p(cut_values)`` before the drawing.

    Parameters
    ----------
    source : DetectorRaster | Float64[Array | NDArray, "n_u n_v n_e"]
        Expected-count carrier or one raw counts block.
    channel : int, optional
        Polarization-channel index for a carrier. A raw block ignores
        this value. Default 0.
    cut_axis : Literal["u", "v"], optional
        Surviving angular axis of the cut image. Default is ``"u"``.
    position : Optional[float], optional
        Physical value on the other angular axis in radians. The slice
        uses the nearest bin centre. ``None`` selects the central
        index. Without carrier axes the value addresses bin indices.
    ax : Optional[Axes], optional
        Existing axis for the image. If ``None``, the function creates
        a figure and axis.
    cmap : str, optional
        Matplotlib colormap name. Default is ``"magma"``.
    log_counts : bool, optional
        If True, plot ``log(1 + counts)``. Default True.
    colorbar : bool, optional
        If True, add a colorbar.
    colorbar_label : str, optional
        Colorbar label text. Default is ``"log(1 + counts)"``. The
        default text resolves to ``"expected counts"`` when
        ``log_counts`` is False.
    xlabel : Optional[str], optional
        x-axis label text. ``None`` resolves to the surviving detector
        axis in radians, or to its bin index without carrier axes.
    ylabel : Optional[str], optional
        y-axis label text. ``None`` resolves to ``"energy (eV)"`` with
        carrier axes and to ``"energy bin"`` without them.
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

    Raises
    ------
    ValueError
        If ``channel`` lies outside the carrier channel axis.
    """
    counts: Float64[NDArray, "n_u n_v n_e"]
    u_axis: Optional[Float64[NDArray, " n_u"]]
    v_axis: Optional[Float64[NDArray, " n_v"]]
    energy_axis: Optional[Float64[NDArray, " n_e"]]
    counts, u_axis, v_axis, energy_axis = _resolve_counts(source, channel)

    slice_size: int = counts.shape[1] if cut_axis == "u" else counts.shape[0]
    other_axis: Optional[Float64[NDArray, " n_o"]] = (
        v_axis if cut_axis == "u" else u_axis
    )
    slice_coordinate: Float64[NDArray, " n_o"] = (
        other_axis
        if other_axis is not None
        else np.arange(slice_size, dtype=np.float64)
    )
    slice_index: int = (
        slice_size // 2
        if position is None
        else int(np.argmin(np.abs(slice_coordinate - position)))
    )
    cut_values: Float64[NDArray, "n_a n_e"] = (
        counts[:, slice_index, :]
        if cut_axis == "u"
        else counts[slice_index, :, :]
    )
    if log_counts:
        cut_values = np.log1p(cut_values)
    angular_axis: Optional[Float64[NDArray, " n_a"]] = (
        u_axis if cut_axis == "u" else v_axis
    )

    fig: Union[Figure, SubFigure]
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure
    ax: Axes

    extent: Optional[Tuple[float, float, float, float]] = None
    if angular_axis is not None and energy_axis is not None:
        extent = (
            float(angular_axis[0]),
            float(angular_axis[-1]),
            float(energy_axis[0]),
            float(energy_axis[-1]),
        )
    image: AxesImage = ax.imshow(
        cut_values.T,
        origin="lower",
        aspect="auto",
        cmap=cmap,
        extent=extent,
    )
    resolved_colorbar_label: str = colorbar_label
    if not log_counts and colorbar_label == "log(1 + counts)":
        resolved_colorbar_label = "expected counts"
    if colorbar:
        fig.colorbar(image, ax=ax, label=resolved_colorbar_label)

    resolved_xlabel: str = (
        xlabel
        if xlabel is not None
        else (
            f"detector {cut_axis} (rad)"
            if angular_axis is not None
            else f"detector {cut_axis} bin"
        )
    )
    resolved_ylabel: str = (
        ylabel
        if ylabel is not None
        else ("energy (eV)" if energy_axis is not None else "energy bin")
    )
    ax.set_xlabel(resolved_xlabel)
    ax.set_ylabel(resolved_ylabel)
    ax.set_title(title)
    plot_result: Tuple[Union[Figure, SubFigure], Axes, AxesImage] = (
        fig,
        ax,
        image,
    )
    return plot_result


@jaxtyped(typechecker=beartype)
def plot_detector_comparison(  # noqa: DOC105, PLR0915
    expected: Union[
        DetectorRaster,
        Float64[Array, "n_u n_v n_e"],
        Float64[NDArray, "n_u n_v n_e"],
    ],
    observed: Union[
        DetectorRaster,
        Float64[Array, "m_u m_v m_e"],
        Float64[NDArray, "m_u m_v m_e"],
    ],
    channel: int = 0,
    view: Literal["angular", "energy"] = "angular",
    axes: Optional[Tuple[Axes, Axes]] = None,
    cmap: str = "magma",
    log_counts: bool = False,
    share_scale: bool = True,
    colorbar: bool = True,
    titles: Tuple[str, str] = ("expected", "observed"),
) -> Tuple[Union[Figure, SubFigure], Tuple[Axes, Axes], List[AxesImage]]:
    """Plot expected and observed detector counts side by side.

    The function renders two panels with one shared color scale. The
    left panel shows the expected counts. The right panel shows the
    observed counts, for example one channel of Poisson-sampled counts.
    The ``view`` selector switches between the energy-summed angular
    image and the central energy-versus-``u`` cut.

    :see: :class:`~.test_detector_views.TestPlotDetectorComparison`

    Implementation Logic
    --------------------
    1. **Resolve both counts blocks and check the shapes**::

           if expected_counts.shape != observed_counts.shape:
               raise ValueError(msg)

       The physical extent comes from the first input with carrier
       axes. Both panels share this extent.

    2. **Draw one image per panel**::

           panel_values = (
               np.sum(block, axis=-1)
               if view == "angular"
               else block[:, block.shape[1] // 2, :]
           )

       The ``"angular"`` view sums over energy. The ``"energy"`` view
       slices at the central ``v`` index. With ``log_counts`` the
       values become ``np.log1p(panel_values)``.

    3. **Share the color scale and the colorbar**::

           fig.colorbar(images[0], ax=axes_pair, shrink=0.86)

       With ``share_scale`` both images receive the joint value range
       as their color limits.

    Parameters
    ----------
    expected : DetectorRaster | Float64[Array | NDArray, "n_u n_v n_e"]
        Expected-count carrier or one raw counts block.
    observed : DetectorRaster | Float64[Array | NDArray, "n_u n_v n_e"]
        Observed-count carrier or one raw counts block.
    channel : int, optional
        Polarization-channel index for each carrier input. A raw block
        ignores this value. Default 0.
    view : Literal["angular", "energy"], optional
        ``"angular"`` shows energy-summed ``u``-``v`` images.
        ``"energy"`` shows central-``v`` energy-versus-``u`` cuts.
        Default is ``"angular"``.
    axes : Optional[Tuple[Axes, Axes]], optional
        Existing axis pair for the two panels. If ``None``, the
        function creates one figure with two constrained-layout panels.
    cmap : str, optional
        Matplotlib colormap name. Default is ``"magma"``.
    log_counts : bool, optional
        If True, plot ``log(1 + counts)`` in both panels. Default
        False.
    share_scale : bool, optional
        If True, apply one shared ``(vmin, vmax)`` across both panels.
        Default True.
    colorbar : bool, optional
        If True, add one colorbar that spans both panels.
    titles : Tuple[str, str], optional
        Panel titles. Default is ``("expected", "observed")``.

    Returns
    -------
    fig : Figure
        Matplotlib figure object.
    axes_pair : Tuple[Axes, Axes]
        The two panel axes in ``(expected, observed)`` order.
    images : List[AxesImage]
        The two image artists in ``(expected, observed)`` order.

    Raises
    ------
    ValueError
        If the two counts blocks disagree in shape, or if ``channel``
        lies outside a carrier channel axis.
    """
    expected_counts: Float64[NDArray, "n_u n_v n_e"]
    observed_counts: Float64[NDArray, "m_u m_v m_e"]
    expected_u: Optional[Float64[NDArray, " n_u"]]
    expected_v: Optional[Float64[NDArray, " n_v"]]
    expected_energy: Optional[Float64[NDArray, " n_e"]]
    observed_u: Optional[Float64[NDArray, " m_u"]]
    observed_v: Optional[Float64[NDArray, " m_v"]]
    observed_energy: Optional[Float64[NDArray, " m_e"]]
    expected_counts, expected_u, expected_v, expected_energy = _resolve_counts(
        expected, channel
    )
    observed_counts, observed_u, observed_v, observed_energy = _resolve_counts(
        observed, channel
    )
    if expected_counts.shape != observed_counts.shape:
        msg: str = "The expected and observed counts blocks disagree in shape."
        raise ValueError(msg)
    u_axis: Optional[Float64[NDArray, " n_u"]] = (
        expected_u if expected_u is not None else observed_u
    )
    v_axis: Optional[Float64[NDArray, " n_v"]] = (
        expected_v if expected_v is not None else observed_v
    )
    energy_axis: Optional[Float64[NDArray, " n_e"]] = (
        expected_energy if expected_energy is not None else observed_energy
    )

    fig: Union[Figure, SubFigure]
    axes_pair: Tuple[Axes, Axes]
    left_axis: Axes
    right_axis: Axes
    if axes is None:
        fig, (left_axis, right_axis) = plt.subplots(
            1, 2, constrained_layout=True
        )
        axes_pair = (left_axis, right_axis)
    else:
        axes_pair = axes
        fig = axes[0].figure

    extent: Optional[Tuple[float, float, float, float]] = None
    if view == "angular":
        aspect: Literal["equal", "auto"] = "equal"
        if u_axis is not None and v_axis is not None:
            extent = (
                float(u_axis[0]),
                float(u_axis[-1]),
                float(v_axis[0]),
                float(v_axis[-1]),
            )
        resolved_xlabel: str = (
            "detector u (rad)" if u_axis is not None else "detector u bin"
        )
        resolved_ylabel: str = (
            "detector v (rad)" if v_axis is not None else "detector v bin"
        )
    else:
        aspect = "auto"
        if u_axis is not None and energy_axis is not None:
            extent = (
                float(u_axis[0]),
                float(u_axis[-1]),
                float(energy_axis[0]),
                float(energy_axis[-1]),
            )
        resolved_xlabel = (
            "detector u (rad)" if u_axis is not None else "detector u bin"
        )
        resolved_ylabel = (
            "energy (eV)" if energy_axis is not None else "energy bin"
        )

    images: List[AxesImage] = []
    panel_axis: Axes
    block: Float64[NDArray, "n_u n_v n_e"]
    panel_title: str
    for panel_axis, block, panel_title in zip(
        axes_pair, (expected_counts, observed_counts), titles, strict=True
    ):
        panel_values: Float64[NDArray, "n_x n_y"] = (
            np.sum(block, axis=-1)
            if view == "angular"
            else block[:, block.shape[1] // 2, :]
        )
        if log_counts:
            panel_values = np.log1p(panel_values)
        panel_image: AxesImage = panel_axis.imshow(
            panel_values.T,
            origin="lower",
            aspect=aspect,
            cmap=cmap,
            extent=extent,
        )
        panel_axis.set_xlabel(resolved_xlabel)
        panel_axis.set_ylabel(resolved_ylabel)
        panel_axis.set_title(panel_title)
        images.append(panel_image)

    if share_scale:
        vmin: float = min(
            float(np.min(np.asarray(image.get_array()))) for image in images
        )
        vmax: float = max(
            float(np.max(np.asarray(image.get_array()))) for image in images
        )
        shared_image: AxesImage
        for shared_image in images:
            shared_image.set_clim(vmin, vmax)
    if colorbar:
        fig.colorbar(images[0], ax=axes_pair, shrink=0.86)

    plot_result: Tuple[
        Union[Figure, SubFigure], Tuple[Axes, Axes], List[AxesImage]
    ] = (fig, axes_pair, images)
    return plot_result


@jaxtyped(typechecker=beartype)
def plot_detector_residual(  # noqa: DOC105
    expected: Union[
        DetectorRaster,
        Float64[Array, "n_u n_v n_e"],
        Float64[NDArray, "n_u n_v n_e"],
    ],
    observed: Union[
        DetectorRaster,
        Float64[Array, "m_u m_v m_e"],
        Float64[NDArray, "m_u m_v m_e"],
    ],
    channel: int = 0,
    ax: Optional[Axes] = None,
    cmap: str = "RdBu_r",
    colorbar: bool = True,
    colorbar_label: str = "standard deviations",
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    title: str = "",
) -> Tuple[Union[Figure, SubFigure], Axes, AxesImage]:
    r"""Plot the standardized Poisson residual image of detector counts.

    The function sums both counts blocks over the energy axis. It then
    standardizes the angular difference with the Poisson standard
    deviation of the expected image. The diverging color scale is
    symmetric around zero. The function accepts an existing axis. It
    creates a figure and axis when the caller supplies none.

    :see: :class:`~.test_detector_views.TestPlotDetectorResidual`

    Implementation Logic
    --------------------
    1. **Resolve both counts blocks and check the shapes**::

           if expected_counts.shape != observed_counts.shape:
               raise ValueError(msg)

       The physical extent comes from the first input with carrier
       axes.

    2. **Standardize the energy-summed difference**::

           residual = (observed_image - expected_image) / np.sqrt(
               np.maximum(expected_image, 1.0)
           )

       The floor of one count in the denominator keeps empty expected
       bins finite.

    3. **Draw the residual with symmetric color limits**::

           image = ax.imshow(
               residual.T, vmin=-scale, vmax=scale, origin="lower"
           )

       The limit ``scale`` equals :math:`\max |residual|`.

    Parameters
    ----------
    expected : DetectorRaster | Float64[Array | NDArray, "n_u n_v n_e"]
        Expected-count carrier or one raw counts block.
    observed : DetectorRaster | Float64[Array | NDArray, "n_u n_v n_e"]
        Observed-count carrier or one raw counts block.
    channel : int, optional
        Polarization-channel index for each carrier input. A raw block
        ignores this value. Default 0.
    ax : Optional[Axes], optional
        Existing axis for the image. If ``None``, the function creates
        a figure and axis.
    cmap : str, optional
        Matplotlib colormap name. Default is ``"RdBu_r"``.
    colorbar : bool, optional
        If True, add a colorbar with ``colorbar_label``.
    colorbar_label : str, optional
        Colorbar label text. Default is ``"standard deviations"``.
    xlabel : Optional[str], optional
        x-axis label text. ``None`` resolves to ``"detector u (rad)"``
        with carrier axes and to ``"detector u bin"`` without them.
    ylabel : Optional[str], optional
        y-axis label text. ``None`` resolves to ``"detector v (rad)"``
        with carrier axes and to ``"detector v bin"`` without them.
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

    Raises
    ------
    ValueError
        If the two counts blocks disagree in shape, or if ``channel``
        lies outside a carrier channel axis.

    Notes
    -----
    An identically zero residual gives a degenerate color range. The
    color limits then fall back to ``(-1.0, 1.0)``.
    """
    expected_counts: Float64[NDArray, "n_u n_v n_e"]
    observed_counts: Float64[NDArray, "m_u m_v m_e"]
    expected_u: Optional[Float64[NDArray, " n_u"]]
    expected_v: Optional[Float64[NDArray, " n_v"]]
    observed_u: Optional[Float64[NDArray, " m_u"]]
    observed_v: Optional[Float64[NDArray, " m_v"]]
    expected_counts, expected_u, expected_v, _ = _resolve_counts(
        expected, channel
    )
    observed_counts, observed_u, observed_v, _ = _resolve_counts(
        observed, channel
    )
    if expected_counts.shape != observed_counts.shape:
        msg: str = "The expected and observed counts blocks disagree in shape."
        raise ValueError(msg)
    u_axis: Optional[Float64[NDArray, " n_u"]] = (
        expected_u if expected_u is not None else observed_u
    )
    v_axis: Optional[Float64[NDArray, " n_v"]] = (
        expected_v if expected_v is not None else observed_v
    )

    expected_image: Float64[NDArray, "n_u n_v"] = np.sum(
        expected_counts, axis=-1
    )
    observed_image: Float64[NDArray, "n_u n_v"] = np.sum(
        observed_counts, axis=-1
    )
    residual: Float64[NDArray, "n_u n_v"] = (
        observed_image - expected_image
    ) / np.sqrt(np.maximum(expected_image, 1.0))
    scale: float = float(np.max(np.abs(residual)))
    if scale == 0.0:
        scale = 1.0

    fig: Union[Figure, SubFigure]
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure
    ax: Axes

    extent: Optional[Tuple[float, float, float, float]] = None
    if u_axis is not None and v_axis is not None:
        extent = (
            float(u_axis[0]),
            float(u_axis[-1]),
            float(v_axis[0]),
            float(v_axis[-1]),
        )
    image: AxesImage = ax.imshow(
        residual.T,
        origin="lower",
        aspect="equal",
        cmap=cmap,
        extent=extent,
        vmin=-scale,
        vmax=scale,
    )
    if colorbar:
        fig.colorbar(image, ax=ax, label=colorbar_label)

    resolved_xlabel: str = (
        xlabel
        if xlabel is not None
        else ("detector u (rad)" if u_axis is not None else "detector u bin")
    )
    resolved_ylabel: str = (
        ylabel
        if ylabel is not None
        else ("detector v (rad)" if v_axis is not None else "detector v bin")
    )
    ax.set_xlabel(resolved_xlabel)
    ax.set_ylabel(resolved_ylabel)
    ax.set_title(title)
    plot_result: Tuple[Union[Figure, SubFigure], Axes, AxesImage] = (
        fig,
        ax,
        image,
    )
    return plot_result


__all__: list[str] = [
    "plot_detector_comparison",
    "plot_detector_energy_cut",
    "plot_detector_image",
    "plot_detector_residual",
]
