"""Plot ARPES spectra with analysis utilities.

Extended Summary
----------------
The module provides Matplotlib functions that render publication-style ARPES
intensity maps. Inputs use :class:`~diffpes.types.ArpesSpectrum` directly.

Routine Listings
----------------
:func:`apply_kpath_ticks`
    Apply symmetry-point ticks/labels from KPathInfo to an axis.
:func:`list_band_scatter_presets`
    Return supported preset names for projected band scatter plots.
:func:`plot_arpes_spectrum`
    Plot an ARPES intensity map from an ArpesSpectrum PyTree.
:func:`plot_arpes_with_kpath`
    Plot ARPES spectrum and annotate k-axis using KPathInfo.
:func:`plot_band_scatter_preset`
    Plot projected bands as marker-size-weighted scatter points.
:func:`plot_band_scatter_with_kpath`
    Plot projected band scatter and annotate x-axis with k-path labels.

Notes
-----
These functions operate on host-side NumPy arrays and Matplotlib objects. Use
them for analysis visualizations. Do not include them in functions that
``jax.jit`` compiles.
"""

import numpy as np
from beartype import beartype
from beartype.typing import Literal, Optional, Tuple, Union
from jaxtyping import Float64, Int32, jaxtyped
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.collections import PathCollection
from matplotlib.figure import Figure, SubFigure
from matplotlib.image import AxesImage
from numpy.typing import NDArray

from diffpes.types import (
    BAND_NDIM,
    ENERGY_AXIS_NDIM,
    INTENSITY_NDIM,
    ORBITAL_INDEX,
    PRESET_NAMES,
    ArpesSpectrum,
    BandStructure,
    KPathInfo,
    OrbitalProjection,
    SpinOrbitalProjection,
)


@beartype
def _prepare_plot_arrays(
    spectrum: ArpesSpectrum,
) -> Tuple[Float64[NDArray, "K E"], Float64[NDArray, " E"]]:
    """Convert and validate spectrum arrays for plotting.

    Extended Summary
    ----------------
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
    clim: Optional[tuple[float, float]] = None,
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
    clim : Optional[tuple[float, float]], optional
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

           indices: list[int] = np.asarray(
               kpath.label_indices, dtype=np.int32
           ).tolist()
           labels: list[str] = list(kpath.labels)

       Host-side lists match the Matplotlib tick API and preserve label order.

    2. **Apply the shared label count**::

           n_labels: int = min(len(indices), len(labels))
           ticks: list[float] = [float(idx) for idx in indices[:n_labels]]

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


@jaxtyped(typechecker=beartype)
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
    clim : Optional[tuple[float, float]], optional
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
def list_band_scatter_presets() -> tuple[str, ...]:
    """Return supported preset names for projected band scatter plots.

    Returns the canonical immutable names accepted by the projected-band
    plotting functions.

    :see: :class:`~.test_plotting.TestListBandScatterPresets`

    Returns
    -------
    presets : tuple[str, ...]
        Available preset names accepted by
        :func:`plot_band_scatter_preset`.

    Notes
    -----
    The function returns the tuple without allocating or reordering it.
    """
    presets: tuple[str, ...] = PRESET_NAMES
    return presets


@beartype
def _prepare_band_arrays(
    bands: BandStructure,
) -> tuple[Float64[NDArray, "K B"], float]:
    """Convert and validate band arrays for scatter plotting."""
    eigenvalues: Float64[NDArray, "K B"] = np.asarray(
        bands.eigenvalues, dtype=np.float64
    )
    if eigenvalues.ndim != BAND_NDIM:
        msg: str = "Expected bands.eigenvalues to have shape (K, B)."
        raise ValueError(msg)
    fermi: float = float(np.asarray(bands.fermi_energy, dtype=np.float64))
    band_arrays: tuple[Float64[NDArray, "K B"], float] = (
        eigenvalues,
        fermi,
    )
    return band_arrays


@beartype
def _subset_atom_axis(
    data: Float64[NDArray, "K B A C"],
    atom_indices: Optional[list[int]],
) -> Float64[NDArray, "K B A2 C"]:
    """Subset an array on atom axis when the caller provides atom indices."""
    if atom_indices is None:
        subset: Float64[NDArray, "K B A2 C"] = data
    else:
        idx: Int32[NDArray, " A"] = np.asarray(atom_indices, dtype=np.int32)
        subset = data[:, :, idx, :]
    return subset


@beartype
def _weights_from_preset(  # noqa: PLR0912
    orb_proj: Union[OrbitalProjection, SpinOrbitalProjection],
    preset: str,
    atom_indices: Optional[list[int]],
) -> tuple[Float64[NDArray, "K B"], bool]:
    """Resolve a band-weight matrix from a preset name.

    Parameters
    ----------
    orb_proj : Union[OrbitalProjection, SpinOrbitalProjection]
        Projection carrier containing orbital and optional spin/OAM data.
    preset : str
        Preset name selecting the reduction to compute.
    atom_indices : Optional[list[int]]
        Optional zero-based atom subset.

    Returns
    -------
    weights : Float64[NDArray, "K B"]
        Band weights with shape ``(K, B)``.
    signed : bool
        Whether the weights have signs and require a color map.

    Raises
    ------
    ValueError
        If the preset is unknown or requires unavailable spin or OAM data.
    """
    key: str = preset.lower()
    proj: Float64[NDArray, "K B A O"] = _subset_atom_axis(
        np.asarray(orb_proj.projections, dtype=np.float64),
        atom_indices,
    )

    orbital_shells: dict[str, slice] = {
        "p": slice(1, 4),
        "d": slice(4, 9),
        "non_s": slice(1, 9),
    }
    weights: Optional[Float64[NDArray, "K B"]] = None
    signed: bool = False

    if key in ORBITAL_INDEX:
        idx: int = ORBITAL_INDEX[key]
        weights = np.sum(proj[..., idx], axis=2)
    elif key in orbital_shells:
        weights = np.sum(proj[..., orbital_shells[key]], axis=(2, 3))
    elif key == "total":
        weights = np.sum(proj, axis=(2, 3))

    spin_arr: Optional[Float64[NDArray, "K B A 6"]] = None
    if orb_proj.spin is not None:
        spin_arr = _subset_atom_axis(
            np.asarray(orb_proj.spin, dtype=np.float64),
            atom_indices,
        )
    spin_channel: dict[str, int] = {
        "spin_x_up": 0,
        "spin_x_down": 1,
        "spin_y_up": 2,
        "spin_y_down": 3,
        "spin_z_up": 4,
        "spin_z_down": 5,
    }
    spin_net: dict[str, tuple[int, int]] = {
        "spin_x": (0, 1),
        "spin_y": (2, 3),
        "spin_z": (4, 5),
    }
    if weights is None and key.startswith("spin_"):
        if spin_arr is None:
            msg: str = (
                f"Preset '{preset}' requires spin data, but spin is None."
            )
            raise ValueError(msg)
        if key in spin_channel:
            weights = np.sum(spin_arr[..., spin_channel[key]], axis=2)
        elif key in spin_net:
            up_idx: int
            dn_idx: int
            up_idx, dn_idx = spin_net[key]
            weights = np.sum(
                spin_arr[..., up_idx] - spin_arr[..., dn_idx],
                axis=2,
            )
            signed = True

    oam_arr: Optional[Float64[NDArray, "K B A C"]] = None
    if orb_proj.oam is not None:
        oam_arr = _subset_atom_axis(
            np.asarray(orb_proj.oam, dtype=np.float64),
            atom_indices,
        )
    oam_component: dict[str, int] = {
        "oam_p": 0,
        "oam_d": 1,
        "oam_total": 2,
    }
    if weights is None and key.startswith("oam_"):
        if oam_arr is None:
            msg = f"Preset '{preset}' requires OAM data, but oam is None."
            raise ValueError(msg)
        if key in oam_component:
            weights = np.sum(oam_arr[..., oam_component[key]], axis=2)
            signed = True
        elif key == "oam_abs_total":
            weights = np.sum(np.abs(oam_arr[..., 2]), axis=2)
            signed = False

    if weights is None:
        presets: str = ", ".join(PRESET_NAMES)
        msg = f"Unknown preset '{preset}'. Available presets: {presets}."
        raise ValueError(msg)
    preset_weights: tuple[Float64[NDArray, "K B"], bool] = (weights, signed)
    return preset_weights


@jaxtyped(typechecker=beartype)
def plot_band_scatter_preset(  # noqa: PLR0913
    bands: BandStructure,
    orb_proj: Union[OrbitalProjection, SpinOrbitalProjection],
    preset: str = "p",
    atom_indices: Optional[list[int]] = None,
    ax: Optional[Axes] = None,
    shift_fermi: bool = True,
    size_scale: float = 250.0,
    min_size: float = 0.5,
    alpha: float = 0.75,
    color: str = "tab:blue",
    cmap: str = "coolwarm",
    colorbar: bool = False,
    xlabel: str = "Momentum (k)",
    ylabel: str = "Energy (eV)",
    title: str = "Projected Band Scatter",
) -> tuple[Union[Figure, SubFigure], Axes, PathCollection]:
    """Plot projected bands as marker-size-weighted scatter points.

    Builds a fat-band style scatter plot from a named preset. Marker
    size encodes projection magnitude. For signed presets (spin net
    components and OAM channels), marker color encodes sign/value via a
    colormap.

    :see: :class:`~.test_plotting.TestPlotBandScatterPreset`

    Implementation Logic
    --------------------
    1. **Resolve the preset weights**::

           weights, signed = _weights_from_preset(
               orb_proj, preset, atom_indices
           )

       This reduces the requested orbital data to one value per band.

    2. **Scale the marker areas**::

           marker_sizes: Float64[NDArray, "K B"] = np.maximum(
               np.abs(weights) * size_scale,
               min_size,
           )

       The magnitude controls area while the lower bound keeps points visible.

    3. **Return the plot objects**::

           return plot_result

       The caller receives the figure, axis, and scatter artist.

    Parameters
    ----------
    bands : BandStructure
        Band structure with ``eigenvalues`` shape ``(K, B)``.
    orb_proj : Union[OrbitalProjection, SpinOrbitalProjection]
        Projection object containing orbital weights and optional spin/OAM.
    preset : str, optional
        Preset key from :func:`list_band_scatter_presets`.
    atom_indices : Optional[list[int]], optional
        Optional 0-based atom indices used before reduction.
    ax : Optional[Axes], optional
        Existing axis for the plot. If ``None``, the function creates a figure
        and axis.
    shift_fermi : bool, optional
        If True, plot ``E - E_F`` on y-axis.
    size_scale : float, optional
        Linear scale factor for marker sizes.
    min_size : float, optional
        Minimum marker size in points^2.
    alpha : float, optional
        Marker alpha.
    color : str, optional
        Solid marker color for non-signed presets.
    cmap : str, optional
        Colormap name used for signed presets.
    colorbar : bool, optional
        If ``True`` and the preset has signs, draw a colorbar.
    xlabel : str, optional
        X-axis label.
    ylabel : str, optional
        Y-axis label.
    title : str, optional
        Plot title.

    Returns
    -------
    fig : Figure or SubFigure
        Parent figure.
    ax : Axes
        Axis used for plotting.
    scatter : PathCollection
        Scatter artist returned by Matplotlib.

    Raises
    ------
    ValueError
        If the preset weights and band eigenvalues have different shapes.
    """
    eigenvalues: Float64[NDArray, "K B"]
    fermi: float
    eigenvalues, fermi = _prepare_band_arrays(bands)
    weights: Float64[NDArray, "K B"]
    signed: bool
    weights, signed = _weights_from_preset(orb_proj, preset, atom_indices)
    if weights.shape != eigenvalues.shape:
        msg: str = (
            "Preset weights must have shape matching bands.eigenvalues (K, B)."
        )
        raise ValueError(msg)

    yvals: Float64[NDArray, "K B"] = (
        eigenvalues - fermi if shift_fermi else eigenvalues
    )
    nkpoints: int = yvals.shape[0]
    xvals: Float64[NDArray, "K B"] = np.broadcast_to(
        np.arange(nkpoints, dtype=np.float64)[:, np.newaxis],
        yvals.shape,
    )
    marker_sizes: Float64[NDArray, "K B"] = np.maximum(
        np.abs(weights) * size_scale,
        min_size,
    )

    fig: Union[Figure, SubFigure]
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure
    ax: Axes

    scatter: PathCollection
    if signed:
        scatter = ax.scatter(
            xvals.ravel(),
            yvals.ravel(),
            s=marker_sizes.ravel(),
            c=weights.ravel(),
            cmap=cmap,
            alpha=alpha,
            edgecolors="none",
        )
        if colorbar:
            fig.colorbar(scatter, ax=ax, label=f"{preset} weight")
    else:
        scatter = ax.scatter(
            xvals.ravel(),
            yvals.ravel(),
            s=marker_sizes.ravel(),
            color=color,
            alpha=alpha,
            edgecolors="none",
        )

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    plot_result: tuple[Union[Figure, SubFigure], Axes, PathCollection] = (
        fig,
        ax,
        scatter,
    )
    return plot_result


@jaxtyped(typechecker=beartype)
def plot_band_scatter_with_kpath(  # noqa: PLR0913
    bands: BandStructure,
    orb_proj: Union[OrbitalProjection, SpinOrbitalProjection],
    kpath: KPathInfo,
    preset: str = "p",
    atom_indices: Optional[list[int]] = None,
    ax: Optional[Axes] = None,
    shift_fermi: bool = True,
    size_scale: float = 250.0,
    min_size: float = 0.5,
    alpha: float = 0.75,
    color: str = "tab:blue",
    cmap: str = "coolwarm",
    colorbar: bool = False,
    xlabel: str = "Momentum (k)",
    ylabel: str = "Energy (eV)",
    title: str = "Projected Band Scatter",
    draw_symmetry_lines: bool = True,
) -> tuple[Union[Figure, SubFigure], Axes, PathCollection]:
    """Plot projected band scatter and annotate x-axis with k-path labels.

    Combines a projected-band scatter plot with the symmetry labels from
    one :class:`KPathInfo` carrier.

    :see: :class:`~.test_plotting.TestPlotBandScatterWithKpath`

    Implementation Logic
    --------------------
    1. **Create the projected-band scatter plot**::

           fig, ax, scatter = plot_band_scatter_preset(
               bands=bands,
               orb_proj=orb_proj,
               preset=preset,
               atom_indices=atom_indices,
               ax=ax,
               shift_fermi=shift_fermi,
               size_scale=size_scale,
               min_size=min_size,
               alpha=alpha,
               color=color,
               cmap=cmap,
               colorbar=colorbar,
               xlabel=xlabel,
               ylabel=ylabel,
               title=title,
           )

       The base plotter owns the marker construction and axis styling.

    2. **Apply the k-path labels**::

           apply_kpath_ticks(
               ax=ax,
               kpath=kpath,
               draw_symmetry_lines=draw_symmetry_lines,
               line_color="black",
               line_alpha=0.35,
           )

       This adds symmetry metadata to the existing axis.

    3. **Return the plot objects**::

           return plot_result

       The caller receives the figure, axis, and scatter artist.

    Parameters
    ----------
    bands : BandStructure
        Band structure with ``eigenvalues`` shape ``(K, B)``.
    orb_proj : Union[OrbitalProjection, SpinOrbitalProjection]
        Projection object containing orbital weights and optional spin/OAM.
    kpath : KPathInfo
        Symmetry labels and indices for the plotted k-point path.
    preset : str, optional
        Preset key from :func:`list_band_scatter_presets`.
    atom_indices : Optional[list[int]], optional
        Optional zero-based atom indices used before reduction.
    ax : Optional[Axes], optional
        Existing axis to draw on, or ``None`` to create one.
    shift_fermi : bool, optional
        Whether to plot energies relative to the Fermi level.
    size_scale : float, optional
        Linear scale factor for marker sizes.
    min_size : float, optional
        Minimum marker size in points squared.
    alpha : float, optional
        Marker opacity.
    color : str, optional
        Solid marker color for unsigned presets.
    cmap : str, optional
        Colormap used for signed presets.
    colorbar : bool, optional
        Whether to draw a colorbar for signed presets.
    xlabel : str, optional
        Horizontal axis label.
    ylabel : str, optional
        Vertical energy-axis label in eV.
    title : str, optional
        Plot title.
    draw_symmetry_lines : bool, optional
        Whether to draw vertical lines at symmetry points.

    Returns
    -------
    fig : Figure or SubFigure
        Parent figure.
    ax : Axes
        Axis used for plotting.
    scatter : PathCollection
        Scatter artist returned by Matplotlib.
    """
    fig: Union[Figure, SubFigure]
    scatter: PathCollection
    fig, ax, scatter = plot_band_scatter_preset(
        bands=bands,
        orb_proj=orb_proj,
        preset=preset,
        atom_indices=atom_indices,
        ax=ax,
        shift_fermi=shift_fermi,
        size_scale=size_scale,
        min_size=min_size,
        alpha=alpha,
        color=color,
        cmap=cmap,
        colorbar=colorbar,
        xlabel=xlabel,
        ylabel=ylabel,
        title=title,
    )
    ax: Axes
    apply_kpath_ticks(
        ax=ax,
        kpath=kpath,
        draw_symmetry_lines=draw_symmetry_lines,
        line_color="black",
        line_alpha=0.35,
    )
    plot_result: tuple[Union[Figure, SubFigure], Axes, PathCollection] = (
        fig,
        ax,
        scatter,
    )
    return plot_result


__all__: list[str] = [
    "apply_kpath_ticks",
    "list_band_scatter_presets",
    "plot_arpes_spectrum",
    "plot_arpes_with_kpath",
    "plot_band_scatter_preset",
    "plot_band_scatter_with_kpath",
]
