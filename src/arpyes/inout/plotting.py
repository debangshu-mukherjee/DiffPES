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
from beartype.typing import Literal, Optional, Tuple, Union
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.collections import PathCollection
from matplotlib.figure import Figure, SubFigure
from matplotlib.image import AxesImage

from arpyes.types import (
    ArpesSpectrum,
    BandStructure,
    KPathInfo,
    OrbitalProjection,
    SpinOrbitalProjection,
)

_INTENSITY_NDIM: int = 2
_ENERGY_AXIS_NDIM: int = 1
_BAND_NDIM: int = 2

_ORBITAL_INDEX: dict[str, int] = {
    "s": 0,
    "py": 1,
    "pz": 2,
    "px": 3,
    "dxy": 4,
    "dyz": 5,
    "dz2": 6,
    "dxz": 7,
    "dx2y2": 8,
}
_PRESET_NAMES: tuple[str, ...] = (
    "s",
    "py",
    "pz",
    "px",
    "p",
    "dxy",
    "dyz",
    "dz2",
    "dxz",
    "dx2y2",
    "d",
    "non_s",
    "total",
    "spin_x_up",
    "spin_x_down",
    "spin_y_up",
    "spin_y_down",
    "spin_z_up",
    "spin_z_down",
    "spin_x",
    "spin_y",
    "spin_z",
    "oam_p",
    "oam_d",
    "oam_total",
    "oam_abs_total",
)


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
) -> Tuple[Union[Figure, SubFigure], Axes, AxesImage]:
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

    fig: Union[Figure, SubFigure]
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
) -> Tuple[Union[Figure, SubFigure], Axes, AxesImage]:
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
    apply_kpath_ticks(
        ax=ax,
        kpath=kpath,
        draw_symmetry_lines=draw_symmetry_lines,
    )
    return fig, ax, image


@beartype
def list_band_scatter_presets() -> tuple[str, ...]:
    """Return supported preset names for projected band scatter plots.

    Returns
    -------
    tuple[str, ...]
        Available preset names accepted by
        :func:`plot_band_scatter_preset`.
    """
    return _PRESET_NAMES


@beartype
def _prepare_band_arrays(
    bands: BandStructure,
) -> tuple[np.ndarray, float]:
    """Convert and validate band arrays for scatter plotting."""
    eigenvalues: np.ndarray = np.asarray(bands.eigenvalues, dtype=np.float64)
    if eigenvalues.ndim != _BAND_NDIM:
        msg: str = "Expected bands.eigenvalues to have shape (K, B)."
        raise ValueError(msg)
    fermi: float = float(np.asarray(bands.fermi_energy, dtype=np.float64))
    return eigenvalues, fermi


@beartype
def _subset_atom_axis(
    data: np.ndarray,
    atom_indices: Optional[list[int]],
) -> np.ndarray:
    """Subset an array on atom axis (axis=2) when atom indices are provided."""
    if atom_indices is None:
        return data
    idx: np.ndarray = np.asarray(atom_indices, dtype=np.int32)
    return data[:, :, idx, :]


@beartype
def _weights_from_preset(  # noqa: PLR0912
    orb_proj: Union[OrbitalProjection, SpinOrbitalProjection],
    preset: str,
    atom_indices: Optional[list[int]],
) -> tuple[np.ndarray, bool]:
    """Resolve a band-weight matrix from a preset name.

    Returns
    -------
    weights : np.ndarray
        Band weights with shape ``(K, B)``.
    signed : bool
        Whether weights are signed and should be color-mapped.
    """
    key: str = preset.lower()
    proj: np.ndarray = _subset_atom_axis(
        np.asarray(orb_proj.projections, dtype=np.float64),
        atom_indices,
    )

    orbital_shells: dict[str, slice] = {
        "p": slice(1, 4),
        "d": slice(4, 9),
        "non_s": slice(1, 9),
    }
    weights: Optional[np.ndarray] = None
    signed: bool = False

    if key in _ORBITAL_INDEX:
        idx: int = _ORBITAL_INDEX[key]
        weights = np.sum(proj[..., idx], axis=2)
    elif key in orbital_shells:
        weights = np.sum(proj[..., orbital_shells[key]], axis=(2, 3))
    elif key == "total":
        weights = np.sum(proj, axis=(2, 3))

    spin_arr: Optional[np.ndarray] = None
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
            msg = f"Preset '{preset}' requires spin data, but spin is None."
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

    oam_arr: Optional[np.ndarray] = None
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
        presets: str = ", ".join(_PRESET_NAMES)
        msg = f"Unknown preset '{preset}'. Available presets: {presets}."
        raise ValueError(msg)
    return weights, signed


@beartype
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

    Extended Summary
    ----------------
    Builds a fat-band style scatter plot from a named preset. Marker
    size encodes projection magnitude. For signed presets (spin net
    components and OAM channels), marker color encodes sign/value via a
    colormap.

    Parameters
    ----------
    bands : BandStructure
        Band structure with ``eigenvalues`` shape ``(K, B)``.
    orb_proj : OrbitalProjection or SpinOrbitalProjection
        Projection object containing orbital weights and optional spin/OAM.
    preset : str, optional
        Preset key from :func:`list_band_scatter_presets`.
    atom_indices : Optional[list[int]], optional
        Optional 0-based atom indices used before reduction.
    ax : Optional[Axes], optional
        Existing axis to draw on. If None, a new figure/axis is created.
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
        If True and preset is signed, draw a colorbar.
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
    """
    eigenvalues: np.ndarray
    fermi: float
    eigenvalues, fermi = _prepare_band_arrays(bands)
    weights: np.ndarray
    signed: bool
    weights, signed = _weights_from_preset(orb_proj, preset, atom_indices)
    if weights.shape != eigenvalues.shape:
        msg = (
            "Preset weights must have shape matching "
            "bands.eigenvalues (K, B)."
        )
        raise ValueError(msg)

    yvals: np.ndarray = eigenvalues - fermi if shift_fermi else eigenvalues
    nkpoints: int = yvals.shape[0]
    xvals: np.ndarray = np.broadcast_to(
        np.arange(nkpoints, dtype=np.float64)[:, np.newaxis],
        yvals.shape,
    )
    marker_sizes: np.ndarray = np.maximum(
        np.abs(weights) * size_scale,
        min_size,
    )

    fig: Union[Figure, SubFigure]
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

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
    return fig, ax, scatter


@beartype
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
    apply_kpath_ticks(
        ax=ax,
        kpath=kpath,
        draw_symmetry_lines=draw_symmetry_lines,
        line_color="black",
        line_alpha=0.35,
    )
    return fig, ax, scatter


__all__: list[str] = [
    "apply_kpath_ticks",
    "list_band_scatter_presets",
    "plot_band_scatter_preset",
    "plot_band_scatter_with_kpath",
    "plot_arpes_spectrum",
    "plot_arpes_with_kpath",
]
