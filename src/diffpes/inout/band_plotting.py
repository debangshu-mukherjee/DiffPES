r"""Plot projected tight-binding bands with analysis utilities.

Extended Summary
----------------
This module renders projected bands as marker-size-weighted scatter plots
and provides named orbital, spin, and angular-momentum presets.

Routine Listings
----------------
:func:`list_band_scatter_presets`
    Return supported preset names for projected band scatter plots.
:func:`plot_band_scatter_preset`
    Plot projected bands as marker-size-weighted scatter points.
:func:`plot_band_scatter_with_kpath`
    Plot projected band scatter and annotate x-axis with k-path labels.

Notes
-----
These functions operate on host-side NumPy arrays and Matplotlib objects.
Use them for analysis visualizations outside JAX-compiled functions.
"""

import numpy as np
from beartype import beartype
from beartype.typing import Dict, List, Optional, Tuple, Union
from jaxtyping import Float64, Int32, jaxtyped
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.collections import PathCollection
from matplotlib.figure import Figure, SubFigure
from numpy.typing import NDArray

from diffpes.constants import BAND_NDIM, ORBITAL_INDEX, PRESET_NAMES
from diffpes.types import (
    BandStructure,
    KPathInfo,
    OrbitalProjection,
    SpinOrbitalProjection,
)

from .plotting import apply_kpath_ticks


@jaxtyped(typechecker=beartype)
def list_band_scatter_presets() -> Tuple[str, ...]:
    """Return supported preset names for projected band scatter plots.

    Returns the canonical immutable names accepted by the projected-band
    plotting functions.

    :see: :class:`~.test_band_plotting.TestListBandScatterPresets`

    Returns
    -------
    presets : Tuple[str, ...]
        Available preset names accepted by
        :func:`plot_band_scatter_preset`.

    Notes
    -----
    The function returns the tuple without allocating or reordering it.
    """
    presets: Tuple[str, ...] = PRESET_NAMES
    return presets


@beartype
def _prepare_band_arrays(
    bands: BandStructure,
) -> Tuple[Float64[NDArray, "K B"], float]:
    """PRIVATE: Convert and validate band arrays for scatter plotting.

    Parameters
    ----------
    bands : BandStructure
        Band-structure carrier with ``eigenvalues`` in eV and
        ``fermi_energy`` in eV.

    Returns
    -------
    eigenvalues : Float64[NDArray, "K B"]
        Band energies in eV as a NumPy array of shape ``(K, B)``.
    fermi : float
        Fermi energy in eV as a host float.

    Raises
    ------
    ValueError
        If ``bands.eigenvalues`` is not two dimensional.

    Notes
    -----
    Converts both fields with :func:`np.asarray` to ``float64`` and
    checks only the eigenvalue rank; downstream plotting relies on the
    ``(K, B)`` layout.
    """
    eigenvalues: Float64[NDArray, "K B"] = np.asarray(
        bands.eigenvalues, dtype=np.float64
    )
    if eigenvalues.ndim != BAND_NDIM:
        msg: str = "Expected bands.eigenvalues to have shape (K, B)."
        raise ValueError(msg)
    fermi: float = float(np.asarray(bands.fermi_energy, dtype=np.float64))
    band_arrays: Tuple[Float64[NDArray, "K B"], float] = (
        eigenvalues,
        fermi,
    )
    return band_arrays


@beartype
def _subset_atom_axis(
    data: Float64[NDArray, "K B A C"],
    atom_indices: Optional[List[int]],
) -> Float64[NDArray, "K B A2 C"]:
    """PRIVATE: Subset an array on atom axis when the caller provides atom
    indices.

    Parameters
    ----------
    data : Float64[NDArray, "K B A C"]
        Projection data with the atom axis in position two.
    atom_indices : Optional[List[int]]
        Zero-based atom indices to keep, or ``None`` for all atoms.

    Returns
    -------
    subset : Float64[NDArray, "K B A2 C"]
        The input array unchanged for ``None``, otherwise the selected
        atom rows in the requested order.

    Notes
    -----
    Fancy-indexes axis two with an ``int32`` index array; repeated
    indices duplicate rows and out-of-range indices raise the NumPy
    ``IndexError``.
    """
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
    atom_indices: Optional[List[int]],
) -> Tuple[Float64[NDArray, "K B"], bool]:
    """PRIVATE: Resolve a band-weight matrix from a preset name.

    Parameters
    ----------
    orb_proj : Union[OrbitalProjection, SpinOrbitalProjection]
        Projection carrier containing orbital and optional spin/OAM data.
    preset : str
        Preset name selecting the reduction to compute.
    atom_indices : Optional[List[int]]
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

    orbital_shells: Dict[str, slice] = {
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
    spin_channel: Dict[str, int] = {
        "spin_x_up": 0,
        "spin_x_down": 1,
        "spin_y_up": 2,
        "spin_y_down": 3,
        "spin_z_up": 4,
        "spin_z_down": 5,
    }
    spin_net: Dict[str, Tuple[int, int]] = {
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
    oam_component: Dict[str, int] = {
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
    preset_weights: Tuple[Float64[NDArray, "K B"], bool] = (weights, signed)
    return preset_weights


@jaxtyped(typechecker=beartype)
def plot_band_scatter_preset(  # noqa: PLR0913, PLR0917
    bands: BandStructure,
    orb_proj: Union[OrbitalProjection, SpinOrbitalProjection],
    preset: str = "p",
    atom_indices: Optional[List[int]] = None,
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
) -> Tuple[Union[Figure, SubFigure], Axes, PathCollection]:
    """Plot projected bands as marker-size-weighted scatter points.

    Builds a fat-band style scatter plot from a named preset. Marker
    size encodes projection magnitude. For signed presets (spin net
    components and OAM channels), marker color encodes sign/value via a
    colormap.

    :see: :class:`~.test_band_plotting.TestPlotBandScatterPreset`

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
    atom_indices : Optional[List[int]], optional
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
    plot_result: Tuple[Union[Figure, SubFigure], Axes, PathCollection] = (
        fig,
        ax,
        scatter,
    )
    return plot_result


@jaxtyped(typechecker=beartype)
def plot_band_scatter_with_kpath(  # noqa: PLR0913, PLR0917
    bands: BandStructure,
    orb_proj: Union[OrbitalProjection, SpinOrbitalProjection],
    kpath: KPathInfo,
    preset: str = "p",
    atom_indices: Optional[List[int]] = None,
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
) -> Tuple[Union[Figure, SubFigure], Axes, PathCollection]:
    """Plot projected band scatter and annotate x-axis with k-path labels.

    Combines a projected-band scatter plot with the symmetry labels from
    one :class:`KPathInfo` carrier.

    :see: :class:`~.test_band_plotting.TestPlotBandScatterWithKpath`

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
    atom_indices : Optional[List[int]], optional
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
    plot_result: Tuple[Union[Figure, SubFigure], Axes, PathCollection] = (
        fig,
        ax,
        scatter,
    )
    return plot_result


__all__: list[str] = [
    "list_band_scatter_presets",
    "plot_band_scatter_preset",
    "plot_band_scatter_with_kpath",
]
