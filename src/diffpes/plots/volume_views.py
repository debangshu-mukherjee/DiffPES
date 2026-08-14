"""Render three-dimensional views of ARPES cubes and band surfaces.

Extended Summary
----------------
The module provides Matplotlib functions that render :class:`ArpesCube`
volumes and band-structure eigenvalue sheets as three-dimensional
scenes. Point clouds, outer cube faces, and band surfaces share one axis
convention: momentum in the horizontal plane and energy on the vertical
axis.

Routine Listings
----------------
:func:`plot_band_surface`
    Plot band eigenvalues as three-dimensional surfaces on a momentum mesh.
:func:`plot_cube_faces`
    Plot the three outer faces of an ARPES cube as opaque surfaces.
:func:`plot_cube_scatter`
    Plot an ARPES cube as a translucent three-dimensional point cloud.

Notes
-----
These functions operate on host-side NumPy arrays and Matplotlib
objects. Use them for analysis visualizations outside JAX-compiled
functions. A caller-supplied axis must be an ``Axes3D`` created with
``fig.add_subplot(projection="3d")``.
"""

import numpy as np
from beartype import beartype
from beartype.typing import List, Optional, Tuple, Union
from jaxtyping import Array, Bool, Float64, jaxtyped
from matplotlib import colormaps
from matplotlib import pyplot as plt
from matplotlib.colors import Colormap
from matplotlib.figure import Figure, SubFigure
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Path3DCollection, Poly3DCollection
from numpy.typing import NDArray

from diffpes.types import ArpesCube


@beartype
def _resolve_volume_axes(
    ax: Optional[Axes3D],
    computed_zorder: bool = True,
) -> Tuple[Union[Figure, SubFigure], Axes3D]:
    """PRIVATE: Resolve the target figure and three-dimensional axis.

    The helper reuses a caller-supplied ``Axes3D`` together with its
    parent figure. It creates a new figure with one three-dimensional
    subplot when the caller supplies no axis.

    Parameters
    ----------
    ax : Optional[Axes3D]
        Existing three-dimensional axis. If ``None``, the helper creates
        a figure and a three-dimensional subplot.
    computed_zorder : bool, optional
        Value of ``computed_zorder`` for a newly created subplot. The
        value has no effect on a caller-supplied axis. Default True.

    Returns
    -------
    fig : Union[Figure, SubFigure]
        Parent figure of the resolved axis.
    resolved_ax : Axes3D
        Three-dimensional axis for drawing.

    Notes
    -----
    A new subplot uses ``fig.add_subplot(projection="3d")``, which
    returns an ``Axes3D`` instance.
    """
    fig: Union[Figure, SubFigure]
    resolved_ax: Axes3D
    if ax is None:
        fig = plt.figure()
        resolved_ax = fig.add_subplot(
            projection="3d", computed_zorder=computed_zorder
        )
    else:
        fig = ax.figure
        resolved_ax = ax
    resolved: Tuple[Union[Figure, SubFigure], Axes3D] = (fig, resolved_ax)
    return resolved


@jaxtyped(typechecker=beartype)
def _normalized_face(
    face_values: Float64[NDArray, "dim_a dim_b"],
) -> Float64[NDArray, "dim_a dim_b"]:
    """PRIVATE: Normalize one cube face to its own maximum.

    Parameters
    ----------
    face_values : Float64[NDArray, "dim_a dim_b"]
        Intensity values on one outer cube face.

    Returns
    -------
    normalized_face : Float64[NDArray, "dim_a dim_b"]
        Face values divided by the face maximum. A zero maximum falls
        back to a divisor of 1.0, so an all-zero face maps to zeros.

    Notes
    -----
    Computes the face maximum on the host. Divides by that maximum when
    it is positive and by 1.0 otherwise.
    """
    face_maximum: float = float(np.max(face_values))
    divisor: float = face_maximum if face_maximum > 0.0 else 1.0
    normalized_face: Float64[NDArray, "dim_a dim_b"] = face_values / divisor
    return normalized_face


@jaxtyped(typechecker=beartype)
def plot_cube_scatter(  # noqa: PLR0913, PLR0917
    cube: ArpesCube,
    ax: Optional[Axes3D] = None,
    intensity_floor: float = 0.12,
    cmap: str = "magma",
    point_size: float = 2.2,
    alpha_power: float = 1.6,
    alpha_scale: float = 0.25,
    jitter: bool = True,
    seed: int = 0,
    floor_slice_index: Optional[int] = None,
    elev: float = 20.0,
    azim: float = -56.0,
    box_aspect: Tuple[float, float, float] = (1.0, 1.0, 1.15),
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    zlabel: str = r"$E - E_F$ (eV)",
    title: str = "",
) -> Tuple[Union[Figure, SubFigure], Axes3D, Path3DCollection]:
    r"""Plot an ARPES cube as a translucent three-dimensional point cloud.

    The function draws one scatter point for each voxel whose normalized
    intensity exceeds ``intensity_floor``. Point color and opacity
    follow the normalized intensity, so bright bands stand out inside
    the translucent volume. The function accepts an existing
    three-dimensional axis. It creates a figure and axis when the caller
    supplies none.

    :see: :class:`~.test_volume_views.TestPlotCubeScatter`

    Implementation Logic
    --------------------
    1. **Normalize the voxel intensities**::

           normalized = intensity / max_intensity

       The maximum comes from the complete cube. A zero maximum raises
       a ``ValueError`` because no voxel can pass the floor.

    2. **Select and position the visible voxels**::

           keep = normalized > intensity_floor
           kx_mesh, ky_mesh, energy_mesh = np.meshgrid(
               kx_values, ky_values, energy_values, indexing="ij"
           )

       With ``jitter`` each coordinate receives a uniform sub-voxel
       offset of at most half a grid step from
       ``np.random.default_rng(seed)``.

    3. **Color and draw the point cloud**::

           colors = np.asarray(colormap(normalized[keep]))
           colors[:, 3] = (
               np.clip(normalized[keep] ** alpha_power, 0.0, 1.0)
               * alpha_scale
           )
           scatter = ax.scatter(..., s=point_size, depthshade=False)

       The alpha channel follows a power law, so faint voxels fade out.

    4. **Style the scene**::

           ax.view_init(elev=elev, azim=azim)
           ax.set_box_aspect(box_aspect)

       An optional ``contourf`` floor map shows one energy slice at the
       bottom of the box.

    Parameters
    ----------
    cube : ArpesCube
        Source cube with intensity on a Cartesian momentum and energy
        grid.
    ax : Optional[Axes3D], optional
        Existing three-dimensional axis. The axis must be an ``Axes3D``.
        If ``None``, the function creates a figure and axis.
    intensity_floor : float, optional
        Normalized intensity threshold for voxel selection. Default
        0.12.
    cmap : str, optional
        Matplotlib colormap name. Default is ``"magma"``.
    point_size : float, optional
        Marker area of each scatter point in points squared. Default
        2.2.
    alpha_power : float, optional
        Exponent of the opacity power law. Default 1.6.
    alpha_scale : float, optional
        Scale of the opacity after the power law. Default 0.25.
    jitter : bool, optional
        If True, add a uniform sub-voxel jitter of at most half a grid
        step per axis. Default True.
    seed : int, optional
        Seed of the jitter random generator. Equal seeds give equal
        offsets. Default 0.
    floor_slice_index : Optional[int], optional
        Energy index of an optional filled contour map on the box
        floor. If ``None``, the function draws no floor map.
    elev : float, optional
        Camera elevation angle in degrees. Default 20.0.
    azim : float, optional
        Camera azimuth angle in degrees. Default -56.0.
    box_aspect : Tuple[float, float, float], optional
        Relative box lengths along the three axes. Default
        (1.0, 1.0, 1.15).
    xlabel : Optional[str], optional
        x-axis label text. If ``None``, the label names :math:`k_x` in
        inverse angstroms.
    ylabel : Optional[str], optional
        y-axis label text. If ``None``, the label names :math:`k_y` in
        inverse angstroms.
    zlabel : str, optional
        z-axis label text. Default is :math:`E - E_F` in eV.
    title : str, optional
        Axis title text. Default is empty.

    Returns
    -------
    fig : Figure
        Matplotlib figure object.
    ax : Axes3D
        Three-dimensional axis used for plotting.
    scatter : Path3DCollection
        Scatter artist that carries the thresholded voxels.

    Raises
    ------
    ValueError
        If the maximum of the cube intensity is zero, so normalization
        is impossible.

    Notes
    -----
    The function reads host values from the cube, so it cannot run
    inside JAX-compiled code.
    """
    intensity: Float64[NDArray, "n_kx n_ky n_e"] = np.asarray(
        cube.intensity, dtype=np.float64
    )
    kx_values: Float64[NDArray, " n_kx"] = np.asarray(
        cube.kx_axis, dtype=np.float64
    )
    ky_values: Float64[NDArray, " n_ky"] = np.asarray(
        cube.ky_axis, dtype=np.float64
    )
    energy_values: Float64[NDArray, " n_e"] = np.asarray(
        cube.energy_axis, dtype=np.float64
    )
    max_intensity: float = float(np.max(intensity))
    if max_intensity == 0.0:
        msg: str = "plot_cube_scatter: the cube intensity maximum is zero."
        raise ValueError(msg)
    normalized: Float64[NDArray, "n_kx n_ky n_e"] = intensity / max_intensity
    keep: Bool[NDArray, "n_kx n_ky n_e"] = normalized > intensity_floor

    kx_mesh: Float64[NDArray, "n_kx n_ky n_e"]
    ky_mesh: Float64[NDArray, "n_kx n_ky n_e"]
    energy_mesh: Float64[NDArray, "n_kx n_ky n_e"]
    kx_mesh, ky_mesh, energy_mesh = np.meshgrid(
        kx_values, ky_values, energy_values, indexing="ij"
    )
    scatter_kx: Float64[NDArray, " n_points"] = kx_mesh[keep]
    scatter_ky: Float64[NDArray, " n_points"] = ky_mesh[keep]
    scatter_energy: Float64[NDArray, " n_points"] = energy_mesh[keep]
    if jitter:
        rng: np.random.Generator = np.random.default_rng(seed)
        half_kx: float = 0.5 * float(kx_values[1] - kx_values[0])
        half_ky: float = 0.5 * float(ky_values[1] - ky_values[0])
        half_energy: float = 0.5 * float(energy_values[1] - energy_values[0])
        n_points: int = scatter_kx.shape[0]
        scatter_kx = scatter_kx + rng.uniform(-half_kx, half_kx, n_points)
        scatter_ky = scatter_ky + rng.uniform(-half_ky, half_ky, n_points)
        scatter_energy = scatter_energy + rng.uniform(
            -half_energy, half_energy, n_points
        )

    colormap: Colormap = colormaps[cmap]
    colors: Float64[NDArray, "n_points 4"] = np.asarray(
        colormap(normalized[keep]), dtype=np.float64
    )
    colors[:, 3] = (
        np.clip(normalized[keep] ** alpha_power, 0.0, 1.0) * alpha_scale
    )

    fig: Union[Figure, SubFigure]
    resolved_ax: Axes3D
    fig, resolved_ax = _resolve_volume_axes(ax)
    scatter: Path3DCollection = resolved_ax.scatter(
        scatter_kx,
        scatter_ky,
        scatter_energy,
        c=colors,
        s=point_size,
        linewidths=0.0,
        depthshade=False,
    )
    if floor_slice_index is not None:
        kx_mesh_2d: Float64[NDArray, "n_kx n_ky"]
        ky_mesh_2d: Float64[NDArray, "n_kx n_ky"]
        kx_mesh_2d, ky_mesh_2d = np.meshgrid(
            kx_values, ky_values, indexing="ij"
        )
        resolved_ax.contourf(
            kx_mesh_2d,
            ky_mesh_2d,
            intensity[:, :, floor_slice_index],
            levels=60,
            zdir="z",
            offset=float(energy_values[0]),
            cmap=cmap,
        )

    resolved_ax.view_init(elev=elev, azim=azim)
    resolved_ax.set_box_aspect(box_aspect)
    resolved_xlabel: str = (
        xlabel if xlabel is not None else r"$k_x$ ($\mathrm{\AA}^{-1}$)"
    )
    resolved_ylabel: str = (
        ylabel if ylabel is not None else r"$k_y$ ($\mathrm{\AA}^{-1}$)"
    )
    resolved_ax.set_xlabel(resolved_xlabel)
    resolved_ax.set_ylabel(resolved_ylabel)
    resolved_ax.set_zlabel(zlabel)
    resolved_ax.set_title(title)
    resolved_ax.grid(False)
    plot_result: Tuple[Union[Figure, SubFigure], Axes3D, Path3DCollection] = (
        fig,
        resolved_ax,
        scatter,
    )
    return plot_result


@jaxtyped(typechecker=beartype)
def plot_cube_faces(  # noqa: PLR0913, PLR0917
    cube: ArpesCube,
    ax: Optional[Axes3D] = None,
    cmap: str = "inferno",
    elev: float = 20.0,
    azim: float = -55.0,
    box_aspect: Tuple[float, float, float] = (1.0, 1.0, 0.85),
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    zlabel: str = r"$E - E_F$ (eV)",
    title: str = "",
) -> Tuple[Union[Figure, SubFigure], Axes3D, List[Poly3DCollection]]:
    r"""Plot the three outer faces of an ARPES cube as opaque surfaces.

    The function paints the last-``k_x`` face, the last-``k_y`` face,
    and the top-energy face of the cube on one three-dimensional axis.
    Each face carries colors from its own normalized intensity, so every
    face uses the full colormap range. The function accepts an existing
    three-dimensional axis. It creates a figure and axis when the caller
    supplies none.

    :see: :class:`~.test_volume_views.TestPlotCubeFaces`

    Implementation Logic
    --------------------
    1. **Normalize each face to its own maximum**::

           face_colors = colormap(_normalized_face(face_values))

       A zero face maximum falls back to a divisor of 1.0, so an
       all-zero face maps to the lowest colormap entry.

    2. **Draw the three outer faces**::

           ax.plot_surface(
               ..., facecolors=face_colors,
               rstride=1, cstride=1, shade=False,
           )

       Constant-coordinate meshes place each face on the cube boundary.

    3. **Style the scene**::

           ax.view_init(elev=elev, azim=azim)
           ax.set_box_aspect(box_aspect)

       A new subplot uses ``computed_zorder=False``, so the drawing
       order keeps the outer faces visible.

    Parameters
    ----------
    cube : ArpesCube
        Source cube with intensity on a Cartesian momentum and energy
        grid.
    ax : Optional[Axes3D], optional
        Existing three-dimensional axis. The axis must be an ``Axes3D``.
        If ``None``, the function creates a figure and axis with
        ``computed_zorder=False``.
    cmap : str, optional
        Matplotlib colormap name. Default is ``"inferno"``.
    elev : float, optional
        Camera elevation angle in degrees. Default 20.0.
    azim : float, optional
        Camera azimuth angle in degrees. Default -55.0.
    box_aspect : Tuple[float, float, float], optional
        Relative box lengths along the three axes. Default
        (1.0, 1.0, 0.85).
    xlabel : Optional[str], optional
        x-axis label text. If ``None``, the label names :math:`k_x` in
        inverse angstroms.
    ylabel : Optional[str], optional
        y-axis label text. If ``None``, the label names :math:`k_y` in
        inverse angstroms.
    zlabel : str, optional
        z-axis label text. Default is :math:`E - E_F` in eV.
    title : str, optional
        Axis title text. Default is empty.

    Returns
    -------
    fig : Figure
        Matplotlib figure object.
    ax : Axes3D
        Three-dimensional axis used for plotting.
    surfaces : List[Poly3DCollection]
        Face surface artists, ordered as the last-``k_x`` face, the
        last-``k_y`` face, and the top-energy face.

    Notes
    -----
    The function reads host values from the cube, so it cannot run
    inside JAX-compiled code.
    """
    intensity: Float64[NDArray, "n_kx n_ky n_e"] = np.asarray(
        cube.intensity, dtype=np.float64
    )
    kx_values: Float64[NDArray, " n_kx"] = np.asarray(
        cube.kx_axis, dtype=np.float64
    )
    ky_values: Float64[NDArray, " n_ky"] = np.asarray(
        cube.ky_axis, dtype=np.float64
    )
    energy_values: Float64[NDArray, " n_e"] = np.asarray(
        cube.energy_axis, dtype=np.float64
    )
    colormap: Colormap = colormaps[cmap]

    fig: Union[Figure, SubFigure]
    resolved_ax: Axes3D
    fig, resolved_ax = _resolve_volume_axes(ax, computed_zorder=False)
    surfaces: List[Poly3DCollection] = []

    ky_mesh_kx_face: Float64[NDArray, "n_ky n_e"]
    energy_mesh_kx_face: Float64[NDArray, "n_ky n_e"]
    ky_mesh_kx_face, energy_mesh_kx_face = np.meshgrid(
        ky_values, energy_values, indexing="ij"
    )
    kx_face: Float64[NDArray, "n_ky n_e"] = _normalized_face(
        intensity[-1, :, :]
    )
    surfaces.append(
        resolved_ax.plot_surface(
            np.full_like(ky_mesh_kx_face, float(kx_values[-1])),
            ky_mesh_kx_face,
            energy_mesh_kx_face,
            facecolors=colormap(kx_face),
            rstride=1,
            cstride=1,
            shade=False,
        )
    )

    kx_mesh_ky_face: Float64[NDArray, "n_kx n_e"]
    energy_mesh_ky_face: Float64[NDArray, "n_kx n_e"]
    kx_mesh_ky_face, energy_mesh_ky_face = np.meshgrid(
        kx_values, energy_values, indexing="ij"
    )
    ky_face: Float64[NDArray, "n_kx n_e"] = _normalized_face(
        intensity[:, -1, :]
    )
    surfaces.append(
        resolved_ax.plot_surface(
            kx_mesh_ky_face,
            np.full_like(kx_mesh_ky_face, float(ky_values[-1])),
            energy_mesh_ky_face,
            facecolors=colormap(ky_face),
            rstride=1,
            cstride=1,
            shade=False,
        )
    )

    kx_mesh_top_face: Float64[NDArray, "n_kx n_ky"]
    ky_mesh_top_face: Float64[NDArray, "n_kx n_ky"]
    kx_mesh_top_face, ky_mesh_top_face = np.meshgrid(
        kx_values, ky_values, indexing="ij"
    )
    top_face: Float64[NDArray, "n_kx n_ky"] = _normalized_face(
        intensity[:, :, -1]
    )
    surfaces.append(
        resolved_ax.plot_surface(
            kx_mesh_top_face,
            ky_mesh_top_face,
            np.full_like(kx_mesh_top_face, float(energy_values[-1])),
            facecolors=colormap(top_face),
            rstride=1,
            cstride=1,
            shade=False,
        )
    )

    resolved_ax.view_init(elev=elev, azim=azim)
    resolved_ax.set_box_aspect(box_aspect)
    resolved_xlabel: str = (
        xlabel if xlabel is not None else r"$k_x$ ($\mathrm{\AA}^{-1}$)"
    )
    resolved_ylabel: str = (
        ylabel if ylabel is not None else r"$k_y$ ($\mathrm{\AA}^{-1}$)"
    )
    resolved_ax.set_xlabel(resolved_xlabel)
    resolved_ax.set_ylabel(resolved_ylabel)
    resolved_ax.set_zlabel(zlabel)
    resolved_ax.set_title(title)
    plot_result: Tuple[
        Union[Figure, SubFigure], Axes3D, List[Poly3DCollection]
    ] = (fig, resolved_ax, surfaces)
    return plot_result


@jaxtyped(typechecker=beartype)
def plot_band_surface(  # noqa: PLR0913, PLR0917
    eigenvalues: Float64[Array, "nk nbands"] | Float64[NDArray, "nk nbands"],
    kx_axis: Float64[Array, " nkx"] | Float64[NDArray, " nkx"],
    ky_axis: Float64[Array, " nky"] | Float64[NDArray, " nky"],
    band_indices: Optional[Tuple[int, ...]] = None,
    ax: Optional[Axes3D] = None,
    cmaps: Tuple[str, ...] = ("viridis", "plasma"),
    rcount: int = 180,
    ccount: int = 180,
    alpha: float = 0.85,
    elev: float = 14.0,
    azim: float = -63.0,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    zlabel: str = r"$E - E_F$ (eV)",
    title: str = "",
) -> Tuple[Union[Figure, SubFigure], Axes3D, List[Poly3DCollection]]:
    r"""Plot band eigenvalues as three-dimensional surfaces on a momentum mesh.

    The function reshapes each selected band from a flattened Cartesian
    mesh to the ``(len(kx_axis), len(ky_axis))`` grid and draws it as
    one colored surface. The first selected band is opaque. Later bands
    use ``alpha``, so crossing sheets stay visible. The function accepts
    an existing three-dimensional axis. It creates a figure and axis
    when the caller supplies none.

    :see: :class:`~.test_volume_views.TestPlotBandSurface`

    Implementation Logic
    --------------------
    1. **Validate the flattened mesh size**::

           if eigenvalue_grid.shape[0] != n_kx * n_ky:
               raise ValueError(msg)

       The eigenvalue rows must cover the Cartesian mesh exactly once.

    2. **Reshape each selected band to the mesh**::

           band_values = eigenvalue_grid[:, band_index].reshape(
               n_kx, n_ky
           )

       The reshape inverts the row-major ``indexing="ij"`` flattening,
       so the values match
       ``np.meshgrid(kx_values, ky_values, indexing="ij")``.

    3. **Draw one surface per band**::

           ax.plot_surface(
               kx_mesh, ky_mesh, band_values,
               cmap=cmaps[position % len(cmaps)],
               rcount=rcount, ccount=ccount,
               linewidth=0.0, antialiased=True,
           )

       The colormap tuple cycles, so adjacent bands stay
       distinguishable.

    Parameters
    ----------
    eigenvalues : Float64[Array, "nk nbands"] | Float64[NDArray, "nk nbands"]
        Band eigenvalues in eV on a flattened Cartesian mesh. The
        flattening uses ``indexing="ij"``, so ``k_x`` varies slowest.
    kx_axis : Float64[Array, " nkx"] | Float64[NDArray, " nkx"]
        Cartesian ``k_x`` axis in inverse angstroms.
    ky_axis : Float64[Array, " nky"] | Float64[NDArray, " nky"]
        Cartesian ``k_y`` axis in inverse angstroms.
    band_indices : Optional[Tuple[int, ...]], optional
        Band columns to draw. If ``None``, the function draws every
        band.
    ax : Optional[Axes3D], optional
        Existing three-dimensional axis. The axis must be an ``Axes3D``.
        If ``None``, the function creates a figure and axis.
    cmaps : Tuple[str, ...], optional
        Colormap names that cycle over the selected bands. Default is
        ``("viridis", "plasma")``.
    rcount : int, optional
        Maximum surface samples along the ``k_x`` direction. Default
        180.
    ccount : int, optional
        Maximum surface samples along the ``k_y`` direction. Default
        180.
    alpha : float, optional
        Opacity of every band after the first. Default 0.85.
    elev : float, optional
        Camera elevation angle in degrees. Default 14.0.
    azim : float, optional
        Camera azimuth angle in degrees. Default -63.0.
    xlabel : Optional[str], optional
        x-axis label text. If ``None``, the label names :math:`k_x` in
        inverse angstroms.
    ylabel : Optional[str], optional
        y-axis label text. If ``None``, the label names :math:`k_y` in
        inverse angstroms.
    zlabel : str, optional
        z-axis label text. Default is :math:`E - E_F` in eV.
    title : str, optional
        Axis title text. Default is empty.

    Returns
    -------
    fig : Figure
        Matplotlib figure object.
    ax : Axes3D
        Three-dimensional axis used for plotting.
    surfaces : List[Poly3DCollection]
        Surface artists in the order of the selected bands.

    Raises
    ------
    ValueError
        If the number of eigenvalue rows differs from
        ``len(kx_axis) * len(ky_axis)``.

    Notes
    -----
    The function operates on host values, so it cannot run inside
    JAX-compiled code.
    """
    eigenvalue_grid: Float64[NDArray, "nk nbands"] = np.asarray(
        eigenvalues, dtype=np.float64
    )
    kx_values: Float64[NDArray, " nkx"] = np.asarray(kx_axis, dtype=np.float64)
    ky_values: Float64[NDArray, " nky"] = np.asarray(ky_axis, dtype=np.float64)
    n_kx: int = kx_values.shape[0]
    n_ky: int = ky_values.shape[0]
    if eigenvalue_grid.shape[0] != n_kx * n_ky:
        msg: str = (
            "plot_band_surface: eigenvalues rows must equal "
            "len(kx_axis) * len(ky_axis)."
        )
        raise ValueError(msg)
    selected_bands: Tuple[int, ...] = (
        band_indices
        if band_indices is not None
        else tuple(range(eigenvalue_grid.shape[1]))
    )

    fig: Union[Figure, SubFigure]
    resolved_ax: Axes3D
    fig, resolved_ax = _resolve_volume_axes(ax)
    kx_mesh: Float64[NDArray, "nkx nky"]
    ky_mesh: Float64[NDArray, "nkx nky"]
    kx_mesh, ky_mesh = np.meshgrid(kx_values, ky_values, indexing="ij")
    surfaces: List[Poly3DCollection] = []
    position: int
    band_index: int
    for position, band_index in enumerate(selected_bands):
        band_values: Float64[NDArray, "nkx nky"] = eigenvalue_grid[
            :, band_index
        ].reshape(n_kx, n_ky)
        band_alpha: float = 1.0 if position == 0 else alpha
        surfaces.append(
            resolved_ax.plot_surface(
                kx_mesh,
                ky_mesh,
                band_values,
                cmap=cmaps[position % len(cmaps)],
                rcount=rcount,
                ccount=ccount,
                linewidth=0.0,
                antialiased=True,
                alpha=band_alpha,
            )
        )

    resolved_ax.view_init(elev=elev, azim=azim)
    resolved_xlabel: str = (
        xlabel if xlabel is not None else r"$k_x$ ($\mathrm{\AA}^{-1}$)"
    )
    resolved_ylabel: str = (
        ylabel if ylabel is not None else r"$k_y$ ($\mathrm{\AA}^{-1}$)"
    )
    resolved_ax.set_xlabel(resolved_xlabel)
    resolved_ax.set_ylabel(resolved_ylabel)
    resolved_ax.set_zlabel(zlabel)
    resolved_ax.set_title(title)
    plot_result: Tuple[
        Union[Figure, SubFigure], Axes3D, List[Poly3DCollection]
    ] = (fig, resolved_ax, surfaces)
    return plot_result


__all__: list[str] = [
    "plot_band_surface",
    "plot_cube_faces",
    "plot_cube_scatter",
]
