"""Export the README hero figures from the public diffpes API.

Extended Summary
----------------
The script builds honeycomb tight-binding models and renders the README
gallery: band structures beside simulated ARPES cuts, full-zone
constant-energy maps, three-dimensional band and intensity-cube views,
densities of states, temperature and linewidth series, and the calibrated
detector chain through one Poisson acquisition. Every physical quantity
comes from public diffpes calls, and every panel renders through the
public :mod:`diffpes.plots` functions.

Routine Listings
----------------
:func:`main`
    Render and save all README figures.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, cast

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "True")

import jax
import jax.numpy as jnp
import matplotlib
import numpy as np
from jax import Array

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import diffpes as dp  # noqa: E402

OUTPUT_DIR = Path(__file__).resolve().parent / "source" / "_static" / "readme"
BI2SE3_DIR = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "DFT"
    / "Bi2Se3"
    / "6QL"
    / "Output few bands"
)
BI2SE3_BULK_DIR = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "DFT"
    / "Bi2Se3"
    / "Bulk"
    / "Output"
)

LATTICE_ANG = 2.46
HOPPING_EV = 2.7
TEMPERATURE_K = 300.0
PATH_GAMMA_EV = 0.09
MAP_GAMMA_EV = 0.12
CONE_GAMMA_EV = 0.045
ARC_DEDUPE_TOL = 1.0e-12
CUBE_INTENSITY_FLOOR = 0.12
DOS_WINDOW_EV = 2.5
K_POINT_FRAC = (1.0 / 3.0, 2.0 / 3.0, 0.0)

BAND_COLOR_PI = "#0066cc"
BAND_COLOR_PI_STAR = "#cc5500"
SERIES_COLORS = ("#0066cc", "#cc5500", "#2e9e62")

FIGURES: list[str] = []


def save(fig: plt.Figure, name: str, tight: bool = True) -> None:
    """Save one gallery figure and record its file name."""
    fig.savefig(
        OUTPUT_DIR / name,
        dpi=200,
        bbox_inches="tight" if tight else None,
    )
    plt.close(fig)
    FIGURES.append(name)


def build_honeycomb_model(
    onsite_a_ev: float = 0.0, onsite_b_ev: float = 0.0
) -> tuple[dp.types.CrystalGeometry, dp.types.TBModel]:
    """Build a nearest-neighbour honeycomb pi-band model."""
    lattice = jnp.asarray(
        [
            [LATTICE_ANG, 0.0, 0.0],
            [LATTICE_ANG / 2.0, LATTICE_ANG * jnp.sqrt(3.0) / 2.0, 0.0],
            [0.0, 0.0, 20.0],
        ]
    )
    crystal = dp.types.make_crystal_geometry(
        lattice=lattice,
        positions=jnp.asarray([[0.0, 0.0, 0.0], [1.0 / 3.0, 1.0 / 3.0, 0.0]]),
        species=("C", "C"),
    )
    basis = dp.types.make_orbital_basis(
        atom_indices=(0, 1),
        n=(2, 2),
        l=(0, 0),
        m=(0, 0),
        labels=("pz_A", "pz_B"),
    )
    model = dp.types.make_tb_model(
        hopping_amplitudes=-HOPPING_EV * jnp.ones(6, dtype=jnp.complex128),
        onsite_energies=jnp.asarray([onsite_a_ev, onsite_b_ev]),
        soc_lambdas=jnp.zeros(0),
        geometry=crystal,
        basis=basis,
        hopping_pairs=((0, 1), (0, 1), (0, 1), (1, 0), (1, 0), (1, 0)),
        hopping_cells=(
            (0, 0, 0),
            (-1, 0, 0),
            (0, -1, 0),
            (0, 0, 0),
            (1, 0, 0),
            (0, 1, 0),
        ),
        shell_index=(-1, -1),
        depths=jnp.zeros(2),
    )
    return crystal, model


def spectral_intensity(
    model: dp.types.TBModel,
    kpoints_frac: Array,
    energy_axis: Array,
    gamma_ev: float,
    fermi_ev: float = 0.0,
    temperature_k: float = TEMPERATURE_K,
) -> tuple[dp.types.DiagonalizedBands, Array]:
    """Occupied spectral intensity with unit band weights on a k set."""
    bands = dp.tightb.diagonalize_tb(model, kpoints_frac)
    weights = jnp.ones(
        (
            kpoints_frac.shape[0],
            energy_axis.shape[0],
            bands.eigenvalues.shape[1],
        ),
        dtype=jnp.float64,
    )
    intensity = dp.simul.assemble_spectral_intensity_bands_chunk(
        bands.eigenvalues,
        weights,
        energy_axis,
        dp.types.make_self_energy_model(gamma=gamma_ev),
        jnp.asarray(fermi_ev),
        temperature_k,
        allow_degenerate_value_only=True,
    )
    return bands, intensity


def dedupe_path(
    path: dp.types.KPath, crystal: dp.types.CrystalGeometry
) -> tuple[Array, np.ndarray, list[int]]:
    """Drop repeated segment anchors from a labeled path."""
    full_arc = np.asarray(dp.tightb.kpath_arc_length(path, crystal))
    keep = np.concatenate(([True], np.diff(full_arc) > ARC_DEDUPE_TOL))
    kpoints = jnp.asarray(np.asarray(path.kpoints)[keep])
    arc = full_arc[keep]
    label_indices = [
        int(keep[: index + 1].sum() - 1)
        for index in np.asarray(path.label_indices)
    ]
    return kpoints, arc, label_indices


def brillouin_zone_vertices(crystal: dp.types.CrystalGeometry) -> np.ndarray:
    """Cartesian vertices of the first-zone hexagon, closed for plotting."""
    corner = np.asarray(
        dp.tightb.kpoints_frac_to_cart(jnp.asarray([K_POINT_FRAC]), crystal)
    )[0, :2]
    angles = np.arange(7) * np.pi / 3.0
    rotations = np.stack(
        [
            np.stack([np.cos(angles), -np.sin(angles)], axis=-1),
            np.stack([np.sin(angles), np.cos(angles)], axis=-1),
        ],
        axis=-2,
    )
    return rotations @ corner


def cut_through_k(
    model: dp.types.TBModel,
    crystal: dp.types.CrystalGeometry,
    energy_axis: Array,
    gamma_ev: float,
    n_points: int = 301,
    temperature_k: float = TEMPERATURE_K,
) -> tuple[np.ndarray, np.ndarray]:
    """Spectral cut on a straight momentum line through the K point."""
    path = dp.tightb.build_kpath(
        jnp.asarray([[0.16, 0.32, 0.0], [0.5067, 1.0133, 0.0]]),
        crystal,
        n_points,
        ("start", "end"),
    )
    arc = np.asarray(dp.tightb.kpath_arc_length(path, crystal))
    k_corner = float(
        np.linalg.norm(
            np.asarray(
                dp.tightb.kpoints_frac_to_cart(
                    jnp.asarray([K_POINT_FRAC]), crystal
                )
            )[0]
            - np.asarray(
                dp.tightb.kpoints_frac_to_cart(path.kpoints, crystal)
            )[0]
        )
    )
    _, intensity = spectral_intensity(
        model,
        path.kpoints,
        energy_axis,
        gamma_ev,
        temperature_k=temperature_k,
    )
    return arc - k_corner, np.asarray(intensity)


def figure_bands_to_arpes(
    crystal: dp.types.CrystalGeometry, model: dp.types.TBModel
) -> None:
    """Band structure beside the simulated ARPES cut on the same path."""
    anchors = jnp.asarray(
        [
            [0.0, 0.0, 0.0],
            [1.0 / 3.0, 2.0 / 3.0, 0.0],
            [0.5, 0.5, 0.0],
            [0.0, 0.0, 0.0],
        ]
    )
    path = dp.tightb.build_kpath(
        anchors, crystal, 241, ("Gamma", "K", "M", "Gamma")
    )
    kpoints, arc, label_indices = dedupe_path(path, crystal)
    energy_axis = jnp.linspace(-9.2, 1.4, 530)
    bands, intensity = spectral_intensity(
        model, kpoints, energy_axis, PATH_GAMMA_EV
    )
    eigenvalues = np.asarray(bands.eigenvalues)

    spectrum = dp.types.make_arpes_spectrum(
        intensity,
        energy_axis,
        jnp.asarray(arc),
        dp.tightb.kpoints_frac_to_cart(kpoints, crystal),
    )
    kpath_info = dp.types.make_kpath_info(
        num_kpoints=arc.shape[0],
        label_indices=label_indices,
        points_per_segment=241,
        segments=3,
        kpoints=kpoints,
        labels=(r"$\Gamma$", "K", "M", r"$\Gamma$"),
    )

    fig, (ax_bands, ax_arpes) = plt.subplots(
        1, 2, figsize=(12.6, 4.9), constrained_layout=True
    )
    dp.plots.plot_band_dispersion(
        eigenvalues[:, :1],
        arc,
        ax=ax_bands,
        color=BAND_COLOR_PI,
        linewidth=2.2,
        fermi_line=False,
    )
    dp.plots.plot_band_dispersion(
        eigenvalues[:, 1:],
        arc,
        ax=ax_bands,
        color=BAND_COLOR_PI_STAR,
        linewidth=2.2,
        fermi_line=True,
        xlabel="",
        title="tight-binding band structure",
    )
    ax_bands.text(
        arc[-1] * 0.985, 0.25, r"$E_F$", color="0.35", ha="right", fontsize=11
    )
    ax_bands.text(
        arc[label_indices[1]], -7.0, r"$\pi$", color=BAND_COLOR_PI, fontsize=13
    )
    ax_bands.text(
        arc[label_indices[1]],
        5.9,
        r"$\pi^{*}$",
        color=BAND_COLOR_PI_STAR,
        fontsize=13,
    )
    for index in label_indices:
        ax_bands.axvline(arc[index], color="0.85", lw=0.8, zorder=0)
    ax_bands.set_xticks(
        [arc[i] for i in label_indices], (r"$\Gamma$", "K", "M", r"$\Gamma$")
    )
    ax_bands.set_xlim(arc[0], arc[-1])
    ax_bands.spines[["top", "right"]].set_visible(False)

    dp.plots.plot_arpes_with_kpath(
        spectrum,
        kpath_info,
        ax=ax_arpes,
        cmap="magma",
        xlabel="",
        ylabel=r"$E - E_F$ (eV)",
        title="simulated ARPES, 300 K",
    )
    save(fig, "graphene-bands-to-arpes.png", tight=False)


def figure_constant_energy_maps(
    crystal: dp.types.CrystalGeometry, model: dp.types.TBModel
) -> None:
    """Full-zone constant-energy slices at four binding energies."""
    axis = jnp.linspace(-2.35, 2.35, 361)
    kgrid = dp.tightb.build_arpes_kmesh(axis, axis, 0.0, 0.0, crystal)
    slice_energies = jnp.asarray([-0.5, -1.6, -2.7, -5.5])
    _, intensity = spectral_intensity(
        model, kgrid.kpoints, slice_energies, MAP_GAMMA_EV
    )
    maps = (
        np.asarray(intensity)
        .reshape((*kgrid.mesh_shape, slice_energies.shape[0]))
        .transpose((1, 0, 2))
    )
    hexagon = brillouin_zone_vertices(crystal)

    fig, axes = plt.subplots(
        1, 4, figsize=(15.2, 4.4), constrained_layout=True
    )
    for index, (ax, energy) in enumerate(
        zip(axes, np.asarray(slice_energies), strict=True)
    ):
        dp.plots.plot_momentum_map(
            jnp.asarray(maps[:, :, index]),
            axis,
            axis,
            ax=ax,
            cmap="inferno",
            colorbar=False,
            xlabel=r"$k_x$ ($\mathrm{\AA}^{-1}$)",
            ylabel=r"$k_y$ ($\mathrm{\AA}^{-1}$)" if index == 0 else "",
            title=rf"$E - E_F = {energy:+.1f}$ eV",
        )
        ax.plot(hexagon[:, 0], hexagon[:, 1], color="w", lw=0.9, ls="--")
        if index > 0:
            ax.set_yticklabels([])
        ax.set_aspect("equal")
    save(fig, "graphene-constant-energy-maps.png")


def build_cone_cube(
    crystal: dp.types.CrystalGeometry, model: dp.types.TBModel
) -> dp.types.ArpesCube:
    """Assemble the intensity cube on a raster around the K point."""
    dirac_cart = np.asarray(
        dp.tightb.kpoints_frac_to_cart(jnp.asarray([K_POINT_FRAC]), crystal)
    )[0]
    half_width = 0.42
    mesh_axis = jnp.linspace(-half_width, half_width, 121)
    kgrid = dp.tightb.build_arpes_kmesh(
        dirac_cart[0] + mesh_axis, dirac_cart[1] + mesh_axis, 0.0, 0.0, crystal
    )
    energy_axis = jnp.linspace(-2.1, 0.4, 141)
    _, intensity = spectral_intensity(
        model, kgrid.kpoints, energy_axis, CONE_GAMMA_EV
    )
    cube_values = (
        np.asarray(intensity)
        .reshape((*kgrid.mesh_shape, energy_axis.shape[0]))
        .transpose((1, 0, 2))
    )
    return dp.types.make_arpes_cube(
        jnp.asarray(cube_values),
        mesh_axis,
        mesh_axis,
        energy_axis,
        provenance="graphene Dirac cone README figure",
    )


def figure_cone_cube(cube: dp.types.ArpesCube) -> None:
    """Translucent three-dimensional ARPES cube around one Dirac cone."""
    fig = plt.figure(figsize=(8.6, 7.2))
    ax = fig.add_subplot(projection="3d")
    dp.plots.plot_cube_scatter(
        cube,
        ax=ax,
        intensity_floor=CUBE_INTENSITY_FLOOR,
        seed=20260814,
        floor_slice_index=8,
        xlabel=r"$k_x - K$ ($\mathrm{\AA}^{-1}$)",
        ylabel=r"$k_y - K$ ($\mathrm{\AA}^{-1}$)",
        title="ARPES intensity cube at the Dirac point",
    )
    for pane in (ax.xaxis, ax.yaxis, ax.zaxis):
        pane_axis = cast("Any", pane)
        pane_axis.set_pane_color((0.06, 0.05, 0.10, 1.0))
        pane_axis.line.set_color("0.4")
    half_width = float(np.asarray(cube.kx_axis)[-1])
    floor_energy = float(np.asarray(cube.energy_axis)[0])
    ax.set_xlim(-half_width, half_width)
    ax.set_ylim(-half_width, half_width)
    ax.set_zlim(floor_energy, 0.2)
    save(fig, "graphene-dirac-cone-cube.png")


def figure_energy_window_maps(cube: dp.types.ArpesCube) -> None:
    """Energy-window integrals of the Dirac-cone cube."""
    windows = ((-1.8, -1.2), (-0.9, -0.4), (-0.15, 0.1))
    fig, axes = plt.subplots(
        1, 3, figsize=(11.6, 3.9), constrained_layout=True
    )
    for index, (ax, bounds) in enumerate(zip(axes, windows, strict=True)):
        dp.plots.plot_momentum_map(
            dp.simul.energy_window_map(cube, bounds[0], bounds[1]),
            cube.kx_axis,
            cube.ky_axis,
            ax=ax,
            cmap="inferno",
            colorbar=False,
            xlabel=r"$k_x - K$ ($\mathrm{\AA}^{-1}$)",
            ylabel=r"$k_y - K$ ($\mathrm{\AA}^{-1}$)" if index == 0 else "",
            title=f"{bounds[0]:+.2f} to {bounds[1]:+.2f} eV",
        )
        if index > 0:
            ax.set_yticklabels([])
        ax.set_aspect("equal")
    save(fig, "graphene-energy-window-maps.png")


def figure_band_surface(
    crystal: dp.types.CrystalGeometry, model: dp.types.TBModel
) -> None:
    """Three-dimensional pi and pi-star band surfaces over the zone."""
    axis = jnp.linspace(-2.1, 2.1, 221)
    kgrid = dp.tightb.build_arpes_kmesh(axis, axis, 0.0, 0.0, crystal)
    bands = dp.tightb.diagonalize_tb(model, kgrid.kpoints)
    shape = kgrid.mesh_shape
    n_bands = bands.eigenvalues.shape[1]
    eigen_mesh = (
        np.asarray(bands.eigenvalues)
        .reshape((*shape, n_bands))
        .transpose((1, 0, 2))
        .reshape((shape[0] * shape[1], n_bands))
    )

    fig = plt.figure(figsize=(8.6, 7.0))
    ax = fig.add_subplot(projection="3d")
    dp.plots.plot_band_surface(
        eigen_mesh,
        np.asarray(axis),
        np.asarray(axis),
        ax=ax,
        title=r"$\pi$ and $\pi^{*}$ band surfaces with six Dirac points",
    )
    save(fig, "graphene-band-surface-3d.png")


def figure_dos(
    crystal: dp.types.CrystalGeometry, model: dp.types.TBModel
) -> None:
    """Gaussian-broadened density of states of the pi bands."""
    axis = jnp.linspace(-2.05, 2.05, 181)
    kgrid = dp.tightb.build_arpes_kmesh(axis, axis, 0.0, 0.0, crystal)
    bands = dp.tightb.diagonalize_tb(model, kgrid.kpoints)
    n_k = kgrid.kpoints.shape[0]
    energy_axis = jnp.linspace(-9.5, 9.5, 601)
    dos = dp.tightb.dos_gaussian(
        bands.eigenvalues,
        jnp.full((n_k,), 1.0 / n_k),
        energy_axis,
        0.07,
    )
    energies = np.asarray(dos.energy)
    values = np.asarray(dos.total_dos)

    fig, ax = plt.subplots(figsize=(7.6, 4.2), constrained_layout=True)
    dp.plots.plot_dos(
        dos,
        ax=ax,
        color=BAND_COLOR_PI,
        title="pi-band density of states, occupied part shaded",
    )
    ax.annotate(
        "van Hove",
        xy=(-HOPPING_EV, float(values[np.argmin(np.abs(energies + 2.7))])),
        xytext=(-5.6, 0.85 * values.max()),
        arrowprops={"arrowstyle": "-", "color": "0.5", "lw": 0.9},
        color="0.35",
        fontsize=10,
    )
    ax.spines[["top", "right"]].set_visible(False)
    save(fig, "graphene-dos.png")


def figure_edc_mdc(
    crystal: dp.types.CrystalGeometry, model: dp.types.TBModel
) -> None:
    """Energy and momentum distribution curves through the Dirac cone."""
    energy_axis = jnp.linspace(-2.0, 0.3, 231)
    momentum, intensity = cut_through_k(
        model, crystal, energy_axis, CONE_GAMMA_EV
    )

    fig, (ax_edc, ax_mdc) = plt.subplots(
        1, 2, figsize=(12.2, 4.8), constrained_layout=True
    )
    dp.plots.plot_distribution_curves(
        intensity,
        momentum,
        np.asarray(energy_axis),
        kind="edc",
        positions=tuple(float(k) for k in np.linspace(-0.3, 0.3, 11)),
        ax=ax_edc,
        offset=4.5,
        cmap="viridis",
        legend=False,
        ylabel="intensity, offset per EDC",
        title=r"EDCs from $k - K = -0.3$ to $+0.3$ $\mathrm{\AA}^{-1}$",
    )
    ax_edc.spines[["top", "right"]].set_visible(False)

    dp.plots.plot_distribution_curves(
        intensity,
        momentum,
        np.asarray(energy_axis),
        kind="mdc",
        positions=tuple(float(e) for e in np.linspace(-1.5, -0.1, 8)),
        ax=ax_mdc,
        offset=6.0,
        cmap="plasma",
        legend=False,
        xlabel=r"$k - K$ ($\mathrm{\AA}^{-1}$)",
        ylabel="intensity, offset per MDC",
        title=r"MDCs from $-1.5$ to $-0.1$ eV",
    )
    ax_mdc.spines[["top", "right"]].set_visible(False)
    save(fig, "graphene-edc-mdc.png")


def figure_fermi_edge(
    crystal: dp.types.CrystalGeometry, model: dp.types.TBModel
) -> None:
    """Momentum-summed Fermi edge at three temperatures."""
    energy_axis = jnp.linspace(-0.35, 0.35, 281)
    temperatures = (25.0, 100.0, 300.0)
    edges: list[np.ndarray] = []
    for temperature in temperatures:
        _, intensity = cut_through_k(
            model,
            crystal,
            energy_axis,
            0.03,
            temperature_k=temperature,
        )
        edges.append(np.asarray(intensity).sum(axis=0))

    fig, ax = plt.subplots(figsize=(7.2, 4.2), constrained_layout=True)
    dp.plots.plot_curve_family(
        np.asarray(energy_axis),
        tuple(edges),
        labels=tuple(f"{t:.0f} K" for t in temperatures),
        ax=ax,
        colors=SERIES_COLORS,
        legend=False,
        xlabel=r"$E - E_F$ (eV)",
        ylabel="momentum-summed intensity",
        title="Fermi edge at three temperatures",
    )
    ax.axvline(0.0, color="0.45", lw=0.9, ls="--")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    save(fig, "graphene-fermi-edge.png")


def figure_linewidth_series(
    crystal: dp.types.CrystalGeometry, model: dp.types.TBModel
) -> None:
    """Render the same Dirac-cone cut at three linewidths."""
    energy_axis = jnp.linspace(-2.0, 0.3, 231)
    gammas = (0.02, 0.08, 0.25)
    momentum = np.zeros(0)
    intensities: list[np.ndarray] = []
    for gamma in gammas:
        momentum, intensity = cut_through_k(model, crystal, energy_axis, gamma)
        intensities.append(intensity)

    fig, axes = plt.subplots(
        1, 3, figsize=(12.6, 4.2), constrained_layout=True
    )
    dp.plots.plot_spectral_cut_series(
        tuple(intensities),
        momentum,
        np.asarray(energy_axis),
        titles=tuple(rf"$\Gamma = {1000.0 * g:.0f}$ meV" for g in gammas),
        axes=tuple(axes),
        share_scale=False,
        colorbar_label="intensity",
        fermi_line=False,
        xlabel=r"$k - K$ ($\mathrm{\AA}^{-1}$)",
    )
    for index, ax in enumerate(axes):
        ax.set_xlim(-0.45, 0.45)
        if index > 0:
            ax.set_yticklabels([])
    save(fig, "graphene-linewidth-series.png")


def figure_gapped_honeycomb(crystal: dp.types.CrystalGeometry) -> None:
    """Bands and ARPES cut of a gapped honeycomb lattice."""
    _, gapped_model = build_honeycomb_model(1.0, -1.0)
    energy_axis = jnp.linspace(-3.2, 0.6, 301)
    momentum, intensity = cut_through_k(
        gapped_model, crystal, energy_axis, 0.05
    )
    path = dp.tightb.build_kpath(
        jnp.asarray([[0.16, 0.32, 0.0], [0.5067, 1.0133, 0.0]]),
        crystal,
        301,
        ("start", "end"),
    )
    bands = dp.tightb.diagonalize_tb(gapped_model, path.kpoints)
    eigenvalues = np.asarray(bands.eigenvalues)

    fig, (ax_bands, ax_cut) = plt.subplots(
        1, 2, figsize=(12.2, 4.6), constrained_layout=True
    )
    dp.plots.plot_band_dispersion(
        eigenvalues[:, :1],
        momentum,
        ax=ax_bands,
        color=BAND_COLOR_PI,
        linewidth=2.0,
        fermi_line=False,
    )
    dp.plots.plot_band_dispersion(
        eigenvalues[:, 1:],
        momentum,
        ax=ax_bands,
        color=BAND_COLOR_PI_STAR,
        linewidth=2.0,
        fermi_line=True,
        xlabel=r"$k - K$ ($\mathrm{\AA}^{-1}$)",
        title="staggered onsite energies open a 2 eV gap",
    )
    ax_bands.set_xlim(-0.6, 0.6)
    ax_bands.set_ylim(-3.4, 3.4)
    ax_bands.spines[["top", "right"]].set_visible(False)

    dp.plots.plot_spectral_cut(
        intensity,
        momentum,
        np.asarray(energy_axis),
        ax=ax_cut,
        colorbar_label="intensity",
        fermi_line=False,
        xlabel=r"$k - K$ ($\mathrm{\AA}^{-1}$)",
        title="simulated ARPES of the gapped valence band",
    )
    ax_cut.set_xlim(-0.6, 0.6)
    save(fig, "honeycomb-gap-bands-arpes.png")


def figure_doped_fermi_surface(
    crystal: dp.types.CrystalGeometry, model: dp.types.TBModel
) -> None:
    """Fermi surface of the hole-doped lattice across the full zone."""
    axis = jnp.linspace(-2.35, 2.35, 361)
    kgrid = dp.tightb.build_arpes_kmesh(axis, axis, 0.0, 0.0, crystal)
    _, intensity = spectral_intensity(
        model,
        kgrid.kpoints,
        jnp.asarray([0.0]),
        0.06,
        fermi_ev=-1.0,
    )
    fermi_map = (
        np.asarray(intensity)
        .reshape((*kgrid.mesh_shape, 1))
        .transpose((1, 0, 2))[:, :, 0]
    )
    hexagon = brillouin_zone_vertices(crystal)

    fig, ax = plt.subplots(figsize=(5.6, 5.0), constrained_layout=True)
    dp.plots.plot_momentum_map(
        jnp.asarray(fermi_map),
        axis,
        axis,
        ax=ax,
        cmap="inferno",
        xlabel=r"$k_x$ ($\mathrm{\AA}^{-1}$)",
        ylabel=r"$k_y$ ($\mathrm{\AA}^{-1}$)",
        title="Fermi surface, hole-doped by 1 eV",
    )
    ax.plot(hexagon[:, 0], hexagon[:, 1], color="w", lw=0.9, ls="--")
    ax.set_aspect("equal")
    save(fig, "graphene-doped-fermi-surface.png")


def figure_bi2se3_dft() -> None:
    """Bi2Se3 VASP bands beside their occupied ARPES-style map."""
    fermi_energy_ev = float(
        dp.inout.read_outcar(str(BI2SE3_DIR / "OUTCAR_SCF")).fermi_energy
    )
    geometry = dp.inout.read_poscar(str(BI2SE3_DIR / "POSCAR"))
    bands = dp.inout.read_eigenval(
        str(BI2SE3_DIR / "MGM" / "EIGENVAL"), fermi_energy=fermi_energy_ev
    )
    kpath_info = dp.inout.read_kpoints(str(BI2SE3_DIR / "MGM" / "KPOINTS"))
    bands, kpath_info, _ = dp.inout.dedupe_band_path(bands, kpath_info)
    k_distance = np.asarray(
        dp.tightb.kpath_arc_length(
            dp.types.make_kpath(bands.kpoints), geometry
        )
    )
    relative_energies = np.asarray(bands.eigenvalues) - fermi_energy_ev
    gamma_index = int(
        np.argmin(np.linalg.norm(np.asarray(bands.kpoints), axis=1))
    )
    centered = k_distance - k_distance[gamma_index]

    energy_axis = jnp.linspace(-1.25, 0.20, 301)
    visible = np.any(
        (relative_energies >= float(energy_axis[0]) - 0.15)
        & (relative_energies <= float(energy_axis[-1]) + 0.15),
        axis=0,
    )
    eigenvalues = jnp.asarray(np.asarray(bands.eigenvalues)[:, visible])
    weights = jnp.ones(
        (eigenvalues.shape[0], energy_axis.shape[0], eigenvalues.shape[1])
    )
    intensity = dp.simul.assemble_spectral_intensity_bands_chunk(
        eigenvalues,
        weights,
        energy_axis,
        dp.types.make_self_energy_model(gamma=0.030),
        jnp.asarray(fermi_energy_ev),
        35.0,
        allow_degenerate_value_only=True,
    )

    fig, (ax_bands, ax_map) = plt.subplots(
        1, 2, figsize=(12.6, 4.9), constrained_layout=True
    )
    dp.plots.plot_band_dispersion(
        relative_energies,
        centered,
        ax=ax_bands,
        color=BAND_COLOR_PI,
        linewidth=0.4,
        fermi_line=True,
        xlabel=r"M-$\Gamma$-M distance ($\mathrm{\AA}^{-1}$)",
        title=r"Bi$_2$Se$_3$ slab bands from VASP EIGENVAL",
    )
    ax_bands.set_xlim(-0.35, 0.35)
    ax_bands.set_ylim(-1.1, 0.35)
    ax_bands.spines[["top", "right"]].set_visible(False)

    dp.plots.plot_spectral_cut(
        intensity,
        centered,
        np.asarray(energy_axis),
        ax=ax_map,
        fermi_line=False,
        xlabel=r"M-$\Gamma$-M distance ($\mathrm{\AA}^{-1}$)",
        title="occupied ARPES-style map at 35 K",
    )
    ax_map.set_xlim(-0.35, 0.35)
    save(fig, "bi2se3-dft-bands-to-arpes.png", tight=False)


def load_bi2se3_path(
    calc_dir: Path, band_subdir: str
) -> tuple[np.ndarray, np.ndarray, float]:
    """Load one Bi2Se3 VASP band path centered at Gamma."""
    fermi_energy_ev = float(
        dp.inout.read_outcar(str(calc_dir / "OUTCAR_SCF")).fermi_energy
    )
    geometry = dp.inout.read_poscar(str(calc_dir / "POSCAR"))
    bands = dp.inout.read_eigenval(
        str(calc_dir / band_subdir / "EIGENVAL"),
        fermi_energy=fermi_energy_ev,
    )
    kpath_info = dp.inout.read_kpoints(str(calc_dir / band_subdir / "KPOINTS"))
    bands, kpath_info, _ = dp.inout.dedupe_band_path(bands, kpath_info)
    k_distance = np.asarray(
        dp.tightb.kpath_arc_length(
            dp.types.make_kpath(bands.kpoints), geometry
        )
    )
    gamma_index = int(
        np.argmin(np.linalg.norm(np.asarray(bands.kpoints), axis=1))
    )
    centered = k_distance - k_distance[gamma_index]
    return centered, np.asarray(bands.eigenvalues), fermi_energy_ev


def vasp_intensity_map(
    eigenvalues: np.ndarray,
    fermi_energy_ev: float,
    energy_axis: Array,
    gamma_ev: float,
) -> np.ndarray:
    """Occupied unit-weight intensity map from parsed VASP eigenvalues."""
    relative = eigenvalues - fermi_energy_ev
    visible = np.any(
        (relative >= float(energy_axis[0]) - 0.15)
        & (relative <= float(energy_axis[-1]) + 0.15),
        axis=0,
    )
    kept = jnp.asarray(eigenvalues[:, visible])
    weights = jnp.ones((kept.shape[0], energy_axis.shape[0], kept.shape[1]))
    return np.asarray(
        dp.simul.assemble_spectral_intensity_bands_chunk(
            kept,
            weights,
            energy_axis,
            dp.types.make_self_energy_model(gamma=gamma_ev),
            jnp.asarray(fermi_energy_ev),
            35.0,
            allow_degenerate_value_only=True,
        )
    )


def figure_bi2se3_surface_state() -> None:
    """Near-Fermi window of the slab: states inside the bulk gap."""
    centered, eigenvalues, fermi_ev = load_bi2se3_path(BI2SE3_DIR, "MGM")
    energy_axis = jnp.linspace(-0.45, 0.12, 241)
    intensity = vasp_intensity_map(eigenvalues, fermi_ev, energy_axis, 0.012)
    fig, ax = plt.subplots(figsize=(6.4, 5.2), constrained_layout=True)
    dp.plots.plot_spectral_cut(
        intensity,
        centered,
        np.asarray(energy_axis),
        ax=ax,
        fermi_line=False,
        xlabel=r"M-$\Gamma$-M distance ($\mathrm{\AA}^{-1}$)",
        title="near-Fermi window of the slab calculation",
    )
    ax.set_xlim(-0.16, 0.16)
    save(fig, "bi2se3-surface-state-window.png", tight=False)


def figure_bi2se3_slab_vs_bulk() -> None:
    """Slab and bulk maps on the same path and energy window."""
    energy_axis = jnp.linspace(-1.05, 0.15, 241)
    panels = (
        ("6QL slab", *load_bi2se3_path(BI2SE3_DIR, "MGKM")),
        ("bulk", *load_bi2se3_path(BI2SE3_BULK_DIR, "MGKM")),
    )
    fig, axes = plt.subplots(
        1, 2, figsize=(12.6, 4.9), constrained_layout=True
    )
    image = None
    for index, (ax, (title, centered, eigenvalues, fermi_ev)) in enumerate(
        zip(axes, panels, strict=True)
    ):
        intensity = vasp_intensity_map(
            eigenvalues, fermi_ev, energy_axis, 0.030
        )
        _, _, image = dp.plots.plot_spectral_cut(
            intensity,
            centered,
            np.asarray(energy_axis),
            ax=ax,
            colorbar=False,
            fermi_line=False,
            xlabel=r"M-$\Gamma$-K-M distance ($\mathrm{\AA}^{-1}$)",
            ylabel=r"$E - E_F$ (eV)" if index == 0 else "",
            title=title,
        )
        ax.set_xlim(-0.5, 0.5)
    fig.colorbar(image, ax=axes, label="intensity (1/eV)", shrink=0.9)
    save(fig, "bi2se3-slab-vs-bulk.png", tight=False)


def figure_bi2se3_edc_stack() -> None:
    """Energy distribution curves from the slab band-path map."""
    centered, eigenvalues, fermi_ev = load_bi2se3_path(BI2SE3_DIR, "MGM")
    energy_axis = jnp.linspace(-1.25, 0.20, 301)
    intensity = vasp_intensity_map(eigenvalues, fermi_ev, energy_axis, 0.030)
    fig, ax = plt.subplots(figsize=(6.8, 4.6), constrained_layout=True)
    dp.plots.plot_distribution_curves(
        intensity,
        centered,
        np.asarray(energy_axis),
        kind="edc",
        positions=tuple(float(k) for k in np.linspace(-0.15, 0.15, 9)),
        ax=ax,
        offset=90.0,
        cmap="viridis",
        legend=False,
        ylabel="intensity, offset per EDC",
        title=(
            r"EDCs from $-0.15$ to $+0.15$ $\mathrm{\AA}^{-1}$ about $\Gamma$"
        ),
    )
    ax.spines[["top", "right"]].set_visible(False)
    save(fig, "bi2se3-edc-stack.png")


def figure_bi2se3_dos() -> None:
    """Overlay the normalized slab and bulk DOSCAR densities."""
    dos_curves = tuple(
        cast(
            dp.types.DensityOfStates,
            dp.inout.read_doscar(str(calc_dir / "DOSCAR")),
        )
        for calc_dir in (BI2SE3_DIR, BI2SE3_BULK_DIR)
    )
    fig, ax = plt.subplots(figsize=(7.6, 4.2), constrained_layout=True)
    dp.plots.plot_dos_overlay(
        dos_curves,
        labels=("6QL slab", "bulk"),
        ax=ax,
        energy_window=(-DOS_WINDOW_EV, DOS_WINDOW_EV),
        colors=SERIES_COLORS[:2],
        legend=False,
        title="DOSCAR densities of states",
    )
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    save(fig, "bi2se3-dos.png")


def figure_bi2se3_charge_profile() -> None:
    """Planar-averaged CHGCAR charge density across the slab."""
    volume = dp.inout.read_chgcar(str(BI2SE3_DIR / "CHGCAR"))
    positions, profile = dp.inout.planar_average(volume)
    fig, ax = plt.subplots(figsize=(8.6, 3.9), constrained_layout=True)
    dp.plots.plot_planar_average(
        positions,
        profile,
        ax=ax,
        color=BAND_COLOR_PI,
        title="CHGCAR charge density across six quintuple layers",
    )
    ax.spines[["top", "right"]].set_visible(False)
    save(fig, "bi2se3-charge-profile.png")


def build_metal_model() -> tuple[dp.types.CrystalGeometry, dp.types.TBModel]:
    """Build a square-lattice s-band metal with its dome at Gamma."""
    crystal = dp.types.make_crystal_geometry(
        lattice=2.0 * jnp.pi * jnp.eye(3),
        positions=jnp.zeros((1, 3)),
        species=("X",),
    )
    basis = dp.types.make_orbital_basis(
        atom_indices=(0,), n=(1,), l=(0,), m=(0,), labels=("1s",)
    )
    model = dp.types.make_tb_model(
        hopping_amplitudes=0.18 * jnp.ones(4, dtype=jnp.complex128),
        onsite_energies=jnp.asarray([-0.36]),
        soc_lambdas=jnp.zeros(0),
        geometry=crystal,
        basis=basis,
        hopping_pairs=((0, 0), (0, 0), (0, 0), (0, 0)),
        hopping_cells=((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0)),
        shell_index=(-1,),
        depths=jnp.asarray([0.25]),
    )
    return crystal, model


def figures_detector_chain() -> None:
    """Detector-chain figures: counts, one acquisition, and contrast.

    The detector sees a window of emission angles around the surface
    normal, so the source is a band whose occupied dome sits at Gamma.
    """
    crystal, model = build_metal_model()
    mesh_axis = jnp.linspace(-0.22, 0.22, 21)
    kgrid = dp.tightb.build_arpes_kmesh(
        mesh_axis, mesh_axis, 0.0, 0.0, crystal
    )
    hamiltonians = dp.tightb.bloch_hamiltonian_batch(model, kgrid.kpoints)
    bands = dp.tightb.diagonalize_tb(model, kgrid.kpoints)
    energy_axis = jnp.linspace(-0.24, 0.08, 49)
    self_energy = dp.types.make_self_energy_model(gamma=0.035)

    basis = model.basis
    radial_spec = dp.types.make_radial_spec(
        basis,
        (0,),
        mode="fixed",
        fixed_integrals_shell=jnp.asarray([[0.0, 1.0]]),
    )
    matrix_element_params = dp.types.make_matrix_element_params(
        basis,
        (0,),
        sigma_shell=jnp.asarray([1.0]),
        phase_shift_angles_shell=jnp.asarray([0.15]),
    )
    calibration = dp.types.make_detector_calibration(
        u_bin_edges=jnp.linspace(-0.075, 0.075, 17),
        v_bin_edges=jnp.linspace(-0.075, 0.075, 17),
        energy_bin_edges_ev=jnp.linspace(-0.22, 0.06, 25),
        psf_fwhm_u=0.006,
        psf_fwhm_v=0.006,
        psf_fwhm_energy_ev=0.020,
        transmission_reference_domain_ev=jnp.asarray([44.5, 46.0]),
    )
    detector_effects = dp.types.make_detector_effects(
        domain_logits=jnp.asarray([0.0]),
        domain_euler_angles_rad=jnp.zeros((1, 3)),
        transmission_raw_slopes=jnp.asarray([-0.65, 0.30]),
        background_coefficients=jnp.asarray([-8.0]),
        sensitivity_coefficients=jnp.asarray([]),
        exposure=1.0e11,
        background_mode="flat",
        sensitivity_mode="constant",
        domain_frame_ids=("org.diffpes.frame.sample_cartesian",),
    )

    def run(polarization: Array) -> np.ndarray:
        experiment = dp.types.make_experiment_geometry(
            photon_energy_ev=50.0,
            polarization=polarization,
            work_function_ev=4.5,
            temperature_k=25.0,
            mean_free_path_ang=8.0,
        )
        detector = dp.simul.simulate_arpes(
            (hamiltonians,),
            (bands,),
            radial_spec,
            matrix_element_params,
            dp.types.make_radial_quadrature_spec(),
            dp.types.make_final_state_spec(),
            experiment,
            self_energy,
            kgrid,
            energy_axis,
            calibration,
            detector_effects,
            k_chunk=64,
            energy_chunk=16,
            checkpoint=True,
        )
        return np.asarray(detector.expected_counts)

    expected = run(jnp.asarray([1.0 + 0.0j, 0.25j, 0.0j]))
    rotated = run(jnp.asarray([0.0j, 1.0 + 0.0j, 0.0j]))
    observed = np.asarray(
        dp.simul.sample_poisson_counts(
            jax.random.key(20260814), jnp.asarray(expected)
        ),
        dtype=np.float64,
    )

    fig, ax = plt.subplots(figsize=(5.6, 4.8), constrained_layout=True)
    dp.plots.plot_detector_image(
        expected[0],
        ax=ax,
        colorbar_label="expected events",
        xlabel="detector u bin",
        ylabel="detector v bin",
        title="expected counts on the detector plane",
    )
    save(fig, "detector-expected-counts.png")

    fig, (ax_expected, ax_observed) = plt.subplots(
        1, 2, figsize=(11.6, 4.8), constrained_layout=True
    )
    _, _, images = dp.plots.plot_detector_comparison(
        expected[0].sum(axis=1, keepdims=True),
        observed[0].sum(axis=1, keepdims=True),
        view="energy",
        axes=(ax_expected, ax_observed),
        colorbar=False,
        titles=("expected image", "one Poisson acquisition"),
    )
    for ax in (ax_expected, ax_observed):
        ax.set_ylabel("detector energy bin")
    fig.colorbar(images[0], ax=(ax_expected, ax_observed), label="events")
    save(fig, "detector-poisson-acquisition.png")

    expected_map = expected[0].sum(axis=-1)
    contrast = expected_map - rotated[0].sum(axis=-1)
    fig, ax = plt.subplots(figsize=(5.6, 4.8), constrained_layout=True)
    dp.plots.plot_difference_map(
        contrast,
        np.arange(contrast.shape[0], dtype=np.float64),
        np.arange(contrast.shape[1], dtype=np.float64),
        ax=ax,
        colorbar_label="count difference",
        xlabel="detector u bin",
        ylabel="detector v bin",
        title="polarization contrast of expected counts",
    )
    ax.set_aspect("equal")
    save(fig, "detector-polarization-contrast.png")

    figure_intrinsic_vs_measured(
        crystal, model, float(bands.fermi_energy), observed
    )


def figure_intrinsic_vs_measured(
    crystal: dp.types.CrystalGeometry,
    model: dp.types.TBModel,
    fermi_ev: float,
    observed: np.ndarray,
) -> None:
    """Compare the intrinsic spectral function with counted data."""
    energy_axis_dense = jnp.linspace(-0.24, 0.08, 231)
    path = dp.tightb.build_kpath(
        jnp.asarray([[-0.22, 0.0, 0.0], [0.22, 0.0, 0.0]]),
        crystal,
        301,
        ("start", "end"),
    )
    arc = np.asarray(dp.tightb.kpath_arc_length(path, crystal))
    momentum = arc - 0.5 * float(arc[-1])
    _, intrinsic_jax = spectral_intensity(
        model,
        path.kpoints,
        energy_axis_dense,
        0.035,
        fermi_ev=fermi_ev,
        temperature_k=25.0,
    )
    intrinsic = np.asarray(intrinsic_jax)
    fig, (ax_intrinsic, ax_measured) = plt.subplots(
        1, 2, figsize=(11.6, 4.8), constrained_layout=True
    )
    dp.plots.plot_spectral_cut(
        intrinsic,
        momentum,
        np.asarray(energy_axis_dense),
        ax=ax_intrinsic,
        colorbar=False,
        fermi_line=False,
        xlabel=r"$k_x$ ($\mathrm{\AA}^{-1}$)",
        title="intrinsic spectral function",
    )
    dp.plots.plot_detector_energy_cut(
        observed[0].sum(axis=1, keepdims=True),
        cut_axis="u",
        ax=ax_measured,
        log_counts=False,
        colorbar=False,
        xlabel="detector u bin",
        ylabel="detector energy bin",
        title="after optics, resolution, background, counting",
    )
    save(fig, "intrinsic-vs-measured.png")


def main() -> None:
    """Render and save all README figures."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    crystal, model = build_honeycomb_model()
    probe = dp.tightb.diagonalize_tb(
        model,
        jnp.asarray([list(K_POINT_FRAC), [0.5, 0.5, 0.0], [0.0, 0.0, 0.0]]),
    )
    probe_energies = np.asarray(probe.eigenvalues)
    np.testing.assert_allclose(probe_energies[0], 0.0, atol=1.0e-9)
    np.testing.assert_allclose(
        np.abs(probe_energies[1]), HOPPING_EV, atol=1.0e-9
    )
    np.testing.assert_allclose(
        np.abs(probe_energies[2]), 3.0 * HOPPING_EV, atol=1.0e-9
    )
    figure_bands_to_arpes(crystal, model)
    figure_constant_energy_maps(crystal, model)
    cube = build_cone_cube(crystal, model)
    figure_cone_cube(cube)
    figure_energy_window_maps(cube)
    figure_band_surface(crystal, model)
    figure_dos(crystal, model)
    figure_edc_mdc(crystal, model)
    figure_fermi_edge(crystal, model)
    figure_linewidth_series(crystal, model)
    figure_gapped_honeycomb(crystal)
    figure_doped_fermi_surface(crystal, model)
    figure_bi2se3_dft()
    figure_bi2se3_surface_state()
    figure_bi2se3_slab_vs_bulk()
    figure_bi2se3_edc_stack()
    figure_bi2se3_dos()
    figure_bi2se3_charge_profile()
    figures_detector_chain()
    for name in FIGURES:
        print("wrote", OUTPUT_DIR / name)


if __name__ == "__main__":
    main()
