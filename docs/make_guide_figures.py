"""Generate the committed spectra figures for the documentation guides.

Extended Summary
----------------
Every figure under ``docs/source/guides/figures`` comes from this script and
the public diffpes API. The sources are a graphene tight-binding model with
Dirac cones, the square-lattice model from the simulation guide, and the
compact bulk model used for the photon-energy guide. Rerun the script after
an API change that alters any depicted spectrum.

Run the script from the repository root::

    .venv/bin/python docs/make_guide_figures.py
"""

from __future__ import annotations

from pathlib import Path

import equinox
import jax
import matplotlib

matplotlib.use("Agg")

import jax.numpy as jnp
import matplotlib.pyplot as plt

from diffpes import matrixel, simul, tightb, types
from diffpes.plots import (
    plot_band_dispersion,
    plot_cube_faces,
    plot_curve_family,
    plot_detector_energy_cut,
    plot_detector_image,
    plot_difference_map,
    plot_momentum_map,
    plot_momentum_map_grid,
    plot_spectral_cut,
    plot_spectral_cut_series,
)

FIGURE_DIRECTORY: Path = (
    Path(__file__).parent / "source" / "guides" / "figures"
)
SAVE_OPTIONS: dict = {"dpi": 200, "bbox_inches": "tight"}
GRAPHENE_HOPPING_EV: float = -2.7
GRAPHENE_FERMI_EV: float = 0.4
GRAPHENE_GAMMA_EV: float = 0.06
TEMPERATURE_K: float = 30.0


def build_graphene() -> types.TBModel:
    """Return the nearest-neighbor graphene pz model.

    Returns
    -------
    model : types.TBModel
        Two-orbital model with Hermitian-closed hoppings.
    """
    a: float = 2.46
    geometry: types.CrystalGeometry = types.make_crystal_geometry(
        lattice=jnp.asarray(
            [
                [a, 0.0, 0.0],
                [0.5 * a, 0.5 * jnp.sqrt(3.0) * a, 0.0],
                [0.0, 0.0, 10.0],
            ]
        ),
        positions=jnp.asarray([[0.0, 0.0, 0.0], [1.0 / 3.0, 1.0 / 3.0, 0.0]]),
        species=("C", "C"),
    )
    basis: types.OrbitalBasis = types.make_orbital_basis(
        atom_indices=(0, 1),
        n=(2, 2),
        l=(1, 1),
        m=(0, 0),
        labels=("A_pz", "B_pz"),
    )
    model: types.TBModel = types.make_tb_model(
        hopping_amplitudes=GRAPHENE_HOPPING_EV
        * jnp.ones(6, dtype=jnp.complex128),
        onsite_energies=jnp.asarray([0.008, -0.008]),
        soc_lambdas=jnp.zeros(0),
        geometry=geometry,
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
    )
    return model


def intrinsic_spectrum(
    eigenvalues: jnp.ndarray,
    weights: jnp.ndarray,
    energy_axis: jnp.ndarray,
    gamma_ev: float = GRAPHENE_GAMMA_EV,
    fermi_ev: float = GRAPHENE_FERMI_EV,
) -> jnp.ndarray:
    """Return the occupied intrinsic spectrum on the sampled energy axis.

    Parameters
    ----------
    eigenvalues : jnp.ndarray
        Band energies with shape ``[K, B]``.
    weights : jnp.ndarray
        Gauge-invariant band weights with shape ``[K, B]``.
    energy_axis : jnp.ndarray
        Sampled energies relative to the Fermi level in eV.
    gamma_ev : float
        Constant self-energy linewidth in eV.
    fermi_ev : float
        Fermi energy on the eigenvalue scale in eV.

    Returns
    -------
    intensity : jnp.ndarray
        Occupied spectral intensity with shape ``[K, E]``.
    """
    self_energy: types.SelfEnergyModel = types.make_self_energy_model(
        gamma=gamma_ev
    )
    weights_ke: jnp.ndarray = jnp.broadcast_to(
        weights[:, None, :],
        (weights.shape[0], energy_axis.shape[0], weights.shape[1]),
    )
    intensity: jnp.ndarray = simul.assemble_spectral_intensity_bands_chunk(
        eigenvalues,
        weights_ke,
        energy_axis,
        self_energy,
        jnp.asarray(fermi_ev),
        TEMPERATURE_K,
    )
    return intensity


def graphene_cube() -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Return a Cartesian graphene spectral cube around the zone center.

    Returns
    -------
    axis : jnp.ndarray
        Shared Cartesian kx and ky axis in 1/Angstrom.
    energy_axis : jnp.ndarray
        Sampled energies relative to the Fermi level in eV.
    cube : jnp.ndarray
        Occupied intensity with shape ``[n_kx, n_ky, n_e]``.
    """
    model: types.TBModel = build_graphene()
    n_k: int = 141
    axis: jnp.ndarray = jnp.linspace(-2.2, 2.2, n_k)
    mesh_x, mesh_y = jnp.meshgrid(axis, axis, indexing="ij")
    kpoints_cart: jnp.ndarray = jnp.stack(
        (mesh_x, mesh_y, jnp.zeros_like(mesh_x)), axis=-1
    ).reshape((-1, 3))
    kpoints_frac: jnp.ndarray = tightb.kpoints_cart_to_frac(
        kpoints_cart, model.geometry
    )
    eigenvalues: jnp.ndarray = tightb.eigvalsh_bands(model, kpoints_frac)
    energy_axis: jnp.ndarray = jnp.linspace(-2.6, 0.5, 121)
    weights: jnp.ndarray = jnp.ones_like(eigenvalues)
    cube: jnp.ndarray = intrinsic_spectrum(
        eigenvalues, weights, energy_axis
    ).reshape((n_k, n_k, -1))
    return axis, energy_axis, cube


def graphene_path_spectrum() -> tuple[
    jnp.ndarray, jnp.ndarray, jnp.ndarray, tuple, tuple
]:
    """Return the graphene spectrum along the Gamma-K-M-Gamma path.

    Returns
    -------
    distance : jnp.ndarray
        Cartesian arc length along the path in 1/Angstrom.
    energy_axis : jnp.ndarray
        Sampled energies relative to the Fermi level in eV.
    intensity : jnp.ndarray
        Occupied intensity with shape ``[K, E]``.
    tick_positions : tuple
        Arc-length positions of the labeled anchors.
    tick_labels : tuple
        Anchor labels.
    """
    model: types.TBModel = build_graphene()
    anchors: jnp.ndarray = jnp.asarray(
        [
            [0.0, 0.0, 0.0],
            [2.0 / 3.0, 1.0 / 3.0, 0.0],
            [0.5, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ]
    )
    path: types.KPath = tightb.build_kpath(
        anchors,
        model.geometry,
        n_per_segment=161,
        labels=("$\\Gamma$", "K", "M", "$\\Gamma$"),
    )
    distance: jnp.ndarray = tightb.kpath_arc_length(path, model.geometry)
    eigenvalues: jnp.ndarray = tightb.eigvalsh_bands(model, path.kpoints)
    energy_axis: jnp.ndarray = jnp.linspace(-8.6, 0.5, 241)
    intensity: jnp.ndarray = intrinsic_spectrum(
        eigenvalues, jnp.ones_like(eigenvalues), energy_axis
    )
    ticks: tuple = tuple(
        float(distance[index]) for index in path.label_indices
    )
    labels: tuple = tuple(path.labels)
    return distance, energy_axis, intensity, ticks, labels


def figure_pipeline_cube() -> None:
    """Draw the three-face spectral cube for the simulation guide.

    The cube corner sits on one K point, so the outer momentum faces are
    momentum-energy slices through Dirac cones while the top face is a
    constant-energy map. Each face is normalized to its own maximum for
    display.
    """
    model: types.TBModel = build_graphene()
    corner_x: float = 4.0 * jnp.pi / (3.0 * 2.46)
    corner_y: float = 2.0 * jnp.pi / (jnp.sqrt(3.0) * 2.46)
    n_k: int = 161
    axis_x: jnp.ndarray = jnp.linspace(-corner_x - 0.4, corner_x, n_k)
    axis_y: jnp.ndarray = jnp.linspace(-corner_y - 0.4, corner_y, n_k)
    mesh_x, mesh_y = jnp.meshgrid(axis_x, axis_y, indexing="ij")
    kpoints_cart: jnp.ndarray = jnp.stack(
        (mesh_x, mesh_y, jnp.zeros_like(mesh_x)), axis=-1
    ).reshape((-1, 3))
    kpoints_frac: jnp.ndarray = tightb.kpoints_cart_to_frac(
        kpoints_cart, model.geometry
    )
    eigenvalues: jnp.ndarray = tightb.eigvalsh_bands(model, kpoints_frac)
    energy_axis: jnp.ndarray = jnp.linspace(-3.5, -0.1, 121)
    intensity: jnp.ndarray = intrinsic_spectrum(
        eigenvalues,
        jnp.ones_like(eigenvalues),
        energy_axis,
        gamma_ev=0.09,
    ).reshape((n_k, n_k, -1))
    cube: types.ArpesCube = types.make_arpes_cube(
        intensity, axis_x, axis_y, energy_axis
    )
    figure = plt.figure(figsize=(7.2, 6.4))
    axes3d = figure.add_subplot(projection="3d", computed_zorder=False)
    plot_cube_faces(
        cube,
        ax=axes3d,
        cmap="inferno",
        title="graphene spectral cube $I(k_x, k_y, E)$",
    )
    axes3d.set_zlabel(r"$E-E_F$ (eV)", labelpad=12.0)
    figure.savefig(FIGURE_DIRECTORY / "pipeline-cube.png", dpi=200)
    plt.close(figure)


def figure_pipeline_ek_and_map() -> None:
    """Draw the path spectrum and the Fermi-surface map."""
    distance, energy_axis, intensity, ticks, labels = graphene_path_spectrum()
    figure, axes = plt.subplots(figsize=(6.4, 4.2))
    plot_spectral_cut(
        intensity,
        distance,
        energy_axis,
        ax=axes,
        cmap="inferno",
        fermi_line=False,
        xlabel="",
        ylabel=r"$E-E_F$ (eV)",
        title=r"graphene $\pi$ bands: occupied spectral intensity",
    )
    axes.set_xticks(ticks, labels)
    figure.savefig(FIGURE_DIRECTORY / "pipeline-ek-cut.png", **SAVE_OPTIONS)
    plt.close(figure)

    axis, cube_energy_axis, cube = graphene_cube()
    fermi_index: int = int(jnp.argmin(jnp.abs(cube_energy_axis)))
    figure, axes = plt.subplots(figsize=(5.4, 4.6))
    plot_momentum_map(
        cube[:, :, fermi_index],
        axis,
        axis,
        ax=axes,
        cmap="inferno",
        title=r"Fermi-surface map at $E_F$",
    )
    figure.savefig(FIGURE_DIRECTORY / "pipeline-fermi-map.png", **SAVE_OPTIONS)
    plt.close(figure)


def figure_geometry() -> None:
    """Draw kinematics curves and constant-energy slices."""
    k_parallel: jnp.ndarray = jnp.asarray([0.0, 0.6, 1.2, 1.8])
    photon_energies: jnp.ndarray = jnp.linspace(15.0, 120.0, 400)
    kz_curves: list = []
    kz_labels: list = []
    for k_par in k_parallel:
        kz_values, propagating = jax.vmap(
            lambda hv, kp=k_par: simul.kz_from_inner_potential(
                hv, 4.5, 12.0, jnp.asarray(0.0), jnp.asarray(kp)
            )
        )(photon_energies)
        kz_curves.append(jnp.where(propagating, jnp.real(kz_values), jnp.nan))
        kz_labels.append(rf"$k_\parallel = {float(k_par):.1f}$")
    figure, axes = plt.subplots(figsize=(6.2, 3.8))
    plot_curve_family(
        photon_energies,
        tuple(kz_curves),
        labels=tuple(kz_labels),
        ax=axes,
        legend_title=r"$\mathrm{\AA}^{-1}$",
        xlabel=r"photon energy $h\nu$ (eV)",
        ylabel=r"$k_z$ ($\mathrm{\AA}^{-1}$)",
        title=r"free-electron $k_z(h\nu)$ with $V_0 = 12$ eV",
    )
    figure.savefig(FIGURE_DIRECTORY / "geometry-kz-hv.png", **SAVE_OPTIONS)
    plt.close(figure)

    axis, energy_axis, cube = graphene_cube()
    slice_energies: tuple = (0.0, -0.8, -1.6)
    slice_maps: list = []
    slice_titles: list = []
    for target in slice_energies:
        index: int = int(jnp.argmin(jnp.abs(energy_axis - target)))
        slice_maps.append(cube[:, :, index])
        slice_titles.append(rf"$E-E_F = {target:.1f}$ eV")
    figure, axes_row = plt.subplots(
        1, 3, figsize=(11.4, 3.9), sharey=True, constrained_layout=True
    )
    plot_momentum_map_grid(
        tuple(slice_maps),
        axis,
        axis,
        titles=tuple(slice_titles),
        axes=tuple(axes_row),
        colorbar_label="intensity (1/eV)",
    )
    figure.savefig(
        FIGURE_DIRECTORY / "geometry-energy-slices.png", **SAVE_OPTIONS
    )
    plt.close(figure)


def figure_kz_guide() -> None:
    """Draw wrapped escape-depth weights for the kz guide."""
    edges: jnp.ndarray = jnp.linspace(-0.5, 0.5, 257)
    nodes: jnp.ndarray = simul.kz_fractional_nodes(256)
    reciprocal_period: jnp.ndarray = jnp.asarray(2.0 * jnp.pi / 3.2)
    mean_free_paths: jnp.ndarray = jnp.asarray([4.0, 7.5, 15.0])
    weights: jnp.ndarray = jax.vmap(
        simul.kz_wrapped_lorentzian_bin_weights,
        in_axes=(None, None, 0, None),
    )(edges, jnp.asarray(0.0), mean_free_paths, reciprocal_period)
    curves: tuple = tuple(values * nodes.shape[0] for values in weights)
    labels: tuple = tuple(
        rf"$\lambda = {float(path_length):.1f}$ $\mathrm{{\AA}}$"
        for path_length in mean_free_paths
    )
    figure, axes = plt.subplots(figsize=(6.2, 3.8))
    plot_curve_family(
        nodes,
        curves,
        labels=labels,
        ax=axes,
        xlabel=r"surface-fractional $k_z$",
        ylabel="wrapped Lorentzian density",
        title="escape depth controls the $k_z$ width",
    )
    figure.savefig(FIGURE_DIRECTORY / "kz-wrapped-weights.png", **SAVE_OPTIONS)
    plt.close(figure)


def bulk_scan_inputs() -> dict:
    """Return the shared carriers for the bulk photon-energy figures.

    Returns
    -------
    carriers : dict
        Keyword inputs for :func:`diffpes.simul.simulate_hv_scan`.
    """
    lattice_scale: float = 3.2
    crystal: types.CrystalGeometry = types.make_crystal_geometry(
        lattice_scale * jnp.eye(3), jnp.zeros((1, 3)), ("X",)
    )
    basis: types.OrbitalBasis = types.make_orbital_basis(
        atom_indices=(0,), n=(1,), l=(0,), m=(0,), labels=("1s",)
    )
    bulk_model: types.TBModel = types.make_tb_model(
        hopping_amplitudes=jnp.asarray(
            (-0.12, -0.12, -0.38, -0.38), dtype=jnp.complex128
        ),
        onsite_energies=jnp.asarray((-0.05,)),
        soc_lambdas=jnp.zeros((0,)),
        geometry=crystal,
        basis=basis,
        hopping_pairs=((0, 0), (0, 0), (0, 0), (0, 0)),
        hopping_cells=((1, 0, 0), (-1, 0, 0), (0, 0, 1), (0, 0, -1)),
        shell_index=(-1,),
    )
    surface_cell: types.SurfaceCell = types.make_surface_cell(
        in_plane_vectors=lattice_scale
        * jnp.asarray(((1.0, 0.0, 0.0), (0.0, 1.0, 0.0))),
        stacking_vector=lattice_scale * jnp.asarray((0.0, 0.0, 1.0)),
        rotation=jnp.eye(3),
        interlayer_spacing_ang=lattice_scale,
        miller=(0, 0, 1),
        in_plane_coeffs=((1, 0, 0), (0, 1, 0)),
        stacking_coeffs=(0, 0, 1),
    )
    path_coordinate: jnp.ndarray = jnp.linspace(-0.09, 0.09, 25)
    path_points: jnp.ndarray = jnp.stack(
        (
            path_coordinate,
            jnp.zeros_like(path_coordinate),
            jnp.zeros_like(path_coordinate),
        ),
        axis=-1,
    )
    carriers: dict = {
        "bulk_model": bulk_model,
        "surface_cell": surface_cell,
        "basis": basis,
        "kpath": types.make_kpath(path_points, kz=0.0),
        "path_coordinate": path_coordinate,
        "radial_spec": types.make_radial_spec(
            basis,
            (0,),
            mode="fixed",
            fixed_integrals_shell=jnp.asarray(((0.0, 1.0),)),
        ),
        "matrix_element_params": types.make_matrix_element_params(
            basis,
            (0,),
            sigma_shell=jnp.asarray((1.13,)),
            phase_shift_angles_shell=jnp.asarray((0.21,)),
        ),
        "experiment": types.make_experiment_geometry(
            photon_energy_ev=28.0,
            polarization=simul.polarization_from_angles(0.75, 0.0, "p"),
            incidence_theta=0.75,
            incidence_phi=0.0,
            work_function_ev=4.3,
            inner_potential_ev=11.0,
            temperature_k=45.0,
            mean_free_path_ang=7.5,
        ),
    }
    return carriers


def figure_hv_scan() -> None:
    """Draw the photon-energy map for the kz guide."""
    carriers: dict = bulk_scan_inputs()
    energy_axis: jnp.ndarray = jnp.linspace(-1.15, 0.05, 61)
    photon_energies: jnp.ndarray = jnp.linspace(18.0, 80.0, 32)
    scan: jnp.ndarray = simul.simulate_hv_scan(
        None,
        None,
        carriers["radial_spec"],
        carriers["matrix_element_params"],
        types.make_radial_quadrature_spec(),
        types.make_final_state_spec(),
        carriers["experiment"],
        types.make_self_energy_model(gamma=0.035),
        carriers["kpath"],
        energy_axis,
        photon_energies,
        k_chunk=25,
        energy_chunk=61,
        checkpoint=True,
        bulk_model=carriers["bulk_model"],
        surface_cell=carriers["surface_cell"],
        kz_nodes_frac=simul.kz_fractional_nodes(64),
        kz_mode="bulk_kz",
    )
    normal_emission: jnp.ndarray = scan[:, scan.shape[1] // 2, :]
    figure, axes = plt.subplots(figsize=(6.2, 4.2))
    plot_spectral_cut(
        normal_emission,
        photon_energies,
        energy_axis,
        ax=axes,
        cmap="magma",
        colorbar_label="intensity",
        fermi_line=False,
        xlabel=r"photon energy $h\nu$ (eV)",
        ylabel=r"$E-E_F$ (eV)",
        title="normal-emission photon-energy scan, bulk $k_z$ mode",
    )
    figure.savefig(FIGURE_DIRECTORY / "kz-hv-scan.png", **SAVE_OPTIONS)
    plt.close(figure)


def figure_kz_broadening_compare() -> None:
    """Draw the exact-center versus kz-broadened comparison."""
    carriers: dict = bulk_scan_inputs()
    energy_axis: jnp.ndarray = jnp.linspace(-1.15, 0.05, 81)
    photon_energy: jnp.ndarray = jnp.asarray([43.0])
    maps: list = []
    modes: tuple = (
        ("bulk_direct", None, "exact final-state $k_z$ (bulk_direct)"),
        (
            "bulk_kz",
            simul.kz_fractional_nodes(512),
            r"$\lambda = 7.5$ $\mathrm{\AA}$ escape depth (bulk_kz)",
        ),
    )
    for kz_mode, nodes, _ in modes:
        scan: jnp.ndarray = simul.simulate_hv_scan(
            None,
            None,
            carriers["radial_spec"],
            carriers["matrix_element_params"],
            types.make_radial_quadrature_spec(),
            types.make_final_state_spec(),
            carriers["experiment"],
            types.make_self_energy_model(gamma=0.02),
            carriers["kpath"],
            energy_axis,
            photon_energy,
            k_chunk=25,
            energy_chunk=81,
            checkpoint=True,
            bulk_model=carriers["bulk_model"],
            surface_cell=carriers["surface_cell"],
            kz_nodes_frac=nodes,
            kz_mode=kz_mode,
        )
        maps.append(scan[0] / float(scan[0].max()))
    path_coordinate: jnp.ndarray = carriers["path_coordinate"]
    figure, axes_row = plt.subplots(
        1, 2, figsize=(9.6, 3.9), sharey=True, constrained_layout=True
    )
    plot_spectral_cut_series(
        tuple(maps),
        path_coordinate,
        energy_axis,
        titles=tuple(title for _, _, title in modes),
        axes=tuple(axes_row),
        cmap="magma",
        colorbar_label="normalized intensity",
        fermi_line=False,
        xlabel="fractional path coordinate",
        ylabel=r"$E-E_F$ (eV)",
    )
    figure.savefig(
        FIGURE_DIRECTORY / "kz-broadening-compare.png", **SAVE_OPTIONS
    )
    plt.close(figure)


def polarized_band_weights(
    polarization: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, tuple, tuple]:
    """Return graphene path weights for one laboratory polarization.

    Parameters
    ----------
    polarization : jnp.ndarray
        Complex Cartesian polarization amplitude.

    Returns
    -------
    distance : jnp.ndarray
        Path arc length in 1/Angstrom.
    eigenvalues : jnp.ndarray
        Band energies with shape ``[K, B]``.
    weights : jnp.ndarray
        Matrix-element band weights with shape ``[K, B]``.
    ticks : tuple
        Labeled anchor positions.
    labels : tuple
        Anchor labels.
    """
    model: types.TBModel = build_graphene()
    anchors: jnp.ndarray = jnp.asarray(
        [
            [0.0, 0.0, 0.0],
            [2.0 / 3.0, 1.0 / 3.0, 0.0],
            [0.5, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ]
    )
    path: types.KPath = tightb.build_kpath(
        anchors,
        model.geometry,
        n_per_segment=161,
        labels=("$\\Gamma$", "K", "M", "$\\Gamma$"),
    )
    distance: jnp.ndarray = tightb.kpath_arc_length(path, model.geometry)
    bands: types.DiagonalizedBands = tightb.diagonalize_tb(model, path.kpoints)
    experiment: types.ExperimentGeometry = types.make_experiment_geometry(
        photon_energy_ev=60.0,
        polarization=polarization,
        incidence_theta=0.75,
        incidence_phi=0.0,
        work_function_ev=4.5,
        temperature_k=TEMPERATURE_K,
        mean_free_path_ang=8.0,
    )
    radial: types.RadialSpec = types.make_radial_spec(
        model.basis,
        (0, 1),
        mode="fixed",
        fixed_integrals_shell=jnp.asarray([[0.6, 1.0], [0.6, 1.0]]),
    )
    me_params: types.MatrixElementParams = types.make_matrix_element_params(
        model.basis,
        (0, 1),
        sigma_shell=jnp.asarray([1.0, 1.0]),
        phase_shift_angles_shell=jnp.asarray([0.15, 0.4, 0.15, 0.4]),
    )
    kpoints_cart: jnp.ndarray = tightb.kpoints_frac_to_cart(
        path.kpoints, model.geometry
    )
    kinetic, emission_valid = simul.kinetic_energy_ev(
        60.0, 4.5, jnp.zeros(kpoints_cart.shape[0])
    )
    k_f_magnitude, _ = simul.final_state_k_inv_ang(kinetic)
    k_parallel_sq: jnp.ndarray = (
        kpoints_cart[:, 0] ** 2 + kpoints_cart[:, 1] ** 2
    )
    k_f_z: jnp.ndarray = jnp.sqrt(
        jnp.maximum(k_f_magnitude**2 - k_parallel_sq, 1.0e-12)
    )
    k_f_cart: jnp.ndarray = jnp.stack(
        (kpoints_cart[:, 0], kpoints_cart[:, 1], k_f_z), axis=-1
    )
    channels: jnp.ndarray = matrixel.assemble_orbital_transition_channels(
        bands,
        radial,
        me_params,
        types.make_radial_quadrature_spec(),
        types.make_final_state_spec(),
        experiment,
        k_f_cart,
        emission_valid,
    )
    band_channels: jnp.ndarray = matrixel.project_band_channels(
        channels, bands.eigenvectors
    )
    polarized: jnp.ndarray = simul.contract_experiment_polarization(
        band_channels, experiment
    )
    weights: jnp.ndarray = matrixel.matrix_element_intensity(polarized)
    ticks: tuple = tuple(
        float(distance[index]) for index in path.label_indices
    )
    return distance, bands.eigenvalues, weights, ticks, tuple(path.labels)


def figure_matrix_elements() -> None:
    """Draw polarization contrast and atomic cross sections."""
    polarizations: tuple = (
        (simul.polarization_from_angles(0.75, 0.0, "s"), "s polarization"),
        (simul.polarization_from_angles(0.75, 0.0, "p"), "p polarization"),
    )
    energy_axis: jnp.ndarray = jnp.linspace(-8.6, 0.5, 241)
    intensities: list = []
    panel_titles: list = []
    for polarization, label in polarizations:
        distance, eigenvalues, weights, ticks, tick_labels = (
            polarized_band_weights(polarization)
        )
        intensities.append(
            intrinsic_spectrum(eigenvalues, weights, energy_axis)
        )
        panel_titles.append(label)
    figure, axes_row = plt.subplots(
        1, 2, figsize=(10.4, 3.9), sharey=True, constrained_layout=True
    )
    plot_spectral_cut_series(
        tuple(intensities),
        distance,
        energy_axis,
        titles=tuple(panel_titles),
        axes=tuple(axes_row),
        cmap="inferno",
        fermi_line=False,
        xlabel="",
        ylabel=r"$E-E_F$ (eV)",
    )
    for axes in axes_row:
        axes.set_xticks(ticks, tick_labels)
    figure.savefig(
        FIGURE_DIRECTORY / "matrixel-polarization.png", **SAVE_OPTIONS
    )
    plt.close(figure)

    subshells: tuple = (
        (29, 3, 2, "Cu 3d"),
        (6, 2, 1, "C 2p"),
        (83, 6, 1, "Bi 6p"),
    )
    figure, axes = plt.subplots(figsize=(6.2, 3.8))
    for position, (atomic_number, n_quantum, l_quantum, label) in enumerate(
        subshells
    ):
        energy, sigma = simul.yeh_lindau_cross_section_table(
            atomic_number, n_quantum, l_quantum
        )[:2]
        plot_curve_family(
            energy,
            (sigma,),
            labels=(label,),
            ax=axes,
            log_x=True,
            log_y=True,
            legend=position == len(subshells) - 1,
            xlabel="photon energy (eV)",
            ylabel="cross section (Mb)",
            title="Yeh--Lindau atomic photoionization cross sections",
        )
    figure.savefig(
        FIGURE_DIRECTORY / "matrixel-cross-sections.png", **SAVE_OPTIONS
    )
    plt.close(figure)


def figure_broadening() -> None:
    """Draw lineshape profiles and self-energy comparisons."""
    energy_axis: jnp.ndarray = jnp.linspace(-0.5, 0.5, 601)
    profiles: tuple = (
        simul.gaussian(energy_axis, 0.0, 0.05),
        simul.voigt(energy_axis, 0.0, 0.05, 0.05),
        simul.voigt(energy_axis, 0.0, 0.008, 0.1),
    )
    profile_labels: tuple = (
        r"Gaussian $\sigma = 50$ meV",
        r"Voigt $\sigma = \gamma = 50$ meV",
        r"near-Lorentzian $\gamma = 100$ meV",
    )
    figure, axes = plt.subplots(figsize=(6.2, 3.8))
    plot_curve_family(
        energy_axis,
        profiles,
        labels=profile_labels,
        ax=axes,
        xlabel=r"$E - E_0$ (eV)",
        ylabel="normalized profile (1/eV)",
        title="energy lineshapes",
    )
    figure.savefig(
        FIGURE_DIRECTORY / "broadening-profiles.png", **SAVE_OPTIONS
    )
    plt.close(figure)

    distance, energy_axis, _, ticks, labels = graphene_path_spectrum()
    model: types.TBModel = build_graphene()
    anchors: jnp.ndarray = jnp.asarray(
        [
            [0.0, 0.0, 0.0],
            [2.0 / 3.0, 1.0 / 3.0, 0.0],
            [0.5, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ]
    )
    path: types.KPath = tightb.build_kpath(
        anchors,
        model.geometry,
        n_per_segment=161,
        labels=("$\\Gamma$", "K", "M", "$\\Gamma$"),
    )
    eigenvalues: jnp.ndarray = tightb.eigvalsh_bands(model, path.kpoints)
    weights: jnp.ndarray = jnp.ones_like(eigenvalues)
    gammas: tuple = (0.03, 0.25)
    figure, axes_row = plt.subplots(
        1, 2, figsize=(10.4, 3.9), sharey=True, constrained_layout=True
    )
    plot_spectral_cut_series(
        tuple(
            intrinsic_spectrum(eigenvalues, weights, energy_axis, gamma)
            for gamma in gammas
        ),
        distance,
        energy_axis,
        titles=tuple(
            rf"$\Gamma = {1000 * gamma:.0f}$ meV" for gamma in gammas
        ),
        axes=tuple(axes_row),
        cmap="inferno",
        fermi_line=False,
        xlabel="",
        ylabel=r"$E-E_F$ (eV)",
    )
    for axes in axes_row:
        axes.set_xticks(ticks, labels)
    figure.savefig(
        FIGURE_DIRECTORY / "broadening-linewidth-ek.png", **SAVE_OPTIONS
    )
    plt.close(figure)

    narrow: jnp.ndarray = intrinsic_spectrum(
        eigenvalues, weights, energy_axis, gamma_ev=0.03
    )
    convolved: jnp.ndarray = simul.convolve_energy(narrow, energy_axis, 0.12)
    figure, axes_row = plt.subplots(
        1, 2, figsize=(10.4, 3.9), sharey=True, constrained_layout=True
    )
    plot_spectral_cut_series(
        (narrow, convolved),
        distance,
        energy_axis,
        titles=("intrinsic", "with 120 meV Gaussian resolution"),
        axes=tuple(axes_row),
        cmap="inferno",
        fermi_line=False,
        xlabel="",
        ylabel=r"$E-E_F$ (eV)",
    )
    for axes in axes_row:
        axes.set_xticks(ticks, labels)
    figure.savefig(
        FIGURE_DIRECTORY / "broadening-resolution.png", **SAVE_OPTIONS
    )
    plt.close(figure)


def detector_example() -> types.DetectorRaster:
    """Return the expected counts from the simulation-guide example.

    Returns
    -------
    detector : types.DetectorRaster
        Native-raster expected counts.
    """
    crystal: types.CrystalGeometry = types.make_crystal_geometry(
        lattice=2.0 * jnp.pi * jnp.eye(3),
        positions=jnp.zeros((1, 3)),
        species=("X",),
    )
    basis: types.OrbitalBasis = types.make_orbital_basis(
        atom_indices=(0,), n=(1,), l=(0,), m=(0,), labels=("1s",)
    )
    model: types.TBModel = types.make_tb_model(
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
    k_axis: jnp.ndarray = jnp.linspace(-0.22, 0.22, 13)
    mesh_x, mesh_y = jnp.meshgrid(k_axis, k_axis, indexing="xy")
    kpoints: jnp.ndarray = jnp.stack(
        (mesh_x, mesh_y, jnp.zeros_like(mesh_x)), axis=-1
    ).reshape((-1, 3))
    kgrid: types.KGrid = types.make_kgrid(kpoints, mesh_shape=(13, 13), kz=0.0)
    hamiltonians: jnp.ndarray = tightb.bloch_hamiltonian_batch(model, kpoints)
    bands: types.DiagonalizedBands = tightb.diagonalize_tb(model, kpoints)
    experiment: types.ExperimentGeometry = types.make_experiment_geometry(
        photon_energy_ev=50.0,
        polarization=jnp.asarray([1.0 + 0.0j, 0.25j, 0.0j]),
        work_function_ev=4.5,
        temperature_k=25.0,
        mean_free_path_ang=8.0,
    )
    calibration: types.DetectorCalibration = types.make_detector_calibration(
        u_bin_edges=jnp.linspace(-0.050, 0.050, 17),
        v_bin_edges=jnp.linspace(-0.050, 0.050, 17),
        energy_bin_edges_ev=jnp.linspace(-0.22, 0.06, 29),
        psf_fwhm_u=0.008,
        psf_fwhm_v=0.010,
        psf_fwhm_energy_ev=0.025,
        transmission_reference_domain_ev=jnp.asarray([44.5, 46.0]),
    )
    detector_effects: types.DetectorEffects = types.make_detector_effects(
        domain_logits=jnp.asarray([0.0]),
        domain_euler_angles_rad=jnp.zeros((1, 3)),
        transmission_raw_slopes=jnp.asarray([-0.65, 0.30]),
        background_coefficients=jnp.asarray([-12.0]),
        sensitivity_coefficients=jnp.asarray([]),
        exposure=2.0e9,
        background_mode="flat",
        sensitivity_mode="constant",
        domain_frame_ids=("org.diffpes.frame.sample_cartesian",),
    )
    detector: types.DetectorRaster = simul.simulate_arpes(
        (hamiltonians,),
        (bands,),
        types.make_radial_spec(
            basis,
            (0,),
            mode="fixed",
            fixed_integrals_shell=jnp.asarray([[0.0, 1.0]]),
        ),
        types.make_matrix_element_params(
            basis,
            (0,),
            sigma_shell=jnp.asarray([1.0]),
            phase_shift_angles_shell=jnp.asarray([0.15]),
        ),
        types.make_radial_quadrature_spec(),
        types.make_final_state_spec(),
        experiment,
        types.make_self_energy_model(gamma=0.035),
        kgrid,
        jnp.linspace(-0.24, 0.08, 33),
        calibration,
        detector_effects,
        k_chunk=13,
        energy_chunk=11,
        checkpoint=True,
    )
    return detector


def figure_detector_counts(
    file_name: str = "pipeline-detector-counts.png",
) -> None:
    """Draw the expected-count raster and one Poisson acquisition.

    Parameters
    ----------
    file_name : str
        Output file name inside the figure directory.
    """
    detector: types.DetectorRaster = detector_example()
    observed: jnp.ndarray = jnp.asarray(
        simul.sample_poisson_counts(
            jax.random.key(20260813), detector.expected_counts
        )[0],
        dtype=jnp.float64,
    )
    figure, axes_row = plt.subplots(
        1, 3, figsize=(12.4, 3.6), constrained_layout=True
    )
    plot_detector_energy_cut(
        detector,
        ax=axes_row[0],
        log_counts=False,
        colorbar_label="events",
        title="expected counts, central $v$ row",
    )
    plot_detector_image(
        detector,
        ax=axes_row[1],
        colorbar_label="events",
        title="expected counts, energy sum",
    )
    plot_detector_image(
        observed,
        ax=axes_row[2],
        colorbar_label="events",
        title="one Poisson acquisition",
    )
    figure.savefig(FIGURE_DIRECTORY / file_name, **SAVE_OPTIONS)
    plt.close(figure)


def figure_pytree_update() -> None:
    """Draw spectra from a carrier update through equinox.tree_at."""
    model: types.TBModel = build_graphene()
    softened: types.TBModel = equinox.tree_at(
        lambda tree: tree.hopping_amplitudes,
        model,
        0.6 * model.hopping_amplitudes,
    )
    anchors: jnp.ndarray = jnp.asarray(
        [
            [0.0, 0.0, 0.0],
            [2.0 / 3.0, 1.0 / 3.0, 0.0],
            [0.5, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ]
    )
    path: types.KPath = tightb.build_kpath(
        anchors,
        model.geometry,
        n_per_segment=161,
        labels=("$\\Gamma$", "K", "M", "$\\Gamma$"),
    )
    distance: jnp.ndarray = tightb.kpath_arc_length(path, model.geometry)
    energy_axis: jnp.ndarray = jnp.linspace(-8.6, 0.5, 241)
    intensities: list = []
    for tree in (model, softened):
        eigenvalues: jnp.ndarray = tightb.eigvalsh_bands(tree, path.kpoints)
        intensities.append(
            intrinsic_spectrum(
                eigenvalues, jnp.ones_like(eigenvalues), energy_axis
            )
        )
    ticks: tuple = tuple(
        float(distance[index]) for index in path.label_indices
    )
    figure, axes_row = plt.subplots(
        1, 2, figsize=(10.4, 3.9), sharey=True, constrained_layout=True
    )
    plot_spectral_cut_series(
        tuple(intensities),
        distance,
        energy_axis,
        titles=(r"$t = -2.7$ eV", r"$t = -1.62$ eV via tree_at"),
        axes=tuple(axes_row),
        cmap="inferno",
        fermi_line=False,
        xlabel="",
        ylabel=r"$E-E_F$ (eV)",
    )
    for axes in axes_row:
        axes.set_xticks(ticks, path.labels)
    figure.savefig(
        FIGURE_DIRECTORY / "pytree-hopping-update.png", **SAVE_OPTIONS
    )
    plt.close(figure)


def figure_gradient_map() -> None:
    """Draw the linewidth sensitivity map for the gradients guide."""
    model: types.TBModel = build_graphene()
    anchors: jnp.ndarray = jnp.asarray(
        [
            [0.0, 0.0, 0.0],
            [2.0 / 3.0, 1.0 / 3.0, 0.0],
            [0.5, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ]
    )
    path: types.KPath = tightb.build_kpath(
        anchors,
        model.geometry,
        n_per_segment=161,
        labels=("$\\Gamma$", "K", "M", "$\\Gamma$"),
    )
    distance: jnp.ndarray = tightb.kpath_arc_length(path, model.geometry)
    eigenvalues: jnp.ndarray = tightb.eigvalsh_bands(model, path.kpoints)
    energy_axis: jnp.ndarray = jnp.linspace(-8.6, 0.5, 241)
    weights: jnp.ndarray = jnp.ones_like(eigenvalues)

    def spectrum_of_gamma(gamma: jnp.ndarray) -> jnp.ndarray:
        self_energy: types.SelfEnergyModel = types.make_self_energy_model(
            gamma=gamma
        )
        weights_ke: jnp.ndarray = jnp.broadcast_to(
            weights[:, None, :],
            (weights.shape[0], energy_axis.shape[0], weights.shape[1]),
        )
        return simul.assemble_spectral_intensity_bands_chunk(
            eigenvalues,
            weights_ke,
            energy_axis,
            self_energy,
            jnp.asarray(GRAPHENE_FERMI_EV),
            TEMPERATURE_K,
        )

    sensitivity: jnp.ndarray = jax.jacfwd(spectrum_of_gamma)(
        jnp.asarray(GRAPHENE_GAMMA_EV)
    )
    figure, axes = plt.subplots(figsize=(6.4, 4.2))
    plot_difference_map(
        sensitivity,
        distance,
        energy_axis,
        ax=axes,
        scale_percentile=99.5,
        colorbar_label=r"$\partial I/\partial\Gamma$",
        ylabel=r"$E-E_F$ (eV)",
        title=r"$\partial I(k,\omega)/\partial\Gamma$ through jax.jacfwd",
    )
    ticks: tuple = tuple(
        float(distance[index]) for index in path.label_indices
    )
    axes.set_xticks(ticks, path.labels)
    figure.savefig(FIGURE_DIRECTORY / "jax-gradient-map.png", **SAVE_OPTIONS)
    plt.close(figure)


def figure_band_carrier_spectrum() -> None:
    """Draw parsed-style bands beside their broadened spectrum."""
    n_k: int = 241
    path_coordinate: jnp.ndarray = jnp.linspace(-1.0, 1.0, n_k)
    eigenvalues: jnp.ndarray = jnp.stack(
        (
            -2.1 + 1.3 * jnp.cos(jnp.pi * path_coordinate),
            -1.15 + 0.65 * jnp.cos(2.0 * jnp.pi * path_coordinate),
            0.3 - 0.9 * jnp.cos(jnp.pi * path_coordinate),
        ),
        axis=1,
    )
    kpoints: jnp.ndarray = jnp.stack(
        (
            path_coordinate,
            jnp.zeros_like(path_coordinate),
            jnp.zeros_like(path_coordinate),
        ),
        axis=-1,
    )
    bands: types.BandStructure = types.make_band_structure(
        eigenvalues, kpoints, fermi_energy=0.0
    )
    energy_axis: jnp.ndarray = jnp.linspace(-3.6, 0.4, 221)
    intensity: jnp.ndarray = intrinsic_spectrum(
        bands.eigenvalues,
        jnp.ones_like(bands.eigenvalues),
        energy_axis,
        gamma_ev=0.08,
        fermi_ev=float(bands.fermi_energy),
    )
    figure, axes_row = plt.subplots(
        1, 2, figsize=(10.4, 3.9), sharey=True, constrained_layout=True
    )
    plot_band_dispersion(
        bands,
        momentum_axis=path_coordinate,
        ax=axes_row[0],
        xlabel="fractional path coordinate",
        title="BandStructure eigenvalues",
    )
    plot_spectral_cut(
        intensity,
        path_coordinate,
        energy_axis,
        ax=axes_row[1],
        cmap="inferno",
        fermi_line=False,
        xlabel="fractional path coordinate",
        ylabel="",
        title="occupied spectrum from the same carrier",
    )
    figure.savefig(FIGURE_DIRECTORY / "bands-to-spectrum.png", **SAVE_OPTIONS)
    plt.close(figure)


def main() -> None:
    """Generate every guide figure."""
    FIGURE_DIRECTORY.mkdir(parents=True, exist_ok=True)
    steps: tuple = (
        figure_pipeline_cube,
        figure_pipeline_ek_and_map,
        figure_detector_counts,
        figure_geometry,
        figure_kz_guide,
        figure_hv_scan,
        figure_kz_broadening_compare,
        figure_matrix_elements,
        figure_broadening,
        figure_pytree_update,
        figure_gradient_map,
        figure_band_carrier_spectrum,
    )
    for step in steps:
        print(f"generating {step.__name__} ...", flush=True)
        step()
    print(f"figures written to {FIGURE_DIRECTORY}")


if __name__ == "__main__":
    main()
