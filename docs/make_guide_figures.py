"""Generate the committed spectra figures for the documentation guides.

Extended Summary
----------------
Every figure under ``docs/source/guides/figures`` comes from this script and
the public diffpes API. The sources are a graphene tight-binding model with
Dirac cones, the square-lattice model from the simulation guide, and the
compact bulk model from the photon-energy tutorial. Rerun the script after an
API change that alters any depicted spectrum.

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


def spectrum_extent(
    distance: jnp.ndarray, energy_axis: jnp.ndarray
) -> tuple[float, float, float, float]:
    """Return the imshow extent for one path spectrum.

    Parameters
    ----------
    distance : jnp.ndarray
        Path coordinate axis.
    energy_axis : jnp.ndarray
        Energy axis in eV.

    Returns
    -------
    extent : tuple[float, float, float, float]
        Axis limits in imshow order.
    """
    extent: tuple[float, float, float, float] = (
        float(distance[0]),
        float(distance[-1]),
        float(energy_axis[0]),
        float(energy_axis[-1]),
    )
    return extent


def figure_pipeline_cube() -> None:
    """Draw the three-face spectral cube for the simulation guide.

    The cube corner sits on one K point, so the front and side faces are
    momentum-energy slices through Dirac cones while the top face is a
    constant-energy map. Each face is normalized to its own maximum for
    display.
    """
    model: types.TBModel = build_graphene()
    corner_x: float = -4.0 * jnp.pi / (3.0 * 2.46)
    corner_y: float = -2.0 * jnp.pi / (jnp.sqrt(3.0) * 2.46)
    n_k: int = 161
    axis_x: jnp.ndarray = jnp.linspace(corner_x, -corner_x + 0.4, n_k)
    axis_y: jnp.ndarray = jnp.linspace(corner_y, -corner_y + 0.4, n_k)
    mesh_x, mesh_y = jnp.meshgrid(axis_x, axis_y, indexing="ij")
    kpoints_cart: jnp.ndarray = jnp.stack(
        (mesh_x, mesh_y, jnp.zeros_like(mesh_x)), axis=-1
    ).reshape((-1, 3))
    kpoints_frac: jnp.ndarray = tightb.kpoints_cart_to_frac(
        kpoints_cart, model.geometry
    )
    eigenvalues: jnp.ndarray = tightb.eigvalsh_bands(model, kpoints_frac)
    energy_axis: jnp.ndarray = jnp.linspace(-3.5, -0.1, 121)
    cube: jnp.ndarray = intrinsic_spectrum(
        eigenvalues,
        jnp.ones_like(eigenvalues),
        energy_axis,
        gamma_ev=0.09,
    ).reshape((n_k, n_k, -1))
    top_index: int = int(energy_axis.shape[0] - 1)
    faces: tuple = (
        cube[:, :, top_index],
        cube[:, 0, :],
        cube[0, :, :],
    )
    normalized: tuple = tuple(
        jnp.asarray(face) / float(face.max()) for face in faces
    )
    mesh_a, mesh_b = jnp.meshgrid(axis_x, axis_y, indexing="ij")
    mesh_kxe, mesh_exk = jnp.meshgrid(axis_x, energy_axis, indexing="ij")
    mesh_kye, mesh_eyk = jnp.meshgrid(axis_y, energy_axis, indexing="ij")
    colormap = plt.get_cmap("inferno")
    figure = plt.figure(figsize=(7.2, 6.4))
    axes3d = figure.add_subplot(projection="3d", computed_zorder=False)
    axes3d.plot_surface(
        mesh_kxe,
        jnp.full_like(mesh_kxe, float(axis_y[0])),
        mesh_exk,
        facecolors=colormap(normalized[1]),
        rstride=1,
        cstride=1,
        shade=False,
    )
    axes3d.plot_surface(
        jnp.full_like(mesh_kye, float(axis_x[0])),
        mesh_kye,
        mesh_eyk,
        facecolors=colormap(normalized[2]),
        rstride=1,
        cstride=1,
        shade=False,
    )
    axes3d.plot_surface(
        mesh_a,
        mesh_b,
        jnp.full_like(mesh_a, float(energy_axis[top_index])),
        facecolors=colormap(normalized[0]),
        rstride=1,
        cstride=1,
        shade=False,
    )
    axes3d.set_xlabel(r"$k_x$ ($\mathrm{\AA}^{-1}$)")
    axes3d.set_ylabel(r"$k_y$ ($\mathrm{\AA}^{-1}$)")
    axes3d.set_zlabel(r"$E-E_F$ (eV)", labelpad=12.0)
    axes3d.set_title("graphene spectral cube $I(k_x, k_y, E)$")
    axes3d.set_box_aspect((1.0, 1.0, 0.85))
    axes3d.view_init(elev=20.0, azim=-55.0)
    figure.savefig(FIGURE_DIRECTORY / "pipeline-cube.png", dpi=200)
    plt.close(figure)


def figure_pipeline_ek_and_map() -> None:
    """Draw the path spectrum and the Fermi-surface map."""
    distance, energy_axis, intensity, ticks, labels = graphene_path_spectrum()
    figure, axes = plt.subplots(figsize=(6.4, 4.2))
    image = axes.imshow(
        intensity.T,
        origin="lower",
        aspect="auto",
        extent=spectrum_extent(distance, energy_axis),
        cmap="inferno",
    )
    axes.set_xticks(ticks, labels)
    axes.set_ylabel(r"$E-E_F$ (eV)")
    axes.set_title(r"graphene $\pi$ bands: occupied spectral intensity")
    figure.colorbar(image, ax=axes, label="intensity (1/eV)")
    figure.savefig(FIGURE_DIRECTORY / "pipeline-ek-cut.png", **SAVE_OPTIONS)
    plt.close(figure)

    axis, cube_energy_axis, cube = graphene_cube()
    fermi_index: int = int(jnp.argmin(jnp.abs(cube_energy_axis)))
    figure, axes = plt.subplots(figsize=(5.4, 4.6))
    image = axes.imshow(
        cube[:, :, fermi_index].T,
        origin="lower",
        extent=(float(axis[0]), float(axis[-1])) * 2,
        cmap="inferno",
    )
    axes.set_xlabel(r"$k_x$ ($\mathrm{\AA}^{-1}$)")
    axes.set_ylabel(r"$k_y$ ($\mathrm{\AA}^{-1}$)")
    axes.set_title(r"Fermi-surface map at $E_F$")
    figure.colorbar(image, ax=axes, label="intensity (1/eV)")
    figure.savefig(
        FIGURE_DIRECTORY / "pipeline-fermi-map.png", **SAVE_OPTIONS
    )
    plt.close(figure)


def figure_geometry() -> None:
    """Draw kinematics curves and constant-energy slices."""
    k_parallel: jnp.ndarray = jnp.asarray([0.0, 0.6, 1.2, 1.8])
    photon_energies: jnp.ndarray = jnp.linspace(15.0, 120.0, 400)
    figure, axes = plt.subplots(figsize=(6.2, 3.8))
    for k_par in k_parallel:
        kz_values, propagating = jax.vmap(
            lambda hv, kp=k_par: simul.kz_from_inner_potential(
                hv, 4.5, 12.0, jnp.asarray(0.0), jnp.asarray(kp)
            )
        )(photon_energies)
        kz_real: jnp.ndarray = jnp.where(
            propagating, jnp.real(kz_values), jnp.nan
        )
        axes.plot(
            photon_energies,
            kz_real,
            label=rf"$k_\parallel = {float(k_par):.1f}$",
        )
    axes.set_xlabel(r"photon energy $h\nu$ (eV)")
    axes.set_ylabel(r"$k_z$ ($\mathrm{\AA}^{-1}$)")
    axes.set_title(r"free-electron $k_z(h\nu)$ with $V_0 = 12$ eV")
    axes.legend(title=r"$\mathrm{\AA}^{-1}$")
    figure.savefig(FIGURE_DIRECTORY / "geometry-kz-hv.png", **SAVE_OPTIONS)
    plt.close(figure)

    axis, energy_axis, cube = graphene_cube()
    slice_energies: tuple = (0.0, -0.8, -1.6)
    figure, axes_row = plt.subplots(
        1, 3, figsize=(11.4, 3.9), sharey=True, constrained_layout=True
    )
    for target, axes in zip(slice_energies, axes_row, strict=True):
        index: int = int(jnp.argmin(jnp.abs(energy_axis - target)))
        image = axes.imshow(
            cube[:, :, index].T,
            origin="lower",
            extent=(float(axis[0]), float(axis[-1])) * 2,
            cmap="inferno",
        )
        axes.set_xlabel(r"$k_x$ ($\mathrm{\AA}^{-1}$)")
        axes.set_title(rf"$E-E_F = {target:.1f}$ eV")
    axes_row[0].set_ylabel(r"$k_y$ ($\mathrm{\AA}^{-1}$)")
    figure.colorbar(image, ax=axes_row, label="intensity (1/eV)", shrink=0.86)
    figure.savefig(
        FIGURE_DIRECTORY / "geometry-energy-slices.png", **SAVE_OPTIONS
    )
    plt.close(figure)


def figure_kz_guide() -> None:
    """Draw wrapped escape-depth weights and a photon-energy scan."""
    edges: jnp.ndarray = jnp.linspace(-0.5, 0.5, 257)
    nodes: jnp.ndarray = simul.kz_fractional_nodes(256)
    reciprocal_period: jnp.ndarray = jnp.asarray(2.0 * jnp.pi / 3.2)
    mean_free_paths: jnp.ndarray = jnp.asarray([4.0, 7.5, 15.0])
    weights: jnp.ndarray = jax.vmap(
        simul.kz_wrapped_lorentzian_bin_weights,
        in_axes=(None, None, 0, None),
    )(edges, jnp.asarray(0.0), mean_free_paths, reciprocal_period)
    figure, axes = plt.subplots(figsize=(6.2, 3.8))
    for path_length, values in zip(mean_free_paths, weights, strict=True):
        axes.plot(
            nodes,
            values * nodes.shape[0],
            label=rf"$\lambda = {float(path_length):.1f}$ $\mathrm{{\AA}}$",
        )
    axes.set_xlabel(r"surface-fractional $k_z$")
    axes.set_ylabel("wrapped Lorentzian density")
    axes.set_title("escape depth controls the $k_z$ width")
    axes.legend()
    figure.savefig(
        FIGURE_DIRECTORY / "kz-wrapped-weights.png", **SAVE_OPTIONS
    )
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
    image = axes.imshow(
        normal_emission.T,
        origin="lower",
        aspect="auto",
        extent=(
            float(photon_energies[0]),
            float(photon_energies[-1]),
            float(energy_axis[0]),
            float(energy_axis[-1]),
        ),
        cmap="magma",
    )
    axes.set_xlabel(r"photon energy $h\nu$ (eV)")
    axes.set_ylabel(r"$E-E_F$ (eV)")
    axes.set_title("normal-emission photon-energy scan, bulk $k_z$ mode")
    figure.colorbar(image, ax=axes, label="intensity")
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
        maps.append(scan[0])
    path_coordinate: jnp.ndarray = carriers["path_coordinate"]
    figure, axes_row = plt.subplots(
        1, 2, figsize=(9.6, 3.9), sharey=True, constrained_layout=True
    )
    for values, (_, _, title), axes in zip(
        maps, modes, axes_row, strict=True
    ):
        image = axes.imshow(
            values.T / float(values.max()),
            origin="lower",
            aspect="auto",
            extent=(
                float(path_coordinate[0]),
                float(path_coordinate[-1]),
                float(energy_axis[0]),
                float(energy_axis[-1]),
            ),
            cmap="magma",
        )
        axes.set_xlabel("fractional path coordinate")
        axes.set_title(title)
    axes_row[0].set_ylabel(r"$E-E_F$ (eV)")
    figure.colorbar(
        image, ax=axes_row, label="normalized intensity", shrink=0.86
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
    bands: types.DiagonalizedBands = tightb.diagonalize_tb(
        model, path.kpoints
    )
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
    figure, axes_row = plt.subplots(
        1, 2, figsize=(10.4, 3.9), sharey=True, constrained_layout=True
    )
    for (polarization, label), axes in zip(
        polarizations, axes_row, strict=True
    ):
        distance, eigenvalues, weights, ticks, labels = (
            polarized_band_weights(polarization)
        )
        intensity: jnp.ndarray = intrinsic_spectrum(
            eigenvalues, weights, energy_axis
        )
        image = axes.imshow(
            intensity.T,
            origin="lower",
            aspect="auto",
            extent=spectrum_extent(distance, energy_axis),
            cmap="inferno",
        )
        axes.set_xticks(ticks, labels)
        axes.set_title(label)
    axes_row[0].set_ylabel(r"$E-E_F$ (eV)")
    figure.colorbar(
        image, ax=axes_row, label="intensity (1/eV)", shrink=0.86
    )
    figure.savefig(
        FIGURE_DIRECTORY / "matrixel-polarization.png", **SAVE_OPTIONS
    )
    plt.close(figure)

    figure, axes = plt.subplots(figsize=(6.2, 3.8))
    for atomic_number, n_quantum, l_quantum, label in (
        (29, 3, 2, "Cu 3d"),
        (6, 2, 1, "C 2p"),
        (83, 6, 1, "Bi 6p"),
    ):
        energy, sigma = simul.yeh_lindau_cross_section_table(
            atomic_number, n_quantum, l_quantum
        )[:2]
        axes.plot(energy, sigma, label=label)
    axes.set_xlabel("photon energy (eV)")
    axes.set_ylabel("cross section (Mb)")
    axes.set_xscale("log")
    axes.set_yscale("log")
    axes.set_title("Yeh--Lindau atomic photoionization cross sections")
    axes.legend()
    figure.savefig(
        FIGURE_DIRECTORY / "matrixel-cross-sections.png", **SAVE_OPTIONS
    )
    plt.close(figure)


def figure_broadening() -> None:
    """Draw lineshape profiles and self-energy comparisons."""
    energy_axis: jnp.ndarray = jnp.linspace(-0.5, 0.5, 601)
    figure, axes = plt.subplots(figsize=(6.2, 3.8))
    axes.plot(
        energy_axis,
        simul.gaussian(energy_axis, 0.0, 0.05),
        label=r"Gaussian $\sigma = 50$ meV",
    )
    axes.plot(
        energy_axis,
        simul.voigt(energy_axis, 0.0, 0.05, 0.05),
        label=r"Voigt $\sigma = \gamma = 50$ meV",
    )
    axes.plot(
        energy_axis,
        simul.voigt(energy_axis, 0.0, 0.008, 0.1),
        label=r"near-Lorentzian $\gamma = 100$ meV",
    )
    axes.set_xlabel(r"$E - E_0$ (eV)")
    axes.set_ylabel("normalized profile (1/eV)")
    axes.set_title("energy lineshapes")
    axes.legend()
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
    for gamma, axes in zip(gammas, axes_row, strict=True):
        intensity: jnp.ndarray = intrinsic_spectrum(
            eigenvalues, weights, energy_axis, gamma_ev=gamma
        )
        image = axes.imshow(
            intensity.T,
            origin="lower",
            aspect="auto",
            extent=spectrum_extent(distance, energy_axis),
            cmap="inferno",
        )
        axes.set_xticks(ticks, labels)
        axes.set_title(rf"$\Gamma = {1000 * gamma:.0f}$ meV")
    axes_row[0].set_ylabel(r"$E-E_F$ (eV)")
    figure.colorbar(
        image, ax=axes_row, label="intensity (1/eV)", shrink=0.86
    )
    figure.savefig(
        FIGURE_DIRECTORY / "broadening-linewidth-ek.png", **SAVE_OPTIONS
    )
    plt.close(figure)

    narrow: jnp.ndarray = intrinsic_spectrum(
        eigenvalues, weights, energy_axis, gamma_ev=0.03
    )
    convolved: jnp.ndarray = simul.convolve_energy(
        narrow, energy_axis, 0.12
    )
    figure, axes_row = plt.subplots(
        1, 2, figsize=(10.4, 3.9), sharey=True, constrained_layout=True
    )
    for values, title, axes in zip(
        (narrow, convolved),
        ("intrinsic", "with 120 meV Gaussian resolution"),
        axes_row,
        strict=True,
    ):
        image = axes.imshow(
            values.T,
            origin="lower",
            aspect="auto",
            extent=spectrum_extent(distance, energy_axis),
            cmap="inferno",
        )
        axes.set_xticks(ticks, labels)
        axes.set_title(title)
    axes_row[0].set_ylabel(r"$E-E_F$ (eV)")
    figure.colorbar(
        image, ax=axes_row, label="intensity (1/eV)", shrink=0.86
    )
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
    kgrid: types.KGrid = types.make_kgrid(
        kpoints, mesh_shape=(13, 13), kz=0.0
    )
    hamiltonians: jnp.ndarray = tightb.bloch_hamiltonian_batch(
        model, kpoints
    )
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
    expected: jnp.ndarray = detector.expected_counts[0]
    observed: jnp.ndarray = simul.sample_poisson_counts(
        jax.random.key(20260813), detector.expected_counts
    )[0]
    figure, axes_row = plt.subplots(
        1, 3, figsize=(12.4, 3.6), constrained_layout=True
    )
    image = axes_row[0].imshow(
        expected[:, expected.shape[1] // 2, :].T,
        origin="lower",
        aspect="auto",
        cmap="magma",
    )
    axes_row[0].set_xlabel("detector $u$ bin")
    axes_row[0].set_ylabel("detector energy bin")
    axes_row[0].set_title("expected counts, central $v$ row")
    figure.colorbar(image, ax=axes_row[0], label="events")
    image = axes_row[1].imshow(
        expected.sum(axis=-1).T, origin="lower", cmap="magma"
    )
    axes_row[1].set_xlabel("detector $u$ bin")
    axes_row[1].set_ylabel("detector $v$ bin")
    axes_row[1].set_title("expected counts, energy sum")
    figure.colorbar(image, ax=axes_row[1], label="events")
    image = axes_row[2].imshow(
        observed.sum(axis=-1).T, origin="lower", cmap="magma"
    )
    axes_row[2].set_xlabel("detector $u$ bin")
    axes_row[2].set_ylabel("detector $v$ bin")
    axes_row[2].set_title("one Poisson acquisition")
    figure.colorbar(image, ax=axes_row[2], label="events")
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
    figure, axes_row = plt.subplots(
        1, 2, figsize=(10.4, 3.9), sharey=True, constrained_layout=True
    )
    for tree, title, axes in zip(
        (model, softened),
        (r"$t = -2.7$ eV", r"$t = -1.62$ eV via tree_at"),
        axes_row,
        strict=True,
    ):
        eigenvalues: jnp.ndarray = tightb.eigvalsh_bands(
            tree, path.kpoints
        )
        intensity: jnp.ndarray = intrinsic_spectrum(
            eigenvalues, jnp.ones_like(eigenvalues), energy_axis
        )
        image = axes.imshow(
            intensity.T,
            origin="lower",
            aspect="auto",
            extent=spectrum_extent(distance, energy_axis),
            cmap="inferno",
        )
        ticks: tuple = tuple(
            float(distance[index]) for index in path.label_indices
        )
        axes.set_xticks(ticks, path.labels)
        axes.set_title(title)
    axes_row[0].set_ylabel(r"$E-E_F$ (eV)")
    figure.colorbar(
        image, ax=axes_row, label="intensity (1/eV)", shrink=0.86
    )
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
    bound: float = float(jnp.percentile(jnp.abs(sensitivity), 99.5))
    figure, axes = plt.subplots(figsize=(6.4, 4.2))
    image = axes.imshow(
        sensitivity.T,
        origin="lower",
        aspect="auto",
        extent=spectrum_extent(distance, energy_axis),
        cmap="RdBu_r",
        vmin=-bound,
        vmax=bound,
    )
    ticks: tuple = tuple(
        float(distance[index]) for index in path.label_indices
    )
    axes.set_xticks(ticks, path.labels)
    axes.set_ylabel(r"$E-E_F$ (eV)")
    axes.set_title(
        r"$\partial I(k,\omega)/\partial\Gamma$ through jax.jacfwd"
    )
    figure.colorbar(image, ax=axes, label=r"$\partial I/\partial\Gamma$")
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
    axes_row[0].plot(path_coordinate, bands.eigenvalues)
    axes_row[0].axhline(0.0, color="0.5", linewidth=0.8)
    axes_row[0].set_xlabel("fractional path coordinate")
    axes_row[0].set_ylabel(r"$E-E_F$ (eV)")
    axes_row[0].set_title("BandStructure eigenvalues")
    image = axes_row[1].imshow(
        intensity.T,
        origin="lower",
        aspect="auto",
        extent=(
            float(path_coordinate[0]),
            float(path_coordinate[-1]),
            float(energy_axis[0]),
            float(energy_axis[-1]),
        ),
        cmap="inferno",
    )
    axes_row[1].set_xlabel("fractional path coordinate")
    axes_row[1].set_title("occupied spectrum from the same carrier")
    figure.colorbar(image, ax=axes_row[1], label="intensity (1/eV)")
    figure.savefig(
        FIGURE_DIRECTORY / "bands-to-spectrum.png", **SAVE_OPTIONS
    )
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
