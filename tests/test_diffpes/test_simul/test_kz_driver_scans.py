"""Verify the bulk-kz driver modes and photon-energy scan surface.

The tests lock exact finite-energy kinematics and mutually exclusive carriers.
They cover native/coherent behavior, scan stacking, interpolation, and
registered derivatives for bulk-kz and photon-energy scans.
"""

import inspect
import math
from pathlib import Path

import chex
import equinox as eqx
import jax
import jax.numpy as jnp
import pytest
from beartype.typing import Any, Dict, List, Optional, Tuple
from jaxtyping import Array, Bool, Complex128, Float64

from diffpes import simul
from diffpes.matrixel import (
    assemble_orbital_transition_channels,
    transition_source,
)
from diffpes.simul import (
    _kz_spectrum,
    assemble_spectral_intensity_chunk,
    contract_experiment_polarization,
    final_state_k_inv_ang,
    kinetic_energy_ev,
    kz_broadening,
    kz_fractional_nodes,
    kz_from_inner_potential,
    spectrum,
)
from diffpes.simul._kz_spectrum import (
    _blockwise_exact_folded_center_and_mask,
    _bulk_domain_intensity,
    _bulk_orbital_positions_surface_cartesian,
    _bulk_source_parallel_cartesian,
    _exact_folded_surface_center,
)
from diffpes.simul._spectrum_stream import _stream_cartesian_intensity
from diffpes.simul.kz_broadening import (
    _kz_wrapped_lorentzian_bin_weight,
    _map_surface_fractional_to_bulk,
    _surface_kz_frame,
)
from diffpes.tightb import bloch_hamiltonian_batch
from diffpes.types import (
    ArpesSpectrum,
    DetectorCalibration,
    DetectorEffects,
    DetectorRaster,
    DiagonalizedBands,
    ExperimentGeometry,
    KGrid,
    KPath,
    SurfaceCell,
    TBModel,
    make_arpes_spectrum,
    make_crystal_geometry,
    make_detector_calibration,
    make_detector_effects,
    make_diagonalized_bands,
    make_experiment_geometry,
    make_final_state_spec,
    make_kgrid,
    make_kpath,
    make_matrix_element_params,
    make_orbital_basis,
    make_radial_quadrature_spec,
    make_radial_spec,
    make_self_energy_model,
    make_surface_cell,
    make_tb_model,
)
from tests._gradients import assert_grad_matches_fd


def _driver_fixture() -> Dict[str, object]:
    """PRIVATE: Build one dispersive identity-surface driver fixture.

    Returns
    -------
    fixture : Dict[str, object]
        Complete native, coherent-slab, and bulk scan inputs.

    Notes
    -----
    The z hopping makes both escape length and inner potential observable.
    Every path point lies in the identity surface plane.
    """
    lattice_scale: float = 3.2
    crystal: Any = make_crystal_geometry(
        lattice_scale * jnp.eye(3, dtype=jnp.float64),
        jnp.zeros((1, 3), dtype=jnp.float64),
        ("X",),
    )
    basis: Any = make_orbital_basis(
        atom_indices=(0,),
        n=(1,),
        l=(0,),
        m=(0,),
        labels=("1s",),
    )
    model: TBModel = make_tb_model(
        hopping_amplitudes=jnp.asarray((-0.38, -0.38), dtype=jnp.complex128),
        onsite_energies=jnp.asarray((-0.05,), dtype=jnp.float64),
        soc_lambdas=jnp.zeros((0,), dtype=jnp.float64),
        geometry=crystal,
        basis=basis,
        hopping_pairs=((0, 0), (0, 0)),
        hopping_cells=((0, 0, 1), (0, 0, -1)),
        shell_index=(-1,),
    )
    surface_cell: SurfaceCell = make_surface_cell(
        in_plane_vectors=lattice_scale
        * jnp.asarray(((1.0, 0.0, 0.0), (0.0, 1.0, 0.0))),
        stacking_vector=lattice_scale * jnp.asarray((0.0, 0.0, 1.0)),
        rotation=jnp.eye(3, dtype=jnp.float64),
        interlayer_spacing_ang=lattice_scale,
        miller=(0, 0, 1),
        in_plane_coeffs=((1, 0, 0), (0, 1, 0)),
        stacking_coeffs=(0, 0, 1),
    )
    path_points: Float64[Array, "2 3"] = jnp.asarray(
        ((-0.045, -0.025, 0.0), (0.018, 0.012, 0.0)),
        dtype=jnp.float64,
    )
    kpath: KPath = make_kpath(path_points, n_per_segment=1, kz=0.0)
    hamiltonian: Complex128[Array, "..."] = bloch_hamiltonian_batch(
        model, path_points
    )
    eigenvalues: Float64[Array, "..."] = jnp.real(hamiltonian[:, :1, 0])
    eigenvectors: Complex128[Array, "..."] = jnp.ones(
        (path_points.shape[0], 1, 1), dtype=jnp.complex128
    )
    bands: DiagonalizedBands = make_diagonalized_bands(
        eigenvalues,
        eigenvectors,
        path_points,
        crystal,
        basis,
        fermi_energy=0.0,
    )
    coherent_bands: DiagonalizedBands = make_diagonalized_bands(
        eigenvalues,
        eigenvectors,
        path_points,
        crystal,
        basis,
        fermi_energy=0.0,
        depths=jnp.asarray((0.7,), dtype=jnp.float64),
    )
    radial: Any = make_radial_spec(
        basis,
        (0,),
        mode="fixed",
        fixed_integrals_shell=jnp.asarray(((0.0, 1.0),)),
    )
    matrix_params: Any = make_matrix_element_params(
        basis,
        (0,),
        sigma_shell=jnp.asarray((1.13,)),
        phase_shift_angles_shell=jnp.asarray((0.21,)),
    )
    geometry: ExperimentGeometry = make_experiment_geometry(
        photon_energy_ev=28.0,
        polarization=jnp.asarray((1.0, 0.35j, 0.0), dtype=jnp.complex128),
        sample_azimuth=0.14,
        work_function_ev=4.3,
        inner_potential_ev=11.0,
        temperature_k=45.0,
        mean_free_path_ang=7.5,
    )
    fixture: Dict[str, object] = {
        "bands": bands,
        "coherent_bands": coherent_bands,
        "energy_axis": jnp.asarray((-0.38, -0.21, -0.06)),
        "final_state": make_final_state_spec(),
        "geometry": geometry,
        "hamiltonian": hamiltonian,
        "kpath": kpath,
        "lattice_scale": lattice_scale,
        "matrix_params": matrix_params,
        "model": model,
        "photon_energies": jnp.asarray((25.5, 28.0, 31.5)),
        "quadrature": make_radial_quadrature_spec(),
        "radial": radial,
        "self_energy": make_self_energy_model(gamma=0.055),
        "surface_cell": surface_cell,
    }
    return fixture


def _geometry_at_hv(
    geometry: ExperimentGeometry,
    photon_energy_ev: Float64[Array, "..."],
) -> ExperimentGeometry:
    """PRIVATE: Replace one geometry photon-energy leaf.

    Parameters
    ----------
    geometry : ExperimentGeometry
        Template experiment geometry.
    photon_energy_ev : Float64[Array, "..."]
        Scalar photon energy in eV.

    Returns
    -------
    updated : ExperimentGeometry
        Geometry with every other field unchanged.
    """
    updated: ExperimentGeometry = eqx.tree_at(
        lambda item: item.photon_energy_ev,
        geometry,
        photon_energy_ev,
    )
    return updated


def _sample_intensity(  # noqa: PLR0913
    fixture: Dict[str, object],
    bands: DiagonalizedBands,
    *,
    geometry: ExperimentGeometry,
    hamiltonian: Float64[Array, "..."],
    omega: Float64[Array, "..."],
    final_momentum: Float64[Array, "..."],
    valid: Float64[Array, "..."],
) -> Float64[Array, "..."]:
    """PRIVATE: Assemble one independent sampled-energy reference column.

    Parameters
    ----------
    fixture : Dict[str, object]
        Shared physical inputs.
    bands : DiagonalizedBands
        Native or depth-bearing source metadata.
    geometry : ExperimentGeometry
        Geometry at the current photon energy.
    hamiltonian : Float64[Array, "..."]
        Absolute-energy Hamiltonian at this sampled center.
    omega : Float64[Array, "..."]
        Scalar sampled energy relative to the Fermi level.
    final_momentum : Float64[Array, "..."]
        Exact final momentum for every path point.
    valid : Float64[Array, "..."]
        Propagating mask for every path point.

    Returns
    -------
    intensity : Float64[Array, "..."]
        Occupied intrinsic intensity for every path point.

    Notes
    -----
    Public matrix-element and spectral primitives form this independent
    reference.
    """
    channels: Any = assemble_orbital_transition_channels(
        bands,
        fixture["radial"],
        fixture["matrix_params"],
        fixture["quadrature"],
        fixture["final_state"],
        geometry,
        final_momentum,
        valid,
    )
    rows: Float64[Array, "..."] = contract_experiment_polarization(
        channels, geometry
    )
    sources: Complex128[Array, "..."] = transition_source(rows)[:, None, :, :]
    sampled: Float64[Array, "..."] = assemble_spectral_intensity_chunk(
        hamiltonian,
        sources,
        omega[None],
        fixture["self_energy"],
        bands.fermi_energy,
        geometry.temperature_k,
        1.0e-4,
    )
    intensity: Float64[Array, "..."] = sampled[:, 0]
    return intensity


def _inner_potential_reference(
    fixture: Dict[str, object],
    photon_energy_ev: Float64[Array, "..."],
    *,
    mode: str,
    center_energy_axis: Optional[Float64[Array, "..."]] = None,
) -> Float64[Array, "..."]:
    """PRIVATE: Derive one exact inner-potential spectrum independently.

    Parameters
    ----------
    fixture : Dict[str, object]
        Shared bulk and coherent inputs.
    photon_energy_ev : Float64[Array, "..."]
        Scalar photon energy in eV.
    mode : str
        ``bulk_direct`` or ``coherent_slab``.
    center_energy_axis : Optional[Float64[Array, "..."]]
        Optional planted center schedule for a negative control.

    Returns
    -------
    intensity : Float64[Array, "..."]
        Reference intensity with shape ``(n_k, n_e)``.

    Raises
    ------
    ValueError
        If the requested reference mode has no contract.

    Notes
    -----
    The bulk branch folds the exact center before Hamiltonian evaluation.
    """
    if mode not in {"bulk_direct", "coherent_slab"}:
        raise ValueError("reference mode must be bulk_direct or coherent_slab")
    geometry: ExperimentGeometry = _geometry_at_hv(
        fixture["geometry"], photon_energy_ev
    )
    energy_axis: Float64[Array, "..."] = fixture["energy_axis"]
    center_axis: Float64[Array, "..."] = (
        energy_axis if center_energy_axis is None else center_energy_axis
    )
    bands: DiagonalizedBands = (
        fixture["bands"]
        if mode == "bulk_direct"
        else fixture["coherent_bands"]
    )
    k_parallel: Float64[Array, "..."] = (
        bands.kpoints @ bands.geometry.reciprocal
    )
    k_parallel_norm: Float64[Array, "..."] = jnp.linalg.norm(
        k_parallel[:, :2], axis=-1
    )
    columns: List[Float64[Array, "..."]] = []
    energy_index: int
    for energy_index in range(energy_axis.shape[0]):
        omega: Float64[Array, "..."] = energy_axis[energy_index]
        center_omega: Float64[Array, "..."] = center_axis[energy_index]
        kz_complex: Float64[Array, "..."]
        propagating: Float64[Array, "..."]
        kz_complex, propagating = kz_from_inner_potential(
            geometry.photon_energy_ev,
            geometry.work_function_ev,
            geometry.inner_potential_ev,
            jnp.broadcast_to(center_omega, k_parallel_norm.shape),
            k_parallel_norm,
        )
        kz_real: Float64[Array, "..."] = jnp.real(kz_complex)
        final_momentum: Float64[Array, "..."] = k_parallel.at[:, 2].set(
            kz_real
        )
        if mode == "bulk_direct":
            u_unfolded: Float64[Array, "..."] = (
                kz_real * fixture["lattice_scale"] / (2.0 * math.pi)
            )
            u_folded: Float64[Array, "..."] = u_unfolded - jnp.floor(
                u_unfolded + 0.5
            )
            mapped_points: Float64[Array, "..."] = bands.kpoints.at[:, 2].set(
                u_folded
            )
            hamiltonian: Complex128[Array, "..."] = bloch_hamiltonian_batch(
                fixture["model"], mapped_points
            )
        else:
            hamiltonian = fixture["hamiltonian"]
        column: Float64[Array, "..."] = _sample_intensity(
            fixture,
            bands,
            geometry=geometry,
            hamiltonian=hamiltonian,
            omega=omega,
            final_momentum=final_momentum,
            valid=propagating,
        )
        columns.append(column)
    intensity: Float64[Array, "..."] = jnp.stack(columns, axis=-1)
    return intensity


def _vacuum_reference(
    fixture: Dict[str, object],
    photon_energy_ev: Float64[Array, "..."],
) -> Float64[Array, "..."]:
    """PRIVATE: Derive the retained native-direct spectrum independently.

    Parameters
    ----------
    fixture : Dict[str, object]
        Shared native source inputs.
    photon_energy_ev : Float64[Array, "..."]
        Scalar photon energy in eV.

    Returns
    -------
    intensity : Float64[Array, "..."]
        Native vacuum-final-state intensity with shape ``(n_k, n_e)``.

    Notes
    -----
    The reference retains the exact 08a vacuum-momentum convention.
    """
    geometry: ExperimentGeometry = _geometry_at_hv(
        fixture["geometry"], photon_energy_ev
    )
    bands: DiagonalizedBands = fixture["bands"]
    energy_axis: Float64[Array, "..."] = fixture["energy_axis"]
    k_cart: Float64[Array, "..."] = bands.kpoints @ bands.geometry.reciprocal
    k_parallel_squared: Float64[Array, "..."] = jnp.sum(
        k_cart[:, :2] ** 2, axis=-1
    )
    columns: List[Float64[Array, "..."]] = []
    omega: Float64[Array, "..."]
    for omega in energy_axis:
        kinetic: Float64[Array, "..."]
        energy_valid: Bool[Array, "..."]
        kinetic, energy_valid = kinetic_energy_ev(
            geometry.photon_energy_ev,
            geometry.work_function_ev,
            omega,
        )
        final_norm: Float64[Array, "..."]
        momentum_valid: Bool[Array, "..."]
        final_norm, momentum_valid = final_state_k_inv_ang(kinetic)
        normal_squared: Float64[Array, "..."] = (
            final_norm**2 - k_parallel_squared
        )
        propagating: Float64[Array, "..."] = jnp.broadcast_to(
            energy_valid & momentum_valid, normal_squared.shape
        ) & (normal_squared > 0.0)
        final_kz: Float64[Array, "..."] = jnp.sqrt(
            jnp.where(propagating, normal_squared, 1.0)
        )
        final_momentum: Float64[Array, "..."] = k_cart.at[:, 2].set(final_kz)
        column: Float64[Array, "..."] = _sample_intensity(
            fixture,
            bands,
            geometry=geometry,
            hamiltonian=fixture["hamiltonian"],
            omega=omega,
            final_momentum=final_momentum,
            valid=propagating,
        )
        columns.append(column)
    intensity: Float64[Array, "..."] = jnp.stack(columns, axis=-1)
    return intensity


def _simulate_scan(  # noqa: PLR0913
    fixture: Dict[str, object],
    photon_energies_ev: Float64[Array, "..."],
    *,
    mode: str,
    geometry: Optional[ExperimentGeometry] = None,
    energy_axis: Optional[Float64[Array, "..."]] = None,
    kz_nodes_frac: Optional[Float64[Array, "..."]] = None,
    checkpoint: bool = False,
) -> Float64[Array, "..."]:
    """PRIVATE: Return one registered scan route with canonical carriers.

    Parameters
    ----------
    fixture : Dict[str, object]
        Shared native, coherent, and bulk inputs.
    photon_energies_ev : Float64[Array, "..."]
        Caller-owned photon-energy samples.
    mode : str
        Registered out-of-plane mode.
    geometry : Optional[ExperimentGeometry]
        Optional geometry override.
    energy_axis : Optional[Float64[Array, "..."]]
        Optional sampled-energy override.
    kz_nodes_frac : Optional[Float64[Array, "..."]]
        Required registered nodes for ``bulk_kz``.
    checkpoint : bool
        Static rematerialization selector.

    Returns
    -------
    scan : Float64[Array, "..."]
        Pre-detector scan with axes ``(hnu, k, energy)``.
    """
    resolved_geometry: ExperimentGeometry = (
        fixture["geometry"] if geometry is None else geometry
    )
    resolved_energy_axis: Float64[Array, "..."] = (
        fixture["energy_axis"] if energy_axis is None else energy_axis
    )
    hamiltonian: Any = None
    bands: Any = None
    bulk_model: Any = None
    surface_cell: Any = None
    if mode == "native_direct":
        hamiltonian = fixture["hamiltonian"]
        bands = fixture["bands"]
    elif mode == "coherent_slab":
        hamiltonian = fixture["hamiltonian"]
        bands = fixture["coherent_bands"]
        surface_cell = fixture["surface_cell"]
    elif mode in {"bulk_direct", "bulk_kz"}:
        bulk_model = fixture["model"]
        surface_cell = fixture["surface_cell"]
    scan: Float64[Array, "..."] = spectrum.simulate_hv_scan(
        hamiltonian,
        bands,
        fixture["radial"],
        fixture["matrix_params"],
        fixture["quadrature"],
        fixture["final_state"],
        resolved_geometry,
        fixture["self_energy"],
        fixture["kpath"],
        resolved_energy_axis,
        photon_energies_ev,
        1.0e-4,
        k_chunk=2,
        energy_chunk=2,
        checkpoint=checkpoint,
        bulk_model=bulk_model,
        surface_cell=surface_cell,
        kz_nodes_frac=kz_nodes_frac,
        kz_mode=mode,
    )
    return scan


def _single_domain_detector_context(
    *, raster: bool = False
) -> Tuple[DetectorCalibration, DetectorEffects]:
    """PRIVATE: Build explicit one-domain detector inputs.

    Parameters
    ----------
    raster : bool, optional
        Give the detector two native ``v`` bins instead of the path cut's
        one-bin slit. Default is ``False``.

    Returns
    -------
    calibration : DetectorCalibration
        Native detector axes, resolution widths, and transmission domain.
    detector_effects : DetectorEffects
        Deterministic one-domain nuisance parameters.

    Notes
    -----
    Both paths use nontrivial resolution, transmission, exposure, and
    background so the tests exercise the complete detector composition.
    """
    v_edges: Float64[Array, "..."] = (
        jnp.asarray((-0.05, 0.0, 0.05), dtype=jnp.float64)
        if raster
        else jnp.asarray((-0.05, 0.05), dtype=jnp.float64)
    )
    calibration: DetectorCalibration = make_detector_calibration(
        u_bin_edges=jnp.asarray((-0.05, -0.012, 0.025)),
        v_bin_edges=v_edges,
        energy_bin_edges_ev=jnp.asarray((-0.34, -0.2, -0.08)),
        psf_fwhm_u=0.008,
        psf_fwhm_v=0.01,
        psf_fwhm_energy_ev=0.018,
        transmission_reference_domain_ev=jnp.asarray((20.0, 32.0)),
    )
    detector_effects: DetectorEffects = make_detector_effects(
        domain_logits=jnp.asarray((0.0,)),
        domain_euler_angles_rad=jnp.zeros((1, 3)),
        transmission_raw_slopes=jnp.asarray((-0.3, 0.18)),
        background_coefficients=jnp.asarray((-2.1,)),
        sensitivity_coefficients=jnp.asarray(()),
        exposure=1.7,
        background_mode="flat",
        sensitivity_mode="constant",
        domain_frame_ids=("org.diffpes.frame.sample_cartesian",),
    )
    returned: Tuple[DetectorCalibration, DetectorEffects] = (
        calibration,
        detector_effects,
    )
    return returned


def _bulk_kgrid(fixture: Dict[str, object]) -> KGrid:
    """PRIVATE: Expand the path coordinates into a separable 2x2 bulk grid.

    Parameters
    ----------
    fixture : Dict[str, object]
        Shared identity-frame bulk driver inputs.

    Returns
    -------
    kgrid : KGrid
        Fractional bulk points whose physical in-plane image is separable.

    Notes
    -----
    The public bulk driver applies the model reciprocal lattice and surface
    rotation before it derives source axes; the fixture makes both maps exact.
    """
    path_points: Float64[Array, "..."] = fixture["kpath"].kpoints
    mesh_x: Float64[Array, "..."]
    mesh_y: Float64[Array, "..."]
    mesh_x, mesh_y = jnp.meshgrid(
        path_points[:, 0], path_points[:, 1], indexing="xy"
    )
    points: Float64[Array, "..."] = jnp.stack(
        (mesh_x, mesh_y, jnp.zeros_like(mesh_x)), axis=-1
    ).reshape((-1, 3))
    kgrid: KGrid = make_kgrid(points, mesh_shape=(2, 2), kz=0.0)
    return kgrid


def _full_detector_bulk_kz_counts(  # noqa: PLR0913
    fixture: Dict[str, object],
    coordinates: Float64[Array, "..."],
    calibration: DetectorCalibration,
    detector_effects: DetectorEffects,
    nodes: Float64[Array, "..."],
) -> Float64[Array, "..."]:
    """PRIVATE: Evaluate one canonical bulk-kz cut at four coordinates.

    Parameters
    ----------
    fixture : Dict[str, object]
        Shared bulk driver inputs.
    coordinates : Float64[Array, "..."]
        Mean free path, inner potential, work function, and energy translation.
    calibration : DetectorCalibration
        Explicit native slit calibration.
    detector_effects : DetectorEffects
        Explicit one-domain detector nuisance parameters.
    nodes : Float64[Array, "..."]
        Caller-owned registered fractional quadrature nodes.

    Returns
    -------
    counts : Float64[Array, "..."]
        Complete native detector expected-count raster.

    Notes
    -----
    Eight nodes are an explicitly caller-recalibrated reduced diagnostic. The
    helper makes no claim about the separately certified quadrature profile.
    """
    geometry: ExperimentGeometry = eqx.tree_at(
        lambda item: item.mean_free_path_ang,
        fixture["geometry"],
        coordinates[0],
    )
    geometry = eqx.tree_at(
        lambda item: item.inner_potential_ev,
        geometry,
        coordinates[1],
    )
    geometry = eqx.tree_at(
        lambda item: item.work_function_ev,
        geometry,
        coordinates[2],
    )
    energy_axis: Float64[Array, "..."] = (
        fixture["energy_axis"] + coordinates[3]
    )
    raster: DetectorRaster = spectrum.simulate_arpes_cut(
        (),
        (),
        fixture["radial"],
        fixture["matrix_params"],
        fixture["quadrature"],
        fixture["final_state"],
        geometry,
        fixture["self_energy"],
        fixture["kpath"],
        energy_axis,
        calibration,
        detector_effects,
        1.0e-4,
        k_chunk=2,
        energy_chunk=2,
        checkpoint=False,
        bulk_models_by_domain=(fixture["model"],),
        surface_cells_by_domain=(fixture["surface_cell"],),
        kz_nodes_frac=nodes,
        kz_mode="bulk_kz",
    )
    counts: Float64[Array, "..."] = raster.expected_counts
    return counts


def _eager_bulk_kz_reference(  # noqa: PLR0913
    fixture: Dict[str, object],
    model: TBModel,
    source_kpoints: Float64[Array, "..."],
    nodes: Float64[Array, "..."],
    *,
    k_chunk: int,
) -> Float64[Array, "..."]:
    """PRIVATE: Build the former full-k finite-width node loop.

    Parameters
    ----------
    fixture : Dict[str, object]
        Shared physical and numerical driver inputs.
    model : TBModel
        Candidate differentiable bulk model.
    source_kpoints : Float64[Array, "..."]
        Caller-owned bulk-fractional source points.
    nodes : Float64[Array, "..."]
        Registered uniform finite-width quadrature nodes.
    k_chunk : int
        Static spectral k-point chunk size.

    Returns
    -------
    intensity : Float64[Array, "..."]
        Eager full-k reference intensity with shape ``(n_k, n_e)``.

    Notes
    -----
    This reference intentionally materializes each node's complete Hamiltonian
    before calling the established spectral streamer. It is independent of the
    production outer k-block map and freezes the pre-optimization evaluation
    order for every physical point.
    """
    surface_cell: SurfaceCell = fixture["surface_cell"]
    geometry: ExperimentGeometry = fixture["geometry"]
    energy_axis: Float64[Array, "..."] = fixture["energy_axis"]
    k_parallel: Float64[Array, "..."] = _bulk_source_parallel_cartesian(
        source_kpoints,
        model,
        surface_cell,
    )
    center_folded: Float64[Array, "..."]
    propagating: Float64[Array, "..."]
    center_folded, propagating, _, _ = (  # noqa: SLF001
        _exact_folded_surface_center(
            k_parallel,
            energy_axis,
            geometry,
            surface_cell,
            model.geometry,
        )
    )
    positions_surface: Float64[Array, "..."] = (  # noqa: SLF001
        _bulk_orbital_positions_surface_cartesian(
            model,
            surface_cell,
        )
    )
    period_inv_ang: Float64[Array, "..."] = _surface_kz_frame(
        surface_cell,
        model.geometry,
    )[3]
    edges: Float64[Array, "..."] = jnp.linspace(
        -0.5,
        0.5,
        nodes.shape[0] + 1,
        dtype=jnp.float64,
    )
    zero_depths: Float64[Array, "..."] = jnp.zeros(
        (len(model.basis.n),),
        dtype=jnp.float64,
    )
    intensity: Float64[Array, "..."] = jnp.zeros(
        (source_kpoints.shape[0], energy_axis.shape[0]),
        dtype=jnp.float64,
    )
    node_index: int
    for node_index in range(nodes.shape[0]):
        folded_nodes: Float64[Array, "..."] = jnp.broadcast_to(
            nodes[node_index],
            (source_kpoints.shape[0],),
        )
        surface_points: Float64[Array, "..."]
        bulk_points: Float64[Array, "..."]
        surface_points, bulk_points = (  # noqa: SLF001
            _map_surface_fractional_to_bulk(
                k_parallel,
                folded_nodes,
                surface_cell,
                model.geometry,
            )
        )
        hamiltonians: Complex128[Array, "..."] = bloch_hamiltonian_batch(
            model, bulk_points
        )
        node_intensity: Float64[Array, "..."] = (  # noqa: SLF001
            _stream_cartesian_intensity(
                hamiltonians,
                surface_points,
                model.basis,
                positions_surface,
                zero_depths,
                jnp.asarray(0.0, dtype=jnp.float64),
                energy_axis,
                fixture["radial"],
                fixture["matrix_params"],
                fixture["quadrature"],
                fixture["final_state"],
                geometry,
                fixture["self_energy"],
                1.0e-4,
                k_chunk=k_chunk,
                energy_chunk=2,
                checkpoint=False,
                use_inner_potential=True,
            )
        )
        weight: Float64[Array, "..."] = (  # noqa: SLF001
            _kz_wrapped_lorentzian_bin_weight(
                edges[node_index],
                edges[node_index + 1],
                center_folded,
                geometry.mean_free_path_ang,
                period_inv_ang,
            )
        )
        contribution: Float64[Array, "..."] = jnp.where(
            propagating,
            node_intensity * weight,
            0.0,
        )
        intensity = intensity + contribution
    return intensity


class TestBulkKzKBlockStreaming:
    """Verify the allocation-bounded finite-width k-block implementation.

    The cases compare block masks, values, and reverse gradients with a full
    eager reference for centered and off-center nodes.
    """

    @pytest.mark.parametrize(
        "n_k",
        [2, 3],
        ids=("divisible", "nondivisible"),
    )
    @pytest.mark.big_mem
    @pytest.mark.rss_limit_mb(800)
    def test_center_mask_values_and_reverse_gradients_match_full_reference(
        self,
        n_k: int,
    ) -> None:
        """Preserve exact kinematics across finite-width block padding.

        Compare the production two-output K-block map with the independent
        complete-K helper.

        Notes
        -----
        The odd case exercises the padded final block and its crop; the reverse
        derivative traverses the exact complex root.
        """
        fixture: Dict[str, object] = _driver_fixture()
        fraction: Float64[Array, "..."] = jnp.linspace(
            0.0,
            1.0,
            n_k,
            dtype=jnp.float64,
        )
        source_kpoints: Float64[Array, "..."] = jnp.stack(
            (
                -0.045 + 0.063 * fraction,
                -0.025 + 0.037 * fraction,
                jnp.zeros_like(fraction),
            ),
            axis=-1,
        )
        k_parallel: Float64[Array, "..."] = (  # noqa: SLF001
            _bulk_source_parallel_cartesian(
                source_kpoints,
                fixture["model"],
                fixture["surface_cell"],
            )
        )
        k_chunk: int = 2
        padded_k: int = ((n_k + k_chunk - 1) // k_chunk) * k_chunk
        padded_parallel: Float64[Array, "..."] = jnp.pad(
            k_parallel,
            ((0, padded_k - n_k), (0, 0)),
        )
        k_parallel_blocks: Float64[Array, "..."] = jnp.reshape(
            padded_parallel,
            (-1, k_chunk, 3),
        )
        direct_surface: Float64[Array, "..."]
        normal_hat: Float64[Array, "..."]
        direct_surface, _, normal_hat, _ = _surface_kz_frame(
            fixture["surface_cell"],
            fixture["model"].geometry,
        )

        def blockwise(
            candidate: Float64[Array, "..."],
        ) -> Tuple[Float64[Array, "..."], Float64[Array, "..."]]:
            """Return production blockwise centres and their mask."""
            geometry: ExperimentGeometry = eqx.tree_at(
                lambda item: item.inner_potential_ev,
                fixture["geometry"],
                candidate,
            )
            result: Tuple[Float64[Array, "..."], Float64[Array, "..."]] = (  # noqa: SLF001
                _blockwise_exact_folded_center_and_mask(
                    k_parallel_blocks,
                    n_k,
                    fixture["energy_axis"],
                    geometry,
                    direct_surface,
                    normal_hat,
                )
            )
            return result

        def complete(
            candidate: Float64[Array, "..."],
        ) -> Tuple[Float64[Array, "..."], Float64[Array, "..."]]:
            """Return the independent complete-K centres and their mask."""
            geometry: ExperimentGeometry = eqx.tree_at(
                lambda item: item.inner_potential_ev,
                fixture["geometry"],
                candidate,
            )
            center: Float64[Array, "..."]
            mask: Bool[Array, "..."]
            center, mask, _, _ = (  # noqa: SLF001
                _exact_folded_surface_center(
                    k_parallel,
                    fixture["energy_axis"],
                    geometry,
                    fixture["surface_cell"],
                    fixture["model"].geometry,
                )
            )
            returned: Tuple[Float64[Array, "..."], Float64[Array, "..."]] = (
                center,
                mask,
            )
            return returned

        objective_weights: Float64[Array, "..."] = jnp.arange(
            1,
            n_k * fixture["energy_axis"].shape[0] + 1,
            dtype=jnp.float64,
        ).reshape((n_k, fixture["energy_axis"].shape[0]))

        def blockwise_loss(
            candidate: Float64[Array, "..."],
        ) -> Float64[Array, "..."]:
            """Return a weighted sum of production blockwise centres."""
            center: Float64[Array, "..."]
            center, _ = blockwise(candidate)
            returned: Float64[Array, "..."] = jnp.sum(
                center * objective_weights
            )
            return returned

        def complete_loss(
            candidate: Float64[Array, "..."],
        ) -> Float64[Array, "..."]:
            """Return a weighted sum of independent complete-K centres."""
            center: Float64[Array, "..."]
            center, _ = complete(candidate)
            returned: Float64[Array, "..."] = jnp.sum(
                center * objective_weights
            )
            return returned

        candidate: Float64[Array, "..."] = fixture[
            "geometry"
        ].inner_potential_ev
        actual_center: Float64[Array, "..."]
        actual_mask: Bool[Array, "..."]
        expected_center: Float64[Array, "..."]
        expected_mask: Bool[Array, "..."]
        actual_center, actual_mask = blockwise(candidate)
        expected_center, expected_mask = complete(candidate)
        actual_gradient: Float64[Array, "..."] = jax.grad(blockwise_loss)(
            candidate
        )
        expected_gradient: Float64[Array, "..."] = jax.grad(complete_loss)(
            candidate
        )

        chex.assert_trees_all_close(
            actual_center,
            expected_center,
            rtol=1.0e-12,
            atol=1.0e-14,
        )
        chex.assert_trees_all_equal(actual_mask, expected_mask)
        chex.assert_trees_all_close(
            actual_gradient,
            expected_gradient,
            rtol=1.0e-12,
            atol=1.0e-14,
        )

    @pytest.mark.slow
    @pytest.mark.parametrize(
        "n_k",
        [2, 3],
        ids=("divisible", "nondivisible"),
    )
    @pytest.mark.big_mem
    @pytest.mark.rss_limit_mb(1800)
    def test_values_and_gradients_match_eager_full_k_reference(
        self,
        n_k: int,
    ) -> None:
        """Preserve values and model gradients across final-block padding.

        The eager control builds one complete Hamiltonian per quadrature node.
        Compare it with production on divisible and one-point padded K axes.
        Production constructs Hamiltonians only inside live k blocks.

        Notes
        -----
        Differentiate a weighted intensity sum with respect to the on-site
        energy so the comparison traverses the complete spectral kernel.
        """
        fixture: Dict[str, object] = _driver_fixture()
        fraction: Float64[Array, "..."] = jnp.linspace(
            0.0,
            1.0,
            n_k,
            dtype=jnp.float64,
        )
        source_kpoints: Float64[Array, "..."] = jnp.stack(
            (
                -0.045 + 0.063 * fraction,
                -0.025 + 0.037 * fraction,
                jnp.zeros_like(fraction),
            ),
            axis=-1,
        )
        nodes: Float64[Array, "..."] = kz_fractional_nodes(4)
        k_chunk: int = 2
        objective_weights: Float64[Array, "..."] = jnp.arange(
            1,
            n_k * fixture["energy_axis"].shape[0] + 1,
            dtype=jnp.float64,
        ).reshape((n_k, fixture["energy_axis"].shape[0]))

        def streamed(
            candidate: Float64[Array, "..."],
        ) -> Float64[Array, "..."]:
            """Evaluate the allocation-bounded production path."""
            candidate_model: TBModel = eqx.tree_at(
                lambda item: item.onsite_energies,
                fixture["model"],
                jnp.reshape(candidate, (1,)),
            )
            intensity: Float64[Array, "..."]
            intensity, _ = _bulk_domain_intensity(
                candidate_model,
                fixture["surface_cell"],
                source_kpoints,
                fixture["energy_axis"],
                fixture["radial"],
                fixture["matrix_params"],
                fixture["quadrature"],
                fixture["final_state"],
                fixture["geometry"],
                fixture["self_energy"],
                1.0e-4,
                nodes,
                "bulk_kz",
                k_chunk=k_chunk,
                energy_chunk=2,
                checkpoint=False,
            )
            return intensity

        def eager(candidate: Float64[Array, "..."]) -> Float64[Array, "..."]:
            """Evaluate the independent full-k node-loop control."""
            candidate_model: TBModel = eqx.tree_at(
                lambda item: item.onsite_energies,
                fixture["model"],
                jnp.reshape(candidate, (1,)),
            )
            intensity: Float64[Array, "..."] = _eager_bulk_kz_reference(
                fixture,
                candidate_model,
                source_kpoints,
                nodes,
                k_chunk=k_chunk,
            )
            return intensity

        def streamed_loss(
            candidate: Float64[Array, "..."],
        ) -> Float64[Array, "..."]:
            """Return a weighted production intensity sum."""
            value: Float64[Array, "..."] = jnp.sum(
                streamed(candidate) * objective_weights
            )
            return value

        def eager_loss(
            candidate: Float64[Array, "..."],
        ) -> Float64[Array, "..."]:
            """Return the weighted eager intensity sum."""
            value: Float64[Array, "..."] = jnp.sum(
                eager(candidate) * objective_weights
            )
            return value

        candidate: Float64[Array, "..."] = fixture["model"].onsite_energies[0]
        actual: Float64[Array, "..."] = streamed(candidate)
        expected: Float64[Array, "..."] = eager(candidate)
        actual_gradient: Float64[Array, "..."] = jax.grad(streamed_loss)(
            candidate
        )
        expected_gradient: Float64[Array, "..."] = jax.grad(eager_loss)(
            candidate
        )

        chex.assert_trees_all_close(
            actual, expected, rtol=1.0e-12, atol=1.0e-14
        )
        chex.assert_trees_all_close(
            actual_gradient,
            expected_gradient,
            rtol=1.0e-11,
            atol=1.0e-13,
        )


class TestKzDriverPublicSurface:
    """Verify the owned bulk-kz API and mutually exclusive carriers.

    The cases inspect the public signatures and reject mixed source modes,
    invalid node combinations, and inconsistent bulk-model tuples.
    """

    def test_signatures_preserve_detector_arguments_and_own_bulk_inputs(
        self,
    ) -> None:
        """Keep one canonical driver and separate scan carrier arguments.

        The test locks every positional 08a input and every new keyword input.
        It also rejects an unowned dense bulk-band carrier across the tree.

        Notes
        -----
        Inspect signatures, exports, aliases, and Python source text directly.
        """
        arpes_signature: inspect.Signature = inspect.signature(
            spectrum.simulate_arpes
        )
        arpes_names: Tuple[str, ...] = tuple(arpes_signature.parameters)
        cut_signature: inspect.Signature = inspect.signature(
            spectrum.simulate_arpes_cut
        )
        cut_names: Tuple[str, ...] = tuple(cut_signature.parameters)
        expected_arpes: Tuple[str, ...] = (
            "hamiltonians_by_domain",
            "bands_by_domain",
            "radial_spec",
            "matrix_element_params",
            "radial_quadrature",
            "final_state",
            "geometry",
            "self_energy",
            "kgrid",
            "energy_axis",
            "detector_calibration",
            "detector_effects",
            "eta",
            "k_chunk",
            "energy_chunk",
            "checkpoint",
            "bulk_models_by_domain",
            "surface_cells_by_domain",
            "kz_nodes_frac",
            "kz_mode",
        )
        expected_cut: Tuple[str, ...] = (
            "hamiltonians_by_domain",
            "bands_by_domain",
            "radial_spec",
            "matrix_element_params",
            "radial_quadrature",
            "final_state",
            "geometry",
            "self_energy",
            "kpath",
            "energy_axis",
            "detector_calibration",
            "detector_effects",
            "eta",
            "k_chunk",
            "energy_chunk",
            "checkpoint",
            "bulk_models_by_domain",
            "surface_cells_by_domain",
            "kz_nodes_frac",
            "kz_mode",
        )
        scan_signature: inspect.Signature = inspect.signature(
            spectrum.simulate_hv_scan
        )
        scan_names: Tuple[str, ...] = tuple(scan_signature.parameters)
        expected_scan: Tuple[str, ...] = (
            "hamiltonian",
            "bands",
            "radial_spec",
            "matrix_element_params",
            "radial_quadrature",
            "final_state",
            "geometry",
            "self_energy",
            "kpath",
            "energy_axis",
            "photon_energies_ev",
            "eta",
            "k_chunk",
            "energy_chunk",
            "checkpoint",
            "bulk_model",
            "surface_cell",
            "kz_nodes_frac",
            "kz_mode",
        )
        keyword_names: Tuple[str, ...] = expected_scan[12:]
        keyword_name: str
        for keyword_name in keyword_names:
            parameter: inspect.Parameter = scan_signature.parameters[
                keyword_name
            ]
            assert parameter.kind is inspect.Parameter.KEYWORD_ONLY
        driver_signature: inspect.Signature
        driver_keyword: str
        for driver_signature in (arpes_signature, cut_signature):
            for driver_keyword in expected_arpes[13:]:
                parameter = driver_signature.parameters[driver_keyword]
                assert parameter.kind is inspect.Parameter.KEYWORD_ONLY
        forbidden_carrier: str = "".join(("Bulk", "Band", "Tensor"))
        repository_root: Path = Path(__file__).resolve().parents[3]
        source_paths: List[Path] = list(
            (repository_root / "src").rglob("*.py")
        ) + list((repository_root / "tests").rglob("*.py"))
        violations: List[Path] = [
            path
            for path in source_paths
            if forbidden_carrier in path.read_text(encoding="utf-8")
        ]

        assert arpes_names == expected_arpes
        assert cut_names == expected_cut
        assert scan_names == expected_scan
        assert simul.simulate_hv_scan is spectrum.simulate_hv_scan
        assert simul.hv_map_at_energy is spectrum.hv_map_at_energy
        assert not hasattr(simul, "simulate_arpes_kz")
        assert violations == []

    def test_rejects_mixed_mode_and_node_combinations(self) -> None:
        """Reject every planted carrier mixture before physical evaluation.

        Native rejects surface metadata, bulk direct rejects nodes, and bulk
        kz rejects both native carriers and a literal one-node quadrature.

        Notes
        -----
        Call the public eager boundary with one defect in each invocation.
        """
        fixture: Dict[str, object] = _driver_fixture()
        nodes: Float64[Array, "..."] = kz_fractional_nodes(4)
        common: Tuple[object, ...] = (
            fixture["radial"],
            fixture["matrix_params"],
            fixture["quadrature"],
            fixture["final_state"],
            fixture["geometry"],
            fixture["self_energy"],
            fixture["kpath"],
            fixture["energy_axis"],
            fixture["photon_energies"][:1],
        )
        with pytest.raises(ValueError, match="native_direct rejects"):
            spectrum.simulate_hv_scan(
                fixture["hamiltonian"],
                fixture["bands"],
                *common,
                surface_cell=fixture["surface_cell"],
                kz_mode="native_direct",
            )
        with pytest.raises(ValueError, match="bulk_direct rejects"):
            spectrum.simulate_hv_scan(
                None,
                None,
                *common,
                bulk_model=fixture["model"],
                surface_cell=fixture["surface_cell"],
                kz_nodes_frac=nodes,
                kz_mode="bulk_direct",
            )
        with pytest.raises(ValueError, match="at least two"):
            spectrum.simulate_hv_scan(
                None,
                None,
                *common,
                bulk_model=fixture["model"],
                surface_cell=fixture["surface_cell"],
                kz_nodes_frac=jnp.asarray((0.0,)),
                kz_mode="bulk_kz",
            )
        with pytest.raises(ValueError, match="empty native"):
            spectrum.simulate_hv_scan(
                fixture["hamiltonian"],
                fixture["bands"],
                *common,
                bulk_model=fixture["model"],
                surface_cell=fixture["surface_cell"],
                kz_nodes_frac=nodes,
                kz_mode="bulk_kz",
            )
        with pytest.raises(ValueError, match="depth-bearing"):
            spectrum.simulate_hv_scan(
                fixture["hamiltonian"],
                fixture["bands"],
                *common,
                surface_cell=fixture["surface_cell"],
                kz_mode="coherent_slab",
            )

    def test_rejects_bulk_model_surface_tuple_length_mismatch(self) -> None:
        """Reject unequal public bulk-domain carrier tuples at the boundary.

        A downstream zip cannot truncate the planted two-model, one-surface
        input or silently interpret it as a one-domain acquisition.

        Notes
        -----
        Invoke the canonical path driver so validation precedes all physics.
        """
        fixture: Dict[str, object] = _driver_fixture()
        calibration: DetectorCalibration
        detector_effects: DetectorEffects
        calibration, detector_effects = _single_domain_detector_context()
        with pytest.raises(ValueError, match="equal nonempty"):
            spectrum.simulate_arpes_cut(
                (),
                (),
                fixture["radial"],
                fixture["matrix_params"],
                fixture["quadrature"],
                fixture["final_state"],
                fixture["geometry"],
                fixture["self_energy"],
                fixture["kpath"],
                fixture["energy_axis"],
                calibration,
                detector_effects,
                1.0e-4,
                k_chunk=2,
                energy_chunk=2,
                checkpoint=False,
                bulk_models_by_domain=(fixture["model"], fixture["model"]),
                surface_cells_by_domain=(fixture["surface_cell"],),
                kz_mode="bulk_direct",
            )


class TestDirectAndCoherentModes:
    """Verify exact centers and preserve both explicit-H source routes.

    The cases check off-grid centers, direct and coherent sources, exact
    nonpropagating zeros, and the direct route's independence from wrapped
    weights.
    """

    @pytest.mark.slow
    @pytest.mark.big_mem
    @pytest.mark.rss_limit_mb(1800)
    def test_bulk_direct_matches_exact_off_grid_centers(self) -> None:
        """Evaluate bulk Hamiltonians at every exact finite-energy center.

        The independent reference folds each off-grid center before evaluating
        the dispersive model. A planted Fermi-level broadcast changes all bins.

        Notes
        -----
        Compare production with public kinematic and spectral primitives.
        """
        fixture: Dict[str, object] = _driver_fixture()
        photon_energy: Float64[Array, "..."] = fixture["photon_energies"][1]
        actual: Float64[Array, "..."] = _simulate_scan(
            fixture,
            photon_energy[None],
            mode="bulk_direct",
        )[0]
        expected: Float64[Array, "..."] = _inner_potential_reference(
            fixture,
            photon_energy,
            mode="bulk_direct",
        )
        at_fermi: Float64[Array, "..."] = _inner_potential_reference(
            fixture,
            photon_energy,
            mode="bulk_direct",
            center_energy_axis=jnp.zeros_like(fixture["energy_axis"]),
        )
        energy_index: int
        for energy_index in range(fixture["energy_axis"].shape[0]):
            assert not jnp.allclose(
                expected[:, energy_index],
                at_fermi[:, energy_index],
                rtol=1.0e-10,
                atol=1.0e-13,
            )
        assert jnp.allclose(actual, expected, rtol=1.0e-10, atol=1.0e-13)

    @pytest.mark.slow
    @pytest.mark.big_mem
    @pytest.mark.rss_limit_mb(1800)
    def test_native_default_and_coherent_exact_paths_are_preserved(
        self,
    ) -> None:
        """Preserve native vacuum values and coherent inner-potential values.

        Omitted and explicit native modes stay bitwise equal. The coherent
        depth path matches an independently assembled finite-energy reference.

        Notes
        -----
        Compare both explicit-H modes without invoking any kz quadrature.
        """
        fixture: Dict[str, object] = _driver_fixture()
        photon_energy: Float64[Array, "..."] = fixture["photon_energies"][1]
        explicit_native: Float64[Array, "..."] = _simulate_scan(
            fixture,
            photon_energy[None],
            mode="native_direct",
        )[0]
        default_native: Float64[Array, "..."] = spectrum.simulate_hv_scan(
            fixture["hamiltonian"],
            fixture["bands"],
            fixture["radial"],
            fixture["matrix_params"],
            fixture["quadrature"],
            fixture["final_state"],
            fixture["geometry"],
            fixture["self_energy"],
            fixture["kpath"],
            fixture["energy_axis"],
            photon_energy[None],
            k_chunk=2,
            energy_chunk=2,
            checkpoint=False,
        )[0]
        native_reference: Float64[Array, "..."] = _vacuum_reference(
            fixture, photon_energy
        )
        coherent: Float64[Array, "..."] = _simulate_scan(
            fixture,
            photon_energy[None],
            mode="coherent_slab",
        )[0]
        coherent_reference: Float64[Array, "..."] = _inner_potential_reference(
            fixture,
            photon_energy,
            mode="coherent_slab",
        )

        assert jnp.array_equal(default_native, explicit_native)
        assert jnp.allclose(
            explicit_native, native_reference, rtol=1.0e-12, atol=1.0e-14
        )
        assert jnp.allclose(
            coherent, coherent_reference, rtol=1.0e-10, atol=1.0e-13
        )

    @pytest.mark.slow
    @pytest.mark.big_mem
    @pytest.mark.rss_limit_mb(2600)
    def test_nonpropagating_bulk_and_coherent_are_exact_zero(self) -> None:
        """Keep every forbidden bulk and coherent channel at exact zero.

        A photon energy below the work-function threshold invalidates all
        energy samples. Direct, finite-width, and coherent public modes must
        retain the validity mask instead of evaluating a safe sentinel state.

        Notes
        -----
        Eight bulk-kz nodes are a caller-recalibrated reduced diagnostic and
        carry no quadrature-accuracy claim.
        """
        fixture: Dict[str, object] = _driver_fixture()
        geometry: ExperimentGeometry = _geometry_at_hv(
            fixture["geometry"], jnp.asarray(3.0, dtype=jnp.float64)
        )
        nodes: Float64[Array, "..."] = kz_fractional_nodes(8)
        direct: Float64[Array, "..."] = _simulate_scan(
            fixture,
            geometry.photon_energy_ev[None],
            mode="bulk_direct",
            geometry=geometry,
        )
        finite_width: Float64[Array, "..."] = _simulate_scan(
            fixture,
            geometry.photon_energy_ev[None],
            mode="bulk_kz",
            geometry=geometry,
            kz_nodes_frac=nodes,
        )
        coherent: Float64[Array, "..."] = _simulate_scan(
            fixture,
            geometry.photon_energy_ev[None],
            mode="coherent_slab",
            geometry=geometry,
        )
        values: Float64[Array, "..."]
        for values in (direct, finite_width, coherent):
            assert jnp.array_equal(values, jnp.zeros_like(values))

    @pytest.mark.big_mem
    @pytest.mark.rss_limit_mb(1800)
    def test_bulk_direct_never_calls_wrapped_kz_weights(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Keep the zero-width direct route independent of wrapped weights.

        Counting failures replace the scalar integrator and public dense weight
        helper. A successful nonzero public direct scan proves that the direct
        route dispatches neither finite-width implementation.

        Notes
        -----
        Patch only the weight seams and call the exported photon-energy scan.
        """
        fixture: Dict[str, object] = _driver_fixture()
        calls: Dict[str, int] = {"count": 0}

        def count_and_fail(
            *arguments: Any,
            **keywords: Any,
        ) -> Float64[Array, "..."]:
            """Record an invalid direct-route weight call and fail."""
            del arguments, keywords
            calls["count"] += 1
            raise AssertionError("bulk_direct called a wrapped kz weight")

        monkeypatch.setattr(
            _kz_spectrum,
            "_kz_wrapped_lorentzian_bin_weight",
            count_and_fail,
        )
        monkeypatch.setattr(
            kz_broadening,
            "kz_wrapped_lorentzian_bin_weights",
            count_and_fail,
        )
        scan: Float64[Array, "..."] = _simulate_scan(
            fixture,
            fixture["photon_energies"][1:2],
            mode="bulk_direct",
        )
        assert calls["count"] == 0
        assert jnp.all(jnp.isfinite(scan))
        assert jnp.any(scan > 0.0)


class TestCanonicalBulkDomainComposition:
    """Verify the public two-domain bulk source-to-detector seam.

    The case composes the public bulk scans with detector effects and compares
    the resulting cut with an explicit two-domain construction.
    """

    @pytest.mark.slow
    @pytest.mark.big_mem
    @pytest.mark.rss_limit_mb(3200)
    def test_cut_matches_public_scans_then_detector_effects(self) -> None:
        """Compose both bulk domains only after their detector mappings.

        Independent single-domain scans become public source carriers before
        the shared detector chain applies unequal rotations and softmax
        weights.

        Notes
        -----
        Compare the complete canonical raster with the public staged result.
        """
        fixture: Dict[str, object] = _driver_fixture()
        first_model: TBModel = fixture["model"]
        second_model: TBModel = eqx.tree_at(
            lambda item: item.onsite_energies,
            first_model,
            first_model.onsite_energies + 0.16,
        )
        models: Tuple[TBModel, ...] = (first_model, second_model)
        surface_cells: Tuple[SurfaceCell, ...] = (
            fixture["surface_cell"],
            fixture["surface_cell"],
        )
        calibration: DetectorCalibration = make_detector_calibration(
            u_bin_edges=jnp.asarray((-0.05, -0.012, 0.025)),
            v_bin_edges=jnp.asarray((-0.05, 0.05)),
            energy_bin_edges_ev=jnp.asarray((-0.34, -0.2, -0.08)),
            psf_fwhm_u=0.008,
            psf_fwhm_v=0.01,
            psf_fwhm_energy_ev=0.018,
            transmission_reference_domain_ev=jnp.asarray((20.0, 30.0)),
        )
        detector_effects: DetectorEffects = make_detector_effects(
            domain_logits=jnp.asarray((-0.45, 0.65)),
            domain_euler_angles_rad=jnp.asarray(
                ((0.0, 0.0, 0.0), (0.035, 0.0, 0.0))
            ),
            transmission_raw_slopes=jnp.asarray((-0.3, 0.18)),
            background_coefficients=jnp.asarray((-2.1,)),
            sensitivity_coefficients=jnp.asarray(()),
            exposure=1.7,
            background_mode="flat",
            sensitivity_mode="constant",
            domain_frame_ids=(
                "org.diffpes.frame.sample_cartesian",
                "org.diffpes.frame.sample_cartesian",
            ),
        )
        actual: DetectorRaster = spectrum.simulate_arpes_cut(
            (),
            (),
            fixture["radial"],
            fixture["matrix_params"],
            fixture["quadrature"],
            fixture["final_state"],
            fixture["geometry"],
            fixture["self_energy"],
            fixture["kpath"],
            fixture["energy_axis"],
            calibration,
            detector_effects,
            1.0e-4,
            k_chunk=2,
            energy_chunk=2,
            checkpoint=False,
            bulk_models_by_domain=models,
            surface_cells_by_domain=surface_cells,
            kz_mode="bulk_direct",
        )
        photon_energy: Float64[Array, "..."] = fixture[
            "geometry"
        ].photon_energy_ev[None]
        source_points: Float64[Array, "..."] = (
            fixture["kpath"].kpoints @ first_model.geometry.reciprocal
        )
        step_lengths: Float64[Array, "..."] = jnp.linalg.norm(
            jnp.diff(source_points, axis=0), axis=-1
        )
        k_axis: Float64[Array, "..."] = jnp.concatenate(
            (jnp.zeros((1,), dtype=jnp.float64), jnp.cumsum(step_lengths))
        )
        sources: List[ArpesSpectrum] = []
        model: TBModel
        for model in models:
            scan: Float64[Array, "..."] = spectrum.simulate_hv_scan(
                None,
                None,
                fixture["radial"],
                fixture["matrix_params"],
                fixture["quadrature"],
                fixture["final_state"],
                fixture["geometry"],
                fixture["self_energy"],
                fixture["kpath"],
                fixture["energy_axis"],
                photon_energy,
                1.0e-4,
                k_chunk=2,
                energy_chunk=2,
                checkpoint=False,
                bulk_model=model,
                surface_cell=fixture["surface_cell"],
                kz_mode="bulk_direct",
            )
            source: ArpesSpectrum = make_arpes_spectrum(
                scan[0],
                fixture["energy_axis"],
                k_axis,
                source_points,
                cartesian_frame_id="org.diffpes.frame.sample_cartesian",
            )
            sources.append(source)
        expected: DetectorRaster = simul.apply_detector_effects(
            tuple(sources),
            fixture["geometry"],
            calibration,
            detector_effects,
        )

        chex.assert_trees_all_equal(actual, expected)


class TestCanonicalBulkRaster:
    """Verify the public full-raster bulk driver success path.

    The case runs the canonical direct bulk source over the complete detector
    raster and checks the returned spectrum structure.
    """

    @pytest.mark.slow
    @pytest.mark.big_mem
    @pytest.mark.rss_limit_mb(2600)
    def test_public_simulate_arpes_bulk_direct_raster_succeeds(self) -> None:
        """Produce finite nonzero counts on every declared detector axis.

        The input is a true separable 2x2 bulk grid. The public canonical
        raster driver owns reciprocal mapping, source-cube construction, and
        the complete explicitly configured detector effects chain.

        Notes
        -----
        Compare native raster shape and coordinate centres with calibration.
        """
        fixture: Dict[str, object] = _driver_fixture()
        calibration: DetectorCalibration
        detector_effects: DetectorEffects
        calibration, detector_effects = _single_domain_detector_context(
            raster=True
        )
        geometry: ExperimentGeometry = eqx.tree_at(
            lambda item: item.sample_azimuth,
            fixture["geometry"],
            jnp.asarray(0.0, dtype=jnp.float64),
        )
        raster: DetectorRaster = spectrum.simulate_arpes(
            (),
            (),
            fixture["radial"],
            fixture["matrix_params"],
            fixture["quadrature"],
            fixture["final_state"],
            geometry,
            fixture["self_energy"],
            _bulk_kgrid(fixture),
            fixture["energy_axis"],
            calibration,
            detector_effects,
            1.0e-4,
            k_chunk=2,
            energy_chunk=2,
            checkpoint=False,
            bulk_models_by_domain=(fixture["model"],),
            surface_cells_by_domain=(fixture["surface_cell"],),
            kz_mode="bulk_direct",
        )
        expected_u: Float64[Array, "..."] = 0.5 * (
            calibration.u_bin_edges[:-1] + calibration.u_bin_edges[1:]
        )
        expected_v: Float64[Array, "..."] = 0.5 * (
            calibration.v_bin_edges[:-1] + calibration.v_bin_edges[1:]
        )
        expected_energy: Float64[Array, "..."] = 0.5 * (
            calibration.energy_bin_edges_ev[:-1]
            + calibration.energy_bin_edges_ev[1:]
        )

        assert raster.expected_counts.shape == (1, 2, 2, 2)
        assert jnp.array_equal(raster.detector_u_axis, expected_u)
        assert jnp.array_equal(raster.detector_v_axis, expected_v)
        assert jnp.array_equal(raster.energy_axis, expected_energy)
        assert jnp.all(jnp.isfinite(raster.expected_counts))
        assert jnp.all(raster.expected_counts >= 0.0)
        assert jnp.any(raster.expected_counts > 0.0)


class TestCanonicalBulkKzDetectorDerivatives:
    """Verify full-detector bulk-kz transformations and derivatives.

    The cases compare full-detector gradients with finite differences and
    require JIT and vectorization to preserve the expected counts.
    """

    @pytest.mark.slow
    @pytest.mark.big_mem
    @pytest.mark.rss_limit_mb(4800)
    def test_full_detector_bulk_kz_gradients_match_fd(self) -> None:
        """Differentiate four physical coordinates through detector counts.

        Mean free path, inner potential, work function, and sampled-energy
        translation pass the shared float64 FD ladder. Their derivatives remain
        nonzero after calibration, detector mapping, resolution, and effects.

        Notes
        -----
        Eight nodes are a caller-recalibrated reduced diagnostic and carry no
        quadrature-accuracy claim. Run this check in a fresh serial process.
        """
        fixture: Dict[str, object] = _driver_fixture()
        calibration: DetectorCalibration
        detector_effects: DetectorEffects
        calibration, detector_effects = _single_domain_detector_context()
        nodes: Float64[Array, "..."] = kz_fractional_nodes(8)
        coordinates: Float64[Array, "4"] = jnp.asarray(
            (7.5, 11.0, 4.3, 0.0), dtype=jnp.float64
        )

        def loss(values: Float64[Array, "..."]) -> Float64[Array, "..."]:
            """Return a generic weighted full-detector bulk-kz loss."""
            counts: Float64[Array, "..."] = _full_detector_bulk_kz_counts(
                fixture,
                values,
                calibration,
                detector_effects,
                nodes,
            )
            weights: Float64[Array, "..."] = jnp.linspace(
                0.7, 1.4, counts.size, dtype=jnp.float64
            ).reshape(counts.shape)
            value: Float64[Array, "..."] = jnp.sum(counts * weights)
            return value

        assert_grad_matches_fd(
            loss,
            coordinates,
            regime="smooth",
            modes=("fwd", "rev"),
        )
        gradient: Float64[Array, "..."] = jax.grad(loss)(coordinates)
        assert jnp.all(jnp.isfinite(gradient))
        assert jnp.all(jnp.abs(gradient) > 1.0e-12)

    @pytest.mark.slow
    @pytest.mark.big_mem
    @pytest.mark.rss_limit_mb(4400)
    def test_full_detector_bulk_kz_jit_and_vmap_preserve_counts(self) -> None:
        """Preserve complete expected counts under JIT and parameter vmap.

        The vectorized coordinate includes mean free path, inner potential,
        work function, and sampled-energy translation, so every registered
        bulk-kz detector coordinate crosses both transformation boundaries.

        Notes
        -----
        Eight nodes are a caller-recalibrated reduced diagnostic and carry no
        quadrature-accuracy claim. Run this check in a fresh serial process.
        """
        fixture: Dict[str, object] = _driver_fixture()
        calibration: DetectorCalibration
        detector_effects: DetectorEffects
        calibration, detector_effects = _single_domain_detector_context()
        nodes: Float64[Array, "..."] = kz_fractional_nodes(8)
        coordinates: Float64[Array, "4"] = jnp.asarray(
            (7.5, 11.0, 4.3, 0.0), dtype=jnp.float64
        )

        def rates(values: Float64[Array, "..."]) -> Float64[Array, "..."]:
            """Return full-detector bulk-kz counts at four coordinates."""
            counts: Float64[Array, "..."] = _full_detector_bulk_kz_counts(
                fixture,
                values,
                calibration,
                detector_effects,
                nodes,
            )
            return counts

        baseline: Float64[Array, "..."] = rates(coordinates)
        compiled: Float64[Array, "..."] = jax.jit(rates)(coordinates)
        perturbation: Float64[Array, "4"] = jnp.asarray(
            (0.07, 0.03, 0.01, 0.0015), dtype=jnp.float64
        )
        batch: Float64[Array, "..."] = jnp.stack(
            (coordinates, coordinates + perturbation)
        )
        vectorized: Float64[Array, "..."] = jax.jit(jax.vmap(rates))(batch)
        sequential: Float64[Array, "..."] = jnp.stack(
            (baseline, rates(coordinates + perturbation))
        )

        assert jnp.all(jnp.isfinite(baseline))
        assert jnp.all(baseline >= 0.0)
        assert jnp.any(baseline > 0.0)
        assert jnp.allclose(compiled, baseline, rtol=1.0e-10, atol=1.0e-13)
        assert jnp.allclose(vectorized, sequential, rtol=1.0e-10, atol=1.0e-13)


class TestPhotonEnergyScan:
    """Verify lax-scan stacking, transformation behavior, and slicing.

    The case compares the photon-energy scan with a loop reference under eager,
    compiled, and vectorized execution.
    """

    @pytest.mark.slow
    @pytest.mark.big_mem
    @pytest.mark.rss_limit_mb(2600)
    def test_scan_matches_loop_jit_and_vmap(self) -> None:
        """Compare exact single-energy runs under every transformation.

        A Python loop supplies the external stacking truth. JIT and vmap must
        preserve values for the same bulk-direct photon-energy coordinates.

        Notes
        -----
        Compare complete pre-detector arrays at strict float64 tolerances.
        """
        fixture: Dict[str, object] = _driver_fixture()
        photon_energies: Float64[Array, "..."] = fixture["photon_energies"]
        scan: Float64[Array, "..."] = _simulate_scan(
            fixture, photon_energies, mode="bulk_direct"
        )
        loop: Float64[Array, "..."] = jnp.stack(
            tuple(
                _simulate_scan(fixture, hv[None], mode="bulk_direct")[0]
                for hv in photon_energies
            )
        )

        def scan_function(
            values: Float64[Array, "..."],
        ) -> Float64[Array, "..."]:
            """Return one bulk-direct scan for transformed photon energies."""
            result: Float64[Array, "..."] = _simulate_scan(
                fixture, values, mode="bulk_direct"
            )
            return result

        def one_energy(value: Float64[Array, "..."]) -> Float64[Array, "..."]:
            """Return one bulk-direct spectrum for vectorized stacking."""
            result: Float64[Array, "..."] = _simulate_scan(
                fixture, value[None], mode="bulk_direct"
            )[0]
            return result

        compiled: Float64[Array, "..."] = jax.jit(scan_function)(
            photon_energies
        )
        vectorized: Float64[Array, "..."] = jax.jit(jax.vmap(one_energy))(
            photon_energies
        )

        chex.assert_trees_all_close(scan, loop, rtol=1.0e-12, atol=1.0e-14)
        chex.assert_trees_all_close(compiled, scan, rtol=1.0e-12, atol=1.0e-14)
        chex.assert_trees_all_close(
            vectorized, scan, rtol=1.0e-12, atol=1.0e-14
        )


class TestKzDriverGradients:
    """Verify bulk-kz finite-difference and nonzero-gradient checks.

    The cases perturb the mean free path, inner potential, work function,
    energy, kinematic Jacobian, and photon-energy scan coordinates.
    """

    @pytest.mark.slow
    @pytest.mark.big_mem
    @pytest.mark.rss_limit_mb(3200)
    def test_coherent_slab_mean_free_path_gradient_matches_fd(self) -> None:
        """Differentiate coherent depth attenuation independently of kz width.

        A positive orbital depth gives a finite nonzero mean-free-path row.
        Setting only that depth to exact zero removes the row, demonstrating
        that the coherent route does not borrow bulk-kz quadrature weights.

        Notes
        -----
        Compare both autodiff modes with the shared float64 FD ladder.
        """
        fixture: Dict[str, object] = _driver_fixture()
        zero_depth_fixture: Dict[str, object] = dict(fixture)
        coherent_bands: DiagonalizedBands = fixture["coherent_bands"]
        zero_depth_fixture["coherent_bands"] = eqx.tree_at(
            lambda item: item.depths,
            coherent_bands,
            jnp.zeros_like(coherent_bands.depths),
        )
        photon_energies: Float64[Array, "..."] = fixture["photon_energies"][
            1:2
        ]

        def loss(candidate: Float64[Array, "..."]) -> Float64[Array, "..."]:
            """Return a weighted coherent-slab loss at one escape length."""
            geometry: ExperimentGeometry = eqx.tree_at(
                lambda item: item.mean_free_path_ang,
                fixture["geometry"],
                candidate,
            )
            scan: Float64[Array, "..."] = _simulate_scan(
                fixture,
                photon_energies,
                mode="coherent_slab",
                geometry=geometry,
            )
            weights: Float64[Array, "..."] = jnp.linspace(
                0.8, 1.3, scan.size, dtype=jnp.float64
            ).reshape(scan.shape)
            value: Float64[Array, "..."] = jnp.sum(scan * weights)
            return value

        def zero_depth_loss(
            candidate: Float64[Array, "..."],
        ) -> Float64[Array, "..."]:
            """Return the planted zero-depth coherent negative control."""
            geometry: ExperimentGeometry = eqx.tree_at(
                lambda item: item.mean_free_path_ang,
                fixture["geometry"],
                candidate,
            )
            scan: Float64[Array, "..."] = _simulate_scan(
                zero_depth_fixture,
                photon_energies,
                mode="coherent_slab",
                geometry=geometry,
            )
            weights: Float64[Array, "..."] = jnp.linspace(
                0.8, 1.3, scan.size, dtype=jnp.float64
            ).reshape(scan.shape)
            value: Float64[Array, "..."] = jnp.sum(scan * weights)
            return value

        coordinate: Float64[Array, "..."] = fixture[
            "geometry"
        ].mean_free_path_ang
        assert_grad_matches_fd(
            loss,
            coordinate,
            regime="smooth",
            modes=("fwd", "rev"),
        )
        derivative: Float64[Array, "..."] = jax.grad(loss)(coordinate)
        zero_depth_derivative: Float64[Array, "..."] = jax.grad(
            zero_depth_loss
        )(coordinate)
        assert jnp.isfinite(derivative)
        assert jnp.abs(derivative) > 1.0e-12
        assert zero_depth_derivative == 0.0

    @pytest.mark.slow
    @pytest.mark.big_mem
    @pytest.mark.rss_limit_mb(3600)
    @pytest.mark.parametrize("mean_free_path", [5.0, 10.0, 50.0])
    def test_bulk_kz_mean_free_path_gradient_matches_fd(
        self,
        mean_free_path: float,
    ) -> None:
        """Differentiate the additional bulk-kz width at all registered scales.

        The dispersive model prevents the wrapped average from becoming a
        constant. Every escape-length derivative must remain finite and
        nonzero.

        Notes
        -----
        Compare reverse and forward autodiff with the shared float64 FD check.
        """
        fixture: Dict[str, object] = _driver_fixture()
        nodes: Float64[Array, "..."] = kz_fractional_nodes(8)
        photon_energies: Float64[Array, "..."] = fixture["photon_energies"][
            1:2
        ]

        def loss(candidate: Float64[Array, "..."]) -> Float64[Array, "..."]:
            """Return one weighted bulk-kz scan loss at a candidate length."""
            geometry: ExperimentGeometry = eqx.tree_at(
                lambda item: item.mean_free_path_ang,
                fixture["geometry"],
                candidate,
            )
            scan: Float64[Array, "..."] = _simulate_scan(
                fixture,
                photon_energies,
                mode="bulk_kz",
                geometry=geometry,
                kz_nodes_frac=nodes,
            )
            weights: Float64[Array, "..."] = jnp.linspace(
                0.7, 1.4, scan.size
            ).reshape(scan.shape)
            value: Float64[Array, "..."] = jnp.sum(scan * weights)
            return value

        coordinate: Float64[Array, "..."] = jnp.asarray(mean_free_path)
        assert_grad_matches_fd(
            loss,
            coordinate,
            regime="smooth",
            modes=("fwd", "rev"),
        )
        derivative: Float64[Array, "..."] = jax.grad(loss)(coordinate)
        assert jnp.isfinite(derivative)
        assert jnp.abs(derivative) > 1.0e-12

    @pytest.mark.slow
    @pytest.mark.big_mem
    @pytest.mark.rss_limit_mb(3600)
    def test_bulk_kz_inner_potential_gradient_matches_fd(self) -> None:
        """Differentiate the exact bulk-kz center through inner potential.

        Generic kz dispersion supplies a nonzero inner-potential Fisher row.
        A detached or Fermi-frozen center fails both FD and tripwire checks.

        Notes
        -----
        Compare both autodiff modes with the shared finite-difference harness.
        """
        fixture: Dict[str, object] = _driver_fixture()
        nodes: Float64[Array, "..."] = kz_fractional_nodes(8)
        photon_energies: Float64[Array, "..."] = fixture["photon_energies"][
            1:2
        ]

        def loss(candidate: Float64[Array, "..."]) -> Float64[Array, "..."]:
            """Return weighted bulk-kz loss at a candidate inner potential."""
            geometry: ExperimentGeometry = eqx.tree_at(
                lambda item: item.inner_potential_ev,
                fixture["geometry"],
                candidate,
            )
            scan: Float64[Array, "..."] = _simulate_scan(
                fixture,
                photon_energies,
                mode="bulk_kz",
                geometry=geometry,
                kz_nodes_frac=nodes,
            )
            weights: Float64[Array, "..."] = jnp.linspace(
                1.3, 0.6, scan.size
            ).reshape(scan.shape)
            value: Float64[Array, "..."] = jnp.sum(scan * weights)
            return value

        coordinate: Float64[Array, "..."] = fixture[
            "geometry"
        ].inner_potential_ev
        assert_grad_matches_fd(
            loss,
            coordinate,
            regime="smooth",
            modes=("fwd", "rev"),
        )
        derivative: Float64[Array, "..."] = jax.grad(loss)(coordinate)
        assert jnp.isfinite(derivative)
        assert jnp.abs(derivative) > 1.0e-12

    def test_exact_kinematic_jacobian_identity_matches_fd(self) -> None:
        """Preserve exact hnu, work-function, and omega center Jacobians.

        Before downstream factors, the exact center obeys equal hnu and omega
        derivatives and the opposite work-function derivative.

        Notes
        -----
        Compare the complete coordinate gradient with the shared FD harness.
        """
        coordinate: Float64[Array, "3"] = jnp.asarray((28.0, 4.3, -0.21))
        parallel_momentum: Float64[Array, ""] = jnp.asarray(0.17)

        def center(values: Float64[Array, "..."]) -> Float64[Array, "..."]:
            """Return one propagating exact inner-potential center."""
            kz_value: Float64[Array, "..."]
            valid: Bool[Array, "..."]
            kz_value, valid = kz_from_inner_potential(
                values[0],
                values[1],
                11.0,
                values[2],
                parallel_momentum,
            )
            checked: Float64[Array, "..."] = jnp.where(
                valid, jnp.real(kz_value), jnp.nan
            )
            return checked

        assert_grad_matches_fd(
            center,
            coordinate,
            regime="smooth",
            modes=("fwd", "rev"),
        )
        jacobian: Float64[Array, "..."] = jax.grad(center)(coordinate)
        assert jnp.allclose(jacobian[0], jacobian[2], rtol=1.0e-12)
        assert jnp.allclose(jacobian[0], -jacobian[1], rtol=1.0e-12)

    @pytest.mark.slow
    @pytest.mark.big_mem
    @pytest.mark.rss_limit_mb(3200)
    def test_driver_work_function_and_omega_gradients_match_fd(
        self,
    ) -> None:
        """Differentiate work function and every translated energy sample.

        The bulk-direct driver receives both coordinates before exact center
        construction. Generic weights keep both downstream derivatives nonzero.

        Notes
        -----
        Compare both autodiff modes with the shared elementwise FD harness.
        """
        fixture: Dict[str, object] = _driver_fixture()
        coordinate: Float64[Array, "..."] = jnp.asarray(
            (fixture["geometry"].work_function_ev, 0.0)
        )
        photon_energies: Float64[Array, "..."] = fixture["photon_energies"][
            1:2
        ]

        def loss(values: Float64[Array, "..."]) -> Float64[Array, "..."]:
            """Return a weighted scan after work and omega translations."""
            geometry: ExperimentGeometry = eqx.tree_at(
                lambda item: item.work_function_ev,
                fixture["geometry"],
                values[0],
            )
            energy_axis: Float64[Array, "..."] = (
                fixture["energy_axis"] + values[1]
            )
            scan: Float64[Array, "..."] = _simulate_scan(
                fixture,
                photon_energies,
                mode="bulk_direct",
                geometry=geometry,
                energy_axis=energy_axis,
            )
            weights: Float64[Array, "..."] = jnp.linspace(
                0.6, 1.6, scan.size
            ).reshape(scan.shape)
            value: Float64[Array, "..."] = jnp.sum(scan * weights)
            return value

        assert_grad_matches_fd(
            loss,
            coordinate,
            regime="smooth",
            modes=("fwd", "rev"),
        )
        gradient: Float64[Array, "..."] = jax.grad(loss)(coordinate)
        assert jnp.all(jnp.isfinite(gradient))
        assert jnp.all(jnp.abs(gradient) > 1.0e-12)

    @pytest.mark.slow
    @pytest.mark.big_mem
    @pytest.mark.rss_limit_mb(3000)
    def test_scan_photon_energy_gradients_match_fd(self) -> None:
        """Differentiate every caller-owned photon-energy scan coordinate.

        Both scan entries influence a generic weighted bulk-direct loss.
        A Python interpolation or detached five-point schedule fails this
        check.

        Notes
        -----
        Compare elementwise autodiff with the shared float64 FD harness.
        """
        fixture: Dict[str, object] = _driver_fixture()
        coordinate: Float64[Array, "..."] = fixture["photon_energies"][:2]

        def loss(values: Float64[Array, "..."]) -> Float64[Array, "..."]:
            """Return one weighted bulk-direct photon-energy scan loss."""
            scan: Float64[Array, "..."] = _simulate_scan(
                fixture, values, mode="bulk_direct"
            )
            weights: Float64[Array, "..."] = jnp.linspace(
                0.8, 1.5, scan.size
            ).reshape(scan.shape)
            value: Float64[Array, "..."] = jnp.sum(scan * weights)
            return value

        assert_grad_matches_fd(
            loss,
            coordinate,
            regime="smooth",
            modes=("fwd", "rev"),
        )
        gradient: Float64[Array, "..."] = jax.grad(loss)(coordinate)
        assert jnp.all(jnp.isfinite(gradient))
        assert jnp.all(jnp.abs(gradient) > 1.0e-12)
