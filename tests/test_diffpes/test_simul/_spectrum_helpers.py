"""Provide shared fixtures for the mirrored split test modules.

The helpers preserve the inputs and independent reference calculations.
"""

import jax.numpy as jnp
from beartype.typing import Any, Dict, Tuple
from jaxtyping import Array, Complex128, Float64

from diffpes.simul._source_carriers import _physical_cubes
from diffpes.types import (
    ArpesCube,
    ArpesSpectrum,
    DetectorCalibration,
    DetectorEffects,
    DetectorRaster,
    DiagonalizedBands,
    ExperimentGeometry,
    FinalStateSpec,
    KGrid,
    MatrixElementParams,
    RadialQuadratureSpec,
    RadialSpec,
    SelfEnergyModel,
    make_crystal_geometry,
    make_detector_calibration,
    make_detector_effects,
    make_detector_raster,
    make_diagonalized_bands,
    make_experiment_geometry,
    make_final_state_spec,
    make_kgrid,
    make_matrix_element_params,
    make_orbital_basis,
    make_radial_quadrature_spec,
    make_radial_spec,
    make_self_energy_model,
)

_FRAME_ID: str = "org.diffpes.frame.sample_cartesian"


def _fixture() -> Dict[str, object]:
    """PRIVATE: Build one asymmetric manufactured source-raster fixture.

    The single-orbital model makes the resolvent truth analytic.  A positive
    depth, asymmetric k mesh, generic optical field, and off-Fermi energies
    keep every registered geometry derivative exposed.

    Returns
    -------
    fixture : Dict[str, object]
        Complete canonical-driver inputs.
    """
    crystal: Any = make_crystal_geometry(
        2.0 * jnp.pi * jnp.eye(3, dtype=jnp.float64),
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
    kx: Float64[Array, "2"] = jnp.asarray([0.025, 0.13], dtype=jnp.float64)
    ky: Float64[Array, "2"] = jnp.asarray([-0.04, 0.075], dtype=jnp.float64)
    mesh_x: Float64[Array, "..."]
    mesh_y: Float64[Array, "..."]
    mesh_x, mesh_y = jnp.meshgrid(kx, ky, indexing="xy")
    kpoints: Float64[Array, "..."] = jnp.stack(
        (mesh_x, mesh_y, jnp.zeros_like(mesh_x)), axis=-1
    ).reshape((-1, 3))
    eigenvalues: Float64[Array, "..."] = -0.08 + 0.7 * jnp.sum(
        kpoints[:, :2] ** 2, axis=-1
    )
    bands: DiagonalizedBands = make_diagonalized_bands(
        eigenvalues[:, None],
        jnp.ones((4, 1, 1), dtype=jnp.complex128),
        kpoints,
        crystal,
        basis,
        fermi_energy=0.0,
        depths=jnp.asarray([0.65]),
    )
    hamiltonians: Complex128[Array, "..."] = eigenvalues[:, None, None].astype(
        jnp.complex128
    )
    radial: RadialSpec = make_radial_spec(
        basis,
        (0,),
        mode="fixed",
        fixed_integrals_shell=jnp.asarray([[0.0, 1.0]]),
    )
    matrix_params: MatrixElementParams = make_matrix_element_params(
        basis,
        (0,),
        sigma_shell=jnp.asarray([1.17]),
        phase_shift_angles_shell=jnp.asarray([0.23]),
    )
    quadrature: RadialQuadratureSpec = make_radial_quadrature_spec()
    final_state: FinalStateSpec = make_final_state_spec()
    geometry: ExperimentGeometry = make_experiment_geometry(
        photon_energy_ev=50.0,
        polarization=jnp.asarray([1.0, 0.4j, 0.0], dtype=jnp.complex128),
        incidence_theta=0.0,
        incidence_phi=0.0,
        sample_azimuth=0.17,
        work_function_ev=4.5,
        temperature_k=30.0,
        mean_free_path_ang=8.0,
    )
    self_energy: SelfEnergyModel = make_self_energy_model(gamma=0.04)
    energy_axis: Float64[Array, "5"] = jnp.asarray(
        [-0.22, -0.09, -0.015, 0.055, 0.18], dtype=jnp.float64
    )
    kgrid: KGrid = make_kgrid(kpoints, mesh_shape=(2, 2), kz=0.0)
    calibration: DetectorCalibration = make_detector_calibration(
        u_bin_edges=jnp.asarray([-0.1, 0.0, 0.15]),
        v_bin_edges=jnp.asarray([-0.08, 0.1]),
        energy_bin_edges_ev=jnp.asarray([-0.25, -0.05, 0.2]),
        psf_fwhm_u=0.01,
        psf_fwhm_v=0.01,
        psf_fwhm_energy_ev=0.02,
        transmission_reference_domain_ev=jnp.asarray([45.0, 46.0]),
    )
    detector_effects: DetectorEffects = make_detector_effects(
        domain_logits=jnp.asarray([0.0]),
        domain_euler_angles_rad=jnp.zeros((1, 3)),
        transmission_raw_slopes=jnp.asarray([-0.4, 0.2]),
        background_coefficients=jnp.asarray([-2.0]),
        sensitivity_coefficients=jnp.asarray([]),
        exposure=1.0,
        background_mode="flat",
        sensitivity_mode="constant",
        domain_frame_ids=(_FRAME_ID,),
    )
    returned: Dict[str, object] = {
        "bands": bands,
        "hamiltonians": hamiltonians,
        "radial": radial,
        "matrix_params": matrix_params,
        "quadrature": quadrature,
        "final_state": final_state,
        "geometry": geometry,
        "self_energy": self_energy,
        "energy_axis": energy_axis,
        "kgrid": kgrid,
        "calibration": calibration,
        "detector_effects": detector_effects,
    }
    return returned


def _physical_cube_fixture(
    fixture: Dict[str, object],
    *,
    hamiltonians: Any = None,
    geometry: Any = None,
) -> Tuple[ArpesCube, ...]:
    """PRIVATE: Build source cubes with nondivisible scan extents.

    Parameters
    ----------
    fixture : Dict[str, object]
        Complete manufactured driver inputs.
    hamiltonians : Any, optional
        Explicit Hamiltonian override.
    geometry : Any, optional
        Experiment-geometry override.

    Returns
    -------
    cubes : Tuple[ArpesCube, ...]
        One manufactured physical source cube.
    """
    resolved_hamiltonians: Any = (
        fixture["hamiltonians"] if hamiltonians is None else hamiltonians
    )
    resolved_geometry: Any = (
        fixture["geometry"] if geometry is None else geometry
    )
    cubes: Tuple[ArpesCube, ...] = _physical_cubes(
        (resolved_hamiltonians,),
        (fixture["bands"],),
        fixture["radial"],
        fixture["matrix_params"],
        fixture["quadrature"],
        fixture["final_state"],
        resolved_geometry,
        fixture["self_energy"],
        fixture["kgrid"],
        fixture["energy_axis"],
        1.0e-4,
        k_chunk=3,
        energy_chunk=4,
        checkpoint=False,
    )
    return cubes


def _identity_detector_chain(
    physical_by_domain: Tuple[ArpesCube | ArpesSpectrum, ...],
    _geometry: ExperimentGeometry,
    _calibration: DetectorCalibration,
    _effects: DetectorEffects,
) -> DetectorRaster:
    """PRIVATE: Preserve source values in a typed detector-shaped carrier.

    This test-only seam isolates the spectral source driver from detector-map
    correctness, which has its own focused module.

    Parameters
    ----------
    physical_by_domain : Tuple[ArpesCube | ArpesSpectrum, ...]
        Single-domain physical source carrier.
    _geometry : ExperimentGeometry
        Unused detector geometry.
    _calibration : DetectorCalibration
        Unused target calibration.
    _effects : DetectorEffects
        Unused nuisance state.

    Returns
    -------
    raster : DetectorRaster
        Source values with an explicit leading detector-channel axis.
    """
    source: ArpesCube | ArpesSpectrum = physical_by_domain[0]
    if isinstance(source, ArpesCube):
        returned: DetectorRaster = make_detector_raster(
            source.intensity[None, ...],
            source.kx_axis,
            source.ky_axis,
            source.energy_axis,
            ("total",),
            "hemispherical_angles",
        )
        return returned
    returned: DetectorRaster = make_detector_raster(
        source.intensity[None, :, None, :],
        source.k_axis,
        jnp.zeros((1,), dtype=jnp.float64),
        source.energy_axis,
        ("total",),
        "hemispherical_angles",
    )
    return returned
