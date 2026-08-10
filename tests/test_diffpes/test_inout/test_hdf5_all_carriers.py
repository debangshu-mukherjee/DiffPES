"""Verify HDF5 round trips for every types-owned Equinox carrier.

Extended Summary
----------------
Exercises the introspected recursive codec over every registered carrier class,
including nested modules, static tuple metadata, complex arrays, and absent
optional leaves.
"""

import tempfile
from pathlib import Path

import chex
import equinox as eqx
import jax.numpy as jnp
from beartype.typing import Any, Callable, Dict
from jaxtyping import Array

import diffpes
from diffpes.inout import load_from_h5, save_to_h5
from diffpes.types import (
    make_arpes_cube,
    make_arpes_spectrum,
    make_band_structure,
    make_density_of_states,
    make_detector_calibration,
    make_detector_effects,
    make_detector_raster,
    make_diagonalized_bands,
    make_experiment_geometry,
    make_final_state_spec,
    make_full_density_of_states,
    make_kgrid,
    make_kpath,
    make_kpath_info,
    make_matrix_element_params,
    make_orbital_projection,
    make_radial_quadrature_spec,
    make_radial_spec,
    make_self_energy_model,
    make_soc_volumetric_data,
    make_spin_band_structure,
    make_spin_orbital_projection,
    make_tb_model,
    make_volumetric_data,
    make_workflow_context,
)
from diffpes.types.wannier import make_wannier_operator_data
from tests._factories import make_1d_chain_model


def _all_carriers() -> Dict[str, eqx.Module]:
    """PRIVATE: Construct one deterministic instance of every carrier class.

    Returns
    -------
    carriers : dict[str, eqx.Module]
        Mapping from a short label to one instance of each registered
        types-owned Equinox carrier. Nested carriers such as the
        tight-binding model keep their geometry and basis.

    Notes
    -----
    Builds small fixed-value arrays with two k-points and one band or
    orbital. Reuses the one-dimensional chain factory for the
    tight-binding pieces, so every HDF5 round-trip test sees
    identical inputs.
    """
    energy: Array
    kpoints: Array
    bands: diffpes.types.BandStructure
    projections: diffpes.types.OrbitalProjection
    tb_model: diffpes.types.TBModel
    geometry: diffpes.types.CrystalGeometry
    basis: diffpes.types.OrbitalBasis
    diagonalized: diffpes.types.DiagonalizedBands
    charge: Array
    cartesian_path: Array

    energy = jnp.array([-1.0, 1.0], dtype=jnp.float64)
    kpoints = jnp.zeros((2, 3), dtype=jnp.float64)
    bands = make_band_structure(energy[:, None], kpoints)
    projections = make_orbital_projection(jnp.ones((2, 1, 1, 9)))
    template: diffpes.types.TBModel = make_1d_chain_model()
    orbital_positions: Array = jnp.asarray(
        [[0.125, 0.0, 0.0]],
        dtype=jnp.float64,
    )
    tb_model = make_tb_model(
        hopping_amplitudes=template.hopping_amplitudes,
        onsite_energies=template.onsite_energies,
        soc_lambdas=template.soc_lambdas,
        geometry=template.geometry,
        basis=template.basis,
        hopping_pairs=template.hopping_pairs,
        hopping_cells=template.hopping_cells,
        shell_index=template.shell_index,
        spinor=template.spinor,
        orbital_positions=orbital_positions,
    )
    geometry = tb_model.geometry
    basis = tb_model.basis
    diagonalized = make_diagonalized_bands(
        eigenvalues=energy[:, None],
        eigenvectors=jnp.ones((2, 1, 1), dtype=jnp.complex128),
        kpoints=kpoints,
        geometry=geometry,
        basis=basis,
        orbital_positions=orbital_positions,
    )
    charge = jnp.ones((2, 2, 2), dtype=jnp.float64)
    cartesian_path = jnp.asarray(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
        dtype=jnp.float64,
    )
    carriers: Dict[str, eqx.Module] = {
        "arpes_cube": make_arpes_cube(
            jnp.ones((2, 2, 2)),
            energy,
            energy,
            energy,
            provenance="hdf5-round-trip",
        ),
        "arpes": make_arpes_spectrum(
            jnp.ones((2, 2)),
            energy,
            jnp.array([0.0, 1.0]),
            cartesian_path,
        ),
        "bands": bands,
        "spin_bands": make_spin_band_structure(
            energy[:, None], energy[:, None], kpoints
        ),
        "projection": projections,
        "spin_projection": make_spin_orbital_projection(
            jnp.ones((2, 1, 1, 9)), jnp.zeros((2, 1, 1, 6))
        ),
        "dos": make_density_of_states(energy, jnp.ones(2)),
        "full_dos": make_full_density_of_states(
            energy, jnp.ones(2), jnp.arange(2.0), natoms=0
        ),
        "geometry": geometry,
        "experiment": make_experiment_geometry(
            21.2,
            jnp.asarray([1.0, 0.0, 0.0], dtype=jnp.complex128),
        ),
        "detector_calibration": make_detector_calibration(
            u_bin_edges=jnp.array([-0.2, 0.0, 0.2]),
            v_bin_edges=jnp.array([-0.05, 0.05]),
            energy_bin_edges_ev=jnp.array([-1.5, -0.5, 0.5, 1.5]),
            psf_fwhm_u=0.01,
            psf_fwhm_v=0.02,
            psf_fwhm_energy_ev=0.05,
            transmission_reference_domain_ev=jnp.array([10.0, 30.0]),
        ),
        "detector_effects": make_detector_effects(
            domain_logits=jnp.array([0.0]),
            domain_euler_angles_rad=jnp.zeros((1, 3)),
            transmission_raw_slopes=jnp.array([-0.4, 0.2]),
            background_coefficients=jnp.array([-2.0]),
            sensitivity_coefficients=jnp.empty((0,)),
            exposure=100.0,
            background_mode="flat",
            sensitivity_mode="constant",
            domain_frame_ids=("org.diffpes.frame.sample_cartesian",),
        ),
        "detector_raster": make_detector_raster(
            expected_counts=jnp.ones((1, 2, 1, 2)),
            detector_u_axis=jnp.array([-0.1, 0.1]),
            detector_v_axis=jnp.array([0.0]),
            energy_axis=energy,
            channel_labels=("intensity",),
            coordinate_system="hemispherical_angles",
        ),
        "generated_kpath": make_kpath(
            kpoints,
            labels=("G", "X"),
            label_indices=(0, 1),
            n_per_segment=2,
        ),
        "kgrid": make_kgrid(kpoints, mesh_shape=(1, 2)),
        "kpath": make_kpath_info(2, [0, 1], segments=1, labels=("G", "X")),
        "basis": basis,
        "radial": make_radial_spec(basis, (0,)),
        "matrix_element": make_matrix_element_params(
            basis,
            (0,),
            phase_shift_angles_shell=jnp.asarray((0.37,)),
        ),
        "radial_quadrature": make_radial_quadrature_spec(),
        "final_state": make_final_state_spec(),
        "self_energy": make_self_energy_model(),
        "diagonalized": diagonalized,
        "tb_model": tb_model,
        "volumetric": make_volumetric_data(
            jnp.eye(3),
            jnp.zeros((1, 3)),
            charge,
            grid_shape=(2, 2, 2),
            symbols=("X",),
            atom_counts=jnp.ones(1, dtype=jnp.int32),
        ),
        "soc_volumetric": make_soc_volumetric_data(
            jnp.eye(3),
            jnp.zeros((1, 3)),
            charge,
            charge,
            jnp.ones((2, 2, 2, 3)),
            grid_shape=(2, 2, 2),
            symbols=("X",),
            atom_counts=jnp.ones(1, dtype=jnp.int32),
        ),
        "wannier": make_wannier_operator_data(
            position_matrices=jnp.asarray(
                [
                    [[[(0.25 + 0.5j), 1.0, -0.5j]]],
                    [[[(0.75 - 0.25j), -1.0j, 2.0]]],
                ],
                dtype=jnp.complex128,
            ),
            centres_cart=jnp.asarray([[0.1, 0.2, 0.3]]),
            cells=((0, 0, 0), (1, -2, 3)),
            degeneracies=(1, 4),
            spin_layout="interleaved_up_down",
            source_format="tb",
        ),
        "context": make_workflow_context(bands, projections),
    }
    return carriers


def test_all_carriers_round_trip_bitwise() -> None:
    """Round-trip every carrier with exact leaves and static metadata.

    Extended Summary
    ----------------
    Saves all deterministic carriers into one HDF5 file and reloads
    them together. Each reconstructed module must retain its exact class,
    numerical leaves, nested modules, optional ``None`` leaves, and static
    metadata.

    Notes
    -----
    Uses :func:`equinox.tree_equal` for an exact recursive comparison after a
    single multi-group save/load cycle.
    """
    temporary_directory: str
    name: str
    carrier: eqx.Module

    carriers: Dict[str, eqx.Module] = _all_carriers()
    with tempfile.TemporaryDirectory() as temporary_directory:
        path: Path = Path(temporary_directory) / "all_carriers.h5"
        save_to_h5(path, **carriers)
        loaded: Dict[str, eqx.Module] = load_from_h5(path)

    chex.assert_equal(set(loaded), set(carriers))
    for name, carrier in carriers.items():
        chex.assert_equal(type(loaded[name]) is type(carrier), True)
        chex.assert_equal(eqx.tree_equal(loaded[name], carrier), True)
    matrix_element: diffpes.types.MatrixElementParams = loaded[
        "matrix_element"
    ]
    assert matrix_element.phase_channel_keys == ((0, 1),)
    chex.assert_shape(matrix_element.phase_shift_angles_shell, (1,))


def test_wannier_hr_round_trip_preserves_absent_position_matrices() -> None:
    """Round-trip the ``hr.dat`` sidecar with its optional array absent.

    The case protects optional Wannier operator metadata through persistence.

    Notes
    -----
    Compare the complete reconstructed carrier with Equinox tree equality.
    """
    temporary_directory: str

    carrier: eqx.Module = make_wannier_operator_data(
        position_matrices=None,
        centres_cart=jnp.asarray(
            [[0.0, 0.25, 0.5], [0.75, 1.0, 1.25]],
            dtype=jnp.float64,
        ),
        cells=((-1, 0, 0), (0, 0, 0), (1, 0, 0)),
        degeneracies=(2, 1, 2),
        spin_layout="block_down_up",
        source_format="hr",
    )
    with tempfile.TemporaryDirectory() as temporary_directory:
        path: Path = Path(temporary_directory) / "wannier_hr.h5"
        save_to_h5(path, wannier=carrier)
        loaded: eqx.Module = load_from_h5(path, name="wannier")

    chex.assert_equal(type(loaded) is type(carrier), True)
    chex.assert_equal(eqx.tree_equal(loaded, carrier), True)
