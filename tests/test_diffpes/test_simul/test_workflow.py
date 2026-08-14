"""Validate high-level workflow helpers in :mod:`diffpes.simul.workflow`.

Extended Summary
----------------
Validates VASP context loading and projection preparation with controlled
temporary inputs.
"""

from pathlib import Path
from unittest.mock import Mock

import chex
import jax.numpy as jnp
import pytest
from beartype.typing import Any, Dict, Tuple, cast
from jaxtyping import Array, Complex128, Float64

import diffpes
from diffpes.simul import (
    load_vasp_context,
    prepare_projection,
    workflow,
)
from diffpes.types import (
    CrystalGeometry,
    DetectorCalibration,
    DetectorEffects,
    DetectorRaster,
    ExperimentGeometry,
    FinalStateSpec,
    KPath,
    MatrixElementParams,
    OrbitalBasis,
    RadialQuadratureSpec,
    RadialSpec,
    SelfEnergyModel,
    SpinOrbitalProjection,
    make_crystal_geometry,
    make_detector_calibration,
    make_detector_effects,
    make_detector_raster,
    make_experiment_geometry,
    make_final_state_spec,
    make_matrix_element_params,
    make_orbital_basis,
    make_orbital_projection,
    make_radial_quadrature_spec,
    make_radial_spec,
    make_self_energy_model,
    make_spin_orbital_projection,
)

_FIXTURES_DIR: Path = (
    Path(__file__).resolve().parents[1] / "test_inout" / "fixtures"
)


@pytest.fixture
def workflow_carriers() -> Dict[str, Any]:
    """Build explicit matrix-element, spectral, and detector carriers."""
    crystal: CrystalGeometry = make_crystal_geometry(
        2.0 * jnp.pi * jnp.eye(3, dtype=jnp.float64),
        jnp.zeros((1, 3), dtype=jnp.float64),
        ("X",),
    )
    basis: OrbitalBasis = make_orbital_basis(
        atom_indices=(0,),
        n=(1,),
        l=(0,),
        m=(0,),
        labels=("1s",),
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
        sigma_shell=jnp.asarray([1.0]),
        phase_shift_angles_shell=jnp.asarray([0.0]),
    )
    experiment: ExperimentGeometry = make_experiment_geometry(
        photon_energy_ev=50.0,
        polarization=jnp.asarray([1.0, 0.0, 0.0], dtype=jnp.complex128),
        work_function_ev=4.5,
        temperature_k=25.0,
    )
    energy_axis: Float64[Array, " 3"] = jnp.asarray(
        [-0.2, 0.0, 0.2], dtype=jnp.float64
    )
    calibration: DetectorCalibration = make_detector_calibration(
        u_bin_edges=jnp.asarray([-0.05, 0.08, 0.18]),
        v_bin_edges=jnp.asarray([-0.02, 0.02]),
        energy_bin_edges_ev=jnp.asarray([-0.3, -0.1, 0.1, 0.3]),
        psf_fwhm_u=0.01,
        psf_fwhm_v=0.01,
        psf_fwhm_energy_ev=0.02,
        transmission_reference_domain_ev=jnp.asarray([45.0, 46.0]),
    )
    effects: DetectorEffects = make_detector_effects(
        domain_logits=jnp.asarray([0.0]),
        domain_euler_angles_rad=jnp.zeros((1, 3)),
        transmission_raw_slopes=jnp.asarray([-0.2, 0.1]),
        background_coefficients=jnp.asarray([-3.0]),
        sensitivity_coefficients=jnp.asarray([]),
        exposure=1.0,
        background_mode="flat",
        sensitivity_mode="constant",
        domain_frame_ids=("org.diffpes.frame.sample_cartesian",),
    )
    hamiltonians: Complex128[Array, "2 1 1"] = jnp.asarray(
        [[[-1.5]], [[-1.0]]], dtype=jnp.complex128
    )
    quadrature: RadialQuadratureSpec = make_radial_quadrature_spec()
    final_state: FinalStateSpec = make_final_state_spec()
    self_energy: SelfEnergyModel = make_self_energy_model(gamma=0.04)
    carriers: Dict[str, Any] = {
        "hamiltonians": hamiltonians,
        "crystal": crystal,
        "basis": basis,
        "radial": radial,
        "matrix_params": matrix_params,
        "quadrature": quadrature,
        "final_state": final_state,
        "experiment": experiment,
        "self_energy": self_energy,
        "energy_axis": energy_axis,
        "calibration": calibration,
        "effects": effects,
    }
    return carriers


class TestLoadVaspContextEdgeCases(chex.TestCase):
    """Validate additional paths in :func:`diffpes.simul.load_vasp_context`.

    :see: :func:`~diffpes.simul.load_vasp_context`
    """

    def test_no_doscar_fermi_defaults_to_zero(self) -> None:
        """Verify the default Fermi energy without DOSCAR input.

        The test passes ``doscar_file=None`` and ``fermi_energy=None``,
        exercising
        workflow.py line 142 (``resolved_fermi = 0.0``). Asserts the
        returned band structure has fermi_energy == 0.0.

        Notes
        -----
        The test builds inputs in its body and checks the stated property with
        the documented numerical or structural assertions.
        """
        context: diffpes.types.WorkflowContext

        context = load_vasp_context(
            directory=str(_FIXTURES_DIR),
            eigenval_file="EIGENVAL_spin",
            procar_file="PROCAR_spin",
            doscar_file=None,
            kpoints_file=None,
            fermi_energy=None,
            check_dimensions=True,
        )
        chex.assert_trees_all_close(
            context.bands.fermi_energy, jnp.float64(0.0), atol=1e-12
        )
        assert context.dos is None

    def test_missing_doscar_raises(self) -> None:
        """Verify that a missing required DOSCAR raises FileNotFoundError.

        The test passes a non-existent ``doscar_file`` with
        ``fermi_energy=None``,
        exercising workflow.py lines 146-150 (FileNotFoundError path).

        Notes
        -----
        The test builds inputs in its body and checks the stated property with
        the documented numerical or structural assertions.
        """
        with pytest.raises(
            FileNotFoundError,
            match="DOSCAR is required.*not found",
        ):
            load_vasp_context(
                directory=str(_FIXTURES_DIR),
                eigenval_file="EIGENVAL_spin",
                procar_file="PROCAR_spin",
                doscar_file="DOES_NOT_EXIST",
                kpoints_file=None,
                fermi_energy=None,
                check_dimensions=False,
            )

    def test_explicit_fermi_reads_doscar_optionally(self) -> None:
        """Verify optional DOSCAR loading with an explicit Fermi energy.

        The test passes ``fermi_energy=1.5`` and a valid ``doscar_file``,
        exercising
        workflow.py lines 154-158 (optional DOSCAR read). The explicit
        Fermi energy controls the bands, and the file supplies ``dos``.

        Notes
        -----
        The test builds inputs in its body and checks the stated property with
        the documented numerical or structural assertions.
        """
        context: diffpes.types.WorkflowContext

        context = load_vasp_context(
            directory=str(_FIXTURES_DIR),
            eigenval_file="EIGENVAL_spin",
            procar_file="PROCAR_spin",
            doscar_file="DOSCAR",
            kpoints_file=None,
            fermi_energy=1.5,
            check_dimensions=True,
        )
        chex.assert_trees_all_close(
            context.bands.fermi_energy, jnp.float64(1.5), atol=1e-12
        )
        assert context.dos is not None


class TestLoadVaspContext(chex.TestCase):
    """Validate :func:`diffpes.simul.load_vasp_context`.

    :see: :func:`~diffpes.simul.load_vasp_context`
    """

    def test_loads_context_with_optional_dos_and_kpath(self) -> None:
        """Verify context loading with inferred Fermi level and checks.

        The test establishes the optional DOS and k-path contract for loading
        VASP context with the concrete values and array shapes described below.

        Notes
        -----
        The test builds inputs in its body and checks the stated property with
        the documented numerical or structural assertions.
        """
        context: diffpes.types.WorkflowContext

        context = load_vasp_context(
            directory=str(_FIXTURES_DIR),
            eigenval_file="EIGENVAL_spin",
            procar_file="PROCAR_spin",
            doscar_file="DOSCAR",
            kpoints_file="KPOINTS_line_fallback",
            procar_mode="full",
            check_dimensions=True,
        )
        chex.assert_shape(context.bands.eigenvalues, (2, 2))
        chex.assert_shape(context.orb_proj.projections, (2, 2, 1, 9))
        assert context.orb_proj.spin is not None
        assert context.kpath is not None
        assert context.dos is not None
        chex.assert_trees_all_close(
            context.bands.fermi_energy,
            jnp.float64(0.5),
            atol=1e-12,
        )


class TestPrepareProjection(chex.TestCase):
    """Validate :func:`diffpes.simul.prepare_projection`.

    :see: :func:`~diffpes.simul.prepare_projection`
    """

    def test_spin_orbital_projection_attaches_oam(self) -> None:
        """Verify OAM attachment works for SpinOrbitalProjection input.

        The test constructs a SpinOrbitalProjection and calls
        ``prepare_projection``
        with ``attach_oam=True``. Asserts the returned object is still a
        SpinOrbitalProjection with OAM attached, covering workflow.py
        line 224 (make_spin_orbital_projection with oam).

        Notes
        -----
        The test builds inputs in its body and checks the stated property with
        the documented numerical or structural assertions.
        """
        projections: Float64[Array, "..."]
        spin: Float64[Array, "..."]
        orb: diffpes.types.SpinOrbitalProjection
        prepared: (
            diffpes.types.OrbitalProjection
            | diffpes.types.SpinOrbitalProjection
        )

        projections = jnp.ones((2, 2, 2, 9), dtype=jnp.float64)
        spin = jnp.ones((2, 2, 2, 6), dtype=jnp.float64)
        orb = make_spin_orbital_projection(projections=projections, spin=spin)
        prepared = prepare_projection(orb_proj=orb, attach_oam=True)
        assert isinstance(prepared, SpinOrbitalProjection)
        assert prepared.oam is not None
        chex.assert_shape(prepared.oam, (2, 2, 2, 3))

    def test_selects_atoms_and_attaches_oam(self) -> None:
        """Verify atom sub-selection and OAM attachment in one call.

        The test establishes atom selection and OAM attachment for
        ``prepare_projection``
        projection with the concrete values and array shapes described below.

        Notes
        -----
        The test builds inputs in its body and checks the stated property with
        the documented numerical or structural assertions.
        """
        projections: Float64[Array, "..."]
        orb: diffpes.types.OrbitalProjection
        prepared: (
            diffpes.types.OrbitalProjection
            | diffpes.types.SpinOrbitalProjection
        )

        projections = jnp.ones((2, 2, 3, 9), dtype=jnp.float64)
        orb = make_orbital_projection(projections=projections)
        prepared = prepare_projection(
            orb_proj=orb,
            atom_indices=[0, 2],
            attach_oam=True,
        )
        chex.assert_shape(prepared.projections, (2, 2, 2, 9))
        assert prepared.oam is not None
        chex.assert_shape(prepared.oam, (2, 2, 2, 3))


class TestRunVaspWorkflow:
    """Validate :func:`diffpes.simul.run_vasp_workflow`.

    :see: :func:`~diffpes.simul.run_vasp_workflow`
    """

    def test_fixture_round_trip_keeps_explicit_h_authority(
        self,
        monkeypatch: pytest.MonkeyPatch,
        workflow_carriers: Dict[str, Any],
    ) -> None:
        """Keep the explicit Hamiltonian and every coherent carrier unchanged.

        The test loads the committed EIGENVAL, PROCAR, and incomplete KPOINTS
        fixtures, intercepts the canonical cut call, and checks its complete
        typed argument surface. It also requires the PROCAR phase-loss warning.

        Notes
        -----
        The empty second KPOINTS label makes the plotting metadata invalid, so
        the physical EIGENVAL path must survive with no labels.
        """
        sentinel: DetectorRaster = make_detector_raster(
            expected_counts=jnp.ones((1, 2, 1, 3)),
            detector_u_axis=jnp.asarray([0.015, 0.13]),
            detector_v_axis=jnp.asarray([0.0]),
            energy_axis=jnp.asarray([-0.2, 0.0, 0.2]),
            channel_labels=("total",),
            coordinate_system="hemispherical_angles",
        )
        canonical_call: Mock = Mock(return_value=sentinel)
        monkeypatch.setattr(
            workflow,
            "simulate_arpes_cut",
            canonical_call,
        )

        with pytest.warns(RuntimeWarning, match="phase"):
            result: DetectorRaster = workflow.run_vasp_workflow(
                workflow_carriers["hamiltonians"],
                crystal_geometry=workflow_carriers["crystal"],
                orbital_basis=workflow_carriers["basis"],
                radial_spec=workflow_carriers["radial"],
                matrix_element_params=workflow_carriers["matrix_params"],
                radial_quadrature=workflow_carriers["quadrature"],
                final_state=workflow_carriers["final_state"],
                experiment_geometry=workflow_carriers["experiment"],
                self_energy=workflow_carriers["self_energy"],
                energy_axis=workflow_carriers["energy_axis"],
                detector_calibration=workflow_carriers["calibration"],
                detector_effects=workflow_carriers["effects"],
                directory=str(_FIXTURES_DIR),
                eigenval_file="EIGENVAL_spin",
                procar_file="PROCAR_spin",
                doscar_file=None,
                kpoints_file="KPOINTS_line_fallback",
                fermi_energy=0.0,
                phase_loss="warn",
                k_chunk=2,
                energy_chunk=3,
                checkpoint=False,
            )

        assert result is sentinel
        canonical_call.assert_called_once()
        forwarded: Dict[str, Any] = canonical_call.call_args.kwargs
        hamiltonians_by_domain: Tuple[Any, ...] = forwarded[
            "hamiltonians_by_domain"
        ]
        assert len(hamiltonians_by_domain) == 1
        assert hamiltonians_by_domain[0] is workflow_carriers["hamiltonians"]
        assert forwarded["radial_spec"] is workflow_carriers["radial"]
        assert (
            forwarded["matrix_element_params"]
            is workflow_carriers["matrix_params"]
        )
        assert (
            forwarded["radial_quadrature"] is workflow_carriers["quadrature"]
        )
        assert forwarded["final_state"] is workflow_carriers["final_state"]
        assert forwarded["geometry"] is workflow_carriers["experiment"]
        assert forwarded["self_energy"] is workflow_carriers["self_energy"]
        assert forwarded["energy_axis"] is workflow_carriers["energy_axis"]
        assert (
            forwarded["detector_calibration"]
            is workflow_carriers["calibration"]
        )
        assert forwarded["detector_effects"] is workflow_carriers["effects"]
        assert forwarded["k_chunk"] == 2
        assert forwarded["energy_chunk"] == 3
        assert forwarded["checkpoint"] is False
        path: KPath = cast(KPath, forwarded["kpath"])
        assert path.labels == ()
        assert path.label_indices == ()
        chex.assert_trees_all_close(
            path.kpoints,
            jnp.asarray([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]]),
            atol=0.0,
            rtol=0.0,
        )
        assert path.kz is not None
        chex.assert_trees_all_close(path.kz, jnp.asarray(0.0), atol=1e-12)

    @pytest.mark.slow
    @pytest.mark.rss_limit_mb(1024)
    def test_committed_fixture_reaches_finite_detector_counts(
        self,
        workflow_carriers: Dict[str, Any],
    ) -> None:
        """Run miniature VASP files through the real effects chain.

        The test combines the parsed two-point path with a supplied Hermitian
        raster. It checks native detector shape, finiteness, and count sign.

        Notes
        -----
        Small momentum and energy chunks exercise the production padding seam
        while keeping this end-to-end check bounded.
        """
        result: DetectorRaster = workflow.run_vasp_workflow(
            workflow_carriers["hamiltonians"],
            crystal_geometry=workflow_carriers["crystal"],
            orbital_basis=workflow_carriers["basis"],
            radial_spec=workflow_carriers["radial"],
            matrix_element_params=workflow_carriers["matrix_params"],
            radial_quadrature=workflow_carriers["quadrature"],
            final_state=workflow_carriers["final_state"],
            experiment_geometry=workflow_carriers["experiment"],
            self_energy=workflow_carriers["self_energy"],
            energy_axis=workflow_carriers["energy_axis"],
            detector_calibration=workflow_carriers["calibration"],
            detector_effects=workflow_carriers["effects"],
            directory=str(_FIXTURES_DIR),
            eigenval_file="EIGENVAL_spin",
            procar_file="PROCAR_spin",
            doscar_file=None,
            kpoints_file="KPOINTS_line_fallback",
            fermi_energy=0.0,
            phase_loss="ignore",
            k_chunk=2,
            energy_chunk=3,
            checkpoint=False,
        )
        counts: Float64[Array, "1 2 1 3"] = result.expected_counts
        chex.assert_shape(counts, (1, 2, 1, 3))
        assert bool(jnp.all(jnp.isfinite(counts)))
        assert bool(jnp.all(counts >= 0.0))

    def test_valid_kpoints_labels_survive_fixture_loading(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        workflow_carriers: Dict[str, Any],
    ) -> None:
        """Retain verified reciprocal line-mode labels on the generated path.

        The test copies the committed miniature EIGENVAL and PROCAR files and
        supplies a matching two-anchor KPOINTS file. It intercepts the cut
        driver and inspects the self-describing KPath metadata.

        Notes
        -----
        Both anchor coordinates and their calculated indices agree with the
        EIGENVAL path, so the workflow keeps both labels.
        """
        filename: str
        for filename in ("EIGENVAL_spin", "PROCAR_spin"):
            (tmp_path / filename).write_bytes(
                (_FIXTURES_DIR / filename).read_bytes()
            )
        (tmp_path / "KPOINTS").write_text(
            "k-path\n"
            "2\n"
            "Line-mode\n"
            "Reciprocal\n"
            "0.0 0.0 0.0 ! G\n"
            "0.5 0.0 0.0 ! X\n",
            encoding="utf-8",
        )
        sentinel: DetectorRaster = make_detector_raster(
            expected_counts=jnp.ones((1, 2, 1, 3)),
            detector_u_axis=jnp.asarray([0.015, 0.13]),
            detector_v_axis=jnp.asarray([0.0]),
            energy_axis=jnp.asarray([-0.2, 0.0, 0.2]),
            channel_labels=("total",),
            coordinate_system="hemispherical_angles",
        )
        canonical_call: Mock = Mock(return_value=sentinel)
        monkeypatch.setattr(
            workflow,
            "simulate_arpes_cut",
            canonical_call,
        )

        result: DetectorRaster = workflow.run_vasp_workflow(
            workflow_carriers["hamiltonians"],
            crystal_geometry=workflow_carriers["crystal"],
            orbital_basis=workflow_carriers["basis"],
            radial_spec=workflow_carriers["radial"],
            matrix_element_params=workflow_carriers["matrix_params"],
            radial_quadrature=workflow_carriers["quadrature"],
            final_state=workflow_carriers["final_state"],
            experiment_geometry=workflow_carriers["experiment"],
            self_energy=workflow_carriers["self_energy"],
            energy_axis=workflow_carriers["energy_axis"],
            detector_calibration=workflow_carriers["calibration"],
            detector_effects=workflow_carriers["effects"],
            directory=str(tmp_path),
            eigenval_file="EIGENVAL_spin",
            procar_file="PROCAR_spin",
            doscar_file=None,
            kpoints_file="KPOINTS",
            fermi_energy=0.0,
            phase_loss="ignore",
            checkpoint=False,
        )

        assert result is sentinel
        forwarded: Dict[str, Any] = canonical_call.call_args.kwargs
        path: KPath = cast(KPath, forwarded["kpath"])
        assert path.labels == ("G", "X")
        assert path.label_indices == (0, 1)
        assert path.n_per_segment == 2

    def test_rejects_explicit_h_kpoint_mismatch(
        self,
        workflow_carriers: Dict[str, Any],
    ) -> None:
        """Reject a Hamiltonian raster that disagrees with parsed EIGENVAL K.

        The test supplies three Hamiltonians beside the committed two-point
        VASP fixture and checks the boundary diagnostic before simulation.

        Notes
        -----
        Reconstructing a Hamiltonian from VASP eigenpairs cannot fix this
        mismatch; the explicit tensor remains mandatory.
        """
        mismatched: Complex128[Array, "3 1 1"] = jnp.zeros(
            (3, 1, 1), dtype=jnp.complex128
        )
        with pytest.raises(ValueError, match="VASP n_k"):
            workflow.run_vasp_workflow(
                mismatched,
                crystal_geometry=workflow_carriers["crystal"],
                orbital_basis=workflow_carriers["basis"],
                radial_spec=workflow_carriers["radial"],
                matrix_element_params=workflow_carriers["matrix_params"],
                radial_quadrature=workflow_carriers["quadrature"],
                final_state=workflow_carriers["final_state"],
                experiment_geometry=workflow_carriers["experiment"],
                self_energy=workflow_carriers["self_energy"],
                energy_axis=workflow_carriers["energy_axis"],
                detector_calibration=workflow_carriers["calibration"],
                detector_effects=workflow_carriers["effects"],
                directory=str(_FIXTURES_DIR),
                eigenval_file="EIGENVAL_spin",
                procar_file="PROCAR_spin",
                doscar_file=None,
                kpoints_file="KPOINTS_line_fallback",
                fermi_energy=0.0,
                phase_loss="ignore",
            )
