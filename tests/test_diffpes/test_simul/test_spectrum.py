"""Verify the coherent single-kz source and detector driver.

The tests cover Hamiltonian ownership, grids, streaming, detector composition,
display normalization, and real-chain geometry derivatives.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest
from beartype.typing import Any, Dict, Tuple

import diffpes.simul.spectrum as spectrum
from diffpes.simul.matrixel import (
    assemble_orbital_transition_channels,
    contract_experiment_polarization,
    transition_source,
)
from diffpes.types import (
    K_PREFACTOR_INV_ANG_SQRT_EV,
    KB_EV_PER_K,
    ArpesCube,
    ArpesSpectrum,
    DetectorCalibration,
    DetectorEffects,
    DetectorRaster,
    DiagonalizedBands,
    ExperimentGeometry,
    FinalStateSpec,
    KGrid,
    KPath,
    MatrixElementParams,
    RadialQuadratureSpec,
    RadialSpec,
    SelfEnergyModel,
    make_arpes_cube,
    make_crystal_geometry,
    make_detector_calibration,
    make_detector_effects,
    make_detector_raster,
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
)
from tests._gradients import assert_grad_matches_fd, gradient_gate

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
    kx: jax.Array = jnp.asarray([0.025, 0.13], dtype=jnp.float64)
    ky: jax.Array = jnp.asarray([-0.04, 0.075], dtype=jnp.float64)
    mesh_x: jax.Array
    mesh_y: jax.Array
    mesh_x, mesh_y = jnp.meshgrid(kx, ky, indexing="xy")
    kpoints: jax.Array = jnp.stack(
        (mesh_x, mesh_y, jnp.zeros_like(mesh_x)), axis=-1
    ).reshape((-1, 3))
    eigenvalues: jax.Array = -0.08 + 0.7 * jnp.sum(
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
    hamiltonians: jax.Array = eigenvalues[:, None, None].astype(jnp.complex128)
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
    energy_axis: jax.Array = jnp.asarray(
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
    return {
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


def _physical_cubes(
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
    cubes: Tuple[ArpesCube, ...] = spectrum._physical_cubes(
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

    This test-only seam isolates the WP8.5 source driver from detector-map
    correctness, which has its own G4b/G8 module.

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
        return make_detector_raster(
            source.intensity[None, ...],
            source.kx_axis,
            source.ky_axis,
            source.energy_axis,
            ("total",),
            "hemispherical_angles",
        )
    return make_detector_raster(
        source.intensity[None, :, None, :],
        source.k_axis,
        jnp.zeros((1,), dtype=jnp.float64),
        source.energy_axis,
        ("total",),
        "hemispherical_angles",
    )


class TestSimulateArpes:
    """Verify :func:`diffpes.simul.simulate_arpes`."""

    def test_g4_source_cube_pads_without_changing_physical_axes(self) -> None:
        """Keep arbitrary physical sizes while scanning static padded chunks.

        The internal scan pads four k points and five energies to six and
        eight.  Only the physical 2x2x5 cube reaches the carrier.

        Notes
        -----
        Compare the returned axes, shape, provenance, and value domain.
        """
        fixture: Dict[str, object] = _fixture()
        cube: ArpesCube = _physical_cubes(fixture)[0]
        assert cube.intensity.shape == (2, 2, 5)
        assert jnp.all(jnp.isfinite(cube.intensity))
        assert jnp.all(cube.intensity >= 0.0)
        assert jnp.array_equal(cube.kx_axis, jnp.asarray([0.025, 0.13]))
        assert jnp.array_equal(cube.ky_axis, jnp.asarray([-0.04, 0.075]))
        assert cube.provenance == "simulate_arpes/domain=0/single-kz"

    def test_explicit_hamiltonian_owns_the_resolvent_value(self) -> None:
        """Verify H ownership with deliberately stale eigensystem metadata.

        Silent H reconstruction from ``bands.eigenvalues`` returns the same
        cube and fails this ownership gate.

        Notes
        -----
        Shift only the explicit Hamiltonian and compare both physical cubes.
        """
        fixture: Dict[str, object] = _fixture()
        baseline: jax.Array = _physical_cubes(fixture)[0].intensity
        shifted_hamiltonians: Any = fixture["hamiltonians"] + (
            0.11 * jnp.ones_like(fixture["hamiltonians"])
        )
        shifted: jax.Array = _physical_cubes(
            fixture, hamiltonians=shifted_hamiltonians
        )[0].intensity
        assert not jnp.allclose(baseline, shifted, rtol=1.0e-6, atol=1.0e-9)

    def test_calls_one_shared_detector_chain(self, monkeypatch: Any) -> None:
        """Verify exactly one downstream composition for a physical cube.

        The test double preserves source values and records one invocation; it
        cannot introduce a tier-specific occupation or convolution branch.

        Notes
        -----
        Replace only the detector seam and compare its captured source values.
        """
        fixture: Dict[str, object] = _fixture()
        calls: list[Tuple[ArpesCube | ArpesSpectrum, ...]] = []

        def record_and_apply(
            physical_by_domain: Tuple[ArpesCube | ArpesSpectrum, ...],
            geometry: ExperimentGeometry,
            calibration: DetectorCalibration,
            effects: DetectorEffects,
        ) -> DetectorRaster:
            calls.append(physical_by_domain)
            return _identity_detector_chain(
                physical_by_domain, geometry, calibration, effects
            )

        monkeypatch.setattr(
            spectrum._effects,
            "apply_detector_effects",
            record_and_apply,
            raising=False,
        )
        raster: DetectorRaster = spectrum.simulate_arpes(
            (fixture["hamiltonians"],),
            (fixture["bands"],),
            fixture["radial"],
            fixture["matrix_params"],
            fixture["quadrature"],
            fixture["final_state"],
            fixture["geometry"],
            fixture["self_energy"],
            fixture["kgrid"],
            fixture["energy_axis"],
            fixture["calibration"],
            fixture["detector_effects"],
            k_chunk=3,
            energy_chunk=4,
            checkpoint=False,
        )
        assert len(calls) == 1
        assert isinstance(calls[0][0], ArpesCube)
        assert jnp.array_equal(
            raster.expected_counts[0], calls[0][0].intensity
        )

    def test_rejects_nonseparable_cartesian_grid(self) -> None:
        """Reject a rotated raster that one-dimensional axes cannot encode.

        The planted grid keeps its static 2x2 shape but mixes x into the row
        direction, exposing any source-array relabeling shortcut.

        Notes
        -----
        Rotate both declared grid points and matching band metadata.
        """
        fixture: Dict[str, object] = _fixture()
        points: jax.Array = fixture["kgrid"].kpoints
        angle: float = 0.23
        rotation: jax.Array = jnp.asarray(
            [
                [jnp.cos(angle), -jnp.sin(angle), 0.0],
                [jnp.sin(angle), jnp.cos(angle), 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        rotated_points: jax.Array = points @ rotation.T
        rotated_grid: KGrid = make_kgrid(rotated_points, (2, 2), kz=0.0)
        rotated_bands: DiagonalizedBands = eqx.tree_at(
            lambda item: item.kpoints,
            fixture["bands"],
            rotated_points,
        )
        rotated_fixture: Dict[str, object] = dict(fixture)
        rotated_fixture["kgrid"] = rotated_grid
        rotated_fixture["bands"] = rotated_bands
        with pytest.raises(
            eqx.EquinoxRuntimeError, match="separable sample-Cartesian"
        ):
            _physical_cubes(rotated_fixture)

    @pytest.mark.big_mem
    @pytest.mark.rss_limit_mb(3500)
    def test_registered_geometry_coordinates_are_fd_correct(self) -> None:
        """Differentiate every registered geometry coordinate through counts.

        The real detector chain includes a two-eV band sample at 15 kelvin.

        Notes
        -----
        A transverse Jones chart activates both complex polarization
        quadratures, while a strictly interior target keeps mapping smooth.
        """
        fixture: Dict[str, object] = _fixture()
        hamiltonians: jax.Array = (
            jnp.asarray(fixture["hamiltonians"]).at[0, 0, 0].set(2.0 + 0.0j)
        )
        energy_axis: jax.Array = jnp.asarray(
            [-0.006, -0.0015, 0.002, 2.0], dtype=jnp.float64
        )
        calibration: DetectorCalibration = make_detector_calibration(
            u_bin_edges=jnp.asarray([0.012, 0.02, 0.028]),
            v_bin_edges=jnp.asarray([-0.01, 0.0, 0.01]),
            energy_bin_edges_ev=jnp.asarray(
                [-0.0055, -0.0025, 0.0, 0.004, 1.95, 2.05]
            ),
            psf_fwhm_u=0.01,
            psf_fwhm_v=0.01,
            psf_fwhm_energy_ev=0.02,
            transmission_reference_domain_ev=jnp.asarray([40.0, 55.0]),
        )
        weights: jax.Array = jnp.linspace(0.3, 1.7, 20).reshape((1, 2, 2, 5))

        def rates(coordinates: jax.Array) -> jax.Array:
            """Compute one real-chain expected-count raster."""
            hnu: jax.Array
            temperature: jax.Array
            theta: jax.Array
            phi: jax.Array
            azimuth: jax.Array
            work: jax.Array
            mfp: jax.Array
            jones_r: jax.Array
            jones_i: jax.Array
            (
                hnu,
                temperature,
                theta,
                phi,
                azimuth,
                work,
                mfp,
                jones_r,
                jones_i,
            ) = coordinates
            s_axis: jax.Array = jnp.asarray(
                [-jnp.sin(phi), jnp.cos(phi), 0.0], dtype=jnp.complex128
            )
            p_axis: jax.Array = jnp.asarray(
                [
                    jnp.cos(theta) * jnp.cos(phi),
                    jnp.cos(theta) * jnp.sin(phi),
                    -jnp.sin(theta),
                ],
                dtype=jnp.complex128,
            )
            polarization: jax.Array = (
                s_axis + (jones_r + 1.0j * jones_i) * p_axis
            )
            geometry: ExperimentGeometry = make_experiment_geometry(
                photon_energy_ev=hnu,
                polarization=polarization,
                incidence_theta=theta,
                incidence_phi=phi,
                sample_azimuth=azimuth,
                work_function_ev=work,
                temperature_k=temperature,
                mean_free_path_ang=mfp,
            )
            raster: DetectorRaster = spectrum.simulate_arpes(
                (hamiltonians,),
                (fixture["bands"],),
                fixture["radial"],
                fixture["matrix_params"],
                fixture["quadrature"],
                fixture["final_state"],
                geometry,
                fixture["self_energy"],
                fixture["kgrid"],
                energy_axis,
                calibration,
                fixture["detector_effects"],
                k_chunk=2,
                energy_chunk=4,
                checkpoint=False,
            )
            value: jax.Array = raster.expected_counts
            return value

        def loss(coordinates: jax.Array) -> jax.Array:
            """Compute one weighted real-chain detector-count loss."""
            value: jax.Array = jnp.sum(rates(coordinates) * weights)
            return value

        coordinates: jax.Array = jnp.asarray(
            [50.0, 15.0, 0.36, -0.29, 0.17, 4.5, 8.0, 0.41, -0.27]
        )
        baseline: jax.Array = loss(coordinates)
        assert jnp.isfinite(baseline)

        def high_energy_count(temperature: jax.Array) -> jax.Array:
            """Extract the two-eV detector-bin count at one temperature."""
            candidate: jax.Array = coordinates.at[1].set(temperature)
            value: jax.Array = jnp.sum(rates(candidate)[..., -1])
            return value

        high_value: jax.Array = high_energy_count(coordinates[1])
        high_derivative: jax.Array = jax.grad(high_energy_count)(
            coordinates[1]
        )
        assert jnp.isfinite(high_value)
        assert jnp.isfinite(high_derivative)
        assert_grad_matches_fd(
            high_energy_count, coordinates[1], regime="smooth"
        )
        gradient_gate(
            loss,
            coordinates,
            regime="smooth",
            elementwise=True,
        )
        coordinate_gradient: jax.Array = jax.grad(loss)(coordinates)
        assert coordinate_gradient[0] != 0.0
        assert jnp.allclose(
            coordinate_gradient[0],
            -coordinate_gradient[5],
            rtol=1.0e-12,
            atol=1.0e-13,
        )
        compiled: jax.Array = jax.jit(loss)(coordinates)
        assert jnp.allclose(compiled, baseline, rtol=1.0e-10, atol=0.0)
        perturbation: jax.Array = jnp.asarray(
            [0.01, 0.02, 0.001, -0.001, 0.001, 0.001, 0.002, 0.001, -0.001]
        )
        batch: jax.Array = jnp.stack((coordinates, coordinates + perturbation))
        vectorized: jax.Array = jax.jit(jax.vmap(loss))(batch)
        sequential: jax.Array = jnp.stack(
            (loss(coordinates), loss(coordinates + perturbation))
        )
        assert jnp.allclose(vectorized, sequential, rtol=1.0e-10, atol=0.0)


class TestSimulateArpesCut:
    """Verify :func:`diffpes.simul.simulate_arpes_cut`."""

    def test_preserves_complete_cartesian_path(self, monkeypatch: Any) -> None:
        """Carry both cumulative distance and full vectors into the map seam.

        The path is not axis aligned, so dropping the vectors loses physical
        information even though cumulative distance remains valid.

        Notes
        -----
        Capture the source carrier and compare both geometric representations.
        """
        fixture: Dict[str, object] = _fixture()
        path_points: jax.Array = fixture["kgrid"].kpoints[:3]
        path: KPath = make_kpath(path_points, n_per_segment=1, kz=0.0)
        bands: DiagonalizedBands = fixture["bands"]
        path_bands: DiagonalizedBands = make_diagonalized_bands(
            bands.eigenvalues[:3],
            bands.eigenvectors[:3],
            path_points,
            bands.geometry,
            bands.basis,
            fermi_energy=bands.fermi_energy,
            depths=bands.depths,
        )
        captured: list[ArpesSpectrum] = []

        def capture(
            physical_by_domain: Tuple[ArpesCube | ArpesSpectrum, ...],
            geometry: ExperimentGeometry,
            calibration: DetectorCalibration,
            effects: DetectorEffects,
        ) -> DetectorRaster:
            assert isinstance(physical_by_domain[0], ArpesSpectrum)
            captured.append(physical_by_domain[0])
            return _identity_detector_chain(
                physical_by_domain, geometry, calibration, effects
            )

        monkeypatch.setattr(
            spectrum._effects,
            "apply_detector_effects",
            capture,
            raising=False,
        )
        raster: DetectorRaster = spectrum.simulate_arpes_cut(
            (fixture["hamiltonians"][:3],),
            (path_bands,),
            fixture["radial"],
            fixture["matrix_params"],
            fixture["quadrature"],
            fixture["final_state"],
            fixture["geometry"],
            fixture["self_energy"],
            path,
            fixture["energy_axis"],
            fixture["calibration"],
            fixture["detector_effects"],
            k_chunk=2,
            energy_chunk=3,
            checkpoint=False,
        )
        assert raster.expected_counts.shape == (1, 3, 1, 5)
        assert len(captured) == 1
        assert jnp.array_equal(
            captured[0].kpoints_cart_inv_ang,
            path_points @ bands.geometry.reciprocal,
        )
        expected_steps: jax.Array = jnp.linalg.norm(
            jnp.diff(captured[0].kpoints_cart_inv_ang, axis=0), axis=-1
        )
        assert jnp.allclose(jnp.diff(captured[0].k_axis), expected_steps)


class TestNormalizeIntensity:
    """Verify :func:`diffpes.simul.normalize_intensity`."""

    def test_returns_plain_arrays_for_all_modes(self) -> None:
        """Normalize without mutating or relabeling the physical carrier.

        Sum and z-score modes use complete-array statistics; none returns the
        original physical array object.

        Notes
        -----
        Compare each statistic and confirm that source intensity stays exact.
        """
        cube: ArpesCube = make_arpes_cube(
            jnp.arange(1.0, 9.0).reshape((2, 2, 2)),
            jnp.asarray([-0.1, 0.1]),
            jnp.asarray([-0.2, 0.2]),
            jnp.asarray([-0.3, 0.1]),
        )
        assert spectrum.normalize_intensity(cube, "none") is cube.intensity
        normalized_sum: jax.Array = spectrum.normalize_intensity(cube, "sum")
        normalized_zscore: jax.Array = spectrum.normalize_intensity(
            cube, "zscore"
        )
        assert jnp.isclose(jnp.sum(normalized_sum), 1.0)
        assert jnp.isclose(jnp.mean(normalized_zscore), 0.0, atol=1.0e-15)
        assert jnp.isclose(jnp.std(normalized_zscore), 1.0)
        assert jnp.array_equal(
            cube.intensity, jnp.arange(1.0, 9.0).reshape((2, 2, 2))
        )

    @pytest.mark.parametrize("mode", ["sum", "zscore"])
    def test_rejects_singular_display_normalization(self, mode: str) -> None:
        """Reject a zero denominator instead of fabricating display values.

        Sum fails on an all-zero array; z-score fails on any constant array.

        Notes
        -----
        Parameterize both singular modes and require the traced runtime guard.
        """
        value: float = 0.0 if mode == "sum" else 2.0
        cube: ArpesCube = make_arpes_cube(
            jnp.full((2, 2, 2), value),
            jnp.asarray([-0.1, 0.1]),
            jnp.asarray([-0.2, 0.2]),
            jnp.asarray([-0.3, 0.1]),
        )
        with pytest.raises(eqx.EquinoxRuntimeError):
            spectrum.normalize_intensity(cube, mode)


@pytest.mark.big_mem
@pytest.mark.rss_limit_mb(800)
def test_analytic_one_level_occupation_convention() -> None:
    """Match a direct one-level Lorentzian times sampled Fermi occupation.

    The independent spectral expression consumes Plan-06 source norms but no
    Plan-07 production function.  The test evaluates occupation at every
    sampled omega.  Replacing it with one band-energy value fails the gate.

    Notes
    -----
    Assemble source norms separately and compare the closed-form full cube.
    """
    fixture: Dict[str, object] = _fixture()
    cube: ArpesCube = _physical_cubes(fixture)[0]
    omega: jax.Array = jnp.asarray(fixture["energy_axis"])
    bands: DiagonalizedBands = fixture["bands"]
    geometry: ExperimentGeometry = fixture["geometry"]
    k_cart: jax.Array = bands.kpoints @ bands.geometry.reciprocal
    source_weights: list[jax.Array] = []
    energy: jax.Array
    for energy in omega:
        kinetic_energy: jax.Array = (
            geometry.photon_energy_ev - geometry.work_function_ev + energy
        )
        final_norm: jax.Array = K_PREFACTOR_INV_ANG_SQRT_EV * jnp.sqrt(
            kinetic_energy
        )
        final_kz: jax.Array = jnp.sqrt(
            final_norm**2 - jnp.sum(k_cart[:, :2] ** 2, axis=-1)
        )
        final_momentum: jax.Array = jnp.stack(
            (k_cart[:, 0], k_cart[:, 1], final_kz), axis=-1
        )
        channels: Any = assemble_orbital_transition_channels(
            bands,
            fixture["radial"],
            fixture["matrix_params"],
            fixture["quadrature"],
            fixture["final_state"],
            geometry,
            final_momentum,
            jnp.ones((k_cart.shape[0],), dtype=jnp.bool_),
        )
        rows: jax.Array = contract_experiment_polarization(channels, geometry)
        sources: jax.Array = transition_source(rows)
        source_weights.append(
            jnp.sum(jnp.real(jnp.conj(sources) * sources), axis=(-2, -1))
        )
    weights: jax.Array = jnp.stack(source_weights, axis=-1)
    band_energy: jax.Array = jnp.real(fixture["hamiltonians"][:, 0, 0])
    width: float = 0.04 + 1.0e-4
    lorentzian: jax.Array = (
        weights
        * width
        / jnp.pi
        / ((omega[None, :] - band_energy[:, None]) ** 2 + width**2)
    )
    occupation: jax.Array = jax.nn.sigmoid(
        -omega / (KB_EV_PER_K * fixture["geometry"].temperature_k)
    )
    expected: jax.Array = (
        jnp.transpose(lorentzian.reshape((2, 2, omega.shape[0])), (1, 0, 2))
        * occupation[None, None, :]
    )
    assert jnp.allclose(
        cube.intensity,
        expected,
        rtol=1.0e-12,
        atol=1.0e-14,
    )
