"""Validate the spectrum module.

The cases use analytic values, invariants, and finite differences.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest
from beartype.typing import Any, Dict, List, Tuple
from jaxtyping import Array, Complex128, Float64

from diffpes.constants import (
    K_PREFACTOR_INV_ANG_SQRT_EV,
    KB_EV_PER_K,
)
from diffpes.matrixel import (
    assemble_orbital_transition_channels,
    transition_source,
)
from diffpes.simul import (
    contract_experiment_polarization,
    effects,
    spectrum,
)
from diffpes.types import (
    ArpesCube,
    ArpesSpectrum,
    DetectorCalibration,
    DetectorEffects,
    DetectorRaster,
    DiagonalizedBands,
    ExperimentGeometry,
    KGrid,
    KPath,
    make_arpes_cube,
    make_detector_calibration,
    make_diagonalized_bands,
    make_experiment_geometry,
    make_kgrid,
    make_kpath,
)
from tests._gradients import (
    assert_grad_matches_fd,
    assert_gradients_match_finite_differences,
)

from ._spectrum_helpers import (
    _fixture,
    _identity_detector_chain,
    _physical_cube_fixture,
)


class TestSimulateArpes:
    """Verify :func:`diffpes.simul.simulate_arpes`.

    The cases check source padding, explicit Hamiltonians, detector-chain
    ownership, Cartesian-grid rejection, and registered geometry derivatives.
    """

    def test_source_cube_pads_without_changing_physical_axes(self) -> None:
        """Keep arbitrary physical sizes while scanning static padded chunks.

        The internal scan pads four k points and five energies to six and
        eight.  Only the physical 2x2x5 cube reaches the carrier.

        Notes
        -----
        Compare the returned axes, shape, provenance, and value domain.
        """
        fixture: Dict[str, object] = _fixture()
        cube: ArpesCube = _physical_cube_fixture(fixture)[0]
        assert cube.intensity.shape == (2, 2, 5)
        assert jnp.all(jnp.isfinite(cube.intensity))
        assert jnp.all(cube.intensity >= 0.0)
        assert jnp.array_equal(cube.kx_axis, jnp.asarray([0.025, 0.13]))
        assert jnp.array_equal(cube.ky_axis, jnp.asarray([-0.04, 0.075]))
        assert cube.provenance == "simulate_arpes/domain=0/single-kz"

    @pytest.mark.slow
    def test_explicit_hamiltonian_owns_the_resolvent_value(self) -> None:
        """Verify H ownership with deliberately stale eigensystem metadata.

        Silent H reconstruction from ``bands.eigenvalues`` returns the same
        cube and fails this ownership check.

        Notes
        -----
        Shift only the explicit Hamiltonian and compare both physical cubes.
        """
        fixture: Dict[str, object] = _fixture()
        baseline: Float64[Array, "..."] = _physical_cube_fixture(fixture)[
            0
        ].intensity
        shifted_hamiltonians: Any = fixture["hamiltonians"] + (
            0.11 * jnp.ones_like(fixture["hamiltonians"])
        )
        shifted: Float64[Array, "..."] = _physical_cube_fixture(
            fixture, hamiltonians=shifted_hamiltonians
        )[0].intensity
        assert not jnp.allclose(baseline, shifted, rtol=1.0e-6, atol=1.0e-9)

    @pytest.mark.slow
    def test_calls_one_shared_detector_chain(self, monkeypatch: Any) -> None:
        """Verify exactly one downstream composition for a physical cube.

        The test double preserves source values and records one invocation; it
        cannot introduce a tier-specific occupation or convolution branch.

        Notes
        -----
        Replace only the detector seam and compare its captured source values.
        """
        fixture: Dict[str, object] = _fixture()
        calls: List[Tuple[ArpesCube | ArpesSpectrum, ...]] = []

        def record_and_apply(
            physical_by_domain: Tuple[ArpesCube | ArpesSpectrum, ...],
            geometry: ExperimentGeometry,
            calibration: DetectorCalibration,
            effects: DetectorEffects,
        ) -> DetectorRaster:
            calls.append(physical_by_domain)
            returned: DetectorRaster = _identity_detector_chain(
                physical_by_domain, geometry, calibration, effects
            )
            return returned

        monkeypatch.setattr(
            effects,
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

    @pytest.mark.slow
    def test_rejects_nonseparable_cartesian_grid(self) -> None:
        """Reject a rotated raster that one-dimensional axes cannot encode.

        The planted grid keeps its static 2x2 shape but mixes x into the row
        direction, exposing any source-array relabeling shortcut.

        Notes
        -----
        Rotate both declared grid points and matching band metadata.
        """
        fixture: Dict[str, object] = _fixture()
        points: Float64[Array, "..."] = fixture["kgrid"].kpoints
        angle: float = 0.23
        rotation: Float64[Array, "..."] = jnp.asarray(
            [
                [jnp.cos(angle), -jnp.sin(angle), 0.0],
                [jnp.sin(angle), jnp.cos(angle), 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        rotated_points: Float64[Array, "..."] = points @ rotation.T
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
            _physical_cube_fixture(rotated_fixture)

    @pytest.mark.slow
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
        hamiltonians: Complex128[Array, "..."] = (
            jnp.asarray(fixture["hamiltonians"]).at[0, 0, 0].set(2.0 + 0.0j)
        )
        energy_axis: Float64[Array, "4"] = jnp.asarray(
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
        weights: Float64[Array, "..."] = jnp.linspace(0.3, 1.7, 20).reshape(
            (1, 2, 2, 5)
        )

        def rates(coordinates: Float64[Array, "..."]) -> Float64[Array, "..."]:
            """Compute one real-chain expected-count raster."""
            hnu: Float64[Array, "..."]
            temperature: Float64[Array, "..."]
            theta: Float64[Array, "..."]
            phi: Float64[Array, "..."]
            azimuth: Float64[Array, "..."]
            work: Float64[Array, "..."]
            mfp: Float64[Array, "..."]
            jones_r: Float64[Array, "..."]
            jones_i: Float64[Array, "..."]
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
            s_axis: Complex128[Array, "..."] = jnp.asarray(
                [-jnp.sin(phi), jnp.cos(phi), 0.0], dtype=jnp.complex128
            )
            p_axis: Complex128[Array, "..."] = jnp.asarray(
                [
                    jnp.cos(theta) * jnp.cos(phi),
                    jnp.cos(theta) * jnp.sin(phi),
                    -jnp.sin(theta),
                ],
                dtype=jnp.complex128,
            )
            polarization: Complex128[Array, "..."] = (
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
            value: Float64[Array, "..."] = raster.expected_counts
            return value

        def loss(coordinates: Float64[Array, "..."]) -> Float64[Array, "..."]:
            """Compute one weighted real-chain detector-count loss."""
            value: Float64[Array, "..."] = jnp.sum(
                rates(coordinates) * weights
            )
            return value

        coordinates: Float64[Array, "9"] = jnp.asarray(
            [50.0, 15.0, 0.36, -0.29, 0.17, 4.5, 8.0, 0.41, -0.27]
        )
        baseline: Float64[Array, "..."] = loss(coordinates)
        assert jnp.isfinite(baseline)

        def high_energy_count(
            temperature: Float64[Array, "..."],
        ) -> Float64[Array, "..."]:
            """Extract the two-eV detector-bin count at one temperature."""
            candidate: Float64[Array, "..."] = coordinates.at[1].set(
                temperature
            )
            value: Float64[Array, "..."] = jnp.sum(rates(candidate)[..., -1])
            return value

        high_value: Float64[Array, "..."] = high_energy_count(coordinates[1])
        high_derivative: Float64[Array, "..."] = jax.grad(high_energy_count)(
            coordinates[1]
        )
        assert jnp.isfinite(high_value)
        assert jnp.isfinite(high_derivative)
        assert_grad_matches_fd(
            high_energy_count, coordinates[1], regime="smooth"
        )
        assert_gradients_match_finite_differences(
            loss,
            coordinates,
            regime="smooth",
            elementwise=True,
        )
        coordinate_gradient: Float64[Array, "..."] = jax.grad(loss)(
            coordinates
        )
        assert coordinate_gradient[0] != 0.0
        assert jnp.allclose(
            coordinate_gradient[0],
            -coordinate_gradient[5],
            rtol=1.0e-12,
            atol=1.0e-13,
        )
        compiled: Float64[Array, "..."] = jax.jit(loss)(coordinates)
        assert jnp.allclose(compiled, baseline, rtol=1.0e-10, atol=0.0)
        perturbation: Float64[Array, "9"] = jnp.asarray(
            [0.01, 0.02, 0.001, -0.001, 0.001, 0.001, 0.002, 0.001, -0.001]
        )
        batch: Float64[Array, "..."] = jnp.stack(
            (coordinates, coordinates + perturbation)
        )
        vectorized: Float64[Array, "..."] = jax.jit(jax.vmap(loss))(batch)
        sequential: Float64[Array, "..."] = jnp.stack(
            (loss(coordinates), loss(coordinates + perturbation))
        )
        assert jnp.allclose(vectorized, sequential, rtol=1.0e-10, atol=0.0)


class TestSimulateArpesCut:
    """Verify :func:`diffpes.simul.simulate_arpes_cut`.

    The case supplies a complete Cartesian momentum path and requires the cut
    driver to preserve every path coordinate.
    """

    def test_preserves_complete_cartesian_path(self, monkeypatch: Any) -> None:
        """Carry both cumulative distance and full vectors into the map seam.

        The path is not axis aligned, so dropping the vectors loses physical
        information even though cumulative distance remains valid.

        Notes
        -----
        Capture the source carrier and compare both geometric representations.
        """
        fixture: Dict[str, object] = _fixture()
        path_points: Float64[Array, "..."] = fixture["kgrid"].kpoints[:3]
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
        captured: List[ArpesSpectrum] = []

        def capture(
            physical_by_domain: Tuple[ArpesCube | ArpesSpectrum, ...],
            geometry: ExperimentGeometry,
            calibration: DetectorCalibration,
            effects: DetectorEffects,
        ) -> DetectorRaster:
            assert isinstance(physical_by_domain[0], ArpesSpectrum)
            captured.append(physical_by_domain[0])
            returned: DetectorRaster = _identity_detector_chain(
                physical_by_domain, geometry, calibration, effects
            )
            return returned

        monkeypatch.setattr(
            effects,
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
        expected_steps: Float64[Array, "..."] = jnp.linalg.norm(
            jnp.diff(captured[0].kpoints_cart_inv_ang, axis=0), axis=-1
        )
        assert jnp.allclose(jnp.diff(captured[0].k_axis), expected_steps)


class TestSimulateHvScan:
    """Verify :func:`diffpes.simul.simulate_hv_scan`.

    The case compares the stacked photon-energy result with repeated calls to
    the public single-energy driver.
    """

    @pytest.mark.slow
    @pytest.mark.big_mem
    @pytest.mark.rss_limit_mb(1800)
    def test_stacks_public_native_single_hv_calls(self) -> None:
        """Compare native spectra along the caller-owned photon-energy axis.

        The two-row public scan must equal two independent one-row public
        calls while preserving explicit Hamiltonian ownership and axis order.

        Notes
        -----
        Compare complete float64 values and the declared hnu-k-energy shape.
        """
        fixture: Dict[str, object] = _fixture()
        path_points: Float64[Array, "..."] = fixture["kgrid"].kpoints[:2]
        path: KPath = make_kpath(path_points, n_per_segment=1, kz=0.0)
        bands: DiagonalizedBands = fixture["bands"]
        path_bands: DiagonalizedBands = make_diagonalized_bands(
            bands.eigenvalues[:2],
            bands.eigenvectors[:2],
            path_points,
            bands.geometry,
            bands.basis,
            fermi_energy=bands.fermi_energy,
            depths=bands.depths,
        )
        photon_energies: Float64[Array, "2"] = jnp.asarray((48.0, 52.0))
        scan: Float64[Array, "..."] = spectrum.simulate_hv_scan(
            fixture["hamiltonians"][:2],
            path_bands,
            fixture["radial"],
            fixture["matrix_params"],
            fixture["quadrature"],
            fixture["final_state"],
            fixture["geometry"],
            fixture["self_energy"],
            path,
            fixture["energy_axis"],
            photon_energies,
            k_chunk=2,
            energy_chunk=3,
            checkpoint=False,
        )
        expected: Float64[Array, "..."] = jnp.stack(
            tuple(
                spectrum.simulate_hv_scan(
                    fixture["hamiltonians"][:2],
                    path_bands,
                    fixture["radial"],
                    fixture["matrix_params"],
                    fixture["quadrature"],
                    fixture["final_state"],
                    fixture["geometry"],
                    fixture["self_energy"],
                    path,
                    fixture["energy_axis"],
                    photon_energy[None],
                    k_chunk=2,
                    energy_chunk=3,
                    checkpoint=False,
                )[0]
                for photon_energy in photon_energies
            )
        )

        assert scan.shape == (2, 2, 5)
        assert jnp.allclose(scan, expected, rtol=1.0e-12, atol=1.0e-14)


class TestHvMapAtEnergy:
    """Verify :func:`diffpes.simul.hv_map_at_energy`.

    The case interpolates one energy plane from a photon-energy scan and checks
    the documented output-axis transpose.
    """

    def test_interpolates_energy_and_transposes_axes(self) -> None:
        """Interpolate the sampled energy axis into a k-by-hnu map.

        The query lies strictly between two bins and exposes both interpolation
        order and the required output-axis transpose.

        Notes
        -----
        Compare the public helper and its compiled value with linear algebra.
        """
        energy_axis: Float64[Array, "4"] = jnp.asarray((-0.4, -0.2, 0.1, 0.3))
        scan: Float64[Array, "..."] = jnp.arange(24.0).reshape((3, 2, 4)) + 1.0
        query: Float64[Array, ""] = jnp.asarray(-0.08)
        fraction: Float64[Array, "..."] = (query - energy_axis[1]) / (
            energy_axis[2] - energy_axis[1]
        )
        expected: Float64[Array, "..."] = (
            (1.0 - fraction) * scan[:, :, 1] + fraction * scan[:, :, 2]
        ).T
        actual: Float64[Array, "..."] = spectrum.hv_map_at_energy(
            scan, energy_axis, query
        )
        compiled: Float64[Array, "..."] = jax.jit(spectrum.hv_map_at_energy)(
            scan, energy_axis, query
        )

        assert actual.shape == (2, 3)
        assert jnp.allclose(actual, expected, rtol=1.0e-13, atol=1.0e-14)
        assert jnp.allclose(compiled, actual, rtol=1.0e-13, atol=1.0e-14)


class TestNormalizeIntensity:
    """Verify :func:`diffpes.simul.normalize_intensity`.

    The cases check every normalization mode returns a plain array and reject
    a singular display normalization.
    """

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
        normalized_sum: Float64[Array, "..."] = spectrum.normalize_intensity(
            cube, "sum"
        )
        normalized_zscore: Float64[Array, "..."] = (
            spectrum.normalize_intensity(cube, "zscore")
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
        with pytest.raises(
            eqx.EquinoxRuntimeError,
            match="normalization requires nonzero",
        ):
            spectrum.normalize_intensity(cube, mode)


@pytest.mark.slow
@pytest.mark.big_mem
@pytest.mark.rss_limit_mb(800)
def test_analytic_one_level_occupation_convention() -> None:
    """Match a direct one-level Lorentzian times sampled Fermi occupation.

    The independent spectral expression consumes matrix-element source norms
    but no production spectral function. The test evaluates occupation at every
    sampled omega.  Replacing it with one band-energy value fails the check.

    Notes
    -----
    Assemble source norms separately and compare the closed-form full cube.
    """
    fixture: Dict[str, object] = _fixture()
    cube: ArpesCube = _physical_cube_fixture(fixture)[0]
    omega: Float64[Array, "..."] = jnp.asarray(fixture["energy_axis"])
    bands: DiagonalizedBands = fixture["bands"]
    geometry: ExperimentGeometry = fixture["geometry"]
    k_cart: Float64[Array, "..."] = bands.kpoints @ bands.geometry.reciprocal
    source_weights: List[Float64[Array, "..."]] = []
    energy: Float64[Array, "..."]
    for energy in omega:
        kinetic_energy: Float64[Array, "..."] = (
            geometry.photon_energy_ev - geometry.work_function_ev + energy
        )
        final_norm: Float64[Array, "..."] = (
            K_PREFACTOR_INV_ANG_SQRT_EV * jnp.sqrt(kinetic_energy)
        )
        final_kz: Float64[Array, "..."] = jnp.sqrt(
            final_norm**2 - jnp.sum(k_cart[:, :2] ** 2, axis=-1)
        )
        final_momentum: Float64[Array, "..."] = jnp.stack(
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
        rows: Float64[Array, "..."] = contract_experiment_polarization(
            channels, geometry
        )
        sources: Complex128[Array, "..."] = transition_source(rows)
        source_weights.append(
            jnp.sum(jnp.real(jnp.conj(sources) * sources), axis=(-2, -1))
        )
    weights: Float64[Array, "..."] = jnp.stack(source_weights, axis=-1)
    band_energy: Float64[Array, "..."] = jnp.real(
        fixture["hamiltonians"][:, 0, 0]
    )
    width: float = 0.04 + 1.0e-4
    lorentzian: Float64[Array, "..."] = (
        weights
        * width
        / jnp.pi
        / ((omega[None, :] - band_energy[:, None]) ** 2 + width**2)
    )
    occupation: Float64[Array, "..."] = jax.nn.sigmoid(
        -omega / (KB_EV_PER_K * fixture["geometry"].temperature_k)
    )
    expected: Float64[Array, "..."] = (
        jnp.transpose(lorentzian.reshape((2, 2, omega.shape[0])), (1, 0, 2))
        * occupation[None, None, :]
    )
    assert jnp.allclose(
        cube.intensity,
        expected,
        rtol=1.0e-12,
        atol=1.0e-14,
    )


def _make_ramp_cube() -> ArpesCube:
    """PRIVATE: Create a small cube whose intensity is linear in energy.

    Returns
    -------
    ArpesCube
        Cube with intensity ``|E|`` on a five-by-five raster and a
        seven-node energy axis from -0.5 eV to 0.1 eV.
    """
    kx_axis: Float64[Array, " 5"] = jnp.linspace(-0.2, 0.2, 5)
    energy_axis: Float64[Array, " 7"] = jnp.linspace(-0.5, 0.1, 7)
    intensity: Float64[Array, "5 5 7"] = jnp.broadcast_to(
        jnp.abs(energy_axis), (5, 5, 7)
    )
    cube: ArpesCube = make_arpes_cube(intensity, kx_axis, kx_axis, energy_axis)
    return cube


class TestConstantEnergySlice:
    """Verify :func:`diffpes.simul.constant_energy_slice`.

    The cases check linear interpolation between sampled energies and the
    out-of-domain guard.
    """

    def test_interpolates_between_energy_nodes(self) -> None:
        """Interpolate the ramp cube midway between two nodes.

        The intensity equals ``|E|`` everywhere, so the slice at
        -0.25 eV holds 0.25 on every pixel.

        Notes
        -----
        Compare the eager and compiled values with the closed form.
        """
        cube: ArpesCube = _make_ramp_cube()
        actual: Float64[Array, "5 5"] = spectrum.constant_energy_slice(
            cube, -0.25
        )
        compiled: Float64[Array, "5 5"] = jax.jit(
            spectrum.constant_energy_slice
        )(cube, -0.25)
        assert actual.shape == (5, 5)
        assert jnp.allclose(actual, 0.25, rtol=1.0e-13, atol=1.0e-14)
        assert jnp.allclose(compiled, actual, rtol=1.0e-13, atol=1.0e-14)

    def test_out_of_domain_query_raises(self) -> None:
        """Reject a query outside the sampled cube domain.

        The query sits above the last sampled energy, so the traced
        domain guard trips.

        Notes
        -----
        The guard trips before any interpolation output forms.
        """
        cube: ArpesCube = _make_ramp_cube()
        with pytest.raises(
            eqx.EquinoxRuntimeError,
            match="the energy query must lie inside the sampled cube domain",
        ):
            spectrum.constant_energy_slice(cube, 0.5)


class TestEnergyWindowMap:
    """Verify :func:`diffpes.simul.energy_window_map`.

    The cases check trapezoid integration over interior segments and the
    empty-window guard.
    """

    def test_integrates_interior_segments(self) -> None:
        """Integrate the ramp cube over an interior window.

        The window from -0.4 eV to -0.1 eV covers three complete
        segments of the ``|E|`` ramp. The trapezoid sum equals 0.075 eV
        times the unit map.

        Notes
        -----
        Compare the eager and compiled values with the closed form.
        """
        cube: ArpesCube = _make_ramp_cube()
        actual: Float64[Array, "5 5"] = spectrum.energy_window_map(
            cube, -0.4, -0.1
        )
        compiled: Float64[Array, "5 5"] = jax.jit(spectrum.energy_window_map)(
            cube, -0.4, -0.1
        )
        assert actual.shape == (5, 5)
        assert jnp.allclose(actual, 0.075, rtol=1.0e-12, atol=1.0e-14)
        assert jnp.allclose(compiled, actual, rtol=1.0e-12, atol=1.0e-14)

    def test_empty_window_raises(self) -> None:
        """Reject a window without one complete sampled segment.

        The traced guard trips because the window covers no complete
        sampled segment.

        Notes
        -----
        The window sits between two adjacent nodes, so no segment has
        both ends inside it.
        """
        cube: ArpesCube = _make_ramp_cube()
        with pytest.raises(
            eqx.EquinoxRuntimeError,
            match="the energy window must cover at least one sampled segment",
        ):
            spectrum.energy_window_map(cube, -0.29, -0.21)
