"""Validate the private detector-cube module.

The cases use analytic values, invariants, and finite differences.
"""

import chex
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from beartype.typing import List, Tuple
from jaxtyping import Array, Float64
from numpy.typing import NDArray

from diffpes.constants import (
    K_PREFACTOR_INV_ANG_SQRT_EV,
)
from diffpes.simul._detector_map import (
    _map_source_to_detector,
    _map_source_to_detector_with_order,
)
from diffpes.types import (
    ArpesCube,
    DetectorCalibration,
    ExperimentGeometry,
    make_arpes_cube,
    make_detector_calibration,
)
from tests._assertions import assert_rejects
from tests._gradients import assert_gradients_match_finite_differences

from ._detector_map_helpers import (
    _cube,
    _geometry,
    _map_calibration,
)


class TestCubeDetectorMap:
    """Verify conservative mapping of a Cartesian source cube.

    The cases check bin mass, boundary behavior, convergence, validation, and
    geometry derivatives against analytic or finite-difference references.
    """

    def test_uses_explicit_target_bins_and_converges_at_eight_points(
        self,
    ) -> None:
        """Preserve explicit unequal bins and converge under quadrature order.

        The target retains every declared native edge.

        Notes
        -----
        The smooth affine fixture lies strictly inside the source support, so
        four- and eight-point rules agree without a boundary crossing.
        """
        source: ArpesCube = _cube()
        geometry: ExperimentGeometry = _geometry()
        calibration: DetectorCalibration = _map_calibration()
        mapped_four: Float64[Array, "u v e"]
        fraction_four: Float64[Array, ""]
        mapped_four, fraction_four = _map_source_to_detector_with_order(
            source, geometry, calibration, order=4
        )
        mapped_eight: Float64[Array, "u v e"]
        fraction_eight: Float64[Array, ""]
        mapped_eight, fraction_eight = _map_source_to_detector_with_order(
            source, geometry, calibration, order=8
        )
        chex.assert_shape(mapped_four, (2, 2, 2))
        chex.assert_trees_all_close(
            mapped_four, mapped_eight, rtol=2.0e-11, atol=1.0e-12
        )
        chex.assert_trees_all_close(
            fraction_four, fraction_eight, rtol=2.0e-11, atol=1.0e-14
        )
        assert 0.0 < float(fraction_four) < 1.0

    def test_exterior_half_cell_matches_analytic_mass_and_stops_at_face(
        self,
    ) -> None:
        """Integrate endpoint density through a half-cell and nowhere beyond.

        The analytic reference integrates the complete detector Jacobian.

        Notes
        -----
        A constant source makes the H-slit change-of-variables integral
        separable.  The accepted target lies wholly between the lower ``kx``
        exterior face and endpoint centre.  A second target lies beyond the
        face and must receive zero density.
        """
        energy_half_span: float = 1.0e-8
        source: ArpesCube = make_arpes_cube(
            intensity=jnp.ones((2, 2, 2)),
            kx_axis=jnp.array([-0.1, 0.1]),
            ky_axis=jnp.array([-1.0, 1.0]),
            energy_axis=jnp.array([-energy_half_span, energy_half_span]),
        )
        kinetic_centre: float = 46.0
        momentum: float = float(
            K_PREFACTOR_INV_ANG_SQRT_EV * np.sqrt(kinetic_centre)
        )
        u_edges: Float64[NDArray, " 2"] = np.arcsin(
            np.array([-0.19, -0.11]) / momentum
        )
        calibration: DetectorCalibration = make_detector_calibration(
            u_bin_edges=jnp.asarray(u_edges),
            v_bin_edges=jnp.array([-0.02, 0.0, 0.02]),
            energy_bin_edges_ev=jnp.array(
                [-2.0 * energy_half_span, 2.0 * energy_half_span]
            ),
            psf_fwhm_u=0.01,
            psf_fwhm_v=0.01,
            psf_fwhm_energy_ev=0.01,
            transmission_reference_domain_ev=jnp.array([40.0, 55.0]),
        )
        density: Float64[Array, "u v e"]
        fraction: Float64[Array, ""]
        density, fraction = _map_source_to_detector(
            source, _geometry(), calibration
        )
        captured_mass: Float64[Array, ""] = jnp.sum(
            density
            * jnp.diff(calibration.u_bin_edges)[:, None, None]
            * jnp.diff(calibration.v_bin_edges)[None, :, None]
            * jnp.diff(calibration.energy_bin_edges_ev)[None, None, :]
        )
        u0: float = float(u_edges[0])
        u1: float = float(u_edges[1])
        v0: float = -0.02
        v1: float = 0.02
        e0: float = -2.0 * energy_half_span
        e1: float = 2.0 * energy_half_span
        energy_integral: float = K_PREFACTOR_INV_ANG_SQRT_EV**2 * (
            kinetic_centre * (e1 - e0) + 0.5 * (e1 * e1 - e0 * e0)
        )
        u_integral: float = 0.5 * (u1 - u0) + 0.25 * (
            np.sin(2.0 * u1) - np.sin(2.0 * u0)
        )
        v_integral: float = np.sin(v1) - np.sin(v0)
        analytic_mass: float = energy_integral * u_integral * v_integral
        chex.assert_trees_all_close(
            captured_mass,
            jnp.asarray(analytic_mass),
            rtol=1.0e-11,
            atol=0.0,
        )
        source_flux: float = 0.4 * 4.0 * (4.0 * energy_half_span)
        chex.assert_trees_all_close(
            fraction,
            jnp.asarray(analytic_mass / source_flux),
            rtol=1.0e-11,
            atol=0.0,
        )

        outside_u_edges: Float64[NDArray, " 2"] = np.arcsin(
            np.array([-0.30, -0.21]) / momentum
        )
        outside_calibration: DetectorCalibration = make_detector_calibration(
            u_bin_edges=jnp.asarray(outside_u_edges),
            v_bin_edges=calibration.v_bin_edges,
            energy_bin_edges_ev=calibration.energy_bin_edges_ev,
            psf_fwhm_u=0.01,
            psf_fwhm_v=0.01,
            psf_fwhm_energy_ev=0.01,
            transmission_reference_domain_ev=jnp.array([40.0, 55.0]),
        )
        outside_density: Float64[Array, "u v e"] = _map_source_to_detector(
            source, _geometry(), outside_calibration
        )[0]
        chex.assert_trees_all_close(
            outside_density,
            jnp.zeros_like(outside_density),
            rtol=0.0,
            atol=0.0,
        )

    def test_projected_rotation_rejects_singular_tilt(self) -> None:
        """Reject a domain plane projected edge-on to the detector.

        The rejection protects the inverse density Jacobian.

        Notes
        -----
        A z--y--z beta angle of pi over two makes the in-plane projected
        rotation singular and cannot define a conservative inverse map.
        """
        assert_rejects(
            _map_source_to_detector_with_order,
            _cube(),
            _geometry(),
            _map_calibration(),
            jnp.array([0.0, 0.5 * jnp.pi, 0.0]),
            match="singular projected",
            order=4,
        )

    @pytest.mark.big_mem
    @pytest.mark.rss_limit_mb(1100)
    def test_sample_azimuth_and_euler_derivatives_match_fd(self) -> None:
        """Match nonzero azimuth/Euler gradients with the shared FD harness.

        Every continuous rotation coordinate remains active.

        Notes
        -----
        Asymmetric target weights prevent integrated-flux conservation from
        hiding the coordinate sensitivities.
        """
        source: ArpesCube = _cube()
        calibration: DetectorCalibration = _map_calibration()
        loss_weights: Float64[Array, "u v e"] = jnp.array(
            [
                [[0.7, -0.2], [0.4, 1.1]],
                [[-0.3, 0.8], [0.5, -0.6]],
            ]
        )

        def loss(
            parameters: Tuple[Float64[Array, ""], Float64[Array, " 3"]],
        ) -> Float64[Array, ""]:
            """Compute a generic weighted scalar from one rotated cube."""
            azimuth: Float64[Array, ""]
            euler_angles: Float64[Array, " 3"]
            azimuth, euler_angles = parameters
            geometry: ExperimentGeometry = eqx.tree_at(
                lambda candidate: candidate.sample_azimuth,
                _geometry(),
                azimuth,
            )
            density: Float64[Array, "u v e"] = (
                _map_source_to_detector_with_order(
                    source,
                    geometry,
                    calibration,
                    euler_angles,
                    order=4,
                )[0]
            )
            value: Float64[Array, ""] = jnp.sum(loss_weights * density)
            return value

        theta: Tuple[Float64[Array, ""], Float64[Array, " 3"]] = (
            jnp.array(0.13),
            jnp.array([0.08, 0.045, -0.025]),
        )
        assert_gradients_match_finite_differences(
            loss, theta, regime="smooth", elementwise=True
        )

    @pytest.mark.big_mem
    @pytest.mark.rss_limit_mb(1100)
    def test_energy_geometry_and_source_derivatives_match_fd(self) -> None:
        """Match photon-energy, work-function, and source gradients with FD.

        The weighted loss also retains the captured-flux fraction derivative.

        Notes
        -----
        A two-point affine source activates every source corner while the
        generic rotated target remains strictly inside all exterior faces.
        """
        kx_axis: Float64[Array, " x"] = jnp.array([-0.6, 0.6])
        ky_axis: Float64[Array, " y"] = jnp.array([-0.55, 0.65])
        energy_axis: Float64[Array, " e"] = jnp.array([-0.4, 0.4])
        intensity: Float64[Array, "x y e"] = (
            1.8
            + 0.21 * kx_axis[:, None, None]
            + 0.13 * ky_axis[None, :, None]
            + 0.09 * energy_axis[None, None, :]
        )
        source: ArpesCube = make_arpes_cube(
            intensity, kx_axis, ky_axis, energy_axis
        )
        base_geometry: ExperimentGeometry = _geometry(
            sample_azimuth=jnp.array(0.13)
        )
        calibration: DetectorCalibration = _map_calibration()
        angles: Float64[Array, " 3"] = jnp.array([0.08, 0.045, -0.025])
        loss_weights: Float64[Array, "u v e"] = jnp.array(
            [
                [[0.7, -0.2], [0.4, 1.1]],
                [[-0.3, 0.8], [0.5, -0.6]],
            ]
        )

        def loss(
            parameters: Tuple[
                Float64[Array, ""],
                Float64[Array, ""],
                Float64[Array, "x y e"],
            ],
        ) -> Float64[Array, ""]:
            """Compute one weighted map and captured-flux loss."""
            photon_energy: Float64[Array, ""]
            work_function: Float64[Array, ""]
            source_intensity: Float64[Array, "x y e"]
            photon_energy, work_function, source_intensity = parameters
            geometry: ExperimentGeometry = eqx.tree_at(
                lambda candidate: (
                    candidate.photon_energy_ev,
                    candidate.work_function_ev,
                ),
                base_geometry,
                (photon_energy, work_function),
            )
            candidate_source: ArpesCube = eqx.tree_at(
                lambda candidate: candidate.intensity,
                source,
                source_intensity,
            )
            density: Float64[Array, "u v e"]
            fraction: Float64[Array, ""]
            density, fraction = _map_source_to_detector_with_order(
                candidate_source,
                geometry,
                calibration,
                angles,
                order=4,
            )
            value: Float64[Array, ""] = (
                jnp.sum(loss_weights * density) + 0.31 * fraction
            )
            return value

        theta: Tuple[
            Float64[Array, ""],
            Float64[Array, ""],
            Float64[Array, "x y e"],
        ] = (
            jnp.array(50.0),
            jnp.array(4.0),
            intensity,
        )
        assert_gradients_match_finite_differences(loss, theta, regime="smooth")

    @pytest.mark.big_mem
    @pytest.mark.rss_limit_mb(1100)
    def test_target_edge_design_finite_differences_converge(self) -> None:
        """Compare two central-FD scales for every target-bin edge.

        The design study moves each native edge outside a compiled map claim.

        Notes
        -----
        The generic target stays strictly interior at both perturbation scales;
        all nine edge coordinates produce finite nonzero design sensitivities.
        """
        source: ArpesCube = _cube()
        geometry: ExperimentGeometry = _geometry(
            sample_azimuth=jnp.array(0.13)
        )
        calibration: DetectorCalibration = _map_calibration()
        angles: Float64[Array, " 3"] = jnp.array([0.08, 0.045, -0.025])
        weights: Float64[Array, "u v e"] = jnp.array(
            [
                [[0.7, -0.2], [0.4, 1.1]],
                [[-0.3, 0.8], [0.5, -0.6]],
            ]
        )
        design: Float64[Array, " 9"] = jnp.concatenate(
            (
                calibration.u_bin_edges,
                calibration.v_bin_edges,
                calibration.energy_bin_edges_ev,
            )
        )

        def evaluate(candidate: Float64[Array, " 9"]) -> Float64[Array, ""]:
            """Evaluate the weighted map at one target-edge design."""
            candidate_calibration: DetectorCalibration = (
                make_detector_calibration(
                    u_bin_edges=candidate[:3],
                    v_bin_edges=candidate[3:6],
                    energy_bin_edges_ev=candidate[6:],
                    psf_fwhm_u=calibration.psf_fwhm_u,
                    psf_fwhm_v=calibration.psf_fwhm_v,
                    psf_fwhm_energy_ev=calibration.psf_fwhm_energy_ev,
                    transmission_reference_domain_ev=(
                        calibration.transmission_reference_domain_ev
                    ),
                )
            )
            density: Float64[Array, "u v e"] = (
                _map_source_to_detector_with_order(
                    source,
                    geometry,
                    candidate_calibration,
                    angles,
                    order=4,
                )[0]
            )
            value: Float64[Array, ""] = jnp.sum(weights * density)
            return value

        coarse_step: float = 2.0e-5
        fine_step: float = 0.5 * coarse_step
        coarse_derivatives: List[float] = []
        fine_derivatives: List[float] = []
        coordinate: int
        for coordinate in range(design.shape[0]):
            coarse_plus: Float64[Array, " 9"] = design.at[coordinate].add(
                coarse_step
            )
            coarse_minus: Float64[Array, " 9"] = design.at[coordinate].add(
                -coarse_step
            )
            fine_plus: Float64[Array, " 9"] = design.at[coordinate].add(
                fine_step
            )
            fine_minus: Float64[Array, " 9"] = design.at[coordinate].add(
                -fine_step
            )
            coarse_derivatives.append(
                float(evaluate(coarse_plus) - evaluate(coarse_minus))
                / (2.0 * coarse_step)
            )
            fine_derivatives.append(
                float(evaluate(fine_plus) - evaluate(fine_minus))
                / (2.0 * fine_step)
            )
        coarse_array: Float64[NDArray, " 9"] = np.asarray(
            coarse_derivatives, dtype=np.float64
        )
        fine_array: Float64[NDArray, " 9"] = np.asarray(
            fine_derivatives, dtype=np.float64
        )
        assert np.all(np.isfinite(fine_array))
        assert np.min(np.abs(fine_array)) > 1.0e-6
        np.testing.assert_allclose(
            coarse_array, fine_array, rtol=5.0e-6, atol=5.0e-8
        )

    def test_generic_interior_converges_and_crossing_rejects(self) -> None:
        """Compare a smooth generic rotation and reject possible face loss.

        The enclosure preserves the bounded general-rotation contract.

        Notes
        -----
        Four- and eight-node maps agree on a strictly interior target. A
        wider target triggers the same traced guard eagerly and under JIT.
        """
        source: ArpesCube = _cube()
        geometry: ExperimentGeometry = _geometry(
            sample_azimuth=jnp.array(0.13)
        )
        calibration: DetectorCalibration = _map_calibration()
        angles: Float64[Array, " 3"] = jnp.array([0.08, 0.045, -0.025])
        mapped_four: Float64[Array, "u v e"]
        fraction_four: Float64[Array, ""]
        mapped_four, fraction_four = _map_source_to_detector_with_order(
            source, geometry, calibration, angles, order=4
        )
        mapped_eight: Float64[Array, "u v e"]
        fraction_eight: Float64[Array, ""]
        mapped_eight, fraction_eight = _map_source_to_detector_with_order(
            source, geometry, calibration, angles, order=8
        )
        chex.assert_trees_all_close(
            mapped_four, mapped_eight, rtol=1.0e-12, atol=1.0e-13
        )
        chex.assert_trees_all_close(
            fraction_four, fraction_eight, rtol=1.0e-12, atol=1.0e-14
        )

        crossing: DetectorCalibration = make_detector_calibration(
            u_bin_edges=jnp.array([-0.25, 0.0, 0.25]),
            v_bin_edges=jnp.array([-0.12, 0.0, 0.12]),
            energy_bin_edges_ev=calibration.energy_bin_edges_ev,
            psf_fwhm_u=calibration.psf_fwhm_u,
            psf_fwhm_v=calibration.psf_fwhm_v,
            psf_fwhm_energy_ev=calibration.psf_fwhm_energy_ev,
            transmission_reference_domain_ev=(
                calibration.transmission_reference_domain_ev
            ),
        )
        assert_rejects(
            _map_source_to_detector_with_order,
            source,
            geometry,
            crossing,
            angles,
            match="strictly inside source exterior faces",
            order=4,
        )

    def test_eager_jit_and_vmap_success_paths_agree(self) -> None:
        """Verify eager, compiled, and vectorized domain rotations.

        Every transformation preserves the same detector density.

        Notes
        -----
        The vmap covers distinct traced Euler rotations while target shapes
        and quadrature order remain static.
        """
        source: ArpesCube = _cube()
        geometry: ExperimentGeometry = _geometry(sample_azimuth=jnp.array(0.1))
        calibration: DetectorCalibration = _map_calibration()
        angles: Float64[Array, " 3"] = jnp.array([0.05, 0.03, -0.02])
        eager: Tuple[Float64[Array, "..."], Float64[Array, "..."]] = (
            _map_source_to_detector_with_order(
                source, geometry, calibration, angles, order=4
            )
        )
        compiled: Tuple[Float64[Array, "..."], Float64[Array, "..."]] = (
            eqx.filter_jit(_map_source_to_detector_with_order)(
                source, geometry, calibration, angles, order=4
            )
        )
        chex.assert_trees_all_close(eager, compiled, rtol=1.0e-13, atol=0.0)

        angle_batch: Float64[Array, "d 3"] = jnp.array(
            [[0.05, 0.03, -0.02], [-0.04, 0.02, 0.06]]
        )
        vectorized: Float64[Array, "d u v e"] = jax.vmap(
            lambda candidate: _map_source_to_detector_with_order(
                source,
                geometry,
                calibration,
                candidate,
                order=4,
            )[0]
        )(angle_batch)
        sequential: Float64[Array, "d u v e"] = jnp.stack(
            [
                _map_source_to_detector_with_order(
                    source,
                    geometry,
                    calibration,
                    candidate,
                    order=4,
                )[0]
                for candidate in angle_batch
            ]
        )
        chex.assert_trees_all_close(
            vectorized, sequential, rtol=1.0e-13, atol=0.0
        )
