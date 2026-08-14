"""Plant permanent counterexamples at detector scientific-contract seams.

The battery isolates plausible defects while every unrelated seam stays fixed.
"""

import equinox as eqx
import jax.numpy as jnp
import numpy as np
import pytest
from beartype.typing import Dict, Tuple
from jaxtyping import Array, Float64
from numpy.typing import NDArray
from scipy import ndimage, special

from diffpes.simul import (
    apply_detector_effects,
    convolve_energy,
    convolve_kpath,
    map_source_to_detector,
)
from diffpes.types import (
    ArpesCube,
    DetectorCalibration,
    DetectorEffects,
    DetectorRaster,
    ExperimentGeometry,
    make_arpes_cube,
    make_detector_calibration,
    make_detector_effects,
    make_experiment_geometry,
)

_FRAME_ID: str = "org.diffpes.frame.sample_cartesian"


def _normal_second_antiderivative(
    displacement: Float64[NDArray, "..."], sigma: float
) -> Float64[NDArray, "..."]:
    """PRIVATE: Evaluate an independent Gaussian second antiderivative.

    Parameters
    ----------
    displacement : Float64[NDArray, "..."]
        Gaussian displacement coordinates.
    sigma : float
        Positive Gaussian standard deviation.

    Returns
    -------
    result : Float64[NDArray, "..."]
        Evaluated second antiderivative.
    """
    scaled: Float64[NDArray, "..."] = displacement / sigma
    result: Float64[NDArray, "..."] = displacement * special.ndtr(
        scaled
    ) + sigma * np.exp(-0.5 * scaled**2) / np.sqrt(2.0 * np.pi)
    return result


def _finite_volume_matrix(
    edges: Float64[NDArray, " n_plus_one"], sigma: float
) -> Float64[NDArray, "n n"]:
    """PRIVATE: Build the analytic common-edge finite-volume Gaussian matrix.

    Parameters
    ----------
    edges : Float64[NDArray, " n_plus_one"]
        Strictly increasing common cell edges.
    sigma : float
        Positive Gaussian standard deviation.

    Returns
    -------
    matrix : Float64[NDArray, "n n"]
        Analytic source-to-target finite-volume matrix.
    """
    left: Float64[NDArray, " n"] = edges[:-1]
    right: Float64[NDArray, " n"] = edges[1:]
    integrated: Float64[NDArray, "n n"] = (
        _normal_second_antiderivative(right[:, None] - left[None, :], sigma)
        - _normal_second_antiderivative(left[:, None] - left[None, :], sigma)
        - _normal_second_antiderivative(right[:, None] - right[None, :], sigma)
        + _normal_second_antiderivative(left[:, None] - right[None, :], sigma)
    )
    matrix: Float64[NDArray, "n n"] = (
        np.maximum(integrated, 0.0) / np.diff(edges)[:, None]
    )
    return matrix


def _compact_detector_fixture() -> Tuple[
    ArpesCube,
    ExperimentGeometry,
    DetectorCalibration,
    DetectorEffects,
]:
    """PRIVATE: Build a compact fixture with every downstream seam active.

    Returns
    -------
    fixture : Tuple
        Source, geometry, calibration, and detector effects.
    """
    kx: Float64[Array, "3"] = jnp.array([-0.5, 0.0, 0.5])
    ky: Float64[Array, "3"] = jnp.array([-0.45, 0.05, 0.55])
    energy: Float64[Array, "3"] = jnp.array([-0.4, 0.0, 0.4])
    intensity: Float64[Array, "..."] = (
        2.0
        + 0.35 * kx[:, None, None]
        + 0.22 * ky[None, :, None]
        + 0.17 * energy[None, None, :]
        + 0.08 * kx[:, None, None] * ky[None, :, None]
    )
    source: ArpesCube = make_arpes_cube(intensity, kx, ky, energy)
    geometry: ExperimentGeometry = make_experiment_geometry(
        photon_energy_ev=50.0,
        polarization=jnp.array([1.0 + 0.0j, 0.0j, 0.0j]),
        work_function_ev=4.0,
    )
    calibration: DetectorCalibration = make_detector_calibration(
        u_bin_edges=jnp.array([-0.055, -0.012, 0.047]),
        v_bin_edges=jnp.array([-0.048, 0.009, 0.052]),
        energy_bin_edges_ev=jnp.array([-0.24, -0.03, 0.21]),
        psf_fwhm_u=0.01,
        psf_fwhm_v=0.012,
        psf_fwhm_energy_ev=0.02,
        transmission_reference_domain_ev=jnp.array([40.0, 55.0]),
    )
    effects: DetectorEffects = make_detector_effects(
        domain_logits=jnp.array([0.0]),
        domain_euler_angles_rad=jnp.zeros((1, 3)),
        transmission_raw_slopes=jnp.array([-0.4, 0.2]),
        background_coefficients=jnp.array([-2.0]),
        sensitivity_coefficients=jnp.array(
            [0.08, -0.05, 0.03, 0.04, -0.02, 0.06]
        ),
        exposure=2.5,
        background_mode="flat",
        sensitivity_mode="smooth",
        domain_frame_ids=(_FRAME_ID,),
    )
    returned: Tuple[
        ArpesCube,
        ExperimentGeometry,
        DetectorCalibration,
        DetectorEffects,
    ] = source, geometry, calibration, effects
    return returned


class TestExplicitTargetCounterexample:
    """Reject target inference from a source carrier.

    The case applies two detector calibrations to one source and requires two
    explicitly declared target maps.
    """

    @pytest.mark.slow
    def test_two_calibrations_require_two_declared_target_maps(self) -> None:
        """Reject a planted mapper that silently reuses source-inferred bins.

        The case compares two explicit calibrations on one source.

        Notes
        -----
        Map the source independently and require distinct detector densities.
        """
        source: ArpesCube
        geometry: ExperimentGeometry
        first: DetectorCalibration
        source, geometry, first, _ = _compact_detector_fixture()
        second: DetectorCalibration = make_detector_calibration(
            u_bin_edges=jnp.array([-0.047, -0.004, 0.055]),
            v_bin_edges=jnp.array([-0.054, 0.003, 0.046]),
            energy_bin_edges_ev=jnp.array([-0.21, 0.0, 0.24]),
            psf_fwhm_u=0.01,
            psf_fwhm_v=0.012,
            psf_fwhm_energy_ev=0.02,
            transmission_reference_domain_ev=jnp.array([40.0, 55.0]),
        )
        first_density: Float64[Array, "..."] = map_source_to_detector(
            source, geometry, first
        )[0]
        second_density: Float64[Array, "..."] = map_source_to_detector(
            source, geometry, second
        )[0]
        planted_inferred_target: Float64[Array, "..."] = first_density

        assert first_density.shape == second_density.shape
        assert not np.allclose(
            planted_inferred_target,
            second_density,
            rtol=1.0e-10,
            atol=1.0e-12,
        )


class TestFinitePathBoundaryCounterexamples:
    """Reject nonphysical alternatives to finite-path loss semantics.

    The case compares a boundary signal with planted path-length variants and
    requires every nonphysical variant to differ from the reference.
    """

    def test_boundary_signal_exposes_every_planted_path_variant(self) -> None:
        """Reject row-normalized, extended, and sampled nonuniform kernels.

        The fixture places asymmetric signal at finite-path boundaries.

        Notes
        -----
        Compare each planted construction with the analytic finite-volume
        result.
        """
        centres: Float64[NDArray, " k"] = np.array(
            [-0.9, -0.35, -0.05, 0.5, 1.2]
        )
        density: Float64[NDArray, "k e"] = np.array(
            [[1.7, 0.3], [0.2, 1.2], [0.8, 0.4], [0.1, 0.9], [1.3, 0.2]]
        )
        sigma: float = 0.28
        interior: Float64[NDArray, " k_minus_one"] = 0.5 * (
            centres[:-1] + centres[1:]
        )
        edges: Float64[NDArray, " k_plus_one"] = np.concatenate(
            (
                [centres[0] - 0.5 * (centres[1] - centres[0])],
                interior,
                [centres[-1] + 0.5 * (centres[-1] - centres[-2])],
            )
        )
        matrix: Float64[NDArray, "k k"] = _finite_volume_matrix(edges, sigma)
        truth: Float64[Array, "..."]
        captured: Float64[Array, "..."]
        truth, captured, _ = convolve_kpath(
            jnp.asarray(density), jnp.asarray(centres), sigma
        )
        mean_spacing: float = float(np.mean(np.diff(centres)))
        gaussian_samples: Float64[NDArray, "target source"] = np.exp(
            -0.5 * ((centres[:, None] - centres[None, :]) / sigma) ** 2
        ) / (sigma * np.sqrt(2.0 * np.pi))
        midpoint: Float64[NDArray, "k e"] = gaussian_samples @ (
            density * np.diff(edges)[:, None]
        )
        trapezoid: Float64[NDArray, "k e"] = np.stack(
            [
                np.stack(
                    [
                        np.trapezoid(
                            gaussian_samples[target] * density[:, channel],
                            centres,
                        )
                        for channel in range(density.shape[1])
                    ]
                )
                for target in range(centres.size)
            ]
        )
        variants: Dict[str, Float64[NDArray, "k e"]] = {
            "row_normalized": (
                matrix / np.sum(matrix, axis=1, keepdims=True) @ density
            ),
            "reflected": ndimage.gaussian_filter1d(
                density,
                sigma=sigma / mean_spacing,
                axis=0,
                mode="reflect",
            ),
            "replicated": ndimage.gaussian_filter1d(
                density,
                sigma=sigma / mean_spacing,
                axis=0,
                mode="nearest",
            ),
            "periodic": ndimage.gaussian_filter1d(
                density,
                sigma=sigma / mean_spacing,
                axis=0,
                mode="wrap",
            ),
            "sampled_midpoint": midpoint,
            "sampled_trapezoid": trapezoid,
            "crop_renormalized": (
                matrix / np.sum(matrix, axis=0, keepdims=True) @ density
            ),
        }

        assert 0.0 < float(captured) < 1.0
        name: str
        planted: Float64[NDArray, "k e"]
        for name, planted in variants.items():
            assert not np.allclose(
                planted, truth, rtol=1.0e-11, atol=1.0e-13
            ), f"{name} unexpectedly passed finite-volume truth"


class TestManufacturedSeamCounterexamples:
    """Verify perturbations of complete detector-chain seams one at a time.

    The case changes each remaining effects seam separately and requires a
    corresponding change in the expected counts.
    """

    @pytest.mark.slow
    def test_each_remaining_effect_seam_changes_expected_counts(self) -> None:
        """Expose every detector-response component.

        The test perturbs each seam while every other seam stays fixed.

        Notes
        -----
        Recompute final counts and require every planted result to change.
        """
        source: ArpesCube
        geometry: ExperimentGeometry
        calibration: DetectorCalibration
        effects: DetectorEffects
        source, geometry, calibration, effects = _compact_detector_fixture()
        accepted: DetectorRaster = apply_detector_effects(
            (source,), geometry, calibration, effects
        )
        planted_effects: Dict[str, DetectorEffects] = {
            "domain_map": eqx.tree_at(
                lambda item: item.domain_euler_angles_rad,
                effects,
                jnp.array([[0.5 * np.pi, 0.0, 0.0]]),
            ),
            "transmission": eqx.tree_at(
                lambda item: item.transmission_raw_slopes,
                effects,
                jnp.zeros_like(effects.transmission_raw_slopes),
            ),
            "background": eqx.tree_at(
                lambda item: item.background_coefficients,
                effects,
                effects.background_coefficients + 0.4,
            ),
            "sensitivity": eqx.tree_at(
                lambda item: item.sensitivity_coefficients,
                effects,
                effects.sensitivity_coefficients.at[0].add(0.12),
            ),
            "exposure": eqx.tree_at(
                lambda item: item.exposure,
                effects,
                effects.exposure * 1.1,
            ),
        }
        planted_calibration: DetectorCalibration = eqx.tree_at(
            lambda item: item.psf_fwhm_u,
            calibration,
            calibration.psf_fwhm_u * 1.4,
        )
        planted_counts: Dict[str, Float64[Array, "..."]] = {
            name: apply_detector_effects(
                (source,), geometry, calibration, candidate
            ).expected_counts
            for name, candidate in planted_effects.items()
        }
        planted_counts["native_psf"] = apply_detector_effects(
            (source,), geometry, planted_calibration, effects
        ).expected_counts

        name: str
        planted: Float64[Array, "..."]
        for name, planted in planted_counts.items():
            assert not np.allclose(
                planted,
                accepted.expected_counts,
                rtol=1.0e-10,
                atol=1.0e-12,
            ), f"{name} perturbation unexpectedly preserved final counts"


class TestNumericalParityIsNotFiniteVolumeTruth:
    """Show sampled parity cannot certify bin-integrated physics.

    The case constructs an operator that matches SciPy at sample points but
    disagrees with an independent integral on coarse detector bins.
    """

    def test_one_sampled_operator_passes_scipy_but_fails_coarse_truth(
        self,
    ) -> None:
        """Pass SciPy exactly before failing the two-cell analytic result.

        The case compares one sampled operator with finite-volume truth.

        Notes
        -----
        Match SciPy first, then require disagreement with analytic integration.
        """
        centres: Float64[Array, "2"] = jnp.array([0.0, 1.0])
        density: Float64[Array, "2 1"] = jnp.ones((2, 1))
        sigma: float = 0.01
        sampled: Float64[Array, "..."] = convolve_energy(
            density.T, centres, sigma
        ).T
        scipy_sampled: Float64[NDArray, "k e"] = ndimage.gaussian_filter1d(
            np.ones((2, 1)),
            sigma=sigma,
            axis=0,
            mode="constant",
            radius=48,
        )
        finite_volume: Float64[Array, "..."] = convolve_kpath(
            density, centres, sigma
        )[0]

        np.testing.assert_allclose(
            sampled, scipy_sampled, rtol=1.0e-10, atol=1.0e-13
        )
        assert not np.allclose(
            sampled, finite_volume, rtol=1.0e-11, atol=1.0e-13
        )
