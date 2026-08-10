"""Verify the complete manufactured source-to-count detector chain.

The tests compare production mapping and detector stages with a frozen
NumPy/SciPy reference, then plant plausible stage-order and measure defects.
"""

from pathlib import Path

import chex
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from beartype.typing import Dict, Tuple
from jaxtyping import Array, Float64
from numpy.typing import NDArray

from diffpes.simul._detector_map import (
    _map_and_mix_domains,
    _map_source_to_detector_with_order,
)
from diffpes.simul.effects import (
    apply_detector_effects,
    apply_resolution,
    apply_transmission,
    background_density,
    detector_bin_volumes,
    expected_counts,
    fixed_total_probabilities,
    sensitivity_field,
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
from tests._gradients import assert_grad_matches_fd, gradient_gate

_FRAME_ID: str = "org.diffpes.frame.sample_cartesian"
_REFERENCE_PATH: Path = (
    Path(__file__).resolve().parents[1]
    / "_reference_data"
    / "detector_chain_manufactured_reference.npz"
)
_EffectsLeaves = Tuple[
    Float64[Array, " d"],
    Float64[Array, "d 3"],
    Float64[Array, " q"],
    Float64[Array, " b"],
    Float64[Array, " s"],
    Float64[Array, ""],
    Float64[Array, " k"],
]


def _load_reference() -> Dict[str, Float64[NDArray, "..."]]:
    """PRIVATE: Load the inert independent detector-chain archive.

    Returns
    -------
    reference : Dict[str, Float64[NDArray, "..."]]
        Named float64 fixture inputs, physical seams, and expected counts.
    """
    archive: np.lib.npyio.NpzFile
    with np.load(_REFERENCE_PATH, allow_pickle=False) as archive:
        reference: Dict[str, Float64[NDArray, "..."]] = {
            name: np.asarray(archive[name], dtype=np.float64)
            for name in archive.files
        }
    return reference


def _manufactured_fixture() -> Tuple[
    Tuple[ArpesCube, ArpesCube],
    ExperimentGeometry,
    DetectorCalibration,
    DetectorEffects,
]:
    """PRIVATE: Build the production side of the frozen fixture.

    Returns
    -------
    fixture : Tuple
        Two source domains, geometry, calibration, and detector effects.
    """
    reference: Dict[str, Float64[NDArray, "..."]] = _load_reference()
    source: ArpesCube = make_arpes_cube(
        intensity=jnp.asarray(reference["source_intensity"]),
        kx_axis=jnp.asarray(reference["kx_axis"]),
        ky_axis=jnp.asarray(reference["ky_axis"]),
        energy_axis=jnp.asarray(reference["source_energy_axis"]),
    )
    geometry: ExperimentGeometry = make_experiment_geometry(
        photon_energy_ev=50.0,
        polarization=jnp.array([1.0 + 0.0j, 0.0j, 0.0j]),
        work_function_ev=4.5,
    )
    calibration: DetectorCalibration = make_detector_calibration(
        u_bin_edges=jnp.asarray(reference["u_edges"]),
        v_bin_edges=jnp.asarray(reference["v_edges"]),
        energy_bin_edges_ev=jnp.asarray(reference["energy_edges"]),
        psf_fwhm_u=0.009,
        psf_fwhm_v=0.013,
        psf_fwhm_energy_ev=0.021,
        transmission_reference_domain_ev=jnp.array([40.0, 50.0]),
    )
    effects: DetectorEffects = make_detector_effects(
        domain_logits=jnp.array([0.0, np.log(2.0)]),
        domain_euler_angles_rad=jnp.array(
            [[0.0, 0.0, 0.0], [0.5 * np.pi, 0.0, 0.0]]
        ),
        transmission_raw_slopes=jnp.array([-0.4, 0.2]),
        background_coefficients=jnp.array([-2.0]),
        sensitivity_coefficients=jnp.array(
            [0.08, -0.05, 0.03, 0.04, -0.02, 0.06]
        ),
        exposure=100.0,
        background_mode="flat",
        sensitivity_mode="smooth",
        domain_frame_ids=(_FRAME_ID, _FRAME_ID),
    )
    fixture: Tuple[
        Tuple[ArpesCube, ArpesCube],
        ExperimentGeometry,
        DetectorCalibration,
        DetectorEffects,
    ] = ((source, source), geometry, calibration, effects)
    return fixture


def _differentiable_fixture() -> Tuple[
    Tuple[ArpesCube, ArpesCube],
    ExperimentGeometry,
    DetectorCalibration,
    Float64[Array, "1 u v e"],
]:
    """PRIVATE: Build a compact strictly interior full-chain fixture.

    Returns
    -------
    fixture : Tuple
        Two affine sources, geometry, calibration, and asymmetric loss weights.
    """
    kx_axis: Float64[Array, " x"] = jnp.array([-0.6, 0.6])
    ky_axis: Float64[Array, " y"] = jnp.array([-0.55, 0.65])
    energy_axis: Float64[Array, " e"] = jnp.array([-0.4, 0.4])
    first_intensity: Float64[Array, "x y e"] = (
        1.8
        + 0.21 * kx_axis[:, None, None]
        + 0.13 * ky_axis[None, :, None]
        + 0.09 * energy_axis[None, None, :]
    )
    second_intensity: Float64[Array, "x y e"] = (
        1.5
        - 0.16 * kx_axis[:, None, None]
        + 0.19 * ky_axis[None, :, None]
        + 0.07 * energy_axis[None, None, :]
        + 0.06 * kx_axis[:, None, None] * ky_axis[None, :, None]
    )
    first: ArpesCube = make_arpes_cube(
        first_intensity, kx_axis, ky_axis, energy_axis
    )
    second: ArpesCube = make_arpes_cube(
        second_intensity, kx_axis, ky_axis, energy_axis
    )
    geometry: ExperimentGeometry = make_experiment_geometry(
        photon_energy_ev=50.0,
        polarization=jnp.array([1.0 + 0.0j, 0.0j, 0.0j]),
        sample_azimuth=0.11,
        work_function_ev=4.0,
    )
    calibration: DetectorCalibration = make_detector_calibration(
        u_bin_edges=jnp.array([-0.05, -0.008, 0.047]),
        v_bin_edges=jnp.array([-0.043, 0.006, 0.052]),
        energy_bin_edges_ev=jnp.array([-0.22, -0.015, 0.19]),
        psf_fwhm_u=0.012,
        psf_fwhm_v=0.015,
        psf_fwhm_energy_ev=0.025,
        transmission_reference_domain_ev=jnp.array([40.0, 55.0]),
    )
    weights: Float64[Array, "1 u v e"] = jnp.array(
        [
            [
                [[0.7, -0.2], [0.4, 1.1]],
                [[-0.3, 0.8], [0.5, -0.6]],
            ]
        ]
    )
    fixture: Tuple[
        Tuple[ArpesCube, ArpesCube],
        ExperimentGeometry,
        DetectorCalibration,
        Float64[Array, "1 u v e"],
    ] = ((first, second), geometry, calibration, weights)
    return fixture


def _effects_from_leaves(candidate: _EffectsLeaves) -> DetectorEffects:
    """PRIVATE: Build the full smooth effects carrier from traced leaves.

    Parameters
    ----------
    candidate : _EffectsLeaves
        Every continuous detector-effects coordinate in public field order.

    Returns
    -------
    effects : DetectorEffects
        Two-domain smooth carrier with calibrated post-count response.
    """
    logits: Float64[Array, " d"]
    rotations: Float64[Array, "d 3"]
    transmission: Float64[Array, " q"]
    background: Float64[Array, " b"]
    sensitivity: Float64[Array, " s"]
    exposure: Float64[Array, ""]
    kernel: Float64[Array, " k"]
    (
        logits,
        rotations,
        transmission,
        background,
        sensitivity,
        exposure,
        kernel,
    ) = candidate
    effects: DetectorEffects = make_detector_effects(
        domain_logits=logits,
        domain_euler_angles_rad=rotations,
        transmission_raw_slopes=transmission,
        background_coefficients=background,
        sensitivity_coefficients=sensitivity,
        exposure=exposure,
        background_mode="smooth",
        sensitivity_mode="smooth",
        post_count_mode="calibrated",
        post_count_kernel=kernel,
        domain_frame_ids=(_FRAME_ID, _FRAME_ID),
    )
    return effects


def _effects_leaves() -> _EffectsLeaves:
    """PRIVATE: Return generic continuous detector-effects coordinates.

    Returns
    -------
    leaves : _EffectsLeaves
        Asymmetric interior values for every continuous public field.
    """
    leaves: _EffectsLeaves = (
        jnp.array([-0.25, 0.35]),
        jnp.array([[0.08, 0.045, -0.025], [-0.06, 0.035, 0.055]]),
        jnp.array([-0.4, 0.2]),
        jnp.array([-1.8, 0.08, -0.04, 0.06, 0.03, -0.05, 0.02]),
        jnp.array([0.08, -0.05, 0.03, 0.04, -0.02, 0.06]),
        jnp.array(2.3),
        jnp.array([0.25, 0.6, 0.15]),
    )
    return leaves


class TestManufacturedDetectorChain:
    """Verify independent truth and deliberately broken detector chains."""

    def test_domain_maps_match_independent_segmented_truth(self) -> None:
        """Match both rotated maps and captured fractions independently.

        The comparison pins the conservative boundary-loss result.

        Notes
        -----
        The frozen reference analytically integrates secondary momentum and
        applies 96-node segmented quadrature to the other smooth coordinates.
        """
        reference: Dict[str, Float64[NDArray, "..."]] = _load_reference()
        sources: Tuple[ArpesCube, ArpesCube]
        geometry: ExperimentGeometry
        calibration: DetectorCalibration
        effects: DetectorEffects
        sources, geometry, calibration, effects = _manufactured_fixture()
        identity: Float64[Array, "u v e"]
        identity_fraction: Float64[Array, ""]
        identity, identity_fraction = _map_source_to_detector_with_order(
            sources[0],
            geometry,
            calibration,
            effects.domain_euler_angles_rad[0],
            order=4,
        )
        quarter_turn: Float64[Array, "u v e"]
        quarter_fraction: Float64[Array, ""]
        quarter_turn, quarter_fraction = _map_source_to_detector_with_order(
            sources[1],
            geometry,
            calibration,
            effects.domain_euler_angles_rad[1],
            order=4,
        )
        actual_fractions: Float64[Array, " 2"] = jnp.stack(
            (identity_fraction, quarter_fraction)
        )

        np.testing.assert_allclose(
            identity,
            reference["identity_density"],
            rtol=1.0e-9,
            atol=1.0e-12,
        )
        np.testing.assert_allclose(
            quarter_turn,
            reference["quarter_density"],
            rtol=1.0e-9,
            atol=1.0e-12,
        )
        np.testing.assert_allclose(
            actual_fractions,
            reference["captured_fractions"],
            rtol=1.0e-9,
            atol=1.0e-12,
        )
        np.testing.assert_allclose(
            actual_fractions,
            jnp.array([0.9170695116692185, 0.9170695116692177]),
            rtol=0.0,
            atol=5.0e-15,
        )

    def test_every_physical_seam_and_final_counts_match(self) -> None:
        """Match mapping, transmission, resolution, sensitivity, and counts.

        The public raster must reproduce the same final observable.

        Notes
        -----
        The test evaluates each production stage once and compares it with a
        separately assembled NumPy/SciPy seam before checking the public chain.
        """
        reference: Dict[str, Float64[NDArray, "..."]] = _load_reference()
        sources: Tuple[ArpesCube, ArpesCube]
        geometry: ExperimentGeometry
        calibration: DetectorCalibration
        effects: DetectorEffects
        sources, geometry, calibration, effects = _manufactured_fixture()
        mixed: Float64[Array, "u v e"]
        mixed_fraction: Float64[Array, ""]
        mixed, mixed_fraction = _map_and_mix_domains(
            sources, geometry, calibration, effects
        )
        recorded_energy: Float64[Array, " e"] = 0.5 * (
            calibration.energy_bin_edges_ev[:-1]
            + calibration.energy_bin_edges_ev[1:]
        )
        kinetic_energy: Float64[Array, " e"] = (
            geometry.photon_energy_ev
            - geometry.work_function_ev
            + recorded_energy
        )
        transmitted: Float64[Array, "u v e"] = apply_transmission(
            mixed,
            kinetic_energy,
            effects.transmission_raw_slopes,
            calibration,
        )
        resolved: Float64[Array, "u v e"] = apply_resolution(
            transmitted, calibration
        )[0]
        sensitivity: Float64[Array, "u v e"] = sensitivity_field(
            calibration, effects
        )
        volumes: Float64[Array, "u v e"] = detector_bin_volumes(calibration)
        counts: Float64[Array, "1 u v e"] = expected_counts(
            resolved[None, ...], calibration, effects
        )
        raster: DetectorRaster = apply_detector_effects(
            sources, geometry, calibration, effects
        )

        assert float(mixed_fraction) > 0.0
        np.testing.assert_allclose(
            mixed, reference["mixed_density"], rtol=1.0e-9, atol=1.0e-12
        )
        np.testing.assert_allclose(
            transmitted,
            reference["transmitted_density"],
            rtol=1.0e-9,
            atol=1.0e-12,
        )
        np.testing.assert_allclose(
            resolved,
            reference["resolved_density"],
            rtol=1.0e-9,
            atol=1.0e-12,
        )
        np.testing.assert_allclose(
            sensitivity,
            reference["sensitivity"],
            rtol=1.0e-10,
            atol=1.0e-12,
        )
        np.testing.assert_allclose(
            volumes, reference["volumes"], rtol=1.0e-13, atol=0.0
        )
        np.testing.assert_allclose(
            counts, reference["counts"], rtol=1.0e-8, atol=1.0e-12
        )
        np.testing.assert_allclose(
            raster.expected_counts,
            reference["counts"],
            rtol=1.0e-8,
            atol=1.0e-12,
        )

    def test_planted_mixing_measure_and_order_defects_fail_truth(self) -> None:
        """Reject source-wise mixing, omitted volumes, and reordered stages.

        The three controls must each exceed the registered truth tolerance.

        Notes
        -----
        Each planted implementation reuses valid production primitives but
        violates one chain seam and must leave the frozen count tolerance.
        """
        reference: Dict[str, Float64[NDArray, "..."]] = _load_reference()
        sources: Tuple[ArpesCube, ArpesCube]
        geometry: ExperimentGeometry
        calibration: DetectorCalibration
        effects: DetectorEffects
        sources, geometry, calibration, effects = _manufactured_fixture()
        identity: Float64[Array, "u v e"] = _map_source_to_detector_with_order(
            sources[0],
            geometry,
            calibration,
            effects.domain_euler_angles_rad[0],
            order=4,
        )[0]
        mixed: Float64[Array, "u v e"] = _map_and_mix_domains(
            sources, geometry, calibration, effects
        )[0]
        recorded_energy: Float64[Array, " e"] = 0.5 * (
            calibration.energy_bin_edges_ev[:-1]
            + calibration.energy_bin_edges_ev[1:]
        )
        kinetic_energy: Float64[Array, " e"] = (
            geometry.photon_energy_ev
            - geometry.work_function_ev
            + recorded_energy
        )
        source_mixed_transmitted: Float64[Array, "u v e"] = apply_transmission(
            identity,
            kinetic_energy,
            effects.transmission_raw_slopes,
            calibration,
        )
        source_mixed_resolved: Float64[Array, "u v e"] = apply_resolution(
            source_mixed_transmitted, calibration
        )[0]
        source_mixed_counts: Float64[Array, "1 u v e"] = expected_counts(
            source_mixed_resolved[None, ...], calibration, effects
        )
        transmitted: Float64[Array, "u v e"] = apply_transmission(
            mixed,
            kinetic_energy,
            effects.transmission_raw_slopes,
            calibration,
        )
        resolved: Float64[Array, "u v e"] = apply_resolution(
            transmitted, calibration
        )[0]
        background: Float64[Array, "1 u v e"] = background_density(
            resolved[None, ...], calibration, effects
        )
        sensitivity: Float64[Array, "u v e"] = sensitivity_field(
            calibration, effects
        )
        missing_measure_counts: Float64[Array, "1 u v e"] = (
            effects.exposure
            * sensitivity[None, ...]
            * (resolved[None, ...] + background)
        )
        prematurely_resolved: Float64[Array, "u v e"] = apply_resolution(
            mixed, calibration
        )[0]
        reordered_density: Float64[Array, "u v e"] = apply_transmission(
            prematurely_resolved,
            kinetic_energy,
            effects.transmission_raw_slopes,
            calibration,
        )
        reordered_counts: Float64[Array, "1 u v e"] = expected_counts(
            reordered_density[None, ...], calibration, effects
        )
        incorrect: Tuple[Float64[Array, "1 u v e"], ...] = (
            source_mixed_counts,
            missing_measure_counts,
            reordered_counts,
        )
        candidate: Float64[Array, "1 u v e"]
        for candidate in incorrect:
            assert not bool(
                jnp.allclose(
                    candidate,
                    reference["counts"],
                    rtol=1.0e-8,
                    atol=1.0e-12,
                )
            )


class TestDetectorChainDerivatives:
    """Verify every continuous effects leaf through the complete chain."""

    @pytest.mark.big_mem
    @pytest.mark.rss_limit_mb(2500)
    def test_expected_rate_leaves_pass_fd_jit_and_vmap(self) -> None:
        """Differentiate every effects leaf through expected detector rates.

        Eager, compiled, and batched evaluations must preserve the same loss.

        Notes
        -----
        Two generic rotated domains keep the mapper strictly inside source
        support while every downstream smooth mode remains active.
        """
        sources: Tuple[ArpesCube, ArpesCube]
        geometry: ExperimentGeometry
        calibration: DetectorCalibration
        weights: Float64[Array, "1 u v e"]
        sources, geometry, calibration, weights = _differentiable_fixture()
        theta: _EffectsLeaves = _effects_leaves()

        def rate_loss(candidate: _EffectsLeaves) -> Float64[Array, ""]:
            """Compute a weighted full-chain expected-rate loss."""
            effects: DetectorEffects = _effects_from_leaves(candidate)
            raster: DetectorRaster = apply_detector_effects(
                sources, geometry, calibration, effects
            )
            value: Float64[Array, ""] = jnp.sum(
                weights * raster.expected_counts
            )
            return value

        gradient_gate(rate_loss, theta, regime="smooth")
        eager: Float64[Array, ""] = rate_loss(theta)
        compiled: Float64[Array, ""] = jax.jit(rate_loss)(theta)
        chex.assert_trees_all_close(compiled, eager, rtol=1.0e-10, atol=0.0)
        batched_theta: _EffectsLeaves = jax.tree.map(
            lambda leaf: jnp.stack((leaf, leaf * 1.01)), theta
        )
        vectorized: Float64[Array, " batch"] = jax.jit(jax.vmap(rate_loss))(
            batched_theta
        )
        sequential: Float64[Array, " batch"] = jnp.stack(
            (
                rate_loss(theta),
                rate_loss(jax.tree.map(lambda leaf: leaf * 1.01, theta)),
            )
        )
        chex.assert_trees_all_close(
            vectorized, sequential, rtol=1.0e-10, atol=0.0
        )

    @pytest.mark.big_mem
    @pytest.mark.rss_limit_mb(2500)
    def test_probability_leaves_pass_fd_jit_and_vmap(self) -> None:
        """Differentiate every effects leaf through event probabilities.

        Global normalization must remove only the exposure-scale derivative.

        Notes
        -----
        The shared FD harness includes exposure, while explicit checks require
        every other leaf to retain a finite nonzero probability sensitivity.
        """
        sources: Tuple[ArpesCube, ArpesCube]
        geometry: ExperimentGeometry
        calibration: DetectorCalibration
        weights: Float64[Array, "1 u v e"]
        sources, geometry, calibration, weights = _differentiable_fixture()
        theta: _EffectsLeaves = _effects_leaves()

        def probability_loss(candidate: _EffectsLeaves) -> Float64[Array, ""]:
            """Compute a weighted full-chain event-probability loss."""
            effects: DetectorEffects = _effects_from_leaves(candidate)
            raster: DetectorRaster = apply_detector_effects(
                sources, geometry, calibration, effects
            )
            probabilities: Float64[Array, "1 u v e"] = (
                fixed_total_probabilities(raster.expected_counts)
            )
            value: Float64[Array, ""] = jnp.sum(weights * probabilities)
            return value

        assert_grad_matches_fd(probability_loss, theta, regime="smooth")
        gradient: _EffectsLeaves = jax.grad(probability_loss)(theta)
        leaf_index: int
        leaf: Float64[Array, "..."]
        for leaf_index, leaf in enumerate(gradient):
            if leaf_index == 5:  # noqa: PLR2004
                chex.assert_trees_all_close(
                    leaf, jnp.zeros_like(leaf), rtol=0.0, atol=1.0e-13
                )
            else:
                assert float(jnp.linalg.norm(jnp.ravel(leaf))) > 1.0e-10
        eager: Float64[Array, ""] = probability_loss(theta)
        compiled: Float64[Array, ""] = jax.jit(probability_loss)(theta)
        chex.assert_trees_all_close(compiled, eager, rtol=1.0e-10, atol=0.0)
        batched_theta: _EffectsLeaves = jax.tree.map(
            lambda candidate: jnp.stack((candidate, candidate * 1.01)), theta
        )
        vectorized: Float64[Array, " batch"] = jax.jit(
            jax.vmap(probability_loss)
        )(batched_theta)
        sequential: Float64[Array, " batch"] = jnp.stack(
            (
                probability_loss(theta),
                probability_loss(
                    jax.tree.map(lambda candidate: candidate * 1.01, theta)
                ),
            )
        )
        chex.assert_trees_all_close(
            vectorized, sequential, rtol=1.0e-10, atol=0.0
        )
