"""Validate the effects module.

The cases use analytic values, invariants, and finite differences.
"""

import chex
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float64

from diffpes.simul import (
    apply_detector_effects,
    apply_resolution,
    apply_transmission,
    expected_counts,
    map_source_to_detector,
)
from diffpes.types import (
    ArpesCube,
    DetectorCalibration,
    DetectorEffects,
    DetectorRaster,
    ExperimentGeometry,
)

from ._effects_helpers import (
    _detector_chain_fixture,
)


class TestMapSourceToDetector:
    """Verify :func:`diffpes.simul.map_source_to_detector`.

    The class owns the public conservative-mapping surface and diagnostics.
    """

    def test_returns_finite_density_and_captured_flux(self) -> None:
        """Convert a named Cartesian source without inferring target bins.

        The case verifies the conservative public mapping boundary.

        Notes
        -----
        The focused public check complements the private analytic/Jacobian
        battery and preserves the reported boundary-loss diagnostic.
        """
        source: ArpesCube
        geometry: ExperimentGeometry
        calibration: DetectorCalibration
        source, geometry, calibration, _ = _detector_chain_fixture()
        density: Float64[Array, "..."]
        captured: Float64[Array, "..."]
        density, captured = map_source_to_detector(
            source, geometry, calibration
        )

        chex.assert_shape(density, (2, 2, 2))
        assert bool(jnp.all(jnp.isfinite(density)))
        assert bool(jnp.all(density >= 0.0))
        assert 0.0 < float(captured) <= 1.0


class TestApplyDetectorEffects:
    """Verify :func:`diffpes.simul.apply_detector_effects`.

    The class owns the public stage order and native-count carrier boundary.
    """

    def test_matches_explicit_ordered_stage_composition(self) -> None:
        """Match the complete deterministic chain stage by stage.

        The case pins transmission before resolution and count assembly.

        Notes
        -----
        Transmission uses true kinetic energy before resolution; expected
        counts then apply background, sensitivity, exposure, volume, and the
        optional recorded response.
        """
        source: ArpesCube
        geometry: ExperimentGeometry
        calibration: DetectorCalibration
        effects: DetectorEffects
        source, geometry, calibration, effects = _detector_chain_fixture()
        mapped: Float64[Array, "..."]
        mapped, _ = map_source_to_detector(source, geometry, calibration)
        recorded_energy: Float64[Array, "..."] = 0.5 * (
            calibration.energy_bin_edges_ev[:-1]
            + calibration.energy_bin_edges_ev[1:]
        )
        kinetic_energy: Float64[Array, "..."] = (
            geometry.photon_energy_ev
            - geometry.work_function_ev
            + recorded_energy
        )
        transmitted: Float64[Array, "..."] = apply_transmission(
            mapped,
            kinetic_energy,
            effects.transmission_raw_slopes,
            calibration,
        )
        resolved: Float64[Array, "..."] = apply_resolution(
            transmitted, calibration
        )[0]
        desired: Float64[Array, "..."] = expected_counts(
            resolved[None, ...], calibration, effects
        )

        raster: DetectorRaster = apply_detector_effects(
            (source,), geometry, calibration, effects
        )

        chex.assert_trees_all_close(raster.expected_counts, desired)
        chex.assert_trees_all_equal(
            raster.detector_u_axis,
            0.5 * (calibration.u_bin_edges[:-1] + calibration.u_bin_edges[1:]),
        )
        chex.assert_trees_all_equal(
            raster.detector_v_axis,
            0.5 * (calibration.v_bin_edges[:-1] + calibration.v_bin_edges[1:]),
        )
        chex.assert_trees_all_equal(raster.energy_axis, recorded_energy)
        assert raster.channel_labels == ("intensity",)

    def test_jit_success_path_preserves_counts(self) -> None:
        """Compile the whole deterministic source-to-count chain.

        The case requires compiled and eager detector carriers to agree.

        Notes
        -----
        The compiled result includes mapping, transmission, native resolution,
        and expected-count construction rather than a stage-local surrogate.
        """
        source: ArpesCube
        geometry: ExperimentGeometry
        calibration: DetectorCalibration
        effects: DetectorEffects
        source, geometry, calibration, effects = _detector_chain_fixture()
        eager: DetectorRaster = apply_detector_effects(
            (source,), geometry, calibration, effects
        )
        compiled: DetectorRaster = jax.jit(apply_detector_effects)(
            (source,), geometry, calibration, effects
        )

        chex.assert_trees_all_close(compiled, eager)
