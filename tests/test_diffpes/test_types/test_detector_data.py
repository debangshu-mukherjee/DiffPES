"""Validate the detector data contracts.

The tests cover public carrier behavior, validation, and JAX
transformations for this implementation module.
"""

import chex
import jax
import jax.numpy as jnp
from beartype.typing import Dict
from jaxtyping import Array, Float64

from diffpes.types import (
    DetectorCalibration,
    DetectorRaster,
    make_detector_calibration,
    make_detector_raster,
)
from tests._assertions import assert_rejects

_CARTESIAN_FRAME_ID: str = "org.diffpes.frame.sample_cartesian"


class TestDetectorRaster:
    """Validate :class:`~diffpes.types.DetectorRaster` metadata.

    The carrier must preserve native axes, nonempty channels, and optional
    per-pixel quantization vectors.

    :see: :class:`~diffpes.types.DetectorRaster`
    """

    def test_stores_native_axes_and_static_metadata(self) -> None:
        """Preserve native axes, channels, and quantization vectors.

        The check also confirms static strings do not become numerical PyTree
        leaves.

        Notes
        -----
        Construct two spin channels with one detector row and inspect every
        carrier boundary.
        """
        quantization_axis: Float64[Array, "2 1 3"] = jnp.broadcast_to(
            jnp.array([0.0, 0.0, 1.0]),
            (2, 1, 3),
        )
        raster: DetectorRaster = make_detector_raster(
            expected_counts=jnp.ones((2, 2, 1, 3)),
            detector_u_axis=jnp.array([-0.1, 0.1]),
            detector_v_axis=jnp.array([0.0]),
            energy_axis=jnp.array([-1.0, 0.0, 1.0]),
            channel_labels=("spin_up", "spin_down"),
            coordinate_system="hemispherical_angles",
            quantization_axis=quantization_axis,
        )

        chex.assert_shape(raster.expected_counts, (2, 2, 1, 3))
        chex.assert_shape(raster.quantization_axis, (2, 1, 3))
        assert raster.channel_labels == ("spin_up", "spin_down")
        assert not any(
            isinstance(leaf, str) for leaf in jax.tree.leaves(raster)
        )


class TestMakeDetectorRaster:
    """Validate :func:`~diffpes.types.make_detector_raster` rejection.

    The factory must reject non-finite counts and an empty channel axis in
    eager and compiled execution.

    :see: :func:`~diffpes.types.make_detector_raster`
    """

    def test_rejects_nonfinite_and_empty_channel_counts(self) -> None:
        """Reject NaN counts and a zero-length channel axis.

        The two cases distinguish traced value validation from the static
        acquisition-shape contract.

        Notes
        -----
        Apply the shared eager/JIT rejection helper with otherwise valid
        native detector axes.
        """
        arguments: Dict[str, object] = {
            "detector_u_axis": jnp.array([-0.1, 0.1]),
            "detector_v_axis": jnp.array([0.0]),
            "energy_axis": jnp.array([-1.0, 0.0, 1.0]),
            "coordinate_system": "hemispherical_angles",
        }
        assert_rejects(
            make_detector_raster,
            expected_counts=jnp.full((1, 2, 1, 3), jnp.nan),
            channel_labels=("intensity",),
            match="expected counts finite",
            **arguments,
        )
        assert_rejects(
            make_detector_raster,
            expected_counts=jnp.empty((0, 2, 1, 3)),
            channel_labels=(),
            match="channel axis requires at least one channel",
            **arguments,
        )


class TestDetectorCalibration:
    """Validate :class:`~diffpes.types.DetectorCalibration` metadata.

    The carrier must preserve detector edges, point-spread widths, and its
    fixed coordinate and boundary policies.

    :see: :class:`~diffpes.types.DetectorCalibration`
    """

    def test_stores_edges_and_policy_metadata(self) -> None:
        """Preserve detector edges, PSFs, and policy metadata.

        The check covers one slit row and three recorded energy bins.

        Notes
        -----
        Construct a valid calibration and inspect its edge shape and static
        selector values.
        """
        calibration: DetectorCalibration = make_detector_calibration(
            u_bin_edges=jnp.array([-0.2, 0.0, 0.2]),
            v_bin_edges=jnp.array([-0.05, 0.05]),
            energy_bin_edges_ev=jnp.array([-1.5, -0.5, 0.5, 1.5]),
            psf_fwhm_u=0.01,
            psf_fwhm_v=0.02,
            psf_fwhm_energy_ev=0.05,
            transmission_reference_domain_ev=jnp.array([10.0, 30.0]),
        )

        chex.assert_shape(calibration.energy_bin_edges_ev, (4,))
        assert calibration.coordinate_system == "hemispherical_angles"
        assert calibration.boundary_policy == "loss"


class TestMakeDetectorCalibration:
    """Validate :func:`~diffpes.types.make_detector_calibration` checks.

    The factory must reject a nonpositive detector point-spread width in eager
    and compiled execution.

    :see: :func:`~diffpes.types.make_detector_calibration`
    """

    def test_rejects_nonpositive_energy_psf(self) -> None:
        """Reject a zero energy point-spread FWHM.

        The check isolates the traced positivity contract from static policy
        validation.

        Notes
        -----
        Pass valid detector edges and transmission bounds with one zero width
        through the shared eager/JIT helper.
        """
        arguments: Dict[str, object] = {
            "u_bin_edges": jnp.array([-0.2, 0.0, 0.2]),
            "v_bin_edges": jnp.array([-0.05, 0.05]),
            "energy_bin_edges_ev": jnp.array([-1.5, -0.5, 0.5, 1.5]),
            "psf_fwhm_u": 0.01,
            "psf_fwhm_v": 0.02,
            "psf_fwhm_energy_ev": 0.0,
            "transmission_reference_domain_ev": jnp.array([10.0, 30.0]),
        }
        assert_rejects(
            make_detector_calibration,
            match="energy FWHM finite and positive",
            **arguments,
        )
