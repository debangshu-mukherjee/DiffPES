"""Validate the private detector-spectrum module.

The cases use analytic values, invariants, and finite differences.
"""

import chex
import jax
import jax.numpy as jnp
import pytest
from jaxtyping import Array, Float64

from diffpes.simul._detector_map import (
    _map_and_mix_domains,
    _map_source_to_detector,
    _map_source_to_detector_with_order,
)
from diffpes.types import (
    ArpesSpectrum,
    DetectorCalibration,
    DetectorEffects,
    ExperimentGeometry,
    make_detector_calibration,
)
from tests._assertions import assert_rejects

from ._detector_map_helpers import (
    _effects,
    _geometry,
    _slit_calibration,
    _spectrum,
)


class TestSpectrumDetectorMap:
    """Verify the explicitly bound slit-line-density interpretation.

    The cases check the transverse aperture, detector rotations, mixture
    semantics, path monotonicity, and transverse-domain rejection.
    """

    def test_line_flux_includes_declared_transverse_aperture(self) -> None:
        """Recover line flux after multiplying by the explicit v width.

        Native-volume integration must reproduce the reported capture ratio.

        Notes
        -----
        The mapper divides the active ``u``/energy density by the one-bin
        aperture width.  Reintegrating native volume therefore gives the
        captured line flux and the reported fraction.
        """
        source: ArpesSpectrum = _spectrum("x")
        calibration: DetectorCalibration = _slit_calibration()
        density: Float64[Array, "u 1 e"]
        fraction: Float64[Array, ""]
        density, fraction = _map_source_to_detector(
            source, _geometry(), calibration
        )
        volumes: Float64[Array, "u 1 e"] = (
            jnp.diff(calibration.u_bin_edges)[:, None, None]
            * jnp.diff(calibration.v_bin_edges)[None, :, None]
            * jnp.diff(calibration.energy_bin_edges_ev)[None, None, :]
        )
        captured_flux: Float64[Array, ""] = jnp.sum(density * volumes)
        path_widths: Float64[Array, " k"] = jnp.array([0.2, 0.2, 0.2])
        energy_widths: Float64[Array, " e"] = jnp.array([0.2, 0.2, 0.2])
        source_flux: Float64[Array, ""] = jnp.einsum(
            "k,e,ke->", path_widths, energy_widths, source.intensity
        )
        chex.assert_trees_all_close(
            captured_flux / source_flux,
            fraction,
            rtol=1.0e-13,
            atol=0.0,
        )
        assert 0.0 < float(fraction) < 1.0

    @pytest.mark.slow
    def test_gamma_x_gamma_y_rotation_and_detector_space_mixture(self) -> None:
        """Verify Gamma-X/Gamma-Y maps before unequal-logit mixing.

        Distinct Cartesian paths reach one common native detector target.

        Notes
        -----
        The equal-length cuts carry distinct Cartesian vectors.  An active
        rotation places Gamma-Y on a valid oblique slit chart.  The weighted
        result matches an independent detector-space mixture.
        """
        gamma_x: ArpesSpectrum = _spectrum("x")
        gamma_y: ArpesSpectrum = _spectrum("y")
        geometry: ExperimentGeometry = _geometry()
        calibration: DetectorCalibration = _slit_calibration()
        rotations: Float64[Array, "d 3"] = jnp.array(
            [[0.0, 0.0, 0.0], [-jnp.pi / 3.0, 0.0, 0.0]]
        )
        effects: DetectorEffects = _effects(jnp.array([-0.4, 0.7]), rotations)
        mixed: Float64[Array, "u 1 e"]
        mixed_fraction: Float64[Array, ""]
        mixed, mixed_fraction = _map_and_mix_domains(
            (gamma_x, gamma_y), geometry, calibration, effects
        )
        first: Float64[Array, "u 1 e"] = _map_source_to_detector_with_order(
            gamma_x,
            geometry,
            calibration,
            rotations[0],
            order=4,
        )[0]
        second: Float64[Array, "u 1 e"] = _map_source_to_detector_with_order(
            gamma_y,
            geometry,
            calibration,
            rotations[1],
            order=4,
        )[0]
        weights: Float64[Array, " d"] = jax.nn.softmax(effects.domain_logits)
        expected: Float64[Array, "u 1 e"] = (
            weights[0] * first + weights[1] * second
        )
        chex.assert_trees_all_close(mixed, expected, rtol=2.0e-16, atol=0.0)
        assert 0.0 < float(mixed_fraction) < 1.0
        assert not bool(jnp.allclose(first, second, rtol=1.0e-8, atol=0.0))

    def test_rejects_nonmonotone_path_and_transverse_escape(self) -> None:
        """Reject an unresolved 2-D cut and a line outside its aperture.

        Both failures preserve the declared one-dimensional source meaning.

        Notes
        -----
        Unrotated Gamma-Y has singular ``u(s)`` in an H slit.  A partially
        rotated cut is monotone but leaves a deliberately narrow v aperture.
        Neither case invents an off-path interpolation rule.
        """
        gamma_y: ArpesSpectrum = _spectrum("y")
        assert_rejects(
            _map_source_to_detector_with_order,
            gamma_y,
            _geometry(),
            _slit_calibration(),
            jnp.zeros(3),
            match="strictly monotone",
            order=4,
        )
        narrow_aperture: DetectorCalibration = make_detector_calibration(
            u_bin_edges=jnp.array([-0.048, -0.007, 0.044]),
            v_bin_edges=jnp.array([-0.005, 0.005]),
            energy_bin_edges_ev=jnp.array([-0.18, 0.01, 0.17]),
            psf_fwhm_u=0.01,
            psf_fwhm_v=0.012,
            psf_fwhm_energy_ev=0.02,
            transmission_reference_domain_ev=jnp.array([40.0, 55.0]),
        )
        assert_rejects(
            _map_source_to_detector_with_order,
            gamma_y,
            _geometry(),
            narrow_aperture,
            jnp.array([-jnp.pi / 3.0, 0.0, 0.0]),
            match="transverse v aperture",
            order=4,
        )
