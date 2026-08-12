"""Validate the private detector-geometry module.

The cases use analytic values, invariants, and finite differences.
"""

import chex
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float64

from diffpes.constants import (
    K_PREFACTOR_INV_ANG_SQRT_EV,
)
from diffpes.simul import detector_angles_to_kpar
from diffpes.simul._detector_geometry import (
    _analytic_angle_jacobian,
    _inverse_map_abs_jacobian,
)


class TestAnalyticDetectorJacobian:
    """Verify the detector inverse map's analytic Jacobian.

    The case compares the closed-form Jacobian with automatic differentiation
    in the horizontal-slit and vertical-slit frames.
    """

    def test_matches_autodiff_for_both_slit_frames(self) -> None:
        """Match determinant and inverse matrix against independent autodiff.

        The comparison exercises both registered slit orientations.

        Notes
        -----
        The generic off-origin point exposes every nonzero analytic term in
        both static slit branches.
        """
        angles: Float64[Array, " 2"] = jnp.array([0.17, -0.11])
        kinetic_energy: Float64[Array, ""] = jnp.array(37.0)
        momentum: Float64[Array, ""] = K_PREFACTOR_INV_ANG_SQRT_EV * jnp.sqrt(
            kinetic_energy
        )
        slit: str
        for slit in ("H", "V"):

            def coordinate_map(
                candidate: Float64[Array, " 2"],
                slit_value: str = slit,
            ) -> Float64[Array, " 2"]:
                """Compute laboratory momentum from one angle pair."""
                mapped: Float64[Array, " 2"] = detector_angles_to_kpar(
                    candidate[0], candidate[1], kinetic_energy, slit_value
                )
                return mapped

            automatic: Float64[Array, "2 2"] = jax.jacfwd(coordinate_map)(
                angles
            )
            analytic_determinant: Float64[Array, ""] = (
                _inverse_map_abs_jacobian(angles[0], angles[1], momentum, slit)
            )
            analytic_inverse: Float64[Array, "2 2"] = _analytic_angle_jacobian(
                angles[0], angles[1], momentum, slit
            )
            chex.assert_trees_all_close(
                analytic_determinant,
                jnp.abs(jnp.linalg.det(automatic)),
                rtol=1.0e-13,
                atol=0.0,
            )
            chex.assert_trees_all_close(
                analytic_inverse,
                jnp.linalg.inv(automatic),
                rtol=1.0e-13,
                atol=1.0e-14,
            )
