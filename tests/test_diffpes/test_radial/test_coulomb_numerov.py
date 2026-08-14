"""Validate the coulomb numerov module.

The cases use analytic values, invariants, and finite differences.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from beartype import beartype
from jaxtyping import Array, Complex128, Float64, jaxtyped

from diffpes.radial import (
    final_state_radial,
)
from diffpes.types import FinalStateSpec, make_final_state_spec


class TestFinalStateRadial:
    """Validate :func:`diffpes.radial.final_state_radial`.

    The cases cover plane and Coulomb dispatch at the origin for every
    supported final-state order. They compare the zero-charge limit with
    spherical Bessel values and differentiate a charged endpoint.

    :see: :func:`~diffpes.radial.final_state_radial`
    """

    @pytest.mark.slow
    @pytest.mark.rss_limit_mb(700)
    def test_plane_wave_limit_origin_and_charge_gradient(self) -> None:
        """Match the plane limit and retain a nonzero charge gradient.

        The test covers the origin and every supported final-state order.

        Notes
        -----
        Direct comparisons and reverse-mode differentiation verify continuity.
        """
        plane_spec: FinalStateSpec = make_final_state_spec()
        coulomb_zero_spec: FinalStateSpec = make_final_state_spec(
            mode="coulomb",
            effective_charge=0.0,
        )
        momentum: Float64[Array, ""] = jnp.asarray(1.2)
        radius: Float64[Array, " n_r"] = jnp.asarray([0.0, 1.0e-5, 0.1, 2.0])
        order: int
        for order in range(6):
            plane: Complex128[Array, " n_r"] = final_state_radial(
                order,
                momentum,
                radius,
                plane_spec,
            )
            coulomb_zero: Complex128[Array, " n_r"] = final_state_radial(
                order,
                momentum,
                radius,
                coulomb_zero_spec,
            )
            np.testing.assert_allclose(
                coulomb_zero,
                plane,
                rtol=1.0e-10,
                atol=1.0e-12,
            )
            assert bool(jnp.all(jnp.isfinite(coulomb_zero)))

        @jaxtyped(typechecker=beartype)
        def _charged_value(
            charge: Float64[Array, ""],
        ) -> Float64[Array, ""]:
            """PRIVATE: Evaluate a charged final-state endpoint.

            Parameters
            ----------
            charge : Float64[Array, ""]
                Effective Coulomb charge in units of the elementary charge.

            Returns
            -------
            result : Float64[Array, ""]
                Real p-wave radial value at the outer radius.

            Notes
            -----
            Builds a Coulomb final-state specification for each traced charge.
            """
            spec: FinalStateSpec = make_final_state_spec(
                mode="coulomb",
                effective_charge=charge,
            )
            radial: Complex128[Array, " n_r"] = final_state_radial(
                1,
                momentum,
                radius,
                spec,
            )
            result: Float64[Array, ""] = jnp.real(radial[-1])
            return result

        charge_gradient: Float64[Array, ""] = jax.grad(_charged_value)(
            jnp.asarray(0.3)
        )
        assert bool(jnp.isfinite(charge_gradient))
        assert float(jnp.abs(charge_gradient)) > 1.0e-4

    def test_coulomb_rejects_zero_momentum(self) -> None:
        """Reject zero momentum for a charged Coulomb final state.

        The test exercises the singular Sommerfeld-parameter boundary.

        Notes
        -----
        The public runtime guard supplies the expected diagnostic.
        """
        spec: FinalStateSpec = make_final_state_spec(
            mode="coulomb",
            effective_charge=0.2,
        )
        with pytest.raises(
            (eqx.EquinoxRuntimeError, jax.errors.JaxRuntimeError),
            match="positive momentum",
        ):
            final_state_radial(
                0,
                jnp.asarray(0.0),
                jnp.asarray([0.0, 1.0]),
                spec,
            )
