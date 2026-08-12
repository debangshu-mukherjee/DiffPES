"""Validate the radial profiles contracts.

The tests cover public carrier behavior, validation, and JAX
transformations for this implementation module.
"""

import chex
import jax
import jax.numpy as jnp
import pytest
from beartype.typing import List
from jaxtyping import Array, Float64

from diffpes.types import (
    FinalStateSpec,
    RadialQuadratureSpec,
    make_final_state_spec,
    make_radial_quadrature_spec,
)
from tests._assertions import assert_rejects


class TestRadialQuadratureSpec(chex.TestCase):
    """Validate :class:`diffpes.types.RadialQuadratureSpec`.

    The case rejects a direct construction that asserts an uncertified
    quadrature tolerance.
    """

    def test_raw_constructor_rejects_self_asserted_tolerance(self) -> None:
        """Reject a profile whose claimed tolerance differs from its identity.

        The false control changes only ``value_rtol`` on the initial profile.

        Notes
        -----
        Require exact registry-property matching in the raw constructor.
        """
        with pytest.raises(ValueError, match="certified profile"):
            RadialQuadratureSpec(
                profile_id="gl1024-r120-k4-l9-v1",
                n_nodes=1024,
                r_max_bohr=120.0,
                k_max_bohr_inv=4.0,
                l_prime_max=9,
                value_rtol=1.0e-14,
                gradient_rtol=1.0e-8,
                tail_bound_method_id="analytic-exp-r120-or-compact-v1",
                coefficient_condition_max=32.0,
                min_decay_parameter=0.5,
                max_decay_parameter=4.0,
            )


class TestFinalStateSpec(chex.TestCase):
    """Validate :class:`diffpes.types.FinalStateSpec`.

    The case separates the traced Coulomb charge from the static final-state
    mode.
    """

    def test_keeps_charge_traced_and_mode_static(self) -> None:
        """Preserve a Coulomb charge as the carrier's only numerical leaf.

        The direct Coulomb selector stays static.

        Notes
        -----
        Flatten the carrier and inspect its exact fields.
        """
        spec: FinalStateSpec = make_final_state_spec(
            mode="coulomb",
            effective_charge=1.5,
        )
        leaves: List[Float64[Array, "..."]] = jax.tree.leaves(spec)
        assert len(leaves) == 1
        chex.assert_trees_all_close(leaves[0], jnp.asarray(1.5))
        assert spec.mode == "coulomb"


class TestMakeRadialQuadratureSpec(chex.TestCase):
    """Validate :func:`diffpes.types.make_radial_quadrature_spec`.

    The case selects both certified profiles and rejects an unknown profile
    identity.
    """

    def test_selects_both_profiles_and_rejects_unknown_identity(self) -> None:
        """Resolve the production and reference profiles without overrides.

        An invented identifier provides the false control.

        Notes
        -----
        Compare node counts and require explicit unknown-profile rejection.
        """
        production: RadialQuadratureSpec = make_radial_quadrature_spec()
        reference: RadialQuadratureSpec = make_radial_quadrature_spec(
            "gl2048-r120-k4-l9-reference-v1"
        )
        assert production.n_nodes == 1024
        assert reference.n_nodes == 2048
        assert production.coefficient_condition_max == 32.0
        assert production.min_decay_parameter == 0.5
        assert production.max_decay_parameter == 4.0
        with pytest.raises(ValueError, match="unknown certified"):
            make_radial_quadrature_spec("gl128-unverified")


class TestMakeFinalStateSpec(chex.TestCase):
    """Validate :func:`diffpes.types.make_final_state_spec`.

    The case rejects an incompatible plane-wave charge and an uncertified
    radial accelerator.
    """

    def test_rejects_plane_wave_charge_and_uncertified_acceleration(
        self,
    ) -> None:
        """Reject incompatible numerical and static final-state selections.

        The cases prevent a charged plane wave and tabulated Coulomb radial.

        Notes
        -----
        Exercise the traced charge check and the eager mode check.
        """
        assert_rejects(
            make_final_state_spec,
            effective_charge=jnp.asarray(0.1),
            match="require zero effective charge",
        )
        with pytest.raises(
            ValueError, match="failed the frozen radial accelerator"
        ):
            make_final_state_spec(radial_accelerator="hermite")
        with pytest.raises(
            ValueError, match="failed the frozen radial accelerator"
        ):
            FinalStateSpec(
                effective_charge=jnp.asarray(0.0),
                mode="plane_wave",
                radial_accelerator="hermite",
                table_n_points=1025,
            )
        with pytest.raises(
            ValueError, match="failed the frozen radial accelerator"
        ):
            make_final_state_spec(
                mode="coulomb",
                effective_charge=1.0,
                radial_accelerator="hermite",
            )
