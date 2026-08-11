"""Verify spectral-tail and streamed-source carrier ownership.

The tests exercise scalar tail storage, compact transition scheduling, public
factory construction, and static axis rejection without evaluating physics.
"""

import chex
import jax.numpy as jnp
import pytest
from beartype import beartype
from jaxtyping import Array, Float64, TypeCheckError, jaxtyped

from diffpes.types import (
    MatrixElementParams,
    OrbitalBasis,
    Power2TailSpec,
    RadialQuadratureSpec,
    RadialSpec,
    TransitionSourceSchedule,
    make_final_state_spec,
    make_matrix_element_params,
    make_orbital_basis,
    make_power2_tail_spec,
    make_radial_quadrature_spec,
    make_radial_spec,
    make_transition_source_schedule,
)


@jaxtyped(typechecker=beartype)
def _schedule() -> TransitionSourceSchedule:
    """PRIVATE: Build one valid compact transition-source schedule.

    Returns
    -------
    schedule : TransitionSourceSchedule
        Two-momentum, three-energy, one-orbital schedule.

    Notes
    -----
    Construct every nested carrier through its public factory. Use explicit
    float64 and complex128 arrays for all numerical schedule leaves.
    """
    basis: OrbitalBasis = make_orbital_basis(
        atom_indices=(0,),
        n=(1,),
        l=(0,),
        m=(0,),
    )
    radial: RadialSpec = make_radial_spec(
        basis,
        (0,),
        mode="fixed",
        fixed_integrals_shell=jnp.asarray([[0.0, 1.0]], dtype=jnp.float64),
    )
    matrix_element: MatrixElementParams = make_matrix_element_params(
        basis,
        (0,),
    )
    quadrature: RadialQuadratureSpec = make_radial_quadrature_spec()
    schedule: TransitionSourceSchedule = make_transition_source_schedule(
        k_i_cart=jnp.asarray(
            [[0.1, 0.0, 0.0], [0.2, 0.0, 0.0]],
            dtype=jnp.float64,
        ),
        final_norm=jnp.asarray([1.0, 1.1, 1.2], dtype=jnp.float64),
        emission_energy_valid=jnp.asarray([True, True, False]),
        positions_cart=jnp.zeros((1, 3), dtype=jnp.float64),
        depths=jnp.zeros((1,), dtype=jnp.float64),
        polarization_sample_cart=jnp.asarray(
            [1.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
            dtype=jnp.complex128,
        ),
        mean_free_path_ang=jnp.asarray(10.0, dtype=jnp.float64),
        radial=radial,
        matrix_element=matrix_element,
        quadrature=quadrature,
        final_state=make_final_state_spec(),
    )
    return schedule


class TestPower2TailSpec:
    """Verify :class:`diffpes.types.Power2TailSpec` scalar storage.

    The cases cover exact field order and non-scalar structural rejection.
    """

    def test_stores_six_scalar_coefficients(self) -> None:
        """Preserve every scalar coefficient without numerical conversion.

        The carrier must expose the left and right parameters exactly.

        Notes
        -----
        Build the carrier directly with six distinct float64 scalars. Compare
        their stacked values against the construction order.
        """
        values: Float64[Array, " 6"] = jnp.arange(
            1.0,
            7.0,
            dtype=jnp.float64,
        )
        spec: Power2TailSpec = Power2TailSpec(*values)
        actual: Float64[Array, " 6"] = jnp.stack(
            (
                spec.amplitude_left,
                spec.alpha_left,
                spec.beta_left,
                spec.amplitude_right,
                spec.alpha_right,
                spec.beta_right,
            )
        )
        chex.assert_trees_all_equal(actual, values)

    def test_rejects_a_nonscalar_coefficient(self) -> None:
        """Reject a tail coefficient with an extra array axis.

        The structural check must prevent an ambiguous broadcast contract.

        Notes
        -----
        Construct the carrier directly so runtime type checking receives one
        two-element coefficient and raises before use.
        """
        scalar: Float64[Array, ""] = jnp.asarray(1.0)
        vector: Float64[Array, " 2"] = jnp.ones(2)
        with pytest.raises(TypeCheckError, match="amplitude_left"):
            Power2TailSpec(
                vector,
                scalar,
                scalar,
                scalar,
                scalar,
                scalar,
            )


class TestMakePower2TailSpec:
    """Verify :func:`diffpes.types.make_power2_tail_spec` construction.

    The case binds the public factory to the types-owned carrier.
    """

    def test_returns_the_types_owned_carrier(self) -> None:
        """Return a Power2TailSpec with exact float64 leaves.

        The factory must preserve a derived beta coefficient exactly.

        Notes
        -----
        Pass six scalar arrays through the runtime-typechecked factory. Inspect
        the returned class and one representative field.
        """
        scalar: Float64[Array, ""] = jnp.asarray(0.25)
        spec: Power2TailSpec = make_power2_tail_spec(
            scalar,
            scalar,
            scalar,
            scalar,
            scalar,
            scalar,
        )
        assert isinstance(spec, Power2TailSpec)
        chex.assert_trees_all_equal(spec.beta_right, scalar)


class TestTransitionSourceSchedule:
    """Verify :class:`diffpes.types.TransitionSourceSchedule` axes.

    The cases cover the compact momentum, energy, and orbital dimensions.
    """

    def test_preserves_only_compact_schedule_arrays(self) -> None:
        """Preserve separate compact momentum and energy schedule arrays.

        The schedule must expose independent K and energy axes.

        Notes
        -----
        Build the shared fixture and inspect every shape that determines the
        streamed source schedule. No physics routine executes.
        """
        schedule: TransitionSourceSchedule = _schedule()
        chex.assert_shape(schedule.k_i_cart, (2, 3))
        chex.assert_shape(schedule.final_norm, (3,))
        chex.assert_shape(schedule.emission_energy_valid, (3,))
        chex.assert_shape(schedule.positions_cart, (1, 3))
        chex.assert_shape(schedule.depths, (1,))


class TestMakeTransitionSourceSchedule:
    """Verify :func:`diffpes.types.make_transition_source_schedule` checks.

    The cases bind public construction and static shape rejection.
    """

    def test_rejects_mismatched_orbital_depths(self) -> None:
        """Reject a depth axis that disagrees with the orbital positions.

        The factory must fail before a streamed matrix-element evaluation.

        Notes
        -----
        Reuse every valid nested carrier from the shared schedule. Plant a
        second depth entry while retaining one orbital position.
        """
        valid: TransitionSourceSchedule = _schedule()
        with pytest.raises(TypeCheckError, match="depths"):
            make_transition_source_schedule(
                k_i_cart=valid.k_i_cart,
                final_norm=valid.final_norm,
                emission_energy_valid=valid.emission_energy_valid,
                positions_cart=valid.positions_cart,
                depths=jnp.zeros((2,), dtype=jnp.float64),
                polarization_sample_cart=valid.polarization_sample_cart,
                mean_free_path_ang=valid.mean_free_path_ang,
                radial=valid.radial,
                matrix_element=valid.matrix_element,
                quadrature=valid.quadrature,
                final_state=valid.final_state,
            )
