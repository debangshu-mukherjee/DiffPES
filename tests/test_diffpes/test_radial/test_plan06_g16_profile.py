"""Certify the complete Plan 06 radial-profile envelope and false controls."""

import math
from collections.abc import Callable

import chex
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jaxtyping import Array
from scipy.integrate import quad
from scipy.special import eval_genlaguerre, gamma, gammaincc

from diffpes.radial import (
    evaluate_radial,
    momentum_inv_ang_to_bohr_inv,
    radial_bvals,
)
from diffpes.types import (
    RadialQuadratureSpec,
    RadialSpec,
    make_final_state_spec,
    make_orbital_basis,
    make_radial_quadrature_spec,
    make_radial_spec,
)


def _node_doubling_with_parameter_gradient(
    base: RadialSpec,
    parameter_getter: Callable[[RadialSpec], Array],
) -> None:
    """Compare profile values and full vector parameter tangents."""
    production: RadialQuadratureSpec = make_radial_quadrature_spec()
    reference: RadialQuadratureSpec = make_radial_quadrature_spec(
        "gl2048-r120-k4-l9-reference-v1"
    )
    final_state = make_final_state_spec()
    momenta: Array = jnp.asarray((0.0, 0.7, 2.1, 4.0))
    parameter: Array = parameter_getter(base)

    def evaluated(value: Array, profile: RadialQuadratureSpec) -> Array:
        candidate: RadialSpec = eqx.tree_at(
            parameter_getter,
            base,
            value,
        )
        return radial_bvals(candidate, momenta, profile, final_state)

    production_values: Array = evaluated(parameter, production)
    reference_values: Array = evaluated(parameter, reference)
    chex.assert_trees_all_close(
        production_values,
        reference_values,
        rtol=1.0e-10,
        atol=1.0e-12,
    )
    tangent: Array = jnp.ones_like(parameter)
    production_gradient: Array = jax.jvp(
        lambda value: evaluated(value, production),
        (parameter,),
        (tangent,),
    )[1]
    reference_gradient: Array = jax.jvp(
        lambda value: evaluated(value, reference),
        (parameter,),
        (tangent,),
    )[1]
    chex.assert_trees_all_close(
        production_gradient,
        reference_gradient,
        rtol=1.0e-8,
        atol=1.0e-10,
    )


def test_g16_sto_and_hydrogenic_envelope_values_gradients_and_tails() -> None:
    """Cover both analytic envelopes at their slowest-decay boundaries."""
    sto_basis = make_orbital_basis(
        atom_indices=(0,),
        n=(6,),
        l=(4,),
        m=(0,),
    )
    sto: RadialSpec = make_radial_spec(
        sto_basis,
        (0,),
        zeta_shell=jnp.asarray(((0.5,),)),
        n_star_shell=(4.2,),
    )
    _node_doubling_with_parameter_gradient(
        sto,
        lambda item: item.zeta_shell,
    )

    hydrogen_basis = make_orbital_basis(
        atom_indices=(0,),
        n=(7,),
        l=(0,),
        m=(0,),
    )
    hydrogenic: RadialSpec = make_radial_spec(
        hydrogen_basis,
        (0,),
        mode="hydrogenic",
        effective_charge_shell=jnp.asarray((3.5,)),
    )
    _node_doubling_with_parameter_gradient(
        hydrogenic,
        lambda item: item.effective_charge_shell,
    )

    cutoff: float = 120.0
    effective_principal: float = 4.2
    exponent: float = 0.5
    sto_norm: float = (2.0 * exponent) ** (
        effective_principal + 0.5
    ) / math.sqrt(float(gamma(2.0 * effective_principal + 1.0)))
    sto_tail: float = (
        sto_norm
        * float(gamma(effective_principal + 3.0))
        * float(gammaincc(effective_principal + 3.0, exponent * cutoff))
        / exponent ** (effective_principal + 3.0)
    )
    assert sto_tail < 1.57e-14

    principal: int = 7
    angular: int = 0
    charge: float = 3.5
    laguerre_order: int = principal - angular - 1
    laguerre_alpha: int = 2 * angular + 1
    hydrogen_norm: float = (2.0 * charge / principal) ** 1.5 * math.sqrt(
        math.factorial(laguerre_order)
        / (2.0 * principal * math.factorial(principal + angular))
    )

    def hydrogenic_tail_integrand(radius: float) -> float:
        """Return the independent absolute missing-tail integrand."""
        rho: float = 2.0 * charge * radius / principal
        radial: float = (
            hydrogen_norm
            * math.exp(-rho / 2.0)
            * rho**angular
            * float(eval_genlaguerre(laguerre_order, laguerre_alpha, rho))
        )
        return abs(radial) * radius**3

    hydrogenic_tail: float = quad(
        hydrogenic_tail_integrand,
        cutoff,
        np.inf,
        epsabs=1.0e-18,
        epsrel=1.0e-10,
    )[0]
    assert hydrogenic_tail <= 1.04e-11
    assert hydrogenic_tail >= 1.03e-11


def test_g16_all_boundary_quantum_numbers_and_compact_modes() -> None:
    """Accept every static boundary and keep grid/fixed profile-independent."""
    n_star_values: tuple[float, ...] = (1.0, 2.0, 3.0, 3.7, 4.0, 4.2)
    principal: int
    n_star: float
    for principal, n_star in enumerate(n_star_values, start=1):
        angular: int = min(4, principal - 1)
        basis = make_orbital_basis(
            atom_indices=(0,),
            n=(principal,),
            l=(angular,),
            m=(0,),
        )
        for exponent in (0.5, 4.0):
            spec: RadialSpec = make_radial_spec(
                basis,
                (0,),
                zeta_shell=jnp.asarray(((exponent,),)),
                n_star_shell=(n_star,),
            )
            chex.assert_tree_all_finite(
                evaluate_radial(spec, jnp.asarray((0.0, 1.0, 120.0)))
            )

    for principal in range(1, 8):
        for angular in range(min(4, principal - 1) + 1):
            basis = make_orbital_basis(
                atom_indices=(0,),
                n=(principal,),
                l=(angular,),
                m=(0,),
            )
            for decay in (0.5, 4.0):
                spec = make_radial_spec(
                    basis,
                    (0,),
                    mode="hydrogenic",
                    effective_charge_shell=jnp.asarray((decay * principal,)),
                )
                chex.assert_tree_all_finite(
                    evaluate_radial(
                        spec,
                        jnp.asarray((0.0, 1.0, 120.0)),
                    )
                )

    s_basis = make_orbital_basis(
        atom_indices=(0,),
        n=(1,),
        l=(0,),
        m=(0,),
    )
    grid: Array = jnp.linspace(0.0, 24.0, 1201)
    samples: Array = jnp.exp(-grid).at[-1].set(0.0)[None, :]
    grid_spec: RadialSpec = make_radial_spec(
        s_basis,
        (0,),
        mode="grid",
        r_grid=grid,
        grid_values_shell=samples,
    )
    fixed_spec: RadialSpec = make_radial_spec(
        s_basis,
        (0,),
        mode="fixed",
        fixed_integrals_shell=jnp.asarray(((0.0, 1.0),)),
    )
    production = make_radial_quadrature_spec()
    reference = make_radial_quadrature_spec("gl2048-r120-k4-l9-reference-v1")
    final_state = make_final_state_spec()
    momenta: Array = jnp.asarray((0.0, 1.3, 4.0))
    for spec in (grid_spec, fixed_spec):
        chex.assert_trees_all_equal(
            radial_bvals(spec, momenta, production, final_state),
            radial_bvals(spec, momenta, reference, final_state),
        )


def test_g16_profile_rejections_and_unit_false_control() -> None:
    """Reject every caller-controlled escape from the registered envelope."""
    with pytest.raises(
        ValueError,
        match="unknown certified radial quadrature profile",
    ):
        make_radial_quadrature_spec("gl256-r30-k4-l9-v0")

    base: RadialQuadratureSpec = make_radial_quadrature_spec()
    with pytest.raises(ValueError, match="certified profile"):
        RadialQuadratureSpec(
            profile_id=base.profile_id,
            n_nodes=base.n_nodes,
            r_max_bohr=120.01,
            k_max_bohr_inv=base.k_max_bohr_inv,
            l_prime_max=base.l_prime_max,
            value_rtol=base.value_rtol,
            gradient_rtol=base.gradient_rtol,
            tail_bound_method_id=base.tail_bound_method_id,
            coefficient_condition_max=base.coefficient_condition_max,
            min_decay_parameter=base.min_decay_parameter,
            max_decay_parameter=base.max_decay_parameter,
        )
    with pytest.raises(ValueError, match="certified profile"):
        RadialQuadratureSpec(
            profile_id=base.profile_id,
            n_nodes=base.n_nodes,
            r_max_bohr=base.r_max_bohr,
            k_max_bohr_inv=base.k_max_bohr_inv,
            l_prime_max=10,
            value_rtol=base.value_rtol,
            gradient_rtol=base.gradient_rtol,
            tail_bound_method_id=base.tail_bound_method_id,
            coefficient_condition_max=base.coefficient_condition_max,
            min_decay_parameter=base.min_decay_parameter,
            max_decay_parameter=base.max_decay_parameter,
        )
    basis = make_orbital_basis(
        atom_indices=(0,),
        n=(1,),
        l=(0,),
        m=(0,),
    )
    spec: RadialSpec = make_radial_spec(basis, (0,))
    with pytest.raises(Exception, match="certified quadrature domain"):
        radial_bvals(
            spec,
            jnp.asarray(4.0001),
            base,
            make_final_state_spec(),
        )

    momentum_ang: Array = jnp.asarray(2.3)
    converted: Array = momentum_inv_ang_to_bohr_inv(momentum_ang)
    dimensionless_reference: Array = momentum_ang * 0.529177210903 * 1.7
    chex.assert_trees_all_close(
        converted * 1.7,
        dimensionless_reference,
        rtol=0.0,
        atol=0.0,
    )
    reciprocal_false_control: Array = momentum_ang / 0.529177210903
    assert not bool(
        jnp.isclose(
            converted,
            reciprocal_false_control,
            rtol=1.0e-12,
            atol=1.0e-12,
        )
    )
