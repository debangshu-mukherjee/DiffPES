"""Certify the complete Plan 06 radial-profile envelope and false controls.

The tests exercise node doubling, missing tails, static boundaries, compact
modes, rejection paths, and the inverse-Angstrom unit seam.
"""

import math
from collections.abc import Callable

import chex
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jaxtyping import Array, Float
from numpy.typing import NDArray
from scipy.integrate import quad
from scipy.special import eval_genlaguerre, gamma, gammaincc, spherical_jn

from diffpes.radial import (
    evaluate_radial,
    momentum_inv_ang_to_bohr_inv,
    radial_bvals,
)
from diffpes.types import (
    FinalStateSpec,
    OrbitalBasis,
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
    directions: tuple[Array, ...] | None = None,
) -> None:
    """Compare profile values and selected parameter tangents."""
    production: RadialQuadratureSpec = make_radial_quadrature_spec()
    reference: RadialQuadratureSpec = make_radial_quadrature_spec(
        "gl2048-r120-k4-l9-reference-v1"
    )
    final_state: FinalStateSpec = make_final_state_spec()
    momenta: Array = jnp.asarray((0.0, 0.1, 0.7, 2.1, 4.0))
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
    selected_directions: tuple[Array, ...] = (
        (jnp.ones_like(parameter),) if directions is None else directions
    )
    tangent: Array
    for tangent in selected_directions:
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
    """Validate both analytic envelopes at slowest-decay boundaries.

    The test checks node doubling, parameter gradients, and missing tails.

    Notes
    -----
    It evaluates boundary carriers and independent incomplete-gamma integrals.
    """
    sto_basis: OrbitalBasis = make_orbital_basis(
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

    hydrogen_basis: OrbitalBasis = make_orbital_basis(
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
    sto_primitive_tail: float = (
        sto_norm
        * float(gamma(effective_principal + 3.0))
        * float(gammaincc(effective_principal + 3.0, exponent * cutoff))
        / exponent ** (effective_principal + 3.0)
    )
    coefficient_condition_max: float = 32.0
    sto_contraction_tail_bound: float = (
        coefficient_condition_max * sto_primitive_tail
    )
    assert sto_contraction_tail_bound < 1.57e-14

    principal: int = 7
    angular: int = 0
    charge: float = 3.5
    laguerre_order: int = principal - angular - 1
    laguerre_alpha: int = 2 * angular + 1
    hydrogen_norm: float = (2.0 * charge / principal) ** 1.5 * math.sqrt(
        math.factorial(laguerre_order)
        / (2.0 * principal * math.factorial(principal + angular))
    )

    def hydrogenic_radial_value(radius: float) -> float:
        """Return the independently evaluated signed hydrogenic radial row."""
        rho: float = 2.0 * charge * radius / principal
        radial: float = (
            hydrogen_norm
            * math.exp(-rho / 2.0)
            * rho**angular
            * float(eval_genlaguerre(laguerre_order, laguerre_alpha, rho))
        )
        return radial

    def hydrogenic_tail_integrand(radius: float) -> float:
        """Return the independent absolute missing-tail integrand."""
        return abs(hydrogenic_radial_value(radius)) * radius**3

    hydrogenic_tail: float = quad(
        hydrogenic_tail_integrand,
        cutoff,
        np.inf,
        epsabs=1.0e-18,
        epsrel=1.0e-10,
    )[0]
    assert hydrogenic_tail <= 1.04e-11
    assert hydrogenic_tail >= 1.03e-11

    tail_momenta: tuple[float, ...] = (
        0.0,
        1.0e-4,
        0.1,
        0.3,
        0.7,
        1.25,
        2.1,
        2.7,
        3.5,
        4.0,
    )
    reference_profile: RadialQuadratureSpec = make_radial_quadrature_spec(
        "gl2048-r120-k4-l9-reference-v1"
    )
    final_state: FinalStateSpec = make_final_state_spec()
    sto_reference: Float[NDArray, "n_orb n_k n_branch"] = np.asarray(
        radial_bvals(
            sto,
            jnp.asarray(tail_momenta),
            reference_profile,
            final_state,
        )
    )
    hydrogenic_reference: Float[NDArray, "n_orb n_k n_branch"] = np.asarray(
        radial_bvals(
            hydrogenic,
            jnp.asarray(tail_momenta),
            reference_profile,
            final_state,
        )
    )

    def sto_signed_tail_integrand(
        radius: float,
        momentum_value: float,
        degree: int,
    ) -> float:
        """Return one signed STO-Bessel missing-tail integrand."""
        return (
            sto_norm
            * radius ** (effective_principal + 2.0)
            * math.exp(-exponent * radius)
            * float(spherical_jn(degree, momentum_value * radius))
        )

    def hydrogenic_signed_tail_integrand(
        radius: float,
        momentum_value: float,
    ) -> float:
        """Return one signed hydrogenic-Bessel missing-tail integrand."""
        return (
            hydrogenic_radial_value(radius)
            * radius**3
            * float(spherical_jn(1, momentum_value * radius))
        )

    momentum_index: int
    momentum: float
    final_degree: int
    channel: int
    for momentum_index, momentum in enumerate(tail_momenta):
        for channel, final_degree in enumerate((3, 5)):
            sto_signed_tail: float = quad(
                sto_signed_tail_integrand,
                cutoff,
                np.inf,
                args=(momentum, final_degree),
                epsabs=1.0e-18,
                epsrel=1.0e-10,
                limit=400,
            )[0]
            sto_half_budget: float = 0.5 * (
                1.0e-12
                + 1.0e-10 * abs(sto_reference[momentum_index, 0, channel])
            )
            assert (
                coefficient_condition_max * abs(sto_signed_tail)
                <= sto_half_budget
            )

        hydrogenic_signed_tail: float = quad(
            hydrogenic_signed_tail_integrand,
            cutoff,
            np.inf,
            args=(momentum,),
            epsabs=1.0e-18,
            epsrel=1.0e-10,
            limit=400,
        )[0]
        hydrogenic_half_budget: float = 0.5 * (
            1.0e-12 + 1.0e-10 * abs(hydrogenic_reference[momentum_index, 0, 1])
        )
        assert abs(hydrogenic_signed_tail) <= hydrogenic_half_budget


def test_g16_near_condition_limit_cancellation_battery() -> None:
    """Validate an admitted STO contraction immediately below kappa 32.

    The test checks mixed coefficient/exponent directions and an absolute
    missing-tail integral. A nearby contraction over the limit must fail.

    Notes
    -----
    Use independent overlap algebra to locate the cancellation condition.
    """
    basis: OrbitalBasis = make_orbital_basis(
        atom_indices=(0,),
        n=(6,),
        l=(4,),
        m=(0,),
    )
    effective_principal: float = 4.2
    exponents: Array = jnp.asarray(((0.5, 0.52),))
    coefficients: Array = jnp.asarray(((1.0, -0.98),))
    spec: RadialSpec = make_radial_spec(
        basis,
        (0,),
        zeta_shell=exponents,
        coefficients_shell=coefficients,
        n_star_shell=(effective_principal,),
    )
    _node_doubling_with_parameter_gradient(
        spec,
        lambda item: item.zeta_shell,
        (
            jnp.ones_like(exponents),
            jnp.asarray(((1.0, -1.0),)),
        ),
    )
    _node_doubling_with_parameter_gradient(
        spec,
        lambda item: item.coefficients_shell,
        (
            jnp.ones_like(coefficients),
            jnp.asarray(((1.0, -1.0),)),
            jnp.asarray(((1.0, 0.0),)),
        ),
    )

    exponent_values: Float[NDArray, " n_prim"] = np.asarray(exponents[0])
    coefficient_values: Float[NDArray, " n_prim"] = np.asarray(coefficients[0])
    primitive_norms: Float[NDArray, " n_prim"] = (2.0 * exponent_values) ** (
        effective_principal + 0.5
    ) / math.sqrt(float(gamma(2.0 * effective_principal + 1.0)))
    overlap: Float[NDArray, "n_prim n_prim"] = (
        primitive_norms[:, None]
        * primitive_norms[None, :]
        * float(gamma(2.0 * effective_principal + 1.0))
        / (exponent_values[:, None] + exponent_values[None, :])
        ** (2.0 * effective_principal + 1.0)
    )
    contraction_norm: float = math.sqrt(
        float(coefficient_values @ overlap @ coefficient_values)
    )
    coefficient_condition: float = (
        float(np.sum(np.abs(coefficient_values))) / contraction_norm
    )
    assert coefficient_condition == pytest.approx(31.54723984446624)
    assert 31.0 < coefficient_condition < 32.0

    def contracted_tail_integrand(radius: float) -> float:
        """Return the independently normalized absolute tail integrand."""
        primitives: Float[NDArray, " n_prim"] = (
            primitive_norms
            * radius ** (effective_principal - 1.0)
            * np.exp(-exponent_values * radius)
        )
        radial: float = float(coefficient_values @ primitives)
        return abs(radial / contraction_norm) * radius**3

    missing_tail: float = quad(
        contracted_tail_integrand,
        120.0,
        np.inf,
        epsabs=1.0e-18,
        epsrel=1.0e-10,
    )[0]
    assert missing_tail <= 5.0e-13
    assert missing_tail == pytest.approx(7.005817765179824e-15, rel=1.0e-8)

    with pytest.raises(Exception, match="coefficient condition"):
        make_radial_spec(
            basis,
            (0,),
            zeta_shell=exponents,
            coefficients_shell=jnp.asarray(((1.0, -0.99),)),
            n_star_shell=(effective_principal,),
        )


def test_g16_hydrogenic_sharp_and_high_angular_gradient_battery() -> None:
    """Validate node doubling at sharp and high-angular hydrogenic edges.

    The test differentiates charge at both ends of the admitted decay range.

    Notes
    -----
    Exercise n=1, l=0 at decay four and n=7, l=4 at decay one half.
    """
    quantum_numbers: tuple[tuple[int, int, float], ...] = (
        (1, 0, 4.0),
        (7, 4, 3.5),
    )
    principal: int
    angular: int
    charge: float
    for principal, angular, charge in quantum_numbers:
        basis: OrbitalBasis = make_orbital_basis(
            atom_indices=(0,),
            n=(principal,),
            l=(angular,),
            m=(0,),
        )
        spec: RadialSpec = make_radial_spec(
            basis,
            (0,),
            mode="hydrogenic",
            effective_charge_shell=jnp.asarray((charge,)),
        )
        _node_doubling_with_parameter_gradient(
            spec,
            lambda item: item.effective_charge_shell,
        )


def test_g16_all_boundary_quantum_numbers_and_compact_modes() -> None:
    """Accept each static boundary and preserve compact-mode invariance.

    The test checks all quantum limits plus grid and fixed radial modes.

    Notes
    -----
    It enumerates certified boundaries and compares both registered profiles.
    """
    n_star_values: tuple[float, ...] = (1.0, 2.0, 3.0, 3.7, 4.0, 4.2)
    principal: int
    n_star: float
    basis: OrbitalBasis
    exponent: float
    angular: int
    decay: float
    for principal, n_star in enumerate(n_star_values, start=1):
        angular = min(4, principal - 1)
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

    s_basis: OrbitalBasis = make_orbital_basis(
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
    assert grid_spec.grid_values_shell is not None
    grid_direction: Array = jnp.cos(grid)[None, :].at[:, -1].set(0.0)

    def grid_parameter(item: RadialSpec) -> Array:
        """Return a compact grid parameter leaf."""
        assert item.grid_values_shell is not None
        return item.grid_values_shell

    _node_doubling_with_parameter_gradient(
        grid_spec,
        grid_parameter,
        (samples, grid_direction),
    )

    def fixed_parameter(item: RadialSpec) -> Array:
        """Return a fixed-integral parameter leaf."""
        assert item.fixed_integrals_shell is not None
        return item.fixed_integrals_shell

    _node_doubling_with_parameter_gradient(
        fixed_spec,
        fixed_parameter,
        (jnp.asarray(((0.0, 1.0),)),),
    )


def test_g16_profile_rejections_and_unit_false_control() -> None:
    """Reject each caller-controlled escape from the registered envelope.

    The test checks profile forgery, domain excess, and reciprocal conversion.

    Notes
    -----
    It exercises public rejection paths and one dimensionless unit identity.
    """
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
    basis: OrbitalBasis = make_orbital_basis(
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
