"""Certify the complete radial-profile envelope and false controls.

Extended Summary
----------------
The tests exercise node doubling, missing tails, static boundaries, compact
modes, rejection paths, and the inverse-Angstrom unit seam.
"""

import math

import chex
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from beartype import beartype
from beartype.typing import Callable, Tuple
from jaxtyping import Array, Complex128, Float64, jaxtyped
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


@jaxtyped(typechecker=beartype)
def _node_doubling_with_parameter_gradient(
    base: RadialSpec,
    parameter_getter: Callable[[RadialSpec], Float64[Array, " ..."]],
    directions: Tuple[Float64[Array, " ..."], ...] | None = None,
) -> None:
    """PRIVATE: Compare profile values and selected parameter tangents.

    Parameters
    ----------
    base : RadialSpec
        Shell-shared radial carrier whose envelope the check probes.
    parameter_getter : Callable[[RadialSpec], Float64[Array, " ..."]]
        Accessor that selects the differentiated leaf of ``base``.
    directions : Tuple[Float64[Array, " ..."], ...] | None
        Tangent directions for the selected leaf; ``None`` selects the
        all-ones direction.

    Notes
    -----
    Evaluates :func:`radial_bvals` on five fixed momenta in inverse
    Bohr under the production quadrature profile and under the doubled
    ``gl2048-r120-k4-l9-reference-v1`` profile. Asserts value agreement
    at ``rtol=1e-10`` and, for each direction, agreement of the JVP
    through the selected parameter at ``rtol=1e-8``. Node doubling
    leaves a converged quadrature unchanged, so any disagreement is a
    resolution defect.
    """
    production: RadialQuadratureSpec = make_radial_quadrature_spec()
    reference: RadialQuadratureSpec = make_radial_quadrature_spec(
        "gl2048-r120-k4-l9-reference-v1"
    )
    final_state: FinalStateSpec = make_final_state_spec()
    momenta: Float64[Array, " 5"] = jnp.asarray((0.0, 0.1, 0.7, 2.1, 4.0))
    parameter: Float64[Array, " ..."] = parameter_getter(base)

    @jaxtyped(typechecker=beartype)
    def _evaluated(
        value: Float64[Array, " ..."], profile: RadialQuadratureSpec
    ) -> Complex128[Array, "5 n_orb 2"]:
        """PRIVATE: Evaluate one parameter leaf under a quadrature profile.

        Parameters
        ----------
        value : Float64[Array, " ..."]
            Candidate value for the selected radial parameter leaf.
        profile : RadialQuadratureSpec
            Fixed quadrature calibration to evaluate.

        Returns
        -------
        result : Complex128[Array, "5 n_orb 2"]
            Radial channels for five momenta and every orbital.

        Notes
        -----
        Replaces the selected leaf before calling the public assembler.
        """
        candidate: RadialSpec = eqx.tree_at(
            parameter_getter,
            base,
            value,
        )
        result: Complex128[Array, "5 n_orb 2"] = radial_bvals(
            candidate, momenta, profile, final_state
        )
        return result

    production_values: Complex128[Array, "5 n_orb 2"] = _evaluated(
        parameter, production
    )
    reference_values: Complex128[Array, "5 n_orb 2"] = _evaluated(
        parameter, reference
    )
    chex.assert_trees_all_close(
        production_values,
        reference_values,
        rtol=1.0e-10,
        atol=1.0e-12,
    )
    selected_directions: Tuple[Float64[Array, " ..."], ...] = (
        (jnp.ones_like(parameter),) if directions is None else directions
    )
    tangent: Float64[Array, " ..."]
    for tangent in selected_directions:
        production_gradient: Complex128[Array, "5 n_orb 2"] = jax.jvp(
            lambda value: _evaluated(value, production),
            (parameter,),
            (tangent,),
        )[1]
        reference_gradient: Complex128[Array, "5 n_orb 2"] = jax.jvp(
            lambda value: _evaluated(value, reference),
            (parameter,),
            (tangent,),
        )[1]
        chex.assert_trees_all_close(
            production_gradient,
            reference_gradient,
            rtol=1.0e-8,
            atol=1.0e-10,
        )


def test_sto_and_hydrogenic_envelope_values_gradients_and_tails(  # noqa: PLR0915
) -> None:
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

    def _hydrogenic_radial_value(radius: float) -> float:
        """PRIVATE: Evaluate the signed hydrogenic radial row.

        Parameters
        ----------
        radius : float
            Radial coordinate in Bohr.

        Returns
        -------
        radial : float
            Signed radial wavefunction value in inverse Bohr to the
            three-halves power.

        Notes
        -----
        Evaluates the normalized generalized-Laguerre expression directly.
        """
        rho: float = 2.0 * charge * radius / principal
        radial: float = (
            hydrogen_norm
            * math.exp(-rho / 2.0)
            * rho**angular
            * float(eval_genlaguerre(laguerre_order, laguerre_alpha, rho))
        )
        return radial

    def _hydrogenic_tail_integrand(radius: float) -> float:
        """PRIVATE: Evaluate the absolute hydrogenic tail integrand.

        Parameters
        ----------
        radius : float
            Radial coordinate in Bohr.

        Returns
        -------
        result : float
            Absolute radial value multiplied by the cubic radial measure.

        Notes
        -----
        Uses the independently evaluated hydrogenic radial expression.
        """
        result: float = abs(_hydrogenic_radial_value(radius)) * radius**3
        return result

    hydrogenic_tail: float = quad(
        _hydrogenic_tail_integrand,
        cutoff,
        np.inf,
        epsabs=1.0e-18,
        epsrel=1.0e-10,
    )[0]
    assert hydrogenic_tail <= 1.04e-11
    assert hydrogenic_tail >= 1.03e-11

    tail_momenta: Tuple[float, ...] = (
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
    sto_reference: Float64[NDArray, "n_orb n_k n_branch"] = np.asarray(
        radial_bvals(
            sto,
            jnp.asarray(tail_momenta),
            reference_profile,
            final_state,
        )
    )
    hydrogenic_reference: Float64[NDArray, "n_orb n_k n_branch"] = np.asarray(
        radial_bvals(
            hydrogenic,
            jnp.asarray(tail_momenta),
            reference_profile,
            final_state,
        )
    )

    def _sto_signed_tail_integrand(
        radius: float,
        momentum_value: float,
        degree: int,
    ) -> float:
        """PRIVATE: Evaluate one signed Slater-Bessel tail integrand.

        Parameters
        ----------
        radius : float
            Radial coordinate in Bohr.
        momentum_value : float
            Momentum in inverse Bohr.
        degree : int
            Static spherical-Bessel order.

        Returns
        -------
        result : float
            Signed radial integrand at the requested radius.

        Notes
        -----
        Multiplies the normalized Slater primitive by the cubic measure and
        the selected spherical Bessel function.
        """
        result: float = (
            sto_norm
            * radius ** (effective_principal + 2.0)
            * math.exp(-exponent * radius)
            * float(spherical_jn(degree, momentum_value * radius))
        )
        return result

    def _hydrogenic_signed_tail_integrand(
        radius: float,
        momentum_value: float,
    ) -> float:
        """PRIVATE: Evaluate one signed hydrogenic-Bessel tail integrand.

        Parameters
        ----------
        radius : float
            Radial coordinate in Bohr.
        momentum_value : float
            Momentum in inverse Bohr.

        Returns
        -------
        result : float
            Signed radial integrand at the requested radius.

        Notes
        -----
        Multiplies the independent p-state radial row by the cubic measure
        and the order-one spherical Bessel function.
        """
        result: float = (
            _hydrogenic_radial_value(radius)
            * radius**3
            * float(spherical_jn(1, momentum_value * radius))
        )
        return result

    momentum_index: int
    momentum: float
    final_degree: int
    channel: int
    for momentum_index, momentum in enumerate(tail_momenta):
        for channel, final_degree in enumerate((3, 5)):
            sto_signed_tail: float = quad(
                _sto_signed_tail_integrand,
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
            _hydrogenic_signed_tail_integrand,
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


def test_near_condition_limit_cancellation() -> None:
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
    exponents: Float64[Array, "1 2"] = jnp.asarray(((0.5, 0.52),))
    coefficients: Float64[Array, "1 2"] = jnp.asarray(((1.0, -0.98),))
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

    exponent_values: Float64[NDArray, " n_prim"] = np.asarray(exponents[0])
    coefficient_values: Float64[NDArray, " n_prim"] = np.asarray(
        coefficients[0]
    )
    primitive_norms: Float64[NDArray, " n_prim"] = (2.0 * exponent_values) ** (
        effective_principal + 0.5
    ) / math.sqrt(float(gamma(2.0 * effective_principal + 1.0)))
    overlap: Float64[NDArray, "n_prim n_prim"] = (
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

    def _contracted_tail_integrand(radius: float) -> float:
        """PRIVATE: Evaluate the normalized contracted tail integrand.

        Parameters
        ----------
        radius : float
            Radial coordinate in Bohr.

        Returns
        -------
        result : float
            Absolute contracted radial row times the cubic radial measure.

        Notes
        -----
        Constructs both primitives independently and divides their contraction
        by its analytic overlap normalization.
        """
        primitives: Float64[NDArray, " n_prim"] = (
            primitive_norms
            * radius ** (effective_principal - 1.0)
            * np.exp(-exponent_values * radius)
        )
        radial: float = float(coefficient_values @ primitives)
        result: float = abs(radial / contraction_norm) * radius**3
        return result

    missing_tail: float = quad(
        _contracted_tail_integrand,
        120.0,
        np.inf,
        epsabs=1.0e-18,
        epsrel=1.0e-10,
    )[0]
    assert missing_tail <= 5.0e-13
    assert missing_tail == pytest.approx(7.005817765179824e-15, rel=1.0e-8)

    with pytest.raises(
        (eqx.EquinoxRuntimeError, jax.errors.JaxRuntimeError),
        match="coefficient condition",
    ):
        make_radial_spec(
            basis,
            (0,),
            zeta_shell=exponents,
            coefficients_shell=jnp.asarray(((1.0, -0.99),)),
            n_star_shell=(effective_principal,),
        )


def test_hydrogenic_sharp_and_high_angular_gradients() -> None:
    """Validate node doubling at sharp and high-angular hydrogenic edges.

    The test differentiates charge at both ends of the admitted decay range.

    Notes
    -----
    Exercise n=1, l=0 at decay four and n=7, l=4 at decay one half.
    """
    quantum_numbers: Tuple[Tuple[int, int, float], ...] = (
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


def test_boundary_quantum_numbers_and_compact_modes() -> None:
    """Accept each static boundary and preserve compact-mode invariance.

    The test checks all quantum limits plus grid and fixed radial modes.

    Notes
    -----
    It enumerates certified boundaries and compares both registered profiles.
    """
    n_star_values: Tuple[float, ...] = (1.0, 2.0, 3.0, 3.7, 4.0, 4.2)
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
    grid: Float64[Array, " 1201"] = jnp.linspace(0.0, 24.0, 1201)
    samples: Float64[Array, "1 1201"] = jnp.exp(-grid).at[-1].set(0.0)[None, :]
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
    grid_direction: Float64[Array, "1 1201"] = (
        jnp.cos(grid)[None, :].at[:, -1].set(0.0)
    )

    def _grid_parameter(
        item: RadialSpec,
    ) -> Float64[Array, "n_shell n_r"]:
        """PRIVATE: Return a compact grid parameter leaf.

        Parameters
        ----------
        item : RadialSpec
            Grid-mode radial specification.

        Returns
        -------
        result : Float64[Array, "n_shell n_r"]
            Stored compact-support radial samples.

        Notes
        -----
        Verifies that the optional grid leaf is present before returning it.
        The selector omits runtime checking because :func:`equinox.tree_at`
        also calls it with internal leaf wrappers.
        """
        assert item.grid_values_shell is not None
        result: Float64[Array, "n_shell n_r"] = item.grid_values_shell
        return result

    _node_doubling_with_parameter_gradient(
        grid_spec,
        _grid_parameter,
        (samples, grid_direction),
    )

    def _fixed_parameter(item: RadialSpec) -> Float64[Array, "n_shell 2"]:
        """PRIVATE: Return a fixed-integral parameter leaf.

        Parameters
        ----------
        item : RadialSpec
            Fixed-mode radial specification.

        Returns
        -------
        result : Float64[Array, "n_shell 2"]
            Stored lower- and upper-channel fixed integrals.

        Notes
        -----
        Verifies that the optional fixed-integral leaf is present before
        returning it. The selector omits runtime checking because
        :func:`equinox.tree_at` also calls it with internal leaf wrappers.
        """
        assert item.fixed_integrals_shell is not None
        result: Float64[Array, "n_shell 2"] = item.fixed_integrals_shell
        return result

    _node_doubling_with_parameter_gradient(
        fixed_spec,
        _fixed_parameter,
        (jnp.asarray(((0.0, 1.0),)),),
    )


def test_profile_rejections_and_unit_false_control() -> None:
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
    with pytest.raises(
        (eqx.EquinoxRuntimeError, jax.errors.JaxRuntimeError),
        match="certified quadrature domain",
    ):
        radial_bvals(
            spec,
            jnp.asarray(4.0001),
            base,
            make_final_state_spec(),
        )

    momentum_ang: Float64[Array, ""] = jnp.asarray(2.3)
    converted: Float64[Array, ""] = momentum_inv_ang_to_bohr_inv(momentum_ang)
    dimensionless_reference: Float64[Array, ""] = (
        momentum_ang * 0.529177210903 * 1.7
    )
    chex.assert_trees_all_close(
        converted * 1.7,
        dimensionless_reference,
        rtol=0.0,
        atol=0.0,
    )
    reciprocal_false_control: Float64[Array, ""] = (
        momentum_ang / 0.529177210903
    )
    assert not bool(
        jnp.isclose(
            converted,
            reciprocal_false_control,
            rtol=1.0e-12,
            atol=1.0e-12,
        )
    )
