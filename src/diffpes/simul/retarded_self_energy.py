"""Evaluate causal retarded self-energy models.

Extended Summary
----------------
This module evaluates analytic and grid self-energy carriers.
It also evaluates polynomial and Fermi-liquid carriers.

Routine Listings
----------------
:func:`evaluate_self_energy`
    Evaluate the complex retarded self-energy for one causal model.
"""

from functools import partial

import jax
import jax.numpy as jnp
from beartype import beartype
from beartype.typing import Any, Optional, Tuple
from jaxtyping import Array, Complex128, Float64, jaxtyped

from diffpes.types import Power2TailSpec, SelfEnergyModel

from ._kramers_kronig import (
    _kk_transform_impl,
    _materialize_tangent,
    _tangent_is_symbolic_zero,
)
from ._principal_value import (
    _check_trusted_interval,
    _cubic_core_pv,
    _cubic_edge_slopes,
    _hat_core_pv,
    _power2_spec_from_edges,
    _power2_tail_pv,
)


def _dynamic_imag(
    mode: str,
    coefficients: Float64[Array, " n_coef"],
    points: Float64[Array, " n_points"],
) -> Float64[Array, " n_points"]:
    """PRIVATE: Evaluate the transform-side dynamic imaginary part.

    The Fermi-liquid mode excludes its constant baseline, because a
    constant has a vanishing subtracted transform. The polynomial mode
    keeps its complete strictly negative softplus profile.

    Parameters
    ----------
    mode : str
        Smooth carrier mode, ``poly`` or ``fermi_liquid``.
    coefficients : Float64[Array, " n_coef"]
        Unconstrained raw model coordinates.
    points : Float64[Array, " n_points"]
        Evaluation energies in eV.

    Returns
    -------
    dynamic : Float64[Array, " n_points"]
        Dynamic imaginary part at every point in eV.

    Raises
    ------
    ValueError
        If the mode has no numerical transform contract.

    Notes
    -----
    The softplus map keeps every profile strictly negative, so the
    tail amplitudes stay strictly positive at both edges.
    """
    if mode == "fermi_liquid":
        beta: Float64[Array, ""] = jnp.logaddexp(coefficients[1], 0.0)
        omega_c: Float64[Array, ""] = jnp.logaddexp(coefficients[2], 0.0)
        dynamic: Float64[Array, " n_points"] = (
            -beta * points**2 / (1.0 + (points / omega_c) ** 4)
        )
    elif mode == "poly":
        dynamic = -jnp.logaddexp(jnp.polyval(coefficients, points), 0.0)
    else:
        msg: str = f"mode {mode!r} has no numerical transform contract"
        raise ValueError(msg)
    return dynamic


def _dynamic_imag_derivative(
    mode: str,
    coefficients: Float64[Array, " n_coef"],
    points: Float64[Array, " n_points"],
) -> Float64[Array, " n_points"]:
    """PRIVATE: Differentiate the dynamic imaginary part analytically.

    The composite frequency-derivative route consumes these analytic
    samples instead of differentiating the discrete interpolant.

    Parameters
    ----------
    mode : str
        Smooth carrier mode, ``poly`` or ``fermi_liquid``.
    coefficients : Float64[Array, " n_coef"]
        Unconstrained raw model coordinates.
    points : Float64[Array, " n_points"]
        Evaluation energies in eV.

    Returns
    -------
    derivative : Float64[Array, " n_points"]
        Analytic frequency derivative at every point.

    Raises
    ------
    ValueError
        If the mode has no numerical transform contract.

    Notes
    -----
    The Fermi-liquid branch evaluates
    ``2*beta*w*(q - 1)/(1 + q)**2`` with ``q = (w/omega_c)**4``. The
    polynomial branch chains the sigmoid with the derivative
    polynomial.
    """
    if mode == "fermi_liquid":
        beta: Float64[Array, ""] = jnp.logaddexp(coefficients[1], 0.0)
        omega_c: Float64[Array, ""] = jnp.logaddexp(coefficients[2], 0.0)
        quartic: Float64[Array, " n_points"] = (points / omega_c) ** 4
        derivative: Float64[Array, " n_points"] = (
            2.0 * beta * points * (quartic - 1.0) / (1.0 + quartic) ** 2
        )
    elif mode == "poly":
        profile: Float64[Array, " n_points"] = jnp.polyval(
            coefficients, points
        )
        degree: int = coefficients.shape[0] - 1
        if degree == 0:
            derivative = jnp.zeros_like(points)
        else:
            slope_coefficients: Float64[Array, " n_deriv"] = coefficients[
                :-1
            ] * jnp.arange(degree, 0, -1)
            derivative = -jax.nn.sigmoid(profile) * jnp.polyval(
                slope_coefficients, points
            )
    else:
        msg: str = f"mode {mode!r} has no numerical transform contract"
        raise ValueError(msg)
    return derivative


def _frozen_base_grid(
    model_domain: Float64[Array, " 2"],
    n_kk: int,
) -> Tuple[Float64[Array, " n_kk"], Float64[Array, ""]]:
    """PRIVATE: Construct the frozen uniform base grid on the carrier domain.

    The grid follows the index construction ``x_j = a + j * h`` with
    ``h = (b - a) / (n_kk - 1)``. The construction never reads the
    query window.

    Parameters
    ----------
    model_domain : Float64[Array, " 2"]
        Increasing carrier domain ``[a, b]`` in eV.
    n_kk : int
        Static number of base grid nodes.

    Returns
    -------
    grid_and_spacing : Tuple[Float64[Array, " n_kk"], Float64[Array, ""]]
        Base grid nodes and the uniform spacing ``h``.

    Notes
    -----
    The spacing evaluates first, so refinements embed the base nodes
    through the shared index expression.
    """
    spacing: Float64[Array, ""] = (model_domain[1] - model_domain[0]) / (
        n_kk - 1
    )
    grid: Float64[Array, " n_kk"] = model_domain[0] + spacing * jnp.arange(
        n_kk, dtype=jnp.float64
    )
    grid_and_spacing: Tuple[Float64[Array, " n_kk"], Float64[Array, ""]] = (
        grid,
        spacing,
    )
    return grid_and_spacing


def _smooth_real_impl(
    mode: str,
    n_kk: int,
    omega_rel_fermi_ev: Float64[Array, " n_omega"],
    coefficients: Float64[Array, " n_coef"],
    tail_raw: Float64[Array, " 2"],
    subtraction: Float64[Array, ""],
    model_domain: Float64[Array, " 2"],
) -> Float64[Array, " n_omega"]:
    """PRIVATE: Evaluate the subtracted smooth-mode real part without a rule.

    The routine samples the dynamic imaginary part on the frozen grid.
    It derives the C1 tail contract from the cubic edge stencils and
    evaluates the transform at the stacked queries. The subtraction
    happens after the transform at the carrier subtraction point.

    Parameters
    ----------
    mode : str
        Smooth carrier mode, ``poly`` or ``fermi_liquid``.
    n_kk : int
        Static number of base grid nodes.
    omega_rel_fermi_ev : Float64[Array, " n_omega"]
        Query energies relative to the Fermi level in eV.
    coefficients : Float64[Array, " n_coef"]
        Unconstrained raw model coordinates.
    tail_raw : Float64[Array, " 2"]
        Raw delta-beta tail coordinates, left then right.
    subtraction : Float64[Array, ""]
        Carrier subtraction point in eV.
    model_domain : Float64[Array, " 2"]
        Increasing carrier domain ``[a, b]`` in eV.

    Returns
    -------
    real_subtracted : Float64[Array, " n_omega"]
        Subtracted real part at every query in eV.

    Notes
    -----
    The custom-rule wrapper shares this body, so the primal and the
    derivative rule evaluate identical validated values.
    """
    grid: Float64[Array, " n_kk"]
    spacing: Float64[Array, ""]
    grid, spacing = _frozen_base_grid(model_domain, n_kk)
    values: Float64[Array, " n_kk"] = _dynamic_imag(mode, coefficients, grid)
    slope_left: Float64[Array, ""]
    slope_right: Float64[Array, ""]
    slope_left, slope_right = _cubic_edge_slopes(values, spacing)
    spec: Power2TailSpec = _power2_spec_from_edges(
        values[0],
        slope_left,
        values[-1],
        slope_right,
        tail_raw[0],
        tail_raw[1],
    )
    stacked: Float64[Array, " n_stacked"] = jnp.concatenate(
        [omega_rel_fermi_ev, subtraction[None]]
    )
    total: Float64[Array, " n_stacked"] = _kk_transform_impl(
        (grid, values), model_domain, spec, stacked, 256
    )
    real_subtracted: Float64[Array, " n_omega"] = total[:-1] - total[-1]
    return real_subtracted


def _smooth_query_composite(
    mode: str,
    n_kk: int,
    points: Float64[Array, " n_points"],
    coefficients: Float64[Array, " n_coef"],
    tail_raw: Float64[Array, " 2"],
    model_domain: Float64[Array, " 2"],
) -> Float64[Array, " n_points"]:
    """PRIVATE: Evaluate the analytic composite frequency derivative.

    The rule applies the core operator to the analytic mode-supplied
    derivative samples. It adds the finite-core boundary terms
    ``(1 / pi) * [Sigma''(a) / (a - w) - Sigma''(b) / (b - w)]``. It
    finally adds the exact forward-mode derivative of both tails.

    Parameters
    ----------
    mode : str
        Smooth carrier mode, ``poly`` or ``fermi_liquid``.
    n_kk : int
        Static number of base grid nodes.
    points : Float64[Array, " n_points"]
        Stacked evaluation energies in eV.
    coefficients : Float64[Array, " n_coef"]
        Unconstrained raw model coordinates.
    tail_raw : Float64[Array, " 2"]
        Raw delta-beta tail coordinates, left then right.
    model_domain : Float64[Array, " 2"]
        Increasing carrier domain ``[a, b]`` in eV.

    Returns
    -------
    composite : Float64[Array, " n_points"]
        Composite frequency derivative of the unsubtracted transform.

    Notes
    -----
    The derivative of the transform equals the transform of the
    derivative plus boundary terms, by partial integration on the
    finite core.
    """
    grid: Float64[Array, " n_kk"]
    spacing: Float64[Array, ""]
    grid, spacing = _frozen_base_grid(model_domain, n_kk)
    values: Float64[Array, " n_kk"] = _dynamic_imag(mode, coefficients, grid)
    derivative_samples: Float64[Array, " n_kk"] = _dynamic_imag_derivative(
        mode, coefficients, grid
    )
    core_derivative: Float64[Array, " n_points"] = _cubic_core_pv(
        grid, derivative_samples, points
    )
    boundary: Float64[Array, " n_points"] = (
        values[0] / (model_domain[0] - points)
        - values[-1] / (model_domain[1] - points)
    ) / jnp.pi
    slope_left: Float64[Array, ""]
    slope_right: Float64[Array, ""]
    slope_left, slope_right = _cubic_edge_slopes(values, spacing)
    spec: Power2TailSpec = _power2_spec_from_edges(
        values[0],
        slope_left,
        values[-1],
        slope_right,
        tail_raw[0],
        tail_raw[1],
    )

    def _tail_only(
        stacked: Float64[Array, " n_points"],
    ) -> Float64[Array, " n_points"]:
        """PRIVATE: Evaluate both tail quadratures for the derivative closure.

        Parameters
        ----------
        stacked : Float64[Array, " n_points"]
            Stacked evaluation energies in eV.

        Returns
        -------
        contribution : Float64[Array, " n_points"]
            Unsubtracted tail contribution at every point.

        Notes
        -----
        Forward-mode differentiation of this closure supplies the
        exact tail derivative.
        """
        contribution: Float64[Array, " n_points"] = _power2_tail_pv(
            model_domain, spec, stacked, 256
        )
        return contribution

    tail_derivative: Float64[Array, " n_points"]
    _, tail_derivative = jax.jvp(
        _tail_only, (points,), (jnp.ones_like(points),)
    )
    composite: Float64[Array, " n_points"] = (
        core_derivative + boundary + tail_derivative
    )
    return composite


@partial(jax.custom_jvp, nondiff_argnums=(0, 1))
def _smooth_real_subtracted(
    mode: str,
    n_kk: int,
    omega_rel_fermi_ev: Float64[Array, " n_omega"],
    coefficients: Float64[Array, " n_coef"],
    tail_raw: Float64[Array, " 2"],
    subtraction: Float64[Array, ""],
    model_domain: Float64[Array, " 2"],
) -> Float64[Array, " n_omega"]:
    """PRIVATE: Evaluate the subtracted smooth-mode real part with its rule.

    The custom derivative rule binds the public frequency derivative to
    the analytic composite route. Its transpose supplies reverse mode,
    so public ``jax.jvp`` and ``jax.grad`` share one contract.

    Parameters
    ----------
    mode : str
        Smooth carrier mode, ``poly`` or ``fermi_liquid``.
    n_kk : int
        Static number of base grid nodes.
    omega_rel_fermi_ev : Float64[Array, " n_omega"]
        Query energies relative to the Fermi level in eV.
    coefficients : Float64[Array, " n_coef"]
        Unconstrained raw model coordinates.
    tail_raw : Float64[Array, " 2"]
        Raw delta-beta tail coordinates, left then right.
    subtraction : Float64[Array, ""]
        Carrier subtraction point in eV.
    model_domain : Float64[Array, " 2"]
        Increasing carrier domain ``[a, b]`` in eV.

    Returns
    -------
    real_subtracted : Float64[Array, " n_omega"]
        Subtracted real part at every query in eV.

    Notes
    -----
    The wrapper defers to the shared implementation body. Only the
    attached derivative rule distinguishes it from the plain call.
    """
    real_subtracted: Float64[Array, " n_omega"] = _smooth_real_impl(
        mode,
        n_kk,
        omega_rel_fermi_ev,
        coefficients,
        tail_raw,
        subtraction,
        model_domain,
    )
    return real_subtracted


@partial(_smooth_real_subtracted.defjvp, symbolic_zeros=True)
def _smooth_real_subtracted_jvp(
    mode: str,
    n_kk: int,
    primals: Any,
    tangents: Any,
) -> Tuple[Float64[Array, " n_omega"], Float64[Array, " n_omega"]]:
    """PRIVATE: Bind the public frequency tangent to the composite route.

    Frequency and subtraction tangents multiply the analytic composite
    derivative. Coefficient, tail, and domain tangents flow through the
    primal linearization at fixed frequencies. Symbolic-zero detection
    skips every unperturbed argument group.

    Parameters
    ----------
    mode : str
        Smooth carrier mode, ``poly`` or ``fermi_liquid``.
    n_kk : int
        Static number of base grid nodes.
    primals : Any
        Primal inputs ``(omega, coefficients, tail_raw, subtraction,
        domain)``.
    tangents : Any
        Matching tangent structure for the primal inputs.

    Returns
    -------
    pair : Tuple[Float64[Array, " n_omega"], Float64[Array, " n_omega"]]
        Primal subtracted real part and its tangent.

    Notes
    -----
    The composite evaluates once on the stacked queries. The last row
    carries the subtraction-point derivative with a negative sign.
    """
    omega_rel_fermi_ev: Float64[Array, " n_omega"]
    coefficients: Float64[Array, " n_coef"]
    tail_raw: Float64[Array, " 2"]
    subtraction: Float64[Array, ""]
    model_domain: Float64[Array, " 2"]
    (
        omega_rel_fermi_ev,
        coefficients,
        tail_raw,
        subtraction,
        model_domain,
    ) = primals
    omega_tangent: Any
    coefficient_tangent: Any
    tail_tangent: Any
    subtraction_tangent: Any
    domain_tangent: Any
    (
        omega_tangent,
        coefficient_tangent,
        tail_tangent,
        subtraction_tangent,
        domain_tangent,
    ) = tangents
    primal_out: Float64[Array, " n_omega"] = _smooth_real_impl(
        mode,
        n_kk,
        omega_rel_fermi_ev,
        coefficients,
        tail_raw,
        subtraction,
        model_domain,
    )
    tangent_out: Float64[Array, " n_omega"] = jnp.zeros_like(primal_out)

    def _fixed_frequencies(
        raw_coefficients: Float64[Array, " n_coef"],
        raw_tail: Float64[Array, " 2"],
        domain: Float64[Array, " 2"],
    ) -> Float64[Array, " n_omega"]:
        """PRIVATE: Evaluate the real part with every frequency held fixed.

        Parameters
        ----------
        raw_coefficients : Float64[Array, " n_coef"]
            Unconstrained raw model coordinates.
        raw_tail : Float64[Array, " 2"]
            Raw delta-beta tail coordinates, left then right.
        domain : Float64[Array, " 2"]
            Increasing carrier domain ``[a, b]`` in eV.

        Returns
        -------
        value : Float64[Array, " n_omega"]
            Subtracted real part at the closed-over queries.

        Notes
        -----
        Linearizing this closure yields the exact parameter tangents.
        """
        value: Float64[Array, " n_omega"] = _smooth_real_impl(
            mode,
            n_kk,
            omega_rel_fermi_ev,
            raw_coefficients,
            raw_tail,
            subtraction,
            domain,
        )
        return value

    parameter_perturbed: bool = not (
        _tangent_is_symbolic_zero(coefficient_tangent)
        and _tangent_is_symbolic_zero(tail_tangent)
        and _tangent_is_symbolic_zero(domain_tangent)
    )
    if parameter_perturbed:
        parameter_tangent: Float64[Array, " n_omega"]
        _, parameter_tangent = jax.jvp(
            _fixed_frequencies,
            (coefficients, tail_raw, model_domain),
            (
                _materialize_tangent(coefficient_tangent),
                _materialize_tangent(tail_tangent),
                _materialize_tangent(domain_tangent),
            ),
        )
        tangent_out = tangent_out + parameter_tangent
    frequency_perturbed: bool = not (
        _tangent_is_symbolic_zero(omega_tangent)
        and _tangent_is_symbolic_zero(subtraction_tangent)
    )
    if frequency_perturbed:
        stacked: Float64[Array, " n_stacked"] = jnp.concatenate(
            [omega_rel_fermi_ev, subtraction[None]]
        )
        composite: Float64[Array, " n_stacked"] = _smooth_query_composite(
            mode, n_kk, stacked, coefficients, tail_raw, model_domain
        )
        tangent_out = (
            tangent_out
            + composite[:-1] * _materialize_tangent(omega_tangent)
            - composite[-1] * _materialize_tangent(subtraction_tangent)
        )
    pair: Tuple[Float64[Array, " n_omega"], Float64[Array, " n_omega"]] = (
        primal_out,
        tangent_out,
    )
    return pair


def _hat_real_subtracted(
    omega_rel_fermi_ev: Float64[Array, " n_omega"],
    nodes: Float64[Array, " n_nodes"],
    coefficients: Float64[Array, " n_nodes"],
    tail_raw: Float64[Array, " 2"],
    subtraction: Float64[Array, ""],
    model_domain: Float64[Array, " 2"],
    n_kk: int,
) -> Float64[Array, " n_omega"]:
    """PRIVATE: Evaluate the exact subtracted grid-mode real part.

    The hat interpolant owns grid mode. The exact piecewise-linear
    transform runs on the carrier nodes, and the tail slopes come from
    the outer hat segments. The cubic reconstruction never touches this
    carrier class, because it can overshoot between negative samples.

    Parameters
    ----------
    omega_rel_fermi_ev : Float64[Array, " n_omega"]
        Query energies relative to the Fermi level in eV.
    nodes : Float64[Array, " n_nodes"]
        Strictly increasing carrier nodes in eV.
    coefficients : Float64[Array, " n_nodes"]
        Unconstrained raw hat coordinates.
    tail_raw : Float64[Array, " 2"]
        Raw delta-beta tail coordinates, left then right.
    subtraction : Float64[Array, ""]
        Carrier subtraction point in eV.
    model_domain : Float64[Array, " 2"]
        Increasing carrier domain ``[a, b]`` in eV.
    n_kk : int
        Static node count that fixes the trusted-interval margin.

    Returns
    -------
    real_subtracted : Float64[Array, " n_omega"]
        Subtracted real part at every query in eV.

    Notes
    -----
    The trusted-interval margin derives from the frozen base spacing
    ``(b - a) / (n_kk - 1)`` even though the hat core runs on the
    carrier nodes.
    """
    spacing: Float64[Array, ""] = (model_domain[1] - model_domain[0]) / (
        n_kk - 1
    )
    ordinates: Float64[Array, " n_nodes"] = -jnp.logaddexp(coefficients, 0.0)
    slope_left: Float64[Array, ""] = (ordinates[1] - ordinates[0]) / (
        nodes[1] - nodes[0]
    )
    slope_right: Float64[Array, ""] = (ordinates[-1] - ordinates[-2]) / (
        nodes[-1] - nodes[-2]
    )
    spec: Power2TailSpec = _power2_spec_from_edges(
        ordinates[0],
        slope_left,
        ordinates[-1],
        slope_right,
        tail_raw[0],
        tail_raw[1],
    )
    stacked: Float64[Array, " n_stacked"] = jnp.concatenate(
        [omega_rel_fermi_ev, subtraction[None]]
    )
    checked: Float64[Array, " n_stacked"] = _check_trusted_interval(
        stacked, model_domain, spacing
    )
    core: Float64[Array, " n_stacked"] = _hat_core_pv(
        nodes, ordinates, checked
    )
    tails: Float64[Array, " n_stacked"] = _power2_tail_pv(
        model_domain, spec, checked, 256
    )
    total: Float64[Array, " n_stacked"] = core + tails
    real_subtracted: Float64[Array, " n_omega"] = total[:-1] - total[-1]
    return real_subtracted


def _kink_real_part(
    points: Float64[Array, " n_points"],
    coupling: Float64[Array, ""],
    omega_0: Float64[Array, ""],
    width: Float64[Array, ""],
) -> Float64[Array, " n_points"]:
    """PRIVATE: Evaluate the analytic bosonic-kink real pole pair.

    Parameters
    ----------
    points : Float64[Array, " n_points"]
        Evaluation energies in eV.
    coupling : Float64[Array, ""]
        Positive kink coupling in eV.
    omega_0 : Float64[Array, ""]
        Positive boson energy in eV.
    width : Float64[Array, ""]
        Positive pole width in eV.

    Returns
    -------
    real : Float64[Array, " n_points"]
        Analytic real part at every point in eV.

    Notes
    -----
    The pair reads ``g**2 * Re[1 / (w - w0 + i*W) + 1 / (w + w0 +
    i*W)]`` and vanishes at zero frequency.
    """
    lower: Float64[Array, " n_points"] = points - omega_0
    upper: Float64[Array, " n_points"] = points + omega_0
    real: Float64[Array, " n_points"] = coupling**2 * (
        lower / (lower**2 + width**2) + upper / (upper**2 + width**2)
    )
    return real


@jaxtyped(typechecker=beartype)
def evaluate_self_energy(  # noqa: DOC502, DOC503 -- JAX runtime guards.
    omega_rel_fermi_ev: Float64[Array, " n_omega"],
    model: SelfEnergyModel,
    n_kk: int = 4096,
) -> Complex128[Array, " n_omega"]:
    r"""Evaluate the complex retarded self-energy for one causal model.

    The function returns :math:`\Sigma(E - E_F) = \Sigma' + i\Sigma''`
    for every carrier mode. Constant mode returns a purely imaginary
    result with an exactly zero subtracted real part. The bosonic kink
    evaluates its analytic complex pole pair. The numerical modes
    ``poly``, ``grid``, and ``fermi_liquid`` obtain the subtracted real
    part from the certified cell-integrated Kramers--Kronig operator.
    That operator lives on the declared carrier domain and carries C1
    ``power2`` tails.

    :see: :class:`~.test_retarded_self_energy.TestEvaluateSelfEnergy`

    Implementation Logic
    --------------------
    1. **Dispatch on the static carrier mode**::

           mode = model.mode

       The Python string selects one code path outside tracing.
    2. **Evaluate the analytic modes directly**::

           real = jnp.zeros_like(omega_rel_fermi_ev)
           real = _kink_real_part(omega, g, omega_0, width) - baseline

       Constant mode keeps an exactly zero subtracted real part. The
       kink subtracts its pole pair at the carrier subtraction point.
    3. **Evaluate the numerical modes through the transform**::

           real = _smooth_real_subtracted(mode, n_kk, omega, ...)
           real = _hat_real_subtracted(omega, nodes, ...)

       The frozen grid comes from ``kk_domain_rel_fermi_ev``, never
       from the query extrema. Grid mode uses the exact hat transform.
    4. **Assemble the complex retarded result**::

           result = jax.lax.complex(real, imag)

       The imaginary part evaluates the mode closed form directly.

    Parameters
    ----------
    omega_rel_fermi_ev : Float64[Array, " n_omega"]
        Query energies relative to the Fermi level in eV. Numerical
        modes require every query inside the trusted interval
        ``[a + 2h, b - 2h]``.
    model : SelfEnergyModel
        Validated causal self-energy carrier.
    n_kk : int, optional
        Static internal Kramers--Kronig grid length. Default is 4096.

    Returns
    -------
    sigma : Complex128[Array, " n_omega"]
        Complex retarded self-energy at every query in eV.

    Raises
    ------
    ValueError
        If ``n_kk`` cannot support the certified operator stencils.
    EquinoxRuntimeError
        If one query or the subtraction point leaves the trusted
        interval, eagerly and inside compiled code.

    Notes
    -----
    The public frequency derivative follows the composite route through
    ``jax.custom_jvp``. The rule applies the same principal-value
    operator to the analytic mode-supplied
    :math:`\partial_\omega\Sigma''`. It then adds the boundary terms
    and the exact tail derivatives. The rule transpose supplies reverse
    mode, so ``jax.jvp`` and ``jax.grad`` agree. Parameter tangents
    flow through the primal linearization. Grid mode differentiates its
    exact closed form directly. Its derivative contract holds only away
    from the hat knots, where the hat transform stays smooth. The
    ``TestEvaluateSelfEnergyDerivatives`` and
    ``TestGridModeHatTransform`` classes pin these contracts.
    """
    mode: str = model.mode
    minimum_n_kk: int = 8
    if n_kk < minimum_n_kk:
        msg: str = (
            "n_kk must reach eight nodes so the certified operator "
            "stencils stay defined"
        )
        raise ValueError(msg)
    if mode == "constant":
        real: Float64[Array, " n_omega"] = jnp.zeros_like(omega_rel_fermi_ev)
        imag: Float64[Array, " n_omega"] = jnp.broadcast_to(
            -jnp.logaddexp(model.coefficients[0], 0.0),
            omega_rel_fermi_ev.shape,
        )
    elif mode == "bosonic_kink":
        gamma_0: Float64[Array, ""] = jnp.logaddexp(model.coefficients[0], 0.0)
        coupling: Float64[Array, ""] = jnp.logaddexp(
            model.coefficients[1], 0.0
        )
        omega_0: Float64[Array, ""] = jnp.logaddexp(model.coefficients[2], 0.0)
        width: Float64[Array, ""] = jnp.logaddexp(model.coefficients[3], 0.0)
        baseline: Float64[Array, " one"] = _kink_real_part(
            model.subtraction_point_rel_fermi_ev[None],
            coupling,
            omega_0,
            width,
        )
        real = (
            _kink_real_part(omega_rel_fermi_ev, coupling, omega_0, width)
            - baseline[0]
        )
        lower: Float64[Array, " n_omega"] = omega_rel_fermi_ev - omega_0
        upper: Float64[Array, " n_omega"] = omega_rel_fermi_ev + omega_0
        imag = -gamma_0 - coupling**2 * width * (
            1.0 / (lower**2 + width**2) + 1.0 / (upper**2 + width**2)
        )
    else:
        domain: Optional[Float64[Array, " 2"]] = model.kk_domain_rel_fermi_ev
        tail_raw: Optional[Float64[Array, " 2"]] = model.tail_coefficients
        if domain is None or tail_raw is None:
            msg = (
                "numerical Kramers-Kronig modes require a declared "
                "domain and tail coordinates"
            )
            raise ValueError(msg)
        if mode == "grid":
            nodes: Optional[Float64[Array, " n_nodes"]] = (
                model.energy_nodes_rel_fermi_ev
            )
            if nodes is None:
                msg = "grid mode requires carrier energy nodes"
                raise ValueError(msg)
            real = _hat_real_subtracted(
                omega_rel_fermi_ev,
                nodes,
                model.coefficients,
                tail_raw,
                model.subtraction_point_rel_fermi_ev,
                domain,
                n_kk,
            )
            imag = jnp.interp(
                omega_rel_fermi_ev,
                nodes,
                -jnp.logaddexp(model.coefficients, 0.0),
            )
        else:
            real = _smooth_real_subtracted(
                mode,
                n_kk,
                omega_rel_fermi_ev,
                model.coefficients,
                tail_raw,
                model.subtraction_point_rel_fermi_ev,
                domain,
            )
            if mode == "fermi_liquid":
                imag = -jnp.logaddexp(
                    model.coefficients[0], 0.0
                ) + _dynamic_imag(mode, model.coefficients, omega_rel_fermi_ev)
            else:
                imag = _dynamic_imag(
                    mode, model.coefficients, omega_rel_fermi_ev
                )
    sigma: Complex128[Array, " n_omega"] = jax.lax.complex(real, imag)
    return sigma


__all__: list[str] = [
    "evaluate_self_energy",
]
