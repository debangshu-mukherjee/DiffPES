"""PRIVATE: Apply the certified retarded Kramers--Kronig map.

Extended Summary
----------------
This private module owns the retarded principal-value transform.
It also owns the custom derivative.
"""

from functools import partial

import jax
import jax.numpy as jnp
from beartype.typing import Any, List, Tuple
from jax.custom_derivatives import SymbolicZero
from jaxtyping import Array, Float64

from ._principal_value import (
    _check_frozen_core_grid,
    _check_tail_spec,
    _check_trusted_interval,
    _cubic_core_pv,
    _derivative_samples_sixth_order,
    _power2_tail_pv,
)


def _tangent_is_symbolic_zero(tangent: Any) -> bool:
    """PRIVATE: Report whether one tangent tree carries no perturbation.

    The custom derivative rules use this static predicate to skip the
    linearization of unperturbed argument groups.

    Parameters
    ----------
    tangent : Any
        Tangent tree from a ``symbolic_zeros`` custom derivative rule.

    Returns
    -------
    all_zero : bool
        Whether every tangent leaf is a symbolic zero.

    Notes
    -----
    The check flattens the tree with symbolic zeros as leaves.
    It then tests every leaf for the symbolic zero type.
    """
    leaves: List[Any] = jax.tree_util.tree_leaves(
        tangent, is_leaf=lambda value: isinstance(value, SymbolicZero)
    )
    all_zero: bool = all(isinstance(value, SymbolicZero) for value in leaves)
    return all_zero


def _materialize_tangent(tangent: Any) -> Any:
    """PRIVATE: Replace symbolic zero tangent leaves with explicit zero arrays.

    Parameters
    ----------
    tangent : Any
        Tangent tree from a ``symbolic_zeros`` custom derivative rule.

    Returns
    -------
    materialized : Any
        Tangent tree with every leaf as an ordinary array.

    Notes
    -----
    The map visits every leaf with symbolic zeros as leaves.
    It builds a matching zero array for each symbolic leaf.
    """

    def _leaf(value: Any) -> Any:
        """PRIVATE: Convert one symbolic zero leaf to an explicit zero array.

        Parameters
        ----------
        value : Any
            One tangent tree leaf.

        Returns
        -------
        value : Any
            Unchanged leaf or a matching explicit zero array.

        Notes
        -----
        The symbolic leaf exposes its shape and dtype directly.
        """
        if isinstance(value, SymbolicZero):
            zero_leaf: Float64[Array, "..."] = jnp.zeros(
                value.shape, value.dtype
            )
            return zero_leaf
        return value

    materialized: Any = jax.tree_util.tree_map(
        _leaf, tangent, is_leaf=lambda value: isinstance(value, SymbolicZero)
    )
    return materialized


def _kk_transform_impl(  # noqa: DOC502, DOC503 -- JAX runtime guards.
    core_grid: Tuple[Float64[Array, " n_kk"], Float64[Array, " n_kk"]],
    model_domain: Float64[Array, " 2"],
    tail_spec: Any,
    queries: Float64[Array, " n_query"],
    n_tail: int,
) -> Float64[Array, " n_query"]:
    """PRIVATE: Evaluate the validated transform without a rule.

    The routine validates the grid, the trusted interval, and the tail
    contract. It then adds the piecewise-cubic core principal value and
    both semi-infinite tail quadratures at every query.

    Parameters
    ----------
    core_grid : Tuple[Float64[Array, " n_kk"], Float64[Array, " n_kk"]]
        Pair of frozen node positions and sampled imaginary values.
    model_domain : Float64[Array, " 2"]
        Increasing carrier domain ``[a, b]`` in eV.
    tail_spec : Any
        Six-attribute left-then-right ``power2`` tail contract.
    queries : Float64[Array, " n_query"]
        Query energies inside the trusted interval in eV.
    n_tail : int
        Positive number of tail quadrature nodes per side.

    Returns
    -------
    transformed : Float64[Array, " n_query"]
        Unsubtracted principal-value transform at every query.

    Raises
    ------
    ValueError
        If ``n_tail`` truncates the tails or the grid pair shapes
        disagree.
    EquinoxRuntimeError
        If a traced grid, query, or tail predicate fails.

    Notes
    -----
    The custom-rule wrapper shares this body, so the primal and the
    derivative rule evaluate identical validated values.
    """
    positions: Float64[Array, " n_kk"]
    values: Float64[Array, " n_kk"]
    positions, values = core_grid
    if n_tail <= 0:
        msg: str = (
            "n_tail must stay positive; a zero order truncates the "
            "semi-infinite tail quadrature"
        )
        raise ValueError(msg)
    if positions.ndim != 1 or positions.shape != values.shape:
        msg = (
            "core_grid must pair one grid vector with one matching "
            "value vector"
        )
        raise ValueError(msg)
    checked_positions: Float64[Array, " n_kk"] = _check_frozen_core_grid(
        positions, model_domain
    )
    spacing: Float64[Array, ""] = (model_domain[1] - model_domain[0]) / (
        positions.shape[0] - 1
    )
    checked_values: Float64[Array, " n_kk"] = _check_tail_spec(
        values, spacing, tail_spec
    )
    checked_queries: Float64[Array, " n_query"] = _check_trusted_interval(
        queries, model_domain, spacing
    )
    core: Float64[Array, " n_query"] = _cubic_core_pv(
        checked_positions, checked_values, checked_queries
    )
    tails: Float64[Array, " n_query"] = _power2_tail_pv(
        model_domain, tail_spec, checked_queries, n_tail
    )
    transformed: Float64[Array, " n_query"] = core + tails
    return transformed


def _seam_query_composite(
    core_grid: Tuple[Float64[Array, " n_kk"], Float64[Array, " n_kk"]],
    model_domain: Float64[Array, " 2"],
    tail_spec: Any,
    queries: Float64[Array, " n_query"],
    n_tail: int,
) -> Float64[Array, " n_query"]:
    """PRIVATE: Evaluate the composite seam derivative from grid samples alone.

    The mode-agnostic seam has no analytic derivative source. The rule
    therefore differentiates the samples with sixth-order stencils. It
    applies the same core operator to those samples, adds the boundary
    terms, and adds the exact tail derivatives.

    Parameters
    ----------
    core_grid : Tuple[Float64[Array, " n_kk"], Float64[Array, " n_kk"]]
        Pair of frozen node positions and sampled imaginary values.
    model_domain : Float64[Array, " 2"]
        Increasing carrier domain ``[a, b]`` in eV.
    tail_spec : Any
        Six-attribute left-then-right ``power2`` tail contract.
    queries : Float64[Array, " n_query"]
        Query energies inside the trusted interval in eV.
    n_tail : int
        Positive number of tail quadrature nodes per side.

    Returns
    -------
    composite : Float64[Array, " n_query"]
        Composite query derivative of the unsubtracted transform.

    Notes
    -----
    The derivative of the transform equals the transform of the
    derivative plus boundary terms, by partial integration on the
    finite core.
    """
    positions: Float64[Array, " n_kk"]
    values: Float64[Array, " n_kk"]
    positions, values = core_grid
    spacing: Float64[Array, ""] = (model_domain[1] - model_domain[0]) / (
        positions.shape[0] - 1
    )
    derivative_samples: Float64[Array, " n_kk"] = (
        _derivative_samples_sixth_order(values, spacing)
    )
    core_derivative: Float64[Array, " n_query"] = _cubic_core_pv(
        positions, derivative_samples, queries
    )
    boundary: Float64[Array, " n_query"] = (
        values[0] / (model_domain[0] - queries)
        - values[-1] / (model_domain[1] - queries)
    ) / jnp.pi

    def _tail_only(
        points: Float64[Array, " n_query"],
    ) -> Float64[Array, " n_query"]:
        """PRIVATE: Evaluate both tail quadratures for the derivative closure.

        Parameters
        ----------
        points : Float64[Array, " n_query"]
            Query energies in eV.

        Returns
        -------
        contribution : Float64[Array, " n_query"]
            Unsubtracted tail contribution at every query.

        Notes
        -----
        Forward-mode differentiation of this closure supplies the
        exact tail derivative.
        """
        contribution: Float64[Array, " n_query"] = _power2_tail_pv(
            model_domain, tail_spec, points, n_tail
        )
        return contribution

    tail_derivative: Float64[Array, " n_query"]
    _, tail_derivative = jax.jvp(
        _tail_only, (queries,), (jnp.ones_like(queries),)
    )
    composite: Float64[Array, " n_query"] = (
        core_derivative + boundary + tail_derivative
    )
    return composite


@partial(jax.custom_jvp, nondiff_argnums=(4,))
def _kk_transform(  # noqa: DOC502 -- eqx.error_if raises under JAX execution.
    core_grid: Tuple[Float64[Array, " n_kk"], Float64[Array, " n_kk"]],
    model_domain: Float64[Array, " 2"],
    tail_spec: Any,
    queries: Float64[Array, " n_query"],
    n_tail: int,
) -> Float64[Array, " n_query"]:
    """PRIVATE: Evaluate the cell-integrated principal-value transform seam.

    The seam performs direct per-query evaluation with no kernel matrix
    and no post-transform interpolation. Its custom derivative rule
    routes query tangents through the composite derivative and keeps
    every other tangent on the primal linearization.

    :see: :class:`~.test_spectral.TestKkTransformSeam`
    :see: :class:`~.test_spectral.TestPlantedNoncompliantConstructions`

    Parameters
    ----------
    core_grid : Tuple[Float64[Array, " n_kk"], Float64[Array, " n_kk"]]
        Pair of frozen node positions and sampled imaginary values.
    model_domain : Float64[Array, " 2"]
        Increasing carrier domain ``[a, b]`` in eV.
    tail_spec : Any
        Six-attribute left-then-right ``power2`` tail contract.
    queries : Float64[Array, " n_query"]
        Query energies inside the trusted interval in eV.
    n_tail : int
        Positive number of tail quadrature nodes per side.

    Returns
    -------
    transformed : Float64[Array, " n_query"]
        Unsubtracted principal-value transform at every query.

    Raises
    ------
    ValueError
        If ``n_tail`` truncates the tails or the grid pair shapes
        disagree.
    EquinoxRuntimeError
        If a traced grid, query, or tail predicate fails.

    Notes
    -----
    The wrapper defers to the shared implementation body. Only the
    attached derivative rule distinguishes it from the plain call.
    """
    transformed: Float64[Array, " n_query"] = _kk_transform_impl(
        core_grid, model_domain, tail_spec, queries, n_tail
    )
    return transformed


@partial(_kk_transform.defjvp, symbolic_zeros=True)
def _kk_transform_jvp(
    n_tail: int,
    primals: Any,
    tangents: Any,
) -> Tuple[Float64[Array, " n_query"], Float64[Array, " n_query"]]:
    """PRIVATE: Dispatch seam tangents through the composite contract.

    Query tangents multiply the composite seam derivative. Grid, domain,
    and tail tangents pass through the linearized primal, which stays
    exact because the transform is linear in the samples. Symbolic-zero
    detection skips every unperturbed argument group.

    Parameters
    ----------
    n_tail : int
        Positive number of tail quadrature nodes per side.
    primals : Any
        Seam primal inputs ``(core_grid, model_domain, tail_spec,
        queries)``.
    tangents : Any
        Matching tangent structure for the seam primal inputs.

    Returns
    -------
    pair : Tuple[Float64[Array, " n_query"], Float64[Array, " n_query"]]
        Primal seam output and its tangent.

    Notes
    -----
    The rule stays linear in every tangent, so the transform transpose
    supplies reverse mode without a separate rule.
    """
    core_grid: Tuple[Float64[Array, " n_kk"], Float64[Array, " n_kk"]]
    model_domain: Float64[Array, " 2"]
    tail_spec: Any
    queries: Float64[Array, " n_query"]
    core_grid, model_domain, tail_spec, queries = primals
    core_tangent: Any
    domain_tangent: Any
    spec_tangent: Any
    query_tangent: Any
    core_tangent, domain_tangent, spec_tangent, query_tangent = tangents
    primal_out: Float64[Array, " n_query"] = _kk_transform_impl(
        core_grid, model_domain, tail_spec, queries, n_tail
    )
    tangent_out: Float64[Array, " n_query"] = jnp.zeros_like(primal_out)

    def _fixed_queries(
        grid_pair: Tuple[Float64[Array, " n_kk"], Float64[Array, " n_kk"]],
        domain: Float64[Array, " 2"],
        spec: Any,
    ) -> Float64[Array, " n_query"]:
        """PRIVATE: Evaluate the transform with the query set held fixed.

        Parameters
        ----------
        grid_pair : Tuple[Float64[Array, " n_kk"], Float64[Array, " n_kk"]]
            Pair of node positions and sampled imaginary values.
        domain : Float64[Array, " 2"]
            Increasing carrier domain ``[a, b]`` in eV.
        spec : Any
            Six-attribute ``power2`` tail contract.

        Returns
        -------
        value : Float64[Array, " n_query"]
            Unsubtracted transform at the closed-over queries.

        Notes
        -----
        Linearizing this closure yields the exact sample, domain, and
        tail tangents.
        """
        value: Float64[Array, " n_query"] = _kk_transform_impl(
            grid_pair, domain, spec, queries, n_tail
        )
        return value

    linear_perturbed: bool = not (
        _tangent_is_symbolic_zero(core_tangent)
        and _tangent_is_symbolic_zero(domain_tangent)
        and _tangent_is_symbolic_zero(spec_tangent)
    )
    if linear_perturbed:
        linear_tangent: Float64[Array, " n_query"]
        _, linear_tangent = jax.jvp(
            _fixed_queries,
            (core_grid, model_domain, tail_spec),
            (
                _materialize_tangent(core_tangent),
                _materialize_tangent(domain_tangent),
                _materialize_tangent(spec_tangent),
            ),
        )
        tangent_out = tangent_out + linear_tangent
    if not _tangent_is_symbolic_zero(query_tangent):
        composite: Float64[Array, " n_query"] = _seam_query_composite(
            core_grid, model_domain, tail_spec, queries, n_tail
        )
        tangent_out = tangent_out + composite * query_tangent
    pair: Tuple[Float64[Array, " n_query"], Float64[Array, " n_query"]] = (
        primal_out,
        tangent_out,
    )
    return pair


__all__: list[str] = []
