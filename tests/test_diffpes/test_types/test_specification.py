"""Validate the specification contracts.

The tests cover public carrier behavior, validation, and JAX
transformations for this implementation module.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest
from beartype.typing import Any
from jaxtyping import Array, Float64

import diffpes.types
from diffpes.types import (
    make_artifact_ref,
    make_convention_ref,
    make_domain_result,
    make_forward_model_spec,
)
from tests._assertions import assert_trees_close


class TestArtifactref:
    """Verify :class:`~diffpes.types.ArtifactRef`.

    The cases cover the public carrier or factory contract in JAX PyTrees.

    :see: :class:`~diffpes.types.ArtifactRef`
    """

    def test_public_symbol_has_expected_kind(self) -> None:
        """Expose ``ArtifactRef`` through its canonical types package path.

        The case uses explicit inputs in the supported certification regime.
        It checks the public result or the documented failure state.

        Notes
        -----
        The test compares the result with explicit numerical or structural
        assertions.
        """
        symbol: object = diffpes.types.ArtifactRef
        assert isinstance(symbol, type)
        assert issubclass(symbol, eqx.Module)


class TestCheckfunction:
    """Verify :obj:`~diffpes.types.CheckFunction`.

    The cases cover the public carrier or factory contract in JAX PyTrees.

    :see: :obj:`~diffpes.types.CheckFunction`
    """

    def test_public_symbol_has_expected_kind(self) -> None:
        """Expose ``CheckFunction`` through its canonical types package path.

        The case uses explicit inputs in the supported certification regime.
        It checks the public result or the documented failure state.

        Notes
        -----
        The test compares the result with explicit numerical or structural
        assertions.
        """
        symbol: object = diffpes.types.CheckFunction
        assert symbol is not None


class TestConventionref:
    """Verify :class:`~diffpes.types.ConventionRef`.

    The cases cover the public carrier or factory contract in JAX PyTrees.

    :see: :class:`~diffpes.types.ConventionRef`
    """

    def test_public_symbol_has_expected_kind(self) -> None:
        """Expose ``ConventionRef`` through its canonical types package path.

        The case uses explicit inputs in the supported certification regime.
        It checks the public result or the documented failure state.

        Notes
        -----
        The test compares the result with explicit numerical or structural
        assertions.
        """
        symbol: object = diffpes.types.ConventionRef
        assert isinstance(symbol, type)
        assert issubclass(symbol, eqx.Module)


class TestDomainpredicate:
    """Verify :class:`~diffpes.types.DomainPredicate`.

    The cases cover the public carrier or factory contract in JAX PyTrees.

    :see: :class:`~diffpes.types.DomainPredicate`
    """

    def test_public_symbol_has_expected_kind(self) -> None:
        """Expose ``DomainPredicate`` through its canonical types package path.

        The case uses explicit inputs in the supported certification regime.
        It checks the public result or the documented failure state.

        Notes
        -----
        The test compares the result with explicit numerical or structural
        assertions.
        """
        symbol: object = diffpes.types.DomainPredicate
        assert isinstance(symbol, type)
        assert issubclass(symbol, eqx.Module)


class TestDomainresult:
    """Verify :class:`~diffpes.types.DomainResult`.

    The cases cover the public carrier or factory contract in JAX PyTrees.

    :see: :class:`~diffpes.types.DomainResult`
    """

    def test_public_symbol_has_expected_kind(self) -> None:
        """Expose ``DomainResult`` through its canonical types package path.

        The case uses explicit inputs in the supported certification regime.
        It checks the public result or the documented failure state.

        Notes
        -----
        The test compares the result with explicit numerical or structural
        assertions.
        """
        symbol: object = diffpes.types.DomainResult
        assert isinstance(symbol, type)
        assert issubclass(symbol, eqx.Module)


class TestForwardmodelspec:
    """Verify :class:`~diffpes.types.ForwardModelSpec`.

    The cases cover the public carrier or factory contract in JAX PyTrees.

    :see: :class:`~diffpes.types.ForwardModelSpec`
    """

    def test_public_symbol_has_expected_kind(self) -> None:
        """Expose ``ForwardModelSpec`` through its canonical types package
        path.

        The case uses explicit inputs in the supported certification regime.
        It checks the public result or the documented failure state.

        Notes
        -----
        The test compares the result with explicit numerical or structural
        assertions.
        """
        symbol: object = diffpes.types.ForwardModelSpec
        assert isinstance(symbol, type)
        assert issubclass(symbol, eqx.Module)


class TestMakeArtifactRef:
    """Verify :func:`~diffpes.types.make_artifact_ref`.

    The cases cover the public carrier or factory contract in JAX PyTrees.

    :see: :func:`~diffpes.types.make_artifact_ref`
    """

    def test_public_symbol_has_expected_kind(self) -> None:
        """Expose ``make_artifact_ref`` through its canonical types package
        path.

        The case uses explicit inputs in the supported certification regime.
        It checks the public result or the documented failure state.

        Notes
        -----
        The test compares the result with explicit numerical or structural
        assertions.
        """
        symbol: object = diffpes.types.make_artifact_ref
        assert callable(symbol)


class TestMakeConventionRef:
    """Verify :func:`~diffpes.types.make_convention_ref`.

    The cases cover the public carrier or factory contract in JAX PyTrees.

    :see: :func:`~diffpes.types.make_convention_ref`
    """

    def test_public_symbol_has_expected_kind(self) -> None:
        """Expose ``make_convention_ref`` through its canonical types package
        path.

        The case uses explicit inputs in the supported certification regime.
        It checks the public result or the documented failure state.

        Notes
        -----
        The test compares the result with explicit numerical or structural
        assertions.
        """
        symbol: object = diffpes.types.make_convention_ref
        assert callable(symbol)


class TestMakeDomainPredicate:
    """Verify :func:`~diffpes.types.make_domain_predicate`.

    The cases cover the public carrier or factory contract in JAX PyTrees.

    :see: :func:`~diffpes.types.make_domain_predicate`
    """

    def test_public_symbol_has_expected_kind(self) -> None:
        """Expose ``make_domain_predicate`` through its canonical types
        package path.

        The case uses explicit inputs in the supported certification regime.
        It checks the public result or the documented failure state.

        Notes
        -----
        The test compares the result with explicit numerical or structural
        assertions.
        """
        symbol: object = diffpes.types.make_domain_predicate
        assert callable(symbol)


class TestMakeDomainResult:
    """Verify :func:`~diffpes.types.make_domain_result`.

    The cases cover the public carrier or factory contract in JAX PyTrees.

    :see: :func:`~diffpes.types.make_domain_result`
    """

    def test_public_symbol_has_expected_kind(self) -> None:
        """Expose ``make_domain_result`` through its canonical types package
        path.

        The case uses explicit inputs in the supported certification regime.
        It checks the public result or the documented failure state.

        Notes
        -----
        The test compares the result with explicit numerical or structural
        assertions.
        """
        symbol: object = diffpes.types.make_domain_result
        assert callable(symbol)

    def test_domain_results_vmap_with_traced_status_and_margin(self) -> None:
        """Batch domain checks without concretizing status arrays.

        The case uses explicit inputs in the supported certification regime.
        It checks the public result or the documented failure state.

        Notes
        -----
        The test compares the result with explicit numerical or structural
        assertions.
        """
        result: Any

        def evaluate(value: Float64[Array, ""]) -> Any:
            margin: Any
            margin = 1.0 - jnp.abs(value)
            result: Any = make_domain_result(
                "bounded",
                measured=value,
                reference=0.0,
                residual=value,
                tolerance=0.0,
                margin=margin,
                passed=margin >= 0.0,
                in_domain=margin >= 0.0,
            )
            return result

        result = eqx.filter_jit(jax.vmap(evaluate))(jnp.array([-0.5, 1.5]))
        assert_trees_close(result.margin, jnp.array([0.5, -0.5]))
        assert jnp.array_equal(result.passed, jnp.array([True, False]))


class TestMakeForwardModelSpec:
    """Verify :func:`~diffpes.types.make_forward_model_spec`.

    The cases cover the public carrier or factory contract in JAX PyTrees.

    :see: :func:`~diffpes.types.make_forward_model_spec`
    """

    def test_public_symbol_has_expected_kind(self) -> None:
        """Expose ``make_forward_model_spec`` through its canonical types
        package path.

        The case uses explicit inputs in the supported certification regime.
        It checks the public result or the documented failure state.

        Notes
        -----
        The test compares the result with explicit numerical or structural
        assertions.
        """
        symbol: object = diffpes.types.make_forward_model_spec
        assert callable(symbol)

    def test_factories_reject_malformed_static_structure(self) -> None:
        """Reject empty identities, invalid JSON, and overlapping path classes.

        The case uses explicit inputs in the supported certification regime.
        It checks the public result or the documented failure state.

        Notes
        -----
        The test compares the result with explicit numerical or structural
        assertions.
        """
        with pytest.raises(ValueError, match="artifact_id must be non-empty"):
            make_artifact_ref("", "text/plain", None, "a", "b", None, "input")
        with pytest.raises(ValueError, match="valid JSON"):
            make_convention_ref("convention", "1", "{")
        with pytest.raises(ValueError, match="must be disjoint"):
            make_forward_model_spec(
                "model",
                "1",
                "observable",
                "implementation",
                (),
                (),
                (),
                ("x",),
                ("x",),
            )


class TestArtifactResolver:
    """Verify :obj:`~diffpes.types.ArtifactResolver`.

    The case checks the canonical types package path for the resolver alias.

    :see: :obj:`~diffpes.types.ArtifactResolver`
    """

    def test_public_alias_is_available(self) -> None:
        """Expose the artifact resolver alias from the types package.

        The public package must provide one canonical alias object.

        Notes
        -----
        The test resolves the alias by its exact public name.
        """
        symbol: object = diffpes.types.ArtifactResolver
        assert symbol is not None
