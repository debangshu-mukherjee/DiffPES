"""Validate the shared invalid-input assertion helper.

The tests exercise successful eager and compiled rejection checks.
They also verify the failure for an accepted invalid value.
"""

import equinox as eqx
import jax.numpy as jnp
import pytest
from beartype import beartype
from jaxtyping import Array, Float, jaxtyped

from tests._assertions import assert_rejects


class TestAssertRejects:
    """Validate :func:`~tests._assertions.assert_rejects`.

    Covers matched validation errors in eager and JIT execution and rejects a
    callable that accepts the invalid value.

    :see: :func:`~tests._assertions.assert_rejects`
    """

    def test_assert_rejects_eager_and_jit(self) -> None:
        """Accept a value-threaded eager and compiled rejection.

        The test confirms the helper recognizes the same Equinox runtime error
        with and without JIT compilation for the scalar value ``-1``.

        Notes
        -----
        Threads the scalar through ``eqx.error_if`` and matches the stable
        validation message in both execution modes.
        """

        @jaxtyped(typechecker=beartype)
        def reject_negative(
            value: Float[Array, ""],
        ) -> Float[Array, ""]:
            checked: Float[Array, ""] = eqx.error_if(
                value,
                value < 0,
                "value must be non-negative",
            )
            return checked

        assert_rejects(
            reject_negative,
            jnp.asarray(-1.0),
            match="value must be non-negative",
        )

    def test_assert_rejects_fails_for_passing_callable(self) -> None:
        """Fail when a callable accepts invalid test data.

        The test confirms the helper raises ``AssertionError`` instead of
        accepting a callable with no validation behavior.

        Notes
        -----
        The test passes scalar ``-1`` through an identity function. It disables
        the redundant JIT repeat and matches the missing-exception diagnostic.
        """

        @jaxtyped(typechecker=beartype)
        def accept(value: Float[Array, ""]) -> Float[Array, ""]:
            accepted: Float[Array, ""] = value
            return accepted

        with pytest.raises(AssertionError, match="DID NOT RAISE"):
            assert_rejects(
                accept,
                jnp.asarray(-1.0),
                match="unused",
                under_jit=False,
            )
