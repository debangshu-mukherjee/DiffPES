"""Validate the stable pure-JAX certification-check registry.

The tests exercise explicit registration, exact lookup, and deterministic
listing without treating registry bookkeeping as a physics result.
"""

import uuid

import jax.numpy as jnp
import pytest
from beartype import beartype
from beartype.typing import Tuple
from jaxtyping import Array, jaxtyped

from diffpes.certify import get_check, list_checks, register_check
from diffpes.types import CheckFunction, DomainResult, make_domain_result


@jaxtyped(typechecker=beartype)
def _positive_check(value: Array) -> DomainResult:
    """PRIVATE: Return a traced positive-value domain result.

    Parameters
    ----------
    value : Array
        JAX value whose sign the check measures.

    Returns
    -------
    result : DomainResult
        Domain result that passes and is in domain when the value is
        positive.

    Notes
    -----
    Casts the value to float64 and reuses it as the measured, residual,
    and margin fields against a zero reference and a zero tolerance.
    """
    margin: Array = jnp.asarray(value, dtype=jnp.float64)
    result: DomainResult = make_domain_result(
        predicate_id="org.diffpes.domain.test.positive",
        measured=margin,
        reference=jnp.zeros(()),
        residual=margin,
        tolerance=jnp.zeros(()),
        margin=margin,
        passed=margin > 0.0,
        in_domain=margin > 0.0,
    )
    return result


class TestRegisterCheck:
    """Verify :func:`~diffpes.certify.register_check`.

    The cases cover the public behavior in the supported certification regime.

    :see: :func:`~diffpes.certify.register_check`
    """

    def test_registered_check_is_resolved_exactly(self) -> None:
        """Verify lookup returns the registered pure JAX predicate.

        The case uses explicit inputs in the supported certification regime.
        It checks the public result or the documented failure state.

        Notes
        -----
        Registers one unique identity and evaluates the resolved callable on a
        scalar JAX input.
        """
        check_id: str = f"org.diffpes.domain.test.{uuid.uuid4().hex}"

        register_check(check_id, _positive_check)
        resolved: CheckFunction = get_check(check_id)
        assert bool(resolved(jnp.asarray(1.0)).passed)
        assert check_id in list_checks()


class TestGetCheck:
    """Verify :func:`~diffpes.certify.get_check`.

    The cases cover exact lookup and explicit failure for an unknown identity.

    :see: :func:`~diffpes.certify.get_check`
    """

    def test_unknown_check_is_rejected(self) -> None:
        """Reject an unregistered stable check identity.

        The case uses explicit inputs in the supported certification regime.
        It checks the public result or the documented failure state.

        Notes
        -----
        The test compares the result with explicit numerical or structural assertions.
        """
        with pytest.raises(KeyError, match="unknown certification check"):
            get_check("org.diffpes.domain.test.absent")


class TestListChecks:
    """Verify :func:`~diffpes.certify.list_checks`.

    The cases cover the deterministic order of process-local check identities.

    :see: :func:`~diffpes.certify.list_checks`
    """

    def test_registered_checks_are_sorted(self) -> None:
        """List registered identities in deterministic lexical order.

        The case uses explicit inputs in the supported certification regime.
        It checks the public result or the documented failure state.

        Notes
        -----
        The test compares the result with explicit numerical or structural assertions.
        """
        check_id: str = f"org.diffpes.domain.test.{uuid.uuid4().hex}"
        register_check(check_id, _positive_check)
        checks: Tuple[str, ...] = list_checks()
        assert checks == tuple(sorted(checks))
        assert check_id in checks
