"""Validate bounded policy waivers at explicit UTC times.

The tests cover active, expired, malformed, and duplicate waiver records.
They confirm that a waiver never changes a failed claim to passed.
"""

import jax.numpy as jnp
import pytest
from beartype.typing import Any

from diffpes.certify import (
    evaluate_policy,
    require_active_waivers,
    validate_waiver,
    validate_waivers,
)
from diffpes.types import make_certification_claim, make_waiver_record


def _waiver(expires_at_utc: str = "2026-07-22T00:00:00Z") -> Any:
    """PRIVATE: Build one research-policy waiver with explicit UTC limits.

    Parameters
    ----------
    expires_at_utc : str
        Exclusive expiry time of the waiver in UTC.

    Returns
    -------
    waiver : Any
        Waiver record for the finite-output claim, issued at
        2026-07-20T00:00:00Z under the research policy.

    Notes
    -----
    Fixes the waiver identity, the policy identity, the author, and the
    reason; only the expiry time varies.
    """
    waiver: Any = make_waiver_record(
        waiver_id="waiver-test",
        policy_id="org.diffpes.policy.research.v1",
        claim_ids=("claim.output.finite",),
        author="reviewer",
        reason="The independent reference is pending.",
        issued_at_utc="2026-07-20T00:00:00Z",
        expires_at_utc=expires_at_utc,
    )
    return waiver


class TestValidateWaiver:
    """Verify :func:`~diffpes.certify.validate_waiver`.

    The cases distinguish valid structure from an active UTC interval.

    :see: :func:`~diffpes.certify.validate_waiver`
    """

    def test_active_interval_is_valid(self) -> None:
        """Accept an evaluation time inside the half-open waiver interval.

        The report must mark the waiver as valid and active.

        Notes
        -----
        The selected time is one day after issue and one day before expiry.
        """
        report: Any = validate_waiver(
            _waiver(),
            as_of_utc="2026-07-21T00:00:00Z",
        )
        assert bool(report.valid)
        assert bool(report.active)

    def test_expired_interval_is_inactive(self) -> None:
        """Reject an evaluation time at the exclusive expiry boundary.

        The report must keep valid structure and mark inactive scope.

        Notes
        -----
        The selected time equals the absolute UTC expiry time.
        """
        report: Any = validate_waiver(
            _waiver(),
            as_of_utc="2026-07-22T00:00:00Z",
        )
        assert bool(report.valid)
        assert not bool(report.active)


class TestValidateWaivers:
    """Verify :func:`~diffpes.certify.validate_waivers`.

    The case rejects duplicate waiver identities before policy evaluation.

    :see: :func:`~diffpes.certify.validate_waivers`
    """

    def test_duplicate_waiver_id_is_rejected(self) -> None:
        """Reject two waiver records with the same permanent identity.

        Duplicate records must raise a static validation error.

        Notes
        -----
        The test supplies the same immutable waiver object twice.
        """
        waiver: Any = _waiver()
        with pytest.raises(ValueError, match="unique"):
            validate_waivers(
                (waiver, waiver),
                as_of_utc="2026-07-21T00:00:00Z",
            )


class TestRequireActiveWaivers:
    """Verify :func:`~diffpes.certify.require_active_waivers`.

    The cases enforce active UTC scope and preserve failed claim outcomes.

    :see: :func:`~diffpes.certify.require_active_waivers`
    """

    def test_expired_waiver_is_rejected(self) -> None:
        """Raise an error when a waiver has expired before execution.

        The eager boundary must not accept inactive review scope.

        Notes
        -----
        The test selects a time one day after the recorded expiry.
        """
        with pytest.raises(ValueError, match="not active"):
            require_active_waivers(
                (_waiver(),),
                as_of_utc="2026-07-23T00:00:00Z",
            )

    def test_waiver_does_not_pass_failed_claim(self) -> None:
        """Keep a failed claim false when a policy contains a waiver.

        The waiver must not change the numerical outcome of the claim.

        Notes
        -----
        The test evaluates one failed required claim under research policy.
        """
        claim: Any = make_certification_claim(
            claim_id="claim.output.finite",
            subject_id="org.diffpes.observable.test",
            predicate_id="output.finite",
            evidence_ids=(),
            measured=jnp.array([1.0]),
            reference=jnp.array([0.0]),
            residual=jnp.array([1.0]),
            tolerance=jnp.array([0.0]),
            passed=False,
            checked=True,
            in_domain=True,
            margin=-1.0,
            severity_code=1,
        )
        report: Any = evaluate_policy(
            (claim,),
            "org.diffpes.policy.research.v1",
            waivers=(_waiver(),),
        )
        assert not bool(jnp.all(report.achieved))

    def test_waiver_policy_must_match_selected_policy(self) -> None:
        """Reject a waiver that belongs to a different policy.

        Policy evaluation must not apply a research waiver to publication.

        Notes
        -----
        The helper creates a research-policy waiver for this mismatch case.
        """
        with pytest.raises(ValueError, match="does not match selected policy"):
            evaluate_policy(
                (),
                "org.diffpes.policy.publication.v1",
                waivers=(_waiver(),),
            )
