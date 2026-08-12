"""Validate bounded policy-waiver records at the I/O boundary.

Extended Summary
----------------
The module validates absolute UTC intervals before compiled certification.
A waiver records a policy exception but never changes a claim outcome.

Routine Listings
----------------
:func:`require_active_waivers`
    Reject malformed, expired, or premature waiver records.
:func:`validate_waiver`
    Validate one waiver against an explicit absolute UTC time.
:func:`validate_waivers`
    Validate multiple waivers against one explicit absolute UTC time.
"""

from datetime import UTC, datetime

from beartype import beartype
from beartype.typing import List, Tuple
from jaxtyping import jaxtyped

from diffpes.constants import (
    CERTIFICATION_POLICY_IDS,
)
from diffpes.types import (
    WaiverRecord,
    WaiverReport,
    make_waiver_report,
)


def _parse_utc(value: str, name: str) -> datetime:
    """PRIVATE: Parse one absolute UTC timestamp with a terminal Z
    marker.

    Parameters
    ----------
    value : str
        Timestamp text, for example ``"2026-01-01T00:00:00Z"``.
    name : str
        Field label used in error messages.

    Returns
    -------
    parsed : datetime
        Timezone-aware UTC datetime.

    Raises
    ------
    ValueError
        If the text does not end in ``Z``, does not parse as an ISO
        timestamp, or does not resolve to UTC.

    Notes
    -----
    Replaces the terminal ``Z`` with an explicit ``+00:00`` offset for
    :func:`datetime.fromisoformat`, then verifies the parsed zone is
    UTC. Waiver interval comparisons therefore never mix naive and
    aware datetimes.
    """
    error: ValueError
    if not value.endswith("Z"):
        message: str = f"{name} must be an absolute UTC timestamp ending in Z"
        raise ValueError(message)
    try:
        parsed: datetime = datetime.fromisoformat(f"{value[:-1]}+00:00")
    except ValueError as error:
        message = f"{name} must be a valid absolute UTC timestamp"
        raise ValueError(message) from error
    if parsed.tzinfo != UTC:
        message = f"{name} must use UTC"
        raise ValueError(message)
    return parsed


@jaxtyped(typechecker=beartype)
def validate_waiver(
    waiver: WaiverRecord,
    *,
    as_of_utc: str,
) -> WaiverReport:
    """Validate one waiver against an explicit absolute UTC time.

    The report distinguishes malformed records from records outside their
    active interval.

    :see: :class:`~.test_waivers.TestValidateWaiver`

    Parameters
    ----------
    waiver : WaiverRecord
        Static waiver declaration.
    as_of_utc : str
        Absolute UTC evaluation time ending in ``Z``.

    Returns
    -------
    report : WaiverReport
        Structural validity, temporal activity, and validation errors.

    Notes
    -----
    The function performs static I/O-boundary validation. It does not run in a
    compiled certification kernel.
    """
    errors: List[str] = []
    issued: datetime | None = None
    expires: datetime | None = None
    selected: datetime | None = None
    value: str
    name: str
    error: ValueError
    for value, name in (
        (waiver.issued_at_utc, "issued_at_utc"),
        (waiver.expires_at_utc, "expires_at_utc"),
        (as_of_utc, "as_of_utc"),
    ):
        try:
            parsed: datetime = _parse_utc(value, name)
        except ValueError as error:
            errors.append(str(error))
            continue
        if name == "issued_at_utc":
            issued = parsed
        elif name == "expires_at_utc":
            expires = parsed
        else:
            selected = parsed
    if waiver.policy_id not in CERTIFICATION_POLICY_IDS:
        errors.append(f"unknown certification policy: {waiver.policy_id}")
    if issued is not None and expires is not None and issued >= expires:
        errors.append("issued_at_utc must precede expires_at_utc")
    valid: bool = not errors
    active: bool = bool(
        valid
        and issued is not None
        and expires is not None
        and selected is not None
        and issued <= selected < expires
    )
    if valid and not active:
        errors.append("waiver is not active at as_of_utc")
    report: WaiverReport = make_waiver_report(
        waiver_id=waiver.waiver_id,
        valid=valid,
        active=active,
        errors=tuple(errors),
    )
    return report


@jaxtyped(typechecker=beartype)
def validate_waivers(
    waivers: Tuple[WaiverRecord, ...],
    *,
    as_of_utc: str,
) -> Tuple[WaiverReport, ...]:
    """Validate multiple waivers against one explicit absolute UTC time.

    The function rejects duplicate identities. It then validates each interval.

    :see: :class:`~.test_waivers.TestValidateWaivers`

    Implementation Logic
    --------------------
    1. **Validate each waiver**::

           reports = tuple(
               validate_waiver(waiver, as_of_utc=as_of_utc)
               for waiver in waivers
           )

       Input ordering remains stable in the returned report tuple.

    Parameters
    ----------
    waivers : Tuple[WaiverRecord, ...]
        Static waiver declarations.
    as_of_utc : str
        Absolute UTC evaluation time ending in ``Z``.

    Returns
    -------
    reports : Tuple[WaiverReport, ...]
        Validation reports in input order.

    Raises
    ------
    ValueError
        If waiver identities contain duplicates.
    """
    waiver_ids: Tuple[str, ...] = tuple(waiver.waiver_id for waiver in waivers)
    if len(waiver_ids) != len(set(waiver_ids)):
        raise ValueError("waiver identities must be unique")
    reports: Tuple[WaiverReport, ...] = tuple(
        validate_waiver(waiver, as_of_utc=as_of_utc) for waiver in waivers
    )
    return reports


@jaxtyped(typechecker=beartype)
def require_active_waivers(
    waivers: Tuple[WaiverRecord, ...],
    *,
    as_of_utc: str,
) -> None:
    """Reject malformed, expired, or premature waiver records.

    The eager boundary rejects every record without active temporal scope.

    :see: :class:`~.test_waivers.TestRequireActiveWaivers`

    Parameters
    ----------
    waivers : Tuple[WaiverRecord, ...]
        Static waiver declarations.
    as_of_utc : str
        Absolute UTC evaluation time ending in ``Z``.

    Raises
    ------
    ValueError
        If one waiver lacks valid and active temporal scope.

    Notes
    -----
    Successful validation records the waiver only. It does not change any
    failed, unchecked, or out-of-domain claim.
    """
    reports: Tuple[WaiverReport, ...] = validate_waivers(
        waivers,
        as_of_utc=as_of_utc,
    )
    failures: Tuple[str, ...] = tuple(
        f"{report.waiver_id}: {', '.join(report.errors)}"
        for report in reports
        if not bool(report.valid) or not bool(report.active)
    )
    if failures:
        raise ValueError("invalid waiver records: " + "; ".join(failures))


__all__: list[str] = [
    "require_active_waivers",
    "validate_waiver",
    "validate_waivers",
]
