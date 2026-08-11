"""Store types-owned records from certificate inspection.

Extended Summary
----------------
Inspection records categorize scientific, numerical, environment, and audit
differences without evaluating any new physical claim.

Routine Listings
----------------
:class:`CertificateDiff`
    Store categorized differences between two forward certificates.
:func:`make_certificate_diff`
    Construct a validated certificate-difference record.
"""

import equinox as eqx
from beartype import beartype
from beartype.typing import List, Tuple
from jaxtyping import jaxtyped


class CertificateDiff(eqx.Module):
    """Store categorized differences between two forward certificates.

    The record groups differences by scientific meaning, numerical evidence,
    execution environment, and audit metadata. It does not run the model
    again.

    :see: :class:`~.test_inspection.TestCertificateDiff`

    Attributes
    ----------
    scientific : Tuple[str, ...]
        Differing scientific fields (**static** -- compile-time constants;
        changing them triggers retracing).
    numerical : Tuple[str, ...]
        Differing numerical-evidence fields (**static** -- compile-time
        constants; changing them triggers retracing).
    environment : Tuple[str, ...]
        Differing execution-environment fields (**static** -- compile-time
        constants; changing them triggers retracing).
    audit : Tuple[str, ...]
        Differing audit fields (**static** -- compile-time constants; changing
        them triggers retracing).

    Notes
    -----
    Inspection compares persisted metadata only. This carrier has no numerical
    leaves and does not reevaluate or differentiate a forward model.

    See Also
    --------
    make_certificate_diff : Construct a validated certificate-difference
        record.
    """

    scientific: Tuple[str, ...] = eqx.field(static=True)
    numerical: Tuple[str, ...] = eqx.field(static=True)
    environment: Tuple[str, ...] = eqx.field(static=True)
    audit: Tuple[str, ...] = eqx.field(static=True)

    @property
    @jaxtyped(typechecker=beartype)
    def identical(self) -> bool:
        """Return whether the record contains no categorized difference.

        :see: :class:`~.test_inspection.TestCertificateDiff`

        Returns
        -------
        identical : bool
            Whether every difference category is empty.
        """
        identical: bool = not any(
            (self.scientific, self.numerical, self.environment, self.audit)
        )
        return identical

    @property
    @jaxtyped(typechecker=beartype)
    def summary(self) -> str:
        """Return a one-line categorized comparison summary.

        :see: :class:`~.test_inspection.TestCertificateDiff`

        Returns
        -------
        summary : str
            Human-readable list of nonempty difference categories.
        """
        if self.identical:
            summary: str = "Certificates are identical."
            return summary
        parts: List[str] = []
        label: str
        values: Tuple[str, ...]
        for label, values in (
            ("scientific", self.scientific),
            ("numerical", self.numerical),
            ("environment", self.environment),
            ("audit", self.audit),
        ):
            if values:
                parts.append(f"{label}: {', '.join(values)}")
        summary = "; ".join(parts)
        return summary  # noqa: RET504 -- assign-before-return is required.


def _difference_names(value: Tuple[str, ...], name: str) -> Tuple[str, ...]:
    """PRIVATE: Validate one immutable sequence of differing field names.

    Parameters
    ----------
    value : Tuple[str, ...]
        Candidate tuple of differing certificate field names.
    name : str
        Category name used in the static error message.

    Returns
    -------
    value : Tuple[str, ...]
        The validated input tuple, unchanged.

    Raises
    ------
    ValueError
        If ``value`` is not a tuple, or if any entry is not a nonempty
        string. This is the static construction-time contract.

    Notes
    -----
    Check the container type first and then every entry. Return the same
    tuple so the factory can bind the result directly.
    """
    if not isinstance(value, tuple) or any(
        not isinstance(item, str) or not item for item in value
    ):
        msg: str = f"{name} must be a tuple of nonempty field names"
        raise ValueError(msg)
    return value


@jaxtyped(typechecker=beartype)
def make_certificate_diff(  # noqa: DOC502
    *,
    scientific: Tuple[str, ...] = (),
    numerical: Tuple[str, ...] = (),
    environment: Tuple[str, ...] = (),
    audit: Tuple[str, ...] = (),
) -> CertificateDiff:
    """Construct a validated certificate-difference record.

    Validate and freeze field names in each comparison category.

    :see: :class:`~.test_inspection.TestMakeCertificateDiff`

    Implementation Logic
    --------------------
    1. **Validate category names**::

           scientific=_difference_names(scientific, "scientific")

       Require immutable tuples of nonempty field names in every category.
    2. **Construct the difference**::

           difference = CertificateDiff(...)

       Bind and return the categorized comparison carrier.

    Parameters
    ----------
    scientific : Tuple[str, ...]
        Differing scientific fields (**static** -- compile-time constants;
        changing them triggers retracing). Default is empty.
    numerical : Tuple[str, ...]
        Differing numerical fields (**static** -- compile-time constants;
        changing them triggers retracing). Default is empty.
    environment : Tuple[str, ...]
        Differing environment fields (**static** -- compile-time constants;
        changing them triggers retracing). Default is empty.
    audit : Tuple[str, ...]
        Differing audit fields (**static** -- compile-time constants; changing
        them triggers retracing). Default is empty.

    Returns
    -------
    difference : CertificateDiff
        Validated immutable certificate difference.

    Raises
    ------
    ValueError
        If a category is not a tuple of nonempty field names.

    Notes
    -----
    Validation is static and does not introduce a gradient path.
    """
    difference: CertificateDiff = CertificateDiff(
        scientific=_difference_names(scientific, "scientific"),
        numerical=_difference_names(numerical, "numerical"),
        environment=_difference_names(environment, "environment"),
        audit=_difference_names(audit, "audit"),
    )
    return difference


__all__: list[str] = [
    "CertificateDiff",
    "make_certificate_diff",
]
