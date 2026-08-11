"""Validate certificate-difference carriers and their public factory.

The cases cover empty and categorized comparisons plus rejection of malformed
immutable category sequences in ``diffpes.types.inspection``.
"""

import pytest

from diffpes.types import CertificateDiff, make_certificate_diff


class TestCertificateDiff:
    """Validate :class:`~diffpes.types.CertificateDiff` derived properties.

    The carrier must distinguish an identical comparison from categorized
    scientific, numerical, environment, and audit differences.

    :see: :class:`~diffpes.types.CertificateDiff`
    """

    def test_summarizes_categorized_differences(self) -> None:
        """Report nonempty comparison categories in a stable one-line form.

        The check verifies both the ``identical`` predicate and the exact
        scientific and audit category names included in ``summary``.

        Notes
        -----
        The test constructs a difference through the public factory, then
        evaluates the
        two derived properties without rerunning any physical model.
        """
        difference: CertificateDiff = make_certificate_diff(
            scientific=("model_id",),
            audit=("created_at",),
        )

        assert difference.identical is False
        assert difference.summary == (
            "scientific: model_id; audit: created_at"
        )


class TestMakeCertificateDiff:
    """Validate :func:`~diffpes.types.make_certificate_diff`.

    The factory must produce an identical empty comparison and reject blank
    or mutable category entries.

    :see: :func:`~diffpes.types.make_certificate_diff`
    """

    def test_builds_identical_empty_difference(self) -> None:
        """Represent an empty comparison as identical certificates.

        The check verifies the factory defaults and the corresponding stable
        human-readable summary.

        Notes
        -----
        The test calls the factory with no categories and compares both public
        derived
        properties with their independently expected values.
        """
        difference: CertificateDiff = make_certificate_diff()

        assert difference.identical is True
        assert difference.summary == "Certificates are identical."

    def test_rejects_blank_category_name(self) -> None:
        """Reject a blank field name in any difference category.

        The check exercises the immutable category validation contract with an
        invalid scientific field name.

        Notes
        -----
        Supplies a one-element tuple containing an empty string and matches
        the factory's nonempty-field diagnostic.
        """
        with pytest.raises(ValueError, match="nonempty field names"):
            make_certificate_diff(scientific=("",))
