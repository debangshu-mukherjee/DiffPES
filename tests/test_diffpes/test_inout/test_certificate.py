"""Validate canonical forward-certificate identity handling.

The tests cover public behavior, validation, and stable scientific identity
in the supported certification regime.
"""

from diffpes.inout import certificate_identity, finalize_certificate
from diffpes.types import ForwardCertificate
from tests._factories import sample_forward_certificate


class TestCertificateIdentity:
    """Verify :func:`~diffpes.inout.certificate_identity`.

    The cases separate scientific identity from audit execution fields.

    :see: :func:`~diffpes.inout.certificate_identity`
    """

    def test_audit_fields_do_not_change_scientific_identity(self) -> None:
        """Keep one identity across distinct execution IDs and timestamps.

        The scientific identity must exclude both declared audit fields.

        Notes
        -----
        The test changes only the two fields classified as audit metadata.
        """
        left: ForwardCertificate = sample_forward_certificate(
            execution_id="audit-left",
            started_at_utc="2026-07-20T00:00:00Z",
        )
        right: ForwardCertificate = sample_forward_certificate(
            execution_id="audit-right",
            started_at_utc="2026-07-21T00:00:00Z",
        )
        assert left.manifest.execution_id != right.manifest.execution_id
        assert certificate_identity(left) == certificate_identity(right)


class TestFinalizeCertificate:
    """Verify :func:`~diffpes.inout.finalize_certificate`.

    The case replaces the compiled placeholder at the canonical I/O boundary.

    :see: :func:`~diffpes.inout.finalize_certificate`
    """

    def test_final_identity_matches_canonical_record(self) -> None:
        """Replace an arbitrary checksum with the computed scientific identity.

        The stored identity must equal a new computation from the result.

        Notes
        -----
        The test finalizes the complete shared certificate fixture once.
        """
        certificate: ForwardCertificate = sample_forward_certificate()
        finalized: ForwardCertificate = finalize_certificate(certificate)
        assert finalized.certificate_checksum == certificate_identity(
            finalized
        )
