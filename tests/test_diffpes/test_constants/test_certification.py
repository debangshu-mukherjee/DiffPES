"""Validate constants for certification identities and schemas.

The tests exercise the frozen prefixes and regular expressions with accepted
and rejected examples from the public certification vocabulary.
"""

from diffpes.constants import (
    CANONICAL_JSON_PREFIX,
    CERTIFICATION_IDENTIFIER_PATTERN,
    CERTIFICATION_SEMVER_PATTERN,
    CHECKSUM_ALGORITHM,
)


class TestCertificationConstants:
    """Validate stable certification identifiers and prefixes.

    The cases match representative valid and invalid text against the frozen
    schema expressions.

    Notes
    -----
    Use full matches because certification identities are complete records.
    """

    def test_identity_vocabulary_is_stable(self) -> None:
        """Preserve canonical identity constants.

        The check verifies one binary prefix, one algorithm name, and two
        complete regular-expression contracts.

        Notes
        -----
        Match accepted and rejected examples to detect broadened schemas.
        """
        assert CANONICAL_JSON_PREFIX == b"DIFFPES-CANONICAL-JSON-V1\x00"
        assert CHECKSUM_ALGORITHM == "sha256"
        assert CERTIFICATION_IDENTIFIER_PATTERN.fullmatch(
            "org.diffpes.model.arpes"
        )
        assert not CERTIFICATION_IDENTIFIER_PATTERN.fullmatch("DiffPES")
        assert CERTIFICATION_SEMVER_PATTERN.fullmatch("1.2.3")
        assert not CERTIFICATION_SEMVER_PATTERN.fullmatch("1.2")
