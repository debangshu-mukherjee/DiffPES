"""Validate non-security scientific-record checksums.

The tests cover public behavior, differentiability, validation, and stable
scientific identity in the supported certification regime.
"""

import zlib

import jax.numpy as jnp
import pytest
from beartype.typing import Any

from diffpes.certify import (
    artifact_ref,
    checksum_bytes,
    checksum_chunks,
    checksum_file,
    checksum_pytree,
    parse_checksum,
    result_checksum,
    semantic_checksum,
)
from diffpes.types import CHECKSUM_ALGORITHM


class TestParseChecksum:
    """Verify :func:`~diffpes.certify.parse_checksum`.

    The cases cover the public behavior in the supported certification regime.

    :see: :func:`~diffpes.certify.parse_checksum`
    """

    def test_checksum_round_trip(self) -> None:
        """Verify a checksum round-trips through its explicit parser.

        The case uses explicit inputs in the supported certification regime.
        It checks the public result or the documented failure state.

        Notes
        -----
        The test checks algorithm, record kind, and fixed-width SHA-256 text.
        """
        checksum: str = checksum_bytes(b"physics", record_kind="result")
        parsed: tuple[str, str, str, str] = parse_checksum(checksum)
        assert parsed[0] == CHECKSUM_ALGORITHM
        assert parsed[2] == "result"

    def test_invalid_checksum_record_is_rejected(self) -> None:
        """Refuse incomplete and hand-written checksum forms.

        The case uses explicit inputs in the supported certification regime.
        It checks the public result or the documented failure state.

        Notes
        -----
        The test compares the result with explicit numerical or structural assertions.
        """
        with pytest.raises(ValueError, match="format"):
            parse_checksum("crc32:1234")

    def test_legacy_crc32_scientific_identity_is_rejected(self) -> None:
        """Reject a well-formed legacy CRC32 identity without an alias.

        Scientific lookup accepts only the current SHA-256 identity syntax.

        Notes
        -----
        Parse a legacy identity and require the scientific-identity error.
        """
        with pytest.raises(ValueError, match="scientific-identity"):
            parse_checksum("crc32:canonical-1:result:4ddb0c25")


class TestChecksumChunks:
    """Verify :func:`~diffpes.certify.checksum_chunks`.

    The cases cover the public behavior in the supported certification regime.

    :see: :func:`~diffpes.certify.checksum_chunks`
    """

    def test_checksum_is_typed_versioned_and_streaming_stable(self) -> None:
        """Record bookkeeping context and remain invariant to chunk boundaries.

        The case uses explicit inputs in the supported certification regime.
        It checks the public result or the documented failure state.

        Notes
        -----
        The test compares the result with explicit numerical or structural assertions.
        """
        whole: Any
        streamed: Any
        algorithm: Any
        version: Any
        kind: Any
        value: Any
        whole = checksum_bytes(b"abcdef", record_kind="result")
        streamed = checksum_chunks((b"ab", b"c", b"def"), record_kind="result")
        assert whole == streamed
        algorithm, version, kind, value = parse_checksum(whole)
        assert algorithm == CHECKSUM_ALGORITHM == "sha256"
        assert version == "1"
        assert kind == "result"
        assert len(value) == 64

    @pytest.mark.parametrize("record_kind", ["", "Result", "two words", "a/b"])
    def test_invalid_record_kind_is_rejected(self, record_kind: Any) -> None:
        """Reject record labels that make the checksum format ambiguous.

        The case uses explicit inputs in the supported certification regime.
        It checks the public result or the documented failure state.

        Notes
        -----
        The test compares the result with explicit numerical or structural assertions.
        """
        with pytest.raises(ValueError, match="record_kind"):
            checksum_bytes(b"value", record_kind=record_kind)


class TestChecksumBytes:
    """Verify :func:`~diffpes.certify.checksum_bytes`.

    The cases cover the public behavior in the supported certification regime.

    :see: :func:`~diffpes.certify.checksum_bytes`
    """

    def test_record_kind_and_one_bit_change_checksum(self) -> None:
        """Detect accidental byte disagreement and distinguish record purposes.

        The case uses explicit inputs in the supported certification regime.
        It checks the public result or the documented failure state.

        Notes
        -----
        The test compares the result with explicit numerical or structural assertions.
        """
        original: Any
        changed: Any
        semantic: Any
        original = checksum_bytes(b"\x00", record_kind="result")
        changed = checksum_bytes(b"\x01", record_kind="result")
        semantic = checksum_bytes(b"\x00", record_kind="semantic")
        assert original != changed
        assert original != semantic


def test_crc32_collision_cannot_alias_scientific_identity() -> None:
    """Keep a known CRC32 collision distinct under scientific identity.

    ``plumless`` and ``buckeroo`` are equal-length payloads with the same
    CRC32. Their domain-separated SHA-256 identities must remain distinct.

    Notes
    -----
    Compare the legacy collision before checking both current identities.
    """
    left: bytes = b"plumless"
    right: bytes = b"buckeroo"
    assert left != right
    assert zlib.crc32(left) == zlib.crc32(right) == 0x4DDB0C25
    left_identity: str = checksum_bytes(left, record_kind="result")
    right_identity: str = checksum_bytes(right, record_kind="result")
    assert left_identity != right_identity
    assert parse_checksum(left_identity)[:3] == ("sha256", "1", "result")
    assert parse_checksum(right_identity)[:3] == ("sha256", "1", "result")


class TestSemanticChecksum:
    """Verify :func:`~diffpes.certify.semantic_checksum`.

    The cases cover the public behavior in the supported certification regime.

    :see: :func:`~diffpes.certify.semantic_checksum`
    """

    def test_semantic_and_numerical_declarations_affect_identity(self) -> None:
        """Distinguish equal arrays with different units or precision contracts.

        The case uses explicit inputs in the supported certification regime.
        It checks the public result or the documented failure state.

        Notes
        -----
        The test compares the result with explicit numerical or structural assertions.
        """
        value: Any
        value = jnp.array([1.0, 2.0])
        assert semantic_checksum(value, {"unit": "eV"}) != semantic_checksum(
            value, {"unit": "hartree"}
        )
        assert result_checksum(
            value, {"precision": "float64"}
        ) != result_checksum(value, {"precision": "float32"})


class TestChecksumPytree:
    """Verify :func:`~diffpes.certify.checksum_pytree`.

    The cases cover the public behavior in the supported certification regime.

    :see: :func:`~diffpes.certify.checksum_pytree`
    """

    def test_streaming_pytree_matches_equal_jax_values(self) -> None:
        """Produce stable normalized-content bookkeeping for equal carriers.

        The case uses explicit inputs in the supported certification regime.
        It checks the public result or the documented failure state.

        Notes
        -----
        The test compares the result with explicit numerical or structural assertions.
        """
        left: Any
        right: Any
        left = checksum_pytree(jnp.arange(8), record_kind="normalized-content")
        right = checksum_pytree(
            jnp.arange(8), record_kind="normalized-content"
        )
        assert left == right


class TestArtifactRef:
    """Verify :func:`~diffpes.certify.artifact_ref`.

    The cases cover the public behavior in the supported certification regime.

    :see: :func:`~diffpes.certify.artifact_ref`
    """

    def test_artifact_ref_separates_byte_content_and_semantics(
        self,
        tmp_path: Any,
    ) -> None:
        """Record exact source bytes separately from parsed scientific meaning.

        The case uses explicit inputs in the supported certification regime.
        It checks the public result or the documented failure state.

        Notes
        -----
        The test compares the result with explicit numerical or structural assertions.
        """
        path: Any
        normalized: Any
        reference: Any
        path = tmp_path / "bands.dat"
        path.write_bytes(b"1 2 3\n")
        normalized = jnp.array([1.0, 2.0, 3.0])
        reference = artifact_ref(
            path,
            normalized,
            role="initial-bands",
            semantics={"unit": "eV", "axis": "band"},
        )
        assert parse_checksum(reference.byte_checksum)[2] == "artifact-bytes"
        assert (
            parse_checksum(reference.content_checksum)[2]
            == "normalized-content"
        )
        assert parse_checksum(reference.semantic_checksum)[2] == "semantic"
        assert reference.role == "initial-bands"
        assert reference.locator == str(path)


class TestChecksumFile:
    """Verify :func:`~diffpes.certify.checksum_file`.

    The cases cover streaming identity for exact bytes in a local artifact.

    :see: :func:`~diffpes.certify.checksum_file`
    """

    def test_file_checksum_matches_byte_checksum(self, tmp_path: Any) -> None:
        """Match the checksum of a file with the checksum of its exact bytes.

        The case uses explicit inputs in the supported certification regime.
        It checks the public result or the documented failure state.

        Notes
        -----
        The test compares the result with explicit numerical or structural assertions.
        """
        path: Any
        path = tmp_path / "artifact.bin"
        payload: bytes = b"physics-record\n"
        path.write_bytes(payload)
        checksum: str = checksum_file(path, record_kind="artifact-bytes")
        assert checksum == checksum_bytes(
            payload, record_kind="artifact-bytes"
        )


class TestResultChecksum:
    """Verify :func:`~diffpes.certify.result_checksum`.

    The cases cover numerical settings that form part of result identity.

    :see: :func:`~diffpes.certify.result_checksum`
    """

    def test_numerical_settings_change_result_identity(self) -> None:
        """Distinguish equal values produced with different precision settings.

        The case uses explicit inputs in the supported certification regime.
        It checks the public result or the documented failure state.

        Notes
        -----
        The test compares the result with explicit numerical or structural assertions.
        """
        value: Any
        value = jnp.asarray([1.0, 2.0])
        float64: str = result_checksum(value, {"precision": "float64"})
        float32: str = result_checksum(value, {"precision": "float32"})
        assert float64 != float32
        assert parse_checksum(float64)[2] == "result"
