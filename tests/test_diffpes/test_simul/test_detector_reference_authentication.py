"""Verify the inert manufactured detector-chain reference.

The consumer binds the archive bytes to literal artifact and generator
identities without importing or reading the plans-side generator.
"""

import hashlib
from pathlib import Path

from beartype.typing import Any

_REPOSITORY_ROOT: Path = Path(__file__).resolve().parents[3]
_REFERENCE_DIRECTORY: Path = (
    _REPOSITORY_ROOT / "tests" / "test_diffpes" / "_reference_data"
)
_ARTIFACT_PATH: Path = (
    _REFERENCE_DIRECTORY / "detector_chain_manufactured_reference.npz"
)
_REFERENCE_MANIFEST_PATH: Path = _REFERENCE_DIRECTORY / "MANIFEST.md"
_RETIRED_GENERATOR_PATH: Path = (
    _REPOSITORY_ROOT
    / "tests"
    / "_reference_tools"
    / "generate_detector_chain_manufactured_reference.py"
)
_GENERATOR_AUTHORITY: str = (
    "diffpes-plans/verification/detector_chain_manufactured/"
    "generate_detector_chain_manufactured_reference.py"
)
_ARTIFACT_SHA256: str = (
    "04e41d5f0fa2fe6111718bdc039f49344f48689d74ef0783408585cac76b55c3"
)
_GENERATOR_SHA256: str = (
    "9789939629293cdaa039f98cdaa9119014b3bfa7d9abad7389847c2ea1c758d6"
)


def _sha256(path: Path) -> str:
    """PRIVATE: Return the hexadecimal SHA-256 digest of one file.

    Parameters
    ----------
    path : Path
        File whose complete bytes define the identity.

    Returns
    -------
    digest : str
        Lowercase hexadecimal SHA-256 digest.
    """
    digest: Any = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def test_manufactured_detector_archive_has_authenticated_authority() -> None:
    """Bind the inert archive to literal artifact and generator identities.

    The plans-side generator remains outside the pytest execution boundary.
    Its literal digest and authority path cross through the inert manifest.
    The consumer recomputes the archive digest from the consumed bytes.

    Notes
    -----
    Also rejects resurrection of the retired test-side mutable generator.
    """
    reference_manifest: str = _REFERENCE_MANIFEST_PATH.read_text(
        encoding="utf-8"
    )

    assert _sha256(_ARTIFACT_PATH) == _ARTIFACT_SHA256
    assert f"`{_ARTIFACT_SHA256}`" in reference_manifest
    assert f"`{_GENERATOR_SHA256}`" in reference_manifest
    assert f"`{_GENERATOR_AUTHORITY}`" in reference_manifest
    assert not _RETIRED_GENERATOR_PATH.exists()
