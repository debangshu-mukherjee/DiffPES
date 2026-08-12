"""Freeze source identities for the external slab scalability evidence."""

from __future__ import annotations

import hashlib
from pathlib import Path

from beartype.typing import List, Tuple

REPOSITORY_ROOT: Path = Path(__file__).resolve().parents[2]
EVIDENCE_DIRECTORY: Path = (
    REPOSITORY_ROOT.parent / "diffpes-plans/verification/slab_scalability"
)
SOURCE_MANIFEST_PATH: Path = EVIDENCE_DIRECTORY / "SOURCE_SHA256SUMS"
ARTIFACT_MANIFEST_PATH: Path = EVIDENCE_DIRECTORY / "SHA256SUMS"
ENVIRONMENT_PATH: Path = EVIDENCE_DIRECTORY / "environment.txt"

SOURCE_PATHS: Tuple[str, ...] = (
    "pyproject.toml",
    "uv.lock",
    "tests/_reference_tools/measure_chunked_slab_scaling.py",
    "tests/_reference_tools/freeze_slab_scalability_provenance.py",
    "src/diffpes/__init__.py",
    "src/diffpes/constants/__init__.py",
    "src/diffpes/constants/carriers.py",
    "src/diffpes/constants/shared.py",
    "src/diffpes/maths/__init__.py",
    "src/diffpes/maths/rotations.py",
    "src/diffpes/maths/safe.py",
    "src/diffpes/tightb/__init__.py",
    "src/diffpes/tightb/diagonalize.py",
    "src/diffpes/tightb/hamiltonian.py",
    "src/diffpes/tightb/slab.py",
    "src/diffpes/tightb/slab_assembly.py",
    "src/diffpes/tightb/slab_rotation.py",
    "src/diffpes/tightb/slab_surface_cell.py",
    "src/diffpes/tightb/slab_topology.py",
    "src/diffpes/tightb/soc.py",
    "src/diffpes/types/__init__.py",
    "src/diffpes/types/aliases.py",
    "src/diffpes/types/diagonalized_bands.py",
    "src/diffpes/types/electronic_structure_validation.py",
    "src/diffpes/types/geometry.py",
    "src/diffpes/types/orbital_basis.py",
    "src/diffpes/types/slab_geometry.py",
    "src/diffpes/types/slab_topology.py",
    "src/diffpes/types/tb_model.py",
)

ARTIFACT_PATHS: Tuple[str, ...] = (
    "README.md",
    "environment.txt",
    "plan05_s1_s2_cpu.json",
    "plan05_s1_spinor_stretch_cpu.json",
    "SOURCE_SHA256SUMS",
)


def _sha256(path: Path) -> str:
    """PRIVATE: Return the SHA-256 identity of one complete file.

    Parameters
    ----------
    path : Path
        File whose complete bytes define the identity.

    Returns
    -------
    digest : str
        Lowercase hexadecimal SHA-256 digest.

    Notes
    -----
    Binary reads preserve the exact committed byte identity.
    """
    digest: str = hashlib.sha256(path.read_bytes()).hexdigest()
    return digest


def _source_line(relative: str) -> str:
    """PRIVATE: Format one source-manifest line.

    Parameters
    ----------
    relative : str
        Repository-relative source path.

    Returns
    -------
    line : str
        Digest and evidence-relative path in sha256sum format.

    Notes
    -----
    The evidence directory reaches the DiffPES repository through three
    parent components.
    """
    path: Path = REPOSITORY_ROOT / relative
    line: str = f"{_sha256(path)}  ../../../DiffPES/{relative}"
    return line


def _replace_snapshot_hash(environment_text: str, digest: str) -> str:
    """PRIVATE: Replace the source snapshot identity in environment text.

    Parameters
    ----------
    environment_text : str
        Existing environment record.
    digest : str
        New source-manifest SHA-256 identity.

    Returns
    -------
    updated : str
        Environment record with exactly one current snapshot identity.

    Raises
    ------
    RuntimeError
        If the record lacks one snapshot identity line.
    """
    prefix: str = "source_snapshot_sha256: "
    lines: List[str] = environment_text.splitlines()
    matches: List[int] = [
        index for index, line in enumerate(lines) if line.startswith(prefix)
    ]
    if len(matches) != 1:
        raise RuntimeError("environment must contain one snapshot hash")
    lines[matches[0]] = f"{prefix}{digest}"
    updated: str = "\n".join(lines) + "\n"
    return updated


def main() -> None:
    """Freeze source, environment, and artifact checksum manifests.

    Notes
    -----
    The command computes every identity from current bytes. It never changes
    either numerical evidence JSON file.
    """
    source_text: str = (
        "\n".join(_source_line(relative) for relative in SOURCE_PATHS) + "\n"
    )
    SOURCE_MANIFEST_PATH.write_text(source_text, encoding="utf-8")
    source_digest: str = _sha256(SOURCE_MANIFEST_PATH)
    environment_text: str = ENVIRONMENT_PATH.read_text(encoding="utf-8")
    ENVIRONMENT_PATH.write_text(
        _replace_snapshot_hash(environment_text, source_digest),
        encoding="utf-8",
    )
    artifact_lines: List[str] = [
        f"{_sha256(EVIDENCE_DIRECTORY / relative)}  {relative}"
        for relative in ARTIFACT_PATHS
    ]
    relative: str
    for relative in (
        "tests/_reference_tools/measure_chunked_slab_scaling.py",
        "tests/_reference_tools/freeze_slab_scalability_provenance.py",
    ):
        artifact_lines.append(
            f"{_sha256(REPOSITORY_ROOT / relative)}  "
            f"../../../DiffPES/{relative}"
        )
    ARTIFACT_MANIFEST_PATH.write_text(
        "\n".join(artifact_lines) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
