"""Certify the Chinook sample portion of Plan 06 G4 from inert data."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from diffpes.radial.screening import slater_zeff

REFERENCE_PATH: Path = (
    Path(__file__).parents[1]
    / "_reference_data"
    / "plan06_chinook_screening_reference.json"
)


class TestPlan06ChinookScreeningReference:
    """Compare Slater screening with the authenticated frozen sample."""

    def test_sample_is_bit_equal_after_pinned_rounding(self) -> None:
        """Match each inert Chinook value after its declared rounding."""
        artifact: dict[str, Any] = json.loads(
            REFERENCE_PATH.read_text(encoding="utf-8")
        )
        assert artifact["gate"] == "06.G4"
        digits: int = int(artifact["round_digits"])
        samples: list[dict[str, Any]] = artifact["samples"]
        assert len(samples) == 6
        sample: dict[str, Any]
        for sample in samples:
            actual: float = slater_zeff(
                int(sample["atomic_number"]),
                int(sample["n"]),
                int(sample["l"]),
            )
            assert round(actual, digits) == sample["rounded_zeff"]

    def test_artifact_pins_chinook_source_and_inert_policy(self) -> None:
        """Require commit and source-table authentication metadata."""
        artifact: dict[str, Any] = json.loads(
            REFERENCE_PATH.read_text(encoding="utf-8")
        )
        assert artifact["chinook_commit"] == (
            "24913de8cc5b8c162f7c1b4acc64bd1b54dd548b"
        )
        assert len(artifact["chinook_module_sha256"]) == 64
        assert len(artifact["chinook_configuration_sha256"]) == 64
        assert "inert JSON" in artifact["policy"]
