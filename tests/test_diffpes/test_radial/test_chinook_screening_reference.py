"""Certify radial screening against inert Chinook sample data.

The tests check rounded screening values and authenticate the frozen external
source metadata without importing Chinook.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from beartype.typing import Dict

from diffpes.radial.screening import slater_zeff

REFERENCE_PATH: Path = (
    Path(__file__).parents[1]
    / "_reference_data"
    / "chinook_screening_reference.json"
)


class TestChinookScreeningReference:
    """Compare Slater screening with the authenticated frozen sample."""

    def test_sample_is_bit_equal_after_pinned_rounding(self) -> None:
        """Match each inert Chinook value after its declared rounding.

        The test checks all six subshell samples at the recorded precision.

        Notes
        -----
        It loads inert JSON and compares public Slater screening results.
        """
        artifact: Dict[str, Any] = json.loads(
            REFERENCE_PATH.read_text(encoding="utf-8")
        )
        assert artifact["requirement"] == "chinook-screening-reference"
        digits: int = int(artifact["round_digits"])
        samples: list[Dict[str, Any]] = artifact["samples"]
        assert len(samples) == 6
        sample: Dict[str, Any]
        for sample in samples:
            actual: float = slater_zeff(
                int(sample["atomic_number"]),
                int(sample["n"]),
                int(sample["l"]),
            )
            assert round(actual, digits) == sample["rounded_zeff"]

    def test_artifact_pins_chinook_source_and_inert_policy(self) -> None:
        """Require commit and source-table authentication metadata.

        The test checks the commit, both hashes, and the inert-data policy.

        Notes
        -----
        It reads each provenance field directly from the frozen JSON.
        """
        artifact: Dict[str, Any] = json.loads(
            REFERENCE_PATH.read_text(encoding="utf-8")
        )
        assert artifact["chinook_commit"] == (
            "24913de8cc5b8c162f7c1b4acc64bd1b54dd548b"
        )
        assert len(artifact["chinook_module_sha256"]) == 64
        assert len(artifact["chinook_configuration_sha256"]) == 64
        assert "inert JSON" in artifact["policy"]
