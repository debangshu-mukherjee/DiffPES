"""Certify radial screening against inert Chinook sample data.

Extended Summary
----------------
The tests check rounded screening values and authenticate the frozen external
source metadata without importing Chinook.
"""

from __future__ import annotations

import json
from pathlib import Path

from beartype.typing import Any, Dict, List

from diffpes.radial import slater_zeff

REFERENCE_PATH: Path = (
    Path(__file__).parents[1]
    / "_reference_data"
    / "chinook_screening_reference.json"
)


class TestChinookScreeningReference:
    """Validate Slater screening against authenticated Chinook data.

    The cases cover six inert subshell samples and their provenance boundary.
    They compare rounded public values and inspect the commit, hashes, and
    inert-data policy in the frozen JSON artifact.

    :see: :func:`~diffpes.radial.slater_zeff`
    """

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
        samples: List[Dict[str, Any]] = artifact["samples"]
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
