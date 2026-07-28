"""Validate the reproducible Plan 06 S1--S3 benchmark artifact."""

from __future__ import annotations

import gzip
import hashlib
import json
import statistics
from pathlib import Path
from typing import Any

import numpy as np

ARTIFACT_DIRECTORY: Path = (
    Path(__file__).parents[1] / "_reference_data" / "plan06_scalability"
)
ARTIFACT_PATH: Path = ARTIFACT_DIRECTORY / "plan06_s1_s3_cpu.json"


def _artifact() -> dict[str, Any]:
    """Load the committed literal-shape benchmark record."""
    artifact: dict[str, Any] = json.loads(
        ARTIFACT_PATH.read_text(encoding="utf-8")
    )
    return artifact


def _sha256(path: Path) -> str:
    """Return one retained artifact digest."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


class TestPlan06ScalabilityEvidence:
    """Check structural, allocation, and raw-timing evidence."""

    def test_s1_sublinear_equations_and_compile_reuse(self) -> None:
        """Require constant graph size and one fixed-shape channel compile."""
        artifact: dict[str, Any] = _artifact()
        assert artifact["process_peak_rss_bytes_non_authoritative"] > 0
        s1: dict[str, Any] = artifact["s1"]
        orbital_counts: list[int] = s1["orbital_counts"]
        equation_counts: list[int] = s1["recursive_jaxpr_equation_counts"]
        assert orbital_counts == [9, 18, 36]
        assert len(set(equation_counts)) == 1
        assert s1["equation_count_growth"] < (
            orbital_counts[-1] - orbital_counts[0]
        )
        assert s1["compile_cache_sizes"] == [0, 1, 1, 1]
        assert s1["result"] == "pass"

    def test_s2_dynamic_arguments_live_allocation_and_retained_ir(
        self,
    ) -> None:
        """Require literal dimensions, XLA authority, and no K-E-B cube."""
        s2: dict[str, Any] = _artifact()["s2"]
        assert (s2["n_k"], s2["n_orb"], s2["n_energy"]) == (4096, 18, 8)
        assert s2["output_shape"] == [8, 4096, 6]
        assert s2["scalar_output_shape"] == [4096, 6]
        assert s2["gradient_shape"] == [18]
        assert s2["forbidden_k_e_b_shape_present"] is False
        memory: dict[str, Any] = s2["memory_analysis"]
        assert memory["authority_available"] is True
        live_values: list[int] = []
        name: str
        for name in ("scalar_value_and_gradient", "reduced_scan"):
            record: dict[str, Any] = memory[name]
            assert record["authority_available"] is True
            assert record["argument_size_bytes"] > 1_000_000
            recomputed_live: int = (
                record["argument_size_bytes"]
                + record["output_size_bytes"]
                + record["temporary_size_bytes"]
                - record["alias_size_bytes"]
            )
            assert recomputed_live == record["compiler_live_allocation_bytes"]
            live_values.append(recomputed_live)
        assert (
            max(live_values)
            == (memory["authoritative_maximum_live_allocation_bytes"])
        )
        assert max(live_values) < memory["limit_bytes"]

        jaxpr_path: Path = ARTIFACT_DIRECTORY / s2["jaxpr_gzip"]
        hlo_path: Path = ARTIFACT_DIRECTORY / s2["hlo_gzip"]
        assert _sha256(jaxpr_path) == s2["jaxpr_gzip_sha256"]
        assert _sha256(hlo_path) == s2["hlo_gzip_sha256"]
        jaxpr_text: str = gzip.decompress(jaxpr_path.read_bytes()).decode()
        hlo_text: str = gzip.decompress(hlo_path.read_bytes()).decode()
        for text in (jaxpr_text, hlo_text):
            assert "SCALAR-ENERGY VALUE+GRADIENT" in text
            assert "EIGHT-ENERGY REDUCED SCAN" in text
            compact: str = text.replace(" ", "")
            assert "8x4096x18" not in compact
            assert "4096x8x18" not in compact
            assert "8,4096,18" not in compact
            assert "4096,8,18" not in compact
        assert s2["result"] == "pass"

    def test_s3_raw_repetitions_recompute_medians_and_both_ratios(
        self,
    ) -> None:
        """Recompute every reported S3 statistic from seven raw runs."""
        s3: dict[str, Any] = _artifact()["s3"]
        assert s3["warmups"] == 2
        assert s3["repetitions"] == 7
        assert s3["synchronized"] is True
        raw: dict[str, list[float]] = s3["raw_seconds"]
        medians: dict[str, float] = s3["median_seconds"]
        name: str
        values: list[float]
        for name, values in raw.items():
            assert len(values) == 7
            assert all(value > 0.0 for value in values)
            np.testing.assert_allclose(
                medians[name],
                statistics.median(values),
                rtol=0.0,
                atol=0.0,
            )
        contraction_ratio: float = (
            medians["batched_contraction"]
            / medians["six_sequential_contractions"]
        )
        pipeline_ratio: float = (
            medians["late_reuse_pipeline"] / medians["six_rebuild_pipelines"]
        )
        np.testing.assert_allclose(
            s3["contraction_ratio"],
            contraction_ratio,
            rtol=0.0,
            atol=0.0,
        )
        np.testing.assert_allclose(
            s3["pipeline_ratio"],
            pipeline_ratio,
            rtol=0.0,
            atol=0.0,
        )
        assert contraction_ratio < 1.5
        assert pipeline_ratio < 1.0
        assert s3["result"] == "pass"
