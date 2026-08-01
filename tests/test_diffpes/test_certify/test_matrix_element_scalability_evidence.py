"""Validate the reproducible matrix-element scalability benchmark artifact.

The tests check graph scaling, compiler memory, retained IR, and raw timing
statistics against the committed literal-shape measurement.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import re
import statistics
from pathlib import Path
from typing import Any

import numpy as np

ARTIFACT_DIRECTORY: Path = (
    Path(__file__).parents[1]
    / "_reference_data"
    / "matrix_element_scalability"
)
ARTIFACT_PATH: Path = ARTIFACT_DIRECTORY / "cpu_benchmark.json"
REPOSITORY_ROOT: Path = Path(__file__).parents[3]


def _artifact() -> dict[str, Any]:
    """Load the committed literal-shape benchmark record."""
    artifact: dict[str, Any] = json.loads(
        ARTIFACT_PATH.read_text(encoding="utf-8")
    )
    return artifact


def _sha256(path: Path) -> str:
    """Return one retained artifact digest."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _array_shapes(ir_text: str) -> set[tuple[int, ...]]:
    """Extract numeric dimensions from the retained compiler text."""
    shapes: set[tuple[int, ...]] = set()
    match: re.Match[str]
    for match in re.finditer(r"\[([0-9,\s]+)\]", ir_text):
        dimensions: tuple[int, ...] = tuple(
            int(value) for value in match.group(1).split(",") if value.strip()
        )
        if dimensions:
            shapes.add(dimensions)
    return shapes


class TestMatrixElementScalabilityEvidence:
    """Check structural, allocation, and raw-timing evidence."""

    def test_s1_sublinear_equations_and_compile_reuse(self) -> None:
        """Require constant graph size and one fixed-shape channel compile.

        The test checks all three orbital counts and each recorded cache size.

        Notes
        -----
        It loads the JSON artifact and compares exact structural counters.
        """
        artifact: dict[str, Any] = _artifact()
        assert artifact["schema"] == "diffpes.matrix-element-scalability.v2"
        relative_path: str
        digest: str
        for relative_path, digest in artifact["source_sha256"].items():
            source_path: Path = REPOSITORY_ROOT / relative_path
            assert _sha256(source_path) == digest
        assert artifact["process_peak_rss_bytes_non_authoritative"] > 0
        s1: dict[str, Any] = artifact["s1"]
        orbital_counts: list[int] = s1["orbital_counts"]
        equation_counts: list[int] = s1["recursive_jaxpr_equation_counts"]
        assert orbital_counts == [9, 18, 36]
        assert len(set(equation_counts)) == 1
        assert s1["equation_count_growth"] < (
            orbital_counts[-1] - orbital_counts[0]
        )
        assert s1["compile_cache_sizes"] == [0, 1, 1]
        assert s1["composed_sweep_compile_cache_sizes"] == [
            0,
            1,
            1,
            1,
            1,
            1,
            1,
        ]
        assert s1["composed_sweep_trace_counts"] == [0, 1, 1, 1, 1, 1, 1]
        assert s1["result"] == "pass"

    def test_s2_dynamic_arguments_live_allocation_and_retained_ir(
        self,
    ) -> None:
        """Require literal dimensions, XLA authority, and no K-E-B cube.

        The test checks dynamic allocation, retained IR, and reduced outputs.

        Notes
        -----
        It recomputes live bytes and verifies both compressed artifact hashes.
        """
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
        text: str
        for text in (jaxpr_text, hlo_text):
            assert "SCALAR-ENERGY VALUE+GRADIENT" in text
            assert "EIGHT-ENERGY REDUCED SCAN" in text
        parsed_shapes: set[tuple[int, ...]] = _array_shapes(
            f"{jaxpr_text}\n{hlo_text}"
        )
        forbidden_shapes: set[tuple[int, ...]] = {
            shape
            for shape in parsed_shapes
            if len(shape) == 3 and sorted(shape) == [8, 18, 4096]
        }
        assert forbidden_shapes == set()
        assert s2["forbidden_k_e_b_shapes"] == []
        assert s2["parsed_array_shape_count"] == len(parsed_shapes)
        assert s2["result"] == "pass"

    def test_s3_raw_repetitions_recompute_medians_and_both_ratios(
        self,
    ) -> None:
        """Recompute every reported S3 statistic from seven raw runs.

        The test checks synchronized samples, medians, and both reuse ratios.

        Notes
        -----
        It derives each statistic directly from the four raw timing series.
        """
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
