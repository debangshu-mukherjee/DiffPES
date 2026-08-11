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

import numpy as np
from beartype.typing import Any, Dict, List, Tuple

ARTIFACT_DIRECTORY: Path = (
    Path(__file__).parents[1]
    / "_reference_data"
    / "matrix_element_scalability"
)
ARTIFACT_PATH: Path = ARTIFACT_DIRECTORY / "cpu_benchmark.json"
REPOSITORY_ROOT: Path = Path(__file__).parents[3]


def _artifact() -> Dict[str, Any]:
    """PRIVATE: Load the committed literal-shape benchmark record.

    Returns
    -------
    artifact : Dict[str, Any]
        Parsed JSON benchmark artifact.

    Notes
    -----
    Reads the UTF-8 JSON file at the committed artifact path.
    """
    artifact: Dict[str, Any] = json.loads(
        ARTIFACT_PATH.read_text(encoding="utf-8")
    )
    return artifact


def _sha256(path: Path) -> str:
    """PRIVATE: Return one retained artifact digest.

    Parameters
    ----------
    path : Path
        File whose bytes the digest covers.

    Returns
    -------
    digest : str
        Hexadecimal SHA-256 digest of the complete file content.

    Notes
    -----
    Reads the file bytes in one call and hashes them with SHA-256.
    """
    digest: str = hashlib.sha256(path.read_bytes()).hexdigest()
    return digest


def _array_shapes(ir_text: str) -> set[Tuple[int, ...]]:
    r"""PRIVATE: Extract numeric dimensions from the retained compiler text.

    Parameters
    ----------
    ir_text : str
        Retained compiler IR text with bracketed shape literals.

    Returns
    -------
    shapes : set[Tuple[int, ...]]
        Distinct non-empty dimension tuples found in the text.

    Notes
    -----
    Matches bracketed integer lists with the pattern
    ``\[([0-9,\s]+)\]`` and converts each list to a tuple of ints.
    """
    shapes: set[Tuple[int, ...]] = set()
    match: re.Match[str]
    for match in re.finditer(r"\[([0-9,\s]+)\]", ir_text):
        dimensions: Tuple[int, ...] = tuple(
            int(value) for value in match.group(1).split(",") if value.strip()
        )
        if dimensions:
            shapes.add(dimensions)
    return shapes


class TestMatrixElementScalabilityEvidence:
    """Check structural, allocation, and raw-timing evidence.

    The cases authenticate the frozen benchmark and enforce each memory bound.
    """

    def test_sublinear_equations_and_compile_reuse(self) -> None:
        """Require constant graph size and one fixed-shape channel compile.

        The test checks all three orbital counts and each recorded cache size.

        Notes
        -----
        It loads the JSON artifact and compares exact structural counters.
        """
        artifact: Dict[str, Any] = _artifact()
        assert artifact["schema"] == "diffpes.matrix-element-scalability.v2"
        relative_path: str
        digest: str
        for relative_path, digest in artifact["source_sha256"].items():
            source_path: Path = REPOSITORY_ROOT / relative_path
            assert _sha256(source_path) == digest
        assert artifact["process_peak_rss_bytes_non_authoritative"] > 0
        compile_reuse: Dict[str, Any] = artifact["compile_reuse"]
        orbital_counts: List[int] = compile_reuse["orbital_counts"]
        equation_counts: List[int] = compile_reuse[
            "recursive_jaxpr_equation_counts"
        ]
        assert orbital_counts == [9, 18, 36]
        assert len(set(equation_counts)) == 1
        assert compile_reuse["equation_count_growth"] < (
            orbital_counts[-1] - orbital_counts[0]
        )
        assert compile_reuse["compile_cache_sizes"] == [0, 1, 1]
        assert compile_reuse["composed_sweep_compile_cache_sizes"] == [
            0,
            1,
            1,
            1,
            1,
            1,
            1,
        ]
        assert compile_reuse["composed_sweep_trace_counts"] == [
            0,
            1,
            1,
            1,
            1,
            1,
            1,
        ]
        assert compile_reuse["result"] == "pass"

    def test_dynamic_arguments_live_allocation_and_retained_ir(
        self,
    ) -> None:
        """Require literal dimensions, XLA authority, and no K-E-B cube.

        The test checks dynamic allocation, retained IR, and reduced outputs.

        Notes
        -----
        It recomputes live bytes and verifies both compressed artifact hashes.
        """
        literal: Dict[str, Any] = _artifact()["literal_allocation"]
        assert (literal["n_k"], literal["n_orb"], literal["n_energy"]) == (
            4096,
            18,
            8,
        )
        assert literal["output_shape"] == [8, 4096, 6]
        assert literal["scalar_output_shape"] == [4096, 6]
        assert literal["gradient_shape"] == [18]
        assert literal["forbidden_k_e_b_shape_present"] is False
        memory: Dict[str, Any] = literal["memory_analysis"]
        assert memory["authority_available"] is True
        live_values: List[int] = []
        name: str
        for name in ("scalar_value_and_gradient", "reduced_scan"):
            record: Dict[str, Any] = memory[name]
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

        jaxpr_path: Path = ARTIFACT_DIRECTORY / literal["jaxpr_gzip"]
        hlo_path: Path = ARTIFACT_DIRECTORY / literal["hlo_gzip"]
        assert _sha256(jaxpr_path) == literal["jaxpr_gzip_sha256"]
        assert _sha256(hlo_path) == literal["hlo_gzip_sha256"]
        jaxpr_text: str = gzip.decompress(jaxpr_path.read_bytes()).decode()
        hlo_text: str = gzip.decompress(hlo_path.read_bytes()).decode()
        text: str
        for text in (jaxpr_text, hlo_text):
            assert "SCALAR-ENERGY VALUE+GRADIENT" in text
            assert "EIGHT-ENERGY REDUCED SCAN" in text
        parsed_shapes: set[Tuple[int, ...]] = _array_shapes(
            f"{jaxpr_text}\n{hlo_text}"
        )
        forbidden_shapes: set[Tuple[int, ...]] = {
            shape
            for shape in parsed_shapes
            if len(shape) == 3 and sorted(shape) == [8, 18, 4096]
        }
        assert forbidden_shapes == set()
        assert literal["forbidden_k_e_b_shapes"] == []
        assert literal["parsed_array_shape_count"] == len(parsed_shapes)
        assert literal["result"] == "pass"

    def test_raw_repetitions_recompute_medians_and_both_ratios(
        self,
    ) -> None:
        """Recompute every reported throughput statistic from seven raw runs.

        The test checks synchronized samples, medians, and both reuse ratios.

        Notes
        -----
        It derives each statistic directly from the four raw timing series.
        """
        throughput: Dict[str, Any] = _artifact()["throughput"]
        assert throughput["warmups"] == 2
        assert throughput["repetitions"] == 7
        assert throughput["synchronized"] is True
        raw: Dict[str, List[float]] = throughput["raw_seconds"]
        medians: Dict[str, float] = throughput["median_seconds"]
        name: str
        values: List[float]
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
            throughput["contraction_ratio"],
            contraction_ratio,
            rtol=0.0,
            atol=0.0,
        )
        np.testing.assert_allclose(
            throughput["pipeline_ratio"],
            pipeline_ratio,
            rtol=0.0,
            atol=0.0,
        )
        assert contraction_ratio < 1.5
        assert pipeline_ratio < 1.0
        assert throughput["result"] == "pass"
