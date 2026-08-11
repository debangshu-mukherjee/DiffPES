"""Validate authenticated detector-driver scaling and small runtime checks.

The committed artifact owns literal full-cube compiler allocation and JAXPR
shape evidence.  Small executable checks independently preserve checkpointed
values and gradients, fixed-shape compilation reuse, and geometry batching.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from beartype.typing import Any, Dict

from tests._reference_tools import measure_detector_scaling as scaling


def _repository_root() -> Path:
    """PRIVATE: Return the repository root for source authentication.

    Returns
    -------
    root : Path
        Absolute repository root.
    """
    root: Path = Path(__file__).resolve().parents[3]
    return root


def _artifact_path() -> Path:
    """PRIVATE: Return the detector-scaling JSON path.

    Returns
    -------
    path : Path
        Committed reference-data path.
    """
    path: Path = (
        Path(__file__).resolve().parents[1]
        / "_reference_data"
        / "detector_scalability"
        / "cpu_benchmark.json"
    )
    return path


def _artifact_digest() -> str:
    """PRIVATE: Return the frozen detector-scaling artifact digest.

    Returns
    -------
    digest : str
        SHA-256 digest of the committed literal CPU record.
    """
    digest: str = (
        "ca642cd6d4b1276937508f404c46487c505b1f1888e726747f6b21d345d3d0b4"
    )
    return digest


def _sha256(path: Path) -> str:
    """PRIVATE: Return the SHA-256 identity of one file.

    Parameters
    ----------
    path : Path
        File to authenticate.

    Returns
    -------
    digest : str
        Lowercase hexadecimal digest.
    """
    digest: str = hashlib.sha256(path.read_bytes()).hexdigest()
    return digest


def _artifact() -> Dict[str, Any]:
    """PRIVATE: Load the authenticated literal scaling record.

    Returns
    -------
    artifact : Dict[str, Any]
        Parsed benchmark record.

    Notes
    -----
    Authentication always precedes JSON parsing.
    """
    path: Path = _artifact_path()
    digest: str = _artifact_digest()
    assert _sha256(path) == digest
    artifact: Dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return artifact


class TestDetectorDriverScalingArtifact:
    """Validate allocation, rematerialization, and batching evidence.

    The cases authenticate bounded-allocation records and small companion
    records for rematerialization, compile reuse, and geometry batching.
    """

    def test_literal_cube_uses_bounded_allocation_without_full_carriers(
        self,
    ) -> None:
        """Require the exact full-cube shape and both memory budgets.

        The artifact fixes dimensions, source hashes, and XLA allocation
        ceilings.

        Notes
        -----
        XLA executable allocation supplies the compile-only device authority.
        The recursive JAXPR audit rejects every flattened or raster KBE shape
        and every complete final-momentum carrier.
        """
        artifact: Dict[str, Any] = _artifact()
        assert artifact["schema"] == "diffpes.detector-scalability.v1"
        assert artifact["backend"] == "cpu"
        assert artifact["x64_enabled"] is True
        assert artifact["requirements"] == [
            "detector-forward-memory",
            "complete-hamiltonian-gradient-memory",
            "fixed-shape-compile-reuse",
            "geometry-vmap-parity",
        ]
        relative_path: str
        digest: str
        for relative_path, digest in artifact["source_sha256"].items():
            assert _sha256(_repository_root() / relative_path) == digest

        literal: Dict[str, Any] = artifact["literal_detector_target"]
        assert (
            literal["n_kx"],
            literal["n_ky"],
            literal["n_energy"],
            literal["n_band"],
            literal["n_orbital"],
            literal["k_chunk"],
            literal["energy_chunk"],
        ) == (256, 256, 400, 20, 20, 32, 16)
        assert literal["expected_count_cube_shape"] == [1, 256, 256, 400]
        assert literal["expected_count_cube_bytes"] == 209_715_200
        assert literal["programs_executed"] is False
        assert literal["mapping_chart"] == (
            "signed-diagonal boundary-aware cubature"
        )
        shape_audit: Dict[str, Any] = literal["jaxpr_shape_audit"]
        assert shape_audit["contains_full_kbe_materialization"] is False
        assert shape_audit["contains_full_kinematics_materialization"] is False
        assert shape_audit["compact_kinematics_invariant"] == (
            "no canonical, flattened, or factored shape with K*E*3 elements"
        )
        assert shape_audit["full_kbe_element_count"] == 524_288_000
        assert shape_audit["full_kinematics_element_count"] == 78_643_200
        assert shape_audit["full_kbe_shape_matches"] == []
        assert shape_audit["full_kinematics_shape_matches"] == []
        assert shape_audit["forbidden_shape_matches"] == []
        assert shape_audit["result"] == "pass"

        name: str
        budget: int
        for name, budget in (
            ("forward", 2_000_000_000),
            ("forward_and_gradient", 12_000_000_000),
        ):
            measurement: Dict[str, Any] = literal[name]
            memory: Dict[str, Any] = measurement["memory_analysis"]
            assert measurement["budget_bytes"] == budget
            assert measurement["compilation_seconds"] > 0.0
            assert memory["authority_available"] is True
            assert memory["result"] == "measured"
            live: int = (
                memory["argument_size_bytes"]
                + memory["output_size_bytes"]
                + memory["temporary_size_bytes"]
                - memory["alias_size_bytes"]
            )
            assert live == memory["compiler_live_allocation_bytes"]
            assert live <= budget
            assert measurement["passes_budget"] is True
        assert literal["result"] == "pass"

    def test_small_companions_certify_remat_compile_reuse_and_vmap(
        self,
    ) -> None:
        """Require every executable companion verdict and tolerance.

        The artifact also records full-driver executable companion checks.

        Notes
        -----
        The artifact records full-driver Hamiltonian gradients and a complete
        ``ExperimentGeometry`` batch rather than proxy kernels.
        """
        artifact: Dict[str, Any] = _artifact()
        remat: Dict[str, Any] = artifact["rematerialization_comparison"]
        assert remat["mapping_chart"] == (
            "general rotation with strict target enclosure"
        )
        assert remat["value_passes_rtol_1e_12"] is True
        assert remat["gradient_passes_rtol_1e_12"] is True
        assert remat["nonzero_gradient"] is True
        assert remat["maximum_reference_gradient"] > 1.0e-10
        assert remat["result"] == "pass"

        batched: Dict[str, Any] = artifact[
            "compile_reuse_and_geometry_batching"
        ]
        assert batched["mapping_chart"] == (
            "general rotation with strict target enclosure"
        )
        assert batched["batch_length"] == 2
        assert batched["fixed_output_shape"] == [2, 1, 2, 2, 2]
        assert batched["trace_count"] == 1
        assert batched["compile_cache_sizes"] == [0, 1, 1, 1]
        assert batched["one_compilation"] is True
        assert batched["vmap_matches_direct_rtol_1e_12"] is True
        assert batched["result"] == "pass"
        assert artifact["result"] == "pass"


class TestDetectorDriverRuntimeScaling:
    """Execute small full-driver remat and geometry-batching checks.

    The cases compare checkpointed values and Hamiltonian gradients, then
    verify compile reuse and vectorized geometry for fixed shapes.
    """

    @pytest.mark.big_mem
    @pytest.mark.rss_limit_mb(1600)
    def test_checkpointing_preserves_values_and_hamiltonian_gradients(
        self,
    ) -> None:
        """Match checkpointed and plain full-driver derivatives.

        The executable comparison uses an asymmetric full-driver scalar loss.

        Notes
        -----
        The asymmetric scalar loss keeps the complete Hamiltonian gradient
        finite and nonzero.
        """
        record: Dict[str, Any] = scaling._remat_record()  # noqa: SLF001
        assert record["mapping_chart"] == (
            "general rotation with strict target enclosure"
        )
        assert record["value_passes_rtol_1e_12"] is True
        assert record["gradient_passes_rtol_1e_12"] is True
        assert record["nonzero_gradient"] is True
        assert record["result"] == "pass"

    @pytest.mark.big_mem
    @pytest.mark.rss_limit_mb(1600)
    def test_fixed_shape_sweeps_reuse_compilation_and_vmap_geometry(
        self,
    ) -> None:
        """Trace once across widths and fixed-length geometry sweeps.

        The sweep changes dynamic widths and photon energies at fixed shapes.

        Notes
        -----
        Direct single-geometry rows provide the independent batched-value
        comparison.
        """
        record: Dict[str, Any] = scaling._compile_and_vmap_record()  # noqa: SLF001
        assert record["mapping_chart"] == (
            "general rotation with strict target enclosure"
        )
        assert record["trace_count"] == 1
        assert record["compile_cache_sizes"] == [0, 1, 1, 1]
        assert record["one_compilation"] is True
        assert record["vmap_matches_direct_rtol_1e_12"] is True
        assert record["result"] == "pass"
