"""Validate the inert bulk-kz scan scalability evidence.

The mutable JAX harness remains outside pytest. These tests consume its frozen
JSON mirror. They bind every graph-defining source and inspect literal
bulk-kz allocations and companions without compiling the target.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from beartype.typing import Any, Dict, Tuple

_REPOSITORY_ROOT: Path = Path(__file__).resolve().parents[3]
_REFERENCE_DIRECTORY: Path = (
    _REPOSITORY_ROOT / "tests" / "test_diffpes" / "_reference_data"
)
_ARTIFACT_PATH: Path = _REFERENCE_DIRECTORY / "kz_scan_scalability.json"
_CONSUMER_MANIFEST_PATH: Path = (
    _REFERENCE_DIRECTORY / "kz_scan_scalability_manifest.json"
)
_ARTIFACT_SHA256: str = (
    "0f7d5ef8df5874c02ca3ecd0d4092efbc6ad2ab11660c0a53cb7e8219d71ffa8"
)
_CONSUMER_MANIFEST_SHA256: str = (
    "12bf9b7cd7d646c8194ae4b3a82ac920f6f91db9c7c4f2e5ec3e9c471691dde2"
)
_HARNESS_SHA256: str = (
    "607320aea72976b82e185e92b6be582aca0a6f645ef68d792c992db8c17ca47a"
)


def _sha256(path: Path) -> str:
    """PRIVATE: Return the SHA-256 identity of one complete file.

    Parameters
    ----------
    path : Path
        File whose bytes define the identity.

    Returns
    -------
    digest : str
        Lowercase hexadecimal digest.
    """
    digest: str = hashlib.sha256(path.read_bytes()).hexdigest()
    return digest


def _load_json(path: Path) -> Dict[str, Any]:
    """PRIVATE: Load one authenticated top-level JSON object.

    Parameters
    ----------
    path : Path
        JSON file to parse.

    Returns
    -------
    record : Dict[str, Any]
        Parsed top-level object.
    """
    value: Any = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    record: Dict[str, Any] = value
    return record


def test_kz_scan_scalability_authenticates_authority_and_graph() -> None:
    """Bind the inert mirror to its sole harness and complete source graph.

    Hash both copies and both manifests before parsing any evidence fields.

    Notes
    -----
    The test imports neither JAX nor the mutable external harness.
    """
    assert _sha256(_ARTIFACT_PATH) == _ARTIFACT_SHA256
    assert _sha256(_CONSUMER_MANIFEST_PATH) == _CONSUMER_MANIFEST_SHA256

    consumer_manifest: Dict[str, Any] = _load_json(_CONSUMER_MANIFEST_PATH)
    assert consumer_manifest["schema_version"] == 1
    assert consumer_manifest["artifact"] == {
        "path_from_diffpes_root": (
            "tests/test_diffpes/_reference_data/kz_scan_scalability.json"
        ),
        "sha256": _ARTIFACT_SHA256,
    }
    assert consumer_manifest["plans_artifact"] == {
        "path": "diffpes-plans/verification/kz_scan_scalability.json",
        "sha256": _ARTIFACT_SHA256,
    }
    assert consumer_manifest["generator"] == {
        "mutable_authority_count": 1,
        "path": ("diffpes-plans/verification/measure_kz_scan_scalability.py"),
        "sha256": _HARNESS_SHA256,
    }
    assert (
        "without importing or executing"
        in consumer_manifest["consumer_boundary"]
    )

    artifact: Dict[str, Any] = _load_json(_ARTIFACT_PATH)
    assert artifact["generator"] == {
        "path": ("diffpes-plans/verification/measure_kz_scan_scalability.py"),
        "sha256": _HARNESS_SHA256,
    }
    relative_path: str
    digest: str
    for relative_path, digest in artifact["source_sha256"].items():
        assert _sha256(_REPOSITORY_ROOT / relative_path) == digest

    convergence: Dict[str, Any] = artifact["quadrature_convergence_authority"]
    assert artifact["selected_n_kz"] == 2048
    assert convergence["selected_n_kz"] == 2048
    assert convergence["artifact"]["path"] == (
        "diffpes-plans/verification/kz_quadrature_convergence.json"
    )
    assert convergence["manifest"]["path"] == (
        "diffpes-plans/verification/kz_quadrature_convergence_manifest.json"
    )
    assert convergence["generator"]["path"] == (
        "diffpes-plans/verification/generate_kz_quadrature_convergence.py"
    )


def test_kz_scan_scalability_freezes_literal_shape_and_allocation() -> None:
    """Require the literal dimensions, recursive audit, and memory budgets.

    Pin both compiler-live allocation authorities at the registered target.

    Notes
    -----
    The literal target is compile-only; XLA ``memory_analysis`` is
    authoritative.
    """
    artifact: Dict[str, Any] = _load_json(_ARTIFACT_PATH)
    assert artifact["schema_version"] == 1
    assert artifact["schema"] == "org.diffpes.verification.kz-scalability.v1"
    assert artifact["requirements"] == [
        "kz-node-local-memory",
        "kz-rematerialization-and-photon-scan-memory",
    ]
    assert artifact["backend"] == "cpu"
    assert artifact["x64_enabled"] is True
    assert artifact["preallocate_environment"] == "false"

    literal: Dict[str, Any] = artifact["literal_kz_target"]
    assert literal["dimensions"] == {
        "energy_chunk": 16,
        "k_chunk": 32,
        "n_band": 20,
        "n_energy": 400,
        "n_k": 65_536,
        "n_kx": 256,
        "n_ky": 256,
        "n_kz": 2048,
    }
    assert literal["cube_shape"] == [256, 256, 400]
    assert literal["cube_bytes"] == 209_715_200
    assert literal["gradient_shape"] == [20, 20]
    assert literal["checkpoint"] is True
    assert literal["programs_executed"] is False
    assert literal["finite_execution"] is None
    assert literal["nonzero_full_h_gradient"] is None

    audit: Dict[str, Any] = literal["jaxpr_shape_audit"]
    assert "constvars" in audit["audit_scope"]
    assert audit["contains_forbidden_all_node_carrier"] is False
    for program_name in ("forward", "forward_and_full_h_gradient"):
        program_audit: Dict[str, Any] = audit[program_name]
        assert program_audit["contains_forbidden_all_node_carrier"] is False
        assert program_audit["result"] == "pass"
        assert program_audit["shape_matches"] == {
            "full_k_band_band_kz": [],
            "full_k_band_energy_kz": [],
            "full_k_energy_kz": [],
            "full_k_kz_cartesian": [],
        }
        assert (
            program_audit["constvar_shape_matches"]
            == (program_audit["shape_matches"])
        )
    assert audit["result"] == "pass"

    expected: Dict[str, Tuple[int, int, int, int, int]] = {
        "forward": (
            2_000_000_000,
            6400,
            209_715_200,
            865_148_448,
            1_074_870_048,
        ),
        "forward_and_full_h_gradient": (
            12_000_000_000,
            6400,
            6424,
            2_567_789_224,
            2_567_802_048,
        ),
    }
    program_name: str
    measurements: Tuple[int, int, int, int, int]
    budget: int
    arguments: int
    output: int
    temporary: int
    live: int
    for program_name, measurements in expected.items():
        budget, arguments, output, temporary, live = measurements
        program: Dict[str, Any] = literal[program_name]
        memory: Dict[str, Any] = program["memory_analysis"]
        assert program["budget_bytes"] == budget
        assert program["compilation_seconds"] > 0.0
        assert program["passes_budget"] is True
        assert memory == {
            "alias_size_bytes": 0,
            "argument_size_bytes": arguments,
            "authority_available": True,
            "compiler_live_allocation_bytes": live,
            "output_size_bytes": output,
            "result": "measured",
            "temporary_size_bytes": temporary,
        }
        assert arguments + output + temporary == live
        assert live <= budget
    assert literal["result"] == "pass"


def test_kz_scan_scalability_freezes_remat_and_photon_scan() -> None:
    """Require small remat equality and flat forward/reverse hnu overhead.

    Inspect exact nonzero-gradient and auxiliary-allocation measurements.

    Notes
    -----
    Subtract the scan output, photon input, gradient output, and scalar output
    before applying the 64 KiB auxiliary-memory flatness requirement.
    """
    artifact: Dict[str, Any] = _load_json(_ARTIFACT_PATH)
    remat: Dict[str, Any] = artifact["rematerialization_comparison"]
    assert remat["dimensions"] == [3, 5, 7, 3, 8, 5, 7]
    assert remat["value_rtol"] == 1.0e-12
    assert remat["value_atol"] == 1.0e-14
    assert remat["maximum_value_absolute_error"] == 0
    assert remat["value_passes"] is True
    assert remat["gradient_rtol"] == 1.0e-12
    assert remat["gradient_atol"] == 1.0e-14
    assert remat["maximum_gradient_absolute_error"] == (1.734723475976807e-18)
    assert remat["maximum_reference_gradient"] == 0.010309586568000314
    assert remat["gradient_passes"] is True
    assert remat["nonzero_full_h_gradient"] is True
    assert remat["result"] == "pass"

    scan: Dict[str, Any] = artifact["photon_scan_auxiliary_memory"]
    assert scan["lengths"] == [1, 5, 20]
    assert scan["flat_tolerance_bytes"] == 65_536
    assert scan["required_output_scaling"] == ("8*n_hv*n_k*n_energy bytes")
    expected: Dict[int, Tuple[int, int]] = {
        1: (42_672, 123_048),
        5: (44_080, 134_696),
        20: (44_208, 160_040),
    }
    measurement: Dict[str, Any]
    expected_forward: int
    expected_reverse: int
    for measurement in scan["measurements"]:
        n_hv: int = measurement["n_hv"]
        expected_forward, expected_reverse = expected[n_hv]
        assert measurement["output_shape"] == [n_hv, 15, 7]
        assert (
            measurement["forward"]["auxiliary_beyond_scan_and_hv_input_bytes"]
            == expected_forward
        )
        assert (
            measurement["reverse_hv_gradient"][
                "auxiliary_beyond_scan_hv_input_gradient_scalar_bytes"
            ]
            == expected_reverse
        )
        assert (
            measurement["forward"]["memory_analysis"]["authority_available"]
            is True
        )
        assert (
            measurement["reverse_hv_gradient"]["memory_analysis"][
                "authority_available"
            ]
            is True
        )
    assert scan["forward"] == {
        "authority_available": True,
        "auxiliary_range_bytes": 1536,
        "result": "pass",
        "temporary_range_bytes": 1536,
    }
    assert scan["reverse_hv_gradient"] == {
        "authority_available": True,
        "auxiliary_range_bytes": 36_992,
        "result": "pass",
        "temporary_range_bytes": 36_992,
    }
    assert scan["result"] == "pass"
    assert artifact["result"] == "pass"
