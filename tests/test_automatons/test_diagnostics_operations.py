"""Validate diagnostic and operational experiment results.

The tests run the compact diagnostics through their public command interface.
They compare fixed-seed results, scientific assertions, host-device agreement,
certificate reproduction, and the available export evidence.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest
from beartype.typing import Any, Dict, List, Tuple

pytestmark: List[pytest.MarkDecorator] = [
    pytest.mark.big_mem,
    pytest.mark.xdist_group("automatons_cpu"),
]


REPOSITORY_ROOT: Path = Path(__file__).resolve().parents[2]
AUTOMATON_DIRECTORY: Path = REPOSITORY_ROOT / "automatons"
PYTHON_EXECUTABLE: Path = REPOSITORY_ROOT / ".venv" / "bin" / "python"
DIAGNOSTIC_SCRIPTS: Tuple[str, ...] = (
    "convergence_study.py",
    "parameter_grid.py",
    "certify_forward.py",
    "export_model.py",
)
EXPECTED_ROLES: Dict[str, Tuple[str, ...]] = {
    "convergence_study.py": (
        "convergence_curves",
        "convergence_arrays",
        "metrics",
    ),
    "parameter_grid.py": (
        "grid_heatmap",
        "grid_arrays",
        "grid_table",
        "metrics",
    ),
    "certify_forward.py": (
        "certificate",
        "certificate_report",
        "metrics",
    ),
    "export_model.py": (
        "stablehlo_artifact",
        "export_manifest",
        "export_arrays",
        "metrics",
    ),
}


def _run_smoke(
    script_name: str,
    output_directory: Path,
    extra_environment: Tuple[Tuple[str, str], ...] = (),
) -> Dict[str, Any]:
    """PRIVATE: Run one fixed-seed compact command and parse its result.

    Parameters
    ----------
    script_name : str
        Direct executable filename in the automaton directory.
    output_directory : Path
        Dedicated temporary artifact root for this process.
    extra_environment : Tuple[Tuple[str, str], ...]
        Additional environment pairs for the isolated subprocess.

    Returns
    -------
    payload : Dict[str, Any]
        Parsed final JSON payload from the completed command.

    Notes
    -----
    Uses the repository virtual environment, CPU-only JAX, and a writable
    temporary Matplotlib configuration directory for every subprocess.
    """
    matplotlib_directory: Path = Path("/tmp/dp-mpl")  # noqa: S108
    matplotlib_directory.mkdir(parents=True, exist_ok=True)
    environment: Dict[str, str] = os.environ.copy()
    environment["JAX_PLATFORMS"] = "cpu"
    environment["MPLCONFIGDIR"] = str(matplotlib_directory)
    name: str
    value: str
    for name, value in extra_environment:
        environment[name] = value
    command: Tuple[str, ...] = (
        str(PYTHON_EXECUTABLE),
        str(AUTOMATON_DIRECTORY / script_name),
        "--smoke",
        "--seed",
        "123",
        "--outdir",
        str(output_directory),
        "--json",
    )
    completed: subprocess.CompletedProcess[str] = subprocess.run(  # noqa: S603
        list(command),
        capture_output=True,
        check=False,
        cwd=REPOSITORY_ROOT,
        env=environment,
        text=True,
        timeout=60.0,
    )
    lines: List[str] = [line for line in completed.stdout.splitlines() if line]
    decoded: Any = json.loads(lines[-1]) if lines else {}

    assert completed.returncode == 0, completed.stderr
    assert isinstance(decoded, dict)
    payload: Dict[str, Any] = decoded
    return payload


class TestDiagnosticOperations:
    """Validate repeated diagnostic executions and their scientific summaries.

    Each case uses independent output roots so fixed-seed comparisons traverse
    the same process boundary that an external caller receives.
    """

    @pytest.mark.parametrize("script_name", DIAGNOSTIC_SCRIPTS)
    def test_fixed_seed_results_and_scientific_assertions(
        self,
        script_name: str,
        tmp_path: Path,
    ) -> None:
        """Compare two fixed-seed results and assert their observed properties.

        The test requires stable parameters, metrics, artifact paths, and
        result identity before applying the script-specific numerical checks.

        Notes
        -----
        Reduced inputs keep every subprocess below the documented CPU ceiling.
        """
        first_payload: Dict[str, Any] = _run_smoke(
            script_name,
            tmp_path / f"{script_name}-first",
        )
        second_payload: Dict[str, Any] = _run_smoke(
            script_name,
            tmp_path / f"{script_name}-second",
        )
        first_metrics: Dict[str, Any] = first_payload["metrics"]
        second_metrics: Dict[str, Any] = second_payload["metrics"]
        first_roles: List[str] = [
            str(artifact["role"]) for artifact in first_payload["artifacts"]
        ]
        second_roles: List[str] = [
            str(artifact["role"]) for artifact in second_payload["artifacts"]
        ]
        first_paths: List[str] = [
            str(artifact["path"]) for artifact in first_payload["artifacts"]
        ]
        second_paths: List[str] = [
            str(artifact["path"]) for artifact in second_payload["artifacts"]
        ]

        assert first_payload["status"] == "ok"
        assert second_payload["status"] == "ok"
        assert float(first_payload["wall_seconds"]) <= 60.0
        assert float(second_payload["wall_seconds"]) <= 60.0
        assert first_payload["params"] == second_payload["params"]
        assert first_metrics == second_metrics
        assert first_paths == second_paths
        assert first_payload["result_key"] == second_payload["result_key"]
        assert set(EXPECTED_ROLES[script_name]) <= set(first_roles)
        assert first_roles == second_roles

        if script_name == "convergence_study.py":
            residuals: Dict[str, List[float]] = first_metrics["residuals"]
            assert first_metrics["monotone"] is True
            assert int(first_metrics["converged_level"]) == 2
            assert all(
                values[index] >= values[index + 1]
                for values in residuals.values()
                for index in range(len(values) - 1)
            )
        elif script_name == "parameter_grid.py":
            assert int(first_metrics["device_count"]) == 1
            assert float(first_metrics["sharded_max_abs_error"]) <= 1.0e-12
        elif script_name == "certify_forward.py":
            assert first_metrics["verified"] is True
            assert float(first_metrics["reproduction_max_abs_error"]) == 0.0
            assert first_metrics["certificate_sha256"]
        else:
            assert float(first_metrics["same_result_max_abs_error"]) < 1.0e-10
            sizes: List[List[int]] = first_metrics["sizes_verified"]
            assert len(sizes) == 2
            if first_metrics["portable_serialization"] is True:
                assert first_metrics["separate_process_ok"] is True
            else:
                assert first_metrics["separate_process_ok"] is False

    def test_host_device_grid_matches_the_vectorized_result(
        self,
        tmp_path: Path,
    ) -> None:
        """Match an eight-device parameter grid to its ordinary vectorization.

        The command internally evaluates both paths before exposing their
        maximum absolute difference and the selected host-device count.

        Notes
        -----
        Set the XLA flag in a fresh subprocess, then preserve the ordinary
        single-device CPU configuration for every other test.
        """
        payload: Dict[str, Any] = _run_smoke(
            "parameter_grid.py",
            tmp_path / "parameter-grid-eight-device",
            (("XLA_FLAGS", "--xla_force_host_platform_device_count=8"),),
        )
        metrics: Dict[str, Any] = payload["metrics"]

        assert payload["status"] == "ok"
        assert int(metrics["device_count"]) == 8
        assert float(metrics["sharded_max_abs_error"]) <= 1.0e-12
