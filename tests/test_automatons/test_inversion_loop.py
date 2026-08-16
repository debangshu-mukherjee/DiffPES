"""Validate deterministic planted recoveries from inversion scripts.

The module runs every inversion script twice with a fixed seed. It checks the
reported recovery, derivative evidence, runtime ceiling, and result identity.
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
INVERSION_SCRIPT_NAMES: Tuple[str, ...] = (
    "fit_hopping_parameters.py",
    "fit_self_energy.py",
    "fit_experiment_geometry.py",
)


def _run_seeded_smoke(
    script_path: Path,
    output_directory: Path,
) -> Dict[str, Any]:
    """PRIVATE: Run one inversion script with a deterministic smoke command.

    Parameters
    ----------
    script_path : Path
        Absolute path to the executable experiment file.
    output_directory : Path
        Dedicated artifact directory for the command.

    Returns
    -------
    result : Dict[str, Any]
        Parsed JSON result from the final standard-output line.

    Notes
    -----
    Uses the project interpreter and CPU environment. Captures both streams
    so a failed command reports its diagnostics through the test assertion.
    """
    matplotlib_directory: Path = Path("/tmp/dp-mpl")  # noqa: S108
    matplotlib_directory.mkdir(parents=True, exist_ok=True)
    environment: Dict[str, str] = os.environ.copy()
    command: Tuple[str, ...] = (
        str(PYTHON_EXECUTABLE),
        str(script_path),
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
        env={
            **environment,
            "JAX_PLATFORMS": "cpu",
            "MPLCONFIGDIR": str(matplotlib_directory),
        },
        text=True,
        timeout=90.0,
    )
    stdout_lines: List[str] = [
        line for line in completed.stdout.splitlines() if line
    ]
    decoded: Any = json.loads(stdout_lines[-1]) if stdout_lines else {}

    assert completed.returncode == 0, completed.stderr
    assert isinstance(decoded, dict)
    result: Dict[str, Any] = decoded
    return result


class TestInversionRecoveries:
    """Validate planted inversion recovery and fixed-seed reproducibility.

    The case compares isolated smoke results for every catalogued inversion
    script and checks their independent recovery tolerances.
    """

    @pytest.mark.parametrize("script_name", INVERSION_SCRIPT_NAMES)
    def test_fixed_seed_recovers_the_planted_parameters(
        self,
        script_name: str,
        tmp_path: Path,
    ) -> None:
        """Recover planted parameters with finite, sensitive residual maps.

        The test runs each compact fit twice with the same seed and compares
        result content before enforcing the reported recovery tolerances.

        Notes
        -----
        The derivative checks compare automatic Jacobians against central
        finite differences inside each executable experiment.
        """
        script_path: Path = AUTOMATON_DIRECTORY / script_name
        first_directory: Path = tmp_path / "first"
        second_directory: Path = tmp_path / "second"
        first_result: Dict[str, Any] = _run_seeded_smoke(
            script_path,
            first_directory,
        )
        second_result: Dict[str, Any] = _run_seeded_smoke(
            script_path,
            second_directory,
        )
        first_metrics: Dict[str, Any] = first_result["metrics"]
        second_metrics: Dict[str, Any] = second_result["metrics"]
        first_paths: List[str] = [
            artifact["path"] for artifact in first_result["artifacts"]
        ]
        second_paths: List[str] = [
            artifact["path"] for artifact in second_result["artifacts"]
        ]

        assert first_result["status"] == "ok"
        assert second_result["status"] == "ok"
        assert float(first_result["wall_seconds"]) < 60.0
        assert float(second_result["wall_seconds"]) < 60.0
        assert first_result["params"] == second_result["params"]
        assert first_metrics == second_metrics
        assert first_paths == second_paths
        assert first_result["result_key"] == second_result["result_key"]
        assert first_metrics["converged"] is True
        assert first_metrics["jacobian_finite"] is True
        assert float(first_metrics["jacobian_min_column_norm"]) > 0.0
        assert float(first_metrics["jacobian_fd_relative_error"]) <= 1.0e-6
        if script_name == "fit_hopping_parameters.py":
            assert float(first_metrics["hopping_rel_error"]) < 1.0e-6
        elif script_name == "fit_self_energy.py":
            assert float(first_metrics["coefficient_rel_error"]) < 1.0e-6
        else:
            assert float(first_metrics["angle_abs_error_rad"]) < 1.0e-5
            assert float(first_metrics["scale_rel_error"]) < 1.0e-6
