"""Validate local information, acquisition ranking, and derivative evidence.

The module runs the three identifiability experiment files in isolated paths.
It checks their deterministic scientific summaries and their failure controls.
"""

from __future__ import annotations

import json
import os
import subprocess
from importlib.machinery import ModuleSpec
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import ModuleType

import jax.numpy as jnp
import pytest
from beartype.typing import Any, Dict, List, Tuple
from jaxtyping import Array, Float64

pytestmark: List[pytest.MarkDecorator] = [
    pytest.mark.big_mem,
    pytest.mark.xdist_group("automatons_cpu"),
]


REPOSITORY_ROOT: Path = Path(__file__).resolve().parents[2]
AUTOMATON_DIRECTORY: Path = REPOSITORY_ROOT / "automatons"
PYTHON_EXECUTABLE: Path = REPOSITORY_ROOT / ".venv" / "bin" / "python"


def _run_seeded_smoke(
    script_path: Path,
    output_directory: Path,
    extra_args: Tuple[str, ...] = (),
) -> Dict[str, Any]:
    """PRIVATE: Run one identifiability script with a deterministic command.

    Parameters
    ----------
    script_path : Path
        Absolute path to the executable experiment file.
    output_directory : Path
        Dedicated artifact directory for the command.
    extra_args : Tuple[str, ...]
        Additional inherited or declared command-line arguments.

    Returns
    -------
    result : Dict[str, Any]
        Parsed JSON result from the final standard-output line.

    Notes
    -----
    Uses the project interpreter with CPU-only JAX and a temporary Matplotlib
    configuration directory. Captured diagnostics remain available on failure.
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
        *extra_args,
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


def _load_automaton_module(script_name: str) -> ModuleType:
    """PRIVATE: Load one executable experiment module for a failure probe.

    Parameters
    ----------
    script_name : str
        Filename of the executable experiment module.

    Returns
    -------
    module : ModuleType
        Loaded module object with its local diagnostic helper.

    Raises
    ------
    ImportError
        If Python cannot create a module specification for the file.

    Notes
    -----
    Loads the file through a temporary module name. The probe reuses the
    executable helper without adding a public package import surface.
    """
    script_path: Path = AUTOMATON_DIRECTORY / script_name
    specification: ModuleSpec | None = spec_from_file_location(
        "identifiability_probe",
        script_path,
    )
    if specification is None or specification.loader is None:
        message: str = "cannot load the executable experiment module"
        raise ImportError(message)
    module: ModuleType = module_from_spec(specification)
    specification.loader.exec_module(module)
    result: ModuleType = module
    return result


class TestInformationSpectrum:
    """Validate the information-spectrum executable experiment.

    The case covers rank, gauge-null evidence, reproducibility, and smoke time.
    """

    def test_scale_gauge_is_null_and_rank_is_reduced(
        self,
        tmp_path: Path,
    ) -> None:
        """Require a null intensity scale and a reduced effective rank.

        The normalized spectral map removes the overall-scale coordinate. The
        resulting active rank cannot exceed the non-gauge coordinate count.

        Notes
        -----
        Runs two fixed-seed smoke commands in separate artifact roots. Compares
        their summaries and enforces the analytic scale-null tolerance.
        """
        script_path: Path = AUTOMATON_DIRECTORY / "information_spectrum.py"
        first_result: Dict[str, Any] = _run_seeded_smoke(
            script_path,
            tmp_path / "information-first",
        )
        second_result: Dict[str, Any] = _run_seeded_smoke(
            script_path,
            tmp_path / "information-second",
        )
        first_metrics: Dict[str, Any] = first_result["metrics"]
        second_metrics: Dict[str, Any] = second_result["metrics"]

        assert first_result["status"] == "ok"
        assert second_result["status"] == "ok"
        assert first_result["params"] == second_result["params"]
        assert first_metrics == second_metrics
        assert first_result["result_key"] == second_result["result_key"]
        assert float(first_result["wall_seconds"]) <= 60.0
        assert float(second_result["wall_seconds"]) <= 60.0
        assert float(first_metrics["gauge_nullspace_max_residual"]) < 1.0e-8
        assert int(first_metrics["effective_rank"]) <= (
            int(first_metrics["n_parameters"]) - int(first_metrics["n_gauge"])
        )


class TestExperimentDesignCompare:
    """Validate deterministic ranking of compact acquisition candidates.

    The case compares repeated process results and ordered information scores.
    """

    def test_larger_equal_noise_energy_range_ranks_no_lower(
        self,
        tmp_path: Path,
    ) -> None:
        """Require a larger equal-noise photon range above smaller ranges.

        The default candidates share polarization, temperature, and resolution.
        Their increasing photon energies represent increasing covered ranges.

        Notes
        -----
        Runs two fixed-seed smoke commands and compares their result identity.
        Checks strict information growth and the selected final candidate.
        """
        script_path: Path = (
            AUTOMATON_DIRECTORY / "experiment_design_compare.py"
        )
        first_result: Dict[str, Any] = _run_seeded_smoke(
            script_path,
            tmp_path / "design-first",
        )
        second_result: Dict[str, Any] = _run_seeded_smoke(
            script_path,
            tmp_path / "design-second",
        )
        first_metrics: Dict[str, Any] = first_result["metrics"]
        second_metrics: Dict[str, Any] = second_result["metrics"]
        logdets: List[float] = [
            float(value) for value in first_metrics["logdet_information"]
        ]
        traces: List[float] = [
            float(value) for value in first_metrics["crb_trace"]
        ]

        assert first_result["status"] == "ok"
        assert second_result["status"] == "ok"
        assert first_result["params"] == second_result["params"]
        assert first_metrics == second_metrics
        assert first_result["result_key"] == second_result["result_key"]
        assert float(first_result["wall_seconds"]) <= 60.0
        assert float(second_result["wall_seconds"]) <= 60.0
        assert logdets[0] < logdets[1] < logdets[2]
        assert traces[0] > traces[1] > traces[2]
        assert int(first_metrics["best_design_index"]) == 2


class TestDerivativeAudit:
    """Validate finite-difference evidence and the zero-column tripwire.

    The case checks a sensitive reference map and a planted unused coordinate.
    """

    def test_reference_map_passes_with_failure_control_enabled(
        self,
        tmp_path: Path,
    ) -> None:
        """Pass the reference derivative audit when every coordinate matters.

        The reference chain spectrum depends on hopping and linewidth. The
        failure control remains enabled to make a silent defect process-fatal.

        Notes
        -----
        Runs two isolated seeded commands with the failure control. Compares
        their results and verifies finite-difference and column metrics.
        """
        script_path: Path = AUTOMATON_DIRECTORY / "audit_derivatives.py"
        extra_args: Tuple[str, ...] = ("--fail-on-violation",)
        first_result: Dict[str, Any] = _run_seeded_smoke(
            script_path,
            tmp_path / "audit-first",
            extra_args,
        )
        second_result: Dict[str, Any] = _run_seeded_smoke(
            script_path,
            tmp_path / "audit-second",
            extra_args,
        )
        first_metrics: Dict[str, Any] = first_result["metrics"]
        second_metrics: Dict[str, Any] = second_result["metrics"]

        assert first_result["status"] == "ok"
        assert second_result["status"] == "ok"
        assert first_result["params"] == second_result["params"]
        assert first_metrics == second_metrics
        assert first_result["result_key"] == second_result["result_key"]
        assert float(first_result["wall_seconds"]) <= 60.0
        assert float(second_result["wall_seconds"]) <= 60.0
        assert first_metrics["all_passed"] is True
        assert int(first_metrics["n_passed"]) == int(
            first_metrics["n_parameters"]
        )
        assert int(first_metrics["zero_column_count"]) == 0
        assert float(first_metrics["max_relative_error"]) <= 1.0e-6

    def test_unused_coordinate_raises_with_failure_control(self) -> None:
        """Raise for a planted coordinate that does not affect the output.

        The selected forward map reads only its first coordinate. The local
        audit must count the unused second column and raise when requested.

        Notes
        -----
        Loads the executable's local audit helper. Passes a two-coordinate map
        with a constant second derivative and requires its documented failure.
        """
        module: ModuleType = _load_automaton_module("audit_derivatives.py")
        evaluator: Any = module._evaluate_derivatives  # noqa: SLF001
        unused_parameters: Float64[Array, " 2"] = jnp.asarray(
            (1.0, 2.0),
            dtype=jnp.float64,
        )

        def unused_forward(
            parameters: Float64[Array, " 2"],
        ) -> Float64[Array, " 4"]:
            """Build a planted output that omits one coordinate.

            The forward map repeats its first coordinate into four samples. It
            deliberately ignores the second coordinate for the tripwire.
            """
            output: Float64[Array, " 4"] = jnp.full(
                (4,),
                parameters[0],
                dtype=jnp.float64,
            )
            return output

        with pytest.raises(RuntimeError, match="failed coordinate condition"):
            evaluator(
                unused_forward,
                unused_parameters,
                1.0e-5,
                1.0e-6,
                True,
            )
