"""Validate deterministic outputs from forward simulation scripts.

The module runs every forward script twice with one fixed seed. It compares
the result identity, parameters, metrics, and artifact manifest paths.
"""

from __future__ import annotations

import json
import math
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
FORWARD_SCRIPT_NAMES: Tuple[str, ...] = (
    "forward_bands.py",
    "forward_spectral_cut.py",
    "forward_arpes_cube.py",
    "forward_detector_acquisition.py",
    "photon_energy_scan.py",
    "polarization_dichroism.py",
    "vasp_bands_to_arpes.py",
)


def _run_seeded_smoke(
    script_path: Path,
    output_directory: Path,
) -> Dict[str, Any]:
    """PRIVATE: Run one forward script with a deterministic smoke command.

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
            "MPLCONFIGDIR": "/tmp/dp-mpl",  # noqa: S108
        },
        text=True,
        timeout=120.0,
    )
    stdout_lines: List[str] = [
        line for line in completed.stdout.splitlines() if line
    ]
    payload: Any = json.loads(stdout_lines[-1]) if stdout_lines else {}

    assert completed.returncode == 0, completed.stderr
    assert isinstance(payload, dict)
    result: Dict[str, Any] = payload
    return result


class TestForwardReproducibility:
    """Validate seeded behavior for every forward experiment script.

    The case compares two isolated smoke results for each catalogued script.
    It preserves one artifact root per result to avoid file reuse.
    """

    @pytest.mark.parametrize("script_name", FORWARD_SCRIPT_NAMES)
    def test_fixed_seed_repeats_result_content(
        self,
        script_name: str,
        tmp_path: Path,
    ) -> None:
        """Require equal seeded parameters, metrics, paths, and identity.

        The test verifies deterministic output at the forward process boundary.
        It also enforces the CPU smoke ceiling for each completed command.

        Notes
        -----
        Runs each executable twice in separate temporary directories. It parses
        the final JSON lines and compares the documented output fields.
        """
        script_path: Path = AUTOMATON_DIRECTORY / script_name
        first_directory: Path = tmp_path / "first"
        second_directory: Path = tmp_path / "second"

        if not script_path.is_file():
            pytest.skip(f"{script_name} is not present yet")

        first_result: Dict[str, Any] = _run_seeded_smoke(
            script_path,
            first_directory,
        )
        second_result: Dict[str, Any] = _run_seeded_smoke(
            script_path,
            second_directory,
        )
        first_paths: List[str] = [
            artifact["path"] for artifact in first_result["artifacts"]
        ]
        second_paths: List[str] = [
            artifact["path"] for artifact in second_result["artifacts"]
        ]

        assert first_result["status"] == "ok"
        assert second_result["status"] == "ok"
        assert first_result["wall_seconds"] <= 60.0
        assert second_result["wall_seconds"] <= 60.0
        assert first_result["params"] == second_result["params"]
        assert first_result["metrics"] == second_result["metrics"]
        assert first_paths == second_paths
        assert first_result["result_key"] == second_result["result_key"]


class TestIntrinsicForwardMetrics:
    """Validate physical summary values from intrinsic forward scripts.

    The class runs each compact intrinsic executable and checks dispersion,
    spectral-cut, or cube metrics against observable-specific bounds.
    """

    @pytest.mark.parametrize(
        "script_name",
        [
            "forward_bands.py",
            "forward_spectral_cut.py",
            "forward_arpes_cube.py",
        ],
    )
    def test_smoke_metrics_are_physical(
        self,
        script_name: str,
        tmp_path: Path,
    ) -> None:
        """Require finite, positive intensity or dispersion observables.

        Parameters
        ----------
        script_name : str
            Name of the intrinsic executable under test.
        tmp_path : Path
            Temporary location for the smoke artifacts.

        Notes
        -----
        Each branch checks quantities appropriate to its observable rather
        than comparing an incidental pixel or array element.
        """
        script_path: Path = AUTOMATON_DIRECTORY / script_name
        result: Dict[str, Any] = _run_seeded_smoke(script_path, tmp_path)
        metrics: Dict[str, Any] = result["metrics"]

        assert result["status"] == "ok"
        assert result["wall_seconds"] <= 60.0
        if script_name == "forward_bands.py":
            assert metrics["n_bands"] >= 2
            assert metrics["bandwidth_ev"] > 0.0
            assert metrics["min_direct_gap_ev"] >= 0.0
        elif script_name == "forward_spectral_cut.py":
            assert metrics["max_intensity"] > 0.0
            assert metrics["integrated_intensity"] > 0.0
            assert metrics["mdc_fwhm_inv_ang_at_ef"] >= 0.0
            assert -0.5 <= metrics["edc_peak_ev_at_k_index"] <= 0.5
        else:
            assert metrics["cube_shape"] == [16, 16, 32]
            assert metrics["max_intensity"] > 0.0
            assert 0.0 < metrics["fermi_map_fraction_above_half_max"] <= 1.0


class TestExperimentalForwardMetrics:
    """Validate physically bounded summaries from experimental forward scripts.

    The class runs each compact executable and checks output metrics against
    count, momentum, asymmetry, or VASP-carrier observable constraints.
    """

    @pytest.mark.parametrize(
        "script_name",
        [
            "forward_detector_acquisition.py",
            "photon_energy_scan.py",
            "polarization_dichroism.py",
            "vasp_bands_to_arpes.py",
        ],
    )
    def test_smoke_metrics_respect_observable_constraints(
        self,
        script_name: str,
        tmp_path: Path,
    ) -> None:
        """Require finite values and bounds intrinsic to each experiment.

        Parameters
        ----------
        script_name : str
            Name of the experimental executable under test.
        tmp_path : Path
            Temporary location for the smoke artifacts.

        Notes
        -----
        The assertions avoid incidental raster values. They instead check
        nonnegative counts, ordered vertical momenta, bounded dichroism, and
        the finite VASP-carrier summaries produced by the public APIs.
        """
        script_path: Path = AUTOMATON_DIRECTORY / script_name
        result: Dict[str, Any] = _run_seeded_smoke(script_path, tmp_path)
        metrics: Dict[str, Any] = result["metrics"]

        assert result["status"] == "ok"
        assert result["wall_seconds"] <= 60.0
        if script_name == "forward_detector_acquisition.py":
            assert metrics["total_counts"] > 0
            assert 0 <= metrics["max_counts"] <= metrics["total_counts"]
            assert math.isfinite(metrics["poisson_chi2_per_dof"])
            assert metrics["poisson_chi2_per_dof"] >= 0.0
            assert metrics["mean_expected_counts"] > 0.0
        elif script_name == "photon_energy_scan.py":
            assert metrics["n_hv"] == 6
            assert math.isfinite(metrics["kz_min_inv_ang"])
            assert math.isfinite(metrics["kz_max_inv_ang"])
            assert metrics["kz_min_inv_ang"] <= metrics["kz_max_inv_ang"]
            assert metrics["intensity_periodicity_ev"] >= 0.0
        elif script_name == "polarization_dichroism.py":
            assert 0.0 <= metrics["max_abs_asymmetry"] <= 1.0 + 1.0e-12
            assert (
                abs(metrics["mean_asymmetry"]) <= metrics["max_abs_asymmetry"]
            )
            assert metrics["sign_change_count"] >= 0
        else:
            assert metrics["n_bands"] >= 1
            assert metrics["n_kpoints"] >= 2
            assert math.isfinite(metrics["fermi_energy_ev"])
            assert math.isfinite(metrics["max_intensity"])
            assert metrics["max_intensity"] > 0.0
