"""Validate scientific measurement results from the automation scripts.

The tests run each reduced experiment through its command interface. They
check recovered edge parameters, candidate selection, broadening, and Poisson
signal-to-noise scaling from the emitted metrics.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from beartype.typing import Any, Dict, List

pytestmark: List[pytest.MarkDecorator] = [
    pytest.mark.big_mem,
    pytest.mark.xdist_group("automatons_cpu"),
]


REPOSITORY_ROOT: Path = Path(__file__).resolve().parents[2]


def _run_measurement_script(
    script_name: str,
    tmp_path: Path,
) -> Dict[str, Any]:
    """PRIVATE: Run one reduced measurement script and parse its result.

    Parameters
    ----------
    script_name : str
        Direct automaton filename inside the repository catalog.
    tmp_path : Path
        Per-test artifact directory supplied by pytest.

    Returns
    -------
    payload : Dict[str, Any]
        Final JSON result object from the completed command.

    Notes
    -----
    Uses the active virtual-environment interpreter and CPU-only JAX. The
    helper also verifies the reported reduced execution remains bounded.
    """
    output_directory: Path = tmp_path / script_name.removesuffix(".py")
    matplotlib_directory: Path = Path("/tmp/dp-mpl")  # noqa: S108
    matplotlib_directory.mkdir(parents=True, exist_ok=True)
    environment: Dict[str, str] = os.environ.copy()
    environment["JAX_PLATFORMS"] = "cpu"
    environment["MPLCONFIGDIR"] = str(matplotlib_directory)
    command: List[str] = [
        sys.executable,
        str(REPOSITORY_ROOT / "automatons" / script_name),
        "--smoke",
        "--seed",
        "123",
        "--outdir",
        str(output_directory),
        "--json",
    ]
    completed: subprocess.CompletedProcess[str] = subprocess.run(  # noqa: S603
        command,
        capture_output=True,
        check=False,
        cwd=REPOSITORY_ROOT,
        env=environment,
        text=True,
        timeout=60.0,
    )
    assert completed.returncode == 0, (
        f"{script_name} failed:\n{completed.stderr}\n"
        f"{completed.stdout[-1000:]}"
    )
    lines: List[str] = [line for line in completed.stdout.splitlines() if line]
    assert lines
    decoded: Any = json.loads(lines[-1])
    assert isinstance(decoded, dict)
    payload: Dict[str, Any] = decoded
    assert payload["status"] == "ok"
    assert float(payload["wall_seconds"]) <= 60.0
    return payload


class TestMeasurementAutomatons:
    """Validate measurement-derived quantities from the experiment catalog.

    The case exercises reduced executions so each assertion follows the same
    command path available to an analyst.
    """

    def test_ingest_recovers_the_planted_fermi_edge(
        self,
        tmp_path: Path,
    ) -> None:
        """Recover the planted Fermi edge and temperature.

        The test compares fit metrics against the in-code graphene measurement.
        It uses tolerances that reflect the sampled energy resolution.

        Notes
        -----
        The reduced path keeps the optimizer deterministic for its fixed seed.
        """
        payload: Dict[str, Any] = _run_measurement_script(
            "arpes_ingest.py",
            tmp_path,
        )
        metrics: Dict[str, Any] = payload["metrics"]
        recovered_edge_ev: float = float(metrics["fermi_edge_ev"])
        planted_edge_ev: float = float(metrics["planted_fermi_edge_ev"])
        recovered_temperature_k: float = float(
            metrics["effective_temperature_k"]
        )
        planted_temperature_k: float = float(metrics["planted_temperature_k"])
        temperature_tolerance_k: float = 0.10 * planted_temperature_k
        assert abs(recovered_edge_ev - planted_edge_ev) <= 2.0e-3
        assert abs(recovered_temperature_k - planted_temperature_k) <= (
            temperature_tolerance_k
        )

    def test_matching_selects_the_planted_grid_cell(
        self,
        tmp_path: Path,
    ) -> None:
        """Select the planted linewidth and temperature grid cell.

        The test compares the winning candidate metrics against synthetic data.
        It confirms the ranking measures recover the known generating values.

        Notes
        -----
        The planted candidate exists in the default three-by-three grid.
        """
        payload: Dict[str, Any] = _run_measurement_script(
            "match_measured_to_simulated.py",
            tmp_path,
        )
        metrics: Dict[str, Any] = payload["metrics"]
        assert float(metrics["best_gamma_ev"]) == float(
            metrics["planted_gamma_ev"]
        )
        assert float(metrics["best_temperature_k"]) == float(
            metrics["planted_temperature_k"]
        )
        assert float(metrics["best_ncc"]) >= 0.999

    def test_resolution_broadening_increases_the_apparent_width(
        self,
        tmp_path: Path,
    ) -> None:
        """Verify apparent momentum broadening from paired resolution values.

        The test reads the reported width series from public convolution calls.
        It checks the monotonic flag and every adjacent numerical relation.

        Notes
        -----
        The Fermi-level momentum distribution receives both convolutions.
        """
        payload: Dict[str, Any] = _run_measurement_script(
            "resolution_sweep.py",
            tmp_path,
        )
        metrics: Dict[str, Any] = payload["metrics"]
        widths: List[float] = [
            float(value) for value in metrics["apparent_fwhm_inv_ang"]
        ]
        assert metrics["monotone_in_energy_resolution"] is True
        assert len(widths) == 3
        assert widths[0] < widths[1] < widths[2]

    def test_counting_statistics_follow_square_root_scaling(
        self,
        tmp_path: Path,
    ) -> None:
        """Measure the Poisson square-root signal-to-noise relation.

        The test fits the exposure trend after independent Poisson
        acquisitions.
        It accepts a narrow interval around the expected one-half exponent.

        Notes
        -----
        The observed total count series must rise across the exposure ladder.
        """
        payload: Dict[str, Any] = _run_measurement_script(
            "counting_statistics.py",
            tmp_path,
        )
        metrics: Dict[str, Any] = payload["metrics"]
        slope: float = float(metrics["snr_loglog_slope"])
        totals: List[float] = [
            float(value) for value in metrics["total_counts"]
        ]
        assert 0.45 <= slope <= 0.55
        assert len(totals) == 3
        assert totals[0] < totals[1] < totals[2]
