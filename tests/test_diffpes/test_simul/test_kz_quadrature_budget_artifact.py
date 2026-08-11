"""Bind and inspect the inert Plan-08b quadrature budget.

Keep the independent NumPy/SciPy generator outside the pytest boundary.
Consume only its frozen JSON mirror in these tests.
"""

import hashlib
import json
import math
from pathlib import Path

from beartype.typing import Any, Dict, List

_REPOSITORY_ROOT: Path = Path(__file__).resolve().parents[3]
_REFERENCE_DIRECTORY: Path = (
    _REPOSITORY_ROOT / "tests" / "test_diffpes" / "_reference_data"
)
_ARTIFACT_PATH: Path = _REFERENCE_DIRECTORY / "08b_kz_quadrature_budget.json"
_MANIFEST_PATH: Path = (
    _REFERENCE_DIRECTORY / "08b_kz_quadrature_budget_manifest.json"
)
_ARTIFACT_SHA256: str = (
    "399160eea90aba56b9c289680e157d24f714464ab7798a4cfba6ba800e364e82"
)
_MANIFEST_SHA256: str = (
    "9da398556e50349c20ad38f43232c480ec4bd6778ea05bcef96265e2f93a468c"
)
_GENERATOR_SHA256: str = (
    "34f6c52813b2afd045f21f0e5078f9c665fcaf3730adaca5c4ae24ec5afea626"
)
_GENERATOR_AUTHORITY: str = (
    "diffpes-plans/verification/generate_08b_kz_quadrature_budget.py"
)
_PLANS_ARTIFACT_AUTHORITY: str = (
    "diffpes-plans/verification/08b_kz_quadrature_budget.json"
)


def _sha256(path: Path) -> str:
    """PRIVATE: Calculate the SHA-256 identity of one complete file.

    Parameters
    ----------
    path : Path
        File whose complete bytes define the identity.

    Returns
    -------
    digest : str
        Lowercase hexadecimal SHA-256 identity.
    """
    digest: str = hashlib.sha256(path.read_bytes()).hexdigest()
    return digest


def _load_json(path: Path) -> Dict[str, Any]:
    """PRIVATE: Load one authenticated inert JSON object.

    Parameters
    ----------
    path : Path
        JSON file to parse.

    Returns
    -------
    result : Dict[str, Any]
        Parsed top-level JSON object.
    """
    value: Any = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    result: Dict[str, Any] = value
    return result


def test_kz_quadrature_budget_has_authenticated_external_authority() -> None:
    """Bind the inert mirror to artifact and generator identities.

    Require literal digests and paths for both independent authorities.

    Notes
    -----
    Hash the consumed bytes before parsing their authentication records.
    """
    assert _sha256(_ARTIFACT_PATH) == _ARTIFACT_SHA256
    assert _sha256(_MANIFEST_PATH) == _MANIFEST_SHA256

    manifest: Dict[str, Any] = _load_json(_MANIFEST_PATH)
    assert manifest["schema_version"] == 1
    assert manifest["artifact"] == {
        "path_from_diffpes_root": (
            "tests/test_diffpes/_reference_data/08b_kz_quadrature_budget.json"
        ),
        "sha256": _ARTIFACT_SHA256,
    }
    assert manifest["plans_artifact"] == {
        "path": _PLANS_ARTIFACT_AUTHORITY,
        "sha256": _ARTIFACT_SHA256,
    }
    assert manifest["generator"] == {
        "mutable_authority_count": 1,
        "path": _GENERATOR_AUTHORITY,
        "sha256": _GENERATOR_SHA256,
    }

    artifact: Dict[str, Any] = _load_json(_ARTIFACT_PATH)
    assert artifact["generator"]["path"] == _GENERATOR_AUTHORITY
    assert artifact["generator"]["sha256"] == _GENERATOR_SHA256


def test_kz_quadrature_budget_freezes_the_preregistered_selection() -> None:
    """Require the full G6 profile and measured 2048-node selection.

    Keep every lower candidate red and pin the immediate predecessor's errors.

    Notes
    -----
    Parse the frozen table and compare each preregistered selection field.
    """
    artifact: Dict[str, Any] = _load_json(_ARTIFACT_PATH)
    assert artifact["schema_version"] == 1
    assert artifact["artifact_id"] == "plan08b-kz-quadrature-budget-v1"
    assert artifact["status"] == "green"
    assert artifact["candidate_n_kz"] == [32, 64, 128, 256, 512, 1024, 2048]
    assert artifact["reference_n_kz"] == 4096
    assert artifact["selected_n_kz"] == 2048
    assert artifact["thresholds"] == {
        "integrated_count_relative_error": 1.0e-6,
        "max_gradient_relative_error": 1.0e-4,
        "max_scaled_pointwise_or_count_error": 1.0e-5,
        "reference_uniform_remainder": 1.0e-12,
        "relative_error_floor": 1.0e-14,
    }

    profile: Dict[str, Any] = artifact["profile"]
    cases: List[Dict[str, Any]] = profile["cases"]
    assert len(cases) == 12
    assert {case["cell_id"] for case in cases} == {"cubic", "oblique"}
    assert {case["mean_free_path_ang"] for case in cases} == {
        5.0,
        10.0,
        50.0,
    }
    assert profile["parameters_with_gradients"] == [
        "lambda",
        "V0",
        "hnu",
        "W",
        "omega",
    ]
    assert all(case["kz_center_inv_ang"] > 0.0 for case in cases)
    assert all(case["gamma_u"] > 0.0 for case in cases)

    results: Dict[int, Dict[str, Any]] = {
        result["n_kz"]: result for result in artifact["candidate_results"]
    }
    assert list(results) == [32, 64, 128, 256, 512, 1024, 2048]
    assert not any(
        results[node]["passes_all_budgets"] for node in results if node < 2048
    )
    assert results[2048]["passes_all_budgets"]

    # The immediate predecessor misses all three budgets.  Pinning these
    # values makes it impossible to obscure the discriminating selection by
    # reporting only the first green row.
    predecessor: Dict[str, Any] = results[1024]
    assert (
        predecessor["versus_doubled"]["max_scaled_pointwise_or_count_error"]
        == 1.3616218626885728e-5
    )
    assert (
        predecessor["versus_doubled"]["integrated_count_relative_error"]
        == 2.3135226506392403e-6
    )
    assert (
        predecessor["versus_doubled"]["max_gradient_relative_error"]
        == 1.6750607640281796e-4
    )
    assert (
        predecessor["versus_reference_4096"][
            "max_scaled_pointwise_or_count_error"
        ]
        == 1.7020310322071123e-5
    )
    assert (
        predecessor["versus_reference_4096"]["integrated_count_relative_error"]
        == 2.8919083113911932e-6
    )
    assert (
        predecessor["versus_reference_4096"]["max_gradient_relative_error"]
        == 2.0938538919477642e-4
    )

    selected: Dict[str, Any] = results[2048]
    comparison_name: str
    for comparison_name in ("versus_doubled", "versus_reference_4096"):
        comparison: Dict[str, Any] = selected[comparison_name]
        assert (
            comparison["max_scaled_pointwise_or_count_error"]
            == 3.404045344959762e-6
        )
        assert (
            comparison["integrated_count_relative_error"]
            == 5.783869988633758e-7
        )
        assert (
            comparison["max_gradient_relative_error"] == 4.1879118882321295e-5
        )


def test_kz_quadrature_budget_covers_reference_and_geometry_truth() -> None:
    """Require wrapped values, analytic gradients, and the oblique trap.

    Check reference remainders, cell identities, and conserved bin masses.

    Notes
    -----
    Inspect the inert numeric records without executing their generator.
    """
    artifact: Dict[str, Any] = _load_json(_ARTIFACT_PATH)
    wrapped: Dict[str, Any] = artifact["wrapped_reference"]
    assert wrapped["max_uniform_remainder_bound"] == 2.487665750739011e-14
    assert wrapped["max_uniform_remainder_bound"] <= 1.0e-12
    assert wrapped["max_cauchy_analytic_fourier_residual"] <= 2.4e-14
    assert len(wrapped["wrapped_cauchy"]) == 4
    assert len(wrapped["wrapped_voigt"]) == 4
    record: Dict[str, Any]
    for record in wrapped["wrapped_cauchy"]:
        assert len(record["analytic_values"]) == 5
        assert len(record["fourier_values"]) == 5
        assert record["uniform_remainder_bound"] <= 1.0e-12
        assert all(
            math.isfinite(value) and value > 0.0
            for value in record["analytic_values"]
        )
    for record in wrapped["wrapped_voigt"]:
        assert len(record["fourier_values"]) == 5
        assert record["uniform_remainder_bound"] <= 1.0e-12
        assert all(
            math.isfinite(value) and value > 0.0
            for value in record["fourier_values"]
        )

    geometry: Dict[str, Dict[str, Any]] = {
        record["cell_id"]: record for record in artifact["profile"]["geometry"]
    }
    assert geometry["oblique"]["direct_cell_ang"] == [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.4, 0.2, 2.0],
    ]
    assert geometry["cubic"]["oblique_naive_center_error"] == 0.0
    assert geometry["oblique"]["oblique_naive_center_error"] > 0.0
    for record in geometry.values():
        assert record["unit_plane_advance"] == 1
        assert record["max_cartesian_roundtrip_error"] <= 2.0e-15
        assert record["max_third_fractional_coordinate_error"] <= 2.0e-15

    values_key: str
    for values_key in ("selected_values", "reference_4096_values"):
        values: Dict[str, Any] = artifact[values_key]
        assert values["min_weight"] > 0.0
        assert values["max_weight_sum_error"] <= 5.0e-16
        assert values["max_derivative_sum_error"] <= 2.0e-14
        assert set(values["observable_gradients"]) == {
            "lambda",
            "V0",
            "hnu",
            "W",
            "omega",
        }
        assert set(values["count_gradients"]) == {
            "lambda",
            "V0",
            "hnu",
            "W",
            "omega",
        }
