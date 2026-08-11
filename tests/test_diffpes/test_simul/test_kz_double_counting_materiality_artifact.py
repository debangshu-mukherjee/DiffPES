"""Verify the independent Plan-08b double-counting evidence.

The tests consume only a frozen inert JSON mirror. They bind its bytes to the
sole plans-side NumPy/SciPy generator and inspect the preregistered materiality
and crop-renormalization records without importing or executing that authority.
"""

import ast
import hashlib
import json
import math
from pathlib import Path

from beartype.typing import Any, Dict, Tuple

_REPOSITORY_ROOT: Path = Path(__file__).resolve().parents[3]
_REFERENCE_DIRECTORY: Path = (
    _REPOSITORY_ROOT / "tests" / "test_diffpes" / "_reference_data"
)
_ARTIFACT_PATH: Path = (
    _REFERENCE_DIRECTORY / "08b_double_counting_materiality.json"
)
_CONSUMER_MANIFEST_PATH: Path = (
    _REFERENCE_DIRECTORY / "08b_double_counting_materiality_manifest.json"
)
_GENERATOR_PATH: Path = (
    _REPOSITORY_ROOT
    / "diffpes-plans"
    / "verification"
    / "generate_08b_double_counting_materiality.py"
)
_PLANS_ARTIFACT_PATH: Path = (
    _REPOSITORY_ROOT
    / "diffpes-plans"
    / "verification"
    / "08b_double_counting_materiality.json"
)
_PLANS_MANIFEST_PATH: Path = (
    _REPOSITORY_ROOT
    / "diffpes-plans"
    / "verification"
    / "08b_double_counting_materiality_manifest.json"
)

_ARTIFACT_SHA256: str = (
    "d6df6e636a01f8982c1b1f1f358d5966aca41bca013da21d6c22a4e0292ae50c"
)
_CONSUMER_MANIFEST_SHA256: str = (
    "0e29dc46cc989ccc5651b723da61947d92a5d34f8e34304d8e562115aa3fb363"
)
_GENERATOR_SHA256: str = (
    "51ececca60f17f909fd134c7544e7dee0beb53c26223c90bf7e7e4f1ee146469"
)
_PLANS_MANIFEST_SHA256: str = (
    "0f7aa64638cc81db333772f1deb3f15265a1bd13a4f253d3c69055c6d017f4d5"
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


def _import_roots(path: Path) -> Tuple[str, ...]:
    """PRIVATE: Return every top-level imported package name.

    Parameters
    ----------
    path : Path
        Python source file to inspect.

    Returns
    -------
    roots : Tuple[str, ...]
        Sorted unique top-level imported names.
    """
    tree: ast.Module = ast.parse(path.read_text(encoding="utf-8"))
    roots_set: set[str] = set()
    node: ast.AST
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            alias: ast.alias
            for alias in node.names:
                roots_set.add(alias.name.split(".", maxsplit=1)[0])
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            roots_set.add(node.module.split(".", maxsplit=1)[0])
    roots: Tuple[str, ...] = tuple(sorted(roots_set))
    return roots


def test_materiality_artifact_authenticates_authority_and_mirror() -> None:
    """Bind every inert byte to one plans-side numeric authority.

    The test hashes the generator, both artifacts, and both manifests. It also
    parses imports and mutates one in-memory artifact byte as a tamper control.

    Notes
    -----
    Compare the plans artifact and consumer mirror byte for byte. Never import
    or execute the generator inside pytest.
    """
    assert _sha256(_ARTIFACT_PATH) == _ARTIFACT_SHA256
    assert _sha256(_PLANS_ARTIFACT_PATH) == _ARTIFACT_SHA256
    assert _sha256(_CONSUMER_MANIFEST_PATH) == _CONSUMER_MANIFEST_SHA256
    assert _sha256(_GENERATOR_PATH) == _GENERATOR_SHA256
    assert _sha256(_PLANS_MANIFEST_PATH) == _PLANS_MANIFEST_SHA256
    assert _ARTIFACT_PATH.read_bytes() == _PLANS_ARTIFACT_PATH.read_bytes()

    consumer_manifest: Dict[str, Any] = _load_json(_CONSUMER_MANIFEST_PATH)
    assert consumer_manifest["schema_version"] == 1
    assert consumer_manifest["artifact"] == {
        "path_from_diffpes_root": (
            "tests/test_diffpes/_reference_data/"
            "08b_double_counting_materiality.json"
        ),
        "sha256": _ARTIFACT_SHA256,
    }
    assert consumer_manifest["plans_artifact"] == {
        "path": (
            "diffpes-plans/verification/08b_double_counting_materiality.json"
        ),
        "sha256": _ARTIFACT_SHA256,
    }
    assert consumer_manifest["generator"] == {
        "mutable_authority_count": 1,
        "path": (
            "diffpes-plans/verification/"
            "generate_08b_double_counting_materiality.py"
        ),
        "sha256": _GENERATOR_SHA256,
    }

    plans_manifest: Dict[str, Any] = _load_json(_PLANS_MANIFEST_PATH)
    assert plans_manifest["artifact"]["sha256"] == _ARTIFACT_SHA256
    assert plans_manifest["generator"]["sha256"] == _GENERATOR_SHA256
    assert "sole mutable numeric authority" in plans_manifest["authority"]
    assert set(_import_roots(_GENERATOR_PATH)).isdisjoint({"diffpes", "jax"})

    original: bytes = _ARTIFACT_PATH.read_bytes()
    tampered: bytearray = bytearray(original)
    tampered[len(tampered) // 2] ^= 1
    tampered_digest: str = hashlib.sha256(tampered).hexdigest()
    assert tampered_digest != _ARTIFACT_SHA256


def test_materiality_artifact_freezes_the_registered_battery() -> None:
    """Require every escape-length and resolution counterexample.

    The artifact crosses four escape lengths with three detector widths. Each
    forbidden combined spectrum must differ from both single-counted routes.

    Notes
    -----
    Inspect only frozen metrics and probes. Treat ``1e-6`` as this fixture's
    roundoff-separation witness, never as a universal materiality threshold.
    """
    artifact: Dict[str, Any] = _load_json(_ARTIFACT_PATH)
    assert artifact["schema_version"] == 1
    assert artifact["artifact_id"] == (
        "plan08b-double-counting-materiality-v1"
    )
    assert artifact["status"] == "green"
    assert artifact["gate_ids"] == [
        "08b.risk.double_counting",
        "08b.G4",
    ]
    assert "no universal physical materiality threshold" in artifact["scope"]
    assert artifact["numeric_nonidentity_floor"] == 1.0e-6
    assert artifact["numeric_floor_semantics"] == (
        "roundoff-separation witness for this frozen fixture only"
    )

    profile: Dict[str, Any] = artifact["profile"]
    assert profile["mean_free_path_ang"] == [5.0, 10.0, 20.0, 50.0]
    assert profile["resolution_fwhm_ev"] == [0.0, 0.06, 0.18]
    assert profile["layer_depths_ang"] == [0.0, 2.6, 5.2, 7.8, 10.4, 13.0]
    assert profile["center_frac"] == 0.173
    assert profile["energy_axis_ev"] == {
        "minimum": -2.4,
        "maximum": 1.2,
        "step": 0.002,
        "count": 1801,
    }
    assert profile["probe_energies_ev"] == [-1.4, -1.0, -0.6, -0.2, 0.2]
    assert profile["resolution_operator"]["ordering"] == (
        "after each physical observable"
    )
    assert profile["resolution_operator"]["two_sided_tail_bound"] <= 1.0e-14

    cases: list[Dict[str, Any]] = artifact["cases"]
    assert len(cases) == 12
    combinations: set[Tuple[float, float]] = {
        (case["mean_free_path_ang"], case["resolution_fwhm_ev"])
        for case in cases
    }
    assert combinations == {
        (mean_free_path, resolution)
        for mean_free_path in (5.0, 10.0, 20.0, 50.0)
        for resolution in (0.0, 0.06, 0.18)
    }
    case: Dict[str, Any]
    for case in cases:
        comparison_name: str
        for comparison_name in (
            "combined_vs_bulk_kz",
            "combined_vs_coherent_slab",
        ):
            comparison: Dict[str, float] = case[comparison_name]
            assert all(math.isfinite(value) for value in comparison.values())
            assert comparison["relative_l1"] > 1.0e-6
            assert comparison["relative_linf"] > 1.0e-6
            assert comparison["absolute_integrated_relative"] >= 0.0
        probes: Dict[str, list[float]] = case["probe_values"]
        assert set(probes) == {
            "bulk_kz",
            "coherent_slab",
            "combined_forbidden",
        }
        assert all(len(values) == 5 for values in probes.values())
        assert all(
            math.isfinite(value)
            for values in probes.values()
            for value in values
        )


def test_materiality_artifact_records_nonconstant_95_percent_crop() -> None:
    """Expose the forbidden finite-window renormalization on a real observable.

    The positive cosine observable is nonconstant. The symmetric Cauchy window
    contains exactly 95% mass before the forbidden division by ``0.95``.

    Notes
    -----
    Compare the frozen analytic full expectation with SciPy's cropped integral.
    Keep the literal 5.263% division gain separate from the observable bias.
    """
    artifact: Dict[str, Any] = _load_json(_ARTIFACT_PATH)
    crop: Dict[str, Any] = artifact["crop_95_percent_counterexample"]
    assert crop["target_mass"] == 0.95
    assert crop["gamma"] == 0.2
    assert crop["observable"] == "1 + 0.4*cos(x)"
    assert crop["observable_min"] == 0.6
    assert crop["observable_max"] == 1.4
    assert math.isclose(
        crop["numeric_window_mass"],
        0.95,
        rel_tol=0.0,
        abs_tol=1.0e-12,
    )
    assert math.isclose(
        crop["division_gain_relative"],
        1.0 / 0.95 - 1.0,
        rel_tol=0.0,
        abs_tol=1.0e-16,
    )
    assert crop["mass_quadrature_error_estimate"] <= 1.0e-12
    assert crop["crop_quadrature_error_estimate"] <= 1.0e-12
    assert abs(crop["raw_crop_relative_bias"]) > 1.0e-6
    assert abs(crop["renormalized_relative_bias"]) > 1.0e-6
    assert not math.isclose(
        crop["crop_renormalized_expectation"],
        crop["full_expectation"],
        rel_tol=1.0e-6,
        abs_tol=1.0e-12,
    )
