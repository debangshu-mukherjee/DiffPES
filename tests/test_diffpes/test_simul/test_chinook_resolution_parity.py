"""Verify the frozen Chinook response-ordering compatibility seam.

The tests authenticate the offline source bundle, replay the isolated response
adapter, and keep wrong-spacing and wrong-axis controls permanently red.
"""

import hashlib
import json
from pathlib import Path

import numpy as np
from beartype.typing import Any, Dict
from jaxtyping import Float64
from jsonschema import Draft202012Validator
from numpy.typing import NDArray

from tests._reference_tools.chinook_resolution_adapter import (
    comparison_metrics,
    matched_resolution,
    wrong_axis_order_resolution,
    wrong_nominal_spacing_resolution,
)


def _sha256(path: Path) -> str:
    """PRIVATE: Return one frozen file's SHA-256 digest.

    Parameters
    ----------
    path : Path
        Frozen file to authenticate.

    Returns
    -------
    digest : str
        Lowercase hexadecimal SHA-256 digest.

    Notes
    -----
    The helper hashes the complete byte payload in one operation.
    """
    digest: str = hashlib.sha256(path.read_bytes()).hexdigest()
    return digest


def _reference_root() -> Path:
    """PRIVATE: Return the frozen response-parity consumer directory.

    Returns
    -------
    root : Path
        Directory containing inert numeric and provenance artifacts.

    Notes
    -----
    The path derives from this collected test rather than the process working
    directory.
    """
    root: Path = (
        Path(__file__).resolve().parents[1]
        / "_reference_data/chinook_resolution_parity"
    )
    return root


def test_lithium_chain_resolution_bundle_authenticates() -> None:
    """Verify the frozen model, environment, source, axes, and schema.

    The check binds every consumer input to the immutable resolution-parity
    bundle before any numerical replay contributes acceptance evidence.

    Notes
    -----
    Literal root hashes anchor the manifest and source adapter. The normative
    JSON Schema rejects missing or unknown summary fields.
    """
    reference_root: Path = _reference_root()
    manifest_path: Path = reference_root / "manifest.json"
    summary_path: Path = reference_root / "summary.json"
    schema_path: Path = reference_root / "schema.json"
    model_path: Path = reference_root / "model_spec.json"
    environment_path: Path = reference_root / "chinook_env_freeze.txt"
    adapter_path: Path = (
        Path(__file__).resolve().parents[2]
        / "_reference_tools/chinook_resolution_adapter.py"
    )
    assert _sha256(manifest_path) == (
        "458d3b790b5db86d24a1084beab914e07fcfb9f5b0f6e5312b521a2edd1a2260"
    )
    assert _sha256(model_path) == (
        "bf11fed1cd03bee97b255af4951b552d64ee181436492aed0c16727d6c49abbe"
    )
    assert _sha256(environment_path) == (
        "6d00cb4df251508b6392273b1df166f6a17abe8f6691cffead45c636e8ef2531"
    )
    assert _sha256(adapter_path) == (
        "4980a4b1642b1b6ffbeb37e61290712b9775f12ead89403ad70e1ff44eb84797"
    )
    manifest: Dict[str, Any] = json.loads(
        manifest_path.read_text(encoding="utf-8")
    )
    summary: Dict[str, Any] = json.loads(
        summary_path.read_text(encoding="utf-8")
    )
    schema: Dict[str, Any] = json.loads(
        schema_path.read_text(encoding="utf-8")
    )
    validator: Any = Draft202012Validator(schema)
    validator.validate(summary)
    assert summary["chinook_commit"] == (
        "24913de8cc5b8c162f7c1b4acc64bd1b54dd548b"
    )
    assert (
        summary["source_authentication"]["chinook_package_content_sha256"]
        == "e2a12678f55c74317dc7309d2a565794fecfdafaea9ab5d57d5c4a73b4aa4b94"
    )
    assert summary["source_authentication"]["adapter_sha256"] == _sha256(
        adapter_path
    )
    source_map: Dict[str, str] = summary["source_authentication"][
        "production_source_sha256"
    ]
    relative_path: str
    digest: str
    for relative_path, digest in source_map.items():
        assert _sha256(
            Path(__file__).resolve().parents[3] / relative_path
        ) == (digest)
    assert (
        summary["scope"]["classification"] == "K-only response compatibility"
    )
    assert summary["scope"]["complete_shape"] == [1, 241, 601]
    assert summary["diagnostics"]["public_long_tail_expected_peak_ratio"] == (
        2.078353654782797e-5
    )
    assert manifest["axes_serialized"]["ky_axis_inv_ang"] == [0.0]
    archive: np.lib.npyio.NpzFile
    with np.load(
        reference_root / "chinook_reference.npz",
        allow_pickle=False,
    ) as archive:
        expected_keys: set[str] = {
            "kx_axis_inv_ang",
            "ky_axis_inv_ang",
            "omega_rel_ev",
            "intensity_raw",
            "intensity_broadened",
            "band_energies_ev",
            "matrix_elements",
            "pks_state",
            "retained_k_indices",
            "resolution_fwhm_axis_order_ky_kx_omega",
            "resolution_sigma_pixels_axis_order_ky_kx_omega",
            "diagnostic_k_inv_ang",
            "chinook_hamiltonians_ev",
        }
        assert set(archive.files) == expected_keys
        assert archive["intensity_raw"].shape == (1, 241, 601)
        assert archive["intensity_broadened"].shape == (1, 241, 601)
        assert np.all(np.isfinite(archive["intensity_raw"]))


def test_lithium_chain_matched_response_passes_frozen_envelopes() -> None:
    """Replay Chinook's exact sampled response on the complete reference cut.

    The acceptance check fits exactly one nonnegative scale on the declared
    five-sigma interior and requires elementwise and strict peak envelopes.

    Notes
    -----
    The test sends the authenticated pre-resolution cut through only the
    test-side response adapter, with no Chinook import or production branch.
    """
    reference_root: Path = _reference_root()
    archive: np.lib.npyio.NpzFile
    with np.load(
        reference_root / "chinook_reference.npz",
        allow_pickle=False,
    ) as archive:
        raw: Float64[NDArray, "ky kx omega"] = np.asarray(
            archive["intensity_raw"],
            dtype=float,
        )
        reference: Float64[NDArray, "ky kx omega"] = np.asarray(
            archive["intensity_broadened"],
            dtype=float,
        )
        kx_axis: Float64[NDArray, " kx"] = np.asarray(
            archive["kx_axis_inv_ang"],
            dtype=float,
        )
        omega_axis: Float64[NDArray, " omega"] = np.asarray(
            archive["omega_rel_ev"],
            dtype=float,
        )
    candidate: Float64[NDArray, "ky kx omega"] = matched_resolution(
        raw,
        kx_axis,
        omega_axis,
        energy_fwhm_ev=0.03,
        momentum_fwhm_inv_ang=0.02,
    )
    metrics: Dict[str, object] = comparison_metrics(
        candidate,
        reference,
        kx_axis,
        omega_axis,
        energy_fwhm_ev=0.03,
        momentum_fwhm_inv_ang=0.02,
    )
    assert metrics["crop_cells_axis_order_ky_kx_omega"] == [0, 5, 13]
    assert metrics["interior_shape"] == [1, 231, 575]
    assert metrics["elementwise_pass"] is True
    assert metrics["strict_peak_scaled_pass"] is True
    fitted_scale: float = float(metrics["profiled_nonnegative_scale"])
    strict_ratio: float = float(metrics["max_absolute_error_over_peak"])
    integral_ratio: float = float(
        metrics["candidate_to_reference_integral_ratio_full_grid"]
    )
    assert np.isclose(fitted_scale, 1.0, rtol=0.0, atol=1.0e-12)
    assert strict_ratio <= 1.0e-6
    assert np.isclose(integral_ratio, 1.0, rtol=0.0, atol=1.0e-12)


def test_lithium_chain_wrong_spacing_and_axis_controls_fail() -> None:
    """Reject planted sigma-spacing and energy-momentum axis defects.

    Both controls keep the frozen input and kernel family while corrupting one
    binding Chinook response convention, so profiling cannot hide either bug.

    Notes
    -----
    The test evaluates each control under the same crop and single-scale
    metrics as the accepted adapter and requires both envelopes to reject it.
    """
    reference_root: Path = _reference_root()
    archive: np.lib.npyio.NpzFile
    with np.load(
        reference_root / "chinook_reference.npz",
        allow_pickle=False,
    ) as archive:
        raw: Float64[NDArray, "ky kx omega"] = np.asarray(
            archive["intensity_raw"],
            dtype=float,
        )
        reference: Float64[NDArray, "ky kx omega"] = np.asarray(
            archive["intensity_broadened"],
            dtype=float,
        )
        kx_axis: Float64[NDArray, " kx"] = np.asarray(
            archive["kx_axis_inv_ang"],
            dtype=float,
        )
        omega_axis: Float64[NDArray, " omega"] = np.asarray(
            archive["omega_rel_ev"],
            dtype=float,
        )
    wrong_spacing: Float64[NDArray, "ky kx omega"] = (
        wrong_nominal_spacing_resolution(
            raw,
            kx_axis,
            omega_axis,
            energy_fwhm_ev=0.03,
            momentum_fwhm_inv_ang=0.02,
        )
    )
    wrong_axis: Float64[NDArray, "ky kx omega"] = wrong_axis_order_resolution(
        raw,
        kx_axis,
        omega_axis,
        energy_fwhm_ev=0.03,
        momentum_fwhm_inv_ang=0.02,
    )
    spacing_metrics: Dict[str, object] = comparison_metrics(
        wrong_spacing,
        reference,
        kx_axis,
        omega_axis,
        energy_fwhm_ev=0.03,
        momentum_fwhm_inv_ang=0.02,
    )
    axis_metrics: Dict[str, object] = comparison_metrics(
        wrong_axis,
        reference,
        kx_axis,
        omega_axis,
        energy_fwhm_ev=0.03,
        momentum_fwhm_inv_ang=0.02,
    )
    assert spacing_metrics["elementwise_pass"] is False
    assert spacing_metrics["strict_peak_scaled_pass"] is False
    assert axis_metrics["elementwise_pass"] is False
    assert axis_metrics["strict_peak_scaled_pass"] is False
