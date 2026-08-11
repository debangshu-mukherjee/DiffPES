"""Generate the Chinook spectral-function comparison artifact.

This script assembles the inert, frozen behavioral-comparison artifact
``tests/test_diffpes/_reference_data/chinook_spectral_reference.npz``
together with its provenance manifest ``chinook_spectral_manifest.json``.
Never import, link, or invoke Chinook here. Keep external-project Python in
the isolated verification repository. This boundary follows the external
reference policy and the Chinook parity protocol. Only inert numeric data,
hashes, and provenance enter the DiffPES test tree.

Offline authority workflow (run manually outside this repository)
-----------------------------------------------------------------
1. Build the isolated pinned environment (parity protocol section 1.2):
   Python 3.11, ``numpy==1.26.4``, ``scipy==1.11.4``, ``matplotlib==3.8.4``,
   then ``pip install --no-deps
   "chinook @ git+https://github.com/qmlab-ubc-ca/chinook@24913de8cc5b8c1
   62f7c1b4acc64bd1b54dd548b"``.  The sorted ``pip freeze --all`` must hash
   to ``EXPECTED_ENVIRONMENT_SHA256`` (the same environment identity as the
   committed slab and matrix-element Chinook artifacts).
2. Execute, from a directory outside the DiffPES checkout::

       MPLBACKEND=Agg MPLCONFIGDIR=/tmp/diffpes-mpl \
           <chinook-venv>/bin/python \
           <verification-repository>/spectral_chinook_crosscheck/\
run_chinook_spectral.py <raw-output-dir>

3. Run this script with ``--raw-dir <raw-output-dir>``.  It authenticates
   the offline provenance, revalidates every frozen model and axis
   parameter, and writes the immutable reference archive plus manifest.

Stop with a clear message when the offline workflow has not run. Keep Chinook
absent from the DiffPES environment. Never make it a runtime or test
dependency.

Use the parity-protocol graphene two-site reference model. It contains two
carbon 2p-z orbitals and nearest-neighbour hopping of -2.7 eV. Chinook
``experiment.spectral`` uses constant ``Sigma = -i*0.02`` eV. It also uses
its hard-coded ``-i*5e-5`` eta floor and T = 4.2 K. The registered axis stores
relative energy. Record Chinook ``gen_SE_KK`` rejection of DiffPES's negative
retarded ``Sigma'' <= 0`` convention. This expected divergence carries no
acceptance tolerance.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from beartype.typing import Any, Dict, Tuple, Union
from jaxtyping import Complex128, Float64
from numpy.typing import NDArray

CHINOOK_REPOSITORY: str = "https://github.com/qmlab-ubc-ca/chinook"
CHINOOK_COMMIT: str = "24913de8cc5b8c162f7c1b4acc64bd1b54dd548b"
CHINOOK_DISTRIBUTION_VERSION: str = "1.1.3"
CHINOOK_SNAPSHOT_ARCHIVE_SHA256: str = (
    "2cf3c830236af5c6311949cf8a655467fd7d10e75e191250e3bac4e37a00e01b"
)
CHINOOK_PACKAGE_CONTENT_SHA256: str = (
    "e2a12678f55c74317dc7309d2a565794fecfdafaea9ab5d57d5c4a73b4aa4b94"
)
EXPECTED_ENVIRONMENT_SHA256: str = (
    "6d00cb4df251508b6392273b1df166f6a17abe8f6691cffead45c636e8ef2531"
)
OFFLINE_RUNNER: str = "run_chinook_spectral.py"

LATTICE_CONSTANT_ANGSTROM: float = 2.46
HOPPING_EV: float = -2.7
PHOTON_ENERGY_EV: float = 21.2
WORK_FUNCTION_EV: float = 4.5
MEAN_FREE_PATH_ANGSTROM: float = 10.0
TEMPERATURE_K: float = 4.2
GAMMA_EV: float = 0.02
ETA_FLOOR_EV: float = 5.0e-5
MAP_HALF_WIDTH_INVERSE_ANGSTROM: float = 0.18
MAP_POINTS_PER_AXIS: int = 31
OMEGA_MIN_EV: float = -2.5
OMEGA_MAX_EV: float = 0.2
OMEGA_POINTS: int = 261
N_ORBITALS: int = 2
HERMITIAN_ABSOLUTE_TOLERANCE: float = 1.0e-12
RECONSTRUCTION_ABSOLUTE_TOLERANCE: float = 1.0e-14
FORENSIC_REJECTED_SIGMA_EV: complex = -0.01j

RAW_ARCHIVE_NAME: str = "chinook_spectral_raw.npz"
REFERENCE_KEYS: Tuple[str, ...] = (
    "kx_axis_inverse_angstrom",
    "ky_axis_inverse_angstrom",
    "omega_rel_ev",
    "valley_centre_inverse_angstrom",
    "intensity_raw",
    "intensity_resolution_broadened",
    "fermi_distribution",
    "m_factor_state",
    "pks_state",
    "kpoints_cartesian_inverse_angstrom",
    "band_energies_k_band_ev",
    "hamiltonians_k_orb_orb_ev",
    "graphene_lattice_angstrom",
    "graphene_positions_cartesian_angstrom",
    "graphene_hopping_records",
    "forensic_omega_rel_ev",
    "forensic_negative_retarded_input",
    "forensic_rejected_sigma",
    "forensic_accepted_sigma",
    "forensic_negative_coefficient_sigma",
    "forensic_positive_coefficient_sigma",
)

MISSING_RAW_MESSAGE: str = (
    "Chinook raw output not found: {path}\n"
    "Chinook is intentionally absent from the DiffPES environment and is "
    "never a runtime or test dependency (external-reference policy; "
    "chinook-parity protocol section 1.4).  Generate the raw archive "
    "offline in the isolated pinned Chinook environment with\n"
    f"  {OFFLINE_RUNNER}\n"
    "and re-run this script with --raw-dir pointing at its output "
    "directory."
)


def _sha256_bytes(payload: bytes) -> str:
    """PRIVATE: Return the SHA-256 hex digest of a byte string.

    Parameters
    ----------
    payload : bytes
        Bytes to digest.

    Returns
    -------
    digest : str
        Lowercase hexadecimal SHA-256 of the payload.

    Notes
    -----
    One ``hashlib.sha256`` call digests the whole payload at once.
    """
    digest: str = hashlib.sha256(payload).hexdigest()
    return digest


def _sha256(path: Path) -> str:
    """PRIVATE: Return the SHA-256 hex digest of one file.

    Parameters
    ----------
    path : Path
        File to digest.

    Returns
    -------
    digest : str
        Lowercase hexadecimal SHA-256 of the file bytes.

    Notes
    -----
    The function reads the complete file into memory and delegates to
    :func:`_sha256_bytes`.
    """
    digest: str = _sha256_bytes(path.read_bytes())
    return digest


def _expected_axes() -> Dict[str, Float64[NDArray, "..."]]:
    """PRIVATE: Build the frozen momentum and relative-energy axes.

    Returns
    -------
    axes : Dict[str, Float64[NDArray, "..."]]
        Valley centre in 1/Angstrom, both 31-point momentum axes in
        1/Angstrom, and the 261-point relative-energy axis in eV.

    Notes
    -----
    The valley centre is ``(2 pi / a) (1/sqrt(3), 1/3, 0)`` for the
    frozen graphene lattice constant ``a = 2.46`` Angstrom; both
    momentum axes span ``+-0.18`` 1/Angstrom around it.
    """
    valley: Float64[NDArray, "3"] = (
        2.0 * np.pi / LATTICE_CONSTANT_ANGSTROM
    ) * np.asarray([1.0 / np.sqrt(3.0), 1.0 / 3.0, 0.0], dtype=np.float64)
    axes: Dict[str, Float64[NDArray, " n_axis"]] = {
        "valley": valley,
        "kx": np.linspace(
            valley[0] - MAP_HALF_WIDTH_INVERSE_ANGSTROM,
            valley[0] + MAP_HALF_WIDTH_INVERSE_ANGSTROM,
            MAP_POINTS_PER_AXIS,
        ),
        "ky": np.linspace(
            valley[1] - MAP_HALF_WIDTH_INVERSE_ANGSTROM,
            valley[1] + MAP_HALF_WIDTH_INVERSE_ANGSTROM,
            MAP_POINTS_PER_AXIS,
        ),
        "omega": np.linspace(OMEGA_MIN_EV, OMEGA_MAX_EV, OMEGA_POINTS),
    }
    return axes


def _require(condition: bool, message: str) -> None:
    """PRIVATE: Raise a validation error when a frozen requirement fails.

    Parameters
    ----------
    condition : bool
        Result of one frozen-requirement check.
    message : str
        Failure description for the exit message.

    Raises
    ------
    SystemExit
        If ``condition`` is false, with ``validation failed:`` and the
        message.

    Notes
    -----
    ``SystemExit`` stops the assembler immediately, so no partially
    validated artifact ever reaches the reference directory.
    """
    if not condition:
        raise SystemExit(f"validation failed: {message}")


def _validate_provenance(raw_dir: Path) -> Dict[str, Any]:
    """PRIVATE: Validate the offline run's environment and pins.

    Parameters
    ----------
    raw_dir : Path
        Directory written by the offline Chinook runner.

    Returns
    -------
    metadata : Dict[str, Any]
        Parsed ``run_metadata.json`` of the authenticated offline run.

    Implementation Logic
    --------------------
    Hash the environment freeze. Compare it with the recorded digest and the
    expected parity-environment digest. Verify the raw-archive digest and
    pinned library versions from the metadata record.

    Raises
    ------
    SystemExit
        If a required offline file is absent. The delegated requirement
        checks also stop on a failed commit, environment, archive, library,
        or forensic-record comparison.
    """
    metadata_path: Path = raw_dir / "run_metadata.json"
    freeze_path: Path = raw_dir / "chinook_env_freeze.txt"
    archive_path: Path = raw_dir / RAW_ARCHIVE_NAME
    path: Path
    for path in (metadata_path, freeze_path, archive_path):
        if not path.is_file():
            raise SystemExit(MISSING_RAW_MESSAGE.format(path=path))
    metadata: Dict[str, Any] = json.loads(metadata_path.read_text())
    _require(
        metadata.get("chinook_commit") == CHINOOK_COMMIT,
        "offline run does not pin the protocol Chinook commit",
    )
    environment_hash: str = _sha256(freeze_path)
    _require(
        environment_hash == metadata.get("environment_sha256"),
        "environment freeze does not match the recorded hash",
    )
    _require(
        environment_hash == EXPECTED_ENVIRONMENT_SHA256,
        "environment freeze differs from the pinned parity environment",
    )
    _require(
        metadata.get("archive_sha256") == _sha256(archive_path),
        "raw archive digest differs from the offline record",
    )
    _require(
        metadata.get("numpy_version") == "1.26.4",
        "offline run must use numpy 1.26.4 (parity protocol section 1.1)",
    )
    _require(
        metadata.get("scipy_version") == "1.11.4",
        "offline run must use scipy 1.11.4 (parity protocol section 1.1)",
    )
    forensic: Dict[str, Any] = metadata.get("forensic_gen_se_kk", {})
    _require(
        forensic.get("rejected_equals_constant_fallback") == "True",
        "forensic record must show the constant -0.01j fallback",
    )
    return metadata


def _validate_arrays(
    raw: Dict[
        str,
        Union[
            Float64[NDArray, "..."],
            Complex128[NDArray, "..."],
        ],
    ],
) -> None:
    """PRIVATE: Validate every frozen model and axis parameter.

    Parameters
    ----------
    raw : Dict[str, Union[Float64[NDArray, "..."], Complex128[NDArray, "..."]]]
        Arrays loaded from the offline raw archive.

    Notes
    -----
    Rebuild the frozen axes. Compare endpoints bitwise and other values to
    1e-13. Reconstruct the complete intensity cube from serialized states.
    Each retained state supplies one Lorentzian. Scale it by its
    matrix-element factor and the shared Fermi factor.

    Delegated requirement checks stop validation when an array, axis, shape,
    value range, Hamiltonian, spectral reconstruction, or forensic record
    differs from the frozen authority.
    """
    key: str
    for key in REFERENCE_KEYS:
        _require(key in raw, f"missing array {key}")
    axes: Dict[str, Float64[NDArray, " n_axis"]] = _expected_axes()
    name: str
    expected: Float64[NDArray, " n_axis"]
    for name, expected in (
        ("kx_axis_inverse_angstrom", axes["kx"]),
        ("ky_axis_inverse_angstrom", axes["ky"]),
        ("omega_rel_ev", axes["omega"]),
        ("forensic_omega_rel_ev", axes["omega"]),
        ("valley_centre_inverse_angstrom", axes["valley"]),
    ):
        _require(
            raw[name].shape == expected.shape
            and bool(np.allclose(raw[name], expected, rtol=0.0, atol=1e-13))
            and raw[name][0] == expected[0]
            and raw[name][-1] == expected[-1],
            f"axis {name} differs from the frozen specification",
        )
    n_axis: int = MAP_POINTS_PER_AXIS
    cube_shape: Tuple[int, int, int] = (n_axis, n_axis, OMEGA_POINTS)
    _require(
        raw["intensity_raw"].shape == cube_shape,
        "intensity_raw must be a (ky, kx, omega_rel) cube",
    )
    _require(
        raw["intensity_resolution_broadened"].shape == cube_shape,
        "broadened cube shape mismatch",
    )
    for key in REFERENCE_KEYS:
        _require(
            bool(np.all(np.isfinite(raw[key]))),
            f"non-finite value in {key}",
        )
    _require(
        bool(np.all(raw["intensity_raw"] >= 0.0)),
        "spectral intensity must be nonnegative",
    )
    fermi: Float64[NDArray, " n_omega"] = np.asarray(
        raw["fermi_distribution"], dtype=np.float64
    )
    _require(
        bool(np.all((fermi >= 0.0) & (fermi <= 1.0))),
        "Fermi-Dirac factor out of [0, 1]",
    )
    pks: Float64[NDArray, "n_state 4"] = np.asarray(
        raw["pks_state"], dtype=np.float64
    )
    _require(
        bool(np.all(pks[:, 1:3] == np.floor(pks[:, 1:3])))
        and bool(np.all(pks[:, 1:3] >= 0))
        and bool(np.all(pks[:, 1:3] < n_axis)),
        "state map indices out of range",
    )
    n_kpoints: int = n_axis * n_axis
    _require(
        raw["band_energies_k_band_ev"].shape == (n_kpoints, N_ORBITALS),
        "band energy table shape mismatch",
    )
    _require(
        raw["hamiltonians_k_orb_orb_ev"].shape
        == (n_kpoints, N_ORBITALS, N_ORBITALS),
        "diagnostic Hamiltonian shape mismatch",
    )
    hermitian_defect: np.float64 = np.max(
        np.abs(
            raw["hamiltonians_k_orb_orb_ev"]
            - np.conj(np.swapaxes(raw["hamiltonians_k_orb_orb_ev"], 1, 2))
        )
    )
    _require(
        bool(hermitian_defect < HERMITIAN_ABSOLUTE_TOLERANCE),
        "diagnostic Hamiltonians must be Hermitian",
    )
    reconstructed: Float64[NDArray, "n_ky n_kx n_omega"] = np.zeros(
        cube_shape, dtype=np.float64
    )
    omega: Float64[NDArray, " n_omega"] = np.asarray(
        raw["omega_rel_ev"], dtype=np.float64
    )
    sigma_total: complex = -1.0j * GAMMA_EV - 1.0j * ETA_FLOOR_EV
    state: int
    for state in range(pks.shape[0]):
        row: int = int(pks[state, 1])
        column: int = int(pks[state, 2])
        lorentzian: Float64[NDArray, " n_omega"] = np.imag(
            -1.0 / (np.pi * (omega - pks[state, 3] - sigma_total))
        )
        reconstructed[row, column, :] += (
            raw["m_factor_state"][state] * lorentzian * fermi
        )
    defect: np.float64 = np.max(np.abs(reconstructed - raw["intensity_raw"]))
    _require(
        bool(defect < RECONSTRUCTION_ABSOLUTE_TOLERANCE),
        "declared spectral semantics do not reproduce intensity_raw "
        f"(max abs defect {defect:.3e})",
    )
    _require(
        bool(
            np.all(
                raw["forensic_rejected_sigma"] == FORENSIC_REJECTED_SIGMA_EV
            )
            and np.array_equal(
                raw["forensic_negative_coefficient_sigma"],
                raw["forensic_positive_coefficient_sigma"],
            )
            and np.all(raw["forensic_negative_retarded_input"] <= 0.0)
        ),
        "forensic arrays do not show the documented gen_SE_KK behavior",
    )


def _model_parameters() -> Dict[str, Any]:
    """PRIVATE: Return the frozen model and axis parameters.

    Returns
    -------
    parameters : Dict[str, Any]
        Manifest section with the two-site graphene model, the frozen
        experiment constants in eV, Kelvin, and Angstrom units, the
        axis specifications, and the intensity semantics.

    Notes
    -----
    Every value comes from the module-level frozen constants; the
    function only arranges them for the manifest.
    """
    parameters: Dict[str, Any] = {
        "model_id": "graphene-two-site-spectral",
        "protocol_model": (
            "chinook-parity protocol section 2.2 graphene p_z two-site "
            "reference model"
        ),
        "lattice_constant_angstrom": LATTICE_CONSTANT_ANGSTROM,
        "lattice_vectors": (
            "a1=a*(sqrt(3)/2,-1/2,0), a2=a*(sqrt(3)/2,1/2,0), a3=(0,0,10)"
        ),
        "positions_fractional": [
            [0.0, 0.0, 0.0],
            [1.0 / 3.0, 1.0 / 3.0, 0.0],
        ],
        "species": ["C", "C"],
        "orbitals": ["2p_z", "2p_z"],
        "onsite_ev": 0.0,
        "nearest_neighbor_hopping_ev": HOPPING_EV,
        "spin": "spinless",
        "photon_energy_ev": PHOTON_ENERGY_EV,
        "work_function_ev": WORK_FUNCTION_EV,
        "mean_free_path_angstrom": MEAN_FREE_PATH_ANGSTROM,
        "temperature_k": TEMPERATURE_K,
        "polarization_cartesian": [1.0, 0.0, 0.0],
        "self_energy": {
            "mode": "constant",
            "sigma_ev": "-i*0.02",
            "gamma_ev": GAMMA_EV,
            "eta_floor_ev": ETA_FLOOR_EV,
            "eta_floor_origin": (
                "Chinook ARPES_lib.experiment.spectral hard-codes -0.00005j "
                "in the resolvent denominator (declared eta-floor "
                "difference)"
            ),
        },
        "radial_mode": (
            "fixed; p-to-s and p-to-d values (1, -1) after the i**lprime "
            "phase, matching the committed Chinook matrix-element parity "
            "artifact"
        ),
        "valley_centre": "(2*pi/a)*(1/sqrt(3), 1/3, 0)",
        "map_half_width_inverse_angstrom": MAP_HALF_WIDTH_INVERSE_ANGSTROM,
        "map_points_per_axis": MAP_POINTS_PER_AXIS,
        "kz_inverse_angstrom": 0.0,
        "omega_rel_axis_ev": [OMEGA_MIN_EV, OMEGA_MAX_EV, OMEGA_POINTS],
        "omega_meaning": (
            "relative energy E - E_F; the half-filled graphene toy puts "
            "E_F = 0 exactly, so Chinook's cube energy axis is the "
            "program-wide omega_rel axis and the Fermi factor midpoint "
            "sits at omega_rel = 0"
        ),
        "resolution_recorded_not_applied": {
            "E_ev": 0.02,
            "k_inverse_angstrom": 0.01,
            "note": (
                "the comparison array intensity_raw is Chinook's "
                "pre-resolution cube; the Gaussian-filtered cube is stored "
                "as informative data only and is outside the registered "
                "spectral comparison"
            ),
        },
        "state_selection": (
            "Chinook datacube retains eigenstates with energy in "
            "[E0 - 5*dE_bin, E1 + 5*dE_bin], dE_bin = (E1 - E0)/n_omega "
            "= 2.7/261 eV; excluded states contribute no Lorentzian tail. "
            "The retained states are serialized in pks_state as "
            "(flat_state_index, ky_index, kx_index, energy_ev)"
        ),
        "intensity_semantics": (
            "I[ky, kx, :] = sum_states M_factor * "
            "Im(-1 / (pi * (omega - eps - (-i*Gamma - i*eta)))) * "
            "f_FD(omega, 0, T); axis order (ky, kx, omega_rel)"
        ),
    }
    return parameters


def _forensic_section(metadata: Dict[str, Any]) -> Dict[str, Any]:
    """PRIVATE: Return the diagnostic sign-convention forensic record.

    Parameters
    ----------
    metadata : Dict[str, Any]
        Authenticated offline run metadata.

    Returns
    -------
    section : Dict[str, Any]
        Manifest section that documents how Chinook ``gen_SE_KK``
        rejects the negative retarded convention, with the captured
        offline behavior attached.

    Notes
    -----
    The record is evidence only: it carries no acceptance tolerance
    and determines no acceptance outcome.
    """
    forensic: Dict[str, Any] = metadata.get("forensic_gen_se_kk", {})
    section: Dict[str, Any] = {
        "name": "self-energy-sign-convention-divergence",
        "diagnostic_only": True,
        "acceptance_tolerance": None,
        "classification": (
            "documented expected divergence; forensic evidence only "
            "(chinook-parity protocol section 9 triage doctrine)"
        ),
        "diffpes_convention": (
            "retarded self-energy with Sigma''(omega) <= 0 everywhere; "
            "KK real part from the verified causal-pair/KK operator"
        ),
        "chinook_behavior": (
            "ARPES_lib.gen_SE_KK requires the magnitude |Sigma''| as "
            "input: its single-signed validation passes only when every "
            "sample is positive, after which it flips the sign "
            "internally.  A callable already returning the negative "
            "retarded Sigma'' is declared invalid and silently replaced "
            "by the constant -0.01j.  Polynomial-coefficient input "
            "cannot express a negative Sigma'' at all: coefficients pass "
            "through abs(), so an all-negative coefficient list with "
            "nonzero max is silently rectified, while one whose max is "
            "-0.0 is misclassified as imaginary-valued input, zeroed by "
            "np.imag, and crashes with an IndexError"
        ),
        "evidence_arrays": [
            "forensic_omega_rel_ev",
            "forensic_negative_retarded_input",
            "forensic_rejected_sigma",
            "forensic_accepted_sigma",
            "forensic_negative_coefficient_sigma",
            "forensic_positive_coefficient_sigma",
        ],
        "captured_behavior": forensic,
        "disposition": (
            "convention divergence documented, not emulated; DiffPES "
            "production and the registered spectral comparison are "
            "unaffected and no reference-code sign behavior enters "
            "production"
        ),
    }
    return section


def main() -> None:
    """Assemble and freeze the spectral reference artifact and manifest.

    Raises
    ------
    SystemExit
        If the caller omits the raw directory or any authority check fails.

    Notes
    -----
    The assembler authenticates an offline Chinook run. It then copies only
    the registered arrays and provenance into the inert repository artifacts.
    """
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description=__doc__
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=None,
        help="directory written by the offline Chinook runner",
    )
    arguments: argparse.Namespace = parser.parse_args()
    if arguments.raw_dir is None:
        raise SystemExit(
            MISSING_RAW_MESSAGE.format(path="<--raw-dir not provided>")
        )
    raw_dir: Path = arguments.raw_dir.resolve()
    metadata: Dict[str, Any] = _validate_provenance(raw_dir)
    archive: np.lib.npyio.NpzFile
    with np.load(raw_dir / RAW_ARCHIVE_NAME) as archive:
        raw: Dict[
            str,
            Union[
                Float64[NDArray, "..."],
                Complex128[NDArray, "..."],
            ],
        ] = {key: np.asarray(archive[key]) for key in archive.files}
    _validate_arrays(raw)

    data_dir: Path = (
        Path(__file__).resolve().parents[1]
        / "test_diffpes"
        / "_reference_data"
    )
    reference_path: Path = data_dir / "chinook_spectral_reference.npz"
    manifest_path: Path = data_dir / "chinook_spectral_manifest.json"
    np.savez_compressed(
        reference_path,
        **{key: raw[key] for key in REFERENCE_KEYS},
    )

    model_spec_path: Path = raw_dir / "model_spec.json"
    manifest: Dict[str, Any] = {
        "schema": "diffpes.chinook-spectral-reference.v1",
        "requirement": "chinook-spectral-parity",
        "classification": "K-type behavioral compatibility",
        "comparison": {
            "comparison_array": "intensity_raw",
            "quantity": "I(k, omega_rel)",
            "rtol": 1e-6,
            "policy": (
                "pytest compares I(k, omega_rel) using artifact data "
                "only, after the independent correctness evidence "
                "passes; agreement establishes compatibility, never "
                "correctness"
            ),
        },
        "chinook_repository": CHINOOK_REPOSITORY,
        "chinook_commit": CHINOOK_COMMIT,
        "chinook_distribution_version": CHINOOK_DISTRIBUTION_VERSION,
        "chinook_distribution": (
            f"chinook @ git+{CHINOOK_REPOSITORY}@{CHINOOK_COMMIT}"
        ),
        "chinook_snapshot_archive": {
            "source": (
                f"{CHINOOK_REPOSITORY}/archive/{CHINOOK_COMMIT}.tar.gz"
            ),
            "sha256": CHINOOK_SNAPSHOT_ARCHIVE_SHA256,
            "package_content_sha256": CHINOOK_PACKAGE_CONTENT_SHA256,
            "package_content_hash_rule": (
                "sha256 of the sorted per-file sha256sum listing of every "
                "*.py and *.txt file under the chinook/ package directory "
                "(paths relative to that directory, __pycache__ excluded); "
                "identical for the snapshot tarball, the installed pinned "
                "distribution, and the local read-only snapshot"
            ),
        },
        "environment": {
            "environment_sha256": metadata["environment_sha256"],
            "python_version": metadata["python_version"],
            "numpy_version": metadata["numpy_version"],
            "scipy_version": metadata["scipy_version"],
            "matplotlib_version": "3.8.4",
            "platform": metadata["platform"],
            "machine": metadata["machine"],
            "numpy_build_configuration": metadata["numpy_build_configuration"],
            "note": (
                "identical environment identity (same freeze hash) as the "
                "committed slab and matrix-element Chinook artifacts"
            ),
        },
        "offline_generation": {
            "runner": OFFLINE_RUNNER,
            "runner_generator_field": metadata["generator"],
            "raw_archive": RAW_ARCHIVE_NAME,
            "raw_archive_sha256": metadata["archive_sha256"],
            "model_specification_sha256": metadata[
                "model_specification_sha256"
            ],
            "policy": (
                "generated manually in the isolated pinned environment "
                "outside the DiffPES repository; DiffPES production code, "
                "pytest, and CI never import or invoke Chinook, this "
                "runner, or this assembler's offline inputs"
            ),
        },
        "assembler": {
            "script": "tests/_reference_tools/"
            "generate_chinook_spectral_reference.py",
            "script_sha256": _sha256(Path(__file__).resolve()),
            "boundary": (
                "imports NumPy only; consumes the offline raw archive and "
                "re-freezes it after full parameter revalidation"
            ),
        },
        "diffpes_baseline_commit": (
            "551f7fca10dde5adbd856a526b63c6e77ad8f1d3"
        ),
        "model_and_axis_parameters": _model_parameters(),
        "archive": reference_path.name,
        "archive_sha256": _sha256(reference_path),
        "arrays": {
            key: {
                "shape": list(raw[key].shape),
                "dtype": str(raw[key].dtype),
            }
            for key in REFERENCE_KEYS
        },
        "state_count": int(raw["pks_state"].shape[0]),
        "forensic_self_energy_sign_convention": _forensic_section(metadata),
        "policy": (
            "Immutable frozen K-type comparator artifact. Chinook is a "
            "behavioral comparison target, never a correctness oracle; "
            "independent analytic and invariant evidence adjudicates "
            "correctness."
        ),
    }
    if model_spec_path.is_file():
        manifest["offline_generation"]["model_specification"] = json.loads(
            model_spec_path.read_text()
        )
    manifest_path.write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(f"wrote {reference_path}")
    print(f"  archive_sha256 {manifest['archive_sha256']}")
    print(f"wrote {manifest_path}")


if __name__ == "__main__":
    main()
