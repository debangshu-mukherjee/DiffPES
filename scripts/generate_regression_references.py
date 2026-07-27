"""Generate deterministic WP6.1 forward-model regression references.

Extended Summary
----------------
Builds the three CPU/x64 behavioral baselines first established before the
Equinox migration. Plan 04 intentionally repins the tight-binding cases to the
basis-position gauge and carrier-native orbital bases. The archives capture
behavior rather than independent physics truth and use deterministic ZIP
metadata so unchanged arrays produce identical files and SHA-256 digests.
"""

import hashlib
import io
import platform
import sys
import zipfile
from pathlib import Path

import jax
import numpy as np

import diffpes

_REPOSITORY_ROOT: Path = Path(__file__).resolve().parents[1]
_TESTS_DIRECTORY: Path = _REPOSITORY_ROOT / "tests"
sys.path.insert(0, str(_REPOSITORY_ROOT))

from diffpes.simul import simulate_novice  # noqa: E402
from diffpes.types import ArpesSpectrum  # noqa: E402
from tests._factories import (  # noqa: E402
    toy_band_structure,
    toy_orbital_projection,
    toy_simulation_params,
)

_REFERENCE_DIRECTORY: Path = (
    _TESTS_DIRECTORY / "test_diffpes" / "_reference_data"
)
_SEED: int = 20260713
_FIXED_ZIP_TIME: tuple[int, int, int, int, int, int] = (2026, 7, 19, 0, 0, 0)
_PLAN04_FACTORY_MANIFEST: tuple[str, ...] = (
    "- `plan04_chinook_tightb_reference`: offline Chinook 0.1.1 compatibility",
    "  outputs for the independently C-gated graphene, square-lattice "
    "Rashba, and",
    "  atomic t2g+SOC models. The generator and isolated environment freeze "
    "live",
    "  outside the DiffPES repository under",
    "  `diffpes-plans/verification/tightb/`; pytest reads only this inert "
    "JSON.",
    "- `plan04_wannier90_wse2_reference`: independent NumPy parsing, Fourier",
    "  assembly, and eigensolution of the publicly distributed dynamics-w90",
    "  `data/WSe2_soc/wse2_soc_11bnd_hr.dat` at Γ and reduced-coordinate",
    "  X = (1/2, 0, 0). The exact normative input is stored losslessly "
    "compressed;",
    "  its decompressed SHA-256 authenticates the local public snapshot.",
)
_PLAN04_ARTIFACT_MANIFEST: tuple[str, ...] = (
    "### `plan04_chinook_tightb_reference.json`",
    "",
    "- Classification: Plan 04 gate 04.G6, K-type behavioral compatibility "
    "only",
    "- Chinook commit: `24913de8cc5b8c162f7c1b4acc64bd1b54dd548b`",
    "- Isolated-environment SHA-256:",
    "  `6d00cb4df251508b6392273b1df166f6a17abe8f6691cffead45c636e8ef2531`",
    "- Artifact SHA-256:",
    "  `db52d72562f2efb49d25f9ce2b9affefed1af6f6fac927d1e20f9bb96f1510dc`",
    "- Arrays encoded as JSON numbers:",
    "  - graphene eigenvalues: shape `(33, 2)`, eV",
    "  - square-lattice Rashba eigenvalues: shape `(5, 2)`, eV",
    "  - atomic t2g+SOC eigenvalues: shape `(3, 6)`, eV",
    "",
    "### `plan04_wse2_soc_11bnd_hr.dat.xz`",
    "",
    "- Classification: Plan 04 gate 04.G7, publicly distributed "
    "normative-format",
    "  input",
    "- Upstream repository:",
    "  `https://github.com/michaelschueler/dynamics-w90`",
    "- Upstream snapshot path: `data/WSe2_soc/wse2_soc_11bnd_hr.dat`",
    "- Upstream commit:",
    "  `6f6d99e7fe4b2839a735c609d7df19d1886e8deb` (byte-for-byte verified)",
    "- License qualification: the upstream repository displays no license, "
    "so no",
    "  license grant is claimed; only this normative input crosses the",
    "  independent-implementation boundary",
    "- Decompressed size: `5,543,022` bytes",
    "- Decompressed SHA-256:",
    "  `8ea8140e4fb3d1e56c188d5d680ab077b9ad57070f9205c7365cbb24a7c40dd1`",
    "- Compressed SHA-256:",
    "  `756fdcf2541aa75dad69ae172327fd5cdf6ba044812c918efb9c62a690ece9d4`",
    "",
    "### `plan04_wannier90_wse2_reference.json`",
    "",
    "- Classification: Plan 04 gate 04.G7, K-type published-input companion",
    "  benchmark; normative-format and analytic gates remain authoritative",
    "- Generator:",
    "  `diffpes-plans/verification/tightb/gen_wannier90_wse2_reference.py`",
    "- Generator SHA-256:",
    "  `9bea0278924325526d458094ecfad5b7896d86bfca31c17505f6dd9cf174bac8`",
    "- Artifact SHA-256:",
    "  `afd95f0e6f26771b10e6d825f4e487f88bab0bdc5b326348d43bb6a24194d18c`",
    "- Arrays encoded as JSON numbers:",
    "  - Γ eigenvalues: shape `(22,)`, eV",
    "  - X = `(0.5, 0.0, 0.0)` eigenvalues: shape `(22,)`, eV",
)


def _spectrum_arrays(spectrum: ArpesSpectrum) -> dict[str, np.ndarray]:
    """Convert one spectrum to named NumPy reference arrays."""
    arrays: dict[str, np.ndarray] = {
        "leaf_000_intensity": np.asarray(spectrum.intensity),
        "leaf_001_energy_axis": np.asarray(spectrum.energy_axis),
    }
    return arrays


def build_payloads() -> dict[str, dict[str, np.ndarray]]:
    """Build the retained fixed-seed incoherent reference payload."""
    key: jax.Array = jax.random.key(_SEED)
    novice_spectrum: ArpesSpectrum = simulate_novice(
        toy_band_structure(key),
        toy_orbital_projection(key),
        toy_simulation_params(fidelity=512),
    )
    payloads: dict[str, dict[str, np.ndarray]] = {
        "novice_toy": _spectrum_arrays(novice_spectrum),
    }
    return payloads


def _write_deterministic_npz(
    path: Path,
    arrays: dict[str, np.ndarray],
) -> None:
    """Write an NPZ with stable member order, timestamps, and permissions."""
    with zipfile.ZipFile(path, mode="w") as archive:
        for name in sorted(arrays):
            buffer = io.BytesIO()
            np.save(buffer, arrays[name], allow_pickle=False)
            member = zipfile.ZipInfo(f"{name}.npy", _FIXED_ZIP_TIME)
            member.compress_type = zipfile.ZIP_DEFLATED
            member.external_attr = 0o100644 << 16
            archive.writestr(member, buffer.getvalue())


def _sha256(path: Path) -> str:
    """Calculate the SHA-256 digest of one generated artifact."""
    digest: str = hashlib.sha256(path.read_bytes()).hexdigest()
    return digest


def _manifest(payloads: dict[str, dict[str, np.ndarray]]) -> str:
    """Render provenance, array metadata, and hashes for all artifacts."""
    lines: list[str] = [
        "# WP6.1 regression-reference manifest",
        "",
        "> These files pin deterministic behavior, not independent physics",
        "> truth.",
        "> The tight-binding cases were repinned for Plan 04's basis-position",
        "> gauge and carrier-native orbital bases.",
        "> Regenerate only with a stated physics or migration",
        "> justification.",
        "",
        "- Generation date: 2026-07-22",
        f"- Seed: `{_SEED}`",
        "- Device policy: CPU, JAX x64 enabled",
        f"- Platform: `{platform.platform()}`",
        f"- Python: `{platform.python_version()}`",
        f"- diffpes: `{diffpes.__version__}`",
        f"- JAX: `{jax.__version__}`",
        f"- NumPy: `{np.__version__}`",
        "",
        "## Factory calls",
        "",
        "- `novice_toy`: `simulate_novice(toy_band_structure(key), "
        "toy_orbital_projection(key), "
        "toy_simulation_params(fidelity=512))`",
        *_PLAN04_FACTORY_MANIFEST,
        "",
        "## Artifacts",
        "",
    ]
    for artifact_name, arrays in payloads.items():
        artifact_path: Path = _REFERENCE_DIRECTORY / f"{artifact_name}.npz"
        lines.extend(
            [
                f"### `{artifact_name}.npz`",
                "",
                f"- SHA-256: `{_sha256(artifact_path)}`",
                "- Arrays:",
            ]
        )
        for array_name in sorted(arrays):
            array: np.ndarray = arrays[array_name]
            lines.append(
                f"  - `{array_name}`: shape `{array.shape}`, dtype "
                f"`{array.dtype}`"
            )
        lines.append("")
    lines.extend(_PLAN04_ARTIFACT_MANIFEST)
    manifest: str = "\n".join(lines)
    return manifest


def main() -> None:
    """Generate, verify, and document the deterministic references."""
    _REFERENCE_DIRECTORY.mkdir(parents=True, exist_ok=True)
    first_payloads: dict[str, dict[str, np.ndarray]] = build_payloads()
    second_payloads: dict[str, dict[str, np.ndarray]] = build_payloads()
    for artifact_name, first_arrays in first_payloads.items():
        second_arrays: dict[str, np.ndarray] = second_payloads[artifact_name]
        for array_name, first_array in first_arrays.items():
            np.testing.assert_allclose(
                first_array,
                second_arrays[array_name],
                rtol=1e-12,
                atol=0.0,
            )
        artifact_path: Path = _REFERENCE_DIRECTORY / f"{artifact_name}.npz"
        _write_deterministic_npz(artifact_path, first_arrays)
    manifest_path: Path = _REFERENCE_DIRECTORY / "MANIFEST.md"
    manifest_path.write_text(_manifest(first_payloads).rstrip() + "\n")


if __name__ == "__main__":
    main()
