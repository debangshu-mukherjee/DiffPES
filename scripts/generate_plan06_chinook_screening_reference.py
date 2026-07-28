"""Generate the inert pinned-Chinook screening sample for Plan 06 G4.

This script is for the isolated Chinook environment only. DiffPES tests read
the frozen JSON and never import or execute Chinook.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import sys
import types
from pathlib import Path
from typing import Any

CHINOOK_COMMIT: str = "24913de8cc5b8c162f7c1b4acc64bd1b54dd548b"
MODULE_SHA256: str = (
    "93341ec1b7cee8cd4982a968391294d91d4669d44a0a3e8fc51b0a9adc34fbf3"
)
CONFIG_SHA256: str = (
    "2e4d1977365dfd17df5b4738f9782298050be007e97d131d3b26d150da200b04"
)
ROUND_DIGITS: int = 12
CASES: tuple[tuple[int, int, int, str], ...] = (
    (6, 2, 1, "2p"),
    (7, 2, 1, "2p"),
    (8, 2, 1, "2p"),
    (24, 3, 2, "3d"),
    (26, 3, 2, "3d"),
    (26, 4, 0, "4s"),
)


def _sha256(path: Path) -> str:
    """Return the SHA-256 digest of one source file."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        block: bytes
        for block in iter(lambda: stream.read(65536), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_chinook_module(source: Path) -> Any:
    """Load Chinook's module from an authenticated source checkout."""
    module_path: Path = source / "electron_configs.py"
    config_path: Path = source / "electron_configs.txt"
    if _sha256(module_path) != MODULE_SHA256:
        message: str = "unexpected pinned Chinook electron_configs.py hash"
        raise RuntimeError(message)
    if _sha256(config_path) != CONFIG_SHA256:
        message = "unexpected pinned Chinook electron_configs.txt hash"
        raise RuntimeError(message)
    package = types.ModuleType("chinook")
    package.__path__ = [str(source)]
    sys.modules["chinook"] = package
    module: Any = importlib.import_module("chinook.electron_configs")
    return module


def generate(source: Path, output: Path) -> None:
    """Evaluate the registered sample and write inert JSON."""
    module: Any = _load_chinook_module(source)
    rows: list[dict[str, int | float | str]] = []
    atomic_number: int
    n_value: int
    l_value: int
    orbital: str
    for atomic_number, n_value, l_value, orbital in CASES:
        raw_value: float = float(module.Z_eff(atomic_number, orbital))
        rows.append(
            {
                "atomic_number": atomic_number,
                "n": n_value,
                "l": l_value,
                "chinook_orbital": orbital,
                "chinook_raw_repr": repr(raw_value),
                "rounded_zeff": round(raw_value, ROUND_DIGITS),
            }
        )
    artifact: dict[str, object] = {
        "gate": "06.G4",
        "chinook_commit": CHINOOK_COMMIT,
        "chinook_module_sha256": MODULE_SHA256,
        "chinook_configuration_sha256": CONFIG_SHA256,
        "round_digits": ROUND_DIGITS,
        "samples": rows,
        "policy": (
            "Generated offline in the isolated pinned Chinook environment; "
            "DiffPES pytest consumes inert JSON only."
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(artifact, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    """Parse source and output locations and generate the artifact."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--chinook-source",
        type=Path,
        required=True,
        help="Path containing Chinook electron_configs.py and its text table.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "tests/test_diffpes/_reference_data/"
            "plan06_chinook_screening_reference.json"
        ),
    )
    arguments: argparse.Namespace = parser.parse_args()
    generate(arguments.chinook_source, arguments.output)


if __name__ == "__main__":
    main()
