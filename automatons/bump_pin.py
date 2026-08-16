#!/usr/bin/env python3
"""Update diffpes dependency pins in automaton scripts.

The tool reads the project version unless a version argument overrides it.
It updates direct automaton files and reports the changed-file count.
"""

from __future__ import annotations

import argparse
import re
import tomllib
from collections.abc import Sequence
from pathlib import Path
from typing import Any


def _project_version(root: Path) -> str:
    """PRIVATE: Read the declared package version from one project file.

    Parameters
    ----------
    root : Path
        Repository root that contains ``pyproject.toml``.

    Returns
    -------
    version : str
        Declared project version.

    Raises
    ------
    ValueError
        Raised when the project file lacks a text version.

    Notes
    -----
    Parses the TOML document with the standard library. It then checks the
    project version before the rewrite starts.
    """
    project_path: Path = root / "pyproject.toml"
    with project_path.open("rb") as project_file:
        configuration: dict[str, Any] = tomllib.load(project_file)
    project: Any = configuration.get("project")
    version: Any = (
        project.get("version") if isinstance(project, dict) else None
    )
    if not isinstance(version, str) or not version:
        msg: str = "pyproject.toml must define a text project version"
        raise ValueError(msg)
    return version


def _replace_pin(source: str, version: str) -> tuple[str, int]:
    """PRIVATE: Replace diffpes pins in one script source document.

    Parameters
    ----------
    source : str
        Original source text.
    version : str
        Version to place in every matching dependency pin.

    Returns
    -------
    updated_source : str
        Source text with normalized dependency pins.
    replacements : int
        Number of rewritten pins.

    Notes
    -----
    Matches the published dependency form. It normalizes an optional CUDA
    extra to the base package pin.
    """
    updated_source: str
    replacements: int
    updated_source, replacements = re.subn(
        r"diffpes(?:\[cuda\])?==[0-9][^\"']*",
        f"diffpes=={version}",
        source,
    )
    return updated_source, replacements


def _script_paths(root: Path) -> tuple[Path, ...]:
    """PRIVATE: List direct automaton scripts in deterministic order.

    Parameters
    ----------
    root : Path
        Repository root that contains the automaton directory.

    Returns
    -------
    paths : tuple[Path, ...]
        Sorted Python scripts that carry dependency pins.

    Notes
    -----
    Excludes this maintenance tool because it has no diffpes dependency.
    """
    directory: Path = root / "automatons"
    paths: tuple[Path, ...] = tuple(
        path
        for path in sorted(directory.glob("*.py"))
        if path.name != __file__.split("/")[-1]
    )
    return paths


def main(argv: Sequence[str] | None = None) -> int:
    """Update dependency pins for direct automaton scripts.

    The command uses the declared project version by default. A positional
    value lets a release update all pins before the project change lands.

    Parameters
    ----------
    argv : Sequence[str] or None
        Optional command-line arguments. Default ``None`` reads process input.

    Returns
    -------
    exit_code : int
        Zero after every matching script has been updated.

    Notes
    -----
    Reads each script before writing it. Unchanged files stay untouched.
    """
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description="Update diffpes pins in automaton scripts."
    )
    parser.add_argument("version", nargs="?", help="Version to write.")
    parser.add_argument(
        "--root",
        default=Path(__file__).resolve().parents[1],
        type=Path,
        help="Repository root. Default uses this file location.",
    )
    parsed: argparse.Namespace = parser.parse_args(argv)
    root: Path = parsed.root.resolve()
    version: str = parsed.version or _project_version(root)
    changed_files: int = 0
    script_path: Path
    for script_path in _script_paths(root):
        source: str = script_path.read_text(encoding="utf-8")
        updated_source: str
        replacements: int
        updated_source, replacements = _replace_pin(source, version)
        if replacements and updated_source != source:
            script_path.write_text(updated_source, encoding="utf-8")
            changed_files += 1
    print(f"pinned diffpes=={version}; updated {changed_files} file(s)")
    exit_code: int = 0
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
