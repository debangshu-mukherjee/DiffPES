"""Validate the executable tutorial source pairs.

Extended Summary
----------------
The documentation keeps stripped ``.ipynb`` files where Sphinx can discover
them and reviewable Jupytext percent scripts in the repository ``tutorials``
directory. This module verifies that both sides exist, contain identical input
cells, and keep execution results out of version control.

Run the gate directly::

    .venv/bin/python tests/_tutorials.py

Routine Listings
----------------
:func:`discover_tutorial_pairs`
    Return paired notebook and percent-script paths.
:func:`check_tutorial_pairs`
    Return every pairing or output-policy defect.
:func:`strip_tutorial_outputs`
    Remove execution counts and outputs from registered notebooks.
:func:`main`
    Run the tutorial gate and return a process exit status.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from beartype.typing import Any, Dict, List, Tuple

REPOSITORY_ROOT: Path = Path(__file__).resolve().parents[1]
NOTEBOOK_DIRECTORY: Path = Path("docs/source/tutorials")
SCRIPT_DIRECTORY: Path = Path("tutorials")


def discover_tutorial_pairs(
    repository_root: Path,
) -> Tuple[List[Tuple[Path, Path]], List[str]]:
    """Return paired notebook and percent-script paths.

    Parameters
    ----------
    repository_root : Path
        Root that contains the documentation and tutorial directories.

    Returns
    -------
    result : Tuple[List[Tuple[Path, Path]], List[str]]
        Discovered ``(notebook, script)`` pairs and structural defects.
    """
    notebook_root: Path = repository_root / NOTEBOOK_DIRECTORY
    script_root: Path = repository_root / SCRIPT_DIRECTORY
    notebooks: Dict[Path, Path] = {
        path.relative_to(notebook_root).with_suffix(""): path
        for path in notebook_root.rglob("*.ipynb")
    }
    scripts: Dict[Path, Path] = {
        path.relative_to(script_root).with_suffix(""): path
        for path in script_root.rglob("*.py")
    }
    defects: List[str] = []
    missing_scripts: List[Path] = sorted(notebooks.keys() - scripts.keys())
    missing_notebooks: List[Path] = sorted(scripts.keys() - notebooks.keys())
    name: Path
    for name in missing_scripts:
        defects.append(f"missing percent script for {name}.ipynb")
    for name in missing_notebooks:
        defects.append(f"missing notebook for {name}.py")
    pairs: List[Tuple[Path, Path]] = [
        (notebooks[name], scripts[name])
        for name in sorted(notebooks.keys() & scripts.keys())
    ]
    if not pairs:
        defects.append("no paired tutorials were discovered")
    result: Tuple[List[Tuple[Path, Path]], List[str]] = (pairs, defects)
    return result


def _cell_inputs(notebook: Any) -> Tuple[Tuple[str, str], ...]:
    """PRIVATE: Return normalized cell types and sources from one notebook.

    Parameters
    ----------
    notebook : Any
        Notebook node returned by Jupytext or nbformat.

    Returns
    -------
    inputs : Tuple[Tuple[str, str], ...]
        Ordered cell types and input sources without trailing whitespace.
    """
    inputs: Tuple[Tuple[str, str], ...] = tuple(
        (str(cell.cell_type), str(cell.source).rstrip())
        for cell in notebook.cells
    )
    return inputs


def check_tutorial_pairs(repository_root: Path) -> List[str]:
    """Return every pairing or output-policy defect.

    Parameters
    ----------
    repository_root : Path
        Root that contains the documentation and tutorial directories.

    Returns
    -------
    defects : List[str]
        Empty for synchronized, output-free pairs. Otherwise, one message for
        each missing side, input mismatch, kernel defect, or retained output.
    """
    import jupytext
    import nbformat

    pairs: List[Tuple[Path, Path]]
    defects: List[str]
    pairs, defects = discover_tutorial_pairs(repository_root)
    notebook_path: Path
    script_path: Path
    for notebook_path, script_path in pairs:
        notebook: Any = nbformat.read(notebook_path, as_version=4)
        script_notebook: Any = jupytext.read(script_path)
        relative_notebook: Path = notebook_path.relative_to(repository_root)
        relative_script: Path = script_path.relative_to(repository_root)
        if _cell_inputs(notebook) != _cell_inputs(script_notebook):
            defects.append(
                f"tutorial inputs differ: {relative_notebook} != "
                f"{relative_script}"
            )
        kernelspec: Any = notebook.metadata.get("kernelspec", {})
        if kernelspec.get("name") != "python3":
            defects.append(f"missing python3 kernelspec: {relative_notebook}")
        index: int
        cell: Any
        for index, cell in enumerate(notebook.cells):
            if cell.cell_type != "code":
                continue
            if cell.get("execution_count") is not None or cell.get("outputs"):
                defects.append(
                    f"committed output in {relative_notebook} cell {index}"
                )
    return defects


def strip_tutorial_outputs(repository_root: Path) -> List[Path]:
    """Remove execution counts and outputs from registered notebooks.

    Parameters
    ----------
    repository_root : Path
        Root that contains the documentation tutorial directory.

    Returns
    -------
    changed : List[Path]
        Notebook paths rewritten because they retained execution state.
    """
    notebook_root: Path = repository_root / NOTEBOOK_DIRECTORY
    changed: List[Path] = []
    notebook_path: Path
    for notebook_path in sorted(notebook_root.rglob("*.ipynb")):
        notebook: Dict[str, Any] = json.loads(notebook_path.read_text())
        modified: bool = False
        cell: Dict[str, Any]
        for cell in notebook.get("cells", []):
            if cell.get("cell_type") != "code":
                continue
            if cell.get("execution_count") is not None:
                cell["execution_count"] = None
                modified = True
            if cell.get("outputs"):
                cell["outputs"] = []
                modified = True
        if modified:
            notebook_path.write_text(
                json.dumps(notebook, indent=1, ensure_ascii=False) + "\n"
            )
            changed.append(notebook_path)
    return changed


def main() -> int:
    """Run the tutorial gate and return a process exit status.

    Returns
    -------
    status : int
        Zero means every tutorial pair matches and contains no output. One
        reports defects, and two reports invalid command-line arguments.
    """
    arguments: List[str] = sys.argv[1:]
    if arguments not in ([], ["--strip"]):
        print("usage: tests/_tutorials.py [--strip]")
        return 2
    changed: List[Path] = (
        strip_tutorial_outputs(REPOSITORY_ROOT) if arguments else []
    )
    path: Path
    for path in changed:
        print(f"stripped outputs: {path.relative_to(REPOSITORY_ROOT)}")
    defects: List[str] = check_tutorial_pairs(REPOSITORY_ROOT)
    print(f"tutorial defects: {len(defects)}")
    defect: str
    for defect in defects:
        print(f"- {defect}")
    status: int = 1 if defects else 0
    return status


if __name__ == "__main__":
    sys.exit(main())
