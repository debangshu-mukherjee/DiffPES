"""Validate canonical tutorial notebooks and their Markdown exports.

Extended Summary
----------------
The repository ``tutorials`` directory is the canonical tutorial home. It
contains only stripped ``.ipynb`` notebooks. The documentation directory
``docs/source/tutorials`` carries one executed Markdown export, with code
and outputs, for each canonical notebook, beside the Markdown-only tutorial
pages. This module verifies that every canonical notebook is output-free,
declares the ``python3`` kernel, and has a present export. Committed
exports are authoritative: the documentation workflow executes only
notebooks that lack one. Contributors regenerate exports locally, where the
untracked DFT data tree is available.

Run the check directly::

    .venv/bin/python tests/_tutorials.py
"""

from __future__ import annotations

import ast
import json
import re
import sys
from pathlib import Path

import nbformat
from beartype.typing import Any, Dict, List, Set

REPOSITORY_ROOT: Path = Path(__file__).resolve().parents[1]
NOTEBOOK_DIRECTORY: Path = Path("tutorials")
EXPORT_DIRECTORY: Path = Path("docs/source/tutorials")
MINIMUM_TUTORIAL_FIGURES: int = 10


def discover_tutorial_notebooks(
    repository_root: Path,
) -> List[Path]:
    """Return the canonical tutorial notebook paths.

    Parameters
    ----------
    repository_root : Path
        Root that contains the canonical tutorial directory.

    Returns
    -------
    notebooks : List[Path]
        Sorted ``.ipynb`` paths under the canonical tutorial directory.

    Notes
    -----
    The discovery includes notebooks in nested canonical directories.
    """
    notebook_root: Path = repository_root / NOTEBOOK_DIRECTORY
    notebooks: List[Path] = sorted(notebook_root.rglob("*.ipynb"))
    return notebooks


def _layout_defects(
    repository_root: Path,
    notebooks: List[Path],
) -> List[str]:
    """PRIVATE: Return defects in the tutorial directory layout.

    Parameters
    ----------
    repository_root : Path
        Root that contains the tutorial and documentation directories.
    notebooks : List[Path]
        Discovered canonical notebooks.

    Returns
    -------
    defects : List[str]
        Messages for missing notebooks and forbidden file types.

    Notes
    -----
    The canonical tree accepts only notebooks. The documentation tree rejects
    Python scripts and notebook files.
    """
    defects: List[str] = []
    if not notebooks:
        defects.append("no canonical tutorial notebooks were discovered")
    canonical_path: Path
    for canonical_path in sorted(
        (repository_root / NOTEBOOK_DIRECTORY).rglob("*")
    ):
        if canonical_path.is_file() and canonical_path.suffix != ".ipynb":
            defects.append(
                f"forbidden canonical tutorial file: "
                f"{canonical_path.relative_to(repository_root)}"
            )
    stray: Path
    for stray in sorted((repository_root / EXPORT_DIRECTORY).rglob("*.py")):
        defects.append(
            f"forbidden script in documentation tutorials: "
            f"{stray.relative_to(repository_root)}"
        )
    for stray in sorted((repository_root / EXPORT_DIRECTORY).rglob("*.ipynb")):
        defects.append(
            f"forbidden notebook in documentation tutorials: "
            f"{stray.relative_to(repository_root)}"
        )
    for stray in sorted((repository_root / NOTEBOOK_DIRECTORY).rglob("*.md")):
        defects.append(
            f"forbidden markdown page in canonical tutorials: "
            f"{stray.relative_to(repository_root)}"
        )
    return defects


def _export_defects(
    repository_root: Path,
    notebook_path: Path,
) -> List[str]:
    """PRIVATE: Return defects in one executed Markdown export.

    Parameters
    ----------
    repository_root : Path
        Root that contains the tutorial and documentation directories.
    notebook_path : Path
        Canonical notebook that owns the export.

    Returns
    -------
    defects : List[str]
        Messages for a missing export or inconsistent image assets.

    Notes
    -----
    The check rejects both missing referenced images and unreferenced images.
    """
    defects: List[str] = []
    relative_notebook: Path = notebook_path.relative_to(repository_root)
    relative_export: Path = notebook_path.relative_to(
        repository_root / NOTEBOOK_DIRECTORY
    ).with_suffix(".md")
    export_path: Path = repository_root / EXPORT_DIRECTORY / relative_export
    if not export_path.exists():
        defects.append(f"missing markdown export for {relative_notebook}")
        return defects

    export_text: str = export_path.read_text(encoding="utf-8")
    asset_directory: Path = export_path.with_name(f"{export_path.stem}_files")
    referenced_assets: List[Path] = [
        export_path.parent / match
        for match in re.findall(
            rf"\((?P<asset>{re.escape(export_path.stem)}_files/[^)]+)\)",
            export_text,
        )
    ]
    referenced_asset: Path
    for referenced_asset in referenced_assets:
        if not referenced_asset.is_file():
            defects.append(
                f"missing export asset for {relative_notebook}: "
                f"{referenced_asset.name}"
            )
    actual_assets: List[Path] = (
        sorted(path for path in asset_directory.rglob("*") if path.is_file())
        if asset_directory.is_dir()
        else []
    )
    referenced_set: Set[Path] = set(referenced_assets)
    if len(referenced_set) < MINIMUM_TUTORIAL_FIGURES:
        defects.append(
            f"tutorial has {len(referenced_set)} executed figures; "
            f"requires at least {MINIMUM_TUTORIAL_FIGURES}: "
            f"{relative_notebook}"
        )
    orphaned_asset: Path
    for orphaned_asset in actual_assets:
        if orphaned_asset not in referenced_set:
            defects.append(
                f"orphaned export asset for {relative_notebook}: "
                f"{orphaned_asset.name}"
            )
    return defects


def _notebook_defects(
    repository_root: Path,
    notebook_path: Path,
) -> List[str]:
    """PRIVATE: Return structural and output defects in one notebook.

    Parameters
    ----------
    repository_root : Path
        Root that contains the tutorial and documentation directories.
    notebook_path : Path
        Canonical notebook to inspect.

    Returns
    -------
    defects : List[str]
        Messages for metadata, structure, code, or committed-output defects.

    Notes
    -----
    Tutorial code calls the public API. It does not define local functions or
    classes.
    """
    defects: List[str] = []
    relative_notebook: Path = notebook_path.relative_to(repository_root)
    notebook: Any = nbformat.read(notebook_path, as_version=4)
    kernelspec: Any = notebook.metadata.get("kernelspec", {})
    if kernelspec.get("name") != "python3":
        defects.append(f"missing python3 kernelspec: {relative_notebook}")
    if not notebook.cells or notebook.cells[0].cell_type != "markdown":
        defects.append(f"missing opening markdown cell: {relative_notebook}")
    else:
        opening_text: str = str(notebook.cells[0].source).strip()
        opening_lines: List[str] = opening_text.splitlines()
        if not opening_lines or not opening_lines[0].startswith("# "):
            defects.append(f"missing tutorial title: {relative_notebook}")
        abstract_lines: List[str] = [
            line for line in opening_lines[1:] if line.strip()
        ]
        if not abstract_lines:
            defects.append(f"missing tutorial abstract: {relative_notebook}")
    markdown_text: str = "\n".join(
        str(cell.source)
        for cell in notebook.cells
        if cell.cell_type == "markdown"
    )
    if "\n## " not in f"\n{markdown_text}":
        defects.append(f"missing tutorial sections: {relative_notebook}")
    index: int
    cell: Any
    for index, cell in enumerate(notebook.cells):
        if cell.cell_type != "code":
            continue
        source: str = str(cell.source)
        try:
            tree: ast.Module = ast.parse(source)
        except SyntaxError:
            defects.append(
                f"invalid Python in {relative_notebook} cell {index}"
            )
        else:
            if any(
                isinstance(
                    definition,
                    (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef),
                )
                for definition in ast.walk(tree)
            ):
                defects.append(
                    f"local definition in {relative_notebook} cell {index}"
                )
        if cell.get("execution_count") is not None or cell.get("outputs"):
            defects.append(
                f"committed output in {relative_notebook} cell {index}"
            )
    return defects


def check_tutorials(repository_root: Path) -> List[str]:
    """Return every layout, output-policy, or missing-export defect.

    Parameters
    ----------
    repository_root : Path
        Root that contains the tutorial and documentation directories.

    Returns
    -------
    defects : List[str]
        Empty for compliant canonical notebooks and their exports.

    Notes
    -----
    The check combines directory, export, and notebook inspections.
    """
    notebooks: List[Path] = discover_tutorial_notebooks(repository_root)
    defects: List[str] = _layout_defects(repository_root, notebooks)
    notebook_path: Path
    for notebook_path in notebooks:
        defects.extend(_export_defects(repository_root, notebook_path))
        defects.extend(_notebook_defects(repository_root, notebook_path))
    return defects


def strip_tutorial_outputs(repository_root: Path) -> List[Path]:
    """Remove execution counts and outputs from canonical notebooks.

    Parameters
    ----------
    repository_root : Path
        Root that contains the canonical tutorial directory.

    Returns
    -------
    changed : List[Path]
        Notebook paths rewritten because they retained execution state.

    Notes
    -----
    The serializer preserves cell order and writes deterministic indentation.
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
    """Run the tutorial check and return a process exit status.

    Returns
    -------
    status : int
        Zero means every canonical notebook is output-free with a present
        export. One reports defects, and two reports invalid command-line
        arguments.

    Notes
    -----
    The optional ``--strip`` argument removes output state before validation.
    """
    arguments: List[str] = sys.argv[1:]
    if arguments not in ([], ["--strip"]):
        print("usage: tests/_tutorials.py [--strip]")
        returned: int = 2
        return returned
    changed: List[Path] = (
        strip_tutorial_outputs(REPOSITORY_ROOT) if arguments else []
    )
    path: Path
    for path in changed:
        print(f"stripped outputs: {path.relative_to(REPOSITORY_ROOT)}")
    defects: List[str] = check_tutorials(REPOSITORY_ROOT)
    defect: str
    for defect in defects:
        print(defect)
    print(f"tutorial defects: {len(defects)}")
    status: int = 1 if defects else 0
    return status


if __name__ == "__main__":
    sys.exit(main())
