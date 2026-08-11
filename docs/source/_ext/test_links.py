"""Resolve compact test references for the Sphinx API pages.

Extended Summary
----------------
The extension scans the test tree once during a documentation build. It maps
test classes and top-level test functions to their complete import paths.
The processor then rewrites compact ``:see:`` targets before Sphinx parses
each docstring.

Routine Listings
----------------
:func:`setup`
    Register the compact test-reference resolver.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

from beartype.typing import Any, Callable, Dict, List, Optional, Tuple
from sphinx.application import Sphinx

_ROLE_RE: re.Pattern[str] = re.compile(
    r":(?P<role>class|func|meth|obj):`(?P<target>~?\.?[\w.]+)`"
)


def _scan_tests(
    repository_root: Path,
) -> Tuple[
    Dict[str, Dict[str, Tuple[str, str]]],
    Dict[str, Tuple[str, str]],
]:
    """PRIVATE: Index each documented test object under the test tree.

    Parameters
    ----------
    repository_root : Path
        Root directory of the repository.

    Returns
    -------
    test_index : Dict[str, Dict[str, Tuple[str, str]]]
        Test objects grouped by their module basename.
    global_names : Dict[str, Tuple[str, str]]
        First registered role and import path for each test object name.

    Notes
    -----
    The scan skips files that the parser cannot read. Sphinx reports a missing
    target later if a skipped file owns a requested object.
    """
    by_module: Dict[str, Dict[str, Tuple[str, str]]] = {}
    global_names: Dict[str, Tuple[str, str]] = {}
    tests_directory: Path = repository_root / "tests"
    test_path: Path
    for test_path in sorted(tests_directory.rglob("test_*.py")):
        relative_path: Path = test_path.relative_to(repository_root)
        module_name: str = ".".join(relative_path.with_suffix("").parts)
        try:
            source_text: str = test_path.read_text(encoding="utf-8")
            tree: ast.Module = ast.parse(source_text, filename=str(test_path))
        except (OSError, SyntaxError):
            continue

        names: Dict[str, Tuple[str, str]] = {}
        node: ast.stmt
        for node in tree.body:
            role: Optional[str] = None
            node_name: Optional[str] = None
            if isinstance(node, ast.ClassDef):
                role = "class"
                node_name = node.name
            elif isinstance(
                node, (ast.FunctionDef, ast.AsyncFunctionDef)
            ) and node.name.startswith("test_"):
                role = "func"
                node_name = node.name
            if role is None or node_name is None:
                continue
            qualified_name: str = f"{module_name}.{node_name}"
            entry: Tuple[str, str] = (role, qualified_name)
            names.setdefault(node_name, entry)
            global_names.setdefault(node_name, entry)
        by_module[test_path.stem] = names
    scan_result: Tuple[
        Dict[str, Dict[str, Tuple[str, str]]],
        Dict[str, Tuple[str, str]],
    ] = (by_module, global_names)
    return scan_result


def _resolve(
    core: str,
    by_module: Dict[str, Dict[str, Tuple[str, str]]],
    global_names: Dict[str, Tuple[str, str]],
) -> Optional[Tuple[str, str]]:
    """PRIVATE: Resolve one stripped target to its role and import path.

    Parameters
    ----------
    core : str
        Compact target without Sphinx display prefixes.
    by_module : Dict[str, Dict[str, Tuple[str, str]]]
        Test objects grouped by their module basename.
    global_names : Dict[str, Tuple[str, str]]
        First registered role and path for each test object name.

    Returns
    -------
    resolved : Optional[Tuple[str, str]]
        The role and import path, or ``None`` when no target matches.

    Notes
    -----
    A module-qualified match takes precedence over a global name match.
    """
    parts: List[str] = core.split(".")
    name: str = parts[-1]
    if "." in core:
        module_basename: str = parts[-2]
        module_entry: Optional[Dict[str, Tuple[str, str]]] = by_module.get(
            module_basename
        )
        if module_entry is not None and name in module_entry:
            resolved_by_module: Tuple[str, str] = module_entry[name]
            return resolved_by_module
    resolved: Optional[Tuple[str, str]] = global_names.get(name)
    return resolved


def _make_processor(
    app: Sphinx,
) -> Callable[[Sphinx, str, str, object, object, List[str]], None]:
    """PRIVATE: Create a processor with a cached test-object index.

    Parameters
    ----------
    app : Sphinx
        Active Sphinx application.

    Returns
    -------
    processor : Callable[[Sphinx, str, str, object, object, List[str]], None]
        Callback for the ``autodoc-process-docstring`` event.

    Notes
    -----
    The callback creates the index on its first docstring event. It reuses
    that index for the remainder of the build.
    """
    state: Dict[str, Any] = {"by_module": None, "global_names": None}

    def _process_docstring(
        _app: Sphinx,
        _what: str,
        _name: str,
        _obj: object,
        _options: object,
        lines: List[str],
    ) -> None:
        """PRIVATE: Rewrite compact test targets in one docstring.

        Parameters
        ----------
        _app : Sphinx
            Sphinx application supplied by the event.
        _what : str
            Documented object category supplied by autodoc.
        _name : str
            Complete name of the documented object.
        _obj : object
            Documented Python object.
        _options : object
            Active autodoc options.
        lines : List[str]
            Mutable lines of the current docstring.

        Notes
        -----
        The processor changes only lines that contain a compact test target.
        """
        if state["by_module"] is None:
            repository_root: Path = Path(app.srcdir).resolve().parents[1]
            state["by_module"], state["global_names"] = _scan_tests(
                repository_root
            )

        def _replace(match: re.Match[str]) -> str:
            """PRIVATE: Replace one compact Sphinx role target.

            Parameters
            ----------
            match : re.Match[str]
                Regular-expression match for one Sphinx role.

            Returns
            -------
            replacement : str
                Complete target when resolution succeeds, or the input text.

            Notes
            -----
            The replacement preserves unresolved targets for Sphinx to report.
            """
            target: str = match.group("target")
            core: str = target.lstrip("~.")
            resolved: Optional[Tuple[str, str]] = _resolve(
                core,
                state["by_module"],
                state["global_names"],
            )
            if resolved is None:
                unresolved: str = match.group(0)
                return unresolved
            role: str
            qualified_name: str
            role, qualified_name = resolved
            replacement: str = f":{role}:`~{qualified_name}`"
            return replacement

        index: int
        line: str
        for index, line in enumerate(lines):
            if "test_" in line and _ROLE_RE.search(line) is not None:
                lines[index] = _ROLE_RE.sub(_replace, line)

    processor: Callable[
        [Sphinx, str, str, object, object, List[str]], None
    ] = _process_docstring
    return processor


def setup(app: Sphinx) -> Dict[str, object]:
    """Register the compact test-reference resolver.

    Parameters
    ----------
    app : Sphinx
        Active Sphinx application.

    Returns
    -------
    metadata : Dict[str, object]
        Extension version and parallel-read safety metadata.

    Notes
    -----
    Sphinx invokes this function when it loads the extension.
    """
    app.connect("autodoc-process-docstring", _make_processor(app))
    metadata: Dict[str, object] = {
        "version": "1.0",
        "parallel_read_safe": True,
    }
    return metadata


__all__: list[str] = ["setup"]
