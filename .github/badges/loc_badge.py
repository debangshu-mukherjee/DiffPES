"""Regenerate the local lines-of-code badge JSON.

Extended Summary
----------------
Counts logical lines of code in ``src/diffpes`` with pygount and
writes ``.github/badges/loc.json`` in the shields.io endpoint schema.
The local pre-commit hook updates the badge during a normal commit.
No continuous-integration job commits this file.
The file is rewritten only when the count changes, keeping the hook
silent on unrelated commits.

Routine Listings
----------------
:func:`main`
    Count lines of code and rewrite the badge JSON if it changed.
"""

import json
from pathlib import Path

import pygount.analysis
from beartype.typing import Any, Iterable
from pygments.lexers.python import PythonLexer

_REPO_ROOT: Path = Path(__file__).resolve().parents[2]
_BADGE_PATH: Path = _REPO_ROOT / ".github" / "badges" / "loc.json"
_COUNT_TARGET: Path = _REPO_ROOT / "src" / "diffpes"


def _count_python_loc(count_target: Path) -> int:
    """PRIVATE: Count Python lines without Pygments language guessing.

    Parameters
    ----------
    count_target : Path
        Directory containing the Python source tree.

    Returns
    -------
    loc : int
        Pygount source-line total with every ``.py`` file parsed as Python.

    Notes
    -----
    Pygount delegates filename classification to Pygments. When IPython is
    installed, Pygments can classify ordinary ``.py`` files as either Python
    or IPython depending on entry-point discovery. Pygount treats Python
    docstrings as comments but IPython docstrings as code, so the badge count
    would otherwise depend on the invocation environment.
    """
    duplicate_pool: pygount.analysis.DuplicatePool = (
        pygount.analysis.DuplicatePool()
    )
    original_guess_lexer: Any = pygount.analysis.guess_lexer

    def _python_lexer(
        _source_path: str,
        _source_code: str,
    ) -> PythonLexer:
        """PRIVATE: Select the Python lexer for one source file.

        Parameters
        ----------
        _source_path : str
            Path supplied by pygount.
        _source_code : str
            Source text supplied by pygount.

        Returns
        -------
        lexer : PythonLexer
            Fresh Python lexer for the source analysis.

        Notes
        -----
        The fixed lexer removes environment-dependent language guessing.
        """
        lexer: PythonLexer = PythonLexer()
        return lexer

    pygount.analysis.guess_lexer = _python_lexer
    try:
        analyses: Iterable[pygount.analysis.SourceAnalysis] = (
            pygount.analysis.SourceAnalysis.from_file(
                str(source_path),
                group="diffpes",
                duplicate_pool=duplicate_pool,
            )
            for source_path in sorted(count_target.rglob("*.py"))
        )
        loc: int = sum(analysis.source_count for analysis in analyses)
        return loc
    finally:
        pygount.analysis.guess_lexer = original_guess_lexer


def main() -> int:
    r"""Count lines of code and rewrite the badge JSON if it changed.

    Implementation Logic
    --------------------
    1. **Count source lines**::

           loc = str(_count_python_loc(_COUNT_TARGET))

       The fixed Python lexer makes the count independent of plugins.
    2. **Serialize badge data**::

           badge = json.dumps(payload, indent=2) + "\\n"

       The trailing newline keeps the generated JSON text stable.
    3. **Write only changed content**::

           _BADGE_PATH.write_text(badge)

       An unchanged badge leaves the working tree clean.

    Returns
    -------
    exit_code : int
        Zero on success. Pre-commit detects a modified badge file itself.
    """
    loc: str = str(_count_python_loc(_COUNT_TARGET))
    badge: str = (
        json.dumps(
            {
                "schemaVersion": 1,
                "label": "lines of code",
                "message": loc,
                "color": "blue",
            },
            indent=2,
        )
        + "\n"
    )
    if _BADGE_PATH.exists() and _BADGE_PATH.read_text() == badge:
        exit_code: int = 0
    else:
        _BADGE_PATH.write_text(badge)
        print(f"updated {_BADGE_PATH.relative_to(_REPO_ROOT)}: {loc} lines")
        exit_code = 0
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())


__all__: list[str] = ["main"]
