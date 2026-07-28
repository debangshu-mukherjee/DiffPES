"""Verify deterministic source-line counting for the repository badge."""

import importlib.util
from pathlib import Path
from types import ModuleType

import pygount.analysis


def _load_loc_badge_module() -> ModuleType:
    """Load the badge script without executing its command-line entry point."""
    root: Path = Path(__file__).resolve().parents[1]
    script: Path = root / ".github" / "badges" / "loc_badge.py"
    spec = importlib.util.spec_from_file_location("diffpes_loc_badge", script)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_loc_badge_forces_python_lexer(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Count docstrings consistently even if language guessing is unusable."""
    package: Path = tmp_path / "package"
    package.mkdir()
    (package / "module.py").write_text(
        '"""Module documentation.\n\nMore documentation.\n"""\n\nvalue = 1\n',
        encoding="utf-8",
    )
    (package / "second.py").write_text(
        '"""Second module."""\n\nother = value + 1\n',
        encoding="utf-8",
    )

    def fail_if_guessed(_source_path: str, _source_code: str):
        raise AssertionError("language guessing must be replaced")

    monkeypatch.setattr(pygount.analysis, "guess_lexer", fail_if_guessed)
    module: ModuleType = _load_loc_badge_module()

    assert module._count_python_loc(package) == 2
    assert pygount.analysis.guess_lexer is fail_if_guessed
