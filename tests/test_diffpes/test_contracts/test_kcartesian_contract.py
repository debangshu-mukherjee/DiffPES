"""Validate the fractional-to-Cartesian momentum contract.

Extended Summary
----------------
The test parses every source module and detects direct normalization of
fractional momentum. It pins an empty allowlist so no legacy site can return.
"""

import ast
import re
from pathlib import Path

_FRACTIONAL_K_NAME: re.Pattern[str] = re.compile(
    r"^(?:k_frac|kpoints_frac|k_crystal|kpoints)$"
)
_LEGACY_ALLOWLIST: frozenset[Path] = frozenset()


def _matching_names(node: ast.AST) -> set[str]:
    """Return fractional-momentum names below one syntax node.

    The helper walks the supplied expression and selects identifiers that
    match the contract's pinned fractional-coordinate vocabulary.

    Parameters
    ----------
    node : ast.AST
        Syntax node to inspect.

    Returns
    -------
    names : set[str]
        Matching fractional-momentum identifiers.
    """
    names: set[str] = {
        child.id
        for child in ast.walk(node)
        if isinstance(child, ast.Name)
        and _FRACTIONAL_K_NAME.fullmatch(child.id) is not None
    }
    return names


def _is_norm_call(node: ast.AST) -> bool:
    """Return whether a syntax node calls a norm function.

    The helper recognizes attribute and direct calls whose function name is
    ``norm`` or ``safe_norm``.

    Parameters
    ----------
    node : ast.AST
        Syntax node to inspect.

    Returns
    -------
    is_norm : bool
        True when the syntax node is a recognized norm call.
    """
    if not isinstance(node, ast.Call):
        return False
    function: ast.expr = node.func
    is_norm: bool = (
        isinstance(function, ast.Name) and function.id in {"norm", "safe_norm"}
    ) or (
        isinstance(function, ast.Attribute)
        and function.attr in {"norm", "safe_norm"}
    )
    return is_norm


def _fractional_norm_lines(tree: ast.AST) -> set[int]:
    """Return source lines that normalize fractional momentum.

    The helper finds norm calls on pinned names and division expressions that
    divide a pinned name by its own norm.

    Parameters
    ----------
    tree : ast.AST
        Parsed source-module syntax tree.

    Returns
    -------
    lines : set[int]
        One-based source lines containing contract violations.
    """
    lines: set[int] = set()
    node: ast.AST
    for node in ast.walk(tree):
        if _is_norm_call(node):
            call: ast.Call = node
            if call.args and _matching_names(call.args[0]):
                lines.add(node.lineno)
        if not isinstance(node, ast.BinOp) or not isinstance(node.op, ast.Div):
            continue
        if not _is_norm_call(node.right):
            continue
        denominator: ast.Call = node.right
        if not denominator.args:
            continue
        numerator_names: set[str] = _matching_names(node.left)
        denominator_names: set[str] = _matching_names(denominator.args[0])
        if numerator_names & denominator_names:
            lines.add(node.lineno)
    return lines


def test_fractional_momentum_is_converted_before_normalization() -> None:
    """Reject direct normalization of fractional momentum anywhere.

    The test parses all Python source files, collects matching syntax lines,
    and requires an empty finding set.

    Notes
    -----
    Walk each parsed module once. Compare the finding set with the pinned
    empty allowlist.
    """
    repository_root: Path = Path(__file__).resolve().parents[3]
    source_root: Path = repository_root / "src" / "diffpes"
    findings: dict[Path, set[int]] = {}
    source_path: Path
    for source_path in source_root.rglob("*.py"):
        tree: ast.AST = ast.parse(source_path.read_text(encoding="utf-8"))
        lines: set[int] = _fractional_norm_lines(tree)
        if lines:
            relative_path: Path = source_path.relative_to(source_root)
            findings[relative_path] = lines

    found_files: frozenset[Path] = frozenset(findings)
    assert found_files == _LEGACY_ALLOWLIST, (
        "fractional momentum normalization sites differ from the exact "
        f"allowlist: {findings}"
    )
