"""Regenerate the active deterministic spectrum references.

Extended Summary
----------------
The true-Voigt implementation replaced the production-derived pseudo-Voigt novice baseline with an
independently assembled SciPy true-Voigt artifact. This compatibility entry
point delegates to the canonical SciPy reference generator so older developer
workflows cannot recreate the superseded ``novice_toy.npz`` file.

Routine Listings
----------------
:func:`main`
    Run the canonical SciPy Voigt reference generator.
"""

from __future__ import annotations

import runpy
from collections.abc import Callable
from pathlib import Path
from typing import cast


def main() -> None:
    """Run the canonical independent Voigt and novice reference generator."""
    generator_path: Path = Path(__file__).with_name(
        "generate_voigt_scipy_reference.py"
    )
    namespace: dict[str, object] = runpy.run_path(str(generator_path))
    generate: Callable[[], None] = cast(
        "Callable[[], None]",
        namespace["main"],
    )
    generate()


if __name__ == "__main__":
    main()
