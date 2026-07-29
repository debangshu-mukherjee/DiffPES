"""Regenerate the active deterministic spectrum references.

Extended Summary
----------------
Plan 07 replaced the production-derived pseudo-Voigt novice baseline with an
independently assembled SciPy true-Voigt artifact. This compatibility entry
point delegates to the canonical Plan-07 generator so older developer
workflows cannot recreate the superseded ``novice_toy.npz`` file.

Routine Listings
----------------
:func:`main`
    Run the canonical Plan-07 reference generator.
"""

from __future__ import annotations

import runpy
from collections.abc import Callable
from pathlib import Path
from typing import cast


def main() -> None:
    """Run the canonical independent Voigt and novice reference generator."""
    generator_path: Path = Path(__file__).with_name(
        "generate_plan07_voigt_reference.py"
    )
    namespace: dict[str, object] = runpy.run_path(str(generator_path))
    generate: Callable[[], None] = cast(
        "Callable[[], None]",
        namespace["main"],
    )
    generate()


if __name__ == "__main__":
    main()
