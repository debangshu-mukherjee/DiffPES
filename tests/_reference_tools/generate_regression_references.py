"""Generate the deterministic SciPy spectrum references.

Extended Summary
----------------
This entry point invokes the canonical SciPy generator. The generator writes
the independently assembled true-Voigt authority and its novice comparison.

Routine Listings
----------------
:func:`main`
    Run the canonical SciPy Voigt reference generator.
"""

from __future__ import annotations

import runpy
from collections.abc import Callable
from pathlib import Path

from beartype.typing import Dict, cast


def main() -> None:
    """Run the independent Voigt and novice reference generator.

    Notes
    -----
    The function loads the canonical generator as a script namespace. It then
    invokes that generator's ``main`` function.
    """
    generator_path: Path = Path(__file__).with_name(
        "generate_voigt_scipy_reference.py"
    )
    namespace: Dict[str, object] = runpy.run_path(str(generator_path))
    generate: Callable[[], None] = cast(
        "Callable[[], None]",
        namespace["main"],
    )
    generate()


if __name__ == "__main__":
    main()
