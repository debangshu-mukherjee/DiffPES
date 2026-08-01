"""Generate the frozen independent SymPy authority for Plan 06 G3.

Run this offline with SymPy 1.14 or newer. Package tests consume only the
resulting CSV and therefore do not import SymPy at collection or runtime.
"""

from __future__ import annotations

import argparse
import csv
import importlib
from pathlib import Path
from typing import Any


def generate(output: Path, l_max: int = 4) -> None:
    """Write exact and 50-digit real-Gaunt values over the dense domain."""
    try:
        sp: Any = importlib.import_module("sympy")
        wigner: Any = importlib.import_module("sympy.physics.wigner")
    except ImportError as error:
        message: str = (
            "SymPy is an offline generator dependency; run with "
            "`uv run --with 'sympy>=1.14' "
            "tests/_reference_tools/generate_plan06_gaunt_sympy_reference.py`"
        )
        raise RuntimeError(message) from error
    real_gaunt: Any = wigner.real_gaunt

    rows: list[dict[str, str]] = []
    l_initial: int
    m_initial: int
    q_value: int
    l_final: int
    m_final: int
    for l_initial in range(l_max + 1):
        for m_initial in range(-l_initial, l_initial + 1):
            for q_value in (-1, 0, 1):
                for l_final in range(l_max + 2):
                    for m_final in range(-l_final, l_final + 1):
                        exact: Any = sp.simplify(
                            real_gaunt(
                                l_final,
                                1,
                                l_initial,
                                m_final,
                                q_value,
                                m_initial,
                            )
                        )
                        rows.append(
                            {
                                "l": str(l_initial),
                                "m": str(m_initial),
                                "q": str(q_value),
                                "l_prime": str(l_final),
                                "m_prime": str(m_final),
                                "exact_sympy": str(exact),
                                "decimal_50": str(sp.N(exact, 50)),
                                "is_exact_zero": str(int(exact == 0)),
                                "sympy_version": sp.__version__,
                                "authority": (
                                    "sympy.physics.wigner.real_gaunt"
                                ),
                            }
                        )
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    """Parse generator options and write the frozen CSV."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "tests/test_diffpes/_reference_data/"
            "plan06_real_gaunt_sympy_reference.csv"
        ),
    )
    parser.add_argument("--l-max", type=int, default=4)
    arguments: argparse.Namespace = parser.parse_args()
    if arguments.l_max < 0:
        message: str = "l-max must be non-negative"
        raise ValueError(message)
    generate(arguments.output, arguments.l_max)


if __name__ == "__main__":
    main()
