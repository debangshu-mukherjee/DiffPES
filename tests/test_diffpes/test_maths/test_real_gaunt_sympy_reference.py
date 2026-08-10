"""Certify real-Gaunt coefficients against a frozen exact SymPy table.

The tests compare every dense physical coordinate and preserve the offline
generator version and authority metadata.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import TextIO

import numpy as np
from beartype.typing import Dict

from diffpes.maths.gaunt import gaunt_lookup

REFERENCE_PATH: Path = (
    Path(__file__).parents[1]
    / "_reference_data"
    / "real_gaunt_sympy_reference.csv"
)


class TestSympyGauntReference:
    """Compare the complete production domain with independent exact values."""

    def test_dense_table_matches_exact_sympy_authority(self) -> None:
        """Match every physical dense-table coordinate and exact zero.

        The test checks allowed coefficients and forbidden selection entries.

        Notes
        -----
        It reads the frozen CSV and compares each row with public lookup.
        """
        stream: TextIO
        with REFERENCE_PATH.open(encoding="utf-8", newline="") as stream:
            rows: list[Dict[str, str]] = list(csv.DictReader(stream))

        assert len(rows) == 2700
        zero_count: int = 0
        nonzero_count: int = 0
        row: Dict[str, str]
        for row in rows:
            actual: float = gaunt_lookup(
                int(row["l"]),
                int(row["m"]),
                int(row["q"]),
                int(row["l_prime"]),
                int(row["m_prime"]),
            )
            if row["is_exact_zero"] == "1":
                zero_count += 1
                assert actual == 0.0, row
            else:
                nonzero_count += 1
                np.testing.assert_allclose(
                    actual,
                    float(row["decimal_50"]),
                    rtol=1.0e-14,
                    atol=0.0,
                    err_msg=(
                        f"SymPy exact value {row['exact_sympy']} at "
                        f"(l,m,q,l',m')="
                        f"({row['l']},{row['m']},{row['q']},"
                        f"{row['l_prime']},{row['m_prime']})"
                    ),
                )
        assert zero_count > nonzero_count > 0

    def test_frozen_reference_pins_sympy_provenance(self) -> None:
        """Require one versioned exact authority across the frozen table.

        The test checks uniform generator and symbolic-authority metadata.

        Notes
        -----
        It collects both metadata columns from every frozen CSV row.
        """
        stream: TextIO
        with REFERENCE_PATH.open(encoding="utf-8", newline="") as stream:
            rows: list[Dict[str, str]] = list(csv.DictReader(stream))

        assert {row["sympy_version"] for row in rows} == {"1.14.0"}
        assert {row["authority"] for row in rows} == {
            "sympy.physics.wigner.real_gaunt"
        }
