"""Validate neutral configurations and Slater screening rules.

Extended Summary
----------------
The tests cover the full supported atomic-number domain, measured
configuration exceptions, published worked screening values, and effective
principal-number conversion.
"""

import pytest
from beartype.typing import Dict, Tuple

from diffpes.radial import (
    electron_configuration,
    slater_zeff,
    slater_zeta,
)


class TestElectronConfiguration:
    """Validate :func:`diffpes.radial.electron_configuration`.

    The cases cover every atomic number from hydrogen through lawrencium and
    both invalid table boundaries. They sum all occupancies and inspect four
    representative ground-state exceptions directly.

    :see: :func:`~diffpes.radial.electron_configuration`
    """

    def test_domain_totals_and_ground_state_exceptions(self) -> None:
        """Preserve electron totals and measured exceptional occupations.

        The check covers every supported element and inspects chromium,
        palladium, gadolinium, and lawrencium explicitly.

        Notes
        -----
        Sum each returned occupancy and compare selected subshell rows.
        """
        atomic_number: int
        for atomic_number in range(1, 104):
            configuration: Tuple[Tuple[int, int, int], ...] = (
                electron_configuration(atomic_number)
            )
            assert sum(row[2] for row in configuration) == atomic_number

        chromium: Dict[Tuple[int, int], int] = {
            row[:2]: row[2] for row in electron_configuration(24)
        }
        palladium: Dict[Tuple[int, int], int] = {
            row[:2]: row[2] for row in electron_configuration(46)
        }
        gadolinium: Dict[Tuple[int, int], int] = {
            row[:2]: row[2] for row in electron_configuration(64)
        }
        lawrencium: Dict[Tuple[int, int], int] = {
            row[:2]: row[2] for row in electron_configuration(103)
        }
        assert chromium[(3, 2)] == 5
        assert chromium[(4, 0)] == 1
        assert (5, 0) not in palladium
        assert palladium[(4, 2)] == 10
        assert gadolinium[(4, 3)] == 7
        assert gadolinium[(5, 2)] == 1
        assert lawrencium[(5, 3)] == 14
        assert lawrencium[(7, 1)] == 1

    def test_rejects_atomic_numbers_outside_table(self) -> None:
        """Reject values below hydrogen and above lawrencium.

        The two cases establish both finite boundaries of the static table.

        Notes
        -----
        Require the same explicit domain diagnostic at either boundary.
        """
        with pytest.raises(ValueError, match="1 through 103"):
            electron_configuration(0)
        with pytest.raises(ValueError, match="1 through 103"):
            electron_configuration(104)


class TestSlaterZeff:
    """Validate :func:`diffpes.radial.slater_zeff`.

    The cases cover four published s-, p-, and d-subshell screening values and
    one unoccupied subshell. They compare the worked values at strict decimal
    tolerance and require the documented occupancy error.

    :see: :func:`~diffpes.radial.slater_zeff`
    """

    @pytest.mark.parametrize(
        ("atomic_number", "n", "l", "expected"),
        [
            (6, 2, 1, 3.25),
            (7, 2, 1, 3.90),
            (26, 3, 2, 6.25),
            (26, 4, 0, 3.75),
        ],
    )
    def test_matches_published_worked_values(
        self,
        atomic_number: int,
        n: int,
        l: int,
        expected: float,
    ) -> None:
        """Match representative s, p, and d screening calculations.

        The cases include the four registered worked values.

        Notes
        -----
        Compare decimal-rule outputs exactly at binary float precision.
        """
        actual: float = slater_zeff(atomic_number, n, l)
        assert actual == pytest.approx(expected, rel=0.0, abs=1.0e-14)

    def test_rejects_an_unoccupied_subshell(self) -> None:
        """Reject quantum numbers absent from the neutral configuration.

        Carbon has no occupied 3d subshell.

        Notes
        -----
        Match the occupied-subshell diagnostic.
        """
        with pytest.raises(ValueError, match="must be occupied"):
            slater_zeff(6, 3, 2)


class TestSlaterZeta:
    """Validate :func:`diffpes.radial.slater_zeta`.

    The cases cover carbon 2p and iron 4s effective principal numbers. They
    compare both public exponents with independent effective-charge ratios at
    strict decimal tolerance.

    :see: :func:`~diffpes.radial.slater_zeta`
    """

    def test_uses_effective_principal_number(self) -> None:
        """Divide effective charge by the registered Slater number.

        Carbon 2p uses ``n_star=2`` and iron 4s uses ``n_star=3.7``.

        Notes
        -----
        Compare both ratios with independently evaluated worked values.
        """
        carbon: float = slater_zeta(6, 2, 1)
        iron: float = slater_zeta(26, 4, 0)
        assert carbon == pytest.approx(3.25 / 2.0, rel=0.0, abs=1.0e-14)
        assert iron == pytest.approx(3.75 / 3.7, rel=0.0, abs=1.0e-14)
