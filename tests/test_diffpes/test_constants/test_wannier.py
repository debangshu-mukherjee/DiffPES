"""Validate constants for Wannier90 and hopping-list formats.

The tests compare stable suffixes and record widths with independent format
values used by the parsers.
"""

from diffpes.constants import (
    HOPPING_LIST_COMPLEX_FIELDS,
    HOPPING_LIST_REAL_FIELDS,
    WANNIER_HR_SUFFIX,
    WANNIER_TB_SUFFIX,
)


class TestWannierConstants:
    """Validate the registered Wannier and hopping-list formats.

    The cases compare file suffixes and field counts with their documented
    text layouts.

    Notes
    -----
    Use independent literals to expose accidental parser-format changes.
    """

    def test_format_constants_match_registered_text_layouts(self) -> None:
        """Preserve suffixes and hopping record widths.

        The check compares the four public format constants with independent
        values.

        Notes
        -----
        Verify both real and complex hopping-list record layouts.
        """
        assert WANNIER_HR_SUFFIX == "_hr.dat"
        assert WANNIER_TB_SUFFIX == "_tb.dat"
        assert HOPPING_LIST_REAL_FIELDS == 6
        assert HOPPING_LIST_COMPLEX_FIELDS == 7
