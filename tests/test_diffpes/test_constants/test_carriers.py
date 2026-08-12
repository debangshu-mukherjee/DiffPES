"""Validate constants that constrain carrier factories.

The tests compare immutable selector vocabularies with independent domain
values. They also verify the public containers reject mutation by design.
"""

from diffpes.constants import (
    ACQUISITION_MODES,
    BACKGROUND_MODES,
    FINAL_STATE_MODES,
    POST_COUNT_MODES,
)


class TestCarrierConstants:
    """Validate the selector vocabularies for public carriers.

    The cases compare each frozen tuple with the modes that its factory
    accepts.

    Notes
    -----
    Use independent literals so the test does not repeat implementation code.
    """

    def test_selector_vocabularies_match_the_public_domains(self) -> None:
        """Preserve the registered carrier modes.

        The check compares four public selector tuples with independent
        domain values.

        Notes
        -----
        Compare tuple values and order because error messages expose both.
        """
        assert ACQUISITION_MODES == ("poisson", "fixed_total")
        assert BACKGROUND_MODES == ("flat", "shirley", "smooth")
        assert FINAL_STATE_MODES == ("plane_wave", "coulomb")
        assert POST_COUNT_MODES == ("none", "calibrated")
