"""Validate shared physical, orbital, and parser constants.

The tests cover the canonical orbital order and parser selectors. They also
verify the zero-legacy boundary after constants moved out of types.
"""

from types import MappingProxyType

from beartype.typing import Tuple

import diffpes
from diffpes.constants import (
    COORDINATE_MODE_TOKENS,
    D_ORBITAL_SLICE,
    L_MAX,
    N_ORBITALS,
    ORBITAL_INDEX,
    P_ORBITAL_SLICE,
)


class TestSharedConstants:
    """Validate the shared orbital order and parser selectors.

    The cases compare the public constants with independent basis and parser
    conventions.

    Notes
    -----
    Check immutable container types as well as their values.
    """

    def test_orbital_and_parser_conventions_are_stable(self) -> None:
        """Preserve the shared orbital and coordinate conventions.

        The check verifies nine orbital indices, two slices, four tokens, and
        the angular-momentum bound.

        Notes
        -----
        Compare independent literals with the public constants package.
        """
        expected_tokens: frozenset[str] = frozenset(
            {"cartesian", "direct", "fractional", "reciprocal"}
        )

        assert isinstance(ORBITAL_INDEX, MappingProxyType)
        assert len(ORBITAL_INDEX) == N_ORBITALS
        assert slice(1, 4) == P_ORBITAL_SLICE
        assert slice(4, 9) == D_ORBITAL_SLICE
        assert expected_tokens == COORDINATE_MODE_TOKENS
        assert L_MAX == 4

    def test_types_has_no_legacy_constant_exports(self) -> None:
        """Keep constants absent from the types surface.

        The check samples scientific, parser, schema, numerical, and format
        names across the new and old owners.

        Notes
        -----
        Require each name on constants and reject it on types.
        """
        names: Tuple[str, ...] = (
            "BOHR_TO_ANGSTROM",
            "CERTIFICATE_FORMAT",
            "COORDINATE_MODE_TOKENS",
            "FADDEEVA_WEIDEMAN_COEFFICIENTS",
            "GAUNT_TABLE",
            "KB_EV_PER_K",
            "WANNIER_HR_SUFFIX",
        )

        name: str
        for name in names:
            assert hasattr(diffpes.constants, name)
            assert not hasattr(diffpes.types, name)
