"""Validate centralized parser and orbital-ordering constants.

The tests cover immutable parser tokens and canonical VASP channel indices.
"""

from types import MappingProxyType

from diffpes.types import (
    COORDINATE_MODE_TOKENS,
    D_ORBITAL_SLICE,
    L_MAX,
    N_ORBITALS,
    ORBITAL_INDEX,
    P_ORBITAL_SLICE,
)


class TestOrbitalConstants:
    """Validate the shared VASP orbital-ordering constants.

    Indices and slices must describe one nine-orbital basis.

    :see: :data:`~diffpes.types.ORBITAL_INDEX`
    """

    def test_share_one_orbital_ordering(self) -> None:
        """Keep orbital indices and slices aligned.

        The check verifies nine immutable indices and the standard p and d
        slices.

        Notes
        -----
        Compare the public mapping and slice constants with independent
        literal values.
        """
        assert isinstance(ORBITAL_INDEX, MappingProxyType)
        assert len(ORBITAL_INDEX) == N_ORBITALS
        assert slice(1, 4) == P_ORBITAL_SLICE
        assert slice(4, 9) == D_ORBITAL_SLICE


class TestParserConstants:
    """Validate immutable parser tokens and angular-momentum bounds.

    The tests verify shared coordinate selectors and the maximum orbital
    angular momentum. Parsers do not define local values for these conventions.

    :see: :data:`~diffpes.types.COORDINATE_MODE_TOKENS`
    :see: :data:`~diffpes.types.L_MAX`
    """

    def test_tokens_are_immutable_conventions(self) -> None:
        """Preserve the coordinate-token set and maximum angular momentum.

        The check expects a frozen four-token convention and ``L_MAX=4``.

        Notes
        -----
        The test compares the public token container type and contents plus
        the integer
        angular-momentum bound against independent literal values.
        """
        expected_tokens: frozenset[str] = frozenset(
            {"cartesian", "direct", "fractional", "reciprocal"}
        )

        assert isinstance(COORDINATE_MODE_TOKENS, frozenset)
        assert expected_tokens == COORDINATE_MODE_TOKENS
        assert L_MAX == 4
