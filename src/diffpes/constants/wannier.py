"""Define constants for Wannier90 and Cartesian hopping-list formats.

Extended Summary
----------------
This module owns immutable field counts, suffixes, tolerances, and selector
vocabularies for Wannier90 and Cartesian hopping-list ingestion.

Routine Listings
----------------
:obj:`HOPPING_LIST_COMPLEX_FIELDS`
    Number of fields in a complex Cartesian hopping-list row.
:obj:`HOPPING_LIST_REAL_FIELDS`
    Number of fields in a real Cartesian hopping-list row.
:obj:`WANNIER_CELL_FIELDS`
    Number of integer components in a Wannier translation.
:obj:`WANNIER_CENTRE_CONSISTENCY_TOLERANCE`
    Cartesian tolerance for centres assigned to one atom.
:obj:`WANNIER_CENTRE_NDIM`
    Expected dimensionality of Wannier centre arrays.
:obj:`WANNIER_DEGENERACIES_PER_LINE`
    Maximum degeneracy weights on one Wannier90 line.
:obj:`WANNIER_HERMITICITY_TOLERANCE`
    Absolute tolerance for real-space Hermitian closure.
:obj:`WANNIER_HR_HAMILTONIAN_FIELDS`
    Number of fields in a Wannier90 HR Hamiltonian row.
:obj:`WANNIER_HR_SUFFIX`
    Required suffix for a Wannier90 HR file.
:obj:`WANNIER_INTEGER_RECOVERY_TOLERANCE`
    Fractional tolerance for recovering an exact translation.
:obj:`WANNIER_POSITION_NDIM`
    Expected dimensionality of Wannier position-operator arrays.
:obj:`WANNIER_SOURCE_FORMATS`
    Source formats accepted by Wannier operator carriers.
:obj:`WANNIER_SPIN_LAYOUTS`
    Spin layouts accepted by Wannier operator carriers.
:obj:`WANNIER_TB_HAMILTONIAN_FIELDS`
    Number of fields in a Wannier90 TB Hamiltonian row.
:obj:`WANNIER_TB_POSITION_FIELDS`
    Number of fields in a Wannier90 TB position row.
:obj:`WANNIER_TB_SUFFIX`
    Required suffix for a Wannier90 TB file.
"""

from beartype.typing import Final, Tuple

HOPPING_LIST_COMPLEX_FIELDS: Final[int] = 7
HOPPING_LIST_REAL_FIELDS: Final[int] = 6
WANNIER_CELL_FIELDS: Final[int] = 3
WANNIER_CENTRE_CONSISTENCY_TOLERANCE: Final[float] = 1e-10
WANNIER_CENTRE_NDIM: Final[int] = 2
WANNIER_DEGENERACIES_PER_LINE: Final[int] = 15
WANNIER_HERMITICITY_TOLERANCE: Final[float] = 1e-12
WANNIER_HR_HAMILTONIAN_FIELDS: Final[int] = 7
WANNIER_HR_SUFFIX: Final[str] = "_hr.dat"
WANNIER_INTEGER_RECOVERY_TOLERANCE: Final[float] = 1e-10
WANNIER_POSITION_NDIM: Final[int] = 4
WANNIER_SOURCE_FORMATS: Final[Tuple[str, ...]] = ("hr", "tb")
WANNIER_SPIN_LAYOUTS: Final[Tuple[str, ...]] = (
    "block_down_up",
    "interleaved_up_down",
)
WANNIER_TB_HAMILTONIAN_FIELDS: Final[int] = 4
WANNIER_TB_POSITION_FIELDS: Final[int] = 8
WANNIER_TB_SUFFIX: Final[str] = "_tb.dat"

__all__: list[str] = [
    "HOPPING_LIST_COMPLEX_FIELDS",
    "HOPPING_LIST_REAL_FIELDS",
    "WANNIER_CELL_FIELDS",
    "WANNIER_CENTRE_CONSISTENCY_TOLERANCE",
    "WANNIER_CENTRE_NDIM",
    "WANNIER_DEGENERACIES_PER_LINE",
    "WANNIER_HERMITICITY_TOLERANCE",
    "WANNIER_HR_HAMILTONIAN_FIELDS",
    "WANNIER_HR_SUFFIX",
    "WANNIER_INTEGER_RECOVERY_TOLERANCE",
    "WANNIER_POSITION_NDIM",
    "WANNIER_SOURCE_FORMATS",
    "WANNIER_SPIN_LAYOUTS",
    "WANNIER_TB_HAMILTONIAN_FIELDS",
    "WANNIER_TB_POSITION_FIELDS",
    "WANNIER_TB_SUFFIX",
]
