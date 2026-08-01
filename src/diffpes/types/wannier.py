"""Define operator metadata carried alongside an ingested Wannier model.

Extended Summary
----------------
The module defines :class:`WannierOperatorData`, the typed sidecar paired
with a :class:`~diffpes.types.TBModel` parsed from Wannier90 output. It keeps
Wannier centres and optional real-space position-operator matrices separate
from tight-binding Hamiltonian parameters.

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
:obj:`WANNIER_TB_HAMILTONIAN_FIELDS`
    Number of fields in a Wannier90 TB Hamiltonian row.
:obj:`WANNIER_TB_POSITION_FIELDS`
    Number of fields in a Wannier90 TB position row.
:obj:`WANNIER_TB_SUFFIX`
    Required suffix for a Wannier90 TB file.
:class:`WannierOperatorData`
    Store operator metadata for a parsed Wannier tight-binding model.
:func:`make_wannier_operator_data`
    Create validated Wannier operator metadata.

Notes
-----
Position matrices use axes ``(cell, source_orbital, target_orbital, xyz)`` in
Angstrom. Static integer cells and their Wigner--Seitz degeneracies preserve
the exact serialization context.
"""

import equinox as eqx
import jax.numpy as jnp
from beartype import beartype
from beartype.typing import Optional
from jaxtyping import Array, Complex, Complex128, Float, Float64, jaxtyped

HOPPING_LIST_COMPLEX_FIELDS: int = 7
HOPPING_LIST_REAL_FIELDS: int = 6
WANNIER_CELL_FIELDS: int = 3
WANNIER_CENTRE_CONSISTENCY_TOLERANCE: float = 1e-10
WANNIER_DEGENERACIES_PER_LINE: int = 15
WANNIER_HERMITICITY_TOLERANCE: float = 1e-12
WANNIER_HR_HAMILTONIAN_FIELDS: int = 7
WANNIER_HR_SUFFIX: str = "_hr.dat"
WANNIER_INTEGER_RECOVERY_TOLERANCE: float = 1e-10
WANNIER_TB_HAMILTONIAN_FIELDS: int = 4
WANNIER_TB_POSITION_FIELDS: int = 8
WANNIER_TB_SUFFIX: str = "_tb.dat"

_CELL_COMPONENTS: int = 3
_POSITION_COMPONENTS: int = 3
_CENTRE_NDIM: int = 2
_POSITION_NDIM: int = 4
_SOURCE_FORMATS: tuple[str, ...] = ("hr", "tb")
_SPIN_LAYOUTS: tuple[str, ...] = (
    "block_down_up",
    "interleaved_up_down",
)


def _validate_wannier_operator_structure(  # noqa: PLR0912
    position_matrices: Optional[Complex[Array, "n_R n_orb n_orb 3"]],
    centres_cart: Float[Array, "n_orb 3"],
    cells: tuple[tuple[int, int, int], ...],
    degeneracies: tuple[int, ...],
    spin_layout: str,
    source_format: str,
) -> None:
    """Validate static axes and serialization metadata."""
    if (
        centres_cart.ndim != _CENTRE_NDIM
        or centres_cart.shape[1] != _POSITION_COMPONENTS
    ):
        message: str = "centres_cart must have shape (n_orb, 3)"
        raise ValueError(message)
    if type(cells) is not tuple or type(degeneracies) is not tuple:
        message = "cells and degeneracies must be tuples"
        raise ValueError(message)
    if not cells:
        message = "cells must contain at least one translation"
        raise ValueError(message)
    if len(cells) != len(degeneracies):
        message = "cells and degeneracies must have the same length"
        raise ValueError(message)
    if any(
        type(cell) is not tuple
        or len(cell) != _CELL_COMPONENTS
        or any(type(component) is not int for component in cell)
        for cell in cells
    ):
        message = "cells must contain exact integer triples"
        raise ValueError(message)
    if len(set(cells)) != len(cells):
        message = "cells must be unique"
        raise ValueError(message)
    if any(type(weight) is not int or weight <= 0 for weight in degeneracies):
        message = "degeneracies must contain positive integers"
        raise ValueError(message)
    if spin_layout not in _SPIN_LAYOUTS:
        message = (
            "spin_layout must be 'block_down_up' or 'interleaved_up_down'"
        )
        raise ValueError(message)
    if source_format not in _SOURCE_FORMATS:
        message = "source_format must be 'hr' or 'tb'"
        raise ValueError(message)
    if source_format == "hr" and position_matrices is not None:
        message = "hr operator data must not contain position_matrices"
        raise ValueError(message)
    if source_format == "tb" and position_matrices is None:
        message = "tb operator data requires position_matrices"
        raise ValueError(message)
    if position_matrices is not None and (
        position_matrices.ndim != _POSITION_NDIM
        or position_matrices.shape[0] != len(cells)
        or position_matrices.shape[1] != centres_cart.shape[0]
        or position_matrices.shape[2] != centres_cart.shape[0]
        or position_matrices.shape[3] != _POSITION_COMPONENTS
    ):
        message = (
            "position_matrices must have shape (len(cells), n_orb, n_orb, 3)"
        )
        raise ValueError(message)


class WannierOperatorData(eqx.Module):
    """Store operator metadata for a parsed Wannier tight-binding model.

    Keep optional position matrices and explicit centres beside exact
    serialization metadata without mixing them into Hamiltonian parameters.

    :see: :class:`~.test_wannier.TestWannierOperatorData`

    Attributes
    ----------
    position_matrices : Optional[Complex128[Array, "n_R n_orb n_orb 3"]]
        Real-space position-operator matrices in Angstrom with trailing
        Cartesian axis ``(x, y, z)``. ``hr.dat`` has no such block and stores
        ``None``.
    centres_cart : Float64[Array, "n_orb 3"]
        Wannier centres in Cartesian Angstrom.
    cells : tuple[tuple[int, int, int], ...]
        Exact serialized lattice translations (**static** -- changing them
        triggers retracing).
    degeneracies : tuple[int, ...]
        Wigner--Seitz degeneracy weight for each cell (**static** -- changing
        them triggers retracing).
    spin_layout : str
        Serialized spin layout, ``"block_down_up"`` or
        ``"interleaved_up_down"`` (**static**).
    source_format : str
        Source grammar, ``"hr"`` or ``"tb"`` (**static**).

    Notes
    -----
    Numerical fields are ordinary JAX leaves and remain available to later
    differentiable operator construction. Parsers normalize matrix entries
    by their corresponding degeneracy before creating this carrier while
    retaining the original integer weights as provenance.

    See Also
    --------
    make_wannier_operator_data : Validating carrier factory.
    """

    position_matrices: Optional[Complex128[Array, "n_R n_orb n_orb 3"]]
    centres_cart: Float64[Array, "n_orb 3"]
    cells: tuple[tuple[int, int, int], ...] = eqx.field(static=True)
    degeneracies: tuple[int, ...] = eqx.field(static=True)
    spin_layout: str = eqx.field(static=True)
    source_format: str = eqx.field(static=True)

    def __check_init__(self) -> None:
        """Validate static metadata and numerical axes again."""
        _validate_wannier_operator_structure(
            self.position_matrices,
            self.centres_cart,
            self.cells,
            self.degeneracies,
            self.spin_layout,
            self.source_format,
        )


@jaxtyped(typechecker=beartype)
def make_wannier_operator_data(  # noqa: DOC502
    position_matrices: Optional[Complex[Array, "n_R n_orb n_orb 3"]],
    centres_cart: Float[Array, "n_orb 3"],
    cells: tuple[tuple[int, int, int], ...],
    degeneracies: tuple[int, ...],
    spin_layout: str,
    source_format: str,
) -> WannierOperatorData:
    """Create validated Wannier operator metadata.

    Normalize numerical precision, enforce format-specific operator presence,
    and retain exact cells and degeneracies as static metadata.

    :see: :class:`~.test_wannier.TestMakeWannierOperatorData`

    Parameters
    ----------
    position_matrices : Optional[Complex[Array, "n_R n_orb n_orb 3"]]
        Degeneracy-normalized position matrices in Angstrom, or ``None`` for
        an ``hr.dat`` source.
    centres_cart : Float[Array, "n_orb 3"]
        Explicit Cartesian Wannier centres in Angstrom.
    cells : tuple[tuple[int, int, int], ...]
        Exact integer translations in serialized order.
    degeneracies : tuple[int, ...]
        Positive Wigner--Seitz weights in the same order as ``cells``.
    spin_layout : str
        Serialized spin ordering.
    source_format : str
        ``"hr"`` or ``"tb"``.

    Returns
    -------
    data : WannierOperatorData
        Validated double-precision sidecar.

    Raises
    ------
    ValueError
        If axes, cells, weights, or static selectors are inconsistent.
    EquinoxRuntimeError
        If a numerical value is non-finite.

    Notes
    -----
    ``hr`` requires absent position matrices; ``tb`` requires them. The
    factory casts centres to float64 and position matrices to complex128.
    """
    centre_array: Float64[Array, "n_orb 3"] = jnp.asarray(
        centres_cart,
        dtype=jnp.float64,
    )
    position_array: Optional[Complex[Array, "n_R n_orb n_orb 3"]] = None
    if position_matrices is not None:
        position_array = jnp.asarray(
            position_matrices,
            dtype=jnp.complex128,
        )
    _validate_wannier_operator_structure(
        position_array,
        centre_array,
        cells,
        degeneracies,
        spin_layout,
        source_format,
    )
    centre_array = eqx.error_if(
        centre_array,
        ~jnp.all(jnp.isfinite(centre_array)),
        "make_wannier_operator_data: centres finite",
    )
    if position_array is not None:
        position_array = eqx.error_if(
            position_array,
            ~jnp.all(jnp.isfinite(position_array)),
            "make_wannier_operator_data: position matrices finite",
        )
    data: WannierOperatorData = WannierOperatorData(
        position_matrices=position_array,
        centres_cart=centre_array,
        cells=cells,
        degeneracies=degeneracies,
        spin_layout=spin_layout,
        source_format=source_format,
    )
    return data


__all__: list[str] = [
    "HOPPING_LIST_COMPLEX_FIELDS",
    "HOPPING_LIST_REAL_FIELDS",
    "WANNIER_CELL_FIELDS",
    "WANNIER_CENTRE_CONSISTENCY_TOLERANCE",
    "WANNIER_DEGENERACIES_PER_LINE",
    "WANNIER_HERMITICITY_TOLERANCE",
    "WANNIER_HR_HAMILTONIAN_FIELDS",
    "WANNIER_HR_SUFFIX",
    "WANNIER_INTEGER_RECOVERY_TOLERANCE",
    "WANNIER_TB_HAMILTONIAN_FIELDS",
    "WANNIER_TB_POSITION_FIELDS",
    "WANNIER_TB_SUFFIX",
    "WannierOperatorData",
    "make_wannier_operator_data",
]
