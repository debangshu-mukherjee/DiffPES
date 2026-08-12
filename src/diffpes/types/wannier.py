"""Define operator metadata carried alongside an ingested Wannier model.

Extended Summary
----------------
The module defines :class:`WannierOperatorData`, the typed sidecar paired
with a :class:`~diffpes.types.TBModel` parsed from Wannier90 output. It keeps
Wannier centres and optional real-space position-operator matrices separate
from tight-binding Hamiltonian parameters.

Routine Listings
----------------
:class:`HamiltonianBlocks`
    Store normalized Hamiltonian matrices with exact block metadata.
:class:`HoppingRecord`
    Store one parsed non-onsite hopping and its source line.
:class:`TextLineCursor`
    Record strict line-numbered parsing for one text file.
:class:`WannierOperatorData`
    Store operator metadata for a parsed Wannier tight-binding model.
:func:`make_hamiltonian_blocks`
    Create normalized Hamiltonian blocks without changing parsed values.
:func:`make_hopping_record`
    Create one parsed hopping record without changing its values.
:func:`make_text_line_cursor`
    Create a line cursor from one UTF-8 text file.
:func:`make_wannier_operator_data`
    Create validated Wannier operator metadata.

Notes
-----
Position matrices use axes ``(cell, source_orbital, target_orbital, xyz)`` in
Angstrom. Static integer cells and their Wigner--Seitz degeneracies preserve
the exact serialization context.
"""

from pathlib import Path

import equinox as eqx
import jax.numpy as jnp
from beartype import beartype
from beartype.typing import Optional, Tuple
from jaxtyping import Array, Complex128, Float64, Int64, jaxtyped
from numpy.typing import NDArray

from diffpes.constants import (
    CARTESIAN_COMPONENTS,
    WANNIER_CENTRE_NDIM,
    WANNIER_POSITION_NDIM,
    WANNIER_SOURCE_FORMATS,
    WANNIER_SPIN_LAYOUTS,
)


class HamiltonianBlocks(eqx.Module):
    """Store normalized Hamiltonian matrices with exact block metadata.

    This carrier keeps parsed complex Hamiltonian blocks together with their
    physical source lines, exact cells, and Wigner--Seitz degeneracies.

    :see: :class:`~.test_wannier.TestHamiltonianBlocks`

    Attributes
    ----------
    matrices : Complex128[NDArray, "n_cell n_orb n_orb"]
        Degeneracy-normalized Hamiltonian matrices in eV.
    source_lines : Int64[NDArray, "n_cell n_orb n_orb"]
        One-based physical source line for each matrix element.
    cells : Tuple[Tuple[int, int, int], ...]
        Exact integer lattice translations (**static** -- parser metadata
        that does not enter compiled kernels).
    degeneracies : Tuple[int, ...]
        Wigner--Seitz degeneracy for each cell (**static** -- parser metadata
        that does not enter compiled kernels).

    Notes
    -----
    The parser owns all format validation and normalization. This carrier
    retains the arrays and exact metadata without an additional reduction.

    See Also
    --------
    make_hamiltonian_blocks : Create normalized Hamiltonian blocks without
        changing parsed values.
    """

    matrices: Complex128[NDArray, "n_cell n_orb n_orb"]
    source_lines: Int64[NDArray, "n_cell n_orb n_orb"]
    cells: Tuple[Tuple[int, int, int], ...] = eqx.field(static=True)
    degeneracies: Tuple[int, ...] = eqx.field(static=True)


class HoppingRecord(eqx.Module):
    """Store one parsed non-onsite hopping and its source line.

    This carrier retains the orbital pair, lattice translation, complex
    amplitude, and physical source line used by Hermitian-closure checks.

    :see: :class:`~.test_wannier.TestHoppingRecord`

    Attributes
    ----------
    pair : Tuple[int, int]
        Zero-based source and target orbital indices (**static** -- parser
        metadata that does not enter compiled kernels).
    cell : Tuple[int, int, int]
        Exact integer lattice translation (**static** -- parser metadata that
        does not enter compiled kernels).
    amplitude : complex
        Complex hopping amplitude in eV.
    line_number : int
        One-based physical source line (**static** -- parser metadata that
        does not enter compiled kernels).

    Notes
    -----
    The parser excludes onsite diagonal terms before it builds this carrier.
    The carrier does not alter the complex amplitude.

    See Also
    --------
    make_hopping_record : Create one parsed hopping record without changing
        its values.
    """

    pair: Tuple[int, int] = eqx.field(static=True)
    cell: Tuple[int, int, int] = eqx.field(static=True)
    amplitude: complex
    line_number: int = eqx.field(static=True)


class TextLineCursor(eqx.Module):
    """Record strict line-numbered parsing for one text file.

    This host-side cursor retains every physical line and advances one exact
    index as a parser consumes records. It can skip blank lines only when a
    caller requests a nonempty record.

    :see: :class:`~.test_wannier.TestTextLineCursor`

    Attributes
    ----------
    path : Path
        Source text file (**static** -- host parser metadata that does not
        enter compiled kernels).
    lines : Tuple[str, ...]
        Physical text lines without newline terminators (**static** -- host
        parser data that does not enter compiled kernels).
    index : int
        Zero-based index of the next unread line (**static** -- mutable host
        parser state that does not enter compiled kernels).

    Notes
    -----
    Equinox makes module attributes immutable through normal assignment. The
    cursor uses ``object.__setattr__`` to retain its original host-side index
    mutation behavior.

    See Also
    --------
    make_text_line_cursor : Create a line cursor from one UTF-8 text file.
    """

    path: Path = eqx.field(static=True)
    lines: Tuple[str, ...] = eqx.field(static=True)
    index: int = eqx.field(static=True)

    def next_line(self, context: str) -> Tuple[int, str]:
        """Return the next physical line without skipping blanks.

        Advance the cursor by exactly one physical line and retain blank text
        unchanged for strict fixed-record parsing.

        Parameters
        ----------
        context : str
            Parser context named in an unexpected-end diagnostic.

        Returns
        -------
        result : Tuple[int, str]
            One-based physical line number and its text without a newline.

        Raises
        ------
        ValueError
            If the cursor has no unread physical line.

        Notes
        -----
        The method reads ``lines[index]``, advances ``index`` through
        ``object.__setattr__``, and returns the original text.
        """
        if self.index >= len(self.lines):
            message: str = (
                f"{self.path}: unexpected end of file while reading {context}"
            )
            raise ValueError(message)
        line_number: int = self.index + 1
        text: str = self.lines[self.index]
        object.__setattr__(self, "index", self.index + 1)
        result: Tuple[int, str] = (line_number, text)
        return result

    def next_nonempty(self, context: str) -> Tuple[int, str]:
        """Return the next nonblank physical line.

        Advance through blank physical lines until the cursor reaches one
        record with non-whitespace text.

        Parameters
        ----------
        context : str
            Parser context named in an unexpected-end diagnostic.

        Returns
        -------
        result : Tuple[int, str]
            One-based physical line number and the next nonblank text.

        Raises
        ------
        ValueError
            If no unread nonblank line remains.

        Notes
        -----
        The method calls :meth:`next_line` for every candidate and accepts the
        first text whose stripped value is nonempty.
        """
        while self.index < len(self.lines):
            line_number: int
            text: str
            line_number, text = self.next_line(context)
            if text.strip():
                result: Tuple[int, str] = (line_number, text)
                return result
        message: str = (
            f"{self.path}: unexpected end of file while reading {context}"
        )
        raise ValueError(message)

    def ensure_exhausted(self) -> None:
        """Reject a trailing nonblank record.

        Consume remaining blank lines and fail on the first unexpected record
        so a strict grammar cannot silently ignore extra content.

        Raises
        ------
        ValueError
            If an unread line contains non-whitespace text.

        Notes
        -----
        The method calls :meth:`next_line` until the cursor reaches the end.
        It reports the exact physical line of the first nonblank record.
        """
        while self.index < len(self.lines):
            line_number: int
            text: str
            line_number, text = self.next_line("trailing records")
            if text.strip():
                message: str = (
                    f"{self.path}: line {line_number}: unexpected trailing "
                    "record"
                )
                raise ValueError(message)


def _validate_wannier_operator_structure(  # noqa: PLR0912
    position_matrices: Optional[Complex128[Array, "n_R n_orb n_orb 3"]],
    centres_cart: Float64[Array, "n_orb 3"],
    cells: Tuple[Tuple[int, int, int], ...],
    degeneracies: Tuple[int, ...],
    spin_layout: str,
    source_format: str,
) -> None:
    """PRIVATE: Validate static axes and serialization metadata.

    Implementation Logic
    --------------------
    Check plain Python metadata with exact ``type`` comparisons and the
    array axes through ``ndim`` and ``shape``. ``hr`` data must store
    ``None`` position matrices, and ``tb`` data must store one
    ``(n_orb, n_orb, 3)`` block per cell.

    Parameters
    ----------
    position_matrices : Optional[Complex128[Array, "n_R n_orb n_orb 3"]]
        Real-space position-operator matrices in Angstrom, or ``None``
        for ``hr`` operator data.
    centres_cart : Float64[Array, "n_orb 3"]
        Wannier centres in Cartesian Angstrom.
    cells : Tuple[Tuple[int, int, int], ...]
        Exact integer lattice translations.
    degeneracies : Tuple[int, ...]
        Wigner--Seitz degeneracy weights, one per cell.
    spin_layout : str
        Spinor orbital-ordering label.
    source_format : str
        Source serialization format, ``"hr"`` or ``"tb"``.

    Raises
    ------
    ValueError
        If ``centres_cart`` has the wrong shape. If cell or degeneracy
        metadata has invalid types, lengths, values, or duplicates. If
        a selector has an unknown label. If ``position_matrices``
        contradicts ``source_format`` or has the wrong shape. This is
        the static construction-time contract.
    """
    if (
        centres_cart.ndim != WANNIER_CENTRE_NDIM
        or centres_cart.shape[1] != CARTESIAN_COMPONENTS
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
        or len(cell) != CARTESIAN_COMPONENTS
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
    if spin_layout not in WANNIER_SPIN_LAYOUTS:
        message = (
            "spin_layout must be 'block_down_up' or 'interleaved_up_down'"
        )
        raise ValueError(message)
    if source_format not in WANNIER_SOURCE_FORMATS:
        message = "source_format must be 'hr' or 'tb'"
        raise ValueError(message)
    if source_format == "hr" and position_matrices is not None:
        message = "hr operator data must not contain position_matrices"
        raise ValueError(message)
    if source_format == "tb" and position_matrices is None:
        message = "tb operator data requires position_matrices"
        raise ValueError(message)
    if position_matrices is not None and (
        position_matrices.ndim != WANNIER_POSITION_NDIM
        or position_matrices.shape[0] != len(cells)
        or position_matrices.shape[1] != centres_cart.shape[0]
        or position_matrices.shape[2] != centres_cart.shape[0]
        or position_matrices.shape[3] != CARTESIAN_COMPONENTS
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
    cells : Tuple[Tuple[int, int, int], ...]
        Exact serialized lattice translations (**static** -- changing them
        triggers retracing).
    degeneracies : Tuple[int, ...]
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
    cells: Tuple[Tuple[int, int, int], ...] = eqx.field(static=True)
    degeneracies: Tuple[int, ...] = eqx.field(static=True)
    spin_layout: str = eqx.field(static=True)
    source_format: str = eqx.field(static=True)

    def __check_init__(self) -> None:  # noqa: DOC502
        """PRIVATE: Validate static metadata and numerical axes again.

        Raises
        ------
        ValueError
            If the static metadata or numerical axes violate the carrier
            structure.

        Notes
        -----
        Equinox calls this hook after direct carrier construction. The method
        delegates every structural check to
        :func:`_validate_wannier_operator_structure`.
        """
        _validate_wannier_operator_structure(
            self.position_matrices,
            self.centres_cart,
            self.cells,
            self.degeneracies,
            self.spin_layout,
            self.source_format,
        )


@jaxtyped(typechecker=beartype)
def make_hamiltonian_blocks(
    matrices: Complex128[NDArray, "n_cell n_orb n_orb"],
    source_lines: Int64[NDArray, "n_cell n_orb n_orb"],
    cells: Tuple[Tuple[int, int, int], ...],
    degeneracies: Tuple[int, ...],
) -> HamiltonianBlocks:
    """Create normalized Hamiltonian blocks without changing parsed values.

    Bind already validated NumPy matrices to their physical source lines,
    exact cells, and degeneracy metadata for later parser reductions.

    :see: :class:`~.test_wannier.TestMakeHamiltonianBlocks`

    Parameters
    ----------
    matrices : Complex128[NDArray, "n_cell n_orb n_orb"]
        Degeneracy-normalized Hamiltonian matrices in eV.
    source_lines : Int64[NDArray, "n_cell n_orb n_orb"]
        One-based physical source line for each matrix element.
    cells : Tuple[Tuple[int, int, int], ...]
        Exact integer lattice translations.
    degeneracies : Tuple[int, ...]
        Wigner--Seitz degeneracy for each cell.

    Returns
    -------
    blocks : HamiltonianBlocks
        Parser carrier containing the supplied arrays and exact metadata.

    Notes
    -----
    Parsing functions validate shapes, ordering, and values before this
    factory call. The factory retains the supplied objects without casting or
    copying them.
    """
    blocks: HamiltonianBlocks = HamiltonianBlocks(
        matrices=matrices,
        source_lines=source_lines,
        cells=cells,
        degeneracies=degeneracies,
    )
    return blocks


@jaxtyped(typechecker=beartype)
def make_hopping_record(
    pair: Tuple[int, int],
    cell: Tuple[int, int, int],
    amplitude: complex,
    line_number: int,
) -> HoppingRecord:
    """Create one parsed hopping record without changing its values.

    Bind one validated orbital pair, cell, complex amplitude, and physical
    source line for later Hermitian-closure checks.

    :see: :class:`~.test_wannier.TestMakeHoppingRecord`

    Parameters
    ----------
    pair : Tuple[int, int]
        Zero-based source and target orbital indices.
    cell : Tuple[int, int, int]
        Exact integer lattice translation.
    amplitude : complex
        Complex hopping amplitude in eV.
    line_number : int
        One-based physical source line.

    Returns
    -------
    record : HoppingRecord
        Parser carrier containing the supplied values.

    Notes
    -----
    Parsing functions validate indices and finite amplitudes before this
    factory call. The factory retains every supplied value unchanged.
    """
    record: HoppingRecord = HoppingRecord(
        pair=pair,
        cell=cell,
        amplitude=amplitude,
        line_number=line_number,
    )
    return record


@jaxtyped(typechecker=beartype)
def make_text_line_cursor(path: Path) -> TextLineCursor:  # noqa: DOC502
    """Create a line cursor from one UTF-8 text file.

    Read every physical line once and initialize the next unread index at the
    start of the file.

    :see: :class:`~.test_wannier.TestMakeTextLineCursor`

    Parameters
    ----------
    path : Path
        UTF-8 text file to parse.

    Returns
    -------
    cursor : TextLineCursor
        Line-numbered cursor positioned before the first physical line.

    Raises
    ------
    OSError
        If the operating system cannot read ``path``.
    UnicodeError
        If the file is not valid UTF-8 text.

    Notes
    -----
    The factory applies :meth:`Path.read_text`, splits the text with
    :meth:`str.splitlines`, and stores the resulting immutable tuple with
    ``index=0``.
    """
    text: str = path.read_text(encoding="utf-8")
    lines: Tuple[str, ...] = tuple(text.splitlines())
    cursor: TextLineCursor = TextLineCursor(
        path=path,
        lines=lines,
        index=0,
    )
    return cursor


@jaxtyped(typechecker=beartype)
def make_wannier_operator_data(  # noqa: DOC502
    position_matrices: Optional[Complex128[Array, "n_R n_orb n_orb 3"]],
    centres_cart: Float64[Array, "n_orb 3"],
    cells: Tuple[Tuple[int, int, int], ...],
    degeneracies: Tuple[int, ...],
    spin_layout: str,
    source_format: str,
) -> WannierOperatorData:
    """Create validated Wannier operator metadata.

    Normalize numerical precision, enforce format-specific operator presence,
    and retain exact cells and degeneracies as static metadata.

    :see: :class:`~.test_wannier.TestMakeWannierOperatorData`

    Parameters
    ----------
    position_matrices : Optional[Complex128[Array, "n_R n_orb n_orb 3"]]
        Degeneracy-normalized position matrices in Angstrom, or ``None`` for
        an ``hr.dat`` source.
    centres_cart : Float64[Array, "n_orb 3"]
        Explicit Cartesian Wannier centres in Angstrom.
    cells : Tuple[Tuple[int, int, int], ...]
        Exact integer translations in serialized order.
    degeneracies : Tuple[int, ...]
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
    position_array: Optional[Complex128[Array, "n_R n_orb n_orb 3"]] = None
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
    "HamiltonianBlocks",
    "HoppingRecord",
    "TextLineCursor",
    "WannierOperatorData",
    "make_hamiltonian_blocks",
    "make_hopping_record",
    "make_text_line_cursor",
    "make_wannier_operator_data",
]
