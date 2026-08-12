"""Define orbital-basis metadata for radial models.

Extended Summary
----------------
This module defines the static orbital quantum numbers and labels
that connect electronic states to radial shells.

Routine Listings
----------------
:class:`OrbitalBasis`
    Store orbital quantum-number metadata in a JAX PyTree.
:func:`make_orbital_basis`
    Create a validated ``OrbitalBasis`` instance.
"""

import equinox as eqx
from beartype import beartype
from beartype.typing import Optional, Tuple
from jaxtyping import jaxtyped


def _validate_orbital_basis_structure(
    atom_indices: Tuple[int, ...],
    n: Tuple[int, ...],
    l: Tuple[int, ...],  # noqa: E741
    m: Tuple[int, ...],
    spin: Tuple[int, ...],
    labels: Tuple[str, ...],
) -> None:
    """PRIVATE: Validate static orbital-basis metadata.

    Implementation Logic
    --------------------
    Use exact ``type`` comparisons to reject bools and NumPy integers.
    Check quantum-number consistency pairwise with
    ``zip(..., strict=True)`` over ``(n, l)`` and ``(l, m)``.

    Parameters
    ----------
    atom_indices : Tuple[int, ...]
        Atom-row index for each orbital.
    n : Tuple[int, ...]
        Principal quantum numbers, one per orbital.
    l : Tuple[int, ...]
        Angular momentum quantum numbers, one per orbital.
    m : Tuple[int, ...]
        Magnetic quantum numbers, one per orbital.
    spin : Tuple[int, ...]
        Spin channels: empty for a spinless basis, otherwise one ``+1``
        or ``-1`` entry per orbital.
    labels : Tuple[str, ...]
        Human-readable orbital labels.

    Raises
    ------
    ValueError
        If a field has the wrong container type or length. If an atom,
        principal, or angular quantum number is invalid. If ``spin``
        has the wrong length or channel values. If a label is not a
        string. This is the static construction-time contract.
    """
    if any(
        type(values) is not tuple
        for values in (atom_indices, n, l, m, spin, labels)
    ):
        message: str = "all OrbitalBasis fields must be tuples"
        raise ValueError(message)
    n_orbitals: int = len(n)
    if not (
        len(atom_indices) == len(l) == len(m) == len(labels) == n_orbitals
    ):
        message: str = (
            "atom_indices, n, l, m, and labels must have the same length"
        )
        raise ValueError(message)
    if any(type(index) is not int or index < 0 for index in atom_indices):
        message = "atom_indices must contain non-negative integers"
        raise ValueError(message)
    if any(type(value) is not int or value < 1 for value in n):
        message = "n must contain integers of at least 1"
        raise ValueError(message)
    if any(
        type(angular) is not int or angular < 0 or angular >= principal
        for principal, angular in zip(n, l, strict=True)
    ):
        message = "l must contain integers satisfying 0 <= l < n"
        raise ValueError(message)
    if any(
        type(magnetic) is not int or abs(magnetic) > angular
        for angular, magnetic in zip(l, m, strict=True)
    ):
        message = "m must contain integers satisfying abs(m) <= l"
        raise ValueError(message)
    if spin and len(spin) != n_orbitals:
        message = "spin must be empty or have one entry per orbital"
        raise ValueError(message)
    if any(
        type(channel) is not int or channel not in (-1, 1) for channel in spin
    ):
        message = "spin entries must be +1 or -1"
        raise ValueError(message)
    if any(type(label) is not str for label in labels):
        message = "labels must contain strings"
        raise ValueError(message)


class OrbitalBasis(eqx.Module):
    """Store orbital quantum-number metadata in a JAX PyTree.

    This type describes the orbital basis for dipole matrix-element
    calculations for the differentiable Chinook pipeline. The quantum
    numbers (n, l, m) parameterize the radial wavefunctions (via
    Slater-type orbitals) and angular parts (spherical harmonics) that
    enter the photoemission matrix element.

    All fields contain static auxiliary data because quantum numbers control
    code paths. They determine recurrence depths in spherical Bessel functions
    and associated Legendre polynomials. They also index the Gaunt coefficient
    table. A quantum-number change alters the computational graph. JAX must
    therefore recompile after this change.


    :see: :class:`~.test_orbital_basis.TestOrbitalBasis`

    Attributes
    ----------
    atom_indices : Tuple[int, ...]
        Atom-row index for each orbital. Each entry refers to a row of
        :attr:`~diffpes.types.CrystalGeometry.positions` (**static** -- a
        compile-time constant; changing it triggers retracing).
    n : Tuple[int, ...]
        Principal quantum numbers, one per orbital. Each value controls the
        radial node count and the power of *r*. The Slater form
        R_nl(r) ~ r^{n-1} exp(-zeta*r) uses static compile-time values;
        changing them triggers retracing.
    l : Tuple[int, ...]
        Angular momentum quantum numbers, one per orbital (0=s, 1=p,
        2=d, 3=f). Determines the spherical harmonic Y_l^m used in
        the matrix element integral (**static** -- compile-time constants;
        changing them triggers retracing).
    m : Tuple[int, ...]
        Magnetic quantum numbers, one per orbital. Ranges from -l to
        +l for each orbital. Selects the specific spherical harmonic
        component (**static** -- compile-time constants; changing them
        triggers retracing).
    spin : Tuple[int, ...]
        Spin channel for each orbital. The empty tuple denotes a spinless
        basis; a spinor basis stores ``+1`` or ``-1`` for every orbital
        (**static** -- a compile-time constant; changing it triggers
        retracing).
    labels : Tuple[str, ...]
        Human-readable orbital labels (e.g. ``("2s", "2px", ...)``).
        Used for plotting and debugging (**static** -- compile-time constants;
        changing them triggers retracing).

    Notes
    -----
    Implemented as an immutable :class:`equinox.Module` PyTree.
    All fields are auxiliary data (no JAX array children) because
    changing any quantum number changes the computational graph and
    requires JIT recompilation. The children tuple is always empty.

    See Also
    --------
    RadialSpec : Wraps shell-shared radial parameters alongside this basis.
    make_orbital_basis : Factory function with length validation and
        default label generation.
    """

    atom_indices: Tuple[int, ...] = eqx.field(static=True)
    n: Tuple[int, ...] = eqx.field(static=True)
    l: Tuple[int, ...] = eqx.field(static=True)  # noqa: E741
    m: Tuple[int, ...] = eqx.field(static=True)
    spin: Tuple[int, ...] = eqx.field(static=True)
    labels: Tuple[str, ...] = eqx.field(static=True)

    def __check_init__(self) -> None:
        """Validate the static orbital-basis invariants again."""
        _validate_orbital_basis_structure(
            self.atom_indices,
            self.n,
            self.l,
            self.m,
            self.spin,
            self.labels,
        )


@jaxtyped(typechecker=beartype)
def make_orbital_basis(  # noqa: DOC502
    atom_indices: Tuple[int, ...],
    n: Tuple[int, ...],
    l: Tuple[int, ...],  # noqa: E741
    m: Tuple[int, ...],
    spin: Tuple[int, ...] = (),
    labels: Optional[Tuple[str, ...]] = None,
) -> OrbitalBasis:
    """Create a validated ``OrbitalBasis`` instance.

    The factory validates quantum number tuples and
    constructs an ``OrbitalBasis`` PyTree. The three quantum number
    tuples must all have the same length (one entry per orbital).
    If ``labels`` is absent, the factory generates generic labels such as
    ``"orb_0"`` and ``"orb_1"``.

    Use this factory instead of the raw ``OrbitalBasis`` constructor
    to get automatic length validation and default label generation.

    :see: :class:`~.test_orbital_basis.TestMakeOrbitalBasis`

    Implementation Logic
    --------------------
    1. **Prepare the normalized values**::

           n_orbitals = len(n)

       This expression gives the later validation steps a stable shape and
       dtype.

    2. **Apply static validation**::

           _validate_orbital_basis_structure(...)

       This predicate rejects invalid structure before JAX traces the
       numerical checks.

    3. **Return the named instance**::

           return basis

       The explicit name keeps the implementation and the Returns section
       synchronized.

    Parameters
    ----------
    atom_indices : Tuple[int, ...]
        Atom-row indices (**static** -- compile-time constants; changing them
        triggers retracing), one per orbital.
    n : Tuple[int, ...]
        Principal quantum numbers (**static** -- compile-time constants;
        changing them triggers retracing), one per orbital.
    l : Tuple[int, ...]
        Angular momentum quantum numbers (**static** -- compile-time
        constants; changing them triggers retracing), one per orbital.
    m : Tuple[int, ...]
        Magnetic quantum numbers (**static** -- compile-time constants;
        changing them triggers retracing), one per orbital.
    spin : Tuple[int, ...], optional
        Spin channels (**static** -- compile-time constants; changing them
        triggers retracing). The empty tuple denotes a spinless basis;
        otherwise every entry must be ``+1`` or ``-1``. Default is empty.
    labels : Optional[Tuple[str, ...]], optional
        Human-readable orbital labels (**static** -- compile-time constants;
        changing them triggers retracing). Defaults to
        ``("orb_0", "orb_1", ...)``.

    Returns
    -------
    basis : OrbitalBasis
        Validated orbital basis with consistent lengths.

    Raises
    ------
    ValueError
        If any per-orbital tuple has a different length. The function also
        rejects invalid atom indices, quantum numbers, or spin channels.

    Notes
    -----
    Every ``OrbitalBasis`` field uses ``eqx.field(static=True)``, so the
    factory performs static validation. Invalid tuple lengths or quantum
    numbers raise ``ValueError`` before tracing. No ``eqx.error_if`` checks
    apply.

    See Also
    --------
    OrbitalBasis : The PyTree class constructed by this factory.
    """
    n_orbitals: int = len(n)
    resolved_labels: Tuple[str, ...] = (
        tuple(f"orb_{i}" for i in range(n_orbitals))
        if labels is None
        else labels
    )
    _validate_orbital_basis_structure(
        atom_indices,
        n,
        l,
        m,
        spin,
        resolved_labels,
    )
    basis: OrbitalBasis = OrbitalBasis(
        atom_indices=atom_indices,
        n=n,
        l=l,
        m=m,
        spin=spin,
        labels=resolved_labels,
    )
    return basis


__all__: list[str] = [
    "OrbitalBasis",
    "make_orbital_basis",
]
