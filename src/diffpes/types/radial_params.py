"""Define radial-wavefunction parameter structures.

Extended Summary
----------------
This module defines PyTree types for orbital basis metadata and Slater-type
radial wavefunction parameters used by the differentiable dipole
matrix element pipeline.

Routine Listings
----------------
:class:`OrbitalBasis`
    Store orbital quantum-number metadata in a JAX PyTree.
:class:`SlaterParams`
    Store Slater radial-wavefunction parameters in a JAX PyTree.
:class:`SlaterKosterParams`
    Store differentiable Slater--Koster two-center integrals.
:func:`make_orbital_basis`
    Create a validated ``OrbitalBasis`` instance.
:func:`make_slater_params`
    Create a validated ``SlaterParams`` instance.
:func:`make_slater_koster_params`
    Create validated Slater--Koster two-center parameters.

Notes
-----
``OrbitalBasis`` contains only static auxiliary data. Atom assignments,
quantum numbers, spin channels, and labels define the traced program shape.
``SlaterParams`` wraps differentiable Slater exponents alongside
the static orbital basis. ``SlaterKosterParams`` separates differentiable
two-center values from their static material/channel identifiers.
"""

import equinox as eqx
import jax.numpy as jnp
from beartype import beartype
from beartype.typing import Optional
from jaxtyping import Array, Float, jaxtyped


def _validate_orbital_basis_structure(
    atom_indices: tuple[int, ...],
    n: tuple[int, ...],
    l: tuple[int, ...],  # noqa: E741
    m: tuple[int, ...],
    spin: tuple[int, ...],
    labels: tuple[str, ...],
) -> None:
    """Validate static orbital-basis metadata."""
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


    :see: :class:`~.test_radial_params.TestOrbitalBasis`

    Attributes
    ----------
    atom_indices : tuple[int, ...]
        Atom-row index for each orbital. Each entry refers to a row of
        :attr:`~diffpes.types.CrystalGeometry.positions` (**static** -- a
        compile-time constant; changing it triggers retracing).
    n : tuple[int, ...]
        Principal quantum numbers, one per orbital. Each value controls the
        radial node count and the power of *r*. The Slater form
        R_nl(r) ~ r^{n-1} exp(-zeta*r) uses static compile-time values;
        changing them triggers retracing.
    l : tuple[int, ...]
        Angular momentum quantum numbers, one per orbital (0=s, 1=p,
        2=d, 3=f). Determines the spherical harmonic Y_l^m used in
        the matrix element integral (**static** -- compile-time constants;
        changing them triggers retracing).
    m : tuple[int, ...]
        Magnetic quantum numbers, one per orbital. Ranges from -l to
        +l for each orbital. Selects the specific spherical harmonic
        component (**static** -- compile-time constants; changing them
        triggers retracing).
    spin : tuple[int, ...]
        Spin channel for each orbital. The empty tuple denotes a spinless
        basis; a spinor basis stores ``+1`` or ``-1`` for every orbital
        (**static** -- a compile-time constant; changing it triggers
        retracing).
    labels : tuple[str, ...]
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
    SlaterParams : Wraps differentiable Slater exponents alongside
        this static basis metadata.
    make_orbital_basis : Factory function with length validation and
        default label generation.
    """

    atom_indices: tuple[int, ...] = eqx.field(static=True)
    n: tuple[int, ...] = eqx.field(static=True)
    l: tuple[int, ...] = eqx.field(static=True)  # noqa: E741
    m: tuple[int, ...] = eqx.field(static=True)
    spin: tuple[int, ...] = eqx.field(static=True)
    labels: tuple[str, ...] = eqx.field(static=True)

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


class SlaterParams(eqx.Module):
    """Store Slater radial-wavefunction parameters in a JAX PyTree.

    This type contains differentiable Slater exponents and linear-combination
    coefficients alongside the static orbital basis metadata.
    Inverse fitting workflows optimize the Slater exponents (zeta).
    ``jax.grad`` gives the gradient of the simulated ARPES intensity with
    respect to the radial wavefunction shape.

    In a multi-zeta basis (C > 1), the ``coefficients`` matrix combines C
    Slater-type functions for each orbital. These functions have different
    exponents. In a single-zeta basis (C=1), each orbital has one exponent and
    a coefficient of 1.0.


    :see: :class:`~.test_radial_params.TestSlaterParams`

    Attributes
    ----------
    zeta : Float[Array, " O"]
        Slater exponents (inverse Bohr), one per orbital. Controls
        the radial decay rate of each Slater-type orbital
        R(r) ~ r^{n-1} exp(-zeta * r). JAX-traced (differentiable)
        -- this is the primary optimization target for inverse
        fitting.
    coefficients : Float[Array, "O C"]
        Linear combination coefficients for multi-zeta expansion.
        Shape is (O, C) where O is the number of orbitals and C is
        the contraction length. For single-zeta basis sets, C=1 and
        all coefficients are 1.0. JAX-traced (differentiable).
    orbital_basis : OrbitalBasis
        Quantum numbers (n, l, m) and labels for each orbital.
        **Static** -- a compile-time constant; changing it triggers retracing.

    Notes
    -----
    Implemented as an immutable :class:`equinox.Module` PyTree.
    ``zeta`` and ``coefficients`` are JAX array children (on the
    gradient tape), while ``orbital_basis`` is auxiliary data
    (static at trace time). This separation means that
    ``jax.grad(loss)(slater_params)`` differentiates through the
    exponents and coefficients while keeping the quantum number
    structure fixed.

    See Also
    --------
    OrbitalBasis : The static quantum number metadata nested inside.
    make_slater_params : Factory function with validation and default
        coefficient generation.
    """

    zeta: Float[Array, " O"]
    coefficients: Float[Array, "O C"]
    orbital_basis: OrbitalBasis = eqx.field(static=True)


def _validate_slater_koster_structure(
    values: Float[Array, " n_sk"],
    keys: tuple[str, ...],
) -> None:
    """Validate Slater--Koster parameter axes and static identifiers."""
    if values.ndim != 1:
        message: str = "SlaterKosterParams values must be one-dimensional"
        raise ValueError(message)
    if type(keys) is not tuple:
        message = "SlaterKosterParams keys must be a tuple"
        raise ValueError(message)
    if len(keys) != values.shape[0]:
        message = (
            "SlaterKosterParams values and keys must have the same length"
        )
        raise ValueError(message)
    if any(type(key) is not str or not key for key in keys):
        message = "SlaterKosterParams keys must contain non-empty strings"
        raise ValueError(message)
    if len(set(keys)) != len(keys):
        message = "SlaterKosterParams keys must be unique"
        raise ValueError(message)


class SlaterKosterParams(eqx.Module):
    """Store differentiable Slater--Koster two-center integrals.

    The numerical values are the flat-real optimization coordinates for a
    Slater--Koster material model. Their keys are static identifiers such as
    ``"C-C:pp_sigma"`` or ``"Ru-O:pd_pi"``. A key change alters the material
    topology and therefore triggers JAX retracing.

    :see: :class:`~.test_radial_params.TestSlaterKosterParams`

    Attributes
    ----------
    values : Float[Array, " n_sk"]
        Fundamental two-center hopping integrals in eV. These values remain
        differentiable JAX leaves.
    keys : tuple[str, ...]
        Unique material/channel identifiers (**static** -- changing them
        triggers retracing).

    Notes
    -----
    The carrier deliberately does not prescribe distance scaling. The
    Slater--Koster builder interprets the identifiers and assigns them to
    frozen neighbor shells.

    See Also
    --------
    make_slater_koster_params : Validating factory for this carrier.
    """

    values: Float[Array, " n_sk"]
    keys: tuple[str, ...] = eqx.field(static=True)

    def __check_init__(self) -> None:
        """Validate the traced axis against the static key tuple."""
        _validate_slater_koster_structure(self.values, self.keys)


@jaxtyped(typechecker=beartype)
def make_orbital_basis(  # noqa: DOC502
    atom_indices: tuple[int, ...],
    n: tuple[int, ...],
    l: tuple[int, ...],  # noqa: E741
    m: tuple[int, ...],
    spin: tuple[int, ...] = (),
    labels: Optional[tuple[str, ...]] = None,
) -> OrbitalBasis:
    """Create a validated ``OrbitalBasis`` instance.

    The factory validates quantum number tuples and
    constructs an ``OrbitalBasis`` PyTree. The three quantum number
    tuples must all have the same length (one entry per orbital).
    If ``labels`` is absent, the factory generates generic labels such as
    ``"orb_0"`` and ``"orb_1"``.

    Use this factory instead of the raw ``OrbitalBasis`` constructor
    to get automatic length validation and default label generation.

    :see: :class:`~.test_radial_params.TestMakeOrbitalBasis`

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
    atom_indices : tuple[int, ...]
        Atom-row indices (**static** -- compile-time constants; changing them
        triggers retracing), one per orbital.
    n : tuple[int, ...]
        Principal quantum numbers (**static** -- compile-time constants;
        changing them triggers retracing), one per orbital.
    l : tuple[int, ...]
        Angular momentum quantum numbers (**static** -- compile-time
        constants; changing them triggers retracing), one per orbital.
    m : tuple[int, ...]
        Magnetic quantum numbers (**static** -- compile-time constants;
        changing them triggers retracing), one per orbital.
    spin : tuple[int, ...], optional
        Spin channels (**static** -- compile-time constants; changing them
        triggers retracing). The empty tuple denotes a spinless basis;
        otherwise every entry must be ``+1`` or ``-1``. Default is empty.
    labels : Optional[tuple[str, ...]], optional
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
    resolved_labels: tuple[str, ...] = (
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


@jaxtyped(typechecker=beartype)
def make_slater_params(  # noqa: DOC503
    zeta: Float[Array, " Oz"],
    orbital_basis: OrbitalBasis,
    coefficients: Optional[Float[Array, "Oc C"]] = None,
) -> SlaterParams:
    """Create a validated ``SlaterParams`` instance.

    The factory validates Slater radial wavefunction
    parameters and constructs a ``SlaterParams`` PyTree. The ``zeta``
    array length must match the orbital count in ``orbital_basis``. If
    ``coefficients`` is absent, the factory uses a single-zeta basis. It
    creates a column of ones with shape ``(O, 1)``.

    Use this factory for automatic size validation and default coefficient
    generation. The factory also guarantees ``float64`` dtypes.

    :see: :class:`~.test_radial_params.TestMakeSlaterParams`

    Implementation Logic
    --------------------
    1. **Prepare the normalized values**::

           zeta_arr = jnp.asarray(zeta, dtype=jnp.float64)

       This expression gives the later validation steps a stable shape and
       dtype.

    2. **Apply static validation**::

           len(orbital_basis.n) != n_orbitals

       This predicate rejects invalid structure before JAX traces the
       numerical checks.

    3. **Apply traced validation**::

           ~jnp.all(jnp.isfinite(zeta_arr))

       This predicate remains active during eager and compiled execution.

    4. **Return the named instance**::

           return params

       The explicit name keeps the implementation and the Returns section
       synchronized.

    Parameters
    ----------
    zeta : Float[Array, " Oz"]
        Slater exponents (inverse Bohr), one per orbital.
    orbital_basis : OrbitalBasis
        Orbital quantum number metadata (**static** -- a compile-time
        constant; changing it triggers retracing). Must have the same number
        of orbitals as the length of ``zeta``.
    coefficients : Optional[Float[Array, "Oc C"]], optional
        Multi-zeta linear combination coefficients. Shape ``(O, C)``
        where C is the contraction length. Defaults to ones with
        C=1 (single-zeta basis).

    Returns
    -------
    params : SlaterParams
        Validated Slater parameters with ``float64`` arrays.

    Raises
    ------
    ValueError
        If ``zeta`` and ``orbital_basis`` contain different orbital counts.
        The function also rejects a different orbital count in
        ``coefficients``.
    EquinoxRuntimeError
        If ``zeta`` is non-finite or non-positive, or if
        ``coefficients`` contains a non-finite value.

    Notes
    -----
    Static validation raises ``ValueError`` before traced construction when
    the orbital count differs from either leading array dimension. Traced
    validation uses ``eqx.error_if`` and raises ``EquinoxRuntimeError`` when
    an exponent is non-finite or non-positive, or a coefficient is non-finite.

    See Also
    --------
    SlaterParams : The PyTree class constructed by this factory.
    make_orbital_basis : Factory for the ``OrbitalBasis`` argument.
    """
    zeta_arr: Float[Array, " O"] = jnp.asarray(zeta, dtype=jnp.float64)
    n_orbitals: int = zeta_arr.shape[0]
    if len(orbital_basis.n) != n_orbitals:
        msg: str = "zeta length must match orbital_basis size"
        raise ValueError(msg)
    if coefficients is None:
        coeff_arr: Float[Array, "O C"] = jnp.ones(
            (n_orbitals, 1), dtype=jnp.float64
        )
    else:
        coeff_arr = jnp.asarray(coefficients, dtype=jnp.float64)
    if coeff_arr.shape[0] != n_orbitals:
        msg = "coefficients first dimension must match orbital_basis size"
        raise ValueError(msg)

    def validate_and_create() -> SlaterParams:
        nonlocal zeta_arr
        zeta_arr = eqx.error_if(
            zeta_arr,
            ~(jnp.all(jnp.isfinite(zeta_arr))),
            "make_slater_params: zeta finite",
        )
        zeta_arr = eqx.error_if(
            zeta_arr,
            ~(jnp.all(zeta_arr > 0.0)),
            "make_slater_params: zeta positive",
        )
        coeff_arr_checked: Float[Array, "O C"] = eqx.error_if(
            coeff_arr,
            ~(jnp.all(jnp.isfinite(coeff_arr))),
            "make_slater_params: coefficients finite",
        )
        validated_params: SlaterParams = SlaterParams(
            zeta=zeta_arr,
            coefficients=coeff_arr_checked,
            orbital_basis=orbital_basis,
        )
        return validated_params

    params: SlaterParams = validate_and_create()
    return params


@jaxtyped(typechecker=beartype)
def make_slater_koster_params(  # noqa: DOC502, DOC503
    values: Float[Array, " n_sk"],
    keys: tuple[str, ...],
) -> SlaterKosterParams:
    """Create validated Slater--Koster two-center parameters.

    The factory normalizes numerical values and validates every static channel
    identifier before constructing the carrier.

    :see: :class:`~.test_radial_params.TestMakeSlaterKosterParams`

    Parameters
    ----------
    values : Float[Array, " n_sk"]
        Fundamental two-center hopping integrals in eV.
    keys : tuple[str, ...]
        Unique static identifiers, one for every value. Material builders use
        identifiers such as ``"C-C:pp_sigma"``.

    Returns
    -------
    params : SlaterKosterParams
        Parameter carrier with float64 differentiable values and static keys.

    Raises
    ------
    ValueError
        If values are not one-dimensional, keys are not a tuple, lengths
        differ, or a key is empty or duplicated.
    EquinoxRuntimeError
        If any value is non-finite, in eager or compiled execution.

    Notes
    -----
    Values may have either sign and may be zero. Only finiteness is a
    numerical invariant; channel and material semantics belong to the
    Slater--Koster model builder.

    See Also
    --------
    SlaterKosterParams : Carrier constructed by this factory.
    """
    value_array: Float[Array, " n_sk"] = jnp.asarray(
        values,
        dtype=jnp.float64,
    )
    _validate_slater_koster_structure(value_array, keys)
    value_array = eqx.error_if(
        value_array,
        ~jnp.all(jnp.isfinite(value_array)),
        "make_slater_koster_params: values finite",
    )
    params: SlaterKosterParams = SlaterKosterParams(
        values=value_array,
        keys=keys,
    )
    return params


__all__: list[str] = [
    "OrbitalBasis",
    "SlaterKosterParams",
    "SlaterParams",
    "make_orbital_basis",
    "make_slater_koster_params",
    "make_slater_params",
]
