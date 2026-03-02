"""Radial wavefunction parameter data structures.

Extended Summary
----------------
Defines PyTree types for orbital basis metadata and Slater-type
radial wavefunction parameters used by the differentiable dipole
matrix element pipeline.

Routine Listings
----------------
:class:`OrbitalBasis`
    PyTree for orbital quantum number metadata.
:class:`SlaterParams`
    PyTree for differentiable Slater radial parameters.
:func:`make_orbital_basis`
    Factory for OrbitalBasis.
:func:`make_slater_params`
    Factory for SlaterParams.

Notes
-----
``OrbitalBasis`` is purely static (all auxiliary data) because the
quantum numbers (n, l, m) control code paths (recurrence depths in
spherical Bessel functions and associated Legendre polynomials).
``SlaterParams`` wraps differentiable Slater exponents alongside
the static orbital basis.
"""

import jax.numpy as jnp
from beartype import beartype
from beartype.typing import NamedTuple, Optional, Tuple
from jax.tree_util import register_pytree_node_class
from jaxtyping import Array, Float, jaxtyped


@register_pytree_node_class
class OrbitalBasis(NamedTuple):
    """PyTree for orbital quantum number metadata.

    Extended Summary
    ----------------
    Describes the orbital basis set used in dipole matrix element
    calculations for the differentiable Chinook pipeline. The quantum
    numbers (n, l, m) parameterize the radial wavefunctions (via
    Slater-type orbitals) and angular parts (spherical harmonics) that
    enter the photoemission matrix element.

    All fields are static (auxiliary data) because quantum numbers
    control code paths: they determine recurrence depths in spherical
    Bessel functions and associated Legendre polynomials, and they
    index into the Gaunt coefficient table. Changing any quantum
    number alters the computational graph structure, so JAX must
    recompile when these values change.

    Attributes
    ----------
    n_values : tuple[int, ...]
        Principal quantum numbers, one per orbital. Determines the
        radial node count and the power of *r* in the Slater-type
        radial function R_nl(r) ~ r^{n-1} exp(-zeta*r). Static
        (auxiliary data, not JAX-traced).
    l_values : tuple[int, ...]
        Angular momentum quantum numbers, one per orbital (0=s, 1=p,
        2=d, 3=f). Determines the spherical harmonic Y_l^m used in
        the matrix element integral. Static (auxiliary data).
    m_values : tuple[int, ...]
        Magnetic quantum numbers, one per orbital. Ranges from -l to
        +l for each orbital. Selects the specific spherical harmonic
        component. Static (auxiliary data).
    labels : tuple[str, ...]
        Human-readable orbital labels (e.g. ``("2s", "2px", ...)``).
        Used for plotting and debugging. Static (auxiliary data).

    Notes
    -----
    Registered as a JAX PyTree with ``@register_pytree_node_class``.
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

    n_values: tuple[int, ...]
    l_values: tuple[int, ...]
    m_values: tuple[int, ...]
    labels: tuple[str, ...]

    def tree_flatten(
        self,
    ) -> Tuple[
        Tuple[()],
        Tuple[
            tuple[int, ...],
            tuple[int, ...],
            tuple[int, ...],
            tuple[str, ...],
        ],
    ]:
        """Flatten into JAX children and auxiliary data.

        Separates the PyTree into children (JAX-traced arrays) and
        auxiliary data (static Python values). For ``OrbitalBasis``,
        there are *no* JAX array children -- all fields are static.

        Implementation Logic
        --------------------
        1. **Children**: empty tuple ``()`` -- there are no numerical
           JAX arrays in this type. All quantum number data is static.
        2. **Auxiliary data**: ``(n_values, l_values, m_values,
           labels)`` -- four tuples of Python ints/strings. JAX
           treats these as compile-time constants; any change triggers
           JIT recompilation.

        Returns
        -------
        children : tuple
            Empty tuple (no JAX array children).
        aux_data : tuple
            ``(n_values, l_values, m_values, labels)`` -- all four
            fields packed as static auxiliary data.
        """
        return ((), (self.n_values, self.l_values, self.m_values, self.labels))

    @classmethod
    def tree_unflatten(
        cls,
        aux_data: Tuple[
            tuple[int, ...],
            tuple[int, ...],
            tuple[int, ...],
            tuple[str, ...],
        ],
        _children: Tuple[()],
    ) -> "OrbitalBasis":
        """Reconstruct an ``OrbitalBasis`` from flattened components.

        Inverse of :meth:`tree_flatten`. JAX calls this method
        automatically when unflattening a PyTree after a
        transformation. Because ``OrbitalBasis`` has no array
        children, this method simply reconstructs from auxiliary data.

        Implementation Logic
        --------------------
        1. **Children**: ignored (always an empty tuple) because
           ``OrbitalBasis`` has no JAX array fields.
        2. **Reconstruction**: unpacks ``aux_data`` into the four
           static tuples ``(n_values, l_values, m_values, labels)``
           and passes them to the ``NamedTuple`` constructor.

        Parameters
        ----------
        aux_data : tuple
            ``(n_values, l_values, m_values, labels)`` recovered
            from auxiliary data.
        _children : tuple
            Empty tuple (unused -- no JAX array children).

        Returns
        -------
        basis : OrbitalBasis
            Reconstructed instance with identical quantum numbers.
        """
        n_values: tuple[int, ...]
        l_values: tuple[int, ...]
        m_values: tuple[int, ...]
        labels: tuple[str, ...]
        n_values, l_values, m_values, labels = aux_data
        basis: OrbitalBasis = cls(
            n_values=n_values,
            l_values=l_values,
            m_values=m_values,
            labels=labels,
        )
        return basis


@register_pytree_node_class
class SlaterParams(NamedTuple):
    """PyTree for Slater radial wavefunction parameters.

    Extended Summary
    ----------------
    Wraps differentiable Slater exponents and linear combination
    coefficients alongside the static orbital basis metadata.
    The Slater exponents (zeta) are the primary quantities to be
    optimized in inverse fitting workflows -- ``jax.grad`` with
    respect to ``zeta`` gives the gradient of the simulated ARPES
    intensity with respect to the radial wavefunction shape.

    In a multi-zeta basis (C > 1), each orbital's radial function
    is expressed as a linear combination of C Slater-type functions
    with different exponents, weighted by the ``coefficients``
    matrix. For single-zeta (C=1), each orbital has exactly one
    exponent and a coefficient of 1.0.

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
        Static (auxiliary data) -- changing quantum numbers
        triggers JIT recompilation.

    Notes
    -----
    Registered as a JAX PyTree with ``@register_pytree_node_class``.
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
    orbital_basis: OrbitalBasis

    def tree_flatten(
        self,
    ) -> Tuple[
        Tuple[Float[Array, " O"], Float[Array, "O C"]],
        OrbitalBasis,
    ]:
        """Flatten into JAX children and auxiliary data.

        Separates JAX-traced arrays (children) from static Python
        values (auxiliary data) for ``jax.tree_util`` compatibility.

        Implementation Logic
        --------------------
        1. **Children** (JAX arrays, participate in autodiff):
           ``(zeta, coefficients)`` -- both are dense float64 arrays
           that live on the gradient tape. ``jax.grad`` sees and
           differentiates through these leaves.
        2. **Auxiliary data** (static, not traced by JAX):
           ``orbital_basis`` -- an ``OrbitalBasis`` NamedTuple of
           Python ints and strings. Because ``OrbitalBasis`` is itself
           a registered PyTree with all-auxiliary fields, it nests
           cleanly as static data here.

        Returns
        -------
        children : tuple of Array
            ``(zeta, coefficients)`` -- differentiable JAX arrays.
        aux_data : OrbitalBasis
            The orbital basis metadata (quantum numbers and labels).
        """
        return ((self.zeta, self.coefficients), self.orbital_basis)

    @classmethod
    def tree_unflatten(
        cls,
        aux_data: OrbitalBasis,
        children: Tuple[Float[Array, " O"], Float[Array, "O C"]],
    ) -> "SlaterParams":
        """Reconstruct a ``SlaterParams`` from flattened components.

        Inverse of :meth:`tree_flatten`. JAX calls this method
        automatically when unflattening a PyTree after a
        transformation (e.g., inside ``jax.jit`` or ``jax.grad``).

        Implementation Logic
        --------------------
        1. Unpack ``children`` into ``(zeta, coefficients)``.
        2. Receive ``aux_data`` as the ``OrbitalBasis`` instance
           carrying the static quantum number metadata.
        3. Pass all three fields to the constructor, restoring
           ``orbital_basis`` from ``aux_data``.

        Parameters
        ----------
        aux_data : OrbitalBasis
            The orbital basis metadata recovered from auxiliary data.
        children : tuple of Array
            ``(zeta, coefficients)`` as returned by
            :meth:`tree_flatten`.

        Returns
        -------
        params : SlaterParams
            Reconstructed instance with the original array and basis
            data.
        """
        zeta: Float[Array, " O"]
        coefficients: Float[Array, "O C"]
        zeta, coefficients = children
        params: SlaterParams = cls(
            zeta=zeta,
            coefficients=coefficients,
            orbital_basis=aux_data,
        )
        return params


@jaxtyped(typechecker=beartype)
def make_orbital_basis(
    n_values: tuple[int, ...],
    l_values: tuple[int, ...],
    m_values: tuple[int, ...],
    labels: Optional[tuple[str, ...]] = None,
) -> OrbitalBasis:
    """Create a validated ``OrbitalBasis`` instance.

    Extended Summary
    ----------------
    Factory function that validates quantum number tuples and
    constructs an ``OrbitalBasis`` PyTree. The three quantum number
    tuples must all have the same length (one entry per orbital).
    If ``labels`` is not provided, generic labels ``"orb_0"``,
    ``"orb_1"``, ... are generated automatically.

    Use this factory instead of the raw ``OrbitalBasis`` constructor
    to get automatic length validation and default label generation.

    Implementation Logic
    --------------------
    1. **Infer orbital count** from ``len(n_values)``.
    2. **Validate lengths**: raise ``ValueError`` if ``l_values`` or
       ``m_values`` differ in length from ``n_values``.
    3. **Default labels**: if ``labels`` is ``None``, generate
       ``("orb_0", "orb_1", ...)`` with length matching the orbital
       count.
    4. **Validate label length**: raise ``ValueError`` if ``labels``
       has a different length from the quantum number tuples.
    5. **Construct** the ``OrbitalBasis`` NamedTuple and return it.

    Parameters
    ----------
    n_values : tuple[int, ...]
        Principal quantum numbers, one per orbital.
    l_values : tuple[int, ...]
        Angular momentum quantum numbers, one per orbital.
    m_values : tuple[int, ...]
        Magnetic quantum numbers, one per orbital.
    labels : tuple[str, ...], optional
        Human-readable orbital labels. Defaults to
        ``("orb_0", "orb_1", ...)``.

    Returns
    -------
    basis : OrbitalBasis
        Validated orbital basis with consistent lengths.

    Raises
    ------
    ValueError
        If ``n_values``, ``l_values``, and ``m_values`` do not all
        have the same length, or if ``labels`` has a mismatched
        length.

    See Also
    --------
    OrbitalBasis : The PyTree class constructed by this factory.
    """
    n_orbitals: int = len(n_values)
    if len(l_values) != n_orbitals or len(m_values) != n_orbitals:
        msg: str = "n_values, l_values, m_values must have the same length"
        raise ValueError(msg)
    if labels is None:
        labels = tuple(f"orb_{i}" for i in range(n_orbitals))
    if len(labels) != n_orbitals:
        msg: str = "labels must have the same length as quantum numbers"
        raise ValueError(msg)
    basis: OrbitalBasis = OrbitalBasis(
        n_values=n_values,
        l_values=l_values,
        m_values=m_values,
        labels=labels,
    )
    return basis


@jaxtyped(typechecker=beartype)
def make_slater_params(
    zeta: Float[Array, " O"],
    orbital_basis: OrbitalBasis,
    coefficients: Optional[Float[Array, "O C"]] = None,
) -> SlaterParams:
    """Create a validated ``SlaterParams`` instance.

    Extended Summary
    ----------------
    Factory function that validates Slater radial wavefunction
    parameters and constructs a ``SlaterParams`` PyTree. The ``zeta``
    array length must match the number of orbitals in the provided
    ``orbital_basis``. If ``coefficients`` is not provided, a
    single-zeta basis is assumed and a column of ones is created
    (shape ``(O, 1)``).

    Use this factory instead of the raw ``SlaterParams`` constructor
    to get automatic size validation, default coefficient generation,
    and guaranteed ``float64`` dtypes.

    Implementation Logic
    --------------------
    1. **Cast zeta** to ``jnp.float64`` via ``jnp.asarray``.
    2. **Infer orbital count** from ``zeta_arr.shape[0]``.
    3. **Validate consistency**: raise ``ValueError`` if the orbital
       count from ``zeta`` does not match
       ``len(orbital_basis.n_values)``.
    4. **Default coefficients**: if ``coefficients`` is ``None``,
       create ``jnp.ones((O, 1), dtype=jnp.float64)`` for a
       single-zeta expansion. Otherwise cast the provided array to
       ``jnp.float64``.
    5. **Construct** the ``SlaterParams`` NamedTuple and return it.

    Parameters
    ----------
    zeta : Float[Array, " O"]
        Slater exponents (inverse Bohr), one per orbital.
    orbital_basis : OrbitalBasis
        Orbital quantum number metadata. Must have the same number
        of orbitals as the length of ``zeta``.
    coefficients : Float[Array, "O C"], optional
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
        If the length of ``zeta`` does not match the number of
        orbitals in ``orbital_basis``.

    See Also
    --------
    SlaterParams : The PyTree class constructed by this factory.
    make_orbital_basis : Factory for the ``OrbitalBasis`` argument.
    """
    zeta_arr: Float[Array, " O"] = jnp.asarray(zeta, dtype=jnp.float64)
    n_orbitals: int = zeta_arr.shape[0]
    if len(orbital_basis.n_values) != n_orbitals:
        msg: str = "zeta length must match orbital_basis size"
        raise ValueError(msg)
    if coefficients is None:
        coeff_arr: Float[Array, "O C"] = jnp.ones(
            (n_orbitals, 1), dtype=jnp.float64
        )
    else:
        coeff_arr = jnp.asarray(coefficients, dtype=jnp.float64)
    params: SlaterParams = SlaterParams(
        zeta=zeta_arr,
        coefficients=coeff_arr,
        orbital_basis=orbital_basis,
    )
    return params


__all__: list[str] = [
    "OrbitalBasis",
    "SlaterParams",
    "make_orbital_basis",
    "make_slater_params",
]
