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

    Describes the orbital basis set used in matrix element
    calculations. All fields are static (auxiliary data) because
    quantum numbers control code paths: they determine recurrence
    depths in spherical Bessel functions and associated Legendre
    polynomials, and they index into the Gaunt coefficient table.

    Attributes
    ----------
    n_values : tuple[int, ...]
        Principal quantum numbers, one per orbital.
    l_values : tuple[int, ...]
        Angular momentum quantum numbers, one per orbital.
    m_values : tuple[int, ...]
        Magnetic quantum numbers, one per orbital.
    labels : tuple[str, ...]
        Human-readable orbital labels (e.g. ``("2s", "2px", ...)``).

    Notes
    -----
    Registered as a JAX PyTree with ``@register_pytree_node_class``.
    All fields are auxiliary data (no JAX array children) because
    changing any quantum number changes the computational graph and
    requires JIT recompilation.
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

        Returns
        -------
        children : tuple
            Empty tuple (no JAX array children).
        aux_data : tuple
            ``(n_values, l_values, m_values, labels)``.
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
        """Reconstruct an OrbitalBasis from flattened components.

        Parameters
        ----------
        aux_data : tuple
            ``(n_values, l_values, m_values, labels)``.
        _children : tuple
            Empty tuple (unused).

        Returns
        -------
        basis : OrbitalBasis
            Reconstructed instance.
        """
        n_values, l_values, m_values, labels = aux_data
        return cls(
            n_values=n_values,
            l_values=l_values,
            m_values=m_values,
            labels=labels,
        )


@register_pytree_node_class
class SlaterParams(NamedTuple):
    """PyTree for Slater radial wavefunction parameters.

    Wraps differentiable Slater exponents and linear combination
    coefficients alongside the static orbital basis metadata.
    The Slater exponents (zeta) are the primary quantities to be
    optimized in inverse fitting workflows.

    Attributes
    ----------
    zeta : Float[Array, " O"]
        Slater exponents, one per orbital. Differentiable.
    coefficients : Float[Array, "O C"]
        Linear combination coefficients for multi-zeta basis.
        C=1 for single-zeta. Differentiable.
    orbital_basis : OrbitalBasis
        Quantum numbers for each orbital. Static (auxiliary data).

    Notes
    -----
    Registered as a JAX PyTree with ``@register_pytree_node_class``.
    ``zeta`` and ``coefficients`` are JAX array children (on the
    gradient tape), while ``orbital_basis`` is auxiliary data
    (static at trace time).
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

        Returns
        -------
        children : tuple of Array
            ``(zeta, coefficients)``.
        aux_data : OrbitalBasis
            The orbital basis metadata.
        """
        return ((self.zeta, self.coefficients), self.orbital_basis)

    @classmethod
    def tree_unflatten(
        cls,
        aux_data: OrbitalBasis,
        children: Tuple[Float[Array, " O"], Float[Array, "O C"]],
    ) -> "SlaterParams":
        """Reconstruct a SlaterParams from flattened components.

        Parameters
        ----------
        aux_data : OrbitalBasis
            The orbital basis metadata.
        children : tuple of Array
            ``(zeta, coefficients)``.

        Returns
        -------
        params : SlaterParams
            Reconstructed instance.
        """
        zeta, coefficients = children
        return cls(
            zeta=zeta,
            coefficients=coefficients,
            orbital_basis=aux_data,
        )


@jaxtyped(typechecker=beartype)
def make_orbital_basis(
    n_values: tuple[int, ...],
    l_values: tuple[int, ...],
    m_values: tuple[int, ...],
    labels: Optional[tuple[str, ...]] = None,
) -> OrbitalBasis:
    """Create a validated OrbitalBasis instance.

    Parameters
    ----------
    n_values : tuple[int, ...]
        Principal quantum numbers.
    l_values : tuple[int, ...]
        Angular momentum quantum numbers.
    m_values : tuple[int, ...]
        Magnetic quantum numbers.
    labels : tuple[str, ...], optional
        Orbital labels. Defaults to ``("orb_0", "orb_1", ...)``.

    Returns
    -------
    basis : OrbitalBasis
        Validated orbital basis.
    """
    n_orbitals = len(n_values)
    if len(l_values) != n_orbitals or len(m_values) != n_orbitals:
        msg = "n_values, l_values, m_values must have the same length"
        raise ValueError(msg)
    if labels is None:
        labels = tuple(f"orb_{i}" for i in range(n_orbitals))
    if len(labels) != n_orbitals:
        msg = "labels must have the same length as quantum numbers"
        raise ValueError(msg)
    return OrbitalBasis(
        n_values=n_values,
        l_values=l_values,
        m_values=m_values,
        labels=labels,
    )


@jaxtyped(typechecker=beartype)
def make_slater_params(
    zeta: Float[Array, " O"],
    orbital_basis: OrbitalBasis,
    coefficients: Optional[Float[Array, "O C"]] = None,
) -> SlaterParams:
    """Create a validated SlaterParams instance.

    Parameters
    ----------
    zeta : Float[Array, " O"]
        Slater exponents, one per orbital.
    orbital_basis : OrbitalBasis
        Orbital quantum number metadata.
    coefficients : Float[Array, "O C"], optional
        Multi-zeta coefficients. Defaults to ones with C=1.

    Returns
    -------
    params : SlaterParams
        Validated Slater parameters with float64 arrays.
    """
    zeta_arr: Float[Array, " O"] = jnp.asarray(zeta, dtype=jnp.float64)
    n_orbitals = zeta_arr.shape[0]
    if len(orbital_basis.n_values) != n_orbitals:
        msg = "zeta length must match orbital_basis size"
        raise ValueError(msg)
    if coefficients is None:
        coeff_arr: Float[Array, "O C"] = jnp.ones(
            (n_orbitals, 1), dtype=jnp.float64
        )
    else:
        coeff_arr = jnp.asarray(coefficients, dtype=jnp.float64)
    return SlaterParams(
        zeta=zeta_arr,
        coefficients=coeff_arr,
        orbital_basis=orbital_basis,
    )


__all__: list[str] = [
    "OrbitalBasis",
    "SlaterParams",
    "make_orbital_basis",
    "make_slater_params",
]
