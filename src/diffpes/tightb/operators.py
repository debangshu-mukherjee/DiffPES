r"""Construct Hermitian observables in a tight-binding orbital basis.

Extended Summary
----------------
The builders in this module construct spin, atomic :math:`L\cdot S`, and
orbital-selection operators using the package-wide real-harmonic and spinor
conventions.

Routine Listings
----------------
:func:`spin_operator`
    Construct :math:`S_{\widehat n}=\widehat n\cdot\sigma/2`.
:func:`ls_operator`
    Construct unit-strength atomic :math:`L\cdot S` by shell.
:func:`orbital_projector`
    Construct a diagonal projector onto selected basis orbitals.
"""

import equinox as eqx
import jax.numpy as jnp
from beartype import beartype
from jaxtyping import Array, Complex, Float, jaxtyped

from diffpes.types import EPS, OrbitalBasis

from .soc import soc_matrix


def _validate_spin_pairs(basis: OrbitalBasis) -> None:
    """Require one down and one up state for every spatial orbital."""
    n_orbitals: int = len(basis.n)
    if len(basis.spin) != n_orbitals:
        message: str = "spin_operator requires an explicit spinor basis"
        raise ValueError(message)
    groups: dict[tuple[int, int, int, int], list[int]] = {}
    orbital: int
    for orbital in range(n_orbitals):
        key: tuple[int, int, int, int] = (
            basis.atom_indices[orbital],
            basis.n[orbital],
            basis.l[orbital],
            basis.m[orbital],
        )
        groups.setdefault(key, []).append(basis.spin[orbital])
    if any(sorted(spins) != [-1, 1] for spins in groups.values()):
        message = (
            "spin_operator requires exactly one -1 and one +1 state for "
            "every spatial orbital"
        )
        raise ValueError(message)


def _validate_orbital_selection(
    basis: OrbitalBasis,
    orbital_select: tuple[int, ...],
) -> None:
    """Validate one static orbital selection."""
    if type(orbital_select) is not tuple:
        message: str = "orbital_select must be a tuple"
        raise ValueError(message)
    if not orbital_select:
        message = "orbital_select must contain at least one index"
        raise ValueError(message)
    if any(type(index) is not int for index in orbital_select):
        message = "orbital_select must contain integers"
        raise ValueError(message)
    n_orbitals: int = len(basis.n)
    if any(index < 0 or index >= n_orbitals for index in orbital_select):
        message = f"orbital_select indices must lie in [0, {n_orbitals})"
        raise ValueError(message)
    if len(set(orbital_select)) != len(orbital_select):
        message = "orbital_select must not contain duplicate indices"
        raise ValueError(message)


@jaxtyped(typechecker=beartype)
def spin_operator(  # noqa: DOC502 -- validation is delegated.
    basis: OrbitalBasis,
    axis: Float[Array, " 3"],
) -> Complex[Array, "n_so n_so"]:
    r"""Construct :math:`S_{\widehat n}=\widehat n\cdot\sigma/2`.

    The operator follows the declared ``(down, up)`` convention, including
    :math:`\sigma_y=[[0,i],[-i,0]]` and
    :math:`\sigma_z=\operatorname{diag}(-1,+1)`. Spatial partners are matched
    from basis metadata, so spin blocks need not be contiguous.

    :see: :class:`~.test_operators.TestSpinOperator`

    Parameters
    ----------
    basis : OrbitalBasis
        Spinor basis containing one ``-1`` and one ``+1`` state for every
        spatial orbital.
    axis : Float[Array, "3"]
        Finite Cartesian unit vector in the sample frame.

    Returns
    -------
    operator : Complex[Array, "n_so n_so"]
        Hermitian spin-one-half operator in the supplied basis order.

    Raises
    ------
    ValueError
        If the basis does not contain complete, unique spin pairs.
    EquinoxRuntimeError
        If ``axis`` is non-finite or not a unit vector.

    Notes
    -----
    Static orbital metadata identifies the partner pairs. Array operations
    then place the three Pauli components in the supplied basis order.
    """
    _validate_spin_pairs(basis)
    checked_axis: Float[Array, " 3"] = eqx.error_if(
        axis,
        ~jnp.all(jnp.isfinite(axis)),
        "spin_operator: axis must be finite",
    )
    norm_squared: Float[Array, ""] = jnp.sum(checked_axis * checked_axis)
    checked_axis = eqx.error_if(
        checked_axis,
        ~jnp.isclose(norm_squared, 1.0, rtol=EPS, atol=EPS),
        "spin_operator: axis must be a unit vector",
    )

    keys: tuple[tuple[int, int, int, int], ...] = tuple(
        (
            basis.atom_indices[index],
            basis.n[index],
            basis.l[index],
            basis.m[index],
        )
        for index in range(len(basis.n))
    )
    same_spatial: Float[Array, "n_so n_so"] = jnp.asarray(
        [
            [float(row_key == column_key) for column_key in keys]
            for row_key in keys
        ],
        dtype=jnp.float64,
    )
    spin: Float[Array, " n_so"] = jnp.asarray(
        basis.spin,
        dtype=jnp.float64,
    )
    row_spin: Float[Array, "n_so 1"] = spin[:, None]
    column_spin: Float[Array, "1 n_so"] = spin[None, :]
    sigma_x: Complex[Array, "n_so n_so"] = (
        same_spatial * (row_spin != column_spin)
    ).astype(jnp.complex128)
    sigma_y: Complex[Array, "n_so n_so"] = same_spatial * jnp.where(
        (row_spin == -1.0) & (column_spin == 1.0),
        1.0j,
        jnp.where(
            (row_spin == 1.0) & (column_spin == -1.0),
            -1.0j,
            0.0j,
        ),
    )
    sigma_z: Complex[Array, "n_so n_so"] = (
        same_spatial * (row_spin == column_spin) * row_spin
    ).astype(jnp.complex128)
    operator: Complex[Array, "n_so n_so"] = 0.5 * (
        checked_axis[0] * sigma_x
        + checked_axis[1] * sigma_y
        + checked_axis[2] * sigma_z
    )
    return operator


@jaxtyped(typechecker=beartype)
def ls_operator(  # noqa: DOC502 -- validation is delegated.
    basis: OrbitalBasis,
    shell_index: tuple[int, ...],
) -> Complex[Array, "n_so n_so"]:
    r"""Construct unit-strength atomic :math:`L\cdot S` by shell.

    Use the SOC shell map while replacing every physical coupling with one.
    The resulting operator exposes atomic angular correlations directly.

    :see: :class:`~.test_operators.TestLsOperator`

    Parameters
    ----------
    basis : OrbitalBasis
        Real-harmonic spinor basis.
    shell_index : tuple[int, ...]
        Static orbital-to-shell IDs. ``-1`` excludes an orbital; nonnegative
        IDs must follow the :func:`diffpes.tightb.soc.soc_matrix` contract.

    Returns
    -------
    operator : Complex[Array, "n_so n_so"]
        Hermitian sum of unit-strength :math:`L\cdot S` shell blocks.

    Raises
    ------
    ValueError
        If shell metadata or spin partners violate the SOC contract.

    Notes
    -----
    The result is dimensionless in units with :math:`\hbar=1`. Multiplication
    by shell-specific couplings is part of Hamiltonian construction, not this
    diagnostic operator.
    """
    n_shells: int = max(shell_index, default=-1) + 1
    strengths: Float[Array, " n_shells"] = jnp.ones(
        (n_shells,),
        dtype=jnp.float64,
    )
    operator: Complex[Array, "n_so n_so"] = soc_matrix(
        basis,
        shell_index,
        strengths,
    )
    return operator


@jaxtyped(typechecker=beartype)
def orbital_projector(  # noqa: DOC502 -- validation is delegated.
    basis: OrbitalBasis,
    orbital_select: tuple[int, ...],
) -> Complex[Array, "n n"]:
    """Construct a diagonal projector onto selected basis orbitals.

    Mark each requested static index with one and leave all other diagonal
    entries at zero.

    :see: :class:`~.test_operators.TestOrbitalProjector`

    Parameters
    ----------
    basis : OrbitalBasis
        Model basis defining the operator dimension.
    orbital_select : tuple[int, ...]
        Fixed unique orbital indices (**static** -- changing them retraces).

    Returns
    -------
    projector : Complex[Array, "n n"]
        Hermitian idempotent diagonal selection matrix.

    Raises
    ------
    ValueError
        If the selection is empty, duplicated, or outside the basis.

    Notes
    -----
    The static tuple controls selection and therefore participates in JAX
    tracing. The returned complex matrix remains a numerical array leaf.
    """
    _validate_orbital_selection(basis, orbital_select)
    diagonal: Float[Array, " n"] = jnp.zeros(
        (len(basis.n),),
        dtype=jnp.float64,
    )
    diagonal = diagonal.at[jnp.asarray(orbital_select)].set(1.0)
    projector: Complex[Array, "n n"] = jnp.diag(diagonal).astype(
        jnp.complex128
    )
    return projector


__all__: list[str] = [
    "ls_operator",
    "orbital_projector",
    "spin_operator",
]
