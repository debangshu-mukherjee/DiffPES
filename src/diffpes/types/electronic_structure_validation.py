"""Validate shared electronic-structure geometry.

Extended Summary
----------------
This private module validates geometry, orbital assignments, and
optional depths for tight-binding and diagonalized-band carriers.
"""

import equinox as eqx
import jax.numpy as jnp
from beartype.typing import Optional
from jaxtyping import Array, Float64

from .geometry import CrystalGeometry
from .orbital_basis import OrbitalBasis


def _validate_depths_shape(
    depths: Optional[Float64[Array, " n_depth"]],
    n_orbitals: int,
) -> None:
    """PRIVATE: Validate the optional orbital-depth axis.

    Parameters
    ----------
    depths : Optional[Float64[Array, " n_depth"]]
        Orbital depths in Angstrom below the top surface, or ``None``
        for a bulk model.
    n_orbitals : int
        Number of orbitals the depth axis must match.

    Raises
    ------
    ValueError
        If ``depths`` is present and is not one-dimensional with one
        entry per orbital. This is the static construction-time
        contract.

    Notes
    -----
    ``None`` passes untouched. Checks only static shape metadata here.
    The factory keeps value nonnegativity traced.
    """
    if depths is not None and (
        depths.ndim != 1 or depths.shape[0] != n_orbitals
    ):
        message: str = "depths must have shape (n_orbitals,)"
        raise ValueError(message)


def _validate_basis_geometry(
    basis: OrbitalBasis,
    geometry: CrystalGeometry,
) -> None:
    """PRIVATE: Validate the orbital-to-atom mapping against a geometry.

    Parameters
    ----------
    basis : OrbitalBasis
        Orbital metadata that supplies ``atom_indices`` for checking.
    geometry : CrystalGeometry
        Crystal geometry that provides the atomic position rows.

    Raises
    ------
    ValueError
        If any ``basis.atom_indices`` entry is not a valid row index of
        ``geometry.positions``. This is the static construction-time
        contract.

    Notes
    -----
    Compare each index against the static atom count
    ``geometry.positions.shape[0]``. Nonnegativity is already
    guaranteed by the ``OrbitalBasis`` invariants.
    """
    n_atoms: int = geometry.positions.shape[0]
    if any(index >= n_atoms for index in basis.atom_indices):
        message: str = (
            "basis atom_indices must refer to geometry.positions rows"
        )
        raise ValueError(message)


def _checked_geometry(
    geometry: CrystalGeometry, context: str
) -> CrystalGeometry:
    """PRIVATE: Attach finite-value runtime checks to every geometry leaf.

    Parameters
    ----------
    geometry : CrystalGeometry
        Geometry that supplies ``lattice``, ``reciprocal``, and
        ``positions`` leaves for guarding.
    context : str
        Caller name used as the prefix of each traced error message.

    Returns
    -------
    checked : CrystalGeometry
        The same geometry with the runtime checks attached.

    Notes
    -----
    Attach one traced ``eqx.error_if`` guard per array leaf instead of
    raising a static ``ValueError``. Each guard fails at run time under
    JIT when its leaf contains a nonfinite element. Rebuild the carrier
    with ``eqx.tree_at`` so that the guarded leaves replace the
    originals.
    """
    lattice: Float64[Array, "3 3"] = eqx.error_if(
        geometry.lattice,
        ~jnp.all(jnp.isfinite(geometry.lattice)),
        f"{context}: geometry lattice finite",
    )
    reciprocal: Float64[Array, "3 3"] = eqx.error_if(
        geometry.reciprocal,
        ~jnp.all(jnp.isfinite(geometry.reciprocal)),
        f"{context}: geometry reciprocal finite",
    )
    positions: Float64[Array, "n_atoms 3"] = eqx.error_if(
        geometry.positions,
        ~jnp.all(jnp.isfinite(geometry.positions)),
        f"{context}: geometry positions finite",
    )
    checked: CrystalGeometry = eqx.tree_at(
        lambda item: (item.lattice, item.reciprocal, item.positions),
        geometry,
        (lattice, reciprocal, positions),
    )
    return checked


__all__: list[str] = []
