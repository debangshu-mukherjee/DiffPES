"""Provide shared fixtures for the mirrored split test modules.

The helpers preserve the inputs and independent reference calculations.
"""

import jax.numpy as jnp
import numpy as np
from beartype.typing import Dict, Tuple
from jaxtyping import Array, Float64

from diffpes.types import (
    CrystalGeometry,
    OrbitalBasis,
    make_crystal_geometry,
    make_orbital_basis,
)

_TABLE_DIRECTIONS: int = 50


_ALL_SK_KEYS: Tuple[str, ...] = (
    "X-X:ss_sigma",
    "X-X:sp_sigma",
    "X-X:sd_sigma",
    "X-X:pp_sigma",
    "X-X:pp_pi",
    "X-X:pd_sigma",
    "X-X:pd_pi",
    "X-X:dd_sigma",
    "X-X:dd_pi",
    "X-X:dd_delta",
)


def _reference_d_tensors() -> Float64[Array, "5 3 3"]:
    """PRIVATE: Return the normalized Table-I d-orbital Cartesian tensors.

    Returns
    -------
    tensors : Float64[Array, "5 3 3"]
        Symmetric traceless rank-two tensors for the real d orbitals in
        the fixed order xy, yz, z2, xz, x2-y2, each normalized to unit
        Frobenius norm.

    Notes
    -----
    The off-diagonal tensors carry ``1/sqrt(2)`` entries and the z2
    tensor carries ``diag(-1, -1, 2)/sqrt(6)``. Contracting these
    tensors with direction cosines reproduces the Slater--Koster
    Table-I d polynomials independently of the production kernel.
    """
    inverse_sqrt_two: float = 1.0 / np.sqrt(2.0)
    inverse_sqrt_six: float = 1.0 / np.sqrt(6.0)
    tensors: Float64[Array, "5 3 3"] = jnp.asarray(
        (
            (
                (0.0, inverse_sqrt_two, 0.0),
                (inverse_sqrt_two, 0.0, 0.0),
                (0.0, 0.0, 0.0),
            ),
            (
                (0.0, 0.0, 0.0),
                (0.0, 0.0, inverse_sqrt_two),
                (0.0, inverse_sqrt_two, 0.0),
            ),
            (
                (-inverse_sqrt_six, 0.0, 0.0),
                (0.0, -inverse_sqrt_six, 0.0),
                (0.0, 0.0, 2.0 * inverse_sqrt_six),
            ),
            (
                (0.0, 0.0, inverse_sqrt_two),
                (0.0, 0.0, 0.0),
                (inverse_sqrt_two, 0.0, 0.0),
            ),
            (
                (inverse_sqrt_two, 0.0, 0.0),
                (0.0, -inverse_sqrt_two, 0.0),
                (0.0, 0.0, 0.0),
            ),
        ),
        dtype=jnp.float64,
    )
    return tensors


def _table_i_blocks(
    direction: Float64[Array, " 3"],
    values: Float64[Array, " 10"],
) -> Dict[Tuple[int, int], Float64[Array, "m1 m2"]]:
    """PRIVATE: Evaluate the direction-cosine polynomials for all ten
    channels.

    Parameters
    ----------
    direction : Float64[Array, " 3"]
        Unit bond direction cosines.
    values : Float64[Array, " 10"]
        The ten fundamental SK integrals in eV in the order ss_sigma,
        sp_sigma, sd_sigma, pp_sigma, pp_pi, pd_sigma, pd_pi, dd_sigma,
        dd_pi, dd_delta.

    Returns
    -------
    blocks : Dict[Tuple[int, int], Float64[Array, "m1 m2"]]
        Hopping blocks in eV for the six canonical shell pairs keyed by
        ``(l1, l2)`` with ``l1 <= l2``.

    Notes
    -----
    Builds each block from tensor identities instead of the tabulated
    entry-by-entry formulas. The p vector comes from projecting the
    direction onto the real p axes. The d sigma vector is
    ``sqrt(3/2)`` times the double tensor contraction with the
    direction. The d pi projector is twice the difference of the
    tensor-vector Gram matrix and the tensor-direction dyad. The dd
    block follows from the sigma, pi, and delta projector
    decomposition. Agreement with :func:`sk_block` on random
    directions therefore checks the production polynomials against an
    algebraically independent construction.
    """
    p_axes: Float64[Array, "3 3"] = jnp.asarray(
        ((0.0, 1.0, 0.0), (0.0, 0.0, 1.0), (1.0, 0.0, 0.0)),
        dtype=jnp.float64,
    )
    tensors: Float64[Array, "5 3 3"] = _reference_d_tensors()
    p_direction: Float64[Array, " 3"] = p_axes @ direction
    tensor_direction: Float64[Array, " 5"] = jnp.einsum(
        "aij,i,j->a",
        tensors,
        direction,
        direction,
    )
    d_sigma: Float64[Array, " 5"] = jnp.sqrt(3.0 / 2.0) * tensor_direction
    tensor_vectors: Float64[Array, "5 3"] = jnp.einsum(
        "aij,j->ai",
        tensors,
        direction,
    )
    p_dot_tensor: Float64[Array, "3 5"] = p_axes @ tensor_vectors.T
    pd_pi_coefficients: Float64[Array, "3 5"] = jnp.sqrt(2.0) * (
        p_dot_tensor - p_direction[:, None] * tensor_direction[None, :]
    )
    d_sigma_projector: Float64[Array, "5 5"] = jnp.outer(
        d_sigma,
        d_sigma,
    )
    d_pi_projector: Float64[Array, "5 5"] = 2.0 * (
        tensor_vectors @ tensor_vectors.T
        - jnp.outer(tensor_direction, tensor_direction)
    )
    d_identity: Float64[Array, "5 5"] = jnp.eye(5, dtype=jnp.float64)

    blocks: Dict[Tuple[int, int], Float64[Array, "m1 m2"]] = {
        (0, 0): values[0:1, None],
        (0, 1): values[1] * p_direction[None, :],
        (0, 2): values[2] * d_sigma[None, :],
        (1, 1): (
            values[4] * jnp.eye(3, dtype=jnp.float64)
            + (values[3] - values[4]) * jnp.outer(p_direction, p_direction)
        ),
        (1, 2): (
            values[5] * jnp.outer(p_direction, d_sigma)
            + values[6] * pd_pi_coefficients
        ),
        (2, 2): (
            values[9] * d_identity
            + (values[7] - values[9]) * d_sigma_projector
            + (values[8] - values[9]) * d_pi_projector
        ),
    }
    return blocks


def _graphene_geometry() -> CrystalGeometry:
    """PRIVATE: Construct the two-atom honeycomb geometry for the shell check.

    Returns
    -------
    geometry : CrystalGeometry
        Hexagonal two-carbon cell with lattice constant 2.46 Angstrom,
        a 10 Angstrom vacuum axis, and the B atom at fractional
        ``(1/3, 1/3, 0)``.

    Notes
    -----
    The neighbor-shell tests count hand-enumerable honeycomb
    coordination shells on this fixed geometry.
    """
    lattice_constant: float = 2.46
    lattice: Float64[Array, "3 3"] = jnp.asarray(
        (
            (lattice_constant, 0.0, 0.0),
            (
                lattice_constant / 2.0,
                lattice_constant * np.sqrt(3.0) / 2.0,
                0.0,
            ),
            (0.0, 0.0, 10.0),
        ),
        dtype=jnp.float64,
    )
    geometry: CrystalGeometry = make_crystal_geometry(
        lattice,
        jnp.asarray(
            ((0.0, 0.0, 0.0), (1.0 / 3.0, 1.0 / 3.0, 0.0)),
            dtype=jnp.float64,
        ),
        ("C", "C"),
    )
    return geometry


def _compact_spd_basis() -> OrbitalBasis:
    """PRIVATE: Construct generic s, px, and dxy orbitals on each of two
    atoms.

    Returns
    -------
    basis : OrbitalBasis
        Six-orbital basis with quantum numbers ``(l, m)`` equal to
        ``(0, 0)``, ``(1, 1)``, and ``(2, -2)`` on each atom.

    Notes
    -----
    One orbital from each angular-momentum sector activates every
    inter-shell SK channel pair without a complete shell, which keeps
    the model-builder tests compact.
    """
    basis: OrbitalBasis = make_orbital_basis(
        (0, 0, 0, 1, 1, 1),
        (1, 2, 3, 1, 2, 3),
        (0, 1, 2, 0, 1, 2),
        (0, 1, -2, 0, 1, -2),
    )
    return basis
