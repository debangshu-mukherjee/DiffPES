"""Orbital angular momentum calculation.

Extended Summary
----------------
Computes the z-component of orbital angular momentum (OAM)
from orbital projections, separating p-orbital and d-orbital
contributions.

Routine Listings
----------------
:func:`compute_oam`
    Compute OAM_z from orbital projections.

Notes
-----
OAM_z = sum(m * |projection(m)|^2) where m is the magnetic
quantum number. For p-orbitals, m = {+1, 0, -1} corresponding
to px+ipy, pz, px-ipy. For d-orbitals, m = {-2, -1, 0, +1, +2}.
"""

import jax.numpy as jnp
from beartype import beartype
from jaxtyping import Array, Float, jaxtyped

_M_P: Float[Array, " 3"] = jnp.array(
    [1.0, 0.0, -1.0], dtype=jnp.float64
)

_M_D: Float[Array, " 5"] = jnp.array(
    [-2.0, -1.0, 0.0, 1.0, 2.0], dtype=jnp.float64
)


@jaxtyped(typechecker=beartype)
def compute_oam(
    projections: Float[Array, "K B A 9"],
) -> Float[Array, "K B A 3"]:
    """Compute orbital angular momentum z-component.

    Parameters
    ----------
    projections : Float[Array, "K B A 9"]
        Orbital projections with 9 orbitals per atom.

    Returns
    -------
    oam : Float[Array, "K B A 3"]
        OAM_z for [p-contribution, d-contribution, total].

    Notes
    -----
    Orbital indices: [s(0), py(1), pz(2), px(3),
    dxy(4), dyz(5), dz2(6), dxz(7), dx2-y2(8)].
    p-orbitals use indices 1-3, d-orbitals use indices 4-8.
    """
    p_proj: Float[Array, "K B A 3"] = projections[
        ..., 1:4
    ]
    p_oam: Float[Array, "K B A"] = jnp.sum(
        _M_P * p_proj**2, axis=-1
    )
    d_proj: Float[Array, "K B A 5"] = projections[
        ..., 4:9
    ]
    d_oam: Float[Array, "K B A"] = jnp.sum(
        _M_D * d_proj**2, axis=-1
    )
    total_oam: Float[Array, "K B A"] = p_oam + d_oam
    oam: Float[Array, "K B A 3"] = jnp.stack(
        [p_oam, d_oam, total_oam], axis=-1
    )
    return oam


__all__: list[str] = [
    "compute_oam",
]
