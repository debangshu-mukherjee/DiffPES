"""Compute orbital angular momentum.

Extended Summary
----------------
The module computes the z-component of orbital angular momentum (OAM)
from orbital projections, separating p-orbital and d-orbital
contributions.

Routine Listings
----------------
:func:`compute_oam`
    Compute orbital angular momentum z-component.

Notes
-----
``OAM_z = sum(m * |projection(m)|^2)`` where m is the magnetic
quantum number. For p-orbitals, m = {+1, 0, -1} corresponding
to px+ipy, pz, px-ipy. For d-orbitals, m = {-2, -1, 0, +1, +2}.
"""

import jax.numpy as jnp
from beartype import beartype
from jaxtyping import Array, Float64, jaxtyped

from diffpes.types import (
    D_ORBITAL_SLICE,
    M_D,
    M_P,
    P_ORBITAL_SLICE,
)


@jaxtyped(typechecker=beartype)
def compute_oam(
    projections: Float64[Array, "K B A 9"],
) -> Float64[Array, "K B A 3"]:
    """Compute orbital angular momentum z-component.

    Evaluates the expectation value of the z-component of orbital
    angular momentum from orbital-resolved projections::

        OAM_z = sum_m  m * |c_m|^2

    Here, m is the magnetic quantum number, and c_m is the orbital projection
    coefficient. The function computes the p-orbital and d-orbital
    contributions separately. It then adds them.

    :see: :class:`~.test_oam.TestComputeOam`

    Implementation Logic
    --------------------
    1. **Extract the p-orbital projections**::

           p_proj: Float64[Array, "K B A 3"] = projections[
               ..., P_ORBITAL_SLICE
           ]

       Selects the three p-orbital coefficients [py, pz, px]
       corresponding to magnetic quantum numbers m = {+1, 0, -1}.

    2. **Compute the p-orbital OAM**::

           p_oam: Float64[Array, "K B A"] = jnp.sum(
               M_P * p_proj**2, axis=-1
           )

       Weights each squared projection by its magnetic quantum
       number m_p = [+1, 0, -1] and sums over the p-orbital
       subspace.

    3. **Extract the d-orbital projections**::

           d_proj: Float64[Array, "K B A 5"] = projections[
               ..., D_ORBITAL_SLICE
           ]

       Selects the five d-orbital coefficients [dxy, dyz, dz2,
       dxz, dx2-y2] corresponding to m = {-2, -1, 0, +1, +2}.

    4. **Compute the d-orbital OAM**::

           d_oam: Float64[Array, "K B A"] = jnp.sum(
               M_D * d_proj**2, axis=-1
           )

       Weights each squared projection by its magnetic quantum
       number m_d = [-2, -1, 0, +1, +2] and sums over the
       d-orbital subspace.

    5. **Stack the p, d, and total results**::

           total_oam: Float64[Array, "K B A"] = p_oam + d_oam
           oam: Float64[Array, "K B A 3"] = jnp.stack(
               [p_oam, d_oam, total_oam], axis=-1
           )

       Returns all three components so that downstream analysis
       can inspect orbital-resolved or total OAM.

    Parameters
    ----------
    projections : Float64[Array, "K B A 9"]
        Orbital projections with 9 orbitals per atom.

    Returns
    -------
    oam : Float64[Array, "K B A 3"]
        OAM_z for [p-contribution, d-contribution, total].

    Notes
    -----
    Orbital indices follow VASP ordering: [s(0), py(1), pz(2),
    px(3), dxy(4), dyz(5), dz2(6), dxz(7), dx2-y2(8)].
    The s-orbital (index 0) has m = 0 and does not contribute to
    the OAM.
    """
    p_proj: Float64[Array, "K B A 3"] = projections[..., P_ORBITAL_SLICE]
    p_oam: Float64[Array, "K B A"] = jnp.sum(M_P * p_proj**2, axis=-1)
    d_proj: Float64[Array, "K B A 5"] = projections[..., D_ORBITAL_SLICE]
    d_oam: Float64[Array, "K B A"] = jnp.sum(M_D * d_proj**2, axis=-1)
    total_oam: Float64[Array, "K B A"] = p_oam + d_oam
    oam: Float64[Array, "K B A 3"] = jnp.stack(
        [p_oam, d_oam, total_oam], axis=-1
    )
    return oam


__all__: list[str] = [
    "compute_oam",
]
