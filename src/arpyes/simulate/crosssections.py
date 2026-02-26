"""Photoionization cross-section weights for ARPES.

Extended Summary
----------------
Provides orbital-dependent photoionization cross-section
calculations based on Yeh-Lindau tabulated data and simple
heuristic models for different photon energies.

Routine Listings
----------------
:func:`heuristic_weights`
    Energy-dependent heuristic orbital weights.
:func:`yeh_lindau_weights`
    Interpolated Yeh-Lindau cross-section weights.

Notes
-----
Cross-section data is based on simplified tabulations from:
Yeh & Lindau, Atomic Data and Nuclear Data Tables 32, 1-155
(1985). The tabulated values at 20, 40, 60 eV provide
approximate cross-sections for s, p, and d orbitals.
"""

import jax.numpy as jnp
from beartype import beartype
from jaxtyping import Array, Float, jaxtyped

from arpyes.types import ScalarFloat

_ENERGIES: Float[Array, " 3"] = jnp.array(
    [20.0, 40.0, 60.0], dtype=jnp.float64
)
_SIGMA_S: Float[Array, " 3"] = jnp.array(
    [0.1, 0.08, 0.06], dtype=jnp.float64
)
_SIGMA_P: Float[Array, " 3"] = jnp.array(
    [0.6, 0.9, 1.1], dtype=jnp.float64
)
_SIGMA_D: Float[Array, " 3"] = jnp.array(
    [2.0, 1.5, 1.2], dtype=jnp.float64
)


@jaxtyped(typechecker=beartype)
def heuristic_weights(
    photon_energy: ScalarFloat,
) -> Float[Array, " 9"]:
    """Compute heuristic orbital weights based on photon energy.

    Parameters
    ----------
    photon_energy : ScalarFloat
        Incident photon energy in eV.

    Returns
    -------
    weights : Float[Array, " 9"]
        Orbital weights for [s, py, pz, px, dxy, dyz, dz2,
        dxz, dx2-y2].

    Notes
    -----
    Below 50 eV, p-orbitals are enhanced (weight 2).
    Above 50 eV, d-orbitals are enhanced (weight 2).
    """
    low_e: Float[Array, " 9"] = jnp.array(
        [1.0, 2.0, 2.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        dtype=jnp.float64,
    )
    high_e: Float[Array, " 9"] = jnp.array(
        [1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 2.0],
        dtype=jnp.float64,
    )
    _threshold_ev: float = 50.0
    weights: Float[Array, " 9"] = jnp.where(
        photon_energy < _threshold_ev, low_e, high_e
    )
    return weights


def _interp_cross_section(
    photon_energy: Float[Array, " "],
    sigma_vals: Float[Array, " 3"],
) -> Float[Array, " "]:
    """Linearly interpolate cross-section with extrapolation.

    Parameters
    ----------
    photon_energy : Float[Array, " "]
        Photon energy in eV.
    sigma_vals : Float[Array, " 3"]
        Cross-section values at [20, 40, 60] eV.

    Returns
    -------
    sigma : Float[Array, " "]
        Interpolated cross-section value.
    """
    sigma: Float[Array, " "] = jnp.interp(
        photon_energy, _ENERGIES, sigma_vals
    )
    return sigma


@jaxtyped(typechecker=beartype)
def yeh_lindau_weights(
    photon_energy: ScalarFloat,
) -> Float[Array, " 9"]:
    """Compute Yeh-Lindau cross-section weights per orbital.

    Parameters
    ----------
    photon_energy : ScalarFloat
        Incident photon energy in eV.

    Returns
    -------
    weights : Float[Array, " 9"]
        Cross-section weights for [s, py, pz, px, dxy, dyz,
        dz2, dxz, dx2-y2].

    Notes
    -----
    Uses linear interpolation of tabulated Yeh-Lindau values
    at 20, 40, and 60 eV with extrapolation outside this range.
    """
    pe: Float[Array, " "] = jnp.asarray(
        photon_energy, dtype=jnp.float64
    )
    s_w: Float[Array, " "] = _interp_cross_section(
        pe, _SIGMA_S
    )
    p_w: Float[Array, " "] = _interp_cross_section(
        pe, _SIGMA_P
    )
    d_w: Float[Array, " "] = _interp_cross_section(
        pe, _SIGMA_D
    )
    weights: Float[Array, " 9"] = jnp.array(
        [s_w, p_w, p_w, p_w, d_w, d_w, d_w, d_w, d_w],
        dtype=jnp.float64,
    )
    return weights


__all__: list[str] = [
    "heuristic_weights",
    "yeh_lindau_weights",
]
