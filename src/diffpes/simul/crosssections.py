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

from diffpes.types import ScalarFloat

_ENERGIES: Float[Array, " 3"] = jnp.array(
    [20.0, 40.0, 60.0], dtype=jnp.float64
)
_SIGMA_S: Float[Array, " 3"] = jnp.array([0.1, 0.08, 0.06], dtype=jnp.float64)
_SIGMA_P: Float[Array, " 3"] = jnp.array([0.6, 0.9, 1.1], dtype=jnp.float64)
_SIGMA_D: Float[Array, " 3"] = jnp.array([2.0, 1.5, 1.2], dtype=jnp.float64)


@jaxtyped(typechecker=beartype)
def heuristic_weights(
    photon_energy: ScalarFloat,
) -> Float[Array, " 9"]:
    """Compute heuristic orbital weights based on photon energy.

    Provides a simple two-regime model for orbital-dependent
    photoionization cross-sections. This is a coarse approximation
    useful when tabulated cross-section data is unavailable.

    Implementation Logic
    --------------------
    The function selects between two pre-defined weight vectors
    based on a 50 eV energy threshold:

    1. **Below 50 eV (low-energy regime)**:
       weights = [1, 2, 2, 2, 1, 1, 1, 1, 1]
       - p-orbitals (indices 1-3) are enhanced with weight 2,
         reflecting the stronger p-orbital cross-section at low
         photon energies typical of He-I or laser ARPES.

    2. **Above 50 eV (high-energy regime)**:
       weights = [1, 1, 1, 1, 2, 2, 2, 2, 2]
       - d-orbitals (indices 4-8) are enhanced with weight 2,
         reflecting the resonant enhancement of d-orbital
         cross-sections at higher photon energies (e.g., He-II,
         synchrotron).

    The selection is performed via ``jnp.where`` for JIT
    compatibility (no Python-level branching).

    Parameters
    ----------
    photon_energy : ScalarFloat
        Incident photon energy in eV.

    Returns
    -------
    weights : Float[Array, " 9"]
        Orbital weights for [s, py, pz, px, dxy, dyz, dz2,
        dxz, dx2-y2].
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

    Performs piecewise-linear interpolation of tabulated cross-section
    values at the given photon energy, with constant extrapolation
    outside the tabulated range.

    Implementation Logic
    --------------------
    1. **Interpolate via jnp.interp**:
       sigma = jnp.interp(photon_energy, _ENERGIES, sigma_vals)
       - ``_ENERGIES`` = [20, 40, 60] eV are the tabulation points.
       - ``sigma_vals`` are the corresponding cross-section values.
       - ``jnp.interp`` performs piecewise-linear interpolation
         between adjacent tabulation points.
       - At the boundaries: energies below 20 eV return
         sigma_vals[0] and energies above 60 eV return
         sigma_vals[-1] (constant extrapolation), which is the
         default behavior of ``jnp.interp``.

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
    sigma: Float[Array, " "] = jnp.interp(photon_energy, _ENERGIES, sigma_vals)
    return sigma


@jaxtyped(typechecker=beartype)
def yeh_lindau_weights(
    photon_energy: ScalarFloat,
) -> Float[Array, " 9"]:
    """Compute Yeh-Lindau cross-section weights per orbital.

    Interpolates tabulated photoionization cross-sections from
    Yeh & Lindau (1985) to produce orbital-resolved weights at
    the specified photon energy.

    Implementation Logic
    --------------------
    1. **Cast photon energy to float64**:
       pe = jnp.asarray(photon_energy, dtype=float64)
       - Ensures consistent precision for the interpolation.

    2. **Interpolate s, p, d cross-sections independently**:
       s_w = _interp_cross_section(pe, _SIGMA_S)
       p_w = _interp_cross_section(pe, _SIGMA_P)
       d_w = _interp_cross_section(pe, _SIGMA_D)
       - Each call linearly interpolates the corresponding
         tabulated values at 20, 40, and 60 eV. The tabulated
         data (_SIGMA_S, _SIGMA_P, _SIGMA_D) encodes simplified
         Yeh-Lindau cross-sections for s, p, and d subshells.

    3. **Broadcast to 9-orbital weight vector**:
       weights = [s_w, p_w, p_w, p_w, d_w, d_w, d_w, d_w, d_w]
       - Maps the three subshell cross-sections onto the full
         9-orbital basis: 1 s-orbital, 3 p-orbitals (each
         receiving p_w), and 5 d-orbitals (each receiving d_w).

    Parameters
    ----------
    photon_energy : ScalarFloat
        Incident photon energy in eV.

    Returns
    -------
    weights : Float[Array, " 9"]
        Cross-section weights for [s, py, pz, px, dxy, dyz,
        dz2, dxz, dx2-y2].

    References
    ----------
    .. [1] Yeh & Lindau, "Atomic subshell photoionization cross
       sections and asymmetry parameters: 1 <= Z <= 103", Atomic
       Data and Nuclear Data Tables 32, 1-155 (1985).
    """
    pe: Float[Array, " "] = jnp.asarray(photon_energy, dtype=jnp.float64)
    s_w: Float[Array, " "] = _interp_cross_section(pe, _SIGMA_S)
    p_w: Float[Array, " "] = _interp_cross_section(pe, _SIGMA_P)
    d_w: Float[Array, " "] = _interp_cross_section(pe, _SIGMA_D)
    weights: Float[Array, " 9"] = jnp.array(
        [s_w, p_w, p_w, p_w, d_w, d_w, d_w, d_w, d_w],
        dtype=jnp.float64,
    )
    return weights


__all__: list[str] = [
    "heuristic_weights",
    "yeh_lindau_weights",
]
