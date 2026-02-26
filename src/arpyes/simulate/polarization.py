"""Photon polarization and dipole matrix element calculations.

Extended Summary
----------------
Computes electric field polarization vectors from incident photon
geometry and evaluates dipole transition matrix elements for each
atomic orbital, following standard ARPES selection rules.

Routine Listings
----------------
:func:`build_polarization_vectors`
    Construct s- and p-polarization basis vectors.
:func:`build_efield`
    Compute electric field vector from polarization config.
:func:`dipole_matrix_elements`
    Compute |e dot d_orbital|^2 for all 9 orbitals.

Notes
-----
Orbital direction vectors follow VASP orbital ordering:
[s, py, pz, px, dxy, dyz, dz2, dxz, dx2-y2].
The s-orbital has zero directionality.
"""

import jax.numpy as jnp
from beartype import beartype
from beartype.typing import Tuple
from jaxtyping import Array, Complex, Float, jaxtyped

from arpyes.types import PolarizationConfig, ScalarFloat

_ORBITAL_DIRS: Float[Array, "9 3"] = jnp.array(
    [
        [0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 1.0],
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 1.0],
        [1.0, -1.0, 0.0],
    ],
    dtype=jnp.float64,
)

_ORBITAL_NORMS: Float[Array, " 9"] = jnp.where(
    jnp.linalg.norm(_ORBITAL_DIRS, axis=1) > 0.0,
    jnp.linalg.norm(_ORBITAL_DIRS, axis=1),
    1.0,
)

ORBITAL_DIRS_NORMALIZED: Float[Array, "9 3"] = (
    _ORBITAL_DIRS / _ORBITAL_NORMS[:, jnp.newaxis]
)


@jaxtyped(typechecker=beartype)
def build_polarization_vectors(
    theta: ScalarFloat,
    phi: ScalarFloat,
) -> Tuple[Float[Array, " 3"], Float[Array, " 3"]]:
    """Construct s- and p-polarization basis vectors.

    Parameters
    ----------
    theta : ScalarFloat
        Incident angle from surface normal in radians.
    phi : ScalarFloat
        In-plane azimuthal angle in radians.

    Returns
    -------
    e_s : Float[Array, " 3"]
        s-polarization unit vector (perpendicular to
        incidence plane).
    e_p : Float[Array, " 3"]
        p-polarization unit vector (in incidence plane,
        perpendicular to photon wavevector).
    """
    k_photon: Float[Array, " 3"] = jnp.array(
        [
            jnp.sin(theta) * jnp.cos(phi),
            jnp.sin(theta) * jnp.sin(phi),
            jnp.cos(theta),
        ],
        dtype=jnp.float64,
    )
    k_photon = k_photon / jnp.linalg.norm(k_photon)
    z_hat: Float[Array, " 3"] = jnp.array(
        [0.0, 0.0, 1.0], dtype=jnp.float64
    )
    y_hat: Float[Array, " 3"] = jnp.array(
        [0.0, 1.0, 0.0], dtype=jnp.float64
    )
    _collinear_threshold: float = 0.99
    ref: Float[Array, " 3"] = jnp.where(
        jnp.abs(jnp.dot(k_photon, z_hat))
        < _collinear_threshold,
        z_hat,
        y_hat,
    )
    e_s_raw: Float[Array, " 3"] = jnp.cross(
        k_photon, ref
    )
    e_s: Float[Array, " 3"] = e_s_raw / jnp.linalg.norm(
        e_s_raw
    )
    e_p_raw: Float[Array, " 3"] = jnp.cross(e_s, k_photon)
    e_p: Float[Array, " 3"] = e_p_raw / jnp.linalg.norm(
        e_p_raw
    )
    return e_s, e_p


@jaxtyped(typechecker=beartype)
def build_efield(
    config: PolarizationConfig,
) -> Complex[Array, " 3"]:
    """Compute electric field vector from polarization config.

    Parameters
    ----------
    config : PolarizationConfig
        Polarization geometry specification.

    Returns
    -------
    efield : Complex[Array, " 3"]
        Complex electric field polarization vector.

    Notes
    -----
    For unpolarized light, returns the s-polarization vector.
    Unpolarized averaging is handled in the simulation loop.
    """
    e_s: Float[Array, " 3"]
    e_p: Float[Array, " 3"]
    e_s, e_p = build_polarization_vectors(
        config.theta, config.phi
    )
    e_s_c: Complex[Array, " 3"] = e_s.astype(
        jnp.complex128
    )
    e_p_c: Complex[Array, " 3"] = e_p.astype(
        jnp.complex128
    )
    pol_type: str = config.polarization_type.lower()
    if pol_type == "lvp":
        efield: Complex[Array, " 3"] = e_s_c
    elif pol_type == "lhp":
        efield = e_p_c
    elif pol_type == "lap":
        efield = (
            jnp.cos(config.polarization_angle) * e_s_c
            + jnp.sin(config.polarization_angle) * e_p_c
        )
    elif pol_type == "rcp":
        efield = (e_s_c + 1j * e_p_c) / jnp.sqrt(2.0)
    elif pol_type == "lcp":
        efield = (e_s_c - 1j * e_p_c) / jnp.sqrt(2.0)
    else:
        efield = e_s_c
    return efield


@jaxtyped(typechecker=beartype)
def dipole_matrix_elements(
    efield: Complex[Array, " 3"],
) -> Float[Array, " 9"]:
    """Compute dipole matrix elements for all 9 orbitals.

    Parameters
    ----------
    efield : Complex[Array, " 3"]
        Electric field polarization vector.

    Returns
    -------
    matrix_elements : Float[Array, " 9"]
        |e dot d_orbital|^2 for each orbital.

    Notes
    -----
    The s-orbital has zero directionality and thus zero
    dipole matrix element with any polarization.
    """
    dots: Complex[Array, " 9"] = jnp.dot(
        ORBITAL_DIRS_NORMALIZED,
        efield,
    )
    matrix_elements: Float[Array, " 9"] = jnp.abs(dots) ** 2
    return matrix_elements


__all__: list[str] = [
    "ORBITAL_DIRS_NORMALIZED",
    "build_efield",
    "build_polarization_vectors",
    "dipole_matrix_elements",
]
