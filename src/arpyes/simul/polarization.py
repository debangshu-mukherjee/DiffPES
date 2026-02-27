"""Photon polarization and dipole matrix element calculations.

Extended Summary
----------------
Computes electric field polarization vectors from incident photon
geometry and evaluates dipole transition matrix elements for each
atomic orbital, following standard ARPES selection rules.

Routine Listings
----------------
:func:`build_efield`
    Compute electric field vector from polarization config.
:func:`build_polarization_vectors`
    Construct s- and p-polarization basis vectors.
:func:`dipole_matrix_elements`
    Compute |e dot d_orbital|^2 for all 9 orbitals.
:func:`photon_wavevector`
    Unit photon propagation vector from incidence angles.

Notes
-----
Orbital direction vectors follow VASP orbital ordering:
[s, py, pz, px, dxy, dyz, dz2, dxz, dx2-y2].
The s-orbital has zero directionality.
"""

import jax
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

    Builds an orthonormal pair of polarization vectors (e_s, e_p)
    from the photon incidence angles, defining the s-polarization
    (perpendicular to the incidence plane) and p-polarization (in
    the incidence plane, perpendicular to the wavevector).

    Implementation Logic
    --------------------
    1. **Construct photon wavevector k from spherical coordinates**:
       k = [sin(theta)*cos(phi), sin(theta)*sin(phi), cos(theta)]
       k = k / ||k||
       - Converts the incidence angles (theta from surface normal,
         phi azimuthal) into a unit wavevector in Cartesian
         coordinates.

    2. **Choose reference axis**:
       ref = z_hat  unless  |k . z_hat| >= 0.99
       ref = y_hat  if k is nearly collinear with z_hat
       - The reference axis is used to define the incidence plane.
         When k is nearly parallel to z_hat, the cross product
         k x z_hat would be poorly conditioned, so y_hat is used
         as a fallback.

    3. **Compute s-polarization: e_s = normalize(k x ref)**:
       e_s_raw = cross(k, ref)
       e_s = e_s_raw / ||e_s_raw||
       - The s-polarization vector is perpendicular to both the
         wavevector and the reference axis, hence perpendicular to
         the incidence plane.

    4. **Compute p-polarization: e_p = normalize(e_s x k)**:
       e_p_raw = cross(e_s, k)
       e_p = e_p_raw / ||e_p_raw||
       - The p-polarization vector lies in the incidence plane and
         is perpendicular to the wavevector, completing the
         right-handed orthonormal basis {k, e_s, e_p}.

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
def photon_wavevector(
    theta: ScalarFloat,
    phi: ScalarFloat,
) -> Float[Array, " 3"]:
    """Unit photon wavevector from incidence angles.

    Builds the unit vector in the direction of photon propagation
    from spherical coordinates (theta from surface normal, phi
    azimuthal): k = [sin(theta)*cos(phi), sin(theta)*sin(phi),
    cos(theta)] / ||...||. Used in spin-orbit ARPES simulations
    to form the S·k_photon correction for circular dichroism.

    Parameters
    ----------
    theta : ScalarFloat
        Incident angle from surface normal in radians.
    phi : ScalarFloat
        In-plane azimuthal angle in radians.

    Returns
    -------
    k_photon : Float[Array, " 3"]
        Unit wavevector in Cartesian coordinates.

    See Also
    --------
    build_polarization_vectors : Builds the same k internally for
        the s- and p-polarization basis; use this when only the
        propagation direction is needed.
    """
    k: Float[Array, " 3"] = jnp.array(
        [
            jnp.sin(theta) * jnp.cos(phi),
            jnp.sin(theta) * jnp.sin(phi),
            jnp.cos(theta),
        ],
        dtype=jnp.float64,
    )
    return k / jnp.linalg.norm(k)


@jaxtyped(typechecker=beartype)
def build_efield(
    config: PolarizationConfig,
) -> Complex[Array, " 3"]:
    """Compute electric field vector from polarization config.

    Constructs the complex electric field polarization vector for the
    specified photon geometry and polarization type.

    Implementation Logic
    --------------------
    1. **Build s- and p-polarization basis**:
       e_s, e_p = build_polarization_vectors(theta, phi)
       - Computes the real-valued orthonormal basis vectors from the
         incidence angles in the config. Both are cast to complex128
         for compatibility with circular polarization states.

    2. **Dispatch on polarization type**:
       The ``polarization_type`` string (case-insensitive) selects
       the electric field vector:

       - **"lvp"** (linear vertical polarization):
         efield = e_s
         Pure s-polarization.

       - **"lhp"** (linear horizontal polarization):
         efield = e_p
         Pure p-polarization.

       - **"lap"** (linear arbitrary polarization):
         efield = cos(angle) * e_s + sin(angle) * e_p
         Linear combination at the angle specified by
         ``config.polarization_angle``.

       - **"rcp"** (right circular polarization):
         efield = (e_s + i * e_p) / sqrt(2)
         Right-handed circular polarization with equal s and p
         amplitudes and 90-degree phase shift.

       - **"lcp"** (left circular polarization):
         efield = (e_s - i * e_p) / sqrt(2)
         Left-handed circular polarization with equal s and p
         amplitudes and -90-degree phase shift.

       - **else** (fallback / unpolarized):
         efield = e_s
         Defaults to s-polarization. Unpolarized averaging is
         handled externally in the simulation loop.

    3. **JAX switch over branches**:
       An operand tuple (e_s_c, e_p_c, polarization_angle) is built so
       that all branch functions receive the same traced inputs. The
       polarization type is mapped to an integer index (lvp=0, lhp=1,
       lap=2, rcp=3, lcp=4, default=5). ``jax.lax.switch(index, branches,
       operand)`` invokes the corresponding branch function on the
       operand and returns the resulting electric field vector.

    Parameters
    ----------
    config : PolarizationConfig
        Polarization geometry specification.

    Returns
    -------
    efield : Complex[Array, " 3"]
        Complex electric field polarization vector.
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
    angle: Float[Array, " "] = jnp.asarray(
        config.polarization_angle, dtype=jnp.float64
    )
    operand: Tuple[
        Complex[Array, " 3"],
        Complex[Array, " 3"],
        Float[Array, " "],
    ] = (e_s_c, e_p_c, angle)

    def branch_lvp(op: Tuple) -> Complex[Array, " 3"]:
        """Linear vertical polarization: electric field equals s-polarization basis vector.

        Parameters
        ----------
        op : Tuple
            Operand (e_s_c, e_p_c, angle). Only the first element is used.

        Returns
        -------
        Complex[Array, " 3"]
            The s-polarization complex electric field vector.
        """
        return op[0]

    def branch_lhp(op: Tuple) -> Complex[Array, " 3"]:
        """Linear horizontal polarization: electric field equals p-polarization basis vector.

        Parameters
        ----------
        op : Tuple
            Operand (e_s_c, e_p_c, angle). Only the second element is used.

        Returns
        -------
        Complex[Array, " 3"]
            The p-polarization complex electric field vector.
        """
        return op[1]

    def branch_lap(op: Tuple) -> Complex[Array, " 3"]:
        """Linear arbitrary polarization: cos(angle)*e_s + sin(angle)*e_p.

        Parameters
        ----------
        op : Tuple
            Operand (e_s_c, e_p_c, angle). op[2] is the polarization angle in radians.

        Returns
        -------
        Complex[Array, " 3"]
            The linear combination of s- and p-polarization vectors.
        """
        return (
            jnp.cos(op[2]) * op[0]
            + jnp.sin(op[2]) * op[1]
        )

    def branch_rcp(op: Tuple) -> Complex[Array, " 3"]:
        """Right circular polarization: (e_s + i*e_p)/sqrt(2).

        Parameters
        ----------
        op : Tuple
            Operand (e_s_c, e_p_c, angle). First two elements are used.

        Returns
        -------
        Complex[Array, " 3"]
            Right-handed circular polarization electric field.
        """
        return (op[0] + 1j * op[1]) / jnp.sqrt(2.0)

    def branch_lcp(op: Tuple) -> Complex[Array, " 3"]:
        """Left circular polarization: (e_s - i*e_p)/sqrt(2).

        Parameters
        ----------
        op : Tuple
            Operand (e_s_c, e_p_c, angle). First two elements are used.

        Returns
        -------
        Complex[Array, " 3"]
            Left-handed circular polarization electric field.
        """
        return (op[0] - 1j * op[1]) / jnp.sqrt(2.0)

    def branch_default(op: Tuple) -> Complex[Array, " 3"]:
        """Fallback for unknown or unpolarized type: return s-polarization.

        Parameters
        ----------
        op : Tuple
            Operand (e_s_c, e_p_c, angle). Only the first element is used.

        Returns
        -------
        Complex[Array, " 3"]
            The s-polarization vector as the default.
        """
        return op[0]

    _POL_INDEX: dict[str, int] = {
        "lvp": 0,
        "lhp": 1,
        "lap": 2,
        "rcp": 3,
        "lcp": 4,
    }
    index: int = _POL_INDEX.get(pol_type, 5)
    branches: Tuple = (
        branch_lvp,
        branch_lhp,
        branch_lap,
        branch_rcp,
        branch_lcp,
        branch_default,
    )
    efield: Complex[Array, " 3"] = jax.lax.switch(
        index, branches, operand
    )
    return efield


@jaxtyped(typechecker=beartype)
def dipole_matrix_elements(
    efield: Complex[Array, " 3"],
) -> Float[Array, " 9"]:
    """Compute dipole matrix elements for all 9 orbitals.

    Evaluates the squared modulus of the dipole transition matrix
    element for each orbital:

        M_i = |e . d_i|^2

    where e is the electric field polarization vector and d_i is
    the normalized direction vector of orbital i.

    Implementation Logic
    --------------------
    1. **Dot product of E-field with each orbital direction**:
       dots = ORBITAL_DIRS_NORMALIZED @ efield
       - Computes the inner product of the complex electric field
         vector with each of the 9 normalized orbital direction
         vectors (shape [9, 3] @ [3] -> [9]). The result is a
         complex-valued array of length 9.
       - The s-orbital direction vector is [0, 0, 0] (zero vector),
         so its dot product is always zero regardless of the
         E-field, reflecting the isotropic (zero directionality)
         character of the s-orbital.

    2. **Square modulus |e . d|^2**:
       matrix_elements = |dots|^2
       - Takes the absolute value squared of each complex dot
         product. For real-valued E-fields this reduces to the
         squared real dot product. For circular polarization
         (complex E-field) this correctly accounts for the phase.

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
    The s-orbital has a zero direction vector and therefore always
    produces a zero dipole matrix element with any polarization.
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
    "photon_wavevector",
]
