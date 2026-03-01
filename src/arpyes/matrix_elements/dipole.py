r"""Full dipole matrix element assembly.

Extended Summary
----------------
Combines radial integrals :math:`B^{l'}(k)`, Gaunt coefficients,
real spherical harmonics :math:`Y_{l'm'}(\hat{k})`, and the
polarization vector :math:`\hat{e}` to compute photoemission
dipole matrix elements from first principles:

.. math::

    M(\mathbf{k}, n, l, m) = \sum_{l', m'} B_{n,l}^{l'}(|\mathbf{k}|)
        \cdot G(l, m, l', m') \cdot Y_{l'}^{m'}(\hat{k})
        \cdot \hat{e}_{q(m'-m)}

where :math:`q = m' - m` selects the dipole component and
:math:`\hat{e}_q` is the corresponding spherical component of
the polarization vector.
"""

import jax.numpy as jnp
from beartype import beartype
from jaxtyping import Array, Complex, Float, jaxtyped

from arpyes.radial import radial_integral, slater_radial
from arpyes.types import SlaterParams

from .gaunt import GAUNT_TABLE, L_MAX
from .spherical_harmonics import real_spherical_harmonic


def _cartesian_to_spherical_dipole(
    efield: Complex[Array, " 3"],
) -> Complex[Array, " 3"]:
    """Convert Cartesian E-field to real-harmonic dipole components.

    Maps (e_x, e_y, e_z) to (e_{q=-1}, e_{q=0}, e_{q=+1}) where
    the q index matches the real spherical harmonic convention for
    the dipole operator (l=1):

    - q = -1 corresponds to Y_1^{-1}(real) ~ sin(theta)sin(phi) ~ y
    - q =  0 corresponds to Y_1^0(real) ~ cos(theta) ~ z
    - q = +1 corresponds to Y_1^{+1}(real) ~ sin(theta)cos(phi) ~ x

    Returns
    -------
    e_spherical : Complex[Array, " 3"]
        ``(e_{q=-1}, e_{q=0}, e_{q=+1})``.
    """
    ex = efield[0]
    ey = efield[1]
    ez = efield[2]
    # q=+1 ~ x, q=-1 ~ y, q=0 ~ z (real spherical harmonic convention)
    return jnp.array([ey, ez, ex], dtype=jnp.complex128)


@jaxtyped(typechecker=beartype)
def dipole_matrix_element_single(
    k_vec: Float[Array, " 3"],
    r_grid: Float[Array, " R"],
    radial_values: Float[Array, " R"],
    l: int,
    m: int,
    efield: Complex[Array, " 3"],
) -> Complex[Array, " "]:
    r"""Compute dipole matrix element for a single orbital (n, l, m).

    .. math::

        M = \sum_{q} \hat{e}_q \sum_{l'} B^{l'}(|k|) \cdot
            G(l, m, l', m+q) \cdot Y_{l'}^{m+q}(\hat{k})

    where the sum is over dipole components q in {-1, 0, +1} and
    final-state angular momenta l' in {l-1, l+1}.

    Parameters
    ----------
    k_vec : Float[Array, " 3"]
        Photoelectron wavevector in Cartesian coordinates.
    r_grid : Float[Array, " R"]
        Radial grid for integration.
    radial_values : Float[Array, " R"]
        R(r) sampled on r_grid.
    l : int
        Angular momentum quantum number of the initial orbital.
    m : int
        Magnetic quantum number of the initial orbital.
    efield : Complex[Array, " 3"]
        Polarization vector in Cartesian coordinates.

    Returns
    -------
    M : Complex[Array, " "]
        Complex dipole matrix element.
    """
    k_mag = jnp.linalg.norm(k_vec)
    safe_k = jnp.where(k_mag > 1e-12, k_mag, 1e-12)
    k_hat = k_vec / safe_k

    # Convert k_hat to spherical angles
    theta_k = jnp.arccos(jnp.clip(k_hat[2], -1.0, 1.0))
    phi_k = jnp.arctan2(k_hat[1], k_hat[0])

    # Polarization in spherical dipole components
    e_sph = _cartesian_to_spherical_dipole(efield)

    M_total = jnp.zeros((), dtype=jnp.complex128)

    for q_idx, q in enumerate((-1, 0, 1)):
        mp = m + q  # final state m'
        eq = e_sph[q_idx]

        for lp in (l - 1, l + 1):
            if lp < 0 or lp > L_MAX + 1:
                continue
            if abs(mp) > lp:
                continue

            # Radial integral B^{l'}(|k|)
            B_lp = radial_integral(k_mag, r_grid, radial_values, lp)

            # Gaunt coefficient G(l, m, l', m')
            G = GAUNT_TABLE[l, m + L_MAX, q + 1, lp, mp + L_MAX]

            # Spherical harmonic Y_{l'}^{m'}(k_hat)
            Y_lp_mp = real_spherical_harmonic(lp, mp, theta_k, phi_k)

            M_total = M_total + eq * B_lp * G * Y_lp_mp

    return M_total


@jaxtyped(typechecker=beartype)
def dipole_intensity_orbital(
    k_vec: Float[Array, " 3"],
    r_grid: Float[Array, " R"],
    radial_values: Float[Array, " R"],
    l: int,
    m: int,
    efield: Complex[Array, " 3"],
) -> Float[Array, " "]:
    """Compute |M|^2 for one orbital.

    Parameters
    ----------
    k_vec : Float[Array, " 3"]
        Photoelectron wavevector.
    r_grid : Float[Array, " R"]
        Radial grid.
    radial_values : Float[Array, " R"]
        Radial wavefunction on grid.
    l : int
        Angular momentum.
    m : int
        Magnetic quantum number.
    efield : Complex[Array, " 3"]
        Polarization vector.

    Returns
    -------
    intensity : Float[Array, " "]
        Squared modulus of the matrix element.
    """
    M = dipole_matrix_element_single(k_vec, r_grid, radial_values, l, m, efield)
    return jnp.abs(M) ** 2


@jaxtyped(typechecker=beartype)
def dipole_intensities_all_orbitals(
    k_vec: Float[Array, " 3"],
    r_grid: Float[Array, " R"],
    slater_params: SlaterParams,
    efield: Complex[Array, " 3"],
) -> Float[Array, " O"]:
    r"""Compute |M|^2 for all orbitals in the basis.

    Unrolls the orbital loop at Python level because each orbital
    has different static (l, m) controlling recurrence depth.

    Parameters
    ----------
    k_vec : Float[Array, " 3"]
        Photoelectron wavevector.
    r_grid : Float[Array, " R"]
        Radial grid.
    slater_params : SlaterParams
        Slater exponents and orbital basis.
    efield : Complex[Array, " 3"]
        Polarization vector.

    Returns
    -------
    intensities : Float[Array, " O"]
        |M|^2 per orbital.
    """
    basis = slater_params.orbital_basis
    n_orbitals = len(basis.n_values)
    results = []

    for o in range(n_orbitals):
        n_o = basis.n_values[o]
        l_o = basis.l_values[o]
        m_o = basis.m_values[o]
        zeta_o = slater_params.zeta[o]

        # Compute radial wavefunction on the grid
        R_values = slater_radial(r_grid, n_o, zeta_o)

        # Weight by multi-zeta coefficient (first column for single-zeta)
        R_values = R_values * slater_params.coefficients[o, 0]

        intensity = dipole_intensity_orbital(
            k_vec, r_grid, R_values, l_o, m_o, efield
        )
        results.append(intensity)

    return jnp.stack(results)


__all__: list[str] = [
    "dipole_intensities_all_orbitals",
    "dipole_intensity_orbital",
    "dipole_matrix_element_single",
]
