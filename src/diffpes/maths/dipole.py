r"""Convert dipole channels and compute Cartesian dipole gauges.

Extended Summary
----------------
The public polarization convention is always Cartesian sample-frame order
``(x, y, z)``.  Complex spherical photon components use order
``(-1, 0, +1)`` and the Condon--Shortley convention

.. math::

   \epsilon_{-1}=(\epsilon_x-i\epsilon_y)/\sqrt2,\qquad
   \epsilon_0=\epsilon_z,\qquad
   \epsilon_{+1}=-(\epsilon_x+i\epsilon_y)/\sqrt2.

The real :math:`l=1` harmonic order is ``(y, z, x)``.  Explicit forward and
inverse maps keep these two three-component bases distinct.  The static
channel table retains every final real harmonic, so a real-orbital label is
never subjected to the complex-basis shortcut :math:`m'=m+q`.

The Cartesian length- and momentum-gauge contractions operate on sampled
wavefunctions.  Both use the final-state bra conjugate and preserve generic
complex phases.  They are independent contractions; neither invokes the
local-potential commutator identity.

Routine Listings
----------------
:func:`channel_tables`
    Build padded real-harmonic dipole channel tables.
:func:`dipole_length_cartesian`
    Compute a sampled Cartesian length-gauge contraction.
:func:`dipole_momentum_cartesian`
    Compute a sampled Cartesian momentum-gauge contraction.
:func:`polarization_cart_to_complex`
    Convert Cartesian polarization to complex spherical components.
:func:`polarization_cart_to_real`
    Convert Cartesian polarization to real-harmonic channel order.
:func:`polarization_complex_to_cart`
    Convert complex spherical polarization to Cartesian components.
:func:`polarization_real_to_cart`
    Convert real-harmonic polarization back to Cartesian order.
"""

import math

import jax.numpy as jnp
import numpy as np
from beartype import beartype
from beartype.typing import Tuple
from jaxtyping import Array, Complex128, Float64, jaxtyped
from numpy.typing import NDArray

from diffpes.types import L_MAX, OrbitalBasis

from .gaunt import GAUNT_TABLE


@jaxtyped(typechecker=beartype)
def polarization_cart_to_complex(
    efield_cart: Complex128[Array, " 3"],
) -> Complex128[Array, " 3"]:
    r"""Convert Cartesian polarization to complex spherical components.

    The input order is ``(x, y, z)`` and the returned order is
    :math:`(-1,0,+1)`.  The transformation is unitary and applies equally to
    real linear, circular, and generic complex elliptic polarization.

    :see: :class:`~.test_dipole.TestPolarizationCartToComplex`

    Parameters
    ----------
    efield_cart : Complex128[Array, " 3"]
        Cartesian sample-frame polarization in ``(x, y, z)`` order.

    Returns
    -------
    efield_complex : Complex128[Array, " 3"]
        Complex spherical components in ``(-1, 0, +1)`` order.

    Notes
    -----
    The explicit three-row stack leaves generic complex phases intact.
    """
    inverse_sqrt_two: float = 1.0 / math.sqrt(2.0)
    ex: Complex128[Array, ""] = efield_cart[0]
    ey: Complex128[Array, ""] = efield_cart[1]
    ez: Complex128[Array, ""] = efield_cart[2]
    efield_complex: Complex128[Array, " 3"] = jnp.stack(
        (
            inverse_sqrt_two * (ex - 1j * ey),
            ez,
            -inverse_sqrt_two * (ex + 1j * ey),
        )
    )
    return efield_complex


@jaxtyped(typechecker=beartype)
def polarization_complex_to_cart(
    efield_complex: Complex128[Array, " 3"],
) -> Complex128[Array, " 3"]:
    r"""Convert complex spherical polarization to Cartesian components.

    This function is the exact inverse of
    :func:`polarization_cart_to_complex`.

    :see: :class:`~.test_dipole.TestPolarizationComplexToCart`

    Parameters
    ----------
    efield_complex : Complex128[Array, " 3"]
        Complex spherical components in ``(-1, 0, +1)`` order.

    Returns
    -------
    efield_cart : Complex128[Array, " 3"]
        Cartesian sample-frame polarization in ``(x, y, z)`` order.

    Notes
    -----
    The inverse combines both transverse spherical components without
    conjugating them.
    """
    inverse_sqrt_two: float = 1.0 / math.sqrt(2.0)
    e_minus: Complex128[Array, ""] = efield_complex[0]
    e_zero: Complex128[Array, ""] = efield_complex[1]
    e_plus: Complex128[Array, ""] = efield_complex[2]
    efield_cart: Complex128[Array, " 3"] = jnp.stack(
        (
            inverse_sqrt_two * (e_minus - e_plus),
            1j * inverse_sqrt_two * (e_minus + e_plus),
            e_zero,
        )
    )
    return efield_cart


@jaxtyped(typechecker=beartype)
def polarization_cart_to_real(
    efield_cart: Complex128[Array, " 3"],
) -> Complex128[Array, " 3"]:
    r"""Convert Cartesian polarization to real-harmonic channel order.

    Real :math:`l=1` harmonic rows follow ``(p_y, p_z, p_x)``, corresponding
    to magnetic labels ``(-1, 0, +1)``.

    :see: :class:`~.test_dipole.TestPolarizationCartToReal`

    Parameters
    ----------
    efield_cart : Complex128[Array, " 3"]
        Cartesian sample-frame polarization in ``(x, y, z)`` order.

    Returns
    -------
    efield_real : Complex128[Array, " 3"]
        Real-harmonic components in ``(y, z, x)`` order.

    Notes
    -----
    Static integer indexing applies the real-harmonic permutation.
    """
    efield_real: Complex128[Array, " 3"] = efield_cart[jnp.asarray((1, 2, 0))]
    return efield_real


@jaxtyped(typechecker=beartype)
def polarization_real_to_cart(
    efield_real: Complex128[Array, " 3"],
) -> Complex128[Array, " 3"]:
    r"""Convert real-harmonic polarization back to Cartesian order.

    This function is the exact inverse of
    :func:`polarization_cart_to_real`.

    :see: :class:`~.test_dipole.TestPolarizationRealToCart`

    Parameters
    ----------
    efield_real : Complex128[Array, " 3"]
        Real-harmonic components in ``(y, z, x)`` order.

    Returns
    -------
    efield_cart : Complex128[Array, " 3"]
        Cartesian sample-frame polarization in ``(x, y, z)`` order.

    Notes
    -----
    Static integer indexing applies the inverse permutation.
    """
    efield_cart: Complex128[Array, " 3"] = efield_real[jnp.asarray((2, 0, 1))]
    return efield_cart


@jaxtyped(typechecker=beartype)
def channel_tables(
    basis: OrbitalBasis,
) -> Tuple[
    Float64[Array, "n_orb 2 3 n_y"],
    Float64[Array, "n_orb 2 3 n_y"],
]:
    r"""Build padded real-harmonic dipole channel tables.

    Each orbital row stores candidate final partial waves in order
    ``(l-1, l+1)`` and real photon channels ``(y, z, x)``.  It pads final
    real harmonics through :math:`l'=5`.  The flattened final-harmonic index
    is ``l_prime**2 + m_prime + l_prime``.

    ``channel_valid`` marks the complete allowed final-harmonic block.  It is
    deliberately distinct from a nonzero-coefficient mask: symmetry may make
    individual real-basis Gaunt coefficients exactly zero inside a valid
    block.

    :see: :class:`~.test_dipole.TestChannelTables`

    Parameters
    ----------
    basis : OrbitalBasis
        Static orbital quantum numbers.  Initial angular momenta must not
        exceed the package limit of four.

    Returns
    -------
    coupling_coeffs : Float64[Array, "n_orb 2 3 n_y"]
        Real-basis Gaunt coefficients.
    channel_valid : Float64[Array, "n_orb 2 3 n_y"]
        Zero/one mask for allowed partial-wave blocks and static padding.

    Raises
    ------
    ValueError
        If an initial angular momentum exceeds the supported table.

    Notes
    -----
    The table is host-side static data.  Production code multiplies by the
    mask and coefficients without data-dependent indexing.  The
    :math:`i^{l'}` plane-wave phase is absent here and belongs exclusively to
    the radial integral layer.
    """
    n_orbitals: int = len(basis.l)
    n_final_harmonics: int = (L_MAX + 2) ** 2
    final_m_offset: int = L_MAX + 1
    coupling_numpy: Float64[NDArray, "n_orb 2 3 n_y"] = np.zeros(
        (n_orbitals, 2, 3, n_final_harmonics),
        dtype=np.float64,
    )
    valid_numpy: Float64[NDArray, "n_orb 2 3 n_y"] = np.zeros_like(
        coupling_numpy
    )
    gaunt_numpy: Float64[NDArray, "..."] = np.asarray(GAUNT_TABLE)
    orbital_index: int
    l_initial: int
    m_initial: int
    branch_index: int
    l_final: int
    q_index: int
    m_final: int
    harmonic_index: int
    for orbital_index, (l_initial, m_initial) in enumerate(
        zip(basis.l, basis.m, strict=True)
    ):
        if l_initial > L_MAX:
            message: str = f"initial l={l_initial} exceeds L_MAX={L_MAX}"
            raise ValueError(message)
        for branch_index, l_final in enumerate((l_initial - 1, l_initial + 1)):
            if l_final < 0 or l_final > L_MAX + 1:
                continue
            for q_index in range(3):
                for m_final in range(-l_final, l_final + 1):
                    harmonic_index = l_final * l_final + m_final + l_final
                    coupling_numpy[
                        orbital_index,
                        branch_index,
                        q_index,
                        harmonic_index,
                    ] = float(
                        gaunt_numpy[
                            l_initial,
                            m_initial + L_MAX,
                            q_index,
                            l_final,
                            m_final + final_m_offset,
                        ]
                    )
                    valid_numpy[
                        orbital_index,
                        branch_index,
                        q_index,
                        harmonic_index,
                    ] = 1.0
    coupling_coeffs: Float64[Array, "n_orb 2 3 n_y"] = jnp.asarray(
        coupling_numpy
    )
    channel_valid: Float64[Array, "n_orb 2 3 n_y"] = jnp.asarray(valid_numpy)
    tables: Tuple[
        Float64[Array, "n_orb 2 3 n_y"],
        Float64[Array, "n_orb 2 3 n_y"],
    ] = (coupling_coeffs, channel_valid)
    return tables


@jaxtyped(typechecker=beartype)
def dipole_length_cartesian(
    psi_final: Complex128[Array, " n_q"],
    psi_initial: Complex128[Array, " n_q"],
    position_bohr: Float64[Array, "n_q 3"],
    volume_weights_bohr3: Float64[Array, " n_q"],
    polarization_cart: Complex128[Array, " 3"],
) -> Complex128[Array, ""]:
    r"""Compute a sampled Cartesian length-gauge contraction.

    The returned amplitude is
    :math:`\sum_j w_j\psi_f(j)^*
    [\boldsymbol\epsilon\mathbin{\cdot}\mathbf r_j]\psi_i(j)`.
    Polarization is not conjugated.

    :see: :class:`~.test_dipole.TestDipoleLengthCartesian`

    Parameters
    ----------
    psi_final : Complex128[Array, " n_q"]
        Final-state ket samples.
    psi_initial : Complex128[Array, " n_q"]
        Initial-state ket samples.
    position_bohr : Float64[Array, "n_q 3"]
        Cartesian quadrature positions in Bohr.
    volume_weights_bohr3 : Float64[Array, " n_q"]
        Volume quadrature weights in Bohr cubed.
    polarization_cart : Complex128[Array, " 3"]
        Cartesian complex polarization.

    Returns
    -------
    amplitude : Complex128[Array, ""]
        Length-gauge amplitude in Bohr.

    Notes
    -----
    The dot product uses polarization directly and conjugates only the
    final-state ket.
    """
    polarized_position: Complex128[Array, " n_q"] = (
        position_bohr @ polarization_cart
    )
    integrand: Complex128[Array, " n_q"] = (
        jnp.conj(psi_final) * polarized_position * psi_initial
    )
    amplitude: Complex128[Array, ""] = jnp.sum(
        volume_weights_bohr3 * integrand
    )
    return amplitude


@jaxtyped(typechecker=beartype)
def dipole_momentum_cartesian(
    psi_final: Complex128[Array, " n_q"],
    grad_psi_initial_bohr_inv: Complex128[Array, "n_q 3"],
    volume_weights_bohr3: Float64[Array, " n_q"],
    polarization_cart: Complex128[Array, " 3"],
) -> Complex128[Array, ""]:
    r"""Compute a sampled Cartesian momentum-gauge contraction.

    The returned amplitude applies
    :math:`\mathbf p=-i\nabla` directly:
    :math:`-i\sum_j w_j\psi_f(j)^*
    [\boldsymbol\epsilon\mathbin{\cdot}\nabla\psi_i(j)]`.
    It does not call or assume a length--momentum commutator identity.

    :see: :class:`~.test_dipole.TestDipoleMomentumCartesian`

    Parameters
    ----------
    psi_final : Complex128[Array, " n_q"]
        Final-state ket samples.
    grad_psi_initial_bohr_inv : Complex128[Array, "n_q 3"]
        Cartesian gradient of the initial-state ket in inverse Bohr.
    volume_weights_bohr3 : Float64[Array, " n_q"]
        Volume quadrature weights in Bohr cubed.
    polarization_cart : Complex128[Array, " 3"]
        Cartesian complex polarization.

    Returns
    -------
    amplitude : Complex128[Array, ""]
        Momentum-gauge amplitude in atomic momentum units.

    Notes
    -----
    The dot product applies ``-1j`` to the supplied initial-state gradient.
    """
    polarized_gradient: Complex128[Array, " n_q"] = (
        grad_psi_initial_bohr_inv @ polarization_cart
    )
    integrand: Complex128[Array, " n_q"] = (
        -1j * jnp.conj(psi_final) * polarized_gradient
    )
    amplitude: Complex128[Array, ""] = jnp.sum(
        volume_weights_bohr3 * integrand
    )
    return amplitude


__all__: list[str] = [
    "channel_tables",
    "dipole_length_cartesian",
    "dipole_momentum_cartesian",
    "polarization_cart_to_complex",
    "polarization_cart_to_real",
    "polarization_complex_to_cart",
    "polarization_real_to_cart",
]
