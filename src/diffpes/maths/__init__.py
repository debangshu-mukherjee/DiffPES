r"""Compute angular matrix elements for dipole photoemission.

Extended Summary
----------------
The subpackage provides Gaunt coefficients, harmonic basis maps, and
gradient-safe angular primitives.  Dipole helpers keep the coherent
complex-amplitude channels intact for the matrix-element assembler.

The following list describes the submodules:

- :mod:`dipole`
    Convert dipole channels and compute Cartesian dipole gauges.
- :mod:`gaunt`
    Build the Gaunt coefficient table for dipole transitions.
- :mod:`rotations`
    Construct differentiable three-dimensional rotations.
- :mod:`safe`
    Provide named gradient-safe elementary operations.
- :mod:`spherical_harmonics`
    Compute real spherical harmonics in JAX.

Routine Listings
----------------
:func:`bond_angles`
    Convert a Cartesian bond to safe polar and azimuthal angles.
:func:`build_gaunt_table`
    Build the dipole Gaunt coefficient lookup table.
:func:`channel_tables`
    Build padded real-harmonic dipole channel tables.
:func:`dipole_length_cartesian`
    Compute a sampled Cartesian length-gauge contraction.
:func:`dipole_momentum_cartesian`
    Compute a sampled Cartesian momentum-gauge contraction.
:func:`gaunt_lookup`
    Look up a single Gaunt coefficient from the precomputed table.
:func:`polarization_cart_to_complex`
    Convert Cartesian polarization to complex spherical components.
:func:`polarization_cart_to_real`
    Convert Cartesian polarization to real-harmonic channel order.
:func:`polarization_complex_to_cart`
    Convert complex spherical polarization to Cartesian components.
:func:`polarization_real_to_cart`
    Convert real-harmonic polarization back to Cartesian order.
:func:`real_harmonic_unitary`
    Construct the complex-to-real harmonic basis-function unitary.
:func:`real_spherical_harmonic`
    Evaluate a single real spherical harmonic.
:func:`real_spherical_harmonics_all`
    Evaluate all real spherical harmonics up to l_max.
:func:`rodrigues_rotation`
    Construct a rotation matrix with Rodrigues' formula.
:func:`safe_arccos`
    Evaluate arccos with saturated values and zero boundary gradients.
:func:`safe_arctan2`
    Evaluate arctan2 with a zero value and gradient at the origin.
:func:`safe_divide`
    Divide with a fallback and zero quotient gradients at zero denominators.
:func:`safe_log`
    Evaluate log with a finite floor and zero gradients below it.
:func:`safe_norm`
    Compute a Euclidean norm with a zero gradient at zero vectors.
:func:`safe_power`
    Raise positive inputs to a power and return zero otherwise.
:func:`safe_sqrt`
    Evaluate sqrt on positive inputs and return zero otherwise.
:func:`wigner_d`
    Construct a Wigner D matrix for an active z--y--z rotation.
:func:`wigner_small_d`
    Construct a Wigner small-d matrix from its finite factorial sum.
:obj:`GAUNT_TABLE`
    Module-level precomputed Gaunt coefficient table for l_max=4.

Notes
-----
All functions support JAX transformations and automatic differentiation. Pure
Python computes the Gaunt table once during import. The module stores the
table as a JAX array for constant-time lookup during traced computation.
"""

from .dipole import (
    channel_tables,
    dipole_length_cartesian,
    dipole_momentum_cartesian,
    polarization_cart_to_complex,
    polarization_cart_to_real,
    polarization_complex_to_cart,
    polarization_real_to_cart,
)
from .gaunt import GAUNT_TABLE, build_gaunt_table, gaunt_lookup
from .rotations import (
    bond_angles,
    real_harmonic_unitary,
    rodrigues_rotation,
    wigner_d,
    wigner_small_d,
)
from .safe import (
    safe_arccos,
    safe_arctan2,
    safe_divide,
    safe_log,
    safe_norm,
    safe_power,
    safe_sqrt,
)
from .spherical_harmonics import (
    real_spherical_harmonic,
    real_spherical_harmonics_all,
)

__all__: list[str] = [
    "bond_angles",
    "build_gaunt_table",
    "channel_tables",
    "dipole_length_cartesian",
    "dipole_momentum_cartesian",
    "gaunt_lookup",
    "polarization_cart_to_complex",
    "polarization_cart_to_real",
    "polarization_complex_to_cart",
    "polarization_real_to_cart",
    "real_harmonic_unitary",
    "real_spherical_harmonic",
    "real_spherical_harmonics_all",
    "rodrigues_rotation",
    "safe_arccos",
    "safe_arctan2",
    "safe_divide",
    "safe_log",
    "safe_norm",
    "safe_power",
    "safe_sqrt",
    "wigner_d",
    "wigner_small_d",
    "GAUNT_TABLE",
]
