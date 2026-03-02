r"""Angular matrix elements for dipole photoemission.

Extended Summary
----------------
Provides Gaunt coefficients, real spherical harmonics, and full
dipole matrix element assembly for the differentiable ARPES
forward model.  The dipole matrix element for orbital
:math:`(n, l, m)` combines radial integrals, Gaunt coefficients,
and spherical harmonics of the photoelectron direction:

.. math::

    M(\mathbf{k}, n, l, m) = \sum_{l', m'} B_{n,l}^{l'}(|\mathbf{k}|)
        \cdot G(l, m, l', m') \cdot Y_{l'}^{m'}(\hat{k})
        \cdot \hat{e}_{q(m'-m)}

Routine Listings
----------------
:data:`GAUNT_TABLE`
    Precomputed dense array of real-valued Gaunt coefficients for
    dipole transitions up to l_max = 4 (s through g orbitals).
:data:`L_MAX`
    Maximum angular momentum supported by the precomputed Gaunt table.
:func:`build_gaunt_table`
    Build a dipole Gaunt coefficient lookup table for a given l_max.
:func:`gaunt_lookup`
    Look up a single Gaunt coefficient from the precomputed table.
:func:`real_spherical_harmonic`
    Evaluate a single real spherical harmonic :math:`Y_l^m(\theta, \varphi)`.
:func:`real_spherical_harmonics_all`
    Evaluate all real spherical harmonics up to a given l_max.
:func:`dipole_matrix_element_single`
    Compute the complex dipole matrix element for one orbital (n, l, m).
:func:`dipole_intensity_orbital`
    Compute :math:`|M|^2` for a single orbital.
:func:`dipole_intensities_all_orbitals`
    Compute :math:`|M|^2` for every orbital in a Slater basis set.

Notes
-----
All functions are JAX-compatible and support JIT compilation and
automatic differentiation.  The Gaunt table is computed once at
import time using pure Python and stored as a JAX array for O(1)
lookup during traced computation.
"""

from .dipole import (
    dipole_intensities_all_orbitals,
    dipole_intensity_orbital,
    dipole_matrix_element_single,
)
from .gaunt import GAUNT_TABLE, L_MAX, build_gaunt_table, gaunt_lookup
from .spherical_harmonics import (
    real_spherical_harmonic,
    real_spherical_harmonics_all,
)

__all__: list[str] = [
    "GAUNT_TABLE",
    "L_MAX",
    "build_gaunt_table",
    "dipole_intensities_all_orbitals",
    "dipole_intensity_orbital",
    "dipole_matrix_element_single",
    "gaunt_lookup",
    "real_spherical_harmonic",
    "real_spherical_harmonics_all",
]
