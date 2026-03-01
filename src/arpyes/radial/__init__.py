r"""Differentiable radial primitives for ARPES matrix elements.

Extended Summary
----------------
Provides JAX-compatible spherical Bessel functions, atomic radial
wavefunctions, and fixed-grid radial-integral evaluation used by the
dipole-matrix-element pipeline.  The central quantity is the radial
integral

.. math::

    B^{l'}(k) = (i)^{l'} \int_0^\infty R(r)\, r^3\, j_{l'}(kr)\, dr

evaluated on a fixed radial grid via composite trapezoidal quadrature.

Routine Listings
----------------
:func:`spherical_bessel_jl`
    Spherical Bessel function :math:`j_l(x)` via stable upward
    recurrence with a small-argument limit.
:func:`radial_integral`
    Fixed-grid trapezoidal evaluation of the dipole radial integral
    :math:`B^{l'}(k)`.
:func:`slater_radial`
    Normalized Slater-type radial function
    :math:`R(r) = N\, r^{n-1} e^{-\zeta r}`.
:func:`hydrogenic_radial`
    Normalized hydrogenic radial function :math:`R_{n,l}(r)` using
    associated Laguerre polynomials.

Notes
-----
All functions are JAX-compatible and support JIT compilation and
automatic differentiation.  Recurrences (Bessel, Laguerre) use
``jax.lax.fori_loop`` for traceability.
"""

from .bessel import spherical_bessel_jl
from .integrate import radial_integral
from .wavefunctions import hydrogenic_radial, slater_radial

__all__: list[str] = [
    "hydrogenic_radial",
    "radial_integral",
    "slater_radial",
    "spherical_bessel_jl",
]
