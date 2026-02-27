"""Differentiable radial primitives for ARPES matrix elements.

Extended Summary
----------------
Provides JAX-compatible spherical Bessel functions, atomic radial
wavefunctions, and fixed-grid radial-integral evaluation used by the
dipole-matrix-element pipeline.
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
