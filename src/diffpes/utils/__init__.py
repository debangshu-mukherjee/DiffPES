"""Provide utility functions for ARPES simulations.

Extended Summary
----------------
The subpackage provides mathematical utilities for ARPES simulations. The
Faddeeva function uses a fixed-order Weideman rational approximation. The
Voigt broadening profile uses this function. Z-score normalization prepares
spectra for comparisons with experiments. Complex packing functions provide
the required real optimizer boundary for complex physics parameters.

The following list describes the submodules:

- :mod:`math`
    Compute mathematical utilities for ARPES simulations.

Routine Listings
----------------
:func:`faddeeva`
    Evaluate the Faddeeva function w(z) = exp(-z^2) erfc(-iz).
:func:`pack_complex`
    Pack complex parameters as stacked real values.
:func:`unpack_complex`
    Unpack stacked real parameters into complex values.
:func:`zscore_normalize`
    Apply z-score normalization (zero-mean, unit-variance).

Notes
-----
All functions support JAX transformations and automatic differentiation. The
Faddeeva implementation uses one rational region in the upper half-plane.
"""

from .math import faddeeva, pack_complex, unpack_complex, zscore_normalize

__all__: list[str] = [
    "faddeeva",
    "pack_complex",
    "unpack_complex",
    "zscore_normalize",
]
