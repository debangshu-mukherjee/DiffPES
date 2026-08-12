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
- :mod:`sharding`
    Compute static-shape padding and checkpointed k-point reductions.

Routine Listings
----------------
:func:`faddeeva`
    Evaluate the Faddeeva function w(z) = exp(-z^2) erfc(-iz).
:func:`pack_complex`
    Pack complex parameters as stacked real values.
:func:`pad_with_mask`
    Pad physical k points with finite repeated values and an f64 mask.
:func:`sharded_kmap`
    Map a checkpointed function over static k-point chunks.
:func:`sharded_ksum`
    Sum scalar outputs over static k-point chunks.
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
from .sharding import pad_with_mask, sharded_kmap, sharded_ksum

__all__: list[str] = [
    "faddeeva",
    "pack_complex",
    "pad_with_mask",
    "sharded_kmap",
    "sharded_ksum",
    "unpack_complex",
    "zscore_normalize",
]
