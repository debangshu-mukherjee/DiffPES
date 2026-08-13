"""Expose the public :mod:`diffpes.utils` surface.

Extended Summary
----------------
- :mod:`math`
    Compute mathematical utilities for ARPES simulations.
- :mod:`sharding`
    Compute static-shape sharding operations.

Routine Listings
----------------
:func:`faddeeva`
    Evaluate the Faddeeva function w(z) = exp(-z^2) erfc(-iz).
:func:`pack_complex`
    Pack complex parameters as stacked real values.
:func:`pad_with_mask`
    Compute the ``pad_with_mask`` public contract.
:func:`sharded_kmap`
    Compute the ``sharded_kmap`` public contract.
:func:`sharded_ksum`
    Compute the ``sharded_ksum`` public contract.
:func:`unpack_complex`
    Unpack stacked real parameters into complex values.
:func:`zscore_normalize`
    Apply z-score normalization (zero-mean, unit-variance).
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
