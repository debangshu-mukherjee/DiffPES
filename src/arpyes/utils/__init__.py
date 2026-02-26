"""Utility functions for ARPES simulations.

Extended Summary
----------------
Provides mathematical utilities including the Faddeeva function
and data normalization routines.

Routine Listings
----------------
:func:`faddeeva`
    Faddeeva function via Weideman's rational approximation.
:func:`zscore_normalize`
    Z-score normalization.

Notes
-----
All functions are JAX-compatible and support JIT compilation.
"""

from .math import faddeeva, zscore_normalize

__all__: list[str] = [
    "faddeeva",
    "zscore_normalize",
]
