"""Utility functions for ARPES simulations.

Extended Summary
----------------
Provides mathematical utilities used throughout the ARPES simulation
pipeline. The Faddeeva function (complex error function) is implemented
via a 64-term Taylor series and is used by the Voigt broadening profile.
Z-score normalization is provided for preprocessing spectra before
comparison with experiment.

Routine Listings
----------------
:func:`faddeeva`
    Faddeeva function w(z) = exp(-z^2) erfc(-iz) via Taylor series.
:func:`zscore_normalize`
    Z-score (zero-mean, unit-variance) normalization over all elements.

Notes
-----
All functions are JAX-compatible and support JIT compilation and
automatic differentiation. The Faddeeva implementation uses
``jax.lax.scan`` for the coefficient recurrence (no Python for loops).
"""

from .math import faddeeva, zscore_normalize

__all__: list[str] = [
    "faddeeva",
    "zscore_normalize",
]
