"""Provide differentiable ARPES simulations in JAX.

Extended Summary
----------------
The package provides Angle-Resolved PhotoEmission Spectroscopy (ARPES)
simulations with JAX automatic differentiation and GPU acceleration.
The same differentiable physics maps an electronic structure to ARPES
spectra and supports inverse recovery of band-structure parameters.
The package provides coherent matrix-element and intrinsic spectral
primitives through separate top-level modules.

The following list describes the top-level modules:

- :mod:`certify`
    Certify differentiable DiffPES forward-model executions.
- :mod:`constants`
    Expose the declarative constants of diffpes.
- :mod:`inout`
    Expose the public :mod:`diffpes.inout` surface.
- :mod:`maths`
    Compute angular matrix elements for dipole photoemission.
- :mod:`matrixel`
    Assemble coherent orbital and band photoemission matrix elements.
- :mod:`radial`
    Provide differentiable radial primitives for ARPES matrix elements.
- :mod:`simul`
    Expose the public :mod:`diffpes.simul` surface.
- :mod:`tightb`
    Provide native tight-binding tools and ARPES-side adapters.
- :mod:`types`
    Expose the public :mod:`diffpes.types` surface.
- :mod:`utils`
    Expose the public :mod:`diffpes.utils` surface.

Routine Listings
----------------
:mod:`certify`
    Certify differentiable DiffPES forward-model executions.
:mod:`constants`
    Expose the declarative constants of diffpes.
:mod:`inout`
    Parse VASP files for ARPES simulation input.
:mod:`maths`
    Compute angular matrix elements for dipole photoemission.
:mod:`matrixel`
    Assemble coherent orbital and band photoemission matrix elements.
:mod:`radial`
    Provide differentiable radial primitives for ARPES matrix elements.
:mod:`simul`
    Provide differentiable ARPES simulation primitives.
:mod:`tightb`
    Provide native tight-binding tools and ARPES-side adapters.
:mod:`types`
    Define types and factory functions for diffpes.
:mod:`utils`
    Provide utility functions for ARPES simulations.

Examples
--------
>>> import diffpes
>>> import jax.numpy as jnp
>>> omega = jnp.linspace(-1.0, 1.0, 101)
>>> model = diffpes.types.make_self_energy_model(gamma=0.1)
>>> sigma = diffpes.simul.evaluate_self_energy(omega, model)

Notes
-----
All computations support JAX transformations and automatic differentiation
of ARPES simulation parameters. The package enables 64-bit precision during
import. It also sets the XLA CPU threading flags before it imports JAX.
"""

import collections.abc
import os
from importlib.metadata import version

if not hasattr(collections.abc, "ByteString"):
    setattr(  # noqa: B010 -- Python 3.14 compatibility for beartype 0.22.9.
        collections.abc,
        "ByteString",
        collections.abc.Buffer,
    )

os.environ.setdefault(
    "XLA_FLAGS",
    "--xla_cpu_multi_thread_eigen=true intra_op_parallelism_threads=0",
)

import jax  # noqa: E402

jax.config.update("jax_enable_x64", True)

from . import (  # noqa: E402
    certify,
    constants,
    inout,
    maths,
    matrixel,
    radial,
    simul,
    tightb,
    types,
    utils,
)

__version__: str = version("diffpes")

__all__: list[str] = [
    "certify",
    "constants",
    "inout",
    "maths",
    "matrixel",
    "radial",
    "simul",
    "tightb",
    "types",
    "utils",
]
