"""Provide differentiable ARPES simulations in JAX.

Extended Summary
----------------
The package provides Angle-Resolved PhotoEmission Spectroscopy (ARPES)
simulations with JAX automatic differentiation and GPU acceleration.
The same differentiable physics maps an electronic structure to ARPES
spectra and supports inverse recovery of band-structure parameters.
The package provides coherent matrix-element and intrinsic spectral
primitives.

Routine Listings
----------------
:mod:`inout`
    Parse VASP files for ARPES simulation input.
:mod:`certify`
    JAX-native scientific assurance for differentiable forward models.
:mod:`maths`
    Compute angular matrix elements for dipole photoemission.
:mod:`radial`
    Provide differentiable radial primitives for ARPES matrix elements.
:mod:`simul`
    Provide coherent matrix-element and spectral simulation primitives.
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
    inout,
    maths,
    radial,
    simul,
    tightb,
    types,
    utils,
)

__version__: str = version("diffpes")

__all__: list[str] = [
    "certify",
    "inout",
    "maths",
    "radial",
    "simul",
    "tightb",
    "types",
    "utils",
]
