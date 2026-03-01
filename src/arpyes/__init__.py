"""Differentiable ARPES simulations in JAX.

Extended Summary
----------------
A comprehensive toolkit for Angle-Resolved PhotoEmission
Spectroscopy (ARPES) simulations using JAX for automatic
differentiation and GPU acceleration. Supports five levels
of physical sophistication from basic Gaussian convolution
to full polarization-dependent dipole matrix element
calculations.

Routine Listings
----------------
:mod:`inout`
    VASP file parsers (POSCAR, EIGENVAL, KPOINTS, DOSCAR, PROCAR).
:mod:`radial`
    Differentiable radial primitives (Bessel, wavefunctions, integrals).
:mod:`simul`
    ARPES simulation functions at five complexity levels.
:mod:`types`
    PyTree data structures and factory functions.
:mod:`utils`
    Mathematical utilities (Faddeeva function, normalization).

Examples
--------
>>> import arpyes
>>> bands = arpyes.inout.read_eigenval("EIGENVAL", fermi_energy=-1.5)
>>> orb = arpyes.inout.read_procar("PROCAR")
>>> params = arpyes.types.make_simulation_params(sigma=0.04)
>>> spectrum = arpyes.simul.simulate_basic(bands, orb, params)

Notes
-----
All computations are JAX-compatible and support automatic
differentiation for gradient-based optimization of ARPES
simulation parameters.
"""

import os
from importlib.metadata import version

os.environ.setdefault(
    "XLA_FLAGS",
    "--xla_cpu_multi_thread_eigen=true intra_op_parallelism_threads=0",
)

import jax  # noqa: E402

jax.config.update("jax_enable_x64", True)

from . import inout, matrix_elements, radial, simul, tb, types, utils  # noqa: E402

__version__: str = version("arpyes")

__all__: list[str] = [
    "__version__",
    "inout",
    "matrix_elements",
    "radial",
    "simul",
    "tb",
    "types",
    "utils",
]
