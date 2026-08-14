API Reference
=============

JAX-based differentiable ARPES simulation package.

diffpes provides a differentiable pipeline connecting electronic band
structure to Angle-Resolved PhotoEmission Spectroscopy (ARPES) spectra.
Built on JAX and Equinox, the same forward physics that maps band
structures to spectra also supports gradient-based inverse recovery.
Coherent primitives preserve orbital amplitudes through radial, angular,
polarization, band-projection, and intrinsic spectral stages.

Submodules
----------

.. toctree::
   :maxdepth: 1
   :hidden:

   certify
   constants
   inout
   maths
   matrixel
   plots
   radial
   simul
   tightb
   types

:mod:`diffpes.certify`
    JAX-native scientific certification, provenance, evidence, policy
    evaluation, information-flow diagnostics, and inspection.

:mod:`diffpes.constants`
    Immutable physical values, schema identifiers, parser tokens, selector
    vocabularies, and validation tolerances.

:mod:`diffpes.inout`
    VASP file parsers (POSCAR, EIGENVAL, KPOINTS, DOSCAR, PROCAR, CHGCAR),
    HDF5 persistence, and workflow helpers for ARPES simulation input.

:mod:`diffpes.maths`
    Angular matrix elements for dipole photoemission: Gaunt coefficients,
    real spherical harmonics, and dipole matrix element assembly. Also the
    Faddeeva function, complex parameter packing, and z-score
    normalization.

:mod:`diffpes.matrixel`
    Coherent photoemission channels, band projection, late polarization
    contraction, and differentiable inversion coordinates.

:mod:`diffpes.plots`
    Matplotlib rendering of diffpes carriers: ARPES maps, projected band
    scatters, distribution curves, and detector-plane images.

:mod:`diffpes.radial`
    Differentiable radial primitives: spherical Bessel functions, atomic
    radial wavefunctions, and fixed-grid radial integrals.

:mod:`diffpes.simul`
    Spectral assembly, detector mapping, broadening, cross sections,
    polarization, orbital angular momentum, free-electron kinematics, and
    static-shape sharding.

:mod:`diffpes.tightb`
    Native tight-binding model construction, diagonalization, and
    ARPES-side adapters for external electronic-structure sources.

:mod:`diffpes.types`
    PyTree-compatible physical inputs, forward outputs, certification records,
    transformation contracts, and provenance carriers.

Examples
--------

.. code-block:: python

    import diffpes as dp

    import jax.numpy as jnp

    omega = jnp.linspace(-1.0, 1.0, 101)
    model = dp.types.make_self_energy_model(gamma=0.1)
    sigma = dp.simul.evaluate_self_energy(omega, model)

Notes
-----

All computations are JAX-compatible and support automatic differentiation
for gradient-based recovery of band-structure parameters from measured
ARPES spectra.
