# diffpes Theory and Architecture Guides

These guides describe the physics and software architecture of diffpes.
diffpes uses JAX for differentiable Angle-Resolved PhotoEmission Spectroscopy
(ARPES) simulations.

## Target Audience

These guides support **physics researchers** who work with ARPES and want to
understand:

- The photoemission physics across coherent assembly stages
- Electronic-structure data flow through the simulation pipeline
- The physical meaning of simulation parameters and outputs
- Differentiability for inverse recovery of band-structure parameters

## Guide Overview

### Physics Foundations

| Guide | Description |
|-------|-------------|
| [ARPES Geometry and Kinematics](arpes-geometry-and-kinematics.md) | Photoemission geometry, energy and momentum conservation, and detector coordinates |
| [Coherent Spectral Assembly](simulation-levels.md) | Matrix-element sources, resolvent/eigen spectral paths, and the detector boundary |
| [Matrix Elements and Polarization](matrix-elements-and-polarization.md) | Radial integrals, Gaunt coefficients, spherical harmonics, and light-polarization effects |
| [Spectral Broadening and Self-Energy](spectral-broadening-and-self-energy.md) | Voigt profiles, resolution convolution, and self-energy models |

### Data and Architecture

| Guide | Description |
|-------|-------------|
| [PyTree Architecture](pytree-architecture.md) | Equinox data structures enabling GPU acceleration and autodiff |
| [JAX Transformability and Gradients](jax-transformability-and-gradients.md) | Support for `grad`, `vmap`, and `jit`, with gradient flow through the forward model |
| [Certified Forward Models](certified-forward-models.md) | Bounded scientific claims, provenance, differentiable evidence, information flow, and portable records |
| [VASP Data Ingestion](vasp-data-ingestion.md) | Parsing POSCAR, EIGENVAL, KPOINTS, DOSCAR, PROCAR, and CHGCAR into PyTrees |

## Quick Start

For hands-on examples, see the [tutorials](../tutorials/index.md).

## Mathematical Notation

The guides use these symbols:

- $\mathbf{k}$ for wavevectors (in $\text{Å}^{-1}$)
- $E_B$ for binding energy and $E_F$ for the Fermi level (in eV)
- $h\nu$ for photon energy (in eV)
- $(n, l, m)$ for orbital quantum numbers
- $\theta$ for polar emission angle
- $\phi$ for azimuthal angle
- $\Sigma(\omega)$ for the electron self-energy

GitHub and MathJax render the equations from compatible LaTeX notation.
