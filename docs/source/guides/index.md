# diffpes Guides

diffpes simulates Angle-Resolved PhotoEmission Spectroscopy (ARPES) with
differentiable JAX programs. To simulate a spectrum, you build an
electronic-structure source, matrix-element carriers, an experiment
geometry, a self-energy, and a detector calibration. Then run
{func}`~diffpes.simul.simulate_arpes` or
{func}`~diffpes.simul.simulate_arpes_cut`.

Start with [Simulating ARPES Spectra](simulating-arpes-spectra.md). It
builds every driver input and runs one complete simulation from a
tight-binding model to Poisson-sampled detector counts. The executed
[tutorials](../tutorials/index.md) run the same pipelines with plots at
every stage. Every spectrum figure in these guides comes from the public
API; `docs/make_guide_figures.py` regenerates all of them.

## Physics Guides

| Guide | What it covers |
|-------|-------------|
| [Simulating ARPES Spectra](simulating-arpes-spectra.md) | The complete forward pipeline: sources, drivers, `kz_mode` selection, and the two spectral paths |
| [ARPES Geometry and Kinematics](arpes-geometry-and-kinematics.md) | Photoemission geometry, energy and momentum conservation, and detector coordinates |
| [kz Broadening and Photon-Energy Scans](kz-broadening-and-photon-energy-scans.md) | Bulk-kz integration with finite escape depth and differentiable $h\nu$ scans |
| [Matrix Elements and Polarization](matrix-elements-and-polarization.md) | Radial integrals, Gaunt couplings, atomic-centre phases, and light-polarization effects |
| [Spectral Broadening and Self-Energy](spectral-broadening-and-self-energy.md) | Voigt profiles, self-energy models, and the instrument response chain |

## Data and Architecture Guides

| Guide | What it covers |
|-------|-------------|
| [VASP Data Ingestion](vasp-data-ingestion.md) | Parsing POSCAR, EIGENVAL, KPOINTS, DOSCAR, PROCAR, and CHGCAR into PyTrees |
| [PyTree Architecture](pytree-architecture.md) | Equinox carriers, static versus traced fields, and factory validation |
| [JAX Transformability and Gradients](jax-transformability-and-gradients.md) | `grad`, `vmap`, and `jit` through the forward model, and optimizer coordinates |
| [Certified Forward Models](certified-forward-models.md) | Recording provenance, validity checks, and derivative evidence for one run |

## Mathematical Notation

- $\mathbf{k}$ for wavevectors (in $\text{Å}^{-1}$)
- $E_B$ for binding energy and $E_F$ for the Fermi level (in eV)
- $h\nu$ for photon energy (in eV)
- $(n, l, m)$ for orbital quantum numbers
- $\theta$ for polar emission angle and $\phi$ for azimuthal angle
- $\Sigma(\omega)$ for the electron self-energy
