# diffpes

[![License](https://img.shields.io/pypi/l/diffpes.svg)](https://github.com/debangshu-mukherjee/diffpes/blob/main/LICENSE)
[![PyPI Downloads](https://static.pepy.tech/badge/diffpes)](https://pepy.tech/projects/diffpes)
[![PyPI version](https://img.shields.io/pypi/v/diffpes.svg)](https://pypi.python.org/pypi/diffpes)
[![Python Versions](https://img.shields.io/pypi/pyversions/diffpes.svg)](https://pypi.python.org/pypi/diffpes)
[![Documentation Status](https://readthedocs.org/projects/diffpes/badge/?version=latest)](https://diffpes.readthedocs.io/en/latest/?badge=latest)
[![tests](https://github.com/debangshu-mukherjee/diffpes/actions/workflows/tests.yml/badge.svg)](https://github.com/debangshu-mukherjee/diffpes/actions/workflows/tests.yml)
[![codecov](https://codecov.io/gh/debangshu-mukherjee/diffpes/graph/badge.svg)](https://codecov.io/gh/debangshu-mukherjee/diffpes)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19037631.svg)](https://doi.org/10.5281/zenodo.19037631)
[![Ruff](https://img.shields.io/badge/lint%20and%20format-ruff-D7FF64?logo=ruff&logoColor=1D1D1D)](https://github.com/astral-sh/ruff)
[![ty](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ty/main/assets/badge/v0.json)](https://github.com/astral-sh/ty)
[![jax_badge](https://tinyurl.com/mucknrvu)](https://docs.jax.dev/)
[![Lines of Code](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/debangshu-mukherjee/diffpes/main/.github/badges/loc.json)](https://github.com/debangshu-mukherjee/diffpes)

Differentiable ARPES simulation in JAX: from tight-binding or DFT electronic
structure to band structures, spectra, Fermi surfaces, and detector-level
photoemission data. Gradients flow through the entire chain.

<p align="center">
  <b><a href="https://diffpes.readthedocs.io/en/latest/tutorials/">Tutorials</a></b> ·
  <b><a href="https://diffpes.readthedocs.io/en/latest/guides/">Guides</a></b> ·
  <b><a href="https://diffpes.readthedocs.io/en/latest/api/">API reference</a></b>
</p>

```bash
pip install diffpes
```

## From bands to spectra

<p align="center">
  <img src="docs/source/_static/readme/graphene-bands-to-arpes.png" alt="Graphene tight-binding band structure along Gamma-K-M-Gamma next to the simulated ARPES spectrum of the same path" width="900">
</p>

The graphene π bands and their simulated ARPES spectrum: the occupied band
is bright. The Fermi function at 300 K cuts off the unoccupied band.

<p align="center">
  <img src="docs/source/_static/readme/graphene-linewidth-series.png" alt="The same Dirac cone cut simulated with 20, 80, and 250 meV self-energy linewidths" width="900">
</p>

The self-energy is an explicit model parameter — the same cut at 20, 80,
and 250 meV linewidth.

<p align="center">
  <img src="docs/source/_static/readme/honeycomb-gap-bands-arpes.png" alt="Gapped honeycomb band structure and the simulated ARPES spectrum showing only the occupied valence band" width="900">
</p>

Stagger the two onsite energies and the Dirac crossing opens into a gap;
ARPES sees the valence band.

<p align="center">
  <img src="docs/source/_static/readme/graphene-fermi-edge.png" alt="Momentum-summed Fermi edge simulated at 25, 100, and 300 kelvin" width="620">
</p>

The Fermi edge at 25, 100, and 300 K.

## Across the Brillouin zone

<p align="center">
  <img src="docs/source/_static/readme/graphene-constant-energy-maps.png" alt="Four constant-energy ARPES maps of graphene: Dirac points, trigonally warped pockets, the van Hove crossing at M, and a Gamma-centered ring" width="900">
</p>

Constant-energy slices: Dirac points grow into trigonally warped pockets,
touch at the van Hove singularity, and close into a ring around Γ.

<p align="center">
  <img src="docs/source/_static/readme/graphene-doped-fermi-surface.png" alt="Simulated Fermi surface of hole-doped graphene with pockets at every zone corner" width="520">
</p>

Move the Fermi level and the Fermi surface follows — hole-doped by 1 eV.

<p align="center">
  <img src="docs/source/_static/readme/graphene-band-surface-3d.png" alt="Three-dimensional pi and pi-star band surfaces of graphene with six Dirac points" width="620">
</p>

<p align="center">
  <img src="docs/source/_static/readme/graphene-dos.png" alt="Gaussian-broadened pi-band density of states with van Hove peaks and the occupied part shaded" width="620">
</p>

## The intensity cube

<p align="center">
  <img src="docs/source/_static/readme/graphene-dirac-cone-cube.png" alt="Translucent three-dimensional rendering of the simulated ARPES intensity cube around the Dirac point" width="620">
</p>

The full cube I(kx, ky, E) around one Dirac point, rendered by intensity
transparency over its deepest constant-energy slice.

<p align="center">
  <img src="docs/source/_static/readme/graphene-energy-window-maps.png" alt="Three energy-window integrals of the Dirac cone cube showing rings collapsing onto the apex" width="900">
</p>

Energy-window integrals of the cube — rings collapse onto the cone apex.

<p align="center">
  <img src="docs/source/_static/readme/graphene-edc-mdc.png" alt="Stacked energy distribution curves and momentum distribution curves through the Dirac cone" width="900">
</p>

EDC and MDC stacks pulled from the same cut.

## Through the detector

<p align="center">
  <img src="docs/source/_static/readme/intrinsic-vs-measured.png" alt="Intrinsic spectral function beside the same spectrum after analyser optics, resolution, background, and Poisson counting" width="900">
</p>

The same physics before and after the instrument: analyser calibration,
point-spread functions, transmission, background, exposure, and counting.

<p align="center">
  <img src="docs/source/_static/readme/detector-poisson-acquisition.png" alt="Expected detector image beside one Poisson-sampled acquisition of the same spectrum" width="900">
</p>

Expected counts and one reproducible Poisson acquisition.

<p align="center">
  <img src="docs/source/_static/readme/detector-expected-counts.png" alt="Expected photoelectron counts over the detector plane" width="520">
</p>

<p align="center">
  <img src="docs/source/_static/readme/detector-polarization-contrast.png" alt="Difference of expected detector counts between two photon polarizations" width="520">
</p>

Rotate the photon polarization and difference the two acquisitions —
matrix-element contrast at the detector.

## First spectrum

This script is complete — a graphene π-band model, a momentum cut through
the Dirac point, and the finite-temperature spectral function:

```python
import jax.numpy as jnp
import matplotlib.pyplot as plt

from diffpes.plots import plot_arpes_spectrum
from diffpes.simul import assemble_spectral_intensity_bands_chunk
from diffpes.tightb import (
    build_kpath,
    diagonalize_tb,
    kpath_arc_length,
    kpoints_frac_to_cart,
)
from diffpes.types import (
    make_arpes_spectrum,
    make_crystal_geometry,
    make_orbital_basis,
    make_self_energy_model,
    make_tb_model,
)

# Graphene pi bands: two carbon sites, six nearest-neighbour hoppings.
a = 2.46
crystal = make_crystal_geometry(
    lattice=jnp.asarray(
        [[a, 0.0, 0.0], [a / 2, a * 3**0.5 / 2, 0.0], [0.0, 0.0, 20.0]]
    ),
    positions=jnp.asarray([[0.0, 0.0, 0.0], [1 / 3, 1 / 3, 0.0]]),
    species=("C", "C"),
)
basis = make_orbital_basis(
    atom_indices=(0, 1), n=(2, 2), l=(0, 0), m=(0, 0), labels=("pz_A", "pz_B")
)
model = make_tb_model(
    hopping_amplitudes=-2.7 * jnp.ones(6, dtype=jnp.complex128),
    onsite_energies=jnp.zeros(2),
    soc_lambdas=jnp.zeros(0),
    geometry=crystal,
    basis=basis,
    hopping_pairs=((0, 1), (0, 1), (0, 1), (1, 0), (1, 0), (1, 0)),
    hopping_cells=(
        (0, 0, 0), (-1, 0, 0), (0, -1, 0),
        (0, 0, 0), (1, 0, 0), (0, 1, 0),
    ),
    shell_index=(-1, -1),
)

# A straight momentum cut through the Dirac point at K.
path = build_kpath(
    jnp.asarray([[0.0, 0.0, 0.0], [0.5, 1.0, 0.0]]),
    crystal,
    301,
    ("Gamma", "K'"),
)
bands = diagonalize_tb(model, path.kpoints)

# Occupied spectral function: Lorentzian self-energy + Fermi cutoff at 300 K.
energies = jnp.linspace(-9.2, 1.2, 480)
intensity = assemble_spectral_intensity_bands_chunk(
    bands.eigenvalues,
    jnp.ones((path.kpoints.shape[0], energies.shape[0], 2)),
    energies,
    make_self_energy_model(gamma=0.09),
    jnp.asarray(0.0),
    300.0,
    allow_degenerate_value_only=True,
)
spectrum = make_arpes_spectrum(
    intensity,
    energies,
    kpath_arc_length(path, crystal),
    kpoints_frac_to_cart(path.kpoints, crystal),
)
plot_arpes_spectrum(spectrum, cmap="magma")
plt.show()
```

## From DFT

<p align="center">
  <img src="docs/source/_static/readme/bi2se3-dft-bands-to-arpes.png" alt="Bi2Se3 slab band structure read from a VASP EIGENVAL file next to its simulated occupied ARPES map along M-Gamma-M" width="900">
</p>

A real material, straight from VASP output: the Bi₂Se₃ slab bands along
M–Γ–M and their occupied spectrum at 35 K.

<p align="center">
  <img src="docs/source/_static/readme/bi2se3-surface-state-window.png" alt="Near-Fermi window of the Bi2Se3 slab spectrum with quantum-well-split conduction states above the valence manifold" width="620">
</p>

The near-Fermi window of the same calculation, sharpened to a 12 meV
linewidth.

<p align="center">
  <img src="docs/source/_static/readme/bi2se3-slab-vs-bulk.png" alt="Simulated spectra of the Bi2Se3 six-quintuple-layer slab and of bulk Bi2Se3 on the same M-Gamma-K-M path, with in-gap states only in the slab" width="900">
</p>

Slab and bulk calculations on the same M–Γ–K–M path: the slab carries
states inside the bulk gap.

<p align="center">
  <img src="docs/source/_static/readme/bi2se3-edc-stack.png" alt="Stacked energy distribution curves around Gamma from the Bi2Se3 slab spectrum" width="620">
</p>

EDCs around Γ from the same map.

<p align="center">
  <img src="docs/source/_static/readme/bi2se3-dos.png" alt="Normalized densities of states of the Bi2Se3 slab and bulk from DOSCAR files" width="620">
</p>

<p align="center">
  <img src="docs/source/_static/readme/bi2se3-charge-profile.png" alt="Planar-averaged CHGCAR charge density of the slab showing six quintuple layers separated by van der Waals gaps" width="900">
</p>

DOSCAR densities of states and the CHGCAR charge density, resolving all
six quintuple layers.

`diffpes.inout` reads `EIGENVAL`, `PROCAR`, `POSCAR`, `KPOINTS`, `OUTCAR`,
`DOSCAR`, `CHGCAR`, `WAVECAR`, Wannier90 `hr.dat`/`tb.dat`, and Cartesian
hopping lists. Parsed eigenvalues and orbital projections drop into the same
spectral calls as tight-binding models, so a converged VASP calculation
becomes a simulated ARPES measurement.

## Differentiable end to end

`jax.grad`, `jax.vmap`, and `jax.jit` work through the whole pipeline —
crystal geometry, hoppings, self-energy, matrix elements, experiment
geometry, and detector response to expected counts. Fit any of it to
measured spectra by gradient descent.

## Agent-runnable experiments

The `automatons/` directory provides standalone experiments for automated
diffpes workflows.
Each file supports discovery, parameter validation, smoke execution, and a
final JSON result.

Run a small example with:

```bash
JAX_PLATFORMS=cpu MPLCONFIGDIR=/tmp/dp-mpl .venv/bin/python \
  automatons/forward_bands.py --smoke --outdir /tmp/dp-bands --json
```

Read the [experiment catalog](https://github.com/debangshu-mukherjee/diffpes/blob/main/automatons/INDEX.md)
before selecting a file.
Read the [agent guide](https://github.com/debangshu-mukherjee/diffpes/blob/main/docs/source/guides/running-experiments-as-an-agent.md)
for the complete execution protocol.
