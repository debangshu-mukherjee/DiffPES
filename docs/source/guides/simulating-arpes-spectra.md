# Simulating ARPES Spectra

An ARPES simulation in diffpes runs one pipeline:

$$
|\hat\epsilon\!\cdot\!M|^2
\rightarrow A(k,\omega)
\rightarrow f_{FD}(\omega;T)
\rightarrow \widetilde T(E_{kin})
\rightarrow \text{detector resolution}
\rightarrow \text{expected counts}.
$$

Matrix elements weight each band. The spectral function and Fermi occupation
distribute that weight over energy. The detector chain converts the
result into expected counts on native analyser bins.
{func}`~diffpes.simul.simulate_arpes` runs the whole pipeline from a raster;
{func}`~diffpes.simul.simulate_arpes_cut` does the same along a momentum
path. This guide builds every input those drivers need and runs one complete
simulation. Every figure on this page comes from the public API through
`docs/make_guide_figures.py`.

```{figure} figures/pipeline-cube.png
:alt: Three-face graphene spectral cube with Dirac cones on the side faces
:width: 88%

A graphene spectral cube $I(k_x, k_y, E)$. The vertical faces are
momentum-energy slices through Dirac cones, and the top face is a
constant-energy map showing the K-point pockets.
```

## Step 1: An Electronic-Structure Source

The driver needs Bloch Hamiltonians and diagonalized bands on a momentum
raster. Here a one-orbital square-lattice tight-binding model keeps the
example small. `diffpes.tightb` builds arbitrary multi-orbital models with
Slater--Koster parameters and spin--orbit coupling.
[VASP Data Ingestion](vasp-data-ingestion.md) covers DFT-derived inputs.

```python
import jax.numpy as jnp

from diffpes.tightb import bloch_hamiltonian_batch, diagonalize_tb
from diffpes.types import (
    make_crystal_geometry,
    make_kgrid,
    make_orbital_basis,
    make_tb_model,
)

crystal = make_crystal_geometry(
    lattice=2.0 * jnp.pi * jnp.eye(3),
    positions=jnp.zeros((1, 3)),
    species=("X",),
)
basis = make_orbital_basis(
    atom_indices=(0,), n=(1,), l=(0,), m=(0,), labels=("1s",)
)
model = make_tb_model(
    hopping_amplitudes=0.18 * jnp.ones(4, dtype=jnp.complex128),
    onsite_energies=jnp.asarray([-0.36]),
    soc_lambdas=jnp.zeros(0),
    geometry=crystal,
    basis=basis,
    hopping_pairs=((0, 0), (0, 0), (0, 0), (0, 0)),
    hopping_cells=((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0)),
    shell_index=(-1,),
    depths=jnp.asarray([0.25]),
)

k_axis = jnp.linspace(-0.22, 0.22, 7)
mesh_x, mesh_y = jnp.meshgrid(k_axis, k_axis, indexing="xy")
kpoints = jnp.stack(
    (mesh_x, mesh_y, jnp.zeros_like(mesh_x)), axis=-1
).reshape((-1, 3))
kgrid = make_kgrid(kpoints, mesh_shape=(7, 7), kz=0.0)

hamiltonians = bloch_hamiltonian_batch(model, kpoints)
bands = diagonalize_tb(model, kpoints)
```

## Step 2: Matrix-Element Carriers

Photoemission intensity depends on the dipole matrix element between the
initial orbital and the outgoing photoelectron. Four carriers describe it:
the radial wavefunction model, per-shell scales and scattering phases, the
radial quadrature, and the final-state model. See
[Matrix Elements and Polarization](matrix-elements-and-polarization.md) for
the physics of each.

```python
from diffpes.types import (
    make_final_state_spec,
    make_matrix_element_params,
    make_radial_quadrature_spec,
    make_radial_spec,
)

radial_spec = make_radial_spec(
    basis,
    (0,),
    mode="fixed",
    fixed_integrals_shell=jnp.asarray([[0.0, 1.0]]),
)
matrix_element_params = make_matrix_element_params(
    basis,
    (0,),
    sigma_shell=jnp.asarray([1.0]),
    phase_shift_angles_shell=jnp.asarray([0.15]),
)
quadrature = make_radial_quadrature_spec()
final_state = make_final_state_spec()
```

The `fixed` radial mode pins the two dipole-allowed radial integrals
directly, which is the simplest starting point. Use `slater` or
`hydrogenic` modes for photon-energy-dependent radial channels, or `grid`
for tabulated wavefunctions.

## Step 3: Experiment Geometry and Self-Energy

{class}`~diffpes.types.ExperimentGeometry` holds the beamline state: photon
energy, complex polarization, work function, temperature, and escape depth.
The self-energy sets the intrinsic linewidth; `gamma=0.035` is a constant
35 meV imaginary part, and [Spectral Broadening and
Self-Energy](spectral-broadening-and-self-energy.md) covers the
energy-dependent models.

```python
from diffpes.types import make_experiment_geometry, make_self_energy_model

experiment = make_experiment_geometry(
    photon_energy_ev=50.0,
    polarization=jnp.asarray([1.0 + 0.0j, 0.25j, 0.0j]),
    work_function_ev=4.5,
    temperature_k=25.0,
    mean_free_path_ang=8.0,
)
self_energy = make_self_energy_model(gamma=0.035)
```

## Step 4: Detector Calibration and Effects

{class}`~diffpes.types.DetectorCalibration` declares the native analyser
raster: angular `u`/`v` bin edges, recorded-energy bin edges, and
point-spread FWHM values. {class}`~diffpes.types.DetectorEffects` holds the
nuisance state fitted alongside physics parameters: transmission slopes,
background, sensitivity, and exposure.

```python
from diffpes.types import make_detector_calibration, make_detector_effects

calibration = make_detector_calibration(
    u_bin_edges=jnp.linspace(-0.050, 0.050, 9),
    v_bin_edges=jnp.linspace(-0.050, 0.050, 9),
    energy_bin_edges_ev=jnp.linspace(-0.22, 0.06, 15),
    psf_fwhm_u=0.008,
    psf_fwhm_v=0.010,
    psf_fwhm_energy_ev=0.025,
    transmission_reference_domain_ev=jnp.asarray([44.5, 46.0]),
)
detector_effects = make_detector_effects(
    domain_logits=jnp.asarray([0.0]),
    domain_euler_angles_rad=jnp.zeros((1, 3)),
    transmission_raw_slopes=jnp.asarray([-0.65, 0.30]),
    background_coefficients=jnp.asarray([-8.0]),
    sensitivity_coefficients=jnp.asarray([]),
    exposure=2.0e8,
    background_mode="flat",
    sensitivity_mode="constant",
    domain_frame_ids=("org.diffpes.frame.sample_cartesian",),
)
```

## Step 5: Run the Driver

`simulate_arpes` joins everything. It assembles coherent matrix-element
sources and evaluates the occupied spectral function on the requested
energy axis. It then maps the source into detector bins and applies
transmission, resolution, background, sensitivity, exposure, and bin
volume. The result is
a {class}`~diffpes.types.DetectorRaster` of expected counts.

```python
from diffpes.simul import sample_poisson_counts, simulate_arpes
import jax

energy_axis = jnp.linspace(-0.24, 0.08, 33)
detector = simulate_arpes(
    (hamiltonians,),
    (bands,),
    radial_spec,
    matrix_element_params,
    quadrature,
    final_state,
    experiment,
    self_energy,
    kgrid,
    energy_axis,
    calibration,
    detector_effects,
    k_chunk=8,
    energy_chunk=8,
    checkpoint=True,
)
print(detector.expected_counts.shape)   # (1, 8, 8, 14): domain, u, v, E

observed = sample_poisson_counts(
    jax.random.key(0), detector.expected_counts
)
```

```{figure} figures/pipeline-detector-counts.png
:alt: Expected detector counts and one Poisson acquisition

Driver output on the native analyser raster: a momentum-energy slice of the
expected counts, the energy-summed detector image, and one Poisson-sampled
acquisition from the same expected counts.
```

Fit and design against `expected_counts`; call
{func}`~diffpes.simul.sample_poisson_counts` only when you need one
simulated acquisition. Every continuous input above -- hoppings, radial
scales, phases, polarization, self-energy, transmission slopes, exposure --
supports `jax.grad` through the driver. The executed
[tight-binding-to-detector tutorial](../tutorials/coherent-detector-paper-path.md)
runs this same pipeline with plots at every stage.

## Bulk Samples and Photon-Energy Scans

The keyword-only `kz_mode` argument selects how the out-of-plane momentum is
treated:

| Mode | Use when |
|---|---|
| `native_direct` | The Hamiltonian already fixes $k_z$ (2D or slab models). This is the default route above. |
| `bulk_direct` | A bulk model evaluated at the exact free-electron final-state $k_z$; no escape-depth broadening. |
| `bulk_kz` | A bulk model with finite escape depth: the driver integrates a wrapped Lorentzian over one $k_z$ period. |
| `coherent_slab` | A slab model summed coherently over layer depth amplitudes. |

{func}`~diffpes.simul.simulate_hv_scan` evaluates the same source physics
across a photon-energy array and returns a `[n_hv, n_k, n_e]` stack.
{func}`~diffpes.simul.hv_map_at_energy` slices it at fixed binding energy
for $k_\parallel$--$h\nu$ maps. See
[kz Broadening and Photon-Energy Scans](kz-broadening-and-photon-energy-scans.md)
for the bulk carriers and mode rules.

## The Two Intrinsic Spectral Paths

Below the drivers sit two interchangeable spectral primitives, both
consuming the retarded self-energy from
{func}`~diffpes.simul.evaluate_self_energy`:

- {func}`~diffpes.simul.spectral_intensity_resolvent` evaluates
  $-\tfrac{1}{\pi}\operatorname{Im}\sum_\alpha s_\alpha^\dagger
  [(\omega+i\eta-\Sigma)I-H]^{-1}s_\alpha$
  from the Hamiltonian and complex transition sources. It never
  differentiates eigenvectors, so it is the safe path at band degeneracies
  and for Hamiltonian-parameter gradients.
- {func}`~diffpes.simul.spectral_intensity_eigen` consumes eigenvalues and
  gauge-invariant band weights. It is faster away from degeneracies.

The chunk assemblers
{func}`~diffpes.simul.assemble_spectral_intensity_chunk` and
{func}`~diffpes.simul.assemble_spectral_intensity_bands_chunk` wrap these
with sampled-energy Fermi occupation for streaming over large
`(k, omega)` blocks. Use them directly when you want the intrinsic
spectrum before detector effects, as in the
[quickstart](../tutorials/quickstart.md).

```{figure} figures/pipeline-ek-cut.png
:alt: Occupied spectral intensity along the graphene Gamma-K-M-Gamma path
:width: 78%

Occupied intrinsic spectrum along the graphene $\Gamma$-K-M-$\Gamma$ path
from the eigen assembler: the $\pi$ bands broadened by a 60 meV self-energy
and cut off by Fermi occupation.
```

```{figure} figures/pipeline-fermi-map.png
:alt: Graphene Fermi-surface map with six K-point pockets
:width: 62%

The same physics as a momentum-momentum slice: a Fermi-surface map of the
electron-doped cones.
```

## Why Amplitudes, Not Probabilities

diffpes keeps complex transition amplitudes until the outgoing-channel
reduction $\sum_s |M_s|^2$, so orbital, sublattice, atomic-centre,
polarization, and surface-attenuation interference all survive. A VASP
`PROCAR` projection stores only probabilities $|c_{no}(\mathbf{k})|^2$ and
cannot reconstruct those relative phases. Quantitative matrix-element
simulation therefore starts from a phase-complete Hamiltonian, with parsed
projections serving as path and basis metadata. This is the reason
`simulate_arpes` takes Hamiltonians and eigenvector-bearing bands rather
than projection weights.

## Where Each Physical Knob Lives

| Physics | Carrier / argument | Guide |
|---|---|---|
| Band structure | `TBModel`, `DiagonalizedBands`, bulk models | [VASP Data Ingestion](vasp-data-ingestion.md) |
| Photon energy, polarization, temperature | `ExperimentGeometry` | [ARPES Geometry and Kinematics](arpes-geometry-and-kinematics.md) |
| Radial integrals, phases, cross sections | `RadialSpec`, `MatrixElementParams` | [Matrix Elements and Polarization](matrix-elements-and-polarization.md) |
| Linewidths and lineshapes | `SelfEnergyModel` | [Spectral Broadening and Self-Energy](spectral-broadening-and-self-energy.md) |
| Escape depth and $k_z$ integration | `kz_mode`, `SurfaceCell` | [kz Broadening and Photon-Energy Scans](kz-broadening-and-photon-energy-scans.md) |
| Analyser response and counts | `DetectorCalibration`, `DetectorEffects` | [Spectral Broadening and Self-Energy](spectral-broadening-and-self-energy.md) |
| Gradients and fitting coordinates | `pack_matrixel_params` | [JAX Transformability and Gradients](jax-transformability-and-gradients.md) |
