# Matrix Elements and Polarization

The coherent matrix-element layer preserves complex amplitudes until the
observable boundary. It combines normalized initial-state radial functions,
final-state partial waves, angular dipole couplings, atomic-centre phases,
surface attenuation, complex band coefficients, and laboratory
polarization.

## Input Carriers

`OrbitalBasis` identifies every atom, subshell, real harmonic, and optional
spin block. The radial and final-state factories build validated PyTrees:

```python
from diffpes.types import (
    make_final_state_spec,
    make_matrix_element_params,
    make_radial_quadrature_spec,
    make_radial_spec,
)

radial = make_radial_spec(
    basis=basis,
    radial_shell_index=shell_index,
    mode="slater",
    zeta_shell=zeta_shell,
    coefficients_shell=coefficients_shell,
)
me_params = make_matrix_element_params(
    basis=basis,
    radial_shell_index=shell_index,
    sigma_shell=sigma_shell,
    phase_shift_angles_shell=phase_angles,
)
quadrature = make_radial_quadrature_spec()
final_state = make_final_state_spec(mode="plane_wave")
```

`RadialSpec` supports `slater`, `hydrogenic`, `grid`, and `fixed` modes.
Slater contractions are normalized after combining their primitive rows, so
a common coefficient scale is a gauge direction rather than a physical
amplitude. `fixed` stores calibrated lower/upper channel integrals. A
nonexistent $l-1$ channel for an s shell remains invalid; it does not acquire
a fitted phase.

## Radial Channels

`radial_bvals` evaluates the two dipole-allowed final angular momenta
$l'=l-1$ and $l'=l+1$:

$$
B_{l'}(k)=(-i)^{l'}\int_0^\infty r^3 R_{nl}(r)j_{l'}(kr)\,dr .
$$

The default quadrature is a fixed certified Gauss--Legendre profile. A fixed
node set makes JAX derivatives reproducible. `FinalStateSpec` selects the
plane-wave or Coulomb radial model explicitly.

## Angular and Orbital Channels

`assemble_orbital_transition_channels` joins `radial_bvals` to real Gaunt
couplings and Cartesian spherical harmonics. Its output has shape
`[K, S, O, 3]`, where `S` is outgoing spin and the last axis is the real
dipole-channel order `(y, z, x)`.

For orbital $o$ at position $\mathbf{R}_o$ and depth $d_o$, its amplitude
contains

$$
\sigma_o e^{-d_o/(2\lambda)}
e^{i(\mathbf{k}_i-\mathbf{k}_f)\cdot\mathbf{R}_o}
\sum_{l'=l\pm1} B_{l'}e^{i\delta_{l'}}A_{l'm'}.
$$

The prefactors multiply the partial-wave sum. Every listed factor remains at
amplitude level. Attenuation uses
$e^{-d/(2\lambda)}$ because $\lambda$ is an intensity mean free path.

The function requires:

- an explicit vacuum `k_f_cart` in sample Cartesian inverse Angstrom,
- an explicit `emission_valid` mask,
- the registered zero in-plane reciprocal shift,
- carrier metadata that agree on basis and radial shell mapping.

It never substitutes `kz_from_inner_potential` for the outgoing vacuum
momentum.

## Late Cartesian Polarization

Polarization is a complex Cartesian vector in the laboratory frame.
`contract_experiment_polarization` rotates it into the sample frame and
contracts it only after the three transition channels are complete.
`contract_polarization` performs the same late contraction when the caller
already has sample-frame polarization.

Keeping polarization late preserves arbitrary elliptical phase and avoids
turning a coherent vector relation into three separate intensities.

```python
orbital_channels = diffpes.matrixel.assemble_orbital_transition_channels(
    bands,
    radial,
    me_params,
    quadrature,
    final_state,
    experiment,
    k_f_cart,
    emission_valid,
)
polarized_orbitals = diffpes.simul.contract_experiment_polarization(
    orbital_channels,
    experiment,
)
band_amplitudes = diffpes.matrixel.project_band_channels(
    orbital_channels,
    bands.eigenvectors,
)
polarized_bands = diffpes.simul.contract_experiment_polarization(
    band_amplitudes,
    experiment,
)
intensity = diffpes.matrixel.matrix_element_intensity(polarized_bands)
```

`project_band_channels` follows the stored ket-coefficient convention and
does not conjugate the band coefficients. For spinors, it retains a distinct
outgoing-spin amplitude. `matrix_element_intensity` then computes
$\sum_s |M_s|^2$: outgoing spins are unresolved and incoherent, while all
orbital interference inside a given $M_s$ survives.

```{figure} figures/matrixel-polarization.png
:alt: Graphene path spectra under s and p polarization

Graphene $\pi$-band spectra weighted by this pipeline for s and p
polarization at the same grazing incidence. Polarization selects which
band and which part of the path light up, and sublattice interference
darkens entire corridors.
```

## Atomic Cross Sections

The packaged Yeh--Lindau data support the inexpensive `basic` projection
tier and independent comparisons:

```python
energy, sigma = diffpes.simul.yeh_lindau_cross_section_table(29, 3, 2)
cu_3d = diffpes.simul.yeh_lindau_cross_section(80.0, 29, 3, 2)
weights = diffpes.simul.yeh_lindau_orbital_weights(
    80.0, basis, atomic_numbers
)
```

These isolated-atom cross sections are not a substitute for coherent
solid-state amplitudes. The interpolator stays inside valid tabulated
segments and reports unsupported energies rather than extrapolating.

```{figure} figures/matrixel-cross-sections.png
:alt: Yeh-Lindau cross sections for Cu 3d, C 2p, and Bi 6p
:width: 74%

Packaged Yeh--Lindau photoionization cross sections. Relative subshell
weights shift by orders of magnitude across the photon-energy range, which
is why photon energy is a practical orbital-contrast knob.
```

## Inversion Coordinates and Gauges

`pack_matrixel_params` exposes the active real optimization coordinates;
`unpack_matrixel_params` reconstructs the carriers. The overall final-state
phase and common Slater coefficient scales do not alter intensity.
`matrix_element_phase_gauge_direction` and
`radial_coefficient_scale_gauge_directions` return their normalized packed
tangents so inverse analyses can identify or project these null directions.

Use `band_group_weight_sensitivity` only for complete isolated band groups.
Degenerate subspaces must be grouped as a whole because individual
eigenvectors inside them are gauge dependent. The helper returns
matrix-element weights and `dw/dtheta`, not expected detector counts.
