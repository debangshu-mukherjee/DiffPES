# Tutorials

The numbered notebooks are an approximation ladder: every one puts an ARPES
calculation on screen, then adds a missing layer of physics. They are not one
material calculation carried unchanged through five files. Tutorials 1, 2, and
5 use compact tight-binding examples; 3 and 4 use independent VASP examples.

## Approximation Ladder

| Tutorial | Adds to the working calculation | Deliberately still assumed |
| --- | --- | --- |
| 1. [Minimal intrinsic spectrum](01-intrinsic/01-simulate-an-arpes-spectrum.md) | Band dispersion, Fermi occupation, and a fixed lifetime on one momentum cut | Two-orbital graphene, uniform band visibility, no photoemission geometry or instrument |
| 2. [Intrinsic ARPES cube](01-intrinsic/02-explore-an-arpes-cube.md) | Both in-plane momentum dimensions and source-cube views | The same compact, uniform-weight source; fixed linewidth; no beamline or detector |
| 3. [VASP bands to a map](02-material-proxies/03-vasp-bands-to-arpes.md) | Material-specific VASP eigenvalues on a physical path | Uniform weights and constant linewidth; no phases, matrix elements, or detector |
| 4. [PROCAR orbital contrast](02-material-proxies/04-orbital-resolved-arpes.md) | Orbital-population-dependent spectral weights | Projection magnitudes are not coherent dipole cross sections; no instrument response |
| 5. [Coherent detector acquisition](03-coherent-detector/05-detector-arpes-acquisition.md) | Phase-complete tight-binding input, matrix elements, beamline geometry, detector response, and counting noise | The Hamiltonian, radial and final-state choices, and detector calibration must be supplied or calibrated |

All of these quick examples are two-dimensional. A bulk `kz`-dependent
calculation needs an appropriate material model and momentum sampling.

```{toctree}
:maxdepth: 1

01-intrinsic/01-simulate-an-arpes-spectrum
01-intrinsic/02-explore-an-arpes-cube
02-material-proxies/03-vasp-bands-to-arpes
02-material-proxies/04-orbital-resolved-arpes
03-coherent-detector/05-detector-arpes-acquisition
01-intrinsic/tight-binding-models
02-material-proxies/slabs-and-surfaces
03-coherent-detector/matrix-element-sensitivity
01-intrinsic/quickstart
03-coherent-detector/certified-forward-model
```

## Follow the Ladder

1. [Simulate an ARPES spectrum](01-intrinsic/01-simulate-an-arpes-spectrum.md): start with
   an occupied high-resolution energy-momentum image, EDCs, MDCs, and one
   explicit linewidth control.
2. [Explore an ARPES cube](01-intrinsic/02-explore-an-arpes-cube.md): retain the intrinsic
   source approximation while adding a transparent `I(kx, ky, E)` volume,
   orthogonal cuts, constant-energy maps, and energy windows.
3. [Use VASP bands](02-material-proxies/03-vasp-bands-to-arpes.md): replace the toy dispersion
   with a line-mode DFT calculation and make an intrinsic ARPES-style map.
4. [Add PROCAR weights](02-material-proxies/04-orbital-resolved-arpes.md): reveal orbital contrast
   while keeping the boundary between projection weights and coherent matrix
   elements explicit.
5. [Simulate detector counts](03-coherent-detector/05-detector-arpes-acquisition.md): begin with a
   phase-complete tight-binding or Wannier input. Then add matrix elements,
   native detector bins, and Poisson noise.

## Focused Workflows

- [Native tight-binding models](01-intrinsic/tight-binding-models.md): build multi-orbital
  models with Slater--Koster parameters and spin--orbit coupling, or import a
  phase-complete Wannier90 model for Tutorial 5.
- [Slabs and surfaces](02-material-proxies/slabs-and-surfaces.md): construct surface models and
  inspect depth-weighted bands.
- [Matrix-element sensitivity](03-coherent-detector/matrix-element-sensitivity.md): differentiate
  complete isolated band-group weights through polarization-dependent contrast.
- [Quickstart](01-intrinsic/quickstart.md): compare the two intrinsic spectral paths and
  their differentiation behavior.
- [Inspect and persist a certified forward run](03-coherent-detector/certified-forward-model.md):
  store bounded claims and differentiable evidence with a result.

Read the [guides](../guides/index.md) for theory and API choices, and the
[API reference](../api/index.rst) for complete function documentation.
