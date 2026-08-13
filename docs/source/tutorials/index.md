# Tutorials

The diffpes tutorials use executable notebooks. Registered tutorials pair a
stripped notebook in this directory with a reviewable Jupytext percent script
in ``tutorials/``. The documentation build executes the notebook cells and
stores outputs only in its cache.

```{toctree}
:maxdepth: 1

quickstart
certified-forward-model
geometry-and-kinematics
coherent-detector-paper-path
bulk-kz-and-photon-energy
tight-binding-models
slabs-and-surfaces
matrix-element-sensitivity
90-laser-window-audit
91-anchor-band-structures
92-orbital-character
93-chgcar-floor-dry-run
94-tb-dirac-cone-spectrum
```

- [Quickstart](quickstart.md): Assemble a coherent intrinsic spectrum through
  the eigen and resolvent paths, map it into native expected counts, then
  differentiate a spectral observable with `jax.grad`.
- [Inspect and persist a certified forward run](certified-forward-model.md):
  Read bounded claims and differentiable evidence. Save canonical JSON and
  attach the same record to an HDF5 result.
- [Geometry and kinematics](geometry-and-kinematics.md): Build k-space rasters,
  detector frames, inner-potential scans, and a geometry Jacobian.
- [Coherent tight-binding model to detector counts](coherent-detector-paper-path.md):
  Build a coherent ARPES cube, inspect its Fermi-surface map, fit analyser
  transmission, and run the canonical native-detector count driver.
- [Bulk kz integration and photon-energy scans](bulk-kz-and-photon-energy.md):
  Distinguish the four out-of-plane modes, inspect wrapped-kz weights, and
  evaluate a compact bulk photon-energy map.
- [Native tight-binding models](tight-binding-models.md): Build graphene by
  hand and with Slater--Koster parameters. Add spin--orbit coupling, then
  inspect fat bands, spin texture, and density of states.
- [Slabs and surfaces](slabs-and-surfaces.md): Build a Miller-index slab,
  verify an analytic finite-chain spectrum, and inspect depth-weighted bands.
- [Matrix-element sensitivity](matrix-element-sensitivity.md): Differentiate
  complete isolated band-group weights through a synthetic dark corridor and
  apply the logarithmic validity mask.
- [Anchor material band structures](91-anchor-band-structures.md): Load local
  DFT bands and compare their paths against one Fermi reference.
- [Laser window audit](90-laser-window-audit.md): Compare low-energy
  photoemission access against local DFT bands and a calibrated Dirac cone.
- [Orbital character maps](92-orbital-character.md): Resolve local DFT bands
  by orbital family, atomic species, and layer.
- [Charge-density floor checks](93-chgcar-floor-dry-run.md): Inspect local
  volumetric densities and surface-sensitive escape-depth weighting.
- [Calibrated Dirac cone spectrum](94-tb-dirac-cone-spectrum.md): Run a
  DFT-calibrated cone through matrix elements, detector response, and one
  Poisson acquisition at 6.05 eV.

The project is developing more complete examples:

- Loading phase-complete electronic-structure inputs for coherent ARPES
- Polarization-dependent matrix element effects
- Gradient-based recovery of band-structure parameters from spectra

Read the [guides](../guides/index.md) for theory and architecture. Read the
[API reference](../api/index.rst) for complete function documentation.
