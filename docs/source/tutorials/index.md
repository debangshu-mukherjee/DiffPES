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
tight-binding-models
slabs-and-surfaces
matrix-element-sensitivity
```

- [Quickstart](quickstart.md): Assemble a coherent intrinsic spectrum through
  the eigen and resolvent paths, then differentiate a spectral observable
  with `jax.grad`.
- [Inspect and persist a certified forward run](certified-forward-model.md):
  Read bounded claims and differentiable evidence. Save canonical JSON and
  attach the same record to an HDF5 result.
- [Geometry and kinematics](geometry-and-kinematics.ipynb): Build k-space rasters,
  detector frames, inner-potential scans, and a geometry Jacobian.
- [Native tight-binding models](tight-binding-models.md): Build graphene by
  hand and with Slater--Koster parameters. Add spin--orbit coupling, then
  inspect fat bands, spin texture, and density of states.
- [Slabs and surfaces](slabs-and-surfaces.md): Build a Miller-index slab,
  verify an analytic finite-chain spectrum, and inspect depth-weighted bands.
- [Matrix-element sensitivity](matrix-element-sensitivity.md): Differentiate
  complete isolated band-group weights through a synthetic dark corridor and
  apply the logarithmic validity mask.

The project is developing more complete examples:

- Loading phase-complete electronic-structure inputs for coherent ARPES
- Applying the Plan 08a detector/count driver once that chain is certified
- Polarization-dependent matrix element effects
- Gradient-based recovery of band-structure parameters from spectra

Read the [guides](../guides/index.md) for theory and architecture. Read the
[API reference](../api/index.rst) for complete function documentation.
