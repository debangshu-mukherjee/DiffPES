# Tutorials

The diffpes tutorials use MyST text notebooks. These Markdown files contain
executable code cells. The documentation build executes the cells and shows
their outputs. Git stores the reviewable plain-text sources.

```{toctree}
:maxdepth: 1

quickstart
certified-forward-model
geometry-and-kinematics
tight-binding-models
```

- [Quickstart](quickstart.md): Build a synthetic band structure. Simulate two
  ARPES fidelity levels and differentiate the spectrometer model with
  `jax.grad`.
- [Inspect and persist a certified forward run](certified-forward-model.md):
  Read bounded claims and differentiable evidence. Save canonical JSON and
  attach the same record to an HDF5 result.
- [Geometry and kinematics](geometry-and-kinematics.md): Build k-space rasters,
  detector frames, inner-potential scans, and a geometry Jacobian.
- [Native tight-binding models](tight-binding-models.md): Build graphene by
  hand and with Slater--Koster parameters, add atomic and Rashba spin--orbit
  coupling, and inspect fat bands, spin texture, and the density of states.

The project is developing more complete examples:

- Loading VASP output and simulating a basic ARPES spectrum
- Stepping through the six simulation fidelity levels
- Polarization-dependent matrix element effects
- Gradient-based recovery of band-structure parameters from spectra

Read the [guides](../guides/index.md) for theory and architecture. Read the
[API reference](../api/index.rst) for complete function documentation.
