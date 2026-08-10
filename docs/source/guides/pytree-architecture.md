# PyTree Architecture

diffpes represents physical data with Equinox modules. Numerical leaves are
JAX arrays; structural metadata are static fields. Factory functions validate
both before a carrier reaches a compiled kernel.

## Carrier Families

| Module | Principal carriers | Factories |
|---|---|---|
| `types/bands.py` | `BandStructure`, `OrbitalProjection` | `make_band_structure`, `make_orbital_projection` |
| `types/geometry.py` | `CrystalGeometry` | `make_crystal_geometry` |
| `types/experiment.py` | `ExperimentGeometry` | `make_experiment_geometry` |
| `types/tb_model.py` | `TBModel`, `DiagonalizedBands` | `make_tb_model`, `make_diagonalized_bands` |
| `types/radial_params.py` | `OrbitalBasis`, `RadialSpec`, `MatrixElementParams`, `RadialQuadratureSpec`, `FinalStateSpec` | corresponding `make_*` factories |

`OrbitalBasis` stores atom indices and quantum numbers as static tuples.
Changing a basis can retrace a compiled function. Radial exponents,
contraction coefficients, shell scales, phase angles, experimental geometry,
and band coefficients are numerical leaves and can participate in autodiff.

## Static Structure and Traced Values

Static values choose array topology or code paths:

- radial mode and final-state mode,
- shell-to-orbital mapping,
- orbital quantum numbers and spin-block layout,
- quadrature profile,
- the self-energy and radial model modes,
- complete band-group membership.

Traced values vary without changing topology:

- radial exponents, coefficients, and effective charges,
- matrix-element shell scales and valid phase angles,
- polarization and experiment geometry,
- band energies and eigenvectors,
- detector-calibration widths and temperature.

Use factory functions outside `jax.jit`. They perform ordinary Python checks
for static structure and JAX-compatible checks for numerical leaves.

```python
basis = diffpes.types.make_orbital_basis(
    atom_indices=(0, 0, 0),
    n=(2, 2, 2),
    l=(1, 1, 1),
    m=(-1, 0, 1),
)
radial = diffpes.types.make_radial_spec(
    basis=basis,
    radial_shell_index=(0, 0, 0),
    mode="hydrogenic",
    effective_charge_shell=jnp.array([1.0]),
)
```

## Incoherent and Coherent Data

`OrbitalProjection.projections` stores probabilities with shape
`[K, B, A, 9]`. It remains an ingestion and diagnostic carrier, but no
production spectrum assembler treats those probabilities as recoverable
complex amplitudes.

`DiagonalizedBands.eigenvectors` stores complex basis-position-gauge
coefficients with shape `[K, B, O]`. The coherent matrix-element pipeline
keeps their phase through `project_band_channels`. Do not convert one carrier
into the other unless losing interference is intentional.

## Optimizer Boundaries

`pack_matrixel_params` maps the active numerical leaves to one real vector.
Its metadata and PyTree definition are returned alongside the vector.
`unpack_matrixel_params` restores the carriers while retaining static and
calibrated fields from templates.

The packing deliberately omits grid samples and fixed calibrated radial
channels. It excludes the invalid lower phase of s shells. Gauge helpers
identify overall phase and normalized-contraction scale directions in the
same coordinates.

## Transformations

Pass PyTrees directly through `jax.jit`, `jax.vmap`, `jax.grad`, `jax.jvp`,
and `jax.vjp`. Construct new carriers with factories, or use
`equinox.tree_at` for controlled updates when the static structure stays
unchanged. Avoid mutation: Equinox modules are immutable values.

See [JAX Transformability and Gradients](jax-transformability-and-gradients.md)
for derivative boundaries and
[Matrix Elements and Polarization](matrix-elements-and-polarization.md) for
the coherent carrier flow.
