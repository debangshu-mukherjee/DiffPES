# Changelog

This file documents all notable changes to diffpes.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and the project uses calendar versioning.

## [Unreleased]

### Changed

- Tight-binding models and diagonalized bands now carry optional differentiable
  per-orbital surface depths in Angstrom. Native diagonalization and HDF5
  persistence preserve the carrier exactly; ``None`` retains bulk semantics.
- Plan 04 completes the native tight-binding core. Bloch assembly now uses
  exact integer hopping cells and the basis-position gauge, with complex
  hoppings, traced onsite/SOC parameters, atom-resolved geometry, and a
  degeneracy-regularized eigensystem. This migration intentionally repins the
  graphene radial-gradient and deterministic tight-binding references.
- Wannier ingestion now keeps explicit fractional positions for every
  orbital through Hamiltonian assembly and diagonalization. Noncoincident
  centres assigned to one atom require separately supplied atomic geometry
  instead of being silently collapsed into one position.
- The real-harmonic convention now fixes positive ``m=1`` to ``+p_x`` and
  keeps Gaunt transformations consistent with that sign.
- `CrystalGeometry` now follows the roadmap field contract. It uses
  `lattice`, `reciprocal`, `positions`, and static per-atom `species`.
  `read_poscar` expands VASP species counts at the parser boundary.
- The package merges `orbital_constants` and `vasp_constants` into
  `diffpes.types.constants`. The package removes both old modules without
  compatibility shims. Cross-subpackage constants are now public and omit
  their leading underscores. Examples include `_EPS` to `EPS` and
  `_N_ORBITALS` to `N_ORBITALS`. Another example is `_PHASE_LOSS_MESSAGE` to
  `PHASE_LOSS_MESSAGE`. `diffpes.types` re-exports these constants. Only
  module-internal intermediate values remain private. The constants module
  now imports JAX because orbital direction tables are device arrays.
- The project adopts the generalized import rule from CONTRIBUTING.
  Cross-subpackage imports use the source subpackage's public surface.
  They do not import a file inside that subpackage. The update fixes the deep
  `diffpes.inout` imports in `simul/workflow.py`.

- Plan 01 scopes the pre-commit Ruff hooks to source, tests, and project
  metadata. The continuous integration workflow now supports manual gate
  verification.
- Every registered carrier now uses a types-owned `equinox.Module` instead of
  a `NamedTuple`. All carrier factories now belong to `diffpes.types`.
  HDF5 serialization now introspects array fields. It also handles nested
  modules, optional fields, and static fields. Carrier construction now
  requires keywords.
  Use `equinox.tree_at` for immutable updates instead of
  `NamedTuple._replace`.
- `diffpes.types` now owns the declarative constants, orbital conventions,
  parser schemas, and lookup tables.
- `diffpes.types` now owns the workflow context and its projection and DOS
  aliases. The context PyTree now uses an Equinox module.
- Repository links, documentation, and release surfaces now use lowercase
  `diffpes`.
- The two-tier factory validation system is now active. Structural violations
  raise `ValueError`. Traced value violations use value-threaded
  `equinox.error_if` checks that survive JIT compilation.
- `read_eigenval(..., fermi_energy=...)` now accepts `ScalarFloat`.
  Workflow Fermi energies remain traced scalar leaves instead of host floats.

### Added

- Plan 05 adds exact primitive Miller-index surface cells and complete-shell
  Cartesian/orbital rotations. It also adds finite depth-tagged slabs, exact
  bulk-to-slab hopping propagation, and open-normal adjacency validation.
  ``SurfaceCell`` and ``SlabSpec`` record the static construction provenance.
- Surface probability operators now provide raw off-degeneracy band weights
  and complement-isolated fixed-group traces. A separate slab seam propagates
  explicit Wannier centres and position-operator matrices without silently
  discarding Plan 04 metadata.
- Plan 03 adds `ExperimentGeometry`, generated `KPath`, and fixed-shape
  `KGrid` carriers. Their factories keep numerical geometry inside JAX.
- Plan 04 adds independently derived s/p/d Slater--Koster construction and
  neighbor-shell discovery. It adds spin doubling and atomic L·S coupling.
  Fixed-group observables, fat bands, Gaussian DOS, Fermi levels, and
  flat-real inversion views complete the layer.
- Distinct strict parsers now ingest explicit hopping lists and normative
  Wannier90 `_hr.dat` and `_tb.dat` files. A typed `WannierOperatorData`
  sidecar preserves required centres and optional position matrices. It keeps
  exact cells, degeneracies, source grammar, and normalized spin layout
  through HDF5 round trips.
- Frozen offline Chinook eigenvalue artifacts cover graphene, square-lattice
  Rashba, and projected-t2g SOC compatibility without importing Chinook at
  runtime or in tests. Analytic spectra, symmetry laws, normative formats,
  and independent calculations remain the correctness authorities.
- The tight-binding layer now builds labeled paths, first-zone masks, fixed
  ARPES rasters, and photon-energy rasters. It uses one explicit conversion
  between fractional and Cartesian momentum. First-zone masks use a static
  shell with a conservative completeness proof. They raise an error when the
  selected shell or reciprocal basis cannot certify the requested geometry.
- The simulation layer now provides free-electron final-state kinematics and
  complex inner-potential momentum. It also provides invertible detector-angle
  maps for both slit conventions.
- The polarization layer now constructs explicit complex states and converts
  them to spherical components. It maps a fixed laboratory photon field into
  the sample frame independently of detector pixels. A separate detector-axis
  composition and shared Rodrigues primitive rotate detector-fixed real frame
  vectors.
- JAX-native certified forward execution is now a defining capability.
  It provides typed certificate PyTrees and deterministic registries for models
  and transformations. It also provides provenance graphs, information-loss
  graphs, JAXPR dependency maps, and reusable JVP/VJP evidence. Other features
  include matrix-free information spectra, cumulative assurance policies, and
  compiled domain checks.
- The package now provides an explicitly registered radial ARPES certification
  surface. It supports portable canonical JSON and HDF5 certificate storage.
  It also supports offline inspection, verification, and user and API
  documentation. Domain-separated SHA-256 supplies scientific content
  identities; the certificate document's separate CRC32 detects transport
  corruption only. Neither provides authenticity or physical assurance.
- A tag-gated, uv-native PyPI Trusted Publishing workflow now tests wheels and
  source distributions.
- Equinox, Optimistix, Lineax, and Optax now form the differentiable software
  stack. They provide types, nonlinear solvers, linear solvers, and optimizers.
  The project adopted this stack on 2026-07-13.
- The test environment now includes Hypothesis for property-based verification.
  It also includes psutil for memory guards during execution.
- The shared pytest foundation enforces x64 and deterministic random keys.
  It also cleans JAX caches, limits RSS leaks, and groups xdist tests by memory.
- The test suite now provides typed deterministic toy factories and strict
  numerical tree assertions. It also provides an NPZ reference comparison
  scaffold.
- The program-wide gradient harness now checks scaled finite differences,
  Wirtinger derivatives, and unexpected zero gradients.
- GitHub Actions now tests Python 3.12 through 3.14 and uploads informational
  Codecov reports. Lock-aligned Ruff and ty hooks run before each commit.
  Install the hooks with `uv run pre-commit install`.
- Deterministic regression references now preserve pre-refactor novice and
  tight-binding radial results. They include established zeta-gradient
  baselines and provenance.
- Seven named gradient-safe mathematical primitives now define values and
  subgradients on their guarded sets.
- `pack_complex` and `unpack_complex` now define the boundary between real
  optimizer PyTrees and complex physics. A JAX test pins the Wirtinger
  convention.

### Removed

- The project removes the unused `difftb` dependency and its broken editable
  `[tool.uv.sources]` path. diffpes now installs as a standalone package.
- Legacy real-only tight-binding storage, source-package analytic fixtures,
  silent Hamiltonian Hermitianization, and obsolete projection accessors are
  removed without compatibility shims.
- The development environment no longer includes Black, isort, jupyter-black,
  build, or Twine. Ruff formats the code. uv builds and publishes the package.

### Fixed

- Plan 03 evidence now exercises reciprocal identities with generated
  lattices. It covers the full photon-energy raster memory target.
  Complex polarization checks include phase, gradient, and
  machine-precision complex-step evidence.
- Python 3.14 imports now work while beartype 0.22.9 references the removed
  `collections.abc.ByteString` name.
- The supported Python range is now `>=3.12,<3.15`. The documentation now
  states support for Python 3.12.
- JAX project metadata is now platform-independent. The project removes unused
  setuptools configuration and aligns interrogate with NumPy docstrings.
  Ruff and runtime type checks now cover the test suite.
- The real-to-complex Gaunt transformation coefficients now satisfy their
  complex-valued runtime type contract.
- A stable sigmoid replaces the overflow-prone reciprocal-exponential
  Fermi-Dirac expression. Values and gradients remain finite across the
  realistic-spectrum audit range.
- The Thompson-Cox-Hastings pseudo-Voigt implementation now has defined
  gradients on both positive-width boundary rays. It rejects the undefined
  zero-width intersection before and during JIT execution.

## [2026.03.01] - 2026-07-13

### Added

- The initial release establishes the differentiable ARPES package.

[unreleased]: https://github.com/debangshu-mukherjee/diffpes/compare/v2026.03.01...HEAD
[2026.03.01]: https://github.com/debangshu-mukherjee/diffpes/releases/tag/v2026.03.01
