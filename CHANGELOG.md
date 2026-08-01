# Changelog

This file documents all notable changes to diffpes.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and the project uses calendar versioning.

## [Unreleased]

### Removed

- The public `diffpes.types.N_TAYLOR` implementation detail is removed.
  `diffpes.utils.faddeeva` now uses a certified fixed-order rational method.
- The Thompson--Cox--Hastings pseudo-Voigt approximation is removed with
  its empirical mixing constants without retaining a compatibility shim.
- The kinematics and geometry update removes `diffpes.types.PolarizationConfig`,
  `diffpes.types.make_polarization_config`, and
  `diffpes.simul.build_efield`. Construct explicit complex Cartesian fields
  with `diffpes.simul.polarization_from_angles` and store experiment geometry
  with `diffpes.types.ExperimentGeometry`.
- `SimulationParams` no longer stores `temperature` or `photon_energy`.
  `ExperimentGeometry.temperature_k` and
  `ExperimentGeometry.photon_energy_ev` own those experiment properties.
  The retained incoherent spectrum functions accept the scalars explicitly
  at their physics boundaries.
- The obsolete coherent prototype is removed:
  `diffpes.simul.simulate_tb_radial` and its `simul.forward` module. Use the
  matrix-element channel and contraction APIs in `diffpes.simul.matrixel`.
- The heuristic polarization symbols
  `diffpes.simul.dipole_matrix_elements`,
  `diffpes.types.ORBITAL_DIRS_NORMALIZED`, and the private
  `types.constants._ORBITAL_DIRS` direction table are removed. Public
  matrix-element APIs now accept canonical complex Cartesian polarization.
- The legacy scalar dipole helpers
  `diffpes.maths.dipole_matrix_element_single`,
  `diffpes.maths.dipole_intensity_orbital`, and
  `diffpes.maths.dipole_intensities_all_orbitals` are removed.
- The projection levels `diffpes.simul.simulate_basicplus`,
  `diffpes.simul.simulate_advanced`, `diffpes.simul.simulate_expert`, and
  `diffpes.simul.simulate_soc` are removed together with
  `simulate_basicplus_expanded`, `simulate_advanced_expanded`,
  `simulate_expert_expanded`, and `simulate_soc_expanded`. The expanded
  dispatcher now accepts only `novice` and `basic`.
- `diffpes.simul.heuristic_weights`, `diffpes.simul.yeh_lindau_weights`, and
  the toy `CROSS_SECTION_ENERGIES`, `CROSS_SECTION_SIGMA_S`,
  `CROSS_SECTION_SIGMA_P`, and `CROSS_SECTION_SIGMA_D` tables are removed.
  The retained basic tier requires explicit `OrbitalBasis` and atomic numbers
  and consumes the authenticated element/subshell Yeh--Lindau tables.
- `diffpes.types.SlaterParams` and `diffpes.types.make_slater_params` are
  removed. Use the shell-shared `RadialSpec` carrier and
  `make_radial_spec`.

### Changed

- Two pinned `generator_sha256` values are re-pinned:
  `tests/test_diffpes/test_radial/data/coulomb_mpmath_80digit.manifest.json` and
  `src/diffpes/simul/data/yeh_lindau_1985.json`. Their generator scripts changed
  through documentation edits and NumPy type annotations. No generator logic
  changed. Every reference archive stays byte-identical, and each
  `archive_sha256` still validates against unchanged data. The re-pin records a
  documentation change to the generator, not new scientific evidence.
- Certification owners and evidence identifiers now use scientific domain names.
  This breaking identity re-issue invalidates records that use the former identifiers.
- Every NumPy array annotation now carries a jaxtyping dtype and shape, in the
  form `Float[NDArray, "m n p"]`. The source imports `NDArray` from
  `numpy.typing`. The previous `from numpy import ndarray as NDArray` alias and
  its `# noqa: N812` suppression are removed. A bare `np.ndarray` annotation and
  a bare `NDArray` annotation are now defects. `CONTRIBUTING.md` states the rule,
  and `tests/test_repo_floor.py` enforces it across the source and the test tree.
- `SelfEnergyConfig` is renamed to `SelfEnergyModel`, and
  `make_self_energy_config` to `make_self_energy_model`. The mode value
  `"polynomial"` becomes `"poly"`, and `"tabulated"` becomes `"grid"`.
  The carrier gains `kk_domain_rel_fermi_ev`, `tail_coefficients`,
  `subtraction_point_rel_fermi_ev`, `kk_consistent`, and `tail_mode`, and
  renames `energy_nodes` to `energy_nodes_rel_fermi_ev`. Energy-dependent
  modes require a KK domain and an explicit tail contract.
- **Silent numerical break in `coefficients`.** `coefficients` are now
  unconstrained real coordinates mapped through `softplus`, not linewidths in
  eV. `make_self_energy_model(coefficients=[0.1], mode="constant")` gives
  `Gamma = softplus(0.1) = 0.744 eV`, where
  `make_self_energy_config(coefficients=[0.1])` gave `Gamma = 0.1 eV`. This is
  a factor of 7.4 and it raises no error. The same applies to any
  `SelfEnergyConfig` written to HDF5 before this release: the file still loads
  and its coefficients now mean something different. To migrate, either pass
  the linewidth through the `gamma` shortcut, which stores
  `softplus_inverse(gamma)` and reproduces the previous value exactly, or
  convert stored coefficients with `log(expm1(gamma))`. The reparameterization
  keeps the imaginary self-energy strictly negative through a smooth,
  gradient-alive map. Clipping would instead zero the gradient and the Fisher
  row at the bound.
- `diffpes.utils.faddeeva` now covers its declared upper-half-plane
  `abs(z) <= 1e8` envelope with an order-40 Weideman rational approximation.
  Invalid or lower-half-plane inputs raise instead of returning divergent
  Taylor-polynomial values.
- `diffpes.simul.voigt` now evaluates the normalized true Voigt convolution.
  Positive widths share the certified Faddeeva `abs(z) <= 1e8` envelope;
  exact Gaussian and Cauchy endpoints are value-only. The migration changes
  core values modestly but can produce much larger relative changes in tails,
  so no uniform percentage-shift claim is made.
- `simulate_novice` and `simulate_basic` are explicitly documented as
  incoherent projection tiers. `simulate_basic` now accepts `basis` and
  `atomic_numbers` and applies one probability-level orbital reduction with
  element- and subshell-resolved Yeh--Lindau weights.
- Tight-binding models and diagonalized bands now carry optional differentiable
  per-orbital surface depths in Angstrom. Native diagonalization and HDF5
  persistence preserve the carrier exactly; ``None`` retains bulk semantics.
- The native tight-binding core is complete. Bloch assembly now uses
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
- `CrystalGeometry` now follows the field contract. It uses
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

- The pre-commit Ruff hooks cover source, tests, and project metadata.
  The continuous integration workflow now supports manual verification.
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

- The package adds shell-shared `RadialSpec`, `MatrixElementParams`,
  `RadialQuadratureSpec`, and `FinalStateSpec` carriers. New radial APIs cover
  Slater screening, normalized Slater/hydrogenic/grid/fixed rows, hardened
  spherical Bessel functions, certified direct quadrature, and regular
  Coulomb final states. The optional Hermite accelerator rejects because its
  frozen refinement evidence does not certify any selectable default.
- Coherent transition-channel assembly now has explicit Wannier
  centres, vacuum final momentum, and outgoing-spin rows. Escape-depth
  attenuation precedes late Cartesian polarization and one final incoherent
  spin reduction. The update also adds stacked-real parameter packing and named
  phase and radial-scale gauge tangents. Complete isolated band-group
  sensitivities include dark-point log masks.
- Authenticated Yeh--Lindau element/subshell cross sections now come from
  the exact Figshare v3 workbook. The package preserves table gaps and zeros,
  uses log--log PCHIP interpolation, rejects extrapolation, and ships source
  provenance with the generated data.
- Independent Coulomb, length--momentum gauge, Chinook
  dark-corridor/polarization, radial-profile, dense-resolvent, and scalability
  evidence are added. The certification registry exposes the complete radial,
  matrix-element, differentiation, and scalability handshake.
- The package adds exact primitive Miller-index surface cells and complete-shell
  Cartesian/orbital rotations. It also adds finite depth-tagged slabs, exact
  bulk-to-slab hopping propagation, and open-normal adjacency validation.
  ``SurfaceCell`` and ``SlabSpec`` record the static construction provenance.
- Surface probability operators now provide raw off-degeneracy band weights
  and complement-isolated fixed-group traces. A separate slab seam propagates
  explicit Wannier centres and position-operator matrices without silently
  discarding tight-binding metadata.
- The package adds `ExperimentGeometry`, generated `KPath`, and fixed-shape
  `KGrid` carriers. Their factories keep numerical geometry inside JAX.
- The package adds independently derived s/p/d Slater--Koster construction and
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
- A release-tag-triggered, uv-native PyPI Trusted Publishing workflow now tests wheels and
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

- Kinematics and geometry evidence now exercises reciprocal identities with generated
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
