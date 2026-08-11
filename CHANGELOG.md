# Changelog

This file documents all notable changes to diffpes.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and the project uses calendar versioning.

## [Unreleased]

### Removed

- The interim real-linewidth evaluator module `simul/self_energy.py` is
  removed together with its interim tests. `diffpes.simul.evaluate_self_energy`
  now lives in `diffpes.simul.spectral` and returns the complex retarded
  self-energy. A caller that needs an imaginary linewidth takes `-sigma.imag`
  from the complex result.
- The public `diffpes.types.N_TAYLOR` implementation detail is removed.
  `diffpes.utils.faddeeva` now uses a certified fixed-order rational method.
- The Thompson--Cox--Hastings pseudo-Voigt approximation is removed with
  its empirical mixing constants without retaining a compatibility shim.
- The kinematics and geometry update removes `diffpes.types.PolarizationConfig`,
  `diffpes.types.make_polarization_config`, and
  `diffpes.simul.build_efield`. Construct explicit complex Cartesian fields
  with `diffpes.simul.polarization_from_angles` and store experiment geometry
  with `diffpes.types.ExperimentGeometry`.
- `SimulationParams`, `make_simulation_params`,
  `make_expanded_simulation_params`, and `types.params` are removed. Callers
  now own the sampled energy axis explicitly; lifetime structure belongs to
  `SelfEnergyModel`, and native resolution belongs to `DetectorCalibration`.
- `ExperimentGeometry.energy_resolution_ev` and
  `ExperimentGeometry.momentum_resolution_inv_ang` are removed so
  `DetectorCalibration` is the sole authority for native detector PSF
  widths.
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
- All projection-probability spectrum tiers and the level-string dispatcher
  are removed without compatibility shims. This includes
  `diffpes.simul.simulate_novice`, `diffpes.simul.simulate_basic`, every
  expanded wrapper, `diffpes.simul.simulate_expanded`, the former
  `simul.spectrum` implementation, and the `simul.expanded` module. The
  rebuilt `simul.spectrum` module retains the canonical `simulate_arpes` and
  `simulate_arpes_cut` drivers and adds the physically separate Plan-08b
  `simulate_hv_scan`; none of the removed tier dispatch survives.
- `diffpes.simul.simulate_context` is removed. The old implicit-H,
  projection-probability `run_vasp_workflow` signature is also removed. Its
  rebuilt coherent API requires an explicit Hamiltonian. It also requires
  every physical carrier named by the canonical detector/count driver.
- The tests and expanded-wrapper guide dedicated only to the removed tiers
  are deleted. The frozen true- and pseudo-Voigt novice archives remain as
  non-live historical provenance and are checked for integrity, not replayed
  through production code.
- `diffpes.simul.heuristic_weights`, `diffpes.simul.yeh_lindau_weights`, and
  the toy `CROSS_SECTION_ENERGIES`, `CROSS_SECTION_SIGMA_S`,
  `CROSS_SECTION_SIGMA_P`, and `CROSS_SECTION_SIGMA_D` tables are removed.
  The authenticated element/subshell Yeh--Lindau tables remain available as
  explicit probability-level diagnostics; they are not a spectrum assembler.
- `diffpes.types.SlaterParams` and `diffpes.types.make_slater_params` are
  removed. Use the shell-shared `RadialSpec` carrier and
  `make_radial_spec`.

### Changed

- `simulate_arpes` and `simulate_arpes_cut` retain the complete Plan 08a
  positional surface and add only keyword-only `bulk_models_by_domain`,
  `surface_cells_by_domain`, `kz_nodes_frac`, and `kz_mode` arguments. The
  default `native_direct` route reproduces the single-kz call. The mutually
  exclusive `bulk_direct`, `bulk_kz`, and `coherent_slab` routes reject mixed
  carriers instead of guessing which escape-depth model the caller intended.
- **Breaking:** `run_vasp_workflow` now uses VASP files only for parsed path,
  Fermi-level, crystal, and projection metadata. Callers must supply the
  phase-complete Hamiltonian plus all radial, matrix-element, self-energy,
  geometry, calibration, and detector-effect carriers. PROCAR weights never
  become a hidden coherent Hamiltonian or inversion coordinate.

- **Breaking:** `ArpesSpectrum` now requires cumulative Cartesian path
  distance, every Cartesian path vector, and the registered sample-frame ID.
  The HDF5 loader rejects retired two-field spectrum files with an actionable
  schema error because it cannot reconstruct their missing geometry.
- `DetectorRaster` now rejects a zero-length channel axis in eager and
  compiled construction. Every expected-count raster contains at least one
  explicitly labeled acquisition channel.
- The geometry-and-kinematics tutorial now uses a stripped Jupyter notebook
  paired with a reviewable Jupytext percent script. Documentation CI verifies
  pair synchronization and committed-output absence, reuses a content-keyed
  execution cache, and fails immediately on an unexpected notebook error.
- **Breaking:** `make_self_energy_model` now rejects a `gamma` shortcut
  supplied together with explicit `coefficients` instead of silently ignoring
  `gamma`. The `gamma` default becomes `None`; an absent `gamma` with absent
  `coefficients` still constructs the `0.1` eV constant carrier.
- Three authenticated reference artifacts are re-issued:
  `chinook_tightb_reference.json`, `chinook_slab_reference.json`, and
  `wannier90_wse2_reference.json`. Each carried one plan-named metadata key.
  The key becomes `requirement` or `requirements` with a descriptive value, and
  each pinned artifact digest is recomputed. A guard confirmed that every field
  outside the metadata block stays identical, so all compared numeric data is
  unchanged. The artifacts now differ from the raw external generator output by
  that metadata key alone.
- Three pinned `generator_sha256` values are re-pinned:
  `tests/test_diffpes/test_radial/data/coulomb_mpmath_80digit.manifest.json` and
  `src/diffpes/simul/data/yeh_lindau_1985.json`, plus
  `tests/test_diffpes/_reference_data/voigt_scipy_manifest.json`. Their
  generator scripts changed through documentation edits and NumPy type
  annotations. No generator logic changed. Every reference archive stays
  byte-identical, and each `archive_sha256` still validates against unchanged
  data. The re-pin records a documentation change to the generator, not new
  scientific evidence.
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

- Plan 08b adds an explicit differentiable bulk-kz integral and
  photon-energy scans. `kz_fractional_nodes`,
  `kz_wrapped_lorentzian_bin_weights`, and `broaden_kz` implement positive
  analytic wrapped-Cauchy bin masses over one primitive surface reciprocal
  period, centred by exact finite-energy inner-potential kinematics. The G6
  calibration selects `n_kz=2048` as the smallest registered count meeting
  its value, integrated-count, gradient, and reference-series budgets; there
  is no silent public count default. A normal integration-coordinate
  reciprocal shift leaves the complete gauge-covariant integrand invariant
  at fixed detected $k_\parallel$ and $k_f$. A move to a neighboring detected
  surface zone changes those momenta. It retains physical repeated-zone
  matrix-element contrast.
- `simulate_hv_scan` returns a single-domain pre-detector
  `[n_hv, n_k, n_e]` stack through a checkpointable `jax.lax.scan`, and
  `hv_map_at_energy` returns an interpolated `[n_k, n_hv]` map. Production
  integration remains node-local and forbids a complete all-node band,
  source, kinematics, or intensity carrier.
- The authenticated Plan-08b literal scalability record
  `4f83fd4f85974ff7065e04846f48003960b1ddbe3ced3db537d7fa92b5caa3c4`
  compiles the exact `256 x 256 x 400`, 20-band, 2048-node target. It records
  1,074,870,048-byte forward and 2,567,802,048-byte full-H-gradient live
  allocations. It also records zero forbidden all-node carriers,
  rematerialization equality, and flat photon-scan auxiliary allocation. The
  source-handshake refresh rebinds the spectral and detector records as
  `08a917ff8dabbcfb78858c4a3b5f3a408834df36a6b55336b2a0f7ed04a9e5cd`
  and `afb70466c0468b616bb66b36b4c6cf23f539116f98ccbe1e5c6a1ad30ee65760`.
  All registered budgets and companions remain green.
- The certification registry adds the `org.diffpes.kz` owner and immutable
  wrapped-integration/photon-energy-scan transformations. Registration
  requires exact `org.diffpes.kspace`, `org.diffpes.surface`,
  `org.diffpes.matrixel`, `org.diffpes.spectral`, and
  `org.diffpes.detector` upstream handshakes; missing or drifted declarations
  fail closed.
- Plan 08a adds one coherent single-kz forward surface. `simulate_arpes`
  accepts separable Cartesian source rasters. `simulate_arpes_cut` accepts
  self-describing momentum paths. The rebuilt `run_vasp_workflow` requires an
  explicit Hamiltonian. The drivers stream block-local Plan-06
  transition sources through the degeneracy-safe Plan-07 resolvent, then call
  one shared detector chain. `map_source_to_detector` performs conservative
  native-bin density mapping before detector-space domain mixing.
  `apply_detector_effects` applies fixed-domain analyser transmission,
  finite-volume native-coordinate resolution, nonnegative backgrounds,
  normalized sensitivity, exposure, explicit bin volumes, and an optional
  calibrated post-count response. Poisson and fixed-total acquisition remain
  explicit-key operations outside the differentiable expected-rate graph.
  Boundary-intersecting cube maps are supported only when the projected
  rotation is signed diagonal or antidiagonal. General rotations require the
  complete inverse detector target to lie strictly inside the source support.
  A path cut is a slit-integrated line density with one declared transverse
  aperture, not an inferred two-dimensional source density.
  The frozen RM-2 Chinook comparison is K-only response compatibility. It
  replays a test-only matched Gaussian adapter on a common authenticated raw
  cut and makes no production-driver, conservation, or absolute-scale claim.

- The coherent intrinsic spectral seam now joins matrix-element sources and
  causal self-energy models. `spectral_intensity_resolvent` uses a complex128
  Lineax solve and remains differentiable at exact degeneracies. Its source
  contract is `[n_out, n_orb]` with a mandatory nonempty outgoing-channel
  axis. Each right-hand side receives an independent solve before the
  real-valued reduction. The chunk assembler accepts
  `[n_k, n_omega, n_out, n_orb]`. The streamed path constructs only
  block-local transition sources. It never materializes a full
  momentum-by-energy-by-basis source carrier.
  `projected_spectral_density_resolvent` preserves Hermitian channel
  coherences. `spectral_intensity_eigen` provides the nondegenerate
  gauge-invariant band-weight path. The two chunk assemblers subtract the
  Fermi energy exactly once and apply the Fermi distribution at sampled
  relative energy. Instrument response, normalization, backgrounds, and
  counts remain downstream operations.
- `diffpes.simul.spectral` adds the complex retarded self-energy evaluation
  `evaluate_self_energy` with the certified cell-integrated principal-value
  Kramers--Kronig operator. Grid mode uses the exact hat transform; the
  smooth modes use the piecewise-cubic transform with C1 `power2` tails and
  256-node semi-infinite quadratures. Queries outside the trusted interval
  `[a + 2h, b - 2h]` raise eagerly and under `jit`. Public `jax.jvp` and
  `jax.grad` in the frequency follow the composite Kramers--Kronig
  derivative route through a `jax.custom_jvp` rule.
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
