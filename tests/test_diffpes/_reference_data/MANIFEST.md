# Reference artifact manifest

> These files pin deterministic behavior, not independent physics
> truth.
> The tight-binding cases were repinned for Plan 04's basis-position
> gauge and carrier-native orbital bases.
> Regenerate only with a stated physics or migration
> justification.

- Generation date: 2026-07-22
- Seed: `20260713`
- Device policy: CPU, JAX x64 enabled
- Platform: `Linux-5.15.0-185-generic-x86_64-with-glibc2.35`
- Python: `3.13.6`
- diffpes: `2026.6.4`
- JAX: `0.9.0.1`
- NumPy: `2.4.2`

## Factory calls

- `novice_toy`: `simulate_novice(toy_band_structure(key), toy_orbital_projection(key), toy_simulation_params(fidelity=512), 15.0)`
- `plan04_chinook_tightb_reference`: offline Chinook 0.1.1 compatibility
  outputs for the independently C-gated graphene, square-lattice Rashba, and
  atomic t2g+SOC models. The generator and isolated environment freeze live
  outside the DiffPES repository under
  `diffpes-plans/verification/tightb/`; pytest reads only this inert JSON.
- `plan04_wannier90_wse2_reference`: independent NumPy parsing, Fourier
  assembly, and eigensolution of the publicly distributed dynamics-w90
  `data/WSe2_soc/wse2_soc_11bnd_hr.dat` at Γ and reduced-coordinate
  X = (1/2, 0, 0). The exact normative input is stored losslessly compressed;
  its decompressed SHA-256 authenticates the local public snapshot.

## Artifacts

### `novice_toy.npz`

- SHA-256: `7585907bef8075904117b13506491ba488038154ff2ec331c5059a2a7ec5d56f`
- Classification: active Plan-02 Thompson-Cox-Hastings pseudo-Voigt
  behavioral reference until the Plan-07 production closeout
- Arrays:
  - `leaf_000_intensity`: shape `(8, 512)`, dtype `float64`
  - `leaf_001_energy_axis`: shape `(512,)`, dtype `float64`

### Plan-07 Voigt preregistration

- Classification: Plan-07 gates 07.G2 and 07.D1, independent SciPy/analytic
  physics truth frozen before the WP7.2 production edit
- Generator:
  `tests/_reference_tools/generate_plan07_voigt_reference.py`
- Generator SHA-256:
  `cf5b8927dac24c42a0cec6ed5a95171b78c981110bae06d8c940c8ee399f27b1`
- Generator boundary: NumPy/SciPy only; it imports no DiffPES or JAX module
  and calls neither `voigt` nor `simulate_novice`
- Provenance manifest: `plan07_voigt_manifest.json`
- Provenance-manifest SHA-256:
  `25d73f8b6283b7d81447ae059c0598c8f01ac5a3f6441ee3819153285496125b`

#### `plan07_voigt_scipy_reference.npz`

- SHA-256:
  `43b1b38836fb2cabf683423a8315b7bf2ca2c11a03cab5ec31a4ede7471c29d0`
- Truth engines: `scipy.special.voigt_profile`, `scipy.special.wofz`,
  analytic Gaussian/Cauchy endpoints, and the Faddeeva ODE derivative
- Evidence: complete positive-width table, exact representable-input
  endpoints, one-sided convergence rates, scaled 256-to-512 full-line
  normalization, shared-envelope coordinates, analytic point derivatives,
  contracted D1 truth, and three five-point finite-difference rungs
- Archive contract: 40 named arrays, all `float64`, deterministic ZIP
  metadata, and pickle disabled

#### `novice_toy_plan07_true_voigt.npz`

- SHA-256:
  `ca410005c45faa46ca9e9a7bc949e7954fafdaf494b391bec8cd30bddac50440`
- Classification: preregistered Plan-07 true-Voigt novice behavioral
  reference; strict-red against the retained TCH production path
- Truth: manual seed-20260713 fixture assembly with SciPy
  `voigt_profile` and an overflow-safe analytic Fermi function
- Arrays:
  - `leaf_000_intensity`: shape `(8, 512)`, dtype `float64`
  - `leaf_001_energy_axis`: shape `(512,)`, dtype `float64`

#### `novice_toy_plan02_pseudo_voigt.npz`

- SHA-256:
  `7585907bef8075904117b13506491ba488038154ff2ec331c5059a2a7ec5d56f`
- Classification: superseded Plan-02 pseudo-Voigt historical evidence,
  retained for provenance only; it is not a compatibility shim
- Byte-for-byte archive of the pre-WP7.2 `novice_toy.npz`; repository-floor
  replay remains on the active filename until production closeout

### `plan04_chinook_tightb_reference.json`

- Classification: Plan 04 gate 04.G6, K-type behavioral compatibility only
- Chinook commit: `24913de8cc5b8c162f7c1b4acc64bd1b54dd548b`
- Isolated-environment SHA-256:
  `6d00cb4df251508b6392273b1df166f6a17abe8f6691cffead45c636e8ef2531`
- Artifact SHA-256:
  `db52d72562f2efb49d25f9ce2b9affefed1af6f6fac927d1e20f9bb96f1510dc`
- Arrays encoded as JSON numbers:
  - graphene eigenvalues: shape `(33, 2)`, eV
  - square-lattice Rashba eigenvalues: shape `(5, 2)`, eV
  - atomic t2g+SOC eigenvalues: shape `(3, 6)`, eV

### `plan04_wse2_soc_11bnd_hr.dat.xz`

- Classification: Plan 04 gate 04.G7, publicly distributed normative-format
  input
- Upstream repository:
  `https://github.com/michaelschueler/dynamics-w90`
- Upstream snapshot path: `data/WSe2_soc/wse2_soc_11bnd_hr.dat`
- Upstream commit:
  `6f6d99e7fe4b2839a735c609d7df19d1886e8deb` (byte-for-byte verified)
- License qualification: the upstream repository displays no license, so no
  license grant is claimed; only this normative input crosses the
  independent-implementation boundary
- Decompressed size: `5,543,022` bytes
- Decompressed SHA-256:
  `8ea8140e4fb3d1e56c188d5d680ab077b9ad57070f9205c7365cbb24a7c40dd1`
- Compressed SHA-256:
  `756fdcf2541aa75dad69ae172327fd5cdf6ba044812c918efb9c62a690ece9d4`

### `plan04_wannier90_wse2_reference.json`

- Classification: Plan 04 gate 04.G7, K-type published-input companion
  benchmark; normative-format and analytic gates remain authoritative
- Generator:
  `diffpes-plans/verification/tightb/gen_wannier90_wse2_reference.py`
- Generator SHA-256:
  `9bea0278924325526d458094ecfad5b7896d86bfca31c17505f6dd9cf174bac8`
- Artifact SHA-256:
  `afd95f0e6f26771b10e6d825f4e487f88bab0bdc5b326348d43bb6a24194d18c`
- Arrays encoded as JSON numbers:
  - Γ eigenvalues: shape `(22,)`, eV
  - X = `(0.5, 0.0, 0.0)` eigenvalues: shape `(22,)`, eV

### `plan06_chinook/`

- Classification: Plan 06 gates 06.G6 and 06.G7, K-type behavioral
  compatibility; the analytic C gates remain authoritative
- Chinook commit:
  `24913de8cc5b8c162f7c1b4acc64bd1b54dd548b`
- Isolated-environment SHA-256:
  `6d00cb4df251508b6392273b1df166f6a17abe8f6691cffead45c636e8ef2531`
- Archive SHA-256:
  `9e857413fce56a3d4af45e88b040a0b85d9af0b445f240f53cb7b1de19365cb1`
- Model-specification SHA-256:
  `8c2b00c99242b539e694620bc744fb89eb7898a2402ad3f684263e4e2a50827e`
- Pytest reconstructs current public-API amplitudes on all frozen points; it
  does not trust the saved DiffPES replay or import Chinook.

### `plan06_g12_reference.npz`

- Classification: Plan 06 gate 06.G12 independent generic-complex and
  local/nonlocal length-versus-momentum gauge evidence
- Artifact SHA-256:
  `e136dfd8214cd4e1e83d11b1d20d87a8597c66e61f54636b949d3c159fc579f0`
- The tracked code-tree copy is byte-equal to
  `diffpes-plans/verification/matrixel_gauge/g12_reference.npz`.

### `plan06_yeh_lindau_authority/`

- Classification: Plan 06 gate 06.G5 authority metadata for the dated
  Figshare-as-numerical-authority amendment
- Figshare v3 metadata SHA-256:
  `c908a3c855ffe98dabd4660fa4d3c17849ac0b9563c26c7bde0197026a7bda44`
- Regoutz project-page SHA-256:
  `2e4c3cc0dbb73cecced5d8608fa44286ac444636e8d9142f4bbe4b042d236703`
- The test authenticates the version, license, file identity, manual-mining
  statement, review statement, and Lindau-permission statement.
