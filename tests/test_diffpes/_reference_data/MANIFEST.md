# Reference artifact manifest

> These files pin deterministic behavior unless an entry explicitly
> classifies an independent physics truth.
> The tight-binding cases were repinned for the basis-position
> gauge and carrier-native orbital bases.
> Regenerate only with a stated physics or migration
> justification.

- Generation date: 2026-07-29
- Seed: `20260713`
- Device policy: CPU, JAX x64 enabled
- Platform: `Linux-5.15.0-185-generic-x86_64-with-glibc2.35`
- Python: `3.13.6`
- diffpes: `2026.6.4`
- JAX: `0.9.0.1`
- NumPy: `2.4.2`

## Factory calls

- `novice_toy_true_voigt`: fixed seed-`20260713` carriers assembled
  manually with SciPy `voigt_profile` and the analytic Fermi function; neither
  `voigt` nor `simulate_novice` is called by the generator.
- `chinook_tightb_reference`: offline Chinook 0.1.1 compatibility
  outputs for the independently C-classified graphene, square-lattice Rashba, and
  atomic t2g+SOC models. The generator and isolated environment freeze live
  outside the DiffPES repository under
  external `verification/tightb/` tooling; pytest reads only this inert JSON.
- `wannier90_wse2_reference`: independent NumPy parsing, Fourier
  assembly, and eigensolution of the publicly distributed dynamics-w90
  `data/WSe2_soc/wse2_soc_11bnd_hr.dat` at Γ and reduced-coordinate
  X = (1/2, 0, 0). The exact normative input is stored losslessly compressed;
  its decompressed SHA-256 authenticates the local public snapshot.

## Artifacts

### `detector_chain_manufactured_reference.npz`

- Classification: independent C-type manufactured detector-chain truth
- Single mutable generator authority:
  `diffpes-plans/verification/detector_chain_manufactured/generate_detector_chain_manufactured_reference.py`
- Generator SHA-256:
  `9789939629293cdaa039f98cdaa9119014b3bfa7d9abad7389847c2ea1c758d6`
- Artifact SHA-256:
  `04e41d5f0fa2fe6111718bdc039f49344f48689d74ef0783408585cac76b55c3`
- Provenance and tolerance metadata:
  `diffpes-plans/verification/detector_chain_manufactured/manifest.json`
- Metadata SHA-256:
  `e783075d4b68086582e53153587a1baf744880148d601a4edf29b26eef5f32fc`
- Truth boundary: NumPy/SciPy only; the generator imports neither DiffPES nor
  JAX and evaluates no production detector routine
- Mapping truth: analytic secondary-momentum integration plus 96-node
  Gauss--Legendre integration on every recorded smooth seam cell
- Downstream truth: independent fixed-domain transmission, Gaussian
  finite-volume matrices, smooth sensitivity, background, exposure, and
  native-bin volumes
- Archive contract: 17 named float64 arrays with deterministic ZIP metadata
  and pickle disabled
- Registered tolerances: mapping seams `rtol=1e-9`, final expected counts
  `rtol=1e-8`, and captured-fraction absolute error at most `1e-10`

### Voigt evidence

- Classification: `voigt-scipy-reference` and `spectral-broadening-gradient`, independent SciPy/analytic
  physics truth frozen before and retained after the true-Voigt production edit
- Generator:
  `tests/_reference_tools/generate_voigt_scipy_reference.py`
- Generator SHA-256:
  `0d47779b97b872c5a35f6ff970f7cb16473ef6311adde9535f6579def4ec9e23`
- Generator boundary: NumPy/SciPy only; it imports no DiffPES or JAX module
  and calls neither `voigt` nor `simulate_novice`
- Provenance manifest: `voigt_scipy_manifest.json`
- Provenance-manifest SHA-256:
  `70785578cc72cc312ddf33c4b68c06da82486c0904bb83ae0aed7a72263ecfca`

#### `voigt_scipy_reference.npz`

- SHA-256:
  `43b1b38836fb2cabf683423a8315b7bf2ca2c11a03cab5ec31a4ede7471c29d0`
- Truth engines: `scipy.special.voigt_profile`, `scipy.special.wofz`,
  analytic Gaussian/Cauchy endpoints, and the Faddeeva ODE derivative
- Evidence includes the complete positive-width table, exact endpoints,
  one-sided rates, and scaled 256-to-512 full-line normalization.
- It also includes shared-envelope coordinates, analytic point derivatives,
  contracted D1 truth, and three five-point finite-difference rungs.
- Archive contract: 40 named arrays, all `float64`, deterministic ZIP
  metadata, and pickle disabled

#### `novice_toy_true_voigt.npz`

- SHA-256:
  `ca410005c45faa46ca9e9a7bc949e7954fafdaf494b391bec8cd30bddac50440`
- Classification: retired true-Voigt historical evidence, retained for
  provenance only; it makes no live production-behavior claim
- Truth: manual seed-20260713 fixture assembly with SciPy
  `voigt_profile` and an overflow-safe analytic Fermi function
- Arrays:
  - `leaf_000_intensity`: shape `(8, 512)`, dtype `float64`
  - `leaf_001_energy_axis`: shape `(512,)`, dtype `float64`

#### `novice_toy_pseudo_voigt.npz`

- SHA-256:
  `7585907bef8075904117b13506491ba488038154ff2ec331c5059a2a7ec5d56f`
- Classification: superseded pseudo-Voigt historical evidence,
  retained for provenance only; it is not a compatibility shim
- Byte-for-byte archive of the earlier pseudo-Voigt `novice_toy.npz`; the
  repository-floor integrity gate preserves both archives without replay

### `chinook_tightb_reference.json`

- Classification: `chinook-tightbinding-parity`, K-type behavioral compatibility only
- Chinook commit: `24913de8cc5b8c162f7c1b4acc64bd1b54dd548b`
- Isolated-environment SHA-256:
  `6d00cb4df251508b6392273b1df166f6a17abe8f6691cffead45c636e8ef2531`
- Artifact SHA-256:
  `db52d72562f2efb49d25f9ce2b9affefed1af6f6fac927d1e20f9bb96f1510dc`
- Arrays encoded as JSON numbers:
  - graphene eigenvalues: shape `(33, 2)`, eV
  - square-lattice Rashba eigenvalues: shape `(5, 2)`, eV
  - atomic t2g+SOC eigenvalues: shape `(3, 6)`, eV

### `wannier90_wse2_soc_11bnd_hr.dat.xz`

- Classification: `wannier90-wse2-parity`, publicly distributed normative-format
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

### `wannier90_wse2_reference.json`

- Classification: `wannier90-wse2-parity`, K-type published-input companion
  benchmark; normative-format and analytic checks remain authoritative
- Generator:
  external `verification/tightb/gen_wannier90_wse2_reference.py` tooling
- Generator SHA-256:
  `9bea0278924325526d458094ecfad5b7896d86bfca31c17505f6dd9cf174bac8`
- Artifact SHA-256:
  `afd95f0e6f26771b10e6d825f4e487f88bab0bdc5b326348d43bb6a24194d18c`
- Arrays encoded as JSON numbers:
  - Γ eigenvalues: shape `(22,)`, eV
  - X = `(0.5, 0.0, 0.0)` eigenvalues: shape `(22,)`, eV

### `chinook_matrix_element_parity/`

- Classification: `chinook-pointwise-matrix-element-parity` and
  `chinook-polarization-intensity-parity`, K-type behavioral compatibility;
  the analytic C checks remain authoritative
- Chinook commit:
  `24913de8cc5b8c162f7c1b4acc64bd1b54dd548b`
- Isolated-environment SHA-256:
  `6d00cb4df251508b6392273b1df166f6a17abe8f6691cffead45c636e8ef2531`
- Archive SHA-256:
  `9e857413fce56a3d4af45e88b040a0b85d9af0b445f240f53cb7b1de19365cb1`
- Model-specification SHA-256:
  `c1e6679986e8812313f4c75b9b28daa67e82c69c5707a104d4d744e69bf9c439`
- Pytest reconstructs current public-API amplitudes on all frozen points; it
  does not trust the saved DiffPES replay or import Chinook.

### `local_nonlocal_gauge_reference.npz`

- Classification: independent generic-complex and
  local/nonlocal length-versus-momentum gauge evidence
- Artifact SHA-256:
  `e136dfd8214cd4e1e83d11b1d20d87a8597c66e61f54636b949d3c159fc579f0`
- The tracked code-tree copy is byte-equal to
  external `verification/matrixel_gauge/g12_reference.npz` tooling.

### `yeh_lindau_authority/`

- Classification: authority metadata for the dated Yeh--Lindau cross-sections
  Figshare-as-numerical-authority amendment
- Figshare v3 metadata SHA-256:
  `c908a3c855ffe98dabd4660fa4d3c17849ac0b9563c26c7bde0197026a7bda44`
- Regoutz project-page SHA-256:
  `2e4c3cc0dbb73cecced5d8608fa44286ac444636e8d9142f4bbe4b042d236703`
- The test authenticates the version, license, file identity, manual-mining
  statement, review statement, and Lindau-permission statement.
