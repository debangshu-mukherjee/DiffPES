# diffpes

[![License](https://img.shields.io/pypi/l/diffpes.svg)](https://github.com/debangshu-mukherjee/diffpes/blob/main/LICENSE)
[![PyPI Downloads](https://static.pepy.tech/badge/diffpes)](https://pepy.tech/projects/diffpes)
[![PyPI version](https://img.shields.io/pypi/v/diffpes.svg)](https://pypi.python.org/pypi/diffpes)
[![Python Versions](https://img.shields.io/pypi/pyversions/diffpes.svg)](https://pypi.python.org/pypi/diffpes)
[![Documentation Status](https://readthedocs.org/projects/diffpes/badge/?version=latest)](https://diffpes.readthedocs.io/en/latest/?badge=latest)
[![tests](https://github.com/debangshu-mukherjee/diffpes/actions/workflows/tests.yml/badge.svg)](https://github.com/debangshu-mukherjee/diffpes/actions/workflows/tests.yml)
[![codecov](https://codecov.io/gh/debangshu-mukherjee/diffpes/graph/badge.svg)](https://codecov.io/gh/debangshu-mukherjee/diffpes)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19037631.svg)](https://doi.org/10.5281/zenodo.19037631)
[![Ruff](https://img.shields.io/badge/lint%20and%20format-ruff-D7FF64?logo=ruff&logoColor=1D1D1D)](https://github.com/astral-sh/ruff)
[![ty](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ty/main/assets/badge/v0.json)](https://github.com/astral-sh/ty)
[![jax_badge](https://tinyurl.com/mucknrvu)](https://docs.jax.dev/)
[![Lines of Code](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/debangshu-mukherjee/diffpes/main/.github/badges/loc.json)](https://github.com/debangshu-mukherjee/diffpes)

diffpes is a JAX-based ARPES simulation toolkit with Python-native APIs and
certified forward execution. A certified run stores its observable and
scientific evidence in the same differentiable PyTree. The evidence includes
bounded physics claims, provenance, domain margins, derivatives, local
information-flow diagnostics, and a named assurance policy. JAX compiles and
batches the numerical certification path. Portable serialization stays at the
filesystem boundary.

Certification here means bounded scientific evidence, not a security
credential. Storage consistency markers detect accidental mismatches only.

The geometry layer converts crystal coordinates, detector angles, and photon
energy into fixed-shape momentum rasters. Its JAX derivatives expose
calibration sensitivity to the work function, inner potential, sample
azimuth, and detector frame.

## Expanded-input workflows

The package provides expanded-input wrappers for plain arrays and scalars.
These wrappers run the same JAX kernels as the typed interfaces.

### Function mapping

- `ARPES_simulation_Novice` -> `diffpes.simul.simulate_novice_expanded`
- `ARPES_simulation_Basic` -> `diffpes.simul.simulate_basic_expanded`
- Dynamic dispatch by level -> `diffpes.simul.simulate_expanded`

These two wrappers are deliberately incoherent: they consume VASP projection
probabilities and cannot reconstruct orbital phases. Quantitative
polarization-dependent calculations use the coherent primitives in
`diffpes.simul.matrixel`.

### Notes

- Default energy-axis padding behavior:
  `min(eigenbands)-1` to `max(eigenbands)+1`.
- Wrappers return the standard `ArpesSpectrum` PyTree.
- The `basic` tier requires an atom-major `OrbitalBasis` and one atomic
  number per atom so that it can select Yeh--Lindau subshell data.

### Python indexing conventions

Use standard Python/NumPy indexing everywhere (zero-based, end-exclusive).

- Non-s orbitals: `slice(1, 9)` -> indices 1..8
- p orbitals: `slice(1, 4)` -> indices 1..3
- d orbitals: `slice(4, 9)` -> indices 4..8

Do not use MATLAB-style indexing notation in Python code.

### Example

```python
import jax.numpy as jnp

from diffpes.simul import simulate_expanded

# [nkpt, nband]
eigenbands = jnp.linspace(-2.0, 0.5, 100).reshape(20, 5)
# [nkpt, nband, natom, 9]
surface_orb = jnp.ones((20, 5, 2, 9)) * 0.1

spectrum = simulate_expanded(
    level="novice",
    eigenbands=eigenbands,
    surface_orb=surface_orb,
    ef=0.0,
    sigma=0.04,
    gamma=0.1,
    fidelity=2500,
    temperature=15.0,
    photon_energy=11.0,
)
```

## Test coverage

Test coverage identifies the source lines that the tests execute. Run the
coverage check with this command:

```bash
source .venv/bin/activate
pytest tests/ --cov=src/diffpes --cov-report=term-missing
```

Use these priorities to increase coverage toward 100%:

1. **Simulation and types:** These modules already have good coverage.
   Add a test for each new coherent matrix-element or dispatch branch.
2. **Expanded dispatch:** Test both `simulate_expanded(level=...)` branches.
   Also test the `ValueError` for an unknown level.
3. **HDF5:** Round-trip every PyTree type. Test each load and save error path.
4. **VASP file readers:** Test `read_doscar`, `read_eigenval`, `read_kpoints`,
   `read_poscar`, and `read_procar` with minimal repository fixtures.
5. **Plotting:** Exercise the public plotting API in tests. GUI code can use a
   lower coverage target.
6. **Edge branches:** Cover optional arguments and their error messages.
   Include `make_band_structure(..., kpoint_weights=...)`.
