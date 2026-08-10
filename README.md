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

## Coherent spectral workflows

The production spectral surface preserves complex transition sources through
the final observable. `diffpes.simul.spectral_intensity_resolvent` is the
degeneracy-safe path. `spectral_intensity_eigen` is a faster path for
gauge-invariant band weights away from degeneracies. Both consume the causal
self-energy returned by `evaluate_self_energy`.

Detector mapping, resolution, transmission, and counts are intentionally
outside this intrinsic spectral boundary. Plan 08a is constructing the single
canonical detector/count driver; there is currently no level-string workflow
or projection-probability compatibility dispatcher.

### Python indexing conventions

Use standard Python/NumPy indexing everywhere (zero-based, end-exclusive).

- Non-s orbitals: `slice(1, 9)` -> indices 1..8
- p orbitals: `slice(1, 4)` -> indices 1..3
- d orbitals: `slice(4, 9)` -> indices 4..8

Do not use MATLAB-style indexing notation in Python code.

### Example

```python
import jax.numpy as jnp

import jax

from diffpes.simul import evaluate_self_energy, spectral_intensity_eigen
from diffpes.types import make_self_energy_model

omega = jnp.linspace(-1.0, 1.0, 501)
self_energy = evaluate_self_energy(
    omega,
    make_self_energy_model(gamma=0.08),
)
eigenvalues = jnp.array([-0.25, 0.30])
band_weights = jnp.array([0.8, 0.2])
intrinsic = jax.vmap(
    lambda energy, sigma: spectral_intensity_eigen(
        eigenvalues,
        band_weights,
        energy,
        sigma,
        1.0e-4,
    )
)(
    omega,
    self_energy,
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
   Add a test for each coherent matrix-element or spectral branch.
2. **HDF5:** Round-trip every PyTree type. Test each load and save error path.
3. **VASP file readers:** Test `read_doscar`, `read_eigenval`, `read_kpoints`,
   `read_poscar`, and `read_procar` with minimal repository fixtures.
4. **Plotting:** Exercise the public plotting API in tests. GUI code can use a
   lower coverage target.
5. **Edge branches:** Cover optional arguments and their error messages.
   Include `make_band_structure(..., kpoint_weights=...)`.
