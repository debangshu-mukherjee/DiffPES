# arpyes

JAX-based ARPES simulation toolkit with Python-native APIs.

## Expanded-input workflows

The package includes expanded-input wrappers that let you call the
simulator with plain arrays/scalars while still running JAX kernels.

### Function mapping

- `ARPES_simulation_Novice` -> `arpyes.simul.simulate_novice_expanded`
- `ARPES_simulation_Basic` -> `arpyes.simul.simulate_basic_expanded`
- `ARPES_simulation_Basicplus` -> `arpyes.simul.simulate_basicplus_expanded`
- `ARPES_simulation_Advanced` -> `arpyes.simul.simulate_advanced_expanded`
- `ARPES_simulation_Expert` -> `arpyes.simul.simulate_expert_expanded`
- `ARPES_simulation_SOC` -> `arpyes.simul.simulate_soc_expanded`
- Dynamic dispatch by level -> `arpyes.simul.simulate_expanded`
  (use `level="soc"` with `surface_spin` for SOC)

### Notes

- Default energy-axis padding behavior:
  `min(eigenbands)-1` to `max(eigenbands)+1`.
- Incident angles for expanded wrappers are interpreted in degrees.
- Wrappers return the standard `ArpesSpectrum` PyTree.

### Python indexing conventions

Use standard Python/NumPy indexing everywhere (zero-based, end-exclusive).

- Non-s orbitals: `slice(1, 9)` -> indices 1..8
- p orbitals: `slice(1, 4)` -> indices 1..3
- d orbitals: `slice(4, 9)` -> indices 4..8

Do not use MATLAB-style indexing notation in Python code.

### Example

```python
import jax.numpy as jnp

from arpyes.simul import simulate_expanded

# [nkpt, nband]
eigenbands = jnp.linspace(-2.0, 0.5, 100).reshape(20, 5)
# [nkpt, nband, natom, 9]
surface_orb = jnp.ones((20, 5, 2, 9)) * 0.1

spectrum = simulate_expanded(
    level="advanced",
    eigenbands=eigenbands,
    surface_orb=surface_orb,
    ef=0.0,
    sigma=0.04,
    fidelity=2500,
    temperature=15.0,
    photon_energy=11.0,
    polarization="unpolarized",
    incident_theta=45.0,
    incident_phi=0.0,
    polarization_angle=0.0,
)
```
