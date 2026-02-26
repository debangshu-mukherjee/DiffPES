# arpyes

JAX-based ARPES simulation toolkit ported from MATLAB workflows.

## MATLAB compatibility

The package now includes MATLAB-style wrappers that mirror the
`ARPES_simulation_*` entry points while running the JAX kernels.

### Function mapping

- `ARPES_simulation_Novice` -> `arpyes.simul.simulate_novice_matlab`
- `ARPES_simulation_Basic` -> `arpyes.simul.simulate_basic_matlab`
- `ARPES_simulation_Basicplus` -> `arpyes.simul.simulate_basicplus_matlab`
- `ARPES_simulation_Advanced` -> `arpyes.simul.simulate_advanced_matlab`
- `ARPES_simulation_Expert` -> `arpyes.simul.simulate_expert_matlab`
- Dynamic dispatch by level -> `arpyes.simul.simulate_matlab`

### Notes

- MATLAB energy-axis behavior is preserved by default:
  `min(eigenbands)-1` to `max(eigenbands)+1`.
- Incident angles for MATLAB wrappers are interpreted in degrees.
- Wrappers return the standard `ArpesSpectrum` PyTree.

### Example

```python
import jax.numpy as jnp

from arpyes.simul import simulate_matlab

# [nkpt, nband]
eigenbands = jnp.linspace(-2.0, 0.5, 100).reshape(20, 5)
# [nkpt, nband, natom, 9]
surface_orb = jnp.ones((20, 5, 2, 9)) * 0.1

spectrum = simulate_matlab(
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
