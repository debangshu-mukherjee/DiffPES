# Spectral Broadening and Self-Energy

Broadening converts discrete band energies into a sampled spectral map. It is
separate from coherent matrix-element assembly: matrix-element weights set
the band brightness, while lineshapes distribute that weight over energy and
momentum.

## Energy Profiles

`gaussian(x, center, sigma)` returns a normalized Gaussian. The retained
`basic` incoherent tier uses it.

`voigt(x, center, sigma, gamma)` returns the normalized true Voigt
convolution through the certified Faddeeva evaluator. The retained `novice`
tier uses it. `sigma` is the Gaussian standard deviation and `gamma` is the
Lorentzian half-width at half-maximum, both in eV.

```python
from diffpes.simul import gaussian, voigt

g = gaussian(energy_axis, center=-0.3, sigma=0.04)
v = voigt(energy_axis, center=-0.3, sigma=0.04, gamma=0.08)
```

Both spectrum tiers multiply the band profile by the finite-temperature
`fermi_dirac` occupation before summing bands.

## Self-Energy

`evaluate_self_energy` evaluates the imaginary self-energy from a
`SelfEnergyConfig`. Supported static models and their parameters are
documented in the API reference. The output can define an
energy-dependent linewidth for a caller-built spectral function.

Self-energy and broadening do not create orbital coherence. If an input
consists only of projection probabilities, applying a more elaborate
lineshape cannot restore phase-sensitive matrix elements.

## Momentum Resolution

`apply_momentum_broadening` convolves an intensity array along a one-
dimensional k-path:

```python
broadened = diffpes.simul.apply_momentum_broadening(
    spectrum.intensity,
    k_dist,
    dk=0.02,
)
```

`simulate_context` and `run_vasp_workflow` expose this operation through the
optional `dk` argument after either retained incoherent tier. The energy axis
does not change.

## Combining Spectral and Coherent Physics

A coherent workflow first computes outgoing-spin band amplitudes:

1. assemble orbital transition channels,
2. project them with complex band coefficients,
3. contract Cartesian polarization late,
4. sum outgoing-spin modulus squares once.

The resulting band weights can then multiply a spectral function built from
occupation, self-energy, and instrumental resolution. This ordering
preserves orbital and inter-centre interference while keeping spectral
broadening conceptually separate.

`band_group_weight_sensitivity` returns derivatives of the matrix-element
weights before spectral, exposure, background, or detector factors. Expected
detector counts begin in Plan 08.

## Numerical Guidance

- Keep widths strictly positive for fitted interior parameters. Exact
  Gaussian (`gamma=0`) and Cauchy (`sigma=0`) calls are value-only endpoints;
  the double-zero delta limit is rejected.
- For positive widths, keep the complete sampled array inside the certified
  Faddeeva envelope `abs(z) <= 1e8`, where
  `z=(x-center+1j*gamma)/(sigma*sqrt(2))`.
- Sample the energy axis finely enough to resolve its narrowest profile.
- Include sufficient padding so normalized tails are not clipped.
- Treat the momentum grid as ordered when applying one-dimensional
  convolution.
- Keep validity masks for domain-limited quantities such as emission
  kinematics and logarithmic band-group derivatives.

See [Simulation Tiers and the Coherent Pipeline](simulation-levels.md) for
the model boundary and
[JAX Transformability and Gradients](jax-transformability-and-gradients.md)
for derivative semantics.
