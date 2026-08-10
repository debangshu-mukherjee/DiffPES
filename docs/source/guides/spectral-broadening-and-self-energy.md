# Spectral Broadening and Self-Energy

Broadening converts discrete band energies into a sampled spectral map. It is
separate from coherent matrix-element assembly: matrix-element weights set
the band brightness, while lineshapes distribute that weight over energy and
momentum.

## Energy Profiles

`gaussian(x, center, sigma)` returns a normalized Gaussian for explicit
profile diagnostics.

`voigt(x, center, sigma, gamma)` returns the normalized true Voigt
convolution through the certified Faddeeva evaluator. `sigma` is the Gaussian
standard deviation and `gamma` is the Lorentzian half-width at half-maximum,
both in eV.

```python
from diffpes.simul import gaussian, voigt

g = gaussian(energy_axis, center=-0.3, sigma=0.04)
v = voigt(energy_axis, center=-0.3, sigma=0.04, gamma=0.08)
```

The coherent spectral chunk assemblers apply finite-temperature
`fermi_dirac` occupation on the sampled energy axis.

## Self-Energy

`evaluate_self_energy` evaluates the complex retarded self-energy from a
`SelfEnergyModel`. The real part comes from the certified once-subtracted
Kramers--Kronig operator for the numerical modes. The imaginary part
evaluates the causal, strictly negative model profile. Take `-sigma.imag`
for a positive Lorentzian linewidth.

Use `"constant"` for uniform broadening and `"poly"` for a polynomial model.
Use `"grid"` for interpolation on relative-energy nodes. The
`"fermi_liquid"` and `"bosonic_kink"` modes provide the other supported
energy-dependent models. Construct each carrier with
`make_self_energy_model`. Numerical modes require every query inside the
trusted interval of the declared domain.

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

This is a narrow Cartesian-path parity helper, not the canonical detector
resolution stage. Plan 08a is constructing native-coordinate resolution and
the detector/count driver; no current high-level workflow applies this helper
implicitly. The energy axis does not change.

## Combining Spectral and Coherent Physics

A coherent workflow first computes outgoing-spin band amplitudes:

1. assemble orbital transition channels,
2. project them with complex band coefficients,
3. contract Cartesian polarization late,
4. convert each outgoing row to its resolvent source independently.

The resolvent input has shape `[n_k, n_omega, n_out, n_orb]`. It solves every
outgoing-channel RHS independently and sums real quadratic responses only
after those solves; adding source rows coherently before solving is forbidden.
The spinless case still carries the explicit nonempty axis with `n_out=1`.
The eigen fast path instead consumes gauge-invariant band weights after the
outgoing-spin modulus-square reduction. Both feed the coherent intrinsic
spectral APIs directly:

```python
from diffpes.simul import (
    assemble_spectral_intensity_bands_chunk,
    assemble_spectral_intensity_chunk,
)

intrinsic = assemble_spectral_intensity_chunk(
    hamiltonians_ev,
    transition_sources,
    omega_rel_fermi_ev,
    self_energy,
    fermi_energy_ev,
    temperature_k,
)
fast_intrinsic = assemble_spectral_intensity_bands_chunk(
    eigenvalues_ev,
    band_weights,
    omega_rel_fermi_ev,
    self_energy,
    fermi_energy_ev,
    temperature_k,
)
```

The resolvent path applies a complex128 Lineax solve. It remains safe at exact
degeneracies. The eigen path is faster on nondegenerate k paths and consumes
only gauge-invariant weights. Both return
`A(k, omega) * f_FD(omega, T)`. The Fermi factor uses the sampled
relative-energy axis. The assembler subtracts the absolute Fermi energy from
the Hamiltonian or eigenvalues exactly once.

`spectral_intensity_resolvent` accepts `[n_out, n_orb]` sources and returns
their post-solve incoherent sum, while
`projected_spectral_density_resolvent` retains the complete Hermitian channel
density. The latter is the appropriate seam for spin or projector channels;
an elementwise imaginary part would discard off-diagonal coherence.

This ordering preserves orbital and inter-centre interference while keeping
instrument effects separate. The intrinsic assembly performs no Gaussian
resolution, normalization, background, transmission, or count conversion.

`band_group_weight_sensitivity` returns derivatives of the matrix-element
weights before spectral, exposure, background, or detector factors.
`expected_counts` now converts an already mapped detector density into counts
on explicit `DetectorCalibration` bins. Plan 08a still owns the canonical
source-to-detector mapping and driver.

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
- Keep differentiated eigen calls at adjacent gaps of at least
  `1e3 * EPS_DEG`. Use `allow_degenerate_value_only=True` only for primal
  checks with complete invariant weights. The flag makes no derivative claim.
  Use the resolvent path for high-symmetry degeneracies and
  Hamiltonian-parameter gradients. Each sample costs one cubic-time LU solve
  per `(k, omega, n_out)`. Scan fixed-size energy chunks for large cubes.
- Construct transition sources inside each live `(k_chunk, omega_chunk)` scan
  step. Do not materialize a complete `[K, E, n_out, n_orb]` source carrier.
- Keep the resolvent operator, right-hand side, and solution in complex128.
  The solve is intentionally never demoted to mixed precision.
- Keep validity masks for domain-limited quantities such as emission
  kinematics and logarithmic band-group derivatives.

## Registered Scaling Evidence

The WP7.6 CPU gate compiled the literal `256 k x 512 omega x 32 orbital`
spinless value-and-Hamiltonian-gradient target with static `32 x 32` chunks,
checkpointing, `n_kk=4096`, and `n_tail=256`. XLA reported `7,483,224` argument
bytes, `4,194,328` output bytes, `48,595,264` temporary bytes, and zero aliased
bytes: `60,272,816` compiler-live bytes in total. This is below the registered
spinless solve-tape estimate of `134,217,728` bytes and its `1.5x` ceiling of
`201,326,592` bytes. The target was compiled for allocation analysis but was
not executed; that fact is explicit in the authenticated artifact. Host RSS
(`478,945,280` to `693,555,200` bytes) is diagnostic only.

The companion comparison matched an unchunked production assembly to
`4.336808689942018e-19` maximum absolute value error and exactly zero maximum
Hamiltonian-gradient error. Three active shapes inside one padded schedule
produced one trace, and the lowered Lineax operator, RHS, and solution were all
complex128. The reproducible record is
`tests/test_diffpes/_reference_data/spectral_scalability/cpu_benchmark.json`.
Its committed SHA-256 is
`3b4248a9498281f09fe9152f4fcc2db42ed92e2d82dedf93cf12cf229e352bca`.

See [Simulation Tiers and the Coherent Pipeline](simulation-levels.md) for
the model boundary and
[JAX Transformability and Gradients](jax-transformability-and-gradients.md)
for derivative semantics.
